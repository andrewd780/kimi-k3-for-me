"""Adaptive loop depth and the default loop (docs/notes/layer-loop.md), on the tiny K3.

--loop-exit EPS makes the loop's depth per token: every token runs the span twice, then keeps going while a pass moves
its span output by EPS or more (relative L2), up to --loop-count passes. Pass p's layers see only the tokens that reached
pass p. The CLI runs this by default on the AttnRes block holding the middle layer, up to 4 passes, at EPS 0.05;
--no-loop is the released model. Like tests/test_layer_loop.py, these gates ask whether the engine computes the stack it
claims to, never whether that stack is any good:
  - identities: EPS 0 never stops early, so it is bit-identical to the fixed loop at the same count; an EPS no change can
    reach stops every token after pass 2, so it is bit-identical to the fixed loop at count 2;
  - reference: with an EPS that splits the tokens between depths, the per-position pass counts equal those of an
    independent torch implementation of the semantics (tools/k3_ref.py forward_adaptive) and the logits match it within
    tools/cmp_logits.py's yardstick; the fixed loop at full depth must fall OUTSIDE that yardstick (the gate can fail);
    greedy continuations match;
  - mode parity at mixed depths: incremental, --kv-latent, --mla-split, a streamed trunk, a streamed lm_head and
    --expert-pipeline are bit-identical to full recompute;
  - the no-compaction mutant (K3_LOOP_MUTANT_NO_COMPACT, test builds only: stopped tokens leak into later passes' state)
    is caught by the reference;
  - the default is the adaptive middle-block loop, and --no-loop is the released model;
  - refusals name their reason.

Needs numpy and torch. Build first (`make all mutants`).
K3_TEST_BIN=bin/k3 K3_TEST_MUT_BIN=bin/k3-mut python3 -m unittest discover -s tests -p test_loop_adaptive.py -v
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_trunk  # noqa: E402

PROMPT = "3,7,11,5,9,1,2,4,250,17,33,8,90,61"     # long enough for tokens to split between depths
LAYERS = 13                                       # tiny_config: blocks of 3, boundaries at 0, 3, 6, 9, 12
BUDGET = 1.2e-7 * math.sqrt(128) * 50             # tools/cmp_logits.py at hidden 128
VOCAB = 256
MARGIN = 0.02                                     # a change this close (relative) to EPS is a coin flip


class LoopAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.binary = (ROOT / os.environ.get("K3_TEST_BIN", "bin/k3")).resolve()
        cls.mutant = (ROOT / os.environ.get("K3_TEST_MUT_BIN", "bin/k3-mut")).resolve()
        cls.env = {**os.environ, "OMP_NUM_THREADS": "2", "K3_NOHUGE": "1"}
        for k in ("K3_LOOP_MUTANT_SHARED_STATE", "K3_LOOP_MUTANT_NO_COMPACT"):
            cls.env.pop(k, None)
        cls.model = cls.root / "plain"
        subprocess.run([sys.executable, str(ROOT / "tools/make_tiny_checkpoint.py"), str(cls.model)],
                       check=True, capture_output=True, env=cls.env)
        cls.trunk = cls.root / "trunk"
        if pack_trunk.main([str(cls.model), str(cls.trunk), str(LAYERS)]):
            raise AssertionError("tiny trunk packing failed")
        import k3_ref
        import make_tiny_checkpoint
        cls.ref = k3_ref
        _, cls.torch_model = make_tiny_checkpoint.reference_model()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.counter = 0

    def run_cli(self, args, gen=4, binary=None, env=None, ok=True, ids=PROMPT, rc=2):
        self.counter += 1
        out = self.path / ("%d.json" % self.counter)
        logits = self.path / ("%d.f32" % self.counter)
        result = subprocess.run([str(binary or self.binary), str(self.model), "--ids", ids, "--gen", str(gen),
                                 "--cache-gb", "0.0001", "--out", str(out), "--dump-all-logits", str(logits),
                                 *map(str, args)],
                                text=True, errors="replace", capture_output=True,
                                env={**self.env, **(env or {})}, timeout=180)
        if not ok:
            self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
            self.assertFalse(out.exists(), "a refused run must not write a report")
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text(errors="replace"))
        self.assertEqual(report["expert_drops"], 0)
        raw = np.fromfile(logits, dtype=np.float32)
        self.assertEqual(raw.size, gen * VOCAB)
        return report, raw

    def assert_same(self, a, b):
        self.assertEqual(a[0]["full_ids"], b[0]["full_ids"])
        self.assertEqual(a[1].tobytes(), b[1].tobytes(), "every generated token's logits must be bit-identical")

    @staticmethod
    def rel(c, r):
        return float(np.abs(c.astype(np.float64) - r.astype(np.float64)).max()) / max(float(np.abs(r).max()), 1e-30)

    @staticmethod
    def loop(span, count, eps=None):
        return ["--loop-layers", span, "--loop-count", str(count), *([] if eps is None else ["--loop-exit", eps])]

    def split_eps(self, span, count, gen=1, extra=()):
        """An EPS that splits the tokens between depths, every change the run then measures sitting clearly to one
        side of it. Candidates are the geometric midpoints of the gaps between the pass-2 changes of an EPS-0 run
        (pass 2 runs on every token whatever EPS is), widest gap first, each leaving at least two tokens on either
        side; the first whose own run is clear and mixed is taken."""
        cached = getattr(self, "_eps_cache", {})
        self._eps_cache = cached
        key = (span, count, gen, tuple(extra))
        if key in cached:
            return cached[key]
        report, _ = self.run_cli([*extra, *self.loop(span, count, "0")], gen=gen)
        d2 = sorted(ch[0] for ch in report["loop_adaptive"]["change"] if ch)
        gaps = sorted(((math.log(d2[i + 1] / d2[i]), i) for i in range(1, len(d2) - 2) if d2[i] > 0), reverse=True)
        for width, i in gaps[:8]:
            if width < 2 * MARGIN:
                break
            eps = "%.9g" % math.sqrt(d2[i] * d2[i + 1])
            run, _ = self.run_cli([*extra, *self.loop(span, count, eps)], gen=gen)
            changes = [d for ch in run["loop_adaptive"]["change"] for d in ch]
            passes = [p for p in run["loop_adaptive"]["passes"] if p]
            if all(abs(math.log(d / float(eps))) > MARGIN for d in changes) and len(set(passes)) > 1:
                cached[key] = eps
                return eps
        self.fail("no EPS splits %s x%d cleanly" % (span, count))

    @staticmethod
    def change_gap(report, ref_changes, n):
        """Largest relative difference between a run's per-pass changes and the reference's, over the passes both
        recorded for the first n positions."""
        worst = 0.0
        for e, r in zip(report["loop_adaptive"]["change"][:n], ref_changes):
            for a, b in zip(e, r):
                worst = max(worst, abs(a - b) / max(abs(b), 1e-30))
        return worst

    def assert_clear(self, report, eps):
        """Every change the run measured sits clearly to one side of EPS, so torch's float order and the engine's
        cannot decide a token differently."""
        e = float(eps)
        for i, ch in enumerate(report["loop_adaptive"]["change"]):
            for d in ch:
                self.assertGreater(abs(math.log(d / e)), MARGIN, "position %d: change %g is too close to %g" % (i, d, e))

    # ------------------------------------------------------------------ identities
    def test_exit_zero_is_the_fixed_loop(self):
        for mode in ([], ["--incremental"], ["--incremental", "--kv-latent"]):
            for span, count in (("3-5", 3), ("3-8", 2), ("12-12", 4)):
                with self.subTest(mode=mode, span=span, count=count):
                    fixed = self.run_cli([*mode, *self.loop(span, count)])
                    adapt = self.run_cli([*mode, *self.loop(span, count, "0")])
                    self.assert_same(fixed, adapt)
                    n = len(adapt[0]["full_ids"]) - 1
                    self.assertEqual(adapt[0]["loop_adaptive"]["passes"][:n], [count] * n)

    def test_an_unreachable_exit_is_two_passes(self):
        for mode in ([], ["--incremental"]):
            for span in ("3-5", "12-12"):
                with self.subTest(mode=mode, span=span):
                    fixed = self.run_cli([*mode, *self.loop(span, 2)])
                    adapt = self.run_cli([*mode, *self.loop(span, 4, "1e300")])
                    self.assert_same(fixed, adapt)
                    n = len(adapt[0]["full_ids"]) - 1
                    self.assertEqual(adapt[0]["loop_adaptive"]["passes"][:n], [2] * n)

    # ------------------------------------------------------------------- reference
    def test_mixed_depths_match_the_torch_reference_and_the_gate_can_fail(self):
        ids = [int(v) for v in PROMPT.split(",")]
        for span, count in (("3-5", 4), ("3-8", 3), ("12-12", 4)):
            with self.subTest(span=span, count=count):
                lo, hi = (int(v) for v in span.split("-"))
                eps = self.split_eps(span, count)
                report, got = self.run_cli(self.loop(span, count, eps), gen=1)
                self.assert_clear(report, eps)
                logits, passes, changes = self.ref.forward_adaptive(self.torch_model, ids, lo, hi, count, float(eps))
                n = len(ids)
                self.assertEqual(report["loop_adaptive"]["passes"][:n], passes, "per-position depth")
                self.assertGreater(len(set(passes)), 1, "the EPS must split the tokens between depths")
                want = logits[-1].float().numpy()
                self.assertLessEqual(self.rel(got, want), BUDGET, "adaptive engine vs adaptive reference")
                # the plant: every token at full depth (EPS 0) must NOT pass: its later passes see tokens the
                # reference's did not, so their changes move far past the tolerance the real run meets, and its
                # logits leave the yardstick
                good = self.change_gap(report, changes, n)
                self.assertLess(good, 1e-4, "per-pass changes vs the reference")
                if span != "12-12":
                    # (on the MLA-only tail span a pass's other tokens barely move a token's output in this tiny
                    # model, 9e-5 against the real run's 1.4e-5, so the plant is read on the mixed spans)
                    full, fixed = self.run_cli(self.loop(span, count, "0"), gen=1)
                    self.assertGreater(self.change_gap(full, changes, n), 20 * max(good, 1e-6),
                                       "adaptive depth must change the passes")
                    self.assertGreater(self.rel(fixed, want), BUDGET, "adaptive depth must change the logits")
                # greedy continuation
                geps = self.split_eps(span, count, gen=3)
                cont = self.run_cli(self.loop(span, count, geps), gen=3)
                self.assert_clear(cont[0], geps)
                self.assertEqual(cont[0]["generated_ids"],
                                 self.ref.greedy_adaptive(self.torch_model, ids, 3, lo, hi, count, float(geps)))

    # ------------------------------------------------------------------ mode parity
    def test_every_supported_mode_is_bit_identical_at_mixed_depths(self):
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.0002"]
        modes = (["--incremental"], ["--incremental", "--kv-latent"],
                 ["--incremental", "--kv-latent", "--mla-split", "2"], trunk,
                 ["--incremental", *trunk, "--stream-lm-head"], ["--incremental", "--expert-pipeline"],
                 ["--expert-pipeline"])
        for span, count in (("3-5", 4), ("12-12", 4)):
            eps = self.split_eps(span, count)
            full = self.run_cli(self.loop(span, count, eps), gen=6)
            self.assertGreater(len(set(full[0]["loop_adaptive"]["passes"][:-1])), 1)
            for mode in modes:
                with self.subTest(span=span, mode=mode):
                    other = self.run_cli([*self.loop(span, count, eps), *mode], gen=6)
                    self.assert_same(full, other)
                    self.assertEqual(full[0]["loop_adaptive"]["passes"][:-1], other[0]["loop_adaptive"]["passes"][:-1])

    def test_streaming_row_and_low_memory_modes_at_mixed_depths(self):
        # the row pipeline, its kv_b cache and ultra-low-memory bind layers on demand, so a
        # repeated layer is simply visited again; each must be the looped full recompute
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.0002"]
        modes = ([*trunk, "--trunk-rows"], ["--incremental", *trunk, "--trunk-rows"],
                 ["--incremental", "--kv-latent", *trunk, "--trunk-rows", "--kvb-cache", "active"],
                 ["--incremental", "--kv-latent", *trunk, "--trunk-rows", "--kvb-cache", "pin"],
                 [*trunk, "--ultra-low-memory"], ["--incremental", *trunk, "--ultra-low-memory"])
        for loop in (self.loop("3-5", 4, self.split_eps("3-5", 4)), self.loop("3-8", 3), []):
            full = self.run_cli(loop, gen=5)
            for mode in modes:
                with self.subTest(loop=loop, mode=mode):
                    self.assert_same(full, self.run_cli([*loop, *mode], gen=5))

    def test_cache_trace_records_the_executed_order(self):
        # the expert requests come layer by layer in the order the token executes them,
        # repeats included (layer 0 is dense and asks for none)
        from itertools import groupby
        import struct
        out = self.root / "trace"
        out.mkdir(exist_ok=True)
        self.run_cli([*self.loop("3-5", 3), "--dump-cache-trace", out], gen=1)
        pairs = list(struct.iter_unpack("<ii", (out / "expert_trace.bin").read_bytes()))
        order = [layer for layer, _ in groupby(pairs, key=lambda p: p[0])]
        self.assertEqual(order, [1, 2, 3, 4, 5] + [3, 4, 5] * 2 + list(range(6, LAYERS)))

    # ---------------------------------------------------------------------- mutant
    def test_the_no_compaction_mutant_is_caught(self):
        ids = [int(v) for v in PROMPT.split(",")]
        env = {"K3_LOOP_MUTANT_NO_COMPACT": "1"}
        span, count = "3-5", 4
        eps = self.split_eps(span, count)
        logits, passes, changes = self.ref.forward_adaptive(self.torch_model, ids, 3, 5, count, float(eps))
        self.assertGreater(max(passes), 2, "some token must reach a pass others skipped")
        good = self.change_gap(self.run_cli(self.loop(span, count, eps), gen=1)[0], changes, len(ids))
        self.assertLess(good, 1e-4, "the real engine meets the reference")
        bad = self.run_cli(self.loop(span, count, eps), gen=1, binary=self.mutant, env=env)[0]
        self.assertTrue(bad["loop_adaptive"]["passes"][:len(ids)] != passes
                        or self.change_gap(bad, changes, len(ids)) > 20 * max(good, 1e-6),
                        "stopped tokens leaking into a pass must move the later passes")
        # the mutant build without the switch is the real thing
        self.assert_same(self.run_cli(self.loop(span, count, eps)),
                         self.run_cli(self.loop(span, count, eps), binary=self.mutant))

    def test_production_refuses_the_mutant_switch(self):
        err = self.run_cli(self.loop("3-5", 4, "0.05"), env={"K3_LOOP_MUTANT_NO_COMPACT": "1"}, ok=False)
        self.assertIn("K3_LOOP_MUTANT_NO_COMPACT", err)

    # ---------------------------------------------------------------------- default
    def test_the_default_is_the_adaptive_middle_block(self):
        for mode in ([], ["--incremental"]):
            with self.subTest(mode=mode):
                default = self.run_cli(mode)
                explicit = self.run_cli([*mode, *self.loop("6-8", 4, "0.05")])
                self.assert_same(default, explicit)
                self.assertEqual(default[0]["loop"], {"layers": [6, 8], "count": 4, "executed_layers": LAYERS + 9,
                                                      "exit": 0.05, "default": True})
                released = self.run_cli([*mode, "--no-loop"])
                self.assertIsNone(released[0]["loop"])
                self.assertIsNone(released[0]["loop_adaptive"])
                self.assertEqual(released[0]["layers_completed"], LAYERS)
                self.assert_same(released, self.run_cli([*mode, *self.loop("6-8", 1)]))
                self.assertNotEqual(default[1].tobytes(), released[1].tobytes(), "the default changes the model")
        # with --layers, the default is the block holding the middle of the bound stack
        part = self.run_cli(["--layers", "7"], gen=1)
        self.assertEqual(part[0]["loop"]["layers"], [3, 5])

    # ---------------------------------------------------------------------- refusals
    def test_refusals_name_the_reason(self):
        cases = [
            (["--loop-exit", "0.05"], "--loop-exit needs --loop-layers"),
            (["--loop-layers", "3-5", "--loop-exit", "-1"], "--loop-exit takes"),
            (["--loop-layers", "3-5", "--loop-exit", "x"], "--loop-exit takes"),
            (["--loop-layers", "3-5", "--loop-exit", "nan"], "--loop-exit takes"),
            (["--loop-layers", "3-5", "--loop-exit", "inf"], "--loop-exit takes"),
            (["--no-loop", "--loop-layers", "3-5"], "--no-loop"),
        ]
        for args, reason in cases:
            with self.subTest(args=args):
                self.assertIn(reason, self.run_cli(args, ok=False))

    # ------------------------------------------------------------- speculative decode
    def repeating_ids(self, loop):
        """A prompt the n-gram drafter proposes on (tests/test_offline_cli.py's construction): it opens and ends with
        the same 4-gram `a`, and after the opening copy comes `b`, the model's own continuation of the whole prompt,
        iterated to a fixed point, so the drafter proposes `b` and the model accepts it."""
        import random
        for seed in range(11, 19):
            rnd = random.Random(seed)
            a = [rnd.randrange(256) for _ in range(4)]
            r = [rnd.randrange(256) for _ in range(6)]
            b = [rnd.randrange(256) for _ in range(8)]
            for _ in range(8):
                got = self.run_cli(["--incremental", *loop], gen=8, ids=",".join(map(str, a + b + r + a)))[0]
                g = got["generated_ids"]
                # the drafter fires once the first generated id is b[0] (the ids after `a` then match); a longer
                # agreed prefix makes the sweep accept some of its drafts as well as reject the rest
                if g[:2] == b[:2]:
                    return ",".join(map(str, a + b + r + a))
                b = g
        self.fail("no prompt made the drafter propose under %s" % loop)

    def spec_loops(self):
        return (self.loop("3-5", 3), self.loop("3-5", 4, self.split_eps("3-5", 4)),
                self.loop("12-12", 4, self.split_eps("12-12", 4)), [])

    def test_spec_is_serial_decode_at_mixed_depths(self):
        for loop in self.spec_loops():
            ids = self.repeating_ids(loop)
            for extra in ([], ["--kv-latent"]):
                with self.subTest(loop=loop, extra=extra):
                    serial = self.run_cli(["--incremental", *extra, *loop], gen=8, ids=ids)
                    spec = self.run_cli(["--incremental", *extra, "--spec", "3", *loop], gen=8, ids=ids)
                    self.assert_same(serial, spec)
                    self.assertGreater(spec[0]["spec_sweeps"], 0, "the drafter must have proposed")
                    self.assertGreater(spec[0]["spec_accepted"], 0, "some draft must have been accepted")
                    n = len(serial[0]["full_ids"]) - 1
                    if loop:
                        self.assertEqual(serial[0]["loop_adaptive"] and serial[0]["loop_adaptive"]["passes"][:n],
                                         spec[0]["loop_adaptive"] and spec[0]["loop_adaptive"]["passes"][:n])

    def test_draft_trunk_is_serial_decode_at_mixed_depths(self):
        # the draft model proposes unlooped; the looped exact model verifies, so the output
        # must still be serial decode of the looped model, bit for bit
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.01"]
        for loop in (self.loop("3-5", 4, "0.05"), self.loop("3-8", 3), []):
            with self.subTest(loop=loop):
                serial = self.run_cli(["--incremental", *loop], gen=12)
                hybrid = self.run_cli(["--incremental", *loop, *trunk, "--draft-trunk", self.trunk], gen=12)
                self.assert_same(serial, hybrid)
                self.assertGreater(hybrid[0]["spec_sweeps"], 0)

    # ------------------------------------------------------------------ saved state
    def test_state_round_trips_under_the_loop_and_spec_saves_the_serial_state(self):
        for loop in self.spec_loops():
            with self.subTest(loop=loop):
                states = [self.path / ("s%d.%s.bin" % (len(loop), k)) for k in ("serial", "spec")]
                ids = self.repeating_ids(loop)
                serial = self.run_cli(["--incremental", *loop, "--save-state", states[0]], gen=5, ids=ids)
                spec = self.run_cli(["--incremental", *loop, "--spec", "3", "--save-state", states[1]],
                                    gen=5, ids=ids)
                self.assert_same(serial, spec)
                self.assertGreater(spec[0]["spec_sweeps"], 0, "the drafter must have proposed")
                self.assertEqual(states[0].read_bytes(), states[1].read_bytes(),
                                 "a state saved after --spec must be the serial state, byte for byte")
                # resuming continues exactly as one uninterrupted run over the same ids
                whole = self.run_cli(["--incremental", *loop], gen=3,
                                     ids=",".join(map(str, serial[0]["full_ids"] + [7, 2])))
                resumed = self.run_cli(["--incremental", *loop, "--load-state", states[0]], gen=3, ids="7,2")
                self.assertEqual(whole[0]["generated_ids"], resumed[0]["generated_ids"])
                self.assertEqual(whole[1].tobytes(), resumed[1].tobytes())
        # a state saved under one loop is refused by a run under another
        bad = self.path / "fixed.bin"
        self.run_cli(["--incremental", *self.loop("3-5", 3), "--save-state", bad], gen=1)
        for other in (self.loop("3-5", 4, "0.05"), self.loop("3-5", 2), ["--no-loop"], []):
            with self.subTest(other=other):
                err = self.run_cli(["--incremental", *other, "--load-state", bad], gen=1, ids="7", ok=False, rc=1)
                self.assertIn("loop", err)

    # ------------------------------------------------------------------ tf-check
    def test_tf_check_counts_the_reference_agreement(self):
        ids = [int(v) for v in PROMPT.split(",")]
        span, count = "3-5", 4
        eps = self.split_eps(span, count)
        logits, _, _ = self.ref.forward_adaptive(self.torch_model, ids, 3, 5, count, float(eps))
        want = sum(int(logits[i].argmax()) == ids[i + 1] for i in range(len(ids) - 1))
        out = self.path / "tf.json"
        result = subprocess.run([str(self.binary), str(self.model), "--ids", PROMPT, "--tf-check",
                                 "--cache-gb", "0.0001", "--out", str(out), *self.loop(span, count, eps)],
                                text=True, capture_output=True, env=self.env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(out.read_text())["tf_matches"], want)

    def test_score_prompt_reports_the_adaptive_loop(self):
        out = self.path / "score.json"
        result = subprocess.run([str(self.binary), str(self.model), "--ids", PROMPT, "--score-prompt",
                                 "--cache-gb", "0.0001", "--out", str(out)],
                                text=True, capture_output=True, env=self.env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text())
        self.assertTrue(report["loop"]["default"])
        passes = report["loop_adaptive"]["passes"]
        self.assertEqual(len(passes), len(PROMPT.split(",")))
        self.assertTrue(all(2 <= p <= 4 for p in passes[:-1]), passes)
        self.assertEqual(passes[-1], 0, "the last position has no target and is not forwarded")
        self.assertEqual(sum(report["loop_adaptive"]["passes_histogram"]), len(passes))
        self.assertTrue(math.isfinite(report["mean_nll"]))


if __name__ == "__main__":
    unittest.main()
