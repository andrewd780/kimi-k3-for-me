"""--loop-layers gates (docs/notes/layer-loop.md): re-running whole AttnRes blocks of the tiny K3.

--loop-layers changes the model, so these gates ask whether the engine computes the looped stack it claims to, never
whether that stack is any good:
  - identity: --loop-count 1 is bit-identical to no flag (every generated token's logits and the ids);
  - reference: the looped prompt logits match an independent torch run of the same order over the same quantized weights
    (tools/k3_ref.py loop_order) within tools/cmp_logits.py's yardstick, and the looped greedy ids match; the unlooped run
    must fall OUTSIDE that yardstick, so the gate can fail and the loop provably changes the model;
  - mode parity: incremental, --kv-latent, --mla-split, a streamed trunk, a streamed lm_head and --expert-pipeline are
    bit-identical to looped full recompute;
  - the shared-state mutant (K3_LOOP_MUTANT_SHARED_STATE, test builds only) is caught: on a span with KDA layers by the
    reference and by parity, on an MLA-only span by incremental parity (full recompute keeps no MLA state to corrupt);
  - refusals: malformed, misaligned or out-of-range spans, bad counts and every unsupported combination exit 2 and say
    why; a production build refuses the mutant switch.

Needs numpy and torch. Build first (`make all mutants`).
K3_TEST_BIN=bin/k3 K3_TEST_MUT_BIN=bin/k3-mut python3 -m unittest discover -s tests -p test_layer_loop.py -v
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

PROMPT = "3,7,11,5,9"
LAYERS = 13                     # tiny_config: blocks of 3, boundaries at 0, 3, 6, 9, 12
REF_SPECS = ("3-5:2", "3-8:3", "12-12:3", "0-2:2")   # a mixed block, two blocks x3, the MLA-only tail, block 0
BUDGET = 1.2e-7 * math.sqrt(128) * 50                # tools/cmp_logits.py at hidden 128
VOCAB = 256


class LayerLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.binary = (ROOT / os.environ.get("K3_TEST_BIN", "bin/k3")).resolve()
        cls.mutant = (ROOT / os.environ.get("K3_TEST_MUT_BIN", "bin/k3-mut")).resolve()
        cls.env = {**os.environ, "OMP_NUM_THREADS": "2", "K3_NOHUGE": "1"}
        cls.env.pop("K3_LOOP_MUTANT_SHARED_STATE", None)
        cls.model = cls.root / "plain"
        loops = [arg for spec in REF_SPECS for arg in ("--loop", spec)]
        subprocess.run([sys.executable, str(ROOT / "tools/make_tiny_checkpoint.py"), str(cls.model),
                        "--prompt-ids", PROMPT, *loops], check=True, capture_output=True, env=cls.env)
        cls.trunk = cls.root / "trunk"
        if pack_trunk.main([str(cls.model), str(cls.trunk), str(LAYERS)]):
            raise AssertionError("tiny trunk packing failed")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.counter = 0

    def run_cli(self, args, gen=6, binary=None, env=None, ok=True):
        """Every generated token's logits (--dump-all-logits) and the report."""
        self.counter += 1
        out = self.path / ("%d.json" % self.counter)
        logits = self.path / ("%d.f32" % self.counter)
        result = subprocess.run([str(binary or self.binary), str(self.model), "--ids", PROMPT, "--gen", str(gen),
                                 "--cache-gb", "0.0001", "--out", str(out), "--dump-all-logits", str(logits),
                                 *map(str, args)],
                                text=True, errors="replace", capture_output=True,
                                env={**self.env, **(env or {})}, timeout=120)
        if not ok:
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertFalse(out.exists(), "a refused run must not write a report")
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text(errors="replace"))
        self.assertEqual(report["expert_drops"], 0)
        raw = np.fromfile(logits, dtype=np.float32)
        self.assertEqual(raw.size, gen * VOCAB, "one complete vector per generated token")
        return report, raw

    def assert_same(self, a, b):
        self.assertEqual(a[0]["full_ids"], b[0]["full_ids"])
        self.assertEqual(a[1].tobytes(), b[1].tobytes(), "every generated token's logits must be bit-identical")

    @staticmethod
    def loop(spec):
        span, count = spec.split(":")
        return ["--loop-layers", span, "--loop-count", count]

    def reference(self, spec):
        span, count = spec.split(":")
        ref = json.loads((self.model / ("ref_logits_loop_%s_%s.json" % (span, count))).read_text())
        self.assertEqual(ref["prompt_ids"], [int(v) for v in PROMPT.split(",")])
        return ref, np.array(ref["logits_bits"], dtype=np.uint32).view(np.float32)

    @staticmethod
    def rel(c, r):
        return float(np.abs(c.astype(np.float64) - r.astype(np.float64)).max()) / max(float(np.abs(r).max()), 1e-30)

    # ------------------------------------------------------------------ identity
    def test_count_one_is_the_unlooped_model(self):
        for mode in ([], ["--incremental"], ["--incremental", "--kv-latent"]):
            for span in ("3-5", "0-2", "12-12"):
                with self.subTest(mode=mode, span=span):
                    base = self.run_cli(mode)
                    once = self.run_cli([*mode, "--loop-layers", span, "--loop-count", "1"])
                    self.assert_same(base, once)
                    self.assertIsNone(base[0]["loop"])
                    self.assertEqual(once[0]["loop"]["executed_layers"], LAYERS)
                    self.assertEqual(once[0]["layers_completed"], LAYERS)

    # ----------------------------------------------------------------- reference
    def test_matches_the_torch_reference_and_the_gate_can_fail(self):
        plain = self.run_cli([], gen=1)[1]
        for spec in REF_SPECS:
            with self.subTest(spec=spec):
                ref, r = self.reference(spec)
                looped = self.run_cli(self.loop(spec), gen=1)
                span, count = spec.split(":")
                lo, hi = (int(v) for v in span.split("-"))
                executed = LAYERS + (int(count) - 1) * (hi - lo + 1)
                self.assertEqual(looped[0]["layers_completed"], executed)
                self.assertEqual(looped[0]["loop"], {"layers": [lo, hi], "count": int(count),
                                                     "executed_layers": executed})
                self.assertLessEqual(self.rel(looped[1], r), BUDGET, "looped engine vs looped reference")
                self.assertEqual(int(np.argmax(looped[1])), ref["argmax"])
                # the plant: the unlooped engine must NOT pass against the looped reference
                self.assertGreater(self.rel(plain, r), 100 * BUDGET, "the loop must change the model")
                generated = self.run_cli(self.loop(spec), gen=len(ref["greedy_ids"]))
                self.assertEqual(generated[0]["generated_ids"], ref["greedy_ids"])

    # --------------------------------------------------------------- mode parity
    def test_every_supported_mode_is_bit_identical_to_full_recompute(self):
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.0002"]
        modes = (["--incremental"], ["--incremental", "--kv-latent"],
                 ["--incremental", "--kv-latent", "--mla-split", "2"], trunk,
                 ["--incremental", *trunk, "--stream-lm-head"], ["--incremental", "--expert-pipeline"],
                 ["--expert-pipeline"])
        for spec in ("3-5:2", "12-12:3", "3-8:3"):
            full = self.run_cli(self.loop(spec))
            for mode in modes:
                with self.subTest(spec=spec, mode=mode):
                    self.assert_same(full, self.run_cli([*self.loop(spec), *mode]))

    # ------------------------------------------------------------------- mutant
    def test_the_shared_state_mutant_is_caught(self):
        env = {"K3_LOOP_MUTANT_SHARED_STATE": "1"}
        # a span with KDA layers: the reference itself sees it
        _, r = self.reference("3-5:2")
        mutated = self.run_cli(self.loop("3-5:2"), gen=1, binary=self.mutant, env=env)
        self.assertGreater(self.rel(mutated[1], r), 2 * BUDGET)
        # parity sees it for both span kinds; the MLA-only span only once a cache exists
        for spec, modes in (("3-5:2", ([], ["--incremental"])),
                            ("12-12:3", (["--incremental"], ["--incremental", "--kv-latent"]))):
            full = self.run_cli(self.loop(spec))
            for mode in modes:
                with self.subTest(spec=spec, mode=mode):
                    bad = self.run_cli([*self.loop(spec), *mode], binary=self.mutant, env=env)
                    self.assertNotEqual(full[1].tobytes(), bad[1].tobytes(), "the mutant must change the logits")
        # and the mutant build without the switch is the real thing
        self.assert_same(self.run_cli(self.loop("3-5:2")), self.run_cli(self.loop("3-5:2"), binary=self.mutant))

    def test_production_refuses_the_mutant_switch(self):
        err = self.run_cli(self.loop("3-5:2"), env={"K3_LOOP_MUTANT_SHARED_STATE": "1"}, ok=False)
        self.assertIn("K3_LOOP_MUTANT_SHARED_STATE", err)

    # ----------------------------------------------------------------- refusals
    def test_refusals_name_the_reason(self):
        cases = [
            (["--loop-layers", "2-5"], "whole AttnRes blocks"),     # starts mid-block
            (["--loop-layers", "3-4"], "whole AttnRes blocks"),     # ends mid-block
            (["--loop-layers", "12-13"], "whole AttnRes blocks"),   # past the last layer
            (["--loop-layers", "6-8", "--layers", "6"], "whole AttnRes blocks"),   # outside a partial stack
            (["--loop-layers", "5-3"], "ends before it starts"),
            (["--loop-layers", "3"], "A-B"), (["--loop-layers", "3-"], "A-B"), (["--loop-layers", "-5"], "A-B"),
            (["--loop-layers", "a-b"], "A-B"), (["--loop-layers", "3-5x"], "A-B"),
            (["--loop-layers", "3-5", "--loop-count", "0"], "1 to 8"),
            (["--loop-layers", "3-5", "--loop-count", "9"], "1 to 8"),
            (["--loop-layers", "3-5", "--loop-count", "2x"], "1 to 8"),
            (["--loop-count", "2"], "needs --loop-layers"),
            (["--loop-layers", "3-5", "--incremental", "--spec", "2"], "--spec"),
            (["--loop-layers", "3-5", "--save-state", self.root / "s.bin"], "--save-state"),
            (["--loop-layers", "3-5", "--incremental", "--load-state", self.root / "s.bin"], "--load-state"),
            (["--loop-layers", "3-5", "--trunk", self.trunk, "--trunk-gb", "0.0002", "--trunk-rows"], "--trunk-rows"),
            (["--loop-layers", "3-5", "--trunk", self.trunk, "--preset", "ultra"], "--ultra-low-memory"),
            (["--loop-layers", "3-5", "--dump-cache-trace", self.root], "--dump-cache-trace"),
            (["--loop-layers", "3-5", "--tf-check"], "--tf-check"),
        ]
        for args, reason in cases:
            with self.subTest(args=args):
                err = self.run_cli(args, ok=False)
                self.assertIn(reason, err)
                self.assertIn("--loop", err)

    def test_score_prompt_reports_the_loop(self):
        out = self.path / "score.json"
        result = subprocess.run([str(self.binary), str(self.model), "--ids", "3,7,11,5,9,1,2,4",
                                 "--score-prompt", "--cache-gb", "0.0001", "--out", str(out),
                                 *self.loop("3-5:2")], text=True, capture_output=True, env=self.env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["layers_completed"], 16)
        self.assertEqual(report["loop"]["executed_layers"], 16)
        self.assertTrue(math.isfinite(report["mean_nll"]))


if __name__ == "__main__":
    unittest.main()
