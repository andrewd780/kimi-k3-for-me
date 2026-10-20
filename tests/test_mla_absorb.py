"""--mla-absorb, the absorbed attention over the latent cache, on the tiny K3.

Under --kv-latent --mla-absorb each query is scored against the cached latents through kv_b's key rows once, and the
probability-weighted latent sum is taken through its value rows once, instead of rebuilding k and v for every cached
position for every query. It is the same function in another floating-point order, so these gates hold it to a
tolerance where the exact layouts are held to each other bit for bit:
  - against the exact latent layout and the torch reference (tools/k3_ref.py): every generated token's logits within
    tools/cmp_logits.py's yardstick, and the same greedy ids;
  - deterministic: bit-identical at any --prefill-chunk, on one thread or three, under the layer loop, and across a
    paused and resumed read-in (--checkpoint);
  - the gate can fail: the causal mutant (K3_ABSORB_MUTANT_CAUSAL, a query sees one position too few) leaves the
    yardstick. The rope mutant (K3_ABSORB_MUTANT_NO_ROPE) does NOT on this tiny model, whose rope slot is a small
    share of each score; tests/unit/test_mla_variants.c holds the engine's absorbed form to the plain scalar formula
    bit for bit (its check 10), where both mutants fail, and `make test` runs the rope mutant against it;
  - refusals name their reason.

Needs numpy and torch. Build first (`make all mutants`).
K3_TEST_BIN=bin/k3 K3_TEST_MUT_BIN=bin/k3-mut python3 -m unittest discover -s tests -p test_mla_absorb.py -v
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

VOCAB = 256
LAYERS = 13
PROMPT = [3, 7, 11, 5, 9, 1, 2, 4, 250, 17, 33, 8, 90, 61, 44, 12, 200, 6, 77, 19, 23]
BUDGET = 1.2e-7 * math.sqrt(128) * 50             # tools/cmp_logits.py at hidden 128
ABSORB = ["--incremental", "--kv-latent", "--mla-absorb"]


class MlaAbsorbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.binary = (ROOT / os.environ.get("K3_TEST_BIN", "bin/k3")).resolve()
        cls.mutant = (ROOT / os.environ.get("K3_TEST_MUT_BIN", "bin/k3-mut")).resolve()
        cls.env = {**os.environ, "OMP_NUM_THREADS": "2", "K3_NOHUGE": "1"}
        for k in ("K3_ABSORB_MUTANT_NO_ROPE", "K3_ABSORB_MUTANT_CAUSAL"):
            cls.env.pop(k, None)
        cls.model = cls.root / "plain"
        subprocess.run([sys.executable, str(ROOT / "tools/make_tiny_checkpoint.py"), str(cls.model)],
                       check=True, capture_output=True, env=cls.env)
        cls.trunk = cls.root / "trunk"
        if pack_trunk.main([str(cls.model), str(cls.trunk), str(LAYERS)]):
            raise AssertionError("tiny trunk packing failed")
        import make_tiny_checkpoint
        _, cls.torch_model = make_tiny_checkpoint.reference_model()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.counter = 0

    def run_cli(self, args, ids=PROMPT, gen=6, binary=None, env=None, ok=True, rc=2):
        self.counter += 1
        out = self.path / ("%d.json" % self.counter)
        logits = self.path / ("%d.f32" % self.counter)
        result = subprocess.run([str(binary or self.binary), str(self.model), "--ids", ",".join(map(str, ids)),
                                 "--gen", str(gen), "--cache-gb", "0.0001", "--out", str(out),
                                 "--dump-all-logits", str(logits), *map(str, args)],
                                text=True, errors="replace", capture_output=True,
                                env={**self.env, **(env or {})}, timeout=300)
        if not ok:
            self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
            self.assertFalse(out.exists(), "a refused or paused run must not write a report")
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text(errors="replace"))
        self.assertEqual(report["expert_drops"], 0)
        raw = np.fromfile(logits, dtype=np.float32).reshape(gen, VOCAB)
        return report, raw

    @staticmethod
    def rel(c, r):
        c, r = np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)
        return float(np.abs(c - r).max()) / max(float(np.abs(r).max()), 1e-30)

    def assert_same(self, a, b):
        self.assertEqual(a[0]["full_ids"], b[0]["full_ids"])
        self.assertEqual(a[1].tobytes(), b[1].tobytes(), "every generated token's logits must be bit-identical")

    def worst(self, a, b):
        return max(self.rel(x, y) for x, y in zip(a[1], b[1]))

    # ------------------------------------------------------------------ the right function
    def test_absorbed_logits_match_the_exact_layout_and_the_torch_reference(self):
        import torch
        exact = self.run_cli(["--incremental", "--kv-latent"])
        absorbed = self.run_cli(ABSORB)
        self.assertTrue(absorbed[0]["mla_absorb"])
        self.assertFalse(exact[0]["mla_absorb"])
        self.assertEqual(absorbed[0]["full_ids"], exact[0]["full_ids"])
        worst = self.worst(absorbed, exact)
        self.assertLessEqual(worst, BUDGET, "absorbed vs exact, worst generated token")
        self.assertNotEqual(absorbed[1].tobytes(), exact[1].tobytes(),
                            "the absorbed form is another floating-point order; identical bits would mean it never ran")
        # the torch reference: greedy ids, and the logits behind each generated token from a full re-forward
        want = self.torch_model.greedy(torch.tensor([PROMPT]), 6)[0].tolist()
        self.assertEqual(absorbed[0]["full_ids"], want)
        with torch.no_grad():
            ref, _ = self.torch_model.forward(torch.tensor([want[:-1]]))
        ref = ref[0, len(PROMPT) - 1:].float().numpy()
        for i in range(6):
            with self.subTest(token=i):
                self.assertLessEqual(self.rel(absorbed[1][i], ref[i]), BUDGET, "absorbed vs the torch reference")
                self.assertLessEqual(self.rel(exact[1][i], ref[i]), BUDGET, "exact vs the torch reference")

    def test_the_causal_mutant_leaves_the_yardstick(self):
        absorbed = self.run_cli(ABSORB)
        bad = self.run_cli(ABSORB, binary=self.mutant, env={"K3_ABSORB_MUTANT_CAUSAL": "1"})
        self.assertGreater(self.worst(bad, absorbed), BUDGET)
        # the rope mutant moves the logits, but by less than the yardstick on this model: the unit gate catches it
        rope = self.run_cli(ABSORB, binary=self.mutant, env={"K3_ABSORB_MUTANT_NO_ROPE": "1"})
        self.assertGreater(self.worst(rope, absorbed), 10 * max(self.worst(absorbed, self.run_cli(
            ["--incremental", "--kv-latent"])), 1e-9))
        for name in ("K3_ABSORB_MUTANT_NO_ROPE", "K3_ABSORB_MUTANT_CAUSAL"):
            with self.subTest(production=name):
                self.assertIn("gate mutant", self.run_cli(ABSORB, env={name: "1"}, ok=False))

    # ------------------------------------------------------------------ deterministic
    def test_absorbed_is_bit_identical_across_chunking_threads_loop_and_resume(self):
        for loop in ([], ["--loop"]):
            with self.subTest(loop=loop):
                one = self.run_cli([*ABSORB, *loop])
                for args, env in ((["--prefill-chunk", 1], None), (["--prefill-chunk", 5], None),
                                  ([], {"OMP_NUM_THREADS": "1"}), (["--prefill-chunk", 7], {"OMP_NUM_THREADS": "3"})):
                    with self.subTest(args=args, env=env):
                        self.assert_same(one, self.run_cli([*ABSORB, *loop, *args], env=env))
                ck = self.path / ("ck%d.state" % len(loop))
                self.run_cli([*ABSORB, *loop, "--prefill-chunk", 4, "--checkpoint", ck, "--pause-after", 2],
                             ok=False, rc=3)
                resumed = self.run_cli([*ABSORB, *loop, "--prefill-chunk", 4, "--checkpoint", ck])
                self.assertEqual(resumed[0]["checkpoint_resumed_ids"], 8)
                self.assert_same(one, resumed)

    # ------------------------------------------------------------------ refusals
    def test_refusals_name_the_reason(self):
        cases = [(["--incremental", "--mla-absorb"], "needs --kv-latent"),
                 ([*ABSORB, "--mla-split", 4], "--mla-split"),
                 ([*ABSORB, "--trunk", self.trunk, "--trunk-rows"], "--trunk-rows"),
                 ([*ABSORB, "--trunk", self.trunk, "--draft-trunk", self.trunk], "--draft-trunk")]
        for args, reason in cases:
            with self.subTest(args=args):
                self.assertIn(reason, self.run_cli(args, ok=False))


if __name__ == "__main__":
    unittest.main()
