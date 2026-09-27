"""--mla-split gates (the private research notes, handoff review B1: split L1s, rebuild once per query block).

With --kv-latent, a forward over T positions rebuilt each visible position's k and v once PER QUERY; --mla-split B
rebuilds them once per block of B queries. The unit test (test_mla_variants, 7b) holds every score, normaliser,
quotient, accumulator and appended row bitwise and checks the new closed form. These CLI gates check, on the tiny model:
  - every emitted position's logits and ids are identical to --mla-split 0, for prompt passes and for speculative verify
    sweeps with 0..3 accepted drafts and a later full acceptance, on a resident model and through --trunk-rows with the
    kv_b buffer;
  - the rebuild count through the kv_b buffer's counters: for a single forward over T prompt positions, every MLA layer
    serves 2 x sum over blocks of (block end) passes instead of T(T+1);
  - a causal-cut mutant (K3_MLA_SPLIT_MUTANT_CAUSAL) changes the logits;
  - out-of-range block sizes are refused.

Build with ZSTD=1 first. K3_TEST_BIN=bin/k3 python3 -m unittest tests.test_mla_split -v
"""
from __future__ import annotations

import unittest

import test_offline_cli as toc   # not imported by name, so unittest does not rerun its tests

MLA_LAYERS = 4


class MlaSplitTests(unittest.TestCase):
    setUpClass = classmethod(toc.OfflineCliTests.setUpClass.__func__)
    setUp = toc.OfflineCliTests.setUp
    run_cli = toc.OfflineCliTests.run_cli
    run_logits = toc.OfflineCliTests.run_logits
    assert_same_run = toc.OfflineCliTests.assert_same_run
    assert_no_replay = toc.OfflineCliTests.assert_no_replay
    crafted = toc.OfflineCliTests.crafted
    SPEC = toc.OfflineCliTests.SPEC
    _crafted = {}

    def test_prompt_and_speculative_trajectories_are_identical(self):
        for accept, second in [(0, False), (1, False), (2, True), (3, False)]:
            prompt, _ = self.crafted(accept, second)
            ids, gen = ",".join(map(str, prompt)), str(accept + self.SPEC + 3)
            for where in ([], ["--trunk", self.trunk, "--trunk-gb", "0.0002", "--trunk-rows", "--kvb-cache", "active"]):
                common = ["--ids", ids, "--gen", gen, "--incremental", "--kv-latent", *where]
                base = self.run_logits(self.selective, common)
                for B in ("2", "4", "8"):
                    with self.subTest(accept=accept, rows=bool(where), B=B):
                        self.assert_same_run(base, self.run_logits(self.selective, [*common, "--mla-split", B]))
                        spec = self.run_logits(self.selective, [*common, "--mla-split", B, "--spec", str(self.SPEC)])
                        self.assert_same_run(base, spec)
                        self.assert_no_replay(spec[0])
                        self.assertEqual(spec[0]["spec_trace"][0], [self.SPEC, accept, accept + 1])

    def test_rebuilds_fall_to_the_closed_form(self):
        T = 8
        ids = ",".join(str((7 * i + 3) % 256) for i in range(T))
        common = ["--ids", ids, "--gen", "1", "--incremental", "--kv-latent", "--trunk", self.trunk,
                  "--trunk-gb", "0.0002", "--trunk-rows", "--kvb-cache", "active"]
        def served(r):
            return r["trunk_kvb_fills"] + r["trunk_kvb_hits"]
        base = self.run_cli(self.selective, common)
        self.assertEqual(served(base[0]), MLA_LAYERS * T * (T + 1))
        for B in (2, 3, 8):
            r = self.run_cli(self.selective, [*common, "--mla-split", str(B)])
            self.assertEqual(r[1], base[1])
            want = 2 * sum(min(T, t0 + B) for t0 in range(0, T, B))
            self.assertEqual(served(r[0]), MLA_LAYERS * want, "B=%d" % B)

    def test_causal_cut_mutant_changes_the_logits(self):
        ids = ",".join(str((7 * i + 3) % 256) for i in range(8))
        common = ["--ids", ids, "--gen", "2", "--incremental", "--kv-latent", "--mla-split", "4"]
        base = self.run_cli(self.selective, common)
        bad = self.run_cli(self.selective, common, env={"K3_MLA_SPLIT_MUTANT_CAUSAL": "1"})
        self.assertNotEqual(base[1], bad[1], "a query that sees later positions of its block must change the logits")

    def test_refusals(self):
        for b in ("-1", "5000"):
            r = self.run_cli(self.selective, ["--ids", "3", "--mla-split", b], ok=False)
            self.assertIn("--mla-split takes", r.stderr)


if __name__ == "__main__":
    unittest.main()
