"""Tests for tools/spec_replay.py: the P0 port, the incremental drafters, the suffix
automaton, the cost model and the step accounting. Synthetic sequences only; runs in
seconds and needs neither corpora nor the tokenizers package."""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spec_replay as sr


def _random_seqs(seed, count, max_len=60):
    rng = random.Random(seed)
    for _ in range(count):
        alpha = rng.randint(2, 5)
        yield [rng.randint(1, alpha) for _ in range(rng.randint(1, max_len))]


class SpecDraftPortTests(unittest.TestCase):
    """Expected drafts worked by hand from the C source of spec_draft()."""

    def test_single_occurrence_copies_to_the_end_of_history(self):
        seq = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 11, 8), [5, 6, 7, 1, 2, 3, 4])
        seq = [1, 2, 3, 4, 7, 9, 1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 10, 8), [7, 9, 1, 2, 3, 4])
        self.assertEqual(sr.spec_draft(seq, 10, 2), [7, 9])

    def test_two_occurrences_stop_where_they_disagree(self):
        seq = [1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5, 6, 8, 1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 18, 8), [5, 6])

    def test_disagreeing_first_id_means_no_draft(self):
        seq = [1, 2, 3, 4, 7, 1, 2, 3, 4, 8, 9, 1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 15, 8), [])

    def test_falls_back_to_the_trigram_when_the_4gram_disagrees(self):
        seq = [1, 2, 3, 4, 6, 9, 1, 2, 3, 4, 5, 9, 7, 2, 3, 4, 5, 9, 1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 22, 8), [5, 9])

    def test_older_continuation_may_not_run_into_the_newer_match(self):
        # period 4: at n=4 the older continuation starts at the newer match (no draft);
        # at n=3 one id fits before it does.
        seq = [1, 2, 3, 4] * 3
        self.assertEqual(sr.spec_draft(seq, 12, 8), [1])

    def test_cap_is_clamped_to_k3_spec_max(self):
        seq = list(range(1, 30)) + [1, 2, 3, 4]
        self.assertEqual(sr.spec_draft(seq, 33, 20), list(range(5, 13)))
        self.assertEqual(sr.spec_draft(seq, 33, 0), [])

    def test_short_or_unmatched_history(self):
        self.assertEqual(sr.spec_draft([1, 2, 3], 3, 8), [])
        self.assertEqual(sr.spec_draft([1, 2, 3, 1], 4, 8), [])
        # a 2-gram match is below P0's minimum; P1 takes it
        seq = [1, 2, 5, 1, 2]
        self.assertEqual(sr.spec_draft(seq, 5, 8), [])
        self.assertEqual(sr.eager_draft(seq, 5, 8), [5, 1, 2])

    def test_uses_only_the_prefix_before_T(self):
        seq = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 6, 6]
        self.assertEqual(sr.spec_draft(seq, 9, 8), [5, 1, 2, 3, 4])

    def test_incremental_index_equals_the_port(self):
        for seq in _random_seqs(1, 400):
            idx = sr.NgramIndex(seq)
            for T in range(1, len(seq) + 1):
                idx.advance()
                for cap in (1, 2, 3, 8, 11):
                    self.assertEqual(idx.p0(cap), sr.spec_draft(seq, T, cap), (seq, T, cap))
                    self.assertEqual(idx.p1(cap), sr.eager_draft(seq, T, cap), (seq, T, cap))


class SuffixAutomatonTests(unittest.TestCase):
    def test_longest_match_most_recent_end_and_counts(self):
        for seq in _random_seqs(2, 300, max_len=50):
            sam = sr.SuffixAutomaton()
            for pos in range(len(seq)):
                v = sam.extend(seq[pos], pos)
                L, e, count = sr.longest_suffix_match(seq, pos + 1)
                self.assertEqual(sam.length[v] if v > 0 else 0, L)
                if v > 0:
                    self.assertEqual(sam.last_end[v], e)
                    self.assertEqual(sam.cnt[v], count)
                    # the first draft id's count: occurrences of suffix + d0 in the prefix
                    d0 = seq[e + 1]
                    w = sam.nxt[v][d0]
                    pat = seq[pos + 1 - L:pos + 1] + [d0]
                    brute = sum(seq[i:i + len(pat)] == pat
                                for i in range(pos + 2 - len(pat)))
                    self.assertEqual(sam.cnt[w], brute)
                sam.commit(pos)

    def test_periodic_copy_continues_past_the_history(self):
        seq = [5, 6, 7, 5, 6]
        L, e, _ = sr.longest_suffix_match(seq, 5)
        self.assertEqual((L, e), (2, 1))
        self.assertEqual(sr.periodic_copy(seq, e + 1, 4 - e, 8), [7, 5, 6, 7, 5, 6, 7, 5])


class CostModelTests(unittest.TestCase):
    def test_plain_step_costs_one(self):
        for tier in sr.TIERS:
            self.assertAlmostEqual(sr.distinct_experts(1), 1.0)
            self.assertEqual(sr.verify_cost(0, tier), 1.0)
            c = sr.TIERS[tier]
            self.assertAlmostEqual(c["a_T"] + c["a_E"] + c["a_C"], 1.0)

    def test_verify_and_replay(self):
        c = sr.TIERS["8GB"]
        u5 = 56 * (1 - (55 / 56) ** 5)
        self.assertAlmostEqual(sr.verify_cost(4, "8GB"), 0.78 + 0.19 * u5 + 0.03 * 5)
        self.assertAlmostEqual(sr.step_cost(4, 4, "8GB", replay=True),
                               sr.verify_cost(4, "8GB"))
        u3 = 56 * (1 - (55 / 56) ** 3)
        self.assertAlmostEqual(sr.step_cost(4, 2, "8GB", replay=True),
                               sr.verify_cost(4, "8GB") + c["a_T"] + c["a_E"] * u3
                               + c["a_C"] * 3)
        self.assertAlmostEqual(sr.step_cost(4, 2, "8GB", replay=False),
                               sr.verify_cost(4, "8GB"))
        self.assertAlmostEqual(sr.verify_cost(4, "8GB", eps=0.05),
                               sr.verify_cost(4, "8GB") + 0.05)

    def test_first_draft_marginal_cost_is_delta(self):
        self.assertAlmostEqual(sr.marginal_cost(1, "8GB"), 0.19 * 55 / 56 + 0.03)
        self.assertAlmostEqual(sr.marginal_cost(1, "8GB"), sr.DELTA["8GB"], delta=0.005)
        self.assertAlmostEqual(sr.marginal_cost(1, "128GB"), sr.DELTA["128GB"], delta=0.005)
        self.assertLess(sr.marginal_cost(8, "8GB"), sr.marginal_cost(1, "8GB"))


class SimulationTests(unittest.TestCase):
    def test_orbit_jumps_by_emitted(self):
        self.assertEqual(sr.orbit([3, 1, 1, 2, 1, 1]), [0, 3, 5])

    def test_step_accounting(self):
        rem = np.array([6, 5, 4, 3, 2, 1])
        nd = np.array([2, 2, 2, 2, 2, 2])
        acc = np.array([2, 0, 1, 0, 0, 0])
        X = sr.simulate(nd, acc, rem, np.zeros(6, np.int64), 1)[0]
        g = lambda k: X[sr.COL[k]]  # noqa: E731
        # t=0 accepts 2 (emits 3); t=3, 4 and 5 accept none (emit 1 each)
        self.assertEqual((g("steps"), g("fired"), g("drafted"), g("accepted"), g("emitted")),
                         (4, 4, 8, 2, 6))
        v, r = sr.verify_cost(2, "8GB"), sr.replay_cost(0, "8GB")
        self.assertAlmostEqual(g("cost_8GB_free"), 4 * v)
        self.assertAlmostEqual(g("cost_8GB_today"), 4 * v + 3 * r)
        self.assertEqual((g("att1"), g("acc1"), g("att2"), g("acc2")), (4, 1, 1, 1))

    def test_copy_of_the_prompt(self):
        USER, END, ASSISTANT = 100, 101, 102
        body = [1, 2, 3, 4, 5, 6, 7, 8]
        seq = [USER] + body + [END, ASSISTANT] + body + [END]
        A = sr.analyze_sequence(seq, [(11, 20)])
        self.assertEqual(list(A["rem"]), [8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(int(A["L2"][0]), 1)             # only "1" has been seen
        self.assertEqual(int(A["a2"][0]), 8)             # its copy runs through END
        self.assertEqual(int(A["l0"][0]), 0)             # no 3-gram yet
        # (1,2,3) matched: P0 copies 4..8, END, ASSISTANT, 1 and the turn keeps 6 of them
        self.assertEqual((int(A["l0"][2]), int(A["a0"][2])), (8, 6))
        nd = np.where(A["L2"] >= 2, 8, 0)
        X = sr.simulate(nd, A["a2"], A["rem"], np.zeros(8, np.int64), 1)[0]
        self.assertEqual((X[sr.COL["steps"]], X[sr.COL["emitted"]]), (2, 8))
        self.assertAlmostEqual(X[sr.COL["cost_8GB_free"]], 1 + sr.verify_cost(8, "8GB"))

    def test_turn_boundaries_are_respected(self):
        # two turns; a draft may never be credited with ids of the next prompt
        seq = [9, 1, 2, 3, 7, 8, 1, 2, 3, 7, 9, 1, 2, 3, 7, 8, 1, 2, 3, 7]
        A = sr.analyze_sequence(seq, [(5, 10), (15, 20)])
        self.assertEqual(len(A["rem"]), 8)
        self.assertTrue(np.all(A["a0"] <= A["rem"]) and np.all(A["a2"] <= A["rem"]))
        for nd, acc in ((A["l1"], A["a1"]), (np.full(8, 8), A["a2"])):
            X = sr.simulate(nd, acc, A["rem"], np.zeros(8, np.int64), 1)[0]
            self.assertEqual(X[sr.COL["emitted"]], 8)


class AdaptiveLengthTests(unittest.TestCase):
    def test_cumulative_threshold(self):
        q = np.full(sr.N_CODES, 0.5)
        q[10], q[11], q[12] = 0.9, 0.8, 0.95
        f3 = np.array([[10, 11, 12, 12, 12, 12, 12, 12],
                       [sr.NO_DRAFT] * 8], np.uint8)
        # cumulative 0.9, 0.72, 0.684, 0.65, 0.617, 0.586, 0.557, 0.529
        self.assertEqual(list(sr.adaptive_lengths(f3, q, 0.22)), [8, 0])
        self.assertEqual(list(sr.adaptive_lengths(f3, q, 0.55)), [7, 0])
        self.assertEqual(list(sr.adaptive_lengths(f3, q, 0.7)), [2, 0])
        self.assertEqual(list(sr.adaptive_lengths(f3, q, 0.95)), [0, 0])
        self.assertEqual(list(sr.adaptive_lengths(f3, q, 0.85, cumulative=False)), [1, 0])

    def test_cost_aware_rule(self):
        q = np.full(sr.N_CODES, 0.5)
        q[7] = 0.9
        f3 = np.array([[7] * 8, [3] * 8, [sr.NO_DRAFT] * 8], np.uint8)
        # q = 0.5 at 8 GB: 0.5 and 0.25 beat the ~0.21 marginal cost, 0.125 does not;
        # with today's replay the first draft needs 0.5 > 0.217 + 0.5 * 1.0, which fails
        self.assertEqual(list(sr.cost_aware_lengths(f3, q, "8GB", replay=False)), [8, 2, 0])
        self.assertEqual(list(sr.cost_aware_lengths(f3, q, "8GB", replay=True)), [8, 0, 0])
        # q = 0.9 at 128 GB: 0.9^6 = 0.531 > 0.514 but 0.9^7 = 0.478 < 0.507; today,
        # draft 4 needs 0.656 > 0.529 + (0.729 - 0.656) * 2.632 = 0.721, which fails
        self.assertEqual(int(sr.cost_aware_lengths(f3, q, "128GB", replay=False)[0]), 6)
        self.assertEqual(int(sr.cost_aware_lengths(f3, q, "128GB", replay=True)[0]), 3)

    def test_table_recovers_rates(self):
        rng = np.random.default_rng(0)
        n = 20000
        code, code2 = sr.feature_code(1, 5, 3, 3), sr.feature_code(2, 6, 3, 3)
        f3 = np.full((n, 8), code2, np.uint8)
        f3[:, 0] = code
        acc = (rng.random(n) < 0.7).astype(np.int64)     # accept the first id or not
        q, info = sr.fit_acceptance_table(f3, acc, np.full(n, 8))
        self.assertAlmostEqual(q[code], 0.7, delta=0.02)
        self.assertLess(q[code2], 0.01)                  # never accepted at depth 2
        self.assertEqual(info["samples"], n + int(acc.sum()))

    def test_codes_fit_in_a_byte(self):
        codes = {sr.feature_code(j, ell, n, k) for j in range(1, 9) for ell in range(1, 90)
                 for n in range(0, 6) for k in range(0, 6)}
        self.assertEqual(min(codes), 0)
        self.assertLess(max(codes), sr.N_CODES)
        self.assertLess(sr.N_CODES, sr.NO_DRAFT)


class BootstrapTests(unittest.TestCase):
    def test_interval_contains_the_point(self):
        rng = np.random.default_rng(1)
        X = np.zeros((40, len(sr.STAT_COLS)))
        X[:, sr.COL["steps"]] = rng.integers(50, 100, 40)
        X[:, sr.COL["fired"]] = X[:, sr.COL["steps"]] // 2
        X[:, sr.COL["emitted"]] = X[:, sr.COL["steps"]] * 1.5
        for tier in sr.TIERS:
            for mode in ("free", "today"):
                X[:, sr.COL["cost_%s_%s" % (tier, mode)]] = X[:, sr.COL["steps"]] * 1.2
        s = sr.summarize(X, sr.bootstrap_weights(40, 500, seed=3))
        pt, lo, hi = s["fire_rate"]
        self.assertLessEqual(lo, pt)
        self.assertLessEqual(pt, hi)
        self.assertAlmostEqual(s["speedup_8GB_free"][0], 1.25)
        self.assertIsNone(s["acceptance_rate"])          # nothing drafted: 0/0


class ReplayIdsTests(unittest.TestCase):
    def test_replay_of_emitted_ids(self):
        body = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        docs = [{"id": "a", "turns": [[[1] + body + [2], body + [3]]]},
                {"id": "b", "turns": [[[1, 5, 6], [7, 8, 3]], [[4] + body, body + [3]]]}]
        ref = {"p3_table": {"q": [0.9] * sr.N_CODES},
               "headline_configs": {"8GB": {"P0": "P0/N8", "P3": "P3/d0.22/N8"}}}
        with tempfile.TemporaryDirectory() as d:
            ids, rj, out = (str(Path(d) / n) for n in ("ids.jsonl", "ref.json", "out.json"))
            Path(ids).write_text("".join(json.dumps(x) + "\n" for x in docs))
            Path(rj).write_text(json.dumps(ref))
            with contextlib.redirect_stdout(io.StringIO()):
                sr.main(["replay-ids", ids, "--reference", rj, "--out", out,
                         "--boot", "20"])
            res = json.loads(Path(out).read_text())
        self.assertEqual(res["documents"], 2)
        r = res["results"]["ids"]
        self.assertEqual(set(r), {"P0/N8", "P3/d0.22/N8"})
        # a verbatim copy of the prompt: both policies emit more than one id per step
        self.assertGreater(r["P0/N8"]["tokens_per_step"][0], 1.5)
        self.assertGreater(r["P3/d0.22/N8"]["speedup_8GB_free"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
