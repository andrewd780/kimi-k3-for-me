"""Check mathematical identities using independent finite enumerations.

These are proof/model checks, not repeated timing runs or hardware simulations.
"""
from fractions import Fraction as F
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import streaming_bounds as sb


class BoundsTests(unittest.TestCase):
    def test_pipeline_closed_form_against_precedence_schedule(self):
        for n, read, compute in itertools.product(range(1, 33), range(1, 8), range(1, 8)):
            formula = read + compute + (n - 1) * max(read, compute)
            self.assertEqual(sb.pipeline_schedule(n, read, compute), formula)
            self.assertEqual(sb.pipeline_schedule(n, read, compute, 1), n * (read + compute))
            self.assertGreaterEqual(formula, max(n * read, n * compute))
            self.assertLessEqual(formula, n * (read + compute))

    def test_uniform_union_against_all_small_routing_sequences(self):
        for count, topk, batch in ((3, 1, 3), (5, 2, 3), (4, 4, 3), (5, 3, 2)):
            choices = list(itertools.combinations(range(count), topk))
            total = sum(len(set().union(*sets))
                        for sets in itertools.product(choices, repeat=batch))
            self.assertEqual(sb.uniform_union(count, topk, batch), F(total, len(choices)**batch))

    def test_prefetch_accounting_against_set_difference(self):
        subsets = [set(s) for n in range(5) for s in itertools.combinations(range(4), n)]
        for wanted, guessed in itertools.product(subsets[1:], subsets):
            actual = len(guessed) + len(wanted - guessed)
            self.assertEqual(sb.prefetch_reads(len(wanted), len(guessed),
                                             len(wanted & guessed)), actual)

    def test_speculation_against_all_accept_reject_paths(self):
        for p in (F(0), F(1, 8), F(1, 2), F(9, 10), F(1)):
            for length in range(6):
                mean = F(0)
                for path in itertools.product((False, True), repeat=length):
                    probability = F(1)
                    for accepted in path:
                        probability *= p if accepted else (1-p)
                    leading = next((i for i, accepted in enumerate(path) if not accepted), length)
                    mean += probability * (leading + 1)
                self.assertEqual(sb.accepted_tokens(p, length), mean)

    def test_codec_break_even_and_byte_conservation(self):
        for ratio, bandwidth in itertools.product((F(0), F(1, 8), F(9, 10)), (F(1), F(3), F(6))):
            D = bandwidth / (1-ratio)
            for scale in (F(1), F(1032192)):
                self.assertEqual(scale / bandwidth, ratio * scale / bandwidth + scale / D)
                self.assertGreater(ratio * scale / bandwidth + scale / (D/2), scale / bandwidth)
        report = sb.build_report()
        b = report["bytes"]
        self.assertEqual(b["expert_per_token"], 1472 * 17547264)
        self.assertEqual(b["scale_per_token"] * 17, b["expert_per_token"])
        self.assertLess(b["total_fraction_saved"], F(1, 100))
        self.assertGreater(b["checkpoint_GB_after_scales_projection"], 1000)
        self.assertEqual(report["batching_uniform_independent_model"][0]["ideal_IO_throughput_ratio"], 1)


if __name__ == "__main__":
    unittest.main()
