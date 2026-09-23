"""Break-even arithmetic of the decode-under-contention gate, on hand-checkable rates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_decode_contention as gate


class BreakEvenTest(unittest.TestCase):
    def test_ssd_bound_decode_hidden(self):
        # B=3, r=.75, D=12, M_rest=6, M_all=8: raw 1/3 s/GB, compressed .25 s/GB.
        raw, fd, limit = gate.stage_times(0.75, 3, 12, 6, 8)
        self.assertAlmostEqual(raw, 1 / 3)
        self.assertAlmostEqual(fd, 0.25)
        self.assertEqual(limit, "ssd")
        row = gate.break_even(0.75, 12, 6, 8, ssd=(3,))["streamed"][0]
        self.assertAlmostEqual(row["speedup"], 4 / 3)
        self.assertTrue(row["decode_hidden"])
        self.assertAlmostEqual(row["decode_needed_for_full_gain_GBps"], 4)

    def test_matmul_on_fewer_cores_can_erase_the_gain(self):
        # B=6: raw max(1/6, 1/8) = 1/6; compressed max(.125, 1/12, 1/6) = 1/6.
        row = gate.break_even(0.75, 12, 6, 8, ssd=(6,))["streamed"][0]
        self.assertAlmostEqual(row["speedup"], 1.0)
        self.assertEqual(row["limiting_stage"], "matmul")

    def test_slow_decode_is_named_as_the_limit(self):
        row = gate.break_even(0.75, 2, 6, 8, ssd=(3,))["streamed"][0]
        self.assertEqual(row["limiting_stage"], "decode")
        self.assertFalse(row["decode_hidden"])
        self.assertAlmostEqual(row["speedup"], (1 / 3) / (1 / 2))
        # A tie between decode and another stage is reported as decode.
        self.assertEqual(gate.stage_times(0.5, 2, 4, 4, 8)[2], "decode")

    def test_resident_regime_and_cpu_cost(self):
        result = gate.break_even(0.75, 12, 6, 8)
        self.assertAlmostEqual(result["resident"]["slowdown"], 8 / 6)
        self.assertTrue(result["resident"]["decode_hidden"])
        self.assertAlmostEqual(result["resident"]["matmul_core_loss"], 8 / 6)
        self.assertAlmostEqual(result["core_seconds_per_token"], 108.81 / 12)
        self.assertAlmostEqual(result["compressed_GB_read_per_token"], 0.75 * 108.81)
        self.assertFalse(gate.break_even(0.75, 5, 6, 8)["resident"]["decode_hidden"])
        with self.assertRaises(ValueError):
            gate.stage_times(0.75, 3, 0, 6, 8)

    def test_summary_uses_contended_rates_for_the_pipeline(self):
        arms = {name: {"median": value, "min": value / 2} for name, value in (
            ("decode_alone", 16), ("decode_concurrent", 12), ("matmul_all_threads", 8),
            ("matmul_rest_threads", 7), ("matmul_concurrent", 6))}
        summary = gate.summarize({"arms": arms, "packed_bytes": 3, "raw_bytes": 4})
        median = summary["derived"]["median"]
        self.assertAlmostEqual(median["decode_slowdown_under_matmul"], 16 / 12)
        self.assertAlmostEqual(median["matmul_slowdown_under_decode"], 7 / 6)
        self.assertAlmostEqual(median["contended"]["resident"]["slowdown"], 8 / 6)
        self.assertAlmostEqual(median["core_seconds_per_token_contended"], 108.81 / 12)
        self.assertAlmostEqual(summary["derived"]["min"]["core_seconds_per_token_alone"],
                               108.81 / 8)

    def test_break_even_table_reads_the_contended_rates(self):
        # r = .75, contended D = 12, M_c = 6, M_all = 8, as in the first test; the
        # uncontended arms are deliberately different so a swap would show.
        arms = {name: {"median": value, "min": value} for name, value in (
            ("decode_alone", 1), ("decode_concurrent", 12), ("matmul_all_threads", 8),
            ("matmul_rest_threads", 100), ("matmul_concurrent", 6))}
        run = {"arms": arms, "packed_bytes": 3, "raw_bytes": 4, "index_bits": 4,
               "native": "ssse3_pshufb", "input": "stream"}
        run["summary"] = gate.summarize(run)
        header, rule, row = gate.break_even_markdown({"runs": [run]}, "median").splitlines()
        self.assertEqual(header.count("|"), rule.count("|"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[:4], ["FD4B ssse3_pshufb", "stream", "median", "0.7500"])
        # B = 2.5, 3, 4, 5, 6: raw max(1/B, 1/8); compressed max(.75/B, 1/12, 1/6).
        self.assertEqual(cells[4:9], ["1.333 (ssd)", "1.333 (ssd)", "1.333 (ssd)",
                                      "1.200 (matmul)", "1.000 (matmul)"])
        self.assertEqual(cells[9], "1.333")


if __name__ == "__main__":
    unittest.main()
