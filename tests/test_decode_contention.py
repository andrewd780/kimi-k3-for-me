"""Break-even arithmetic of the decode-under-contention gate, on hand-checkable rates."""
from pathlib import Path
import random
import statistics
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_decode_contention as gate


def arm(runs):
    """An arm as the benchmark reports it: every run, its median and its minimum."""
    return {"GBps_runs": list(runs), "median": statistics.median(runs), "min": min(runs)}


def arms_from(runs):
    return {name: arm(values) for name, values in runs.items()}


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
        # Medians 16, 12, 8, 7, 6; every arm's slowest run is half, its fastest 1.5x.
        arms = arms_from({name: (value / 2, value, 1.5 * value) for name, value in (
            ("decode_alone", 16), ("decode_concurrent", 12), ("matmul_all_threads", 8),
            ("matmul_rest_threads", 7), ("matmul_concurrent", 6))})
        summary = gate.summarize({"arms": arms, "packed_bytes": 3, "raw_bytes": 4})
        median = summary["derived"]["median"]
        self.assertAlmostEqual(median["decode_slowdown_under_matmul"], 16 / 12)
        self.assertAlmostEqual(median["matmul_slowdown_under_decode"], 7 / 6)
        self.assertAlmostEqual(median["contended"]["resident"]["slowdown"], 8 / 6)
        self.assertAlmostEqual(median["core_seconds_per_token_contended"], 108.81 / 12)
        # Worst case: the baseline and the uncontended arms at their fastest where that
        # hurts, every decode and matmul feeding the compressed pipeline at its slowest.
        worst = summary["derived"]["worst"]
        self.assertAlmostEqual(worst["core_seconds_per_token_alone"], 108.81 / 8)
        self.assertAlmostEqual(worst["core_seconds_per_token_contended"], 108.81 / 6)
        self.assertAlmostEqual(worst["decode_slowdown_under_matmul"], 24 / 6)
        self.assertAlmostEqual(worst["matmul_slowdown_under_decode"], 10.5 / 3)
        self.assertAlmostEqual(worst["contended"]["resident"]["slowdown"], 12 / 3)
        self.assertAlmostEqual(worst["alone"]["resident"]["slowdown"], 12 * max(1 / 8, 1 / 3.5))

    def test_a_disturbed_baseline_run_cannot_pass_the_gate(self):
        # r = .75, D_c = 12, M_c = 5 in every repeat; M_all has median 12 and one
        # disturbed repeat at 4. At B = 6 the median fails: raw max(1/6, 1/12) = 1/6,
        # compressed max(.125, 1/12, 1/5) = .2, speedup .833. A per-arm minimum would
        # take M_all = 4, raw .25 and speedup 1.25, and pass; the worst case must not.
        arms = arms_from({"decode_alone": [14] * 9, "decode_concurrent": [12] * 9,
                          "matmul_all_threads": [11, 12, 13, 12, 12.5, 11.5, 12, 4, 12.2],
                          "matmul_rest_threads": [9] * 9, "matmul_concurrent": [5] * 9})
        summary = gate.summarize({"arms": arms, "packed_bytes": 3, "raw_bytes": 4})
        median = summary["derived"]["median"]["contended"]
        worst = summary["derived"]["worst"]["contended"]
        self.assertAlmostEqual(median["streamed"][-1]["speedup"], (1 / 6) / 0.2)
        self.assertAlmostEqual(worst["streamed"][-1]["speedup"], (1 / 6) / 0.2)
        self.assertLess(worst["streamed"][-1]["speedup"], 1)
        self.assertAlmostEqual(median["resident"]["slowdown"], 12 / 5)
        self.assertAlmostEqual(worst["resident"]["slowdown"], 13 / 5)
        for m, w in zip(median["streamed"], worst["streamed"]):
            self.assertLessEqual(w["speedup"], m["speedup"])

    def test_worst_case_bounds_every_repeat(self):
        # The worst speedup is at or below each repeat's own speedup (its five arms
        # paired together, the diagonal pairing) and the worst resident slowdown at or
        # above. That the bound holds for every other pairing of runs follows from the
        # monotonicity of break_even in each rate; it is not exercised here.
        names = ("decode_alone", "decode_concurrent", "matmul_all_threads",
                 "matmul_rest_threads", "matmul_concurrent")
        rng = random.Random(20260923)
        for _ in range(200):
            repeats = rng.randint(1, 9)
            runs = {name: [rng.uniform(1, 30) for _ in range(repeats)] for name in names}
            r = rng.uniform(0.6, 0.8)
            worst = gate.summarize({"arms": arms_from(runs), "packed_bytes": r,
                                    "raw_bytes": 1})["derived"]["worst"]["contended"]
            for i in range(repeats):
                paired = gate.break_even(r, runs["decode_concurrent"][i],
                                         runs["matmul_concurrent"][i],
                                         runs["matmul_all_threads"][i])
                for w, p in zip(worst["streamed"], paired["streamed"]):
                    self.assertLessEqual(w["speedup"], p["speedup"] * (1 + 1e-12))
                self.assertGreaterEqual(worst["resident"]["slowdown"] * (1 + 1e-12),
                                        paired["resident"]["slowdown"])

    def test_break_even_table_reads_the_contended_rates(self):
        # r = .75, contended D = 12, M_c = 6, M_all = 8, as in the first test; the
        # uncontended arms are deliberately different so a swap would show.
        arms = arms_from({name: (value,) for name, value in (
            ("decode_alone", 1), ("decode_concurrent", 12), ("matmul_all_threads", 8),
            ("matmul_rest_threads", 100), ("matmul_concurrent", 6))})
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
        # With one run per arm the worst case is the same table under its own label.
        worst = gate.break_even_markdown({"runs": [run]}, "worst").splitlines()[2]
        self.assertEqual(worst, row.replace("| median |", "| worst |"))
        rates = gate.markdown({"runs": [run]}).splitlines()
        self.assertEqual([line.split("|")[3].strip() for line in rates[2:]],
                         ["median", "min", "max"])


if __name__ == "__main__":
    unittest.main()
