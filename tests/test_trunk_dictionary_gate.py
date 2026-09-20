"""Adversarial checks for the falsifier; execution is in hosted CI only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import trunk_dictionary_gate as gate


def hist(counts):
    return [counts.get(i, 0) for i in range(256)]


def entries(histograms):
    return [({"tensor": f"layer_{i}"}, h) for i, h in enumerate(histograms)]


class DictionaryGateTest(unittest.TestCase):
    def test_plane_preserves_sign_and_ignores_low_byte(self):
        raw = bytes([0, 0x3F, 0xFF, 0xBF, 0x80, 0x3F])
        self.assertEqual(gate.high_histogram(raw), hist({0x3F: 2, 0xBF: 1}))
        for invalid in (b"", raw[:-1]):
            with self.assertRaises(ValueError):
                gate.high_histogram(invalid)

    def test_coverage_support_uses_integer_ceiling(self):
        row = gate.describe(hist({0: 9900, 1: 90, 2: 9, 3: 1}), [0])
        self.assertEqual(row["values_for_coverage_percent"],
                         {"90": 1, "99": 1, "99.9": 2, "99.99": 3})
        self.assertTrue(row["global_15"]["passes_99_percent"])
        self.assertAlmostEqual(row["global_15"]["payload_ratio"], 0.755)
        self.assertFalse(gate.dictionary_stats(hist({0: 9899, 1: 101}), [0])[
            "passes_99_percent"])

    def test_uniform_support_kills_scheme(self):
        report = gate.analyze(entries([hist(dict.fromkeys(range(256), 1))] * 8))
        self.assertEqual(report["gate"]["status"], "STOP")
        self.assertFalse(report["gate"]["continue_to_codec"])
        self.assertEqual(report["pooled"]["values_for_coverage_percent"],
                         {"90": 231, "99": 254, "99.9": 256, "99.99": 256})

    def test_local_dictionaries_cannot_substitute_for_global(self):
        report = gate.analyze(entries([hist(dict.fromkeys(range(i*15, (i+1)*15), 1))
                                       for i in range(8)]))
        self.assertTrue(all(r["best_local_15"]["passes_99_percent"] for r in report["ranges"]))
        self.assertEqual(report["pooled"]["global_15"]["dictionary"], list(range(15)))
        self.assertEqual(report["gate"]["status"], "STOP")

    def test_pooled_is_count_weighted_and_cannot_hide_bad_range(self):
        good, bad = hist({0: 10000}), hist(dict.fromkeys(range(256), 1))
        report = gate.analyze(entries([good] * 7 + [bad]))
        self.assertEqual(report["pooled"]["bf16_values"], 70256)
        self.assertEqual(report["pooled"]["histogram"][0], 70001)
        self.assertTrue(report["pooled"]["global_15"]["passes_99_percent"])
        self.assertEqual(report["gate"]["status"], "STOP")
        self.assertEqual(report["gate"]["failed_global_dictionary_ranges"], ["layer_7"])

    def test_positive_control_can_pass(self):
        report = gate.analyze(entries([hist(dict.fromkeys(range(15), 100))] * 8))
        self.assertEqual(report["gate"]["status"], "PASS")
        self.assertEqual(report["pooled"]["global_15"]["p_escape"], 0)
        self.assertEqual(report["pooled"]["global_15"]["payload_ratio"], 0.75)
        self.assertEqual(report["pooled"]["global_15"]["saturated_stream_decode_GBps_at_B_3"], 4)

    def test_all_eight_and_valid_counts_required(self):
        for invalid in ([], [0] * 256, [-1] + [1] * 255, [1.0] * 256):
            with self.assertRaises(ValueError):
                gate.rank(invalid)
        with self.assertRaises(ValueError):
            gate.analyze(entries([hist({0: 1})] * 7))

    def test_manifest_missing_duplicate_alignment_and_bounds_rejected(self):
        manifest = json.loads(gate.MANIFEST.read_bytes())
        rows = gate.dense_ranges(manifest)
        self.assertEqual(len(rows), 8)
        for mutation in (lambda m: m.update(revision="main"),
                         lambda m: m.update(samples=m["samples"][:-1]),
                         lambda m: m["samples"].append(copy.deepcopy(rows[0])),
                         lambda m: next(r for r in m["samples"] if r["kind"] == "dense").update(
                             offset=1),
                         lambda m: next(r for r in m["samples"] if r["kind"] == "dense").update(
                             bytes=1 << 30)):
            broken = copy.deepcopy(manifest)
            mutation(broken)
            with self.assertRaises(ValueError):
                gate.dense_ranges(broken)

    def test_corrupt_or_truncated_range_cannot_be_scored(self):
        raw = b"\x80\xbf\x00\x3f"
        row = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.assertEqual(gate.verified_range(row, raw), raw)
        for broken in (raw[:-1], raw + b"\x00", b"\x81" + raw[1:]):
            with self.assertRaises(ValueError):
                gate.verified_range(row, broken)


if __name__ == "__main__":
    unittest.main()
