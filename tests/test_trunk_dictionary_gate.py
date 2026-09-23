"""Adversarial checks for the falsifier; execution is in hosted CI only."""
import copy
from fractions import Fraction
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


def by_scheme(curve):
    return {s["scheme"]: s for s in curve["schemes"]}


class BitWidthCurveTest(unittest.TestCase):
    # 7 values x 100, 8 values x 10, 16 values x 1: n = 796, 31 distinct values.
    TIERED = hist({**dict.fromkeys(range(7), 100), **dict.fromkeys(range(7, 15), 10),
                   **dict.fromkeys(range(15, 31), 1)})
    # Dyadic and sign-symmetric: magnitudes 60..56 split evenly, 55 and 54 one sign each.
    SIGNED = hist({0x3C: 32, 0xBC: 32, 0x3D: 16, 0xBD: 16, 0x3B: 8, 0xBB: 8, 0x3A: 4,
                   0xBA: 4, 0x39: 2, 0xB9: 2, 0x38: 1, 0xB8: 1, 0x37: 1, 0xB6: 1})

    def test_fixed_width_ratios_are_exact_integer_bit_counts(self):
        curve = gate.bit_width_curve(self.TIERED)
        rows, n = by_scheme(curve), 796
        # k bits + 8 low bits per value, 8 more per escape; ratio = bits / (16 n).
        for k, escapes in ((3, 96), (4, 16), (5, 0)):
            row = rows[f"fixed_{k}bit"]
            self.assertEqual(row["dictionary_capacity"], 2**k - 1)
            self.assertEqual(row["escape_values"], escapes)
            self.assertEqual(row["payload_bits"], n * (k + 8) + 8 * escapes)
            # The float is the correctly rounded exact fraction; the bits are exact.
            self.assertEqual(row["payload_ratio"],
                             float(Fraction(n * (k + 8) + 8 * escapes, 16 * n)))
        self.assertEqual(rows["fixed_3bit"]["payload_bits"], 9524)
        self.assertEqual(rows["fixed_4bit"]["payload_bits"], 9680)
        self.assertEqual(rows["fixed_5bit"]["payload_ratio"], 0.8125)
        self.assertAlmostEqual(rows["fixed_3bit"]["gain_vs_fixed_4bit_points"],
                               100 * 156 / 12736)
        self.assertEqual(rows["fixed_3bit"]["dictionary"], list(range(7)))

    def test_sign_split_folds_sign_and_ranks_magnitudes(self):
        rows, n = by_scheme(gate.bit_width_curve(self.SIGNED)), 128
        self.assertEqual(rows["sign_split_3bit"]["dictionary"], [60, 61, 59])
        # Ties on one count break to the smaller magnitude: 54 before 55.
        self.assertEqual(rows["sign_split_4bit"]["dictionary"], [60, 61, 59, 58, 57, 56, 54])
        for k, escapes in ((3, 16), (4, 1), (5, 0)):
            self.assertEqual(rows[f"sign_split_{k}bit"]["escape_values"], escapes)
            self.assertEqual(rows[f"sign_split_{k}bit"]["payload_bits"],
                             n * (k + 8) + 8 * escapes)
        # The signed table ranks ties by byte value: 0x3C before 0xBC.
        self.assertEqual(rows["fixed_3bit"]["dictionary"], [60, 188, 61, 189, 59, 187, 58])
        self.assertEqual(rows["fixed_3bit"]["escape_values"], 12)
        self.assertEqual(rows["fixed_3bit"]["payload_ratio"], 1504 / 2048)
        self.assertEqual(rows["fixed_4bit"]["escape_values"], 0)

    def test_entropy_and_huffman_bounds_are_hand_checkable(self):
        bounds = gate.bit_width_curve(self.SIGNED)["bounds"]
        # Probabilities 1/4,1/4,1/8,1/8,...,1/128 x4: dyadic, so Huffman meets entropy.
        self.assertAlmostEqual(bounds["high_byte_entropy_bits"], 2.96875, places=12)
        self.assertEqual(bounds["huffman_high_byte_bits"], 380)
        self.assertEqual(bounds["huffman_high_byte_ratio"], (8 * 128 + 380) / 2048)
        self.assertAlmostEqual(bounds["high_byte_entropy_ratio"], (8 + 2.96875) / 16)
        # Non-dyadic: Huffman lengths 1,2,2 -> 1.5 bits against 1.584963 entropy.
        self.assertEqual(gate.huffman_bits([1, 1, 1]), 5)
        self.assertAlmostEqual(gate.entropy_bits([1, 1, 1]), 1.584962500721156)
        self.assertEqual(gate.huffman_bits([0, 7, 0]), 7)
        for invalid in ([], [0, 0], [-1, 2], [1.0, 1]):
            with self.assertRaises(ValueError):
                gate.huffman_bits(invalid)
            with self.assertRaises(ValueError):
                gate.entropy_bits(invalid)

    def test_full_bf16_bound_uses_whole_values_and_must_match_high_plane(self):
        # Two high bytes x four uniform low bytes: H = 3, H(high) = 1, H(low|high) = 2.
        full = {high << 8 | low: 5 for high in (0x3C, 0xBC) for low in (0, 1, 2, 3)}
        high = hist({0x3C: 20, 0xBC: 20})
        bounds = gate.bit_width_curve(high, full_histogram=full)["bounds"]
        self.assertAlmostEqual(bounds["full_bf16_entropy_bits"], 3.0, places=12)
        self.assertAlmostEqual(bounds["full_bf16_entropy_ratio"], 3 / 16)
        self.assertAlmostEqual(bounds["low_byte_entropy_bits"], 2.0, places=12)
        self.assertAlmostEqual(bounds["low_given_high_entropy_bits"], 2.0, places=12)
        self.assertEqual(bounds["full_bf16_distinct_values"], 8)
        raw = bytes([0x02, 0x3C, 0x02, 0x3C, 0x01, 0xBC])
        self.assertEqual(gate.bf16_histogram(raw), {0x3C02: 2, 0xBC01: 1})
        for mismatch in ({**full, 0x3C00: 6}, {0x13C00: 40}):
            with self.assertRaises(ValueError):
                gate.bit_width_curve(high, full_histogram=mismatch)

    def test_prototype_threshold_is_exact_at_one_and_a_half_points(self):
        def curve(escaped):
            # n = 1000, 15 distinct values, so fixed_4bit never escapes.
            counts = dict.fromkeys(range(7, 15), 0)
            for i in range(escaped):
                counts[7 + i % 8] += 1
            seven = 1000 - escaped
            counts.update({v: seven // 7 + (v < seven % 7) for v in range(7)})
            return gate.bit_width_curve(hist(counts))
        at, below = curve(95), curve(96)
        self.assertEqual(at["decision"]["best_scheme"], "fixed_3bit")
        self.assertAlmostEqual(at["decision"]["best_gain_vs_fixed_4bit_points"], 1.5)
        self.assertTrue(at["decision"]["prototype"])
        self.assertFalse(below["decision"]["prototype"])

    def test_every_range_is_scored_with_the_pooled_dictionaries(self):
        ranges = [hist({0: 10, 1: 5, 2: 1})] * 7 + [hist({2: 100, 3: 1})]
        fulls = [{v << 8: c for v, c in enumerate(h) if c} for h in ranges]
        report = gate.analyze(entries(ranges), fulls)
        pooled = report["pooled"]["bit_width_curve"]
        self.assertEqual(by_scheme(pooled)["fixed_3bit"]["dictionary"], [2, 0, 1, 3])
        last = by_scheme(report["ranges"][7]["bit_width_curve"])["fixed_3bit"]
        self.assertEqual(last["dictionary"], [2, 0, 1, 3])
        self.assertEqual(pooled["bf16_values"], 7 * 16 + 101)
        self.assertAlmostEqual(pooled["bounds"]["full_bf16_entropy_bits"],
                               pooled["bounds"]["high_byte_entropy_bits"], places=12)
        with self.assertRaises(ValueError):
            gate.analyze(entries(ranges), fulls[:7])

    def test_committed_four_range_curve_recomputes_from_its_source(self):
        record = json.loads((gate.ROOT / "docs/measurements/fixed-width-curve-four-ranges.json")
                            .read_bytes())
        again = gate.committed_curve(gate.ROOT / record["source"]["file"],
                                     record["source"]["pointer"])
        self.assertEqual(again["source"], record["source"])
        self.assertEqual(again["histogram"], record["histogram"])
        for old, new in zip(record["bit_width_curve"]["schemes"],
                            again["bit_width_curve"]["schemes"]):
            for key, value in old.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(new[key], value, places=12)
                else:
                    self.assertEqual(new[key], value)
        self.assertEqual(record["bit_width_curve"]["decision"]["best_scheme"], "fixed_3bit")

    def test_committed_histogram_agrees_with_independent_fd4b_escape_counts(self):
        # Two committed records from different CI runs describe the same four ranges:
        # the Huffman job's pooled high-byte counts and the FD4B rate job's escapes.
        source = json.loads((gate.ROOT / "docs/measurements/research-two-symbol.json")
                            .read_bytes())["experiments"][1]["report"]
        rate = json.loads((gate.ROOT / "docs/measurements/fixed-dictionary-rate-x86_64.json")
                          .read_bytes())
        tensors = [s["tensor"] for s in source["samples"]]
        cases = [c for c in rate["cases"] if c["sample"].get("tensor") in tensors]
        self.assertEqual(len(cases), 4)
        escaped = sum(source["high_byte_histogram"]) - sum(
            source["high_byte_histogram"][v] for v in rate["dictionary"])
        self.assertEqual(escaped, sum(c["escape_values"] for c in cases))
        self.assertEqual(escaped, 1077)

    def test_pointer_resolution_and_bad_widths(self):
        node, ancestors = gate.resolve_pointer({"a/b": [{"x~": [7]}]}, "/a~1b/0/x~0/0")
        self.assertEqual(node, 7)
        self.assertEqual(len(ancestors), 2)
        with self.assertRaises(ValueError):
            gate.resolve_pointer({}, "a")
        for widths in ((3, 5), (1, 4), (4, 9)):
            with self.assertRaises(ValueError):
                gate.bit_width_curve(self.TIERED, widths=widths)


if __name__ == "__main__":
    unittest.main()
