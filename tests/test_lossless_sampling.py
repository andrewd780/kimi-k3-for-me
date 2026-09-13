"""Plan validation needs no network or model weights."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_lossless import entropy, sample_plan, summarize


class SamplingTest(unittest.TestCase):
    def header(self):
        header, offset = {}, 0
        for expert in (0, 2, 10, 100):
            for matrix in ("w1", "w2", "w3"):
                for suffix, shape in (("packed", [32, 64]), ("scale", [32, 4])):
                    n = shape[0] * shape[1]
                    key = f"model.layers.1.block_sparse_moe.experts.{expert}.{matrix}.weight_{suffix}"
                    header[key] = {"dtype": "U8", "shape": shape,
                                   "data_offsets": [offset, offset + n]}
                    offset += n
        return header

    def test_complete_experts_and_pairs(self):
        plan = sample_plan(self.header(), 128, "sample", 3)
        packed = [r for r in plan if r["kind"] == "expert"]
        scales = [r for r in plan if r["kind"] == "scale"]
        self.assertEqual(len(packed), 9)
        self.assertEqual({int(r["tensor"].split("experts.")[1].split(".")[0]) for r in packed},
                         {0, 2, 100})
        self.assertEqual({r["paired_tensor"] for r in packed}, {r["tensor"] for r in scales})
        self.assertTrue(all(r["bytes"] == r["tensor_bytes"] == 128 for r in scales))
        self.assertEqual(sum(r["bytes"] for r in plan), 9 * (2048 + 128))

    def test_bad_scale_refused(self):
        header = self.header()
        name = next(n for n in header if n.endswith("weight_scale"))
        for mutate in (lambda h: h.pop(name),
                       lambda h: h[name].update(dtype="BF16"),
                       lambda h: h[name].update(shape=[32, 3])):
            broken = copy.deepcopy(header)
            mutate(broken)
            with self.assertRaises(ValueError):
                sample_plan(broken, 128, "sample", 4)

    def test_entropy_and_byte_weighting(self):
        self.assertEqual(entropy(bytes(32)), 0)
        self.assertEqual(entropy(bytes(range(256))), 8)
        rows, counts = [], {}
        for kind in ("expert", "scale", "dense"):
            counts[kind] = {0: 3, 1: 1}
            for size, bits, packed in ((3, 0, 2), (1, 1, 1)):
                rows.append({"kind": kind, "bytes": size, "byte_entropy_bits": bits,
                             "codecs": {"zstd3": {"bytes": packed}}})
        # summarize needs zero counts for absent symbols, as Counter supplies.
        from collections import Counter
        ratios, summary = summarize(rows, {k: Counter(v) for k, v in counts.items()})
        self.assertEqual(ratios["scale"]["zstd3"], 0.75)
        self.assertEqual(summary["scale"]["weighted_tensor_entropy_bits_per_byte"], 0.25)
        self.assertAlmostEqual(summary["scale"]["pooled_entropy_bits_per_byte"], 0.811278124459)


if __name__ == "__main__":
    unittest.main()
