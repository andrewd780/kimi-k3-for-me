"""Hand-checkable controls for the per-family gate; real sampling runs in hosted CI only."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import trunk_family_gate as fam


def hist(counts):
    return [counts.get(i, 0) for i in range(256)]


def tensor(name, rows, cols, offset=0, shard="model-00001-of-000096.safetensors", dtype="BF16"):
    return {"name": name, "shard": shard, "dtype": dtype, "shape": [rows, cols],
            "offset": offset, "bytes": 2 * rows * cols}


class InventoryTest(unittest.TestCase):
    def test_config_inventory_matches_engine_shapes_and_documented_total(self):
        report = fam.inventory_report()
        by = {r["family"]: r for r in report["families"]}
        # 2-D matrices including the BF16 router: 54,379,315,200 parameters.
        self.assertEqual(report["matrix_bytes"], 108_758_630_400)
        self.assertLess(report["non_matrix_bytes_estimate"], 60_000_000)
        self.assertEqual(fam.mla_layers()[:3], [3, 7, 11])
        self.assertEqual(fam.mla_layers()[-2:], [91, 92])
        self.assertEqual(len(fam.mla_layers()), 24)
        self.assertEqual({w["cols"] for w in report["row_widths"]},
                         {128, 512, 1536, 3584, 6144, 7168, 12288, 33792})
        kda = sum(by[f"kda.{m}_proj"]["bytes"] for m in "qkvgo")
        self.assertEqual(kda, 5 * 69 * 12288 * 7168 * 2)
        shared = sum(by[f"moe.shared_{m}"]["bytes"] for m in ("gate", "up", "down"))
        self.assertEqual(shared, 3 * 92 * 6144 * 7168 * 2)
        self.assertEqual((by["kda.f_b_proj"]["rows"], by["kda.f_b_proj"]["cols"]), (12288, 128))
        self.assertEqual((by["mla.kv_b_proj"]["rows"], by["mla.kv_b_proj"]["cols"]), (24576, 512))
        self.assertEqual(by["dense.down_proj"]["matrices"], 1)

    def test_row_index_bytes_are_exact(self):
        # Per row: 16-byte header + one u32 per row; grouped: one u32 per ceil(8192/cols) rows.
        self.assertEqual(fam.row_index_bytes(12288, 7168, 1), 16 + 4 * 12288)
        self.assertEqual(fam.grouped(7168), 2)
        self.assertEqual(fam.row_index_bytes(12288, 7168, 2), 16 + 4 * 6144)
        self.assertEqual(fam.grouped(128), 64)
        self.assertEqual(fam.row_index_bytes(12288, 128, 64), 16 + 4 * 192)
        self.assertEqual(fam.grouped(12288), 1)
        self.assertEqual(fam.grouped(3584), 3)
        self.assertEqual(fam.row_index_bytes(7168, 3584, 3), 16 + 4 * 2390)
        self.assertEqual(fam.row_index_bytes(0, 7, 1), 16)
        for invalid in ((1, 0, 1), (1, 1, 0), (-1, 1, 1)):
            with self.assertRaises(ValueError):
                fam.row_index_bytes(*invalid)
        by = {r["family"]: r for r in fam.inventory_report()["families"]}
        self.assertAlmostEqual(by["kda.f_b_proj"]["row_index_per_row_points"],
                               100 * (16 + 4 * 12288) / (12288 * 128 * 2))

    def test_layer_kind_decides_o_and_g_projection_families(self):
        names = ["language_model.model.layers.3.self_attn.kv_b_proj.weight",
                 "language_model.model.layers.3.self_attn.o_proj.weight",
                 "language_model.model.layers.3.self_attn.g_proj.weight",
                 "language_model.model.layers.4.self_attn.o_proj.weight",
                 "language_model.model.layers.4.self_attn.g_proj.weight",
                 "language_model.model.layers.4.block_sparse_moe.experts.7.w1.weight_packed",
                 "language_model.model.layers.4.input_layernorm.weight",
                 "language_model.model.embed_tokens.weight"]
        families, total = fam.header_inventory([tensor(n, 4, 8) for n in names])
        self.assertEqual(sorted(families), ["elementwise", "kda.g_proj", "kda.o_proj",
                                            "mla.g_proj", "mla.kv_b_proj", "mla.o_proj"])
        self.assertEqual(total, 6 * 64)
        self.assertFalse(families["elementwise"]["sampled"])
        self.assertTrue(families["mla.o_proj"]["sampled"])
        wide = fam.header_inventory([tensor(names[3], 4, 8, dtype="F32")])[0]
        self.assertFalse(wide["kda.o_proj"]["sampled"])


class PlanTest(unittest.TestCase):
    def test_systematic_plan_is_byte_proportional_and_inside_tensors(self):
        mib = 1 << 20
        # 1, 2, 3 and 2 MiB tensors: cumulative [0,1) [1,3) [3,6) [6,8) MiB.
        tensors = [tensor(f"t{i}", rows, 1024, offset=1000 + i * 10 * mib)
                   for i, rows in enumerate((512, 1024, 1536, 1024))]
        plan = fam.plan_family(tensors, samples=4)
        # Centers at 1, 3, 5, 7 MiB; each 1 MiB range is centered there, then clamped.
        self.assertEqual([r["tensor"] for r in plan], ["t1", "t2", "t2", "t3"])
        base = {f"t{i}": 1000 + i * 10 * mib for i in range(4)}
        self.assertEqual([r["offset"] - base[r["tensor"]] for r in plan],
                         [0, 0, 3 * mib // 2, mib // 2])
        self.assertEqual([r["position"] for r in plan], [1 / 8, 3 / 8, 5 / 8, 7 / 8])
        self.assertTrue(all(r["bytes"] == mib and r["offset"] % 2 == 0 for r in plan))

    def test_single_tensor_and_tiny_tensor_families(self):
        mib = 1 << 20
        single = fam.plan_family([tensor("dense", 4096, 1024)], samples=4)
        starts = [r["offset"] for r in single]
        self.assertEqual(starts, [mib // 2, 5 * mib // 2, 9 * mib // 2, 13 * mib // 2])
        tiny = fam.plan_family([tensor("tiny", 3, 5)], samples=4)
        self.assertEqual(len(tiny), 1)
        self.assertEqual((tiny[0]["offset"], tiny[0]["bytes"]), (0, 30))
        for bad in ([], [tensor("odd", 1, 1) | {"bytes": 3}]):
            with self.assertRaises(ValueError):
                fam.plan_family(bad)
        with self.assertRaises(ValueError):
            fam.plan_family([tensor("x", 2, 2)], samples=0)

    def test_pins_must_cover_every_planned_range(self):
        plan = [{"shard": "s", "offset": 2, "bytes": 4, "tensor": "t"}]
        self.assertEqual(fam.verify_pins(plan, [{"shard": "s", "offset": 2, "bytes": 4,
                                                  "sha256": "ab"}]), {("s", 2, 4): "ab"})
        with self.assertRaises(ValueError):
            fam.verify_pins(plan, [{"shard": "s", "offset": 4, "bytes": 4, "sha256": "ab"}])


class WeightingTest(unittest.TestCase):
    # Family A: 15 values, no 4-bit escapes; family B: one value outside A's table.
    A = hist({**dict.fromkeys(range(7), 10), **dict.fromkeys(range(7, 15), 5)})   # n = 110
    B = hist({0: 60, 200: 40})                                                     # n = 100

    def families(self, a_bytes=3, b_bytes=1):
        return {"a": {"bytes": a_bytes, "histogram": self.A, "full": None},
                "b": {"bytes": b_bytes, "histogram": self.B, "full": None}}

    def test_one_dictionary_from_the_byte_weighted_mixture(self):
        mix, ranking, _ = fam.mixture(self.families())
        self.assertEqual(sum(mix), 1)
        # 0: 3/4*10/110 + 1/4*60/100 is the heaviest; 200 (1/4*40/100 = 1/10) is next.
        self.assertEqual(mix[0], Fraction(3, 4) * Fraction(10, 110) + Fraction(1, 4) * Fraction(60, 100))
        self.assertEqual(ranking[:2], [0, 200])
        report = fam.analyze_families(self.families())
        self.assertEqual(report["dictionary_ranking"][:15], ranking[:15])
        self.assertEqual(report["families"]["b"]["global_15"]["escape_values"], 0)
        # 200 displaces value 14 (the last of the 5-count ties) from the 15-entry table.
        self.assertEqual(report["families"]["a"]["global_15"]["escape_values"], 5)

    def test_pooled_ratios_are_exact_byte_weighted_means(self):
        report = fam.analyze_families(self.families())
        rows = {name: {s["scheme"]: s for s in r["bit_width_curve"]["schemes"]}
                for name, r in report["families"].items()}
        pooled = {s["scheme"]: s for s in report["byte_weighted_pooled"]["schemes"]}
        for scheme in ("fixed_3bit", "fixed_4bit", "fixed_5bit", "sign_split_4bit"):
            expected = (Fraction(3, 4) * Fraction(rows["a"][scheme]["payload_bits"], 16 * 110) +
                        Fraction(1, 4) * Fraction(rows["b"][scheme]["payload_bits"], 16 * 100))
            self.assertEqual(pooled[scheme]["payload_ratio"], float(expected))
        # fixed_4bit: a escapes 5 of 110, b none -> 3/4*(12*110+40)/1760 + 1/4*12/16.
        self.assertEqual(pooled["fixed_4bit"]["payload_ratio"],
                         float(Fraction(3, 4) * Fraction(1360, 1760) + Fraction(1, 4) * Fraction(3, 4)))
        coverage = report["byte_weighted_pooled"]["coverage_global_15"]
        self.assertAlmostEqual(coverage, 0.75 * 105 / 110 + 0.25)

    def test_gate_names_every_family_below_99_percent(self):
        report = fam.analyze_families(self.families(), gate1_dictionary=list(range(15)))
        self.assertEqual(report["gate"]["status"], "STOP")
        self.assertEqual(report["gate"]["failed_families"], ["a"])
        self.assertEqual(report["families"]["b"]["gate1_15"]["escape_values"], 40)
        good = {"a": {"bytes": 1, "histogram": self.A, "full": None}}
        self.assertEqual(fam.analyze_families(good)["gate"]["status"], "PASS")
        with self.assertRaises(ValueError):
            fam.analyze_families({})

    def test_full_value_bounds_pool_only_when_every_family_has_them(self):
        full_a = Counter({v << 8: c for v, c in enumerate(self.A) if c})
        full_b = Counter({v << 8 | 1: c for v, c in enumerate(self.B) if c})
        families = self.families()
        families["a"]["full"], families["b"]["full"] = full_a, full_b
        pooled = fam.analyze_families(families)["byte_weighted_pooled"]
        self.assertIn("full_bf16_entropy_ratio", pooled["per_family_code_bounds"])
        self.assertIn("full_bf16_mixture_entropy_ratio", pooled["single_code_bounds"])
        # Low bytes are constant per family, so whole-value entropy equals high-byte entropy.
        self.assertAlmostEqual(pooled["per_family_code_bounds"]["full_bf16_entropy_ratio"] * 16,
                               pooled["per_family_code_bounds"]["high_byte_entropy_ratio"] * 16 - 8)
        families["b"]["full"] = None
        pooled = fam.analyze_families(families)["byte_weighted_pooled"]
        self.assertNotIn("full_bf16_entropy_ratio", pooled["per_family_code_bounds"])


class FamilyPlanTest(unittest.TestCase):
    # "base" carries the bytes and the pooled table: seven values of 100, eight of 1.
    # "own" is fifteen values the pooled table lacks, so it fails the gate but its own
    # table covers all of it. "wide" spreads over thirty values, so no 15-entry table
    # reaches 99% of it.
    BASE = hist({**dict.fromkeys(range(7), 100), **dict.fromkeys(range(7, 15), 1)})
    OWN = hist(dict.fromkeys(range(100, 115), 10))
    WIDE = hist(dict.fromkeys(range(200, 230), 10))

    def test_failed_families_get_their_own_table_or_stay_raw(self):
        report = fam.analyze_families({
            "base": {"bytes": 100, "histogram": self.BASE, "full": None},
            "own": {"bytes": 1, "histogram": self.OWN, "full": None},
            "wide": {"bytes": 1, "histogram": self.WIDE, "full": None}})
        self.assertEqual(report["gate"]["failed_families"], ["own", "wide"])
        self.assertTrue(report["byte_weighted_pooled"]["decision"]["prototype"])
        plan = fam.family_plan(report)
        rows = plan["families"]
        # base: 708 values, 8 escapes at 3 bits: 708*11 + 64 bits beat 708*12.
        self.assertEqual(rows["base"], {"choice": "pooled_3bit",
                                        "payload_ratio": float(Fraction(7852, 16 * 708))})
        # own: its own seven of fifteen equal values escape 80 at 3 bits (2290 bits), so
        # its own 15-entry table at 4 bits (1800 bits, no escape) wins.
        self.assertEqual(rows["own"], {"choice": "own_4bit", "payload_ratio": 0.75})
        self.assertEqual(rows["wide"], {"choice": "raw", "payload_ratio": 1.0})
        expected = (100 * Fraction(7852, 16 * 708) + Fraction(3, 4) + 1) / 102
        self.assertEqual(plan["payload_ratio"], float(expected))

    def test_committed_family_report_recomputes_from_its_counts(self):
        # The per-family CI run's report, committed from its stdout line: every field must
        # follow from the histograms it carries, except the whole-BF16 entropy bounds,
        # whose value histograms are only in the run's artifact.
        root = Path(__file__).resolve().parents[1] / "docs/measurements"
        record = json.loads((root / "trunk-family-gate.json").read_bytes())
        report = record["report"]
        gate1 = json.loads((root / "trunk-dictionary-gate1.json").read_bytes())[
            "report"]["pooled"]["global_15"]["dictionary"]
        again = fam.analyze_families({name: {"bytes": f["bytes"], "histogram": f["histogram"],
                                             "full": None, "samples": f["samples"]}
                                      for name, f in report["families"].items()}, gate1)
        whole = ("full_bf16", "low_byte_entropy_bits", "low_given_high_entropy_bits",
                 "gap_to_full_bf16_entropy_points")

        def check(old, new, path):
            if isinstance(old, dict):
                for key, value in old.items():
                    if not key.startswith(whole):
                        self.assertIn(key, new, path + "/" + key)
                        check(value, new[key], path + "/" + key)
            elif isinstance(old, list):
                self.assertEqual(len(old), len(new), path)
                for i, (a, b) in enumerate(zip(old, new)):
                    check(a, b, f"{path}/{i}")
            elif isinstance(old, float):
                self.assertAlmostEqual(old, new, places=12, msg=path)
            else:
                self.assertEqual(old, new, path)

        for key in ("families", "byte_weighted_pooled", "dictionary_ranking", "gate",
                    "limits", "schema"):
            check(report[key], again[key], "/" + key)
        self.assertEqual(report["gate"]["failed_families"], ["moe.router", "moe.shared_down"])
        self.assertTrue(all(row["match"] for row in report["config_check"].values()))
        self.assertEqual(report["execution"]["run_id"], str(record["workflow_run"]))
        # The figure the note quotes: shared_down raw, the router on a table of its own.
        plan = fam.family_plan(report)
        self.assertEqual(plan["families"]["moe.shared_down"]["choice"], "raw")
        self.assertEqual(plan["families"]["moe.router"]["choice"], "own_3bit")
        self.assertEqual(round(plan["payload_ratio"], 6), 0.732863)
        pins = json.loads((root / "trunk-family-pins.json").read_bytes())["samples"]
        planned = [s for f in report["families"].values() for s in f["samples"]]
        self.assertEqual(len(fam.verify_pins(planned, pins)), 92)


if __name__ == "__main__":
    unittest.main()
