"""Calibration checks on artificial routes, not a K3 recall measurement."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import routing_probe as rp  # noqa: E402


class ProbeTests(unittest.TestCase):
    def fixture(self):
        return {"x": np.tile(np.eye(4), (3, 1)),
                "routes": np.tile(np.arange(4), 3)[:, None],
                "prompt": np.repeat(["calibration", "held-a", "held-b"], 4),
                "position": np.tile(np.arange(4), 3), "layer": np.full(12, 4),
                "source_layer": np.full(12, 3), "source_position": np.tile(np.arange(4), 3),
                "phase": np.full(12, "decode", dtype="U16"),
                "feature_site": np.array("router_input"), "evidence_kind": np.array("synthetic"),
                "n_experts": np.array(4)}

    def test_separable_routes_generalize_to_other_prompts(self):
        report = rp.evaluate(self.fixture(), {"calibration"}, lead=1)
        self.assertEqual(report["heldout_prompts"], ["held-a", "held-b"])
        self.assertEqual(report["evidence_kind"], "synthetic")
        self.assertEqual(report["gate_status"], "not_evaluated")
        self.assertIn("equal-slot global static-pin null", report["unmeasured"])
        for row in report["results"]:
            self.assertEqual(row["topk_recall"], 1.0)
            self.assertNotIn("expert_read_multiplier", row)
            self.assertNotIn("clears_70_percent_recall_gate", row)

    def test_replayed_prefills_and_leaking_split_refused(self):
        data = self.fixture()
        data["position"][1] = 0
        data["source_position"][1] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rp.evaluate(data, {"calibration"}, lead=1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            rp.evaluate(self.fixture(), {"calibration", "held-a", "held-b"}, lead=1)

    def test_shuffled_heldout_routes_have_no_artificial_agreement(self):
        data = self.fixture()
        data["routes"][4:] = (data["routes"][4:] + 1) % 4
        for row in rp.evaluate(data, {"calibration"}, lead=1)["results"]:
            self.assertEqual(row["topk_recall"], 0.0)

    def test_unique_prefill_rows_are_still_ineligible(self):
        data = self.fixture()
        data["phase"][0] = "prefill"
        with self.assertRaisesRegex(ValueError, "prefill/replay"):
            rp.evaluate(data, {"calibration"}, lead=1)

    def test_oracle_cannot_be_labelled_as_early_features(self):
        data = self.fixture()
        data["source_layer"][:] = 4
        with self.assertRaisesRegex(ValueError, "declared lead"):
            rp.evaluate(data, {"calibration"}, lead=1)
        report = rp.evaluate(data, {"calibration"}, lead=0)
        self.assertTrue(report["oracle_diagnostic"])
        self.assertEqual(report["gate_status"], "not_evaluated")
        self.assertTrue(all(row["oracle_diagnostic"] for row in report["results"]))

    def test_each_requested_lead_is_recorded(self):
        for lead in (1, 2, 4):
            with self.subTest(lead=lead):
                data = self.fixture()
                data["source_layer"][:] = 4 - lead
                report = rp.evaluate(data, {"calibration"}, lead=lead)
                self.assertEqual(report["lead_layers"], lead)
                for row in report["results"]:
                    self.assertEqual(row["source_layer"], 4 - lead)
                    self.assertEqual(row["lead_layers"], lead)
                    self.assertFalse(row["oracle_diagnostic"])

    def test_future_or_other_position_features_are_refused(self):
        for field, value, error in (("source_layer", 5, "declared lead"),
                                    ("source_layer", -1, "declared lead"),
                                    ("source_position", 1, "same generated position")):
            with self.subTest(field=field, value=value):
                data = self.fixture()
                data[field][0] = value
                with self.assertRaisesRegex(ValueError, error):
                    rp.evaluate(data, {"calibration"}, lead=1)

    def test_legacy_ambiguous_data_and_wrong_feature_site_are_refused(self):
        data = self.fixture()
        del data["source_layer"]
        with self.assertRaisesRegex(ValueError, "missing capture/lead metadata"):
            rp.evaluate(data, {"calibration"}, lead=1)
        data = self.fixture()
        data["feature_site"] = np.array("layer_output")
        with self.assertRaisesRegex(ValueError, "feature_site"):
            rp.evaluate(data, {"calibration"}, lead=1)

    def test_capture_metadata_does_not_authenticate_real_data(self):
        data = self.fixture()
        data["evidence_kind"] = np.array("generation_capture")
        report = rp.evaluate(data, {"calibration"}, lead=1)
        self.assertEqual(report["gate_status"], "not_evaluated")
        self.assertIn("bytes per decode token", report["unmeasured"])


if __name__ == "__main__":
    unittest.main()
