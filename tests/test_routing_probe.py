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
                "position": np.tile(np.arange(4), 3), "layer": np.ones(12, dtype=int),
                "n_experts": np.array(4)}

    def test_separable_routes_generalize_to_other_prompts(self):
        report = rp.evaluate(self.fixture(), {"calibration"})
        self.assertEqual(report["heldout_prompts"], ["held-a", "held-b"])
        for row in report["results"]:
            self.assertEqual(row["topk_recall"], 1.0)

    def test_replayed_prefills_and_leaking_split_refused(self):
        data = self.fixture()
        data["position"][1] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rp.evaluate(data, {"calibration"})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            rp.evaluate(self.fixture(), {"calibration", "held-a", "held-b"})

    def test_shuffled_heldout_routes_have_no_artificial_agreement(self):
        data = self.fixture()
        data["routes"][4:] = (data["routes"][4:] + 1) % 4
        for row in rp.evaluate(data, {"calibration"})["results"]:
            self.assertEqual(row["topk_recall"], 0.0)
            self.assertEqual(row["expert_read_multiplier"], 2.0)


if __name__ == "__main__":
    unittest.main()
