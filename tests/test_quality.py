"""Scoring protocol tests only. This does not run the checkpoint quality harness."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from quality import metric, read_prompts, validate_result, windows


class QualityTests(unittest.TestCase):
    def test_every_target_once_even_at_partial_final_window(self):
        for size in (2, 3, 8, 9, 16, 17, 41):
            for context in (2, 4, 8, 16):
                for stride in range(1, context):
                    ids = list(range(size))
                    targets = []
                    for offset, piece, first in windows(ids, context, stride):
                        self.assertEqual(piece, ids[offset:offset + len(piece)])
                        self.assertGreaterEqual(first, 1)
                        self.assertLessEqual(len(piece), context)
                        targets.extend(piece[first:])
                    self.assertEqual(targets, ids[1:])

    def test_bad_windows_refused(self):
        for ids, context, stride in (([0], 8, 4), ([0, 1], 1, 1),
                                     ([0, 1], 8, 0), ([0, 1], 8, 8)):
            with self.assertRaises(ValueError):
                list(windows(ids, context, stride))

    def test_token_weighted_aggregate(self):
        # One difficult target and three easy ones: averaging document perplexity
        # would produce 5.5 instead of the correct 10**0.25.
        score = metric([math.log(10)] + [0] * 3)
        self.assertAlmostEqual(score["perplexity"], 10 ** 0.25)
        self.assertIsNone(metric([2000])["perplexity"])
        for bad in ([], [float("nan")], [float("inf")], [-1], [True]):
            with self.assertRaises(ValueError):
                metric(bad)

    def test_native_response_must_match_targets_and_complete_layers(self):
        response = {"schema": 1, "task": "perplexity", "input_ids": [3, 7, 5],
                    "score_start": 1, "scored_tokens": 2, "layers": 13,
                    "layers_completed": 13, "expert_drops": 0,
                    "token_nll": [1, 3], "nll_sum": 4, "mean_nll": 2}
        self.assertEqual(validate_result(response, [3, 7, 5], 1, 13), [1, 3])
        for changes in ({"input_ids": [3, 7, 6]}, {"score_start": 2},
                        {"scored_tokens": 1}, {"layers_completed": 12},
                        {"expert_drops": 1}, {"nll_sum": 5}, {"token_nll": [1]}):
            broken = copy.deepcopy(response)
            broken.update(changes)
            with self.assertRaises(ValueError):
                validate_result(broken, [3, 7, 5], 1, 13)

    def test_prompt_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            for record in ({"id": "a", "token_ids": [True]},
                           {"id": "a", "token_ids": [-1]},
                           {"id": "a", "text": "x", "token_ids": [1]},
                           {"id": "a", "text": ""}):
                path.write_text(json.dumps(record) + "\n")
                with self.assertRaises(ValueError):
                    read_prompts(path)
            path.write_text('{"id":"a","token_ids":[1,2]}\n' * 2)
            with self.assertRaises(ValueError):
                read_prompts(path)


if __name__ == "__main__":
    unittest.main()
