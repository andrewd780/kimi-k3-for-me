"""Adversarial trace conversion and held-out profile replay, without model weights."""
from collections import OrderedDict
import contextlib
import io
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import expert_profile as ep  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_fixture_is_twelve_positions_not_sixty_eight_new_tokens(self):
        pairs, _ = ep.read_trace(ROOT / "tests/fixtures/expert_trace.bin", 93, 896)
        ordered, info = ep.legacy_prefixes(pairs, 16)
        self.assertEqual(info["prefix_lengths"], list(range(5, 13)))
        self.assertEqual(len(ordered), 12 * 1472)
        self.assertEqual(len(set(ordered)), 10010)
        self.assertEqual(info["coverage"][4]["distinct_experts"], 5682)
        old_keys = [layer * 896 + expert for layer, expert in pairs]
        new_keys = [layer * 896 + expert for layer, expert in ordered]
        self.assertEqual(ep.replay(old_keys, 455)["resident_reuses"], 36272)
        self.assertEqual(ep.replay(new_keys, 455)["resident_reuses"], 0)
        pairs[0] = (pairs[0][0], (pairs[0][1] + 1) % 896)
        with self.assertRaisesRegex(ValueError, "not identical"):
            ep.legacy_prefixes(pairs, 16)

    def test_conversion_requires_real_repeated_prefixes(self):
        for pairs in ([(1, 0)], [(1, 0), (2, 0)],
                      [(1, 0), (2, 0), (1, 1), (2, 0)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                ep.legacy_prefixes(pairs, 1)

    def test_binary_validation_and_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.bin"
            for data in (b"", b"1234", struct.pack("<ii", -1, 0),
                         struct.pack("<ii", 1, 896)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    ep.read_trace(path, 93, 896)

    def test_lazy_pins_use_empty_capacity_and_pay_cold_misses(self):
        # Key 99 never arrives, so its pin must not waste an empty cache slot.
        self.assertEqual(ep.replay([1, 2, 3, 1, 2, 3], 3, [99])["loads"], 3)
        self.assertEqual(ep.replay([99, 99], 3, [99])["loads"], 1)

    def test_replay_matches_independent_full_resident_lru(self):
        rng = random.Random(73)
        for cap in (3, 8, 21):
            keys = [rng.randrange(25) for _ in range(500)]
            hot = {0, 1}
            resident = OrderedDict()
            hits = 0
            for key in keys:
                if key in resident:
                    hits += 1
                    resident.move_to_end(key)
                else:
                    if len(resident) == cap:
                        victim = next(k for k in resident if k not in hot)
                        del resident[victim]
                    resident[key] = None
            self.assertEqual(ep.replay(keys, cap, hot)["resident_reuses"], hits)

    def test_ranking_has_no_test_data_and_deterministic_ties(self):
        train = [5, 1, 5, 1, 2]
        ranked = ep.rank(train)
        self.assertEqual(ranked, [(1, 2), (5, 2), (2, 1)])
        self.assertEqual(ep.rank(ep.Counter(train)), ranked)
        self.assertNotIn(99, dict(ranked))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.txt"
            ep.write_profile(path, ranked, 1, 8, 2)
            self.assertEqual(path.read_text(), "K3EXPERTS 1 1 8 2\n0 1 2\n0 5 2\n0 2 1\n")

    def test_evaluation_reports_one_deterministic_result_per_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            trace, out = Path(temp) / "trace.bin", Path(temp) / "result.json"
            trace.write_bytes(b"".join(struct.pack("<ii", 0, key)
                                      for key in [0, 1, 0, 2, 0, 1, 0, 2]))
            args = ["evaluate", str(trace), "--n-layers", "1", "--n-experts", "4",
                    "--topk", "1", "--slots", "3", "--train-requests", "4",
                    "--pin-fractions", "0,1", "--out", str(out)]
            self.assertEqual(ep.main(args), 0)
            report = json.loads(out.read_text())
            self.assertEqual(len(report["arms"]), 2)
            for arm in report["arms"]:
                self.assertNotIn("runs", arm)
                self.assertEqual(arm["result"]["requests"], 4)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ep.main([*args, "--runs", "3"])


if __name__ == "__main__":
    unittest.main()
