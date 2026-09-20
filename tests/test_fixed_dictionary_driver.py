"""Independent layout and negative timing gate controls (run in CI)."""
import copy
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_fixed_dictionary import encode, rate_gate


class DriverTest(unittest.TestCase):
    def test_independent_golden_nibble_order_escape_and_odd_tail(self):
        raw = bytes.fromhex("e100e201e3ffe40ee5")
        expected = (b"FD4B" + struct.pack("<III", 9, 1, 0) + bytes(range(15)) +
                    bytes.fromhex("0010efe1e2e3e4e5ff"))
        self.assertEqual(encode(raw, list(range(15))), expected)

    def test_invalid_dictionary_rejected(self):
        for dictionary in (list(range(14)), [0]*15, list(range(14)) + [256]):
            with self.assertRaises(ValueError):
                encode(b"", dictionary)

    def test_every_run_and_every_range_is_required(self):
        arm = {"name": "ssse3_pshufb", "decoded_BF16_GBps_runs": [3, 4, 5]}
        good = [{"native": "ssse3_pshufb", "byte_exact": True,
                 "arms": [copy.deepcopy(arm), copy.deepcopy(arm)]} for _ in range(9)]
        self.assertTrue(rate_gate(good))
        for rate in (2.999999999, float("nan"), float("inf"), 0):
            broken = copy.deepcopy(good)
            broken[3]["arms"][1]["decoded_BF16_GBps_runs"][1] = rate
            self.assertFalse(rate_gate(broken))
        for mutation in (lambda c: c.pop(),
                         lambda c: c[0].update(byte_exact=False),
                         lambda c: c[0].update(native="scalar_fallback"),
                         lambda c: c[0]["arms"][1]["decoded_BF16_GBps_runs"].pop()):
            broken = copy.deepcopy(good)
            mutation(broken)
            self.assertFalse(rate_gate(broken))


if __name__ == "__main__":
    unittest.main()
