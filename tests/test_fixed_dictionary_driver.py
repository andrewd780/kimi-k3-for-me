"""Independent layout and negative timing gate controls (run in CI)."""
import copy
from pathlib import Path
import random
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench_fixed_dictionary import FORMATS, GOLDEN, encode, encode3, rate_gate


def reference_decode3(packed):
    """Bit-by-bit FD3B reader, written independently of the grouped encoder."""
    raw_bytes, escapes, reserved = struct.unpack("<III", packed[4:16])
    n, table = raw_bytes // 2, packed[16:23]
    index_bytes, low_bytes = (3 * n + 7) // 8, (raw_bytes + 1) // 2
    index = packed[32:32 + index_bytes]
    low = packed[32 + index_bytes:32 + index_bytes + low_bytes]
    escape = packed[32 + index_bytes + low_bytes:]
    if (packed[:4] != b"FD3B" or reserved or packed[23:32] != bytes(9)
            or len(escape) != escapes + 32 or escape[escapes:] != bytes(32)):
        raise ValueError("bad FD3B framing")
    out, e = bytearray(), 0
    for i in range(n):
        code = sum(((index[(3 * i + b) // 8] >> ((3 * i + b) % 8)) & 1) << b for b in range(3))
        out += bytes([low[i], escape[e] if code == 7 else table[code]])
        e += code == 7
    if raw_bytes % 2:
        out.append(low[n])
    if e != escapes or ((3 * n) % 8 and index[-1] >> ((3 * n) % 8)):
        raise ValueError("escape count or padding mismatch")
    return bytes(out)


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
        for dictionary in (list(range(6)), [0] * 7, list(range(6)) + [256], list(range(15))):
            with self.assertRaises(ValueError):
                encode3(b"", dictionary)

    def test_fd3b_independent_golden_bit_order_escape_tail_and_slack(self):
        # Codes 1, 6, escape are bits 001 110 111, packed from bit 0: 0xF1 then 0x01.
        expected = (b"FD3B" + struct.pack("<III", 7, 1, 0) + bytes([3, 20, 37, 54, 71, 88, 105])
                    + bytes(9) + bytes.fromhex("f101") + bytes.fromhex("e1e2e3e4") + b"\xaa"
                    + bytes(32))
        self.assertEqual(encode3(bytes.fromhex("e114e269e3aae4"), [3, 20, 37, 54, 71, 88, 105]),
                         expected)
        self.assertEqual(GOLDEN[3][2], expected)
        self.assertEqual({bits: entries for bits, (_, entries) in FORMATS.items()}, {3: 7, 4: 15})

    def test_fd3b_round_trips_through_a_bitwise_reference(self):
        rng = random.Random(3)
        dictionary = [188, 60, 61, 189, 59, 187, 58]
        for n in [*range(0, 70), 1001, 4096]:
            raw = bytes(rng.choice(dictionary + [0, 57, 255]) if i % 2 else rng.randrange(256)
                        for i in range(n))
            packed = encode3(raw, dictionary)
            self.assertEqual(reference_decode3(packed), raw)
        broken = bytearray(encode3(bytes(range(64)), dictionary))
        broken[-1] = 1
        with self.assertRaises(ValueError):
            reference_decode3(bytes(broken))

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
