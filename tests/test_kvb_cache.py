"""The --kvb-cache gates (the private research notes, findings 98-99, active-layer kv_b reuse).

Under --trunk-rows --kv-latent every cached position rebuilds its k and v from kv_b rows, so the
row pipeline reread kv_b once per position per MLA layer. --kvb-cache active holds the current
layer's kv_b in one buffer, read once per layer visit; --kvb-cache pin keeps every layer's.
These gates check, on the tiny synthetic model (4 MLA layers with distinct weights):
  - generated ids and dumped logits are bit-identical to the unbuffered row pipeline, on a plain
    and on a compressed trunk;
  - reader counts are exact: every kv_b pass the buffer serves is one fewer streamed matrix pass
    (base.matrix_calls - kvb.matrix_calls == kvb.fills + kvb.hits), fills happen once per MLA
    layer visit (active) or once per MLA layer (pin), each fill reads exactly the kv_b tensor's
    bytes, and nothing falls back;
  - a stale-slot mutant (K3_KVB_MUTANT_STALE: no invalidation, no key check) changes the logits,
    so the logits gate can fail;
  - an unaffordable budget, a missing --trunk-rows or --kv-latent, and an unknown mode are refused.

Build with ZSTD=1 first. K3_TEST_BIN=bin/k3 python3 -m unittest tests.test_kvb_cache -v
"""
from __future__ import annotations

import unittest

import test_offline_cli as toc   # not imported by name, so unittest does not rerun its tests

MLA_LAYERS = 4          # tiny_k3: full_attn_layers 4, 8, 12, 13
KVB_BYTES = 4 * (24 + 16) * 32 * 2   # heads x (qk_nope + v_head) x kv_lora x BF16


class KvbCacheTests(unittest.TestCase):
    setUpClass = classmethod(toc.OfflineCliTests.setUpClass.__func__)
    setUp = toc.OfflineCliTests.setUp
    run_cli = toc.OfflineCliTests.run_cli
    assert_same = toc.OfflineCliTests.assert_same
    long_ids = staticmethod(toc.OfflineCliTests.long_ids)
    run_logits = toc.OfflineCliTests.run_logits
    assert_same_run = toc.OfflineCliTests.assert_same_run
    assert_no_replay = toc.OfflineCliTests.assert_no_replay
    crafted = toc.OfflineCliTests.crafted
    SPEC = toc.OfflineCliTests.SPEC
    _crafted = {}

    def test_all_positions_and_speculative_rollback_match(self):
        # Gate the complete trajectory against serial OFF, including each rejection
        # length 0..3 and a later full acceptance of all four after a partial rollback.
        for accept, second in [(0, False), (1, False), (2, True), (3, False)]:
            prompt, _ = self.crafted(accept, second)
            ids = ",".join(map(str, prompt))
            gen = str(accept + self.SPEC + 3)
            for trunk in (self.trunk, self.ztrunk):
                base = self.run_logits(self.selective, self.args(trunk, ids=ids, gen=gen))
                # Observer negative control: first vector and all IDs remain right;
                # only a later vector is damaged. The old first-only gate misses it.
                corrupted = bytearray(base[2]); corrupted[-1] ^= 1
                with self.assertRaises(AssertionError):
                    self.assert_same_run(base, (base[0], base[1], bytes(corrupted)))
                for mode in ("off", "active", "pin"):
                    with self.subTest(accept=accept, second=second, trunk=trunk.name, mode=mode):
                        serial = self.run_logits(self.selective, self.args(trunk,
                            "--kvb-cache", mode, ids=ids, gen=gen))
                        self.assert_same_run(base, serial)
                        spec = self.run_logits(self.selective, self.args(trunk,
                            "--kvb-cache", mode, "--spec", str(self.SPEC), ids=ids, gen=gen))
                        self.assert_same_run(base, spec)
                        self.assert_no_replay(spec[0])
                        self.assertEqual(spec[0]["spec_trace"][0], [self.SPEC, accept, accept + 1])
                        if second:
                            self.assertEqual(spec[0]["spec_trace"][1], [self.SPEC, self.SPEC, self.SPEC + 1])
                        if mode != "off":
                            want = MLA_LAYERS * spec[0]["forward_sweeps"] if mode == "active" else MLA_LAYERS
                            self.assertEqual(spec[0]["trunk_kvb_fills"], want)
                            self.assertEqual(spec[0]["trunk_kvb_fallbacks"], 0)


    def args(self, trunk, *extra, ids="3,7,11,5,2,8,1,4", gen="3"):
        return ["--ids", ids, "--gen", gen, "--trunk", trunk, "--trunk-gb", "0.0002",
                "--trunk-rows", "--incremental", "--kv-latent", *extra]

    def test_active_and_pin_are_bit_identical_and_read_kvb_once(self):
        for trunk in (self.trunk, self.ztrunk):
            with self.subTest(trunk=trunk.name):
                base = self.run_cli(self.selective, self.args(trunk))
                self.assertEqual(base[0]["trunk_kvb_mode"], "off")
                self.assertEqual(base[0]["trunk_kvb_fills"], 0)
                for mode in ("active", "pin"):
                    kvb = self.run_cli(self.selective, self.args(trunk, "--kvb-cache", mode))
                    self.assert_same(base, kvb)
                    b, k = base[0], kvb[0]
                    self.assertEqual(k["trunk_kvb_mode"], mode)
                    import struct
                    metadata = 13 * (3 * struct.calcsize("P") + 2 * 8) if mode == "pin" else 0
                    self.assertEqual(k["trunk_kvb_metadata_bytes"], metadata)
                    self.assertGreater(k["trunk_kvb_fill_seconds"], 0)
                    self.assertEqual(k["trunk_kvb_fallbacks"], 0)
                    self.assertGreater(k["trunk_kvb_hits"], 0)
                    served = k["trunk_kvb_fills"] + k["trunk_kvb_hits"]
                    self.assertEqual(b["trunk_matrix_calls"] - k["trunk_matrix_calls"], served,
                                     "every kv_b pass served from the buffer is one fewer streamed pass")
                    sweeps = k["forward_sweeps"]
                    want_fills = MLA_LAYERS * sweeps if mode == "active" else MLA_LAYERS
                    self.assertEqual(k["trunk_kvb_fills"], want_fills)
                    self.assertEqual(k["trunk_kvb_fill_bytes"], want_fills * KVB_BYTES)
                    if trunk == self.trunk:
                        self.assertLess(k["trunk_bytes_read"], b["trunk_bytes_read"])
                    slot = KVB_BYTES + 2 * 4096
                    self.assertEqual(k["trunk_kvb_buffer_bytes"], slot if mode == "active" else MLA_LAYERS * slot)

    def test_longer_context_is_still_bit_identical(self):
        base = self.run_cli(self.selective, self.args(self.trunk, ids=self.long_ids(40), gen="4"))
        for mode in ("active", "pin"):
            kvb = self.run_cli(self.selective, self.args(self.trunk, "--kvb-cache", mode,
                                                         ids=self.long_ids(40), gen="4"))
            self.assert_same(base, kvb)

    def test_stale_slot_mutant_changes_the_logits(self):
        base = self.run_cli(self.selective, self.args(self.trunk))
        stale = self.run_cli(self.selective, self.args(self.trunk, "--kvb-cache", "active"),
                             env={"K3_KVB_MUTANT_STALE": "1"})
        self.assertNotEqual(base[1], stale[1], "a stale kv_b slot must change the logits")

    def test_a_failed_fill_is_sticky_and_emits_nothing(self):
        # K3_KVB_FAULT_FILL makes the kv_b fill itself fail: the run must abort with no output
        # (read_error is sticky) rather than compute on an unfilled buffer. The control run
        # without the hook passes, so the failure is the fill's.
        self.run_cli(self.selective, self.args(self.trunk, "--kvb-cache", "active"))
        for mode in ("active", "pin"):
            with self.subTest(mode=mode):
                r = self.run_cli(self.selective, self.args(self.trunk, "--kvb-cache", mode), ok=False,
                                 env={"K3_KVB_FAULT_FILL": "1"})
                self.assertIn("forward pass failed", r.stderr)

    def test_a_truncated_trunk_fails_in_every_mode(self):
        # cut trunk.bin in the middle of the last MLA layer's kv_b: every mode must refuse to emit
        import json
        import shutil
        cut = self.path / "cut"
        shutil.copytree(self.trunk, cut)
        meta = json.loads((cut / "trunk.json").read_text())
        layers = meta["layers"]
        last = layers[12]
        kvb = [t for name, t in last["tensors"].items() if "kv_b" in name][0]
        size = last["file_off"] + kvb["off"] + kvb["nbytes"] // 2
        with open(cut / "trunk.bin", "r+b") as f:
            f.truncate(size)
        for mode in ("off", "active", "pin"):
            with self.subTest(mode=mode):
                self.run_cli(self.selective, self.args(cut, "--kvb-cache", mode), ok=False)

    def test_refusals(self):
        for extra, why in ((["--kvb-cache", "active", "--trunk-gb", "0.00002"], "cannot pay"),
                           (["--kvb-cache", "pin", "--trunk-gb", "0.00005"], "cannot pay"),
                           (["--kvb-cache", "sometimes"], "takes off, active or pin")):
            with self.subTest(extra=extra):
                r = self.run_cli(self.selective, [*self.args(self.trunk), *extra], ok=False)
                self.assertIn(why, r.stdout + r.stderr)
        for drop in ("--trunk-rows", "--kv-latent"):
            with self.subTest(drop=drop):
                args = [a for a in self.args(self.trunk, "--kvb-cache", "active") if a != drop]
                r = self.run_cli(self.selective, args, ok=False)
                self.assertIn("--kvb-cache needs --trunk-rows and --kv-latent", r.stderr)


if __name__ == "__main__":
    unittest.main()
