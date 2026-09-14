"""Native byte parity and adversarial archives; needs C compiler and libzstd.

python3 -m unittest discover -s tests -p test_offline_storage.py -v
No model download, Python packages or network access required.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import offline_model as om  # noqa: E402
import pack_trunk  # noqa: E402
from test_remote_storage import make_shard  # noqa: E402


class OfflineStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cflags = shlex.split(os.environ.get("ZSTD_CFLAGS", ""))
        libs = shlex.split(os.environ.get("ZSTD_LIBS", "-lzstd"))
        omp = shlex.split(os.environ.get("TEST_OMP_FLAGS", "-fopenmp"))
        includes = ["-I" + str(ROOT / p) for p in (
            "src/io", "include/k3", "include", "src/model", "src/cache",
            "src/core", "third_party")]
        base = [os.environ.get("CC", "cc"), "-O1", "-std=gnu99", "-Wall", "-Wextra",
                "-Werror", "-Wshadow", "-Wpointer-arith", *includes]
        cls.native = cls.root / "native"
        cls.no_codec = cls.root / "no_codec"
        cls.parity = cls.root / "parity"
        cls.stream = cls.root / "stream"
        native_flags = shlex.split(os.environ.get("K3_TEST_NATIVE_CFLAGS", ""))
        subprocess.run([*base, *cflags, *native_flags, "-DK3_WITH_ZSTD", *omp,
                        str(ROOT / "tests/unit/test_zfile.c"), *libs,
                        "-o", str(cls.native)], check=True)
        subprocess.run([*base, str(ROOT / "tests/unit/test_zfile.c"),
                        "-o", str(cls.no_codec)], check=True)
        subprocess.run([*base, *cflags, "-DK3_WITH_ZSTD",
                        str(ROOT / "tests/unit/test_remote.c"),
                        str(ROOT / "src/io/k3_st.c"), *libs,
                        "-o", str(cls.parity)], check=True)
        subprocess.run([*base, *cflags, "-DK3_WITH_ZSTD", "-ffp-contract=off",
                        "-Wno-unknown-pragmas", *[str(ROOT / p) for p in (
                            "tests/unit/test_model_stream.c", "src/model/k3_bind.c",
                            "src/core/k3_ops.c", "src/io/k3_st.c")],
                        *libs, "-lm", "-o", str(cls.stream)], check=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def run_c(self, program, *args, ok=True):
        result = subprocess.run([str(program), *map(str, args)], text=True,
                                capture_output=True, timeout=60,
                                env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertNotIn("ERROR: AddressSanitizer", result.stderr)
        self.assertNotIn("ERROR: LeakSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        return result

    def archive(self, raw=None, shuffle="off"):
        if raw is None:
            raw = random.Random(7823).randbytes(3 * (64 << 10) + 37)
        source, dest = self.path / "raw", self.path / "raw.k3z"
        source.write_bytes(raw)
        report = om.pack_file(source, dest, block=64 << 10, shuffle=shuffle)
        self.assertEqual(report["stored_bytes"], dest.stat().st_size)
        self.assertEqual(report["sha256"], hashlib.sha256(raw).hexdigest())
        return source, dest

    def reject(self, source, dest):
        with self.assertRaises((ValueError, OSError)):
            with om.Reader(dest) as reader:
                reader.read()
        self.run_c(self.native, source, dest, ok=False)

    def test_random_bytes_concurrent_native_and_python_reads(self):
        source, dest = self.archive()
        self.run_c(self.native, source, dest)
        raw = source.read_bytes()
        with om.Reader(dest) as reader:
            for offset, n in ((0, 0), (3, 123), (65529, 65551),
                              (len(raw) - 3, 99), (len(raw) + 5, 10)):
                reader.seek(offset)
                self.assertEqual(reader.read(n), raw[offset:offset + n])
            reader.seek(-37, os.SEEK_END)
            self.assertEqual(reader.read(), raw[-37:])
            with self.assertRaises(ValueError):
                reader.seek(-1)

    def selective(self, random_scales=False):
        # Odd boundaries, scales longer than a block, three separate matrices,
        # all 256 byte values, and multi-MiB raw runs. No released weights.
        generator = random.Random(63011)
        header, payload = {}, bytearray()
        for matrix in ("w1", "w2", "w3"):
            prefix = "language_model.model.layers.1.block_sparse_moe.experts.0." + matrix
            for suffix, shape, raw in (
                    ("weight_packed", [257, 4112], generator.randbytes(257 * 4112)),
                    ("weight_scale", [257, 257], generator.randbytes(257 * 257)
                     if random_scales else (bytes(range(256)) + b"y" * (257 * 257 - 256)))):
                name = prefix + "." + suffix
                lo = len(payload)
                payload.extend(raw)
                header[name] = {"dtype": "U8", "shape": shape,
                                "data_offsets": [lo, len(payload)]}
        encoded = json.dumps(header).encode()
        source, dest = self.path / "select.safetensors", self.path / "select.safetensors.k3z"
        source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        report = om.pack_file(source, dest, block=64 << 10, policy="scales")
        return source, dest, report

    def test_selective_raw_bypass_and_complete_random_range_parity(self):
        source, dest, report = self.selective()
        self.run_c(self.native, source, dest)  # concurrent reads + injected decoder failure
        want = source.read_bytes()
        self.assertEqual(report["stored_bytes"], dest.stat().st_size)
        self.assertEqual(report["source_bytes"], len(want))
        self.assertEqual(report["sha256"], hashlib.sha256(want).hexdigest())
        self.assertEqual(report["scale_bytes"], 3 * 257 * 257)
        self.assertEqual(report["compressed_scale_bytes"], report["scale_bytes"])
        with om.Reader(dest) as reader:
            self.assertEqual(reader.read(), want)
            boundaries = [0, *reader.ends]
            for boundary in boundaries:
                off = max(0, boundary - 3)
                reader.seek(off)
                self.assertEqual(reader.read(37), want[off:off + 37])
            # A failing decoder must not matter when only a raw extent is requested.
            with mock.patch.object(reader.codec, "decompress", side_effect=AssertionError):
                for index, (_offset, stored, kind) in enumerate(reader.entries):
                    if kind == om.RAW_EXTENT:
                        start = reader.ends[index - 1] if index else 0
                        reader.seek(start)
                        self.assertEqual(reader.read(min(37, stored)), want[start:start + min(37, stored)])
        self.run_c(self.no_codec, source, dest, ok=False)

    def test_selective_incompressible_scales_fall_back_to_raw(self):
        source, dest, report = self.selective(random_scales=True)
        self.assertEqual(report["compressed_scale_bytes"], 0)
        self.assertEqual(report["raw_payload_bytes"], source.stat().st_size)
        self.assertEqual(report["stored_bytes"], report["source_bytes"] + report["index_bytes"])
        self.run_c(self.native, source, dest)

    @staticmethod
    def repair_map_index(data):
        count = om.HEADER.unpack_from(data)[3]
        end = om.HEADER.size + count * om.MAP_ENTRY.size
        checksum = om.index_hash(data[om.HEADER.size:end], om.index_hash(data[:40]))
        struct.pack_into("<Q", data, 40, checksum)

    def test_selective_corruption_and_invalid_extent_geometry_fail_closed(self):
        source, dest, _ = self.selective()
        original = dest.read_bytes()
        for offset in (8, 16, 20, 24, 40, 48, 56, 64, 72, 76):
            with self.subTest(offset=offset):
                data = bytearray(original)
                data[offset] ^= 128
                dest.write_bytes(data)
                self.reject(source, dest)
        # Recompute the metadata checksum to exercise bounds independently of it.
        for entry_index, field, value in ((0, 0, 0), (0, 0, len(source.read_bytes()) + 1),
                                          (0, 1, 0), (0, 2, (1 << 64) - 1),
                                          (0, 3, 7), (0, 4, 1), (1, 0, 1)):
            data = bytearray(original)
            at = om.HEADER.size + entry_index * om.MAP_ENTRY.size
            entry = list(om.MAP_ENTRY.unpack_from(data, at))
            entry[field] = value
            om.MAP_ENTRY.pack_into(data, at, *entry)
            self.repair_map_index(data)
            dest.write_bytes(data)
            self.reject(source, dest)
        for data in (original[:10], original[:-1], original + b"x"):
            dest.write_bytes(data)
            self.reject(source, dest)
        dest.write_bytes(original)
        with om.Reader(dest) as reader:
            compressed = [(off, n) for off, n, flag in reader.entries if flag == 0]
        for off, n in compressed:
            data = bytearray(original)
            data[off + n - 1] ^= 1
            dest.write_bytes(data)
            self.reject(source, dest)

    def test_selective_scale_frame_identity_and_position_are_bound(self):
        source, dest, _ = self.selective()
        original = dest.read_bytes()
        with om.Reader(dest) as reader:
            index = next(i for i, e in enumerate(reader.entries) if e[2] == 0)
            off, n, _ = reader.entries[index]
            start = reader.ends[index - 1] if index else 0
            length = reader.ends[index] - start
            raw = reader.codec.decompress(original[off:off + n], length + om.PREFIX.size)
            for prefix_offset in (0, 16, 24, 28):
                changed = bytearray(raw)
                changed[prefix_offset] ^= 1
                frame = reader.codec.compress(bytes(changed))
                data = bytearray(original[:off] + frame + original[off + n:])
                for i in range(len(reader.entries)):
                    at = om.HEADER.size + i * om.MAP_ENTRY.size
                    entry = list(om.MAP_ENTRY.unpack_from(data, at))
                    if i == index:
                        entry[2] = len(frame)
                    elif i > index:
                        entry[1] += len(frame) - n
                    om.MAP_ENTRY.pack_into(data, at, *entry)
                self.repair_map_index(data)
                dest.write_bytes(data)
                self.reject(source, dest)

    def test_selective_raw_damage_needs_full_hash_verification(self):
        _source, dest, report = self.selective()
        data = bytearray(dest.read_bytes())
        _, off, _, kind, _ = om.MAP_ENTRY.unpack_from(data, om.HEADER.size)
        self.assertEqual(kind, om.RAW_EXTENT)
        data[off] ^= 1
        dest.write_bytes(data)
        # Raw payload deliberately has ordinary-file integrity, not frame checksums.
        with om.Reader(dest) as reader:
            self.assertNotEqual(hashlib.sha256(reader.read()).hexdigest(), report["sha256"])

    def test_selective_plan_preflight_and_output_cap(self):
        source, _dest, _ = self.selective()
        raw = source.read_bytes()
        bad = self.path / "bad.safetensors"
        bad.write_bytes(raw[:100])
        with self.assertRaises(ValueError):
            om.pack_file(bad, self.path / "bad.k3z", policy="scales")
        n = struct.unpack_from("<Q", raw)[0]
        header = json.loads(raw[8:8 + n])
        name = next(k for k in header if k.endswith("weight_scale"))
        header[name]["shape"][1] -= 1
        encoded = json.dumps(header).encode()
        bad.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + n:])
        with self.assertRaisesRegex(ValueError, "paired E8M0"):
            om.scale_plan(bad)
        other = self.path / "capped.k3z"
        with self.assertRaisesRegex(OSError, "exceeds"):
            om.pack_file(source, other, policy="scales", limit=80000)
        self.assertFalse(other.exists())
        self.assertFalse(Path(str(other) + ".part").exists())
        self.assertEqual(source.read_bytes(), raw)

    def test_selective_all_raw_shard_and_trunk_packing(self):
        source = self.model()
        packed = self.path / "mapped-model"
        with contextlib.redirect_stdout(io.StringIO()):
            om.pack_model(source, packed, policy="scales")
            self.assertEqual(pack_trunk.main([str(source), str(self.path / "ta"), "1"]), 0)
            self.assertEqual(pack_trunk.main([str(packed), str(self.path / "tb"), "1"]), 0)
        self.run_c(self.parity, source, packed)
        self.assertEqual((self.path / "ta/trunk.bin").read_bytes(),
                         (self.path / "tb/trunk.bin").read_bytes())

    def test_shuffle_restores_odd_length_and_cross_block_reads(self):
        source, dest = self.archive(shuffle="on")
        self.run_c(self.native, source, dest)
        with om.Reader(dest) as reader:
            self.assertEqual(reader.read(), source.read_bytes())

    def test_empty_file(self):
        source, dest = self.archive(b"")
        self.run_c(self.native, source, dest)
        with om.Reader(dest) as reader:
            self.assertEqual(reader.read(), b"")

    def test_no_codec_fails_with_build_instruction(self):
        source, dest = self.archive()
        result = self.run_c(self.no_codec, source, dest, ok=False)
        self.assertIn("make ZSTD=1", result.stderr)

    def test_corrupt_payload_rejected(self):
        source, dest = self.archive()
        data = bytearray(dest.read_bytes())
        # A checksum bit is guaranteed to matter; compressed padding bits need not
        # change the decoded bytes and are legitimately ignored by the codec.
        data[-1] ^= 1
        dest.write_bytes(data)
        self.reject(source, dest)

    def test_swapped_valid_frames_rejected(self):
        # Incompressible equal-size blocks have equal-size frames, so swapping
        # payloads passes all offset/length checks and isolates the identity check.
        source, dest = self.archive()
        data = bytearray(dest.read_bytes())
        a, na, _ = om.ENTRY.unpack_from(data, om.HEADER.size)
        b, nb, _ = om.ENTRY.unpack_from(data, om.HEADER.size + om.ENTRY.size)
        self.assertEqual(na, nb)
        data[a:a + na], data[b:b + nb] = data[b:b + nb], data[a:a + na]
        dest.write_bytes(data)
        self.reject(source, dest)

    def test_transform_flag_flip_rejected(self):
        source, dest = self.archive()
        data = bytearray(dest.read_bytes())
        data[om.HEADER.size + 12] ^= 1
        dest.write_bytes(data)
        self.reject(source, dest)

    def test_header_and_index_rejected(self):
        source, dest = self.archive()
        original = dest.read_bytes()
        for offset in (0, 8, 16, 20, 24, 40, 48, 56, 60):
            with self.subTest(offset=offset):
                data = bytearray(original)
                data[offset] ^= 128
                dest.write_bytes(data)
                self.reject(source, dest)
        for data in (original[:10], original[:-1], original + b"x"):
            dest.write_bytes(data)
            self.reject(source, dest)

    def test_unchecksummed_frame_rejected(self):
        source, dest = self.archive()
        data = bytearray(dest.read_bytes())
        offset, _, _ = om.ENTRY.unpack_from(data, om.HEADER.size)
        data[offset + 4] &= ~4
        dest.write_bytes(data)
        self.reject(source, dest)

    def test_output_limit_cleans_partial_and_preserves_input(self):
        source = self.path / "raw"
        raw = random.Random(98).randbytes(160000)
        source.write_bytes(raw)
        dest = self.path / "raw.k3z"
        with self.assertRaisesRegex(OSError, "exceeds"):
            om.pack_file(source, dest, block=64 << 10, limit=80000)
        self.assertEqual(source.read_bytes(), raw)
        self.assertFalse(dest.exists())
        self.assertFalse(Path(str(dest) + ".part").exists())

    def test_existing_output_and_part_never_overwritten(self):
        source, dest = self.archive()
        before = dest.read_bytes()
        with self.assertRaises(FileExistsError):
            om.pack_file(source, dest)
        self.assertEqual(dest.read_bytes(), before)
        other = self.path / "other.k3z"
        part = Path(str(other) + ".part")
        part.write_bytes(b"another conversion")
        with self.assertRaises(FileExistsError):
            om.pack_file(source, other)
        self.assertEqual(part.read_bytes(), b"another conversion")

    def test_failed_roundtrip_not_published(self):
        source = self.path / "raw"
        source.write_bytes(b"payload")
        dest = self.path / "raw.k3z"
        with mock.patch.object(om.Zstd, "decompress", return_value=b"wrong"):
            with self.assertRaisesRegex(ValueError, "roundtrip"):
                om.pack_file(source, dest)
        self.assertFalse(dest.exists())
        self.assertFalse(Path(str(dest) + ".part").exists())

    def model(self):
        source = self.path / "model"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(make_shard())
        return source

    def test_safetensors_raw_aligned_widened_and_trunk_packing(self):
        source = self.model()
        packed = self.path / "packed"
        with contextlib.redirect_stdout(io.StringIO()):
            om.pack_model(source, packed, block=1 << 20)
        self.run_c(self.parity, source, packed)
        self.assertFalse((packed / ".k3-incomplete").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pack_trunk.main([str(source), str(self.path / "ta"), "1"]), 0)
            self.assertEqual(pack_trunk.main([str(packed), str(self.path / "tb"), "1"]), 0)
        self.assertEqual((self.path / "ta/trunk.bin").read_bytes(),
                         (self.path / "tb/trunk.bin").read_bytes())

    def test_streamed_embeddings_and_logits(self):
        source = self.path / "fixture"
        shutil.copytree(ROOT / "tests/fixtures/st", source)
        (source / "config.json").write_text("{}")
        packed = self.path / "packed"
        with contextlib.redirect_stdout(io.StringIO()):
            om.pack_model(source, packed)
        self.run_c(self.parity, source, packed)
        self.run_c(self.stream, packed)

    def test_partial_model_is_not_runnable(self):
        source = self.model()
        packed = self.path / "packed"
        with self.assertRaises(OSError):
            om.pack_model(source, packed, limit=2000)
        self.assertTrue((packed / ".k3-incomplete").exists())
        self.assertLessEqual(sum(p.stat().st_size for p in packed.iterdir()), 2000)
        result = self.run_c(self.parity, source, packed, ok=False)
        self.assertIn("incomplete", result.stderr)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            pack_trunk.main([str(packed), str(self.path / "trunk"), "1"])

    def test_copied_index_names_files_that_exist(self):
        # A verbatim copy would advertise .safetensors shards the packed directory does
        # not contain: an external tool reading the index sees a complete checkpoint and
        # gets missing files. Assert against the directory, not against a literal, so
        # this fails if either the index or the shard naming drifts.
        source = self.model()
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 1}, "weight_map": {"a": "model.safetensors"}}))
        packed = self.path / "packed"
        with contextlib.redirect_stdout(io.StringIO()):
            om.pack_model(source, packed)
        index = json.loads((packed / "model.safetensors.index.json").read_text())
        self.assertEqual(index["weight_map"], {"a": "model.safetensors.k3z"})
        for target in index["weight_map"].values():
            self.assertTrue((packed / target).is_file(), target)
        self.assertEqual(index["metadata"], {"total_size": 1})

    def test_missing_shards_and_header_only_source_refused(self):
        source = self.model()
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "model.safetensors", "b": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "index"):
            om.pack_model(source, self.path / "bad")
        (source / ".k3-remote").write_text("header-only")
        with self.assertRaisesRegex(ValueError, "complete local"):
            om.pack_model(source, self.path / "bad2")


if __name__ == "__main__":
    unittest.main()
