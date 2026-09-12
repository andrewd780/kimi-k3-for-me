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
        subprocess.run([*base, *cflags, "-DK3_WITH_ZSTD", *omp,
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
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
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
