"""Offline fixtures: actual C range reads, HTTP failures, cache bounds and parity.

Run: python3 -m unittest discover -s tests -p test_remote_storage.py -v
No released model, Python packages, or external network access is needed.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import http.server
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_trunk  # noqa: E402
import remote_model as rm  # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        name = urllib.parse.urlparse(self.path).path.lstrip("/")
        payload = self.server.files.get(name)
        if payload is None:
            self.send_error(404)
            return
        start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        mode = self.server.mode
        data = payload[start:end + 1]
        self.server.requests.append((name, start, end))
        self.send_response(200 if mode == "full" else 206)
        total = len(payload) + (1 if mode == "size" else 0)
        claimed_start = start + (1 if mode == "wrong" else 0)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (claimed_start, end, total))
        if mode == "encoded":
            self.send_header("Content-Encoding", "gzip")
        if mode == "short":
            data = data[:-1]
        if mode == "extra":
            data += b"x"
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)


def make_shard():
    data = bytearray()
    header = {}
    values = [
        ("large.u8", "U8", [rm.BLOCK * 2 + 73],
         bytes(range(256)) * (rm.BLOCK // 128) + bytes(range(73))),
        ("plain.bf16", "BF16", [6], struct.pack("<6H", 0, 0x3f80, 0xbf80,
                                               0x7f80, 0x7fc1, 1)),
        ("plain.f16", "F16", [6], struct.pack("<6H", 0, 0x3c00, 0xbc00,
                                             0x7c00, 0x7e01, 1)),
        ("plain.f32", "F32", [4], struct.pack("<4f", -3.5, 0, 1.25, 100)),
        ("language_model.model.layers.0.norm.weight", "F32", [2],
         struct.pack("<2f", 0.75, -0.5)),
    ]
    for name, dtype, shape, raw in values:
        start = len(data)
        data.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [start, len(data)]}
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + data


class RemoteStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.workpath = Path(cls.work.name)
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.files = {"model.safetensors": make_shard()}
        cls.server.files.update({p.name: p.read_bytes()
                                 for p in (ROOT / "tests/fixtures/st").glob("*.safetensors")})
        cls.server.mode = "ok"
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.addClassCleanup(cls.thread.join)
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)
        cls.base = "http://127.0.0.1:%d/" % cls.server.server_port
        includes = ["-I" + str(ROOT / path) for path in (
            "src/io", "include/k3", "include", "src/model", "src/cache",
            "src/core", "third_party")]
        compiler = os.environ.get("CC", "cc")
        cls.driver = cls.workpath / "test_remote"
        subprocess.run([compiler, "-O1", "-std=gnu99", "-Wall", "-Wextra", "-Werror",
                        "-Wshadow", "-Wpointer-arith", *includes,
                        str(ROOT / "tests/unit/test_remote.c"),
                        str(ROOT / "src/io/k3_st.c"), "-o", str(cls.driver)], check=True)
        cls.stream = cls.workpath / "test_model_stream"
        subprocess.run([compiler, "-O1", "-std=gnu99", "-ffp-contract=off",
                        "-Wno-unknown-pragmas", *includes,
                        *[str(ROOT / path) for path in (
                            "tests/unit/test_model_stream.c", "src/model/k3_bind.c",
                            "src/core/k3_ops.c", "src/io/k3_st.c")],
                        "-lm", "-o", str(cls.stream)], check=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.workpath)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server.mode = "ok"
        self.server.requests.clear()

    def cache(self, blocks=2):
        cache = rm.DiskCache(self.root / "cache", blocks * (rm.BLOCK + 32))
        self.addCleanup(cache.close)
        return cache

    def mirror(self, names=("model.safetensors",)):
        local, remote = self.root / "local", self.root / "remote"
        local.mkdir()
        remote.mkdir()
        manifest = {"version": 1, "repo": "fixture/model", "revision": "a" * 40,
                    "base_url": self.base, "shards": {}, "metadata": {}}
        for name in names:
            data = self.server.files[name]
            (local / name).write_bytes(data)
            size = struct.unpack("<Q", data[:8])[0] + 8
            header = data[:size]
            (remote / name).write_bytes(header)
            manifest["shards"][name] = {"size": len(data), "header_sha256": rm.digest(header)}
        rm.save_manifest(remote, manifest)
        return local, remote

    def test_exact_http_range(self):
        data, size = rm.read_range(self.base + "model.safetensors", 17, 193)
        self.assertEqual(data, self.server.files["model.safetensors"][17:210])
        self.assertEqual(size, len(self.server.files["model.safetensors"]))

    def test_refuse_bad_http_responses(self):
        total = len(self.server.files["model.safetensors"])
        for mode in ("full", "wrong", "size", "encoded", "short", "extra"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                with self.assertRaises(ValueError):
                    rm.read_range(self.base + "model.safetensors", 0, 16, total)

    def test_invalid_http_range_fails_before_request(self):
        for offset, length in ((-1, 8), (0, 0), (0, rm.MAX_HEADER + 1)):
            with self.assertRaises(ValueError):
                rm.read_range(self.base + "model.safetensors", offset, length)
        self.assertEqual(self.server.requests, [])

    def test_header_overlap_and_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            rm.decode(b'{"x":1,"x":2}')
        header = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                  "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}}
        raw = json.dumps(header).encode()
        with self.assertRaises(ValueError):
            rm.validate_header(struct.pack("<Q", len(raw)) + raw, 11 + len(raw))

    def test_manifest_or_header_changes_are_rejected(self):
        _, remote = self.mirror()
        original = (remote / "model.safetensors").read_bytes()
        (remote / "model.safetensors").write_bytes(original[:-1] + b"!")
        with self.assertRaises(ValueError):
            rm.Source(remote)
        (remote / "model.safetensors").write_bytes(original)
        data = json.loads((remote / "remote.json").read_text())
        data["revision"] = "main"
        (remote / "remote.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            rm.Source(remote)

    def test_cache_lru_bound_and_restart(self):
        cache = self.cache(2)
        keys = [rm.digest(str(i).encode()) for i in range(3)]
        value = b"a" * rm.BLOCK
        for key in keys[:2]:
            self.assertEqual(cache.get(key, len(value), lambda: value), value)
        cache.get(keys[0], len(value), lambda: self.fail("cache miss"))
        cache.get(keys[2], len(value), lambda: value)
        self.assertNotIn(keys[1], cache.entries)
        self.assertLessEqual(cache.used, cache.limit)
        self.assertLessEqual(sum(p.stat().st_size for p in cache.root.glob("*.blk")),
                             cache.limit)
        cache.close()
        reopened = rm.DiskCache(cache.root, rm.BLOCK + 32)
        try:
            self.assertLessEqual(reopened.used, reopened.limit)
            self.assertEqual(len(reopened.entries), 1)
        finally:
            reopened.close()

    def test_cache_refuses_second_writer(self):
        cache = self.cache()
        with self.assertRaises(RuntimeError):
            rm.DiskCache(cache.root, cache.limit)

    def test_corrupt_cache_is_refetched(self):
        cache = self.cache()
        key = rm.digest(b"corrupt")
        cache.get(key, 3, lambda: b"abc")
        path = cache.root / (key + ".blk")
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        self.assertEqual(cache.get(key, 3, lambda: b"def"), b"def")

    def test_failed_fill_releases_reservation(self):
        cache = self.cache()
        key = rm.digest(b"failed")
        with self.assertRaises(ValueError):
            cache.get(key, 3, lambda: b"a")
        self.assertEqual(cache.reserved, 0)
        self.assertFalse(cache.pending)
        self.assertFalse(list(cache.root.glob("*.part")))
        self.assertEqual(cache.get(key, 3, lambda: b"abc"), b"abc")

    def test_duplicate_concurrent_misses_fetch_once(self):
        cache = self.cache()
        key = rm.digest(b"same")
        calls = []
        def fetch():
            calls.append(1)
            time.sleep(0.03)
            return b"abc"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda _: cache.get(key, 3, fetch), range(8)))
        self.assertEqual(values, [b"abc"] * 8)
        self.assertEqual(len(calls), 1)

    def test_concurrent_distinct_fills_stay_bounded(self):
        cache = self.cache(1)
        value = b"x" * rm.BLOCK
        peak = []
        def fill(i):
            key = rm.digest(str(i).encode())
            def fetch():
                with cache.cv:
                    peak.append(cache.used + cache.reserved)
                time.sleep(0.01)
                return value
            return cache.get(key, len(value), fetch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(fill, range(4))), [value] * 4)
        self.assertLessEqual(max(peak), cache.limit)
        self.assertLessEqual(cache.used, cache.limit)

    def test_cross_block_eof_and_offline_hit(self):
        _, remote = self.mirror()
        source = rm.Source(remote, self.cache(3))
        offset = rm.BLOCK - 9
        expected = self.server.files["model.safetensors"][offset:offset + 31]
        self.assertEqual(source.read("model.safetensors", offset, 31), expected)
        source.offline = True
        self.assertEqual(source.read("model.safetensors", offset, 31), expected)
        with self.assertRaises(OSError):
            source.read("model.safetensors", 2 * rm.BLOCK + 1, 7)
        with self.assertRaises(ValueError):
            source.read("model.safetensors", source.size("model.safetensors") - 1, 2)

    def test_c_reader_matches_local_bytes(self):
        local, remote = self.mirror()
        source = rm.Source(remote, self.cache(3))
        with rm.bridge(source) as path:
            result = subprocess.run([str(self.driver), str(local), str(remote)],
                                    env=dict(os.environ, K3_REMOTE_SOCKET=path),
                                    capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("stored/local byte parity: PASSED", result.stdout)
        self.assertLess(sum(p.stat().st_size for p in remote.glob("*.safetensors")), 4096)

    def test_c_streamed_logits_match_fixture(self):
        names = tuple(p.name for p in (ROOT / "tests/fixtures/st").glob("*.safetensors"))
        _, remote = self.mirror(names)
        source = rm.Source(remote, self.cache())
        with rm.bridge(source) as path:
            result = subprocess.run([str(self.stream), str(remote)],
                                    env=dict(os.environ, K3_REMOTE_SOCKET=path),
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("model stream parity: PASSED", result.stdout)

    def test_missing_bridge_is_fatal(self):
        local, remote = self.mirror()
        env = dict(os.environ)
        env.pop("K3_REMOTE_SOCKET", None)
        result = subprocess.run([str(self.driver), str(local), str(remote)],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("needs tools/remote_model.py run", result.stderr)

    def test_wrong_daemon_identity_is_fatal(self):
        local, remote = self.mirror()
        source = rm.Source(remote, self.cache())
        source.manifest["id"] = "0" * 64
        with rm.bridge(source) as path:
            result = subprocess.run([str(self.driver), str(local), str(remote)],
                                    env=dict(os.environ, K3_REMOTE_SOCKET=path),
                                    capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)

    def test_pack_fetches_only_the_trunk_and_matches_local(self):
        local, remote = self.mirror()
        local_out, remote_out = self.root / "lt", self.root / "rt"
        self.assertEqual(pack_trunk.main([str(local), str(local_out), "1"]), 0)
        self.server.requests.clear()
        self.assertEqual(pack_trunk.main([str(remote), str(remote_out), "1"]), 0)
        self.assertEqual((local_out / "trunk.bin").read_bytes(),
                         (remote_out / "trunk.bin").read_bytes())
        self.assertEqual(json.loads((local_out / "trunk.json").read_text()),
                         json.loads((remote_out / "trunk.json").read_text()))
        self.assertEqual(sum(end - start + 1 for _, start, end in self.server.requests), 8)
        self.assertFalse((remote / "range-cache").exists())

    def test_pack_failure_does_not_publish_partial_trunk(self):
        _, remote = self.mirror()
        self.server.mode = "short"
        out = self.root / "bad-trunk"
        with self.assertRaises(ValueError):
            pack_trunk.main([str(remote), str(out), "1"])
        self.assertFalse((out / "trunk.bin").exists())
        self.assertFalse((out / "trunk.json").exists())
        self.assertTrue((out / "trunk.bin.part").exists())

    def test_prepare_downloads_only_metadata(self):
        out = self.root / "prepared"
        info = {"sha": "a" * 40, "siblings": [{"rfilename": "model.safetensors"}]}
        payload = self.server.files["model.safetensors"]
        def small(url, limit=0):
            return json.dumps(info).encode() if "/api/models/" in url else b"{}"
        def byte_range(url, offset, length, total=None):
            return payload[offset:offset + length], len(payload)
        args = type("Args", (), {"repo": "fixture/model", "revision": "main",
                                 "directory": str(out)})()
        with mock.patch.object(rm, "read_small", side_effect=small), \
             mock.patch.object(rm, "read_range", side_effect=byte_range) as ranges:
            rm.prepare(args)
        self.assertEqual(len(ranges.call_args_list), 2)
        self.assertEqual(rm.load_manifest(out)["revision"], "a" * 40)
        self.assertLess((out / "model.safetensors").stat().st_size, 4096)


if __name__ == "__main__":
    unittest.main()
