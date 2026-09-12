#!/usr/bin/env python3
"""Exact HTTP range storage for K3; Python 3.9+, stdlib only, Linux/macOS/WSL.

prepare downloads metadata only. pack copies just the dense trunk. run supplies
missing expert ranges to the C reader through a private Unix socket. This trades
local capacity for network traffic; it is not weight compression or a speedup.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

BLOCK = 8 << 20
MAX_HEADER = 64 << 20
ERROR = (1 << 64) - 1
REQUEST = struct.Struct("<64sIQQ")
REPLY = struct.Struct("<Q")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object)


def read_small(url, limit=32 << 20):
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise OSError("metadata request did not return HTTP 200")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("metadata exceeds size limit")
    return data


def read_range(url, offset, length, total=None):
    """Reject full-file responses, wrong ranges, transformed or truncated bytes."""
    if offset < 0 or length <= 0 or length > MAX_HEADER:
        raise ValueError("invalid HTTP range")
    end = offset + length - 1
    request = urllib.request.Request(url, headers={
        "Range": "bytes=%d-%d" % (offset, end), "Accept-Encoding": "identity",
    })
    # Never retry malformed successful responses. Only transient HTTP/transport errors.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = re.fullmatch(
                    r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                    response.headers.get("Content-Range", ""),
                )
                if response.status != 206 or expected is None:
                    raise ValueError("server must honor byte ranges with HTTP 206")
                start, stop, size = map(int, expected.groups())
                if (start, stop) != (offset, end) or size <= end:
                    raise ValueError("server returned the wrong byte range")
                if total is not None and size != total:
                    raise ValueError("remote shard size changed")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("compressed HTTP response is not a weight range")
                data = response.read(length + 1)
                if len(data) != length:
                    raise ValueError("truncated or oversized HTTP range")
                return data, size
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise OSError("range retries exhausted")


def validate_header(raw, total):
    if len(raw) < 8:
        raise ValueError("short safetensors header")
    length = struct.unpack("<Q", raw[:8])[0]
    if length == 0 or length > MAX_HEADER or length + 8 != len(raw):
        raise ValueError("invalid safetensors header length")
    header = decode(raw[8:])
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    sizes = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        # Structure first, and as ValueError: these bytes came from the origin, and
        # main() catches ValueError/KeyError but not the TypeError that indexing or
        # unpacking a non-object would raise (entry=5 -> 'int' not subscriptable).
        if not isinstance(entry, dict):
            raise ValueError("invalid tensor entry: " + str(name))
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid tensor span: " + str(name))
        start, end = offsets
        if (entry.get("dtype") not in sizes or len(shape) > 4
                or any(type(n) is not int or n < 0 for n in shape)
                or type(start) is not int or type(end) is not int
                or start < 0 or end < start
                or end > total - len(raw)
                or end - start != math.prod(shape) * sizes[entry["dtype"]]):
            raise ValueError("invalid tensor span: " + name)
        spans.append((start, end))
    pos = 0
    for start, end in sorted(spans):
        if start != pos:
            raise ValueError("overlap or gap in safetensors data")
        pos = end
    if pos + len(raw) != total:
        raise ValueError("header does not cover the full remote shard")
    return header


def identity(manifest):
    body = {key: value for key, value in manifest.items() if key != "id"}
    return digest(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def save_manifest(root, manifest):
    manifest["id"] = identity(manifest)
    (root / "remote.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Published last: an interrupted prepare is never a usable checkpoint.
    (root / ".k3-remote").write_text(manifest["id"] + "\n")


def load_manifest(root):
    root = Path(root)
    manifest = decode((root / "remote.json").read_bytes())
    if (manifest.get("version") != 1 or manifest.get("id") != identity(manifest)
            or (root / ".k3-remote").read_text() != manifest["id"] + "\n"):
        raise ValueError("incomplete or changed remote manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("remote revision is not immutable")
    for name, item in manifest["shards"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name):
            raise ValueError("unsafe shard name")
        raw = (root / name).read_bytes()
        if digest(raw) != item["header_sha256"]:
            raise ValueError("local header changed: " + name)
        validate_header(raw, item["size"])
    for name, expected in manifest["metadata"].items():
        if name not in ("config.json", "tokenizer_config.json", "tiktoken.model"):
            raise ValueError("unexpected metadata filename")
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("local metadata changed: " + name)
    return manifest


def prepare(args):
    repo = args.repo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("expected owner/model for --repo")
    revision = urllib.parse.quote(args.revision, safe="")
    info = decode(read_small("https://huggingface.co/api/models/%s/revision/%s"
                             % (repo, revision)))
    commit = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hub did not return a commit SHA")
    names = sorted(item["rfilename"] for item in info["siblings"]
                   if item["rfilename"].endswith(".safetensors"))
    if not names or any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", n)
                        for n in names):
        raise ValueError("expected top-level safetensors shards")
    base = "https://huggingface.co/%s/resolve/%s/" % (repo, commit)
    root = Path(args.directory).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "repo": repo, "revision": commit,
                "base_url": base, "shards": {}, "metadata": {}}
    for name in ("config.json", "tokenizer_config.json", "tiktoken.model"):
        raw = read_small(base + name)
        (root / name).write_bytes(raw)
        manifest["metadata"][name] = digest(raw)
    for i, name in enumerate(names, 1):
        prefix, total = read_range(base + name, 0, 8)
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= MAX_HEADER or length + 8 > total:
            raise ValueError("impossible remote header: " + name)
        header, _ = read_range(base + name, 8, length, total)
        raw = prefix + header
        validate_header(raw, total)
        (root / name).write_bytes(raw)
        manifest["shards"][name] = {"size": total, "header_sha256": digest(raw)}
        print("header %d/%d: %s" % (i, len(names), name), flush=True)
    save_manifest(root, manifest)
    stored = sum(p.stat().st_size for p in root.iterdir())
    print("Prepared %s@%s; %.2f MB local metadata, no weight payloads."
          % (repo, commit, stored / 1e6))
    print("Next: python3 tools/remote_model.py pack %s <trunk_dir>" % root)


class DiskCache:
    """One writer per directory; reservations cap concurrent fills before writing.

    Cache files contain a SHA-256 followed by a block. The bound includes this
    digest and all in-flight .part bytes, excluding filesystem directory overhead.
    """
    def __init__(self, root, limit):
        import fcntl  # POSIX only; the CLI checks the platform before opening a cache.

        if limit < BLOCK + 32:
            raise ValueError("cache must hold at least one 8 MiB block plus its digest")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.root / ".lock").open("a+b")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError("this cache is already in use") from None
        self.limit = limit
        self.cv = threading.Condition()
        self.entries = collections.OrderedDict()
        self.used = 0
        self.reserved = 0
        self.pending = set()
        self.hits = 0
        self.downloaded = 0
        paths = []
        for path in self.root.iterdir():
            if re.fullmatch(r"[0-9a-f]{64}\.part", path.name):
                path.unlink()
            elif re.fullmatch(r"[0-9a-f]{64}\.blk", path.name):
                paths.append(path)
        for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns):
            size = path.stat().st_size
            if size < 32 or size > BLOCK + 32:
                path.unlink()
                continue
            self.entries[path.stem] = size
            self.used += size
        self._evict(0)

    def _evict(self, need):
        while self.entries and self.used + self.reserved + need > self.limit:
            key, size = self.entries.popitem(last=False)
            (self.root / (key + ".blk")).unlink(missing_ok=True)
            self.used -= size

    def get(self, key, length, fetch):
        need = length + 32
        if not 0 < length <= BLOCK or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid cache block")
        path = self.root / (key + ".blk")
        while True:
            with self.cv:
                while key in self.pending:
                    self.cv.wait()
                cached = key in self.entries
                if cached:
                    # Claim recency before releasing: an evictor running while we read
                    # must not pick this key as its LRU victim.
                    self.entries.move_to_end(key)
                else:
                    self._evict(need)
                    if self.used + self.reserved + need > self.limit:
                        self.cv.wait()
                        continue
                    self.reserved += need
                    self.pending.add(key)
            if not cached:
                break
            # Read and verify OUTSIDE the lock. Holding it across an 8 MiB read plus its
            # SHA-256 serializes every bridge worker on each hit, which is precisely the
            # warm-cache path the concurrency exists for. Racing an evictor is safe: an
            # unlink after our open still reads the old inode, the key IS the content
            # digest, and fills land via an atomic replace -- so the check below accepts
            # only the block actually asked for, whichever version it read.
            try:
                raw = path.read_bytes()
            except OSError:
                raw = b""
            if len(raw) == need and raw[:32] == hashlib.sha256(raw[32:]).digest():
                with self.cv:
                    if key in self.entries:
                        self.entries.move_to_end(key)
                    self.hits += length
                # Advisory only, and the block may have been evicted while we read it.
                with contextlib.suppress(OSError):
                    os.utime(path, None)
                return raw[32:]
            with self.cv:
                stale = self.entries.pop(key, None)
                if stale is not None:
                    self.used -= stale
            path.unlink(missing_ok=True)
        partial = self.root / (key + ".part")
        committed = False
        try:
            data = fetch()
            if len(data) != length:
                raise ValueError("short cache fill")
            with partial.open("xb") as output:
                output.write(hashlib.sha256(data).digest())
                output.write(data)
            with self.cv:
                partial.replace(self.root / (key + ".blk"))
                self.entries[key] = need
                self.used += need
                self.downloaded += length
                self.reserved -= need
                self.pending.remove(key)
                committed = True
                self.cv.notify_all()
            return data
        finally:
            if not committed:
                try:
                    partial.unlink(missing_ok=True)
                finally:
                    with self.cv:
                        self.reserved -= need
                        self.pending.remove(key)
                        self.cv.notify_all()

    def close(self):
        self.lock_file.close()


class Source:
    def __init__(self, root, cache=None, offline=False):
        self.root = Path(root)
        self.manifest = load_manifest(root)
        self.cache = cache
        self.offline = offline

    def size(self, name):
        return self.manifest["shards"][name]["size"]

    def read(self, name, offset, length):
        total = self.size(name)
        if offset < 0 or length < 0 or offset > total - length:
            raise ValueError("range outside remote shard")
        result = bytearray()
        while length:
            block_offset = (offset // BLOCK) * BLOCK if self.cache else offset
            block_length = min(BLOCK, total - block_offset) if self.cache else min(
                BLOCK, length)
            url = self.manifest["base_url"] + name

            def fetch(start=block_offset, count=block_length, source_url=url):
                if self.offline:
                    raise OSError("offline cache miss; original weights are required")
                return read_range(source_url, start, count, total)[0]

            if self.cache:
                key = digest(("%s/%s/%d" % (
                    self.manifest["id"], name, block_offset)).encode())
                data = self.cache.get(key, block_length, fetch)
            else:
                data = fetch()
            within = offset - block_offset
            take = min(length, len(data) - within)
            result.extend(data[within:within + take])
            offset += take
            length -= take
        return bytes(result)

    def reader(self, name):
        return RemoteReader(self, name)


class RemoteReader:
    def __init__(self, source, name):
        self.source, self.name, self.position = source, name, 0

    def seek(self, offset):
        if not 0 <= offset <= self.source.size(self.name):
            raise ValueError("seek outside shard")
        self.position = offset

    def read(self, count):
        count = min(count, self.source.size(self.name) - self.position)
        data = self.source.read(self.name, self.position, count)
        self.position += count
        return data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def receive(stream, length):
    result = bytearray()
    while len(result) < length:
        data = stream.recv(length - len(result))
        if not data:
            raise OSError("short bridge request")
        result.extend(data)
    return bytes(result)


class RangeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(600)
        sent_header = False
        try:
            ident, name_length, offset, length = REQUEST.unpack(
                receive(self.request, REQUEST.size))
            if ident.decode("ascii") != self.server.source.manifest["id"]:
                raise ValueError("wrong checkpoint identity")
            if not 1 <= name_length <= 255 or length > BLOCK:
                raise ValueError("invalid bridge request")
            name = receive(self.request, name_length).decode("ascii")
            with self.server.slots:
                if length == 0:
                    answer = REPLY.pack(self.server.source.size(name))
                else:
                    data = self.server.source.read(name, offset, length)
                    answer = REPLY.pack(len(data))
                self.request.sendall(answer)
                sent_header = True
                if length:
                    self.request.sendall(data)
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            print("remote read failed: %s" % exc, file=sys.stderr)
            if not sent_header:
                with contextlib.suppress(OSError):
                    self.request.sendall(REPLY.pack(ERROR))


_UnixServer = getattr(socketserver, "ThreadingUnixStreamServer",
                      socketserver.ThreadingTCPServer)


class RangeServer(_UnixServer):
    # Non-daemon workers are joined on close before the cache lock is released.
    request_queue_size = 64

    def __init__(self, path, source):
        self.source = source
        self.slots = threading.Semaphore(4)
        super().__init__(path, RangeHandler)


@contextlib.contextmanager
def bridge(source):
    # A short private path fits macOS's smaller sockaddr_un and protects access.
    with tempfile.TemporaryDirectory(prefix="k3-", dir="/tmp") as directory:
        path = str(Path(directory) / "ranges.sock")
        with RangeServer(path, source) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                yield path
            finally:
                server.shutdown()
                thread.join()


def run(args):
    root = Path(args.directory).expanduser().absolute()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run needs a command after --")
    if not math.isfinite(args.cache_gb) or args.cache_gb <= 0:
        raise ValueError("--cache-gb must be finite and positive")
    cache = DiskCache(root / "range-cache", int(args.cache_gb * 1e9))
    try:
        source = Source(root, cache, args.offline)
        print("Remote weights: %.2f GB disk cache; HTTP on misses: %s."
              % (args.cache_gb, "OFF" if args.offline else "ON"), flush=True)
        print("A cold token can download tens of GB. This is a storage trade-off.",
              flush=True)
        with bridge(source) as path:
            env = dict(os.environ, K3_REMOTE_SOCKET=path)
            return subprocess.call(command, env=env)
    finally:
        print("range cache: %.3f GB fetched, %.3f GB cache hits, %.3f GB stored"
              % (cache.downloaded / 1e9, cache.hits / 1e9, cache.used / 1e9),
              file=sys.stderr)
        cache.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="download only headers, config and tokenizer")
    prep.add_argument("directory")
    prep.add_argument("--repo", default="moonshotai/Kimi-K3")
    prep.add_argument("--revision", default="main")
    pack = sub.add_parser("pack", help="download only the dense trunk (~109 GB)")
    pack.add_argument("directory")
    pack.add_argument("trunk_directory")
    pack.add_argument("--layers", type=int, default=93)
    runner = sub.add_parser("run", help="start a bounded cache and run a C command")
    runner.add_argument("--cache-gb", type=float, default=100.0,
                        help="decimal GB for disk blocks, not the C RAM cache")
    runner.add_argument("--offline", action="store_true",
                        help="fail on a cache miss instead of downloading")
    runner.add_argument("directory")
    runner.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if os.name != "posix":
            raise ValueError("remote tools need Linux, macOS, or WSL")
        if args.action == "prepare":
            prepare(args)
            return 0
        if args.action == "pack":
            import pack_trunk
            return pack_trunk.main([str(Path(args.directory).expanduser()),
                                    str(Path(args.trunk_directory).expanduser()),
                                    str(args.layers)])
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
