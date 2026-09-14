#!/usr/bin/env python3
"""Pack local weights losslessly; the C engine reads .k3z directly without Python.

No network requests are made by this tool. Conversion keeps the source intact and
publishes only fully written files. See docs/OFFLINE_STORAGE.md for costs and limits.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct

from k3_zstd import Zstd

HEADER = struct.Struct("<8sQII16sQ")
ENTRY = struct.Struct("<QII")
PREFIX = struct.Struct("<16sQII")
MAGIC = b"K3ZSTD1\0"
MAP_MAGIC = b"K3ZMAP1\0"
MAP_ENTRY = struct.Struct("<QQQII")  # logical end, physical offset/size, kind, reserved
RAW_EXTENT = 2
BLOCK = 1 << 20
MAX_BLOCK = 8 << 20
MAX_COUNT = 1 << 22


def index_hash(data, value=14695981039346656037):
    """FNV-1a detects accidental index damage; this is not authentication."""
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def block_count(size, block):
    if (not 0 <= size <= (1 << 63) - 1 or not (64 << 10) <= block <= MAX_BLOCK
            or block & (block - 1)):
        raise ValueError("invalid file size or block size (power of two, 64 KiB..8 MiB)")
    count = (size + block - 1) // block
    if count > MAX_COUNT:
        raise ValueError("too many blocks; choose a larger --block-kib")
    return count


def unshuffle(data):
    result = bytearray(len(data))
    half = (len(data) + 1) // 2
    result[0::2], result[1::2] = data[:half], data[half:]
    return bytes(result)


class Reader(io.RawIOBase):
    """Seekable logical file, with at most one decoded block cached in memory.

    Like an ordinary Python file, an instance has a shared cursor. Use separate
    Readers per thread. The inference engine has its own thread-safe C reader.
    """
    def __init__(self, path):
        super().__init__()
        self.file = open(path, "rb")
        try:
            self.codec = Zstd()
            h = self.file.read(HEADER.size)
            if len(h) != HEADER.size:
                raise ValueError("short archive header")
            magic, self.size, self.block, count, self.id, reserved = HEADER.unpack(h)
            self.mapped = magic == MAP_MAGIC
            # Validate common size/block limits without imposing the fixed-block count
            # on a variable-extent archive. A raw extent may span many blocks.
            block_count(0, self.block)
            if (self.size > (1 << 63) - 1 or count > MAX_COUNT or
                    (not self.mapped and (magic != MAGIC or reserved or
                     count != block_count(self.size, self.block)))):
                raise ValueError("invalid archive header")
            physical = os.fstat(self.file.fileno()).st_size
            entry_type = MAP_ENTRY if self.mapped else ENTRY
            cursor = HEADER.size + count * entry_type.size
            if cursor > physical:
                raise ValueError("truncated block index")
            table = self.file.read(count * entry_type.size)
            if len(table) != count * entry_type.size:
                raise ValueError("short block index")
            if self.mapped and index_hash(table, index_hash(h[:40])) != reserved:
                raise ValueError("archive index checksum mismatch")
            self.entries, self.ends = [], []
            bound = self.codec.lib.ZSTD_compressBound(self.block + PREFIX.size)
            end = 0
            for entry in entry_type.iter_unpack(table):
                if self.mapped:
                    last, offset, size, flags, zero = entry
                    length = last - end
                    if (zero or not 0 < length or last > self.size or
                            flags not in (0, RAW_EXTENT) or
                            (flags == RAW_EXTENT and size != length) or
                            (flags == 0 and length > self.block)):
                        raise ValueError("invalid extent index")
                    end = last
                    self.ends.append(end)
                else:
                    offset, size, flags = entry
                if (offset != cursor or size > physical - cursor or
                        (not self.mapped and flags not in (0, 1)) or
                        (flags != RAW_EXTENT and
                         (not 9 <= size <= bound or flags not in (0, 1)))):
                    raise ValueError("invalid block index")
                self.entries.append((offset, size, flags))
                cursor += size
            if self.mapped and end != self.size:
                raise ValueError("extent index does not cover the logical file")
            if cursor != physical:
                raise ValueError("trailing archive bytes")
            self.pos = 0
            self.cached_index, self.cached = -1, b""
        except BaseException:
            self.file.close()
            raise

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        self._checkClosed()
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError("invalid whence")
        pos = offset + (self.pos if whence == os.SEEK_CUR else
                        self.size if whence == os.SEEK_END else 0)
        if pos < 0:
            raise ValueError("negative seek")
        self.pos = pos
        return pos

    def _block(self, index):
        if index == self.cached_index:
            return self.cached
        offset, packed_size, flags = self.entries[index]
        start = self.ends[index - 1] if self.mapped and index else 0
        size = (self.ends[index] - start if self.mapped else
                min(self.block, self.size - index * self.block))
        self.file.seek(offset)
        packed = self.file.read(packed_size)
        if len(packed) != packed_size:
            raise ValueError("short block read")
        raw = self.codec.decompress(packed, size + PREFIX.size)
        if PREFIX.unpack_from(raw) != (self.id, index, flags, 0):
            raise ValueError("block identity, position or transform mismatch")
        payload = raw[PREFIX.size:]
        self.cached = unshuffle(payload) if flags else payload
        self.cached_index = index
        return self.cached

    def read(self, size=-1):
        self._checkClosed()
        left = max(0, self.size - self.pos)
        size = left if size is None or size < 0 else min(size, left)
        result = bytearray()
        while len(result) < size:
            if self.mapped:
                index = bisect.bisect_right(self.ends, self.pos)
                start = self.ends[index - 1] if index else 0
                within = self.pos - start
                offset, _, kind = self.entries[index]
                if kind == RAW_EXTENT:
                    # Never decode or allocate the entire raw extent (possibly GB).
                    take = min(size - len(result), self.ends[index] - self.pos)
                    self.file.seek(offset + within)
                    raw = self.file.read(take)
                    if len(raw) != take:
                        raise ValueError("short raw extent read")
                    result.extend(raw)
                    self.pos += take
                    continue
            else:
                index, within = divmod(self.pos, self.block)
            raw = self._block(index)
            take = min(size - len(result), len(raw) - within)
            result.extend(raw[within:within + take])
            self.pos += take
        return bytes(result)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self.file.close()
            self.cached = b""
        super().close()


def open_weight(path):
    return Reader(path) if str(path).endswith(".k3z") else open(path, "rb")


def _unique_object(pairs):
    obj = {}
    for name, value in pairs:
        if name in obj:
            raise ValueError("duplicate safetensors JSON key: " + name)
        obj[name] = value
    return obj


def scale_plan(source, block=BLOCK):
    """Partition the *whole* logical file, cutting exactly at scale boundaries.

    Only paired U8 group-32 expert scales are selected. Unknown tensors and gaps
    remain raw. The safetensors header stays byte-identical, including offsets.
    The plan needs only a bounded header, never weight data.
    """
    block_count(0, block)
    source = Path(source)
    size = source.stat().st_size
    if not 8 <= size <= (1 << 63) - 1:
        raise ValueError("invalid safetensors file size")
    with source.open("rb") as src:
        prefix = src.read(8)
        if len(prefix) != 8:
            raise ValueError("short safetensors header")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= min(64 << 20, size - 8):
            raise ValueError("invalid safetensors header length")
        raw_header = src.read(length)
        if len(raw_header) != length:
            raise ValueError("short safetensors header")
        header = json.loads(raw_header, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    base = 8 + length
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError("invalid tensor metadata")
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or
                not 0 <= offsets[0] <= offsets[1] <= size - base):
            raise ValueError("invalid tensor offsets: " + name)
        if offsets[0] != offsets[1]:
            spans.append((offsets[0], offsets[1]))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping tensor ranges")
    selected = []
    for name, tensor in header.items():
        if (".block_sparse_moe.experts." not in name or
                not any(name.endswith(f".{w}.weight_scale") for w in ("w1", "w2", "w3"))):
            continue
        packed = header.get(name.removesuffix("weight_scale") + "weight_packed")
        shapes = [x.get("shape") for x in (tensor, packed) if isinstance(x, dict)]
        if (len(shapes) != 2 or any(not isinstance(s, list) or len(s) != 2 or
                any(type(v) is not int or v <= 0 for v in s) for s in shapes) or
                tensor.get("dtype") != "U8" or packed.get("dtype") != "U8" or
                shapes[0][0] != shapes[1][0] or shapes[0][1] * 16 != shapes[1][1] or
                any(math.prod(x["shape"]) != x["data_offsets"][1] - x["data_offsets"][0]
                    for x in (tensor, packed))):
            raise ValueError("missing or incompatible paired E8M0 tensor: " + name)
        lo, hi = tensor["data_offsets"]
        selected.append((base + lo, base + hi))
    plan, cursor = [], 0
    for lo, hi in sorted(selected):
        if cursor < lo:
            plan.append((cursor, lo, RAW_EXTENT))
        while lo < hi:
            end = min(hi, lo + block)
            plan.append((lo, end, 0))
            lo = end
        cursor = hi
    if cursor < size:
        plan.append((cursor, size, RAW_EXTENT))
    if len(plan) > MAX_COUNT:
        raise ValueError("too many extents; choose a larger --block-kib")
    return size, plan


def pack_scales(source, destination, *, block=BLOCK, level=3, limit=None):
    """Raw positioned reads for packed bytes; bounded Zstd frames for scales only.

    Raw payload has the same integrity properties as an ordinary safetensors
    file. A full logical SHA-256 is recorded for explicit whole-archive verify.
    Compressed extents have mandatory frame checksums and index/identity binding.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("destination must be new: " + str(destination))
    before = source.stat()
    size, plan = scale_plan(source, block)
    cursor = HEADER.size + len(plan) * MAP_ENTRY.size
    if limit is not None and cursor > limit:
        raise OSError("output limit cannot hold the archive index")
    codec, archive_id = Zstd(), os.urandom(16)
    part = destination.with_name(destination.name + ".part")
    entries, sha = [], hashlib.sha256()
    scale_bytes = scale_stored = raw_bytes = compressed_scale_bytes = 0
    header = HEADER.pack(MAP_MAGIC, size, block, len(plan), archive_id, 0)
    with part.open("xb") as dst:
        try:
            dst.write(header)
            dst.seek(cursor)
            with source.open("rb") as src:
                for index, (lo, hi, kind) in enumerate(plan):
                    offset = cursor
                    if kind == RAW_EXTENT:
                        left = hi - lo
                        while left:
                            chunk = src.read(min(block, left))
                            if not chunk:
                                raise ValueError("source truncated during packing")
                            if limit is not None and cursor + len(chunk) > limit:
                                raise OSError("raw output exceeds --max-output-gb")
                            sha.update(chunk)
                            dst.write(chunk)
                            cursor += len(chunk)
                            left -= len(chunk)
                        raw_bytes += hi - lo
                    else:
                        raw = src.read(hi - lo)
                        if len(raw) != hi - lo:
                            raise ValueError("source truncated during packing")
                        sha.update(raw)
                        plain = PREFIX.pack(archive_id, index, 0, 0) + raw
                        packed = codec.compress(plain, level)
                        if codec.decompress(packed, len(plain)) != plain:
                            raise ValueError("lossless roundtrip verification failed")
                        if len(packed) >= len(raw):
                            packed, kind = raw, RAW_EXTENT
                            raw_bytes += len(raw)
                        else:
                            compressed_scale_bytes += len(raw)
                        if limit is not None and cursor + len(packed) > limit:
                            raise OSError("scale output exceeds --max-output-gb")
                        dst.write(packed)
                        cursor += len(packed)
                        scale_bytes += len(raw)
                        scale_stored += len(packed)
                    entries.append((hi, offset, cursor - offset, kind, 0))
                if src.read(1):
                    raise ValueError("source grew during packing")
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("source changed during packing")
            dst.seek(HEADER.size)
            checksum = index_hash(header[:40])
            for entry in entries:
                encoded = MAP_ENTRY.pack(*entry)
                checksum = index_hash(encoded, checksum)
                dst.write(encoded)
            dst.seek(40)
            dst.write(struct.pack("<Q", checksum))
            dst.flush()
            os.fsync(dst.fileno())
        except BaseException:
            dst.close()
            part.unlink()
            raise
    try:
        os.link(part, destination)
    finally:
        part.unlink()
    return {"file": destination.name, "format": "K3ZMAP1", "policy": "scales",
            "source_bytes": size, "stored_bytes": cursor, "ratio": cursor / size,
            "sha256": sha.hexdigest(), "block_bytes": block,
            "extents": len(entries), "raw_payload_bytes": raw_bytes,
            "scale_bytes": scale_bytes, "scale_stored_bytes": scale_stored,
            "compressed_scale_bytes": compressed_scale_bytes,
            "index_bytes": HEADER.size + len(entries) * MAP_ENTRY.size}


def pack_file(source, destination, *, block=BLOCK, level=3, shuffle="auto", limit=None,
              policy="all"):
    """Bounded-memory conversion; limit includes the file header, index and frames."""
    if policy == "scales":
        return pack_scales(source, destination, block=block, level=level, limit=limit)
    if policy != "all":
        raise ValueError("invalid compression policy")
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("destination must be new: " + str(destination))
    if shuffle not in ("off", "auto", "on"):
        raise ValueError("invalid shuffle mode")
    codec = Zstd()
    before = source.stat()
    size = before.st_size
    count = block_count(size, block)
    cursor = HEADER.size + count * ENTRY.size
    if limit is not None and cursor > limit:
        raise OSError("output limit cannot hold the archive index")
    archive_id = os.urandom(16)
    part = destination.with_name(destination.name + ".part")
    entries, sha = [], hashlib.sha256()
    # Exclusive creation matters: cleanup must never remove someone else's .part.
    with part.open("xb") as dst:
        try:
            dst.write(HEADER.pack(MAGIC, size, block, count, archive_id, 0))
            dst.seek(cursor)
            with source.open("rb") as src:
                for index in range(count):
                    want = min(block, size - index * block)
                    raw = src.read(want)
                    if len(raw) != want:
                        raise ValueError("source changed or was truncated during packing")
                    sha.update(raw)
                    flags = 0
                    plain = PREFIX.pack(archive_id, index, 0, 0) + raw
                    # Only "auto" needs both candidates to compare. Under "on" the
                    # unshuffled one is discarded unconditionally, so compressing it
                    # would burn a second full pass over every byte for nothing.
                    packed = None if shuffle == "on" else codec.compress(plain, level)
                    if shuffle != "off":
                        shuffled = (PREFIX.pack(archive_id, index, 1, 0)
                                    + raw[0::2] + raw[1::2])
                        candidate = codec.compress(shuffled, level)
                        # Avoid paying unshuffle CPU for negligible expert-byte savings.
                        if packed is None or len(candidate) < len(packed) * 0.98:
                            packed, plain, flags = candidate, shuffled, 1
                    if codec.decompress(packed, len(plain)) != plain:
                        raise ValueError("lossless roundtrip verification failed")
                    if limit is not None and cursor + len(packed) > limit:
                        raise OSError("compressed output exceeds --max-output-gb; "
                                      "these weights do not compress enough")
                    dst.write(packed)
                    entries.append((cursor, len(packed), flags))
                    cursor += len(packed)
                if src.read(1):
                    raise ValueError("source grew during packing")
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("source changed during packing")
            dst.seek(HEADER.size)
            for entry in entries:
                dst.write(ENTRY.pack(*entry))
            dst.flush()
            os.fsync(dst.fileno())
        except BaseException:
            dst.close()
            part.unlink()
            raise
    try:
        # Same-directory hard link atomically publishes without overwriting a file
        # created by another process while compression was in progress.
        os.link(part, destination)
    finally:
        part.unlink()
    return {"file": destination.name, "source_bytes": size, "stored_bytes": cursor,
            "ratio": cursor / size if size else None, "sha256": sha.hexdigest(),
            "block_bytes": block, "shuffled_blocks": sum(e[2] for e in entries)}


def pack_model(source, destination, **options):
    source, destination = Path(source), Path(destination)
    if (source / ".k3-remote").exists() or (source / ".k3-incomplete").exists():
        raise ValueError("conversion needs complete local shards, not header-only files")
    shards = sorted(source.glob("*.safetensors"))
    if not shards or not (source / "config.json").is_file():
        raise ValueError("source needs config.json and complete .safetensors shards")
    # When an HF index exists, do not mistake a partially downloaded model for a
    # complete one. Fixtures and single-shard checkpoints need not have an index.
    index = source / "model.safetensors.index.json"
    if index.exists():
        expected = set(json.loads(index.read_text())["weight_map"].values())
        if expected != {p.name for p in shards}:
            raise ValueError("shards do not match model.safetensors.index.json")
    limit = options.pop("limit", None)
    marker_text = "Conversion has not finished. Do not run this directory.\n"
    if limit is not None and len(marker_text) > limit:
        raise OSError("output limit cannot hold the incomplete marker")
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".k3-incomplete"
    marker.write_text(marker_text)
    used = marker.stat().st_size
    report = {"format": "K3ZMAP1" if options.get("policy") == "scales" else "K3ZSTD1",
              "files": []}
    # Preserve tokenizer/config metadata, never carry over remote manifests or outputs.
    metadata = [p for p in source.iterdir() if p.is_file()
                and p.suffix in (".json", ".model", ".txt")
                and p.name not in ("remote.json", "trunk.json", "offline.json")]
    for path in sorted(metadata):
        if path.name == "model.safetensors.index.json":
            # Copied verbatim this would publish a weight_map naming .safetensors files
            # the directory does not contain: a checkpoint that looks complete to any
            # external tool and is not. The check above already proved the values are
            # exactly the shard names, and each is written as <name>.k3z below.
            remapped = json.loads(path.read_text())
            remapped["weight_map"] = {k: v + ".k3z"
                                      for k, v in remapped["weight_map"].items()}
            blob = (json.dumps(remapped, indent=2) + "\n").encode()
            if limit is not None and used + len(blob) > limit:
                raise OSError("metadata exceeds --max-output-gb")
            (destination / path.name).write_bytes(blob)
            used += len(blob)
            continue
        n = path.stat().st_size
        if limit is not None and used + n > limit:
            raise OSError("metadata exceeds --max-output-gb")
        with path.open("rb") as src, (destination / path.name).open("xb") as dst:
            left = n
            while left:
                chunk = src.read(min(BLOCK, left))
                if not chunk:
                    raise ValueError("metadata changed during packing")
                dst.write(chunk)
                left -= len(chunk)
            if src.read(1):
                raise ValueError("metadata grew during packing")
        used += n
    for path in shards:
        item = pack_file(path, destination / (path.name + ".k3z"),
                         limit=None if limit is None else limit - used, **options)
        used += item["stored_bytes"]
        report["files"].append(item)
        print(json.dumps(item), flush=True)
    report["stored_bytes_without_report"] = used - marker.stat().st_size
    raw_report = (json.dumps(report, indent=2) + "\n").encode()
    if limit is not None and used + len(raw_report) > limit:
        raise OSError("conversion report exceeds --max-output-gb")
    (destination / "offline.json").write_bytes(raw_report)
    marker.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("file", "model"):
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("destination")
        p.add_argument("--block-kib", type=int, default=1024)
        p.add_argument("--level", type=int, default=3)
        p.add_argument("--shuffle", choices=("off", "auto", "on"), default="auto")
        p.add_argument("--policy", choices=("all", "scales"), default="all",
                       help="scales: compress only expert E8M0 tensors; copy all other "
                            "bytes raw (safetensors input only; ignores --shuffle)")
        p.add_argument("--max-output-gb", type=float,
                       help="stop before exceeding this output cap (decimal GB)")
    p = sub.add_parser("verify", help="decode every block and print the original SHA-256")
    p.add_argument("source")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            sha = hashlib.sha256()
            with Reader(args.source) as src:
                while data := src.read(BLOCK):
                    sha.update(data)
            print(sha.hexdigest())
        else:
            cap = args.max_output_gb
            if cap is not None and (not math.isfinite(cap) or cap <= 0):
                raise ValueError("--max-output-gb must be finite and positive")
            pack = pack_file if args.command == "file" else pack_model
            result = pack(args.source, args.destination, block=args.block_kib * 1024,
                          level=args.level, shuffle=args.shuffle,
                          policy=args.policy,
                          limit=None if cap is None else int(cap * 1e9))
            if args.command == "file":
                print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, "offline_model: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
