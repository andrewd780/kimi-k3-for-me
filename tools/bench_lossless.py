#!/usr/bin/env python3
"""Bounded stratified real-weight samples; no full-checkpoint compression claim."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import struct
import time
import zlib

from k3_zstd import Zstd
from remote_model import MAX_HEADER, decode, digest, read_range, read_small, validate_header


def entropy(data):
    counts = collections.Counter(data)
    return -sum((n / len(data)) * math.log2(n / len(data)) for n in counts.values())


def measure(raw, zstd):
    out = {}
    variants = {
        "zstd3": (lambda x: zstd.compress(x, 3), lambda x: zstd.decompress(x, len(raw)), raw),
        "zlib1": (lambda x: zlib.compress(x, 1), zlib.decompress, raw),
        "shuffle2_zstd3": (lambda x: zstd.compress(x, 3),
                          lambda x: zstd.decompress(x, len(raw)),
                          raw[0::2] + raw[1::2]),
    }
    for name, (encode, restore, transformed) in variants.items():
        packed = encode(transformed)
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            result = restore(packed)
            if name == "shuffle2_zstd3":
                original = bytearray(len(raw))
                half = (len(raw) + 1) // 2
                original[0::2], original[1::2] = result[:half], result[half:]
                result = bytes(original)
            elapsed = time.perf_counter() - start
            if result != raw:
                raise ValueError("codec roundtrip mismatch")
            runs.append(len(raw) / max(elapsed, 1e-9) / 1e6)
        out[name] = {"bytes": len(packed), "ratio": len(packed) / len(raw),
                     "decode_MBps_runs": runs, "decode_MBps_median": statistics.median(runs)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="lossless-samples.json")
    parser.add_argument("--revision", default="f831ab66814297da540d832a5235f8e904f29d06")
    args = parser.parse_args()
    repo = "moonshotai/Kimi-K3"
    info = decode(read_small("https://huggingface.co/api/models/%s/revision/%s"
                             % (repo, args.revision)))
    revision = info["sha"]
    names = sorted(x["rfilename"] for x in info["siblings"]
                   if x["rfilename"].endswith(".safetensors"))
    if not names:
        raise ValueError("model has no safetensors shards")
    chosen = sorted({min(i, len(names) - 1) for i in (1, 12, 24, 36, 48, 60, 72, 92)})
    zstd = Zstd()
    results = []
    for index in chosen:
        shard = names[index]
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
        prefix, total = read_range(url, 0, 8)
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER:
            raise ValueError("header too large")
        raw_header, _ = read_range(url, 8, size, total)
        header = validate_header(prefix + raw_header, total)
        experts = sorted(k for k in header if ".block_sparse_moe.experts." in k
                         and k.endswith(".weight_packed"))
        dense = sorted(k for k, v in header.items() if k != "__metadata__"
                       and ".block_sparse_moe.experts." not in k and v["dtype"] == "BF16"
                       and v["data_offsets"][1] - v["data_offsets"][0] >= (1 << 20))
        sampled = ([experts[i] for i in sorted({0, len(experts) // 2, len(experts) - 1})]
                   if experts else [])
        if dense:
            sampled.append(dense[len(dense) // 2])
        for name in sampled:
            first, last = header[name]["data_offsets"]
            length = min(1 << 20, last - first)
            offset = first + ((last - first - length) // 2 // 2) * 2
            raw, _ = read_range(url, size + 8 + offset, length, total)
            row = {"shard": shard, "tensor": name,
                   "kind": "expert" if name in experts else "dense",
                   "offset": size + 8 + offset, "bytes": len(raw), "sha256": digest(raw),
                   "byte_entropy_bits": entropy(raw), "codecs": measure(raw, zstd)}
            results.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for kind in ("expert", "dense"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            raise ValueError("no samples for " + kind)
        n = sum(r["bytes"] for r in rows)
        summary[kind] = {codec: sum(r["codecs"][codec]["bytes"] for r in rows) / n
                         for codec in rows[0]["codecs"]}
    report = {"revision": revision, "zstd": zstd.lib.ZSTD_versionString().decode(),
              "samples": results, "summary_ratios": summary,
              "limits": "Stratified tensor samples, not a census or a full-model speed test; "
                        "scales and metadata are not represented by the expert samples."}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
