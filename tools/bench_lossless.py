#!/usr/bin/env python3
"""Bounded stratified real-weight samples; no full-checkpoint compression claim."""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
import math
import platform
import re
from pathlib import Path
import statistics
import struct
import time
import zlib

from k3_zstd import Zstd
from offline_model import unshuffle
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
                result = unshuffle(result)
            elapsed = time.perf_counter() - start
            if result != raw:
                raise ValueError("codec roundtrip mismatch")
            runs.append(len(raw) / max(elapsed, 1e-9) / 1e6)
        out[name] = {"bytes": len(packed), "ratio": len(packed) / len(raw),
                     "decode_MBps_runs": runs, "decode_MBps_median": statistics.median(runs)}
    return out


def sample_plan(header, header_bytes, shard, experts_per_shard=0):
    """Pair packed samples with the entire E8M0 scale tensor of the same matrix."""
    experts = sorted(k for k in header if ".block_sparse_moe.experts." in k
                     and k.endswith(".weight_packed"))
    if experts_per_shard:
        groups = collections.defaultdict(dict)
        for name in experts:
            prefix, matrix, _ = name.rsplit(".", 2)
            groups[prefix][matrix] = name
        complete = sorted((prefix for prefix, matrices in groups.items()
                           if set(matrices) == {"w1", "w2", "w3"}),
                          key=lambda prefix: tuple(map(int, re.findall(r"[0-9]+", prefix))))
        if len(complete) < experts_per_shard:
            raise ValueError("not enough complete experts in " + shard)
        count = experts_per_shard
        indices = [i * (len(complete) - 1) // max(1, count - 1) for i in range(count)]
        sampled = [groups[complete[i]][matrix] for i in indices for matrix in ("w1", "w2", "w3")]
    else:
        sampled = ([experts[i] for i in sorted({0, len(experts) // 2, len(experts) - 1})]
                   if experts else [])
    dense = sorted(k for k, entry in header.items() if k != "__metadata__"
                   and ".block_sparse_moe.experts." not in k and entry["dtype"] == "BF16"
                   and entry["data_offsets"][1] - entry["data_offsets"][0] >= (1 << 20))
    pairs = []
    for name in sampled:
        scale = name.removesuffix("weight_packed") + "weight_scale"
        pk, sc = header[name], header.get(scale)
        if (sc is None or pk["dtype"] != "U8" or sc["dtype"] != "U8"
                or len(pk["shape"]) != 2 or len(sc["shape"]) != 2
                or pk["shape"][0] != sc["shape"][0]
                or pk["shape"][1] != 16 * sc["shape"][1]):
            raise ValueError("missing or incompatible E8M0 scale plane for " + name)
        pairs.extend(((name, "expert", scale), (scale, "scale", name)))
    if dense:
        pairs.append((dense[len(dense) // 2], "dense", None))
    result = []
    for name, kind, pair in pairs:
        first, last = header[name]["data_offsets"]
        length = last - first if kind == "scale" else min(1 << 20, last - first)
        if not 0 < length <= MAX_HEADER:
            raise ValueError("invalid sample length")
        offset = first + ((last - first - length) // 2 // 2) * 2
        result.append({"shard": shard, "tensor": name, "kind": kind,
                       "paired_tensor": pair, "dtype": header[name]["dtype"],
                       "shape": header[name]["shape"], "offset": header_bytes + offset,
                       "bytes": length, "tensor_bytes": last - first})
    return result


def summarize(results, counts):
    ratios, summary = {}, {}
    for kind in ("expert", "scale", "dense"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            raise ValueError("no samples for " + kind)
        n = sum(r["bytes"] for r in rows)
        ratios[kind] = {codec: sum(r["codecs"][codec]["bytes"] for r in rows) / n
                        for codec in rows[0]["codecs"]}
        histogram = counts[kind]
        summary[kind] = {
            "tensors": len(rows), "bytes": n,
            "weighted_tensor_entropy_bits_per_byte":
                sum(r["bytes"] * r["byte_entropy_bits"] for r in rows) / n,
            "pooled_entropy_bits_per_byte":
                -sum((v / n) * math.log2(v / n) for v in histogram.values()),
            "histogram": [histogram[i] for i in range(256)],
        }
    return ratios, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="lossless-samples.json")
    parser.add_argument("--revision", default="f831ab66814297da540d832a5235f8e904f29d06")
    parser.add_argument("--experts-per-shard", type=int, default=0,
                        help="0: legacy three matrices; 32: ~1.07 GB of matched samples")
    parser.add_argument("--max-sample-bytes", type=int, default=128 << 20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.experts_per_shard <= 32 or not 1 <= args.workers <= 8:
        parser.error("experts-per-shard must be 0..32; workers must be 1..8")
    if not 0 < args.max_sample_bytes <= 1125_000_000:
        parser.error("sample byte limit must be positive and at most 1.125 GB")
    repo = "moonshotai/Kimi-K3"
    info = decode(read_small("https://huggingface.co/api/models/%s/revision/%s"
                             % (repo, args.revision)))
    revision = info["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("checkpoint revision must resolve to an immutable commit")
    names = sorted(x["rfilename"] for x in info["siblings"]
                   if x["rfilename"].endswith(".safetensors"))
    if not names:
        raise ValueError("model has no safetensors shards")
    chosen = sorted({min(i, len(names) - 1) for i in (1, 12, 24, 36, 48, 60, 72, 92)})
    plan, sources = [], {}
    header_bytes = 0
    for index in chosen:
        shard = names[index]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", shard):
            raise ValueError("unsafe shard name")
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
        prefix, total = read_range(url, 0, 8)
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER:
            raise ValueError("header too large")
        raw_header, _ = read_range(url, 8, size, total)
        header = validate_header(prefix + raw_header, total)
        header_bytes += size + 8
        sources[shard] = {"url": url, "bytes": total, "header_sha256": digest(prefix + raw_header)}
        plan.extend(sample_plan(header, size + 8, shard, args.experts_per_shard))
    planned = sum(r["bytes"] for r in plan)
    if planned > args.max_sample_bytes:
        raise ValueError("planned %d sample bytes exceeds limit %d" % (planned, args.max_sample_bytes))
    print("PLAN " + json.dumps({"tensors": len(plan), "sample_bytes": planned,
                               "header_bytes": header_bytes}), flush=True)
    zstd = Zstd()
    results, counts = [], collections.defaultdict(collections.Counter)

    def fetch(row):
        source = sources[row["shard"]]
        return read_range(source["url"], row["offset"], row["bytes"], source["bytes"])[0]

    # Bounded read-ahead: at most `workers` samples resident, never a whole shard.
    # Decode timings run serially after each batch's downloads have completed.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(plan), args.workers):
            batch = plan[start:start + args.workers]
            fetched = list(pool.map(fetch, batch))
            for entry, raw in zip(batch, fetched):
                row = {**entry, "sha256": digest(raw), "byte_entropy_bits": entropy(raw),
                       "codecs": measure(raw, zstd)}
                counts[row["kind"]].update(raw)
                results.append(row)
                print(json.dumps(row), flush=True)
    ratios, summary = summarize(results, counts)
    report = {"revision": revision, "zstd": zstd.lib.ZSTD_versionString().decode(),
              "platform": platform.platform(), "sources": sources,
              "sample_bytes": planned, "header_bytes": header_bytes,
              "experts_per_shard": args.experts_per_shard,
              "samples": results, "summary_ratios": ratios, "summary": summary,
              "limits": "Stratified samples, not a census or a model speed test. "
                        "Packed samples cover a central 1 MiB; paired scales cover the full "
                        "same matrix. Pooled byte entropy is a zero-order coding bound, "
                        "not a compression ratio. No deployment codec is added."}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps({"sample_bytes": planned, "summary": summary,
                                   "summary_ratios": ratios}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
