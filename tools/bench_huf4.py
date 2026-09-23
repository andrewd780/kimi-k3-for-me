#!/usr/bin/env python3
"""Independent encoder and CI benchmark driver for the four-stream prototype.

Default input is explicitly synthetic, NOT the measured K3 histogram. --raw can
benchmark an independently obtained, pinned BF16 range (at most 16 MiB).
The HF4B transport is benchmark-only and cannot be read as a K3 archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import platform
import random
import struct
import subprocess
import tempfile


def codes_for(raw):
    counts = Counter(raw)
    floor = 1
    while True:
        lengths = [0] * 256
        queue = [(max(weight, floor), (symbol,)) for symbol, weight in counts.items()]
        heapq.heapify(queue)
        while len(queue) > 1:
            a, sa = heapq.heappop(queue)
            b, sb = heapq.heappop(queue)
            for symbol in sa + sb:
                lengths[symbol] += 1
            heapq.heappush(queue, (a + b, sa + sb))
        if len(queue) == 1 and len(queue[0][1]) == 1:
            lengths[queue[0][1][0]] = 1
        if max(lengths) <= 12:
            break
        # Reweight and REBUILD a valid tree; never truncate code lengths. At a
        # floor >= max(counts), at most 256 equally weighted leaves need <=8 bits.
        floor *= 2
    code, previous, codes = 0, 0, {}
    for width, symbol in sorted((width, s) for s, width in enumerate(lengths) if width):
        code <<= width - previous
        codes[symbol] = code, width
        code += 1
        previous = width
    return lengths, codes


def bits(raw, codes):
    out, acc, held = bytearray(), 0, 0
    for value in raw:
        code, width = codes[value]
        acc = (acc << width) | code
        held += width
        while held >= 8:
            held -= 8
            out.append((acc >> held) & 255)
        acc &= (1 << held) - 1
    if held:
        out.append(acc << (8 - held))
    return bytes(out)


def encode(raw):
    high, low = raw[1::2], raw[::2]
    lengths, codes = codes_for(high)
    if not high:
        lengths[0] = 1
    one = bits(high, codes)
    lanes = [bits(high[j::4], codes) for j in range(4)]
    header = b"HF4B" + struct.pack("<6I", len(raw), len(one), *map(len, lanes)) + bytes(lengths)
    return header + one + b"".join(lanes) + low + raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--sample-k3", action="store_true",
                        help="4 MiB of SHA-verified HTTP ranges from the committed pinned sample manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(39487)
    sample_ids = []
    if args.raw and args.sample_k3:
        parser.error("choose --raw or --sample-k3")
    if args.sample_k3:
        from remote_model import read_range
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               "docs/measurements/lossless-samples.json").read_text())
        chunks = []
        for row in [r for r in manifest["samples"] if r["kind"] == "dense"][:4]:
            if row["bytes"] != 1 << 20:
                raise ValueError("expected four bounded 1 MiB ranges")
            url = ("https://huggingface.co/moonshotai/Kimi-K3/resolve/" +
                   manifest["revision"] + "/" + row["shard"])
            chunk, _ = read_range(url, row["offset"], row["bytes"])
            if hashlib.sha256(chunk).hexdigest() != row["sha256"]:
                raise ValueError("pinned sample hash mismatch")
            chunks.append(chunk)
            sample_ids.append({key: row[key] for key in ("shard", "tensor", "offset", "bytes", "sha256")})
        if len(chunks) != 4:
            raise ValueError("four dense samples required")
        raw = b"".join(chunks)
    elif args.raw:
        if not 1 <= args.raw.stat().st_size <= 16 << 20:
            parser.error("--raw must contain 1 byte to 16 MiB")
        raw = args.raw.read_bytes()
    else:
        n = 2 << 20
        raw = bytearray(2*n)
        raw[::2] = rng.randbytes(n)
        raw[1::2] = bytes(rng.choices(range(256), weights=[600, 200, 100, 50] + [0.1]*252, k=n))
        raw = bytes(raw)
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "input.hf4b"
        encoded = encode(raw)
        path.write_bytes(encoded)
        result = subprocess.run([str(args.binary.resolve()), str(path)], capture_output=True,
                                text=True, check=True, timeout=60)
    report = json.loads(result.stdout)
    report.update({"machine": platform.machine(), "system": platform.platform(),
                   "source": "pinned_K3_ranges" if args.sample_k3 else
                             ("user_supplied_range" if args.raw else "synthetic_heavy_tailed_bytes"),
                   "samples": sample_ids,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "scope": "kernel only; no full-model compression or inference speed claim"})
    sizes = struct.unpack_from("<6I", encoded, 4)
    report["four_stream_payload_ratio"] = (sum(sizes[2:]) + (len(raw)+1)//2) / len(raw)
    histogram = Counter(raw[1::2])
    report["high_byte_histogram"] = [histogram[i] for i in range(256)]
    report["four_streams_clear_1_GBps_all_runs"] = all(
        rate >= 1 for rate in report["arms"][1]["decoded_BF16_GBps_runs"])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
