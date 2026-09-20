#!/usr/bin/env python3
"""Independent FD4B encoder; ordered correctness and kernel-rate gates in CI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import tempfile

from remote_model import read_range
from trunk_dictionary_gate import MANIFEST, dense_ranges, high_histogram, verified_range


def encode(raw, dictionary):
    if (len(raw) > 16 << 20 or len(dictionary) != 15 or len(set(dictionary)) != 15
            or any(type(v) is not int or not 0 <= v <= 255 for v in dictionary)):
        raise ValueError("invalid raw length or dictionary")
    lookup = {v: i for i, v in enumerate(dictionary)}
    codes = [lookup.get(v, 15) for v in raw[1::2]]
    indexes = bytearray((len(codes) + 1) // 2)
    for i, code in enumerate(codes):
        indexes[i // 2] |= code << (4 * (i % 2))
    escapes = bytes(v for v in raw[1::2] if v not in lookup)
    return (b"FD4B" + struct.pack("<III", len(raw), len(escapes), 0) +
            bytes(dictionary) + b"\0" + indexes + raw[::2] + escapes)


def rate_gate(cases):
    if len(cases) != 9:
        return False
    for case in cases:
        arms = case.get("arms", [])
        if (case.get("byte_exact") is not True or len(arms) != 2
                or case.get("native") not in ("ssse3_pshufb", "neon_tbl")
                or arms[1].get("name") != case["native"]):
            return False
        for arm in arms:
            rates = arm.get("decoded_BF16_GBps_runs", [])
            if len(rates) != 3 or any(not math.isfinite(r) or r <= 0 for r in rates):
                return False
        if any(r < 3 for r in arms[1]["decoded_BF16_GBps_runs"]):
            return False
    return True


def sample_inputs(histogram_path):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise ValueError("range acquisition and execution are restricted to hosted CI")
    manifest = json.loads(MANIFEST.read_bytes())
    histogram = json.loads(histogram_path.read_bytes())
    if (not histogram["gate"]["continue_to_codec"]
            or histogram["manifest_sha256"] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest()):
        raise ValueError("a passing falsifier for this exact manifest is required")
    samples = []
    for row, measured in zip(dense_ranges(manifest), histogram["ranges"], strict=True):
        url = ("https://huggingface.co/moonshotai/Kimi-K3/resolve/" +
               manifest["revision"] + "/" + row["shard"])
        raw, _ = read_range(url, row["offset"], row["bytes"])
        verified_range(row, raw)
        if high_histogram(raw) != measured["histogram"]:
            raise ValueError("falsifier counts changed")
        samples.append((measured["sample"], raw))
    return samples, histogram["pooled"]["global_15"]["dictionary"]


def invoke(binary, raw, encoded, mode):
    with tempfile.TemporaryDirectory() as directory:
        raw_path, packed_path = Path(directory) / "raw", Path(directory) / "fd4b"
        raw_path.write_bytes(raw)
        packed_path.write_bytes(encoded)
        result = subprocess.run([str(binary.resolve()), str(packed_path), str(raw_path), mode],
                                capture_output=True, text=True, timeout=60)
    return result


def correctness_controls(binary, dictionary):
    # Independent golden byte layout: four high bytes and a final unpaired low byte.
    golden_raw = bytes.fromhex("e100e201e3ffe40ee5")
    golden = (b"FD4B" + struct.pack("<III", 9, 1, 0) + bytes(range(15)) +
              b"\0\x10\xef" + bytes.fromhex("e1e2e3e4e5ff"))
    if encode(golden_raw, list(range(15))) != golden:
        raise ValueError("independent golden layout failed")
    if invoke(binary, golden_raw, golden, "verify").returncode:
        raise ValueError("C golden decode failed")
    raw = bytes(range(256)) * 3 + b"\xff"
    encoded = encode(raw, dictionary)
    positive = invoke(binary, raw, encoded, "verify")
    if positive.returncode:
        raise ValueError("positive control failed: " + positive.stderr)
    corrupt_payload = bytearray(encoded)
    corrupt_payload[32 + (len(raw)//2 + 1)//2] ^= 1
    controls = {"truncated_header": (raw, encoded[:31]),
                "truncated_payload": (raw, encoded[:-1]),
                "excess_payload": (raw, encoded + b"\0"),
                "corrupt_low_byte": (raw, bytes(corrupt_payload)),
                "wrong_reference_byte": (bytes([raw[0] ^ 1]) + raw[1:], encoded)}
    for name, (reference, packed) in controls.items():
        result = invoke(binary, reference, packed, "verify")
        # Sanitizer faults use distinct exit codes in CI: cannot count as a clean rejection.
        if result.returncode != 1:
            raise ValueError(name + " did not fail the byte-exact gate cleanly: " + result.stderr)
    return {"golden_layout_pass": True, "odd_tail_pass": True,
            "negative_controls_rejected": list(controls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", choices=("correctness", "rate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--histogram", type=Path, default=Path("trunk-dictionary-histogram.json"),
                        help="passing gate-1 artifact from this CI workflow")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("experimental execution is restricted to hosted CI")
    samples, dictionary = sample_inputs(args.histogram)
    controls = correctness_controls(args.binary, dictionary) if args.mode == "correctness" else None
    pooled = b"".join(raw for _, raw in samples)
    cases = []
    for identity, raw in samples + [({"tensor": "pooled_eight_ranges"}, pooled)]:
        packed = encode(raw, dictionary)
        result = invoke(args.binary, raw, packed, "verify" if args.mode == "correctness" else "time")
        if result.returncode:
            raise ValueError("native reconstruction failed: " + result.stderr)
        row = json.loads(result.stdout)
        if row["native"] not in ("ssse3_pshufb", "neon_tbl"):
            raise ValueError("hosted SIMD target absent")
        row.update({"sample": identity, "sha256": hashlib.sha256(raw).hexdigest(),
                    "payload_ratio": (len(packed) - 32) / len(raw),
                    "framed_ratio": len(packed) / len(raw)})
        for arm in row["arms"]:
            runs = arm["decoded_BF16_GBps_runs"]
            arm.update(mean=statistics.mean(runs), median=statistics.median(runs), minimum=min(runs))
        cases.append(row)
    report = {"schema": "fixed-dictionary-v1", "mode": args.mode, "index_bits": 4,
              "dictionary": dictionary, "cases": cases, "controls": controls,
              "execution": {"machine": platform.machine(), "system": platform.platform(),
                            "cpu": platform.processor(), "cpu_count": os.cpu_count(),
                            "compiler_flags": os.environ.get("CODEC_FLAGS"),
                            "head_sha": os.environ.get("RESEARCH_COMMIT"),
                            "checkout_sha": os.environ.get("GITHUB_SHA"),
                            "run_id": os.environ.get("GITHUB_RUN_ID")},
              "scope": "repeated 1 MiB and pooled 8 MiB buffers, cache-friendly single-thread "
                       "kernel; all escape handling and BF16 assembly timed, parse/allocation/"
                       "I/O/encoding excluded; no competing model compute, storage or speedup claim"}
    if args.mode == "rate":
        report["rate_gate"] = {"status": "PASS" if rate_gate(cases) else "STOP",
                               "rule": "SIMD >=3 reconstructed BF16 GB/s in every one of 27 runs; "
                                       "scalar is a reference, not the proposed SIMD decoder",
                               "target_4_GBps_all_runs": all(
                                   r >= 4 for c in cases for r in c["arms"][1]["decoded_BF16_GBps_runs"])}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("CODEC_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
