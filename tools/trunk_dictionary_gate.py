#!/usr/bin/env python3
"""Gate 1 only: exact high-byte histograms for eight pinned BF16 dense ranges.

The proposed plane is raw[1::2], matching bench_huf4: sign plus the upper seven
exponent bits. The low byte retains the last exponent bit and seven fraction
bits. This is deliberately NOT a histogram of the mathematical 8-bit exponent.
No encoder, decoder or timing arm runs before this falsifier passes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re

from remote_model import read_range

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/measurements/lossless-samples.json"
COVERAGE = (("90", 9000), ("99", 9900), ("99.9", 9990), ("99.99", 9999))


def dense_ranges(manifest):
    if not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"]):
        raise ValueError("checkpoint revision must be immutable")
    rows = [r for r in manifest["samples"] if r["kind"] == "dense"]
    if len(rows) != 8:
        raise ValueError("exactly eight dense ranges required")
    seen = set()
    for row in rows:
        identity = row["shard"], row["offset"]
        if (not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{6}\.safetensors", row["shard"])
                or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
                or type(row["offset"]) is not int or row["offset"] < 0
                or row["offset"] % 2 or row["bytes"] != 1 << 20
                or identity in seen or ".experts." in row["tensor"]):
            raise ValueError("invalid or duplicate pinned dense range")
        seen.add(identity)
    return rows


def verified_range(row, raw):
    if len(raw) != row["bytes"]:
        raise ValueError("pinned sample length mismatch")
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("pinned sample hash mismatch")
    return raw


def high_histogram(raw):
    if not raw or len(raw) % 2:
        raise ValueError("expected complete, nonempty BF16 values")
    counts = Counter(raw[1::2])
    return [counts[i] for i in range(256)]


def rank(histogram):
    if (len(histogram) != 256 or any(type(n) is not int or n < 0 for n in histogram)
            or not sum(histogram)):
        raise ValueError("invalid histogram")
    # Deterministic ties: ascending byte value. Never select entries per test row
    # when evaluating the proposed single pooled dictionary.
    return sorted((v for v in range(256) if histogram[v]),
                  key=lambda v: (-histogram[v], v))


def dictionary_stats(histogram, dictionary):
    n = sum(histogram)
    hits = sum(histogram[v] for v in dictionary)
    escaped = n - hits
    return {"dictionary": dictionary, "covered_values": hits,
            "coverage": hits / n, "escape_values": escaped, "p_escape": escaped / n,
            "bits_per_bf16": 12 + 8 * escaped / n,
            "payload_ratio": 0.75 + 0.5 * escaped / n,
            "saturated_stream_decode_GBps_at_B_3": 3 / (0.75 + 0.5 * escaped / n),
            "passes_99_percent": hits * 100 >= n * 99}


def describe(histogram, dictionary):
    ordered = rank(histogram)
    n, support = sum(histogram), {}
    for label, basis_points in COVERAGE:
        cumulative = 0
        for k, value in enumerate(ordered, 1):
            cumulative += histogram[value]
            if cumulative * 10000 >= n * basis_points:
                support[label] = k
                break
    return {"bf16_values": n, "distinct_values": len(ordered),
            "histogram": histogram, "values_for_coverage_percent": support,
            "best_local_15": dictionary_stats(histogram, ordered[:15]),
            "global_15": dictionary_stats(histogram, dictionary)}


def analyze(entries):
    if len(entries) != 8:
        raise ValueError("all eight histograms required")
    for _, hist in entries:
        rank(hist)
    pooled = [sum(hist[i] for _, hist in entries) for i in range(256)]
    dictionary = rank(pooled)[:15]
    ranges = [{"sample": sample, **describe(hist, dictionary)} for sample, hist in entries]
    summary = describe(pooled, dictionary)
    failed = [r["sample"]["tensor"] for r in ranges
              if not r["global_15"]["passes_99_percent"]]
    passed = not failed and summary["global_15"]["passes_99_percent"]
    return {"schema": "trunk-dictionary-falsifier-v1", "ranges": ranges, "pooled": summary,
            "gate": {"status": "PASS" if passed else "STOP",
                     "rule": "15 pooled entries cover at least 99% in every range and pooled; "
                             "integer counts decide, not rounded percentages",
                     "failed_global_dictionary_ranges": failed,
                     "continue_to_codec": passed},
            "plane": "raw[1::2]: sign + upper 7 exponent bits; low byte stays raw",
            "ratio_scope": "4-bit indexes + one raw byte per escape + raw low plane; "
                           "excludes dictionary, framing, row offsets and padding",
            "limits": "Eight fixed central 1 MiB ranges, not a census. Pooled dictionary is "
                      "fit and scored on the same samples; no unseen-layer generalization. "
                      "No codec, rate, full-model speedup or storage-size claim. "
                      "Pinned trunk bytes are not reread at every budget."}


def markdown(report):
    lines = ["| Range | Distinct | 90% | 99% | 99.9% | 99.99% | Local 15 % | Global 15 % | r |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["ranges"] + [report["pooled"]]:
        tensor = row.get("sample", {}).get("tensor", "Pooled")
        layer = re.search(r"layers\.([0-9]+)\.", tensor)
        label = "Layer " + layer[1] if layer else tensor
        values = row["values_for_coverage_percent"]
        local, glob = row["best_local_15"], row["global_15"]
        lines.append(f"| {label} | {row['distinct_values']} | " +
                     " | ".join(str(values[key]) for key, _ in COVERAGE) +
                     f" | {100 * local['coverage']:.6f} | {100 * glob['coverage']:.6f} | " +
                     f"{glob['payload_ratio']:.9f} |")
    lines.append("\nGate 1: **" + report["gate"]["status"] + "**.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true",
                        help="return 2 on scientific STOP; normal reporting succeeds on STOP")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("real-range acquisition is restricted to GitHub Actions")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = []
    for row in dense_ranges(manifest):
        url = ("https://huggingface.co/moonshotai/Kimi-K3/resolve/" +
               manifest["revision"] + "/" + row["shard"])
        raw, _ = read_range(url, row["offset"], row["bytes"])
        histogram = high_histogram(verified_range(row, raw))
        identity = {key: row[key] for key in ("shard", "tensor", "offset", "bytes", "sha256")}
        entries.append((identity, histogram))
    report = analyze(entries)
    report.update({"checkpoint_revision": manifest["revision"],
                   "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "sample_bytes": sum(row["bytes"] for row, _ in entries),
                   "execution": {"platform": platform.platform(), "machine": platform.machine(),
                                 "head_sha": os.environ.get("RESEARCH_COMMIT"),
                                 "checkout_sha": os.environ.get("GITHUB_SHA"),
                                 "run_id": os.environ.get("GITHUB_RUN_ID"),
                                 "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}})
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    table = markdown(report)
    print(table, flush=True)
    print("FALSIFIER_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(table)
    return 2 if args.require_pass and not report["gate"]["continue_to_codec"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
