#!/usr/bin/env python3
"""Gate 1: exact high-byte histograms for eight pinned BF16 dense ranges, plus the
fixed-width bit-width curve and order-0 entropy bounds computed from the same counts.

The proposed plane is raw[1::2], matching bench_huf4: sign plus the upper seven
exponent bits. The low byte retains the last exponent bit and seven fraction
bits. This is deliberately NOT a histogram of the mathematical 8-bit exponent.
No encoder, decoder or timing arm runs before this falsifier passes.

`--curve-from FILE --pointer /json/pointer` evaluates the curve on a histogram
that is already committed (no range acquisition, runs anywhere).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import re
import struct

from remote_model import read_range

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/measurements/lossless-samples.json"
COVERAGE = (("90", 9000), ("99", 9900), ("99.9", 9990), ("99.99", 9999))
CURVE_WIDTHS = (3, 4, 5)


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


def bf16_histogram(raw):
    """Sparse order-0 histogram of whole little-endian BF16 values."""
    if not raw or len(raw) % 2:
        raise ValueError("expected complete, nonempty BF16 values")
    return Counter(struct.unpack(f"<{len(raw) // 2}H", raw))


def entropy_bits(counts):
    """Empirical order-0 entropy, bits per symbol, from integer counts."""
    counts = list(counts)
    if any(type(c) is not int or c < 0 for c in counts) or not sum(counts):
        raise ValueError("invalid counts")
    n = sum(counts)
    return math.log2(n) - sum(c * math.log2(c) for c in counts if c) / n


def huffman_bits(counts):
    """Total bits of an optimal unconstrained prefix (Huffman) code for these counts.

    This is sum(count * code length); tie-breaking cannot change it. A one-symbol
    alphabet is charged one bit per value, as a decodable prefix code would be.
    The research decoder's 12-bit length limit and stream padding are not modeled."""
    weights = list(counts)
    if any(type(c) is not int or c < 0 for c in weights) or not sum(weights):
        raise ValueError("invalid counts")
    weights = [c for c in weights if c]
    if len(weights) == 1:
        return weights[0]
    heapq.heapify(weights)
    total = 0
    while len(weights) > 1:
        merged = heapq.heappop(weights) + heapq.heappop(weights)
        total += merged
        heapq.heappush(weights, merged)
    return total


def magnitude_histogram(histogram):
    """Fold the sign bit away: counts of the upper seven exponent bits."""
    return [histogram[m] + histogram[m | 0x80] for m in range(128)]


def magnitude_rank(magnitudes):
    # Same deterministic order as rank(): count descending, ascending value.
    return sorted((m for m in range(128) if magnitudes[m]), key=lambda m: (-magnitudes[m], m))


def _scheme(name, n, index_bits, capacity, entries, covered):
    escapes = n - covered
    # Per BF16: index bits + raw low byte; per escape: one raw byte, byte-aligned
    # like FD4B's escape stream. Exact: ratio = payload_bits / (16 * n).
    payload_bits = n * (index_bits + 8) + 8 * escapes
    return {"scheme": name, "index_bits": index_bits, "dictionary_capacity": capacity,
            "dictionary": entries, "escape_values": escapes, "p_escape": escapes / n,
            "payload_bits": payload_bits, "bits_per_bf16": payload_bits / n,
            "payload_ratio": payload_bits / (16 * n)}


def fixed_width(histogram, ranking, index_bits):
    """k-bit index into the first 2**k - 1 pooled entries; code 2**k - 1 escapes."""
    capacity = (1 << index_bits) - 1
    entries = ranking[:capacity]
    return _scheme(f"fixed_{index_bits}bit", sum(histogram), index_bits, capacity, entries,
                   sum(histogram[v] for v in entries))


def sign_split(histogram, ranking, index_bits):
    """Raw sign bit + (k-1)-bit index into 2**(k-1) - 1 pooled magnitudes + escape.

    Same k + 8 bits per BF16 as fixed_width; an escape stores the raw high byte."""
    capacity = (1 << (index_bits - 1)) - 1
    magnitudes = magnitude_histogram(histogram)
    entries = ranking[:capacity]
    return _scheme(f"sign_split_{index_bits}bit", sum(histogram), index_bits, capacity,
                   entries, sum(magnitudes[m] for m in entries))


def bit_width_curve(histogram, ranking=None, magnitude_ranking=None, full_histogram=None,
                    widths=CURVE_WIDTHS):
    """Exact payload ratios of fixed-width index schemes and their order-0 bounds.

    Rankings default to this histogram's own; analyze() passes the pooled ones so that
    every range is scored with the single pooled dictionary. `full_histogram` maps
    whole 16-bit BF16 values to counts and adds the full-value entropy bound."""
    own = rank(histogram)
    n = sum(histogram)
    ranking = own if ranking is None else ranking
    if magnitude_ranking is None:
        magnitude_ranking = magnitude_rank(magnitude_histogram(histogram))
    if 4 not in widths or any(type(k) is not int or not 2 <= k <= 8 for k in widths):
        raise ValueError("widths must include the 4-bit reference and lie in 2..8")
    schemes = ([fixed_width(histogram, ranking, k) for k in widths] +
               [sign_split(histogram, magnitude_ranking, k) for k in widths])
    huffman = huffman_bits(histogram)
    high = entropy_bits(histogram)
    bounds = {"high_byte_entropy_bits": high, "high_byte_entropy_ratio": (8 + high) / 16,
              "huffman_high_byte_bits": huffman, "huffman_high_byte_bits_per_bf16": huffman / n,
              "huffman_high_byte_ratio": (8 * n + huffman) / (16 * n)}
    if full_histogram is not None:
        marginal, low = [0] * 256, [0] * 256
        for value, count in full_histogram.items():
            if type(value) is not int or not 0 <= value <= 0xFFFF:
                raise ValueError("full histogram keys must be 16-bit values")
            marginal[value >> 8] += count
            low[value & 0xFF] += count
        if marginal != list(histogram):
            raise ValueError("full BF16 histogram does not match the high-byte histogram")
        full = entropy_bits(full_histogram.values())
        bounds.update({"full_bf16_distinct_values": sum(1 for c in full_histogram.values() if c),
                       "full_bf16_entropy_bits": full, "full_bf16_entropy_ratio": full / 16,
                       "low_byte_entropy_bits": entropy_bits(low),
                       "low_given_high_entropy_bits": full - high})
    reference = next(s for s in schemes if s["scheme"] == "fixed_4bit")
    for s in schemes:
        s["gain_vs_fixed_4bit_points"] = 100 * (reference["payload_ratio"] - s["payload_ratio"])
        s["gap_to_huffman_points"] = 100 * (s["payload_ratio"] - bounds["huffman_high_byte_ratio"])
        s["gap_to_high_entropy_points"] = 100 * (s["payload_ratio"] -
                                                 bounds["high_byte_entropy_ratio"])
        if full_histogram is not None:
            s["gap_to_full_bf16_entropy_points"] = 100 * (s["payload_ratio"] -
                                                          bounds["full_bf16_entropy_ratio"])
    best = min((s for s in schemes if s is not reference), key=lambda s: s["payload_bits"])
    # 1.5 ratio points, in integers: (bits4 - bits) / (16 n) >= 15 / 1000.
    prototype = 1000 * (reference["payload_bits"] - best["payload_bits"]) >= 240 * n
    return {"bf16_values": n, "schemes": schemes, "bounds": bounds,
            "decision": {"rule": "prototype a decoder only if a scheme beats fixed_4bit by at "
                                 "least 1.5 ratio points; integer payload bits decide",
                         "best_scheme": best["scheme"],
                         "best_gain_vs_fixed_4bit_points": best["gain_vs_fixed_4bit_points"],
                         "prototype": prototype},
            "scope": "payload only: index bits + raw low byte + one raw byte per escape; "
                     "excludes dictionary table, framing, row index and alignment"}


def analyze(entries, full_histograms=None):
    if len(entries) != 8:
        raise ValueError("all eight histograms required")
    if full_histograms is not None and len(full_histograms) != len(entries):
        raise ValueError("one full BF16 histogram per range required")
    for _, hist in entries:
        rank(hist)
    pooled = [sum(hist[i] for _, hist in entries) for i in range(256)]
    ranking = rank(pooled)
    dictionary = ranking[:15]
    magnitudes = magnitude_rank(magnitude_histogram(pooled))
    fulls = full_histograms or [None] * len(entries)
    pooled_full = sum((Counter(f) for f in fulls), Counter()) if full_histograms else None
    ranges = [{"sample": sample, **describe(hist, dictionary),
               "bit_width_curve": bit_width_curve(hist, ranking, magnitudes, full)}
              for (sample, hist), full in zip(entries, fulls)]
    summary = describe(pooled, dictionary)
    summary["bit_width_curve"] = bit_width_curve(pooled, ranking, magnitudes, pooled_full)
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


def curve_markdown(curve, title):
    bounds, points = curve["bounds"], "gain_vs_fixed_4bit_points"
    reference = next(s for s in curve["schemes"] if s["scheme"] == "fixed_4bit")
    lines = [f"Bit-width curve, {title}: {curve['bf16_values']} BF16 values, payload only.", "",
             "| Scheme | Entries | Escapes | p escape % | Payload r | Gain over 4-bit, points | "
             "Excess over Huffman, points |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in curve["schemes"]:
        lines.append(f"| {s['scheme']} | {s['dictionary_capacity']} | {s['escape_values']} | "
                     f"{100 * s['p_escape']:.6f} | {s['payload_ratio']:.9f} | {s[points]:+.4f} | "
                     f"{s['gap_to_huffman_points']:+.4f} |")
    rows = [("Huffman high byte + raw low", bounds["huffman_high_byte_ratio"]),
            ("Order-0 entropy, high byte + raw low", bounds["high_byte_entropy_ratio"])]
    if "full_bf16_entropy_ratio" in bounds:
        rows.append(("Order-0 entropy, whole BF16", bounds["full_bf16_entropy_ratio"]))
    for label, ratio in rows:
        lines.append(f"| {label} | | | | {ratio:.9f} | "
                     f"{100 * (reference['payload_ratio'] - ratio):+.4f} | "
                     f"{100 * (ratio - bounds['huffman_high_byte_ratio']):+.4f} |")
    decision = curve["decision"]
    lines.append(f"\nBest non-4-bit scheme: {decision['best_scheme']} "
                 f"({decision['best_gain_vs_fixed_4bit_points']:+.4f} points); "
                 f"prototype at >=1.5 points: **{decision['prototype']}**.")
    return "\n".join(lines) + "\n"


def resolve_pointer(document, pointer):
    """RFC 6901 JSON pointer, plus the dict ancestors passed on the way down."""
    if pointer and not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    node, ancestors = document, []
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            ancestors.append(node)
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node, ancestors


PROVENANCE_KEYS = ("commit", "workflow_run", "source_url", "stage", "job", "job_id", "machine",
                   "source", "samples", "sha256", "raw_bytes")


def committed_curve(path, pointer):
    """Curve for a histogram already recorded in the repository; no range is fetched."""
    histogram, ancestors = resolve_pointer(json.loads(path.read_bytes()), pointer)
    provenance = {}
    for node in ancestors:
        provenance.update({key: node[key] for key in PROVENANCE_KEYS if key in node})
    canonical = json.dumps(histogram, separators=(",", ":")).encode()
    return {"schema": "fixed-width-curve-v1",
            "source": {"file": path.resolve().relative_to(ROOT).as_posix(), "pointer": pointer,
                       "histogram_sha256": hashlib.sha256(canonical).hexdigest(),
                       "recorded_with": provenance},
            "histogram": histogram, "bit_width_curve": bit_width_curve(histogram)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true",
                        help="return 2 on scientific STOP; normal reporting succeeds on STOP")
    parser.add_argument("--curve-from", type=Path,
                        help="committed JSON holding a 256-count high-byte histogram: compute "
                             "the bit-width curve only, without acquiring any range")
    parser.add_argument("--pointer", default="",
                        help="RFC 6901 pointer to the histogram inside --curve-from")
    args = parser.parse_args()
    if args.curve_from:
        record = committed_curve(args.curve_from, args.pointer)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(curve_markdown(record["bit_width_curve"], record["source"]["file"] +
                             "#" + args.pointer), flush=True)
        print("CURVE_REPORT " + json.dumps(record, separators=(",", ":")), flush=True)
        return 0
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("real-range acquisition is restricted to GitHub Actions")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries, fulls = [], []
    for row in dense_ranges(manifest):
        url = ("https://huggingface.co/moonshotai/Kimi-K3/resolve/" +
               manifest["revision"] + "/" + row["shard"])
        raw, _ = read_range(url, row["offset"], row["bytes"])
        histogram = high_histogram(verified_range(row, raw))
        identity = {key: row[key] for key in ("shard", "tensor", "offset", "bytes", "sha256")}
        entries.append((identity, histogram))
        fulls.append(bf16_histogram(raw))
    report = analyze(entries, fulls)
    report.update({"checkpoint_revision": manifest["revision"],
                   "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "sample_bytes": sum(row["bytes"] for row, _ in entries),
                   "execution": {"platform": platform.platform(), "machine": platform.machine(),
                                 "head_sha": os.environ.get("RESEARCH_COMMIT"),
                                 "checkout_sha": os.environ.get("GITHUB_SHA"),
                                 "run_id": os.environ.get("GITHUB_RUN_ID"),
                                 "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}})
    # Whole-value counts go to the artifact only; logs carry the derived entropies.
    pooled_full = sum((Counter(f) for f in fulls), Counter())
    artifact = {**report, "pooled_full_bf16_histogram": sorted(pooled_full.items())}
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    table = markdown(report) + "\n" + curve_markdown(report["pooled"]["bit_width_curve"],
                                                     "eight ranges pooled")
    print(table, flush=True)
    print("FALSIFIER_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)
    print("CURVE_REPORT " + json.dumps(report["pooled"]["bit_width_curve"],
                                       separators=(",", ":")), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(table)
    return 2 if args.require_pass and not report["gate"]["continue_to_codec"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
