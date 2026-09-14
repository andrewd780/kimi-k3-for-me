#!/usr/bin/env python3
"""Reproduce conditional streaming bounds, never hardware measurements.

Inputs distinguish measured sample ratios from hypothetical bandwidth/compute.
Fractions keep the identities exact until human-readable JSON conversion.
No checkpoint, network, native kernels, or optional Python package is needed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs/measurements/scale-plane-summary.json"


def positive(value):
    try:
        result = F(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def pipeline_schedule(count, read, compute, buffers=2):
    """One reader, one compute resource; task j needs its own completed read.

    Buffer j%buffers may be overwritten only after its old compute completes.
    This is a precedence-graph calculation, not a disk simulation or benchmark.
    """
    if count < 1 or buffers < 1 or read <= 0 or compute <= 0:
        raise ValueError("positive task count, buffers and durations required")
    free = [F(0)] * buffers
    io_done = compute_done = F(0)
    for j in range(count):
        slot = j % buffers
        io_done = max(io_done, free[slot]) + read
        compute_done = max(io_done, compute_done) + compute
        free[slot] = compute_done
    return compute_done


def uniform_union(experts, topk, batch):
    """E[unique experts] for independent uniform top-k sets across requests.

    Uniform balanced marginal routing alone does NOT imply this independence.
    """
    if not 1 <= topk <= experts or batch < 1:
        raise ValueError("invalid routing geometry")
    return experts * (1 - (1 - F(topk, experts)) ** batch)


def prefetch_reads(topk, guesses, correct):
    """Expected/exact reads with no cache hits; all wrong guesses finish reading.

    Correct guesses are retained. Cache pollution and canceled/partial I/O are
    omitted. `correct` may be fractional for an explicitly assumed expectation.
    """
    if topk < 1 or guesses < 0 or not 0 <= correct <= min(topk, guesses):
        raise ValueError("invalid prediction counts")
    return guesses + topk - correct


def accepted_tokens(alpha, draft_length):
    """IID conditional agreement toy model, including one verifier/bonus token."""
    if not 0 <= alpha <= 1 or draft_length < 0:
        raise ValueError("invalid speculation assumptions")
    return sum((alpha ** i for i in range(draft_length + 1)), F(0))


def build_report(ssd_gbps=F(3), decode_gbps=F("0.68"), compute_seconds=F(216)):
    if min(ssd_gbps, decode_gbps, compute_seconds) <= 0:
        raise ValueError("rates and compute time must be positive")
    evidence = SAMPLES.read_bytes()
    sample = json.loads(evidence, parse_float=F)
    r = sample["summary_ratios"]["scale"]["zstd3"]
    if not 0 < r < 1:
        raise ValueError("expected a compressible measured scale sample")
    n_layer, n_expert, topk, expert_size = 92, 896, 16, 17547264
    if expert_size % 17:
        raise ValueError("expert geometry does not match 16 packed bytes per scale")
    # Trunk is an approximate published total; expert arithmetic uses exact geometry.
    trunk = 108810000000
    expert = n_layer * topk * expert_size
    scale = expert // 17
    packed = expert - scale
    raw = trunk + expert
    selective = trunk + packed + r * scale
    B, D = ssd_gbps * 10**9, decode_gbps * 10**9
    C = compute_seconds
    histogram = sample["summary"]["scale"]["histogram"]
    top3 = sorted(range(256), key=lambda x: (-histogram[x], x))[:3]
    observed = sum(histogram)
    exceptions = observed - sum(histogram[x] for x in top3)
    # A hypothetical lossless 2-bit selector: three literal bytes + full-byte escape.
    palette_bits = 2 * observed + 8 * exceptions
    storage_raw = 1560000000000
    storage_scale = sample["scale_plane_projection"]["assumed_raw_bytes"]
    batch = []
    for b in (1, 2, 4, 8, 16, 32):
        unique = uniform_union(n_expert, topk, b)
        experts_per_output = n_layer * expert_size * unique / b
        total = F(trunk, b) + experts_per_output
        batch.append({"independent_requests": b, "expected_experts_per_layer": unique,
                      "expert_GB_per_output": experts_per_output / 10**9,
                      "total_GB_per_output": total / 10**9,
                      "ideal_IO_throughput_ratio": raw / total})
    schedules = []
    for read, comp in ((F(1), F(1)), (F(1), F(4)), (F(4), F(1))):
        serial = topk * (read + comp)
        piped = pipeline_schedule(topk, read, comp)
        schedules.append({"read_cost_units": read, "compute_cost_units": comp,
                          "barrier_cost_units": serial, "two_buffer_cost_units": piped,
                          "ideal_expert_stage_speed_ratio": serial / piped})
    return {
        "schema": 1,
        "kind": "conditional mathematical model; not measured latency or throughput",
        "evidence": {"path": str(SAMPLES.relative_to(ROOT)),
                     "sha256": hashlib.sha256(evidence).hexdigest(),
                     "checkpoint_revision": sample["revision"],
                     "scale_retained_fraction_sample": r},
        "assumptions": {"SSD_GBps": ssd_gbps, "aggregate_scale_decode_GBps": decode_gbps,
                        "compute_seconds_if_executed_alone": compute_seconds,
                        "compute_is_hypothetical": True, "trunk_GB_approx": F(trunk, 10**9),
                        "expert_bytes": expert_size, "MoE_layers": n_layer,
                        "topk": topk, "experts_per_layer": n_expert,
                        "persistent_cache_hits": 0,
                        "limits": "No contention, metadata, alignment, page cache or startup "
                                  "cost calibration. Bandwidth and compute are scenario inputs."},
        "bytes": {"expert_per_token": expert, "scale_per_token": scale,
                  "raw_total_per_token": raw, "selective_total_per_token": selective,
                  "expert_fraction_saved": 1 - (16 + r) / 17,
                  "total_fraction_saved": 1 - selective / raw,
                  "ideal_IO_speed_ratio": raw / selective,
                  "ideal_IO_ratio_if_all_expert_reads_vanished": F(raw, trunk),
                  "ideal_IO_ratio_if_all_trunk_reads_vanished": F(raw, expert),
                  "scale_plane_GB_assumed": F(storage_scale, 10**9),
                  "scale_plane_GB_saved_projection": storage_scale * (1-r) / 10**9,
                  "checkpoint_GB_after_scales_projection":
                      (storage_raw - storage_scale * (1-r)) / 10**9},
        "codec_bounds": {
            "serial_break_even_decode_GBps": ssd_gbps / (1-r),
            "ideal_dedicated_decode_GBps_to_feed_expert_stream":
                ssd_gbps / (16+r),
            "raw_serial_scenario_seconds": C + raw / B,
            "selective_serial_scenario_seconds": C + selective / B + scale / D,
            "raw_perfect_overlap_lower_bound_seconds": max(C, raw / B),
            "selective_perfect_overlap_dedicated_decoder_lower_bound_seconds":
                max(C, selective / B, scale / D),
            "selective_perfect_overlap_shared_CPU_work_lower_bound_seconds":
                max(C + scale / D, selective / B),
            "shared_CPU_assumption": "C and scale/D are normalized service times on the "
                                     "same exclusive CPU resource; actual scaling is unknown."},
        "known_route_pipeline": {
            "assumptions": "One fixed-bandwidth reader, compute overlaps without slowing; "
                           "known experts consumed in original order; bytes unchanged.",
            "two_expert_buffer_bytes": 2 * expert_size,
            "current_topk_plus_one_arena_payload_bytes": (topk + 1) * expert_size,
            "memory_is_scheduling_floor_not_supported_engine_setting": True,
            "scenarios": schedules},
        "prediction": [{"assumed_recall_at_16": recall,
                        "expert_traffic_multiplier": prefetch_reads(16, 16, 16*recall)/16,
                        "total_traffic_multiplier":
                            (trunk + expert * prefetch_reads(16,16,16*recall)/16) / raw}
                       for recall in (F(0), F("0.35"), F("0.7"), F(1))],
        "batching_uniform_independent_model": batch,
        "lossy_topk_IO_only": [{"topk": k,
                               "total_GB_per_token": (trunk + expert * F(k, topk)) / 10**9,
                               "ideal_IO_speed_ratio": raw / (trunk + expert * F(k, topk))}
                              for k in (1, 4, 8, 16)],
        "speculation": [{"assumed_conditional_agreement": a, "draft_length": 4,
                         "expected_verified_outputs": accepted_tokens(a, 4)}
                        for a in (F(0), F("0.125"), F("0.5"), F("0.9"), F(1))],
        "unimplemented_scale_palette": {
            "common_bytes": top3, "sample_values": observed, "escape_values": exceptions,
            "selector_plus_escape_bits_per_value": F(palette_bits, observed),
            "retained_fraction_excluding_index_header": F(palette_bits, 8 * observed),
            "limits": "An exact representation size on the observed histogram, not a "
                      "codec implementation, decoded speed, or unseen-data size guarantee."},
    }


def jsonable(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-gbps", type=positive, default=F(3))
    parser.add_argument("--decode-gbps", type=positive, default=F("0.68"))
    parser.add_argument("--compute-seconds", type=positive, default=F(216),
                        help="hypothetical exclusive compute service time, NOT a Mac estimate")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = jsonable(build_report(args.ssd_gbps, args.decode_gbps, args.compute_seconds))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
