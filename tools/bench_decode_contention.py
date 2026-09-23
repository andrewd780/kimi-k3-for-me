#!/usr/bin/env python3
"""Decode-under-contention gate: run bench_decode_contention, derive the break-even.

Per layer of W raw trunk bytes, retained fraction r, SSD bandwidth B, decode rate D
(reconstructed GB/s, one core) and matmul consumption M (weight GB/s):

  uncompressed, streamed   t_raw = max(W/B, W/M_all)             read overlaps matmul
  compressed, streamed     t_fd  = max(rW/B, W/D_c, W/M_c)       read, decode, matmul
  resident                 t_raw = W/M_all;  t_fd = max(W/D_c, W/M_c)

M_all uses every core; D_c and M_c are measured while the decoder holds one core and
the matmul the rest. Decoding a layer is hidden when W/D_c <= max(rW/B, W/M_c): the
decoder finishes its 0.75*W compressed bytes (FD4B) before the stage it feeds.
Per-token CPU cost of decoding the whole trunk: 108.81 GB / D core-seconds.

Every derived figure is computed twice: from each arm's median, and "worst", from
each arm at the run least favourable to compression. The speedup falls as the
uncompressed baseline M_all rises and rises with D_c and M_c, so the worst speedup
takes M_all's fastest run with the slowest D_c and M_c: it is at or below the
speedup of every repeat, however the runs pair up, and the worst resident slowdown
is at or above every repeat's. Slowdown ratios take the uncontended arm's fastest
run over the contended arm's slowest. A per-arm minimum is not conservative: it
also takes M_all's slowest run, so one disturbed baseline repeat raises the
speedup and lowers the resident slowdown.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

TRUNK_BYTES = 108.81e9
SSD_GBPS = (2.5, 3.0, 4.0, 5.0, 6.0)


def stage_times(r, b, decode, matmul_rest, matmul_all):
    """Seconds per GB of raw weights for each pipeline, and the limiting stage."""
    if min(r, b, decode, matmul_rest, matmul_all) <= 0:
        raise ValueError("rates and ratio must be positive")
    raw = max(1 / b, 1 / matmul_all)
    stages = {"ssd": r / b, "decode": 1 / decode, "matmul": 1 / matmul_rest}
    compressed = max(stages.values())
    return raw, compressed, max(stages, key=lambda k: (stages[k], k == "decode"))


def break_even(r, decode, matmul_rest, matmul_all, ssd=SSD_GBPS):
    streamed = []
    for b in ssd:
        raw, fd, limit = stage_times(r, b, decode, matmul_rest, matmul_all)
        streamed.append({"ssd_GBps": b, "raw_s_per_GB": raw, "compressed_s_per_GB": fd,
                         "speedup": raw / fd, "limiting_stage": limit,
                         "decode_hidden": 1 / decode <= max(r / b, 1 / matmul_rest),
                         "decode_needed_for_full_gain_GBps": b / r,
                         "decode_needed_to_break_even_GBps": b})
    return {"streamed": streamed,
            "resident": {"slowdown": matmul_all * max(1 / decode, 1 / matmul_rest),
                         "decode_hidden": decode >= matmul_rest,
                         "decode_needed_GBps": matmul_rest,
                         "matmul_core_loss": matmul_all / matmul_rest},
            "core_seconds_per_token": TRUNK_BYTES / (decode * 1e9),
            "compressed_GB_read_per_token": r * TRUNK_BYTES / 1e9}


def arm_rate(arm, statistic):
    """An arm's median as the benchmark reports it, or its slowest or fastest run."""
    if statistic == "median":
        return arm["median"]
    return {"min": min, "max": max}[statistic](arm["GBps_runs"])


def figures(r, low, high):
    """Derived figures from rates `low`, used where a lower rate is worse for
    compression, and `high`, used where a higher one is. The median passes the same
    rates as both; the worst case passes each arm's slowest and fastest run."""
    return {
        "contended": break_even(r, low["decode_concurrent"], low["matmul_concurrent"],
                                high["matmul_all_threads"]),
        "alone": break_even(r, low["decode_alone"], low["matmul_rest_threads"],
                            high["matmul_all_threads"]),
        "decode_slowdown_under_matmul": high["decode_alone"] / low["decode_concurrent"],
        "matmul_slowdown_under_decode": high["matmul_rest_threads"] / low["matmul_concurrent"],
        "core_seconds_per_token_alone": TRUNK_BYTES / (low["decode_alone"] * 1e9),
        "core_seconds_per_token_contended": TRUNK_BYTES / (low["decode_concurrent"] * 1e9)}


def summarize(run):
    arms = run["arms"]
    r = run["packed_bytes"] / run["raw_bytes"]
    rates = {stat: {name: arm_rate(arm, stat) for name, arm in arms.items()}
             for stat in ("median", "min", "max")}
    return {"payload_and_framing_ratio": r,
            "derived": {"median": figures(r, rates["median"], rates["median"]),
                        "worst": figures(r, rates["min"], rates["max"])}}


def markdown(report):
    lines = ["| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | "
             "Matmul rest | Matmul + decode | Core-s/token (contended) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for stat in ("median", "min", "max"):
            a = {k: arm_rate(v, stat) for k, v in run["arms"].items()}
            lines.append(f"| FD{run['index_bits']}B {run['native']} | {run['input']} | {stat} | "
                         f"{a['decode_alone']:.3f} | "
                         f"{a['decode_concurrent']:.3f} | {a['matmul_all_threads']:.3f} | "
                         f"{a['matmul_rest_threads']:.3f} | {a['matmul_concurrent']:.3f} | "
                         f"{TRUNK_BYTES / (a['decode_concurrent'] * 1e9):.2f} |")
    return "\n".join(lines) + "\n"


def break_even_markdown(report, statistic):
    """Streamed speedup over raw reads at each SSD rate, from the contended rates, with
    the limiting stage; then the resident slowdown. `statistic` is "median" or "worst".
    r excludes the FDRX index, which is small enough (0.028 points here) to keep
    resident rather than stream."""
    lines = ["| Format | Input | Stat | r | " +
             " | ".join(f"B = {b:g} GB/s" for b in SSD_GBPS) + " | Resident slowdown |",
             "|---|---|---|---:|" + "---:|" * (len(SSD_GBPS) + 1)]
    for run in report["runs"]:
        derived = run["summary"]["derived"][statistic]["contended"]
        cells = [f"{row['speedup']:.3f} ({row['limiting_stage']})"
                 for row in derived["streamed"]]
        lines.append(f"| FD{run['index_bits']}B {run['native']} | {run['input']} | "
                     f"{statistic} | {run['summary']['payload_and_framing_ratio']:.4f} | " +
                     " | ".join(cells) + f" | {derived['resident']['slowdown']:.3f} |")
    return "\n".join(lines) + "\n"


def load_average():
    try:
        return os.getloadavg()
    except OSError:
        return None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu model")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 3], choices=(3, 4))
    parser.add_argument("--inputs", nargs="+", default=["stream", "hot"],
                        choices=("stream", "hot"))
    args = parser.parse_args()
    load_before = load_average()
    runs = []
    for placement in args.inputs:
        for bits in args.bits:
            result = subprocess.run([str(args.binary.resolve()), str(bits), str(args.threads),
                                     str(args.seconds), str(args.repeats), placement],
                                    capture_output=True, text=True, timeout=900)
            if result.returncode:
                raise SystemExit("contention benchmark failed: " + result.stderr)
            run = json.loads(result.stdout)
            if (run["byte_exact"] is not True or run["index_bits"] != bits or not run["openmp"]
                    or run["input"] != placement):
                raise SystemExit("benchmark must be byte-exact, as requested, with OpenMP")
            run["summary"] = summarize(run)
            runs.append(run)
    report = {"schema": "decode-contention-v1", "runs": runs,
              "execution": {"machine": platform.machine(), "system": platform.platform(),
                            "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
                            "threads": args.threads, "seconds_per_arm": args.seconds,
                            "repeats": args.repeats, "load_average_before": load_before,
                            "load_average_after": load_average(),
                            "head_sha": os.environ.get("RESEARCH_COMMIT"),
                            "run_id": os.environ.get("GITHUB_RUN_ID")},
              "scope": "synthetic weights with the committed four-range high-byte "
                       "distribution; one decoder thread against k3_matmul_bf16 on the "
                       "remaining threads; matmul weights from DRAM; decoder input either "
                       "streamed from DRAM or one cache-resident chunk; no disk, no "
                       "full-model claim"}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    table = (markdown(report) + "\n" + break_even_markdown(report, "median") + "\n" +
             break_even_markdown(report, "worst"))
    print(table, flush=True)
    print("CONTENTION_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
