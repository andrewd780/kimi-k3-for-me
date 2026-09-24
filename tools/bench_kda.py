#!/usr/bin/env python3
"""Three interleaved synthetic timing runs per KDA arm, with bytewise oracle gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def run(command):
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.batches <= 100000:
        parser.error("--batches must be in [1, 100000]")
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        arch, isa = "", "NEON"
    elif machine in ("x86_64", "amd64"):
        arch, isa = "-mavx2 -mfma", "AVX2"
    else:
        parser.error("this comparison requires an AVX2 or aarch64 runner")
    arms = {}
    for name in ("original_c", "simd"):
        directory = Path("build") / ("kda-" + name)
        binary = directory / "bin"
        flags = arch + (" -DK3_KDA_FORCE_SCALAR" if name == "original_c" else " -DK3_KDA_SIMD")
        build = ["make", "-j2", "BUILD=" + str(directory), "BIN=" + str(binary),
                 "ARCH=" + flags, "OMP_CFLAGS=", "OMP_LDFLAGS="]
        targets = ["test_ops", "test_kda_exact", "k3_model", "bench_kda"]
        print(run(build + [str(binary / t) for t in targets]), flush=True)
        logs = {}
        for target, extra in (("test_ops", ["tests/fixtures/ops"]),
                              ("test_kda_exact", []),
                              ("k3_model", ["tests/fixtures", str(directory / "logits.bin")])):
            log = run([str(binary / target)] + extra)
            (directory / (target + ".log")).write_text(log)
            print(name + " " + log, flush=True)
            logs[target] = log
        data = (directory / "logits.bin").read_bytes()
        if not data or "25 passed, 0 failed, 0 skipped" not in logs["test_ops"]:
            raise ValueError("empty/missing fixture gate")
        arms[name] = {"arch_flags": flags.strip(), "oracle_bytes": len(data),
                      "oracle_sha256": hashlib.sha256(data).hexdigest(),
                      "ops_log_sha256": hashlib.sha256(logs["test_ops"].encode()).hexdigest(),
                      "runs": [], "binary": str(binary / "bench_kda")}
    for field in ("oracle_sha256", "ops_log_sha256", "oracle_bytes"):
        if arms["original_c"][field] != arms["simd"][field]:
            raise ValueError("cross-build byte mismatch: " + field)
    order = []
    for trial in range(3):
        for name in (("original_c", "simd") if trial % 2 == 0 else ("simd", "original_c")):
            measured = json.loads(run([arms[name]["binary"], str(args.batches)]))
            arms[name]["runs"].append(measured)
            order.append({"arm": name, "run": trial + 1})
            print(json.dumps({"arm": name, "run": trial + 1, **measured}), flush=True)
    hashes = {r["state_output_fnv1a"] for a in arms.values() for r in a["runs"]}
    if len(hashes) != 1:
        raise ValueError("benchmark state/output bytes differ")
    for arm in arms.values():
        values = [r["ms_per_96_heads"] for r in arm["runs"]]
        arm["median_ms_per_96_heads"] = statistics.median(values)
        arm["spread_fraction"] = (max(values) - min(values)) / min(values)
        del arm["binary"]
    if platform.system() == "Darwin":
        cpu = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    else:
        cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text()
                    .splitlines() if s.startswith("model name")), "unknown")
    report = {"schema": 1, "kind": "synthetic KDA kernel; no model I/O or s/token claim",
              "platform": platform.platform(), "machine": machine, "cpu": cpu, "isa": isa,
              "compiler": run(["cc", "--version"]).splitlines()[0],
              "ci_run": os.environ.get("GITHUB_RUN_ID"), "revision": run(["git", "rev-parse", "HEAD"]).strip(),
              "threads": 1, "heads": 96, "head_dim": 128, "order": order, "arms": arms,
              "median_speed_ratio": arms["original_c"]["median_ms_per_96_heads"] /
                                    arms["simd"]["median_ms_per_96_heads"],
              "limits": "Original C allows compiler auto-vectorisation. One serial batch of 96 "
                        "heads, repeated warm synthetic inputs. Not an end-to-end decode benchmark."}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("KDA SUMMARY " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
