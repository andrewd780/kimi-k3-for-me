#!/usr/bin/env python3
"""Run one CI fixture inside an already capped cgroup v2 and record the actual cap."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.limit <= 0:
        parser.error("a positive cap and command are required")
    entries = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line[3:] for line in entries if line.startswith("0::")), None)
    if relative is None:
        raise RuntimeError("cgroup v2 is required; refusing an uncapped substitute")
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    maximum = (group / "memory.max").read_text().strip()
    swap = (group / "memory.swap.max").read_text().strip()
    if maximum != str(args.limit) or swap != "0":
        raise RuntimeError("the requested memory/no-swap cap was not applied")
    result = subprocess.run(command, check=False)
    events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
    report = {"schema": 1, "cgroup": relative, "memory_max_bytes": int(maximum),
              "memory_swap_max_bytes": int(swap),
              "memory_peak_bytes": int((group / "memory.peak").read_text()),
              "events": {key: int(value) for key, value in events.items()},
              "returncode": result.returncode}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return result.returncode if result.returncode else int(report["events"].get("oom", 0) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
