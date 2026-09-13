#!/usr/bin/env python3
"""Calibrate lazy expert pins and replay held-out requests, using only the stdlib.

The legacy fixture repeats full prefixes. Conversion to a token-major replay is
explicit and validates every repeated prefix; it is not a measured decode trace.
Profiles affect residency only, never routing or weight precision.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import struct

EXPERT_BYTES = 17_547_264
SLOT_BYTES = 17_555_456


def read_trace(path, n_layers, n_experts):
    raw = Path(path).read_bytes()
    if not raw or len(raw) % 8:
        raise ValueError("trace must contain nonempty little-endian int32 pairs")
    pairs = list(struct.iter_unpack("<ii", raw))
    if any(not (0 <= layer < n_layers and 0 <= expert < n_experts)
           for layer, expert in pairs):
        raise ValueError("trace coordinates do not match model geometry")
    return pairs, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                   "recorded_requests": len(pairs)}


def legacy_prefixes(pairs, topk):
    """Validate layer-major repeated prefixes, then transpose the final pass only."""
    runs = [(layer, list(group)) for layer, group in groupby(pairs, key=lambda p: p[0])]
    width = next((i for i in range(1, len(runs)) if runs[i][0] <= runs[i-1][0]), 0)
    if not width or len(runs) % width:
        raise ValueError("not a sequence of repeated layer-major prefix passes")
    layers = [layer for layer, _ in runs[:width]]
    final = runs[-width:]
    lengths = []
    for start in range(0, len(runs), width):
        block = runs[start:start + width]
        size = len(block[0][1])
        if ([layer for layer, _ in block] != layers or size % topk or
                any(len(rows) != size for _, rows in block)):
            raise ValueError("prefix passes have inconsistent layers or top-k counts")
        if lengths and size // topk <= lengths[-1]:
            raise ValueError("prefix lengths must grow strictly")
        for (_, rows), (_, last) in zip(block, final):
            if rows != last[:size]:
                raise ValueError("repeated prefix routing is not identical")
        lengths.append(size // topk)
    ordered = [pair for pos in range(lengths[-1]) for _, rows in final
               for pair in rows[pos * topk:(pos + 1) * topk]]
    seen = set()
    coverage = []
    per_position = width * topk
    for start in range(0, len(ordered), per_position):
        seen.update(ordered[start:start + per_position])
        coverage.append({"positions": start // per_position + 1,
                         "distinct_experts": len(seen),
                         "expert_payload_bytes": len(seen) * EXPERT_BYTES})
    return ordered, {"kind": "derived token-major replay, not measured decode",
                     "prefix_lengths": lengths, "unique_positions": lengths[-1],
                     "requests_per_position": per_position, "coverage": coverage}


def rank(keys):
    """Accept requests or their Counter; both build and evaluation share tie order."""
    return sorted(Counter(keys).items(), key=lambda item: (-item[1], item[0]))


def write_profile(path, ranked, n_layers, n_experts, topk):
    text = "K3EXPERTS 1 %d %d %d\n" % (n_layers, n_experts, topk)
    for key, count in ranked:
        text += "%d %d %d\n" % (key // n_experts, key % n_experts, count)
    Path(path).write_text(text, encoding="ascii")


def replay(keys, capacity, hot=()):
    """Serial native-policy model, cold start; unused pins do not reserve empty slots.

    No top-k prefetch scheduling is modeled. Every first load, including a hot key,
    pays a miss. Only training data may determine hot; never rank the test requests.
    """
    hot = set(hot)
    if capacity < 1 or len(hot) >= capacity:
        raise ValueError("replay needs positive evictable capacity")
    loaded = set()
    cold = OrderedDict()
    hits = 0
    for key in keys:
        if key in loaded:
            hits += 1
        elif key in cold:
            hits += 1
            cold.move_to_end(key)
        else:
            if len(cold) + len(loaded) == capacity:
                cold.popitem(last=False)
            if key in hot:
                loaded.add(key)
            else:
                cold[key] = None
    return {"requests": len(keys), "resident_reuses": hits,
            "loads": len(keys) - hits,
            "projected_expert_payload_bytes": (len(keys) - hits) * EXPERT_BYTES}


def load(path, args):
    pairs, info = read_trace(path, args.n_layers, args.n_experts)
    if args.layout == "legacy-prefixes":
        pairs, derived = legacy_prefixes(pairs, args.topk)
        info.update(derived)
    else:
        info["kind"] = "recorded get() requests; no token-count inference"
    return [layer * args.n_experts + expert for layer, expert in pairs], info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    for command in ("build", "evaluate"):
        cmd = commands.add_parser(command)
        cmd.add_argument("traces", nargs="+" if command == "build" else None)
        cmd.add_argument("--layout", choices=("requests", "legacy-prefixes"),
                         default="requests")
        cmd.add_argument("--n-layers", type=int, default=93)
        cmd.add_argument("--n-experts", type=int, default=896)
        cmd.add_argument("--topk", type=int, default=16)
        cmd.add_argument("--out", type=Path, required=True)
        if command == "build":
            cmd.add_argument("--take-requests", type=int,
                             help="use only this many requests from each input")
        else:
            cmd.add_argument("--train-requests", type=int, required=True)
            cmd.add_argument("--slots", default="28,455,615,1344,2073,3647,6208")
            cmd.add_argument("--pin-fractions", default="0,0.5,1")
    args = ap.parse_args(argv)
    try:
        if not (0 < args.n_layers <= 4096 and 0 < args.n_experts <= 65536 and
                0 < args.topk <= min(64, args.n_experts)):
            raise ValueError("invalid model geometry")
        if args.command == "build":
            counts = Counter()
            for path in args.traces:
                keys, _ = load(path, args)
                if args.take_requests is not None:
                    if not 0 < args.take_requests <= len(keys):
                        raise ValueError("take-requests exceeds a trace or is nonpositive")
                    keys = keys[:args.take_requests]
                counts.update(keys)
            ranked = rank(counts)
            write_profile(args.out, ranked, args.n_layers, args.n_experts, args.topk)
            print("wrote %s: %d ranked experts from %d calibration requests" %
                  (args.out, len(ranked), sum(counts.values())))
            return 0
        keys, info = load(args.traces, args)
        if not 0 < args.train_requests < len(keys):
            raise ValueError("split must leave nonempty training and test requests")
        slots = [int(value) for value in args.slots.split(",")]
        fractions = [float(value) for value in args.pin_fractions.split(",")]
        if any(cap < args.topk + 1 for cap in slots):
            raise ValueError("each capacity must leave at least topk+1 slots")
        if any(not 0 <= fraction <= 1 for fraction in fractions):
            raise ValueError("pin fractions must be in [0,1]")
        train, test = keys[:args.train_requests], keys[args.train_requests:]
        ranked = rank(train)
        report = {"kind": "deterministic replay; one result per arm",
                  "source": info, "training_requests": len(train),
                  "heldout_requests": len(test), "training_distinct": len(set(train)),
                  "heldout_distinct": len(set(test)),
                  "limitations": ["one context; held-out suffix is not prompt diversity",
                                  "serial request replay excludes batch prefetch",
                                  "projected expert payload, not measured SSD traffic",
                                  "no timing or whole-model speed claim"], "arms": []}
        for cap in slots:
            for fraction in fractions:
                count = min(int(cap * fraction), cap - args.topk - 1, len(ranked))
                hot = [key for key, _ in ranked[:count]]
                report["arms"].append({"slots": cap, "arena_bytes": cap * SLOT_BYTES,
                                       "requested_pin_fraction": fraction, "pins": count,
                                       "result": replay(test, cap, hot)})
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    except (OSError, ValueError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
