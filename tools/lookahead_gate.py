#!/usr/bin/env python3
"""Bounded speculation economics; exact rational math, not K3 measurements.

Model: independent per-draft acceptance p, window w, one verification costing V,
an additional proposal pass costing D, and replay cost R if any draft is rejected.
All costs are multiples of a plain decode step. Real acceptance is correlated;
recording measured verified-prefix lengths is required before enabling lookahead.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path


def analyze(p, window, verify, draft, replay):
    if not 0 <= p <= 1 or not 1 <= window <= 8 or min(verify, draft, replay) < 0:
        raise ValueError("invalid acceptance probability, window or nonnegative costs")
    tokens = sum((p**i for i in range(window + 1)), F(0))
    cost = verify + draft + (1 - p**window) * replay
    if not cost:
        raise ValueError("total cost must be positive")
    # maximum_extra_draft_cost is the break-even proposal cost: profitable only for a
    # draft cost strictly below it, so a value of 0 means no positive draft cost pays.
    return {"expected_tokens": str(tokens), "expected_step_cost": str(cost),
            "speed_ratio": str(tokens / cost), "profitable_under_assumptions": tokens > cost,
            "maximum_extra_draft_cost": str(tokens - verify - (1 - p**window) * replay)}


def jacobi_proposal(next_token, prefix, guess, rounds):
    """Small, stateless reference only. Each round uses the OLD guess everywhere.

    Calling a real K3 verifier here costs another model sweep. Keeping all rounds
    in the cost account is the point; this is not an engine implementation.
    """
    if not 1 <= len(guess) <= 8 or not 0 <= rounds <= len(guess):
        raise ValueError("bounded window/round count required")
    current = list(guess)
    for _ in range(rounds):
        current = [next_token(prefix + current[:j]) for j in range(len(current))]
    return current


def verify_prefix(next_token, prefix, draft):
    """Emit only matching drafts and the exact first correction/bonus token."""
    accepted = []
    for candidate in draft:
        true = next_token(prefix + accepted)
        if true != candidate:
            return accepted + [true]
        accepted.append(candidate)
    return accepted + [next_token(prefix + accepted)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=F, required=True)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--verify-cost", type=F, required=True)
    parser.add_argument("--draft-cost", type=F, required=True)
    parser.add_argument("--replay-cost", type=F, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "conditional mathematical model, not a speed measurement",
              "assumptions": {key: str(value) for key, value in vars(args).items() if key != "out"},
              "result": analyze(args.acceptance, args.window, args.verify_cost,
                                args.draft_cost, args.replay_cost)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
