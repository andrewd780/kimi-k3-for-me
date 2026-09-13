#!/usr/bin/env python3
"""Prove that a shared expert base collapses to THREE matmuls per layer, not 35.

WHY THIS MATTERS
    If experts are stored as W_e = Base + U_e V_e^T, the w1 and w3 bases obviously
    factor out of the top-k loop: the latent input z is identical for all 16 selected
    experts (src/core/k3_ops.c:596), so Base1 @ z and Base3 @ z are computed once.

    The w2 base looks like it CANNOT factor out, because SiTU-GLU puts a nonlinearity
    in between and each expert's activation differs (k3_ops.c:630). The natural reading
    is that Base2 must be replayed 16 times per layer -- which would mean re-reading it
    16x92 times per token, roughly 8.6 GB/token of RAM traffic, and would need a 16-RHS
    GEMM this matvec-only engine does not have.

    It does factor out. src/core/k3_ops.c:640-642 accumulates

        accL += wt[j] * edn[j]

    and the RMSNorm is applied to the AGGREGATE, not per expert (k3.h:102). That makes
    the expert sum linear in w2, so by linearity:

        sum_j wt_j (Base2 + U2_j V2_j^T) act_j
              = Base2 @ (sum_j wt_j act_j)  +  sum_j wt_j U2_j (V2_j^T act_j)

    Accumulate the weighted activations FIRST, then apply Base2 once.

    This script checks that claim numerically against the naive computation. If it ever
    fails, the low-rank container's whole performance argument is void and the design
    must fall back to replaying Base2 per expert.

usage: python3 ideas/verify_factorization.py
"""
from __future__ import annotations

import sys

import numpy as np

# Shapes follow include/k3/k3.h (latent 3584, moe_inter 3072, topk 16), scaled down so
# the test runs instantly. The algebra does not depend on the sizes.
LATENT, INTER, TOPK, RANK = 96, 80, 16, 8


def situ_glu(gate, up, b1=1.0, b2=0.0):
    """Stand-in for k3_situ_glu. The exact nonlinearity is irrelevant to the proof --
    what matters is that it is applied PER EXPERT, between the matmuls."""
    return (gate / (1.0 + np.exp(-b1 * gate)) + b2 * gate) * up


def build(rng):
    return {
        "B1": rng.standard_normal((INTER, LATENT)),
        "B3": rng.standard_normal((INTER, LATENT)),
        "B2": rng.standard_normal((LATENT, INTER)),
        "U1": rng.standard_normal((TOPK, INTER, RANK)),
        "V1": rng.standard_normal((TOPK, LATENT, RANK)),
        "U3": rng.standard_normal((TOPK, INTER, RANK)),
        "V3": rng.standard_normal((TOPK, LATENT, RANK)),
        "U2": rng.standard_normal((TOPK, LATENT, RANK)),
        "V2": rng.standard_normal((TOPK, INTER, RANK)),
    }


def naive(p, z, wt):
    """What the engine does today, with every expert materialised: k3_ops.c:628-642."""
    acc = np.zeros(LATENT)
    for j in range(TOPK):
        w1 = p["B1"] + p["U1"][j] @ p["V1"][j].T
        w3 = p["B3"] + p["U3"][j] @ p["V3"][j].T
        w2 = p["B2"] + p["U2"][j] @ p["V2"][j].T
        act = situ_glu(w1 @ z, w3 @ z)
        acc += wt[j] * (w2 @ act)          # k3_ops.c:641-642
    return acc


def factored(p, z, wt):
    """Three base matmuls per layer, total, regardless of top-k."""
    b1z, b3z = p["B1"] @ z, p["B3"] @ z    # once per layer, z is shared
    agg = np.zeros(INTER)                   # sum_j wt_j act_j
    tail = np.zeros(LATENT)
    for j in range(TOPK):
        gate = b1z + p["U1"][j] @ (p["V1"][j].T @ z)
        up = b3z + p["U3"][j] @ (p["V3"][j].T @ z)
        act = situ_glu(gate, up)            # nonlinearity stays per expert, but is cheap
        agg += wt[j] * act                  # accumulate BEFORE the base matmul
        tail += wt[j] * (p["U2"][j] @ (p["V2"][j].T @ act))
    return p["B2"] @ agg + tail             # Base2 applied exactly once


def macs(rank):
    """MACs per layer per token, counting only the expert matmuls."""
    full = INTER * LATENT
    lowrank = 3 * rank * (INTER + LATENT)
    return {
        "today": TOPK * 3 * full,
        "base2_replayed": 2 * full + TOPK * (lowrank + full),
        "base2_collapsed": 3 * full + TOPK * lowrank,
    }


def main():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _trial in range(25):
        p = build(rng)
        z = rng.standard_normal(LATENT)
        wt = rng.random(TOPK)
        wt /= wt.sum()                      # k3_ops.c:592 normalises the top-k weights
        a, b = naive(p, z, wt), factored(p, z, wt)
        worst = max(worst, np.abs(a - b).max() / np.abs(a).max())

    print(f"trials: 25   worst relative difference: {worst:.3e}")
    ok = worst < 1e-10
    print("FACTORIZATION HOLDS" if ok else "FACTORIZATION FAILED")

    m = macs(RANK)
    print(f"\nMACs per layer per token (top-{TOPK}, rank {RANK}):")
    print(f"  today, every expert materialised   {m['today']:>12,}")
    print(f"  base+delta, Base2 replayed 16x     {m['base2_replayed']:>12,}"
          f"   {m['today']/m['base2_replayed']:5.1f}x less")
    print(f"  base+delta, Base2 collapsed        {m['base2_collapsed']:>12,}"
          f"   {m['today']/m['base2_collapsed']:5.1f}x less")
    print("\nCollapsing Base2 also avoids re-reading it once per selected expert,")
    print("which at real sizes is ~8.6 GB/token of extra traffic, and removes any")
    print("need for a 16-RHS GEMM -- this engine is matrix-vector only (k3.h:265).")

    print("\nTWO THINGS THIS DOES NOT PROVE")
    print("  1. That a rank-r delta reconstructs a real K3 expert well enough.")
    print("     That is ideas/probe_expert_redundancy.py, and it is unmeasured.")
    print("  2. That the reordering is safe in fixed precision. It changes summation")
    print("     ORDER, so it must be checked against the repo's reduction-order")
    print("     contract (k3_ops.c:8-29) before it is trusted in the engine.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
