#!/usr/bin/env python3
"""Size-budget calculator for getting the K3 checkpoint under a target.

Every constant here is taken from include/k3/k3.h and verified against the
engine's own figures: one expert is 17.55 MB at MXFP4, which is exactly the
cache slot size in src/cache/k3_cache.c, and the 82,432 experts sum to the
1.45 TB the header documents.

Usage:  python3 ideas/size_model.py [target_gb]
"""
from __future__ import annotations
import sys

# --- architecture, from include/k3/k3.h -------------------------------------
HIDDEN, LATENT, MOE_INTER = 7168, 3584, 3072
N_LAYERS, N_MOE_LAYERS, N_EXPERTS = 93, 92, 896
MXFP4_GROUP = 32

ACTIVE_PARAMS = 56_743_648_000          # trunk + embed + lm_head, always active
EXPERTS_TOTAL = N_EXPERTS * N_MOE_LAYERS  # 82,432

# One expert: w1[moe_inter][latent] + w3[moe_inter][latent] + w2[latent][moe_inter]
EXPERT_PARAMS = 3 * MOE_INTER * LATENT
MXFP4_BITS = 4 + 8 / MXFP4_GROUP        # 4-bit value + one fp8 scale per 32 => 4.25

GB = 1e9


def size_gb(params: float, bits: float) -> float:
    return params * bits / 8 / GB


def banner(text: str) -> None:
    print("\n" + text)
    print("-" * len(text))


def main() -> int:
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0

    expert_params_total = EXPERTS_TOTAL * EXPERT_PARAMS
    experts_now = size_gb(expert_params_total, MXFP4_BITS)
    trunk_bf16 = size_gb(ACTIVE_PARAMS, 16)

    banner("BASELINE (what ships today)")
    print(f"  experts          {EXPERTS_TOTAL:,} x {EXPERT_PARAMS/1e6:.2f}M params "
          f"@ {MXFP4_BITS} bits = {experts_now:8.1f} GB")
    print(f"  per expert       {EXPERT_PARAMS * MXFP4_BITS / 8 / 1e6:.2f} MB"
          f"   (cache slot in k3_cache.c: 17.55 MB)")
    print(f"  active (bf16)    {ACTIVE_PARAMS/1e9:.2f}B params      = {trunk_bf16:8.1f} GB")
    print(f"  TOTAL                                        {experts_now + trunk_bf16:8.1f} GB")
    print(f"  target                                       {target:8.1f} GB")

    banner("THE WALL: quantization only, all 2.72T expert params kept")
    for name, tb in (("bf16", 16), ("int8", 8), ("int4", 4)):
        head = size_gb(ACTIVE_PARAMS, tb)
        left = target - head
        if left <= 0:
            print(f"  trunk {name:>4}: {head:6.1f} GB -- already over target on its own")
            continue
        bits = left * GB * 8 / expert_params_total
        print(f"  trunk {name:>4}: {head:6.1f} GB -> experts get {left:6.1f} GB "
              f"=> {bits:.3f} bits/param")
    print("  For scale: credible post-hoc methods bottom out near 2 bits.")
    print("  => No compression-only route reaches the target. Params must go.")

    banner("OPTION A: shared base per layer + per-expert low-rank delta")
    print("  W_e = Base_layer + U_e V_e^T, computed as Base@x + U@(V^T@x).")
    print("  Base: 3 matrices/layer x 92 layers. Delta: r*(3072+3584) params/matrix.")
    base_params = 3 * MOE_INTER * LATENT * N_MOE_LAYERS
    for bits, label in ((16, "bf16"), (8, "int8")):
        base_gb = size_gb(base_params, bits)
        print(f"\n  deltas @ {label}   (bases {base_gb:.1f} GB, int8 trunk "
              f"{size_gb(ACTIVE_PARAMS, 8):.1f} GB)")
        for r in (8, 16, 32, 64, 128):
            delta_params = EXPERTS_TOTAL * 3 * r * (MOE_INTER + LATENT)
            d = size_gb(delta_params, bits)
            total = d + base_gb + size_gb(ACTIVE_PARAMS, 8)
            flag = "  <-- FITS" if total <= target else ""
            print(f"    rank {r:4d}  deltas {d:7.1f} GB   TOTAL {total:7.1f} GB{flag}")

    banner("OPTION A-grid: rank x delta precision (the real design space)")
    print("  Higher rank at lower delta precision keeps more capacity for the same GB.")
    print(f"  {'rank':>5} {'kept params':>12} {'%cap':>6} " +
          "".join(f"{b:>12}" for b in ("bf16", "int8", "int4")))
    trunk8 = size_gb(ACTIVE_PARAMS, 8)
    for r in (32, 64, 128, 256):
        kept = base_params + EXPERTS_TOTAL * 3 * r * (MOE_INTER + LATENT)
        row = f"  {r:>5} {kept/1e9:>10.1f} B {100*kept/expert_params_total:>5.2f}%"
        for bits in (16, 8, 4):
            total = size_gb(kept, bits) + trunk8
            row += f"{total:>10.0f} GB" + ("*" if total <= target else " ")
        print(row)
    print("  * fits the target. Delta precision is a second lever independent of rank:")
    print("    rank 128 @ int4 keeps 4x the capacity of rank 32 @ int8 for similar GB.")

    banner("OPTION B: expert pruning (keep the top-N most-routed)")
    for bits, label in ((MXFP4_BITS, "MXFP4"), (8, "int8")):
        head = size_gb(ACTIVE_PARAMS, 8)
        per = EXPERT_PARAMS * bits / 8 / GB
        keep = int((target - head) / per)
        print(f"  experts @ {label:>5}: {per*1000:6.2f} MB each -> keep {keep:,} of "
              f"{EXPERTS_TOTAL:,}  ({100*keep/EXPERTS_TOTAL:.1f}% of the pool)")
    print("  Trace precedent: a 68-token trace touched 12.14% of the pool (175.65 GB).")
    print("  That is ONE trace. Coverage growth across diverse prompts is unmeasured.")

    banner("OPTION C: shared codebook / vector quantization")
    head = size_gb(ACTIVE_PARAMS, 8)
    left = target - head
    for bits in (1.0, 1.5, 2.0, 2.5, 3.0):
        g = size_gb(expert_params_total, bits)
        total = g + head
        flag = "  <-- FITS" if total <= target else ""
        print(f"  {bits:>4.1f} bits/param  experts {g:7.1f} GB   TOTAL {total:7.1f} GB{flag}")
    print(f"  Budget available for experts: {left:.1f} GB "
          f"=> {left*GB*8/expert_params_total:.3f} bits/param needed.")

    banner("OPTION D: hybrid -- prune to a kept set, then low-rank the rest")
    trunk8 = size_gb(ACTIVE_PARAMS, 8)
    for keep_frac in (0.02, 0.05, 0.10):
        kept = int(EXPERTS_TOTAL * keep_frac)
        kept_gb = size_gb(kept * EXPERT_PARAMS, MXFP4_BITS)
        rest = EXPERTS_TOTAL - kept
        for r in (16, 32):
            rest_gb = size_gb(rest * 3 * r * (MOE_INTER + LATENT), 8)
            total = kept_gb + rest_gb + trunk8
            flag = "  <-- FITS" if total <= target else ""
            print(f"  keep {keep_frac:.0%} exact ({kept:,}) = {kept_gb:6.1f} GB + "
                  f"rest rank-{r} int8 = {rest_gb:5.1f} GB  TOTAL {total:6.1f} GB{flag}")

    banner("PER-TOKEN I/O AND COMPUTE -- the reason Option A is not just a size win")
    reads = N_MOE_LAYERS * 16                      # 1,472 expert reads per token
    today_io = reads * EXPERT_PARAMS * MXFP4_BITS / 8 / GB
    print(f"  today: {reads:,} expert reads x 17.55 MB = {today_io:.2f} GB read per token")
    print("         (matches the measured 25.83 GB/token in docs/data/memory-ladder.tsv)")
    print()
    print("  With a shared base, Base@x is IDENTICAL for all 16 routed experts in a")
    print("  layer, so it is computed ONCE per layer, not 16 times:")
    print("    sum_e g_e (Base + U_e V_e^T) x  =  Base@x + sum_e g_e U_e (V_e^T x)")
    print()
    base_resident = size_gb(base_params, 8)
    print(f"  bases at int8 = {base_resident:.2f} GB -- small enough to stay RESIDENT")
    for r in (16, 32, 64):
        delta_bytes = 3 * r * (MOE_INTER + LATENT)          # int8
        io = reads * delta_bytes / GB
        macs_today = 16 * EXPERT_PARAMS
        macs_new = EXPERT_PARAMS + 16 * 3 * 2 * r * (MOE_INTER + LATENT)
        print(f"    rank {r:3d}: delta {delta_bytes/1e3:6.1f} KB/expert -> "
              f"{io:5.2f} GB/token read ({today_io/io:5.1f}x less I/O), "
              f"expert MACs {macs_today/macs_new:4.1f}x less")
    print()
    print("  So Option A attacks the 200 GB target AND the 25.83 GB/token bottleneck")
    print("  with the same change. That is the only idea here that does both.")

    banner("SAMPLING COST: what it takes to TEST any of this on real weights")
    per_expert_mb = EXPERT_PARAMS * MXFP4_BITS / 8 / 1e6
    for n in (8, 32, 128, 896):
        label = "one full layer" if n == 896 else f"{n} experts"
        print(f"  {label:<16} {n * per_expert_mb / 1000:6.2f} GB via HTTP range reads")
    print("  tools/remote_model.py read_range() already does pinned-revision range")
    print("  reads against the real checkpoint. No full download needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
