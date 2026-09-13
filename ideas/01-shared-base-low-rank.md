# Shared base + low-rank delta

**Status: the only candidate that survived the council. Blocked on one unmeasured
scalar that costs ~1 GB of download to settle.**

Store each expert as a per-layer shared base plus a per-expert low-rank correction:

```
W_e  =  Base_layer  +  U_e V_e^T           U_e is [rows][r], V_e is [cols][r]
```

## Size — not the constraint

Delta is `3 · r · (3072 + 3584) = 19,968·r` params per expert. Bases are 3.04 B params
over 92 layers: **1.6 GB at MXFP4**, resident and trivial.

| rank | params/expert | vs dense | deltas @ MXFP4 | deltas @ int8 | **total (+ int8 trunk)** |
|---|---|---|---|---|---|
| 8 | 159,744 | 0.48% | 7.0 GB | 13.2 GB | **65.3 GB** |
| 16 | 319,488 | 0.97% | 14.0 GB | 26.3 GB | **72.3 GB** |
| 32 | 638,976 | 1.94% | 28.0 GB | 52.7 GB | **86.3 GB** |
| 64 | 1,277,952 | 3.87% | 56.0 GB | 105.3 GB | **114.3 GB** |
| 128 | 2,555,904 | 7.74% | 111.9 GB | 210.7 GB | **170.2 GB** |

**Every rank fits, with room to spare.** Prefer MXFP4 deltas: they are 4.25 bits, and
`k3_matmul_mxfp4` consumes them with its existing signature, so **no new kernel is
needed**.

## Use rank 8–16. Do not spend rank 128.

**This reverses the advice an earlier draft of this folder gave.** The correction comes
from a measurement, and it matters enough to state plainly.

Write an expert as `W_i = √ρ·C + √(1−ρ)·E_i`, where ρ is the pairwise correlation
between experts in a layer. The delta is the *uncorrelated* part, and total relative
error is `√((1−ρ)(1−E_δ(r)))`. The ρ required to hit a given error is almost
independent of r:

| r | ρ needed for 10% err | for 20% | for 30% |
|---|---|---|---|
| 16 | 0.9898 | 0.9592 | 0.9083 |
| 64 | 0.9892 | 0.9569 | 0.9031 |
| 128 | 0.9884 | 0.9537 | 0.8958 |

Going from r=16 to r=128 is **8× the storage to move the bar by 0.0014.** That is the
signature of a flat residual spectrum, and it was confirmed directly: a real K3 expert
slice (`tests/fixtures/mxfp4.json`, `layers.1.experts.0.w1`, 64×3584) has **stable rank
48.3 against 50.6 for iid noise** — Marchenko–Pastur indistinguishable. Individual
experts are not low-rank, so rank cannot be the lever.

> **If the scheme works, it works at r=8–16 (65–72 GB). If it fails at r=16, raising r
> will not save it.**

## Why it also fixes the speed problem

The base is identical for every expert in a layer, and **all three base matmuls collapse
to one each per layer**, regardless of top-16.

`w1` and `w3` factor out trivially — the latent input `z` is shared across all selected
experts (`src/core/k3_ops.c:596`). `w2` looks like it cannot, because SiTU-GLU puts a
nonlinearity in between and each expert's activation differs (`k3_ops.c:630`).

**It factors out anyway.** `k3_ops.c:640-642` accumulates `accL += wt[j] * edn[j]` and
the RMSNorm is applied to the **aggregate**, not per expert (`k3.h:102`). That makes the
expert sum linear in `w2`:

```
sum_j wt_j (Base2 + U2_j V2_j^T) act_j  =  Base2 @ (sum_j wt_j act_j) + sum_j wt_j U2_j (V2_j^T act_j)
```

Accumulate the weighted activations first, then apply `Base2` once.

**Two independent council seats both concluded `w2`'s base must be replayed per expert.
Both were wrong.** `ideas/verify_factorization.py` proves the collapse numerically —
25 trials, worst relative error **8.58e-16**. Run it before building on this.

It matters for three reasons:
- expert MACs drop ~4× further than the replayed version
- it avoids re-reading `Base2` once per selected expert — **~8.6 GB/token of RAM traffic**
- it removes any need for a 16-RHS GEMM, which this matrix-vector-only engine
  (`k3.h:265`) does not have

Per-token expert reads fall from **25.83 GB to ~0.5 GB at r=32** (51×). At r ≤ 128 the
whole delta set is RAM-resident on a large box, and **K3 stops being a streaming model.**

## The honest part

At r=16 you keep **0.97% of the expert parameters**. Calling that "without losing
quality" is a claim that needs evidence. It rests entirely on ρ ≳ 0.95 — that the 896
experts in a layer are near-duplicates.

There is reason for doubt. LoRA-style deltas are genuinely low-rank when experts were
**upcycled from a shared dense FFN**; K3's 896 fine-grained experts appear trained from
scratch, which predicts low ρ. Supporting prior art exists — MC-SMoE (Li et al., ICLR
2024) merges router-similar experts then low-rank-decomposes the residual, reporting up
to ~80% memory reduction — but on Switch/T5-scale models with few experts and with
fine-tuning. Transfer to an 896-expert 2.78 T model is genuinely uncertain.

## The experiment that decides it

One scalar decides everything:

```
rho = ||mean(W_i)||²_F / mean(||W_i||²_F)
```

`ideas/probe_expert_redundancy.py` computes it on the real pinned checkpoint via the
range-read path the project already ships. **~1 GB of download, an afternoon.**

```bash
python3 ideas/probe_expert_redundancy.py --layer 3 --experts 32 --json findings.json
```

**Decision rule, fixed before the measurement:**

| ρ | verdict |
|---|---|
| **≥ 0.95** | build it — shared base + rank-16 delta, ~72 GB |
| **0.85 – 0.95** | K=1 is not enough; cluster to K=8–16 centroids/layer + rank-16 delta (~84–97 GB) |
| **< 0.85** | **stop.** No post-hoc structural scheme reaches 200 GB. Say so. |

Do **not** respond to a weak ρ by raising the rank. See the table above.

It is disproportionate to keep designing formats around a number this cheap to measure.

## What building it would touch

- `k3_ops.c:607-643` — hoist `B1@z`/`B3@z` out of the top-k loop; accumulate `wt_j·act_j`;
  apply `B2` once after the loop. Same restructure at `k3_ops.c:756-778` for
  `moe_prefill_chunk`, where it composes with the measured 1.96× chunk-union win.
- `k3.h:323-327` — `K3ExpertQ` goes from 6 to 12 pointers. Touches `k3_ops.c:612`, `:759`,
  `k3_cache.c:24-33`, `test_cache.c:121,142,175`.
- `k3_load.h:30-44` + `k3_load.c:15-105` — resolve 12 tensors instead of 6.
- `k3_bind.c:226-231` — bind the per-layer base like the router gate.

**~500 LOC of C, no new kernel** if U/V ship as MXFP4. Best ratio of anything the
council considered.

Three things worth knowing before starting:

1. **It does not break the oracle gate.** The oracle runs the tiny model on the resident
   fp32 expert bank (`k3.h:360-363`, `k3_ops.c:684-685`) and **never executes
   `k3_matmul_mxfp4`.** GATE 1/2/3 are safe. What it does touch: `tests/fixtures/mxfp4.json`,
   `tests/unit/test_expert.c:186-219`, and the golden token stream, which re-baselines.
2. **Budget-invariance survives if the order is unconditional.** "Byte-identical at every
   memory budget" is a claim about the same weights at 12 cache sizes, not about
   quantization. Apply `Base@x` then `+U(Vᵀx)` always, never conditional on residency,
   and never paper over a miss (`k3_ops.c:616-627`).
3. **There is no expert packer in this repo.** `tools/pack_trunk.py:87-89` skips every
   `.block_sparse_moe.experts.` tensor; experts are read in place from 96 original
   shards. Budget **250–400 LOC of Python and a multi-day I/O job**, plus an SVD over
   82,432 experts. This is the most under-estimated line item in the whole plan.

*Alignment freebie:* `19,968·r` bytes is a multiple of 4096 whenever **r is a multiple of
8** (int8) or 4 (bf16), so O_DIRECT slot alignment in `k3_cache.c:301-302` holds by
construction. Pick r ∈ {8, 16, 24, 32}.
