# ideas/ — getting the checkpoint under 200 GB

Working notes on one question: **can the 1,559.9 GB K3 checkpoint fit in 200 GB without
losing quality?**

Proposals and measurements, not decisions. Nothing here is built. The one candidate that
survived review is blocked on a single unmeasured number, and there is a script here to
measure it for about a gigabyte of download.

## Read in this order

| file | what it settles |
|---|---|
| [`00-the-wall.md`](00-the-wall.md) | why no amount of compression reaches 200 GB — read first |
| [`01-shared-base-low-rank.md`](01-shared-base-low-rank.md) | the only surviving candidate, and the experiment that decides it |
| [`02-ruled-out.md`](02-ruled-out.md) | what was tried and killed, with the measurement that killed it |

| script | what it does |
|---|---|
| [`size_model.py`](size_model.py) | budget calculator — every size claim here is reproducible |
| [`probe_expert_redundancy.py`](probe_expert_redundancy.py) | **the decisive experiment.** ~1 GB of download |
| [`verify_factorization.py`](verify_factorization.py) | proves the shared base collapses to 3 matmuls per layer |
| [`size_model_output.txt`](size_model_output.txt) | saved calculator output if you'd rather not run it |

## The short version

**Compression cannot get there.** Keeping all 2.72 T expert parameters inside 200 GB
needs 0.42 bits per parameter, and **even deleting the trunk entirely only relaxes that
to 0.588**. Below one bit you cannot give each weight its own code, so codes must be
shared between weights — which is parameter removal, not compression. Best honest
compression-only result: ~1,455 GB.

**Pruning cannot get there either, and this was measured rather than assumed.** Held-out
expert coverage plateaus at 35.4% no matter how much you keep: at the budget point the
single highest-scoring expert is missing 42% of the time. The router is trained with
Quantile Balancing, which deliberately flattens expert usage — **there is no hot subset
to prune toward.**

**One candidate survives.** Store each expert as a shared per-layer base plus a low-rank
delta. At rank 16 that is ~72 GB total, and because the base is shared across all 16
routed experts it also cuts per-token disk reads from 25.83 GB to ~0.5 GB. It is the only
idea that addresses the size target and the speed bottleneck with one change — and it
would stop K3 being a streaming model at all.

It rests entirely on ρ, the pairwise correlation between experts in a layer, which
nobody has measured. `probe_expert_redundancy.py` measures it. The decision rule is
fixed in advance, including the outcome that kills the idea.

## Two cheap wins worth banking regardless

Neither reaches the target; both are nearly free.

1. **int8 trunk — saves 56.8 GB.** The container already exists (`tools/int8_trunk.py`,
   measured 90.9% teacher-forced agreement). Built, never shipped.
2. **Entropy-code the E8M0 scale plane — an estimated 48 GB, losslessly.** The only
   entropy coding that survives this project's decode-speed test: the scale plane is
   1.03 MB per expert against 17.55 MB of nibbles, so a token decodes 1.52 GB of scales
   rather than 25.83 GB.
   **Unverified:** `tools/bench_lossless.py:77` filters to `.weight_packed` and has never
   sampled `.weight_scale`. One-line change to find out.

## Corrections this review made to earlier work

Worth reading if you saw an earlier version of any of this.

- **"Only 12% of experts are ever used" is an artifact.** The trace is 12 distinct token
  positions re-prefilled 8 times, not 68 tokens. Coverage is still adding ~680 new
  experts per position at position 12 and projects to ~77% of the pool by 128 tokens.
  Any plan budgeting against 12% is budgeting against a measurement artifact.
- **Rank is not the lever — ρ is.** An earlier draft recommended rank 128. That was
  wrong: required ρ is almost independent of r, and going r=16 → 128 is 8× the storage
  to move the bar by 0.0014. Use r=8–16, and if it fails there it fails.
- **The `w2` base does collapse.** Two council seats independently concluded it must be
  replayed per expert. `verify_factorization.py` proves otherwise at 8.58e-16, which is
  worth ~8.6 GB/token of avoided traffic and removes the need for a 16-RHS GEMM the
  engine does not have.
- **The oracle gate is not at risk.** It runs on the resident fp32 expert bank and never
  executes `k3_matmul_mxfp4`, so an expert-format change cannot break GATE 1/2/3.

## Ground rules these notes hold to

- Every size claim is reproducible with `size_model.py`, whose constants come from
  `include/k3/k3.h` and agree with the engine's own figures (one expert = 17.55 MB, the
  cache slot size in `src/cache/k3_cache.c`).
- Measured, estimated, and assumed are labelled differently.
- Decision rules were written down *before* the measurement, including the outcomes that
  kill the idea.
- Weight-space error is treated as a proxy for quality, never as quality.
