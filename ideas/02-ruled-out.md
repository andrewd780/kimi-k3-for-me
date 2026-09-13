# Ruled out, with the measurement that ruled it out

Hand this to anyone about to start work on the target. Each of these looks obvious from
the outside and each has a number against it. Several were measured during this review,
not recalled from literature.

---

## Expert pruning — measured, and it fails

**The idea:** keep the most-routed experts, drop the rest. Reaching 200 GB unchanged
means keeping 88 of 896 per layer (9.9%).

**The measurement.** Using `tests/fixtures/expert_trace.bin`, pick the hot set from the
first half of the trace, then score coverage on the held-out half:

| kept/layer | GB @ MXFP4 | coverage on calibration | **coverage held out** |
|---|---|---|---|
| 32 | 51.7 | 56.4% | 21.7% |
| **88** (budget) | **142.1** | 89.0% | **35.4%** |
| 128 | 206.6 | 99.5% | 35.4% |
| 512 | 826.5 | 100% | **35.4%** |

**Held-out coverage plateaus at 35.4% and more storage buys literally nothing.**
Breaking it down by routing rank at the budget point is worse than the average suggests:

```
routing rank:   0    1    2    3    4    5  ...  15
coverage %:    58   53   46   44   42   34  ...  25
```

**The single highest-scoring expert is missing 42% of the time.** Roughly 10 of 16 slots
are empty per layer per token, across all 92 routing layers. Every token, every layer.

**Why it cannot work here:** the router is trained with Quantile Balancing, which
*deliberately flattens expert usage*. The engine's own data shows the consequence —
expert retention pinned at exactly 0.0% from 28 cache slots to 1,344, a 48× increase
buying nothing (`docs/data/memory-ladder.tsv`). **There is no hot subset to prune
toward; the checkpoint was trained not to have one.**

Literature that supports expert pruning is on 8-expert Mixtral-class models at 25–50%
pruning, often task-narrowed or with recovery fine-tuning. Nothing supports 90% pruning
of a fine-grained 896-expert pool, where the design argument is explicitly combinatorial.

---

## "Only 12% of experts are ever used" — the statistic is an artifact

An earlier version of this analysis (and the first brief handed to Astra) treated
"10,010 distinct experts, 12.14% of the pool" as evidence of a small working set. **That
reading is wrong**, and it was corrected by re-parsing the trace directly.

`tests/fixtures/expert_trace.bin` contains **8 forward passes over prefixes of length
5, 6, …, 12** — that is **12 distinct token positions re-prefilled 8 times**, not 68
tokens. Re-prefill is 100.0% deterministic (460/460 `(layer, position)` blocks route
identically), so the 8 passes carry zero extra information.

The true coverage curve:

| distinct positions | distinct experts | % of pool | new per position |
|---|---|---|---|
| 1 | 1,472 | 1.79% | 1,472 |
| 4 | 4,892 | 5.93% | 1,018 |
| 8 | 7,922 | 9.61% | 739 |
| 12 | 10,010 | 12.14% | **680** |

**Still adding ~680 new experts (≈12 GB) at position 12 — nowhere near saturation.** A
Heaps fit (`D(T) = 1610·T^0.756`, R² = 0.993) projects ~77% of the pool at 128 tokens and
saturation by ~500. Uniform random routing would give 16,028 at T=12, so real routing is
only **1.6× more concentrated than random**.

**12.14% is the coverage of a 12-token context, not a property of the model.** Any plan
that budgets storage against it is budgeting against an artifact.

---

## Plain per-expert SVD — measured, dead on arrival

**The idea:** factor each expert as a truncated SVD.

**The measurement**, on real checkpoint bytes (`tests/fixtures/mxfp4.json`,
`layers.1.block_sparse_moe.experts.0.w1`, a 64×3584 slice):

| rank | K3 expert energy | iid random matrix |
|---|---|---|
| 1 | 2.07% | 1.98% |
| 4 | 8.10% | 7.76% |
| 16 | 30.37% | 29.45% |
| 32 | 56.73% | 55.49% |

**Stable rank 48.3 against 50.6 for iid noise** (max 64). Row-normalizing changes
nothing. K3 expert matrices are Marchenko–Pastur-indistinguishable from random.

The budget-feasible rank is 163 of 3072, which on an MP spectrum is **91% relative
Frobenius error**. Getting to even 20% error needs rank ~2,102 — *larger than storing
the dense matrix* (break-even at r=1654).

This is why `01-shared-base-low-rank.md` factors the **residual after a shared base**,
not the expert itself, and why rank is not the lever there either.

---

## Quantization below 4 bits — arithmetically incapable

See `00-the-wall.md` for the full argument. The short version: reaching 200 GB with all
2.72 T expert params needs 0.42 bits/param, and **even deleting the trunk entirely only
relaxes that to 0.588**. Real methods land at 2 bits (737 GB, and needing fine-tuning);
1 bit per weight is 397 GB and destroys the model.

Cheapest variant worth knowing: changing `K3_MXFP4_GROUP` from 32 to 64 is a **10-line
change** (the kernel already takes `group` as a runtime parameter and the `group > 64`
abort at `k3_ops.c:1376-1384` permits exactly 64) and buys about 3%. Useful only stacked
on top of a structural scheme, never alone.

---

## Vector quantization / shared codebooks — misses, and inverts the engine's advantage

**Size:** d=8/K=256 is 1.0 bits/param = 454 GB. d=16/K=256 is 0.5 bits = 284 GB. Still
over, and d=32 into 256 codewords is not a quantizer, it is a lossy hash.

**Sharing the codebook across 896 experts is free and pointless** — it amortizes the
codebook to under 0.001 bits/param, but the rate lives in the *indices*, and every
subvector needs one regardless of who owns the table.

**Worse, it inverts what makes the current kernel fast.** `k3_matmul_mxfp4` streams the
packed row linearly, which is what puts it at the memory floor. A codebook path replaces
that with a data-dependent gather into a ~1 MB fp64 table per row, on a path the code
already calls shuffle-port-bound (`k3_ops.c:1436-1438`), while the engine is already
~45% compute. It also defeats chunk-union prefill (`k3_ops.c:717-808`, measured 1.96×),
because the table is per-token and that loop is expert-major.

---

## Structured sparsity — 3–6× short, and the cheap sparsity is already spent

Ceilings including index overhead: 2:4 → 904 GB, 1:4 → 452 GB, 1:8 → 226 GB. All short
of 200 GB before any quality cost. And the checkpoint is **already 11.6% exact zeros**
(E2M1 encodes 0 directly), so the easy sparsity has been taken.

---

## Cross-layer expert sharing — no

92 layers are not interchangeable, and K3's AttnRes design (block-of-12 residual
snapshots) makes depth position *more* semantically loaded, not less. Sharing only the
*base* across layers would save 1.6 GB → 0.02 GB, which is irrelevant at this scale.

---

## Reducing top-k (16 → 8) — right instinct, wrong axis

Cuts bandwidth and FLOPs per token. Cuts **zero** disk bytes: every expert still has to
exist on disk to be routable. It is a speed lever, not a size lever.

---

## Distillation to fewer experts — the one honest alternative, but it is training

64 experts/layer → 160 GB total (a ~251 B model); 88/layer → 199 GB (~324 B). This is the
only route with a defensible quality argument at this compression ratio, and the only one
that does not depend on an unmeasured correlation.

It is also a full pretraining-scale project, not a post-hoc conversion. If ρ comes back
below 0.85, **this is the remaining option**, and it should be costed honestly rather
than approached as an afternoon's work.
