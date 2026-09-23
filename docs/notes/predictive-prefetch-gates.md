# Predictive prefetch: ordered gate audit

2026-09-20, following Andrew's research memo and review of PR #12. This records
what the repository establishes, not a new K3 experiment. No checkpoint host,
hidden-state generation capture or predictive reader exists here. Work on the
predictive reader is **closed**, independently of any future generation capture.
Nothing runs on Andrew's machines. At assumed 70% recall, uncancelled misses add
about 5.76% to the stated whole-token traffic baseline; experts are only 19.2%
of that baseline. Andrew's follow-up closes this research route and directs the
next experiment to the [fixed-width trunk falsifier](fixed-width-trunk.md).
The audit below is retained as the historical gate/accounting record, not a
request to collect a generation trajectory or reopen predictive prefetch.

## 1. Oracle and generation-trajectory gate: blocked

**There is no real calibration to re-score.** The only inputs previously run
through `tools/routing_probe.py` were the artificial cases in
`tests/test_routing_probe.py`. The previous NPZ schema named a target layer but
did not record the source layer or execution phase. Consequently it established
neither positive lead time nor the absence of oracle features. Deduplicating its
rows could not turn prefill into decode. The old `2 - recall` output and
`clears_70_percent_recall_gate` flag are removed: neither measured reads nor
justified proceeding to deployment.

Define `h_L` as the full-width input passed to the router in `k3_moe` at layer
L, **before** the latent down-projection. At that point the true router can run;
using `h_L` supplies no cross-layer lead. This does not claim the router itself
has zero compute cost. A different feature site needs its own availability
definition, rather than silently sharing the name `h_L`.

| Feature for target L | Lead in layers | Real held-out recall | Decision |
|---|---:|---|---|
| `h_L` | 0 | unmeasured | Oracle diagnostic only; cannot pass |
| `h_(L-1)` | 1 | unmeasured | Blocked on generation capture |
| `h_(L-2)` | 2 | unmeasured | Blocked on generation capture |
| `h_(L-4)` | 4 | unmeasured | Blocked on generation capture |

Unmeasured is neither near-baseline nor successful. Unavailable exact routing
also does not prove that earlier features carry no statistical information.
It is not possible to
recover these curves from `tests/fixtures/expert_trace.bin`: that file contains
only expert IDs, from eight prefix recomputations of lengths 5 through 12.
The 68 position evaluations contain 12 distinct positions. Its final-pass
transpose is a derived replay, not an observed incremental generation trace.
See [the trace audit](../EXPERT_PROFILES.md#what-the-existing-trace-actually-records).

The former gate would have required an actual autoregressive generation run
after prompt prefill, with the immutable checkpoint/engine revisions, flags,
tokenizer, original prompt identity, emitted token sequence, seed/sampling
settings, absolute positions, execution phase and feature hook in a manifest.
Use the engine's generated tokens, not teacher-forced tokens or recomputed
prefixes. Record each new decode position once, all relevant source-layer
vectors, true routed IDs and event timestamps. Exclude the prompt-prefill pass
that emits the first output token. Layer distance alone is not a time budget.

All continuations of the same original prompt must remain in one split. Freeze
calibration/validation/held-out prompt groups and tuning choices before scoring.
Compare k=1,2,4 on the **same target layers, positions and held-out prompts**;
with this router-input convention K3's dense layer 0 is not a source, so the
common target set begins at layer 5. Unsupported early targets still count as
ordinary demand reads in any whole-token byte result. Report per-prompt and
per-layer curves as well as aggregate recall, with uncertainty across prompts.
If positive-lead arms are indistinguishable from the calibration-only static
null, stop; do not proceed on the k=0 result or a fixed 70% threshold.

The existing probe is now a **recall diagnostic only**. It requires an explicit
`--lead 0|1|2|4`, matching `source_layer` and `source_position`, `phase="decode"`,
`feature_site="router_input"`, and declared `evidence_kind` (`synthetic` or
`generation_capture`). Its remaining arrays are documented in the tool. A
future eligible capture can be checked, for example, with:

```sh
python3 tools/routing_probe.py generation-k1.npz --lead 1 \
  --train-prompts calibration-a,calibration-b --out recall-k1.json
```

Those filenames are placeholders; no such K3 capture is committed. The tool
rejects missing/mismatched lead metadata and prefill even when positions are
unique. It labels k=0 as an oracle and always reports `gate_status=not_evaluated`:
self-declared metadata cannot authenticate a capture, and the equal-slot null,
byte, timing and pipeline comparisons are **not implemented by this probe**.
The synthetic tests only check this contract and the scoring mechanics.

## 2. Equal-slot static-pin null: decode comparison blocked

Rank `(layer, expert)` keys globally using calibration requests only, resolving
ties as the engine's profile tool does. Pin the first N lazily; predict nothing.
Use the same slot count, checkpoint format, starting cache state, prefill warm
state policy and known-route pipeline for all arms. Charge compulsory loads and
all predictor/state/buffer memory. Do not rank the test data or replace this
null with a per-layer top-k classifier. Choose N on calibration/validation only;
keep at least `topk+1` evictable slots as the current engine requires.

The historical shipping 8 GB configuration has **28 slots**, at most **11 pins**.
Here is the existing null evidence, verified against
[expert-profile-replay.json](../measurements/expert-profile-replay.json).
**Every row below is the old seven-position derived replay, not decode data.**

| Slots | Arena GB, excluding the rest of the process | Pins | Loads | Logical bytes / derived position | Reduction vs same-slot LRU |
|---:|---:|---:|---:|---:|---:|
| 28 | 0.491553 | 0 | 10,304 | 25,829,572,608 | 0% |
| 28 | 0.491553 | 11 | 10,259 | 25,716,768,768 | 0.4367% |
| 615 | 10.796605 | 0 | 10,304 | 25,829,572,608 | 0% |
| 615 | 10.796605 | 598 | 9,793 | 24,548,622,336 | 4.9592% |
| 1,344 | 23.594533 | 0 | 10,304 | 25,829,572,608 | 0% |
| 1,344 | 23.594533 | 1,327 | 9,289 | 23,285,219,328 | 9.8505% |

Arithmetic: arena = slots × 17,555,456; bytes/position = loads × 17,547,264 / 7.
The larger rows are context, **not alternatives at 8.24 GB process RSS**. The
0.44% result remains negative; static pinning is not being revived as a solution.
A future predictor must beat this policy on new, matched decode trajectories,
starting at 28 slots. It currently has no measured comparison at any capacity.

Slot accounting matters to scheduling: if 11 distinct pins are resident and
16 other current-layer experts are protected, a 28-slot arena has only **one**
spare slot. Speculative reads cannot assume another 16-expert buffer is free.
Pins overlapping the current set and slots released after consumption change
that count and must be modeled explicitly.

## 3. Whole-expert byte gate: blocked, accounting specified

One expert's payload is **17,547,264 bytes** across three matrices, each with a
packed-weight and scale tensor. The review's "three preads" is not the current
loader contract: `src/io/k3_load.c` coalesces all six contiguous tensors into
**one range**, otherwise it reads **six tensor ranges**. Short-read loops,
alignment and compressed blocks can change the actual syscall/physical-byte
count. Cache admission and the current predictor proposal still use a **whole
expert**. Nearly correct IDs earn no partial credit.

For each arm report expert logical bytes, trunk logical bytes and their sum
**per incremental decode step**, with the step count and prefill separately.
These are workload counters; measure actual aligned/compressed/device bytes
separately rather than renaming payload counters as SSD traffic. Include wrong
guesses, reloads after eviction, partial/cancelled transfers and any predictor
weight reads. Drain or attribute reads crossing the measurement boundary.
The existing CLI's whole-run totals divided by generated outputs can include
prefill, so they cannot supply this decode-only metric without phase accounting.

Under the explicit toy assumptions of no cache hits, k guesses, all wrong
guesses fully read, and every correct guess retained until use:

$$B_{expert}=92\,16\,17{,}547{,}264=25{,}829{,}572{,}608\text{ bytes/step},$$
$$B_{predicted}=B_{expert}(2-r).$$

At assumed recall r=0.7 this is **33.5784443904 GB**, **7.7488717824 GB extra**
per step. Adding the approximately 108.81 GB trunk gives 142.3884443904 GB,
about **5.76% more total traffic**. This is conditional arithmetic, not an
upper bound with eviction, a measurement, or a speedup. Cancellation and real
residency require a schedule; recall by itself cannot determine bytes.

Do not evaluate this as a prefill lever. With independent uniform top-16 sets
among 896 experts, expected coverage is `1-(1-16/896)^n`: **83.50% at n=100**,
**99.55% at n=300**. Balanced marginal use does not establish independence, so
these are illustrative calculations, not measured K3 coverage. Deduplicating
prefill requests does not make this a generation trajectory.

## What prediction must improve upon

[Known-route pipelining](expert-pipeline.md) is already implemented and needs
no learned predictor. It is **opt-in**, not enabled by default. Its mathematical
equal-cost example has 16 experts with read time i and compute time c:

$$T_{serial}=16(i+c),\quad T_{pipe}=i+c+15\max(i,c).$$

For i=c, the ideal expert-stage ratio is 32/17 = **1.8824x**. Even having all
experts ready before that stage would then improve 17i to 16i: **1.0625x
additional expert-stage headroom**, assuming all earlier reads fit, finish and
do not delay other work. This is not a general K3 speed limit or a benchmark;
different service ratios and contention change it.

Likewise 134.64/108.81 ≈ **1.2374x** is the disk-service ratio if all expert
bytes vanished. Prefetch does not make them vanish. It can reduce exposed
latency by overlap; the byte ratio is not a universal latency bound. Comparing
1.24x whole-I/O and 1.88x expert-stage figures does **not** prove dominance.

The baseline must include known-route pipelining and applicable existing read
ordering. PR #12 also implements next-row double buffering for the trunk
(`--trunk-rows`); cross-matrix/layer row prefetch remains absent. All are
predictor-free alternatives. Whether predictive prefetch adds enough benefit
requires the three gates above, then matched three-run timings including
predictor cost and I/O contention. No such result exists. Bank the exact
mechanisms and stop predictor deployment work here.

The memo's latent-space findings motivate careful controls, not a transfer
result for K3 routing. Identity-regularized hidden-to-hidden maps are square;
an identity penalty cannot simply be copied onto a 7,168-to-896 classifier.
No new probe family, generation capture or predictive reader is planned. The
route remains closed; the next research item attacks trunk decode rate.
