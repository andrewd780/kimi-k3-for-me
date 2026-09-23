# Research queue: implementation and remaining gates

Written in [#12](https://github.com/andrewd780/kimi-k3-for-me/pull/12) from the five
proposals first queued in #11, and updated through #14 (2026-09-23); the work it lists
is merged. It replaces #11's ranked queue: the per-machine ordering that queue drew
from the memory ladder does not hold (see [the arithmetic](#what-the-arithmetic-establishes)).
[Research results](research-results.md) records scope, measurements, usage and
remaining work.

| Proposal | Implemented in this follow-up | Gate still open |
|---|---|---|
| Smaller trunk ring | Opt-in `--trunk-rows`: two row buffers, unchanged matrix arithmetic, current-layer vector arena; no cross-layer read in flight | Real-model latency, queue depth and prefill cost |
| Compact Huffman decoder | Four streams, bounded word refill and two-symbol lookup; independent encoder, corruption checks, x86/ARM CI timings on synthetic bytes and four pinned K3 ranges, all KDA `f_a_proj` (0.12% of trunk bytes) | Results in the results note; production container, concurrent reader and resource-cost gate remain separate |
| Predictive expert prefetch | Lead-labelled centroid/ridge diagnostic, synthetic tests only; [ordered gate audit](predictive-prefetch-gates.md) | Closed; do not reopen on a generation capture |
| Fixed-width trunk dictionary | [Eight-range falsifier, per-family samples, FD4B and FD3B SIMD codecs, FDRX row index, rate and contention gates](fixed-width-trunk.md), benchmark-only | Gates 1–3 passed in CI: 99.95% coverage on the eight gate-1 ranges, r = 0.7502, SIMD decode 14.8–26.7 reconstructed GB/s. Gate 4: 3 bits beat 4 by 4.68 points on those ranges and 3.78 byte-weighted over all 23 families; FD3B byte-exact and 9.6–15.3 GB/s in hosted CI. Gate 5: FDRX costs 0.034 points, zero padding. Per-family gate: STOP on `moe.router` and `moe.shared_down`; by the rules fixed before the data r = 0.7329 over all matrices, 5.58 points above per-family Huffman. Contention: worst case above 1 up to B = 4 GB/s in every run on a 4-vCPU VM with the shipped kernels, marginal at 5 (passed in the idle runs, just below 1 in one run with background downloads), below 1 at 6 for streamed input, above 1 at every B to 6 on both hosted runners. Open: a supported reader, and the placement a real row pipeline sees |
| Bounded lookahead | Bounded Jacobi reference, exhaustive toy exactness checks and rational break-even calculator | Useful early acceptance on K3 and measured total work. State rollback is no longer a gate: since #14 `--spec` commits verify sweeps from a KDA log with no replay or snapshot, and a lookahead mode would reuse that |
| Linux async submission | Raw-syscall `io_uring` versus blocking-pool experiment, queue depths 1/2/4/8/16, three runs per arm | A repeatable benefit under concurrent real compute before adding an engine backend |

## What the arithmetic establishes

The small-budget one-position baseline streams about 108.81 GB of trunk and
25.83 GB of experts: 134.64 GB total. The ladder's read column is experts only.
See the [option map](streaming-options.md) for the definitions and checked math.
For independent disk and compute resources, every schedule satisfies

$$t \ge \max(C, B_{\rm required}/D_{\rm disk}).$$

The only split on record is the memory ladder's 8 GB row: 32.69 s per token with
57.3% of it I/O ([memory-ladder.tsv](../data/memory-ladder.tsv)), so 18.73 s of disk
and 13.96 s of everything else. Fully overlapped, that would bound the step at
18.73 s, a 1.75x ceiling. Four things keep that from describing the engine today:

- Both figures are averages over a whole 8-step run whose first step is the prompt
  prefill (99.70 GB of expert reads against 25.83 GB per later step), and the ladder
  is meant as a comparison across budgets, not a per-token speed
  ([PERFORMANCE.md](../PERFORMANCE.md#longer-runs-are-faster)).
- The campaign (captured 2026-07-31) predates the asynchronous trunk reader
  (2026-08-05), so no budget in it had read-ahead.
- It also predates v1.0.0's fused kernels, which cut per-token compute about eightfold;
  with the compute term divided by 8, the same model gives a ceiling of about 1.09x on
  that 124-core host.
- None of it measures the row pipeline, and none of it a small machine.

Smaller buffers, startup, queue depth, shared cores, decompression, expert
dependencies and prompt batches can change those service times or prevent full
overlap. Dividing server compute time by core count does not establish Mac
performance. No Mac or core-limited full-model measurement exists here.

## Corrections to the original proposals

- Matrix-row boundaries can be finer than whole-tensor boundaries. Each output
  row retains the original reduction order. All elementwise vectors must also
  have an explicit lifetime; counting two largest tensors is insufficient.
- Four independent Huffman streams need an encoder layout change. The shelved
  single-stream archive is not compatible. Exponent-plane GB/s and reconstructed
  BF16 GB/s are different units; the new benchmark names its unit.
- Predicting 16 experts at 70% recall costs `16 + 16*(1-.7) = 20.8`
  reads on average with no cache hits, no cancellation and correct guesses
  retained until use. Eviction can cost more; this is not an upper bound.
  `1 + (25.83/134.64)*.3 = 1.0576` is **more total traffic**, not a speedup
  ceiling. Earlier arrival helps only when its scheduling benefit exceeds cost.
- Blocking read threads sleep while waiting for I/O. Thread count is not a count
  of continuously occupied compute cores. `io_uring` can reduce submission/wakeup
  overhead, but does not guarantee a faster drive or free decode resources.
- A byte-reduction factor cannot apply identically at every memory budget:
  resident/pinned trunk bytes are not reread from disk each token.

## Constraints and next decisions

Native gates run in CI, including hosted ARM; the kernel, batching, contention and
MLA timings were taken on a 4-vCPU cloud VM, with conditions in each note. Nothing
runs on Andrew's Macs. The Huffman benchmark fetches four SHA-verified 1 MiB BF16
ranges of the existing immutable sample manifest and the fixed-width gates its eight,
plus 92 1 MiB family samples whose hashes are recorded; nothing downloads a checkpoint.

No standing checkpoint host exists. Full-model timing, core/thread sweeps
and quality evaluation remain blocked on one. Predictive prefetch is closed. This work
does not rent a machine, change precision, prune experts, reduce top-k, or claim
a sub-200-GB model. The negative static-pinning result remains closed.

Use results to choose what comes next: measure row streaming on an already
provisioned checkpoint host; keep codec, lookahead and async-backend
deployment gates explicit. No result here establishes a world-first technique.
