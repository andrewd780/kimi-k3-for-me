# Research queue: implementation and remaining gates

Updated 2026-09-20 in [PR #12](https://github.com/andrewd780/kimi-k3-for-me/pull/12),
following the five proposals in #11. The implementation is under review; this
page does not call it merged. [Research results](research-results.md) records
scope, measurements, usage and remaining work.

| Proposal | Implemented in this follow-up | Gate still open |
|---|---|---|
| Smaller trunk ring | Opt-in `--trunk-rows`: two row buffers, unchanged matrix arithmetic, current-layer vector arena; no cross-layer read in flight | Real-model latency, queue depth and prefill cost |
| Compact Huffman decoder | Four streams, bounded word refill and two-symbol lookup; independent encoder, corruption checks, x86/ARM CI timings on synthetic and pinned K3 ranges | Results in the results note; production container, concurrent reader and resource-cost gate remain separate |
| Predictive expert prefetch | Lead-labelled centroid/ridge diagnostic, synthetic tests only; [ordered gate audit](predictive-prefetch-gates.md) | Closed; do not reopen on a generation capture |
| Fixed-width trunk dictionary | [Eight-range falsifier, SIMD codec and rate gates](fixed-width-trunk.md), one pooled 15-entry dictionary, benchmark-only | Gates 1–3 passed in CI: 99.95% coverage, r = 0.7502, SIMD decode 14.8–26.7 reconstructed GB/s, every run above the 4 GB/s target. Gates 4–5: on committed `f_a_proj` counts 3 bits beats 4 by 4.60 points (FD3B built), 5 bits buys nothing; FDRX row index costs 0.034 points, zero padding. Open: eight-range and per-family CI samples, decode under matmul contention (benchmark built, not measured), supported reader |
| Bounded lookahead | Bounded Jacobi reference, exhaustive toy exactness checks and rational break-even calculator | Useful early acceptance on K3, engine snapshot/replay integration, measured total work |
| Linux async submission | Raw-syscall `io_uring` versus blocking-pool experiment, queue depths 1/2/4/8/16, three runs per arm | A repeatable benefit under concurrent real compute before adding an engine backend |

## What the arithmetic establishes

The small-budget one-position baseline streams about 108.81 GB of trunk and
25.83 GB of experts: 134.64 GB total. The ladder's read column is experts only.
See the [option map](streaming-options.md) for the definitions and checked math.
For independent disk and compute resources, every schedule satisfies

$$t \ge \max(C, B_{\rm required}/D_{\rm disk}).$$

Applying the historical **serialized** 18.63 s I/O + 14.06 s compute split to an
ideal fully overlapping schedule gives a lower bound of 18.63 s, or a conditional
1.75x ceiling. It is not a result of the new reader. Smaller buffers, startup,
queue depth, shared cores, decompression, expert dependencies and prompt batches
can change those service times or prevent full overlap. Dividing server compute
time by core count does not establish Mac performance. No Mac or core-limited
full-model measurement exists here.

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

All native execution is in CI, including hosted ARM. Nothing runs on Andrew's
Macs. The decoder fetches only four SHA-verified 1 MiB BF16 ranges from the
existing immutable sample manifest; it does not download a checkpoint.

No standing checkpoint host exists. Full-model timing, core/thread sweeps
and quality evaluation remain blocked on one. Predictive prefetch is closed. This work
does not rent a machine, change precision, prune experts, reduce top-k, or claim
a sub-200-GB model. The negative static-pinning result remains closed.

Use results to choose what comes next: measure row streaming on an already
provisioned checkpoint host; keep codec, lookahead and async-backend
deployment gates explicit. No result here establishes a world-first technique.
