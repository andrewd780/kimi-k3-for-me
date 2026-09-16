# Research queue: what is left to build without the checkpoint, ranked

Written 2026-09-16 from a survey of the repository's own option map and of 2024 to
2026 literature on expert offloading, lossless weight coding, training-free
speculative decoding and I/O mechanics. Everything here is exact (bitwise identical
output) and buildable and testable on the synthetic model. Anything that would change
the output was not queued; it needs the quality harness first, and that needs a
checkpoint host.

## The arithmetic that orders the queue

Per generated token at an 8 GB budget the engine reads 134.6 GB: 108.8 GB of bf16
trunk (81%) and 25.8 GB of MXFP4 experts (19%). On the rented 124-core host at 8 GB
the token took 32.69 s: 18.63 s of disk and 14.06 s of compute, serialized, because at
that budget the trunk ring holds one slot and read-ahead is lost
([PERFORMANCE.md](../PERFORMANCE.md)).

| Lever | 124 cores, 8 GB (measured split) | 8-core laptop, 2 GB/s SSD (estimate) |
|---|---|---|
| Today | 18.6 + 14.1 = 32.7 s | 67 + ~218 = ~285 s |
| Restore overlap only | max(18.6, 14.1) = 18.6 s, **1.75x** | max(67, 218) = 218 s, **1.31x** |
| Trunk 1.45x smaller only | 14.0 + 14.1 = 28.0 s, 1.17x | 50 + 7 (decode) + 218 = 275 s, 1.04x |
| Both | ~14.1 s, 2.3x | 218 s, 1.31x |

The laptop column scales the measured 14.06 s of compute by core count, which is an
estimate, not a measurement; the status page already calls a small machine
compute-bound for that reason. The conclusion does not depend on the exact figure:
overlap pays on every machine, fewer trunk bytes pays only where disk time exceeds
compute time, and once overlap exists the byte saving is hidden behind compute
unless the disk is the longer term.

## Ranked

### 1. Tensor-granular trunk ring (the asymmetric ring, done at tensor size)

Today a ring slot must hold the largest layer that streams through it. At the 8 GB
floor nothing is pinned, so the slot must hold the dense layer 0 at 2.34 GB, and a
second slot costs 2.37 GB, which is why the laptop preset falls back to one slot
(`src/io/k3_trunk.c`, the comment above `RING_WANT`). No single trunk tensor exceeds
0.49 GB (the dense MLP matrices, 33792 x 7168 at bf16). Streaming the trunk one
tensor at a time, in the fixed order the layer consumes them, lets two slots cost
about 1 GB in total instead of 2.37 GB for one, so read-ahead survives at 8 GB and
the floor drops by roughly 1.4 GB at the same time.

- Exact: the kernels, the reduction order and the bytes are untouched; only the
  lifetime of each tensor's buffer changes. The option map's warning stands: every
  consumer of a tensor must finish before its slot is reused, and small tensors used
  more than once in a layer (norms, router, biases) stay resident rather than ringed.
- Gate: the oracle gates at a tiny memory cap, the allocator gate, ThreadSanitizer,
  and the CLI suite byte-identical with the feature on and off. The 93-layer
  wraparound is a drain at the seam, tested explicitly.
- Effort: large. It touches the trunk reader, the binder every layer type calls, and
  the memory plan. Staged: first tensor-sized slots with the existing two-slot
  ring, then the prefetch cursor across layers.
- Ceiling: 1.75x on the measured host at 8 GB, about 1.3x on a compute-bound laptop,
  plus the memory it frees for expert cache.

### 2. Small multi-stream Huffman decoder for the byte-plane trunk

The shelved prototype ([compressed-trunk.md](compressed-trunk.md),
`k3_huf.h.shelved`, `huf_encode.py.shelved`) already round-trips real trunk bytes
at 1.45x with a single 12-bit-table decoder at 0.31 GB/s per core. Its own note names
the unlock: four interleaved bitstreams per stripe for instruction-level parallelism,
or a double-symbol table, in a decoder that stays a few hundred lines. Reference
FSE/Huff0 reached 1.72 GB/s on the same bytes, so the target of 1 GB/s per core is
known to be physically available. Prior art with the same ratio: ZipNN (Nov 2024,
arXiv:2411.05239, CPU), DFloat11 (Apr 2025, arXiv:2504.11651, GPU only).

- Exact: the decoded bytes are the checkpoint's bytes.
- Gate: round-trip on synthetic bf16 tensors with the measured exponent histogram,
  and a decode-throughput measurement with the repository's three-run rule; the
  feature ships only if the decoder clears 1 GB/s per core on the CI measure host
  and locally.
- Effort: medium. Encoder format can stay; the decoder and the trunk reader's
  compressed path are new.
- Ceiling: 1.34x fewer bytes at every budget; wall-clock 1.17x on the measured host
  without overlap, near nothing on a compute-bound laptop. Ranked second for that
  reason, not for difficulty.

### 3. Next-layer expert prefetch from the current hidden state

Predict layer L+1's experts while layer L computes and issue cancelable low-priority
reads; true routing still decides what is computed, so it is exact by construction.
The option map's own worked example, 70% recall on top-16, gives 1.3x expert traffic
for earlier arrival: at most a 1.06x change in total bytes because experts are 19%
of the per-token reads. The predictor needs real routing data to calibrate, which
the replay fixture only partly supplies. Prior art: Pre-gated MoE (ISCA 2024,
arXiv:2308.12066), MoE-Infinity (arXiv:2401.14361). Large effort, small ceiling:
queued, not recommended next.

### 4. Bounded lookahead verification, flagged as a trap

Extend the shipped `--spec` n-gram drafter with Jacobi-style candidate generation
(Lookahead Decoding, ICML 2024, arXiv:2402.02057) so drafts exist without a matching
suffix in the history. Exact by the same accept-exact-prefix rule. The engine's own
comment in `src/cli/k3_run.c` records why this is dangerous here: an eager drafter
measured 0.91x on code, because partial acceptances pay a full replay sweep, and the
expert bytes of a verification batch grow with the union of routes across it. Only
worth building with a very small window and the same evidence gating `--spec`
already uses, and its acceptance rate is unmeasurable without a checkpoint host.

### 5. io_uring read submission

Replace the blocking pread pool with a single-thread submission ring on Linux, raw
syscalls, no library. Exact, and its throughput-versus-core-count curve is measurable
on any Linux box with no model. It is an enabler for item 1 on small-core machines,
where blocking reads occupy cores the compute needs, not a lever on its own; queued
behind item 1.

## Not queued, with the reason

- Prompt-lookup and n-gram drafting: already shipped as `--spec`.
- Vectorized trunk kernels: already done, AVX2 and NEON with double accumulation in
  `k3_matmul_bf16`, exact by construction.
- Whole offloading systems (PowerInfer-2, MoE-Lightning, kTransformers, Fiddler,
  HOBBIT): GPU-centric; only their prefetch sub-idea transfers (item 3).
- Anything lossy (pruning, int8 or int4 trunk, top-k reduction): closed in
  [STATUS.md](../STATUS.md) until the quality harness has run on the real model.

## Recommendation

Build item 1 first, in two stages, each gated bitwise on the synthetic model. Item 2
second, only if its decoder clears the speed bar. Items 3 to 5 stay recorded.
