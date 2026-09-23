# Five proposals: code, experiments and unresolved gates

Updated 2026-09-23. Follow-up to [the research queue](research-queue.md), written in
[#12](https://github.com/andrewd780/kimi-k3-for-me/pull/12) and extended in #13 and
#14, all merged. Native gates run in hosted CI. The batched-kernel timings in
section 1 and the decode-under-contention measurements in section 2b were taken on a
4-vCPU cloud development VM, with conditions beside each. Andrew's machines were not
used. There is no full-checkpoint host and no new full-model seconds/token result.

## 1. Bounded trunk rows: implemented, opt-in

`--trunk-rows` replaces whole-layer residency with two buffers of at most 8 MiB
payload each. While the existing matrix kernel consumes one, a reader fills the
other. The binder uses its existing canonical tensor plan. Matrices become
descriptors; **all** elementwise tensors, including F32 norms, convolutions and
router weights, are copied into a separate current-layer arena.

For output row `j`, the input vector, ordered weight bytes, accumulator lanes
and reduction tree are unchanged. Only the outer loop over independent rows is
partitioned. A buffer is reusable after that matmul returns. Every matrix call
drains its reads, so transitions including layer 92 back to 0 cannot leave an
old read writing into a live buffer. Read errors are sticky and checked before
a layer's output can be emitted.

At released dimensions, the largest current-layer vector arena is **26,519,936
bytes**. Two 8 MiB payload buffers plus alignment slack need **16,793,600 bytes**:
**43,313,536 bytes combined**, plus small reader state. This is an allocation
calculation, not full-model RSS. Tensor metadata, activations, recurrent/KV state,
expert cache and embedding/head remain additional. Allocated sizes appear in JSON.
On a host already holding the packed checkpoint, add to an existing command:

```sh
--trunk /path/to/packed-trunk --trunk-rows --trunk-gb 0.05
```

Use an explicit budget: without one the default `--trunk-gb` of 16 applies, which the
row buffers never approach but the memory plan charges in full. Auto selection and
draft trunks are rejected for this mode. The existing whole-layer reader remains the
default.

**Tradeoff:** every trunk matrix is applied to all positions of a forward in one
pass (`k3_mmw_batch`, through `K3WeightStream.apply_batch`), so multi-token prefill
and speculative verification read each matrix once per batch, as whole-layer
streaming does. That holds at any prompt length: the MoE deduplicates routed experts
over sub-chunks of 64 positions, but its five trunk matrices span the whole batch.
`test_offline_cli.py` checks that 1-, 3-, 8-, 65-, 129- and 130-token forwards read
identical bytes. Before that change prefill reread each matrix for every token.
`--kv-latent` still rereads `kv_b` for every cached position it rebuilds.
Inspect `trunk_matrix_calls` and `trunk_bytes_read`; lower memory does not prove
lower latency. Cross-matrix and cross-layer prefetch are not implemented.
Compressed archives may decode a block repeatedly across small reads. The byte
counter is logical/requested traffic, not a physical-device measurement.

Tests cover two full 93-layer walks, unaligned row starts, ragged row/tile tails,
wraparound, insufficient budget and truncation. The fixture's bytes differ at every
position, so a tile read from the wrong offset, applied to the wrong rows or shifted by
a wrong O_DIRECT prefix changes the products the test compares; the 93-layer build,
whose 257-row matrices span several tiles, runs in `make test` as `test_trunk_rows`.
ThreadSanitizer and ASan/UBSan run it without OpenMP in the research workflow. CLI
checks compare all dumped vocabulary logits and generated IDs across plain/compressed
trunk, full-recompute/incremental/latent-cache modes, and expert pipelining. A Linux
check uses a 64 MiB cgroup with swap disabled. These prove synthetic
mechanism/exactness, not full-model speed.

Sanitizers also exposed the old reader's leaked parsed JSON tree. Tensor names
now keep an explicit owner freed on close/error. The old parallel read loop's
shared error flag now uses an OpenMP reduction.

### Batched kernel timing

The batched matmul is measured on its own with `bench_batch`: weights resident in
RAM (176 MB and 2.35 GB, far beyond cache), no SSD reads, the same bf16 inputs
through `k3_matmul_bf16_batch` and through a loop of one-position `k3_matmul_bf16`
calls. The loop streams the matrix once per position; the batch streams it once per
pass and widens each weight once per register block of positions.

Conditions, 2026-09-23, commit 6cd0668, which carries the exact decode kernels of
[decode-kernels.md](decode-kernels.md): a shared cloud development VM with 4 vCPUs
reported as "Intel(R) Xeon(R) Processor @ 2.80GHz" (AVX-512F/BW/DQ/VL/VNNI), 15 GB
RAM, Linux 6.18, gcc 13.3.0, the Makefile's `-O3 -march=native -ffp-contract=off
-fopenmp` unless stated, `OMP_NUM_THREADS=4 OMP_PROC_BIND=close`. Each run times one
thread, then four. Three runs per arm, interleaved (trunk shape; trunk shape with
blocks forced to 4; trunk shape on the AVX2 baseline; AVX2 with blocks forced to 8;
lm_head shape; three times over); within a run, 15 calls per cell (7 at lm_head's
shape) after one untimed call. Cells give the median of the three run medians and, in
parentheses, the fastest call of all runs; speedup is loop median over batched median.
Before every run the runner waited for the 1-minute load to fall below 1.0; it read
0.24 to 0.87, with no other job running. **In all fifteen runs every batched output
was bit-identical to the per-position loop, at every T and both thread counts.**
Across the 72 cells of the three tables below, the largest of a cell's three run
medians exceeded the smallest by 5.7% at the median and by 42% at worst, in one of the
shortest cells (four threads, T = 1, batched: 5.2 to 7.4 ms); read differences under
about 10% as noise. Every run's cells are in
[batched-matmul-x86_64.json](../measurements/batched-matmul-x86_64.json).

An earlier version of these tables was taken at 9ca67f5, before the exact decode
kernels were merged. Its per-position loop ran the older one-position kernel, 31.2 ms
per call at one thread against 20.9 ms here, so its speedups (1.76x, 2.66x and 3.04x
at T = 2, 4 and 8 on one thread; 3.14x at T = 8 on four; 2.78x and 3.09x for lm_head
blocks of 9 and 16) measured batching against a kernel the engine no longer ships.
The batched calls themselves take the same time as then (81.7 ms at T = 8 on one
thread then, 82.6 now); what shrank is the alternative.

Trunk shape, 12288 x 7168 (a KDA q/k/v/g projection), `-march=native`: AVX-512VL,
so positions share a widened weight in register blocks of 8 (`bench_batch 15 12288 16`).

| Threads | T | Per-position loop, ms | Batched, ms | Batched ms/position | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 20.9 (20.1) | 20.9 (20.0) | 20.90 | 1.00x |
| 1 | 2 | 40.9 (38.5) | 37.2 (36.1) | 18.59 | 1.10x |
| 1 | 4 | 83.6 (77.7) | 49.2 (47.4) | 12.29 | 1.70x |
| 1 | 8 | 160.3 (154.8) | 82.6 (80.3) | 10.32 | 1.94x |
| 1 | 9 | 185.4 (176.6) | 103.6 (97.2) | 11.51 | 1.79x |
| 1 | 16 | 332.1 (311.4) | 167.2 (159.3) | 10.45 | 1.99x |
| 4 | 1 | 6.3 (5.2) | 5.8 (5.0) | 5.82 | 1.09x |
| 4 | 2 | 10.2 (9.7) | 9.3 (9.1) | 4.65 | 1.09x |
| 4 | 4 | 23.3 (20.4) | 14.4 (12.2) | 3.59 | 1.62x |
| 4 | 8 | 45.1 (40.5) | 21.3 (20.3) | 2.66 | 2.12x |
| 4 | 9 | 50.2 (45.7) | 25.8 (24.8) | 2.87 | 1.95x |
| 4 | 16 | 94.5 (83.3) | 49.5 (41.3) | 3.10 | 1.91x |

The same shape on the shipping AVX2 baseline, `ARCH='-mavx2 -mfma'`: blocks of 4.

| Threads | T | Per-position loop, ms | Batched, ms | Batched ms/position | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 21.7 (20.9) | 21.5 (20.6) | 21.49 | 1.01x |
| 1 | 2 | 43.5 (42.0) | 36.9 (35.6) | 18.46 | 1.18x |
| 1 | 4 | 87.3 (83.3) | 50.5 (49.5) | 12.62 | 1.73x |
| 1 | 8 | 174.5 (166.3) | 95.5 (93.1) | 11.93 | 1.83x |
| 1 | 9 | 196.9 (187.1) | 114.9 (109.7) | 12.76 | 1.71x |
| 1 | 16 | 352.6 (338.1) | 190.6 (182.9) | 11.91 | 1.85x |
| 4 | 1 | 6.0 (5.3) | 5.5 (5.3) | 5.54 | 1.08x |
| 4 | 2 | 12.5 (10.8) | 11.1 (9.0) | 5.53 | 1.13x |
| 4 | 4 | 26.2 (21.3) | 13.5 (12.4) | 3.36 | 1.95x |
| 4 | 8 | 47.0 (43.1) | 25.8 (24.0) | 3.23 | 1.82x |
| 4 | 9 | 55.5 (49.1) | 30.5 (28.9) | 3.38 | 1.82x |
| 4 | 16 | 93.4 (86.6) | 52.5 (47.1) | 3.28 | 1.78x |

lm_head shape, 163840 x 7168, `-march=native` (`bench_batch 7 163840 16`). T = 9 is
a `--spec 8` verify sweep, T = 16 one `--score-prompt` / `--tf-check` block.

| Threads | T | Per-position loop, ms | Batched, ms | Batched ms/position | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 275 (247) | 269 (248) | 269.2 | 1.02x |
| 1 | 2 | 551 (496) | 492 (430) | 246.0 | 1.12x |
| 1 | 4 | 1,122 (1,036) | 657 (617) | 164.2 | 1.71x |
| 1 | 8 | 2,212 (2,164) | 1,133 (1,069) | 141.6 | 1.95x |
| 1 | 9 | 2,526 (2,343) | 1,406 (1,346) | 156.2 | 1.80x |
| 1 | 16 | 4,491 (4,179) | 2,230 (2,133) | 139.4 | 2.01x |
| 4 | 1 | 83 (69) | 79 (69) | 79.0 | 1.05x |
| 4 | 2 | 165 (149) | 134 (116) | 67.1 | 1.23x |
| 4 | 4 | 333 (279) | 169 (160) | 42.2 | 1.98x |
| 4 | 8 | 649 (611) | 297 (283) | 37.1 | 2.19x |
| 4 | 9 | 730 (619) | 383 (342) | 42.6 | 1.90x |
| 4 | 16 | 1,319 (1,156) | 581 (556) | 36.3 | 2.27x |

Reading them:

- At T = 1 the batch hands the position to the existing kernel, so its 1.00x to
  1.09x is the noise floor.
- The per-position loop is now close to the memory roof: its weights stream at 8.4 to
  8.8 GB/s on one thread and about 30 on four, against the plain read of about 9.8 and
  36.5 GB/s measured in [decode-kernels.md](decode-kernels.md). The batch streams each
  weight once per pass (2.1 GB/s at T = 8 on one thread, 8.3 on four) and is
  compute-bound there: 17.1 GFLOP/s on one thread and 66.2 on four at T = 8, 1.9x and
  2.1x the loop, scaling 3.9x from one thread to four. So batching saves the repeated
  weight stream, and the widening and arithmetic it still does per position cap the
  gain near 2x at these shapes.
- At T = 8 on the trunk shape a position costs 2.66 ms instead of 5.63 ms on four
  threads (2.12x), 10.32 ms instead of 20.04 on one (1.94x). The `-march=native`
  build's batched call takes 13% less time than the AVX2 baseline's at T = 8 on one
  thread (82.6 vs 95.5 ms) and 18% less on four (21.3 vs 25.8 ms); at T = 4 it is 3%
  less on one thread and 7% more on four, and the same at T = 1. The builds differ in
  instruction set as well as block width, so this does not isolate the block width;
  the forced comparison below does.
- At lm_head's shape, a `--spec 8` sweep (T = 9) takes 383 ms on four threads instead
  of 730 ms (1.90x). It costs 42.6 ms per position against 37.1 at T = 8 because
  9 positions are a block of 8 plus a block of 1. A 16-position block takes 581 ms
  instead of 1,319 (2.27x).
- Arithmetic from these cells, not a model measurement: with the head resident,
  the projection of a 512-position `--score-prompt` would be 32 blocks x 0.581 s,
  about 19 s, instead of 512 x 0.083 s, about 42 s, on four threads. Under
  `--stream-lm-head` the block also reads the 2.35 GB head once instead of 16 times;
  that I/O is not timed here.

**Register block size.** The same runs time each ISA with its default block and with
the other one forced (`-DK3_MM_TB`), trunk shape, batched median in ms; the loop does
not depend on the block. With AVX-512VL a block of 8 takes 7% to 13% less time than 4
at T = 8, 9 and 16 on one thread and at T = 8 and 9 on four, and ties at T = 16 on
four; on the AVX2 baseline a block of 8 takes 2% to 9% more time than 4. That is what
the default of 8 with AVX-512VL and 4 without rests on.

| Build | Threads | T | Block 8 | Block 4 | 8 against 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `-march=native` | 1 | 8 | 82.6 | 92.8 | -11.0% |
| `-march=native` | 1 | 9 | 103.6 | 111.7 | -7.3% |
| `-march=native` | 1 | 16 | 167.2 | 185.0 | -9.6% |
| `-march=native` | 4 | 8 | 21.3 | 24.4 | -12.5% |
| `-march=native` | 4 | 9 | 25.8 | 29.7 | -13.2% |
| `-march=native` | 4 | 16 | 49.5 | 49.5 | 0.0% |
| `-mavx2 -mfma` | 1 | 8 | 104.2 | 95.5 | +9.1% |
| `-mavx2 -mfma` | 1 | 9 | 120.6 | 114.9 | +5.0% |
| `-mavx2 -mfma` | 1 | 16 | 202.3 | 190.6 | +6.2% |
| `-mavx2 -mfma` | 4 | 8 | 26.4 | 25.8 | +2.2% |
| `-mavx2 -mfma` | 4 | 9 | 32.6 | 30.5 | +6.9% |
| `-mavx2 -mfma` | 4 | 16 | 55.9 | 52.5 | +6.5% |

The header line prints `built AVX2` for both x86 builds; it names the vector
kernel, not the register block, which comes from `__AVX512VL__`.

**Not run: full-checkpoint A/B.** The planned comparisons of the pre-batching
build (b0c8b74) against this one, a 64-token `--trunk-rows` prefill (`wall_seconds`,
`trunk_bytes_read`, `trunk_matrix_calls`), `--incremental --spec 8 --gen 64
--stream-lm-head` (`seconds_per_token`, `lm_head_bytes_read`, `spec_accepted`,
identical ids) and a 512-token `--score-prompt` with and without
`--stream-lm-head` (wall time, lm_head bytes, identical `token_nll`), all at
`--trunk-gb 8 --cache-gb 4`, need the packed checkpoint. The VM these timings come
from has none: the BF16 trunk alone is 108.81 GB against 20 GB of free disk. They
remain for a host that holds it.

## 2. Compact Huffman decoder: research prototype

`benchmarks/huf4.h` has four streams, bounded word refills and two-symbol lookup
when two codes fit the 12-bit prefix. Tables occupy 24 KiB. Low bytes stay raw;
high bytes are reconstructed exactly. `tools/bench_huf4.py` independently encodes
canonical codes. Overlong trees are rebuilt with a higher minimum frequency
until valid, never truncated.

The benchmark checks every decoded byte after each timed run and reports three
runs per arm as **reconstructed BF16 GB/s**, including low-plane assembly. Native
sanitizers cover all byte values, odd tails, truncation, invalid padding, invalid
code space and single/pair lookup paths. The transport is benchmark-only, not a
supported archive format.

Inputs are explicitly synthetic bytes and four 1 MiB BF16 ranges identified by
immutable revision, offset and SHA in the committed sample manifest. All four are
slices of one small tensor family, KDA `self_attn.f_a_proj.weight` at layers 1, 12, 24
and 36: 128 x 7168 per KDA layer, 0.127 GB over the 69 KDA layers, 0.12% of the
108.81 GB trunk. Both the ratio and the decode speed depend on the high-byte
distribution, so they describe that family, not the trunk; the fixed-width
[per-family run](fixed-width-trunk.md#per-family-result) shows how much families
differ. No model weights are committed. The real-range payload retains **0.672227**
of the original bytes, excluding archive framing/indexes. This is a sample result, not
a full checkpoint-size guarantee.

Earlier word-refill results are preserved in
[research-word-refill.json](../measurements/research-word-refill.json): real-range
x86 runs were 0.84305, 0.84308, 0.83993 GB/s; hosted ARM ran 0.97730, 0.97566,
1.02588 GB/s. Neither cleared a strict 1 GB/s **every-run** gate. The two-symbol
experiment is recorded separately with its own commit.

The [two-symbol results](../measurements/research-two-symbol.json), from
[CI run 35462000448](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35462000448),
are:

| Hosted CI ISA, the four `f_a_proj` ranges | Original, three GB/s runs | Four streams + pairs, three GB/s runs | Median kernel ratio | Every run >=1 GB/s |
|---|---|---|---:|---|
| x86_64 | 0.41880, 0.41934, 0.41816 | 1.10005, 1.09141, 1.09938 | 2.63x | yes |
| arm64 | 0.38270, 0.35989, 0.36222 | 1.35033, 1.21364, 0.94272 | 3.35x | no |

The synthetic x86 runs also remain just below 1 GB/s. The combined deployment
gate therefore **does not pass**. The decoder stays a research prototype; it is
not wired into inference. These repeated 4 MiB buffer timings are cache-friendly,
exclude disk/encoding and do not compete with model compute. The improvement
over the old kernel is real in these arms; a full-model inference improvement
has not been measured. All repeats, sample hashes and the slower first attempt
are preserved, rather than selecting the fastest run.

A kernel pass alone does not make deployment profitable. For raw bytes `W`,
retained fraction `r`, disk rate `B` and reconstructed decode rate `D`, serial
read+decode helps only if

$$rW/B+W/D < W/B \quad\Longleftrightarrow\quad D>B/(1-r).$$

At `B=3 GB/s` and historical `r=.689`, this requires **9.65 GB/s**, not 1 GB/s.
With independent overlapping stages, decode must exceed 3 GB/s to beat raw disk
time and reach `B/r = 4.35 GB/s` to feed a saturated compressed stream. Sharing
cores with matmuls changes this again. The format/reader, bounded workspace,
parallel decode and concurrent-compute gates remain before engine integration.

## 2b. Fixed-width high-byte dictionary: gates 1–3 passed, per-family STOP on two families

Follow-up to the entropy decoder above, as the [fixed-width note](fixed-width-trunk.md)
specifies: a 4-bit index per BF16 high byte into one pooled 15-entry table plus
an escape code, the low byte raw. `benchmarks/fixed_dictionary.h` decodes with a
scalar reference, SSSE3 `pshufb` and AArch64 NEON `tbl`; `tools/bench_fixed_dictionary.py`
independently encodes. Benchmark-only; not an archive format and not in inference.

Gate 1, the histogram falsifier, ran on all eight dense 1 MiB ranges (seven KDA
`f_a_proj`, one MLA `g_proj`; the job's full report is
[trunk-dictionary-gate1.json](../measurements/trunk-dictionary-gate1.json)): one pooled
dictionary covers **99.954653%** of high bytes, 1,902 escapes in 4,194,304, worst
range 99.941254% (layer 12), payload r = **0.7502** before framing. Gate 2, byte-exact
scalar and SIMD round trips under ASan/UBSan on both ISAs with five negative
controls rejected, passed. Gate 3, the rate gate, from
[CI run 35498176696](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35498176696)
at head `9f0c07f`, three runs per arm, reconstructed BF16 GB/s in `bench_huf4`
units, recorded in
[fixed-dictionary-rate-x86_64.json](../measurements/fixed-dictionary-rate-x86_64.json) and
[fixed-dictionary-rate-arm64.json](../measurements/fixed-dictionary-rate-arm64.json):

| Hosted CI ISA, the eight gate-1 ranges | Scalar reference | SIMD, eight 1 MiB ranges, 24 runs | SIMD, pooled 8 MiB, 3 runs | Every run >=3 GB/s | Every run >=4 GB/s |
|---|---:|---:|---:|---|---|
| x86_64 `ssse3_pshufb` | 1.50–1.58 | min 24.274, median 25.302, max 25.705 | 16.220, 16.263, 16.332 | yes | yes |
| arm64 `neon_tbl` | 1.20–2.38 | min 16.361, median 19.764, max 26.662 | 14.844, 17.473, 19.071 | yes | yes |

`rate_gate.status = PASS` on both ISAs. The slowest of the 54 SIMD runs is
14.844 GB/s. Against the compact Huffman decoder's best run on the same ISA in §2
(1.100 on x86_64, 1.350 on arm64, the four `f_a_proj` ranges), each ISA's slowest FD4B
run is 14.7x faster on x86_64 (16.220) and 11.0x on arm64 (14.844); the pooled medians
give 14.8x and 12.9x. That rate is bought with a ratio premium measured on the same
four ranges: FD4B retains 0.750257 against Huffman's 0.672227, **7.80 points**, which
misses the proposal's six-point limit. (The 6.1 points once quoted here compared
FD4B with the historical r = 0.689 of a different, unidentified 4 MB sample; it is a
lower bound, not the premium.) Over every trunk family, byte-weighted, the 4-bit
premium over per-family Huffman codes is 7.40 points. The 8 MiB pooled figure is
the one to quote; the 1 MiB cases sit in cache. The arm64 spread on identical
input (16.4–26.7 GB/s) is the shared three-core hosted runner; the gate is on
the minimum.

These are warm-buffer, single-thread kernel ceilings with no competing model
compute, as the report's `scope` field states, and §2's break-even arithmetic
applies unchanged: at `B = 3 GB/s` and `r = .75`, serial read+decode needs
`D > 12 GB/s` and the overlapped saturated stream needs `D > 4 GB/s`. The kernel
clears both on the hosted runners. Sharing cores with the matmuls, the row-seekable
layout, the bit width and every tensor family were the gates left open here; the
follow-up below closes each on its evidence, and a supported reader remains. No
full-model speedup, storage-size result or ratio win over Huffman is claimed.

The measurement files were recorded from the CI jobs' stdout `CODEC_REPORT`
lines rather than copied from the artifact zips; every derived field (per-arm
mean, median, minimum, payload and framed ratios, escape counts, gate status)
was recomputed from the primitives and matched exactly before the files were
written. The run's artifacts `dictionary-rate-ubuntu-latest` (ID 10601747318)
and `dictionary-rate-macos-14` (ID 10600672892) hold the originals.

**Follow-up, 2026-09-22 to 2026-09-23** (details in the [fixed-width note](fixed-width-trunk.md);
the hosted results are from [CI run 35845709912](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35845709912)
at head `cb4ab38`).
*Bit-width curve:* the gate tool scores 3/4/5-bit tables, a sign-split variant,
unconstrained Huffman and order-0 entropy bounds. On all eight gate-1 ranges, from
committed counts, 3 bits retains 0.703403 against 0.750227 for 4 bits (+4.68 points),
5 bits 0.8125, Huffman 0.673839, high-byte entropy 0.670093 (whole-BF16 entropy
0.658087 in CI). *FD3B:* a 3-bit decoder with branch-free escape expansion (scalar,
SSSE3, AVX2, NEON) is byte-exact in hosted CI under ASan/UBSan on both ISAs and
decodes the pooled 8 MiB at 9.6 (SSSE3), 15.3 (AVX2) and 11.0 to 13.5 (NEON)
reconstructed GB/s, every run above 4. *Row index:* FDRX makes any whole-row range
decodable for 0.0340 points of matrix bytes per row, or 0.0148 grouped, with zero
padding at K3 widths. *Families:* gate 1 sampled two of 23 matrix families (4.00% of
trunk bytes); the `families` job sampled all 23 (92 MiB,
[trunk-family-gate.json](../measurements/trunk-family-gate.json)). Its gate is a
**STOP**: the byte-weighted pooled table covers under 99% of the router (98.49%) and
the shared-expert down projection (98.88%), and gate 1's own table under 99% of all
three shared-expert projections, 22.35% of trunk bytes. By the rules fixed before the
data, the router takes its own table, the shared-expert down projection stays raw, and
each family takes its better width: **r = 0.7329** over all 108.76 GB of matrices,
payload only, 5.58 points above per-family Huffman codes (3 bits everywhere would be
0.7133, 4 bits 0.7511). *Contention:* a byte-exact benchmark runs one decoder thread
against `k3_matmul_bf16` on the other cores. Re-measured with the shipped kernels, on a
quiet 4-vCPU VM (load 0.10 and 0.46 before the runs, 9 interleaved
repeats), the worst-case contended streamed speedup exceeds 1 up to B = 5 GB/s in every
format, placement and thread count (4 and 2) and is the full 1.333 (FD4B) / 1.420
(FD3B) up to about 4.3 / 4.0 GB/s; at 6 GB/s it fails for streamed input at both
thread counts (0.937 to 0.982, decoder-limited) and passes for cache-hot input. The
hosted x86_64 (Xeon Platinum 8573C) and arm64 legs stay above 1 at every B up to 6.
Pinned layers should stay raw: the resident slowdown is 2.3 to 6.1 in the worst case.
The first VM tables, taken with the kernel it replaced, had the 2-thread case
matmul-limited at B = 6 and resident slowdowns of 2.1 to 3.6. No full-model speedup is
claimed.

## 3. Predictive expert reads: closed

Andrew's follow-up closes this route independently of any future generation
capture. The 5.76% is toy arithmetic, not a measurement: a predictor that fetches 16
experts per layer ahead of time at an assumed 70% recall, with no cache hits, no
cancellation of wrong reads and every correct guess kept until use, reads
`16 + 16 * 0.3 = 20.8` experts' bytes for every 16 used, and experts are 25.83 GB of
the 134.64 GB one-position baseline (19.2%), so whole-token traffic grows by
`0.192 * 0.3 = 5.76%`. Cache hits or cancellation would lower that; eviction of
still-needed experts could raise it. The audit remains the stopping record. No further
capture or predictor work is planned; the next item is the
[fixed-width trunk dictionary falsifier](fixed-width-trunk.md).

The [ordered review audit](predictive-prefetch-gates.md) supersedes the earlier
"calibration ready" description. **Only synthetic tests were run.** There is no
real generation capture to re-score at k=1,2,4, and no measured static-pin or
byte comparison. The prefix-replay fixture cannot supply that evidence.

`tools/routing_probe.py` retains centroid/ridge scoring as a recall diagnostic.
It now requires explicit source layer, source position, decode phase, feature
site, evidence kind and `--lead`. Zero lead is labelled oracle-only; ambiguous
old NPZ files and even deduplicated prefill rows are refused. Prompt-separated
splits, duplicate checks and the bounded dual solve remain. Declared metadata
does not authenticate a real capture. Synthetic tests establish validation and
scoring mechanics only.

The recall-derived read multiplier and 70% gate flag have been removed. The
tool explicitly marks the equal-slot global static-pin null, bytes per decode
token, time available and gain over known-route pipelining as **unmeasured**.
No predictor reader is being built. Uncancelled
wrong predictions can add traffic; true routing remains authoritative. The primary
[Pre-gated MoE paper](https://arxiv.org/html/2308.12066v3) is prior art for earlier
routing, not evidence for a training-free K3 predictor or a CPU speedup.

## 4. Bounded lookahead: reference and cost gate, no new engine mode

`tools/lookahead_gate.py` implements bounded Jacobi proposals and an
accept-exact-prefix reference. Tests enumerate every binary second-order model,
starting prefix, three-token guess and 0–3 Jacobi rounds. All emitted tokens
equal greedy output. This proves the reference, not K3 state rollback or speed.

For window `w`, independent draft agreement `p`, verification cost `V`, proposal
cost `D` and rejection replay cost `R`, normalized to ordinary-step cost (the model
was written when `--spec` replayed after a partial acceptance; since #14 the engine's
rollback costs no replay, so `R = 0` for it):

$$E[N]=\sum_{j=0}^w p^j,\qquad E[T]=V+D+(1-p^w)R.$$

It pays only if `E[N] > E[T]`. Assuming `p=.9,w=2,V=D=R=1` gives
`2.71/2.19 = 1.2374`, and with `R = 0` it gives `2.71/2 = 1.355`; at `p=.5` it loses
either way (`1.75` against `2.75` or `2`). Those are assumptions, not K3 acceptance.
Correlated acceptance needs empirical prefix-length frequencies. Guaranteed
convergence via `w` full Jacobi passes plus verification costs at least `w+1`
sweeps to emit at most `w+1` tokens: no pass-count improvement.

`--spec` itself changed in #14: a verify sweep is tentative, and only the positions
behind emitted ids are committed, from a per-position log of each KDA layer's
recurrence inputs (MLA rows are positional), bit-identical to serial decode with no
replay sweep and no state snapshot. Oracle GATE 4 and the CLI tests hold that
rollback across memory modes. A lookahead mode could reuse it; it still needs cheap
early proposals or reused computation, and real K3 acceptance including rejected-work
I/O. The primary
[Lookahead Decoding paper](https://arxiv.org/abs/2402.02057) establishes prior art,
not an offloaded K3 speed claim.

## 5. io_uring: standalone experiment, no backend replacement

`benchmarks/bench_uring.c` uses Linux raw syscalls without liburing, polling or
registered buffers. It compares `IORING_OP_READ` and a blocking pread pool at
queue depths 1/2/4/8/16, three runs per arm, alternating arm order. A generated
64 MiB file is read with O_DIRECT. Every byte and completion identity/size is
checked. Both arms pay verification cost; setup is untimed. Wall and process CPU
time are reported. Unsupported/denied APIs are reported without changing policy.

The recorded runs have broad timing dispersion. Arms overlap, with no consistent
33%-plus improvement across queue depths. This does not justify an engine backend
change. Process CPU service is much shorter than wall time: sleeping read threads
do not each occupy a full core. Process CPU excludes some kernel-worker cost,
so it is not a complete system-energy measurement. Read completion alone does
not establish overlap with K3 compute. The
[Linux API documentation](https://man7.org/linux/man-pages/man7/io_uring.7.html)
defines the shared queues and out-of-order completion contract used here.

## What remains blocked

Three-run full-model comparisons of row streaming, thread/core count
and speculation require the checkpoint on local NVMe. No such host exists here.
None of this changes precision, restores the invalidated static-hot-set claim,
or makes the roughly 1.5 TB checkpoint fit Andrew's disks. The quality harness
still needs a real-model run before lossy experiments can be evaluated.
