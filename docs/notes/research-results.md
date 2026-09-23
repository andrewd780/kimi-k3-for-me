# Five proposals: code, experiments and unresolved gates

Updated 2026-09-20. Follow-up to [the research queue](research-queue.md), in
[PR #12](https://github.com/andrewd780/kimi-k3-for-me/pull/12). All native runs are
hosted CI. Andrew's machines were not used. There is no full-checkpoint host and
no new full-model seconds/token result.

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

Use an explicit budget; auto selection and draft trunks are rejected for this mode.
The existing whole-layer reader remains the default.

**Tradeoff:** the current attention/MLP loops are token-major. Multi-token prefill
can reread each matrix for every token, whereas whole-layer streaming keeps it
during that batch. `--kv-latent` can reread `kv_b` for every cached position too.
Inspect `trunk_matrix_calls` and `trunk_bytes_read`; lower memory does not prove
lower latency. Cross-matrix and cross-layer prefetch are not implemented.
Compressed archives may decode a block repeatedly across small reads. The byte
counter is logical/requested traffic, not a physical-device measurement.

Tests cover two full 93-layer walks, unaligned row starts, ragged row/tile tails,
wraparound, insufficient budget and truncation. ThreadSanitizer and ASan/UBSan
run the lifetime test without OpenMP. CLI checks compare all dumped vocabulary
logits and generated IDs across plain/compressed trunk, full-recompute/incremental/
latent-cache modes, and expert pipelining. A Linux check uses a 64 MiB cgroup with
swap disabled. These prove synthetic mechanism/exactness, not full-model speed.

Sanitizers also exposed the old reader's leaked parsed JSON tree. Tensor names
now keep an explicit owner freed on close/error. The old parallel read loop's
shared error flag now uses an OpenMP reduction.

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
immutable revision, offset and SHA in the committed sample manifest. No model
weights are committed. The real-range payload retains **0.672227** of the original
bytes, excluding archive framing/indexes. This is a sample result, not a full
checkpoint-size guarantee.

Earlier word-refill results are preserved in
[research-word-refill.json](../measurements/research-word-refill.json): real-range
x86 runs were 0.84305, 0.84308, 0.83993 GB/s; hosted ARM ran 0.97730, 0.97566,
1.02588 GB/s. Neither cleared a strict 1 GB/s **every-run** gate. The two-symbol
experiment is recorded separately with its own commit.

The [two-symbol results](../measurements/research-two-symbol.json), from
[CI run 35462000448](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35462000448),
are:

| Hosted CI ISA, real K3 ranges | Original, three GB/s runs | Four streams + pairs, three GB/s runs | Median kernel ratio | Every run >=1 GB/s |
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

## 2b. Fixed-width high-byte dictionary: gates 1–3 passed

Follow-up to the entropy decoder above, as the [fixed-width note](fixed-width-trunk.md)
specifies: a 4-bit index per BF16 high byte into one pooled 15-entry table plus
an escape code, the low byte raw. `benchmarks/fixed_dictionary.h` decodes with a
scalar reference, SSSE3 `pshufb` and AArch64 NEON `tbl`; `tools/bench_fixed_dictionary.py`
independently encodes. Benchmark-only; not an archive format and not in inference.

Gate 1, the histogram falsifier, ran on all eight dense 1 MiB ranges: one pooled
dictionary covers **99.954653%** of high bytes, 1,902 escapes in 4,194,304, worst
range 99.941254% (layer 12), payload r = **0.7502** before framing. Gate 2, byte-exact
scalar and SIMD round trips under ASan/UBSan on both ISAs with five negative
controls rejected, passed. Gate 3, the rate gate, from
[CI run 35498176696](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35498176696)
at head `9f0c07f`, three runs per arm, reconstructed BF16 GB/s in `bench_huf4`
units, recorded in
[fixed-dictionary-rate-x86_64.json](../measurements/fixed-dictionary-rate-x86_64.json) and
[fixed-dictionary-rate-arm64.json](../measurements/fixed-dictionary-rate-arm64.json):

| Hosted CI ISA, real K3 ranges | Scalar reference | SIMD, eight 1 MiB ranges, 24 runs | SIMD, pooled 8 MiB, 3 runs | Every run >=3 GB/s | Every run >=4 GB/s |
|---|---:|---:|---:|---|---|
| x86_64 `ssse3_pshufb` | 1.50–1.58 | min 24.274, median 25.302, max 25.705 | 16.220, 16.263, 16.332 | yes | yes |
| arm64 `neon_tbl` | 1.20–2.38 | min 16.361, median 19.764, max 26.662 | 14.844, 17.473, 19.071 | yes | yes |

`rate_gate.status = PASS` on both ISAs. The slowest of the 54 SIMD runs is
14.844 GB/s, against the compact Huffman decoder's best real-range runs of
1.100 (x86_64) and 1.350 (arm64) in §2: a 13x–15x kernel-rate difference bought
with a 6.1-point ratio premium (0.750 versus 0.689). The 8 MiB pooled figure is
the one to quote; the 1 MiB cases sit in cache. The arm64 spread on identical
input (16.4–26.7 GB/s) is the shared three-core hosted runner; the gate is on
the minimum.

These are warm-buffer, single-thread kernel ceilings with no competing model
compute, as the report's `scope` field states, and §2's break-even arithmetic
applies unchanged: at `B = 3 GB/s` and `r = .75`, serial read+decode needs
`D > 12 GB/s` and the overlapped saturated stream needs `D > 4 GB/s`. The kernel
clears both on the hosted runners. Whether it does so while sharing cores with
the matmuls, the row-seekable layout and its padding cost, the 5-bit variant,
and a supported reader are the open gates. No full-model speedup, storage-size
result or ratio win over Huffman is claimed.

The measurement files were recorded from the CI jobs' stdout `CODEC_REPORT`
lines rather than copied from the artifact zips; every derived field (per-arm
mean, median, minimum, payload and framed ratios, escape counts, gate status)
was recomputed from the primitives and matched exactly before the files were
written. The run's artifacts `dictionary-rate-ubuntu-latest` (ID 10601747318)
and `dictionary-rate-macos-14` (ID 10600672892) hold the originals.

**2026-09-22 follow-up** (details in the [fixed-width note](fixed-width-trunk.md)).
The three open gates now have tooling. *Bit-width curve:* the gate tool scores
3/4/5-bit tables, a sign-split variant, unconstrained Huffman and order-0 entropy
bounds, per range and pooled, in CI. On the only committed real counts (four of
the gate-1 `f_a_proj` ranges) 3 bits retains 0.704128 against 0.750153 for 4 bits
(+4.60 points), 5 bits 0.8125, Huffman 0.672050, high-byte entropy 0.670303; the
sample-matched FD4B premium over the four-stream Huffman payload is 7.80 points.
*FD3B:* a 3-bit decoder with branch-free escape expansion (scalar, SSSE3, AVX2,
NEON) passes byte-exact tests locally; its rates await the hosted rate job (a
reading on this VM under another agent's load, with no report kept, is orientation
only). *Row index:* FDRX makes any
whole-row range decodable for 0.0340 points of matrix bytes per row, or 0.0148
grouped, with zero padding at K3 widths. *Families:* gate 1 sampled two of 23
matrix families (4.00% of trunk bytes); the `families` CI job samples all of
them, under per-family decision rules fixed in the note before the data.
*Contention:* a byte-exact benchmark runs one decoder thread against
`k3_matmul_bf16` on the other cores. On a quiet 4-vCPU VM (load 0.10 and 0.29
before the runs, 9 interleaved repeats) the worst-case contended streamed speedup
exceeds 1 up to B = 5 GB/s in every format, placement and thread count (4 and 2),
and is the full 1.333 (FD4B) / 1.420 (FD3B) up to about 4.2 GB/s on 4 threads; at
6 GB/s it fails for streamed input on 4 threads (0.970, 0.998, decode-limited) and
everywhere on 2 (about 0.85, matmul-limited). Pinned layers should stay raw
(resident slowdown 2.08 to 3.62). Hosted numbers for all of these await CI. No
full-model speedup is claimed.

## 3. Predictive expert reads: closed

Andrew's follow-up closes this route independently of any future generation
capture. The assumed 70% recall adds about 5.76% whole-token traffic while
experts account for only 19.2% of baseline bytes. The audit remains the stopping
record. No further capture or predictor work is planned; the next item is the
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
cost `D` and rejection replay cost `R`, normalized to ordinary-step cost:

$$E[N]=\sum_{j=0}^w p^j,\qquad E[T]=V+D+(1-p^w)R.$$

It pays only if `E[N] > E[T]`. Assuming `p=.9,w=2,V=D=R=1` gives
`2.71/2.19 = 1.2374`; at `p=.5` it loses. Those are assumptions, not K3 acceptance.
Correlated acceptance needs empirical prefix-length frequencies. Guaranteed
convergence via `w` full Jacobi passes plus verification costs at least `w+1`
sweeps to emit at most `w+1` tokens: no pass-count improvement.

The existing `--spec` is unchanged. A new mode needs cheap early proposals or
reused computation, complete KDA/KV/AttnRes rollback tests, and real acceptance
including rejected-work I/O. The primary
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
