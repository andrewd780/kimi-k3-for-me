# Expert and trunk streaming: options, proofs, and gates

The largest open exact *latency* lever is scheduling reads against useful work,
especially the trunk. The measured scale entropy gives a smaller but concrete
*storage* lever. This follow-up implements the latter as
[`--policy scales`](../SELECTIVE_SCALES.md): packed weights use raw reads, scales
use bounded independent frames, and the reconstructed bytes are unchanged.

No finite note can enumerate every future algorithm. The map below covers the
main ways to change required bytes, hide transfers, increase bandwidth, reduce
compute, or change the model. Nothing here establishes an unmeasured speedup or
a world first. Mathematical bounds eliminate bad premises and state necessary
conditions; they do not supply missing hardware or quality measurements.

## Evidence and constraints

The [14 earlier lessons](https://github.com/andrewd780/j-research/blob/main/qwen-heavy/reports/council-brief-2026-09-07-expert-streaming.md#3-what-kimi-k3-in-c-actually-does-and-what-transfers)
were checked before choosing this work. The later corrected trace analysis
supersedes that brief's historical 36% hit-rate/12% coverage interpretation.
The [profile result](../EXPERT_PROFILES.md), negative [KDA SIMD result](kda-simd.md),
negative [int8 self-draft result](int8-draft-container.md), and shelved
[Huffman trunk decoder](compressed-trunk.md) remain evidence, not fresh leads.

The small-budget one-position reference streams approximately 108.81 GB of trunk
and exactly `92 * 16 * 17,547,264 = 25,829,572,608` expert bytes, or **134.640 GB
total**, with zero persistent expert hits. The ladder's 25.83 GB column counts
experts only. These are logical payload totals; alignment, metadata and actual
device traffic are additional. `--stream-lm-head` adds output-table reads and
is not included in that baseline. The independently packed trunk also occupies
storage in addition to a full original checkpoint; do not count duplicate files
as model parameters.

The [campaign environment](../data/environment.txt) was a rented 124-core EPYC
host. Its 3.2 GB/s cold direct-I/O figure and 5.4–6.1 GB/s in-engine logical trunk
rates came from different tests and must not be mixed into a single calibrated
model. No standing host or full checkpoint is available. All native execution in
this follow-up is in CI, including hosted ARM; nothing runs on Andrew's Macs.
Three timing runs per arm remain required for future performance gates. The
33% observed campaign spread remains the conservative bar for claimed effects.

## The equations that decide whether an idea can help

Let `T` be uncached trunk bytes, `P` packed expert bytes, `S` expert scale bytes,
`B` effective disk bytes/s, `C` compute service time, and `D` scale decode bytes/s.
For released group-32 MXFP4, **P = 16S**. For one device, even perfect overlap
must respect its total byte demand:

$$
 t \ge \max\left(C,\frac{T+P+S}{B}\right).
$$

This is an optimistic lower bound, not the engine's measured clock. Expert and
trunk readers sharing one device cannot each independently receive its full
bandwidth. Prefetch changes when bytes arrive; it does not remove this constraint.
If a hypothetical CPU takes 216 seconds while the disk takes 45 seconds, hiding
all disk time still leaves 216 seconds. The current data does not establish that
compute time on any Mac.

### Selective scale compression

For sample retained fraction `r = 0.15531521373324925`, the expert fraction is

$$
 \frac{P+rS}{P+S}=\frac{16+r}{17}=0.95031266.
$$

That saves **4.969% of expert payload, 0.953% of total payload**. Even free decode
buys only **1.0096x** of total I/O-limited speed in this baseline. Applied to an
assumed 85 GB scale plane, storage falls by about 71.8 GB. Applied to a 1,560 GB
checkpoint, that leaves approximately **1,488 GB**, excluding extra trunk copies
and filesystem overhead. This is still a sample-based projection.

Serial scale read+decode helps only when

$$
 \frac{rS}{B}+\frac{S}{D}<\frac{S}{B}
 \quad\Longleftrightarrow\quad D>\frac{B}{1-r}.
$$

At an assumed 3 GB/s device this requires **3.552 GB/s decode**. A codec does not
become serially profitable just because it handles a small fraction of bytes:
the fraction cancels in this inequality. With a *dedicated overlapping decoder*,
the steady-state expert-stream condition is instead

$$
 D\ge\frac{SB}{P+rS}=\frac{B}{16+r}=0.186\ \text{GB/s at }B=3.
$$

These are different resource models. The sample's roughly 0.68 GB/s aggregate
decode rate passes the second numeric condition but not the first. It includes
Python overhead and is not a native deployed rate. A decoder sharing scarce
compute cores, buffered-read pressure, and queue contention can erase the overlap.
The new reader makes selective decoding possible without claiming it is faster.

### Known-route pipelining: no prediction is necessary within a layer

`k3_moe` already knows all 16 selected IDs before `getmany` is called. Today the
cache reserves slots, reads the whole batch, publishes, and only then returns
for expert matmuls. There is a dependency barrier between all reads and all
expert computation. It is sufficient for correctness to wait for the *particular*
expert being consumed, while keeping its slot protected until computation ends.

For `k` known tasks with equal read time `i`, compute time `c`, one fixed-bandwidth
reader and independent compute resource, the current barrier costs `k(i+c)`.
A two-buffer pipeline has the exact idealized schedule

$$
 t_2=i+c+(k-1)\max(i,c).
$$

Proof: the first result requires `i+c`. Thereafter the slower stage determines
the initiation interval. With two buffers, one can be read while the previous
one is consumed; the buffer cannot be reused until that consume completes.
Induction on tasks gives the formula. The accompanying precedence scheduler
checks every `k=1..32`, `i,c=1..7` case against it, including the one-buffer
serial control. This is a schedule proof under assumptions, not a benchmark.

At `k=16` and `i=c`, the **expert stage** bound is `32/17 = 1.882x`. When one
stage takes four times the other it is only `80/65 = 1.231x`. Neither is a
whole-model speed claim. Lower queue depth may reduce NVMe bandwidth, so `i`
cannot simply be held fixed in a real comparison.

The mathematical two-expert payload floor is **35,094,528 bytes**. The current
cache's `topk+1` contract reserves at least **298,303,488 bytes**, and practical
queue depth can require more. This does not identify an old allocation bug or
change the engine's supported minimum. It motivates a separate streaming queue
whose ownership contract is “retire before refill”, not “all top-k are resident”.
Memory was already solved; any extra freed budget is useful only if it changes
trunk overlap or actual retained reuse.

An implementable next experiment should first retain the current cache capacity:
reserve all missing slots serially; have a worker fill them without touching
replacement metadata; publish readiness under a lock; let `get` wait only for its
own requested slot; keep every batch slot protected; drain/join before reusing a
slot, resetting stats or closing. Issue demand-order reads or bounded waves so
the first consumed expert is not last in an offset-sorted batch. Never reorder
the floating-point weighted sum. Stress failed reads, duplicate IDs, already
resident IDs, short caches, prefill unions, shutdown and thread cancellation in
CI before exposing a flag. Only a later real-model comparison decides wave size
and whether CPU/I/O contention defeats it. It is **not implemented here**.

### Trunk buffers and granularity

Whole-layer buffers require at least the live layer and its prefetched successor.
The sum of their actual sizes, not twice the largest layer, is the relevant
pairwise payload bound. A 2.34 GB dense layer followed by a 1.17 GB layer needs
3.51 GB of payload in that pair, versus 4.68 GB for two uniform largest-layer
slots. Widening, alignment and other layers are additional; those rounded sizes
do not prove a feasible complete allocation.

Two fixed unequal slots also need a valid assignment across **all 93 layers**
and token wraparound. An odd cycle cannot be assigned alternating two colors:
the large layer eventually lands in the small slot if a rotating index is
blindly continued. A boundary drain/reset or explicit lifetime-based placement
is necessary. Test every transition and prefetch in flight. This is a new
scheduling proposal, not a claim that changing one slot size is safe.

Splitting large trunk matrices into output-row tiles can lower the buffer floor
further. For `y=Wx`, output rows are independent. Retaining the original ascending
input-column reduction within every row preserves its arithmetic; partitioning
that reduction and adding partial sums generally does not. All consumers must
finish before a tile is overwritten. A tensor/sub-layer binder and pipeline
would therefore be a larger exact change with an oracle and allocator gate.
It buys memory and overlap, not fewer weight bytes. The existing independent
lm_head stream is the small completed example of this principle.

### Prediction must pay for wrong guesses

If `k` experts are needed, `n` are guessed, and `h` guesses are correct and retained,
the reader fetches `n + k - h` experts, before any eviction pollution. For top-16
prediction of 16 experts with 70% recall this is **1.3x expert traffic**, about
**1.058x total traffic** in this baseline. Merely exceeding 70% recall is not a
proof of net speedup: useful lead time must hide correct reads, and wasted reads
must not delay trunk/demand traffic. A cancelable low-priority reader and bandwidth
reservation are part of the experiment. Prediction changes fetch timing only;
the real router must still select the exact experts.

### Amortization, speculation, and approximate top-k

Batching independent requests amortizes the trunk read. If each request draws a
uniform independent top-k set from `N` experts, expected distinct experts per
layer are `N[1-(1-k/N)^b]`. Linearity of expectation over “expert used at least
once” indicators proves the expression; exhaustive small set enumerations check
it. Balanced marginals alone do not establish independence across requests.

Under that toy model, eight independent requests need about **37.87 GB per
output token**, versus 134.64 GB at batch one: **3.56x I/O throughput potential**.
Each round still executes all eight requests, needs eight states, and can increase
individual latency. It cannot batch unknown future tokens of one autoregressive
answer. Existing batched MoE prefill already reuses experts within a prompt;
that mechanism is not being rediscovered as new.

A cheap *independent* draft model or exact prompt-lookup proposal could expose
future candidates to batched verification. For a deliberately hypothetical
constant conditional acceptance `a` and `m` candidates, expected verified outputs
including the correction/bonus token are `1+a+...+a^m`. A speedup requires
`draft_cost + verification_cost < expected_outputs * ordinary_token_cost`.
Every verified position still needs real trunk computation, real expert routes,
and state rollback on rejection. The failed int8 self-draft does not satisfy a
cheap-draft premise, and its agreement figure is not main-model quality evidence.

Reducing top-k from 16 to 4 is **lossy**. It quarters expert reads but, with the
trunk unchanged, cuts total bytes only from 134.64 to **115.27 GB**: at most
**1.168x** I/O-only improvement. Compute could change too; quality is unmeasured.
It cannot be presented as a fourfold full-model speedup. The existing quality
harness is the prerequisite to any deliberate lossy branch.

## Option map

“Exact” below means preserving all released weight bits and arithmetic/output
contracts, subject to the stated implementation gate. Mathematical real-number
equivalence alone does not satisfy the bytewise oracle.

| Mechanism | What it can change | Exactness, evidence, and next gate |
| --- | --- | --- |
| Selective scale compression with raw packed reads | Storage; a little I/O; decode allocation | Implemented here, opt-in, synthetic native byte gates. Real conversion size, buffered/direct comparison and timing remain unmeasured. |
| A fast scale palette plus escape bytes | Fewer decoder instructions, at a worse compression ratio | A 2-bit selector for bytes 121/120/122 plus 8-bit escapes would retain 25.662% on the measured histogram, before headers. Exact for all byte values with escapes. Needs an implemented decoder, position index and three-run speed gate; histogram arithmetic proves only representation size. |
| Conditional/delta scale coding or ANS | Potentially improve ratio or speed | Exact if fully reversible; order correlations and decode costs need raw samples. Marginal histograms do not measure conditional entropy. No claim of a new entropy coder. |
| Compress all packed nibbles | Modest extra storage reduction, substantial decode work | Existing Zstd sample retains 94.26%; the 6% is known, not a route to 200 GB. Use only after measuring the complete read/decode path. |
| BF16 trunk coding or byte-plane transforms | Trunk bytes, the majority of I/O | Existing Huffman decoder was too slow. Eight BF16 samples are not representative of every trunk tensor. Reopen only with new fast native decode evidence; no precision change disguised as lossless coding. |
| Compress resident cached experts | Roughly 5% more capacity under current scale ratio | Decode-on-use and scratch can consume the gain; an extra slot is not proof of reuse. Needs corrected position-major held-out traces. Not a new solution to the refuted budget gap. |
| Current-layer asynchronous expert consumption | Overlap exact known reads and matmuls | Strong next scheduling lead. Keep slot ownership and original sum order; concurrency and real workload gates described above. |
| Next-layer dynamic expert prediction | Earlier I/O, with extra wrong-guess traffic | Real hidden-state/routing data and a probe are required. Keep true routing and demand priority; static pinning's negative transfer result is not dynamic prediction evidence. |
| Asymmetric trunk ring or row/tensor streaming | Less buffer waste and more overlap | Full lifetime/93-layer wraparound proof, allocator gate and unchanged oracle required. Pairwise arithmetic alone is insufficient. |
| Demand priority, bounded waves, queue-depth tuning | Device utilization and tail latency | Existing concurrent, offset-sorted reads already exist. Trunk and expert readers contend for one device. Sweep real workload at fixed memory with three runs; no universal best depth. |
| Coalesce more reads or repack per-expert extents | Fewer system calls or seeks | Six original tensors are already usually one contiguous expert read. Selective extents trade this for cheap decoding. Reading gaps adds bytes; benchmark both, especially on HDD. |
| Direct-I/O alignment or Darwin cache bypass for selective raw spans | Page-cache pressure and copies | Still open. Must map physical alignment, preserve compressed neighbors, bound scratch and validate on hosted platforms. “pread into destination” is not O_DIRECT. |
| Reuse decoder contexts or decode directly into destination | Allocation/copy overhead | Bounded per-worker state, ownership and corrupt-frame behavior need tests. Cannot eliminate entropy decode work. Raw spans already need no decoder scratch. |
| Avoid duplicated trunk tensors in an export | Deployment disk duplication | A full source plus packed trunk has redundant bytes. Needs a manifest-aware exporter/binder, source preservation and complete tensor-coverage verification. Does not shrink the original model's parameter payload. |
| More trunk pinning; stationary component placement | Actual read reduction for always-used weights | Established upstream allocation lesson. Must fit the complete memory plan and beat loss of double buffering. Expert cache memory gets no assumed 36% hit-rate windfall. |
| Multi-request batching or larger prefill reuse windows | Throughput by amortizing trunk/expert reads | Existing prefill reuse already works. Serving adds per-request states and waiting; it does not lower one-stream dependency latency. Uniform union formula is conditional. |
| Independent cheap drafting or prompt lookup plus exact verification | More output tokens per trunk traversal | Different from expensive self-drafting, but not automatically beneficial. Needs compatible tokens, exact rollback, acceptance on held-out prompts, and real verifier cost. Existing lookup/speculation paths must be the baseline. |
| SIMD/GEMV scheduling, w1+w3 input reuse, avoiding repeated widening | Compute time and memory bandwidth | The KDA SIMD trial already regressed ARM. Fuse only independent outputs while preserving each dot-product order; time the specific remaining kernel rather than extrapolating. |
| Exact activation or result memoization | Skip work only for genuinely repeated full inputs/state | Approximate similarity or temporal “small deltas” do not imply equality. Hash collisions need byte comparison; changing sum decomposition loses bitwise identity. Unknown reuse is not a capacity claim. |
| Exact zero skipping or certified output pruning | Potentially less arithmetic or reads on special inputs | Dense SiTU is not ReLU sparsity. Signed zero, NaN, Inf and reduction order matter. Certified argmax equality is weaker than equal logits/full next-token state and needs a separate contract. No approximate threshold is enabled. |
| Several independent SSDs or storage-side compute | More physical bandwidth or less host traffic | Sum device rates only until a shared bus/CPU becomes limiting. Requires hardware; no rental or purchase is made. Does not reduce stored model bytes. |
| GPU/Metal acceleration of streamed tiles | Compute, potentially the dominant small-machine term | Architectural alternative outside this repository's standing CPU-only scope. Cannot remove the storage floor and has no Apple GPU measurement here. |
| Reduced top-k, lower trunk precision, expert pruning | Compute/bytes by changing the function | Deliberately lossy, fails the present identity gate. Known pruning and quantization failures remain closed; real quality evaluation is needed before revisiting. |
| Shared base, low rank, centroids, codebooks or trained replacement | Smaller representation of a changed model | Existing K3-specific negatives remain closed. Exact dense deltas retain their bytes; real-arithmetic factorization is not float-order equivalence. A distilled smaller model is a different model. |
| Remote cache/cloud experts | Less local disk by moving weights elsewhere | Incompatible with the offline goal and dependent on network misses. Not pursued. |
| Regenerate arbitrary weights from a small seed | Replace stored information with computation | No such generator for these released trained weights is known here. Counting forbids a universally shorter injective code for arbitrary byte strings; it does not prove this particular checkpoint has no exploitable structure. |

## What compression math does and does not rule out

A one-byte histogram entropy is the expected coding rate of a memoryless symbol
model. It is **not** a universal lower bound on every context-aware or structural
lossless compressor for this checkpoint. Similarly, a poor rank/quantization
proxy is not a perplexity result. The evidence supports rejecting those measured
approaches; it does not justify a proof that all possible exact algorithms fail.

What is certain is the information-accounting obligation: an exact reconstruction
must preserve whatever information is not supplied by a valid shared dictionary,
predictor, generator or other side information, and that side information counts
toward storage. Decoder instructions/time count too. Caching moves bytes to RAM;
remote loading moves them to a server; pruning changes the model. None is a
lossless storage reduction simply by changing the accounting boundary.

Even eliminating all expert reads would improve the baseline *I/O component* by
at most `134.64/108.81 = 1.237x`. Eliminating all trunk reads would make that ratio
`134.64/25.83 = 5.213x`, but requires somewhere to retain or avoid those weights.
These are byte-only ceilings, not wall-clock effects. They explain why improving
the trunk or compute deserves more weight than a clever expert cache at laptop
budgets.

## Reproduce the calculations and inspect the work

```sh
python3 tools/streaming_bounds.py --out streaming-bounds.json
python3 -m unittest discover -s tests -p test_streaming_bounds.py -v
```

[`streaming-bounds.json`](../measurements/streaming-bounds.json) records the sample
file hash, model geometry, explicit scenario rates, conditional formulas and
results. Fractions are used internally; only JSON presentation rounds to floats.
Change `--ssd-gbps`, `--decode-gbps`, and `--compute-seconds` to explore a scenario.
The default 216 seconds is a hypothetical compute service time from the earlier
discussion, **not a Mac prediction or a measurement**. Deterministic calculations
run once; repeating them three times would add no evidence.

The new reader's first native gates ran in
[CI 34803774252](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/34803774252)
at commit `08a9503e09be5d0a147c23f8be13f1895872d48a`: 24 storage test groups on
Linux and hosted ARM, existing C oracle gates, and 14 synthetic CLI groups on
Linux. A raw-only read was also tested with the decoder deliberately disabled.
No timing result, new weight sample download campaign or whole-model inference
is needed to establish those implementation properties.

## Relevant prior work, with scope kept explicit

- [Pre-gated MoE, Hwang et al., 2023 preprint / 2024 revision](https://arxiv.org/abs/2308.12066v3)
  changes gating to enable early host-to-GPU transfer. It motivates lookahead;
  it does not establish a drop-in byte-exact predictor or SSD speedup for K3.
- [MoE-Infinity, Xue et al., 2025 revision](https://arxiv.org/abs/2401.14361v3)
  uses request tracing and expert caching/prefetch for personal-machine serving.
  Its DeepSeek/Mixtral results do not establish reuse for K3's corrected trace.
- [EdgeMoE, Yi et al.](https://arxiv.org/abs/2308.14352)
  combines expert management with bitwidth adaptation. The latter changes weights
  and cannot be imported as lossless compression.
- [LLM in a flash, Alizadeh et al.](https://arxiv.org/abs/2312.11514)
  studies flash-resident inference with windowing and row/column bundling.
  Its sparsity premises need model-specific verification for this SiTU MoE.
- [FlexGen, Sheng et al.](https://arxiv.org/abs/2303.06865)
  optimizes offloading and batching for throughput. That objective explains why
  its gains cannot be translated directly into one interactive stream's latency.
- [Zstandard's primary API documentation](https://facebook.github.io/zstd/zstd_manual.html)
  defines bounded output buffers and independent decompression calls. Those
  contracts underlie the selective reader; generic codec speed figures are not
  substituted for a K3 measurement.
