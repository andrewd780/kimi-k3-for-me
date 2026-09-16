# Where everything stands

*Plain-language status of this fork, written for the project owner. Last updated
2026-09-14 with pull request #8. Every number links to the file it
was measured in. Engineering priorities stay in [ROADMAP.md](ROADMAP.md).*

## The short version

**The engine works and is exact.** A 2.78-trillion-parameter Kimi K3 runs on one CPU
in 8 GB of RAM, and its output is byte-identical at every memory budget. That is
proven by tests that run in seconds with no weights (`make test`), by the
[memory ladder](data/memory-ladder.tsv), and by a
[four-run proof on an 8 GB Jetson](results/jetson-orin-nano-super/README.md).

**It is slow, and the model is enormous on disk.** The released checkpoint is
1.56 TB. At 8 GB of RAM the measured speed was 32.7 seconds per token, on a rented
124-core server with a fast NVMe ([environment](data/environment.txt)). On the 8 GB
Jetson it was about 16 minutes per token. The best measured full-model speed on that
same rental, with memory allocated well rather than at the 8 GB floor, is **10.66
seconds per token in sustained decode, at 127.9 GB peak RSS**
([README](../README.md), [PERFORMANCE.md](PERFORMANCE.md#longer-runs-are-faster)). A
laptop with 8 to 12 cores is expected to be slower than the server, because compute
rather than disk becomes the limit there. That expectation is an estimate; it has not
been measured.

**Full quality at 20 to 50 tokens per second.** No Mac configuration reaches that,
at any price. Four 512 GB M3 Ultra Studios pipelined across the layers project to
about 6 tokens/second for a single stream; splitting the model across them with
tensor parallelism instead projects to about 12 to 16. Both are estimates, not
measurements, and both are short of 20. Only a data-center node that holds the whole
model in fast accelerator memory gets there. An 8 GB Mac can only be a *client* of
such a node, never the machine doing the work; see
[notes/remote-k3.md](notes/remote-k3.md) for what that costs today.

**No lossless trick gets it under 200 GB, or even under 1 TB.** Every compression
route was measured or argued to the end. The best lossless result leaves about
1.49 TB ([bounds](measurements/streaming-bounds.json)). A 200 GB model would need
0.42 bits per parameter on average, and the released weights already sit at 4.25.
Below one bit per parameter, weights must share codes, which is the same thing as
removing parameters, which is a different model. A different model needs a quality
evaluation, and the harness for that exists now but has never been run on K3.

**Neither of Andrew's Macs can hold the checkpoint at all.** The M1 Air has a 256 GB
disk and the M4 Max has about 90 GB free. Disk, not RAM, is the first wall. A 2 TB
external NVMe would hold the checkpoint plus the 109 GB packed trunk; what speed that
gives is unmeasured.

## What fits where

| | GB |
|---|---:|
| Released checkpoint | 1,560 |
| After every lossless trick measured so far (projection) | 1,488 |
| M1 Air disk, total | 256 |
| The goal | 200 |
| M4 Max free space | 90 |

The context window has its own, separate cost. The attention cache is stored
**expanded**, in fp32, across the 24 MLA layers: 2.37 MB per position, which is
19.38 GB of cache at an 8,192-token context, read again every token
([README](../README.md)). A latent cache, `--kv-latent`, is in progress and would
remove most of that by keeping the cache compressed instead of expanding it; see the
[proposed techniques](#proposed-not-started) below.

## Every technique, where it stands

Status words: **done** means merged, tested and exact; **negative** means it works but
measured no useful gain; **measured** means a number exists and no code was needed;
**shelved** or **dead** means investigated and closed; **proposal** means written up,
not built; **blocked** means it needs a machine holding the full checkpoint.

### Shipped, exact, tested

| Technique | What it does | Status | How deep the evidence goes | What is left |
|---|---|---|---|---|
| Streamed inference engine (upstream) | Reads weights from disk during each token instead of holding the model in RAM | done | 12 memory budgets, one identical token sequence, 33% run-to-run speed noise ([ladder](data/memory-ladder.tsv), [noise](data/replication.tsv)) | Re-run with 3 repeats; needs a checkpoint host |
| Trunk read-ahead ring (upstream) | Reads the next layer while the current one computes | done | 1.70x measured on the campaign host | At 8 GB only one slot fits, so the overlap is lost there |
| Hybrid draft decode, `--draft-trunk` (upstream) | A cheaper copy proposes tokens; the exact model verifies them | done, but slower unless the draft fits in RAM (64 GB and up) | One real flight accepted 66.7% of drafts ([note](notes/int8-draft-container.md)) | Int8 container never built; irrelevant on a small laptop |
| Offline lossless archives, K3ZSTD1 (#1) | Compressed weights the engine reads directly, no network | done, opt-in | Native tests; 32 real samples ([doc](OFFLINE_STORAGE.md)) | Experts shrink only 5.7%, so not a route to a smaller model |
| Prompt rereading (#1) | Processes the prompt twice before generating | done, exact | CLI tests | Nothing |
| Calibrated expert pinning (#2, #4) | Keeps the most-used experts resident | done, opt-in; **negative** | Held-out replay saves 0.44% of expert reads at 8 GB; the earlier 36% hit rate was an artifact of repeated prompts ([doc](EXPERT_PROFILES.md)) | Dead as a speed lever |
| Exact SIMD KDA recurrence (#5) | Hand-vectorized inner loop of the recurrent layers | done, opt-in (`KDA_SIMD=1`); **negative** | Bitwise identical across 9,504 checks and under sanitizers; 1.11x on x86, 0.72x on ARM, both inside the noise floor ([note](notes/kda-simd.md)) | Leave off by default |
| lm_head streaming, `--stream-lm-head` (#5) | Streams the 2.35 GB output table instead of keeping it resident, funding a second trunk slot | done | Synthetic mechanism gate under a 64 MiB memory cap: ring grows from 1 to 2 slots with identical logits ([note](notes/stream-lm-head.md)) | Full-model effect unmeasured |
| Quality harness, `tools/quality.py` (#5) | Scores how well a model predicts held-out text, so a changed model can be compared with the exact one | built, never run on K3 | Protocol tests and a tiny native check ([doc](QUALITY.md)) | Needs a checkpoint host; about 1.4 TB of reads per 128-token window |
| Selective scale archives, K3ZMAP1 (#6) | Compresses only the 5.9% of expert bytes that compress (the scales) and reads the rest raw | done, opt-in | 24 storage tests, sanitizers, synthetic CLI parity ([doc](SELECTIVE_SCALES.md)) | Saves 0.95% of bytes per token; unmeasured on the real model |
| Direct I/O for selective raw extents (#8) | Reads the raw parts of a selective archive past the page cache, the way plain shards are read | done | Native test proves the direct path is taken and falls back cleanly; bytes identical | Speed unmeasured on the real model |
| Known-route expert pipelining, `--expert-pipeline` (#9) | Publishes each routed expert as its read lands instead of waiting for the whole top-k | done, opt-in | Native pipeline-mode cases plus sanitizers; CLI parity for full recompute and `--incremental` at pool sizes 1 and 16 ([note](notes/expert-pipeline.md)) | Speed unmeasured on the real model |
| MLA latent KV cache, `--kv-latent` (#10) | Keeps the attention cache in its 512-wide latent form and rebuilds keys and values on use: 55 KB per position instead of 2.37 MB, 42.8x less | done, opt-in | GATE 3b holds all 20 incremental steps bit-identical between layouts; state save/load round-trips; 20 CI checks green ([note](notes/kv-latent.md)) | Costs one kv_b matmul per cached position per use; speed unmeasured on the real model |

### Measured, no new code

| Technique | What it answers | Status | How deep the evidence goes | What is left |
|---|---|---|---|---|
| Scale-plane entropy campaign (#5) | How compressible the expert scale bytes are | measured | 264 MB from 256 real experts across 8 shards: 15.5% retained under Zstd ([note](notes/scale-plane.md)) | Projects 85 GB to 13 GB; the checkpoint stays about 1.49 TB |
| Packed-expert compressibility (#1) | How compressible the main expert bytes are | measured | 24 MiB of real samples: 94.3% retained ([samples](measurements/lossless-samples.json)) | Dead end; these bytes are already at their entropy |
| Memory ladder and I/O split (upstream) | Speed against RAM budget | measured, single sample | 32.7 s/token at 8 GB, 19.2 s/token at 224 GB, on 124 cores; disk is 57% of the time at 8 GB | Share on a small core count unknown |
| Jetson Orin Nano Super proof (upstream) | Full checkpoint on an 8 GB-class board | measured | 4 runs, 949 s per token including a 5-token prefill, 208 GB read per run | Proof of life, not usable speed |
| Streaming option map and bounds (#6) | 26 ways to change bytes, overlap or compute, with the math checked | written | Ceilings: removing all expert reads gives at most 1.24x; removing all trunk reads at most 5.2x, disk time only ([map](notes/streaming-options.md)) | A menu, not code |

### Built and shelved, or studied and closed

| Technique | What it would do | Status | Why it closed | What would reopen it |
|---|---|---|---|---|
| Huffman-coded trunk (upstream) | A 1.45x smaller trunk, lossless | shelved, queued second | Decoder ran at 0.31 GB/s per core; reference Huff0 reached 1.72 GB/s on the same bytes, so the bar is reachable ([note](notes/compressed-trunk.md)) | A small multi-stream decoder above 1 GB/s per core; pays only where disk time exceeds compute ([queue](notes/research-queue.md)) |
| Expert pruning (studied) | Drop rarely used experts | dead | Held-out coverage plateaus at 35%; usage is deliberately flattened by the router | It changes the model, so the quality harness first |
| Shared base plus low-rank delta (studied, closed PR #3) | Store one expert per layer plus small differences | dead on paper | Needs 0.99 correlation between experts; real expert weights look random. The write-up itself had errors and was closed unmerged | Kept only at PR #3 for the record |
| Int8 trunk as the main model (upstream note) | Halve the trunk by rounding | not validated | The 90.9% figure was a 22-token draft with the exact model verifying, not a quality result | A real quality evaluation |

### Proposed, not started

| Technique | What it would do | Status | What is known | Size of the job |
|---|---|---|---|---|
| Tensor-granular trunk ring (#6 map) | Streams the trunk one tensor at a time so two ring slots cost about 1 GB instead of 2.37 GB for one, restoring read-ahead at 8 GB | proposal, next to build | Arithmetic from the reader's own slot rule and the 0.49 GB largest tensor; 1.75x ceiling at 8 GB on the measured host ([queue](notes/research-queue.md)) | Large exact change, staged; oracle, allocator and ThreadSanitizer gates |
| Next-layer expert prefetch | Predicts the next layer's experts from the current hidden state and reads them early; routing still decides what is computed | proposal | Option-map arithmetic: 1.3x expert traffic for earlier arrival, at most 1.06x on total bytes ([queue](notes/research-queue.md)) | Large; predictor needs real routing data |
| Bounded lookahead verification | Drafts without a history match, verified by the exact model | proposal, flagged | The engine's own 0.91x eager-drafter result says wide windows lose ([queue](notes/research-queue.md)) | Small window and evidence gating only; acceptance unmeasurable here |
| io_uring reads | Async read submission without a thread per read, Linux only | proposal | Measurable on any Linux box without a model ([queue](notes/research-queue.md)) | An enabler for the ring on small-core machines, not a lever alone |
| Speculative decoding on resident hardware | A cheap draft proposes tokens, the exact model verifies; ~1.7x fewer weight bytes per accepted token at the measured 66.7% acceptance ([note](notes/int8-draft-container.md)) | proposal | Only pays off once the model is resident in RAM; nothing on 8 GB, where both draft and exact stream from disk | Needs a large-memory host to pay for |
| Chunked prefill, sampling, chat template, vision, HTTP serving | Usability features from the upstream roadmap | not started | n/a | Do not change size or speed |

### Blocked on a machine that holds the checkpoint

Full-model seconds per token on a laptop-like core count, a thread-count sweep, real
8 GB overlap timing, and any real quality number. Nothing has run on a full checkpoint
since the rental. Doing it needs roughly 3 TB of NVMe and hours to download 1.56 TB.
Each quality window reads about 1.4 TB, so a 10,000-token corpus is on the order of
a day of disk time.

## Three decisions

1. **Organization, now.** Delete the merged branches (each merged pull request page
   has a "Delete branch" button) and turn on *Settings, General, Pull Requests,
   Automatically delete head branches* so this stops recurring.
2. **Checkpoint-free engineering, in order of value per risk.** The lint findings
   and direct I/O for selective raw extents landed in #8; known-route expert
   pipelining landed in #9; the latent KV cache landed in #10. The ranked list of
   what is left is [notes/research-queue.md](notes/research-queue.md): the
   tensor-granular trunk ring first (overlap pays on every machine), the small
   Huffman decoder second (pays only where disk time exceeds compute), three more
   recorded and not recommended.
3. **The rental remains the only way to measure.** Everything marked blocked needs one
   machine with the checkpoint for a day or two. Without it there will never be a
   laptop speed or quality number; with it, one core-limited ladder run answers the
   decisive question.
4. **Remote K3 or not.** Full quality at 20 to 50 tokens/second is not a laptop
   outcome at any price; it needs a data-center node or Moonshot's own API. See
   [notes/remote-k3.md](notes/remote-k3.md) for the real prices found and what each
   route costs the project owner in money versus engineering time.

## Where things live

| | |
|---|---|
| [ROADMAP.md](ROADMAP.md) | engineering priorities, in order |
| [PERFORMANCE.md](PERFORMANCE.md) and [data/](data/) | every measured number and its raw output |
| [TESTING.md](TESTING.md) | what each test proves |
| [notes/](notes/) | one write-up per experiment, including the negative ones |
| [measurements/](measurements/) | machine-readable records behind the notes |
