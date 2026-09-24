# Where everything stands

*Plain-language status of this fork, written for the project owner. Last updated
2026-09-23 with merged work through #14. Measurements and projections are
distinguished below. Engineering priorities stay in [ROADMAP.md](ROADMAP.md).*

## The short version

**The engine works and is exact.** A 2.78-trillion-parameter Kimi K3 runs on one CPU
in 8 GB of RAM, and its output is byte-identical at every memory budget. That is
proven by tests that run in seconds with no weights (`make test`), by the
[memory ladder](data/memory-ladder.tsv), and by a
[four-run proof on an 8 GB Jetson](results/jetson-orin-nano-super/README.md).

**It is slow, and the model is enormous on disk.** The released checkpoint is
1.56 TB. At 8 GB of RAM the measured speed was 32.7 seconds per token, on a rented
124-core server with a fast NVMe ([environment](data/environment.txt)). The Jetson's
949-second run includes a five-token prompt prefill; it is not an isolated decode
estimate for a Mac. The best reported full-model speed on that
same rental, with memory allocated well rather than at the 8 GB floor, is **10.66
seconds per token in sustained decode, at 127.9 GB peak RSS**
([README](../README.md), [PERFORMANCE.md](PERFORMANCE.md#longer-runs-are-faster)). A
laptop with 8 to 12 cores is expected to be slower than the server, because compute
rather than disk becomes the limit there. That expectation is an estimate; it has not
been measured.

**No demonstrated path here reaches 20 to 50 tokens per second on Andrew's Macs.**
Server-to-Mac core-count scaling and multi-machine projections are not benchmarks.
The engine's actual Mac speed remains unmeasured. A remote service is a different
deployment choice and does not satisfy offline inference.

**No implemented technique demonstrates storage below 1 TB, let alone 200 GB.**
Selective scale compression projects about 1.49 TB
([bounds](measurements/streaming-bounds.json)). The new small trunk-codec samples
are not a full-checkpoint result. Compression samples do not prove every future
lossless method impossible. A lossy change still needs real quality evaluation;
the harness exists but has never been run on K3.

**Neither of Andrew's Macs can hold the checkpoint at all.** The M1 Air has a 256 GB
disk and the M4 Max has about 90 GB free. Disk, not RAM, is the first wall. A 2 TB
external NVMe would hold the checkpoint plus the 109 GB packed trunk; what speed that
gives is unmeasured.

## What fits where

| | GB |
|---|---:|
| Released checkpoint | 1,560 |
| After selective scale compression (sample-based projection) | 1,488 |
| M1 Air disk, total | 256 |
| The goal | 200 |
| M4 Max free space | 90 |

The context window has its own, separate cost. The attention cache is stored
**expanded**, in fp32, across the 24 MLA layers: 2.37 MB per position, which is
19.38 GB of cache at an 8,192-token context, read again every token
([README](../README.md)). The merged opt-in `--kv-latent` reduces this allocation
by keeping latent values and rebuilding keys/values on use. It adds matrix work;
its full-model latency is unmeasured. See the shipped row below.

## Every technique, where it stands

Status words: **done** means merged, tested and exact; **negative** means it works but
measured no useful gain; **measured** means a number exists and no code was needed;
**shelved** or **dead** means investigated and closed; **proposal** means written up,
not built; **blocked** means it needs a machine holding the full checkpoint.

### Shipped, exact, tested

| Technique | What it does | Status | How deep the evidence goes | What is left |
|---|---|---|---|---|
| Streamed inference engine (upstream) | Reads weights from disk during each token instead of holding the model in RAM | done | 12 memory budgets, one identical token sequence, 33% run-to-run speed noise ([ladder](data/memory-ladder.tsv), [noise](data/replication.tsv)) | Re-run with 3 repeats; needs a checkpoint host |
| Trunk read-ahead ring (upstream) | Reads the next layer while the current one computes | done | The 1.70x once quoted here is the memory ladder's 224 GB against 8 GB ratio (19.21 against 32.69 s per token), captured before the ring existed; the ring's own gain is unmeasured ([queue](notes/research-queue.md)) | At 8 GB only one slot fits, so the overlap is lost there |
| Hybrid draft decode, `--draft-trunk` (upstream) | A cheaper copy proposes tokens; the exact model verifies them | built; negative streaming result | The int8 container and kernel were built; streaming routed experts made the draft slower ([note](notes/int8-draft-container.md)) | Draft agreement with a verifier is not main-model quality validation |
| Offline lossless archives, K3ZSTD1 (#1) | Compressed weights the engine reads directly, no network | done, opt-in | Native tests; 32 real samples ([doc](OFFLINE_STORAGE.md)) | Experts shrink only 5.7%, so not a route to a smaller model |
| Prompt rereading (#1) | Processes the prompt twice before generating | done, exact | CLI tests | Nothing |
| Calibrated expert pinning (#2, #4) | Keeps the most-used experts resident | done, opt-in; **negative** | Held-out replay saves 0.44% of expert reads at 8 GB; the earlier 36% hit rate was an artifact of repeated prompts ([doc](EXPERT_PROFILES.md)) | Dead as a speed lever |
| Exact SIMD KDA recurrence (#5) | Hand-vectorized inner loop of the recurrent layers | done, opt-in (`KDA_SIMD=1`); **negative** | Bitwise identical across 9,504 checks and under sanitizers; 1.11x on x86, 0.72x on ARM, both inside the noise floor ([note](notes/kda-simd.md)) | Leave off by default |
| lm_head streaming, `--stream-lm-head` (#5) | Streams the 2.35 GB output table instead of keeping it resident, funding a second trunk slot | done | Synthetic mechanism gate under a 64 MiB memory cap: ring grows from 1 to 2 slots with identical logits ([note](notes/stream-lm-head.md)) | Full-model effect unmeasured |
| Quality harness, `tools/quality.py` (#5) | Scores how well a model predicts held-out text, so a changed model can be compared with the exact one | built, never run on K3 | Protocol tests and a tiny native check ([doc](QUALITY.md)) | Needs a checkpoint host; about 1.4 TB of reads per 128-token window |
| Selective scale archives, K3ZMAP1 (#6) | Compresses only the 5.9% of expert bytes that compress (the scales) and reads the rest raw | done, opt-in | 24 storage tests, sanitizers, synthetic CLI parity ([doc](SELECTIVE_SCALES.md)) | Saves 0.95% of bytes per token; unmeasured on the real model |
| Direct I/O for selective raw extents (#8) | Reads the raw parts of a selective archive past the page cache, the way plain shards are read | done | Native test proves the direct path is taken and falls back cleanly; bytes identical | Speed unmeasured on the real model |
| Known-route expert pipelining, `--expert-pipeline` (#9) | Publishes each routed expert as its read lands instead of waiting for the whole top-k | done, opt-in | Native pipeline-mode cases plus sanitizers; CLI parity for full recompute and `--incremental` at pool sizes 1 and 16 ([note](notes/expert-pipeline.md)) | Speed unmeasured on the real model |
| MLA latent KV cache, `--kv-latent` (#10) | Keeps the attention cache in its 512-wide latent form and rebuilds keys and values on use: 55 KB per position instead of 2.37 MB, 42.8x less | done, opt-in | GATE 3b holds all 20 incremental steps bit-identical between layouts; state save/load round-trips; 20 CI checks green ([note](notes/kv-latent.md)) | Costs one kv_b matmul's worth per cached position per query token, applied as its key rows to score and its value rows to weight; until ea6f419 it was two whole matmuls, each using half its rows (found in the 2026-09-24 review, then split at identical bits); speed unmeasured on the real model |
| Bounded trunk rows, `--trunk-rows` (#12) | Streams each trunk matrix through two small double-buffered row tiles instead of whole layers, with unchanged arithmetic | done, opt-in | Two 93-layer walks over position-dependent fixture bytes, ThreadSanitizer and ASan/UBSan lifetime runs, CLI logit parity across trunk, cache and expert modes, a 64 MiB cgroup ([results](notes/research-results.md)) | Real speed unmeasured; `--kv-latent` rereads `kv_b` for every rebuilt position, from a plain `trunk.bin` only the half each pass applies |
| Replay-free speculative rollback (#14) | A partly accepted `--spec` sweep commits only its accepted positions, from a per-position KDA log, instead of restoring a snapshot and replaying them | done | Oracle GATE 4: 14 verify sweeps at every accept count 0 to 4, both KV layouts, no replay; CLI parity with serial decode in every memory mode, saved state equal to serial's | Speed unmeasured on the real model; the drafting rule is unchanged ([study](notes/spec-replay.md)) |
| Batched trunk matmul and lm_head blocks (#14) | Applies each trunk matrix, and lm_head, to every position of a prefill or verify sweep in one pass | done | Bitwise against the one-position kernel on cancelling data (`test_ops`), oracle GATE 3c, CLI byte counts at 1 to 130 positions; about 2x per position at 8 positions on the VM ([timing](notes/research-results.md#batched-kernel-timing)) | Full-checkpoint comparison needs a checkpoint host |
| Faster exact decode kernels and router (#14) | Less work per weight in the decode matmuls and the MoE router, the same bits | done | `test_matmul_exact` and `test_ops` hold every partition and reduction order on cancelling data; bf16 at 82% to 93% of the VM's read rate, router 1.8x faster ([note](notes/decode-kernels.md)) | Kernel timings only; no s/token claim |

### Measured, no new code

| Technique | What it answers | Status | How deep the evidence goes | What is left |
|---|---|---|---|---|
| Scale-plane entropy campaign (#5) | How compressible the expert scale bytes are | measured | 264 MB from 256 real experts across 8 shards: 15.5% retained under Zstd ([note](notes/scale-plane.md)) | Projects 85 GB to 13 GB; the checkpoint stays about 1.49 TB |
| Packed-expert compressibility (#1) | How compressible the main expert bytes are | measured | 24 MiB of real samples: 94.3% retained ([samples](measurements/lossless-samples.json)) | Dead end; these bytes are already at their entropy |
| Memory ladder and I/O split (upstream) | Speed against RAM budget | measured, single sample | 32.7 s/token at 8 GB, 19.2 s/token at 224 GB, on 124 cores; disk is 57% of the time at 8 GB | Share on a small core count unknown |
| Jetson Orin Nano Super proof (upstream) | Full checkpoint on an 8 GB-class board | measured | 4 runs, 949 s per token including a 5-token prefill, 208 GB read per run | Proof of life, not usable speed |
| Streaming option map and bounds (#6) | 26 ways to change bytes, overlap or compute, with the math checked | written | Ceilings: removing all expert reads gives at most 1.24x; removing all trunk reads at most 5.2x, disk time only ([map](notes/streaming-options.md)) | A menu, not code |
| MLA cache variants (#14) | What each way of caching MLA attention costs, and whether absorbing `kv_b` stays exact | measured, benchmark only | Three variants held bitwise to the engine in `make test`; the absorbed one moves outputs by about 1e-6 relative, so it cannot be a mode; kv_b counts and timings on the VM and hosted arm64 ([note](notes/mla-variants.md)) | Rebuilding each position once per call (L1) is the candidate for `--kv-latent` at T > 1; not in the engine. The engine's loop (L0) now applies only the key rows, then only the value rows, of kv_b (ea6f419), which halved its count at identical bits and matches L1's at T = 1 when L1 holds every value row (vcap >= N) |
| `--spec` drafting policies (#14) | Which n-gram drafting rule pays, per workload, under the verify-step cost model | measured on a text proxy, not K3 | Teacher-forced replay of four corpora with a proxy tokenizer: the current rule pays on whole-file edits and little elsewhere ([note](notes/spec-replay.md)) | K3's own outputs through `spec_replay.py replay-ids` |

### Built and shelved, or studied and closed

| Technique | What it would do | Status | Why it closed | What would reopen it |
|---|---|---|---|---|
| Huffman-coded trunk (upstream) | Smaller byte-identical trunk | old prototype shelved; new decoder experiment built in #12 | Historical single-stream decoder was too slow; new x86/ARM range benchmarks explicitly report reconstructed BF16 GB/s ([results](notes/research-results.md)) | Supported container/reader and concurrent-compute gate; kernel speed alone is insufficient |
| Expert pruning (studied) | Drop rarely used experts | dead | Held-out coverage plateaus at 35%; usage is deliberately flattened by the router | It changes the model, so the quality harness first |
| Shared base plus low-rank delta (studied, closed PR #3) | Store one expert per layer plus small differences | dead on paper | Needs 0.99 correlation between experts; real expert weights look random. The write-up itself had errors and was closed unmerged | Kept only at PR #3 for the record |
| Int8 trunk as the main model (upstream note) | Halve the trunk by rounding | not validated | The 90.9% figure was a 22-token draft with the exact model verifying, not a quality result | A real quality evaluation |

### Research gates from #12 to #14

| Technique | What it would do | Status | What is known | Size of the job |
|---|---|---|---|---|
| Fixed-width trunk dictionary | A 4-bit high-byte index into one 15-entry table, low byte raw; SIMD table lookup instead of entropy decode | gates 1–3 passed in CI (#13); the per-family gate is a STOP on two families (#14); benchmark-only, not in inference | 99.95% coverage on eight dense ranges, r = 0.7502; byte-exact on x86/ARM under sanitizers; 14.8–26.7 reconstructed GB/s with every SIMD run above the 4 GB/s target. In #14: 3 bits beat 4 by 4.68 points on those ranges and 3.78 byte-weighted over all 23 families; the 3-bit decoder is byte-exact and 9.6–15.3 GB/s in hosted CI; the row index costs 0.034 points. By the rules fixed before the data, the router takes its own table, `moe.shared_down` stays raw, and per-family widths give r = 0.7329 over all matrices. Decode beside matmul threads keeps a worst-case streamed speedup above 1 up to a 4 GB/s SSD in every run on a 4-vCPU VM (5 GB/s is marginal: it passed in the idle runs and fell just below 1 in one run taken during background downloads), and at every rate to 6 GB/s on both hosted runners ([note](notes/fixed-width-trunk.md), [results](notes/research-results.md)) | A supported reader, and the placement a real row pipeline sees. By design the ratio stays above Huffman's: 7.80 points on the four `f_a_proj` ranges, 5.58 points above the per-family Huffman bound for the plan (about 2.3 of them the raw `shared_down` fallback the bound does not take; 3.57 with every family coded). The design has public precedents, ZipServ (ASPLOS'26) and two GitHub projects a day earlier ([note](notes/fixed-width-trunk.md#prior-art-added-2026-09-24-after-review)) |
| Next-layer expert prefetch | Would predict upcoming routes; true routing still decides computation | closed in #13 on the traffic arithmetic; diagnostic only in #12 | Synthetic validation only; the [audit](notes/predictive-prefetch-gates.md) records k=1/2/4, the equal-slot static null and bytes per decode token as unmeasured. Toy arithmetic, not a measurement: fetching 16 experts at an assumed 70% recall, with no cache hits, no cancellation and every correct guess kept until use, adds about 5.76% to whole-token traffic in the streamed-trunk regime, because experts are 25.83 of the 134.64 GB one position streams (with the trunk resident the same guesses would add about 30% to disk bytes; not analysed) | Not reopened by a generation capture; no engine predictor |
| Bounded lookahead verification | Drafts without a matching history suffix | reference and cost gate built in #12; no engine mode | Exhaustive toy-model exactness; proposal and replay work explicitly charged ([results](notes/research-results.md)). Since #14 `--spec` rolls back without replay, so a lookahead mode would pay no replay cost (R = 0) | K3 acceptance and total work, including rejected-work reads, need a checkpoint host |
| io_uring reads | Linux asynchronous read submission | standalone experiment built in #12 | Three runs per arm at five queue depths; noisy overlapping timings, no consistent meaningful gain; no report committed, byte verification inside the timed region, no compute arm, so the queue's gate (a benefit under concurrent compute) was not exercised ([results](notes/research-results.md)) | No engine backend replacement justified |
| Speculative decoding on resident hardware | A cheap draft proposes tokens, the exact model verifies; ~1.7x fewer weight bytes per accepted token at the measured 66.7% acceptance ([note](notes/int8-draft-container.md)) | proposal | Only pays off once the model is resident in RAM; nothing on 8 GB, where both draft and exact stream from disk | Needs a large-memory host to pay for |
| Chunked prefill, sampling, chat template, vision, HTTP serving | Usability features from the upstream roadmap | not started | n/a | Do not change size or speed |

### Blocked on a machine that holds the checkpoint

Full-model seconds per token on a laptop-like core count, a thread-count sweep, real
8 GB overlap timing, and any real quality number. Nothing has run on a full checkpoint
since the rental. Doing it needs roughly 3 TB of NVMe and hours to download 1.56 TB.
Each quality window reads about 1.4 TB, so a 10,000-token corpus is on the order of
a day of disk time.

## Next decisions

1. **Organization, now.** Delete the merged branches (each merged pull request page
   has a "Delete branch" button) and turn on *Settings, General, Pull Requests,
   Automatically delete head branches* so this stops recurring.
2. **Checkpoint-free engineering, in order of value per risk.** The lint findings
   and direct I/O for selective raw extents landed in #8; known-route expert
   pipelining landed in #9; the latent KV cache landed in #10. #12 implemented bounded
   trunk rows and gave the four other proposals executable research gates; #13 closed
   predictive prefetch and built the fixed-width dictionary falsifier, whose first three
   gates passed in CI; #14 added replay-free `--spec` rollback, the batched and faster
   exact kernels, and the fixed-width follow-up gates.
   [notes/research-queue.md](notes/research-queue.md) records what each proposal became
   and which gates remain open. Read the measured results before selecting further
   integration work.
3. **The rental remains the only way to measure.** Everything marked blocked needs one
   machine with the checkpoint for a day or two. Without it there will never be a
   laptop speed or quality number; with it, one core-limited ladder run answers the
   decisive question.
4. **Remote K3 or not.** Remote serving changes the offline requirement. Its
   feasibility and current cost are a separate decision; no remote service is
   provisioned by this work.

## Where things live

| | |
|---|---|
| [ROADMAP.md](ROADMAP.md) | engineering priorities, in order |
| [PERFORMANCE.md](PERFORMANCE.md) and [data/](data/) | every measured number and its raw output |
| [TESTING.md](TESTING.md) | what each test proves |
| [notes/](notes/) | one write-up per experiment, including the negative ones |
| [measurements/](measurements/) | machine-readable records behind the notes |
