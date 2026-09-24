# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Exact batched trunk matmul** for T > 1 positions (prefill, speculative
  verification, draft prefill): `k3_matmul_bf16_batch`, `k3_matmul_batch` and the
  `k3_mmw_batch` dispatcher apply a matrix to every position in one pass, widening each
  weight once per block of positions, with every output bit-identical to the
  single-position kernel (same partition, fma order, reduction tree and tail; scalar,
  AVX2 and NEON). KDA, MLA, the MoE trunk and the dense layer use it; under
  `--trunk-rows` a batch reads each matrix once instead of once per position, at any
  prompt length: the streamed MoE prefill deduplicates routed experts over 64-position
  sub-chunks but applies its five trunk matrices to the whole batch.
  T == 1 takes the existing kernel. `k3_moe_scratch` now takes T. Gated bitwise by
  `test_ops` on cancelling inputs that expose a wrong summation order, by the oracle's
  GATE 3c, by a rows-mode byte-count check in `test_offline_cli.py` at 1 to 130
  positions, and by 65-, 129- and 130-token prompts matched against the per-token MoE
  (`K3_NO_BATCH_PREFILL`). That CLI check runs the tiny checkpoint's top-2 routing,
  where a position's two routed terms sum to the same float in either order, so the
  MoE prefill's one ordering hazard, summing a position's experts in fetch order
  instead of top-k order, is held by `test_ops`'s `moe_prefill` gate instead: top-16
  of 40 experts over 130 positions, bitwise against `k3_moe`, with an in-test check
  that a reversed or fetch-order sum would change most positions, and one expert read
  per unique expert per sub-chunk. Positions share a widened weight in register blocks
  of 4 on AVX2, 8 when AVX-512VL gives the compiler 32 vector registers, and 2 on NEON.
  On the reference VM a block of 8 takes 7% to 13% less time than 4 with AVX-512VL (a
  tie at 16 positions on four threads) and 2% to 9% more on plain AVX2 (`bench_batch`,
  8 to 16 positions; the AVX2 differences sit inside the timing note's 10% noise
  band, so the AVX2 default of 4 rests on the register count as much as on them); the
  block is a loop shape only and cannot move an output. Against
  the exact one-position decode kernel, batching takes a 12288 x 7168 projection of 8
  positions from 160 to 83 ms on one thread and from 45 to 21 ms on four, and a
  16-position lm_head block from 1,319 to 581 ms on four: about 2x, not the 3x an
  earlier timing against the older kernel showed (that earlier timing's raw data is
  not in the repository). Every run is in
  [docs/notes/research-results.md](docs/notes/research-results.md#batched-kernel-timing).
  **Public API:** `K3WeightStream` (include/k3/k3.h), the streamed-matrix descriptor
  that `--trunk-rows` introduces, carries an optional `apply_batch` callback besides
  `apply`. `k3_mmw_batch` calls it for a `K3_WSTREAM` matrix, so every layer op reaches
  it on a multi-position forward; each output must be bit-identical to `apply` on that
  position alone. NULL falls back to one `apply` per position, so a stream built on the
  stack must zero the field.
- **lm_head batched over positions** where a forward needs every position's logits:
  a `--spec` verify sweep projects its positions in one pass over the head, and
  `--tf-check` / `--score-prompt` in blocks of up to 16, through `k3_mmw_batch`, or
  under `--stream-lm-head` through `k3_model_stream_project_batch`, which reads each
  chunk of the head once per block instead of streaming the 2.35 GB head once per
  position. Every logit is bit-identical to the one-position projection; the block
  (10.5 MB at most on K3) is counted in the memory plan and reported as
  `logit_block_bytes`. `test_offline_cli.py` checks NLLs and `--tf-check`
  predictions on both sides of each block boundary against one-position runs, and
  that a streamed-head verify sweep reads the head exactly once.
- **MLA cache variants study** (benchmark-only; the engine's arithmetic is unchanged):
  one MLA layer's cached attention five ways in `benchmarks/mla_variants.h` -- expanded
  (E), the same arithmetic threaded (E+), the engine's latent loop (L0), a latent loop
  that rebuilds each position once per call (L1), and absorbed (A). E+, L0 and L1 are
  held to `k3_mla_cached` bit for bit by `test_mla_variants`, now in `make test`, on
  ordinary, cancelling and sharp layers, including the double softmax normalisers and
  probability quotients the output rounds away, which the engine exposes through a new
  test-only hook, `k3_mla_trace` (NULL outside tests); a mutation run shows the gate
  catches reordered chains in the variants and in both engine layouts. `bench_mla
  counts` counts kv_b applications: at a 256-token prefill L0 makes 65,792 per layer and
  L1 256 (L0 applies all of kv_b on both of its passes and uses only the key rows in
  one and the value rows in the other, so a row split would halve its count at
  identical bits; found in review, not done). A differs from E by ~1e-6 relative, as
  much as E differs from a double reference, with no argmax change in 3 x 10,000
  attention-score rows of a flat synthetic softmax; it is not bitwise, so it
  cannot be a mode under the exactness contract. Timed on the idle 4-core VM with the
  shipped kernels (one layer, four threads): E+ is 4.8-5.5x faster than E for one new
  token and 6.6-15x for five; L1 halves L0 at decode and takes a 256-token prefill from
  43 s to 0.53 s per layer, but rebuilding stays 34-51x slower than E+ at decode; A is
  the fastest at long contexts. The hosted macOS arm64 and Ubuntu runs pass the same
  gates and counts. See [the note](docs/notes/mla-variants.md).
- **Fixed-width trunk dictionary gates** (benchmark-only, not in inference): a
  4-bit high-byte index into one pooled 15-entry table with an escape code, the
  low byte raw, decoded by SSSE3 `pshufb` / NEON `tbl`. The eight-range histogram
  falsifier passes (99.95% coverage, r = 0.7502); byte-exact round trips pass on
  x86/ARM under sanitizers; the rate gate passes with every SIMD run at
  14.8–26.7 reconstructed BF16 GB/s against a 4 GB/s target. Each ISA's slowest run
  is 14.7x (x86_64) and 11.0x (arm64) the compact Huffman kernel's best run on the
  four KDA `f_a_proj` ranges (different CI runs, runners and buffer sizes, so
  indicative rather than matched), where its ratio trails Huffman's by 7.80 points
  (0.7503 against 0.6722), missing the six-point requirement. The design has public
  precedents (ZipServ, ASPLOS'26; dgpp and NWC on GitHub a day earlier), recorded in
  the note's prior-art section. Follow-up gates, run in
  hosted CI (run 35845709912) except where a VM is named: a bit-width curve with
  entropy bounds (3 bits beat 4 by 4.68 points on the eight ranges and by 3.78
  byte-weighted over all 23 families); an FD3B 3-bit decoder, byte-exact under
  ASan/UBSan on SSSE3, AVX2 and NEON at 9.6 to 15.3 reconstructed GB/s pooled; the
  FDRX row index for whole-row seeking (0.034 points per row, zero padding, exact
  from the released shapes); samples of all 23 trunk matrix families, a STOP on two
  (the pooled table covers 98.49% of `moe.router` and 98.88% of `moe.shared_down`),
  after which the plan fixed before the data gives the router its own table, keeps
  `shared_down` raw and reaches r = 0.7329 over all 108.76 GB of matrices, 5.58
  points above per-family Huffman (about 2.3 of those points are the raw fallback the
  bound does not take: coding every family at its better width gives 0.7128, 3.57
  points above Huffman; the STOP rule is scored in 4-bit coverage while the widths
  chosen are 3-bit, see the note's correction); and a decode-under-matmul-contention benchmark,
  whose worst-case streamed speedup stays above 1 up to a 4 GB/s SSD in every run on a
  4-vCPU VM with the shipped kernels (5 GB/s is marginal: it passes in the idle runs and
  fell just below 1 in one run taken during background downloads; it fails at 6 GB/s for
  streamed input at 4 and 2 threads) and at every rate to 6 GB/s on both hosted runners. A supported reader
  remains. See [the note](docs/notes/fixed-width-trunk.md) and
  [results](docs/notes/research-results.md).
- **Bounded trunk row streaming**, opt-in `--trunk-rows`: two small read/compute
  buffers, unchanged per-row arithmetic, and an explicitly sized current-layer
  vector arena. Synthetic gates cover compressed/plain logits, a 64 MiB cgroup,
  two 93-layer row walks with wraparound over position-dependent fixture bytes
  (`test_trunk_rows`), ThreadSanitizer and ASan/UBSan. A batch of
  positions reads each matrix once (see the batched matmul entry above); `--kv-latent`
  still rereads `kv_b` per rebuilt position. `--trunk-gb` caps the two buffers in this
  mode and the memory plan charges the whole budget, so pass a small one; the run
  report gives the mode's binds, matrix passes, reader tile time and main-thread
  waits. No full-model speedup is claimed. Fixes the trunk JSON ownership leak, which
  ASan exposed, and a parallel read error-flag race (the sanitizer jobs build the row
  fixture without OpenMP, so ThreadSanitizer did not exercise the parallel loop that
  held it). Every refusal of a malformed
  `trunk.json` now names its reason (no layers, a negative or overflowing run, a tensor
  outside its layer's run, a misaligned offset).
- **Executable research gates** for the other four proposals: a compact
  four-stream/two-symbol Huffman decoder benchmark with pinned K3 range samples,
  prompt-separated routing diagnostics, bounded Jacobi/lookahead cost analysis,
  and raw-syscall io_uring versus pread-pool experiments. They do not add default
  inference behavior. See [research results](docs/notes/research-results.md).
  The decoder is 2.63x/3.35x faster than the old kernel by median on the four
  sampled K3 ranges (hosted x86/ARM respectively), all KDA `f_a_proj` and together
  0.12% of trunk bytes, but fails the strict every-run throughput gate and remains
  outside inference. The comparison has two arms, so it does not separate the three
  changes; the ARM median rests on three runs spread from 0.94 to 1.35 GB/s; and the
  shelved note's Huff0 figure, 1724 MB/s on the same kind of bytes, is about three
  times this prototype's high-byte symbol rate. No model-level gain is claimed.
- **Predictive-prefetch gate correction:** routing diagnostics require declared
  source-layer lead and decode phase; zero lead is oracle-only. Removed inferred
  read counters and the 70% promotion flag. The [ordered audit](docs/notes/predictive-prefetch-gates.md)
  records all real k=1/2/4 scores as unmeasured, the equal-slot static null, and
  whole-expert byte accounting. The route is closed on its traffic arithmetic: at an
  assumed 70% recall with 16 guesses per layer, wrong guesses that are never cancelled
  add about 5.76% to whole-token reads in the streamed-trunk regime, where experts are
  19.2% of the bytes (with the trunk resident the same guesses would add about 30% to
  disk bytes; that regime is not analysed), and no generation capture reopens it. The
  existing prefix replay is not eligible evidence.

- **Research queue**, `docs/notes/research-queue.md`: what each of the five
  checkpoint-free proposals became, the gates each still has open, and the arithmetic
  that bounds them; a per-machine ordering drafted from the memory ladder was withdrawn
  because its premises (a prefill-averaged ladder captured before the asynchronous
  reader and the fused kernels) no longer hold, not because a measurement contradicted
  it. The status board points at it, carries the research-gate rows, and
  lists `--kv-latent`, `--trunk-rows` and replay-free rollback under shipped with their
  evidence.
- **`--kv-latent`**, off by default: the incremental decoder's MLA KV cache holds only
  the `kv_lora_rank` latent and the shared rope row per position, and rebuilds the
  per-head k and v through `kv_b` on every use, which is what MLA's own design caches.
  That is 0.055 MB per position across the 24 MLA layers instead of 2.37 MB, 42.8x less,
  and it turns a 131,072-position context from 310.04 GB of cache into 7.25 GB. It is
  paid for in arithmetic: every cached position is re-expanded twice per decode step,
  once to score and once to weight the values. Output is BITWISE identical, not close:
  the latent stored is exactly the bytes the expanded path fed to `kv_b`, the rebuild
  uses the same kernel, and every softmax reduction keeps its order, so GATE 3b of the
  oracle compares all logits of all steps rather than tokens. The memory plan, the KV
  line and `k3_run.json` report the layout actually allocated; `--save-state` records it
  in the header and a cross-layout `--load-state` is refused by name rather than read at
  the wrong stride. Needs `--incremental`; full recompute is untouched. See
  [docs/notes/kv-latent.md](docs/notes/kv-latent.md).

- **`--expert-pipeline`** (also `K3_EXPERT_PIPELINE=1`), off by default: the routed-expert
  batch prefetch stops waiting for the whole top-k before it returns. The routes are known
  before the first read is issued, so the reads are handed to a small pthread pool in
  CONSUMPTION order and each slot is published the instant its read lands; `get()` then
  blocks only on the expert the MoE needs next instead of on the slowest read in the batch.
  `K3_EXPERT_PIPELINE_THREADS` sets the pool size (default 4, capped at 16). Nothing about
  routing, top-k, the combining weights or the float summation order changes, and with the
  flag off the cache is byte-for-byte the code that shipped, stats included. In pipeline
  mode `load_seconds` is the wall clock from a batch's launch to its last completion, since
  summing overlapped per-read durations would report a bandwidth the device never delivered.

- **`--stop-id N`** (repeatable, up to 8): generation halts as soon as the model emits
  a listed token id. Off by default, so `--gen N` still means exactly N tokens for
  every benchmark and oracle gate. The stop id stays in the sequence, so `--save-state`
  and a later `--load-state` continue from what the model actually produced, and the
  check runs at emit time so a `--spec` sweep is truncated at the stop exactly like
  serial decode. `k3_run.json` gains `"stopped_at"` (the id, or -1). Parsing is with
  `strtol` and refuses a non-integer, a negative, or an id past the vocabulary, since a
  stop the model can never emit is indistinguishable from a model that never emitted
  one.
- **Prefill on `--gen 0`**: `--gen 0 --incremental --save-state` now runs the prompt's
  prefill and saves its exact KV and recurrent state with zero generated tokens, so a
  shared prefix (a system prompt, a long document) can be warmed once and resumed many
  times with `--load-state`. Previously `--gen 0` skipped the decode loop and saved
  nothing useful.
- **Windows support**: builds natively via MSYS2's MinGW-w64 GCC, no WSL required.
  `make`, `make test`, and `make test-all` pass every gate unmodified, including the
  full-model oracle and tokenizer parity (45/45) against real Kimi K3 weights.
  `src/io/k3_portable_io.h` gained a Windows branch alongside the existing Darwin one,
  porting `O_DIRECT` (via `FILE_FLAG_NO_BUFFERING`, intercepted at `open()` since
  Windows -- unlike Darwin -- cannot add it to an already-open handle), `pread` (via
  `ReadFile`'s `OVERLAPPED` offset fields, chosen specifically because it does not
  share mutable file-pointer state across threads the way `SetFilePointerEx` +
  `ReadFile` would), `posix_memalign` (via `_aligned_malloc`), and `getrusage`/
  `MemAvailable` (via `GetProcessMemoryInfo`/`GlobalMemoryStatusEx`). `make asan`/
  `make ubsan` switch to Clang on Windows (MinGW-w64's GCC package ships no sanitizer
  runtime at all, confirmed directly rather than assumed).

### Changed

- **Decode matmul kernels do less work per weight, with the same bits.** `k3_matmul`,
  `k3_matmul_bf16` and `k3_matmul_mxfp4` widen x to double once per call instead of once
  per row (x86 reads it in an even/odd layout that drops the bf16 zero-extend shuffles);
  x86 bf16 and the AVX-512 MXFP4 path take two rows per pass sharing each x load; x86
  issues a software prefetch ahead of each streamed row; and an AVX-512 build now has
  its own bf16, fp32 and MXFP4 flat paths instead of running the AVX2 ones. Every path
  keeps the accumulation partition and reduction tree it had, so bf16/fp32 stay
  bit-identical across scalar, AVX2, AVX-512 and NEON, and MXFP4 keeps its per-ISA bits
  (AVX-512 reproduces AVX2). A new gate, `test_matmul_exact`, re-implements each order
  in plain C and compares every bit on cancelling data built so that a different
  partition or tree changes the float; in-test it sums the same data in four named
  wrong orders (sequential, the other tree, rotated lanes, lanes 0 and 1 swapped) and
  requires each to be rejected in at least 10% of the eligible rows (39% to 92% in
  practice: 2593, 1571, 1050 and 1087 of 2818 or 1241), so the data is shown to
  discriminate against those orders rather than against every conceivable one; a
  CI job runs it on the AVX-512 build whenever the runner has AVX-512. Every NaN output
  of `k3_matmul` and `k3_matmul_bf16` is now the quiet NaN `0x7FC00000`: a NaN's sign
  and payload are not fixed by the summation order, and the two rows of a bf16 pair
  could pass on different ones, so a NaN row's bits followed its place in a call, and
  the row pipeline's call lengths follow the memory budget. On x86, `k3_matmul_mxfp4`
  with a group that is a multiple of 16 (K3's is 32) aborts if it cannot allocate its
  copy of x, instead of falling back to the grouped path, whose different order would
  have changed the bits. `bench_kernels` reports the median and best call
  (`K3_BENCH_REPS`) and the machine's streaming read bandwidth beside the bf16 rate.
  Quiet-machine timings, old kernels against new in one harness on a 4-vCPU AVX-512
  Xeon guest, are in [docs/notes/decode-kernels.md](docs/notes/decode-kernels.md): bf16
  12288 x 7168 is 1.50x faster at 1 thread with AVX-512 and 1.34x with AVX2 (1.44x and
  1.27x at 4 threads), now 82% to 93% of the machine's plain read rate; MXFP4 is 2.4x to
  2.9x faster on AVX-512 and unchanged on AVX2. Kernel figures only, no s/token claim.
- **The MoE router scores eight experts per pass over x, with the same bits.**
  `k3_router` used to run each expert's 7168-term double chain alone, so a core waited
  out the add latency on every element; it now walks eight experts side by side, each
  still receiving its terms in order, one add per term, so every score, index and
  weight is unchanged. `test_ops` holds it bitwise to the one-expert loop on
  cancelling data where a reversed or split chain changes every score (checked
  in-test), at 1 to 896 experts. The new `bench_router` times one call at 896 x 7168:
  on the 4-vCPU AVX-512 guest, old kernel against new in one harness, 8.4 to 4.6 ms on
  one thread and 2.1 to 1.2 ms on four (AVX2 build: 8.4 to 4.8 and 2.1 to 1.2), about
  0.09 s per token over the 92 MoE layers at four threads; every run is in
  [docs/notes/decode-kernels.md](docs/notes/decode-kernels.md#the-moe-router).
- **Speculative decode never replays.** A partially accepted `--spec` sweep used to
  restore a copy of the whole carried state and replay the accepted prefix through a
  second forward, re-reading the trunk and the prefix's experts (at K3 scale 108.81 GB
  plus the routed experts, per rejection). Verify sweeps are now tentative: each KDA
  layer runs on a one-layer work copy and records its recurrence inputs
  (`K3KdaLog`, `k3_kda_layer_log`), and only the positions behind emitted ids are
  committed with `k3_kda_advance`, bit-identical to serial decode and reading no
  weights. MLA needs nothing: its KV rows are positional. The 626 MB state snapshot
  (all 93 layers at K3 size, 69 of which carry state) is gone; `--spec 4` now holds a
  102 MB log (five positions at 20.4 MB each) plus one 6.7 MB work layer, counted in
  the memory plan. "Never replays" means no second forward and no weight re-read; the
  commit still recomputes the KDA recurrence for the kept positions. The
  hybrid `--draft-trunk` path commits the same way and folds its catch-up into the
  next round's first call, so it runs no replay, catch-up or lockstep sweeps either.
  The run report and `--out` JSON count verify sweeps, acceptances and forward sweeps
  per decode step; the new `--dump-all-logits` writes the logits behind every token,
  and under `--spec` the memory plan counts the sweep's worth of logits it holds
  (`K3_SPEC_MAX + 1` vocabulary rows, `all_logits_bytes` in `k3_run.json`).
  Gated by `test_kda_exact`, oracle GATE 4 and CLI parity tests across memory modes.
- **Trunk layers are read in parallel chunks.** `load_run()` streamed each layer with
  one sequential `pread` loop, so the device saw queue depth 1. It now splits the layer
  into 64 MiB chunks (a multiple of `K3_TRUNK_ALIGN`, so every chunk stays aligned for
  `O_DIRECT` and `F_NOCACHE`) issued under an OpenMP parallel for, matching what the
  expert path already does. Without OpenMP the loop still runs one chunk at a time. A
  short read in any chunk fails the whole layer, as before, and the decoded output and
  `trunk_bytes_read` are unchanged.

### Fixed

- **`--spec` saved a state ahead of its sequence** when a `--stop-id` cut a verify
  sweep short: the carried state kept every accepted position, so `--save-state` wrote
  a state that had consumed ids the saved sequence did not contain and a resumed run
  continued from the wrong context. The sweep now commits exactly the positions behind
  the ids it emits, and the saved file is byte-identical to serial decode's.
- **`--dump-logits` with `--draft-trunk` recorded the draft model's logits**: the draft
  wrote its prefill logits into the exact model's buffer. The draft has its own now.
- **`k3_run.json` was not valid JSON after a run that generated nothing.** With
  `nout == 0` the `seconds_per_token` field computed `t_total / nout` and emitted a
  bare `inf`, so a harness driving `--gen 0 --save-state` failed on the one run it
  needed to parse. It now reports `0`.
- **Heap corruption on Windows** (`STATUS_HEAP_CORRUPTION`) in the trunk and expert-
  cache arena allocators: `_aligned_malloc`, which backs the Windows `posix_memalign`
  shim, must be freed with `_aligned_free`, not plain `free`. POSIX's `posix_memalign`
  carries no such restriction, so this compiled cleanly and only crashed once the
  corrupted allocator metadata was actually used, well after the allocation itself.
  Three call sites needed the fix: `k3_cache.c`'s cache arena, and `k3_trunk.c`'s
  trunk arena and per-layer pinned buffers.
- **`SHARD_DIR`/`TOK_FILES` unquoted in the Makefile**: a path containing a space
  (routine on Windows, e.g. an "AI LOCAL MODELS" folder) silently split into extra
  argv entries instead of failing loudly, and `test_expert`/`test_real_layer`/
  `test_tok`/`test_cfg` read whichever truncated token happened to resolve to a path,
  rather than refusing outright.

## [1.0.0] - 2026-08-07

Verified end to end on the full released checkpoint, and made substantially faster, with
byte-identical output preserved at every step. The first-run experience, which was broken
on a clean clone, now works.

### Added

- **`--preset auto`**: sizes the trunk and expert-cache budgets from the machine's own free
  RAM, trunk-first, so a user need not pick a preset by hand. A gigabyte given to the trunk
  is worth far more than a gigabyte of expert cache, and auto pins accordingly, capping the
  pin below the RAM ceiling after a heavy-pin regression was measured.
- **Chunk-union prefill**: a batched-prefill MoE that fetches each unique routed expert once
  per chunk instead of once per token, measured to read about half the expert bytes on a
  prompt, with the generated token bit-identical to the per-token path.
- **Conversation resume** (`--save-state` / `--load-state`): carries the recurrent state and
  KV cache to disk so a second turn resumes instead of re-reading the whole prompt, measured
  3.9x faster on turn two with identical output. Refuses to restore state from a different
  architecture.
- **`--spec N`**: speculative decode by n-gram drafting with batched greedy verification;
  output is exactly the serial greedy decode by construction.
- `--tf-check`, teacher-forced agreement over an id sequence in one sweep, for measuring
  draft quality; `tools/qdq_trunk.py` and `tools/int8_trunk.py` for deriving quantized
  trunks.

### Changed

- **Fused matmul kernels** (fp32, bf16, MXFP4): sixteen partitioned accumulators with
  explicitly fused products, taking the trunk matmul to its memory floor (about eight times
  less per-token compute) while keeping the scalar and AVX2 paths bitwise identical.
- **KDA recurrence parallelised over heads**, bit-identical to the serial form.
- `scripts/k3-doctor.sh` per-preset speed expectations refreshed to the v1.0.0 numbers, with
  the streaming presets noted as disk-bound and the resident tier as compute-bound.

### Fixed

- All shell scripts are committed executable; the first documented command no longer fails
  with Permission denied on a clean clone.
- `scripts/download-model.sh` uses the current `hf` CLI and pins an immutable revision with
  checksum verification; it no longer attempts a pip install that cannot succeed on the
  target OS, and refuses to start without free space for the checkpoint.
- `scripts/k3-doctor.sh` no longer fails a machine that can build and test the engine; the
  memory floor is a warning about running the checkpoint, not a hard stop.
- The config-refusal fixtures the docs describe now exist and are gated in `make test`,
  ctest and CI; the tokenizer leg reports NOT RUN rather than passing silently; CI runs
  `make test` rather than a hand-picked subset.
- A silent-corruption path in the MLA KV overflow and one in the single-slot trunk reader,
  both of which could emit a plausible wrong token, now abort or are prevented.
- The MXFP4 packer alignment and the tiny-checkpoint scale rule.

### Research notes, not shipped as features

- Lossless trunk compression and a quantized-self-draft hybrid were both built and measured,
  and both turned out to help only narrow regimes. The findings and prototypes are kept in
  [`docs/notes/`](docs/notes/).

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
