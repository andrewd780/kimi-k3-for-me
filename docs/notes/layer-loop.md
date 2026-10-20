# Layer looping, `--loop`, `--loop-layers` and adaptive depth (opt-in; changes the model)

**Status:** plan and design written 2026-10-19 before any code; the status table at the end is updated as work lands.

Andrew asked for "real looped Kimi K3 with the loop architecture" (2026-10-19), after reading two looped-transformer
papers (j-research `findings/105`, `findings/107`). He chose, from three options, **"Loop K3's own layers"**: an experimental,
off-by-default flag that re-runs a span of K3's existing layers per token, with no training.

## What this is, and what it is not

- **It is** the only "looped K3" that exists without training: a span of whole AttnRes blocks runs `R` times per token,
  as if the span appeared `R` times in the stack (the layer-duplication used by passthrough "frankenmerges").
- **It is not** a looped model in the trained sense. Ouro and Huginn are trained so that every loop is meaningful; K3 was
  trained as one 93-layer stack. **The expected effect on quality is a loss**, and the size of that loss is unmeasured. This
  repository has no full checkpoint to measure it on.
- **It is opt-in:** `--loop` runs the middle block adaptively, and with no loop flag every code path is byte-identical to
  the released model; the existing gates prove that, and a `--loop-count 1` gate proves the flag's own identity path. It
  was the default from Andrew's word of 2026-10-19 until 2026-10-20, when he left the call to Claude and it went back to
  opt-in (below).
- **It is not faster.** Each extra pass over a span costs that span's trunk and expert bytes again.

## Semantics

`--loop-layers A-B` (zero-based, inclusive) with `--loop-count R` (total passes over the span, 1 to 8; 2 if omitted).
Execution order per token: layers `0..B`, then `A..B` another `R-1` times, then `B+1..N-1`.

1. **Whole AttnRes blocks only.** `A` must start a block (`A % attn_res_block == 0`), and `B` must end one
   (`(B + 1) % attn_res_block == 0`, or `B` is the last layer). K3 does not pass a plain residual stream between layers. Each
   layer rebuilds its input from per-block snapshots plus the running sum of the current block, and a snapshot fires when
   `layer_idx % attn_res_block == 0`. With whole-block spans, every executed layer keeps the position inside its block that
   it was trained at, so `layer_idx` stays the physical index. Each repeated block adds one snapshot source, which the
   softmax aggregators already handle for any count. A span cutting through a block would move every later boundary and is
   refused.
2. **Each executed layer has its own recurrent state and its own KV cache.** A KDA layer's ShortConv history and
   recurrent matrix, and an MLA layer's cache rows, belong to one executed position in the stack, not to the weights. The
   second pass of layer `L` has its own slot, as a duplicated layer would. Sharing the slot would make pass 2 consume pass 1's
   update of the same token. That is the gate mutant (`K3_LOOP_MUTANT_SHARED_STATE`, test builds only).
3. **The AttnRes stride grows** by the extra snapshots: `(R - 1) * ceil((B - A + 1) / attn_res_block)`. One helper,
   `k3_attn_res_stride()`, replaces the five copies of `n_layers / attn_res_block + 2`. It is a runtime field
   (`attn_res_extra`), zero for every loaded config, so the flag-off value is the old one.
4. **Weights are shared, never copied.** The trunk binds the physical layer and the expert cache serves physical
   `(layer, expert)` keys. A repeated layer re-reads or re-hits the same bytes.

## Adaptive depth, `--loop-exit EPS`, and the default

Andrew, 2026-10-19, verbatim: *"Also find a way in the looped layers to make it like recursive depth or something. Also
make all the changes default in Kimi k3"*. Asked what he meant, he chose **"Adaptive per-token depth"** and **"Default on
now, unmeasured"** (over keeping it opt-in, and over defaulting only after a quality gate), with span **"One middle block,
36-47"**, cap **"R max 4"** and exit **"Threshold 0.05"**. Recorded in j-research `workspace/loopcd/PLAN.md`. On
2026-10-20, shown the cost (1 to 3 extra passes, short-context tokens slower than the original engine), he chose **"On, R
max 2"**: the default's cap is 2. Pass 2 always runs, so at that cap no token exits early and the default is the fixed loop
at count 2; per-token depth is `--loop-count 3` or more with `--loop-exit`.

**Semantics.** With `--loop-layers A-B --loop-count R --loop-exit EPS`:

1. Layers `0..B` run on every token (pass 1 of the span is the stack's own).
2. Pass `p = 2..R` of `A..B` runs on the tokens still active. After pass `p`, token `t` stops if
   `||h_p - h_(p-1)|| / ||h_(p-1)|| < EPS`, where `h_p` is its running block sum after the span's last layer in pass `p`
   (what K3 passes between layers; the snapshot the next boundary takes), or if `p = R`. The norms are one double sum
   each over the row in index order. Pass 2 always runs: the first comparison needs two passes, so a token runs 2 to R
   passes.
3. **Pass `p`'s layers see exactly the tokens that reached pass `p`, in order.** Its KDA recurrent state and ShortConv
   history advance only on them, and its MLA cache holds only their rows, compactly. K3's MLA carries no position
   encoding (the rope slot is unrotated), so a compact cache attends exactly as a sparse one would. This is
   Mixture-of-Recursions' recursion-wise caching, without the router.
4. Layers `B+1..N-1` run on every token, each with its own AttnRes stack height: a token that ran `r` passes has
   `r` snapshots of the span's last block. `k3_decoder_layer_inc_logv` takes one block count per token; with every
   count equal it computes exactly `k3_decoder_layer_inc_log`'s floats.

`EPS = 0` never stops early, so it is the fixed loop at count `R`, bit for bit; an `EPS` no change can reach stops every
token after pass 2, the fixed loop at count 2. Batched prompts work: a pass gathers its active rows in order, runs, and
scatters them back. Full recompute, incremental decode and every supported mode make the same decisions on the same
floats, so they stay bit-identical to each other.

**`--loop`, and the default.** `--loop` runs `--loop-layers` on the AttnRes block holding layer `N/2` of the bound
stack (36-47 on K3; 6-8 on the 13-layer tiny model) at `--loop-count 4 --loop-exit 0.05`, bit for bit; `--loop-count` and
`--loop-exit` change its cap and threshold. With no loop flag the CLI runs the released model; `--no-loop` says so
explicitly. `tests/test_loop_adaptive.py` gates both.

**Why it is opt-in (2026-10-20).** The loop was the default for a day, first at up to 4 passes, then at Andrew's "On, R
max 2". At a cap of 2 no token can exit early (the exit compares two passes), so the default was a fixed one-extra-pass
loop. Andrew then wrote *"Lowkey I’m kinda not literate in ai research so whatever you think is good do it"*, and Claude
made it opt-in, for three reasons:
- K3 was not trained to repeat layers;
- no quality measurement exists, and j-research's toy test of training-free looping (`findings/106`) lost accuracy;
- it costs layers per token.

With the loop off by default, the speed work of the same week (j-research `findings/114`) is what a default run gets.

**Cost.** `--loop`'s 2 to 4 passes of a 12-layer block: 13% to 39% more layers per token, with their trunk and
expert bytes. No quality measurement exists; K3 was not trained to repeat layers.

## Modes

Every mode runs the loop, fixed or adaptive, and is held bit-identical to looped full recompute (or, for `--spec` and
`--draft-trunk`, to looped serial decode):

| mode | what the loop needed there | gate |
|---|---|---|
| full recompute, `--incremental`, `--kv-latent`, `--mla-split` | per-executed-position KDA slots and MLA rows; under adaptive depth, per-pass compact MLA rows (`loop_pcached`) | `test_layer_loop.py`, `test_loop_adaptive.py` mode parity |
| `--trunk` ring and pinning, `--stream-lm-head`, `--expert-pipeline`, `--expert-profile`/`--pin-experts` | nothing: bind, prefetch and cache take any layer order; a hint for a pass that adaptive depth then skips is wasted, never wrong | mode parity |
| `--trunk-rows`, `--kvb-cache`, `--ultra-low-memory` | nothing: rows bind on demand; ultra reuses one state slot because each executed position runs once per forward | `test_streaming_row_and_low_memory_modes_at_mixed_depths` |
| `--spec`, `--draft-trunk` | a KDA rollback log per executed KDA position; under adaptive depth, a sweep commits to pass `p` only the kept positions that reached it, and cuts pass `p`'s MLA row count back the same way. The draft proposes unlooped; the looped model verifies | `test_spec_is_serial_decode_at_mixed_depths`, `test_draft_trunk_is_serial_decode_at_mixed_depths` |
| `--save-state`, `--load-state` | state version 2: the loop record (span, count, exit, slots), each pass's cached count and each position's passes; a state is refused by a run under another loop; a state saved after `--spec` is the serial state byte for byte | `test_state_round_trips_under_the_loop_and_spec_saves_the_serial_state` |
| `--tf-check`, `--score-prompt` | nothing beyond forward | `test_tf_check_counts_the_reference_agreement` (against the torch reference), `test_score_prompt_reports_the_adaptive_loop` |
| `--dump-cache-trace` | records requests in executed order, repeats included. `tools/expert_profile.py` parses only unlooped prefix traces and rejects a looped one by its own validation; calibrate profiles with `--no-loop` | `test_cache_trace_records_the_executed_order` |
| `--layers N` | the span must lie inside the first `N` layers; the default takes the middle block of the `N` | refusal and default tests |

## Gates (each must be able to fail)

- **Flag off:** `make test` (the full-model oracle and every existing gate) unchanged.
- **Identity:** `--loop-count 1` gives ids and logits bit-identical to no flag (full recompute, incremental, latent).
- **Reference:** the looped final-position logits match an independent torch reference (`tools/k3_ref.py`, the same
  quantized weights) within `cmp_logits.py`'s yardstick. Two plants show it can fail: the unlooped reference must be
  outside the yardstick of the looped run, and so must the shared-state mutant.
- **Mode parity:** looped incremental, `--kv-latent`, streamed trunk and expert pipeline give ids and logits bit-identical
  to looped full recompute. That is the engine's invariant, extended to the loop.
- **Refusals:** a misaligned span, an out-of-range span, a bad count and every refused combination exit 2 and name
  `--loop-layers`. A production build refuses the mutant switch.

## Not claimed

No quality result and no speed result. The torch reference shows the engine computes the looped stack it says it does;
it says nothing about whether the looped stack is any good.

## Status

| item | state |
|---|---|
| plan and design (this note) | written first, unchanged in substance by the build |
| engine (`src/cli/k3_run.c`, `k3_attn_res_stride()` in `include/k3/k3.h`) | built |
| reference (`tools/k3_ref.py` `loop_order`, `tools/make_tiny_checkpoint.py --loop`) | built |
| gates (`tests/test_layer_loop.py`, 7 tests; `make test` loop-layers contract; CI step in `offline-storage.yml`) | passing |
| adaptive depth, the `--loop` preset (the default for a day, then opt-in), every mode (`--loop-exit`, `--no-loop`, `k3_decoder_layer_inc_logv`, `forward_adaptive`, state version 2, per-executed-position `--spec` logs) | built; `tests/test_loop_adaptive.py`, 15 tests, passing, and under ASan + UBSan |

Adaptive depth and the default, measured on the tiny model, 2026-10-20, Linux x86-64, gcc 13.3:

- **Identities:** `--loop-exit 0` is the fixed loop and `--loop-exit 1e300` the fixed loop at 2, bit for bit, in full
  recompute, incremental and `--kv-latent`.
- **Reference:** at an EPS chosen to split the 14-token prompt between depths (and checked to sit at least 2 % away from
  every change the run measures), per-position depth equals `forward_adaptive`'s exactly on spans 3-5 x4, 3-8 x3 and
  12-12 x4; per-pass changes agree to 1.4e-6 relative on 3-5 (under 1e-4 on all); last-position logits are inside the
  6.8e-5 yardstick; 3-token greedy continuations match. The plant (every token at full depth) misses the reference's
  changes by 2.2e-3 on 3-5, 1,500 times the real run's gap.
- **Modes:** at mixed depths, incremental, `--kv-latent`, `--mla-split 2`, streamed trunk, `--stream-lm-head`,
  `--expert-pipeline`, `--trunk-rows`, `--kvb-cache active` and `pin`, `--ultra-low-memory` are bit-identical to full
  recompute; `--spec 3` and `--draft-trunk` are serial decode bit for bit, with drafts proposed and accepted; a state
  saved after `--spec` is the serial state byte for byte and resumes as one uninterrupted run; a state is refused under
  another loop; `--tf-check` counts the torch reference's agreement; the cache trace lists layers in executed order.
- **Default and `--loop`:** no flag and `--no-loop` are `--loop-count 1`, bit for bit; `--loop` is `--loop-layers 6-8
  --loop-count 4 --loop-exit 0.05` on the tiny model, bit for bit; putting the loop back on by default fails 2 tests; the
  offline CLI, kv_b cache, MLA split, quality and expert-profile suites pass.
- **Plants, each reverted:** stopped tokens run in later passes (`K3_LOOP_MUTANT_NO_COMPACT`): caught by the reference;
  every post-span token reading token 0's block count: 12 parity failures; a sweep committing every position to every
  pass: `--spec` and state gates; a state that drops each pass's cached count: state gates; a stale change reference:
  the reference.

Measured on the tiny model (13 layers, blocks of 3), 2026-10-19, Linux x86-64, gcc 13.3:

- **Flag off:** ids and every generated token's logits byte-identical to a `main` build in 9 modes (full recompute,
  incremental, `--kv-latent`, `--mla-split 2`, streamed trunk, streamed trunk with `--stream-lm-head`,
  `--expert-pipeline`, `--spec 3`, `--kv-latent --spec 2`). `make test` gate lines identical to `main`'s.
- **Reference:** spans 3-5 x2, 3-8 x3, 12-12 x3 and 0-2 x2 match the torch reference at 4.3e-6 to 6.8e-6 relative
  (yardstick 6.8e-5); the unlooped run is 0.46 to 0.84 away from the same references; looped greedy ids match.
- **Plants:** running the repeats after the tail fails the reference on all three spans it changes; an AttnRes stride
  that does not grow fails mode parity on 12-12 and 3-8. Both were reverted and the source hash re-checked.
- **Mutant:** shared state is 3.6e-4 to 4.8e-3 from the reference on spans with KDA layers; on the MLA-only span it
  is invisible to full recompute (no MLA state exists there) and caught by incremental parity in both cache layouts.
- **Sanitizers:** the 7 tests pass under ASan + UBSan. LeakSanitizer reports 3,734 bytes in 67 allocations, the same
  as a `main` build with no flag; the leak predates this change and is not fixed here.
- **CMake:** 17 of 18 ctest targets run and pass; the tokenizer target is skipped (no tokenizer file on the box).
