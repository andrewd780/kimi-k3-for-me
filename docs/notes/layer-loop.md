# Layer looping, `--loop-layers` (experimental; changes the model)

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
- **It is not the default and never changes a run that does not ask for it.** Without `--loop-layers`, every code path is
  byte-identical to before. The existing gates prove that, and a `--loop-count 1` gate proves the flag's own identity path.
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

## Modes

| mode | with `--loop-layers` | why |
|---|---|---|
| full recompute, `--incremental`, `--kv-latent`, `--mla-split` | supported, gated | the per-executed-layer state/cache slots are the whole design |
| `--trunk` ring and pinning, `--stream-lm-head`, `--expert-pipeline`, `--expert-profile`/`--pin-experts` | supported, gated | bind/prefetch/cache take any layer order; the prefetch hint follows the executed order |
| `--score-prompt` | supported | the way to measure what looping does to perplexity on a real checkpoint |
| `--layers N` | supported if the span lies inside the first `N` layers | |
| `--spec`, `--draft-trunk` | **refused** | the rollback log is per physical KDA layer; extending it is future work |
| `--save-state`, `--load-state` | **refused** | a saved state's slot layout would silently differ |
| `--trunk-rows`, `--kvb-cache` | **refused** | the row pipeline is not covered by these gates |
| `--ultra-low-memory` | **refused** | not covered by these gates |
| `--dump-cache-trace`, `--tf-check` | **refused** | the trace would double-count repeated layers; tf-check is not covered |

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
