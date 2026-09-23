# Testing

Every test in `make test` runs **without model weights**. The checkpoint is 1.56 TB; if
correctness depended on having it, correctness would not get checked.

```bash
make test          # everything below; about 20 s on four cores, peak RSS ~1.7 GB
make test-all SHARD_DIR=~/k3model   # adds the checkpoint-dependent tests
```

## What each gate proves

**`test_kda_exact`** compares complete recurrent-state and output bytes against the
original scalar-order C through 9,504 repeated steps, including zero skips,
misalignment, tails and heap-sized widths. The dedicated KDA workflow compiles
both the original C and opt-in SIMD, compares full oracle-logit traces within each
ISA, runs sanitizer coverage, and records three synthetic timing runs per arm.

**`test_mla_variants`** holds the MLA cache variants of `benchmarks/mla_variants.h` to
the engine's own `k3_mla_cached` by `memcmp`: outputs, accumulators, appended cache
rows, raw scores and every softmax normaliser, at three value-row budgets and on one
thread and every thread, with each variant's kv_b application count checked against its
closed form (including the prefill shape C=0, T=256, where L0 makes 65,792 and L1 256).
Ordinary random layers cannot see a reordered score chain, a double sum rounded to
float, and the test prints an order witness showing 0.0% sensitivity there, so it also
runs *cancelling* layers (exactly negated huge terms built into the weights, where
94-100% of reordered chains round differently) and *sharp* layers (where the double
softmax normaliser stops being exact in any order). Mutants that reorder a chain in E+,
L1, A or E itself pass every ordinary case and fail the cancelling ones; see [the
variants note](notes/mla-variants.md).

**`test_quality`** checks stable NLL arithmetic on synthetic logits. Python quality
tests cover overlapping target windows and token-weighted aggregation. A tiny CLI
primitive check compares native scores with ordinary prefix logits. These tests
do not invoke the corpus harness or measure full-checkpoint model quality.

**The Linux lm_head mechanism check** uses a scaled synthetic checkpoint in two
64 MiB cgroup v2 units with swap disabled. It verifies the actual cap, equal memory
plans, one versus two trunk ring slots, no drops and identical full-vocabulary
logits. It refuses an uncapped fallback. See [the mechanism note](notes/stream-lm-head.md).

**`test_ops`**, every kernel against reference values, at a tolerance declared in the
fixture manifest rather than hardcoded. Several fixtures are adversarial by construction:
the router fixture reorders its top-2 on 5 of 6 rows, so an implementation that ignores
the routing bias fails; the SiTU-GLU fixture drives the activation to its exact
analytic cap.

**`test_cache`**, the streaming expert cache: prefetch, eviction, and mixed batch/serial
access. Uses a synthetic shard of structurally faithful experts, a few KB. With
`K3_EXPERT_PIPELINE=1` it also runs a second pass covering the opt-in reader-thread
pool: pipelined batches, duplicate and already-resident ids, a batch bigger than the
slot count, draining on free/reset_stats before any `get()`, and an injected short
read; see [the pipelining note](notes/expert-pipeline.md).

**`test_trunk`**, the streaming trunk: ring-slot budget enforcement, one-slot
guard, async prefetch, slot-isolation under concurrency, ring wrap-around, and
truncated-read failure isolation. Uses a synthetic 3-layer trunk fixture of a few
KB that is generated inline, so no checkpoint is required. The one-slot guard check
fails against any build that starts the reader thread unconditionally, which is the
condition the real model exhibited as silent token corruption: with one ring slot
the reader would write layer L+1 over layer L while the caller was still computing
on it, producing fluent but wrong tokens with no diagnostic.

**`test_st`**, the safetensors reader: dtype widening, offsets, tail bytes, escaped
tensor names, and a tensor deliberately containing non-finite values.

**`test_model_stream`**, the ultra-low-memory model-table reader. It gathers one BF16
embedding row and projects through the lm_head in bounded aligned chunks, then requires
both results to be bit-identical to the resident kernels. It also injects a corrupt file
offset and requires the resulting short read to fail without being counted as valid I/O.

**`test_cfg`**, the config reader against the fixture layout, plus three malformed configs
in `tests/fixtures/cfg/` it must **refuse**: `no_layermap.json` (no `full_attn_layers` at
all), `bad_layer_index.json` (a one-based index outside `1..n_layers`), and
`bad_topk.json` (a top-k above `K3_MAX_TOPK`). Each is the working config with exactly one
field mutated, so a rejection can only come from that field. This matters more than it
looks: a config reader that substitutes defaults for missing fields produces a model that
loads, runs, and is architecturally wrong, with nothing to indicate it.

**`scale_test`**, the same kernels at the real released dimensions: 7168 wide, 93 layers,
96 heads, 896 experts. It checks the layer map really is 69 KDA + 24 MLA, that every
scratch-sizing helper returns a sane value at full width rather than only at fixture
width, and it allocates and runs one complete 7168-wide KDA layer. That last step is a
single **1.77 GB** allocation, which is the only real resource requirement anywhere in
`make test`; on a machine too small for it the test says so and fails rather than
skipping.

**`test_tok`**, byte-exact roundtrip (encode then decode recovers the input exactly).
With `tools/tok_parity.py` it also compares token-for-token against the reference
tokenizer across CJK, emoji, ZWJ sequences, accents, contractions and whitespace runs.
This is the one gate that cannot run on a clean checkout: `tiktoken.model` has 163,584
entries and ships with the checkpoint, not with this repository. `make test` reports it as
**NOT RUN** rather than passing it quietly.

You do not need the 1.56 TB checkpoint to run it. Four small files are enough, about
2.8 MB in total:

```bash
hf download moonshotai/Kimi-K3 \
    tiktoken.model tokenizer_config.json config.json tokenization_kimi.py \
    --local-dir ~/k3tok

make test TOK_FILES=~/k3tok                 # the roundtrip leg now runs
make tok  TOK_FILES=~/k3tok                 # token-for-token parity, needs `pip install tiktoken`
./bin/test_cfg real ~/k3tok/config.json     # the released nested config
```

`tokenization_kimi.py` is required by `tools/tok_parity.py`, which reads the split regex
out of it rather than restating it; without that file the parity run stops before its
first case. The roundtrip leg in `make test` needs only `tiktoken.model` and
`tokenizer_config.json`.

**`k3_model`**, the end-to-end gate, on a tiny model whose tensor graph matches the released
architecture exactly:

- teacher forcing: every position matches the reference
- full-recompute state reuse: every logit is bit-identical with one recurrent-state slot
- greedy decode: every generated token matches
- incremental decode: same tokens as full recompute, with KV cache and carried
  recurrent state
- KV latent layout: the same incremental decode run again with `--kv-latent`'s
  compressed cache, requiring every logit of every step to be bit-identical to the
  expanded one; see [the latent-cache note](notes/kv-latent.md)

Every one of these must be *exact*. There is no tolerance on token identity, and none
on the logits of the two KV layouts either.

## Checkpoint-dependent tests

Need `SHARD_DIR`:

- **`test_expert`**, reads an expert from released shards and compares against
  independently fetched bytes.
- **`test_real_layer`**, runs one released layer at full width against the reference.
- **`tools/conform_all.py`**, all 93 layers individually against the reference.

## Adding a test

Fixtures are generated by `tools/emit_fixtures.py` and carry their own tolerance in a
manifest. A test that cannot fail is not a test, make the fixture adversarial: if a
plausible wrong implementation would pass, change the input until it would not.
