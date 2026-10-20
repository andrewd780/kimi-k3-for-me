# The MLA latent KV cache, `--kv-latent`

Off by default. It changes the *layout* of the incremental decoder's KV cache and
nothing else: the logits are bitwise identical to the expanded cache, and GATE 3b of
`tests/unit/k3_model.c` compares every logit of every decode step to prove it.

## What was stored before

MLA projects a token into one 576-float vector (512 latent + 64 shared rope), norms the
latent, and expands it through `kv_b` into per-head keys and values. The incremental
decoder cached the **output** of that expansion: `n_heads * (qk_nope + v_head)` floats
per position per layer, plus the one shared rope row.

## What is stored now

With the flag on, the cache holds the **input** to `kv_b` instead: the post-norm
`kv_lora_rank` latent, plus the same shared rope row, which is never projected and so is
never rebuilt. `k` and `v` are rebuilt through `kv_b` at the moment they are used.

| per position, per MLA layer | expanded | latent | ratio |
|---|---:|---:|---:|
| released model (96 heads, 128+128, kv_lora 512, rope 64) | 98,560 B | 2,304 B | 42.8x |
| all 24 MLA layers | 2,365,440 B (2.37 MB) | 55,296 B (0.055 MB) | 42.8x |
| 8,192 positions | 19.38 GB | 0.45 GB | |
| 131,072 positions | 310.04 GB | 7.25 GB | |
| fixture (4 heads, 24+16, kv_lora 32, rope 8) | 672 B | 160 B | 4.2x |
| fixture, all 4 MLA layers | 2,688 B | 640 B | 4.2x |

## The compute that buys it

Every cached position must be rebuilt on every step, and `kv_b` is 24576x512, 12.6M
MACs. The rebuild comes in two halves, because softmax needs every score before any
value may be used: the score pass applies only `kv_b`'s key rows (W_uk, 12,288 rows) and
the value pass only its value rows (W_uv, the other 12,288), each through `k3_mmw_rows`.
Holding the rebuilt block across the two passes would mean holding the expanded cache
again, which is the thing being avoided. That is one `kv_b`'s worth per cached position
per query token, ~302 MMAC per cached position per token across the 24 MLA layers,
against zero for the expanded cache. Until commit ea6f419 (2026-09-24) each pass applied
the whole matrix and used half of it, twice that arithmetic (~604 MMAC); the rows each
pass uses are the same floats either way.

Under `--trunk-rows`, where `kv_b` is streamed, every pass rereads it for every cached
position. From a plain `trunk.bin` a pass requests only its half, one read per head's
128 rows (96 reads of 128 KiB instead of three 8 MiB tiles; O_DIRECT rounds each to
whole 4 KiB pages, a small overhead at this size and none of the saving on a matrix
whose runs are shorter than a page). A compressed `trunk.bin.k3z` decodes a whole 1 MiB
block behind every read, four heads' worth of `kv_b`, so there a pass reads the whole
matrix, as before, and applies only its half.

**No speed claim is made here.** Nothing in this note was timed on the released
checkpoint. The direction is not in question -- rebuilding is strictly more arithmetic
than reading -- but the size of the effect on a real machine depends on bandwidth,
threads and context length, and an untimed number would be a guess.

## Why the output is identical rather than close

The stored latent is exactly the bytes the expanded path fed to `kv_b`, and the rebuild
computes each row it uses with the same kernel code, which is deterministic per output
row: applying the key rows in one call and the value rows in another changes which rows
are computed, never how a row is summed. The attention loops are transposed (position
outer, head inner) so that one rebuild serves all 96 heads, but every reduction keeps
its order: scores are formed position-ascending, the running max is taken
position-ascending, and each output element accumulates position-ascending, per head.
Reordering any of those would be a different number in the last bits, which is why the
gate compares floats and not argmaxes.

## Interop

`--kv-latent` needs `--incremental`; full recompute holds no cache to compress and the
flag is refused rather than ignored. The state file records floats-per-position in its
header, so a latent state saves and resumes normally, and loading a state written in the
other layout is refused by name instead of being read at the wrong stride.

## `--mla-absorb`: the absorbed form, rounding-level and opt-in

The rebuild costs one `kv_b` matmul's worth, 512 x 24,576 multiply-adds, per cached
position per query token per layer. That is what makes the exact latent cache slow at
long context: measured at K3's shapes on a 4-vCPU AVX-512 box, one MLA layer for one
answer token took 0.66 s at 1,024 cached positions and 4.6 s at 8,192 (pilot of
j-research `k3/DIAG-2026-10-20-absorb.md`).

`--kv-latent --mla-absorb` uses the algebra DeepSeek's MLA was designed for. With
`W_uk[h]` and `W_uv[h]` head h's key and value rows of `kv_b`:

    score[h][s] = scale * ((W_uk[h]^T q_nope[h]) . c_s + q_rope[h] . r_s)
    out[h]      = W_uv[h] (sum_s p[h][s] c_s)

so `kv_b` is applied twice per query, not twice per cached position: about 96 x
(576 + 512) multiply-adds per cached position per layer, about 1,000x fewer. At 8,192
positions the same call took 0.057 s.

It is the same function in another floating-point order: the exact paths round k and v
to float per position before scoring and weighting them, the absorbed form never forms
them. Its logits therefore differ from the exact layouts by rounding: 4.7e-7 relative on
the tiny model, 2.2e-6 on one MLA layer's output at K3's shapes and 8,192 positions.
That is why it is a flag and not a replacement. What is held exactly:

- `tests/unit/test_mla_variants.c` check 10: the engine's absorbed form is the
  benchmark's variant A bit for bit (outputs, scores, normalisers, quotients, appended
  rows), which check 6 holds to a plain scalar rendering of the formula. `make test`
  shows the gate failing on an engine with the rope term dropped.
- `tests/test_mla_absorb.py`: within `tools/cmp_logits.py`'s yardstick of the exact
  layout and of the torch reference, the same greedy ids, and bit-identical across
  `--prefill-chunk`, thread counts, the layer loop and a resumed read-in.

It reads `kv_b`'s rows directly, so it needs them resident in fp32 or bf16: it is refused
with `--trunk-rows` and `--draft-trunk`, and with `--mla-split`, whose grouped rebuilds
it replaces. The cache it reads and writes is the latent cache itself, so a state saved
under one mode loads under the other.
