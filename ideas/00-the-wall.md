# The wall: why no amount of compression reaches 200 GB

Read this before evaluating any proposal. It is the constraint every idea in this
folder has to clear, and it disqualifies a whole category of approach that looks
obvious from the outside.

Reproduce every number here with `python3 ideas/size_model.py 200`.

## What the checkpoint is made of

| store | params | format | size |
|---|---|---|---|
| routed experts | 2,722,740,830,208 | MXFP4, 4.25 bits | 1,446.5 GB |
| — nibble plane | | 4.00 bits | 1,361.4 GB |
| — E8M0 scale plane | | 0.25 bits | 85.1 GB |
| always-active trunk + embed | 56,743,648,000 | bf16 | 113.5 GB |
| **total** | | | **1,559.9 GB** |

One expert is `w1[3072][3584] + w3[3072][3584] + w2[3584][3072]` = 33,030,144 params
= 17,547,264 bytes. There are 82,432 of them (896 experts × 92 MoE layers). That
17.55 MB figure is the cache slot size in `src/cache/k3_cache.c`, so the arithmetic
above is the same arithmetic the engine runs on.

## The arithmetic that kills compression

To reach 200 GB while keeping all 2.72 T expert parameters:

| trunk choice | trunk GB | left for experts | bits/param required |
|---|---|---|---|
| bf16 (today) | 113.5 | 86.5 | **0.254** |
| int8 | 56.7 | 143.3 | **0.421** |
| int4 | 28.4 | 171.6 | **0.504** |
| **trunk deleted entirely** | 0.0 | 200.0 | **0.588** |

The last row is the one that settles it. **Even if the trunk cost nothing, the experts
would still need 0.588 bits per parameter.** The entire bf16-vs-int8-vs-int4 trunk
debate moves the requirement only from 0.25 to 0.59 — the trunk is a rounding error in
this problem.

Below one bit per parameter you cannot assign an independent code to each weight at
all. The code has to be *shared* between weights, and sharing codes between weights is
parameter sharing, which is not compression. **200 GB is a parameter-count problem, not
a compression problem.**

## Where real methods actually land

| bits/param | experts | + int8 trunk | vs target |
|---|---|---|---|
| 4.25 (today) | 1,446 GB | 1,503 GB | 7.5× over |
| 3.75 (measured lossless floor) | 1,277 GB | 1,334 GB | 6.7× over |
| 2.0 (AQLM / QuIP# / QTIP, **needs fine-tuning**) | 681 GB | 737 GB | 3.7× over |
| 1.0 (one bit per weight — model destroyed) | 340 GB | 397 GB | 2.0× over |
| 0.421 (required) | 143 GB | 200 GB | — |

Nothing genuinely one-shot post-hoc goes below ~3 bits with the model intact. Every
method that reaches 2 bits needs calibration *plus* fine-tuning, and at 2.72 T expert
params that is an estimated 40–80 GPU-weeks — to land 3.7× short.

## The rate-distortion check, anchored on this repo's own measurement

`docs/data/trunk-quantisation.txt` measured int8 → ~1% mean relative weight error and
int4 → ~17% on real K3 tensors. That is 2.03× error per bit removed, against the
Gaussian theoretical 2.00×. K3's weights track the 6.02 dB/bit slope almost exactly,
so extrapolating is trustworthy:

| bits | extrapolated K3 error | Shannon-optimal ∞-dim VQ floor |
|---|---|---|
| 4 | 17% (measured) | 6% |
| 2 | 70% | 25% |
| 1 | 142% | 50% |
| **0.421** | **215%** | **75%** |

At the required rate, a vector quantizer *of infinite dimension* — better than any real
method — still yields ~75% RMS weight error. `docs/ROADMAP.md` already rejected 17% as
too much. 75% is not a model, it is noise with a shape.

## Why shared codebooks don't rescue it

Sharing one codebook across the 896 experts in a layer amortizes the codebook to under
0.001 bits/param. But the rate lives in the **indices**, not the codebook — every
d-dimensional subvector needs its own index no matter who owns the table:

| scheme | index cost | codebook if shared over 896 |
|---|---|---|
| d=8, 2^16 centroids | 2.000 b/p | 0.00028 b/p |
| d=16, 2^16 centroids | 1.000 b/p | 0.00057 b/p |

Sharing removes a term that was already under half a bit. It cannot touch a 0.42-bit
target where the index term alone is 1–2 bits. Sharing is the right engineering choice
and an irrelevant one.

## The structural argument, which is the important one

For compression to reach 0.42 bits/param, the experts would have to be roughly 90%
mutually redundant. **If that is true, the right response is to remove the redundancy,
not to entropy-code it** — a model whose 2.72 T expert parameters carry only ~270 B
parameters' worth of information should be stored as ~270 B parameters.

So the redundancy question decides everything, and it decides it in favour of
restructuring rather than compression either way:

- if experts **are** highly redundant → restructure (see `01-shared-base-low-rank.md`)
- if experts **are not** → compression cannot find the redundancy either, and 200 GB
  is unreachable at any quality

That question is currently **unmeasured**. `ideas/probe_expert_redundancy.py` measures
it, on the real checkpoint, for under a gigabyte of download.

## What compression *can* honestly contribute

Not the target, but not nothing, and both of these are worth banking:

1. **int8 trunk: 113.5 GB → 56.7 GB.** Saves 56.8 GB, and the container already exists
   (`tools/int8_trunk.py`, measured 90.9% teacher-forced agreement). Built, not shipped.
2. **Entropy-code the E8M0 scale plane only: 85.1 GB → ~37 GB.** Lossless, and it is the
   *only* entropy coding here that survives this project's decode-speed test. The scale
   plane is 1.03 MB per expert against 17.55 MB of nibbles, so a token decodes 1.52 GB
   of scales rather than 25.83 GB — about 0.9 s/token at the FSE rate this repo already
   measured (1,724 MB/s), under 5% overhead. Compressing the *nibbles* instead would
   cost 83 s/token at the measured scalar-Huffman rate, which is exactly why
   `docs/notes/compressed-trunk.md` shelved that work.

   **The 3.5-bit entropy estimate behind the ~48 GB saving is unverified.**
   `tools/bench_lossless.py:77` filters to `.weight_packed` and has never sampled
   `.weight_scale`. Measuring it is a one-line change and worth doing first.

Together those are ~105 GB off 1,560 GB. Real, cheap, and 7× short on their own.
