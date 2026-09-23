# MLA KV-cache variants: what each one costs, and what absorbing `kv_b` changes

**A benchmark and a numerical study. Nothing in the engine's arithmetic changed**; the
only engine change is a recording hook for the test (`k3_mla_trace`, below). Four ways to run
one MLA layer's cached attention, plus one bit-exact restructuring added as a fair
baseline, are implemented side by side in `benchmarks/mla_variants.h`. They are held to
the engine bit for bit by `tests/unit/test_mla_variants.c`, which is part of `make test`,
and measured by `benchmarks/bench_mla.c`. `benchmarks/mla-study.sh` reproduces every
number below, and the raw JSON lines are in
[`docs/measurements/mla-variants-x86_64.jsonl`](../measurements/mla-variants-x86_64.jsonl).

This version of the note has the **exact half** of the study: the bitwise gates, the
kv_b application counts and the numerics of the absorbed variant. None of those depend
on machine load. **The wall-time tables are placeholders** (marked TIMING PENDING),
because the only machine available so far was a shared 4-core VM whose other jobs, in
an informal check, turned a 0.9 ms matmul into 25 ms. They are filled by the timing phase of the study script on a
quiet machine; the exact commands are under [Reproducing](#reproducing).

## The variants

One MLA layer at the released geometry: 96 heads, qk_nope 128, qk_rope 64, v_head 128,
kv_lora 512. `kv_b` is 24,576 x 512 (12.6 M multiply-adds, 25.2 MB in bf16). A call
appends T new tokens after C cached ones and attends over N = C + T positions, causally.

| | cache per position per layer | kv_b applications per call | exact? | what it is |
|---|---:|---|---|---|
| **E** | 98,560 B | T | reference | the engine's default loop, copied statement for statement |
| **E+** | 98,560 B | T | bitwise = E | the same arithmetic threaded over heads, each cached row read once per call instead of once per query token |
| **L0** | 2,304 B | 2T(C+1) + T(T-1) | bitwise = E | the engine's `--kv-latent` loop: every visible position rebuilt through kv_b twice per query token |
| **L1** | 2,304 B, plus up to 49,152 B per position transiently, for one layer at a time | 2N - vcap | bitwise = E | each position rebuilt once per call; value rows kept for the first vcap positions, the rest rebuilt a second time |
| **A** | 2,304 B | none (T kv_b's worth of multiply-adds in per-head slices) | **no** | absorbed: W_uk folded into the query, W_uv applied after the latent-weighted sum |

The header comment of `benchmarks/mla_variants.h` carries the exactness argument for E+
and L1 and the exact formulas A evaluates. In one line: L1 performs every floating-point
operation E performs, on the same operands and in the same order, and only changes how
often the deterministic kv_b kernel is called; A computes the same real number as E by a
different association, and so different floats.

## How the exact variants are held exact

`test_mla_variants` runs each case through the **engine's own** `k3_mla_cached` in both
cache layouts and through every variant, from the same hidden states, and compares with
`memcmp`: layer outputs, pre-gate attention accumulators, the rows appended to the cache,
the raw scores before the softmax, the double softmax normaliser z of every row and the
double probability quotient e/z of every score. It compares them between the engine's
two layouts, between E and the expanded engine, between L0 and the latent engine, and
between each variant and E. L1 runs at three value-row budgets (all, half, none); E+, L1
and A rerun on every thread (on the cases of up to 48 positions, which span three of
their 16-position blocks); A is compared bitwise with a plain scalar rendering of its
formulas. The count of kv_b applications each call makes is checked against the closed
form in the table above.

The engine's intermediates come from a recording hook, `k3_mla_trace` in `k3.h`, which
is NULL outside tests. It is needed because the output cannot show them: z and e/z are
doubles that reach the output only as p = (float)(e / z), and summing z in another order,
or forming the quotient as e * (1 / z), moves p by a float ulp about once in 2^29 values.
A gate on outputs alone would pass an engine whose two layouts summed z differently,
which is a different reduction tree and so, somewhere in a long enough run, a different
logit. With the hook set, `k3_mla_cached` copies each value out as it forms it (the
quotient is named before it is rounded, so the recorded double is the one rounded);
nothing reads the copies back, and with the hook NULL the cost is a pointer test per
row and per probability. A replacement for the engine's latent loop, such as L1, is
held by this gate to the expanded layout's normaliser and quotient as well as its
output.

Ordinary random layers are not enough for a gate like this, and the test measures why.
Every score is a double chain rounded to float once, and on terms of one size a
reordered chain rounds to the same float: an **order witness** re-sums every score of
every case in three other orders and counts how often the float changes. So the test
also runs two constructed kinds of layer through the same gates:

- **Cancelling layers** build the cancellation into the weights, so the engine runs on
  them too: kv_a has exactly negated row pairs sharing a huge rmsnorm weight (2^30 to
  2^44, and 2^44 to 2^47 in the rope slot), and q_b and kv_b are selection matrices. Every
  score chain then meets huge terms that cancel exactly, and its ordinary terms lose bits
  that depend on the partial sums they meet.
- **Sharp layers** scale the query norm by 16 (exact), for score ranges of tens of nats.
  Until softmax terms fall below about 2^-29 the double normaliser is a sum of floats that
  is exact in any order; on sharp rows it is not.

| order witness (this commit) | score chains | changed if reversed | nope and rope summed apart | two lanes | normalisers of 3+ terms changed if reversed | quotients changed as e*(1/z) |
|---|---:|---:|---:|---:|---:|---:|
| ordinary layers | 305,712 | 0.0% | 0.0% | 0.0% | 0.0% of 3,397 | 25.9% |
| cancelling layers | 25,459 | 100.0% | 100.0% | 93.7% | 0.0% of 1,024 | 25.4% |
| sharp layers | 34,384 | 0.0% | 0.0% | 0.0% | 51.7% of 944 | 21.0% |

The quotient needs no special layer: e * (1 / z) rounds twice where e / z rounds once, so
the doubles differ often, on every kind of layer, while the float p they round to almost
never does.

The witness only says whether a reordering *could* be seen. A mutation run shows that
the test does see one: each mutant below was compiled into a scratch copy of the header,
or for the engine rows of `src/core/k3_ops.c`, and run through the unchanged test on
four threads (57 cases: 29 ordinary, 16 cancelling, 12 sharp).

| mutant | ordinary cases failed | cancelling | sharp |
|---|---:|---:|---:|
| E+ score chain, rope terms before nope terms | 0 / 29 | 13 / 16 | 0 / 12 |
| E+ score chain, nope and rope summed apart | 0 / 29 | 13 / 16 | 0 / 12 |
| L1 score chain, two lanes over the nope terms | 0 / 29 | 13 / 16 | 0 / 12 |
| A score step, four terms regrouped pairwise | 0 / 29 | 10 / 16 | 0 / 12 |
| E itself, rope terms before nope terms (E drifts from the engine) | 0 / 29 | 16 / 16 | 0 / 12 |
| engine, latent layout: score chain with rope terms before nope terms | 0 / 29 | 16 / 16 | 0 / 12 |
| softmax normaliser summed in reverse (E+, L1 and A) | 0 / 29 | 0 / 16 | 11 / 12 |
| engine, latent layout: normaliser summed in reverse | 0 / 29 | 0 / 16 | 11 / 12 |
| engine, both layouts: normaliser summed in reverse | 0 / 29 | 0 / 16 | 11 / 12 |
| A score chain, rope terms before latent terms | 26 / 29 | 14 / 16 | 12 / 12 |
| L1 value sums, positions reversed within a block | 23 / 29 | 14 / 16 | 12 / 12 |
| A value sums, positions reversed within a block | 23 / 29 | 14 / 16 | 12 / 12 |
| E+ lane swap, positions 1 and 2 of a group exchanged | 21 / 29 | 13 / 16 | 12 / 12 |
| quotient formed as e * (1 / z) (E+, L1 and A) | 24 / 29 | 14 / 16 | 12 / 12 |
| engine, latent layout: quotient formed as e * (1 / z) | 24 / 29 | 14 / 16 | 12 / 12 |
| engine, both layouts: quotient formed as e * (1 / z) | 24 / 29 | 14 / 16 | 12 / 12 |

The first six would pass a test built on ordinary layers alone, and the three reversed
normalisers would pass without the sharp layers. Five rows also need the recorded
doubles: the two engine normaliser rows and the three quotient rows. The test before
the trace hook compared the engine only on outputs and appended rows and recorded no
quotient anywhere, and all five passed it, 57 of 57 cases: a z or e/z formed another
way moves no output float on these inputs. In the both-layouts rows the engine's two
layouts still agree with each other; the change is caught because E, the copy of the
engine's original order, no longer matches either. Every mutant fails the test as
committed.

## Counts: kv_b applications, multiply-adds and bytes

`bench_mla counts` runs each variant on the fixture geometry with a counter on every
kv_b application, fails if a count differs from its closed form, and quotes the
multiply-adds and bytes at K3 geometry. The count depends only on (C, T, vcap), never on
the geometry, which is why counting on the small one is enough. "x24" is all 24 MLA
layers per token. "kv_b read" is whole-matrix passes over kv_b's 25.2 MB: resident, that
is DRAM or cache traffic; under `--trunk-rows`, where the MLA weights are bound as
streamed (`src/model/k3_bind.c`), each pass is a read through the row stream. For A it is
the key half once per call plus the value half once per query token. The non-attention
work of the whole model is about 103 GMAC per token, for scale.

### The prefill shape: no cache, T new tokens

| T | L0 kv_b applications | L1 applications (all value rows held) | L0 / L1 | L0 GMAC per token x24 | L1 GMAC per token x24 | A GMAC per token x24 | L0 kv_b read | L1 kv_b read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1 | 2.0x | 0.61 | 0.30 | 0.30 | 0.05 GB | 0.03 GB |
| 16 | 272 | 16 | 17.0x | 5.14 | 0.31 | 0.32 | 6.8 GB | 0.40 GB |
| 64 | 4,160 | 64 | 65.0x | 19.7 | 0.33 | 0.38 | 105 GB | 1.6 GB |
| 256 | 65,792 | 256 | 257.0x | 77.7 | 0.40 | 0.62 | 1,656 GB | 6.4 GB |

L0's count is T(T+1): **quadratic** in the prompt length. At a 256-token prompt its
attention alone is 77.7 GMAC per token, three quarters of the rest of the model, and a
1.66 TB pass over kv_b per layer. L1's count is T, the same as appending to the
expanded cache, with 12.6 MB of transient value rows for one layer at a time; holding
no value rows at all it is 2T, still linear (512 at T = 256). The engine's `--kv-latent`
path runs L0's loop today.

### The decode shapes: C cached positions, T = 1 or 5 new tokens

L1's value-row budget here is the benchmark default, 2,560 MB for one layer, which holds
52,083 positions at K3 geometry; past that L1 rebuilds the rest twice.

| C | T | L0 applications | L1 applications | L0 / L1 | L0 GMAC/token x24 | L1 GMAC/token x24 | A GMAC/token x24 | E GMAC/token x24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 514 | 257 | 2.0x | 155 | 78 | 0.95 | 0.49 |
| 256 | 5 | 2,590 | 261 | 9.9x | 157 | 16 | 0.95 | 0.49 |
| 1,024 | 1 | 2,050 | 1,025 | 2.0x | 620 | 310 | 2.87 | 1.06 |
| 1,024 | 5 | 10,270 | 1,029 | 10.0x | 621 | 63 | 2.88 | 1.06 |
| 4,096 | 1 | 8,194 | 4,097 | 2.0x | 2,478 | 1,240 | 10.6 | 3.32 |
| 4,096 | 5 | 40,990 | 4,101 | 10.0x | 2,479 | 251 | 10.6 | 3.32 |
| 16,384 | 1 | 32,770 | 16,385 | 2.0x | 9,908 | 4,960 | 41.4 | 12.4 |
| 16,384 | 5 | 163,870 | 16,389 | 10.0x | 9,909 | 1,002 | 41.4 | 12.4 |
| 65,536 | 1 | 131,074 | 78,991 | 1.7x | 39,631 | 23,903 | 165 | 48.6 |
| 65,536 | 5 | 655,390 | 78,999 | 8.3x | 39,633 | 4,820 | 165 | 48.6 |

Three things follow, none of them a timing. At T = 1 a latent cache costs at least one
kv_b application per cached position per layer, whichever loop runs it; L1 halves L0's
count and no exact reorganisation can go below N. Verifying T drafted tokens at once
(T = 5 here) is where L0's cost multiplies and L1's does not. And A's arithmetic is at
most 3.4 times E's (the per-position ratio it approaches at long contexts), not the
hundreds of times a rebuilding cache pays, because it never expands the cache: each
cached position costs H(2 kv_lora + qk_rope) = 104,448 multiply-adds per query token
against E's H(qk_nope + qk_rope + v_head) = 30,720, read from 2,304 cached bytes instead
of 98,560.

## Numerics of the absorbed variant

A is the only way to keep a 2,304-byte cache without rebuilding it, and it is not
bitwise equal to E. `bench_mla numerics` measures how far it moves, on one layer at the
full 7,168 hidden width: weights ~ N(0, 0.02) rounded to bf16, rmsnorm weights
1 + N(0, 0.1), hidden states ~ N(0, 1), all through the engine's own projections and
norms; C cached positions, then 105 query tokens x 96 heads, each compared between A, E
and a reference computed in double throughout (k and v never rounded to float, exact
scale, `exp` in double). The stress run scales the queries by 6 for a sharper softmax
than these weights give (score rms 4.9, largest score 32).

| | C = 1,024 | C = 16,384 | stress: C = 4,096, queries x6 |
|---|---:|---:|---:|
| **attention output, A vs E**: relative L2 | 8.1e-7 | 2.8e-6 | 1.0e-6 |
| max abs difference / max abs value | 1.3e-6 | 3.9e-6 | 1.7e-6 |
| elements bitwise equal | 3.3% | 0.9% | 2.8% |
| **E vs double reference**: relative L2 | 5.8e-7 | 2.0e-6 | 7.9e-7 |
| max abs error | 1.6e-7 | 2.1e-7 | 3.7e-6 |
| **A vs double reference**: relative L2 | 5.7e-7 | 2.0e-6 | 7.9e-7 |
| max abs error | 6.7e-8 | 9.6e-8 | 1.6e-6 |
| **layer output after gate and o_proj, A vs E**: relative L2 | 8.1e-7 | 2.8e-6 | 1.0e-6 |
| **raw scores** compared | 10.3 M | 165 M | 41.3 M |
| scores bitwise equal | 74.2% | 74.1% | 74.1% |
| \|score_A - score_E\|, 99th percentile | 1.2e-7 | 1.2e-7 | 5.0e-7 |
| \|score_A - score_E\|, max (score rms) | 4.8e-7 (0.83) | 4.8e-7 (0.82) | 3.8e-6 (4.95) |
| **argmax changed**, of 10,000 softmax rows | 0 | 0 | 0 |
| smallest top-2 score gap among those rows | 9.1e-6 | 1.9e-5 | 2.6e-4 |
| rows whose gap is within 2x their largest score difference | 0 | 0 | 0 |

What the table says:

- **A is not less accurate than E.** Against the double reference their relative L2
  errors agree to within 1% at every size, and A's worst element is 2.1 to 2.3 times
  closer than E's. A moves the output by about the size of E's own rounding error.
- The difference grows with the context, 8e-7 at 1K to 2.8e-6 at 16K, and so does E's own
  error: both are dominated by the float value sums over positions, which E and A both
  accumulate in float, over 128-wide value rows in E and 512-wide latent rows in A.
- Three quarters of the scores are the same float in both; the rest differ by one or a
  few float ulps. No softmax row came near an argmax change: the smallest top-2 gap was
  19 times the largest score difference seen anywhere at 1K, and no row's gap was within
  twice its own largest difference.
- The deviation after the gate and o_proj is the same size as before them: o_proj
  neither amplifies nor averages it away.

**What that means under this project's rules.** The exactness contract asks for
byte-identical logits at every memory budget and mode, so A cannot be an option beside E:
turning it on would change the logits. It could only be adopted as the engine's single
arithmetic in every mode at once, which the numbers above neither justify nor rule out;
they are one synthetic layer, and how a 1e-6 relative change per MLA layer compounds
through 93 layers of real weights is not measured here. L1 needs no such decision: it is
the same floats as the engine's latent path with a linear rather than quadratic kv_b
count, and it is the variant a follow-up would put into `k3_mla_cached`.

## Timing

**TIMING PENDING.** Every table in this section is filled by the timing phase of
`benchmarks/mla-study.sh` on a quiet machine. Until then no speed is claimed for any
variant. The timed region is `mla_attend`: appending the T new tokens (a kv_b matmul each
for the expanded layout, a copy for the latent one) plus the attention. Each (C, T)
configuration interleaves the variants run by run, reports the median of 5 runs, records
the Linux PSI CPU-stall share and the process CPU time of every run, repeats a short run
whose stall share exceeds 10% (at most three times), and waits up to 120 s for the
machine to go idle before each run. A run longer than `--max-run-s` is PROJECTED from the
measured cost of one kv_b application scaled by the measured-to-modelled ratio of the
runs that did complete, and labelled so; E and E+ at 65,536 positions (6.4 GB of cache
for one layer) are NOT RUN.

### The unit: one kv_b application

| threads | ms per application | GMAC/s |
|---:|---:|---:|
| 1 | TIMING PENDING | TIMING PENDING |
| 4 | TIMING PENDING | TIMING PENDING |

### Decode, one thread

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 .. 65,536 | 1, 5 | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING |

### Decode, four threads

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 .. 65,536 | 1, 5 | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING |

### Prefill, C = 0 and T = 256

| threads | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|
| 1 | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING |
| 4 | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING | TIMING PENDING |

### The work every variant shares

The projections, the output gate and o_proj, per token per layer (`--common`):
TIMING PENDING.

### Apple Silicon

The `MLA cache variants` workflow runs the test, the counts, the 1,024-position numerics
and a short timing sweep on `macos-14` (arm64, NEON kernels) as well as Ubuntu. It has
not run yet; its JSON artifact is the arm64 evidence when it does.

## Reproducing

```bash
make -j2 bin/test_mla_variants bin/bench_mla
./bin/test_mla_variants                                   # the gates, ~20 s
./bin/bench_mla counts                                    # prefill counts, < 1 s
./bin/bench_mla counts --C 256,1024,4096,16384,65536 --T 1,5
benchmarks/mla-study.sh <out_dir> 4 exact                 # test, counts, numerics: ~10 min
benchmarks/mla-study.sh <out_dir> 4 timing                # quiet machine: ~1.5 h
```

The numerics are deterministic: the JSON committed here was produced on two threads,
and an earlier run of the same code on four threads gave identical deviation figures.
The `seconds_per_query` fields in the numerics lines are incidental wall times from a
shared machine, kept only because the JSON format has them; they are not results.

## What this does not show

- Real weights. The layer is synthetic; a real head's scores may be sharper than even
  the stress run, and its value rows differently scaled.
- More than one layer. How A's per-layer difference compounds through the model's 24
  MLA and 69 KDA layers into the logits is not measured.
- Any speed. See [Timing](#timing).
