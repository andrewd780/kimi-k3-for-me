# MLA KV-cache variants: what each one costs, and what absorbing `kv_b` changes

**A benchmark and a numerical study. Nothing in the engine's arithmetic changed**; the
only engine change is a recording hook for the test (`k3_mla_trace`, below). Four ways to run
one MLA layer's cached attention, plus one bit-exact restructuring added as a fair
baseline, are implemented side by side in `benchmarks/mla_variants.h`. They are held to
the engine bit for bit by `tests/unit/test_mla_variants.c`, which is part of `make test`,
and measured by `benchmarks/bench_mla.c`. `benchmarks/mla-study.sh` reproduces every
number below, and the raw JSON lines are in
[`docs/measurements/mla-variants-x86_64.jsonl`](../measurements/mla-variants-x86_64.jsonl).

The study has two halves. The **exact half**, the bitwise gates, the kv_b application
counts and the numerics of the absorbed variant, does not depend on machine load. The
**wall times** do: the machine is a shared 4-core VM whose other jobs, in an informal
check, turned a 0.9 ms matmul into 25 ms. They were taken by the timing phase of the
study script with that VM otherwise idle, gated on load and CPU pressure run by run;
the conditions are under [Timing](#timing) and the commands under
[Reproducing](#reproducing).

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

Measured on 2026-09-23 on the 4-core VM the rest of this note uses: Intel Xeon
Processor @ 2.80GHz, 4 cores with one thread each, 33 MB L3, 16 GB RAM. The CPU has
AVX-512; the kernels are the AVX2 ones, as `bench_mla` reports. gcc 13.3.0 with the
Makefile's flags, `-O3 -std=gnu99 -march=native -fopenmp -ffp-contract=off`, at commit
475a7a8. Nothing else ran. Each of the four arms (one and four threads, decode and
prefill) waited for the 1-minute load average to fall under 1.0, and they started at
0.33, 0.63, 0.81 and 0.90. Each run then waited for the machine to be 75% idle (one
thread) or 90% idle (four threads); the least idle any kept run started on was 87% and
95%. The largest PSI CPU-stall share of any kept run was
3.6% on one thread and 7.4% on four. Of 455 kept runs, 3 were repeats of runs discarded
for a stall share over 10%, all on four threads. Process CPU time was 0.94 to 1.00 of wall
time in every one-thread run. The phase took 63 minutes.

Method, as in `benchmarks/bench_mla.c`: the timed region is `mla_attend`, appending the
T new tokens (a kv_b matmul each for the expanded layout, a copy for the latent one)
plus the attention, for **one layer**. Each (C, T) configuration interleaves the
variants run by run. A cell is the **median of 5 runs, in milliseconds per call, with
the fastest run in brackets**. A configuration whose run would take longer than
`--max-run-s` (70 s on one thread, 90 s on four) is PROJECTED from the unit cost of one
kv_b application, scaled by the measured-to-modelled ratio of the latent runs already
completed in the same process. E and E+ at 65,536 positions (6.4 GB of cache for one
layer, over the 3 GB `--mem-mb`) are NOT RUN. The raw lines, with every run's time,
stall share, idle fraction and CPU time, are the `time`, `kv_b` and `model` lines of
[`docs/measurements/mla-variants-x86_64.jsonl`](../measurements/mla-variants-x86_64.jsonl).

### The unit: one kv_b application

A 24,576 x 512 bf16 matmul, 12.58 M multiply-adds, median of 31. Each of the four
bench processes measures it once, at startup. The last column checks the latent-path
model: every measured L0 and L1 time divided by its counted applications times the unit.

| threads | arm | ms per application | GMAC/s | measured / modelled, latent runs |
|---:|---|---:|---:|---|
| 1 | decode | 3.860 | 3.26 | 1.092 over 13 runs (1.02 to 1.14 each) |
| 1 | prefill | 4.315 | 2.92 | 1.907 over 1 run, L1 at T = 256 |
| 4 | decode | 0.929 | 13.54 | 1.037 over 17 runs (0.96 to 1.09 each) |
| 4 | prefill | 1.066 | 11.80 | 1.421 over 2 runs, L0 and L1 at T = 256 |

The same matmul varied by 12% (one thread) and 15% (four) between processes on the idle
machine. Within a row of the tables below the variants are interleaved, so ratios in a
row are tighter than absolute times from row to row.

### Decode, one thread

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 20.5 (18.2) | 13.2 (12.4) | 2,108 (2,041) | 1,013 (967) | 11.5 (11.3) |
| 256 | 5 | 88.9 (86.6) | 43.4 (42.9) | 10,940 (10,322) | 1,135 (1,082) | 42.4 (40.9) |
| 1,024 | 1 | 68.5 (66.6) | 43.6 (42.7) | 8,363 (8,158) | 4,205 (4,160) | 24.0 (23.8) |
| 1,024 | 5 | 348 (339) | 120 (109) | 43,013 (42,323) | 4,471 (4,247) | 105 (104) |
| 4,096 | 1 | 383 (354) | 252 (234) | 34,344 (33,416) | 17,192 (17,011) | 76.6 (72.8) |
| 4,096 | 5 | 1,765 (1,704) | 421 (411) | ~171,900 projected | 18,047 (17,309) | 398 (396) |
| 16,384 | 1 | 1,766 (1,502) | 1,119 (939) | ~137,600 projected | 69,862 (69,150) | 322 (319) |
| 16,384 | 5 | 7,866 (7,257) | 1,962 (1,786) | ~690,700 projected | 72,295 (69,883) | 1,517 (1,470) |
| 65,536 | 1 | not run | not run | ~552,500 projected | ~333,000 projected | 1,134 (1,107) |
| 65,536 | 5 | not run | not run | ~2,763,000 projected | ~333,000 projected | 6,113 (5,929) |

### Decode, four threads

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 16.3 (14.3) | 4.21 (3.88) | 514 (443) | 230 (213) | 3.49 (3.18) |
| 256 | 5 | 71.9 (68.9) | 11.2 (10.7) | 2,507 (2,344) | 261 (227) | 13.1 (12.2) |
| 1,024 | 1 | 64.7 (62.8) | 13.8 (12.5) | 2,025 (1,809) | 989 (833) | 7.22 (6.77) |
| 1,024 | 5 | 328 (322) | 32.3 (28.4) | 9,922 (9,762) | 1,013 (894) | 35.4 (29.1) |
| 4,096 | 1 | 375 (359) | 74.5 (63.4) | 8,212 (7,847) | 3,931 (3,768) | 22.1 (20.7) |
| 4,096 | 5 | 1,844 (1,742) | 137 (108) | 39,599 (38,031) | 4,146 (3,750) | 136 (112) |
| 16,384 | 1 | 1,773 (1,704) | 303 (290) | 30,416 (29,243) | 14,829 (14,618) | 104 (93.0) |
| 16,384 | 5 | 8,695 (8,248) | 534 (484) | ~158,600 projected | 16,065 (15,662) | 397 (378) |
| 65,536 | 1 | not run | not run | ~126,500 projected | 72,631 (72,282) | 351 (327) |
| 65,536 | 5 | not run | not run | ~631,200 projected | 74,053 (73,419) | 1,578 (1,560) |

### Prefill, C = 0 and T = 256

| threads | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,668 (2,439) | 1,565 (1,377) | ~541,400 projected; about 270,000 to 310,000, see below | 2,107 (2,094) | 2,116 (2,069) |
| 4 | 1,892 (1,711) | 375 (323) | 60,360 (59,501) | 541 (531) | 570 (537) |

### The work every variant shares

The projections, the output gate and o_proj, per token per layer (`--common`, timed in
the four-thread arm only): **27.7 ms** on four threads, median of 5, for 219.6 M
multiply-adds (7.9 GMAC/s); 0.665 s per token over the 24 MLA layers. Every variant adds
this to the times above.

### Per token, all 24 MLA layers, four threads

The four-thread tables divided by T and multiplied by 24, in seconds per token, against
the ~16 s per token the rest of the model takes on this VM (`NONATTN_SECONDS`) and the
0.665 s of shared work above.

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (prefill) | 256 | 0.177 | 0.035 | 5.66 | 0.051 | 0.053 |
| 256 | 1 | 0.392 | 0.101 | 12.3 | 5.52 | 0.084 |
| 256 | 5 | 0.345 | 0.054 | 12.0 | 1.25 | 0.063 |
| 1,024 | 1 | 1.55 | 0.330 | 48.6 | 23.7 | 0.173 |
| 1,024 | 5 | 1.57 | 0.155 | 47.6 | 4.86 | 0.170 |
| 4,096 | 1 | 9.00 | 1.79 | 197 | 94.3 | 0.532 |
| 4,096 | 5 | 8.85 | 0.659 | 190 | 19.9 | 0.654 |
| 16,384 | 1 | 42.5 | 7.28 | 730 | 356 | 2.50 |
| 16,384 | 5 | 41.7 | 2.56 | ~761 projected | 77.1 | 1.91 |
| 65,536 | 1 | not run | not run | ~3,035 projected | 1,743 | 8.43 |
| 65,536 | 5 | not run | not run | ~3,030 projected | 355 | 7.57 |

### What the timing says

- **E leaves three cores idle; E+ does not, and is the same floats.** Only E's kv_b
  append is threaded, as in the engine, so from C = 4,096 up its four-thread time is its
  one-thread time (process CPU time equals wall time). E+ splits heads across threads
  and reads each cached row once per call instead of once per query token. It is 1.5 to
  1.6 times faster than E on one thread at T = 1 and 2.1 to 4.2 times at T = 5. On four
  threads it is 3.9 to 5.9 times faster at T = 1 and 6.4 to 16 times at T = 5. At 16,384
  positions and T = 1 that is 7.3 s per token over 24 layers instead of 42.5 s.
- **A is the fastest variant at long contexts, and E+ matches it when verifying five
  tokens.** On four threads at T = 1, A is 1.2 times faster than E+ at 256 positions,
  1.9 times at 1,024, 3.4 at 4,096 and 2.9 at 16,384. At T = 5, E+ is faster at 256 and
  1,024 (11.2 against 13.1 ms, 32.3 against 35.4), equal at 4,096 (137 against 136)
  and 1.3 times slower at 16,384. A does up to 3.4 times E's multiply-adds but reads
  2,304 cached bytes per position instead of 98,560. At 16,384 positions E+ moves 1.6 GB
  per call, 5.3 GB/s in 303 ms. A is not exact (see
  [Numerics](#numerics-of-the-absorbed-variant)), so under this project's contract its
  speed is not on offer beside E.
- **Rebuilding the cache costs what the counts say, and that is far too much for
  decode.** L1 is 1.97 to 2.24 times faster than L0 at T = 1 and 9.6 to 9.8 times at
  T = 5 where both ran; the counts give 2.0 and 9.9 to 10.0. Every latent decode run
  took 0.96 to 1.14 times its applications times the unit cost. But each rebuilt
  position is a 12.6 M multiply-add matmul, and L1 is 49 to 72 times slower than E+ at
  T = 1 on four threads (23 to 31 times at T = 5). At 4,096 positions that is 94 s per
  token over 24 layers, against E+'s 1.8 s and the ~16 s of the rest of the model. The
  engine's `--kv-latent` path, L0, takes 197 s per token there.
- **At prefill, L1 is on par with the expanded cache.** At C = 0 and T = 256, L1 makes
  as many kv_b applications as E. On four threads it takes 541 ms, against E+'s 375,
  A's 570 and E's 1,892. L0 at the same point was measured on four threads at 60.4 s
  for one layer, 112 times L1. That is roughly what `--kv-latent` pays for a 256-token
  prompt today: 24 minutes over 24 layers. L0 / L1 is 112 rather than the counted 257
  because half of L1's time at T = 256 is the attention itself, not kv_b.
- **Beside a fast variant the shared work is not small.** At 27.7 ms per token per
  layer it is more than A's attention at 4,096 positions (22.1 ms) and 0.37 of E+'s
  (74.5 ms).

Three cautions about individual cells:

- **L0's one-thread prefill projection overstates.** The bench calibrates a projection
  on the latent runs already completed in the same process. In the one-thread prefill
  process the only one is L1 at T = 256, whose ratio is 1.91 because half its time is
  attention. L0's own ratio on one thread was 1.06 to 1.09 at the five decode points it
  ran. On four threads its measured prefill was 0.86 to 0.99 times its count times the
  unit. Scaled by its own ratio, the one-thread L0 prefill is about 270 to 310 s (65,792
  applications x 3.86 to 4.32 ms x 1.06 to 1.09), not the 541 s in the JSON. The decode
  projections are calibrated on runs whose ratios are 0.96 to 1.14, and the four-thread
  prefill needed no projection.
- **E and E+ at 65,536 positions are not measured.** The JSON's `seconds_per_call` for
  those lines (status `not_run_memory`) is the bench's linear extrapolation from 16,384.
  At T = 1 it gives E 6.3 s per call on one thread and 7.0 s on four, and E+ 1.57 s and
  0.43 s.
- **One synthetic layer on one VM.** The per-token columns multiply one layer by 24.
  They are not a model measurement.

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
benchmarks/mla-study.sh <out_dir> 4 timing                # quiet machine: 63 min here
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
- Speed anywhere else. The times are one synthetic layer on one 4-core x86-64 VM with
  the AVX2 kernels; the per-token figures multiply that layer by 24, and no arm64 timing
  has run yet.
