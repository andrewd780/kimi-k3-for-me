# MLA KV-cache variants: what each one costs, and what absorbing `kv_b` changes

**A benchmark and a numerical study. Nothing in the engine's arithmetic changed**: the
study added a recording hook for the test (`k3_mla_trace`, below), and the review that
followed it led to a row split of the engine's latent path (commit ea6f419, see the
[correction](#the-decode-shapes-c-cached-positions-t--1-or-5-new-tokens) under Counts)
that computes the same floats with half the kv_b work. Four ways to run
one MLA layer's cached attention, plus one bit-exact restructuring added as a fair
baseline, are implemented side by side in `benchmarks/mla_variants.h`. They are held to
the engine bit for bit by `tests/unit/test_mla_variants.c`, which is part of `make test`,
and measured by `benchmarks/bench_mla.c`. `benchmarks/mla-study.sh` reproduces every
number below except two, and the raw JSON lines are in
[`docs/measurements/mla-variants-x86_64.jsonl`](../measurements/mla-variants-x86_64.jsonl):
the shared-work timing (17.8 ms per layer) is printed by `time_common` but not written
to JSON, and the mutation table's mutants are not committed, so it can be re-derived
only by hand. The JSON predates the row split: its L0 lines count, and time, the loop
that applied all of kv_b in both passes, twice the kv_b work of today's L0.

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
kv_b applications are counted in whole-matrix equivalents, kv_b rows applied over its
24,576: L0's key-rows call and value-rows call are half an application each at this
geometry.

| | cache per position per layer | kv_b applications per call | exact? | what it is |
|---|---:|---|---|---|
| **E** | 98,560 B | T | reference | the engine's default loop, copied statement for statement |
| **E+** | 98,560 B | T | bitwise = E | the same arithmetic threaded over heads, each cached row read once per call instead of once per query token |
| **L0** | 2,304 B | T(C+1) + T(T-1)/2 | bitwise = E | the engine's `--kv-latent` loop: every visible position rebuilt through kv_b twice per query token, its key rows to score it and its value rows to weight its values, one application's worth in all (before ea6f419 each rebuild applied all of kv_b's rows and the count was twice this; see the correction under the decode counts) |
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
formulas. The count of kv_b applications each call makes, in whole-matrix equivalents,
is checked against the closed form in the table above, for every variant and for the
engine itself, whose trace hook counts the kv_b rows it applies (`kvb_rows`).

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
  While every softmax term is at least about 2^-29 of the normaliser itself (a float has
  24 significant bits and a double 53), the double normaliser is a sum of floats that is
  exact in any order; on sharp rows the smallest terms fall below that, and it is not.

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
four threads (57 cases: 29 ordinary, 16 cancelling, 12 sharp). The mutants are not
committed, so the table is a record, not a reproducible artifact; the first row's
13 of 16 holds for the mutation applied to the four-chain E+ path alone, and applied
to both E+ chains it fails all 16 cancelling cases.

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

`bench_mla counts` runs each variant on the fixture geometry with a counter on the kv_b
rows every call applies, turns it into applications (whole-matrix equivalents: rows
applied over kv_b's rows), fails if a count differs from its closed form or is not a
whole number, and quotes the multiply-adds and bytes at K3 geometry. The count depends
only on (C, T, vcap), never on the geometry, which is why counting on the small one is
enough. "x24" is all 24 MLA layers per token. "kv_b read" is whole-matrix passes over
kv_b's 25.2 MB, each of L0's half calls counting half a pass: resident, that is DRAM or
cache traffic; under `--trunk-rows`, where the MLA weights are bound as streamed
(`src/model/k3_bind.c`), it is what the row stream reads from a plain `trunk.bin`, where
each of L0's calls reads only its half. A compressed trunk reads the whole matrix on
each call, twice this for L0, because reading one head's rows at a time would decode
each 1 MiB block once per head it holds (`rows_run` in `src/io/k3_trunk.c`). For A it is
the key half once per call plus the value half once per query token. The non-attention
work of the whole model is about 103 GMAC per token, for scale.

### The prefill shape: no cache, T new tokens

| T | L0 kv_b applications | L1 applications (all value rows held) | L0 / L1 | L0 GMAC per token x24 | L1 GMAC per token x24 | A GMAC per token x24 | L0 kv_b read | L1 kv_b read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1.0x | 0.30 | 0.30 | 0.30 | 0.03 GB | 0.03 GB |
| 16 | 136 | 16 | 8.5x | 2.57 | 0.31 | 0.32 | 3.4 GB | 0.40 GB |
| 64 | 2,080 | 64 | 32.5x | 9.84 | 0.33 | 0.38 | 52 GB | 1.6 GB |
| 256 | 32,896 | 256 | 128.5x | 38.9 | 0.40 | 0.62 | 828 GB | 6.4 GB |

L0's count is T(T+1)/2: **quadratic** in the prompt length. At a 256-token prompt its
attention alone is 38.9 GMAC per token, more than a third of the rest of the model, and
828 GB of kv_b traffic per layer (77.7 GMAC and 1.66 TB before the row split, when the
count was T(T+1)). L1's count is T, the same as appending to the
expanded cache, with 12.6 MB of transient value rows for one layer at a time; holding
no value rows at all it is 2T, still linear (512 at T = 256). The engine's `--kv-latent`
path runs L0's loop today.

### The decode shapes: C cached positions, T = 1 or 5 new tokens

L1's value-row budget here is the benchmark default, 2,560 MB for one layer, which holds
52,083 positions at K3 geometry; past that L1 rebuilds the rest twice.

| C | T | L0 applications | L1 applications | L0 / L1 | L0 GMAC/token x24 | L1 GMAC/token x24 | A GMAC/token x24 | E GMAC/token x24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 257 | 257 | 1.0x | 78 | 78 | 0.95 | 0.49 |
| 256 | 5 | 1,295 | 261 | 5.0x | 78 | 16 | 0.95 | 0.49 |
| 1,024 | 1 | 1,025 | 1,025 | 1.0x | 310 | 310 | 2.87 | 1.06 |
| 1,024 | 5 | 5,135 | 1,029 | 5.0x | 311 | 63 | 2.88 | 1.06 |
| 4,096 | 1 | 4,097 | 4,097 | 1.0x | 1,240 | 1,240 | 10.6 | 3.32 |
| 4,096 | 5 | 20,495 | 4,101 | 5.0x | 1,241 | 251 | 10.6 | 3.32 |
| 16,384 | 1 | 16,385 | 16,385 | 1.0x | 4,960 | 4,960 | 41.4 | 12.4 |
| 16,384 | 5 | 81,935 | 16,389 | 5.0x | 4,961 | 1,002 | 41.4 | 12.4 |
| 65,536 | 1 | 65,537 | 78,991 | 0.83x | 19,840 | 23,903 | 165 | 48.6 |
| 65,536 | 5 | 327,695 | 78,999 | 4.1x | 19,840 | 4,820 | 165 | 48.6 |

**Correction (2026-09-24 review), and the change it led to.** L0's two applications per
position were each a full 24,576-row kv_b, of which the score pass reads only the 128
key rows per head and the value pass only the 128 value rows (`src/core/k3_ops.c`, the
latent branch of `k3_mla_cached`). Applying only W_uk's rows in the first pass and only
W_uv's in the second is bitwise identical, since the matvec is row-independent, and
halves L0's kv_b multiply-adds and, from a plain `trunk.bin` under `--trunk-rows`, the
kv_b bytes it requests at K3 geometry, where a head's run is 128 KiB (under O_DIRECT a
run is rounded to whole 4 KiB pages, which on a small matrix can cancel the saving). The
engine has done so since commit ea6f419 (2026-09-24), through `k3_mmw_rows`, and L0
since 5f5208c; `test_mla_variants` and GATE 3b hold it bitwise, `test_mla_variants` also
holds the engine's own kv_b row count (its trace hook's `kvb_rows`) to L0's closed form,
and the tables above are the counts after it. Before it, L0's counts were twice these,
2T(C+1) + T(T-1) (514 at C = 256, T = 1; 65,792 at the 256-token prefill), which is what
the committed JSON records. Counted in whole applications, L0 makes N per query token at
T = 1, the same as L1 holding every value row, so the "L0 / L1 = 2.0x" that stood in the
decode table before the change was L0's own waste, not an advantage of L1's loop order.
L1's advantage at T > 1, rebuilding each position once per call rather than once per
query token, stands: 5.0x at T = 5. L1's second rebuild of the positions past its
value-row budget still applies the whole matrix where only the value rows are read;
split the same way it would count N at every budget. It is not the engine's loop and was
left as it was.

Three things follow, none of them a timing. At T = 1 a latent cache costs at least one
kv_b application per cached position per layer, whichever loop runs it; L0 and L1 with
every value row held (vcap >= N) both make exactly N now, and no exact reorganisation
can go below N. Verifying T drafted tokens at once (T = 5 here) is where L0's cost
multiplies and L1's does not. And A's arithmetic is at most 3.4 times E's (the
per-position ratio it approaches at long contexts), not the hundreds of times a
rebuilding cache pays, because it never expands the cache: each cached position costs
H(2 kv_lora + qk_rope) = 104,448 multiply-adds per query token against E's H(qk_nope +
qk_rope + v_head) = 30,720, read from 2,304 cached bytes instead of 98,560.

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
| **argmax changed**, of 10,000 softmax rows (the first 10,000 of the 10,080, the `--rows` cap) | 0 | 0 | 0 |
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
Processor @ 2.80GHz, 4 cores with one thread each, 33 MB L3, 16 GB RAM. gcc 13.3.0
with the Makefile's flags, `-O3 -std=gnu99 -march=native -fopenmp -ffp-contract=off`,
at commit cb4ab38, so kv_b runs the shipped exact decode kernel on its AVX-512 path
([decode-kernels.md](decode-kernels.md)). The JSON lines' `isa` field
says AVX2 because `bench_mla` then named a build by `__AVX2__` alone; it now reports
AVX-512 for such a build. These tables replace ones taken at 475a7a8 with the kernel
before that change, when one kv_b application took 3.86 ms on one thread and 0.93 ms
on four; it now takes 2.36 and 0.58.

**L0 in every table of this section is the loop before the row split** (ea6f419 in the
engine, 5f5208c in L0), which applied all of kv_b in both passes: twice the kv_b work of
today's L0, whose counts are the ones under
[Counts](#counts-kv_b-applications-multiply-adds-and-bytes). The tables were not
retaken. An informal check after the change, not a measurement against variance:
`bench_mla time --threads 4 --C 256 --T 1 --variants L0,L1 --runs 5`, three processes
each with the binary from before the split and the one after, interleaved, on this VM.
L0's per-process medians were 338, 381 and 325 ms before and 205, 177 and 210 ms after.
L1's were 181, 194 and 174 ms, then 163, 160 and 147 ms: every one lower, a shift rather
than a spread. L1 is no control for noise here, because ea6f419 also refactored
`k3_matmul_bf16`, the kernel L1's rebuilds run on, so its shift cannot be separated from
noise, and L0's change cannot be attributed to the row split alone by these runs.

Each of the four arms (one and four threads, decode and prefill) waited for the
1-minute load average to fall under 1.0, and they started at 0.94, 0.95, 0.98 and
0.94. Nothing else of this work ran during the phase but light file reads and
edits, and a background git sync of a few seconds every five minutes. Each run then
waited for the machine to be 75% idle (one thread) or 90% idle (four threads); the
least idle any kept run started on was 78% and 92%. The largest PSI CPU-stall share of
any kept run was 1.6% on one thread and 9.7% on four. Of 460 kept runs,
20 were repeats of runs discarded for a stall share over 10%: 3 on one thread
and 17 on four. Process CPU time was 0.95 to 1.00 of wall time in 203 of the
215 kept one-thread runs; in the other 12 it fell as low as 0.78, and 11 of those
were the slowest or second slowest of their five, so the medians move little. The
phase took 52 minutes.

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
| 1 | decode | 2.359 | 5.33 | 1.122 over 13 runs (1.06 to 1.22 each) |
| 1 | prefill | 2.347 | 5.36 | 3.048 over 1 run, L1 at T = 256 |
| 4 | decode | 0.579 | 21.74 | 1.160 over 18 runs (1.09 to 1.36 each) |
| 4 | prefill | 0.614 | 20.51 | 2.217 over 2 runs, L0 and L1 at T = 256 |

The same matmul varied by 0.5% (one thread) and 6% (four) between the processes of
this phase. Within a row of the tables below the variants are interleaved, so ratios in a
row are tighter than absolute times from row to row.

### Decode, one thread

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 20.0 (18.2) | 13.6 (12.1) | 1,280 (1,220) | 671 (618) | 11.1 (11.1) |
| 256 | 5 | 86.9 (83.3) | 36.9 (34.0) | 6,619 (6,386) | 752 (708) | 46.3 (43.6) |
| 1,024 | 1 | 76.6 (70.6) | 45.1 (42.2) | 5,477 (5,233) | 2,813 (2,637) | 24.6 (24.0) |
| 1,024 | 5 | 354 (322) | 108 (103) | 26,270 (25,621) | 2,774 (2,713) | 117 (115) |
| 4,096 | 1 | 356 (348) | 252 (216) | 20,574 (19,755) | 10,361 (9,780) | 78.5 (73.7) |
| 4,096 | 5 | 1,690 (1,596) | 414 (392) | ~108,100 projected | 11,291 (11,029) | 332 (313) |
| 16,384 | 1 | 1,651 (1,602) | 1,140 (1,084) | ~86,370 projected | 43,029 (41,509) | 359 (347) |
| 16,384 | 5 | 8,056 (7,639) | 2,021 (1,949) | ~433,600 projected | 45,405 (43,862) | 1,579 (1,536) |
| 65,536 | 1 | not run | not run | ~346,800 projected | ~209,000 projected | 1,348 (1,264) |
| 65,536 | 5 | not run | not run | ~1,734,000 projected | ~209,000 projected | 6,200 (6,124) |

### Decode, four threads

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 18.1 (15.9) | 3.81 (3.56) | 351 (327) | 193 (173) | 3.36 (3.11) |
| 256 | 5 | 79.0 (71.5) | 12.0 (11.0) | 1,758 (1,708) | 205 (178) | 12.9 (11.4) |
| 1,024 | 1 | 70.0 (64.9) | 12.6 (11.2) | 1,321 (1,222) | 645 (580) | 7.14 (6.61) |
| 1,024 | 5 | 365 (331) | 32.1 (31.2) | 6,778 (6,652) | 672 (658) | 28.0 (26.3) |
| 4,096 | 1 | 384 (377) | 78.2 (75.8) | 5,340 (5,133) | 2,735 (2,581) | 23.2 (20.7) |
| 4,096 | 5 | 1,741 (1,727) | 144 (113) | 26,116 (25,495) | 2,742 (2,600) | 102 (91.1) |
| 16,384 | 1 | 1,642 (1,496) | 313 (295) | 21,215 (20,802) | 10,499 (10,376) | 86.8 (80.8) |
| 16,384 | 5 | 8,884 (8,019) | 605 (452) | ~110,500 projected | 11,784 (11,627) | 427 (386) |
| 65,536 | 1 | not run | not run | 85,778 (83,197) | 49,598 (49,294) | 397 (365) |
| 65,536 | 5 | not run | not run | ~440,000 projected | 54,265 (53,132) | 1,727 (1,576) |

### Prefill, C = 0 and T = 256

| threads | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,343 (2,241) | 1,106 (1,017) | ~470,700 projected; about 163,000 to 176,000, see below | 1,832 (1,742) | 1,975 (1,906) |
| 4 | 1,729 (1,720) | 272 (240) | 43,266 (42,058) | 528 (452) | 553 (544) |

### The work every variant shares

The projections, the output gate and o_proj, per token per layer (`--common`, timed in
the four-thread arm only): **17.8 ms** on four threads, median of 5, for 219.6 M
multiply-adds (12.3 GMAC/s); 0.427 s per token over the 24 MLA layers. Every variant adds
this to the times above.

### Per token, all 24 MLA layers, four threads

The four-thread tables divided by T and multiplied by 24, in seconds per token. For
scale, `bench_mla` assumes 16 s per token for the rest of the model
(`NONATTN_SECONDS`): a round figure, not a measurement, since no full model has run on
the VM. The shared work above adds 0.427 s.

| C | T | E | E+ | L0 | L1 | A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (prefill) | 256 | 0.162 | 0.026 | 4.06 | 0.050 | 0.052 |
| 256 | 1 | 0.435 | 0.091 | 8.42 | 4.64 | 0.081 |
| 256 | 5 | 0.379 | 0.058 | 8.44 | 0.982 | 0.062 |
| 1,024 | 1 | 1.68 | 0.303 | 31.7 | 15.5 | 0.171 |
| 1,024 | 5 | 1.75 | 0.154 | 32.5 | 3.23 | 0.135 |
| 4,096 | 1 | 9.22 | 1.88 | 128 | 65.6 | 0.556 |
| 4,096 | 5 | 8.36 | 0.693 | 125 | 13.2 | 0.489 |
| 16,384 | 1 | 39.4 | 7.50 | 509 | 252 | 2.08 |
| 16,384 | 5 | 42.6 | 2.91 | ~530 projected | 56.6 | 2.05 |
| 65,536 | 1 | not run | not run | 2,059 | 1,190 | 9.53 |
| 65,536 | 5 | not run | not run | ~2,112 projected | 260 | 8.29 |

### What the timing says

- **E leaves three cores idle; E+ does not, and is the same floats.** Only E's kv_b
  append is threaded, as in the engine, so from C = 4,096 up its four-thread time is its
  one-thread time. E+ splits heads across threads and reads each cached row once per
  call instead of once per query token. It is 1.4 to 1.7 times faster than E on one
  thread at T = 1 and 2.3 to 4.1 times at T = 5. On four threads it is 4.8 to 5.5 times
  faster at T = 1 and 6.6 to 15 times at T = 5. At 16,384 positions and T = 1 that is
  7.5 s per token over 24 layers instead of 39.4 s.
- **A is the fastest variant at long contexts.** On four threads at T = 1, A is 1.1
  times faster than E+ at 256 positions, 1.8 times at 1,024, 3.4 at 4,096 and 3.6 at
  16,384. At T = 5, E+ is faster only at 256 (12.0 against 12.9 ms); A is 1.15 times
  faster at 1,024 and 1.4 times at 4,096 and 16,384. A does up to 3.4 times E's
  multiply-adds but reads 2,304 cached bytes per position instead of 98,560. At 16,384
  positions E+ moves 1.6 GB per call, 5.2 GB/s in 313 ms. A is not exact (see
  [Numerics](#numerics-of-the-absorbed-variant)), so under this project's contract its
  speed is not on offer beside E.
- **Rebuilding the cache costs what the counts say, and that is far too much for
  decode.** L1 was 1.8 to 2.05 times faster than the old L0 at T = 1 up to 16,384
  positions and 8.6 to 10.1 times at T = 5 where both ran, when the counts gave 2.0 and
  9.9 to 10.0 (since the row split they give 1.0 and 5.0). At 65,536 positions, where L1
  holds value rows for only 52,083 positions and the counts then gave 1.66, the old L0
  measured 85.8 s per call on four threads against L1's 49.6 s, 1.73 times. The latent
  decode configurations took 1.06 to 1.36 times their applications times the unit cost
  (medians): with the faster kv_b, the attention loops around it are a larger share. But
  each rebuilt position is a 12.6 M multiply-add matmul, and L1 is 34 to 51 times slower
  than E+ at T = 1 on four threads (17 to 21 times at T = 5). At 4,096 positions that is
  66 s per token over 24 layers, against E+'s 1.9 s and the 16 s assumed for the rest of
  the model (`NONATTN_SECONDS`, a round figure, not a measurement). The engine's
  `--kv-latent` path, L0, took 128 s per token there before the row split, which halved
  its kv_b work.
- **At prefill, L1 sits among the expanded variants.** At C = 0 and T = 256, L1 makes
  as many kv_b applications as E. On four threads it takes 528 ms: about twice E+'s
  272, level with A's 553 and under a third of E's 1,729. L0 at the same point was
  measured on four threads at 43.3 s for one layer, 82 times L1. That is roughly what
  `--kv-latent` paid for a 256-token prompt before the row split: 17 minutes over 24
  layers, of which the split removes about half the kv_b work. L0 / L1 was 82 rather
  than the 257 then counted because most of L1's time at T = 256 is the attention
  itself, not kv_b.
- **Beside a fast variant the shared work is not small.** At 17.8 ms per token per
  layer it is three quarters of A's attention at 4,096 positions (23.2 ms) and a
  quarter of E+'s (78.2 ms).

Three cautions about individual cells:

- **L0's one-thread prefill projection overstates.** The bench calibrates a projection
  on the latent runs already completed in the same process. In the one-thread prefill
  process the only one is L1 at T = 256, whose ratio is 3.05 because most of its time
  is attention. L0's own ratio on one thread was 1.06 to 1.13 at the five decode points
  it ran, and on four threads its measured prefill was 1.07 times its count times the
  unit. Scaled by its own ratio, the one-thread L0 prefill is about 163 to 176 s
  (65,792 applications x 2.35 to 2.36 ms x 1.06 to 1.13), not the 471 s in the JSON,
  which keeps that projected value under `status: projected`; read it with this
  caveat. The decode projections are calibrated on runs whose ratios are
  1.06 to 1.36, and the four-thread prefill needed no projection.
- **E and E+ at 65,536 positions are not measured.** The JSON's `seconds_per_call` for
  those lines (status `not_run_memory`) is the bench's linear extrapolation from 16,384.
  At T = 1 it gives E 6.4 s per call on one thread and 7.1 s on four, and E+ 1.62 s and
  0.48 s.
- **One synthetic layer on one VM.** The per-token columns multiply one layer by 24.
  They are not a model measurement.

### Apple Silicon and the hosted runners

The hosted figures in this section come from the job logs and artifacts named below;
GitHub keeps those only for its retention period, and this note is their durable record.

The `MLA cache variants` workflow ran at head `cb4ab38` (run 35845709990) on `macos-14`
(Apple M1 (Virtual), 3 cores, NEON kernels; job 107131235227, artifact
`mla-variants-macOS-ARM64`, ID 10742829385) and on `ubuntu-latest` (AMD EPYC 7763, AVX2
build; job 107131235027). On both, `test_mla_variants` passed 57 of 57 cases (2,080
checks) and every counted kv_b application matched its closed form. The 1,024-position
numerics on Ubuntu repeat the VM's to every printed digit. On the M1 the figures for E,
the scores, the argmax and the top-2 gap repeat too (74.18% of scores bitwise equal, no
argmax change in 10,000 rows, smallest gap 9.06e-6), and A's differ slightly: its
relative L2 against E is 8.06e-7 on both, its worst element against the double reference
6.9e-8 there and 6.7e-8 on x86.

The hosted timing sweeps are short (C up to 4,096 on one thread and 16,384 on every
core, `--max-run-s 20`) and ran on shared runners with no stall measurement, so they
show the shape, not figures to quote. On the M1 one kv_b application took 1.413 ms on
one thread and 0.502 ms on three (8.9 and 25.1 GMAC/s), and the latent runs took 1.05
and 1.11 times their counted applications times that unit. At T = 1 the ordering is
the VM's: at 4,096 positions on three threads E+ took 35.6 ms, against E's 133, A's
22.2 and L1's 2,221 (62 times E+). At T = 5 it differs: E+ took 77.6 ms against A's
109, where on the VM A was the faster.

## Reproducing

```bash
make -j2 bin/test_mla_variants bin/bench_mla
./bin/test_mla_variants                                   # the gates, ~20 s
./bin/bench_mla counts                                    # prefill counts, < 1 s
./bin/bench_mla counts --C 256,1024,4096,16384,65536 --T 1,5
benchmarks/mla-study.sh <out_dir> 4 exact                 # test, counts, numerics: ~10 min
benchmarks/mla-study.sh <out_dir> 4 timing                # quiet machine: 52 min on the VM
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
- Speed anywhere else. The tables are one synthetic layer on one 4-core x86-64 VM with
  the AVX-512 kernels; the per-token figures multiply that layer by 24, and the hosted
  arm64 and Ubuntu sweeps above are short runs on shared machines.
