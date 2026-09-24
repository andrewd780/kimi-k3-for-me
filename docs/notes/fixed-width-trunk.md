# Fixed-width trunk dictionary: falsifier first

2026-09-20, updated 2026-09-23. Predictive prefetch remains closed. This separate
research item targets the BF16 trunk, 80.8% of the stated one-position streamed-byte
baseline. Everything that reads checkpoint bytes (gate 1, the per-family samples) and
the hosted exactness, rate and contention legs ran in hosted CI; the FD3B orientation
rates and the decode-under-contention tables were measured on a 4-vCPU cloud VM, with
the conditions beside each. No checkpoint host or rental is used.

## Prior art (added 2026-09-24, after review)

The design here is not new; the review of #12 to #14 found these published or public
precedents, checked as stated. **ZipServ** (Fan et al., ASPLOS'26; arXiv 2603.17435;
code `HPMLL/ZipServ_ASPLOS26@6f8a209`, dated 2025-12-19) encodes BF16 weights with
3 bits per weight from three bitmaps: codes 1 to 7 index a numerically contiguous
window of seven exponent values seeded by the seven most frequent, and code 0 stores
the whole 16-bit value in a separate stream, with the decode fused into the GEMM on
GPU (read from its source, `kernel_benchmark/utils.h` and `csrc/L_Kernel.cuh`; its
abstract, seen through search only, claims up to 30% size reduction, bit-exact).
**dgpp** (`leloch/dgpp@73ddb9f7`, 2026-09-19 09:37 UTC) streams BF16 decode weights
from "the sign+mantissa byte plus a 4-bit exponent code against a per-row window,
exact side tables for the weights outside it", 0.75 of the bytes; **NWC** layout 2
(`parda21/NWC@648b35f`, 2026-09-19 21:04 UTC) stores a `[sign | rank]` nibble per
weight with per-block exception lists. Both were published the day before this note's
first commit (`c03cf52`, 2026-09-20 07:35 UTC). The entropy facts behind all of them,
that BF16 exponents carry about 2.6 to 2.8 bits and the mantissa byte is close to
incompressible, are published in ZipNN (arXiv 2411.05239, exponent bytes separated and
Huffman-coded, about 33% saved) and DFloat11 (arXiv 2504.11651, Huffman-coded
exponents, about 70% size). What differs here: the code indexes the high byte (sign
plus seven exponent bits) with a frequency-ranked, non-contiguous table, an escape
costs 8 extra bits rather than ZipServ's 16, the decode runs on CPU with SSSE3, AVX2
and NEON table lookups rather than fused into a GPU GEMM, and the sign-split
alternative is measured (0.99 points worse at 3 bits). None of that is a new
technique; the measurements on K3's families are the contribution, within the
limits stated throughout.

## Gate 1 contract

`tools/trunk_dictionary_gate.py` reads exactly the eight dense 1 MiB ranges in
`docs/measurements/lossless-samples.json`, at its immutable checkpoint revision.
The existing strict HTTP range reader rejects full-file responses. Length and
SHA-256 must match before a range enters the analysis. No other weight ranges
are requested and no weight bytes are committed or uploaded as artifacts.

The "exponent plane" here is the existing Huffman prototype's **high byte**:
`raw[1::2]`, sign plus the top seven exponent bits. The unchanged low byte holds
the remaining exponent bit and seven fraction bits. Extracting the mathematical
8-bit exponent would test a different bit layout and is not substituted here.

Report all 256 byte counts per range and pooled; minimum support at 90%, 99%,
99.9%, and 99.99%; local best-15 coverage; and coverage using one pooled top-15
dictionary, with ascending-byte tie breaks. The pooled dictionary is fitted on
these same ranges: passing would not prove generalization to unseen tensors.

Operationalize "about 99%" as **at least 99% in every range and pooled**, using
integer counts rather than rounded percentages. A failure stops the codec work.
The CI job can succeed at measuring a scientific STOP; the JSON gate status is
authoritative. `--require-pass` returns 2 on STOP for a later dependent gate.
Adversarial CI tests include diffuse support, individually compressible but
incompatible local dictionaries, a bad range hidden by pooling, threshold edges,
wrong plane/sign handling, missing ranges, truncation and hash corruption.

Gate 1 **passed** in [CI run 35497304209](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35497304209)
at head `c03cf52e328eebb064d7aaab76775fbfc8c667a1`. All nine adversarial tests
passed. The job's complete report (all 256 counts per range and pooled, the support,
local-15 and global-15 columns, and every sample's identity) is committed as
[trunk-dictionary-gate1.json](../measurements/trunk-dictionary-gate1.json), recorded
unchanged from the stdout `FALSIFIER_REPORT` line of job 106042549106; a unit test
recomputes every derived field from its counts. The run's artifact (ID 10601201984)
also holds the pooled whole-BF16 histogram, which the log omits, and expires on
2026-12-19. The pooled global-dictionary coverage is **99.954653%**.

| Range | Distinct | 90% | 99% | 99.9% | 99.99% | Local 15 % | Global 15 % | Payload r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Layer 1 | 25 | 6 | 9 | 13 | 18 | 99.968910 | 99.951553 | 0.750242233 |
| Layer 12 | 25 | 6 | 9 | 13 | 18 | 99.970436 | 99.941254 | 0.750293732 |
| Layer 24 | 22 | 6 | 9 | 12 | 17 | 99.977493 | 99.947929 | 0.750260353 |
| Layer 36 | 23 | 6 | 9 | 13 | 18 | 99.965096 | 99.953842 | 0.750230789 |
| Layer 48 | 23 | 6 | 9 | 14 | 18 | 99.953079 | 99.952698 | 0.750236511 |
| Layer 60 | 25 | 6 | 9 | 14 | 17 | 99.961472 | 99.960518 | 0.750197411 |
| Layer 72 | 22 | 5 | 10 | 14 | 17 | 99.970055 | 99.970055 | 0.750149727 |
| Layer 92 | 24 | 6 | 11 | 14 | 18 | 99.959373 | 99.959373 | 0.750203133 |
| Pooled | 29 | 6 | 9 | 14 | 18 | 99.954653 | 99.954653 | 0.750226736 |

The pooled high-byte count is 4,194,304 with **1,902 escapes** (0.0453472%).
The shared table, in code order, is
`[188,60,61,189,59,187,58,186,185,57,190,62,56,184,183]`.
All ranges are `self_attn.f_a_proj.weight` except layer 92's `self_attn.g_proj.weight`;
this is not coverage of every trunk tensor family. The historical "roughly a
dozen at 99.9%" becomes 14 values pooled, and on these eight ranges 15 entries clear 99%
with room to spare. Across every family they do not: the
[per-family run](#per-family-result) finds the pooled table under 99% on the router and
the shared-expert down projection, and this gate-1 table under 99% on all three
shared-expert projections.

The 4-bit FD4B prototype now implements scalar reference, SSSE3 `pshufb` and
AArch64 NEON `tbl` decoders. It is benchmark-only. CI first repeats the histogram,
then runs ASan/UBSan correctness on both ISAs, then enables rate jobs. Real-range
round trips, independent golden bytes, all byte values, escape underflow/surplus,
odd tails, corrupt payload/reference and deliberate sanitizer faults guard the
comparison. Timing includes low-plane assembly and escape patching.

The benchmark reports only public model identifiers, byte counts, dictionaries,
hashes, timings and hosted-runner metadata. Artifact paths are explicit; raw
model ranges and unrelated workspace files are not uploaded. No user secrets
or personal files are inputs. Standard GitHub logs retain the public repository
identity. Gates 2 and 3 passed in hosted CI at head `9f0c07f`, recorded below.
Gate 4 (the bit-width curve) and gate 5 (row-boundary cost) have results from
committed counts and exact shapes. The every-family samples, FD3B's hosted exactness
and rates, and the hosted legs of decode under matmul contention ran in CI at head
`cb4ab38` (run 35845709912), and the contention benchmark was re-measured on a 4-vCPU
VM with the shipped kernels. The per-family gate is a STOP that names two
families; a supported reader remains ([status](#remaining-gates)).

## Gates 2 and 3: passed in CI run 35498176696

Head `9f0c07f5f98688ffab51f23cf17051bf74e2e324`, hosted `ubuntu-latest` (x86_64,
gcc 13.3, `-O3 -mssse3`, 4 cores) and `macos-14` (arm64, Apple clang 15, `-O3`,
3 cores). The reports are the run's artifacts `dictionary-rate-ubuntu-latest`
(ID 10601747318) and `dictionary-rate-macos-14` (ID 10600672892); the copies in
[fixed-dictionary-rate-x86_64.json](../measurements/fixed-dictionary-rate-x86_64.json)
and [fixed-dictionary-rate-arm64.json](../measurements/fixed-dictionary-rate-arm64.json)
were recorded from the jobs' stdout `CODEC_REPORT` lines, with every derived
field (per-arm mean, median, minimum, both ratios, escape counts, gate status)
recomputed from the primitives and matched exactly before writing.

**Gate 2, exactness** (`-O1 -fsanitize=address,undefined`, both ISAs): the
independent golden layout, the odd-tail case, all eight ranges and the pooled
8 MiB decode byte-exact through both the scalar and the SIMD path; the five
negative controls (truncated header, truncated payload, excess payload, corrupt
low byte, wrong reference byte) are each rejected with exit 1; the deliberate
ASan and UBSan faults trip their sanitizers. Artifacts
`dictionary-exactness-{ubuntu-latest,macos-14}` (IDs 10600847541, 10601433078).

**Gate 3, rate**, reconstructed BF16 GB/s, three runs per arm, every escape
patched and both planes assembled inside the timed region, `bench_huf4` units:

| Input | x86_64 `ssse3_pshufb` min / median / max | arm64 `neon_tbl` min / median / max | Scalar reference |
|---|---:|---:|---:|
| Eight 1 MiB ranges, 24 runs | 24.274 / 25.302 / 25.705 | 16.361 / 19.764 / 26.662 | 1.20–2.11 |
| Pooled 8 MiB, 3 runs | 16.220 / 16.263 / 16.332 | 14.844 / 17.473 / 19.071 | 1.54–2.38 |

`rate_gate.status = PASS` on both ISAs: all 27 SIMD runs per ISA are at or above
3 GB/s and all are at or above the 4 GB/s target. The slowest run anywhere is
14.844 GB/s (arm64, pooled). Against the compact Huffman decoder's best
real-range run on the same ISA (1.100 x86_64, 1.350 arm64, on the four `f_a_proj`
ranges; [results](research-results.md#2-compact-huffman-decoder-research-prototype)),
each ISA's slowest FD4B run is 14.7x faster on x86_64 (16.220) and 11.0x on arm64
(14.844); the pooled medians give 14.8x and 12.9x. Those ratios compare different CI
runs on different hosted runners and buffer sizes (FD4B from run 35498176696 on the
pooled 8 MiB buffer, the Huffman kernel from run 35462000448 on repeated 4 MiB
buffers), so they are indicative, not a matched measurement: the same pooled FD4B
x86_64 case read 16.2 GB/s in that run and 25.4 GB/s in run 35845709912
([below](#fd3b-the-3-bit-prototype)). The ratio cost of that rate, on the
same four ranges, is 7.80 points (0.750257 against 0.672227, [below](#remaining-gates)).
The arm64 spread on identical input (16.4–26.7 GB/s) is the shared three-core hosted
runner; the gate is on the minimum, not the median. At `B = 3 GB/s` and
`r = .75` the serial break-even is `D > 12 GB/s` and the saturated-stream bar is
`D > 4 GB/s`; the pooled minimum clears both.

These are cache-friendly, single-thread, warm-buffer kernel ceilings with no
competing model compute, exactly as the report's `scope` field states. The 8 MiB
pooled case is the less cache-resident of the two and is the figure to quote.

## Remaining gates

The gates listed here on 2026-09-20, and the two the family and contention
questions added, stand as follows on 2026-09-23. Each has a section below with
its provenance.

| Gate | Status | What is known | What is pending |
|---|---|---|---|
| 4. Bit-width curve | **Result on all eight gate-1 ranges and on every family** | Eight ranges, from committed counts: 3 bits retain 0.703403, 4 bits 0.750227, 5 bits 0.8125; 3 bits gain 4.68 points. Every family, byte-weighted (CI run 35845709912): 0.713279 against 0.751122, +3.78 points. Both clear the 1.5-point bar, so FD3B is carried. FD3B is byte-exact in hosted CI under ASan/UBSan (SSSE3, AVX2, NEON) and decodes at 9.6 (SSSE3), 15.3 (AVX2) and 11.0 to 13.5 (NEON) reconstructed GB/s pooled | Nothing for the curve |
| 5. Row-boundary cost | **Exact, from released shapes** | FDRX: zero padding at every K3 width; 0.0340 points per-row index, 0.0148 grouped; three range reads per chunk | Nothing for the cost; a container still needs a per-chunk checksum |
| Every tensor family | **STOP on two families; plan applied** | All 23 families sampled, inventory equal to the config; the pooled table covers under 99% of `moe.router` (98.49%) and `moe.shared_down` (98.88%). By the plan the router takes its own table, `shared_down` stays raw, and per-family widths give **r = 0.7329** over all 108.76 GB of matrices; coding `shared_down` at 3 bits, as the plan's own per-matrix rule would, gives 0.7128 ([result](#per-family-result)) | A census instead of four 1 MiB samples per family; a STOP rule stated in the width the plan chooses |
| Decode under matmul contention | **Measured on a VM with the shipped kernels, and hosted** | VM, worst case: above 1 up to B = 4 GB/s in every run; B = 5 is marginal (all eight cases pass in the idle runs, worst 1.124, but one 4-thread run taken during background downloads fell to 0.969 FD4B and 0.999 FD3B); the full 1/r up to about 4.0 to 4.3 GB/s for streamed input; at B = 6 it fails for streamed input at 4 and 2 threads, decoder-limited. Hosted x86_64 and arm64: above 1 at every B up to 6. Resident slowdown 2.3 to 6.1 ([results](#results-on-a-4-vcpu-vm)) | Which placement a real row pipeline sees |
| Supported container/reader | Not started | | Everything |

Kernel speed alone still does not make deployment profitable, as the Huffman
note already says.

For escape fraction `p`, `r = .75 + .5*p` before dictionary/framing/row costs.
For disk bandwidth `B=3 GB/s`, decode must exceed 3 GB/s to beat raw reads under
independent overlap, and exceed `B/r` to feed the saturated compressed stream.
These are different thresholds; neither is a full-model speed measurement.

The proposed strict "at most six percentage points" ratio premium is already
incompatible with historical `r=.689`: even zero escapes cost **6.1 points**.
The four-range Huffman payload result `.672227` is a different sample arm and
must not be silently compared as if it were an all-eight-range measurement.
Any continued campaign must report its actual sample-matched premium and label
a missed six-point requirement, rather than rounding it into a pass.
**Sample-matched premium, reported:** on the same four ranges (layers 1, 12, 24,
36 `f_a_proj`), FD4B with the gate-1 dictionary retains 0.750257 against the
four-stream Huffman payload's 0.672227: **7.80 points, the six-point requirement
is missed**. FD3B (below) retains 0.704132 there: 3.19 points. Both figures cover
KDA `f_a_proj` only. Over every family, byte-weighted, 4-bit retains 0.751122
against the per-family Huffman bound of 0.677074 (7.40 points). The plan's
per-family choice, 0.732863, is 5.58 points above that bound, but the two are not
like for like (correction, 2026-09-24 review): the plan stores `moe.shared_down` raw
(r = 1) while the bound Huffman-codes it (0.687925), and that raw fallback alone is
about 2.32 of the 5.58 points (0.0745 x (1 - 0.6879)). With every family coded at
its better width, `shared_down` at its 3-bit 0.730941, the byte-weighted ratio is
0.712812 and the premium over per-family Huffman is **3.57 points**.

No full-model speedup, full-model storage number or ratio win over Huffman is
claimed. The streamed-byte reduction is budget-dependent: pinned trunk bytes
are not reread every token.

## Bit-width curve: 3, 4 and 5 bits, sign split, entropy bounds

`tools/trunk_dictionary_gate.py` now scores, for any 256-count high-byte
histogram, the exact payload of a k-bit index into the first `2^k - 1` pooled
entries plus an escape (k = 3, 4, 5), a sign-split variant (raw sign bit plus a
(k-1)-bit index into `2^(k-1) - 1` pooled magnitudes, same k + 8 bits per value),
the unconstrained Huffman code for the high byte, and the order-0 entropy of the
high byte. Payloads are integer bit counts, `k + 8` per value plus 8 per escape,
so `r = bits / 16n` exactly; framing, dictionary and row index are excluded as in
gate 1. Given raw bytes it adds the whole-BF16 order-0 entropy, `H(low)` and
`H(low | high)`. The histogram job computes all of it per range and pooled on the
eight pinned ranges, and uploads the pooled whole-BF16 value histogram. Unit
tests use hand-checkable histograms: exact bit counts, a dyadic signed histogram
where Huffman meets entropy (2.96875 bits), whole-value bounds, and the prototype
threshold decided in integers at exactly 1.5 points.

**Real numbers, from committed counts.** Two committed high-byte histograms of real
K3 bytes feed the curve. The first is the Huffman research job's pool of four of the
gate-1 ranges, KDA `f_a_proj` at layers 1, 12, 24 and 36:
[research-two-symbol.json](../measurements/research-two-symbol.json)
`/experiments/1/report/high_byte_histogram`, from CI run 35462000448 (job
105947405103), 2,097,152 values. That identity is cross-checked by a unit test:
under the gate-1 dictionary those counts predict 1,077 escapes, exactly the sum
of the four per-range escape counts that the separate FD4B rate job recorded
(254 + 308 + 273 + 242, CI run 35498176696). The tool's output on them is
[fixed-width-curve-four-ranges.json](../measurements/fixed-width-curve-four-ranges.json),
recomputed by a unit test:

| Scheme on the four `f_a_proj` ranges | Entries | Escapes | p escape | Payload r | Gain over 4-bit, points | Above Huffman, points |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 3-bit | 7 | 69,742 | 3.3256% | 0.704128 | +4.60 | 3.21 |
| Fixed 4-bit | 15 | 643 | 0.0307% | 0.750153 | 0 | 7.81 |
| Fixed 5-bit | 31 | 0 | 0 | 0.812500 | -6.23 | 14.04 |
| Sign split, 1 + 2 bits | 3 | 111,453 | 5.3145% | 0.714072 | +3.61 | 4.20 |
| Sign split, 1 + 3 bits | 7 | 869 | 0.0414% | 0.750207 | -0.01 | 7.82 |
| Sign split, 1 + 4 bits | 15 | 0 | 0 | 0.812500 | -6.23 | 14.04 |
| Unconstrained Huffman, high byte | | | | 0.672050 | +7.81 | 0 |
| Order-0 entropy, high byte (2.7248 bits) | | | | 0.670303 | +7.99 | -0.17 |
| Order-0 entropy, whole BF16 | | | | not computed | | |

The whole-BF16 bound needs raw bytes, which that job did not keep; the eight-range
table below has it. Here the tables are this subset's own ranking. With the gate-1
dictionary and its first seven entries instead, the same bytes give 0.750257 (1,077 escapes) and
0.704132 (69,758 escapes): 4.61 points. Five bits buys nothing: the support is
only 27 values here (29 pooled over eight ranges) and 4 bits already covers
99.97%, so a 5-bit index pays a full bit per value for 0.03% of escapes. The only
direction that clears the 1.5-point bar is **down**, to 3 bits, where seven entries
hold 96.67% and the 3.33% of escapes cost less than the saved bit. The sign is
close to a fair coin, so a (k-1)-bit magnitude table covers `2^k - 2` signed values
in pairs while the k-bit table covers the same pairs plus one more value: the sign
split trails at 3 and 4 bits, by 0.99 points at 3 bits, where that extra entry
holds 1.99% of values, and ties at 5 bits, where neither escapes. Huffman is a further 3.21 points below 3-bit, and the
high-byte entropy only 0.17 below Huffman. Beyond that only the low byte is left:
an order-0 coder of whole values can gain at most `8 - H(low | high)` bits per
value over these bounds, which the CI job now measures.

The second is all eight gate-1 ranges, pooled, from
[trunk-dictionary-gate1.json](../measurements/trunk-dictionary-gate1.json)
`/report/pooled/histogram` (4,194,304 values): the tool's output is
[fixed-width-curve-eight-ranges.json](../measurements/fixed-width-curve-eight-ranges.json),
recomputed by the same unit test. The CI histogram job at `cb4ab38` (job 107131235189)
read the same eight ranges again, with the same hashes and counts, and adds the
whole-BF16 bounds that need raw bytes; they are its figures, not recomputable from the
committed counts:

| Scheme on the eight gate-1 ranges | Entries | Escapes | p escape | Payload r | Gain over 4-bit, points | Above Huffman, points |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 3-bit | 7 | 133,404 | 3.1806% | 0.703403 | +4.68 | 2.96 |
| Fixed 4-bit | 15 | 1,902 | 0.0453% | 0.750227 | 0 | 7.64 |
| Fixed 5-bit | 31 | 0 | 0 | 0.812500 | -6.23 | 13.87 |
| Sign split, 1 + 2 bits | 3 | 206,903 | 4.9330% | 0.712165 | +3.81 | 3.83 |
| Sign split, 1 + 3 bits | 7 | 3,013 | 0.0718% | 0.750359 | -0.01 | 7.65 |
| Sign split, 1 + 4 bits | 15 | 0 | 0 | 0.812500 | -6.23 | 13.87 |
| Unconstrained Huffman, high byte | | | | 0.673839 | +7.64 | 0 |
| Order-0 entropy, high byte (2.7215 bits) | | | | 0.670093 | +8.01 | -0.37 |
| Order-0 entropy, whole BF16 (10.529 bits; CI only) | | | | 0.658087 | +9.21 | -1.58 |

Adding the layer-92 `g_proj` range and three more `f_a_proj` ranges moves nothing
that matters: 3 bits still gain 4.7 points over 4 and sit 3.0 above Huffman. The low
byte carries 7.971 bits of entropy alone and 7.808 given the high byte, so an order-0
coder of whole values could gain only 0.19 bits per value, 1.2 points, over the
high-byte bounds.

**Scope.** The two tables cover two matrix families between them, KDA `f_a_proj`
(0.116% of the trunk, and all of the four-range table) and MLA `g_proj`, 4.00% of the
trunk's bytes together. The `families` job below computes the same curve for every
family.

## Every tensor family, not two

Gate 1's eight ranges are seven KDA `f_a_proj` ranges and one MLA `g_proj` range,
families holding **4.00% of trunk bytes**; every committed dense sample in the
repository is one of those eight. The 1,251 BF16 matrices of the trunk, from the
released shapes (`python3 tools/trunk_family_gate.py inventory`, checked by unit
tests against the engine's binder), with the exact row-index cost of the next
section:

| Family | Shape | Matrices | GB | Trunk % | Gate-1 samples | FDRX per row, points | Rows per entry | FDRX grouped, points |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `kda.q_proj`, `k_proj`, `v_proj`, `g_proj` (each) | 12288 x 7168 | 69 | 12.155 | 11.171 | none | 0.0279 | 2 | 0.0140 |
| `kda.o_proj` | 7168 x 12288 | 69 | 12.155 | 11.171 | none | 0.0163 | 1 | 0.0163 |
| `kda.f_a_proj` | 128 x 7168 | 69 | 0.127 | 0.116 | 7 ranges | 0.0288 | 2 | 0.0148 |
| `kda.f_b_proj` | 12288 x 128 | 69 | 0.217 | 0.199 | none | 1.5630 | 64 | 0.0249 |
| `kda.b_proj` | 96 x 7168 | 69 | 0.095 | 0.087 | none | 0.0291 | 2 | 0.0151 |
| `mla.q_a_proj` | 1536 x 7168 | 24 | 0.528 | 0.486 | none | 0.0280 | 2 | 0.0140 |
| `mla.q_b_proj` | 18432 x 1536 | 24 | 1.359 | 1.249 | none | 0.1302 | 6 | 0.0217 |
| `mla.kv_a_proj_with_mqa` | 576 x 7168 | 24 | 0.198 | 0.182 | none | 0.0281 | 2 | 0.0141 |
| `mla.kv_b_proj` | 24576 x 512 | 24 | 0.604 | 0.555 | none | 0.3907 | 16 | 0.0245 |
| `mla.o_proj` | 7168 x 12288 | 24 | 4.228 | 3.886 | none | 0.0163 | 1 | 0.0163 |
| `mla.g_proj` | 12288 x 7168 | 24 | 4.228 | 3.886 | 1 range | 0.0279 | 2 | 0.0140 |
| `moe.latent_down` (`routed_expert_down_proj`) | 3584 x 7168 | 92 | 4.727 | 4.344 | none | 0.0279 | 2 | 0.0140 |
| `moe.latent_up` (`routed_expert_up_proj`) | 7168 x 3584 | 92 | 4.727 | 4.344 | none | 0.0558 | 3 | 0.0186 |
| `moe.shared_gate`, `shared_up` (each) | 6144 x 7168 | 92 | 8.103 | 7.447 | none | 0.0279 | 2 | 0.0140 |
| `moe.shared_down` | 7168 x 6144 | 92 | 8.103 | 7.447 | none | 0.0326 | 2 | 0.0163 |
| `moe.router` (`gate.weight`) | 896 x 7168 | 92 | 1.182 | 1.086 | none | 0.0280 | 2 | 0.0141 |
| `dense.gate_proj`, `up_proj` (each, layer 0) | 33792 x 7168 | 1 | 0.484 | 0.445 | none | 0.0279 | 2 | 0.0140 |
| `dense.down_proj` (layer 0) | 7168 x 33792 | 1 | 0.484 | 0.445 | none | 0.0059 | 1 | 0.0059 |
| **All 1,251 matrices** | | | 108.759 | 99.95 | 4.00% of bytes | 0.0340 | | 0.0148 |

KDA q/k/v/g/o hold 55.86% of trunk bytes, the shared experts 22.34%, the latent
projections 8.69%, MLA o/g 7.77%; none was sampled. Shares use the documented
108.81 GB; the router counts as BF16 because an F32 router would exceed that
total, and the remaining ~0.05 GB is norms, convolutions and biases, which the
codec does not target.

The `families` CI job (`tools/trunk_family_gate.py sample`) reads every shard
header at the pinned revision for the exact inventory and dtypes, then four
systematic 1 MiB samples per BF16 matrix family (centred at 1/8, 3/8, 5/8, 7/8 of
the family's bytes in layer order, about 92 MiB in all), and reports per family
the coverage, the full bit-width curve and the whole-BF16 entropy under **one**
dictionary ranked on the byte-weighted mixture of the families, plus
byte-weighted pooled ratios (exact weighted means) and both per-family and
single-code entropy bounds. It also scores the gate-1 dictionary on each family,
which is the unseen-family generalization test gate 1 could not make. Its gate is
the gate-1 rule applied per family: at least 99% coverage in every family.
Sample hashes are recorded on first observation, not pinned in advance; `--pins`
takes a committed record of them and then requires the same bytes. Unit tests
cover the classification (MLA versus KDA `o_proj`/`g_proj` by layer), the plan, the
exact weighting and the per-family STOP; the run's result is
[below](#per-family-result).

### Per-family plan

The rules for reading the `families` run are fixed here, before its data exist,
so the result cannot pick its own test. In order:

1. **Inventory first.** `config_check` compares every family's header bytes with
   the config-derived bytes in the table above. Any mismatch corrects this table
   (and the shares quoted from it) before any ratio is read.
2. **Coverage gate, per family.** PASS needs at least 99% of each sampled
   family's high bytes inside the one byte-weighted pooled 15-entry table, by
   integer counts (`gate.status`). The gate-1 table is scored on every family as
   well (`gate1_15`): fitted on 4.00% of the bytes and scored on the rest, it is
   the generalization test gate 1 could not make, and it is reported whether it
   passes or not.
3. **A STOP names families, not the campaign.** Every FD stream carries its own
   magic, length and table, so each matrix is encoded on its own. A family in
   `gate.failed_families` gets its own table if its `best_local_15` coverage is at
   least 99% (15 bytes per matrix), and otherwise stays raw BF16. The ratio is
   then restated over all 108.76 GB of matrices with raw families at r = 1, never
   over the passing families alone.
4. **Width, per family.** The 1.5-point rule decides whether FD3B is worth
   carrying at all, on the byte-weighted curve (`byte_weighted_pooled.decision`).
   If it is, each family takes the smaller of its own 3- and 4-bit payloads; the
   magic selects the decoder, so mixing widths costs no format change. The
   byte-weighted ratio of that per-family choice is the figure to quote.
5. **Bounds, per family.** The Huffman and order-0 bounds per family and for one
   shared code state the fixed-width premium family by family; the whole-BF16
   order-0 bound says how much any memoryless model of the low byte could still add
   (a coder that models context across neighbouring values or rows is not bounded
   by it).
6. **Provenance.** Record the run ID, head and artifact ID; commit the report as
   `docs/measurements/trunk-family-gate.json` with every derived field recomputed
   from the committed histograms and matched before writing (the gate 2/3
   procedure); commit the observed sample identities (`shard`, `offset`, `bytes`,
   `sha256` from each family's `samples`) as `{"samples": [...]}` so a rerun with
   `--pins` must read the same bytes.

Limits of the sample: four 1 MiB ranges per family are systematic, not a census,
and the table is fitted and scored on the same bytes. The deployment decision does
not rest on the sample, though. A container writer sees every value of every
matrix, knows each matrix's exact escape count before it writes, and can write any
matrix raw whose FD payload would not be smaller; that choice is exact and costs
nothing at decode time.

### Per-family result

[CI run 35845709912](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35845709912),
job 107131350952, head `cb4ab38`: every shard header at the pinned revision, then four
1 MiB samples from each of the 23 BF16 matrix families, 92 MiB in all. The report is
committed as [trunk-family-gate.json](../measurements/trunk-family-gate.json), recorded
unchanged from the job's stdout `FAMILY_REPORT` line; a unit test recomputes every
field from its histograms except the whole-BF16 bounds, whose value histograms are only
in the run's artifact (ID 10743097432). The 92 sample identities are committed as
[trunk-family-pins.json](../measurements/trunk-family-pins.json) for `--pins`. Read by
the plan above:

1. **Inventory.** All 23 families' header bytes equal the config-derived bytes in the
   table above, so the table and its shares stand.
2. **Coverage gate: STOP.** One byte-weighted pooled 15-entry table covers 99.78% of
   the sampled bytes but under 99% of `moe.router` (98.49%) and `moe.shared_down`
   (98.88%). The gate-1 table, fitted on 4.00% of the bytes, covers 99.54% byte-weighted
   and under 99% of all three shared-expert projections (`shared_down` 97.59%,
   `shared_gate` 98.61%, `shared_up` 98.54%; together 22.35% of matrix bytes, 22.34%
   of the documented trunk): gate 1's pass does not generalize to them.
3. **The STOP names families.** The router's own best 15 entries cover 99.96% of it, so
   it takes a table of its own; `shared_down`'s own best 15 cover 98.94%, so it stays
   raw BF16.
4. **Width.** On the byte-weighted curve 3 bits beat 4 by 3.78 points (0.713279 against
   0.751122), over the 1.5-point bar, so FD3B is carried. Each family then takes the
   smaller of its 3- and 4-bit payloads: 3 bits for 19 families (91.21% of matrix
   bytes, the router on its own table), 4 bits for the three dense-layer MLP matrices
   (1.34%, whose high bytes are spread too widely for seven entries) and raw for
   `shared_down` (7.45%). Restated over all 108.76 GB of matrices, the raw family at
   r = 1, that is **r = 0.732863** (`trunk_family_gate.family_plan` on the committed
   report, unit-tested), the figure to quote: 29.05 GB fewer bytes before framing and
   the row index (0.0148 to 0.034 points).
   **Correction (2026-09-24 review).** The STOP rule in step 3 is scored on 15-entry
   (4-bit) coverage, while step 4 chooses 3-bit widths, so the rule and the decision it
   controls are in different units: `shared_down`'s 3-bit payload is 0.730941, well
   under 1, and the deployment rule stated above (a writer stores raw only a matrix
   whose FD payload would not be smaller) would code it. Coding it gives 0.712812
   over the same matrices, 2.00 points below the plan's figure; the router's own table
   moves its ratio by 0.005 points (0.707432 against 0.707483 pooled) and the trunk
   total by under 0.0001. The pre-registered figure is kept as what the rule yields;
   quote 0.7128 beside it, and state any future STOP in the width being decided.
5. **Bounds.** Per-family Huffman codes for the high byte give 0.677074 byte-weighted;
   the order-0 entropy of the byte-weighted mixture, the bound for one shared code, is
   0.677387 (an entropy, not a Huffman code); the order-0 whole-BF16 entropy is
   0.661496 per family and 0.663900 for the mixture. The plan's r is 5.58 points above
   the per-family Huffman bound, of which about 2.32 points are the raw `shared_down`
   fallback that the bound does not take; like for like, every family coded, the
   premium is 3.57 points. 4 bits everywhere would be 7.40 points above the bound.

Bold coverages are under 99%; "own best 15" is each family's own 15 most frequent high
bytes.

| Family | Trunk % | Pooled 15 % | Gate-1 15 % | Own best 15 % | r 3-bit | r 4-bit | Huffman r | Plan |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dense.down_proj` | 0.445 | 99.7509 | 99.1339 | 99.7509 | 0.788420 | 0.751246 | 0.716373 | 4-bit |
| `dense.gate_proj` | 0.445 | 99.7583 | 99.0766 | 99.7583 | 0.784362 | 0.751209 | 0.714405 | 4-bit |
| `dense.up_proj` | 0.445 | 99.7648 | 99.0805 | 99.7650 | 0.784547 | 0.751176 | 0.714411 | 4-bit |
| `kda.b_proj` | 0.087 | 99.9750 | 99.9349 | 99.9769 | 0.707550 | 0.750125 | 0.672755 | 3-bit |
| `kda.f_a_proj` | 0.116 | 99.8953 | 99.9559 | 99.9564 | 0.701896 | 0.750524 | 0.672340 | 3-bit |
| `kda.f_b_proj` | 0.199 | 99.6071 | 99.6059 | 99.6824 | 0.740979 | 0.751965 | 0.698875 | 3-bit |
| `kda.g_proj` | 11.171 | 99.9817 | 99.9399 | 99.9817 | 0.706118 | 0.750092 | 0.671951 | 3-bit |
| `kda.k_proj` | 11.171 | 99.9620 | 99.9050 | 99.9634 | 0.707229 | 0.750190 | 0.673793 | 3-bit |
| `kda.o_proj` | 11.171 | 99.9778 | 99.9392 | 99.9778 | 0.706862 | 0.750111 | 0.672551 | 3-bit |
| `kda.q_proj` | 11.171 | 99.9640 | 99.9029 | 99.9642 | 0.707156 | 0.750180 | 0.673798 | 3-bit |
| `kda.v_proj` | 11.171 | 99.9779 | 99.9465 | 99.9779 | 0.704504 | 0.750110 | 0.671664 | 3-bit |
| `mla.g_proj` | 3.885 | 99.9503 | 99.8479 | 99.9517 | 0.734207 | 0.750249 | 0.692105 | 3-bit |
| `mla.kv_a_proj_with_mqa` | 0.182 | 99.9832 | 99.9374 | 99.9842 | 0.706346 | 0.750084 | 0.671868 | 3-bit |
| `mla.kv_b_proj` | 0.555 | 99.4282 | 99.9442 | 99.9447 | 0.708527 | 0.752859 | 0.676164 | 3-bit |
| `mla.o_proj` | 3.885 | 99.9722 | 99.9087 | 99.9729 | 0.716802 | 0.750139 | 0.683066 | 3-bit |
| `mla.q_a_proj` | 0.486 | 99.9790 | 99.9259 | 99.9808 | 0.711020 | 0.750105 | 0.675102 | 3-bit |
| `mla.q_b_proj` | 1.249 | 99.6161 | 99.9436 | 99.9436 | 0.706995 | 0.751920 | 0.677021 | 3-bit |
| `moe.latent_down` | 4.344 | 99.9852 | 99.9425 | 99.9865 | 0.705731 | 0.750074 | 0.671827 | 3-bit |
| `moe.latent_up` | 4.344 | 99.9851 | 99.9484 | 99.9851 | 0.704828 | 0.750075 | 0.671447 | 3-bit |
| `moe.router` | 1.086 | **98.4896** | 99.9621 | 99.9621 | 0.707483 | 0.757552 | 0.674823 | 3-bit, own table (0.707432) |
| `moe.shared_down` | 7.447 | **98.8763** | **97.5853** | **98.9352** | 0.730941 | 0.755619 | 0.687925 | raw BF16 (1) |
| `moe.shared_gate` | 7.447 | 99.3912 | **98.6058** | 99.3912 | 0.723351 | 0.753044 | 0.682003 | 3-bit |
| `moe.shared_up` | 7.447 | 99.3688 | **98.5392** | 99.3920 | 0.722976 | 0.753156 | 0.682474 | 3-bit |
| **Byte-weighted** | 99.951 | 99.7756 | 99.5373 | | 0.713279 | 0.751122 | 0.677074 | **0.732863** |

The r columns score every family on the pooled tables (its first 7 or 15 entries);
the plan column scores the router on its own. The shared experts' high bytes spread
wider than the KDA projections', so their 3-bit payloads sit 1.6 to 2.6 points above
the KDA q/k/v/g/o ones and `shared_down` falls out of FD altogether under the plan's
rule. Limits as the plan states: four systematic 1 MiB samples per
family, fitted and scored on the same bytes, and a container writer that sees every
matrix would decide each matrix's escapes, and raw or not, exactly. No speed,
full-model or storage-size claim.

## FD3B: the 3-bit prototype

On the committed counts the curve clears the 1.5-point bar at 3 bits, so
`benchmarks/fixed_dictionary.h`
now has FD3B beside FD4B: 3-bit codes as a little-endian bitstream (code i in
bits 3i..3i+2), code 7 escapes to a raw byte, then the raw low bytes, the escape
bytes and 32 zero bytes of read slack. Escapes are 3.2% of values pooled over the
eight ranges (133,404 of 4,194,304) and 3.3% on the four `f_a_proj` ranges, so the
FD4B approach of patching each escape in a branchy loop would mispredict; FD3B
computes an exclusive prefix count of the escape lanes and shuffles the next
escape bytes into place with no branch. Decoders: scalar reference, SSSE3,
AVX2 (both halves of 32 values in one register) and NEON; `bench_fixed_dictionary`
dispatches on the magic. `tools/bench_fixed_dictionary.py --index-bits 3` is an
independent encoder, and CI runs FD3B through the same exactness (ASan/UBSan,
both ISAs, all eight ranges, golden bytes and negative controls) and rate jobs
as FD4B.

Tests: every length 0..131 and 8193, an independently hand-packed golden stream,
all byte values, no and all escapes, escape rates 0.1% to 90% and bursts of 1 to
40, truncation at every length, header, table, padding-bit and slack corruption,
escape underflow and surplus, and payload corruption caught by the byte
comparison. Every stream, and every row index a range is decoded through, is
parsed and decoded from an exact-size heap copy, so ASan sees any read past its
end. That matters for escape codes that outnumber the header's count: an FD3B
vector step may consume up to 32 escapes and so end past the count, and only the
check after each step keeps the scalar tail from reading on past the slack. The
tests lower the count by 1 to 200 (both sides of `FWD3_SLACK`) with every value an
escape or escapes only in the last 300 values, at four lengths in both formats;
with FD3B's three per-step checks removed they fail with a heap over-read under
ASan (SSSE3, AVX2) and on a guard page (NEON). They passed on the 4-vCPU VM for the
scalar, SSSE3 and AVX2 paths under ASan/UBSan with gcc 13.3 and at -O3 with clang 18,
and for NEON under qemu-aarch64 (gcc 13.3 cross, -O3 and UBSan; ASan does not run
under qemu user mode, so the NEON ASan run is the macos-14 job's).

**Hosted exactness and rates.** [CI run 35845709912](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35845709912)
at head `cb4ab38` ran FD3B through the gate 2 and gate 3 jobs. Exactness (jobs
107131351050 on `ubuntu-latest`, SSSE3 and AVX2, and 107131351014 on `macos-14`,
NEON; ASan and UBSan): the unit test binary passed, and all eight ranges and the
pooled 8 MiB decoded byte-exact, with the golden layout, the odd tail and the five
negative controls rejected. Rates (jobs 107131626457 and 107131626496, the gate 3
protocol: three runs per arm, reconstructed BF16 GB/s), recorded unchanged from the
jobs' stdout `CODEC_REPORT` lines, their derived fields recomputed and matched, in
[fixed-dictionary-rate-fd3b-x86_64.json](../measurements/fixed-dictionary-rate-fd3b-x86_64.json),
[fixed-dictionary-rate-fd3b-avx2-x86_64.json](../measurements/fixed-dictionary-rate-fd3b-avx2-x86_64.json)
and [fixed-dictionary-rate-fd3b-arm64.json](../measurements/fixed-dictionary-rate-fd3b-arm64.json):

| Input | x86_64 `ssse3_pshufb` | x86_64 `avx2_vpshufb` | arm64 `neon_tbl` |
|---|---:|---:|---:|
| Eight 1 MiB ranges, 24 runs, min / median / max | 9.479 / 9.658 / 9.721 | 15.187 / 15.288 / 15.404 | 6.274 / 11.807 / 13.619 |
| Pooled 8 MiB, 3 runs | 9.641, 9.637, 9.646 | 15.249, 15.265, 15.251 | 11.029, 11.544, 13.464 |

`rate_gate.status = PASS` for all three, every run above the 4 GB/s target. The same
jobs timed FD4B again on these runners: pooled 25.4 GB/s on x86_64 (a faster runner than
gate 3's 16.2) and 15.4 to 19.4 on arm64. So FD3B decodes at 38% (SSSE3) and 60%
(AVX2) of FD4B's pooled rate on x86_64 and about 71% on arm64: its escape expansion
costs rate, as the orientation runs below suggested. That FD4B re-timing exists only
in the job logs, and the committed FD4B rates are gate 3's from a different run and
runner, so these fractions are orientation rather than a matched measurement; no arm
times the SIMD unpack with conventional scalar escape patching, so whether the
branch-free expansion beats it at this escape rate is not measured. The AVX2 report's
`compiler_flags` field repeats the job's SSSE3 flags; the binary it timed was built
with `-mavx2`, as its `native` field says. Figures in this note that are quoted from
hosted job logs rather than from a committed file (this paragraph's FD4B re-timing and
the FD3B/FD4B ratios, the whole-BF16 and low-byte entropies) are identified by run and job
id; GitHub keeps those logs only for its retention period, so the committed JSON files
are the durable record. The workflow now records the AVX2 flags for
those runs.

Kernel rates, **for orientation only and not admissible as a result**: they were
taken on the 4-vCPU cloud VM while unrelated builds ran on it (load average 5.7), and
no report file was kept. The hosted FD3B rates above replace them. (Intel Xeon
2.8 GHz Cascade Lake VM, 4 vCPU; synthetic bytes drawn from the committed four-range
histogram; `bench_huf4` units, reconstructed GB/s, median and minimum of 15 runs
interleaved with the other arms.)

| Input | FD4B `ssse3_pshufb` | FD3B `ssse3_pshufb` | FD3B `avx2_vpshufb` |
|---|---:|---:|---:|
| 1 MiB | 15.41 (min 13.75) | 6.37 (min 6.13) | 9.52 (min 8.06) |
| 8 MiB | 12.61 (min 10.09) | 5.86 (min 5.26) | 8.49 (min 7.45) |

With the escape patch removed (wrong output, timing only), the SSSE3 FD3B loop
ran at FD4B's speed (about 12 GB/s on 8 MiB), which suggests the escape expansion
costs it about half its rate and the AVX2 path wins part of it back; the same
caveat applies, and the hosted rates above bear it out.

## Row-seekable layout: FDRX

A whole-row range must decode without reading the rows before it. In the planar
FD layout the index and low planes are already seekable: rows `[a, b)` of a
matrix with `C` columns are index bytes `[a*C*k/8, b*C*k/8)` and low bytes
`[a*C, b*C)`. Only the escape stream is variable. FDRX stores, per group of `G`
rows, the number of escapes before the group: `E(jG)` as u32, plus a 16-byte
header (magic, rows, cols, G). For any row `r`,
`E(r) = E(G*floor(r/G)) + (escape codes in rows G*floor(r/G) .. r-1)`, so the
range's escapes are bytes `[E(a), E(b))` and the range is an ordinary FD view.
With `G = 1` no code is ever counted; with `G = ceil(8192/C)` a start inside a
group counts at most 8,191 codes, under one 4 KiB page of 4-bit index that an
aligned read fetches anyway.

Exact costs for K3 (table above): **padding is zero bytes** for every K3 width,
since all are multiples of 128 and so `C*k` is a multiple of 8 for k = 3 and 4;
the per-row index costs 16 + 4R bytes per matrix, **0.0340 points** of the
108.76 GB of matrices (36,962,224 bytes), dominated in relative terms by
`kda.f_b_proj` (C = 128: 1.56 points of its own bytes) and `mla.kv_b_proj`
(0.39); the grouped index costs **0.0148 points** (16,134,544 bytes) and at most
0.025 points for any family. Framing is 32 header bytes per matrix, plus 32 bytes
of slack for FD3B. A chunk needs three range reads (index, low, escapes) instead
of one; with O_DIRECT each rounds out to 4 KiB, at most 24 KiB more traffic per
8 MiB chunk (0.29%), which is read traffic, not storage. In the terms of the
engine's `rows_apply` (unchanged here), the reader would fetch the three slices of
the next `count` rows into its idle buffer and a decoder would expand them into
the matmul's input: the raw chunk stays at most 8 MiB and its compressed slices
are 6 MiB (FD4B) or 5.5 MiB (FD3B) plus escapes, about 0.13 MiB at 3.3%.

`fwd_rows_parse` checks that checkpoints start at zero, never decrease and never
exceed the group's value count; `fwd_rows_view` returns rows `[a, b)` as a view
that both decoders accept. The index is not self-authenticating: a checkpoint off
by one inside a group can shift escapes without a count mismatch, and the tests
show only the independent byte comparison catches it. A deployment container needs
a checksum per chunk. The entry count is `ceil(R/G)` computed as `R/G + (R%G !=
0)`: the usual `(R + G - 1)/G` wraps with a 32-bit `size_t` at `G` near 2^32, and
an index with no checkpoints then parsed and had one read from past its end (shown
with a freestanding i386 build under qemu; the hosted jobs are 64-bit). Tests (all
under ASan/UBSan): every `[a, b)` of a 13 x 64 matrix in FD4B and FD3B and an 11 x
128 FD3B matrix, G = 1, 2, 3, 7, R and R + 5; 40 x 512 (both widths) and 6 x 7168
(both widths) matrices with ranges starting and ending inside groups mid-matrix;
empty ranges, reversed and out-of-range bounds; structural corruption refused at
parse; an off-by-one checkpoint caught; G = 2^32 - 1 needing exactly one entry.
The contention benchmark below also decodes a whole 12288 x 7168 matrix through a
per-row FDRX index in 22 chunks of up to 585 rows, byte-exact before and after
timing.

## Decode under matmul contention

> **Measured on 2026-09-23 on a 4-vCPU cloud VM, at 4 and 2 threads, by the protocol
> below and with the shipped kernels; the hosted x86_64 and arm64 legs ran
> in CI run 35845709912.** The tables are the driver's output pasted as printed; the
> reports are [decode-contention-x86_64-t4.json](../measurements/decode-contention-x86_64-t4.json)
> and [decode-contention-x86_64-t2.json](../measurements/decode-contention-x86_64-t2.json)
> (VM), [decode-contention-x86_64-hosted.json](../measurements/decode-contention-x86_64-hosted.json)
> and [decode-contention-arm64-hosted.json](../measurements/decode-contention-arm64-hosted.json)
> (CI).

Kernel rates alone (gate 3) had the decoder to themselves. In a streamed trunk the
decoder shares the machine with the matmuls it feeds.
`benchmarks/bench_decode_contention.c` puts both on one machine at once: one
decoder thread reconstructs a 12288 x 7168 BF16 matrix (the KDA `q_proj` shape) in
22 chunks of up to 585 rows, the most whole rows that fit the row pipeline's 8 MiB
buffer, each chunk located through a per-row FDRX index and written into two
alternating 8 MiB buffers, while the engine's own `k3_matmul_bf16` multiplies the
same shape on the remaining OpenMP threads. Four arms, repeated in interleaved
order: decode alone, the matmul on all T threads (the uncompressed baseline), the
matmul on T - 1 threads, and decoder plus matmul at once. In the concurrent arm
only work finished before the deadline counts, and each side keeps running until
the other has finished, so every counted unit ran against a live competitor. The
input is either `stream` (all 22 chunks in turn, so the decoder reads DRAM like a
cold row pipeline) or `hot` (one chunk re-decoded in place, as right behind the read
that landed it; at 5.6 to 6.0 MiB compressed (8 MiB x r) the chunk exceeds the 1 MiB
per-core L2 and shares the 33 MiB L3 with the matmul's 176 MB stream, so `hot` means
recently touched, not verified cache-resident). Every chunk is compared byte for byte with the raw matrix through
the scalar and native decoders before timing, and through the native one after.

Exactness is not at stake here: the decoder reproduces the BF16 bytes exactly, so
the matmul sees the same operands in the same partition and reduction order, and
logits are unchanged by construction. Only time is measured.

Setup, exact and deterministic (fixed seed, not timings): high bytes are drawn
from the committed four-range histogram, low bytes uniform; FD4B uses the gate-1
table and FD3B its first seven entries. Of 88,080,384 values, FD4B escapes 44,671
(0.0507%, against 0.0514% for the same table on the real four ranges) and FD3B
2,927,952 (3.324%, against 3.326%). Payload plus framing gives r = 0.750254 (FD4B)
and 0.704121 (FD3B); the FDRX index is 49,168 bytes, 0.028 points, small enough to
keep resident rather than stream.

**Model** (`tools/bench_decode_contention.py`, unit-tested on hand-checkable
rates). Per GB of raw weights, with SSD rate B, contended decode rate D_c and
matmul rates M_all (all threads) and M_c (the rest, against the decoder):

- uncompressed, streamed: `max(1/B, 1/M_all)`
- compressed, streamed: `max(r/B, 1/D_c, 1/M_c)`, and the largest term names the
  limiting stage
- resident: `1/M_all` against `max(1/D_c, 1/M_c)`

Decoding is hidden when `1/D_c <= max(r/B, 1/M_c)`. The CPU cost of decoding the
whole trunk is `108.81 / D_c` core-seconds per token.

Each figure is derived twice: from every arm's median, and as the **worst case**,
from every arm at its run least favourable to compression. The speedup falls as
M_all rises and rises with D_c and M_c, so the worst case pairs M_all's fastest
run with the slowest D_c and M_c; it is at or below the speedup of every repeat,
however the runs pair up, and the worst resident slowdown at or above every
repeat's (unit-tested on random arms). The worst case is an envelope over
run-to-run variance only: an extreme order statistic whose value depends on the
repeat count (9 on the VM, 5 hosted, so the two are not comparable), with no
confidence level, and it does not bound the model's own error (perfect overlap of
the stages, no disk in the loop, no fill or drain across the 22 chunks and two
buffers).

**Decision rules, fixed before the data.** A streamed deployment at SSD rate B is
worth building only if the contended **worst-case** speedup at that B exceeds 1;
the median is reported beside it. (Amended on 2026-09-23, before the runs tabulated
below but after the 2026-09-22 orientation reading noted at the end of this
section: the rule first named the per-arm minimum, which also takes M_all's slowest
run, so one disturbed run of the uncompressed baseline could pass the gate while the
median failed, and would report the most favourable resident slowdown. On the data
below the amendment changes nothing: every `Matmul all` run exceeds 6 GB/s, so raw
streaming is SSD-bound at every tabulated B under either rule.) The
resident slowdown is expected to exceed 1: a pinned trunk layer is read once, so
it should be decoded once when pinned and kept raw, and this measurement prices
that choice rather than gating it. The VM has 4 cores, where the decoder's core is
a quarter of the matmul's; the 2-thread run stands in for a smaller machine, with
the caveat that two threads on a 4-vCPU VM keep the whole L3 and memory bandwidth,
so it is not a bound on a real 2-core host; larger machines lose a smaller share.

**Protocol.** Build with the engine's flags, as the `contention` CI job does:

```sh
flags='-O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter -pthread -ffp-contract=off'
inc='-Iinclude -Iinclude/k3 -Ithird_party -Isrc/core -Ibenchmarks'
mkdir -p build/contention
cc $flags -march=native -fopenmp $inc -c src/core/k3_ops.c -o build/contention/k3_ops.o
cc $flags -Werror -march=native -fopenmp $inc -c benchmarks/bench_decode_contention.c -o build/contention/contention.o
cc build/contention/contention.o build/contention/k3_ops.o -o build/contention/bench-decode-contention -lm -pthread -fopenmp
```

Then, with no other build or benchmark running and a one-minute load average
below 0.5 (the report records it before and after), run all four
format x placement cases with 2 s per arm and 9 repeats, first on every core and
then on two:

```sh
RESEARCH_COMMIT=$(git rev-parse HEAD) python3 tools/bench_decode_contention.py \
  --binary build/contention/bench-decode-contention --threads 4 --seconds 2 --repeats 9 \
  --out docs/measurements/decode-contention-x86_64-t4.json
RESEARCH_COMMIT=$(git rev-parse HEAD) python3 tools/bench_decode_contention.py \
  --binary build/contention/bench-decode-contention --threads 2 --seconds 2 --repeats 9 \
  --out docs/measurements/decode-contention-x86_64-t2.json
```

The driver prints three tables per run (rates, then the break-even from the
medians and from the worst case); the results below paste them with the CPU model,
load averages and head from the report's `execution` field. The hosted
x86_64 and arm64 legs come from the `contention` CI job (`decode-contention-*`
artifacts, 1 s per arm, 5 repeats) and are recorded from the jobs' stdout
`CONTENTION_REPORT` lines.

### Results on a 4-vCPU VM

**Conditions.** Both runs with the benchmark and kernels at head `cb4ab38`:
`bench_decode_contention` is unchanged since `516f010`, and `k3_matmul_bf16` is the
exact decode kernel of [decode-kernels.md](decode-kernels.md), whose AVX-512 path this
`-march=native` build takes. `Intel(R) Xeon(R) Processor @ 2.80GHz` (family 6 model 85;
4 vCPUs, one thread per core, 1 MiB L2 per core, 33 MiB shared L3; SSSE3, AVX2 and
AVX-512F/BW/VL), Linux 6.18, gcc 13.3.0 with the flags above (`-O3 -march=native
-fopenmp -ffp-contract=off`), so FD4B dispatches to `ssse3_pshufb` and FD3B to
`avx2_vpshufb`. 2 s per arm and 9 repeats of the four arms in interleaved order, per
format x placement case, the cases run in turn. Every case was byte-exact (scalar and
native decoders before timing, native after), with the 44,671 FD4B and 2,927,952 FD3B
escapes stated above. The one-minute load average was 0.10 before the 4-thread run and
0.46 before the 2-thread run, under the 0.5 bar; 2.79 and 1.56 after them are the
benchmark's own threads. Nothing else of this work ran except a background git sync of
a few seconds every five minutes, which overlapped the last seconds of the 4-thread run
and the middle of the 2-thread one. A first 4-thread run was taken while CI logs were
being downloaded on the VM. The protocol above forbids other builds and benchmarks and
asks for a load average below 0.5, and that run did not record its load, so it is
reported here rather than discarded: its worst case at B = 5 was 0.969 (FD4B) and 0.999
(FD3B) for streamed input, below 1. Its raw report was not kept. The tables below are
the two idle runs, so B = 5 is a pass only for them. A VM's vCPUs can still share physical cores with other
tenants. The tables first published here were measured at `516f010` with the previous
`k3_matmul_bf16`; they remain in this note's history, and
[what the shipped kernel changed](#what-the-shipped-kernel-changed) compares them.

**4 threads** (the contended arm: decoder on one core, matmul on three).
Rates, reconstructed or consumed BF16 GB/s:

| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | Matmul rest | Matmul + decode | Core-s/token (contended) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | median | 7.461 | 6.002 | 29.518 | 23.444 | 22.950 | 18.13 |
| FD4B ssse3_pshufb | stream | min | 6.516 | 5.767 | 27.291 | 21.648 | 22.770 | 18.87 |
| FD4B ssse3_pshufb | stream | max | 7.741 | 6.489 | 32.386 | 24.286 | 24.872 | 16.77 |
| FD3B avx2_vpshufb | stream | median | 6.669 | 5.986 | 29.245 | 23.582 | 23.187 | 18.18 |
| FD3B avx2_vpshufb | stream | min | 6.279 | 5.622 | 26.461 | 22.953 | 21.767 | 19.35 |
| FD3B avx2_vpshufb | stream | max | 7.723 | 6.164 | 33.226 | 26.275 | 24.503 | 17.65 |
| FD4B ssse3_pshufb | hot | median | 9.843 | 8.019 | 29.473 | 24.436 | 21.848 | 13.57 |
| FD4B ssse3_pshufb | hot | min | 8.510 | 7.066 | 28.539 | 22.596 | 20.777 | 15.40 |
| FD4B ssse3_pshufb | hot | max | 11.245 | 8.836 | 31.761 | 26.694 | 24.917 | 12.31 |
| FD3B avx2_vpshufb | hot | median | 7.809 | 6.393 | 29.547 | 23.295 | 22.270 | 17.02 |
| FD3B avx2_vpshufb | hot | min | 6.865 | 6.209 | 26.971 | 21.603 | 20.528 | 17.52 |
| FD3B avx2_vpshufb | hot | max | 8.169 | 7.302 | 32.037 | 24.399 | 22.586 | 14.90 |

Streamed speedup over raw reads from the contended rates, limiting stage in
brackets, and resident slowdown; medians, then the worst case:

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | median | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.200 (decode) | 1.000 (decode) | 4.918 |
| FD3B avx2_vpshufb | stream | median | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.197 (decode) | 0.998 (decode) | 4.886 |
| FD4B ssse3_pshufb | hot | median | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 3.676 |
| FD3B avx2_vpshufb | hot | median | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.279 (decode) | 1.066 (decode) | 4.621 |

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.153 (decode) | 0.961 (decode) | 5.616 |
| FD3B avx2_vpshufb | stream | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.405 (decode) | 1.124 (decode) | 0.937 (decode) | 5.910 |
| FD4B ssse3_pshufb | hot | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.178 (decode) | 4.495 |
| FD3B avx2_vpshufb | hot | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.242 (decode) | 1.035 (decode) | 5.160 |

**2 threads** (the contended arm: decoder on one core, matmul on one). Rates:

| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | Matmul rest | Matmul + decode | Core-s/token (contended) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | median | 6.306 | 6.593 | 16.376 | 8.639 | 8.590 | 16.50 |
| FD4B ssse3_pshufb | stream | min | 5.512 | 5.744 | 15.882 | 8.333 | 7.977 | 18.94 |
| FD4B ssse3_pshufb | stream | max | 7.204 | 7.295 | 16.989 | 8.967 | 8.735 | 14.92 |
| FD3B avx2_vpshufb | stream | median | 6.882 | 6.263 | 16.665 | 8.822 | 8.480 | 17.37 |
| FD3B avx2_vpshufb | stream | min | 5.768 | 5.893 | 15.494 | 8.106 | 7.012 | 18.46 |
| FD3B avx2_vpshufb | stream | max | 7.455 | 7.225 | 17.277 | 9.078 | 9.275 | 15.06 |
| FD4B ssse3_pshufb | hot | median | 9.365 | 8.664 | 16.646 | 8.773 | 8.615 | 12.56 |
| FD4B ssse3_pshufb | hot | min | 8.129 | 7.619 | 16.031 | 8.509 | 8.093 | 14.28 |
| FD4B ssse3_pshufb | hot | max | 10.034 | 9.431 | 17.491 | 9.115 | 8.911 | 11.54 |
| FD3B avx2_vpshufb | hot | median | 8.131 | 7.998 | 16.896 | 8.841 | 8.709 | 13.60 |
| FD3B avx2_vpshufb | hot | min | 7.280 | 7.641 | 16.017 | 8.450 | 8.499 | 14.24 |
| FD3B avx2_vpshufb | hot | max | 8.815 | 8.403 | 17.614 | 9.278 | 8.832 | 12.95 |

Speedup and resident slowdown, medians, then the worst case:

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | median | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.319 (decode) | 1.099 (decode) | 2.484 |
| FD3B avx2_vpshufb | stream | median | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.253 (decode) | 1.044 (decode) | 2.661 |
| FD4B ssse3_pshufb | hot | median | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.932 |
| FD3B avx2_vpshufb | hot | median | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.333 (decode) | 2.112 |

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.149 (decode) | 0.957 (decode) | 2.958 |
| FD3B avx2_vpshufb | stream | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.179 (decode) | 0.982 (decode) | 2.932 |
| FD4B ssse3_pshufb | hot | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.270 (decode) | 2.296 |
| FD3B avx2_vpshufb | hot | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.273 (decode) | 2.305 |

**Decision, by the rule fixed above** (a streamed deployment at B is worth building
only if the contended worst-case speedup at B exceeds 1). Every `Matmul all` run
(at least 26.461 GB/s on 4 threads, 15.494 on 2) exceeds 6 GB/s, so raw streaming is
SSD-bound at every tabulated B, `t_raw = 1/B`, and the speedup
`(1/B) / max(r/B, 1/D_c, 1/M_c)` exceeds 1 exactly when `B < min(D_c, M_c)` (r < 1)
and is the full `1/r` while `B <= r * min(D_c, M_c)`. The last two columns below are
that arithmetic on the printed `min` rows, not new measurements.

| Threads | Format | Input | Worst case > 1 at B (GB/s) | Worst case <= 1 at B | Break-even B = min(D_c, M_c) | Full 1/r up to B = r min(D_c, M_c) |
|---:|---|---|---|---|---:|---:|
| 4 | FD4B | stream | 2.5, 3, 4, 5 | 6 (0.961, decode) | 5.767 | 4.327 |
| 4 | FD3B | stream | 2.5, 3, 4, 5 | 6 (0.937, decode) | 5.622 | 3.958 |
| 4 | FD4B | hot | 2.5, 3, 4, 5, 6 | none | 7.066 | 5.301 |
| 4 | FD3B | hot | 2.5, 3, 4, 5, 6 | none | 6.209 | 4.372 |
| 2 | FD4B | stream | 2.5, 3, 4, 5 | 6 (0.957, decode) | 5.744 | 4.309 |
| 2 | FD3B | stream | 2.5, 3, 4, 5 | 6 (0.982, decode) | 5.893 | 4.150 |
| 2 | FD4B | hot | 2.5, 3, 4, 5, 6 | none | 7.619 | 5.716 |
| 2 | FD3B | hot | 2.5, 3, 4, 5, 6 | none | 7.641 | 5.380 |

- **B up to 4 GB/s: the condition holds in every run**, including the one taken during
  background downloads (by the model above, a run whose worst case is 0.969 at B = 5 has
  min(D_c, M_c) = 4.85 GB/s and so gives 1.21 at B = 4; that run's raw report was not
  kept, so its B = 4 figure is back-derived from its B = 5 worst case rather than read
  from a recorded table),
  so a streamed FD4B or FD3B trunk is worth building there by this gate. **At B = 5 it
  is marginal:** all eight format x placement x thread-count cases pass in the idle runs
  (worst 1.124), but the run with background activity fell just below 1, so B = 5 is
  not established as a pass on this VM. In the idle runs decoding is fully hidden up to
  about 4.3 GB/s for FD4B and 4.0 for FD3B (streamed input, either thread count), where
  the modelled gain is the whole byte saving, 1.333 and 1.420; in the run with
  background activity that limit falls to about 3.6 and 3.5 GB/s. At B = 3 GB/s the
  whole saving holds with margin in every run.
- **B = 6 GB/s: the condition fails for streamed input** on 4 and on 2 threads, and
  the decoder is the limiting stage in every failing case: one decoder thread at its
  slowest (5.62 to 5.89 GB/s) is below 6. Cache-hot input passes on both (1.178 and
  1.035 on 4 threads, 1.270 and 1.273 on 2). Which placement a real row pipeline sees
  is not measured here; `stream` is the conservative bracket. A second decoder thread
  is a different design, not measured.
- **Margins at B = 5 are thin:** 1.153 (FD4B) and 1.124 (FD3B) for streamed input on
  4 threads, 1.149 and 1.179 on 2, and the run taken during background downloads fell to
  0.969 and 0.999.
- **What contention costs.** On 4 threads the medians show the decoder 10% to 20%
  slower beside the matmul and the 3-thread matmul 2% to 11% slower beside the
  decoder; the decoder's core itself leaves the matmul at 0.79 to 0.83 of its
  4-thread rate. On 2 threads the decoder moves by -5% to +9%, within the run spread,
  and the matmul loses 0.6% to 4%, but one matmul thread is 0.52 to 0.53 of two: on a
  small machine the cost is the core, not memory traffic.
- **Resident slowdown** is 4.50 to 5.91 in the worst case on 4 threads and 2.30 to
  2.96 on 2 (medians 3.68 to 4.92 and 1.93 to 2.66): a pinned layer should be decoded
  once and kept raw.
- Decode rates here (6.7 to 7.5 GB/s streamed, 7.8 to 9.8 hot, medians alone) are
  below gate 3's hosted single-kernel rates (16.22 to 16.33 GB/s, pooled 8 MiB,
  x86_64): a different CPU, a whole 12288 x 7168 matrix in 22 FDRX chunks and output
  to two 8 MiB buffers. The gap is not diagnosed here.

### What the shipped kernel changed

The tables first published in this section (head `516f010`, the same benchmark, the
previous `k3_matmul_bf16`) and these differ only in the matmul kernel. Its faster
widening and two-row passes move the matmul arms from 19.8 to 29.5 GB/s (4 threads,
all) and 15.5 to 23.4 (3 threads), and from 10.4 to 16.4 and 5.4 to 8.6 on 2 threads
(FD4B streamed medians). Three conclusions changed:

1. **On 2 threads the matmul is no longer the limiting stage.** At B = 6 the old
   tables failed in every case, matmul-limited (0.846 to 0.854 worst): the decoder's
   core left one matmul thread at about 5.1 GB/s. Now cache-hot input passes at 6,
   streamed input fails on the decoder instead, and the full byte saving holds to 4.1
   to 4.3 GB/s for streamed input instead of 3.6 to 3.8.
2. **The full byte saving holds to a slightly lower B** on 4 threads for streamed FD3B,
   about 4.0 GB/s instead of 4.2, because the contended decoder is a little slower
   beside a faster matmul; B <= 4 passes in every run, B = 5 passes only in the idle runs,
   and B = 6 still fails for streamed input.
3. **Resident decoding costs more,** 4.50 to 5.91 worst on 4 threads instead of 2.67
   to 3.62, since the raw matmul it is compared with got faster.

### Hosted legs

[CI run 35845709912](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35845709912),
head `cb4ab38`, the `contention` job's protocol: 1 s per arm, 5 repeats, every core,
recorded unchanged from the jobs' stdout; the driver's summaries recompute from the
runs.

**x86_64**, job 107131626491: `INTEL(R) XEON(R) PLATINUM 8573C`, 4 vCPUs, one-minute
load 0.16 before. Rates, then the worst case:

| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | Matmul rest | Matmul + decode | Core-s/token (contended) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | median | 11.726 | 6.450 | 36.990 | 26.942 | 24.998 | 16.87 |
| FD4B ssse3_pshufb | stream | min | 11.679 | 6.434 | 36.179 | 26.530 | 24.468 | 16.91 |
| FD4B ssse3_pshufb | stream | max | 11.752 | 6.654 | 37.057 | 27.558 | 25.042 | 16.35 |
| FD3B avx2_vpshufb | stream | median | 11.627 | 6.370 | 36.725 | 27.502 | 25.049 | 17.08 |
| FD3B avx2_vpshufb | stream | min | 11.439 | 6.142 | 36.151 | 27.326 | 24.827 | 17.72 |
| FD3B avx2_vpshufb | stream | max | 11.692 | 6.402 | 37.613 | 27.608 | 25.952 | 17.00 |
| FD4B ssse3_pshufb | hot | median | 16.061 | 9.229 | 36.277 | 27.287 | 22.426 | 11.79 |
| FD4B ssse3_pshufb | hot | min | 15.934 | 8.929 | 35.605 | 26.896 | 21.983 | 12.19 |
| FD4B ssse3_pshufb | hot | max | 16.198 | 9.313 | 36.930 | 27.398 | 22.466 | 11.68 |
| FD3B avx2_vpshufb | hot | median | 14.555 | 7.475 | 36.495 | 26.669 | 25.086 | 14.56 |
| FD3B avx2_vpshufb | hot | min | 14.533 | 7.426 | 35.569 | 24.498 | 23.441 | 14.65 |
| FD3B avx2_vpshufb | hot | max | 14.589 | 7.476 | 37.007 | 27.623 | 25.549 | 14.55 |

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B ssse3_pshufb | stream | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.287 (decode) | 1.072 (decode) | 5.759 |
| FD3B avx2_vpshufb | stream | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.228 (decode) | 1.024 (decode) | 6.124 |
| FD4B ssse3_pshufb | hot | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 4.136 |
| FD3B avx2_vpshufb | hot | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.238 (decode) | 4.983 |

**arm64**, job 107131626525: `macos-14`, 3 cores, NEON `tbl` for both formats. This
runner was not idle: its one-minute load average was 4.8 before the run and 12.6 after,
and the all-core matmul arm's runs spread from 21.2 to 47.9 GB/s, so these are the
least controlled numbers here. Rates, then the worst case:

| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | Matmul rest | Matmul + decode | Core-s/token (contended) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD4B neon_tbl | stream | median | 12.874 | 8.644 | 25.108 | 21.412 | 15.617 | 12.59 |
| FD4B neon_tbl | stream | min | 10.797 | 7.192 | 21.885 | 16.283 | 12.796 | 15.13 |
| FD4B neon_tbl | stream | max | 14.757 | 9.061 | 43.067 | 33.368 | 19.683 | 12.01 |
| FD3B neon_tbl | stream | median | 9.313 | 8.959 | 25.107 | 20.629 | 19.190 | 12.14 |
| FD3B neon_tbl | stream | min | 8.159 | 7.343 | 21.209 | 18.805 | 15.763 | 14.82 |
| FD3B neon_tbl | stream | max | 10.484 | 10.069 | 32.420 | 26.346 | 23.285 | 10.81 |
| FD4B neon_tbl | hot | median | 12.196 | 9.223 | 33.913 | 29.604 | 20.273 | 11.80 |
| FD4B neon_tbl | hot | min | 11.481 | 8.339 | 29.816 | 23.607 | 16.930 | 13.05 |
| FD4B neon_tbl | hot | max | 15.116 | 9.645 | 39.827 | 31.629 | 25.700 | 11.28 |
| FD3B neon_tbl | hot | median | 10.571 | 11.074 | 36.942 | 26.140 | 27.136 | 9.83 |
| FD3B neon_tbl | hot | min | 9.796 | 9.324 | 23.790 | 20.918 | 17.929 | 11.67 |
| FD3B neon_tbl | hot | max | 11.833 | 12.568 | 47.947 | 32.622 | 31.541 | 8.66 |

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B neon_tbl | stream | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.199 (decode) | 5.988 |
| FD3B neon_tbl | stream | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.224 (decode) | 4.415 |
| FD4B neon_tbl | hot | worst | 0.7503 | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 1.333 (ssd) | 4.776 |
| FD3B neon_tbl | hot | worst | 0.7041 | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 1.420 (ssd) | 5.142 |

On both hosted machines the worst-case streamed speedup exceeds 1 at every tabulated
B up to 6 GB/s, in both formats and placements: the x86_64 host's decoder runs at
11.4 to 16.2 GB/s alone and 6.1 to 9.3 beside the matmul, against 5.6 to 8.8 on the
VM. The worst-case resident slowdown is 4.1 to 6.1 (x86_64) and 4.4 to 6.0 (arm64).

Scope, as the reports' `scope` field says: synthetic weights with the committed
four-range high-byte distribution, one matrix shape, the SSD rate B a model
parameter with no disk in the loop, three machines, no full-model claim. An earlier
orientation reading taken while other builds loaded the VM (2026-09-22, no report
kept) is superseded by these tables.
