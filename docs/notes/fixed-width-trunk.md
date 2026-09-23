# Fixed-width trunk dictionary: falsifier first

2026-09-20. Predictive prefetch remains closed. This separate research item
targets the BF16 trunk, 80.8% of the stated one-position streamed-byte baseline.
All experimental execution is hosted CI; no checkpoint host or rental is used.

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
passed. [The complete counts and identities](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/35497304209/artifacts/10601201984)
are preserved; the pooled global-dictionary coverage is **99.954653%**.

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
dozen at 99.9%" becomes 14 values pooled; 15 entries comfortably clear 99%.

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
Gate 4 (the bit-width curve) and gate 5 (row-boundary cost) now have results from
committed counts and exact shapes; the every-family samples, decode under matmul
contention and the deployment gates remain ([status](#remaining-gates)).

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
real-range runs (1.100 x86_64, 1.350 arm64,
[results](research-results.md#2-compact-huffman-decoder-research-prototype))
that is a 13x–15x kernel-rate difference at a 6.1-point ratio cost. The arm64
spread on identical input (16.4–26.7 GB/s) is the shared three-core hosted
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
| 4. Bit-width curve | **Result on committed counts** | On the four committed `f_a_proj` ranges (CI run 35462000448): 3 bits retains 0.704128, 4 bits 0.750153, 5 bits 0.8125; 3 bits gains 4.60 points over 4, above the 1.5-point prototype bar. FD3B built, byte-exact under local ASan/UBSan (scalar, SSSE3, AVX2). | Eight-range and whole-BF16 figures (histogram CI job); FD3B hosted rates |
| 5. Row-boundary cost | **Exact, from released shapes** | FDRX: zero padding at every K3 width; 0.0340 points per-row index, 0.0148 grouped; three range reads per chunk | Nothing for the cost; a container still needs a per-chunk checksum |
| Every tensor family | **Tooling and CI job built** | Gate 1 sampled 2 of 23 families, 4.00% of trunk bytes; inventory exact | The `families` CI job's samples and the [per-family plan](#per-family-plan) decisions |
| Decode under matmul contention | **Benchmark and model built; not measured** | Byte-exact benchmark on the engine's `k3_matmul_bf16`, break-even arithmetic unit-tested | The quiet-machine measurement ([placeholder](#decode-under-matmul-contention)); hosted x86 and arm64 legs |
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
KDA `f_a_proj` only.

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

**Real numbers now, from committed counts.** The eight-range pooled histogram is
not committed (it is in the gate-1 artifact). The only committed high-byte counts
of real K3 bytes are the Huffman research job's pool of four of those ranges,
KDA `f_a_proj` at layers 1, 12, 24 and 36:
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
| Order-0 entropy, whole BF16 | | | | awaits CI | | |

Here the tables are this subset's own ranking. With the gate-1 dictionary and its
first seven entries instead, the same bytes give 0.750257 (1,077 escapes) and
0.704132 (69,758 escapes): 4.61 points. Five bits buys nothing: the support is
only 27 values here (29 pooled over eight ranges) and 4 bits already covers
99.97%, so a 5-bit index pays a full bit per value for 0.03% of escapes. The only
direction that clears the 1.5-point bar is **down**, to 3 bits, where seven entries
hold 96.67% and the 3.33% of escapes cost less than the saved bit. The sign is
close to a fair coin, so a (k-1)-bit magnitude table covers `2^k - 2` signed values
in pairs while the k-bit table covers the same pairs plus one more value: the sign
split trails at every width, by 0.99 points at 3 bits, where that extra entry
holds 1.99% of values. Huffman is a further 3.21 points below 3-bit, and the
high-byte entropy only 0.17 below Huffman. Beyond that only the low byte is left:
an order-0 coder of whole values can gain at most `8 - H(low | high)` bits per
value over these bounds, which the CI job now measures.

**Scope.** These are KDA `f_a_proj` bytes only, 0.116% of the trunk. The CI
histogram job computes the eight-range curve; the `families` job computes it for
every family (next section).

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

The new `families` CI job (`tools/trunk_family_gate.py sample`) reads every shard
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
takes a committed record of them and then requires the same bytes. Results
await CI; unit tests cover the classification (MLA versus KDA `o_proj`/`g_proj`
by layer), the plan, the exact weighting and the per-family STOP.

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
   bound says how much any model of the low byte could still add.
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

## FD3B: the 3-bit prototype

On the committed counts the curve clears the 1.5-point bar at 3 bits, so
`benchmarks/fixed_dictionary.h`
now has FD3B beside FD4B: 3-bit codes as a little-endian bitstream (code i in
bits 3i..3i+2), code 7 escapes to a raw byte, then the raw low bytes, the escape
bytes and 32 zero bytes of read slack. Escapes are 3.3% of values here, so the
FD4B approach of patching each escape in a branchy loop would mispredict; FD3B
computes an exclusive prefix count of the escape lanes and shuffles the next
escape bytes into place with no branch. Decoders: scalar reference, SSSE3,
AVX2 (both halves of 32 values in one register) and NEON; `bench_fixed_dictionary`
dispatches on the magic. `tools/bench_fixed_dictionary.py --index-bits 3` is an
independent encoder, and CI runs FD3B through the same exactness (ASan/UBSan,
both ISAs, all eight ranges, golden bytes and negative controls) and rate jobs
as FD4B.

Tests: every length 0..131 and 8193, an independently hand-packed golden stream,
all byte values, no and all escapes, escape rates 0.1% to 90% and bursts of 1
to 40, truncation at every length, header, table, padding-bit and slack
corruption, escape underflow and surplus, and payload corruption caught by the
byte comparison. They pass here for the scalar, SSSE3 and AVX2 paths under
ASan/UBSan with gcc 13.3 and at -O3 with clang 18. The NEON path is only
compile-checked here (clang, aarch64 target); it runs in the macos-14 jobs.

Kernel rates here (Intel Xeon 2.8 GHz Cascade Lake VM, 4 vCPU, shared with other
builds; synthetic bytes drawn from the committed four-range histogram; `bench_huf4`
units, reconstructed GB/s, median and minimum of 15 runs interleaved with the
other arms, load average 5.7):

| Input | FD4B `ssse3_pshufb` | FD3B `ssse3_pshufb` | FD3B `avx2_vpshufb` |
|---|---:|---:|---:|
| 1 MiB | 15.41 (min 13.75) | 6.37 (min 6.13) | 9.52 (min 8.06) |
| 8 MiB | 12.61 (min 10.09) | 5.86 (min 5.26) | 8.49 (min 7.45) |

With the escape patch removed (wrong output, timing only), the SSSE3 FD3B loop
runs at FD4B's speed (about 12 GB/s on 8 MiB): the escape expansion costs it half
its rate, which is what the AVX2 path wins back in part.
The hosted rates and the arm64 NEON rate await CI.

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
show only the independent byte comparison catches it. A deployment container
needs a checksum per chunk. Tests (all under ASan/UBSan): every `[a, b)` of a 13 x
64 matrix in FD4B and FD3B and an 11 x 128 FD3B matrix, G = 1, 2, 3, 7, R and
R + 5; 40 x 512 (both widths) and 6 x 7168 (both widths) matrices with ranges
starting and ending inside groups mid-matrix;
empty ranges, reversed and out-of-range bounds; structural corruption refused at
parse; an off-by-one checkpoint caught. The contention benchmark below also
decodes a whole 12288 x 7168 matrix through a per-row FDRX index in 22 chunks of
up to 585 rows, byte-exact before and after timing.

## Decode under matmul contention

> **PLACEHOLDER: this measurement has not been taken.** The benchmark, the
> break-even arithmetic and the protocol below are committed and tested; the
> tables are empty until the run on a quiet machine fills them. No contention
> number in this note is a result.

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
cold row pipeline) or `hot` (one cache-resident chunk, as right behind the read
that landed it). Every chunk is compared byte for byte with the raw matrix through
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

**Decision rules, fixed before the data.** A streamed deployment at SSD rate B is
worth building only if the contended **minimum** speedup at that B exceeds 1; the
median is reported beside it. The resident slowdown is expected to exceed 1: a
pinned trunk layer is read once, so it should be decoded once when pinned and
kept raw, and this measurement prices that choice rather than gating it. The
machine here has 4 cores, where the decoder's core is a quarter of the matmul's;
the 2-thread run bounds a smaller machine, and larger machines lose a smaller
share.

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

The driver prints the three tables below in this form; paste them with the CPU
model, load averages and head from the report's `execution` field. The hosted
x86_64 and arm64 legs come from the `contention` CI job (`decode-contention-*`
artifacts, 1 s per arm, 5 repeats) and are recorded the same way.

**Rates, reconstructed or consumed BF16 GB/s (PLACEHOLDER, not measured):**

| Format | Input | Stat | Decode alone | Decode + matmul | Matmul all | Matmul rest | Matmul + decode | Core-s/token (contended) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD4B `ssse3_pshufb` | stream | median / min | pending | pending | pending | pending | pending | pending |
| FD3B `avx2_vpshufb` | stream | median / min | pending | pending | pending | pending | pending | pending |
| FD4B `ssse3_pshufb` | hot | median / min | pending | pending | pending | pending | pending | pending |
| FD3B `avx2_vpshufb` | hot | median / min | pending | pending | pending | pending | pending | pending |

**Streamed speedup over raw reads, contended rates, limiting stage in brackets;
resident slowdown (PLACEHOLDER, not measured):**

| Format | Input | Stat | r | B = 2.5 GB/s | B = 3 GB/s | B = 4 GB/s | B = 5 GB/s | B = 6 GB/s | Resident slowdown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD4B | stream, hot | median, min | 0.7503 | pending | pending | pending | pending | pending | pending |
| FD3B | stream, hot | median, min | 0.7041 | pending | pending | pending | pending | pending | pending |

For orientation only, and **not admissible as a result**: one FD4B `stream` run
on 2026-09-22, on this 4-vCPU VM while another agent's builds held the load
average near 5 (no report file kept), gave decode 6.48 GB/s alone and 6.46 with
the matmul running; the matmul gave 19.6 on four threads, 15.1 on three and 14.8
on three against the decoder. The quiet run replaces these.
