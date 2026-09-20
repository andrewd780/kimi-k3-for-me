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
identity. Gates 2 and 3 passed in hosted CI at head `9f0c07f`, recorded below;
gates 4 and 5 and the deployment gates remain.

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

1. The 4-bit versus 5-bit ratio-versus-rate curve, now that the 4-bit kernel has
   a rate.
2. Row-boundary padding and offset-index costs, measured separately. Fixed-width
   indexes alone do not make a variable-length escape stream row-seekable;
   independent row escape offsets or another explicitly costed layout is needed.
3. Decode while sharing cores with the matmuls on a core count like the target
   machines, and a supported container/reader. Kernel speed alone does not make
   deployment profitable, as the Huffman note already says.

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

No full-model speedup, full-model storage number or ratio win over Huffman is
claimed. The streamed-byte reduction is budget-dependent: pinned trunk bytes
are not reread every token.
