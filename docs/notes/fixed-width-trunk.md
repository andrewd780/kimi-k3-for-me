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

Results are pending the first CI run. No codec or performance result is claimed.

## Remaining gates, only if gate 1 passes

1. All-eight-range byte-exact round trips, sanitizers, escape and odd-tail paths,
   with failure controls for each gate.
2. Reconstructed BF16 bytes / elapsed seconds / 1e9, three runs per arm on x86
   and hosted ARM. Hard floor 3 GB/s on every run, target 4 GB/s. Include plane
   assembly and all escape work, as `bench_huf4` includes reconstruction.
3. Only after those gates, the 4-bit/5-bit ratio-versus-rate curve.
4. Row-boundary padding and offset-index costs, measured separately. Fixed-width
   indexes alone do not make a variable-length escape stream row-seekable;
   independent row escape offsets or another explicitly costed layout is needed.

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
