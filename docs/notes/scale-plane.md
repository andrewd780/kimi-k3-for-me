# E8M0 scales: the entropy measurement

The sampled scale plane is highly compressible. Across **768 matched scale
tensors / 264,241,152 bytes**, pooled byte entropy is **0.713913 bits per byte**;
the byte-weighted mean within tensors is **0.683339 bits per byte**. Zstd level 3
retained **15.5315%** of the scale bytes, with checksummed, byte-exact round trips
for every sample. The most common byte is 121 (223,749,731 observations); 120 is
next (36,977,770). Entropy is a zero-order coding bound, not a measured codec ratio.

If the Zstd sample ratio generalizes to an 85 GB full scale plane, it projects to
**13.20 GB**, saving **71.80 GB**. This is an extrapolation, not a complete-checkpoint
compression result. It does not make a 1.56 TB checkpoint fit under 200 GB, and no
new deployment codec or format is added by this measurement.

A later [selective reader](../SELECTIVE_SCALES.md) implements scale-only archives
with raw packed-weight extents. Its synthetic correctness gates do not change
the measurement's scope or turn the size projection into a full-checkpoint result.

## What was actually read

The [CI campaign](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/34781927101)
used immutable Hugging Face revision
`f831ab66814297da540d832a5235f8e904f29d06` of `moonshotai/Kimi-K3`.
Eight stratified shards each supplied 32 complete experts, with all `w1/w2/w3`
matrices: 256 experts total. Each central 1 MiB packed-weight sample was paired
with the **entire scale tensor of that same matrix**. Eight 1 MiB BF16 samples
were retained as the trunk comparison. Total tensor payload was 1,077,936,128
bytes (1.078 GB), including 805 MB packed weights and 264 MB scales; bounded
metadata reads are additional. No full shard or checkpoint was downloaded.

| Sample class | Tensors | Pooled bits/byte | Zstd retained bytes |
| --- | ---: | ---: | ---: |
| Packed expert nibbles | 768 | 7.506931 | 94.2584% |
| Matched E8M0 scales | 768 | 0.713913 | 15.5315% |
| BF16 trunk | 8 | 6.236738 | 78.2315% |

The packed-weight result reproduces the earlier near-incompressibility result.
These classes must not be merged as though their sample proportions represented
the checkpoint: the scales were deliberately oversampled relative to nibble bytes.

Every codec sample has three serial decode timings. Aggregating the scale Zstd
payload over summed per-sample decode time gives **660.8, 679.7, 694.6 MB/s** in
the three passes. These include ctypes allocation/check overhead on a shared CI
CPU. They do not establish the speed of a native integrated reader, cache behavior,
or full-model token throughput. A scale-only reader still needs its own gate.

Full per-sample offsets, shapes, byte counts, SHA-256 hashes, entropy and codec
timings are in [`scale-plane-samples.jsonl`](../measurements/scale-plane-samples.jsonl).
Aggregates, histograms, provenance and the projection are in
[`scale-plane-summary.json`](../measurements/scale-plane-summary.json).
The original CI artifact is also available from the linked run.

Reproduce on a permitted cloud/CI host with:

```sh
python3 tools/bench_lossless.py --experts-per-shard 32 \
  --max-sample-bytes 1125000000 --out scale-plane.json
```

The default invocation remains a smaller paired sample. The larger CI campaign
requires an explicit workflow dispatch or `[measure-scales]` in the head commit
subject, so later edits to the PR do not silently download another gigabyte.
The sampler validates tensor pairing and shapes and checks the complete planned
payload against the cap before any tensor reads. HTTP 200/full-file responses,
wrong ranges, size changes and truncated data are rejected.
