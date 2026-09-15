# Selective, offline scale compression

`offline_model.py --policy scales` writes a native **K3ZMAP1** archive. Only
paired E8M0 expert scale tensors are candidates for Zstandard compression.
Packed MXFP4 weights, BF16 trunk tensors, embeddings, headers and all other bytes
are copied verbatim. The C reader issues positioned reads directly into the
destination for those raw extents and never sends them through a decoder. Into an
aligned destination, the aligned interior of a raw extent bypasses the page cache
through the same direct descriptor the plain shards use (`O_DIRECT` on Linux,
`F_NOCACHE` on Darwin, unbuffered on Windows).
An incompressible scale block also falls back to raw storage.

This is an opt-in storage mechanism. Full-checkpoint conversion and model
throughput have not been measured. Existing ordinary weights and K3ZSTD1
archives retain their behavior. This is established selective compression and
extent mapping applied to this engine, not a claim of a new coding theorem.

## Why separate the planes?

For each group of 32 MXFP4 weights there are 16 packed bytes and one scale byte.
The [scale campaign](notes/scale-plane.md) measured a 0.155315 retained fraction
on matched scale samples, while packed weights retained 0.942584. Compressing
the scales alone therefore projects to `(16 + 0.155315)/17 = 0.950313` of expert
payload, about **4.97% fewer expert bytes**. It avoids decoding the other 94.12%
of the logical expert data. This distinction matters on a small CPU.

The sample-based 85 GB scale-plane projection still saves about 71.8 GB across
the checkpoint. Exact tensor boundaries avoid decoding packed bytes merely
because they share a fixed-size compression block with a scale tensor.
No full-checkpoint size result follows from the samples. Codec prefixes,
extent indexes and different block boundaries add overhead.

## Conversion and inference

On a permitted host with the complete local checkpoint, build `make ZSTD=1`
as described in [offline storage](OFFLINE_STORAGE.md). Then:

```sh
python3 tools/offline_model.py model /path/to/checkpoint /path/to/new-scale-archive \
  --policy scales --block-kib 1024
python3 tools/offline_model.py verify /path/to/new-scale-archive/model-00002-of-000096.safetensors.k3z
```

Compare the printed whole-file SHA-256 with that file's entry in `offline.json`.
`verify` computes the hash; it does not discover a publisher hash automatically.
The `file` subcommand also accepts `--policy scales` for a single safetensors
shard. It rejects an ordinary `trunk.bin`, which has no tensor header. `--shuffle`
is ignored by the scales policy; no permutation is applied to either plane.

Use the archive directory as the ordinary checkpoint argument to the native C
engine. A packed trunk remains separate and can be read plain or in the existing
K3ZSTD1 format. The trunk packer accepts selective archives and produces the same
trunk bytes. `--stream-lm-head` and `--reread-prompt` work with this format.
Inference makes no network requests and needs no Python process or decompressed
disk copy. Conversion keeps the source intact, requires a new output destination,
honors `--max-output-gb`, and retains the existing incomplete-directory guard.
Source plus destination must fit during conversion.

## Memory and scheduling limits

A read touching only raw extents allocates **zero decoder scratch** and reads
only the requested byte ranges. A read touching scales allocates at most one
`block_bytes + 32` output buffer and one `ZSTD_compressBound(block_bytes + 32)`
input buffer, reused until that call finishes. Libzstd's own working memory is
additional. Each concurrent reader owns its buffers; no decoder state is shared.
A large raw extent does not cause a whole-extent allocation.

The immutable native index costs `sizeof(K3ZExtent)` per extent (32 bytes on
the tested 64-bit ABIs). Three scale tensors plus three intervening raw runs
per expert suggest roughly 16 MB for 82,432 experts, before other extents.
This is geometry arithmetic, not full-checkpoint RSS. Conversion and Python
metadata objects use additional memory bounded by the shard header/index.

**Raw extents use direct I/O for their aligned interior.** The writer pads each
raw extent so that its physical offset equals its logical offset modulo 4,096, and
the shard reader widens every read to 4,096-byte boundaries into an aligned slot,
so the interior of a raw extent is read through the same direct descriptor as a
plain shard. The head and tail of each extent (under 4 KiB each) and every
compressed scale frame still go through the page cache, and a filesystem that
refuses direct I/O falls back to buffered reads of the whole span. Extra system
calls at extent boundaries and serial scale decoding can still outweigh byte
savings. In particular,
for a scale compression ratio `r`, device rate `B` and decode rate `D`, serial
read-plus-decode beats raw reading only if `D > B/(1-r)`, assuming equal device
rates and ignoring other overhead. At 3 GB/s and `r=0.155315`, that threshold is
3.55 GB/s. The earlier sample decoder did not establish that rate.

Overlapping different reads/compute can change the critical path. This reader
does not introduce a new asynchronous scheduler; it uses the existing concurrent
expert batch reader and trunk prefetcher. No token-speed improvement is claimed.

## Format and corruption handling

All integers are little-endian. The 48-byte header reuses the K3ZSTD1 layout:

| Offset | Bytes | Meaning |
| --- | ---: | --- |
| 0 | 8 | `K3ZMAP1` followed by NUL |
| 8 | 8 | Original logical file size, at most INT64_MAX |
| 16 | 4 | Maximum decoded scale block, power of two, 64 KiB through 8 MiB |
| 20 | 4 | Extent count, at most 2^22 |
| 24 | 16 | Random archive identity |
| 40 | 8 | FNV-1a-64 of header bytes 0..39 followed by the complete index |

Each 32-byte index entry stores `logical_end:u64, physical_offset:u64,
stored_size:u64, kind:u32, reserved_zero:u32`. Extents must cover the complete
logical file exactly once, with strictly increasing ends. Stored extents follow
the index in order, without overlaps or trailing bytes; up to 4,095 zero bytes of
padding may precede an extent, and the writer uses that before every raw extent so
that its physical offset equals its logical offset modulo 4,096. A gap of 4,096 or
more is refused. Kind 2 is raw
and its stored length must equal its logical length. Kind 0 is one checksummed
Zstandard frame; its logical length must fit the block limit. Other kinds fail.

A compressed frame expands to the existing 32-byte prefix followed by scale
bytes: `identity[16], extent_index:u64, zero:u32, zero:u32`. The reader validates
the checksum, identity, position and exact output length before returning those
bytes. This catches accidental damage and misplaced otherwise-valid frames.

**Raw payload has no per-read checksum**, matching ordinary safetensors reads.
Use the full logical SHA-256 in the conversion report to detect raw corruption.
The index checksum detects accidental metadata damage; neither it nor a locally
generated manifest authenticates the publisher. No stronger guarantee is implied.

CI tests cover concurrent ranges, boundary/odd-length reads, all 256 scale values,
incompressible fallback, index bounds, truncated payloads, damaged compressed
frames, valid frames with wrong identity/position, conversion caps, native tensor
and trunk parity, and full synthetic CLI logits. Decoder failure injection proves
that raw-only reads bypass the decoder. The native test also checks that the
aligned interior of every raw extent goes through the direct descriptor when one is
present, byte for byte, and that a descriptor which cannot serve a read falls back
to buffered reads. Real K3 quality and throughput remain unmeasured.
