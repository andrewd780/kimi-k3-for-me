# Offline lossless weights and prompt rereading

The engine can read `.safetensors.k3z` and `trunk.bin.k3z` directly in C. Each archive
contains independently compressed Zstandard blocks. Only blocks covering the requested
weights are decoded into memory. Inference needs no internet, Python helper, socket,
FUSE mount or decompressed copy on disk. All released weight bits are preserved,
including the existing MXFP4 codes and scales. This does not recover precision that
was already removed when the released checkpoint was quantized.

**The full K3 checkpoint still does not fit under 1 TB with the tested codecs.**
The 1.56 TB checkpoint needs about 36% savings to reach 1 TB, before adding a packed
trunk, sessions or filesystem overhead. Sampled packed expert weights saved only
5.7%. If 1 TB is a hard total limit, do not start a full conversion expecting it to
fit. A smaller model or a change to the weights would require a separate choice and
quality evaluation; this implementation makes neither change.

## What was measured

[Recorded results](measurements/lossless-samples.json) contain 32 one-MiB slices at
pinned model revision `f831ab66814297da540d832a5235f8e904f29d06`: three packed expert
tensors and one dense BF16 tensor from each of eight shards spanning layers
1, 12, 24, 36, 48, 60, 72 and 92. Every codec result was decoded and compared byte for
byte. Results below are weighted by input bytes, using local libzstd 1.5.5.

| Sample class | Plain Zstd level 3 | Byte shuffle + Zstd level 3 | zlib level 1 |
|---|---:|---:|---:|
| Packed experts, 24 MiB | 94.263% retained | 94.261% retained | 94.887% retained |
| Dense BF16, 8 MiB | 78.231% retained | 70.944% retained | 80.146% retained |

These are stratified samples, not a full-checkpoint compression result. Dense samples
are `self_attn.f_a_proj.weight`; they do not represent every dense tensor. Expert
samples exclude scale tensors and metadata. The JSON records offsets, input SHA-256,
sizes and all three timing runs per codec. Those timings include Python/ctypes copies
on a shared Linux worker; they are not native engine throughput or laptop benchmarks.
The sample frames omit the archive's small identity/index overhead.

Reproduce the bounded experiment (this command downloads public samples):

```bash
python3 tools/bench_lossless.py --out lossless-samples.json
```

Zstandard and byte shuffling are established techniques. This repository already
investigated [compressed trunks](notes/compressed-trunk.md). This work adds native
random access for both shards and trunks, integrity checks and integration tests;
it does not claim a previously unknown compression algorithm or a world first.

## Build

Install the native library once. Inference can then run disconnected.

```bash
# Debian / Ubuntu
sudo apt-get install libzstd-dev

# macOS (OpenMP is also needed by the existing engine)
brew install zstd libomp
```

Use separate build directories because Make does not track changes in compiler flags:

```bash
make ZSTD=1 BUILD=build/zstd BIN=bin/zstd -j4
make ZSTD=1 BUILD=build/zstd BIN=bin/zstd test
```

For a nonstandard installation, set `ZSTD_CFLAGS` and `ZSTD_LIBS` for Make and
`K3_ZSTD_LIBRARY` for the Python converter. CMake also supports the option:

```bash
cmake -S . -B build/cmake-zstd -DK3_ZSTD=ON
cmake --build build/cmake-zstd -j4
ctest --test-dir build/cmake-zstd --output-on-failure
```

The default build still runs ordinary local checkpoints and requires no libzstd.
It reports a build instruction if asked to open compressed weights. Native compressed
reads are tested on Linux and macOS in the new workflow. The existing Windows build
remains covered without Zstandard; compressed Windows operation has not been validated.

## Convert a complete local checkpoint

`offline_model.py` itself makes no network requests. It requires complete local
shards, not the earlier remote mode's header-only files. The source is left intact.
Use a new destination and budget for **source plus compressed output** during
conversion; the source may reside on a separate drive. Conversion is sequential and
uses bounded memory, but reading and compressing terabytes can take many hours.

```bash
K3_SOURCE=/path/to/complete/checkpoint
K3_COMPACT=/path/to/new/compressed-checkpoint

python3 tools/offline_model.py model "$K3_SOURCE" "$K3_COMPACT"
```

`--max-output-gb N` enforces a decimal-GB output cap, including archive indexes,
metadata, completed files and the current partial file, excluding filesystem
allocation/directory overhead. It stops if the weights do not compress enough.
It is a guard, not a way to obtain an arbitrarily small model. For example,
`--max-output-gb 950` is expected to fail on the full K3 checkpoint. Leave room for
other files and filesystem overhead separately.

Defaults are 1 MiB blocks and Zstd level 3. `--shuffle auto` tries an exact permutation
of alternating bytes and uses it only when it saves at least another 2% on that block.
This usually helps BF16 more than packed experts. `--shuffle off` avoids that extra
conversion work and the runtime unshuffle loop. `--block-kib` accepts powers of two
from 64 through 8192; smaller blocks reduce over-read, while larger blocks reduce
index size. No setting is a guarantee of faster inference.

Each frame is decoded and compared during conversion. Files use a `.part` suffix
until complete and are published without replacing existing destinations. A failed
conversion keeps its completed archives and `.k3-incomplete` marker; the engine and
trunk packer refuse that directory. There is no automatic resume yet. Keep the source
and restart into a fresh destination after resolving the failure. File publication
requires a filesystem supporting same-directory hard links.

`offline.json` records original SHA-256 values and actual stored sizes. To check an
archive again, decode every block and compare the printed hash with that report:

```bash
python3 tools/offline_model.py verify "$K3_COMPACT/model-00002-of-000096.safetensors.k3z"
```

These archives are for this fork's reader. Other safetensors/Hugging Face tools will
not automatically understand the `.k3z` extension.

## Packed trunk and an offline run

The trunk packer accepts compressed shards. Its output is byte-identical to packing
the corresponding ordinary shards. It still creates an ordinary `trunk.bin` first,
which takes about 109 GB for the released model. Compressing that file temporarily
needs room for both versions, in addition to the checkpoint.

```bash
K3_TRUNK=/path/to/new/trunk-directory
python3 tools/pack_trunk.py "$K3_COMPACT" "$K3_TRUNK"
python3 tools/offline_model.py file "$K3_TRUNK/trunk.bin" "$K3_TRUNK/trunk.bin.k3z"
```

The engine prefers `trunk.bin` if present and falls back to `trunk.bin.k3z` when it is
absent. After conversion and verification, move the ordinary trunk to another drive
or remove your redundant copy if you want to use the compressed trunk. No automatic
deletion is performed. Keep a backup of the source weights until you are satisfied
with the converted checkpoint.

Run the C binary directly, without `remote_model.py run`:

```bash
./bin/zstd/k3 "$K3_COMPACT" --trunk "$K3_TRUNK" --preset laptop \
  --tok "$K3_COMPACT" --prompt-file prompt.txt --reread-prompt \
  --gen 1 --incremental
```

The existing expert RAM cache, trunk pinning and asynchronous trunk prefetch remain
in use. A compressed read adds roughly two block-sized temporary buffers per active
reader (about 2 MiB at the default), plus an immutable 16-byte index entry per block.
Whole boundary blocks are decoded even for a small slice. Compressed files use
buffered positioned reads; ordinary files retain the existing direct-I/O path.
Repeated reads can decode a block again; there is no additional decoded disk cache.

Reducing disk traffic introduces CPU decoding work. This may be slower on a fast
SSD, especially for expert bytes that shrink by only 5.7%. The engine's existing byte
counters describe logical weight bytes, not physical compressed traffic; their
MB/s figures are effective read/decode rates. No full-model tokens/sec, laptop speed,
or full-checkpoint peak memory measurement has been established for this mode.

## Reread the prompt before generation

`--reread-prompt` appends exactly one copy of the input token IDs before generation.
It works with `--ids`, `--prompt` and `--prompt-file`, with incremental and full
recompute modes. For example, input IDs `1,2,3` become `1,2,3,1,2,3`.
Text is tokenized once; copying IDs avoids merging tokens across the repeat boundary.

This gives the model another pass over the prompt. It is inspired by
[Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/abs/2512.14982),
but adherence improvements have not been evaluated on K3. It does not fact-check an
answer or guarantee instruction following. It doubles input context and adds prefill
work; a long prompt can be substantially slower. It is opt-in so existing prompts
and oracle tests retain their original semantics.

The original prompt must fit within 16,384 tokens when rereading, given this build's
32,768-token input ceiling. The repeated input, generation allowance and any saved
history are checked together before weights are loaded. On `--load-state`, only the
new request is repeated; saved history is not duplicated.

Output JSON includes the actual processed `prompt_ids`, `original_prompt_tokens`
and `reread_prompt`. Malformed or oversized raw-ID input is rejected instead of
hanging or silently truncating the request.

## Archive format and validation

All integers are little-endian. The archive starts with a 48-byte header:

| Offset | Type | Value |
|---:|---|---|
| 0 | 8 bytes | `K3ZSTD1` followed by NUL |
| 8 | uint64 | Original byte length |
| 16 | uint32 | Block size, power of two, 64 KiB to 8 MiB |
| 20 | uint32 | Block count, ceiling of length/block size, at most 2^22 |
| 24 | 16 bytes | Random archive identity |
| 40 | uint64 | Reserved, must be zero |

Next are `count` 16-byte index entries: uint64 physical offset, uint32 frame length,
uint32 transform flag (0 = unchanged, 1 = alternating-byte shuffle). Frames follow
contiguously in block order; gaps, overlaps, trailing bytes and impossible lengths
are rejected. Each frame is a standard Zstandard frame with a mandatory content
checksum. Its uncompressed content is a 32-byte prefix followed by the block payload:

`archive identity[16] | block number u64 | transform u32 | reserved zero u32 | payload`

The prefix binds identity, position and transform to the frame checksum. Swapping
otherwise valid frames or flipping the index's transform flag cannot silently change
the returned weights. For flag 1 the payload is `original[0::2] + original[1::2]`;
the last block can have an odd length. Readers require an exact decompressed length,
one frame per entry and matching prefix fields. Bounds checks precede allocations.

Checksums detect accidental damage, not malicious replacement with newly generated
valid archives. They are not publisher signatures or publisher-supplied hashes.
Untouched blocks are validated when read; `verify` checks all blocks. The native
reader returns a failed/short read on corruption; callers fail instead of substituting
missing expert bytes. Decoder state is private to each call and safe for concurrent
positioned reads.

Validation commands:

```bash
python3 -m unittest discover -s tests -p test_offline_storage.py -v
# Additional synthetic full-model tests require numpy and torch:
K3_TEST_BIN=bin/zstd/k3 python3 -m unittest discover -s tests -p test_offline_cli.py -v
```

The 16 storage tests cover random/concurrent C reads, exact raw/aligned/widened
weights, odd lengths, corrupted frames/indexes, frame swaps, missing codec support,
output caps, partial conversion rejection and streamed embedding/logit parity.
Seven CLI test groups compare bit-identical logits and generated IDs on a complete
synthetic 13-layer model, including compressed trunks, streaming, prompt repetition,
UTF-8 input, context limits and saved-session continuation. These tests validate the
implementation; they do not establish full K3 quality or performance.
