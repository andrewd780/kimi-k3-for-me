# Run with less local disk: remote weights and a bounded cache

This fork adds an **experimental, internet-dependent storage mode**. It preserves the
released weight bytes. It does not shrink the model, quantize it further, or establish
a new compression algorithm.

With the default 100 GB disk cache, plan for **about 210 GB of local storage**:
the upstream ~109 GB packed trunk, up to 100 GB of cached ranges, and metadata.
This is a capacity estimate from the existing trunk size and the configured cache cap,
not a measured full-model installation. Leave extra space for filesystem overhead,
saved sessions, and other files.

**Expect very slow generation on ordinary home internet.** The upstream expert
traffic estimate is ~25.8 GB per cold token before this mode's block overfetch.
At 100 Mbps, transferring 25.8 GB alone takes about 34 minutes; at 1 Gbps, about
3.4 minutes. Those are ideal transfer lower bounds, excluding compute, latency,
retries and extra downloaded block bytes. Cache reuse varies with routing and prompts.
Packing the trunk also downloads ~109 GB once.

## Setup

Use Linux, macOS, or WSL, Python 3.9+, and the existing C compiler/toolchain.
No extra Python packages, FUSE driver, HTTP C library, or GPU is required.
On a Mac, the ordinary engine build still needs Homebrew libomp.

Run these from the repository directory. Choose **new directories** for remote
metadata and trunk output. Do not reuse an existing full-checkpoint directory.

1. Build and run the tests:

   ```bash
   make -j
   make test
   python3 -m unittest discover -s tests -p test_remote_storage.py -v
   ```

2. Fetch the config, tokenizer, and shard headers only:

   ```bash
   python3 tools/remote_model.py prepare "$HOME/k3-remote"
   ```

   The default source is moonshotai/Kimi-K3. The tool resolves main to an immutable
   commit and records it in remote.json. Use --revision COMMIT to choose a specific
   snapshot. It supports public model downloads without login. It never executes
   Python files from the model repository.

3. Pack only the dense trunk, without downloading complete shards first:

   ```bash
   python3 tools/remote_model.py pack "$HOME/k3-remote" "$HOME/k3-remote-trunk"
   ```

   This takes roughly 109 GB of disk and network traffic. The exact output size is
   computed from the headers and checked against free space before copying.
   The packer bypasses the range cache, avoiding a duplicate trunk copy.

4. Start a short generation with a 100 GB **disk** cache:

   ```bash
   python3 tools/remote_model.py run --cache-gb 100 "$HOME/k3-remote" -- \
     ./bin/k3 "$HOME/k3-remote" --trunk "$HOME/k3-remote-trunk" \
     --preset laptop --tok "$HOME/k3-remote" \
     --prompt "The capital of France is" --gen 1 --incremental
   ```

   Start with one token because internet transfer time can be substantial.
   The laptop preset is a conservative RAM starting point, including for a 36 GB Mac.
   The engine's existing preset/auto options still control RAM. The wrapper's
   --cache-gb controls disk and must appear **before** the remote directory.
   Options after the directory are passed to the command being launched.

## Storage and failure behavior

- Metadata files contain **only headers**, not 1.56 TB sparse placeholders. Old
  binaries reject these files as truncated rather than treating missing weights as zero.
- C raw reads, aligned expert reads, and float-widening reads all use the same bridge.
  Ordinary local checkpoints retain their direct local I/O path.
- Each request includes the manifest identity. Accidentally connecting a directory
  to a daemon for another snapshot is rejected.
- The helper requires HTTP 206, an exact Content-Range and total file size, identity
  encoding, and exactly the requested payload length. A server returning a whole
  shard cannot silently start a terabyte download.
- Persistent 8 MiB blocks use LRU eviction. Concurrent misses for a block share one
  download. Four requests can fetch concurrently. Space is reserved and old blocks
  evicted **before** new files are written, including temporary fills.
- The configured cache cap includes each block's 32-byte digest and in-flight cache
  files. It excludes directory/allocation overhead and files outside range-cache.
  Startup enforces a reduced cap and removes interrupted temporary cache fills.
- SHA-256 protects cached blocks against subsequent corruption. This is **not**
  verification against a publisher-supplied per-range checksum: the initial payload
  still relies on the pinned HTTPS origin. Whole-shard checksum verification would
  require reading the full checkpoint.
- One process owns each cache directory. A second concurrent owner is rejected.
- A failed download or missing cache block fails the read; the engine never invents
  a replacement expert. Cached data persists after the wrapper exits.
- Use --offline before the remote directory to forbid downloads. A cache miss then
  fails immediately. This does not make a partial cache a complete offline model.
- Failed packing leaves trunk.bin.part, not a usable trunk manifest. To restart,
  choose another empty output directory or remove that partial output yourself.
  Packing does not yet resume midway through a layer or checkpoint.
- Interrupted prepare can leave an incomplete metadata directory. Use a new
  directory on retry; the completion marker is only written after all headers arrive.

An example cap of 20 GB needs about 130 GB total; 100 GB needs about 210 GB;
500 GB needs about 610 GB. Smaller caches tend to repeat more downloads. These are
storage budgets, not throughput promises or guarantees that a prompt will work offline.

## Validation and limits

The added fixture suite compiles the actual C reader, compares raw/aligned/widened
reads with local shard bytes, runs the existing streamed embedding/logit parity test,
and checks HTTP refusal, corrupt cache recovery, concurrent eviction, restart,
snapshot mismatch, missing daemon, and byte-identical trunk packing. It uses a local
HTTP server and synthetic/repository fixtures, never the released checkpoint.
GitHub Actions runs it on Linux and macOS.

The 19 fixture tests passed on Linux and macOS in
[the initial CI run](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/34696996154).
The editing environment itself could not execute C/Python, so validation ran on GitHub.
A separate Linux CI step probes a pinned public shard header and 4 KiB payload using
tools/probe_remote_source.py; it never downloads the full shard. Check the latest
workflow for that live probe's result. No full-model generation has been run.
Passing fixtures and a small network probe do not establish full-model generation
speed, answer quality, or long-term service availability.

The engineering contribution here is the integration into this fork: header-only
checkpoint setup, direct trunk extraction, and one bounded cache across the C read
paths. HTTP range loading and LRU caching are established techniques; no claim of
world-first novelty is made.

Upstream measurements and format context:
[README](../README.md), [architecture](ARCHITECTURE.md), and
[measurement data](data/README.md).
