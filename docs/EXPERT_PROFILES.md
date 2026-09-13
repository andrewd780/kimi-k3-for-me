# Calibrated expert residency

This fork adds **opt-in lazy pinning of calibrated experts**. Selected experts load on
their first normal request and stay resident; the remaining slots use LRU. Routing,
top-k, weight precision and arithmetic order are unchanged. It works with ordinary
local shards and the native offline compressed reader. No network is needed for either
local path. All experts remain available on disk: this does **not** solve the 1 TB
storage target.

There is no claimed new compression algorithm or world-first cache policy. This is an
experimental implementation and measurement pass for this fork. No full-checkpoint
benchmark was available for this change. Default memory presets are unchanged.

## Use a separate calibration workload

Build normally, or with `make ZSTD=1` for compressed local weights. Record a workload
representative of what you expect to run, using the same checkpoint and decode mode:

```bash
mkdir -p calibration
./bin/k3 /models/k3 --trunk /models/k3-trunk --preset laptop --incremental \
  --tok /models/k3 --prompt "The capital of France is" --gen 32 \
  --dump-cache-trace calibration
python3 tools/expert_profile.py build calibration/expert_trace.bin --out experts.profile
./bin/k3 /models/k3 --trunk /models/k3-trunk --preset laptop --incremental \
  --tok /models/k3 --prompt "The capital of Japan is" --gen 32 \
  --expert-profile experts.profile --pin-experts 8
```

`build` accepts multiple trace paths, so separate prompts can contribute to calibration.
Keep evaluation prompts out of that collection. Its geometry defaults to 93 layers,
896 experts and top-16; pass `--n-layers`, `--n-experts`, `--topk` for other configurations.
It runs offline with the Python standard library. Inference reads the profile in C.

Both inference flags are required together. The pin count is explicit: there is no
universally good fraction. The engine refuses counts that leave fewer than `topk+1`
evictable slots, incompatible geometry, missing selected experts, duplicate keys,
unsorted counts, overflow, or malformed lines. First-touch loads still cost I/O. An
unrequested hot expert occupies no slot. Installing a profile reads metadata only;
it does not preload weights. Profiles can be installed only before cache accesses.
Path, format and row count are checked during argument processing, before opening
the checkpoint. Model geometry, selected tensor availability and pin capacity are
rechecked after metadata indexing and before any weight binding. Profile geometry is
bounded to 4,096 layers, 65,536 experts and top-64, matching the producer.

Profile format is ASCII with a final newline on every line:

```text
K3EXPERTS 1 93 896 16
4 520 12
24 136 12
32 276 11
```

These rows are a format example, not a recommended calibration. After the version and
geometry header, rows contain zero-based layer, expert, and positive request count.
Rank by decreasing count, then increasing `(layer, expert)` to break ties. The engine
validates the entire file before applying the first N rows. Geometry is not a model
fingerprint; use calibration from the same checkpoint for useful predictions.

## What the existing trace actually records

`tests/fixtures/expert_trace.bin` has SHA-256
`d6a3d7a3353e9028e3a5f2bb74a9acfc792af396e8153839581822bdb2866879`.
Its 100,096 requests are eight full-recompute passes over prefixes of lengths
**5, 6, 7, 8, 9, 10, 11, 12**. That is 68 position evaluations but only **12 distinct
input positions**, not 68 new generated tokens. Every repeated layer prefix matches
the corresponding part of the final pass exactly.

At 455 slots, serial LRU replay of the original layer-major requests has 36,272 hits
(36.24%). Taking the final pass and transposing its requests into position-major order
gives **0 hits at those same 455 slots**. The original trace even gives 31.14% at 28
slots. The 36%-versus-0% comparison therefore cannot establish a budget-allocation bug.
The capacities differ, and the workloads differ too.

The new tool converts this layout only with `--layout legacy-prefixes`, after checking
every repeated prefix. The resulting sequence is a **derived token-major replay**, not
a new engine trace. Current chunk-union prefill fetches each expert once per chunk;
the legacy trace also predates that access pattern. A `get()` trace counts cache
requests, not every router selection in such a prefill.

The coverage curve still grows within this one context:

| Distinct positions | Distinct experts | Expert payload represented, decimal GB |
|---:|---:|---:|
| 1 | 1,472 | 25.83 |
| 5 | 5,682 | 99.70 |
| 8 | 7,922 | 139.01 |
| 12 | 10,010 | 175.65 |

This supplies no evidence that coverage plateaus across diverse prompts. It cannot
justify deleting the other 88% of experts. Prompt-diverse full-model traces are still
needed before evaluating any approximate subset model.

## Held-out deterministic replay

Train on the first 5 derived positions (7,360 requests, 5,682 distinct experts), test on
the last 7 (10,304 requests, 5,908 distinct experts). Hot keys are ranked from training
only. Test caches start empty and pay compulsory misses. Unused pins do not waste empty
capacity. This serial policy model excludes batch prefetch scheduling.

```bash
python3 tools/expert_profile.py evaluate tests/fixtures/expert_trace.bin \
  --layout legacy-prefixes --train-requests 7360 \
  --out docs/measurements/expert-profile-replay.json
```

Selected arms below; all fractions, all counts, source hash, coverage and projected
bytes are in [the replay report](measurements/expert-profile-replay.json). Capacities
are expert slots, not total process RAM. A real slot is 17,555,456 bytes including
alignment room; its expert payload is 17,547,264 bytes.

| Slots | Pins | Deterministic loads |
|---:|---:|---:|
| 28 | 0 | 10,304 |
| 28 | 11 | 10,259 |
| 615 | 0 | 10,304 |
| 615 | 598 | 9,793 |
| 1,344 | 0 | 10,304 |
| 1,344 | 1,327 | 9,289 |
| 3,647 | 0 | 6,102 |
| 3,647 | 1,823 | 6,125 |

The first comparison saves **0.44% of expert payload reads**; the 615-slot comparison
saves 4.96%, and the 1,344-slot comparison saves 9.85%. At 3,647 slots, half pinning
slightly increases reads. Re-executing this deterministic calculation adds no evidence;
the former `--runs` option and three identical results per arm have been removed.
Actual timed native/CLI comparisons still require three runs per arm. These replay
results are projections on seven held-out positions in the
same context, not SSD measurements or expected speedups on a laptop.

**Small-machine outcome: negative.** The historical 8 GB configuration has 28 slots;
45 avoided loads out of 10,304 is only **0.44% of expert reads**, an even smaller share
of total weight I/O. Static hot-set pinning is not being pursued further for small
machines. Larger-slot rows are not forecasts for an 8 GB laptop.

## Memory and I/O accounting

The 8.24 GB RSS result is already a working memory floor. Its non-expert memory is
mostly accounted for by allocations described in `src/cli/k3_run.c` and
`src/io/k3_trunk.c`:

| Component, ordinary path | Decimal GB | Basis |
|---|---:|---|
| Resident embedding, output table and small model parts | 4.698 | 163,840 vocabulary × 7,168 width, two BF16 tables, small FP32 parts |
| One streamed trunk ring slot | about 2.34 | Largest layer plus aligned-read padding |
| Recurrent state allocation | 0.626 | 93 × (96×128×128 + 3×96×128×3) FP32 elements |
| Safetensors index | about 0.078 | Existing full-scale estimate in `peak_rss_bytes` comments |
| KV cache, scratch, metadata, allocator and page rounding | varies | Context and actual reservations |

The first four terms explain about 7.74 GB before variable buffers. This is an allocation
audit, not a fresh exact decomposition of the historical process's peak RSS. Budgets
round down to slots, and RSS counts touched pages. The startup plan now includes saved
history and the actual larger incremental scratch requirement, which it previously
omitted. It remains a forecast; measured RSS is authoritative.

The ladder's `gb_read` column is **expert payload only**. With no trunk retention,
the documented 108.81 GB dense trunk adds to the 25.83 GB experts: approximately
134.64 GB of logical weight payload per decoded position. O_DIRECT padding, compression
and OS caching affect physical storage traffic. Expert pinning changes only the expert
part; a 5% expert reduction is not a 5% reduction in all traffic. Nothing in this
analysis warrants taking memory away from the trunk by default.

The engine now reports successful `expert_requests` and `expert_resident_reuses` in
JSON. A reuse is a `get()` served from a load already consumed at least once. The first
consumption of a prefetched load is not a reuse, even across a statistics reset.
`requests - evictions` was incorrect because it credited cold fills; raw hits credited
fresh prefetches. The step table now shows `REUSE %` and `EXPERT GB`. Byte counters
remain logical loaded payload, not compressed physical bytes. The ladder/split
harnesses read whole-run expert bytes from JSON and divide by emitted tokens, including
prefill, instead of parsing the final step's human-readable report.

Batch prefetch also protects requested experts already resident while selecting its
victims. A mixed batch must not evict and reload its own members. If a prefill union
exceeds available slots, prefetch stops reserving and normal demand reads finish it.
Membership is marked once per batch in a byte array (83,328 bytes for K3) and checked
in O(1) per victim candidate. Marks are cleared after reservation, including no-work
and capacity-limited batches; the LRU victim scan itself is still linear in slots.

## Native checks and remaining limits

The cache fixture tests exact bytes under pressure, lazy pins, pin limits, atomic
profile rejection, first-use accounting across resets, and mixed-residency batch
prefetch. In the adversarial batch, only one of four experts needs a payload read.

`tests/test_offline_cli.py` calibrates on synthetic prompt IDs `[1,2,3]` and evaluates
`[4,5,6]`, generating four tokens with six cache slots and two pins. It compares every
dumped first-step vocabulary logit byte and all generated IDs across ordinary,
pinned, and compressed-pinned runs, in both full-recompute and incremental modes.
All three runs per arm match exactly. [All native measurements](measurements/expert-profile-tiny.json)
include timings, which are not used to infer a speed improvement:

| Mode and arm | Expert bytes run 1 | Run 2 | Run 3 | Mean bytes |
|---|---:|---:|---:|---:|
| Full, ordinary | 1,031,424 | 1,031,424 | 1,031,424 | 1,031,424 |
| Full, pinned | 992,256 | 992,256 | 992,256 | 992,256 |
| Full, compressed pinned | 992,256 | 992,256 | 992,256 | 992,256 |
| Incremental, ordinary | 698,496 | 698,496 | 698,496 | 698,496 |
| Incremental, pinned | 678,912 | 678,912 | 678,912 | 678,912 |
| Incremental, compressed pinned | 678,912 | 678,912 | 678,912 | 678,912 |

These tiny weights establish implementation parity, not full K3 quality or performance.
The full 1.56 TB checkpoint, diverse calibration prompts, three full-model runs per
arm, and a thread-count sweep remain unmeasured here. Exact KDA vectorization is a
separate open item. Storage below 1 TB would require a different demonstrated lossless
result or a deliberate approximate-model project with quality evaluation.
