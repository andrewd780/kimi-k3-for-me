# Independent lm_head streaming

`--stream-lm-head` uses the existing exact BF16/F32 chunk reader for the output
projection while retaining the normal embedding and recurrent-state policies.
`--ultra-low-memory` still selects streaming for both model tables and its separate
state-reuse policy. No tensor format or model weights change.

The released BF16 lm_head is 2,348,810,240 bytes. Streaming replaces that resident
table with a 4,202,496-byte I/O buffer, releasing 2,344,607,744 bytes. It also reads
the lm_head again for every forward that projects logits: once per decode step and once
per `--spec` verify sweep, whose positions are projected together from each chunk read
(`k3_model_stream_project_batch`), and once per block of up to 16 positions for
`--tf-check` and `--score-prompt`, which need every position's logits. Each position's
logits are bit-identical to projecting it alone. Those are geometry and mechanism
calculations, not measured full-model memory or throughput results. The batched
projection's compute alone, with the head resident and no reads, is timed at the
head's shape in [the batched kernel timing](research-results.md#batched-kernel-timing):
on four threads a `--spec 8` sweep takes 383 ms instead of 730 ms and a 16-position
block 581 ms instead of 1,319 ms, against the one-position kernel this branch ships.

Fixed presets and explicit `--trunk-gb` values stay explicit. The flag alone frees
memory; to fund the second trunk slot, give that memory to the trunk budget. For
example, `--preset laptop --stream-lm-head --trunk-gb 4.84` illustrates the budget
trade. It is not a verified full-checkpoint 8 GB configuration. `--preset auto`
does include the smaller model-table reserve in its existing K3 budget estimate.
Streaming with `--draft-trunk` is refused; ordinary incremental and n-gram
speculative decoding continue to use the same exact model.

## Mechanism gate

The Linux offline CI job builds the existing tiny synthetic checkpoint, enlarges
only the two vocabulary tables by repeating existing BF16 rows, and compares:

- Resident lm_head with a trunk budget that selects one ring slot.
- Streamed lm_head with precisely its net released bytes added to that budget.

Both native runs execute inside separate systemd cgroup v2 units with the same
64 MiB `MemoryMax` and `MemorySwapMax=0`. The child checks the actual cgroup files
before starting the engine and records peak charged memory and OOM events. The
test refuses an uncapped substitute. It requires the same planned memory total,
ring sizes 1 and 2 respectively, all layers completed, no expert drops, identical
generated IDs, and byte-identical full-vocabulary logits.

This is a scaled mechanism check. It does not allocate the real 2.35 GB lm_head,
establish the real model's 8 GB RSS, measure overlap benefit or measure s/token.
The exported mechanism artifact deliberately omits all fixture timings. Full-model
timing remains blocked until a permitted host has the entire checkpoint locally.

The gate passed in [CI run 34783226202](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/34783226202).
The toy checkpoint was 12,680,201 bytes, with a 4,501,504-byte lm_head. Replacing
that table with the stream buffer made 299,008 bytes available for the trunk.

| Observed quantity | Resident head | Streamed head |
| --- | ---: | ---: |
| Total planned bytes | 9,602,560 | 9,602,560 |
| Trunk budget bytes | 299,008 | 598,016 |
| Trunk slot bytes | 245,760 | 245,760 |
| Ring slots | 1 | 2 |
| Cgroup peak charged bytes | 17,686,528 | 17,948,672 |
| Cgroup limit bytes | 67,108,864 | 67,108,864 |
| OOM events | 0 | 0 |

The cap includes runtime overhead beyond the allocation plan; this does not claim
that the plan equals RSS or that streaming lowers total RSS after reinvesting the
freed memory. Full-vocabulary logits matched byte for byte and both runs generated
`[83, 31]`. The [raw mechanism record](../measurements/lm-head-mechanism.json)
includes the logit hash, actual cgroup paths and memory events.

The ordinary tiny CLI tests also check resident-versus-streamed logits for full
recompute and incremental decode, including a compressed trunk and compressed
checkpoint. Machine-readable run JSON now includes `lm_head_streamed`,
`trunk_ring_slots`, slot and budget bytes, resident model bytes, stream-buffer
bytes and the planned memory total.
