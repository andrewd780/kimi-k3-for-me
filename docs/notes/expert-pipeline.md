# Known-route expert pipelining

`--expert-pipeline` (also `K3_EXPERT_PIPELINE=1`) changes when the routed-expert batch
prefetch returns, not what it returns. Off by default; with it off the cache is the
code that shipped, byte for byte, stats included.

## The barrier it removes

`cache_getmany` reserves every slot of a top-k, reads every slot, and only then
returns. `k3_moe` then multiplies the first expert, but by construction that wait
was already over: routing picks the whole top-k before the first byte is read, and
`k3_moe` consumes it in a known, fixed order. There was never anything to gain by
waiting for the *slowest* of sixteen 17.55 MB reads before starting on the first.

## The design

Reads are handed to a small pthread pool (`K3_EXPERT_PIPELINE_THREADS`, default 4,
capped at 16) in **consumption order**, not disk-offset order. Disk-offset order is
the right choice when the caller waits for the whole batch anyway, turning scattered
seeks into a forward sweep; it is the wrong choice here, since it would place the
expert `k3_moe` multiplies first at an arbitrary point in the queue. Each worker
claims the next unclaimed entry off a shared cursor and reads it unlocked; publishing
a slot (`key_of`, `slot_of`, the LRU clock, the byte counters) happens under one
mutex, exactly the phase-3 bookkeeping the serial path already does, one entry at a
time. `get()` no longer waits for the batch: it checks residency, waits on a condvar
only if its own expert is still pending in the current batch, and otherwise falls
back to a synchronous miss read, same as pipelining was never on.

## Ownership rules

Workers exist only between a batch's launch and its `drain()`, which is the first
thing every batch-launching call does, so slot reservation and eviction always run on
the calling thread with no worker active. No disk read ever happens with the mutex
held. `drain()` also runs first in `free`, `reset_stats`, `pin`, `prefetch`,
`dump_trace`, `load_profile` and `report`, so nothing outside `getmany`/`get` can
observe a batch mid-flight.

## Stats in this mode

`load_seconds` is the wall clock from a batch's launch to its **last** completion,
recorded once by whichever worker finishes last. Summing each worker's own read
duration would count overlapped time multiple times and report a bandwidth the
device never delivered; the batch's elapsed wall time is what keeps
`bytes_read / load_seconds` an honest rate. Every other counter (hits, misses,
prefetch_reads, evictions) keeps exactly the meaning it has on the serial path.

## Failure path

A short read releases the slot's reservation (`key_of` back to empty) rather than
publishing it, so a truncated or failed read can never be served as a hit. The
expert that failed falls through to `get()`'s ordinary synchronous miss path on
demand; the rest of the batch is unaffected.

## Enabling it

`--expert-pipeline` on the CLI, or `K3_EXPERT_PIPELINE=1` directly (the flag just
sets the same environment variable before `k3_cache_init`). `K3_EXPERT_PIPELINE_THREADS`
sizes the pool. If thread creation fails outright, the cache logs once and falls back
to the old synchronous batch read — slower, not wrong.

## No speedup is claimed

The math bound from the earlier [streaming option map](streaming-options.md) is
**1.88x on the expert stage only, at equal read and compute time, never
whole-model** — and that bound assumed perfect overlap, which real thread scheduling,
mutex contention and NVMe queue behavior will not deliver exactly. This note ships
native tests and CLI parity, not a benchmark: `test_cache`'s pipeline cases and the
CLI's `--expert-pipeline` parity test run on tiny synthetic fixtures with no timing
assertion at all. The real number needs a checkpoint host, the same blocker every
other unmeasured entry in [STATUS.md](../STATUS.md) is waiting on.
