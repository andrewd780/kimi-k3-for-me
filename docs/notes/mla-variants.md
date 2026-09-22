# MLA KV-cache variants: what each one costs, and what absorbing `kv_b` changes

**A benchmark and a numerical study. Nothing in the engine changed.** Four ways to run
one MLA layer's cached attention, plus one bit-exact restructuring added as a fair
baseline, are implemented side by side in `benchmarks/mla_variants.h`. They are held to
the engine bit for bit by `tests/unit/test_mla_variants.c`, which is part of `make test`,
and measured by `benchmarks/bench_mla.c`. `benchmarks/mla-study.sh` reproduces every
number below, and the raw JSON lines are in
[`docs/measurements/mla-variants-x86_64.jsonl`](../measurements/mla-variants-x86_64.jsonl).

PENDING
