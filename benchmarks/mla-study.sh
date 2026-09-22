#!/usr/bin/env bash
# The full MLA KV-cache variant study behind docs/notes/mla-variants.md.
#
#   benchmarks/mla-study.sh <out_dir> [threads]
#
# Steps, strictly one after another (two benchmarks at once would time each other):
#   1. the bitwise test: E+, L0 and L1 against the engine's own k3_mla_cached;
#   2. the numerical study of the absorbed variant A at 1,024 and 16,384 cached
#      positions, and a sharper-softmax stress run (queries scaled by 6) at 4,096;
#   3. timing on one thread, then on [threads] (default: every online CPU), at 256, 1,024,
#      4,096, 16,384 and 65,536 cached positions, T = 1 and 5.
#
# One layer at the released geometry, synthetic bf16 weights, no checkpoint. At 65,536
# positions the expanded cache is 6.4 GB for ONE layer, over the benchmark's 3 GB default
# budget, so E and E+ are reported NOT RUN there and projected from 16,384. A run that
# would take longer than --max-run-s is not run either; it is PROJECTED from the measured
# cost of one kv_b application, calibrated against the runs that did complete, and every
# output line says which. See benchmarks/bench_mla.c for the contention gating.
#
# Output: <out_dir>/mla-study.jsonl (one JSON object per line) and a .log per step.
# Runtime on the 4-core VM of the note: about an hour and a half when it is quiet.
set -euo pipefail
OUT="${1:?usage: mla-study.sh <out_dir> [threads]}"
THREADS="${2:-$(getconf _NPROCESSORS_ONLN)}"
[ -x ./bin/bench_mla ] && [ -x ./bin/test_mla_variants ] || {
    echo "build first: make bin/bench_mla bin/test_mla_variants"
    exit 1
}
mkdir -p "$OUT"
J="$OUT/mla-study.jsonl"
CS=256,1024,4096,16384,65536

./bin/test_mla_variants | tee "$OUT/test.log"
./bin/bench_mla numerics --threads "$THREADS" --C 1024,16384 --queries 105 --json "$J" \
    | tee "$OUT/numerics.log"
./bin/bench_mla numerics --threads "$THREADS" --C 4096 --queries 105 --qscale 6 \
    --json "$J" | tee "$OUT/numerics-stress.log"
./bin/bench_mla --threads 1 --C "$CS" --T 1,5 --max-run-s 70 --json "$J" \
    | tee "$OUT/time-1.log"
./bin/bench_mla --threads "$THREADS" --C "$CS" --T 1,5 --max-run-s 90 --common \
    --json "$J" | tee "$OUT/time-$THREADS.log"
echo "done: $J"
