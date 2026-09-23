#!/usr/bin/env bash
# The full MLA KV-cache variant study behind docs/notes/mla-variants.md.
#
#   benchmarks/mla-study.sh <out_dir> [threads] [phase]
#
# phase is `exact` (steps 1-3), `timing` (step 4) or `all` (default). The exact phase
# gives the same bits on any machine under any load; the timing phase wants a quiet one.
# Steps, strictly one after another (two benchmarks at once would time each other):
#   1. the bitwise test: E+, L0 and L1 against the engine's own k3_mla_cached, on
#      ordinary, cancelling and sharp layers, with the kv_b application counts;
#   2. counts, no timing: kv_b applications per call, COUNTED, at the prefill shape
#      (C = 0, T = 1, 16, 64, 256) and at the decode shapes (C = 256 .. 65,536, T = 1, 5);
#   3. the numerical study of the absorbed variant A at 1,024 and 16,384 cached
#      positions, and a sharper-softmax stress run (queries scaled by 6) at 4,096;
#   4. timing on one thread, then on [threads] (default: every online CPU), at 256, 1,024,
#      4,096, 16,384 and 65,536 cached positions, T = 1 and 5, and the prefill point
#      C = 0, T = 256.
#
# One layer at the released geometry, synthetic bf16 weights, no checkpoint. At 65,536
# positions the expanded cache is 6.4 GB for ONE layer, over the benchmark's 3 GB default
# budget, so E and E+ are reported NOT RUN there and projected from 16,384. A run that
# would take longer than --max-run-s is not run either; it is PROJECTED from the measured
# cost of one kv_b application, calibrated against the runs that did complete, and every
# output line says which. See benchmarks/bench_mla.c for the contention gating.
#
# Output: <out_dir>/mla-study.jsonl (one JSON object per line) and a .log per step.
# Runtime on the 4-core VM of the note: the exact phase about ten minutes, the timing
# phase about an hour and a half when the machine is quiet.
set -euo pipefail
OUT="${1:?usage: mla-study.sh <out_dir> [threads] [exact|timing|all]}"
THREADS="${2:-$(getconf _NPROCESSORS_ONLN)}"
PHASE="${3:-all}"
case "$PHASE" in exact|timing|all) ;; *) echo "phase must be exact, timing or all"; exit 2 ;; esac
[ -x ./bin/bench_mla ] && [ -x ./bin/test_mla_variants ] || {
    echo "build first: make bin/bench_mla bin/test_mla_variants"
    exit 1
}
mkdir -p "$OUT"
J="$OUT/mla-study.jsonl"
CS=256,1024,4096,16384,65536

if [ "$PHASE" != timing ]; then
    ./bin/test_mla_variants | tee "$OUT/test.log"
    ./bin/bench_mla counts --threads 1 --C 0 --T 1,16,64,256 --json "$J" \
        | tee "$OUT/counts-prefill.log"
    ./bin/bench_mla counts --threads 1 --C "$CS" --T 1,5 --json "$J" \
        | tee "$OUT/counts-decode.log"
    ./bin/bench_mla numerics --threads "$THREADS" --C 1024,16384 --queries 105 --json "$J" \
        | tee "$OUT/numerics.log"
    ./bin/bench_mla numerics --threads "$THREADS" --C 4096 --queries 105 --qscale 6 \
        --json "$J" | tee "$OUT/numerics-stress.log"
fi
if [ "$PHASE" != exact ]; then
    ./bin/bench_mla --threads 1 --C "$CS" --T 1,5 --runs 5 --max-run-s 70 --json "$J" \
        | tee "$OUT/time-1.log"
    ./bin/bench_mla --threads 1 --C 0 --T 256 --runs 5 --max-run-s 70 --json "$J" \
        | tee "$OUT/time-prefill-1.log"
    ./bin/bench_mla --threads "$THREADS" --C "$CS" --T 1,5 --runs 5 --max-run-s 90 --common \
        --json "$J" | tee "$OUT/time-$THREADS.log"
    ./bin/bench_mla --threads "$THREADS" --C 0 --T 256 --runs 5 --max-run-s 90 --json "$J" \
        | tee "$OUT/time-prefill-$THREADS.log"
fi
echo "done: $J"
