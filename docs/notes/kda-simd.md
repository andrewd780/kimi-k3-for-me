# Exact KDA SIMD: measured, opt-in

The default stays the original optimized C. Three SIMD layouts were byte-exact,
but none established a useful improvement on both CI architectures. The original
C already permits compiler auto-vectorization; it is not an artificially slowed
scalar control. No full-model or Andrew-machine benchmark was run.

The retained experiment combines decay with the key read and the rank-one write
with the query read. Independent value columns use AVX2 (eight float lanes) or
NEON (four). Key rows accumulate in ascending order. Multiply and add remain
separate under `-ffp-contract=off`, and zero-key/query skips are preserved.
State access stays row-major; the earlier column tiles regressed on ARM.

Enable explicitly with `make KDA_SIMD=1 BUILD=build/kda BIN=bin/kda`, or configure
CMake with `-DK3_KDA_SIMD=ON`. Use separate build directories when changing flags.
The portable fallback remains available. This is an experiment, not a recommended
speed setting.

## Three-run timing gate

Commands and raw observations are preserved in
[`kda-simd-ci.json`](../measurements/kda-simd-ci.json). The final row-major experiment
was measured in [CI run 34768596191](https://github.com/andrewd780/kimi-k3-for-me/actions/runs/34768596191).
Each observation averages 4,096 serial batches of 96 heads at head dimension 128.
Every invocation starts from identical synthetic inputs; one warm batch is
excluded. Arms alternate order. Times below are **milliseconds per 96-head batch**.

| CI ISA / CPU | Arm | Run 1 | Run 2 | Run 3 | Median |
| --- | --- | ---: | ---: | ---: | ---: |
| AVX2 / EPYC 7763 runner | Original C | 0.718909 | 0.586558 | 0.586791 | 0.586791 |
| AVX2 / EPYC 7763 runner | SIMD | 0.529286 | 0.529468 | 0.532081 | 0.529468 |
| NEON / Apple M1 virtual runner | Original C | 0.567039 | 0.767477 | 0.735936 | 0.735936 |
| NEON / Apple M1 virtual runner | SIMD | 0.952165 | 1.028658 | 1.043498 | 1.028658 |

The x86 median is 1.11x the control rate, below the campaign's conservative 33%
effect threshold. ARM is slower, with substantial control dispersion. These are
shared CI runners, not the campaign host or Andrew's Macs. The results establish
neither laptop throughput nor an end-to-end speedup.

Reproduce with `python3 tools/bench_kda.py --out kda.json`. This builds both arms,
checks the existing 22 op fixtures (including KDA decay), compares all 526,336
oracle-logit bytes across GATE 1/2/3 within each ISA, and performs 9,504 repeated
state/output comparisons over tails, misalignment, zero skips and widths above
the stack temporary bound. Both ISAs also run the differential check under ASan
and UBSan. The timing gate hashes the complete final state and output, not just
argmax tokens. Existing fixtures are unchanged. Oracle hashes differ between
architectures due to existing platform arithmetic; only within-ISA equality is
claimed.
