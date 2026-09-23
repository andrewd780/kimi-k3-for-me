# Exact decode kernels: quiet-machine timings

The decode matmuls (`k3_matmul`, `k3_matmul_bf16`, `k3_matmul_mxfp4`) now widen x once
per call, take two rows per pass on x86 bf16 and AVX-512 MXFP4, prefetch streamed rows
on x86, and have AVX-512 paths of their own. None of this moves a bit: every path keeps
its accumulation partition and reduction tree, and `test_matmul_exact` holds them to it
(see [TESTING.md](../TESTING.md)). This note records what the change is worth in time on
one machine, old kernels against new, with the output hashes checked in the same log.

These are kernel microbenchmarks at K3's shapes. They make no s/token claim.

## Conditions

| | |
|---|---|
| CPU | Intel Xeon @ 2.80 GHz (family 6, model 85, stepping 7), AVX2 + AVX-512F/BW/VL |
| Machine | KVM guest, 4 vCPUs (1 thread per core), L2 1 MiB per core, L3 33 MiB, 15 GB RAM |
| Compiler | GCC 13.3.0, `-O3 -std=gnu99 -fopenmp -pthread -ffp-contract=off` plus the ISA flags |
| Builds | `native` = `-march=native`; `avx2` = `-mavx2 -mfma` |
| Old kernels | `src/core/k3_ops.c` at 49f5ccb |
| New kernels | `src/core/k3_ops.c` at 3612040 |
| Harness | `benchmarks/bench_kernels.c` at 3612040, linked against each kernel object |
| Threads | 1 and 4; `OMP_PROC_BIND=close OMP_PLACES=cores` |
| Repeats | 5 runs per binary and thread count, each the median and best of `K3_BENCH_REPS=11` timed calls after one warm call |
| Order | per round: old native, new native, old avx2, new avx2; five rounds at 1 thread, then five at 4 |
| Load | before every run the 1-minute load average had to be below 1.0 (polled every 30 s); it was 0.36 to 0.54 over the 1-thread runs and 0.54 to 0.66 over the 4-thread runs; no other job was running |

Both kernel objects are linked into the SAME harness, so the timing code, the inputs and
the read-bandwidth probe are identical and only the kernels differ. The base commit has
no AVX-512 kernels, so its `native` build runs the AVX2 intrinsics compiled with
`-march=native`; the harness banner still says "built WITH AVX2 and AVX-512" there,
because it reports the harness's own flags, not the kernel object's.

## Results

Each figure is the median over the five runs of the per-run median call; "best" is the
fastest single call in all five runs. The bf16 matrix (12288 x 7168, 176 MB) is far
larger than the L3, so its rate is a DRAM streaming rate; "read bw" is the harness's
plain parallel read over 256 MB at the same thread count, and "of read bw" is the median
of the per-run ratios the harness prints.

**bf16 matmul 12288 x 7168** (KDA q_proj)

| threads | build | old ms | new ms | speedup | old best | new best | new GB/s | read bw GB/s | old / new of read bw |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | native | 29.94 | 19.97 | 1.50x | 28.44 | 18.72 | 8.8 | 9.8 | 61% / 93% |
| 1 | avx2 | 30.06 | 22.36 | 1.34x | 28.52 | 21.70 | 7.9 | 9.4 | 62% / 82% |
| 4 | native | 8.06 | 5.60 | 1.44x | 7.57 | 5.07 | 31.5 | 36.5 | 61% / 84% |
| 4 | avx2 | 8.01 | 6.32 | 1.27x | 7.42 | 5.46 | 27.9 | 35.2 | 61% / 83% |

**MXFP4 matmul 3072 x 3584** (one expert projection; 5.8 MB packed with its scales)

| threads | build | old ms | new ms | speedup | old best | new best |
|---:|---|---:|---:|---:|---:|---:|
| 1 | native | 3.95 | 1.37 | 2.88x | 3.63 | 1.30 |
| 1 | avx2 | 3.91 | 3.90 | 1.00x | 3.64 | 3.67 |
| 4 | native | 1.12 | 0.47 | 2.38x | 0.98 | 0.35 |
| 4 | avx2 | 1.16 | 1.19 | 0.97x | 1.01 | 1.01 |

**Exactness in the same log.** All 40 runs print the same two output hashes, old
kernels and new, native and AVX2, 1 and 4 threads:

```
     40              bf16  OUTPUT FNV1a = 83c8504a4cb3fac6
     40              mxfp4 OUTPUT FNV1a = a231061237b5579d
```

### Every run

Per-run median call in ms, in run order (CONTRIBUTING asks for all of them).

| threads | binary | bf16 runs 1-5 | MXFP4 runs 1-5 |
|---:|---|---|---|
| 1 | old native | 29.94 29.67 30.09 29.71 30.73 | 3.95 3.92 3.82 3.97 4.04 |
| 1 | new native | 19.97 19.11 20.17 19.44 20.62 | 1.36 1.37 1.37 1.38 1.41 |
| 1 | old avx2 | 30.06 30.68 30.18 29.42 29.50 | 3.72 4.08 3.71 4.04 3.91 |
| 1 | new avx2 | 22.36 22.80 22.23 22.25 23.25 | 3.79 4.34 3.74 3.90 3.92 |
| 4 | old native | 11.27 8.06 9.24 7.86 7.97 | 1.13 1.11 1.13 1.12 1.11 |
| 4 | new native | 7.03 5.60 8.44 5.26 5.24 | 0.78 0.44 0.47 0.49 0.40 |
| 4 | old avx2 | 8.01 10.27 7.79 8.32 7.74 | 1.21 1.16 1.57 1.14 1.02 |
| 4 | new avx2 | 6.32 5.68 6.77 5.71 9.21 | 1.75 2.10 1.13 1.19 1.16 |

Read bandwidth per run, GB/s: 1 thread 8.5 to 10.1; 4 threads 20.6 to 37.8.

## What the numbers say

- **bf16, the trunk kernel, is now near the memory roof on this machine.** The old
  kernels streamed weights at about 61% of what a plain read achieves, at both thread
  counts and both ISAs; the new AVX-512 build reaches 84% to 93%, the new AVX2 build
  82% to 83%. What is left is small: past this point a faster trunk needs fewer bytes,
  not faster arithmetic.
- **At 1 thread every bf16 difference is clear of the noise**: the five per-run medians
  of old and new never overlap (old 29.4 to 30.7 ms, new 19.1 to 20.6 native and 22.2 to
  23.3 AVX2), and each run is itself a median of 11 calls.
- **At 4 threads the runs are noisier**, in both arms alike: one or two runs in five are
  slow (old native 11.27 ms, old avx2 10.27, new native 8.44, new avx2 9.21), and in two
  old-native runs the read probe fell to 20.6 and 22.6 GB/s against a usual 32 to 38.
  The guest's load stayed below 0.7 throughout, so this points at the host sharing the
  four vCPUs rather than at the kernels. The medians still show 1.44x (native) and 1.27x
  (AVX2), but the AVX2 run ranges overlap (old 7.74 to 10.27 ms, new 5.68 to 9.21), so
  the 4-thread AVX2 figure is the weakest one here.
- **MXFP4 gains only where it has a new path.** The AVX-512 flat path is 2.4x to 2.9x
  faster; the AVX2 build, whose MXFP4 path only lost the per-row widening of x, is
  unchanged within noise (1.00x and 0.97x). The 5.8 MB matrix fits in the 33 MiB L3 and
  is re-read every call, so these are cache rates: a routed expert read from RAM or SSD
  per token runs at the memory rate, and this table does not predict that.

## Not measured

- **aarch64 / NEON.** No native arm64 machine was available; timings under qemu mean
  nothing, so none were taken. The NEON paths are covered for bits by `test_matmul_exact`
  under qemu, not for speed.
- **Full-model s/token.** A kernel microbenchmark on one 4-vCPU guest does not establish
  an end-to-end speedup, whose floor is set by SSD reads at small memory budgets.

## Reproduce

```sh
D=/path/to/kern-timing; rm -rf $D; mkdir -p $D/base && git archive 49f5ccb | tar -x -C $D/base
for v in "native:-march=native" "avx2:-mavx2 -mfma"; do n=${v%%:*}; a=${v#*:}
  make -C $D/base -j2 ARCH="$a" BUILD=build/$n BIN=bin/$n build/$n/src/core/k3_ops.o
  make -j2 ARCH="$a" BUILD=build/t-$n BIN=bin/t-$n build/t-$n/src/core/k3_ops.o
  for k in old:$D/base/build/$n new:build/t-$n; do
    cc -O3 -std=gnu99 $a -fopenmp -pthread -ffp-contract=off -Iinclude -Iinclude/k3 \
       -Ithird_party -Isrc/core benchmarks/bench_kernels.c ${k#*:}/src/core/k3_ops.o \
       -o $D/bench_${k%%:*}_$n -lm
  done
done
for t in 1 4; do for i in 1 2 3 4 5; do
  for b in old_native new_native old_avx2 new_avx2; do
    echo "== $b threads=$t run=$i"
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores K3_BENCH_REPS=11 $D/bench_$b
  done
done; done 2>&1 | tee $D/timing.log
grep FNV $D/timing.log | sort | uniq -c    # exactly two distinct lines
```

Wait for a quiet machine (1-minute load below 1.0) before each run; the runs above did.
