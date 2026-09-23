/* bench_kernels.c - where does the arithmetic actually go?
 *
 * The engine's decode time at the memory floor splits roughly 36 s trunk read, 11 s
 * expert read, 10 s compute. The read paths are now within 10-20% of what the device can
 * deliver, so the only lossless win left is the compute. Before optimising it, measure
 * which kernel owns it -- guessing which loop is hot is how people spend a week making
 * something 2% faster.
 *
 * Dimensions here are the REAL ones, per token:
 *   bf16 trunk matmuls   the attention projections, latent down/up, shared experts and
 *                        the dense MLP. Sized from the actual layer shapes.
 *   MXFP4 expert matmuls 16 experts x 3 matrices x 92 layers, in latent space.
 *
 * Reports GFLOP/s and the projected per-token seconds for each, so the two can be
 * compared directly against the measured 10 s compute budget.
 *
 * TIMING. Each kernel is warmed once, then timed call by call; the headline figure is
 * the MEDIAN call and the best call is reported beside it. A shared or busy machine
 * makes the mean of a few calls wander by 2x; the median and best of the same calls
 * wander far less. K3_BENCH_REPS sets the number of timed calls (default 5).
 *
 * THE ROOF. A decode matmul reads every weight once, so its ceiling on a given machine
 * is how fast that machine can stream memory at all. The bench measures that with a
 * plain parallel read over a buffer larger than any cache, at the same thread count
 * (OMP_NUM_THREADS), and prints the bf16 kernel's weight traffic as a fraction of it:
 * near 100% means the kernel is bandwidth-bound there and only fewer bytes can make it
 * faster; well below means compute is the limit.
 */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Exact bits of a whole output vector.
 *
 * Building this file with and without AVX2 and comparing these hashes is the ONLY real
 * proof that a vector path is bit-identical rather than merely close. A tolerance check
 * would happily pass a kernel that quietly reassociated the reduction, which is exactly
 * the mistake worth catching: this engine's claim is that its output equals the
 * reference, and "equals" has to mean equals.
 *
 * This only means anything while the inputs are FINITE. NaN compares unequal to itself
 * and carries a payload that propagates differently through scalar and vector code, so
 * a NaN input turns this from a proof into a guaranteed false alarm. See fillbf16. */
static void fnv(const char *label, const float *v, int n)
{
    unsigned long long h = 1469598103934665603ull;
    for (int k = 0; k < n; k++) {
        union { float f; unsigned u; } b; b.f = v[k];
        for (int t = 0; t < 4; t++) { h ^= (b.u >> (8 * t)) & 0xFFu; h *= 1099511628211ull; }
    }
    printf("             %s OUTPUT FNV1a = %016llx\n", label, h);
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int bench_reps(void)
{
    const char *e = getenv("K3_BENCH_REPS");
    const int n = e ? atoi(e) : 5;
    return n < 1 ? 1 : (n > 1000 ? 1000 : n);
}

/* Per-call times -> median and best, in seconds. t is sorted in place. */
static void med_min(double *t, int n, double *med, double *best)
{
    qsort(t, (size_t)n, sizeof t[0], cmp_double);
    *best = t[0];
    *med = (n & 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

/* STREAM-like read bandwidth: a parallel sum over a 256 MB buffer, eight independent
 * accumulators per thread so the adds never limit it, best of three passes. Integer
 * sums, so the compiler may vectorise freely; the volatile store only keeps the loop
 * from being discarded. */
static volatile uint64_t g_read_sink;

static double read_bandwidth_gbs(void)
{
    const size_t n = (size_t)32 << 20;            /* 32 M x 8 bytes = 256 MB */
    uint64_t *buf = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!buf) return 0.0;
    for (size_t i = 0; i < n; i++) buf[i] = (uint64_t)i;
    double best = 0.0;
    uint64_t sink = 0;
    for (int pass = 0; pass < 3; pass++) {
        uint64_t total = 0;
        const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:total)
#endif
        for (long blk = 0; blk < (long)(n / 4096); blk++) {
            const uint64_t *p = buf + (size_t)blk * 4096;
            uint64_t a[8] = {0};
            for (int i = 0; i < 4096; i += 8)
                for (int l = 0; l < 8; l++) a[l] += p[i + l];
            total += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        }
        const double gbs = (double)(n * sizeof(uint64_t)) / (now_s() - t0) / 1e9;
        if (gbs > best) best = gbs;
        sink += total;
    }
    free(buf);
    g_read_sink = sink;
    return best;
}

static void fillf(float *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}
static void fillb(unsigned char *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s >> 13);
    }
}

/* bf16 weights that are FINITE, which fillb cannot promise.
 *
 * Random bytes reinterpreted as bf16 hit the all-ones exponent about one time in 256,
 * so a 7168-term row is NaN with probability 1 - (255/256)^7168, i.e. essentially
 * always. Every output of the bf16 section then comes out NaN, and hashing NaN compares
 * PAYLOADS rather than arithmetic: those propagate differently through libm's fma() and
 * through _mm256_fmadd_pd, so the two builds disagree unconditionally and the check
 * described above reports a bit-identity failure that is not there.
 *
 * Same xorshift and the same distribution as fillf, truncated to the top half, which is
 * what bf16 storage is. No intermediate float array: at 12288 x 7168 that would be a
 * 352 MB temporary for values consumed once. */
static void fillbf16(uint16_t *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float f = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        unsigned u; memcpy(&u, &f, sizeof u);
        p[i] = (unsigned short)(u >> 16);
    }
}

int main(void)
{
    printf("kernel benchmark at REAL Kimi K3 dimensions\n");
#if defined(__AVX512F__)
    printf("built WITH AVX2 and AVX-512\n");
#elif defined(__AVX2__)
    printf("built WITH AVX2\n");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    printf("built WITH NEON\n");
#else
    printf("built WITHOUT AVX2 or NEON (scalar)\n");
#endif
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    const int reps = bench_reps();
    printf("%d thread(s), median and best of %d timed calls per kernel\n\n", threads, reps);
    double *tc = (double *)malloc((size_t)reps * sizeof(double));
    if (!tc) { printf("alloc failed\n"); return 1; }
    double bf16_gbs = 0.0;

    /* ---------- bf16 trunk matmul: KDA q_proj, 7168 -> 12288 ---------- */
    {
        const int in = 7168, out = 12288;
        uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)out * sizeof(float));
        if (!W || !x || !y) { printf("alloc failed\n"); return 1; }
        fillbf16(W, (size_t)in * out, 12345u);
        fillf(x, in, 999u);

        k3_matmul_bf16(y, x, W, in, out);              /* warm */
        for (int r = 0; r < reps; r++) {
            const double t0 = now_s();
            k3_matmul_bf16(y, x, W, in, out);
            tc[r] = now_s() - t0;
        }
        double dt, best;
        med_min(tc, reps, &dt, &best);
        const double gflop = 2.0 * in * out / 1e9;
        const double wbytes = 2.0 * in * out;
        printf("bf16 matmul  %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               out, in, dt * 1e3, gflop / dt);
        bf16_gbs = wbytes / dt / 1e9;
        printf("             best %.2f ms (%.1f GFLOP/s); weights stream at %.1f GB/s median\n",
               best * 1e3, gflop / best, bf16_gbs);

        /* Per token the trunk is 56.74 G always-active params = 113.49 GFLOP of
         * multiply-add. Project from the rate just measured. */
        fnv("bf16 ", y, out);
        printf("             trunk is 56.74 G params/token -> %.2f s/token at this rate\n",
               2.0 * 56.74e9 / 1e9 / (gflop / dt));
        free(W); free(x); free(y);
    }

    /* ---------- MXFP4 expert matmul: w1, latent 3584 -> inter 3072 ---------- */
    {
        const int in = 3584, rows = 3072, group = K3_MXFP4_GROUP;
        const int pcols = in / 2, ngrp = in / group;
        unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
        unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)rows * sizeof(float));
        if (!pk || !sc || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb(pk, (size_t)rows * pcols, 777u);
        memset(sc, 127, (size_t)rows * ngrp);          /* scale 2^0, none skipped */
        fillf(x, in, 4242u);

        k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        for (int r = 0; r < reps; r++) {
            const double t0 = now_s();
            k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
            tc[r] = now_s() - t0;
        }
        double dt, best;
        med_min(tc, reps, &dt, &best);
        const double gflop = 2.0 * in * rows / 1e9;
        printf("\nMXFP4 matmul %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               rows, in, dt * 1e3, gflop / dt);
        /* The same packed matrix is re-read on every call, so where the last-level
         * cache holds it this is a cache rate, and where it does not, a DRAM rate. The
         * bench does not know which: it states the size and leaves that to the reader. */
        printf("             best %.2f ms (%.1f GFLOP/s); the same %.1f MB matrix every call "
               "(a cache rate if it fits the last-level cache)\n",
               best * 1e3, gflop / best,
               ((double)rows * pcols + (double)rows * ngrp) / 1e6);

        /* One expert is w1 + w3 (both 3072x3584) + w2 (3584x3072) = 3 of these.
         * 16 experts x 92 MoE layers per token. */
        fnv("mxfp4", y, rows);
        const double per_tok = dt * 3.0 * 16 * 92;
        printf("             16 experts x 3 mats x 92 layers -> %.2f s/token\n", per_tok);
        free(pk); free(sc); free(x); free(y);
    }

    {
        const double roof = read_bandwidth_gbs();
        printf("\nread bandwidth, %d thread(s): %.1f GB/s (streaming sum over 256 MB, best of 3)\n",
               threads, roof);
        if (roof > 0.0)
            printf("             bf16 weights stream at %.0f%% of it: %s\n",
                   100.0 * bf16_gbs / roof,
                   bf16_gbs >= 0.8 * roof ? "bandwidth-bound here"
                                          : "compute-bound here (more cores or wider SIMD help)");
    }
    free(tc);

    printf("\nmeasured compute budget at the floor is about 10 s/token; whichever line\n"
           "above dominates it is the one worth vectorising.\n");
    return 0;
}
