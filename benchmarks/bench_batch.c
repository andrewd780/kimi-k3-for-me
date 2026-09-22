/* bench_batch.c - what applying a trunk matrix to T positions at once buys.
 *
 * Prefill, speculative verification and draft prefill apply every trunk matrix to T
 * positions. The per-position loop streams the matrix through the core T times and redoes
 * the bf16 -> f32 -> f64 widening, the kernel's real cost, T times; k3_matmul_bf16_batch
 * streams it once per pass and widens each weight once per block of positions. This times
 * both at the REAL shape of a KDA q/k/v/g projection, 12288 x 7168, for T = 1, 2, 4, 8,
 * at one thread and at the OpenMP default (four on the reference VM), and checks that
 * both produce the same bits.
 *
 * Columns:
 *   ms median / min   wall time of one call over all T positions, over the repetitions
 *   ms/pos            median divided by T
 *   GFLOP/s           2*in*out*T / median
 *   weight GB/s       bytes of weights streamed through the core per second: the loop
 *                     streams the matrix T times, the batch once per pass of positions
 *                     (all of these T fit one pass). Batching lowers this on purpose: the
 *                     same work needs less weight traffic.
 *
 * Timings are wall clock on whatever else the machine is doing; repeat them, and read the
 * median and the minimum together.
 *
 * usage: bench_batch [reps]   (default 7)
 */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Finite bf16 weights and fp32 activations, the same distribution as bench_kernels. */
static void fill(uint16_t *w, size_t nw, float *x, size_t nx, unsigned s)
{
    for (size_t i = 0; i < nw; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float f = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        unsigned u; memcpy(&u, &f, sizeof u);
        w[i] = (uint16_t)(u >> 16);
    }
    for (size_t i = 0; i < nx; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void loop_tokens(float *Y, const float *X, const uint16_t *W, int in, int out, int T)
{
    for (int t = 0; t < T; t++) k3_matmul_bf16(Y + (size_t)t * out, X + (size_t)t * in, W, in, out);
}

int main(int argc, char **argv)
{
    const int in = 7168, out = 12288, reps = argc > 1 ? atoi(argv[1]) : 7;
    const int Ts[] = {1, 2, 4, 8}, nT = (int)(sizeof Ts / sizeof *Ts), Tmax = 8;
    if (reps < 1 || reps > 1000) { fprintf(stderr, "reps must be 1..1000\n"); return 2; }
    uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
    float *X  = (float *)malloc((size_t)Tmax * in * sizeof(float));
    float *Yl = (float *)malloc((size_t)Tmax * out * sizeof(float));
    float *Yb = (float *)malloc((size_t)Tmax * out * sizeof(float));
    double *ts = (double *)malloc((size_t)reps * sizeof(double));
    if (!W || !X || !Yl || !Yb || !ts) { fprintf(stderr, "alloc failed\n"); return 1; }
    fill(W, (size_t)in * out, X, (size_t)Tmax * in, 12345u);

#ifdef __AVX2__
    const char *isa = "AVX2";
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const char *isa = "NEON";
#else
    const char *isa = "scalar";
#endif
    int threads[2] = {1, 1}, nth = 1;
#ifdef _OPENMP
    threads[1] = omp_get_max_threads();
    if (threads[1] > 1) nth = 2;
#endif
    const double wbytes = 2.0 * in * out;
    printf("batched bf16 matmul, %d x %d (%.0f MB of bf16 weights), built %s, %d reps\n\n",
           out, in, wbytes / 1e6, isa, reps);
    printf("threads  T  kernel     ms median     min   ms/pos  GFLOP/s  weight GB/s  bits\n");
    int all_exact = 1;
    for (int th = 0; th < nth; th++) {
#ifdef _OPENMP
        omp_set_num_threads(threads[th]);
#endif
        for (int a = 0; a < nT; a++) {
            const int T = Ts[a];
            for (int batched = 0; batched < 2; batched++) {
                float *Y = batched ? Yb : Yl;
                /* one untimed warm call, then the repetitions */
                if (batched) k3_matmul_bf16_batch(Y, X, W, in, out, T);
                else         loop_tokens(Y, X, W, in, out, T);
                for (int r = 0; r < reps; r++) {
                    const double t0 = now_s();
                    if (batched) k3_matmul_bf16_batch(Y, X, W, in, out, T);
                    else         loop_tokens(Y, X, W, in, out, T);
                    ts[r] = now_s() - t0;
                }
                qsort(ts, (size_t)reps, sizeof *ts, cmp_d);
                const double med = ts[reps / 2], mn = ts[0];
                const double passes = batched ? 1.0 : (double)T;
                const int exact = !batched || !memcmp(Yl, Yb, (size_t)T * out * sizeof(float));
                if (!exact) all_exact = 0;
                printf("%7d %2d  %-8s %9.2f %9.2f %8.2f %8.1f %12.2f  %s\n",
                       threads[th], T, batched ? "batched" : "loop", med * 1e3, mn * 1e3,
                       med * 1e3 / T, 2.0 * in * out * T / med / 1e9,
                       wbytes * passes / med / 1e9,
                       batched ? (exact ? "== loop" : "DIFFER") : "");
            }
        }
        printf("\n");
    }
    printf("%s\n", all_exact ? "every batched output bit-identical to the per-position loop"
                             : "BATCHED OUTPUT DIFFERS FROM THE PER-POSITION LOOP");
    free(W); free(X); free(Yl); free(Yb); free(ts);
    return all_exact ? 0 : 1;
}
