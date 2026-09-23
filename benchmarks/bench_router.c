/* bench_router.c - what one MoE router call costs, and the bits it produces.
 *
 * k3_router runs once per token in each of the 92 MoE layers: 896 dot products of
 * length 7168 in double, a sigmoid per expert, then the top-16. It reads no weights from
 * disk (the gate is resident), so it is pure arithmetic on the per-token critical path.
 * This times one call at the released shape and hashes the chosen indices and weights
 * over 64 inputs, so that builds of different router code can be timed in the same
 * harness and shown to agree bit for bit (link it against each k3_ops.o, as
 * docs/notes/decode-kernels.md does).
 *
 * The data is the ordinary router distribution (W in +-0.05, x in +-1, bias in +-0.01).
 * That is right for timing and for a same-build hash, but it is NOT a test of summation
 * order: a reordered double chain almost never moves a float logit on such data.
 * test_ops holds the order on cancelling data built for that.
 *
 * usage: bench_router [reps]   (default 51 timed calls after one warm call)
 * Threads follow OMP_NUM_THREADS; the engine runs the router threaded over experts.
 */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
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

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static float unit(unsigned *s, float a)
{
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return ((float)(*s >> 8) / 8388608.0f - 1.0f) * a;
}

int main(int argc, char **argv)
{
    enum { E = 896, H = 7168, K = 16, NX = 64 };
    const int reps = argc > 1 ? atoi(argv[1]) : 51;
    if (reps < 1 || reps > 100000) { fprintf(stderr, "reps must be 1..100000\n"); return 2; }
    float  *W  = (float *)malloc((size_t)E * H * sizeof(float));
    float  *x  = (float *)malloc((size_t)NX * H * sizeof(float));
    float  *b  = (float *)malloc((size_t)E * sizeof(float));
    double *ts = (double *)malloc((size_t)reps * sizeof(double));
    if (!W || !x || !b || !ts) { fprintf(stderr, "alloc failed\n"); return 1; }
    unsigned s = 12345u;
    for (size_t i = 0; i < (size_t)E * H; i++) W[i] = unit(&s, 0.05f);
    for (size_t i = 0; i < (size_t)NX * H; i++) x[i] = unit(&s, 1.0f);
    for (int e = 0; e < E; e++) b[e] = unit(&s, 0.01f);

    int idx[K]; float w[K];
    unsigned long long h = 1469598103934665603ULL;          /* FNV-1a over 64 calls */
    for (int r = 0; r < NX; r++) {
        k3_router(idx, w, x + (size_t)r * H, W, b, H, E, K, 1, 2.5f);
        const unsigned char *p = (const unsigned char *)idx;
        for (size_t i = 0; i < sizeof idx; i++) { h ^= p[i]; h *= 1099511628211ULL; }
        p = (const unsigned char *)w;
        for (size_t i = 0; i < sizeof w; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    }

    k3_router(idx, w, x, W, b, H, E, K, 1, 2.5f);           /* warm */
    for (int r = 0; r < reps; r++) {
        const double t0 = now_s();
        k3_router(idx, w, x + (size_t)(r % NX) * H, W, b, H, E, K, 1, 2.5f);
        ts[r] = now_s() - t0;
    }
    qsort(ts, (size_t)reps, sizeof *ts, cmp_d);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("router %d x %d top-%d, %d thread(s): median %.3f ms  best %.3f ms  (%d calls)\n",
           E, H, K, threads, ts[reps / 2] * 1e3, ts[0] * 1e3, reps);
    printf("             92 MoE layers -> %.3f s/token at the median\n", ts[reps / 2] * 92);
    printf("             router OUTPUT FNV1a = %016llx\n", h);
    free(W); free(x); free(b); free(ts);
    return 0;
}
