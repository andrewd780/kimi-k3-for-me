/* One serial batch of 96 independent 128x128 heads. Synthetic kernel timing only;
 * there are no model weights, I/O, layer projections or per-token estimates here. */
#define _POSIX_C_SOURCE 200809L
#include "k3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static unsigned rng = 20260913u;
static float sample(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((float)(rng >> 8) / 8388608.0f - 1.0f) * 0.03125f;
}

int main(int argc, char **argv)
{
    const int heads = 96, dim = 128;
    const int batches = argc == 2 ? atoi(argv[1]) : 512;
    if (batches < 1 || batches > 100000) return 2;
    const size_t n = (size_t)heads * dim, ns = n * dim;
    float *s = (float *)malloc(ns * sizeof(float));
    float *o = (float *)malloc(n * sizeof(float));
    float *q = (float *)malloc(n * sizeof(float));
    float *k = (float *)malloc(n * sizeof(float));
    float *v = (float *)malloc(n * sizeof(float));
    float *a = (float *)malloc(n * sizeof(float));
    if (!s || !o || !q || !k || !v || !a) return 1;
    for (size_t i = 0; i < ns; i++) s[i] = sample();
    for (size_t i = 0; i < n; i++) {
        q[i] = sample(); k[i] = sample(); v[i] = sample(); a[i] = 0.96f + sample();
    }
    /* One untimed warm batch; each invocation starts from the same inputs. */
    double start = 0.0;
    for (int r = -1; r < batches; r++) {
        if (r == 0) start = now_s();
        for (int h = 0; h < heads; h++) {
            const size_t p = (size_t)h * dim;
            k3_kda_step(s + p * dim, o + p, q + p, k + p, v + p, a + p,
                        0.5f, dim, dim);
        }
    }
    const double elapsed = now_s() - start;
    unsigned long long hash = 14695981039346656037ull;
    for (int part = 0; part < 2; part++) {
        const float *data = part ? o : s;
        const size_t count = part ? n : ns;
        for (size_t i = 0; i < count; i++) {
            unsigned bits;
            if (!isfinite(data[i])) return 1;
            memcpy(&bits, data + i, sizeof bits);
            for (int b = 0; b < 4; b++) {
                hash ^= (bits >> (8 * b)) & 255u;
                hash *= 1099511628211ull;
            }
        }
    }
    printf("{\"heads\":96,\"head_dim\":128,\"batches\":%d,\"seconds\":%.9f,"
           "\"ms_per_96_heads\":%.9f,\"state_output_fnv1a\":\"%016llx\"}\n",
           batches, elapsed, elapsed * 1000.0 / batches, hash);
    free(s); free(o); free(q); free(k); free(v); free(a);
    return 0;
}
