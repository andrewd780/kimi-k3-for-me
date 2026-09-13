/* Bitwise, repeated-state differential test against the original four-pass C.
 * Tails, misalignment, heap-sized widths, signed zeros and cancellation matter:
 * an argmax/tolerance check would miss a changed reduction or fused operation. */
#include "k3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rng = 20260913u;
static float sample(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((float)(rng >> 8) / 8388608.0f - 1.0f) * 0.03125f;
}

static void reference(float *s, float *o, const float *q, const float *k,
                      const float *v, const float *a, float beta, int dk, int dv)
{
    float *u = (float *)calloc((size_t)dv, sizeof(float));
    if (!u) abort();
    for (int i = 0; i < dk; i++)
        for (int j = 0; j < dv; j++) s[(size_t)i * dv + j] *= a[i];
    for (int i = 0; i < dk; i++) {
        if (k[i] == 0.0f) continue;
        for (int j = 0; j < dv; j++) u[j] += k[i] * s[(size_t)i * dv + j];
    }
    for (int i = 0; i < dk; i++) {
        if (k[i] == 0.0f) continue;
        for (int j = 0; j < dv; j++)
            s[(size_t)i * dv + j] += k[i] * beta * (v[j] - u[j]);
    }
    for (int j = 0; j < dv; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        if (q[i] == 0.0f) continue;
        for (int j = 0; j < dv; j++) o[j] += q[i] * s[(size_t)i * dv + j];
    }
    free(u);
}

int main(void)
{
    const int widths[] = {1, 3, 4, 7, 8, 9, 31, 127, 128, 129, 257};
    const int keys[] = {1, 7, 128};
    unsigned cases = 0;
    for (size_t w = 0; w < sizeof widths / sizeof *widths; w++) {
        for (size_t d = 0; d < sizeof keys / sizeof *keys; d++) {
            const int dv = widths[w], dk = keys[d];
            const size_t n = (size_t)dk * dv;
            float *s0 = (float *)malloc((n + 2) * sizeof(float));
            float *s1 = (float *)malloc((n + 2) * sizeof(float));
            float *o0 = (float *)malloc((size_t)(dv + 2) * sizeof(float));
            float *o1 = (float *)malloc((size_t)(dv + 2) * sizeof(float));
            float *q = (float *)malloc((size_t)dk * sizeof(float));
            float *k = (float *)malloc((size_t)dk * sizeof(float));
            float *a = (float *)malloc((size_t)dk * sizeof(float));
            float *v = (float *)malloc((size_t)dv * sizeof(float));
            if (!s0 || !s1 || !o0 || !o1 || !q || !k || !a || !v) abort();
            for (int trial = 0; trial < 12; trial++) {
                for (size_t i = 0; i < n + 2; i++) s0[i] = sample();
                memcpy(s1, s0, (n + 2) * sizeof(float));
                for (int j = 0; j < dv + 2; j++) o0[j] = o1[j] = 123.25f;
                for (int step = 0; step < 24; step++) {
                    for (int i = 0; i < dk; i++) {
                        q[i] = sample(); k[i] = sample(); a[i] = 0.96f + sample();
                        if ((i + trial) % 5 == 0) k[i] = trial % 2 ? -0.0f : 0.0f;
                        if ((i + step) % 7 == 0) q[i] = step % 2 ? -0.0f : 0.0f;
                        if (trial == 0) k[i] = q[i] = 0.0f;
                        if (trial == 1) a[i] = i % 2 ? 1.0f : 0.0f;
                    }
                    for (int j = 0; j < dv; j++) v[j] = sample();
                    const float beta = step % 3 == 0 ? 0.0f : 0.5f + sample();
                    reference(s0 + 1, o0 + 1, q, k, v, a, beta, dk, dv);
                    k3_kda_step(s1 + 1, o1 + 1, q, k, v, a, beta, dk, dv);
                    if (memcmp(s0, s1, (n + 2) * sizeof(float)) ||
                        memcmp(o0, o1, (size_t)(dv + 2) * sizeof(float))) {
                        fprintf(stderr, "KDA mismatch dk=%d dv=%d trial=%d step=%d\n",
                                dk, dv, trial, step);
                        return 1;
                    }
                    cases++;
                }
            }
            free(s0); free(s1); free(o0); free(o1); free(q); free(k); free(a); free(v);
        }
    }
    printf("KDA EXACT: %u state/output comparisons passed (including guards)\n", cases);
    return 0;
}
