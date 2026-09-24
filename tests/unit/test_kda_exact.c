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

/* ---- tentative sweeps: k3_kda_layer_log + k3_kda_advance -------------------------------
 * The contract a speculative verify sweep relies on: a logged call computes the same
 * output bits as a plain one while leaving the carried state untouched, and committing n
 * of its positions with k3_kda_advance leaves EXACTLY the state feeding those n positions
 * would have, both the recurrent matrix and the ShortConv history, bit for bit. Checked
 * against fresh runs rather than against a formula, and then used: decoding continues
 * from the committed state and must stay bit-identical to decoding from the fresh one.
 * Also checked: a log built by several short calls (the hybrid draft's shape) behaves as
 * the same calls would have in place. */
typedef struct {
    K3Cfg c;
    K3KdaW w;
    float *buf[16];
    int nbuf;
} RwModel;

static float *rw_fill(RwModel *m, size_t n, float scale, float offset)
{
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) abort();
    for (size_t i = 0; i < n; i++) p[i] = offset + sample() * scale;
    m->buf[m->nbuf++] = p;
    return p;
}

static void rw_model(RwModel *m, int E, int H, int D, int K)
{
    memset(m, 0, sizeof *m);
    m->c.hidden = E; m->c.kda_heads = H; m->c.kda_head_dim = D; m->c.conv_k = K;
    m->c.gate_lb = -5.0f; m->c.rms_eps = 1e-5f;
    const size_t P = (size_t)H * D;
    /* sample() is +-1/32, so the scales bring each projection to O(1) outputs: the
     * recurrence must see real decay, real deltas and a history that matters. */
    m->w.q = rw_fill(m, P * E, 24.0f, 0.0f);
    m->w.k = rw_fill(m, P * E, 24.0f, 0.0f);
    m->w.v = rw_fill(m, P * E, 24.0f, 0.0f);
    m->w.q_conv = rw_fill(m, P * K, 16.0f, 0.0f);
    m->w.k_conv = rw_fill(m, P * K, 16.0f, 0.0f);
    m->w.v_conv = rw_fill(m, P * K, 16.0f, 0.0f);
    m->w.f_a = rw_fill(m, (size_t)D * E, 24.0f, 0.0f);
    m->w.f_b = rw_fill(m, P * D, 24.0f, 0.0f);
    m->w.A_log = rw_fill(m, (size_t)H, 16.0f, 0.0f);
    m->w.dt_bias = rw_fill(m, P, 16.0f, 0.0f);
    m->w.b = rw_fill(m, (size_t)H * E, 24.0f, 0.0f);
    m->w.g = rw_fill(m, P * E, 24.0f, 0.0f);
    m->w.o_norm = rw_fill(m, (size_t)D, 8.0f, 1.0f);
    m->w.o = rw_fill(m, (size_t)E * P, 24.0f, 0.0f);
    m->w.wdt = K3_WF32;
}

static void rw_free(RwModel *m) { for (int i = 0; i < m->nbuf; i++) free(m->buf[i]); }

static int rw_fail(const char *what, const RwModel *m, int T, int n)
{
    fprintf(stderr, "KDA LOG mismatch: %s (H=%d D=%d conv_k=%d T=%d n=%d)\n", what,
            m->c.kda_heads, m->c.kda_head_dim, m->c.conv_k, T, n);
    return 1;
}

/* One geometry. Returns nonzero on any failure and adds to *checks. */
static int log_suite(int E, int H, int D, int K, unsigned *checks)
{
    RwModel m;
    rw_model(&m, E, H, D, K);
    const K3Cfg *c = &m.c;
    const int P = H * D, hist = K - 1, TMAX = 7, WARM = 5, TAIL = 3;
    const size_t ns = k3_kda_state_floats(c);
    const size_t nt = (size_t)(WARM + TMAX + TAIL);
    float *x = rw_fill(&m, nt * E, 32.0f, 0.0f);
    float *sc = (float *)malloc(k3_kda_scratch(c, (int)nt) * sizeof(float));
    float *entry = (float *)calloc(ns, sizeof(float));
    float *sa = (float *)malloc(ns * sizeof(float));
    float *sb = (float *)malloc(ns * sizeof(float));
    float *work = (float *)malloc(ns * sizeof(float));
    float *oa = (float *)malloc(nt * E * sizeof(float));
    float *ob = (float *)malloc(nt * E * sizeof(float));
    float *rows = (float *)malloc((size_t)TMAX * k3_kda_log_row(c) * sizeof(float));
    if (!sc || !entry || !sa || !sb || !work || !oa || !ob || !rows) abort();
    if (ns != (size_t)P * D + (size_t)3 * P * hist) return rw_fail("state size", &m, 0, 0);
    int bad = 0;

    /* A non-trivial entry state: WARM real positions, so S has structure and every
     * history slot holds a genuine earlier input, including ones a short commit keeps. */
    k3_kda_layer(oa, x, &m.w, c, WARM, entry, sc);
    const float *xs = x + (size_t)WARM * E;

    for (int T = 1; T <= TMAX; T++) {
        for (int cap = T - 2 > 0 ? T - 2 : 0; cap <= T; cap++) {
            for (int n = 0; n <= cap; n++) {
                /* The logged sweep computes exactly what the plain one does, and leaves
                 * the carried state exactly as it found it. */
                K3KdaLog lg = { rows, work, cap, 0 };
                memcpy(sa, entry, ns * sizeof(float));
                memcpy(sb, entry, ns * sizeof(float));
                k3_kda_layer_log(oa, xs, &m.w, c, T, sa, sc, &lg);
                k3_kda_layer(ob, xs, &m.w, c, T, sb, sc);
                if (memcmp(oa, ob, (size_t)T * E * sizeof(float)))
                    bad |= rw_fail("a log changed the layer's output", &m, T, n);
                if (memcmp(sa, entry, ns * sizeof(float)))
                    bad |= rw_fail("a logged call wrote the carried state", &m, T, n);

                /* Commit n, against feeding the first n positions alone. */
                k3_kda_advance(sa, &lg, n, c);
                memcpy(sb, entry, ns * sizeof(float));
                if (n > 0) k3_kda_layer(ob, xs, &m.w, c, n, sb, sc);
                if (memcmp(sa, sb, (size_t)P * D * sizeof(float)))
                    bad |= rw_fail("recurrent matrix", &m, T, n);
                if (memcmp(sa + (size_t)P * D, sb + (size_t)P * D,
                           (size_t)3 * P * hist * sizeof(float)))
                    bad |= rw_fail("ShortConv history", &m, T, n);

                /* And the committed state is live: continuing from it is continuing
                 * from the fresh one, output and state. */
                k3_kda_layer(oa, xs + (size_t)n * E, &m.w, c, TAIL, sa, sc);
                k3_kda_layer(ob, xs + (size_t)n * E, &m.w, c, TAIL, sb, sc);
                if (memcmp(oa, ob, (size_t)TAIL * E * sizeof(float)) ||
                    memcmp(sa, sb, ns * sizeof(float)))
                    bad |= rw_fail("decoding after the commit", &m, T, n);
                (*checks)++;
            }
        }
    }

    /* A log built by several calls: a two-position call, then single positions, as the
     * hybrid draft builds it. Each call must compute what the same call computes in place
     * after the earlier ones, while the carried state stays put. */
    {
        K3KdaLog lg = { rows, work, TMAX, 0 };
        memcpy(sa, entry, ns * sizeof(float));
        memcpy(sb, entry, ns * sizeof(float));
        k3_kda_layer_log(oa, xs, &m.w, c, 2, sa, sc, &lg);
        k3_kda_layer(ob, xs, &m.w, c, 2, sb, sc);
        if (memcmp(oa, ob, (size_t)2 * E * sizeof(float)))
            bad |= rw_fail("first call of a multi-call log", &m, 2, 0);
        for (int r = 2; r < TMAX; r++) {
            lg.row0 = r;
            k3_kda_layer_log(oa, xs + (size_t)r * E, &m.w, c, 1, sa, sc, &lg);
            k3_kda_layer(ob, xs + (size_t)r * E, &m.w, c, 1, sb, sc);
            if (memcmp(oa, ob, (size_t)E * sizeof(float)))
                bad |= rw_fail("later call of a multi-call log", &m, 1, r);
        }
        if (memcmp(sa, entry, ns * sizeof(float)))
            bad |= rw_fail("a multi-call log wrote the carried state", &m, TMAX, 0);
        for (int n = 0; n <= TMAX; n++) {
            memcpy(sa, entry, ns * sizeof(float));
            k3_kda_advance(sa, &lg, n, c);
            memcpy(sb, entry, ns * sizeof(float));
            if (n > 0) k3_kda_layer(ob, xs, &m.w, c, n, sb, sc);
            if (memcmp(sa, sb, ns * sizeof(float)))
                bad |= rw_fail("commit from a multi-call log", &m, TMAX, n);
            (*checks)++;
        }
    }

    /* No carried state: the call starts from the zero state k3_kda_layer assumes. */
    {
        K3KdaLog lg = { rows, work, TMAX, 0 };
        k3_kda_layer_log(oa, xs, &m.w, c, TMAX, NULL, sc, &lg);
        k3_kda_layer(ob, xs, &m.w, c, TMAX, NULL, sc);
        if (memcmp(oa, ob, (size_t)TMAX * E * sizeof(float)))
            bad |= rw_fail("log without a carried state", &m, TMAX, 0);
        for (int n = 0; n <= TMAX; n++) {
            memset(sa, 0, ns * sizeof(float));
            memset(sb, 0, ns * sizeof(float));
            k3_kda_advance(sa, &lg, n, c);
            if (n > 0) k3_kda_layer(ob, xs, &m.w, c, n, sb, sc);
            if (memcmp(sa, sb, ns * sizeof(float)))
                bad |= rw_fail("commit without a carried state", &m, TMAX, n);
            (*checks)++;
        }
    }

    free(sc); free(entry); free(sa); free(sb); free(work); free(oa); free(ob); free(rows);
    rw_free(&m);
    return bad;
}

int main(void)
{
    {
        /* Head widths off and on the SIMD lane count, and every history length that
         * behaves differently: none, shorter than a sweep, and the released conv_k 4.
         * K3's own width (D = 128, H = 96) is not among them: the bit-exactness argument
         * does not depend on D, and the SIMD tile split at that width is covered by the
         * oracle gates, not by this suite. */
        const int geo[][4] = { {24, 3, 8, 4}, {20, 2, 5, 4}, {16, 2, 8, 1},
                               {16, 2, 9, 2}, {40, 4, 16, 4} };
        unsigned checks = 0;
        int bad = 0;
        for (size_t g = 0; g < sizeof geo / sizeof *geo; g++)
            bad |= log_suite(geo[g][0], geo[g][1], geo[g][2], geo[g][3], &checks);
        if (bad) return 1;
        printf("KDA LOG: %u commits bit-identical to feeding the kept prefix alone "
               "(S, ShortConv history, decoding after); logged calls leave the state "
               "untouched\n", checks);
    }

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
