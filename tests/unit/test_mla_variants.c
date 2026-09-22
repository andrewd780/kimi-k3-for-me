/* test_mla_variants.c - the MLA cache variants of benchmarks/mla_variants.h, bit for bit.
 *
 * WHAT THIS PROVES
 *   1. E and L0 are faithful copies. Each case prefills a cache with C positions, then
 *      runs T new tokens through the ENGINE's k3_mla_cached in both layouts and through
 *      every variant, from the same hidden states. The engine's two outputs, E, E+, L0
 *      and L1 must be the same floats (memcmp, not a tolerance), and so must the cache
 *      rows the engine and the variants append.
 *   2. L1 is bitwise identical to E and L0 at every value-row budget: all rows held,
 *      half held (the rest rebuilt a second time), none held. So is E+.
 *   3. The raw scores before the softmax agree bitwise across E, E+, L0 and L1 too, so
 *      the output match is not two errors cancelling.
 *   4. Causal masking across the new positions: with T >= 2, changing the LAST new
 *      token's hidden state must leave the outputs of the earlier new tokens unchanged
 *      bit for bit, in L1 and in A, and must change the last one.
 *   5. Thread-count independence: L1, E+ and A rerun on every available thread give the
 *      bits they gave on one.
 *   6. A is the plain formula: it matches a scalar, loop-by-loop rendering of the
 *      absorbed attention bitwise, so bench_mla's numerical study measures absorption
 *      itself, not an artefact of the vectorised loop. It is NOT equal to E, and the
 *      test only checks that it is close (a wrong W_uk/W_uv split would not be).
 *
 * The geometries are the reference fixture's (4 heads, 24+8, v 16, kv_lora 32), an odd
 * one that leaves a tail in every unrolled or vectorised loop, and the released
 * attention geometry (96 heads, 128+64, v 128, kv_lora 512, q_lora 1536) with the
 * residual width cut to 512 to keep the test fast; the hidden width only enters through
 * the shared projections. bench_mla's `numerics` mode runs the full 7168 width.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3.h"
#include "../../benchmarks/mla_variants.h"

static int failures = 0, checks = 0;
/* Threads available at start-up. The comparisons run on one thread, and the threaded
 * variants are rerun on all of these: on a loaded machine every OpenMP fork and join
 * costs milliseconds, and the fixture geometry makes thousands of tiny ones. */
static int g_threads = 1;

#define CHECK(cond, ...)                                                             \
    do {                                                                             \
        checks++;                                                                    \
        if (!(cond)) {                                                               \
            failures++;                                                              \
            printf("  FAIL (line %d): ", __LINE__);                                  \
            printf(__VA_ARGS__);                                                     \
            printf("\n");                                                            \
        }                                                                            \
    } while (0)

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

static int same(const void *a, const void *b, size_t bytes) { return !memcmp(a, b, bytes); }

/* Relative L2 distance, for the one comparison that is not exact. */
static double rel_l2(const float *a, const float *b, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static void set_threads(int n)
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

static int max_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* ------------------------------------------------ scalar absorbed reference ---- */
/* The absorbed attention written as plainly as possible: per query token, per head,
 * one loop per formula. mla_attend_A must reproduce this bit for bit. */
static void attend_A_scalar(float *acc, const float *q, int T, int C, const MlaCache *k,
                            const K3MlaW *w, const K3Cfg *c)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, kvl = c->kv_lora, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    float *qa = (float *)xmalloc((size_t)kvl * sizeof(float));
    float *sc = (float *)xmalloc((size_t)N * sizeof(float));
    float *u  = (float *)xmalloc((size_t)kvl * sizeof(float));
    for (int t = 0; t < T; t++) {
        const int p = C + t;
        for (int h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            for (int j = 0; j < kvl; j++) {
                double d = 0.0;
                for (int i = 0; i < qn; i++) {
                    const size_t e = ((size_t)h * kvd + i) * kvl + j;
                    const double wv = w->wdt == K3_WBF16
                                          ? (double)k3_bf16f(((const uint16_t *)w->kv_b)[e])
                                          : (double)((const float *)w->kv_b)[e];
                    d += wv * (double)qt[i];
                }
                qa[j] = (float)d;
            }
            for (int s = 0; s <= p; s++) {
                const float *cs = k->kv + (size_t)s * kvl;
                const float *kr = k->rope + (size_t)s * qr;
                double d = 0.0;
                for (int j = 0; j < kvl; j++) d += (double)qa[j] * (double)cs[j];
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                sc[s] = (float)d * scale;
            }
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) if (sc[s] > m) m = sc[s];
            double z = 0.0;
            for (int s = 0; s <= p; s++) { sc[s] = expf(sc[s] - m); z += sc[s]; }
            for (int j = 0; j < kvl; j++) u[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const float pr = (float)(sc[s] / z);
                const float *cs = k->kv + (size_t)s * kvl;
                for (int j = 0; j < kvl; j++) u[j] += pr * cs[j];
            }
            k3_mmw(acc + ((size_t)t * H + h) * vh, u,
                   mla_row(w->kv_b, w->wdt, kvl, (size_t)h * kvd + qn), w->wdt, kvl, vh);
        }
    }
    free(qa); free(sc); free(u);
}

/* ------------------------------------------------------------------ a case ---- */
typedef struct {
    const K3Cfg  *c;
    const K3MlaW *w;
    int C, T, cap;
    float *xpre, *xnew;       /* [C][E], [T][E] */
    MlaCache lat, exp_;       /* prefilled with the C prefix positions */
} Case;

static void cache_alloc(MlaCache *k, const K3Cfg *c, int cap, int latent)
{
    const size_t per = latent ? (size_t)c->kv_lora
                              : (size_t)c->n_heads * (size_t)(c->qk_nope + c->v_head);
    k->cap  = cap;
    k->kv   = (float *)xmalloc((size_t)cap * per * sizeof(float));
    k->rope = (float *)xmalloc((size_t)cap * (size_t)c->qk_rope * sizeof(float));
    /* Poison, so a variant that reads a slot it did not fill cannot pass by luck. */
    memset(k->kv, 0x7F, (size_t)cap * per * sizeof(float));
    memset(k->rope, 0x7F, (size_t)cap * (size_t)c->qk_rope * sizeof(float));
}

static void cache_copy(MlaCache *dst, const MlaCache *src, const K3Cfg *c, int latent)
{
    const size_t per = latent ? (size_t)c->kv_lora
                              : (size_t)c->n_heads * (size_t)(c->qk_nope + c->v_head);
    cache_alloc(dst, c, src->cap, latent);
    memcpy(dst->kv, src->kv, (size_t)src->cap * per * sizeof(float));
    memcpy(dst->rope, src->rope, (size_t)src->cap * (size_t)c->qk_rope * sizeof(float));
}

static void cache_free(MlaCache *k) { free(k->kv); free(k->rope); k->kv = k->rope = NULL; }

/* The prefix, stored the way the engine stores it: the latent layout keeps the post-norm
 * latent, the expanded layout keeps kv_b of it, both keep the rope row. */
static void case_prefill(Case *K)
{
    const K3Cfg *c = K->c;
    const int kvw = c->kv_lora + c->qk_rope, H = c->n_heads;
    const int kvd = c->qk_nope + c->v_head;
    cache_alloc(&K->lat, c, K->cap, 1);
    cache_alloc(&K->exp_, c, K->cap, 0);
    float *ct = (float *)xmalloc((size_t)(K->C ? K->C : 1) * kvw * sizeof(float));
    mla_project_kv(ct, K->xpre, K->w, c, K->C);
    for (int s = 0; s < K->C; s++) {
        const float *cs = ct + (size_t)s * kvw;
        memcpy(K->lat.kv + (size_t)s * c->kv_lora, cs, (size_t)c->kv_lora * sizeof(float));
        memcpy(K->lat.rope + (size_t)s * c->qk_rope, cs + c->kv_lora,
               (size_t)c->qk_rope * sizeof(float));
        memcpy(K->exp_.rope + (size_t)s * c->qk_rope, cs + c->kv_lora,
               (size_t)c->qk_rope * sizeof(float));
        k3_mmw(K->exp_.kv + (size_t)s * H * kvd, cs, K->w->kv_b, K->w->wdt, c->kv_lora,
               H * kvd);
    }
    free(ct);
}

typedef struct {
    float *acc, *out, *probe;   /* [T][H][vh], [T][E], [T][H][N] */
    MlaCache cache;             /* after the call */
} Run;

static void run_free(Run *r) { free(r->acc); free(r->out); free(r->probe); cache_free(&r->cache); }

/* One whole layer in variant v on a private copy of the prefilled cache. */
static void run_variant(Run *r, const Case *K, const float *xnew, int v, int vcap)
{
    const K3Cfg *c = K->c;
    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;
    const int kvw = c->kv_lora + c->qk_rope, E = c->hidden, T = K->T, N = K->C + K->T;
    float *q    = (float *)xmalloc((size_t)T * H * qh * sizeof(float));
    float *ct   = (float *)xmalloc((size_t)T * kvw * sizeof(float));
    float *ql   = (float *)xmalloc((size_t)c->q_lora * sizeof(float));
    float *gbuf = (float *)xmalloc((size_t)H * vh * sizeof(float));
    void  *scr  = xmalloc(mla_scratch_bytes(v, c, T, N, vcap));
    r->acc   = (float *)xmalloc((size_t)T * H * vh * sizeof(float));
    r->out   = (float *)xmalloc((size_t)T * E * sizeof(float));
    r->probe = (float *)xmalloc((size_t)T * H * N * sizeof(float));
    memset(r->probe, 0, (size_t)T * H * N * sizeof(float));
    cache_copy(&r->cache, mla_is_latent(v) ? &K->lat : &K->exp_, c, mla_is_latent(v));

    mla_project(q, ct, xnew, K->w, c, T, ql);
    if (mla_attend(v, r->acc, q, ct, T, K->C, &r->cache, K->w, c, scr, vcap, r->probe)) {
        fprintf(stderr, "variant %s refused the layer\n", MLA_NAME[v]);
        exit(1);
    }
    /* The gate works in place, so keep the pre-gate accumulator for comparison. */
    float *acc2 = (float *)xmalloc((size_t)T * H * vh * sizeof(float));
    memcpy(acc2, r->acc, (size_t)T * H * vh * sizeof(float));
    mla_finish(r->out, acc2, xnew, K->w, c, T, gbuf);
    free(acc2); free(q); free(ct); free(ql); free(gbuf); free(scr);
}

/* The engine itself, in either layout, on a private copy of the prefilled cache. */
static void run_engine(Run *r, const Case *K, int latent)
{
    const K3Cfg *c = K->c;
    const size_t n = k3_mla_scratch_cached(c, K->T, K->cap, 1, latent);
    float *scr = (float *)xmalloc(n * sizeof(float));
    memset(r, 0, sizeof *r);
    r->out = (float *)xmalloc((size_t)K->T * c->hidden * sizeof(float));
    cache_copy(&r->cache, latent ? &K->lat : &K->exp_, c, latent);
    k3_mla_cached(r->out, K->xnew, K->w, c, K->T, scr, r->cache.kv, r->cache.rope, K->C,
                  K->cap, latent);
    free(scr);
}

/* A itself, and the scalar rendering, on the same prefilled latent cache. */
static void run_A_scalar(float *acc, const Case *K, const float *xnew)
{
    const K3Cfg *c = K->c;
    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, kvw = c->kv_lora + c->qk_rope;
    float *q  = (float *)xmalloc((size_t)K->T * H * qh * sizeof(float));
    float *ct = (float *)xmalloc((size_t)K->T * kvw * sizeof(float));
    float *ql = (float *)xmalloc((size_t)c->q_lora * sizeof(float));
    MlaCache k;
    cache_copy(&k, &K->lat, c, 1);
    mla_project(q, ct, xnew, K->w, c, K->T, ql);
    mla_append(&k, 1, ct, K->T, K->C, K->w, c);
    attend_A_scalar(acc, q, K->T, K->C, &k, K->w, c);
    cache_free(&k); free(q); free(ct); free(ql);
}

static int test_case(const char *geom, const K3Cfg *c, const K3MlaW *w, int C, int T,
                     uint64_t seed)
{
    const int f0 = failures, E = c->hidden, H = c->n_heads, vh = c->v_head;
    const int N = C + T, kvd = c->qk_nope + c->v_head;
    Case K;
    memset(&K, 0, sizeof K);
    K.c = c; K.w = w; K.C = C; K.T = T; K.cap = N + 3;   /* slack: cap > N is legal */
    K.xpre = (float *)xmalloc((size_t)(C ? C : 1) * E * sizeof(float));
    K.xnew = (float *)xmalloc((size_t)T * E * sizeof(float));
    mla_fill_normal(K.xpre, (size_t)C * E, 1.0f, seed);
    mla_fill_normal(K.xnew, (size_t)T * E, 1.0f, seed + 1);
    case_prefill(&K);

    const size_t accb = (size_t)T * H * vh * sizeof(float);
    const size_t outb = (size_t)T * E * sizeof(float);
    const size_t prb  = (size_t)T * H * N * sizeof(float);
    const size_t latb = (size_t)T * c->kv_lora * sizeof(float);
    const size_t expb = (size_t)T * H * kvd * sizeof(float);
    const size_t ropb = (size_t)T * c->qk_rope * sizeof(float);

    Run eng_x, eng_l, rE, rEP, rL0, rA;
    run_engine(&eng_x, &K, 0);
    run_engine(&eng_l, &K, 1);
    run_variant(&rE, &K, K.xnew, MLA_E, 0);
    run_variant(&rEP, &K, K.xnew, MLA_EP, 0);
    run_variant(&rL0, &K, K.xnew, MLA_L0, 0);
    run_variant(&rA, &K, K.xnew, MLA_A, 0);

    /* 1. the engine's two layouts, and E, against each other */
    CHECK(same(eng_x.out, eng_l.out, outb), "%s C=%d T=%d: engine expanded != engine latent",
          geom, C, T);
    CHECK(same(rE.out, eng_x.out, outb), "%s C=%d T=%d: E != engine", geom, C, T);
    CHECK(same(rL0.out, eng_l.out, outb), "%s C=%d T=%d: L0 != engine latent", geom, C, T);
    CHECK(same(rEP.out, rE.out, outb) && same(rEP.acc, rE.acc, accb),
          "%s C=%d T=%d: E+ != E", geom, C, T);
    CHECK(same(rL0.acc, rE.acc, accb), "%s C=%d T=%d: L0 acc != E acc", geom, C, T);
    /* the appended rows are the engine's rows */
    CHECK(same(rE.cache.kv + (size_t)C * H * kvd, eng_x.cache.kv + (size_t)C * H * kvd, expb)
              && same(rE.cache.rope + (size_t)C * c->qk_rope,
                      eng_x.cache.rope + (size_t)C * c->qk_rope, ropb),
          "%s C=%d T=%d: E appended different expanded rows", geom, C, T);
    CHECK(same(rL0.cache.kv + (size_t)C * c->kv_lora, eng_l.cache.kv + (size_t)C * c->kv_lora,
               latb)
              && same(rL0.cache.rope + (size_t)C * c->qk_rope,
                      eng_l.cache.rope + (size_t)C * c->qk_rope, ropb),
          "%s C=%d T=%d: latent append differs from the engine's", geom, C, T);
    /* 3. scores agree before the softmax */
    CHECK(same(rEP.probe, rE.probe, prb) && same(rL0.probe, rE.probe, prb),
          "%s C=%d T=%d: raw scores differ between E, E+ and L0", geom, C, T);

    /* 2. L1 at three value-row budgets */
    const int vcaps[3] = {N, N / 2, 0};
    for (int i = 0; i < 3; i++) {
        Run r;
        run_variant(&r, &K, K.xnew, MLA_L1, vcaps[i]);
        CHECK(same(r.out, eng_x.out, outb) && same(r.acc, rE.acc, accb),
              "%s C=%d T=%d: L1 (vcap %d) != E", geom, C, T, vcaps[i]);
        CHECK(same(r.probe, rE.probe, prb), "%s C=%d T=%d: L1 (vcap %d) scores != E",
              geom, C, T, vcaps[i]);
        CHECK(same(r.cache.kv + (size_t)C * c->kv_lora, eng_l.cache.kv + (size_t)C * c->kv_lora,
                   latb),
              "%s C=%d T=%d: L1 appended different latent rows", geom, C, T);
        run_free(&r);
    }

    /* 6. A is the scalar formula, and is close to E but is its own arithmetic */
    {
        float *acc = (float *)xmalloc(accb);
        run_A_scalar(acc, &K, K.xnew);
        CHECK(same(acc, rA.acc, accb), "%s C=%d T=%d: A != scalar absorbed reference", geom,
              C, T);
        free(acc);
        const double d_acc = rel_l2(rA.acc, rE.acc, (size_t)T * H * vh);
        const double d_out = rel_l2(rA.out, rE.out, (size_t)T * E);
        CHECK(d_acc < 1e-4 && d_out < 1e-4, "%s C=%d T=%d: A far from E (acc %.3g, out %.3g)",
              geom, C, T, d_acc, d_out);
    }

    /* 4. causal masking across the new positions */
    if (T >= 2) {
        float *x2 = (float *)xmalloc((size_t)T * E * sizeof(float));
        memcpy(x2, K.xnew, (size_t)T * E * sizeof(float));
        mla_fill_normal(x2 + (size_t)(T - 1) * E, (size_t)E, 1.0f, seed + 99);
        const int vs[2] = {MLA_L1, MLA_A};
        const Run *base[2] = {&rE, &rA};
        for (int i = 0; i < 2; i++) {
            Run r;
            run_variant(&r, &K, x2, vs[i], N);
            const size_t early = (size_t)(T - 1) * E * sizeof(float);
            CHECK(same(r.out, base[i]->out, early),
                  "%s C=%d T=%d: %s leaks the last new token into earlier ones", geom, C, T,
                  MLA_NAME[vs[i]]);
            CHECK(!same(r.out + (size_t)(T - 1) * E, base[i]->out + (size_t)(T - 1) * E,
                        (size_t)E * sizeof(float)),
                  "%s C=%d T=%d: %s ignored the last new token", geom, C, T, MLA_NAME[vs[i]]);
            run_free(&r);
        }
        free(x2);
    }

    /* 5. thread-count independence: everything above ran on one thread; the variants
     *    that split work across threads run again on all of them. */
    if (g_threads > 1) {
        set_threads(g_threads);
        const int vs[3] = {MLA_L1, MLA_EP, MLA_A};
        const Run *base[3] = {&rE, &rEP, &rA};
        for (int i = 0; i < 3; i++) {
            Run r;
            run_variant(&r, &K, K.xnew, vs[i], N / 2);
            CHECK(same(r.acc, base[i]->acc, accb) && same(r.out, base[i]->out, outb),
                  "%s C=%d T=%d: %s changes with the thread count", geom, C, T,
                  MLA_NAME[vs[i]]);
            run_free(&r);
        }
        set_threads(1);
    }

    run_free(&eng_x); run_free(&eng_l); run_free(&rE); run_free(&rEP); run_free(&rL0);
    run_free(&rA);
    cache_free(&K.lat); cache_free(&K.exp_);
    free(K.xpre); free(K.xnew);
    printf("  %-22s C=%-4d T=%d  %s\n", geom, C, T, failures == f0 ? "ok" : "FAILED");
    return failures == f0;
}

int main(void)
{
    struct { int C, T; } tiny[] = {{0, 1}, {0, 3}, {1, 1}, {5, 2}, {7, 5}, {31, 4},
                                   {64, 1}, {200, 5}, {257, 3}};
    struct { int C, T; } odd[] = {{0, 2}, {3, 5}, {13, 3}, {50, 7}};
    struct { int C, T; } real[] = {{0, 1}, {1, 5}, {6, 2}, {33, 5}, {40, 1}};
    int ncase = 0, nok = 0;

    g_threads = max_threads();
    set_threads(1);
    printf("MLA cache variants: E, E+, L0, L1 bitwise against the engine; A against its\n"
           "scalar formula (one thread, threaded variants rerun on %d)\n", g_threads);

    for (int wdt = 0; wdt < 2; wdt++) {           /* fp32 and bf16 kv_b */
        K3Cfg c;
        MlaSynth S;
        mla_cfg_tiny(&c);
        const int t = wdt ? K3_WBF16 : K3_WF32;
        if (mla_synth_layer(&S, &c, t, 0.1f, 0.1f, 11 + (uint64_t)wdt, 1)) return 1;
        for (size_t i = 0; i < sizeof tiny / sizeof tiny[0]; i++, ncase++)
            nok += test_case(wdt ? "fixture geometry bf16" : "fixture geometry fp32", &c, &S.w,
                             tiny[i].C, tiny[i].T, 1000 + 17 * (uint64_t)i);
        mla_synth_free(&S);
    }
    {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_tiny(&c);
        c.hidden = 72; c.n_heads = 3; c.q_lora = 48; c.kv_lora = 40;
        c.qk_nope = 20; c.qk_rope = 6; c.v_head = 12;
        if (mla_synth_layer(&S, &c, K3_WBF16, 0.1f, 0.1f, 23, 1)) return 1;
        for (size_t i = 0; i < sizeof odd / sizeof odd[0]; i++, ncase++)
            nok += test_case("odd geometry bf16", &c, &S.w, odd[i].C, odd[i].T,
                             2000 + 17 * (uint64_t)i);
        mla_synth_free(&S);
    }
    {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_k3(&c, 512);
        if (mla_synth_layer(&S, &c, K3_WBF16, 0.02f, 0.1f, 37, 1)) return 1;
        for (size_t i = 0; i < sizeof real / sizeof real[0]; i++, ncase++)
            nok += test_case("K3 attention bf16", &c, &S.w, real[i].C, real[i].T,
                             3000 + 17 * (uint64_t)i);
        mla_synth_free(&S);
    }

    printf("%d/%d cases, %d checks, %d failed\n", nok, ncase, checks, failures);
    if (failures) {
        printf("MLA VARIANTS: FAIL\n");
        return 1;
    }
    printf("MLA VARIANTS: L1 AND E+ BITWISE IDENTICAL TO E, L0 AND THE ENGINE\n");
    return 0;
}
