/* test_mla_variants.c - the MLA cache variants of benchmarks/mla_variants.h, bit for bit.
 *
 * WHAT THIS PROVES
 *   1. E and L0 are faithful copies. Each case prefills a cache with C positions, then
 *      runs T new tokens through the ENGINE's k3_mla_cached in both layouts and through
 *      every variant, from the same hidden states. The engine's two outputs, E, E+, L0
 *      and L1 must be the same floats (memcmp, not a tolerance), and so must the cache
 *      rows the engine and the variants append and the pre-gate accumulators.
 *   2. L1 is bitwise identical to E and L0 at every value-row budget: all rows held,
 *      half held (the rest rebuilt a second time), none held. So is E+.
 *   3. The raw scores before the softmax, every softmax normaliser z and every
 *      probability quotient e/z agree bitwise too: between the engine's two layouts,
 *      between E and the expanded engine, between L0 and the latent engine, and between
 *      E, E+, L0 and L1. The engine's are read through its trace hook (k3_mla_trace in
 *      k3.h), the variants' through mla_probe_row, mla_zprobe and mla_qprobe. So the
 *      output match is not two errors cancelling, and the doubles the output rounds
 *      away (see 9) are held to the engine's own, not only to a copy of it.
 *   4. Causal masking across the new positions: with T >= 2, changing the LAST new
 *      token's hidden state must leave the outputs of the earlier new tokens unchanged
 *      bit for bit, in L1 and in A, and must change the last one.
 *   5. Thread-count independence: L1, E+ and A rerun on every available thread give the
 *      bits they gave on one, on the cases of up to 48 positions (see test_case).
 *   6. A is the plain formula: it matches a scalar, loop-by-loop rendering of the
 *      absorbed attention bitwise, so bench_mla's numerical study measures absorption
 *      itself, not an artefact of the vectorised loop. It is NOT equal to E, and the
 *      test only checks that it is close (a wrong W_uk/W_uv split would not be).
 *   7. The kv_b applications each variant makes are exactly mla_rebuilds(): T for E
 *      and E+, 2T(C+1) + T(T-1) for L0, 2N - vcap for L1, none for A. The prefill-
 *      shaped case C=0, T=256 is where L0's count is quadratic: 65,792 against L1's 256.
 *   8. All of the above again on CANCELLING layers (see synth_cancelling), built so that
 *      summing any score chain in another order changes its float. On the ordinary
 *      layers it almost never would, so without these the memcmp gates could not see a
 *      reordered chain at all. The order witness at the end prints both rates.
 *
 * The geometries are the reference fixture's (4 heads, 24+8, v 16, kv_lora 32), an odd
 * one that leaves a tail in every unrolled or vectorised loop, and the released
 * attention geometry (96 heads, 128+64, v 128, kv_lora 512, q_lora 1536) with the
 * residual width cut to 512 to keep the test fast; the hidden width only enters through
 * the shared projections. bench_mla's `numerics` mode runs the full 7168 width.

 *   9. Why 3. records z and e/z as doubles. z is a double sum of positive terms that
 *      reaches the output only as p = (float)(e / z); no input short of ~2^29 positions
 *      moves p by a float ulp, so a reordered z would pass every other check here, in
 *      a variant or in the engine. Compared as a double, a reorder can show -- but only
 *      on SHARP rows: while every e = expf(s - max) is above ~2^-29, the terms are floats
 *      whose sum is exact in double in any order. The sharp layers (query norm scaled
 *      by 16, an exact power of two) give score ranges of tens of nats, as the real
 *      model's sharper heads do, and there the order of z is visible. The quotient is
 *      the same story without the special layer: formed as e * (1 / z) it rounds twice
 *      and differs from e / z in about a fifth of all quotients, yet p almost never
 *      moves. The order witness prints both rates.
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
/* Order witness totals, [0] ordinary layers, [1] cancelling layers: scores and value sums
 * re-summed in other orders, and how many of them rounded differently. */
typedef struct {
    double scores, rev, split, lanes;   /* score chains: count, and differing floats */
    double vals, vrev;                  /* value sums of 3+ terms, and differing     */
    double zrows, zrev;                 /* normalisers of 3+ terms, and differing    */
    double quots, qrecip;               /* quotients e/z, and differing as e*(1/z)   */
} Witness;
enum { LAYER_ORDINARY, LAYER_CANCELLING, LAYER_SHARP, LAYER_KINDS };
static Witness g_wit[LAYER_KINDS];
/* Threads available at start-up. The comparisons run on one thread, and the threaded
 * variants are rerun on all of these: on a loaded machine every OpenMP fork and join
 * costs milliseconds, and the fixture geometry makes thousands of tiny ones. So only
 * the attention step itself (mla_attend, which is what differs between variants) runs
 * on g_attend_threads; the projections, gate and o_proj around it stay on one thread,
 * as they are the same calls in every variant and their kernels' thread independence is
 * test_ops's to prove. */
static int g_threads = 1, g_attend_threads = 1;

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
static void attend_A_scalar(float *acc, double *zs, double *qs, const float *q, int T,
                            int C, const MlaCache *k, const K3MlaW *w, const K3Cfg *c)
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
            zs[(size_t)t * H + h] = z;
            for (int j = 0; j < kvl; j++) u[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const double pq = sc[s] / z;
                qs[((size_t)t * H + h) * N + s] = pq;
                const float pr = (float)pq;
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
    double *z;                  /* [T][H] softmax normalisers */
    double *quot;               /* [T][H][N] probability quotients e/z, before rounding */
    MlaCache cache;             /* after the call */
    unsigned long long kvb;     /* kv_b applications the call made */
} Run;

static void run_free(Run *r)
{
    free(r->acc); free(r->out); free(r->probe); free(r->z); free(r->quot);
    cache_free(&r->cache);
}

/* The recorded intermediates of a run, zeroed so that a row nobody wrote compares equal
 * only to another row nobody wrote. */
static void run_alloc_trace(Run *r, int T, int H, int vh, int N)
{
    r->acc   = (float *)xmalloc((size_t)T * H * vh * sizeof(float));
    r->probe = (float *)xmalloc((size_t)T * H * N * sizeof(float));
    r->z     = (double *)xmalloc((size_t)T * H * sizeof(double));
    r->quot  = (double *)xmalloc((size_t)T * H * N * sizeof(double));
    memset(r->acc, 0, (size_t)T * H * vh * sizeof(float));
    memset(r->probe, 0, (size_t)T * H * N * sizeof(float));
    memset(r->z, 0, (size_t)T * H * sizeof(double));
    memset(r->quot, 0, (size_t)T * H * N * sizeof(double));
}

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
    run_alloc_trace(r, T, H, vh, N);
    r->out   = (float *)xmalloc((size_t)T * E * sizeof(float));
    cache_copy(&r->cache, mla_is_latent(v) ? &K->lat : &K->exp_, c, mla_is_latent(v));

    mla_project(q, ct, xnew, K->w, c, T, ql);
    const unsigned long long k0 = mla_kvb_calls;
    mla_zprobe = r->z;
    mla_qprobe = r->quot;
    set_threads(g_attend_threads);
    if (mla_attend(v, r->acc, q, ct, T, K->C, &r->cache, K->w, c, scr, vcap, r->probe)) {
        fprintf(stderr, "variant %s refused the layer\n", MLA_NAME[v]);
        exit(1);
    }
    set_threads(1);
    mla_zprobe = NULL;
    mla_qprobe = NULL;
    r->kvb = mla_kvb_calls - k0;
    /* The gate works in place, so keep the pre-gate accumulator for comparison. */
    float *acc2 = (float *)xmalloc((size_t)T * H * vh * sizeof(float));
    memcpy(acc2, r->acc, (size_t)T * H * vh * sizeof(float));
    mla_finish(r->out, acc2, xnew, K->w, c, T, gbuf);
    free(acc2); free(q); free(ct); free(ql); free(gbuf); free(scr);
}

/* The engine itself, in either layout, on a private copy of the prefilled cache, with
 * its trace hook (k3_mla_trace) recording the same intermediates the variants record:
 * the raw scores, every softmax normaliser and probability quotient, and the pre-gate
 * accumulator. Those are what the engine's output cannot show; see k3.h. */
static void run_engine(Run *r, const Case *K, int latent)
{
    const K3Cfg *c = K->c;
    const int N = K->C + K->T;
    const size_t n = k3_mla_scratch_cached(c, K->T, K->cap, 1, latent);
    float *scr = (float *)xmalloc(n * sizeof(float));
    memset(r, 0, sizeof *r);
    run_alloc_trace(r, K->T, c->n_heads, c->v_head, N);
    r->out = (float *)xmalloc((size_t)K->T * c->hidden * sizeof(float));
    cache_copy(&r->cache, latent ? &K->lat : &K->exp_, c, latent);
    K3MlaTrace tr = {N, r->probe, r->z, r->quot, r->acc};
    k3_mla_trace = &tr;
    k3_mla_cached(r->out, K->xnew, K->w, c, K->T, scr, r->cache.kv, r->cache.rope, K->C,
                  K->cap, latent);
    k3_mla_trace = NULL;
    free(scr);
}

/* A itself, and the scalar rendering, on the same prefilled latent cache. */
static void run_A_scalar(float *acc, double *zs, double *qs, const Case *K,
                         const float *xnew)
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
    attend_A_scalar(acc, zs, qs, q, K->T, K->C, &k, K->w, c);
    cache_free(&k); free(q); free(ct); free(ql);
}

static int fsame(float a, float b) { return !memcmp(&a, &b, sizeof a); }

/* The order witness. Re-derives every score of the call from q and the cache E filled,
 * in E's order, which must reproduce E's probe bit for bit, and in three others:
 * reversed, nope and rope summed separately and then added, and two lanes (alternate
 * terms) added at the end, the shapes a vectorised or restructured loop would take.
 * Then re-sums every value element from E's probabilities, s ascending (must equal E's
 * accumulator) and s descending. It counts how often the other orders round to a
 * different float, i.e. how often the memcmp gates in test_case could see such a
 * reordering at all. Returns 1 if E's own order was reproduced exactly. */
static int witness(Witness *W, const Case *K, const Run *rE)
{
    const K3Cfg *c = K->c;
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, T = K->T, C = K->C, N = C + T;
    const int kvw = c->kv_lora + qr;
    const float scale = 1.0f / sqrtf((float)qh);
    float *q  = (float *)xmalloc((size_t)T * H * qh * sizeof(float));
    float *ct = (float *)xmalloc((size_t)T * kvw * sizeof(float));
    float *ql = (float *)xmalloc((size_t)c->q_lora * sizeof(float));
    float *pr = (float *)xmalloc((size_t)N * sizeof(float));
    double *term = (double *)xmalloc((size_t)qh * sizeof(double));
    int ok = 1;
    mla_project(q, ct, K->xnew, K->w, c, T, ql);
    for (int t = 0; t < T; t++) {
        const int p = C + t;
        for (int h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            const float *probe = rE->probe + ((size_t)t * H + h) * N;
            for (int s = 0; s <= p; s++) {
                const float *ks = rE->cache.kv + (size_t)s * H * kvd + (size_t)h * kvd;
                const float *kr = rE->cache.rope + (size_t)s * qr;
                for (int i = 0; i < qn; i++) term[i] = (double)qt[i] * (double)ks[i];
                for (int i = 0; i < qr; i++) term[qn + i] = (double)qt[qn + i] * (double)kr[i];
                double fwd = 0.0, rev = 0.0, dn = 0.0, dr = 0.0, l0 = 0.0, l1 = 0.0;
                for (int i = 0; i < qh; i++) fwd += term[i];
                for (int i = qh - 1; i >= 0; i--) rev += term[i];
                for (int i = 0; i < qn; i++) dn += term[i];
                for (int i = qn; i < qh; i++) dr += term[i];
                for (int i = 0; i < qh; i++) { if (i & 1) l1 += term[i]; else l0 += term[i]; }
                const float f = (float)fwd * scale;
                ok &= fsame(f, probe[s]);
                W->scores += 1.0;
                W->rev   += !fsame((float)rev * scale, f);
                W->split += !fsame((float)(dn + dr) * scale, f);
                W->lanes += !fsame((float)(l0 + l1) * scale, f);
            }
            /* the normaliser, E's order and reversed */
            {
                float m = -INFINITY;
                for (int s = 0; s <= p; s++) if (probe[s] > m) m = probe[s];
                double zf = 0.0, zr = 0.0;
                for (int s = 0; s <= p; s++) zf += expf(probe[s] - m);
                for (int s = p; s >= 0; s--) zr += expf(probe[s] - m);
                ok &= !memcmp(&zf, rE->z + (size_t)t * H + h, sizeof zf);
                if (p >= 2) { W->zrows += 1.0; W->zrev += zr != zf; }
                /* the quotients, e / z as E forms them and as e * (1 / z) */
                const double *qe = rE->quot + ((size_t)t * H + h) * N;
                const double rz = 1.0 / zf;
                for (int s = 0; s <= p; s++) {
                    const float e = expf(probe[s] - m);
                    const double qf = e / zf;
                    ok &= !memcmp(&qf, qe + s, sizeof qf);
                    W->quots += 1.0;
                    W->qrecip += e * rz != qf;
                }
            }
            memcpy(pr, probe, (size_t)(p + 1) * sizeof(float));
            mla_softmax_row(pr, p, NULL);
            const float *acc = rE->acc + ((size_t)t * H + h) * vh;
            for (int j = 0; j < vh; j++) {
                float up = 0.0f, dn2 = 0.0f;
                for (int s = 0; s <= p; s++)
                    up += pr[s] * rE->cache.kv[(size_t)s * H * kvd + (size_t)h * kvd + qn + j];
                for (int s = p; s >= 0; s--)
                    dn2 += pr[s] * rE->cache.kv[(size_t)s * H * kvd + (size_t)h * kvd + qn + j];
                ok &= fsame(up, acc[j]);
                if (p >= 2) {            /* two terms commute exactly; three need not */
                    W->vals += 1.0;
                    W->vrev += !fsame(dn2, up);
                }
            }
        }
    }
    free(q); free(ct); free(ql); free(pr); free(term);
    return ok;
}

/* kv_b applications the call made, against the closed form. */
static void check_count(const Run *r, int v, int C, int T, int vcap, const char *geom)
{
    const double want = mla_rebuilds(v, T, C, vcap);
    CHECK((double)r->kvb == want, "%s C=%d T=%d: %s (vcap %d) made %llu kv_b applications, "
          "mla_rebuilds says %.0f", geom, C, T, MLA_NAME[v], vcap, r->kvb, want);
}

/* kind is the layer's LAYER_*: its order witness is added to that kind's totals, and on a
 * cancelling layer A's closeness to E is not checked (there A's own order moves every
 * score by far more than on a real layer; A is still held to its scalar formula). */
static int test_case(const char *geom, const K3Cfg *c, const K3MlaW *w, int C, int T,
                     uint64_t seed, int kind)
{
    const int cancelling = kind == LAYER_CANCELLING;
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
    const size_t zb   = (size_t)T * H * sizeof(double);
    const size_t qb   = (size_t)T * H * N * sizeof(double);
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

    /* 1. the engine's two layouts, and E and L0, against each other: outputs, and the
     *    engine's traced accumulators (3. has its scores, normalisers and quotients) */
    CHECK(same(eng_x.out, eng_l.out, outb) && same(eng_x.acc, eng_l.acc, accb),
          "%s C=%d T=%d: engine expanded != engine latent", geom, C, T);
    CHECK(same(rE.out, eng_x.out, outb) && same(rE.acc, eng_x.acc, accb),
          "%s C=%d T=%d: E != engine", geom, C, T);
    CHECK(same(rL0.out, eng_l.out, outb) && same(rL0.acc, eng_l.acc, accb),
          "%s C=%d T=%d: L0 != engine latent", geom, C, T);
    CHECK(same(rEP.out, rE.out, outb) && same(rEP.acc, rE.acc, accb),
          "%s C=%d T=%d: E+ != E", geom, C, T);
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
    /* 3. scores agree before the softmax, and so do the softmax normalisers and the
     *    probability quotients, first between the engine's own two layouts, then between
     *    each copy and the engine layout it copies, then between the variants and E */
    CHECK(same(eng_l.probe, eng_x.probe, prb) && same(eng_l.z, eng_x.z, zb)
              && same(eng_l.quot, eng_x.quot, qb),
          "%s C=%d T=%d: the engine's two layouts differ in scores, normalisers or "
          "quotients", geom, C, T);
    CHECK(same(rE.probe, eng_x.probe, prb) && same(rE.z, eng_x.z, zb)
              && same(rE.quot, eng_x.quot, qb),
          "%s C=%d T=%d: E's scores, normalisers or quotients != the engine's", geom, C,
          T);
    CHECK(same(rL0.probe, eng_l.probe, prb) && same(rL0.z, eng_l.z, zb)
              && same(rL0.quot, eng_l.quot, qb),
          "%s C=%d T=%d: L0's scores, normalisers or quotients != the latent "
          "engine's", geom, C, T);
    CHECK(same(rEP.probe, rE.probe, prb) && same(rL0.probe, rE.probe, prb),
          "%s C=%d T=%d: raw scores differ between E, E+ and L0", geom, C, T);
    CHECK(same(rEP.z, rE.z, zb) && same(rL0.z, rE.z, zb),
          "%s C=%d T=%d: softmax normalisers differ between E, E+ and L0", geom, C, T);
    CHECK(same(rEP.quot, rE.quot, qb) && same(rL0.quot, rE.quot, qb),
          "%s C=%d T=%d: probability quotients differ between E, E+ and L0", geom, C, T);
    /* 7. kv_b applications */
    check_count(&rE, MLA_E, C, T, 0, geom);
    check_count(&rEP, MLA_EP, C, T, 0, geom);
    check_count(&rL0, MLA_L0, C, T, 0, geom);
    check_count(&rA, MLA_A, C, T, 0, geom);
    /* 8. how often a reordering would have shown on these inputs */
    {
        Witness wk = {0};
        CHECK(witness(&wk, &K, &rE), "%s C=%d T=%d: the witness did not reproduce E's order",
              geom, C, T);
        Witness *acc = &g_wit[kind];
        acc->scores += wk.scores; acc->rev += wk.rev; acc->split += wk.split;
        acc->lanes += wk.lanes; acc->vals += wk.vals; acc->vrev += wk.vrev;
        acc->zrows += wk.zrows; acc->zrev += wk.zrev;
        acc->quots += wk.quots; acc->qrecip += wk.qrecip;
    }

    /* 2. L1 at three value-row budgets */
    const int vcaps[3] = {N, N / 2, 0};
    for (int i = 0; i < 3; i++) {
        Run r;
        run_variant(&r, &K, K.xnew, MLA_L1, vcaps[i]);
        CHECK(same(r.out, eng_x.out, outb) && same(r.acc, rE.acc, accb),
              "%s C=%d T=%d: L1 (vcap %d) != E", geom, C, T, vcaps[i]);
        CHECK(same(r.probe, rE.probe, prb) && same(r.z, rE.z, zb)
                  && same(r.quot, rE.quot, qb),
              "%s C=%d T=%d: L1 (vcap %d) scores, normalisers or quotients != E", geom, C,
              T, vcaps[i]);
        CHECK(same(r.cache.kv + (size_t)C * c->kv_lora, eng_l.cache.kv + (size_t)C * c->kv_lora,
                   latb),
              "%s C=%d T=%d: L1 appended different latent rows", geom, C, T);
        check_count(&r, MLA_L1, C, T, vcaps[i], geom);
        run_free(&r);
    }

    /* 6. A is the scalar formula, and is close to E but is its own arithmetic */
    {
        float *acc = (float *)xmalloc(accb);
        double *zs = (double *)xmalloc(zb);
        double *qs = (double *)xmalloc(qb);
        memset(qs, 0, qb);
        run_A_scalar(acc, zs, qs, &K, K.xnew);
        CHECK(same(acc, rA.acc, accb) && same(zs, rA.z, zb) && same(qs, rA.quot, qb),
              "%s C=%d T=%d: A != scalar absorbed reference", geom, C, T);
        free(acc); free(zs); free(qs);
        const double d_acc = rel_l2(rA.acc, rE.acc, (size_t)T * H * vh);
        const double d_out = rel_l2(rA.out, rE.out, (size_t)T * E);
        CHECK(cancelling || (d_acc < 1e-4 && d_out < 1e-4),
              "%s C=%d T=%d: A far from E (acc %.3g, out %.3g)", geom, C, T, d_acc, d_out);
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
     *    that split work across threads run again on all of them, on the cases of up to
     *    48 positions. Those include three blocks of L1 and A (so A's unsynchronised
     *    block loop is exercised), the 96 heads of K3's geometry and every kind of
     *    layer; the larger cases take no path they do not, and each would add hundreds
     *    of fork/joins of the tiny kv_b kernel, which a loaded machine turns into
     *    seconds (~10 ms apiece measured at load 7 on four cores). */
    if (g_threads > 1 && N <= 48) {
        g_attend_threads = g_threads;
        const int vs[3] = {MLA_L1, MLA_EP, MLA_A};
        const Run *base[3] = {&rE, &rEP, &rA};
        for (int i = 0; i < 3; i++) {
            Run r;
            run_variant(&r, &K, K.xnew, vs[i], N / 2);
            CHECK(same(r.acc, base[i]->acc, accb) && same(r.out, base[i]->out, outb)
                      && same(r.z, base[i]->z, zb) && same(r.quot, base[i]->quot, qb),
                  "%s C=%d T=%d: %s changes with the thread count", geom, C, T,
                  MLA_NAME[vs[i]]);
            run_free(&r);
        }
        g_attend_threads = 1;
    }

    run_free(&eng_x); run_free(&eng_l); run_free(&rE); run_free(&rEP); run_free(&rL0);
    run_free(&rA);
    cache_free(&K.lat); cache_free(&K.exp_);
    free(K.xpre); free(K.xnew);
    printf("  %-22s C=%-4d T=%d  %s\n", geom, C, T, failures == f0 ? "ok" : "FAILED");
    return failures == f0;
}

/* ------------------------------------------------------- a cancelling layer ---- */
/* WHY THE ORDINARY LAYERS ARE NOT ENOUGH
 *   Every score is a double chain rounded to float once, at the end. With terms of one
 *   size, the same terms summed in another order differ by ~1e-16 relative and round to
 *   the same float almost always, so a variant that reordered its chain would pass every
 *   memcmp in test_case. The order witness measures this: on the ordinary layers only a
 *   few percent of reordered chains change their float.
 *
 * WHAT THIS LAYER DOES INSTEAD
 *   Every chain CANCELS. Huge terms, in exactly negated pairs, arrive at shuffled points
 *   among ordinary ones; the partial sums swing out to between 2^30 and 2^48, where one
 *   double ulp is 2^-22 to 2^-4, and every ordinary term added out there loses bits that
 *   depend on the partial sum it met. The exact sum is the ordinary terms' sum, O(1), so
 *   those lost bits are many float ulps: another order meets other partial sums and
 *   rounds differently, nearly always (the witness says how nearly). The pairs come in
 *   different sizes, 2^30 to 2^44 in the latent and 2^44 to 2^47 in the rope slot, so
 *   that orders which open and close the same pairs still round at different grains.
 *
 * BUILT FROM THE ENGINE'S OWN PROJECTIONS
 *   The engine forms q and the latent from x itself, so the cancellation is put into the
 *   weights. Every weight involved is 0, 1, a power-of-two scale or an exact negation of
 *   another row, and the fp32 and bf16 kernels reproduce each of those exactly (negation
 *   commutes with round-to-nearest; a single 1.0 in a row makes the output an exact copy
 *   of one input, whatever the reduction tree):
 *   - kv_a holds PAIRS of latent rows, one the negation of the other, whose rmsnorm
 *     weights are the same huge float: their latent values are exactly c and -c. It
 *     holds pairs of rope rows too, scaled by a power of two and negated (the rope slot
 *     is not normed).
 *   - q_b and kv_b are SELECTION matrices: a single 1.0 per row, so every q, k_nope and
 *     v element is an exact copy of one q_lora or latent element. Head h's key rows pick
 *     both members of every huge pair, in slots shuffled per head among ordinary ones,
 *     and its query rows for those two slots copy the same q_lora element, so the
 *     products are Q*c and Q*(-c). Rope pairs likewise, one q_lora element per pair.
 *   - A absorbs the same pairs (q_abs is Q at both latent indices of a pair and 0 is
 *     never involved), so its chain over the latent, in latent-index order, cancels too.
 *   - Value rows copy huge and ordinary latent elements alike, so the float value sums
 *     carry huge terms of both signs across positions.
 *   Ordinary elements are N(0, sd) weights as in mla_synth_layer. Returns 0/-1. */
static int synth_cancelling(MlaSynth *S, const K3Cfg *c, int wdt, float sd, uint64_t seed)
{
    const int E = c->hidden, H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope;
    const int vh = c->v_head, kvl = c->kv_lora, ql = c->q_lora;
    const int qh = qn + qr, kvd = qn + vh, kvw = kvl + qr;
    int np = (qn - 1) / 4 < kvl / 4 ? (qn - 1) / 4 : kvl / 4;   /* huge latent pairs */
    if (np > 8) np = 8;
    int nr = qr / 4 > 0 ? qr / 4 : (qr >= 2);          /* huge rope pairs */
    if (nr > 4) nr = 4;
    if (np < 1 || H > MLA_MAX_HEADS) return -1;
    const size_t rows[6] = {(size_t)ql, (size_t)H * qh, (size_t)kvw, (size_t)H * kvd,
                            (size_t)E, (size_t)H * vh};
    const size_t cols[6] = {(size_t)E, (size_t)ql, (size_t)E, (size_t)kvl, (size_t)H * vh,
                            (size_t)E};
    float *m[6] = {0};
    int *lat_role = (int *)xmalloc((size_t)kvl * sizeof(int));   /* -1 ordinary, else pair */
    int *rope_role = (int *)xmalloc((size_t)(qr ? qr : 1) * sizeof(int));
    int *perm = (int *)xmalloc((size_t)(qn > kvl ? qn : kvl) * sizeof(int));
    int *ord = (int *)xmalloc((size_t)kvl * sizeof(int));
    int *pl = (int *)xmalloc((size_t)np * 2 * sizeof(int)); /* latent index of pair k, +/- */
    int *pr = (int *)xmalloc((size_t)(nr ? nr : 1) * 2 * sizeof(int));
    MlaRng r;
    mla_rng_seed(&r, seed);
    memset(S, 0, sizeof *S);
    for (int i = 0; i < 6; i++) {
        m[i] = (float *)calloc(rows[i] * cols[i], sizeof(float));
        if (!m[i]) goto fail;
    }
    /* q_a, o and g are ordinary; kv_a's rows are ordinary until the pairs overwrite them */
    mla_fill_normal(m[0], rows[0] * cols[0], sd, seed + 1);
    mla_fill_normal(m[2], rows[2] * cols[2], sd, seed + 3);
    mla_fill_normal(m[4], rows[4] * cols[4], sd, seed + 5);
    mla_fill_normal(m[5], rows[5] * cols[5], sd, seed + 6);

    /* Which latent indices are huge pairs: a shuffle, so the pairs sit at scattered
     * indices and A's latent-order chain meets them among ordinary terms. */
#define SHUFFLE(a, n)                                                                 \
    do {                                                                              \
        for (int i_ = 0; i_ < (n); i_++) (a)[i_] = i_;                                \
        for (int i_ = (n) - 1; i_ > 0; i_--) {                                        \
            const int j_ = (int)(mla_rng_next(&r) % (uint64_t)(i_ + 1));              \
            const int t_ = (a)[i_]; (a)[i_] = (a)[j_]; (a)[j_] = t_;                  \
        }                                                                             \
    } while (0)
    SHUFFLE(perm, kvl);
    for (int i = 0; i < kvl; i++) lat_role[i] = -1;
    for (int k = 0; k < np; k++) {
        pl[2 * k] = perm[2 * k]; pl[2 * k + 1] = perm[2 * k + 1];
        lat_role[pl[2 * k]] = lat_role[pl[2 * k + 1]] = k;
    }
    int nord = 0;
    for (int i = 0; i < kvl; i++) if (lat_role[i] < 0) ord[nord++] = i;
    SHUFFLE(perm, qr);
    for (int i = 0; i < qr; i++) rope_role[i] = -1;
    for (int k = 0; k < nr; k++) {
        pr[2 * k] = perm[2 * k]; pr[2 * k + 1] = perm[2 * k + 1];
        rope_role[pr[2 * k]] = rope_role[pr[2 * k + 1]] = k;
    }
    /* kv_a: the second row of each pair is the exact negation of the first */
    for (int k = 0; k < np; k++) {
        const float *a = m[2] + (size_t)pl[2 * k] * E;
        float *b = m[2] + (size_t)pl[2 * k + 1] * E;
        for (int e = 0; e < E; e++) b[e] = -a[e];
    }
    for (int k = 0; k < nr; k++) {
        float *a = m[2] + (size_t)(kvl + pr[2 * k]) * E;
        float *b = m[2] + (size_t)(kvl + pr[2 * k + 1]) * E;
        for (int e = 0; e < E; e++) { a[e] = ldexpf(a[e], 44 + k); b[e] = -a[e]; }
    }
    /* q_b and kv_b, head by head: key slots shuffled, the first 2*np carry the pairs */
    for (int h = 0; h < H; h++) {
        float *qb = m[1] + (size_t)h * qh * ql;
        float *kb = m[3] + (size_t)h * kvd * kvl;
        /* The LAST nope slot is always an ordinary term, added after every nope pair
         * has closed, so the nope part ends with bits a huge rope partial sum must
         * round away: that is what makes "nope and rope summed apart" visible. */
        SHUFFLE(perm, qn - 1);
        perm[qn - 1] = qn - 1;
        for (int k = 0; k < np; k++) {
            const int qsrc = (int)(mla_rng_next(&r) % (uint64_t)ql);
            for (int e = 0; e < 2; e++) {
                const int slot = perm[2 * k + e];
                qb[(size_t)slot * ql + qsrc] = 1.0f;
                kb[(size_t)slot * kvl + pl[2 * k + e]] = 1.0f;
            }
        }
        for (int i = 2 * np; i < qn; i++) {
            const int slot = perm[i];
            qb[(size_t)slot * ql + mla_rng_next(&r) % (uint64_t)ql] = 1.0f;
            kb[(size_t)slot * kvl + ord[mla_rng_next(&r) % (uint64_t)nord]] = 1.0f;
        }
        /* rope query rows: the two slots of a rope pair copy one q_lora element */
        int *rsrc = perm;                    /* reuse: one q_lora index per rope pair */
        for (int k = 0; k < nr; k++) rsrc[k] = (int)(mla_rng_next(&r) % (uint64_t)ql);
        for (int i = 0; i < qr; i++) {
            const int src = rope_role[i] >= 0 ? rsrc[rope_role[i]]
                                              : (int)(mla_rng_next(&r) % (uint64_t)ql);
            qb[(size_t)(qn + i) * ql + src] = 1.0f;
        }
        /* value rows: any latent element, huge or ordinary */
        for (int j = 0; j < vh; j++)
            kb[(size_t)(qn + j) * kvl + mla_rng_next(&r) % (uint64_t)kvl] = 1.0f;
    }
#undef SHUFFLE
    for (int i = 0; i < 6; i++) {
        if (i == 5 && !c->mla_out_gate) { free(m[5]); m[5] = NULL; continue; }
        const size_t n = rows[i] * cols[i];
        if (wdt == K3_WBF16) {
            uint16_t *b = (uint16_t *)malloc(n * sizeof(uint16_t));
            if (!b) goto fail;
            for (size_t j = 0; j < n; j++) b[j] = mla_bf16(m[i][j]);
            free(m[i]);
            S->m[i] = b;
        } else {
            S->m[i] = m[i];
        }
        m[i] = NULL;
    }
    /* norms: q_a_norm ordinary; kv_a_norm the same huge float for both members of a
     * pair, so c and -c stay exact negatives through the norm */
    S->norm = (float *)xmalloc((size_t)(ql + kvl) * sizeof(float));
    mla_fill_normal(S->norm, (size_t)(ql + kvl), 0.1f, seed + 7);
    for (int i = 0; i < ql + kvl; i++) S->norm[i] += 1.0f;
    for (int k = 0; k < np; k++) {
        const float big = ldexpf(S->norm[ql + pl[2 * k]], 30 + 2 * k);
        S->norm[ql + pl[2 * k]] = S->norm[ql + pl[2 * k + 1]] = big;
    }
    S->w.q_a = S->m[0]; S->w.q_b = S->m[1]; S->w.kv_a = S->m[2];
    S->w.kv_b = S->m[3]; S->w.o = S->m[4]; S->w.g = S->m[5];
    S->w.q_a_norm = S->norm;
    S->w.kv_a_norm = S->norm + ql;
    S->w.wdt = wdt;
    free(lat_role); free(rope_role); free(perm); free(ord); free(pl); free(pr);
    return 0;
fail:
    for (int i = 0; i < 6; i++) free(m[i]);
    mla_synth_free(S);
    free(lat_role); free(rope_role); free(perm); free(ord); free(pl); free(pr);
    return -1;
}

static double pct(double a, double n) { return n > 0.0 ? 100.0 * a / n : 0.0; }

static void print_witness(const char *label, const Witness *w)
{
    printf("order witness, %s: of %.0f scores, a reordered double chain rounds to a\n"
           "  different float in %.1f%% (reversed), %.1f%% (nope and rope summed apart), "
           "%.1f%% (two lanes);\n  of %.0f float value sums of 3+ terms, %.1f%% change "
           "when summed in reverse;\n  of %.0f softmax normalisers of 3+ terms, %.1f%% "
           "change when summed in reverse;\n  of %.0f probability quotients e/z, %.1f%% "
           "change when formed as e*(1/z)\n", label, w->scores, pct(w->rev, w->scores),
           pct(w->split, w->scores), pct(w->lanes, w->scores), w->vals,
           pct(w->vrev, w->vals), w->zrows, pct(w->zrev, w->zrows), w->quots,
           pct(w->qrecip, w->quots));
}

int main(void)
{
    /* {0, 256} is the prefill shape: no cache, 256 new tokens, where L0 makes 65,792
     * kv_b applications and L1 makes 256. */
    struct { int C, T; } tiny[] = {{0, 1}, {0, 3}, {1, 1}, {5, 2}, {7, 5}, {31, 4},
                                   {64, 1}, {200, 5}, {257, 3}, {0, 256}};
    struct { int C, T; } odd[] = {{0, 2}, {3, 5}, {13, 3}, {50, 7}};
    struct { int C, T; } real[] = {{0, 1}, {1, 5}, {6, 2}, {33, 5}, {40, 1}};
    struct { int C, T; } canc[] = {{0, 1}, {0, 5}, {3, 4}, {17, 3}, {40, 5}, {0, 64}};
    struct { int C, T; } canc_odd[] = {{0, 3}, {9, 5}, {30, 2}};
    struct { int C, T; } sharp[] = {{0, 5}, {7, 5}, {64, 3}, {200, 5}, {0, 64}, {3, 40}};
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
                             tiny[i].C, tiny[i].T, 1000 + 17 * (uint64_t)i, LAYER_ORDINARY);
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
                             2000 + 17 * (uint64_t)i, LAYER_ORDINARY);
        mla_synth_free(&S);
    }
    {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_k3(&c, 512);
        if (mla_synth_layer(&S, &c, K3_WBF16, 0.02f, 0.1f, 37, 1)) return 1;
        for (size_t i = 0; i < sizeof real / sizeof real[0]; i++, ncase++)
            nok += test_case("K3 attention bf16", &c, &S.w, real[i].C, real[i].T,
                             3000 + 17 * (uint64_t)i, LAYER_ORDINARY);
        mla_synth_free(&S);
    }
    /* The same gates on cancelling layers, where a reordered chain cannot hide. */
    for (int wdt = 0; wdt < 2; wdt++) {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_tiny(&c);
        const int t = wdt ? K3_WBF16 : K3_WF32;
        if (synth_cancelling(&S, &c, t, 0.1f, 41 + (uint64_t)wdt)) return 1;
        for (size_t i = 0; i < sizeof canc / sizeof canc[0]; i++, ncase++)
            nok += test_case(wdt ? "cancelling fixture bf16" : "cancelling fixture fp32", &c,
                             &S.w, canc[i].C, canc[i].T, 4000 + 17 * (uint64_t)i,
                             LAYER_CANCELLING);
        mla_synth_free(&S);
    }
    {
        /* every loop gets a tail: kv_lora 37 is odd for A's four-term steps, qk_nope 21
         * and qk_rope 5 leave remainders everywhere */
        K3Cfg c;
        MlaSynth S;
        mla_cfg_tiny(&c);
        c.hidden = 72; c.n_heads = 3; c.q_lora = 48; c.kv_lora = 37;
        c.qk_nope = 21; c.qk_rope = 5; c.v_head = 12;
        if (synth_cancelling(&S, &c, K3_WBF16, 0.1f, 43)) return 1;
        for (size_t i = 0; i < sizeof canc_odd / sizeof canc_odd[0]; i++, ncase++)
            nok += test_case("cancelling odd bf16", &c, &S.w, canc_odd[i].C, canc_odd[i].T,
                             5000 + 17 * (uint64_t)i, LAYER_CANCELLING);
        mla_synth_free(&S);
    }
    {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_k3(&c, 512);
        if (synth_cancelling(&S, &c, K3_WBF16, 0.02f, 47)) return 1;
        nok += test_case("cancelling K3 bf16", &c, &S.w, 13, 4, 6000, LAYER_CANCELLING);
        ncase++;
        mla_synth_free(&S);
    }
    /* Sharp rows: the same kind of layer with the query norm scaled by 16, which scales
     * q, and so every score, by exactly 16 (a power of two commutes with every rounding
     * on the way). Score ranges of tens of nats put softmax terms far below 2^-29, where
     * the double normaliser stops being exact and its order shows. */
    for (int wdt = 0; wdt < 2; wdt++) {
        K3Cfg c;
        MlaSynth S;
        mla_cfg_tiny(&c);
        const int t = wdt ? K3_WBF16 : K3_WF32;
        if (mla_synth_layer(&S, &c, t, 0.1f, 0.1f, 53 + (uint64_t)wdt, 1)) return 1;
        for (int i = 0; i < c.q_lora; i++) S.norm[i] *= 16.0f;
        for (size_t i = 0; i < sizeof sharp / sizeof sharp[0]; i++, ncase++)
            nok += test_case(wdt ? "sharp fixture bf16" : "sharp fixture fp32", &c, &S.w,
                             sharp[i].C, sharp[i].T, 7000 + 17 * (uint64_t)i, LAYER_SHARP);
        mla_synth_free(&S);
    }
    print_witness("ordinary layers", &g_wit[LAYER_ORDINARY]);
    print_witness("cancelling layers", &g_wit[LAYER_CANCELLING]);
    print_witness("sharp layers", &g_wit[LAYER_SHARP]);
    /* The cancelling and sharp layers must keep doing their jobs: nearly every
     * reordered chain rounds differently on the first (the ordinary layers print ~0%),
     * and a good share of reordered normalisers differ on the second (again ~0% on the
     * ordinary layers). The seeds are fixed, so the rates do not wander; the floors sit
     * under them and trip only if a construction stops working. */
    {
        const Witness *w = &g_wit[LAYER_CANCELLING];
        CHECK(w->scores > 20000 && w->rev >= 0.95 * w->scores && w->split >= 0.95 * w->scores
                  && w->lanes >= 0.85 * w->scores,
              "cancelling layers no longer expose reordered score chains");
        const Witness *z = &g_wit[LAYER_SHARP];
        CHECK(z->zrows > 400 && z->zrev >= 0.3 * z->zrows,
              "sharp layers no longer expose a reordered softmax normaliser");
        /* The quotient needs no special layer: e*(1/z) rounds twice where e/z rounds
         * once, and differs in about a fifth to a quarter of the quotients anywhere. */
        for (int k = 0; k < LAYER_KINDS; k++)
            CHECK(g_wit[k].quots > 20000 && g_wit[k].qrecip >= 0.1 * g_wit[k].quots,
                  "layer kind %d no longer exposes a reciprocal-multiply quotient", k);
    }

    printf("%d/%d cases, %d checks, %d failed\n", nok, ncase, checks, failures);
    if (failures) {
        printf("MLA VARIANTS: FAIL\n");
        return 1;
    }
    printf("MLA VARIANTS: L1 AND E+ BITWISE IDENTICAL TO E, L0 AND THE ENGINE\n");
    return 0;
}
