/* mla_variants.h - one Gated-MLA layer's CACHED attention, four ways, for measurement.
 *
 * Benchmark and test code only. Nothing in src/ includes this file and the engine's own
 * k3_mla_cached is untouched by it; tests/unit/test_mla_variants.c holds the copies
 * below to the engine bit for bit (its outputs, and through its k3_mla_trace hook the
 * scores, normalisers, quotients and accumulators the output cannot show), and
 * benchmarks/bench_mla.c times them and measures how far the one inexact variant
 * moves. docs/notes/mla-variants.md has the results.
 *
 * WHAT IS BEING COMPARED
 *   Every variant runs the engine's own projections (q_a, q_a_norm, q_b for the query;
 *   kv_a, kv_a_norm for the latent and the shared rope row) and the engine's own output
 *   gate and o_proj: mla_project and mla_finish below are those calls, in that order.
 *   What differs is how the T new tokens are appended to the cache and how attention
 *   over the C cached positions plus the T new ones is formed. N = C + T positions in
 *   all; query token t sits at position C + t and sees positions 0..C+t (causal).
 *
 *   E   EXPANDED cache: per-head k_nope and v, 98,304 B per position per layer, plus
 *       the 256 B rope row. The loop in k3_mla_cached today, copied statement for
 *       statement: per query token, per head, positions ascending, on one thread.
 *   E+  NOT one of the four requested variants. The same arithmetic as E, restructured:
 *       heads split across OpenMP threads, positions outer and query tokens inner so the
 *       cache is read once per call rather than once per query token, and four position
 *       chains in flight at a time. It exists so that A is not credited with a speedup
 *       that is really only a better loop. Bitwise identical to E.
 *   L0  LATENT cache: the 512-float post-norm latent plus the rope row, 2,304 B per
 *       position per layer. The loop in k3_mla_cached today: every cached position is
 *       rebuilt through kv_b TWICE PER QUERY TOKEN (score pass, value pass).
 *   L1  LATENT cache, each position rebuilt ONCE PER CALL. Bitwise identical to E and
 *       L0; the argument is below.
 *   A   ABSORBED: kv_b's per-head key rows are folded into the query and its value rows
 *       applied after the latent-weighted sum. The same real-number function as E, NOT
 *       the same floats; bench_mla's `numerics` mode measures the difference.
 *
 * THE kv_b ROW LAYOUT, which A depends on and which is verified rather than assumed
 *   kv_b is [H*(qk_nope+v_head)][kv_lora]. The expanded path fills one position's block
 *   with a single k3_mmw over all H*kvd rows, then reads head h's key at block + h*kvd
 *   and its value at block + h*kvd + qk_nope (K3_KV_AT in k3_mla_cached), and the
 *   reference does kv.view(B, -1, H, qk_nope + v_head).split([qk_nope, v_head]). So
 *   head h owns rows [h*kvd, h*kvd + qk_nope) -- W_uk[h], producing k_nope -- and rows
 *   [h*kvd + qk_nope, (h+1)*kvd) -- W_uv[h], producing v. Both are VIEWS of the bf16
 *   tensor: A copies, transposes and re-rounds no weight.
 *
 * L1 IS BITWISE IDENTICAL TO E AND L0, AND WHY
 *   Every float E produces comes from one of four computations, and L1 performs each
 *   of them on the same operands in the same order:
 *     k and v    E stores k3_mmw(kv_b, latent) at append time; L1 calls the same kernel
 *                on the same latent bytes at use time. k3_matmul_bf16 computes each
 *                output row on one thread with a fixed sixteen-accumulator tree, so its
 *                result does not depend on when, how often, or on how many threads it
 *                runs. The latent L1 reads is a memcpy of the vector E fed to kv_b.
 *     a score    one double chain per (t, h, s): qk_nope terms ascending, then the
 *                qk_rope terms ascending, starting from 0.0, then (float)d * scale.
 *                L1 evaluates exactly that expression; only WHEN it runs changes
 *                (position outer, so one rebuild serves every query token and head).
 *     softmax    per (t, h): the max over s ascending, expf(sc - m) and the double sum
 *                z over s ascending, p = (float)(e / z). Same statements.
 *     the value  per (t, h, j): o starts at 0.0f and receives p_s * v_s[j] for
 *                s = 0, 1, ..., C+t in that order, float multiply then float add
 *                (-ffp-contract=off). L1 walks positions in blocks, ascending, and
 *                within a block ascending, and one thread owns each head throughout,
 *                so no term moves.
 *   What L1 changes is only how often the kernel runs: N rebuilds per call instead of
 *   2 * sum_t (C+t+1). The value rows rebuilt in the score pass are kept in a transient
 *   buffer for the value pass. That buffer is H*v_head floats = 49,152 B per position
 *   -- half an expanded position -- for ONE layer at a time, reused by the next layer.
 *   `vcap` bounds it: positions at or beyond vcap are rebuilt a second time instead,
 *   which is still the same kernel on the same bytes, so the cap trades compute for
 *   memory without touching a bit.
 *
 * E+ IS BITWISE IDENTICAL TO E, AND WHY
 *   Heads are independent: no reduction crosses them, so splitting them across threads
 *   moves no term. The four interleaved chains are four separate doubles, each fed its
 *   own position's terms in E's order; nothing is summed across them. Running query
 *   tokens inside the position loop reorders when a score is computed, never how.
 *
 * A: WHAT IS COMPUTED, AND WHERE IT ROUNDS DIFFERENTLY FROM E
 *   q_abs[t,h]  = W_uk[h]^T q_nope[t,h]                 kv_lora wide; double sum over
 *                                                       the qk_nope rows, ascending,
 *                                                       rounded to float like every
 *                                                       matmul output in the engine
 *   score       = scale * (q_abs . c_s + q_rope . r_s)  one double chain, as in E
 *   softmax                                             exactly the engine's form
 *   u[t,h]      = sum_s p_s c_s                         float, s ascending, as E sums v
 *   out[t,h]    = W_uv[h] u[t,h]                        k3_mmw: the engine kernel
 *   In exact arithmetic q_nope . (W_uk c) = (W_uk^T q_nope) . c and
 *   sum_s p_s (W_uv c_s) = W_uv (sum_s p_s c_s). In floats E rounds k_nope and v (a
 *   kv_b output) to float and A rounds q_abs and out instead, and E accumulates 128-wide
 *   value rows over s where A accumulates 512-wide latent rows. Neither is the more
 *   exact one by construction; bench_mla measures both against a double reference.
 *   Cost per position per query token: E does H*(qk_nope+qk_rope+v_head) = 30,720 MACs
 *   over 98,560 cached bytes; A does H*(2*kv_lora+qk_rope) = 104,448 MACs over 2,304
 *   bytes, plus H*qk_nope*kv_lora*2 = 12.6M MACs per query token for the absorption and
 *   the output, i.e. one kv_b's worth, independent of C.
 *   The score loop runs the chains for all heads side by side (head-minor q_abs), which
 *   vectorises across heads without reassociating any chain: each lane is one head's
 *   sequential sum. tests/unit/test_mla_variants.c checks A against a plain scalar
 *   rendering of the formulas above, bitwise.
 *
 * THREADING
 *   kv_b goes through k3_mmw everywhere, which splits output rows across OpenMP threads.
 *   E and L0 are otherwise single-threaded, as in the engine. E+, L1 and A split heads,
 *   positions or (query, head) pairs across threads, never a reduction. Where A calls
 *   k3_mmw inside its own parallel loop the inner region is inactive under the default
 *   max-active-levels of 1, and the kernel's result is the same either way. Every
 *   variant's output is independent of the thread count; the test checks that too.
 */
#ifndef MLA_VARIANTS_H
#define MLA_VARIANTS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"

enum { MLA_E, MLA_EP, MLA_L0, MLA_L1, MLA_A, MLA_NVAR };
static const char *const MLA_NAME[MLA_NVAR] = {"E", "E+", "L0", "L1", "A"};

/* A keeps one double accumulator per head, and one converted kv_b row, on the stack. */
#define MLA_MAX_HEADS 256
#define MLA_MAX_LORA  2048
/* Positions per block in L1 and A: L1 rebuilds this many, one k3_mmw call each, then
 * scores them in one parallel region; A's value pass keeps this many latent rows hot
 * while every head uses them. The block changes no order, only locality. */
#define MLA_BLOCK 16

static inline int mla_is_latent(int v) { return v == MLA_L0 || v == MLA_L1 || v == MLA_A; }

typedef struct {
    float *kv;    /* expanded [cap][H][qk_nope+v_head], or latent [cap][kv_lora] */
    float *rope;  /* [cap][qk_rope]: the shared slot, in both layouts, never projected */
    int    cap;
} MlaCache;

/* Persistent floats per cached position per layer, rope row included. */
static inline size_t mla_cache_floats(const K3Cfg *c, int latent)
{
    const size_t per = latent ? (size_t)c->kv_lora
                              : (size_t)c->n_heads * (size_t)(c->qk_nope + c->v_head);
    return per + (size_t)c->qk_rope;
}

/* Row `row` of a tagged weight matrix whose rows are `in` wide. */
static inline const void *mla_row(const void *W, int wdt, int in, size_t row)
{
    return (const void *)((const unsigned char *)W + row * k3_row_bytes(wdt, in));
}

/* Transient bytes the attention step needs for T query tokens over N positions; vcap is
 * L1's value-row budget in positions (clamped to [0, N]). */
static inline size_t mla_scratch_bytes(int v, const K3Cfg *c, int T, int N, int vcap)
{
    const size_t H = (size_t)c->n_heads, kvd = (size_t)(c->qk_nope + c->v_head);
    const size_t kvl = (size_t)c->kv_lora, qr = (size_t)c->qk_rope, vh = (size_t)c->v_head;
    const size_t t = (size_t)T, n = (size_t)N;
    if (vcap > N) vcap = N;
    if (vcap < 0) vcap = 0;
    switch (v) {
    case MLA_E:  return n * sizeof(float);
    case MLA_EP: return H * t * n * sizeof(float);
    case MLA_L0: return (H * n + H * kvd) * sizeof(float);
    case MLA_L1:
        return (t * H * n + (size_t)MLA_BLOCK * H * kvd + (size_t)vcap * H * vh)
               * sizeof(float);
    case MLA_A:
        return (t * (kvl + qr) * H + H * t * kvl) * sizeof(double)
               + (t * H * n + t * H * kvl) * sizeof(float);
    }
    return 0;
}

/* ------------------------------------------------------------ shared steps ---- */

/* The engine's per-token projections, the same calls in the same order as the top of
 * k3_mla_cached: q[t] is [H][qk_nope+qk_rope]; ct[t] is [kv_lora + qk_rope], the
 * post-norm latent followed by the UN-normed rope slot. ql is [q_lora]. */
static void mla_project(float *q, float *ct, const float *x, const K3MlaW *w,
                        const K3Cfg *c, int T, float *ql)
{
    const int E = c->hidden, H = c->n_heads, qh = c->qk_nope + c->qk_rope;
    const int kvw = c->kv_lora + c->qk_rope;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        k3_mmw(ql, xt, w->q_a, w->wdt, E, c->q_lora);
        k3_rmsnorm(ql, ql, w->q_a_norm, c->q_lora, c->rms_eps);
        k3_mmw(q + (size_t)t * H * qh, ql, w->q_b, w->wdt, c->q_lora, H * qh);
        float *ctt = ct + (size_t)t * kvw;
        k3_mmw(ctt, xt, w->kv_a, w->wdt, E, kvw);
        k3_rmsnorm(ctt, ctt, w->kv_a_norm, c->kv_lora, c->rms_eps);
    }
}

/* The kv side only, for filling a cache with a prefix nobody queries from. */
static void mla_project_kv(float *ct, const float *x, const K3MlaW *w, const K3Cfg *c,
                           int T)
{
    const int E = c->hidden, kvw = c->kv_lora + c->qk_rope;
    for (int t = 0; t < T; t++) {
        float *ctt = ct + (size_t)t * kvw;
        k3_mmw(ctt, x + (size_t)t * E, w->kv_a, w->wdt, E, kvw);
        k3_rmsnorm(ctt, ctt, w->kv_a_norm, c->kv_lora, c->rms_eps);
    }
}

/* ONE application of kv_b: one position's latent in, all H heads' k_nope and v out.
 * Every full-matrix kv_b call in this file goes through here, so mla_kvb_calls is an
 * exact count of them. The test holds that count to mla_rebuilds() for every variant,
 * and bench_mla's `counts` mode reports it; counts, unlike timings, do not move with
 * machine load. The calls are all made from serial code (k3_mmw threads INSIDE the
 * kernel), so a plain counter is race-free. */
static unsigned long long mla_kvb_calls;

static inline void mla_kvb(float *out, const float *lat, const K3MlaW *w, const K3Cfg *c)
{
    mla_kvb_calls++;
    k3_mmw(out, lat, w->kv_b, w->wdt, c->kv_lora, c->n_heads * (c->qk_nope + c->v_head));
}

/* Store the T new tokens at positions C..C+T-1: the latent layout keeps the kv_b INPUT,
 * the expanded layout the kv_b OUTPUT, and both keep the rope row. */
static void mla_append(MlaCache *k, int latent, const float *ct, int T, int C,
                       const K3MlaW *w, const K3Cfg *c)
{
    const int H = c->n_heads, kvd = c->qk_nope + c->v_head, qr = c->qk_rope;
    const int kvl = c->kv_lora, kvw = kvl + qr;
    for (int t = 0; t < T; t++) {
        const int p = C + t;
        const float *ctt = ct + (size_t)t * kvw;
        memcpy(k->rope + (size_t)p * qr, ctt + kvl, (size_t)qr * sizeof(float));
        if (latent) memcpy(k->kv + (size_t)p * kvl, ctt, (size_t)kvl * sizeof(float));
        else        mla_kvb(k->kv + (size_t)p * H * kvd, ctt, w, c);
    }
}

/* The engine's output gate then o_proj, per token. acc is [T][H][v_head] and is gated
 * in place, exactly as k3_mla_cached gates its per-token accumulator. gbuf is [H*v_head]. */
static void mla_finish(float *out, float *acc, const float *x, const K3MlaW *w,
                       const K3Cfg *c, int T, float *gbuf)
{
    const int E = c->hidden, HV = c->n_heads * c->v_head;
    for (int t = 0; t < T; t++) {
        float *a = acc + (size_t)t * HV;
        if (w->g) {
            k3_mmw(gbuf, x + (size_t)t * E, w->g, w->wdt, E, HV);
            for (int i = 0; i < HV; i++) a[i] *= 1.0f / (1.0f + expf(-gbuf[i]));
        }
        k3_mmw(out + (size_t)t * E, a, w->o, w->wdt, HV, E);
    }
}

/* Raw scaled scores, before the softmax, for the numerical study: probe[(t*H+h)*N + s].
 * NULL everywhere except bench_mla's `numerics` mode and the test. */
static inline void mla_probe_row(float *probe, const float *row, int t, int h, int H,
                                 int N, int p)
{
    if (probe)
        memcpy(probe + ((size_t)t * H + h) * N, row, (size_t)(p + 1) * sizeof(float));
}

/* The softmax normaliser z of row (t, h), for the test: mla_zprobe[t*H + h]. z is a
 * double sum of positive terms that only ever reaches the output as p = (float)(e / z),
 * where a reordered sum would round to the same float almost surely; recording z itself
 * is what lets the test hold its ORDER to E's, and E's to the engine's (k3_mla_trace),
 * bit for bit. NULL (no recording) except in tests/unit/test_mla_variants.c. Each (t, h)
 * is written by one thread. */
static double *mla_zprobe;

static inline void mla_probe_z(int t, int h, int H, double z)
{
    if (mla_zprobe) mla_zprobe[(size_t)t * H + h] = z;
}

/* The probability quotients of row (t, h), for the test: mla_qprobe[(t*H + h)*N + s] is
 * the double e_s / z that p_s is rounded from. Like z, it reaches the output only
 * through that rounding, which a reciprocal multiply or any other way of forming it
 * would survive almost surely; recording the double is what lets the test hold the step
 * itself to the engine's. NULL except in the test. mla_qrow is the row to record into,
 * or NULL. */
static double *mla_qprobe;

static inline double *mla_qrow(int t, int h, int H, int N)
{
    return mla_qprobe ? mla_qprobe + ((size_t)t * H + h) * N : NULL;
}

/* The engine's softmax form, in place over row[0..p]: max ascending, expf(x - m) and a
 * double sum ascending, then p = (float)(e / z) folded back into the row, the quotient
 * named first so that the value recorded in qrow (when not NULL) is the value rounded,
 * exactly as in k3_mla_cached. Returns z. */
static inline double mla_softmax_row(float *row, int p, double *qrow)
{
    float m = -INFINITY;
    for (int s = 0; s <= p; s++) if (row[s] > m) m = row[s];
    double z = 0.0;
    for (int s = 0; s <= p; s++) { row[s] = expf(row[s] - m); z += row[s]; }
    for (int s = 0; s <= p; s++) {
        const double pq = row[s] / z;
        if (qrow) qrow[s] = pq;
        row[s] = (float)pq;
    }
    return z;
}

/* -------------------------------------------------------------------- E ---- */
/* k3_mla_cached's expanded branch, statement for statement. acc is [T][H][v_head]
 * rather than the engine's per-token [H][v_head]; nothing else differs. */
static void mla_attend_E(float *acc, const float *q, int T, int C, const MlaCache *k,
                         const K3Cfg *c, void *scratch, float *probe)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    float *sc = (float *)scratch;                                   /* [N] */
    for (int t = 0; t < T; t++) {
        const int p = C + t;
        for (int h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) {                          /* causal: s <= p */
                const float *ks = k->kv + (size_t)s * H * kvd + (size_t)h * kvd;
                const float *kr = k->rope + (size_t)s * qr;         /* shared slot */
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                sc[s] = (float)d * scale;
                if (sc[s] > m) m = sc[s];
            }
            mla_probe_row(probe, sc, t, h, H, N, p);
            double z = 0.0;
            for (int s = 0; s <= p; s++) { sc[s] = expf(sc[s] - m); z += sc[s]; }
            mla_probe_z(t, h, H, z);
            double *qrow = mla_qrow(t, h, H, N);

            float *o = acc + ((size_t)t * H + h) * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const double pq = sc[s] / z;
                if (qrow) qrow[s] = pq;
                const float pr = (float)pq;
                const float *vs = k->kv + (size_t)s * H * kvd + (size_t)h * kvd + qn;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }
    }
}

/* ------------------------------------------------------------------- E+ ---- */
/* E's score chain for one (query, position): qk_nope terms, then qk_rope terms. */
static inline double mla_dot_E(const float *qt, const float *ks, const float *kr, int qn,
                               int qr)
{
    double d = 0.0;
    for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
    for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
    return d;
}

/* Four of E's chains in flight: four separate doubles, each fed only its own position's
 * terms in E's order, so each equals mla_dot_E bit for bit. The point is the four
 * independent add-latency chains, not any sharing between them. */
static inline void mla_dot4_E(double *d, const float *qt, const float *const *ks,
                              const float *const *kr, int qn, int qr)
{
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    for (int i = 0; i < qn; i++) {
        const double qi = (double)qt[i];
        d0 += qi * (double)ks[0][i]; d1 += qi * (double)ks[1][i];
        d2 += qi * (double)ks[2][i]; d3 += qi * (double)ks[3][i];
    }
    for (int i = 0; i < qr; i++) {
        const double qi = (double)qt[qn + i];
        d0 += qi * (double)kr[0][i]; d1 += qi * (double)kr[1][i];
        d2 += qi * (double)kr[2][i]; d3 += qi * (double)kr[3][i];
    }
    d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

static void mla_attend_EP(float *acc, const float *q, int T, int C, const MlaCache *k,
                          const K3Cfg *c, void *scratch, float *probe)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    float *scb = (float *)scratch;                                  /* [H][T][N] */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        float *sh = scb + (size_t)h * T * N;
        /* Scores. Positions outer, so a key row is fetched once for every query token
         * that can see it; query token t sees position s iff t >= s - C. */
        for (int s0 = 0; s0 < N; s0 += 4) {
            const int nb = N - s0 < 4 ? N - s0 : 4;
            const float *ks[4], *kr[4];
            for (int b = 0; b < 4; b++) {
                const int s = s0 + (b < nb ? b : 0);
                ks[b] = k->kv + (size_t)s * H * kvd + (size_t)h * kvd;
                kr[b] = k->rope + (size_t)s * qr;
            }
            for (int t = (s0 > C ? s0 - C : 0); t < T; t++) {
                const float *qt = q + ((size_t)t * H + h) * qh;
                const int vis = (C + t + 1) - s0;                   /* >= 1 here */
                float *r = sh + (size_t)t * N + s0;
                if (nb == 4 && vis >= 4) {
                    double d[4];
                    mla_dot4_E(d, qt, ks, kr, qn, qr);
                    for (int b = 0; b < 4; b++) r[b] = (float)d[b] * scale;
                } else {
                    const int e = nb < vis ? nb : vis;
                    for (int b = 0; b < e; b++)
                        r[b] = (float)mla_dot_E(qt, ks[b], kr[b], qn, qr) * scale;
                }
            }
        }
        for (int t = 0; t < T; t++) {
            float *r = sh + (size_t)t * N;
            mla_probe_row(probe, r, t, h, H, N, C + t);
            mla_probe_z(t, h, H, mla_softmax_row(r, C + t, mla_qrow(t, h, H, N)));
            float *o = acc + ((size_t)t * H + h) * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
        }
        /* Values: positions ascending, each value row fetched once for all T. */
        for (int s = 0; s < N; s++) {
            const float *vs = k->kv + (size_t)s * H * kvd + (size_t)h * kvd + qn;
            for (int t = (s > C ? s - C : 0); t < T; t++) {
                const float pr = sh[(size_t)t * N + s];
                float *o = acc + ((size_t)t * H + h) * vh;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }
    }
}

/* ------------------------------------------------------------------- L0 ---- */
/* k3_mla_cached's latent branch, statement for statement: per query token, rebuild
 * every visible position to score it, softmax, rebuild every visible position again to
 * weight its values. The score rows have stride N (the engine's last + 1). */
static void mla_attend_L0(float *acc, const float *q, int T, int C, const MlaCache *k,
                          const K3MlaW *w, const K3Cfg *c, void *scratch, float *probe)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, kvl = c->kv_lora, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    float *sc = (float *)scratch;                                   /* [H][N] */
    float *kb = sc + (size_t)H * N;                                 /* [H][kvd] */
    for (int t = 0; t < T; t++) {
        const int p = C + t;
        for (int s = 0; s <= p; s++) {
            mla_kvb(kb, k->kv + (size_t)s * kvl, w, c);
            const float *kr = k->rope + (size_t)s * qr;
            for (int h = 0; h < H; h++) {
                const float *qt = q + ((size_t)t * H + h) * qh;
                const float *ks = kb + (size_t)h * kvd;
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                sc[(size_t)h * N + s] = (float)d * scale;
            }
        }
        for (int h = 0; h < H; h++) {
            float *sh = sc + (size_t)h * N;
            mla_probe_row(probe, sh, t, h, H, N, p);
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) if (sh[s] > m) m = sh[s];
            double z = 0.0;
            for (int s = 0; s <= p; s++) { sh[s] = expf(sh[s] - m); z += sh[s]; }
            mla_probe_z(t, h, H, z);
            double *qrow = mla_qrow(t, h, H, N);
            for (int s = 0; s <= p; s++) {
                const double pq = sh[s] / z;
                if (qrow) qrow[s] = pq;
                sh[s] = (float)pq;
            }
            float *o = acc + ((size_t)t * H + h) * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
        }
        for (int s = 0; s <= p; s++) {
            mla_kvb(kb, k->kv + (size_t)s * kvl, w, c);
            for (int h = 0; h < H; h++) {
                const float pr = sc[(size_t)h * N + s];
                float *o = acc + ((size_t)t * H + h) * vh;
                const float *vs = kb + (size_t)h * kvd + qn;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }
    }
}

/* ------------------------------------------------------------------- L1 ---- */
/* One rebuild per position per call. See the header comment for the exactness
 * argument; vcap is the number of leading positions whose value rows are held. */
static void mla_attend_L1(float *acc, const float *q, int T, int C, const MlaCache *k,
                          const K3MlaW *w, const K3Cfg *c, void *scratch, int vcap,
                          float *probe)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, kvl = c->kv_lora, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    if (vcap > N) vcap = N;
    if (vcap < 0) vcap = 0;
    float *sc   = (float *)scratch;                                 /* [T][H][N] */
    float *kbuf = sc + (size_t)T * H * N;                           /* [BLOCK][H][kvd] */
    float *vbuf = kbuf + (size_t)MLA_BLOCK * H * kvd;               /* [vcap][H][vh] */

    /* Pass one: rebuild each position once, score it for every query token that can
     * see it and every head, and keep its value rows if they fit. */
    for (int s0 = 0; s0 < N; s0 += MLA_BLOCK) {
        const int nb = N - s0 < MLA_BLOCK ? N - s0 : MLA_BLOCK;
        for (int b = 0; b < nb; b++)
            mla_kvb(kbuf + (size_t)b * H * kvd, k->kv + (size_t)(s0 + b) * kvl, w, c);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < H; h++) {
            for (int b = 0; b < nb; b++) {
                const int s = s0 + b;
                const float *ks = kbuf + ((size_t)b * H + h) * kvd;
                const float *kr = k->rope + (size_t)s * qr;
                for (int t = (s > C ? s - C : 0); t < T; t++) {
                    const float *qt = q + ((size_t)t * H + h) * qh;
                    double d = 0.0;
                    for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                    for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                    sc[((size_t)t * H + h) * N + s] = (float)d * scale;
                }
                if (s < vcap)
                    memcpy(vbuf + ((size_t)s * H + h) * vh, ks + qn,
                           (size_t)vh * sizeof(float));
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int th = 0; th < T * H; th++) {
        const int t = th / H, h = th % H;
        float *r = sc + (size_t)th * N;
        mla_probe_row(probe, r, t, h, H, N, C + t);
        mla_probe_z(t, h, H, mla_softmax_row(r, C + t, mla_qrow(t, h, H, N)));
        float *o = acc + (size_t)th * vh;
        for (int j = 0; j < vh; j++) o[j] = 0.0f;
    }
    /* Pass two: every output element receives its terms s ascending. Positions past the
     * budget are rebuilt again, through the same kernel. */
    for (int s0 = 0; s0 < N; s0 += MLA_BLOCK) {
        const int nb = N - s0 < MLA_BLOCK ? N - s0 : MLA_BLOCK;
        for (int b = 0; b < nb; b++)
            if (s0 + b >= vcap)
                mla_kvb(kbuf + (size_t)b * H * kvd, k->kv + (size_t)(s0 + b) * kvl, w, c);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < H; h++) {
            for (int b = 0; b < nb; b++) {
                const int s = s0 + b;
                const float *vs = s < vcap ? vbuf + ((size_t)s * H + h) * vh
                                           : kbuf + ((size_t)b * H + h) * kvd + qn;
                for (int t = (s > C ? s - C : 0); t < T; t++) {
                    const float pr = sc[((size_t)t * H + h) * N + s];
                    float *o = acc + ((size_t)t * H + h) * vh;
                    for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
                }
            }
        }
    }
}

/* -------------------------------------------------------------------- A ---- */
/* Returns 0, or -1 for a layout it cannot run: A reads kv_b's elements directly, so it
 * needs a resident fp32 or bf16 kv_b rather than an int8 or streamed one. */
static int mla_attend_A(float *acc, const float *q, int T, int C, const MlaCache *k,
                        const K3MlaW *w, const K3Cfg *c, void *scratch, float *probe)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr, kvd = qn + vh, kvl = c->kv_lora, N = C + T;
    const float scale = 1.0f / sqrtf((float)qh);
    if ((w->wdt != K3_WBF16 && w->wdt != K3_WF32) || H > MLA_MAX_HEADS || kvl > MLA_MAX_LORA)
        return -1;
    double *qa = (double *)scratch;                        /* [T][kvl][H]  q_abs       */
    double *qp = qa + (size_t)T * kvl * H;                 /* [T][qr][H]   q_rope      */
    double *ab = qp + (size_t)T * qr * H;                  /* [H][T][kvl]  absorb sums */
    float  *sc = (float *)(ab + (size_t)H * T * kvl);      /* [T][H][N]    scores, p   */
    float  *u  = sc + (size_t)T * H * N;                   /* [T][H][kvl]  sum p c     */

    /* 1. q_abs[t,h,j] = (float) sum_i W_uk[h][i][j] * q_nope[t,h][i], i ascending, in
     *    double. Each bf16 row is widened once and used for all T query tokens; the
     *    product of a widened bf16 and a float is exact in double, so each step is one
     *    rounding, as in the kernels. Stored head-minor for the score loop. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        double wd[MLA_MAX_LORA];
        double *ah = ab + (size_t)h * T * kvl;
        for (size_t j = 0; j < (size_t)T * kvl; j++) ah[j] = 0.0;
        for (int i = 0; i < qn; i++) {
            const void *row = mla_row(w->kv_b, w->wdt, kvl, (size_t)h * kvd + i);
            if (w->wdt == K3_WBF16)
                for (int j = 0; j < kvl; j++)
                    wd[j] = (double)k3_bf16f(((const uint16_t *)row)[j]);
            else
                for (int j = 0; j < kvl; j++) wd[j] = (double)((const float *)row)[j];
            for (int t = 0; t < T; t++) {
                const double qi = (double)q[((size_t)t * H + h) * qh + i];
                double *at = ah + (size_t)t * kvl;
                for (int j = 0; j < kvl; j++) at[j] += wd[j] * qi;
            }
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < kvl; j++)
                qa[((size_t)t * kvl + j) * H + h] = (double)(float)ah[(size_t)t * kvl + j];
            for (int i = 0; i < qr; i++)
                qp[((size_t)t * qr + i) * H + h] = (double)q[((size_t)t * H + h) * qh + qn + i];
        }
    }

    /* 2. Scores: for each position, all heads' chains side by side. d[h] receives
     *    q_abs[h][0]*c[0], q_abs[h][1]*c[1], ... then the rope terms, in that order,
     *    whatever the vector width: the four-term statement below is the sequential
     *    sum written out, not a regrouping of it. */
    for (int t = 0; t < T; t++) {
        const double *qat = qa + (size_t)t * kvl * H;
        const double *qpt = qp + (size_t)t * qr * H;
        const int p = C + t;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int s = 0; s <= p; s++) {
            double d[MLA_MAX_HEADS];
            const float *cs = k->kv + (size_t)s * kvl;
            const float *kr = k->rope + (size_t)s * qr;
            for (int h = 0; h < H; h++) d[h] = 0.0;
            int j = 0;
            for (; j + 3 < kvl; j += 4) {
                const double c0 = (double)cs[j], c1 = (double)cs[j + 1];
                const double c2 = (double)cs[j + 2], c3 = (double)cs[j + 3];
                const double *q0 = qat + (size_t)j * H, *q1 = q0 + H;
                const double *q2 = q1 + H, *q3 = q2 + H;
                for (int h = 0; h < H; h++)
                    d[h] = (((d[h] + q0[h] * c0) + q1[h] * c1) + q2[h] * c2) + q3[h] * c3;
            }
            for (; j < kvl; j++) {
                const double cj = (double)cs[j];
                const double *qj = qat + (size_t)j * H;
                for (int h = 0; h < H; h++) d[h] += qj[h] * cj;
            }
            for (int i = 0; i < qr; i++) {
                const double ri = (double)kr[i];
                const double *qi = qpt + (size_t)i * H;
                for (int h = 0; h < H; h++) d[h] += qi[h] * ri;
            }
            for (int h = 0; h < H; h++) sc[((size_t)t * H + h) * N + s] = (float)d[h] * scale;
        }
    }

    /* 3. Softmax, the engine's form. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int th = 0; th < T * H; th++) {
        const int t = th / H, h = th % H;
        float *r = sc + (size_t)th * N;
        mla_probe_row(probe, r, t, h, H, N, C + t);
        mla_probe_z(t, h, H, mla_softmax_row(r, C + t, mla_qrow(t, h, H, N)));
        float *ut = u + (size_t)th * kvl;
        for (int j = 0; j < kvl; j++) ut[j] = 0.0f;
    }

    /* 4. u[t,h] = sum_s p_s c_s in float, s ascending: blocks ascending, positions
     *    ascending within a block, and each head owned by one thread for the whole loop,
     *    so no term changes place. Blocks outer so that one block of latent rows serves
     *    every head while it is in cache. No barrier per block (nowait): OpenMP assigns
     *    iterations of static loops with the same trip count in the same parallel
     *    region to the same threads, so a thread only ever touches its own heads and
     *    has nothing to wait for. With a barrier, one descheduled thread stalls all the
     *    others once per block, a thousand times per call at 16K positions. */
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (int s0 = 0; s0 < N; s0 += MLA_BLOCK) {
        const int s1 = s0 + MLA_BLOCK < N ? s0 + MLA_BLOCK : N;
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++) {
                const int e = s1 < C + t + 1 ? s1 : C + t + 1;
                const float *pt = sc + ((size_t)t * H + h) * N;
                float *ut = u + ((size_t)t * H + h) * kvl;
                for (int s = s0; s < e; s++) {
                    const float pr = pt[s];
                    const float *cs = k->kv + (size_t)s * kvl;
                    for (int j = 0; j < kvl; j++) ut[j] += pr * cs[j];
                }
            }
        }
    }

    /* 5. out[t,h] = W_uv[h] u[t,h], through the engine kernel, on a view of kv_b. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int th = 0; th < T * H; th++) {
        const int h = th % H;
        k3_mmw(acc + (size_t)th * vh, u + (size_t)th * kvl,
               mla_row(w->kv_b, w->wdt, kvl, (size_t)h * kvd + qn), w->wdt, kvl, vh);
    }
    return 0;
}

/* ---------------------------------------------------------------- driver ---- */
/* Append the T new tokens in the variant's layout, then attend. ct is mla_project's
 * output; the cache must hold C earlier positions of the right layout and room for T
 * more. Returns 0, or -1 if the variant cannot run this weight layout. */
static int mla_attend(int v, float *acc, const float *q, const float *ct, int T, int C,
                      MlaCache *k, const K3MlaW *w, const K3Cfg *c, void *scratch, int vcap,
                      float *probe)
{
    mla_append(k, mla_is_latent(v), ct, T, C, w, c);
    switch (v) {
    case MLA_E:  mla_attend_E(acc, q, T, C, k, c, scratch, probe); return 0;
    case MLA_EP: mla_attend_EP(acc, q, T, C, k, c, scratch, probe); return 0;
    case MLA_L0: mla_attend_L0(acc, q, T, C, k, w, c, scratch, probe); return 0;
    case MLA_L1: mla_attend_L1(acc, q, T, C, k, w, c, scratch, vcap, probe); return 0;
    case MLA_A:  return mla_attend_A(acc, q, T, C, k, w, c, scratch, probe);
    }
    return -1;
}

/* Full kv_b applications (mla_kvb calls) per call, the unit of latent-path cost (and,
 * under --trunk-rows, of 25 MB kv_b re-reads from disk at K3 dimensions). They depend
 * only on (C, T, vcap), never on the geometry, which is why the test and bench_mla's
 * `counts` mode can count them on the fixture geometry and quote them for K3's.
 *   E, E+  T: each new token's k and v are built once, when it is appended.
 *   L0     2 * sum_t (C + t + 1) = 2T(C + 1) + T(T - 1): every visible position, twice,
 *          per query token. At C = 0 (a prefill) that is T(T + 1), QUADRATIC in T.
 *   L1     N + (N - vcap) with N = C + T: once per position per call, plus a second
 *          time for the positions beyond the value-row budget. LINEAR in T.
 *   A      0: A never applies the whole of kv_b to anything. Its absorption (W_uk^T q)
 *          and output (W_uv u) are per-head slices, T kv_b's worth of multiply-adds in
 *          all, which mla_macs counts. */
static inline double mla_rebuilds(int v, int T, int C, int vcap)
{
    const double t = T, n = (double)C + T;
    if (vcap > C + T) vcap = C + T;
    if (vcap < 0) vcap = 0;
    switch (v) {
    case MLA_E: case MLA_EP: return t;                        /* appending new tokens */
    case MLA_L0: return 2.0 * (t * ((double)C + 1.0) + t * (t - 1.0) / 2.0);
    case MLA_L1: return n + (n - (double)vcap);
    case MLA_A:  return 0.0;
    }
    return 0.0;
}

/* Multiply-adds per call, counted from the loops above (the latent paths' kv_b MACs
 * included), so the timings can be read against a count that does not move. */
static inline double mla_macs(int v, const K3Cfg *c, int T, int C, int vcap)
{
    const double H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const double kvl = c->kv_lora, kvb = H * (qn + vh) * kvl;
    double seen = 0.0;                                   /* sum over t of (C + t + 1) */
    for (int t = 0; t < T; t++) seen += (double)C + t + 1;
    const double attn = H * (qn + qr + vh) * seen;
    switch (v) {
    case MLA_E: case MLA_EP: case MLA_L0: case MLA_L1:
        return mla_rebuilds(v, T, C, vcap) * kvb + attn;
    case MLA_A:
        return T * kvb + H * (2.0 * kvl + qr) * seen;
    }
    return 0.0;
}

/* ------------------------------------------------------- synthetic inputs ---- */
/* Shared by the benchmark and the test so both draw the same kind of layer. A layer is
 * reproducible from (config, weight type, seed); each matrix has its own stream so that
 * dropping the gate does not shift the others. */
typedef struct { uint64_t s; } MlaRng;

static inline uint64_t mla_rng_next(MlaRng *r)
{
    r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17;   /* xorshift64 */
    return r->s;
}
static inline void mla_rng_seed(MlaRng *r, uint64_t seed)
{
    r->s = seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
    if (!r->s) r->s = 1;
    for (int i = 0; i < 8; i++) (void)mla_rng_next(r);
}
/* Uniform on the open interval (0, 1), 53 bits. */
static inline double mla_rng_unit(MlaRng *r)
{
    return ((double)(mla_rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
/* n normal deviates with standard deviation sd, Box-Muller in pairs. */
static void mla_fill_normal(float *p, size_t n, float sd, uint64_t seed)
{
    MlaRng r;
    mla_rng_seed(&r, seed);
    for (size_t i = 0; i < n; i += 2) {
        const double a = sqrt(-2.0 * log(mla_rng_unit(&r)));
        const double b = 6.283185307179586 * mla_rng_unit(&r);
        p[i] = (float)(sd * a * cos(b));
        if (i + 1 < n) p[i + 1] = (float)(sd * a * sin(b));
    }
}
/* float -> bf16, round to nearest even. The inputs are finite normals. */
static inline uint16_t mla_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

typedef struct {
    K3MlaW w;
    void  *m[6];                    /* q_a, q_b, kv_a, kv_b, o, g */
    float *norm;                    /* q_a_norm then kv_a_norm */
} MlaSynth;

static void mla_synth_free(MlaSynth *S)
{
    for (int i = 0; i < 6; i++) free(S->m[i]);
    free(S->norm);
    memset(S, 0, sizeof *S);
}

/* Weights ~ N(0, sd) stored as wdt (K3_WBF16 or K3_WF32); norm weights 1 + N(0, nsd).
 * need_all = 0 builds kv_b alone, which is all the timing runs touch. Returns 0/-1. */
static int mla_synth_layer(MlaSynth *S, const K3Cfg *c, int wdt, float sd, float nsd,
                           uint64_t seed, int need_all)
{
    const size_t E = (size_t)c->hidden, H = (size_t)c->n_heads;
    const size_t qh = (size_t)(c->qk_nope + c->qk_rope), kvd = (size_t)(c->qk_nope + c->v_head);
    const size_t kvw = (size_t)(c->kv_lora + c->qk_rope), HV = H * (size_t)c->v_head;
    const size_t rows[6] = {(size_t)c->q_lora, H * qh, kvw, H * kvd, E, HV};
    const size_t cols[6] = {E, (size_t)c->q_lora, E, (size_t)c->kv_lora, HV, E};
    memset(S, 0, sizeof *S);
    for (int i = 0; i < 6; i++) {
        if (!need_all && i != 3) continue;
        if (i == 5 && !c->mla_out_gate) continue;
        const size_t n = rows[i] * cols[i];
        float *f = (float *)malloc(n * sizeof(float));
        if (!f) { mla_synth_free(S); return -1; }
        mla_fill_normal(f, n, sd, seed + 101u * (uint64_t)(i + 1));
        if (wdt == K3_WBF16) {
            uint16_t *b = (uint16_t *)malloc(n * sizeof(uint16_t));
            if (!b) { free(f); mla_synth_free(S); return -1; }
            for (size_t j = 0; j < n; j++) b[j] = mla_bf16(f[j]);
            free(f);
            S->m[i] = b;
        } else {
            S->m[i] = f;
        }
    }
    const size_t nn = (size_t)c->q_lora + (size_t)c->kv_lora;
    S->norm = (float *)malloc(nn * sizeof(float));
    if (!S->norm) { mla_synth_free(S); return -1; }
    mla_fill_normal(S->norm, nn, nsd, seed + 7u);
    for (size_t j = 0; j < nn; j++) S->norm[j] += 1.0f;
    S->w.q_a = S->m[0]; S->w.q_b = S->m[1]; S->w.kv_a = S->m[2];
    S->w.kv_b = S->m[3]; S->w.o = S->m[4]; S->w.g = S->m[5];
    S->w.q_a_norm = S->norm;
    S->w.kv_a_norm = S->norm + c->q_lora;
    S->w.wdt = wdt;
    return 0;
}

/* The released geometry, and the reference fixture's. hidden may be overridden. */
static inline void mla_cfg_k3(K3Cfg *c, int hidden)
{
    memset(c, 0, sizeof *c);
    c->hidden = hidden; c->rms_eps = 1e-5f;
    c->n_heads = 96; c->q_lora = 1536; c->kv_lora = 512;
    c->qk_nope = 128; c->qk_rope = 64; c->v_head = 128; c->mla_out_gate = 1;
}
static inline void mla_cfg_tiny(K3Cfg *c)
{
    memset(c, 0, sizeof *c);
    c->hidden = 128; c->rms_eps = 1e-5f;
    c->n_heads = 4; c->q_lora = 64; c->kv_lora = 32;
    c->qk_nope = 24; c->qk_rope = 8; c->v_head = 16; c->mla_out_gate = 1;
}

#endif /* MLA_VARIANTS_H */
