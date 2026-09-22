/* test_matmul_exact.c - the decode matmuls against plain re-implementations of their
 * own summation order, bit for bit.
 *
 * WHY THIS EXISTS
 *   k3_matmul, k3_matmul_bf16 and k3_matmul_mxfp4 each have several implementations
 *   (scalar, AVX2, AVX-512, NEON; one row per pass or two; x widened in place or read
 *   from a hoisted copy in a permuted layout). The engine's claim is that the same
 *   binary emits the same bits at every memory budget and mode, and that the x86 builds
 *   agree with each other, so every one of those implementations must reproduce ONE
 *   summation order exactly. test_ops checks k3_matmul_bf16 against k3_matmul, but two
 *   kernels in the same build can share a mistake; bench_kernels compares hashes across
 *   builds, but only on benign data, where a double sum narrowed to float almost never
 *   shows a reordering. This file writes each order down independently, in the plainest
 *   C, and compares every output bit on data built so that a different order does NOT
 *   round to the same float.
 *
 * THE ORDERS
 *   bf16 and fp32 (every build): element i < in & ~15 is fused (one fma) into
 *   accumulator a[i % 16] in ascending i; b_l = (a[l] + a[l+4]) + (a[l+8] + a[l+12]);
 *   acc = (b0 + b1) + (b2 + b3); the in % 16 tail is fused sequentially into acc.
 *
 *   MXFP4 is per-ISA by contract (tests/unit/test_expert.c holds it to 1e-6 of
 *   dequantise-then-matmul; the exact bits are what each ISA emits):
 *     x86 (AVX2, and AVX-512 which must equal it), group % 16 == 0: the FLAT path.
 *       Weight = (double)(float)(E2M1 * scale), 16-element chunks into a[i % 16],
 *       chunks whose scale byte is 255 skipped; q_l = (a[l] + a[l+8]) + (a[l+4] +
 *       a[l+12]); acc = (q0 + q2) + (q1 + q3); tail fused sequentially, 255 NOT skipped.
 *     x86, other groups: per group, unscaled weights into a[j % 16] over 16- and then
 *       8-element steps, sub = (q0 + q1) + (q2 + q3), sub-8 tail fused into sub, then
 *       acc += sub * scale; 255 groups skipped.
 *     scalar and NEON: per group, s[j % 8] over 8-element steps, sub = ((s0 + s4) +
 *       (s1 + s5)) + ((s2 + s6) + (s3 + s7)), tail fused into sub, acc += sub * scale;
 *       255 groups skipped.
 *
 * WHY THE DATA HAS TEETH
 *   Every eleventh element of x (i % 11 == 3) holds the same HUGE value, 2^60. Each row
 *   pairs one or two of those positions with the next one (11 later), with weights of
 *   opposite sign, so each pair's products cancel EXACTLY in real arithmetic while
 *   being some 2^60 times larger than the ordinary products everywhere else. Because 11
 *   and 16 are coprime, the halves of a pair land in unrelated accumulators, and until
 *   the reduction brings them together each one swamps its accumulator, and whatever
 *   partial sum it is added to on the way: ordinary terms there lose all their bits.
 *   Which terms are lost depends on exactly which accumulator every element went to, in
 *   what order, and where the pair meets in the tree, and the loss is large next to the
 *   float result. So a misplaced lane, a different tree or a sequential sum changes the
 *   float, not merely the last bit of a double. Only one or two pairs per row: with many,
 *   every accumulator ends up huge and rounded so coarsely that the tree's additions
 *   become exact and stop telling trees apart. For MXFP4 a second scenario draws each
 *   row's scale from a range of 2^210 and pairs the HUGE positions ACROSS groups, so
 *   group sums cancel in the running total.
 *
 *   This is proved rather than assumed: every check also computes WRONG orders on the
 *   same data -- a sequential sum, the neighbouring tree, the lanes rotated by one, and
 *   lanes 0 and 1 swapped -- and requires each to be rejected by some rows of every
 *   large enough check and by at least a tenth of all such rows overall, so if the data
 *   ever stops discriminating, the test fails instead of passing vacuously. Only
 *   permutations that are not symmetries of the tree in question count as wrong:
 *   swapping lanes 2k and 2k+1 for every k, or the two halves, leaves both trees'
 *   values unchanged on any data (addition commutes), so a kernel doing that would
 *   still be exact; the same holds for a rotation by one under the MXFP4 flat tree,
 *   which is why that mutant is not applied there.
 *
 * Inputs are finite except where a case is ABOUT non-finite values (the NaN scale byte
 * skipping an infinite x, and E8M0 scales that overflow a float weight to inf). NaN
 * outputs, which only those cases can produce, compare equal to NaN.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static int g_pass = 0, g_fail = 0;

/* ------------------------------------------------------------- utilities ---- */
static uint32_t g_rng = 0x9E3779B9u;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
static int rndi(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

/* sign * 2^e * (1 + fraction), a float with a random 23-bit significand. */
static float rndf(int elo, int ehi)
{
    const float m = 1.0f + (float)(rnd() >> 9) / 8388608.0f;
    const float v = ldexpf(m, rndi(elo, ehi));
    return (rnd() & 1) ? -v : v;
}

static float bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}
static uint16_t f2bf16(float f)                   /* truncation: any bf16 will do */
{
    union { uint32_t u; float f; } v;
    v.f = f;
    return (uint16_t)(v.u >> 16);
}

static int same_bits(float a, float b)
{
    union { float f; uint32_t u; } x, y;
    x.f = a; y.f = b;
    if (isnan(a) && isnan(b)) return 1;
    return x.u == y.u;
}

static void report(int ok, const char *what)
{
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) g_pass++; else g_fail++;
}

static int is_huge(int i) { return i % 11 == 3; }

/* x: the shared HUGE value at the pairing positions, ordinary values elsewhere. */
static void fill_x(float *x, int in)
{
    for (int i = 0; i < in; i++) x[i] = is_huge(i) ? ldexpf(1.0f, 60) : rndf(-4, 4);
}

/* One row's HUGE pairs: role[i] is 1 for the first half, 2 for the second (i + 11),
 * 0 otherwise. `want` pairs at random starts where that fits; group > 0 keeps both
 * halves of a pair in one group. */
static void pick_pairs(unsigned char *role, int in, int group, int want)
{
    for (int i = 0; i < in; i++) role[i] = 0;
    for (int tries = 0; want > 0 && tries < 64; tries++) {
        const int i = 3 + 11 * rndi(0, in / 11);
        if (i + 11 >= in || role[i] || role[i + 11]) continue;
        if (group > 0 && i / group != (i + 11) / group) continue;
        role[i] = 1; role[i + 11] = 2; want--;
    }
}

/* Teeth bookkeeping: per mutant, rows rejecting it and rows eligible, over all checks. */
static int g_teeth_differ[8], g_teeth_rows[8];

/* Mutants: wrong orders the data must reject. */
enum { M_RIGHT, M_SEQ, M_TREE, M_ROT, M_SWAP, M_COUNT };
static const char *MNAME[M_COUNT] = { "right", "sequential", "other tree",
                                      "lanes rotated", "lanes 0,1 swapped" };

/* Where element slot l goes: accumulator l, or a mutant's misplacement of it. */
static int lane(int l, int mut, int nl)
{
    if (mut == M_ROT) return (l + 1) % nl;
    if (mut == M_SWAP && l < 2) return l ^ 1;
    return l;
}

/* ----------------------------------------------- bf16 / fp32 reference ---- */
static float ref_dot16(const float *w, const float *x, int in, int mut)
{
    const int n16 = in & ~15;
    double acc;
    if (mut == M_SEQ) {
        acc = 0.0;
        for (int i = 0; i < n16; i++) acc = fma((double)w[i], (double)x[i], acc);
    } else {
        double a[16] = {0};
        for (int i = 0; i < n16; i += 16)
            for (int l = 0; l < 16; l++) {
                const int k = lane(l, mut, 16);
                a[k] = fma((double)w[i + l], (double)x[i + l], a[k]);
            }
        if (mut != M_TREE) {
            const double b0 = (a[0] + a[4]) + (a[8] + a[12]);
            const double b1 = (a[1] + a[5]) + (a[9] + a[13]);
            const double b2 = (a[2] + a[6]) + (a[10] + a[14]);
            const double b3 = (a[3] + a[7]) + (a[11] + a[15]);
            acc = (b0 + b1) + (b2 + b3);
        } else {                                  /* the MXFP4 flat path's tree */
            double q[4];
            for (int l = 0; l < 4; l++) q[l] = (a[l] + a[l + 8]) + (a[l + 4] + a[l + 12]);
            acc = (q[0] + q[2]) + (q[1] + q[3]);
        }
    }
    for (int i = n16; i < in; i++) acc = fma((double)w[i], (double)x[i], acc);
    return (float)acc;
}

/* One row's weights as bf16 bit patterns: the HUGE pairs with opposite signs, zero at
 * the other HUGE positions, ordinary weights elsewhere. `raw` rows use uniformly random
 * finite bit patterns instead -- denormals, zeros, -0, huge and tiny exponents -- for
 * coverage of the widening rather than of the order. */
static void fill_w_bf16(uint16_t *w, int in, int raw, const unsigned char *role)
{
    for (int i = 0; i < in; i++) {
        if (raw) {
            uint16_t h = (uint16_t)(rnd() >> 16);
            if (((h >> 7) & 0xFF) == 0xFF) h &= 0x7F7Fu;   /* no inf/NaN */
            w[i] = h;
        } else if (role[i] == 1) {
            w[i] = f2bf16(rndf(0, 8));
        } else if (role[i] == 2) {
            w[i] = (uint16_t)(w[i - 11] ^ 0x8000u);        /* exact negation */
        } else if (is_huge(i)) {
            w[i] = 0;
        } else {
            w[i] = f2bf16(rndf(-4, 4));
        }
    }
}

static void check_dense(int in, int out)
{
    uint16_t *wb = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
    float *wf = (float *)malloc((size_t)in * out * sizeof(float));
    float *x = (float *)malloc((size_t)in * sizeof(float));
    float *yb = (float *)malloc((size_t)out * sizeof(float));
    float *yf = (float *)malloc((size_t)out * sizeof(float));
    float *wrow = (float *)malloc((size_t)in * sizeof(float));
    unsigned char *role = (unsigned char *)malloc((size_t)in);
    if (!wb || !wf || !x || !yb || !yf || !wrow || !role) {
        report(0, "dense (alloc)");
        goto done;
    }

    fill_x(x, in);
    for (int o = 0; o < out; o++) {
        const int raw = (o % 5) == 4;
        const uint16_t *br = wb + (size_t)o * in;
        float *fr = wf + (size_t)o * in;
        pick_pairs(role, in, 0, 1 + (o & 1));
        fill_w_bf16(wb + (size_t)o * in, in, raw, role);
        /* The fp32 rows are the bf16 values plus random low bits, so the fp32 kernel is
         * checked on full 24-bit weights, not only on widened bf16 ones; each HUGE pair
         * is re-negated after, so it still cancels exactly. */
        for (int i = 0; i < in; i++) {
            union { uint32_t u; float f; } v;
            v.f = bf16f(br[i]);
            if ((v.u & 0x7F800000u) != 0 && (v.u & 0x7F800000u) != 0x7F800000u)
                v.u |= rnd() & 0xFFFFu;
            fr[i] = v.f;
            if (!raw && role[i] == 2) fr[i] = -fr[i - 11];
        }
    }
    k3_matmul_bf16(yb, x, wb, in, out);
    k3_matmul(yf, x, wf, in, out);

    int bad_b = 0, bad_f = 0, structured = 0, differ[M_COUNT] = {0};
    for (int o = 0; o < out; o++) {
        const float *fr = wf + (size_t)o * in;
        for (int i = 0; i < in; i++) wrow[i] = bf16f(wb[(size_t)o * in + i]);
        const float rb = ref_dot16(wrow, x, in, M_RIGHT);
        const float rf = ref_dot16(fr, x, in, M_RIGHT);
        bad_b += !same_bits(rb, yb[o]);
        bad_f += !same_bits(rf, yf[o]);
        if ((o % 5) != 4) {                       /* both weight sets, every mutant */
            structured += 2;
            for (int m = M_SEQ; m < M_COUNT; m++) {
                differ[m] += !same_bits(rb, ref_dot16(wrow, x, in, m));
                differ[m] += !same_bits(rf, ref_dot16(fr, x, in, m));
            }
        }
    }

    char msg[320];
    snprintf(msg, sizeof msg, "bf16  in=%-5d out=%-4d %d/%d rows differ from the reference order",
             in, out, bad_b, out);
    report(bad_b == 0, msg);
    snprintf(msg, sizeof msg, "fp32  in=%-5d out=%-4d %d/%d rows differ from the reference order",
             in, out, bad_f, out);
    report(bad_f == 0, msg);
    /* Teeth, where there is room for an order to go wrong: sixteen full chunks. */
    if (in >= 256 && structured >= 8) {
        int ok = 1, n = snprintf(msg, sizeof msg, "      in=%-5d wrong orders rejected:", in);
        for (int m = M_SEQ; m < M_COUNT; m++) {
            ok &= differ[m] > 0;
            g_teeth_differ[m] += differ[m];
            g_teeth_rows[m] += structured;
            n += snprintf(msg + n, sizeof msg - (size_t)n, " %s %d/%d%s", MNAME[m],
                          differ[m], structured, m + 1 < M_COUNT ? "," : "");
        }
        report(ok, msg);
    }
done:
    free(wb); free(wf); free(x); free(yb); free(yf); free(wrow); free(role);
}

/* ----------------------------------------------------------- MXFP4 ---- */
static const float E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static float e8m0(unsigned b) { return b == 255 ? 0.0f : ldexpf(1.0f, (int)b - 127); }
static unsigned nib(const unsigned char *pr, int i)
{
    return (i & 1) ? (unsigned)(pr[i >> 1] >> 4) : (unsigned)(pr[i >> 1] & 0x0F);
}
static void set_nib(unsigned char *pr, int i, unsigned c)
{
    if (i & 1) pr[i >> 1] = (unsigned char)((pr[i >> 1] & 0x0F) | (c << 4));
    else       pr[i >> 1] = (unsigned char)((pr[i >> 1] & 0xF0) | c);
}

enum { MX_FLAT, MX_GROUPED16, MX_GROUPED8 };

static float ref_mx(int kind, const unsigned char *pr, const unsigned char *sr,
                    const float *x, int in, int group, int mut)
{
    if (kind == MX_FLAT) {
        double a[16] = {0}, acc = 0.0;
        int i = 0;
        for (; i + 15 < in; i += 16) {
            const unsigned sb = sr[i / group];
            if (sb == 255) continue;
            for (int l = 0; l < 16; l++) {
                const double w = (double)(E2M1[nib(pr, i + l)] * e8m0(sb));
                if (mut == M_SEQ) { acc = fma(w, (double)x[i + l], acc); continue; }
                const int k = lane(l, mut, 16);
                a[k] = fma(w, (double)x[i + l], a[k]);
            }
        }
        if (mut != M_SEQ) {
            double q[4];
            for (int l = 0; l < 4; l++) q[l] = (a[l] + a[l + 8]) + (a[l + 4] + a[l + 12]);
            acc = (mut == M_TREE) ? (q[0] + q[1]) + (q[2] + q[3])
                                  : (q[0] + q[2]) + (q[1] + q[3]);
        }
        for (; i < in; i++)
            acc = fma((double)(E2M1[nib(pr, i)] * e8m0(sr[i / group])), (double)x[i], acc);
        return (float)acc;
    }
    const int ngrp = (in + group - 1) / group;
    const int nl = kind == MX_GROUPED16 ? 16 : 8;
    double acc = 0.0;
    for (int g = 0; g < ngrp; g++) {
        const unsigned sb = sr[g];
        if (sb == 255) continue;
        const int base = g * group;
        const int n = (in - base < group) ? in - base : group;
        double a[16] = {0}, sub = 0.0;
        int i = 0;
        /* grouped16 takes 16-element steps, then one 8-element step, into a[j % 16];
         * grouped8 takes 8-element steps into s[j % 8]. */
        for (; i + nl - 1 < n; i += nl)
            for (int l = 0; l < nl; l++) {
                const double t = (double)E2M1[nib(pr, base + i + l)];
                if (mut == M_SEQ) { sub = fma(t, (double)x[base + i + l], sub); continue; }
                const int k = lane(l, mut, nl);
                a[k] = fma(t, (double)x[base + i + l], a[k]);
            }
        if (nl == 16)
            for (; i + 7 < n; i += 8)
                for (int l = 0; l < 8; l++) {
                    const double t = (double)E2M1[nib(pr, base + i + l)];
                    if (mut == M_SEQ) { sub = fma(t, (double)x[base + i + l], sub); continue; }
                    const int k = lane(l, mut, 16);
                    a[k] = fma(t, (double)x[base + i + l], a[k]);
                }
        if (mut != M_SEQ) {
            if (nl == 16) {
                double q[4];
                for (int l = 0; l < 4; l++) q[l] = (a[l] + a[l + 8]) + (a[l + 4] + a[l + 12]);
                sub = (mut == M_TREE) ? (q[0] + q[2]) + (q[1] + q[3])
                                      : (q[0] + q[1]) + (q[2] + q[3]);
            } else {
                sub = (mut == M_TREE)
                    ? ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
                    : ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            }
        }
        for (; i < n; i++)
            sub = fma((double)E2M1[nib(pr, base + i)], (double)x[base + i], sub);
        acc += sub * (double)e8m0(sb);
    }
    return (float)acc;
}

/* Which order this build's k3_matmul_mxfp4 promises (see the top of the file). */
static int mx_kind(int group)
{
#if defined(__AVX2__)
    return (group & 15) == 0 ? MX_FLAT : MX_GROUPED16;
#else
    (void)group;
    return MX_GROUPED8;
#endif
}

enum { SC_HUGE, SC_INF, SC_NAN, SC_NAN_TAIL, SC_WIDE, SC_COUNT };

/* SC_HUGE  HUGE pairs within each group (where the two scales agree), ordinary scales
 *          2^-3..2^3, plus three groups -- the same group index in every row -- that
 *          exercise the scale bytes with special handling, their HUGE positions given
 *          zero weights:
 *            HIGH  scales 250..254 with x shrunk by 2^-120 there, so the products stay
 *                  ordinary. At 253 and 254 the codes are limited to |w| <= 1.5,
 *                  which is where the float weight does not yet overflow.
 *            LOW   scales 0..3 with x grown by 2^120 (so still below 2^128): 0 and 1
 *                  give SUBNORMAL float weights, 2 and 3 the smallest normal ones.
 *            NAN   scale 255 in about half the rows: the group is skipped there.
 * SC_INF   one group per row with scale 253 or 254 and weights that DO overflow to
 *          +inf in float; x is tiny and positive there, so a table that stayed finite
 *          would give a finite row where the right one gives +inf.
 * SC_NAN   scale 255 on one group for every row, with x = +inf at one element of that
 *          group inside the whole 16-element chunks: a skipped chunk contributes
 *          nothing, a zero-weight product would be 0 * inf = NaN.
 * SC_NAN_TAIL  the same with the +inf in the last in % 16 elements, where the flat
 *          path's scalar tail does NOT skip a 255 group (so those rows are NaN there,
 *          and finite on the grouped paths), and the reference says so.
 * SC_WIDE  one scale per row, drawn from 2^-105..2^105, and the HUGE positions paired
 *          ACROSS groups, so whole group sums cancel in the running total.
 * HUGE positions outside a pair, and all of them in the three special groups, get
 * weight zero (code 0). */
static void check_mxfp4(int in, int rows, int group, int scenario)
{
    const int pcols = in / 2, ngrp = (in + group - 1) / group;
    unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
    unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
    float *x = (float *)malloc((size_t)in * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float));
    unsigned char *role = (unsigned char *)malloc((size_t)in);
    if (!pk || !sc || !x || !y || !role) { report(0, "mxfp4 (alloc)"); goto done; }

    fill_x(x, in);
    const int gspec = rndi(0, ngrp - 1);
    const int huge_groups = scenario == SC_HUGE && ngrp >= 4;
    const int ghigh = huge_groups ? 1 : -1, glow = huge_groups ? 2 : -1;
    const int gnan = huge_groups ? 3 : -1;
    if (scenario == SC_HUGE)
        for (int i = 0; i < in; i++) {
            if (is_huge(i)) continue;             /* their weights are zero there */
            if (i / group == ghigh) x[i] = ldexpf(x[i], -120);
            if (i / group == glow)  x[i] = ldexpf(x[i], 120);
        }
    int gnanx = gspec;                            /* the group holding x = +inf */
    if (scenario == SC_NAN) {
        const int at = rndi(0, (in & ~15) - 1);
        x[at] = INFINITY;
        gnanx = at / group;
    }
    if (scenario == SC_NAN_TAIL) {
        const int at = rndi(in & ~15, in - 1);
        x[at] = INFINITY;
        gnanx = at / group;
    }
    for (int r = 0; r < rows; r++) {
        unsigned char *pr = pk + (size_t)r * pcols;
        unsigned char *sr = sc + (size_t)r * ngrp;
        for (int j = 0; j < pcols; j++) pr[j] = (unsigned char)(rnd() >> 24);
        const int wide = rndi(22, 232);
        for (int g = 0; g < ngrp; g++)
            sr[g] = (unsigned char)(scenario == SC_WIDE ? wide : rndi(124, 130));
        /* HUGE pairs with opposite signs: within a group, where the two scales agree,
         * or (SC_WIDE, one scale per row) across groups. */
        pick_pairs(role, in, scenario == SC_WIDE ? 0 : group, 1 + (r & 1));
        for (int i = 0; i < in; i++) {
            if (!is_huge(i)) continue;
            const int g = i / group;
            const int special = g == ghigh || g == glow || g == gnan;
            if (role[i] == 1 && !special)
                set_nib(pr, i, (unsigned)rndi(1, 7) | (rnd() & 8u));
            else if (role[i] == 2 && !special && nib(pr, i - 11) != 0)
                set_nib(pr, i, nib(pr, i - 11) ^ 8u);
            else
                set_nib(pr, i, 0);
        }
        if (scenario == SC_HUGE && huge_groups) {
            sr[ghigh] = (unsigned char)rndi(250, 254);
            if (sr[ghigh] >= 253)                 /* codes 0..3 and 8..11: |w| <= 1.5 */
                for (int i = ghigh * group; i < (ghigh + 1) * group && i < in; i++)
                    set_nib(pr, i, nib(pr, i) & 0xBu);
            sr[glow] = (unsigned char)rndi(0, 3);
            if (rnd() & 1) sr[gnan] = 255;
        }
        if (scenario == SC_INF) {
            const int lo = gspec * group, hi = (lo + group < in) ? lo + group : in;
            sr[gspec] = (unsigned char)(253 + (r & 1));
            for (int i = lo; i < hi; i++) set_nib(pr, i, (unsigned)rndi(4, 7));  /* 2..6 */
        }
        if (scenario == SC_NAN || scenario == SC_NAN_TAIL) sr[gnanx] = 255;
    }
    if (scenario == SC_INF) {
        const int lo = gspec * group, hi = (lo + group < in) ? lo + group : in;
        for (int i = lo; i < hi; i++) x[i] = ldexpf(1.0f, -100);
    }

    k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);

    const int kind = mx_kind(group);
    int bad = 0, differ[M_COUNT] = {0}, other = 0;
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = pk + (size_t)r * pcols;
        const unsigned char *sr = sc + (size_t)r * ngrp;
        const float want = ref_mx(kind, pr, sr, x, in, group, M_RIGHT);
        bad += !same_bits(want, y[r]);
        for (int m = M_SEQ; m < M_COUNT; m++)
            differ[m] += !same_bits(want, ref_mx(kind, pr, sr, x, in, group, m));
        other += !same_bits(want, ref_mx(kind == MX_FLAT ? MX_GROUPED16 : MX_FLAT,
                                         pr, sr, x, in, group, M_RIGHT));
    }
    static const char *names[] = { "flat", "grouped16", "grouped8" };
    static const char *scen[SC_COUNT] = { "huge+special", "inf-weight", "nan-scale-skip",
                                          "nan-scale-tail", "wide-scale" };
    char msg[320];
    snprintf(msg, sizeof msg, "mxfp4 in=%-5d rows=%-4d group=%-2d %-9s %-14s %d/%d rows differ",
             in, rows, group, names[kind], scen[scenario], bad, rows);
    report(bad == 0, msg);
    /* Teeth: enough elements for several HUGE pairs, and a few rows. The grouped
     * paths' mutants act only inside a group, so they need groups of 16 or more. */
    if ((scenario == SC_HUGE || scenario == SC_WIDE) && in >= 256 && rows >= 8 &&
        (kind == MX_FLAT || group >= 16)) {
        int ok = other > 0;
        int n = snprintf(msg, sizeof msg, "      in=%-5d group=%-2d wrong orders rejected: "
                         "other partition %d/%d,", in, group, other, rows);
        for (int m = M_SEQ; m < M_COUNT; m++) {
            if (m == M_ROT && kind == MX_FLAT) continue;   /* a symmetry of that tree */
            ok &= differ[m] > 0;
            g_teeth_differ[m] += differ[m];
            g_teeth_rows[m] += rows;
            n += snprintf(msg + n, sizeof msg - (size_t)n, " %s %d/%d%s", MNAME[m],
                          differ[m], rows, m + 1 < M_COUNT ? "," : "");
        }
        report(ok, msg);
    }
    if (scenario == SC_INF && kind == MX_FLAT) {  /* grouped paths scale after the sum */
        int inf_rows = 0;
        for (int r = 0; r < rows; r++) inf_rows += (y[r] == INFINITY);
        snprintf(msg, sizeof msg, "      in=%-5d group=%-2d overflowing scale gives +inf in %d/%d rows",
                 in, group, inf_rows, rows);
        report(inf_rows == rows, msg);
    }
done:
    free(pk); free(sc); free(x); free(y); free(role);
}

int main(void)
{
    printf("decode matmuls against plain re-implementations of their summation order\n");
#if defined(__AVX512F__)
    printf("built WITH AVX-512 (MXFP4 order: the AVX2 flat/grouped16 paths)\n\n");
#elif defined(__AVX2__)
    printf("built WITH AVX2 (MXFP4 order: flat for group %% 16 == 0, else grouped16)\n\n");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    printf("built WITH NEON (MXFP4 order: grouped8)\n\n");
#else
    printf("built WITHOUT AVX2 or NEON (MXFP4 order: grouped8)\n\n");
#endif

    /* Shapes with every tail: in % 16 and in % 32 nonzero, fewer elements than one
     * chunk, odd row counts (a row paired with itself), and row counts either side of
     * the kernels' OpenMP threshold of 64. */
    static const int din[] = { 1, 5, 15, 16, 17, 31, 33, 48, 63, 65, 100, 257, 1030, 7174 };
    static const int dout[] = { 1, 2, 3, 7, 64, 65, 66, 129 };
    for (size_t a = 0; a < sizeof din / sizeof din[0]; a++)
        check_dense(din[a], dout[(a * 3) % (sizeof dout / sizeof dout[0])]);
    check_dense(7168, 67);                        /* the trunk's width, exactly */

    /* MXFP4: in even (a precondition), groups that take each path on x86 (16, 32, 48,
     * 64 flat; 8, 10, 24, 40 grouped), K3's 3584 and 3072 widths. */
    static const int mgroup[] = { 32, 16, 48, 64, 8, 10, 24, 40 };
    static const int min_[] = { 2, 14, 30, 34, 62, 96, 98, 142, 254, 258, 1090, 3584, 3072 };
    static const int mrows[] = { 1, 3, 8, 65, 66, 129 };
    for (size_t g = 0; g < sizeof mgroup / sizeof mgroup[0]; g++)
        for (size_t a = 0; a < sizeof min_ / sizeof min_[0]; a++) {
            const int rows = mrows[(g + a) % (sizeof mrows / sizeof mrows[0])];
            check_mxfp4(min_[a], rows, mgroup[g], SC_HUGE);
        }
    for (size_t g = 0; g < sizeof mgroup / sizeof mgroup[0]; g++) {
        check_mxfp4(3584, 65, mgroup[g], SC_WIDE);
        check_mxfp4(1090, 66, mgroup[g], SC_WIDE);
        check_mxfp4(3584, 65, mgroup[g], SC_INF);
        check_mxfp4(1090, 66, mgroup[g], SC_INF);
        for (int rep = 0; rep < 3; rep++) {
            check_mxfp4(3584, 9, mgroup[g], SC_NAN);
            check_mxfp4(1094, 9, mgroup[g], SC_NAN);
            check_mxfp4(1094, 9, mgroup[g], SC_NAN_TAIL);   /* 1094 % 16 == 6 */
            check_mxfp4(1066, 9, mgroup[g], SC_NAN_TAIL);   /* 1066 % 16 == 10 */
        }
    }

#if defined(__AVX2__)
    /* The same order under flush-to-zero and denormals-are-zero. The engine never sets
     * them, but a host built with -ffast-math does (crtfastmath), and the AVX-512 MXFP4
     * table is built so that it still agrees with AVX2 there: scale bytes 0 and 1, whose
     * float weights are subnormal, go through the float multiply AVX2 uses rather than
     * the exact double one, and flush the same way. rows <= 64 keeps these calls on this
     * thread, the only one whose MXCSR is changed. */
    {
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040u);                /* FTZ (bit 15) | DAZ (bit 6) */
        for (size_t g = 0; g < 4; g++) {          /* the flat-path groups */
            check_mxfp4(3584, 64, mgroup[g], SC_HUGE);
            check_mxfp4(1090, 33, mgroup[g], SC_HUGE);
        }
        check_dense(7168, 64);
        _mm_setcsr(csr);
        printf("  (the nine checks above ran with FTZ and DAZ set)\n");
    }
#endif

    /* Overall teeth: each wrong order rejected by at least a tenth of eligible rows. */
    {
        char msg[320];
        int ok = 1, n = snprintf(msg, sizeof msg, "all   wrong orders rejected overall:");
        for (int m = M_SEQ; m < M_COUNT; m++) {
            ok &= g_teeth_rows[m] > 0 && g_teeth_differ[m] * 10 >= g_teeth_rows[m];
            n += snprintf(msg + n, sizeof msg - (size_t)n, " %s %d/%d%s", MNAME[m],
                          g_teeth_differ[m], g_teeth_rows[m], m + 1 < M_COUNT ? "," : "");
        }
        report(ok, msg);
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail) {
        printf("MATMUL EXACTNESS FAILED\n");
        return 1;
    }
    printf("MATMUL EXACTNESS: every path reproduces its order bit for bit\n");
    return 0;
}
