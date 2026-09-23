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
 * THE SCALE BYTES WITH SPECIAL HANDLING get data of their own, where the ordinary
 *   tree data would not show a mistake: a NaN scale byte (255) beside an infinite x,
 *   on both rows of an AVX-512 row pair and on only one of them; E8M0 scales (253, 254)
 *   that overflow a float weight to inf, in whole chunks and in the scalar tail; and
 *   scales 0..3, whose subnormal float weights must flush under FTZ and DAZ exactly as
 *   AVX2's float multiply flushes them. Each carries its own check that the data can
 *   tell the right handling from the wrong one (a skipped row is finite, an overflow
 *   gives +inf, a weight formed in double differs under FTZ).
 *
 * Inputs are finite except where a case is ABOUT non-finite values (those scale-byte
 * cases, and the NaN rows below). Where the order is the question, NaN outputs compare
 * equal to NaN; check_nan_rows then holds the bf16 and fp32 kernels' NaN outputs to
 * the one quiet NaN, bit for bit, wherever a row sits in the call.
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

/* Mutants: wrong orders the data must reject. M_DOUBLEW is not an order but a wrong
 * MXFP4 flat weight (see mx_wflat), checked only by the scenarios built for it. */
enum { M_RIGHT, M_SEQ, M_TREE, M_ROT, M_SWAP, M_COUNT, M_DOUBLEW = M_COUNT };
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

/* NaN outputs of the bf16 and fp32 kernels: every one is the quiet NaN 0x7FC00000, so
 * a row's bits do not depend on where it sits in the call. The row pipeline splits a
 * matrix into calls whose length follows the memory budget, which moves a row between
 * the first and second place of a k3_matmul_bf16 pair, and the two places are separate
 * instruction sequences that can pass on different input NaNs. So each matrix is
 * multiplied whole and again from its second row on (every row one place earlier,
 * its pair parity flipped), and the two must agree bit for bit, NaN or not. Rows get
 * several NaNs with random signs and payloads, in x and in the weights, both in the
 * whole chunks and in the tail, so that NaNs meet in fmas and in the tree. */
static uint32_t fbits(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return v.u;
}
static float nan_bits(void)
{
    union { uint32_t u; float f; } v;
    v.u = 0x7FC00000u | (rnd() & 0x803FFFFFu);   /* quiet, random sign and payload */
    return v.f;
}

static void check_nan_rows(int in, int out, int xnan)
{
    uint16_t *wb = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
    float *wf = (float *)malloc((size_t)in * out * sizeof(float));
    float *x = (float *)malloc((size_t)in * sizeof(float));
    float *y = (float *)malloc((size_t)out * 4 * sizeof(float));
    if (!wb || !wf || !x || !y) { report(0, "nan rows (alloc)"); goto done; }

    for (int i = 0; i < in; i++) x[i] = rndf(-4, 4);
    if (xnan) x[rndi(0, in - 1)] = nan_bits();   /* then every row is NaN */
    for (int o = 0; o < out; o++) {
        for (int i = 0; i < in; i++) {
            wf[(size_t)o * in + i] = rndf(-4, 4);
            wb[(size_t)o * in + i] = f2bf16(rndf(-4, 4));
        }
        for (int k = rndi(0, 3); k > 0; k--) {
            const int at = rndi(0, in - 1);
            const float v = nan_bits();
            wf[(size_t)o * in + at] = v;
            wb[(size_t)o * in + at] = f2bf16(v);
        }
    }
    float *yb = y, *yb1 = y + out, *yf = y + 2 * out, *yf1 = y + 3 * out;
    k3_matmul_bf16(yb, x, wb, in, out);
    k3_matmul_bf16(yb1, x, wb + in, in, out - 1);
    k3_matmul(yf, x, wf, in, out);
    k3_matmul(yf1, x, wf + in, in, out - 1);

    int nan_b = 0, nan_f = 0, canon = 0, moved = 0;
    for (int o = 0; o < out; o++) {
        nan_b += yb[o] != yb[o];
        nan_f += yf[o] != yf[o];
        canon += (yb[o] != yb[o] && fbits(yb[o]) != 0x7FC00000u) +
                 (yf[o] != yf[o] && fbits(yf[o]) != 0x7FC00000u);
        if (o > 0)
            moved += (fbits(yb[o]) != fbits(yb1[o - 1])) + (fbits(yf[o]) != fbits(yf1[o - 1]));
    }
    char msg[320];
    snprintf(msg, sizeof msg, "nan   in=%-5d out=%-4d NaN rows bf16 %d, fp32 %d: %d not 0x7FC00000, "
             "%d change bits one place earlier", in, out, nan_b, nan_f, canon, moved);
    /* Some rows must be NaN, or the check says nothing. */
    report(canon == 0 && moved == 0 && nan_b > 0 && nan_f > 0, msg);
done:
    free(wb); free(wf); free(x); free(y);
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

/* The flat path's weight: the E2M1 value times the scale IN FLOAT, then widened, which
 * is how AVX2 forms it, so a product that is subnormal flushes to zero under FTZ and
 * one past 2^128 becomes inf. M_DOUBLEW forms the same product exactly in double
 * instead: it agrees everywhere else, and the SC_LOW and SC_INF data exist to tell the
 * two apart. */
static double mx_wflat(unsigned code, unsigned sb, int mut)
{
    if (mut == M_DOUBLEW) return sb == 255 ? 0.0 : (double)E2M1[code] * ldexp(1.0, (int)sb - 127);
    return (double)(E2M1[code] * e8m0(sb));
}

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
                const double w = mx_wflat(nib(pr, i + l), sb, mut);
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
            acc = fma(mx_wflat(nib(pr, i), sr[i / group], mut), (double)x[i], acc);
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

enum { SC_HUGE, SC_INF, SC_INF_TAIL, SC_NAN, SC_NAN_TAIL, SC_WIDE, SC_LOW, SC_COUNT };

/* Set while main runs checks with FTZ and DAZ on: SC_LOW's expectations depend on it. */
static int g_ftz = 0;

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
 * SC_INF_TAIL  the same in the LAST group, the one holding the in % 16 tail, with the
 *          overflowing weights only in the tail (the group's whole chunks, if it has
 *          any, keep |w| <= 1.5 and stay finite): the flat path's scalar tail must fold
 *          the scale in float too, where a double product would stay finite.
 * SC_NAN   scale 255 on one group in two rows of every three, with x = +inf at one
 *          element of that group inside the whole 16-element chunks: a skipped chunk
 *          contributes nothing, a zero-weight product would be 0 * inf = NaN. The
 *          pattern puts, among the kernels' row pairs (0,1), (2,3), ..., pairs where
 *          only the first row has the 255, only the second, and both, so a kernel that
 *          lets a 255 row through with its partner's decision is caught. A row without
 *          the 255 has its own non-finite result, which the reference gives too.
 * SC_NAN_TAIL  the same with the +inf in the last in % 16 elements, where the flat
 *          path's scalar tail does NOT skip a 255 group (so those rows are NaN there,
 *          and finite on the grouped paths), and the reference says so.
 * SC_WIDE  one scale per row, drawn from 2^-105..2^105, and the HUGE positions paired
 *          ACROSS groups, so whole group sums cancel in the running total.
 * SC_LOW   every group's scale 0..3 (0 and 1 make the float weights of codes +-0.5 and
 *          more SUBNORMAL, 2 and 3 are the smallest normal ones) with x grown by 2^120
 *          away from the HUGE positions, whose weights are zero, so no accumulator is
 *          swamped and a subnormal weight flushed or kept moves the float. Under FTZ and
 *          DAZ this is where a weight formed in double, not flushed as AVX2's float
 *          multiply flushes it, must show; without them the two agree, and must.
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
    int gspec = rndi(0, ngrp - 1);
    if (scenario == SC_INF_TAIL) gspec = ngrp - 1;
    const int huge_groups = scenario == SC_HUGE && ngrp >= 4;
    const int ghigh = huge_groups ? 1 : -1, glow = huge_groups ? 2 : -1;
    const int gnan = huge_groups ? 3 : -1;
    if (scenario == SC_HUGE)
        for (int i = 0; i < in; i++) {
            if (is_huge(i)) continue;             /* their weights are zero there */
            if (i / group == ghigh) x[i] = ldexpf(x[i], -120);
            if (i / group == glow)  x[i] = ldexpf(x[i], 120);
        }
    if (scenario == SC_LOW)
        for (int i = 0; i < in; i++)
            if (!is_huge(i)) x[i] = ldexpf(x[i], 120);
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
            sr[g] = (unsigned char)(scenario == SC_WIDE ? wide
                                    : scenario == SC_LOW ? rndi(0, 3) : rndi(124, 130));
        /* HUGE pairs with opposite signs: within a group, where the two scales agree,
         * or (SC_WIDE, one scale per row) across groups. */
        pick_pairs(role, in, scenario == SC_WIDE ? 0 : group, 1 + (r & 1));
        for (int i = 0; i < in; i++) {
            if (!is_huge(i)) continue;
            const int g = i / group;
            const int special = g == ghigh || g == glow || g == gnan || scenario == SC_LOW;
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
        if (scenario == SC_INF_TAIL) {
            /* Whole chunks: codes 0..3 and 8..11, |w| <= 1.5, finite at 253 and 254.
             * Tail: 2..6, the last element 4 or 6, which overflow at either scale. */
            const int lo = gspec * group, n16 = in & ~15;
            sr[gspec] = (unsigned char)(253 + (r & 1));
            for (int i = lo; i < in; i++)
                set_nib(pr, i, i < n16 ? nib(pr, i) & 0xBu : (unsigned)rndi(4, 7));
            set_nib(pr, in - 1, (unsigned)rndi(6, 7));
        }
        if (scenario == SC_NAN && r % 3 != 1) sr[gnanx] = 255;
        if (scenario == SC_NAN_TAIL) sr[gnanx] = 255;
    }
    if (scenario == SC_INF || scenario == SC_INF_TAIL) {
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
    static const char *scen[SC_COUNT] = { "huge+special", "inf-weight", "inf-weight-tail",
                                          "nan-scale-skip", "nan-scale-tail", "wide-scale",
                                          "low-scale" };
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
    if ((scenario == SC_INF || scenario == SC_INF_TAIL) && kind == MX_FLAT) {
        /* grouped paths scale after the sum, so only the flat path overflows */
        int inf_rows = 0;
        for (int r = 0; r < rows; r++) inf_rows += (y[r] == INFINITY);
        snprintf(msg, sizeof msg, "      in=%-5d group=%-2d overflowing scale%s gives +inf in %d/%d rows",
                 in, group, scenario == SC_INF_TAIL ? " in the tail" : "", inf_rows, rows);
        report(inf_rows == rows, msg);
    }
    if (scenario == SC_LOW && kind == MX_FLAT) {
        /* Teeth for the FTZ run, and the reason only it has any: without FTZ every
         * product here is exact in float too, so the double weights must agree. */
        int dw = 0;
        for (int r = 0; r < rows; r++)
            dw += !same_bits(ref_mx(kind, pk + (size_t)r * pcols, sc + (size_t)r * ngrp,
                                    x, in, group, M_RIGHT),
                             ref_mx(kind, pk + (size_t)r * pcols, sc + (size_t)r * ngrp,
                                    x, in, group, M_DOUBLEW));
        snprintf(msg, sizeof msg, "      in=%-5d group=%-2d weights formed in double differ in %d/%d rows%s",
                 in, group, dw, rows, g_ftz ? " (FTZ: at least half must)" : " (none may)");
        report(g_ftz ? dw * 2 >= rows : dw == 0, msg);
    }
    if (scenario == SC_NAN) {
        /* Every path skips a 255 group's whole chunks, so each row that has one is
         * finite, whatever its pair partner's scale; and the pattern must have produced
         * both kinds of mixed pair, or this proves nothing about them. */
        int nan_rows = 0, finite = 0, first = 0, second = 0;
        for (int r = 0; r < rows; r++) {
            const int has = sc[(size_t)r * ngrp + gnanx] == 255;
            nan_rows += has;
            finite += has && isfinite(y[r]);
            if ((r & 1) == 0 && r + 1 < rows) {
                const int has1 = sc[(size_t)(r + 1) * ngrp + gnanx] == 255;
                first += has && !has1;
                second += !has && has1;
            }
        }
        snprintf(msg, sizeof msg, "      in=%-5d group=%-2d scale 255 beside +inf x skipped in %d/%d rows "
                 "(mixed pairs %d+%d)", in, group, finite, nan_rows, first, second);
        report(finite == nan_rows && first > 0 && second > 0, msg);
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
    for (int t = 0; t < 8; t++)                   /* x NaN too in every other one */
        check_nan_rows(din[(size_t)t % (sizeof din / sizeof din[0])] + 40, 66, t & 1);

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
    /* The overflowing scale in the last group's in % 16 tail: at group 48 that group
     * also has two whole chunks, at 16, 32 and 64 it is the tail alone. */
    for (size_t g = 0; g < sizeof mgroup / sizeof mgroup[0]; g++) {
        check_mxfp4(1090, 65, mgroup[g], SC_INF_TAIL);      /* 1090 % 16 == 2 */
        check_mxfp4(1066, 66, mgroup[g], SC_INF_TAIL);      /* 1066 % 16 == 10 */
        check_mxfp4(1090, 64, mgroup[g], SC_LOW);
    }

#if defined(__AVX2__)
    /* The same order under flush-to-zero and denormals-are-zero. The engine never sets
     * them, but a host built with -ffast-math does (crtfastmath), and the AVX-512 MXFP4
     * table is built so that it still agrees with AVX2 there: scale bytes 0 and 1, whose
     * float weights are subnormal, go through the float multiply AVX2 uses rather than
     * the exact double one, and flush the same way; SC_LOW is the data on which a
     * double weight would show (SC_HUGE's LOW group alone is mostly swamped). rows <= 64
     * keeps these calls on this thread, the only one whose MXCSR is changed. */
    {
        const unsigned csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8040u);                /* FTZ (bit 15) | DAZ (bit 6) */
        g_ftz = 1;
        printf("  -- FTZ and DAZ set --\n");
        for (size_t g = 0; g < 4; g++) {          /* the flat-path groups */
            check_mxfp4(3584, 64, mgroup[g], SC_HUGE);
            check_mxfp4(1090, 33, mgroup[g], SC_HUGE);
            check_mxfp4(3584, 64, mgroup[g], SC_LOW);
            check_mxfp4(1090, 33, mgroup[g], SC_LOW);
        }
        check_dense(7168, 64);
        g_ftz = 0;
        _mm_setcsr(csr);
        printf("  -- FTZ and DAZ cleared --\n");
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
