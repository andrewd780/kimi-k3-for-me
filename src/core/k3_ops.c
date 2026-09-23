/* k3_ops.c - the numeric core of the Kimi K3 engine.
 *
 * Every routine here is gated on a JSON fixture under tests/fixtures/ops/, generated
 * by tools/emit_fixtures.py from the pure-torch reference. Per-op fixtures exist
 * alongside the full-model oracle because the oracle is pass-or-fail: it proves the
 * stack is wrong without indicating which of ~40 kernels is responsible.
 *
 * FLOATING-POINT CONTRACT. This file requires -ffp-contract=off and no -ffast-math;
 * both build systems set them. The kernels are written so that three implementations
 * of the same operation agree exactly:
 *
 *   - the scalar C99 path, which is the reference,
 *   - the OpenMP path (guarded by _OPENMP), which parallelises only over independent
 *     output rows, so it introduces no reduction and changes no arithmetic,
 *   - the AVX2 path (guarded by __AVX2__), which reproduces the scalar code's
 *     four-accumulator partition and reduction tree exactly rather than choosing a
 *     more natural one,
 *   - the NEON path (guarded by __ARM_NEON on aarch64), which maps the same scalar
 *     accumulators onto 2-lane double vectors, element i in the same accumulator and
 *     the same reduction tree, so it is bound by the identical bit-exactness contract.
 *
 * That last point is the reason several loops here look hand-unrolled for no visible
 * gain: the unrolling fixes a summation ORDER that the vector path must match. Reduce
 * them to the obvious form and the paths diverge in the last bits, which shows up as
 * a fixture failure on one machine and a pass on another.
 *
 * Matmul accumulators are double; the KDA recurrence retains float sums. Hidden size is 7168 and expert rows are 2048
 * wide; a float32 accumulator loses precision the reference comparisons can see.
 *
 * Bit-identity covers every result that is not a NaN. When two NaNs meet in one
 * operation, which payload and sign survive is left open by C and IEEE 754 and is
 * decided by the compiler's instruction selection (vfmadd132/213/231, the operand order
 * of a commutative add): measured, the same fma(w, x, acc) tail keeps the weight's NaN
 * in the scalar build and the activation's in the AVX2 build, and the batched tile can
 * differ from the single-position kernel the same way. So a NaN is reproduced as a NaN,
 * not as a particular NaN. A NaN logit means the weights or the state are already
 * corrupt, and greedy selection (v[i] > v[best]) reads no payload bits. test_ops draws
 * finite weights for that reason.
 */
#include "k3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* --------------------------------------------------------- fatal errors ---- */
/* Several kernels here need a small temporary that cannot be hoisted into caller-owned
 * scratch without changing a published signature. They are hundreds of bytes to a few
 * kilobytes, on a path where the engine has already reserved tens of gigabytes, so a
 * failure means the process is finished either way.
 *
 * What matters is HOW it finishes. These kernels return void, so a failed allocation
 * could only be handled by returning early, which leaves the output buffer holding
 * whatever was in it before, and the caller consumes it as a result. The run then
 * completes and prints a plausible token computed from uninitialised memory. That is the
 * one failure mode this engine treats as unacceptable, so allocation failure aborts
 * loudly instead.
 *
 * This is deliberately NOT how streamed experts are handled: an expert that fails to
 * load is recoverable in principle, so it is counted in k3_expert_drops and the decision
 * is left to the caller. Memory exhaustion inside a kernel is not recoverable. */
static void k3_fatal_oom(const char *what, size_t bytes)
{
    fprintf(stderr,
            "k3: FATAL, could not allocate %zu bytes for %s.\n"
            "    Aborting rather than continuing with an uninitialised buffer, which\n"
            "    would produce plausible-looking but meaningless output.\n",
            bytes, what);
    abort();
}

/* The same rule, for a bound that is exceeded rather than an allocation that fails.
 * k3_mla_cached writes its output only after the capacity check, so returning early
 * would hand the caller back whatever the scratch buffer held from the previous layer,
 * and k3_decoder_layer_inc folds that straight into the residual. The run finishes and
 * prints a token derived from the wrong layer's activations. Abort instead. */
static void k3_fatal_bound(const char *what, long value, long limit)
{
    fprintf(stderr,
            "k3: FATAL, %s is %ld, which exceeds the limit of %ld.\n"
            "    Aborting rather than returning without writing the output buffer,\n"
            "    which would fold the previous layer's values into the residual and\n"
            "    produce plausible-looking but meaningless output.\n"
            "    Shorten the prompt, lower --gen, or drop --incremental.\n",
            what, value, limit);
    abort();
}

/* ------------------------------------------------------------- layer map ---- */
/* The released config lists full_attn_layers ONE-BASED, and
 * configuration_kimi_k3.py:152-156 tests (layer_idx + 1) in kda_layers. Getting this
 * off by one silently swaps KDA and MLA layers throughout the stack. */
int k3_is_mla(const K3Cfg *c, int layer)
{
    for (int i = 0; i < c->n_full_attn; i++)
        if (c->full_attn[i] == layer + 1) return 1;
    return 0;
}
int k3_is_kda(const K3Cfg *c, int layer)   { return !k3_is_mla(c, layer); }
int k3_is_dense(const K3Cfg *c, int layer) { return layer < c->first_dense; }

/* --------------------------------------------------------------- rmsnorm ---- */
void k3_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    /* double accumulator: 7168 squared terms in float32 loses real precision, and
     * every downstream comparison against the reference depends on this. */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = w[i] * x[i] * inv;
}

/* -------------------------------------------------------------- SiTU-GLU ---- */
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* See k3.h. Incremented whenever a streamed expert cannot be fetched. */
long k3_expert_drops = 0;

void k3_situ_glu(float *y, const float *x, int n, float b1, float b2)
{
    const float *gate = x;
    const float *up   = x + n;
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        /* The sigmoid takes the UNCAPPED gate. Feeding it the capped value instead
         * still yields a bounded, plausible function and is WRONG.
         * modeling_kimi_linear.py:79 */
        const float a = b1 * tanhf(g / b1) * sigmoidf_(g);
        const float u = b2 * tanhf(up[i] / b2);
        y[i] = a * u;
    }
}

/* Automatic-storage bound for the k3_kda_step temporary. K3 uses kda_head_dim 128; the
 * heap fallback keeps a larger configuration slower, never wrong. */
#define K3_KDA_STEP_DV 256

/* ------------------------------------------------------------- ShortConv ---- */
/* Causal depthwise conv, SiLU fused, exactly as ShortConvolution(activation='silu').
 * state[c*(k-1) + j] holds the previous inputs for channel c, oldest first.
 * Updated in place so a decode loop can carry it forward. */
void k3_shortconv(float *y, const float *x, const float *w, float *state,
                  int channels, int k, int T)
{
    const int hist = k - 1;
    /* hist can be 0 when k == 1, and malloc(0) is permitted to return NULL. Guard on
     * hist rather than on buf, or a legitimate k == 1 configuration silently skips the
     * whole convolution and leaves y untouched. */
    float *buf = hist ? (float *)malloc((size_t)hist * sizeof(float)) : NULL;
    if (hist && !buf) k3_fatal_oom("ShortConv history", (size_t)hist * sizeof(float));

    for (int c = 0; c < channels; c++) {
        if (hist) {   /* memcpy/memset with a NULL pointer is UB even at length 0 */
            if (state) memcpy(buf, state + (size_t)c * hist, (size_t)hist * sizeof(float));
            else       memset(buf, 0, (size_t)hist * sizeof(float));
        }

        for (int t = 0; t < T; t++) {
            const float cur = x[(size_t)t * channels + c];
            /* taps are ordered oldest..newest, matching conv1d over a left-padded
             * sequence: w[k-1] multiplies the CURRENT input. */
            float acc = w[(size_t)c * k + hist] * cur;
            for (int j = 0; j < hist; j++)
                acc += w[(size_t)c * k + j] * buf[j];

            for (int j = 0; j + 1 < hist; j++) buf[j] = buf[j + 1];
            if (hist > 0) buf[hist - 1] = cur;

            y[(size_t)t * channels + c] = acc * sigmoidf_(acc);   /* SiLU, fused */
        }
        if (state && hist) memcpy(state + (size_t)c * hist, buf, (size_t)hist * sizeof(float));
    }
    free(buf);
}

/* ------------------------------------------------------------ KDA decay ----- */
void k3_kda_decay(float *g, float *alpha, const float *z, const float *A_log,
                  const float *dt_bias, int H, int D, float lb)
{
    for (int h = 0; h < H; h++) {
        /* PER HEAD. The checkpoint stores head_dim floats but only the first H are
         * nonzero. Indexing this per channel is a silent, fatal error. */
        const float a = expf(A_log[h]);
        for (int d = 0; d < D; d++) {
            const int i = h * D + d;
            const float u  = a * (z[i] + dt_bias[i]);
            const float gi = lb * sigmoidf_(u);   /* in (lb, 0] */
            g[i] = gi;
            alpha[i] = expf(gi);                  /* in (e^lb, 1] */
        }
    }
}

/* -------------------------------------------------------- KDA recurrence ---- */
#if defined(K3_KDA_SIMD) && !defined(K3_KDA_FORCE_SCALAR) && defined(__AVX2__)
typedef __m256 KdaVec;
#define KDA_WIDTH 8
#define kda_load _mm256_loadu_ps
#define kda_store _mm256_storeu_ps
#define kda_splat _mm256_set1_ps
#define kda_add _mm256_add_ps
#define kda_sub _mm256_sub_ps
#define kda_mul _mm256_mul_ps
#elif defined(K3_KDA_SIMD) && !defined(K3_KDA_FORCE_SCALAR) && defined(__ARM_NEON) && defined(__aarch64__)
typedef float32x4_t KdaVec;
#define KDA_WIDTH 4
#define kda_load vld1q_f32
#define kda_store vst1q_f32
#define kda_splat vdupq_n_f32
#define kda_add vaddq_f32
#define kda_sub vsubq_f32
#define kda_mul vmulq_f32
#endif

#ifdef KDA_WIDTH
void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv)
{
    /* Lanes own independent value columns; key rows are still reduced in their
     * original ascending order. Separate multiply/add instructions retain the float
     * rounding points. Fuse decay + read and delta write + output, but keep state
     * access row-major: column tiles regressed on the ARM CI runner.
     * K3_KDA_FORCE_SCALAR selects the original optimised C control below. */
    float ubuf[K3_KDA_STEP_DV];
    float *u = dv <= K3_KDA_STEP_DV ? ubuf : (float *)malloc((size_t)dv * sizeof(float));
    if (!u) k3_fatal_oom("KDA recurrence temporary", (size_t)dv * sizeof(float));
    for (int j = 0; j < dv; j++) u[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const KdaVec av = kda_splat(alpha[i]), kv = kda_splat(k[i]);
        int j = 0;
        for (; j <= dv - KDA_WIDTH; j += KDA_WIDTH) {
            const KdaVec state = kda_mul(kda_load(row + j), av);
            kda_store(row + j, state);
            if (k[i] != 0.0f)
                kda_store(u + j, kda_add(kda_load(u + j), kda_mul(kv, state)));
        }
        for (; j < dv; j++) {
            row[j] *= alpha[i];
            if (k[i] != 0.0f) u[j] += k[i] * row[j];
        }
    }
    /* Prediction error is column-local and invariant over the rank-one write. */
    for (int j = 0; j < dv; j++) { u[j] = v[j] - u[j]; o[j] = 0.0f; }
    for (int i = 0; i < dk; i++) {
        if (k[i] == 0.0f && q[i] == 0.0f) continue;
        float *row = S + (size_t)i * dv;
        const KdaVec kb = kda_splat(k[i] * beta), qv = kda_splat(q[i]);
        int j = 0;
        for (; j <= dv - KDA_WIDTH; j += KDA_WIDTH) {
            KdaVec state = kda_load(row + j);
            if (k[i] != 0.0f) {
                state = kda_add(state, kda_mul(kb, kda_load(u + j)));
                kda_store(row + j, state);
            }
            if (q[i] != 0.0f)
                kda_store(o + j, kda_add(kda_load(o + j), kda_mul(qv, state)));
        }
        for (; j < dv; j++) {
            if (k[i] != 0.0f) row[j] += k[i] * beta * u[j];
            if (q[i] != 0.0f) o[j] += q[i] * row[j];
        }
    }
    if (u != ubuf) free(u);
}
#undef KDA_WIDTH
#undef kda_load
#undef kda_store
#undef kda_splat
#undef kda_add
#undef kda_sub
#undef kda_mul
#else
void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv)
{
    /* 1. channel-wise decay: scale ROW i of S by alpha[i]. The gate is per key
     *    channel, not a scalar, which is what "channel-wise forget gate" means. */
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float a = alpha[i];
        for (int j = 0; j < dv; j++) row[j] *= a;
    }

    /* 2. read the state along k:  u = S^T k */
    /* Allocated AFTER the decay above has already modified S. Returning early here
     * would leave the recurrent state permanently scaled but never updated -- silent,
     * unrecoverable corruption of every subsequent token.
     *
     * AUTOMATIC STORAGE at the sizes that occur. This is the innermost call in the
     * engine: once per head per token per KDA layer, which at K3 scale is 96 x 69 =
     * 6,624 calls per token, and k3_kda_layer runs them from an OpenMP loop, so a heap
     * temporary here is 6,624 malloc/free pairs per token with sixteen threads
     * contending for the allocator. dv is 128 for K3, so the array below covers it and
     * the heap path is dead code in practice. */
    float  ubuf[K3_KDA_STEP_DV];
    float *uheap = NULL;
    float *u = ubuf;
    if (dv > K3_KDA_STEP_DV) {
        uheap = (float *)calloc((size_t)dv, sizeof(float));
        if (!uheap) k3_fatal_oom("KDA recurrence temporary", (size_t)dv * sizeof(float));
        u = uheap;
    } else {
        for (int j = 0; j < dv; j++) u[j] = 0.0f;   /* calloc's zeroing, explicitly */
    }
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) u[j] += ki * row[j];
    }

    /* 3. rank-one delta write. (v - u) is the prediction error: this is what makes
     *    it a DELTA rule rather than plain accumulation. */
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) row[j] += ki * beta * (v[j] - u[j]);
    }

    /* 4. output from the ALREADY UPDATED state: o = S^T q */
    for (int j = 0; j < dv; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float qi = q[i];
        if (qi == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) o[j] += qi * row[j];
    }
    free(uheap);                                  /* free(NULL) is a no-op */
}
#endif

/* ---------------------------------------------------------------- matmul ---- */
/* The dominant cost in the engine: essentially all compute time is spent here or in
 * k3_matmul_mxfp4. Two properties of this kernel are load bearing.
 *
 *   1. OUTPUT ROWS ARE INDEPENDENT. Parallelising the outer loop introduces no
 *      reduction and no race, and changes no arithmetic at all, each row is summed
 *      by exactly one thread in exactly the order below. Results are therefore
 *      identical at any thread count, which the fixtures rely on.
 *
 *   2. SIXTEEN ACCUMULATORS, PARTITIONED BY i%16, REDUCED BY ONE FIXED TREE (see
 *      k3_tree16). The split keeps the FMA pipeline full, a single accumulator
 *      serialises on the latency chain, but it is written out explicitly rather than
 *      left to the compiler because it fixes a summation ORDER. k3_matmul_bf16 and
 *      every vector path (AVX2, AVX-512, NEON) reproduce this exact partition and this
 *      exact tree, which is what makes the implementations agree bit for bit.
 *
 * Floating-point addition is not associative, so the sixteen-way split is a real change
 * to the arithmetic relative to a sequential sum. Keeping the accumulators in double
 * bounds the difference far below fp32 output precision; making them float would not.
 */

/* THE PARTITION, WRITTEN DOWN ONCE. For every element i below n16 = in & ~15, the
 * product row[i] * x[i] is fused (one fma() in double, one rounding) into accumulator
 * a[i % 16], each accumulator in ascending i. The sixteen are then reduced by the tree
 * below, and the last in % 16 elements are fused sequentially into the result. Every
 * implementation of k3_matmul and k3_matmul_bf16 -- scalar, AVX2, AVX-512, NEON, one row
 * at a time or two, x widened in place or read from a hoisted copy -- computes exactly
 * this, which tests/unit/test_matmul_exact.c checks against a plain re-implementation
 * on data built so that any other order changes the result. */
static inline double k3_tree16(const double *a)
{
    const double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
    const double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
    const double b2 = (a[2] + a[6]) + (a[10] + a[14]);
    const double b3 = (a[3] + a[7]) + (a[11] + a[15]);
    return (b0 + b1) + (b2 + b3);
}

/* EVERY NaN OUTPUT IS THE SAME NaN. The partition and tree fix every other output's
 * bits, but not a NaN's sign and payload: IEEE 754 leaves unspecified which input NaN
 * an operation passes on, x86 takes the one in the first source operand of whichever
 * instruction form the compiler picked, and compilers treat the operands of a*b, a+b
 * and fma's product as interchangeable. k3_matmul_bf16 sums the two rows of a pair
 * with separate instruction sequences, so the same row can leave with different NaN
 * bits as the first of a pair or the second (observed on the AVX2 and scalar builds),
 * and the row pipeline in k3_trunk.c splits a matrix into calls whose length follows
 * the memory budget, which moves rows between the two. Storing every NaN as the one
 * quiet NaN 0x7FC00000 makes a NaN output, like every other output, the same bits at
 * every budget and on every path. Nothing else changes: a value that is not NaN is
 * stored exactly as (float)acc. k3_matmul does the same so that the two kernels, which
 * share one order, also share one NaN. */
static inline float k3_out_f32(double acc)
{
    union { uint32_t u; float f; } v;
    v.f = (float)acc;
    if (v.f != v.f) v.u = 0x7FC00000u;
    return v.f;
}

/* The vector paths read x from a double copy made once per call (see "WIDEN x ONCE" in
 * k3_matmul_mxfp4); without a vector unit there is nothing to hoist. */
#if defined(__AVX2__) || (defined(__ARM_NEON) && defined(__aarch64__))
#define K3_MM_HOIST 1
#else
#define K3_MM_HOIST 0
#endif

/* The whole-chunk part of one fp32 row, the reference form: portable C, x widened per
 * element. It is the scalar build's path and the vector builds' fallback when the
 * hoisted copy of x could not be allocated, so that failure costs speed and nothing
 * else: the partition and tree are the ones every vector path reproduces. */
static inline double k3_f32_row_c(const float *row, const float *x, int n16)
{
    double a[16] = {0};
    for (int i = 0; i < n16; i += 16)
        for (int l = 0; l < 16; l++)
            a[l] = fma((double)row[i + l], (double)x[i + l], a[l]);
    return k3_tree16(a);
}

#if K3_MM_HOIST
/* The same sum from the hoisted xd[i] == (double)x[i]. Float to double is exact, so the
 * operands of every fma are the ones k3_f32_row_c forms in place.
 *
 *   AVX-512  two __m512d: z0 lane l is a[l], z1 lane l is a[8 + l].
 *   AVX2     four __m256d: v0..v3 hold a[0..3], a[4..7], a[8..11], a[12..15]; the
 *            lanewise (v0+v1)+(v2+v3) is b0..b3 of k3_tree16, then (b0+b1)+(b2+b3).
 *   NEON     eight float64x2_t: wk holds {a[2k], a[2k+1]}; (w0+w2)+(w4+w6) is {b0,b1}
 *            and (w1+w3)+(w5+w7) is {b2,b3}.
 *
 * _mm512_fmadd_pd, _mm256_fmadd_pd and vfmaq_f64 are the IEEE fused multiply-add per
 * lane, the same single rounding as fma(). */
static inline double k3_f32_row_v(const float *row, const double *xd, int n16)
{
#if defined(__AVX512F__)
    __m512d z0 = _mm512_setzero_pd(), z1 = _mm512_setzero_pd();
    for (int i = 0; i < n16; i += 16) {
        z0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(row + i)),
                             _mm512_loadu_pd(xd + i), z0);
        z1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_loadu_ps(row + i + 8)),
                             _mm512_loadu_pd(xd + i + 8), z1);
    }
    double a[16];
    _mm512_storeu_pd(a, z0);
    _mm512_storeu_pd(a + 8, z1);
    return k3_tree16(a);
#elif defined(__AVX2__)
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
    for (int i = 0; i < n16; i += 16) {
        v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i)),
                             _mm256_loadu_pd(xd + i), v0);
        v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 4)),
                             _mm256_loadu_pd(xd + i + 4), v1);
        v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 8)),
                             _mm256_loadu_pd(xd + i + 8), v2);
        v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 12)),
                             _mm256_loadu_pd(xd + i + 12), v3);
    }
    double b[4];
    _mm256_storeu_pd(b, _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)));
    return (b[0] + b[1]) + (b[2] + b[3]);
#else
    float64x2_t w0 = vdupq_n_f64(0.0), w1 = vdupq_n_f64(0.0);
    float64x2_t w2 = vdupq_n_f64(0.0), w3 = vdupq_n_f64(0.0);
    float64x2_t w4 = vdupq_n_f64(0.0), w5 = vdupq_n_f64(0.0);
    float64x2_t w6 = vdupq_n_f64(0.0), w7 = vdupq_n_f64(0.0);
    for (int i = 0; i < n16; i += 16) {
        const float32x4_t f0 = vld1q_f32(row + i),     f1 = vld1q_f32(row + i + 4);
        const float32x4_t f2 = vld1q_f32(row + i + 8), f3 = vld1q_f32(row + i + 12);
        w0 = vfmaq_f64(w0, vcvt_f64_f32(vget_low_f32(f0)), vld1q_f64(xd + i));
        w1 = vfmaq_f64(w1, vcvt_high_f64_f32(f0),          vld1q_f64(xd + i + 2));
        w2 = vfmaq_f64(w2, vcvt_f64_f32(vget_low_f32(f1)), vld1q_f64(xd + i + 4));
        w3 = vfmaq_f64(w3, vcvt_high_f64_f32(f1),          vld1q_f64(xd + i + 6));
        w4 = vfmaq_f64(w4, vcvt_f64_f32(vget_low_f32(f2)), vld1q_f64(xd + i + 8));
        w5 = vfmaq_f64(w5, vcvt_high_f64_f32(f2),          vld1q_f64(xd + i + 10));
        w6 = vfmaq_f64(w6, vcvt_f64_f32(vget_low_f32(f3)), vld1q_f64(xd + i + 12));
        w7 = vfmaq_f64(w7, vcvt_high_f64_f32(f3),          vld1q_f64(xd + i + 14));
    }
    const float64x2_t t0 = vaddq_f64(vaddq_f64(w0, w2), vaddq_f64(w4, w6));
    const float64x2_t t1 = vaddq_f64(vaddq_f64(w1, w3), vaddq_f64(w5, w7));
    return vaddvq_f64(t0) + vaddvq_f64(t1);
#endif
}
#endif /* K3_MM_HOIST */

void k3_matmul(float *y, const float *x, const float *W, int in, int out)
{
    const int n16 = in & ~15;

    /* x widened once per call rather than once per row; see k3_matmul_mxfp4. A failed
     * allocation selects k3_f32_row_c, the same sum without the copy. */
#if K3_MM_HOIST
    double *const xd = (double *)malloc((size_t)in * sizeof(double));
    if (xd) for (int i = 0; i < in; i++) xd[i] = (double)x[i];
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
#if K3_MM_HOIST
        double acc = xd ? k3_f32_row_v(row, xd, n16) : k3_f32_row_c(row, x, n16);
#else
        double acc = k3_f32_row_c(row, x, n16);
#endif
        for (int i = n16; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
        y[o] = k3_out_f32(acc);
    }

#if K3_MM_HOIST
    free(xd);                                     /* free(NULL) is a no-op */
#endif
}

/* ------------------------------------------------------- batched matmul ---- */
/* Y[t][o] = W[o] . X[t] for T positions in one call, every output BIT-IDENTICAL to
 * k3_matmul or k3_matmul_bf16 applied to that position alone.
 *
 * WHY THIS EXISTS
 *   Prefill, speculative verification and draft prefill apply every trunk matrix to T
 *   positions. Called once per position, the matrix streams through the core T times and
 *   the single-position kernel's real bottleneck, widening bf16 -> f32 -> f64, is redone
 *   T times for the same weights; under the row pipeline each call also rereads the matrix
 *   from disk. Here a row's weights are widened ONCE per register block of positions and
 *   every position in the block consumes the same widened register, and a streamed matrix
 *   is applied through K3WeightStream.apply_batch so it is read once per batch.
 *
 * WHY IT IS EXACT
 *   A block of positions is several independent copies of the single-position
 *   computation that happen to share the register holding the widened weight. For every
 *   (position, output) pair nothing else differs:
 *     - same operands. The weight goes through the same widening (bf16 -> f32 is a 16-bit
 *       shift, f32 -> f64 is exact) and x through the same f32 -> f64 conversion. A
 *       widened value is a value, whichever position it is shared with.
 *     - same partition. Element i of the row lands in accumulator i % 16 of ITS position:
 *       four __m256d per position on AVX2 (lane l of vector j is accumulator 4j+l), eight
 *       float64x2_t per position on NEON (vector k holds accumulators 2k and 2k+1), a
 *       [16] array per position in scalar C. Positions never share an accumulator.
 *     - same order within an accumulator: i ascending, one fma per element, the product
 *       first and the running sum last, exactly as the single-position kernels issue it.
 *     - same reduction tree: (a[l] + a[4+l]) + (a[8+l] + a[12+l]) for each l, then
 *       (b0 + b1) + (b2 + b3), per position.
 *     - same tail: the elements past the last full 16 are fma'd into the reduced sum in
 *       ascending order, and the final (float) rounding is the same.
 *     - same store: k3_out_f32, so a NaN output is the one quiet NaN here too. The tiles
 *       are other instruction sequences than the single-position kernels and may pass
 *       on a different input NaN; without it a NaN's sign and payload would depend on
 *       whether its position was prefilled in a batch or decoded alone.
 *   OpenMP splits OUTPUT ROWS across threads, as the single-position kernels do, so each
 *   output is still summed by one thread in the order above, at any thread count.
 *   test_ops compares every output bitwise against the per-position kernels over many
 *   shapes, tails and position counts, in whatever ISA the binary was built for.
 *
 * WHAT IS NOT DONE, AND WHY
 *   AVX-512 is not used. Measured on an AVX-512 Xeon, a 512-bit form of this loop was no
 *   faster than the 256-bit one: with the weight widened once per block, what remains is
 *   the per-position f32 -> f64 conversion of x, and the wider form does not remove it.
 *   Widening X to double once per call instead turns the loop into an L2 stream of
 *   doubles, and that measured slower still, including with the tile packed for L1.
 *   So an AVX-512 build runs the 256-bit tiles below beside the single-position
 *   kernels' 512-bit rows. That is not a second arithmetic: the tiles and every
 *   k3_matmul / k3_matmul_bf16 path hold the same sixteen accumulators, each fed its
 *   own elements in ascending order and reduced by the one tree (k3_tree16); the lane
 *   layouts differ only in which register lane holds which accumulator.
 *   test_matmul_exact holds the single-position kernels to that partition and tree,
 *   and test_ops holds these tiles to the single-position kernels, on every ISA.
 *
 * POSITIONS PER REGISTER BLOCK (K3_MM_TB)
 *   Each position in a block holds four __m256d accumulators on AVX2, so the block size
 *   is set by the vector register file. With 16 ymm registers (plain AVX2) a block of 4
 *   fills them, and 8 spills: bench_batch at 12288 x 7168, one thread, -march=haswell on
 *   the reference VM, measured blocks of 8 about 8% SLOWER than 4 at T = 8 and 16 (and
 *   2% to 9% slower with -mavx2 -mfma at T = 8, 9 and 16, one thread and four). With
 *   AVX-512VL the compiler may use ymm16-31 for the same 256-bit code, and there a block
 *   of 8 measured 7% to 13% faster than 4 at T = 8, 9 and 16 on one thread and at T = 8
 *   and 9 on four, tying at 16 on four (runs in docs/notes/research-results.md).
 *   So 8 when __AVX512VL__ is defined, else 4; NEON keeps 2, since each position needs
 *   eight of its 32 registers for accumulators. The block size is a loop shape only:
 *   each position keeps its own accumulators whatever the block, so no output can
 *   depend on it, and test_ops checks every block size and remainder bitwise. It may be
 *   forced with -DK3_MM_TB=1, 2, 4 or 8 to compare them. */

/* Force inlining so that each call below, made with a literal block size, gets its own
 * copy with the position loops unrolled and the accumulators held in registers. */
#if defined(__GNUC__)
#define K3_ALWAYS_INLINE static inline __attribute__((always_inline))
#else
#define K3_ALWAYS_INLINE static inline
#endif

#ifndef K3_MM_TB
#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(__AVX2__)
#define K3_MM_TB 2
#elif defined(__AVX2__) && defined(__AVX512VL__)
#define K3_MM_TB 8
#else
#define K3_MM_TB 4
#endif
#endif
#if K3_MM_TB != 1 && K3_MM_TB != 2 && K3_MM_TB != 4 && K3_MM_TB != 8
#error "K3_MM_TB must be 1, 2, 4 or 8: the remainder dispatch below covers those"
#endif

/* Positions per pass over the matrix. Every row reads the X rows of all positions in the
 * pass, so the pass is sized to keep them (about 512 KiB of floats) cache resident while
 * the weights stream past; the matrix is then read once per pass rather than once per
 * position. This chooses a loop order only. It cannot change an output. */
static int k3_mm_pass(int in)
{
    long g = (512L * 1024L) / ((long)(in > 0 ? in : 1) * (long)sizeof(float));
    g -= g % K3_MM_TB;
    if (g < K3_MM_TB) g = K3_MM_TB;
    if (g > 4096)     g = 4096;
    return (int)g;
}

/* One output row for nb positions: y[t*ldy] for t < nb. fp32 weights, k3_matmul's
 * arithmetic. */
K3_ALWAYS_INLINE void k3_mm_f32_tile(float *y, int ldy, const float *X, int ldx,
                                     const float *row, int in, const int nb)
{
    double a[K3_MM_TB][16];
    double acc[K3_MM_TB];
    for (int t = 0; t < nb; t++)
        for (int l = 0; l < 16; l++) a[t][l] = 0.0;
    int i = 0;
    for (; i + 15 < in; i += 16)
        for (int t = 0; t < nb; t++) {
            const float *xt = X + (size_t)t * ldx + i;
            for (int l = 0; l < 16; l++)
                a[t][l] = fma((double)row[i + l], (double)xt[l], a[t][l]);
        }
    for (int t = 0; t < nb; t++) {
        const double b0 = (a[t][0] + a[t][4]) + (a[t][8]  + a[t][12]);
        const double b1 = (a[t][1] + a[t][5]) + (a[t][9]  + a[t][13]);
        const double b2 = (a[t][2] + a[t][6]) + (a[t][10] + a[t][14]);
        const double b3 = (a[t][3] + a[t][7]) + (a[t][11] + a[t][15]);
        acc[t] = (b0 + b1) + (b2 + b3);
    }
    for (; i < in; i++) {
        const double wi = (double)row[i];
        for (int t = 0; t < nb; t++) acc[t] = fma(wi, (double)X[(size_t)t * ldx + i], acc[t]);
    }
    for (int t = 0; t < nb; t++) y[(size_t)t * ldy] = k3_out_f32(acc[t]);
}

/* One output row for nb positions, bf16 weights, k3_matmul_bf16's arithmetic: its
 * partition, fma order, tree and tail, in a lane layout of the tile's own (AVX2 and
 * scalar forms below; NEON the same layout as k3_matmul_bf16's NEON rows). */
K3_ALWAYS_INLINE void k3_mm_bf16_tile(float *y, int ldy, const float *X, int ldx,
                                      const uint16_t *row, int in, const int nb)
{
    double acc[K3_MM_TB];
    int i = 0;
#if defined(__AVX2__)
    {
        __m256d v[K3_MM_TB][4];
        for (int t = 0; t < nb; t++)
            for (int j = 0; j < 4; j++) v[t][j] = _mm256_setzero_pd();
        for (; i + 15 < in; i += 16) {
            /* Sixteen weights, widened once: bf16 -> f32 is the 16-bit shift and
             * f32 -> f64 exact, so each is the value k3_matmul_bf16 multiplies (its x86
             * rows split even and odd elements with a shift and a mask instead, which
             * yields the same floats). wv[j] holds elements i+4j .. i+4j+3, i.e.
             * accumulators 4j .. 4j+3, in natural order. */
            __m256d wv[4];
            for (int j = 0; j < 4; j++) {
                const __m128i h = _mm_loadl_epi64((const __m128i *)(row + i + 4 * j));
                wv[j] = _mm256_cvtps_pd(
                    _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h), 16)));
            }
            for (int t = 0; t < nb; t++) {
                const float *xt = X + (size_t)t * ldx + i;
                for (int j = 0; j < 4; j++)
                    v[t][j] = _mm256_fmadd_pd(wv[j], _mm256_cvtps_pd(_mm_loadu_ps(xt + 4 * j)),
                                              v[t][j]);
            }
        }
        /* (v0+v1)+(v2+v3) lanewise is b0..b3 of k3_tree16, then (b0+b1)+(b2+b3): the one
         * tree, per position */
        for (int t = 0; t < nb; t++) {
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v[t][0], v[t][1]),
                                             _mm256_add_pd(v[t][2], v[t][3]));
            double a[4];
            _mm256_storeu_pd(a, vt);
            acc[t] = (a[0] + a[1]) + (a[2] + a[3]);
        }
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    {
        /* v[t][k] holds accumulators {2k, 2k+1} of position t, as w0..w7 do in
         * k3_matmul_bf16's NEON path: quad q of the chunk feeds v[t][2q] with its low
         * half and v[t][2q+1] with its high half, vfmaq_f64(acc, weight, x). */
        float64x2_t v[K3_MM_TB][8];
        for (int t = 0; t < nb; t++)
            for (int k = 0; k < 8; k++) v[t][k] = vdupq_n_f64(0.0);
        for (; i + 15 < in; i += 16) {
            const uint16x8_t h0 = vld1q_u16(row + i);
            const uint16x8_t h1 = vld1q_u16(row + i + 8);
            const float32x4_t f[4] = {
                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)),
                vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16)),
                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)),
                vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16))
            };
            float64x2_t wv[8];
            for (int q = 0; q < 4; q++) {
                wv[2 * q]     = vcvt_f64_f32(vget_low_f32(f[q]));
                wv[2 * q + 1] = vcvt_high_f64_f32(f[q]);
            }
            for (int t = 0; t < nb; t++) {
                const float *xt = X + (size_t)t * ldx + i;
                for (int q = 0; q < 4; q++) {
                    const float32x4_t xq = vld1q_f32(xt + 4 * q);
                    v[t][2 * q]     = vfmaq_f64(v[t][2 * q], wv[2 * q],
                                                vcvt_f64_f32(vget_low_f32(xq)));
                    v[t][2 * q + 1] = vfmaq_f64(v[t][2 * q + 1], wv[2 * q + 1],
                                                vcvt_high_f64_f32(xq));
                }
            }
        }
        /* {b0,b1} = (v0+v2)+(v4+v6), {b2,b3} = (v1+v3)+(v5+v7), then (b0+b1)+(b2+b3):
         * the scalar tree, as k3_matmul_bf16's NEON path reduces it. */
        for (int t = 0; t < nb; t++) {
            const float64x2_t s0 = vaddq_f64(vaddq_f64(v[t][0], v[t][2]),
                                             vaddq_f64(v[t][4], v[t][6]));
            const float64x2_t s1 = vaddq_f64(vaddq_f64(v[t][1], v[t][3]),
                                             vaddq_f64(v[t][5], v[t][7]));
            acc[t] = vaddvq_f64(s0) + vaddvq_f64(s1);
        }
    }
#else
    {
        double a[K3_MM_TB][16];
        for (int t = 0; t < nb; t++)
            for (int l = 0; l < 16; l++) a[t][l] = 0.0;
        for (; i + 15 < in; i += 16)
            for (int l = 0; l < 16; l++) {
                const double wl = (double)k3_bf16f(row[i + l]);
                for (int t = 0; t < nb; t++)
                    a[t][l] = fma(wl, (double)X[(size_t)t * ldx + i + l], a[t][l]);
            }
        for (int t = 0; t < nb; t++) {
            const double b0 = (a[t][0] + a[t][4]) + (a[t][8]  + a[t][12]);
            const double b1 = (a[t][1] + a[t][5]) + (a[t][9]  + a[t][13]);
            const double b2 = (a[t][2] + a[t][6]) + (a[t][10] + a[t][14]);
            const double b3 = (a[t][3] + a[t][7]) + (a[t][11] + a[t][15]);
            acc[t] = (b0 + b1) + (b2 + b3);
        }
    }
#endif
    for (; i < in; i++) {
        const double wi = (double)k3_bf16f(row[i]);
        for (int t = 0; t < nb; t++) acc[t] = fma(wi, (double)X[(size_t)t * ldx + i], acc[t]);
    }
    for (int t = 0; t < nb; t++) y[(size_t)t * ldy] = k3_out_f32(acc[t]);
}

/* One pass: every row, positions in register blocks of K3_MM_TB, the remainder in at
 * most two smaller blocks (a block of 4 first when K3_MM_TB is 8). Each literal block
 * size below is its own inlined copy of the tile. */
static void k3_mm_f32_pass(float *Y, int ldy, const float *X, int ldx,
                           const float *W, int in, int out, int T)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        int t = 0;
        for (; t + K3_MM_TB <= T; t += K3_MM_TB)
            k3_mm_f32_tile(Y + (size_t)t * ldy + o, ldy, X + (size_t)t * ldx, ldx,
                           row, in, K3_MM_TB);
#if K3_MM_TB > 4
        if (T - t >= 4) {
            k3_mm_f32_tile(Y + (size_t)t * ldy + o, ldy, X + (size_t)t * ldx, ldx,
                           row, in, 4);
            t += 4;
        }
#endif
        float *yt = Y + (size_t)t * ldy + o;
        const float *xt = X + (size_t)t * ldx;
        switch (T - t) {
#if K3_MM_TB > 3
        case 3: k3_mm_f32_tile(yt, ldy, xt, ldx, row, in, 3); break;
#endif
#if K3_MM_TB > 2
        case 2: k3_mm_f32_tile(yt, ldy, xt, ldx, row, in, 2); break;
#endif
        case 1: k3_mm_f32_tile(yt, ldy, xt, ldx, row, in, 1); break;
        default: break;
        }
    }
}

static void k3_mm_bf16_pass(float *Y, int ldy, const float *X, int ldx,
                            const uint16_t *W, int in, int out, int T)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int t = 0;
        for (; t + K3_MM_TB <= T; t += K3_MM_TB)
            k3_mm_bf16_tile(Y + (size_t)t * ldy + o, ldy, X + (size_t)t * ldx, ldx,
                            row, in, K3_MM_TB);
#if K3_MM_TB > 4
        if (T - t >= 4) {
            k3_mm_bf16_tile(Y + (size_t)t * ldy + o, ldy, X + (size_t)t * ldx, ldx,
                            row, in, 4);
            t += 4;
        }
#endif
        float *yt = Y + (size_t)t * ldy + o;
        const float *xt = X + (size_t)t * ldx;
        switch (T - t) {
#if K3_MM_TB > 3
        case 3: k3_mm_bf16_tile(yt, ldy, xt, ldx, row, in, 3); break;
#endif
#if K3_MM_TB > 2
        case 2: k3_mm_bf16_tile(yt, ldy, xt, ldx, row, in, 2); break;
#endif
        case 1: k3_mm_bf16_tile(yt, ldy, xt, ldx, row, in, 1); break;
        default: break;
        }
    }
}

void k3_matmul_batch_ld(float *Y, int ldy, const float *X, int ldx, const float *W,
                        int in, int out, int T)
{
    if (T <= 0 || out <= 0) return;
    const int g = k3_mm_pass(in);
    for (int t0 = 0; t0 < T; t0 += g) {
        const int n = T - t0 < g ? T - t0 : g;
        float *Yp = Y + (size_t)t0 * ldy;
        const float *Xp = X + (size_t)t0 * ldx;
        if (n == 1) k3_matmul(Yp, Xp, W, in, out);         /* the decode kernel itself */
        else        k3_mm_f32_pass(Yp, ldy, Xp, ldx, W, in, out, n);
    }
}

void k3_matmul_bf16_batch_ld(float *Y, int ldy, const float *X, int ldx,
                             const uint16_t *W, int in, int out, int T)
{
    if (T <= 0 || out <= 0) return;
    const int g = k3_mm_pass(in);
    for (int t0 = 0; t0 < T; t0 += g) {
        const int n = T - t0 < g ? T - t0 : g;
        float *Yp = Y + (size_t)t0 * ldy;
        const float *Xp = X + (size_t)t0 * ldx;
        if (n == 1) k3_matmul_bf16(Yp, Xp, W, in, out);    /* the decode kernel itself */
        else        k3_mm_bf16_pass(Yp, ldy, Xp, ldx, W, in, out, n);
    }
}

void k3_matmul_batch(float *Y, const float *X, const float *W, int in, int out, int T)
{
    k3_matmul_batch_ld(Y, out, X, in, W, in, out, T);
}

void k3_matmul_bf16_batch(float *Y, const float *X, const uint16_t *W, int in, int out,
                          int T)
{
    k3_matmul_bf16_batch_ld(Y, out, X, in, W, in, out, T);
}

/* ------------------------------------------------------------- Gated MLA ---- */
/* MLA with an optional KV cache, which is what makes incremental decode possible.
 *
 * WHY MLA IS THE ONLY PIECE THAT NEEDS ONE
 *   KDA already carries everything it needs: k3_kda_layer updates its recurrent state
 *   and its ShortConv history in place, so a decode step just declines to clear them.
 *   The attn-res block stack is per token with no cross-token dependency. MLA is the
 *   exception, because softmax attention must see every previous key and value, and
 *   this function recomputes them all from x on every call. That is what makes decode
 *   O(T^2).
 *
 * WHAT IS CACHED: TWO LAYOUTS, ONE ARITHMETIC
 *   By default the EXPANDED per-head keys and values are cached: 96 heads x 256 floats
 *   = 98,304 B per position per layer, and 2.37 MB per position across the 24 MLA
 *   layers. Nothing is recomputed, and a 64-token generation is 151 MB of KV cache.
 *
 *   kv_latent != 0 selects MLA's own design instead: kvc holds only the kv_lora_rank
 *   latent (the post-norm output of kv_a, 512 floats), and the per-head k and v are
 *   rebuilt through kv_b on every use. That is 2,304 B per position per layer with the
 *   rope slot, 0.055 MB per position across the 24 layers -- 42.8x smaller -- and it
 *   is what makes a long context a memory question rather than an impossible one.
 *
 *   The trade is compute, and it is not small: kv_b is 24576x512, so every cached
 *   position costs a 12.6M-MAC matmul, paid TWICE per query token (once to score, once
 *   to weight the values) because softmax needs every score before any value is used.
 *   Holding the rebuilt block across the two passes would mean holding the expanded
 *   cache again, which is the thing being avoided.
 *
 *   The latent is stored exactly as it was fed to kv_b in the expanded path, and the
 *   rebuild calls the SAME kernel with the SAME reduction order, so the two layouts are
 *   bitwise identical, not merely close. Nothing here may reorder the softmax: scores
 *   are still formed s ascending, the running max is still taken s ascending, and the
 *   value accumulation is still s ascending, per head. The loops are transposed (s
 *   outer, h inner) only so that ONE rebuild serves all 96 heads.
 *
 *   The rope slot is cached separately in both layouts because it is SHARED across
 *   heads: 64 values per position, not per head. Folding it into the per-head block
 *   would waste 96x the space and, worse, invites treating it as per-head somewhere.
 *   It is never rebuilt, because it is never projected through kv_b.
 *
 * kvc == NULL selects the self-contained path, which recomputes all keys and values
 * from x and caches nothing. All three paths must produce identical output; the op
 * fixtures gate the uncached path and tests/unit/k3_model.c gates them against each
 * other, logit for logit.
 *
 * THE TRACE HOOK (k3_mla_trace, see k3.h) copies out the raw scores, the double
 * normaliser z, the double quotient e/z and the pre-gate accumulator of every row, which
 * is how tests/unit/test_mla_variants.c holds the two layouts to each other on the
 * doubles the output rounds away. The quotient is named before it is rounded so that the
 * value recorded IS the value used: `pq = sc[s] / z; (float)pq` is `(float)(sc[s] / z)`,
 * since a double quotient is evaluated in double on every target this file builds for
 * (FLT_EVAL_METHOD 0 on x86-64 and aarch64). Nothing else here reads the trace.
 */
K3MlaTrace *k3_mla_trace = NULL;

void k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
                   int T, float *scratch,
                   float *kvc, float *ropec, int cached, int cap, int kv_latent)
{
    const int E  = c->hidden, H = c->n_heads;
    const int qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr;                       /* 192: the FULL head width      */
    const int kvw = c->kv_lora + qr;              /* 576: latent + shared rope slot */
    const int kvd = qn + vh;                      /* 256: cached width per head    */
    const float scale = 1.0f / sqrtf((float)qh);  /* :359, over qh not qn           */
    if (!kvc) cached = 0;
    /* The latent layout is a property of the cache, so it cannot exist without one. */
    const int lat = (kvc && kv_latent) ? 1 : 0;
    const int last = cached + T - 1;              /* highest absolute position      */
    if (kvc && last >= cap)
        k3_fatal_bound("MLA KV cache position", (long)last, (long)cap - 1);
    /* The test hook, read once per call: NULL outside tests. */
    K3MlaTrace *const tr = k3_mla_trace;
    if (tr && (tr->scores || tr->quot) && last + 1 > tr->n)
        k3_fatal_bound("MLA trace row length", (long)last + 1, (long)tr->n);

    /* Scratch layout. Every region below is DISJOINT and must stay so. Overlapping
     * any two of them can appear to work, aliasing the gate buffer onto q, say, is
     * safe only while H*vh < H*qh holds, but that is an accident of the released
     * dimensions, not an invariant, and it breaks silently the moment v_head grows.
     * Size the buffer with k3_mla_scratch_cached(); do not compute it by hand.
     *
     * ct, ql, acc and gbuf hold one row PER POSITION so that every projection is applied
     * to all T positions in one pass (k3_mmw_batch): the matrix is read and widened once
     * per call rather than once per position. Nothing about a position's arithmetic
     * changes; only where its intermediate rows live. */
    float *q    = scratch;                          /* [T][H][qh]     */
    float *ct   = q    + (size_t)T * H * qh;        /* [T][kvw] latent + rope slot */
    float *ql   = ct   + (size_t)T * kvw;           /* [T][q_lora]    */
    float *acc  = ql   + (size_t)T * c->q_lora;     /* [T][H][vh]     */
    float *gbuf = acc  + (size_t)T * H * vh;        /* [T][H][vh] gate */
    /* Scores are per head in the latent layout, because the s loop moves outside the h
     * loop there and every head's row must survive until its own softmax runs. */
    float *sc   = gbuf + (size_t)T * H * vh;        /* [last+1], latent [H][last+1] */
    const size_t scn = lat ? (size_t)H * (size_t)(last + 1) : (size_t)(last + 1);
    float *kb   = sc   + scn;                       /* latent: [H][kvd] one position */
    /* Without a cache the keys/values live in scratch and cover only this call. */
    float *kvs  = kb   + (lat ? (size_t)H * kvd : 0);       /* [T][H][kvd]    */
    float *rps  = kvs  + (kvc ? 0 : (size_t)T * H * kvd);   /* [T][qr] */

    #define K3_KV_AT(p)   (kvc   ? kvc   + (size_t)(p) * H * kvd : kvs + (size_t)(p) * H * kvd)
    #define K3_ROPE_AT(p) (ropec ? ropec + (size_t)(p) * qr      : rps + (size_t)(p) * qr)
    #define K3_LAT_AT(p)  (kvc + (size_t)(p) * c->kv_lora)

    /* ---- projections, each matrix applied to every position in one pass ---- */
    k3_mmw_batch(ql, x, w->q_a, w->wdt, E, c->q_lora, T);
    for (int t = 0; t < T; t++) {
        float *qlt = ql + (size_t)t * c->q_lora;
        k3_rmsnorm(qlt, qlt, w->q_a_norm, c->q_lora, c->rms_eps);
    }
    k3_mmw_batch(q, ql, w->q_b, w->wdt, c->q_lora, H * qh, T);

    /* ONE projection emits the compressed latent AND the shared rope slot */
    k3_mmw_batch(ct, x, w->kv_a, w->wdt, E, kvw, T);
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        float *ctt = ct + (size_t)t * kvw;
        /* the norm covers the latent only, never the rope slot */
        k3_rmsnorm(ctt, ctt, w->kv_a_norm, c->kv_lora, c->rms_eps);
        memcpy(K3_ROPE_AT(p), ctt + c->kv_lora, (size_t)qr * sizeof(float));
        /* The latent layout stores the kv_b INPUT and expands on use; the expanded
         * layout stores the kv_b OUTPUT, below. Same bytes into the same kernel either
         * way. */
        if (lat) memcpy(K3_LAT_AT(p), ctt, (size_t)c->kv_lora * sizeof(float));
    }
    /* Positions cached .. cached+T-1 are consecutive rows of the expanded cache (or of
     * kvs), so kv_b writes all of them in one pass, reading each position's normalised
     * latent in place at stride kvw. */
    if (!lat)
        k3_mmw_batch_ld(K3_KV_AT(cached), H * kvd, ct, kvw, w->kv_b, w->wdt,
                        c->kv_lora, H * kvd, T);

    /* ---- attention, per head, causal. Position t leaves its heads' outputs in its own
     * row of acc, so the gate and o_proj below can take every position in one pass. ---- */
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        float *acct = acc + (size_t)t * H * vh;
        if (lat) {
            /* Pass one: rebuild each cached position ONCE and score it against every
             * head. Transposing the loops is what keeps the rebuild count at one per
             * position rather than one per (position, head). */
            for (int s = 0; s <= p; s++) {
                k3_mmw(kb, K3_LAT_AT(s), w->kv_b, w->wdt, c->kv_lora, H * kvd);
                const float *kr = K3_ROPE_AT(s);
                for (int h = 0; h < H; h++) {
                    const float *qt = q + ((size_t)t * H + h) * qh;
                    const float *ks = kb + (size_t)h * kvd;
                    double d = 0.0;
                    for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                    for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                    sc[(size_t)h * (last + 1) + s] = (float)d * scale;
                }
            }
            /* Softmax per head, s ascending in both sweeps, exactly as above. The
             * probability is folded back into sc so the second rebuild pass needs no
             * per-head denominator: (float)(sc[s]/z) is the same value either way. */
            for (int h = 0; h < H; h++) {
                float *sh = sc + (size_t)h * (last + 1);
                const size_t row = (size_t)t * H + h;          /* trace row (t, h) */
                if (tr && tr->scores)
                    memcpy(tr->scores + row * tr->n, sh, (size_t)(p + 1) * sizeof(float));
                float m = -INFINITY;
                for (int s = 0; s <= p; s++) if (sh[s] > m) m = sh[s];
                double z = 0.0;
                for (int s = 0; s <= p; s++) { sh[s] = expf(sh[s] - m); z += sh[s]; }
                if (tr && tr->z) tr->z[row] = z;
                double *qrow = tr && tr->quot ? tr->quot + row * tr->n : NULL;
                for (int s = 0; s <= p; s++) {
                    const double pq = sh[s] / z;
                    if (qrow) qrow[s] = pq;
                    sh[s] = (float)pq;
                }
                float *o = acct + (size_t)h * vh;
                for (int j = 0; j < vh; j++) o[j] = 0.0f;
            }
            /* Pass two: rebuild again and accumulate the values. Each o[j] still
             * receives its terms s ascending, which is the order the sum must keep. */
            for (int s = 0; s <= p; s++) {
                k3_mmw(kb, K3_LAT_AT(s), w->kv_b, w->wdt, c->kv_lora, H * kvd);
                for (int h = 0; h < H; h++) {
                    const float pr = sc[(size_t)h * (last + 1) + s];
                    float *o = acct + (size_t)h * vh;
                    const float *vs = kb + (size_t)h * kvd + qn;
                    for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
                }
            }
        } else for (int h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) {                 /* causal: s <= p */
                const float *ks = K3_KV_AT(s) + (size_t)h * kvd;
                const float *kr = K3_ROPE_AT(s);           /* shared slot */
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                /* the rope slot is UNROTATED but still scored, and the SAME 64
                 * values serve every head. Dropping this term is the silent bug. */
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                sc[s] = (float)d * scale;
                if (sc[s] > m) m = sc[s];
            }
            const size_t row = (size_t)t * H + h;              /* trace row (t, h) */
            if (tr && tr->scores)
                memcpy(tr->scores + row * tr->n, sc, (size_t)(p + 1) * sizeof(float));
            double z = 0.0;
            for (int s = 0; s <= p; s++) { sc[s] = expf(sc[s] - m); z += sc[s]; }
            if (tr && tr->z) tr->z[row] = z;
            double *qrow = tr && tr->quot ? tr->quot + row * tr->n : NULL;

            float *o = acct + (size_t)h * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const double pq = sc[s] / z;
                if (qrow) qrow[s] = pq;
                const float pr = (float)pq;
                const float *vs = K3_KV_AT(s) + (size_t)h * kvd + qn;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }

        if (tr && tr->acc)
            memcpy(tr->acc + (size_t)t * H * vh, acct, (size_t)H * vh * sizeof(float));
    }

    /* ---- output gate then projection, every position in one pass each. Gate BEFORE
     * o_proj, and no norm on it, unlike KDA which norms first. :470-473 ---- */
    if (w->g) {
        k3_mmw_batch(gbuf, x, w->g, w->wdt, E, H * vh, T);
        for (size_t i = 0; i < (size_t)T * H * vh; i++)
            acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));
    }
    k3_mmw_batch(out, acc, w->o, w->wdt, H * vh, E, T);
    #undef K3_KV_AT
    #undef K3_ROPE_AT
    #undef K3_LAT_AT
}

void k3_mla(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
            int T, float *scratch)
{
    k3_mla_cached(out, x, w, c, T, scratch, NULL, NULL, 0, 0, 0);
}

/* ---------------------------------------------------------------- router ---- */
void k3_router(int *idx, float *w, const float *x, const float *W,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale)
{
    /* Returning early here would leave idx[] and w[] untouched, and k3_moe forms
     * `w->w1 + idx[j]*I*L` from them one line later -- an arbitrary pointer built from
     * uninitialised stack. */
    float *score  = (float *)malloc((size_t)n_experts * sizeof(float));
    float *choice = (float *)malloc((size_t)n_experts * sizeof(float));
    if (!score || !choice) k3_fatal_oom("router scores", (size_t)n_experts * sizeof(float) * 2);

    /* logits in float32 with no bias, then an independent sigmoid per expert. The
     * reference upcasts both operands explicitly; a double accumulator here matches
     * it and costs nothing at this width. */
    /* PARALLEL over experts, and bit-identical because of it.
     *
     * This is 896 dot products of length 7168 = 6.4M multiply-adds, per token, per MoE
     * layer, so 590M across the 92 of them -- and it ran on ONE core while every other
     * matmul in the engine was already threaded. It is pure arithmetic with no I/O to
     * hide behind, so it sat squarely on the critical path.
     *
     * Each iteration writes only its own score[e] and choice[e], and the ACCUMULATION
     * ORDER INSIDE an expert is untouched: thread t still sums i = 0..hidden-1 in
     * sequence into its own double. Splitting the outer loop therefore cannot change a
     * single bit, which is why this needs no tolerance and no re-gating.
     *
     * EIGHT EXPERTS PER PASS OVER x, for the same reason. One expert's sum is a single
     * chain of 7168 dependent double adds, so a core running one expert at a time waits
     * out the add latency on every element and leaves the rest of its pipeline idle.
     * Walking K3_ROUTER_BLOCK experts side by side gives the core that many independent
     * chains, each still receiving its terms i = 0..hidden-1 in order, one add per term,
     * exactly as before: the interleaving changes WHEN each add issues, never which adds
     * happen or in what order within an expert, so every score is bit-identical to the
     * one-expert loop. The product needs no care either way: a float times a float fits
     * in double's 53 bits, so it is exact whether or not the compiler fuses it. Timed
     * with bench_router at the released shape (896 x 7168) against the one-expert loop,
     * on a 4-vCPU AVX-512 guest with GCC 13.3, -march=native: 8.4 -> 4.6 ms per call on
     * one thread, and 2.1 -> 1.2 ms on four, the engine's threaded case, which is about
     * 0.09 s per token across the 92 MoE layers; the output hash was the same in all 40
     * runs. docs/notes/decode-kernels.md has every run. test_ops holds this form bitwise
     * to the one-expert loop on data where a reordered chain changes the scores. The
     * tail block (n_experts % K3_ROUTER_BLOCK experts) runs the same per-expert loop. */
    enum { K3_ROUTER_BLOCK = 8 };
    const int nblk = (n_experts + K3_ROUTER_BLOCK - 1) / K3_ROUTER_BLOCK;
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < nblk; b++) {
        const int e0 = b * K3_ROUTER_BLOCK;
        const int n  = (n_experts - e0) < K3_ROUTER_BLOCK ? (n_experts - e0)
                                                          : K3_ROUTER_BLOCK;
        double acc[K3_ROUTER_BLOCK] = {0};
        const float *row[K3_ROUTER_BLOCK];
        for (int k = 0; k < n; k++) row[k] = W + (size_t)(e0 + k) * hidden;
        if (n == K3_ROUTER_BLOCK) {
            for (int i = 0; i < hidden; i++) {
                const double xi = (double)x[i];
                for (int k = 0; k < K3_ROUTER_BLOCK; k++)
                    acc[k] += (double)row[k][i] * xi;
            }
        } else {
            for (int k = 0; k < n; k++)
                for (int i = 0; i < hidden; i++)
                    acc[k] += (double)row[k][i] * (double)x[i];
        }
        for (int k = 0; k < n; k++) {
            const int e = e0 + k;
            score[e]  = 1.0f / (1.0f + expf(-(float)acc[k]));
            choice[e] = score[e] + (bias ? bias[e] : 0.0f);   /* selection score only */
        }
    }

    /* top-k by repeated max. n_experts is 896 and topk is 16, so this is 14k
     * comparisons per token per layer: cheap next to an 18 MB expert read, and it
     * avoids a sort. Marking taken entries with -INFINITY keeps ties deterministic
     * in first-index order, matching a stable selection. */
    for (int j = 0; j < topk; j++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < n_experts; e++)
            if (choice[e] > bv) { bv = choice[e]; best = e; }
        if (best < 0) { idx[j] = 0; w[j] = 0.0f; continue; }
        idx[j] = best;
        w[j]   = score[best];              /* UNBIASED score, not choice[best] */
        choice[best] = -INFINITY;
    }

    if (renorm && topk > 1) {
        double s = 0.0;
        for (int j = 0; j < topk; j++) s += (double)w[j];
        const float inv = (float)(1.0 / (s + 1e-20));
        for (int j = 0; j < topk; j++) w[j] *= inv;
    }
    for (int j = 0; j < topk; j++) w[j] *= routed_scale;

    free(score); free(choice);
}

/* --------------------------------------------------------------- AttnRes ---- */
void k3_attn_res(float *out, const float *src, const float *fold,
                 int nsrc, int n, float eps)
{
    /* Returning early would leave `out` holding the previous layer's residual, which
     * the caller cannot distinguish from a computed one. */
    float *score = (float *)malloc((size_t)nsrc * sizeof(float));
    if (!score) k3_fatal_oom("AttnRes scores", (size_t)nsrc * sizeof(float));

    for (int s = 0; s < nsrc; s++) {
        const float *v = src + (size_t)s * n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
        /* key is the NORMALISED source; fold already carries norm.weight*proj.weight */
        double acc = 0.0;
        for (int i = 0; i < n; i++) acc += (double)(v[i] * inv) * (double)fold[i];
        score[s] = (float)acc;
    }

    float m = score[0];
    for (int s = 1; s < nsrc; s++) if (score[s] > m) m = score[s];
    double z = 0.0;
    for (int s = 0; s < nsrc; s++) { score[s] = expf(score[s] - m); z += score[s]; }

    for (int i = 0; i < n; i++) out[i] = 0.0f;
    for (int s = 0; s < nsrc; s++) {
        const float p = (float)(score[s] / z);
        const float *v = src + (size_t)s * n;   /* the RAW source, not the key */
        for (int i = 0; i < n; i++) out[i] += p * v[i];
    }
    free(score);
}

/* Exact scratch requirement for k3_mla. Callers should use this rather than
 * duplicating the arithmetic; getting it wrong overruns silently. */
/* Scratch for the self-contained path: keys and values live here, so they scale with T.
 * `cap` is the highest position that will be attended over plus one; without a cache
 * that is just T. */
size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode,
                             int kv_latent)
{
    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;
    const size_t kvd = (size_t)(c->qk_nope + vh);
    const int lat = (cached_mode && kv_latent) ? 1 : 0;
    /* The latent layout keeps one score row per head, and one rebuilt position. */
    size_t scores = (size_t)(cap > T ? cap : T);
    if (lat) scores *= (size_t)H;
    /* ct, ql, acc and gbuf are per position so each projection is one batched pass */
    size_t n = (size_t)T * H * qh                      /* q            */
             + (size_t)T * (c->kv_lora + c->qk_rope)   /* ct           */
             + (size_t)T * c->q_lora                   /* ql           */
             + (size_t)2 * T * H * vh                  /* acc, gbuf    */
             + scores;
    if (lat) n += (size_t)H * kvd;                     /* rebuilt k/v  */
    if (!cached_mode) n += (size_t)T * H * kvd + (size_t)T * c->qk_rope;
    return n;
}

size_t k3_mla_scratch(const K3Cfg *c, int T)
{
    return k3_mla_scratch_cached(c, T, T, 0, 0);
}

/* ------------------------------------------------------- Stable LatentMoE ---- */
/* Verified against modeling_kimi_linear.py:815-838. The ORDER is load bearing:
 *   1. route on the FULL hidden width, before any projection      :818
 *   2. down-project to the latent width                            :822
 *   3. run the selected experts IN LATENT SPACE and sum, weighted  :825
 *   4. RMSNorm the AGGREGATE, never per expert                     :831
 *   5. up-project back to hidden                                   :832
 *   6. add the shared expert computed on the ORIGINAL input,
 *      with NO routing weight and NO scaling                       :837
 *
 * Routed experts live at the latent width (latent -> moe_inter -> latent); the
 * shared expert is one wider MLP at full width with intermediate moe_inter*n_shared.
 *
 * Every scratch region below is DISJOINT and must stay so. Reusing the gate/up buffer
 * for the down-projection output is safe only while 2*moe_inter >= latent. That holds
 * for the released config and for the tiny one, but it is an accident of the
 * numbers rather than an invariant, and it is the same class of hazard documented at
 * the scratch layout in k3_mla_cached. Size with k3_moe_scratch().
 */
/* The MoE scratch, laid out in ONE place so k3_moe, k3_moe_prefill and
 * k3_moe_scratch cannot disagree. The [T] regions hold one row per position, so each
 * trunk matrix -- down, up and the shared expert's three -- is applied to every position
 * in one pass (k3_mmw_batch) instead of once per position; the three per-expert buffers
 * are reused for each routed expert in turn. */
typedef struct {
    float *z;      /* [T][L]     latent inputs, the down projection            */
    float *accL;   /* [T][L]     weighted expert aggregates                    */
    float *sgu;    /* [T][2*SI]  shared gate|up; SiTU overwrites the gate half */
    float *sdn;    /* [T][E]     shared down projection                        */
    float *gu;     /* [2*I]      gate|up, one routed expert                    */
    float *act;    /* [I]        after SiTU                                    */
    float *edn;    /* [L]        expert down projection                        */
} K3MoeScratch;

static K3MoeScratch moe_layout(float *scratch, const K3Cfg *c, int T)
{
    const size_t n = T > 0 ? (size_t)T : 1;
    const size_t L = (size_t)c->latent, I = (size_t)c->moe_inter;
    const size_t SI = I * (size_t)c->n_shared;
    K3MoeScratch s;
    s.z    = scratch;
    s.accL = s.z    + n * L;
    s.sgu  = s.accL + n * L;
    s.sdn  = s.sgu  + n * 2 * SI;
    s.gu   = s.sdn  + n * (size_t)c->hidden;
    s.act  = s.gu   + 2 * I;
    s.edn  = s.act  + I;
    return s;
}

/* 6. The shared expert on the ORIGINAL full-width input, for every position, added
 * UNWEIGHTED to out. sh1 and sh3 land interleaved per position as [gate | up]; SiTU
 * writes its result over the gate half, which k3_situ_glu permits because element i of
 * the output depends only on element i of each half; sh2 then reads that half at stride
 * 2*SI. Per position this is exactly the sequence of k3_mmw calls it replaces. */
static void moe_shared(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                       int T, const K3MoeScratch *s)
{
    const int E = c->hidden, SI = c->moe_inter * c->n_shared;
    k3_mmw_batch_ld(s->sgu,      2 * SI, x, E, w->sh1, w->wdt, E, SI, T);
    k3_mmw_batch_ld(s->sgu + SI, 2 * SI, x, E, w->sh3, w->wdt, E, SI, T);
    for (int t = 0; t < T; t++) {
        float *sg = s->sgu + (size_t)t * 2 * SI;
        k3_situ_glu(sg, sg, SI, c->situ_b1, c->situ_b2);
    }
    k3_mmw_batch_ld(s->sdn, E, s->sgu, 2 * SI, w->sh2, w->wdt, SI, E, T);
    for (size_t i = 0; i < (size_t)T * E; i++) out[i] += s->sdn[i];
}

void k3_moe(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
            int T, int *idx, float *wt, float *scratch)
{
    const int E = c->hidden, L = c->latent, I = c->moe_inter;
    const K3MoeScratch s = moe_layout(scratch, c, T);
    float *gu = s.gu, *act = s.act, *edn = s.edn;

    /* 2. down-project every position into the latent space, one pass over `down`. It
     * reads x, exactly as routing does, so hoisting it above step 1 changes nothing. */
    k3_mmw_batch(s.z, x, w->down, w->wdt, E, L, T);

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        const float *z  = s.z + (size_t)t * L;
        float *accL = s.accL + (size_t)t * L;

        /* 1. route on the FULL width x, never on the latent z */
        k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, c->topk,
                  c->moe_renorm, c->routed_scale);

        int nk = c->topk;
        /* Draft cache-only routing: keep only the top-k experts already resident, and
         * renormalise their weights so the mixture still sums as intended. This makes a
         * draft token read ZERO new expert bytes. It is an approximation, which is exactly
         * what a draft is; the exact model verifies every proposed token. */
        if (w->cache_only && w->src && w->src->resident) {
            int m = 0; float wsum = 0.0f;
            for (int j = 0; j < c->topk; j++) {
                if (w->src->resident(w->src, w->layer, idx[j], NULL)) {
                    idx[m] = idx[j]; wt[m] = wt[j]; wsum += wt[j]; m++;
                }
            }
            nk = m;
            if (wsum > 0.0f) for (int j = 0; j < nk; j++) wt[j] /= wsum;
        }

        /* 3. the selected experts, in latent space, weighted and summed */
        for (int i = 0; i < L; i++) accL[i] = 0.0f;
        /* Hand the WHOLE top-k to the source first, so its reads can overlap. Without
         * this the loop below misses, blocks on a 17.55 MB read, computes, misses
         * again: a queue depth of one against a drive that needs depth to reach its
         * rated bandwidth. getmany is optional and may be NULL, in which case nothing
         * changes and the loop reads them one at a time exactly as before. */
        if (!w->cache_only && w->src && w->src->getmany)
            w->src->getmany(w->src, w->layer, idx, nk);
        for (int j = 0; j < nk; j++) {
            if (w->src) {
                /* Streamed: the expert stays MXFP4 and the matmul reads nibbles. In
                 * cache-only mode every idx[j] is known resident, so resident() serves it
                 * with no disk read; otherwise get() may read it. */
                K3ExpertQ q;
                int miss = w->cache_only
                    ? !w->src->resident(w->src, w->layer, idx[j], &q)
                    : (w->src->get(w->src, w->layer, idx[j], &q) != 0);
                if (miss) {
                    /* A cache-only draft filtered to resident experts already, so a miss
                     * here is a benign race at worst; skip it, since the draft is
                     * approximate by construction and the exact model verifies. On the
                     * exact path a miss is the unacceptable silent-corruption case: count
                     * it in k3_expert_drops so the caller fails the run (see docs/API.md). */
                    if (w->cache_only) continue;
                    k3_expert_drops++;
                    fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                                    "this token is CORRUPT\n", w->layer, idx[j]);
                    continue;
                }
                k3_matmul_mxfp4(gu,     z, q.p1, q.s1, L, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4(gu + I, z, q.p3, q.s3, L, I, K3_MXFP4_GROUP);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, K3_MXFP4_GROUP);
            } else {
                const float *e1 = w->w1 + (size_t)idx[j] * I * L;   /* gate */
                const float *e3 = w->w3 + (size_t)idx[j] * I * L;   /* up   */
                const float *e2 = w->w2 + (size_t)idx[j] * L * I;   /* down */
                k3_matmul(gu,     z, e1, L, I);
                k3_matmul(gu + I, z, e3, L, I);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul(edn, act, e2, I, L);
            }
            const float wj = wt[j];
            for (int i = 0; i < L; i++) accL[i] += wj * edn[i];
        }

        /* 4. RMSNorm the AGGREGATE (not per expert) */
        if (c->latent_norm) k3_rmsnorm(accL, accL, w->latent_norm, L, c->rms_eps);
    }

    /* 5. up-project every position, one pass over `up`, then 6. the shared expert */
    k3_mmw_batch(out, s.accL, w->up, w->wdt, L, E, T);
    moe_shared(out, x, w, c, T, &s);
}

/* Floats of scratch k3_moe and k3_moe_prefill need for T positions: see moe_layout. */
size_t k3_moe_scratch(const K3Cfg *c, int T)
{
    const size_t n  = T > 0 ? (size_t)T : 1;
    const size_t SI = (size_t)c->moe_inter * c->n_shared;
    return n * ((size_t)2 * c->latent     /* z, accL             */
              + 2 * SI                    /* shared gate|up      */
              + (size_t)c->hidden)        /* shared down         */
         + (size_t)3 * c->moe_inter       /* gu (2*I) + act (I)  */
         + (size_t)c->latent;             /* edn                 */
}

/* Batched MoE for PREFILL over T tokens, streamed experts only.
 *
 * k3_moe walks the top-k for each token independently, so across a T-token chunk it
 * fetches an expert once per token that routes to it. Under near-uniform routing that is
 * mostly waste: measured on the released trace, a 32-token chunk touches only ~2.7x fewer
 * unique experts than 16*32 draws, so reading each unique expert ONCE and reusing it for
 * every token in the chunk cuts prefill expert bytes ~3-4x. Prefill is where that matters,
 * because decode feeds one token at a time and has nothing to batch.
 *
 * Exactness is preserved to the last bit. Per token the arithmetic is identical to
 * k3_moe: the routed latent contributions are accumulated in the ORIGINAL top-k order
 * (j = 0..k-1) from a per-(token, slot) buffer, then normalised, up-projected and given
 * the shared expert exactly as before. Only the ORDER in which experts are fetched from
 * disk changes, and that touches no floating-point result.
 *
 * Two widths, deliberately different. The trunk matrices -- down, up and the shared
 * expert's three -- are each applied to ALL T positions in one pass (k3_mmw_batch), so
 * under --trunk-rows a forward reads each of them once however long the prompt is; their
 * [T] scratch rows are already sized by k3_moe_scratch(c, T). Only the routed-expert
 * dedup runs in sub-chunks of MOE_DEDUP_CHUNK positions, because its contribution buffer
 * is [n][K][L] and would otherwise grow with the prompt. Neither width reaches a
 * floating-point result: k3_mmw_batch is per-position bit-identical to k3_mmw at any T,
 * and a token's routed sum reads only its own contribution rows.
 *
 * out/x are [T][E], idx/wt scratch are topk-wide (reused per token), scratch holds
 * k3_moe_scratch(c, T) floats. This path requires w->src (streamed); the resident path
 * stays on k3_moe, which is what the oracle gates exercise. */
#define MOE_DEDUP_CHUNK 64

/* Per-sub-chunk buffers of the routed dedup, allocated once per k3_moe_prefill call and
 * reused by every sub-chunk; each is sized for MOE_DEDUP_CHUNK positions at most. */
typedef struct {
    int   *ridx;     /* [n][K]    routing decisions; -1 marks a dropped expert */
    float *rwt;      /* [n][K]    routing weights                             */
    float *contrib;  /* [n][K][L] every routed expert's latent output         */
    int   *uniq;     /* [n*K]     the sub-chunk's unique experts, first-seen  */
    char  *seen;     /* [n_experts]                                           */
} K3MoeDedup;

static void moe_prefill_routed(const K3MoeScratch *s, const float *x, const K3MoeW *w,
                               const K3Cfg *c, int t0, int T, const K3MoeDedup *d);

void k3_moe_prefill(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                    int T, int *idx, float *wt, float *scratch)
{
    /* K3_NO_BATCH_PREFILL forces the per-token path, so one binary can produce both the
     * batched and the reference token streams for a bit-identity A/B. */
    static int no_batch = -1;
    if (no_batch < 0) no_batch = getenv("K3_NO_BATCH_PREFILL") ? 1 : 0;
    /* cache_only renormalises per token over the resident subset, which the per-token
     * path already does; the draft's prompt prefill is one-time, so defer rather than
     * duplicate the renorm in the batch. */
    if (!w->src || T <= 1 || no_batch || w->cache_only) {
        k3_moe(out, x, w, c, T, idx, wt, scratch);
        return;
    }
    const int E = c->hidden, Ll = c->latent, K = c->topk;
    const K3MoeScratch s = moe_layout(scratch, c, T);

    /* Fixed sub-chunks bound the contribution buffer (14.7 MB at 64 tokens) no matter
     * how long the prompt is; a 32k prefill would otherwise want 7.3 GB of it. Most of
     * the dedup is already captured at this width: the unique-expert count grows far
     * slower than the request count under near-uniform routing. */
    const int nmax = T < MOE_DEDUP_CHUNK ? T : MOE_DEDUP_CHUNK;
    K3MoeDedup d;
    d.ridx    = (int *)  malloc((size_t)nmax * K * sizeof(int));
    d.rwt     = (float *)malloc((size_t)nmax * K * sizeof(float));
    d.contrib = (float *)malloc((size_t)nmax * K * Ll * sizeof(float));
    d.uniq    = (int *)  malloc((size_t)nmax * K * sizeof(int));
    d.seen    = (char *) malloc((size_t)c->n_experts);
    if (!d.ridx || !d.rwt || !d.contrib || !d.uniq || !d.seen)
        k3_fatal_oom("MoE prefill batch", (size_t)nmax * K * Ll * sizeof(float));

    /* 2. down-project every position in one pass over `down`. Routing (step 1) reads x,
     * not z, so hoisting this above it changes nothing, as in k3_moe. */
    k3_mmw_batch(s.z, x, w->down, w->wdt, E, Ll, T);
    /* 1, 3, 4. route, run the unique experts and aggregate, one sub-chunk at a time; each
     * leaves its positions' normalised aggregates in s.accL. */
    for (int t0 = 0; t0 < T; t0 += MOE_DEDUP_CHUNK) {
        const int n = (T - t0) < MOE_DEDUP_CHUNK ? (T - t0) : MOE_DEDUP_CHUNK;
        moe_prefill_routed(&s, x, w, c, t0, n, &d);
    }
    /* 5, 6. up-project every position and add the shared expert: one pass per matrix. */
    k3_mmw_batch(out, s.accL, w->up, w->wdt, Ll, E, T);
    moe_shared(out, x, w, c, T, &s);

    free(d.ridx); free(d.rwt); free(d.contrib); free(d.uniq); free(d.seen);
}

/* The routed half of the batched MoE for positions [t0, t0 + T), T <= MOE_DEDUP_CHUNK:
 * route each position, fetch each unique expert once, and leave each position's
 * normalised aggregate in its s->accL row. Reads x and s->z rows t0.. only. */
static void moe_prefill_routed(const K3MoeScratch *s, const float *x, const K3MoeW *w,
                               const K3Cfg *c, int t0, int T, const K3MoeDedup *d)
{
    const int E = c->hidden, Ll = c->latent, I = c->moe_inter;
    const int K = c->topk;
    int   *ridx = d->ridx, *uniq = d->uniq;
    float *rwt = d->rwt, *contrib = d->contrib;
    const float *z = s->z + (size_t)t0 * Ll;

    /* 1. route every token on the FULL width x and collect the sub-chunk's unique
     * experts in first-seen order. */
    memset(d->seen, 0, (size_t)c->n_experts);
    int nu = 0;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)(t0 + t) * E;
        int   *it = ridx + (size_t)t * K;
        float *wtt = rwt + (size_t)t * K;
        k3_router(it, wtt, xt, w->gate, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);
        for (int j = 0; j < K; j++) {
            const int e = it[j];
            if (e >= 0 && e < c->n_experts && !d->seen[e]) { d->seen[e] = 1; uniq[nu++] = e; }
        }
    }

    /* 3. expert-major: fetch each unique expert ONCE, apply it to every (token, slot)
     * that selected it. gu/act/edn are reused per (expert, token). */
    float *gu = s->gu, *act = s->act, *edn = s->edn;
    if (w->src->getmany) w->src->getmany(w->src, w->layer, uniq, nu);
    for (int u = 0; u < nu; u++) {
        const int e = uniq[u];
        K3ExpertQ q;
        if (w->src->get(w->src, w->layer, e, &q) != 0) {
            k3_expert_drops++;
            fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                            "this chunk is CORRUPT\n", w->layer, e);
            /* k3_moe skips a dropped expert's term; mark its slots so the sum below
             * does the same instead of reading a contribution row never written. */
            for (int i = 0; i < T * K; i++) if (ridx[i] == e) ridx[i] = -1;
            continue;
        }
        for (int t = 0; t < T; t++) {
            const int   *it = ridx + (size_t)t * K;
            const float *zt = z + (size_t)t * Ll;
            for (int j = 0; j < K; j++) {
                if (it[j] != e) continue;
                k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
            }
        }
    }

    /* 4. per token, sum contributions in the ORIGINAL top-k order and normalise, exactly
     * as k3_moe does it, so every float matches the per-token path. Summing in the
     * fetch order of uniq[] instead is the natural slip here and changes the floats;
     * test_ops's moe_prefill gate catches it at top-16. The CLI's K3_NO_BATCH_PREFILL
     * comparison cannot: its tiny checkpoint routes to the top 2, where the order of
     * two terms added to zero never changes a float. */
    for (int t = 0; t < T; t++) {
        const int   *it  = ridx + (size_t)t * K;
        const float *wtt = rwt + (size_t)t * K;
        float *acc = s->accL + (size_t)(t0 + t) * Ll;
        for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            if (it[j] < 0) continue;
            const float wj = wtt[j];
            const float *cb = contrib + ((size_t)t * K + j) * Ll;
            for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
        }
        if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
    }
}

/* --------------------------------------------------------- KDA full layer ---- */
/* L2 normalisation over the last dimension. The reference uses the SUM of squares
 * with eps inside the rsqrt, NOT the mean: k3_ref.py l2norm(). Using the mean here
 * scales every q and k by sqrt(d_k) and quietly changes the attention temperature. */
static void l2norm_(float *v, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
    const float inv = (float)(1.0 / sqrt(ss + (double)eps));
    for (int i = 0; i < n; i++) v[i] *= inv;
}

size_t k3_kda_scratch(const K3Cfg *c, int T)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return 3 * (size_t)T * P        /* q, k, v after conv                      */
         + 2 * (size_t)T * P        /* z (then g, then the output gate), alpha */
         + (size_t)T * c->kda_heads /* beta                                    */
         + (size_t)T * P            /* recurrence output                       */
         + P                        /* one work row, a slice per head          */
         + (size_t)T * c->kda_head_dim; /* f_a output, per position            */
}

/* Floats in one row of a K3KdaLog: k and v as the recurrence reads them, alpha, beta, and
 * the three pre-conv inputs. See the layout in k3.h. */
size_t k3_kda_log_row(const K3Cfg *c)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return 6 * P + (size_t)c->kda_heads;
}

size_t k3_kda_state_floats(const K3Cfg *c)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return P * (size_t)c->kda_head_dim + 3 * P * (size_t)(c->conv_k - 1);
}

/* A log row or commit past the recorded rows would replay whatever the buffer held last
 * time: a fluent, wrong state. Refuse loudly, as the other kernels do. */
static void kda_log_bound_(const char *what, int n, int cap)
{
    fprintf(stderr, "k3: FATAL, %s is %d, but the KDA log holds %d rows.\n", what, n, cap);
    abort();
}

/* Apply log rows [0, n) to ONE head's block of S, in place. The heart of every commit and
 * of every replay into a work copy.
 *
 * WHY THE RESULT IS BIT-IDENTICAL, not merely close, to the sweep that recorded the rows:
 * this is k3_kda_step, the function the sweep called, on the same S bits, with the same
 * k, v, alpha and beta bits, in the same t order. Its update of S reads nothing else: q
 * enters only the output o, in both the scalar and the SIMD build (the SIMD path tests q
 * only to skip a row whose k is also zero, and a zero k never writes S). So passing zeros
 * for q, which also lets the step skip its output pass, cannot move a bit of S. Heads are
 * independent, so the thread count cannot either. */
static void kda_replay_head_(float *Sh, const float *rows, size_t lrow, int n, int h,
                             int D, int P, const float *qz, float *oh)
{
    for (int t = 0; t < n; t++) {
        const float *r = rows + (size_t)t * lrow;
        k3_kda_step(Sh, oh, qz, r + (size_t)h * D, r + P + (size_t)h * D,
                    r + 2 * (size_t)P + (size_t)h * D, r[3 * (size_t)P + h], D, D);
    }
}

/* The ShortConv history after n logged rows. k3_shortconv keeps, per channel, the last
 * conv_k-1 inputs, oldest first, out of [history it started with, x_0, x_1, ...]; after n
 * inputs that is entries n .. n+conv_k-2 of that sequence, and the log holds each of them
 * bit for bit. Pure data movement.
 *
 * dst may be src. Writing slot j reads slot n+j of the same channel, and for n >= 1 that
 * slot lies AHEAD of every slot written so far, so ascending j never reads a slot it has
 * already overwritten. src == NULL is the zero history of a fresh sequence. */
static void kda_replay_conv_(float *dst, const float *src, const float *rows, size_t lrow,
                             int n, const K3Cfg *c)
{
    const int P = c->kda_heads * c->kda_head_dim, H = c->kda_heads, hist = c->conv_k - 1;
    for (int which = 0; which < 3; which++) {
        const size_t blk = (size_t)which * P * hist;
        const size_t pre = 3 * (size_t)P + H + (size_t)which * P;   /* pre-conv slice */
        for (int ch = 0; ch < P; ch++) {
            for (int j = 0; j < hist; j++) {
                const int i = n + j;
                const size_t at = blk + (size_t)ch * hist;
                dst[at + j] = i < hist ? (src ? src[at + i] : 0.0f)
                                       : rows[(size_t)(i - hist) * lrow + pre + ch];
            }
        }
    }
}

void k3_kda_layer(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                  int T, float *state, float *scratch)
{
    k3_kda_layer_log(out, x, w, c, T, state, scratch, NULL);
}

void k3_kda_layer_log(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                      int T, float *state, float *scratch, const K3KdaLog *log)
{
    const int E = c->hidden, H = c->kda_heads, D = c->kda_head_dim;
    const int P = H * D, K = c->conv_k, hist = K - 1;

    float *q  = scratch;                 float *k  = q + (size_t)T * P;
    float *v  = k + (size_t)T * P;       float *z  = v + (size_t)T * P;
    float *al = z + (size_t)T * P;       float *bt = al + (size_t)T * P;
    float *o  = bt + (size_t)T * H;      float *wr = o + (size_t)T * P;
    float *fa = wr + P;                  /* [T][D] */

    /* 1. projections, each matrix applied to every position in one pass (k3_mmw_batch):
     * read and widened once per call, not once per position, and per position
     * bit-identical to k3_mmw. */
    k3_mmw_batch(q,  x, w->q, w->wdt, E, P, T);
    k3_mmw_batch(k,  x, w->k, w->wdt, E, P, T);
    k3_mmw_batch(v,  x, w->v, w->wdt, E, P, T);
    k3_mmw_batch(bt, x, w->b, w->wdt, E, H, T);
    /* ONE shared low-rank pair feeds every head: [E->D] then [D->H*D] */
    k3_mmw_batch(fa, x,  w->f_a, w->wdt, E, D, T);
    k3_mmw_batch(z,  fa, w->f_b, w->wdt, D, P, T);

    /* A logged call is tentative (see K3KdaLog): it computes on log->work and leaves
     * `state` as it found it. `st` is the state this call actually carries forward. The
     * work copy starts as `state` advanced by the rows earlier calls recorded, so the call
     * computes exactly what it would have computed had those calls updated `state` in
     * place. Everything the log adds is a copy either into the work state or out of
     * buffers the layer computes anyway, so the arithmetic below is unchanged and the
     * output is the same bits with or without a log. */
    const size_t lrow = log ? k3_kda_log_row(c) : 0;
    float *st = state;
    int nrec = 0;
    if (log) {
        if (log->row0 < 0 || log->row0 > log->cap) kda_log_bound_("KDA log row0", log->row0,
                                                                  log->cap);
        st = log->work;
        /* Positions past cap are computed but not recorded: the caller sizes the log for
         * the longest prefix it can keep. */
        nrec = log->cap - log->row0;
        if (nrec > T) nrec = T;
        kda_replay_conv_(st + (size_t)H * D * D,
                         state ? state + (size_t)H * D * D : NULL,
                         log->rows, lrow, log->row0, c);
        /* The pre-conv inputs, before k3_shortconv overwrites them in place. They are all
         * the ShortConv history is ever made of. */
        for (int t = 0; t < nrec; t++) {
            float *r = log->rows + (size_t)(log->row0 + t) * lrow + 3 * (size_t)P + H;
            memcpy(r,                 q + (size_t)t * P, (size_t)P * sizeof(float));
            memcpy(r + P,             k + (size_t)t * P, (size_t)P * sizeof(float));
            memcpy(r + 2 * (size_t)P, v + (size_t)t * P, (size_t)P * sizeof(float));
        }
    }

    /* 2. ShortConv with fused SiLU, carrying state across calls */
    float *cs = st ? st + (size_t)H * D * D : NULL;
    k3_shortconv(q, q, w->q_conv, cs ? cs : NULL, P, K, T);
    k3_shortconv(k, k, w->k_conv, cs ? cs + (size_t)P * hist : NULL, P, K, T);
    k3_shortconv(v, v, w->v_conv, cs ? cs + (size_t)2 * P * hist : NULL, P, K, T);

    /* 3. L2Norm on q and k ONLY, per head. v is deliberately left alone. */
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            l2norm_(q + (size_t)t * P + (size_t)h * D, D, 1e-6f);
            l2norm_(k + (size_t)t * P + (size_t)h * D, D, 1e-6f);
        }

    /* 4/5. beta and the decay chain */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) bt[(size_t)t * H + h] = sigmoidf_(bt[(size_t)t * H + h]);
        k3_kda_decay(z + (size_t)t * P, al + (size_t)t * P, z + (size_t)t * P,
                     w->A_log, w->dt_bias, H, D, c->gate_lb);
    }

    /* The recurrence operands, exactly as k3_kda_step is about to receive them below:
     * k normalised, v raw, alpha, and beta after its sigmoid. Nothing between here and
     * the step modifies them. */
    if (log) for (int t = 0; t < nrec; t++) {
        float *r = log->rows + (size_t)(log->row0 + t) * lrow;
        memcpy(r,                 k  + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(r + P,             v  + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(r + 2 * (size_t)P, al + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(r + 3 * (size_t)P, bt + (size_t)t * H, (size_t)H * sizeof(float));
    }

    /* 6. recurrence, per head, with q pre-scaled by d_k^-0.5 */
    float *S = st;
    float *Sown = NULL;
    if (!S) {
        /* Dereferenced at a computed offset immediately below; an unchecked NULL here
         * is a wild write, not a missing result. */
        Sown = (float *)calloc((size_t)H * D * D, sizeof(float));
        if (!Sown) k3_fatal_oom("KDA recurrent state", (size_t)H * D * D * sizeof(float));
        S = Sown;
    }
    /* Only a replay into the work copy needs these: a zero q and a throwaway output row
     * per head for k3_kda_step, see kda_replay_head_. */
    float *rtmp = NULL;
    if (log && log->row0 > 0) {
        rtmp = (float *)calloc((size_t)(H + 1) * D, sizeof(float));
        if (!rtmp) k3_fatal_oom("KDA log replay", (size_t)(H + 1) * D * sizeof(float));
    }
    const float qscale = 1.0f / sqrtf((float)D);
    /* Heads are independent: each reads and writes only its own S block, its own D-wide
     * slice of q/k/v/al/o, and its own beta column. The recurrence is sequential in t
     * WITHIN a head, so the loops nest head-outer here and each head walks its own t in
     * order; per-head arithmetic is untouched and the results are bit-identical to the
     * serial form (gated by test_ops' kda fixtures under 1 vs N threads). The recurrence
     * is 0.4% of FLOPs but, serial, it is a majority of non-matmul wall time at high
     * core counts. wr is a full P-wide work row, so wr + h*D gives each head a private
     * slice with no new allocation. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        float *wh = wr + (size_t)h * D;
        /* The work copy of this head's block, taken here rather than by one big copy
         * before the loop: the block is about to be streamed through the cache by the
         * recurrence anyway, and every thread copies its own heads. */
        if (log) {
            float *Sh = S + (size_t)h * D * D;
            if (state) memcpy(Sh, state + (size_t)h * D * D, (size_t)D * D * sizeof(float));
            else       memset(Sh, 0, (size_t)D * D * sizeof(float));
            if (log->row0 > 0)
                kda_replay_head_(Sh, log->rows, lrow, log->row0, h, D, P, rtmp,
                                 rtmp + (size_t)(h + 1) * D);
        }
        for (int t = 0; t < T; t++) {
            const size_t off = (size_t)t * P + (size_t)h * D;
            for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
            k3_kda_step(S + (size_t)h * D * D, o + off, wh, k + off, v + off,
                        al + off, bt[(size_t)t * H + h], D, D);
        }
    }
    free(rtmp);

    /* 7/8/9. head-wise RMSNorm, THEN the gate, THEN the output projection. Each stage
     * covers every position before the next begins, so g_proj and o_proj are each one
     * pass over their matrix. Positions share nothing here, so each position still sees
     * exactly norm, then gate, then projection, on the same values.
     *
     * The gate rows reuse z. After step 5 z holds only the log-decay g, which nothing
     * reads once alpha has been formed from it, and it is [T][P], exactly the gate's
     * shape, so the batched gate costs no scratch beyond what the layer already had. */
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            float *oh = o + (size_t)t * P + (size_t)h * D;
            k3_rmsnorm(oh, oh, w->o_norm, D, c->rms_eps);
        }
    float *gt = z;
    k3_mmw_batch(gt, x, w->g, w->wdt, E, P, T);
    for (size_t i = 0; i < (size_t)T * P; i++) o[i] *= sigmoidf_(gt[i]);
    k3_mmw_batch(out, o, w->o, w->wdt, P, E, T);
    free(Sown);
}

/* Commit n recorded positions to the carried state. See K3KdaLog in k3.h, and
 * kda_replay_head_ and kda_replay_conv_ for why the result is the state feeding exactly
 * those positions would have left, to the bit. Nothing is read from the weights, which is
 * why a streamed trunk costs no I/O here. */
void k3_kda_advance(float *state, const K3KdaLog *log, int n, const K3Cfg *c)
{
    const int H = c->kda_heads, D = c->kda_head_dim, P = H * D;
    if (n < 0 || n > log->cap) kda_log_bound_("KDA commit length", n, log->cap);
    if (n == 0) return;
    const size_t lrow = k3_kda_log_row(c);

    float *tmp = (float *)calloc((size_t)(H + 1) * D, sizeof(float));
    if (!tmp) k3_fatal_oom("KDA commit temporaries", (size_t)(H + 1) * D * sizeof(float));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++)
        kda_replay_head_(state + (size_t)h * D * D, log->rows, lrow, n, h, D, P, tmp,
                         tmp + (size_t)(h + 1) * D);
    free(tmp);
    kda_replay_conv_(state + (size_t)H * D * D, state + (size_t)H * D * D, log->rows, lrow,
                     n, c);
}

/* ----------------------------------------------------------- decoder layer ---- */
/* The region after the layer's own buffers, which attention, then the MLP, take turns
 * in: [2*dense_inter][sub-block]. Attention and the MoE use the sub-block; the dense MLP
 * (layer 0) uses the whole region, since nothing else is live by then. */
static size_t layer_region(const K3Cfg *c, int T)
{
    size_t a = k3_mla_scratch(c, T);
    size_t b = k3_kda_scratch(c, T);
    size_t m = k3_moe_scratch(c, T);
    size_t sub = a > b ? a : b;
    if (m > sub) sub = m;
    return (size_t)2 * c->dense_inter + sub;
}

/* Positions per block of the dense MLP. Each position needs one [gate | up] row of
 * 2*dense_inter floats (SiTU then writes over the gate half in place), and a block is as
 * many rows as the region holds, so the three dense matrices are applied to a whole
 * block per pass without reserving anything beyond what attention already needs. For the
 * released config and the fixture config alike that covers every position: the KDA
 * sub-block alone is larger than 2*dense_inter per position. */
static int dense_block(const K3Cfg *c, int T)
{
    const size_t row = (size_t)2 * (c->dense_inter > 0 ? c->dense_inter : 1);
    size_t n = layer_region(c, T) / row;
    if (n < 1) n = 1;
    if (n > (size_t)T) n = (size_t)(T > 0 ? T : 1);
    return (int)n;
}

size_t k3_layer_scratch(const K3Cfg *c, int T)
{
    /* prefix_sum, tmp, hin, fold vectors, one attn_res source stack, plus the region */
    return (size_t)3 * T * c->hidden
         + (size_t)2 * c->hidden
         + (size_t)(c->n_layers / c->attn_res_block + 2) * c->hidden
         + layer_region(c, T);
}

/* The incremental form. Everything except MLA already carries its own state:
 *   - k3_kda_layer updates the recurrent matrix and the ShortConv history in place, so
 *     a decode step simply does not clear them.
 *   - The attn-res block stack is built per token from that token's own hidden states,
 *     with no dependency on earlier tokens, so T=1 needs nothing carried.
 * Only softmax attention has to see every earlier position, which is why the KV cache
 * exists and why it is threaded through here rather than hidden inside k3_mla.
 *
 * kvc == NULL gives exactly the behaviour k3_decoder_layer has always had. */
void k3_decoder_layer_inc(float *h, float *block_residual, int *n_blocks,
                          const K3LayerW *w, const K3Cfg *c, int layer_idx,
                          int T, float *state, float *scratch,
                          float *kvc, float *ropec, int cached, int cap, int kv_latent)
{
    k3_decoder_layer_inc_log(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                             scratch, kvc, ropec, cached, cap, kv_latent, NULL);
}

/* kda_log reaches only the KDA module; see K3KdaLog. Everything else in the layer is
 * per token (norms, AttnRes, the MoE) or positional (the MLA cache), so the log is the
 * only thing a speculative rollback needs from here. */
void k3_decoder_layer_inc_log(float *h, float *block_residual, int *n_blocks,
                              const K3LayerW *w, const K3Cfg *c, int layer_idx,
                              int T, float *state, float *scratch,
                              float *kvc, float *ropec, int cached, int cap, int kv_latent,
                              const K3KdaLog *kda_log)
{
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;

    float *pref   = scratch;                    /* [T][E] the running residual   */
    float *tmp    = pref + (size_t)T * E;       /* [T][E] module output          */
    float *hin    = tmp  + (size_t)T * E;       /* [T][E] normalised layer input */
    float *foldA  = hin  + (size_t)T * E;       /* [E] attention aggregator      */
    float *foldM  = foldA + E;                  /* [E] mlp aggregator            */
    float *src    = foldM + E;                  /* [maxb+1][E] source stack      */
    float *dgu    = src + (size_t)(maxb) * E;   /* the region: [2*dense_inter] ... */
    float *sub    = dgu + (size_t)2 * c->dense_inter;   /* ... then the sub-block    */

    /* The norm gain and the scoring projection collapse to ONE vector. Folding them
     * here costs 2*hidden multiplies per layer; a real engine folds at load time. */
    for (int i = 0; i < E; i++) {
        foldA[i] = w->attn_res_norm[i] * w->attn_res_proj[i];
        foldM[i] = w->mlp_res_norm[i]  * w->mlp_res_proj[i];
    }

    memcpy(pref, h, (size_t)T * E * sizeof(float));
    int have_prefix = 1;                        /* mirrors "prefix_sum is not None" */

    /* aggregation before attention, only when snapshots already exist */
    if (*n_blocks > 0) {
        for (int t = 0; t < T; t++) {
            for (int b = 0; b < *n_blocks; b++)
                memcpy(src + (size_t)b * E,
                       block_residual + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)(*n_blocks) * E, pref + (size_t)t * E,
                   (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, foldA, *n_blocks + 1, E, c->rms_eps);
        }
    }

    /* block boundary: snapshot the running residual, then CLEAR it */
    if (layer_idx % c->attn_res_block == 0) {
        for (int t = 0; t < T; t++)
            memcpy(block_residual + ((size_t)t * maxb + *n_blocks) * E,
                   pref + (size_t)t * E, (size_t)E * sizeof(float));
        (*n_blocks)++;
        have_prefix = 0;
    }

    /* attention */
    for (int t = 0; t < T; t++)
        k3_rmsnorm(hin + (size_t)t * E, h + (size_t)t * E, w->in_norm, E, c->rms_eps);
    if (w->kda) k3_kda_layer_log(tmp, hin, w->kda, c, T, state, sub, kda_log);
    else        k3_mla_cached(tmp, hin, w->mla, c, T, sub, kvc, ropec, cached, cap,
                              kv_latent);

    if (have_prefix) for (size_t i = 0; i < (size_t)T * E; i++) pref[i] += tmp[i];
    else             { memcpy(pref, tmp, (size_t)T * E * sizeof(float)); have_prefix = 1; }

    /* aggregation before the MLP. NO emptiness guard in the reference. */
    for (int t = 0; t < T; t++) {
        for (int b = 0; b < *n_blocks; b++)
            memcpy(src + (size_t)b * E,
                   block_residual + ((size_t)t * maxb + b) * E,
                   (size_t)E * sizeof(float));
        memcpy(src + (size_t)(*n_blocks) * E, pref + (size_t)t * E,
               (size_t)E * sizeof(float));
        k3_attn_res(h + (size_t)t * E, src, foldM, *n_blocks + 1, E, c->rms_eps);
    }

    for (int t = 0; t < T; t++)
        k3_rmsnorm(hin + (size_t)t * E, h + (size_t)t * E, w->post_norm, E, c->rms_eps);

    if (w->moe) {
        int   idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];
        /* Prefill batches (T > 1, streamed source) fetch each unique expert once for
         * the whole chunk; decode (T == 1) and the resident path fall straight through
         * to k3_moe inside, byte-identical. */
        k3_moe_prefill(tmp, hin, w->moe, c, T, idx, wt, sub);
    } else {
        /* Dense MLP (layer 0 only), in blocks of dense_block() positions: gate and up land
         * interleaved per position as [gate | up], SiTU writes its result over the gate
         * half (element i of the output depends only on element i of each half), and down
         * reads that half at stride 2*dense_inter. Each matrix is one pass per block, and
         * per position every value is what the per-position k3_mmw sequence produced. */
        const int di = c->dense_inter, nb = dense_block(c, T);
        for (int t0 = 0; t0 < T; t0 += nb) {
            const int n = T - t0 < nb ? T - t0 : nb;
            const float *ht = hin + (size_t)t0 * E;
            k3_mmw_batch_ld(dgu,      2 * di, ht, E, w->dense_gate, w->wdt, E, di, n);
            k3_mmw_batch_ld(dgu + di, 2 * di, ht, E, w->dense_up,   w->wdt, E, di, n);
            for (int t = 0; t < n; t++) {
                float *gu_t = dgu + (size_t)t * 2 * di;
                k3_situ_glu(gu_t, gu_t, di, c->situ_b1, c->situ_b2);
            }
            k3_mmw_batch_ld(tmp + (size_t)t0 * E, E, dgu, 2 * di, w->dense_down, w->wdt,
                            di, E, n);
        }
    }

    for (size_t i = 0; i < (size_t)T * E; i++) pref[i] += tmp[i];
    memcpy(h, pref, (size_t)T * E * sizeof(float));
}

void k3_decoder_layer(float *h, float *block_residual, int *n_blocks,
                      const K3LayerW *w, const K3Cfg *c, int layer_idx,
                      int T, float *state, float *scratch)
{
    k3_decoder_layer_inc(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                         scratch, NULL, NULL, 0, 0, 0);
}

/* ---------------------------------------------------------------- MXFP4 ---- */
/* OCP MX E2M1: index by the 4-bit code; bit 3 is the sign. */
static const float K3_E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* y[out] = W[out][in] . x[in], with W stored as bf16 and widened on read.
 *
 * WHY THIS EXISTS
 *   The checkpoint ships the dense trunk as bf16. Holding it as fp32 would double the
 *   resident set and double the weight traffic per token. These kernels are bandwidth
 *   bound, so halving the bytes read makes them faster, not slower, the same reason
 *   the MXFP4 path beats dequantise-then-multiply.
 *
 * WHY IT LOSES NOTHING
 *   bf16 is what the checkpoint already contains. Widening bf16 to fp32 is a pure left
 *   shift by 16 bits with no rounding, so multiplying from bf16 storage computes with
 *   exactly the values an fp32 copy would have supplied. The only difference from
 *   k3_matmul is WHERE the widening happens, not what is widened.
 *
 * The accumulator layout mirrors k3_matmul deliberately: the same sixteen partial sums
 * in double, reduced by the same k3_tree16, so the two kernels agree to the bit on
 * identical input (test_ops asserts it).
 *
 * WHERE THE DECODE TIME WENT. On x86 every conversion below issues on the one shuffle
 * port, and that port, not the multiply-adds, was the limit: per 16 weights the old AVX2
 * loop ran four zero-extends and four float-to-double conversions for the weights, and
 * four more conversions that re-widened x, which does not depend on the row at all.
 *
 *   1. x is widened to double ONCE PER CALL (xd), as k3_matmul_mxfp4 already did.
 *
 *   2. No zero-extend. A 32-bit lane of the raw row holds TWO bf16, the even-indexed
 *      element in its low half and the odd-indexed one in its high half. Shifting the
 *      lane left by 16 turns the even one into its float; masking off the low half
 *      turns the odd one into its float. Both are plain ALU operations. The vector loop
 *      therefore sees the eight even-indexed weights of a 16-element chunk in one
 *      register and the eight odd-indexed ones in another, and xd is stored to match
 *      (k3_widen_eo16): per chunk, the eight even-indexed x, then the eight odd ones.
 *
 *   3. Two rows per iteration, sharing every xd load. At the trunk's widths xd is 57 KB,
 *      larger than L1, so this halves the L2 traffic the loop generates; the two rows are
 *      still two independent sums.
 *
 * THE VECTOR PATHS ARE BIT-IDENTICAL TO THE SCALAR PATH, not merely close. The even/odd
 * split decides only which register lane holds which of the sixteen accumulators, never
 * which accumulator an element goes to:
 *
 *   AVX-512  e0 lane k is a[2k], o0 lane k is a[2k + 1];
 *   AVX2     el lane k is a[2k], eh lane k is a[8 + 2k], ol and oh the odd ones;
 *   NEON     natural order, as in k3_f32_row_v (wk holds {a[2k], a[2k + 1]}).
 *
 * Each accumulator still takes the elements i == its index (mod 16) in ascending i, one
 * fused multiply-add each: fma(), _mm256_fmadd_pd, _mm512_fmadd_pd and vfmaq_f64 are the
 * same IEEE operation with a single rounding, and the build's -ffp-contract=off keeps
 * the compiler from fusing anything else. The lanes are stored back to a[16] in natural
 * order and reduced by k3_tree16, and every path finishes the in % 16 tail with the same
 * sequential fma() loop. (The products are exact in any case: a bf16 times a float needs
 * 8 + 24 = 32 significand bits of double's 53, so the only rounding is the sum's.)
 */

/* The whole-chunk part of one bf16 row, the reference form. See k3_f32_row_c: it is the
 * scalar build's path and the vector builds' fallback when xd could not be allocated. */
static inline double k3_bf16_row_c(const uint16_t *row, const float *x, int n16)
{
    double a[16] = {0};
    for (int i = 0; i < n16; i += 16)
        for (int l = 0; l < 16; l++)
            a[l] = fma((double)k3_bf16f(row[i + l]), (double)x[i + l], a[l]);
    return k3_tree16(a);
}

#if defined(__AVX2__)
/* SOFTWARE PREFETCH, x86 only. A decode-time matmul streams its weights from DRAM once
 * per token. With compute between the loads, the hardware prefetcher may not run far
 * enough ahead of a row's read pointer to keep enough misses in flight, and the loop
 * then reads below the machine's streaming bandwidth. One prefetcht0 per 64-byte line,
 * 1 KB ahead of each row's read pointer (512 B for MXFP4, whose 16-byte groups are
 * consumed faster per byte), asks for the lines early. The distances come from
 * exploratory runs on a shared x86 VM that are not recorded, so no speed figure is
 * claimed for them: bench_kernels prints the bf16 kernel's weight traffic beside the
 * machine's read bandwidth, which is the measurement to take on a quiet machine before
 * tuning them. A prefetch is only a hint -- it reads nothing into a register, cannot
 * fault, and changes no arithmetic -- so the one thing it can change is speed. The NEON
 * loops are left without one. */
#define K3_PF_BF16  1024
#define K3_PF_MXFP4 512

/* x widened to double in the EVEN/ODD CHUNK LAYOUT the x86 vector loops read: within the
 * chunk starting at c, xd[c + k] = x[c + 2k] and xd[c + 8 + k] = x[c + 2k + 1] for
 * k = 0..7. The in % 16 tail stays in natural order. Float to double is exact, so every
 * xd value is the one the scalar path forms in place; only the addresses move. Shared
 * with the AVX-512 path of k3_matmul_mxfp4, which splits its nibbles the same way. */
static void k3_widen_eo16(double *xd, const float *x, int in)
{
    const int n16 = in & ~15;
    for (int c = 0; c < n16; c += 16)
        for (int k = 0; k < 8; k++) {
            xd[c + k]     = (double)x[c + 2 * k];
            xd[c + 8 + k] = (double)x[c + 2 * k + 1];
        }
    for (int i = n16; i < in; i++) xd[i] = (double)x[i];
}

/* ev[k] is accumulator a[2k] and od[k] is a[2k + 1]: put them back in natural order and
 * reduce with the one tree. */
static inline double k3_tree16_eo(const double *ev, const double *od)
{
    double a[16];
    for (int k = 0; k < 8; k++) { a[2 * k] = ev[k]; a[2 * k + 1] = od[k]; }
    return k3_tree16(a);
}
#endif

#if defined(__AVX512F__)
/* Rows r0 and r1 over the first n16 elements, xd in the even/odd chunk layout. Per 16
 * elements and row: one 32-byte load, a shift and a mask, two float-to-double
 * conversions and two FMAs. r1 may equal r0 (an odd row count's last row). */
static inline void k3_bf16_rows2_v(double *acc0, double *acc1, const uint16_t *r0,
                                   const uint16_t *r1, const double *xd, int n16)
{
    const __m256i hi = _mm256_set1_epi32((int)0xFFFF0000u);
    __m512d e0 = _mm512_setzero_pd(), o0 = _mm512_setzero_pd();
    __m512d e1 = _mm512_setzero_pd(), o1 = _mm512_setzero_pd();
    for (int i = 0; i < n16; i += 16) {
        const __m512d xe = _mm512_loadu_pd(xd + i);       /* x[i + 2k]     */
        const __m512d xo = _mm512_loadu_pd(xd + i + 8);   /* x[i + 2k + 1] */
        const __m256i w0 = _mm256_loadu_si256((const __m256i *)(r0 + i));
        const __m256i w1 = _mm256_loadu_si256((const __m256i *)(r1 + i));
        if ((i & 31) == 0) {                      /* once per 64-byte line of each row */
            _mm_prefetch((const char *)(r0 + i) + K3_PF_BF16, _MM_HINT_T0);
            _mm_prefetch((const char *)(r1 + i) + K3_PF_BF16, _MM_HINT_T0);
        }
        e0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(
                 _mm256_slli_epi32(w0, 16))), xe, e0);
        o0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(
                 _mm256_and_si256(w0, hi))), xo, o0);
        e1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(
                 _mm256_slli_epi32(w1, 16))), xe, e1);
        o1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(
                 _mm256_and_si256(w1, hi))), xo, o1);
    }
    double ev[8], od[8];
    _mm512_storeu_pd(ev, e0); _mm512_storeu_pd(od, o0); *acc0 = k3_tree16_eo(ev, od);
    _mm512_storeu_pd(ev, e1); _mm512_storeu_pd(od, o1); *acc1 = k3_tree16_eo(ev, od);
}
#elif defined(__AVX2__)
/* The same with 256-bit registers: eight accumulators for the pair, each 16-element
 * chunk loaded as two 16-byte halves so the conversions need no lane extract. Two rows
 * share every xd load, as on AVX-512; the body is ordered so that the eight
 * accumulators, one x vector and two temporaries fit the sixteen ymm registers without
 * a spill. The gain was seen only in unrecorded exploratory runs on a shared VM, so no
 * figure is claimed for it (see the CHANGELOG). */
static inline void k3_bf16_rows2_v(double *acc0, double *acc1, const uint16_t *r0,
                                   const uint16_t *r1, const double *xd, int n16)
{
    const __m128i hi = _mm_set1_epi32((int)0xFFFF0000u);
    __m256d el0 = _mm256_setzero_pd(), eh0 = _mm256_setzero_pd();
    __m256d ol0 = _mm256_setzero_pd(), oh0 = _mm256_setzero_pd();
    __m256d el1 = _mm256_setzero_pd(), eh1 = _mm256_setzero_pd();
    __m256d ol1 = _mm256_setzero_pd(), oh1 = _mm256_setzero_pd();
    for (int i = 0; i < n16; i += 16) {
        const __m128i a0 = _mm_loadu_si128((const __m128i *)(r0 + i));
        const __m128i b0 = _mm_loadu_si128((const __m128i *)(r0 + i + 8));
        const __m128i a1 = _mm_loadu_si128((const __m128i *)(r1 + i));
        const __m128i b1 = _mm_loadu_si128((const __m128i *)(r1 + i + 8));
        if ((i & 31) == 0) {                      /* once per 64-byte line of each row */
            _mm_prefetch((const char *)(r0 + i) + K3_PF_BF16, _MM_HINT_T0);
            _mm_prefetch((const char *)(r1 + i) + K3_PF_BF16, _MM_HINT_T0);
        }
        /* Each x vector feeds both rows at once, which keeps the live registers to
         * the eight accumulators, one x and two temporaries. */
        __m256d xv = _mm256_loadu_pd(xd + i);                  /* x[i + 0, 2, 4, 6]    */
        el0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(a0, 16))),
                              xv, el0);
        el1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(a1, 16))),
                              xv, el1);
        xv = _mm256_loadu_pd(xd + i + 8);                      /* x[i + 1, 3, 5, 7]    */
        ol0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_and_si128(a0, hi))),
                              xv, ol0);
        ol1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_and_si128(a1, hi))),
                              xv, ol1);
        xv = _mm256_loadu_pd(xd + i + 4);                      /* x[i + 8, 10, 12, 14] */
        eh0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(b0, 16))),
                              xv, eh0);
        eh1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(b1, 16))),
                              xv, eh1);
        xv = _mm256_loadu_pd(xd + i + 12);                     /* x[i + 9, 11, 13, 15] */
        oh0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_and_si128(b0, hi))),
                              xv, oh0);
        oh1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_and_si128(b1, hi))),
                              xv, oh1);
    }
    double ev[8], od[8];
    _mm256_storeu_pd(ev, el0); _mm256_storeu_pd(ev + 4, eh0);
    _mm256_storeu_pd(od, ol0); _mm256_storeu_pd(od + 4, oh0);
    *acc0 = k3_tree16_eo(ev, od);
    _mm256_storeu_pd(ev, el1); _mm256_storeu_pd(ev + 4, eh1);
    _mm256_storeu_pd(od, ol1); _mm256_storeu_pd(od + 4, oh1);
    *acc1 = k3_tree16_eo(ev, od);
}
#elif defined(__ARM_NEON) && defined(__aarch64__)
/* One row, natural order: wk holds {a[2k], a[2k + 1]}, exactly as k3_f32_row_v. bf16 to
 * f32 is the usual 16-bit left shift; vshll_n_u16 widens and shifts in one instruction.
 * The x widening this loop used to repeat per row (eight vcvt per 16 elements) is gone:
 * xd holds x already widened, in natural order. */
static inline double k3_bf16_row_neon(const uint16_t *row, const double *xd, int n16)
{
    float64x2_t w0 = vdupq_n_f64(0.0), w1 = vdupq_n_f64(0.0);
    float64x2_t w2 = vdupq_n_f64(0.0), w3 = vdupq_n_f64(0.0);
    float64x2_t w4 = vdupq_n_f64(0.0), w5 = vdupq_n_f64(0.0);
    float64x2_t w6 = vdupq_n_f64(0.0), w7 = vdupq_n_f64(0.0);
    for (int i = 0; i < n16; i += 16) {
        const uint16x8_t h0 = vld1q_u16(row + i);
        const uint16x8_t h1 = vld1q_u16(row + i + 8);
        const float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
        const float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
        const float32x4_t f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        const float32x4_t f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        w0 = vfmaq_f64(w0, vcvt_f64_f32(vget_low_f32(f0)), vld1q_f64(xd + i));
        w1 = vfmaq_f64(w1, vcvt_high_f64_f32(f0),          vld1q_f64(xd + i + 2));
        w2 = vfmaq_f64(w2, vcvt_f64_f32(vget_low_f32(f1)), vld1q_f64(xd + i + 4));
        w3 = vfmaq_f64(w3, vcvt_high_f64_f32(f1),          vld1q_f64(xd + i + 6));
        w4 = vfmaq_f64(w4, vcvt_f64_f32(vget_low_f32(f2)), vld1q_f64(xd + i + 8));
        w5 = vfmaq_f64(w5, vcvt_high_f64_f32(f2),          vld1q_f64(xd + i + 10));
        w6 = vfmaq_f64(w6, vcvt_f64_f32(vget_low_f32(f3)), vld1q_f64(xd + i + 12));
        w7 = vfmaq_f64(w7, vcvt_high_f64_f32(f3),          vld1q_f64(xd + i + 14));
    }
    /* (a[l]+a[4+l])+(a[8+l]+a[12+l]) lanewise -- t0 = {b0,b1}, t1 = {b2,b3} -- then
     * (b0+b1)+(b2+b3): k3_tree16 exactly. */
    const float64x2_t t0 = vaddq_f64(vaddq_f64(w0, w2), vaddq_f64(w4, w6));
    const float64x2_t t1 = vaddq_f64(vaddq_f64(w1, w3), vaddq_f64(w5, w7));
    return vaddvq_f64(t0) + vaddvq_f64(t1);
}

/* NEON keeps one row per pass: nothing here can measure a two-row form on Apple
 * Silicon, and its three load ports make the shared-xd saving small there. */
static inline void k3_bf16_rows2_v(double *acc0, double *acc1, const uint16_t *r0,
                                   const uint16_t *r1, const double *xd, int n16)
{
    *acc0 = k3_bf16_row_neon(r0, xd, n16);
    *acc1 = (r1 != r0) ? k3_bf16_row_neon(r1, xd, n16) : *acc0;
}
#endif

void k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
    const int n16 = in & ~15;

    /* WIDEN x ONCE, in the order the vector loop reads it (see the notes above and
     * "WIDEN x ONCE" in k3_matmul_mxfp4). Read-only and shared by every thread. NULL is
     * a valid state: the rows then take k3_bf16_row_c, the same sum without the copy,
     * so an allocation failure costs speed and nothing else. */
#if K3_MM_HOIST
    double *const xd = (double *)malloc((size_t)in * sizeof(double));
    if (xd) {
#if defined(__AVX2__)
        k3_widen_eo16(xd, x, in);
#else
        for (int i = 0; i < in; i++) xd[i] = (double)x[i];
#endif
    }
#endif

    /* Row PAIRS are the unit of parallel work. Output rows stay independent -- each is
     * summed by exactly one thread in exactly the order above -- so pairing them changes
     * no arithmetic and results remain identical at any thread count. An odd last row
     * is paired with itself and stored once. */
    const int npair = (out + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int p = 0; p < npair; p++) {
        const int o0 = 2 * p;
        const int o1 = (o0 + 1 < out) ? o0 + 1 : o0;
        const uint16_t *r0 = W + (size_t)o0 * in;
        const uint16_t *r1 = W + (size_t)o1 * in;
        double acc0, acc1;
#if K3_MM_HOIST
        if (xd) k3_bf16_rows2_v(&acc0, &acc1, r0, r1, xd, n16);
        else
#endif
        {
            acc0 = k3_bf16_row_c(r0, x, n16);
            acc1 = (o1 != o0) ? k3_bf16_row_c(r1, x, n16) : acc0;
        }
        for (int i = n16; i < in; i++) {
            acc0 = fma((double)k3_bf16f(r0[i]), (double)x[i], acc0);
            acc1 = fma((double)k3_bf16f(r1[i]), (double)x[i], acc1);
        }
        y[o0] = k3_out_f32(acc0);
        if (o1 != o0) y[o1] = k3_out_f32(acc1);
    }

#if K3_MM_HOIST
    free(xd);                                     /* free(NULL) is a no-op */
#endif
}

/* Per-row int8 matmul for the draft model: each row is [f32 scale][int8 * in]. The int8
 * weights are widened to float, dotted with the fp32 activation, and the row's scale is
 * applied once at the end. Unlike the trunk kernels this carries NO cross-path
 * determinism contract (K3_WI8 is draft-only, and the exact model decides every emitted
 * token), so it accumulates in float with fused products and the natural AVX2 reduction,
 * which is what makes it fast. */
void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out)
{
    const unsigned char *base = (const unsigned char *)W;
    const size_t rowb = (size_t)4 + (size_t)in;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const unsigned char *row = base + (size_t)o * rowb;
        float scale;
        memcpy(&scale, row, 4);
        const int8_t *w = (const int8_t *)(row + 4);
        int i = 0;
        float acc;
#if defined(__AVX2__)
        {
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (; i + 15 < in; i += 16) {
                const __m128i b0 = _mm_loadl_epi64((const __m128i *)(w + i));
                const __m128i b1 = _mm_loadl_epi64((const __m128i *)(w + i + 8));
                v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),
                                     _mm256_loadu_ps(x + i), v0);
                v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),
                                     _mm256_loadu_ps(x + i + 8), v1);
            }
            __m256 vs = _mm256_add_ps(v0, v1);
            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs),
                                   _mm256_extractf128_ps(vs, 1));
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            acc = _mm_cvtss_f32(lo);
        }
#elif defined(__ARM_NEON) && defined(__aarch64__)
        {
            /* Draft-only kernel, no determinism contract: fused float accumulation
             * and the natural NEON reduction, same as the AVX2 form's spirit. */
            float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
            float32x4_t v2 = vdupq_n_f32(0.0f), v3 = vdupq_n_f32(0.0f);
            for (; i + 15 < in; i += 16) {
                const int8x16_t b = vld1q_s8(w + i);
                const int16x8_t s0 = vmovl_s8(vget_low_s8(b));
                const int16x8_t s1 = vmovl_s8(vget_high_s8(b));
                v0 = vfmaq_f32(v0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))),
                               vld1q_f32(x + i));
                v1 = vfmaq_f32(v1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))),
                               vld1q_f32(x + i + 4));
                v2 = vfmaq_f32(v2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))),
                               vld1q_f32(x + i + 8));
                v3 = vfmaq_f32(v3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))),
                               vld1q_f32(x + i + 12));
            }
            acc = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        }
#else
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (; i + 3 < in; i += 4) {
                a0 += (float)w[i]     * x[i];
                a1 += (float)w[i + 1] * x[i + 1];
                a2 += (float)w[i + 2] * x[i + 2];
                a3 += (float)w[i + 3] * x[i + 3];
            }
            acc = (a0 + a1) + (a2 + a3);
        }
#endif
        for (; i < in; i++) acc += (float)w[i] * x[i];
        y[o] = acc * scale;
    }
}

/* A whole BYTE to its two E2M1 values, so the inner loop does one 8-byte load instead
 * of masking, shifting and two separate lookups. 2 KB, built once, shared by all
 * threads after initialisation.
 *
 * SCALAR PATH ONLY. The AVX2 path decodes nibbles with permutevar8x32 in register and
 * never touches this table, so building it there would be 2 KB of cold cache and an
 * unused-symbol warning. See k3_matmul_mxfp4. */
#if !defined(__AVX2__)
static float K3_E2M1_PAIR[256][2];
static int   k3_pair_ready = 0;

static void k3_pair_init(void)
{
    for (int b = 0; b < 256; b++) {
        K3_E2M1_PAIR[b][0] = K3_E2M1[b & 0x0F];   /* low nibble  = EVEN element */
        K3_E2M1_PAIR[b][1] = K3_E2M1[b >> 4];     /* high nibble = ODD element  */
    }
    k3_pair_ready = 1;
}
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
/* Every E2M1 value's f32 bit pattern has zero low 16 bits, so a 4-bit code expands to
 * (B3[code] << 24) | (B2[code] << 16). Two 16-entry byte tables therefore cover the
 * whole lookup, and vqtbl1q_u8 resolves 16 codes per instruction -- this is what lets
 * the NEON path expand a group in registers instead of through the wf[] buffer. The
 * expanded bit patterns are identical to K3_E2M1_PAIR's floats, so using them changes
 * no arithmetic. */
static const uint8_t K3_E2M1_B2[16] = {
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0
};
static const uint8_t K3_E2M1_B3[16] = {
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
    0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0
};
#endif

/* E8M0 byte to its power of two. 255 is NaN by spec and maps to zero. Precomputed
 * because ldexpf in the group loop is a function call the compiler will not inline
 * into a vectorised body. */
static float K3_E8M0[256];
static int   k3_e8m0_ready = 0;

static void k3_e8m0_init(void)
{
    for (int b = 0; b < 256; b++) K3_E8M0[b] = (b == 255) ? 0.0f : ldexpf(1.0f, b - 127);
    k3_e8m0_ready = 1;
}

#if defined(__AVX512F__)
/* AVX-512 FORM OF THE AVX2 FLAT ROW PATH, BIT-IDENTICAL TO IT. See the FLAT ROW PATH
 * comment inside k3_matmul_mxfp4 for the arithmetic; this changes only how each
 * operand reaches its lane.
 *
 * DECODE BY TABLE, IN DOUBLE. The AVX2 loop decodes a nibble to a float through a
 * permute and a sign XOR, then widens it with _mm256_cvtps_pd; with the zero-extends,
 * lane extracts and scale broadcast, that is thirteen shuffle-port operations per 16
 * weights, and the shuffle port is what bounds it. Here the sixteen possible weights of
 * a group -- every E2M1 code times that group's scale -- are built once per group as
 * DOUBLES in two registers, and _mm512_permutex2var_pd, which indexes sixteen doubles by
 * the low four bits of each 64-bit lane, turns a code straight into its widened weight.
 * Per 16 weights: one 8-byte zero-extend, one shift, two lookups, two FMAs.
 *
 * WHY THE TABLE HOLDS EXACTLY THE AVX2 VALUES. AVX2 uses w = (double)(float)(E2M1[c] *
 * K3_E8M0[sb]), the product rounded to float and then widened. For 2 <= sb <= 252 that
 * float product is exact -- a 3-bit significand times a power of two, neither
 * overflowing (6 * 2^125 < FLT_MAX) nor falling below the normal range (0.5 * 2^-125 is
 * normal) -- so computing it in double from the exact double scale gives the same value.
 * sb 0, 1, 253 and 254 are where the float product is subnormal or overflows to inf;
 * for those the table is built the AVX2 way, a float multiply then a widen, so it
 * reproduces the subnormal (and any flush-to-zero mode) and the inf exactly. Negative
 * codes are the negated magnitudes times the scale: IEEE multiplication is symmetric in
 * sign, so -(a*s) == (-a)*s bit for bit, the same value AVX2 gets by XORing the sign bit
 * into a*s (including -0.0 for code 8). */
static inline void k3_e2m1_table512(__m512d *lo, __m512d *hi, unsigned sb)
{
    if (sb >= 2 && sb <= 252) {
        const __m512d s = _mm512_set1_pd((double)K3_E8M0[sb]);   /* exact: a power of two */
        *lo = _mm512_mul_pd(_mm512_setr_pd(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), s);
        *hi = _mm512_mul_pd(_mm512_setr_pd(-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), s);
    } else {
        const __m256 s = _mm256_set1_ps(K3_E8M0[sb]);
        *lo = _mm512_cvtps_pd(_mm256_mul_ps(
            _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f), s));
        *hi = _mm512_cvtps_pd(_mm256_mul_ps(
            _mm256_setr_ps(-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f), s));
    }
}

/* The AVX2 flat path's reduction, in accumulator terms. Its lanewise (v0+v2)+(v1+v3)
 * is q[l] = (a[l] + a[l+8]) + (a[l+4] + a[l+12]); it then adds the two 128-bit halves,
 * (q0+q2, q1+q3), and finally those two lanes. That pairing differs from k3_tree16's,
 * and each must stay as it is: changing either tree changes bits. ev[k] is a[2k] and
 * od[k] is a[2k+1], as in k3_tree16_eo. */
static inline double k3_tree16_flat_eo(const double *ev, const double *od)
{
    double a[16], q[4];
    for (int k = 0; k < 8; k++) { a[2 * k] = ev[k]; a[2 * k + 1] = od[k]; }
    for (int l = 0; l < 4; l++) q[l] = (a[l] + a[l + 8]) + (a[l + 4] + a[l + 12]);
    return (q[0] + q[2]) + (q[1] + q[3]);
}

/* Two rows of the flat path, xd in the even/odd chunk layout (k3_widen_eo16): per
 * 16-element chunk, byte k of the 8 packed bytes holds element 2k in its low nibble and
 * element 2k+1 in its high nibble, so the zero-extended bytes index the even weights
 * directly (the lookup reads only bits 3..0) and the same bytes shifted right by 4
 * index the odd ones. Lane k of e is accumulator a[2k] and lane k of o is a[2k+1]:
 * every element still lands in a[i % 16], in ascending i, one FMA each.
 *
 * A NaN scale byte (255) SKIPS its chunks, exactly as the AVX2 loop's goto does: the
 * FMAs run under a zero write-mask, which leaves the accumulators untouched, rather
 * than adding a product of zero (0 * inf would be NaN, and a skipped chunk must not be
 * able to produce one). The in % 16 tail is the AVX2 path's scalar loop verbatim,
 * which does NOT skip a 255 group -- also reproduced. r1 may equal r0 (an odd row
 * count's last row).
 *
 * e8d[b] is (double)K3_E8M0[b], widened once per call so a group's scale reaches a
 * register as one broadcast load. The common case -- a whole 32-element group (K3's
 * group size) whose two scale bytes both lie in 2..252 -- runs straight-line with no
 * mask and a two-multiply table per row; everything else (other group sizes, the last
 * partial group, 255 and the four float-table scale bytes) takes the general loop,
 * which computes the same values the slower way. */
static inline void k3_mxfp4_rows2_avx512(double *acc, const unsigned char *p0,
                                         const unsigned char *s0, const unsigned char *p1,
                                         const unsigned char *s1, const double *xd,
                                         const double *e8d, int in, int group)
{
    const int n16 = in & ~15;
    const __m512d LO = _mm512_setr_pd(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    const __m512d HI = _mm512_setr_pd(-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
    __m512d e0 = _mm512_setzero_pd(), o0 = _mm512_setzero_pd();
    __m512d e1 = _mm512_setzero_pd(), o1 = _mm512_setzero_pd();
    int i = 0;
    for (int g = 0; i < n16; g++) {               /* i == g * group at the top */
        const int gend = (n16 - i > group) ? i + group : n16;
        const unsigned b0 = s0[g], b1 = s1[g];
        if (gend - i == 32 && b0 - 2u <= 250u && b1 - 2u <= 250u) {
            const __m512d c0 = _mm512_set1_pd(e8d[b0]), c1 = _mm512_set1_pd(e8d[b1]);
            const __m512d t0lo = _mm512_mul_pd(LO, c0), t0hi = _mm512_mul_pd(HI, c0);
            const __m512d t1lo = _mm512_mul_pd(LO, c1), t1hi = _mm512_mul_pd(HI, c1);
            const unsigned char *q0 = p0 + ((unsigned)i >> 1);
            const unsigned char *q1 = p1 + ((unsigned)i >> 1);
            const double *xg = xd + i;
            if ((g & 3) == 0) {                   /* four 16-byte groups per line */
                _mm_prefetch((const char *)q0 + K3_PF_MXFP4, _MM_HINT_T0);
                _mm_prefetch((const char *)q1 + K3_PF_MXFP4, _MM_HINT_T0);
            }
            for (int h = 0; h < 2; h++) {         /* the group's two chunks, in order */
                const __m512d xe = _mm512_loadu_pd(xg + 16 * h);
                const __m512d xo = _mm512_loadu_pd(xg + 16 * h + 8);
                const __m512i n0 = _mm512_cvtepu8_epi64(
                    _mm_loadl_epi64((const __m128i *)(q0 + 8 * h)));
                const __m512i n1 = _mm512_cvtepu8_epi64(
                    _mm_loadl_epi64((const __m128i *)(q1 + 8 * h)));
                e0 = _mm512_fmadd_pd(_mm512_permutex2var_pd(t0lo, n0, t0hi), xe, e0);
                o0 = _mm512_fmadd_pd(
                    _mm512_permutex2var_pd(t0lo, _mm512_srli_epi64(n0, 4), t0hi), xo, o0);
                e1 = _mm512_fmadd_pd(_mm512_permutex2var_pd(t1lo, n1, t1hi), xe, e1);
                o1 = _mm512_fmadd_pd(
                    _mm512_permutex2var_pd(t1lo, _mm512_srli_epi64(n1, 4), t1hi), xo, o1);
            }
            i += 32;
            continue;
        }
        const __mmask8 k0 = (b0 == 255) ? 0 : 0xFF;
        const __mmask8 k1 = (b1 == 255) ? 0 : 0xFF;
        __m512d t0lo, t0hi, t1lo, t1hi;
        k3_e2m1_table512(&t0lo, &t0hi, b0);
        k3_e2m1_table512(&t1lo, &t1hi, b1);
        for (; i < gend; i += 16) {
            const __m512d xe = _mm512_loadu_pd(xd + i);       /* x[i + 2k]     */
            const __m512d xo = _mm512_loadu_pd(xd + i + 8);   /* x[i + 2k + 1] */
            const __m512i n0 = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64((const __m128i *)(p0 + ((unsigned)i >> 1))));
            const __m512i n1 = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64((const __m128i *)(p1 + ((unsigned)i >> 1))));
            e0 = _mm512_mask3_fmadd_pd(_mm512_permutex2var_pd(t0lo, n0, t0hi),
                                       xe, e0, k0);
            o0 = _mm512_mask3_fmadd_pd(
                _mm512_permutex2var_pd(t0lo, _mm512_srli_epi64(n0, 4), t0hi), xo, o0, k0);
            e1 = _mm512_mask3_fmadd_pd(_mm512_permutex2var_pd(t1lo, n1, t1hi),
                                       xe, e1, k1);
            o1 = _mm512_mask3_fmadd_pd(
                _mm512_permutex2var_pd(t1lo, _mm512_srli_epi64(n1, 4), t1hi), xo, o1, k1);
        }
    }
    double ev[8], od[8];
    _mm512_storeu_pd(ev, e0); _mm512_storeu_pd(od, o0); acc[0] = k3_tree16_flat_eo(ev, od);
    _mm512_storeu_pd(ev, e1); _mm512_storeu_pd(od, o1); acc[1] = k3_tree16_flat_eo(ev, od);
    /* Tail, at most 15 elements in one group; xd is in natural order here. */
    for (; i < in; i++) {
        const unsigned char n0 = (i & 1) ? (p0[i >> 1] >> 4) : (p0[i >> 1] & 0x0F);
        const unsigned char n1 = (i & 1) ? (p1[i >> 1] >> 4) : (p1[i >> 1] & 0x0F);
        acc[0] = fma((double)(K3_E2M1[n0] * K3_E8M0[s0[i / group]]), xd[i], acc[0]);
        acc[1] = fma((double)(K3_E2M1[n1] * K3_E8M0[s1[i / group]]), xd[i], acc[1]);
    }
}
#endif

/* y[rows] = W[rows][in] . x[in], with W read straight out of packed MXFP4 and never
 * materialised as floats. This is not an optimisation; it is what makes streaming
 * experts possible at all.
 *
 * One routed expert is 33,030,144 parameters. Dequantised to fp32 that is 132 MB, so
 * the 1,472 experts a single token touches would be 194 GB of materialised weights. As
 * packed nibbles the same expert is 17.55 MB. A matrix-vector product is memory bound,
 * so reading 7.5x fewer bytes makes this kernel FASTER than dequantising first.
 *
 * The loop is structured around the 32-element group because the scale is constant
 * within one: it factors out of the inner sum and is applied once per group instead of
 * once per element. At group 32 each group is exactly 16 packed bytes.
 *
 * PRECONDITIONS:
 *   - group <= 64. The expanded group goes into a fixed wf[64] stack buffer.
 *     Checked below; violating it aborts with a FATAL message rather than
 *     overflowing the buffer.
 *   - `in` is even. The packed row stride is in/2 bytes, two elements per byte.
 *     Checked below for the same reason.
 *   - `packed` is rows x (in/2) bytes; `scales` is rows x ceil(in/group) bytes.
 *     Not checked; the caller owns these buffer sizes.
 *   - A scale byte of 255 is NaN by the OCP MX spec and zeroes its whole group.
 *
 * ACCURACY CONTRACT. This kernel is deliberately NOT bit-identical to
 * dequantise-then-k3_matmul, and no caller should assume it is. On AVX2 with a
 * group that is a multiple of 16 (K3 uses 32) it runs the FLAT ROW PATH below: the
 * E8M0 power-of-two scale is folded into each E2M1 weight exactly, and the whole
 * row is summed by sixteen independent double accumulator lanes (four __m256d).
 * Otherwise it sums each
 * group of 32 and applies that group's scale before accumulating. Both orderings
 * differ from dequantise-then-matmul's single accumulator set.
 *
 * The difference is bounded and tiny. Every individual product is EXACT in double, an
 * E2M1 value carries 3 mantissa bits and x carries 24, so the product needs 27 of the
 * 53 available, so only the additions round, and reassociating exact terms moves the
 * result by roughly 1 ULP of double, order 1e-16 relative. The required agreement is
 * 1e-6 against dequantise-then-matmul on real checkpoint weights, gated by
 * tests/unit/test_expert.c. The margin is nine orders of magnitude.
 */
void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                     const unsigned char *scales, int in, int rows, int group)
{
    if (in & 1) {
        fprintf(stderr,
                "k3: FATAL, k3_matmul_mxfp4 called with in=%d, which is odd.\n"
                "    Packed rows are in/2 bytes, two elements per byte; an odd `in`\n"
                "    truncates that stride below what the trailing group's odd\n"
                "    remainder reads, a heap read past the caller's buffer instead\n"
                "    of failing loudly.\n",
                in);
        abort();
    }
    if (group > 64) {
        fprintf(stderr,
                "k3: FATAL, k3_matmul_mxfp4 called with group=%d, which exceeds 64.\n"
                "    Each group is expanded into a fixed wf[64] stack buffer before\n"
                "    the dot product; a larger group overflows it instead of failing\n"
                "    loudly.\n",
                group);
        abort();
    }

    const int pcols = in / 2;                     /* two elements per byte */
    const int ngrp  = (in + group - 1) / group;
    const int gbyte = group / 2;

#if !defined(__AVX2__)
    if (!k3_pair_ready)  k3_pair_init();
#endif
    if (!k3_e8m0_ready)  k3_e8m0_init();

    /* WIDEN x ONCE, NOT ONCE PER ROW. The accumulators are double, so every row used to
     * re-run the same `in` float-to-double conversions -- 3072 rows x 3584 elements is
     * 11 M conversions per call to produce 3584 distinct values. x does not depend on r,
     * so it is hoisted here and the row loop reads doubles directly.
     *
     * BIT-IDENTICAL: float to double is exact (24 mantissa bits into 53), so the widened
     * copy holds precisely what _mm256_cvtps_pd produced in place.
     *
     * Read-only and shared by every thread, so one copy serves the whole parallel
     * region. At the K3 shapes it is 28 KB, which stays in L2 while the packed weights
     * stream past it.
     *
     * A FAILED ALLOCATION ABORTS ON x86 when group % 16 == 0. The flat row path below
     * exists only with the copy; without it the rows would take the grouped path, whose
     * summation order is different, so the output bits would follow the allocator -- not
     * a speed cost but a change of result, which the engine does not allow (see the
     * fatal-error note at the top of this file). Elsewhere NULL is a valid state: the
     * group loop widens each group into a small stack buffer, the same values the copy
     * holds, so there an allocation failure costs speed and nothing else.
     *
     * On AVX-512 the flat path reads x in the even/odd chunk layout its nibble split
     * produces (k3_widen_eo16); every other path reads natural order. */
    double *const xd = (double *)malloc((size_t)in * sizeof(double));
#if defined(__AVX2__)
    if (!xd && (group & 15) == 0)
        k3_fatal_oom("the widened x of an MXFP4 matmul", (size_t)in * sizeof(double));
#endif
#if defined(__AVX512F__)
    const int flat512 = xd && (group & 15) == 0;
    if (flat512) k3_widen_eo16(xd, x, in);
    else
#endif
    if (xd) for (int i = 0; i < in; i++) xd[i] = (double)x[i];

#if defined(__AVX512F__)
    /* The AVX-512 flat path, two rows per iteration sharing each xd load. It runs under
     * exactly the condition the AVX2 flat path does, so an AVX-512 build and an AVX2
     * build of this file agree bit for bit on every input, including the
     * group % 16 != 0 case, which falls through to the unchanged loop below (and a
     * failed xd, which aborted above in both builds).
     * Rows stay independent: pairing them changes no arithmetic. */
    if (flat512) {
        double e8d[256];                          /* exact: float to double */
        for (int b = 0; b < 256; b++) e8d[b] = (double)K3_E8M0[b];
        const int npair = (rows + 1) / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 64)
#endif
        for (int p = 0; p < npair; p++) {
            const int r0 = 2 * p;
            const int r1 = (r0 + 1 < rows) ? r0 + 1 : r0;
            double acc[2];
            k3_mxfp4_rows2_avx512(acc, packed + (size_t)r0 * pcols,
                                  scales + (size_t)r0 * ngrp,
                                  packed + (size_t)r1 * pcols,
                                  scales + (size_t)r1 * ngrp, xd, e8d, in, group);
            y[r0] = (float)acc[0];
            if (r1 != r0) y[r1] = (float)acc[1];
        }
        free(xd);
        return;
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 64)
#endif
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        double acc = 0.0;

#if defined(__AVX2__)
        if (xd && (group & 15) == 0) {
            /* FLAT ROW PATH. Every E8M0 scale is a power of two (K3_E8M0[sb] =
             * 2^(sb-127), 0 for the NaN byte 255), so folding it into the E2M1 weight
             * before the FMA is EXACT in fp32: a 3-bit mantissa shifted by an exponent
             * does not round, and the only overflow case (sb >= 251 times weight 6)
             * produces inf exactly as the dequantised reference does. That folds the
             * per-group scale step out of the accumulation, so the whole row is one
             * flat vectorised dot product: four independent double accumulator chains
             * v0..v3, a single horizontal reduction at the end, and none of the
             * per-group scalar reduction, the a[4] store-forwarding, or the serial
             * `acc` chain that the grouped path below pays once per group.
             *
             * The accumulator count is a deliberate trade-off. An earlier version
             * ran eight chains over 32-element blocks, but eight accumulators plus
             * the decode temporaries exceed the 16 ymm registers and the compiler
             * spills one accumulator to the stack every block - reintroducing the
             * store-forwarding this path exists to avoid. Four chains over 16-element
             * chunks fit without a spill, and the loop is decode-bound on the shuffle
             * port (permutevar8x32, cvtepu8_epi32 and cvtps_pd all issue there), not
             * FMA-latency-bound, so the shorter chain depth is not what limits it.
             *
             * Lane k of vj holds the elements == 4j + k (mod 16), and the final tree
             * is ((v0+v2)+(v1+v3)) lane-wise, then the two 128-bit halves added, then
             * the last two lanes, so every lane holds a sum of the same element classes
             * in the same order as the dequantised reference and the error stays a few
             * ulps of double, far inside the 1e-6 gate. These are the bits the engine
             * emits on x86; the AVX-512 path (k3_mxfp4_rows2_avx512) reproduces them.
             *
             * Requires `group` to be a multiple of 16 so no 16-element chunk straddles
             * a scale boundary (K3 uses group 32). Anything else takes the grouped path
             * below, which handles arbitrary group <= 64. */
            const __m256  LUT  = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f,
                                                2.0f, 3.0f, 4.0f, 6.0f);
            const __m128i m0f  = _mm_set1_epi8(0x0F);
            const __m256i m07  = _mm256_set1_epi32(7);
            const __m256i m08  = _mm256_set1_epi32(8);
            __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
            int i = 0, g = 0, c = 0;
            const int cpg = group >> 4;           /* 16-element chunks per group */

            for (; i + 15 < in; i += 16) {
                const unsigned char sb = sr[g];
                if (sb == 255) goto flat_skip;
                {
                    const __m256 LUTs = _mm256_mul_ps(
                        LUT, _mm256_set1_ps(K3_E8M0[sb]));
                    const __m128i b = _mm_loadl_epi64(
                        (const __m128i *)(pr + (i >> 1)));
                    const __m128i u = _mm_unpacklo_epi8(
                        _mm_and_si128(b, m0f),
                        _mm_and_si128(_mm_srli_epi16(b, 4), m0f));
                    const __m256i c0 = _mm256_cvtepu8_epi32(u);
                    const __m256i c1 = _mm256_cvtepu8_epi32(_mm_srli_si128(u, 8));
                    const __m256  w0 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUTs, _mm256_and_si256(c0, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c0, m08), 28)));
                    const __m256  w1 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUTs, _mm256_and_si256(c1, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c1, m08), 28)));
                    v0 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_castps256_ps128(w0)),
                        _mm256_loadu_pd(xd + i), v0);
                    v1 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)),
                        _mm256_loadu_pd(xd + i + 4), v1);
                    v2 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_castps256_ps128(w1)),
                        _mm256_loadu_pd(xd + i + 8), v2);
                    v3 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)),
                        _mm256_loadu_pd(xd + i + 12), v3);
                }
            flat_skip:
                if (++c == cpg) { c = 0; g++; }
            }
            {
                /* Horizontal reduction without touching memory. Lanewise
                 * q = (v0+v2)+(v1+v3), then (q0+q2)+(q1+q3): the 128-bit halves are
                 * added first. That is NOT the grouped path's (q0+q1)+(q2+q3), and the
                 * AVX-512 path (k3_tree16_flat_eo) reproduces this one exactly. */
                const __m256d q = _mm256_add_pd(
                    _mm256_add_pd(v0, v2), _mm256_add_pd(v1, v3));
                const __m128d t = _mm_add_pd(
                    _mm256_castpd256_pd128(q), _mm256_extractf128_pd(q, 1));
                acc = _mm_cvtsd_f64(_mm_add_sd(t, _mm_unpackhi_pd(t, t)));
            }
            /* Scalar tail, at most 15 elements, all inside one group (i is 16-aligned
             * and group is a multiple of 16, so a tail this short cannot cross a scale
             * boundary). Scale folded the same way as the vector path. */
            for (; i < in; i++) {
                const unsigned char by = pr[i >> 1];
                const unsigned char nib = (i & 1) ? (by >> 4) : (by & 0x0F);
                acc = fma((double)(K3_E2M1[nib] * K3_E8M0[sr[i / group]]), xd[i], acc);
            }
            y[r] = (float)acc;
            continue;
        }
#endif
        for (int g = 0; g < ngrp; g++) {
            const unsigned char sb = sr[g];
            if (sb == 255) continue;              /* NaN scale: contribute nothing */
            const unsigned char *pb = pr + (size_t)g * gbyte;
            const float *xg = x + (size_t)g * group;

            int n = in - g * group;
            if (n > group) n = group;

            /* One pointer for both paths, so the hot loop carries no test. The fallback
             * branch is per group, not per element, and only runs when the hoist above
             * could not allocate. */
            double xlocal[64];                    /* group <= 64, same bound as wf */
            const double *xdg;
            if (xd) {
                xdg = xd + (size_t)g * group;
            } else {
                for (int j = 0; j < n; j++) xlocal[j] = (double)xg[j];
                xdg = xlocal;
            }

            double sub;
#if defined(__ARM_NEON) && defined(__aarch64__)
            if (n == 32) {
                /* Full-group fast path: expand all 32 nibbles in registers and feed
                 * the dot product directly, no wf[] round-trip. vzip1/vzip2 on the
                 * (low nibble, high nibble) pair restores element order -- low nibble
                 * is the EVEN element -- then vqtbl1q_u8 resolves 16 codes at a time
                 * against the two byte tables and vzip+vshll assembles the f32 bit
                 * patterns. The values, the element order, and the u0..u3 accumulator
                 * partition are exactly the wf path's, so this stays bit-identical. */
                const uint8x16_t pkv = vld1q_u8(pb);
                const uint8x16_t nlo = vandq_u8(pkv, vdupq_n_u8(0x0F));
                const uint8x16_t nhi = vshrq_n_u8(pkv, 4);
                const uint8x16_t t2  = vld1q_u8(K3_E2M1_B2);
                const uint8x16_t t3  = vld1q_u8(K3_E2M1_B3);
                float64x2_t u0 = vdupq_n_f64(0.0), u1 = vdupq_n_f64(0.0);
                float64x2_t u2 = vdupq_n_f64(0.0), u3 = vdupq_n_f64(0.0);
                for (int k = 0; k < 2; k++) {
                    const uint8x16_t c = k ? vzip2q_u8(nlo, nhi)  /* codes 16..31 */
                                           : vzip1q_u8(nlo, nhi); /* codes  0..15 */
                    const uint8x16_t b2 = vqtbl1q_u8(t2, c);
                    const uint8x16_t b3 = vqtbl1q_u8(t3, c);
                    /* byte pairs (b2,b3) as u16 = b3<<8 | b2; shift 16 more for f32 */
                    const uint16x8_t h0 = vreinterpretq_u16_u8(vzip1q_u8(b2, b3));
                    const uint16x8_t h1 = vreinterpretq_u16_u8(vzip2q_u8(b2, b3));
                    const float32x4_t w0v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
                    const float32x4_t w1v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
                    const float32x4_t w2v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
                    const float32x4_t w3v =
                        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
                    /* x comes from xdg, already widened: the eight vcvt per 16
                     * elements that re-widened x for every row are gone. xdg[j] is
                     * (double)xg[j] exactly, so every fma sees the operands it did
                     * before, in the same order per accumulator. */
                    const double *xk = xdg + 16 * k;
                    u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(w0v)), vld1q_f64(xk));
                    u1 = vfmaq_f64(u1, vcvt_high_f64_f32(w0v),          vld1q_f64(xk + 2));
                    u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(w1v)), vld1q_f64(xk + 4));
                    u3 = vfmaq_f64(u3, vcvt_high_f64_f32(w1v),          vld1q_f64(xk + 6));
                    u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(w2v)), vld1q_f64(xk + 8));
                    u1 = vfmaq_f64(u1, vcvt_high_f64_f32(w2v),          vld1q_f64(xk + 10));
                    u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(w3v)), vld1q_f64(xk + 12));
                    u3 = vfmaq_f64(u3, vcvt_high_f64_f32(w3v),          vld1q_f64(xk + 14));
                }
                const float64x2_t t0 = vaddq_f64(u0, u2);
                const float64x2_t t1 = vaddq_f64(u1, u3);
                sub = vaddvq_f64(t0) + vaddvq_f64(t1);
                acc += sub * (double)K3_E8M0[sb];
                continue;
            }
#endif

            /* Four double lanes partitioned by i%4, reduced as (s0+s1)+(s2+s3), in
             * the scalar path, so on machines without AVX2 the reduction is the
             * same on every compiler. The split is written out rather than left to
             * the compiler because a sequential floating-point reduction may not be
             * reassociated without -ffast-math, which this build does not set:
             * expressed as one serial accumulator, the hottest loop in the engine
             * compiles to scalar adds no matter what the surrounding code looks
             * like.
             *
             * The lane split changes the summation order. See the accuracy contract
             * on the function above for why that is bounded at ~1e-16 relative. */
            /* The AVX2 grouped path below uses four __m256d accumulators over each
             * 16-element chunk, so its intra-lane summation order differs from the
             * scalar path; see the NIBBLE DECODE comment below for the accuracy
             * contract. The group is short, so four accumulators suffice to break
             * the add-latency chain. On aarch64 without AVX2, the NEON path above
             * already took the fast n==32 case, so the wf-based path below is the
             * general-group fallback with its own two-accumulator NEON loop. */
            int i = 0;
#if defined(__AVX2__)
            {
                /* NIBBLE DECODE IN REGISTER. The expand-to-wf[64]-then-reload form this
                 * replaces cost more than the arithmetic it fed: per 32-element group it
                 * ran 16 scalar table lookups and 32 four-byte stores, then reloaded all
                 * 32 floats one vector at a time, and the reload of a just-written stack
                 * slot is a store-forwarding stall on every group.
                 *
                 * E2M1 is small enough to decode with a shuffle instead of a table. The
                 * eight magnitudes {0,.5,1,1.5,2,3,4,6} are indexed by the low three bits
                 * of the code, which is exactly _mm256_permutevar8x32_ps of a register
                 * constant, and bit 3 is the sign, which is that bit moved to 31 and
                 * XORed in. Code 8 gives 0.0f ^ 0x80000000 = -0.0f, which is what
                 * K3_E2M1[8] holds, so the negative zero survives.
                 *
                 * ACCURACY CONTRACT (test_expert.c:219) is maxrel < 1e-6 against
                 * dequant-then-matmul, NOT bit-identity. This grouped path is the
                 * FALLBACK for group not a multiple of 16 (a failed xd hoist aborts
                 * above rather than landing here); on the normal K3 shape the flat row
                 * path above is taken instead. It keeps four independent
                 * accumulators (v0..v3) to break the FMA latency chain, so its
                 * intra-lane accumulation order differs from the scalar path
                 * below; the difference is a few ulps of double,
                 * orders of magnitude inside the 1e-6 gate. The bench FNV1a of the
                 * mxfp4 output differs from the pre-optimisation value -- that hash
                 * is a determinism check, not a correctness oracle. */
                const __m256  LUT = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f,
                                                   2.0f, 3.0f, 4.0f, 6.0f);
                const __m128i m0f = _mm_set1_epi8(0x0F);
                const __m256i m07 = _mm256_set1_epi32(7);
                const __m256i m08 = _mm256_set1_epi32(8);
                /* 16-element iteration: ONE 8-byte load yields all 16 nibbles, decoded
                 * into c0/c1 by unpacking the low/high nibble masks. Each block feeds its
                 * own accumulator (v0..v3), so per iteration every accumulator chain is a
                 * single FMA -- four independent depth-1 chains hide both the decode
                 * latency and the FMA latency. Group 32 collapses to two iterations. The
                 * scalar tail handles 8-15 and 0-7 remainders. */
                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
                __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
                for (; i + 15 < n; i += 16) {
                    const __m128i b  = _mm_loadl_epi64((const __m128i *)(pb + (i >> 1)));
                    const __m128i lo = _mm_and_si128(b, m0f);
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                    /* unpacklo interleaves the low 8 bytes of lo and hi, which together
                     * cover all 16 elements in order: [e0,e1,e2,...,e15]. Split that 16
                     * bytes into the low 8 (elems 0-7) and high 8 (elems 8-15). */
                    const __m128i u16 = _mm_unpacklo_epi8(lo, hi);
                    const __m256i c0 = _mm256_cvtepu8_epi32(u16);
                    const __m256i c1 = _mm256_cvtepu8_epi32(_mm_srli_si128(u16, 8));
                    const __m256  w0 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c0, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c0, m08), 28)));
                    const __m256  w1 = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c1, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c1, m08), 28)));
                    v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(w0)),
                                         _mm256_loadu_pd(xdg + i), v0);
                    v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)),
                                         _mm256_loadu_pd(xdg + i + 4), v1);
                    v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(w1)),
                                         _mm256_loadu_pd(xdg + i + 8), v2);
                    v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)),
                                         _mm256_loadu_pd(xdg + i + 12), v3);
                }
                /* 8-element remainder: same AVX2 decode + FMA as before. Runs at most
                 * once, for the 8-15 remainder after the 16-element loop. */
                for (; i + 7 < n; i += 8) {
                    int32_t four;
                    memcpy(&four, pb + (i >> 1), 4);
                    const __m128i b  = _mm_cvtsi32_si128(four);
                    const __m128i lo = _mm_and_si128(b, m0f);
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                    const __m256i c  = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));
                    const __m256  wv = _mm256_xor_ps(
                        _mm256_permutevar8x32_ps(LUT, _mm256_and_si256(c, m07)),
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(_mm256_and_si256(c, m08), 28)));
                    v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(wv)),
                                         _mm256_loadu_pd(xdg + i), v0);
                    v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(wv, 1)),
                                         _mm256_loadu_pd(xdg + i + 4), v1);
                }
                double a[4];
                _mm256_storeu_pd(a, _mm256_add_pd(_mm256_add_pd(v0, v2),
                                                  _mm256_add_pd(v1, v3)));
                sub = (a[0] + a[1]) + (a[2] + a[3]);
            }
            /* Sub-8 remainder, decoded one nibble at a time. K3 never reaches it (group
             * 32 divides evenly) but a short final group must still be correct. */
            for (; i < n; i++) {
                const unsigned char by = pb[i >> 1];
                sub = fma((double)K3_E2M1[(i & 1) ? (by >> 4) : (by & 0x0F)],
                          xdg[i], sub);
            }
#else
            {
                /* Expand the group to floats first, then take a plain dot product. The
                 * split exists so the second loop can vectorise, which it cannot do
                 * while a table lookup sits in the middle of the accumulation. */
                float wf[64];                     /* group is 32 for K3; 64 is headroom */
                const int half = n >> 1;
                for (int j = 0; j < half; j++) {
                    const float *pv = K3_E2M1_PAIR[pb[j]];
                    wf[2 * j]     = pv[0];
                    wf[2 * j + 1] = pv[1];
                }
                if (n & 1) wf[n - 1] = K3_E2M1_PAIR[pb[half]][0];

#if defined(__ARM_NEON) && defined(__aarch64__)
                {
                    /* uk holds the scalar path's {s[2k], s[2k+1]}: element i in
                     * accumulator i%8, vfmaq_f64 the same IEEE fma() per lane. u0+u2 is
                     * lanewise {s0+s4, s1+s5} = {b0,b1} and u1+u3 is {b2,b3}, so the
                     * reduction below is the scalar (b0+b1)+(b2+b3) tree exactly. */
                    float64x2_t u0 = vdupq_n_f64(0.0), u1 = vdupq_n_f64(0.0);
                    float64x2_t u2 = vdupq_n_f64(0.0), u3 = vdupq_n_f64(0.0);
                    for (; i + 7 < n; i += 8) {
                        const float32x4_t wv0 = vld1q_f32(wf + i);
                        const float32x4_t wv1 = vld1q_f32(wf + i + 4);
                        u0 = vfmaq_f64(u0, vcvt_f64_f32(vget_low_f32(wv0)),
                                           vld1q_f64(xdg + i));
                        u1 = vfmaq_f64(u1, vcvt_high_f64_f32(wv0), vld1q_f64(xdg + i + 2));
                        u2 = vfmaq_f64(u2, vcvt_f64_f32(vget_low_f32(wv1)),
                                           vld1q_f64(xdg + i + 4));
                        u3 = vfmaq_f64(u3, vcvt_high_f64_f32(wv1), vld1q_f64(xdg + i + 6));
                    }
                    const float64x2_t t0 = vaddq_f64(u0, u2);
                    const float64x2_t t1 = vaddq_f64(u1, u3);
                    sub = vaddvq_f64(t0) + vaddvq_f64(t1);
                }
#else
                {
                    double s[8] = {0};
                    for (; i + 7 < n; i += 8)
                        for (int l = 0; l < 8; l++)
                            s[l] = fma((double)wf[i + l], xdg[i + l], s[l]);
                    double b0 = s[0] + s[4], b1 = s[1] + s[5];
                    double b2 = s[2] + s[6], b3 = s[3] + s[7];
                    sub = (b0 + b1) + (b2 + b3);
                }
#endif
                for (; i < n; i++) sub = fma((double)wf[i], xdg[i], sub);
            }
#endif
            acc += sub * (double)K3_E8M0[sb];
        }
        y[r] = (float)acc;
    }

    free(xd);                                     /* free(NULL) is a no-op */
}

void k3_mxfp4_dequant(float *out, const unsigned char *packed,
                      const unsigned char *scales, int rows, int pcols, int group)
{
    const int width = pcols * 2;                  /* logical elements per row */
    const int ngrp  = (width + group - 1) / group;

    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        float *orow = out + (size_t)r * width;

        for (int g = 0; g < ngrp; g++) {
            /* E8M0: a bare biased exponent. 255 is NaN by spec; map it to zero so one
             * bad byte cannot poison the row. ldexpf is exact for powers of two. */
            const unsigned char sb = sr[g];
            const float mult = (sb == 255) ? 0.0f : ldexpf(1.0f, (int)sb - 127);

            const int lo = g * group;
            int hi = lo + group;
            if (hi > width) hi = width;

            for (int i = lo; i < hi; i++) {
                const unsigned char byte = pr[i >> 1];
                /* low nibble = EVEN element. Reversing this gives right values in
                 * wrong places, which every statistical check would pass. */
                const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                orow[i] = K3_E2M1[nib] * mult;
            }
        }
    }
}
