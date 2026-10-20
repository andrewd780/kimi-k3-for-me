/* bench_token.c - one decode token's compute at the released geometry, by layer kind.
 *
 *   bench_token [--C 256,1024,4096,16384] [--calls 3] [--threads N] [--label s]
 *
 * Synthetic weights at Kimi K3's real shapes (hidden 7168, 96 KDA heads of 128, MLA 96
 * heads, 896 experts top-16 in a 3584 latent, 2 shared experts, dense layer 0 at 33792,
 * vocab 163840), bf16 where the checkpoint is bf16 and MXFP4 for routed experts; no
 * checkpoint is read. A token of K3 runs layer 0 (KDA attention + the dense MLP), 68
 * more KDA layers and 24 MLA layers (each with a LatentMoE), then the head. This times
 * ONE layer of each kind through the engine's own k3_decoder_layer_inc at T = 1, and
 * the head (final norm + lm_head), and composes the token:
 *
 *     token(C) = L0 + 68 x KDA layer + 24 x MLA layer(C) + head
 *
 * for every C in the list (one cache of the largest C serves them all: a call at C reads
 * its first C positions). It also times the pieces inside a layer (KDA attention, MLA
 * attention at each C, MoE, the dense MLP) and prints the bytes each reads and the rate
 * that implies, so a module running far below the machine's read bandwidth stands out.
 *
 * WHAT IT IS NOT. A composed figure from one layer of each kind, each re-run on the same
 * weights. Weights larger than the last-level cache stream from DRAM on every call, as
 * a real decode's do, but the real model's weights do not fit in this box's memory, and
 * a real run streams most of them from disk: on such a host bytes per token, not this
 * compute, set the speed. Routed experts come from an in-memory bank of 32 through the
 * same K3ExpertSrc interface the streamed cache implements, so their arithmetic is the
 * engine's and their bytes are DRAM reads, never disk reads.
 *
 * Every timed call is preceded by a quiet wait (85% idle, up to 120 s); every call's time
 * is printed. The output hash of each layer kind is printed so two builds can be held to
 * the same bytes.
 */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3.h"

#define MAX_CALLS 32
#define BANK 32
#define MAX_C 8

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void sleep_s(double s)
{
    struct timespec t;
    t.tv_sec = (time_t)s;
    t.tv_nsec = (long)((s - (double)t.tv_sec) * 1e9);
    nanosleep(&t, NULL);
}

static int read_cpu_stat(unsigned long long *idle, unsigned long long *total)
{
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return -1;
    unsigned long long v[10] = {0};
    const int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0],
                         &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
    fclose(f);
    if (n < 4) return -1;
    *idle = v[3] + v[4];
    *total = 0;
    for (int i = 0; i < 8; i++) *total += v[i];
    return 0;
}

static double wait_quiet(double need, double max_wait)
{
    const double t0 = now_s();
    for (;;) {
        unsigned long long i0, a0, i1, a1;
        if (read_cpu_stat(&i0, &a0)) return -1.0;
        sleep_s(0.25);
        if (read_cpu_stat(&i1, &a1) || a1 <= a0) return -1.0;
        const double idle = (double)(i1 - i0) / (double)(a1 - a0);
        if (idle >= need || now_s() - t0 >= max_wait) return idle;
        sleep_s(0.75);
    }
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%.1f MB)\n", (double)n / 1e6); exit(1); }
    return p;
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}

/* finite normal floats of magnitude 2^-9 .. 2^-2, either sign */
static float *f32s(size_t n, float scale)
{
    float *p = (float *)xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        p[i] = scale * (float)(((double)(rnd() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0);
    return p;
}

/* bf16 with exponents 118..125 (about 2^-9 .. 2^-2), random sign and mantissa */
static uint16_t *bf16s(size_t n)
{
    uint16_t *p = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i += 4) {
        const uint64_t r = rnd();
        for (int k = 0; k < 4 && i + k < n; k++) {
            const unsigned b = (unsigned)(r >> (16 * k)) & 0xFFFFu;
            p[i + k] = (uint16_t)((b & 0x8000u) | ((118u + ((b >> 7) & 7u)) << 7) | (b & 0x7Fu));
        }
    }
    return p;
}

static float *ones(size_t n, float v)
{
    float *p = (float *)xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = v;
    return p;
}

/* ---- an in-memory expert source: BANK packed experts, expert e served from e % BANK */
typedef struct { unsigned char *p[3], *s[3]; } BankExpert;
static BankExpert g_bank[BANK];

static int bank_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    (void)self; (void)layer;
    const BankExpert *b = &g_bank[expert % BANK];
    out->p1 = b->p[0]; out->s1 = b->s[0];
    out->p3 = b->p[1]; out->s3 = b->s[1];
    out->p2 = b->p[2]; out->s2 = b->s[2];
    return 0;
}

static uint64_t fnv(const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

typedef struct {
    const char *name;
    double t[MAX_CALLS];
    double bytes;
    uint64_t hash;
} Timing;

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n)
{
    double t[MAX_CALLS];
    memcpy(t, v, (size_t)n * sizeof(double));
    qsort(t, (size_t)n, sizeof(double), cmp_d);
    return n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

int main(int argc, char **argv)
{
    int Cs[MAX_C] = {256, 1024, 4096, 16384}, nC = 4, calls = 3, threads = 0;
    const char *label = "";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
        if (!strcmp(a, "--C")) {
            nC = 0;
            for (const char *q = v; *q && nC < MAX_C; ) {
                Cs[nC++] = atoi(q);
                const char *cm = strchr(q, ',');
                if (!cm) break;
                q = cm + 1;
            }
        }
        else if (!strcmp(a, "--calls")) calls = atoi(v);
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--label")) label = v;
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
        i++;
    }
    int Cmax = 0;
    for (int i = 0; i < nC; i++) {
        if (Cs[i] < 0) nC = 0;
        if (Cs[i] > Cmax) Cmax = Cs[i];
    }
    if (nC < 1 || calls < 1 || calls > MAX_CALLS) {
        fprintf(stderr, "usage: bench_token [--C n,n,...] [--calls 1..%d] [--threads n] [--label s]\n",
                MAX_CALLS);
        return 2;
    }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif

    /* the released configuration (tests/unit/scale_test.c) */
    K3Cfg c; memset(&c, 0, sizeof c);
    c.hidden = 7168;  c.n_layers = 93;   c.vocab = 163840; c.rms_eps = 1e-5f;
    c.kda_heads = 96; c.kda_head_dim = 128; c.conv_k = 4;  c.gate_lb = -5.0f;
    c.n_heads = 96;   c.q_lora = 1536;   c.kv_lora = 512;
    c.qk_nope = 128;  c.qk_rope = 64;    c.v_head = 128;   c.mla_out_gate = 1;
    c.n_experts = 896; c.topk = 16;      c.n_shared = 2;
    c.latent = 3584;  c.moe_inter = 3072; c.routed_scale = 1.0f;
    c.moe_renorm = 1; c.latent_norm = 1;
    c.first_dense = 1; c.dense_inter = 33792;
    c.attn_res_block = 12;
    c.situ_b1 = 4.0f; c.situ_b2 = 25.0f;
    static int fa[24];
    int nfa = 0;
    for (int i = 4; i <= 93; i += 4) fa[nfa++] = i;
    fa[nfa++] = 93;
    c.n_full_attn = nfa; c.full_attn = fa;

    const size_t E = (size_t)c.hidden, P = (size_t)c.kda_heads * c.kda_head_dim;
    const size_t D = (size_t)c.kda_head_dim, H = (size_t)c.n_heads;
    const size_t L = (size_t)c.latent, I = (size_t)c.moe_inter, SI = I * (size_t)c.n_shared;
    const size_t qh = (size_t)(c.qk_nope + c.qk_rope), kvd = (size_t)(c.qk_nope + c.v_head);
    const size_t kvw = (size_t)(c.kv_lora + c.qk_rope), HV = H * (size_t)c.v_head;

    /* which layer indices are which kind, and representative block counts */
    int iK = -1, iM = -1;
    for (int l = 13; l < c.n_layers && (iK < 0 || iM < 0); l++) {
        if (k3_is_mla(&c, l) && iM < 0) iM = l;
        if (!k3_is_mla(&c, l) && iK < 0) iK = l;
    }

    /* ---- weights ---- */
    K3KdaW kda; memset(&kda, 0, sizeof kda);
    kda.q = bf16s(P * E); kda.k = bf16s(P * E); kda.v = bf16s(P * E); kda.g = bf16s(P * E);
    kda.o = bf16s(E * P);
    kda.q_conv = f32s(P * c.conv_k, 0.4f); kda.k_conv = f32s(P * c.conv_k, 0.4f);
    kda.v_conv = f32s(P * c.conv_k, 0.4f);
    kda.f_a = bf16s(D * E); kda.f_b = bf16s(P * D); kda.b = bf16s((size_t)c.kda_heads * E);
    kda.A_log = f32s((size_t)c.kda_heads, 1.0f); kda.dt_bias = f32s(P, 0.5f);
    kda.o_norm = ones(D, 1.0f);
    kda.wdt = K3_WBF16;

    K3MlaW mla; memset(&mla, 0, sizeof mla);
    mla.q_a = bf16s((size_t)c.q_lora * E); mla.q_b = bf16s(H * qh * (size_t)c.q_lora);
    mla.kv_a = bf16s(kvw * E); mla.kv_b = bf16s(H * kvd * (size_t)c.kv_lora);
    mla.o = bf16s(E * HV); mla.g = bf16s(HV * E);
    mla.q_a_norm = ones((size_t)c.q_lora, 1.0f); mla.kv_a_norm = ones((size_t)c.kv_lora, 1.0f);
    mla.wdt = K3_WBF16;

    for (int b = 0; b < BANK; b++)
        for (int m = 0; m < 3; m++) {
            const size_t rows = m == 2 ? L : I, cols = m == 2 ? I : L;
            g_bank[b].p[m] = (unsigned char *)xmalloc(rows * cols / 2);
            g_bank[b].s[m] = (unsigned char *)xmalloc(rows * cols / K3_MXFP4_GROUP);
            for (size_t i = 0; i < rows * cols / 2; i++) g_bank[b].p[m][i] = (unsigned char)rnd();
            for (size_t i = 0; i < rows * cols / K3_MXFP4_GROUP; i++)
                g_bank[b].s[m][i] = (unsigned char)(118 + rnd() % 6);
        }
    K3ExpertSrc src; memset(&src, 0, sizeof src);
    src.get = bank_get;
    K3MoeW moe; memset(&moe, 0, sizeof moe);
    moe.gate = f32s((size_t)c.n_experts * E, 0.02f);
    float *bias = (float *)xmalloc((size_t)c.n_experts * sizeof(float));
    for (int e = 0; e < c.n_experts; e++) bias[e] = -0.15f + 0.3f * (float)e / (float)(c.n_experts - 1);
    moe.bias = bias;
    moe.latent_norm = ones(L, 1.0f);
    moe.down = bf16s(L * E); moe.up = bf16s(E * L);
    moe.sh1 = bf16s(SI * E); moe.sh3 = bf16s(SI * E); moe.sh2 = bf16s(E * SI);
    moe.wdt = K3_WBF16;
    moe.src = &src;

    uint16_t *dg = bf16s((size_t)c.dense_inter * E), *du = bf16s((size_t)c.dense_inter * E);
    uint16_t *dd = bf16s(E * (size_t)c.dense_inter);
    uint16_t *head = bf16s((size_t)c.vocab * E);
    float *fnorm = ones(E, 1.0f);

    K3LayerW lay0, layK, layM;
    memset(&lay0, 0, sizeof lay0);
    lay0.in_norm = ones(E, 1.0f); lay0.post_norm = ones(E, 1.0f);
    lay0.attn_res_norm = ones(E, 1.0f); lay0.attn_res_proj = f32s(E, 0.05f);
    lay0.mlp_res_norm = ones(E, 1.0f); lay0.mlp_res_proj = f32s(E, 0.05f);
    lay0.kda = &kda;
    lay0.dense_gate = dg; lay0.dense_up = du; lay0.dense_down = dd; lay0.wdt = K3_WBF16;
    layK = lay0; layK.dense_gate = layK.dense_up = layK.dense_down = NULL; layK.moe = &moe;
    layM = layK; layM.kda = NULL; layM.mla = &mla;

    /* ---- state, caches, buffers ---- */
    const int maxb = k3_attn_res_stride(&c);
    const int cap = Cmax + 1;
    float *kvc = f32s((size_t)cap * H * kvd, 1.0f), *ropec = f32s((size_t)cap * c.qk_rope, 1.0f);
    float *kst = f32s(k3_kda_state_floats(&c), 0.01f);
    float *h = f32s(E, 1.0f), *h0 = (float *)xmalloc(E * sizeof(float));
    memcpy(h0, h, E * sizeof(float));
    float *br = f32s((size_t)maxb * E, 1.0f);
    size_t ns = k3_layer_scratch(&c, 1);
    const size_t nm = k3_mla_scratch_cached(&c, 1, cap, 1, 0);
    if (nm > ns) ns = nm;
    float *sc = (float *)xmalloc(ns * sizeof(float));
    float *lg = (float *)xmalloc((size_t)c.vocab * sizeof(float));
    float *xin = f32s(E, 1.0f), *yo = (float *)xmalloc(E * sizeof(float));
    int idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];

    /* bytes each piece reads per token (weights, plus the KV rows for MLA at its C) */
    const double b_kda = 2.0 * (5.0 * P * E + D * E + P * D + (double)c.kda_heads * E);
    const double b_mla_w = 2.0 * ((double)c.q_lora * E + H * qh * c.q_lora + kvw * E
                                  + H * kvd * c.kv_lora + 2.0 * E * HV);
    const double b_exp = (double)c.topk * 3.0 * I * L * (0.5 + 1.0 / K3_MXFP4_GROUP);
    const double b_moe = 4.0 * c.n_experts * E + b_exp + 2.0 * (2.0 * L * E + 3.0 * SI * E);
    const double b_dense = 2.0 * 3.0 * c.dense_inter * E;
    const double b_head = 2.0 * (double)c.vocab * E;

    /* pieces 0..5 do not depend on C; then mla_attention and layer_mla for each C */
    enum { N_FIXED = 6 };
    const int N_T = N_FIXED + 2 * nC;
    Timing tm[N_FIXED + 2 * MAX_C];
    memset(tm, 0, sizeof tm);
    const char *fixed_names[N_FIXED] = {"kda_attention", "moe", "dense_mlp", "layer0", "layer_kda", "head"};
    const double fixed_bytes[N_FIXED] = {b_kda, b_moe, b_dense, b_kda + b_dense, b_kda + b_moe, b_head};
    static char names[2 * MAX_C][32];
    for (int m = 0; m < N_FIXED; m++) { tm[m].name = fixed_names[m]; tm[m].bytes = fixed_bytes[m]; }
    for (int j = 0; j < nC; j++) {
        const double kvb = 4.0 * ((double)(Cs[j] + 1) * (H * kvd + c.qk_rope));
        snprintf(names[2 * j], sizeof names[0], "mla_attention_C%d", Cs[j]);
        snprintf(names[2 * j + 1], sizeof names[0], "layer_mla_C%d", Cs[j]);
        tm[N_FIXED + 2 * j].name = names[2 * j];         tm[N_FIXED + 2 * j].bytes = b_mla_w + kvb;
        tm[N_FIXED + 2 * j + 1].name = names[2 * j + 1]; tm[N_FIXED + 2 * j + 1].bytes = b_mla_w + kvb + b_moe;
    }

    for (int k = -1; k < calls; k++) {          /* k = -1: untimed warm-up */
        for (int m = 0; m < N_T; m++) {
            if (k >= 0) wait_quiet(0.85, 120.0);
            int nb = 0;
            const int Cm = m >= N_FIXED ? Cs[(m - N_FIXED) / 2] : 0;
            double t0 = now_s();
            const float *res = yo;
            size_t rn = E;
            switch (m) {
            case 0: k3_kda_layer(yo, xin, &kda, &c, 1, kst, sc); break;
            case 1: k3_moe(yo, xin, &moe, &c, 1, idx, wt, sc); break;
            case 2: {
                float *gu = sc, *g2 = sc + c.dense_inter;
                k3_mmw(gu, xin, dg, K3_WBF16, (int)E, c.dense_inter);
                k3_mmw(g2, xin, du, K3_WBF16, (int)E, c.dense_inter);
                k3_situ_glu(gu, gu, c.dense_inter, c.situ_b1, c.situ_b2);
                k3_mmw(yo, gu, dd, K3_WBF16, c.dense_inter, (int)E);
                break;
            }
            case 3: memcpy(h, h0, E * sizeof(float)); nb = 0; t0 = now_s();
                    k3_decoder_layer_inc(h, br, &nb, &lay0, &c, 0, 1, kst, sc, NULL, NULL, 0, 0, 0);
                    res = h; break;
            case 4: memcpy(h, h0, E * sizeof(float)); nb = 4; t0 = now_s();
                    k3_decoder_layer_inc(h, br, &nb, &layK, &c, iK, 1, kst, sc, NULL, NULL, 0, 0, 0);
                    res = h; break;
            case 5: k3_rmsnorm(sc, xin, fnorm, (int)E, c.rms_eps);
                    k3_mmw(lg, sc, head, K3_WBF16, (int)E, c.vocab);
                    res = lg; rn = (size_t)c.vocab; break;
            default:
                if ((m - N_FIXED) % 2 == 0) {
                    k3_mla_cached(yo, xin, &mla, &c, 1, sc, kvc, ropec, Cm, cap, 0);
                } else {
                    memcpy(h, h0, E * sizeof(float)); nb = 4; t0 = now_s();
                    k3_decoder_layer_inc(h, br, &nb, &layM, &c, iM, 1, kst, sc, kvc, ropec, Cm, cap, 0);
                    res = h;
                }
            }
            const double dt = now_s() - t0;
            if (k >= 0) tm[m].t[k] = dt;
            tm[m].hash = fnv(res, rn * sizeof(float));
        }
    }

    double med[N_FIXED + 2 * MAX_C];
    printf("{\"bench\":\"token\",\"label\":\"%s\",\"threads\":%d,\"pieces\":{", label, threads);
    for (int m = 0; m < N_T; m++) {
        med[m] = median(tm[m].t, calls);
        printf("%s\"%s\":{\"median_s\":%.6f,\"bytes\":%.0f,\"gbps\":%.3f,\"hash\":\"%016llx\",\"times_s\":[",
               m ? "," : "", tm[m].name, med[m], tm[m].bytes, tm[m].bytes / med[m] / 1e9,
               (unsigned long long)tm[m].hash);
        for (int k = 0; k < calls; k++) printf("%s%.6f", k ? "," : "", tm[m].t[k]);
        printf("]}");
    }
    /* token(C) = layer 0 + 68 KDA layers + 24 MLA layers at C + head, from medians */
    printf("},\"token_s\":{");
    for (int j = 0; j < nC; j++)
        printf("%s\"%d\":%.4f", j ? "," : "", Cs[j],
               med[3] + 68.0 * med[4] + 24.0 * med[N_FIXED + 2 * j + 1] + med[5]);
    printf("}}\n");
    return 0;
}
