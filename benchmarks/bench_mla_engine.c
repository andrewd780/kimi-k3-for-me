/* bench_mla_engine.c - wall time of ONE call of the engine's own k3_mla_cached.
 *
 *   bench_mla_engine --C <cached> --T <new> [--threads N] [--calls N] [--latent]
 *                    [--label s] [--quiet-wait s]
 *
 * One MLA layer at the released geometry (hidden 7168, 96 heads, qk_nope 128, qk_rope 64,
 * v_head 128, kv_lora 512, q_lora 1536, output gate on), synthetic bf16 weights and a
 * synthetic cache of C positions; no checkpoint is read. The timed region is the whole
 * call the decoder makes per MLA layer: projections, the append of T new positions,
 * attention, gate and o_proj. bench_mla times copies of the attention step side by side;
 * this times the engine itself, so two builds of the engine can be compared with the same
 * harness, and prints a hash of what each call produced so that the comparison also
 * shows whether they computed the same bytes.
 *
 * The inputs depend only on (C, T): every value is drawn from fixed seeds, so two builds
 * given the same arguments see the same bytes, and the hashes (FNV-1a 64 over the output
 * rows, then over the appended cache rows and rope slots) must agree between builds that
 * compute the same floats. Every call writes the same positions C .. C+T-1 from the same
 * inputs, so repeating it is idempotent and every call is hashed.
 *
 * Contention: before each call the harness waits (up to --quiet-wait seconds, default
 * 120) for the machine to be at least 85% idle over 250 ms, and records the share of the
 * call's wall time with CPU pressure (PSI "some", /proc/pressure/cpu; -1 where absent).
 * Nothing is discarded: every call's time, idle fraction and stall share are printed, as
 * one JSON line on stdout, after one untimed warm-up call.
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

#define MAX_CALLS 64

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

/* Idle fraction of the whole machine over `s` seconds, or -1 where unknown. */
static double idle_fraction(double s)
{
    unsigned long long i0, t0, i1, t1;
    if (read_cpu_stat(&i0, &t0)) return -1.0;
    sleep_s(s);
    if (read_cpu_stat(&i1, &t1) || t1 <= t0) return -1.0;
    return (double)(i1 - i0) / (double)(t1 - t0);
}

static double wait_quiet(double need, double max_wait)
{
    const double t0 = now_s();
    double idle = idle_fraction(0.25);
    while (idle >= 0.0 && idle < need && now_s() - t0 < max_wait) {
        sleep_s(0.75);
        idle = idle_fraction(0.25);
    }
    return idle;
}

/* Cumulative microseconds of CPU "some" pressure, or -1 where unavailable. */
static double psi_some_us(void)
{
    FILE *f = fopen("/proc/pressure/cpu", "r");
    if (!f) return -1.0;
    char line[256];
    double us = -1.0;
    if (fgets(line, sizeof line, f)) {
        const char *p = strstr(line, "total=");
        if (p) us = atof(p + 6);
    }
    fclose(f);
    return us;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%.1f MB)\n", (double)n / 1e6); exit(1); }
    return p;
}

/* Cheap uniform fill, as bench_mla's (the same xorshift64 and seeding as
 * benchmarks/mla_variants.h): values need only be finite, normal and plausibly sized;
 * Box-Muller over the 1.6 GB cache at C = 16,384 would outlast the calls. */
static void fill_fast(float *p, size_t n, float amp, uint64_t seed)
{
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
    if (!s) s = 1;
    for (int i = 0; i < 8; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; }
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const double u = ((double)(s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        p[i] = amp * (float)(2.0 * u - 1.0);
    }
}

/* float -> bf16, round to nearest even; the inputs are finite. */
static uint16_t to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static uint16_t *bf16_matrix(size_t n, float amp, uint64_t seed)
{
    float *f = (float *)xmalloc(n * sizeof(float));
    fill_fast(f, n, amp, seed);
    uint16_t *b = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) b[i] = to_bf16(f[i]);
    free(f);
    return b;
}

static uint64_t fnv1a(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(int argc, char **argv)
{
    int C = -1, T = 1, threads = 0, calls = 3, latent = 0;
    double quiet_wait = 120.0;
    const char *label = "";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--latent")) { latent = 1; continue; }
        if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
        if (!strcmp(a, "--C")) C = atoi(v);
        else if (!strcmp(a, "--T")) T = atoi(v);
        else if (!strcmp(a, "--threads")) threads = atoi(v);
        else if (!strcmp(a, "--calls")) calls = atoi(v);
        else if (!strcmp(a, "--label")) label = v;
        else if (!strcmp(a, "--quiet-wait")) quiet_wait = atof(v);
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
        i++;
    }
    if (C < 0 || T < 1 || calls < 1 || calls > MAX_CALLS) {
        fprintf(stderr, "usage: bench_mla_engine --C <cached> --T <new> [--threads N] "
                        "[--calls 1..%d] [--latent] [--label s] [--quiet-wait s]\n", MAX_CALLS);
        return 2;
    }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif

    K3Cfg c;                         /* the released attention geometry */
    memset(&c, 0, sizeof c);
    c.hidden = 7168; c.rms_eps = 1e-5f;
    c.n_heads = 96; c.q_lora = 1536; c.kv_lora = 512;
    c.qk_nope = 128; c.qk_rope = 64; c.v_head = 128; c.mla_out_gate = 1;
    const size_t E = (size_t)c.hidden, H = (size_t)c.n_heads;
    const size_t qh = (size_t)(c.qk_nope + c.qk_rope), kvd = (size_t)(c.qk_nope + c.v_head);
    const size_t kvw = (size_t)(c.kv_lora + c.qk_rope), HV = H * (size_t)c.v_head;
    /* Uniform on [-a, a] has sd a / sqrt(3): a = 0.035 gives about bench_mla's 0.02. */
    const float wa = 0.035f;
    K3MlaW w;
    memset(&w, 0, sizeof w);
    w.q_a  = bf16_matrix((size_t)c.q_lora * E, wa, 11);
    w.q_b  = bf16_matrix(H * qh * (size_t)c.q_lora, wa, 12);
    w.kv_a = bf16_matrix(kvw * E, wa, 13);
    w.kv_b = bf16_matrix(H * kvd * (size_t)c.kv_lora, wa, 14);
    w.o    = bf16_matrix(E * HV, wa, 15);
    w.g    = bf16_matrix(HV * E, wa, 16);
    w.wdt  = K3_WBF16;
    float *norm = (float *)xmalloc(((size_t)c.q_lora + c.kv_lora) * sizeof(float));
    fill_fast(norm, (size_t)c.q_lora + c.kv_lora, 0.1f, 17);
    for (int i = 0; i < c.q_lora + c.kv_lora; i++) norm[i] += 1.0f;
    w.q_a_norm = norm;
    w.kv_a_norm = norm + c.q_lora;

    const int cap = C + T;
    const size_t row = latent ? (size_t)c.kv_lora : H * kvd;
    float *kvc   = (float *)xmalloc((size_t)cap * row * sizeof(float));
    float *ropec = (float *)xmalloc((size_t)cap * c.qk_rope * sizeof(float));
    /* cached rows of about the size the projections give them */
    fill_fast(kvc, (size_t)C * row, 1.0f, 21);
    fill_fast(ropec, (size_t)C * c.qk_rope, 1.0f, 22);
    float *x = (float *)xmalloc((size_t)T * E * sizeof(float));
    fill_fast(x, (size_t)T * E, 1.7f, 23);
    float *out = (float *)xmalloc((size_t)T * E * sizeof(float));
    const size_t ns = k3_mla_scratch_cached(&c, T, cap, 1, latent);
    float *scr = (float *)xmalloc(ns * sizeof(float));

    /* warm-up, untimed: page in the weights, cache and scratch */
    k3_mla_cached(out, x, &w, &c, T, scr, kvc, ropec, C, cap, latent);

    double t[MAX_CALLS], idle[MAX_CALLS], stall[MAX_CALLS];
    uint64_t hash[MAX_CALLS];
    for (int k = 0; k < calls; k++) {
        idle[k] = wait_quiet(0.85, quiet_wait);
        const double p0 = psi_some_us(), t0 = now_s();
        k3_mla_cached(out, x, &w, &c, T, scr, kvc, ropec, C, cap, latent);
        const double t1 = now_s(), p1 = psi_some_us();
        t[k] = t1 - t0;
        stall[k] = (p0 >= 0.0 && p1 >= 0.0) ? (p1 - p0) * 1e-6 / t[k] : -1.0;
        uint64_t h = 0xcbf29ce484222325ull;
        h = fnv1a(h, out, (size_t)T * E * sizeof(float));
        h = fnv1a(h, kvc + (size_t)C * row, (size_t)T * row * sizeof(float));
        h = fnv1a(h, ropec + (size_t)C * c.qk_rope, (size_t)T * c.qk_rope * sizeof(float));
        hash[k] = h;
    }

    printf("{\"bench\":\"mla_engine\",\"label\":\"%s\",\"C\":%d,\"T\":%d,\"threads\":%d,"
           "\"layout\":\"%s\",\"scratch_floats\":%zu,\"times_s\":[", label, C, T, threads,
           latent ? "latent" : "expanded", ns);
    for (int k = 0; k < calls; k++) printf("%s%.6f", k ? "," : "", t[k]);
    printf("],\"idle_before\":[");
    for (int k = 0; k < calls; k++) printf("%s%.3f", k ? "," : "", idle[k]);
    printf("],\"stall_share\":[");
    for (int k = 0; k < calls; k++) printf("%s%.4f", k ? "," : "", stall[k]);
    printf("],\"hash\":[");
    for (int k = 0; k < calls; k++) printf("%s\"%016llx\"", k ? "," : "", (unsigned long long)hash[k]);
    printf("]}\n");

    free(scr); free(out); free(x); free(kvc); free(ropec); free(norm);
    free((void *)w.q_a); free((void *)w.q_b); free((void *)w.kv_a); free((void *)w.kv_b);
    free((void *)w.o); free((void *)w.g);
    return 0;
}
