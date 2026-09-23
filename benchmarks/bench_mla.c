/* bench_mla.c - what each MLA cache variant costs, and what the inexact one changes.
 *
 * Two modes, both at the released attention geometry (96 heads, qk_nope 128, qk_rope
 * 64, v_head 128, kv_lora 512) with synthetic bf16 weights; no checkpoint is read.
 *
 *   bench_mla [time] [options]     wall time of ONE layer's attention step for T new
 *                                  tokens after C cached ones, per variant: E, E+, L0,
 *                                  L1, A of benchmarks/mla_variants.h. The timed region
 *                                  is mla_attend: appending the new tokens (a kv_b matmul
 *                                  each for the expanded layout, a copy for the latent
 *                                  one) and the attention itself. The projections, gate
 *                                  and o_proj are the same calls in every variant and are
 *                                  timed once, separately (--common).
 *   bench_mla numerics [options]   the absorbed variant A against E on the same inputs,
 *                                  and both against a double-precision reference:
 *                                  attention output before and after o_proj, raw score
 *                                  differences, and argmax changes over 10,000 softmax
 *                                  rows. Hidden states ~ N(0,1) go through the engine's
 *                                  own projections and rmsnorms with weights ~ N(0,0.02)
 *                                  rounded to bf16, exactly as k3_mla_cached runs them.
 *   bench_mla counts [options]     no timing at all: per variant, the kv_b applications
 *                                  one call makes (COUNTED, by running it), and the
 *                                  multiply-adds and bytes that follow at K3 geometry.
 *                                  The prefill shape C=0, T=256 is the point of it: L0's
 *                                  count is quadratic in T there, L1's linear.
 *
 * WHAT IS AND IS NOT MEASURED
 *   Latent-path time is dominated by kv_b applications, whose count per call is exact
 *   (mla_rebuilds) and whose unit cost is measured here first. A configuration whose
 *   single run would exceed --max-run-s is not run; it is reported as PROJECTED from
 *   that unit cost, scaled by the measured-to-modelled ratio of the runs that did
 *   complete, and marked so in every output. A configuration whose buffers exceed
 *   --mem-mb is not run either (the expanded cache at 65,536 positions is 6.4 GB for
 *   one layer). Multiply-add counts are printed next to every time: they do not move
 *   with machine load, which on a shared VM the times certainly do.
 *
 * Every timed run of a (C, T) configuration interleaves the variants, so background load
 * lands on all of them rather than on whichever ran last. Medians of --runs runs (>= 5).
 */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3.h"
#include "mla_variants.h"

#define MAX_RUNS 64
#define MAX_LIST 16
/* Model-level context the note uses: 24 MLA layers, and the non-attention per-token
 * work of the whole model, ~103 GMAC (trunk 54.4 + routed experts 48.6). NONATTN_SECONDS
 * is an ASSUMED round figure for how long that work takes per token, there only to give
 * the per-token columns a scale: no full model has been timed on the VM these numbers
 * come from, so it is not a measurement. */
#define N_MLA_LAYERS    24
#define NONATTN_GMAC    103.0
#define NONATTN_SECONDS 16.0

/* A double as a JSON value: the number in `fmt`, or null when it is not finite. printf
 * writes nan and inf for a projection that could not be made, a ratio with nothing to
 * divide or a gap with no second score, and strict JSON parsers reject both. Returns
 * one of JN_BUFS rotating buffers, so one fprintf may use up to that many. */
#define JN_BUFS 16
static const char *jnum(double x, const char *fmt)
{
    static char buf[JN_BUFS][40];
    static int next = 0;
    char *b = buf[next];
    next = (next + 1) % JN_BUFS;
    if (isfinite(x)) snprintf(b, sizeof buf[0], fmt, x);
    else snprintf(b, sizeof buf[0], "null");
    return b;
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* CPU time of the whole process, all threads. For a one-thread run this is what the
 * run would take on an idle machine, near enough, whatever else is running; for a
 * threaded run it also counts time threads spend spinning at barriers. */
static double cpu_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static double load1(void)
{
    double l[1] = {-1.0};
    return getloadavg(l, 1) == 1 ? l[0] : -1.0;
}

/* ---- contention on a shared machine -------------------------------------------
 * Timings here come from a 4-core VM that other jobs use in bursts; an OpenMP barrier
 * whose fourth thread is descheduled turns a 0.9 ms kv_b application into 25 ms. Two
 * Linux counters make that visible instead of silently averaged in:
 *   /proc/stat           before a run: the fraction of CPU time idle over 250 ms while
 *                        this process sleeps, i.e. what the OTHER jobs are using;
 *   /proc/pressure/cpu   during a run: the share of wall time in which some runnable
 *                        task, ours included, was waiting for a CPU (PSI "some").
 * A run waits (up to --quiet-wait seconds) for the machine to go quiet, and a short run
 * whose stall share exceeds 10% is discarded and repeated, at most three times. Every
 * run's stall share is kept in the JSON. Elsewhere (no /proc) both are skipped. */
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
    for (int i = 0; i < 8; i++) *total += v[i];     /* guest time is already in user */
    return 0;
}

static void sleep_s(double s)
{
    struct timespec t;
    t.tv_sec = (time_t)s;
    t.tv_nsec = (long)((s - (double)t.tv_sec) * 1e9);
    nanosleep(&t, NULL);
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

static double g_quiet_wait = 120.0;

/* Wait until at least `need` of the machine is idle; returns the idle fraction seen. */
static double wait_quiet(double need)
{
    const double t0 = now_s();
    double idle = idle_fraction(0.25);
    while (idle >= 0.0 && idle < need && now_s() - t0 < g_quiet_wait) {
        sleep_s(0.75);
        idle = idle_fraction(0.25);
    }
    return idle;
}

static int threads_now(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%.1f MB)\n", (double)n / 1e6); exit(1); }
    return p;
}

/* Cheap uniform fill for timing buffers: values only need to be finite, normal and of
 * a plausible size; Box-Muller over a 1.6 GB cache would take longer than the runs. */
static void fill_fast(float *p, size_t n, float amp, uint64_t seed)
{
    MlaRng r;
    mla_rng_seed(&r, seed);
    for (size_t i = 0; i < n; i++)
        p[i] = amp * (float)(2.0 * mla_rng_unit(&r) - 1.0);
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n)
{
    double t[MAX_RUNS];
    memcpy(t, v, (size_t)n * sizeof(double));
    qsort(t, (size_t)n, sizeof(double), cmp_d);
    return n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

static int parse_list(const char *s, int *out)
{
    int n = 0;
    while (*s && n < MAX_LIST) {
        char *e;
        const long v = strtol(s, &e, 10);
        if (e == s || v < 0 || v > 1000000) return -1;
        out[n++] = (int)v;
        s = *e == ',' ? e + 1 : e;
        if (*e && *e != ',') return -1;
    }
    return n;
}

static int variant_of(const char *s)
{
    for (int v = 0; v < MLA_NVAR; v++)
        if (!strcmp(s, MLA_NAME[v])) return v;
    if (!strcmp(s, "EP")) return MLA_EP;
    return -1;
}

/* The engine kernels timed here (k3_mmw, k3_matmul_bf16) take their AVX-512 paths
 * whenever __AVX512F__ is defined, so a -march=native build on an AVX-512 machine is
 * labelled by that, not by the AVX2 it also has. */
static const char *isa(void)
{
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__ARM_NEON) && defined(__aarch64__)
    return "NEON";
#else
    return "scalar";
#endif
}

static void cpu_name(char *buf, size_t n)
{
    snprintf(buf, n, "unknown");
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "model name", 10)) {
                const char *p = strchr(line, ':');
                if (p) {
                    snprintf(buf, n, "%s", p + 2);
                    buf[strcspn(buf, "\n")] = 0;
                }
                break;
            }
        fclose(f);
        return;
    }
#ifdef __APPLE__
    f = popen("sysctl -n machdep.cpu.brand_string", "r");
    if (f) {
        if (fgets(buf, (int)n, f)) buf[strcspn(buf, "\n")] = 0;
        pclose(f);
    }
#endif
}

/* ================================================================== timing ==== */

typedef struct {
    int    v, C, T, threads;
    int    status;                 /* 1 measured, 0 projected (too slow), 2 memory */
    int    nruns;
    double runs[MAX_RUNS];
    double med, lo, hi, projected;
    double rebuilds, macs;
    double persistent_pp;          /* bytes per cached position per layer          */
    double transient;              /* bytes of scratch for this call               */
    double load_before, load_after;
    double stall[MAX_RUNS];        /* PSI "some" share of each kept run, -1 unknown */
    double cpu[MAX_RUNS];          /* process CPU seconds of each kept run          */
    double idle_before[MAX_RUNS];  /* machine idle fraction just before each run    */
    int    discarded;              /* runs repeated because they were contended     */
    int    vcap;
} Result;

static double result_seconds(const Result *r) { return r->status == 1 ? r->med : r->projected; }

static void print_result(const Result *r, FILE *json, const char *cpu)
{
    const double s = result_seconds(r);
    const double per_tok_layer = s / r->T;
    const double per_tok_model = per_tok_layer * N_MLA_LAYERS;
    const char *tag = r->status == 1 ? "measured" : r->status == 0 ? "PROJECTED" : "NOT RUN";
    printf("%-3s C=%-6d T=%d  %9.4f s/call  %9.3f ms/token/layer  %8.3f s/token x24 "
           "(%6.2f%% of %.0f s)  %8.3f GMAC/token x24  %s",
           MLA_NAME[r->v], r->C, r->T, s, per_tok_layer * 1e3, per_tok_model,
           100.0 * per_tok_model / NONATTN_SECONDS, NONATTN_SECONDS,
           r->macs / r->T * N_MLA_LAYERS / 1e9, tag);
    if (r->status == 1) {
        double smax = -1.0;
        for (int i = 0; i < r->nruns; i++) if (r->stall[i] > smax) smax = r->stall[i];
        printf(" [%d runs %.4f..%.4f, cpu %.4f, max stall %.0f%%, %d redone]", r->nruns,
               r->lo, r->hi, median(r->cpu, r->nruns), smax < 0 ? -1.0 : 100.0 * smax,
               r->discarded);
    }
    printf("\n");
    if (!json) return;
    fprintf(json,
            "{\"mode\":\"time\",\"variant\":\"%s\",\"C\":%d,\"T\":%d,\"threads\":%d,"
            "\"status\":\"%s\",\"seconds_per_call\":%s,\"runs\":[",
            MLA_NAME[r->v], r->C, r->T, r->threads,
            r->status == 1 ? "measured" : r->status == 0 ? "projected" : "not_run_memory",
            jnum(s, "%.9g"));
    for (int i = 0; i < r->nruns; i++) fprintf(json, "%s%.9g", i ? "," : "", r->runs[i]);
    fprintf(json, "],\"stall_share\":[");
    for (int i = 0; i < r->nruns; i++) fprintf(json, "%s%.4f", i ? "," : "", r->stall[i]);
    fprintf(json, "],\"idle_before\":[");
    for (int i = 0; i < r->nruns; i++)
        fprintf(json, "%s%.4f", i ? "," : "", r->idle_before[i]);
    fprintf(json, "],\"cpu_seconds\":[");
    for (int i = 0; i < r->nruns; i++) fprintf(json, "%s%.6g", i ? "," : "", r->cpu[i]);
    fprintf(json, "],\"redone\":%d,", r->discarded);
    fprintf(json,
            "\"min\":%s,\"max\":%s,\"ms_per_token_per_layer\":%s,"
            "\"s_per_token_x24\":%s,\"fraction_of_16s\":%s,\"kv_b_applications\":%.9g,"
            "\"macs_per_call\":%.9g,\"gmac_per_token_x24\":%.9g,"
            "\"persistent_bytes_per_position_per_layer\":%.0f,\"transient_bytes\":%.0f,"
            "\"vcap\":%d,\"load_before\":%.2f,\"load_after\":%.2f,\"isa\":\"%s\","
            "\"cpu\":\"%s\"}\n",
            jnum(r->status == 1 ? r->lo : s, "%.9g"),
            jnum(r->status == 1 ? r->hi : s, "%.9g"),
            jnum(per_tok_layer * 1e3, "%.9g"), jnum(per_tok_model, "%.9g"),
            jnum(per_tok_model / NONATTN_SECONDS, "%.9g"), r->rebuilds, r->macs,
            r->macs / r->T * N_MLA_LAYERS / 1e9, r->persistent_pp, r->transient, r->vcap,
            r->load_before, r->load_after, isa(), cpu);
    fflush(json);
}

typedef struct {
    int    threads, runs;
    int    nC, Cs[MAX_LIST], nT, Ts[MAX_LIST];
    int    use[MLA_NVAR];
    double max_run_s, vbuf_mb, mem_mb;
    int    common, ncpu;
    const char *json;
} TimeOpt;

/* One application of kv_b at the released geometry: the latent paths' unit of work. */
static double time_rebuild(const K3MlaW *w, const K3Cfg *c, const float *lat, int nlat)
{
    const int kvd = c->qk_nope + c->v_head, H = c->n_heads;
    float *kb = (float *)xmalloc((size_t)H * kvd * sizeof(float));
    double t[31];
    for (int i = 0; i < 3; i++)
        k3_mmw(kb, lat + (size_t)(i % nlat) * c->kv_lora, w->kv_b, w->wdt, c->kv_lora, H * kvd);
    for (int i = 0; i < 31; i++) {
        const double t0 = now_s();
        k3_mmw(kb, lat + (size_t)(i % nlat) * c->kv_lora, w->kv_b, w->wdt, c->kv_lora, H * kvd);
        t[i] = now_s() - t0;
    }
    free(kb);
    return median(t, 31);
}

/* The work every variant shares: projections for one token, then gate and o_proj. */
static void time_common(const K3Cfg *c, int runs)
{
    MlaSynth S;
    printf("\ncommon per-token MLA work (q_a, q_b, kv_a, rmsnorms, gate, o_proj): "
           "building the full layer...\n");
    if (mla_synth_layer(&S, c, K3_WBF16, 0.02f, 0.1f, 5, 1)) {
        printf("  not run: could not allocate the layer\n");
        return;
    }
    const int E = c->hidden, H = c->n_heads, qh = c->qk_nope + c->qk_rope;
    float *x   = (float *)xmalloc((size_t)E * sizeof(float));
    float *q   = (float *)xmalloc((size_t)H * qh * sizeof(float));
    float *ct  = (float *)xmalloc((size_t)(c->kv_lora + c->qk_rope) * sizeof(float));
    float *ql  = (float *)xmalloc((size_t)c->q_lora * sizeof(float));
    float *acc = (float *)xmalloc((size_t)H * c->v_head * sizeof(float));
    float *g   = (float *)xmalloc((size_t)H * c->v_head * sizeof(float));
    float *out = (float *)xmalloc((size_t)E * sizeof(float));
    fill_fast(x, (size_t)E, 1.0f, 3);
    double t[MAX_RUNS];
    for (int r = -1; r < runs; r++) {
        fill_fast(acc, (size_t)H * c->v_head, 0.1f, 4);
        const double t0 = now_s();
        mla_project(q, ct, x, &S.w, c, 1, ql);
        mla_finish(out, acc, x, &S.w, c, 1, g);
        if (r >= 0) t[r] = now_s() - t0;
    }
    const double m = median(t, runs);
    const double macs = (double)c->q_lora * E + (double)H * qh * c->q_lora
                        + (double)(c->kv_lora + c->qk_rope) * E
                        + 2.0 * (double)H * c->v_head * E;
    printf("  %.3f ms/token/layer (median of %d), %.1f MMAC; x24 = %.3f s/token\n",
           m * 1e3, runs, macs / 1e6, m * N_MLA_LAYERS);
    free(x); free(q); free(ct); free(ql); free(acc); free(g); free(out);
    mla_synth_free(&S);
}

static int run_time(const TimeOpt *o)
{
    K3Cfg c;
    mla_cfg_k3(&c, 7168);
    const int H = c.n_heads, qh = c.qk_nope + c.qk_rope, kvl = c.kv_lora, qr = c.qk_rope;
    const int kvw = kvl + qr, vh = c.v_head;
    char cpu[256];
    cpu_name(cpu, sizeof cpu);
    FILE *json = o->json ? fopen(o->json, "a") : NULL;
    if (o->json && !json) { fprintf(stderr, "cannot open %s\n", o->json); return 1; }

    MlaSynth S;
    if (mla_synth_layer(&S, &c, K3_WBF16, 0.02f, 0.1f, 1, 0)) return 1;
    int Tmax = 1;
    for (int i = 0; i < o->nT; i++) if (o->Ts[i] > Tmax) Tmax = o->Ts[i];

    printf("MLA attention variants, one layer, K3 geometry; %s build, %d thread(s), %s\n",
           isa(), o->threads, cpu);
    printf("timed region: append T new tokens + attention (mla_attend); medians of %d runs\n",
           o->runs);

    /* the unit cost of the latent paths */
    float *lat0 = (float *)xmalloc((size_t)64 * kvl * sizeof(float));
    fill_fast(lat0, (size_t)64 * kvl, 1.7f, 9);
    const double t_rb = time_rebuild(&S.w, &c, lat0, 64);
    free(lat0);
    printf("kv_b application (24576x512 bf16, 12.58 MMAC): %.3f ms median of 31  "
           "(%.2f GMAC/s)\n\n", t_rb * 1e3, 12.582912e6 / t_rb / 1e9);
    if (json)
        fprintf(json, "{\"mode\":\"kv_b\",\"threads\":%d,\"seconds\":%.9g,\"isa\":\"%s\","
                "\"cpu\":\"%s\"}\n", o->threads, t_rb, isa(), cpu);

    if (o->common) time_common(&c, o->runs);

    Result all[MAX_LIST * MAX_LIST * MLA_NVAR];
    int nall = 0;
    double last_per_work[MLA_NVAR];      /* s per (query, position) from the last run */
    for (int v = 0; v < MLA_NVAR; v++) last_per_work[v] = -1.0;
    double model_ratio_sum = 0.0;         /* measured / (rebuilds x t_rb), latent paths */
    int model_ratio_n = 0;

    for (int ic = 0; ic < o->nC; ic++) {
        const int C = o->Cs[ic], cap = C + Tmax;
        int need_lat = 0, need_exp = 0;
        for (int v = 0; v < MLA_NVAR; v++)
            if (o->use[v]) { if (mla_is_latent(v)) need_lat = 1; else need_exp = 1; }
        const double exp_bytes = (double)cap * mla_cache_floats(&c, 0) * 4.0;
        const int exp_fits = exp_bytes <= o->mem_mb * 1e6;
        MlaCache lat = {0}, ex = {0};
        if (need_lat) {
            lat.cap = cap;
            lat.kv = (float *)xmalloc((size_t)cap * kvl * sizeof(float));
            lat.rope = (float *)xmalloc((size_t)cap * qr * sizeof(float));
            fill_fast(lat.kv, (size_t)cap * kvl, 1.7f, 100 + (uint64_t)C);
            fill_fast(lat.rope, (size_t)cap * qr, 2.9f, 200 + (uint64_t)C);
        }
        if (need_exp && exp_fits) {
            ex.cap = cap;
            ex.kv = (float *)xmalloc((size_t)cap * H * (c.qk_nope + vh) * sizeof(float));
            ex.rope = (float *)xmalloc((size_t)cap * qr * sizeof(float));
            fill_fast(ex.kv, (size_t)cap * H * (c.qk_nope + vh), 0.78f, 300 + (uint64_t)C);
            fill_fast(ex.rope, (size_t)cap * qr, 2.9f, 200 + (uint64_t)C);
        }
        float *q = (float *)xmalloc((size_t)Tmax * H * qh * sizeof(float));
        float *ct = (float *)xmalloc((size_t)Tmax * kvw * sizeof(float));
        float *acc = (float *)xmalloc((size_t)Tmax * H * vh * sizeof(float));
        fill_fast(q, (size_t)Tmax * H * qh, 1.4f, 400);
        fill_fast(ct, (size_t)Tmax * kvw, 1.7f, 500);

        for (int it = 0; it < o->nT; it++) {
            const int T = o->Ts[it], N = C + T;
            Result *res[MLA_NVAR] = {0};
            void *scr[MLA_NVAR] = {0};
            int measure[MLA_NVAR] = {0}, nmeasure = 0;
            double work = 0.0;                       /* sum over t of (C + t + 1) */
            for (int t = 0; t < T; t++) work += (double)C + t + 1;
            const int vcap_budget = (int)fmin((double)N, o->vbuf_mb * 1e6 / (H * vh * 4.0));
            /* measured/modelled ratio of the latent runs so far, 1 until there are some */
            const double kmodel = model_ratio_n ? model_ratio_sum / model_ratio_n : 1.0;

            for (int v = 0; v < MLA_NVAR; v++) {
                if (!o->use[v]) continue;
                Result *r = &all[nall++];
                memset(r, 0, sizeof *r);
                res[v] = r;
                r->v = v; r->C = C; r->T = T; r->threads = o->threads;
                r->vcap = v == MLA_L1 ? vcap_budget : 0;
                r->rebuilds = mla_rebuilds(v, T, C, r->vcap);
                r->macs = mla_macs(v, &c, T, C, r->vcap);
                r->persistent_pp = (double)mla_cache_floats(&c, mla_is_latent(v)) * 4.0;
                r->transient = (double)mla_scratch_bytes(v, &c, T, N, r->vcap);
                double est;
                if (v == MLA_L0 || v == MLA_L1) est = r->rebuilds * t_rb * kmodel;
                else if (last_per_work[v] > 0) est = last_per_work[v] * work;
                else est = 0.0;
                const double mem = r->transient + (mla_is_latent(v) ? 0.0 : exp_bytes);
                if (!mla_is_latent(v) && !exp_fits) {
                    r->status = 2;
                } else if (mem > o->mem_mb * 1e6) {
                    r->status = 2;
                } else if (est > o->max_run_s) {
                    r->status = 0;
                } else {
                    r->status = 1;
                    measure[v] = 1;
                    nmeasure++;
                    scr[v] = xmalloc((size_t)r->transient);
                    memset(scr[v], 0, (size_t)r->transient);
                }
                r->projected = est;
            }

            /* Warm-up (untimed) for anything short, then interleaved timed runs. */
            for (int v = 0; v < MLA_NVAR; v++)
                if (measure[v] && (res[v]->projected < 5.0))
                    mla_attend(v, acc, q, ct, T, C, mla_is_latent(v) ? &lat : &ex, &S.w, &c,
                               scr[v], res[v]->vcap, NULL);
            for (int v = 0; v < MLA_NVAR; v++)
                if (measure[v]) res[v]->load_before = load1();
            /* Idle share of the whole machine to wait for before each run: 90% (the
             * other jobs using at most 0.4 of 4 cores) when every core is ours, 75%
             * otherwise; memory bandwidth and L3 are shared even when cores are not. */
            const double need = o->threads >= o->ncpu ? 0.90 : 0.75;
            for (int r = 0; r < o->runs && nmeasure; r++)
                for (int v = 0; v < MLA_NVAR; v++) {
                    if (!measure[v]) continue;
                    Result *R = res[v];
                    for (int attempt = 0;; attempt++) {
                        const double idle = wait_quiet(need);
                        const double p0 = psi_some_us(), c0 = cpu_s(), t0 = now_s();
                        mla_attend(v, acc, q, ct, T, C, mla_is_latent(v) ? &lat : &ex, &S.w,
                                   &c, scr[v], R->vcap, NULL);
                        const double dt = now_s() - t0, dc = cpu_s() - c0;
                        const double p1 = psi_some_us();
                        const double stall = (p0 >= 0 && p1 >= 0) ? (p1 - p0) / (dt * 1e6)
                                                                  : -1.0;
                        if (stall > 0.10 && dt < 10.0 && attempt < 3) {
                            R->discarded++;
                            continue;
                        }
                        R->idle_before[R->nruns] = idle;
                        R->stall[R->nruns] = stall;
                        R->cpu[R->nruns] = dc;
                        R->runs[R->nruns++] = dt;
                        break;
                    }
                }
            for (int v = 0; v < MLA_NVAR; v++) {
                if (!measure[v]) continue;
                Result *r = res[v];
                r->load_after = load1();
                r->med = median(r->runs, r->nruns);
                r->lo = r->hi = r->runs[0];
                for (int i = 1; i < r->nruns; i++) {
                    if (r->runs[i] < r->lo) r->lo = r->runs[i];
                    if (r->runs[i] > r->hi) r->hi = r->runs[i];
                }
                last_per_work[v] = r->med / work;
                if (v == MLA_L0 || v == MLA_L1) {
                    model_ratio_sum += r->med / (r->rebuilds * t_rb);
                    model_ratio_n++;
                }
                free(scr[v]);
            }
            /* Projections for what did not run, calibrated by what did. */
            for (int v = 0; v < MLA_NVAR; v++) {
                Result *r = res[v];
                if (!r || r->status == 1) continue;
                if (v == MLA_L0 || v == MLA_L1) {
                    const double k = model_ratio_n ? model_ratio_sum / model_ratio_n : 1.0;
                    r->projected = r->rebuilds * t_rb * k;
                } else if (last_per_work[v] > 0) {
                    r->projected = last_per_work[v] * work;
                } else {
                    r->projected = NAN;
                }
            }
            for (int v = 0; v < MLA_NVAR; v++)
                if (res[v]) print_result(res[v], json, cpu);
            printf("\n");
        }
        free(q); free(ct); free(acc);
        free(lat.kv); free(lat.rope); free(ex.kv); free(ex.rope);
    }

    printf("latent-path model: measured/(kv_b applications x %.3f ms) = %.3f over %d runs\n",
           t_rb * 1e3, model_ratio_n ? model_ratio_sum / model_ratio_n : NAN, model_ratio_n);
    if (json) {
        fprintf(json, "{\"mode\":\"model\",\"threads\":%d,\"kv_b_seconds\":%.9g,"
                "\"measured_over_model\":%.6f,\"points\":%d}\n", o->threads, t_rb,
                model_ratio_n ? model_ratio_sum / model_ratio_n : -1.0, model_ratio_n);
        fclose(json);
    }
    mla_synth_free(&S);
    return 0;
}

/* ================================================================ numerics ==== */

/* Log-binned histogram of non-negative values: 16 bins per octave from 2^-64 to 2^16,
 * plus exact zeros. Quantiles are read at bin upper edges (conservative, within 4.4%). */
#define HB_PER 16
#define HB_LO  (-64)
#define HB_N   ((16 - HB_LO) * HB_PER)
typedef struct {
    double zeros, n, max;
    double bin[HB_N];
} Hist;

static void hist_add(Hist *h, double x)
{
    h->n += 1.0;
    if (x > h->max) h->max = x;
    if (x == 0.0) { h->zeros += 1.0; return; }
    int b = (int)floor((log2(x) - HB_LO) * HB_PER);
    if (b < 0) b = 0;
    if (b >= HB_N) b = HB_N - 1;
    h->bin[b] += 1.0;
}

static double hist_q(const Hist *h, double q)
{
    const double target = q * h->n;
    double acc = h->zeros;
    if (acc >= target) return 0.0;
    for (int b = 0; b < HB_N; b++) {
        acc += h->bin[b];
        if (acc >= target) {
            const double edge = exp2((double)(b + 1) / HB_PER + HB_LO);
            return edge < h->max ? edge : h->max;
        }
    }
    return h->max;
}

typedef struct {
    double max_abs, max_ref, sum_d2, sum_r2, max_rel_elem, n, n_equal;
} Dev;

/* a against b (b is the denominator); every element counted. */
static void dev_add(Dev *d, const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const double x = a[i], y = b[i], e = fabs(x - y);
        if (e > d->max_abs) d->max_abs = e;
        if (fabs(y) > d->max_ref) d->max_ref = fabs(y);
        d->sum_d2 += e * e;
        d->sum_r2 += y * y;
        if (y != 0.0 && e / fabs(y) > d->max_rel_elem) d->max_rel_elem = e / fabs(y);
        d->n_equal += (a[i] == b[i]);
        d->n += 1.0;
    }
}
static void dev_add_d(Dev *d, const float *a, const double *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const double x = a[i], y = b[i], e = fabs(x - y);
        if (e > d->max_abs) d->max_abs = e;
        if (fabs(y) > d->max_ref) d->max_ref = fabs(y);
        d->sum_d2 += e * e;
        d->sum_r2 += y * y;
        if (y != 0.0 && e / fabs(y) > d->max_rel_elem) d->max_rel_elem = e / fabs(y);
        d->n += 1.0;
    }
}

static void dev_print(const char *label, const Dev *d, FILE *json, int C, int first)
{
    printf("  %-34s max|d| %.3e  max|d|/max|ref| %.3e  ||d||/||ref|| %.3e  "
           "max elementwise |d|/|ref| %.3e  bitwise equal %.2f%%\n",
           label, d->max_abs, d->max_abs / d->max_ref, sqrt(d->sum_d2 / d->sum_r2),
           d->max_rel_elem, 100.0 * d->n_equal / d->n);
    if (json)
        fprintf(json, "%s\"%s\":{\"max_abs\":%s,\"max_ref\":%s,"
                "\"max_abs_over_max_ref\":%s,\"rel_l2\":%s,\"max_rel_elementwise\":%s,"
                "\"bitwise_equal_fraction\":%s,\"elements\":%.0f}", first ? "" : ",", label, jnum(d->max_abs, "%.6g"),
                jnum(d->max_ref, "%.6g"), jnum(d->max_abs / d->max_ref, "%.6g"),
                jnum(sqrt(d->sum_d2 / d->sum_r2), "%.6g"), jnum(d->max_rel_elem, "%.6g"),
                jnum(d->n_equal / d->n, "%.6g"), d->n);
    (void)C;
}

/* The same attention in double throughout: k and v never rounded to float, the exact
 * scale, softmax with exp() in double. Computed through the absorbed form, which is the
 * same real number; its own rounding (~1e-16 relative) is far below what is measured. */
static void ref_attend(double *accR, double *scR, const float *qt, int p, const MlaCache *lat,
                       const K3MlaW *w, const K3Cfg *c)
{
    const int H = c->n_heads, qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int kvd = qn + vh, kvl = c->kv_lora, n = p + 1;
    const double scale = 1.0 / sqrt((double)(qn + qr));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int h = 0; h < H; h++) {
        double qa[MLA_MAX_LORA], u[MLA_MAX_LORA];
        const float *qh = qt + (size_t)h * (qn + qr);
        const uint16_t *W = (const uint16_t *)w->kv_b;
        for (int j = 0; j < kvl; j++) qa[j] = 0.0;
        for (int i = 0; i < qn; i++) {
            const uint16_t *row = W + ((size_t)h * kvd + i) * kvl;
            for (int j = 0; j < kvl; j++) qa[j] += (double)k3_bf16f(row[j]) * (double)qh[i];
        }
        double *sc = scR + (size_t)h * n;
        double m = -INFINITY;
        for (int s = 0; s < n; s++) {
            const float *cs = lat->kv + (size_t)s * kvl, *kr = lat->rope + (size_t)s * qr;
            double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
            for (int j = 0; j + 3 < kvl; j += 4) {
                d0 += qa[j] * cs[j]; d1 += qa[j + 1] * cs[j + 1];
                d2 += qa[j + 2] * cs[j + 2]; d3 += qa[j + 3] * cs[j + 3];
            }
            for (int i = 0; i < qr; i++) d0 += (double)qh[qn + i] * (double)kr[i];
            sc[s] = ((d0 + d1) + (d2 + d3)) * scale;
            if (sc[s] > m) m = sc[s];
        }
        double z = 0.0;
        for (int j = 0; j < kvl; j++) u[j] = 0.0;
        for (int s = 0; s < n; s++) z += exp(sc[s] - m);
        for (int s = 0; s < n; s++) {
            const double pr = exp(sc[s] - m) / z;
            const float *cs = lat->kv + (size_t)s * kvl;
            for (int j = 0; j < kvl; j++) u[j] += pr * (double)cs[j];
        }
        for (int i = 0; i < vh; i++) {
            const uint16_t *row = W + ((size_t)h * kvd + qn + i) * kvl;
            double a = 0.0;
            for (int j = 0; j < kvl; j++) a += (double)k3_bf16f(row[j]) * u[j];
            accR[(size_t)h * vh + i] = a;
        }
    }
}

static int argmax_f(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

typedef struct {
    int nC, Cs[MAX_LIST], queries, rows;
    double qscale;
    const char *json;
} NumOpt;

static int run_numerics(const NumOpt *o)
{
    K3Cfg c;
    mla_cfg_k3(&c, 7168);
    const int E = c.hidden, H = c.n_heads, qh = c.qk_nope + c.qk_rope, kvl = c.kv_lora;
    const int qr = c.qk_rope, kvw = kvl + qr, vh = c.v_head, kvd = c.qk_nope + vh;
    char cpu[256];
    cpu_name(cpu, sizeof cpu);
    FILE *json = o->json ? fopen(o->json, "a") : NULL;
    if (o->json && !json) { fprintf(stderr, "cannot open %s\n", o->json); return 1; }
    printf("numerics: absorbed (A) against expanded (E) and a double reference (R)\n"
           "weights ~ N(0, 0.02) in bf16, norm weights 1 + N(0, 0.1), hidden ~ N(0, 1);\n"
           "the engine's projections and rmsnorms; %d query tokens per context%s\n",
           o->queries, o->qscale != 1.0 ? "; STRESS: queries scaled" : "");
    if (o->qscale != 1.0) printf("query scale %g (sharper softmax than the weights give)\n",
                                 o->qscale);
    MlaSynth S;
    const double t_syn = now_s();
    if (mla_synth_layer(&S, &c, K3_WBF16, 0.02f, 0.1f, 2026, 1)) return 1;
    printf("layer built in %.1f s\n", now_s() - t_syn);

    for (int ic = 0; ic < o->nC; ic++) {
        const int C = o->Cs[ic], N = C + 1, cap = N;
        const double t0 = now_s();
        MlaCache lat = {0}, ex = {0};
        lat.cap = ex.cap = cap;
        lat.kv  = (float *)xmalloc((size_t)cap * kvl * sizeof(float));
        lat.rope = (float *)xmalloc((size_t)cap * qr * sizeof(float));
        ex.kv   = (float *)xmalloc((size_t)cap * H * kvd * sizeof(float));
        ex.rope = (float *)xmalloc((size_t)cap * qr * sizeof(float));
        /* The context: hidden states through the engine's kv projection, stored in both
         * layouts exactly as the engine stores them. */
        const int chunk = 256;
        float *x = (float *)xmalloc((size_t)chunk * E * sizeof(float));
        float *ct = (float *)xmalloc((size_t)chunk * kvw * sizeof(float));
        for (int s0 = 0; s0 < C; s0 += chunk) {
            const int nb = C - s0 < chunk ? C - s0 : chunk;
            mla_fill_normal(x, (size_t)nb * E, 1.0f, 77000 + (uint64_t)s0);
            mla_project_kv(ct, x, &S.w, &c, nb);
            for (int b = 0; b < nb; b++) {
                const int s = s0 + b;
                const float *cs = ct + (size_t)b * kvw;
                memcpy(lat.kv + (size_t)s * kvl, cs, (size_t)kvl * sizeof(float));
                memcpy(lat.rope + (size_t)s * qr, cs + kvl, (size_t)qr * sizeof(float));
                memcpy(ex.rope + (size_t)s * qr, cs + kvl, (size_t)qr * sizeof(float));
                k3_mmw(ex.kv + (size_t)s * H * kvd, cs, S.w.kv_b, S.w.wdt, kvl, H * kvd);
            }
        }
        free(x); free(ct);
        printf("\ncontext C=%d built in %.1f s\n", C, now_s() - t0);

        float *xq   = (float *)xmalloc((size_t)E * sizeof(float));
        float *q    = (float *)xmalloc((size_t)H * qh * sizeof(float));
        float *cq   = (float *)xmalloc((size_t)kvw * sizeof(float));
        float *ql   = (float *)xmalloc((size_t)c.q_lora * sizeof(float));
        float *gbuf = (float *)xmalloc((size_t)H * vh * sizeof(float));
        float *accE = (float *)xmalloc((size_t)H * vh * sizeof(float));
        float *accA = (float *)xmalloc((size_t)H * vh * sizeof(float));
        float *accX = (float *)xmalloc((size_t)H * vh * sizeof(float));
        float *outE = (float *)xmalloc((size_t)E * sizeof(float));
        float *outA = (float *)xmalloc((size_t)E * sizeof(float));
        float *scE  = (float *)xmalloc((size_t)H * N * sizeof(float));
        float *scA  = (float *)xmalloc((size_t)H * N * sizeof(float));
        float *scX  = (float *)xmalloc((size_t)H * N * sizeof(float));
        double *accR = (double *)xmalloc((size_t)H * vh * sizeof(double));
        double *scR  = (double *)xmalloc((size_t)H * N * sizeof(double));
        void *sEP = xmalloc(mla_scratch_bytes(MLA_EP, &c, 1, N, 0));
        void *sE  = xmalloc(mla_scratch_bytes(MLA_E, &c, 1, N, 0));
        void *sA  = xmalloc(mla_scratch_bytes(MLA_A, &c, 1, N, 0));

        Dev dAE = {0}, dER = {0}, dAR = {0}, dOut = {0};
        Hist hS, hER, hAR;
        memset(&hS, 0, sizeof hS);
        memset(&hER, 0, sizeof hER); memset(&hAR, 0, sizeof hAR);
        double score_abs_max = 0.0, score_sq = 0.0;
        int rows = 0, flips = 0, at_risk = 0, e_checked = 0, e_same = 1;
        double min_gap = INFINITY;
        double tE = 0, tA = 0, tR = 0;

        for (int qi = 0; qi < o->queries; qi++) {
            mla_fill_normal(xq, (size_t)E, 1.0f, 990000 + (uint64_t)qi);
            mla_project(q, cq, xq, &S.w, &c, 1, ql);
            if (o->qscale != 1.0)      /* stress: sharper softmax rows, same weights */
                for (int i = 0; i < H * qh; i++) q[i] = (float)(q[i] * o->qscale);
            double a = now_s();
            mla_attend(MLA_EP, accE, q, cq, 1, C, &ex, &S.w, &c, sEP, 0, scE);
            tE += now_s() - a;
            if (qi < 2) {         /* E itself, to tie E+ to it on these inputs too */
                mla_attend(MLA_E, accX, q, cq, 1, C, &ex, &S.w, &c, sE, 0, scX);
                e_checked++;
                if (memcmp(accX, accE, (size_t)H * vh * sizeof(float))
                    || memcmp(scX, scE, (size_t)H * N * sizeof(float)))
                    e_same = 0;
            }
            a = now_s();
            if (mla_attend(MLA_A, accA, q, cq, 1, C, &lat, &S.w, &c, sA, 0, scA)) return 1;
            tA += now_s() - a;
            a = now_s();
            ref_attend(accR, scR, q, C, &lat, &S.w, &c);
            tR += now_s() - a;

            dev_add(&dAE, accA, accE, (size_t)H * vh);
            dev_add_d(&dER, accE, accR, (size_t)H * vh);
            dev_add_d(&dAR, accA, accR, (size_t)H * vh);
            for (int h = 0; h < H; h++) {
                const float *se = scE + (size_t)h * N, *sa = scA + (size_t)h * N;
                const double *sr = scR + (size_t)h * N;
                double dmax = 0.0;
                for (int s = 0; s < N; s++) {
                    const double d = fabs((double)se[s] - (double)sa[s]);
                    hist_add(&hS, d);
                    hist_add(&hER, fabs((double)se[s] - sr[s]));
                    hist_add(&hAR, fabs((double)sa[s] - sr[s]));
                    if (d > dmax) dmax = d;
                    if (fabs(se[s]) > score_abs_max) score_abs_max = fabs(se[s]);
                    score_sq += (double)se[s] * (double)se[s];
                }
                if (rows < o->rows) {
                    const int ae = argmax_f(se, N), aa = argmax_f(sa, N);
                    flips += ae != aa;
                    double second = -INFINITY;
                    for (int s = 0; s < N; s++)
                        if (s != ae && se[s] > second) second = se[s];
                    const double gap = N > 1 ? (double)se[ae] - second : INFINITY;
                    if (gap < min_gap) min_gap = gap;
                    at_risk += gap <= 2.0 * dmax;
                    rows++;
                }
            }
            mla_finish(outE, accE, xq, &S.w, &c, 1, gbuf);
            mla_finish(outA, accA, xq, &S.w, &c, 1, gbuf);
            dev_add(&dOut, outA, outE, (size_t)E);
        }

        printf("C=%d: %d query tokens x %d heads; E+ %.3f s, A %.3f s, reference %.3f s "
               "per query token\n", C, o->queries, H, tE / o->queries, tA / o->queries,
               tR / o->queries);
        printf("  E+ matched E bitwise on %d query tokens: %s\n", e_checked,
               e_same ? "yes" : "NO");
        if (json)
            fprintf(json, "{\"mode\":\"numerics\",\"C\":%d,\"queries\":%d,\"heads\":%d,"
                    "\"qscale\":%g,\"cpu\":\"%s\",\"isa\":\"%s\",\"e_plus_equals_e\":%s,", C,
                    o->queries, H, o->qscale, cpu, isa(), e_same ? "true" : "false");
        if (json) fprintf(json, "\"deviation\":{");
        dev_print("attn A vs E (before gate, o_proj)", &dAE, json, C, 1);
        dev_print("attn E vs double ref", &dER, json, C, 0);
        dev_print("attn A vs double ref", &dAR, json, C, 0);
        dev_print("layer out A vs E (after o_proj)", &dOut, json, C, 0);
        if (json) fprintf(json, "},");
        const double score_rms = sqrt(score_sq / hS.n);
        printf("  |score_E - score_A| over %.0f scores (score rms %.3f, max |score| %.2f): "
               "bitwise equal %.2f%%, median %.2e, p90 %.2e, p99 %.2e, p99.9 %.2e, max %.2e\n",
               hS.n, score_rms, score_abs_max, 100.0 * hS.zeros / hS.n, hist_q(&hS, 0.5),
               hist_q(&hS, 0.9), hist_q(&hS, 0.99), hist_q(&hS, 0.999), hS.max);
        printf("  against the double reference: |E - R| median %.2e max %.2e; "
               "|A - R| median %.2e max %.2e\n", hist_q(&hER, 0.5), hER.max,
               hist_q(&hAR, 0.5), hAR.max);
        printf("  argmax changed in %d of %d softmax rows; smallest top-2 gap %.3e; rows "
               "whose gap is within 2x their largest score difference: %d\n", flips, rows,
               min_gap, at_risk);
        if (json) {
            fprintf(json, "\"scores\":{\"count\":%.0f,\"rms_score\":%s,"
                    "\"abs_max_score\":%s,\"bitwise_equal_fraction\":%s,\"median\":%s,"
                    "\"p90\":%s,\"p99\":%s,\"p999\":%s,\"max\":%s,"
                    "\"e_vs_ref_median\":%s,\"e_vs_ref_max\":%s,\"a_vs_ref_median\":%s,"
                    "\"a_vs_ref_max\":%s},",
                    hS.n, jnum(score_rms, "%.6g"), jnum(score_abs_max, "%.6g"),
                    jnum(hS.zeros / hS.n, "%.6g"), jnum(hist_q(&hS, 0.5), "%.6g"),
                    jnum(hist_q(&hS, 0.9), "%.6g"), jnum(hist_q(&hS, 0.99), "%.6g"),
                    jnum(hist_q(&hS, 0.999), "%.6g"), jnum(hS.max, "%.6g"),
                    jnum(hist_q(&hER, 0.5), "%.6g"), jnum(hER.max, "%.6g"),
                    jnum(hist_q(&hAR, 0.5), "%.6g"), jnum(hAR.max, "%.6g"));
            fprintf(json, "\"argmax\":{\"rows\":%d,\"changed\":%d,\"min_top2_gap\":%s,"
                    "\"rows_gap_within_2x_diff\":%d},\"seconds_per_query\":{\"e_plus\":%.6g,"
                    "\"a\":%.6g,\"reference\":%.6g}}\n", rows, flips,
                    jnum(min_gap, "%.6g"), at_risk, tE / o->queries, tA / o->queries,
                    tR / o->queries);
            fflush(json);
        }
        free(xq); free(q); free(cq); free(ql); free(gbuf); free(accE); free(accA); free(accX);
        free(outE); free(outA); free(scE); free(scA); free(scX); free(accR); free(scR);
        free(sEP); free(sE); free(sA);
        free(lat.kv); free(lat.rope); free(ex.kv); free(ex.rope);
        if (!e_same) { fprintf(stderr, "E+ diverged from E\n"); return 1; }
    }
    if (json) fclose(json);
    mla_synth_free(&S);
    return 0;
}

/* ================================================================== counts ==== */
/* How many kv_b applications a call makes depends only on (C, T, vcap): the loops that
 * make them never look at the geometry. So they are counted where counting is cheap, by
 * running each variant on the fixture geometry (4 heads, kv_lora 32) with mla_kvb_calls
 * watching, and held to the closed form mla_rebuilds(); a mismatch fails the run. The
 * multiply-adds (mla_macs), bytes, and kv_b weight bytes touched are then quoted at the
 * released geometry. None of it moves with machine load, which is why this phase of the
 * study reports it and leaves wall time to a quiet machine. */

/* kv_b at K3 geometry in bf16: 24,576 rows of 512, read in full by every application. */
#define KVB_BYTES_K3 (24576.0 * 512.0 * 2.0)

/* kv_b weight traffic per call, in whole-matrix units: one per application, except A,
 * which never applies the whole matrix but reads its key half once per call (absorbing
 * every query token) and its value half once per query token. */
static double kvb_reads(int v, int T, double applications)
{
    return v == MLA_A ? 0.5 * (1.0 + T) : applications;
}

typedef struct {
    int nC, Cs[MAX_LIST], nT, Ts[MAX_LIST];
    double vbuf_mb;
    const char *json;
} CountOpt;

static int run_counts(const CountOpt *o)
{
    K3Cfg k3, fx;
    mla_cfg_k3(&k3, 7168);
    mla_cfg_tiny(&fx);
    const int fH = fx.n_heads, fqh = fx.qk_nope + fx.qk_rope, fkvw = fx.kv_lora + fx.qk_rope;
    const int fkvd = fx.qk_nope + fx.v_head, fvh = fx.v_head;
    FILE *json = o->json ? fopen(o->json, "a") : NULL;
    if (o->json && !json) { fprintf(stderr, "cannot open %s\n", o->json); return 1; }
    MlaSynth S;
    if (mla_synth_layer(&S, &fx, K3_WBF16, 0.1f, 0.1f, 3, 0)) return 1;
    int bad = 0;
    printf("kv_b applications per call, COUNTED on the fixture geometry (the count does not\n"
           "depend on it) and checked against mla_rebuilds(); multiply-adds and bytes at K3\n"
           "geometry (96 heads, 128+64, v 128, kv_lora 512), one MLA layer; x24 = all MLA\n"
           "layers. L1's value-row budget is --vbuf-mb %.0f MB at K3 geometry.\n",
           o->vbuf_mb);
    for (int ic = 0; ic < o->nC; ic++)
        for (int it = 0; it < o->nT; it++) {
            const int C = o->Cs[ic], T = o->Ts[it], N = C + T;
            const int vbudget = (int)fmin((double)N, o->vbuf_mb * 1e6
                                          / (k3.n_heads * k3.v_head * 4.0));
            /* rows: E, E+, L0, L1 at the budget, L1 holding no value rows, A */
            const int rv[6] = {MLA_E, MLA_EP, MLA_L0, MLA_L1, MLA_L1, MLA_A};
            const int rc[6] = {0, 0, 0, vbudget, 0, 0};
            MlaCache lat = {0}, ex = {0};
            lat.cap = ex.cap = N;
            lat.kv = (float *)xmalloc((size_t)N * fx.kv_lora * sizeof(float));
            lat.rope = (float *)xmalloc((size_t)N * fx.qk_rope * sizeof(float));
            ex.kv = (float *)xmalloc((size_t)N * fH * fkvd * sizeof(float));
            ex.rope = (float *)xmalloc((size_t)N * fx.qk_rope * sizeof(float));
            fill_fast(lat.kv, (size_t)N * fx.kv_lora, 1.0f, 11);
            fill_fast(lat.rope, (size_t)N * fx.qk_rope, 1.0f, 12);
            fill_fast(ex.kv, (size_t)N * fH * fkvd, 1.0f, 13);
            fill_fast(ex.rope, (size_t)N * fx.qk_rope, 1.0f, 12);
            float *q = (float *)xmalloc((size_t)T * fH * fqh * sizeof(float));
            float *ct = (float *)xmalloc((size_t)T * fkvw * sizeof(float));
            float *acc = (float *)xmalloc((size_t)T * fH * fvh * sizeof(float));
            fill_fast(q, (size_t)T * fH * fqh, 1.0f, 14);
            fill_fast(ct, (size_t)T * fkvw, 1.0f, 15);
            double counted[6];
            printf("\nC=%d T=%d (N=%d)\n  %-3s %6s  %12s  %11s  %13s  %11s  %13s  %12s\n", C, T,
                   N, "", "vcap", "kv_b applied", "closed form", "GMAC per call",
                   "GMAC/tok x24", "kv_b GB read", "cache B/pos");
            for (int i = 0; i < 6; i++) {
                const int v = rv[i];
                void *scr = xmalloc(mla_scratch_bytes(v, &fx, T, N, rc[i]));
                const unsigned long long k0 = mla_kvb_calls;
                if (mla_attend(v, acc, q, ct, T, C, mla_is_latent(v) ? &lat : &ex, &S.w, &fx,
                               scr, rc[i], NULL)) {
                    fprintf(stderr, "%s refused the layer\n", MLA_NAME[v]);
                    return 1;
                }
                free(scr);
                counted[i] = (double)(mla_kvb_calls - k0);
                const double want = mla_rebuilds(v, T, C, rc[i]);
                const double macs = mla_macs(v, &k3, T, C, rc[i]);
                const double pbytes = (double)mla_cache_floats(&k3, mla_is_latent(v)) * 4.0;
                const double tbytes = (double)mla_scratch_bytes(v, &k3, T, N, rc[i]);
                const int ok = counted[i] == want;
                bad |= !ok;
                char vc[16];
                if (v == MLA_L1) snprintf(vc, sizeof vc, "%d", rc[i]);
                else snprintf(vc, sizeof vc, "-");
                printf("  %-3s %6s  %12.0f  %11.0f%s  %13.3f  %11.3f  %13.3f  %12.0f\n",
                       MLA_NAME[v], vc, counted[i], want, ok ? " " : "!", macs / 1e9,
                       macs / T * N_MLA_LAYERS / 1e9,
                       kvb_reads(v, T, counted[i]) * KVB_BYTES_K3 / 1e9, pbytes);
                if (json)
                    fprintf(json, "{\"mode\":\"counts\",\"variant\":\"%s\",\"C\":%d,\"T\":%d,"
                            "\"vcap\":%d,\"kv_b_applications\":%.0f,"
                            "\"kv_b_applications_closed_form\":%.0f,\"macs_per_call\":%.9g,"
                            "\"gmac_per_token_x24\":%.9g,\"kv_b_bytes_read\":%.9g,"
                            "\"persistent_bytes_per_position_per_layer\":%.0f,"
                            "\"transient_bytes\":%.0f,\"counted_on\":\"fixture geometry\"}\n",
                            MLA_NAME[v], C, T, rc[i], counted[i], want, macs,
                            macs / T * N_MLA_LAYERS / 1e9,
                            kvb_reads(v, T, counted[i]) * KVB_BYTES_K3, pbytes, tbytes);
            }
            printf("  L0 / L1 kv_b applications: %.1fx (L1 holding no value rows: %.1fx)\n",
                   counted[2] / counted[3], counted[2] / counted[4]);
            free(q); free(ct); free(acc);
            free(lat.kv); free(lat.rope); free(ex.kv); free(ex.rope);
        }
    if (json) fclose(json);
    mla_synth_free(&S);
    if (bad) { fprintf(stderr, "a counted kv_b total differs from mla_rebuilds()\n"); return 1; }
    printf("\nall counts equal their closed forms\n");
    return 0;
}

/* ==================================================================== main ==== */

static void usage(void)
{
    fprintf(stderr,
            "usage: bench_mla [time] [--threads N] [--C 256,1024,...] [--T 1,5]\n"
            "                 [--variants E,E+,L0,L1,A] [--runs 5] [--max-run-s 60]\n"
            "                 [--vbuf-mb 2560] [--mem-mb 3072] [--quiet-wait 120]\n"
            "                 [--common] [--json FILE]\n"
            "       bench_mla numerics [--threads N] [--C 1024,16384] [--queries 105]\n"
            "                 [--rows 10000] [--qscale 1] [--json FILE]\n"
            "       bench_mla counts [--C 0] [--T 1,16,64,256] [--vbuf-mb 2560]\n"
            "                 [--json FILE]\n");
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);             /* progress is visible when logged */
    int numerics = 0, counts = 0, a = 1;
    if (a < argc && !strcmp(argv[a], "numerics")) { numerics = 1; a++; }
    else if (a < argc && !strcmp(argv[a], "counts")) { counts = 1; a++; }
    else if (a < argc && !strcmp(argv[a], "time")) a++;

    TimeOpt t;
    memset(&t, 0, sizeof t);
    t.threads = threads_now(); t.runs = 5; t.max_run_s = 60.0; t.vbuf_mb = 2560.0;
    t.mem_mb = 3072.0;
    t.nC = parse_list("256,1024,4096,16384", t.Cs);
    t.nT = parse_list("1,5", t.Ts);
    for (int v = 0; v < MLA_NVAR; v++) t.use[v] = 1;
    NumOpt n;
    memset(&n, 0, sizeof n);
    n.nC = parse_list("1024,16384", n.Cs);
    n.queries = 105; n.rows = 10000; n.qscale = 1.0;

    int c_given = 0, t_given = 0;
    for (; a < argc; a++) {
        const char *k = argv[a], *v = a + 1 < argc ? argv[a + 1] : NULL;
        if (!strcmp(k, "--common")) { t.common = 1; continue; }
        if (!v) { usage(); return 2; }
        a++;
        if (!strcmp(k, "--threads")) t.threads = atoi(v);
        else if (!strcmp(k, "--C")) {
            if ((t.nC = parse_list(v, t.Cs)) < 1 || (n.nC = parse_list(v, n.Cs)) < 1) {
                usage(); return 2;
            }
            c_given = 1;
        } else if (!strcmp(k, "--T")) {
            if ((t.nT = parse_list(v, t.Ts)) < 1) { usage(); return 2; }
            t_given = 1;
        }
        else if (!strcmp(k, "--runs")) t.runs = atoi(v);
        else if (!strcmp(k, "--max-run-s")) t.max_run_s = atof(v);
        else if (!strcmp(k, "--vbuf-mb")) t.vbuf_mb = atof(v);
        else if (!strcmp(k, "--mem-mb")) t.mem_mb = atof(v);
        else if (!strcmp(k, "--quiet-wait")) g_quiet_wait = atof(v);
        else if (!strcmp(k, "--queries")) n.queries = atoi(v);
        else if (!strcmp(k, "--rows")) n.rows = atoi(v);
        else if (!strcmp(k, "--qscale")) n.qscale = atof(v);
        else if (!strcmp(k, "--json")) t.json = n.json = v;
        else if (!strcmp(k, "--variants")) {
            for (int i = 0; i < MLA_NVAR; i++) t.use[i] = 0;
            char buf[128];
            snprintf(buf, sizeof buf, "%s", v);
            for (char *s = strtok(buf, ","); s; s = strtok(NULL, ",")) {
                const int id = variant_of(s);
                if (id < 0) { usage(); return 2; }
                t.use[id] = 1;
            }
        } else { usage(); return 2; }
    }
    if (t.runs < 1 || t.runs > MAX_RUNS || t.threads < 1 || n.queries < 1) {
        usage(); return 2;
    }
    for (int i = 0; i < t.nT; i++) if (t.Ts[i] < 1) { usage(); return 2; }
#ifdef _OPENMP
    t.ncpu = omp_get_num_procs();
    omp_set_num_threads(t.threads);
#else
    t.ncpu = 1;
#endif
    t.threads = threads_now();
    if (counts) {
        /* the prefill shape by default; --C and --T replace it */
        CountOpt k;
        memset(&k, 0, sizeof k);
        k.nC = c_given ? t.nC : parse_list("0", k.Cs);
        k.nT = t_given ? t.nT : parse_list("1,16,64,256", k.Ts);
        if (c_given) memcpy(k.Cs, t.Cs, sizeof k.Cs);
        if (t_given) memcpy(k.Ts, t.Ts, sizeof k.Ts);
        k.vbuf_mb = t.vbuf_mb;
        k.json = t.json;
        return run_counts(&k);
    }
    return numerics ? run_numerics(&n) : run_time(&t);
}
