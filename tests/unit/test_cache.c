/* test_cache.c - the streaming expert cache, which nothing else tests.
 *
 * WHY THIS FILE EXISTS
 *   Every op fixture and all three oracle gates drive the RESIDENT expert bank
 *   (K3MoeW.w1/w3/w2). The streaming path (K3MoeW.src -> K3Cache) is reached only when
 *   running the real 1.56 TB checkpoint. So the component the entire project depends on
 *   had no test at all, and it showed: a batch-prefetch bug that handed one slot to
 *   several experts simultaneously passed 22/22 fixtures and all three gates, and was
 *   caught only because the real model emitted token 65 where it should have emitted
 *   2494. A test that cannot fail is worse than no test; a path with no test is worse
 *   still.
 *
 * WHAT IS CHECKED
 *   1 IDENTITY      every expert read back through the cache is byte-identical to the
 *                   same expert read straight off disk. This is the check the aliasing
 *                   bug fails.
 *   2 EQUIVALENCE   the batch prefetch and the serial path return the SAME bytes. The
 *                   prefetch is an optimisation; if it changes a byte it is wrong.
 *   3 PRESSURE      with a cache far smaller than the working set, so eviction runs
 *                   constantly and slots are recycled aggressively. The bug only
 *                   appeared under pressure, because with a roomy cache pick_victim
 *                   returns genuinely free slots and the aliasing never happens.
 *   4 ACCOUNTING    requests, hits and prefetch_reads stay mutually consistent.
 *   5 PIPELINE      with K3_EXPERT_PIPELINE=1 (see k3_cache.c), a second pass over the
 *                   same fixture: a pipelined batch is byte-exact against the same
 *                   ground truth as the serial path; duplicate and already-resident ids
 *                   in one batch are handled; a batch bigger than the slot count still
 *                   serves every expert via get(), just not all from the prefetch; free
 *                   and reset_stats called right after getmany, before any get(), drain
 *                   cleanly; a truncated shard leaves the short expert's slot empty and
 *                   get() failing for it alone.
 *
 * usage: test_cache <fixture_dir> [n_experts]
 *        fixture_dir comes from tools/make_cache_fixture.py
 */
#define _GNU_SOURCE             /* mkdtemp, ftruncate */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"     /* first: establishes Darwin feature macros */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>             /* ftruncate */
#ifdef _WIN32
#include <windows.h>
#endif

#include "k3.h"
#include "k3_cache.h"
#include "k3_load.h"
#include "k3_st.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

/* Read an expert straight from the store, bypassing the cache entirely. This is the
 * ground truth the cache is measured against. */
static unsigned char *direct_read(const K3St *st, int layer, int e, int64_t *nb)
{
    K3ExpertRef r;
    if (k3_expert_ref(st, layer, e, &r) != 0) return NULL;
    unsigned char *b = (unsigned char *)malloc((size_t)r.nbytes);
    if (!b) return NULL;
    if (k3_expert_load(st, &r, b) != r.nbytes) { free(b); return NULL; }
    *nb = r.nbytes;
    return b;
}

/* The three (packed, scale) pairs the cache hands out must point at bytes equal to the
 * direct read. Compare through the SAME offsets the kernels use, so a wrong pad or a
 * wrong slot base is caught, not just a wrong buffer. */
static int same_expert(const K3St *st, int layer, int e, const K3ExpertQ *q)
{
    int64_t nb = 0;
    unsigned char *truth = direct_read(st, layer, e, &nb);
    if (!truth) return 0;
    K3ExpertRef r;
    if (k3_expert_ref(st, layer, e, &r) != 0) { free(truth); return 0; }

    const unsigned char *got[6] = { q->p1, q->s1, q->p2, q->s2, q->p3, q->s3 };
    int ok = 1;
    for (int i = 0; i < 3 && ok; i++) {
        const unsigned char *tp = truth + r.m[i].p_off, *ts = truth + r.m[i].s_off;
        const int64_t pn = (int64_t)r.m[i].rows * r.m[i].pcols;
        const int64_t sn = (int64_t)r.m[i].rows * r.m[i].scols;
        if (memcmp(got[i * 2], tp, (size_t)pn) != 0) ok = 0;
        if (memcmp(got[i * 2 + 1], ts, (size_t)sn) != 0) ok = 0;
    }
    free(truth);
    return ok;
}

/* ---------------------------------------------------------------- pipeline mode --
 *
 * K3_EXPERT_PIPELINE=1 turns on the reader-thread pool (see k3_cache.c). Every case
 * below reuses the identity check above (same_expert), which is what makes it a
 * parity check against the non-pipeline path: both are compared to the same
 * straight-off-disk ground truth, so a byte pipeline mode gets wrong is a byte the
 * non-pipeline path was already proven right on.
 */

static void set_env(const char *name, const char *val)
{
#ifdef _WIN32
    _putenv_s(name, val ? val : "");
#else
    if (val) setenv(name, val, 1); else unsetenv(name);
#endif
}

/* A temp directory, portable enough for this test: POSIX mkdtemp on Linux/macOS, a
 * small CreateDirectoryA retry loop on Windows (mirrors test_trunk.c). */
static int make_tmpdir(char *out, size_t cap)
{
#if defined(_WIN32)
    char base[MAX_PATH];
    DWORD blen = GetTempPathA((DWORD)sizeof base, base);
    if (blen == 0 || blen >= sizeof base) return -1;
    for (unsigned attempt = 0; attempt < 1000; attempt++) {
        snprintf(out, cap, "%sk3_test_cache_%lu_%u", base,
                 (unsigned long)GetCurrentProcessId(), attempt);
        if (CreateDirectoryA(out, NULL)) return 0;
    }
    return -1;
#else
    snprintf(out, cap, "/tmp/k3_test_cache_XXXXXX");
    return mkdtemp(out) ? 0 : -1;
#endif
}

/* Byte-for-byte file copy, no shelling out. */
static int copy_file(const char *src, const char *dst)
{
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }
    char buf[65536];
    size_t n;
    int ok = 1;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { ok = 0; break; }
    if (ferror(in)) ok = 0;
    fclose(in); fclose(out);
    return ok ? 0 : -1;
}

/* Grow the budget until init accepts. A cache needs room for whole, O_DIRECT-widened
 * slots, so a budget sized only off the raw expert byte count can come up short by a
 * slot; the fixture experts are small enough that this happens routinely. Mirrors the
 * growing loop main() already uses to size its own cache above. */
static int init_grown(K3Cache *cache, const K3St *st, const K3Cfg *cfg, int64_t unit)
{
    int64_t budget = unit * 8;
    for (int tries = 0; tries < 14; tries++, budget *= 2)
        if (k3_cache_init(cache, st, cfg, budget) == 0) return 1;
    return 0;
}

/* (a) a pipelined batch, consumed in the same order getmany was given, must be
 * byte-exact -- exactly the check the serial and non-pipeline-batch paths are held
 * to above. (b) duplicate ids and an already-resident id together in one batch must
 * not corrupt or drop anything. */
static void pipeline_batch_case(K3St *st, K3Cache *cache, int NE, int topk)
{
    int bad = 0, batches = 0;
    for (int start = 0; start + topk <= NE; start += topk) {
        int ids[16];
        for (int j = 0; j < topk; j++) ids[j] = start + j;
        cache->src.getmany(&cache->src, 0, ids, topk);
        batches++;
        for (int j = 0; j < topk; j++) {
            K3ExpertQ q;
            if (cache->src.get(&cache->src, 0, ids[j], &q) != 0) { bad++; continue; }
            if (!same_expert(st, 0, ids[j], &q)) bad++;
        }
    }
    char b[96];
    snprintf(b, sizeof b, "%d batches of %d, %d wrong", batches, topk, bad);
    ck(bad == 0, "pipeline: batch prefetch is byte-exact", b);

    /* Warm expert 0 first, so the mixed batch below also names an already-resident
     * id, alongside expert 1 named twice. */
    K3ExpertQ q0;
    cache->src.get(&cache->src, 0, 0, &q0);
    int mix[6] = { 0, 1, 1, 2, 0, 1 };
    cache->src.getmany(&cache->src, 0, mix, 6);
    int bad_mix = 0;
    for (int j = 0; j < 6; j++) {
        K3ExpertQ q;
        if (cache->src.get(&cache->src, 0, mix[j], &q) != 0) { bad_mix++; continue; }
        if (!same_expert(st, 0, mix[j], &q)) bad_mix++;
    }
    ck(bad_mix == 0, "pipeline: duplicate/resident ids in one batch", NULL);
}

/* (c) a batch bigger than the slot count. Reservation runs out of slots partway
 * through, so getmany launches fewer reads than the batch asked for -- the
 * remaining experts must still come back correct through get()'s own miss path,
 * with nothing dropped overall. */
static void pipeline_short_cache_case(K3St *st, const K3Cfg *cfg, int64_t expert_nbytes)
{
    K3Cache cache;
    if (!init_grown(&cache, st, cfg, expert_nbytes)) {
        ck(0, "pipeline: short cache init", NULL); return;
    }

    int n = cache.nslot + 4;               /* always more unique ids than slots */
    if (n > cfg->n_experts) n = cfg->n_experts;
    if (n > 16) n = 16;
    int ids[16];
    for (int i = 0; i < n; i++) ids[i] = i;
    cache.src.getmany(&cache.src, 0, ids, n);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        K3ExpertQ q;
        if (cache.src.get(&cache.src, 0, ids[i], &q) != 0) { bad++; continue; }
        if (!same_expert(st, 0, ids[i], &q)) bad++;
    }
    char b[96];
    snprintf(b, sizeof b, "%d slots, %d ids requested, %d wrong", cache.nslot, n, bad);
    ck(bad == 0, "pipeline: oversized batch on a short cache, no drop", b);
    k3_cache_free(&cache);
}

/* (d) k3_cache_free and k3_cache_reset_stats called right after getmany, before any
 * get(), must drain the workers cleanly rather than race or hang. */
static void pipeline_drain_case(K3St *st, const K3Cfg *cfg, int64_t expert_nbytes)
{
    K3Cache cache;
    if (!init_grown(&cache, st, cfg, expert_nbytes)) {
        ck(0, "pipeline: drain-on-reset cache init", NULL);
    } else {
        int ids[4] = { 0, 1, 2, 3 };
        cache.src.getmany(&cache.src, 0, ids, 4);
        k3_cache_reset_stats(&cache);          /* drains before touching a counter */
        int bad = 0;
        for (int i = 0; i < 4; i++) {
            K3ExpertQ q;
            if (cache.src.get(&cache.src, 0, ids[i], &q) != 0) { bad++; continue; }
            if (!same_expert(st, 0, ids[i], &q)) bad++;
        }
        ck(bad == 0, "pipeline: reset_stats right after getmany drains clean", NULL);
        k3_cache_free(&cache);
    }

    K3Cache cache2;
    if (!init_grown(&cache2, st, cfg, expert_nbytes)) {
        ck(0, "pipeline: drain-on-free cache init", NULL);
    } else {
        int ids[4] = { 4, 5, 6, 7 };
        cache2.src.getmany(&cache2.src, 0, ids, 4);
        k3_cache_free(&cache2);        /* reaching the next line is the assertion */
        ck(1, "pipeline: free right after getmany drains clean", NULL);
    }
}

/* One K3Cfg for the failed-read case, which builds its own K3St on a temp dir and so
 * cannot simply reuse the caller's. */
static K3Cfg *cfg_here(int NE)
{
    static K3Cfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_layers = 1; cfg.n_experts = NE; cfg.topk = 4;
    return &cfg;
}

/* (e) an injected read failure: copy the fixture shard to a temp file and truncate
 * it mid-way through the LAST expert's bytes (every expert is the same on-disk
 * size, and they are laid out back-to-back in id order, so this shortens exactly
 * one expert). That slot must come back empty and get() must fail for it alone. */
static void pipeline_failed_read_case(const char *fixture_dir, int NE,
                                      int64_t expert_nbytes)
{
    char tmpdir[512];
    if (make_tmpdir(tmpdir, sizeof tmpdir) != 0) {
        ck(0, "pipeline: failed-read tmpdir", "mkdtemp failed"); return;
    }
    char src[600], dst[600];
    snprintf(src, sizeof src, "%s/model-00001-of-00001.safetensors", fixture_dir);
    snprintf(dst, sizeof dst, "%s/model-00001-of-00001.safetensors", tmpdir);
    if (copy_file(src, dst) != 0) {
        ck(0, "pipeline: failed-read fixture copy", NULL); rmdir(tmpdir); return;
    }

    /* Open on the INTACT copy first: k3_st_open validates every tensor's declared
     * span against the file size, so opening on an already-truncated file would be
     * refused at open time rather than exercising a failed READ. Truncate only
     * after the header is parsed and the fd is held open, the same order
     * test_trunk.c uses for its own truncated-read case. */
    K3St st;
    if (k3_st_open(&st, tmpdir) != 0) {
        ck(0, "pipeline: failed-read fixture open", NULL);
        remove(dst); rmdir(tmpdir); return;
    }

    int wfd = open(dst, O_WRONLY);
    if (wfd < 0) {
        ck(0, "pipeline: failed-read open for truncate", NULL);
        k3_st_close(&st); remove(dst); rmdir(tmpdir); return;
    }
    struct stat sb;
    fstat(wfd, &sb);
    const off_t cut = sb.st_size - (off_t)(expert_nbytes / 2);
    const int trunc_ok = ftruncate(wfd, cut) == 0;
    close(wfd);
    if (!trunc_ok) {
        ck(0, "pipeline: failed-read ftruncate", NULL);
        k3_st_close(&st); remove(dst); rmdir(tmpdir); return;
    }

    K3Cache cache;
    if (!init_grown(&cache, &st, cfg_here(NE), expert_nbytes)) {
        ck(0, "pipeline: failed-read cache init", NULL);
    } else {
        const int bad_id = NE - 1;
        const int good[3] = { 0, 1, 2 };
        int ids[4] = { good[0], good[1], bad_id, good[2] };
        cache.src.getmany(&cache.src, 0, ids, 4);

        K3ExpertQ q;
        ck(cache.src.get(&cache.src, 0, bad_id, &q) == -1,
           "pipeline: short read leaves the slot empty", NULL);

        int ok_others = 1;
        for (int i = 0; i < 3; i++) {
            K3ExpertQ q2;
            if (cache.src.get(&cache.src, 0, good[i], &q2) != 0 ||
                !same_expert(&st, 0, good[i], &q2))
                ok_others = 0;
        }
        ck(ok_others, "pipeline: other experts in the batch still serve", NULL);
        k3_cache_free(&cache);
    }
    k3_st_close(&st);
    remove(dst);
    rmdir(tmpdir);
}

static void run_pipeline_tests(const char *dir, int NE, const K3Cfg *cfg, int64_t nbytes)
{
    printf("\npipeline mode (K3_EXPERT_PIPELINE=1)\n\n");
    set_env("K3_EXPERT_PIPELINE", "1");

    K3St st;
    if (k3_st_open(&st, dir) != 0) {
        ck(0, "pipeline: reopen fixture", NULL);
    } else {
        K3Cache cache;
        if (!init_grown(&cache, &st, cfg, nbytes)) {
            ck(0, "pipeline: cache init", NULL);
        } else {
            ck(cache.pipeline != 0, "pipeline: cache.pipeline is set", NULL);
            pipeline_batch_case(&st, &cache, NE, cfg->topk);
            k3_cache_free(&cache);
        }
        pipeline_short_cache_case(&st, cfg, nbytes);
        pipeline_drain_case(&st, cfg, nbytes);
        k3_st_close(&st);
    }

    pipeline_failed_read_case(dir, NE, nbytes);

    set_env("K3_EXPERT_PIPELINE", NULL);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "../fixtures/cache";
    const int NE = argc > 2 ? atoi(argv[2]) : 24;

    K3St st;
    if (k3_st_open(&st, dir) != 0) {
        fprintf(stderr, "TEST ABORTED: cannot open %s\n"
                        "  build it with: python3 tools/make_cache_fixture.py\n", dir);
        return 2;
    }
    printf("streaming expert cache, %d tensors from %d shard(s)\n\n", st.nt, st.nshard);

    K3Cfg c; memset(&c, 0, sizeof c);
    c.n_layers = 1; c.n_experts = NE; c.topk = 4;

    /* Size the budget by asking, not by arithmetic. The cache rounds a slot up to an
     * O_DIRECT-widened, page-aligned size, so for these deliberately tiny fixture
     * experts the requested budget and the usable slot count part company badly. Grow
     * until init accepts, then confirm PRESSURE remains: fewer slots than experts,
     * so eviction actually runs. A roomy cache hides slot-recycling bugs entirely. */
    K3ExpertRef probe;
    if (k3_expert_ref(&st, 0, 0, &probe) != 0) { fprintf(stderr, "no expert 0\n"); return 2; }
    K3Cache cache;
    int ok_init = 0;
    for (int64_t budget = probe.nbytes * 8; budget <= probe.nbytes * 4096; budget *= 2) {
        if (k3_cache_init(&cache, &st, &c, budget) == 0) { ok_init = 1; break; }
    }
    if (!ok_init) { fprintf(stderr, "cache init failed at every budget\n"); return 2; }
    { char b[80]; snprintf(b, sizeof b, "%d slots for %d experts, top-%d",
                           cache.nslot, NE, c.topk);
      ck(cache.nslot >= c.topk + 1 && cache.nslot < NE, "cache under pressure", b); }

    /* ---- 1+3: serial path, every expert, under eviction pressure ---- */
    int bad = 0;
    for (int pass = 0; pass < 3; pass++)
        for (int e = 0; e < NE; e++) {
            K3ExpertQ q;
            if (cache.src.get(&cache.src, 0, e, &q) != 0) { bad++; continue; }
            if (!same_expert(&st, 0, e, &q)) bad++;
        }
    { char b[64]; snprintf(b, sizeof b, "%d of %d reads wrong", bad, 3 * NE);
      ck(bad == 0, "serial reads are byte-exact", b); }

    /* ---- 2: batch prefetch must agree with the serial path ----
     * This is the check the aliasing bug fails. Ask for a whole top-k at once, with the
     * cache too small to hold the previous batch, then verify EVERY expert. */
    k3_cache_reset_stats(&cache);
    int bad2 = 0, batches = 0;
    if (!cache.src.getmany) {
        ck(0, "batch prefetch present", "getmany is NULL");
    } else {
        for (int start = 0; start + c.topk <= NE; start += c.topk) {
            int ids[16];
            for (int j = 0; j < c.topk; j++) ids[j] = start + j;
            cache.src.getmany(&cache.src, 0, ids, c.topk);
            batches++;
            for (int j = 0; j < c.topk; j++) {
                K3ExpertQ q;
                if (cache.src.get(&cache.src, 0, ids[j], &q) != 0) { bad2++; continue; }
                if (!same_expert(&st, 0, ids[j], &q)) bad2++;
            }
        }
        char b[96];
        snprintf(b, sizeof b, "%d batches of %d, %d wrong", batches, c.topk, bad2);
        ck(bad2 == 0, "batch prefetch is byte-exact", b);

        /* ACCOUNTING, checked HERE and only here. The loop above is exactly the pattern
         * k3_moe uses -- prefetch a top-k, then consume that same top-k -- and for that
         * pattern every prefetched expert is still resident when get() asks, so it is
         * recorded as a hit and prefetch_reads can never exceed hits. If it does, the
         * report's "true resident hit rate" underflows and the whole figure is a lie.
         * The mixed test below deliberately prefetches sets it never consumes, so the
         * invariant does NOT hold there and asserting it would be wrong. */
        char a[128];
        snprintf(a, sizeof a, "requests %llu, hits %llu, prefetch %llu",
                 (unsigned long long)(cache.hits + cache.misses),
                 (unsigned long long)cache.hits,
                 (unsigned long long)cache.prefetch_reads);
        ck(cache.prefetch_reads <= cache.hits, "prefetch_reads <= hits", a);
    }

    /* ---- 2b: interleaving the two paths must not corrupt either ---- */
    k3_cache_reset_stats(&cache);
    int bad3 = 0;
    for (int e = 0; e < NE; e++) {
        if (cache.src.getmany && (e % 2) == 0) {
            int ids[4];
            for (int j = 0; j < 4; j++) ids[j] = (e + j) % NE;
            cache.src.getmany(&cache.src, 0, ids, 4);
        }
        K3ExpertQ q;
        if (cache.src.get(&cache.src, 0, e, &q) != 0) { bad3++; continue; }
        if (!same_expert(&st, 0, e, &q)) bad3++;
    }
    { char b[64]; snprintf(b, sizeof b, "%d of %d wrong", bad3, NE);
      ck(bad3 == 0, "mixed batch and serial", b); }

    k3_cache_free(&cache);

    /* Put the requested resident keys at the LRU end. Prefetching a new key must not
     * evict them and then read them again in the very same batch. */
    const int64_t stride = (probe.nbytes + 3 * K3_ST_ALIGN - 1)
                           & ~(int64_t)(K3_ST_ALIGN - 1);
    if (k3_cache_init(&cache, &st, &c, 8 * stride)) return 2;
    char profile[4096];
    snprintf(profile, sizeof profile, "%s/experts.profile", dir);
    ck(k3_cache_check_profile(profile, 1) == 0 &&
       k3_cache_check_profile(profile, 4) == -1,
       "profile preflight without checkpoint", NULL);
    ck(k3_cache_load_profile(&cache, profile, 1) == 0 && cache.profile_pins == 1 &&
       cache.bytes_read == 0 && cache.slot_of[0] == -1,
       "profile pins without preloading", NULL);
    ck(k3_cache_load_profile(&cache, profile, 4) == -1 && cache.profile_pins == 1,
       "failed profile leaves policy intact", NULL);
    K3ExpertQ q;
    for (int e = 0; e < 8; e++)
        if (cache.src.get(&cache.src, 0, e, &q)) return 2;
    ck(k3_cache_load_profile(&cache, profile, 1) == -1,
       "profile cannot change a live cache", NULL);
    k3_cache_reset_stats(&cache);
    const int ids[4] = {8, 0, 1, 2};
    ck(cache.src.getmany(&cache.src, 0, ids, 4) == 1,
       "batch keeps its resident members", "only expert 8 needs a load");
    int exact = 1;
    for (int i = 0; i < 4; i++)
        if (cache.src.get(&cache.src, 0, ids[i], &q) ||
            !same_expert(&st, 0, ids[i], &q)) exact = 0;
    ck(exact && cache.bytes_read == (uint64_t)probe.nbytes,
       "mixed-residency batch byte-exact", "one payload read");
    ck(cache.demand_requests == 4 && cache.demand_reuses == 3,
       "cold prefetch is not reuse", "3 reuses, 4 successful requests");
    ck(cache.src.getmany(&cache.src, 0, ids, 4) == 0,
       "resident batch needs no reads", NULL);
    int clear = 1;
    for (int e = 0; e < NE; e++) if (cache.requested[e]) clear = 0;
    ck(clear, "batch marks clear on no-work return", NULL);

    const int later = 9;
    cache.src.getmany(&cache.src, 0, &later, 1);
    k3_cache_reset_stats(&cache);
    cache.src.get(&cache.src, 0, later, &q);
    ck(cache.demand_requests == 1 && cache.demand_reuses == 0,
       "reset preserves first-use state", "prefetch belongs to an earlier window");
    cache.src.get(&cache.src, 0, later, &q);
    ck(cache.demand_requests == 2 && cache.demand_reuses == 1,
       "second consumption is reuse", NULL);
    ck(!k3_cache_pin(&cache, -1, NE, 1) &&
       k3_cache_prefetch(&cache, 0, NE) == -1 &&
       cache.src.getmany(&cache.src, 1, ids, 4) == -1 &&
       cache.src.getmany(&cache.src, 0, NULL, 4) == -1,
       "invalid cache coordinates refused", NULL);
    ck(k3_cache_pin(&cache, 0, 0, 1) && k3_cache_pin(&cache, 0, 1, 1) &&
       k3_cache_pin(&cache, 0, 2, 1) && !k3_cache_pin(&cache, 0, 9, 1),
       "pins leave topk+1 evictable slots", NULL);
    for (int e = 3; e < NE; e++)
        if (cache.src.get(&cache.src, 0, e, &q) || !same_expert(&st, 0, e, &q)) exact = 0;
    ck(exact && cache.src.resident(&cache.src, 0, 0, &q) &&
       same_expert(&st, 0, 0, &q), "pinned bytes survive pressure", NULL);
    k3_cache_pin(&cache, 0, 0, 0);
    k3_cache_pin(&cache, 0, 1, 0);
    k3_cache_pin(&cache, 0, 2, 0);
    for (int e = 1; e < NE; e++)
        if (cache.src.get(&cache.src, 0, e, &q)) exact = 0;
    ck(exact && cache.src.resident(&cache.src, 0, 0, &q) &&
       same_expert(&st, 0, 0, &q), "lazy profile pin survives pressure", NULL);
    const int union_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, -1, NE};
    cache.src.getmany(&cache.src, 0, union_ids,
                      (int)(sizeof union_ids / sizeof union_ids[0]));
    clear = 1;
    for (int e = 0; e < NE; e++) if (cache.requested[e]) clear = 0;
    ck(clear, "oversized batch releases every mark", "including duplicate/invalid IDs");
    for (int e = 1; e <= 12; e++)
        if (cache.src.get(&cache.src, 0, e, &q) || !same_expert(&st, 0, e, &q)) exact = 0;
    ck(exact, "oversized batch demand fallback exact", NULL);
    k3_cache_free(&cache);

    run_pipeline_tests(dir, NE, &c, probe.nbytes);

    k3_st_close(&st);
    printf("\n%s\n", g_fail ? "CACHE TESTS FAILED" : "CACHE TESTS PASSED");
    return g_fail ? 1 : 0;
}
