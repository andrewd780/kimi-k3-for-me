/* k3_run.c - run the REAL Kimi K3, all 93 layers, from the released checkpoint.
 *
 * WHAT THIS IS
 *   The full engine: safetensors index over 96 shards, resident trunk bound by name,
 *   routed experts streamed from disk through an LRU cache and multiplied straight out
 *   of MXFP4. Greedy decode. Token ids in, token ids out.
 *
 * MEMORY. The banner this program prints before allocating is a PLAN, not a measurement.
 *   It reports requested budgets rather than actual reservations, and in practice it
 *   OVERSTATES: across the 12-rung ladder in docs/data/ the planned total exceeded
 *   measured peak RSS by 0.13-1.84 GB, because both budgets round down to whole slots and
 *   that rounding outweighs the safetensors index it omits. Quote the "PEAK RSS" line
 *   instead, which comes from
 *   getrusage after the run. Fully resident, the weights are 108.81 GB of bf16 trunk plus
 *   4.70 GB of embed and lm_head, so 113.49 GB; streamed, the resident set is whatever
 *   budget is given, down to about 8.2 GB. The 1.45 TB of routed experts is never
 *   resident at any budget.
 *
 * THIS ENGINE IS I/O BOUND at small budgets and roughly balanced at large ones. The
 *   measured I/O share runs 40.9%-60.6% across the 12-rung ladder (docs/data/), dropping
 *   below 50% at 96 GB and above. The "I/O share" line printed at the end of every run
 *   reports it for that run. Going faster still means moving fewer bytes before it means
 *   computing less, which is why docs/TUNING.md is mostly about allocation.
 *
 * DECODE STRATEGY
 *   By default each step re-runs the whole prefix rather than carrying state forward.
 *   That is O(T^2), but it is the path the full-model oracle validates in
 *   tests/unit/k3_model.c. --incremental switches to prefill-then-one-token-at-a-time,
 *   carrying the KDA recurrent state and an MLA KV cache. GATE 3 of the tiny-model
 *   oracle requires it to produce the SAME tokens as full recompute, so the equivalence
 *   is tested rather than assumed. Context is limited by the MLA KV cache
 *   (~2.37 MB/position, or 0.055 with --kv-latent, which caches the compressed latent
 *   and rebuilds k and v on use), not by array sizes; the engine computes the
 *   requirement up front and refuses the run if it will not fit.
 *
 * COMMAND LINE
 *   usage() below is the single source of truth for options and defaults; `k3 --help`
 *   prints it. It is not duplicated here, because a second copy is a second thing to
 *   keep correct and the copy is the one that goes stale.
 */
#define _POSIX_C_SOURCE 200809L
/* _POSIX_C_SOURCE alone hides the BSD rusage fields, ru_maxrss among them, from
 * <sys/resource.h> on Darwin. peak_rss_bytes() below needs it. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <math.h>
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"
#include "k3_trunk.h"
#include "k3_tok.h"   /* text in/out; the --ids path never touches it */
#include "k3_cfg.h"   /* read the checkpoint's own config rather than assuming it */
#include "k3_quality.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void human(double b, char *o, size_t n)
{
    const char *u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0; while (b >= 1000.0 && i < 4) { b /= 1000.0; i++; }
    snprintf(o, n, "%.2f %s", b, u[i]);
}

/* The released constants, kept ONLY as a fallback for runs against a shard directory
 * that has no config.json (partial fixtures, hand-assembled trunks). Every value here
 * matches the released config.json, but a hardcoded table cannot notice a checkpoint
 * revision -- so k3_cfg_load_file() is preferred whenever a config is present, and this
 * path announces itself loudly rather than passing for the real thing. */
static void real_cfg_hardcoded(K3Cfg *c, int *fa)
{
    memset(c, 0, sizeof *c);
    c->hidden = 7168;  c->n_layers = 93;   c->vocab = 163840; c->rms_eps = 1e-5f;
    c->kda_heads = 96; c->kda_head_dim = 128; c->conv_k = 4;  c->gate_lb = -5.0f;
    c->n_heads = 96;   c->q_lora = 1536;   c->kv_lora = 512;
    c->qk_nope = 128;  c->qk_rope = 64;    c->v_head = 128;   c->mla_out_gate = 1;
    c->n_experts = 896; c->topk = 16;      c->n_shared = 2;
    c->latent = 3584;  c->moe_inter = 3072; c->routed_scale = 1.0f;
    c->moe_renorm = 1; c->latent_norm = 1;
    c->first_dense = 1; c->dense_inter = 33792;
    c->attn_res_block = 12;
    c->situ_b1 = 4.0f; c->situ_b2 = 25.0f;
    int n = 0;
    for (int i = 4; i <= 93; i += 4) fa[n++] = i;     /* config lists are ONE-based */
    fa[n++] = 93;
    c->n_full_attn = n; c->full_attn = fa;
}

/* Prefer the checkpoint's own config; fall back only when there is none.
 * cfg_path may be NULL, in which case <shard_dir>/config.json is tried.
 * Returns 1 on success, 0 if a config was found but could not be trusted -- and in
 * that case the caller MUST abort rather than fall back: a config that was found but
 * could not be parsed is evidence that the checkpoint is not what the fallback table
 * describes, which is exactly when the fallback is most dangerous. */
static int real_cfg(K3Cfg *c, int *fa, int fa_max,
                    const char *shard_dir, const char *cfg_path)
{
    char guess[4096];
    if (!cfg_path) {
        snprintf(guess, sizeof guess, "%s/config.json", shard_dir);
        FILE *probe = fopen(guess, "rb");
        if (probe) { fclose(probe); cfg_path = guess; }
    }
    if (cfg_path) return k3_cfg_load_file(c, fa, fa_max, cfg_path);

    real_cfg_hardcoded(c, fa);
    printf("config: NO config.json found under %s\n"
           "        falling back to the built-in Kimi K3 constants (93 layers, 24 MLA).\n"
           "        These match the released checkpoint but are NOT read from it; pass\n"
           "        --config PATH to validate against the real file.\n", shard_dir);
    return 1;
}

static int argmax_(const float *v, int n)
{ int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

static void json_string(FILE *f, const char *s)
{
    if (!s) { fputs("null", f); return; }
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\b': fputs("\\b", f); break;
        case '\f': fputs("\\f", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", (unsigned)*p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

/* ------------------------------------------------------- conversation state ----
 * Everything the engine carries between tokens, on disk. The point is turn two of a
 * conversation: without this, resuming re-reads the whole prompt through all 93 layers,
 * which on a streamed trunk costs minutes; with it, a resumed session pays only for the
 * tokens actually new.
 *
 * Three things are carried, and only three: the KDA recurrent matrices plus ShortConv
 * history (fixed size, independent of context), the MLA KV cache in whichever layout
 * the run chose (kvpp in the header says which, and a mismatch is refused on load), and
 * the shared rope rows. The AttnRes block buffer is NOT carried because forward() clears
 * it on entry and rebuilds it from the layer outputs every pass; saving it would be
 * saving scratch.
 *
 * The KV cache is stored position-major inside each MLA layer's slice, so only the
 * OCCUPIED positions are written and a resumed run may size its cache differently. The
 * header carries a config fingerprint: restoring state built by a different architecture
 * would produce fluent, wrong output with nothing to indicate it, which is the one
 * failure mode this engine refuses to have. */
#define K3_STATE_MAGIC "K3ST"
#define K3_STATE_VER   2      /* 2 adds the layer-loop record; 1 is read as "no loop" */

typedef struct {
    char    magic[4];
    int32_t version;
    int32_t fp[12];        /* config fingerprint */
    int32_t n_bound, n_mla, cached, nseq;
    int64_t kper;          /* KDA+conv floats per layer */
    int64_t kvpp, ropepp;  /* KV / rope floats per position, per MLA layer */
} K3StateHdr;

/* Version 2: the layer loop the state was computed under, right after the header. A loop
 * gives every executed position its own KDA slot and MLA rows (nv slots, not n_bound),
 * and adaptive depth carries how many positions each pass has cached and how many passes
 * each position ran; restoring under another loop would read the slots at the wrong
 * meaning. count == 0 is no loop (and what a version-1 file holds). */
typedef struct {
    int32_t a, b, count, nv;
    double  eps;             /* < 0: fixed loop or none */
} K3StateLoop;

static void k3_state_fp(const K3Cfg *c, int32_t *fp)
{
    fp[0] = c->hidden;      fp[1] = c->n_layers;  fp[2]  = c->vocab;
    fp[3] = c->kda_heads;   fp[4] = c->kda_head_dim; fp[5] = c->conv_k;
    fp[6] = c->n_heads;     fp[7] = c->qk_nope;   fp[8]  = c->qk_rope;
    fp[9] = c->v_head;      fp[10] = c->n_experts; fp[11] = c->topk;
}

#define K3_SPEC_MAX 8

/* Positions whose logits one lm_head pass serves when a run needs EVERY position's logits
 * (--tf-check, --score-prompt, and a --spec verify sweep of spec_n + 1 <= K3_SPEC_MAX + 1
 * positions). One pass of the batched kernel covers 16 positions at hidden 7168 on x86
 * (see k3_mm_pass in k3_ops.c; on NEON, K3_MM_TB = 2, a pass covers 18), so for a
 * resident head a larger block would buy no fewer passes on x86. A streamed head
 * (--stream-lm-head, --ultra-low-memory) is read from disk once
 * per block, 16x less than once per position; a larger block would cut that further but
 * costs a vocab row (0.66 MB) per position, and 16 keeps the buffer at 10.5 MB. */
#define K3_LOGIT_ROWS 16
/* Longest-suffix n-gram drafting for --spec: if the last n ids (n=3, then 2) already
 * appeared earlier in the sequence, propose the ids that followed them there. Costs
 * nothing when it misses: no draft means the step runs exactly as without --spec. The
 * drafts are PROPOSALS only; batched greedy verification accepts precisely the prefix
 * the model itself would have emitted, so the output stream is identical to serial
 * decode by construction, and the A/B gate checks it. */
/* Reads only the header, so the caller can size buffers before committing to a load. */
static int k3_state_peek(const char *path, K3StateHdr *hd, K3StateLoop *lp)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const size_t got = fread(hd, 1, sizeof *hd, f);
    size_t gotl = sizeof *lp;
    memset(lp, 0, sizeof *lp);
    lp->a = lp->b = -1; lp->nv = got == sizeof *hd ? hd->n_bound : 0; lp->eps = -1.0;
    if (got == sizeof *hd && hd->version == 2) gotl = fread(lp, 1, sizeof *lp, f);
    fclose(f);
    if (got != sizeof *hd || memcmp(hd->magic, K3_STATE_MAGIC, 4) != 0 || gotl != sizeof *lp) {
        fprintf(stderr, "%s is not a k3 state file\n", path);
        return -1;
    }
    if (hd->version != 1 && hd->version != K3_STATE_VER) {
        fprintf(stderr, "%s is state version %d, this build reads 1 and %d\n",
                path, hd->version, K3_STATE_VER);
        return -1;
    }
    return 0;
}

/* The saved ids of a state file whose header k3_state_peek has read, into ids[0 .. nseq). */
static int k3_state_read_ids(const char *path, const K3StateHdr *hd, int *ids)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const long off = (long)sizeof *hd + (hd->version >= 2 ? (long)sizeof(K3StateLoop) : 0L);
    const int ok = hd->nseq >= 0 && fseek(f, off, SEEK_SET) == 0
                && fread(ids, sizeof(int), (size_t)hd->nseq, f) == (size_t)hd->nseq;
    fclose(f);
    if (!ok) fprintf(stderr, "%s is truncated\n", path);
    return ok ? 0 : -1;
}

static int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr *hd,
                         const K3StateLoop *lp, const K3StateLoop *run,
                         int *seq, float *ks, float *kvc, float *ropec,
                         int n_bound, int n_mla, int kv_cap,
                         int64_t kvpp, int64_t ropepp, int *pcached, int *passes)
{
    if (lp->count != run->count || (run->count &&
        (lp->a != run->a || lp->b != run->b || lp->nv != run->nv ||
         memcmp(&lp->eps, &run->eps, sizeof lp->eps) != 0))) {
        char sl[96] = "no layer loop (--no-loop)", rl[96] = "no layer loop (--no-loop)";
        if (lp->count) snprintf(sl, sizeof sl, "--loop-layers %d-%d --loop-count %d%s%.9g",
                                lp->a, lp->b, lp->count, lp->eps >= 0 ? " --loop-exit " : "",
                                lp->eps >= 0 ? lp->eps : 0.0);
        if (run->count) snprintf(rl, sizeof rl, "--loop-layers %d-%d --loop-count %d%s%.9g",
                                 run->a, run->b, run->count, run->eps >= 0 ? " --loop-exit " : "",
                                 run->eps >= 0 ? run->eps : 0.0);
        fprintf(stderr, "REFUSING: %s was computed under %s, this run uses %s.\n"
                        "  Its per-layer state would be read at the wrong meaning; run with the "
                        "loop it was saved under.\n", path, sl, rl);
        return -1;
    }
    int32_t fp[12];
    k3_state_fp(c, fp);
    if (memcmp(fp, hd->fp, sizeof fp) != 0) {
        fprintf(stderr, "REFUSING: %s was written by a different model architecture.\n"
                        "  Restoring it would produce fluent, wrong output.\n", path);
        return -1;
    }
    if (hd->n_bound != n_bound || hd->n_mla != n_mla) {
        fprintf(stderr, "REFUSING: %s holds %d bound layers and %d MLA layers, "
                        "this run has %d and %d\n",
                path, hd->n_bound, hd->n_mla, n_bound, n_mla);
        return -1;
    }
    /* The KV layout is not in the fingerprint, because it is a run-time choice rather
     * than a property of the model. It still has to match: an expanded state read into
     * a latent cache would be read at the wrong stride and produce fluent, wrong
     * output, which is the one failure mode this file exists to prevent. */
    if (hd->kvpp != kvpp || hd->ropepp != ropepp) {
        const int saved_latent = hd->kvpp < kvpp;
        fprintf(stderr, "REFUSING: %s holds %lld KV floats per position per MLA layer, "
                        "this run's cache holds %lld.\n"
                        "  The file was saved %s --kv-latent; %s the flag to match it.\n",
                path, (long long)hd->kvpp, (long long)kvpp,
                saved_latent ? "WITH" : "WITHOUT", saved_latent ? "add" : "drop");
        return -1;
    }
    if (hd->cached > kv_cap) {
        fprintf(stderr, "REFUSING: %s holds %d positions, this run's KV cache is %d.\n"
                        "  Raise --gen or shorten the prompt.\n", path, hd->cached, kv_cap);
        return -1;
    }
    /* A saved sequence is the cached positions plus at most one pending id (the last one
     * generated, not yet fed). Anything else was not written by k3_state_save. */
    if (hd->cached < 0 || hd->nseq < hd->cached || hd->nseq > hd->cached + 1) {
        fprintf(stderr, "REFUSING: %s holds %d ids for %d cached positions; a state file "
                        "holds the cached positions and at most one pending id\n",
                path, hd->nseq, hd->cached);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const long off = (long)sizeof *hd + (hd->version >= 2 ? (long)sizeof *lp : 0L);
    if (fseek(f, off, SEEK_SET) != 0) { fclose(f); return -1; }

    int rc = 0;
    if (fread(seq, sizeof(int), (size_t)hd->nseq, f) != (size_t)hd->nseq) rc = -1;
    if (!rc && lp->count) {
        /* a fixed loop's run keeps neither array; the file's are zeros then */
        int scratch_pc[9];
        const size_t np = (size_t)lp->count + 1;
        if (fread(pcached ? pcached : scratch_pc, sizeof(int), np, f) != np) rc = -1;
        for (int i = 0; !rc && i < hd->nseq; i++) {
            int v;
            if (fread(&v, sizeof v, 1, f) != 1) rc = -1;
            else if (passes) passes[i] = v;
        }
    }
    const size_t slots = (size_t)(lp->count ? lp->nv : n_bound);
    if (!rc && fread(ks, sizeof(float), (size_t)hd->kper * slots, f)
               != (size_t)hd->kper * slots) rc = -1;
    /* Position-major inside each layer slice, so a differently-sized destination cache
     * is written slice by slice rather than as one block. */
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = kvc + (size_t)mi * kv_cap * hd->kvpp;
        const size_t n = (size_t)hd->cached * hd->kvpp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = ropec + (size_t)mi * kv_cap * hd->ropepp;
        const size_t n = (size_t)hd->cached * hd->ropepp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    fclose(f);
    if (rc) fprintf(stderr, "%s is truncated\n", path);
    return rc;
}

static int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                         const float *ks, const float *kvc, const float *ropec,
                         int n_bound, int n_mla, int kv_cap, int cached,
                         int64_t kper, int64_t kvpp, int64_t ropepp,
                         const K3StateLoop *lp, const int *pcached, const int *passes)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    K3StateHdr hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, K3_STATE_MAGIC, 4);
    hd.version = K3_STATE_VER;
    k3_state_fp(c, hd.fp);
    hd.n_bound = n_bound; hd.n_mla = n_mla; hd.cached = cached; hd.nseq = nseq;
    hd.kper = kper; hd.kvpp = kvpp; hd.ropepp = ropepp;

    int rc = 0;
    if (fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && fwrite(lp, sizeof *lp, 1, f) != 1) rc = -1;
    if (!rc && fwrite(seq, sizeof(int), (size_t)nseq, f) != (size_t)nseq) rc = -1;
    if (!rc && lp->count) {
        /* a fixed loop carries zeros here: its passes all cache every position */
        static const int zero[9] = {0};
        const int *pc = pcached ? pcached : zero;
        if (fwrite(pc, sizeof(int), (size_t)lp->count + 1, f) != (size_t)lp->count + 1) rc = -1;
        for (int i = 0; !rc && i < nseq; i++) {
            const int v = passes ? passes[i] : 0;
            if (fwrite(&v, sizeof v, 1, f) != 1) rc = -1;
        }
    }
    const size_t slots = (size_t)(lp->count ? lp->nv : n_bound);
    if (!rc && fwrite(ks, sizeof(float), (size_t)kper * slots, f)
               != (size_t)kper * slots) rc = -1;
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = kvc + (size_t)mi * kv_cap * kvpp;
        const size_t n = (size_t)cached * kvpp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = ropec + (size_t)mi * kv_cap * ropepp;
        const size_t n = (size_t)cached * ropepp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    if (fclose(f) != 0) rc = -1;
    if (rc) fprintf(stderr, "failed writing %s\n", path);
    return rc;
}

/* --checkpoint: what a read-in needs to save its progress after a chunk. The file is the
 * ordinary state file (k3_state_save) of the ids read so far, with no pending id, written
 * to PATH.tmp and renamed over PATH, so an interruption mid-write leaves the previous
 * checkpoint whole. */
typedef struct {
    const char  *path;          /* NULL: no checkpoints */
    int          pause_after;   /* chunks this run reads before pausing; 0 = no limit */
    const int   *seq;           /* the whole sequence: seq[0 .. cached) is saved */
    const K3Cfg *c;
    K3StateLoop  loop;
    int64_t      kper, kvpp, ropepp;
    int          written;       /* out: checkpoints written this run */
    int          paused;        /* out: the read-in stopped early */
} K3Checkpoint;

/* Ctrl-C during a checkpointed read-in: finish the chunk, save, stop. A second Ctrl-C
 * finds the default handler again and stops at once. */
static volatile sig_atomic_t k3_pause_requested = 0;

static void k3_on_sigint(int sig)
{
    static const char msg[] = "\nCtrl-C: pausing once this chunk is read and saved; press "
                              "Ctrl-C again to stop now and lose this chunk\n";
    k3_pause_requested = 1;
    signal(sig, SIG_DFL);
#ifdef _WIN32
    (void)_write(2, msg, sizeof msg - 1);
#else
    (void)!write(2, msg, sizeof msg - 1);
#endif
}

static int k3_rename_over(const char *from, const char *to)
{
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
    return rename(from, to);
#endif
}

static int spec_draft(const int *seq, int T, int cap, int *out)
{
    /* Evidence-gated: a draft only fires when the suffix n-gram's occurrences AGREE on
     * what follows. Measured on the released checkpoint, an eager most-recent-match
     * drafter went 0.91x on code, when every partial acceptance also paid a replay sweep.
     * The replay was not the whole cost. Verify sweeps are tentative now (see the
     * speculative step in main), yet the teacher-forced replay of this rule and of an
     * eager drafter (docs/notes/spec-replay.md, a proxy, not K3) still has eager drafts of
     * up to 8 ids losing on every non-edit corpus without any replay: 0.93x to 0.95x at
     * 8 GB and 0.57x to 0.68x with the trunk resident, against 1.07x to 1.13x and 0.87x
     * to 0.92x for this rule (which at --spec 8 loses there too). A rejected draft still
     * costs its position's experts and arithmetic in the sweep. So the gate stays; the
     * study's cost-aware stop rule is the candidate to replace it, not yet measured on
     * K3. Rules: match length 4 (then 3); if the n-gram occurred more than once, every
     * occurrence must propose the same next id, and the draft stops at the first
     * position where historical continuations diverge. */
    if (cap > K3_SPEC_MAX) cap = K3_SPEC_MAX;
    for (int n = 4; n >= 3; n--) {
        if (T < n + 1) continue;
        int m1 = -1, m2 = -1;                            /* two most recent matches */
        for (int j = T - n - 1; j >= 0; j--) {
            int hit = 1;
            for (int i = 0; i < n; i++)
                if (seq[j + i] != seq[T - n + i]) { hit = 0; break; }
            if (!hit) continue;
            if (m1 < 0) m1 = j;
            else { m2 = j; break; }
        }
        if (m1 < 0) continue;
        int nd = 0;
        for (int i = 0; nd < cap && m1 + n + i < T; i++) {
            const int cand = seq[m1 + n + i];
            if (m2 >= 0) {
                /* stop where the two histories stop agreeing */
                if (m2 + n + i >= m1 || seq[m2 + n + i] != cand) break;
            }
            out[nd++] = cand;
        }
        if (nd > 0) return nd;
    }
    return 0;
}

#ifndef K3_VERSION
#define K3_VERSION "1.0.0"
#endif

static void usage(FILE *f)
{
    fprintf(f,
"k3 " K3_VERSION ", Kimi K3 inference engine\n"
"\n"
"usage: k3 <model_dir> [options]\n"
"\n"
"prompt (exactly one):\n"
"  --prompt TEXT         tokenize TEXT and run it\n"
"  --prompt-file PATH    read the prompt from a file; use this for non-ASCII, since\n"
"                        argv is re-encoded by the shell\n"
"  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\n"
"  --ids-file PATH       the same ids read from a file, for prompts longer than one\n"
"                        command-line argument can carry (about 30K ids on Linux)\n"
"  --reread-prompt       process an exact second copy of the prompt before generation\n"
"                        Uses twice the prompt context and adds prefill work.\n"
"                        Does not verify or guarantee the answer.\n"
"\n"
"memory:\n"
"  --preset NAME         auto | ultra | laptop | desktop | workstation | server | max\n"
"                        auto sizes both budgets from this machine's free RAM,\n"
"                        trunk-first; also spelled --trunk-gb auto\n"
"  --list-presets        show each preset's split and expected speed\n"
"  --trunk DIR           packed trunk directory; enables streaming (see scripts/)\n"
"  --trunk-gb X          trunk ring / pinned-layer budget\n"
"  --trunk-rows          exact double-buffered matrix rows; needs --trunk\n"
"                        bounded buffers, no pins; a batch of positions reads each\n"
"                        matrix once (--kv-latent still rereads kv_b per position,\n"
"                        from a plain trunk.bin only the half each pass applies).\n"
"  --mla-split B         with --kv-latent: rebuild each cached position's k and v once per\n"
"                        block of B query positions (verify sweeps, prompts) instead of once\n"
"                        per query; bit-identical; costs B x heads x context floats of\n"
"                        scratch. 0 (default) = the per-query loop\n"
"  --kvb-cache MODE      with --trunk-rows --kv-latent: active = hold the current layer's\n"
"                        kv_b in one buffer, read once per layer visit; pin = keep every\n"
"                        layer's kv_b; off (default) = reread per position. Charged to\n"
"                        --trunk-gb; the output is bit-identical in every mode.\n"
"                        --trunk-gb (default 16) then caps two row buffers of at most\n"
"                        8 MiB each plus the current layer's vectors, and the memory\n"
"                        plan charges the whole budget; pass a small one\n"
"  --cache-gb X          routed-expert cache budget\n"
"  --ultra-low-memory    stream embedding rows and lm_head chunks, and reuse one\n"
"                        recurrent-state slot during full recompute; needs --trunk\n"
"  --expert-pipeline     overlap routed-expert reads with the MoE multiply: publish\n"
"                        each expert as its read lands instead of waiting for the\n"
"                        whole top-k. Off by default; same arithmetic either way\n"
"\n"
"generation:\n"
"  --stream-lm-head      stream exact lm_head chunks while keeping embedding/state\n"
"                        policy unchanged; increase --trunk-gb to use freed memory\n"
"  --gen N               tokens to generate (default 8)\n"
"  --stop-id N           halt after emitting token id N (repeatable, up to 8). The\n"
"                        stop id is kept in the sequence, so --save-state and a later\n"
"                        --load-state continue from what was actually produced.\n"
"                        Off by default: without it --gen N means exactly N tokens,\n"
"                        which the benchmarks and oracle gates rely on. Note the\n"
"                        released checkpoint declares TWO end ids that disagree:\n"
"                        config.json says 163586 (<|end_of_msg|>), tokenizer_config\n"
"                        .json says 163585 ([EOS]), and the model emits 163585.\n"
"                        Pass both to stop on either\n"
"  --incremental         carry KV cache and recurrent state between tokens. The only\n"
"                        mode for prompts over 32,768 ids (up to 1,048,576)\n"
"  --prefill-chunk N     with --incremental: read the prompt in at most N positions per\n"
"                        pass (1 to 32768, default 4096). Bit-identical to one pass;\n"
"                        per-position buffers are sized for N, not for the prompt, and a\n"
"                        long read-in reports its progress after each chunk. A streamed\n"
"                        trunk and experts are read once per chunk\n"
"  --checkpoint PATH     with --incremental: while the prompt is read in, save the\n"
"                        progress to PATH after every chunk but the last (written to\n"
"                        PATH.tmp, then renamed). The same command run again continues\n"
"                        where it stopped, once the saved ids are checked to begin the\n"
"                        prompt. Ctrl-C pauses after the chunk being read is saved;\n"
"                        a second Ctrl-C stops at once. PATH is kept: delete it when done\n"
"  --pause-after N       with --checkpoint: read N chunks in this run, save, and stop\n"
"                        with exit status 3 (paused, not finished). Ctrl-C does the same\n"
"  --kv-latent           cache MLA's compressed latent instead of the expanded k and\n"
"                        v, and rebuild them on use: 0.055 MB per position instead of\n"
"                        2.37, at the cost of one kv_b matmul's worth per cached\n"
"                        position per query token (per step at T = 1; prefill and\n"
"                        --spec verification pay it per position): its key rows to\n"
"                        score, its value rows to weight. Logits are bitwise\n"
"                        identical either way. Needs --incremental\n"
"  --save-state PATH     write the carried state after the run, so the next turn of a\n"
"                        conversation resumes instead of re-reading the whole prompt\n"
"  --load-state PATH     resume from a saved state; the prompt given now is treated as\n"
"                        the CONTINUATION of the saved sequence. Needs --incremental\n"
"  --draft-trunk DIR     hybrid decode: a second packed trunk (typically a quantized\n"
"                        derivation of the real one, see tools/qdq_trunk.py) DRAFTS\n"
"                        tokens which the exact model verifies in batched sweeps.\n"
"                        Output remains exactly the exact model's greedy decode; the\n"
"                        draft only proposes. Needs --incremental; implies --spec 4\n"
"  --draft-trunk-gb X    trunk budget for the draft model (default 6)\n"
"  --spec N              speculative decode: draft up to N tokens by n-gram lookup and\n"
"                        verify them in ONE batched sweep. Output is identical to\n"
"                        serial decode by construction; needs --incremental. A sweep\n"
"                        reads the trunk and lm_head once for all its positions (but\n"
"                        --kv-latent with --trunk-rows rereads kv_b, from a plain\n"
"                        trunk.bin the half each pass applies, for every cached\n"
"                        position it rebuilds); each extra position still costs its\n"
"                        own experts and arithmetic.\n"
"                        No full-checkpoint speedup has been measured.\n"
"                        A rejected draft costs no second sweep: the sweep is tentative\n"
"                        and only the accepted positions are committed\n"
"  --tok DIR             directory with tiktoken.model and tokenizer_config.json\n"
"\n"
"layer loop, CHANGES THE MODEL (docs/notes/layer-loop.md); off by default:\n"
"  --loop                the AttnRes block holding the middle layer (36-47 on K3) loops\n"
"                        adaptively: 2 to 4 passes per token, a token stopping once a\n"
"                        pass moves its span output by under 5%% (relative L2).\n"
"                        --loop-count and --loop-exit change the cap and threshold. K3\n"
"                        was not trained for this; its effect on quality is unmeasured\n"
"  --no-loop             the released model, every layer once: the default, accepted\n"
"                        so that scripts written while the loop was on still work\n"
"  --loop-layers A-B     loop layers A..B instead (zero-based, whole AttnRes blocks), as\n"
"                        if the span appeared again in the stack. Each pass keeps its own\n"
"                        recurrent state and KV rows; weights are shared\n"
"  --loop-count R        passes over the --loop-layers span, 1 to 8 (default 2); with\n"
"                        --loop-exit, the most a token runs. 1 is the released model\n"
"  --loop-exit EPS       adaptive depth: a token stops after pass p >= 2 once the span's\n"
"                        output moved by less than EPS (relative L2) from pass p - 1.\n"
"                        Pass p's layers see only the tokens that reached it. 0 never\n"
"                        stops early (the fixed loop)\n"
"\n"
"diagnostics:\n"
"  --score-prompt       teacher-force the prompt; write per-token NLL to --out.\n"
"                        No generation, implicit BOS/EOS, or state continuation\n"
"  --score-start N      first target position to score (default 1); earlier tokens\n"
"                        remain context. Used for overlapping quality windows\n"
"  --expert-profile PATH ranked calibration profile from tools/expert_profile.py\n"
"  --pin-experts N        retain N profiled experts on first use; both flags required\n"
"  --config PATH         model config; defaults to <model_dir>/config.json\n"
"  --layers N            bind only the first N layers (partial shard sets)\n"
"  --dump-logits PATH    write float32 logits for the first step\n"
"  --dump-all-logits PATH  write the float32 logits behind EVERY generated token, in\n"
"                        order, one vocabulary-length vector each, so two runs can be\n"
"                        compared step by step rather than at the first step only\n"
"  --dump-cache-trace D  write expert_hist.json and expert_trace.bin into D, for\n"
"                        offline analysis with tools/sim_cache.py\n"
"  --out FILE            JSON results (default k3_run.json)\n"
"  --version, --help\n"
"\n"
"Memory is a dial, not a floor: the same model runs in 8 GB and in 224 GB and produces\n"
"identical output. Give memory to the trunk before the expert cache, see\n"
"docs/TUNING.md for why, and scripts/k3-doctor.sh to size this machine.\n");
}

/* ------------------------------------------------------------------- presets ----
 * Named memory budgets, so a user does not have to discover the trunk/cache split
 * empirically.
 *
 * The split is not arbitrary and it is not symmetric. Per token the engine re-reads the
 * ENTIRE 108.81 GB trunk but only ~25.8 GB of routed experts, so a gigabyte given to the
 * trunk removes roughly 1.17 GB/token of guaranteed traffic (one pinned layer) while a
 * gigabyte given to the expert cache removes, below about 36 GB of arena, nothing
 * measurable, K3's router is trained for flat expert usage, which defeats an LRU.
 *
 * Measured consequence: at a fixed 128 GB budget, trunk-first runs 1.69x faster than
 * cache-first. So every preset fills the trunk before it feeds the cache.
 * docs/PERFORMANCE.md carries the data and the noise floor that bounds it. */
typedef struct {
    const char *name;
    double trunk_gb, cache_gb;
    int ultra;
    const char *note;
} K3Preset;

/* The trunk/cache figures are BUDGETS passed to the two allocators. The description
 * quotes measured peak RSS for the whole process, which is the number that decides
 * whether a machine can run the preset, it includes the safetensors index, the KV
 * cache and scratch, none of which appear in either budget. Measured on the reference
 * machine in docs/PERFORMANCE.md; expect a little variation elsewhere. */
static const K3Preset K3_PRESETS[] = {
    { "ultra",       2.5,  0.31, 1,
      "~3 GB planned: streamed model tables, one state slot. Slow." },
    { "laptop",      3.0,   1.0, 0, "8.2 GB peak RSS. The ordinary-path floor." },
    { "desktop",    16.0,  10.0, 0, "31.9 GB peak RSS." },
    { "workstation", 60.0, 30.0, 0,
      "95.5 GB peak RSS; the expert cache starts to matter here." },
    { "server",     110.0, 13.0, 0,
      "~128 GB peak RSS; 90 of 93 trunk layers pinned. Fastest." },
    { "max",        110.0,109.0, 0,
      "~224 GB peak RSS; trunk pinned and a large expert cache." },
};
enum { K3_NPRESET = (int)(sizeof K3_PRESETS / sizeof K3_PRESETS[0]) };

static const K3Preset *k3_preset_find(const char *name)
{
    for (int i = 0; i < K3_NPRESET; i++)
        if (!strcmp(name, K3_PRESETS[i].name)) return &K3_PRESETS[i];
    return NULL;
}

static void k3_preset_list(FILE *f)
{
    fprintf(f, "presets (trunk / expert-cache, in GB):\n");
    for (int i = 0; i < K3_NPRESET; i++)
        fprintf(f, "  %-12s %6.2f / %-6.2f  %s\n", K3_PRESETS[i].name,
                K3_PRESETS[i].trunk_gb, K3_PRESETS[i].cache_gb, K3_PRESETS[i].note);
    fprintf(f, "  %-12s %6s / %-6s  %s\n", "auto", "fit", "fit",
            "sizes both from this machine's free RAM, trunk-first. Recommended.");
    fprintf(f, "\nAll presets stream the trunk, so they need --trunk <packed_dir>.\n"
               "Run scripts/k3-doctor.sh to see which one this machine fits.\n");
}

/* PEAK resident set, in bytes. ru_maxrss is kilobytes on Linux and BYTES on Darwin, so
 * the scale factor differs by platform; applying the Linux one on macOS would overstate
 * the peak by 1024x.
 *
 * This is the authoritative memory figure. The banner printed before allocation is a
 * PLAN and understates: it omits the safetensors index (~78 MB at full scale), reports
 * requested budgets rather than actual reservations, and cannot observe fragmentation.
 * Quote this value, not the plan. */
static double peak_rss_bytes(void)
{
#ifdef _WIN32
    /* PeakWorkingSetSize is Windows' peak-RSS equivalent, already in bytes -- no
     * kilobyte scaling needed, unlike ru_maxrss on Linux. */
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return 0.0;
    return (double)pmc.PeakWorkingSetSize;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
#if defined(__APPLE__)
    return (double)ru.ru_maxrss;            /* already bytes */
#else
    return (double)ru.ru_maxrss * 1024.0;   /* kilobytes */
#endif
#endif
}

/* MemAvailable, which is what the kernel thinks can actually be handed out, not
 * MemFree. Returns 0 if it cannot be read. */
static double mem_available_bytes(void)
{
#ifdef _WIN32
    /* ullAvailPhys is Windows' MemAvailable equivalent: physical memory that can
     * actually be handed out (already accounts for the standby/modified page
     * lists the way MemAvailable accounts for reclaimable cache), not the raw
     * free count. */
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0.0;
    return (double)ms.ullAvailPhys;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
    fclose(f);
    return kb * 1024.0;
#endif
}

/* Floats of KV cache per position, per MLA layer, in whichever layout is in force.
 * Every stride, every plan figure and the state file's kvpp come from this one place:
 * two spellings of the same number is how a cache gets read at the wrong stride. */
static size_t kv_floats_per_pos(const K3Cfg *c, int kv_latent)
{
    return kv_latent ? (size_t)c->kv_lora
                     : (size_t)c->n_heads * (size_t)(c->qk_nope + c->v_head);
}

typedef struct {
    K3LayerBind *lay;
    K3ModelBind  mb;
    K3ModelStream ms;
    int          n_bound;
    int          layers_completed;
    /* --loop-layers (docs/notes/layer-loop.md): the order layers EXECUTE in. Without the
     * flag vmap is NULL, the loop below runs v = 0..n_bound-1 with layer v, and every
     * state and cache slot index is the one it always was. With it, n_virt positions
     * execute; vmap[v] is the physical layer whose weights position v uses, and
     * vslot[v] its recurrent-state slot, which is v itself: each executed position owns
     * its state and its MLA cache rows, as a duplicated layer would. Only the
     * K3_LOOP_MUTANT_SHARED_STATE gate mutant points a repeat back at its first pass. */
    int          n_virt;
    int         *vmap;
    int         *vslot;
    /* --loop-exit (adaptive depth, docs/notes/layer-loop.md): loop_eps < 0 is the fixed
     * loop. Otherwise passes 2..loop_rmax of the span loop_a..loop_b run only on the
     * tokens still active; a token stops after pass p >= 2 once its span output moved by
     * less than loop_eps (relative L2) from pass p - 1, or at loop_rmax. Pass p's layers
     * see exactly the tokens that reached pass p, in order, so their KDA state advances
     * only on those and their MLA cache holds only those rows: loop_pcached[p] of them
     * (incremental decode; pass 1 is the stack's own and uses `cached`). */
    int          loop_a, loop_b, loop_span, loop_rmax;
    double       loop_eps;
    int         *loop_pcached;   /* [loop_rmax + 1] */
    int         *loop_act, *loop_idx, *loop_nbt;   /* [Tmax] per token of a forward */
    float       *loop_hprev;     /* [Tmax][E] each token's span output after its last pass */
    float       *loop_hc, *loop_brc;   /* [Tmax][E], [Tmax][maxb][E]: a pass's active rows */
    int         *loop_passes;    /* [Tmax] passes each absolute position ran */
    int         *loop_sweep_pcached;   /* loop_pcached when a tentative sweep began */
    int          loop_sweep_base;      /* the absolute position a tentative sweep began at */
    double      *loop_d;         /* [Tmax][loop_rmax + 1] the change after each pass */
    int          ultra;
    int          stream_lm_head;
    K3Trunk     *trunk;      /* non-NULL when the trunk is streamed rather than resident */
    /* Incremental decode state. Only MLA layers need a KV cache, so the 24 of them are
     * numbered densely rather than indexing all 93 and wasting 74% of the allocation. */
    float       *kvc, *ropec;
    int         *mla_slot;   /* [executed position] -> dense MLA index, or -1 */
    int          n_mla, kv_cap, cached;
    int          max_rows;   /* positions one forward() may take: what h, br and the
                              * scratch were sized for (the read-in chunk under
                              * --incremental, the whole sequence otherwise) */
    int          kv_latent;  /* kvc holds the latent, not expanded k and v */
    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
    const int   *score_targets; /* optional next-token targets for each input position */
    double      *score_nll;
    int          score_start;   /* first scored input position, zero-based */
    /* [logit_rows][vocab]: when every position needs logits, lm_head is applied to this
     * many positions per pass. NULL (or 1 row) projects one position at a time. */
    float       *logit_block;
    int          logit_rows;
    /* Tentative sweeps for speculative decode: log rows for each KDA layer (see K3KdaLog
     * in k3.h), allocated only with --spec or --draft-trunk. Only KDA layers carry state
     * that absorbs a rejected token, so the MLA layers get no log. log_row0 >= 0 makes
     * the next forward() tentative: KDA state untouched, positions recorded from that row
     * on, until spec_commit applies the ones kept. */
    float       *kda_log;
    float       *kda_work;     /* one KDA layer's state; every layer, both models, share it */
    int         *kda_slot;     /* [executed position] -> dense KDA index, or -1 */
    int          n_kda, log_cap, log_row0;
    size_t       log_floats;   /* floats of log rows per KDA layer */
    /* --dump-all-logits: the logit vector behind every position of a verify sweep. */
    float       *all_logits;
    long         forwards;     /* forward() calls, so the report can show there was no
                                * replay: one per decode step */
    long         prefill_passes;   /* of those, the read-in's (prefill()): one per chunk */
} Weights;

/* One full forward over T tokens, writing logits for the LAST position only. Every
 * step rebuilds state from scratch, matching the path the oracle validates.
 *
 * Returns 0 on success and -1 if the forward could not be completed. The caller MUST
 * check: on failure logits_last is left untouched, and argmaxing an untouched buffer
 * yields a token drawn from uninitialised memory, printed as though it were output. */
/* arg_all: when non-NULL, receives argmax(logits) for EVERY position 0..T-1, which is
 * what batched greedy verification consumes. logits_last still gets the final position's
 * full vector either way. The extra cost is lm_head applied to the additional positions,
 * which with a logit block is one pass over the head per block of positions rather than
 * one per position. (The ~22% of a serial token per extra verified position once measured
 * at streamed-trunk budgets predates batching the trunk matrices and the head over
 * positions; it needs remeasuring.) */
static int forward(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                   float *logits_last, float *scratch, float *h, float *br, float *kstate,
                   int *arg_all)
{
    const int E = c->hidden;
    const int maxb = k3_attn_res_stride(c);
    const int P = c->kda_heads * c->kda_head_dim;
    const size_t kper = (size_t)P * c->kda_head_dim + (size_t)3 * P * (c->conv_k - 1);

    /* Every per-position buffer holds max_rows rows; more would overrun them silently. */
    if (T < 1 || T > w->max_rows) {
        fprintf(stderr, "forward of %d positions, but the buffers hold %d; refusing\n",
                T, w->max_rows);
        return -1;
    }
    w->forwards++;
    for (int t = 0; t < T; t++) {
        if (w->ultra) {
            if (k3_model_stream_embed_row(&w->ms, h + (size_t)t * E, ids[t]) != 0) {
                fprintf(stderr, "embedding row load failed for token %d at position %d\n",
                        ids[t], t);
                return -1;
            }
        } else {
            k3_embed_row(h + (size_t)t * E, w->mb.embed, w->mb.wdt, ids[t], E);
        }
    }

    memset(br, 0, (size_t)T * maxb * E * sizeof(float));
    /* Incremental decode carries the KDA recurrent matrix and ShortConv history across
     * steps, so it must NOT be cleared here; the full-recompute path rebuilds from
     * scratch every step and must be. */
    /* executed positions: n_bound without --loop-layers, more with it */
    const int nv = w->vmap ? w->n_virt : w->n_bound;
    if (!w->kvc) {
        const size_t slots = w->ultra ? 1u : (size_t)nv;
        memset(kstate, 0, kper * slots * sizeof(float));
    }
    w->layers_completed = 0;
    int nb = 0;
    /* --loop-exit: executed positions rep0 .. repN-1 are passes 2..loop_rmax of the span.
     * A pass runs on its active tokens only; when some token has stopped, their rows are
     * gathered in order into loop_hc / loop_brc and scattered back after the pass. Every
     * active token has run the same passes, so the pass shares one block count (nbp);
     * after the span the counts differ and the layers take one per token (loop_nbt). */
    const int adapt = w->vmap && w->loop_eps >= 0.0;
    const int rep0 = adapt ? w->loop_b + 1 : nv;
    const int repN = adapt ? rep0 + (w->loop_rmax - 1) * w->loop_span : nv;
    const int pos0 = w->kvc ? w->cached : 0;      /* absolute position of row 0 */
    int Tp = T, nbp = 0, pass = 0;
    /* GATE MUTANT (test builds only): K3_LOOP_MUTANT_NO_COMPACT runs every pass on EVERY
     * token, stopped ones included, keeping only the active tokens' results. Outputs of
     * stopped tokens are untouched, but the pass's KDA state and MLA rows absorb tokens
     * that never reached it, which the reference must see. */
    int no_compact = 0;
#ifdef K3_TEST_MUTANTS
    no_compact = getenv("K3_LOOP_MUTANT_NO_COMPACT") != NULL;
#endif
    float *hp = h, *bp = br;
    for (int v = 0; v < nv; v++) {
        const int L = w->vmap ? w->vmap[v] : v;    /* whose weights */
        const int sv = w->vslot ? w->vslot[v] : v; /* whose state and cache rows */
        const int in_rep = v >= rep0 && v < repN;
        if (in_rep && (v - rep0) % w->loop_span == 0) {      /* a pass begins */
            pass = 2 + (v - rep0) / w->loop_span;
            Tp = 0;
            for (int t = 0; t < T; t++) if (w->loop_act[t]) w->loop_idx[Tp++] = t;
            if (Tp == 0) {                                   /* every token has stopped */
                v += w->loop_span - 1;
                w->layers_completed = v + 1;
                continue;
            }
            nbp = w->loop_nbt[w->loop_idx[0]];
            if (no_compact && Tp < T) {
                Tp = 0;
                for (int t = 0; t < T; t++) w->loop_idx[Tp++] = t;
                Tp = -T;     /* below: gather all T, as if compacting */
            }
            if (Tp < 0) {
                Tp = T;
                for (int i = 0; i < Tp; i++) {
                    memcpy(w->loop_hc + (size_t)i * E, h + (size_t)i * E, (size_t)E * sizeof(float));
                    memcpy(w->loop_brc + (size_t)i * maxb * E, br + (size_t)i * maxb * E,
                           (size_t)maxb * E * sizeof(float));
                }
                hp = w->loop_hc; bp = w->loop_brc;
            } else if (Tp < T) {
                for (int i = 0; i < Tp; i++) {
                    const int t = w->loop_idx[i];
                    memcpy(w->loop_hc + (size_t)i * E, h + (size_t)t * E, (size_t)E * sizeof(float));
                    memcpy(w->loop_brc + (size_t)i * maxb * E, br + (size_t)t * maxb * E,
                           (size_t)maxb * E * sizeof(float));
                }
                hp = w->loop_hc; bp = w->loop_brc;
            } else {
                hp = h; bp = br;
            }
        }
        /* Streaming: bring this layer in, and hint the next one so its read overlaps
         * this layer's arithmetic. The order is fixed every token (0..92, or the
         * --loop-layers order), so the hint is never wrong. */
        if (w->trunk) {
            if (k3_trunk_bind(w->trunk, c, L, &w->lay[L]) != 0) {
                fprintf(stderr, "trunk bind failed at layer %d\n", L);
                return -1;
            }
            k3_trunk_prefetch(w->trunk,
                              !w->vmap ? L + 1 : v + 1 < nv ? w->vmap[v + 1] : w->n_bound);
        }
        /* Point this layer's MoE at the cache before use. Doing it here rather than at
         * bind time keeps K3LayerBind independent of any particular cache. */
        if (w->lay[L].lay.moe) {
            w->lay[L].moe.src = &cache->src;
            w->lay[L].moe.layer = L;
            /* The draft routes only among resident experts, reading zero new expert bytes;
             * the exact model keeps true routing. This is what makes a draft step cheap. */
            w->lay[L].moe.cache_only = w->draft_mode;
        }
        /* Full recompute consumes a layer's KDA/ShortConv state only while that layer is
         * executing over the complete sequence. Once the layer returns, no later layer
         * can observe it, so ultra mode may clear and reuse one slot. Incremental decode
         * still needs one persistent slot per layer and therefore never takes this
         * path. */
        float *layer_state = kstate + ((w->ultra && !w->kvc) ? 0 : kper * (size_t)sv);
        if (w->ultra && !w->kvc)
            memset(layer_state, 0, kper * sizeof(float));
        const long drops_before = k3_expert_drops;
        /* In a tentative sweep a KDA layer leaves its state alone and records its
         * recurrence inputs instead; see the speculative step in main(). The log belongs
         * to the executed position, so each pass of a looped KDA layer has its own. */
        K3KdaLog rlog, *rl = NULL;
        if (w->kda_log && w->log_row0 >= 0 && w->kda_slot[v] >= 0) {
            rlog.rows = w->kda_log + w->log_floats * (size_t)w->kda_slot[v];
            rlog.work = w->kda_work;
            rlog.cap  = w->log_cap;
            rlog.row0 = w->log_row0;
            rl = &rlog;
        }
        if (adapt && (in_rep || v >= repN)) {
            /* a pass of the span (its active rows, one shared count), or a layer after
             * it (every row, a count per token); MLA in a pass reads and extends that
             * pass's own cache rows */
            const int mi = w->kvc ? w->mla_slot[v] : -1;
            const size_t kvper = (size_t)w->kv_cap * kv_floats_per_pos(c, w->kv_latent);
            const size_t rpper = (size_t)w->kv_cap * c->qk_rope;
            float *kv = mi >= 0 ? w->kvc + kvper * (size_t)mi : NULL;
            float *rp = mi >= 0 ? w->ropec + rpper * (size_t)mi : NULL;
            const int cch = in_rep ? w->loop_pcached[pass] : w->cached;
            if (in_rep)
                k3_decoder_layer_inc_log(hp, bp, &nbp, &w->lay[L].lay, c, L, Tp, layer_state,
                                         scratch, kv, rp, kv ? cch : 0, w->kv_cap, w->kv_latent,
                                         rl);
            else
                k3_decoder_layer_inc_logv(h, br, w->loop_nbt, &w->lay[L].lay, c, L, T,
                                          layer_state, scratch, kv, rp, kv ? cch : 0,
                                          w->kv_cap, w->kv_latent, rl);
        } else if (w->kvc && w->mla_slot[v] >= 0) {
            const size_t kvper = (size_t)w->kv_cap * kv_floats_per_pos(c, w->kv_latent);
            const size_t rpper = (size_t)w->kv_cap * c->qk_rope;
            const int mi = w->mla_slot[v];
            k3_decoder_layer_inc(h, br, &nb, &w->lay[L].lay, c, L, T,
                                 layer_state, scratch,
                                 w->kvc + kvper * (size_t)mi,
                                 w->ropec + rpper * (size_t)mi,
                                 w->cached, w->kv_cap, w->kv_latent);
        } else {
            k3_decoder_layer_inc_log(h, br, &nb, &w->lay[L].lay, c, L, T,
                                     layer_state, scratch,
                                     NULL, NULL, 0, 0, 0, rl);
        }
        if (w->trunk && w->trunk->read_error) {
            fprintf(stderr, "trunk row read failed at layer %d; refusing partial output\n", L);
            return -1;
        }
        if (k3_expert_drops != drops_before) {
            fprintf(stderr, "routed expert load failed at layer %d; refusing partial "
                            "MoE output\n", L);
            return -1;
        }
        if (adapt && v == rep0 - 1) {
            /* pass 1 is done for every token: its span output is the reference for pass 2 */
            memcpy(w->loop_hprev, h, (size_t)T * E * sizeof(float));
            for (int t = 0; t < T; t++) {
                w->loop_act[t] = w->loop_rmax > 1;
                w->loop_nbt[t] = nb;
                w->loop_passes[pos0 + t] = 1;
            }
        }
        if (in_rep && (v - rep0) % w->loop_span == w->loop_span - 1) {   /* a pass ends */
            const int moved = hp != h;                       /* rows were gathered */
            for (int i = 0; i < Tp; i++) {
                const int t = moved ? w->loop_idx[i] : i;
                if (!w->loop_act[t]) continue;               /* only the mutant runs these */
                const float *hn = hp + (size_t)i * E;
                float *ho = w->loop_hprev + (size_t)t * E;
                /* relative L2 change, one double sum each over the row in index order */
                double dd = 0.0, oo = 0.0;
                for (int j = 0; j < E; j++) {
                    const double df = (double)hn[j] - (double)ho[j];
                    dd += df * df;
                    oo += (double)ho[j] * (double)ho[j];
                }
                const double d = oo > 0.0 ? sqrt(dd / oo) : (dd > 0.0 ? INFINITY : 0.0);
                memcpy(ho, hn, (size_t)E * sizeof(float));
                w->loop_d[(size_t)(pos0 + t) * (w->loop_rmax + 1) + pass] = d;
                w->loop_passes[pos0 + t] = pass;
                if (d < w->loop_eps || pass == w->loop_rmax) w->loop_act[t] = 0;
                w->loop_nbt[t] = nbp;
                if (moved) {
                    memcpy(h + (size_t)t * E, hn, (size_t)E * sizeof(float));
                    memcpy(br + (size_t)t * maxb * E, bp + (size_t)i * maxb * E,
                           (size_t)maxb * E * sizeof(float));
                }
            }
            if (w->kvc) w->loop_pcached[pass] += Tp;
            hp = h; bp = br;
        }
        w->layers_completed = v + 1;
    }

    /* The model-level aggregator, beyond the two per layer. Exactly one pair exists in
     * the checkpoint; skipping it is silent. */
    if (w->mb.out_res_norm && w->mb.out_res_proj) {
        float *fold = scratch;
        float *src  = fold + E;
        for (int i = 0; i < E; i++) fold[i] = w->mb.out_res_norm[i] * w->mb.out_res_proj[i];
        for (int t = 0; t < T; t++) {
            const int nbt = adapt ? w->loop_nbt[t] : nb;
            for (int b = 0; b < nbt; b++)
                memcpy(src + (size_t)b * E, br + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)nbt * E, h + (size_t)t * E, (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, fold, nbt + 1, E, c->rms_eps);
        }
    }

    float *nrm = scratch;
    if (arg_all || w->score_targets) {
        /* Every position's logits, lm_head applied to a block of positions per pass
         * (k3_mmw_batch, or one streamed pass over the head per block) instead of once
         * per position: verifying 9 speculative positions otherwise reads the 2.35 GB
         * head 9 times. Each position's vector is bit-identical to the one-position
         * projection, and positions before score_start are still never projected.
         * nrm holds at most T rows, well inside the layer scratch. */
        const int first = w->score_targets ? w->score_start : 0;
        const int rows = (w->logit_block && w->logit_rows > 1) ? w->logit_rows : 1;
        float *blk = rows > 1 ? w->logit_block : logits_last;
        int last = -1;                        /* row of blk holding the final position */
        for (int t0 = first; t0 < T; t0 += rows) {
            const int n = T - t0 < rows ? T - t0 : rows;
            for (int j = 0; j < n; j++)
                k3_rmsnorm(nrm + (size_t)j * E, h + (size_t)(t0 + j) * E, w->mb.norm, E,
                           c->rms_eps);
            if (w->stream_lm_head) {
                if (k3_model_stream_project_batch(&w->ms, blk, nrm, n) != 0) return -1;
            } else {
                k3_mmw_batch(blk, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab, n);
            }
            for (int j = 0; j < n; j++) {
                const int t = t0 + j;
                const float *lt = blk + (size_t)j * c->vocab;
                if (arg_all) arg_all[t] = argmax_(lt, c->vocab);
                if (arg_all && w->all_logits && t <= K3_SPEC_MAX)
                    memcpy(w->all_logits + (size_t)t * c->vocab, lt,
                           (size_t)c->vocab * sizeof(float));
                if (w->score_targets && k3_token_nll(lt, c->vocab,
                                                     w->score_targets[t], w->score_nll + t)) {
                    fprintf(stderr, "invalid logits/target at score position %d\n", t + 1);
                    return -1;
                }
            }
            last = n - 1;
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        if (blk != logits_last && last >= 0)
            memcpy(logits_last, blk + (size_t)last * c->vocab, (size_t)c->vocab * sizeof(float));
        return 0;
    }
    k3_rmsnorm(nrm, h + (size_t)(T - 1) * E, w->mb.norm, E, c->rms_eps);
    if (w->stream_lm_head) {
        if (k3_model_stream_project(&w->ms, logits_last, nrm) != 0) return -1;
    } else {
        k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
    }
    return 0;
}

/* Rows every per-position buffer holds: the whole sequence under full recompute, which
 * forwards all of it each step; under --incremental the read-in chunk, or a --spec verify
 * sweep (K3_SPEC_MAX + 1) if that is more. forward() refuses anything larger. */
static int pass_rows(int incremental, int chunk, int Tmax)
{
    if (!incremental) return Tmax;
    const int r = chunk > K3_SPEC_MAX + 1 ? chunk : K3_SPEC_MAX + 1;
    return r < Tmax ? r : Tmax;
}

/* One --checkpoint: the state after w->cached positions, as k3_state_save writes it for
 * a run that stopped there with nothing pending. */
static int checkpoint_write(const Weights *w, K3Checkpoint *ck, const float *ks)
{
    const size_t n = strlen(ck->path);
    char *tmp = (char *)malloc(n + 5);
    if (!tmp) return -1;
    memcpy(tmp, ck->path, n);
    memcpy(tmp + n, ".tmp", 5);
    const double t0 = now_s();
    int rc = k3_state_save(tmp, ck->c, ck->seq, w->cached, ks, w->kvc, w->ropec, w->n_bound,
                           w->n_mla, w->kv_cap, w->cached, ck->kper, ck->kvpp, ck->ropepp,
                           &ck->loop, w->loop_pcached, w->loop_passes);
    if (rc == 0 && k3_rename_over(tmp, ck->path) != 0) {
        fprintf(stderr, "cannot rename %s to %s\n", tmp, ck->path);
        rc = -1;
    }
    free(tmp);
    if (rc == 0) {
        ck->written++;
        printf("  checkpoint: %d positions saved to %s in %.1f s\n", w->cached, ck->path,
               now_s() - t0);
    }
    return rc;
}

/* Incremental read-in of n ids, at most `chunk` positions per forward(), advancing
 * w->cached past each chunk. A position's arithmetic does not depend on which batch it
 * rides in (GATE 3c of tests/unit/k3_model.c: T positions in one forward give the bits
 * of T one-position steps), so the carried state and the last position's logits are
 * those of one pass over all n; only the per-position buffers shrink, from n rows to
 * chunk. `who` names the model in the progress lines a multi-chunk read-in prints;
 * NULL prints none.
 *
 * With a --checkpoint (ck->path), the state is saved after every chunk that leaves
 * positions unread, never after the last: a resumed run must have at least one position
 * to read, since the logits behind the first generated id are not saved. The read-in
 * pauses after a save once ck->pause_after chunks have been read in this run, or a Ctrl-C
 * came: ck->paused is set and 0 returned with w->cached short of the prompt. */
static int prefill(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int n,
                   int chunk, float *lg, float *sc, float *h, float *br, float *ks,
                   const char *who, K3Checkpoint *ck)
{
    const double t0 = now_s();
    /* The time to go: a position's cost grows with the context its attention reads, so
     * the rate per position is fitted as r0 + k * position from the first chunk and the
     * latest one, and integrated over the positions left. */
    double r1 = 0.0, m1 = 0.0;
    for (int i = 0; i < n; ) {
        const int q = n - i < chunk ? n - i : chunk;
        const double tc = now_s();
#ifdef K3_TEST_MUTANTS
        /* GATE MUTANT (test builds only): K3_PREFILL_MUTANT_RESET starts every chunk after
         * the first from cleared KDA and ShortConv state, as a read-in that forgot what
         * the earlier chunks carried would. One pass, and decode, are untouched. */
        if (i > 0 && getenv("K3_PREFILL_MUTANT_RESET")) {
            const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
            const size_t kper = P * c->kda_head_dim + 3 * P * (size_t)(c->conv_k - 1);
            memset(ks, 0, kper * (size_t)(w->vmap ? w->n_virt : w->n_bound) * sizeof(float));
        }
#endif
        if (forward(w, c, cache, ids + i, q, lg, sc, h, br, ks, NULL) != 0) return -1;
        w->prefill_passes++;
        w->cached += q;
        const double r = (now_s() - tc) / q, m = w->cached - 0.5 * q;   /* this chunk */
        if (i == 0) { r1 = r; m1 = m; }
        i += q;
        if (who && n > chunk) {
            const double s = now_s() - t0;
            printf("  %s read in %d of %d positions (%.1f%%): %.1f s, %.3f s per "
                   "position%s", who, i, n, 100.0 * i / n, s, s / i, i < n ? ", " : "\n");
            if (i < n) {
                double k = m > m1 ? (r - r1) / (m - m1) : 0.0;
                if (k < 0.0) k = 0.0;
                const double r0 = r1 - k * m1, a = w->cached, b = a + (n - i);
                printf("about %.0f s to go\n", r0 * (b - a) + 0.5 * k * (b * b - a * a));
            }
            fflush(stdout);
        }
        if (ck && ck->path && i < n) {
            if (checkpoint_write(w, ck, ks) != 0) return -1;
            if ((ck->pause_after && ck->written >= ck->pause_after) || k3_pause_requested) {
                ck->paused = 1;
                return 0;
            }
        }
    }
    return 0;
}

/* The adaptive loop's record for the report: per absolute position, the passes it ran
 * (0 = never forwarded) and the relative change after each pass from the second on.
 * Writes null when the loop is fixed or off. */
static void write_loop_adaptive(FILE *f, const Weights *w, int npos)
{
    if (w->loop_eps < 0.0 || !w->loop_passes) { fputs("null", f); return; }
    long hist[9] = {0};
    for (int i = 0; i < npos; i++)
        if (w->loop_passes[i] >= 0 && w->loop_passes[i] <= 8) hist[w->loop_passes[i]]++;
    fprintf(f, "{\"eps\":%.17g,\"max_passes\":%d,\"passes_histogram\":[", w->loop_eps,
            w->loop_rmax);
    for (int r = 0; r <= w->loop_rmax; r++) fprintf(f, "%s%ld", r ? "," : "", hist[r]);
    fputs("],\"passes\":[", f);
    for (int i = 0; i < npos; i++) fprintf(f, "%s%d", i ? "," : "", w->loop_passes[i]);
    fputs("],\"change\":[", f);
    for (int i = 0; i < npos; i++) {
        fputs(i ? ",[" : "[", f);
        for (int r = 2; r <= w->loop_passes[i]; r++)
            fprintf(f, "%s%.17g", r > 2 ? "," : "",
                    w->loop_d[(size_t)i * (w->loop_rmax + 1) + r]);
        fputc(']', f);
    }
    fputs("]}", f);
}

/* Commit the first n positions of a tentative sweep (or of a run of tentative draft
 * calls) to every KDA layer's carried state. No weights are read and no forward runs: see
 * k3_kda_advance for why the result is the state serial decode holds after those
 * positions, to the bit. MLA layers need nothing, because their KV rows are positional
 * and the caller sets `cached`. */
static void spec_commit(Weights *w, const K3Cfg *c, float *kstate, int n)
{
    const size_t kper = k3_kda_state_floats(c);
    /* Adaptive depth (--loop-exit): pass p's layers logged only the sweep positions that
     * reached pass p, compacted in order, so the first n positions own the first n_p rows
     * of those logs, n_p = how many of them reached pass p. The MLA rows of pass p are
     * cut back the same way (loop_pcached was advanced by the whole sweep). */
    int np_[9] = {0};
    const int adapt = w->vmap && w->loop_eps >= 0.0;
    const int rep0 = adapt ? w->loop_b + 1 : 0;
    if (adapt) {
        const int base = w->loop_sweep_base;
        for (int p = 2; p <= w->loop_rmax; p++) {
            for (int t = 0; t < n; t++) np_[p] += w->loop_passes[base + t] >= p;
            w->loop_pcached[p] = w->loop_sweep_pcached[p] + np_[p];
        }
    }
    const int nv = w->vmap ? w->n_virt : w->n_bound;
    for (int v = 0; v < nv; v++) {
        if (w->kda_slot[v] < 0) continue;
        const int in_rep = adapt && v >= rep0 && v < rep0 + (w->loop_rmax - 1) * w->loop_span;
        K3KdaLog rl;
        rl.rows = w->kda_log + w->log_floats * (size_t)w->kda_slot[v];
        rl.work = NULL;
        rl.cap  = w->log_cap;
        rl.row0 = 0;
        const int sv = w->vslot ? w->vslot[v] : v;
        k3_kda_advance(kstate + kper * (size_t)sv, &rl,
                       in_rep ? np_[2 + (v - rep0) / w->loop_span] : n, c);
    }
}

/* How many of a step's emitted ids the output loop keeps: all of them, unless --gen runs
 * out or a --stop-id is reached first. A speculative step needs this BEFORE it appends
 * anything, because the carried state must end exactly one position short of the last
 * kept id, the same place serial decode would leave it; otherwise --save-state would write
 * a state that has consumed ids the sequence does not contain. */
static int emit_keep(const int *emit, int emitn, int nout, int gen, int T, int Tmax,
                     const int *stop_id, int n_stop)
{
    int k = 0;
    while (k < emitn && nout + k < gen && T + k < Tmax) {
        const int id = emit[k++];
        for (int s = 0; s < n_stop; s++)
            if (id == stop_id[s]) return k;
    }
    return k;
}

/* `cap` log rows per KDA layer among the first nl, and nothing for MLA layers. work is
 * the one-layer work state the tentative calls run on; NULL allocates it, and a second
 * model passes the first one's, since the two never run at the same time. Returns 0, or
 * -1 when out of memory. */
static int spec_log_alloc(Weights *w, const K3Cfg *c, int nl, int cap, float *work)
{
    /* one log per EXECUTED KDA position: each pass of a looped layer has its own */
    const int nv = w->vmap ? w->n_virt : nl;
    w->kda_slot = (int *)malloc((size_t)nv * sizeof(int));
    if (!w->kda_slot) return -1;
    w->n_kda = 0;
    for (int v = 0; v < nv; v++)
        w->kda_slot[v] = k3_is_mla(c, w->vmap ? w->vmap[v] : v) ? -1 : w->n_kda++;
    w->log_cap = cap;
    w->log_row0 = -1;
    w->log_floats = (size_t)cap * k3_kda_log_row(c);
    w->kda_log = (float *)malloc(w->log_floats * (size_t)(w->n_kda > 0 ? w->n_kda : 1)
                                 * sizeof(float));
    w->kda_work = work ? work : (float *)malloc(k3_kda_state_floats(c) * sizeof(float));
    return (w->kda_log && w->kda_work) ? 0 : -1;
}

int main(int argc, char **argv)
{
    /* Informational flags are answered before anything else, because they must work
     * without a model directory, `k3 --help` on a machine with no checkpoint is the
     * first thing most people type. */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("k3 %s\n", K3_VERSION); return 0; }
        if (!strcmp(argv[i], "--list-presets")) { k3_preset_list(stdout); return 0; }
    }
    if (argc < 2) { usage(stderr); return 2; }
#ifndef K3_TEST_MUTANTS
    /* gate mutants exist only in test builds (make mutants); a production binary refuses the switch */
    if (getenv("K3_MLA_SPLIT_MUTANT_CAUSAL")) {
        fprintf(stderr, "K3_MLA_SPLIT_MUTANT_CAUSAL is a gate mutant for test builds (make mutants); this build refuses it\n");
        return 2;
    }
    if (getenv("K3_LOOP_MUTANT_SHARED_STATE")) {
        fprintf(stderr, "K3_LOOP_MUTANT_SHARED_STATE is a gate mutant for test builds (make mutants); this build refuses it\n");
        return 2;
    }
    if (getenv("K3_LOOP_MUTANT_NO_COMPACT")) {
        fprintf(stderr, "K3_LOOP_MUTANT_NO_COMPACT is a gate mutant for test builds (make mutants); this build refuses it\n");
        return 2;
    }
    if (getenv("K3_PREFILL_MUTANT_RESET")) {
        fprintf(stderr, "K3_PREFILL_MUTANT_RESET is a gate mutant for test builds (make mutants); this build refuses it\n");
        return 2;
    }
    if (getenv("K3_PREFILL_MUTANT_RELATIVE")) {
        fprintf(stderr, "K3_PREFILL_MUTANT_RELATIVE is a gate mutant for test builds (make mutants); this build refuses it\n");
        return 2;
    }
#endif

    const char *dir = argv[1];
    if (dir[0] == '-') {
        fprintf(stderr, "the first argument must be the model directory, got '%s'\n\n", dir);
        usage(stderr);
        return 2;
    }
    const char *ids_s = NULL, *outp = "k3_run.json", *trunk_dir = NULL;
    /* Expert-cache diagnostics are opt-in. They are only meaningful for cache research,
     * and writing them unconditionally drops two undeclared files into whatever
     * directory the user happened to run from. */
    const char *trace_dir = NULL;
    const char *expert_profile = NULL;
    int pin_experts = 0;
    const char *logits_path = NULL, *all_logits_path = NULL;
    const char *prompt_text = NULL, *prompt_file = NULL, *tok_dir = NULL;
    const char *cfg_path = NULL;
    int gen = 8, want_layers = -1;
    int score_prompt = 0, score_start = 1;
    int exit_status = 0;
    int reread_prompt = 0;
    /* --stop-id, repeatable. Generation halts AFTER emitting a listed id, so the state
     * written by --save-state still contains it and a later --load-state continues the
     * sequence the model actually produced. Without this the engine always runs to
     * --gen, which for a chat-tuned checkpoint means paying seconds per token for text
     * past the end-of-message marker that a caller will only throw away. */
    int stop_id[8]; int n_stop = 0, hit_stop = 0, stopped_at = -1;
    double cache_gb = 64.0, trunk_gb = 16.0;
    double memory_plan_bytes = 0.0, all_logits_bytes = 0.0;
    int budget_auto = 0;
    int spec_n = 0;
    int tf_check = 0;
    const char *draft_dir = NULL;
    double draft_gb = 6.0;
    const char *load_state = NULL, *save_state = NULL;
    const char *preset_name = NULL;
    int incremental = 0, ultra = 0, stream_lm_head = 0, expert_pipeline = 0;
    int kv_latent = 0, trunk_rows = 0, kvb_mode = K3_KVB_OFF;
    int loop_a = -1, loop_b = -1, loop_count = 0;   /* --loop-layers A-B, --loop-count R */
    double loop_eps = -1.0;                         /* --loop-exit EPS; < 0 is the fixed loop */
    int no_loop = 0, want_loop = 0, loop_preset = 0; /* --no-loop; --loop; its span in force */
    const char *ids_file = NULL;                    /* --ids-file */
    /* --prefill-chunk: positions per forward of an incremental read-in. 4096 keeps the
     * per-position buffers (about 0.67 MB per row at K3's shapes) near 2.7 GB whatever
     * the prompt, against about 3.6 s of compute per position on a 4-vCPU AVX-512 box:
     * re-reading a streamed trunk once per 4096 positions is small beside that. */
    int prefill_chunk = 4096, prefill_chunk_set = 0;
    const char *ckpt_path = NULL;   /* --checkpoint */
    int pause_after = 0;            /* --pause-after */
    int ckpt_resumed = 0;           /* ids a --checkpoint already held when this run began */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(argv[i], "--ids-file") && i + 1 < argc) ids_file = argv[++i];
        else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) {
            char *end;
            errno = 0;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || errno || value < 1 || value > K3_MAX_PASS) {
                fprintf(stderr, "--prefill-chunk takes positions per pass from 1 to %d, "
                                "got '%s'\n", K3_MAX_PASS, argv[i]);
                return 2;
            }
            prefill_chunk = (int)value;
            prefill_chunk_set = 1;
        }
        else if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) ckpt_path = argv[++i];
        else if (!strcmp(argv[i], "--pause-after") && i + 1 < argc) {
            char *end;
            errno = 0;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || errno || value < 1 || value > INT_MAX) {
                fprintf(stderr, "--pause-after takes a number of chunks from 1, got '%s'\n",
                        argv[i]);
                return 2;
            }
            pause_after = (int)value;
        }
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--reread-prompt")) reread_prompt = 1;
        else if (!strcmp(argv[i], "--score-prompt")) score_prompt = 1;
        else if (!strcmp(argv[i], "--score-start") && i + 1 < argc) {
            char *end;
            errno = 0;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || errno || value < 1 || value >= K3_MAX_PROMPT) {
                fprintf(stderr, "--score-start must be a valid positive target position\n");
                return 2;
            }
            score_start = (int)value;
        }
        else if (!strcmp(argv[i], "--tok") && i + 1 < argc) tok_dir = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stop-id") && i + 1 < argc) {
            if (n_stop >= (int)(sizeof stop_id / sizeof stop_id[0])) {
                fprintf(stderr, "--stop-id given more than %d times\n",
                        (int)(sizeof stop_id / sizeof stop_id[0]));
                return 2;
            }
            /* strtol, not atoi: atoi("abc") is 0, which is a real token id, so a typo
             * would silently arm a stop on a token the model may well emit. Refuse
             * anything that is not entirely a non-negative integer; the range check
             * against the vocabulary has to wait until config.json has been read. */
            {
                char *end;
                const long v = strtol(argv[++i], &end, 10);
                if (*argv[i] == '\0' || *end != '\0' || v < 0) {
                    fprintf(stderr, "--stop-id %s: expected a non-negative token id\n",
                            argv[i]);
                    return 2;
                }
                stop_id[n_stop++] = (int)v;
            }
        }
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--expert-profile") && i + 1 < argc)
            expert_profile = argv[++i];
        else if (!strcmp(argv[i], "--pin-experts") && i + 1 < argc) {
            char *end;
            errno = 0;
            const long v = strtol(argv[++i], &end, 10);
            if (*argv[i] < '0' || *argv[i] > '9' || *end || errno || v < 1 || v > INT_MAX) {
                fprintf(stderr, "--pin-experts requires a positive integer\n");
                return 2;
            }
            pin_experts = (int)v;
        }
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) want_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_dir = argv[++i];
        else if (!strcmp(argv[i], "--trunk-rows")) trunk_rows = 1;
        else if (!strcmp(argv[i], "--mla-split") && i + 1 < argc) {
            /* strict: digits only. atoi read "banana" as 0 and "2junk" as 2 (Astra's K1 review) */
            const char *s = argv[++i];
            const size_t nd = strspn(s, "0123456789");
            if (nd == 0 || s[nd] != '\0' || nd > 4 || atoi(s) > 4096) {
                fprintf(stderr, "--mla-split takes a block size from 0 (off) to 4096, got '%s'\n", s);
                return 2;
            }
            k3_mla_split_block = atoi(s);
        }
        else if (!strcmp(argv[i], "--kvb-cache") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "off")) kvb_mode = K3_KVB_OFF;
            else if (!strcmp(m, "active")) kvb_mode = K3_KVB_ACTIVE;
            else if (!strcmp(m, "pin")) kvb_mode = K3_KVB_PIN;
            else { fprintf(stderr, "--kvb-cache takes off, active or pin\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--loop-layers") && i + 1 < argc) {
            /* strict "A-B", digits only: atoi would read "3-x" as 3 and loop the wrong span */
            const char *s = argv[++i];
            const size_t na = strspn(s, "0123456789");
            const size_t nb_ = (na > 0 && na <= 4 && s[na] == '-')
                ? strspn(s + na + 1, "0123456789") : 0;
            if (nb_ == 0 || nb_ > 4 || s[na + 1 + nb_] != '\0') {
                fprintf(stderr, "--loop-layers takes a zero-based inclusive layer range "
                                "A-B, got '%s'\n", s);
                return 2;
            }
            loop_a = atoi(s);
            loop_b = atoi(s + na + 1);
            if (loop_b < loop_a) {
                fprintf(stderr, "--loop-layers %s: the range ends before it starts\n", s);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--loop-count") && i + 1 < argc) {
            const char *s = argv[++i];
            if (strspn(s, "0123456789") != strlen(s) || !*s || strlen(s) > 1 ||
                atoi(s) < 1 || atoi(s) > 8) {
                fprintf(stderr, "--loop-count takes the number of passes over the "
                                "--loop-layers span, 1 to 8, got '%s'\n", s);
                return 2;
            }
            loop_count = atoi(s);
        }
        else if (!strcmp(argv[i], "--loop-exit") && i + 1 < argc) {
            const char *s = argv[++i];
            char *end = NULL;
            const double e = strtod(s, &end);
            if (end == s || *end != '\0' || !(e >= 0.0) || !isfinite(e)) {
                fprintf(stderr, "--loop-exit takes a relative change >= 0 (0 never exits "
                                "early), got '%s'\n", s);
                return 2;
            }
            loop_eps = e;
        }
        else if (!strcmp(argv[i], "--no-loop")) no_loop = 1;
        else if (!strcmp(argv[i], "--loop")) want_loop = 1;
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf-check")) tf_check = 1;
        else if (!strcmp(argv[i], "--load-state") && i + 1 < argc) load_state = argv[++i];
        else if (!strcmp(argv[i], "--save-state") && i + 1 < argc) save_state = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk-gb") && i + 1 < argc) draft_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "auto")) budget_auto = 1;
            else { trunk_gb = atof(v); budget_auto = 0; }
        }
        else if (!strcmp(argv[i], "--incremental")) incremental = 1;
        else if (!strcmp(argv[i], "--kv-latent")) kv_latent = 1;
        else if (!strcmp(argv[i], "--ultra-low-memory")) ultra = 1;
        else if (!strcmp(argv[i], "--expert-pipeline")) expert_pipeline = 1;
        else if (!strcmp(argv[i], "--stream-lm-head")) stream_lm_head = 1;
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-all-logits") && i + 1 < argc)
            all_logits_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-cache-trace") && i + 1 < argc) trace_dir = argv[++i];
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc && !strcmp(argv[i + 1], "auto")) {
            /* Not in the table: the table is fixed budgets, auto is computed from this
             * machine's MemAvailable at startup, below, once parsing is complete. */
            i++;
            budget_auto = 1;
            preset_name = "auto";
            ultra = 0;
        }
        else if (!strcmp(argv[i], "--preset") && i + 1 < argc) {
            const K3Preset *p = k3_preset_find(argv[++i]);
            if (!p) {
                fprintf(stderr, "unknown preset '%s'\n\n", argv[i]);
                k3_preset_list(stderr);
                return 2;
            }
            /* A preset sets the budget; an explicit --trunk-gb/--cache-gb after it still
             * wins, because the flags are applied in argv order. */
            trunk_gb = p->trunk_gb;
            cache_gb = p->cache_gb;
            preset_name = p->name;
            ultra = p->ultra;
        }
        else if (!strcmp(argv[i], "--list-presets")) { k3_preset_list(stdout); return 0; }
        else if (!strcmp(argv[i], "--version")) {
            printf("k3 %s\n", K3_VERSION);
            return 0;
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown option %s\n\n", argv[i]); usage(stderr); return 2; }
    }

    /* Full recompute carries no KV cache at all, so there is no layout to choose. A
     * flag that silently does nothing is worse than one that refuses. */
    if (kv_latent && !incremental) {
        fprintf(stderr, "--kv-latent needs --incremental: full recompute holds no KV "
                        "cache to compress\n");
        return 2;
    }
    if (ckpt_path && !incremental) {
        fprintf(stderr, "--checkpoint needs --incremental: only an incremental read-in "
                        "carries state that can be saved part way\n");
        return 2;
    }
    if (pause_after && !ckpt_path) {
        fprintf(stderr, "--pause-after needs --checkpoint, or the chunks read would be "
                        "lost\n");
        return 2;
    }
    if (ckpt_path && load_state) {
        fprintf(stderr, "--checkpoint resumes a read-in of this prompt and --load-state "
                        "continues a saved session; give one of them\n");
        return 2;
    }
    if (prefill_chunk_set && !incremental) {
        fprintf(stderr, "--prefill-chunk needs --incremental: full recompute carries no "
                        "cache from one pass to the next, so it cannot read in parts\n");
        return 2;
    }
    const int head_streamed = ultra || stream_lm_head;
    /* An omitted --trunk-gb is not refused: the 16 GB default becomes the row budget,
     * which the row pipeline caps at two 8 MiB buffers plus the current layer's vectors,
     * and which the memory plan below charges in full. Only a computed (auto) budget and
     * a draft trunk are refused. */
    if (trunk_rows && (!trunk_dir || draft_dir || budget_auto)) {
        fprintf(stderr, "--trunk-rows needs --trunk; --trunk-gb auto, --preset auto and "
                        "draft trunks are unsupported\n");
        return 2;
    }
    /* only the --kv-latent passes reread kv_b rows, and only the row pipeline streams them */
    if (kvb_mode != K3_KVB_OFF && (!trunk_rows || !kv_latent)) {
        fprintf(stderr, "--kvb-cache needs --trunk-rows and --kv-latent\n");
        return 2;
    }
    if (trunk_rows && (!isfinite(trunk_gb) || trunk_gb <= 0 || trunk_gb * 1e9 >= (double)INT64_MAX)) {
        fprintf(stderr, "--trunk-rows requires a finite positive trunk budget\n");
        return 2;
    }
    if (stream_lm_head && draft_dir) {
        fprintf(stderr, "--stream-lm-head does not yet support --draft-trunk\n");
        return 2;
    }
    /* --loop-layers changes the model, so every mode it is combined with must have been
     * shown to compute the looped stack; the ones not covered by its gates refuse rather
     * than run something unverified (docs/notes/layer-loop.md, Modes). */
    if (loop_count && loop_a < 0 && !want_loop) {
        fprintf(stderr, "--loop-count needs --loop-layers or --loop\n");
        return 2;
    }
    if (loop_eps >= 0.0 && loop_a < 0 && !want_loop) {
        fprintf(stderr, "--loop-exit needs --loop-layers or --loop\n");
        return 2;
    }
    if (no_loop && (loop_a >= 0 || want_loop)) {
        fprintf(stderr, "--no-loop runs the released stack; it cannot be combined with "
                        "--loop or --loop-layers\n");
        return 2;
    }
    if (want_loop && loop_a >= 0) {
        fprintf(stderr, "--loop loops the AttnRes block holding the middle layer; for "
                        "another span give --loop-layers alone\n");
        return 2;
    }
    /* NO LOOP BY DEFAULT: the released model. The loop ran by default from 2026-10-19 to
     * 2026-10-20 (Andrew's call); on 2026-10-20 he left the call to Claude ("whatever you
     * think is good do it"), and it went opt-in because K3 was not trained to repeat
     * layers, no quality measurement exists, and it costs layers per token
     * (docs/notes/layer-loop.md). --loop is his original adaptive setting: the AttnRes
     * block holding the middle layer, up to 4 passes, a token stopping once a pass moves
     * its span output by under 5%. Its span is settled once the layer count is known. */
    loop_preset = want_loop;
    if (loop_a >= 0 || loop_preset) {
        if (!loop_count) loop_count = loop_preset ? 4 : 2;
        if (loop_preset && loop_eps < 0.0) loop_eps = 0.05;
        /* Every mode runs the loop: each keeps its per-layer state per EXECUTED position
         * (KDA slots, MLA rows, --spec logs, the state file's slots), the trunk binds and
         * the expert cache take any layer order, and --dump-cache-trace records requests
         * in executed order (tools/expert_profile.py's prefix parser accepts only
         * unlooped traces and says so). tests/test_loop_adaptive.py holds each to the
         * looped full recompute. */
    }
    if (score_prompt) {
        if (incremental || load_state || save_state || spec_n || draft_dir || tf_check ||
            reread_prompt || n_stop || want_layers != -1 || trace_dir || logits_path ||
            all_logits_path) {
            fprintf(stderr, "--score-prompt requires a complete teacher-forced model; "
                            "generation/state/partial-layer diagnostics cannot be combined\n");
            return 2;
        }
        gen = 0;
    } else if (score_start != 1) {
        fprintf(stderr, "--score-start needs --score-prompt\n");
        return 2;
    }
    if ((expert_profile != NULL) != (pin_experts > 0)) {
        fprintf(stderr, "--expert-profile and --pin-experts must be supplied together\n");
        return 2;
    }
    if (expert_profile && k3_cache_check_profile(expert_profile, pin_experts)) return 2;
    if (ultra && !trunk_dir) {
        fprintf(stderr, "--ultra-low-memory needs --trunk; resident trunk cannot fit its "
                        "memory contract\n");
        return 2;
    }
    if (ultra && budget_auto) {
        fprintf(stderr, "--ultra-low-memory uses explicit bounded budgets; use "
                        "--preset ultra or pass --trunk-gb/--cache-gb\n");
        return 2;
    }
    if (ultra && (spec_n > 0 || draft_dir)) {
        fprintf(stderr,
                "--ultra-low-memory does not yet support --spec or --draft-trunk; "
                "use deterministic serial decode\n");
        return 2;
    }
    /* Speculative depth is settled here rather than where its buffers are allocated,
     * because the rollback log is sized from it and the memory plan below must count it. */
    if (spec_n < 0) spec_n = 0;
    if (spec_n > 0 && !incremental) {
        fprintf(stderr, "--spec needs --incremental; ignoring --spec\n");
        spec_n = 0;
    }
    if (draft_dir && (!incremental || !trunk_dir)) {
        fprintf(stderr, "--draft-trunk needs --incremental and --trunk; ignoring\n");
        draft_dir = NULL;
    }
    if (draft_dir && spec_n <= 0) spec_n = 4;
    if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;
    {
        int nsrc = (ids_s != NULL) + (ids_file != NULL) + (prompt_text != NULL)
                 + (prompt_file != NULL);
        if (nsrc == 0) {
            fprintf(stderr, "one of --ids, --ids-file, --prompt or --prompt-file is "
                            "required\n");
            return 2;
        }
        if (nsrc > 1) {
            /* Refuse rather than pick: silently preferring one source would make a
             * mistyped invocation run the WRONG prompt for tens of minutes. */
            fprintf(stderr, "--ids, --ids-file, --prompt and --prompt-file are mutually "
                            "exclusive\n");
            return 2;
        }
    }

    /* ---- auto budget ----
     * RAM-first: per token the engine re-reads the ENTIRE streamed trunk but only
     * ~25.8 GB of experts, and steady-state expert caching yields nothing until the
     * arena is tens of GB. Measured: a gigabyte pinned in the trunk is worth roughly
     * 70x a gigabyte of expert cache at the margin. So auto gives the trunk everything
     * this machine has, minus a safety margin, and the cache gets real memory only
     * after the whole 110 GB trunk would be resident. */
    if (budget_auto) {
        const double avail = mem_available_bytes();
        if (avail <= 0.0) {
            fprintf(stderr, "--preset auto needs /proc/meminfo; pass explicit "
                            "--trunk-gb/--cache-gb on this platform\n");
            return 2;
        }
        /* Fixed costs outside both budgets: embeddings + lm_head 4.70 GB, safetensors
         * index, recurrent state 0.63 GB, KV cache and scratch. Reserve them plus a
         * 2 GB + 2% margin so auto never invites the OOM killer. */
        const double tables = head_streamed
            ? 2.35 + ((double)K3_MODEL_STREAM_CHUNK + 2.0 * K3_ST_ALIGN) / 1e9 : 4.70;
        const double reserve = 2.0 + 0.02 * (avail / 1e9) + tables + 1.70;
        double usable = avail / 1e9 - reserve;
        const double slot_min = 2.5;   /* one ring slot + headroom; refuse below */
        const double cache_min = 0.5;  /* topk+1 expert slots is ~0.3 GB */
        if (usable < slot_min + cache_min) {
            fprintf(stderr, "auto: only %.1f GB usable after the %.1f GB reserve; "
                            "below the %.1f GB floor. Pass explicit budgets.\n",
                    usable, reserve, slot_min + cache_min);
            return 2;
        }
        const double trunk_full = 111.0;   /* full packed trunk + widen headroom */
        if (usable - cache_min >= trunk_full) {
            /* Full residency: per-token trunk reads disappear entirely. This is the
             * configuration auto exists for. */
            trunk_gb = trunk_full;
            cache_gb = usable - trunk_full;
        } else {
            /* Partial pinning has WEAK returns and real hazards, both measured on the
             * released checkpoint: pinning 51 of 109 GB ran 14% SLOWER than pinning
             * nothing (48.2 vs 42.1 s/token) because peak RSS at ~90% of RAM put the
             * kernel into reclaim and the device served the remaining tail of the
             * packed trunk a third slower, while a moderate pin stayed neutral to
             * mildly positive (40.1 s/token at 25 GB, device throughput unharmed).
             * So below full residency, auto pins only while the whole process stays
             * comfortably clear of the RAM ceiling. */
            double memtotal = 0.0;
#ifdef _WIN32
            {
                MEMORYSTATUSEX ms;
                ms.dwLength = sizeof ms;
                if (GlobalMemoryStatusEx(&ms)) memtotal = (double)ms.ullTotalPhys;
            }
#else
            FILE *mf = fopen("/proc/meminfo", "r");
            if (mf) {
                char ln[256];
                while (fgets(ln, sizeof ln, mf))
                    if (!strncmp(ln, "MemTotal:", 9)) { memtotal = atof(ln + 9) * 1024.0; break; }
                fclose(mf);
            }
#endif
            const double rss_ceiling = memtotal > 0.0 ? 0.55 * memtotal / 1e9
                                                      : usable;   /* no /proc: keep old cap */
            double cap = rss_ceiling - reserve - cache_min;
            if (cap < slot_min) cap = slot_min;
            trunk_gb = usable - cache_min;
            if (trunk_gb > cap) trunk_gb = cap;
            cache_gb = cache_min;
        }
        printf("auto budget: %.1f GB available, %.1f GB reserved -> trunk %.1f GB / "
               "expert cache %.1f GB\n", avail / 1e9, reserve, trunk_gb, cache_gb);
    }

    /* fa is sized for the released 24 MLA layers with generous headroom; k3_cfg_load
     * refuses a config that would overrun it rather than truncating the layer map. */
    K3Cfg c; static int fa[128];
    if (!real_cfg(&c, fa, 128, dir, cfg_path)) {
        fprintf(stderr, "ABORTED: the model config could not be read with confidence.\n");
        return 2;
    }
    if (want_layers > 0 && want_layers < c.n_layers) {
        printf("NOTE: binding only the first %d of %d layers. Output is NOT the full "
               "model; it is a partial stack for testing the machinery.\n\n",
               want_layers, c.n_layers);
    }

    /* ---- prompt ----
     * Three entry points, one representation. --ids is the reproducible channel every
     * fixture and the oracle use, and stays the default for validation work. --prompt /
     * --prompt-file tokenize here in C, which is what makes the engine text-in/text-out
     * without a Python step. The tokenizer is loaded ONLY when actually needed, so the
     * id path keeps working on a box that has no tokenizer files at all. */
    /* Heap, not stack. This was `int prompt[4096]` and it was the reason the engine
     * refused prompts longer than 4096 ids -- a stack-array size, not a model or memory
     * limit. */
    /* One id past the ceiling, so that a tokenized prompt which reaches it is seen to be
     * too long: tok_encode stops at its max without saying so, which would otherwise
     * answer a truncated prompt. */
    int *prompt = (int *)malloc(((size_t)K3_MAX_PROMPT + 1) * sizeof(int));
    if (!prompt) { fprintf(stderr, "OOM allocating prompt buffer\n"); return 2; }
    int np = 0;
    Tok tok; int have_tok = 0;
    char *ids_text = NULL;   /* --ids-file's contents, parsed as --ids would be */
    if (ids_file) {
        long ilen = 0;
        ids_text = tk_read_file(ids_file, &ilen);   /* exits if unreadable */
        if ((long)strlen(ids_text) != ilen) {
            fprintf(stderr, "--ids-file %s contains a NUL byte\n", ids_file);
            return 2;
        }
        ids_s = ids_text;
    }

    if (prompt_text || prompt_file) {
        if (!tok_dir) {
            fprintf(stderr, "--prompt/--prompt-file need --tok DIR (the directory with "
                            "tiktoken.model and tokenizer_config.json)\n");
            return 2;
        }
        k3_tok_load(&tok, tok_dir);
        have_tok = 1;

        char *ptext = NULL; long plen = 0;
        if (prompt_file) {
            ptext = tk_read_file(prompt_file, &plen);   /* exits if unreadable */
        } else {
            plen  = (long)strlen(prompt_text);
            ptext = (char *)malloc((size_t)plen + 1);
            if (!ptext) { fprintf(stderr, "OOM on prompt\n"); return 2; }
            memcpy(ptext, prompt_text, (size_t)plen + 1);
        }
        np = tok_encode(&tok, ptext, (int)plen, prompt, K3_MAX_PROMPT + 1);
        free(ptext);
        if (np > K3_MAX_PROMPT) {
            fprintf(stderr, "prompt exceeds the %d-id ceiling (%ld bytes of text)\n",
                    K3_MAX_PROMPT, plen);
            return 2;
        }
        printf("  tokenized: %ld bytes -> %d ids\n", plen, np);
    } else {
        for (const char *p = ids_s; *p; ) {
            while (*p == ',' || isspace((unsigned char)*p)) p++;
            if (!*p) break;
            if (np == K3_MAX_PROMPT) {
                fprintf(stderr, "prompt exceeds the %d-id ceiling\n", K3_MAX_PROMPT);
                return 2;
            }
            char *end = NULL;
            errno = 0;
            long id = strtol(p, &end, 10);
            if (end == p || errno == ERANGE || id < 0 || id >= c.vocab ||
                (*end && *end != ',' && !isspace((unsigned char)*end))) {
                fprintf(stderr, "--ids contains an invalid token id\n");
                return 2;
            }
            prompt[np++] = (int)id;
            p = end;
        }
    }
    free(ids_text);
    ids_text = NULL;
    ids_s = NULL;
    if (np <= 0) { fprintf(stderr, "no prompt ids parsed\n"); return 2; }
    if (score_prompt && score_start >= np) {
        fprintf(stderr, "--score-prompt needs at least one target after --score-start\n");
        return 2;
    }
    for (int i = 0; i < np; i++)
        if (prompt[i] < 0 || prompt[i] >= c.vocab) {
            fprintf(stderr, "token id %d is outside the vocabulary of %d\n", prompt[i], c.vocab);
            return 2;
        }

    const int original_prompt_tokens = np;
    if (reread_prompt) {
        if (np > K3_MAX_PROMPT / 2) {
            fprintf(stderr, "--reread-prompt needs twice the prompt context: "
                            "%d original ids exceed the %d-id reread limit\n",
                    np, K3_MAX_PROMPT / 2);
            return 2;
        }
        /* Copy token IDs: concatenating text and tokenizing again can merge across
         * the boundary. A resumed session repeats only this request, not history. */
        memcpy(prompt + np, prompt, (size_t)np * sizeof *prompt);
        np *= 2;
        printf("reread prompt: %d original ids, %d processed before generation\n",
               original_prompt_tokens, np);
    }

    /* Validate the request before allocating anything.
     *
     * Refuse rather than clamp: a caller who asks for more tokens than this build
     * supports should be told, not quietly handed fewer. The decode loop's own guard
     * (T >= Tmax) is a backstop, not a bounds check. */
    if (gen < 0 || gen > K3_MAX_GEN) {
        fprintf(stderr, "--gen %d is out of range: this build generates at most %d "
                        "tokens (outtok[%d])\n", gen, K3_MAX_GEN, K3_MAX_GEN);
        return 2;
    }
    /* A stop id the model can never emit gives a run that never stops, which is
     * indistinguishable from a model that simply did not produce one, and the two
     * need different fixes. Refuse it here, where the vocabulary is finally known. */
    for (int s = 0; s < n_stop; s++)
        if (stop_id[s] >= c.vocab) {
            fprintf(stderr, "--stop-id %d is outside the vocabulary of %d\n",
                    stop_id[s], c.vocab);
            return 2;
        }
    if (np > K3_MAX_PROMPT) {
        fprintf(stderr, "prompt of %d ids exceeds the %d-id ceiling (seq[%d])\n",
                np, K3_MAX_PROMPT, K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    /* Full recompute, --score-prompt and --tf-check put the whole sequence through one
     * forward pass, whose rows stay within K3_MAX_PASS; only --incremental reads in parts. */
    if (!incremental && np > K3_MAX_PASS) {
        fprintf(stderr, "prompt of %d ids exceeds the %d-id ceiling of one forward pass; "
                        "with --incremental it is read in parts, up to %d ids%s\n",
                np, K3_MAX_PASS, K3_MAX_PROMPT,
                score_prompt || tf_check ? " (--score-prompt and --tf-check are one pass)"
                                         : "");
        return 2;
    }
    /* --checkpoint: a file already at PATH is an earlier read-in of this prompt. Its ids
     * must begin the prompt, with at least one position left to read; the run then
     * continues as a --load-state of it with the rest of the prompt. */
    if (ckpt_path) {
        FILE *probe = fopen(ckpt_path, "rb");
        if (probe) {
            fclose(probe);
            K3StateHdr ch;
            K3StateLoop cl;
            if (k3_state_peek(ckpt_path, &ch, &cl) != 0) return 1;
            if (ch.nseq < 0 || ch.nseq > np || ch.cached < 0 || ch.cached >= np) {
                fprintf(stderr, "REFUSING: checkpoint %s holds %d ids, %d of them read in; this "
                                "prompt has %d, so it is not a read-in of this prompt.\n"
                                "  Delete it, or pass another --checkpoint path.\n",
                        ckpt_path, ch.nseq, ch.cached, np);
                return 2;
            }
            int *saved = (int *)malloc((size_t)(ch.nseq > 0 ? ch.nseq : 1) * sizeof(int));
            if (!saved || k3_state_read_ids(ckpt_path, &ch, saved) != 0) {
                free(saved);
                return 1;
            }
            int diff = -1;
            for (int i = 0; i < ch.nseq && diff < 0; i++)
                if (saved[i] != prompt[i]) diff = i;
            if (diff >= 0) {
                fprintf(stderr, "REFUSING: checkpoint %s was read from a different prompt: its "
                                "id at position %d is %d, this prompt's is %d.\n"
                                "  Delete it, or pass another --checkpoint path.\n",
                        ckpt_path, diff, saved[diff], prompt[diff]);
                free(saved);
                return 2;
            }
            free(saved);
            printf("resuming the read-in from %s: %d of %d positions already read\n",
                   ckpt_path, ch.cached, np);
            memmove(prompt, prompt + ch.nseq, (size_t)(np - ch.nseq) * sizeof *prompt);
            np -= ch.nseq;
            ckpt_resumed = ch.nseq;
            load_state = ckpt_path;
        }
    }
    K3StateHdr shd;
    K3StateLoop sloop;
    int prior = 0;
    if (load_state) {
        if (!incremental) {
            fprintf(stderr, "--load-state needs --incremental\n");
            return 2;
        }
        if (k3_state_peek(load_state, &shd, &sloop) != 0) return 1;
        prior = shd.nseq;
    }
    /* The longest read-in this run makes: the prompt, after the saved pending id on a
     * resume, or with a draft trunk on a resume the whole sequence (the draft's state is
     * not saved). A chunk longer than that would only size buffers nothing fills. The
     * header's counts are checked when the state loads; here they only bound a size. */
    const int pending = !load_state ? 0 : prior - shd.cached < 0 ? 0
                      : prior - shd.cached > 1 ? 1 : prior - shd.cached;
    if (incremental) {
        int longest = np + pending;
        if (draft_dir && load_state && prior + np > longest) longest = prior + np;
        if (prefill_chunk > longest) prefill_chunk = longest;
    }
    if (prior < 0 || prior > K3_MAX_PROMPT + K3_MAX_GEN - np - gen - 1) {
        fprintf(stderr, "history %d + prompt %d + gen %d + 1 exceeds the "
                        "%d-position ceiling\n",
                prior, np, gen, K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    /* THE REAL CONTEXT LIMIT is the MLA KV cache, not any array size. Check it against
     * what the kernel says is actually available and refuse with both numbers, rather
     * than letting a long prompt get 40 minutes into a run and then be OOM-killed. Only
     * incremental decode allocates the KV cache; full recompute carries no cache. */
    if (incremental) {
        /* The per-position figure is the LAYOUT'S, not a constant: --kv-latent is the
         * one dial that moves it, so a plan printed with the expanded number while the
         * latent cache is allocated would overstate the limit 43-fold. */
        /* From the config, not the released checkpoint's constants (K3_KV_BYTES_PER_POS
         * and K3_KV_LATENT_BYTES_PER_POS, which this reproduces for K3): a smaller model
         * was refused by K3's figure. A loop's repeated MLA rows come on top; the full
         * plan below counts them. */
        int n_mla_cfg = 0;
        for (int L = 0; L < c.n_layers; L++) n_mla_cfg += k3_is_mla(&c, L) != 0;
        const double per_pos = (double)n_mla_cfg
            * ((double)kv_floats_per_pos(&c, kv_latent) + c.qk_rope) * sizeof(float);
        const double kv_need = (double)(prior + np + gen + 1) * per_pos;
        const double avail   = mem_available_bytes();
        char kb[32], ab[32];
        human(kv_need, kb, sizeof kb);
        human(avail, ab, sizeof ab);
        printf("  KV cache : %s for %d positions (%.3f MB/position, %s)\n",
               kb, prior + np + gen + 1, per_pos / 1e6,
               kv_latent ? "LATENT, k and v rebuilt on use" : "expanded k and v");
        if (avail > 0.0 && kv_need > avail * 0.9) {
            fprintf(stderr,
                "\nREFUSING: the KV cache for %d positions needs %s but only %s is\n"
                "available. This is a MEMORY limit, not an engine ceiling: MLA caches\n"
                "%s in fp32 across %d layers, so context costs %.3f MB per\n"
                "position regardless of budget. %s"
                "Shorten the request%s.\n",
                prior + np + gen + 1, kb, ab,
                kv_latent ? "the compressed latent" : "expanded k and v", n_mla_cfg,
                per_pos / 1e6,
                kv_latent ? "" : "Try --kv-latent, which caches the\n"
                                 "compressed latent instead (0.055 MB per position on "
                                 "K3) and rebuilds k and v.\n",
                np <= K3_MAX_PASS ? ", or use full recompute (drop --incremental),\n"
                                    "which carries no KV cache at all" : "");
            return 2;
        }
    }

    char b1[32];
    printf("Kimi K3, pure C, released checkpoint\n");
    /* The directory, not a shard count: the index has not been built yet at this point.
     * The count is printed by the "indexed N tensors from M shards" line below, once
     * k3_st_open has actually counted them. */
    printf("  model    : %s\n", dir);
    printf("  prompt   : %d tokens, generating %d\n", np, gen);
    /* Echo the preset so a captured log is self-describing: a timing figure is
     * meaningless without the budget that produced it. */
    if (preset_name)
        printf("  preset   : %s (trunk %.2f GB / expert cache %.2f GB)\n",
               preset_name, trunk_gb, cache_gb);
    printf("\n");

    K3St st;
    double t0 = now_s();
    if (k3_st_open(&st, dir) != 0) return 1;
    printf("indexed %d tensors from %d shards in %.2f s\n", st.nt, st.nshard, now_s() - t0);

    /* ---- how much will this take? Report BEFORE allocating, so a box that cannot
     * hold it fails with a number rather than an OOM kill. ---- */
    const int NL = (want_layers > 0 && want_layers < c.n_layers) ? want_layers : c.n_layers;

    /* ---- --loop-layers: the order layers execute in (docs/notes/layer-loop.md) ----
     * Whole AttnRes blocks only. Snapshots fire on layer_idx % attn_res_block == 0, so a
     * span of whole blocks leaves every executed layer at the position inside its block
     * it was trained at, layer_idx can stay the physical index, and each repeated block
     * adds one snapshot. A span that cut a block would shift every later boundary. */
    int NV = NL, n_mla_rep = 0;
    int *vmap = NULL, *vslot = NULL;
    char loop_json[192] = "null";
    if (loop_preset) {
        const int blk = c.attn_res_block, mid = NL / 2;
        loop_a = mid - mid % blk;
        loop_b = loop_a + blk - 1 < NL - 1 ? loop_a + blk - 1 : NL - 1;
    }
    if (loop_a >= 0) {
        const int blk = c.attn_res_block;
        if (loop_b >= NL || loop_a % blk != 0 || ((loop_b + 1) % blk != 0 && loop_b != NL - 1)) {
            fprintf(stderr, "--loop-layers %d-%d: the span must be whole AttnRes blocks of %d "
                            "layers within the %d bound: start at a multiple of %d and end "
                            "just before one, or at layer %d\n",
                    loop_a, loop_b, blk, NL, blk, NL - 1);
            return 2;
        }
        const int span = loop_b - loop_a + 1;
        NV = NL + (loop_count - 1) * span;
        vmap  = (int *)malloc((size_t)NV * sizeof(int));
        vslot = (int *)malloc((size_t)NV * sizeof(int));
        if (!vmap || !vslot) return 1;
        int nv = 0;
        for (int L = 0; L <= loop_b; L++) vmap[nv++] = L;
        for (int r = 1; r < loop_count; r++)
            for (int L = loop_a; L <= loop_b; L++) vmap[nv++] = L;
        for (int L = loop_b + 1; L < NL; L++) vmap[nv++] = L;
        int shared = 0;   /* the gate mutant: repeats reuse the first pass's state and rows */
#ifdef K3_TEST_MUTANTS
        shared = getenv("K3_LOOP_MUTANT_SHARED_STATE") != NULL;
#endif
        /* a repeat's first-pass position is its physical index, since 0..B execute first */
        for (int v = 0; v < NV; v++)
            vslot[v] = (shared && v > loop_b && v <= loop_b + (loop_count - 1) * span)
                       ? vmap[v] : v;
        c.attn_res_extra = (loop_count - 1) * ((span + blk - 1) / blk);
        for (int L = loop_a; L <= loop_b; L++) n_mla_rep += k3_is_mla(&c, L);
        n_mla_rep *= loop_count - 1;
        char ej[32] = "null";
        if (loop_eps >= 0.0) snprintf(ej, sizeof ej, "%.9g", loop_eps);
        snprintf(loop_json, sizeof loop_json,
                 "{\"layers\":[%d,%d],\"count\":%d,\"executed_layers\":%d,\"exit\":%s,"
                 "\"preset\":%s}",
                 loop_a, loop_b, loop_count, NV, ej, loop_preset ? "true" : "false");
        if (loop_eps >= 0.0)
            printf("%s: layers %d-%d loop adaptively, 2 to %d passes per token (a token "
                   "stops once a pass moves its span output by under %g relative), up to %d "
                   "layers per token over %d bound. This is NOT the released model: K3 was "
                   "not trained to repeat layers, and no quality result attaches to the "
                   "output.%s\n\n", loop_preset ? "LAYER LOOP (--loop)" : "EXPERIMENTAL",
                   loop_a, loop_b, loop_count, loop_eps, NV, NL,
                   loop_preset ? " Without --loop the CLI runs the released model." : "");
        else
            printf("EXPERIMENTAL: --loop-layers %d-%d x %d runs %d layers per token over %d "
                   "bound. This is NOT the released model: K3 was not trained to repeat "
                   "layers, and no quality result attaches to the output.\n\n",
                   loop_a, loop_b, loop_count, NV, NL);
    }
    int64_t total = 0; int missing = 0;
    for (int L = 0; L < NL; L++) {
        const int64_t n = k3_bind_layer_bytes(&st, &c, L);
        if (n < 0) { missing++; continue; }
        total += n;
    }
    if (missing) {
        fprintf(stderr, "\n%d of %d layers are missing tensors in this shard set. "
                        "A partial download cannot run the model.\n", missing, NL);
        return 1;
    }
    human((double)total, b1, sizeof b1);
    /* Report the mode actually in effect. The trunk is either resident or streamed and
     * the two have very different memory profiles, so the banner must reflect the real
     * choice rather than a default. */
    if (trunk_dir)
        printf("trunk on disk : %s total (STREAMED from %s, not held in RAM)\n",
               b1, trunk_dir);
    else
        printf("resident trunk: %s in RAM (large matrices kept in the checkpoint's bf16,\n"
               "  fp32 only for the norms and biases that kernels read elementwise)\n", b1);

    /* Runs that need every position's logits project them a block at a time: see
     * K3_LOGIT_ROWS and forward(). Sized here so the memory plan below counts it. */
    int logit_rows = 1;
    if (score_prompt)    logit_rows = np - score_start;   /* the positions scored       */
    else if (tf_check)   logit_rows = np;                 /* every position             */
    else if (spec_n > 0) logit_rows = spec_n + 1;         /* one verify sweep; a draft
                                                           * trunk has set spec_n above */
    if (logit_rows > K3_LOGIT_ROWS) logit_rows = K3_LOGIT_ROWS;
    if (logit_rows < 1) logit_rows = 1;

    /* Add up EVERYTHING before allocating anything. Being OOM-killed halfway through
     * binding wastes the whole load and reports nothing useful; a refusal with the two
     * numbers side by side says exactly what box this needs. */
    {
        const int64_t E64 = c.hidden;
        const double w_trunk = trunk_dir ? trunk_gb * 1e9 : (double)total;
        const double w_model = (ultra ? 0.0 : (double)c.vocab * E64 * 2)
            + (head_streamed ? (double)K3_MODEL_STREAM_CHUNK + 2.0 * K3_ST_ALIGN
                             : (double)c.vocab * E64 * 2)
            + 3.0 * E64 * 4;
        const double w_cache = cache_gb * 1e9;
        const int Tm = prior + np + gen + 1;
        const int Tr = pass_rows(incremental, prefill_chunk, Tm);   /* rows per pass */
        const int mb = k3_attn_res_stride(&c);
        const int Pp = c.kda_heads * c.kda_head_dim;
        const int state_layers = (ultra && !incremental) ? 1 : NV;
        const double w_state = (double)((size_t)Pp * c.kda_head_dim
                              + (size_t)3 * Pp * (c.conv_k - 1)) * state_layers * 4;
        /* the allocation below, the same call */
        const size_t scratch = k3_layer_scratch_cached(&c, Tr, Tm, kv_latent);
        const int adaptive = loop_eps >= 0.0 && loop_a >= 0;
        const double w_buf = ((double)Tr * E64 + (double)Tr * mb * E64
                              + (double)scratch + (double)c.vocab
                              + (logit_rows > 1 ? (double)logit_rows * c.vocab : 0.0)
                              /* adaptive loop: last span outputs, a pass's compacted rows */
                              + (adaptive ? 2.0 * Tr * E64 + (double)Tr * mb * E64 : 0.0)) * 4
                           /* and what it records per absolute position */
                           + (adaptive ? (double)Tm * (sizeof(int)
                                         + sizeof(double) * ((size_t)loop_count + 1)) : 0.0);
        /* The KV cache MUST be in this total: it is the only term that grows with
         * context, so a guard that omits it is blind to the one thing it exists to
         * catch. k3_mla_cached stores expanded per-head k and v plus the shared rope
         * slot, in fp32, across all 24 MLA layers -- 2.37 MB per position, so a
         * 4096-token prompt alone is 9.7 GB. Under --kv-latent it stores the kv_lora
         * latent plus the same rope slot, 0.055 MB per position, and the plan must
         * report what will actually be allocated. */
        int n_mla = 0;
        for (int L = 0; L < c.n_layers; L++) if (k3_is_mla(&c, L)) n_mla++;
        n_mla += n_mla_rep;   /* --loop-layers: each repeated MLA layer has its own rows */
        const double w_kv = incremental
            ? (double)Tm * n_mla
              * ((double)kv_floats_per_pos(&c, kv_latent) + c.qk_rope) * 4
            : 0.0;
        /* Speculative decode: per KDA layer, one log row per position a sweep can commit
         * (spec_n + 1, for the exact model and again for a hybrid draft), plus a single
         * layer's work state that every tentative call runs on. No copy of the carried
         * state, and nothing for MLA layers. See the allocation below. */
        int n_kda = 0;
        for (int L = 0; L < NL; L++) if (!k3_is_mla(&c, L)) n_kda++;
        const double w_spec = spec_n > 0
            ? 4.0 * ((double)n_kda * (spec_n + 1) * (draft_dir ? 2.0 : 1.0)
                     * (double)k3_kda_log_row(&c) + (double)k3_kda_state_floats(&c))
            : 0.0;
        /* --dump-all-logits under --spec (or a draft trunk) keeps every position of a verify
         * sweep's logits until the kept ones are written: K3_SPEC_MAX + 1 vectors, the
         * allocation made with the dump file below. */
        const double w_dump = (all_logits_path && spec_n > 0)
            ? (double)(K3_SPEC_MAX + 1) * c.vocab * sizeof(float) : 0.0;
        const double need_b = w_trunk + w_model + w_cache + w_state + w_buf + w_kv + w_spec
                            + w_dump;
        memory_plan_bytes = need_b;
        all_logits_bytes = w_dump;
        const double have = mem_available_bytes();

        char b2[32], b3[32], b4[32], b5[32], b6[32], b7[32];
        human(w_kv, b7, sizeof b7);
        human(w_trunk, b1, sizeof b1); human(w_model, b2, sizeof b2);
        human(w_cache, b3, sizeof b3); human(w_state, b4, sizeof b4);
        human(w_buf, b5, sizeof b5);   human(need_b, b6, sizeof b6);
        printf("\nmemory plan\n");
        printf("  trunk %-10s %s\n  embed + lm_head  %s %s\n  expert cache     %s\n"
               "  recurrent state  %s\n  buffers          %s\n  KV cache         %s%s\n",
               trunk_rows ? "(ROWS)" : trunk_dir ? "(STREAMED)" : "(resident)", b1, b2,
               ultra ? "(STREAMED)" : (head_streamed ? "(lm_head streamed)" : "(resident)"),
               b3, b4, b5, b7, kv_latent ? "  (latent layout)" : "");
        /* The plan charges a row budget as given; say so, since the row pipeline itself
         * never holds more than two 8 MiB buffers and one layer's vectors. */
        if (trunk_rows)
            printf("                   (the whole --trunk-gb budget is charged; rows allocate "
                   "at most two 8 MiB buffers\n                    and the current layer's "
                   "vectors, reported once the trunk is open)\n");
        if (w_spec > 0.0) {
            human(w_spec, b7, sizeof b7);
            printf("  spec log         %s  (KDA recurrence inputs per position)\n", b7);
        }
        if (w_dump > 0.0) {
            human(w_dump, b7, sizeof b7);
            printf("  all-logits dump  %s  (every position of one verify sweep)\n", b7);
        }
        if (incremental) {
            /* what step 0 feeds: the prompt, and on a resume the saved pending token */
            const int n_in = np + pending;
            const int passes = (n_in + prefill_chunk - 1) / prefill_chunk;
            printf("                   (buffers hold %d positions; %d to read in, in %d "
                   "pass%s of up to %d)\n", Tr, n_in, passes, passes == 1 ? "" : "es",
                   prefill_chunk);
        }
        printf("  TOTAL            %s\n", b6);
        if (have > 0.0) {
            human(have, b1, sizeof b1);
            printf("  available        %s\n", b1);
            if (need_b > have * 0.95) {
                human(need_b - have, b2, sizeof b2);
                fprintf(stderr,
                        "\nREFUSING TO START: this needs %s and the machine has %s "
                        "available, a shortfall of %s.\n"
                        "Options: a larger box, a smaller --cache-gb, or fewer --layers.\n",
                        b6, b1, b2);
                return 1;
            }
        }
        printf("\n");
    }

    /* Reject checkpoint geometry, missing selected tensors and impossible pin budgets
     * after metadata indexing but before binding any trunk, embedding or head weights. */
    K3Cache cache;
    /* The cache reads its own switch out of the environment, so that a run driven by
     * K3_EXPERT_PIPELINE and one driven by the flag are the same run. Set it BEFORE init,
     * which is where the cache looks. */
    if (expert_pipeline) {
#ifdef _WIN32
        _putenv_s("K3_EXPERT_PIPELINE", "1");
#else
        setenv("K3_EXPERT_PIPELINE", "1", 1);
#endif
        printf("expert pipelining: ON (experts are published as their reads land)\n");
    }
    if (k3_cache_init(&cache, &st, &c, (int64_t)(cache_gb * 1e9)) != 0) return 1;
    if (expert_profile) {
        if (k3_cache_load_profile(&cache, expert_profile, pin_experts)) {
            k3_cache_free(&cache);
            return 1;
        }
        printf("expert profile: %d lazy pins, %d slots remain evictable\n",
               cache.profile_pins, cache.nslot - cache.profile_pins);
    }

    Weights w; memset(&w, 0, sizeof w);
    w.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
    if (!w.lay) return 1;
    w.vmap = vmap;      /* NULL without --loop-layers */
    w.loop_a = loop_a; w.loop_b = loop_b; w.loop_rmax = loop_count;
    w.loop_span = loop_a >= 0 ? loop_b - loop_a + 1 : 0;
    w.loop_eps = vmap ? loop_eps : -1.0;
    w.vslot = vslot;
    w.n_virt = NV;

    static K3Trunk trunk;
    t0 = now_s();
    if (trunk_dir) {
        /* STREAMED. Nothing is bound up front: each layer is read from the packed trunk
         * on fast local storage as the forward pass reaches it. RAM stops being a floor
         * and becomes a dial, and unlike quantisation it costs no accuracy, which
         * matters because the K3 report (4.1.4) keeps exactly these tensors in higher
         * precision on purpose. */
        const int rc = trunk_rows
            ? k3_trunk_open_rows_kvb(&trunk, trunk_dir, &c, (int64_t)(trunk_gb * 1e9), kvb_mode)
            : k3_trunk_open(&trunk, trunk_dir, &c, (int64_t)(trunk_gb * 1e9));
        if (rc != 0) return 1;
        if (trunk.n_layers < NL) {
            fprintf(stderr, "packed trunk has %d layers, need %d\n", trunk.n_layers, NL);
            return 1;
        }
        w.trunk = &trunk;
        w.n_bound = NL;
        printf("trunk streaming enabled from %s in %.1f s\n", trunk_dir, now_s() - t0);
    } else {
        for (int L = 0; L < NL; L++) {
            if (k3_bind_layer(&st, &c, L, &w.lay[L]) != 0) {
                fprintf(stderr, "bind failed at layer %d\n", L); return 1;
            }
            w.n_bound = L + 1;
            if ((L + 1) % 10 == 0 || L + 1 == NL) {
                printf("  bound %d/%d layers, %.1f s elapsed\n", L + 1, NL, now_s() - t0);
                fflush(stdout);
            }
        }
        const double t_bind = now_s() - t0;
        printf("trunk loaded in %.1f s (%.0f MB/s from disk)\n",
               t_bind, (double)total / 1e6 / t_bind);
    }

    t0 = now_s();
    w.ultra = ultra;
    w.stream_lm_head = head_streamed;
    if (k3_bind_model_parts(&st, &c, !ultra, !head_streamed, &w.mb) != 0) return 1;
    if (head_streamed && k3_model_stream_init(&w.ms, &st, &c) != 0) return 1;
    human((double)w.mb.nbytes, b1, sizeof b1);
    if (ultra)
        printf("final norms: %s resident; embedding and lm_head streamed in %.1f s\n\n",
               b1, now_s() - t0);
    else if (head_streamed)
        printf("embedding and final norms: %s resident; lm_head streamed in %.1f s\n\n",
               b1, now_s() - t0);
    else
        printf("embedding, final norm and lm_head: %s in %.1f s\n\n", b1, now_s() - t0);

    {   /* The plan is a forecast. This is the outcome. */
        char rb[32];
        human(peak_rss_bytes(), rb, sizeof rb);
        printf("peak RSS after loading weights: %s  (the plan above is a forecast, "
               "this is measured)\n", rb);
    }
    printf("expert cache: %d slots x %.2f MB = %.2f GB (%.2f%% of the 1.45 TB expert pool)\n\n",
           cache.nslot, (double)cache.slot_bytes / 1e6,
           (double)cache.nslot * cache.slot_bytes / 1e9,
           100.0 * cache.nslot / (double)(92 * c.n_experts));

    /* ---- buffers ----
     * A resumed session must hold the saved history as well as the new tokens, so the
     * KV cache and every per-position buffer are sized for both. The header was
     * validated with the prompt before loading weights; the payload is restored after. */
    if (load_state) {
        printf("resuming from %s: %d prior positions, %d new\n\n", load_state, prior, np);
    }
    const int Tmax = prior + np + gen + 1;
    /* Rows of h, br, the scratch and the adaptive loop's per-token buffers: one pass's
     * positions, which under --incremental is the read-in chunk (or a verify sweep), not
     * the whole context. Only the KV cache and the per-position records span Tmax. */
    const int Tb = pass_rows(incremental, prefill_chunk, Tmax);
    w.max_rows = Tb;
    const int E = c.hidden;
    const int maxb = k3_attn_res_stride(&c);
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * c.kda_head_dim + (size_t)3 * P * (c.conv_k - 1);

    /* The model-level aggregator lays out fold[E] followed by (nb+1) source rows of E
     * inside scratch, and nb reaches n_layers/attn_res_block = 8 at full depth. So
     * scratch must hold at least (maxb + 2) * hidden floats. k3_layer_scratch includes
     * exactly that term, but an off-by-one here would overwrite whatever follows
     * without any symptom until the logits came out subtly wrong, so it is checked
     * rather than assumed. */
    {
        const size_t need_scratch = (size_t)(maxb + 2) * E;
        const size_t have_scratch = k3_layer_scratch(&c, Tb);
        if (have_scratch < need_scratch) {
            fprintf(stderr, "scratch is %zu floats, the attn-res aggregator needs %zu\n",
                    have_scratch, need_scratch);
            return 1;
        }
    }

    float *h  = (float *)malloc((size_t)Tb * E * sizeof(float));
    float *br = (float *)malloc((size_t)Tb * maxb * E * sizeof(float));
    const int state_layers = (ultra && !incremental) ? 1 : NV;
    float *ks = (float *)malloc(kper * (size_t)state_layers * sizeof(float));
    /* The cached MLA path sizes its score buffer by cache capacity, not by T (the latent
     * layout one row per head, and per query of an --mla-split block, plus a rebuild
     * buffer), and it gets the region after the layer's own buffers. This was
     * max(layer scratch, MLA scratch), short by those buffers whenever the MLA term won:
     * --kv-latent --mla-split 300 over a 300-id prompt wrote past the end on the tiny
     * model under AddressSanitizer. */
    const size_t sc_need = k3_layer_scratch_cached(&c, Tb, Tmax, kv_latent);
    float *sc = (float *)malloc(sc_need * sizeof(float));
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
    if (!h || !br || !ks || !sc || !lg) { fprintf(stderr, "buffer allocation failed\n"); return 1; }
    if (w.loop_eps >= 0.0) {
        /* adaptive depth: per-token flags and counts, each token's last span output, the
         * compacted rows of a pass, and what every absolute position did (the report) */
        w.loop_pcached = (int *)calloc((size_t)w.loop_rmax + 1, sizeof(int));
        w.loop_sweep_pcached = (int *)calloc((size_t)w.loop_rmax + 1, sizeof(int));
        /* per row of a pass: Tb; per absolute position (the report): Tmax */
        w.loop_act     = (int *)calloc((size_t)Tb, sizeof(int));
        w.loop_idx     = (int *)calloc((size_t)Tb, sizeof(int));
        w.loop_nbt     = (int *)calloc((size_t)Tb, sizeof(int));
        w.loop_passes  = (int *)calloc((size_t)Tmax, sizeof(int));
        w.loop_d       = (double *)calloc((size_t)Tmax * (w.loop_rmax + 1), sizeof(double));
        w.loop_hprev   = (float *)malloc((size_t)Tb * E * sizeof(float));
        w.loop_hc      = (float *)malloc((size_t)Tb * E * sizeof(float));
        w.loop_brc     = (float *)malloc((size_t)Tb * maxb * E * sizeof(float));
        if (!w.loop_pcached || !w.loop_sweep_pcached || !w.loop_act || !w.loop_idx || !w.loop_nbt || !w.loop_passes ||
            !w.loop_d || !w.loop_hprev || !w.loop_hc || !w.loop_brc) {
            fprintf(stderr, "adaptive loop buffer allocation failed\n");
            return 1;
        }
    }
    if (logit_rows > 1) {
        w.logit_block = (float *)malloc((size_t)logit_rows * c.vocab * sizeof(float));
        if (!w.logit_block) { fprintf(stderr, "logit block allocation failed\n"); return 1; }
        w.logit_rows = logit_rows;
    }
    human((double)(kper * state_layers) * 4, b1, sizeof b1);
    if (state_layers == 1 && NL > 1)
        printf("recurrent state: one %s slot, cleared and reused across %d layers\n\n",
               b1, NL);
    else
        printf("recurrent state for %d layers: %s\n\n", state_layers, b1);

    /* ---- generate ----
     * Heap and sized from the ACTUAL request, not from the ceiling. These were
     * `int seq[K3_MAX_PROMPT + K3_MAX_GEN]` and `int outtok[K3_MAX_GEN]` on the stack,
     * which is why the ceiling had to stay small enough to be a stack array. */
    int *seq = (int *)malloc((size_t)(prior + np + gen + 8) * sizeof(int));
    int *outtok = (int *)malloc((size_t)(gen + 8) * sizeof(int));
    if (!seq || !outtok) { fprintf(stderr, "OOM allocating sequence buffers\n"); return 1; }
    /* On a resume the saved history occupies the front of the sequence and the prompt
     * given now is its continuation; the restore below fills seq[0..prior). */
    memcpy(seq + prior, prompt, (size_t)np * sizeof(int));
    int T = prior + np;
    int nout = 0;
    char *generated_text = NULL;

    /* ---- optional incremental decode ----
     * Full recompute re-runs the whole prefix every step, so expert traffic grows with
     * context: measured 99.7 -> 126.0 GB across just three tokens. Incremental prefills
     * once and then feeds ONE token per step, carrying the KDA recurrent state (which
     * k3_kda_layer already updates in place) and an MLA KV cache. Validated by GATE 3
     * of the tiny-model oracle, which requires the SAME tokens as full recompute. */
    if (incremental) {
        /* one MLA slot per executed MLA position; without --loop-layers position v is
         * layer v, and with it a repeat owns its rows unless vslot (the gate mutant)
         * points it at its first pass */
        w.mla_slot = (int *)malloc((size_t)NV * sizeof(int));
        if (!w.mla_slot) return 1;
        w.n_mla = 0;
        for (int v = 0; v < NV; v++) {
            const int L = vmap ? vmap[v] : v;
            const int sv = vslot ? vslot[v] : v;
            w.mla_slot[v] = !k3_is_mla(&c, L) ? -1 : sv == v ? w.n_mla++ : w.mla_slot[sv];
        }
        w.kv_cap = Tmax;
        w.kv_latent = kv_latent;
        const size_t kvper = (size_t)w.kv_cap * kv_floats_per_pos(&c, w.kv_latent);
        const size_t rpper = (size_t)w.kv_cap * c.qk_rope;
        const double kvb = (double)(kvper + rpper) * w.n_mla * sizeof(float);
        human(kvb, b1, sizeof b1);
        printf("incremental decode: KV cache %s for %d MLA layers at %d positions "
               "(%s)\n\n", b1, w.n_mla, w.kv_cap,
               w.kv_latent ? "latent, k and v rebuilt through kv_b on every use"
                           : "expanded k and v");
        w.kvc   = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
        w.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));
        if (!w.kvc || !w.ropec) { fprintf(stderr, "KV cache allocation failed\n"); return 1; }
        memset(ks, 0, kper * (size_t)NV * sizeof(float));
        w.cached = 0;

        if (load_state) {
            const double tl = now_s();
            K3StateLoop rl = {-1, -1, 0, w.n_bound, -1.0};
            if (vmap) { rl.a = loop_a; rl.b = loop_b; rl.count = loop_count; rl.nv = NV;
                        rl.eps = w.loop_eps; }
            if (k3_state_load(load_state, &c, &shd, &sloop, &rl, seq, ks, w.kvc, w.ropec,
                              w.n_bound, w.n_mla, w.kv_cap,
                              (int64_t)kv_floats_per_pos(&c, w.kv_latent),
                              (int64_t)c.qk_rope, w.loop_pcached, w.loop_passes) != 0)
                return 1;
            w.cached = shd.cached;
            printf("restored %d positions in %.2f s: decode continues without "
                   "re-reading the prior context\n\n", w.cached, now_s() - tl);
        }
    }

    /* ---- speculative decode: tentative sweeps ----
     * A verify sweep feeds the pending token and nd drafts through every layer at once,
     * and the model may reject some of them. MLA layers need nothing undone, because
     * their KV rows are positional (see K3KdaLog in k3.h for why no stale row can be
     * read). KDA layers fold every position into their state in place, so the old scheme
     * copied the whole carried state before each sweep, restored it on a rejection and
     * replayed the accepted prefix through a SECOND forward: at K3 scale that re-reads the
     * 108.81 GB trunk and the prefix's experts, part of why an eager drafter measured
     * 0.91x on code (spec_draft says what the rest was).
     *
     * Instead the sweep is tentative. Each KDA layer runs on a one-layer work copy and
     * leaves its carried state alone, recording its recurrence inputs per position; once
     * the accepted count is known, spec_commit applies exactly the kept positions with
     * k3_kda_advance. The result is bit-identical to serial decode, no weight is read
     * twice, and no copy of the carried state exists at all. A sweep can commit all
     * spec_n + 1 of its positions, so that is the row count. */
    if (spec_n > 0) {
        if (spec_log_alloc(&w, &c, NL, spec_n + 1, NULL) != 0) {
            fprintf(stderr, "OOM for the --spec log\n");
            return 1;
        }
        char lb[32], rb[32], wb[32];
        human((double)w.log_floats * w.n_kda * sizeof(float), lb, sizeof lb);
        human((double)k3_kda_log_row(&c) * w.n_kda * sizeof(float), rb, sizeof rb);
        human((double)k3_kda_state_floats(&c) * sizeof(float), wb, sizeof wb);
        printf("speculative decode: up to %d drafted tokens per sweep, %s, verified "
               "batched\n  sweep log %s (%d positions x %s across %d KDA layers) + one "
               "work layer %s;\n  no state snapshot, and a rejected draft costs no second "
               "sweep\n\n", spec_n, draft_dir ? "proposed by the draft trunk"
                                              : "n-gram lookup",
               lb, spec_n + 1, rb, w.n_kda, wb);
    }

    /* ---- hybrid decode: a second, typically quantized, trunk drafts ----
     * The draft model shares everything that is identical between the two models: the
     * embedding, the lm_head, the layer map, and the routed experts (the qdq derivation
     * touches only 2D trunk tensors). It differs ONLY in trunk weights, so it needs its
     * own trunk stream, its own layer bindings, and its own recurrent/KV state. Output
     * exactness is structural: drafts feed the SAME batched greedy verification as
     * --spec, so what gets emitted is precisely what the exact model would have chosen.
     * Measured teacher-forced agreement of an int8-derived draft on the released
     * checkpoint is 94.2 percent against a 96.2 percent measurement ceiling, which is
     * what makes the draft worth consulting at all. */
    static K3Trunk trunk_d;
    Weights dw; memset(&dw, 0, sizeof dw);
    /* The draft writes its logits into dlg, never into lg: lg holds the EXACT model's
     * vector for the step, which is what --dump-logits and --dump-all-logits record. */
    float *dks = NULL, *dlg = NULL;
    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0, draft_dropped = 0;
    if (draft_dir) {
        /* --draft-trunk was checked against --incremental/--trunk and given its spec depth
         * before the memory plan. */
        if (k3_trunk_open(&trunk_d, draft_dir, &c, (int64_t)(draft_gb * 1e9)) != 0)
            return 1;
        dw.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
        dks   = (float *)calloc(k3_kda_state_floats(&c) * (size_t)w.n_bound,
                                sizeof(float));
        dlg   = (float *)malloc((size_t)c.vocab * sizeof(float));
        const size_t kvperd = (size_t)w.kv_cap * kv_floats_per_pos(&c, kv_latent);
        const size_t rpperd = (size_t)w.kv_cap * c.qk_rope;
        dw.kvc   = (float *)calloc(kvperd * (size_t)w.n_mla, sizeof(float));
        dw.ropec = (float *)calloc(rpperd * (size_t)w.n_mla, sizeof(float));
        /* The draft proposes through tentative calls too, and commits what the exact
         * model keeps, from its own log. It needs spec_n + 1 rows as well: after a
         * fully accepted round it has not fed the last accepted draft yet, and its
         * next round's first call feeds that token together with the new pending one
         * instead of paying a sweep of its own to catch up. It shares the exact
         * model's work layer, since the two never run at once. */
        if (!dw.lay || !dks || !dlg || !dw.kvc || !dw.ropec ||
            spec_log_alloc(&dw, &c, NL, spec_n + 1, w.kda_work) != 0) {
            fprintf(stderr, "OOM for the draft model state\n"); return 1;
        }
        dw.mb = w.mb;              /* embed + lm_head are the same tensors */
        dw.trunk = &trunk_d;
        dw.n_bound = w.n_bound;
        dw.mla_slot = w.mla_slot;  /* read-only map, safely shared */
        dw.n_mla = w.n_mla;
        dw.kv_cap = w.kv_cap;
        dw.max_rows = w.max_rows;     /* it runs in the same h, br and scratch */
        dw.kv_latent = w.kv_latent;   /* one layout for both models, or one stride
                                       * is wrong and the draft reads garbage */
        dw.cached = 0;
        dw.draft_mode = 1;   /* cache-only routing: draft tokens read no new experts */
        printf("hybrid decode: draft trunk %s (%.1f GB budget) proposes up to %d "
               "tokens per sweep;\n               the exact model verifies every one "
               "before it is emitted\n\n", draft_dir, draft_gb, spec_n);
    }

    /* What --spec allocated, for the report: log rows for each model plus the one work
     * layer they share. */
    const double spec_log_bytes = (double)sizeof(float)
        * ((w.kda_log ? w.log_floats * w.n_kda + k3_kda_state_floats(&c) : 0)
           + (dw.kda_log ? dw.log_floats * dw.n_kda : 0));

    if (score_prompt) {
        double *losses = (double *)calloc((size_t)np - 1, sizeof(double));
        if (!losses) { exit_status = 1; goto cleanup; }
        w.score_targets = seq + 1;
        w.score_nll = losses;
        w.score_start = score_start - 1;
        k3_expert_drops = 0;
        if (forward(&w, &c, &cache, seq, np - 1, lg, sc, h, br, ks, NULL) != 0 ||
            w.layers_completed != (vmap ? NV : c.n_layers) || k3_expert_drops) {
            fprintf(stderr, "quality run invalid; no scores published\n");
            free(losses);
            exit_status = 1;
            goto cleanup;
        }
        double sum = 0.0;
        for (int i = score_start - 1; i < np - 1; i++) sum += losses[i];
        const int count = np - score_start;
        const double mean = sum / count;
        FILE *sf = fopen(outp, "w");
        if (!sf) { perror(outp); exit_status = 1; }
        else {
            fprintf(sf, "{\"schema\":1,\"task\":\"perplexity\",\"input_ids\":[");
            for (int i = 0; i < np; i++) fprintf(sf, "%s%d", i ? "," : "", seq[i]);
            fputs("],\"loop_adaptive\":", sf);
            write_loop_adaptive(sf, &w, np);
            fprintf(sf, ",\"score_start\":%d,\"scored_tokens\":%d,"
                        "\"layers\":%d,\"layers_completed\":%d,\"loop\":%s,\"expert_drops\":0,"
                        "\"nll_sum\":%.17g,\"mean_nll\":%.17g,\"perplexity\":",
                    score_start, count, c.n_layers, w.layers_completed, loop_json, sum, mean);
            if (mean <= log(DBL_MAX)) fprintf(sf, "%.17g", exp(mean));
            else fputs("null", sf); /* mean_nll remains defined when exp overflows */
            fputs(",\"token_nll\":[", sf);
            for (int i = score_start - 1; i < np - 1; i++)
                fprintf(sf, "%s%.17g", i == score_start - 1 ? "" : ",", losses[i]);
            fputs("]}\n", sf);
            const int failed = ferror(sf);
            if (fclose(sf) || failed) { perror(outp); exit_status = 1; }
        }
        free(losses);
        goto cleanup;
    }

    /* --tf-check: teacher-forced agreement over the whole --ids sequence in ONE sweep.
     * Prediction i is the argmax after positions 0..i; it is compared to the id the
     * sequence actually continues with. This is the acceptance rate a draft model
     * would see under batched greedy verification, measured directly, and it is the
     * one number a quantized-draft design stands on. Free-running comparisons cannot
     * measure it: a single early divergence changes every later context. */
    if (tf_check) {
        if (np < 2) { fprintf(stderr, "--tf-check needs at least 2 ids\n"); return 2; }
        int *arg = (int *)malloc((size_t)np * sizeof(int));
        if (!arg) { fprintf(stderr, "OOM for --tf-check\n"); return 1; }
        const double t0c = now_s();
        if (forward(&w, &c, &cache, seq, np, lg, sc, h, br, ks, arg) != 0) {
            fprintf(stderr, "forward failed in --tf-check\n");
            return 1;
        }
        int match = 0;
        for (int i = 0; i + 1 < np; i++) match += (arg[i] == seq[i + 1]);
        printf("teacher-forced agreement: %d/%d positions (%.1f%%) in %.1f s\n",
               match, np - 1, 100.0 * match / (np - 1), now_s() - t0c);
        printf("  per-position (p=predicted a=actual): ");
        for (int i = 0; i + 1 < np; i++)
            if (arg[i] != seq[i + 1])
                printf("[%d p=%d a=%d] ", i, arg[i], seq[i + 1]);
        printf("\n");
        FILE *tf = fopen(outp, "w");
        if (tf) {
            fprintf(tf, "{\"tf_positions\":%d,\"tf_matches\":%d,\"tf_agreement\":%.4f}\n",
                    np - 1, match, (double)match / (np - 1));
            fclose(tf);
        }
        free(arg);
        return 0;
    }

    printf("%-6s %-10s %-12s %-10s %-10s %s\n",
           "STEP", "TOKEN", "SECONDS", "REUSE %", "EXPERT GB", "TOK/S");
    printf("--------------------------------------------------------------------\n");
    k3_expert_drops = 0;
    double t_total = 0.0;
    /* Per-step cache statistics are reset each iteration so the columns below describe
     * that step alone. The end-of-run summary needs whole-run totals, so accumulate the
     * expert side here; the trunk side is already cumulative. Comparing a cumulative
     * figure against a single step would misstate the I/O share. */
    double expert_s_total = 0.0, expert_gb_total = 0.0;
    uint64_t expert_reqs_total = 0, expert_evict_total = 0, expert_bytes_total = 0;
    uint64_t expert_reuses_total = 0;
    /* Speculative accounting. Every decode step runs exactly ONE exact-model forward, a
     * verify sweep included, so forwards == steps is the proof that nothing was replayed. */
    long spec_sweeps = 0, spec_drafted = 0, spec_accepted = 0, spec_full = 0;
    long spec_partial = 0, spec_cut = 0, spec_dropped = 0;
    int steps = 0;
    /* (drafted, accepted, kept) per verify sweep, for the JSON: a sweep emits at least
     * one id, so there are at most gen of them. */
    /* --checkpoint: saved under the same loop record and layout a --save-state would use */
    K3Checkpoint ckpt;
    memset(&ckpt, 0, sizeof ckpt);
    if (ckpt_path) {
        ckpt.path = ckpt_path;
        ckpt.pause_after = pause_after;
        ckpt.seq = seq;
        ckpt.c = &c;
        ckpt.loop.a = ckpt.loop.b = -1;
        ckpt.loop.count = 0;
        ckpt.loop.nv = w.n_bound;
        ckpt.loop.eps = -1.0;
        if (vmap) { ckpt.loop.a = loop_a; ckpt.loop.b = loop_b; ckpt.loop.count = loop_count;
                    ckpt.loop.nv = NV; ckpt.loop.eps = w.loop_eps; }
        ckpt.kper = (int64_t)kper;
        ckpt.kvpp = (int64_t)kv_floats_per_pos(&c, w.kv_latent);
        ckpt.ropepp = (int64_t)c.qk_rope;
    }
    int *spec_trace = (int *)malloc((size_t)3 * (gen + 1) * sizeof(int));
    if (!spec_trace) { fprintf(stderr, "OOM for the --spec trace\n"); return 1; }
    FILE *all_logits_f = NULL;
    if (all_logits_path) {
        all_logits_f = fopen(all_logits_path, "wb");
        if (!all_logits_f) { perror(all_logits_path); return 1; }
        if (w.kda_log) {
            w.all_logits = (float *)malloc((size_t)(K3_SPEC_MAX + 1) * c.vocab
                                           * sizeof(float));
            if (!w.all_logits) { fprintf(stderr, "OOM for --dump-all-logits\n"); return 1; }
        }
    }
    /* `nout < gen` drives generation; the `g == 0` disjunct additionally runs the
     * incremental prefill once even when --gen 0, so the prompt's KV and recurrent
     * state are computed and can be saved with ZERO generated tokens. That is what
     * lets --gen 0 --save-state warm a reusable prefix (e.g. a chat system prompt)
     * whose recurrent state is exact rather than one generated token past the end. */
    for (int g = 0; nout < gen || (incremental && g == 0); g++) {
        k3_cache_reset_stats(&cache);
        const double ts = now_s();
        int frc;
        int emit[K3_SPEC_MAX + 1];
        int emitn = 0;
        int keep = -1;                   /* ids of emit[] the output keeps; see emit_keep */
        const float *step_logits = NULL; /* per-position logits of a verify sweep */
        if (incremental && g == 0) {
            /* Step 0 feeds everything not yet consumed: the whole prompt on a fresh
             * run, and on a resume the carried pending token PLUS the new prompt.
             * T - base covers both exactly; feeding np here instead dropped the last
             * new token from a resumed batch, and the first generated token then came
             * from a context one token short: fluent, plausible, and wrong. */
            const int base = w.cached;
            const int nT0 = T - base;
            /* in passes of at most --prefill-chunk positions; w.cached ends at T */
            if (ckpt.path) signal(SIGINT, k3_on_sigint);
            frc = prefill(&w, &c, &cache, seq + base, nT0, prefill_chunk, lg, sc, h, br, ks,
                          "prompt", &ckpt);
            if (ckpt.path) signal(SIGINT, SIG_DFL);
            if (frc == 0 && ckpt.paused) {
                printf("\npaused: %d of %d positions read and saved to %s.\n"
                       "Run the same command again to continue from there.\n",
                       w.cached, T, ckpt.path);
                exit_status = 3;
                goto cleanup;
            }
            if (frc == 0 && ckpt.path && ckpt.written)
                printf("  checkpoint %s is kept: the same command again resumes from it "
                       "instead of reading those positions again. Delete it when done.\n",
                       ckpt.path);
            else if (frc == 0 && ckpt.path)
                printf("  checkpoint: nothing saved to %s, the rest of the prompt was read in "
                       "one pass\n", ckpt.path);
            if (frc == 0) emit[emitn++] = argmax_(lg, c.vocab);
            /* The draft model must absorb the same context, or its first proposals
             * come from a shorter one; one draft sweep, paid once. Saved state does
             * not include the draft's, so a resumed run replays the WHOLE sequence
             * through the draft once; correctness never depends on this, only
             * acceptance does. */
            if (dw.trunk && frc == 0) {
                const int db = load_state ? 0 : base;
                dw.cached = db;
                if (prefill(&dw, &c, &cache, seq + db, base + nT0 - db, prefill_chunk, dlg,
                            sc, h, br, dks, "draft", NULL) != 0)
                    frc = -1;
            }
        } else if (incremental) {
            const int base = w.cached;
            int d[K3_SPEC_MAX], nd = 0, dlag = 0;
            frc = 0;
            if (w.kda_log && T + spec_n + 1 < Tmax && base + spec_n + 1 <= w.kv_cap) {
                if (dw.trunk) {
                    /* The draft model proposes, chaining its own argmax one position at a
                     * time. Its first call also feeds any accepted token it has not seen:
                     * after a fully accepted round that is the last draft, which the exact
                     * model consumed but the draft never fed. Folding it in here saves the
                     * draft a sweep per fully accepted round. Every call is tentative and
                     * recorded in the draft's own log, so once the exact model has decided,
                     * the draft commits exactly the positions it keeps, and a rejected
                     * draft costs the draft no replay either. */
                    dlag = base - dw.cached;
                    if (dlag < 0) {
                        fprintf(stderr, "draft model is ahead of the exact model\n");
                        frc = -1;
                    } else if (dlag + spec_n > dw.log_cap) {
                        /* Drafting stops for good once the run nears its end, so the
                         * draft trails by at most one token whenever it drafts. Should
                         * that change, absorb the backlog unrecorded, never past the
                         * log. */
                        if (prefill(&dw, &c, &cache, seq + dw.cached, dlag, prefill_chunk,
                                    dlg, sc, h, br, dks, NULL, NULL) != 0) frc = -1;
                        else dlag = 0;          /* prefill left dw.cached at base */
                    }
                    if (frc == 0) {
                        dw.log_row0 = 0;
                        if (forward(&dw, &c, &cache, seq + dw.cached, dlag + 1, dlg, sc, h,
                                    br, dks, NULL) != 0) frc = -1;
                        else { dw.cached = base + 1; d[nd++] = argmax_(dlg, c.vocab); }
                        while (frc == 0 && nd < spec_n) {
                            int prev = d[nd - 1];
                            dw.log_row0 = dlag + nd;
                            if (forward(&dw, &c, &cache, &prev, 1, dlg, sc, h, br,
                                        dks, NULL) != 0) frc = -1;
                            else { dw.cached += 1; d[nd++] = argmax_(dlg, c.vocab); }
                        }
                        dw.log_row0 = -1;
                        hyb_rounds  += 1;
                        hyb_drafted += nd;
                    }
                } else {
                    nd = spec_draft(seq, T, spec_n, d);
                }
            }
            if (frc == 0 && nd > 0) {
                /* One sweep verifies the pending token plus nd drafts. arg[i] is the
                 * model's own next token after batch position i; the accepted prefix is
                 * exactly what serial decode would have emitted, and arg[m] after it is
                 * clean because its context contains only accepted tokens. The sweep is
                 * tentative: every KDA layer leaves its carried state alone and records
                 * its recurrence inputs instead. */
                int arg[K3_SPEC_MAX + 1];
                for (int i = 0; i < nd; i++) seq[T + i] = d[i];
                if (w.loop_sweep_pcached) {      /* adaptive depth: where the sweep began */
                    memcpy(w.loop_sweep_pcached, w.loop_pcached,
                           ((size_t)w.loop_rmax + 1) * sizeof(int));
                    w.loop_sweep_base = base;
                }
                w.log_row0 = 0;
                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks, arg);
                w.log_row0 = -1;
                if (frc == 0) {
                    int m = 0;
                    while (m < nd && arg[m] == d[m]) m++;
                    for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                    emit[emitn++] = arg[m];
                    /* Commit exactly the positions behind the ids that will be emitted.
                     * Batch position i fed the id just before emit[i], so keeping `keep`
                     * ids keeps positions 0..keep-1 and leaves emit[keep-1] pending, as
                     * serial decode would. Normally keep is m+1; a --stop-id can cut it
                     * shorter, even after a full acceptance, and --save-state must then
                     * see the state serial decode would have saved there.
                     *
                     * The KDA layers get the kept positions applied from the log
                     * (k3_kda_advance, bit-exact); the MLA layers need only `cached`,
                     * since their rows past it are positional and are overwritten by the
                     * next sweep before anything reads them. No second forward runs, so
                     * no weight is read twice. */
                    keep = emit_keep(emit, emitn, nout, gen, T, Tmax, stop_id, n_stop);
                    spec_commit(&w, &c, ks, keep);
                    w.cached = base + keep;
                    step_logits = w.all_logits;
                    if (spec_sweeps < gen + 1) {
                        spec_trace[3 * spec_sweeps]     = nd;
                        spec_trace[3 * spec_sweeps + 1] = m;
                        spec_trace[3 * spec_sweeps + 2] = keep;
                    }
                    spec_sweeps++;
                    spec_drafted += nd;
                    spec_accepted += m;
                    spec_dropped += nd + 1 - keep;
                    if (m == nd) spec_full++; else spec_partial++;
                    if (keep < m + 1) spec_cut++;
                    /* The draft fed positions base-dlag .. base+nd-1 tentatively. It
                     * commits what the exact model kept, up to what it fed: after a full
                     * acceptance it ends one token short, which its next round absorbs. */
                    if (dw.trunk) {
                        const int dkeep = keep < nd ? keep : nd;
                        hyb_accepted += m;
                        spec_commit(&dw, &c, dks, dlag + dkeep);
                        draft_dropped += nd - dkeep;
                        dw.cached = base + dkeep;
                    }
                }
            } else if (frc == 0) {
                /* Nothing drafted: one serial step. The draft model does not follow it;
                 * with a draft trunk this only happens once the run is too close to its
                 * end to draft again, so feeding the draft here would be wasted. */
                frc = forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br, ks, NULL);
                if (frc == 0) { w.cached = base + 1; emit[emitn++] = argmax_(lg, c.vocab); }
            }
        } else {
            frc = forward(&w, &c, &cache, seq, T, lg, sc, h, br, ks, NULL);
            if (frc == 0) emit[emitn++] = argmax_(lg, c.vocab);
        }
        /* Abort the run rather than argmax a buffer the forward never wrote. */
        if (frc != 0 || emitn == 0) {
            fprintf(stderr, "forward pass failed at generation step %d; aborting.\n", g);
            return 1;
        }
        if (keep < 0) keep = emit_keep(emit, emitn, nout, gen, T, Tmax, stop_id, n_stop);
        steps++;
        const int nxt = emit[emitn - 1];
        /* The logits behind every kept id, in order: a verify sweep's position i for its
         * i-th id, otherwise the step's one vector. */
        if (all_logits_f) {
            for (int i = 0; i < keep; i++) {
                const float *v = step_logits ? step_logits + (size_t)i * c.vocab : lg;
                if (fwrite(v, sizeof(float), (size_t)c.vocab, all_logits_f)
                        != (size_t)c.vocab) {
                    fprintf(stderr, "cannot write %s\n", all_logits_path);
                    return 1;
                }
            }
        }
        /* Dump the FIRST step's logits as raw float32 bits.
         * Comparing generated tokens against a reference only compares argmax, which
         * hides near-ties: two engines can agree on every token while disagreeing
         * substantially on the logit vector behind it. tools/ref_forward.py produces the
         * same vector from the same shards in torch, and tools/cmp_logits.py compares
         * them elementwise. That is the only check here that can see a small systematic
         * error in the final norm, the lm_head, or the model-level AttnRes. */
        if (logits_path && g == 0) {
            FILE *lf = fopen(logits_path, "wb");
            if (lf) {
                fwrite(lg, sizeof(float), (size_t)c.vocab, lf);
                fclose(lf);
                printf("wrote %s (%d float32 logits)\n", logits_path, c.vocab);
            } else {
                fprintf(stderr, "cannot open %s for the logits dump\n", logits_path);
            }
        }
        const double dt = now_s() - ts;
        t_total += dt;
        const uint64_t req = cache.demand_requests;
        printf("%-6d %-10d %-12.2f %-10.1f %-10.2f %.3f\n", g, nxt, dt,
               req ? 100.0 * cache.demand_reuses / req : 0.0,
               (double)cache.bytes_read / 1e9, 1.0 / dt);
        fflush(stdout);
        /* Roll the per-step figures up before the next reset wipes them. */
        expert_s_total     += cache.load_seconds;
        expert_gb_total    += (double)cache.bytes_read / 1e9;
        expert_bytes_total += cache.bytes_read;
        expert_reqs_total  += cache.demand_requests;
        expert_reuses_total += cache.demand_reuses;
        expert_evict_total += cache.evictions;
        /* emit_keep already stopped at --gen, the context limit and the first stop id,
         * so a speculative sweep that verified past a stop id is truncated at the stop,
         * exactly like serial decode, and its committed state ends there too. */
        for (int i = 0; i < keep; i++) {
            seq[T++] = emit[i];
            outtok[nout++] = emit[i];
        }
        for (int s = 0; keep > 0 && s < n_stop; s++)
            if (emit[keep - 1] == stop_id[s]) {
                hit_stop = 1;
                stopped_at = emit[keep - 1];
                break;
            }
        if (hit_stop) {
            printf("stop id %d reached after %d of %d tokens\n", stopped_at, nout, gen);
            break;
        }
        if (T >= Tmax) break;
    }
    if (save_state) {
        if (!incremental) {
            fprintf(stderr, "--save-state needs --incremental; nothing written\n");
        } else {
            const double tsv = now_s();
            const int64_t kvpp   = (int64_t)kv_floats_per_pos(&c, w.kv_latent);
            const int64_t ropepp = (int64_t)c.qk_rope;
            if (w.loop_pcached) {
                /* Adaptive depth: rows of a pass's MLA cache past what it committed, and the
                 * pass counts of positions past `cached`, are what a rejected --spec sweep
                 * left behind. Nothing reads them; clear them so the file is the one serial
                 * decode writes, byte for byte. */
                const int rep0 = w.loop_b + 1;
                const size_t kvp = kv_floats_per_pos(&c, w.kv_latent);
                for (int v = rep0; v < rep0 + (w.loop_rmax - 1) * w.loop_span; v++) {
                    const int mi = w.mla_slot[v];
                    if (mi < 0) continue;
                    const int pc = w.loop_pcached[2 + (v - rep0) / w.loop_span];
                    if (pc >= w.cached) continue;
                    memset(w.kvc + ((size_t)mi * w.kv_cap + pc) * kvp, 0,
                           (size_t)(w.cached - pc) * kvp * sizeof(float));
                    memset(w.ropec + ((size_t)mi * w.kv_cap + pc) * c.qk_rope, 0,
                           (size_t)(w.cached - pc) * c.qk_rope * sizeof(float));
                }
                for (int i = w.cached; i < T; i++) w.loop_passes[i] = 0;
            }
            K3StateLoop sl = {-1, -1, 0, w.n_bound, -1.0};
            if (vmap) { sl.a = loop_a; sl.b = loop_b; sl.count = loop_count; sl.nv = NV;
                        sl.eps = w.loop_eps; }
            if (k3_state_save(save_state, &c, seq, T, ks, w.kvc, w.ropec,
                              w.n_bound, w.n_mla, w.kv_cap, w.cached,
                              (int64_t)kper, kvpp, ropepp, &sl, w.loop_pcached,
                              w.loop_passes) == 0) {
                const double bytes = (double)sizeof(K3StateHdr) + sizeof sl
                    + (double)T * sizeof(int) * (sl.count ? 2 : 1)
                    + (double)kper * (sl.count ? NV : w.n_bound) * sizeof(float)
                    + (double)w.cached * (kvpp + ropepp) * w.n_mla * sizeof(float);
                char sb[32]; human(bytes, sb, sizeof sb);
                printf("wrote %s (%s, %d positions) in %.2f s\n",
                       save_state, sb, w.cached, now_s() - tsv);
            }
        }
    }

    if (all_logits_f) {
        if (fclose(all_logits_f) != 0) { perror(all_logits_path); return 1; }
        printf("wrote %s (the logits behind all %d generated tokens)\n",
               all_logits_path, nout);
    }
    if (w.kda_log) {
        printf("\nspeculative decode: %ld verify sweeps, %ld drafted, %ld accepted "
               "(%.1f%%); %ld fully accepted, %ld partially\n",
               spec_sweeps, spec_drafted, spec_accepted,
               spec_drafted ? 100.0 * spec_accepted / spec_drafted : 0.0,
               spec_full, spec_partial);
        printf("  %ld swept positions dropped at commit (%ld sweeps cut short by a stop "
               "id); %ld exact forward sweeps for %d decode steps, so none replayed\n",
               spec_dropped, spec_cut, w.forwards, steps);
    }
    if (dw.trunk) {
        if (hyb_rounds > 0)
            printf("\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "
                   "mean accepted run %.2f; %ld draft sweeps, %ld drafted positions "
                   "dropped at commit\n",
                   hyb_rounds, hyb_drafted, hyb_accepted,
                   hyb_drafted ? 100.0 * hyb_accepted / hyb_drafted : 0.0,
                   (double)hyb_accepted / hyb_rounds, dw.forwards, draft_dropped);
        k3_trunk_close(&trunk_d);
        free(dw.lay); free(dks); free(dlg); free(dw.kvc); free(dw.ropec);
        free(dw.kda_log); free(dw.kda_slot);
    }
    printf("--------------------------------------------------------------------\n");
    if (nout > 0)
        printf("%d tokens in %.1f s, %.2f s/token average\n",
               nout, t_total, t_total / nout);
    else
        printf("prefill only: %d positions cached, 0 tokens generated\n", w.cached);

    /* Decoded text, when a tokenizer is loaded. Printed as a distinct block rather than
     * streamed per token: a partially-decoded multi-byte sequence is not valid UTF-8, so
     * streaming would emit mojibake at every token boundary that splits a codepoint. */
    if (have_tok && nout > 0) {
        generated_text = (char *)malloc((size_t)nout * 8 + 1);
        if (generated_text) {
            int m = tok_decode(&tok, outtok, nout, generated_text, nout * 8);
            generated_text[m] = 0;
            printf("\n--- generated text ---\n%s\n----------------------\n\n",
                   generated_text);
        }
    }
    const double peak_b = peak_rss_bytes();
    {
        char rb[32];
        human(peak_b, rb, sizeof rb);
        printf("PEAK RSS for the whole run: %s   <- quote this, not the plan\n", rb);
        printf("layers completed: %d/%d; routed expert drops: %ld\n\n",
               w.layers_completed, NV, k3_expert_drops);
    }
    k3_cache_report(&cache, "final step");

    FILE *f = fopen(outp, "w");
    if (f) {
        fprintf(f, "{\"prompt_ids\":[");
        for (int i = 0; i < np; i++) fprintf(f, "%s%d", i ? "," : "", prompt[i]);
        fprintf(f, "],\"original_prompt_tokens\":%d,\"reread_prompt\":%s,"
                   "\"generated_ids\":[", original_prompt_tokens,
                reread_prompt ? "true" : "false");
        for (int i = 0; i < nout; i++) fprintf(f, "%s%d", i ? "," : "", outtok[i]);
        fprintf(f, "],\"full_ids\":[");
        for (int i = 0; i < T; i++) fprintf(f, "%s%d", i ? "," : "", seq[i]);
        fputs("],\"loop_adaptive\":", f);
        write_loop_adaptive(f, &w, T);
        fprintf(f,
                ",\"loop\":%s,\"layers\":%d,\"layers_requested\":%d,\"layers_completed\":%d,"
                "\"expert_drops\":%ld,\"peak_rss_bytes\":%.0f,\"wall_seconds\":%.4f,"
                "\"seconds_per_token\":%.4f,\"expert_bytes_read\":%llu,"
                "\"expert_requests\":%llu,\"expert_resident_reuses\":%llu,"
                "\"expert_cache_slots\":%d,\"expert_profile_pins\":%d,"
                "\"trunk_bytes_read\":%llu,\"embedding_bytes_read\":%llu,"
                "\"lm_head_bytes_read\":%llu,\"ultra_low_memory\":%s,"
                "\"lm_head_streamed\":%s,\"kv_latent\":%s,\"prefill_chunk\":%d,"
                "\"prefill_passes\":%ld,\"buffer_rows\":%d,\"checkpoint_resumed_ids\":%d,"
                "\"checkpoints_written\":%d,\"trunk_ring_slots\":%d,"
                "\"trunk_slot_bytes\":%lld,\"trunk_budget_bytes\":%.0f,"
                "\"model_resident_bytes\":%zu,\"model_stream_buffer_bytes\":%zu,"
                "\"memory_plan_bytes\":%.0f,\"trunk_rows\":%s,"
                "\"trunk_row_buffer_bytes\":%llu,\"trunk_small_buffer_bytes\":%llu,"
                "\"trunk_matrix_calls\":%llu,\"trunk_rows_whole_tiles\":%s,"
                "\"trunk_kvb_mode\":\"%s\",\"trunk_kvb_buffer_bytes\":%llu,\"trunk_kvb_fills\":%llu,"
                "\"trunk_kvb_hits\":%llu,\"trunk_kvb_fill_bytes\":%llu,\"trunk_kvb_fallbacks\":%llu,"
                "\"trunk_kvb_metadata_bytes\":%llu,\"trunk_kvb_fill_seconds\":%.9f,"
                "\"stopped_at\":%d,"
                "\"decode_steps\":%d,\"forward_sweeps\":%ld,\"spec_n\":%d,"
                "\"spec_sweeps\":%ld,\"spec_drafted\":%ld,\"spec_accepted\":%ld,"
                "\"spec_full_accepts\":%ld,\"spec_partial_accepts\":%ld,"
                "\"spec_cut_by_stop\":%ld,\"spec_dropped_positions\":%ld,"
                "\"spec_log_bytes\":%.0f,\"logit_block_bytes\":%.0f,\"all_logits_bytes\":%.0f,"
                "\"draft_forward_sweeps\":%ld,"
                "\"draft_accepted\":%ld,\"draft_dropped_positions\":%ld,"
                "\"spec_trace\":[",
                loop_json, NL, NL, w.layers_completed, k3_expert_drops, peak_b, t_total,
                nout ? t_total / nout : 0.0, (unsigned long long)expert_bytes_total,
                (unsigned long long)expert_reqs_total,
                (unsigned long long)expert_reuses_total, cache.nslot, cache.profile_pins,
                (unsigned long long)(w.trunk ? w.trunk->bytes_read : 0),
                (unsigned long long)w.ms.embed_bytes_read,
                (unsigned long long)w.ms.lm_head_bytes_read,
                w.ultra ? "true" : "false", w.stream_lm_head ? "true" : "false",
                w.kv_latent ? "true" : "false",
                incremental ? prefill_chunk : 0, w.prefill_passes, w.max_rows, ckpt_resumed,
                ckpt.written,
                w.trunk ? w.trunk->nslot : 0,
                (long long)(w.trunk ? w.trunk->slot_bytes : 0),
                w.trunk ? trunk_gb * 1e9 : 0.0, w.mb.nbytes, w.ms.bufcap,
                memory_plan_bytes, trunk_rows ? "true" : "false",
                (unsigned long long)(w.trunk ? w.trunk->row_buffer_bytes : 0),
                (unsigned long long)(w.trunk ? w.trunk->small_buffer_bytes : 0),
                (unsigned long long)(w.trunk ? w.trunk->matrix_calls : 0),
                trunk_rows && w.trunk && w.trunk->rows_whole_tiles ? "true" : "false",
                kvb_mode == K3_KVB_PIN ? "pin" : kvb_mode == K3_KVB_ACTIVE ? "active" : "off",
                (unsigned long long)(w.trunk ? w.trunk->kvb_buffer_bytes : 0),
                (unsigned long long)(w.trunk ? w.trunk->kvb_fills : 0),
                (unsigned long long)(w.trunk ? w.trunk->kvb_hits : 0),
                (unsigned long long)(w.trunk ? w.trunk->kvb_fill_bytes : 0),
                (unsigned long long)(w.trunk ? w.trunk->kvb_fallbacks : 0),
                (unsigned long long)(w.trunk ? w.trunk->kvb_metadata_bytes : 0),
                w.trunk ? w.trunk->kvb_fill_seconds : 0.0,
                stopped_at,
                steps, w.forwards, spec_n, spec_sweeps, spec_drafted, spec_accepted,
                spec_full, spec_partial, spec_cut, spec_dropped,
                spec_log_bytes,
                (double)sizeof(float) * (w.logit_block ? (double)w.logit_rows * c.vocab : 0.0),
                w.all_logits ? all_logits_bytes : 0.0,
                dw.forwards, hyb_accepted, draft_dropped);
        for (long i = 0; i < spec_sweeps && i < gen + 1; i++)
            fprintf(f, "%s[%d,%d,%d]", i ? "," : "", spec_trace[3 * i],
                    spec_trace[3 * i + 1], spec_trace[3 * i + 2]);
        fputs("],\"generated_text\":", f);
        json_string(f, generated_text);
        fputs("}\n", f);
        fclose(f);
        printf("\nwrote %s\n", outp);
    }
    free(spec_trace);
    if (trace_dir) {
        char p[4096];
        snprintf(p, sizeof p, "%s/expert_hist.json", trace_dir);
        k3_cache_dump_hist(&cache, p);
        snprintf(p, sizeof p, "%s/expert_trace.bin", trace_dir);
        k3_cache_dump_trace(&cache, p);
    }

    /* Report the compute-versus-I/O split rather than leaving it to be inferred.
     *
     * It cannot be inferred safely: a flat curve across a RAM sweep looks like evidence
     * of a compute-bound engine, but it is equally consistent with the trunk being
     * streamed in full at every point of the sweep, so that the bytes moved barely
     * change. Those two have opposite tuning implications, and only a direct measurement
     * separates them. */
    {
        const double trunk_s = w.trunk ? w.trunk->load_seconds : 0.0;
        const double model_s = w.stream_lm_head ? w.ms.read_seconds : 0.0;
        /* Both terms MUST be whole-run totals over the same window. Mixing a cumulative
         * trunk time with a last-step expert time and dividing by the whole run
         * understates the expert share by roughly the token count. */
        const double io_s = trunk_s + expert_s_total + model_s;
        const double share = t_total > 0 ? 100.0 * io_s / t_total : 0.0;
        printf("I/O share of wall clock: %.1f%%  (trunk %.1f s + experts %.1f s + "
               "model tables %.1f s of %.1f s)\n",
               share, trunk_s, expert_s_total, model_s, t_total);
        printf("  both figures are WHOLE-RUN totals over %d steps\n", nout);
        /* Above 100% is not a bug in the arithmetic: with more than one trunk ring slot
         * the reader thread does device work while the main thread computes, so the two
         * terms genuinely overlap and their sum can exceed wall clock. Say so, rather
         * than printing an impossible percentage with no explanation. */
        if (share > 100.0)
            printf("  over 100%% because trunk reads overlap compute on the reader thread;\n"
                   "  %.1f s of device time was hidden behind arithmetic\n", io_s - t_total);
        /* Count actual reuse. requests-evictions incorrectly credits cold fills,
         * and raw hits incorrectly credits the first use of prefetched weights. */
        const unsigned long long retained = (unsigned long long)expert_reuses_total;
        printf("  experts, whole run: %.2f GB payload read | %llu of %llu requests reused"
               " (%.2f%%) | %llu evictions\n"
               "    (reuse excludes the first consumption of each load, including\n"
               "     prefetched loads; expert bytes exclude trunk and model reads)\n\n",
               expert_gb_total, retained,
               (unsigned long long)expert_reqs_total,
               expert_reqs_total ? 100.0 * (double)retained / (double)expert_reqs_total : 0.0,
               (unsigned long long)expert_evict_total);
    }
cleanup:
    free(w.kvc); free(w.ropec); free(w.mla_slot); free(w.vmap); free(w.vslot);
    free(w.loop_pcached); free(w.loop_sweep_pcached); free(w.loop_act); free(w.loop_idx); free(w.loop_nbt);
    free(w.loop_passes); free(w.loop_d); free(w.loop_hprev); free(w.loop_hc); free(w.loop_brc);
    free(w.kda_log); free(w.kda_slot); free(w.kda_work); free(w.all_logits);
    if (w.trunk) { k3_trunk_report(w.trunk, "final"); k3_trunk_close(w.trunk); }
    k3_cache_free(&cache);
    for (int L = 0; L < w.n_bound; L++) k3_bind_free(&w.lay[L]);
    free(w.lay);
    k3_model_stream_free(&w.ms);
    k3_bind_model_free(&w.mb);
    k3_st_close(&st);
    free(h); free(br); free(ks); free(sc); free(lg); free(w.logit_block); free(generated_text);
    free(prompt); free(seq); free(outtok);

    /* A dropped expert means some token was computed with part of its routed sum
     * missing. The run still produced token ids and they still look plausible, which is
     * exactly why this has to be an error rather than a note: silent numerical
     * corruption that exits 0 is indistinguishable from a good run. */
    if (k3_expert_drops) {
        fprintf(stderr,
                "\nRUN INVALID: %ld routed expert load(s) failed and were dropped from\n"
                "the MoE sum. The token ids above are CORRUPT. Re-run; if this repeats,\n"
                "the shard set or the storage is at fault.\n", k3_expert_drops);
        return 4;
    }
    return exit_status;
}
