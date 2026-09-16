/* k3_cache.c - see k3_cache.h. */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header;
                                * on Windows, supplies posix_memalign */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#ifndef _WIN32
#include <sys/mman.h>   /* MADV_HUGEPAGE; k3_portable_io.h no-ops it on Windows */
#endif

#include "k3_cache.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Resolve a slot to the three (packed, scale) pairs the kernels want. */
static void fill_q(const K3Cache *c, int slot, K3ExpertQ *q)
{
    /* pad is where the expert really begins: an O_DIRECT read starts at the enclosing
     * 4096 boundary, which is at or before the expert's own offset. */
    const unsigned char *b = c->arena + (size_t)slot * c->slot_bytes + c->pad[slot];
    const K3ExpertRef *r = &c->ref[slot];
    q->p1 = b + r->m[0].p_off; q->s1 = b + r->m[0].s_off;
    q->p2 = b + r->m[1].p_off; q->s2 = b + r->m[1].s_off;
    q->p3 = b + r->m[2].p_off; q->s3 = b + r->m[2].s_off;
}

/* Least recently used unpinned slot. Linear, deliberately: a few hundred comparisons
 * against a 17.55 MB read is not where the time goes. */
/* key_of[] has THREE states, not two:
 *     >= 0            holds that key
 *     K3_SLOT_EMPTY   holds nothing, free to take
 *     K3_SLOT_INFLIGHT reserved by a batch prefetch whose read has not finished
 *
 * The third state exists because of a real bug. The batch prefetch marks a slot empty
 * before reading into it, so that a failed read cannot leave the slot claiming an expert
 * it does not hold. But the empty test below is a FAST PATH that returns immediately,
 * ahead of the pinned check and the LRU scan -- so the next expert in the same batch was
 * handed the SAME slot, several parallel reads wrote into one buffer, and the MoE
 * multiplied garbage. It cost one wrong token (65 instead of 2494) on the real model and
 * nothing at all in the fixtures, because no fixture exercises the streaming cache. */
static int pick_victim(K3Cache *c, int protect_batch)
{
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] == K3_SLOT_INFLIGHT) continue;   /* being read into RIGHT NOW */
        if (c->key_of[i] == K3_SLOT_EMPTY) return i;      /* free, take it */
        if (c->pinned[i] || c->hot_key[c->key_of[i]]) continue;
        /* A batch must not evict a resident member of its own requested set. For a
         * prefill union larger than capacity, stop prefetching and let get() stream. */
        if (protect_batch && c->requested[c->key_of[i]]) continue;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

/* Bring (layer, expert) resident and return its slot, or -1. */
static int admit(K3Cache *c, int layer, int expert)
{
    const int32_t key = layer * c->n_experts + expert;
    int slot = c->slot_of[key];
    if (slot >= 0) {
        c->hits++;
        c->used_at[slot] = ++c->clock;
        return slot;
    }
    c->misses++;

    K3ExpertRef r;
    if (k3_expert_ref(c->st, layer, expert, &r) != 0) return -1;
    if (r.nbytes > c->slot_bytes) {
        fprintf(stderr, "k3_cache: L%d expert %d is %lld bytes, slot holds %lld\n",
                layer, expert, (long long)r.nbytes, (long long)c->slot_bytes);
        return -1;
    }

    slot = pick_victim(c, 0);
    if (slot < 0) {
        fprintf(stderr, "k3_cache: every slot is pinned, cannot admit L%d expert %d\n",
                layer, expert);
        return -1;
    }
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }

    const double t0 = now_s();
    int64_t pad = 0;
    const int64_t got = k3_expert_load_direct(c->st, &r,
                            c->arena + (size_t)slot * c->slot_bytes,
                            c->slot_bytes, &pad);
    c->load_seconds += now_s() - t0;
    if (got != r.nbytes) {
        fprintf(stderr, "k3_cache: short load of L%d expert %d (%lld of %lld)\n",
                layer, expert, (long long)got, (long long)r.nbytes);
        c->key_of[slot] = -1;
        return -1;
    }
    c->bytes_read += (uint64_t)got;

    c->ref[slot] = r;
    c->pad[slot] = (int32_t)pad;
    c->key_of[slot] = key;
    c->slot_of[key] = slot;
    c->used_at[slot] = ++c->clock;
    c->fresh[slot] = 1;
    return slot;
}

/* ------------------------------------------------------------ known-route pipelining --
 *
 * WHAT THE BARRIER COSTS. cache_getmany below reserves every slot, reads every slot, and
 * only then returns, so the MoE's first multiply waits for the SLOWEST of sixteen
 * 17.55 MB reads. The routes are already decided before the first read is issued, and
 * k3_moe consumes the top-k in a known order, so there is nothing to wait for: issue
 * the reads in CONSUMPTION order and publish each slot the instant it lands: get()
 * then blocks only on the expert it needs next, while the rest keep arriving behind it.
 *
 * Opt-in, because it trades a barrier for a mutex and a condvar on the hot path and
 * because with it on the reads are no longer issued in disk-offset order.
 * K3_EXPERT_PIPELINE=1 (or --expert-pipeline) turns it on; with it off not one byte of
 * behaviour changes.
 *
 * LOCKING RULES, and both are load-bearing:
 *   - Workers exist ONLY between a batch's launch and its drain(). Slot reservation and
 *     eviction happen on the calling thread with no worker running, because drain() is
 *     the first thing getmany does; everything else that touches the batch takes p->mu.
 *   - No disk read ever happens with p->mu held. A worker reads unlocked into a slot that
 *     is already reserved to it and takes the lock only to publish.
 */
#define K3_PIPE_MAX_THREADS 16
#define K3_PIPE_DEF_THREADS 4

/* State of one reserved entry: it is either still being read, published, or dead. */
#define K3_PIPE_PENDING 0
#define K3_PIPE_DONE    1
#define K3_PIPE_FAILED  2

typedef struct {
    int         slot;
    int         expert;
    int         state;
    K3ExpertRef r;
} K3PipeWork;

struct K3CachePipe {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    K3Cache    *c;
    K3PipeWork  w[K3_MAX_TOPK];
    int         nw;        /* entries in the current batch                          */
    int         cursor;    /* next entry a worker will claim, in consumption order  */
    int         ndone;     /* entries published or failed                           */
    int         layer;     /* every entry of a batch belongs to one layer           */
    int         active;    /* workers are running and still need joining            */
    int         maxth;     /* thread pool ceiling, from the environment             */
    int         nth;
    int         warned;    /* one pthread_create warning per cache, not per batch   */
    double      t0;        /* batch launch, for load_seconds                        */
    pthread_t   th[K3_PIPE_MAX_THREADS];
};

/* Publish (or bury) entry i. CALLED WITH p->mu HELD.
 *
 * Identical bookkeeping to phase 3 of the serial batch path, one entry at a time: a slot
 * is registered to its key only after its read succeeded, and a short read releases the
 * reservation so the slot can never be served as a hit.
 *
 * STATS: load_seconds in pipeline mode is the wall clock from the batch's launch to its
 * LAST completion, recorded here by whichever worker finishes last. Summing per-read
 * durations would count overlapped time several times over and report a bandwidth the
 * device never delivered; the batch's elapsed wall time keeps bytes_read/load_seconds a
 * real rate. Every other counter keeps exactly the meaning it has on the serial path. */
static void pipe_finish(struct K3CachePipe *p, int i, int64_t got, int64_t pad)
{
    K3Cache *c = p->c;
    K3PipeWork *e = &p->w[i];
    if (got != e->r.nbytes) {
        fprintf(stderr, "k3_cache: short prefetch of L%d expert %d (%lld of %lld); "
                        "leaving the slot empty so it cannot be served as a hit\n",
                p->layer, e->expert, (long long)got, (long long)e->r.nbytes);
        c->key_of[e->slot] = K3_SLOT_EMPTY;       /* release the reservation */
        e->state = K3_PIPE_FAILED;
    } else {
        const int32_t key = p->layer * c->n_experts + e->expert;
        c->ref[e->slot] = e->r;
        c->pad[e->slot] = (int32_t)pad;
        c->key_of[e->slot] = key;
        c->slot_of[key] = e->slot;
        c->used_at[e->slot] = ++c->clock;
        c->fresh[e->slot] = 1;
        c->bytes_read += (uint64_t)got;
        c->prefetch_reads++;
        e->state = K3_PIPE_DONE;
    }
    if (++p->ndone == p->nw) c->load_seconds += now_s() - p->t0;
    pthread_cond_broadcast(&p->cv);
}

/* Read one entry and publish it. The read itself is NOT under the lock. */
static void pipe_run_one(struct K3CachePipe *p, int i)
{
    K3Cache *c = p->c;
    int64_t pad = 0;
    const int64_t got = k3_expert_load_direct(
        c->st, &p->w[i].r, c->arena + (size_t)p->w[i].slot * c->slot_bytes,
        c->slot_bytes, &pad);
    pthread_mutex_lock(&p->mu);
    pipe_finish(p, i, got, pad);
    pthread_mutex_unlock(&p->mu);
}

/* Pull entries off the shared cursor IN ORDER, so the expert k3_moe will multiply first
 * is also the one the pool starts on. */
static void *pipe_worker(void *arg)
{
    struct K3CachePipe *p = (struct K3CachePipe *)arg;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        const int i = p->cursor < p->nw ? p->cursor++ : -1;
        pthread_mutex_unlock(&p->mu);
        if (i < 0) return NULL;
        pipe_run_one(p, i);
    }
}

/* Join the current batch's workers and forget it. Every entry point that inspects or
 * mutates cache state outside the mutex calls this first, so from its return until the
 * next launch the calling thread is the only one touching the cache. */
static void drain(K3Cache *c)
{
    struct K3CachePipe *p = c->pipe;
    if (!p || !p->active) return;
    for (int i = 0; i < p->nth; i++) pthread_join(p->th[i], NULL);
    p->nth = p->nw = p->cursor = p->ndone = 0;
    p->active = 0;
}

/* Reserve a slot for one expert and read it on THIS thread, taking the mutex only around
 * the bookkeeping. This is admit()'s miss path, split so that a worker publishing another
 * slot cannot race the reservation. Returns the slot, or -1. */
static int admit_sync_locked(K3Cache *c, int layer, int expert)
{
    struct K3CachePipe *p = c->pipe;
    const int32_t key = layer * c->n_experts + expert;

    K3ExpertRef r;
    if (k3_expert_ref(c->st, layer, expert, &r) != 0) return -1;
    if (r.nbytes > c->slot_bytes) {
        fprintf(stderr, "k3_cache: L%d expert %d is %lld bytes, slot holds %lld\n",
                layer, expert, (long long)r.nbytes, (long long)c->slot_bytes);
        return -1;
    }

    pthread_mutex_lock(&p->mu);
    const int slot = pick_victim(c, 0);
    if (slot < 0) {
        pthread_mutex_unlock(&p->mu);
        fprintf(stderr, "k3_cache: every slot is pinned, cannot admit L%d expert %d\n",
                layer, expert);
        return -1;
    }
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }
    /* INFLIGHT while the read runs, so a worker publishing elsewhere and a later victim
     * search both keep their hands off it. */
    c->key_of[slot] = K3_SLOT_INFLIGHT;
    c->used_at[slot] = ++c->clock;
    pthread_mutex_unlock(&p->mu);

    const double t0 = now_s();
    int64_t pad = 0;
    const int64_t got = k3_expert_load_direct(c->st, &r,
                            c->arena + (size_t)slot * c->slot_bytes,
                            c->slot_bytes, &pad);

    pthread_mutex_lock(&p->mu);
    c->load_seconds += now_s() - t0;
    if (got != r.nbytes) {
        fprintf(stderr, "k3_cache: short load of L%d expert %d (%lld of %lld)\n",
                layer, expert, (long long)got, (long long)r.nbytes);
        c->key_of[slot] = K3_SLOT_EMPTY;
        pthread_mutex_unlock(&p->mu);
        return -1;
    }
    c->bytes_read += (uint64_t)got;
    c->ref[slot] = r;
    c->pad[slot] = (int32_t)pad;
    c->key_of[slot] = key;
    c->slot_of[key] = slot;
    c->used_at[slot] = ++c->clock;
    c->fresh[slot] = 1;
    pthread_mutex_unlock(&p->mu);
    return slot;
}

/* admit() for pipeline mode. Three cases, in this order:
 *   resident              serve it, exactly as admit() does;
 *   pending in this batch  wait on the condvar rather than starting a second read of the
 *                         same expert into a second slot;
 *   anything else         the ordinary synchronous miss path.
 * A batch entry whose read FAILED falls through to that same miss path, so a short read
 * costs a retry rather than a dropped expert. */
static int admit_pipelined(K3Cache *c, int layer, int expert)
{
    struct K3CachePipe *p = c->pipe;
    const int32_t key = layer * c->n_experts + expert;

    pthread_mutex_lock(&p->mu);
    for (;;) {
        const int slot = c->slot_of[key];
        if (slot >= 0) {
            c->hits++;
            c->used_at[slot] = ++c->clock;
            pthread_mutex_unlock(&p->mu);
            return slot;
        }
        int pending = 0;
        if (p->active && p->layer == layer)
            for (int i = 0; i < p->nw; i++)
                if (p->w[i].expert == expert && p->w[i].state == K3_PIPE_PENDING) {
                    pending = 1; break;
                }
        if (!pending) break;
        pthread_cond_wait(&p->cv, &p->mu);
    }
    c->misses++;
    pthread_mutex_unlock(&p->mu);
    return admit_sync_locked(c, layer, expert);
}

/* Hand the reserved batch to the pool and RETURN, without waiting for a single read.
 * Returns the number of entries launched; get() is what waits, per expert. */
static int pipe_launch(K3Cache *c, int layer, int nw)
{
    struct K3CachePipe *p = c->pipe;
    p->layer = layer;
    p->nw = nw;
    p->cursor = p->ndone = p->nth = 0;
    p->t0 = now_s();
    p->active = 1;

    const int want = nw < p->maxth ? nw : p->maxth;
    int rc = 0;
    for (int i = 0; i < want; i++) {
        rc = pthread_create(&p->th[i], NULL, pipe_worker, p);
        if (rc != 0) break;
        p->nth++;
    }
    if (p->nth > 0) return nw;      /* one surviving worker still drains the whole list */

    /* Not a single thread: do the remaining entries here. That is the old barrier back
     * again, which is slower but not wrong. Warn once per cache, not once per batch. */
    p->active = 0;
    if (!p->warned) {
        p->warned = 1;
        fprintf(stderr, "k3_cache: cannot create pipeline threads (%s); falling back to "
                        "synchronous batch reads\n", strerror(rc));
    }
    for (int i = 0; i < nw; i++) pipe_run_one(p, i);
    int ok = 0;
    for (int i = 0; i < nw; i++) if (p->w[i].state == K3_PIPE_DONE) ok++;
    p->nw = p->cursor = p->ndone = 0;
    return ok;
}

/* Bring a whole top-k resident, with the reads issued CONCURRENTLY.
 *
 * The serial path admits one expert per call, so the drive sees a queue depth of one:
 * 17.55 MB, wait, repeat, 16 times per layer. NVMe needs depth to reach rated bandwidth,
 * so that pattern leaves most of the drive idle. This hands the whole set over at once.
 *
 * THREE PHASES, and the split is not cosmetic:
 *   1 SERIAL   resolve each miss and reserve it a slot. Slot allocation touches the LRU
 *              bookkeeping, which is shared mutable state and must not race.
 *   2 PARALLEL do the reads. Every read targets a distinct, already-assigned buffer and
 *              goes through pread, which takes its offset as an argument and so does not
 *              touch any shared file position. Nothing here is shared for writing.
 *   3 SERIAL   publish. A slot is registered to its key ONLY after its read succeeded.
 *
 * Phase 3 is where the danger was. Registering the key up front, then reading, would
 * leave a failed read with a slot that claims to hold an expert it does not -- and the
 * next request for that expert would count a HIT and multiply garbage. That exact bug
 * existed in the trunk ring and is why the order here is deliberate.
 */
static int cache_getmany(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    /* The previous batch owns its slots until its workers are joined. No-op when
     * pipelining is off. */
    drain(c);
    if (layer < 0 || layer >= c->n_layers || n < 0 || (n && !ids)) return -1;
    if (n == 0) return 0;

    typedef struct { int slot; int expert; K3ExpertRef r; int64_t got, pad; } Work;
    /* One entry per expert in a batch prefetch, so it is bounded by top-k. */
    Work w[K3_MAX_TOPK];
    int nw = 0;
    const int cap = (int)(sizeof w / sizeof *w);

    /* Mark once, so each victim candidate gets an O(1) membership check even for
     * a large prefill union. Only this serial reservation phase uses the marks. */
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < c->n_experts)
            c->requested[layer * c->n_experts + ids[i]] = 1;

    /* ---- phase 1: reserve, serially ---- */
    for (int i = 0; i < n && nw < cap; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->slot_of[key] >= 0) continue;             /* already resident */

        int dup = 0;                                    /* the same id twice in one top-k */
        for (int j = 0; j < nw; j++) if (w[j].expert == e) { dup = 1; break; }
        if (dup) continue;

        K3ExpertRef r;
        if (k3_expert_ref(c->st, layer, e, &r) != 0) continue;
        if (r.nbytes > c->slot_bytes) continue;

        const int slot = pick_victim(c, 1);
        if (slot < 0) break;
        if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }
        /* INFLIGHT, not EMPTY. Marking it empty made pick_victim's fast path hand the
         * same slot to the next expert in this very batch. */
        c->key_of[slot] = K3_SLOT_INFLIGHT;
        c->used_at[slot] = ++c->clock;

        w[nw].slot = slot; w[nw].expert = e; w[nw].r = r; w[nw].got = -1; w[nw].pad = 0;
        nw++;
    }
    /* Clear even when reservation stopped early or the entire batch was resident. */
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < c->n_experts)
            c->requested[layer * c->n_experts + ids[i]] = 0;
    if (nw == 0) return 0;

    /* Pipelined: keep CONSUMPTION order. Disk-offset order is the right answer when the
     * caller waits for the whole batch anyway, and the wrong one here -- it would put the
     * expert k3_moe multiplies first at an arbitrary place in the queue. */
    if (c->pipeline) {
        struct K3CachePipe *p = c->pipe;
        for (int i = 0; i < nw; i++) {
            p->w[i].slot = w[i].slot;
            p->w[i].expert = w[i].expert;
            p->w[i].r = w[i].r;
            p->w[i].state = K3_PIPE_PENDING;
        }
        return pipe_launch(c, layer, nw);
    }

    /* Issue in DISK-OFFSET order. Experts are not stored id-ordered inside a shard, so
     * sorting by where the bytes actually live turns a scattered set of seeks into a
     * mostly forward sweep. Insertion sort: nw is at most the top-k. */
    for (int i = 1; i < nw; i++) {
        Work t = w[i]; int j = i - 1;
        while (j >= 0 && (w[j].r.shard > t.r.shard ||
                         (w[j].r.shard == t.r.shard && w[j].r.off > t.r.off))) {
            w[j + 1] = w[j]; j--;
        }
        w[j + 1] = t;
    }

    /* ---- phase 2: read, concurrently ---- */
    const double t0 = now_s();
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < nw; i++) {
        int64_t pad = 0;
        const int64_t got = k3_expert_load_direct(
            c->st, &w[i].r, c->arena + (size_t)w[i].slot * c->slot_bytes,
            c->slot_bytes, &pad);
        w[i].got = got;
        w[i].pad = pad;
    }
    c->load_seconds += now_s() - t0;

    /* ---- phase 3: publish only what actually arrived ---- */
    int ok = 0;
    for (int i = 0; i < nw; i++) {
        if (w[i].got != w[i].r.nbytes) {
            fprintf(stderr, "k3_cache: short prefetch of L%d expert %d (%lld of %lld); "
                            "leaving the slot empty so it cannot be served as a hit\n",
                    layer, w[i].expert, (long long)w[i].got, (long long)w[i].r.nbytes);
            c->key_of[w[i].slot] = K3_SLOT_EMPTY;       /* release the reservation */
            continue;
        }
        const int32_t key = layer * c->n_experts + w[i].expert;
        c->ref[w[i].slot] = w[i].r;
        c->pad[w[i].slot] = (int32_t)w[i].pad;
        c->key_of[w[i].slot] = key;
        c->slot_of[key] = w[i].slot;
        c->used_at[w[i].slot] = ++c->clock;
        c->fresh[w[i].slot] = 1;
        c->bytes_read += (uint64_t)w[i].got;
        c->prefetch_reads++;
        ok++;
    }
    return ok;
}

/* Is this expert already resident, i.e. would get() serve it with no disk read? Used by
 * the draft model's cache-only routing to propose tokens without any expert I/O; if it
 * is resident, fill_q hands back the same bytes get() would. */
static int cache_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return 0;
    const int32_t key = layer * c->n_experts + expert;
    /* Pipelined, a worker may be publishing a slot right now: read slot_of under the same
     * mutex it is published with. This never waits -- "resident" means resident NOW. */
    if (c->pipeline) pthread_mutex_lock(&c->pipe->mu);
    const int slot = c->slot_of[key];
    if (slot >= 0 && out) fill_q(c, slot, out);
    if (c->pipeline) pthread_mutex_unlock(&c->pipe->mu);
    return slot >= 0;
}

static int cache_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;          /* src is the first member, by contract */
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts) {
        fprintf(stderr, "k3_cache: out of range L%d expert %d\n", layer, expert);
        return -1;
    }
    c->hist[layer * c->n_experts + expert]++;

    /* Record the request before serving it. The trace must reflect what the MODEL
     * asked for, independent of what the cache happened to hold, or replaying it at a
     * different capacity would be meaningless. */
    if (c->ntrace + 2 > c->captrace) {
        int64_t nc = c->captrace ? c->captrace * 2 : (1 << 16);
        int32_t *nt = (int32_t *)realloc(c->trace, (size_t)nc * sizeof(int32_t));
        if (nt) { c->trace = nt; c->captrace = nc; }
    }
    if (c->ntrace + 2 <= c->captrace) {
        c->trace[c->ntrace++] = layer;
        c->trace[c->ntrace++] = expert;
    }

    const int slot = c->pipeline ? admit_pipelined(c, layer, expert)
                                 : admit(c, layer, expert);
    if (slot < 0) return -1;
    /* fresh[] is written by the workers, so read-modify-write it under their mutex. */
    if (c->pipeline) pthread_mutex_lock(&c->pipe->mu);
    c->demand_requests++;
    if (!c->fresh[slot]) c->demand_reuses++;
    c->fresh[slot] = 0;
    fill_q(c, slot, out);
    if (c->pipeline) pthread_mutex_unlock(&c->pipe->mu);
    return 0;
}

int k3_cache_init(K3Cache *c, const K3St *st, const K3Cfg *cfg, int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->src.get = cache_get;
    c->src.resident = cache_resident;
    /* K3_NOPREFETCH=1 disables the batch path at runtime. An A/B between two BUILDS
     * compares two binaries; an A/B on one binary compares one decision, which is the
     * only way to attribute a timing difference to the prefetch rather than to the
     * compiler, the layout, or the weather. */
    c->src.getmany = getenv("K3_NOPREFETCH") ? NULL : cache_getmany;
    if (!c->src.getmany)
        fprintf(stderr, "k3_cache: batch prefetch DISABLED by K3_NOPREFETCH\n");
    /* Known-route pipelining is OPT-IN: K3_EXPERT_PIPELINE=1, or --expert-pipeline, which
     * sets it. Off, every path below is the one that shipped, byte for byte. */
    {
        const char *v = getenv("K3_EXPERT_PIPELINE");
        c->pipeline = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
    }
    c->src.ctx = c;
    c->st = st;
    c->n_layers = cfg->n_layers;
    c->n_experts = cfg->n_experts;
    c->topk = cfg->topk;

    /* Size a slot from the checkpoint rather than from arithmetic: find any expert and
     * ask how many bytes it actually occupies. */
    K3ExpertRef probe;
    int found = 0;
    for (int L = 0; L < cfg->n_layers && !found; L++) {
        if (k3_is_dense(cfg, L)) continue;
        if (k3_expert_ref(st, L, 0, &probe) == 0) found = 1;
    }
    if (!found) { fprintf(stderr, "k3_cache: no routed experts in this shard set\n"); return -1; }
    /* Room for an O_DIRECT read widened outward to 4096 boundaries at both ends. */
    /* Round the SLOT STRIDE up to the O_DIRECT alignment, not just the arena base.
     *
     * posix_memalign below aligns the arena, which aligns slot 0 and nothing else: slot
     * N starts at arena + N*slot_bytes, so every slot is aligned only if slot_bytes is
     * itself a multiple of K3_ST_ALIGN. On the real checkpoint an expert is 17,547,264
     * bytes, which is exactly 4284 * 4096, so this held BY COINCIDENCE and the engine
     * worked. With any other expert size -- another model, a repacked container, or the
     * few-KB experts in tests/fixtures/cache -- every O_DIRECT read into every slot
     * after the first returns 0 bytes and the cache silently serves nothing. The
     * fixture deliberately uses a non-conforming expert size so this is gated rather
     * than left to the real checkpoint's coincidence (tests/unit/test_cache.c). */
    c->slot_bytes = probe.nbytes + 2 * K3_ST_ALIGN;
    c->slot_bytes = (c->slot_bytes + K3_ST_ALIGN - 1) & ~(int64_t)(K3_ST_ALIGN - 1);

    c->nslot = (int)(budget_bytes / c->slot_bytes);
    if (c->nslot < cfg->topk + 1) {
        fprintf(stderr,
                "k3_cache: budget %.2f GB gives %d slots of %.2f MB, but top-%d needs at "
                "least %d. A cache smaller than one token's working set would evict an "
                "expert that is still being multiplied.\n",
                (double)budget_bytes / 1e9, c->nslot, (double)c->slot_bytes / 1e6,
                cfg->topk, cfg->topk + 1);
        return -1;
    }

    /* Page aligned so the arena can later be read into with O_DIRECT unchanged. */
    /* 2 MB aligned and hugepage-advised, for the same reason as the trunk arena: every
     * O_DIRECT expert read pins its destination pages, and a 17.55 MB slot on 4 KB pages
     * is 4,284 pins per read, 1,472 reads per token. See k3_trunk.c:k3_alloc_direct.
     * K3_NOHUGE=1 restores 4 KB so the two can be compared on one binary. */
    {
        const int huge = !getenv("K3_NOHUGE");
        const size_t al = huge ? (2u << 20) : 4096u;
        size_t want = (size_t)c->nslot * c->slot_bytes;
        if (huge) want = (want + al - 1) & ~(al - 1);
        if (posix_memalign((void **)&c->arena, al, want) != 0) {
            fprintf(stderr, "k3_cache: cannot allocate %.2f GB arena\n", (double)want / 1e9);
            return -1;
        }
#if defined(MADV_HUGEPAGE)
        if (huge) madvise(c->arena, want, MADV_HUGEPAGE);
#endif
    }
    if (0) {
        fprintf(stderr, "k3_cache: cannot allocate %.2f GB arena\n",
                (double)c->nslot * c->slot_bytes / 1e9);
        return -1;
    }

    const size_t nkey = (size_t)c->n_layers * c->n_experts;
    c->slot_of = (int32_t *)malloc(nkey * sizeof(int32_t));
    c->key_of  = (int32_t *)malloc((size_t)c->nslot * sizeof(int32_t));
    c->used_at = (uint64_t *)calloc((size_t)c->nslot, sizeof(uint64_t));
    c->pinned  = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->hot_key = (unsigned char *)calloc(nkey, 1);
    c->requested = (unsigned char *)calloc(nkey, 1);
    c->fresh   = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->ref     = (K3ExpertRef *)calloc((size_t)c->nslot, sizeof(K3ExpertRef));
    c->pad     = (int32_t *)calloc((size_t)c->nslot, sizeof(int32_t));
    c->hist    = (uint32_t *)calloc(nkey, sizeof(uint32_t));
    if (!c->slot_of || !c->key_of || !c->used_at || !c->pinned || !c->ref ||
        !c->pad || !c->hist || !c->hot_key || !c->fresh || !c->requested) {
        k3_cache_free(c); return -1;
    }
    for (size_t i = 0; i < nkey; i++) c->slot_of[i] = -1;
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = -1;

    if (c->pipeline) {
        struct K3CachePipe *p = (struct K3CachePipe *)calloc(1, sizeof *p);
        if (!p) { k3_cache_free(c); return -1; }
        p->c = c;
        /* A handful of threads is enough to keep an NVMe queue deep; more of them just
         * compete for the same device and for the publishing mutex. */
        p->maxth = K3_PIPE_DEF_THREADS;
        const char *nt = getenv("K3_EXPERT_PIPELINE_THREADS");
        if (nt && *nt) {
            const long v = strtol(nt, NULL, 10);
            if (v > 0)
                p->maxth = (int)(v > K3_PIPE_MAX_THREADS ? K3_PIPE_MAX_THREADS : v);
        }
        if (pthread_mutex_init(&p->mu, NULL) != 0) {
            free(p); k3_cache_free(c); return -1;
        }
        if (pthread_cond_init(&p->cv, NULL) != 0) {
            pthread_mutex_destroy(&p->mu); free(p); k3_cache_free(c); return -1;
        }
        c->pipe = p;
        fprintf(stderr, "k3_cache: known-route pipelining ON, %d reader thread(s)\n",
                p->maxth);
    }
    return 0;
}

void k3_cache_free(K3Cache *c)
{
    drain(c);
    if (c->pipe) {
        pthread_cond_destroy(&c->pipe->cv);
        pthread_mutex_destroy(&c->pipe->mu);
        free(c->pipe);
        c->pipe = NULL;
    }
    k3_aligned_free(c->arena); free(c->slot_of); free(c->key_of);
    free(c->used_at); free(c->pinned); free(c->ref); free(c->pad); free(c->hist);
    free(c->trace);
    free(c->hot_key); free(c->fresh);
    free(c->requested);
    memset(c, 0, sizeof *c);
}

int k3_cache_dump_trace(const K3Cache *c, const char *path)
{
    drain((K3Cache *)c);        /* const is about the TRACE, not about the worker pool */
    if (!c->trace || c->ntrace == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const size_t n = fwrite(c->trace, sizeof(int32_t), (size_t)c->ntrace, f);
    fclose(f);
    printf("wrote %s: %lld requests (%.1f KB)\n",
           path, (long long)(c->ntrace / 2), (double)c->ntrace * 4 / 1024.0);
    return n == (size_t)c->ntrace ? 0 : -1;
}

int k3_cache_pin(K3Cache *c, int layer, int expert, int pin)
{
    drain(c);
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return 0;
    const int32_t key = layer * c->n_experts + expert;
    const int slot = c->slot_of[key];
    if (slot < 0) return 0;
    if (pin && !c->pinned[slot] && !c->hot_key[key]) {
        int npin = c->profile_pins;
        for (int i = 0; i < c->nslot; i++)
            if (c->pinned[i] && c->key_of[i] >= 0 && !c->hot_key[c->key_of[i]]) npin++;
        if (npin >= c->nslot - c->topk - 1) return 0;
    }
    c->pinned[slot] = pin ? 1 : 0;
    return 1;
}

int k3_cache_prefetch(K3Cache *c, int layer, int expert)
{
    drain(c);
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return -1;
    return admit(c, layer, expert) >= 0 ? 0 : -1;
}

/* Parse unsigned decimal fields with explicit overflow/sign/trailing-data checks.
 * scanf's unsigned conversions accept negative inputs; profiles must not. */
static int profile_fields(const char *s, uint64_t *v, int n)
{
    for (int i = 0; i < n; i++) {
        while (*s == ' ' || *s == '\t') s++;
        if (*s < '0' || *s > '9') return -1;
        char *end;
        errno = 0;
        unsigned long long x = strtoull(s, &end, 10);
        if (errno == ERANGE) return -1;
        v[i] = (uint64_t)x;
        s = end;
        if (i + 1 < n && *s != ' ' && *s != '\t') return -1;
    }
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    return *s ? -1 : 0;
}

static int read_profile(K3Cache *c, const char *path, int count)
{
    if (count <= 0) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "k3_cache: cannot open profile %s\n", path); return -1; }
    unsigned char *selected = NULL, *seen = NULL;
    int ok = 0, rows = 0;
    char line[192];
    uint64_t v[4], prev_count = UINT64_MAX;
    int prev_key = -1;
    if (!fgets(line, sizeof line, f) || strncmp(line, "K3EXPERTS ", 10) ||
        !strchr(line, '\n') || profile_fields(line + 10, v, 4) ||
        v[0] != 1 || !v[1] || v[1] > 4096 || !v[2] || v[2] > 65536 ||
        !v[3] || v[3] > K3_MAX_TOPK || v[3] > v[2]) goto done;
    /* Bounds match the producer and keep key products within int32. Preflight has
     * no checkpoint config yet; installation checks the declared geometry again. */
    const int n_layers = (int)v[1], n_experts = (int)v[2];
    if (c && (n_layers != c->n_layers || n_experts != c->n_experts ||
              v[3] != (uint64_t)c->topk)) goto done;
    const size_t nkey = (size_t)n_layers * n_experts;
    if ((size_t)count > nkey) goto done;
    seen = (unsigned char *)calloc(nkey, 1);
    if (c) selected = (unsigned char *)calloc(nkey, 1);
    if (!seen || (c && !selected)) goto done;
    while (fgets(line, sizeof line, f)) {
        if (!strchr(line, '\n') || profile_fields(line, v, 3) ||
            v[0] >= (uint64_t)n_layers || v[1] >= (uint64_t)n_experts ||
            !v[2] || v[2] > prev_count) goto done;
        const int key = (int)v[0] * n_experts + (int)v[1];
        if (seen[key] || (v[2] == prev_count && key <= prev_key)) goto done;
        seen[key] = 1;
        prev_key = key; prev_count = v[2];
        if (c && rows < count) {
            K3ExpertRef ref;
            if (k3_expert_ref(c->st, (int)v[0], (int)v[1], &ref)) goto done;
            selected[key] = 1;
        }
        rows++;
    }
    if (ferror(f) || rows < count) goto done;
    if (c) {
        memcpy(c->hot_key, selected, nkey);
        c->profile_pins = count;
    }
    ok = 1;
done:
    free(selected); free(seen); fclose(f);
    if (!ok) fprintf(stderr, "k3_cache: invalid/incompatible profile %s "
                            "or fewer than %d ranked experts\n", path, count);
    return ok ? 0 : -1;
}

int k3_cache_check_profile(const char *path, int count)
{
    return read_profile(NULL, path, count);
}

int k3_cache_load_profile(K3Cache *c, const char *path, int count)
{
    drain(c);
    if (c->clock || count <= 0 || count > c->nslot - c->topk - 1) {
        fprintf(stderr, "k3_cache: profile needs an unused cache and 1..%d pins "
                        "(keep top-%d plus one slots evictable)\n",
                c->nslot - c->topk - 1, c->topk);
        return -1;
    }
    return read_profile(c, path, count);
}

void k3_cache_reset_stats(K3Cache *c)
{
    drain(c);                   /* a running batch still has counters to add */
    c->hits = c->misses = c->evictions = c->bytes_read = 0;
    c->demand_requests = c->demand_reuses = 0;
    c->load_seconds = 0.0;
    /* Keep per-slot fresh flags: a prefetch before reset is still a first consumption
     * afterward. Subtracting window-level prefetch totals cannot express this. */
    c->prefetch_reads = 0;
}

void k3_cache_report(const K3Cache *c, const char *label)
{
    drain((K3Cache *)c);        /* reporting mid-batch would print half a batch */
    const uint64_t n = c->demand_requests;
    int resident = 0, pinned = 0;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] >= 0) {
            resident++;
            if (c->pinned[i] || c->hot_key[c->key_of[i]]) pinned++;
        }
    }
    printf("cache [%s]\n", label ? label : "");
    printf("  slots        : %d of %.2f MB = %.2f GB arena (%d resident, %d pinned)\n",
           c->nslot, (double)c->slot_bytes / 1e6,
           (double)c->nslot * c->slot_bytes / 1e9, resident, pinned);
    printf("  requests     : %llu  resident reuses %llu (%.2f%%)  evictions %llu\n",
           (unsigned long long)n, (unsigned long long)c->demand_reuses,
           n ? 100.0 * c->demand_reuses / n : 0.0, (unsigned long long)c->evictions);
    printf("  weight payload: %.2f GB in %.2f s (%.0f MB/s while loading)\n",
           (double)c->bytes_read / 1e9, c->load_seconds,
           c->load_seconds > 0 ? (double)c->bytes_read / 1e6 / c->load_seconds : 0.0);
}

int k3_cache_dump_hist(const K3Cache *c, const char *path)
{
    drain((K3Cache *)c);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\"n_layers\":%d,\"n_experts\":%d,\"counts\":{",
            c->n_layers, c->n_experts);
    int first = 1;
    for (int L = 0; L < c->n_layers; L++) {
        for (int e = 0; e < c->n_experts; e++) {
            const uint32_t v = c->hist[L * c->n_experts + e];
            if (!v) continue;                       /* sparse: most are zero */
            fprintf(f, "%s\"%d,%d\":%u", first ? "" : ",", L, e, v);
            first = 0;
        }
    }
    fprintf(f, "}}\n");
    fclose(f);
    return 0;
}
