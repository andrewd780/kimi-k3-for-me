/* k3_trunk.c - see k3_trunk.h for why the trunk is streamed rather than quantised. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header */

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>   /* MADV_HUGEPAGE; k3_portable_io.h no-ops both on Windows */
#endif
#include <pthread.h>

#include "json.h"
#include "k3_st.h"
#include "k3_trunk.h"
#include "k3_zfile.h"

static int k3_alloc_direct(void **out, size_t bytes);   /* defined below */

typedef struct {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    K3Trunk *tr;
    int stop;
    int busy;
    int done;
    int layer;
    int slot;
    int result;
} K3TrunkIO;

static void *trunk_io_main(void *arg);

/* WHERE THE TIME IN A BIND ACTUALLY GOES.
 *
 * k3_trunk_report divides bytes_read by load_seconds, but load_seconds brackets ONLY the
 * read loop. For ordinary files it reports a DEVICE rate; compressed files include
 * decoding and report logical bytes per second. Everything else the bind does --
 * widening bf16 tensors to fp32, resolving names, kernel page bookkeeping -- is invisible
 * to it while still being paid on every layer of every token. That residual is large
 * enough to change conclusions drawn from the device rate alone.
 *
 * These three counters close the gap by measurement rather than estimate: wall clock
 * around the whole of k3_trunk_bind, of which the widen loop is tracked separately, so
 * bind_wall - load_seconds - widen_wall is the remaining unattributed time. */
double k3_trunk_bind_wall = 0.0;    /* total wall inside k3_trunk_bind   */
double k3_trunk_widen_wall = 0.0;   /* of which, inside k3_bind_layer_mem */
long   k3_trunk_binds = 0;

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int dt_of(const char *s)
{
    if (!strcmp(s, "BF16")) return K3_DT_BF16;
    if (!strcmp(s, "F32"))  return K3_DT_F32;
    if (!strcmp(s, "U8"))   return K3_DT_U8;
    if (!strcmp(s, "F16"))  return K3_DT_F16;
    if (!strcmp(s, "I8R"))  return K3_DT_I8R;
    return K3_DT_UNKNOWN;
}

static int json_size(jval *object, const char *key, int64_t *out)
{
    jval *value = json_get(object, key);
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        value->num < 0 || value->num >= 9223372036854775808.0) return -1;
    const int64_t n = (int64_t)value->num;
    if ((double)n != value->num) return -1;
    *out = n;
    return 0;
}

static char *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; fclose(f);
    if (n) *n = (size_t)sz;
    return b;
}

static void trunk_json_free(jval *v)
{
    if (!v) return;
    for (int i = 0; i < v->len; i++) {
        trunk_json_free(v->kids[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->kids); free(v->keys); free(v->str); free(v);
}

/* Resolver handed to k3_bind_layer_mem: linear over one layer's ~28 tensors, which is
 * nothing next to a 1.27 GB read. */
typedef struct { const K3TrunkLayer *L; } Finder;

static int find_in_layer(void *ctx, const char *name,
                         int64_t *off, int64_t *nbytes, int *dtype)
{
    const K3TrunkLayer *L = ((Finder *)ctx)->L;
    for (int i = 0; i < L->nt; i++)
        if (!strcmp(L->t[i].name, name)) {
            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;
            return 0;
        }
    return -1;
}

/* Row buffers never escape apply(). The reader writes only the spare buffer while
 * the existing matmul consumes the other one, then a condition-variable handoff
 * publishes completed bytes. A layer change and the 92 -> 0 seam therefore have
 * no outstanding reads or live matrix pointers to drain. */
typedef struct K3Rows K3Rows;
typedef struct {
    K3WeightStream stream;       /* first: dispatched by k3_mmw */
    K3Rows *owner;
    int64_t off, nbytes;
    int dtype;
} K3RowMatrix;

struct K3Rows {
    K3Trunk *tr;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int started, stop, busy, result, job_slot;
    int64_t job_off;
    size_t job_len, payload, cap, small_cap, small_used;
    unsigned char *buf[2], *small;
    K3RowMatrix matrix[64];
    int count, layer;
};

/* Offset, length and destination must all be aligned for O_DIRECT. Each slot has
 * two extra pages; row starts need not be page aligned. Padding stays inside the
 * packed layer's aligned run. Count actual requested bytes, including padding. */
static int rows_read(K3Rows *r, int slot, int64_t off, size_t len)
{
    if (off < 0 || len > r->payload) return -1;
    const size_t prefix = r->tr->direct ? (size_t)(off % K3_TRUNK_ALIGN) : 0;
    const int64_t base = off - (int64_t)prefix;
    size_t want = len + prefix;
    if (r->tr->direct) want = (want + K3_TRUNK_ALIGN - 1) & ~(size_t)(K3_TRUNK_ALIGN - 1);
    if (want > r->cap) return -1;
    size_t got = 0;
    const double start = now_s();
    while (got < want) {
        const int64_t n = r->tr->zfile
            ? k3_zread(r->tr->zfile, r->buf[slot] + got, (int64_t)(want - got), base + (int64_t)got)
            : pread(r->tr->fd, r->buf[slot] + got, want - got, (off_t)(base + (int64_t)got));
        if (n < 0 && errno == EINTR && !r->tr->zfile) continue;
        if (n <= 0) return -1;
        got += (size_t)n;
        if (r->tr->direct && got < want && got % K3_TRUNK_ALIGN) return -1;
    }
    r->tr->load_seconds += now_s() - start;
    r->tr->bytes_read += got;
    return 0;
}

static void *rows_worker(void *arg)
{
    K3Rows *r = (K3Rows *)arg;
    pthread_mutex_lock(&r->mu);
    for (;;) {
        while (!r->busy && !r->stop) pthread_cond_wait(&r->cv, &r->mu);
        if (r->stop) break;
        pthread_mutex_unlock(&r->mu);
        const int rc = rows_read(r, r->job_slot, r->job_off, r->job_len);
        pthread_mutex_lock(&r->mu);
        r->result = rc;
        r->busy = 0;
        pthread_cond_broadcast(&r->cv);
    }
    pthread_mutex_unlock(&r->mu);
    return NULL;
}

static void rows_submit(K3Rows *r, int slot, int64_t off, size_t len)
{
    pthread_mutex_lock(&r->mu);
    r->job_slot = slot; r->job_off = off; r->job_len = len;
    r->busy = 1;
    pthread_cond_signal(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

/* Called only by the main thread, which is what makes the wait it adds to
 * row_wait_seconds time that no compute overlapped. */
static int rows_wait(K3Rows *r)
{
    const double start = now_s();
    pthread_mutex_lock(&r->mu);
    while (r->busy) pthread_cond_wait(&r->cv, &r->mu);
    const int result = r->result;
    pthread_mutex_unlock(&r->mu);
    r->tr->row_wait_seconds += now_s() - start;
    return result;
}

/* THE ROWS A PASS APPLIES: rows r0 .. r0 + nr - 1 of each block of blk rows, out / blk
 * blocks, as in k3_mmw_rows. A whole matrix is the one block {blk = nr = out, r0 = 0}.
 * Selected row k is matrix row (k / nr) * blk + r0 + k % nr; a block's selected rows are
 * one contiguous run on disk, and consecutive runs are not. */
typedef struct { int blk, r0, nr; } K3RowSel;

static int rows_sel_row(const K3RowSel *s, int k)
{
    return (k / s->nr) * s->blk + s->r0 + k % s->nr;
}

/* Rows in the tile that starts at selected row k: at most `per`, and never past the end
 * of k's run, since the next run is not adjacent on disk. For a whole matrix this is the
 * tiling of old, one run of out rows cut every `per` rows. */
static size_t rows_sel_tile(const K3RowSel *s, int k, size_t per)
{
    const size_t left = (size_t)(s->nr - k % s->nr);
    return left < per ? left : per;
}

/* Apply the selected rows that tile [first, first + count) holds: the part of each run
 * inside it, written to its own rows of Y. A tile cut from the selection's own runs holds
 * one run's part, a single call; a whole-matrix tile holds a part of every run it
 * crosses. The whole matrix itself is one run, so its tile is one call, as always. */
static void rows_apply_tile(const K3RowSel *sel, const unsigned char *weight, size_t row,
                            int wdt, int first, int count, float *Y, int ldy,
                            const float *X, int ldx, int in, int T)
{
    const int end = first + count;
    for (int b = first / sel->blk; b * sel->blk < end; b++) {
        int lo = b * sel->blk + sel->r0, hi = lo + sel->nr;
        if (lo < first) lo = first;
        if (hi > end) hi = end;
        if (lo < hi)
            k3_mmw_batch_ld(Y + lo, ldy, X, ldx, weight + (size_t)(lo - first) * row, wdt, in,
                            hi - lo, T);
    }
}

/* One pass over a streamed matrix for T positions: position t reads X + t*ldx and writes
 * Y + t*ldy. Each row tile is read from disk ONCE and applied to every position before the
 * next tile replaces it, so a T-position batch costs one matrix of I/O rather than T. The
 * product for a tile is k3_mmw_batch_ld over its rows, whose outputs are bit-identical to
 * the per-position kernel's (see k3_ops.c), and tiling splits output ROWS only, so every
 * output is the same float whether the matrix is tiled, resident, or batched. T == 1 takes
 * the single-position kernel through k3_mmw_batch_ld, so decode runs exactly as before.
 *
 * A pass applies only the rows `sel` selects (every row for apply and apply_batch; the
 * rows k3_mmw_rows asks for through apply_rows) and writes each to its own row of Y, as
 * the whole product would. HOW IT READS THEM depends on the file. From a plain trunk.bin
 * the tiles are cut from the selected runs, so only their bytes are read: half of kv_b
 * per --kv-latent pass at K3 geometry, as 96 reads of one head's 128 KiB of rows instead
 * of three 8 MiB tiles. That is fewer bytes in more requests: in an informal O_DIRECT
 * check on the development VM's disk (two runs, the best of seven passes each; not a
 * committed measurement) the 96 reads took 10.1 and 11.4 ms where the three took 16.9
 * and 18.5, and a device whose per-request latency is high relative to its bandwidth
 * gains less. A compressed trunk
 * decodes the whole block behind a read (1 MiB by default, four heads of kv_b), so
 * reading run by run would decode each block once per run it holds; there
 * (rows_whole_tiles) the tiles are whole-matrix tiles, read as a full pass reads them,
 * and only their selected rows are applied.
 *
 * On a failure every row of Y is zeroed and read_error is set. matrix_calls counts passes,
 * which is what the no-reread gate in test_offline_cli.py compares. */
static void rows_run(const K3RowMatrix *m, float *Y, int ldy, const float *X, int ldx,
                     int in, int out, int T, K3RowSel sel)
{
    K3Rows *r = m->owner;
    const size_t esz = m->dtype == K3_DT_BF16 ? 2u : 4u;
    const int wdt = m->dtype == K3_DT_BF16 ? K3_WBF16 : K3_WF32;
    if (T <= 0) return;
    if (in <= 0 || out <= 0 || (uint64_t)in * (uint64_t)out > INT64_MAX / esz ||
        (int64_t)((uint64_t)in * (uint64_t)out * esz) != m->nbytes ||
        (T > 1 && (ldy < out || ldx < in)) ||
        sel.blk <= 0 || out % sel.blk != 0 || sel.nr <= 0 || sel.r0 < 0 ||
        sel.nr > sel.blk - sel.r0) goto failed;
    const size_t row = (size_t)in * esz;
    if (r->tr->read_error || row > r->payload) goto failed;
    const size_t per = r->payload / row;
    const K3RowSel whole = {out, 0, out};
    const K3RowSel cut = r->tr->rows_whole_tiles ? whole : sel;   /* rows the reads cover */
    const int total = out / cut.blk * cut.nr;
    r->tr->matrix_calls++;
    int k = 0, slot = 0;
    size_t count = rows_sel_tile(&cut, 0, per);
    rows_submit(r, slot, m->off + (int64_t)rows_sel_row(&cut, 0) * (int64_t)row, count * row);
    while (k < total) {
        if (rows_wait(r)) goto failed;
        const int next = k + (int)count;
        const size_t nnext = next < total ? rows_sel_tile(&cut, next, per) : 0;
        if (nnext)
            rows_submit(r, 1 - slot,
                        m->off + (int64_t)rows_sel_row(&cut, next) * (int64_t)row,
                        nnext * row);
        const int first = rows_sel_row(&cut, k);
        const int64_t off = m->off + (int64_t)first * (int64_t)row;
        const size_t prefix = r->tr->direct ? (size_t)(off % K3_TRUNK_ALIGN) : 0;
        rows_apply_tile(&sel, r->buf[slot] + prefix, row, wdt, first, (int)count, Y, ldy, X,
                        ldx, in, T);
        k = next; count = nnext; slot = 1 - slot;
    }
    return;
failed:
    r->tr->read_error = 1;
    if (out > 0)
        for (int t = 0; t < T; t++) memset(Y + (size_t)t * ldy, 0, (size_t)out * sizeof(float));
}

static void rows_apply(const K3WeightStream *stream, float *y, const float *x, int in, int out)
{
    const K3RowSel all = {out, 0, out};
    rows_run((const K3RowMatrix *)stream, y, out, x, in, in, out, 1, all);
}

static void rows_apply_batch(const K3WeightStream *stream, float *Y, int ldy,
                             const float *X, int ldx, int in, int out, int T)
{
    const K3RowSel all = {out, 0, out};
    rows_run((const K3RowMatrix *)stream, Y, ldy, X, ldx, in, out, T, all);
}

/* The --kv-latent passes: kv_b's key rows to score a cached position, its value rows to
 * weight it, each applied without the other half and, from a plain trunk.bin, read
 * without it (see rows_run). Selecting no rows reads and writes nothing, as in the
 * resident kernels. */
static void rows_apply_rows(const K3WeightStream *stream, float *y, const float *x, int in,
                            int out, int blk, int r0, int nr)
{
    if (nr == 0) return;
    const K3RowSel sel = {blk, r0, nr};
    rows_run((const K3RowMatrix *)stream, y, out, x, in, in, out, 1, sel);
}

static int rows_acquire(void *ctx, int64_t off, int64_t nb, int dt,
                        int64_t take, int narrow, const void **dest)
{
    K3Rows *r = (K3Rows *)ctx;
    off += r->tr->lay[r->layer].file_off;
    if (narrow) {
        if (r->count == 64) return -1;
        K3RowMatrix *m = &r->matrix[r->count++];
        m->stream.apply = rows_apply; m->stream.apply_batch = rows_apply_batch;
        m->stream.apply_rows = rows_apply_rows;
        m->owner = r;
        m->off = off; m->nbytes = nb; m->dtype = dt;
        *dest = m;
        return 0;
    }
    const size_t start = (r->small_used + 7u) & ~(size_t)7u;
    if ((uint64_t)take > SIZE_MAX / 4 || start > r->small_cap ||
        (size_t)take * 4 > r->small_cap - start) return -1;
    const size_t esz = dt == K3_DT_BF16 ? 2u : 4u;
    size_t done = 0;
    float *dst = (float *)(r->small + start);
    const double sync_start = now_s();   /* main thread, inside bind: never overlapped */
    while (done < (size_t)take) {
        size_t count = (size_t)take - done;
        if (count > r->payload / esz) count = r->payload / esz;
        const int64_t at = off + (int64_t)(done * esz);
        if (rows_read(r, 0, at, count * esz)) return -1;
        const size_t prefix = r->tr->direct ? (size_t)(at % K3_TRUNK_ALIGN) : 0;
        if (dt == K3_DT_F32) memcpy(dst + done, r->buf[0] + prefix, count * 4);
        else {
            const uint16_t *sp = (const uint16_t *)(r->buf[0] + prefix);
            for (size_t j = 0; j < count; j++) dst[done + j] = k3_bf16f(sp[j]);
        }
        done += count;
    }
    r->tr->row_sync_seconds += now_s() - sync_start;
    *dest = dst; r->small_used = start + (size_t)take * 4;
    return 0;
}

static void rows_close(K3Rows *r)
{
    if (!r) return;
    if (r->started) {
        pthread_mutex_lock(&r->mu); r->stop = 1;
        pthread_cond_signal(&r->cv); pthread_mutex_unlock(&r->mu);
        pthread_join(r->thread, NULL);
        pthread_cond_destroy(&r->cv); pthread_mutex_destroy(&r->mu);
    }
    k3_aligned_free(r->buf[0]); k3_aligned_free(r->buf[1]);
    free(r->small); free(r);
}

static int rows_open(K3Trunk *tr, const K3Cfg *c, int64_t budget)
{
    K3Rows *r = (K3Rows *)calloc(1, sizeof *r);
    if (!r) return -1;
    r->tr = tr;
    /* Only the layers the config describes are ever bound; a trunk may hold more. */
    for (int L = 0; L < tr->n_layers && L < c->n_layers; L++) {
        Finder f = { &tr->lay[L] }; K3MemSrc src = { find_in_layer, &f };
        K3LayerBind tmp; size_t small = 0;
        if (k3_bind_layer_stream(c, L, &tmp, &src, NULL, NULL, &small)) {
            fprintf(stderr, "k3_trunk: layer %d's tensors do not match the config's plan "
                            "(names, dtypes or shapes)\n", L);
            goto bad;
        }
        if (small > r->small_cap) r->small_cap = small;
    }
    const uint64_t fixed = r->small_cap + sizeof *r;
    if (budget <= 0 || (uint64_t)budget < fixed + 6u * K3_TRUNK_ALIGN) {
        fprintf(stderr, "k3_trunk: a %lld-byte row budget is below the %llu bytes the "
                        "largest layer's vectors and two minimal row buffers need\n",
                (long long)budget, (unsigned long long)(fixed + 6u * K3_TRUNK_ALIGN));
        goto bad;
    }
    r->cap = (size_t)(((uint64_t)budget - fixed) / 2);
    if (r->cap > (8u << 20) + 2u * K3_TRUNK_ALIGN) r->cap = (8u << 20) + 2u * K3_TRUNK_ALIGN;
    r->cap &= ~(size_t)(K3_TRUNK_ALIGN - 1);
    r->payload = r->cap - 2u * K3_TRUNK_ALIGN;
    const int64_t cols[] = {c->hidden, c->dense_inter, c->q_lora, c->kv_lora,
        (int64_t)c->n_heads * c->v_head, (int64_t)c->kda_heads * c->kda_head_dim,
        (int64_t)c->moe_inter * c->n_shared, c->latent, c->kda_head_dim};
    for (size_t i = 0; i < sizeof cols / sizeof *cols; i++) {
        /* Conservatively allow F32 matrices too; reject an undersized row buffer
         * during opening, before any forward pass can start. */
        if (cols[i] < 0 || (uint64_t)cols[i] > r->payload / 4) {
            fprintf(stderr, "k3_trunk: a %lld-wide matrix row does not fit a %zu-byte row "
                            "buffer; raise the trunk budget\n", (long long)cols[i], r->payload);
            goto bad;
        }
    }
    r->small = (unsigned char *)malloc(r->small_cap ? r->small_cap : 1);
    if (!r->small || posix_memalign((void **)&r->buf[0], K3_TRUNK_ALIGN, r->cap) ||
        posix_memalign((void **)&r->buf[1], K3_TRUNK_ALIGN, r->cap)) {
        fprintf(stderr, "k3_trunk: cannot allocate the row pipeline's buffers\n");
        goto bad;
    }
    if (pthread_mutex_init(&r->mu, NULL)) goto no_thread;
    if (pthread_cond_init(&r->cv, NULL)) { pthread_mutex_destroy(&r->mu); goto no_thread; }
    if (pthread_create(&r->thread, NULL, rows_worker, r)) {
        pthread_cond_destroy(&r->cv); pthread_mutex_destroy(&r->mu); goto no_thread;
    }
    r->started = 1; tr->row_state = r;
    tr->nslot = 2; tr->slot_bytes = (int64_t)r->cap;
    tr->row_buffer_bytes = 2 * r->cap;
    tr->small_buffer_bytes = r->small_cap;
    printf("trunk: row pipeline, two %zu-byte buffers + %zu-byte current-layer vectors\n",
           r->cap, r->small_cap);
    return 0;
no_thread:
    fprintf(stderr, "k3_trunk: cannot start the row pipeline's reader thread\n");
bad:
    rows_close(r);
    return -1;
}

static int trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes, int rows)
{
    memset(tr, 0, sizeof *tr);
    /* memset leaves fd == 0, which is stdin. Every failure path below returns without
     * opening the file, and a caller that then calls k3_trunk_close would close the
     * process's stdin. -1 is the only safe "no file" value. */
    tr->fd = -1;

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.json", dir);
    size_t jn = 0;
    char *txt = slurp(p, &jn);
    if (!txt) { fprintf(stderr, "k3_trunk: cannot read %s\n", p); return -1; }
    /* Every K3TrunkTensor.name points at an object key of the parsed tree: the parser
     * allocates each string on its own (see j_dup in json.h) and the tree owns them. So the
     * tree, and the arena pointer the parser may also return, belong to the K3Trunk and are
     * freed together in k3_trunk_close, not before and not at process exit. */
    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    tr->json_arena = arena;
    tr->json_root = root;
    if (!root) { fprintf(stderr, "k3_trunk: %s is not valid JSON\n", p); free(txt); return -1; }

    /* Every refusal below says which field of trunk.json failed. They guard the offsets
     * the readers seek to and the pointers the binder hands the kernels, so a malformed
     * manifest stops here, with its reason, instead of reading outside a layer's run. */
    jval *jl = json_get(root, "layers");
    if (!jl || jl->t != J_ARR) { fprintf(stderr, "k3_trunk: no layers array\n"); goto bad; }
    tr->n_layers = jl->len;
    /* More layers than the config uses are accepted, as they always were: only layers
     * below the count the caller binds are ever read, and k3_run refuses a trunk with
     * FEWER layers than it needs. rows_open stops its metadata walk at c->n_layers. */
    if (tr->n_layers <= 0) { fprintf(stderr, "k3_trunk: %s lists no layers\n", p); goto bad; }
    tr->lay = (K3TrunkLayer *)calloc((size_t)tr->n_layers, sizeof(K3TrunkLayer));
    if (!tr->lay) goto bad;

    for (int i = 0; i < jl->len; i++) {
        jval *e = jl->kids[i];
        jval *v;
        K3TrunkLayer *L = &tr->lay[i];
        if (json_size(e, "file_off", &L->file_off) || json_size(e, "nbytes", &L->nbytes)) {
            fprintf(stderr, "k3_trunk: layer %d: file_off and nbytes must be non-negative "
                            "integers\n", i);
            goto bad;
        }
        if (L->nbytes <= 0 || L->file_off > INT64_MAX - L->nbytes) {
            fprintf(stderr, "k3_trunk: layer %d: run of %lld bytes at %lld is empty or "
                            "overflows\n", i, (long long)L->nbytes, (long long)L->file_off);
            goto bad;
        }
        jval *ts = json_get(e, "tensors");
        if (!ts || ts->t != J_OBJ) { fprintf(stderr, "k3_trunk: layer %d has no tensors\n", i); goto bad; }
        L->nt = ts->len;
        L->t = (K3TrunkTensor *)calloc((size_t)L->nt, sizeof(K3TrunkTensor));
        if (!L->t) goto bad;
        for (int k = 0; k < ts->len; k++) {
            K3TrunkTensor *t = &L->t[k];
            /* a key of the parsed tree, which k3_trunk_close frees (see above) */
            t->name = ts->keys[k];
            jval *o = ts->kids[k];
            if (json_size(o, "off", &t->off) || json_size(o, "nbytes", &t->nbytes)) {
                fprintf(stderr, "k3_trunk: layer %d, %s: off and nbytes must be non-negative "
                                "integers\n", i, t->name);
                goto bad;
            }
            if ((v = json_get(o, "dtype"))  && v->t == J_STR) t->dtype  = dt_of(v->str);
            if (t->nbytes <= 0 || t->off > L->nbytes || t->nbytes > L->nbytes - t->off) {
                fprintf(stderr, "k3_trunk: layer %d, %s: %lld bytes at %lld do not fit the "
                                "layer's %lld-byte run\n", i, t->name, (long long)t->nbytes,
                        (long long)t->off, (long long)L->nbytes);
                goto bad;
            }
            if ((t->dtype == K3_DT_F32 && t->off % 4) || (t->dtype == K3_DT_BF16 && t->off % 2)) {
                fprintf(stderr, "k3_trunk: layer %d, %s: offset %lld is not a multiple of "
                                "its %d-byte element\n", i, t->name, (long long)t->off,
                        t->dtype == K3_DT_F32 ? 4 : 2);
                goto bad;
            }
        }
    }
    free(txt);                      /* arena holds the strings; txt itself is done */

    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    /* O_DIRECT, because the trunk is the one thing the page cache CANNOT help with.
     * Each streamed layer is read once per token and never reused before eviction, so
     * buffering it only copies every byte twice and evicts whatever else the cgroup was
     * holding. Measured under a 32 GB cap: buffered reads collapsed to 1,878 MB/s
     * against 6,553 MB/s unconstrained.
     *
     * It requires offset, length and buffer all aligned. pack_trunk.py pads every run
     * to 4096 and the slots come from posix_memalign. If the filesystem refuses
     * O_DIRECT, fall back rather than fail: correctness does not depend on it. */
    tr->direct = 1;
    tr->fd = open(p, O_RDONLY | O_DIRECT);
    if (tr->fd >= 0 && k3_set_direct(tr->fd) != 0)
        tr->direct = 0;   /* Darwin refused F_NOCACHE: reads stay buffered but correct */
    if (tr->fd < 0) {
        tr->direct = 0;
        tr->fd = open(p, O_RDONLY);
    }
    /* Capture immediately. errno is process-global, so any call inserted between the
     * open above and this test would silently disable compressed-trunk support: the
     * user would see "cannot open trunk.bin" with trunk.bin.k3z sitting next to it. */
    const int open_err = tr->fd < 0 ? errno : 0;
    if (tr->fd < 0 && open_err == ENOENT) {
        snprintf(p, sizeof p, "%s/trunk.bin.k3z", dir);
        tr->fd = open(p, O_RDONLY);
        if (tr->fd >= 0 && k3_zopen(tr->fd, &tr->zfile)) {
            /* The archive opened but its index is unusable. k3_zopen already said why;
             * close here because a caller that gets -1 need not call k3_trunk_close. */
            close(tr->fd);
            tr->fd = -1;
            return -1;
        }
    }
    if (tr->fd < 0) { fprintf(stderr, "k3_trunk: cannot open %s\n", p); return -1; }
    {
        jval *a = json_get(root, "align");
        const int64_t want = (a && a->t == J_NUM) ? (int64_t)a->num : 0;
        if (tr->direct && want != K3_TRUNK_ALIGN) {
            /* A trunk packed before the alignment change cannot be read with O_DIRECT:
             * its run offsets are arbitrary. Say so rather than fail every read. */
            fprintf(stderr, "k3_trunk: trunk.json reports align %lld, expected %d; "
                            "falling back to buffered reads (repack to enable O_DIRECT)\n",
                    (long long)want, K3_TRUNK_ALIGN);
            close(tr->fd);
            tr->direct = 0;
            tr->fd = open(p, O_RDONLY);
            if (tr->fd < 0) return -1;
        }
    }

    tr->rows_whole_tiles = tr->zfile != NULL;
    if (rows) return rows_open(tr, c, budget_bytes);
    const size_t widen = k3_bind_widen_bytes(c);
    int64_t total = 0;
    for (int i = 0; i < tr->n_layers; i++) total += tr->lay[i].nbytes;

    /* Pin a PREFIX of layers, each in an exact-size allocation, then keep a small ring
     * of uniform slots for everything else. Uniform slots everywhere would size every
     * slot for layer 0 (2.34 GB, the dense MLP) and waste roughly half the budget. */
    tr->slot_of = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    if (!tr->slot_of) return -1;
    for (int i = 0; i < tr->n_layers; i++) tr->slot_of[i] = -1;

    /* Two ring slots: the layer being computed on, plus one asynchronous read in flight.
     *
     * This is a REQUEST, not a guarantee. The second slot costs a full slot's worth of
     * memory, which at the floor is 2.37 GB, and the budget the caller asked for has to
     * come first: measured on the released checkpoint, taking the second slot
     * unconditionally moved the laptop preset from 8.78 GB to 11.12 GB peak RSS, a 27%
     * overshoot of a 3.0 GB trunk budget, and left the printed memory plan understating
     * the real figure. So it is granted only when it fits, and reported when it does not.
     * A single slot is exactly what this file did before the asynchronous reader existed,
     * so falling back is always safe; it costs speed, not correctness. */
    const int RING_WANT = 2;
    int RING = RING_WANT;

    /* Size the ring from the layers that will actually STREAM through it.
     *
     * Pinning is a PREFIX: layers 0..npin-1 are held resident and never touch the ring,
     * so only npin..n_layers-1 ever occupy a slot. Sizing the slot from the maximum over
     * ALL layers therefore reserves room for layer 0 -- which at 2.34 GB is the largest
     * in the model, being the only dense one with a 33792-wide MLP, and which prefix
     * pinning pins FIRST whenever anything is pinned at all. That wasted about 1.17 GB
     * for nothing at every budget above the floor.
     *
     * Ring size and pin count are mutually dependent: a smaller ring frees budget, which
     * pins more layers, which can shrink the ring again. Iterate to a fixed point. It
     * converges in two or three passes and is monotone, so the loop is bounded. At the
     * floor, where npin is 0, this correctly changes nothing: every layer streams and the
     * ring must still hold the biggest of them. */
    int64_t ring_slot = 0, spent = 0;
    int npin = 0;
    for (int pass = 0; pass < 4; pass++) {
        int64_t big = 0;
        for (int i = npin; i < tr->n_layers; i++)
            if (tr->lay[i].nbytes > big) big = tr->lay[i].nbytes;
        if (big == 0) big = tr->lay[tr->n_layers - 1].nbytes;   /* all pinned */
        int64_t rs = (big + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);
        rs += (int64_t)widen;
        rs = (rs + 4095) & ~(int64_t)4095;

        /* The ring itself must fit the budget before any layer is pinned. The loop below
         * only ever tested ADDITIONAL pinned layers against it, so RING * rs was spent
         * whether or not it fitted. Drop to one slot rather than overshoot. */
        RING = RING_WANT;
        while (RING > 1 && (int64_t)RING * rs > budget_bytes) RING--;

        int64_t sp = (int64_t)RING * rs;
        int np = 0;
        while (np < tr->n_layers) {
            const int64_t need = tr->lay[np].nbytes + (int64_t)widen;
            if (sp + need > budget_bytes) break;
            sp += need;
            np++;
        }
        if (np >= tr->n_layers) np = tr->n_layers;
        if (rs == ring_slot && np == npin) { ring_slot = rs; spent = sp; break; }
        ring_slot = rs; npin = np; spent = sp;
    }

    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;

    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(unsigned char *));
    if (!tr->pin) return -1;
    for (int i = 0; i < npin; i++) {
        const size_t need = (size_t)((tr->lay[i].nbytes + K3_TRUNK_ALIGN - 1)
                                     & ~(int64_t)(K3_TRUNK_ALIGN - 1)) + widen;
        if (k3_alloc_direct((void **)&tr->pin[i], need) != 0) {
            fprintf(stderr, "k3_trunk: cannot allocate %.2f GB for pinned layer %d\n",
                    (double)need / 1e9, i);
            return -1;
        }
    }
    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\n",
                (double)RING * ring_slot / 1e9);
        return -1;
    }
    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));
    for (int i = 0; i < RING; i++) tr->layer_of[i] = -1;
    tr->widen_bytes = (int64_t)widen;

    /* The reader is started ONLY when there are at least two slots, and this is a
     * correctness requirement rather than an optimisation.
     *
     * k3_trunk_prefetch claims tr->ring for the incoming layer. With one slot, tr->ring
     * is necessarily the slot k3_trunk_bind just returned to the caller, so the worker
     * preads layer L+1 straight over layer L's bytes while the caller is still computing
     * on them. Nothing detects it: the read succeeds, no bound pointer changes, and the
     * run completes and emits fluent, wrong tokens.
     *
     * Measured on the released checkpoint. With one slot and the reader running, the same
     * prompt that gives 17374 20829 10 427 414 1008 606 142957 instead produced
     * 32609 2329 146429 2539 11 152834 44449 7569, with no diagnostic of any kind.
     *
     * With io_state NULL, trunk_io_wait returns 0 and k3_trunk_prefetch returns
     * immediately, which is exactly the synchronous path this file had before the reader
     * existed. */
    if (RING >= 2) {
        K3TrunkIO *io = (K3TrunkIO *)calloc(1, sizeof *io);
        if (!io) return -1;
        io->tr = tr;
        pthread_mutex_init(&io->mu, NULL);
        pthread_cond_init(&io->cv, NULL);
        tr->io_state = io;
        if (pthread_create(&io->thread, NULL, trunk_io_main, io) != 0) {
            fprintf(stderr, "k3_trunk: cannot start asynchronous reader\n");
            pthread_cond_destroy(&io->cv);
            pthread_mutex_destroy(&io->mu);
            free(io);
            tr->io_state = NULL;
            return -1;
        }
    } else {
        tr->io_state = NULL;
    }

    printf("trunk stream: %.2f GB packed, %d/%d layers PINNED (%.2f GB), "
           "ring %d x %.2f GB\n",
           (double)total / 1e9, npin, tr->n_layers,
           (double)(spent - (int64_t)RING * ring_slot) / 1e9,
           RING, (double)ring_slot / 1e9);
    printf("              reads use %s\n",
           tr->direct ? "O_DIRECT (page cache bypassed)" : "buffered I/O");
    if (RING < RING_WANT)
        printf("              ring held at %d slot: a second slot needs %.2f GB and the "
               "trunk budget is %.2f GB,\n"
               "              so reads are NOT overlapped with compute. Raise --trunk-gb "
               "above %.2f GB to enable it.\n",
               RING, (double)RING_WANT * ring_slot / 1e9,
               (double)budget_bytes / 1e9,
               (double)RING_WANT * ring_slot / 1e9);
    printf("              deterministic hit rate %.1f%% (a cyclic scan defeats LRU, so "
           "a pinned prefix is used instead)\n", 100.0 * npin / tr->n_layers);
    return 0;
bad:
    free(txt);
    return -1;
}

int k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget)
{
    const int result = trunk_open(tr, dir, c, budget, 0);
    if (result) k3_trunk_close(tr);
    return result;
}

int k3_trunk_open_rows(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget)
{
    const int result = trunk_open(tr, dir, c, budget, 1);
    if (result) k3_trunk_close(tr);
    return result;
}

void k3_trunk_close(K3Trunk *tr)
{
    rows_close((K3Rows *)tr->row_state);
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (io) {
        pthread_mutex_lock(&io->mu);
        io->stop = 1;
        pthread_cond_signal(&io->cv);
        pthread_mutex_unlock(&io->mu);
        pthread_join(io->thread, NULL);
        pthread_cond_destroy(&io->cv);
        pthread_mutex_destroy(&io->mu);
        free(io);
    }
    if (tr->fd >= 0) close(tr->fd);
    k3_zfree(tr->zfile);
    if (tr->pin) { for (int i = 0; i < tr->npin; i++) k3_aligned_free(tr->pin[i]); free(tr->pin); }
    k3_aligned_free(tr->arena); free(tr->layer_of); free(tr->slot_of);
    if (tr->lay) { for (int i = 0; i < tr->n_layers; i++) free(tr->lay[i].t); free(tr->lay); }
    trunk_json_free((jval *)tr->json_root);
    free(tr->json_arena);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;            /* see k3_trunk_open: 0 is stdin, not "closed" */
}

/* Read one layer's run into dst. */

/* Allocate an O_DIRECT target on a 2 MB boundary and ask for transparent hugepages.
 *
 * WHY THIS IS NOT COSMETIC. Every O_DIRECT read must pin its destination pages in the
 * kernel (get_user_pages) for the duration of the transfer. A 2.37 GB ring slot backed by
 * 4 KB pages is 578,000 pages pinned and released PER READ, and the trunk is read 93
 * times per token: about 53.8 million pin operations, at a few hundred nanoseconds each.
 * That is on the order of ten seconds per token spent in the kernel doing page
 * bookkeeping, none of which appears in the engine's own I/O timer -- which brackets only
 * the pread loop and therefore reports a device rate that looks like the disk is
 * saturated while a third of the token is unaccounted for.
 *
 * Backing the same buffer with 2 MB pages cuts the count by 512x. The allocation is
 * otherwise identical, so this is lossless and cannot change a single output bit.
 *
 * K3_NOHUGE=1 restores 4 KB alignment so the two can be A/B compared on ONE binary,
 * which is the only way to attribute a timing difference to this decision rather than to
 * the compiler or the weather. */
static int k3_alloc_direct(void **out, size_t bytes)
{
    const int huge = !getenv("K3_NOHUGE");
    const size_t align = huge ? (2u << 20) : 4096u;
    /* Round the LENGTH up too: madvise only covers whole pages, so a 2 MB-aligned start
     * with a ragged tail leaves the last stretch on 4 KB pages. */
    const size_t len = huge ? ((bytes + align - 1) & ~(align - 1)) : bytes;
    if (posix_memalign(out, align, len) != 0) return -1;
#if defined(MADV_HUGEPAGE)
    if (huge) madvise(*out, len, MADV_HUGEPAGE);   /* advisory: failure is not an error */
#endif
    return 0;
}

/* One layer is ~1.17 GB. A single sequential pread loop leaves the drive at queue
 * depth 1 (issue, wait, issue), which on NVMe reaches roughly half its rated rate --
 * measured 3.1 GB/s here against 6.7 GB/s on the expert path, which already reads in
 * parallel (k3_cache.c cache_getmany). Splitting the layer into aligned chunks issued
 * under OpenMP gives the same depth. Chunks are multiples of K3_TRUNK_ALIGN and layer
 * offsets/sizes are too, so every chunk's offset and length stay aligned -- required
 * by O_DIRECT on Linux and harmless to F_NOCACHE on Darwin. */
#define K3_TRUNK_CHUNK ((int64_t)64 << 20)   /* 64 MiB, a multiple of K3_TRUNK_ALIGN */

static int load_run(K3Trunk *tr, int L, unsigned char *dst)
{
    const K3TrunkLayer *lay = &tr->lay[L];
    const double t0 = now_s();
    const int64_t nb = lay->nbytes;
    const int nchunk = (int)((nb + K3_TRUNK_CHUNK - 1) / K3_TRUNK_CHUNK);
    int failed = 0;
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1) reduction(|:failed)
#endif
    for (int ci = 0; ci < nchunk; ci++) {
        const int64_t base = (int64_t)ci * K3_TRUNK_CHUNK;
        const int64_t len = (nb - base < K3_TRUNK_CHUNK) ? nb - base : K3_TRUNK_CHUNK;
        int64_t got = 0;
        while (got < len) {
            int64_t want = len - got;
            if (want > K3_PREAD_MAX) want = K3_PREAD_MAX;
            int64_t r = tr->zfile
                ? k3_zread(tr->zfile, dst + base + got, want, lay->file_off + base + got)
                : pread(tr->fd, dst + base + got, (size_t)want,
                        (off_t)(lay->file_off + base + got));
            if (r <= 0) { failed = 1; break; }
            got += r;
        }
    }
    if (failed) { fprintf(stderr, "k3_trunk: short read on layer %d\n", L); return -1; }
    tr->load_seconds += now_s() - t0;
    tr->bytes_read += (uint64_t)nb;
    return 0;
}

static void *trunk_io_main(void *arg)
{
    K3TrunkIO *io = (K3TrunkIO *)arg;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (!io->busy && !io->stop)
            pthread_cond_wait(&io->cv, &io->mu);
        if (io->stop) {
            pthread_mutex_unlock(&io->mu);
            return NULL;
        }
        const int L = io->layer;
        const int slot = io->slot;
        K3Trunk *tr = io->tr;
        pthread_mutex_unlock(&io->mu);

        const int rc = load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes);

        pthread_mutex_lock(&io->mu);
        io->result = rc;
        io->done = 1;
        io->busy = 0;
        pthread_cond_broadcast(&io->cv);
        pthread_mutex_unlock(&io->mu);
    }
}

static int trunk_io_wait(K3Trunk *tr, int L)
{
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (!io) return 0;
    pthread_mutex_lock(&io->mu);
    if ((io->busy || io->done) && io->layer == L) {
        while (!io->done && !io->stop)
            pthread_cond_wait(&io->cv, &io->mu);
        const int rc = io->result;
        const int slot = io->slot;
        if (!io->stop && rc == 0) {
            tr->layer_of[slot] = L;
            tr->slot_of[L] = slot;
            tr->misses++;
        }
        io->done = 0;
        pthread_mutex_unlock(&io->mu);
        return rc == 0 ? 1 : -1;
    }
    pthread_mutex_unlock(&io->mu);
    return 0;
}

int k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b)
{
    if (L < 0 || L >= tr->n_layers) return -1;
    if (tr->row_state) {
        K3Rows *r = (K3Rows *)tr->row_state;
        if (tr->read_error || rows_wait(r)) return -1;
        r->small_used = 0; r->count = 0; r->layer = L;
        Finder f = { &tr->lay[L] }; K3MemSrc src = { find_in_layer, &f };
        tr->misses++;
        return k3_bind_layer_stream(c, L, b, &src, rows_acquire, r, NULL);
    }
    const double t_bind0 = now_s();
    k3_trunk_binds++;
    unsigned char *base;

    if (L < tr->npin) {
        base = tr->pin[L];
        if (tr->slot_of[L] < 0) {            /* first touch: load once, keep forever */
            if (load_run(tr, L, base) != 0) return -1;
            tr->slot_of[L] = L;
            tr->misses++;
        } else {
            tr->hits++;
        }
    } else {
        int slot = -1;
        const int prefetched = trunk_io_wait(tr, L);
        if (prefetched < 0) return -1;
        if (prefetched > 0) {
            slot = tr->slot_of[L];
        } else {
            for (int i = 0; i < tr->nslot; i++)
                if (tr->layer_of[i] == L) { slot = i; break; }
            if (slot >= 0) {
                tr->hits++;
            } else {
                slot = tr->ring;
                tr->ring = (tr->ring + 1) % tr->nslot;
                if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
                /* Mark the slot EMPTY before reading into it, not after. */
                tr->layer_of[slot] = -1;
                if (load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes) != 0) return -1;
                tr->layer_of[slot] = L;
                tr->misses++;
            }
        }
        base = tr->arena + (size_t)slot * tr->slot_bytes;
    }

    Finder f; f.L = &tr->lay[L];
    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;
    unsigned char *widen = base + (((tr->lay[L].nbytes + K3_TRUNK_ALIGN - 1)
                                    & ~(int64_t)(K3_TRUNK_ALIGN - 1)));
    /* Pinned layers own exactly nbytes + widen; ring slots own slot_bytes. */
    const size_t cap = (size_t)tr->widen_bytes;
    const double tw = now_s();
    const int rc = k3_bind_layer_mem(c, L, b, base, &src, widen, cap, NULL);
    const double tnow = now_s();
    k3_trunk_widen_wall += tnow - tw;
    k3_trunk_bind_wall  += tnow - t_bind0;
    return rc;
}

void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    if (tr->row_state) return;   /* rows_apply owns the reader and both buffers */
    if (L < 0 || L >= tr->n_layers || L < tr->npin) return;
    for (int i = 0; i < tr->nslot; i++) if (tr->layer_of[i] == L) return;

    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (!io) return;
    pthread_mutex_lock(&io->mu);
    if (io->busy || tr->slot_of[L] >= 0) {
        pthread_mutex_unlock(&io->mu);
        return;
    }
    const int slot = tr->ring;
    tr->ring = (tr->ring + 1) % tr->nslot;
    if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
    tr->layer_of[slot] = -1;
    io->layer = L;
    io->slot = slot;
    io->done = 0;
    io->busy = 1;
    pthread_cond_signal(&io->cv);
    pthread_mutex_unlock(&io->mu);
}

void k3_trunk_report(const K3Trunk *tr, const char *label)
{
    const uint64_t n = tr->hits + tr->misses;
    printf("trunk [%s]\n", label ? label : "");
    if (tr->row_state) {
        /* The row pipeline has no pins, no ring and no timed bind: each matrix is read
         * tile by tile inside the matmul that uses it, so the ring's bind-wall breakdown
         * below would divide device time by zero binds and call all of it overlapped.
         * What is measured instead is split by thread. The main thread's own vector reads
         * and its waits for a tile are time no compute overlapped; the reader thread's
         * tile reads are the rest of load_seconds, and whatever of them the main thread
         * did not wait for ran beside a matmul. */
        const double reader = tr->load_seconds - tr->row_sync_seconds;
        printf("  row pipeline: %llu layer binds, %llu matrix passes, two %.2f MiB buffers\n",
               (unsigned long long)n, (unsigned long long)tr->matrix_calls,
               (double)tr->row_buffer_bytes / 2.0 / (1 << 20));
        printf("  read %.2f GB in %.2f s of device time (%.0f MB/s)\n",
               (double)tr->bytes_read / 1e9, tr->load_seconds,
               tr->load_seconds > 0 ? (double)tr->bytes_read / 1e6 / tr->load_seconds : 0.0);
        printf("  reader thread %.2f s of tile reads; main thread waited %.2f s for tiles "
               "and read layer vectors for %.2f s\n",
               reader, tr->row_wait_seconds, tr->row_sync_seconds);
        return;
    }
    printf("  pinned %d/%d layers, ring %d slots\n", tr->npin, tr->n_layers, tr->nslot);
    printf("  binds %llu, hits %llu (%.1f%%), reads %llu\n",
           (unsigned long long)n, (unsigned long long)tr->hits,
           n ? 100.0 * tr->hits / n : 0.0, (unsigned long long)tr->misses);
    printf("  read %.2f GB in %.2f s (%.0f MB/s)\n",
           (double)tr->bytes_read / 1e9, tr->load_seconds,
           tr->load_seconds > 0 ? (double)tr->bytes_read / 1e6 / tr->load_seconds : 0.0);
    /* The rate above is a DEVICE rate: load_seconds brackets the pread loop alone. The
     * breakdown below is the wall clock actually spent inside k3_trunk_bind, so the
     * difference between them is per-bind overhead rather than disk time.
     *
     * Reporting the widen step separately is what distinguishes a slow device from
     * excessive work per bind, two causes with the same symptom and different fixes. */
    {
        /* load_seconds is DEVICE time and, with more than one ring slot, some of it
         * happens on the reader thread while the main thread is computing. Subtracting it
         * from bind wall clock then goes negative by exactly the amount of overlap
         * achieved, which is how the previous form of this line reported the feature
         * working as "other -157.06" and a read share of 207%. Overlapped time is a
         * result, not unattributed overhead, so it is named rather than subtracted. */
        const double serial = tr->load_seconds + k3_trunk_widen_wall;
        const double overlapped = serial - k3_trunk_bind_wall;
        if (overlapped > 0.0) {
            printf("  bind wall %.2f s over %ld binds; read %.2f + widen %.2f = %.2f s of "
                   "device work,\n"
                   "                    of which %.2f s (%.0f%%) overlapped compute on the "
                   "reader thread\n",
                   k3_trunk_bind_wall, k3_trunk_binds, tr->load_seconds,
                   k3_trunk_widen_wall, serial, overlapped,
                   serial > 0.0 ? 100.0 * overlapped / serial : 0.0);
        } else {
            const double other = k3_trunk_bind_wall - serial;
            printf("  bind wall %.2f s over %ld binds  =  read %.2f + widen %.2f + other %.2f\n",
                   k3_trunk_bind_wall, k3_trunk_binds, tr->load_seconds,
                   k3_trunk_widen_wall, other);
            if (k3_trunk_bind_wall > 0.0)
                printf("                    shares:      read %.0f%%  widen %.0f%%  other %.0f%%\n",
                       100.0 * tr->load_seconds / k3_trunk_bind_wall,
                       100.0 * k3_trunk_widen_wall / k3_trunk_bind_wall,
                       100.0 * other / k3_trunk_bind_wall);
        }
    }
}
