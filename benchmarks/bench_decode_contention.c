/* Decode under matmul contention: gate 3 of the fixed-width trunk note.
 *
 * One decoder thread reconstructs a real-shape 12288 x 7168 BF16 matrix (KDA q_proj)
 * from an FD4B or FD3B stream in 585-row chunks (the most whole rows that fit the row
 * pipeline's 8 MiB buffer), each chunk located through the FDRX row index, into two
 * alternating 8 MiB buffers. Meanwhile the engine's own k3_matmul_bf16 multiplies the
 * same shape on the remaining OpenMP threads. Four arms, repeated in interleaved order:
 *
 *   decode_alone      the decoder thread only
 *   matmul_all        k3_matmul_bf16 on all T threads: the uncompressed baseline
 *   matmul_rest       k3_matmul_bf16 on T-1 threads
 *   concurrent        decoder (1 thread) and matmul (T-1 threads) at the same time
 *
 * Rates: decode in reconstructed BF16 GB/s (bench_huf4 units), matmul in weight GB/s
 * consumed. In the concurrent arm only work completed before the deadline counts, and
 * each side keeps working until the other has finished, so every counted unit ran
 * against a live competitor. Every chunk is checked byte-exact before and after timing.
 *
 * Input placement brackets the pipeline. "stream" cycles through all 22 chunks of the
 * whole compressed matrix, so the decoder reads DRAM as a cold row pipeline would;
 * "hot" re-decodes one chunk, whose compressed bytes stay cache-resident, as when the
 * decoder runs right behind the read that landed them. Both write alternating buffers.
 *
 * Synthetic weights: high bytes drawn from the committed four-range high-byte histogram
 * (docs/measurements/research-two-symbol.json, experiments[1], 2,097,152 values), low
 * bytes uniform. The dictionaries are the committed gate-1 pooled table (FD4B) and its
 * first seven entries (FD3B). No model weights are read.
 *
 *   bench_decode_contention <3|4> [threads] [seconds_per_arm] [repeats] [stream|hot]
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "k3.h"
#include "fixed_dictionary.h"

#define ROWS 12288
#define COLS 7168
#define CHUNK_BYTES ((size_t)8 << 20)
#define MAX_REPEATS 16

static const struct { uint8_t value; uint32_t count; } HIGH[] = {
    {60, 519798}, {188, 518923}, {61, 309278}, {189, 308338}, {59, 164903}, {187, 164459},
    {186, 41711}, {58, 41695}, {185, 10473}, {57, 10293}, {56, 2582}, {184, 2573},
    {183, 633}, {55, 624}, {190, 226}, {62, 190}, {54, 179}, {182, 176}, {53, 40},
    {181, 32}, {180, 10}, {52, 7}, {51, 3}, {50, 2}, {179, 2}, {177, 1}, {178, 1}};
static const uint8_t DICT[15] = {188, 60, 61, 189, 59, 187, 58, 186, 185, 57, 190, 62, 56, 184, 183};

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next64(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

/* Exactly 2^21 histogram slots, so a uniform 21-bit draw samples the histogram. */
static void synthesize(uint8_t *raw, size_t n)
{
    uint8_t *slot = (uint8_t *)malloc((size_t)1 << 21);
    size_t at = 0;
    if (!slot) exit(2);
    for (size_t k = 0; k < sizeof HIGH / sizeof *HIGH; k++)
        for (uint32_t c = 0; c < HIGH[k].count; c++) slot[at++] = HIGH[k].value;
    if (at != (size_t)1 << 21) exit(2);
    for (size_t i = 0; i + 1 < n; i += 2) {
        const uint64_t r = next64();
        raw[i] = (uint8_t)(r >> 40);
        raw[i + 1] = slot[r & ((1u << 21) - 1)];
    }
    free(slot);
}

static void put32(uint8_t *p, size_t v)
{
    for (int j = 0; j < 4; j++) p[j] = (uint8_t)(v >> (8 * j));
}

/* Plain encoders; the decode is checked byte-exact against raw, so they need not be fast. */
static size_t encode(uint8_t *p, const uint8_t *raw, size_t n, unsigned bits)
{
    const size_t entries = bits == 3 ? 7 : 15, values = n / 2;
    const size_t indexes = (bits * values + 7) / 8, low = (n + 1) / 2;
    uint8_t code[256];
    memset(code, (int)entries, sizeof code);
    for (size_t k = 0; k < entries; k++) code[DICT[k]] = (uint8_t)k;
    memset(p, 0, 32 + indexes);
    memcpy(p, bits == 3 ? "FD3B" : "FD4B", 4);
    put32(p + 4, n);
    memcpy(p + 16, DICT, entries);
    uint8_t *escape = p + 32 + indexes + low;
    size_t escapes = 0;
    for (size_t j = 0; j < low; j++) p[32 + indexes + j] = raw[2 * j];
    for (size_t j = 0; j < values; j++) {
        const unsigned c = code[raw[2 * j + 1]];
        const size_t bit = bits * j;
        p[32 + bit / 8] |= (uint8_t)(c << (bit % 8));
        if (bit % 8 + bits > 8) p[32 + bit / 8 + 1] |= (uint8_t)(c >> (8 - bit % 8));
        if (c == entries) escape[escapes++] = raw[2 * j + 1];
    }
    put32(p + 8, escapes);
    const size_t slack = bits == 3 ? FWD3_SLACK : 0;
    memset(escape + escapes, 0, slack);
    return 32 + indexes + low + escapes + slack;
}

static uint8_t *row_index(const uint8_t *raw, unsigned bits, size_t *length)
{
    const size_t entries = bits == 3 ? 7 : 15;
    uint8_t known[256] = {0};
    for (size_t k = 0; k < entries; k++) known[DICT[k]] = 1;
    uint8_t *x = (uint8_t *)malloc(16 + 4 * ROWS);
    if (!x) exit(2);
    memcpy(x, "FDRX", 4); put32(x + 4, ROWS); put32(x + 8, COLS); put32(x + 12, 1);
    size_t before = 0;
    for (size_t r = 0; r < ROWS; r++) {
        put32(x + 16 + 4 * r, before);
        for (size_t c = 0; c < COLS; c++) before += !known[raw[2 * (r * COLS + c) + 1]];
    }
    *length = 16 + 4 * ROWS;
    return x;
}

typedef struct {
    const FwdView *whole;
    const FwdRows *rows;
    uint8_t *out[2];
    size_t chunk_rows, chunks, cycle, next;
    double start, deadline, last;
    size_t counted;
    int failed, concurrent;
    volatile int *partner_done;
} Decoder;

static int decode_chunk(Decoder *d, size_t chunk, uint8_t *out)
{
    FwdView v;
    const size_t first = chunk * d->chunk_rows;
    const size_t last = first + d->chunk_rows < ROWS ? first + d->chunk_rows : ROWS;
    if (fwd_rows_view(d->rows, d->whole, first, last, &v)) return -1;
    return fwd_decode_any_native(&v, out, CHUNK_BYTES) ? -1 : (int)(last - first);
}

static void *decode_worker(void *arg)
{
    Decoder *d = (Decoder *)arg;
    for (size_t k = 0;; k++) {
        const int got = decode_chunk(d, d->next, d->out[k & 1]);
        const double t = now();
        if (got < 0) { d->failed = 1; break; }
        d->next = (d->next + 1) % d->cycle;
        if (t <= d->deadline) {
            d->counted += (size_t)got * COLS * 2;
            d->last = t;
        }
        if (d->concurrent ? __atomic_load_n(d->partner_done, __ATOMIC_ACQUIRE) : t > d->deadline)
            break;
    }
    return NULL;
}

typedef struct { double seconds; size_t bytes; } Rate;

static double gbps(Rate r) { return r.seconds > 0 ? r.bytes / r.seconds / 1e9 : 0; }

static void set_threads(int n)
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

/* Matmul calls until the deadline; only calls finishing by it count. */
static Rate matmul_until(float *y, const float *x, const uint16_t *w, double start, double deadline)
{
    Rate r = {0, 0};
    do {
        k3_matmul_bf16(y, x, w, COLS, ROWS);
        const double t = now();
        if (t <= deadline) { r.bytes += (size_t)ROWS * COLS * 2; r.seconds = t - start; }
    } while (now() < deadline);
    return r;
}

static int by_value(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void emit(const char *name, const double *v, int n, int last)
{
    double s[MAX_REPEATS];
    memcpy(s, v, n * sizeof *v);
    qsort(s, n, sizeof *s, by_value);
    printf("\"%s\":{\"GBps_runs\":[", name);
    for (int i = 0; i < n; i++) printf("%s%.6f", i ? "," : "", v[i]);
    printf("],\"median\":%.6f,\"min\":%.6f}%s", n % 2 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2,
           s[0], last ? "" : ",");
}

int main(int argc, char **argv)
{
    const unsigned bits = argc > 1 ? (unsigned)atoi(argv[1]) : 4;
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const int threads = argc > 2 ? atoi(argv[2]) : (int)cpus;
    const double seconds = argc > 3 ? atof(argv[3]) : 1.0;
    const int repeats = argc > 4 ? atoi(argv[4]) : 5;
    const char *input = argc > 5 ? argv[5] : "stream";
    if ((bits != 3 && bits != 4) || threads < 2 || seconds <= 0 || repeats < 1 ||
        repeats > MAX_REPEATS || (strcmp(input, "stream") && strcmp(input, "hot"))) {
        fprintf(stderr, "usage: %s <3|4> [threads>=2] [seconds] [repeats<=16] [stream|hot]\n",
                argv[0]);
        return 2;
    }
    const size_t n = (size_t)ROWS * COLS * 2;
    uint8_t *raw = (uint8_t *)malloc(n);
    uint8_t *packed = (uint8_t *)malloc(n + n / 2 + 64);
    float *x = (float *)malloc(COLS * sizeof *x), *y = (float *)malloc(ROWS * sizeof *y);
    Decoder d;
    memset(&d, 0, sizeof d);
    d.out[0] = (uint8_t *)malloc(CHUNK_BYTES);
    d.out[1] = (uint8_t *)malloc(CHUNK_BYTES);
    if (!raw || !packed || !x || !y || !d.out[0] || !d.out[1]) return 2;
    synthesize(raw, n);
    for (int i = 0; i < COLS; i++) x[i] = (float)((int)(next64() % 2001) - 1000) * 1e-4f;
    const size_t length = encode(packed, raw, n, bits);
    size_t index_length;
    uint8_t *index = row_index(raw, bits, &index_length);
    FwdView whole;
    FwdRows rows;
    if (fwd_parse_any(&whole, packed, length) || whole.bits != bits ||
        fwd_rows_parse(&rows, index, index_length, &whole)) return 1;
    d.whole = &whole;
    d.rows = &rows;
    d.chunk_rows = CHUNK_BYTES / (2 * COLS);
    d.chunks = (ROWS + d.chunk_rows - 1) / d.chunk_rows;
    d.cycle = strcmp(input, "hot") ? d.chunks : 1;
    /* Byte-exact before timing, through both decoders and the row index. */
    for (size_t c = 0; c < d.chunks; c++) {
        FwdView v;
        const size_t first = c * d.chunk_rows;
        const size_t last = first + d.chunk_rows < ROWS ? first + d.chunk_rows : ROWS;
        if (fwd_rows_view(&rows, &whole, first, last, &v) ||
            fwd_decode_any_scalar(&v, d.out[0], CHUNK_BYTES) ||
            memcmp(d.out[0], raw + 2 * first * COLS, v.raw_bytes) ||
            decode_chunk(&d, c, d.out[1]) < 0 ||
            memcmp(d.out[1], raw + 2 * first * COLS, v.raw_bytes)) return 1;
    }
    const uint16_t *w = (const uint16_t *)raw;   /* the matmul multiplies the same matrix */
    set_threads(threads);
    k3_matmul_bf16(y, x, w, COLS, ROWS);         /* warm the team and the pages */

    double dec_alone[MAX_REPEATS], all[MAX_REPEATS], rest[MAX_REPEATS];
    double dec_conc[MAX_REPEATS], mm_conc[MAX_REPEATS];
    struct timespec pause = {0, 50 * 1000 * 1000};
    for (int rep = 0; rep < repeats; rep++) {
        pthread_t t;
        /* decoder alone */
        nanosleep(&pause, NULL);
        d.concurrent = 0; d.counted = 0; d.start = now(); d.last = d.start;
        d.deadline = d.start + seconds;
        if (pthread_create(&t, NULL, decode_worker, &d) || pthread_join(t, NULL) || d.failed) return 1;
        dec_alone[rep] = gbps((Rate){d.last - d.start, d.counted});
        /* matmul on every thread, then on the rest */
        nanosleep(&pause, NULL);
        set_threads(threads);
        double start = now();
        all[rep] = gbps(matmul_until(y, x, w, start, start + seconds));
        nanosleep(&pause, NULL);
        set_threads(threads - 1);
        start = now();
        rest[rep] = gbps(matmul_until(y, x, w, start, start + seconds));
        /* both at once */
        nanosleep(&pause, NULL);
        volatile int done = 0;
        d.concurrent = 1; d.partner_done = &done; d.counted = 0;
        d.start = now(); d.last = d.start; d.deadline = d.start + seconds;
        if (pthread_create(&t, NULL, decode_worker, &d)) return 1;
        const Rate m = matmul_until(y, x, w, d.start, d.deadline);
        __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
        if (pthread_join(t, NULL) || d.failed) return 1;
        mm_conc[rep] = gbps(m);
        dec_conc[rep] = gbps((Rate){d.last - d.start, d.counted});
    }
    /* Byte-exact after timing as well. */
    for (size_t c = 0; c < d.chunks; c++) {
        const int got = decode_chunk(&d, c, d.out[0]);
        if (got < 0 || memcmp(d.out[0], raw + 2 * c * d.chunk_rows * COLS, (size_t)got * COLS * 2))
            return 1;
    }
#ifdef _OPENMP
    const int openmp = 1;
#else
    const int openmp = 0;
#endif
    printf("{\"native\":\"%s\",\"index_bits\":%u,\"input\":\"%s\",\"threads\":%d,\"openmp\":%s,"
           "\"matrix\":[%d,%d],\"raw_bytes\":%zu,\"packed_bytes\":%zu,\"row_index_bytes\":%zu,"
           "\"escape_values\":%zu,\"chunk_rows\":%zu,\"chunks\":%zu,\"seconds_per_arm\":%.3f,"
           "\"byte_exact\":true,\"arms\":{",
           bits == 3 ? fwd3_native_name() : fwd_native_name(), bits, input, threads,
           openmp ? "true" : "false", ROWS, COLS, n, length,
           index_length, whole.escapes, d.chunk_rows, d.chunks, seconds);
    emit("decode_alone", dec_alone, repeats, 0);
    emit("matmul_all_threads", all, repeats, 0);
    emit("matmul_rest_threads", rest, repeats, 0);
    emit("decode_concurrent", dec_conc, repeats, 0);
    emit("matmul_concurrent", mm_conc, repeats, 1);
    printf("}}\n");
    free(raw); free(packed); free(x); free(y); free(index); free(d.out[0]); free(d.out[1]);
    return 0;
}
