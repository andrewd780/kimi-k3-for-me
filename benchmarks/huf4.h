/* Research-only four-stream decoder. The alphabet/table are the shelved codec's;
 * the FOUR independent streams require a new encoder layout. Not an archive API. */
#ifndef K3_HUF4_H
#define K3_HUF4_H
#include "../docs/notes/k3_huf.h.shelved"

typedef struct {
    const uint8_t *p, *end;
    uint64_t bits;
    unsigned held;
} HufReader;

static inline int huf_symbol(HufReader *r, const K3HufTable *t)
{
    if (r->held < 12) {
        while (r->held <= 56 && r->p < r->end) {
            r->bits = (r->bits << 8) | *r->p++;
            r->held += 8;
        }
    }
    const unsigned at = r->held >= 12
        ? (unsigned)(r->bits >> (r->held - 12)) & 4095u
        : (unsigned)(r->bits << (12 - r->held)) & 4095u;
    const unsigned value = t->sym_len[at], n = value & 15u;
    if (!n || n > r->held) return -1;
    r->held -= n;
    return (int)(value >> 4);
}

static int huf4_decode(const K3HufTable *table, const uint8_t *src[4],
                       const size_t len[4], const uint8_t *low, uint8_t *out, size_t n)
{
    HufReader r[4];
    for (int lane = 0; lane < 4; lane++) {
        r[lane].p = src[lane]; r[lane].end = src[lane] + len[lane];
        r[lane].bits = 0; r[lane].held = 0;
    }
    const size_t symbols = n / 2;
    size_t i = 0;
    for (; i + 3 < symbols; i += 4) {
        const int a = huf_symbol(&r[0], table);
        const int b = huf_symbol(&r[1], table);
        const int c = huf_symbol(&r[2], table);
        const int d = huf_symbol(&r[3], table);
        if ((a | b | c | d) < 0) return -1;
        out[2*i] = low[i]; out[2*i+1] = (uint8_t)a;
        out[2*i+2] = low[i+1]; out[2*i+3] = (uint8_t)b;
        out[2*i+4] = low[i+2]; out[2*i+5] = (uint8_t)c;
        out[2*i+6] = low[i+3]; out[2*i+7] = (uint8_t)d;
    }
    for (; i < symbols; i++) {
        const int value = huf_symbol(&r[i % 4], table);
        if (value < 0) return -1;
        out[2*i] = low[i]; out[2*i+1] = (uint8_t)value;
    }
    if (n % 2) out[n-1] = low[n/2];
    /* Accept only the encoder's zero bit padding, no trailing bytes. */
    for (int lane = 0; lane < 4; lane++) {
        if (r[lane].p != r[lane].end || r[lane].held > 7 ||
            (r[lane].bits & (((uint64_t)1 << r[lane].held) - 1))) return -1;
    }
    return 0;
}
#endif
