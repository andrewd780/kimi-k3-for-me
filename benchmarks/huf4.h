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

static inline void huf_refill(HufReader *r)
{
    if (r->held < 12) {
        if ((size_t)(r->end - r->p) >= 4) {
            /* Four-byte refill keeps at most 43 bits live. Compilers recognize
             * this portable big-endian load; no unaligned or speculative overread. */
            const uint32_t word = ((uint32_t)r->p[0] << 24) | ((uint32_t)r->p[1] << 16)
                                | ((uint32_t)r->p[2] << 8) | r->p[3];
            r->bits = (r->bits << 32) | word;
            r->held += 32; r->p += 4;
        } else {
            while (r->p < r->end) {
                r->bits = (r->bits << 8) | *r->p++;
                r->held += 8;
            }
        }
    }
}

static inline unsigned huf_index(const HufReader *r)
{
    return r->held >= 12
        ? (unsigned)(r->bits >> (r->held - 12)) & 4095u
        : (unsigned)(r->bits << (12 - r->held)) & 4095u;
}

static inline int huf_symbol(HufReader *r, const K3HufTable *t)
{
    huf_refill(r);
    const unsigned at = huf_index(r);
    const unsigned value = t->sym_len[at], n = value & 15u;
    if (!n || n > r->held) return -1;
    r->held -= n;
    return (int)(value >> 4);
}

/* A 12-bit prefix holds TWO symbols only when their combined code length fits.
 * Otherwise it holds one. Count and consumed bits are stored explicitly; neither
 * the stream layout nor the symbol count is guessed from output capacity. */
static void huf_pairs(uint32_t pairs[4096], const K3HufTable *table)
{
    for (unsigned i = 0; i < 4096; i++) {
        const unsigned a = table->sym_len[i], na = a & 15u;
        if (!na) { pairs[i] = 0; continue; }
        const unsigned b = table->sym_len[(i << na) & 4095u], nb = b & 15u;
        pairs[i] = (a >> 4) | (na << 16);
        if (nb && na + nb <= 12)
            pairs[i] = (a >> 4) | ((b >> 4) << 8) | ((na + nb) << 16) | (1u << 20);
    }
}

static inline int huf_pair(HufReader *r, const uint32_t *pairs,
                           uint8_t *out, size_t *pos)
{
    huf_refill(r);
    const uint32_t entry = pairs[huf_index(r)];
    const unsigned bits = (entry >> 16) & 15u, two = entry >> 20;
    if (!bits || bits > r->held) return -1;
    r->held -= bits;
    out[2 * *pos + 1] = (uint8_t)entry;
    if (two) out[2 * *pos + 9] = (uint8_t)(entry >> 8);
    *pos += 4u * (1u + two);
    return 0;
}

static int huf4_decode(const K3HufTable *table, const uint32_t pairs[4096], const uint8_t *src[4],
                       const size_t len[4], const uint8_t *low, uint8_t *out, size_t n)
{
    HufReader r[4];
    for (int lane = 0; lane < 4; lane++) {
        r[lane].p = src[lane]; r[lane].end = src[lane] + len[lane];
        r[lane].bits = 0; r[lane].held = 0;
    }
    const size_t symbols = n / 2;
    size_t p0 = 0, p1 = 1, p2 = 2, p3 = 3;
    while (p0 + 4 < symbols && p1 + 4 < symbols && p2 + 4 < symbols && p3 + 4 < symbols) {
        const int a = huf_pair(&r[0], pairs, out, &p0);
        const int b = huf_pair(&r[1], pairs, out, &p1);
        const int c = huf_pair(&r[2], pairs, out, &p2);
        const int d = huf_pair(&r[3], pairs, out, &p3);
        if ((a | b | c | d) < 0) return -1;
    }
    const size_t positions[4] = {p0, p1, p2, p3};
    for (int lane = 0; lane < 4; lane++) {
        for (size_t i = positions[lane]; i < symbols; i += 4) {
            const int value = huf_symbol(&r[lane], table);
            if (value < 0) return -1;
            out[2*i+1] = (uint8_t)value;
        }
    }
    for (size_t i = 0; i < (n + 1)/2; i++) out[2*i] = low[i];
    /* Accept only the encoder's zero bit padding, no trailing bytes. */
    for (int lane = 0; lane < 4; lane++) {
        if (r[lane].p != r[lane].end || r[lane].held > 7 ||
            (r[lane].bits & (((uint64_t)1 << r[lane].held) - 1))) return -1;
    }
    return 0;
}
#endif
