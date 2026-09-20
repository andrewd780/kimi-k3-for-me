/* Benchmark-only lossless BF16 high-byte dictionary; not an inference archive.
 * FD4B: magic, raw-byte count, escape count, reserved u32 (all LE), 16 table
 * bytes (15 distinct values + zero sentinel), low-nibble-first indexes,
 * ceil(raw/2) verbatim low bytes, then escape bytes in occurrence order.
 * An odd final raw byte is preserved. Index padding must be zero.
 */
#ifndef FIXED_DICTIONARY_H
#define FIXED_DICTIONARY_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef struct {
    const uint8_t *table, *index, *low, *escape;
    size_t raw_bytes, values, escapes;
} FwdView;

static uint32_t fwd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static int fwd_parse(FwdView *v, const uint8_t *data, size_t length)
{
    if (length < 32 || memcmp(data, "FD4B", 4) || fwd_u32(data + 12)) return -1;
    const size_t raw = fwd_u32(data + 4), escapes = fwd_u32(data + 8);
    if (raw > (16u << 20) || escapes > raw / 2) return -1;
    const size_t n = raw / 2, indexes = (n + 1) / 2, lows = (raw + 1) / 2;
    if (length != 32 + indexes + lows + escapes || data[31]) return -1;
    for (size_t i = 0; i < 15; i++)
        for (size_t j = 0; j < i; j++)
            if (data[16 + i] == data[16 + j]) return -1;
    if ((n & 1) && (data[32 + indexes - 1] & 0xf0)) return -1;
    v->table = data + 16;
    v->index = data + 32;
    v->low = v->index + indexes;
    v->escape = v->low + lows;
    v->raw_bytes = raw;
    v->values = n;
    v->escapes = escapes;
    return 0;
}

static const char *fwd_native_name(void)
{
#if defined(__SSSE3__)
    return "ssse3_pshufb";
#elif defined(__aarch64__)
    return "neon_tbl";
#else
    return "scalar_fallback";
#endif
}

static int fwd_decode_scalar(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (capacity < v->raw_bytes) return -1;
    size_t e = 0;
    for (size_t i = 0; i < v->values; i++) {
        const unsigned code = (v->index[i / 2] >> (4 * (i & 1))) & 15;
        uint8_t high = v->table[code];
        if (code == 15) {
            if (e == v->escapes) return -1;
            high = v->escape[e++];
        }
        out[2*i] = v->low[i];
        out[2*i + 1] = high;
    }
    if (v->raw_bytes & 1) out[v->raw_bytes - 1] = v->low[v->values];
    return e == v->escapes ? 0 : -1;
}

static int fwd_decode_native(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (capacity < v->raw_bytes) return -1;
    size_t i = 0, e = 0;
#if defined(__SSSE3__)
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i table = _mm_loadu_si128((const __m128i *)v->table);
    for (; i + 32 <= v->values; i += 32) {
        const __m128i packed = _mm_loadu_si128((const __m128i *)(v->index + i/2));
        const __m128i a = _mm_and_si128(packed, mask);
        const __m128i b = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        const __m128i c = _mm_unpacklo_epi8(a, b), d = _mm_unpackhi_epi8(a, b);
        const __m128i hc = _mm_shuffle_epi8(table, c), hd = _mm_shuffle_epi8(table, d);
        const __m128i lc = _mm_loadu_si128((const __m128i *)(v->low + i));
        const __m128i ld = _mm_loadu_si128((const __m128i *)(v->low + i + 16));
        _mm_storeu_si128((__m128i *)(out + 2*i), _mm_unpacklo_epi8(lc, hc));
        _mm_storeu_si128((__m128i *)(out + 2*i + 16), _mm_unpackhi_epi8(lc, hc));
        _mm_storeu_si128((__m128i *)(out + 2*i + 32), _mm_unpacklo_epi8(ld, hd));
        _mm_storeu_si128((__m128i *)(out + 2*i + 48), _mm_unpackhi_epi8(ld, hd));
        uint32_t escaped = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(c, mask)) |
                          (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(d, mask)) << 16;
        while (escaped) {
            const unsigned at = (unsigned)__builtin_ctz(escaped);
            if (e == v->escapes) return -1;
            out[2*(i + at) + 1] = v->escape[e++];
            escaped &= escaped - 1;
        }
    }
#elif defined(__aarch64__)
    const uint8x16_t mask = vdupq_n_u8(15), table = vld1q_u8(v->table);
    for (; i + 32 <= v->values; i += 32) {
        const uint8x16_t packed = vld1q_u8(v->index + i/2);
        const uint8x16_t a = vandq_u8(packed, mask), b = vshrq_n_u8(packed, 4);
        const uint8x16_t c = vzip1q_u8(a, b), d = vzip2q_u8(a, b);
        const uint8x16x2_t first = {{vld1q_u8(v->low + i), vqtbl1q_u8(table, c)}};
        const uint8x16x2_t second = {{vld1q_u8(v->low + i + 16), vqtbl1q_u8(table, d)}};
        vst2q_u8(out + 2*i, first);
        vst2q_u8(out + 2*i + 32, second);
        if (vmaxvq_u8(vorrq_u8(vceqq_u8(a, mask), vceqq_u8(b, mask)))) {
            for (size_t j = 0; j < 32; j++) {
                const unsigned code = (v->index[(i+j)/2] >> (4 * (j & 1))) & 15;
                if (code == 15) {
                    if (e == v->escapes) return -1;
                    out[2*(i+j) + 1] = v->escape[e++];
                }
            }
        }
    }
#endif
    for (; i < v->values; i++) {
        const unsigned code = (v->index[i/2] >> (4 * (i & 1))) & 15;
        uint8_t high = v->table[code];
        if (code == 15) {
            if (e == v->escapes) return -1;
            high = v->escape[e++];
        }
        out[2*i] = v->low[i];
        out[2*i + 1] = high;
    }
    if (v->raw_bytes & 1) out[v->raw_bytes - 1] = v->low[v->values];
    return e == v->escapes ? 0 : -1;
}
#endif
