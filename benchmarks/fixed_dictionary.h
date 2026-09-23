/* Benchmark-only lossless BF16 high-byte dictionary; not an inference archive.
 * FD4B: magic, raw-byte count, escape count, reserved u32 (all LE), 16 table
 * bytes (15 distinct values + zero sentinel), low-nibble-first indexes,
 * ceil(raw/2) verbatim low bytes, then escape bytes in occurrence order.
 * An odd final raw byte is preserved. Index padding must be zero.
 *
 * FD3B: the same header with magic "FD3B" and 7 distinct table values followed
 * by 9 zero bytes; code 7 escapes. Codes are a little-endian bitstream, code i
 * in bits [3i, 3i+3); padding bits are zero. Then ceil(raw/2) low bytes, the
 * escape bytes, and FWD3_SLACK zero bytes so vector code may over-read.
 *
 * FDRX: row index making whole-row ranges of one even-length stream (a matrix
 * of rows x cols BF16, cols * index bits a multiple of 8) independently
 * decodable: magic, rows, cols, group (u32 LE), then ceil(rows/group) u32 LE
 * checkpoints, entry j = escapes stored before row j*group. A range starting
 * inside a group counts escape codes from the group start (fwd_rows_view).
 */
#ifndef FIXED_DICTIONARY_H
#define FIXED_DICTIONARY_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#define FWD3_SLACK 32
/* One stream may hold a whole K3 matrix (at most 484 MB, the dense layer's MLP). */
#define FWD_MAX_RAW ((size_t)1 << 30)

typedef struct {
    const uint8_t *table, *index, *low, *escape;
    size_t raw_bytes, values, escapes;
    size_t escape_readable;     /* bytes readable at escape (FD3B: escapes + slack) */
    unsigned bits;              /* index bits: 4 (FD4B) or 3 (FD3B) */
} FwdView;

static inline uint32_t fwd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static inline int fwd_parse(FwdView *v, const uint8_t *data, size_t length)
{
    if (length < 32 || memcmp(data, "FD4B", 4) || fwd_u32(data + 12)) return -1;
    const size_t raw = fwd_u32(data + 4), escapes = fwd_u32(data + 8);
    if (raw > FWD_MAX_RAW || escapes > raw / 2) return -1;
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
    v->escape_readable = escapes;
    v->bits = 4;
    return 0;
}

static inline const char *fwd_native_name(void)
{
#if defined(__SSSE3__)
    return "ssse3_pshufb";
#elif defined(__aarch64__)
    return "neon_tbl";
#else
    return "scalar_fallback";
#endif
}

static inline int fwd_decode_scalar(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (v->bits != 4 || capacity < v->raw_bytes) return -1;
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

static inline int fwd_decode_native(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (v->bits != 4 || capacity < v->raw_bytes) return -1;
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

/* ------------------------------------------------------------------ FD3B ---- */

static inline int fwd3_parse(FwdView *v, const uint8_t *data, size_t length)
{
    if (length < 32 || memcmp(data, "FD3B", 4) || fwd_u32(data + 12)) return -1;
    const size_t raw = fwd_u32(data + 4), escapes = fwd_u32(data + 8);
    if (raw > FWD_MAX_RAW || escapes > raw / 2) return -1;
    const size_t n = raw / 2, indexes = (3 * n + 7) / 8, lows = (raw + 1) / 2;
    if (length != 32 + indexes + lows + escapes + FWD3_SLACK) return -1;
    for (size_t i = 23; i < 32; i++)
        if (data[i]) return -1;
    for (size_t i = 0; i < 7; i++)
        for (size_t j = 0; j < i; j++)
            if (data[16 + i] == data[16 + j]) return -1;
    if (((3 * n) & 7) && (data[32 + indexes - 1] >> ((3 * n) & 7))) return -1;
    for (size_t i = length - FWD3_SLACK; i < length; i++)
        if (data[i]) return -1;
    v->table = data + 16;
    v->index = data + 32;
    v->low = v->index + indexes;
    v->escape = v->low + lows;
    v->raw_bytes = raw;
    v->values = n;
    v->escapes = escapes;
    v->escape_readable = escapes + FWD3_SLACK;
    v->bits = 3;
    return 0;
}

/* Code i spans at most two bytes: bits [3i, 3i+3) with 3i mod 8 <= 7. */
static inline unsigned fwd3_code(const uint8_t *index, size_t i)
{
    const size_t bit = 3 * i;
    unsigned w = index[bit >> 3];
    if ((bit & 7) > 5) w |= (unsigned)index[(bit >> 3) + 1] << 8;
    return (w >> (bit & 7)) & 7;
}

static inline int fwd3_decode_scalar(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (v->bits != 3 || capacity < v->raw_bytes) return -1;
    size_t e = 0;
    for (size_t i = 0; i < v->values; i++) {
        const unsigned code = fwd3_code(v->index, i);
        uint8_t high = v->table[code];
        if (code == 7) {
            if (e == v->escapes) return -1;
            high = v->escape[e++];
        }
        out[2*i] = v->low[i];
        out[2*i + 1] = high;
    }
    if (v->raw_bytes & 1) out[v->raw_bytes - 1] = v->low[v->values];
    return e == v->escapes ? 0 : -1;
}

/* Vector FD3B, 32 values per step. Codes: gather each code's two bytes into a
 * 16-bit lane, shift right per lane (multiply by 2^(13-s), keep the top three
 * bits), pack. Escapes, frequent at 3 bits: an exclusive prefix count of the
 * escape lanes indexes a shuffle of the next 16 escape bytes, blended in with
 * no branch. The escape stream's slack makes the 16-byte load safe; a surplus
 * of escape codes is detected after every step. That check also bounds the
 * scalar tail, which stops only at e == escapes: a step may end up to 32 past
 * the count, and without the check the tail would read on from there, past the
 * slack, whenever codes outnumber the count by more than FWD3_SLACK. */
static inline const char *fwd3_native_name(void)
{
#if defined(__AVX2__)
    return "avx2_vpshufb";
#else
    return fwd_native_name();
#endif
}

#if defined(__AVX2__)
/* Both halves of a 32-value step in one register: lane 0 gathers codes 0..15 from
 * bytes 0..6 of the broadcast load, lane 1 codes 16..31 from bytes 6..12. Escape
 * ranks are lane-relative; lane 1 reads its escapes after lane 0's. */
static inline unsigned fwd3_count16(unsigned m)
{
#if defined(__POPCNT__)
    return (unsigned)__builtin_popcount(m);
#else
    m -= (m >> 1) & 0x5555u;
    m = (m & 0x3333u) + ((m >> 2) & 0x3333u);
    m = (m + (m >> 4)) & 0x0f0fu;
    return (m + (m >> 8)) & 0x1fu;
#endif
}

static inline __m256i fwd3_high32(const uint8_t *index, __m256i table, const uint8_t *escape,
                                  size_t *e)
{
    const __m256i lo = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3,
                                        6, 7, 6, 7, 6, 7, 7, 8, 7, 8, 7, 8, 8, 9, 8, 9);
    const __m256i hi = _mm256_setr_epi8(3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6,
                                        9, 10, 9, 10, 9, 10, 10, 11, 10, 11, 10, 11, 11, 12, 11, 12);
    const __m256i mul = _mm256_setr_epi16(8192, 1024, 128, 4096, 512, 64, 2048, 256,
                                          8192, 1024, 128, 4096, 512, 64, 2048, 256);
    const __m256i packed = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)index));
    const __m256i a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(packed, lo), mul), 13);
    const __m256i b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(packed, hi), mul), 13);
    const __m256i codes = _mm256_packus_epi16(a, b);
    const __m256i esc = _mm256_cmpeq_epi8(codes, _mm256_set1_epi8(7));
    const __m256i one = _mm256_and_si256(esc, _mm256_set1_epi8(1));
    __m256i s = _mm256_add_epi8(one, _mm256_slli_epi64(one, 8));
    s = _mm256_add_epi8(s, _mm256_slli_epi64(s, 16));
    s = _mm256_add_epi8(s, _mm256_slli_epi64(s, 32));
    s = _mm256_add_epi8(s, _mm256_shuffle_epi8(s, _mm256_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, 7, 7, 7, 7,
        -1, -1, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, 7, 7, 7, 7)));
    const unsigned mask = (unsigned)_mm256_movemask_epi8(esc);
    const size_t first = fwd3_count16(mask & 0xffffu), second = fwd3_count16(mask >> 16);
    const __m256i from = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(escape + *e))),
        _mm_loadu_si128((const __m128i *)(escape + *e + first)), 1);
    *e += first + second;
    return _mm256_blendv_epi8(_mm256_shuffle_epi8(table, codes),
                              _mm256_shuffle_epi8(from, _mm256_sub_epi8(s, one)), esc);
}
#endif

#if defined(__SSSE3__) && !defined(__AVX2__)
static inline __m128i fwd3_codes16(__m128i packed, __m128i lo, __m128i hi, __m128i mul)
{
    const __m128i a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(packed, lo), mul), 13);
    const __m128i b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(packed, hi), mul), 13);
    return _mm_packus_epi16(a, b);
}

static inline __m128i fwd3_high16(__m128i codes, __m128i table, const uint8_t *escape, size_t *e)
{
    const __m128i esc = _mm_cmpeq_epi8(codes, _mm_set1_epi8(7));
    const __m128i one = _mm_and_si128(esc, _mm_set1_epi8(1));
    __m128i s = _mm_add_epi8(one, _mm_slli_si128(one, 1));
    s = _mm_add_epi8(s, _mm_slli_si128(s, 2));
    s = _mm_add_epi8(s, _mm_slli_si128(s, 4));
    s = _mm_add_epi8(s, _mm_slli_si128(s, 8));
    const __m128i from = _mm_loadu_si128((const __m128i *)(escape + *e));
    const __m128i patched = _mm_shuffle_epi8(from, _mm_sub_epi8(s, one));
    *e += (unsigned)_mm_extract_epi16(s, 7) >> 8;
    return _mm_or_si128(_mm_and_si128(esc, patched),
                        _mm_andnot_si128(esc, _mm_shuffle_epi8(table, codes)));
}
#elif defined(__aarch64__)
static inline uint8x16_t fwd3_codes16(uint8x16_t packed, uint8x16_t lo, uint8x16_t hi)
{
    const int16x8_t shift = {0, -3, -6, -1, -4, -7, -2, -5};
    const uint16x8_t seven = vdupq_n_u16(7);
    const uint16x8_t a = vandq_u16(vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(packed, lo)), shift), seven);
    const uint16x8_t b = vandq_u16(vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(packed, hi)), shift), seven);
    return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}

static inline uint8x16_t fwd3_high16(uint8x16_t codes, uint8x16_t table, const uint8_t *escape, size_t *e)
{
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t esc = vceqq_u8(codes, vdupq_n_u8(7));
    const uint8x16_t one = vshrq_n_u8(esc, 7);
    uint8x16_t s = vaddq_u8(one, vextq_u8(zero, one, 15));
    s = vaddq_u8(s, vextq_u8(zero, s, 14));
    s = vaddq_u8(s, vextq_u8(zero, s, 12));
    s = vaddq_u8(s, vextq_u8(zero, s, 8));
    const uint8x16_t patched = vqtbl1q_u8(vld1q_u8(escape + *e), vsubq_u8(s, one));
    *e += vgetq_lane_u8(s, 15);
    return vbslq_u8(esc, patched, vqtbl1q_u8(table, codes));
}
#endif

static inline int fwd3_decode_native(const FwdView *v, uint8_t *out, size_t capacity)
{
    if (v->bits != 3 || capacity < v->raw_bytes || v->escape_readable < v->escapes) return -1;
    size_t i = 0, e = 0;
#if defined(__AVX2__)
    const size_t index_bytes = (3 * v->values + 7) / 8;
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)v->table));
    for (; i + 32 <= v->values && 3 * i / 8 + 16 <= index_bytes &&
           e + 32 <= v->escape_readable; i += 32) {
        const __m256i high = fwd3_high32(v->index + 3 * i / 8, table, v->escape, &e);
        if (e > v->escapes) return -1;
        const __m256i low = _mm256_loadu_si256((const __m256i *)(v->low + i));
        const __m256i first = _mm256_unpacklo_epi8(low, high), second = _mm256_unpackhi_epi8(low, high);
        _mm_storeu_si128((__m128i *)(out + 2*i), _mm256_castsi256_si128(first));
        _mm_storeu_si128((__m128i *)(out + 2*i + 16), _mm256_castsi256_si128(second));
        _mm_storeu_si128((__m128i *)(out + 2*i + 32), _mm256_extracti128_si256(first, 1));
        _mm_storeu_si128((__m128i *)(out + 2*i + 48), _mm256_extracti128_si256(second, 1));
    }
#elif defined(__SSSE3__)
    const size_t index_bytes = (3 * v->values + 7) / 8;
    const __m128i lo = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3);
    const __m128i hi = _mm_setr_epi8(3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6);
    const __m128i six = _mm_set1_epi8(6);
    const __m128i lo2 = _mm_add_epi8(lo, six), hi2 = _mm_add_epi8(hi, six);
    const __m128i mul = _mm_setr_epi16(8192, 1024, 128, 4096, 512, 64, 2048, 256);
    const __m128i table = _mm_loadu_si128((const __m128i *)v->table);
    for (; i + 32 <= v->values && 3 * i / 8 + 16 <= index_bytes &&
           e + 32 <= v->escape_readable; i += 32) {
        const __m128i packed = _mm_loadu_si128((const __m128i *)(v->index + 3 * i / 8));
        const __m128i hc = fwd3_high16(fwd3_codes16(packed, lo, hi, mul), table, v->escape, &e);
        const __m128i hd = fwd3_high16(fwd3_codes16(packed, lo2, hi2, mul), table, v->escape, &e);
        if (e > v->escapes) return -1;
        const __m128i lc = _mm_loadu_si128((const __m128i *)(v->low + i));
        const __m128i ld = _mm_loadu_si128((const __m128i *)(v->low + i + 16));
        _mm_storeu_si128((__m128i *)(out + 2*i), _mm_unpacklo_epi8(lc, hc));
        _mm_storeu_si128((__m128i *)(out + 2*i + 16), _mm_unpackhi_epi8(lc, hc));
        _mm_storeu_si128((__m128i *)(out + 2*i + 32), _mm_unpacklo_epi8(ld, hd));
        _mm_storeu_si128((__m128i *)(out + 2*i + 48), _mm_unpackhi_epi8(ld, hd));
    }
#elif defined(__aarch64__)
    const size_t index_bytes = (3 * v->values + 7) / 8;
    const uint8x16_t lo = {0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3};
    const uint8x16_t hi = {3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6};
    const uint8x16_t lo2 = vaddq_u8(lo, vdupq_n_u8(6)), hi2 = vaddq_u8(hi, vdupq_n_u8(6));
    const uint8x16_t table = vld1q_u8(v->table);
    for (; i + 32 <= v->values && 3 * i / 8 + 16 <= index_bytes &&
           e + 32 <= v->escape_readable; i += 32) {
        const uint8x16_t packed = vld1q_u8(v->index + 3 * i / 8);
        const uint8x16_t hc = fwd3_high16(fwd3_codes16(packed, lo, hi), table, v->escape, &e);
        const uint8x16_t hd = fwd3_high16(fwd3_codes16(packed, lo2, hi2), table, v->escape, &e);
        if (e > v->escapes) return -1;
        const uint8x16x2_t first = {{vld1q_u8(v->low + i), hc}};
        const uint8x16x2_t second = {{vld1q_u8(v->low + i + 16), hd}};
        vst2q_u8(out + 2*i, first);
        vst2q_u8(out + 2*i + 32, second);
    }
#endif
    for (; i < v->values; i++) {
        const unsigned code = fwd3_code(v->index, i);
        uint8_t high = v->table[code];
        if (code == 7) {
            if (e == v->escapes) return -1;
            high = v->escape[e++];
        }
        out[2*i] = v->low[i];
        out[2*i + 1] = high;
    }
    if (v->raw_bytes & 1) out[v->raw_bytes - 1] = v->low[v->values];
    return e == v->escapes ? 0 : -1;
}

/* Either format by magic; the decoders dispatch on the view's width. */
static inline int fwd_parse_any(FwdView *v, const uint8_t *data, size_t length)
{
    if (length >= 4 && !memcmp(data, "FD3B", 4)) return fwd3_parse(v, data, length);
    return fwd_parse(v, data, length);
}

static inline int fwd_decode_any_scalar(const FwdView *v, uint8_t *out, size_t capacity)
{
    return v->bits == 3 ? fwd3_decode_scalar(v, out, capacity) : fwd_decode_scalar(v, out, capacity);
}

static inline int fwd_decode_any_native(const FwdView *v, uint8_t *out, size_t capacity)
{
    return v->bits == 3 ? fwd3_decode_native(v, out, capacity) : fwd_decode_native(v, out, capacity);
}

/* ------------------------------------------------------------------ FDRX ---- */

typedef struct {
    const uint8_t *entry;
    size_t rows, cols, group, entries;
} FwdRows;

/* Escape codes among values [first, last) of a whole view's index plane. */
static inline size_t fwd_count_escapes(const FwdView *v, size_t first, size_t last)
{
    size_t n = 0;
    for (size_t i = first; i < last; i++)
        n += v->bits == 3 ? fwd3_code(v->index, i) == 7
                          : ((v->index[i / 2] >> (4 * (i & 1))) & 15) == 15;
    return n;
}

static inline int fwd_rows_parse(FwdRows *x, const uint8_t *data, size_t length, const FwdView *whole)
{
    if (length < 16 || memcmp(data, "FDRX", 4) || (whole->bits != 3 && whole->bits != 4) ||
        (whole->raw_bytes & 1)) return -1;
    const size_t rows = fwd_u32(data + 4), cols = fwd_u32(data + 8), group = fwd_u32(data + 12);
    if (!cols || !group || rows > whole->values / cols || rows * cols != whole->values ||
        (cols * whole->bits) % 8) return -1;
    /* Not (rows + group - 1) / group: with a 32-bit size_t and group near 2^32 that
     * sum wraps, a zero-entry index would parse and its checkpoint be read past it. */
    const size_t entries = rows / group + (rows % group != 0);
    if (length != 16 + 4 * entries) return -1;
    size_t previous = 0;
    for (size_t j = 0; j < entries; j++) {
        const size_t at = fwd_u32(data + 16 + 4 * j);
        /* Monotone, starts at zero, and a group cannot hold more escapes than values. */
        if ((!j && at) || at < previous || at > whole->escapes ||
            (j && at - previous > group * cols)) return -1;
        previous = at;
    }
    if (entries && whole->escapes - previous > (rows - (entries - 1) * group) * cols) return -1;
    x->entry = data + 16;
    x->rows = rows;
    x->cols = cols;
    x->group = group;
    x->entries = entries;
    return 0;
}

static inline size_t fwd_rows_escapes_before(const FwdRows *x, const FwdView *whole, size_t row)
{
    if (row == x->rows) return whole->escapes;
    const size_t g = row / x->group;
    return fwd_u32(x->entry + 4 * g) +
           fwd_count_escapes(whole, g * x->group * x->cols, row * x->cols);
}

/* Rows [first, last) as an ordinary view: index, low and escape slices only. */
static inline int fwd_rows_view(const FwdRows *x, const FwdView *whole, size_t first, size_t last,
                         FwdView *out)
{
    if (first > last || last > x->rows) return -1;
    const size_t e0 = fwd_rows_escapes_before(x, whole, first);
    const size_t e1 = fwd_rows_escapes_before(x, whole, last);
    if (e1 < e0 || e1 > whole->escapes) return -1;
    *out = *whole;
    out->index = whole->index + first * x->cols * whole->bits / 8;
    out->low = whole->low + first * x->cols;
    out->escape = whole->escape + e0;
    out->values = (last - first) * x->cols;
    out->raw_bytes = 2 * out->values;
    out->escapes = e1 - e0;
    out->escape_readable = whole->escape_readable - e0;
    return 0;
}
#endif
