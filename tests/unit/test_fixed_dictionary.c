/* Adversarial transport/decoder gates, run under ASan+UBSan on both CI ISAs. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../benchmarks/fixed_dictionary.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static void put32(uint8_t *p, size_t n)
{
    for (size_t j = 0; j < 4; j++) p[j] = (uint8_t)(n >> (8*j));
}

static size_t encode(uint8_t *p, const uint8_t *raw, size_t n)
{
    memset(p, 0, 32 + 2*n);
    memcpy(p, "FD4B", 4); put32(p + 4, n);
    for (size_t j = 0; j < 15; j++) p[16+j] = (uint8_t)(17*j + 3);
    size_t escapes = 0, indexes = (n/2 + 1)/2, low = (n+1)/2;
    for (size_t j = 0; j < low; j++) p[32+indexes+j] = raw[2*j];
    for (size_t j = 0; j < n/2; j++) {
        unsigned code = 15;
        for (unsigned k = 0; k < 15; k++) if (p[16+k] == raw[2*j+1]) code = k;
        p[32+j/2] |= (uint8_t)(code << (4*(j&1)));
        if (code == 15) p[32+indexes+low+escapes++] = raw[2*j+1];
    }
    put32(p + 8, escapes);
    return 32 + indexes + low + escapes;
}

static int exact(const uint8_t *packed, size_t length, const uint8_t *raw, size_t n)
{
    FwdView v;
    uint8_t *out = (uint8_t *)malloc(n + 1);
    CHECK(out);
    int ok = !fwd_parse(&v, packed, length) && v.raw_bytes == n;
    if (ok) ok = !fwd_decode_scalar(&v, out, n) && !memcmp(out, raw, n);
    if (ok) ok = !fwd_decode_native(&v, out, n) && !memcmp(out, raw, n);
    free(out);
    return ok;
}

/* FD3B, encoded bit by bit so it shares nothing with the decoder's extraction. */
static const uint8_t TABLE3[7] = {3, 20, 37, 54, 71, 88, 105};

static size_t fd3_size(size_t n)
{
    return 32 + (3 * (n / 2) + 7) / 8 + (n + 1) / 2 + n / 2 + FWD3_SLACK;
}

static size_t encode3(uint8_t *p, const uint8_t *raw, size_t n)
{
    const size_t values = n / 2, indexes = (3 * values + 7) / 8, low = (n + 1) / 2;
    memset(p, 0, fd3_size(n));
    memcpy(p, "FD3B", 4); put32(p + 4, n);
    memcpy(p + 16, TABLE3, 7);
    size_t escapes = 0;
    for (size_t j = 0; j < low; j++) p[32 + indexes + j] = raw[2*j];
    for (size_t j = 0; j < values; j++) {
        unsigned code = 7;
        for (unsigned k = 0; k < 7; k++) if (TABLE3[k] == raw[2*j+1]) code = k;
        for (unsigned b = 0; b < 3; b++)
            if (code >> b & 1) p[32 + (3*j + b) / 8] |= (uint8_t)(1u << ((3*j + b) % 8));
        if (code == 7) p[32 + indexes + low + escapes++] = raw[2*j+1];
    }
    put32(p + 8, escapes);
    return 32 + indexes + low + escapes + FWD3_SLACK;
}

static int exact3(const uint8_t *packed, size_t length, const uint8_t *raw, size_t n)
{
    FwdView v;
    uint8_t *out = (uint8_t *)malloc(n + 1);
    CHECK(out);
    int ok = !fwd3_parse(&v, packed, length) && v.raw_bytes == n;
    if (ok) ok = !fwd3_decode_scalar(&v, out, n) && !memcmp(out, raw, n);
    if (ok) ok = !fwd3_decode_native(&v, out, n) && !memcmp(out, raw, n);
    if (ok) ok = !fwd_decode_any_native(&v, out, n) && !memcmp(out, raw, n);
    /* The widths never mix: an FD3B view is not an FD4B stream. */
    if (ok) ok = fwd_decode_native(&v, out, n) && fwd_decode_scalar(&v, out, n);
    free(out);
    return ok;
}

static uint32_t rng = 2463534242u;
static uint32_t next_random(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

/* High bytes from the table with probability 1 - escape_per_mille/1000. */
static void skewed(uint8_t *raw, size_t n, unsigned escape_per_mille)
{
    for (size_t j = 0; j < n; j++) {
        const uint32_t r = next_random();
        raw[j] = (uint8_t)r;
        if (j & 1) raw[j] = (r >> 8) % 1000 < escape_per_mille ? (uint8_t)(200 + (r >> 20) % 50)
                                                              : TABLE3[(r >> 20) % 7];
    }
}

static int is_escape(unsigned bits, uint8_t high)
{
    const size_t entries = bits == 3 ? 7 : 15;
    for (size_t k = 0; k < entries; k++)
        if ((bits == 3 ? TABLE3[k] : (uint8_t)(17*k + 3)) == high) return 0;
    return 1;
}

/* FDRX from the raw bytes alone: escapes stored before row j*group. */
static size_t build_rows(uint8_t *x, const uint8_t *raw, unsigned bits, size_t rows, size_t cols,
                         size_t group)
{
    memcpy(x, "FDRX", 4); put32(x + 4, rows); put32(x + 8, cols); put32(x + 12, group);
    const size_t entries = (rows + group - 1) / group;
    size_t before = 0;
    for (size_t j = 0; j < entries; j++) {
        put32(x + 16 + 4*j, before);
        const size_t last = (j + 1) * group < rows ? (j + 1) * group : rows;
        for (size_t i = j * group * cols; i < last * cols; i++) before += is_escape(bits, raw[2*i+1]);
    }
    return 16 + 4 * entries;
}

/* Every [first, last) of the matrix through the row index, both decoders. */
static int rows_exact(const FwdView *whole, const uint8_t *index, size_t length, const uint8_t *raw,
                      size_t first, size_t last)
{
    FwdRows x;
    FwdView v;
    if (fwd_rows_parse(&x, index, length, whole) || fwd_rows_view(&x, whole, first, last, &v))
        return 0;
    const size_t n = 2 * (last - first) * x.cols;
    uint8_t *out = (uint8_t *)malloc(n + 1);
    CHECK(out);
    const uint8_t *want = raw + 2 * first * x.cols;
    int ok = v.raw_bytes == n && !fwd_decode_any_scalar(&v, out, n) && !memcmp(out, want, n);
    if (ok) ok = !fwd_decode_any_native(&v, out, n) && !memcmp(out, want, n);
    free(out);
    return ok;
}

static void row_ranges(unsigned bits, size_t rows, size_t cols, unsigned escape_per_mille)
{
    const size_t n = 2 * rows * cols;
    uint8_t *raw = (uint8_t *)malloc(n), *packed = (uint8_t *)malloc(fd3_size(n) + 2 * n + 64);
    uint8_t *index = (uint8_t *)malloc(16 + 4 * rows + 4);
    CHECK(raw && packed && index);
    skewed(raw, n, escape_per_mille);
    if (bits == 4) for (size_t j = 1; j < n; j += 2) if (raw[j] < 200) raw[j] = (uint8_t)(17*(raw[j] % 15) + 3);
    const size_t length = bits == 3 ? encode3(packed, raw, n) : encode(packed, raw, n);
    FwdView whole;
    CHECK(!fwd_parse_any(&whole, packed, length) && whole.bits == bits);
    const size_t groups[] = {1, 2, 3, 7, rows, rows + 5};
    for (size_t g = 0; g < sizeof groups / sizeof *groups; g++) {
        const size_t ilen = build_rows(index, raw, bits, rows, cols, groups[g]);
        for (size_t first = 0; first <= rows; first++)
            for (size_t last = first; last <= rows; last++) {
                /* Exhaustive for small matrices; long ones keep ranges touching the ends
                 * or starting and ending mid-matrix inside a group. */
                if (rows > 16 && first % 5 != 2 && last != rows && last - first > 3) continue;
                CHECK(rows_exact(&whole, index, ilen, raw, first, last));
            }
        FwdRows x;
        CHECK(!fwd_rows_parse(&x, index, ilen, &whole));
        /* Structural corruption is refused when the index is parsed. */
        uint8_t saved[4];
        memcpy(saved, index + 16, 4); put32(index + 16, 1);
        CHECK(fwd_rows_parse(&x, index, ilen, &whole));
        memcpy(index + 16, saved, 4);
        CHECK(fwd_rows_parse(&x, index, ilen - 1, &whole));
        CHECK(fwd_rows_parse(&x, index, ilen + 4, &whole));
        put32(index + 4, (uint32_t)rows + 1);
        CHECK(fwd_rows_parse(&x, index, ilen, &whole));
        put32(index + 4, (uint32_t)rows);
        if (x.entries > 2 && fwd_u32(index + 20) > 0) {
            /* A checkpoint that stays monotone but is off by one: every affected range
             * must fail its own escape count or its independent byte comparison. */
            memcpy(saved, index + 20, 4);
            put32(index + 20, fwd_u32(saved) - 1);
            CHECK(!fwd_rows_parse(&x, index, ilen, &whole));
            const size_t first = x.group + (x.group > 1), last = first + x.group;
            if (last <= rows) CHECK(!rows_exact(&whole, index, ilen, raw, first, last));
            memcpy(index + 20, saved, 4);
        }
    }
    FwdRows x;
    FwdView v;
    const size_t ilen = build_rows(index, raw, bits, rows, cols, 3);
    CHECK(!fwd_rows_parse(&x, index, ilen, &whole));
    CHECK(fwd_rows_view(&x, &whole, 2, 1, &v));
    CHECK(fwd_rows_view(&x, &whole, 0, rows + 1, &v));
    free(raw); free(packed); free(index);
}

static void fd3_suite(void)
{
    uint8_t raw[8193], packed[17000], copy[17001];
    /* Every length modulo the vector block: raw-byte, bit-packing and 32-value tails. */
    for (size_t n = 0; n <= sizeof raw; n++) {
        if (n > 131 && n != sizeof raw) continue;
        for (size_t j = 0; j < n; j++) raw[j] = (uint8_t)(j*37 + j/2);
        const size_t length = encode3(packed, raw, n);
        CHECK(exact3(packed, length, raw, n));
    }
    /* Independent golden bytes: codes 1, 6, escape -> bits 001 110 111 -> F1 01. */
    const uint8_t golden_raw[7] = {0xe1, 20, 0xe2, 105, 0xe3, 0xaa, 0xe4};
    uint8_t golden[32 + 2 + 4 + 1 + FWD3_SLACK] = {'F', 'D', '3', 'B', 7, 0, 0, 0, 1};
    memcpy(golden + 16, TABLE3, 7);
    golden[32] = 0xf1; golden[33] = 0x01;
    golden[34] = 0xe1; golden[35] = 0xe2; golden[36] = 0xe3; golden[37] = 0xe4; golden[38] = 0xaa;
    CHECK(encode3(packed, golden_raw, 7) == sizeof golden && !memcmp(packed, golden, sizeof golden));
    CHECK(exact3(golden, sizeof golden, golden_raw, 7));
    /* All high-byte values, all escapes, no escapes; realistic and bursty escapes. */
    for (size_t j = 0; j < sizeof raw/2; j++) { raw[2*j] = (uint8_t)(j*29); raw[2*j+1] = (uint8_t)j; }
    size_t length = encode3(packed, raw, sizeof raw);
    CHECK(exact3(packed, length, raw, sizeof raw));
    for (size_t j = 0; j < sizeof raw/2; j++) raw[2*j+1] = 255;
    length = encode3(packed, raw, sizeof raw);
    CHECK(exact3(packed, length, raw, sizeof raw));
    for (size_t j = 0; j < sizeof raw/2; j++) raw[2*j+1] = TABLE3[j % 7];
    length = encode3(packed, raw, sizeof raw);
    CHECK(exact3(packed, length, raw, sizeof raw));
    const unsigned rates[] = {1, 33, 250, 900};
    for (size_t r = 0; r < 4; r++) {
        skewed(raw, sizeof raw, rates[r]);
        length = encode3(packed, raw, sizeof raw);
        CHECK(exact3(packed, length, raw, sizeof raw));
    }
    for (size_t burst = 1; burst <= 40; burst++) {
        for (size_t j = 0; j < sizeof raw/2; j++)
            raw[2*j+1] = (j / burst) % 3 == 1 ? (uint8_t)(160 + j % 40) : TABLE3[j % 7];
        length = encode3(packed, raw, sizeof raw);
        CHECK(exact3(packed, length, raw, sizeof raw));
    }
    /* Bounds: every truncation, one excess byte, header fields, table and padding. */
    skewed(raw, sizeof raw, 33);
    length = encode3(packed, raw, sizeof raw);
    for (size_t n = 0; n < length; n++) {
        FwdView v;
        CHECK(fwd3_parse(&v, packed, n));
    }
    memcpy(copy, packed, length); copy[length] = 0;
    CHECK(!exact3(copy, length + 1, raw, sizeof raw));
    for (size_t at = 0; at < 16; at += 4) {
        memcpy(copy, packed, length); copy[at] ^= 255;
        CHECK(!exact3(copy, length, raw, sizeof raw));
    }
    memcpy(copy, packed, length); copy[17] = copy[16];
    CHECK(!exact3(copy, length, raw, sizeof raw));
    for (size_t at = 23; at < 32; at++) {
        memcpy(copy, packed, length); copy[at] = 1;
        CHECK(!exact3(copy, length, raw, sizeof raw));
    }
    memcpy(copy, packed, length); copy[length - 1] = 1;
    CHECK(!exact3(copy, length, raw, sizeof raw));
    /* Nonzero index padding bits: 3 values -> 9 bits used of 16. */
    raw[0] = 1; raw[1] = TABLE3[2]; raw[2] = 2; raw[3] = TABLE3[0]; raw[4] = 3; raw[5] = TABLE3[6];
    length = encode3(packed, raw, 6);
    CHECK(exact3(packed, length, raw, 6));
    packed[33] |= 0x80;
    CHECK(!exact3(packed, length, raw, 6));
    /* Escape underflow and surplus fail inside both decoders. */
    skewed(raw, sizeof raw, 33);
    length = encode3(packed, raw, sizeof raw);
    FwdView v;
    for (size_t j = 0; j < 4096; j++) {
        if (fwd3_code(packed + 32, j) == 7) continue;
        memcpy(copy, packed, length);
        for (unsigned b = 0; b < 3; b++) copy[32 + (3*j + b) / 8] |= (uint8_t)(1u << ((3*j + b) % 8));
        CHECK(!fwd3_parse(&v, copy, length));
        CHECK(fwd3_decode_scalar(&v, raw, sizeof raw));
        CHECK(fwd3_decode_native(&v, raw, sizeof raw));
        break;
    }
    memcpy(copy, packed, length); put32(copy + 8, fwd_u32(packed + 8) + 1);
    memmove(copy + length - FWD3_SLACK + 1, copy + length - FWD3_SLACK, FWD3_SLACK);
    copy[length - FWD3_SLACK] = 255;
    CHECK(!fwd3_parse(&v, copy, length + 1));
    CHECK(fwd3_decode_scalar(&v, raw, sizeof raw));
    CHECK(fwd3_decode_native(&v, raw, sizeof raw));
    /* Corrupt but well-formed payload fails the independent byte comparison. */
    skewed(raw, sizeof raw, 33);
    length = encode3(packed, raw, sizeof raw);
    CHECK(!fwd3_parse(&v, packed, length));
    memcpy(copy, packed, length); copy[v.low - packed + 100] ^= 1;
    CHECK(!exact3(copy, length, raw, sizeof raw));
    memcpy(copy, packed, length); copy[v.escape - packed + 3] ^= 1;
    CHECK(!exact3(copy, length, raw, sizeof raw));
    CHECK(fwd3_decode_native(&v, copy, sizeof raw - 1));
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "trip-asan")) {
        volatile size_t count = 1;
        volatile int *p = (volatile int *)malloc(count * sizeof(int));
        CHECK(p); p[1] = 7; free((void *)p); return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "trip-ubsan")) {
        volatile int n = INT_MAX;
        n += 1; return n == 0;
    }
    CHECK(argc == 1);
    uint8_t raw[8193], packed[17000], copy[17001];
    /* Every length modulo the vector block, both raw-byte and index odd tails. */
    for (size_t n = 0; n <= sizeof raw; n++) {
        if (n > 131 && n != sizeof raw) continue;
        for (size_t j = 0; j < n; j++) raw[j] = (uint8_t)(j*37 + j/2);
        const size_t length = encode(packed, raw, n);
        CHECK(exact(packed, length, raw, n));
    }
    /* All high-byte values, including sign, zeros, special-value bit patterns. */
    for (size_t j = 0; j < sizeof raw/2; j++) {
        raw[2*j] = (uint8_t)(j*29); raw[2*j+1] = (uint8_t)j;
    }
    size_t length = encode(packed, raw, sizeof raw);
    CHECK(exact(packed, length, raw, sizeof raw));
    for (size_t j = 0; j < sizeof raw/2; j++) raw[2*j+1] = 255;
    length = encode(packed, raw, sizeof raw);
    CHECK(exact(packed, length, raw, sizeof raw));
    /* No escapes; every dictionary slot exercises the vector lookup. */
    for (size_t j = 0; j < sizeof raw/2; j++) raw[2*j+1] = (uint8_t)(17*(j%15)+3);
    length = encode(packed, raw, sizeof raw);
    CHECK(exact(packed, length, raw, sizeof raw));
    for (size_t n = 0; n < length; n++) {
        FwdView v;
        CHECK(fwd_parse(&v, packed, n));
    }
    memcpy(copy, packed, length); copy[length] = 0;
    CHECK(!exact(copy, length+1, raw, sizeof raw));
    for (size_t at = 0; at < 16; at += 4) {
        memcpy(copy, packed, length); copy[at] ^= 255;
        CHECK(!exact(copy, length, raw, sizeof raw));
    }
    memcpy(copy, packed, length); copy[17] = copy[16];
    CHECK(!exact(copy, length, raw, sizeof raw));
    memcpy(copy, packed, length); copy[31] = 1;
    CHECK(!exact(copy, length, raw, sizeof raw));
    FwdView v;
    CHECK(!fwd_parse(&v, packed, length));
    CHECK(fwd_decode_native(&v, copy, sizeof raw - 1));
    CHECK(fwd_decode_scalar(&v, copy, sizeof raw - 1));
    /* Corrupt valid payload must fail the independent byte comparison. */
    memcpy(copy, packed, length); copy[32] ^= 1;
    CHECK(!exact(copy, length, raw, sizeof raw));
    memcpy(copy, packed, length); copy[v.low - packed] ^= 1;
    CHECK(!exact(copy, length, raw, sizeof raw));
    /* Escape underflow and surplus are decoder failures, not just bad lengths. */
    memcpy(copy, packed, length); copy[32] |= 15;
    CHECK(!fwd_parse(&v, copy, length));
    CHECK(fwd_decode_scalar(&v, raw, sizeof raw));
    CHECK(fwd_decode_native(&v, raw, sizeof raw));
    memcpy(copy, packed, length); put32(copy+8, 1); copy[length] = 255;
    CHECK(!fwd_parse(&v, copy, length+1));
    CHECK(fwd_decode_scalar(&v, raw, sizeof raw));
    CHECK(fwd_decode_native(&v, raw, sizeof raw));
    /* Single BF16 value: unused high nibble cannot hide garbage. */
    raw[0] = 128; raw[1] = 3;
    length = encode(packed, raw, 2);
    packed[32] |= 0x10;
    CHECK(!exact(packed, length, raw, 2));
    fd3_suite();
    /* Row index: exhaustive small matrices, K3-like widths, both widths, many groups. */
    row_ranges(4, 13, 64, 30);
    row_ranges(3, 13, 64, 33);
    row_ranges(3, 11, 128, 400);
    row_ranges(4, 40, 512, 5);
    row_ranges(3, 40, 512, 33);
    row_ranges(3, 6, 7168, 33);
    row_ranges(4, 6, 7168, 1);
    printf("PASS %s, FD3B %s: tails, all byte values, no/all escapes, bounds, corruption, "
           "escape counts; FD3B bit layout and branch-free escapes; FDRX mid-matrix row ranges\n",
           fwd_native_name(), fwd3_native_name());
    return 0;
}
