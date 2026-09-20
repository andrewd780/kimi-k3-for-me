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
    printf("PASS %s: tails, all byte values, no/all escapes, bounds, corruption, escape counts\n",
           fwd_native_name());
    return 0;
}
