#include <stdio.h>
#include "../../benchmarks/huf4.h"

int main(void)
{
    uint8_t lengths[256]; memset(lengths, 8, sizeof lengths);
    K3HufTable table;
    if (k3_huf_build(&table, lengths)) return 1;
    uint32_t pairs[4096]; huf_pairs(pairs, &table);
    uint8_t lanes[4][128], low[512], raw[1024], out[1024];
    const uint8_t *src[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
    for (size_t n = 1; n <= 1024; n++) {
        size_t len[4] = {0};
        for (size_t i = 0; i < n; i++) raw[i] = (uint8_t)(i * 73 + n);
        for (size_t i = 0; i < (n + 1)/2; i++) low[i] = raw[2*i];
        for (size_t i = 0; i < n/2; i++) lanes[i%4][len[i%4]++] = raw[2*i+1];
        if (huf4_decode(&table, pairs, src, len, low, out, n) || memcmp(raw, out, n)) return 1;
        if (len[0]) {
            len[0]--;
            if (!huf4_decode(&table, pairs, src, len, low, out, n)) return 1;
        }
    }
    memset(lengths, 0, sizeof lengths);
    for (int i = 0; i < 4; i++) lengths[i] = 2;
    if (k3_huf_build(&table, lengths)) return 1;
    huf_pairs(pairs, &table);
    for (size_t n = 1; n <= 1024; n++) {
        size_t len[4] = {0}; memset(lanes, 0, sizeof lanes);
        for (size_t i = 0; i < (n + 1)/2; i++) raw[2*i] = low[i] = (uint8_t)i;
        for (size_t i = 0; i < n/2; i++) {
            const size_t lane = i % 4, index = i / 4;
            const uint8_t symbol = (uint8_t)((i / 5) % 4);
            raw[2*i+1] = symbol;
            lanes[lane][index / 4] |= (uint8_t)(symbol << (6 - 2 * (index % 4)));
            len[lane] = index / 4 + 1;
        }
        if (huf4_decode(&table, pairs, src, len, low, out, n) || memcmp(raw, out, n)) return 1;
    }
    memset(lengths, 0, sizeof lengths); lengths[0] = 1;
    if (k3_huf_build(&table, lengths)) return 1;
    huf_pairs(pairs, &table);
    lanes[0][0] = 1; /* a decoded zero followed by nonzero padding is malformed */
    size_t len[4] = {1, 0, 0, 0};
    if (!huf4_decode(&table, pairs, src, len, low, out, 2)) return 1;
    memset(lengths, 1, sizeof lengths);
    if (!k3_huf_build(&table, lengths)) return 1;
    /* Also exercise the historical baseline on an independently specified stream. */
    memset(lengths, 8, sizeof lengths);
    if (k3_huf_build(&table, lengths) ||
        k3_huf_decode_stripe(&table, (const uint8_t *)"\x81", 1,
                             (const uint8_t *)"\x12", 1, out, 2) ||
        out[0] != 0x12 || out[1] != 0x81) return 1;
    puts("HUF4: all byte values, 1024 lengths, odd tails, truncation, padding and Kraft gates pass");
    return 0;
}
