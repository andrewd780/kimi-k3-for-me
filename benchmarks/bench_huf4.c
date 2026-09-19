/* A/B decoder timing, input encoded independently by tools/bench_huf4.py.
 * All sizes are bounded before allocating; malformed data is a failed gate. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "huf4.h"

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint32_t u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    uint8_t hdr[284];
    if (fread(hdr, 1, sizeof hdr, f) != sizeof hdr || memcmp(hdr, "HF4B", 4)) return 1;
    const uint32_t n = u32(hdr + 4), single = u32(hdr + 8);
    size_t len[4], packed = single;
    if (!n || n > (16u << 20) || single > n) return 1;
    for (int j = 0; j < 4; j++) {
        len[j] = u32(hdr + 12 + 4*j);
        if (len[j] > n) return 1;
        packed += len[j];
    }
    const size_t low_bytes = (n + 1u) / 2;
    const size_t payload = packed + low_bytes + n;
    uint8_t *data = (uint8_t *)malloc(payload), *out = (uint8_t *)malloc(n);
    if (!data || !out) return 2;
    if (fread(data, 1, payload, f) != payload || fgetc(f) != EOF) return 1;
    fclose(f);
    K3HufTable table;
    if (k3_huf_build(&table, hdr + 28)) return 1;
    const uint8_t *lanes[4]; size_t off = single;
    for (int j = 0; j < 4; j++) { lanes[j] = data + off; off += len[j]; }
    const uint8_t *low = data + packed, *raw = low + low_bytes;
    printf("{\"raw_bytes\":%u,\"table_bytes\":%zu,\"arms\":[", n, sizeof table);
    for (int arm = 0; arm < 2; arm++) {
        printf("%s{\"name\":\"%s\",\"decoded_BF16_GBps_runs\":[", arm ? "," : "",
               arm ? "four_streams" : "shelved_single_stream");
        for (int run = 0; run < 3; run++) {
            const double start = now();
            unsigned rounds = 0;
            double elapsed;
            do {
                const int rc = arm ? huf4_decode(&table, lanes, len, low, out, n)
                    : k3_huf_decode_stripe(&table, data, single, low, (uint32_t)low_bytes, out, n);
                if (rc) return 1;
                rounds++;
                elapsed = now() - start;
            } while (elapsed < 0.2);
            if (memcmp(raw, out, n)) return 1;
            printf("%s%.9f", run ? "," : "", (double)n * rounds / elapsed / 1e9);
        }
        printf("]}");
    }
    printf("]}\n");
    free(data); free(out);
    return 0;
}
