/* Independent Python encoding, byte-exact C reconstruction, then optional timing.
 * Units and 0.2-second repeat windows match bench_huf4; setup/I/O are untimed. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fixed_dictionary.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint8_t *read_file(const char *path, size_t *length)
{
    FILE *f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END)) exit(2);
    const long size = ftell(f);
    if (size < 0 || size > (40L << 20) || fseek(f, 0, SEEK_SET)) exit(2);
    *length = (size_t)size;
    uint8_t *p = (uint8_t *)malloc(*length + 1);
    if (!p || fread(p, 1, *length, f) != *length || fgetc(f) != EOF) exit(2);
    fclose(f);
    return p;
}

int main(int argc, char **argv)
{
    if (argc != 4 || (strcmp(argv[3], "verify") && strcmp(argv[3], "time"))) return 2;
    size_t packed_bytes, raw_bytes;
    uint8_t *packed = read_file(argv[1], &packed_bytes);
    uint8_t *raw = read_file(argv[2], &raw_bytes);
    uint8_t *out = (uint8_t *)malloc(raw_bytes + 1);
    FwdView view;
    if (!out || fwd_parse(&view, packed, packed_bytes) || view.raw_bytes != raw_bytes) goto fail;
    if (fwd_decode_scalar(&view, out, raw_bytes) || memcmp(out, raw, raw_bytes)) goto fail;
    if (fwd_decode_native(&view, out, raw_bytes) || memcmp(out, raw, raw_bytes)) goto fail;
    printf("{\"native\":\"%s\",\"raw_bytes\":%zu,\"packed_bytes\":%zu,"
           "\"escape_values\":%zu,\"byte_exact\":true,\"arms\":[",
           fwd_native_name(), raw_bytes, packed_bytes, view.escapes);
    if (!strcmp(argv[3], "time")) {
        if (!raw_bytes) goto fail;
        for (int arm = 0; arm < 2; arm++) {
            printf("%s{\"name\":\"%s\",\"decoded_BF16_GBps_runs\":[", arm ? "," : "",
                   arm ? fwd_native_name() : "scalar_reference");
            for (int run = 0; run < 3; run++) {
                const double start = now();
                unsigned rounds = 0;
                double elapsed;
                do {
                    const int rc = arm ? fwd_decode_native(&view, out, raw_bytes)
                                       : fwd_decode_scalar(&view, out, raw_bytes);
                    if (rc) goto fail;
                    rounds++;
                    elapsed = now() - start;
                } while (elapsed < 0.2);
                if (memcmp(out, raw, raw_bytes)) goto fail;
                printf("%s%.9f", run ? "," : "", (double)raw_bytes * rounds / elapsed / 1e9);
            }
            printf("]}");
        }
    }
    printf("]}\n");
    free(out); free(raw); free(packed);
    return 0;
fail:
    free(out); free(raw); free(packed);
    return 1;
}
