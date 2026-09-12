/* Actual C decoder: concurrent disjoint random reads, boundaries and bad ranges. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "k3_portable_io.h"
#include <unistd.h>
#include "k3_zfile.h"

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    int plain = open(argv[1], O_RDONLY), packed = open(argv[2], O_RDONLY);
    if (plain < 0 || packed < 0) return 1;
    K3ZFile *z = NULL;
    if (k3_zopen(packed, &z)) { close(plain); close(packed); return 1; }
    int64_t size = (int64_t)lseek(plain, 0, SEEK_END);
    if (size < 0 || (uint64_t)size != z->raw_size || size > (64 << 20)) return 1;
    unsigned char *want = (unsigned char *)malloc((size_t)size + 1);
    unsigned char *got = (unsigned char *)malloc((size_t)size + 1);
    if (!want || !got) return 1;
    int bad = pread(plain, want, (size_t)size, 0) != size;
    if (k3_zread(z, got, size, 0) != size || memcmp(want, got, (size_t)size)) bad++;
    free(got);
    /* Each worker has private output and shares only the immutable index and fd.
     * A seek()+read() implementation, instead of pread(), fails this under threads. */
#ifdef _OPENMP
#pragma omp parallel for reduction(+:bad) schedule(dynamic, 1)
#endif
    for (int i = 0; i < 96; i++) {
        size_t off = size ? (size_t)(((uint64_t)i * 2654435761u) % size) : 0;
        size_t n = (size_t)size - off;
        if (n > 70001) n = 70001;
        unsigned char *out = (unsigned char *)malloc(n + 1);
        if (!out) { bad++; continue; }
        if (k3_zread(z, out, (int64_t)n, (int64_t)off) != (int64_t)n ||
            memcmp(want + off, out, n)) bad++;
        free(out);
    }
    unsigned char sentinel = 123;
    if (k3_zread(z, &sentinel, 1, size) || k3_zread(z, &sentinel, 1, -1) ||
        k3_zread(z, &sentinel, INT64_MAX, 1) ||
        k3_zread(z, &sentinel, -1, 0) || sentinel != 123) bad++;
    free(want); k3_zfree(z); close(plain); close(packed);
    printf("native compressed ranges: %s\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
