/* Compare actual local and socket-backed read paths, including float widening. */
#define _POSIX_C_SOURCE 200809L
#include "k3_portable_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "k3_st.h"

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    K3St local, remote;
    if (k3_st_open(&local, argv[1]) || k3_st_open(&remote, argv[2])) return 1;
    if (local.nt != remote.nt || !remote.remote_socket || local.remote_socket) return 1;
    int bad = 0;
    for (int i = 0; i < local.nt; i++) {
        const K3Tensor *a = &local.t[i];
        const K3Tensor *b = k3_st_find(&remote, a->name);
        if (!b || b->nbytes != a->nbytes || b->dtype != a->dtype) return 1;
        size_t n = (size_t)a->nbytes;
        unsigned char *want = (unsigned char *)malloc(n ? n : 1);
        unsigned char *got = (unsigned char *)malloc(n ? n : 1);
        if (!want || !got) return 1;
        if (k3_st_read(&local, a, want) != a->nbytes ||
            k3_st_read(&remote, b, got) != b->nbytes || memcmp(want, got, n)) bad++;

        void *aligned = NULL;
        if (posix_memalign(&aligned, K3_ST_ALIGN, n + 2 * K3_ST_ALIGN)) return 1;
        int64_t pad = -1;
        if (k3_st_read_aligned(&remote, b->shard, b->off, b->nbytes,
                               aligned, (int64_t)n + 2 * K3_ST_ALIGN, &pad) != b->nbytes ||
            pad != 0 || memcmp(want, aligned, n)) bad++;
        if (n && k3_st_read_aligned(&remote, b->shard, b->off, b->nbytes,
                                    aligned, (int64_t)n - 1, &pad) != 0) bad++;
        k3_aligned_free(aligned);

        size_t nf = (size_t)k3_st_numel(a);
        float *wf = (float *)malloc((nf ? nf : 1) * sizeof(float));
        float *gf = (float *)malloc((nf ? nf : 1) * sizeof(float));
        if (!wf || !gf) return 1;
        if (k3_st_read_f32(&local, a, wf) != (int64_t)nf ||
            k3_st_read_f32(&remote, b, gf) != (int64_t)nf ||
            memcmp(wf, gf, nf * sizeof(float))) bad++;
        free(wf); free(gf); free(want); free(got);
    }
    K3Tensor broken = remote.t[0];
    broken.off = INT64_C(1) << 60;
    broken.nbytes = 1;
    unsigned char byte = 123;
    if (k3_st_read(&remote, &broken, &byte) != 0 || byte != 123) bad++;
    k3_st_close(&remote); k3_st_close(&local);
    printf("remote/local byte parity: %s\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
