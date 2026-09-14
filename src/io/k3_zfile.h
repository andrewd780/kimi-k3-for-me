/* Independently checksummed Zstandard blocks. See docs/OFFLINE_STORAGE.md.
 * The descriptor belongs to the caller. Indexes are immutable; each read owns its
 * scratch buffers so parallel expert and trunk reads never share decoder state. */
#ifndef K3_ZFILE_H
#define K3_ZFILE_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* pread, lseek. k3_trunk.c includes <unistd.h> only inside
                       * #ifndef _WIN32, so a ZSTD=1 MinGW build reaches lseek() here
                       * with no declaration. k3_st.c already includes it unguarded
                       * and builds under MinGW, so doing the same is safe. */
#ifdef K3_WITH_ZSTD
#include <zstd.h>
#endif

#define K3_ZHEADER 48
#define K3_ZPREFIX 32
#define K3_ZMAX_BLOCK (8u << 20)
#define K3_ZMAX_COUNT (1u << 22)

typedef struct { uint64_t off; uint32_t size, flags; } K3ZEntry;
typedef struct { uint64_t end, off, size; uint32_t flags; } K3ZExtent;
typedef struct K3ZFile {
    int fd;
    uint64_t raw_size;
    uint32_t block_size, count;
    unsigned char id[16];
    K3ZEntry *entry;
    K3ZExtent *extent;   /* K3ZMAP1: raw spans plus independently coded scale spans */
    int mapped;
} K3ZFile;

static inline int k3_zsuffix(const char *path)
{
    size_t n = strlen(path);
    return n >= 4 && !strcmp(path + n - 4, ".k3z");
}

static inline void k3_zfree(K3ZFile *z)
{
    if (z) { free(z->entry); free(z->extent); free(z); }
}

#ifdef K3_WITH_ZSTD
static inline uint64_t k3_zle(const unsigned char *p, int n)
{
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static inline int k3_zpread(int fd, void *buf, size_t n, uint64_t off)
{
    size_t got = 0;
    while (got < n) {
        size_t take = n - got;
        if (take > (1u << 30)) take = 1u << 30;
        ssize_t r = pread(fd, (unsigned char *)buf + got, take,
                          (off_t)(off + got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}
#include "k3_zmap.h"
#endif

static inline int k3_zopen(int fd, K3ZFile **out)
{
    *out = NULL;
#ifndef K3_WITH_ZSTD
    (void)fd;
    fprintf(stderr, "k3z: compressed weights require a build with make ZSTD=1\n");
    return -1;
#else
    unsigned char h[K3_ZHEADER];
    int64_t physical = (int64_t)lseek(fd, 0, SEEK_END);
    if (physical < K3_ZHEADER || k3_zpread(fd, h, sizeof h, 0)) goto bad_header;
    if (!memcmp(h, "K3ZMAP1\0", 8)) return k3_zmap_open(fd, h, (uint64_t)physical, out);
    if (memcmp(h, "K3ZSTD1\0", 8) || k3_zle(h + 40, 8)) goto bad_header;
    uint64_t raw = k3_zle(h + 8, 8);
    uint32_t block = (uint32_t)k3_zle(h + 16, 4);
    uint32_t count = (uint32_t)k3_zle(h + 20, 4);
    if (raw > INT64_MAX || block < (64u << 10) || block > K3_ZMAX_BLOCK ||
        (block & (block - 1)) || count > K3_ZMAX_COUNT ||
        count != raw / block + (raw % block != 0)) goto bad_header;
    uint64_t cursor = K3_ZHEADER + (uint64_t)count * 16;
    if (cursor > (uint64_t)physical) goto bad_header;
    K3ZFile *z = (K3ZFile *)calloc(1, sizeof *z);
    if (!z) return -1;
    z->fd = fd; z->raw_size = raw; z->block_size = block; z->count = count;
    memcpy(z->id, h + 24, 16);
    z->entry = (K3ZEntry *)calloc(count ? count : 1, sizeof *z->entry);
    unsigned char *table = (unsigned char *)malloc(count ? (size_t)count * 16 : 1);
    if (!z->entry || !table) { free(table); k3_zfree(z); return -1; }
    int bad = k3_zpread(fd, table, (size_t)count * 16, K3_ZHEADER);
    size_t bound = ZSTD_compressBound((size_t)block + K3_ZPREFIX);
    for (uint32_t i = 0; !bad && i < count; i++) {
        const unsigned char *p = table + (size_t)i * 16;
        K3ZEntry *e = &z->entry[i];
        e->off = k3_zle(p, 8);
        e->size = (uint32_t)k3_zle(p + 8, 4);
        e->flags = (uint32_t)k3_zle(p + 12, 4);
        if (e->off != cursor || e->size < 9 || e->size > bound || e->flags > 1 ||
            e->size > (uint64_t)physical - cursor) bad = 1;
        else cursor += e->size;
    }
    free(table);
    if (bad || cursor != (uint64_t)physical) { k3_zfree(z); goto bad_header; }
    *out = z;
    return 0;
bad_header:
    fprintf(stderr, "k3z: invalid archive header or block index\n");
    return -1;
#endif
}

static inline int64_t k3_zread(const K3ZFile *z, void *dst, int64_t n, int64_t off)
{
#ifndef K3_WITH_ZSTD
    (void)z; (void)dst; (void)n; (void)off;
    return 0;
#else
    if (!z || off < 0 || n < 0 || (uint64_t)off > z->raw_size ||
        (uint64_t)n > z->raw_size - (uint64_t)off || (uint64_t)n > SIZE_MAX) return 0;
    if (!n) return 0;
    if (z->mapped) return k3_zmap_read(z, dst, n, off);
    size_t bound = ZSTD_compressBound((size_t)z->block_size + K3_ZPREFIX);
    unsigned char *packed = (unsigned char *)malloc(bound);
    unsigned char *raw = (unsigned char *)malloc((size_t)z->block_size + K3_ZPREFIX);
    if (!packed || !raw) { free(packed); free(raw); return 0; }
    int64_t done = 0;
    while (done < n) {
        uint64_t pos = (uint64_t)(off + done);
        uint32_t index = (uint32_t)(pos / z->block_size);
        const K3ZEntry *e = &z->entry[index];
        uint64_t remain = z->raw_size - (uint64_t)index * z->block_size;
        size_t size = remain < z->block_size ? (size_t)remain : z->block_size;
        if (k3_zpread(z->fd, packed, e->size, e->off) ||
            memcmp(packed, "\x28\xb5\x2f\xfd", 4) || !(packed[4] & 4) ||
            ZSTD_findFrameCompressedSize(packed, e->size) != e->size) break;
        size_t got = ZSTD_decompress(raw, size + K3_ZPREFIX, packed, e->size);
        /* Binding identity, position AND transform to the frame checksum prevents
         * valid frames (or an index flag bit flip) from silently changing weights. */
        if (ZSTD_isError(got) || got != size + K3_ZPREFIX ||
            memcmp(raw, z->id, 16) || k3_zle(raw + 16, 8) != index ||
            k3_zle(raw + 24, 4) != e->flags || k3_zle(raw + 28, 4)) break;
        size_t start = (size_t)(pos % z->block_size);
        size_t take = size - start;
        if ((uint64_t)take > (uint64_t)(n - done)) take = (size_t)(n - done);
        const unsigned char *data = raw + K3_ZPREFIX;
        unsigned char *outp = (unsigned char *)dst + (size_t)done;
        if (!e->flags) memcpy(outp, data + start, take);
        else {
            size_t half = (size + 1) / 2;
            for (size_t j = 0; j < take; j++) {
                size_t k = start + j;
                outp[j] = data[k / 2 + (k % 2 ? half : 0)];
            }
        }
        done += (int64_t)take;
    }
    free(raw); free(packed);
    if (done != n) fprintf(stderr, "k3z: corrupt or unreadable block at byte %lld\n",
                            (long long)(off + done));
    return done;
#endif
}
#endif
