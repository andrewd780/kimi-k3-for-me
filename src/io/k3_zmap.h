/* Internal implementation for k3_zfile.h, included only with K3_WITH_ZSTD.
 * K3ZMAP1 preserves logical offsets while coding only selected tensor extents.
 * Raw spans go straight from pread into the caller's buffer; no decoder or
 * whole-span allocation. All metadata is immutable after open. */
#ifndef K3_ZMAP_H
#define K3_ZMAP_H

#define K3_ZMAP_ENTRY 32
#define K3_ZMAP_RAW 2

static inline uint64_t k3_zmap_hash(const unsigned char *p, size_t n, uint64_t h)
{
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static inline int k3_zmap_open(int fd, const unsigned char *h, uint64_t physical,
                              K3ZFile **out)
{
    uint64_t raw_size = k3_zle(h + 8, 8);
    uint32_t block = (uint32_t)k3_zle(h + 16, 4);
    uint32_t count = (uint32_t)k3_zle(h + 20, 4);
    if (raw_size > INT64_MAX || block < (64u << 10) || block > K3_ZMAX_BLOCK ||
        (block & (block - 1)) || count > K3_ZMAX_COUNT) goto bad_header;
    uint64_t table_size = (uint64_t)count * K3_ZMAP_ENTRY;
    uint64_t cursor = K3_ZHEADER + table_size;
    if (cursor > physical) goto bad_header;
    K3ZFile *z = (K3ZFile *)calloc(1, sizeof *z);
    if (!z) return -1;
    z->fd = fd; z->raw_size = raw_size; z->block_size = block;
    z->count = count; z->mapped = 1;
    memcpy(z->id, h + 24, 16);
    z->extent = (K3ZExtent *)calloc(count ? count : 1, sizeof *z->extent);
    unsigned char *table = (unsigned char *)malloc(table_size ? (size_t)table_size : 1);
    if (!z->extent || !table) { free(table); k3_zfree(z); return -1; }
    int bad = k3_zpread(fd, table, (size_t)table_size, K3_ZHEADER);
    uint64_t checksum = k3_zmap_hash(h, 40, UINT64_C(14695981039346656037));
    if (!bad && k3_zmap_hash(table, (size_t)table_size, checksum) != k3_zle(h + 40, 8))
        bad = 1;
    uint64_t logical = 0;
    size_t bound = ZSTD_compressBound((size_t)block + K3_ZPREFIX);
    for (uint32_t i = 0; !bad && i < count; i++) {
        const unsigned char *p = table + (size_t)i * K3_ZMAP_ENTRY;
        K3ZExtent *e = &z->extent[i];
        e->end = k3_zle(p, 8); e->off = k3_zle(p + 8, 8);
        e->size = k3_zle(p + 16, 8); e->flags = (uint32_t)k3_zle(p + 24, 4);
        if (k3_zle(p + 28, 4) || e->end <= logical || e->end > raw_size ||
            e->off != cursor || e->size > physical - cursor ||
            (e->flags != 0 && e->flags != K3_ZMAP_RAW)) { bad = 1; break; }
        uint64_t length = e->end - logical;
        if ((e->flags == K3_ZMAP_RAW && e->size != length) ||
            (e->flags == 0 && (length > block || e->size < 9 || e->size > bound))) {
            bad = 1; break;
        }
        logical = e->end; cursor += e->size;
    }
    free(table);
    if (bad || logical != raw_size || cursor != physical) {
        k3_zfree(z); goto bad_header;
    }
    *out = z;
    return 0;
bad_header:
    fprintf(stderr, "k3z: invalid selective archive header or extent index\n");
    return -1;
}

static inline int64_t k3_zmap_read(const K3ZFile *z, void *dst, int64_t n, int64_t off)
{
    /* Caller validated the range, including n <= SIZE_MAX. Find the first covering
     * extent once; subsequent extents are contiguous both logically and physically. */
    uint32_t lo = 0, hi = z->count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (z->extent[mid].end <= (uint64_t)off) lo = mid + 1;
        else hi = mid;
    }
    unsigned char *packed = NULL, *raw = NULL;
    int64_t done = 0;
    for (uint32_t index = lo; done < n && index < z->count; index++) {
        const K3ZExtent *e = &z->extent[index];
        uint64_t start = index ? z->extent[index - 1].end : 0;
        uint64_t pos = (uint64_t)(off + done), within = pos - start;
        uint64_t take = e->end - pos;
        if (take > (uint64_t)(n - done)) take = (uint64_t)(n - done);
        unsigned char *outp = (unsigned char *)dst + (size_t)done;
        if (e->flags == K3_ZMAP_RAW) {
            if (k3_zpread(z->fd, outp, (size_t)take, e->off + within)) break;
        } else {
            if (!packed) {
                packed = (unsigned char *)malloc(
                    ZSTD_compressBound((size_t)z->block_size + K3_ZPREFIX));
                raw = (unsigned char *)malloc((size_t)z->block_size + K3_ZPREFIX);
                if (!packed || !raw) break;
            }
            size_t length = (size_t)(e->end - start);
            if (k3_zpread(z->fd, packed, (size_t)e->size, e->off) ||
                memcmp(packed, "\x28\xb5\x2f\xfd", 4) || !(packed[4] & 4) ||
                ZSTD_findFrameCompressedSize(packed, (size_t)e->size) != e->size) break;
            size_t got = ZSTD_decompress(raw, length + K3_ZPREFIX, packed, (size_t)e->size);
            if (ZSTD_isError(got) || got != length + K3_ZPREFIX ||
                memcmp(raw, z->id, 16) || k3_zle(raw + 16, 8) != index ||
                k3_zle(raw + 24, 4) || k3_zle(raw + 28, 4)) break;
            memcpy(outp, raw + K3_ZPREFIX + (size_t)within, (size_t)take);
        }
        done += (int64_t)take;
    }
    free(raw); free(packed);
    if (done != n) fprintf(stderr, "k3z: corrupt or unreadable extent at byte %lld\n",
                            (long long)(off + done));
    return done;
}
#endif
