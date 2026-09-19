/* Research-only Linux O_DIRECT comparison. No engine backend, SQPOLL, registered
 * buffers or liburing dependency. Uses a generated 64 MiB file, never model files.
 * Blocking pread workers sleep in the kernel; process CPU time is reported too.
 * API/lifetime contract: https://man7.org/linux/man-pages/man7/io_uring.7.html */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define BLOCK (1u << 20)
#define BLOCKS 64
#define MAX_QD 16

typedef struct {
    int fd;
    struct io_uring_params p;
    unsigned char *sq, *cq;
    struct io_uring_sqe *entries;
    size_t sq_size, cq_size, entries_size;
} Ring;

static unsigned get(unsigned char *base, unsigned off)
{
    return __atomic_load_n((unsigned *)(base + off), __ATOMIC_ACQUIRE);
}

static void put(unsigned char *base, unsigned off, unsigned value)
{
    __atomic_store_n((unsigned *)(base + off), value, __ATOMIC_RELEASE);
}

static void ring_close(Ring *r)
{
    /* Called only after complete waves or before any operation was submitted. */
    if (r->fd >= 0) close(r->fd);
    if (r->entries && r->entries != MAP_FAILED) munmap(r->entries, r->entries_size);
    if (r->cq && r->cq != MAP_FAILED && r->cq != r->sq) munmap(r->cq, r->cq_size);
    if (r->sq && r->sq != MAP_FAILED) munmap(r->sq, r->sq_size);
}

static int ring_open(Ring *r, unsigned qd)
{
    memset(r, 0, sizeof *r);
    r->fd = (int)syscall(__NR_io_uring_setup, qd, &r->p);
    if (r->fd < 0) return -1;
    r->sq_size = r->p.sq_off.array + r->p.sq_entries * sizeof(unsigned);
    r->cq_size = r->p.cq_off.cqes + r->p.cq_entries * sizeof(struct io_uring_cqe);
    if (r->p.features & IORING_FEAT_SINGLE_MMAP) {
        if (r->cq_size > r->sq_size) r->sq_size = r->cq_size;
        r->cq_size = r->sq_size;
    }
    r->sq = mmap(NULL, r->sq_size, PROT_READ | PROT_WRITE, MAP_SHARED, r->fd, IORING_OFF_SQ_RING);
    if (r->sq == MAP_FAILED) return -1;
    r->cq = r->p.features & IORING_FEAT_SINGLE_MMAP ? r->sq
        : mmap(NULL, r->cq_size, PROT_READ | PROT_WRITE, MAP_SHARED, r->fd, IORING_OFF_CQ_RING);
    if (r->cq == MAP_FAILED) return -1;
    r->entries_size = r->p.sq_entries * sizeof(struct io_uring_sqe);
    r->entries = mmap(NULL, r->entries_size, PROT_READ | PROT_WRITE, MAP_SHARED, r->fd, IORING_OFF_SQES);
    return r->entries == MAP_FAILED ? -1 : 0;
}

static int wave(Ring *r, int fd, unsigned char *buf, unsigned first, unsigned qd)
{
    const unsigned tail = get(r->sq, r->p.sq_off.tail), mask = get(r->sq, r->p.sq_off.ring_mask);
    if (tail - get(r->sq, r->p.sq_off.head) + qd > r->p.sq_entries) return -1;
    unsigned *array = (unsigned *)(r->sq + r->p.sq_off.array);
    for (unsigned i = 0; i < qd; i++) {
        const unsigned idx = (tail + i) & mask;
        struct io_uring_sqe *entry = &r->entries[idx];
        memset(entry, 0, sizeof *entry);
        entry->opcode = IORING_OP_READ; entry->fd = fd;
        entry->off = (uint64_t)(first + i) * BLOCK;
        entry->addr = (uint64_t)(uintptr_t)(buf + (size_t)i * BLOCK);
        entry->len = BLOCK; entry->user_data = i;
        array[idx] = idx;
    }
    put(r->sq, r->p.sq_off.tail, tail + qd);
    unsigned submitted = 0;
    while (submitted < qd) {
        const int n = (int)syscall(__NR_io_uring_enter, r->fd, qd - submitted, 0, 0, NULL, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        submitted += (unsigned)n;
    }
    unsigned completed = 0, seen = 0;
    const unsigned cmask = get(r->cq, r->p.cq_off.ring_mask);
    struct io_uring_cqe *cq = (struct io_uring_cqe *)(r->cq + r->p.cq_off.cqes);
    while (completed < qd) {
        unsigned head = get(r->cq, r->p.cq_off.head), end = get(r->cq, r->p.cq_off.tail);
        if (head == end) {
            const int rc = (int)syscall(__NR_io_uring_enter, r->fd, 0, 1, IORING_ENTER_GETEVENTS, NULL, 0);
            if (rc < 0 && errno != EINTR) return -1;
            continue;
        }
        while (head != end) {
            const struct io_uring_cqe *event = &cq[head & cmask];
            if (event->res != (int)BLOCK || event->user_data >= qd ||
                (seen & (1u << event->user_data))) return -1;
            seen |= 1u << event->user_data;
            completed++; head++;
        }
        put(r->cq, r->p.cq_off.head, head);
    }
    return completed == qd ? 0 : -1;
}

static double clock_s(clockid_t which)
{
    struct timespec t; clock_gettime(which, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void)
{
    char path[] = "/tmp/k3-uring-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 2;
    unsigned char *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, (size_t)MAX_QD * BLOCK)) return 2;
    for (unsigned block = 0; block < BLOCKS; block++) {
        for (unsigned j = 0; j < BLOCK; j++) buf[j] = (unsigned char)(block * 17 + j * 13);
        if (write(fd, buf, BLOCK) != BLOCK) return 2;
    }
    if (fsync(fd)) return 2;
    close(fd);
    fd = open(path, O_RDONLY | O_DIRECT); unlink(path);
    if (fd < 0) { printf("{\"status\":\"O_DIRECT_unavailable\",\"errno\":%d}\n", errno); free(buf); return 0; }
    Ring ring;
    if (ring_open(&ring, MAX_QD)) {
        const int why = errno; ring_close(&ring); close(fd); free(buf);
        printf("{\"status\":\"io_uring_unavailable\",\"errno\":%d}\n", why);
        return (why == EPERM || why == ENOSYS || why == EOPNOTSUPP || why == EACCES) ? 0 : 1;
    }
    printf("{\"status\":\"measured\",\"scope\":\"synthetic 64 MiB O_DIRECT file, no model or compute overlap\",\"runs\":[");
    int comma = 0;
    for (unsigned qd = 1; qd <= MAX_QD; qd *= 2) {
        for (int run = 0; run < 3; run++) {
            for (int order = 0; order < 2; order++) {
                const int arm = (run + order) % 2; /* alternate which arm goes first */
                const double start = clock_s(CLOCK_MONOTONIC), cpu = clock_s(CLOCK_PROCESS_CPUTIME_ID);
                int failed = 0;
                for (unsigned first = 0; first < BLOCKS; first += qd) {
                    if (arm) failed = wave(&ring, fd, buf, first, qd);
                    else {
#pragma omp parallel for num_threads(qd) reduction(|:failed)
                        for (unsigned i = 0; i < qd; i++) {
                            ssize_t n;
                            do { n = pread(fd, buf + (size_t)i * BLOCK, BLOCK, (off_t)(first + i) * BLOCK); }
                            while (n < 0 && errno == EINTR);
                            if (n != BLOCK) failed = 1;
                        }
                    }
                    /* A fatal syscall may leave requests in flight. Terminate without
                     * reusing/freeing their destination; process teardown owns them. */
                    if (failed) return 1;
                    /* Check every byte, including out-of-order CQE placement. Both
                     * arms pay the same check cost; this is not isolated SSD latency. */
                    for (unsigned i = 0; i < qd; i++)
                        for (unsigned j = 0; j < BLOCK; j++)
                            if (buf[(size_t)i * BLOCK + j] != (unsigned char)((first+i)*17 + j*13)) return 1;
                }
                const double elapsed = clock_s(CLOCK_MONOTONIC) - start;
                printf("%s{\"arm\":\"%s\",\"queue_depth\":%u,\"run\":%d,\"seconds\":%.9f,\"process_cpu_seconds\":%.9f}",
                       comma++ ? "," : "", arm ? "io_uring" : "pread_pool", qd, run+1, elapsed,
                       clock_s(CLOCK_PROCESS_CPUTIME_ID) - cpu);
            }
        }
    }
    puts("]}");
    ring_close(&ring); close(fd); free(buf);
    return 0;
}
