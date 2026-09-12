/* Optional local bridge to tools/remote_model.py. No HTTP or model math in C.
 * Each request has its own socket, so OpenMP expert reads never share a stream.
 * All integers on the wire are little endian; UINT64_MAX is a hard failure.
 */
#ifndef K3_REMOTE_H
#define K3_REMOTE_H

/* Include k3_st.h before this header: k3_remote_request takes a K3St. The libc
 * headers below are declared here rather than inherited from whichever .c file
 * includes this one, so the order of the remaining includes cannot matter. */
#include <errno.h>
#include <stdint.h>   /* uint64_t, INT64_MAX */
#include <stdio.h>    /* fprintf, stderr */
#include <string.h>   /* memcpy, memset, strlen, strrchr, strerror */

#define K3_REMOTE_CHUNK (8 << 20)

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>   /* close */

static void k3_remote_put64(unsigned char *p, uint64_t v)
{
    for (int i = 0; i < 8; i++) { p[i] = (unsigned char)v; v >>= 8; }
}

static int k3_remote_send(int fd, const void *buf, size_t n)
{
    const unsigned char *p = (const unsigned char *)buf;
    while (n) {
#ifdef MSG_NOSIGNAL
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
#else
        ssize_t r = send(fd, p, n, 0);
#endif
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static int k3_remote_recv(int fd, void *buf, size_t n)
{
    unsigned char *p = (unsigned char *)buf;
    while (n) {
        ssize_t r = recv(fd, p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}
#endif

/* nbytes == 0 queries the logical shard size. Never read a placeholder's holes. */
static int64_t k3_remote_request(const K3St *s, int shard, int64_t off,
                                 int64_t nbytes, void *buf)
{
#ifdef _WIN32
    (void)s; (void)shard; (void)off; (void)nbytes; (void)buf;
    return -1; /* Local checkpoints still work on Windows; remote mode needs WSL. */
#else
    if (shard < 0 || shard >= s->nshard || off < 0 || nbytes < 0 ||
        nbytes > K3_REMOTE_CHUNK || !s->remote_socket) return -1;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    if (strlen(s->remote_socket) >= sizeof addr.sun_path) return -1;
    memcpy(addr.sun_path, s->remote_socket, strlen(s->remote_socket) + 1);

    const char *name = strrchr(s->path[shard], '/');
    name = name ? name + 1 : s->path[shard];
    size_t len = strlen(name);
    if (len == 0 || len > 255) return -1;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct timeval timeout = {600, 0};
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout) ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout)) {
        close(fd); return -1;
    }
#ifdef SO_NOSIGPIPE
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one)) {
        close(fd); return -1;
    }
#endif
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        fprintf(stderr, "k3_remote: bridge unavailable: %s\n", strerror(errno));
        close(fd); return -1;
    }

    unsigned char req[84], reply[8];
    memcpy(req, s->remote_id, 64);
    for (int i = 0; i < 4; i++) req[64 + i] = (unsigned char)(len >> (8 * i));
    k3_remote_put64(req + 68, (uint64_t)off);
    k3_remote_put64(req + 76, (uint64_t)nbytes);
    if (k3_remote_send(fd, req, sizeof req) ||
        k3_remote_send(fd, name, len) ||
        k3_remote_recv(fd, reply, sizeof reply)) {
        close(fd); return -1;
    }
    uint64_t got = 0;
    for (int i = 7; i >= 0; i--) got = (got << 8) | reply[i];
    if (got > INT64_MAX || (nbytes && got != (uint64_t)nbytes) ||
        (nbytes && k3_remote_recv(fd, buf, (size_t)nbytes))) {
        fprintf(stderr, "k3_remote: failed range for %s at %lld (%lld bytes)\n",
                name, (long long)off, (long long)nbytes);
        close(fd); return -1;
    }
    close(fd);
    return (int64_t)got;
#endif
}

static int64_t k3_remote_read(const K3St *s, int shard, int64_t off,
                              int64_t nbytes, void *buf)
{
    if (off < 0 || nbytes < 0 || off > INT64_MAX - nbytes) return 0;
    int64_t done = 0;
    while (done < nbytes) {
        int64_t take = nbytes - done;
        if (take > K3_REMOTE_CHUNK) take = K3_REMOTE_CHUNK;
        if (k3_remote_request(s, shard, off + done, take,
                              (unsigned char *)buf + done) != take) return done;
        done += take;
    }
    return done;
}
#endif
