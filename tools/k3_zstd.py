"""Small ctypes binding to the native Zstandard library; no Python package needed."""
from __future__ import annotations

import ctypes
import ctypes.util
import os


class Zstd:
    def __init__(self):
        candidates = [os.environ.get("K3_ZSTD_LIBRARY"), ctypes.util.find_library("zstd"),
                      "/opt/homebrew/opt/zstd/lib/libzstd.dylib",
                      "/usr/local/opt/zstd/lib/libzstd.dylib"]
        self.lib = None
        for candidate in candidates:
            if candidate:
                try:
                    self.lib = ctypes.CDLL(candidate)
                    break
                except OSError:
                    pass
        if self.lib is None:
            raise OSError("libzstd is required: brew install zstd, or install libzstd-dev")
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        api = {
            "ZSTD_compressBound": (size, [size]),
            "ZSTD_createCCtx": (ptr, []),
            "ZSTD_freeCCtx": (size, [ptr]),
            "ZSTD_CCtx_setParameter": (size, [ptr, ctypes.c_int, ctypes.c_int]),
            "ZSTD_compress2": (size, [ptr, ptr, size, ptr, size]),
            "ZSTD_decompress": (size, [ptr, size, ptr, size]),
            "ZSTD_isError": (ctypes.c_uint, [size]),
            "ZSTD_getErrorName": (ctypes.c_char_p, [size]),
            "ZSTD_versionString": (ctypes.c_char_p, []),
        }
        for name, (restype, argtypes) in api.items():
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = restype, argtypes

    def check(self, result):
        if self.lib.ZSTD_isError(result):
            raise ValueError(self.lib.ZSTD_getErrorName(result).decode())
        return result

    def compress(self, data, level=3):
        output = ctypes.create_string_buffer(self.lib.ZSTD_compressBound(len(data)))
        ctx = self.lib.ZSTD_createCCtx()
        if not ctx:
            raise MemoryError("ZSTD_createCCtx")
        try:
            self.check(self.lib.ZSTD_CCtx_setParameter(ctx, 100, level))
            self.check(self.lib.ZSTD_CCtx_setParameter(ctx, 201, 1))  # frame checksum
            size = self.check(self.lib.ZSTD_compress2(
                ctx, output, len(output), data, len(data)))
            return output.raw[:size]
        finally:
            self.lib.ZSTD_freeCCtx(ctx)

    def decompress(self, data, size):
        output = ctypes.create_string_buffer(size)
        got = self.check(self.lib.ZSTD_decompress(output, size, data, len(data)))
        if got != size:
            raise ValueError("wrong decompressed length")
        return output.raw
