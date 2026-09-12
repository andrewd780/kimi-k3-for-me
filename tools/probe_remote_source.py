#!/usr/bin/env python3
"""Probe the public K3 origin with bounded reads; never download a full shard."""
from __future__ import annotations

import json
import re
import struct
import sys

from remote_model import MAX_HEADER, decode, digest, read_range, read_small, validate_header


def main():
    repo = "moonshotai/Kimi-K3"
    info = decode(read_small("https://huggingface.co/api/models/%s" % repo))
    revision = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub did not return an immutable commit")
    names = sorted(item["rfilename"] for item in info["siblings"]
                   if re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", item["rfilename"]))
    if not names:
        raise ValueError("no safetensors shards at origin")
    name = names[0]
    url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, name)
    prefix, total = read_range(url, 0, 8)
    length = struct.unpack("<Q", prefix)[0]
    if not 0 < length <= MAX_HEADER or length + 8 >= total:
        raise ValueError("impossible shard header")
    header, _ = read_range(url, 8, length, total)
    tensors = validate_header(prefix + header, total)
    # A small payload read verifies that only metadata is not being served.
    sample, _ = read_range(url, 8 + length, min(4096, total - length - 8), total)
    print(json.dumps({
        "repo": repo, "revision": revision, "shard": name, "shard_bytes": total,
        "tensor_entries": len(tensors) - int("__metadata__" in tensors),
        "downloaded_bytes": 8 + length + len(sample), "sample_sha256": digest(sample),
        "result": "header and payload range accepted; full-model inference not tested",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
