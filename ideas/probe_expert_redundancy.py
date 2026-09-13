#!/usr/bin/env python3
"""probe_expert_redundancy.py - is a K3 expert mostly a shared base plus a small correction?

WHY THIS EXISTS
    Every route to a sub-200 GB checkpoint that does not simply delete parameters rests
    on one unmeasured claim: that the 896 experts in a layer are near-duplicates, so an
    expert can be stored as a shared base plus a low-rank delta.

        W_e  ~=  Base_layer + U_e V_e^T

    If the residual W_e - Base is low rank, rank-32 int8 deltas put the whole checkpoint
    at ~112 GB and cut per-token reads from 25.83 GB to 0.94 GB. If the residual is full
    rank, the entire idea is dead and nobody should spend another day on it.

    Nothing in this repository measures that. This does, on the REAL checkpoint, for
    well under a gigabyte of download, using the range-read path the project already
    ships (tools/remote_model.py). It answers one question and then stops.

WHAT IT REPORTS
    For each of w1/w3 (3072x3584) and w2 (3584x3072), over a sample of experts:

      spectrum of W_e            how low-rank an expert is on its own
      spectrum of W_e - Base     how low-rank the RESIDUAL is, which is what matters
      energy captured at rank r  the number that decides the design
      pairwise similarity        how alike the experts are before any factoring

    The verdict line at the end is the deliverable: the rank needed for 95% and 99% of
    residual energy, and the checkpoint size that rank implies.

READ THE VERDICT HONESTLY
    Energy captured is a WEIGHT-SPACE proxy. It is necessary, not sufficient: a matrix
    can lose 1% of its Frobenius energy and still move the model's output. A good
    spectrum here justifies building the container and measuring perplexity; it does not
    prove quality is preserved. A bad spectrum is conclusive in the other direction.

usage:
    python3 ideas/probe_expert_redundancy.py --layer 3 --experts 16
    python3 ideas/probe_expert_redundancy.py --shard-dir /path/to/shards --layer 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))

# OCP MX FP4. Low nibble is the EVEN element -- reversing this yields a matrix with the
# right values in the wrong places, where every statistic looks perfect. See k3.h.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

EXPERT_FMT = "language_model.model.layers.%d.block_sparse_moe.experts.%d.%s.%s"
REPO = "moonshotai/Kimi-K3"


# ----------------------------------------------------------------- sources --

class LocalShards:
    """Index every tensor across a directory of .safetensors shards."""

    def __init__(self, directory):
        self.index = {}
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(directory, fn)
            with open(path, "rb") as handle:
                n = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(n).decode("utf-8"))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                start, end = entry["data_offsets"]
                self.index[name] = (path, entry["shape"], 8 + n + start, end - start)

    def get_u8(self, name):
        path, shape, off, nbytes = self.index[name]
        with open(path, "rb") as handle:
            handle.seek(off)
            raw = handle.read(nbytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


class RemoteShards:
    """Byte-range reads against the pinned public checkpoint. Downloads only the
    tensors asked for -- one expert is 17.55 MB, not a 1.45 TB checkpoint."""

    def __init__(self):
        import remote_model

        self._read_range = remote_model.read_range
        self._validate = remote_model.validate_header

        info = remote_model.decode(
            remote_model.read_small("https://huggingface.co/api/models/%s" % REPO))
        self.revision = info["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Hub did not return an immutable commit")
        self.shards = sorted(item["rfilename"] for item in info["siblings"]
                             if re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors",
                                             item["rfilename"]))
        self.index = {}
        self._headers_read = 0

    def _url(self, shard):
        return "https://huggingface.co/%s/resolve/%s/%s" % (REPO, self.revision, shard)

    def _load_header(self, shard):
        url = self._url(shard)
        prefix, total = self._read_range(url, 0, 8)
        length = struct.unpack("<Q", prefix[:8])[0]
        header, _ = self._read_range(url, 8, length, total)
        tensors = self._validate(prefix + header, total)
        for name, entry in tensors.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            self.index[name] = (shard, entry["shape"], 8 + length + start, end - start)
        self._headers_read += 1

    def ensure(self, name):
        if name in self.index:
            return
        for shard in self.shards:
            if shard in getattr(self, "_done", set()):
                continue
            self._load_header(shard)
            self._done = getattr(self, "_done", set()) | {shard}
            if name in self.index:
                return
        raise KeyError(name)

    def get_u8(self, name):
        self.ensure(name)
        shard, shape, off, nbytes = self.index[name]
        raw, _ = self._read_range(self._url(shard), off, nbytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


# ------------------------------------------------------------------- maths --

def dequant(src, base):
    """MXFP4 -> float32. Mirrors tools/verify_real_layer.py, which was deliberately
    written in numpy so it shares no line of code with the C implementation."""
    packed = src.get_u8(base + ".weight_packed")
    scales = src.get_u8(base + ".weight_scale")
    rows, pcols = packed.shape
    out = np.empty((rows, 2 * pcols), dtype=np.float32)
    out[:, 0::2] = E2M1[packed & 0x0F]
    out[:, 1::2] = E2M1[packed >> 4]
    mult = np.where(scales == 255, 0.0,
                    np.exp2(scales.astype(np.int32) - 127).astype(np.float32))
    out *= np.repeat(mult, 32, axis=1)[:, : 2 * pcols]
    return out


def randomized_spectrum(matrix, max_rank, oversample=12, power=2, seed=0):
    """Top singular values via randomized range finding. A full SVD of 3072x3584 is
    minutes per matrix; this is seconds and is accurate in the leading singular values,
    which is the only part the energy question depends on."""
    rng = np.random.default_rng(seed)
    k = min(max_rank + oversample, min(matrix.shape))
    omega = rng.standard_normal((matrix.shape[1], k), dtype=np.float32)
    sample = matrix @ omega
    for _ in range(power):                      # power iteration sharpens the decay
        sample = matrix @ (matrix.T @ sample)
    q, _ = np.linalg.qr(sample)
    return np.linalg.svd(q.T @ matrix, compute_uv=False)


def energy_at(svals, total_energy, ranks):
    cumulative = np.cumsum(svals.astype(np.float64) ** 2)
    return {r: float(cumulative[min(r, len(cumulative)) - 1] / total_energy)
            for r in ranks}


def rank_for(svals, total_energy, threshold):
    cumulative = np.cumsum(svals.astype(np.float64) ** 2) / total_energy
    hit = np.searchsorted(cumulative, threshold) + 1
    return int(hit) if hit <= len(cumulative) else None


# -------------------------------------------------------------------- main --

RANKS = (8, 16, 32, 64, 128, 256)


def size_for_rank(rank, bits=8):
    """Total checkpoint GB if every expert is a shared base plus a rank-r delta.
    Constants from include/k3/k3.h; matches ideas/size_model.py."""
    experts, moe_inter, latent, layers = 82_432, 3072, 3584, 92
    active_params = 56_743_648_000
    deltas = experts * 3 * rank * (moe_inter + latent) * bits / 8
    bases = layers * 3 * moe_inter * latent * bits / 8
    trunk = active_params * 8 / 8
    return (deltas + bases + trunk) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=3, help="MoE layer to sample (1..92)")
    ap.add_argument("--experts", type=int, default=16, help="how many experts to pull")
    ap.add_argument("--first-expert", type=int, default=0)
    ap.add_argument("--shard-dir", help="read local shards instead of HTTP ranges")
    ap.add_argument("--max-rank", type=int, default=256)
    ap.add_argument("--matrices", default="w1,w2,w3")
    ap.add_argument("--json", help="write the findings here")
    args = ap.parse_args()

    src = LocalShards(args.shard_dir) if args.shard_dir else RemoteShards()
    ids = list(range(args.first_expert, args.first_expert + args.experts))

    print(f"layer {args.layer}, experts {ids[0]}..{ids[-1]} "
          f"({len(ids)} of 896), source: "
          f"{'local shards' if args.shard_dir else 'HTTP range reads'}")
    print(f"downloading ~{len(ids) * 17.55 / 1000:.2f} GB\n")

    findings = {"layer": args.layer, "experts": ids, "matrices": {}}

    for which in args.matrices.split(","):
        print(f"=== {which} " + "=" * 56)
        mats = []
        for e in ids:
            base = EXPERT_FMT % (args.layer, e, which, "")
            mats.append(dequant(src, base.rstrip(".")))
            print(f"  loaded expert {e}", end="\r", flush=True)
        stack = np.stack(mats)
        del mats
        shared = stack.mean(axis=0)
        print(" " * 40, end="\r")
        print(f"  shape {stack.shape[1]}x{stack.shape[2]}, {len(ids)} experts")

        # How alike are they before any factoring?
        flat = stack.reshape(len(ids), -1)
        norms = np.linalg.norm(flat, axis=1)
        sims = (flat @ flat.T) / np.outer(norms, norms)
        off = sims[~np.eye(len(ids), dtype=bool)]
        print(f"  pairwise cosine similarity: mean {off.mean():+.4f}  "
              f"min {off.min():+.4f}  max {off.max():+.4f}")
        # rho = ||mean(W_i)||^2_F / mean(||W_i||^2_F). This single scalar decides the
        # whole design: the delta is the UNCORRELATED part of an expert, and total
        # relative error is sqrt((1 - rho)(1 - E_delta(r))). Because a flat residual
        # spectrum makes E_delta(r) tiny at any affordable r, rho dominates and rank
        # barely matters -- which is why the decision rule below is on rho, not on rank.
        rho = float(np.linalg.norm(shared) ** 2 /
                    np.mean(np.linalg.norm(flat, axis=1) ** 2))
        base_share = rho
        findings.setdefault("rho", {})[which] = rho
        print(f"  rho (energy in the shared mean): {rho:6.2%}   <-- THE DECIDING NUMBER")
        del flat, sims

        raw_e, res_e = [], []
        for i in range(min(len(ids), 4)):        # 4 experts is enough for the spectrum
            whole = stack[i]
            resid = whole - shared
            tot_w = float(np.sum(whole.astype(np.float64) ** 2))
            tot_r = float(np.sum(resid.astype(np.float64) ** 2))
            sv_w = randomized_spectrum(whole, args.max_rank)
            sv_r = randomized_spectrum(resid, args.max_rank)
            raw_e.append(energy_at(sv_w, tot_w, RANKS))
            res_e.append(energy_at(sv_r, tot_r, RANKS))
            if i == 0:
                r95, r99 = rank_for(sv_r, tot_r, 0.95), rank_for(sv_r, tot_r, 0.99)
                findings.setdefault("rank_needed", {})[which] = {"p95": r95, "p99": r99}
            print(f"  expert {ids[i]} spectra done", end="\r", flush=True)
        print(" " * 40, end="\r")

        print(f"  {'rank':>6}  {'energy of W':>12}  {'energy of W-Base':>17}")
        for r in RANKS:
            a = float(np.mean([d[r] for d in raw_e]))
            b = float(np.mean([d[r] for d in res_e]))
            print(f"  {r:>6}  {a:>11.2%}  {b:>16.2%}")
        findings["matrices"][which] = {
            "pairwise_cosine_mean": float(off.mean()),
            "shared_mean_energy_share": base_share,
            "energy_whole": {str(r): float(np.mean([d[r] for d in raw_e])) for r in RANKS},
            "energy_residual": {str(r): float(np.mean([d[r] for d in res_e])) for r in RANKS},
        }
        del stack, shared
        print()

    print("=" * 64)
    print("VERDICT")
    for which, need in findings.get("rank_needed", {}).items():
        p95, p99 = need["p95"], need["p99"]
        s95 = f"{size_for_rank(p95):.0f} GB" if p95 else "n/a"
        s99 = f"{size_for_rank(p99):.0f} GB" if p99 else "n/a"
        print(f"  {which}: residual needs rank {p95} for 95% ({s95} checkpoint), "
              f"rank {p99} for 99% ({s99})")
    print()
    rhos = findings.get("rho", {})
    if rhos:
        worst = min(rhos.values())
        print("  rho by matrix: " + "  ".join(f"{k}={v:.3f}" for k, v in rhos.items()))
        print(f"  worst: {worst:.3f}")
        if worst >= 0.95:
            print("  => BUILD IT. Shared base + rank-16 delta, ~72 GB total.")
        elif worst >= 0.85:
            print("  => PARTIAL. Try clustering to K=8-16 centroids per layer plus a")
            print("     rank-16 delta (~84-97 GB). Plain K=1 base will not be enough.")
        else:
            print("  => STOP. No post-hoc structural scheme reaches 200 GB at this rho.")
            print("     Say so plainly rather than spending rank on it.")
    print()
    print("  Do NOT respond to a weak rho by raising the rank. Required rho is almost")
    print("  independent of r: going r=16 -> r=128 is 8x the storage and moves the bar")
    print("  by ~0.0014. If it fails at r=16, it fails.")
    print()
    print("  Weight-space energy is a proxy. A good number here justifies building the")
    print("  container and measuring perplexity; it does not prove quality is kept.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(findings, handle, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
