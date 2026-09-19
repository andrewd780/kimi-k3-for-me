#!/usr/bin/env python3
"""Held-out next-layer routing calibration; never changes engine routing.

NPZ: x[N,H] float hidden states available BEFORE the target layer, routes[N,K]
integer target IDs, layer[N], position[N], prompt[N] Unicode identifiers, and
n_experts scalar. One row per (prompt, position, target layer): repeated prefills
must be deduplicated upstream. Split by prompt, never by request or position.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def topk(scores, k):
    # Stable expert-ID tie break makes the calibration reproducible.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(data, train_prompts, ridge=1.0):
    x = np.array(data["x"], dtype=np.float64, copy=True)
    routes, layer, position, prompt = (np.asarray(data[key]) for key in
                                       ("routes", "layer", "position", "prompt"))
    count_value = np.asarray(data["n_experts"])
    if count_value.ndim != 0 or not np.issubdtype(count_value.dtype, np.integer):
        raise ValueError("n_experts must be an integer scalar")
    experts = int(count_value.item())
    if (x.ndim != 2 or routes.ndim != 2 or len(x) != len(routes) or
            not 1 <= routes.shape[1] <= experts <= 65536 or
            any(a.shape != (len(x),) for a in (layer, position, prompt)) or
            not all(np.issubdtype(a.dtype, np.integer) for a in (routes, layer, position)) or
            prompt.dtype.kind != "U" or not np.isfinite(x).all() or
            not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("invalid routing-probe shapes, types or values")
    if ((routes < 0).any() or (routes >= experts).any() or (layer < 0).any() or
            (position < 0).any() or
            any(len(set(row)) != len(row) for row in routes)):
        raise ValueError("invalid or repeated expert IDs")
    keys = list(zip(prompt.tolist(), position.tolist(), layer.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate positions: remove replayed prefills before calibration")
    training = np.isin(prompt, list(train_prompts))
    if not training.any() or training.all():
        raise ValueError("need disjoint, nonempty training and held-out prompt groups")
    if set(train_prompts) - set(prompt.tolist()):
        raise ValueError("unknown training prompt identifier")
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    results = []
    for target in sorted(set(layer.tolist())):
        a, b = training & (layer == target), ~training & (layer == target)
        if not a.any() or not b.any():
            raise ValueError("every target layer needs both training and held-out prompts")
        # Bound this research tool explicitly; it is not a full-trace trainer.
        if a.sum() > 8192 or experts * a.sum() > 16_000_000:
            raise ValueError("calibration too large: sample unique positions per prompt")
        xa, xb = x[a], x[b]
        labels = np.zeros((len(xa), experts))
        labels[np.arange(len(xa))[:, None], routes[a]] = 1.0
        count = labels.sum(axis=0)
        centroids = labels.T @ xa / np.maximum(count[:, None], 1)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-30)
        # Dual ridge avoids an H x H solve for the real hidden width (7168).
        gram = xa @ xa.T
        gram.flat[::len(xa) + 1] += ridge
        dual = np.linalg.solve(gram, labels)
        scores = {"centroid": xb @ centroids.T, "ridge": (xb @ xa.T) @ dual}
        for method, values in scores.items():
            values[:, count == 0] = -np.inf
            selected = topk(values, routes.shape[1])
            overlap = [len(set(p) & set(q)) for p, q in zip(selected, routes[b])]
            recall = sum(overlap) / (len(overlap) * routes.shape[1])
            results.append({"target_layer": int(target), "method": method,
                            "train_positions": int(a.sum()), "heldout_positions": int(b.sum()),
                            "topk_recall": recall, "expert_read_multiplier": 2 - recall,
                            "clears_70_percent_recall_gate": recall >= 0.7})
    return {"scope": "held-out recall; no prefetch latency or K3 speed claim",
            "train_prompts": sorted(train_prompts),
            "heldout_prompts": sorted(set(prompt[~training].tolist())), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--train-prompts", required=True, help="comma-separated prompt IDs")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        report = evaluate(data, set(args.train_prompts.split(",")), args.ridge)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
