#!/usr/bin/env python3
"""Lead-labelled routing recall diagnostic; never changes engine routing.

NPZ: x[N,H] router-input features; routes[N,K] integer target IDs; layer[N],
position[N], source_layer[N], source_position[N]; prompt[N] Unicode group IDs;
phase[N] must be "decode". Scalars: n_experts; feature_site="router_input";
evidence_kind="synthetic" or "generation_capture". --lead is explicit: 0 is
an oracle diagnostic, 1/2/4 use an earlier layer at the SAME generated position.

Metadata is a declaration, not proof of a real generation capture. Unique
prefill rows are ineligible too; deduplicating a prefix replay does not fix it.
Split by original prompt (including all its trajectories), never by position.
See docs/notes/predictive-prefetch-gates.md for the capture contract and blocked
equal-slot static-pin/byte gates. Recall alone cannot pass a deployment gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def topk(scores, k):
    # Stable expert-ID tie break makes the calibration reproducible.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def text_scalar(data, key, allowed):
    value = np.asarray(data[key])
    if value.ndim != 0 or value.dtype.kind != "U" or value.item() not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}")
    return value.item()


def evaluate(data, train_prompts, lead, ridge=1.0):
    required = {"x", "routes", "layer", "position", "prompt", "n_experts",
                "source_layer", "source_position", "phase", "feature_site", "evidence_kind"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing capture/lead metadata: {', '.join(sorted(missing))}")
    if lead not in (0, 1, 2, 4):
        raise ValueError("lead must be 0 (oracle diagnostic), 1, 2 or 4")
    text_scalar(data, "feature_site", {"router_input"})
    evidence_kind = text_scalar(data, "evidence_kind", {"synthetic", "generation_capture"})
    x = np.array(data["x"], dtype=np.float64, copy=True)
    routes, layer, position, prompt = (np.asarray(data[key]) for key in
                                       ("routes", "layer", "position", "prompt"))
    count_value = np.asarray(data["n_experts"])
    if count_value.ndim != 0 or not np.issubdtype(count_value.dtype, np.integer):
        raise ValueError("n_experts must be an integer scalar")
    experts = int(count_value.item())
    source_layer, source_position, phase = (np.asarray(data[key]) for key in
                                           ("source_layer", "source_position", "phase"))
    if (x.ndim != 2 or routes.ndim != 2 or len(x) != len(routes) or
            not x.shape[1] or not len(x) or
            not 1 <= routes.shape[1] <= experts <= 65536 or
            any(a.shape != (len(x),) for a in
                (layer, position, prompt, source_layer, source_position, phase)) or
            not all(np.issubdtype(a.dtype, np.integer) for a in
                    (routes, layer, position, source_layer, source_position)) or
            prompt.dtype.kind != "U" or phase.dtype.kind != "U" or
            not np.isfinite(x).all() or
            not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("invalid routing-probe shapes, types or values")
    if ((routes < 0).any() or (routes >= experts).any() or (layer < 0).any() or
            (position < 0).any() or
            any(len(set(row)) != len(row) for row in routes)):
        raise ValueError("invalid or repeated expert IDs")
    if (phase != "decode").any():
        raise ValueError("only incremental decode rows are eligible; prefill/replay is not decode")
    if (source_position != position).any():
        raise ValueError("source and target must be at the same generated position")
    # Convert to Python ints before subtraction: unsigned layer arrays must not wrap.
    if any(int(src) < 0 or int(dst) - int(src) != lead
           for src, dst in zip(source_layer, layer)):
        raise ValueError("source layer does not match the declared lead")
    keys = list(zip(prompt.tolist(), position.tolist(), layer.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate positions; capture each generated position once per layer")
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
            results.append({"target_layer": int(target), "source_layer": int(target) - lead,
                            "lead_layers": lead, "oracle_diagnostic": lead == 0, "method": method,
                            "train_positions": int(a.sum()), "heldout_positions": int(b.sum()),
                            "topk_recall": recall})
    return {"scope": "recall diagnostic only; no residency, read-byte or speed result",
            "evidence_kind": evidence_kind, "declared_phase": "decode", "lead_layers": lead,
            "oracle_diagnostic": lead == 0,
            "gate_status": "not_evaluated",
            "unmeasured": ["equal-slot global static-pin null", "bytes per decode token",
                           "lead time in seconds", "incremental gain over known-route pipeline"],
            "train_prompts": sorted(train_prompts),
            "heldout_prompts": sorted(set(prompt[~training].tolist())), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--train-prompts", required=True, help="comma-separated prompt IDs")
    parser.add_argument("--lead", type=int, choices=(0, 1, 2, 4), required=True,
                        help="target layer minus source layer; 0 is an oracle diagnostic only")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        report = evaluate(data, set(args.train_prompts.split(",")), args.lead, args.ridge)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
