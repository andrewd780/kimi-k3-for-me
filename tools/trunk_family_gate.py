#!/usr/bin/env python3
"""Per-family trunk gate: gate 1 and the bit-width curve on every BF16 matrix family.

Gate 1's eight ranges are seven KDA f_a_proj ranges and one MLA g_proj range: two
families holding about 4% of trunk bytes. This tool extends the same measurements to
every family of the dense trunk (the 93 per-layer runs; routed experts and embed/head
excluded):

  inventory  exact per-family bytes from the released config's shapes, plus the byte
             cost of the FDRX row index for every row width. No network.
  sample     hosted CI only: reads every shard header at the pinned revision (exact
             inventory, dtypes, offsets), then a systematic sample of 1 MiB ranges from
             every BF16 matrix family, and reports per-family coverage and bit-width
             curve under ONE byte-weighted pooled dictionary, plus byte-weighted
             pooled figures. Sample hashes are recorded on first observation; pass
             --pins with a committed record of them to require the same bytes again.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import platform
import re
import struct

from remote_model import MAX_HEADER, decode, digest, read_range, read_small, validate_header
import trunk_dictionary_gate as gate

REPO = "moonshotai/Kimi-K3"
LAYER = re.compile(r"language_model\.model\.layers\.([0-9]+)\.(.+)")
TRUNK_TOTAL_BYTES_DOCUMENTED = 108_810_000_000  # docs/ARCHITECTURE.md, rounded to 10 MB
ROW_INDEX_HEADER = 16                           # FDRX magic, rows, cols, group
ROW_CHECKPOINT = 4                              # one u32 per group of rows
GROUP_VALUES = 8192                             # grouped index: >= 8192 values per entry

# Released K3 dimensions (include/k3/k3.h, src/cli/k3_run.c real_cfg_hardcoded).
K3 = {"hidden": 7168, "layers": 93, "kda_heads": 96, "kda_head_dim": 128, "heads": 96,
      "q_lora": 1536, "kv_lora": 512, "qk_nope": 128, "qk_rope": 64, "v_head": 128,
      "experts": 896, "shared": 2, "latent": 3584, "moe_inter": 3072, "dense_inter": 33792}

SUFFIX = {
    "self_attn.q_proj.weight": "kda.q_proj", "self_attn.k_proj.weight": "kda.k_proj",
    "self_attn.v_proj.weight": "kda.v_proj", "self_attn.f_a_proj.weight": "kda.f_a_proj",
    "self_attn.f_b_proj.weight": "kda.f_b_proj", "self_attn.b_proj.weight": "kda.b_proj",
    "self_attn.q_a_proj.weight": "mla.q_a_proj", "self_attn.q_b_proj.weight": "mla.q_b_proj",
    "self_attn.kv_a_proj_with_mqa.weight": "mla.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj.weight": "mla.kv_b_proj",
    "block_sparse_moe.routed_expert_down_proj.weight": "moe.latent_down",
    "block_sparse_moe.routed_expert_up_proj.weight": "moe.latent_up",
    "block_sparse_moe.shared_experts.gate_proj.weight": "moe.shared_gate",
    "block_sparse_moe.shared_experts.up_proj.weight": "moe.shared_up",
    "block_sparse_moe.shared_experts.down_proj.weight": "moe.shared_down",
    "block_sparse_moe.gate.weight": "moe.router",
    "mlp.gate_proj.weight": "dense.gate_proj", "mlp.up_proj.weight": "dense.up_proj",
    "mlp.down_proj.weight": "dense.down_proj",
}


def mla_layers(k3=K3):
    """Zero-based MLA layers: every fourth (one-based 4, 8, ..., 92) plus the last."""
    return sorted({i - 1 for i in range(4, k3["layers"] + 1, 4)} | {k3["layers"] - 1})


def family_of(suffix, mla):
    """Family of one per-layer tensor suffix; None for routed experts."""
    if suffix.startswith("block_sparse_moe.experts."):
        return None
    if suffix in ("self_attn.o_proj.weight", "self_attn.g_proj.weight"):
        return ("mla." if mla else "kda.") + suffix.split(".")[1]
    return SUFFIX.get(suffix, "elementwise")


def config_inventory(k3=K3):
    """Every 2-D BF16 matrix of the trunk from config shapes: (family, rows, cols, layers)."""
    h, p = k3["hidden"], k3["kda_heads"] * k3["kda_head_dim"]
    mla = len(mla_layers(k3))
    kda, moe = k3["layers"] - mla, k3["layers"] - 1
    si, qh = k3["shared"] * k3["moe_inter"], k3["qk_nope"] + k3["qk_rope"]
    return [
        ("kda.q_proj", p, h, kda), ("kda.k_proj", p, h, kda), ("kda.v_proj", p, h, kda),
        ("kda.g_proj", p, h, kda), ("kda.o_proj", h, p, kda),
        ("kda.f_a_proj", k3["kda_head_dim"], h, kda), ("kda.f_b_proj", p, k3["kda_head_dim"], kda),
        ("kda.b_proj", k3["kda_heads"], h, kda),
        ("mla.q_a_proj", k3["q_lora"], h, mla), ("mla.q_b_proj", k3["heads"] * qh, k3["q_lora"], mla),
        ("mla.kv_a_proj_with_mqa", k3["kv_lora"] + k3["qk_rope"], h, mla),
        ("mla.kv_b_proj", k3["heads"] * (k3["qk_nope"] + k3["v_head"]), k3["kv_lora"], mla),
        ("mla.o_proj", h, k3["heads"] * k3["v_head"], mla),
        ("mla.g_proj", k3["heads"] * k3["v_head"], h, mla),
        ("moe.latent_down", k3["latent"], h, moe), ("moe.latent_up", h, k3["latent"], moe),
        ("moe.shared_gate", si, h, moe), ("moe.shared_up", si, h, moe),
        ("moe.shared_down", h, si, moe), ("moe.router", k3["experts"], h, moe),
        ("dense.gate_proj", k3["dense_inter"], h, 1), ("dense.up_proj", k3["dense_inter"], h, 1),
        ("dense.down_proj", h, k3["dense_inter"], 1),
    ]


def row_index_bytes(rows, cols, group):
    """Exact FDRX bytes for one matrix: header plus one u32 per group of rows."""
    if rows < 0 or cols <= 0 or group <= 0:
        raise ValueError("invalid matrix or group")
    return ROW_INDEX_HEADER + ROW_CHECKPOINT * ((rows + group - 1) // group)


def grouped(cols):
    """Rows per checkpoint so each entry spans at least GROUP_VALUES values (per row if wider)."""
    return max(1, -(-GROUP_VALUES // cols))


def inventory_report(k3=K3):
    rows, total = [], 0
    for family, r, c, count in config_inventory(k3):
        raw = 2 * r * c * count
        per_row, per_group = row_index_bytes(r, c, 1) * count, row_index_bytes(r, c, grouped(c)) * count
        rows.append({"family": family, "rows": r, "cols": c, "matrices": count, "bytes": raw,
                     "row_index_per_row_bytes": per_row,
                     "row_index_per_row_points": 100 * per_row / raw,
                     "group_rows": grouped(c), "row_index_grouped_bytes": per_group,
                     "row_index_grouped_points": 100 * per_group / raw})
        total += raw
    for row in rows:
        row["share_of_matrices"] = row["bytes"] / total
        row["share_of_documented_trunk"] = row["bytes"] / TRUNK_TOTAL_BYTES_DOCUMENTED
    widths = {}
    for row in rows:
        w = widths.setdefault(row["cols"], {"cols": row["cols"], "bytes": 0, "families": []})
        w["bytes"] += row["bytes"]
        w["families"].append(row["family"])
    per_row = sum(r["row_index_per_row_bytes"] for r in rows)
    per_group = sum(r["row_index_grouped_bytes"] for r in rows)
    return {"schema": "trunk-inventory-v1", "source": "released config shapes (include/k3/k3.h)",
            "matrix_bytes": total, "documented_trunk_bytes": TRUNK_TOTAL_BYTES_DOCUMENTED,
            "non_matrix_bytes_estimate": TRUNK_TOTAL_BYTES_DOCUMENTED - total,
            "families": rows, "row_widths": sorted(widths.values(), key=lambda w: w["cols"]),
            "row_index_total": {"per_row_bytes": per_row, "per_row_points": 100 * per_row / total,
                                "grouped_bytes": per_group,
                                "grouped_points": 100 * per_group / total},
            "limits": "Matrix shapes are exact; the elementwise remainder is the documented "
                      "108.81 GB (rounded) minus matrices. The sample mode reads exact bytes "
                      "and dtypes from every shard header."}


def header_inventory(tensors):
    """Classify every per-layer trunk tensor from shard headers (routed experts excluded)."""
    suffixes = {}
    for t in tensors:
        match = LAYER.fullmatch(t["name"])
        if match:
            suffixes.setdefault(int(match[1]), set()).add(match[2])
    mla = {layer for layer, names in suffixes.items()
           if "self_attn.kv_b_proj.weight" in names}
    families = {}
    for t in sorted(tensors, key=lambda t: (layer_of(t["name"]), t["name"])):
        match = LAYER.fullmatch(t["name"])
        if not match:
            continue
        family = family_of(match[2], int(match[1]) in mla)
        if family is None:
            continue
        f = families.setdefault(family, {"bytes": 0, "tensors": [], "dtypes": Counter()})
        f["bytes"] += t["bytes"]
        f["dtypes"][t["dtype"]] += t["bytes"]
        f["tensors"].append(t)
    total = sum(f["bytes"] for f in families.values())
    for name, f in families.items():
        f["share"] = f["bytes"] / total
        f["sampled"] = (name != "elementwise" and set(f["dtypes"]) == {"BF16"}
                        and all(len(t["shape"]) == 2 for t in f["tensors"]))
    return families, total


def layer_of(name):
    match = LAYER.fullmatch(name)
    return int(match[1]) if match else -1


def plan_family(tensors, samples=4, sample_bytes=1 << 20):
    """Systematic sample: the 1 MiB range centered at (s + 1/2)/samples of the family's
    bytes concatenated in layer order, clamped inside its tensor, 2-byte aligned."""
    total = sum(t["bytes"] for t in tensors)
    if not tensors or samples <= 0 or total <= 0 or any(t["bytes"] % 2 for t in tensors):
        raise ValueError("invalid family or sample count")
    plan, seen = [], set()
    for s in range(samples):
        target, start = (2 * s + 1) * total // (2 * samples), 0
        for t in tensors:
            if target < start + t["bytes"]:
                break
            start += t["bytes"]
        length = min(sample_bytes, t["bytes"])
        within = min(max(target - start - length // 2, 0), t["bytes"] - length)
        within -= within % 2
        key = (t["name"], within)
        if key in seen:
            continue
        seen.add(key)
        plan.append({"shard": t["shard"], "tensor": t["name"], "offset": t["offset"] + within,
                     "bytes": length, "position": target / total})
    return plan


def mixture(families):
    """Byte-weighted mixture of per-family high-byte distributions (exact fractions)."""
    total = sum(f["bytes"] for f in families.values())
    mix = [Fraction(0)] * 256
    for f in families.values():
        n = sum(f["histogram"])
        for v, c in enumerate(f["histogram"]):
            if c:
                mix[v] += Fraction(f["bytes"] * c, total * n)
    ranking = sorted((v for v in range(256) if mix[v]), key=lambda v: (-mix[v], v))
    magnitudes = [mix[m] + mix[m | 0x80] for m in range(128)]
    magnitude_ranking = sorted((m for m in range(128) if magnitudes[m]),
                               key=lambda m: (-magnitudes[m], m))
    return mix, ranking, magnitude_ranking


def entropy_of(probabilities):
    return -sum(float(p) * math.log2(float(p)) for p in probabilities if p)


def analyze_families(families, gate1_dictionary=None):
    """families: name -> {bytes, histogram (256 ints), full (Counter or None), samples}."""
    if not families:
        raise ValueError("no sampled families")
    for f in families.values():
        gate.rank(f["histogram"])
    total = sum(f["bytes"] for f in families.values())
    mix, ranking, magnitude_ranking = mixture(families)
    dictionary = ranking[:15]
    rows = {}
    for name, f in sorted(families.items()):
        curve = gate.bit_width_curve(f["histogram"], ranking, magnitude_ranking, f.get("full"))
        row = {"bytes": f["bytes"], "share_of_sampled": f["bytes"] / total,
               "samples": f.get("samples", []), "bf16_values": sum(f["histogram"]),
               "histogram": f["histogram"], **{k: v for k, v in gate.describe(
                   f["histogram"], dictionary).items() if k not in ("histogram", "bf16_values")},
               "bit_width_curve": curve}
        if gate1_dictionary is not None:
            row["gate1_15"] = gate.dictionary_stats(f["histogram"], gate1_dictionary)
        rows[name] = row
    weights = {name: Fraction(f["bytes"], total) for name, f in families.items()}
    schemes = {}
    for scheme in (s["scheme"] for s in next(iter(rows.values()))["bit_width_curve"]["schemes"]):
        exact = sum(weights[name] * Fraction(s["payload_bits"], 16 * row["bf16_values"])
                    for name, row in rows.items() for s in row["bit_width_curve"]["schemes"]
                    if s["scheme"] == scheme)
        schemes[scheme] = exact
    reference = schemes["fixed_4bit"]
    best = min((s for s in schemes if s != "fixed_4bit"), key=lambda s: schemes[s])
    pooled = {"schemes": [{"scheme": s, "payload_ratio": float(r),
                           "gain_vs_fixed_4bit_points": float(100 * (reference - r))}
                          for s, r in schemes.items()],
              "decision": {"rule": "prototype a decoder only if a scheme beats fixed_4bit by "
                                   "at least 1.5 byte-weighted ratio points",
                           "best_scheme": best,
                           "best_gain_vs_fixed_4bit_points": float(100 * (reference - schemes[best])),
                           "prototype": reference - schemes[best] >= Fraction(15, 1000)},
              "coverage_global_15": float(sum(weights[n] * Fraction(
                  rows[n]["global_15"]["covered_values"], rows[n]["bf16_values"]) for n in rows)),
              "per_family_code_bounds": {
                  key: float(sum(weights[n] * Fraction(rows[n]["bit_width_curve"]["bounds"][key])
                                 for n in rows))
                  for key in ("huffman_high_byte_ratio", "high_byte_entropy_ratio",
                              "full_bf16_entropy_ratio")
                  if all(key in rows[n]["bit_width_curve"]["bounds"] for n in rows)},
              "single_code_bounds": {"high_byte_mixture_entropy_ratio":
                                     (8 + entropy_of(mix)) / 16}}
    if all(f.get("full") for f in families.values()):
        full_mix = Counter()
        for f in families.values():
            n = sum(f["histogram"])
            for value, count in f["full"].items():
                full_mix[value] += Fraction(f["bytes"] * count, total * n)
        pooled["single_code_bounds"]["full_bf16_mixture_entropy_ratio"] = \
            entropy_of(full_mix.values()) / 16
    failed = sorted(n for n, r in rows.items() if not r["global_15"]["passes_99_percent"])
    return {"schema": "trunk-family-gate-v1", "families": rows, "byte_weighted_pooled": pooled,
            "dictionary_ranking": ranking,
            "gate": {"status": "STOP" if failed else "PASS",
                     "rule": "one byte-weighted pooled 15-entry dictionary covers at least 99% "
                             "of every sampled family; integer counts decide",
                     "failed_families": failed},
            "limits": "Systematic 1 MiB samples per family, not a census; the dictionary is "
                      "fit and scored on the same samples. Payload ratios exclude framing, "
                      "the row index and alignment. No speed or full-model claim."}


def family_plan(report):
    """Steps 3 and 4 of the per-family plan in docs/notes/fixed-width-trunk.md, applied to
    an analyze_families() report with exact payload-bit fractions.

    Step 3: a family the pooled 15-entry table fails keeps FD only with a table of its
    own, and only if that table covers at least 99% of it (best_local_15); otherwise the
    family stays raw BF16 at r = 1. Step 4: when the byte-weighted curve carries FD3B (the
    1.5-point rule), each coded family takes the smaller of its 3- and 4-bit payloads,
    scored with its own table where step 3 gave it one; otherwise every coded family is
    4-bit. The ratio is the byte-weighted mean over every sampled family, raw ones at 1,
    never over the coded families alone."""
    families = report["families"]
    failed = set(report["gate"]["failed_families"])
    fd3b = report["byte_weighted_pooled"]["decision"]["prototype"]
    total = sum(f["bytes"] for f in families.values())
    rows, ratio = {}, Fraction(0)
    for name, f in sorted(families.items()):
        if name in failed and not f["best_local_15"]["passes_99_percent"]:
            choice, exact = "raw", Fraction(1)
        else:
            own = name in failed
            curve = gate.bit_width_curve(f["histogram"]) if own else f["bit_width_curve"]
            bits = {s["scheme"]: s["payload_bits"] for s in curve["schemes"]}
            width = 3 if fd3b and bits["fixed_3bit"] < bits["fixed_4bit"] else 4
            choice = ("own" if own else "pooled") + f"_{width}bit"
            exact = Fraction(bits[f"fixed_{width}bit"], 16 * sum(f["histogram"]))
        rows[name] = {"choice": choice, "payload_ratio": float(exact)}
        ratio += Fraction(f["bytes"], total) * exact
    return {"families": rows, "payload_ratio": float(ratio), "fd3b": fd3b}


def verify_pins(plan, pins):
    """Every planned range must equal a pinned identity (same bytes again)."""
    pinned = {(p["shard"], p["offset"], p["bytes"]): p["sha256"] for p in pins}
    for row in plan:
        if (row["shard"], row["offset"], row["bytes"]) not in pinned:
            raise ValueError("planned range is not pinned: " + row["tensor"])
    return pinned


def markdown(report, shares):
    lines = ["| Family | Trunk share % | Samples | Global-15 coverage % | r 3-bit | r 4-bit | "
             "Huffman r |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["families"].items():
        schemes = {s["scheme"]: s for s in row["bit_width_curve"]["schemes"]}
        lines.append(f"| {name} | {100 * shares.get(name, 0):.3f} | {len(row['samples'])} | "
                     f"{100 * row['global_15']['coverage']:.6f} | "
                     f"{schemes['fixed_3bit']['payload_ratio']:.6f} | "
                     f"{schemes['fixed_4bit']['payload_ratio']:.6f} | "
                     f"{row['bit_width_curve']['bounds']['huffman_high_byte_ratio']:.6f} |")
    pooled = report["byte_weighted_pooled"]
    ratios = {s["scheme"]: s["payload_ratio"] for s in pooled["schemes"]}
    lines.append(f"| **byte-weighted** | | | {100 * pooled['coverage_global_15']:.6f} | "
                 f"{ratios['fixed_3bit']:.6f} | {ratios['fixed_4bit']:.6f} | "
                 f"{pooled['per_family_code_bounds'].get('huffman_high_byte_ratio', 0):.6f} |")
    lines.append(f"\nFamily gate: **{report['gate']['status']}**; best scheme "
                 f"{pooled['decision']['best_scheme']} "
                 f"({pooled['decision']['best_gain_vs_fixed_4bit_points']:+.4f} points).")
    return "\n".join(lines) + "\n"


def read_headers(revision):
    info = decode(read_small(f"https://huggingface.co/api/models/{REPO}/revision/{revision}"))
    if info.get("sha") != revision:
        raise ValueError("hub did not resolve the pinned revision")
    files = sorted(x["rfilename"] for x in info["siblings"]
                   if x["rfilename"].endswith(".safetensors"))
    names = [n for n in files if re.fullmatch(r"model-[0-9]{5}-of-[0-9]{6}\.safetensors", n)]
    if not names:
        raise ValueError("no model shards at the pinned revision")
    tensors, shards = [], {"ignored": [n for n in files if n not in names]}
    for name in names:
        url = f"https://huggingface.co/{REPO}/resolve/{revision}/{name}"
        prefix, total = read_range(url, 0, 8)
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER:
            raise ValueError("header too large: " + name)
        raw, _ = read_range(url, 8, size, total)
        header = validate_header(prefix + raw, total)
        shards[name] = {"url": url, "bytes": total, "header_sha256": digest(prefix + raw)}
        for tensor, entry in header.items():
            if tensor == "__metadata__":
                continue
            first, last = entry["data_offsets"]
            tensors.append({"name": tensor, "shard": name, "dtype": entry["dtype"],
                            "shape": entry["shape"], "offset": 8 + size + first,
                            "bytes": last - first})
    return tensors, shards


def sample(args):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("real-range acquisition is restricted to GitHub Actions")
    revision = json.loads(gate.MANIFEST.read_bytes())["revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("checkpoint revision must be immutable")
    tensors, shards = read_headers(revision)
    inventory, trunk_bytes = header_inventory(tensors)
    plans = {name: plan_family(f["tensors"], args.samples)
             for name, f in inventory.items() if f["sampled"]}
    pins = None
    if args.pins:
        pins = verify_pins([r for p in plans.values() for r in p],
                           json.loads(args.pins.read_bytes())["samples"])
    gate1 = (json.loads(args.gate1.read_bytes())["pooled"]["global_15"]["dictionary"]
             if args.gate1 else None)

    def fetch(row):
        raw, _ = read_range(shards[row["shard"]]["url"], row["offset"], row["bytes"],
                            shards[row["shard"]]["bytes"])
        return row, raw

    families = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, plan in plans.items():
            histogram, full, identities = [0] * 256, Counter(), []
            for row, raw in pool.map(fetch, plan):
                sha = digest(raw)
                if pins is not None and pins[(row["shard"], row["offset"], row["bytes"])] != sha:
                    raise ValueError("pinned sample hash mismatch: " + row["tensor"])
                histogram = [a + b for a, b in zip(histogram, gate.high_histogram(raw))]
                full.update(gate.bf16_histogram(raw))
                identities.append({**row, "sha256": sha})
            families[name] = {"bytes": inventory[name]["bytes"], "histogram": histogram,
                              "full": full, "samples": identities}
    report = analyze_families(families, gate1)
    shares = {name: f["share"] for name, f in inventory.items()}
    report.update({
        "checkpoint_revision": revision, "shards": shards, "trunk_bytes": trunk_bytes,
        "inventory": {name: {"bytes": f["bytes"], "share": f["share"], "tensors": len(f["tensors"]),
                             "dtypes": dict(f["dtypes"]), "sampled": f["sampled"],
                             "shapes": sorted({tuple(t["shape"]) for t in f["tensors"]})}
                      for name, f in sorted(inventory.items())},
        "pinned": pins is not None,
        "execution": {"platform": platform.platform(), "machine": platform.machine(),
                      "head_sha": os.environ.get("RESEARCH_COMMIT"),
                      "run_id": os.environ.get("GITHUB_RUN_ID")}})
    # The headers must describe the shapes the engine binds: flag any family that differs.
    expected = {f: 2 * r * c * n for f, r, c, n in config_inventory()}
    report["config_check"] = {
        f: {"config_bytes": expected.get(f), "header_bytes": inventory.get(f, {}).get("bytes", 0),
            "match": expected.get(f) == inventory.get(f, {}).get("bytes", 0)}
        for f in sorted((set(expected) | set(inventory)) - {"elementwise"})}
    fulls = {name: sorted(f["full"].items()) for name, f in families.items()}
    args.out.write_text(json.dumps({**report, "full_bf16_histograms": fulls}, indent=2) + "\n")
    table = markdown(report, shares)
    print(table, flush=True)
    printable = {k: v for k, v in report.items() if k != "shards"}
    print("FAMILY_REPORT " + json.dumps(printable, separators=(",", ":")), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(table)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="config-derived family shares and row-index cost")
    inv.add_argument("--out", type=Path)
    run = sub.add_parser("sample", help="hosted CI: every header, every BF16 family")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--samples", type=int, default=4, help="1 MiB ranges per family")
    run.add_argument("--gate1", type=Path, help="gate-1 artifact: also score its dictionary")
    run.add_argument("--pins", type=Path, help="committed sample identities to require")
    args = parser.parse_args()
    if args.command == "inventory":
        report = inventory_report()
        text = json.dumps(report, indent=2) + "\n"
        if args.out:
            args.out.write_text(text)
        print(text, end="")
        return 0
    if not 1 <= args.samples <= 16:
        parser.error("samples must be 1..16")
    return sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
