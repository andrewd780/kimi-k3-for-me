#!/usr/bin/env python3
"""Offline corpus perplexity for the native K3 engine. No downloads or model conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def windows(ids, context, stride):
    """(document offset, window ids, first target index); score each target once."""
    if not 2 <= context <= 32768 or not 1 <= stride < context or len(ids) < 2:
        raise ValueError("need >=2 tokens, context 2..32768 and stride 1..context-1")
    previous_end, end = 1, min(context, len(ids))
    while True:
        start = max(0, end - context)
        yield start, ids[start:end], previous_end - start
        if end == len(ids):
            return
        previous_end, end = end, min(end + stride, len(ids))


def metric(losses):
    if not losses or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                         for v in losses):
        raise ValueError("missing, negative or non-finite token losses")
    total = math.fsum(losses)
    mean = total / len(losses)
    return {"scored_tokens": len(losses), "nll_sum": total, "mean_nll": mean,
            "perplexity": math.exp(mean) if mean <= math.log(float.fromhex("0x1.fffffffffffffp1023"))
            else None}


def validate_result(result, ids, first, expected_layers):
    if (result.get("schema") != 1 or result.get("task") != "perplexity"
            or result.get("input_ids") != ids or result.get("score_start") != first
            or result.get("layers") != expected_layers
            or result.get("layers_completed") != expected_layers
            or result.get("expert_drops") != 0):
        raise ValueError("native scorer returned a mismatched or incomplete window")
    losses = result.get("token_nll", [])
    stats = metric(losses)
    if len(losses) != len(ids) - first or result.get("scored_tokens") != len(losses):
        raise ValueError("native scorer target count differs")
    for key in ("nll_sum", "mean_nll"):
        value = result.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not math.isclose(stats[key], value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("native scorer aggregate differs: " + key)
    return losses


def read_prompts(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or ("text" in row) == ("token_ids" in row):
                raise ValueError(f"line {line_no}: supply exactly one of text/token_ids")
            name = row.get("id", str(line_no))
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError(f"line {line_no}: id must be a unique nonempty string")
            if "text" in row and (not isinstance(row["text"], str) or not row["text"]):
                raise ValueError(f"line {line_no}: text must be nonempty")
            if "token_ids" in row and (not isinstance(row["token_ids"], list)
                    or not row["token_ids"]
                    or any(type(i) is not int or i < 0 for i in row["token_ids"])):
                raise ValueError(f"line {line_no}: token_ids must be nonnegative integers")
            seen.add(name)
            rows.append({**row, "id": name})
    if not rows:
        raise ValueError("empty prompt set")
    return rows


def tokenize(text, binary, directory):
    # test_tok encodefile has a 4 MiB input buffer and a 1M-token cap. Restrict text
    # to <1 MiB so neither can silently truncate even in the one-token/byte case.
    data = text.encode("utf-8")
    if len(data) >= (1 << 20) or b"\0" in data:
        raise ValueError("text document must be <1 MiB UTF-8 with no embedded NUL")
    with tempfile.TemporaryDirectory(prefix="k3-quality-tokenize-") as temporary:
        path = Path(temporary) / "prompt.txt"
        path.write_bytes(data)
        result = subprocess.run([str(binary), str(directory), "encodefile", str(path)],
                                check=True, capture_output=True, text=True)
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return [int(i) for i in line.split(",")] if line else []


def local_identity(directory):
    if (directory / ".k3-remote").exists():
        raise ValueError("quality evaluation requires local weights, not a remote range model")
    files = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("*.k3z")) |
                   set(directory.glob("trunk.bin")))
    if not files:
        raise ValueError("no local checkpoint/trunk files in " + str(directory))
    # Do not read 1.56 TB merely to label an evaluation. This is explicitly a file
    # inventory, not cryptographic authentication of every weight byte.
    return [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL: id and text or token_ids")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("bin/k3"))
    parser.add_argument("--trunk", type=Path)
    parser.add_argument("--trunk-gb", type=float, default=16)
    parser.add_argument("--cache-gb", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--bos-id", type=int, help="explicitly prepend this token to each document")
    parser.add_argument("--tokenizer", type=Path, help="local tokenizer files for text records")
    parser.add_argument("--tokenizer-binary", type=Path, default=Path("bin/test_tok"))
    parser.add_argument("--baseline", type=Path, help="compare mean NLL on the identical token set")
    parser.add_argument("--checkpoint-label", default="unlabelled",
                        help="user-supplied revision/variant label; not weight authentication")
    args = parser.parse_args()
    list(windows([0, 1], args.window, args.stride))  # validate before any model bind
    for budget in (args.cache_gb, args.trunk_gb):
        if not math.isfinite(budget) or budget <= 0:
            parser.error("memory budgets must be finite and positive")
    checkpoint, binary = args.checkpoint.resolve(), args.binary.resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config = config.get("text_config", config)
    vocab = config.get("vocab_size", config.get("vocab"))
    layers = config.get("num_hidden_layers", config.get("n_layers"))
    if type(vocab) is not int or vocab < 1 or type(layers) is not int or layers < 1:
        raise ValueError("checkpoint config must declare vocabulary and full layer count")
    if args.bos_id is not None and not 0 <= args.bos_id < vocab:
        parser.error("BOS is outside the vocabulary")
    identity = local_identity(checkpoint)
    trunk = args.trunk.resolve() if args.trunk else None
    trunk_identity = local_identity(trunk) if trunk else None
    rows = read_prompts(args.prompts)
    tokenizer_identity = None
    if any("text" in row for row in rows):
        if not args.tokenizer:
            parser.error("text records need --tokenizer and a built bin/test_tok")
        tokenizer_identity = {name: sha256(args.tokenizer / name)
                              for name in ("tiktoken.model", "tokenizer_config.json")}
        tokenizer_identity["binary_sha256"] = sha256(args.tokenizer_binary)
    for row in rows:
        ids = (tokenize(row["text"], args.tokenizer_binary.resolve(), args.tokenizer.resolve())
               if "text" in row else row["token_ids"])
        if args.bos_id is not None:
            ids = [args.bos_id] + ids
        if len(ids) < 2 or any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("document needs >=2 valid token ids: " + row["id"])
        row["ids"] = ids
    token_hash = hashlib.sha256(json.dumps([(r["id"], r["ids"]) for r in rows],
                                          separators=(",", ":")).encode()).hexdigest()
    protocol = {"schema": 1, "tokenized_prompts_sha256": token_hash, "window": args.window,
                "stride": args.stride, "bos_id": args.bos_id, "vocab": vocab,
                "document_boundaries": "reset", "first_token": "context only",
                "template": "none", "implicit_eos": False}
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and (baseline.get("status") != "complete" or baseline.get("protocol") != protocol):
        raise ValueError("baseline used a different token set or scoring protocol")
    command = [str(binary), str(checkpoint), "--score-prompt", "--cache-gb", str(args.cache_gb)]
    if trunk:
        command += ["--trunk", str(trunk), "--trunk-gb", str(args.trunk_gb)]
    report = {"schema": 1, "status": "complete", "protocol": protocol,
              "checkpoint_label": args.checkpoint_label, "checkpoint": str(checkpoint),
              "checkpoint_files": identity, "config_sha256": sha256(config_path),
              "trunk": str(trunk) if trunk else None, "trunk_files": trunk_identity,
              "trunk_manifest_sha256": sha256(trunk / "trunk.json") if trunk else None,
              "binary_sha256": sha256(binary), "command_prefix": command,
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
              "prompts_sha256": sha256(args.prompts), "tokenizer": tokenizer_identity,
              "documents": [], "limits": "Corpus perplexity is not a task-suite result. "
              "Weights are identified by local inventory and metadata, not full-file hashes. "
              "Each window is a fresh native process; model binding is repeated."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = args.out.parent / (args.out.stem + "-windows")
    work.mkdir()  # refuse stale native outputs from a prior/incomplete evaluation
    corpus_losses = []
    for document_no, row in enumerate(rows):
        losses, details = [], []
        for window_no, (offset, ids, first) in enumerate(windows(row["ids"], args.window, args.stride)):
            stem = work / f"{document_no:05d}-{window_no:05d}"
            output = stem.with_suffix(".json")
            call = command + ["--ids", ",".join(map(str, ids)), "--score-start", str(first),
                              "--out", str(output.resolve())]
            with stem.with_suffix(".log").open("w") as log:
                subprocess.run(call, check=True, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads(output.read_text())
            part = validate_result(result, ids, first, layers)
            losses.extend(part)
            details.append({"offset": offset, "first_target": offset + first,
                            "last_target": offset + len(ids) - 1,
                            "raw_result": output.name, **metric(part)})
            print(f"scored {row['id']!r}: targets {offset + first}..{offset + len(ids) - 1}", flush=True)
        if len(losses) != len(row["ids"]) - 1:
            raise ValueError("document target coverage differs")
        report["documents"].append({"id": row["id"], **metric(losses), "windows": details})
        corpus_losses.extend(losses)
    if (identity != local_identity(checkpoint) or report["config_sha256"] != sha256(config_path)
            or report["binary_sha256"] != sha256(binary)
            or (trunk and (trunk_identity != local_identity(trunk)
                           or report["trunk_manifest_sha256"] != sha256(trunk / "trunk.json")))):
        raise ValueError("checkpoint/binary changed during the run; results not published")
    report["corpus"] = metric(corpus_losses)  # token-weighted, never mean document PPL
    if baseline is not None:
        original = baseline["corpus"]
        if (original.get("scored_tokens") != len(corpus_losses)
                or type(original.get("mean_nll")) not in (int, float)
                or not math.isfinite(original["mean_nll"])):
            raise ValueError("invalid baseline aggregate")
        report["comparison"] = {"baseline_sha256": sha256(args.baseline),
                                "delta_mean_nll": report["corpus"]["mean_nll"] - original["mean_nll"]}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(report["corpus"], allow_nan=False))


if __name__ == "__main__":
    main()
