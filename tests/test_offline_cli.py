"""Whole tiny-model generation parity, prompt repetition and saved sessions.

Build with ZSTD=1 first. Requires numpy and torch for the synthetic checkpoint.
K3_TEST_BIN=bin/zstd/k3 python3 -m unittest discover -s tests -p test_offline_cli.py -v
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import offline_model as om  # noqa: E402
import pack_trunk  # noqa: E402
import expert_profile  # noqa: E402


class OfflineCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.binary = (ROOT / os.environ.get("K3_TEST_BIN", "bin/zstd/k3")).resolve()
        cls.env = {**os.environ, "OMP_NUM_THREADS": "2", "K3_NOHUGE": "1"}
        cls.plain, cls.packed = cls.root / "plain", cls.root / "packed"
        cls.selective = cls.root / "selective"
        subprocess.run([sys.executable, str(ROOT / "tools/make_tiny_checkpoint.py"),
                        str(cls.plain)], check=True, capture_output=True, env=cls.env)
        # Minimal all-byte vocabulary exercises text and UTF-8 file entry points.
        (cls.plain / "tiktoken.model").write_text("".join(
            base64.b64encode(bytes([i])).decode() + " %d\n" % i for i in range(256)))
        (cls.plain / "tokenizer_config.json").write_text('{"added_tokens_decoder":{}}')
        cls.trunk = cls.root / "trunk"
        cls.ztrunk = cls.root / "ztrunk"
        with contextlib.redirect_stdout(io.StringIO()):
            om.pack_model(cls.plain, cls.packed, block=64 << 10)
            om.pack_model(cls.plain, cls.selective, block=64 << 10, policy="scales")
            if pack_trunk.main([str(cls.packed), str(cls.trunk), "13"]):
                raise AssertionError("tiny trunk packing failed")
            cls.ztrunk.mkdir()
            shutil.copyfile(cls.trunk / "trunk.json", cls.ztrunk / "trunk.json")
            om.pack_file(cls.trunk / "trunk.bin", cls.ztrunk / "trunk.bin.k3z",
                         block=64 << 10)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.counter = 0

    def run_cli(self, model, args, ok=True):
        self.counter += 1
        out = self.path / (str(self.counter) + ".json")
        logits = self.path / (str(self.counter) + ".f32")
        result = subprocess.run([str(self.binary), str(model), "--gen", "2",
                                 "--cache-gb", "0.0001", "--out", str(out),
                                 "--dump-logits", str(logits), *map(str, args)],
                                text=True, errors="replace", capture_output=True,
                                env=self.env, timeout=60)
        if not ok:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("AddressSanitizer", result.stderr)
            self.assertNotIn("runtime error:", result.stderr)
            self.assertFalse(out.exists())
            return result
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # A synthetic all-byte vocabulary can emit incomplete UTF-8. Compare IDs
        # and binary logits; diagnostic text is not a tokenizer-quality assertion.
        report = json.loads(out.read_text(errors="replace"))
        self.assertEqual(report["expert_drops"], 0)
        self.assertEqual(report["layers_completed"], 13)
        raw_logits = logits.read_bytes()
        self.assertEqual(len(raw_logits), 256 * 4, "expected a complete vocabulary vector")
        return report, raw_logits

    def assert_same(self, a, b):
        self.assertEqual(a[0]["generated_ids"], b[0]["generated_ids"])
        self.assertEqual(a[0]["full_ids"], b[0]["full_ids"])
        self.assertEqual(a[1], b[1], "all dumped logits must be bit-identical")

    def test_compressed_vs_plain_generation_with_evictions(self):
        for mode in ([], ["--incremental"]):
            with self.subTest(mode=mode):
                args = ["--ids", "1,2,3", *mode]
                a = self.run_cli(self.plain, args)
                b = self.run_cli(self.packed, args)
                self.assert_same(a, b)
                self.assertFalse(b[0]["reread_prompt"])
                self.assertEqual(b[0]["original_prompt_tokens"], 3)

    def test_selective_scales_with_evictions_and_trunk_streaming(self):
        for mode in ([], ["--incremental"]):
            for head in ([], ["--stream-lm-head"]):
                with self.subTest(mode=mode, head=head):
                    args = ["--ids", "1,2,3", "--reread-prompt", "--trunk", self.ztrunk,
                            "--trunk-gb", "0.001", *mode, *head]
                    self.assert_same(self.run_cli(self.plain, args),
                                     self.run_cli(self.selective, args))

    def test_expert_pipeline_matches_serial_batch(self):
        # --expert-pipeline overlaps routed-expert reads with the MoE multiply (see
        # docs/notes/expert-pipeline.md); it must not change a single generated id or
        # logit. Cover both consumers of the batch path (full recompute, --incremental)
        # and let the thread-pool size vary, including the single-worker fallback shape
        # and the documented ceiling, since neither changes what gets served.
        for mode in ([], ["--incremental"]):
            for threads in ("1", "16"):
                with self.subTest(mode=mode, threads=threads):
                    args = ["--ids", "1,2,3", *mode]
                    base = self.run_cli(self.plain, args)
                    env = {**self.env, "K3_EXPERT_PIPELINE_THREADS": threads}
                    piped = self.run_cli(self.plain, [*args, "--expert-pipeline"])
                    self.assert_same(base, piped)
                    old_env = self.env
                    self.env = env
                    try:
                        piped_env = self.run_cli(self.plain, [*args, "--expert-pipeline"])
                    finally:
                        self.env = old_env
                    self.assert_same(base, piped_env)

    def test_native_score_primitive_on_synthetic_logits(self):
        # Does not invoke the corpus harness or produce a K3 quality measurement.
        # Compare the new native score path to ordinary prefix logits of this toy.
        ids = [3, 7, 11, 5]
        losses = []
        for position in range(1, len(ids)):
            _, raw = self.run_cli(self.plain, ["--ids", ",".join(map(str, ids[:position]))])
            logits = struct.unpack("=256f", raw)
            maximum = max(logits)
            losses.append(math.log(sum(math.exp(x - maximum) for x in logits))
                          + (maximum - logits[ids[position]]))
        for first in (1, 2):
            output = self.path / f"score-{first}.json"
            process = subprocess.run([str(self.binary), str(self.packed), "--score-prompt",
                                      "--ids", ",".join(map(str, ids)), "--score-start", str(first),
                                      "--trunk", str(self.ztrunk), "--trunk-gb", "0.001",
                                      "--cache-gb", "0.0001", "--out", str(output)],
                                     capture_output=True, text=True, env=self.env, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["scored_tokens"], len(ids) - first)
            self.assertEqual(report["input_ids"], ids)
            self.assertEqual(report["layers_completed"], 13)
            self.assertEqual(report["expert_drops"], 0)
            for got, want in zip(report["token_nll"], losses[first - 1:]):
                self.assertAlmostEqual(got, want, places=9)
            self.assertAlmostEqual(report["nll_sum"], sum(losses[first - 1:]), places=9)

    def test_stream_lm_head_independent_of_ultra(self):
        for mode in ([], ["--incremental"]):
            args = ["--ids", "3,7,11", "--trunk", self.ztrunk, "--trunk-gb", "0.001", *mode]
            resident = self.run_cli(self.packed, args)
            streamed = self.run_cli(self.packed, [*args, "--stream-lm-head"])
            self.assert_same(resident, streamed)
            self.assertFalse(streamed[0]["ultra_low_memory"])
            self.assertTrue(streamed[0]["lm_head_streamed"])
            self.assertEqual(streamed[0]["embedding_bytes_read"], 0)
            self.assertGreater(streamed[0]["lm_head_bytes_read"], 0)
        refused = self.run_cli(self.plain, ["--ids", "3,7", "--stream-lm-head",
                                           "--draft-trunk", "absent"], ok=False)
        self.assertIn("--stream-lm-head does not yet support --draft-trunk", refused.stderr)

    def test_lm_head_ring_under_cgroup_cap(self):
        if os.environ.get("K3_CGROUP_TEST") != "1":
            self.skipTest("set K3_CGROUP_TEST=1 on a Linux CI runner with sudo/systemd")
        cap = 64 << 20
        stream_buffer = (4 << 20) + 2 * 4096
        config = json.loads((self.plain / "config.json").read_text())
        trunk_meta = json.loads((self.trunk / "trunk.json").read_text())
        largest = max(layer["nbytes"] for layer in trunk_meta["layers"])
        # Leave generous room for this tiny config's vector widening. This is a
        # budget choice, not an alternate implementation of the ring allocator.
        one_budget = largest + (64 << 10)
        row_bytes = config["hidden_size"] * 2
        vocab = (stream_buffer + one_budget + row_bytes - 1) // row_bytes
        old_vocab = config["vocab_size"]
        config["vocab_size"] = vocab
        model = self.path / "wide-tables"
        model.mkdir()
        source = (self.plain / "model.safetensors").read_bytes()
        header_size = struct.unpack("<Q", source[:8])[0]
        header = json.loads(source[8:8 + header_size])
        source_payload = memoryview(source)[8 + header_size:]
        new_header, blobs, offset = {}, [], 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            first, last = entry["data_offsets"]
            raw = bytes(source_payload[first:last])
            entry = dict(entry)
            if name in ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight"):
                self.assertEqual(entry["dtype"], "BF16")
                self.assertEqual(len(raw), old_vocab * row_bytes)
                raw += raw[:row_bytes] * (vocab - old_vocab)
                entry["shape"] = [vocab, config["hidden_size"]]
            entry["data_offsets"] = [offset, offset + len(raw)]
            new_header[name] = entry
            blobs.append(raw)
            offset += len(raw)
        encoded = json.dumps(new_header).encode()
        with (model / "model.safetensors").open("wb") as f:
            f.write(struct.pack("<Q", len(encoded)))
            f.write(encoded)
            for blob in blobs:
                f.write(blob)
        (model / "config.json").write_text(json.dumps(config))
        head_bytes = vocab * row_bytes
        freed = head_bytes - stream_buffer
        results, logits = {}, {}
        for arm, budget in (("resident", one_budget), ("streamed", one_budget + freed)):
            output = self.path / (arm + ".json")
            raw_logits = self.path / (arm + ".f32")
            limits = self.path / (arm + "-cgroup.json")
            command = [str(self.binary), str(model), "--ids", "3,7,11", "--gen", "2",
                       "--cache-gb", "0.0001", "--trunk", str(self.trunk),
                       "--trunk-gb", f"{budget / 1e9:.9f}", "--out", str(output),
                       "--dump-logits", str(raw_logits)]
            if arm == "streamed":
                command.append("--stream-lm-head")
            run = ["sudo", "-n", "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
                   f"--unit=k3-head-{os.getpid()}-{arm}", f"--property=MemoryMax={cap}",
                   "--property=MemorySwapMax=0", "--property=MemoryAccounting=yes",
                   "--setenv=OMP_NUM_THREADS=2", "--setenv=K3_NOHUGE=1",
                   sys.executable, str(ROOT / "tools/cgroup_probe.py"), "--limit", str(cap),
                   "--report", str(limits), "--", *command]
            process = subprocess.run(run, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            engine = json.loads(output.read_text())
            group = json.loads(limits.read_text())
            self.assertEqual(engine["layers_completed"], 13)
            self.assertEqual(engine["expert_drops"], 0)
            self.assertEqual(group["memory_max_bytes"], cap)
            self.assertEqual(group["memory_swap_max_bytes"], 0)
            self.assertLessEqual(group["memory_peak_bytes"], cap)
            self.assertEqual(group["events"]["oom"], 0)
            self.assertFalse(engine["ultra_low_memory"])
            self.assertEqual(engine["embedding_bytes_read"], 0)
            self.assertLess(engine["memory_plan_bytes"], cap)
            logits[arm] = raw_logits.read_bytes()
            self.assertEqual(len(logits[arm]), vocab * 4)
            # Mechanism report deliberately excludes all fixture timings.
            fields = ("trunk_ring_slots", "trunk_slot_bytes", "trunk_budget_bytes",
                      "model_resident_bytes", "model_stream_buffer_bytes", "memory_plan_bytes",
                      "peak_rss_bytes", "lm_head_streamed", "lm_head_bytes_read", "generated_ids")
            results[arm] = {"engine": {key: engine[key] for key in fields}, "cgroup": group,
                            "logits_sha256": hashlib.sha256(logits[arm]).hexdigest()}
        a, b = results["resident"]["engine"], results["streamed"]["engine"]
        self.assertEqual(a["trunk_ring_slots"], 1)
        self.assertEqual(b["trunk_ring_slots"], 2)
        self.assertEqual(a["generated_ids"], b["generated_ids"])
        self.assertEqual(logits["resident"], logits["streamed"])
        self.assertEqual(a["model_resident_bytes"] - b["model_resident_bytes"], head_bytes)
        self.assertEqual(b["model_stream_buffer_bytes"], stream_buffer)
        self.assertEqual(a["memory_plan_bytes"], b["memory_plan_bytes"])
        report = {"kind": "scaled synthetic mechanism, not a real K3 memory/speed measurement",
                  "ci_run": os.environ.get("GITHUB_RUN_ID"), "head_bytes": head_bytes,
                  "net_bytes_available_for_trunk": freed, "vocab": vocab,
                  "checkpoint_bytes": (model / "model.safetensors").stat().st_size,
                  "arms": results}
        if os.environ.get("K3_HEAD_REPORT"):
            Path(os.environ["K3_HEAD_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
        print("HEAD MECHANISM " + json.dumps(report), flush=True)

    def test_reread_is_exact_token_repetition(self):
        for mode in ([], ["--incremental"]):
            with self.subTest(mode=mode):
                repeated = self.run_cli(self.packed,
                                        ["--ids", "1,2,3", "--reread-prompt", *mode])
                manual = self.run_cli(self.plain, ["--ids", "1,2,3,1,2,3", *mode])
                self.assert_same(repeated, manual)
                self.assertEqual(repeated[0]["prompt_ids"], [1, 2, 3] * 2)
                self.assertTrue(repeated[0]["reread_prompt"])
                self.assertEqual(repeated[0]["original_prompt_tokens"], 3)

    def test_text_and_utf8_prompt_file(self):
        prompt = "Read café.\n"
        path = self.path / "prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        for flag, value in (("--prompt", prompt), ("--prompt-file", path)):
            with self.subTest(flag=flag):
                once = self.run_cli(self.plain, [flag, value, "--tok", self.plain])
                ids = once[0]["prompt_ids"]
                repeated = self.run_cli(self.packed, [flag, value, "--tok", self.packed,
                                                      "--reread-prompt"])
                manual = self.run_cli(self.plain,
                                      ["--ids", ",".join(map(str, ids * 2))])
                self.assert_same(repeated, manual)
                self.assertEqual(repeated[0]["original_prompt_tokens"], len(ids))

    def test_compressed_trunk_and_streamed_model_parts(self):
        for mode, budget in (([], "0.0005"), ([], "0.001"),
                             (["--ultra-low-memory"], "0.0005")):
            with self.subTest(mode=mode, budget=budget):
                args = ["--ids", "1,2,3", "--reread-prompt", "--trunk-gb", budget]
                a = self.run_cli(self.plain, [*args, "--trunk", self.trunk, *mode])
                b = self.run_cli(self.packed, [*args, "--trunk", self.ztrunk, *mode])
                self.assert_same(a, b)
                self.assertGreater(b[0]["trunk_bytes_read"], 0)

    def test_reread_on_resume_repeats_new_request_only(self):
        state = self.path / "saved.bin"
        self.run_cli(self.plain, ["--ids", "1,2,3", "--incremental",
                                 "--save-state", state])
        args = ["--incremental", "--load-state", state]
        a = self.run_cli(self.packed, [*args, "--ids", "5,6", "--reread-prompt"])
        b = self.run_cli(self.plain, [*args, "--ids", "5,6,5,6"])
        self.assert_same(a, b)
        self.assertEqual(a[0]["prompt_ids"], [5, 6, 5, 6])
        self.assertEqual(a[0]["original_prompt_tokens"], 2)

    def test_prompt_limits_and_malformed_ids_refused_before_loading(self):
        for ids in ("1,x,2", "1.2", "99999999999999999999999", "-1", "256",
                    ",".join(["1"] * 32769)):
            with self.subTest(ids=ids[:24]):
                result = self.run_cli(self.plain, ["--ids", ids], ok=False)
                self.assertIn("--ids" if len(ids) < 100 else "ceiling", result.stderr)
        result = self.run_cli(self.plain, ["--ids", ",".join(["1"] * 16385),
                                          "--reread-prompt"], ok=False)
        self.assertIn("reread limit", result.stderr)

    def test_history_is_included_in_context_limit(self):
        state = self.path / "saved.bin"
        self.run_cli(self.plain, ["--ids", ",".join(["1"] * 12), "--incremental",
                                 "--save-state", state])
        result = self.run_cli(self.plain, ["--ids", ",".join(["1"] * 16380),
                                          "--gen", "4096", "--reread-prompt",
                                          "--incremental", "--load-state", state], ok=False)
        self.assertIn("history", result.stderr)
        self.assertIn("ceiling", result.stderr)

    def test_profiled_pins_preserve_complete_logits_three_runs_per_arm(self):
        trace = self.path / "calibration"
        trace.mkdir()
        self.run_cli(self.plain, ["--ids", "1,2,3", "--incremental",
                                  "--gen", "4", "--dump-cache-trace", trace])
        profile = self.path / "experts.profile"
        with contextlib.redirect_stdout(io.StringIO()):
            expert_profile.main(["build", str(trace / "expert_trace.bin"),
                                 "--n-layers", "13", "--n-experts", "8", "--topk", "2",
                                 "--out", str(profile)])
        measurements = []
        for mode in ([], ["--incremental"]):
            for run in range(3):
                with self.subTest(mode=mode, run=run):
                    args = ["--ids", "4,5,6", "--gen", "4", *mode]
                    base = self.run_cli(self.plain, args)
                    pins = self.run_cli(self.plain, [*args, "--expert-profile", profile,
                                                    "--pin-experts", "2"])
                    compressed = self.run_cli(self.packed, [*args, "--expert-profile",
                                                           profile, "--pin-experts", "2"])
                    self.assert_same(base, pins)
                    self.assert_same(base, compressed)
                    self.assertEqual(pins[0]["expert_profile_pins"], 2)
                    self.assertEqual(base[0]["expert_profile_pins"], 0)
                    for arm, result in (("plain", base), ("pinned", pins),
                                        ("compressed_pinned", compressed)):
                        report = result[0]
                        self.assertLessEqual(report["expert_resident_reuses"],
                                             report["expert_requests"])
                        measurements.append({"mode": "incremental" if mode else "full",
                                             "arm": arm, "run": run + 1,
                                             **{key: report[key] for key in
                                                ("wall_seconds", "expert_bytes_read",
                                                 "expert_requests", "expert_resident_reuses",
                                                 "expert_cache_slots")}})
        if os.environ.get("K3_PROFILE_REPORT"):
            Path(os.environ["K3_PROFILE_REPORT"]).write_text(json.dumps(
                {"fixture": "make_tiny_checkpoint.py, 13 layers, 8 experts, top-2",
                 "calibration_ids": [1, 2, 3], "evaluation_ids": [4, 5, 6],
                 "generated_tokens": 4, "pins": 2,
                 "limitations": "synthetic checkpoint; no real K3 speed or quality claim",
                 "runs": measurements}, indent=2) + "\n")

    def test_invalid_profiles_and_pin_budgets_fail(self):
        profile = self.path / "invalid.profile"
        header = "K3EXPERTS 1 13 8 2\n"
        invalid = ("K3EXPERTS 2 13 8 2\n1 0 3\n",
                   "K3EXPERTS 1 12 8 2\n1 0 3\n",
                   header + "1 0 3\n1 0 2\n",
                   header + "1 0 3\n1 1 4\n",
                   header + "1 1 3\n1 0 3\n",
                   header + "-1 0 3\n", header + "13 0 3\n",
                   header + "1 8 3\n", header + "0 0 3\n",
                   header + "1 0 0\n", header + "1 0 18446744073709551616\n",
                   header + "1 0 3 garbage\n", header + "1 0 3", header,
                   header + "1 0 3\x00hidden\n", header + "1 0 " + "9" * 300 + "\n")
        for text in invalid:
            with self.subTest(text=text[:65]):
                profile.write_bytes(text.encode())
                result = self.run_cli(self.plain, ["--ids", "1", "--expert-profile",
                                                  profile, "--pin-experts", "1"], ok=False)
                self.assertIn("profile", result.stderr)
                self.assertNotIn("bound 13/13", result.stdout)
                self.assertNotIn("embedding, final norm and lm_head:", result.stdout)
        profile.write_text(header + "1 0 3\n")
        for value in ("0", "-1", "2.5", "99999999999999999999", "100000"):
            with self.subTest(value=value):
                self.run_cli(self.plain, ["--ids", "1", "--expert-profile", profile,
                                          "--pin-experts", value], ok=False)
        self.run_cli(self.plain, ["--ids", "1", "--expert-profile", profile], ok=False)
        self.run_cli(self.plain, ["--ids", "1", "--pin-experts", "1"], ok=False)
        self.run_cli(self.plain, ["--ids", "1", "--expert-profile", profile,
                                  "--pin-experts", "2"], ok=False)

    def test_profile_preflight_precedes_even_config_or_shard_access(self):
        absent_model = self.path / "no-model-directory"
        profile = self.path / "profile.txt"
        for text in (None, "not a profile\n", "K3EXPERTS 1 4294967296 8 2\n",
                     "K3EXPERTS 1 13 4294967296 2\n", "K3EXPERTS 1 13 8 0\n",
                     "K3EXPERTS 1 13 8 65\n", "K3EXPERTS 1 13 8 2\n1 0 2\n1 0 1\n"):
            with self.subTest(text=text):
                if text is not None:
                    profile.write_text(text)
                result = self.run_cli(absent_model, ["--ids", "1", "--expert-profile",
                                                     profile, "--pin-experts", "1"], ok=False)
                self.assertIn("profile", result.stderr)
                self.assertNotIn("config:", result.stdout)
                self.assertNotIn("indexed", result.stdout)


if __name__ == "__main__":
    unittest.main()
