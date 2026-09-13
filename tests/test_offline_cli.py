"""Whole tiny-model generation parity, prompt repetition and saved sessions.

Build with ZSTD=1 first. Requires numpy and torch for the synthetic checkpoint.
K3_TEST_BIN=bin/zstd/k3 python3 -m unittest discover -s tests -p test_offline_cli.py -v
"""
from __future__ import annotations

import base64
import contextlib
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
