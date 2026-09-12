"""Whole tiny-model generation parity, prompt repetition and saved sessions.

Build with ZSTD=1 first. Requires numpy and torch for the synthetic checkpoint.
K3_TEST_BIN=bin/zstd/k3 python3 -m unittest discover -s tests -p test_offline_cli.py -v
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import offline_model as om  # noqa: E402
import pack_trunk  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
