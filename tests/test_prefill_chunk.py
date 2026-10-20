"""Chunked reading-in (--prefill-chunk) and the 1M-id prompt ceiling, on the tiny K3.

Under --incremental the prompt is read in passes of at most --prefill-chunk positions, each carrying the KV cache and the
recurrent state to the next. A position's arithmetic does not depend on which pass it rides in (GATE 3c of
tests/unit/k3_model.c), so every mode must give the bits of one pass over the whole prompt:
  - identity: the same ids, every generated token's logits and the saved state file, byte for byte, at chunk sizes that
    split the prompt evenly, unevenly and into single positions, in every incremental mode (both cache layouts,
    --mla-split, the layer loop fixed and adaptive, --spec, a hybrid draft, a streamed trunk and lm_head, the row
    pipeline, --expert-pipeline), and on a resumed session;
  - the gates can fail: two test-build mutants that leave one pass and decode untouched are caught (K3_PREFILL_MUTANT_RESET
    clears the carried KDA state before every chunk after the first; K3_PREFILL_MUTANT_RELATIVE makes the latent cache's
    causal cut use a query's row in the call instead of its absolute position);
  - a prompt past the old 32,768-id ceiling reads in, and two chunkings of it agree;
  - refusals name their reason (chunk range, --prefill-chunk without --incremental, the one-pass ceiling of full
    recompute, --ids-file problems, an inconsistent state file).
--checkpoint PATH saves the read-in's progress after every chunk but the last, and the same command run again resumes
from it; --pause-after N and Ctrl-C stop after a saved chunk with exit status 3:
  - a read-in paused twice and resumed is the uninterrupted run (ids, logits, saved state), and the checkpoint is byte
    for byte the state a run over the ids it holds saves with --gen 0;
  - Ctrl-C mid read-in pauses at the next chunk boundary, and the resumed run is the uninterrupted one;
  - a checkpoint of another prompt, another loop or another cache layout is refused.

Build first (`make all mutants`).
K3_TEST_BIN=bin/k3 K3_TEST_MUT_BIN=bin/k3-mut python3 -m unittest discover -s tests -p test_prefill_chunk.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_trunk  # noqa: E402

VOCAB = 256
LAYERS = 13
PROMPT = [3, 7, 11, 5, 9, 1, 2, 4, 250, 17, 33, 8, 90, 61, 44, 12, 200, 6, 77, 19, 23]   # 21 ids
CHUNKS = (1, 4, 7, 20)                  # single positions, uneven, uneven, one short of the prompt


class PrefillChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.binary = (ROOT / os.environ.get("K3_TEST_BIN", "bin/k3")).resolve()
        cls.mutant = (ROOT / os.environ.get("K3_TEST_MUT_BIN", "bin/k3-mut")).resolve()
        cls.env = {**os.environ, "OMP_NUM_THREADS": "2", "K3_NOHUGE": "1"}
        for k in ("K3_PREFILL_MUTANT_RESET", "K3_PREFILL_MUTANT_RELATIVE"):
            cls.env.pop(k, None)
        cls.model = cls.root / "plain"
        subprocess.run([sys.executable, str(ROOT / "tools/make_tiny_checkpoint.py"), str(cls.model)],
                       check=True, capture_output=True, env=cls.env)
        cls.trunk = cls.root / "trunk"
        if pack_trunk.main([str(cls.model), str(cls.trunk), str(LAYERS)]):
            raise AssertionError("tiny trunk packing failed")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.counter = 0

    def run_cli(self, args, ids=PROMPT, gen=4, binary=None, env=None, ok=True, rc=2, timeout=180, dump=True):
        self.counter += 1
        out = self.path / ("%d.json" % self.counter)
        logits = self.path / ("%d.f32" % self.counter)
        if isinstance(ids, Path):
            source = ["--ids-file", str(ids)]
        else:
            source = ["--ids", ",".join(map(str, ids))]
        result = subprocess.run([str(binary or self.binary), str(self.model), *source, "--gen", str(gen),
                                 "--cache-gb", "0.0001", "--out", str(out),
                                 *(["--dump-all-logits", str(logits)] if dump else []), *map(str, args)],
                                text=True, errors="replace", capture_output=True,
                                env={**self.env, **(env or {})}, timeout=timeout)
        self.last_stdout = result.stdout
        if not ok:
            self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
            self.assertFalse(out.exists(), "a refused run must not write a report")
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(out.read_text(errors="replace"))
        self.assertEqual(report["expert_drops"], 0)
        raw = np.fromfile(logits, dtype=np.float32)
        self.assertEqual(raw.size, gen * VOCAB)
        return report, raw, result.stdout

    def assert_same(self, a, b, what="every generated token's logits must be bit-identical"):
        self.assertEqual(a[0]["full_ids"], b[0]["full_ids"])
        self.assertEqual(a[1].tobytes(), b[1].tobytes(), what)

    def assert_chunked_is_one_pass(self, mode, ids=PROMPT, gen=4, chunks=CHUNKS):
        """One pass (the default chunk covers the prompt) against each chunking: same ids, same logits, same saved
        state, and the report says how many passes the read-in took."""
        states = [self.path / ("%d.state" % i) for i in range(len(chunks) + 1)]
        one = self.run_cli([*mode, "--save-state", states[0]], ids=ids, gen=gen)
        self.assertEqual(one[0]["prefill_passes"], 1)
        for q, state in zip(chunks, states[1:]):
            with self.subTest(mode=mode, chunk=q):
                got = self.run_cli([*mode, "--prefill-chunk", q, "--save-state", state], ids=ids, gen=gen)
                self.assert_same(one, got)
                self.assertEqual(states[0].read_bytes(), state.read_bytes(),
                                 "the state after a chunked read-in must be the one-pass state, byte for byte")
                self.assertEqual(got[0]["prefill_chunk"], q)
                self.assertEqual(got[0]["prefill_passes"], -(-len(ids) // q))
                self.assertEqual(got[0]["buffer_rows"], max(q, 9))
        return one

    # ------------------------------------------------------------------ identity
    def test_chunked_read_in_is_one_pass_in_every_mode(self):
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.0002"]
        modes = (["--incremental"], ["--incremental", "--kv-latent"],
                 ["--incremental", "--kv-latent", "--mla-split", "3"],
                 ["--incremental", "--loop"],
                 ["--incremental", "--loop-layers", "3-8", "--loop-count", "3"],
                 ["--incremental", "--kv-latent", "--loop"],
                 ["--incremental", *trunk, "--stream-lm-head"],
                 ["--incremental", *trunk, "--trunk-rows"],
                 ["--incremental", "--kv-latent", *trunk, "--trunk-rows", "--kvb-cache", "active"],
                 ["--incremental", "--expert-pipeline"])
        for mode in modes:
            self.assert_chunked_is_one_pass(mode)

    def repeating_ids(self, loop):
        """A prompt a + b + r + a whose model continuation starts with b, so the n-gram drafter proposes from the first
        decode step on (its lookup ends at the pending generated id, so b must be the model's own continuation)."""
        for seed in range(11, 19):          # tests/test_loop_adaptive.py's construction and seeds
            rnd = random.Random(seed)
            a = [rnd.randrange(VOCAB) for _ in range(4)]
            r = [rnd.randrange(VOCAB) for _ in range(6)]
            b = [rnd.randrange(VOCAB) for _ in range(8)]
            for _ in range(8):
                got = self.run_cli(["--incremental", *loop], ids=a + b + r + a, gen=8)[0]["generated_ids"]
                if got[:2] == b[:2]:
                    return a + b + r + a
                b = got
        self.fail("no prompt made the drafter propose under %s" % loop)

    def test_chunked_read_in_is_one_pass_under_spec_and_a_draft(self):
        # verify sweeps that run on a cache a chunked read-in filled; the hybrid draft reads the prompt in chunks of its
        # own
        trunk = ["--trunk", self.trunk, "--trunk-gb", "0.01"]
        for loop, mode in (([], ["--incremental", "--spec", "3"]),
                           (["--loop"], ["--incremental", "--kv-latent", "--spec", "3", "--loop"]),
                           ([], ["--incremental", *trunk, "--draft-trunk", self.trunk])):
            ids = self.repeating_ids(loop)
            one = self.assert_chunked_is_one_pass(mode, ids=ids, gen=10, chunks=(1, 6))
            self.assertGreater(one[0]["spec_sweeps"], 0, "the drafter must have proposed")
        ids = self.repeating_ids([])
        serial = self.run_cli(["--incremental"], ids=ids, gen=10)
        for q in (1, 6):
            with self.subTest(spec_vs_serial=q):
                self.assert_same(serial, self.run_cli(["--incremental", "--spec", "3", "--prefill-chunk", q],
                                                      ids=ids, gen=10))

    def test_a_resumed_session_reads_its_new_prompt_in_chunks(self):
        # with a draft trunk the draft, whose state is not saved, re-reads the whole sequence in chunks too
        draft = ["--trunk", self.trunk, "--trunk-gb", "0.01", "--draft-trunk", self.trunk]
        for mode in (["--incremental"], ["--incremental", "--kv-latent", "--loop"], ["--incremental", *draft]):
            with self.subTest(mode=mode):
                state = self.path / "first.state"
                first = self.run_cli([*mode, "--prefill-chunk", 5, "--save-state", state], gen=3)
                more = PROMPT[::-1]
                whole = self.run_cli(mode, ids=first[0]["full_ids"] + more, gen=3)
                for q in (1, 4, len(more) + 1):
                    resumed = self.run_cli([*mode, "--load-state", state, "--prefill-chunk", q], ids=more, gen=3)
                    self.assertEqual(whole[0]["generated_ids"], resumed[0]["generated_ids"])
                    self.assertEqual(whole[1].tobytes(), resumed[1].tobytes())
                    # the saved pending id plus the new prompt
                    self.assertEqual(resumed[0]["prefill_passes"], -(-(len(more) + 1) // q))

    def test_a_multi_chunk_read_in_reports_its_progress(self):
        _, _, out = self.run_cli(["--incremental", "--prefill-chunk", 8])
        lines = [l for l in out.splitlines() if "prompt read in" in l]
        self.assertEqual(len(lines), 3)
        self.assertIn("8 of 21 positions", lines[0])
        self.assertIn("to go", lines[0])
        self.assertIn("21 of 21 positions (100.0%)", lines[-1])
        self.assertIn("in 3 passes of up to 8", out)

    # ------------------------------------------------------------------ the gates can fail
    def test_the_mutants_leave_one_pass_alone_and_are_caught_chunked(self):
        for name, mode in (("K3_PREFILL_MUTANT_RESET", ["--incremental"]),
                           ("K3_PREFILL_MUTANT_RESET", ["--incremental", "--kv-latent"]),
                           ("K3_PREFILL_MUTANT_RELATIVE", ["--incremental", "--kv-latent"])):
            with self.subTest(mutant=name, mode=mode):
                ref = self.run_cli(mode)
                env = {name: "1"}
                self.assert_same(ref, self.run_cli(mode, binary=self.mutant, env=env),
                                 "a mutant that one pass never exercises must leave it alone")
                bad = self.run_cli([*mode, "--prefill-chunk", 7], binary=self.mutant, env=env)
                self.assertNotEqual(ref[1].tobytes(), bad[1].tobytes(), "%s went unseen" % name)
        for name in ("K3_PREFILL_MUTANT_RESET", "K3_PREFILL_MUTANT_RELATIVE"):
            with self.subTest(production=name):
                err = self.run_cli(["--incremental"], env={name: "1"}, ok=False)
                self.assertIn(name, err)

    # ------------------------------------------------------------------ past the old ceiling
    def test_a_prompt_past_the_old_ceiling_reads_in_and_two_chunkings_agree(self):
        rnd = random.Random(11)
        n = 32_768 + 300
        ids = self.path / "long.ids"
        ids.write_text(",".join(str(rnd.randrange(VOCAB)) for _ in range(n)))
        # The first four layers (KDA, KDA, KDA, MLA) keep this near three minutes on two threads. The two cache layouts
        # give the same bits by design, so the two runs differ in chunking and layout at once.
        a = self.run_cli(["--incremental", "--layers", 4, "--prefill-chunk", 4096], ids=ids, gen=2, timeout=1800)
        b = self.run_cli(["--incremental", "--layers", 4, "--kv-latent", "--mla-split", 512, "--prefill-chunk", 3000],
                         ids=ids, gen=2, timeout=1800)
        self.assertEqual(a[0]["layers_completed"], 4)
        self.assertEqual(len(a[0]["full_ids"]), n + 2)
        self.assertEqual(a[0]["prefill_passes"], -(-n // 4096))
        self.assertEqual(b[0]["prefill_passes"], -(-n // 3000))
        self.assert_same(a, b)
        # full recompute, --score-prompt and --tf-check stay one pass of at most 32,768 ids
        for extra in ([], ["--score-prompt"], ["--tf-check"]):
            with self.subTest(one_pass=extra):
                # (--score-prompt refuses --dump-all-logits before it reads a prompt, so none is asked for)
                self.assertIn("ceiling of one forward pass", self.run_cli(extra, ids=ids, gen=2, ok=False, dump=False))

    # ------------------------------------------------------------------ refusals
    def test_refusals_name_the_reason(self):
        for bad in ("0", "32769", "-3", "4k", ""):
            with self.subTest(chunk=bad):
                self.assertIn("--prefill-chunk", self.run_cli(["--incremental", "--prefill-chunk", bad], ok=False))
        self.assertIn("needs --incremental", self.run_cli(["--prefill-chunk", "4"], ok=False))
        both = self.run_cli(["--ids-file", self.path / "x"], ok=False)
        self.assertIn("mutually exclusive", both)
        nul = self.path / "nul.ids"
        nul.write_bytes(b"1,2\x003")
        self.assertIn("NUL", self.run_cli([], ids=nul, ok=False))
        over = self.path / "over.ids"
        over.write_text(",".join(["1"] * (1_048_576 + 1)))
        self.assertIn("ceiling", self.run_cli(["--incremental", "--kv-latent"], ids=over, ok=False))
        # --reread-prompt doubles the prompt, so it takes half the ceiling
        half = self.path / "half.ids"
        half.write_text(",".join(["1"] * (524_288 + 1)))
        self.assertIn("reread limit", self.run_cli(["--incremental", "--reread-prompt"], ids=half, ok=False))

    def test_an_inconsistent_state_file_is_refused(self):
        state = self.path / "s.state"
        self.run_cli(["--incremental", "--save-state", state], gen=2)
        raw = bytearray(state.read_bytes())
        # K3StateHdr: magic[4], version, fp[12], n_bound, n_mla, cached, nseq, ...
        off = 4 + 4 + 12 * 4 + 4 + 4
        cached, nseq = struct.unpack_from("<ii", raw, off)
        self.assertEqual(nseq, cached + 1, "a run that generated leaves one pending id")
        # two pending ids, more cached positions than ids, a negative count
        for c, s in ((cached, cached + 2), (nseq + 1, nseq), (-1, nseq)):
            with self.subTest(cached=c, nseq=s):
                bad = bytearray(raw)
                struct.pack_into("<ii", bad, off, c, s)
                path = self.path / "bad.state"
                path.write_bytes(bytes(bad))
                err = self.run_cli(["--incremental", "--load-state", path], ids=[7], gen=1, ok=False, rc=1)
                self.assertIn("at most one pending id", err)

    # ------------------------------------------------------------------ pause and resume (--checkpoint)
    def test_a_paused_and_resumed_read_in_is_the_uninterrupted_run(self):
        for mode in (["--incremental"], ["--incremental", "--kv-latent", "--loop"],
                     ["--incremental", "--kv-latent", "--mla-split", "3", "--spec", "2"]):
            with self.subTest(mode=mode):
                ck = self.path / "ck.state"
                ck.unlink(missing_ok=True)
                base = [*mode, "--prefill-chunk", 4]
                whole_state, done_state = self.path / "whole.state", self.path / "done.state"
                whole = self.run_cli([*base, "--save-state", whole_state])
                self.run_cli([*base, "--checkpoint", ck, "--pause-after", 2], ok=False, rc=3)
                self.assertIn("paused: 8 of 21 positions", self.last_stdout)
                # the checkpoint is the state a run over its 8 ids saves with nothing generated
                prefix = self.path / "prefix.state"
                self.run_cli([*mode, "--save-state", prefix], ids=PROMPT[:8], gen=0)
                self.assertEqual(ck.read_bytes(), prefix.read_bytes())
                self.run_cli([*base, "--checkpoint", ck, "--pause-after", 1], ok=False, rc=3)
                done = self.run_cli([*base, "--checkpoint", ck, "--save-state", done_state])
                self.assertEqual(done[0]["checkpoint_resumed_ids"], 12)
                self.assert_same(whole, done)
                self.assertEqual(whole_state.read_bytes(), done_state.read_bytes())
                # the checkpoint is kept, at the last chunk boundary before the end: the same command resumes from it
                again = self.run_cli([*base, "--checkpoint", ck])
                self.assertEqual(again[0]["checkpoint_resumed_ids"], 20)
                self.assert_same(whole, again)

    def test_ctrl_c_pauses_after_the_chunk_and_the_resumed_run_is_the_uninterrupted_one(self):
        rnd = random.Random(3)
        ids = self.path / "ctrl_c.ids"
        ids.write_text(",".join(str(rnd.randrange(VOCAB)) for _ in range(4000)))
        ck = self.path / "ck.state"
        base = [str(self.binary), str(self.model), "--ids-file", str(ids), "--gen", "2", "--cache-gb", "0.0001",
                "--incremental", "--prefill-chunk", "16"]
        p = subprocess.Popen([*base, "--checkpoint", str(ck), "--out", str(self.path / "paused.json")],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
        for line in p.stdout:
            if "prompt read in" in line:        # the read-in has begun: Ctrl-C now pauses rather than stops
                p.send_signal(signal.SIGINT)
                break
        out, err = p.communicate(timeout=600)
        self.assertEqual(p.returncode, 3, out + err)
        self.assertIn("Ctrl-C: pausing", err)
        self.assertIn("Run the same command again", out)
        self.assertFalse((self.path / "paused.json").exists())
        resumed = self.run_cli(["--incremental", "--prefill-chunk", 16, "--checkpoint", ck], ids=ids, gen=2)
        self.assertGreater(resumed[0]["checkpoint_resumed_ids"], 0)
        self.assertLess(resumed[0]["checkpoint_resumed_ids"], 4000)
        self.assertEqual(resumed[0]["checkpoint_resumed_ids"] % 16, 0)
        self.assert_same(self.run_cli(["--incremental", "--prefill-chunk", 16], ids=ids, gen=2), resumed)

    def test_a_checkpoint_of_another_prompt_loop_or_layout_is_refused(self):
        ck = self.path / "ck.state"
        self.run_cli(["--incremental", "--prefill-chunk", 4, "--checkpoint", ck, "--pause-after", 1], ok=False, rc=3)
        other = PROMPT[:2] + [99] + PROMPT[3:]
        err = self.run_cli(["--incremental", "--prefill-chunk", 4, "--checkpoint", ck], ids=other, ok=False)
        self.assertIn("different prompt", err)
        self.assertIn("position 2", err)
        err = self.run_cli(["--incremental", "--checkpoint", ck], ids=PROMPT[:4], ok=False)
        self.assertIn("not a read-in of this prompt", err)
        self.assertIn("loop", self.run_cli(["--incremental", "--loop", "--checkpoint", ck], ok=False, rc=1))
        self.assertIn("--kv-latent", self.run_cli(["--incremental", "--kv-latent", "--checkpoint", ck], ok=False, rc=1))
        # nothing above touched it
        self.assertEqual(self.run_cli(["--incremental", "--checkpoint", ck])[0]["checkpoint_resumed_ids"], 4)

    def test_checkpoint_refusals_name_the_reason(self):
        ck = self.path / "ck.state"
        cases = [(["--checkpoint", ck], "--checkpoint needs --incremental"),
                 (["--incremental", "--pause-after", 2], "--pause-after needs --checkpoint"),
                 (["--incremental", "--checkpoint", ck, "--pause-after", 0], "--pause-after takes"),
                 (["--incremental", "--checkpoint", ck, "--pause-after", "x"], "--pause-after takes"),
                 (["--incremental", "--checkpoint", ck, "--load-state", ck], "give one of them")]
        for args, reason in cases:
            with self.subTest(args=args):
                self.assertIn(reason, self.run_cli(args, ok=False))
        self.assertFalse(ck.exists())


if __name__ == "__main__":
    unittest.main()
