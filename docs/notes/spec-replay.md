# `--spec` drafting policies, replayed on text

**This is a teacher-forced proxy, not a measurement of K3.** The "model output" is
human-written text replayed as if it were K3's greedy output. The ids come from a
byte-level BPE trained here, because K3's tokenizer ships with the checkpoint. A draft
counts as accepted when it equals the next ids of that text. The numbers rank drafting
policies and size them under the verify-step cost model. They are not K3 tokens per
second. Data: [`measurements/spec-replay.json`](../measurements/spec-replay.json).
Tool: `tools/spec_replay.py`. Tests: `tests/test_spec_replay.py`.

## What was replayed

| corpus | prompt | output | eval docs | output ids |
|---|---|---|---:|---:|
| code edits | old version of a `.c/.h/.py/.md` file from this repository's history (up to `49f5ccb`), plus one instruction line with the commit subject | the new version, in full | 135 | 538k |
| code writing | first third of a Python 3.11 standard-library module | the rest | 182 | 483k |
| prose | first third of a markdown document (this repository's docs; npm, Ruby and Go package docs) | the rest | 145 | 196k |
| chat (synthetic) | earlier turns, plus a question about the next section of a package document | that section | 79 | 127k |

The chat corpus is built from documentation, so its answers are prose. No conversation
transcripts were used.

- **Tokenizer.** The proxy is a byte-level BPE with 131,072 ids and a split regex from
  the same family as K3's. It was trained on 55 MB of text disjoint from every
  evaluation document: third-party Python packages, C headers, Perl docs, Go sources,
  and the other half of the package markdown, split by package. Training files that
  shared more than 25% of their lines with any evaluation text were dropped. Output
  runs at 3.7 to 4.0 characters per id.
- **Draft points.** These are the steps the engine would take if its greedy output were
  the text: each step jumps ahead by what it emitted. The first id of each answer comes
  from prefill and is excluded.
- **Calibration split.** 20% of each corpus's documents (chosen by hash) fit P3's table
  and choose P2's defaults. Every number below comes from the other 80%.
- **Intervals.** Brackets are 95% bootstrap intervals over documents, with the same
  resamples for every policy.

## Policies

- **P0 current**: `spec_draft()` exactly, at `--spec 8`. The port is checked against
  hand-worked cases taken from the C source, and against an incremental index over 400
  random sequences.
- **P1 eager**: the most recent occurrence of the suffix 4-gram, then 3-gram, then
  2-gram, with no agreement gate.
- **P2 longest**: the longest suffix match (suffix automaton), taking its most recent
  occurrence. The calibration split chose a minimum match of 5 and N = 8 at 8 GB, and a
  minimum of 6 and N = 4 at 128 GB.
- **P3 adaptive**: P2's draft, cut at the first j where the estimated probability that
  drafts 1..j are *all* accepted falls to δ. δ is 0.22 at 8 GB and 0.55 at 128 GB, the
  marginal cost of one more verified position. The estimate is a 234-cell table over
  (match length + j − 1, how often the matched context occurred and was followed by the
  proposed id, depth), and it reads only the history.
- **P3 cost-aware**: the same estimate with the stop rule taken from the cost model
  *including today's replay*. Draft j is added while
  P_j > S(j+1) − S(j) + (P_{j−1} − P_j)·R(j−1).

Cost model: a verify step costs S(n) = a_T + a_E·u(n) + a_C·n, where n is the number of
drafts plus one and u(n) = 56(1 − (55/56)^n). At 8 GB the coefficients are 0.78, 0.19
and 0.03; at 128 GB they are 0.44, 0.45 and 0.11. The replay R(m) runs only today,
after a partial acceptance. Speedup is emitted ids divided by summed step cost; a
plain step costs 1.

"Today" in the tables means the engine as it was before the replay-free rollback
(commit 1804f7e on this branch), which restored a snapshot and replayed the accepted
prefix through a second forward after every partial acceptance. The engine on this
branch no longer does that, so the "replay-free" column is the one that describes it; the
"today" column is kept to show what the rollback changed. The coefficients are estimates
of a plain decode token's cost shares (trunk bytes, expert bytes, compute) at each memory
tier, not measurements.

## 8 GB (trunk streamed)

| corpus | policy | fires | accepted / drafted | tokens / step | replay-free | today |
|---|---|---:|---:|---:|---:|---:|
| code edits | P0 current | 59% | 91% | 4.64 | 2.54 [2.44, 2.64] | 2.41 [2.30, 2.52] |
|  | P1 eager | 76% | 65% | 4.95 | 2.21 [2.11, 2.31] | 1.88 [1.78, 1.98] |
|  | P2 longest | 55% | 95% | 5.16 | 2.72 [2.59, 2.85] | 2.65 [2.52, 2.79] |
|  | P3 adaptive | 78% | 92% | 5.64 | 2.76 [2.64, 2.88] | 2.54 [2.40, 2.69] |
|  | P3 cost-aware | 62% | 95% | 5.31 | 2.75 [2.63, 2.87] | 2.67 [2.54, 2.80] |
| code writing | P0 current | 21% | 35% | 1.41 | 1.13 [1.10, 1.16] | 1.00 [0.98, 1.02] |
|  | P1 eager | 45% | 18% | 1.64 | 0.94 [0.92, 0.97] | 0.71 [0.70, 0.72] |
|  | P2 longest | 14% | 19% | 1.21 | 0.99 [0.97, 1.01] | 0.87 [0.83, 0.90] |
|  | P3 adaptive | 56% | 44% | 1.59 | 1.24 [1.21, 1.26] | 0.94 [0.92, 0.96] |
|  | P3 cost-aware | 23% | 61% | 1.36 | 1.21 [1.18, 1.24] | 1.11 [1.09, 1.13] |
| prose | P0 current | 10% | 36% | 1.22 | 1.08 [1.06, 1.11] | 1.01 [0.99, 1.04] |
|  | P1 eager | 25% | 16% | 1.31 | 0.93 [0.91, 0.95] | 0.78 [0.76, 0.80] |
|  | P2 longest | 5% | 36% | 1.15 | 1.06 [1.04, 1.08] | 1.01 [0.99, 1.02] |
|  | P3 adaptive | 40% | 42% | 1.33 | 1.14 [1.11, 1.17] | 0.91 [0.88, 0.93] |
|  | P3 cost-aware | 12% | 65% | 1.22 | 1.14 [1.11, 1.17] | 1.08 [1.06, 1.11] |
| chat (synthetic) | P0 current | 9% | 33% | 1.19 | 1.07 [1.04, 1.09] | 1.00 [0.98, 1.02] |
|  | P1 eager | 22% | 16% | 1.28 | 0.95 [0.92, 0.98] | 0.80 [0.78, 0.83] |
|  | P2 longest | 4% | 36% | 1.13 | 1.05 [1.03, 1.07] | 1.00 [0.99, 1.02] |
|  | P3 adaptive | 39% | 41% | 1.30 | 1.13 [1.10, 1.15] | 0.90 [0.88, 0.92] |
|  | P3 cost-aware | 10% | 64% | 1.18 | 1.11 [1.09, 1.14] | 1.07 [1.05, 1.09] |

## 128 GB (trunk resident)

| corpus | policy | fires | accepted / drafted | tokens / step | replay-free | today |
|---|---|---:|---:|---:|---:|---:|
| code edits | P0 current | 59% | 91% | 4.64 | 1.49 [1.45, 1.52] | 1.43 [1.39, 1.47] |
|  | P1 eager | 76% | 65% | 4.95 | 1.19 [1.14, 1.23] | 1.05 [1.01, 1.10] |
|  | P2 longest | 66% | 98% | 3.60 | 1.48 [1.46, 1.50] | 1.47 [1.44, 1.49] |
|  | P3 adaptive | 63% | 98% | 5.21 | 1.60 [1.57, 1.62] | 1.57 [1.53, 1.60] |
|  | P3 cost-aware | 55% | 99% | 4.87 | 1.59 [1.56, 1.62] | 1.57 [1.54, 1.60] |
| code writing | P0 current | 21% | 35% | 1.41 | 0.87 [0.86, 0.88] | 0.76 [0.76, 0.77] |
|  | P1 eager | 45% | 18% | 1.64 | 0.57 [0.56, 0.58] | 0.45 [0.44, 0.46] |
|  | P2 longest | 9% | 37% | 1.14 | 0.95 [0.93, 0.96] | 0.87 [0.84, 0.89] |
|  | P3 adaptive | 24% | 73% | 1.32 | 1.07 [1.05, 1.08] | 0.99 [0.98, 1.00] |
|  | P3 cost-aware | 11% | 87% | 1.20 | 1.07 [1.05, 1.08] | 1.04 [1.03, 1.06] |
| prose | P0 current | 10% | 36% | 1.22 | 0.92 [0.91, 0.94] | 0.85 [0.84, 0.87] |
|  | P1 eager | 25% | 16% | 1.31 | 0.64 [0.62, 0.66] | 0.55 [0.53, 0.57] |
|  | P2 longest | 5% | 57% | 1.11 | 1.01 [1.00, 1.01] | 0.97 [0.95, 0.98] |
|  | P3 adaptive | 13% | 75% | 1.20 | 1.05 [1.03, 1.06] | 1.01 [1.00, 1.02] |
|  | P3 cost-aware | 6% | 87% | 1.13 | 1.04 [1.03, 1.06] | 1.03 [1.02, 1.04] |
| chat (synthetic) | P0 current | 9% | 33% | 1.19 | 0.91 [0.90, 0.93] | 0.85 [0.84, 0.86] |
|  | P1 eager | 22% | 16% | 1.28 | 0.68 [0.65, 0.70] | 0.58 [0.56, 0.60] |
|  | P2 longest | 4% | 58% | 1.09 | 1.01 [1.00, 1.01] | 0.97 [0.97, 0.98] |
|  | P3 adaptive | 11% | 73% | 1.16 | 1.04 [1.03, 1.05] | 1.00 [0.99, 1.01] |
|  | P3 cost-aware | 5% | 84% | 1.10 | 1.03 [1.02, 1.04] | 1.02 [1.01, 1.03] |

## What the proxy says

1. **Edits are where lookup drafting pays.** On edits every gated policy runs at
   2.4 to 2.8x at 8 GB and 1.4 to 1.6x at 128 GB. P0 already takes most of that. P2 and
   P3 add about 0.2 at 8 GB; the paired interval for P3 minus P0 replay-free is
   [+0.19, +0.25].
2. **On all other text the gains are small.** At 8 GB, P0 gives 1.07 to 1.13x
   replay-free and 1.00 to 1.01x today. P3 gives 1.13 to 1.24x replay-free, but its δ
   assumes no replay, so today it loses (0.90 to 0.94x).
3. **The cost-aware rule beats P0 in all 16 corpus × tier × mode cells.** Paired
   differences run from +0.05 to +0.28, and every interval excludes zero. Under today's
   costs it gives 1.07 to 1.11x on non-edit text at 8 GB and 1.02 to 1.04x at 128 GB.
   Replay-free it stays within 0.03 of P3, so it needs no retuning when the rollback
   lands. At 8 GB, P3 with δ raised to 0.55 behaves almost the same; the cost-aware
   rule derives that threshold from the cost model instead of tuning it.
4. **At 128 GB, the current rule at `--spec 8` slows non-edit text down.** It runs
   0.87 to 0.92x replay-free and 0.76 to 0.85x today, because it drafts about 6 ids at
   about 35% acceptance. At `--spec 1` or `2`, P0 is 0.96 to 1.05x.
5. **P1 eager at `--spec 8` loses outside edits at both tiers, with or without
   replay.** At 8 GB it runs 0.93 to 0.95x replay-free and 0.71 to 0.80x today.
   Shorter eager drafts do better replay-free at 8 GB (1.09 to 1.19x at `--spec 1` or
   `2`) but still lose today (0.92 to 0.97x). At `--spec 4`, today, on code writing, it
   is 0.84x. The engine measured an eager drafter at 0.91x on code; the proxy points
   the same way at a similar size.
6. **P3's table is calibrated.** On every corpus, predicted and realized first-draft
   acceptance agree to within 2.2 points. The *cumulative* reading of "P(accept next)
   > δ" matters: stopping on the per-draft conditional probability instead costs 0.28 to
   0.31 outside edits at 8 GB, and 0.08 to 0.17 at 128 GB. P2's copy past the end of the
   history makes no measurable difference.
7. **The result survives a much smaller vocabulary.** A 32,768-id proxy moves every
   speedup by 0.04 or less. A per-verify overhead of eps = 0.02 moves them by 0.02 or
   less.

## What would change these numbers

- **K3's own output.** Greedy decoding often copies context more readily than a human
  author: it quotes code, reuses identifiers and restates the question. That would
  raise acceptance, most of all for long drafts. Paraphrasing where a human copied
  would lower it. To measure this, run the prompts through the engine without `--spec`
  and feed the prompt and emitted ids to `spec_replay.py replay-ids`. That uses K3's
  tokenizer and K3's output, with the recorded P3 table and P2 defaults.
- **K3's tokenizer.** K3 has 163,840 ids, multilingual and trained on far more text.
  The 32k check suggests vocabulary size matters little here, but merges tuned to K3's
  data could still move match lengths.
- **The workload.** The edit corpus assumes whole-file rewrites, which is the best case.
  Diffs or search-and-replace output copy less. Real chat quotes user code and earlier
  answers, which the synthetic chat corpus cannot do.
- **Costs.** These are the estimated cost-share coefficients above, with eps = 0. Any per-step cost of the
  replay-free rollback (snapshots per verified position) enters as eps. The JSON gives
  the headline at eps = 0.02 and says how to recompute for others.
- **Calibration and intervals.** P3's table and P2's defaults come from held-out
  documents of the same four kinds, so a different workload would need its own
  calibration. Successive edits of one file are not independent, so the edit-corpus
  intervals are narrower than they should be.

## Reproduce

```
python3 tools/spec_replay.py run --work /tmp/spec-replay --rev 49f5ccb
python3 -m unittest tests.test_spec_replay
```

The run takes about 3 CPU-minutes and 1 GB of RAM. The tokenizer's training text and
the package-doc corpora come from locally installed packages, so another machine will
get slightly different numbers.
