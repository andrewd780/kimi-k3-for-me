# Offline quality evaluation

**Built, not evaluated on K3.** No full checkpoint is available on a permitted
host. The unit tests validate scoring arithmetic and window accounting, not K3
quality. A new lossy model still needs a real evaluation against the exact
checkpoint. Draft acceptance with a BF16 verifier is not that evaluation.

`tools/quality.py` measures teacher-forced corpus perplexity through `bin/k3`.
It uses the local checkpoint and the engine's logits. It neither downloads weights
nor converts them. No network-backed `.k3-remote` model is accepted.

Prepare a held-out JSONL corpus, one independent document per line:

```jsonl
{"id":"prose-1","text":"An actual held-out passage goes here."}
{"id":"tokens-1","token_ids":[3,7,11,5,9]}
```

Text requires local tokenizer files and `make bin/test_tok`. Token IDs require
no tokenizer. No chat template, BOS, EOS or prompt repetition is added implicitly.
Use `--bos-id ID` deliberately if the corpus protocol calls for one; the added
BOS is context, not a scored target. Avoid training/calibration text when assessing
a variant. Preserve the corpus, tokenization and window settings across arms.

On a future machine holding the full checkpoint on local NVMe:

```sh
python3 tools/quality.py \
  --checkpoint /models/Kimi-K3 --trunk /models/Kimi-K3-trunk \
  --prompts heldout.jsonl --tokenizer /models/Kimi-K3 \
  --window 128 --stride 64 --trunk-gb 16 --cache-gb 0.5 \
  --checkpoint-label exact-bf16 --out results/exact.json
```

These are example budgets, not an 8 GB preset or a measured recommendation. The
engine forecasts allocations and refuses a plan above available memory. Larger
windows amortize trunk reads but require more prefill scratch. The current harness
starts a fresh native process for each window and pays binding costs each time.
It is for evaluation, not generation latency measurement.

For a deliberately created variant, use a separate checkpoint/trunk and output:

```sh
python3 tools/quality.py \
  --checkpoint /models/Kimi-K3 --trunk /models/variant-trunk \
  --prompts heldout.jsonl --tokenizer /models/Kimi-K3 \
  --window 128 --stride 64 --trunk-gb 16 --cache-gb 0.5 \
  --baseline results/exact.json --checkpoint-label variant \
  --out results/variant.json
```

The comparison refuses a different token set, vocabulary, BOS choice or window
protocol. It reports change in mean negative log-likelihood (NLL); positive means
the variant assigned lower average probability to the actual targets. This is one
corpus metric, not proof that reasoning or every task remains unchanged.

## Accounting contract

Each document starts fresh. Its first token supplies context. A logit at position
`i` scores the actual token at `i+1`; generated guesses are never fed back. The
first window scores all targets after its first token. Subsequent windows retain
overlapping context and score only new targets. Every original target is scored
once, including the final partial window. Context length varies within each
window; stride therefore changes the protocol and must match between arms.

The native `--score-prompt --score-start N` mode writes full per-target NLL using
double-precision log-sum-exp over the unmodified float logits. Invalid targets,
non-finite logits, incomplete layers and failed expert reads invalidate a run.
Partial stacks, state continuation, speculative generation and prompt rereading
cannot be combined with scoring. No lossy inference setting is enabled here.

Corpus perplexity is `exp(sum(token NLL) / number of scored tokens)`, not an
average of document perplexities. If exponentiation overflows, perplexity is
`null` and the finite mean NLL remains available. Window JSON and logs are kept
beside the final report in `<output-stem>-windows/`; that directory must be new.
A failed evaluation leaves diagnostics there and publishes no new complete report.

Reports include corpus and tokenized-input hashes, tokenizer/binary/config hashes,
local weight file sizes and modification times, and the packed trunk manifest
hash. Reading all 1.56 TB to hash it is not performed. The checkpoint label is
user supplied; the inventory is not cryptographic proof of every weight byte.
Use a pinned, verified checkpoint when an evaluation host becomes available.
