# K3 at full quality and 20 to 50 tokens per second from an 8 GB Mac

## Why this has to be remote

Every decode step on this engine reads about 135 GB of weights (25.83 GB of experts
plus 108.81 GB of trunk, [measured](../PERFORMANCE.md#the-memory-ladder)). To hit
even 1 token per second at that cost you need 135 GB/s of sustained throughput; to
hit 20 tokens/second you need 2.7 TB/s. An M1 Air's RAM delivers about 68 GB/s at
best, which caps it at roughly 0.5 tokens/second even in the impossible case where
the whole 1.56 TB checkpoint sits resident in 8 GB of memory. It cannot: the model
is nearly 200x the RAM. So every token instead streams from the SSD, at around
2 GB/s, which is where this fork's real 8 GB numbers come from (32.7 s/token
measured on a server-class NVMe, worse on a laptop SSD; see
[STATUS.md](../STATUS.md)). No amount of engineering on this codebase changes the
physics: 20 to 50 tokens/second at full quality requires the full checkpoint sitting
in memory that moves multiple terabytes per second, which means HBM on a data-center
accelerator, not a laptop's RAM or SSD. From an 8 GB Mac, the only way to get that
speed is to be a client of a machine that already has it.

## Option A: Moonshot's own API

Moonshot serves Kimi K3 directly. Pricing found today (2026-09-16), reported
consistently across several trackers that cite Moonshot's rate card: **$3.00 per
million input tokens** (cache miss), **$0.30 per million input tokens** (cache hit,
automatic, no cache API), and **$15.00 per million output tokens**. Context length
is **1,048,576 tokens**, one flat rate across the whole window, no long-context
surcharge.
Sources: [BenchLM](https://benchlm.ai/moonshot/api-pricing),
[Morph](https://www.morphllm.com/kimi-api),
[llm-stats.com](https://llm-stats.com/models/kimi-k3), accessed 2026-09-16.

Rate limits are tiered by cumulative account top-up, and the exact numbers per tier
disagree between secondary sources (one says $10 cumulative recharge unlocks 100
RPM / 15 concurrent, another says 200 RPM / 50 concurrent for the same tier). I could
not resolve this discrepancy: the primary docs page,
`https://platform.kimi.ai/docs/pricing/limits`, is blocked by this environment's
network egress policy, and the secondary sources conflict. Treat any specific RPM/TPM
number here as unverified until read from the console directly.
Sources: [Kimi help center](https://www.kimi.ai/help/kimi-api/api-rate-limits),
[kimi-ai.chat rate limits](https://kimi-ai.chat/docs/rate-limits/), accessed
2026-09-16.

```bash
curl https://api.moonshot.ai/v1/chat/completions \
  -H "Authorization: Bearer $MOONSHOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"hello"}]}'
```

## Option B: rent a node that holds the model

The checkpoint needs roughly 1.6 to 2.3 TB of fast accelerator memory in one place:
8x AMD MI355X (288 GB HBM3e each, 2.3 TB total) or a two-node, 16-GPU NVIDIA B200
pod (2x 8x 180 GB, 2.88 TB total). Two prices found today, both for the smaller,
single 8-GPU MI355X node, and both third-party GPU-market trackers rather than a
vendor rate card:

- **Vultr, 8x MI355X**: about $2.65/GPU-hr, roughly $21.20/hr for the node.
  [gpuperhour.com](https://gpuperhour.com/providers/vultr/mi355x),
  [computeprices.com](https://computeprices.com/providers/vultr), accessed
  2026-09-16.
- **Lambda, 8x B200 SXM**: reported between $4.99 and $6.69/GPU-hr depending on
  when the tracker last updated, roughly $40-53/hr for the node; Lambda's own site
  gives a custom quote rather than a fixed public rate.
  [costbench.com](https://costbench.com/software/ai-gpu-cloud/lambda-labs/),
  [gpuperhour.com](https://gpuperhour.com/providers/lambda-labs), accessed
  2026-09-16.

I could not find a public combined price for the larger 16-GPU, two-node B200
configuration; multi-node clusters at that size are generally quoted directly by the
provider, not listed. Renting the hardware is also only step one. Loading the
checkpoint, converting or sharding it for tensor parallelism across 8 or 16
accelerators, and standing up an inference server (vLLM, SGLang, or similar) is real
engineering, unrelated to this repo's streaming engine, and is its own project.

## Hybrid: local draft, remote verify

Speculative decoding: a small (~4B parameter) model runs locally and proposes
tokens; the remote K3 verifies a batch of them in one forward pass and accepts or
rejects each one. Because K3, not the draft, is the only source of an emitted token,
the output distribution is exactly K3's, not an approximation. This fork already
measured a real flight of this idea at 66.7% acceptance
([int8-draft-container.md](int8-draft-container.md)); at that rate the expected
run length per verify step is roughly 1/(1-0.667) ≈ 3, so remote cost per emitted
token drops by roughly that acceptance-length factor if the serving side supports
it. It needs a serving stack that exposes batch verification against draft tokens
(accept/reject per position, ideally with logprobs) — the public chat-completions
API does not offer this. It is only reachable from a custom deployment on option B,
not from option A.

## Decision table

| Option | Cost basis | Speed | Quality | What you must do |
|---|---|---|---|---|
| A. Moonshot API | $3/$15 per M tokens in/out, $0.30 cached in | Provider-side, unmeasured by us; no local hardware ceiling | Exact, Moonshot's own K3 | Get an API key, pay per token; nothing to build |
| B. Rent a node | ~$21-53/hr for a single 8-GPU node (found); larger multi-node unverified | 20-50+ tok/s achievable once served well; unmeasured here | Exact, if you serve the real weights | Rent hardware, then build a whole serving stack: weight sharding, tensor parallelism, an inference server |
| C. Hybrid draft+verify on B | Node rental cost divided by roughly the acceptance-length factor (~3x fewer verify calls at 66.7% acceptance) | Faster than plain remote decode once built | Exact, K3 verifies every emitted token | Everything in B, plus a serving stack that exposes batched draft verification, which nothing public provides today |

Every price above is as found by web search today, 2026-09-16, from the cited URLs.
Anything not cited with a number was not found and is not guessed at.
