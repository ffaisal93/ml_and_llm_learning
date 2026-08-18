# Model Selection & Optimization (16.8%)

> The developer's cost/latency/quality domain: pick the right model, control tokens and context, tune sampling, and pull the cost levers (caching, batching, routing, thinking budgets). Model facts are current as of **August 2026** — always re-check `platform.claude.com/docs/en/about-claude/models` because the lineup moves.

---

## 1. The current Claude lineup (Aug 2026)

| Model | API ID | Context | Max output | Price (in / out per MTok) | Reach for it when |
|---|---|---|---|---|---|
| **Claude Fable 5** | `claude-fable-5` | 1M | 128k | **$10 / $50** | Absolute peak capability; hardest reasoning; adaptive thinking always on |
| **Claude Opus 5** | `claude-opus-5` | 1M | 128k | **$5 / $25** | Complex agentic coding & enterprise work; the **default** for hard tasks |
| **Claude Sonnet 5** | `claude-sonnet-5` | 1M | 128k | **$3 / $15**¹ | Best **balance** of speed, cost, intelligence; general-purpose workhorse |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | 200k | 64k | **$1 / $5** | **Fastest / cheapest**; latency-critical, high-volume, near-frontier intelligence |

¹ Introductory pricing of **$2 / $10** applies through Aug 31, 2026.

All four take **text + image** input and produce **text**. The big three have **1M-token** context; Haiku has **200k**. Extended/adaptive thinking is available across the line (always-on for Fable 5).

> **The selection heuristic:** start at **Sonnet** for general work or **Opus** for hard agentic/coding tasks; drop to **Haiku** when latency or volume dominates and the task is well-scoped; reach for **Fable** only when you truly need the top of the capability curve. Don't pay Opus/Fable prices for a classification task Haiku nails.

### Aliases vs pinned versions (a developer gotcha)
- An **alias** like `claude-sonnet-5` always points at the latest snapshot — convenient, but behavior can shift when a new snapshot lands.
- A **pinned/dated ID** like `claude-haiku-4-5-20251001` is reproducible — the model won't change under you.
- **Production wants pinned versions** for stability + repeatable evals; use aliases for quick experimentation. Test before adopting a new snapshot.

---

## 2. Routing: use different models for different jobs

The highest-leverage optimization is **not** using one model for everything. Route by difficulty:

- Cheap classifier (Haiku) decides the request's complexity → easy → Haiku answers; hard → escalate to Opus.
- Multi-agent systems: a strong **orchestrator** (Opus) plans; cheap **workers** (Haiku/Sonnet) execute mechanical subtasks.
- Drafting with a small model, then a single strong-model pass to polish.

This is the "routing" pattern from `AGENTS_AND_WORKFLOWS.md`, applied for cost.

---

## 3. Tokens and context windows

- **Tokens ≈ word-pieces.** Rough rule: ~4 characters/token, ~0.75 words/token in English. You pay per input + output token; input dominates in long/agentic runs (you resend history).
- **Context window** = the max tokens (input + output) a request can hold. 1M is large but **not free**: bigger prompts cost more, add latency, and dilute attention. **Relevance beats volume.**
- Watch `usage` on every response to *see* token growth; when it balloons, compact context (see `PROMPT_AND_CONTEXT_ENGINEERING.md`).
- `max_tokens` reserves output room; the effective input budget is window minus your `max_tokens`.

---

## 4. Sampling parameters (know what each does)

| Param | Effect | Set it for |
|---|---|---|
| `temperature` (0–1) | Randomness/creativity. 0 ≈ deterministic; higher = more varied | **Low/0** for extraction, code, classification; **higher** for brainstorming/creative |
| `top_p` (nucleus) | Sample from the smallest set of tokens whose cumulative prob ≥ p | Alternative diversity control; usually tune temp **or** top_p, not both hard |
| `top_k` | Restrict to top-k candidate tokens | Rarely needed; a coarse diversity cap |
| `stop_sequences` | Custom strings that end generation | Enforce output boundaries/protocols |

> **Exam-style:** "Your extractor's output is inconsistent run-to-run." → **lower temperature** (toward 0). Determinism matters for structured tasks.

---

## 5. Extended thinking as an optimization dial

Thinking budget is a **quality-vs-cost** control, not just on/off:

```python
thinking={"type": "enabled", "budget_tokens": 4000}
```

- More budget → better hard multi-step reasoning, more tokens + latency.
- Reserve it for planning, math, tricky debugging, agent decisions. For a simple lookup it wastes money and time.
- Pair a **thinking-heavy** planning step with **cheap** execution steps to get quality where it counts without paying everywhere.

---

## 6. The cost/latency levers (memorize the toolbox)

| Lever | What it saves | Mechanism |
|---|---|---|
| **Right-size the model** | \$\$\$ + latency | Haiku for easy/high-volume; Opus only when needed |
| **Prompt caching** | input cost + latency on repeats | Cache stable prefix; keep it unchanged (`APPLICATIONS_INTEGRATION_DEEP_DIVE.md`) |
| **Message Batches** | ~50% on bulk | Async, latency-tolerant jobs |
| **Context management** | input cost | Summarize/compact; retrieve instead of pre-loading |
| **Model routing** | \$\$\$ | Cheap model triages; escalate only hard cases |
| **Streaming** | perceived latency | Tokens appear immediately (doesn't cut cost) |
| **Tune `max_tokens`** | output cost | Don't reserve 4k when you need 200 |
| **Lower temperature** | fewer retries | Deterministic structured output = fewer re-runs |

> **Streaming vs batching confusion (common trap):** streaming cuts *perceived* latency for interactive use; batching cuts *cost* for bulk async. They solve different problems.

---

## 7. Worked example: optimize a support classifier

**Requirement:** classify 2M support tickets/month into 20 categories, cheaply and reliably.

1. **Model:** Haiku 4.5 — classification is easy and volume is huge; Opus would be wasteful.
2. **Structured output:** a schema with an **enum** of the 20 categories → the model can't emit an off-list label.
3. **Temperature 0:** deterministic, repeatable classifications.
4. **Prompt caching:** the long taxonomy + instructions are identical every call → cache the prefix; only the ticket text varies.
5. **Batching:** it's offline/bulk → Message Batches for ~50% savings.
6. **Pin the model version** for stable evals month over month.

Result: near-cheapest possible pipeline that's also reproducible. Each lever is a distinct exam-worthy decision.

---

## 8. Rapid-fire self-check

1. Default model for a general task vs a hard agentic-coding task? *(Sonnet vs Opus.)*
2. Highest-volume, latency-critical, well-scoped task? *(Haiku.)*
3. Alias vs pinned model ID — which for production? *(Pinned/dated, for stability.)*
4. Output varies run-to-run on a structured task — fix? *(Lower temperature toward 0.)*
5. Does streaming reduce cost? *(No — perceived latency only.)*
6. Two levers that cut *cost* on a bulk offline job? *(Batches + caching + a cheaper model.)*
7. Why is a 1M window not a license to include everything? *(Cost, latency, diluted attention.)*
8. When is extra thinking budget worth it? *(Hard multi-step reasoning/planning; not simple lookups.)*

---

## 9. Further reading

- Models overview & IDs — `https://platform.claude.com/docs/en/about-claude/models/overview`
- Choosing a model — `https://platform.claude.com/docs/en/about-claude/models/choosing-a-model`
- Pricing — `https://platform.claude.com/docs/en/about-claude/pricing`
- Context windows — `https://platform.claude.com/docs/en/build-with-claude/context-windows`
- Extended thinking — `https://platform.claude.com/docs/en/build-with-claude/extended-thinking`
- Reducing latency — `https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency`
