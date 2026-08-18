# Claude Certified Developer — Foundations (CCDV-F)

> **What this chapter is.** A complete, self-contained study course for the **Claude Certified Developer – Foundations** exam. Written to teach: intuition → mechanism → runnable example → the decision a developer must make. Read in order for a first pass, then drill `PRACTICE_QUESTIONS.md`.

> ⚠️ **Accuracy note.** Numbers below are cross-checked against public study guides as of **August 2026**; confirm against the **official Exam Guide PDF** on the Anthropic Partner Academy (Skilljar) before booking. Product behavior reflects current Anthropic docs (`platform.claude.com`, `code.claude.com`).

> **Relationship to the Architect cert (`../71_claude_certified_architect/`).** The two exams overlap heavily (API, MCP, Claude Code, agents). The **Architect** exam tests *design trade-offs*; the **Developer** exam tests *implementation* — writing the code, choosing models, handling errors, streaming, caching, security. Where a concept is fully developed in the Architect chapter, this one links to it and focuses on the developer's hands-on angle. **If you're taking both, study them together.**

---

## 1. What the certification tests

A certified developer can **build applications with Claude**: integrate the API, choose and optimize models, wire tools and MCP servers, engineer prompts and context, and handle security and debugging — in code.

The tell that this is a *developer* exam: **Applications & Integration is 33.1%** — by far the heaviest single domain on any Claude certification. Expect concrete questions about messages, tools, streaming, vision, caching, batching, error handling, and async patterns.

---

## 2. Exam logistics (cross-checked; confirm before booking)

| Item | Value |
|---|---|
| Exam code | **CCDV-F** (a.k.a. CCD-F) |
| Questions | **53** scored |
| Time | **120 minutes** (~2.3 min/question) |
| Passing score | **720 / 1000** (scaled) |
| Question types | Multiple-choice and multiple-response |
| Cost | **\$125 USD** |
| Delivery | Pearson VUE — online-proctored or test center |
| Validity | **12 months** |
| Level | Foundations (entry tier) |

---

## 3. Domain weightings — spend time by weight

| # | Domain | Weight | This chapter's file |
|---|---|---|---|
| 1 | **Applications & Integration** | **33.1%** | `APPLICATIONS_INTEGRATION_DEEP_DIVE.md` |
| 2 | **Model Selection & Optimization** | **16.8%** | `MODEL_SELECTION_OPTIMIZATION.md` |
| 3 | **Agents & Workflows** | **14.7%** | `AGENTS_AND_WORKFLOWS.md` |
| 4 | **Prompt & Context Engineering** | **11.0%** | `PROMPT_AND_CONTEXT_ENGINEERING.md` |
| 5 | **Tools & MCPs** | **10.6%** | `TOOLS_AND_MCP.md` |
| 6 | **Security & Safety** | **8.1%** | `SECURITY_AND_SAFETY.md` |
| 7 | **Claude Code** | **3.1%** | `CLAUDE_CODE_AND_DEBUGGING.md` |
| 8 | **Evaluation, Testing & Debugging** | **2.6%** | `CLAUDE_CODE_AND_DEBUGGING.md` |

> Domains 1 + 2 are **~50%** of the exam. If time is short, master API integration and model/cost optimization first.

---

## 4. Six things a Claude developer must be fluent in

1. **The Messages API request/response shape** — roles, `system`, content blocks, `stop_reason`, token accounting. Stateless: you resend history.
2. **Tool use loop in code** — send tool defs → get `tool_use` → run it → return `tool_result` → continue.
3. **Model trade-offs** — Opus vs Sonnet vs Haiku on capability/latency/cost; when extended thinking pays off.
4. **Cost/latency levers** — prompt caching, batching, streaming, token/context management, model routing.
5. **Reliability in code** — structured outputs, error handling, retries with backoff, idempotency, async.
6. **Security** — prompt injection defense, untrusted-input handling, PII, key management, least privilege, hooks.

---

## 5. How to study this chapter

**Pass 1 — read in order:** `APPLICATIONS_INTEGRATION_DEEP_DIVE.md` → `MODEL_SELECTION_OPTIMIZATION.md` → `AGENTS_AND_WORKFLOWS.md` → `PROMPT_AND_CONTEXT_ENGINEERING.md` → `TOOLS_AND_MCP.md` → `SECURITY_AND_SAFETY.md` → `CLAUDE_CODE_AND_DEBUGGING.md`.

**Pass 2 — build:** run the Quickstart, make a tool-use call, add streaming, add prompt caching, and read back the `usage` field to *see* the token/cost effects. Nothing cements the integration domain like one real script.

**Pass 3 — drill:** `PRACTICE_QUESTIONS.md`, cold, explaining every distractor.

---

## 6. Suggested one-week plan

| Day | Focus |
|---|---|
| 1–2 | Applications & Integration (the 33%) + build a real script with tools, streaming, caching |
| 3 | Model selection & optimization; measure token usage and cost |
| 4 | Agents & workflows; build a small tool loop / subagent |
| 5 | Prompt & context engineering + Tools & MCP |
| 6 | Security & safety + Claude Code + eval/debugging |
| 7 | Practice questions; review misses; re-read weak files |

---

## 7. Further reading (authoritative first)

- **Anthropic Partner Academy** (official exam/enrollment) — `https://anthropic-partners.skilljar.com`
- **Claude API docs** — `https://platform.claude.com/docs`
- **Client SDKs (Python/TS)** — `https://platform.claude.com/docs/en/api/client-sdks`
- **Agent SDK / Claude Code** — `https://code.claude.com/docs`
- **Prompt engineering** — `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview`
- **Model Context Protocol** — `https://modelcontextprotocol.io`
