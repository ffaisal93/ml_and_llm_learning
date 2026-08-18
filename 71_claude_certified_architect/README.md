# Claude Certified Architect — Foundations (CCAR-F)

> **What this chapter is.** A complete, self-contained study course for the **Claude Certified Architect – Foundations** exam. It is written to *teach*, not just to summarize: every concept starts with plain-English intuition, builds up the mechanism, shows a small worked example, and ends with the trade-off an architect is expected to reason about. Read the chapters in order for a first pass, then drill `PRACTICE_QUESTIONS.md` until you can answer cold.

> ⚠️ **Accuracy note.** Exam logistics and product features change. The numbers below are cross-checked against multiple public study guides as of **August 2026**, but before you book, confirm everything against the **official Exam Guide PDF** on the Anthropic Partner Academy (Skilljar) page. Product behavior is stated from the current Anthropic docs at `platform.claude.com`, `code.claude.com`, and `modelcontextprotocol.io`.

---

## 1. What the certification actually tests

The one-line description Anthropic uses is that a certified architect can *"make informed decisions about trade-offs when implementing real-world solutions with Claude."*

Read that sentence carefully, because it tells you the **flavor** of the exam. This is **not** a coding exam and it is **not** a memorize-the-API exam. It is a *judgment* exam. Almost every question is a short scenario — "you are building X, which has constraint Y" — followed by "what is the best design choice?" The wrong answers are usually things that *work* but are worse on cost, latency, reliability, or maintainability. Your job is to pick the option a thoughtful senior engineer would defend in a design review.

The exam is organized around **four technology pillars**:

| Pillar | One-sentence role | Where it shows up |
|---|---|---|
| **Claude API** (Messages API) | The stateless HTTP interface to the model | System prompts, tools, structured output, batching, caching |
| **Model Context Protocol (MCP)** | Open standard for connecting Claude to external tools/data | Tools vs resources vs prompts, trust boundaries, error semantics |
| **Claude Code** | Agentic coding tool + configuration model | `CLAUDE.md` hierarchy, scopes, slash commands, plan mode |
| **Claude Agent SDK** | Library to build production agents (the Claude Code engine, as a library) | Agent loop, subagents, hooks, sessions, permissions |

---

## 2. Exam logistics (cross-checked, confirm before booking)

| Item | Value |
|---|---|
| Exam code | **CCAR-F** (a.k.a. CCA-F) |
| Questions | **~60** |
| Time | **120 minutes** (~2 min/question) |
| Passing score | **720 / 1000** (scaled score, not raw %) |
| Question types | Multiple-choice and multiple-response (pick-N) |
| Cost | **\$125 USD** |
| Delivery | Pearson VUE — online-proctored **or** test center |
| Credential validity | **12 months** |
| Level | Foundations (entry tier; no formal prerequisites) |

**What "scaled score" means.** 720/1000 is *not* "answer 72% correctly." Anthropic maps your raw correct count through a scoring model that accounts for question difficulty, then reports a 0–1000 scaled number. Practically: aim for **~75–80% raw** on practice tests to give yourself margin.

---

## 3. Domain weightings — where to spend your time

The five scored domains and their approximate weights. Study time should follow the weights.

| # | Domain | Weight | This chapter's file |
|---|---|---|---|
| 1 | **Agentic Architecture & Orchestration** | **27%** | `AGENTIC_PATTERNS_DEEP_DIVE.md`, `AGENT_SDK_DEEP_DIVE.md` |
| 2 | **Claude Code Configuration & Workflows** | **20%** | `CLAUDE_CODE_DEEP_DIVE.md` |
| 3 | **Prompt Engineering & Structured Output** | **20%** | `CLAUDE_API_DEEP_DIVE.md` |
| 4 | **Tool Design & MCP Integration** | **18%** | `MCP_DEEP_DIVE.md`, `AGENTIC_PATTERNS_DEEP_DIVE.md` |
| 5 | **Context Management & Reliability** | **15%** | `CONTEXT_AND_RELIABILITY_DEEP_DIVE.md` |

> Domain 1 (orchestration) is the single biggest slice. If you are short on time, over-invest there and in tool design — together they are ~45% of the exam and share the same underlying mental model ("who is responsible for what: model, application, tool, or schema?").

---

## 4. The five mental models that unlock most questions

If you internalize these five ideas, a large fraction of the exam becomes pattern-matching. They recur in every deep-dive file.

**1. The model has no memory — context is *your* application state.**
The Messages API is **stateless**. Every request re-sends the entire system prompt + message history. "What Claude knows right now" is exactly "what you put in this request." Most reliability questions reduce to: *what did you choose to include, and did it still fit the window?*

**2. Push reliability into structure, not prose.**
A JSON Schema that makes an invalid output *impossible* beats a paragraph politely asking for valid JSON. Prefer `output_config.format` / tool schemas / enums over "please respond in the following format." This is the single most repeated exam theme.

**3. A good tool interface makes the right action easy and the wrong action hard.**
Tool design is UX design *for the model*. Stable IDs, enums, structured errors, pagination-on-demand — all exist so the model falls into the pit of success.

**4. Classify errors before you handle them.**
Transient (retry with backoff) vs validation (return structured details for correction) vs business-rule (non-retryable, surface to user) vs permission (escalate). Blindly retrying a *write* that timed out is the classic wrong answer.

**5. Match the orchestration pattern to the *shape* of the work.**
Fixed steps → prompt chaining. Distinct categories → routing. Coordinator + specialists → orchestrator-workers. Independent partitions → parallel subagents. Path-depends-on-findings → dynamic decomposition. Don't reach for a multi-agent system when a single prompt chain is more reliable and cheaper.

---

## 5. How to study this chapter

**Pass 1 — understand (read in this order):**

1. `CLAUDE_API_DEEP_DIVE.md` — the foundation; everything else assumes the stateless request model.
2. `MCP_DEEP_DIVE.md` — tools / resources / prompts and the trust model.
3. `CLAUDE_CODE_DEEP_DIVE.md` — configuration hierarchy and workflows.
4. `AGENT_SDK_DEEP_DIVE.md` — building agents; the agent loop.
5. `AGENTIC_PATTERNS_DEEP_DIVE.md` — orchestration patterns + tool design (the 27% + 18% core).
6. `CONTEXT_AND_RELIABILITY_DEEP_DIVE.md` — context strategies + error handling.
7. `EXAM_GUIDE.md` — objective-by-objective checklist to self-assess.

**Pass 2 — drill:** work `PRACTICE_QUESTIONS.md`. For every question, say *out loud* why each distractor is wrong. On this exam the distractors teach you as much as the keys.

**Pass 3 — hands-on (highly recommended):** the exam rewards people who have actually built something. Spin up the Agent SDK quickstart, wire one MCP server, and write a `CLAUDE.md`. Thirty minutes of real usage cements a dozen exam facts.

---

## 6. Suggested one-week plan

| Day | Focus |
|---|---|
| 1 | Claude API deep dive + build a tiny tool-use call |
| 2 | MCP deep dive + connect one MCP server in Claude Code |
| 3 | Claude Code configuration + write a `CLAUDE.md` and a slash command |
| 4 | Agent SDK + agentic patterns (the heavy 27%) |
| 5 | Context management & reliability + error-handling patterns |
| 6 | Practice questions, first cold pass; review every miss |
| 7 | Re-read weak files; second practice pass; skim `EXAM_GUIDE.md` checklist |

---

## 7. Further reading (authoritative first)

- **Anthropic Partner Academy** — official exam guide, policies, and enrollment: `https://anthropic-partners.skilljar.com`
- **Claude Docs / API** — `https://platform.claude.com/docs`
- **Claude Code / Agent SDK Docs** — `https://code.claude.com/docs`
- **Model Context Protocol** — `https://modelcontextprotocol.io`
- **Anthropic "Building Effective Agents"** (the canonical agentic-patterns essay) — `https://www.anthropic.com/engineering/building-effective-agents`
- **Prompt engineering guide** — `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview`

The companion cert is in `72_claude_certified_developer/` (the Developer – Foundations exam), which overlaps heavily but is more code-implementation focused. If you are taking both, study them together.
