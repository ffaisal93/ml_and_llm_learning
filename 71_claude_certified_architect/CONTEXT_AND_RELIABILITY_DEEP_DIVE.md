# Context Management & Reliability Deep Dive

> Covers exam domain **Context Management & Reliability (15%)**. Central principle: *"The model sees a request, not your database. You decide what to include."* Reliability is mostly about controlling context and handling failure deliberately.

---

## 1. The core principle

Because the Messages API is **stateless** (see `CLAUDE_API_DEEP_DIVE.md`), the model's entire knowledge on any turn is the request you assembled. Your database, your logs, the previous 40 turns — none of it exists to the model unless you put it in. So **context engineering is application design**: deciding, per request, what to include, what to summarize, and what to leave out.

Two forces pull against each other:
- **Include more** → the model has the facts it needs.
- **Include less** → lower cost, lower latency, and *less distraction* (a bloated context buries the signal and weakens attention to what matters).

Good context strategy resolves this tension deliberately rather than "just append everything."

---

## 2. Context strategies (know each and *when*)

| Strategy | What it does | Best when |
|---|---|---|
| **Sliding window** | Keep only the last N turns | Casual chat where old turns don't matter |
| **Progressive summarization** | Condense old turns into a running summary as they age | Long sessions where early *decisions* matter but verbatim text doesn't |
| **Structured state object** | Maintain an explicit `{decisions, preferences, facts}` blob you update and re-inject | You need durable, machine-updatable memory of key facts |
| **Persistent reference section** | Keep stable content (policies, schema) verbatim in every request | Content that must always be exactly right |
| **Retrieval / fact store (RAG)** | Fetch exact data on demand from an external store | Large or exact data (prices, inventory) that shouldn't live in the prompt |
| **Tool-result compression** | Summarize/trim tool outputs before they re-enter context | Verbose tool results (logs, big JSON) |
| **API-native compaction / context editing** | Let the platform compact or edit context automatically | Long agent runs approaching the window |

**Why not just a bigger window?** Even with a 1M-token window, stuffing everything is worse: it costs more, is slower, and dilutes attention. **Relevance beats volume.** The exam repeatedly prefers "include the *right* context" over "include *all* context."

### Sliding window vs. summarization — the trade-off
- **Sliding window** is cheap and simple but **forgets** anything older than N turns — bad if an early instruction or decision still matters.
- **Summarization** preserves the *gist* of old turns at some fidelity loss and token cost to summarize. Use it when losing early decisions would break the task.

### Structured state beats raw transcript
For preferences/decisions/facts, a **structured state object** you update ("user prefers metric units; chose plan B; deadline = Sept 1") is more reliable and compact than hoping the model re-derives them from a long transcript. Re-inject it each turn as ground truth.

---

## 3. Retrieval and exact data

Never rely on the model to *remember* exact, changing data (a price, a balance). Put it in a **fact store** and retrieve on demand so the number is always current and correct. RAG also handles corpora far larger than any window: retrieve the top-k relevant chunks per query instead of pre-loading the corpus.

Design notes the exam likes:
- Retrieve **exactly what this request needs**, tagged with provenance, so the model can cite sources and you can audit.
- Compress/trim retrieved chunks; don't paste whole documents when a section suffices.

---

## 4. Returning users and long sessions

- **Returning-user pattern:** on a new session, don't replay the entire old transcript — do a **fresh state lookup** (load their structured state / recent summary) and start clean. Cheaper and avoids dragging stale context.
- **System-prompt versioning:** if you change the system prompt across a long-lived session, do it deliberately and note it — an abrupt mid-session change can confuse behavior *and* (as in `CLAUDE_API_DEEP_DIVE.md`) invalidate prompt caching.
- **Reinforce key instructions** at natural breakpoints, because attention to the (still-present) system prompt weakens as history grows.

---

## 5. Reliability = deliberate failure handling

Reliability on this exam is mostly: (a) constrain outputs, (b) classify errors, (c) don't corrupt state, (d) route uncertainty to humans.

### Recap: classify errors (full table in `AGENTIC_PATTERNS_DEEP_DIVE.md`)
- **Transient** → retry with backoff.
- **Validation** → return structured details; request a correction (source + prior output + exact error), not a blind retry.
- **Business rule** → non-retryable; surface to user.
- **Permission** → escalate/approve.

### The uncertain-write problem (memorize)
A write times out — success unknown. **Blind retry risks double-applying** (double charge, duplicate record). Correct designs:
- **Idempotency keys** — the same key applied twice has the effect of once.
- **Check-then-act** — query state before retrying.
- Treat "uncertain" as its own state, not "failed."

### Confidence calibration & human review
- Have the model (or your validation) express **confidence**, and **route low-confidence results to human review** rather than auto-shipping a guess. Especially for extraction/decisions with real consequences.
- **Feedback loops:** track recurring error patterns and fix them upstream (better schema, better tool description, added validation) instead of patching case-by-case.

### Reliability through structure and redundancy
- **Schema-backed output** eliminates a whole class of parse failures.
- **Voting / best-of-N** (from the patterns file) raises reliability for high-stakes single answers.
- **Verification in the loop** (re-run tests, re-check invariants) catches silent failures before they propagate.

---

## 6. Worked example: a durable customer-support agent

**Requirement:** multi-day support conversations, must recall decisions, act on the live order system, never double-refund.

1. **Structured state object:** `{customer_id, open_issue, agreed_resolution, preferences}` — updated each turn, re-injected as ground truth. Not a 3-day raw transcript.
2. **Retrieval, not memory, for live data:** order status/amounts fetched via a tool each time (always current).
3. **Persistent reference:** refund policy kept verbatim (as an MCP **resource**) so eligibility logic is always exact.
4. **Returning-user:** on reconnect, load state + recent summary; skip replaying everything.
5. **Idempotent refunds:** `issue_refund` takes an idempotency key; a timeout never double-refunds.
6. **Human routing:** refunds over a threshold, or low-confidence intent, go to an agent for approval.
7. **Caching:** stable policy + instructions sit in the cached prefix; only the volatile turn changes.

Each choice is a defensible exam answer and traces to the two principles: *manage what the model sees*, and *handle failure deliberately.*

---

## 7. Rapid-fire self-check

1. Complete: "The model sees a request, not your ___." *(database.)*
2. Sliding window's failure mode? *(Forgets anything older than N turns — including early decisions.)*
3. Preferences/decisions: raw transcript or structured state? *(Structured state object.)*
4. Exact, changing data (prices): in-prompt or retrieved? *(Retrieved from a fact store.)*
5. Why not just use a 1M window and include everything? *(Cost, latency, and diluted attention — relevance beats volume.)*
6. Timed-out write — how to retry safely? *(Idempotency key or check-then-act; never blind retry.)*
7. Low-confidence extraction — ship it? *(No — route to human review.)*

---

## 8. Further reading

- Context engineering / windows — `https://platform.claude.com/docs/en/build-with-claude/context-windows`
- Prompt caching — `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- Reducing hallucinations / increasing reliability — `https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations`
- Building Effective Agents — `https://www.anthropic.com/engineering/building-effective-agents`
