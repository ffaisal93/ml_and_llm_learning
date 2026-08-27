# Claude API Deep Dive (Messages, Tools, Structured Output)

> Covers exam domain **Prompt Engineering & Structured Output (20%)** and underpins every other domain. If you only truly understand one file before the exam, make it this one — the stateless request model is the foundation everything else stands on.

---

## 1. The one idea that everything hangs on: the API is *stateless*

Here is the mental model. When you call the Messages API, you send an HTTP request that contains **the entire conversation so far**. Claude reads it, produces one assistant reply, and *forgets everything*. There is no session on Anthropic's side that "remembers" the previous turn. The next turn only continues the conversation because **you** resend the whole history plus the new user message.

Think of it like a brilliant consultant with total amnesia. Every time you talk to them you must hand over the full case file again. What they "know" is exactly what is in the file you handed them this time — nothing more.

**Why this matters (and why the exam loves it):** almost every reliability or context question reduces to *"what did you choose to put in this request, and did it fit?"* If a long conversation starts giving worse answers, it is not because Claude "got tired" — it is because the history grew, older instructions are now far from the model's focus, or you are approaching the context limit.

### Minimal request shape

```json
{
  "model": "claude-sonnet-5",
  "max_tokens": 1024,
  "system": "You are a terse SQL assistant. Only output SQL.",
  "messages": [
    {"role": "user", "content": "customers in Texas"},
    {"role": "assistant", "content": "SELECT * FROM customers WHERE state = 'TX';"},
    {"role": "user", "content": "only their emails"}
  ]
}
```

Three things to notice, each an exam point:

1. **`system` is a top-level field, not a message.** The system prompt is not part of the `messages` array. It sets role, tone, and constraints, and you must send it on **every** request — it is not "sticky" from a previous call.
2. **`messages` alternates `user` / `assistant`.** You reconstruct the whole thread yourself. To continue a conversation you append the model's previous reply as an `assistant` message and then the new `user` message.
3. **`max_tokens` caps the *output*, not the input.** It is the maximum tokens Claude may generate in this reply. Set it too low and the reply is cut off (`stop_reason: "max_tokens"`).

> **Exam trap.** "How does Claude remember earlier turns?" The answer is *it does not* — the application resends history. Any option implying server-side memory between calls is wrong.

---

## 2. System prompt: what it is for, and how it decays

The system prompt defines **who the model is and what rules bind it**: role ("you are a support agent"), tone, hard constraints ("never reveal internal pricing"), and output conventions.

Key behaviors the exam tests:

- It is sent **every request**. There is no first-turn-only special casing.
- **Attention to it weakens as the conversation grows.** Even though you keep sending it, once there are 40 turns of history, an instruction buried in the system prompt competes with a lot of recent text. Mitigation: **reinforce** critical instructions at natural breakpoints (e.g., restate the key rule in the latest user turn or a lightweight reminder), and keep the system prompt focused.
- **General principles beat long conditional lists** for judgment-heavy behavior. "Prefer the least-privilege action and ask before doing anything irreversible" generalizes; a 30-line if/else tree does not, and the model will hit a case you didn't enumerate.
- **Reserve explicit conditionals for safety-critical triggers.** For the one rule that must never be violated ("if the user asks to delete production data, refuse and escalate"), be explicit and rigid. For everything else, principles.
- **Few-shot examples usually outperform prose.** To control *format*, showing two examples of the exact desired output is more reliable than describing it in words.

> **Worked intuition.** You want dates as `YYYY-MM-DD`. Option A: a paragraph explaining ISO 8601. Option B: two examples showing input → `2026-08-02`. Option B wins on the exam and in practice, because the model pattern-matches format from examples far more reliably than it parses instructions about format.

---

## 3. Tool use (a.k.a. function calling)

Tools let Claude *act* — call your code, hit a database, fetch a page. The loop is always the same four beats:

1. You send the request **with a list of tool definitions** (name, description, JSON-Schema `input_schema`).
2. Claude decides to call one and replies with a **`tool_use`** content block: `{name, input, id}`. The `stop_reason` is `"tool_use"`.
3. **Your code runs the tool** and sends the result back as a **`tool_result`** content block (referencing that `id`) in a new `user` message.
4. Claude reads the result and continues — either another `tool_use` or a final text answer.

The model never runs your code. It only *asks* you to. **You** execute and report back. This is the whole agent loop in miniature.

```json
// Claude's reply (step 2)
{"role": "assistant", "content": [
  {"type": "text", "text": "Let me look that up."},
  {"type": "tool_use", "id": "toolu_01A", "name": "get_weather",
   "input": {"city": "Austin"}}
]}
// You send back (step 3)
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_01A",
   "content": "72F, clear"}
]}
```

### `tool_choice` — the four modes (memorize these)

| Mode | Meaning | Use when |
|---|---|---|
| `auto` | Model decides whether to call a tool or answer directly | Default; general assistants |
| `any` | Model **must** call *some* tool (its pick) | You always need a structured action, never free prose |
| `tool` | Model **must** call a **specific named** tool | You know which tool applies; e.g. forcing an extraction tool |
| `none` | Model may **not** call any tool | Turns off tools for a turn |

> **Exam trap.** "You need every response to be a call to your `record_decision` tool." → `tool_choice: {"type": "tool", "name": "record_decision"}`. Not `auto`.

### Signalling errors back to the model

A `tool_result` can carry `is_error: true`. That tells Claude the tool failed so it can react (retry differently, ask the user, or give up gracefully) instead of assuming success. This is distinct from *protocol* errors — see the MCP file for the `isError` distinction.

### Tool design in one line

**A good tool interface makes the right action easy and the wrong one hard.** Concretely: descriptive names, enums for constrained params, stable IDs in outputs, structured errors, and returning a *page* of results on demand rather than auto-dumping 10,000 rows into the context. (Full treatment in [`AGENTIC_PATTERNS_DEEP_DIVE.md`](AGENTIC_PATTERNS_DEEP_DIVE.md).)

---

## 4. Structured output: two different mechanisms (don't confuse them)

The exam explicitly wants you to distinguish **two ways** to get machine-readable output.

### (a) Tool-based extraction ("extraction tool")
You define a tool whose `input_schema` *is* the shape you want, and force it with `tool_choice: {type: "tool", name: ...}`. Claude "calls" the tool; the `input` it produces is your structured data. This has been the classic pattern and works everywhere tools work.

### (b) Native structured outputs (`output_config.format`)
You attach a **JSON Schema** to the request via `output_config.format` and the API constrains the model's decoding so the returned text **conforms to the schema**. The reliability comes from the *decoder being constrained*, not from asking nicely.

**The key exam sentence:** *schema-backed output is more reliable than asking for free-form text that mimics JSON.* Whenever an option says "instruct the model to reply in JSON" versus "constrain output with a schema," the schema option is the better architecture.

### Schema design that prevents fabrication
- Make fields **optional / nullable** when the source may not contain them. If `phone` is required and the document has no phone, the model may **invent** one to satisfy the schema. Allowing `null` lets it honestly say "not present."
- Validate **semantically**, not just syntactically. JSON-Schema-valid ≠ correct. A date can parse and still be wrong. Add downstream checks (ranges, cross-field consistency, provenance).
- On a validation failure, send a **correction request** that includes the source, the previous (bad) extraction, and the *exact* errors — not a blind retry. Blind retries tend to reproduce the same mistake.

---

## 5. Stop reasons and token budgeting

Every response has a `stop_reason`. Know what each means because they drive control flow:

| `stop_reason` | Meaning | Your move |
|---|---|---|
| `end_turn` | Model finished naturally | Done |
| `max_tokens` | Hit your output cap mid-generation | Raise `max_tokens` or continue |
| `tool_use` | Model wants to call a tool | Run it, send `tool_result` |
| `stop_sequence` | Hit a custom stop string you set | Handle per your protocol |
| `refusal` / `pause_turn` | Safety refusal / long-running pause | Handle gracefully |

**Token budgeting intuition.** `input_tokens` grow every turn (you resend history). `output_tokens` are what you pay to generate and what `max_tokens` caps. In a long agent run, input cost dominates — which is exactly why context management and prompt caching matter.

---

## 6. Prompt caching — the cost lever you must understand

Long requests often repeat a big, stable prefix: a large system prompt, a tool catalog, a reference document. **Prompt caching** lets Anthropic cache that prefix so subsequent requests reusing it are much cheaper and faster (cache reads cost a fraction of normal input tokens).

Mechanics and the trade-off the exam wants:

- You mark a **cache breakpoint**; everything **before** it (the prefix) can be cached.
- **Order matters:** put stable content (system prompt, tool defs, long reference docs) *first*, volatile content (the latest user turn) *last*. Caching only helps for an **unchanged prefix**.
- **Any change to the cached prefix invalidates the cache.** Bump one word of the system prompt and the next request re-pays full price to re-cache. So: **don't** interleave a per-request timestamp into the system prompt if you want caching to hold. System-prompt *versioning* is deliberate for this reason.

> **Exam framing.** "You change the system prompt slightly every request to include the current time. Why are cache hit rates low?" → because the cached prefix changed, invalidating it. Move volatile content out of the cached region.

---

## 7. Message Batches API — throughput over latency

For large, **asynchronous** workloads (extract fields from 50,000 documents, classify a backlog), the **Message Batches API** processes many requests offline at a **significant discount (~50%)** with high throughput, at the cost of immediacy (results come back within a window, not instantly).

Decision rule: **interactive/user-facing → normal synchronous call; bulk/offline and latency-tolerant → Batches.** Combine with prompt caching (shared instructions) and constrained schemas for cheap, reliable large-scale extraction.

---

## 8. Streaming, vision, and extended thinking (know the "when")

- **Streaming** (SSE): tokens arrive incrementally. Use for chat UIs where perceived latency matters. Doesn't change *what* the model produces, only *how* you receive it.
- **Vision:** current Claude models accept image input alongside text (documents, screenshots, charts). Output is text.
- **Extended / adaptive thinking:** the model can spend extra internal reasoning tokens before answering. It improves hard multi-step problems at the cost of more tokens and latency. Turn it up for complex planning/derivation; leave it off for simple, latency-sensitive calls. (More in `../72_claude_certified_developer/MODEL_SELECTION_OPTIMIZATION.md`.)

---

## 9. Worked example: designing a reliable extraction endpoint

**Task:** extract `{invoice_number, total, due_date, vendor}` from uploaded invoices, at scale, cheaply, reliably. How does an architect assemble the pieces?

1. **Structured output over prose.** Attach a JSON Schema (`output_config.format`) so the result conforms by construction. Not "reply in JSON."
2. **Nullable fields.** `due_date` and `vendor` are `["string","null"]` — a scanned receipt may lack them; nullability prevents fabrication.
3. **Force determinism where possible.** Low temperature; the task is extraction, not creativity.
4. **Cache the stable prefix.** The long instruction + schema is identical across all invoices → cache it; only the document bytes vary.
5. **Batch the backlog.** 50k historical invoices are not interactive → Message Batches API for the ~50% discount.
6. **Validate semantically.** Check `total` parses as currency and `due_date >= invoice_date`. On failure, issue a correction request with the source + prior output + exact error.
7. **Route low-confidence to humans.** If a field is null or fails validation twice, flag for review instead of shipping a guess.

Every one of those seven choices is a defensible exam answer, and each maps to one of the five mental models from the README.

---

## 10. Rapid-fire self-check

1. Where does the system prompt live in the request, and how often is it sent? *(Top-level `system` field; every request.)*
2. Name the four `tool_choice` modes. *(auto, any, tool, none.)*
3. Why prefer a JSON Schema over "please reply in JSON"? *(Constrained decoding makes invalid output impossible; prose is a request the model can violate.)*
4. Why make extraction fields nullable? *(To let the model say "absent" instead of fabricating.)*
5. You slightly change the system prompt each call and caching stops helping — why? *(Prefix changed → cache invalidated.)*
6. Interactive chat vs. 50k-doc backlog: which API for the backlog? *(Message Batches.)*
7. What does `stop_reason: "tool_use"` require you to do next? *(Execute the tool, return a `tool_result`.)*

---

## 11. Further reading

- Messages API reference — `https://platform.claude.com/docs/en/api/messages`
- Tool use guide — `https://platform.claude.com/docs/en/build-with-claude/tool-use`
- Structured outputs — `https://platform.claude.com/docs/en/build-with-claude/structured-outputs`
- Prompt caching — `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- Message Batches — `https://platform.claude.com/docs/en/build-with-claude/batch-processing`
- Prompt engineering overview — `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview`
