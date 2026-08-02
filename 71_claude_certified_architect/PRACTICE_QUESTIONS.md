# CCAR-F Practice Questions (with explanations)

> ~30 scenario questions in the style of the exam. Cover the answer, then reveal. For every one, force yourself to say *why the wrong options are wrong* — on this exam the distractors are "works but worse," and spotting that is the skill. Answers and reasoning follow each block.

---

## Domain 1 — Agentic Architecture & Orchestration

**Q1.** A pipeline always runs the same three steps: extract entities → validate against a DB → format a report. Best pattern?
- A) Autonomous agent with dynamic decomposition
- B) Prompt chaining with a validation gate
- C) Orchestrator-workers
- D) Voting with best-of-5

**Q2.** You must fix a bug that may touch an unknown number of files; the coordinator should decide what to delegate at runtime. Pattern?
- A) Parallelization (sectioning)
- B) Routing
- C) Orchestrator-workers
- D) Prompt chaining

**Q3.** You want to embed an agent loop with file/command tools and context management inside your Python backend **without writing the loop yourself**. Choose:
- A) Client SDK (Anthropic API SDK)
- B) Claude Agent SDK
- C) Managed Agents
- D) Claude Code CLI, interactive

**Q4.** Which is the strongest reason to give a subtask its **own subagent**?
- A) It always reduces token cost
- B) It isolates context and can restrict the tool set, keeping the main thread clean
- C) The model can't call tools otherwise
- D) It disables the agent loop

**Q5.** Incoming messages are clearly "billing," "technical," or "account." Cheapest reliable design?
- A) One giant prompt handling all three
- B) Routing to three specialized handlers (and cheaper models for easy classes)
- C) Orchestrator-workers spawning subagents per message
- D) Evaluator-optimizer loop

---

## Domain 2 — Claude Code Configuration & Workflows

**Q6.** Coding conventions that must apply to **everyone** who clones the repo go in:
- A) `~/.claude/CLAUDE.md`
- B) `CLAUDE.local.md`
- C) Root `CLAUDE.md`, committed
- D) `settings.local.json`

**Q7.** A rule that must **always** hold ("never run `terraform destroy` without confirmation"). Most reliable enforcement?
- A) A sentence in the system prompt
- B) A `PreToolUse` hook that blocks it
- C) A note in `CLAUDE.md`
- D) A slash command

**Q8.** You want Claude Code to **research and propose** a big refactor but make **no edits** until you approve. Use:
- A) `tool_choice: none`
- B) Plan mode
- C) A subagent
- D) Voting

**Q9.** `CLAUDE.md` vs `settings.json`:
- A) Both store secrets
- B) `CLAUDE.md` = instructions/context; `settings.json` = behavior/permissions
- C) `settings.json` = coding style; `CLAUDE.md` = tool permissions
- D) They are interchangeable

**Q10.** The team should get two MCP servers automatically on clone. Configure them at which scope?
- A) local
- B) user
- C) project (committed `.mcp.json`)
- D) enterprise only

---

## Domain 3 — Prompt Engineering & Structured Output

**Q11.** After 40 turns, Claude starts ignoring a rule that is still in the system prompt every request. Best explanation + fix?
- A) The API dropped the system prompt; resend it — it isn't being sent (it is)
- B) Attention to it weakens as history grows; reinforce the rule at a breakpoint
- C) The model has server-side memory that expired
- D) Lower `max_tokens`

**Q12.** You need **every** response to be a call to your `log_decision` tool. Set:
- A) `tool_choice: "auto"`
- B) `tool_choice: {"type": "any"}`
- C) `tool_choice: {"type": "tool", "name": "log_decision"}`
- D) `tool_choice: "none"`

**Q13.** Most reliable way to guarantee machine-parseable output?
- A) Ask "reply only in JSON" in the system prompt
- B) Provide a JSON Schema via `output_config.format` (constrained decoding)
- C) Add "no prose" to every user turn
- D) Post-process with regex

**Q14.** Your extractor sometimes **invents** a phone number when the document has none. Best fix?
- A) Raise temperature
- B) Make `phone` nullable/optional in the schema so "absent" is representable
- C) Add "don't make things up" to the prompt
- D) Switch to a bigger model

**Q15.** A structured extraction passes JSON-Schema validation but the `due_date` is wrong. What does this teach?
- A) Schemas are useless
- B) Syntactic validity ≠ semantic correctness; add semantic checks + provenance
- C) Always retry blindly
- D) Use prose instead of schema

**Q16.** To control the **format** of answers, which is usually most effective?
- A) A long prose description of the format
- B) Two or three few-shot examples of the exact output
- C) Raising `max_tokens`
- D) Setting `tool_choice: none`

---

## Domain 4 — Tool Design & MCP

**Q17.** The three MCP primitives are controlled by, respectively:
- A) Tools—user, Resources—model, Prompts—app
- B) Tools—model, Resources—application, Prompts—user
- C) All three—the model
- D) Tools—app, Resources—user, Prompts—model

**Q18.** A refund-policy PDF the agent should consult before deciding eligibility should be exposed as a:
- A) Tool
- B) Resource
- C) Prompt
- D) Hook

**Q19.** An MCP tool is annotated `readOnlyHint: true`. Can you auto-approve it as safe?
- A) Yes — annotations are authoritative
- B) No — annotations are untrusted hints; enforce trust on the client (permissions/sandbox/approval)
- C) Only for local servers
- D) Only if `idempotentHint` is also true

**Q20.** A `search_orders` tool would return 50,000 rows. Best output design?
- A) Return all rows so the model has everything
- B) Return a page + cursor; let the model request more on demand
- C) Return a prose summary of all rows
- D) Refuse if > 1000 rows

**Q21.** Difference between a JSON-RPC protocol error and a tool result with `isError: true`?
- A) None
- B) Protocol error = plumbing/transport failure; `isError` = the tool ran but its task failed (feed to the model to adapt)
- C) `isError` means the server crashed
- D) Protocol errors are shown to the user only

**Q22.** You have 150 tools; the model keeps mis-selecting. Best remedy?
- A) Put all 150 in every request with longer names
- B) Progressive availability — expose a relevant subset per phase; use tool search / `list_changed`
- C) Remove tool descriptions to save tokens
- D) Force `tool_choice: any`

---

## Domain 5 — Context Management & Reliability

**Q23.** Complete the principle: "The model sees a request, not your ___."
- A) prompt
- B) database
- C) tools
- D) schema

**Q24.** A 3-day support conversation must remember the agreed resolution and preferences. Best mechanism?
- A) Resend the full 3-day transcript each turn
- B) A structured state object (decisions/preferences/facts) updated and re-injected
- C) Rely on the model's memory
- D) A sliding window of the last 2 turns

**Q25.** A charge-card tool call **times out**; you don't know if it succeeded. Correct handling?
- A) Immediately retry the charge
- B) Use an idempotency key or check state before retrying; treat "uncertain" as its own state
- C) Assume success and continue
- D) Assume failure and refund

**Q26.** You have a 1M-token window, so you decide to include the entire knowledge base in every request. Problem?
- A) None — bigger context is always better
- B) Higher cost, higher latency, and diluted attention; relevance beats volume
- C) The API will reject it
- D) It disables caching only

**Q27.** Your system prompt embeds the current timestamp each call, and prompt-cache hit rate is near zero. Why?
- A) Caching is disabled by default
- B) The cached prefix changes every request, invalidating the cache; move volatile content to the end
- C) Timestamps are not cacheable data types
- D) `max_tokens` is too low

**Q28.** 50,000 historical invoices need field extraction; it's not user-facing. Cheapest correct approach?
- A) 50,000 synchronous calls
- B) Message Batches API (async, ~50% cheaper) + prompt caching + a JSON Schema
- C) One request with all 50,000 invoices
- D) Streaming responses

**Q29.** An extraction pipeline keeps making the *same* mistake on a field. Best long-term response?
- A) Retry each failure a second time
- B) A feedback loop: fix upstream (schema, tool description, validation) so the class of error stops
- C) Raise temperature
- D) Ignore it below a threshold

**Q30.** For a high-stakes single answer where correctness matters more than cost, which raises reliability?
- A) Lower `max_tokens`
- B) Voting / best-of-N with aggregation, plus verification in the loop
- C) Remove the system prompt
- D) Disable tools

---

## Answer key & reasoning

1. **B.** Fixed known steps → prompt chaining; a gate validates before formatting. An agent (A) adds needless nondeterminism; C/D are overkill for a fixed sequence.
2. **C.** Subtasks decided *at runtime by a coordinator* = orchestrator-workers. Sectioning (A) is for *pre-known* independent partitions.
3. **B.** Agent SDK runs the loop in your process. Client SDK (A) means you write the loop; Managed Agents (C) is hosted/async; CLI (D) is interactive.
4. **B.** Isolation + restricted tools + clean main context. It does *not* always cut cost (A) — coordination has overhead.
5. **B.** Routing with cheaper models for easy classes is the classic cost-efficient answer.
6. **C.** Team-wide, shared → committed root `CLAUDE.md`. A/B are personal; D is personal behavior settings.
7. **B.** Must-always-hold → deterministic hook, not probabilistic prompt text.
8. **B.** Plan mode = research/propose, no mutations until approval.
9. **B.** Instructions/context vs behavior/permissions. Never store secrets in `CLAUDE.md`.
10. **C.** Project scope (committed `.mcp.json`) ships to everyone on clone. local/user are per-machine.
11. **B.** It *is* still sent; attention decays with length. Reinforce at a breakpoint. There is no server-side memory (C is wrong on the API model).
12. **C.** Force a specific named tool. `any` (B) forces *some* tool, not that one.
13. **B.** Constrained decoding via schema. Prose requests (A/C) can be violated; regex (D) is a band-aid.
14. **B.** Nullability lets the model say "absent" instead of fabricating. Prompt pleading (C) is weaker.
15. **B.** Valid-shape ≠ correct-value; add semantic validation + provenance.
16. **B.** Few-shot examples control format better than prose.
17. **B.** Tools—model, Resources—application, Prompts—user. Memorize this.
18. **B.** Reference material to consult = resource.
19. **B.** Annotations are advisory; enforce trust on the client.
20. **B.** Paginate on demand; never dump 50k rows into context.
21. **B.** Plumbing failure vs task failure; the latter is fed to the model to adapt.
22. **B.** Progressive availability / tool search; the visible tool set is a design choice.
23. **B.** "…not your database. You decide what to include."
24. **B.** Structured state object beats replaying transcript or trusting model memory.
25. **B.** Idempotency / check-then-act; never blind-retry an uncertain write.
26. **B.** Relevance beats volume; big windows still cost, slow, and dilute attention.
27. **B.** Volatile content in the cached prefix invalidates the cache; keep stable-first, volatile-last.
28. **B.** Batches (async, ~50% off) + caching + schema. Bulk, non-interactive → Batches.
29. **B.** Feedback loop fixes the error class upstream; retrying (A) just re-hits it.
30. **B.** Voting/best-of-N + in-loop verification trade cost for reliability.

---

### Scoring guide
- **27–30:** exam-ready. Do a light review of any missed domain.
- **21–26:** solid; re-read the deep-dive files behind your misses.
- **< 21:** do another full read pass, focusing on Domains 1 and 4 (the heavy, concept-dense ones), then retake.
