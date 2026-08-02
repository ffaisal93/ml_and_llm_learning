# CCDV-F Practice Questions (with explanations)

> ~33 questions weighted toward the heavy domains (Applications & Integration, Model Selection). Answer first, explain every distractor, then check the key. This is a *developer* exam — expect code-shaped scenarios.

---

## Applications & Integration (heaviest)

**Q1.** Why does a multi-turn conversation retain earlier context?
- A) Anthropic stores the session server-side
- B) Your code resends the full `messages` history each call (stateless API)
- C) The model caches it internally for 24h
- D) `max_tokens` preserves it

**Q2.** `max_tokens=100` and the reply is cut off. `stop_reason` is:
- A) `end_turn`
- B) `tool_use`
- C) `max_tokens`
- D) `stop_sequence`

**Q3.** In the tool-use loop, you continue looping while:
- A) `stop_reason == "end_turn"`
- B) `stop_reason == "tool_use"`
- C) `usage.output_tokens > 0`
- D) the content has text

**Q4.** Which errors should you retry with exponential backoff?
- A) 400, 401, 403
- B) 429, 500, 529
- C) All of them
- D) None — never retry

**Q5.** A request that **charges a card** times out; you can't tell if it succeeded. Best handling?
- A) Retry immediately
- B) Idempotency key or check state before retrying
- C) Assume success
- D) Assume failure and re-charge

**Q6.** You want to reduce **perceived** latency in a chat UI. Use:
- A) Message Batches
- B) Streaming (SSE)
- C) A bigger `max_tokens`
- D) Prompt caching only

**Q7.** 200,000 documents need offline field extraction, not user-facing. Cheapest correct approach?
- A) 200k synchronous calls on Opus
- B) Message Batches (~50% off) + Haiku + a JSON Schema + caching
- C) One request containing all documents
- D) Streaming each document

**Q8.** Prompt caching stopped helping after you added the current time to the system prompt. Why?
- A) Caching expires hourly regardless
- B) The cached prefix changed, invalidating it; keep it stable and put volatile content last
- C) Timestamps can't be cached
- D) You must call a cache API first

**Q9.** You attach an image and text; you should place the instruction:
- A) Before the image
- B) After the image
- C) In the system prompt only
- D) It doesn't matter and images are free

**Q10.** Which field lets you confirm a cache hit and monitor cost?
- A) `stop_reason`
- B) `usage` (incl. `cache_read_input_tokens`)
- C) `id`
- D) `role`

**Q11.** `content[0]` in a response — safe to assume it's text?
- A) Yes, always
- B) No — it may be a `tool_use` (or thinking) block; inspect `block.type`
- C) Only for Haiku
- D) Only when streaming

---

## Model Selection & Optimization

**Q12.** A high-volume, latency-critical, well-scoped classification task. Best model?
- A) Claude Fable 5
- B) Claude Opus 5
- C) Claude Haiku 4.5
- D) Whichever is newest

**Q13.** Complex agentic coding across an enterprise codebase. Default choice?
- A) Haiku 4.5
- B) Opus 5
- C) A non-Claude model
- D) Sonnet 3

**Q14.** Production app needs reproducible behavior and stable evals. Reference the model by:
- A) The alias `claude-sonnet-5`
- B) A pinned dated ID (e.g. `claude-haiku-4-5-20251001`)
- C) "latest"
- D) No model field

**Q15.** Structured extractor gives different output each run. First fix?
- A) Raise temperature
- B) Lower temperature toward 0
- C) Increase `max_tokens`
- D) Switch to streaming

**Q16.** Two levers that cut **cost** on a bulk offline job (pick the best combined answer)?
- A) Streaming + bigger model
- B) Batches + prompt caching (+ a cheaper model)
- C) Higher temperature + top_k
- D) More `max_tokens`

**Q17.** When is spending extra **thinking** budget justified?
- A) Simple key lookups
- B) Hard multi-step reasoning / planning / tricky debugging
- C) To reduce cost
- D) Always, on every call

---

## Agents & Workflows

**Q18.** Steps are fixed and known; you need predictability and low cost. Build a:
- A) Fully autonomous agent
- B) Workflow (code-orchestrated)
- C) Multi-agent swarm
- D) Voting ensemble

**Q19.** Agent SDK vs Client SDK — the difference?
- A) Agent SDK is slower
- B) Agent SDK runs the agent loop for you; Client SDK means you implement the loop
- C) Client SDK can't call tools
- D) They're identical

**Q20.** Orchestrator-workers vs parallel sectioning — the distinguishing question?
- A) Which model is used
- B) Whether subtasks are decided at runtime (orchestrator) or known in advance (sectioning)
- C) Number of tools
- D) Streaming or not

**Q21.** Best reason to run a subtask in a subagent?
- A) It's the only way to call tools
- B) Isolated context + restricted tools (and parallelism)
- C) It's always cheaper
- D) It disables hooks

---

## Prompt & Context Engineering

**Q22.** Most reliable way to control output **format**?
- A) A long prose description
- B) Few-shot examples of the exact output (+ a schema)
- C) Raising temperature
- D) Adding "be precise"

**Q23.** Your extractor invents a value when the source lacks it. Fix?
- A) Add "don't hallucinate" to the prompt only
- B) Make the field nullable/optional in the schema
- C) Raise temperature
- D) Use a bigger model

**Q24.** Even with a 1M-token window, why not include the whole knowledge base every call?
- A) The API rejects it
- B) Cost, latency, and diluted attention — relevance beats volume
- C) It breaks streaming
- D) No reason; always include everything

**Q25.** First structural defense when a prompt includes untrusted web content?
- A) Trust the content
- B) Separate instructions from data — tag the content and treat it as data, not commands
- C) Raise `max_tokens`
- D) Use `tool_choice: any`

---

## Tools & MCPs

**Q26.** What does the model primarily use to decide when to call a tool?
- A) The parameter names
- B) The tool description
- C) The return type
- D) The order in the list

**Q27.** A capability multiple internal apps all need. Best implementation?
- A) Duplicate inline tools in each app
- B) One MCP server, reused across apps
- C) A slash command
- D) A bigger prompt

**Q28.** Signal to the model that a tool failed so it can adapt?
- A) Return empty string
- B) `is_error: true` in the `tool_result`
- C) Raise an HTTP 500
- D) Change `tool_choice`

---

## Security & Safety

**Q29.** An agent can send email and reads arbitrary web pages. Preventing exfiltration via a malicious page relies most on:
- A) Telling the model to ignore malicious instructions
- B) Least privilege + separating data from instructions + human approval before sending
- C) A bigger model
- D) Higher temperature

**Q30.** Where do API keys belong?
- A) In `CLAUDE.md`
- B) In a secrets manager / env vars — never in code or version control
- C) Hardcoded for convenience
- D) In the system prompt

**Q31.** Guarantee "never run a destructive command without confirmation." Use:
- A) A system-prompt sentence
- B) A `PreToolUse` hook that blocks it
- C) A comment in `CLAUDE.md`
- D) Lower temperature

---

## Claude Code + Eval/Debugging

**Q32.** Run Claude Code in CI and parse the result programmatically. Use:
- A) Interactive mode
- B) Headless `-p` with `--output-format json`
- C) A slash command
- D) Streaming to a terminal

**Q33.** Claude "forgot" an instruction after a long chat. Most likely:
- A) Model defect
- B) Integration/context issue (history grew or was truncated); reinforce/manage context
- C) Wrong model
- D) API outage

---

## Answer key & reasoning

1. **B.** Stateless API; you resend history. No server-side session (A/C wrong).
2. **C.** Hitting the output cap → `max_tokens`.
3. **B.** Loop continues while the model keeps asking for tools.
4. **B.** 429/5xx/529 are transient → backoff. 4xx are your bug → fix, don't retry.
5. **B.** Uncertain write → idempotency/check-state; never blind-retry a side effect.
6. **B.** Streaming cuts *perceived* latency. Batches cut cost, not latency.
7. **B.** Bulk offline → Batches + Haiku + schema + caching.
8. **B.** Volatile content invalidated the cached prefix; keep stable-first.
9. **B.** Put the instruction after the image; images cost tokens (not free).
10. **B.** `usage`, including `cache_read_input_tokens`.
11. **B.** A response block may be `tool_use`/thinking; check `block.type`.
12. **C.** Haiku — cheap, fast, high-volume, well-scoped.
13. **B.** Opus 5 is the default for hard agentic coding.
14. **B.** Pin a dated ID for reproducibility.
15. **B.** Lower temperature toward 0 for deterministic structured output.
16. **B.** Batches + caching (+ cheaper model) cut cost.
17. **B.** Thinking budget pays off on hard multi-step reasoning, not simple lookups (and it *adds* cost).
18. **B.** Fixed known steps → workflow.
19. **B.** SDK runs the loop; Client SDK = you write it.
20. **B.** Runtime-decided subtasks (orchestrator) vs pre-known (sectioning).
21. **B.** Context isolation + restricted tools + parallelism (not always cheaper).
22. **B.** Few-shot examples + schema beat prose for format.
23. **B.** Nullable field lets the model report "absent" instead of fabricating.
24. **B.** Relevance beats volume — cost, latency, diluted attention.
25. **B.** Separate instructions from data; treat tagged content as data.
26. **B.** The description is the interface.
27. **B.** One reusable MCP server, not duplicated inline tools.
28. **B.** `is_error: true` in the `tool_result`.
29. **B.** Least privilege + data/instruction separation + human approval. Prompt-pleading alone (A) is weak.
30. **B.** Secrets manager / env; never in code, `CLAUDE.md`, or VCS.
31. **B.** A hook enforces deterministically; prompt text only guides.
32. **B.** Headless `-p` + `--output-format json`.
33. **B.** Usually an integration/context problem, not a model defect.

---

### Scoring guide
- **29–33:** exam-ready.
- **23–28:** solid; re-read the deep-dive behind each miss (weight Applications & Integration).
- **< 23:** another full pass, focusing on Domains 1–2 (~50% of the exam), then retake.
