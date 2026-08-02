# CCAR-F Objective Checklist (Self-Assessment)

Tick each box when you can *explain it out loud* and *justify the trade-off*, not just recognize it. Each objective links to where it's taught in this chapter.

## Domain 1 — Agentic Architecture & Orchestration (27%)
- [ ] Distinguish **workflow** (code-defined path) vs **agent** (model-defined path); default to the simplest. → `AGENTIC_PATTERNS_DEEP_DIVE.md`
- [ ] Match work-shape to pattern: prompt chaining, routing, orchestrator-workers, parallelization (sectioning/voting), evaluator-optimizer, dynamic decomposition. → same
- [ ] Explain orchestrator-workers vs sectioning by *who decides the subtasks*. → same
- [ ] Use **subagents** for isolated context / parallelism; know when a single agent is better. → `AGENT_SDK_DEEP_DIVE.md`
- [ ] State the **agent loop** (gather → act → verify → repeat). → same
- [ ] Pick among **Agent SDK / CLI / Client SDK / Managed Agents**. → same

## Domain 2 — Claude Code Configuration & Workflows (20%)
- [ ] `CLAUDE.md` memory hierarchy (enterprise → user → project → subdirectory) and how levels combine. → `CLAUDE_CODE_DEEP_DIVE.md`
- [ ] `CLAUDE.md` (instructions) vs `settings.json` (behavior/permissions). → same
- [ ] Custom **slash commands**; MCP prompts surface as slash commands. → same
- [ ] **Plan mode** = research/propose, no changes until approval. → same
- [ ] **Hooks** enforce deterministic guardrails vs prompt guidance. → same/`AGENT_SDK`
- [ ] Least-privilege permissions and tool approval. → same

## Domain 3 — Prompt Engineering & Structured Output (20%)
- [ ] Stateless Messages API; system prompt sent every request; attention decays with length. → `CLAUDE_API_DEEP_DIVE.md`
- [ ] Few-shot > prose for format; principles > conditionals for judgment; explicit conditionals for safety-critical. → same
- [ ] Tool use loop; four `tool_choice` modes (auto/any/tool/none). → same
- [ ] **Structured outputs** via JSON Schema (`output_config.format`) vs prompt-only JSON; schema > prose. → same
- [ ] Nullable/optional fields to prevent fabrication; semantic validation; correction-not-blind-retry. → same
- [ ] Extraction tool vs native structured output — the two mechanisms. → same

## Domain 4 — Tool Design & MCP Integration (18%)
- [ ] "Make the right action easy, the wrong action hard." Enums, stable IDs, structured errors, pagination-on-demand. → `AGENTIC_PATTERNS_DEEP_DIVE.md`
- [ ] Tool composition: bundle mechanical steps, keep decision points separate. → same
- [ ] MCP primitives: **tools (model), resources (application), prompts (user)** — and which to use. → `MCP_DEEP_DIVE.md`
- [ ] Annotations are **untrusted hints**, not security. → same
- [ ] Protocol errors vs tool execution errors (`isError: true`). → same
- [ ] MCP scope precedence: **local > project > user**. → same
- [ ] Progressive availability / `list_changed` / tool search. → same

## Domain 5 — Context Management & Reliability (15%)
- [ ] "The model sees a request, not your database." → `CONTEXT_AND_RELIABILITY_DEEP_DIVE.md`
- [ ] Context strategies: sliding window, summarization, structured state, persistent reference, retrieval, tool-result compression, native compaction. → same
- [ ] Relevance beats volume even with huge windows. → same
- [ ] Error classification (transient/validation/business/permission) and correct responses. → `AGENTIC_PATTERNS_DEEP_DIVE.md`
- [ ] **Uncertain writes**: idempotency / check-then-act, never blind retry. → `CONTEXT_AND_RELIABILITY_DEEP_DIVE.md`
- [ ] Confidence calibration → human review; feedback loops. → same
- [ ] Prompt caching: stable prefix first, volatile last; changes invalidate cache. → `CLAUDE_API_DEEP_DIVE.md`

## Cross-cutting "always true" answers
- [ ] Prefer **structure over prose** for reliability.
- [ ] Push each concern to the right owner: **model / application / tool / schema**.
- [ ] Prefer the **simplest** pattern that meets the requirement.
- [ ] Least privilege + human approval for irreversible actions.

If every box is checked and you can defend the trade-off, you're ready. Do a final cold pass of `PRACTICE_QUESTIONS.md`.
