# Claude Code (3.1%) + Evaluation, Testing & Debugging (2.6%)

> Two small domains (~5.7% combined) bundled here. The Claude Code part is the developer's feature view (config in the Architect chapter goes deeper); the eval/debugging part is about diagnosing failures and measuring quality.

---

## Part 1 — Claude Code for developers (3.1%)

Claude Code is Anthropic's agentic coding tool (terminal + IDE) and the engine behind the Agent SDK. The developer-facing building blocks:

| Feature | What it is |
|---|---|
| **`CLAUDE.md` (memory)** | Auto-loaded project/user instructions & conventions. Hierarchy: enterprise → user → project → subdirectory (more specific wins). |
| **`settings.json`** | Behavior & permissions (tool allow/deny, approvals). User / project / local / enterprise-managed layers. |
| **Rules** | Guardrails/conventions Claude Code follows (expressed via `CLAUDE.md` / settings). |
| **Skills** | Packaged reusable capabilities (`SKILL.md` + optional scripts) auto-loaded when relevant; from `.claude/` and `~/.claude/`. |
| **Commands (slash commands)** | Saved prompt workflows invoked with `/name`, from `.claude/commands/`. MCP prompts appear as slash commands. |
| **Agents (subagents)** | Specialized agents with isolated context & tool sets for focused subtasks. |
| **Agent Memory** | Durable context persisted across turns/sessions (files/state) so long or returning work keeps what matters. |
| **Sessions** | Persist, **resume**, or **fork** a conversation; long sessions compact/summarize to fit the window. |

### Headless & streaming modes (developer-specific)
- **Headless mode:** run Claude Code non-interactively — `claude -p "prompt"` — for scripting, CI, and automation. Add `--output-format json` (or `stream-json`) for machine-parseable output. This is also how you drive the agent loop from a **non-Python/TS language** (run the CLI as a subprocess).
- **Streaming mode:** stream results incrementally (e.g., `--output-format stream-json`) for real-time consumption in pipelines/UIs.

> **Exam framing.** "Run Claude Code inside CI and parse the result." → **headless** (`-p`) with `--output-format json`. "Give the team a `/deploy-checklist` workflow." → a **custom slash command** committed under `.claude/commands/`.

Config precedence and `CLAUDE.md` vs `settings.json` are detailed in `../71_claude_certified_architect/CLAUDE_CODE_DEEP_DIVE.md`.

---

## Part 2 — Evaluation, Testing & Debugging (2.6%)

### Diagnose: is it a model problem or an integration problem?
The first debugging question on any Claude app:

- **Integration issue** — your code/config: wrong request shape, context you forgot to include, a tool that threw, a parsing bug, a bad retry, an expired key, truncation from a low `max_tokens`. **Most "the model is dumb" bugs are actually integration bugs.**
- **Model-output issue** — the request was correct but the content is wrong: hallucination, missed instruction, poor reasoning. Fix with prompt/context/model changes (better examples, more relevant context, a stronger model, extended thinking).

> **Exam framing.** "Claude 'forgot' an earlier instruction." Usually an **integration** issue — the history grew and the instruction lost salience, or you truncated context — not a model defect. Fix by reinforcing/managing context.

### Trace analysis
Debug by inspecting the full trace: the **exact request** (system + messages + tools), each **`tool_use`** and the **`tool_result`** returned, the **`stop_reason`**, and **`usage`**. Failures usually reveal themselves here:
- `stop_reason: "max_tokens"` → truncated; raise the cap or continue.
- A tool returned an error/empty result → the model reacted to bad data.
- Context ballooned in `usage` → bloat/near-limit; compact.
- The model never saw a fact → you didn't include it (stateless!).

Log inputs, outputs, tool calls, and token usage so traces are reconstructable — you can't debug what you didn't capture.

### Evaluation (measuring quality)
- **Build an eval set:** representative inputs with expected outputs/criteria. Run it on every prompt/model change to catch regressions — don't eyeball a few examples.
- **Grading:** exact-match/programmatic checks where possible; **LLM-as-judge** (a model scores outputs against a rubric) for open-ended quality; human review for high-stakes.
- **Pin model versions** during evals so results are reproducible (aliases can shift under you).
- **Test the pipeline, not just the prompt:** schema validation, error paths, retries, and tool behavior all need tests.

### Common failure modes & fixes (quick table)

| Symptom | Likely cause | Fix |
|---|---|---|
| Output cut off | `max_tokens` too low | Raise cap / stream / continue |
| Inconsistent structured output | temperature too high; no schema | temp→0; add JSON Schema |
| Invented fields | required fields with missing source | make fields nullable |
| "Ignores" a rule late in chat | context growth / bloat | reinforce rule; compact context |
| Intermittent 429/5xx failures | no/naive retry | exponential backoff + jitter |
| Double side effects | blind retry of a write | idempotency key / check state |
| Slow responses | oversized context / big model | trim context; right-size model; stream |

---

## 3. Rapid-fire self-check

1. First triage question when output is wrong? *(Integration issue or model-output issue?)*
2. Where do you look to debug a tool-using turn? *(The trace: request, `tool_use`/`tool_result`, `stop_reason`, `usage`.)*
3. Run Claude Code in CI with parseable output? *(Headless `-p` + `--output-format json`.)*
4. Why pin model versions for evals? *(Reproducibility — aliases can change.)*
5. Grade open-ended quality at scale? *(LLM-as-judge against a rubric, plus human spot-checks.)*
6. "Claude forgot an instruction" — model or integration? *(Usually integration: context growth/truncation.)*
7. Where do Skills and slash commands load from? *(`.claude/` in the project and `~/.claude/`.)*

---

## 4. Further reading

- Claude Code overview — `https://code.claude.com/docs/en/overview`
- Headless mode — `https://code.claude.com/docs/en/headless`
- Slash commands — `https://code.claude.com/docs/en/slash-commands`
- Skills — `https://code.claude.com/docs/en/skills`
- Create strong empirical evals — `https://platform.claude.com/docs/en/test-and-evaluate/develop-tests`
- Reduce hallucinations — `https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations`
- Companion — `../71_claude_certified_architect/CLAUDE_CODE_DEEP_DIVE.md`
