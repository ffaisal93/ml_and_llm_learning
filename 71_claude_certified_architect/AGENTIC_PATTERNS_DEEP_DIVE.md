# Agentic Patterns & Tool Design Deep Dive

> Covers the two heaviest technical themes: **Agentic Architecture & Orchestration (27%)** and **Tool Design & MCP Integration (18%)**. Together ~45% of the exam. The meta-skill is *matching the pattern to the shape of the work* and *designing tools that steer the model into correct behavior.*

---

## Part A — Workflows vs. Agents

Start with the most important distinction. Anthropic separates two things people lump together:

- **Workflow:** LLM calls orchestrated through **predefined code paths**. *You* decide the control flow; the model fills in steps. Predictable, testable, cheaper.
- **Agent:** the **model decides its own path** and tool usage dynamically. Flexible, handles open-ended tasks, but less predictable and more expensive.

**The architect's default: prefer the simplest thing that works.** Don't build an autonomous agent when a fixed workflow (or even a single well-crafted prompt) suffices. Reach for agency only when the task genuinely requires the model to decide the path because you *can't* enumerate the steps in advance.

> **Exam mantra.** "What's the *simplest* pattern that meets the requirement?" The elaborate multi-agent answer is usually a distractor when a prompt chain would do.

---

## Part B — The orchestration patterns (know each by its "shape")

Learn each pattern as *"the work looks like ___ → use ___."*

### 1. Prompt chaining — *fixed sequence of known steps*
Decompose into a fixed pipeline: output of step 1 feeds step 2, etc. Optionally add programmatic **gates** between steps (validate before continuing).
*Use when:* the steps are known and stable (e.g., outline → draft → polish; extract → translate → format).
*Trade-off:* higher latency (serial), but each step is simpler and more reliable.

### 2. Routing — *distinct input categories*
A classifier routes each input to a specialized handler/prompt.
*Use when:* inputs fall into clean categories that deserve different handling (billing vs. technical vs. refund; simple query → Haiku, hard query → Opus).
*Benefit:* separation of concerns; cheaper models for easy classes.

### 3. Orchestrator-workers — *coordinator delegates to specialists, subtasks not known upfront*
A lead model **dynamically breaks down** the task, delegates to worker subagents, and **synthesizes** their results.
*Use when:* the number/shape of subtasks depends on the input (e.g., "fix this feature across however many files it touches").
*Key difference from parallelization:* the subtasks are **determined dynamically** by the orchestrator, not fixed by you.

### 4. Parallelization — *independent work you can fan out*
Two flavors:
- **Sectioning:** split into independent subtasks run concurrently (review 10 files at once).
- **Voting:** run the *same* task multiple times for diverse outputs, then aggregate (majority vote, best-of-N) to raise reliability.
*Use when:* subtasks are independent (sectioning) or you want confidence via redundancy (voting).
*Benefit:* wall-clock speed and/or reliability.

### 5. Evaluator-optimizer — *iterative refinement against feedback*
One model generates; another **evaluates and critiques**; the generator revises. Loop until the evaluator is satisfied.
*Use when:* quality has clear criteria and iteration helps (translation quality, code that must pass tests, meeting a rubric).

### 6. Dynamic decomposition — *investigation path depends on findings*
The agent explores, and what it does next depends on what it just learned (debugging: the next file to read depends on the last stack trace).
*Use when:* you cannot pre-plan because the path is discovered as you go — this is where genuine **agency** is warranted.

### Decision cheat-sheet

| The work is… | Pattern |
|---|---|
| A fixed known sequence | Prompt chaining |
| Sorted into categories | Routing |
| Broken into subtasks *you* fix in advance, independent | Parallelization (sectioning) |
| The same task repeated for reliability | Parallelization (voting) |
| Subtasks decided dynamically by a coordinator | Orchestrator-workers |
| Generate → critique → improve | Evaluator-optimizer |
| Path unknown, discovered while doing it | Dynamic decomposition (agent) |

> **Exam trap.** Orchestrator-workers vs. parallelization sectioning: the tell is **who decides the subtasks.** Fixed/known by you → sectioning. Decided at runtime by a lead model → orchestrator-workers.

---

## Part C — Tool design (the 18%, and where agents live or die)

**Governing principle (memorize verbatim):** *A good tool interface makes the right action easy and the wrong action difficult.* Tool design is UX design *for the model.*

### Parameter design
- **Use enums** for constrained choices (`status: "open" | "closed"`) so the model can't pass a garbage value.
- **Stable identifiers** — return and accept durable IDs (`order_id`), not fuzzy natural-language references the model might mangle.
- **Descriptive names + descriptions** — the model picks and uses tools by reading descriptions. A vague description is a bug. Include *when* to use it and what it returns.

### Output design
- Return **structured results with IDs**, not prose the model must re-parse.
- **Paginate on demand.** Return a page + a cursor; let the model ask for more. **Don't auto-fetch all 10,000 rows** into context — it blows the window and buries the signal.
- Keep results **compact**; include only what the next decision needs.

### Tool composition — mechanical vs. decision points
- Bundle **mechanical** multi-step sequences into **one** tool (if steps A→B→C always happen together with no decision between them, one tool `do_ABC` is better than three round-trips).
- Keep **decision points** as separate tools so the model can choose. Don't collapse a real branch into a tool that guesses.

### Progressive availability for large tool sets
Don't expose 200 tools at once — the model mis-picks. Reveal a relevant subset per phase; use tool search / dynamic registration (MCP `list_changed`). **The visible tool set is a design decision.**

### Error handling in tools — classify, don't blindly retry
Tools must return errors the model (or your code) can act on. Classify:

| Error class | Nature | Right response |
|---|---|---|
| **Transient** (network, 503, rate limit) | Temporary infra | **Retry with backoff** (often *inside* the tool) |
| **Validation** (bad input, schema violation) | Permanent, caller's fault | Return **structured details**; model corrects and retries |
| **Business rule** (e.g., refund exceeds policy) | Non-retryable by design | Surface to **user**; don't retry |
| **Permission** (not authorized) | Access boundary | **Escalate** / request approval; don't loop |

> **The classic trap: uncertain writes.** A write (charge a card, create a record) **times out**. Did it succeed? **Do not blind-retry** — you risk a double charge. Design for **idempotency** (idempotency keys) or check state before retrying. This exact scenario appears on the exam.

Also recall the MCP distinction: **protocol errors** (JSON-RPC plumbing) vs **tool execution errors** (`isError: true`, the tool ran but failed its job — feed these to the model to adapt). See `MCP_DEEP_DIVE.md`.

---

## Part D — Responsibility allocation (the exam's favorite lens)

For nearly every scenario, ask: **who should own this concern — the model, the application, the tool, or the schema?**

| Concern | Best owner | Why |
|---|---|---|
| Output validity/shape | **Schema** | Constrained decoding makes invalid impossible |
| What Claude sees / memory | **Application** | Model is stateless; app manages context |
| Correct action affordances | **Tool** | Good interfaces prevent wrong actions |
| Judgment / open-ended path | **Model** | Only the model can adapt to novel inputs |

Pushing a concern to the *right* layer is the through-line of the whole certification.

---

## Part E — Worked example: a research assistant

**Requirement:** answer complex questions by searching internal docs + the web, then writing a cited summary.

- **Shape analysis:** the sub-questions aren't known upfront and depend on findings → lean **orchestrator-workers** with some **dynamic decomposition**, not a rigid chain.
- **Parallelize** independent searches (sectioning) to cut latency.
- **Tools:** `search_docs(query, cursor)` and `web_search(query)` return **paginated, ID-tagged** snippets — not entire documents. Descriptions state when to use each.
- **Voting/verification:** for high-stakes claims, cross-check with a second pass (voting) or an **evaluator** that checks every claim has a citation.
- **Errors:** search timeout = transient → retry with backoff; "no results" = business/empty state → tell the model to broaden the query, not retry identically.
- **Context:** each worker subagent gets its own window; the orchestrator keeps only summaries + citations.

Notice how each decision traces to "match the pattern to the shape" and "make the right action easy."

---

## Part F — Rapid-fire self-check

1. Workflow vs agent — who controls the path? *(Workflow = your code; agent = the model.)*
2. Default bias when choosing a pattern? *(Simplest that works; avoid unnecessary agency.)*
3. Orchestrator-workers vs sectioning — the distinguishing question? *(Are subtasks fixed by you or decided at runtime?)*
4. Why paginate tool output on demand? *(Avoid dumping huge results into context; preserve window and signal.)*
5. A write times out — retry? *(No blind retry; ensure idempotency or check state.)*
6. Four error classes and their responses? *(Transient→backoff; validation→structured detail; business→user; permission→escalate.)*
7. "Make the right action easy and the wrong action hard" describes what? *(Tool interface design.)*

---

## Part G — Further reading

- **Building Effective Agents** (the canonical source for these patterns) — `https://www.anthropic.com/engineering/building-effective-agents`
- Tool use & tool design — `https://platform.claude.com/docs/en/build-with-claude/tool-use`
- Writing effective tools for agents — `https://www.anthropic.com/engineering/writing-tools-for-agents`
- Subagents — `https://code.claude.com/docs/en/agent-sdk/subagents`
- MCP — `https://modelcontextprotocol.io`
