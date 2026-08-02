# Agents & Workflows (14.7%)

> Building agents in code: when to use a workflow vs an agent, constructing agents with the Claude Agent SDK, manager/supervisor hierarchies, and subagents. Deep conceptual treatment of the patterns lives in `../71_claude_certified_architect/AGENTIC_PATTERNS_DEEP_DIVE.md` and `AGENT_SDK_DEEP_DIVE.md`; this file is the developer's build-focused view.

---

## 1. Workflow vs. agent — the decision, in code terms

- **Workflow:** *your code* orchestrates a fixed sequence of LLM calls. Predictable, testable, cheaper, easier to debug.
- **Agent:** *the model* decides the next step and which tools to call, in a loop, until done. Flexible for open-ended tasks; less predictable, more expensive.

**Decision criteria (memorize):**

| Choose a workflow when… | Choose an agent when… |
|---|---|
| Steps are known and stable | The path can't be enumerated in advance |
| You need predictability/testability | The task needs runtime adaptation to findings |
| Cost/latency must be tight | Flexibility is worth the extra cost |
| The task is narrow | The task is open-ended/exploratory |

> **Default bias:** the simplest thing that works. A single prompt < a workflow < an agent, in that order of preference. Reach for agency only when you must.

---

## 2. Building an agent with the Agent SDK

The Agent SDK is *Claude Code as a library* (Python/TS) — it runs the **agent loop** (gather context → act → verify → repeat) so you don't hand-roll it. Minimal shape (Python):

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt="You are a bug-fixing agent. Run tests after every edit.",
    allowed_tools=["Read", "Edit", "Bash"],       # least privilege
    permission_mode="acceptEdits",                 # or require approval
)
async for message in query(prompt="Fix the failing test in cart.py", options=options):
    print(message)
```

You configure, the SDK loops. Key knobs (all detailed in the Architect SDK file):
- **Built-in tools** (read/write/edit/bash/web), plus your custom tools and MCP servers.
- **Permissions** — which tools auto-run vs. need approval (least privilege).
- **Hooks** — deterministic code at lifecycle points (block a command, run a linter). Enforce *must-hold* rules with hooks, not prompt text.
- **Sessions** — persist/resume/fork context for long or returning-user runs.
- **Subagents** — spawn specialists with isolated context.

**SDK vs Client SDK vs Managed Agents vs CLI** — the four-way choice is a likely question; see `../71_claude_certified_architect/AGENT_SDK_DEEP_DIVE.md` §2. Short version: Agent SDK runs the loop for you; Client SDK = you write the loop; Managed Agents = hosted/async; CLI = interactive terminal.

---

## 3. Manager / supervisor hierarchies (multi-agent)

For work that decomposes, use a **manager (orchestrator)** agent that delegates to **worker** subagents and synthesizes their results:

- **Manager/supervisor:** owns the goal, breaks it into subtasks, assigns them, integrates outputs. Often a stronger model (Opus).
- **Workers/subagents:** focused scope, **own context window**, **restricted tools**, return a summary — not their full transcript. Often cheaper models (Haiku/Sonnet).

Two shapes to distinguish (same tell as the Architect exam):
- **Orchestrator-workers:** subtasks decided **dynamically** by the manager at runtime.
- **Parallel sectioning:** subtasks **known in advance**, fanned out concurrently.

**When to keep it single-agent:** coordination has real token + latency cost. If the task is a simple fixed sequence, one agent (or a workflow) is more reliable and cheaper. Justify multi-agent with genuine parallelism or a genuine need for isolated context.

---

## 4. Subagents in practice

Reasons to spawn a subagent:
- **Context isolation** — a noisy subtask (reading 40 files) doesn't pollute the main thread; only its result returns.
- **Parallelism** — independent subtasks run concurrently (review N files at once).
- **Least privilege per role** — a "reviewer" subagent gets read-only tools; an "editor" gets write tools.

Reasons **not** to: the subtask is trivial, or it needs the *same* context the parent already has (spawning just adds overhead).

---

## 5. Frameworks & interop

You can build agents directly on the Agent SDK, or use orchestration frameworks like **LangGraph** (graph-based agent/workflow orchestration) on top of the Claude API. The exam may name-drop such frameworks; the point is that **agent orchestration = defining nodes (LLM calls/tools) and edges (control flow)**, whether via the SDK's loop, a graph framework, or your own code. MCP is the standard way to plug tools/data into any of them.

---

## 6. Worked example: a code-review agent

**Requirement:** review a PR touching many files, flag issues, suggest fixes.

1. **Agent, not workflow** — the number of files and issues isn't known upfront (dynamic path).
2. **Manager (Opus)** plans and, for a big PR, spawns **one subagent per file** (parallel sectioning) — each read-only, isolated context.
3. Each subagent returns a **structured** list of findings (schema-backed), not prose.
4. **Hook** runs the linter/test suite so "verify" is deterministic, not model-judged.
5. Manager **synthesizes** findings, dedupes, and writes one review.
6. **Permissions:** review tools are read-only; any auto-fix requires approval.

Each decision is a gradeable developer choice mapping to a pattern.

---

## 7. Rapid-fire self-check

1. Workflow vs agent — who owns the control flow? *(Your code vs the model.)*
2. Default preference order? *(Single prompt < workflow < agent.)*
3. Agent SDK vs Client SDK — who runs the loop? *(SDK vs you.)*
4. Orchestrator-workers vs sectioning — the distinguishing question? *(Are subtasks decided at runtime or known in advance?)*
5. Two concrete reasons to use a subagent? *(Context isolation; parallelism; also least-privilege per role.)*
6. Enforce a must-hold safety rule in an agent — hook or prompt? *(Hook — deterministic.)*
7. When is multi-agent *not* worth it? *(Simple fixed sequences — overhead outweighs benefit.)*

---

## 8. Further reading

- Building Effective Agents — `https://www.anthropic.com/engineering/building-effective-agents`
- Agent SDK overview — `https://code.claude.com/docs/en/agent-sdk/overview`
- Subagents — `https://code.claude.com/docs/en/agent-sdk/subagents`
- Agent loop — `https://code.claude.com/docs/en/agent-sdk/agent-loop`
- Companion deep dives — `../71_claude_certified_architect/AGENT_SDK_DEEP_DIVE.md`, `../71_claude_certified_architect/AGENTIC_PATTERNS_DEEP_DIVE.md`
