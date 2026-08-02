# Claude Agent SDK Deep Dive

> Part of exam domain **Agentic Architecture & Orchestration (27%)** — the biggest slice. The SDK is *"Claude Code as a library."* Understand the agent loop, the four SDK-vs-alternatives choices, and subagents, and you own most of this domain's SDK questions.

---

## 1. What the Agent SDK is

An **agent** is an application that completes a task by **planning its own steps and calling tools** (read files, run commands, hit APIs) rather than following a hard-coded script. The **Claude Agent SDK** gives you the *same tools, agent loop, and context management that power Claude Code*, as a programmable library in **Python and TypeScript**.

The key insight: Anthropic took the engine inside Claude Code — the loop that decides what to do next, when to call a tool, when it's done — and exposed it so you can embed it in your own product. You get built-in file/command/web tools, hooks, subagents, MCP integration, permissions, and session management for free, and you write the glue for *your* domain.

> Language note: the SDK is **Python/TS only**. To drive the same loop from another language, run the **CLI headless** as a subprocess with `-p` and `--output-format json`.

---

## 2. Choosing the right tool — the four-way decision (very testable)

Anthropic frames a table you should be able to reproduce. Given what you're building, pick:

| If you're… | Use | Why |
|---|---|---|
| Building an agent **without writing the tool loop yourself** | **Agent SDK** | Runs the agent loop in *your* process (Python/TS) |
| Doing **interactive** dev / one-off terminal tasks | **Claude Code CLI** | The terminal UX for daily use |
| Calling the API directly and **implementing the loop yourself** | **Client SDK** (Anthropic API SDK) | Raw API access; you own the loop |
| Running **long-running/async** agents **without managing your own sandbox/session infra** | **Managed Agents** | Hosted REST product; Anthropic runs the agent and sandbox |

> **Exam trap.** "You want an agent loop with file/command tools and context management, embedded in your Python service, but you don't want to hand-roll the observe-act loop." → **Agent SDK**, not the Client SDK (that's when you *do* want to own the loop) and not Managed Agents (that's hosted/async with no infra on your side).

---

## 3. The agent loop — the beating heart

Every agent, whether Claude Code or your SDK app, runs the same cycle. A clean way to remember it:

> **Gather context → Take action (call tools) → Verify the work → Repeat until done.**

Expanded:

1. **Observe / gather context** — read the current state: user request, files, prior tool results, memory. Context strategy (what to include) decides what the model can reason over.
2. **Reason / plan** — the model decides the next step. With extended thinking, it can plan more deliberately.
3. **Act** — emit a `tool_use`; the harness executes the tool and feeds back a `tool_result`.
4. **Verify** — check the result (tests pass? file compiles? schema valid?). Verification is where reliable agents separate from flaky ones — build checks into the loop, don't assume success.
5. **Repeat** until the task is complete (`end_turn`) or a stop condition/permission gate halts it.

The SDK runs this loop for you; you shape it with tools, hooks, permissions, and subagents.

---

## 4. Capabilities you configure (know what each is *for*)

| Capability | What it does | Architect's use |
|---|---|---|
| **Built-in tools** | Read/write/edit files, run commands, web search | The default action surface |
| **Hooks** | Run your code at lifecycle points (pre/post tool use, etc.) | Deterministic guardrails, logging, injecting context, blocking dangerous actions |
| **Subagents** | Spawn specialized agents for focused subtasks, each with its **own context** | Isolate/parallelize work; keep main context clean |
| **MCP** | Connect external tools/data via MCP servers | Reuse standardized integrations |
| **Permissions** | Which tools run automatically vs. need approval | Least-privilege; human-in-the-loop for risky actions |
| **Sessions** | Persist context across exchanges; resume or fork | Long-running / returning-user agents |
| **Skills, commands, memory** | Auto-load from `.claude/` and `~/.claude/` (same as Claude Code) | Reuse project conventions and workflows |
| **Plugins** | Package skills/agents/hooks/MCP servers together | Distribute a whole capability bundle |

**Hooks vs. prompt instructions** is a favorite distinction: if a rule *must* hold ("never delete without confirmation"), enforce it with a **hook** (deterministic code), not by asking the model in the prompt (probabilistic). Prompts guide; hooks guarantee.

---

## 5. Subagents and orchestration

Subagents are how the SDK does multi-agent work. A subagent:
- has its **own context window** (so a big, noisy subtask doesn't pollute the main thread),
- can have a **restricted tool set** (least privilege per role),
- returns a **summary/result** to the coordinator, not its whole transcript.

This maps directly onto the orchestration patterns (full treatment in `AGENTIC_PATTERNS_DEEP_DIVE.md`):
- **Orchestrator-workers:** a coordinator delegates subtasks to worker subagents and synthesizes.
- **Parallel subagents:** independent partitions (e.g., review 10 files) run concurrently.

**When *not* to use subagents:** if the task is a simple fixed sequence, a single agent (or a prompt chain) is cheaper and more reliable. Multi-agent adds coordination overhead and token cost — justify it with genuine parallelism or genuine need for isolated context.

---

## 6. Context management for long-running agents

Because the API is stateless and windows are finite, long agents must actively manage context:
- **Compaction / summarization** — condense old turns into a compact state as the window fills (sessions do this).
- **Externalize memory** — write durable facts to files/stores (memory) and re-read on demand instead of carrying everything in-context.
- **Subagent isolation** — push a big subtask's raw context into a subagent and keep only its result.

The reliability failure mode to recognize: an agent that "forgets" mid-task usually blew its context budget or summarized away something it later needed — an application/design problem, not a model defect.

---

## 7. Worked example: a bug-fixing agent

The canonical SDK quickstart is an agent that **finds and fixes bugs**. As an architecture:

1. **Tools:** read files, run the test suite, edit files.
2. **Loop:** gather context (read failing test + source) → act (edit) → **verify** (re-run tests) → repeat until green.
3. **Permissions:** auto-allow reads and test runs; **require approval** before `git push`.
4. **Hook:** `PreToolUse` blocks edits to `/vendor`; `PostToolUse` runs the linter after each edit.
5. **Subagent:** for a large refactor, spawn a subagent to handle one module in isolation and report back.
6. **Session:** persist so the developer can resume tomorrow with summarized context.

Every element is an SDK capability mapped to a real need — exactly the reasoning the exam rewards.

---

## 8. Rapid-fire self-check

1. Agent SDK vs Client SDK — who writes the agent loop? *(SDK runs it for you; Client SDK = you write it.)*
2. When do you pick Managed Agents? *(Long-running/async, no infra to manage on your side; Anthropic hosts.)*
3. State the agent loop in four beats. *(Gather context → act → verify → repeat.)*
4. Guarantee a rule always holds: hook or prompt? *(Hook — deterministic.)*
5. Why give a subtask its own subagent? *(Isolated context + restricted tools; parallelism.)*
6. When is a single agent better than multi-agent? *(Simple fixed sequences — less overhead, more reliable.)*
7. Non-Python/TS language — how do you run the same loop? *(CLI headless: `-p` + `--output-format json`.)*

---

## 9. Further reading

- Agent SDK overview — `https://code.claude.com/docs/en/agent-sdk/overview`
- Agent loop — `https://code.claude.com/docs/en/agent-sdk/agent-loop`
- Subagents — `https://code.claude.com/docs/en/agent-sdk/subagents`
- Hooks — `https://code.claude.com/docs/en/agent-sdk/hooks`
- Sessions — `https://code.claude.com/docs/en/agent-sdk/sessions`
- Managed Agents — `https://platform.claude.com/docs/en/managed-agents/overview`
- "Building Effective Agents" — `https://www.anthropic.com/engineering/building-effective-agents`
