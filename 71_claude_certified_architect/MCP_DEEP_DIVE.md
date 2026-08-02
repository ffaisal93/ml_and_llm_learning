# Model Context Protocol (MCP) Deep Dive

> Covers exam domain **Tool Design & MCP Integration (18%)** and threads through orchestration and Claude Code. MCP questions are very "definitional" — they reward knowing *exactly* which primitive does what and where the trust boundary sits.

---

## 1. What MCP is, in one breath

**MCP is an open standard for connecting AI applications to external tools and data through a uniform interface.** Think of it as *"USB-C for AI context"*: instead of every app inventing its own bespoke way to plug in a database, a filesystem, or an API, MCP defines one plug. Write an MCP server once, and any MCP-compatible client (Claude Code, the Agent SDK, Claude Desktop, third-party hosts) can use it.

**Why it exists:** without a standard, N applications × M integrations = N·M custom connectors. MCP turns that into N + M — each app speaks MCP once, each integration exposes MCP once.

**Architecture:** a **client** (inside the AI app/host) talks to a **server** (which wraps some external system) over **JSON-RPC**. The server advertises what it offers; the client discovers and uses it. Servers can run **locally** (a subprocess over stdio) or **remotely** (over HTTP).

---

## 2. The three primitives — the heart of the exam

MCP servers expose exactly three kinds of things. The exam *will* test which is which, and — crucially — **who controls each**.

| Primitive | Controlled by | Nature | Example | Mental model |
|---|---|---|---|---|
| **Tools** | **Model** | Actions with side effects / computation | `search_orders`, `create_ticket`, `run_query` | Verbs the model may *invoke* |
| **Resources** | **Application** | Passive context the app pulls in | a schema, a catalog, a doc, a file | Nouns the app *reads* and puts in context |
| **Prompts** | **User** | Reusable templates / workflows | a "code review checklist", a "triage playbook" | Saved recipes the user *chooses* |

Read the "controlled by" column three times. It is the single most testable distinction in the whole MCP domain.

- **Tools are *model-controlled*:** Claude decides when to call them during a task. They *do* things.
- **Resources are *application-controlled*:** the app decides what reference material to load into context. They don't *do* anything — they're information to consult.
- **Prompts are *user-controlled*:** the user picks them (in Claude Code they surface as **slash commands**). They template a workflow.

> **The design rule to memorize:** *"If content is reference material the agent might consult before acting, expose it as a resource. If it is an action the agent takes, expose it as a tool. If it is a workflow the user launches, expose it as a prompt."*

### Worked example
You're building an MCP server for a support system:
- `get_customer(id)` and `refund_order(id)` → **tools** (the model calls them to act).
- The product catalog and the refund-policy document → **resources** (the app loads them so the model reasons with correct facts).
- "Run the full refund-eligibility check" → a **prompt** (the agent chooses this saved workflow).

Putting the refund-policy document in as a *tool* the model has to "call" would be a design smell — it's reference material, so it's a **resource**.

---

## 3. The trust model — annotations are hints, not guarantees

MCP tools can carry **annotations** that *describe* their behavior:

| Annotation | Claim | 
|---|---|
| `readOnlyHint` | "I don't modify anything" |
| `destructiveHint` | "I may delete/overwrite" |
| `idempotentHint` | "Calling me twice = calling me once" |
| `openWorldHint` | "I touch external/unbounded systems" |

**The critical exam point:** these are **untrusted hints for UX and heuristics — not security controls.** A malicious or buggy server can label a destructive tool `readOnlyHint: true`. You must **never** make a security or auto-approval decision solely because a tool *claims* to be read-only. Real safety comes from **permissions, sandboxing, and human approval on the client side**, not from believing the server's self-description.

> **Exam trap.** "Can you auto-approve any tool with `readOnlyHint: true`?" → **No.** Annotations are advisory. Trust boundaries are enforced by the host/client, not by the server's honesty.

**Server trust generally:** connecting an MCP server grants it a channel into your agent. Treat third-party servers like any dependency — vet them, scope their permissions, and assume their descriptions could be wrong or adversarial (a prompt-injection vector: a compromised server could return tool descriptions or results that try to steer the model).

---

## 4. Error semantics — two distinct error channels

MCP deliberately separates **two** kinds of failure, and conflating them is a classic wrong answer.

1. **Protocol errors (JSON-RPC level):** the request itself was malformed, the method doesn't exist, the server is unreachable. These surface as JSON-RPC errors — the *plumbing* failed.
2. **Tool execution errors:** the tool ran but *failed its job* (e.g., "order not found," "insufficient funds"). These come back as a **normal tool result with `isError: true`** and a human/model-readable message.

Why the split matters: a **tool execution error is information the model should see and react to** ("the order doesn't exist, so ask the user to recheck the ID"). A **protocol error is an infrastructure problem** for your client to handle (retry the connection, alert an operator) — it usually shouldn't be fed to the model as if it were a task-level result.

> Map this back to the API file's `is_error` on `tool_result`: same spirit — let the model *see* task failures so it can adapt, while keeping infrastructure failures in your control plane.

---

## 5. Tool discovery and progressive availability

Agents degrade when you dump 200 tools on them — the model spends attention choosing and mis-picks. MCP supports **making tools discoverable and available progressively**:

- **Good descriptions** are the primary discovery mechanism — the model picks tools by reading their descriptions, so descriptions are part of the interface, not documentation.
- **Progressive availability / tool search:** expose a small relevant set for the current phase; reveal more on demand rather than all at once.
- **`list_changed` notifications:** a server can tell the client "my tool list changed," enabling **dynamic tool registration** — tools that appear/disappear as state changes (e.g., admin tools only after auth).

The architectural principle: **the set of tools visible to the model is itself a design decision.** Fewer, well-described, well-scoped tools beat a giant flat catalog.

---

## 6. MCP scopes in Claude Code (precedence — memorize the order)

When you configure MCP servers in **Claude Code**, configuration lives at **three scopes**, and precedence is:

> **local > project > user** (most specific wins)

| Scope | Where | Who it's for |
|---|---|---|
| **local** | your machine, current project only (not shared) | personal, per-project experiments; overrides the others |
| **project** | checked into the repo (`.mcp.json`) | the whole team, shared via version control |
| **user** | your machine, across all projects | your personal defaults everywhere |

So a **local** server config overrides a **project** one, which overrides a **user** one. This mirrors the general Claude Code configuration hierarchy (see `CLAUDE_CODE_DEEP_DIVE.md`), where more specific/closer-to-the-work settings win.

> **Exam trap.** "A team wants an MCP server available to everyone who clones the repo." → **project** scope (committed `.mcp.json`), not user (that's only your machine) and not local (not shared).

---

## 7. MCP vs. plain API tool use — when to reach for MCP

Both let the model call functions. The difference is **reusability and boundaries**:

- **Plain API tools** are defined inline in your request — great when the tools are specific to one app and you control both ends.
- **MCP** shines when you want the **same** capability reusable across **many** AI apps, a clean **client/server separation**, or to plug into an ecosystem of existing servers. It standardizes discovery, the tool/resource/prompt split, and the error model.

Architect's rule of thumb: one app, a couple of bespoke functions → inline API tools are simplest. A capability multiple agents/apps should share, or a third-party integration → MCP server.

---

## 8. Rapid-fire self-check

1. Name the three MCP primitives and who controls each. *(Tools—model; Resources—application; Prompts—user.)*
2. A policy PDF the agent should consult before acting: tool or resource? *(Resource.)*
3. Can you auto-approve a tool because `readOnlyHint` is true? *(No — annotations are untrusted hints.)*
4. Difference between a JSON-RPC protocol error and a tool result with `isError: true`? *(Plumbing failure vs. the tool ran but its task failed; the latter is fed to the model to adapt.)*
5. MCP scope precedence in Claude Code? *(local > project > user.)*
6. Team-wide shared server via the repo: which scope? *(project.)*
7. What does a `list_changed` notification enable? *(Dynamic tool discovery/registration.)*

---

## 9. Further reading

- MCP spec & docs — `https://modelcontextprotocol.io`
- MCP in Claude Code — `https://code.claude.com/docs/en/mcp`
- MCP in the Agent SDK — `https://code.claude.com/docs/en/agent-sdk/mcp`
- Anthropic tool-use / tool-design guidance — `https://platform.claude.com/docs/en/build-with-claude/tool-use`
