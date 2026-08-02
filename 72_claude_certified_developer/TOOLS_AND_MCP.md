# Tools & MCPs (10.6%)

> Implementing tools and MCP servers in code: schemas, descriptions, error handling, `tool_choice`, client- vs server-side patterns, and when to use built-in tools, custom tools, or MCP. The conceptual MCP model (primitives, trust, scopes) is in `../71_claude_certified_architect/MCP_DEEP_DIVE.md`; here's the developer's implementation view.

---

## 1. Defining a custom tool

A tool is a name + description + JSON-Schema `input_schema`:

```python
tools = [{
    "name": "create_ticket",
    "description": ("Create a support ticket. Use when the user reports a problem "
                    "that needs tracking. Returns the ticket id."),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "customer_id": {"type": "string"},
        },
        "required": ["title", "priority", "customer_id"],
    },
}]
```

Implementation rules the exam rewards:
- **The description is the interface.** The model picks and uses the tool by reading it — say *what it does, when to use it, and what it returns*. A vague description is a functional bug.
- **Enums for constrained params** (`priority`) so the model can't pass garbage.
- **Stable IDs** in and out (`customer_id`, returned `ticket_id`), not fuzzy references.
- **Right granularity:** bundle always-together mechanical steps into one tool; keep genuine decision points as separate tools.

---

## 2. The tool-use loop and `tool_choice`

The loop (full code in `APPLICATIONS_INTEGRATION_DEEP_DIVE.md`): send tools → model returns `tool_use` → **your code runs it** → return `tool_result` → repeat until `end_turn`.

`tool_choice` modes:

| Mode | Behavior |
|---|---|
| `auto` | Model decides whether to use a tool (default) |
| `any` | Must call *some* tool |
| `tool` (named) | Must call *this specific* tool — great for forcing structured extraction |
| `none` | No tools this turn |

Return failures with `is_error: true` in the `tool_result` so the model can **adapt** (retry differently, ask the user) instead of assuming success.

---

## 3. Built-in vs. custom vs. MCP — choosing

| Option | Use when |
|---|---|
| **Built-in tools** (e.g. web search, code execution, computer use, text editor) | Anthropic already provides the capability — don't reinvent it |
| **Custom API tools** (defined inline in your request) | App-specific functions where you control both ends |
| **MCP server** | You want the capability **reusable across apps**, a clean client/server boundary, or to plug into existing MCP servers |

> **Exam framing.** "Multiple internal apps need the same 'query warehouse' capability." → build it once as an **MCP server**, not duplicated inline tools per app. "One app needs one bespoke function." → inline **custom tool**.

---

## 4. Client-side vs. server-side tools

- **Client-side tool:** your application executes it (runs in *your* process/loop). Most custom tools are client-side — you own the code and secrets.
- **Server-side / hosted tool:** executed by Anthropic's infrastructure (certain built-ins) — no execution code on your side.
- **MCP tool:** executed by the **MCP server** (local subprocess or remote HTTP), decoupled from your app.

The developer decision is about **where the code and credentials live** and **who runs it** — keep secrets on the side that should hold them, and apply least privilege wherever execution happens.

---

## 5. Authoring an MCP server (shape)

An MCP server exposes **tools** (model-controlled actions), **resources** (app-controlled context), and **prompts** (user-controlled workflows) over JSON-RPC. Minimal Python (FastMCP-style):

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("warehouse")

@mcp.tool()
def query_stock(sku: str) -> dict:
    """Return current stock for a SKU."""      # docstring becomes the description
    return {"sku": sku, "qty": lookup(sku)}

@mcp.resource("schema://warehouse")
def schema() -> str:
    """The warehouse DB schema (reference context)."""
    return SCHEMA_TEXT

if __name__ == "__main__":
    mcp.run()   # stdio (local) or HTTP (remote)
```

Deployment notes:
- **Local** servers run as a subprocess over stdio (dev, personal machine); **remote** servers run over HTTP (shared/team, cloud).
- In **Claude Code**, configure servers at **local / project / user** scope (precedence: local > project > user). Team-wide, ship a committed **project** `.mcp.json`.
- **Error handling:** distinguish JSON-RPC **protocol errors** (plumbing) from **tool execution errors** (`isError: true`, the tool ran but failed — feed to the model).
- **Trust:** tool annotations (`readOnlyHint`, etc.) are **untrusted hints**, not security. Enforce permissions/sandboxing/approval on the client.

---

## 6. Worked example: expose an internal DB to several agents

1. Build **one MCP server** wrapping the DB (reusable across apps) rather than inline tools per app.
2. **Tools** for actions (`run_query`, `create_record`); the **schema** as a **resource** (reference context, not an action); a saved analysis workflow as a **prompt**.
3. **Least privilege:** the query tool is read-only; writes are a separate, permissioned tool requiring approval.
4. **Errors:** "table not found" → `isError: true` result the model can react to; a dropped connection → protocol error your client retries.
5. Ship it at **project** scope so every teammate gets it on clone.

---

## 7. Rapid-fire self-check

1. What part of a tool does the model use to decide when to call it? *(The description.)*
2. Force the model to call your `extract` tool every time? *(`tool_choice: {type:"tool", name:"extract"}`.)*
3. Signal a tool failure so the model adapts? *(`is_error: true` in the `tool_result`.)*
4. Same capability needed by many apps — inline tool or MCP server? *(MCP server.)*
5. Three MCP primitives and their controllers? *(Tools—model, Resources—app, Prompts—user.)*
6. Are `readOnlyHint` annotations a security control? *(No — untrusted hints.)*
7. Team-wide MCP server via the repo — which scope? *(Project, committed `.mcp.json`.)*

---

## 8. Further reading

- Tool use — `https://platform.claude.com/docs/en/build-with-claude/tool-use`
- Writing tools for agents — `https://www.anthropic.com/engineering/writing-tools-for-agents`
- MCP spec — `https://modelcontextprotocol.io`
- Build an MCP server — `https://modelcontextprotocol.io/quickstart/server`
- MCP in Claude Code — `https://code.claude.com/docs/en/mcp`
- Companion — `../71_claude_certified_architect/MCP_DEEP_DIVE.md`
