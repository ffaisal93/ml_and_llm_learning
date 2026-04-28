# Build an LLM Agent in 30 Minutes

> Goal: walk into an interview, get asked "design an agent that can answer questions using tools," and write a working agent in 30 minutes that demonstrates real understanding. This file has the canonical patterns + complete code you can write from memory under pressure.

An "agent" in 2024-2025 interview parlance = an LLM in a loop that can call tools, process their results, and decide what to do next. It's three things stacked together: (1) a prompt that includes tool definitions, (2) a parser for tool calls, (3) a loop that executes tools and feeds results back.

---

## 1. The minimal agent loop (the heart of every agent)

```python
def run_agent(query, tools, llm, max_steps=10):
    """
    query: user question (str)
    tools: dict[str, callable] — name -> function
    llm:   callable taking messages list -> response str
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TOOLS(tools)},
        {"role": "user", "content": query},
    ]
    for _ in range(max_steps):
        response = llm(messages)
        messages.append({"role": "assistant", "content": response})

        tool_call = parse_tool_call(response)
        if tool_call is None:
            return response                        # final answer (no tool call)

        try:
            result = tools[tool_call["name"]](**tool_call["args"])
        except Exception as e:
            result = f"ERROR: {e}"

        messages.append({"role": "tool", "content": str(result)})

    return "Max steps reached without final answer."
```

That's the whole thing. Everything below fills in the pieces.

---

## 2. The system prompt with tools

The agent's "API" is a prompt. Two formats dominate:

### JSON tool-calling (modern API native — OpenAI, Anthropic, Llama-3+)

```python
SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful assistant with access to tools.

Tools available:
{tool_descriptions}

To call a tool, respond with EXACTLY this format and nothing else:
<tool_call>{{"name": "tool_name", "args": {{"arg1": "value1"}}}}</tool_call>

To give the final answer (no tool call), respond with:
<final>your answer</final>

Think step by step. Use tools when you need information or computation."""
```

### ReAct format (Yao et al. 2022 — works on weaker models too)

```
Thought: I need to look up X.
Action: search
Action Input: {"query": "..."}
Observation: <tool result inserted here by runtime>
Thought: Now I know X. The user asked about Y...
Final Answer: ...
```

Rule of thumb: JSON tool-calling for production (lower error rate, native API support). ReAct for raw-LLM-on-a-laptop demos.

---

## 3. Tool definitions

Each tool is a function + a schema. Keep both together:

```python
def search_web(query: str) -> str:
    """Search the web. Returns top 3 results as text."""
    # ... implementation (e.g., serpapi, duckduckgo, etc.)
    return formatted_results

def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for arithmetic only."""
    import ast, operator as op
    OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):    return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):  return OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {type(node)}")
    return str(_eval(ast.parse(expression, mode='eval').body))

def read_file(path: str) -> str:
    """Read a file from the local workspace."""
    with open(path) as f:
        return f.read()[:10_000]               # cap output size

TOOLS = {
    "search_web": search_web,
    "calculator": calculator,
    "read_file":  read_file,
}
```

**Interview hot tip**: never use `eval()` or `exec()` on LLM output. Use `ast` parsing or a sandbox. This is the #1 thing to mention when asked "what could go wrong?"

---

## 4. The tool-call parser

```python
import re, json

def parse_tool_call(response: str):
    """Extract tool call from LLM response. Returns dict or None."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if not match:
        return None                             # no tool call -> final answer
    try:
        call = json.loads(match.group(1))
        return {"name": call["name"], "args": call.get("args", {})}
    except (json.JSONDecodeError, KeyError):
        return None                             # malformed -> treat as final answer
```

For the OpenAI / Anthropic native tool-use APIs you don't write this — the API returns structured tool calls directly. But know how to do it from scratch for cases where you're using a raw LLM.

---

## 5. Tool-description generator

```python
def SYSTEM_PROMPT_WITH_TOOLS(tools):
    """Build the system prompt by introspecting tool docstrings."""
    descriptions = []
    for name, fn in tools.items():
        sig = str(inspect.signature(fn))
        doc = (fn.__doc__ or "").strip()
        descriptions.append(f"- {name}{sig}: {doc}")
    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions="\n".join(descriptions)
    )
```

(`inspect.signature` gives you `(query: str) -> str` which the LLM can read.) For production: write a JSON schema explicitly per tool with strict typing.

---

## 6. Putting it all together — a complete runnable agent

```python
import inspect, json, re, ast, operator as op

# ---------- tools ----------
def search_web(query: str) -> str:
    """Search the web. Returns top 3 result snippets."""
    return f"[stub] Top result for '{query}': ..."

def calculator(expression: str) -> str:
    """Evaluate a basic math expression like 2*(3+4)."""
    OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):    return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):  return OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported")
    return str(_eval(ast.parse(expression, mode='eval').body))

TOOLS = {"search_web": search_web, "calculator": calculator}

# ---------- prompt ----------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with access to tools.

Tools:
{tool_descriptions}

To call a tool, respond with EXACTLY:
<tool_call>{{"name": "tool_name", "args": {{...}}}}</tool_call>

For the final answer, respond with:
<final>your answer</final>

Think step by step."""

def system_prompt(tools):
    descs = [f"- {name}{inspect.signature(fn)}: {(fn.__doc__ or '').strip()}"
             for name, fn in tools.items()]
    return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions="\n".join(descs))

# ---------- parser ----------
def parse_tool_call(text):
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m: return None
    try:
        call = json.loads(m.group(1))
        return {"name": call["name"], "args": call.get("args", {})}
    except (json.JSONDecodeError, KeyError):
        return None

def parse_final(text):
    m = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# ---------- loop ----------
def run_agent(query, tools, llm_fn, max_steps=10):
    messages = [
        {"role": "system", "content": system_prompt(tools)},
        {"role": "user",   "content": query},
    ]
    for step in range(max_steps):
        response = llm_fn(messages)
        messages.append({"role": "assistant", "content": response})

        # 1. final answer?
        if (final := parse_final(response)) is not None:
            return final, messages

        # 2. tool call?
        if (call := parse_tool_call(response)) is not None:
            try:
                result = tools[call["name"]](**call["args"])
            except Exception as e:
                result = f"ERROR: {e}"
            messages.append({"role": "tool", "content": str(result)})
            continue

        # 3. neither — model went off-format. Nudge.
        messages.append({"role": "user",
                         "content": "Respond with <tool_call>...</tool_call> or <final>...</final>."})

    return "Max steps reached.", messages
```

That's a complete, working agent in ~70 lines. Drill until you can write it cold in 25 minutes.

---

## 7. Common interview extensions (how to evolve it)

The interviewer will push you to add features. Here's what to do for each:

### "Add memory across conversations"

```python
# Use a vector store keyed by user_id
from collections import defaultdict
memory_store = defaultdict(list)

def remember(user_id, content):
    memory_store[user_id].append(content)

def retrieve_memory(user_id, query, k=5):
    # In production: embed query, ANN over stored embeddings.
    # For interview: simple recency or keyword match.
    return memory_store[user_id][-k:]

# In the agent loop, prepend retrieved memories to messages
messages.insert(1, {"role": "system",
                    "content": "Relevant memories:\n" + "\n".join(retrieve_memory(user_id, query))})
```

### "Handle parallel tool calls"

```python
import asyncio

async def run_tools_parallel(calls, tools):
    async def one(c):
        return await asyncio.to_thread(tools[c["name"]], **c["args"])
    return await asyncio.gather(*[one(c) for c in calls])

# Modify the parser to extract a list of <tool_call>s, run them in parallel.
```

Most modern APIs return *lists* of tool calls per assistant turn. Always assume parallel when productionizing.

### "Add a planner"

Two-tier agent: a planner LLM produces a high-level plan; an executor loop executes each step. Planner is called once at the start (and maybe at replan points); executor is the loop above.

```python
def make_plan(query, tools, llm):
    return llm([{"role": "system", "content": "You produce step-by-step plans..."},
                {"role": "user",   "content": query}])

def run_planner_executor(query, tools, llm):
    plan = make_plan(query, tools, llm)
    for step in plan.steps:
        run_agent(step, tools, llm, max_steps=5)
```

### "Add safety / guardrails"

- Input filter: classifier on user query (block obvious abuse).
- Output filter: classifier on final answer.
- Tool whitelist: only expose tools the current user has permission for.
- Rate limit on tool calls per session.
- Refuse list of dangerous patterns ("ignore previous instructions" → kick to safety prompt).

Talk through these explicitly when asked "what could go wrong?"

### "Add observability"

```python
class AgentTrace:
    def __init__(self):
        self.steps = []

    def log(self, kind, content):
        self.steps.append({"kind": kind, "content": content, "ts": time.time()})

# In the loop, log each LLM call + tool call + result.
# Persist to OpenTelemetry / Langfuse / your own DB.
```

In production you want to see the entire trace per request — debugging an agent without traces is brutal.

### "Stream tokens"

```python
async def run_agent_streaming(query, tools, llm_stream_fn):
    async for chunk in llm_stream_fn(messages):
        yield chunk                           # send to UI
        # parse incrementally for tool calls
```

For UX, stream the assistant's text. When you detect a `<tool_call>` opening tag, stop streaming, run the tool, then continue.

---

## 8. Common failure modes (interview hot-buttons)

The interviewer will ask "what goes wrong?" Have these ready:

| Failure | Cause | Fix |
|---|---|---|
| Hallucinated tool call (tool name doesn't exist) | LLM sometimes invents tools | Validate tool name before executing; if invalid, return error |
| Malformed JSON args | Mostly fixed in modern APIs; still happens with raw LLMs | Use `json_repair` or re-prompt with parser error |
| Infinite loops (tool → tool → tool) | No clear stopping condition | `max_steps` cap; detect repeated identical tool calls |
| Cascading errors | One bad result confuses subsequent reasoning | Summarize / truncate long tool outputs before re-injecting |
| Context bloat | Tool outputs accumulate, exceed context | Summarize older messages every N steps |
| Prompt injection via tool output | Document/web result contains "ignore previous instructions" | Mark tool output as untrusted; system-prompt instructs model not to follow |
| Cost runaway | Long agent chains burn tokens fast | Per-session budget; warn at threshold |
| Permission escalation | Tools have side effects (delete file, send email) | Require confirmation for destructive actions |

---

## 9. Architecture choices to discuss

**Single-agent vs multi-agent.** Single = one LLM in a loop. Multi-agent = multiple specialized LLMs (researcher, writer, critic) collaborating. Multi-agent is more structured but expensive; single-agent with strong tools usually wins for most tasks.

**Native tool calling vs ReAct prompting.** Native (OpenAI / Anthropic / Llama 3+ JSON tool-calls) is more reliable; ReAct is necessary for raw LLMs. Modern frontier models all support native.

**Stateful vs stateless tools.** Stateless (search, calculator) can be retried freely. Stateful (write file, send email) need confirmation, idempotency keys, and rollback.

**Synchronous vs asynchronous.** Voice agents need streaming + parallel tool calls. Background agents (e.g., overnight research) can be sequential.

**On-policy training (Tülu, Llama, Qwen).** Some teams now train models *specifically* on agentic trajectories (tool calls + observations). The model becomes much better at multi-turn tool use after this. Worth mentioning if asked about future improvements.

---

## 10. The interview narrative (5 min answer)

When asked "design an agent":

1. **Frame** (15 sec): "An agent is an LLM in a loop with tools. Three pieces: system prompt with tool definitions, a parser for tool calls, the loop that executes them."
2. **Code the loop** (5 min): the 70-line skeleton above.
3. **Discuss extensions** (5-10 min): memory, parallel tools, planner, observability, streaming. Pick whichever the interviewer pushes on.
4. **Failure modes** (5 min): walk through the table in §8. Always mention prompt injection from tool outputs.
5. **Production touches** (5 min): rate limits, guardrails, trace logging, cost budgets, permission tiers.

---

## 11. Drill plan

- Write the 70-line agent from memory in 25 minutes. Repeat 3 times until automatic.
- For each extension in §7, write the code snippet from scratch.
- For each failure in §8, recite cause and fix in one sentence.
- Practice the 5-min narrative out loud.

---

## 12. Further reading

- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.*
- Anthropic (2024). *Building effective agents.* — the canonical practitioner essay.
- OpenAI Assistants API docs / Anthropic tool use docs — production patterns.
- LangChain / LlamaIndex source — see how a real framework's agent loop is structured.
