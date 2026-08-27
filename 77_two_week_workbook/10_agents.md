# Agents

Agent questions test engineering judgement, not model knowledge. The interviewer wants to know whether you can bound a system that calls itself in a loop. Candidates lose points in two ways. They reach for an agent when a fixed pipeline would be cheaper and more reliable. And they describe the ReAct loop without naming the controls that stop it: a step budget, a token budget, a wall-clock timeout, and a repeat detector. Bring numbers to every claim about reliability and cost.

## The equations

**Compounding reliability.** For $n$ sequential steps, each succeeding independently with probability $p$:

$$P(\text{task succeeds}) = p^n$$

$p$ is the per-step success rate and $n$ is the number of steps; a long chain fails even when every single step looks good, because errors multiply rather than add.

**The concrete case.** With $p = 0.95$ and $n = 10$:

$$0.95^{10} = 0.5987$$

Ten steps at ninety-five percent each give about sixty percent end-to-end, so a plausible-sounding step rate produces a coin flip of a product.

**Per-step rate required for a target.** To reach end-to-end success $S$ over $n$ steps:

$$p = S^{1/n}$$

For $S = 0.95$ and $n = 10$ you need $p = 0.9949$, so a ninety-five percent product needs a ninety-nine and a half percent step.

**Quadratic token growth of a ReAct loop.** If the loop resends the whole history each turn, and turn $k$ adds $d$ tokens on top of a base prompt $b$:

$$T = \sum_{k=1}^{n} \left( b + kd \right) = nb + d\,\frac{n(n+1)}{2}$$

$T$ is total tokens billed across the run; the $n^2$ term means doubling the step count roughly quadruples the token cost.

**Cost of a run.** With price $c$ per input token:

$$\text{cost} = c \cdot T \approx c \cdot \frac{d n^2}{2} \quad \text{for large } n$$

The asymptotic term is the one that kills you, so cap $n$ before you optimise anything else.

**Exponential backoff with full jitter.** On retry attempt $k$ (zero-indexed):

$$\text{delay}_k \sim \mathrm{Uniform}\left(0,\ \min(\text{cap},\ \text{base} \cdot 2^{k})\right)$$

base is the first delay and cap is the ceiling; drawing uniformly over the whole window spreads retries out, whereas fixed backoff makes every client retry at the same instant.

**Total wait before giving up.** Expected total sleep over $m$ retries, ignoring the cap:

$$E[W] = \sum_{k=0}^{m-1} \frac{\text{base} \cdot 2^{k}}{2} = \frac{\text{base}}{2}\left(2^{m} - 1\right)$$

This is why $m$ must be small: five retries at half a second base is about fifteen and a half seconds of expected sleep.

**Step budget.** The loop must stop on the first of several limits:

$$\text{stop} = \left[\, s \ge s_{\max} \ \lor\ T \ge T_{\max} \ \lor\ t \ge t_{\max} \ \lor\ \text{repeat}(a_s, a_{s-1}) \,\right]$$

$s$ is steps taken, $T$ is tokens spent, $t$ is wall-clock seconds, and $\text{repeat}$ fires when the model calls the same tool with the same arguments twice; any one of them ends the run.

**Effective budget from a cost ceiling.** Given a per-task budget $B$ in currency and average marginal cost $c\,d$ per step:

$$s_{\max} \approx \sqrt{\frac{2B}{c\,d}}$$

The square root comes from the quadratic growth above, so doubling your money budget only buys about forty percent more steps.

**Tool-selection reliability.** If the model picks the right tool with probability $q$ and the tool then succeeds with probability $r$:

$$p = q \cdot r$$

Splitting $p$ this way tells you whether to fix the schema and descriptions ($q$) or the tool itself ($r$).

## Code from memory

Purpose: a minimal ReAct loop with a tool registry and a hard step budget, driven by a stub model so it runs offline.

```python
TOOLS = {}

def tool(name, fn):
    TOOLS[name] = fn

tool("add", lambda a, b: a + b)
tool("lookup", lambda key: {"gpu": "H100", "mem": "80GB"}.get(key, "unknown"))

def fake_model(history):
    # Stub policy: act twice, then answer. Real code calls an LLM here.
    n = sum(1 for h in history if h["role"] == "observation")
    if n == 0:
        return {"thought": "I need a sum.", "action": "add", "args": {"a": 2, "b": 3}}
    if n == 1:
        return {"thought": "Now the hardware.", "action": "lookup", "args": {"key": "gpu"}}
    return {"thought": "Done.", "final": "5 on an H100"}

def react(question, max_steps=6):
    history = [{"role": "user", "content": question}]
    for step in range(max_steps):          # hard step budget, never while True
        msg = fake_model(history)
        if "final" in msg:
            return msg["final"], step
        fn = TOOLS.get(msg["action"])
        obs = fn(**msg["args"]) if fn else f"error: no tool named {msg['action']}"
        history.append({"role": "observation", "content": str(obs)})
    return "budget exhausted", max_steps

print(react("what is 2+3 and which gpu?"))
```

Output: `('5 on an H100', 2)`. The `for` loop is the point. A `while True` with a break condition is the same code with the safety removed.

Purpose: a tool schema as a Pydantic model, with a validator whose error text tells the model how to recover.

```python
from pydantic import BaseModel, Field, field_validator, ValidationError

class SearchOrders(BaseModel):
    """Find a customer's orders in a date range. Returns at most `limit` orders."""
    customer_id: str = Field(description="Internal customer id, format CUST-12345")
    start_date: str = Field(description="Inclusive ISO date, YYYY-MM-DD")
    limit: int = Field(default=10, ge=1, le=100, description="Max orders to return")

    @field_validator("customer_id")
    @classmethod
    def check_id(cls, v):
        if not v.startswith("CUST-") or not v[5:].isdigit():
            # Error text is written FOR the model: say how to fix it.
            raise ValueError("customer_id must look like CUST-12345; strip names and emails")
        return v

def call_tool(payload):
    try:
        args = SearchOrders(**payload)
        return {"ok": True, "args": args.model_dump()}
    except ValidationError as e:
        # Return the error to the model as an observation, do not crash the loop.
        return {"ok": False, "error": "; ".join(x["msg"] for x in e.errors())}

print(call_tool({"customer_id": "CUST-42", "start_date": "2026-01-01", "limit": 5}))
print(call_tool({"customer_id": "jane@example.com", "start_date": "2026-01-01"}))
print(call_tool({"customer_id": "CUST-42", "start_date": "2026-01-01", "limit": 500}))
```

Output, in order: `ok True`; `error 'Value error, customer_id must look like CUST-12345; strip names and emails'`; `error 'Input should be less than or equal to 100'`. The bounded `limit` is enforced by the type, so the model cannot request fifty thousand rows.

Purpose: a retry wrapper with exponential backoff and full jitter that retries transient failures and never retries a bad request.

```python
import random

class Retryable(Exception): pass      # 429, 503, timeout
class Fatal(Exception): pass          # 400, 401, schema error

def with_backoff(fn, max_tries=5, base=0.5, cap=8.0, sleep=lambda s: None):
    for attempt in range(max_tries):
        try:
            return fn(attempt)
        except Fatal:
            raise                                  # never retry a bad request
        except Retryable:
            if attempt == max_tries - 1:
                raise
            # full jitter: uniform over the whole exponential window
            delay = random.uniform(0, min(cap, base * 2 ** attempt))
            sleep(delay)
    raise Retryable("unreachable")

random.seed(0)
calls = []
def flaky(attempt):
    calls.append(attempt)
    if attempt < 2:
        raise Retryable("429 rate limited")
    return "ok"

print(with_backoff(flaky), "attempts:", calls)

def bad(attempt):
    raise Fatal("400 malformed tool arguments")
try:
    with_backoff(bad)
except Fatal as e:
    print("not retried:", e)
```

Output: `ok attempts: [0, 1, 2]` then `not retried: 400 malformed tool arguments`. `sleep` is injected so the test runs instantly; in production pass `time.sleep`.

## Questions

### Q1. When is an agent the right choice, and when is a fixed workflow better?

Use a fixed workflow when you know the steps in advance. A workflow is a directed graph you wrote: retrieve, then rank, then summarise. It is testable, cheap, and its latency and cost are known before you run it. Use an agent when the number of steps and their order depend on what the model discovers at run time, and the space of paths is too large to enumerate. Debugging a customer issue that could touch billing, shipping, or account state is a real agent task. The cost is severe. Compounding reliability says $p^n$: at $p = 0.95$ and $n = 10$ you get $0.5987$, about sixty percent. Token cost grows as $nb + dn(n+1)/2$, so it is quadratic in steps. Therefore my default is a workflow, and I promote to an agent only when I can show a workflow cannot cover the branching.

> **Say it.** My default is a fixed workflow, because I can test it, price it, and bound its latency. I only reach for an agent when the path depends on what the model finds at run time and I cannot enumerate the branches. The reason I am reluctant is arithmetic. Ten sequential steps at ninety-five percent each give about sixty percent end to end, and token cost grows quadratically if I resend history. So an agent has to buy me real coverage that a graph cannot.

### Q2. Walk through the ReAct loop step by step.

ReAct interleaves reasoning and acting in one loop. Step one: you send the system prompt, the tool schemas, the user question, and the history so far. Step two: the model emits a thought and either a tool call with arguments or a final answer. Step three: if it is a final answer, you return and stop. Step four: otherwise you validate the arguments against the tool's schema. If validation fails you append the error as an observation and continue, so the model can correct itself. Step five: you execute the tool, truncate and validate the result, and append it to the history as an observation. Step six: you check the budgets, which are step count, token count, wall-clock time, and a repeat detector. Then you go back to step one. The history grows every turn, which is where the quadratic token cost comes from.

> **Say it.** ReAct is thought, action, observation, repeated. I send the prompt, the tool schemas, and the history. The model returns a thought plus either a tool call or a final answer. If it is a tool call, I validate the arguments against the schema, run the tool, truncate the result, and append it as an observation. Then I check my budgets: steps, tokens, wall clock, and whether it just repeated the same call. If none trip, I loop. The history grows each turn, so tokens grow quadratically.

### Q3. ReAct versus Plan-and-Execute. What is the trade-off?

ReAct decides one step at a time, so it sees each observation before choosing the next action. That makes it adaptive. Plan-and-Execute calls the model once to produce the whole plan, then runs the steps, often without the model in the loop for each one. The token argument favours planning. ReAct resends the full history every turn, so cost is $nb + dn(n+1)/2$, quadratic in $n$. Plan-and-Execute pays one large planning call plus $n$ small executions, which is roughly linear. It is also parallelisable, because independent plan steps can run at once, which cuts latency. The cost is rigidity. If step three returns something the plan did not anticipate, the plan is wrong and the remaining steps are wasted. The practical answer is a hybrid: plan, execute, and re-plan only when a step fails or returns something off-plan.

> **Say it.** ReAct picks one step at a time, so it adapts, but it resends the whole history and token cost goes quadratic in the number of steps. Plan-and-Execute pays one big planning call then runs the steps, so cost is roughly linear and independent steps can run in parallel. The weakness is that a surprise at step three invalidates the rest of the plan. In practice I plan, execute, and re-plan only on failure or when an observation contradicts the plan.

### Q4. What are the four common agentic patterns and what is each for?

Reflection: the model produces output, then critiques and revises it. It helps most where a mistake is easier to spot than to avoid, such as code that fails a test or a draft that misses a requirement. Tool use: the model calls external functions for facts, computation, or side effects, because it should not do arithmetic or recall live state from weights. Planning: the model decomposes a goal into ordered sub-goals before acting, which is what makes long tasks tractable and lets independent branches run in parallel. Multi-agent: separate roles with separate prompts and separate tool sets, coordinated by a supervisor or a shared state object. It helps when the roles need genuinely different context or permissions, for example a researcher with read access and a writer with none. The patterns compose. Reflection inside a planned multi-agent system is normal.

> **Say it.** Reflection: produce, critique, revise, which helps when errors are easier to detect than to prevent. Tool use: call external functions for facts, computation, and side effects instead of trusting the weights. Planning: decompose the goal into ordered sub-goals so long tasks become tractable and independent branches parallelise. Multi-agent: separate roles with their own prompts, tools, and permissions, coordinated by a supervisor. They compose. The one I question hardest is multi-agent, because it usually adds cost and handoff loss without adding independence.

### Q5. How do you design a tool so a model can use it correctly?

Three things: naming, schema, and error messages. The name and docstring are the only retrieval signal the model has, so name the tool for the user's intent, not the internal service. `search_customer_orders` beats `os_query_v2`. The docstring says what it returns and when to use it against the near neighbours. The schema is typed and constrained, not free-form strings. Enums instead of strings where the set is closed, bounded integers with `ge` and `le`, and explicit date formats in the field description. Constraints in the type mean the model cannot ask for fifty thousand rows. Error messages are written for the model, not for a human on-call. `customer_id must look like CUST-12345; strip names and emails` tells the model exactly what to change. `ValidationError at field 0` does not. Finally, keep the tool count small. Every extra tool raises the chance of picking the wrong one.

> **Say it.** The name and the docstring are the model's only retrieval signal, so I name for user intent and say in one line what the tool returns and when to prefer it over its neighbours. The schema is typed and constrained: enums for closed sets, bounded integers, explicit date formats. That way a bad call fails before it costs anything. Error strings are written for the model and say how to fix the call, not just that it broke. And I keep the tool list short, because more tools means more wrong picks.

### Q6. Why validate tool output before it reaches the model?

Three reasons, and they are separate. Cost and truncation: a query can return ten megabytes, which will blow the context window and the budget. So you cap length, page the results, and return a count plus the top rows rather than everything. Correctness: if the tool returns an error object or an empty result, you want that surfaced as a clean, explicit observation such as `no orders found for CUST-42`, because a raw stack trace makes the model hallucinate a recovery that will not work. Security: tool output is untrusted data. A retrieved document or a web page can contain text that reads like an instruction. If you paste it into the context unmarked, that is prompt injection. So you delimit it, label it as data, and strip or neutralise anything that looks like a control instruction. The rule is that nothing enters the context without passing a checkpoint you wrote.

> **Say it.** Three reasons. Size, because an unbounded result blows the context window and the budget, so I truncate and page. Clarity, because a raw stack trace makes the model invent a recovery, whereas a clean message like no orders found lets it act correctly. And security, because tool output is untrusted data and text fetched from a document or a web page can read as an instruction. I delimit it, label it as data, and never let anything into context without passing a checkpoint I wrote.

### Q7. How do you stop infinite loops and excessive tool calls?

Four independent controls, and the loop stops on whichever fires first. A step budget: a `for` loop with `max_steps`, never a `while True`. A token budget: track cumulative prompt plus completion tokens and stop at a ceiling, because that is what actually maps to money. A wall-clock timeout, because a slow tool can burn an hour without spending many tokens. And a repeat detector: hash the tool name plus its normalised arguments, and if the same hash appears twice in a row, or three times in the run, stop or force a different branch. Beyond the loop, add per-tool rate limits so one tool cannot be called two hundred times, and cap retries per tool call at three or so with backoff. On budget exhaustion, do not return silence. Return the partial state and an explicit failure so the caller or a human can take over.

> **Say it.** Four limits, first one wins. A step budget as a for loop, never a while True. A token budget, since that is what maps to money. A wall-clock timeout, because a slow tool burns time without burning tokens. And a repeat detector that hashes tool name plus arguments and stops on a duplicate. On top of that, per-tool rate limits and a retry cap of about three with backoff. When a budget trips, I return the partial state and an explicit failure, so a human or the caller can pick it up.

### Q8. Short-term versus long-term memory. What techniques manage the context window?

Short-term memory is the context window: the current run's messages, observations, and scratchpad. It is fast, exact, and it disappears when the run ends. Long-term memory is external storage that survives runs, usually a vector index for semantic recall plus a key-value store for facts. A vector index is a database that stores embeddings and returns the nearest ones to a query embedding. Redis is a common key-value store, an in-memory database used for fast reads of small facts and for session state. The context techniques are: sliding window, keeping the last $k$ turns; summarisation, replacing old turns with a compressed summary once you pass a token threshold; retrieval, pulling only the relevant slices back in on demand; and externalising state to a file or scratchpad that the agent reads and writes by tool call rather than holding in context.

> **Say it.** Short-term memory is the context window, which is exact and dies with the run. Long-term memory is external: a vector index, meaning a database of embeddings that returns the nearest ones to a query, plus a key-value store like Redis, which is an in-memory database for fast small reads. To manage context I use a sliding window over recent turns, summarise older turns once I pass a token threshold, retrieve only the relevant slices on demand, and push large state out to a file the agent reads through a tool.

### Q9. Why are multiple agents sharing one model not independent verifiers?

Because their errors are correlated. Two agents built on the same weights have the same training data, the same biases, and the same blind spots. If the model believes something false, both instances believe it, so asking a second instance to check the first mostly returns agreement. Independence is the whole assumption behind ensembling: if error rates are $\epsilon$ and truly independent, both being wrong has probability $\epsilon^2$. When the errors are correlated the joint error is close to $\epsilon$, so the ensemble buys you almost nothing while doubling your cost. What a second call does buy is sampling variance and a different framing. A critic prompt that asks for specific defects against explicit criteria catches format errors, missing requirements, and internal contradictions. Real verification needs an independent source: a unit test, a type checker, a database lookup, or a human.

> **Say it.** Because their errors are correlated. Same weights, same training data, same blind spots, so if the model believes something false, both copies believe it and the second one agrees. Independence is the entire assumption behind ensembling. If errors were independent at rate epsilon, both wrong is epsilon squared. Correlated, it stays near epsilon and I have doubled cost for nothing. A critic prompt still catches format and contradiction errors, but real verification needs an outside source: a test, a type checker, a database, or a person.

### Q10. How do you handle state and handoffs between agents, and what gets lost?

Use an explicit shared state object rather than passing prose messages. The state holds the original goal, structured intermediate results, a list of what has been tried and failed, open questions, and a confidence or provenance field per item. Each agent reads the fields it needs and writes back typed results. What gets lost in a handoff is everything not written down: the reasoning behind a choice, the alternatives already rejected, the uncertainty attached to a result, and the raw evidence behind a summary. The classic failure is the receiving agent treating an upstream guess as an established fact, because the summary dropped the hedge. So carry confidence and a pointer to the source with every claim, record failed attempts so the next agent does not repeat them, and keep the original user request verbatim in the state, because summarised goals drift step by step.

> **Say it.** I pass an explicit typed state object, not prose. It carries the original goal verbatim, structured results, what has already been tried and failed, open questions, and a confidence plus source for each claim. What gets lost otherwise is the reasoning, the rejected alternatives, the uncertainty, and the raw evidence. The classic bug is the next agent treating an upstream guess as fact because the summary dropped the hedge. Keeping confidence and provenance on every item is what prevents that.

### Q11. Where do you place human-in-the-loop gates?

By consequence and reversibility, not by model confidence. Build a two-by-two. Low consequence and reversible, such as a draft or a read query: no gate, let it run. Low consequence and irreversible, such as sending an internal notification: log it and sample-audit. High consequence and reversible, such as a config change with a one-click rollback: notify after the fact with an undo link. High consequence and irreversible, such as a payment, a deletion, or an email to a customer: gate before the action, always. The gate goes on the action, not on the reasoning, because reviewing a chain of thought is slow and humans rubber-stamp it. Show the human the exact call, the arguments, and the predicted effect, and make approving cost one click. Also gate on budget exhaustion and on low model confidence, and rate-limit gates so reviewers do not habituate and approve blindly.

> **Say it.** I place gates by consequence and reversibility, not by model confidence. Reversible and low stakes runs unattended. Reversible but high stakes gets a notification and an undo. Irreversible and high stakes, like a payment, a deletion, or an outbound email, gets a hard approval before the call. The gate sits on the action, not on the reasoning, because people rubber-stamp reasoning they cannot check. I show the exact call, the arguments, and the predicted effect, and I limit how often gates fire so reviewers do not habituate.

### Q12. What is MCP, and why does the 2026-07-28 revision matter for deployment?

MCP, the Model Context Protocol, is an open protocol that standardises how a model client connects to tool and data providers. Instead of writing a bespoke integration per model per tool, a server exposes tools, resources, and prompts once, and any compliant client can use them. That turns an N-times-M integration problem into N plus M. The 2026-07-28 revision makes the core protocol stateless. Sessions and the `initialize` handshake are removed, so a request no longer depends on server-side state established by an earlier request. The deployment consequence is direct. Previously any request had to reach the same server instance that held the session, which forced sticky routing, session affinity in the load balancer, and careful handling when an instance restarted or scaled down. Now any request can land on any instance. Therefore you can put servers behind a plain round-robin load balancer, autoscale them freely, and deploy them as ordinary stateless services or serverless functions.

> **Say it.** MCP is an open protocol that standardises how a model client talks to tool and data providers, so integrations go from N times M to N plus M. The 2026-07-28 revision made the core stateless: sessions and the initialize handshake are gone. That matters operationally, because before it, a request had to reach the same instance holding its session, which meant sticky routing and pain on restarts and scale-down. Now any request lands on any instance, so I can use plain round-robin, autoscale freely, and run servers as ordinary stateless or serverless workloads.

### Q13. An agent has ninety-five percent per-step reliability and needs twenty steps. What do you do?

First state the problem with the number: $0.95^{20} \approx 0.358$, so about thirty-six percent of runs finish correctly. Then attack $n$, then $p$, in that order. Reduce $n$ by collapsing sequences into single tools. If the agent always does three calls to answer one question, write one tool that does all three, because that converts three chances to fail into one. Reduce $n$ by moving deterministic work out of the loop entirely into ordinary code. Then raise $p$: split it as $p = q \cdot r$, measure tool-selection accuracy $q$ separately from tool success $r$, and fix whichever is worse. Better schemas and fewer, better-named tools raise $q$; retries with backoff and clearer errors raise $r$. Add verification so failures are caught rather than propagated. If you still cannot get there, add human gates at the irreversible steps and accept a supervised system.

> **Say it.** First I say the number out loud: point nine five to the twentieth is about thirty-six percent, so two runs in three fail. Then I cut n before I touch p, because n is the exponent. I collapse fixed call sequences into single tools and move deterministic work into plain code outside the loop. Then I split p into tool-selection accuracy times tool success rate, measure both, and fix the worse one. If it still will not reach target, I gate the irreversible steps with a human and ship it supervised.

### Q14. How do you debug an agent that gives a wrong final answer?

Read the trajectory, not the answer. Log every step: the full prompt sent, the model's thought, the tool name and arguments, the raw tool result, the truncated observation, tokens used, and latency. Then find the first step where the state went wrong, because everything after it is downstream noise. Classify that step into one of four failures. Wrong tool selected: fix the name, the description, or reduce the tool set. Right tool, wrong arguments: fix the schema, add constraints and examples. Right call, bad tool result: the bug is in the tool or its data, not the model. Right observation, wrong conclusion: that is a reasoning failure, and you address it with a better prompt, a reflection step, or a smaller step. Then turn that trajectory into a regression case in the eval suite, so the fix is protected.

> **Say it.** I read the trajectory, not the answer. Every step is logged: prompt, thought, tool name, arguments, raw result, truncated observation, tokens, latency. I find the first step where state went wrong, because everything after it is noise. Then I classify it: wrong tool, right tool with wrong arguments, right call with a broken tool result, or right observation with a wrong conclusion. Each one has a different fix, and only the last is a prompting problem. Finally I add that trajectory to the eval suite as a regression case.

## Done when

- You can state $p^n$ from memory, compute $0.95^{10} = 0.5987$ in your head to two digits, and derive $p = S^{1/n}$ on a whiteboard in under a minute.
- You can write the ReAct loop with a tool registry and a hard step budget from memory in under ten minutes, and it runs first time against a stub model.
- You can name all four loop-termination controls, and say which one catches a slow tool that spends no tokens.
- You can explain in three sentences why the 2026-07-28 stateless MCP core removes the need for sticky routing.
