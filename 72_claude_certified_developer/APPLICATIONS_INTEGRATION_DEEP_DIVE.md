# Applications & Integration Deep Dive (33.1%)

> The single heaviest domain on any Claude cert. This is the "can you actually build it" domain: the Messages API in code, tools, streaming, vision, extended thinking, caching, batching, and — critically — **error handling and async**. Examples are in Python (the TS SDK mirrors them).

---

## 1. The Messages API in code

Install and make a call:

```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY from env

resp = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain a hash map in two sentences."}],
)
print(resp.content[0].text)
print(resp.usage)  # input_tokens / output_tokens — watch these
```

Non-negotiable facts (see also `../71_claude_certified_architect/CLAUDE_API_DEEP_DIVE.md`):
- **Stateless.** No memory between calls; to continue a conversation you append the assistant reply and resend the whole `messages` list.
- **`system` is a top-level field**, not a message; send it every request.
- **`max_tokens` caps output only.** Too low → truncated reply with `stop_reason: "max_tokens"`.
- **`content` is a list of blocks.** Text, images, `tool_use`, `tool_result`, thinking — all are content blocks. Don't assume `content[0]` is always text (it may be a `tool_use`).

### The `usage` object — your cost/latency dashboard
Every response includes `usage` with `input_tokens`, `output_tokens`, and (with caching) `cache_creation_input_tokens` and `cache_read_input_tokens`. Reading `usage` is how you *measure* the optimizations in the next file — get in the habit.

### Stop reasons drive control flow
`end_turn` (done), `max_tokens` (raise cap / continue), `tool_use` (run the tool), `stop_sequence` (hit your stop string), `pause_turn`/`refusal`. Your loop branches on `resp.stop_reason`.

---

## 2. Multi-turn: you rebuild the history

```python
messages = [{"role": "user", "content": "Hi, I'm debugging a race condition."}]
resp = client.messages.create(model="claude-sonnet-5", max_tokens=512, messages=messages)
messages.append({"role": "assistant", "content": resp.content})   # append the reply
messages.append({"role": "user", "content": "It only happens under load."})  # next turn
resp = client.messages.create(model="claude-sonnet-5", max_tokens=512, messages=messages)
```

The conversation continues **only because you resend `messages`.** This is the #1 conceptual anchor for the whole exam.

---

## 3. Tool use in code (the agent loop)

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Use when the user asks about weather.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

messages = [{"role": "user", "content": "Weather in Austin?"}]
while True:
    resp = client.messages.create(
        model="claude-sonnet-5", max_tokens=1024, tools=tools, messages=messages
    )
    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason != "tool_use":
        break
    # find each tool_use block, run it, return a tool_result
    tool_results = []
    for block in resp.content:
        if block.type == "tool_use":
            result = run_tool(block.name, block.input)   # YOUR code executes it
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
                # "is_error": True   # set when the tool failed, so Claude can adapt
            })
    messages.append({"role": "user", "content": tool_results})

print(messages[-1])
```

Memorize the loop: **call → if `tool_use`, execute and append `tool_result`, repeat → else done.** The model asks; your code acts. `tool_choice` modes (`auto`/`any`/`tool`/`none`) control whether/which tool is forced — details in `TOOLS_AND_MCP.md`.

---

## 4. Streaming (SSE)

For chat UIs, stream tokens so the user sees output immediately:

```python
with client.messages.stream(
    model="claude-sonnet-5", max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about caches."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()  # full message + usage once done
```

Streaming changes **delivery**, not content. Use it to cut *perceived* latency. Note: very long responses may **require** streaming to avoid client timeouts. You reassemble deltas into the final message (the SDK helper does this for you).

---

## 5. Vision (multimodal input)

Current models accept images alongside text (screenshots, documents, charts). Output is text.

```python
messages = [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64",
        "media_type": "image/png", "data": b64_png}},
    {"type": "text", "text": "What error is shown in this screenshot?"},
]}]
```

Developer notes: images cost tokens roughly by resolution — downscale huge images; put the instruction *after* the image; for multi-page scans consider batching.

---

## 6. Extended (adaptive) thinking

Let the model reason internally before answering hard problems:

```python
resp = client.messages.create(
    model="claude-sonnet-5", max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2000},
    messages=[{"role": "user", "content": "Prove there are infinitely many primes."}],
)
```

Trade-off: better multi-step reasoning at the cost of extra thinking tokens + latency. Enable for planning/math/complex debugging; leave off for simple, latency-sensitive calls. (More in `MODEL_SELECTION_OPTIMIZATION.md`.)

---

## 7. Prompt caching in code

Cache a large stable prefix so repeat requests are cheaper/faster. You mark a breakpoint with `cache_control`:

```python
resp = client.messages.create(
    model="claude-sonnet-5", max_tokens=1024,
    system=[
        {"type": "text", "text": LONG_STABLE_INSTRUCTIONS,
         "cache_control": {"type": "ephemeral"}},   # cache everything up to here
    ],
    messages=[{"role": "user", "content": user_turn}],  # volatile, not cached
)
# check resp.usage.cache_read_input_tokens to confirm a cache hit
```

Rules that show up as questions:
- **Stable content first, volatile last.** Only an unchanged prefix hits the cache.
- **Any change to the cached prefix invalidates it** — don't inject per-request timestamps into cached regions.
- Great for: long system prompts, tool catalogs, big reference docs reused across calls.

---

## 8. Message Batches API (async bulk)

For large, non-interactive jobs, submit many requests as a batch — **~50% cheaper**, high throughput, results within a processing window (not instant).

```python
batch = client.messages.batches.create(requests=[
    {"custom_id": f"doc-{i}",
     "params": {"model": "claude-haiku-4-5-20251001", "max_tokens": 512,
                "messages": [{"role": "user", "content": doc}]}}
    for i, doc in enumerate(docs)
])
# poll batch.processing_status, then retrieve results by custom_id
```

Decision rule: **interactive → synchronous; bulk + latency-tolerant → Batches.** Stack with caching + a cheap model + structured schema for cheap large-scale extraction/classification.

---

## 9. Error handling & async — where developers actually get graded

This is the part architects gloss over and developers must nail. Know the HTTP error classes and the correct code response.

| Status | Meaning | Correct handling |
|---|---|---|
| `400` invalid_request | Bad params/schema | **Fix the request**; don't retry unchanged |
| `401` authentication | Bad/missing API key | Fix credentials; never retry-loop |
| `403` permission | Key lacks access | Escalate; don't retry |
| `404` not_found | Wrong resource | Fix the reference |
| `413` request_too_large | Payload/context too big | Trim context / split |
| `429` rate_limit | Too many requests | **Retry with exponential backoff**; respect `retry-after` |
| `500` api_error | Server error | Retry with backoff (bounded) |
| `529` overloaded | Service overloaded | **Back off and retry**; consider a fallback model |

**Retry pattern (exponential backoff + jitter):**

```python
import time, random
from anthropic import APIStatusError, RateLimitError

def call_with_retry(**kw):
    for attempt in range(5):
        try:
            return client.messages.create(**kw)
        except (RateLimitError,) as e:            # 429
            wait = min(2 ** attempt + random.random(), 30)
            time.sleep(wait)
        except APIStatusError as e:               # 5xx/529 retryable; 4xx not
            if e.status_code in (500, 529) and attempt < 4:
                time.sleep(min(2 ** attempt, 30)); continue
            raise
    raise RuntimeError("exhausted retries")
```

Principles the exam rewards:
- **Retry only retryable errors** (429/5xx/529) with **exponential backoff + jitter**; never hammer with fixed-interval retries.
- **Do not blindly retry non-idempotent side effects.** If a request that caused a *write* times out, use an **idempotency key** or check state first (double-charge trap — see the Architect reliability file).
- **4xx are your bug**, not transient — fix the request, don't loop.
- **Async / concurrency:** use the async client (`AsyncAnthropic`) or a bounded worker pool for many calls; cap concurrency to stay under rate limits; prefer the **Batches API** for large offline sets instead of hand-rolling thousands of concurrent calls.
- **Timeouts & streaming:** set sensible client timeouts; stream long generations to avoid idle-connection timeouts.

---

## 10. Worked example: a robust extraction service endpoint

```python
async def extract(doc: str) -> dict:
    resp = await call_with_retry_async(
        model="claude-sonnet-5", max_tokens=1024,
        system=[{"type": "text", "text": EXTRACTION_INSTRUCTIONS,
                 "cache_control": {"type": "ephemeral"}}],   # cache stable prefix
        output_config={"format": SCHEMA},                    # constrained output
        messages=[{"role": "user", "content": doc}],
    )
    data = parse(resp)
    if not semantically_valid(data):        # syntactic ≠ semantic
        data = correct(doc, data, errors())  # correction, not blind retry
    if low_confidence(data):
        route_to_human(doc, data)            # don't ship a guess
    return data
```

For the 50k-document backlog, the same logic goes through the **Batches API** with a **Haiku** model. Every choice — schema, caching, retry policy, batch, human routing — is a gradeable developer decision.

---

## 11. Rapid-fire self-check

1. Why does a conversation "remember" earlier turns? *(You resend `messages`; the API is stateless.)*
2. What does `max_tokens` limit? *(Output tokens only.)*
3. In the tool loop, what condition continues the loop? *(`stop_reason == "tool_use"`.)*
4. Which errors do you retry with backoff, and which don't you? *(429/5xx/529 yes; 4xx no.)*
5. A timed-out write — safe to retry? *(Only with idempotency/state check.)*
6. Where do you put volatile content for caching to work? *(At the end; keep the cached prefix stable.)*
7. Bulk 50k offline extractions — which API? *(Message Batches, ~50% cheaper.)*
8. What does `usage.cache_read_input_tokens` tell you? *(That the cached prefix was hit.)*

---

## 12. Further reading

- Messages API — `https://platform.claude.com/docs/en/api/messages`
- Client SDKs — `https://platform.claude.com/docs/en/api/client-sdks`
- Streaming — `https://platform.claude.com/docs/en/build-with-claude/streaming`
- Vision — `https://platform.claude.com/docs/en/build-with-claude/vision`
- Extended thinking — `https://platform.claude.com/docs/en/build-with-claude/extended-thinking`
- Prompt caching — `https://platform.claude.com/docs/en/build-with-claude/prompt-caching`
- Batch processing — `https://platform.claude.com/docs/en/build-with-claude/batch-processing`
- Errors & rate limits — `https://platform.claude.com/docs/en/api/errors`, `https://platform.claude.com/docs/en/api/rate-limits`
