# Multi-Turn Conversation Design — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Multi-turn chat is now the dominant LLM interface. Designing conversational systems brings together long context, memory, persona consistency, agent loops, and serving — all the LLM problems compounded by *time*. Senior interviews probe this when they want to test product-engineering judgment in the LLM era.

---

## 1. The conversation lifecycle

A multi-turn chat system processes:

1. **System prompt**: persona, instructions, constraints, format.
2. **Conversation history**: prior user messages + assistant responses.
3. **User message**: current turn.
4. **Optional context**: RAG retrieved docs, tool outputs, user metadata.
5. → **LLM produces response**.
6. **State updates**: append turn, possibly trigger tools, update memory.

The technical challenge: each of these layers has design decisions, failure modes, and trade-offs.

---

## 2. Memory management strategies

The core tension: more history = more context for the model, but also more cost, latency, and risk of "lost in the middle."

### Strategy 1: Append everything

Append every turn to context. Simple. Fails when conversation gets long (cost, latency, lost-in-middle, hard context limit).

### Strategy 2: Sliding window

Keep last $K$ turns; drop earlier. Simple, predictable cost. Loses long-range coherence.

```python
def sliding_window_context(history, K=10):
    return history[-K:]
```

### Strategy 3: Summarization

Periodically summarize older turns into a condensed form.

```python
if len(history) > threshold:
    old, recent = history[:-keep_n], history[-keep_n:]
    summary = llm.summarize(old)
    return [summary] + recent
```

Trade-off: summary is lossy; subtle context lost.

### Strategy 4: External memory / retrieval

Store facts/preferences in a database. Retrieve relevant ones per turn.

```python
def get_context(user_id, current_message):
    relevant_memories = vector_db.search(query=current_message, user_id=user_id)
    return relevant_memories
```

Trade-off: retrieval can miss context; needs careful indexing strategy.

### Hybrid (most production systems)

- Last $K$ turns verbatim.
- Summary of older turns.
- Retrievable user-fact store (preferences, facts disclosed).
- System prompt with key context.

---

## 3. Persona / character consistency

Common failure: model forgets its persona or contradicts itself across turns.

### Mitigations
- **Strong system prompt**: clear role, constraints, do/don't.
- **Periodic re-injection**: include key persona elements every N turns.
- **Style transfer fine-tuning**: train on persona-consistent dialogues.
- **Constitutional principles**: AI follows explicit principles (safer than persona).

### Specific failure modes
- **Sycophancy**: model agrees with user's last opinion. Mitigate via diverse training data.
- **Roleplay drift**: user pushes model into different persona ("pretend you're..."). Defense: hardened system prompt.
- **Length drift**: responses get shorter/longer over conversation. Mitigate via length specification.
- **Style drift**: tone changes. Mitigate via explicit style instructions.

---

## 4. Multi-turn evaluation

Hard. Single-turn evals don't capture conversational dynamics.

### Conversation-level metrics
- **Coherence**: does the conversation make sense end-to-end?
- **Goal completion**: did the user accomplish their task?
- **Turns to resolution**: efficient or rambling?
- **User satisfaction**: explicit ratings or proxy (length of session).
- **Persona consistency**: model stays in character.

### Methods
- **Simulated user**: another LLM plays the user; measure success.
- **Pairwise turn comparison**: human raters compare model A vs B turn-by-turn.
- **Trajectory comparison**: full conversation A vs B.
- **Production telemetry**: turn count, abandonment rate, satisfaction proxies.

### Pitfalls
- Test contamination: chatbot training data overlaps with eval users.
- Evaluator preference for verbose responses (longer = better is a common bias).
- Single-turn evals don't catch drift.

---

## 5. State management at scale

### Per-conversation state
- Conversation ID.
- User ID.
- History (or pointer to history).
- Active tools / context.
- Cached values.

### Storage tier
- **In-memory**: fast, lost on restart. Fine for short conversations.
- **Persistent (DB)**: longer conversations across sessions.
- **Distributed**: for high-scale serving.

### Concurrency
Same user may have multiple parallel conversations (mobile + web). State management needs to handle this.

### Context truncation strategy
When approaching context limit:
1. Summarize older turns.
2. Drop low-priority turns (e.g., simple acknowledgments).
3. Compress repetitive content.
4. Preserve recent turns + critical context.

---

## 6. Tool use in conversations

Tools let the LLM access external info during conversation.

### Standard tool-call loop

```
User message → LLM → optional tool call → tool execution → result → LLM → response → User
```

### Multi-tool, multi-step conversations

```
User: "Book a flight to NYC for next Friday and reserve a hotel."
LLM: search_flights(NYC, next_friday) → results
LLM: search_hotels(NYC, dates) → results
LLM: present options to user
User: "Book the 9am flight and the Marriott."
LLM: book_flight(...) → confirmation
LLM: reserve_hotel(...) → confirmation
LLM: respond to user
```

Each tool call is a separate LLM forward pass; the conversation history grows with results.

### Failure modes
- **Tool selection drift**: model forgets it has access to tools.
- **Tool result format issues**: malformed JSON, unexpected types.
- **Cascading errors**: bad tool result confuses subsequent reasoning.
- **Infinite tool loops**: model can't decide when to stop.

### Mitigations
- Strict schema validation.
- Step / iteration limits.
- Tool result summarization for long outputs.
- Periodic re-injection of tool list in system prompt.

See `07_llm_problems/` for full agent treatment.

---

## 7. Prompt template structure

Production chat systems use templates like:

```
<|system|>
You are a helpful assistant. Follow these rules: ...
[user-specific context: name, preferences, history summary]
[tool definitions]

<|history|>
[turn 1: user]
[turn 1: assistant]
[turn 2: user]
[turn 2: assistant]
...

<|user|>
[current message]

<|assistant|>
```

### Format conventions
- ChatML (OpenAI): `<|im_start|>system\n...<|im_end|>`
- Llama 2: `<s>[INST]<<SYS>>...<</SYS>>...[/INST]`
- Llama 3+: ChatML-like with `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>` headers per message
- Anthropic Claude: separate `system`, `messages` parameters.

### Why format matters
Models are trained on specific formats. Wrong format → degraded quality.

---

## 8. Latency strategy for multi-turn

### TTFT vs ITL (see paged-attention deep dive)
- TTFT bottleneck: prefill of full conversation history.
- ITL bottleneck: decoding new tokens.

### Mitigations
- **Prompt caching**: cache prefix KV across turns. Long stable system prompt + history → cache hit.
- **Truncation strategies**: keep cached prefix; vary only the recent + new.
- **Speculative decoding**: faster ITL.
- **Streaming**: improve perceived latency.

---

## 9. Personalization

### Per-user customization
- **System prompt with user info**: name, preferences, relevant context.
- **User-specific embeddings**: small custom adaptation per user.
- **User memory store**: facts disclosed across sessions.

### Learning from feedback
- **RLHF on user preferences**: gather thumbs up/down; train.
- **Personalized fine-tuning**: per-user LoRA or adapter.
- **Trade-off**: more personalized = better experience but harder ops.

### Privacy
- Don't include other users' data.
- Honor delete requests.
- Be careful about training on user chats.

---

## 10. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Memory strategy? | "Append all" | Hybrid: recent verbatim + summary + retrievable facts |
| Persona drift fix? | Stronger model | Periodic system-prompt re-injection; persona-consistent training |
| Multi-turn eval — single-turn metric works? | Yes | No — need trajectory-level metrics, simulated users, persona consistency |
| Sycophancy cause? | Bug | RLHF reward correlates with user agreement; needs targeted training |
| Long conversations cost? | Constant | Per-turn cost grows with history (linear or quadratic for attention) |
| Tool failure handling? | Model handles it | Strict schemas + retry + fallback in system layer |
| Prompt format matters? | Not really | Yes — model trained on specific format; wrong format degrades quality |

---

## 11. Eight most-asked multi-turn questions

1. **Design a chat system for our product.** (Memory strategy, persona, tools, eval, serving — full stack.)
2. **How do you manage conversation history at scale?** (Sliding window + summarization + retrieval; hybrid.)
3. **How do you keep the assistant on-character?** (System prompt; periodic re-injection; persona-trained data.)
4. **Multi-turn evaluation — how?** (Simulated users, pairwise turn comparison, persona consistency, trajectory metrics.)
5. **What are the latency optimizations for long conversations?** (Prompt caching of stable prefix, speculative decoding, streaming.)
6. **How does tool use compose with conversation?** (Each tool call adds to history; iteration limits; result summarization.)
7. **Personalization without breaking privacy?** (User-specific context in prompt; per-user memory; respect deletion.)
8. **Why does the bot get more sycophantic over a conversation?** (RLHF reward correlates with agreement; train on diverse preferences.)

---

## 12. Drill plan

- For "design a chat system" — practice 5-minute end-to-end answer.
- For each memory strategy, recite trade-offs.
- For each conversation failure mode (sycophancy, drift, persona), recite cause + mitigation.
- For tool integration, walk through 2-step flow with failure recovery.
- Practice tail-latency optimization recipes for chat (caching, speculative).

---

## 13. Further reading

- Anthropic, *Building effective agents* (2024) — cookbook for chat + agents.
- OpenAI assistants API documentation — production patterns.
- LangChain, LlamaIndex documentation — open-source chat frameworks.
- Liu et al. (2023), *Lost in the Middle* — long-context recall.
- Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models.*
