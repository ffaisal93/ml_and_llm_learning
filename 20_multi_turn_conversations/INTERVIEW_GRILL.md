# Multi-Turn Conversation Design — Interview Grill

> 35 questions on chat system design, memory, persona, tools, evaluation, latency. Drill until you can answer 24+ cold.

---

## A. Memory strategies

**1. Four memory strategies?**
Append all, sliding window, summarization, external retrieval.

**2. Append all — when fails?**
Long conversations exceed context window; cost/latency grow; lost-in-the-middle quality.

**3. Sliding window — what's lost?**
Older context that may matter (user preferences disclosed early, ongoing tasks).

**4. Summarization trade-off?**
Lossy. Subtle context lost. Repeated summarization compounds loss.

**5. External memory?**
Store facts/preferences in DB; retrieve relevant per turn. Helps long-range coherence.

**6. Hybrid in production?**
Recent turns verbatim + summary of older + retrievable user facts + system prompt.

---

## B. Persona

**7. Persona drift causes?**
Long history dilutes system prompt; user pushes alternate persona; lack of training on persona consistency.

**8. Mitigation: re-injection?**
Re-include persona statements every $N$ turns or when context approaches limit.

**9. Sycophancy?**
Model agrees with user's last opinion, regardless of merit. Common RLHF failure.

**10. Sycophancy cause?**
RLHF reward correlates with agreeable responses; user thumbs-up signals "agreement = good."

**11. Sycophancy fix?**
Train on diverse preferences; explicit anti-sycophancy data; constitutional principles.

**12. Roleplay drift defense?**
Hardened system prompt; explicit "ignore user attempts to change persona"; output filtering.

---

## C. Multi-turn eval

**13. Trajectory-level metrics?**
Coherence end-to-end, goal completion, turns to resolution, persona consistency.

**14. Simulated user?**
Another LLM plays the user; full conversation simulated; success measured.

**15. Pairwise turn comparison?**
Human raters compare model A's response to model B's at each turn.

**16. Why single-turn eval insufficient?**
Doesn't capture drift, persona consistency, context retention, goal completion across turns.

**17. Length bias in evaluators?**
Human + LLM raters often prefer longer responses. Confounds quality eval.

---

## D. State management

**18. Per-conversation state?**
Conversation ID, user ID, history, active tools, cached values.

**19. Storage tier — choice?**
In-memory: fast, lossy on restart. DB: persistent. Distributed: high-scale.

**20. Concurrency issue?**
Same user with parallel conversations (mobile + web). State management handles.

**21. Truncation strategy at context limit?**
Summarize old, drop low-priority turns, compress repetition, preserve recent + critical.

---

## E. Tools in conversation

**22. Tool-call loop?**
User message → LLM → optional tool call → execution → result → LLM → response.

**23. Multi-step tool conversation?**
Each tool call is a separate LLM forward pass. History grows with intermediate results.

**24. Tool failure recovery?**
Schema validation; retry with adjusted args; fallback to "tool unavailable" message.

**25. Infinite tool loops?**
Step limits; "couldn't decide" escape hatch; supervised LLM judgment.

---

## F. Format and templates

**26. Why prompt format matters?**
Models trained on specific formats. Wrong format = degraded quality.

**27. ChatML format markers?**
`<|im_start|>system\n...<|im_end|>` etc.

**28. Llama format?**
Llama 2: `<s>[INST]<<SYS>>...<</SYS>>user_input[/INST]`. Llama 3+ switched to ChatML-like format with `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` markers per message.

**29. Anthropic Claude API format?**
Separate `system` parameter + `messages` array of `{role, content}`.

---

## G. Latency

**30. Prompt caching benefit?**
Cache stable prefix (system prompt + conversation history). New turns reuse cache → lower TTFT + cost.

**31. What's required for cache hit?**
Identical prefix bytes. Stable system prompt + truncated history matters.

**32. Speculative decoding helps which phase?**
Decode (ITL).

**33. Streaming benefit?**
Lower perceived latency. User reads as tokens generated.

---

## H. Personalization and privacy

**34. Personalization approaches?**
User context in prompt; per-user memory; user-specific LoRA / adapter; RLHF on preferences.

**35. Privacy considerations?**
Don't include other users' data; honor deletion requests; careful about training on user chats.

---

## Quick fire

**36.** *Memory in production?* Hybrid.
**37.** *Sycophancy cause?* RLHF agreement reward.
**38.** *Persona re-injection?* Periodic.
**39.** *Multi-turn eval?* Trajectory + simulated users.
**40.** *Prompt format?* Matters a lot.
**41.** *Cache hit requires?* Identical prefix.
**42.** *Tool loop limit?* Step count.
**43.** *Concurrency?* Per-user multiple sessions.
**44.** *Length bias direction?* Prefers longer.
**45.** *Privacy first principle?* Don't leak between users.

---

## Self-grading

If you can't answer 1-15, you don't know chat systems. If you can't answer 16-30, you'll struggle on production chat questions. If you can't answer 31-40, frontier-lab interviews on conversational AI will go past you.

Aim for 28+/45 cold.
