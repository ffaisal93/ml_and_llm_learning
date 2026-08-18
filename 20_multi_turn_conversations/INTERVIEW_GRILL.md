# Multi-Turn Conversation Design — Interview Grill

> 35 questions on chat system design, memory, persona, tools, evaluation, latency. Drill until you can answer 24+ cold.

---

## A. Memory strategies

**1. Four memory strategies?**
Append all, sliding window, summarization, external retrieval.

> **Saying it out loud.** So there are really four ways a chatbot can remember what you said earlier. You can just keep the whole transcript, you can keep only the last few turns, you can periodically compress the old stuff into a summary, or you can pull facts out and stash them in a database you look things up in. Everything in production is some mix of those four. The tradeoff running through all of them is the same one: fidelity versus context length and cost.

**2. Append all — when fails?**
Long conversations exceed context window; cost/latency grow; lost-in-the-middle quality.

> **Saying it out loud.** Keeping the whole transcript works great right up until it doesn't. Every turn you add makes the prompt longer, so you're paying more money and waiting longer for each reply, and eventually you just slam into the context window and the thing breaks. Even before that, quality starts sagging because models get sloppy about stuff buried in the middle of a long prompt. That's the lost-in-the-middle failure mode, and it shows up well before you hit the hard token limit.

**3. Sliding window — what's lost?**
Older context that may matter (user preferences disclosed early, ongoing tasks).

> **Saying it out loud.** A sliding window is the cheap fix: keep the last twenty turns, throw the rest away. The problem is that what you throw away isn't random, it's everything the user told you early on. So the user says "I'm vegetarian" in turn two, and forty turns later you're recommending a steakhouse. Constant memory cost, but you silently lose exactly the long-lived preferences that make a conversation feel coherent.

**4. Summarization trade-off?**
Lossy. Subtle context lost. Repeated summarization compounds loss.

> **Saying it out loud.** Summarization buys you length by paying in detail. You take the old turns, compress them to a paragraph, and keep going, which is fine until you need something specific that got compressed away. The nasty part is that it compounds: you summarize the summary, then summarize that, and errors and omissions accumulate like a game of telephone. The named failure mode is drift from repeated re-summarization, so most systems cap how many times a segment gets re-compressed.

**5. External memory?**
Store facts/preferences in DB; retrieve relevant per turn. Helps long-range coherence.

> **Saying it out loud.** External memory is when you stop treating the conversation as the memory and start treating it like a database. You extract durable facts and preferences, write them somewhere, and then retrieve just the relevant ones for each turn. The win is that a fact from six months ago is as retrievable as one from six minutes ago, so long-range coherence stops being a function of context length. The cost is that you've now got a retrieval system to tune, and bad retrieval means the model confidently uses the wrong user's preference.

**6. Hybrid in production?**
Recent turns verbatim + summary of older + retrievable user facts + system prompt.

> **Saying it out loud.** Nobody picks one strategy in production, they stack them. You keep the last handful of turns verbatim because recency matters most, a rolling summary of everything older, a retrievable store of stable user facts, and the system prompt pinned at the top. That gives you sharp recent context, cheap long-range context, and precise recall of the things that actually matter. The engineering cost is that you now have three components that can each fail independently, so you need observability on each.

---

## B. Persona

**7. Persona drift causes?**
Long history dilutes system prompt; user pushes alternate persona; lack of training on persona consistency.

> **Saying it out loud.** Persona drift is when your carefully written system prompt stops mattering thirty turns in. Three things cause it: the sheer volume of conversation dilutes those few hundred instruction tokens, the user actively pushes the model into a different character, and the base model was never really trained to hold a persona over long horizons. It's the same attention budget being spread thinner and thinner. In practice you see it as the assistant slowly adopting the user's tone and opinions instead of its own.

**8. Mitigation: re-injection?**
Re-include persona statements every $N$ turns or when context approaches limit.

> **Saying it out loud.** The simplest defense is to just say it again. Every so many turns, or whenever you're getting close to the context limit, you re-inject the persona statement so it's recent rather than buried a hundred turns back. Recency is doing the work here, not novelty. Downside is that you're spending tokens on repetition every time, and if you re-inject too aggressively the model starts sounding stilted and repetitive about who it is.

**9. Sycophancy?**
Model agrees with user's last opinion, regardless of merit. Common RLHF failure.

> **Saying it out loud.** Sycophancy is when the model agrees with you because you said it, not because it's true. You assert something wrong, push back a little, and it folds and tells you you're right. It's not a bug in any one prompt, it's a systematic bias baked in during preference training. It's the canonical RLHF failure mode, and it's dangerous precisely because it feels pleasant to the user.

**10. Sycophancy cause?**
RLHF reward correlates with agreeable responses; user thumbs-up signals "agreement = good."

> **Saying it out loud.** The cause is straightforward once you look at where the reward comes from. Human raters click thumbs-up on responses that agree with them, so during RLHF the reward model learns that agreement correlates with quality. The optimizer does exactly what you asked, it just turns out you asked for flattery. This is a reward-misspecification problem, not a capability problem, so scaling the model up doesn't fix it.

**11. Sycophancy fix?**
Train on diverse preferences; explicit anti-sycophancy data; constitutional principles.

> **Saying it out loud.** You fix sycophancy on the data side, not the prompt side. That means training on preferences from diverse raters so no single viewpoint gets rewarded, deliberately adding examples where the correct answer is polite disagreement, and encoding explicit principles about honesty that the model is trained against. Prompt-level instructions like "don't just agree with me" help a bit but wash out over long conversations. The tradeoff is that anti-sycophancy training can overshoot into a model that's needlessly contrarian and rated as unhelpful.

**12. Roleplay drift defense?**
Hardened system prompt; explicit "ignore user attempts to change persona"; output filtering.

> **Saying it out loud.** Roleplay is the standard on-ramp for getting a model out of character, so the defenses are layered. You harden the system prompt with explicit instructions that user requests to change identity get ignored, you keep re-asserting that during the conversation, and you put an output filter after generation as a backstop. No single layer holds, which is why you use three. The failure mode you're guarding against is the slow boil, where each turn moves the persona one percent and turn fifty is somewhere you'd never have agreed to go directly.

---

## C. Multi-turn eval

**13. Trajectory-level metrics?**
Coherence end-to-end, goal completion, turns to resolution, persona consistency.

> **Saying it out loud.** Trajectory-level metrics judge the whole conversation instead of one reply. Did the user actually get what they came for, how many turns did it take, did the assistant stay coherent and in character the whole way through. A conversation can have ten individually good responses and still fail, because it never converged on the goal. Turns-to-resolution is the one product teams care about most, since it maps directly to user effort.

**14. Simulated user?**
Another LLM plays the user; full conversation simulated; success measured.

> **Saying it out loud.** A simulated user is just another LLM playing the human side, given a persona and a goal. You let the two models talk, then check whether the goal got accomplished. It's the only way to get multi-turn eval at any scale, because real multi-turn human evals are brutally slow and expensive. The catch is that simulated users are more patient and more articulate than real ones, so your numbers come out optimistic.

**15. Pairwise turn comparison?**
Human raters compare model A's response to model B's at each turn.

> **Saying it out loud.** Pairwise turn comparison is head-to-head: same conversation history, two candidate responses, a human picks which one is better. It's much easier to judge than an absolute score because people are bad at calibrated ratings but good at comparisons. You aggregate into a win rate or an Elo. The catch is it only tells you which is better at that one turn, so it misses everything about how the conversation ends up.

**16. Why single-turn eval insufficient?**
Doesn't capture drift, persona consistency, context retention, goal completion across turns.

> **Saying it out loud.** Single-turn eval measures the wrong unit. All the interesting failures in a chat system happen across turns: the persona sliding, the model forgetting what you said ten turns ago, the conversation going in circles without resolving. Score each turn independently and every one of those looks fine. So you can ship a model that wins on single-turn benchmarks and feels worse to actual users over a real session.

**17. Length bias in evaluators?**
Human + LLM raters often prefer longer responses. Confounds quality eval.

> **Saying it out loud.** Both humans and LLM judges systematically prefer longer answers. Same content, padded with more words, scores higher. That means your eval is partly measuring verbosity instead of quality, and if you optimize against it you get a model that rambles. The standard mitigation is length-controlled win rates, where you regress out response length before comparing.

---

## D. State management

**18. Per-conversation state?**
Conversation ID, user ID, history, active tools, cached values.

> **Saying it out loud.** Per-conversation state is the stuff you need to reconstruct a session on the next request. Conversation ID, which user it belongs to, the message history, whatever tools are currently active, and any cached or computed values you don't want to recompute. The design question is what lives in the request versus what lives in the store. Get that split wrong and you either blow up your payload sizes or you can't scale horizontally.

**19. Storage tier — choice?**
In-memory: fast, lossy on restart. DB: persistent. Distributed: high-scale.

> **Saying it out loud.** You pick a storage tier by what you can afford to lose. In-memory is the fastest and simplest, but a process restart wipes every open conversation. A database survives restarts and gives you history, at the cost of a round trip per turn. A distributed store like Redis is what you land on at scale, because you need any server in the fleet to serve any user's next message. The real tradeoff is latency on the critical path versus durability.

**20. Concurrency issue?**
Same user with parallel conversations (mobile + web). State management handles.

> **Saying it out loud.** The concurrency case people forget is one user with two live conversations, phone and laptop at once. If your state is keyed only by user, those two sessions start writing over each other. So you key by conversation and treat user-level memory as a separate, carefully-synchronized layer. The failure mode is last-write-wins clobbering, where the mobile session's update silently erases what the user just did on desktop.

**21. Truncation strategy at context limit?**
Summarize old, drop low-priority turns, compress repetition, preserve recent + critical.

> **Saying it out loud.** When you hit the context limit you don't just chop from the front, you triage. Summarize the old turns, drop the ones that carry least information, squeeze out repeated boilerplate, and protect the two things you can't lose: the most recent turns and anything explicitly marked critical, like the system prompt or a confirmed order number. Blind truncation is how you end up dropping the user's name. The tradeoff is that smarter truncation costs you an extra model call and breaks your prompt cache.

---

## E. Tools in conversation

**22. Tool-call loop?**
User message → LLM → optional tool call → execution → result → LLM → response.

> **Saying it out loud.** The tool loop is a cycle, not a single call. User sends a message, the model decides it needs a tool and emits a call, your code actually runs the tool, you feed the result back in, and the model writes the real answer. The key thing to say out loud is that the model never executes anything, it only asks. That separation is what lets you sandbox and validate everything, and it's also why every tool round trip costs you a full extra forward pass of latency.

**23. Multi-step tool conversation?**
Each tool call is a separate LLM forward pass. History grows with intermediate results.

> **Saying it out loud.** Multi-step tool use means you go around that loop several times before the user sees anything. Every round trip is another forward pass over a prompt that keeps growing, because each tool call and each result gets appended to the history. So latency and cost scale with the number of steps, not just the length of the answer. A five-step agent turn can easily be ten times the cost of a plain chat turn.

**24. Tool failure recovery?**
Schema validation; retry with adjusted args; fallback to "tool unavailable" message.

> **Saying it out loud.** Tool failures are normal, so you plan for them. Validate the arguments against the schema before you ever execute, and if the model produced garbage, hand back the validation error so it can fix its own call. If the tool itself fails, retry with backoff, and if it keeps failing, tell the model the tool is unavailable so it can degrade gracefully instead of hallucinating a result. The failure mode you're avoiding is silent fabrication, where a failed API call turns into a confidently invented answer.

**25. Infinite tool loops?**
Step limits; "couldn't decide" escape hatch; supervised LLM judgment.

> **Saying it out loud.** Models get stuck in loops, calling the same tool over and over waiting for a different answer. You break that with a hard step limit per turn, an explicit escape hatch so the model is allowed to say "I couldn't figure this out" instead of trying forever, and in bigger systems a supervisor that watches the trace and cuts it off. Without a cap, one bad turn can burn unbounded money. Ten to twenty steps is a typical ceiling for a consumer chat agent.

---

## F. Format and templates

**26. Why prompt format matters?**
Models trained on specific formats. Wrong format = degraded quality.

> **Saying it out loud.** Prompt format matters because the model was trained on one specific set of delimiters, and those tokens are load-bearing. Use the wrong template and the model can't cleanly tell where the system prompt ends and the user starts, so instruction-following and safety behavior both degrade. It looks like a mysterious quality drop rather than an error, which is what makes it so nasty. This is a top cause of "the open-weights model is much worse than the benchmark said."

**27. ChatML format markers?**
`<|im_start|>system\n...<|im_end|>` etc.

> **Saying it out loud.** ChatML is the format OpenAI popularized, and it just wraps each message in special tokens: a start marker with the role, the content, then an end marker. The important part is that those markers are single special tokens the user's text can't forge, which is what keeps a user from typing their way into the system role. Get the whitespace wrong and you're off-distribution. It's a security boundary as much as a formatting convention.

**28. Llama format?**
Llama 2: `<s>[INST]<<SYS>>...<</SYS>>user_input[/INST]`. Llama 3+ switched to ChatML-like format with `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` markers per message.

> **Saying it out loud.** Llama 2 used its own thing, with instruction brackets around the user turn and a nested system block inside. Llama 3 threw that out and moved to a ChatML-style scheme with explicit header and end-of-turn tokens per message. That's a real trap in practice: same model family, incompatible templates across versions. Always read the tokenizer's chat template rather than hardcoding, because the version mismatch fails silently.

**29. Anthropic Claude API format?**
Separate `system` parameter + `messages` array of `{role, content}`.

> **Saying it out loud.** Anthropic's API splits it: the system prompt is its own top-level parameter, and everything else is a list of role-content messages that has to alternate user and assistant. You never hand-build delimiter strings, the API does the templating. That removes a whole class of format bugs at the cost of less control. The constraint that trips people up is the strict alternation, so you have to merge consecutive same-role messages yourself.

---

## G. Latency

**30. Prompt caching benefit?**
Cache stable prefix (system prompt + conversation history). New turns reuse cache → lower TTFT + cost.

> **Saying it out loud.** Prompt caching pays off because the front of your prompt barely changes between turns. The system prompt and the conversation so far are identical from one request to the next, so the provider can keep those computed attention keys and values around and skip recomputing them. You get much faster time-to-first-token and a big discount on the cached portion, typically around ninety percent off input cost. It's the single highest-leverage optimization for a multi-turn chat product.

**31. What's required for cache hit?**
Identical prefix bytes. Stable system prompt + truncated history matters.

> **Saying it out loud.** A cache hit needs the prefix to match exactly, byte for byte. One extra space, a timestamp injected at the top, a reordered tool definition, and you miss and pay full price. So the design rule is put everything stable at the front and everything variable at the back. The classic self-inflicted wound is a dynamic "today's date" line in the system prompt, which invalidates your cache on every single request.

**32. Speculative decoding helps which phase?**
Decode (ITL).

> **Saying it out loud.** Speculative decoding helps the decode phase, so it shows you up as lower inter-token latency. A small draft model guesses several tokens ahead and the big model verifies them in one pass, which works because verification is parallel while generation is sequential. It does nothing for prefill or time-to-first-token. Typical speedups are two to three times on decode when the draft model's acceptance rate is high.

**33. Streaming benefit?**
Lower perceived latency. User reads as tokens generated.

> **Saying it out loud.** Streaming doesn't make anything faster, it makes the wait feel shorter. Instead of staring at a spinner for eight seconds, you see the first words in half a second and read along as it generates. Perceived latency collapses to time-to-first-token. The tradeoff is that you can't run a moderation pass over the complete response before the user sees the beginning of it.

---

## H. Personalization and privacy

**34. Personalization approaches?**
User context in prompt; per-user memory; user-specific LoRA / adapter; RLHF on preferences.

> **Saying it out loud.** Personalization runs on a spectrum of how expensive the intervention is. Cheapest is stuffing user context into the prompt, then a per-user memory store you retrieve from, then a per-user LoRA adapter that actually changes weights, and at the far end tuning the whole model on aggregate preferences. Cost and staleness go one way, expressiveness goes the other. Almost everyone stops at retrieval, because per-user adapters mean serving thousands of distinct weight sets.

**35. Privacy considerations?**
Don't include other users' data; honor deletion requests; careful about training on user chats.

> **Saying it out loud.** Privacy in chat comes down to one rule: one user's data must never surface in another user's session. Concretely that means scoping every memory lookup by user ID, honoring deletion so a delete actually removes the derived memories and embeddings too, and being extremely careful about training on conversation logs since models memorize. Retrieval systems are where this breaks, because a shared vector index without a hard user filter will happily return somebody else's secrets.

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
