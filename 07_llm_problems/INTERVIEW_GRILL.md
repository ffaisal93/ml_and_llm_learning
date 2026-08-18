# LLM Problems & Mitigations — Interview Grill

> 45 questions on long context, hallucination, prompting, jailbreaks, agents. Drill until you can answer 30+ cold.

---

## A. Long context

**1. Why does long context cost more?**
Attention is $O(L^2)$. KV cache is $O(L)$. Both scale with context.

> **Saying it out loud.** Two separate costs and it's worth splitting them. Compute is quadratic — attention compares every token to every other one, so doubling the context quadruples the work, and that lands on prefill, which is the part the user waits through. Memory is linear but still brutal: the KV cache for a single 128K request on a 70B model is around 41 gigabytes, more than a whole A100. The practical consequence: KV cache, not weights, is what caps how many long-context users you can serve at once.

**2. What's lost-in-the-middle?**
LLMs recall info at start and end better than middle. U-shaped recall vs position.

> **Saying it out loud.** It's the finding that recall over context position is U-shaped — the model uses information at the start and the end reliably and sags badly in the middle. It's not a small dip either; you can see over 50% recall at the edges and under 20% in the middle. The consequence people miss is that adding more context can make grounding worse, so a bigger window is a capacity limit, not a promise that anything gets read.

**3. Lost-in-the-middle mitigations?**
Place critical content at edges; structure with delimiters; train on long-context data; RAG instead of stuffing.

> **Saying it out loud.** Cheapest first: put the critical material at the top or the bottom and never bury it, and use clear delimiters so the boundaries are obvious. Then retrieve instead of stuffing — ten relevant chunks beat a thousand pages. Then long-context fine-tuning on data that genuinely requires attending to the middle. The point worth making is that reordering the prompt costs nothing and often buys more recall than doubling the window.

**4. Long context vs RAG?**
Both. RAG for huge corpora + freshness. Long context for in-context reasoning over retrieved chunks.

> **Saying it out loud.** Both, and saying "both" is the answer. RAG when the corpus is huge, when freshness matters, or when you need citations. Long context when the document fits and the reasoning genuinely needs all of it at once — you can't retrieve your way to "find the internal contradictions in this contract." In production it's usually RAG to find the chunks and a generous 32K window to reason over them. The tradeoff: RAG is cheaper and can miss; long context sees everything you give it, badly.

**5. RoPE NTK / YaRN purpose?**
Extend RoPE-based context beyond pre-training length without retraining.

> **Saying it out loud.** They stretch a RoPE-trained model past the length it was trained on without a full retrain — by rescaling the rotary frequencies, so positions the model never saw map into a range it understands. NTK-aware scaling and YaRN differ in how they scale across frequency bands, with YaRN being the more careful version. The tradeoff to name: extension buys you length and costs you some quality at short context, which is why you usually do a short fine-tune afterward to recover it.

---

## B. Hallucination

**6. Hallucination types?**
Factual, faithfulness (vs source), logical (internal inconsistency), source (invented citations).

> **Saying it out loud.** Four buckets, and they need different fixes, which is why bucketing matters. Factual — wrong about the world. Faithfulness — contradicts the document you handed it. Logical — the reasoning contradicts itself. Source — invented a citation. Retrieval fixes the first, entailment checking catches the second, and neither one touches self-contradictory reasoning.

**7. Why does the model hallucinate?**
Pattern matching produces plausible text; greedy decoding picks high-probability even when wrong; coverage gaps; distribution shift.

> **Saying it out loud.** Because it's a pattern-matcher optimized for plausibility, not truth — it produces text shaped like an answer whether or not it has one, and decoding commits to the top token even when the top two are nearly tied. Add coverage gaps and long-tail facts it never really memorized, plus distribution shift when your prompt looks nothing like training data. None of that is a bug; it's what next-token prediction does, which is why you contain it architecturally rather than waiting for it to be fixed.

**8. Does RAG eliminate hallucination?**
No. Reduces factual errors (when retrieval works) but model can still misinterpret sources or contradict them.

> **Saying it out loud.** No — it reduces hallucination substantially, roughly 50 to 80%, and it doesn't eliminate it. The model can still misread a passage, blend two sources, or add unsupported connective tissue that the context never stated. And there's a deeper limit: if retrieval surfaces something wrong, a perfectly faithful answer is still factually wrong. Faithfulness is the ML problem you can fix; factuality is a data problem RAG inherits.

**9. Self-consistency for hallucination?**
Sample multiple answers; majority vote. Errors uncorrelated → reduced via averaging.

> **Saying it out loud.** Sample the same question several times and take the majority answer. It works when the errors are uncorrelated — random mistakes scatter, the correct answer concentrates, so voting cleans it up. The failure mode is exactly the correlated case: if the model is confidently wrong because it memorized misinformation, every sample agrees and the vote confirms the error. Cost is K times inference, which is why it's an escalation tier rather than a default.

**10. Confidence calibration for hallucinations?**
Train model to say "I don't know" when uncertain. Metrics: token logprob, output entropy, post-hoc calibration.

> **Saying it out loud.** The goal is to make the model's stated confidence mean something so it can decline instead of guessing. You measure with log-probs and entropy, then fix with post-hoc calibration like temperature or Platt scaling on a held-out set, and train refusal on top. The thing to name: RLHF is what broke calibration in the first place — it rewarded confidence — so this step is mostly undoing damage from alignment, not adding something new.

**11. Detect hallucination automatically?**
SelfCheckGPT (consistency across samples), NLI-based, fact-check against retrieved sources.

> **Saying it out loud.** Three families. If you have a source, it's entailment — does the retrieved context support each claim. If you don't, you make the model give itself away by sampling several times and checking consistency, which is SelfCheckGPT, or by clustering samples by meaning, which is semantic entropy. And underneath both there's a nearly-free tier of token log-probs and entropy. The caveat on that cheap tier is that post-RLHF calibration is unreliable, so it's a feature in a classifier, not a gate.

---

## C. Prompting

**12. Zero-shot vs few-shot?**
Zero-shot: just instruction. Few-shot: include examples.

> **Saying it out loud.** Zero-shot is just the instruction; few-shot adds worked examples. The useful framing is what each buys: few-shot mostly buys format compliance and task disambiguation, not reasoning ability — which is why adding examples to a hard math problem usually does nothing. Modern instruction-tuned models need few-shot much less than 2022 models did, so the main remaining reason to use it is pinning down an output format you can't easily describe.

**13. Chain-of-thought (CoT)?**
"Think step by step" — reasoning before answer. Improves math/logic.

> **Saying it out loud.** Ask for the reasoning before the answer — "think step by step" — and accuracy on multi-step problems jumps. The mechanism is that it gives the model tokens to compute in: without them it has to produce the answer in a single forward pass, and with them it can decompose. The tradeoffs are latency, cost, and the compounding-error problem — long chains multiply per-step error rates, so more reasoning helps up to a point and then hurts.

**14. Self-consistency?**
Sample multiple CoT chains; majority vote. Better than single CoT.

> **Saying it out loud.** Sample several independent chains of thought and take the majority final answer. It beats a single chain because different chains fail in different ways, so the wrong answers scatter while the right one concentrates. Cost is linear in the number of samples, and the limitation is the same as always — if the model is systematically wrong about the setup, every chain inherits it and the vote just makes you more confident about the error.

**15. Tree of Thoughts?**
Explore multiple reasoning paths, backtrack from dead ends. For complex problems.

> **Saying it out loud.** Instead of one linear chain, explore several branches, evaluate partial states, and backtrack out of dead ends — search over reasoning rather than a single pass. It genuinely helps on puzzle-like problems with a clear notion of a promising partial state. The reason it's rare in production is cost: you're paying for many branches plus an evaluator at each node, and for most tasks self-consistency gets you most of the benefit for a fraction of the tokens.

**16. ReAct?**
Interleave Reason + Act + Observe. Agent loop with tool use.

> **Saying it out loud.** Reason, act, observe, repeat — the model thinks about what it needs, calls a tool, sees the result, and thinks again. The reason it's the workhorse agent pattern is that it's reactive, so it recovers when the world doesn't match the plan. Compare with upfront planning, which is more coherent over long horizons and often wrong by step three. Most production agents are still ReAct with a step limit.

**17. Prompt sensitivity?**
Small wording changes shift benchmark scores 5-10 points. Need robustness testing.

> **Saying it out loud.** Rewording a prompt without changing its meaning can move benchmark scores five to ten points, and few-shot ordering matters too. So any comparison between two models — or two prompt versions — is meaningless unless the spread across paraphrases is smaller than the difference you're claiming. The rule to state: measure your prompt's variance before you trust its mean, and version prompts like code.

**18. System prompt structure?**
Role → instructions → constraints → examples → context → user query.

> **Saying it out loud.** Role, instructions, constraints, examples, context, then the user query. The ordering isn't stylistic — it's driven by lost-in-the-middle and by caching. Static content up front gets attended to reliably and can be cached by the provider so you pay for it once. And the query goes last because the end of the context is the other high-attention region. The mistake is burying constraints after a long RAG dump, which is exactly where they'll be ignored.

---

## D. Jailbreaks and safety

**19. Common jailbreak patterns?**
Roleplay (DAN), authority claim, encoding (base64), multi-turn drift, indirect injection.

> **Saying it out loud.** Five families, named by mechanism. Roleplay — talk the model into a character with no rules. Authority — "I'm a researcher, this is for safety work." Encoding — base64 or leetspeak so the input filter doesn't see the words. Multi-turn drift — nothing in any single message is objectionable and the trajectory is. And indirect injection, which is the one that actually matters in production because the user never typed it.

**20. Indirect prompt injection?**
Malicious instructions embedded in retrieved documents or tool outputs. Hard to defend.

> **Saying it out loud.** It's when the malicious instruction lives inside content the model reads rather than content the user typed — a retrieved document, a web page, a tool result. It's hard to defend because your input filter never sees it and the model has no reliable way to distinguish data from instructions inside its own context. The real defense isn't detection, it's permissions: treat retrieved content as untrusted data and cap what the agent can do, so a successful injection can't spend money or delete anything.

**21. Universal adversarial suffix (Zou et al. 2023)?**
A suffix optimized to make model comply with harmful instructions; transfers across models.

> **Saying it out loud.** It's a token string found by gradient optimization against open models that, appended to a harmful request, makes the model comply — and the striking part is that it transfers to models the attacker never had weights for, including closed ones. That's the finding that killed the idea that safety training is a defensible boundary. The takeaway is that this is adversarial risk management, not a bug with a fix, and it's why output-side filtering matters.

**22. Defense layers?**
Input classifier, system prompt hardening, output classifier, action permission limits.

> **Saying it out loud.** Four layers, and no layer holds alone. Input classifier for known attack patterns. System prompt hardening. Output classifier that blocks harmful completions regardless of how they were elicited — that one matters most, because it doesn't need to anticipate the attack. And permission limits on what actions the model can actually take, which is the only thing that contains indirect injection. Defense in depth, because every individual layer is bypassable.

**23. Why do jailbreaks persist?**
Adversarial co-evolution; helpful-harmless trade-off; new attack patterns constantly.

> **Saying it out loud.** Because it's adversarial and asymmetric — defenses get published and attackers iterate in days while model updates take months. Underneath that is a real tension: helpful and harmless conflict, so every increment of caution costs legitimate refusals, and there's no setting that maximizes both. And transferable adversarial suffixes mean an attacker doesn't even need access to your model. It's an equilibrium to manage, not a problem to close.

**24. Constitutional AI principle?**
Self-critique against principles; revise iteratively. Less RLHF data needed.

> **Saying it out loud.** The model critiques its own output against a written set of principles and revises, so the alignment signal comes from stated rules plus AI feedback rather than from a mountain of human preference labels. The practical benefit is scale and auditability — you can read the constitution and see what the model is being asked to optimize. The limit: it shapes style and behavior far more than knowledge, so it reduces confident wrongness rather than wrongness.

---

## E. Agents and tool use

**25. Tool use mechanics?**
LLM outputs structured tool call (JSON); runtime executes; result fed back; LLM continues.

> **Saying it out loud.** The model emits a structured tool call, your runtime executes it, and the result goes back into the context so the model can continue. The important thing to say is that the model never runs anything — it produces JSON and your code decides whether to honor it. That's where schema validation and permissions live, and it's why "the model called a tool that doesn't exist" is a runtime-handling problem rather than a model problem.

**26. Common tool types?**
Search, code execution, database query, file system, API calls.

> **Saying it out loud.** Search for freshness, code execution for math and data, database queries for structured facts, file access, and general API calls for actions. The way to pick is to ask what the model reliably gets wrong alone — arithmetic and current events, mostly — so a calculator and a search tool cover the bulk of hallucination. The tradeoff is blast radius: read-only tools are nearly free to add, anything that writes or spends money needs confirmation and scoped permissions.

**27. Single-step vs multi-step agent?**
Single-step: ReAct loop. Multi-step: plan upfront, execute. Hierarchical: planner + executor.

> **Saying it out loud.** ReAct interleaves thinking and acting one step at a time, so it adapts when a tool returns something unexpected. Upfront planning produces the whole sequence first, which is more coherent over long horizons and brittle when reality diverges from the plan. Hierarchical splits the two so a planner handles structure and an executor handles steps. The tradeoff: planning buys coherence and loses adaptability, and most shipped agents are ReAct with a step limit for that reason.

**28. Agent failure modes?**
Wrong tool, malformed args, infinite loops, context bloat, hallucinated tools, cascading errors.

> **Saying it out loud.** Wrong tool, malformed arguments, infinite loops, context bloat from huge tool outputs, hallucinated tools, and cascading errors. They're all mundane, which is the point — agents fail boringly, not dramatically. Cascading is the expensive one because the math compounds: 95% reliability per step over twenty steps is about 36% end to end. That number is the best single thing to say about why agents are still brittle.

**29. Mitigations?**
Strict schemas + validation, step limits, output truncation/summarization, clear tool descriptions, human-in-loop for risky actions.

> **Saying it out loud.** Validate every tool call against a strict schema and retry with the error message attached, which recovers most malformed calls in one round. Cap iterations so a loop is bounded. Truncate or summarize long tool outputs before they hit context. Write tool descriptions that say when *not* to use each one. And put a human in front of anything irreversible. The framing: you're not making the model reliable, you're making its failures cheap and recoverable.

**30. Multi-agent architecture?**
Specialist agents (researcher, writer, critic) collaborate. More structured than single-agent.

> **Saying it out loud.** Specialist agents — researcher, writer, critic — each with their own prompt and tools, passing work between them. The appeal is separation of concerns and a natural place for self-critique. The honest assessment is that it usually multiplies cost and failure surface without a proportional quality gain, because every handoff is a place to lose context. The one pattern that reliably earns its cost is a separate critic, since judging is easier than generating.

---

## F. Multi-turn

**31. Memory strategies?**
Append all (simple, bloats), sliding window (forgets), summarization (lossy), external memory (retrieval over user history).

> **Saying it out loud.** Four options on one tradeoff axis. Append everything is perfect fidelity until you run out of context and budget. Sliding window is cheap and forgets whatever fell off — usually the setup from turn one. Summarization keeps the gist and drops the specifics like names and numbers, which are exactly what people expect you to remember. External memory scales and introduces a retrieval problem. In practice: recent turns verbatim, older turns summarized, durable facts written to a store.

**32. Style drift across turns?**
Model adapts to user's style/opinions over time. Can lead to sycophancy.

> **Saying it out loud.** Over a long conversation the model drifts toward the user's style and, worse, toward the user's opinions — that's sycophancy, and it's a behavior failure no capability benchmark catches. It's dangerous in advice products precisely because the model isn't wrong in an obvious way, it's telling the user what they signaled they wanted. Test for it by asserting a wrong answer mid-conversation and seeing whether the model flips.

**33. How to keep critical facts across turns?**
External memory (retrievable), fact extraction + re-injection, fine-tuned summarizer.

> **Saying it out loud.** Don't rely on the context window to remember — extract the facts and store them. Write durable items like names, preferences, and constraints to an external store and re-inject the relevant ones each turn, rather than hoping they survive summarization. A fine-tuned summarizer helps if you must compress. The framing point: treat long-term memory as a retrieval problem with an explicit write step, not as a side effect of keeping the transcript.

---

## G. Latency and cost

**34. TTFT vs ITL?**
TTFT: time to first token (prefill). ITL: inter-token latency (decode).

> **Saying it out loud.** Time to first token is prefill — processing the whole prompt at once, compute-bound, and it's what the user waits through before anything appears. Inter-token latency is decode, which is memory-bandwidth-bound because you read the model's weights to produce each token. They optimize differently, which is the point: a long prompt hurts TTFT and shortening it does nothing for ITL.

**35. Prefill bottleneck?**
Compute (matrix multiplies on full sequence).

> **Saying it out loud.** Compute. Prefill processes every prompt token in parallel, so it's big matrix multiplies and it saturates the GPU's math units — which is also why it's quadratic in length. The lever is fewer tokens: shorten the prompt, or reuse a cached prefix so you skip recomputing it entirely.

**36. Decode bottleneck?**
Memory bandwidth (per-token reads of KV cache).

> **Saying it out loud.** Memory bandwidth. Decoding one token means reading the entire model's weights plus the KV cache to produce a single output, so you're bandwidth-bound rather than compute-bound and the GPU's math units sit mostly idle. That's why batching helps throughput so much — you amortize one weight read across many sequences — and why speculative decoding works, since verifying several tokens costs about the same read as generating one.

**37. Prompt caching?**
Cache long prompt prefixes; subsequent requests reuse the prefix's KV cache. Cuts cost + latency.

> **Saying it out loud.** The provider keeps the KV cache for a prompt prefix, so repeated requests that share it skip prefill for that portion — cutting both cost and time to first token substantially. It's the highest-leverage optimization for chat and RAG products because a long static system prompt gets charged once instead of every turn. The design implication: put everything stable at the top and everything variable at the bottom, or you break the cache on every request.

**38. Smaller-model fallback?**
Route easy queries to cheap model; only escalate hard ones to flagship.

> **Saying it out loud.** Route the easy majority to a small cheap model and reserve the frontier model for the hard tail — usually the single largest cost reduction available, since most traffic is easy. The hard part is the router: it has to decide before it knows the answer, so you either train a classifier on difficulty or use a cheap-model confidence signal to escalate. The failure mode is a router that under-escalates, where quality dies quietly on exactly the queries that mattered.

**39. Streaming benefits?**
Lower perceived latency; user reads tokens as they arrive.

> **Saying it out loud.** It doesn't reduce latency at all — total generation time is identical — it reduces *perceived* latency, because the user starts reading at the first token instead of waiting for the last. That's a huge UX win for anything long-form, and it's basically free. The tradeoff worth mentioning: you can't run an output filter on text you've already streamed, so streaming and pre-publish safety checking are in tension.

---

## H. Evaluation

**40. Why is LLM eval hard?**
Open-ended; subjective; many valid answers; benchmark contamination.

> **Saying it out loud.** Because there's no single right answer, so you can't check equality — every metric is a proxy with its own bias. Outputs are open-ended, quality is partly subjective, many different responses are equally good, and benchmarks get contaminated within months of publication. The framing that lands: an eval is a measurement instrument with bias and variance, and most teams under-invest in the instrument relative to the model.

**41. LLM-as-judge — risk?**
Self-preference bias (judges own outputs higher). Use external strong model as judge.

> **Saying it out loud.** The one people name is self-preference — a judge rates its own family's outputs higher — and the fix is a judge from a different family than either contestant, or an ensemble of three. But I'd list the others too, because they're bigger in practice: length bias, position bias where option A wins more often, and format bias toward bullets and bold. Position bias you fix by swapping the order and only counting agreeing votes. And none of it counts unless the judge has been calibrated against human labels.

**42. Pairwise human preference?**
Show two responses; ask which is better. Aggregates to ELO ratings (Chatbot Arena).

> **Saying it out loud.** Show two anonymized responses, ask which is better, and aggregate the votes into ratings — that's Chatbot Arena. Pairwise works because comparison is a much easier judgment than absolute scoring, so agreement is higher. The aggregation is Bradley-Terry fitted in batch rather than online ELO, which avoids order dependence and gives you confidence intervals. Its limitation is prompt distribution: Arena voters skew casual, so it says little about your domain.

**43. Faithfulness for RAG?**
Does response stay true to retrieved sources? NLI-based or LLM-judge.

> **Saying it out loud.** Faithfulness asks whether the response stays inside what was retrieved, and you measure it by decomposing the answer into atomic claims and checking each for entailment against the context. That's RAGAS faithfulness. The distinction to draw unprompted is faithfulness versus factuality — if the retrieved document is wrong, a perfectly faithful answer is still false. Faithfulness is what you can monitor continuously from your own logs; factuality needs an external source.

**44. Test contamination problem?**
Public benchmark answers leak into training data. Inflated scores without real progress.

> **Saying it out loud.** The benchmark ended up in the training data, so the score measures memorization rather than capability — and it's the default assumption for any public benchmark more than a year or two old. It gets in four ways: crawled from the web, pasted into forums, arriving secondhand through instruction-tuning or synthetic data, or deliberately. The cheapest check is a perturbation test: rephrase the questions and see if accuracy falls off a cliff.

**45. Capability-specific evals?**
Code: HumanEval / SWE-Bench. Math: MATH / GSM8K / AIME. Long context: RULER. Reasoning: BBH.

> **Saying it out loud.** Match the eval to the capability: HumanEval-plus and SWE-Bench-Verified for code, MATH and AIME for math, RULER for long context, BBH or GPQA for reasoning. Code is the privileged case because execution decides — no judge, no ambiguity — which is exactly why verifiable-reward RL works there. And the caveat I'd add: all of it is a sanity check on the model, and for a product, a few hundred prompts from your own traffic decides more than the whole list.

---

## Quick fire

**46.** *Lost in the middle — shape?* U.
**47.** *Self-consistency — for what?* Reasoning errors.
**48.** *CoT trigger phrase?* "Step by step."
**49.** *Tool call format?* Structured (JSON).
**50.** *Agent termination criterion?* Step limit + success signal.
**51.** *Indirect injection source?* Retrieved content / tool output.
**52.** *Adversarial suffix transferability?* Across models.
**53.** *Prompt cache benefit?* Speed + cost.
**54.** *LLM-as-judge bias?* Self-preference.
**55.** *Long-context architecture?* Sliding window / linear attention / SSM.

---

## Self-grading

If you can't answer 1-15, you don't know LLM problems. If you can't answer 16-30, you'll struggle on production LLM questions. If you can't answer 31-45, frontier-lab questions on agents / safety will go past you.

Aim for 35+/55 cold.
