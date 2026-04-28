# LLM Problems & Mitigations — Interview Grill

> 45 questions on long context, hallucination, prompting, jailbreaks, agents. Drill until you can answer 30+ cold.

---

## A. Long context

**1. Why does long context cost more?**
Attention is $O(L^2)$. KV cache is $O(L)$. Both scale with context.

**2. What's lost-in-the-middle?**
LLMs recall info at start and end better than middle. U-shaped recall vs position.

**3. Lost-in-the-middle mitigations?**
Place critical content at edges; structure with delimiters; train on long-context data; RAG instead of stuffing.

**4. Long context vs RAG?**
Both. RAG for huge corpora + freshness. Long context for in-context reasoning over retrieved chunks.

**5. RoPE NTK / YaRN purpose?**
Extend RoPE-based context beyond pre-training length without retraining.

---

## B. Hallucination

**6. Hallucination types?**
Factual, faithfulness (vs source), logical (internal inconsistency), source (invented citations).

**7. Why does the model hallucinate?**
Pattern matching produces plausible text; greedy decoding picks high-probability even when wrong; coverage gaps; distribution shift.

**8. Does RAG eliminate hallucination?**
No. Reduces factual errors (when retrieval works) but model can still misinterpret sources or contradict them.

**9. Self-consistency for hallucination?**
Sample multiple answers; majority vote. Errors uncorrelated → reduced via averaging.

**10. Confidence calibration for hallucinations?**
Train model to say "I don't know" when uncertain. Metrics: token logprob, output entropy, post-hoc calibration.

**11. Detect hallucination automatically?**
SelfCheckGPT (consistency across samples), NLI-based, fact-check against retrieved sources.

---

## C. Prompting

**12. Zero-shot vs few-shot?**
Zero-shot: just instruction. Few-shot: include examples.

**13. Chain-of-thought (CoT)?**
"Think step by step" — reasoning before answer. Improves math/logic.

**14. Self-consistency?**
Sample multiple CoT chains; majority vote. Better than single CoT.

**15. Tree of Thoughts?**
Explore multiple reasoning paths, backtrack from dead ends. For complex problems.

**16. ReAct?**
Interleave Reason + Act + Observe. Agent loop with tool use.

**17. Prompt sensitivity?**
Small wording changes shift benchmark scores 5-10 points. Need robustness testing.

**18. System prompt structure?**
Role → instructions → constraints → examples → context → user query.

---

## D. Jailbreaks and safety

**19. Common jailbreak patterns?**
Roleplay (DAN), authority claim, encoding (base64), multi-turn drift, indirect injection.

**20. Indirect prompt injection?**
Malicious instructions embedded in retrieved documents or tool outputs. Hard to defend.

**21. Universal adversarial suffix (Zou et al. 2023)?**
A suffix optimized to make model comply with harmful instructions; transfers across models.

**22. Defense layers?**
Input classifier, system prompt hardening, output classifier, action permission limits.

**23. Why do jailbreaks persist?**
Adversarial co-evolution; helpful-harmless trade-off; new attack patterns constantly.

**24. Constitutional AI principle?**
Self-critique against principles; revise iteratively. Less RLHF data needed.

---

## E. Agents and tool use

**25. Tool use mechanics?**
LLM outputs structured tool call (JSON); runtime executes; result fed back; LLM continues.

**26. Common tool types?**
Search, code execution, database query, file system, API calls.

**27. Single-step vs multi-step agent?**
Single-step: ReAct loop. Multi-step: plan upfront, execute. Hierarchical: planner + executor.

**28. Agent failure modes?**
Wrong tool, malformed args, infinite loops, context bloat, hallucinated tools, cascading errors.

**29. Mitigations?**
Strict schemas + validation, step limits, output truncation/summarization, clear tool descriptions, human-in-loop for risky actions.

**30. Multi-agent architecture?**
Specialist agents (researcher, writer, critic) collaborate. More structured than single-agent.

---

## F. Multi-turn

**31. Memory strategies?**
Append all (simple, bloats), sliding window (forgets), summarization (lossy), external memory (retrieval over user history).

**32. Style drift across turns?**
Model adapts to user's style/opinions over time. Can lead to sycophancy.

**33. How to keep critical facts across turns?**
External memory (retrievable), fact extraction + re-injection, fine-tuned summarizer.

---

## G. Latency and cost

**34. TTFT vs ITL?**
TTFT: time to first token (prefill). ITL: inter-token latency (decode).

**35. Prefill bottleneck?**
Compute (matrix multiplies on full sequence).

**36. Decode bottleneck?**
Memory bandwidth (per-token reads of KV cache).

**37. Prompt caching?**
Cache long prompt prefixes; subsequent requests reuse the prefix's KV cache. Cuts cost + latency.

**38. Smaller-model fallback?**
Route easy queries to cheap model; only escalate hard ones to flagship.

**39. Streaming benefits?**
Lower perceived latency; user reads tokens as they arrive.

---

## H. Evaluation

**40. Why is LLM eval hard?**
Open-ended; subjective; many valid answers; benchmark contamination.

**41. LLM-as-judge — risk?**
Self-preference bias (judges own outputs higher). Use external strong model as judge.

**42. Pairwise human preference?**
Show two responses; ask which is better. Aggregates to ELO ratings (Chatbot Arena).

**43. Faithfulness for RAG?**
Does response stay true to retrieved sources? NLI-based or LLM-judge.

**44. Test contamination problem?**
Public benchmark answers leak into training data. Inflated scores without real progress.

**45. Capability-specific evals?**
Code: HumanEval / SWE-Bench. Math: MATH / GSM8K / AIME. Long context: RULER. Reasoning: BBH.

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
