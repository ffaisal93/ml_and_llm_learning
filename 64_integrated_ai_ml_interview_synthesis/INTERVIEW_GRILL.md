# Cross-Topic Synthesis — Interview Grill

> 40 questions on cross-topic synthesis: how to bridge theory, systems, methodology, and judgment. Drill until you can answer 28+ cold.

---

## A. Five archetypes

**1. Five common interview-question archetypes?**
Design X, Train X, Why does X work, Debug X, Trade off X vs Y.

**2. "Design X" — what framework?**
6 steps: clarify → frame → data → features+model → serving → monitoring. From `29_system_design_for_ml`.

**3. "Train X" — what framework?**
Frontier training playbook: scaling laws → architecture → data mixture → stages → evaluation. From `62`.

**4. "Why does X work" — what framework?**
First principles: identify the relevant principle (bias-variance, scaling laws, optimization geometry); reduce to it.

**5. "Debug X" — what framework?**
Decision tree: data → model → evaluation → drift → cost asymmetry. Common failures by frequency.

**6. "Trade off X vs Y" — what framework?**
Identify business priorities; explain the curve; pick the operating point with justification.

---

## B. Cross-topic bridges

**7. Cross-entropy connects to which topics?**
LM losses (`43`), RLHF (`08`), distillation (`25`), VAE (`33`), clustering (`19`).

**8. Bias-variance connects to?**
Classical theory (`27`), SLT (`52`), regularization (`11`), estimation (`47`).

**9. Embeddings connect to?**
Retrieval/RAG (`39`), recommendations (`22`), multimodal (`38`), tokenization (`15`), search (`36`).

**10. Attention connects to?**
Transformer (`04`), long context (`07`), KV cache + serving (`63`), positional encoding (`14`).

**11. Data curation connects to?**
Frontier training (`62`), SFT/RLHF (`08`), RAG corpora (`39`), anomaly (`32`).

**12. Why does cross-topic fluency matter?**
Frontier interviews ask multi-part questions where each part lives in a different deep dive. Senior candidates see the connections.

---

## C. Synthesis question patterns

**13. "Build LLM Q&A system" — what topics?**
RAG, prompting, agents, safety, evaluation, serving, A/B testing.

**14. "Why does scale work?" — what topics?**
Scaling laws, bias-variance, over-parameterization, implicit regularization, double descent.

**15. "Reduce model latency by 2x" — what topics?**
Quantization (FP8/INT8), distillation, speculative decoding, KV cache, batching, prompt caching.

**16. "Improve metric without retraining" — what topics?**
Prompting, sampling, post-processing, calibration, retrieval augmentation, threshold tuning.

**17. "Production regression" — what topics?**
Distribution shift, data quality, model rollback, infra failures, evaluation pitfalls.

**18. "Online/offline mismatch" — what topics?**
Position bias, counterfactual eval, label time leakage, novelty effect, distribution shift.

---

## D. The "first principles" pattern

**19. State the goal — why first?**
Determines the whole answer. Without it, you're guessing what to optimize.

**20. State constraints early — why?**
Most ML decisions are constraint-driven (latency, data, cost). Frames trade-offs.

**21. Apply dominant principle — what does it mean?**
Reduce the question to one major framework: scaling laws, bias-variance, cost asymmetry. Anchor the answer.

**22. Recommend a baseline — why?**
Pragmatic. Shows you'd ship something simple before complicating. Strong baselines are often strong enough.

**23. Iterate up — what's the priority order?**
By expected impact / effort. Not every improvement is worth its complexity.

**24. Failure modes — why end with these?**
Shows judgment. Senior candidates see what could go wrong; junior candidates assume it'll work.

**25. Strongest defended conclusion — why?**
"All of the above" loses points. Pick one, justify it, mention alternatives for context.

---

## E. Common mistakes

**26. Listing without judgment — example?**
"You could use logistic regression, decision tree, GBDT, or NN." Picks one, justifies with a reason.

**27. Naming without explaining — example?**
"I'd use FlashAttention." Better: "FlashAttention because $O(L^2) \to O(L)$ memory; matters at our 32K context."

**28. Over-engineering — example?**
Multi-tower transformer + GNN when GBDT would suffice. Pragmatism scores.

**29. Missing failure modes — what to do?**
Always close with: "could fail when X; mitigate via Y."

**30. Forgetting the business — what to do?**
Tie ML answer to user-facing outcome. Senior interviewers value product judgment.

---

## F. Topic-bridging cheatsheet

**31. RLHF in one breath?**
SFT for format → reward model from preferences → PPO/DPO/GRPO; KL penalty prevents drift; reward hacking is the threat.

**32. Production ML pipeline in one breath?**
Data ingestion → features → training → eval → A/B → deployment → monitoring → retraining.

**33. LLM stack in one breath?**
Pre-train → mid-train (long context, code/math) → SFT → preference optimization → eval → serve.

**34. RAG in one breath?**
Index docs → embed query → ANN retrieve → optional rerank → prompt template → LLM → response with citations.

**35. Recommender stack in one breath?**
Two-tower retrieval (ANN over embeddings) → ranker (GBDT or DL) → diversity/exploration → A/B test.

---

## G. Synthesis under pressure

**36. 5-min answer to "design X" — what to cover?**
1 min clarification + 1 min frame + 1.5 min architecture + 1 min eval + 30s monitoring/iteration.

**37. 30-second answer to a tough question?**
"Three things matter here: A, B, C. The dominant one is A because [reason]. I'd start with [solution] and refine if [signal]."

**38. When asked for opinion?**
Have one. State it clearly. Justify with reasoning. Acknowledge where reasonable people disagree.

**39. "What's the next frontier?"**
Reasoning, agents, multimodal, efficiency, alignment robustness. Pick 2 with substance: "Reasoning because [trend]; alignment because [problem]."

**40. Stuck on a question — strategy?**
Re-state to confirm understanding. Decompose into smaller parts. Solve the simplest part. Build up. Don't fake it.

---

## Quick fire

**41.** *First step "design X"?* Clarify.
**42.** *First step "train X"?* Scale + objective.
**43.** *"Why work" framework?* First principles.
**44.** *"Debug" framework?* Data → model → eval → drift.
**45.** *Listing without judgment lesson?* Pick + justify.
**46.** *Strong end of answer?* Failure modes + mitigation.
**47.** *Synthesis bridge for cross-entropy?* LM, RLHF, KD, VAE.
**48.** *5-min answer structure?* Clarify, frame, design, eval, monitor.
**49.** *Most important conclusion sentence?* The strongest claim you'd defend.
**50.** *Pragmatism over fancy?* Always.

---

## Self-grading

If you can't answer 1-15, you can't structure cross-topic answers. If you can't answer 16-30, you'll get tripped up on synthesis questions. If you can't answer 31-45, frontier-lab interviews on real cases will go past you.

Aim for 35+/50 cold.
