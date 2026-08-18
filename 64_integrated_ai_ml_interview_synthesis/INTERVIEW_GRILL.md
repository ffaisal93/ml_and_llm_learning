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

> **Saying it out loud.** The reason to learn the five archetypes is that recognizing which one you're in decides the shape of your answer in the first ten seconds. "Design X" wants a structure walked out loud, ending in monitoring. "Train X" wants the pipeline in decision order, starting from the compute budget. "Why does X work" wants you to name a principle and build back up from it. "Debug X" wants hypotheses ordered by prior probability, boring causes first. "Trade off X versus Y" wants the curve and then a decision. Misclassify the archetype and you'll give a well-structured answer to a question nobody asked — which is a surprisingly common way to lose a round you technically knew the material for.

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

> **Saying it out loud.** Cross-topic fluency is really one skill: noticing that the same object is showing up in two places. Cross-entropy is the pretraining loss, the distillation objective, and the KL penalty in RLHF — same math, three costumes. Attention is the architecture answer, the long-context answer, and the serving-cost answer. Embeddings are retrieval, recommendation, and multimodal alignment. When you can say "this is the same thing as that, for the same reason," you sound like someone who has worked across the stack rather than someone who read seven separate chapters. And that's exactly what the multi-part frontier question is designed to detect, because each part deliberately lives in a different area.

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

> **Saying it out loud.** For any of these sprawling prompts, the move is to name the two or three topics that actually bind and then go deep on one, rather than touring all six at one inch of depth. "Cut latency in half" touches quantization, speculative decoding, caching, and batching — but I'd say up front that decode is memory-bandwidth-bound, so anything that reduces bytes read per token is the real lever and everything else is secondary. "Improve a metric without retraining" sounds like a trick until you realize threshold tuning and calibration are free and frequently worth more than a model upgrade. Naming the dominant lever first, then the supporting ones, is the difference between a synthesis answer and a list.

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

> **Saying it out loud.** The pattern in one breath: what are we optimizing, what's the binding constraint, which principle governs, what's the simplest baseline, how would I iterate, how could it fail, and what would I actually do. The two steps candidates skip are the baseline and the ending. Skipping the baseline makes you sound like you'd rather build the interesting thing than the shippable thing, which is a real hiring signal. And ending on "there are several good options" instead of "I'd do this, because" reads as an inability to commit under uncertainty — which is most of the job. Say the failure modes too; senior candidates volunteer them, junior candidates assume it works.

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

> **Saying it out loud.** All five of these are the same mistake: substituting coverage for judgment. Listing model families proves you read the survey; picking one and saying why proves you'd ship. Naming FlashAttention proves you follow the news; saying it takes attention memory from quadratic to linear by tiling, and that it matters here because we're at 32K context, proves you know when it applies. Over-engineering is the same disease pointed the other way — proposing a graph network where gradient boosting would do reads as inexperience, not sophistication. And the cheapest points on the board are closing every answer with a failure mode and a mitigation, which almost nobody does unprompted.

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

> **Saying it out loud.** These one-breath summaries are worth rehearsing verbatim, because interviewers frequently open with "walk me through RAG" and the first thirty seconds set their prior for the rest of the hour. Say it as a pipeline with the failure point marked: index and chunk the documents, embed the query, approximate-nearest-neighbor retrieve, optionally rerank, drop the results into a template, generate with citations — and the part that breaks is retrieval, not generation, which is why you measure recall at k separately. Same discipline for RLHF: SFT for format, reward model from preferences, then policy optimization with a KL penalty holding you near the reference, and reward hacking is the permanent threat.

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

> **Saying it out loud.** Under time pressure the structure is the answer. For a five-minute design question: one minute clarifying, one framing, ninety seconds on the architecture, one on evaluation, thirty seconds on monitoring — and say the budget out loud so they know you're managing it. For thirty seconds: "three things matter here, A, B, and C; A dominates because of this; I'd start with X and revisit if I see Y." And when you're genuinely stuck, the recovery is to restate the problem, decompose it, and solve the smallest piece out loud — never to bluff. Interviewers score the honest boundary far above a confident wrong answer, and they can almost always tell the difference.

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
