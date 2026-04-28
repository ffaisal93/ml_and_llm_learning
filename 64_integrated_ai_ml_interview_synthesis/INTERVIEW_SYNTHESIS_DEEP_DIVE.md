# Integrated Interview Synthesis — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This is a meta-document. The repo's 50+ deep dives cover individual topics; this one is about the *cross-topic synthesis* that separates senior candidates from junior ones. Frontier interviews increasingly ask questions that span multiple areas — "design an LLM system that handles X" requires synthesizing inference, alignment, evaluation, A/B testing, system design, and prompting fluency in one answer.

---

## 1. The five archetype questions

Most cross-topic interview questions reduce to one of five archetypes. Knowing which you're being asked sets the answer's structure.

### A. "Design X" (system design)

"Design Spotify recommendations." "Build a fraud detector for credit cards." "Design a chatbot."

Use the 6-step framework from `29_system_design_for_ml/`: clarify → frame → data → features+model → serving → monitoring.

### B. "Train X" (training methodology)

"Train a 70B model." "Improve our model's math abilities." "Build a code assistant."

Use frontier training playbook: scaling laws → architecture → data mixture → training stages → evaluation. See `62_frontier_training_playbook/`.

### C. "Why does X work?" (research / theory)

"Why does scale help?" "Why does Adam beat SGD here?" "Why is RLHF important?"

Use mechanistic answer: reduce to first principles. Often involves bias-variance, optimization geometry, capacity arguments, scaling laws.

### D. "Debug X" (debugging methodology)

"Loss is spiking — what's wrong?" "Model regressed in production — investigate." "Eval looks fine but users complain — what's the issue?"

Use debugging tree: data → model → evaluation → distribution shift → cost asymmetry. Common failures by frequency, not exotic-ness.

### E. "Trade off X vs Y" (judgment)

"Bigger model vs more data." "Online vs offline training." "Latency vs accuracy."

Use trade-off framework: list axes; identify which the business cares about; explain the curve.

---

## 2. Cross-topic synthesis questions

Questions that show up at frontier labs:

### "Build an LLM-powered customer support agent"
- **Frame**: agent (LLM + tools); chat interface; multi-turn.
- **Components**: system prompt, RAG over docs, tool use (account lookup, ticket creation), safety filtering.
- **Evaluation**: automated (faithfulness, refusal correctness) + human review.
- **Deployment**: latency budget; failover; monitoring.
- **Cross-topic**: prompting (`07`), RAG (`39`), inference (`06`), agents, A/B testing (`30`), evaluation (`49`).

### "Improve our ranker by 1% NDCG"
- **Frame**: ranking optimization; specific metric.
- **Levers**: features (richer signals), model (DLRM upgrade, transformer ranker), training (more data, hard negatives), loss (listwise vs pairwise).
- **Validation**: A/B test for online lift; offline NDCG ablation.
- **Cross-topic**: recommendations (`22`), evaluation (`49`), A/B testing (`30`), ranking (in case studies `28`).

### "Reduce hallucination in our Q&A system"
- **Frame**: factual accuracy + faithfulness.
- **Levers**: stronger RAG, calibration, refusal training, tool use, self-consistency.
- **Evaluation**: faithfulness vs source, factual accuracy on held-out, refusal rate.
- **Cross-topic**: LLM problems (`07`), alignment (`08`), RAG (`39`), evaluation (`49`).

### "Train a model to do X better"
- **Frame**: capability gap; targeted improvement.
- **Levers**: SFT data, RLHF reward, mid-training, prompting.
- **Validation**: capability-specific eval; broader regression check.
- **Cross-topic**: training playbook (`62`), alignment (`08`), data curation, evaluation (`49`).

### "Why is our offline metric not matching online?"
- **Frame**: distribution shift; counterfactual; selection bias.
- **Causes**: position bias (search/rec), counterfactual issue (offline data from old policy), long-term effects, novelty effects.
- **Cross-topic**: A/B testing (`30`), evaluation (`49`), recommendations (`22`).

---

## 3. The "first principles" answer pattern

Strong synthesis answers follow a pattern:

1. **State the goal clearly**: what are we optimizing? What's the user-facing outcome?
2. **Identify the constraint(s)**: latency, cost, data, scale.
3. **Apply the dominant principle**: scaling laws, bias-variance, cost asymmetry — whatever's most relevant.
4. **Recommend a baseline**: what's the simplest thing that could work?
5. **Iterate up**: how would you improve from there, in priority order?
6. **Discuss what could fail**: 2-3 failure modes; mitigation for each.
7. **State the strongest conclusion you'd defend**: not "all of the above" but "I'd start here, because..."

This earns more points than a comprehensive list. Interviewers value *judgment* over *coverage*.

---

## 4. Topics that bridge multiple areas

Some topics show up everywhere. Owning these unlocks cross-topic answers:

### Cross-entropy / KL divergence
- Pre-training loss (LM losses, `43`).
- RLHF objective (alignment, `08`).
- Knowledge distillation (LoRA, `25`).
- Variational inference (information theory, `33`).
- Clustering (Bregman divergences, `19`).

### Bias-variance trade-off
- Classical ML (advanced theory, `27`).
- Deep learning generalization (SLT, `52`).
- Estimator design (statistical inference, `47`).
- Regularization choice (regularization, `11`).

### Embeddings
- Retrieval (RAG, `39`).
- Recommendations (two-tower, `22`).
- Multimodal (CLIP, `38`).
- Tokenization (`15`).
- Search ranking (BM25 + dense, `36`).

### Attention
- Transformer architecture (`04`, `05`).
- Long context (LLM problems, `07`).
- KV cache + serving (paged attention, `63`).
- Position encoding (`14`).

### Data curation
- Pre-training (frontier training, `62`).
- SFT / RLHF (`08`).
- RAG corpora (`39`).
- Anomaly detection (`32`).

When you can connect these threads in your answer, you sound like someone who has worked across the stack.

---

## 5. Common failure modes in interviews

### Listing without judgment
"You could use logistic regression, decision trees, random forests, GBDT, or neural networks."

This is just a list. Interviewer learns nothing. Better: "I'd default to GBDT here because [reason]."

### Naming without explaining
"I'd use FlashAttention."

OK — but why? When? What does it solve? Strong answer: "FlashAttention reduces attention memory from $O(L^2)$ to $O(L)$ via tiled IO-aware computation. It matters here because we have 32K context and our current attention dominates memory."

### Over-engineering
"I'd build a multi-tower transformer with cross-attention and use a graph neural network for the user side."

Interviewer wants pragmatism. Start simple; add complexity only when needed.

### Missing failure modes
Strong answers always close with: "this could fail when... Here's how I'd mitigate."

### Forgetting the business
ML answers without business context show you don't understand product. Always tie back to the user-facing outcome.

---

## 6. Synthesis cheatsheet by topic

### Logistic regression / classification
- Discriminative model.
- MLE = cross-entropy.
- Linear decision boundary.
- For tabular: GBDT often beats DL.
- Calibration matters for cost-weighted decisions.

### Optimization
- SGD with momentum is robust default.
- Adam handles bad conditioning approximately.
- LR is the most important hyperparameter.
- Warmup + cosine decay is standard.

### Generalization
- Classical: bias-variance.
- Modern: double descent.
- Implicit regularization of SGD finds flat minima.
- Real-world generalization needs distribution-shift defense.

### Inference
- KV cache critical.
- Quantization for memory.
- Speculative decoding for throughput.
- PagedAttention + continuous batching for serving.

### Alignment
- SFT for format.
- RLHF for capability + preference.
- DPO simpler alternative.
- Reward hacking is the eternal threat.

### Evaluation
- Online ground truth via A/B test.
- Offline can mislead (position bias, counterfactuals).
- Calibration matters separately from accuracy.
- Multiple metrics + uncertainty.

### Systems
- 6-step design framework.
- Two-stage retrieval + ranking.
- Latency budget = where time actually goes.
- Always have a fallback.

---

## 7. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| What's the right answer? | Listing options | Pick one, justify, mention alternative for context |
| How would you improve X? | "Try harder model" | Identify bottleneck first; then targeted improvement |
| What if X fails? | "It won't" | Always have failure mode + mitigation |
| Tradeoff between X and Y? | Pick one | Identify what business prioritizes; explain the curve |
| Why is this hard? | "It's complex" | Specific reason: bias-variance, cost asymmetry, distribution shift |
| What's the metric? | "Accuracy" | Business metric → ML proxy → offline → calibration |
| Latency / cost / accuracy — pick? | Pick one | Two of three usually; depends on use case |

---

## 8. Eight cross-topic questions you should be ready for

1. **Design an LLM-powered Q&A system.** (RAG + prompting + agent + safety + evaluation + serving.)
2. **Why does scale work?** (Scaling laws + bias-variance + over-parameterization + implicit reg.)
3. **What's the most important thing in training large models?** (Methodology + data > architecture; ablation rigor.)
4. **Reduce model latency to half.** (Quantization + smaller model + caching + batching + speculative decoding.)
5. **Improve a metric without retraining.** (Inference tricks: prompting, sampling, post-processing, calibration, retrieval.)
6. **You see a regression in production. Walk me through your investigation.** (Data + model + infra + drift + evaluation; rollback + diagnose.)
7. **Online and offline metrics disagree. What do you check?** (Position bias, counterfactual, novelty, distribution shift, label time leakage.)
8. **What's the next frontier in LLMs?** (Reasoning, agents, multimodal, efficiency, alignment robustness — name 2-3 with substance.)

---

## 9. Drill plan

- Practice 5-minute answers for the eight cross-topic questions above.
- For each, identify which 4-5 deep dives in the repo are relevant.
- Pick one deep dive per day; spend 15 min on the grill questions; identify cross-references to other topics.
- Time yourself: aim for 30-45 min total prep per major mock interview.

---

## 10. Further reading

This deep dive is a meta-document. The "further reading" is the rest of the repo:

- For first-principles: SLT (`52`), info theory (`33`), optimization (`48`).
- For systems: ML system design (`29`), large-scale LLM (`61`), paged attention (`63`).
- For methodology: frontier training (`62`), generalization (`49`), A/B testing (`30`).
- For practice: mock interviews (`57`), blind drills (`59`), case studies (`28`).

If you can't connect across these in an interview, drill the bridges (cross-entropy, embeddings, attention, bias-variance, data curation) until they're second nature.
