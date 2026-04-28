# Recommendation Systems — Interview Grill

> 45 questions on collab filtering, matrix factorization, two-tower, sequential models, ranking, evaluation. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What does a recommender output?**
Ranked list of top-$k$ items for a (user, context) input.

**2. Common business targets?**
CTR, conversion, dwell time, retention, revenue. Pick one or weighted combination.

**3. User-user vs item-item CF?**
User-user: similarity between users via shared items. Item-item: similarity between items via shared users. Item-item is Amazon's classic.

**4. Why is CF data sparse?**
Most user-item pairs unobserved (a typical user interacts with $\ll 1\%$ of catalog).

**5. What's the cold-start problem?**
New users / new items have no interaction data. CF can't help; need content features.

---

## B. Matrix factorization

**6. Matrix factorization core idea?**
$R \approx U V^\top$. Low-rank approximation of user-item interaction matrix.

**7. Loss for explicit ratings?**
$\sum_{(u,i)} (R_{ui} - u_u^\top v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$.

**8. How do you handle implicit feedback?**
Weighted ALS, BPR (pairwise ranking), or pointwise BCE on observed clicks vs sampled negatives.

**9. What's BPR loss?**
Bayesian Personalized Ranking: $-\log \sigma(s(u, i^+) - s(u, i^-))$. Pairwise; encourages observed > unobserved.

**10. Strengths of MF?**
Simple, fast, scales, captures latent factors automatically.

**11. Weaknesses?**
Cold start, no side features, static (no sequence).

---

## C. Two-tower retrieval

**12. Two-tower architecture?**
User encoder $f_u$ and item encoder $f_i$ → fixed-dim embeddings. Score = $u^\top v$.

**13. Why do encoders need to be independent?**
To pre-compute item embeddings offline and serve via ANN. If user-item features mixed, every (user, item) pair would need explicit forward pass.

**14. Training objective?**
Contrastive: positive pair (clicked item) → high score; negatives → low. In-batch negatives standard.

**15. Hard negative mining?**
Include "almost-positive" negatives (e.g., items shown but not clicked). Better than random negatives at training discrimination.

**16. Sampled softmax — why?**
Full softmax over millions of items intractable. Sample $K$ negatives from sampling distribution $Q$, then subtract $\log Q(j)$ from each sampled logit (the **logQ correction**) before applying softmax over sample. Yields an unbiased estimator of the full softmax gradient.

**17. Inference flow?**
Embed user → ANN search over precomputed item embeddings → return top-$K$.

**18. Why ANN, not exact?**
Exact KNN over millions is too slow. ANN (HNSW, IVF-PQ) trades small recall for huge speedup.

---

## D. Sequential models

**19. Why sequential models?**
User interests evolve. Recent clicks predict next click better than oldest clicks. Order matters.

**20. GRU4Rec — what's the architecture?**
RNN over session of clicks. At each step, predict next item.

**21. SASRec — improvement?**
Transformer self-attention instead of RNN. Better for long sequences.

**22. BERT4Rec — twist?**
Masked-item prediction (BERT-style) on the sequence. Bidirectional context.

**23. Modern industry sequential setup?**
Transformer over user history; produce user representation; combine with item features for ranking.

---

## E. Two-stage retrieval + ranking

**24. Why two-stage?**
Catalog ($N$) too large to rank exhaustively. Retrieval narrows to $K \ll N$. Then expensive ranking on $K$.

**25. Latency budgets?**
Retrieval ~10ms. Ranking ~30-50ms. Total ~50ms.

**26. Stage 1 methods?**
Two-tower ANN, item-item CF, popularity, rules. Often hybridized.

**27. Stage 2 methods?**
GBDT, DeepFM, DLRM, transformer-based rankers.

**28. When three stages?**
Pinterest, YouTube: retrieval (1M → 1000) → coarse ranking (1000 → 100) → final (100 → top-$K$).

---

## F. Ranking models

**29. Why does GBDT often win for ranking?**
Tabular interaction features; robust; fast; interpretable; handles missing data well.

**30. DeepFM — what's the idea?**
Factorization Machines (low-order interactions) + Deep MLP (high-order). Combined.

**31. DLRM architecture?**
Categorical embeddings → element-wise dot products + dense features → MLP.

**32. LambdaMART — what is it?**
GBDT trained with pairwise/listwise ranking objectives (LambdaRank). Produces ranking-aware models.

**33. Pointwise vs pairwise vs listwise?**
Pointwise: predict score per item. Pairwise: predict $i > j$. Listwise: optimize full list metric (e.g., NDCG).

---

## G. Evaluation

**34. NDCG@K formula intuition?**
$\mathrm{DCG@K} = \sum_{i=1}^K (2^{\mathrm{rel}_i} - 1) / \log_2(i+1)$ (graded relevance form; for binary relevance simplifies to $\sum \mathrm{rel}_i / \log_2(i+1)$). Position-discounted relevance with $i$ 1-indexed (rank). Normalized by ideal ordering: $\mathrm{NDCG} = \mathrm{DCG}/\mathrm{IDCG}$. Higher = better.

**35. MAP@K?**
Mean Average Precision. Average precision at each correct hit position.

**36. MRR?**
Mean Reciprocal Rank: $1/k_1$ where $k_1$ is position of first hit. Captures "did we rank a correct item near top?"

**37. Hit@K?**
1 if any correct item in top-$K$, else 0. Simple recall measure.

**38. AUC for ranking?**
Pairwise: probability that positive item ranks above negative. Threshold-free.

**39. Why does offline often disagree with online?**
Position bias (top items click more), counterfactual (offline data from old policy), selection bias (only see clicks on shown), long-term effects.

**40. Holdback test for recommenders?**
Permanent control population on old model. Catches long-term drift / degradation that A/B doesn't.

---

## H. Cold start and exploration

**41. New user — what do you do?**
Popularity, demographics, onboarding survey, exploration.

**42. New item — what do you do?**
Content features (CLIP for images, text encoder for descriptions), forced exposure, similar-to-existing.

**43. Echo chamber problem?**
Greedy ranking reinforces popular items; users see less diversity over time. Filter bubbles.

**44. Thompson sampling for exploration?**
Maintain posterior over each item's reward. Sample, act greedily under the sample. Natural exploration-exploitation balance.

**45. Diversity bonus?**
Add penalty for items similar to ones already in the recommended list. Encourages variety.

---

## Quick fire

**46.** *MF dimension typical?* 64-256.
**47.** *Two-tower scoring?* Dot product.
**48.** *In-batch negatives?* Items from other queries in same batch.
**49.** *NDCG decay?* $\log_2(i+1)$.
**50.** *Echo chamber fix?* Exploration + diversity.
**51.** *DLRM bottom layers?* Embedding tables.
**52.** *Cold-start tools?* Content features + popularity.
**53.** *GBDT for ranking strength?* Tabular interactions.
**54.** *Stage 1 latency?* ~10ms.
**55.** *Stage 2 latency?* ~30-50ms.

---

## Self-grading

If you can't answer 1-15, you can't talk about recommenders. If you can't answer 16-30, you'll struggle on architecture and ranking questions. If you can't answer 31-45, big-tech recommender system design will go past you.

Aim for 35+/55 cold.
