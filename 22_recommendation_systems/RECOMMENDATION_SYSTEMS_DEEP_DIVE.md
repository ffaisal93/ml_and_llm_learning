# Recommendation Systems — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Recommenders are everywhere — search, social feeds, ads, e-commerce, content discovery. Big-tech ML interviews probe this hard because it's the canonical "design at scale" problem: millions of users, millions of items, sub-50ms latency, business metrics that matter.

---

## 1. The recommendation problem

Given a user $u$ and context $c$, return top-$k$ items from a catalog $\mathcal{I}$ that maximize a target metric.

**Targets** (depend on business):
- CTR (click-through rate)
- Conversion rate (purchase, signup)
- Watch / dwell time
- Long-term retention
- Revenue
- Diversity / engagement balance

**Inputs**:
- User: history, demographics, embeddings.
- Item: features, embeddings, metadata.
- Context: device, location, time, query.

**Output**: ranked list of items.

---

## 2. Classical approaches

### Collaborative filtering (CF)

Idea: users with similar tastes will like similar items.

**User-user CF**: similarity between users via shared item interactions; recommend items liked by similar users.

**Item-item CF**: similarity between items via shared user interactions; recommend items similar to ones the user liked. Amazon's classic algorithm.

Similarity: cosine, Pearson correlation, Jaccard.

### Matrix factorization

User-item interaction matrix $R \in \mathbb{R}^{m \times n}$ is low-rank in practice. Factor:

$$
R \approx U V^\top
$$

with $U \in \mathbb{R}^{m \times d}$ (user embeddings), $V \in \mathbb{R}^{n \times d}$ (item embeddings).

Loss for explicit ratings:

$$
\mathcal{L} = \sum_{(u, i) \in \Omega} (R_{ui} - u_u^\top v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)
$$

For implicit feedback (clicks): use weighted ALS, BPR, or pointwise BCE on observed clicks vs random negatives.

### Why it worked
- Simple and fast.
- Scales to large catalogs.
- Captures latent factors (taste dimensions) automatically.

### Limitations
- Cold start (new users / items have no interactions).
- Sparse (most user-item pairs unrated).
- Doesn't use side features.
- Static (doesn't model sequential interest).

---

## 3. Two-tower models

The dominant retrieval architecture. Two encoders:

$$
f_u(\text{user features, history}) \to u, \quad f_i(\text{item features}) \to v
$$

Score: $s(u, i) = u^\top v$. Trained with contrastive loss (positive pair = clicked items; negatives = random items in batch).

### Architecture
- Each tower: MLP / transformer / GNN over features.
- Embeddings of fixed dim ($d \sim 64$–$256$).
- Independence: query and item encoded separately → can pre-compute item embeddings.

### Inference flow
- Item embeddings precomputed offline; stored in ANN index (HNSW, IVF-PQ).
- At request time: encode user; ANN search for top-K; pass to ranker.

### Training
- Sampled softmax (large catalog → can't compute full softmax).
- Hard negative mining: include "almost-correct" negatives.
- In-batch negatives: items shown to other users in the same batch.
- Cross-batch negatives: large global negative pool.

---

## 4. Sequential models — modeling user history

Users' interests evolve. A user's recent clicks predict future clicks better than their first click.

### GRU4Rec (2015)
Pioneering. RNN over session of clicks predicts next item. Showed sequential matters.

### SASRec (Self-Attentive Sequential Recommendation, 2018)
Transformer decoder over sequence of user's interactions. Each position predicts next item. Better than GRU4Rec on long histories.

### BERT4Rec (2019)
BERT-style masked-item prediction on sequence. Bidirectional context. Strong on sequence-based benchmarks.

### Transformer-based recommenders
Modern systems: encode user history with a transformer; concatenate with item features; predict click probability. Used at Pinterest, Meta, Google.

### Why sequential helps
- Captures order: "user just looked at iPhone case" → recommend phone case accessories.
- Captures recency.
- Captures session intent vs long-term taste.

---

## 5. Two-stage: retrieval + ranking

Standard pattern at scale. Catalogs of 10M-1B items can't be ranked exhaustively per request.

### Stage 1: Retrieval
- Goal: high recall on top-1000 (or so) candidates from full catalog.
- Latency: ~10ms.
- Methods: ANN over two-tower embeddings; collaborative filtering; popularity; rules; hybrids.

### Stage 2: Ranking
- Goal: high precision on top-$K$ from candidates.
- Latency: ~30-50ms.
- Methods: GBDT (LightGBM, XGBoost) on engineered features; or DL (DeepFM, DLRM, transformer-based); use richer (cross-feature) interactions.

### Why two stages?
- Retrieval is fast but coarse. Ranker is slow but precise.
- Total compute: retrieval $\propto N$ items; ranking $\propto K$ candidates. Overall manageable.

### Sometimes three stages
Retrieval → coarse ranker → final ranker. Coarse ranker filters from 1000 to 100; final ranks 100. Used at Pinterest, YouTube.

---

## 6. Ranking models

### GBDT
- Strong baseline. Robust, fast, interpretable.
- Used as a pointwise classifier (predict CTR / conversion).
- LambdaMART for pairwise ranking objectives.

### DeepFM (2017)
Combines low-order interactions (factorization machine) and high-order (deep MLP). Standard production model for ads.

### DLRM (Meta 2019)
Categorical features → embeddings → element-wise dot products + concatenated dense features → MLP. Open-source benchmark for large-scale recommenders.

### Transformer-based rankers
Sequence of user interactions + candidate item → transformer → score. Increasing adoption.

### Loss functions
- **Pointwise** (BCE on click): simple; doesn't model relative ordering.
- **Pairwise** (BPR, RankNet): predict $i$ before $j$.
- **Listwise** (LambdaRank, ListNet): full list optimization. Closer to final metric.

Pairwise often wins for medium-data systems; listwise for high-quality ranked output evaluation.

---

## 7. Cold start

The hardest problem in recommenders.

### New user (no history)
- Popularity-based recommendations.
- Demographic-based ("users like you").
- Onboarding ("pick 3 interests").
- Active exploration (Thompson sampling, $\epsilon$-greedy).

### New item (no clicks)
- Content features (item embedding from text/image alone).
- Forced exposure period: explicitly insert into some sessions.
- Side features: category, brand, similar-to existing items.
- Use multimodal features: CLIP embedding of item image.

---

## 8. Exploration vs exploitation

If you only show items predicted to be best, you never learn about other items. Echo chamber forms.

### Solutions
- **$\epsilon$-greedy**: random item with prob $\epsilon$; otherwise predicted best.
- **Thompson sampling**: sample posterior over user-item preferences; act greedy under sample. Naturally balances exploration with confidence.
- **UCB**: optimistic estimates for less-tried items.
- **Diversity bonus**: add diversity term to ranking score.
- **Periodic forced exploration**: occasionally show items the model has low confidence about.

### Why hard at scale
Greedy maximization → echo chambers, popularity bias, filter bubbles. Exploration costs short-term metric for long-term diversity / discovery.

---

## 9. Evaluation

### Offline metrics
- **Hit@K**: did the actual clicked item appear in top-$K$? Simple recall measure.
- **NDCG@K**: position-discounted relevance score. Standard for ranked output.
- **MAP@K**: mean average precision.
- **MRR**: reciprocal rank of first hit.
- **AUC**: pairwise ranking metric.

### Online metrics (the actual ground truth)
- **CTR / conversion rate**: short-term.
- **Watch time / dwell time**: deeper engagement.
- **Retention / day-7 / day-30**: long-term.
- **Coverage**: fraction of catalog actually shown.
- **Diversity**: Gini of recommended items or per-user variety.

### Offline-online gap
Common problem: model wins offline, loses online. Causes:
- Position bias in offline labels.
- Counterfactual issue: offline data is from old policy.
- Selection bias: only see clicks on items shown.
- Long-term effects offline can't measure.

### A/B testing
The actual decision instrument. Usually 1-4 weeks per launch decision. (See A/B testing deep dive.)

---

## 10. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Use accuracy for click prediction? | Sure | Bad for imbalanced; use AUC or PR-AUC |
| Cold start mitigation? | Wait for clicks | Content features + popularity + exploration |
| Why two-stage? | Tradition | Retrieval is fast/coarse, ranker is slow/precise; overall compute manageable |
| What does NDCG measure? | "Quality" | Position-discounted relevance — early ranks weighted more |
| Echo chamber — fix? | Less greedy | Exploration: Thompson sampling, diversity bonus, forced exposure |
| Offline wins, online loses — why? | "Random" | Position bias, counterfactual issue, long-term effects, selection bias |
| Why GBDT for ranking? | Old-school | Strong on tabular features; fast; robust; often beats DL when feature engineering is good |

---

## 11. Eight most-asked interview questions

1. **Design a recommender for [domain]: walk through end-to-end.** (Two-stage; retrieval + ranking; cold start; serving; monitoring.)
2. **What's collaborative filtering and where does it fail?** (User-user / item-item / matrix factorization; cold start, side features.)
3. **Two-tower model — architecture and training?** (User and item encoders, contrastive loss, in-batch negatives, ANN at serve time.)
4. **Pointwise vs pairwise vs listwise ranking?** (Pointwise easiest; pairwise emphasizes ordering; listwise optimizes the metric directly.)
5. **How do you handle cold start?** (Content features for items; popularity / demographics for users; forced exploration.)
6. **Echo chamber problem — how do you fix it?** (Exploration: Thompson sampling, diversity bonus, forced exposure.)
7. **NDCG vs AUC?** (NDCG is position-discounted relevance for ranked output; AUC is pairwise threshold-free classification.)
8. **Why does offline often disagree with online?** (Position bias, counterfactual issue, off-policy evaluation problems.)

---

## 12. Drill plan

- Practice the 5-minute "design a recommender" answer for 3 domains: e-commerce, video streaming, ads.
- Recite two-tower training: positive pair, negative sources, contrastive loss form.
- Recite NDCG, MAP, MRR formulas.
- For each cold-start mitigation, recite when it applies + trade-off.
- Be able to name 2 industry models per category: retrieval (two-tower), ranking (DLRM / DeepFM / GBDT), sequential (SASRec / BERT4Rec).

---

## 13. Further reading

- Koren, Bell, Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems.*
- Rendle (2010), *Factorization Machines.*
- Hidasi et al. (2015), *Session-based Recommendations with Recurrent Neural Networks* (GRU4Rec).
- Kang & McAuley (2018), *Self-Attentive Sequential Recommendation* (SASRec).
- Naumov et al. (2019), *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (DLRM).
- YouTube paper (2016), *Deep Neural Networks for YouTube Recommendations.*
- Pinterest engineering blog: PinSage, PinnerSAGE, etc.
