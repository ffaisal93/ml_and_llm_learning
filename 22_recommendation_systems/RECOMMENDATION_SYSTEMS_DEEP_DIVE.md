# Recommendation Systems — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

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

> **Saying it out loud.** Strip it down and a recommender is a ranking function: user plus context in, ordered list of items out. The part people underestimate is picking the target. Click-through rate is easy to measure and easy to game, watch time pushes you toward autoplay traps, and what the business actually wants is retention, which you can't observe for thirty days. So every production system trains on short-term proxies and validates on long-term ones, and the gap between those two is where most of the interesting arguments happen.

---

## 2. Classical approaches

### Collaborative filtering (CF)

Idea: users with similar tastes will like similar items.

**User-user CF**: similarity between users via shared item interactions; recommend items liked by similar users.

**Item-item CF**: similarity between items via shared user interactions; recommend items similar to ones the user liked. Amazon's classic algorithm.

Similarity: cosine, Pearson correlation, Jaccard.

> **Saying it out loud.** Collaborative filtering rests on one sentence: people who agreed in the past will agree again. You can run it in two directions. User-user finds people with similar histories and recommends what they liked, item-item finds items that tend to be liked by the same people and recommends neighbors of what you just touched. Item-item is what Amazon shipped and what most people still use, because item-item similarities are stable enough to compute in a nightly batch job while user tastes shift constantly.

### Matrix factorization

> **In plain language.** This section is the classic trick that made recommenders work. Think of a giant grid, users down the side, items across the top, mostly empty. Matrix factorization says that grid is secretly the product of two much skinnier tables, one describing each user in maybe a hundred numbers and one describing each item the same way. Learn those two tables and you can fill in every empty cell with a dot product.

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

> **Saying it out loud.** The loss is squared error between the observed rating and the dot product of the user and item vectors, plus L2 on both embedding tables. The one detail that matters is that the sum runs only over cells you actually observed, because an unrated item is unknown, not a zero. Regularization is essential here since a user with three ratings and a hundred-dimensional vector would otherwise memorize them. For implicit data, where you only ever see positives, you switch to weighted ALS, BPR, or pointwise cross-entropy against sampled negatives.

### Why it worked
- Simple and fast.
- Scales to large catalogs.
- Captures latent factors (taste dimensions) automatically.

> **Saying it out loud.** Matrix factorization won because it was simple, it scaled, and it found taste dimensions nobody had to hand-design. The Netflix Prize made it famous, and the striking thing was that the learned latent factors were sometimes readable, one axis roughly tracking action versus drama, another tracking mainstream versus niche. Serving is a dot product, which is essentially free. It's still the baseline every fancier model has to beat, and plenty of them don't.

### Limitations
- Cold start (new users / items have no interactions).
- Sparse (most user-item pairs unrated).
- Doesn't use side features.
- Static (doesn't model sequential interest).

> **Saying it out loud.** Four limitations, and each one motivates a later architecture. No cold start handling, because a new user or item has no row to look up. Extreme sparsity, since users touch a vanishing fraction of the catalog. No way to use side features like category or price or image. And no notion of order, so the thing you clicked a minute ago counts the same as the thing you clicked last year. Two-tower models fix the first three by replacing lookups with encoders; sequential models fix the fourth.

---

## 3. Two-tower models

> **In plain language.** A two-tower model is matrix factorization where the two embedding tables are replaced by neural networks. One network reads the user and their history and produces a vector; a completely separate network reads an item's features and produces a vector; the score is their dot product. Keeping the two networks apart is the whole design, because it lets you compute every item's vector ahead of time.

The dominant retrieval architecture. Two encoders:

$$
f_u(\text{user features, history}) \to u, \quad f_i(\text{item features}) \to v
$$

Score: $s(u, i) = u^\top v$. Trained with contrastive loss (positive pair = clicked items; negatives = random items in batch).

> **Saying it out loud.** The two-tower model generalizes matrix factorization by replacing lookup tables with encoders. The user tower eats history and context and emits a vector; the item tower eats item features and emits a vector; the score is the dot product. Because items are encoded from features rather than IDs, a brand-new item still gets a sensible embedding, which is how this fixes cold start. Training is contrastive, positives being real engagements and negatives being other items in the batch.

### Architecture
- Each tower: MLP / transformer / GNN over features.
- Embeddings of fixed dim ($d \sim 64$–$256$).
- Independence: query and item encoded separately → can pre-compute item embeddings.

> **Saying it out loud.** Each tower can be whatever fits the features, an MLP over tabular inputs, a transformer over a sequence, a graph network over a co-engagement graph. Embeddings usually land somewhere between sixty-four and two hundred fifty-six dimensions. The one non-negotiable is independence: the towers must not see each other's inputs until the dot product at the very end. Break that and you can't precompute item embeddings, and without precomputation the whole retrieval strategy collapses.

### Inference flow
- Item embeddings precomputed offline; stored in ANN index (HNSW, IVF-PQ).
- At request time: encode user; ANN search for top-K; pass to ranker.

> **Saying it out loud.** Serving is two steps. Offline, a batch job runs the item tower over the whole catalog and loads the vectors into an ANN index like HNSW or IVF-PQ. Online, you run the user tower once on the live request and do a vector search for the top few hundred, which takes a couple of milliseconds. So one neural forward pass per request regardless of catalog size. The operational cost is index staleness, since a new item isn't retrievable until the next build, which is why fresh-content platforms run incremental updates.

### Training
- Sampled softmax (large catalog → can't compute full softmax).
- Hard negative mining: include "almost-correct" negatives.
- In-batch negatives: items shown to other users in the same batch.
- Cross-batch negatives: large global negative pool.

> **Saying it out loud.** Training is all about the negatives. You can't do a full softmax over ten million items, so you use sampled softmax with a logQ correction, subtracting the log sampling probability so the gradient stays unbiased. In-batch negatives give you hundreds of negatives for free by reusing other users' positives. Hard negatives, items that were shown and not clicked, are what actually sharpen the decision boundary. The failure mode is over-mining hard negatives, because some of them are false negatives, items the user would have loved and simply never saw.

---

## 4. Sequential models — modeling user history

Users' interests evolve. A user's recent clicks predict future clicks better than their first click.

### GRU4Rec (2015)
Pioneering. RNN over session of clicks predicts next item. Showed sequential matters.

> **Saying it out loud.** GRU4Rec was the paper that showed session order carries real signal. It runs a recurrent net over the clicks in a session and predicts the next item at every step. The commercially important insight was that it needs no user ID at all, just the current session, which matters because a huge share of e-commerce traffic is logged out. The limitation is the classic RNN one: strictly sequential training and a fading memory of anything far back.

### SASRec (Self-Attentive Sequential Recommendation, 2018)
Transformer decoder over sequence of user's interactions. Each position predicts next item. Better than GRU4Rec on long histories.

> **Saying it out loud.** SASRec replaces the recurrence with causal self-attention, so every position can look straight back at any earlier item instead of relaying information through a chain. Training parallelizes across the sequence and long-range dependencies survive. It's essentially a small GPT over item IDs. The cost is quadratic attention, which is why production versions truncate history to the last fifty or hundred interactions.

### BERT4Rec (2019)
BERT-style masked-item prediction on sequence. Bidirectional context. Strong on sequence-based benchmarks.

> **Saying it out loud.** BERT4Rec trains bidirectionally, masking random items in the sequence and predicting them from both sides. That gives richer training signal and it beats SASRec on most offline benchmarks. The awkward part is the train-serve mismatch, since at prediction time you only have the past, so you tack a mask token on the end. Whether that offline win survives online is genuinely disputed, and noticing that is a good sign in an interview.

### Transformer-based recommenders
Modern systems: encode user history with a transformer; concatenate with item features; predict click probability. Used at Pinterest, Meta, Google.

> **Saying it out loud.** In real systems the sequence model is a feature extractor, not the ranker. You run a transformer over the user's recent interactions to produce a user representation, then feed that representation into the retrieval tower and the ranker alongside everything else. Pinterest, Meta, and Google all do some version of this. The engineering constraint is latency, so the sequence embedding is usually computed asynchronously and cached, refreshed on a delay rather than on every request.

### Why sequential helps
- Captures order: "user just looked at iPhone case" → recommend phone case accessories.
- Captures recency.
- Captures session intent vs long-term taste.

> **Saying it out loud.** Sequence helps for three reasons that are worth separating. Order carries meaning: viewing a phone and then a case implies something different from the reverse. Recency dominates, since the last five minutes predict the next click far better than last year does. And it lets you separate session intent, what you're shopping for right now, from long-term taste, what you generally like. A bag-of-history model conflates all three, which is why sequential models were such a jump on e-commerce data.

---

## 5. Two-stage: retrieval + ranking

Standard pattern at scale. Catalogs of 10M-1B items can't be ranked exhaustively per request.

### Stage 1: Retrieval
- Goal: high recall on top-1000 (or so) candidates from full catalog.
- Latency: ~10ms.
- Methods: ANN over two-tower embeddings; collaborative filtering; popularity; rules; hybrids.

> **Saying it out loud.** Retrieval's job is recall, not precision. From a catalog of ten million you need to hand roughly a thousand candidates to the ranker in about ten milliseconds, and all that matters is that the genuinely good items are somewhere in that thousand. You get there with ANN over two-tower embeddings, plus item-item collaborative filtering, plus popularity, plus rules, all unioned together. Anything retrieval misses is gone permanently, so retrieval recall is a hard ceiling on the whole system's quality.

### Stage 2: Ranking
- Goal: high precision on top-$K$ from candidates.
- Latency: ~30-50ms.
- Methods: GBDT (LightGBM, XGBoost) on engineered features; or DL (DeepFM, DLRM, transformer-based); use richer (cross-feature) interactions.

> **Saying it out loud.** Ranking's job is precision at the very top, and because it only sees a few hundred candidates it can afford real computation. Gradient boosted trees on engineered features remain extremely competitive, and deep models like DeepFM and DLRM take over when you have high-cardinality categoricals. The defining difference from retrieval is that the ranker sees full user-item cross features, the exact thing the two-tower architecture forbade. That's why the two stages aren't redundant: they're allowed to model different things.

### Why two stages?
- Retrieval is fast but coarse. Ranker is slow but precise.
- Total compute: retrieval $\propto N$ items; ranking $\propto K$ candidates. Overall manageable.

> **Saying it out loud.** The arithmetic is the argument. A cheap dot product over ten million items is affordable; a deep model over ten million items is not, by about four orders of magnitude. So you spend a tiny amount of compute per item across the whole catalog, then a large amount per item on a few hundred survivors, and total cost stays flat. Retrieval is fast and coarse, ranking is slow and precise. The structural weakness is that the two stages are trained separately and can disagree, so the ranker sometimes gets candidates it considers worthless.

### Sometimes three stages
Retrieval → coarse ranker → final ranker. Coarse ranker filters from 1000 to 100; final ranks 100. Used at Pinterest, YouTube.

> **Saying it out loud.** You add a third stage when one hop from a million candidates to a hundred is too big for a single model. YouTube and Pinterest go a million to a thousand with retrieval, a thousand to a hundred with a lightweight ranker, then a hundred to the final list with the expensive one. Each stage costs more per item and sees fewer items, so total compute stays roughly constant. The price is another model to train, monitor, and keep aligned with its neighbors.

---

## 6. Ranking models

### GBDT
- Strong baseline. Robust, fast, interpretable.
- Used as a pointwise classifier (predict CTR / conversion).
- LambdaMART for pairwise ranking objectives.

> **Saying it out loud.** Boosted trees are still the default ranker because ranking features are tabular and trees are the best thing we have for tabular data. They find interactions automatically, tolerate missing values, don't care about feature scaling, and train in minutes so you can iterate all day. Usually you run them pointwise, predicting a click probability, and switch to LambdaMART when you want a genuine ranking objective. The underrated advantage is feature importance, so when the model behaves stupidly you can actually find out why.

### DeepFM (2017)
Combines low-order interactions (factorization machine) and high-order (deep MLP). Standard production model for ads.

> **Saying it out loud.** DeepFM runs a factorization machine and a deep MLP on the same embeddings and adds their outputs. The FM half handles every pairwise feature interaction explicitly; the MLP half picks up higher-order ones. The reason to bolt on the FM is that MLPs are surprisingly bad at learning simple products of features from scratch, so you build them in rather than hope. Compared to Wide and Deep, the win is that you don't have to hand-engineer the cross features.

### DLRM (Meta 2019)
Categorical features → embeddings → element-wise dot products + concatenated dense features → MLP. Open-source benchmark for large-scale recommenders.

> **Saying it out loud.** DLRM is Meta's production click model. Sparse categorical features go through embedding tables, dense features through a bottom MLP, then you take explicit pairwise dot products among all those vectors and feed the result to a top MLP. That explicit interaction layer is the architectural statement: multiplicative interactions get built in, not learned. The dominant fact about DLRM in practice is that the embedding tables run to hundreds of gigabytes, so the whole system is designed around sharding them across machines, with communication rather than FLOPs as the bottleneck.

### Transformer-based rankers
Sequence of user interactions + candidate item → transformer → score. Increasing adoption.

> **Saying it out loud.** Transformer rankers take the user's interaction sequence plus the candidate item and score them jointly, so the model can attend from the candidate back into history. That's strictly more expressive than a two-tower dot product because it allows early interaction, which is affordable precisely because ranking only sees a few hundred candidates. Adoption is climbing but it isn't universal, because the latency cost is real and the gain over a well-featured GBDT is often small. The version that wins is usually attending over the candidate list too, so the score depends on what else is being shown.

### Loss functions
- **Pointwise** (BCE on click): simple; doesn't model relative ordering.
- **Pairwise** (BPR, RankNet): predict $i$ before $j$.
- **Listwise** (LambdaRank, ListNet): full list optimization. Closer to final metric.

Pairwise often wins for medium-data systems; listwise for high-quality ranked output evaluation.

> **Saying it out loud.** Pointwise, pairwise, listwise is a ladder from easiest to optimize to closest to the metric you care about. Pointwise predicts a click probability per item independently, which ignores that ranking is comparative but gives you a calibrated number, and calibration genuinely matters if an ad auction consumes the score. Pairwise learns which of two items goes first, which matches the ordering task. Listwise optimizes a list metric like NDCG directly and is the hardest to train because the metric isn't differentiable. Industry runs pointwise more than the literature suggests, and calibration is the reason.

---

## 7. Cold start

The hardest problem in recommenders.

### New user (no history)
- Popularity-based recommendations.
- Demographic-based ("users like you").
- Onboarding ("pick 3 interests").
- Active exploration (Thompson sampling, $\epsilon$-greedy).

> **Saying it out loud.** For a brand-new user you stack fallbacks by how much you know. Popularity within whatever coarse segment you can infer from country, device, and referrer. An onboarding step where they pick a few interests, which buys enormous signal for thirty seconds of user effort. Then explore hard for the first few sessions, since early interactions are worth far more than later ones. The framing worth saying: personalization should ramp within the first session, not over weeks, because the first three clicks are the strongest signal you will ever get.

### New item (no clicks)
- Content features (item embedding from text/image alone).
- Forced exposure period: explicitly insert into some sessions.
- Side features: category, brand, similar-to existing items.
- Use multimodal features: CLIP embedding of item image.

> **Saying it out loud.** For a new item you go all-in on content. Encode the image with something like CLIP, encode the title and description with a text model, add category and brand and price, and drop the item into embedding space beside things it resembles. Then force some exposure, deliberately showing it to a slice of traffic to buy interaction data, because otherwise it never earns impressions and never gets data. That forced exposure is an explicit exploration budget paid in short-term engagement, and on a marketplace it isn't optional, because sellers whose listings get zero impressions leave.

---

## 8. Exploration vs exploitation

If you only show items predicted to be best, you never learn about other items. Echo chamber forms.

### Solutions
- **$\epsilon$-greedy**: random item with prob $\epsilon$; otherwise predicted best.
- **Thompson sampling**: sample posterior over user-item preferences; act greedy under sample. Naturally balances exploration with confidence.
- **UCB**: optimistic estimates for less-tried items.
- **Diversity bonus**: add diversity term to ranking score.
- **Periodic forced exploration**: occasionally show items the model has low confidence about.

> **Saying it out loud.** The exploration menu runs from crude to principled. Epsilon-greedy shows a random item some fraction of the time, which is simple and wasteful because it explores things you already know are bad. UCB is optimistic about items it has seen rarely. Thompson sampling keeps a posterior per item and samples from it, which explores in proportion to genuine uncertainty and needs no rate to tune. On top of those, a diversity bonus and periodic forced exposure for low-confidence items. Thompson sampling is usually the best answer to give, because the exploration decays automatically as evidence accumulates.

### Why hard at scale
Greedy maximization → echo chambers, popularity bias, filter bubbles. Exploration costs short-term metric for long-term diversity / discovery.

> **Saying it out loud.** It's hard at scale because greedy maximization is a feedback loop, not a static bias. You show what the model predicts is best, the user engages, that becomes training data, and the model gets more confident about a shrinking set of items. Popularity bias and filter bubbles are the visible symptoms. You can't fix it by cleaning the data because the data is generated by the policy. And exploration is genuinely costly: every exploratory impression is one you knowingly expect to underperform, which is why it's a percentage of traffic that someone has to defend in a metrics review.

---

## 9. Evaluation

### Offline metrics
- **Hit@K**: did the actual clicked item appear in top-$K$? Simple recall measure.
- **NDCG@K**: position-discounted relevance score. Standard for ranked output.
- **MAP@K**: mean average precision.
- **MRR**: reciprocal rank of first hit.
- **AUC**: pairwise ranking metric.

> **Saying it out loud.** Offline metrics differ mainly in what they choose to ignore. Hit rate at K just asks whether anything relevant made the cut, which is the right check for retrieval. NDCG discounts by position and handles graded relevance, which makes it the default for ranked output. MAP rewards finding all the relevant items, MRR only cares about the first one. AUC measures the global ordering and is threshold-free, which is also its weakness: it weights position five thousand as much as position two, so AUC can improve while NDCG at ten gets worse.

### Online metrics (the actual ground truth)
- **CTR / conversion rate**: short-term.
- **Watch time / dwell time**: deeper engagement.
- **Retention / day-7 / day-30**: long-term.
- **Coverage**: fraction of catalog actually shown.
- **Diversity**: Gini of recommended items or per-user variety.

> **Saying it out loud.** Online metrics are the ones that decide launches, and they're layered by time horizon. Click-through and conversion are immediate and noisy and easy to game. Watch time and dwell are deeper but push toward length over quality. Retention at day seven and day thirty is what actually matters and takes a month to read. Then coverage and diversity as guardrails, catching the case where your metrics went up because you started showing everyone the same twenty items. The rule is that you always pair an engagement metric with a diversity guardrail.

### Offline-online gap
Common problem: model wins offline, loses online. Causes:
- Position bias in offline labels.
- Counterfactual issue: offline data is from old policy.
- Selection bias: only see clicks on items shown.
- Long-term effects offline can't measure.

> **Saying it out loud.** Offline wins that vanish online almost always come from the same root cause: your logged data was produced by the old policy. Position bias means top-slot items got clicked partly for being in the top slot. Selection bias means you only have labels for what the old model chose to show, so a great recommendation it never surfaced looks like a negative example. And offline evaluation measures one session while the business cares about a month. Inverse propensity weighting and dedicated exploration traffic shrink the gap but never close it, which is why nothing ships without an A/B test.

### A/B testing
The actual decision instrument. Usually 1-4 weeks per launch decision. (See A/B testing deep dive.)

> **Saying it out loud.** The A/B test is the actual decision instrument; everything else is a filter to decide what's worth testing. One to four weeks is typical, driven by how long it takes the metric you care about to move past the noise. The thing that trips people up is that recommender A/B tests violate independence, because your treatment users and control users share a catalog and can influence each other's popularity signals. On a marketplace that interference can be big enough that you need cluster-randomized or switchback designs instead of plain user-level splits.

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

> **Saying it out loud.** If I had to compress the gotchas: never quote accuracy on click prediction, because click-through is a couple of percent and predicting no always wins. Two-stage exists for the compute arithmetic, not tradition. NDCG is position-discounted, so the top slots dominate. Echo chambers are fixed by spending traffic on exploration, not by tweaking the ranker's temperature. And GBDT still beats deep learning on well-engineered tabular ranking features more often than people expect.

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
