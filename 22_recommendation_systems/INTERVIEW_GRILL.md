# Recommendation Systems — Interview Grill

> 45 questions on collab filtering, matrix factorization, two-tower, sequential models, ranking, evaluation. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. What does a recommender output?**
Ranked list of top-$k$ items for a (user, context) input.

> **Saying it out loud.** A recommender takes a user plus whatever context you have, the time of day, the device, the page they're on, and hands back an ordered list of items. The output is a ranking, not a prediction, and that distinction matters because it means only the top handful of positions really count. Nobody scrolls to slot forty. So the metrics that matter are all position-weighted, things like NDCG at ten, not overall accuracy.

**2. Common business targets?**
CTR, conversion, dwell time, retention, revenue. Pick one or weighted combination.

> **Saying it out loud.** Business targets are things like click-through rate, conversion, watch time, retention, and revenue, and the hard part is that they conflict. Optimizing purely for clicks gets you clickbait, optimizing purely for watch time gets you autoplay traps, and both of those quietly hurt retention. So real systems train on a weighted blend of several heads and tune the weights by A/B test. The honest framing to give an interviewer: the metric you can measure this week is a proxy for the thing you actually want, which is whether the user comes back next month.

**3. User-user vs item-item CF?**
User-user: similarity between users via shared items. Item-item: similarity between items via shared users. Item-item is Amazon's classic.

> **Saying it out loud.** User-user collaborative filtering says find people like you and recommend what they liked. Item-item flips it: find items that get liked by the same people and recommend things similar to what you just engaged with. Item-item is what Amazon shipped and it's usually the better choice in practice, because item-item similarities are far more stable over time than user tastes, so you can compute them offline in a batch job. Users change every day, the fact that two products go together doesn't.

**4. Why is CF data sparse?**
Most user-item pairs unobserved (a typical user interacts with $\ll 1\%$ of catalog).

> **Saying it out loud.** Because a person interacts with a laughably tiny slice of the catalog. A retailer might have ten million products and a good customer buys fifty, so you're looking at well under one percent, often more like a thousandth of a percent, of the matrix being filled in. Everything else is unknown, not zero, and confusing those two is a common modeling mistake. That extreme sparsity is exactly why low-rank methods work at all: you have no hope unless you assume a small number of latent factors explains everything.

**5. What's the cold-start problem?**
New users / new items have no interaction data. CF can't help; need content features.

> **Saying it out loud.** Cold start is the case where you have no interaction history to learn from, either a brand new user or a brand new item. Collaborative filtering is completely blind here, because it only knows about things through who has touched them. So you fall back on content: features of the item, whatever you know about the user, popularity as a floor. It's the single hardest structural problem in recommenders because new items arrive continuously and, unhandled, they never get shown, which means they never get data, which means they never get shown.

---

## B. Matrix factorization

**6. Matrix factorization core idea?**
$R \approx U V^\top$. Low-rank approximation of user-item interaction matrix.

> **Saying it out loud.** Matrix factorization says the giant user-by-item matrix is really low rank: everybody's taste can be described by maybe a hundred hidden dimensions, and so can every item. So you learn a short vector per user and per item such that their dot product reproduces the observed interactions. Then unobserved cells get predicted by the same dot product, which is how you get recommendations. The magic is that the latent dimensions are learned, not designed, and they often end up corresponding to interpretable things like genre or price tier.

**7. Loss for explicit ratings?**
$\sum_{(u,i)} (R_{ui} - u_u^\top v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$.

> **Saying it out loud.** For explicit ratings you minimize squared error on the cells you actually observed, plus L2 regularization on both embedding matrices. The crucial detail is that the sum runs only over observed pairs, not the whole matrix, because an unrated item is unknown, not a zero rating. Regularization is doing heavy lifting, since a user with three ratings and a hundred-dimensional embedding would otherwise overfit instantly. In practice you also add per-user and per-item bias terms, which alone capture a surprising fraction of the signal.

**8. How do you handle implicit feedback?**
Weighted ALS, BPR (pairwise ranking), or pointwise BCE on observed clicks vs sampled negatives.

> **Saying it out loud.** Implicit feedback breaks the explicit setup because you only ever see positives; nobody tells you they disliked something, they just don't click. Three standard approaches: weighted ALS, which treats all unobserved cells as weak negatives with low confidence and observed ones as strong positives; BPR, which learns a pairwise ranking that observed beats unobserved; or pointwise binary cross-entropy against sampled negatives. The thing that makes or breaks all three is negative sampling, since sampling by popularity versus uniformly changes the model's behavior dramatically.

**9. What's BPR loss?**
Bayesian Personalized Ranking: $-\log \sigma(s(u, i^+) - s(u, i^-))$. Pairwise; encourages observed > unobserved.

> **Saying it out loud.** BPR turns recommendation into a pairwise comparison problem. For each user you take an item they interacted with and one they didn't, and you train the score of the positive to be higher than the score of the negative, through a log-sigmoid of the difference. You're never asking for an absolute score, only for a correct ordering, which lines up with what a ranked list actually needs. The subtlety worth mentioning: with uniform negative sampling most negatives are trivially easy, so the gradients go to nearly zero and training stalls, which is why hard negative mining matters.

**10. Strengths of MF?**
Simple, fast, scales, captures latent factors automatically.

> **Saying it out loud.** Matrix factorization's virtues are that it's simple, trains fast, scales to hundreds of millions of users, and discovers latent structure without anybody hand-engineering features. Serving is a dot product, which is about as cheap as it gets, and it plugs straight into approximate nearest neighbor search. It remains a strong baseline that plenty of fancier deep models fail to beat. If a candidate proposes a transformer before establishing an MF baseline, that's a yellow flag.

**11. Weaknesses?**
Cold start, no side features, static (no sequence).

> **Saying it out loud.** Three real weaknesses. It can't handle cold start at all, because a new user or item has no embedding and nothing to train it from. It has no way to use side features, so your item's category and price and image are all invisible to it. And it's static, treating your interaction history as an unordered bag, so it can't tell that what you clicked five minutes ago matters more than what you clicked last year. The two-tower architecture exists to fix the first two, and sequential models fix the third.

---

## C. Two-tower retrieval

**12. Two-tower architecture?**
User encoder $f_u$ and item encoder $f_i$ → fixed-dim embeddings. Score = $u^\top v$.

> **Saying it out loud.** Two-tower means two separate encoders, one that turns the user and context into a vector and one that turns an item into a vector, with the score being just their dot product. It generalizes matrix factorization by letting each side be an arbitrary network over features rather than a lookup table. That fixes cold start, because a brand-new item still has features to encode. The constraint that defines the architecture is that the two towers never see each other until the final dot product.

**13. Why do encoders need to be independent?**
To pre-compute item embeddings offline and serve via ANN. If user-item features mixed, every (user, item) pair would need explicit forward pass.

> **Saying it out loud.** The towers have to stay separate so you can precompute every item embedding offline and put them in an ANN index. If the model mixed user and item features early, then scoring would require a full forward pass per user-item pair, which at ten million items and fifty milliseconds is completely impossible. Keeping them separate turns retrieval into a vector search, which is sublinear. That's the whole trade: you give up early feature interaction, which costs you accuracy, in exchange for retrieval being feasible at all, and you buy the accuracy back in the ranking stage.

**14. Training objective?**
Contrastive: positive pair (clicked item) → high score; negatives → low. In-batch negatives standard.

> **Saying it out loud.** You train it contrastively: the item the user actually engaged with should score high, and other items should score low, usually via a softmax over the positive and a set of negatives. In-batch negatives are the standard trick, where the positives of the other users in your batch serve as your negatives, so you get hundreds of negatives for free with no extra compute. The catch is that in-batch negatives are drawn by popularity, since popular items appear in batches more often, so you have to apply the logQ correction or the model will systematically under-rank popular items.

**15. Hard negative mining?**
Include "almost-positive" negatives (e.g., items shown but not clicked). Better than random negatives at training discrimination.

> **Saying it out loud.** Hard negatives are items that are plausible but wrong, typically things that were shown to the user and not clicked. Random negatives are usually so obviously irrelevant that the model separates them immediately and stops learning; hard negatives sit near the decision boundary where the gradient is. The standard recipe is a mixture, mostly in-batch randoms plus some fraction of mined hard negatives. Go all-in on hard negatives and training destabilizes, because some of your hard negatives are actually false negatives, items the user would have loved and simply never saw.

**16. Sampled softmax — why?**
Full softmax over millions of items intractable. Sample $K$ negatives from sampling distribution $Q$, then subtract $\log Q(j)$ from each sampled logit (the **logQ correction**) before applying softmax over sample. Yields an unbiased estimator of the full softmax gradient.

> **Saying it out loud.** A full softmax over ten million items is not something you can compute per training example, so you compute it over a sample of negatives instead. But sampling biases the result, because popular items get sampled more often, so you subtract the log of the sampling probability from each sampled logit. That logQ correction is what makes the sampled gradient an unbiased estimate of the full softmax gradient. Skip it and your model systematically penalizes popular items, and you'll see it as your recommendations getting weirdly obscure.

**17. Inference flow?**
Embed user → ANN search over precomputed item embeddings → return top-$K$.

> **Saying it out loud.** At serving time you run the user tower once on the live request to get a user vector, then hit an ANN index of precomputed item embeddings and pull back the top few hundred. One neural forward pass, one vector search, done. The item embeddings come from a batch job that reruns periodically. The staleness that introduces is the operational cost: a brand-new item isn't retrievable until the next index build, which is why fresh-content systems run incremental index updates.

**18. Why ANN, not exact?**
Exact KNN over millions is too slow. ANN (HNSW, IVF-PQ) trades small recall for huge speedup.

> **Saying it out loud.** Exact nearest neighbor over millions of vectors means scoring every one of them, and that's tens of milliseconds of pure compute per request before you've done anything else. ANN methods like HNSW or IVF-PQ give you the top results in a millisecond or two by only exploring part of the space. You lose a bit of recall, maybe one or two percent of true top results. That's a great trade, because the ranking stage downstream is going to reshuffle everything anyway, so a few borderline misses at retrieval barely move the final metric.

---

## D. Sequential models

**19. Why sequential models?**
User interests evolve. Recent clicks predict next click better than oldest clicks. Order matters.

> **Saying it out loud.** Because what you want changes, and recency carries most of the signal. If someone just spent twenty minutes looking at camping gear, that says far more about the next click than anything in their history from last spring. Order matters too: viewing a camera and then a lens means something different than the reverse. Plain matrix factorization treats history as an unordered bag and throws all of that away, which is why session-based models were such a step change on e-commerce data.

**20. GRU4Rec — what's the architecture?**
RNN over session of clicks. At each step, predict next item.

> **Saying it out loud.** GRU4Rec runs a recurrent network over the sequence of items in a session, and at each step predicts the next item. It was the paper that made session-based recommendation work, and the important insight was that you don't even need user IDs, the session itself carries enough signal. That matters commercially because a huge fraction of e-commerce traffic is logged out. The limitation is the usual RNN one: it's sequential so it can't parallelize over the timeline, and it struggles to hold onto anything far back.

**21. SASRec — improvement?**
Transformer self-attention instead of RNN. Better for long sequences.

> **Saying it out loud.** SASRec swaps the RNN for a causal self-attention stack, so every position can look directly at every earlier item rather than passing information down a chain. That means long-range dependencies survive and training parallelizes across the sequence. It's basically a small GPT over item IDs. The gain shows up most on long histories, and the cost is the usual quadratic attention cost, which is why production systems truncate to the last fifty or a hundred interactions.

**22. BERT4Rec — twist?**
Masked-item prediction (BERT-style) on the sequence. Bidirectional context.

> **Saying it out loud.** BERT4Rec trains bidirectionally: instead of predicting the next item, it masks random items in the sequence and predicts them from context on both sides. That gives each position a richer view during training and generally beats SASRec on offline benchmarks. The awkward part is the train-serve mismatch, since at serving time you genuinely only have the past, so you append a mask token at the end and predict there. Whether the offline gain survives online is genuinely contested, and saying that out loud is a good sign of judgment.

**23. Modern industry sequential setup?**
Transformer over user history; produce user representation; combine with item features for ranking.

> **Saying it out loud.** In industry the sequential model is usually one component, not the whole system. You run a transformer over the user's recent history to produce a user representation, and that representation becomes an input to both the retrieval tower and the ranking model alongside all the other features. So the sequence model is a feature extractor, not the ranker. The engineering constraint driving this is latency: you often precompute the user sequence embedding asynchronously and cache it, refreshing on a delay rather than on every request.

---

## E. Two-stage retrieval + ranking

**24. Why two-stage?**
Catalog ($N$) too large to rank exhaustively. Retrieval narrows to $K \ll N$. Then expensive ranking on $K$.

> **Saying it out loud.** Because you cannot run an expensive model over ten million items in fifty milliseconds, but you also can't get good results from a cheap model. So you split it: a cheap model that's very fast and just has to not lose the good stuff narrows millions down to hundreds, then an expensive model that sees rich cross-features carefully orders those hundreds. Retrieval optimizes recall, ranking optimizes precision at the top. The failure mode to name is that anything retrieval drops is gone forever, so retrieval recall is the ceiling on the entire system.

**25. Latency budgets?**
Retrieval ~10ms. Ranking ~30-50ms. Total ~50ms.

> **Saying it out loud.** Roughly ten milliseconds for retrieval and thirty to fifty for ranking, with a total budget around fifty to a hundred milliseconds for the recommendation service. That's not the whole page load, that's just your slice of it. These numbers drive every architecture decision downstream, which is why retrieval is a dot product and ranking only sees a few hundred candidates. And the number that actually governs the system is the p99, not the average, because the slowest one percent of requests is what users notice.

**26. Stage 1 methods?**
Two-tower ANN, item-item CF, popularity, rules. Often hybridized.

> **Saying it out loud.** Stage one is almost never one method, it's a union of several candidate generators run in parallel: a two-tower ANN model, item-item collaborative filtering off the user's recent activity, trending and popular items, and hand-written business rules for things like new releases or promoted content. You take the union and pass it all to ranking. The reason for the mix is coverage: each source has a blind spot, and the union is more robust than any single retriever. It also gives you a clean place to inject freshness and diversity before ranking narrows things down.

**27. Stage 2 methods?**
GBDT, DeepFM, DLRM, transformer-based rankers.

> **Saying it out loud.** Stage two is where you can afford real models, because you're only scoring a few hundred candidates. Gradient boosted trees are still extremely competitive on the tabular feature sets these systems produce. Deep models like DeepFM and DLRM handle high-cardinality categorical features through embeddings, and transformer-based rankers that attend over the candidate set are the current frontier. Whatever you use, the ranker sees full user-item cross features, which is precisely what the retrieval tower was forbidden from doing.

**28. When three stages?**
Pinterest, YouTube: retrieval (1M → 1000) → coarse ranking (1000 → 100) → final (100 → top-$K$).

> **Saying it out loud.** You add a third stage when the gap between retrieval and ranking is too wide to bridge in one step. YouTube and Pinterest go roughly a million down to a thousand with retrieval, a thousand down to a hundred with a lightweight ranker, then a hundred down to the final list with the heavy model. Each stage's model gets more expensive per item as the candidate count shrinks, keeping total compute roughly flat. The cost is another model to train, monitor, and keep consistent with the ones on either side of it.

---

## F. Ranking models

**29. Why does GBDT often win for ranking?**
Tabular interaction features; robust; fast; interpretable; handles missing data well.

> **Saying it out loud.** Because ranking features are tabular, and boosted trees are still the best thing we have for tabular data. They pick up interactions automatically, handle missing values without imputation, don't care that one feature is a count and another is a ratio, and they train in minutes so you can iterate. Neural rankers only pull ahead when you have very high-cardinality categoricals or raw content that needs embedding. The other practical win is that you get feature importances, so when the model does something stupid you can find out why.

**30. DeepFM — what's the idea?**
Factorization Machines (low-order interactions) + Deep MLP (high-order). Combined.

> **Saying it out loud.** DeepFM runs two things side by side on the same embeddings: a factorization machine part that explicitly models every pairwise feature interaction, and a deep MLP that can learn higher-order ones. The outputs get summed before the sigmoid. The point is that MLPs are surprisingly bad at learning simple multiplicative interactions from scratch, so you hand them to the FM component explicitly instead. Compared to the earlier Wide and Deep, the advantage is that no manual cross-feature engineering is required.

**31. DLRM architecture?**
Categorical embeddings → element-wise dot products + dense features → MLP.

> **Saying it out loud.** DLRM is Meta's click-through model. Categorical features go through embedding tables, dense features go through a bottom MLP, then you take explicit pairwise dot products between all those vectors and feed the result plus the dense vector into a top MLP. The explicit dot-product interaction layer is the architectural choice, and it's motivated by the same observation as DeepFM: you should build multiplicative interactions in rather than hope for them. The dominant engineering fact about DLRM is that the embedding tables run to hundreds of gigabytes, so the whole system is built around sharding them across machines.

**32. LambdaMART — what is it?**
GBDT trained with pairwise/listwise ranking objectives (LambdaRank). Produces ranking-aware models.

> **Saying it out loud.** LambdaMART is boosted trees trained with a ranking objective instead of a regression one. The trick is that ranking metrics like NDCG are flat and non-differentiable, so instead of differentiating the metric you directly specify the gradient: for each pair of documents, weight the pairwise gradient by how much NDCG would change if you swapped them. Those weighted pseudo-gradients are the lambdas. It dominated learning-to-rank competitions for a decade and it's still the strong baseline in search ranking.

**33. Pointwise vs pairwise vs listwise?**
Pointwise: predict score per item. Pairwise: predict $i > j$. Listwise: optimize full list metric (e.g., NDCG).

> **Saying it out loud.** Three ways to frame ranking. Pointwise treats each item independently and predicts a score or a click probability, which is simple and lets you use any classifier but ignores that ranking is comparative. Pairwise learns which of two items should come first, which matches the ordering task better, and that's BPR and RankNet. Listwise optimizes a metric over the entire list, like NDCG, which is what you actually care about but is the hardest to optimize because the metric isn't differentiable. Industry mostly runs pointwise for calibration reasons, since you often need a real click probability for downstream ad auctions, not just an order.

---

## G. Evaluation

**34. NDCG@K formula intuition?**
$\mathrm{DCG@K} = \sum_{i=1}^K (2^{\mathrm{rel}_i} - 1) / \log_2(i+1)$ (graded relevance form; for binary relevance simplifies to $\sum \mathrm{rel}_i / \log_2(i+1)$). Position-discounted relevance with $i$ 1-indexed (rank). Normalized by ideal ordering: $\mathrm{NDCG} = \mathrm{DCG}/\mathrm{IDCG}$. Higher = better.

> **Saying it out loud.** NDCG asks how good your ordering is, with credit discounted the further down the list something appears. You sum each item's relevance divided by a log of its position, so position one counts fully and position ten counts about a third as much, then you divide by the score of the perfect ordering so the result lands between zero and one. The normalization is what makes it comparable across queries with different numbers of relevant items. It's the default ranking metric precisely because it handles graded relevance and position discount at once.

**35. MAP@K?**
Mean Average Precision. Average precision at each correct hit position.

> **Saying it out loud.** Mean Average Precision: for each user, you walk down the ranked list and every time you hit a relevant item you record the precision at that point, then average those, then average over users. It rewards getting all your relevant items high, not just the first one. Compared to NDCG, MAP only handles binary relevance, no graded scores. It's the right metric when there are several correct answers and you care about all of them, like retrieval for a research tool.

**36. MRR?**
Mean Reciprocal Rank: $1/k_1$ where $k_1$ is position of first hit. Captures "did we rank a correct item near top?"

> **Saying it out loud.** Mean reciprocal rank is one over the position of the first correct item, averaged over queries. Get it at position one and you score one, position two and you score a half, position ten and you score a tenth. It only cares about the first hit and completely ignores everything after it. That makes it the right metric when there's exactly one right answer, like a lookup or a navigational search, and the wrong one for a feed.

**37. Hit@K?**
1 if any correct item in top-$K$, else 0. Simple recall measure.

> **Saying it out loud.** Hit rate at K is the blunt one: did any relevant item show up in the top K, yes or no, averaged over users. No credit for position, no credit for finding several. It's easy to explain and it's a reasonable proxy for whether the retrieval stage is doing its job. Use it for retrieval recall, not for final ranking quality, because a system that always puts the right answer at position ten looks identical to one that puts it at position one.

**38. AUC for ranking?**
Pairwise: probability that positive item ranks above negative. Threshold-free.

> **Saying it out loud.** AUC in a ranking setting is the probability that a randomly chosen positive scores above a randomly chosen negative. It's threshold-free, so it measures the ordering rather than any particular cutoff. The problem is that it weights every position equally, so improving the ordering down at rank five thousand counts as much as improving rank two, and nobody sees rank five thousand. That's why AUC can go up while NDCG at ten goes down, which is a genuinely common and confusing experience.

**39. Why does offline often disagree with online?**
Position bias (top items click more), counterfactual (offline data from old policy), selection bias (only see clicks on shown), long-term effects.

> **Saying it out loud.** Offline and online disagree because your logged data was generated by your old model, not your new one. Position bias means items shown at the top got clicked partly because they were at the top. Selection bias means you only have labels for things the old policy chose to show, so a great recommendation the old model never surfaced looks like a negative. And offline metrics measure a single session while the business cares about whether the user comes back in a month. The mitigations are inverse propensity weighting for the position bias and exploration traffic so you have unbiased data at all, but nothing removes the gap, which is why everything ships behind an A/B test.

**40. Holdback test for recommenders?**
Permanent control population on old model. Catches long-term drift / degradation that A/B doesn't.

> **Saying it out loud.** A holdback is a small slice of users kept permanently on the old model, or on no personalization at all, rather than being folded back in when a test ends. Regular A/B tests run for a couple of weeks, which is too short to catch slow effects like the feed narrowing or users getting bored. The holdback gives you a long-run baseline so you can measure the cumulative effect of a year of shipped changes, which is frequently smaller than the sum of the individual wins. The cost is real: you're deliberately giving some users a worse experience indefinitely, so you keep the group tiny and rotate it carefully.

---

## H. Cold start and exploration

**41. New user — what do you do?**
Popularity, demographics, onboarding survey, exploration.

> **Saying it out loud.** For a brand-new user you stack fallbacks in order of how much you know. Start with what's popular, ideally popular within whatever coarse segment you can infer from country, device, and referrer. If the product supports it, an onboarding step where they pick a few interests gives you enormous signal for thirty seconds of effort. Then explore aggressively for the first few sessions, because early interactions are worth far more than later ones. And the thing to say is that personalization ramps within a session, not over weeks, since the first few clicks are the strongest signal you'll ever get.

**42. New item — what do you do?**
Content features (CLIP for images, text encoder for descriptions), forced exposure, similar-to-existing.

> **Saying it out loud.** For a new item you lean entirely on content. Encode the image with something like CLIP, encode the title and description with a text model, use the category and the seller and the price, and place the item in embedding space next to items it resembles. Then you have to force some exposure, deliberately showing it to a small slice of traffic to buy interaction data, because otherwise it never earns its way in. That forced exposure costs you short-term engagement and is straightforwardly an exploration budget. On a marketplace it isn't optional, because sellers leave if their new listings get no impressions.

**43. Echo chamber problem?**
Greedy ranking reinforces popular items; users see less diversity over time. Filter bubbles.

> **Saying it out loud.** The echo chamber comes from the system optimizing exactly what you told it to. Ranking greedily by predicted engagement means you show what's worked before, the user engages with it, that becomes training data, and the loop tightens. Over time the user sees a narrower and narrower slice, and the model never learns what else they might have liked because it stopped asking. It's a feedback loop, not a bias in the data, which is why you can't fix it by cleaning the dataset. You fix it by spending some traffic on exploration and adding an explicit diversity term, both of which cost measurable short-term engagement.

**44. Thompson sampling for exploration?**
Maintain posterior over each item's reward. Sample, act greedily under the sample. Natural exploration-exploitation balance.

> **Saying it out loud.** Thompson sampling keeps a probability distribution over how good each item is rather than a single estimate, then draws a sample from each distribution and acts greedily on the samples. Items you're uncertain about have wide distributions, so they sometimes draw high and get shown, which is exactly the exploration you want, and it fades automatically as the distribution narrows. There's no exploration rate to tune, unlike epsilon-greedy. It generally beats UCB empirically and it's a natural fit for recommendation because click data gives you clean Beta-Bernoulli posteriors.

**45. Diversity bonus?**
Add penalty for items similar to ones already in the recommended list. Encourages variety.

> **Saying it out loud.** A diversity bonus means you don't just take the top-scoring items independently, you build the list greedily and penalize each candidate by how similar it is to what you've already picked. MMR is the classic formulation, with one knob trading relevance against novelty. The reason you need it is that the top ten by score are often ten near-duplicates, which is a worse list than eight good items plus two different ones. It costs you predicted click-through and usually buys you session length and retention, which is the tradeoff you argue about with the product team.

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
