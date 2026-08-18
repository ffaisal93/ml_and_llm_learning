# ML System Design — Interview Grill

> 45 questions on the ML system design framework, common patterns, and failure modes. Drill until you can answer 30+ cold.

---

## A. The framework

**1. Six steps in an ML system design answer?**
Clarify requirements → frame as ML problem → data → features+model → serving → monitoring & iteration.

> **Saying it out loud.** Six steps, and the order is the whole point. Clarify what we're building and for whom, turn it into an ML problem with inputs and a loss, figure out where the data and labels come from, then features and model, then how it serves, then how you know it's still working. Model selection is step four, not step one. If you keep that order under pressure, you'll cover the things interviewers actually score, and the ones you skip you can at least name as skipped.

**2. What goes wrong if you skip clarification?**
You design for the wrong problem. Latency targets, scale, cost asymmetry — all change the design.

> **Saying it out loud.** Skipping clarification means you might build a beautiful system for a problem nobody has. Latency changes everything — a hundred-millisecond budget and a nightly batch job are completely different architectures. So does scale: a million items and ten billion items need different retrieval. And cost asymmetry decides your threshold. Three or four questions up front is maybe ninety seconds, and it prevents twenty minutes of designing the wrong thing.

**3. Three things to ask about scale?**
QPS at peak, catalog/user size, latency budget.

> **Saying it out loud.** I want peak queries per second, the size of the thing we're searching over, and the latency budget. QPS tells me whether this is one box or a fleet. Catalog size tells me whether I need approximate nearest neighbors or can brute-force it. And the latency budget decides the model class — under a hundred milliseconds means no cross-encoder over thousands of candidates. Those three numbers determine most of the architecture before I've said a single model name.

**4. Three things to ask about cost asymmetry?**
Cost of false positive vs false negative; cost of latency miss; cost of being unavailable.

> **Saying it out loud.** I ask what a false positive costs, what a false negative costs, and what being slow or down costs. The first two set the operating threshold — fraud is maybe a hundred to one against misses, spam is ten to one the other way. The third decides how much I invest in fallbacks and caching. And there's a fourth flavor worth naming: the cost of being unavailable, because a system that's right ninety-nine percent of the time and down one percent of the time may be worse than a dumber system that never fails.

**5. Why frame the ML problem before talking model?**
The right model class falls out of input/output/loss. Rushing to model selection skips this.

> **Saying it out loud.** The model falls out of the framing, so framing first saves you from arguing about architectures in a vacuum. Once I've said the input is a user and a candidate item, the output is a relevance score, and the loss is pairwise, the space of sensible models is small. If I pick the model first, I end up retrofitting the problem to the tool. The other benefit is that framing exposes the label question early, which is usually the hardest part of the whole design.

---

## B. ML problem framing

**6. Is fraud detection a binary classification or anomaly detection problem?**
Both framings are valid. Often binary classification with class weights / cost-sensitive loss; sometimes anomaly when labels are very rare.

> **Saying it out loud.** Both framings work, and the honest answer is that it depends on labels. If you have confirmed fraud labels — chargebacks, investigations — treat it as supervised binary classification with cost-sensitive weights, because supervised beats unsupervised whenever labels exist. If you're in a new market with essentially no labels, or you're worried about attack types you've never seen, unsupervised anomaly detection buys you coverage. In practice mature systems run both: a supervised model for known patterns and an anomaly layer for the novel stuff, because the supervised model is blind to attacks it wasn't trained on.

**7. Is recommendation a classification problem?**
Usually framed as ranking (pointwise/pairwise/listwise) with implicit feedback.

> **Saying it out loud.** You can train it with a classification loss, but the task is ranking. What matters isn't whether each item is labeled click or no-click, it's whether the right item ends up above the wrong one in the top ten. That's why the metrics are NDCG and recall at K rather than accuracy, and it's why pairwise and listwise losses often beat pointwise. The other wrinkle is that the feedback is implicit — a non-click isn't a negative, it might just mean they never saw it.

**8. Is "predicted user lifetime value" a classification problem?**
Regression. Sometimes binned into ranges and treated as ordinal classification.

> **Saying it out loud.** Lifetime value is a regression problem, since the target is a dollar amount. But the distribution is nasty — heavily skewed, a big spike at zero, and a long tail of whales — so plain squared error gets dominated by the tail. In practice people either log-transform, model it as a two-stage problem (will they spend at all, and if so how much), or bin it into ordinal buckets, which is often what the business wants anyway since they act on tiers, not exact numbers.

**9. When use pointwise vs pairwise vs listwise ranking loss?**
Pointwise: simple, learn $f(u, i)$. Pairwise: relative orderings via $f(u, i_+) > f(u, i_-)$. Listwise: full list (NDCG-like). Pairwise often wins for medium-data systems; listwise for ranked-output evaluation.

> **Saying it out loud.** Pointwise treats each item independently and just predicts a score, which is simple and lets you reuse any regressor. Pairwise learns from comparisons — this item should beat that one — which matches what ranking actually is and usually wins in practice. Listwise optimizes the whole ordering against something like NDCG directly, which is the most faithful but the most complex and data-hungry. My default is pairwise for medium data, and I'd move to listwise only if the evaluation metric is strictly list-based and I have the volume to support it.

---

## C. Data

**10. What's label leakage in churn prediction?**
Using features computed *after* churn determined: e.g., "days since last login" computed including the churn event. Model trivially predicts churn from itself.

> **Saying it out loud.** The classic churn leak is using anything that only exists because the churn already happened. "Days since last login" computed through the cancellation date is really just measuring the cancellation. Same with a support ticket filed on the way out, or a downgrade a week before. The rule is that every feature must be computable at the moment you'd actually score the customer, with a strict cutoff. And the tell is offline AUC around 0.99 — real churn models live in the 0.75 to 0.85 range, so if you're way above that, go hunting.

**11. How do you split for time-series data?**
Train on past, validate on slightly newer past, test on recent. Never random.

> **Saying it out loud.** You split by time, always. Train on the oldest chunk, validate on a slightly newer chunk, test on the most recent — which mirrors what production does. A random split lets the model see Wednesday when predicting Tuesday, and your metrics become fiction. For a serious estimate I'd use walk-forward validation, refitting across several rolling cutoffs so I can see whether performance decays as the gap grows. The extra thing to watch is that features built from rolling windows have to respect the same cutoff, or you leak through the back door.

**12. How do you handle imbalanced data?**
Right metric (PR-AUC, F1), class weights in loss, threshold tuning, focal loss, resample only train. Not test.

> **Saying it out loud.** Imbalance is mostly a metrics and threshold problem, not a data problem. First, use the right metric — PR-AUC or recall at a fixed false-positive rate, since accuracy is meaningless at a one percent base rate. Second, use class weights or focal loss so the rare class isn't drowned out. Third, tune the threshold on the cost ratio instead of leaving it at 0.5. And if you do resample, resample only the training set — touching validation or test gives you numbers that don't reflect production, which is the most common mistake here.

**13. How do you check for data drift?**
KS / PSI on input feature distributions, per-feature distribution monitoring, classifier-based shift detection.

> **Saying it out loud.** Drift detection is about watching the inputs, because that's what you can see immediately. Per-feature, I'd track the distribution and compute something like population stability index or a KS test against a reference window — PSI above 0.2 is the usual alarm. For multivariate drift, train a classifier to tell training data from recent production data; if it can, you've drifted. The reason all this matters is that accuracy needs labels, and labels might be thirty days late, so input monitoring is your early warning system.

**14. What's a feature store?**
Centralized service for feature definitions and values. Reduces online/offline skew (same code computes train and serve features). Examples: Feast, Tecton, in-house.

> **Saying it out loud.** A feature store is one place where feature definitions live, so training and serving compute them the same way. The problem it solves is training-serving skew: someone writes a SQL aggregation for training and an engineer reimplements it slightly differently in the serving path, and now your production model sees features it never trained on. It also gives you point-in-time-correct joins, which is what stops leakage in the first place, plus a low-latency online store for real-time lookups. The cost is real infrastructure, so for a small team a shared library of feature code gets you most of the benefit.

---

## D. Modeling

**15. Why do GBDTs often beat deep learning on tabular?**
Tabular data has heterogeneous, sparse features. GBDTs split on individual features intelligently; DL models need careful preprocessing and may not benefit from depth.

> **Saying it out loud.** Tabular data is a bunch of unrelated columns with different scales and meanings, and trees are built for exactly that. A tree can split on "income above sixty thousand" directly, no normalization, no embedding, missing values handled natively. Neural nets assume smooth structure and locality, which images and text have and spreadsheets don't. Plus trees need almost no tuning to be good. The honest framing is that deep learning wins when there's structure to exploit — sequences, pixels, language — and on plain tables GBDT still matches or beats it with a tenth of the effort.

**16. When should you use deep learning?**
Lots of data + perceptual signals (images, text, audio, sequences). Or end-to-end embedding learning.

> **Saying it out loud.** Deep learning earns its keep when the input has structure a network can exploit: images, audio, text, or sequences where order matters. It also wins when you want to learn embeddings you can reuse across tasks, or when you have very high-cardinality entities like millions of user or item IDs. And it needs data — usually hundreds of thousands of examples minimum. Below that, or on plain tabular features, you're paying in latency, tuning time and ops complexity for accuracy you won't get.

**17. When should you use an LLM?**
Task is fundamentally about language. Or zero-shot with limited labels. Or generative output.

> **Saying it out loud.** Use an LLM when the task is genuinely about language, or when you need something generative, or when you have no labels and zero-shot is the only way to start. It's a fantastic way to get a v0 shipped in a week and to bootstrap labels for a smaller model. What it isn't is a good default for high-volume classification — a fine-tuned DistilBERT will be a hundred times cheaper and faster at the same accuracy. So my rule is: LLM to prototype and to handle the long tail, distilled small model for the head of the distribution.

**18. Why is two-stage retrieval common?**
Cheap retrieval (ANN on embeddings) over millions, expensive ranking on top-K (e.g., 100s). Saves compute.

> **Saying it out loud.** You can't score ten million candidates in fifty milliseconds, so you split the work. Retrieval is cheap per item and runs over everything, usually a dot product against an ANN index, narrowing millions to a thousand. Ranking is expensive per item but runs over almost nothing, so you can afford hundreds of features and a heavy model. The catch is that it's a funnel — whatever retrieval drops is gone forever, so recall at stage one is a hard ceiling on the whole system, and that's the number I'd monitor.

**19. Two-tower model?**
User encoder and item encoder produce embeddings; score is dot product. Used for retrieval. Trained on (user, positive item, negative items).

> **Saying it out loud.** Two towers means the user and the item get encoded separately into the same vector space, and the score is just their dot product. The reason that structure matters is that item embeddings can be precomputed and stuck in an ANN index, so at request time you only encode the user once and do a nearest-neighbor lookup. The price you pay is that user and item features never interact until the final dot product, so it can't learn "this user likes this item *when* it's raining" — that's what the ranker is for. Negative sampling is where most of the difficulty lives; in-batch negatives are the standard trick.

**20. Cold start strategies?**
Content-based features (no need for history), popularity fallback, explicit exploration ($\epsilon$-greedy, Thompson sampling), few-shot embeddings.

> **Saying it out loud.** Cold start is about substituting content for behavior until behavior exists. For a new item, use what you know about it — text, images, category — and score it with a model trained on established items. Fall back to popularity, which is a surprisingly strong default. Then explore deliberately, with epsilon-greedy or Thompson sampling, so uncertain items actually get impressions. And blend from content-based to behavioral with Bayesian shrinkage as data accumulates rather than flipping a switch. The tradeoff is that forced exposure costs a measurable slice of aggregate CTR today to keep supply alive tomorrow.

---

## E. Serving

**21. Online vs batch prediction — when?**
Online: latency-sensitive, request-driven (search, fraud). Batch: pre-computable, lookup-driven (daily recommendations). Mixed: precompute offline, refine online.

> **Saying it out loud.** Online means you compute the prediction when the request arrives — necessary when the input is the request itself, like a fraud check or a search query. Batch means you precompute overnight and just look up the answer, which is far cheaper and simpler when the prediction doesn't depend on live context. The tradeoff is freshness versus cost and latency. The hybrid is common and often the right answer: precompute candidates in batch, then do a light online re-rank with real-time context like what the user did in the last five minutes.

**22. Where does latency go in online serving?**
Network → feature lookup → inference → post-processing. Often feature lookup dominates if not cached.

> **Saying it out loud.** People assume it's the model, and it usually isn't — it's the feature fetch. You're doing several round trips to a key-value store to assemble user features, item features and counters, and each one is a network hop. Then inference, which for a GBDT is well under a millisecond, and post-processing like business rules and deduplication. So the first thing I'd do is profile, then batch or parallelize the feature lookups and cache the hot ones. Optimizing a model that's five percent of your latency is the classic wasted week.

**23. Caching strategies?**
Per-user cache, per-pair cache, popular-prediction cache. Trade staleness for latency.

> **Saying it out loud.** Caching is how you buy latency with staleness. You can cache per user — this user's feed for the next ten minutes — or per user-item pair, or just cache predictions for popular items, which is where the hit rate lives since traffic is power-law distributed. The question that decides your TTL is how fast the right answer changes: a news feed goes stale in minutes, a similar-items module on a product page can be cached for a day. The failure mode is serving a stale cache after a model rollback, so cache keys should include the model version.

**24. How does quantization help inference?**
INT8/FP8 inference is 2-4× faster, smaller memory. Slight accuracy loss usually recoverable with calibration.

> **Saying it out loud.** Quantization means storing and computing weights in lower precision — INT8 or FP8 instead of FP32. You get roughly two to four times faster inference and a big memory reduction, which often matters more since it's what lets a model fit on one GPU. The accuracy loss is usually under a percent if you do post-training quantization with a calibration set, and near zero with quantization-aware training. The tradeoff is engineering time and hardware support, and the thing to watch is that outlier activations in large models can blow up under naive quantization.

**25. Distillation — what for?**
Train small student model on big teacher's outputs. Faster inference at small accuracy cost.

> **Saying it out loud.** Distillation trains a small student model to imitate a big teacher's outputs rather than the hard labels. The reason it works better than just training the small model directly is that the teacher's soft probabilities carry information about how classes relate — that this image is mostly cat but a little bit fox — which is a richer signal than a one-hot label. You typically keep most of the accuracy at a fraction of the cost; DistilBERT is about sixty percent of BERT's size at roughly ninety-seven percent of its performance. The cost is that you need the teacher first and an extra training pipeline.

**26. ANN vs exact KNN?**
ANN (HNSW, IVF, PQ) trades small recall loss for huge speedup. Standard for any retrieval at scale.

> **Saying it out loud.** Exact nearest neighbor means comparing your query against every vector, which is linear in corpus size and hopeless at ten million items. Approximate methods like HNSW build a graph you can walk in roughly logarithmic time, and IVF or product quantization trade memory for speed. You give up a bit of recall — typically you tune to about ninety-five to ninety-nine percent — for a hundred-fold or more speedup. That's essentially always the right trade in retrieval, because the ranker downstream will re-sort things anyway, and a recall miss on the true top item is rare.

**27. p50 vs p99 latency — which to optimize?**
Both, but p99 matters most for user-facing systems. Tail latencies cause cascading failures.

> **Saying it out loud.** You optimize both but you're graded on p99, because the tail is what users and downstream systems actually feel. A great median with a terrible tail means one request in a hundred hangs, and in a fan-out architecture where one page makes fifty backend calls, a one-percent tail means half your pages are slow. Worse, slow requests hold connections and queue up, which is how tail latency turns into an outage. So I'd set an SLO on p99, and use timeouts with a degraded fallback so a slow path fails fast instead of cascading.

---

## F. Monitoring

**28. Three layers of metrics?**
Infra (latency, errors), model (score distribution, calibration), business (CTR, revenue, retention).

> **Saying it out loud.** Three layers, and you need all of them because each catches a different failure. Infra metrics — latency, error rate, throughput — tell you the system is up. Model metrics — score distribution, calibration, feature drift — tell you the model is still sane, and they're available immediately. Business metrics — click-through, revenue, retention — tell you it's actually working, but they're noisy and lagging. The pattern to notice is that the fastest signals are the least meaningful and the most meaningful are the slowest, which is why you monitor all three.

**29. Why monitor input distributions, not just accuracy?**
Accuracy needs labels — usually delayed (days/weeks). Input distributions are immediate signals of drift.

> **Saying it out loud.** Because accuracy needs labels, and labels are late. In churn you find out in thirty days, in fraud you find out when a chargeback lands weeks later, in lending it could be months. If you wait for accuracy to degrade, you've already served bad predictions for a month. Input distributions, by contrast, are available the instant a request comes in, so a feature suddenly going ninety percent null — usually an upstream pipeline break — shows up in minutes. The tradeoff is that input drift is a proxy: it can fire when nothing is actually wrong, and it can miss drift in the input-output relationship.

**30. Calibration drift — what is it?**
Even if AUC stable, predicted probabilities may shift. Matters for downstream cost-sensitive decisions.

> **Saying it out loud.** Calibration drift is when the ordering is still fine but the numbers stop meaning what they say. AUC looks unchanged, yet your "0.3" predictions now happen forty percent of the time. That's fine if you only rank, and it's a real problem the moment you multiply probability by a dollar value to make a decision — every cost-weighted threshold you set is now wrong. So I'd monitor reliability curves or expected calibration error alongside AUC, and refit a calibration layer on recent data, which is cheap and doesn't require retraining the model.

**31. Shadow vs canary deployment?**
Shadow: model serves traffic but predictions discarded; compare to prod. Canary: small live %. A/B: full split.

> **Saying it out loud.** Shadow means the new model scores live traffic but nobody sees its output — you just log it and compare against production. That's your safety check: it catches crashes, latency blowups and crazy score distributions with zero user risk, but it can't tell you anything about user behavior since nothing changed for them. Canary means a small slice of real users actually get the new model, so you do learn behavioral effects, at real risk. The sequence I'd run is shadow, then a one-percent canary, then a proper A/B for the launch decision.

**32. When retrain on a schedule vs trigger?**
Schedule: predictable drift rates. Trigger: when monitoring detects shift. Often both: schedule + trigger.

> **Saying it out loud.** Schedule is the floor, triggers catch the surprises. A fixed cadence — weekly for search, daily for fraud, monthly for churn — is predictable and easy to automate. But drift doesn't respect your calendar, so you also want triggers on feature drift or a performance drop. The best setup is both, plus automated validation gates so a retrained model can't ship if it fails on a held-out set. The failure mode of pure triggers is that a slow, silent decay never crosses the threshold, and the failure mode of pure schedules is that you're two weeks late to a real break.

**33. Echo chamber risk in recommenders?**
Model's recommendations bias future training data → reinforcing loop. Mitigate with exploration, diversity bonuses, popularity floor.

> **Saying it out loud.** The recommender's own output becomes tomorrow's training data, and that's a feedback loop. If it only shows popular items, only popular items get clicks, which confirms they're popular — and the long tail becomes invisible to both the user and the model. Users get narrower feeds, sellers of niche items leave, and your training set stops containing information about anything you didn't already show. The mitigations are exploration, an explicit diversity term, and a floor of impressions for unexplored items. The tradeoff is that all of them cost short-term engagement to buy long-term catalog health.

---

## G. Cost & trade-offs

**34. Compute vs latency vs accuracy — pick two?**
You usually pick two. More accuracy → bigger model → more compute / latency.

> **Saying it out loud.** Compute, latency and accuracy — you get two. A bigger model is more accurate but slower, unless you throw more hardware at it, which is money. Distillation and quantization let you cheat a little at the margins, buying speed for a small accuracy hit. The way to make this concrete in an interview is to state the budget first: "we have fifty milliseconds and this much GPU, so the ranker can be about this big" — that turns an abstract tradeoff into a design constraint you're clearly managing rather than ignoring.

**35. Cost asymmetry — example?**
Fraud: false negatives cost much more than false positives. Adjust threshold (operating point) accordingly.

> **Saying it out loud.** Fraud is the cleanest example: a missed fraud is a direct write-off, while a false decline is friction and some goodwill, so the ratio is something like a hundred to one and the threshold should be far below 0.5. But the direction flips depending on the domain — for a spam filter, losing a real email is much worse than letting spam through, maybe ten to one the other way, so the threshold goes up. The general point is that the threshold is a business input, not a modeling default, and I'd ask for the two costs before setting it.

**36. When pre-compute vs compute on the fly?**
Pre-compute: small candidate set, mostly stable. On-fly: large input space, freshness matters.

> **Saying it out loud.** Precompute when the space of things to predict is small and stable and the answer doesn't change fast — daily recommendations for ten million users is fine, that's a nightly job and a lookup table. Compute on the fly when the input includes something you can't know in advance, like a search query or the current transaction, or when freshness matters within minutes. The hybrid is usually best: precompute the expensive candidate generation, then apply a light online layer with real-time signals. The cost of precomputing is staleness and storage; the cost of on-the-fly is latency and serving spend.

**37. Batch vs streaming — what's the difference?**
Batch: process windows of data, periodic. Streaming: per-event, low latency. Streaming is harder to debug.

> **Saying it out loud.** Batch processes accumulated data on a schedule — big windows, high throughput, easy to reason about and easy to re-run when something breaks. Streaming processes each event as it arrives, giving you seconds-fresh features, which fraud velocity counters absolutely need. The reason streaming is harder is everything around correctness: out-of-order events, late arrivals, exactly-once semantics, and the fact that you can't just rerun yesterday. So my default is batch unless freshness genuinely changes the product, and even then I'd keep a batch path as the source of truth for backfills.

---

## H. Failure modes

**38. What do you do when a model fails in production?**
Roll back to last good version. Then investigate: data shift, feature pipeline bug, code regression, infra issue.

> **Saying it out loud.** First thing is stop the bleeding — roll back to the last known good model. You don't debug in production while users are getting bad predictions. Once you're stable, work backwards through the likely causes: did an upstream feature pipeline change, did the input distribution shift, did someone deploy code, or is it infra? The fastest diagnostic is usually comparing feature distributions between now and training. And the thing that makes all this possible is having versioned models with a one-command rollback ready before you ever need it.

**39. Fallback for serving outage?**
Cached predictions, popularity-based defaults, last-known-good model, rules.

> **Saying it out loud.** You need a degradation ladder, not a single fallback. Serve cached predictions if you have them; if not, drop to a simpler model that needs fewer features; if that's gone too, fall back to popularity or a rules engine; and at the very bottom, a static default that's at least sensible. The principle is graceful degradation — the product stays up with worse recommendations rather than showing an error. And you have to actually test these paths, because an untested fallback is just an outage you haven't discovered yet.

**40. Adversarial users — how do you defend?**
Rate limiting, anomaly detection, robust features, human review of edge cases, retrain on adversarial examples.

> **Saying it out loud.** You defend in layers, because there's no single fix against someone actively trying. Rate limiting and cost-raising measures cut the volume of probing. Robust features help — anything trivially controllable by the attacker, like a self-reported field, is a liability. Anomaly detection catches behavior that's weird even if it isn't a known pattern. Human review handles the ambiguous edge. And you retrain frequently on fresh adversarial examples, because in an adversarial domain your data distribution is being actively steered by your opponent.

**41. Cold-start item — how do you bootstrap?**
Content features, popularity boost initial period, explicit exploration, force-show in some sessions.

> **Saying it out loud.** For a brand-new item, you have content but no behavior, so you score it from content — text and image embeddings, category, seller reputation — using a model trained on items that do have history. Then you buy the missing data: boost it for an initial window or reserve a slice of impressions so it accumulates real signal. As clicks come in, blend from predicted to observed with Bayesian shrinkage rather than switching abruptly, since ten impressions is not evidence. The cost is a small, measurable hit to aggregate CTR, and you accept it because the alternative is a death spiral for new supply.

**42. What's an SRE-style runbook for ML?**
Standard incident response: detection → mitigation (rollback) → investigation (logs, metrics) → root cause → preventive fix. ML adds: data quality checks, feature pipeline diff, retraining ablation.

> **Saying it out loud.** It's the standard incident loop — detect, mitigate, investigate, root cause, prevent — with ML-specific steps bolted on. Detection has to include model metrics and drift, not just five-hundreds. Mitigation is almost always a model rollback, which means rollback has to be one command. The investigation adds things a normal runbook doesn't have: diff the feature distributions against training, check whether an upstream data source changed schema, and check what changed in the last retrain. The part teams forget is that the data pipeline is part of the service, so a silent schema change upstream is a production incident.

---

## I. Worked-example shortcuts

**43. Recommender — first words?**
"Two-stage: retrieval (ANN over embeddings) → ranking (GBDT or DL on top-K). Latency budget determines ranker capacity."

> **Saying it out loud.** For a recommender I'd open with the shape: two stages, retrieval then ranking. Retrieval is a two-tower model with an ANN index pulling maybe a thousand candidates out of millions in a few milliseconds; ranking is a heavier model with rich cross features scoring just those. Then I'd say the latency budget determines how big the ranker can be, which shows I know the constraint drives the design. And I'd flag early that retrieval recall caps everything downstream.

**44. Search — first words?**
"BM25 first-stage + neural reranker. Index built offline. Query rewriting / expansion if recall is low."

> **Saying it out loud.** For search I'd start with BM25 as the first stage, because lexical retrieval is fast, cheap, and still a strong baseline that handles rare terms and exact matches better than embeddings do. Then a neural reranker — a cross-encoder — over the top few hundred. Hybrid retrieval, combining BM25 and dense vectors, is the modern default since they fail in different ways. And I'd mention query rewriting or expansion if recall is the bottleneck. The tradeoff to name is that the cross-encoder is where your latency budget goes, so its candidate count is your main dial.

**45. Fraud — first words?**
"Synchronous binary classifier with cost-asymmetric loss. GBDT for speed and interpretability. Threshold tuned on PR curve at desired false-positive rate."

> **Saying it out loud.** For fraud I'd open with: synchronous binary classifier in the authorization path, gradient-boosted trees for speed and interpretability, cost-asymmetric loss because a miss costs a hundred times a false alarm. The threshold comes off the PR curve at whatever false-positive rate the business will tolerate, not 0.5. Real-time velocity features do most of the work, which means a streaming aggregation layer is the hard part of the build. And daily retraining, because the adversary adapts.

**46. RAG — first words?**
"Index documents into embedding store. At query time: embed → retrieve top-K via ANN → optionally rerank → pass to LLM with prompt template. Monitor: faithfulness, citation quality."

> **Saying it out loud.** For RAG I'd say: chunk and embed documents offline into a vector index; at query time, embed the query, retrieve the top K, optionally rerank with a cross-encoder, then hand the passages to the LLM in a prompt that demands citations. The parts people underrate are chunking strategy and reranking, which usually move quality more than swapping the LLM does. And the metrics aren't accuracy — they're retrieval recall and faithfulness, meaning does the answer actually follow from the retrieved text. The main failure mode is confident answers grounded in nothing retrieved.

**47. Ad ranking — first words?**
"Two-stage with calibration. Retrieval over targeted ads → CTR + price model on top-K → auction (Google/Meta moved to first-price auctions for display ads ~2019–2021; second-price was the historical default). Calibration is critical for revenue."

> **Saying it out loud.** Ads is two-stage like recommendations, but with an auction bolted on the end and calibration as the make-or-break. You retrieve eligible ads by targeting, predict click and conversion probability on the top candidates, and rank by expected value — bid times predicted CTR. Calibration matters more here than almost anywhere, because the predicted probability is literally multiplied by money, so an uncalibrated model misprices every auction and burns revenue directly. And the whole thing is a marketplace, so you're balancing advertiser value, user experience and platform revenue at once.

---

## Quick fire

**48.** *First step in ML system design?* Clarify requirements.
**49.** *Two-stage retrieval order?* Retrieval → ranking.
**50.** *Imbalance metric?* PR-AUC + threshold tuning.
**51.** *Time-series split?* Time-based, never random.
**52.** *Cold start mitigation?* Popularity + content features + exploration.
**53.** *Serving latency tail?* p99 matters most.
**54.** *Drift detection?* Input distribution monitoring + PSI.
**55.** *Model rollback trigger?* Sustained metric regression.

---

## Self-grading

If you can't answer 1-15, you can't structure the answer. If you can't answer 16-30, you'll get tripped up on serving / monitoring questions. If you can't answer 31-45, frontier-lab system-design interviews will go past you.

Aim for 35+/55 cold.
