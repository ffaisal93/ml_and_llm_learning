# ML system design

The design round tests whether you can turn a vague ask into a specification with numbers, and then defend the numbers. Most candidates jump to model architecture in the first two minutes. That is the mistake. The metric, the scale estimate, and the data plan decide the design; the model is usually the least interesting part. Do arithmetic out loud, state assumptions before you use them, and say which requirement you would drop first when the budget is cut.

## The equations

**Queries per second from daily active users.**

$$\text{QPS}_{\text{avg}} = \frac{\text{DAU} \times \text{requests per user per day}}{86400}$$

DAU is daily active users and 86400 is seconds per day; this converts a product number into an infrastructure number in one line.

**Peak load.**

$$\text{QPS}_{\text{peak}} = \text{QPS}_{\text{avg}} \times k, \qquad k \in [2, 5]$$

$k$ is the peak-to-average ratio, driven by time zones and daily rhythm; you size hardware for peak, not average. Ten million DAU at five requests each gives 579 average and about 1736 at $k=3$.

**Latency budget as a sum.**

$$L_{\text{total}} = \sum_{i=1}^{N} L_i + L_{\text{network}}$$

$L_i$ is the latency of stage $i$; this only holds for stages that run in sequence. Stages that run in parallel do not add. Their combined latency is $\max_i L_i$, not the sum, plus fan-out and merge overhead, so parallelising is how you buy latency back.

**Fan-out tail identity.** If a request fans out to $n$ parallel calls and each is slow with probability $p$:

$$P(\max > t) = 1 - (1-p)^n$$

$t$ is the slowness threshold and the identity says the request is as slow as its slowest branch. At $p = 0.01$: $n = 10$ gives 9.6 percent, $n = 100$ gives 63.4 percent. So a one-percent tail becomes the common case under fan-out.

**Per-stage percentile for an end-to-end target.** For $N$ independent sequential stages and an end-to-end p99:

$$q = 0.99^{1/N}$$

$q$ is the per-stage success probability needed. For $N = 5$, $q = 0.99799$, so each stage needs its p99.8, not its p99. The tail budget per stage is $1 - q = 0.201$ percent.

**Storage for an embedding index.**

$$S = n \times d \times b \times o$$

$n$ is the number of vectors, $d$ the dimension, $b$ bytes per component, and $o$ the index structure overhead; one hundred million vectors at 768 dimensions in float32 is 307.2 GB raw, and about 460.8 GB at 1.5 times overhead.

**Quantisation saving.** Moving from float32 to int8 divides $b$ by four:

$$S_{\text{int8}} = S_{\text{fp32}} / 4$$

That turns 460.8 GB into about 115 GB, which is the difference between many machines and one.

**Cost per query.**

$$c_q = \frac{\text{tokens per query}}{1000} \times c_{1k}$$

$c_{1k}$ is the price per thousand tokens; at 1500 tokens and a rate of two tenths of a cent per thousand, that is three tenths of a cent per query, and at fifty million queries per day it is one hundred and fifty thousand dollars per day. The price used here is illustrative, not a vendor quote.

**Two-stage funnel cost.** With a corpus of $M$ items, a cheap retriever scoring all of them at cost $c_r$ and a ranker scoring only $K$ at cost $c_k$:

$$C = M c_r + K c_k$$

The funnel exists because $c_k \gg c_r$; scoring the whole corpus with the ranker costs $M c_k$, which is thousands of times more.

**Throughput and machines needed.**

$$m = \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS per machine}} \right\rceil \times (1 + \text{headroom})$$

Headroom of thirty to fifty percent covers deploys, failures, and estimation error.

**Effective QPS per machine from latency and concurrency.**

$$\text{QPS}_{\text{machine}} = \frac{\text{concurrency}}{L_{\text{service}}}$$

This is Little's law rearranged; a service at 50 ms with 8 concurrent slots handles 160 requests per second.

## Code from memory

Purpose: a back-of-envelope capacity calculator that prints traffic, storage, and cost.

```python
def capacity(dau, req_per_user_day, peak_factor, n_vectors, dim, bytes_per_dim,
             index_overhead, cost_per_1k_tokens_usd, tokens_per_query):
    # 1. traffic
    avg_qps = dau * req_per_user_day / 86_400
    peak_qps = avg_qps * peak_factor

    # 2. storage for the embedding index
    raw_gb = n_vectors * dim * bytes_per_dim / 1e9
    index_gb = raw_gb * index_overhead

    # 3. money, per query and per day at average load
    cost_q = tokens_per_query / 1000 * cost_per_1k_tokens_usd
    cost_day = cost_q * dau * req_per_user_day

    print(f"avg QPS        {avg_qps:,.0f}")
    print(f"peak QPS       {peak_qps:,.0f}")
    print(f"raw vectors    {raw_gb:,.1f} GB")
    print(f"index on disk  {index_gb:,.1f} GB  ({index_overhead}x overhead)")
    print(f"cost/query     {cost_q*100:.3f} US cents")
    print(f"cost/day       {cost_day:,.0f} US dollars")
    return avg_qps, peak_qps, index_gb, cost_q

capacity(dau=10e6, req_per_user_day=5, peak_factor=3,
         n_vectors=100e6, dim=768, bytes_per_dim=4, index_overhead=1.5,
         cost_per_1k_tokens_usd=0.002, tokens_per_query=1500)
```

Output: 579 average QPS, 1736 peak, 307.2 GB raw, 460.8 GB indexed, three tenths of a US cent per query, one hundred and fifty thousand US dollars per day. The token price is an illustrative input, not a quote.

Purpose: the per-stage percentile needed for an end-to-end target, and the fan-out tail identity.

```python
def per_stage_percentile(target_pct, n_stages):
    """If N stages are independent, each must exceed the target's Nth root."""
    q = target_pct ** (1.0 / n_stages)
    return q, 1 - q          # required per-stage success, and its tail mass

def fanout_tail(p, n):
    """P(at least one of n parallel calls is slow) when each is slow w.p. p."""
    return 1 - (1 - p) ** n

for n in (2, 5, 10):
    q, tail = per_stage_percentile(0.99, n)
    print(f"{n} stages -> each needs p{q*100:.3f} (tail {tail*100:.3f}%)")

print()
for n in (1, 10, 100):
    print(f"fan-out {n:>3} at p=1% slow -> P(max slow) = {fanout_tail(0.01, n)*100:.1f}%")
```

Output: two stages need p99.499 each, five stages need p99.799, ten stages need p99.900. Fan-out at a one percent slow rate gives 1.0 percent at $n=1$, 9.6 percent at $n=10$, and 63.4 percent at $n=100$. Both are one-line identities you should be able to reproduce on a whiteboard.

## Questions

### Q1. What framework do you use to answer any ML system design question?

Seven steps in order. First clarify: who the user is, what the product does, what counts as success, what the constraints are on latency, cost, and privacy. Second define metrics: one online business metric, one offline model metric that predicts it, and explicit guardrails such as latency and fairness. Third estimate scale: DAU, requests per user, QPS average and peak, corpus size, storage, and cost per query, using $\text{QPS} = \text{DAU} \times r / 86400$. Fourth design the data: sources, labels, how labels arrive and how late, and how you build training examples without leakage. Fifth choose the model, usually a simple baseline first and a stronger model only where the baseline is measurably short. Sixth design serving: online or batch, the funnel, caching, the latency budget. Seventh evaluate and iterate: offline gate, shadow, canary, rollback, monitoring. Spend the most time on steps one to three.

> **Say it.** Clarify, metrics, scale, data, model, serving, evaluate, iterate. I start by pinning down the user, the success definition, and the latency and cost constraints. Then I name one online business metric and one offline proxy, plus guardrails. Then I do the arithmetic: DAU times requests over eighty-six thousand four hundred for QPS, corpus size, storage, cost per query. Only then data, then model, then serving. I spend most of the time on the first three steps, because they decide everything downstream, and the model choice is usually the least interesting part.

### Q2. How do you turn a vague product ask into a specification with numbers?

Convert every adjective into a measurable quantity with a threshold. "Fast" becomes a p99 latency target, for example three hundred milliseconds end to end. "Relevant" becomes a metric with a number, such as recall at ten of at least eighty percent, or click-through rate up two percent. "Scalable" becomes peak QPS and a growth horizon. "Cheap" becomes a cost ceiling per query or per month. Then state assumptions explicitly and label them: DAU, requests per user, corpus size, item churn rate, peak-to-average factor. Then name what is out of scope, because a specification without exclusions expands forever. Finally write the acceptance test: the exact conditions under which you would call the system done. If the stakeholder cannot supply a number, propose one and ask them to reject it. A proposed number gets corrected; an open question gets ignored.

> **Say it.** I turn every adjective into a quantity with a threshold. Fast becomes a p99 target in milliseconds. Relevant becomes recall at ten above a stated value. Scalable becomes peak QPS with a growth horizon. Cheap becomes a cost ceiling per query. Then I write down my assumptions and label them as assumptions: daily actives, requests per user, corpus size, peak factor. Then I state what is out of scope, and I write the acceptance test. If the stakeholder will not give me a number, I propose one, because a proposal gets corrected and a question gets ignored.

### Q3. Why is retrieval and ranking a two-stage funnel?

Because the good scoring function is too expensive to run on the whole corpus. With $M$ items, cost is $M c_r + K c_k$ for a retriever at $c_r$ per item and a ranker at $c_k$ over only $K$ candidates. If $c_k$ is a thousand times $c_r$, scoring everything with the ranker is impossible within a latency budget of a few hundred milliseconds. So stage one, candidate generation, is cheap, high recall, and low precision: approximate nearest neighbour over embeddings, inverted-index lookup, popularity, and simple co-occurrence, often several sources merged. Its job is to get the right item somewhere in the top few hundred, so recall at $K$ is the metric. Stage two, ranking, is expensive, precise, and uses rich cross features and a heavy model over those few hundred. Its metric is NDCG or click-through. Often there is a third stage that re-ranks for diversity and business rules.

> **Say it.** Because the accurate scorer is too expensive to run over the whole corpus inside a few hundred milliseconds. Stage one is cheap and high recall: approximate nearest neighbour, inverted index, popularity, co-occurrence, usually several sources merged, cutting millions of items to a few hundred. Its metric is recall at K, because a missed item can never be recovered downstream. Stage two is expensive and precise, with rich cross features over those few hundred, scored by NDCG or click-through. A third pass often handles diversity and business rules.

### Q4. What is a feature store and what is training-serving skew?

A feature store is a system that computes feature values once and serves them to both training and inference, with a registry of definitions, an offline store for historical values used in training, and an online store for low-latency lookups at serving time. The online store is usually Redis, an in-memory key-value database that answers reads in about a millisecond. Training-serving skew is any difference between the feature values a model saw in training and the values it sees in production. It has three main causes. Implementation skew: the feature was computed by a Spark job for training and reimplemented in Python for serving, and the two disagree. Time-travel leakage: the training feature used data from after the prediction time, which serving cannot have. Distribution skew: the live inputs differ from the training snapshot. A feature store fixes the first two by having one definition and point-in-time correct joins.

> **Say it.** A feature store computes each feature once from one definition and serves it to both training and inference. It has a registry of definitions, an offline store of historical values for training, and an online store, usually Redis, an in-memory key-value database with sub-millisecond reads. Training-serving skew is any mismatch between what the model saw in training and what it sees live. Causes are reimplementing the feature twice, leaking future data through a bad join, and genuine distribution shift. The store fixes the first two by single definition and point-in-time correct joins.

### Q5. How do you handle cold start?

Split it into three cases and treat each separately. New user: you have no interaction history, so fall back to non-personalised signals, which are popularity, trending, and geography or device, then use whatever the user gives you at sign-up, then switch to personalisation as soon as a few interactions arrive. Explicit onboarding, such as picking three interests, buys a lot cheaply. New item: you have no interactions, so use content features, meaning the item's own text, image, and category embeddings, and give the item a small exploration budget of forced impressions so it can earn data. New system: no data at all, so start with rules and a content-based model, and instrument logging from day one so you can train the real model later. In all three, blend the cold signal with the warm one by confidence, weighting the personalised score more as interaction count grows.

> **Say it.** Three cases. New user: fall back to popularity, trending, and coarse context like geography, ask for a few interests at sign-up, and switch to personalised as soon as a handful of interactions arrive. New item: use content features from its text, image, and category, and give it a small forced-impression budget so it can earn data. New system: rules plus a content-based model, and log everything from day one so I can train later. In every case I blend cold and warm scores by confidence, weighting personalisation up as evidence accumulates.

### Q6. Online versus batch inference. How do you decide?

Decide on three things: whether the input is known in advance, how fresh the output must be, and how many predictions you need. Batch inference precomputes predictions on a schedule and stores them for lookup. Use it when the input set is enumerable, such as one prediction per user per day, and staleness of hours is acceptable. It is cheap, because you get GPU efficiency from large batches, and serving is a key-value read from Redis. Online inference computes at request time. Use it when the input depends on the immediate request, such as a search query or a fraud check on this transaction, or when freshness matters in seconds. It costs more and puts a model in your latency path. The common answer is a hybrid: precompute the expensive user and item embeddings in batch, and do only the light scoring or nearest-neighbour lookup online.

> **Say it.** Three questions. Is the input known in advance, how fresh must the answer be, and how many predictions do I need. If I can enumerate the inputs and hours of staleness is fine, I batch: precompute on a schedule, store the result, and serve it as a Redis lookup, which is cheap and takes the model out of the latency path. If the input arrives with the request, like a query or a card transaction, I go online. Usually I do both: heavy embeddings in batch, light scoring online.

### Q7. Sketch a recommender in five minutes.

Metric: online, session watch time or add-to-cart rate; offline, NDCG at ten as its proxy; guardrails on p99 latency and diversity. Scale: ten million DAU at five requests gives about 579 average and 1736 peak QPS. Architecture is the funnel. Candidate generation merges three sources: two-tower embedding retrieval with approximate nearest neighbour over a vector index, meaning a database of embeddings that returns the nearest to a query vector; item-to-item co-occurrence from recent behaviour; and popularity for cold users. That yields about five hundred candidates. Ranking is a gradient-boosted tree or a small neural network over user, item, and cross features, plus context. Then a re-rank pass for diversity and business rules. User and item embeddings are computed in batch nightly and cached; only the nearest-neighbour lookup and the ranker run online. Train on implicit feedback with sampled negatives, and correct for position bias.

> **Say it.** Metric first: watch time or add-to-cart online, NDCG at ten offline, with latency and diversity guardrails. Ten million actives at five requests is about six hundred average QPS, roughly seventeen hundred at peak. Then the funnel. Candidate generation merges two-tower nearest-neighbour retrieval over a vector index, item-to-item co-occurrence, and popularity for cold users, down to about five hundred. A boosted-tree ranker with cross features orders them, then a diversity re-rank. Embeddings batch nightly, only lookup and ranking online. Implicit feedback with sampled negatives, corrected for position bias.

### Q8. Sketch a search ranker in five minutes.

Metric: online, click-through at position one and session success rate, meaning the user did not reformulate; offline, NDCG at ten against graded relevance labels. Latency target is a p99 around two hundred milliseconds. Pipeline: query understanding first, which is spelling correction, intent classification, and entity extraction. Then retrieval as a hybrid of lexical BM25 over an inverted index and dense retrieval over a vector index, merged by reciprocal rank fusion, because lexical handles rare terms and exact identifiers while dense handles paraphrase. That gives a few hundred candidates. Then a learning-to-rank model, typically LambdaMART or a cross-encoder, trained on click logs with position-bias correction, using query-document features, document quality, and freshness. Then business rules and diversity. Cache aggressively, because query traffic is heavily skewed and the top queries repeat constantly, so a query result cache with a short time to live removes most of the load.

> **Say it.** Online metric is click-through at position one plus session success, meaning no reformulation; offline it is NDCG at ten on graded labels, with a p99 near two hundred milliseconds. Query understanding first: spelling, intent, entities. Then hybrid retrieval, BM25 over an inverted index plus dense retrieval over a vector index, fused by reciprocal rank, because lexical wins on rare terms and dense wins on paraphrase. Then a learning-to-rank model on click logs with position-bias correction. Then rules and diversity. And a query cache, because head traffic repeats.

### Q9. Sketch a fraud detector in five minutes.

The defining features are extreme class imbalance, an adversary, and asymmetric costs. Metric: not accuracy. Use precision-recall area under the curve, and operate at a threshold set by the cost ratio, for example recall at a fixed false-positive rate the review team can absorb. Scale is per transaction, so it must be online with a p99 under about one hundred milliseconds. Architecture: real-time features from a streaming aggregator, such as count and amount in the last minute, hour, and day per card, per device, and per merchant, served from Redis. Then a gradient-boosted tree, because tabular features dominate, plus rules for known patterns and a graph or anomaly component for rings. Two thresholds: block above the high one, send to human review between them, allow below. Labels are delayed by chargebacks, often sixty days or more, so hold out a recent window and monitor proxies. Retrain often, because the adversary adapts.

> **Say it.** Imbalance, an adversary, and asymmetric costs. So not accuracy: precision-recall AUC, with the operating threshold set by the cost of a missed fraud against the cost of a false decline and the review team's capacity. It must be online, under about a hundred milliseconds. Streaming aggregate features per card, device, and merchant over the last minute, hour, and day, served from Redis, into a boosted tree, plus rules and a graph component for rings. Two thresholds: block, review, allow. Chargeback labels come back in sixty days, so I monitor proxies and retrain frequently.

### Q10. Sketch a support copilot in five minutes.

Metric: online, resolution rate without escalation and handle time; offline, faithfulness to the retrieved sources and answer accuracy on a golden set, with a hard guardrail that it never invents policy. Architecture: retrieval-augmented generation over the help centre and past resolved tickets, chunked and embedded into a vector index, with hybrid lexical plus dense retrieval and a re-ranker. The generator is a mid-size model with a strict prompt: answer only from the retrieved passages, cite them, and say you do not know otherwise. Add tools for account lookups so live state is fetched, not recalled. Place a human gate on anything irreversible, such as refunds. Serving: streaming responses so perceived latency is low, plus a cache on repeated questions. Evaluate at retrieval and generation separately, log full traces, and convert every escalation into an eval case. Start assistive, with an agent approving each draft, then automate the safe classes.

> **Say it.** Resolution without escalation online, faithfulness and accuracy on a golden set offline, with a hard rule that it never invents policy. Retrieval-augmented generation over the help centre and past resolved tickets in a vector index, hybrid retrieval with a re-ranker, and a generator instructed to answer only from the passages, cite them, or say it does not know. Tools for live account state. Human gate on refunds and anything irreversible. Streaming output for perceived latency. I ship it assistive first, with agents approving drafts, then automate the safe classes.

### Q11. How do you monitor a model, and what is the difference between feature, label, and concept drift?

Monitor four layers. Operational: QPS, latency percentiles, error rate, saturation. Data: null rates, ranges, cardinality, and schema violations per feature. Model: prediction distribution, confidence, and feature importances. Business: the actual outcome metric. Collect the numbers with Prometheus, a time-series database that scrapes metrics from your services, and display them in Grafana, a dashboarding tool that queries Prometheus and holds the alert rules. The three drifts differ. Feature drift is $P(X)$ changing while the relationship holds, for example a new device type shifting an input distribution; detect with population stability index or a Kolmogorov-Smirnov test per feature. Label drift is $P(Y)$ changing, for example the fraud base rate rising. Concept drift is $P(Y \mid X)$ changing, meaning the same input now implies a different answer; it is the dangerous one because inputs look normal, and you only detect it once labels arrive.

> **Say it.** Four layers: operational, data quality, model behaviour, and business outcome. Prometheus scrapes the numbers as time series, Grafana draws them and holds the alerts. Feature drift is the input distribution moving while the relationship holds, caught with population stability index or a KS test per feature. Label drift is the outcome base rate moving. Concept drift is the mapping itself changing, so the same input now means something different. Concept drift is the dangerous one, because the inputs look completely normal and only labels reveal it.

### Q12. Explain shadow deployment, canary, and rollback.

A shadow deployment runs the new model on real production traffic in parallel with the current one, but its outputs are logged and discarded rather than served. It gives you real-distribution behaviour, latency, and error rates at zero user risk, and it is the only way to test infrastructure under real load before anyone sees the result. It cannot measure user reaction, because nobody sees the output. A canary sends a small fraction of real traffic, typically one to five percent, to the new model and serves its results, then compares metrics between the canary and control group. It does measure user reaction, at bounded blast radius, and you ramp up in stages, for example one, five, twenty-five, then a hundred percent, holding at each stage long enough for the metric to be readable. Rollback is the ability to return to the previous version in one action, which requires the old version still running, versioned artefacts, and automatic triggers on guardrail breach.

> **Say it.** Shadow runs the new model on real traffic in parallel and throws the output away. It gives me real-distribution behaviour, latency, and errors with zero user risk, but it cannot tell me how users react, because nobody sees it. A canary serves the new model to one to five percent of traffic and compares against control, so it does measure user reaction with a bounded blast radius, ramped in stages. Rollback is a one-action return to the previous version, which needs the old one still running, versioned artefacts, and automatic triggers on guardrail breach.

### Q13. How do you train and monitor when labels are delayed?

First quantify the delay and its shape: fraud chargebacks arrive over sixty to ninety days, subscription churn over a month, ad conversions over days. Then respect it in training. Only use a labelling window that is fully mature, which means your most recent data is unusable and your training set is always at least one delay period stale. Never join a label without a point-in-time correct cut, otherwise you leak. For evaluation, either wait for maturity or model the delay distribution and reweight the partially observed period. For monitoring you cannot wait, so use proxies: prediction distribution shift, feature drift, score calibration against the fraction of labels that have arrived, and fast human-labelled samples on a small stream. Run a continuous small review queue so you get some fresh ground truth every day. Accept the consequence: your retraining cadence cannot be faster than the delay unless you use proxy labels.

> **Say it.** First I quantify the delay and its shape, because chargebacks take sixty to ninety days and conversions take days. Then I only train on a fully mature window, which means my training data is always at least one delay period old, and I cut every label join point-in-time so I do not leak. For monitoring I cannot wait, so I use proxies: prediction distribution shift, feature drift, and calibration against the labels that have arrived so far, plus a small daily human review queue for fresh ground truth.

### Q14. What are feedback loops, and how do you handle them?

A feedback loop happens when the model's own outputs become its future training data. A recommender shows ten items, the user clicks one, and that click trains the next model. But the user never saw the items you did not show, so the model learns that what it already recommends is good and narrows over time. This is presentation bias, and it compounds. The consequences are popularity collapse, filter bubbles, and metrics that improve while the product gets worse. The controls are: log impressions, not just clicks, so you know what was shown and can compute a real denominator; use inverse propensity weighting, dividing each observed reward by the probability that item was shown, to debias training; reserve a small randomised exploration slice, typically one to five percent of traffic, which gives unbiased data; and monitor catalogue coverage and long-tail share as guardrails, not just click-through, because click-through rises as the system narrows.

> **Say it.** A feedback loop is when the model's outputs become its own training data. The recommender shows ten items, the user clicks one, that click trains the next model, and the items never shown never get a chance. So it narrows and looks better on click-through while getting worse. I control it by logging impressions and not just clicks, weighting training examples by inverse propensity so I correct for what was shown, holding out one to five percent of traffic for randomised exploration, and monitoring catalogue coverage and long-tail share as guardrails.

### Q15. The budget is cut by half. What do you simplify?

Cut in this order, and say why each is safe. First, model size: distil or replace the large model with a smaller one, and quantise. Quantising an embedding index from float32 to int8 divides storage by four, so 460.8 GB becomes about 115 GB. Second, cache harder: traffic is skewed, so a cache on the head of the distribution removes a large share of the compute for a small staleness cost. Third, move work from online to batch, because precomputing embeddings nightly and serving them as a Redis lookup is far cheaper than computing per request. Fourth, relax the latency target, since going from p99 at one hundred milliseconds to three hundred lets you batch requests on the GPU and raises throughput a lot. Fifth, cut candidate set size $K$ in the ranker, which trades a small quality loss for a linear cost saving. What I never cut is monitoring, logging, and the safety gates.

> **Say it.** In order: shrink the model by distillation and quantisation, since int8 cuts index storage by four. Then cache the head of the traffic, because the distribution is skewed and staleness there is cheap. Then move work from online to batch, precomputing embeddings and serving them as a key-value read. Then relax the latency target, because a looser p99 lets me batch on the GPU and raise throughput sharply. Then cut the ranker's candidate count, which is a linear saving for a small quality loss. I never cut monitoring, logging, or safety gates.

## Done when

- You can compute QPS from DAU, peak factor, embedding index storage, and cost per query on a whiteboard in under three minutes without a calculator.
- You can state $P(\max > t) = 1 - (1-p)^n$ and say from memory that a one percent slow rate becomes 63.4 percent at a fan-out of one hundred.
- You can derive $q = 0.99^{1/N}$ and say that five sequential stages each need their p99.8.
- You can sketch a recommender, a search ranker, a fraud detector, and a support copilot in five minutes each, leading with the metric and the scale estimate rather than the model.
