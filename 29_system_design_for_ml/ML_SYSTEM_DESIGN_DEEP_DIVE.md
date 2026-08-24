# ML System Design: The Complete Guide

## Part 1 — Foundations, Vocabulary, and Delivery

This part gives you three things. First, a picture of what the interview round actually is, so you know what you are being graded on. Second, the six-step framework and how to run it under time pressure. Third — and this is the long one — a from-zero explanation of every piece of production infrastructure that shows up in the worked designs in Parts 2 and 3, so that when you say "we'd put the features in Redis behind the ranker" you know exactly what you are claiming and can defend it when pushed.

The assumption behind this rewrite is that you are strong on the ML and thin on the production plumbing. That is the normal shape of a research background. The plumbing is not deep; it is just unfamiliar, and nobody ever sat you down and explained it. Section C does that.

---
## Section A. How to read this, and how the round works

### What the round actually is

An ML system design interview is a 45-minute conversation in which an interviewer says something deliberately under-specified — "design YouTube's recommender," "design fraud detection for a payment processor" — and then watches how you turn that sentence into a system. There is no correct answer being checked against a key. There is a person forming a judgment about whether they would want you owning a production system on their team.

That framing matters because it changes what you optimize for. You are not trying to produce the best possible recommender. You are trying to demonstrate that you can take an ambiguous business goal, convert it into a machine learning problem with defined inputs and outputs, choose an architecture whose components you can each justify, notice the ways it will break, and say all of that out loud in a structure someone can follow. A mediocre architecture explained with clear reasoning and honest trade-offs beats a state-of-the-art architecture recited without justification, every time.

### Where the 45 minutes goes

The clock is tighter than it sounds. Assume five minutes of introductions and logistics at the front and five minutes of your questions at the end, and you have roughly thirty-five minutes of actual design time. A well-run answer spends the first three to five minutes clarifying, two or three minutes stating the ML framing, five minutes on data, eight to ten on the architecture, five to eight on serving, and three to five on monitoring and iteration. That adds up to about thirty-five, which is the point: there is no slack. Every minute you spend on something the interviewer did not ask about is a minute you do not have for the part they wanted to probe.

The single most common time failure is going deep too early. A candidate hears "recommender," gets excited about the two-tower model, and spends twelve minutes on the loss function and negative sampling strategy before saying a word about how candidates are retrieved or what happens when the model is down. The interviewer now has no signal about anything except loss functions.

### What is being graded

Interviewers at large companies typically score on four or five axes, and although the wording differs, they come out roughly like this.

**Problem framing.** Did you convert a business sentence into a machine learning problem? Did you name the prediction target, the unit of prediction, the label source, and what "good" means numerically? Did you notice the ambiguity that mattered — for instance, that "recommend videos" could mean maximize clicks, watch time, or long-run retention, and that these three give you materially different systems?

**Architecture and trade-offs.** Can you propose a system, and — the part that separates levels — can you say what you gave up? Every architectural choice buys something and costs something. A candidate who says "I'd use approximate nearest neighbour search" is fine. A candidate who says "I'd use approximate nearest neighbour search, which means accepting maybe ninety-five percent recall against exact search in exchange for going from seconds to single-digit milliseconds, and I'd measure that recall explicitly against a brute-force baseline on a sample" is a level up.

**Production judgment.** Do you know that models decay, that features computed differently in training and serving silently destroy accuracy, that you need a rollback path, that a system with no monitoring is a system nobody can operate? This is the axis where research-heavy candidates lose points, and it is exactly the axis Section C is designed to fix.

**Communication.** Are you structured? Do you signpost — "I'm going to cover data, then architecture, then serving, then monitoring" — so the interviewer can follow? Do you check in rather than monologue for eleven minutes? Do you handle being redirected without visible irritation?

**Depth on demand.** When the interviewer picks one box on your diagram and asks you to go three levels down, is there anything there? You do not need depth everywhere. You need genuine depth somewhere, and the honesty to say where you do not have it.

### How to use Parts 2 and 3

Parts 2 and 3 contain the worked designs. Read this part first, all of it, including the vocabulary section even where it feels basic — the worked designs assume every term in Section C and will not re-define them. Then work each design in Parts 2 and 3 twice: once reading, once out loud on a whiteboard with a timer, without looking.

---

## Section B. The six-step framework

Every answer in Parts 2 and 3 follows the same six steps in the same order. The order is not arbitrary. Each step produces a decision that the next step depends on, so running them out of order means guessing at inputs you could have simply asked for.

Here is the shape, and then the explanation of each.

| Step | What you produce | Rough time |
|---|---|---|
| 1. Clarify | Scope, scale, latency budget, success metric | 3-5 min |
| 2. Frame as an ML problem | Inputs, output, label, loss, training and serving regime | 2-3 min |
| 3. Data | Sources, label generation, freshness, leakage risks | 5 min |
| 4. Architecture | The boxes and arrows, model choices, why each | 8-10 min |
| 5. Evaluation | Offline metrics, online metrics, guardrails, the launch decision | 5 min |
| 6. Production concerns | Serving, deployment, monitoring, retraining, failure response | 5-8 min |

### Step 1 — Clarify

**What you are doing.** Converting an underspecified prompt into a bounded problem by asking a small number of questions whose answers change your design. Not questions in general — questions with design consequences.

**Why it is first.** Because almost every later decision is downstream of the answers. Whether you need approximate nearest neighbour retrieval depends on catalog size. Whether features can be computed at request time depends on the latency budget. Whether you can use a large cross-encoder model depends on both. If you skip this step you will make those choices by assumption, and if your assumption is wrong the interviewer has to either correct you — burning time — or watch you build the wrong system.

**How long.** Three to five minutes, and no more. Five sharp questions, not fifteen.

**What going wrong looks like.** Two failure modes. The first is skipping it entirely because the problem "seems clear," which reads as someone who has never been burned by a product spec. The second, less obvious, is asking a long list of generic questions that do not change anything — "what's the tech stack?", "is there an existing system?" — which reads as someone performing the ritual of clarification without understanding its purpose. If you cannot say what your design would do differently under each possible answer, do not ask the question.

**The four things worth asking about.** Goal, scale, constraints, and data. Under goal: what business outcome are we optimizing, and what is the relative cost of a false positive versus a false negative? Under scale: how many requests per second at peak, how large is the catalog or user base, and what is the end-to-end latency budget? Under constraints: real-time or batch, personalized or not, any regulatory or fairness requirements, and what the cold-start situation is for new users and new items? Under data: what is logged today, and is the label cheap or expensive to obtain?

#### Worked micro-example: clarifying "design a system to detect fraudulent transactions"

This is what the first four minutes sound like. Note that each question is followed, in your head, by a branch.

Suppose the interviewer says: authorization-path card fraud, real time.

Suppose: about 100 milliseconds, p99.

Suppose: a false decline costs roughly ten times more than the average fraud loss, because customers who get declined stop using the card.

Suppose: ten thousand transactions per second at peak; chargebacks arrive with a lag of thirty to ninety days.

That is the whole skill. Five questions, each with a stated consequence, ending with an explicit handoff to the next step. Notice that the last answer surfaced the hard part of the problem — delayed labels — which you can now spend real time on, having earned it.

### Step 2 — Frame as an ML problem

**What you are doing.** Stating, in three or four sentences, the exact prediction problem: what the model takes in, what it emits, what the training label is and where it comes from, what loss you train against, and whether inference happens in a batch job or on the request path.

**Why it comes here.** It is the contract for everything downstream. Data collection only makes sense once you know what the label is. Architecture only makes sense once you know the output space. And stating the framing forces you to confront a gap you might otherwise paper over — most commonly, that the thing you want to optimize is not the thing you can label. You want long-term user satisfaction; you have clicks. Naming that gap out loud is a strong senior signal.

**How long.** Two to three minutes. It is short because it is a statement, not an exploration.

**What going wrong looks like.** Vagueness. "It's a classification problem" is not a framing. A framing is: "For a given (user, candidate video, context) triple at request time, predict the probability that the user watches more than ten seconds, trained with binary cross-entropy on impression logs where a positive is a watch over ten seconds and a negative is an impression that was shown and not watched — with the caveat that this label is confounded by position, which I'll come back to." The second failure is a framing that quietly requires information you will not have at serving time, which is the setup for the leakage problem in step 3.

### Step 3 — Data

**What you are doing.** Naming your data sources, explaining how labels are actually produced, stating how fresh each signal is, and calling out leakage risks.

**Why it comes before the architecture.** Because data availability constrains architecture far more than the reverse. If your only labels are implicit engagement logs, a supervised ranker trained on them will learn the biases of whatever system generated those logs, and you need to say how you will handle that before you draw the model. If a signal is only available with a day's delay, you cannot use it as a real-time feature, no matter how predictive it is.

**How long.** About five minutes.

**What going wrong looks like.** The classic is label leakage: using a feature at training time that is computed after the outcome you are predicting, or that encodes it. If you train fraud detection with a feature like "number of chargebacks on this card," and chargebacks are exactly what defines your label, your offline AUC will be spectacular and your production model will be useless. The general test is: at the moment of prediction in production, would this feature's value be known? If you cannot answer that for every feature, you have not designed the data layer, you have listed columns.

The other failure is not saying where labels come from. Every supervised system has a label-generation mechanism — user behavior logs, human annotation, downstream business outcomes, or a heuristic — and each has a bias and a cost. Naming it is not optional.

### Step 4 — Architecture

**What you are doing.** Drawing the system: what components exist, what flows between them, which models sit where, and what each stage's job is. This is where the patterns in Section D get used.

**Why here.** With the goal, scale, latency budget, and data established, the architecture is now mostly determined. That is the pleasant part: if you did steps one through three well, this step feels like reading off consequences rather than inventing.

**How long.** Eight to ten minutes, the largest block.

**What going wrong looks like.** Two things. First, drawing components you cannot defend — putting a box labeled "Kafka" on the diagram because it appears in these diagrams, without being able to say what it buys you over a direct service call. Interviewers probe boxes, so never draw one you cannot explain for ninety seconds. Second, presenting one architecture as though it were the only one. State the alternative and why you rejected it: "I'm using a two-tower retrieval model here rather than an inverted-index keyword approach because the queries are behavioral rather than lexical, but if the catalog were small enough — say under a hundred thousand items — I'd skip retrieval entirely and just rank everything, which removes a whole component."

### Step 5 — Evaluation

**What you are doing.** Saying how you know the system works, offline and online, and what decides a launch.

**Why it comes after architecture and before production concerns.** Because it is the bridge. Offline evaluation is what gates a model going anywhere near traffic; online evaluation is what gates it staying there. Candidates who fold evaluation into "monitoring" at the very end always run out of time and give it thirty seconds, which is a shame because it is a high-signal topic and the one where a research background is a genuine advantage — you already understand statistical inference better than most.

**How long.** About five minutes.

**What going wrong looks like.** Naming only offline metrics. Saying "I'd track AUC" and stopping is a junior answer. The senior answer connects three layers: an offline metric that you can compute cheaply on held-out data, an online metric that reflects the business goal, and a set of guardrail metrics that must not degrade even if the primary metric improves. And the honest acknowledgment that offline and online metrics disagree a substantial fraction of the time, which is why nothing launches on offline numbers alone. Section C's evaluation subsection covers all of this in depth.

### Step 6 — Production concerns

**What you are doing.** Serving mechanics, latency budget arithmetic, deployment strategy, monitoring, retraining cadence, and what happens when something breaks.

**Why last.** It depends on everything above, and it is the step most likely to be cut short by the clock — so if you have been disciplined about time, arriving here with five minutes left is the plan, not a failure.

**How long.** Five to eight minutes.

**What going wrong looks like.** Not budgeting latency numerically. If the requirement is 100ms at p99 — the ninety-ninth percentile, meaning ninety-nine out of a hundred requests are faster than this — then you should be able to say where it goes: perhaps 5ms of network, 10ms for a feature lookup, 15ms for retrieval, 40ms for ranking, 10ms for business logic, leaving 20ms of headroom. That arithmetic, even with made-up numbers you flag as estimates, is one of the strongest seniority signals available, because it is the thing people only do after they have missed a latency target in production.

The other failure is having no answer for "what happens when the model service is down." Every production ML system needs a degraded mode: serve a cached prediction, fall back to a simpler model, fall back to a non-ML heuristic like popularity ranking, or fail open versus fail closed depending on the cost asymmetry you established in step 1.

---

## Section C. Production infrastructure reference

Name the property before the product. For example, say that you need a key-value lookup below 10 ms before you choose Redis. This makes the reason for the component clear.

### Storage and serving

An in-memory key-value store keeps hot features and cached predictions in RAM. It gives low latency, but RAM is expensive. A TTL removes stale values. Cache misses use a slower store, so they can dominate p99 latency.

A relational database stores source-of-truth records with transactions and constraints. A distributed key-value store scales keyed reads and writes across machines, but it gives up flexible joins and can use eventual consistency. Object storage keeps raw events, images, Parquet datasets, and model files at low cost. A warehouse runs large analytical scans and joins. Neither object storage nor a warehouse is suitable for request-time lookup.

A vector index supports approximate nearest-neighbor, or ANN, search. Exact search over $N$ vectors of dimension $d$ costs $O(Nd)$ per query. ANN reduces this work but can miss true neighbors. HNSW gives high recall and supports inserts, but uses more memory. IVF searches selected clusters. Product quantization compresses vectors. IVF-PQ uses less memory, but usually loses more recall. Measure latency, memory, update time, and ANN recall against exact search.

### Data movement and ML components

Kafka or another event log stores ordered event streams. Producers append events. Consumers read and replay them independently. Consumers must be idempotent because at-least-once delivery can repeat an event. A stream processor maintains recent state, such as velocity features and rolling counts. A batch orchestrator runs a DAG of validation, feature, training, and evaluation jobs. Change data capture emits database updates so indexes and caches stay synchronized.

A feature store provides historical feature values for training and current values for serving. It reduces training-serving skew, but the historical join must still be point-in-time correct. An experiment tracker records code, data, parameters, and metrics. A model registry stores versioned artifacts. A model server loads models and can batch requests. Batching improves accelerator throughput, but adds queue delay.

### Reliability and deployment

Metrics show numerical trends. Logs explain events. Traces follow one request across services. Monitor service health, data freshness, model scores, labels, and product outcomes. A service can return HTTP 200 while stale features make every prediction wrong.

An SLI is a measured property. An SLO is its target. A 99.9% SLO over 28 days gives an error budget of about 40 minutes:

$$
28 \times 24 \times 60 \times 0.001 \approx 40.3 \text{ minutes}.
$$

Use tests and offline evaluation before deployment. Shadow traffic measures production behavior without changing results. A canary sends a small fraction of traffic to the new version. An A/B test measures product impact. Keep the previous model and configuration ready for rollback.

## Section D. The architecture menu

Production ML systems are assembled from a small number of recurring patterns. Once you recognize them, most design questions become a matter of choosing which patterns apply and how they compose, rather than inventing something. Each pattern below is presented as a problem and its resolution, because that is how you should present it in an interview — the pattern name alone means nothing until you have said what it fixes.

### D.1 Two-stage retrieval and ranking

**The problem.** You have a catalog of $N$ items and a request. You want to return the ten best. The scoring function you would like to use is expensive — a neural network that attends jointly over the user's history and the item's content, costing perhaps a millisecond per item. With $N = 10^7$ and a 100ms budget, scoring everything is off by four orders of magnitude. Meanwhile, a scoring function cheap enough to run over everything is too crude to order the top ten well.

**The resolution.** Split the problem by the observation that the two ends of the ranked list need different things. Getting from ten million candidates down to a few hundred requires only that you not lose the good ones — a **recall** problem, where mistakes at the bottom of the list are free. Getting from a few hundred to the final ten requires precise ordering — a **precision** problem, where every position matters. So use a cheap high-recall method for the first reduction and an expensive high-precision method for the second.

Stage one, **retrieval**, produces a few hundred to a few thousand candidates using methods that are sublinear or cheap-linear in catalog size: approximate nearest neighbour over embeddings, an inverted keyword index, precomputed item-to-item similarity lists, or simple rules like "items from creators this user follows." It is normal and correct to merge several such sources, since each has a different failure mode and the union has better recall than any one. Stage two, **ranking**, scores those few hundred with a model that can afford to be a hundred times more expensive per item because it sees a thousandth as many items. A third stage, **reranking**, often follows, applying business logic that is not naturally expressible as a per-item score: diversity, so the top ten are not ten near-identical items; freshness boosts; deduplication; policy filters; and exploration, deliberately showing some uncertain items to gather data.

**The properties to state.** The dominant failure mode is that stage one's recall bounds the whole system — an item that retrieval never surfaces cannot be ranked, so no ranker improvement can recover it, and the correct diagnostic is to measure retrieval recall against the set of items the ranker would have placed in the top ten had it seen everything. The second property is that the stages are trained on different distributions: the ranker only ever sees candidates that retrieval produced, so if you swap retrieval you have changed the ranker's input distribution and it needs retraining. The number of candidates passed between stages is the system's main latency-versus-quality dial, and being able to say "I'd start at 500 and tune it by measuring the quality curve against latency" is exactly the right register.

**When it is wrong.** When the catalog is small enough to score exhaustively — under roughly ten thousand items with a cheap model — where a single stage is simpler, has no recall ceiling, and is easier to reason about.

### D.2 Embedding retrieval and the two-tower pattern

**The problem.** You want retrieval to be semantic rather than lexical: to find items relevant to a user or query even when they share no words. And you want it fast over a huge catalog.

**The resolution.** Learn a function that maps queries (or users) and items into a shared vector space where geometric closeness means relevance, then reduce retrieval to nearest-neighbour search. The critical structural constraint is that the query representation and the item representation must be computed **independently** — this is the "two-tower" shape, one encoder per side, with the only interaction being a dot product or cosine similarity at the very end. That constraint is what makes the pattern work operationally: because the item tower does not depend on the query, every item embedding can be computed offline in a batch job and loaded into an ANN index, leaving only one small encoder forward pass at request time.

Contrast this with a **cross-encoder**, which feeds the query and item together into one model so that every layer can attend across both. Cross-encoders are consistently more accurate — the interaction is where the signal is — and are consistently unusable for retrieval, because there is nothing to precompute: you would need one forward pass per candidate. Hence the natural pairing: two-tower for retrieval, cross-encoder for ranking. Explaining that trade-off crisply is one of the highest-value ninety seconds in a search or recommendation interview.

**The properties to state.** Training these models is dominated by **negative sampling**: positives come from observed interactions, and negatives must be constructed. In-batch negatives (treating other items in the batch as negatives) are nearly free and are the standard starting point, but they are easy negatives, so the model learns to distinguish relevant from random rather than relevant from plausible. Hard negative mining — retrieving with the current model and using its high-scoring wrong answers as negatives — is what actually improves top-of-list quality, and is the thing to mention. Second, index freshness: a new item has no embedding until the batch job runs, which is a cold-start problem with a concrete mechanical cause, and the fix is an incremental path where new items get embedded and inserted within minutes. Third, an embedding index built from model version $v$ is only valid for query encoders of version $v$; changing the model means rebuilding the entire index, which for a billion items is a multi-hour job that must be coordinated with the deployment — a genuinely good operational detail to raise.

**When it is wrong.** When matching is genuinely lexical. Exact keyword search still beats dense retrieval for rare terms, product identifiers, names, and codes, because embeddings compress away exactly the rare specifics. The mature answer is **hybrid retrieval**: run both a keyword index and a vector index and fuse the results, which is what most serious production search systems actually do.

### D.3 The cascade

**The problem.** Your best model is expensive, and you cannot afford to run it on everything. But most inputs are easy, and a cheap model handles them perfectly well.

**The resolution.** Arrange models in increasing order of cost and accuracy, and let each stage resolve what it can confidently and escalate only what it cannot. A content moderation pipeline is the canonical example: an exact-hash lookup against known violating content is essentially free and catches a meaningful fraction; a small classifier handles the clear cases; a large multimodal model handles what the small one is uncertain about; and human reviewers see only the residue. If 90% of traffic is resolved at a stage costing a thousandth of the next, the average cost collapses while the accuracy on hard cases is preserved.

This differs from two-stage retrieval in an important way that is worth being precise about. In retrieval-and-ranking, every request passes through every stage and the stages reduce the *number of items*. In a cascade, stages reduce the *number of requests* by terminating early. They compose — a ranking system can itself be one stage of a cascade — but they are not the same idea.

**The properties to state.** The escalation rule is the design. It is usually a confidence threshold, which requires the early model's confidence to be **calibrated** — meaning that among the cases where it outputs 0.9, about 90% really are positive — otherwise the threshold is arbitrary and you are escalating the wrong things. Temperature scaling or isotonic regression on a validation set is the standard fix, and mentioning it shows depth. The second property is that each stage's error compounds: anything the cheap stage wrongly resolves never reaches the expensive one, so the cheap stage's *false confidence* rate, not its accuracy, is the number that matters. Third, latency becomes bimodal — fast for the common path, slow for escalations — so p50 and p99 tell very different stories and you should quote both.

**When it is wrong.** When the expensive model is affordable on everything, since a cascade adds real complexity. And when errors are asymmetric enough that the cheap stage's mistakes are unacceptable at any rate, in which case run the expensive model on everything and pay.

### D.4 Batch, streaming, and on-demand feature computation

**The problem.** A feature has to be computed at some point between the event that generates it and the prediction that uses it. Where you put that computation determines freshness, cost, and complexity, and the three options are genuinely different systems.

**The resolution — three modes, chosen per feature, not per system.**

**Batch** computation runs on a schedule, typically hourly or nightly, over the warehouse, and writes results to the online store. It is by far the cheapest per feature because it amortizes over huge row counts, it is the easiest to test and backfill, and the computation can be arbitrarily complex since nothing is waiting on it. Its cost is staleness bounded by the schedule interval. This is right for features that change slowly: a user's long-run topic affinities, an item's lifetime statistics, aggregate popularity.

**Streaming** computation maintains features continuously from an event stream, updating the online store within seconds. It is more expensive and much harder to operate and backfill, and it buys you freshness measured in seconds. This is right for features where recency is the signal itself: transactions on this card in the last five minutes, items viewed in this session, current trending rate.

**On-demand** computation happens at request time from data in the request. It is trivially fresh because it has not been stored at all, it costs latency directly out of your budget, and it is right for features that only exist at request time — the query text, the device, the time of day, and anything derived from combining the request with a stored value, such as the distance between the request's location and the user's stored home location.

**The properties to state.** Real systems use all three simultaneously, and saying that is the answer. The key insight to volunteer is that freshness is a *per-feature* decision with a cost, so the design move is to identify which few features actually need second-level freshness and pay for streaming only on those. There is also a specific hazard: on-demand and precomputed versions of the same feature will drift apart unless they share a definition, which is exactly the training-serving skew problem, and is why the transformation logic must be shared code rather than parallel implementations.

**When each is wrong.** Batch is wrong when the signal decays in minutes — a fraud velocity feature computed nightly is worthless, since the attack completes in an hour. Streaming is wrong when hours of staleness would have been fine, because you are paying substantial operational cost for freshness nobody uses. On-demand is wrong when the computation touches data that is expensive to fetch, since you have moved a batch job onto the critical path.

### D.5 The read path and the write path

**The problem.** The same system has to do two things with opposite requirements. It must absorb a very high volume of incoming events durably and without ever blocking the user, and it must answer queries with very low latency. Optimizing storage for one hurts the other: a layout good for fast writes is usually bad for fast reads, and indexes that make reads fast make writes slower.

**The resolution.** Stop treating it as one system. Design the **write path** and the **read path** separately, connected by asynchronous processing, and accept a bounded delay between them.

The write path's job is to accept events and never lose them. It should do as little work as possible: append the event to a durable log and return. No joins, no aggregation, no index updates on the critical path. Its requirements are throughput and durability, and its latency is measured in low single-digit milliseconds.

The read path's job is to answer at request time from data structures already shaped for the query. It should do no computation that could have been done earlier: features precomputed and stored by key, embeddings already indexed, candidate lists already materialized. Its requirement is p99 latency.

Between them sit the asynchronous processors — stream jobs and batch jobs — that consume the log and build the read path's data structures. The delay across that gap is the freshness lag, and naming it as an explicit number that you are choosing is a strong signal.

**The properties to state.** This is the same idea as CQRS (command-query responsibility segregation) and event sourcing in general systems design, and the same idea as the Lambda and Kappa architectures in data engineering; naming the family is worth a point but explaining the separation is worth more. The main consequence is that the read path serves data that is slightly stale by construction, so the design question becomes "how stale, and does the product tolerate it?" — a question you should ask rather than assume. The second consequence is that a failure in the asynchronous layer is invisible to users at first: everything responds fine, it is just increasingly wrong. That is precisely why pipeline freshness needs its own monitoring and alerting, independent of service health, and it is the single most commonly missed monitoring item in interview answers.

**When it is wrong.** When you truly need read-your-own-writes semantics — a user updates something and must see it immediately — which requires either a synchronous update or a read-through cache invalidation on the write path. And for small systems, where one database serving both paths is entirely adequate and splitting them is unnecessary machinery.

### D.6 A note on composing them

A typical design uses most of these at once, and it helps to see how they nest. The write path carries interactions into a log. Batch and streaming jobs consume it to maintain features, with the freshness mode chosen per feature. Those features feed a two-tower model whose item embeddings are indexed for approximate nearest neighbour search. At request time, the read path does an on-demand feature computation, a feature store lookup, a retrieval step over the index merged with a couple of other candidate sources, a ranking pass over the merged candidates, and a business rerank. The whole thing is deployed behind a feature flag, shadowed then canaried then A/B tested, gated offline by a regression suite, and watched with metrics, traces, and prediction logs.

That paragraph is essentially the skeleton of every worked design in Parts 2 and 3. What changes between them is which pattern carries the weight and where the hard problem lives.

---

## Design 1 — YouTube's home-feed recommender

### Requirements and assumptions

I will design the logged-in home feed. I will optimize expected watch time per impression. However, watch time is only a proxy for satisfaction. Therefore, I will use day-7 and day-28 retention as guardrails.

I will assume 2.5 billion monthly users, one billion feed requests per day, and one billion available videos. The service must return 50 ranked videos within 150 ms at p99. The client displays about 20 videos above the fold. New uploads must become reachable within one hour. Safety and policy rules are hard filters.

**Functional requirements.** The system must generate a personalized home feed, retrieve candidates from several sources, rank them, apply safety and diversity rules, support new users and videos, log every decision, and return 50 playable videos. It must also support model experiments and fast rollback.

**Non-functional requirements.** The recommendation call must stay below 150 ms at p99 and support about 35,000 QPS at peak. New videos must enter retrieval within one hour. The service must remain available when a feature, retrieval, or ranking dependency fails. Training data must be point-in-time correct. User data must follow privacy and retention rules.

### Step 2 — Frame it as an ML problem

The prediction unit is one `(user, video, context)` impression. Context includes the device, locale, time, network quality, and recent session activity.

For each candidate, the ranker predicts:

$$
P(\text{click}),\quad \mathbb{E}[\text{watch time}],\quad
P(\text{like}),\quad P(\text{hide}).
$$

A possible final score is:

$$
\text{score}
=
P(\text{click})^\alpha
\cdot \mathbb{E}[\text{watch time}]^\beta
\cdot \left(1+\gamma P(\text{like})\right)
\cdot \left(1-\delta P(\text{hide})\right).
$$

The weights $\alpha$, $\beta$, $\gamma$, and $\delta$ represent product priorities. I would tune them with online experiments.

YouTube's published 2016 approach used weighted logistic regression. Positive impressions were weighted by observed watch time. Negative impressions had weight one. Under this training method, the odds approximate expected watch time:

$$
\mathbb{E}[\text{watch time}] \approx \frac{p}{1-p}=e^{Wx+b}.
$$

This method lets a classification model produce a regression-like watch-time estimate.

The labels have three main problems. First, watch time is an implicit and noisy satisfaction signal. Second, outcomes exist only for videos that the current system showed. This creates a feedback loop. Third, higher positions receive more clicks even when relevance is unchanged. This creates position bias. Therefore, the system needs exploration, position-bias correction, and long-term guardrails.

### Step 3 — Data and features

User features include long-term interests, the last 20–50 watched videos, language, country, device type, and account age. I would refresh long-term interests daily. I would update recent watch history within seconds because it represents current intent.

Video features include channel, age, duration, language, topic, title, description, thumbnail, sampled frames, and speech transcript. Content features are critical for new videos because new videos have no engagement history.

Cross features combine user and video information. Examples include the number of videos this user watched from this channel and the cosine similarity between the user-interest vector and the video vector. These features depend on both sides. Therefore, I compute them only after candidate retrieval.

Every training feature must be point-in-time correct. For an impression at time $t$, the training row can use only information available before $t$. Otherwise, future engagement leaks into training. I would also split training and evaluation data by time.

### Step 4 — Architecture

The system uses retrieval followed by ranking. Retrieval reduces one billion videos to about 1,000–1,500 candidates. Ranking applies a more expensive model to those candidates. A final reranker constructs the list.

The main candidate source is a two-tower model. The user tower converts the user and context into a 256-dimensional vector. The video tower converts each video into a vector in the same space. The affinity score is:

$$
s(u,v)=e_u^\top e_v.
$$

The video vector does not depend on the user. Therefore, I compute video vectors offline and store them in an approximate nearest-neighbor, or ANN, index. At request time, I run the user tower once and search the index.

I would train the two-tower model with in-batch negatives and sampled softmax. In a batch of $B=8192$ positive user-video pairs, each user receives one positive and up to 8,191 negatives without extra video-encoder calls. I would apply logQ correction because popular videos occur more often in the batch. I would also add hard negatives, such as videos that were shown but not clicked and high-scoring videos that the user did not watch.

At one billion videos, the ANN index probably needs IVF-PQ or a hybrid design. One billion 256-dimensional FP16 vectors require:

$$
10^9 \times 256 \times 2 = 512\text{ GB}.
$$

An HNSW graph with 32 four-byte neighbor IDs per item adds:

$$
10^9 \times 32 \times 4 = 128\text{ GB}.
$$

The total is about 640 GB before replication. Product quantization can reduce each vector to about 64 bytes. This reduces raw vector storage to about 64 GB, but it lowers ANN recall.

I would not depend on one retrieval source. I would also retrieve videos from subscribed channels, regional trending lists, co-watch neighbors, and a fresh-content exploration source. These sources run in parallel. I would merge and deduplicate their results.

After retrieval, the system fetches video and cross features for about 1,000 candidates. The ranker can use a DLRM or DCN architecture with Multi-gate Mixture-of-Experts, or MMoE, task heads. It predicts click, watch time, watch-through, like, and hide. A shallow bias tower can absorb position effects during training. I would remove that tower, or set position to a constant, during serving.

The final reranker limits repeated channels and topics. It also applies safety rules, freshness, already-watched filtering, and exploration. Exploration is required because the model cannot learn about videos it never shows.

#### Online and offline architecture

```text
 Home-feed request
 (user_id, device, locale, timestamp)
                    |
                    v
 +-------------------------------------+
 | API GATEWAY / FEED SERVICE          |
 | Authentication and request parsing  |
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | USER FEATURE SERVICE                |  ~5 ms
 | Long-term interests                 |
 | Last 20-50 watched videos           |
 | Language, country, device, session  |
 +------------------+------------------+
                    |
                    | user vector + context
                    v
 +-------------------------------------------------------------+
 |              PARALLEL CANDIDATE RETRIEVAL                   |
 |                                                             |
 | +----------------+ +--------------+ +--------------------+  |
 | | TWO-TOWER ANN  | |SUBSCRIPTIONS | | REGIONAL TRENDING  |  |
 | | user tower     | |new videos    | |fast-growing videos |  |
 | | -> 256-D vector| |from followed | |by language/region  |  |
 | | -> HNSW/IVF-PQ | |channels      | |                    |  |
 | | ~500 candidates| | ~200         | | ~100               |  |
 | +----------------+ +--------------+ +--------------------+  |
 |                                                             |
 | +----------------+ +--------------------------------------+ |
 | | CO-WATCH / CF  | | FRESH-CONTENT EXPLORATION            | |
 | | item neighbors | | new videos with little history       | |
 | | ~200 candidates| | ~100 candidates                      | |
 | +----------------+ +--------------------------------------+ |
 +---------------------------+---------------------------------+
                             |
                             | ~1,100 candidates
                             v
 +-------------------------------------+
 | MERGE, DEDUPLICATE, HARD FILTERS    |  ~3 ms
 | Policy, region, availability        |
 | Remove seen and duplicate videos    |
 | Output: ~1,000 candidates           |
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | CANDIDATE FEATURE HYDRATION         |  ~15 ms
 | Video features and cross features   |
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | MULTI-TASK RANKER                   |  ~50 ms
 | DLRM/DCN + MMoE                     |
 | P(click), E[watch time], P(like),   |
 | P(watch-through), P(hide)           |
 | Training-only position-bias tower   |
 +------------------+------------------+
                    |
                    | top ~200
                    v
 +-------------------------------------+
 | LIST-LEVEL RERANKER                 |  ~5 ms
 | Diversity, freshness, policy,       |
 | exploration, near-duplicate removal |
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | RESPONSE: 50 VIDEOS                 |  ~5 ms
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | EVENT LOG                           |
 | Impressions, positions, scores,     |
 | feature snapshots, actions, labels  |
 +------------------+------------------+
                    |
                    v
 +-------------------------------------+
 | OFFLINE TRAINING PIPELINE           |
 | Point-in-time features              |
 | Train retriever and ranker          |
 | Build versioned ANN index           |
 | Evaluate -> registry -> deployment  |
 +-------------------------------------+
```

The event log must store feature values as they existed at request time. It must also store model versions, scores, positions, and outcomes. Otherwise, later training data can disagree with the production decision.

### Step 5 — Evaluation

I would evaluate retrieval and ranking separately. For retrieval, I would use recall@1000. It measures how often the watched video appears in the candidate set. If recall@1000 is 60%, retrieval has already lost 40% of observed positive outcomes. The ranker cannot recover them.

For ranking, I would use per-user AUC, NDCG, log loss, and calibration. Log loss matters because the final score combines probabilities and expected values.

Offline metrics are release gates, not launch decisions. The data came from the current policy, so it favors models that behave like the current model.

The launch decision requires an A/B test randomized by user. The primary metric is watch time per user per day. Guardrails include day-7 and day-28 retention, dislikes, “not interested” actions, satisfaction surveys, content diversity, and creator exposure. I would run the test for at least two weeks. I would also keep about 0.5% of users in a long-term holdback to detect slow retention damage.

### Step 6 — Production concerns

One billion daily requests gives:

$$
\frac{10^9}{86{,}400}\approx 11{,}600 \text{ average QPS}.
$$

With a $3\times$ peak factor, I would provision for about 35,000 QPS. If each request ranks 1,000 candidates, the ranker performs:

$$
35{,}000 \times 1{,}000 = 35\text{ million item scores per second}.
$$

Therefore, the ranker cannot be a large transformer.

The p99 latency budget is:

```text
Request handling                 10 ms
User-feature lookup               5 ms
Parallel candidate retrieval     30 ms
Merge and hard filters            3 ms
Candidate feature hydration      15 ms
Multi-task ranking               50 ms
List-level reranking              5 ms
Response serialization            5 ms
                               -------
Total                           123 ms
Headroom                         27 ms
                               -------
p99 target                      150 ms
```

Each stage needs a fallback. If ANN fails, use subscriptions, trending, fresh content, and co-watch candidates. If user features fail, use a locale-level default profile. If ranking times out, sort by retrieval score and engagement priors. If the full recommendation path fails, return a cached regional popularity feed.

I would retrain the ranker daily. I would retrain the two-tower model weekly because each new retriever requires a large re-index. I would compute new-video embeddings at upload and insert them incrementally.

### The hard tradeoff

The central tradeoff is watch time versus satisfaction.

Watch time is dense, immediate, and easy to optimize. However, it can reward long, repetitive, or emotionally provocative content. This can increase short-term engagement while reducing long-term trust.

Therefore, I would use watch time as the main signal, add explicit positive and negative satisfaction signals, and make long-term retention a hard launch guardrail. I would not launch a model that improves watch time but significantly reduces day-28 retention.

The design has one core shape: retrieve about 1,000 candidates from one billion videos, rank them with a calibrated multi-task model, rerank the list for policy, diversity, freshness, and exploration, and validate the result with long-term online metrics.


---

## Design 2 — Google search ranking

### Requirements and assumptions

I will design query understanding, retrieval, and organic ranking. I will assume a few hundred billion documents, 10–14 billion queries per day, a 200 ms p99 budget, and ten organic results. Human grades and click logs are available. Personalization uses language, locale, and coarse location.

**Functional requirements.** The system must understand and rewrite queries, retrieve lexical and semantic candidates, rank by relevance and quality, remove spam and duplicates, support fresh documents, generate snippets, and log impressions and clicks.

**Non-functional requirements.** The service must stay below 200 ms at p99, support global traffic, keep important documents fresh, survive slow shards, and provide stable quality across languages, regions, and head and tail queries.

### ML framing and data

This is learning to rank. Pointwise loss scores documents independently. Pairwise loss learns which of two documents ranks higher. Listwise methods optimize the result list. LambdaMART is a strong baseline because it weights pairwise gradients by the change in NDCG.

Human grades are small but unbiased by position. Click logs are large but biased. A position-based click model is:

$$
P(\text{click}\mid d,k)=P(\text{examine}\mid k)P(\text{relevant}\mid d)=\theta_k\gamma_d.
$$

I would estimate $\theta_k$ from a small randomized slice and use inverse propensity weights $1/\theta_k$. Query features include spelling, intent, entities, language, and freshness intent. Document features include authority, quality, spam, language, and age. Query-document features include BM25, title match, proximity, dense similarity, and point-in-time behavioral aggregates.

### Architecture

BM25 handles exact terms, names, and identifiers. A bi-encoder and ANN index handle semantics and paraphrases. Their union enters a ranking cascade.

```text
query
 -> query understanding                                      ~10 ms
 -> BM25 inverted index + dense bi-encoder ANN in parallel   ~40 ms
 -> merge and deduplicate: about 30,000 documents             ~5 ms
 -> L1 cheap scorer: BM25, authority, spam -> 1,000          ~20 ms
 -> L2 LambdaMART with rich features -> 100                  ~30 ms
 -> L3 cross-encoder over query+document -> 10-20            ~60 ms
 -> snippets, diversity, and page assembly                   ~15 ms
 -> ten organic results
```

A bi-encoder computes document vectors offline. A cross-encoder processes query and document together. It is more accurate but cannot be precomputed. Therefore, it runs only on the final candidates.

### Evaluation and production

Use recall@k for retrieval and NDCG, MRR, and human relevance for ranking. Segment navigational, informational, local, news, head, and tail queries. Online metrics include long clicks, successful sessions, reformulation, abandonment, and latency.

The stages total 180 ms and leave 20 ms of headroom. Cache popular query-locale results. Use hedged requests for slow shards. If dense retrieval fails, return lexical results. If L3 times out, return L2 ordering.

The main tradeoff is cross-encoder quality versus cost. I would route L3 to difficult queries when L2 confidence is low.


---

## Design 3 — Ads ranking and CTR prediction

### Requirements and assumptions

I will assume 100 million eligible ads, 10 billion impressions per day, and a 50 ms p99 budget. Advertisers bid per click or conversion. The system uses a second-price-style auction, reserve prices, pacing, frequency caps, and about one ad per eight organic items.

**Functional requirements.** The system must enforce targeting, predict click and conversion, calibrate probabilities, pace budgets, run the auction, enforce frequency caps, return a winner, and log impressions, prices, clicks, and delayed conversions.

**Non-functional requirements.** The path must stay below 50 ms at p99, support peak traffic, prevent overspend, remain calibrated by segment, protect privacy, resist click fraud, and provide auditable prices and safe fallbacks.

### ML framing

For click-priced campaigns:

$$
\text{eCPM}=1000P(\text{click})\text{bid}_{\text{CPC}}.
$$

For conversion campaigns:

$$
\text{eCPM}=1000P(\text{click})P(\text{conversion}\mid\text{click})\text{bid}_{\text{CPA}}.
$$

Probabilities must be calibrated. Expected calibration error is:

$$
\mathrm{ECE}=\sum_b\frac{n_b}{N}|\bar p_b-\bar y_b|.
$$

In a simplified second-price auction:

$$
\text{CPC}_{\text{charged}}=
\frac{\text{eCPM}_{\text{second}}}{1000P(\text{click})_{\text{winner}}}.
$$

Therefore, calibration errors directly change price. Clicks arrive quickly but have position and fraud bias. Conversions are sparse and delayed. Train on mature conversions or model the delay.

### Architecture

```text
ad request
 -> targeting index + eligibility, budget, policy, cap filters ~10 ms
    100M ads -> 50K-100K
 -> two-tower and prior-eCPM retrieval -> 1,000-2,000           ~8 ms
 -> DLRM or wide-and-deep ranker                               ~20 ms
 -> segment calibration                                         ~1 ms
 -> pacing controller                                            ~2 ms
 -> auction and reserve price                                    ~2 ms
 -> winner, logging, response                                    ~4 ms
```

Targeting is set intersection, not ML. The ranker uses user, ad, context, and cross features. A wide part memorizes known combinations. A deep part generalizes to new campaigns. Platt scaling or isotonic regression calibrates each placement and objective. The pacing controller adjusts campaign admission or bid multipliers against a target spend curve.

### Evaluation and production

Use log loss, ECE, PR-AUC, and ranking metrics offline. Use revenue, conversions, advertiser return, and budget delivery online. Guardrails include complaints, hides, ad load, latency, concentration, and segment calibration.

Ten billion impressions per day is about 116,000 per second on average. At $3\times$ peak, plan for about 350,000 per second. The latency total is 47 ms.

If ranking fails, use calibrated historical priors. If pacing is stale, use conservative admission. Never skip policy, budget, or frequency-cap rules.

The main tradeoff is short-term revenue versus long-term user and advertiser trust.


---

## Design 4 — Fraud detection at a payment processor

### Requirements and assumptions

I will design processor-side card-not-present fraud detection. The merchant bears chargeback loss. The inline path has a 100 ms p99 budget. Actions are approve, step-up, or decline. High-value cases can enter review. Assume a 0.1% fraud-attempt rate.

**Functional requirements.** The system must apply rules, compute velocity and graph features, produce a calibrated fraud probability, choose an action with reason codes, update state, support review, join delayed chargebacks, and log randomized approvals.

**Non-functional requirements.** It must stay below 100 ms at p99, support about 12,000 peak TPS, remain highly available, keep features fresh within seconds, protect payment data, support audit, and degrade conservatively.

### ML framing and decision rule

Predict $p=P(\text{fraud}\mid x)$. Accuracy is useless because always predicting legitimate gives 99.9% accuracy.

Approve when:

$$
pC_{\mathrm{FN}}<(1-p)C_{\mathrm{FP}}.
$$

The threshold is:

$$
p^*=\frac{C_{\mathrm{FP}}}{C_{\mathrm{FN}}+C_{\mathrm{FP}}}.
$$

Missed-fraud cost grows with transaction amount. Therefore, thresholds depend on amount and merchant risk. Two thresholds create approve, step-up, and decline regions.

Features include amount, merchant, card, device, IP, location consistency, account age, velocity windows, and graph links. Chargebacks arrive 30–90 days later. Declined transactions have no normal outcome. A small randomized approval slice provides unbiased labels.

### Architecture

```text
transaction
 -> schema, blocklist, deterministic rules                    ~1 ms
 -> parallel velocity and graph feature reads                ~20 ms
 -> calibrated GBDT fraud model                               ~8 ms
 -> cost-sensitive policy: approve / step-up / decline        ~3 ms
 -> reason codes, durable log, response                       ~5 ms

events -> stream processor -> online feature store
reviews + chargebacks -> mature labels -> training -> registry
```

A GBDT handles tabular data, nonlinear thresholds, and missing values with low latency. Rules respond to new attacks within minutes. Streaming velocity features detect bursts such as card testing.

### Evaluation and production

Use PR-AUC, recall at a fixed false-positive rate, precision at fixed recall, calibration, and total expected dollar cost. Guardrails are authorization rate, false declines, step-up pass rate, latency, and segment disparities.

Assuming $1.9$ trillion annual volume and a $50 average transaction gives about 1,200 TPS on average. A $10\times$ peak gives 12,000 TPS. The stages total about 43 ms.

Feature freshness is a paging metric. If streaming data is stale, use a fallback model without velocity features and route more traffic to step-up. If the model fails, use conservative rules.

The main tradeoff is fraud loss versus false declines. Choose thresholds with expected dollar cost and enforce authorization rate and fairness guardrails.


---

## Design 5 — Content moderation at scale

### Requirements and assumptions

Assume one billion items per day: 70% text, 25% images, and 5% video. The system supports 30 policy categories and 40 languages. Actions are allow, demote, age-gate, remove, or human review. Text has a 150 ms inline budget. Media decisions complete within 60 seconds. Human capacity is 250,000 reviews per day.

**Functional requirements.** Detect violations across modalities, match known-bad content, produce per-policy scores, select actions, route uncertain or severe cases to humans, support appeals, update policies quickly, and log policy and model versions.

**Non-functional requirements.** Process about 35,000 items per second at peak, keep text below 150 ms at p99, finish media checks within 60 seconds, limit human routing to 0.025%, support 40 languages, resist attacks, and provide auditable and reversible actions.

### ML framing and architecture

This is multi-label classification followed by cost-sensitive routing. Each policy has its own precision, recall, severity, and thresholds. Review priority should approximate expected harm reduced per reviewer-minute. Labels come from reviewers, appeals, reports, random audits, and known-bad sets. Store the policy version with each label. Random audits prevent selection bias from training only on escalated content.

```text
content
 -> normalization, hard rules, and blocklists
 -> perceptual hash / known-bad matching
 -> modality encoder + per-policy classifier heads
      confident safe -> allow
      confident violation -> demote / age-gate / remove
      uncertain -> multimodal LLM judge
      severe or unresolved -> human review
 -> appeal -> corrected label -> training
```

A shared encoder reduces compute. Separate policy heads support independent policy changes. The LLM judge handles only a small uncertain band. Humans handle severe, high-reach, appealed, or unresolved cases.

One billion items per day is 11,600 per second on average and about 35,000 at $3\times$ peak. The peak mix is 24,000 text, 8,700 image, and 1,700 video items per second. Video dominates compute because frame sampling multiplies work.

The inline text path uses about 15 ms for ingress, 1 ms for rules, 15 ms for context, 25 ms for classification, and 10 ms for policy and logging. This totals 66 ms and leaves 84 ms of headroom. LLM and human review stay asynchronous.

### Evaluation and tradeoff

Report precision, recall, calibration, and prevalence by policy, language, region, and modality. Track violating-content exposure, time to action, appeals, overturns, review backlog, false removals, and segment disparities.

If the media fleet fails, keep hash checks active and demote queued media. If the LLM judge fails, demote uncertain content and prioritize humans. Canary threshold changes and rate-limit mass removals.

The main tradeoff is speed versus certainty. Fast automation reduces exposure but increases false actions. Demotion is a reversible action for uncertain cases. Reserve automatic removal for high-confidence or catastrophic categories.


---

## Design 6 — An LLM serving platform

### Requirements and assumptions

Assume 70% interactive and 30% batch traffic. The platform serves an 8 B model, a 70 B model, 20 LoRA adapters, and one vendor mixture-of-experts model. Interactive targets are TTFT below 1 second and inter-token latency below 50 ms at p95. Peak demand is four times average.

**Functional requirements.** Authenticate tenants, enforce request and token quotas, route model versions and adapters, stream tokens, batch requests, manage KV cache, support vendor and self-hosted models, record usage, and deploy versions safely.

**Non-functional requirements.** Meet TTFT and ITL SLOs, isolate tenants, prevent capacity monopolies, maximize GPU use, protect prompts and caches, support overload, provide accurate billing, and enable rollback.

### Serving model and architecture

Prefill processes prompt tokens in parallel and is compute-bound. Decode produces one token per sequence per step and is memory-bandwidth-bound. Continuous batching adds and removes sequences after each decode step.

KV-cache bytes per token are:

$$
2n_{\text{layers}}n_{\text{kv heads}}d_{\text{head}}\times\text{bytes per element}.
$$

PagedAttention stores cache in fixed blocks. Prefix caching reuses shared prompt blocks. Cache keys include tenant ID to prevent cross-tenant timing leaks.

```text
client
 -> gateway: auth, quotas, token limits
 -> router: model, version, adapter, prefix affinity
 -> fair queues and admission control
 -> continuous-batching scheduler
 -> prefill/decode workers: PagedAttention + tensor parallelism
 -> streamed tokens
telemetry -> billing, capacity, quality, safety
vendor models -> same gateway and policies
```

A 70 B FP16 model needs $70\times10^9\times2=140$ GB for weights. TP=4 across four 80 GB H100s gives 320 GB total and leaves useful KV-cache capacity. Estimate about 160 GB of KV cache, or 500,000 cached tokens.

At 200 requests per second, 1,500 input tokens, and 300 output tokens, output demand is 60,000 tokens per second. At 3,520 output tokens per second per four-GPU node, base load needs about 17 nodes or 68 GPUs. Peak-only provisioning would need 272 GPUs. Caching, quantization, batch backfill, and queueing can reduce the practical fleet toward 100–120 GPUs.

### Evaluation and tradeoff

Track TTFT, ITL, queue wait, total latency, goodput, preemption, cache hit rate, GPU memory, batch size, throughput, errors, and cost by model and tenant.

A 1-second TTFT budget can use 10 ms for gateway checks, 5 ms for routing, 300 ms for queueing, 130 ms for prefill, 30 ms for first-token delivery, and 525 ms of headroom.

Reject requests whose predicted KV footprint does not fit. Use chunked prefill for long prompts. Under overload, use fair scheduling and clear admission control. If a vendor fails, route to a compatible self-hosted model.

The main tradeoff is latency versus cost. Larger batches improve throughput but raise queueing, ITL, and KV use. Segment interactive and batch traffic. Batch only until the interactive SLO is reached.


---

## Design 7 — Semantic image search

### Requirements and assumptions

Design hybrid search for 50 million listings and 200 million images. Support text, image, and image-plus-text queries at 1,000 QPS and 300 ms p99. About 500,000 listings change daily and must be searchable within five minutes. The primary metric is add-to-cart rate, with relevance and revenue guardrails.

**Functional requirements.** Validate and crop uploads, encode text and images into one space, retrieve lexical and visual candidates, apply price, stock, rights, region, and safety filters, rerank, deduplicate products, support new listings, and return metadata and CDN URLs.

**Non-functional requirements.** Stay below 300 ms at p99, support 1,000 QPS, index updates within five minutes, scale to 200 million images, preserve ANN recall, keep encoder-index versions consistent, tolerate shard failures, and protect uploaded-image privacy.

### ML framing and architecture

A CLIP-style bi-encoder maps text and images into one space. Contrastive learning uses matched pairs as positives and other batch items as negatives:

$$
\mathcal{L}=-\frac{1}{N}\sum_i
\log\frac{\exp(\operatorname{sim}(v_i,t_i)/\tau)}
{\sum_j\exp(\operatorname{sim}(v_i,t_j)/\tau)}.
$$

Fine-tune a pretrained model on purchases, clicks, and multiple images of one product. Use hard negatives such as visually similar but different models. Use human judgments for unbiased evaluation.

```text
catalog:
listing event -> validate/deduplicate -> image encoder
              -> fresh ANN index -> main-index merge
              -> metadata and lexical index

query:
upload/text -> decode, detect, crop -> image/text encoder
            -> ANN + BM25 in parallel -> fusion + filters
            -> cross-encoder rerank top 100
            -> GBDT business ranker + diversity/dedup
            -> products, metadata, CDN images
```

For 200 million 768-dimensional FP16 vectors:

$$
200\times10^6\times768\times2\approx307\text{ GB}.
$$

With HNSW overhead, estimate 450 GB. Eight shards hold about 56 GB each. Replicate shards because sharding reduces data per machine, not per-shard QPS.

Use a small fresh HNSW index for updates. Merge it into the main index. Version encoder and index together. A V18 query vector is invalid against V17 catalog vectors. Build the full V18 index before a dual-index switch.

### Evaluation and tradeoff

Measure ANN recall against exact search, retrieval recall@k, precision@k, NDCG, human relevance, duplicate rate, and quality by query type. Online metrics are click, add-to-cart, conversion, abandonment, and reformulation. Guardrails include unsafe results, rights violations, latency, stale inventory, and concentration.

A 300 ms budget can use 30 ms for upload, 20 ms for crop detection, 15 ms for encoding, 25 ms for ANN, 15 ms for parallel BM25, 2 ms for fusion, 40 ms for cross-encoder reranking, 5 ms for business ranking, and 8 ms for assembly. This uses 145 ms and leaves 155 ms for network and p99 headroom.

If reranking fails, order by retrieval fusion. If a shard fails, merge remaining results. If a new index fails, keep the previous encoder-index pair.

The main tradeoff is visual similarity versus purchase intent. Retrieval protects relevance. Ranking can optimize value, but it must not introduce irrelevant products.


---

## Cross-cutting questions

These get asked regardless of which design you drew. They are the interviewer's way of finding out whether you've operated a system or only read about one, and the good news is that the answers transfer completely — learn these eight and you have a response to most probes in most designs.

### How do you handle cold start?

Cold start is the problem that a model needs history to make a good prediction, and some entities have none: a user who signed up thirty seconds ago, an item listed this morning, a query nobody has ever typed. The reason it matters more than it seems is that cold entities are not a rare edge case — on a growing platform they're a large and strategically important slice, and they're exactly the population whose experience determines whether the platform keeps growing.

The general answer is to **fall back through a hierarchy of decreasing specificity**, and to say which level you're on rather than pretending you have signal you don't. For a **cold user**: no personal history, so use whatever context the request itself carries — device, locale, referrer, time of day — plus population-level popularity, plus anything an onboarding flow collected. Then update aggressively, because the first few interactions carry enormous information relative to a long-tenured user's next one; a session-based model that conditions on the last handful of actions rather than a long-term embedding is often better for the first day than the main model is. For a **cold item**: you have no behavioural signal but you always have *content* — the text, the image, the category, the seller — so a content-derived embedding places the item in the same space as items that do have history, and it inherits their neighbourhood as a prior. This is the single most valuable property of content-based embeddings and it's why the image-search design leans on them. For a **cold query**: fall back to lexical matching, which needs no history at all, and to query rewriting against a taxonomy.

Two mechanisms deserve naming. **Exploration**: reserve a small budget — a few percent of impressions — for items with high uncertainty, because an item that's never shown never accumulates the data that would let it be shown. Without an explicit exploration budget, a ranker trained on its own logs will permanently suppress everything it wasn't initially confident about, and the catalogue quietly ossifies. Thompson sampling or a simple epsilon-greedy slice both work; the important part is that the budget exists and is measured. **Two-tower architectures with content features** make cold start structurally cheaper: if the item tower takes content features rather than a learned per-item ID embedding, a brand-new item gets a meaningful vector on its first forward pass, with no retraining at all. Designing for cold start at the architecture level beats patching it at serving time, and volunteering that is the senior version of this answer.

### How do you detect and respond to drift?

Drift is the model getting worse because the world changed, not because the code changed, and it comes in three flavours worth distinguishing. **Covariate shift** is the input distribution moving while the input-to-output relationship holds — new traffic from a new country, a UI change that alters what users type. **Label shift** is the base rate moving — fraud attempts triple, or a policy category's prevalence jumps during an election. **Concept drift** is the relationship itself changing — the features that predicted fraud last month don't this month, because the fraudsters adapted. The response differs by type, which is why naming them matters: covariate shift may need only recalibration, concept drift needs retraining, and label shift often needs neither if the model is well-calibrated and you adjust the prior.

Detection runs on three signals, in increasing order of usefulness and decreasing order of availability. **Input distributions**, monitored per feature, using population stability index or KL divergence against a reference window — cheap, immediate, no labels required, and noisy enough that it produces false alarms if you're not careful about which features you watch. **Output distributions**, which are usually the better signal: if the fraction of items scoring above threshold moves without a deploy, something upstream changed, and this single metric catches more real incidents than any input monitor. **Performance metrics** on whatever labels you have, which is the ground truth and is also the slowest and most expensive.

Responding well is mostly about having decided in advance. Automatic retraining on a schedule handles slow drift without anyone thinking about it. Threshold or calibration refits handle the common case where the model's ranking is still fine but its scores have shifted, and they're minutes of compute rather than hours. A **holdback** — a small permanent slice of traffic on a frozen old model — is the most underrated tool here, because comparing current-model to frozen-model performance over time separates "the world got harder" from "our model got worse," which are diagnosed and fixed completely differently. And for anything adversarial, accept that retraining cadence is a design parameter set by how fast your adversary adapts, not by convenience.

### How do you deal with delayed labels?

Many labels arrive long after the prediction. A chargeback lands 60–90 days after the transaction. A moderation appeal resolves in days. A subscription cancellation is a label on an acquisition decision made a year ago. The problem this creates is subtle and it bites people: **at any moment, your recent data looks artificially clean**, because the positives haven't arrived yet. Train on it naively and you learn that recent transactions are safe, which is precisely backwards.

The first discipline is to never treat "no label yet" as "negative." Maintain an explicit label-maturity window per label type, and either exclude immature data from training or model the maturity explicitly. The clean way to model it is to estimate the **label delay distribution** — empirically, what fraction of eventual positives have arrived by day $d$ — and reweight recent examples by the inverse of that fraction, which is a standard survival-analysis correction and lets you use recent data without the bias.

The second move is to find **proxy labels** that arrive fast and correlate with the slow one. Fraud has manual-review outcomes in hours and card-network alerts in days, both long before the chargeback. Moderation has reviewer decisions in hours before appeals resolve. Churn has engagement decay weeks before cancellation. Train on the proxy for freshness, validate against the true label as it matures, and monitor the proxy-to-true correlation, because the day that correlation breaks is the day your fast pipeline starts lying to you.

The third is architectural: **separate the fast and slow loops**. A slow loop retrains the main model on mature, fully-labelled data at a cadence matched to the label delay. A fast loop adapts on top of it — recalibration, threshold adjustment, or a lightweight model on proxy labels — at a much higher cadence. This gets you responsiveness without contaminating the model that has to be right. And for evaluation, always report metrics with the as-of date and the maturity level attached, because comparing a mature month against an immature one is the most common way teams fool themselves into thinking a model improved.

### How do you handle a feedback loop where the model's own output becomes its training data?

This is the most important question in this list, because it's the failure mode that's invisible from inside the system. A recommender shows what it predicts you'll like, you can only click what it showed, and the clicks become training data — so the model learns that its own choices were correct. Moderation escalates what it suspects, humans label the escalations, and the model never sees its own blind spots. Search ranks by predicted relevance, users click the top result, and the click confirms the rank. In every case the system converges toward self-consistency rather than toward truth, and every offline metric improves while real quality degrades.

The clearest symptom is a widening gap between offline and online performance, and a narrowing distribution of what the model outputs — reduced entropy in recommendations, escalation concentrated on a shrinking set of patterns, a long tail that stops being served at all. Both are directly measurable and neither requires labels, so they belong on a dashboard.

The mitigations are structural and there are essentially four. **Random exploration** is the fundamental one: a slice of traffic where selection is randomized rather than model-driven, which gives you unbiased data about what the model *would* have suppressed. It costs a little quality on that slice and it is the only thing that genuinely breaks the loop, so budget for it explicitly and defend it when someone tries to cut it. **Propensity correction** lets you use the biased logged data honestly: record the probability with which each item was shown, then weight training examples by the inverse of that probability, which recovers an unbiased estimate of what would have happened under uniform exposure. Clip the weights, because small propensities produce enormous variance. **Independent evaluation** means a measurement path that doesn't depend on the model's own choices — human judgments, a random audit sample, a holdback on a frozen model. This is what the content-moderation design's random audit stratum is for, and it's the reason to protect it from cost-cutting. And **content-based features** reduce the loop's grip, because a model that scores items on their intrinsic properties rather than their accumulated interaction history can rate something it has never shown.

The framing to offer: any system that both selects and learns from what it selected needs a deliberate source of information it did not select, and if you can't point at that source in your design, you have a feedback loop whether you've noticed it or not.

### How do you debug a model that got worse in production but not offline?

This is the most common real incident in ML, and the value of the answer is having an ordered procedure rather than a list of possibilities. Work from cheapest and most likely to most expensive.

Start with **train/serve skew**, which is the culprit far more often than anything else. The same feature is computed by different code in the training pipeline and the serving path, and they disagree — a different default for missing values, a timezone difference, a unit difference, a normalization applied in one place and not the other. The direct test is to log the actual feature vectors used at serving time, replay the exact same examples through the training pipeline, and diff them feature by feature. If any feature differs, stop; you've found it. This is exactly what feature stores exist to prevent (Part 1), and "same code computes training and serving features" is the structural fix.

Then check **temporal leakage** in the offline evaluation. If the offline split was random rather than time-based, the model saw the future during training — it learned from Wednesday to predict Tuesday — and offline metrics are inflated in a way production can never reproduce. Any aggregate feature computed over the full dataset rather than as-of the prediction time does the same thing more subtly. Re-run the offline evaluation with a strict temporal split and see whether the gap closes on its own.

Then **distribution mismatch**: compare the production input distribution against the training distribution feature by feature. Common causes are a training set filtered in a way production isn't (deduplicated, bot-filtered, or restricted to complete records), or a training set that's simply older than you thought.

Then **feedback and selection effects**: is the offline evaluation set drawn from what the previous model selected? If so, the new model is being graded on the old model's curriculum, which systematically favours models that behave like the old one.

Then **the serving path itself**, which people check too late: a preprocessing difference, a truncation limit, a model-version mismatch between replicas, a batching bug that mixes up rows. Compare a handful of individual predictions between offline and online for the identical input. If the same input produces a different score, it's engineering, not modelling, and the search space just collapsed.

Finally, consider that **the offline metric may be measuring the wrong thing** — a ranking model improving on NDCG while hurting revenue because it optimizes relevance and the business runs on margin. That's not a bug; it's a misalignment, and it's the topic of the last question in this section.

### How do you decide between one model and many?

The question shows up as: one global model, or one per country, per customer, per segment, per language? The honest framing is that this is a bias-variance tradeoff with an engineering-cost term attached, and the engineering term usually dominates in practice.

A **single global model** sees all the data, so it has far more signal per parameter and generalizes better to sparse segments. It's one thing to train, deploy, monitor, and debug. Its weakness is that it fits the average, so it can systematically underperform on segments whose behaviour differs from the majority — and it'll do that invisibly, because the aggregate metric is dominated by the majority.

**Many specialized models** fit each segment's idiosyncrasies exactly, and they fail in three ways: small segments have too little data and overfit, you now have $N$ training pipelines and $N$ monitoring dashboards and $N$ opportunities for one to go stale unnoticed, and cold-start for a new segment has no answer at all.

The resolution that's almost always right is the **hybrid**: one shared model with segment identity as an input. Add a segment embedding, or per-segment feature crosses, or a per-segment output head over a shared trunk. This gets you most of the specialization benefit while keeping one pipeline, and it degrades gracefully — a new segment starts at the population average and specializes as data accumulates, which is exactly the cold-start behaviour you want. The content-moderation design's shared encoder with per-policy heads is this pattern, and so is multi-LoRA serving in the LLM platform: one base, many cheap adapters.

Split into genuinely separate models only when there's a *structural* reason rather than a statistical one: different feature availability (one market has data you're not allowed to use elsewhere), different label semantics (a policy that means something different by jurisdiction), regulatory isolation, or wildly different scale where one segment's volume would swamp training. The diagnostic to actually run: train the global model, evaluate per segment, and look at where per-segment performance lags the aggregate. If the gap is small, you're done. If one segment is badly served, try adding it as a feature before you fork the model — and only fork if that fails.

### How do you handle multi-tenancy and per-customer models?

Multi-tenancy means one system serving many customers whose data must not mix, and it has three faces: isolation, customization, and fairness.

**Isolation** is the non-negotiable one. Customer A's data must not leak into a prediction served to customer B, and the leak paths are more numerous than people expect. The obvious one is training on pooled data, where a memorizing model can reproduce another tenant's data — real for large models, essentially the same concern as training-data extraction. The less obvious ones are shared caches, where a timing difference reveals that another tenant sent the same input (this is the prompt-cache side channel from the LLM-serving design, and it generalizes to any cache keyed on content), and shared feature stores, where a badly-scoped key returns another tenant's aggregates. The defence is that tenant ID is part of every cache key, every feature key, and every index partition, enforced at the framework level rather than left to each service to remember.

**Customization** is the same one-model-or-many question in a different costume, and the same hybrid answer applies with an extra tier. Most tenants get the shared model, because they don't have enough data to beat it and they benefit from everyone else's. Tenants with real volume get a cheap personalization layer — a tenant embedding, a per-tenant calibration, or a LoRA adapter in the LLM case — which is where most of the value is for most of the cost. Only tenants with genuinely different label semantics or a contractual demand for isolation get their own model, and they should pay for it, because a per-tenant model is a per-tenant pipeline and someone has to keep it alive. The economic point worth making: the cost of a customized model is not the training run, it's the perpetual maintenance, and that cost is linear in tenant count while the revenue often isn't.

**Fairness** is resource allocation, and it's the LLM-serving design's weighted fair queueing generalized. One tenant must not be able to degrade another's service, whether through a traffic spike, a pathologically expensive request, or a retraining job that saturates the cluster. That means per-tenant rate limits and quotas, a service metric that reflects real cost rather than request count, and enough isolation in the scheduler that a noisy neighbour degrades their own experience first. And it means per-tenant monitoring, because an aggregate p95 hides the tenant for whom the system is broken — the single most common way a multi-tenant platform is bad without anyone knowing.

### What do you do when the business metric and the model metric disagree?

You launched a model with better AUC and revenue went down. This happens constantly, and how you respond is one of the sharper seniority signals in the whole interview, because the junior instinct is to defend the model and the senior instinct is to distrust the metric.

The first thing to do is **believe the business metric**, provided it's measured properly. The model metric is a proxy chosen for convenience; the business metric is closer to what the company actually wants. If they disagree, the default assumption is that the proxy is wrong, not that the business is wrong.

Then diagnose, because there are only a few things it can be. The most common is that **the proxy is misaligned**: AUC measures ranking quality uniformly across all pairs, but revenue depends on the top few positions and on calibration, and a model can improve average ranking while getting the head of the distribution slightly worse. In ads this is the classic — better AUC with worse calibration destroys revenue, because bids are computed from probabilities and a systematically shifted probability misprices every auction. Check calibration explicitly whenever a model's scores are used as numbers rather than as an ordering. The second possibility is a **distribution or segment effect**: the model improved on the bulk and regressed on a small high-value segment, and revenue is concentrated there. Segment the A/B result by user value, query type, and geography before concluding anything. The third is a **second-order effect**: the model is better at the immediate objective and worse for the system — more relevant results that reduce browsing, or higher short-term engagement that costs retention. These only show up over longer horizons and are why holdbacks and long-run experiments exist. The fourth, and check it early because it's embarrassing, is that **the experiment is broken**: unbalanced assignment, a novelty effect, insufficient power, or a bug that only fires in the treatment arm.

Then act. If the proxy is misaligned, fix the proxy — retrain against a metric closer to the business objective, or add calibration as an explicit constraint — rather than fighting the evidence. If it's a segment regression, either fix the segment or gate the launch to the segments that improved. If it's second-order, you need a longer experiment before you can decide anything at all, and the right answer to the interviewer is "I'd hold the launch and run a holdback for 60 days," not a guess. And the lesson to state, because it's the general one: **the fastest way to prevent this is to choose the offline metric to correlate with the online metric, and to validate that correlation empirically** — take your last twenty experiments, plot offline delta against online delta, and if there's no correlation, your offline metric is not doing its job and no amount of improving it will help.

---
