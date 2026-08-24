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

## Section D. Reusable architecture patterns

### Retrieval followed by ranking

Use two stages when the catalogue is too large for full scoring. Retrieval reduces millions or billions of items to hundreds or thousands. It uses cheap methods and optimizes recall. Ranking uses richer features and a more expensive model to optimize ordering. A final reranker applies list-level rules such as diversity, policy, freshness, and deduplication. Retrieval recall is a hard ceiling because ranking cannot recover an item that retrieval dropped.

### Bi-encoder and cross-encoder

A bi-encoder computes query and item vectors independently. Therefore, item vectors can be computed offline and searched with an ANN index. This makes large-scale retrieval possible. A cross-encoder processes the query and item together. It models richer interactions but needs one forward pass per pair. Therefore, use the bi-encoder for retrieval and the cross-encoder only for a small reranking set.

### Cascades

A cascade applies cheap decisions first and expensive decisions only to unresolved cases. For example, moderation can use rules, a small classifier, a large multimodal model, and human review. The escalation threshold controls cost and quality. Early-stage confidence must be calibrated because a confidently wrong early decision never reaches a stronger stage.

### Feature freshness

Use batch computation for stable long-window features, streaming for signals that change within seconds, and request-time computation for current context. Choose freshness per feature. Streaming every feature adds cost without value. Batch processing a fraud velocity feature makes it useless.

### Read and write paths

The write path collects events, validates them, updates features and indexes, and creates training data. The read path serves requests from prepared state. Design them together. The read path depends on write-path freshness, and the write path must log enough information to reproduce each production decision.


---

## Design 1 — YouTube's home-feed recommender

**The scenario.** The interviewer says: *"Let's design the recommendation system behind YouTube's home feed. When I open the app, I get a grid of videos — how do you build the thing that decides what's in that grid? Assume you have all of Google's infrastructure. Roughly two billion users, a corpus in the billions of videos. Take me through it."* They will probably add something offhand like *"oh and it needs to feel fast"* — that throwaway line is the latency budget, and you should pick it up.

What makes this hard is not the model. It is the **corpus size**. There are on the order of a billion candidate videos and roughly a hundred milliseconds to produce twenty of them, which means you cannot score every video — not with any model, not on any hardware. The design is therefore forced into a shape: something cheap that reduces a billion to a thousand, then something expensive that reduces a thousand to twenty. Everything else in the answer hangs off that. The second thing being tested is whether you notice that **there is no label**. Nobody tells YouTube which video you wanted; the system only sees what it showed you and what you did. You have to manufacture a training signal out of behaviour, and every choice you make there — click versus watch time versus completion — becomes the product. Candidates who jump straight to "I'd use a transformer" fail this question because they never confront either problem.

### Step 1 — Clarify (2 minutes)

These are the questions to actually say out loud, and what each answer buys you.

*"What are we optimizing — clicks, watch time, or something longer-horizon like retention?"* This is the single most consequential question in the whole design, and it is not a formality. If the answer is clicks, the system will learn clickbait: a thumbnail that overpromises maximizes clicks and minimizes satisfaction. If the answer is watch time, you get a different pathology — long, slow, autoplay-friendly content wins over short, excellent content. If the answer is retention, you have a beautiful objective and an unusable one, because you cannot wait thirty days for a label on every impression. The real answer at YouTube is a blend: optimize a short-horizon proxy that correlates with the long-horizon goal, and use the long-horizon goal as a guardrail in experiments.

**Assume, for the rest of this answer:** we optimize expected watch time per impression, with day-7 and day-28 retention as experiment guardrails. Roughly 2.5 billion monthly logged-in users (this is the publicly reported figure; I am treating it as approximately right rather than exact) and about one billion home-feed loads per day. Corpus of one billion servable videos. Budget: 150 ms end-to-end at p99 for the recommendation call. Output: a ranked list of 50, of which the client shows about 20 above the fold. New uploads should be reachable within roughly an hour. Policy filters are hard filters.

**Functional requirements.** Generate a personalized feed, retrieve from several sources, rank candidates, apply safety and diversity, support cold start, log decisions, return 50 playable videos, run experiments, and roll back safely.

**Non-functional requirements.** Stay below 150 ms at p99 and support about 35,000 peak QPS. Index new videos within one hour. Degrade safely when dependencies fail. Keep training data point-in-time correct and follow privacy and retention rules.

### Step 2 — Frame it as an ML problem

Start by writing down the unit of prediction, because everything downstream is determined by it. The unit here is an **impression**: one (user, video, context) triple at one moment in time. Context means time of day, device, app surface, and network conditions. For each impression we want a number that lets us sort — and the number we choose is expected watch time if this video were shown to this user right now.

Now the label. YouTube does not have a "did the user want this" label; it has logs. The logs record that video $v$ was shown to user $u$ at position $k$ and that the user either did nothing, or clicked and watched for $t$ seconds. So the label has to be constructed. The natural construction is: a positive example is an impression that led to a watch of at least some minimum duration, and the *value* of that positive is the watch time. The classic published approach from YouTube's own 2016 paper is elegant and worth knowing by name: they train **weighted logistic regression** on the final layer, where positive impressions are weighted by their observed watch time and negatives get weight one. Under that weighting, the learned odds $\frac{p}{1-p}$ come out approximately equal to expected watch time — because the positives have been inflated in proportion to how long they were watched — so at serving time they simply exponentiate the logit, $e^{Wx+b}$, and use it directly as a watch-time estimate. This is a nice trick to be able to state: it lets a classification model produce a regression-like quantity.

Three things are wrong with this label, and naming them is most of the points on this step.

The first pathology is that the label is **implicit**, meaning it is inferred from behaviour rather than stated. A user who watches ten minutes of a video may have loved it or may have left the tab open. A user who watches ten seconds may have hated it or may have gotten exactly the answer they needed. Implicit labels are noisy in a way that is systematic rather than random, and no amount of data fixes systematic noise.

### Step 3 — The data

**Features about the user.** The strongest signal by far is watch history — the sequence of videos this person has watched, in order, with dwell times. Sequence matters: someone who just watched three videos about guitar pedals is in a different state than someone who watched them last March. So represent history two ways. Long-term interests become a dense vector summarizing months of behaviour, refreshed daily. Short-term intent becomes the last 20 to 50 watched video IDs, fed as a sequence into the model and refreshed within seconds of each action. Alongside these sit stable attributes — country, language, device class, account age — and request context.

The word **embedding** is going to appear constantly, so pin it down here. An embedding is a fixed-length vector of numbers that stands in for a discrete thing. A video ID is just an arbitrary integer, and integers have no useful geometry — video 5 is not "between" videos 4 and 6. So we learn a vector for each video, say 256 floating-point numbers, such that videos watched in similar contexts end up with vectors close together. The mapping from ID to vector is stored in an **embedding table**: literally a big matrix with one row per ID, whose rows are learned parameters updated by gradient descent like any other weight. Once things are vectors, "similar" becomes a dot product, and dot products are something hardware is extremely good at.

**Features about the video.** Metadata is cheap and useful: channel, upload time, duration, language, topic taxonomy. Content features come from the media itself — a text encoder over title and description, a vision encoder over the thumbnail and sampled frames, an ASR transcript. These matter most for new videos, which have no behavioural data at all. Engagement priors — historical click-through rate, average watch-through fraction, like rate — are extremely predictive and extremely dangerous, for reasons in the leakage discussion below.

### Step 4 — The architecture

Walk it as a story. A request arrives from a phone. It carries a user ID, a device, a locale, and a timestamp, and it needs 50 videos back in under 150 milliseconds.

**First stop: the feature service.** This is a low-latency key-value store — Redis or an internal equivalent, described properly in Part 1 — holding precomputed user state keyed by user ID. One read returns the long-term interest vector, the demographic block, and the recent-watch sequence. Budget 5 ms. If this read fails, you do not fail the request; you fall back to a locale-level default profile and continue, which will produce a generic but non-empty feed.

**Second stop: candidate generation.** This is the stage that reduces one billion to about one thousand, and it is the stage that must be cheap. The distinction between **candidate generation and ranking** is the backbone of the whole design: candidate generation optimizes *recall* — did the good videos survive into the shortlist — using a model cheap enough to run against the entire corpus. Ranking optimizes *precision at the top* — of the survivors, which order is best — using a model expensive enough that it could never be run on a billion items. Recall errors at stage one are unrecoverable: a video that does not make the shortlist has zero chance of being shown no matter how good the ranker is.

The workhorse here is a **two-tower model**, and this is the term the reader must be able to say out loud. Picture two separate neural networks. The left tower takes everything about the user — history sequence, interests, context — and outputs a single 256-dimensional vector. The right tower takes everything about a video — content, metadata, priors — and outputs a 256-dimensional vector in the same space. The predicted affinity between a user and a video is just the dot product of the two vectors. The towers never see each other's inputs; they only meet at that final dot product. That restriction is the entire point. Because the video tower's output does not depend on the user, you can run it offline over all one billion videos and store the results. At request time you only run the user tower once, then find which of the billion stored vectors have the largest dot product with it — a pure geometry problem, no neural network involved. A model where user and item features are mixed together in early layers (a **cross-encoder**, which we will meet in Design 2) is more accurate and cannot be precomputed, so it is unusable at this stage.

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp\!\big(s(u_i, v_i)/\tau\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_i, v_j)/\tau - \log q(v_j)\big)}
$$

Once trained, the video tower is run over the corpus offline and the resulting billion vectors are loaded into an **ANN index**. ANN stands for approximate nearest neighbour: given a query vector, return the vectors with the largest dot product, accepting that you might miss a few in exchange for being orders of magnitude faster than checking all one billion. Two structures matter.

$$
\text{score} = \big(p_{\text{click}}\big)^{\alpha} \cdot \big(\mathbb{E}[\text{watch time}]\big)^{\beta} \cdot \big(1 + \gamma \, p_{\text{like}}\big) \cdot \big(1 - \delta \, p_{\text{hide}}\big)
$$

#### Architecture diagram

```
                       home-feed request (user_id, device, locale, ts)
                                        |
                                        v
                        +-------------------------------+
                        |  FEATURE SERVICE (KV store)   |   ~5 ms
                        |  long-term vec, recent watches|
                        +-------------------------------+
                                        |
              +--------------+----------+-----------+---------------+
              v              v                      v               v
      +---------------+ +------------+      +--------------+ +-------------+
      | TWO-TOWER ANN | | SUBSCRIBED |      |  TRENDING    | |  FRESH /    |
      | user vec ->   | | channels   |      |  per region  | |  CF co-watch|
      | HNSW / IVF-PQ | | recent     |      |  counters    | |             |
      |  ~500 cands   | |  ~200      |      |   ~100       | |   ~200      |
      +---------------+ +------------+      +--------------+ +-------------+
              |              |                      |               |
              +--------------+----------+-----------+---------------+
                                        v
                        +-------------------------------+
                        |  MERGE + DEDUPE + HARD FILTER |   ~3 ms
                        |     -> ~1000-1500 candidates  |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |  FEATURE HYDRATION            |   ~15 ms
                        |  per-candidate + cross feats  |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |  RANKER (DLRM + MMoE heads)   |   ~50 ms
                        |  p_click, p_watch50, p_like,  |
                        |  p_hide, E[watch time]        |
                        |  + shallow position tower     |
                        +-------------------------------+
                                        |  top ~200 by blended score
                                        v
                        +-------------------------------+
                        |  RERANKER                     |   ~5 ms
                        |  diversity caps, freshness,   |
                        |  policy, exploration slots    |
                        +-------------------------------+
                                        |
                                        v
                              response: 50 videos
                                        |
                                        v
                        +-------------------------------+
                        |  LOGGING -> event stream      |
                        |  impressions, positions,      |
                        |  scores, feature snapshot     |
                        +-------------------------------+
```

### Step 5 — Evaluation

Offline, evaluate the two stages separately, because they have different jobs. For candidate generation the metric is **recall@k**: over held-out impressions where the user actually watched something, what fraction of the time does the watched video appear in the retrieved top $k$? If recall@1000 is 60%, then 40% of good outcomes are already lost before ranking runs, and no ranker improvement can recover them. For the ranker, the metric is **AUC** or, better, per-user grouped AUC — AUC computed within each user's impression list and then averaged — because global AUC can look great simply by separating heavy users from light users, which is not a ranking skill. Add log loss, which unlike AUC is sensitive to calibration and will catch a model that orders well but predicts nonsense magnitudes.

The honest thing to say next is that all of these are measured on data the current system generated, and they therefore reward agreement with the current system. A new model that surfaces genuinely good videos the old model never showed gets *penalized* offline, because those impressions have no positive label — they have no label at all. This is why offline metrics are a filter for obviously-broken models, not a launch decision.

For long-horizon effects, keep a small **holdback**: a permanent slice of users, perhaps 0.5%, held on an older model for months. It is the only way to measure whether a year of incremental watch-time optimization has quietly damaged retention, since each individual A/B was too short to see it.

### Step 6 — Production concerns

**Throughput.** One billion home-feed loads per day is $10^9 / 86{,}400 \approx 11{,}600$ requests per second on average. Traffic is not flat; peak-to-average of about 3x is a reasonable assumption for a global consumer product, so plan for ~35,000 QPS. Each request scores about 1000 candidates in the ranker, so the ranker performs $35{,}000 \times 1000 = 3.5 \times 10^7$ item-scorings per second at peak. That number is the reason the ranker cannot be a large transformer: at 35 million scorings per second, every microsecond of per-item cost is 35 seconds of aggregate compute per second, i.e. 35 machines. Sizing the ranker is not an accuracy decision, it is a fleet-size decision.

**Storage.** Item embeddings: $10^9 \times 256 \times 2$ bytes (fp16) = 512 GB, plus HNSW edges at $10^9 \times 32 \times 4$ = 128 GB, so ~640 GB, sharded across roughly 10 machines with 64 GB of index each plus replicas for throughput. With PQ at 64 bytes per vector this drops to 64 GB and the sharding is driven by QPS instead. User state: 2.5 billion users at ~2 KB each is 5 TB in the online store, though you would only keep the active subset hot and page the rest.

**Latency budget** at p99, summing to the 150 ms target: network and request parsing 10 ms; feature service read 5 ms; candidate generation 30 ms (all sources in parallel, so this is the slowest source, not the sum); merge and filter 3 ms; feature hydration 15 ms; ranker inference 50 ms; reranker 5 ms; response serialization 5 ms. Total 123 ms, leaving about 27 ms of headroom for tail effects. State the headroom explicitly — interviewers like a budget that does not exactly equal the target, because one that does is a budget that will be blown.

**Failure modes and what happens.** Each stage needs a defined degradation, not a 500. If the ANN index is unreachable, serve from the other retrieval sources; the feed gets less personal but exists. If the feature service is down, use a locale-level default user profile. If the ranker times out, return candidates ordered by retrieval score and engagement priors — noticeably worse, still a feed. If everything is down, serve a cached per-locale popularity list, which should be pregenerated and refreshed hourly precisely so that this path is always warm. The general principle: every stage has a cheaper fallback, and you rehearse them, because a fallback path that has never taken traffic will not work the first time it has to.

**Monitoring** in three layers. Infrastructure: QPS, p50/p95/p99 per stage, error rates, cache hit rates, index staleness. Model: prediction distribution drift measured by population stability index (PSI), feature-level drift, calibration plots comparing predicted watch time to realized watch time in deciles, and coverage — what fraction of the catalog receives any impression at all, which catches a model collapsing onto a narrow slice. Business: watch time, sessions, retention, dislike rate, creator-side distribution. Part 1 covers the dashboard and alerting stack; the relevant point here is that the model layer is the one teams forget, and it is the one that fails silently.

**Retraining.** The ranker retrains daily on a rolling window of recent logs, warm-started from yesterday's weights rather than from scratch. The two-tower retriever retrains weekly, because re-indexing a billion items is expensive and its embeddings move more slowly. New item embeddings are computed continuously at upload and inserted incrementally into the index, so a video is retrievable within about an hour without a full rebuild. Triggers for off-schedule retraining: PSI above a threshold on important features, calibration error exceeding a band, or a step change in a business metric. And the pipeline must support rollback to the previous model artifact in minutes, which means keeping the last several versions loadable and having a deployment flag that flips traffic instantly.

### The hard tradeoff

**Should the ranker optimize watch time, or should it optimize satisfaction?**

The case for watch time is that it is dense, immediate, and unambiguous. Every impression produces a number within minutes. It is machine-readable, it correlates with revenue because more watching means more ad inventory, and it makes the training loop tight — you can retrain daily and see the effect. Sparse alternatives like surveys cover a tiny sample and arrive slowly, so a model trained on them is starved of data.

The case against is that watch time is a proxy that diverges from the goal in specific, predictable ways. It rewards length over quality: a mediocre 40-minute video beats an excellent 4-minute one. It rewards autoplay-friendly, low-effort content. It rewards emotionally activating content, since outrage holds attention. And crucially, the divergence compounds — each retraining cycle bakes in the previous cycle's drift, so the system walks steadily toward content that holds attention without being wanted. You do not see this in a two-week A/B test because the damage accrues over months.

What I would actually do is neither pure option. Optimize a blended objective where watch time is the dominant term but explicit satisfaction signals — likes, surveys where available, and especially negative signals like hides and "not interested" — enter with meaningful weight, and treat long-horizon retention as a hard guardrail rather than an objective. Concretely, refuse to launch a model that improves watch time while regressing day-28 retention, even at high statistical significance on the primary metric.

What would change my mind: evidence from the long-term holdback. If a year of watch-time optimization shows no retention divergence between the holdback and production, the proxy is not diverging and the added complexity of satisfaction modelling is not earning its keep. If the holdback shows divergence, I would go further than the blend and move satisfaction from a weighted term to a constraint.

---

## Design 2 — Google search ranking

### Requirements and assumptions

**Assume for the rest:** I own query understanding through final ordering. Corpus of a few hundred billion documents (Google publicly says "hundreds of billions of webpages," an index over 100 million gigabytes). Query volume on the order of 10-14 billion per day — public estimates vary widely and I am treating this as approximate. Both human relevance grades and click logs are available. Personalization limited to locale, language, and coarse location. Server-side budget 200 ms at p99. Output: ten organic results plus whatever else the page assembles.

**Functional requirements.** The system must understand and rewrite queries, retrieve lexical and semantic candidates, rank by relevance and quality, remove spam and duplicates, support fresh documents, generate snippets, and log impressions and clicks.

**Non-functional requirements.** The service must stay below 200 ms at p99, support global query volume, keep important documents fresh, survive slow or failed shards, and provide stable quality across languages, regions, and head and tail queries.

### Step 2 — Frame it as an ML problem

The naive framing is binary classification: given a (query, document) pair, predict relevant or not. That framing is wrong and saying why is worth real points. Relevance is not a property of a document in isolation, it is a property of a document *relative to the other documents available for that query*. A mediocre page about a very obscure topic may deserve rank one because nothing better exists; an excellent page may deserve rank eight for a competitive query. And the loss you care about is not per-document accuracy, it is the quality of the *ordering of the top few*. Getting position 200 wrong costs nothing. Getting position 1 wrong costs everything.

That is what **learning-to-rank** means: a family of methods whose loss function is defined over a list rather than an example. There are three, and interviewers ask you to compare them.

**Pointwise** treats each (query, document) pair independently and predicts an absolute score or grade — plain regression or classification. It is simple, it reuses all standard tooling, and its weakness is that it optimizes the wrong thing. A model that predicts every document's grade with small error can still order them badly, and it wastes capacity being accurate about documents nobody will ever see.

**Pairwise** takes pairs of documents for the same query where one is graded higher, and learns to score the better one above the worse one. The loss is on the *difference*, so absolute calibration stops mattering and the model concentrates on ordering, which is what you want. RankNet introduced this with a logistic loss on score differences. The weakness is that all pairs count equally: swapping documents at positions 1 and 2 and swapping documents at positions 99 and 100 contribute the same gradient, which is not how anyone uses a search engine.

**Listwise** defines the loss over the whole ranked list and tries to optimize the ranking metric directly. The obstacle is that ranking metrics are step functions of the scores — a document's contribution changes only when it crosses another document — so they have zero gradient almost everywhere. LambdaRank's insight was to skip the loss entirely and specify the *gradient*: take the pairwise gradient and multiply it by how much the ranking metric would change if you swapped those two documents. Pairs that matter get large gradients, pairs deep in the tail get almost none. **LambdaMART** is that gradient rule implemented inside gradient-boosted trees, and for tabular ranking features it remains a genuinely strong baseline, not a historical curiosity.

Now the labels. Two sources with opposite properties.

**Human relevance grades** come from trained raters following a rubric, producing a graded label per (query, document) pair — commonly a 0-4 scale from "off-topic" to "fully meets the need." They are high quality, unbiased by position, and expensive: you get millions of pairs, not billions, and they skew toward queries someone thought to sample. They also measure topical relevance rather than satisfaction; a rater cannot tell you whether the page loaded slowly or was covered in ads.

### Step 3 — The data

**Query-side features** come out of query understanding: the corrected spelling, the segmentation into terms, a predicted intent class (navigational, informational, transactional, local, news-seeking), any linked entities, and the query's own statistics — how frequent it is, whether its volume just spiked, its historical reformulation rate.

**Document-side features** are query-independent and precomputed at index time: link-graph authority in the PageRank family, spam and quality classifiers, page-experience signals like load time and layout stability, language, country, publication and last-update timestamps, and content-quality scores from models run offline over the page.

**Query-document features** are the interesting ones. Lexical match, chiefly **BM25** — the standard bag-of-words scoring function that rewards a document for containing the query's terms, weighted so rare terms count more than common ones and so the tenth occurrence of a word adds less than the second, with a normalization for document length so long documents cannot win by sheer size. Field-specific matches (does the query appear in the title, the URL, the anchor text pointing at this page). Semantic similarity from dense embeddings. Proximity — do the query terms appear near each other. And aggregated historical behaviour for this exact query-document pair: click-through rate, dwell time after clicking, and whether users came back and searched again.

That last family is enormously predictive and carries the sharpest **leakage** risk in the design, so explain it rather than naming it. Suppose you build a training row for query $q$ and document $d$ on some date, and you attach "CTR of $d$ for $q$" computed from the full log dump. That statistic includes the clicks from the very impression you are training on, and from impressions after it. The model then learns "documents that get clicked for this query are relevant," which is circular — at inference for a *new* query-document pair the feature is undefined or based on much less data. Offline metrics look spectacular, live performance does not move. The fix is point-in-time correctness: compute the feature over a window ending strictly before the row's timestamp, and enforce it in the pipeline rather than trusting people to remember. A second, sneakier form: these historical features do not exist for new documents, and if "missing" is encoded as zero, the model learns that new documents are bad, which permanently suppresses fresh content. Encode missing as genuinely missing — GBDTs handle this natively — and give the model a "document age" feature so it can learn how to treat young pages.

### Step 4 — The architecture

A query arrives. Two hundred milliseconds.

**Query understanding** runs first and is cheap. Spelling correction, language identification, segmentation, intent classification, entity linking against a knowledge graph. It also produces query rewrites and expansions — synonyms and related forms that will be sent to retrieval alongside the original. Budget 10 ms. The output is not one query but a small bundle of them.

**Retrieval** runs next and must reduce hundreds of billions to tens of thousands. Two complementary systems run in parallel.

The **inverted index** is the classical structure: for every term in the vocabulary, a posting list of the document IDs containing it, sorted, compressed. To answer "machine learning tutorial" you intersect three posting lists and score the survivors with BM25. This is exact on terms, extremely fast, and it fails on vocabulary mismatch — a page that says "automobile" does not match a query that says "car." At this corpus size the index is sharded across many thousands of machines by document, each holding a slice; the query is broadcast to all shards, each returns its local top-$k$, and a gathering tier merges. Note the fan-out pattern, because it has a consequence: a request's latency is the *slowest* shard's latency, so tail latency at each shard becomes typical latency for the query. The standard mitigation is hedged requests — send the same shard request to two replicas and take whichever answers first — which trades roughly 5% extra load for a large p99 improvement.

**Dense retrieval** covers what the inverted index misses. A **bi-encoder** — the same two-tower idea as Design 1, with a query encoder and a document encoder producing vectors in one space — is run over the corpus offline for documents and at query time for the query, and an ANN index (HNSW or IVF-PQ, defined in Design 1) returns nearest neighbours. This matches on meaning rather than tokens, so it retrieves the "automobile" page for the "car" query. Its weakness is the mirror image: it is fuzzy about exact strings, so it can miss a rare product code or a specific name. Union the two. Neither alone is acceptable, and knowing *why* each fails is the point of running both. Budget 40 ms. Output: on the order of tens of thousands of candidates.

Now the **cascade**. The idea is simple and it is the organizing principle of the whole system: you have a fixed compute budget and a huge candidate set, so you apply a sequence of rankers of increasing cost and decreasing input size, each one's job being to hand a smaller, better set to the next. The economics only work if the cost per document rises faster than the candidate count falls. A useful way to say it: each tier should cost roughly the same in total, so a tier that is 100x more expensive per document should see 100x fewer documents.

**L1** is a cheap, fast scorer over tens of thousands of candidates using only precomputed, cheap features — BM25, PageRank-family authority, spam score, a coarse semantic similarity. It is often a small GBDT or even a linear model, and it must be fast enough to run on 30,000 documents in under 20 ms, which means roughly half a microsecond per document. It cuts to about 1000.

**L2** is a proper learning-to-rank model — LambdaMART over a few hundred features, or a lightweight neural equivalent — running on 1000 documents with full feature hydration. Budget 30 ms. It cuts to about 100.

An optional **L4** exists in modern systems: an LLM used as a relevance judge or answer synthesizer for the hardest queries. Route only queries the earlier tiers flag as ambiguous or low-confidence, because you cannot afford it on all traffic. Say explicitly that this is a routed, minority path.

**Result assembly** then builds the page: snippet generation per result, deduplication so the same site does not take six of ten slots, diversity for ambiguous queries (a query like "jaguar" should show both the animal and the car rather than betting everything on one interpretation), plus knowledge panels and other verticals. Budget 15 ms.

#### Architecture diagram

```
                                query text
                                     |
                                     v
                    +--------------------------------+
                    |  QUERY UNDERSTANDING           |  ~10 ms
                    |  spell, segment, intent,       |
                    |  entity link, rewrites         |
                    +--------------------------------+
                                     |
                 +-------------------+-------------------+
                 v                                       v
     +------------------------+              +------------------------+
     | INVERTED INDEX (BM25)  |              | DENSE RETRIEVAL        |
     | sharded x thousands    |              | bi-encoder + ANN       |
     | exact term match       |              | semantic match         |
     +------------------------+              +------------------------+
                 |   scatter/gather, hedged                |
                 +-------------------+---------------------+
                                     v
                    +--------------------------------+
                    |  MERGE / DEDUPE  ~30k docs     |  ~40 ms total
                    +--------------------------------+
                                     |
                                     v
                    +--------------------------------+
                    |  L1  cheap scorer              |  ~20 ms
                    |  BM25, authority, spam, coarse |
                    |  ~0.5 us per doc  -> top 1000  |
                    +--------------------------------+
                                     |
                                     v
                    +--------------------------------+
                    |  L2  LambdaMART / LTR          |  ~30 ms
                    |  few hundred features          |
                    |  -> top 100                    |
                    +--------------------------------+
                                     |
                                     v
                    +--------------------------------+
                    |  L3  cross-encoder             |  ~60 ms
                    |  query+doc through transformer |
                    |  -> top 10-20                  |
                    +--------------------------------+
                                     |
                          (hard queries only)
                                     v
                    +--------------------------------+
                    |  L4  LLM judge / synthesis     |  routed minority
                    +--------------------------------+
                                     |
                                     v
                    +--------------------------------+
                    |  PAGE ASSEMBLY                 |  ~15 ms
                    |  snippets, host dedupe,        |
                    |  intent diversity, verticals   |
                    +--------------------------------+
                                     |
                                     v
                          10 results  +  logging
                          (positions, scores,
                           propensities)
```

### Step 5 — Evaluation

The offline metric is **NDCG**, and you should be able to derive it, not just name it. Start with discounted cumulative gain: each result contributes a gain based on its relevance grade, discounted by how far down it sits, since users look at the top.
$$
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i+1)}
$$
The $2^{\mathrm{rel}} - 1$ numerator makes highly relevant documents worth disproportionately more than moderately relevant ones — a grade-4 document is worth 15, a grade-2 is worth 3 — which matches the fact that one excellent result beats three mediocre ones. The $\log_2(i+1)$ denominator implements the position discount. DCG is not comparable across queries, because a query with many relevant documents can score higher than one with few no matter how well you rank. So normalize by the ideal ordering:
$$
\mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
$$
where IDCG is DCG of the perfect ranking. Now every query is on a 0-to-1 scale and averaging across queries is meaningful.

What NDCG misses is worth saying plainly. It measures topical relevance against a rater rubric, so it is blind to everything the rater did not judge: page load time, ad density, whether the answer required scrolling past three paragraphs of preamble. It scores each query independently, so it cannot see that your ten results are near-duplicates of one another, and it cannot see intent diversity for ambiguous queries — you need separate metrics such as $\alpha$-NDCG or explicit host-diversity counts for that. It is also only as good as your rater pool's agreement; if two raters disagree 20% of the time, an NDCG difference of half a point is noise.

Guardrails: NDCG on the held-out human-rated set must not regress; per-segment NDCG on tail queries, non-English queries, and each major locale must not regress even if the average improves; spam-domain share of top-10 impressions; and latency, since a ranking win that costs 40 ms may be a net loss.

### Step 6 — Production concerns

**Throughput.** Take 12 billion queries per day as a working figure (public estimates run from roughly 8.5 to 14 billion; I am treating this as an assumption). That is $1.2 \times 10^{10} / 86{,}400 \approx 139{,}000$ queries per second on average. With a peak-to-average factor of 2, plan for about 280,000 QPS. Now the cascade arithmetic. L1 sees 30,000 documents per query, so at peak it processes $2.8\times10^5 \times 3\times10^4 = 8.4 \times 10^9$ document-scorings per second — which is why L1 must cost well under a microsecond per document and cannot be a neural network. L3 sees 100 documents, so $2.8\times10^5 \times 100 = 2.8\times10^7$ cross-encoder evaluations per second. Even a distilled cross-encoder at, say, 1 ms of accelerator time per document implies 28,000 accelerator-seconds per second of wall clock — an enormous fleet. This is the arithmetic that makes **caching** structural rather than an optimization: query frequency follows a heavy-tailed distribution, and if the top queries account for a large share of volume, caching full result pages for common queries at a modest TTL removes most of that load. Say the cache hit rate is 40%; the L3 fleet shrinks by 40% immediately.

**Storage.** A few hundred billion documents with, say, 500 bytes of ranking features each is on the order of 100 TB of feature data, before the inverted index and the document text itself, distributed across many thousands of machines. The dense index at $3\times10^{11}$ documents is not storable as full-precision vectors under any budget — $3\times10^{11} \times 768 \times 4$ bytes is roughly 900 TB — which is why dense retrieval at web scale uses heavy quantization and typically covers a filtered subset of the corpus rather than every crawled page. Be willing to say that out loud: the honest answer is that dense retrieval complements the inverted index over a curated slice, it does not replace it over everything.

**Latency budget** at p99, summing to 200 ms: query understanding 10; retrieval 40 (parallel, dominated by the slowest shard, mitigated by hedged requests); merge and dedupe 5; L1 20; L2 30; L3 60; page assembly 15. Total 180 ms, with 20 ms of slack.

**Failure modes.** If dense retrieval fails, serve from the inverted index alone; results degrade on paraphrased queries and remain acceptable. If L3 times out, return L2's ordering, which is worse but coherent — this is the single most valuable property of a cascade, that every tier's output is a valid answer. If a retrieval shard is unresponsive, return results from the remaining shards and mark the response as partial; missing one shard out of thousands is usually invisible. The dangerous failure is a **poisoned index**: a spam campaign or an ingestion bug causing a class of low-quality documents to surge. That is not a crash, it is a quality collapse, and the only defense is monitoring the composition of top-10 results by domain age, domain reputation, and spam score, with alerts on sudden shifts.

**Monitoring.** Infrastructure: QPS, per-tier latency, shard health, cache hit rate, index freshness lag. Model: score distribution drift, feature drift, the fraction of queries where L3 substantially reorders L2's output (if that fraction collapses, L3 has stopped contributing and you are paying a fleet for nothing), and per-segment NDCG on a continuously refreshed rated sample. Business and quality: reformulation rate, long-click rate, abandonment, and the spam and freshness composition metrics above.

### The hard tradeoff

**Should the cascade end in a cross-encoder, or should you push more work into retrieval and stop earlier?**

The cross-encoder case: joint attention over query and document is qualitatively more capable than any comparison of independently-computed vectors. It is the only tier that can resolve negation, argument roles, and entity relationships — the difference between "who did X beat" and "who beat X." On hard, long-tail, natural-language queries this is not a marginal gain, it is the difference between an answer and a near-miss. And the cost is bounded by design: 100 documents, not 30,000.

The other case: the cross-encoder is by a wide margin the most expensive component in the stack, and per the arithmetic above it dominates serving cost. Every millisecond it consumes is a millisecond unavailable elsewhere, and there is a real alternative use for that budget — a better bi-encoder with hard negative mining, or a larger L2 with richer features, might recover a large fraction of the gain at a fraction of the cost. There is also a distributional argument: head queries are navigational and are answered correctly by BM25 plus authority, so the cross-encoder's benefit concentrates on a minority of traffic while its cost is paid on all of it.

My position is to keep the cross-encoder but **route** it: run it on all traffic only if the tier's contribution justifies the fleet, and otherwise gate it on a cheap confidence signal from L2 — run it when L2's top scores are close together or the query is long and natural-language, skip it when L2 is confident or the query is navigational. That converts a fixed cost into a cost proportional to difficulty. Distillation on top: train a small cross-encoder to imitate a large one, which typically retains most of the quality.

What would change my mind: an ablation showing that on the routed subset the cross-encoder's NDCG lift is under a point, or that a strong bi-encoder trained with hard negatives closes most of the gap. Conversely, if the routed subset shows large gains and represents growing traffic — and natural-language queries are growing as people talk to search engines the way they talk to assistants — I would spend more here and cut L2 instead.

---

## Design 3 — Ads ranking and CTR prediction

### Requirements and assumptions

**Assume for the rest:** advertisers bid per click, with optional conversion-optimized campaigns where they bid per conversion and the platform converts internally. Second-price-style auction with per-slot reserve prices. Fixed ad load of roughly one ad per eight organic items. Budgets are paced across the day. About 100 million eligible ads, 10 billion impressions per day, 50 ms budget at p99 running in parallel with organic ranking.

**Functional requirements.** The system must enforce targeting and eligibility, predict click and conversion probabilities, calibrate them, pace budgets, run the auction, enforce frequency caps, return a winner, and log impressions, prices, clicks, and delayed conversions.

**Non-functional requirements.** The path must stay below 50 ms at p99, support peak auction traffic, prevent budget overspend, remain calibrated by placement and segment, protect user privacy, resist click fraud, and provide auditable prices and safe fallbacks.

### Step 2 — Frame it as an ML problem

The system's job at each request is to pick the ad that maximizes expected value, where value is measured in money. Define the currency first.

**eCPM** stands for effective cost per mille — expected revenue per thousand impressions. It exists because advertisers bid in different units and you need to compare them. If an advertiser bids \$2.00 per click and the model predicts a 1.5% click-through rate, then each impression is worth $0.015 \times \$2.00 = \$0.03$ in expectation, and per thousand impressions that is \$30. So
$$
\text{eCPM} = 1000 \times p_{\text{click}} \times \text{bid}_{\text{CPC}}
$$
For a conversion-optimized campaign bidding per action, you chain the two probabilities:
$$
\text{eCPM} = 1000 \times p_{\text{click}} \times p_{\text{conversion} \mid \text{click}} \times \text{bid}_{\text{CPA}}
$$
And for a CPM campaign the eCPM is just the bid, no model required. Everything now lives on the same axis, measured in dollars per thousand impressions, and the auction can rank across campaign types.

So the ML problem is two probability estimation tasks: $p_{\text{click}}$ given an impression, and $p_{\text{conversion}}$ given a click. Both are binary classification, both are trained with log loss, and both must be *calibrated*, not merely well-ordered.

Say precisely what calibration means, because it is the hinge of this design. A model is calibrated if, among all impressions where it predicted 2%, about 2% actually get clicked. You measure it by bucketing predictions into deciles and comparing the mean prediction in each bucket to the observed rate, or with a single summary number, expected calibration error:
$$
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\big|\,\overline{p}_b - \overline{y}_b\,\big|
$$
where $\overline{p}_b$ is the mean predicted probability in bucket $b$ and $\overline{y}_b$ the observed rate.

Why it matters here and not in Design 1: in a second-price auction the winner does not pay their own bid, they pay the smallest amount that would still have won. Concretely, the winner is the ad with the highest eCPM, and the price they are charged per click is
$$
\text{CPC}_{\text{charged}} = \frac{\text{eCPM}_{\text{second}}}{1000 \times p_{\text{click}}^{(1)}}
$$
Look at where $p_{\text{click}}^{(1)}$ sits — in the denominator of the price. If the model overpredicts the winner's click rate by 10%, the winner still wins, the ordering is unchanged, and the platform charges 10% less per click than it should. Miscalibration converts directly into lost revenue without changing a single ranking decision. If the model *under*predicts, the advertiser is overcharged, which is worse in a different way — it is an advertiser-trust problem and, at scale, a regulatory one.

Now the labels and their pathologies.

**Clicks** are fast, dense, and mostly honest. They arrive within seconds, so a click model can be retrained hourly. Their problems are position and context bias — an ad in the second slot outperforms the same ad in the seventh — and click fraud, which inflates rates for specific ads or specific traffic sources and must be filtered before training or the model learns to love bot traffic.

### Step 3 — The data

**User features** mirror Design 1: long-term interest embeddings from behaviour, recent activity, demographics available under the platform's privacy policy, device and connection, and — importantly here — advertising-specific history such as how many ads this user has seen today, how recently they saw *this* advertiser, and their historical rate of hiding or reporting ads. That last set drives both the model and the frequency-capping rules.

**Ad features** include the creative itself (image and text, encoded by pretrained vision and text models into embeddings), the campaign objective, the advertiser and their vertical, the landing page and its quality, the targeting specification, and historical performance — this creative's click rate, this advertiser's click rate, this vertical's click rate. Historical performance is the strongest single feature and creates the same leakage and cold-start issues as Design 1: point-in-time correctness is mandatory, and "no history" must be encoded as unknown rather than zero, or every new creative will be ranked as though it were terrible.

**Cross features** are where this model earns its keep, and they are why the architecture is what it is. The signal is not "this user clicks ads" or "this ad gets clicked" but "this user clicks *this kind of ad* in *this context*." Recall from Design 1 that a **feature cross** is the conjunction of two categorical features treated as a single new feature — (user's country, advertiser vertical) as one token rather than two independent ones. Linear models cannot represent conjunctions; they can only sum independent contributions. The cross is what lets the model know that people in one country respond to one vertical without inferring it from country alone and vertical alone.

**Feature cardinality and hashing.** The sparse feature space here is enormous: hundreds of millions of ad IDs, a billion user IDs, plus crosses that multiply cardinalities together. A cross of 10,000 user segments with 10,000 ad categories is $10^8$ possible values. The **hashing trick** — mapping each feature string through a hash function into a fixed number of buckets, say $2^{24}$, and learning one embedding per bucket — makes the table size a design parameter instead of a consequence of the data. Collisions are the cost, and they are unevenly harmful: two rare features sharing a row barely matters, two head features sharing a row is a real quality loss. Mitigations are multiple independent hash functions whose embeddings are summed, so a collision in one is unlikely to coincide with a collision in another, and reserving explicit non-hashed rows for the top few million most frequent values.

### Step 4 — The architecture

A feed request arrives and contains an ad slot. Fifty milliseconds.

**Targeting and eligibility filtering** runs first, and it is not machine learning — it is a fast set-intersection problem, and it does the heaviest reduction in the pipeline. Every campaign specifies a targeting predicate: geography, language, age range, interest segments, custom audiences, device type, plus exclusions. Given a user, you need the set of campaigns whose predicates the user satisfies. This is implemented as an inverted index over targeting attributes, the same structure as Design 2's text index but over audience attributes instead of terms: for each attribute value, a posting list of campaign IDs requiring it. Intersect and union the lists implied by this user's attributes and you get eligible campaigns without evaluating a hundred million predicates. Applied at the same time are the hard filters: campaigns that have exhausted their budget, campaigns paused by pacing, ads this user has been frequency-capped on, brand-safety exclusions, and policy blocks. This is where 100 million becomes something like 50,000 to 100,000. Budget 10 ms.

**Candidate retrieval** narrows further. Even 50,000 is too many to score with a heavy model in the remaining budget, so use the same tools as Design 1: a two-tower retriever over user and ad embeddings, plus a cheap eCPM estimate using historical priors (last-known CTR times bid), plus a rule-based path guaranteeing that certain campaign types get considered. Take the union down to roughly 1,000 to 2,000. Budget 8 ms.

**Ranking** scores the survivors with the pCTR and pCVR models. The architecture is the **wide-and-deep** or **DLRM** family described in Design 1 — sparse categorical features through hashed embedding tables, dense features through an MLP, an explicit interaction layer computing pairwise products among embeddings, then a shared trunk with task heads. Two heads matter here: click probability and conversion-given-click. In practice you also add auxiliary heads — probability of hide, probability of a long dwell on the landing page — because they regularize the trunk and because they feed the user-experience term in the final score.

The wide part deserves a sentence of justification in this design specifically. Ads have a lot of *memorizable* structure: a particular advertiser genuinely does perform well with a particular audience segment, and that is a fact to be stored, not generalized. The wide linear component over explicit crosses stores such facts exactly. The deep component generalizes to combinations never observed, which is what handles the constant churn of new campaigns. Both matter, which is precisely the argument in the original wide-and-deep paper.

**Calibration layer.** After the model comes an explicit, separately-fitted calibration stage, and this is a distinct component, not a training detail. The two standard methods: **Platt scaling** fits a one-dimensional logistic regression on the model's output, $p_{\text{cal}} = \sigma(a \cdot z + b)$ where $z$ is the model's logit, learning just two parameters on a held-out set — cheap, stable, and only capable of correcting a monotone S-shaped distortion. **Isotonic regression** fits an arbitrary non-decreasing step function mapping raw scores to calibrated probabilities — far more flexible, capable of fixing weird local distortions, and prone to overfitting on small data, so it needs a decent held-out sample. Fit calibration per meaningful segment, because a globally calibrated model is routinely miscalibrated per country, per placement, and per campaign objective, and the auction runs within segments.

One correction to a common miscalibration source: if you **downsample negatives** during training — which you probably do, since impressions vastly outnumber clicks — the model's outputs are calibrated to the resampled distribution, not the real one. If you kept negatives with probability $w$, invert it analytically:
$$
p = \frac{p_s}{p_s + (1 - p_s)/w}
$$
where $p_s$ is the model's output on the downsampled distribution. Failing to apply this correction produces a model that overpredicts by exactly the downsampling factor, which in an auction is a direct, large revenue error.

#### Architecture diagram

```
             feed request with ad slot (user, context, placement)
                                  |
                                  v
              +-------------------------------------------+
              |  TARGETING + ELIGIBILITY                   |  ~10 ms
              |  inverted index over audience attributes;  |
              |  budget-exhausted, paced-off, freq-capped, |
              |  brand-safety and policy filters removed   |
              |  100M ads  ->  ~50-100k eligible           |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |  CANDIDATE RETRIEVAL                       |  ~8 ms
              |  two-tower ANN + prior-eCPM shortlist      |
              |  -> ~1000-2000 candidates                  |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |  RANKING MODEL (wide-and-deep / DLRM)      |  ~20 ms
              |  hashed sparse embeddings + dense MLP      |
              |  + explicit crosses                        |
              |  heads: p_click, p_conv|click, p_hide      |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |  CALIBRATION                               |  ~1 ms
              |  Platt / isotonic, fitted per segment;     |
              |  downsampling correction applied           |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |  PACING                                    |  ~2 ms
              |  rho from PI controller vs spend curve     |
              +-------------------------------------------+
                                  |
                                  v
              +-------------------------------------------+
              |  AUCTION                                   |  ~2 ms
              |  eCPM = 1000 * pClick * (pConv) * bid      |
              |  x rho x quality; reserve price applied;   |
              |  second-price clearing                     |
              +-------------------------------------------+
                                  |
                                  v
                         winning ad -> rendered
                                  |
        +-------------------------+--------------------------+
        v                         v                          v
  impression log            click stream                conversion
  (features, preds,         (seconds)                   pipeline
   auction state,                                       (hours to days,
   position)                                            attribution join)
```

### Step 5 — Evaluation

Offline, the primary metric is **log loss**, not AUC, and the reason is calibration. AUC depends only on the ordering of predictions, so a model that predicts every probability at exactly twice the true value has unchanged AUC and is catastrophic in an auction. Log loss,
$$
-\frac{1}{N}\sum_i \big[ y_i \log p_i + (1-y_i)\log(1-p_i) \big],
$$
is a proper scoring rule: it is minimized only when the predicted probabilities equal the true ones, so it penalizes both bad ordering and bad magnitude. Report AUC too, since it isolates ranking quality, but treat log loss as the gate. Add expected calibration error and a calibration plot per major segment, and report **relative information gain** — the log-loss improvement over a constant-rate baseline — because raw log loss values are hard to interpret when the base rate is 1.5%.

What offline metrics miss here is severe and specific. First, they are computed on impressions the current system chose to serve, so they say nothing about ads never shown — the same selection bias as Design 1. Second, log loss on the conversion model computed over a recent window is *systematically wrong* because of delayed feedback, so a model can appear to degrade purely because the label window is young. Always evaluate the conversion model on a fully matured window. Third, and most important, offline metrics have no notion of the auction. A model can improve log loss and reduce revenue, because revenue depends on the *relationship* between the winner's and runner-up's predictions rather than on average accuracy.

### Step 6 — Production concerns

**Throughput.** Ten billion impressions per day is $10^{10}/86{,}400 \approx 116{,}000$ per second on average; at a 3x peak factor, about 350,000 per second. Each request ranks roughly 1,500 candidates, so the ranker performs $3.5\times10^5 \times 1.5\times10^3 \approx 5.2\times10^8$ ad-scorings per second at peak. That is more than an order of magnitude above Design 1's ranker load, on a budget less than half as large, which is the quantitative reason ads rankers are shallow and wide rather than deep: the architecture is chosen for arithmetic intensity that suits batched inference on accelerators, and the embedding tables are the memory bottleneck rather than the compute.

**Revenue arithmetic**, useful for grounding the calibration argument. At 10 billion impressions per day and a \$10 eCPM, daily revenue is $10^{10}/1000 \times \$10 = \$100$ million per day, roughly \$36.5 billion per year. A 1% systematic calibration error is therefore about \$365 million per year. That single sentence is usually the moment the interviewer starts nodding, and it is why calibration monitoring is a paging alert rather than a dashboard.

**Storage.** Embedding tables dominate. A hashed table of $2^{24}$ rows at 64 dimensions in fp16 is $1.68\times10^7 \times 64 \times 2 \approx 2.1$ GB per feature field; with 30 sparse fields that is roughly 64 GB of embedding parameters, which does not fit on a single accelerator and must be sharded — the parameters are partitioned across hosts while the dense MLP is replicated on each, which is exactly the hybrid parallelism DLRM was designed around. Budget and pacing state is small but hot: 10 million active campaigns at 200 bytes is 2 GB in an in-memory store, read and written on every request, which makes it a genuine hot-spot requiring sharding by campaign and careful handling of counter contention on popular campaigns.

**Latency budget** at p99, summing under 50 ms: targeting and eligibility 10; retrieval 8; ranking 20; calibration 1; pacing 2; auction 2; logging and response 4. Total 47 ms.

**Failure modes, in order of how much they hurt.**

*Calibration drift* is the top one. It happens silently after any distribution shift — a seasonal change, a new placement, a large advertiser entering — and produces no errors, no latency change, and a steady revenue leak. Defense: monitor predicted versus actual click rate continuously per segment, alert on deviation beyond a tight band, and refit the calibration layer far more frequently than the model itself, since refitting two Platt parameters on recent data is cheap and safe.

**Monitoring.** Infrastructure: QPS, per-stage latency, eligibility-set sizes (a sudden collapse means a targeting index bug and is invisible in latency metrics), pacing-state write latency. Model: calibration per segment, prediction distribution drift, feature drift, coverage of the ad inventory, and the fraction of auctions cleared at the reserve price, which rises when competition thins. Business: eCPM by segment, revenue, delivery rate, advertiser-side cost per acquisition, and user-side hide and report rates.

### The hard tradeoff

**Should the auction rank purely by expected revenue, or should it discount for user experience?**

Pure eCPM ranking is defensible and clean. It maximizes short-run revenue by construction, it is transparent to advertisers — the highest expected value wins, and everyone can reason about how to compete — and it avoids the platform imposing an opaque quality judgment that advertisers cannot see or contest. Any quality multiplier is a thumb on the scale whose weight the platform sets unilaterally, and advertisers reasonably distrust it.

Discounting for user experience is also defensible. Ads impose a real cost on the product: an irrelevant or irritating ad reduces the probability that the user comes back, and that cost is paid by the platform in future impressions, not by the advertiser who caused it. Pure eCPM ranking makes that externality invisible — a high-bidding, low-quality advertiser can outbid a low-bidding, high-quality one and degrade the feed for everyone. Including a quality term internalizes the cost, and it also improves long-run revenue, because feed quality determines how many impressions exist at all.

I would include the quality term, and I would be specific about how, because the vague version of this answer is weak. Estimate the user cost of showing an ad in currency terms — the expected reduction in future sessions, converted to expected lost revenue — and subtract it from eCPM rather than multiplying by an unitless fudge factor. That makes the trade explicit and auditable, and it lets you tell an advertiser exactly why their ad was discounted. Publish a quality score to advertisers so it is actionable rather than mysterious.

What would change my mind: a long-run holdback showing that ad quality has no measurable effect on retention. If users tolerate bad ads without reducing engagement, the externality is imaginary and the quality term is just revenue foregone. My prior is strongly the other way, but it is an empirical question and the holdback is how you answer it.

---

## Design 4 — Fraud detection at a payment processor

### Requirements and assumptions

**Assume for the rest:** processor-side, card-not-present, merchant bears chargeback losses. Inline decision with a 100 ms budget at p99. Actions are approve, decline, or step-up challenge, plus an asynchronous review queue for high-value cases. Volume assumptions below. Fraud attempt rate of roughly 0.1% of transactions by count — I am treating this as an assumption; published figures are usually expressed by *value*, where US card-not-present fraud has run on the order of 10-20 basis points of transaction value in recent Federal Reserve data.

**Functional requirements.** The system must apply hard rules, compute real-time velocity and graph features, produce a calibrated fraud probability, choose approve, step-up, or decline with reason codes, update state, support human review, join delayed chargebacks, and collect unbiased audit data.

**Non-functional requirements.** The system must stay below 100 ms at p99, support about 12,000 peak transactions per second, remain highly available, keep streaming features fresh within seconds, protect payment data, support audit and explanation, and degrade conservatively.

### Step 2 — Frame it as an ML problem

The prediction target is straightforward to state and subtle to define: given everything known at authorization time, estimate $p(\text{fraud} \mid \text{transaction})$. The unit is a single authorization request. The subtlety is in "fraud," which is not one thing. Stolen-card fraud, account takeover, friendly fraud where the legitimate cardholder disputes a purchase they made, merchant collusion, and card testing — where an attacker runs thousands of tiny transactions to find which stolen numbers still work — have different signatures and different costs. A single binary model conflates them. A reasonable answer is one primary model with fraud-type as an auxiliary multi-class head, so you get a single score for the decision and a type prediction for routing and for the human reviewers.

**The label.** Ground truth arrives as a chargeback: the cardholder disputes the transaction, the issuer reverses it, and you find out. Chargebacks arrive on a long tail — a large share within 30 days, most within 90, some out past 120. Three consequences.

*The label is delayed*, exactly like the ad conversion problem in Design 3 but with a longer horizon. Any recent transaction labeled "not fraud" is only provisionally so. If you train on the last 30 days and treat unlabeled as legitimate, you systematically underestimate fraud, and the underestimate is concentrated in the most recent data. Handle it the same way: train the main model on a matured window, and if you need recency, weight recent negatives by the probability that a fraudulent transaction of that age would already have been disputed.

*The label is incomplete in a specific direction.* Not all fraud produces a chargeback — some victims never notice small amounts, and some merchants refund proactively to avoid the dispute fee, which resolves the customer's complaint and destroys your label. So your negatives include real fraud, and your measured fraud rate is a lower bound.

*The label is contaminated by your own decisions*, and this is the one to lead with because it is the deepest problem in the design. You only observe outcomes for transactions you approved. The ones you declined have no outcome — you will never know whether they were fraud. So your training data is a biased sample: it is exactly the set of transactions your current model thought were fine. Train on it naively and the new model learns to reproduce the old model's decision boundary, and it will look excellent on held-out data drawn from the same biased pool while being blind in the region the old model refused to enter. This is **selection bias**, and in credit and fraud it is classically called the **reject inference** problem: how do you learn about the rejected population when you have no labels for it?

With three actions the same logic yields two thresholds: below $\tau_1$ approve, between $\tau_1$ and $\tau_2$ step up, above $\tau_2$ decline — where the step-up band exists because a challenge has a small cost (some legitimate users abandon) and a large benefit (it stops most fraud), so it wins in the middle region where neither approving nor declining is clearly right.

### Step 3 — The data

**Transaction features** are the obvious ones and the weakest: amount, currency, merchant identifier, merchant category code, time, whether the card details were entered manually or from a stored token, the AVS and CVV verification results returned by the issuer, and the BIN, which identifies the issuing bank and card type.

**Velocity features** are the most predictive family in fraud detection, and they need a proper definition since the term is used constantly. A velocity feature is a count, sum, or distinct-count of events over a recent time window, keyed on some entity. Not "the amount of this transaction" but "how many transactions has this card attempted in the last 60 seconds," "how many distinct merchants has this device touched in the last hour," "what is the total amount charged to this card in the last 24 hours versus its 30-day average," "how many distinct cards has this IP address used today." They work because fraud is nearly always a *rate* phenomenon rather than a per-transaction one. A single stolen-card purchase looks exactly like a legitimate purchase; what betrays it is that the same card was tried at four merchants in ninety seconds, or that this device has cycled through fifteen cards this morning. Card testing is invisible per transaction and blindingly obvious in velocity space.

The entities to key on, and this list is worth reciting: card, device fingerprint, IP address, email address, billing address, shipping address, merchant, and the pairs among them. The windows to compute: something like 1 minute, 5 minutes, 1 hour, 24 hours, 7 days, 30 days. The aggregations: count, distinct count, sum, max, and the ratio of a short window to a long one, which is what actually detects a change in behaviour rather than a level.

Two things about velocity features that separate a good answer from a shallow one. They are the hardest part of the infrastructure, because they must be *current to the second* — a velocity feature computed from a batch job that ran an hour ago cannot detect card testing that started five minutes ago, which is the entire use case. And they are the features most vulnerable to leakage, discussed below.

**Historical and behavioural features** cover the entity's normal: card age with this processor, the merchant's own historical fraud rate, whether this transaction's amount and category are typical for this card, the geographic distance and elapsed time from the previous transaction — the "impossible travel" signal, where a card used in New York and then in Singapore twenty minutes later is physically implausible.

### Step 4 — The architecture

An authorization request arrives from a merchant's server. One hundred milliseconds, and the clock includes everything.

**Stage zero: deterministic rules.** Before any model, a rules engine checks the things that are absolute — cards on a confirmed-compromised list, sanctioned entities, merchant-configured hard blocks, obviously malformed requests. These exist for three reasons: they are certain, they are auditable in a way a model is not, and they are the mechanism by which a human can respond to an attack in minutes rather than waiting for a retrain. Under 1 ms, in-memory. Say explicitly that rules and models coexist permanently in fraud systems; a candidate who proposes replacing all rules with a model has not operated one.

**Stage one: feature retrieval.** This is the latency-critical component and the one that determines whether the design is real. Velocity features are read from an online store — Redis or equivalent, covered properly in Part 1 — keyed by card, device, IP, and email. The counters behind them are maintained by a streaming pipeline: every authorization event is published to a log-structured event bus (Kafka, again Part 1), a stream processor (Flink or equivalent) maintains windowed aggregates per key, and it writes the current values into the online store. The critical property is that the write path must complete fast enough that the *next* transaction on the same card sees the update — card testing runs at seconds per attempt, so a pipeline with 30 seconds of lag is blind to precisely the attack it exists to catch. Design for end-to-end freshness under a couple of seconds, and monitor that freshness as a first-class metric.

A practical refinement worth mentioning: for the very shortest windows, do not rely on the streaming round trip at all. Keep a small in-process or co-located counter updated synchronously on the request path for the 1-minute and 5-minute windows, and use the streaming store for longer windows where a few seconds of lag is immaterial. This is the difference between a design that works against card testing and one that does not.

Budget 20 ms for all feature reads, issued in parallel across key types.

**Stage two: the model.** Use gradient-boosted decision trees — recall from Design 2 that GBDTs build an additive ensemble of small trees, each fitted to the residual of the ones before. They are the right choice here for reasons you should be able to list. The features are tabular and heterogeneous, mixing counts, amounts, ratios, and categories, which is exactly the regime where trees beat neural networks empirically. Missing values are pervasive — a new card has no history, a new device has no graph — and GBDTs handle missingness natively by learning a default direction at each split rather than requiring imputation, which matters enormously because *missing is informative* here. Inference is fast: a few hundred shallow trees score in single-digit milliseconds on CPU with no accelerator. And the model is inspectable: you can extract the features that drove a decision, which the review queue needs and which regulators may require.

Add a second model where it earns its place rather than everywhere. A sequence model over the card's recent transaction history captures patterns that flat aggregates miss — the *shape* of a spending sequence rather than its summary statistics. A graph neural network catches ring structure. Both are best run asynchronously, scoring after the inline decision and feeding the review queue and the next model refresh, because their latency does not fit in the inline budget and their value is in catching what the fast model missed.

Budget 8 ms.

**Class imbalance handling**, since this is where the interviewer will push. Four techniques, and the important thing is knowing what each costs.

*Class weighting* multiplies the loss on positive examples by some factor, typically the inverse class ratio, so the optimizer stops ignoring them. Cheap and effective. It distorts the output scale, so the model no longer predicts calibrated probabilities.

#### Architecture diagram

```
              authorization request (card, amount, merchant, device, ip, ...)
                                     |
                                     v
                  +--------------------------------------+
                  |  RULES ENGINE                        |   <1 ms
                  |  blocklists, sanctions, merchant     |
                  |  hard blocks -> immediate decline    |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |  FEATURE RETRIEVAL (parallel)        |   ~20 ms
                  |  velocity by card / device / ip /    |
                  |  email  (online KV store)            |
                  |  graph neighbourhood stats           |
                  |  card + merchant history             |
                  |  in-process counters for 1-5 min     |
                  +--------------------------------------+
                                     ^
                                     |  written by
                  +--------------------------------------+
                  |  STREAMING AGGREGATION               |
                  |  event bus -> stream processor ->    |
                  |  windowed counters -> online store   |
                  |  target freshness < 2 s              |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |  GBDT SCORER                         |   ~8 ms
                  |  few hundred shallow trees, CPU      |
                  |  native missing-value handling       |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |  CALIBRATION                         |   ~1 ms
                  |  undo downsampling; isotonic on      |
                  |  true-distribution holdout           |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |  COST-SENSITIVE DECISION             |   ~3 ms
                  |  thresholds tau1, tau2 as functions  |
                  |  of amount and merchant costs        |
                  |  -> approve / step-up / decline      |
                  |  + reason codes                      |
                  +--------------------------------------+
                                     |
                        response to merchant  (~35 ms typical)
                                     |
                                     v
                  +--------------------------------------+
                  |  ASYNC: heavy models rescore,        |
                  |  graph edges updated, review queue,  |
                  |  randomized-approval slice logged,   |
                  |  chargebacks joined weeks later      |
                  +--------------------------------------+
```

### Step 5 — Evaluation

Do not report accuracy, and say why: at a 0.1% base rate, predicting "legitimate" always yields 99.9% accuracy. Do not lead with ROC-AUC either, and this reason is subtler and worth knowing. ROC-AUC uses the false positive rate, whose denominator is the number of negatives — an enormous number — so a change from 10,000 to 20,000 false positives barely moves it while doubling the operational burden. Under heavy imbalance ROC-AUC is optimistically insensitive.

Use the **precision-recall curve** and the area under it, because precision's denominator is the number of *predicted positives*, which is small, so the metric is sensitive to exactly the errors that matter. But even PR-AUC integrates over thresholds you would never use. In production you care about specific operating points, so report the concrete pair: recall at a fixed false-positive budget ("what fraction of fraud do we catch while declining at most 0.5% of legitimate transactions"), and precision at a fixed recall.

The metric that actually decides things is **cost-weighted**: total dollars of fraud approved plus total dollars of legitimate volume declined times the friction cost, evaluated at the chosen thresholds. This is the only number that maps to the business, and it is the number that will differ from the statistical ones — a model with worse PR-AUC can be better in dollars if its errors are concentrated on small transactions.

What offline metrics miss here is unusually severe. They are computed on approved transactions, so they measure the model on the population the current system lets through; performance in the declined region is unmeasured and unmeasurable without the randomized slice. They use matured labels, so they are evaluating on an attack landscape weeks old. And they cannot see the adversarial response: a model that would provoke fraudsters into a more damaging strategy scores the same offline as one that would not.

### Step 6 — Production concerns

**Throughput.** Take \$1.9 trillion of annual volume, which is Stripe's publicly reported 2025 figure, and an average transaction of \$50 — that average is my assumption and it swings the result, so state it. Then annual transaction count is $1.9\times10^{12} / 50 = 3.8\times10^{10}$, and the average rate is $3.8\times10^{10} / 3.15\times10^{7}\ \text{s} \approx 1{,}200$ transactions per second. Peaks are far above average in payments — Black Friday and Cyber Monday, plus the diurnal cycle — so a 10x peak factor gives roughly 12,000 transactions per second to design for. At 0.1% fraud that is about 12 fraudulent attempts per second at peak, and at 10 basis points of value the annual fraud exposure on \$1.9T is about \$1.9 billion, which is the number that justifies the whole system's budget.

**Feature-store load.** Each transaction reads velocity features for perhaps 5 entity keys across 6 windows and several aggregations, which is a handful of multi-get operations, and it writes updates for the same keys. At 12,000 TPS that is on the order of 60,000 reads and 60,000 writes per second against the online store — well within a sharded in-memory store's capability, but it means the store is on the critical path for availability and must be replicated. Hot keys are a genuine problem: a large merchant's own key is touched by a substantial share of all traffic, so shard by a composite key or maintain merchant-level aggregates separately with relaxed consistency.

**Latency budget** at p99, inside 100 ms: network ingress and parsing 5 ms; rules 1 ms; feature retrieval 20 ms (parallel across key types, so this is the slowest one); model inference 8 ms; calibration 1 ms; decision and reason codes 3 ms; logging and response 5 ms. Total 43 ms, which leaves substantial headroom — and you should keep it, because the tail here is not a quality issue but a timeout, and a timeout is usually treated as a decline. Deliberate over-provisioning is the correct posture in this system in a way it is not in a feed.

**Failure modes.**

*Adversarial probing.* Attackers test the boundary with small transactions to learn what passes. Mitigations: rate-limit per entity independently of the model, since probing is itself a velocity signature; add controlled randomness near the boundary so the response is not perfectly learnable; and never expose granular decline reasons to the transaction initiator, only to the merchant through an authenticated channel.

**Monitoring.** Infrastructure: TPS, per-stage latency with a hard alert on p99 approaching the timeout, feature-store availability, and — most important — feature freshness lag as a paging metric. Model: score distribution drift and per-feature drift measured by population stability index, with the conventional reading that PSI above 0.1 warrants investigation and above 0.25 indicates a substantial shift; calibration plots on matured data; missing-feature rates, since a rise means an upstream pipeline broke. Business: decline rate overall and per segment, step-up rate and pass rate, authorization rate, chargeback rate as it matures, dollars of fraud approved, and review-queue volume and agreement rate between reviewers and the model.

### The hard tradeoff

**Should the inline decision use a fast, interpretable GBDT, or a heavier model — a sequence or graph network — that catches more sophisticated fraud?**

The GBDT case is strong and mostly practical. It fits the latency budget with room to spare, which matters because exceeding the timeout converts into declines and the failure mode is worse than a slightly weaker model. It handles missing features natively, and missingness is everywhere in this data. Its decisions can be attributed to features, which the review queue needs and regulators may demand. It retrains in minutes on commodity hardware, which is what makes daily — or faster — retraining operationally realistic against an adapting adversary. And empirically, on tabular features with good velocity and graph aggregates, it is very hard to beat.

The heavier-model case is also real. Sophisticated fraud shows up in structure that flat aggregates flatten: the ordering and timing of a card's recent transactions, the shape of a ring in the entity graph. A sequence model sees "three small probes then a large purchase" as a pattern; the aggregate sees a count of four. A GNN propagates fraud evidence across a ring, so a card whose device is two hops from confirmed fraud is flagged even with a clean individual history. These are exactly the cases with the largest dollar losses, because organized fraud is where the money is.

The resolution I would argue for is not to choose but to **place them differently in time**. GBDT inline, because the inline path is latency-bound and interpretability-bound. Heavy models asynchronous, scoring within seconds after the decision, where they can trigger a capture cancellation before settlement, feed the review queue, and — crucially — generate features that the inline GBDT consumes on the *next* transaction. That last point is the trick worth stating: a graph model's output becomes a precomputed node score, which is just another input to the fast model, so the inline path benefits from graph reasoning without paying its latency.

---

## Design 5 — Content moderation at scale

### Requirements and assumptions

**Assume, for the rest of this answer:** one billion items per day, roughly 70% text, 25% image, 5% video; about thirty policy categories; actions are allow / demote / age-gate / remove / escalate-to-human; text is moderated in-line with a 150 ms budget, image and video asynchronously within about 60 seconds of upload; a human review capacity of 250,000 items per day; appeals exist and their outcomes are logged; forty languages with region-specific policy overlays.

**Functional requirements.** The system must detect policy violations across modalities, match known-bad content, produce per-policy scores, select an action, route uncertain or severe cases to humans, support appeals, update policies quickly, and log decisions with policy and model versions.

**Non-functional requirements.** The system must process about 35,000 items per second at peak, keep text below 150 ms at p99, finish media checks within 60 seconds, limit human routing to 0.025%, support 40 languages, resist adversarial inputs, and provide auditable and reversible actions.

### Step 2 — Frame it as an ML problem

The naive framing is multi-label classification: input is a piece of content plus its context, output is a probability per policy category, loss is per-head binary cross-entropy. That framing is correct and you should say it, but it is not the whole frame, and the interviewer is waiting to see whether you notice the two things it leaves out.

The first is that the *output of the model is not the output of the system*. The model emits thirty calibrated probabilities. The system emits one of five actions. The map between them is a policy engine: a per-category threshold table plus precedence rules (removal beats demotion; a CSAM hit short-circuits everything; a verified-news-publisher account gets a mandatory human step before removal). Keeping that map outside the model is a deliberate architectural choice, because policy changes weekly and models retrain monthly, and you do not want a policy change to require a training run. Say that explicitly.

The second is that this is a **cascade under a budget**, not a single classifier. Formally, you have tiers with increasing cost and increasing accuracy, and you are choosing, per item, how far down the cascade to send it, subject to a total-cost constraint. If tier $i$ costs $c_i$ per item and you route a fraction $f_i$ of traffic to it, your daily cost is $N \sum_i f_i c_i$ and the human tier's $f$ is pinned by headcount. Framing it as constrained routing rather than "we run some models" is what makes the numbers in Step 6 fall out naturally.

**Calibration** deserves a definition since the whole cascade depends on it: a model is calibrated when a predicted probability of 0.8 means that, among all items scored 0.8, about 80% really are violations. Uncalibrated scores are still fine for ranking but useless for thresholding, and thresholds are how policy talks to your system. You calibrate per policy head on a held-out set, typically with isotonic regression (a monotone step function fit to map raw scores onto empirical frequencies) or Platt scaling (fitting a one-dimensional logistic on the scores), and you re-fit calibration far more often than you retrain the model, because calibration drifts fast and re-fitting it is cheap.

### Step 3 — The data

Labels here are unusually hard, and the reason is worth stating plainly because it bounds everything else: **your model cannot be more accurate than your labels, and moderation labels genuinely disagree.** The measure of this is **inter-annotator agreement** — take the same items, give them to several trained reviewers independently, and measure how often they agree, usually with Cohen's or Fleiss' kappa, which corrects for agreement you'd get by chance. Spam and nudity come in high, often above 0.9. Hate speech and harassment, on published academic and industry numbers, routinely land in the 0.6–0.8 range even among trained reviewers with a written policy in front of them. Assume for this design that hate speech agreement is around 0.7. That has a hard consequence: if two trained humans disagree on 25% of borderline hate-speech items, a model that appears to be 85% accurate against single-annotator labels may be at ceiling already, and the way to improve it is to fix the policy document and the reviewer training, not the architecture. Saying that out loud is a strong senior signal.

Your label sources, in order of volume and inversely in order of quality:

The **human review queue** is your gold set — items a reviewer explicitly adjudicated against a policy. High quality, low volume, and critically it is a *biased sample*: it contains only things the system was already suspicious about, so a model trained purely on it never learns what ordinary content looks like. You fix that by always mixing in a **random audit sample** — a few thousand items per day drawn uniformly from all content, labelled by reviewers regardless of model score. That random sample is expensive and you will be tempted to cut it. Don't: it is the only unbiased estimate of your true recall in production, because it's the only place you see violations the model never flagged.

**Appeals outcomes** are a second gold source with a useful property: an overturned removal is a confirmed false positive, which is the error type you're otherwise blind to. They're also biased — users who appeal skew toward the confident and the English-speaking — so weight them, don't just concatenate.

**User reports** are high volume and low precision. People report things they dislike. Treat a report as a feature (a strong routing signal) rather than a label, and note that reports are gameable: brigading, where a coordinated group mass-reports a target, is a standard attack.

Two techniques bridge the gap between the volume you need and the labels you can afford.

### Step 4 — The architecture

Walk the path of one post. A user hits Submit on a text post with an attached image.

The request first meets the **hard-rule gate**. This is not machine learning: exact-match blocklists, banned URLs and domains, hashes of previously-removed content from this same user, and a small set of regex rules that Trust & Safety owns directly. It runs in under a millisecond against an in-memory hash set, and it exists for two reasons — it's free, and it gives policy a lever they can pull in five minutes during an incident without waiting for a model deploy. Roughly 1–2% of submissions die here, mostly spam.

Next is **hash matching**, which handles known-bad content. The concept you need to define here is **perceptual hashing**. A cryptographic hash like SHA-256 changes completely if you flip one pixel, which makes it useless against an adversary who re-saves the JPEG. A perceptual hash is instead designed so that visually similar images produce *similar* hashes: you downscale the image to something tiny, convert to greyscale, take a frequency transform (a discrete cosine transform in the classic pHash construction), and emit a bit-string from the sign pattern of the low-frequency coefficients — the coefficients that survive resizing, re-compression and mild colour shifts. Two images match if the Hamming distance between their hashes (the number of differing bits) is below a threshold. Microsoft's PhotoDNA is the industry-standard variant for CSAM and works on similar principles over image gradients; matching against NCMEC-supplied hash sets is standard practice and in many jurisdictions effectively mandatory. Video gets the same treatment per sampled frame, plus audio fingerprinting for soundtrack matching. The engineering point: this is a nearest-neighbour lookup under Hamming distance over a set of tens of millions of hashes, which you serve with a multi-index hash table or a small ANN structure, and it comes back in well under 50 ms. The failure mode to name is that perceptual hashes are robust to *re-encoding* but not to *re-composition* — crop it, mirror it, overlay it on a meme template, and the hash moves. Hash matching catches redistribution, not novelty. That's the handoff to the next tier.

Then the **fast classifier ensemble**, which is where most of the actual decisions get made. Architecturally this is one shared encoder per modality with a linear head per policy category. For text, a fine-tuned encoder in the RoBERTa/DeBERTa family, distilled down to something in the 20–60 M parameter range so it runs on CPU at scale or on a small GPU fleet with room to spare. For images, a vision transformer initialized from a **CLIP**-style checkpoint — I define CLIP properly in Design 7; for now it's an image encoder pretrained on hundreds of millions of image-caption pairs, which means its features already encode semantic content rather than just texture, so a linear probe on top learns a new policy from a few thousand examples. For video, sample frames (a few per second, plus scene-change detection so you don't miss a two-second clip inside a ten-minute upload), embed each, and aggregate with attention pooling over the frame sequence. For audio, transcribe with a Whisper-class ASR model and route the transcript into the text classifier, plus a direct audio embedding for things transcription destroys, like a gunshot or a specific piece of copyrighted music.

Items the ensemble is confident about — score far from the threshold in either direction — are actioned immediately. Items in the uncertain band go on.

#### Architecture diagram

```
 user submits post (text + image)
        │
        ▼
 ┌──────────────────────────────┐
 │ 1. Hard-rule gate            │  blocklists, banned URLs, regex     <1 ms
 │    ~1-2% blocked             │  policy-owned, no model deploy
 └──────────────────────────────┘
        │ pass
        ▼
 ┌──────────────────────────────┐
 │ 2. Perceptual-hash match     │  PhotoDNA / pHash / audio prints   <50 ms
 │    ~3-5% matched             │  known-bad redistribution
 └──────────────────────────────┘
        │ no match
        ▼
 ┌──────────────────────────────┐
 │ 3. Fast classifier ensemble  │  shared encoder + 30 policy heads
 │    text  : distilled RoBERTa │  text  ~15 ms CPU/GPU
 │    image : CLIP ViT + heads  │  image ~40 ms GPU
 │    video : frames + pooling  │  video  async
 │    fusion: text+image head   │  cross-modal cases
 └──────────────────────────────┘
        │
        ├── confident violation ──► ACTION (remove / age-gate / demote)
        ├── confident benign ─────► ALLOW
        │
        │ uncertain band (~2-5% of items)
        ▼
 ┌──────────────────────────────┐
 │ 4. LLM judge (multimodal)    │  policy text in prompt, + context
 │    structured verdict +      │  ~500 ms - 2 s, ASYNC only
 │    rationale                 │  reads policy, not labels
 └──────────────────────────────┘
        │ still uncertain, or severe, or high-reach
        ▼
 ┌──────────────────────────────┐
 │ 5. Human review queue        │  priority = severity x uncertainty
 │    ~0.02-0.05% of items      │           x exposure / review-time
 └──────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────┐
 │ Policy engine                │  per-category thresholds,
 │ → allow / demote / age-gate  │  precedence rules, region overlays
 │   / remove / suspend         │  OWNED BY TRUST & SAFETY, not ML
 └──────────────────────────────┘
        │
        ▼
   labels from (4) and (5) ──────► training set  (active-learning loop)
   appeals outcomes ─────────────► false-positive corpus
   random audit sample ──────────► unbiased recall estimate
```

### Step 5 — Evaluation

Offline, you evaluate **per policy category, never in aggregate.** An aggregate F1 across thirty categories is dominated by spam and tells you nothing about the categories anyone cares about. For each category you report precision at the operating threshold, recall at that threshold, and the full precision-recall curve so that Trust & Safety can see what recall costs at each precision. The headline number to quote per policy is *recall at the precision they require* — "at 0.95 precision, which is what policy demands for auto-removal on harassment, we get 0.61 recall" is a sentence a T&S director can act on, and "our AUC is 0.94" is not.

The critical measurement problem is that **you cannot estimate recall from the escalation queue**, because the queue only contains items the model already suspected. Recall requires knowing about violations you never flagged, and the only way to see those is the random audit sample. So: draw a few thousand items per day uniformly at random, have reviewers label them fully against all policies, and estimate recall from that stratum. It's expensive and it's non-negotiable. To get a usable estimate for rare categories without labelling millions of items, use stratified sampling — oversample the high-score strata, then reweight by the inverse of each stratum's sampling probability — which gets you a variance-efficient unbiased estimate for a fraction of the labelling cost.

### Step 6 — Production concerns

**Scale arithmetic.** A billion items a day is $10^9 / 86400 \approx 11{,}600$ items per second on average. Traffic isn't flat; assume a peak-to-average ratio of about 3, so provision for roughly 35,000 items per second. Split by the modality mix: about 24,000/s text, 8,700/s image, 1,700/s video at peak.

**Latency budget** for the in-line text path, against a 150 ms allowance:

**Failure modes**, and what each one does to you. If the *image GPU fleet degrades*, you must not fail open on severe categories; the correct degradation is to keep hash matching (cheap, CPU) running and hold un-classified images in a Kafka backlog while auto-demoting them, so that reach is suppressed until you catch up. If the *LLM judge is down*, uncertain items route to the demote action and into the human queue at a higher rate — you accept a queue backlog rather than accept unreviewed exposure. If *someone changes a threshold badly*, you get a mass false-positive event within minutes; the mitigation is that threshold changes go through the same canary machinery as model changes, plus a rate limiter on total removals per hour per category that trips an alarm rather than silently removing four million posts. If *the human queue backs up*, severity-ordered prioritization means the backlog accumulates in low-severity categories, which is the correct place for it to accumulate — but you need an explicit queue-age SLO per severity tier so you find out. And the one people forget: *a training-data feedback loop*. If you only train on items your system escalated, you learn your own system's blind spots as truth, and your measured recall improves while your real recall degrades. The random audit sample is the circuit-breaker; treat it as load-bearing infrastructure.

**Monitoring.** Infrastructure metrics — queue depth, stage latency, GPU utilization, throughput — come from **Prometheus** (a time-series database that scrapes numeric metrics from your services on an interval) and get displayed and alerted on in **Grafana** (a dashboarding and alerting layer on top of it); Part 1 covers both, and the thing to say here is only that every stage in the cascade emits its own routing rate, because a shift in routing rate between tiers is the earliest signal that something upstream changed. Model metrics — per-policy score distributions, calibration error, escalation rate per category — are computed continuously and compared against the previous week, because a score distribution that shifts without a deploy means the *input* distribution shifted. Business metrics — prevalence, time-to-action, appeal and overturn rates — are computed daily off the audit sample. The single highest-value alarm in the whole system is a jump in a category's escalation rate, because it fires within minutes of a new attack campaign starting and it needs no labels to compute.

**Retraining triggers.** Retrain on a fixed cadence — weekly for the fast heads, since labels accumulate daily and the heads are cheap to fit; monthly or on-demand for the shared encoders. Retrain off-cadence when any of these fire: prevalence rises in a category for three consecutive days; the escalation rate in a category moves more than 30% week over week; a new policy category launches; calibration error on the audit set exceeds its threshold; or a coordinated campaign is detected. Refit calibration weekly regardless, since it's minutes of compute and it's what the thresholds depend on.

### The hard tradeoff

The real one is **speed of action versus certainty of action**, and it is genuinely unresolvable — you're choosing between two different kinds of harm to two different sets of people.

Act fast and automatically, and violating content comes down within seconds, before it accrues views. That's the whole point: harm from content is roughly proportional to exposure, and exposure is roughly exponential in the first hour. But automatic action at speed means acting on model scores, and model scores are wrong on the tail, so you will remove legitimate content — journalism that documents violence, medical discussion that trips a self-harm classifier, reclaimed slurs used inside the community that owns them, satire. Those errors land disproportionately on marginalized users whose speech patterns are underrepresented in training data, which turns a modelling artifact into a fairness failure with a press cycle attached.

Act slowly and carefully, route everything ambiguous to humans, and your false-positive rate collapses. But your queue is three days deep — which is exactly where this scenario started — and content that should have come down in ninety seconds is up for seventy hours, and by then the harm is done and the removal is theatre.

The resolution isn't to pick one. It's to make the choice *per policy and explicit*, and to use the graded action space to escape the binary. Categories with catastrophic and irreversible harm and high annotator agreement — CSAM, credible violent threats, terrorist recruitment — get automatic action at aggressive recall, accepting false positives, with fast human appeal as the correction mechanism. Categories with contested boundaries and reversible harm — hate speech, misinformation, harassment — get a cheap intermediate action: *demote* rather than remove. Demotion cuts exposure by an order of magnitude, which captures most of the harm reduction, while remaining nearly invisible to the user if you got it wrong and being instantly reversible if you did. Reserve removal for the confident tail and the human-adjudicated tail. The sentence to say is: "the existence of a low-cost reversible action is what lets me be aggressive on recall without paying the full false-positive price, and if the product only lets me remove or allow, I'd push back on that constraint before I touched the model."

---

## Design 6 — An LLM serving platform (internal, multi-team, multi-model)

### Requirements and assumptions

**Assume, for the rest of this answer:** a mix of about 70% interactive chat and agentic traffic and 30% batch that tolerates minutes of delay; three base models — an 8 B, a 70 B, and one large mixture-of-experts model accessed through a vendor API rather than self-hosted — plus roughly twenty LoRA adapters over the 8 B; targets of TTFT under 1 s at p95 and ITL under 50 ms at p95 for interactive, no target for batch; peak-to-average around 4 with a predictable daily shape; reserved H100 capacity plus cloud burst; tenants may share batches but prompt caches are isolated per tenant by default.

**Functional requirements.** The platform must authenticate tenants, enforce request and token quotas, route model versions and adapters, stream tokens, batch requests, manage KV cache, support vendor and self-hosted models, record usage, and deploy versions safely.

**Non-functional requirements.** The platform must meet TTFT and ITL SLOs, isolate tenants, prevent one tenant from exhausting capacity, maximize GPU utilization, protect prompts and caches, support graceful overload, provide accurate billing, and enable rollback.

### Step 2 — Frame it as an ML problem

It mostly isn't one, and saying so confidently is the right move. There is no training here and no labels. The framing is: **this is a scheduling and capacity-allocation problem over a heterogeneous, non-preemptible, memory-constrained accelerator pool, with per-tenant fairness and latency constraints.** The ML content is in knowing what the workload does to the hardware.

There is one genuine modelling problem hiding inside it, and volunteering it scores points: **output-length prediction**. The scheduler would make much better decisions if it knew how many tokens a request will generate, since that determines how long the request will hold KV-cache memory. You don't know it in advance. You can predict it — a small regressor on the prompt's embedding, the requested model, the tenant, and the stated `max_tokens`, trained on your own request logs — and get a usable estimate. That prediction feeds admission control (don't admit a request you predict will hold 8 K tokens of cache when you're at 90% memory) and lets you approximate shortest-job-first scheduling, which is the policy that minimizes average waiting time. Get the prediction wrong in the optimistic direction and you preempt; that's a recoverable error, so bias the estimator to over-predict.

The objective to optimize is **goodput**, not utilization, and the distinction is the most important conceptual point in this design. **GPU utilization** as reported by `nvidia-smi` measures the fraction of time at least one kernel was resident on the device. It is close to useless: a GPU running a batch of size 1 shows near-100% utilization while doing perhaps 2% of the useful work it could. **Goodput** is requests (or tokens) completed *within their latency SLO* per unit time. The gap between them is where all the money is. You can always raise throughput by batching harder, but past a point the added queueing pushes requests past their SLO, and tokens delivered too late to a chat user are worth zero. So the objective is: maximize tokens delivered inside SLO per dollar-hour of GPU. Say that sentence. The better proxy metric to actually watch is **MFU**, model FLOPs utilization — achieved FLOPs divided by the hardware's peak — which for well-tuned prefill lands around 40–50% and for decode is intrinsically low because decode is memory-bound, not compute-bound.

### Step 3 — The data

There's no training corpus, but there are three data assets the platform must produce, and forgetting them is a common failure in this answer.

**Request telemetry** is the primary one, and it has to be complete enough to bill against: per request, the tenant and team, the model and adapter, input token count, output token count, cached-prefix token count, queue wait, TTFT, total latency, which worker served it, and the outcome. This is what powers chargeback, capacity forecasting, the output-length predictor, and every debugging session you will ever have. Emit it to **Kafka** (the distributed append-only event log; Part 1 covers it) and land it in a columnar warehouse. The critical detail is that token counts must be measured, not estimated from character counts, since billing correctness depends on it and teams will audit you.

**Prompt and response logging** is a policy minefield and you should raise it unprompted. Logging full prompts makes debugging and evaluation possible and makes you a liability under any privacy regime; not logging them makes the platform very hard to operate. The workable answer is per-tenant opt-in with short retention, redaction of detected secrets and PII on ingest, and a strict separation between the metrics path (always on, no content) and the content path (opt-in, encrypted, 7-day TTL).

**Evaluation data** exists because a serving platform still needs to answer "did this change break anything?" — a quantization change, an engine version bump, a speculative-decoding config, or a vendor model version rotating under you. You need a **golden set** of a few thousand representative requests per model, replayed on every change, scoring both output quality and latency. Which brings us to **automated evaluation**: a pipeline that runs candidate outputs against reference expectations without a human in the loop, typically mixing exact-match or unit-test-style checks for structured outputs, similarity metrics for free text, and LLM-as-judge scoring for open-ended quality. Part 1 covers the mechanics; the platform-specific point is that this pipeline is the *gate* on any change that could alter numerics, and quantization is exactly such a change.

### Step 4 — The architecture

Follow a single chat request through the system.

It arrives at the **API gateway**, which authenticates the caller and resolves it to a tenant. Authentication is a token lookup; the tenant identity is what everything downstream keys on. The gateway then applies **rate limiting**, and the standard mechanism is the **token bucket**: each tenant has a conceptual bucket that refills at a fixed rate $r$ (say 100 requests per second) up to a capacity $b$ (say 500). A request takes one token from the bucket; if the bucket is empty the request is rejected with a 429. The reason this specific algorithm and not a simple counter is that the bucket's capacity $b$ allows a *burst* — a tenant that's been quiet can spend accumulated tokens all at once — while the refill rate $r$ bounds the sustained rate. That matches how real clients behave. For LLMs you run two buckets per tenant, one on requests per minute and one on **tokens** per minute, because one request with a 100 K-token prompt costs vastly more than a hundred short ones, and a request-only limit doesn't protect you at all. Buckets live in **Redis** (the in-memory key-value store; Part 1) so that all gateway replicas share the same counter, implemented as a small atomic script to avoid races.

**Quotas** are the longer-horizon sibling of rate limits: a rate limit is per-second smoothing, a quota is "this team gets 20 million tokens a day" or "this team gets the equivalent of 4 GPUs." Quotas are how finance's chargeback becomes enforceable rather than advisory, and the useful design is soft quotas — exceed your quota and you aren't cut off, you're demoted to a lower scheduling priority. That way a team's overrun degrades their own latency rather than failing their product, and the pressure to fix it is real but not an outage.

The request now hits the **router**, which decides which fleet serves it. Routing considers the requested model, whether an adapter is needed, the tenant's priority tier, and current fleet load. Two routing decisions earn their keep. First, **prefix-aware routing**: if this request shares a long prefix with a recent one (same system prompt, same RAG context, same agent conversation), route it to the worker that already has those KV blocks cached, because a cache hit turns thousands of prefill tokens into a pointer dereference. You implement this by hashing prefix chunks and keeping a Redis map from prefix-hash to worker, with consistent hashing as the fallback. Second, **batch offloading**: requests marked latency-tolerant go to a separate queue that fills GPU capacity during troughs and is preempted by interactive traffic.

Then the **scheduler**, which is where the real work happens, and it's built around **continuous batching**. Here's the intuition. Naive batching collects, say, 32 requests, runs them together until *all* are done, then takes the next 32. Because output lengths vary wildly — one request generates 10 tokens, another generates 2,000 — the whole batch runs at the speed of its slowest member, and for most of that time you're computing padding. Continuous batching (the idea from the Orca paper, and how vLLM and every modern engine work) instead schedules at the granularity of a single decode *step*: after every step, finished sequences leave the batch and waiting requests join it immediately. The batch is a revolving door, not a bus. This alone is typically a several-fold throughput improvement over static batching on realistic mixed-length traffic, and it's the single highest-leverage thing in the stack.

$$\text{bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes per element}$$

#### Architecture diagram

```
                          ┌──────────────────────────────┐
   client ───────────────►│ API gateway                  │
                          │  auth → tenant identity      │
                          │  token-bucket rate limit     │  Redis-backed
                          │   (req/min AND tokens/min)   │  atomic counters
                          │  quota check → priority tier │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │ Router                       │
                          │  model + adapter selection   │
                          │  prefix-aware affinity ──────┼─► Redis: prefix-hash → worker
                          │  interactive vs batch lane   │
                          └──────┬────────────────┬──────┘
                                 │                │
                 ┌───────────────▼──────┐  ┌──────▼─────────────────┐
                 │ Interactive queue    │  │ Batch queue            │
                 │ priority + fair-     │  │ fills troughs,         │
                 │ share (WFQ)          │  │ preempted by interactive│
                 └───────────┬──────────┘  └──────┬─────────────────┘
                             └────────┬───────────┘
                                      ▼
                     ┌────────────────────────────────────┐
                     │ Scheduler — CONTINUOUS BATCHING    │
                     │  admits/evicts at each decode step │
                     │  admission control on KV headroom  │
                     │  output-length prediction → SJF    │
                     └────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
 ┌──────▼───────────────────┐                         ┌─────────────▼──────────────┐
 │ Fleet A: 8B + 20 LoRAs   │                         │ Fleet B: 70B               │
 │  TP=1, 1 GPU per replica │                         │  TP=4 within one node      │
 │  multi-LoRA: one base,   │                         │  NVLink for all-reduce     │
 │  adapters swapped per req│                         │  chunked prefill on        │
 └──────┬───────────────────┘                         └─────────────┬──────────────┘
        │                                                           │
        └───────────────────────────┬───────────────────────────────┘
                                    ▼
                  ┌──────────────────────────────────────┐
                  │ KV cache (PagedAttention)            │
                  │  16-token blocks, no fragmentation   │
                  │  prefix sharing by refcount          │
                  │   (cache key includes tenant id)     │
                  │  FP8 quantized → 2x concurrency      │
                  │  spill to CPU RAM for long tail      │
                  └──────────────────┬───────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │ SSE token stream    │──► client
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────────────────┐
                          │ Telemetry → Kafka → warehouse   │
                          │  tokens in/out/cached, TTFT,    │
                          │  ITL, queue wait, tenant, model │
                          │  → chargeback + capacity model  │
                          └─────────────────────────────────┘

  Vendor-API models (large MoE) sit behind the same gateway as a
  proxied fleet — same auth, quotas, telemetry, chargeback — so
  teams see one endpoint and finance sees one bill.
```

### Step 5 — Evaluation

The platform is evaluated on three axes, and confusing them is the classic mistake.

**Latency**, reported per model and per tenant, not globally, as p50/p95/p99 of TTFT and ITL, plus end-to-end. Track **queue wait separately from execution time** — this is the single most useful decomposition you can have, because when p95 TTFT regresses, queue wait tells you it's a capacity problem and execution time tells you it's a model or config problem, and you'll spend hours guessing without it.

**Throughput and efficiency**: tokens per second per GPU, split into prefill and decode tokens because they're not comparable; batch size distribution over time; KV cache occupancy; MFU. And goodput as defined earlier — the fraction of interactive requests served within SLO — which is the number to put on the executive dashboard, because it's the only one that's simultaneously about users and about money.

**Quality**, which people forget a serving platform has. Every change that touches numerics — quantization, engine upgrade, speculative decoding config, a new kernel, or a vendor model version rotating under you — runs the golden set through automated evaluation before rollout, with an explicit regression threshold. The vendor-rotation case is the sneaky one: you didn't deploy anything, and your quality moved. A weekly golden-set replay against every vendor model with alerting on score deltas catches it.

Rollouts are **canaried** (Part 1 covers the mechanics): a new engine version or quantization config takes 5% of traffic, and the automatic rollback conditions are a latency regression at p95, an error-rate increase, or an evaluation-score drop beyond threshold. For a quantization change specifically I'd also run a **shadow** phase first — mirror real traffic to the candidate, don't return its output, and compare distributions of output length and evaluation scores — because quantization failures are often subtle quality degradation rather than crashes, and a canary measured only on latency will happily pass a model that's gotten dumber.

### Step 6 — Production concerns

**Capacity arithmetic.** Take the 70 B model at FP16 on H100 SXM hardware (80 GB HBM3, roughly 3.35 TB/s of memory bandwidth, about 990 TFLOPS dense BF16 — vendor specifications). Weights are $70 \times 10^9 \times 2 \text{ bytes} = 140$ GB, which does not fit on one 80 GB GPU, so you need tensor parallelism. At TP=2 you have 160 GB total and 140 GB of weights, leaving about 20 GB minus activation and framework overhead — call it 12 GB of usable KV cache, which at 0.31 MB per token is roughly 39,000 tokens of cache total, or about 19 concurrent users at 2 K context each. That's too thin, and it is worth noting that **the original source text claims 240 GB of requirement fits in TP=2 across 160 GB of memory, which is simply wrong** — I'd correct that. At TP=4 you have 320 GB, weights take 140 GB, and after overhead you have roughly 160 GB of KV cache: about 500,000 tokens, or 250 concurrent users at 2 K context. That's a working configuration, and it's what I'd run.

**Cost.** Reserved H100 capacity runs somewhere around \$2–3 per GPU-hour depending on commitment and provider (market rate, changes constantly — state it as an assumption). At \$2.50, the four-GPU node costs \$10 per hour. At the 3,520 output tokens/second computed above, one hour produces $3520 \times 3600 = 1.27 \times 10^7$ output tokens, so:

$$\text{cost per million output tokens} = \frac{\$10}{12.7} \approx \$0.79$$

**Latency budget** for an interactive request against a 1 s TTFT target:

**Failure modes.** *OOM during prefill* is the classic: a request with a 200 K-token prompt arrives, the engine tries to allocate 62 GB of KV cache for it, and the worker dies taking every in-flight request with it. Three defences, all needed: a per-tier maximum prompt length enforced at the gateway; admission control that refuses to admit a request whose predicted KV footprint exceeds current headroom; and chunked prefill so the allocation is incremental and can be aborted. *KV cache exhaustion under load* is the graceful version — you're at 98% cache occupancy and new requests can't be admitted. The engine's answer is preemption: evict a low-priority sequence, either by swapping its KV to CPU memory or by discarding and recomputing it later. Recomputation is often cheaper than swapping because prefill is fast and PCIe is slow, which is a pleasingly counterintuitive fact. *A hot tenant* — the 3 a.m. bug — is contained by the token bucket at the gateway, then by fair queueing (their overrun degrades their own share), then by a circuit breaker that sheds their load entirely if they're causing SLO violations for others. *Vendor API degradation* on the proxied models needs its own handling: timeouts, retries with jitter, and a documented fallback to a self-hosted model with a quality caveat surfaced to the caller. *Silent vendor model rotation* is caught by the weekly golden-set replay. *Speculative decoding going net-negative* when acceptance rate drops is caught by monitoring acceptance and auto-disabling below a threshold. And the operational one people miss: *a model deploy takes minutes*, because you're moving 140 GB of weights onto GPUs, so autoscaling cannot respond to a spike — you must either keep warm standby capacity or accept queueing, and choosing which is a cost conversation with the business, not a technical one.

**Monitoring** splits three ways. Infrastructure: GPU memory occupancy, KV cache utilization, batch size distribution, achieved bandwidth, per-worker health, all scraped by **Prometheus** and charted and alerted in **Grafana** (Part 1). Serving quality: TTFT and ITL percentiles per model and tenant, queue wait separately, goodput, preemption rate, prefix cache hit rate, speculative acceptance rate. Business: tokens per tenant per day, cost per tenant, quota consumption, and cost per million tokens trended over time — that last one is the platform's own KPI and it should go down every quarter or the platform isn't earning its existence. The alerts that actually page someone are goodput below target, KV occupancy above 95% sustained, preemption rate spiking, and error rate on any fleet.

**Retraining triggers** don't apply in the usual sense, but three things get refreshed on a cadence. The output-length predictor retrains weekly on recent logs, since traffic mix shifts. The capacity model — the forecast that says how many GPUs you need next month — refits monthly on observed demand. And the speculative-decoding draft model is re-evaluated quarterly against the current traffic distribution, since acceptance rate is a function of how well the draft matches real prompts, and that drifts as teams ship new features.

### The hard tradeoff

The central one is **latency versus cost**, and it is unusually clean here because the mechanism is a single knob: batch size.

Increasing batch size increases throughput almost linearly at first, because decode is memory-bandwidth-bound — you read the weights once per step regardless of how many sequences are in the batch, so going from batch 1 to batch 64 gets you roughly 64× the tokens for nearly the same memory traffic. This is the fundamental economics of LLM serving and it's why batching is not an optimization but the entire business model. But every sequence in the batch shares the step, so a larger batch means a longer step, which means higher ITL for every user in it. And a larger batch means more KV cache, which means at some point you hit the memory wall and start preempting, which is catastrophic for tail latency.

So you're choosing a point on a curve where one end is "every user gets a dedicated GPU and 5 ms tokens and it costs a hundred times too much" and the other is "batch of 512, wonderful cost per token, and everyone waits four seconds for their first word."

The resolution has three parts and stating all three is what makes this answer complete. First, exploit the *satisfaction ceiling*: nobody perceives ITL below about 25 ms, so the correct policy is to batch up to the point where ITL hits your SLO and not one request further — improvements beyond that are converted into throughput, not given away as speed. Second, **segment the traffic** rather than picking one point, which is the real answer: interactive traffic runs on a fleet configured for moderate batch and tight ITL, batch traffic runs at maximum batch on the same hardware during troughs, and the two are scheduled against each other. Batch workloads are not a nuisance to tolerate; they are the thing that makes the interactive fleet affordable, because they convert idle capacity into revenue. Third, use the mechanisms that shift the curve instead of moving along it: prefix caching, quantization, and speculative decoding all give you more throughput at the *same* latency, and those are strictly better than any batch-size choice. Frame it that way — "first I'd shift the curve, then I'd pick a point on it per traffic class" — and you've answered a question the interviewer usually has to drag out of people.

---

## Design 7 — Semantic image search

### Requirements and assumptions

**Assume, for the rest of this answer:** 50 million active listings averaging four images each, so about 200 million images; text, image, and image-plus-text queries all in scope; hybrid with existing lexical search; the objective is add-to-cart rate with revenue as a guardrail; roughly 500,000 new or updated listings per day with a freshness requirement of under five minutes; 1,000 queries/second at peak with a 300 ms budget end to end; reliable price and stock, noisy seller-provided categories; two years of click and purchase logs from keyword search.

**Functional requirements.** The system must validate and crop uploads, encode text and images into one space, retrieve lexical and visual candidates, apply price, stock, rights, region, and safety filters, rerank results, deduplicate products, support new listings, and return metadata and CDN image URLs.

**Non-functional requirements.** The system must stay below 300 ms at p99, support 1,000 QPS, index updates within five minutes, scale to 200 million images, preserve ANN recall, keep encoder and index versions consistent, tolerate shard failures, and protect uploaded-image privacy.

### Step 2 — Frame it as an ML problem

This is **retrieval followed by ranking**, and the framing has two halves that people conflate.

The retrieval half is a **multimodal embedding** problem. An embedding is a fixed-length vector of real numbers representing an object, learned so that geometric closeness in the vector space corresponds to semantic closeness in the world. A *multimodal* embedding is one where objects of different types — here, images and text — are mapped into the **same** space by different encoders, so that a photograph of a walnut sideboard and the string "mid-century walnut sideboard" land near each other despite having no surface features in common. That shared space is the entire trick, and it's what makes text-to-image search a nearest-neighbour lookup rather than a translation problem.

**CLIP** (Contrastive Language–Image Pretraining, OpenAI 2021) is the canonical way to get one. It's two encoders — a vision transformer for images, a text transformer for text — trained jointly on hundreds of millions of image-caption pairs scraped from the web, with a projection at the end of each that maps both into a common vector space, typically 512 or 768 dimensions depending on the variant. Later models in the same family (SigLIP, EVA-CLIP, and various open reproductions) improve on it with different objectives and data, and in production I'd benchmark a few rather than assume CLIP itself is best; the architecture pattern is what matters.

**Contrastive learning** is how it's trained, and it's worth explaining properly because it's the mechanism behind half of modern retrieval. Take a batch of $N$ image-caption pairs. Encode all $N$ images and all $N$ captions. You now have an $N \times N$ matrix of similarities between every image and every caption. The $N$ diagonal entries are the true pairs and the $N^2 - N$ off-diagonal entries are wrong pairings, and the loss — InfoNCE, applied symmetrically in both directions — pushes the diagonal up and the off-diagonal down:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}$$

where $v_i$ and $t_i$ are the normalized image and text vectors, $\text{sim}$ is cosine similarity, and $\tau$ is a learned temperature that controls how sharply the model separates the positive from the negatives. Two consequences follow directly and both matter in production. First, the other items in the batch serve as the negatives, so **large batches are essential** — with a batch of 32 the task is trivially easy and the model learns a lazy representation; CLIP was trained with batches in the tens of thousands. Second, the model only learns to separate things that appear together in a batch, which is why **hard negative mining** — deliberately including near-misses, like two sideboards that differ only in wood tone — matters enormously when you fine-tune on a product catalogue where everything already looks similar.

One capability falls out of this for free and is worth naming: **zero-shot classification**. Because text and images share a space, you can classify an image into arbitrary categories without any training data for them — encode the candidate label strings ("a photo of a walnut sideboard", "a photo of a pine bookshelf"), encode the image, and take the nearest label. In this design that gives you a free category-inference model for the noisy seller-provided categories, and it costs you one forward pass you're already doing.

### Step 3 — The data

You have three data assets and they do different jobs.

**The catalogue** is 200 million images plus structured metadata. The images are seller-uploaded, which means variable resolution, bad lighting, watermarks, collages of multiple products in one frame, and stock photos reused across sellers. That last one is worth flagging: duplicate and near-duplicate images across listings are extremely common on marketplaces, and if you don't deduplicate them your results page shows the same product eight times from eight sellers. **Perceptual hashing** — a hash designed so that visually similar images produce similar hashes, robust to resizing and re-compression, as defined in Design 5 — is the cheap first pass for exact and near-exact duplicates; embedding-space clustering catches the rest.

**Behavioural logs** from the existing keyword search are your training signal, and the way you turn them into training pairs is the most consequential decision in this design. The naive approach — treat (query, clicked item) as a positive pair, sample random items as negatives — produces a model that learns *popularity* rather than *relevance*, because random negatives are trivially easy to separate and clicks are dominated by position. Three corrections. Use **hard negatives**: items that were shown in the same result page and *not* clicked, which forces the model to learn the fine distinction rather than the coarse one. Correct for **position bias** — users click the top result because it's the top result, not because it's best — by weighting each training example by the inverse of its estimated examination probability at that position (inverse propensity scoring; Part 1 and the ads design cover the mechanics). And prefer **purchase** over click as the positive signal where volume allows, since clicks are noisy and purchases are what you're actually paid for; a common compromise is a weighted objective with clicks at low weight and purchases at high weight.

**Human relevance judgments** are the third asset and the one people skip. You need a few thousand (query, item) pairs rated by humans on a graded relevance scale, because behavioural data can only tell you about items your current system already showed. It cannot tell you whether there's a perfect match sitting at rank 40,000 that you've never surfaced, and that is exactly the failure mode a new retrieval system is supposed to fix. Without a judged set, your offline metrics will improve while the system gets worse at the thing you built it for.

### Step 4 — The architecture

Follow a query. The user uploads a photo of a chair and types "but in oak, under £400."

The **query preprocessing** stage does the boring, necessary work: decode the image, strip EXIF (which carries GPS coordinates and is a privacy incident waiting to happen), resize to the encoder's expected resolution, and — importantly — detect and crop the salient object. A café photo contains a chair, a table, a coffee cup, and someone's elbow; embedding the whole frame gives you a vector that means "café interior," which retrieves other café interiors. Running a lightweight object detector and embedding the largest detected product region, or offering the user a crop box, is the difference between a demo and a product. Meanwhile the text part is parsed for structured constraints — "under £400" becomes a price filter, "oak" is retained as a soft semantic signal — with a small parser or a cheap LLM call handling the extraction.

The image goes through the **image encoder** (a ViT, roughly 10–25 ms on GPU for a base-size model) producing a 768-dimensional vector; the residual text goes through the **text encoder** producing a vector in the same space. For a combined query, the simplest effective approach is a weighted sum of the two normalized vectors, re-normalized — which works better than it has any right to, because the space is approximately linear for compositional attributes. It does *not* handle negation or relational constraints, which is a limitation to name before the interviewer does.

Now **retrieval** against the **vector index**. Define it: a vector index is a data structure over a large set of vectors that answers "which stored vectors are nearest to this query vector?" without comparing against all of them. Brute-force comparison against 200 million 768-dimensional vectors is $200\text{M} \times 768$ multiply-adds per query, about $1.5 \times 10^{11}$ operations — feasible on GPUs but wildly wasteful at 1,000 QPS. So you use **approximate nearest neighbour** search, which trades a small amount of correctness for orders of magnitude of speed.

**ANN recall** is the metric that quantifies that trade: recall@$k$ for an index is the fraction of the *true* top-$k$ nearest neighbours that the approximate search actually returns. An index at 0.95 recall@100 returns 95 of the true top 100. This is a distinct concept from the recall of your search *system* and confusing the two is a common and visible error — you can have 0.99 ANN recall and terrible search quality if your embeddings are bad, and vice versa. ANN recall is tunable at query time and it trades directly against latency.

The two index families to know. **HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph where each vector is a node connected to its approximate neighbours, with sparse long-range links in upper layers; search starts at the top layer, greedily walks toward the query, and descends. It gives excellent recall at low latency, supports incremental insertion (which matters enormously for freshness), and costs memory — the graph edges alone are typically comparable to the vectors themselves. **IVF-PQ** (Inverted File with Product Quantization) instead clusters vectors into cells, searches only the few cells nearest the query, and stores each vector *compressed* by splitting it into sub-vectors and replacing each with a codebook index. It's dramatically more memory-efficient — compression of 16–32× is routine — at some cost in recall, and it's what you use when the vectors won't fit in RAM.

The three legs are fused. **Reciprocal rank fusion** is the simple robust default: each item scores $\sum_{\text{legs}} 1/(k + \text{rank}_{\text{leg}})$ with $k$ around 60, which needs no score calibration across legs — a genuine advantage, since BM25 scores and cosine similarities aren't on comparable scales. A learned fusion is better if you have the data.

#### Architecture diagram

```
  query: photo of a chair  +  "but in oak, under £400"
        │
        ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Query preprocessing                                       │
 │  image: EXIF strip, resize, OBJECT DETECT + CROP          │  ~20 ms
 │         (embedding the whole café gives you cafés)        │
 │  text : parse hard constraints ("<£400" → filter)         │
 │         keep soft attributes ("oak") as semantics         │
 └───────────────────────┬───────────────────────────────────┘
                         ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Bi-encoder — SHARED EMBEDDING SPACE                       │
 │  ViT image encoder ──┐                                    │  ~15 ms
 │                      ├─► 768-d vector (normalized)        │
 │  text encoder ───────┘   weighted sum for combined query  │
 └───────────────────────┬───────────────────────────────────┘
                         ▼
 ┌──────────────┬────────────────────┬───────────────────────┐
 │ DENSE leg    │ LEXICAL leg        │ FILTER leg            │
 │ HNSW ANN,    │ BM25 over titles   │ in-stock, region,     │
 │ 8 shards x   │ + descriptions     │ price < £400          │
 │ 25M vectors  │ (exact brand/SKU   │                       │
 │ recall@100   │  matches dense     │ APPLIED INSIDE the    │
 │  ~0.97       │  retrieval misses) │ ANN search, NOT after │  ~40 ms
 └──────┬───────┴─────────┬──────────┴──────────┬────────────┘
        └─────────────────┼─────────────────────┘
                          ▼
              ┌───────────────────────────┐
              │ Reciprocal rank fusion    │  ~1000 candidates
              │  Σ 1/(60 + rank)          │  no score calibration needed
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ CROSS-ENCODER rerank      │  top 100 → full attention
              │  query + item TOGETHER    │  over query AND item
              │  not precomputable        │  ~40 ms (biggest budget item)
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Business ranker (GBDT)    │  price, seller rating,
              │  trained on add-to-cart   │  conversion, shipping,
              │  "similar" → "good"       │  image quality, freshness
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Diversity pass (MMR)      │  cap per seller / per cluster
              └─────────────┬─────────────┘
                            ▼
                      top 20 results

  ═══════════════ INDEXING PATH (continuous) ═══════════════

### Step 5 — Evaluation

Three layers, and you should distinguish them explicitly because they measure different failures.

**Embedding quality**, measured in isolation on a held-out set of (query, relevant-item) pairs, using recall@$k$ — the fraction of queries whose relevant item appears in the top $k$ retrieved from the *full* catalogue by exhaustive search. This measures the model, with the index removed from the picture. If this is bad, no amount of index tuning saves you.

**Index quality**, measured as ANN recall@$k$ against exhaustive search on the *same* embeddings. This isolates the index. Run it as a scheduled job against a sample of queries, because ANN recall silently degrades as the index accumulates incremental inserts and tombstones, and nothing else will tell you. This is a separate alarm from search quality and mixing them costs you a day of debugging every time.

**End-to-end ranking quality** on human-judged query sets, using NDCG@10 — normalized discounted cumulative gain, which rewards putting highly-relevant items near the top with a logarithmic position discount, normalized so 1.0 is a perfect ordering. Report it separately for text queries, image queries, and combined queries, because they fail differently and an average will hide that image queries are broken.

Online, A/B test on add-to-cart rate as the primary metric, with revenue per session and search abandonment rate as guardrails. The specific counter-metric to watch is **query reformulation rate** — a user who searches again immediately didn't find what they wanted, and this catches "results looked plausible but were wrong" better than click-through does, since a bad-but-attractive result still gets clicked. Also segment by query type: a change that helps image queries and hurts text queries can look neutral in aggregate while making half your users' experience worse.

### Step 6 — Production concerns

**Scale arithmetic.** 200 million images at 768 dimensions: FP32 is 614 GB, FP16 is 307 GB, plus HNSW graph overhead of roughly 40–50% brings it to about 450 GB. Sharded eight ways that's 56 GB per shard, which fits on commodity memory-heavy instances with room for the OS and the graph's working set. At 1,000 QPS across 8 shards, each shard sees 1,000 queries/second — note that sharding does *not* reduce per-shard QPS, since every query goes to every shard; it reduces per-shard *data*. So you also replicate each shard for throughput: HNSW at this size serves on the order of a few thousand queries/second/core with good recall settings, so two to three replicas per shard gives comfortable headroom. Total: roughly 16–24 index machines.

**Latency budget** against 300 ms:

**Failure modes.** *The index and the model fall out of sync* — half the vectors from the old model, half from the new — and the symptom is not an error but quietly bad results for a subset of queries. The fix is a dual-index swap with an explicit model-version tag on every vector and a hard check that all shards report the same version before serving. *ANN recall degrades silently* as incremental inserts and tombstones accumulate; the scheduled recall-against-exhaustive job is the only detection, and periodic compaction is the fix. *A shard goes down* and you're now searching 7/8 of the catalogue, which returns plausible-looking results with a silent quality hit; alarm on shard count, and prefer serving degraded-with-a-flag over failing, but make sure the flag is visible in monitoring. *Post-filtering collapse* when a user applies a narrow price filter and gets four results; handled by in-index filtering plus a fallback that widens the ANN search when the filtered result count is low. *Popularity feedback loop*: the ranker favours items with conversion history, those get shown more, they accumulate more history, and new listings never escape — which on a marketplace means seller churn. The counters are an explicit exploration budget for new listings, a freshness boost that decays, and content-based features that don't depend on history. *Adversarial sellers* upload attractive stock images unrelated to what they ship; detect with image-text consistency scoring (does the CLIP similarity between the listing's image and its own title look normal?) and with the returns rate as a downstream signal. And *query images that are garbage* — screenshots, memes, blurry photos of nothing — should be detected by an embedding-norm or out-of-distribution check and routed to a "we couldn't read that photo" experience rather than returning confident nonsense.

**Monitoring.** Infrastructure metrics — per-shard latency, memory, index size, insert lag from Kafka — go to **Prometheus** and **Grafana** (Part 1). The index-specific ones that earn their place: **index freshness lag**, measured as the p99 age of the newest listing findable in search, alarmed at your five-minute SLO; ANN recall from the scheduled job; and vector count per shard, since a shard drifting in size means the sharding function is unbalanced. Model metrics: embedding-norm distribution on queries (a shift means input distribution changed), fusion-leg contribution rates (if the lexical leg suddenly wins every query, dense retrieval broke), and score distribution of the top result. Business metrics: add-to-cart rate and NDCG on the judged set, both segmented by query type; zero-result rate, which is the most user-visible failure; and reformulation rate.

**Retraining triggers.** The bi-encoder is fine-tuned monthly, or when catalogue composition shifts materially — a new product category with unfamiliar vocabulary is the usual trigger. The cross-encoder and business ranker retrain weekly, since they're cheap and behavioural data accumulates fast. A full re-index happens on any bi-encoder change and nothing else. Off-cadence triggers: zero-result rate rising, NDCG on the judged set dropping beyond threshold, or a seasonal shift (the query distribution in November is genuinely different, and a model fitted on July data underperforms).

### The hard tradeoff

The one worth spending your time on is **visual similarity versus purchase intent**, because it's the tradeoff that determines whether this system makes money.

The embedding space is optimized to place visually and semantically similar things near each other. So the nearest neighbour to a photo of a specific chair is a nearly identical chair — often literally the same product from a different seller, or the same product photographed slightly differently. That is exactly what the model was asked to do, and it is frequently not what the user wants. Someone photographing a chair in a café is usually not asking "find me this exact chair"; they're asking "find me a chair like this that I can afford, that ships to me, and that I'd actually be happy with." The pure nearest-neighbour answer to that is a page of twelve near-identical chairs, several out of stock, sorted by an accident of embedding geometry.

Push too far the other way — rank purely on predicted conversion — and you get a different failure that's harder to see. The system learns to show whatever converts, which is the popular, cheap, heavily-reviewed items, regardless of whether they resemble what the user asked for. Users searching for a specific mid-century sideboard get the best-selling flat-pack unit, because that item converts well for everyone. Search stops being search and becomes a merchandising surface, and users stop trusting it, and the metric that catches this is not add-to-cart on the session — it's retention over months, which is exactly the metric your A/B test isn't long enough to measure.

---

## Cross-cutting questions

### Cold start

For a new user, use locale, language, context, declared interests, and population priors. Update short-term state after each interaction. For a new item, use content features and explicit exploration. Missing history must mean “unknown,” not zero.

### Drift

Monitor feature distributions, missingness, score distributions, calibration, and outcomes by segment. Covariate drift changes inputs. Label drift changes the base rate. Concept drift changes the relationship between inputs and outcomes. First rule out a pipeline bug. Then decide whether to repair data, adjust thresholds, add features, or retrain.

### Delayed and missing labels

Train on cohorts whose outcome window has matured. Use early proxies for fast monitoring and mature labels for final evaluation. In systems such as fraud, declined cases have no normal outcome. Use a small randomized sample or another audit path to measure this rejected region.

### Feedback loops

The model changes what users see, so it changes its next training set. Log exposure probabilities, keep a small exploration budget, and do not treat unexposed items as negatives. Use inverse propensity weighting or doubly robust estimation when the logging policy is known.

### Offline improvement but online regression

Check experiment assignment and instrumentation first. Then check training-serving skew, feature freshness, latency, calibration, segment regressions, metric mismatch, and interactions with product rules. Offline metrics are proxies. Validate their historical correlation with online outcomes.

### One model or several

Start with one shared model. Split only when segments have different labels, constraints, feature meaning, or enough data to support separate models. Multi-task learning, adapters, or segment features often provide specialization with lower operational cost.

### Business and model metrics disagree

Believe a well-measured business result before an offline proxy. Diagnose whether the proxy is misaligned, a high-value segment regressed, a second-order effect appeared, or the experiment is broken. Fix the metric or model rather than optimizing a proxy that has stopped predicting the business outcome.
