# ML System Design — Deep Dive

> **Why this exists.** ML system design is the hardest interview round at Meta/Google/Amazon/Netflix-tier companies because the question is open-ended: "design YouTube's recommender." Strong candidates do five things simultaneously — clarify scope, reason about scale, propose an architecture, defend trade-offs, anticipate failures — without losing the thread. This deep dive gives you the framework + 7 worked examples (recommender, search, ads, fraud, content moderation, LLM serving, image search) at the depth interviewers actually probe.

---

## 1. The framework — six steps, in order

Every ML system design answer follows the same structure. Use it religiously, even when you have a strong instinct for the answer.

| Step | What you do | Time |
|---|---|---|
| **1. Clarify** | Ask 3-6 sharp questions to pin down scope, scale, latency, business goal | 3-5 min |
| **2. Frame as ML problem** | Translate to inputs/outputs/labels/loss; identify training & serving regime | 2-3 min |
| **3. Data** | Sources, labels, freshness, leakage prevention | 5 min |
| **4. Model** | Architecture, training, evaluation; pick from a small menu of patterns | 8-10 min |
| **5. Serving** | Latency budget, infra, deployment, fallbacks | 5-8 min |
| **6. Monitoring & iteration** | Metrics across infra/model/business; retraining cadence; failure response | 3-5 min |

**Common mistake**: skipping step 1 because the problem "seems clear." Even 2 minutes of clarification scores high — it shows judgment.

---

## 2. The clarification checklist (memorize this)

Any open-ended question, ask through this list:

**Goal**
- "What's the business outcome we're optimizing? (Clicks, dwell time, revenue, retention?)"
- "What does success look like in 6 months?"
- "What's the cost of being wrong — false positives vs false negatives?"

**Scale**
- "How many requests per second at peak?"
- "How many users / items / queries in the catalog?"
- "What's the latency budget end-to-end?"

**Constraints**
- "Real-time or batch?"
- "Personalization: per-user, segments, or population-level?"
- "Are there fairness, privacy, or regulatory constraints?"
- "What's the cold-start situation — new users, new items, both?"

**Inputs**
- "What data is available? Logs, user features, item features, behavior?"
- "Is the label easily available or expensive to obtain?"

This usually surfaces 1-2 things you can simplify (e.g., "they care about latency more than accuracy → stronger retrieval, smaller ranker"). Pin those down explicitly.

---

## 3. The architecture menu — patterns you'll reuse across designs

Production ML systems are built from a small set of repeating patterns. Recognize them and you can stitch any design.

### 3.1 Two-stage retrieve-and-rank

For tasks with a large candidate set: search, recommendations, ads, RAG retrieval.

```
[query/user/context]
       │
       ▼
┌──────────────────────┐
│  Stage 1: Retrieval  │   Goal: high recall on top-K
│   (cheap, scales)    │   ANN over embeddings, BM25, rules
└──────────────────────┘
       │  top 100-1000
       ▼
┌──────────────────────┐
│  Stage 2: Ranking    │   Goal: high precision on top-K
│  (expensive, accurate)│  GBDT / DLRM / cross-encoder
└──────────────────────┘
       │  top 10
       ▼
┌──────────────────────┐
│  Stage 3 (optional): │   Goal: re-rank for diversity, freshness, business rules
│   Business reranker  │   Hand-tuned scoring, MMR, exploration
└──────────────────────┘
       │
       ▼
   [response]
```

**Why two-stage?** Retrieval over millions of items in 10ms is achievable only with cheap methods (ANN). Ranking with neural cross-attention over 10M items in 50ms isn't. Split: cheap-and-recall-first, then expensive-and-precision-second.

### 3.2 Embedding indexes (HNSW / IVF-PQ)

For ANN retrieval. Trade-off:
- **HNSW**: fast at query time, fastest recall, high memory cost.
- **IVF-PQ**: smaller memory, slightly worse recall, used at billion-scale.
- Standard: HNSW for ≤100M items, IVF-PQ for ≥1B.

### 3.3 Feature stores (Feast, Tecton, internal)

Two-pane store: offline (Parquet on S3) for training; online (Redis/Cassandra) for serving with <10ms reads. Same code computes both → no online/offline skew. **Skew prevention** is the #1 pain point in production ML.

### 3.4 Real-time vs batch vs streaming

- **Batch**: features and predictions precomputed nightly, served via key-value lookup. Cheapest. Stale by 24h.
- **Real-time**: features computed at request time from event streams (Kafka). Fresh, expensive.
- **Streaming**: features updated continuously (Flink/Spark Streaming). Middle ground.

For most companies: batch features for stable signals + real-time for fresh signals (last 5 min activity), combined at scoring time.

### 3.5 Online evaluation: A/B testing

You can never trust offline metrics for the launch decision. Always A/B with proper statistical rigor (see `30_ab_testing/`). Common variants you'll discuss:

- Holdback: a permanent control group on the old model to measure drift over time.
- Interleaving: instead of A/B at user level, mix items from both rankers within a single page → smaller sample sizes for ranker comparisons.
- Sequential testing: peek-safe analysis with always-valid p-values.

### 3.6 Shadow / canary / blue-green deployment

- **Shadow**: new model receives traffic, predictions discarded, compared offline to current model. Risk-free pre-production check.
- **Canary**: new model gets 1-5% of traffic. Monitor metrics. Roll out if green.
- **Blue-green**: full switch with instant rollback.

Always have a rollback plan. Ship behind a feature flag.

### 3.7 Multi-tier serving

- Hot path: small model with low latency (e.g., BERT-base distilled).
- Reranker: bigger model on top-K.
- Async refinement: even bigger LLM-judge or human-in-loop for uncertain cases.

### 3.8 Cold-start playbook

| Problem | Mitigation |
|---|---|
| New user | Demographic priors, popularity-based recs, onboarding survey, exploration |
| New item | Content embeddings (text/image), forced exposure period, similarity to existing items |
| New query | Query rewriting, fallback to BM25 |
| Cold node in distributed serving | Pre-warm cache, gradual traffic ramp |

---

## 4. Worked example 1 — design YouTube's recommender system

**Interviewer prompt:** "Design a video recommendation system for YouTube's home feed."

### Clarification (2 min)

"To make sure I'm solving the right problem:
- Goal: I'll assume optimizing **watch time** (the metric YouTube actually optimizes), with engagement and long-term retention as guardrails.
- Scale: ~2B monthly users; ~1B videos; ~1B home-feed loads/day.
- Latency: <150ms end-to-end for the home feed.
- Output: ranked list of ~20-50 videos.
- Personalization: per-user, with cold-start handling.
- Fairness: surface diverse content (avoid echo chambers, surface long-tail creators)."

### Frame as ML problem

Two-stage personalized retrieval + ranking:

- **Retrieval**: from 1B videos → top ~1000 candidates.
- **Ranking**: from 1000 → top 50, sorted by predicted watch time.
- **Post-processing**: diversity reranker, freshness boosts, policy filters.

### Data

**User features**:
- Demographics (country, language, device, age band).
- Long-term: subscriptions, channels watched, topic embedding from history.
- Short-term: last N watched videos (sequential signal).
- Context: time of day, day of week, location, device, network.

**Video features**:
- Content: title, description, transcript, thumbnail (multimodal embedding).
- Metadata: channel, duration, language, upload time, category.
- Engagement priors: CTR, watch-through rate, avg watch time on this video so far.

**Labels**:
- Implicit: watch time per impression. Define "click" = >10s watched. "Engaged watch" = >30s or >50% of video.
- Negative samples: shown but not clicked + random negatives.

**Leakage to watch out for**:
- Don't include features computed *after* the user clicked the video (e.g., comments left).
- Time-respecting splits — train on past, validate on later periods.
- Per-user splits if generalizing to new users.

### Architecture

#### Stage 1 — retrieval (3-4 sources merged)

Multiple retrieval sources, each with high recall on different signals:

1. **Two-tower neural retriever** (the big one)
   - User tower: history embedding (transformer over recent watches) + demographic features → 256-d vector.
   - Video tower: content (title + thumbnail CLIP) + metadata + engagement features → 256-d vector.
   - Trained with sampled softmax + in-batch negatives + hard negatives (impression-not-clicked).
   - Item embeddings precomputed nightly; indexed in HNSW.
   - At request time: encode user → ANN top 500.

2. **Subscription-based retrieval**: videos from subscribed channels published in last 7 days.

3. **Trending / popular**: per region/language, refreshed hourly.

4. **Collaborative filtering retrieval**: matrix-factorization-based; videos liked by users similar to me.

Merge by union; deduplicate. Output: ~1000 candidates.

#### Stage 2 — ranking

Inputs: (user, video, context) → score for each candidate. Cross-features matter here (e.g., "video topic × user's preferred topics").

Model: **DLRM-style** architecture
- Sparse features (user_id, video_id, channel_id, topic_id) → embeddings.
- Dense features (engagement priors, recency) → MLP.
- Cross-feature layer (FM-like or DCN).
- Top MLP → multi-task heads.

Multi-task targets (key for serving multiple objectives):
- $p$(click | impression)
- $p$(watch >50% | clicked)
- $p$(like | clicked)
- $p$(comment | clicked)
- Predicted watch-time (regression head)

Final score: weighted combination of heads (weights themselves tuned via Bayesian opt against business KPI). Multi-task lets you trade objectives smoothly.

Trained on ~weeks of impression logs. Daily retraining. Online learning: small LR fine-tuning on most recent hour's logs.

#### Stage 3 — business reranker

- **Diversity**: cap items per channel/topic in the final feed (e.g., max 3 from same channel in top 20).
- **Freshness boost**: small score multiplier for videos < 24h old.
- **Quality / policy filters**: hard-block videos violating policies; soft-down-rank borderline content.
- **Exploration**: small fraction of slots given to high-uncertainty / high-novelty items (Thompson sampling on rerank).

### Serving

```
request
  │
  ▼
[feature service]  ← user features (cached, ~1ms),
                     real-time signals (last 5 min, ~5ms)
  │
  ▼
[retrieval]        ← parallel ANN + subscription + trending + CF (~30ms)
  │  top 1000
  ▼
[feature hydration] ← cross features for ranker (~10ms)
  │
  ▼
[ranker]           ← DLRM batched inference on top 1000 (~50ms)
  │  top 50
  ▼
[business reranker] ← diversity, freshness, policy (~5ms)
  │
  ▼
[response]         ← total ~100-130ms p50, 200ms p99
```

**Caching**:
- User embedding: cached for 5-15 min (recomputed when user does new actions).
- Item embeddings: precomputed offline, refreshed nightly for new uploads.
- Hot trends per region: cached for 1 hour.

**Failure modes / fallbacks**:
- If retrieval service down → fallback to popularity feed (per region).
- If ranker times out → use retrieval scores directly.
- If feature store down → use last-seen cached features.

### Monitoring

Three layers:

**Infra metrics**: p50/p95/p99 latency, error rate, cache hit rate, retrieval recall.

**Model metrics**:
- Score distribution drift (detected via PSI on ranker outputs).
- Per-feature value distribution drift.
- Click-through rate predictions vs realized (calibration).
- Coverage (% of catalog actually shown).

**Business metrics**:
- Watch time per session (primary).
- Daily/monthly active users.
- Retention day-7, day-30.
- Ad revenue (tertiary).
- Diversity proxies (Gini of recommended items, creator diversity).

### Iteration & retraining

- **Daily**: retrain ranker on yesterday's logs.
- **Weekly**: re-index two-tower embeddings.
- **Monthly**: refresh content embeddings with latest CLIP/Multimodal model.
- **Quarterly**: A/B test new architectural ideas.

### What goes wrong (interview hot-buttons)

- **Filter bubbles**: model converges on narrow content. Fix: explicit diversity bonuses + exploration.
- **Engagement traps**: optimizing for clicks promotes clickbait. Fix: optimize for watch-time + thumbs-up + retention, not clicks.
- **Cold-start for new creators**: their videos never get traffic → never learn signal. Fix: forced exposure period.
- **Long-term vs short-term**: aggressive watch-time optimization may hurt long-term retention. Fix: holdback experiments measuring 90-day retention.

---

## 5. Worked example 2 — design Google search ranking

**Prompt:** "Design the ranking system for Google web search."

### Clarification

- Goal: relevance + page quality + freshness, weighted (Google's actual signals are hundreds; we'll model 5-10).
- Scale: ~10B queries/day; ~100B web pages indexed.
- Latency: <200ms end-to-end including network.
- Personalization: minimal (Google personalizes lightly compared to YouTube).

### Architecture (multi-stage)

```
query
  │
  ▼
[query understanding]  ← spell-correct, intent classification, entity link
  │
  ▼
[retrieval — billions of pages]
  ├─ Inverted index (BM25) — exact term match, fast
  ├─ Dense retrieval — semantic
  └─ Knowledge graph for entity queries
  │  ~10K candidates
  ▼
[L1 ranker — feature-based]   ← LightGBM on 100s of features (PageRank,
                                 freshness, click signals, etc.)
  │  top 1000
  ▼
[L2 ranker — neural]         ← cross-encoder transformer (BERT-style),
                                 ~100ms for top-100
  │  top 100
  ▼
[L3 ranker — LLM-augmented]  ← (newer) for hard queries, LLM judges relevance
  │  top 10
  ▼
[result page assembly]       ← snippets, KG card, ads, etc.
```

### Key signals

- **Lexical**: BM25, TF-IDF.
- **Semantic**: dense embedding similarity.
- **Page quality**: PageRank, domain authority, spam signals.
- **Freshness**: publication date, last update.
- **User behavior**: aggregated CTR, dwell time, query reformulation rate.
- **Diversity**: result variety (don't return 10 results from same domain).
- **Entity match**: knowledge graph linking.

### Training

- **Offline**: human-labeled query-document pairs (graded relevance 0-4). Used to train rankers via LambdaMART (pairwise) or cross-encoder.
- **Online**: click-based signals reweighted to remove position bias (IPS/inverse propensity scoring).
- **Multi-objective**: relevance + diversity + freshness combined via Pareto-optimal weights.

### Position bias

A massive issue for click-based training. Top-position results get clicked more regardless of relevance. Common fixes:
- IPS reweighting at training.
- Randomized result interleaving for unbiased click data on a small fraction of traffic.
- Counterfactual evaluation (off-policy).

### Serving

- Inverted index: distributed across thousands of machines, sharded by document.
- Dense retrieval: ANN sharded too, replicated for throughput.
- Rankers: served on dedicated GPU/TPU pools.
- Aggregation tier: fans out to retrieval shards in parallel, gathers top-K from each.

### Monitoring

- NDCG@10 on labeled eval set.
- Click signals on production traffic (with debiasing).
- Reformulation rate (high reformulation → bad rankings).
- Per-query-type performance (head queries vs tail).
- Spam/adversarial monitoring (sudden new domains shooting up rankings).

---

## 6. Worked example 3 — design ads ranking

**Prompt:** "Design the ad ranking system for a social media feed."

### Clarification

- Goal: maximize platform revenue + advertiser satisfaction + user experience. Three-sided objective.
- Scale: ~100M ads in inventory; ~1B users; ~10B ad impressions/day.
- Latency: <50ms (ads sit in the same feed slot as organic content).
- Constraint: pacing (advertiser budgets, frequency caps).

### The ad-tech triangle

Three goals in tension:
1. **Revenue maximization** for the platform: high-bid ads.
2. **Advertiser ROI**: ads must convert (clicks → actions).
3. **User experience**: don't show too many or irrelevant ads.

### Frame: predicted-bid optimization

For each (user, ad) pair, predict:
- $p_{\mathrm{click}}$ = probability of click given impression.
- $p_{\mathrm{conversion} | \mathrm{click}}$ = probability of conversion given click.
- Predicted bid = advertiser's bid for the action being predicted.

Final ranking score: $\text{eCPM} = p_{\mathrm{click}} \times p_{\mathrm{conv|click}} \times \text{bid}$ (expected cost per mille — what the platform expects to earn from this impression).

In a **first-price auction** (modern standard), each ad's score is its eCPM and the highest-eCPM ad wins, paying its bid. (Second-price auctions were the historical default but are rare now.)

### Two-stage retrieval

**Retrieval** (~100M → 10K):
- Targeting filter: user attributes match advertiser's audience.
- Real-time bidding (RTB) calls to advertisers (or DSPs) for actively bidding ads.
- Pacing-aware: skip ads that have hit budget caps.

**Ranking** (~10K → 10):
- pCTR model: deep + wide architecture (DLRM-style with sparse user/ad features + cross features).
- pConversion model: similar architecture; trained on conversion data (much sparser).
- Calibration: predicted CTR must match actual rate (used to compute eCPM correctly). Calibration drift breaks the auction.

### Calibration is critical

Non-calibrated CTR predictions don't matter for ranking (monotone-preserving) but are *life-or-death* for ad auctions: the eCPM directly determines what the platform charges. A 10% calibration drift = 10% revenue loss.

Standard fix: Platt scaling or isotonic regression on a held-out set; recalibrate weekly.

### Serving

```
ad request (user, slot, context)
  │
  ▼
[targeting + pacing filter]  ← which ads are even eligible (~100M → 10K)
  │
  ▼
[bid retrieval]              ← RTB calls or pre-bid lookup
  │
  ▼
[ranker — pCTR & pConversion] ← (~10K candidates, batched, ~30ms)
  │
  ▼
[auction]                    ← compute eCPM = pCTR × pConv × bid
  │  winner
  ▼
[ad delivery + tracking]     ← log impression, await click/conversion
```

### Failure modes

- **Calibration drift**: a model retrained on shifted data may rank well but mis-price → revenue loss.
- **Click bots / fraud**: inflated CTRs from non-human clicks. Detect via behavior anomalies; subtract from billing.
- **Brand safety**: ads shown next to inappropriate content. Filter via content classification.
- **Budget pacing**: a popular ad burns through budget too fast. Smooth via pacing controllers.
- **Cold-start for new advertisers**: no historical CTR. Use bid + basic features; explore via initial impression budget.

### Monitoring

- eCPM by segment.
- Calibration plot per day.
- Advertiser ROI dashboards (CPC, CPA).
- User satisfaction proxy (ad load tolerance, hide-ad rate).
- Pacing fidelity.

---

## 7. Worked example 4 — design fraud detection at a payment processor

**Prompt:** "Design a fraud detection system for credit-card transactions at a Stripe-scale processor."

### Clarification

- Goal: block fraudulent transactions; minimize false positives (legitimate users denied are very costly).
- Scale: ~1000 transactions per second peak; >$1T/year volume.
- Latency: ≤100ms decision (must complete before authorization timeout).
- Cost asymmetry: false negative (fraud succeeds) costs the merchant money. False positive (block a legit txn) costs trust + future spend. Roughly 100:1 dollar-asymmetric depending on segment.

### Frame as ML problem

Real-time binary classification: $p(\mathrm{fraud} | \mathrm{transaction})$. Severely imbalanced (~0.1% fraud).

Decision threshold tuned on cost-weighted precision-recall:
$$
\tau^* = \arg\min_\tau \big[ c_{FN} \cdot \mathrm{FN}(\tau) + c_{FP} \cdot \mathrm{FP}(\tau) \big]
$$

Often two thresholds: hard-block at $\tau_1$, send to review queue between $\tau_1$ and $\tau_2$, allow above $\tau_2$.

### Data

**Transaction features**:
- Amount, currency, merchant ID, MCC code.
- Time, location.

**Velocity features** (the most predictive class):
- Number of transactions from this card in last 1m, 5m, 1h, 24h.
- Distinct merchants in same windows.
- Geographic distance from previous transaction (can't be in NYC and Tokyo within 20 min).
- Amount sum / max in windows.

**Card history features**:
- Account age, prior dispute rate, merchant categories typically used.

**Device / network features**:
- Browser fingerprint, IP, ASN, geolocation, VPN/Tor flags.

**Network features** (graph-based, advanced):
- How many other accounts share this device fingerprint?
- How recently has this card been seen at this merchant?
- Card-merchant clusters: known fraud rings detected via community detection.

### Model

- **Primary**: gradient-boosted trees (LightGBM/XGBoost). Why: fast inference (<5ms), handles missing values natively, robust to mixed feature types, interpretable for review queues.
- **Secondary** (optional): GNN over the card-merchant-device graph for ring detection.
- **Adversarial component**: monitor distribution shift; if attackers shift tactics, model decays fast.

### Class imbalance handling

- Class weights in loss (~100:1).
- Focal loss for very imbalanced datasets.
- Threshold tuned on PR curve, not accuracy.
- Down-sampling negatives during training (calibrate back at inference).

### Real-time feature pipeline

This is the hardest part. Velocity features need real-time aggregations:
- Streaming framework (Kafka + Flink): aggregate into windowed counters per card.
- Online feature store (Redis): served at <5ms read.
- Fallback: if real-time features unavailable, use pre-computed daily aggregates.

### Serving

```
transaction request
  │
  ▼
[hard rules]            ← deterministic blocks (e.g., card on blocklist) <1ms
  │
  ▼
[feature lookup]        ← velocity + history + device <10ms
  │
  ▼
[ML model]              ← GBDT inference ~3ms
  │
  ▼
[threshold + reason]    ← block / review / allow + explanation for support
  │
  ▼
[response to merchant]  ← total <50ms
```

### Adversarial drift

Fraudsters adapt continuously. Mitigations:
- **Frequent retraining**: daily, sometimes hourly for fast-moving attacks.
- **Anomaly monitoring**: flag sudden distribution shifts in features.
- **Active learning**: human reviewers label uncertain cases; back into training data.
- **Adversarial test sets**: red team generates novel attack patterns.

### Monitoring

- **Block rate**: sustained spike → either attack or model regression.
- **Chargeback rate** (lagging label, weeks later): true fraud cost.
- **False positive rate** (customer support tickets, complaints).
- **Distribution shift**: PSI per feature, alerts on >0.25.
- **Feature freshness**: how stale are real-time aggregations?
- **Latency p99**.

### Failure response

- Sudden block-rate spike → temporarily lower threshold, flag for review.
- Model performance drop → roll back to last good model, investigate.
- Real-time feature pipeline down → degrade gracefully to cached daily features (with conservative threshold).

---

## 8. Worked example 5 — design content moderation at scale

**Prompt:** "Design a content moderation system for a social platform with billions of posts/day."

### Clarification

- Goal: identify and act on harmful content (hate, violence, sexual abuse, spam, misinformation).
- Scale: ~1B posts/day across text, image, video.
- Latency: text moderation in-line (<100ms before publish); image/video can be async.
- Action types: remove, demote (lower in feeds), warn, escalate to human.
- Multi-class: dozens of policy categories.
- Multi-modal: text, image, video, audio.

### The moderation pyramid

```
        ┌────────────────────┐
        │  Human moderators  │  ~0.1% of content (highest-uncertainty)
        ├────────────────────┤
        │   LLM judge        │  ~5% of content (uncertain by classifiers)
        ├────────────────────┤
        │  Specialized       │  ~20% of content (per-policy classifiers)
        │  classifiers       │
        ├────────────────────┤
        │   Hash matching    │  ~5% of content (known bad — CSAM, banned)
        ├────────────────────┤
        │  Hard rules        │  ~70% of content (clear OK)
        └────────────────────┘
```

### Data and labels

**Per-policy classifiers**: trained on labeled examples from human moderators. Active learning loop: classifier predicts → low-confidence examples sent to moderators → labels back into training.

**Inter-annotator agreement** is tracked carefully. Hate-speech labels often have <80% agreement among trained moderators — your model can't be more accurate than the labels.

### Architecture

```
post submitted
  │
  ▼
[hard-rule block]            ← exact-match blocklist, banned terms <1ms
  │
  ▼
[hash matching]              ← PhotoDNA, CSAM hashes, audio fingerprints <50ms
  │
  ▼
[fast classifier ensemble]   ← per-policy classifiers (text BERT, image CLIP, etc.) <100ms
  │  if confidence < threshold
  ▼
[LLM judge]                  ← multimodal LLM with policy in system prompt <500ms (async OK)
  │  if still uncertain
  ▼
[human review]               ← async, hours-to-days
  │
  ▼
[action: allow / demote / remove / suspend]
```

### Multi-modal classifiers

- **Text**: fine-tuned RoBERTa or similar; one head per policy.
- **Image**: CLIP-based; or end-to-end CNN trained on policy labels.
- **Video**: frame-sampled CLIP + temporal aggregation; or a video transformer.
- **Audio**: Whisper (transcribe) + text classifier; or audio embedding directly.
- **Cross-modal**: text + image jointly (e.g., for sarcasm where text contradicts image).

Each policy has its own threshold; the union across policies determines action.

### Adversarial robustness

Bad actors actively evade:
- Text: misspellings, leetspeak, foreign characters.
- Images: small perturbations, blur, watermarks over banned content.
- Video: rotated, mirrored, sped up.

Defenses:
- Augmented training with adversarial examples.
- Multi-modal redundancy (text + image must both pass).
- Feedback loop from "appeal granted / denied" labels.

### Monitoring

- Per-policy precision and recall (via human-labeled audit set).
- Appeals rate by policy (high appeals → over-moderation).
- Time-to-removal for severe policy violations.
- Coverage by language and region.
- Inter-annotator agreement on audit sets.
- Reviewer wellness metrics (this is real — moderation work is psychologically heavy).

### Trade-offs

- **Precision vs recall**: high precision avoids over-moderation but lets harmful content slip through. Tune per policy: very high recall on CSAM (zero tolerance), balanced on borderline hate speech.
- **Latency vs accuracy**: in-line synchronous adds latency but prevents harmful content from being seen. Async catches more but lets some content reach users.
- **Coverage vs cost**: more LLM-judge calls catch more borderline cases but cost more.

---

## 9. Worked example 6 — design an LLM serving platform

**Prompt:** "Design an LLM serving platform like the OpenAI API."

### Clarification

- Goal: serve LLM inference at low latency and high throughput.
- Scale: 1M+ requests/sec across multiple model sizes.
- Latency: TTFT < 500ms p95, ITL < 50ms p95 for chat.
- Models: 7B, 70B, 200B+ (some MoE).
- Multi-tenant: thousands of customers with rate limits and SLAs.

### Architecture

```
user request
  │
  ▼
[API gateway]              ← auth, rate limit, routing
  │
  ▼
[model router]             ← chooses which serving fleet (7B vs 70B etc.)
  │
  ▼
[batch scheduler]          ← admits to a continuous-batching engine
  │
  ▼
[serving engine: vLLM / TensorRT-LLM / SGLang]
   │
   ├─ Prefill workers      ← compute-bound, large batches
   └─ Decode workers       ← memory-bound, smaller batches (disaggregated serving)
   │
   ▼
[KV cache: paged, optionally LMCache for spillover to CPU/disk]
   │
   ▼
[response stream]
```

### Key components

- **Continuous batching** (Orca / vLLM): admit new requests at decode-step granularity. Iteration-level scheduling.
- **PagedAttention**: KV cache as fixed-size blocks, no fragmentation.
- **Prefix caching**: shared system-prompt/RAG-context KV reused across requests.
- **Disaggregated serving** (Mooncake / DistServe): split prefill (compute-bound) and decode (memory-bound) into separate worker fleets.
- **Speculative decoding**: small draft model + EAGLE heads → 2-3× ITL speedup.
- **Quantization**: FP8 on H100, FP4 on Blackwell. KV cache quantized to INT8.
- **Multi-LoRA serving**: many fine-tuned LoRA adapters served from one base model (Punica, S-LoRA).

### KV cache management

KV cache memory is the binding constraint on concurrent users. Strategies:
- Block-based allocation (PagedAttention).
- Prefix sharing (block reference counting).
- Eviction (StreamingLLM with attention sinks for very long contexts).
- KV quantization (INT8 / FP8).
- LMCache: spillover to CPU/disk for long-tail KV.

### Multi-tenancy

- **Quotas**: per-customer requests/sec, tokens/min.
- **Priority queues**: paid tiers get priority during congestion.
- **Isolation**: model cache hits are per-tenant for prompt caching (privacy).

### Latency optimizations

- **TTFT**: chunked prefill prevents long prompts from blocking decode.
- **ITL**: speculative decoding + KV optimization.
- **Streaming**: tokens delivered to user as generated.
- **Prompt caching**: 90% cost discount on cache hits in modern APIs.

### Cost / capacity planning

For a 70B model on 8× H100:
- Memory: 140 GB weights + ~80 GB KV + ~20 GB activations ≈ 240 GB total. Fits with TP=2 across 2 GPUs minimum.
- Throughput at batch 32, 1K input + 200 output: ~5K-10K tokens/sec/GPU.
- Cost per 1M tokens: ~$0.3-1 depending on demand and batch utilization.

### Monitoring

- p50/p95/p99 of TTFT and ITL.
- Tokens-per-second per worker.
- KV cache utilization.
- Speculative-decode acceptance rate.
- Per-tenant quota / abuse signals.
- GPU memory pressure → triggers eviction or rebalance.

### Failure modes

- **OOM during prefill**: long-prompt request crashes worker. Mitigations: chunked prefill, max-prompt limits per tier, soft preemption.
- **KV cache fragmentation**: solved by PagedAttention, but block size choice still matters.
- **Hot-tenant traffic spike**: rate limits + circuit breakers + priority demotion.
- **Model decay** (rare for inference but happens with adapters): monitor calibration of safety classifiers etc.

---

## 10. Worked example 7 — design semantic image search

**Prompt:** "Design semantic image search for an e-commerce site (find products from photos)."

### Clarification

- Goal: user uploads or describes a photo → return matching products.
- Scale: 10M-100M product images; 1M queries/day.
- Latency: <500ms.
- Modes: image-to-image, text-to-image, image-to-text.

### Architecture

```
query (image or text)
  │
  ▼
[CLIP-style encoder]            ← image: ViT; text: text transformer; shared 512-d space
  │
  ▼
[ANN index — HNSW or IVF-PQ]    ← over product image embeddings
  │  top 1000
  ▼
[metadata + business filters]   ← in-stock, price range, region availability
  │  top 200
  ▼
[reranker (cross-encoder)]      ← (optional) for high-stakes queries
  │  top 20
  ▼
[response]
```

### Embeddings

- **CLIP-style** (or SigLIP, EVA-CLIP, modern variants): joint image-text embedding.
- Fine-tuned on the product catalog with click pairs as positive (text query, clicked image).
- Re-indexed weekly as catalog updates.

### Hard problems

- **Catalog freshness**: new products need to be indexed within hours.
- **Long-tail products**: rare items need feature engineering (more views, augmented training).
- **Multi-modal queries**: "blue jeans like the photo but cheaper" requires joint reasoning.
- **Visual similarity vs intent**: photo of red shoes — does the user want that exact style or a similar style? Often intent.

### Production touches

- Pre-compute product embeddings; store in vector DB.
- Image upload pipeline: resize, EXIF-strip, run through CLIP, search.
- Caching: popular queries cached; embedding-of-query cached briefly.
- Diversity / category mixing in final results.

---

## 11. Cross-cutting questions interviewers love to ask

After your design, expect probes. Have answers ready:

### "What if you only had 1 month to ship this?"

Cut everything but the core path. For a recommender: use a simple two-tower with off-the-shelf CLIP embeddings + popularity fallback + LightGBM on top. Skip multi-task heads, skip diversity reranker. Ship → measure → iterate.

### "What changes if scale is 10x?"

For 10x users:
- More serving replicas (linear).
- Larger ANN indexes — switch HNSW → IVF-PQ.
- Stronger feature store sharding.
- Daily retraining → maybe hourly.

For 10x catalog:
- ANN index time grows; switch to billion-scale infrastructure (FAISS GPU, or graph-quantized indexes).
- Cold-start gets harder — more reliance on content features.

### "What if labels are expensive?"

- Active learning: uncertain examples sent to humans first.
- Weak supervision: rules + models combine.
- Self-supervised pretraining + small labeled fine-tune.
- Synthetic data from LLMs.

### "How would you deal with bias / fairness?"

- Audit: per-segment performance metrics (gender, age, geography, language).
- Constrained optimization: minimize worst-segment loss instead of average.
- Counterfactual generation: would the model predict differently if a sensitive attribute changed?
- Human review of high-impact decisions.

### "What if the data distribution shifts?"

- Continuous monitoring (PSI, KL on inputs and outputs).
- Automatic retraining triggered by drift.
- Fallback to simpler model (less likely to amplify drift effects).
- Holdback group on old model to detect shifts.

### "How do you handle GDPR / privacy?"

- Data minimization: don't log what you don't need.
- Right to erasure: pipeline to delete user from training data and rebuild affected models on schedule.
- Differential privacy: train with DP-SGD for sensitive features.
- Federated learning: train on-device for very sensitive data.
- Per-region data isolation (Schrems II).

### "How do you cost this out?"

Walk through:
- Compute cost: GPU/CPU hours per training run × frequency.
- Storage: feature store, embedding indexes, logs.
- Serving: GPU/CPU instances at peak QPS × replication.
- Network: egress especially for media-heavy systems.
- Human-in-loop costs.

A mid-sized recommender might run $1-10M/year in infra. An ad system or LLM platform: $100M-1B/year.

---

## 12. The 12 most-asked ML system design prompts

Drill the framework on each. For each: (a) what you'd ask, (b) the architecture, (c) the gotcha.

1. **YouTube recommender** — two-stage retrieve-and-rank, multi-task ranker, watch-time objective; gotcha: filter bubbles + clickbait.
2. **Google search** — multi-stage with BM25 + dense + cross-encoder + LLM judge; gotcha: position bias + spam adversaries.
3. **Ads ranking** — eCPM = pCTR × pConv × bid; gotcha: calibration drift kills revenue.
4. **Stripe-like fraud** — GBDT + velocity features + real-time pipeline; gotcha: adversarial drift requires daily retraining.
5. **Content moderation** — hash → fast classifiers → LLM judge → human review; gotcha: inter-annotator agreement.
6. **LLM serving (OpenAI-like)** — vLLM/SGLang + paged KV + continuous batching + speculative + prompt caching; gotcha: KV cache memory binds throughput.
7. **Image search** — CLIP + ANN + reranker; gotcha: catalog freshness.
8. **Spotify-like music recommender** — collab filter + audio embeddings + sequential model; gotcha: skip rate as negative signal.
9. **Uber ETA prediction** — gradient-boosted regression + map + traffic features; gotcha: surge pricing + driver-side bias.
10. **AirBnB price suggestion** — regression with location features + image quality + market features; gotcha: endogeneity (past prices ↔ past demand).
11. **LinkedIn "people you may know"** — graph neural net + features; gotcha: privacy-respecting (don't reveal sources).
12. **News feed (FB/Insta)** — multi-objective ranker (engagement + diversity + integrity) + freshness; gotcha: misinformation amplification.

---

## 13. The senior-level discussion points

After the design, the interviewer wants to test seniority. Strong topics to volunteer:

- **Multi-objective optimization**: balancing engagement, revenue, integrity, fairness.
- **Counterfactual evaluation**: how to measure off-policy without full A/B.
- **Long-term vs short-term**: holdback experiments to measure 30/60/90-day effects.
- **Position bias and counterfactual learning**: IPS in click-based training.
- **Model decay and continuous training**: pipeline maintenance vs model maintenance.
- **Cross-team coordination**: features owned by team A, models by team B, serving by team C.
- **Cost trade-offs**: when GBDT is more economical than DL, when to upgrade, when to compress.
- **Privacy & differential privacy**: practical DP-SGD costs.

Interviewers love when you bring up one of these unprompted because it shows you've thought about systems beyond just "the model."

---

## 14. The 5-minute structured answer (what to actually say)

When the question lands:

```
[1 min — clarify]
"To make sure I'm solving the right problem, three questions:
 - What's the primary success metric?
 - What's the scale (QPS, catalog size)?
 - What's the latency budget?"
[interviewer answers]

[1 min — frame]
"OK, this is a [recommendation / classification / ranking] problem.
 At [scale], two-stage retrieve-and-rank is standard. I'll structure
 my answer as: data → architecture → serving → monitoring."

[5 min — architecture]
[walk through retrieval + ranking + reranker; draw boxes if on whiteboard]

[3 min — serving]
[latency budget, deployment, fallbacks]

[2 min — monitoring]
[infra + model + business metrics]

[discuss extensions for the rest]
```

Practice this until it's automatic. Most candidates wing it; you should have the structure memorized.

---

## 15. Drill plan

- For each of the 12 prompts in §12, do a 25-min mock answer: 5 min clarify+frame, 10 min architecture, 5 min serving, 5 min monitoring + failure modes.
- Time yourself. Senior interviews want a complete answer in 30-40 min.
- For each prompt, write down the 1-2 specific failure modes the interviewer would push on.
- For each prompt, identify 2 cross-cutting questions (§11) you'd answer well on.

Do 3 prompts/day for a week → 21 mocks. By the end, the framework is muscle memory.

---

## 16. Further reading

- Huyen, *Designing Machine Learning Systems* (2022). The single best book on this.
- Kleppmann, *Designing Data-Intensive Applications*. Not ML-specific but the data-systems foundation you need.
- *Real-World ML* (Chip Huyen blog) — short, current essays.
- Tech blogs: Uber, Lyft, Pinterest, Meta, Netflix engineering. The "we built X" posts are essentially solved system designs.
- Anthropic, OpenAI engineering blogs — when published, contains relevant LLM-systems material.
- Kohavi, Tang, Xu, *Trustworthy Online Controlled Experiments* — for the A/B testing layer.
