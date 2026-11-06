# ML system design

<span class="practice-page"></span>

**Foundation explainer:** Core material · **Priority:** Very important

Very important: turn a vague product request into a measurable, operable system.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f081"></a>

## F081. Structure an ML system-design answer.

**Start here.** Start from the decision the system must support, then connect data, modeling, serving and evaluation.

**Work it through.**

1. Clarify users, target, prediction time, latency, scale and error costs; state assumptions if details are missing.
2. Design data collection and labels, a baseline, offline evaluation, serving interfaces and feedback loops.
3. Estimate capacity and discuss failures, monitoring, rollback and abuse. Spend deep-dive time on the hardest constraint rather than listing every database technology.

**Small example.** For spam detection, the correct latency and false-positive cost differ between pre-publication blocking and later review.

**Common trap.** Drawing infrastructure before defining what counts as a correct model decision.

**Follow-up.** What should you ask first? What action follows the prediction, and which errors are most costly?

---

<a id="f082"></a>

## F082. Design a recommendation system from candidate generation to ranking.

**Start here.** A cascade reduces a huge catalog to a small set worth scoring with expensive models.

**Work it through.**

1. Generate candidates using complementary signals such as collaborative, content and popularity retrieval.
2. Rank with user, item and context features, ensuring training features were available at event time.
3. Apply policy and diversity constraints, log exposures and outcomes, and evaluate retrieval recall, ranking utility and online guardrails separately.

**Small example.** Retrieve 1000 candidates from a million items, rerank 100, then display 10; each stage trades compute for selectivity.

**Common trap.** Training only on clicked items and treating unexposed items as observed negatives.

**Follow-up.** How do you handle a new item? Content features, controlled exploration and uncertainty-aware fallback rather than waiting for interaction history.

---

<a id="f083"></a>

## F083. Design multi-modal harmful-content detection.

**Start here.** Use a calibrated triage system that combines automated decisions with human review capacity.

**Work it through.**

1. Define harm categories, policy versions and separate precision/recall targets per action and category.
2. Run cheap per-modality filters, then stronger joint models on uncertain cases; reserve pre-publication and asynchronous paths with separate latency budgets.
3. Prioritize a bounded review queue, audit decisions and retrain from reviewed labels with sampling-bias checks.

**Small example.** A high-confidence severe violation can be blocked, while ambiguous content may receive limited distribution and human review under the specified policy.

**Common trap.** Optimizing one aggregate accuracy metric while hiding catastrophic errors on a rare category.

**Follow-up.** How do policy changes affect old decisions? Version models and policy rules, and define which content must be re-evaluated.

---

<a id="f084"></a>

## F084. Design fraud or account-takeover detection.

**Start here.** Predict risk at a concrete event and route it to an action whose cost is explicitly understood.

**Work it through.**

1. Collect time-valid device, account and behavior features with privacy and access controls.
2. Use a fast risk model and a policy layer for allow, challenge, review or block; tune thresholds to error costs and review capacity.
3. Evaluate on time-based splits with mature labels, account for adversarial adaptation and monitor false positives by user segment.

**Small example.** A suspicious login can trigger a step-up challenge rather than an irreversible account block, reducing the cost of uncertain predictions.

**Common trap.** Leaking post-investigation labels or future account events into training-time features.

**Follow-up.** How do attackers change the problem? Their behavior adapts to the model and creates feedback-driven distribution shift.

---

<a id="f085"></a>

## F085. Design image-text retrieval.

**Start here.** Learn or use embeddings that place matched images and text near each other, then search and rerank efficiently.

**Work it through.**

1. Build cleaned matched-pair data with duplicate and entity-aware evaluation splits.
2. Train dual encoders with an appropriate contrastive objective, then precompute catalog embeddings and build an index.
3. Evaluate recall@K, hard-negative behavior, latency and subgroup performance; use a stronger reranker if the quality gain justifies cost.

**Small example.** A text query retrieves candidate images by embedding similarity, while a pairwise reranker checks the top candidates more carefully.

**Common trap.** Splitting near-duplicate images across training and test or interpreting false negatives as unrelated pairs automatically.

**Follow-up.** What happens when the encoder changes? Version and rebuild the index; query and item embeddings must remain compatible.

---

<a id="f086"></a>

## F086. Design a feature platform with online/offline consistency.

**Start here.** A feature definition must preserve its meaning both in historical training data and at live prediction time.

**Work it through.**

1. Define event time, availability time, entity keys, transformations, ownership and freshness targets.
2. Use point-in-time joins for historical features and a serving store for low-latency current values.
3. Version transformations, monitor missingness and freshness, and support replay and backfills without silently leaking future information.

**Small example.** A seven-day transaction count for a prediction on Monday must exclude transactions only observed on Tuesday.

**Common trap.** Assuming identical column names guarantee identical online and offline values.

**Follow-up.** How do late events affect correctness? Use explicit watermark and correction policies, and distinguish historical truth from what was known then.

---

<a id="f087"></a>

## F087. Design an online model rollout with rollback.

**Start here.** A model artifact is only one part of a deployable, reversible prediction system.

**Work it through.**

1. Version the model with preprocessing, feature contracts, configuration and decision thresholds.
2. Run shadow traffic or a small canary with health, latency and quality guardrails; compare consistent cohorts and label windows.
3. Keep the previous compatible bundle available and automate rollback on operational failures, while handling delayed quality signals separately.

**Small example.** Rolling back weights alone fails if the new feature schema is incompatible with the old model.

**Common trap.** Treating a healthy server response as proof that predictions remain correct.

**Follow-up.** What is shadow deployment good for? Observing behavior and operational cost without letting the new model drive user decisions.

---

<a id="f088"></a>

## F088. Design an LLM evaluation and annotation platform.

**Start here.** Treat examples, rubrics, judgments and model versions as versioned experimental data.

**Work it through.**

1. Create task schemas, sampled evaluation sets and annotation instructions with gold checks and disagreement resolution.
2. Store immutable model outputs and record reviewer identity, rubric version and adjudication history with appropriate access controls.
3. Provide reproducible aggregate metrics, slice analysis, regression comparisons and an audit trail; avoid exposing benchmark answers into training workflows accidentally.

**Small example.** Changing a rubric halfway through a comparison can create an apparent model improvement without any model change.

**Common trap.** Using annotation speed as the only reviewer-quality metric or overwriting previous judgments irreversibly.

**Follow-up.** How should disagreements be used? Diagnose ambiguity and systematic errors before collapsing them into one label.

---

<a id="f089"></a>

## F089. Mine trustworthy training examples from noisy interaction logs.

**Start here.** A recorded trajectory is not automatically a valid demonstration of the desired behavior.

**Work it through.**

1. Define success criteria, allowed observation/action fields and privacy filtering; deduplicate by task or session.
2. Filter incomplete, corrupted or failed trajectories, then label steps or outcomes under an explicit policy. Preserve ordering and provenance within the internal dataset without including private content unnecessarily.
3. Split by task, user or time as appropriate, and test whether a small clean subset improves a fixed evaluation before scaling collection.

**Small example.** A successful final screen may follow many accidental clicks; copying every action as equally good supervision teaches noise.

**Common trap.** Using future success metadata as an input feature that would not exist during live action selection.

**Follow-up.** How would you prioritize annotation? Choose uncertain, diverse and high-impact trajectories rather than merely the longest logs.

---

<a id="f090"></a>

## F090. Design a rate-limited LLM classification pipeline.

**Start here.** Separate durable work tracking from bounded model calls so retries do not duplicate or lose results.

**Work it through.**

1. Ingest items with stable ids into a durable queue and deduplicate work by an idempotency key.
2. Use bounded concurrency, token/request budgets, deadlines and exponential backoff for retryable failures; validate each output against the schema and label set.
3. Persist versioned results, route permanent failures to review and measure throughput, cost, queue age and end-to-end accuracy.

**Small example.** If a request times out after the provider processed it, a retry must not create two independent downstream records for the same item.

**Common trap.** Dropping invalid responses from accuracy calculations or retrying malformed inputs forever.

**Follow-up.** How do you handle backlog growth? Scale within budget, reduce per-item cost or apply admission and prioritization policies.

---
