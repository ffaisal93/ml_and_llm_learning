# Additional product ML case studies

<span class="practice-page"></span>

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: practice adapting the core design framework to a concrete product.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f181"></a>

## F181. Predict the category of a newly listed item.

**Start here.** Frame the task around the available item fields and a category taxonomy that can change.

**Work it through.**

1. Define single-label versus multi-label output, hierarchy and costs of misclassification.
2. Build text/image baselines with entity-aware splits, handle rare and new categories and allow abstention or suggested categories when uncertain.
3. Serve with taxonomy/model versioning and collect corrected labels without assuming every user edit is reliable.

**Small example.** Predicting a parent category may be acceptable when the fine-grained leaf is uncertain, if the product supports that fallback.

**Common trap.** Using a category name entered after human review as a training feature for the pre-review prediction.

**Follow-up.** How do you evaluate hierarchy errors? Confusing sibling leaves may have a different cost from predicting the wrong top-level branch.

---

<a id="f182"></a>

## F182. Recommend cover photos to improve bookings.

**Start here.** Separate predicting which photo attracts clicks from estimating which selection improves a meaningful booking outcome.

**Work it through.**

1. Define eligible photos, user context, outcome window and safety/quality constraints.
2. Build image and listing-context features, handling exposure bias in historic cover-photo choices.
3. Evaluate online bookings and guardrails with an appropriate randomized policy; monitor whether clickbait photos increase clicks but reduce qualified bookings.

**Small example.** A visually striking photo can get more clicks while misrepresenting the property and increasing abandonment.

**Common trap.** Training on observed clicks without accounting for which photos were ever shown as covers.

**Follow-up.** How do new listings get started? Content features and controlled exploration, with a safe default.

---

<a id="f183"></a>

## F183. Estimate listing lifetime value.

**Start here.** The target combines future activity, retention and value over a specified horizon.

**Work it through.**

1. Define the unit, value measure, horizon, discounting if relevant and censoring policy.
2. Build time-valid features, compare simple cohort and regression baselines and handle incomplete outcome windows explicitly.
3. Evaluate calibration across value ranges and listing ages, connecting errors to the actual ranking or investment decision.

**Small example.** A new listing cannot yet have a complete twelve-month realized value label; treating its observed first month as the final value introduces censoring bias.

**Common trap.** Mixing different observation horizons without correction.

**Follow-up.** Why might a two-stage model help? Separate probability of future activity from value conditional on being active.

---

<a id="f184"></a>

## F184. Design personalized short-video ranking.

**Start here.** Optimize a measured user outcome while controlling feedback loops and content-quality risks.

**Work it through.**

1. Generate candidates from user history, similarity, freshness and exploration channels.
2. Rank with time-valid user, video and context features, distinguishing watch duration from completion and repeated exposure.
3. Evaluate long-term satisfaction, retention, diversity and safety alongside engagement; handle new users and creators explicitly.

**Small example.** A short video can have a high completion rate simply because it is short, so completion needs duration-aware interpretation.

**Common trap.** Treating every unshown video as a negative example or optimizing watch time without quality guardrails.

**Follow-up.** How do you limit popularity feedback loops? Exploration, exposure-aware learning and evaluation on underexposed content.

---

<a id="f185"></a>

## F185. Rank news without rewarding stale or misleading content.

**Start here.** News ranking needs relevance, freshness and quality constraints that can conflict.

**Work it through.**

1. Retrieve by user interests, current topics and recency, with deduplication of syndicated stories.
2. Rank using relevance and freshness features, keeping quality rules and user controls explicit.
3. Evaluate useful engagement, topic coverage and freshness under temporal splits and online experiments.

**Small example.** Ten near-identical articles about one event can inflate click opportunities while reducing coverage of other useful stories.

**Common trap.** Using random train/test splits that let tomorrow's news patterns leak into yesterday's prediction setting.

**Follow-up.** What if the user has no history? Context, location at an appropriate privacy level, broad preferences and transparent defaults can support a cold start.

---

<a id="f186"></a>

## F186. Design a hiring-workflow assistant without making unsupported automated decisions.

**Start here.** Keep evidence organization, workflow help and consequential human decisions clearly separated.

**Work it through.**

1. Define allowed tasks such as summarizing approved materials, answering grounded process questions and scheduling.
2. Enforce access controls, evidence-linked summaries, privacy retention and human review for consequential assessments.
3. Evaluate factual fidelity, omissions, subgroup failure patterns and auditability rather than relying only on a fluent-summary score.

**Small example.** A summary should identify what a candidate actually stated instead of inventing a qualification from a vague phrase.

**Common trap.** Treating a language-model score as a validated hiring decision rule.

**Follow-up.** What should an audit record retain? The relevant authorized evidence, versioned processing configuration and review outcome under the retention policy.

---

<a id="f187"></a>

## F187. Design a support-ticket routing model.

**Start here.** Routing is a decision under capacity constraints, often with ambiguous labels and changing team ownership.

**Work it through.**

1. Define destinations, escalation conditions and the cost of wrong routing versus human triage.
2. Train a text/context baseline using time-valid ticket data, and handle unknown or low-confidence classes explicitly.
3. Integrate queues, overrides and feedback while separating noisy agent transfers from gold routing labels.

**Small example.** A rare critical incident may deserve a recall-focused escalation rule rather than the same threshold used for routine billing tickets.

**Common trap.** Using final resolution notes as input for the initial routing prediction.

**Follow-up.** How do you evaluate beyond accuracy? Time-to-resolution, transfer rate, critical misses and queue balance.

---

<a id="f188"></a>

## F188. Design a flight-search and booking agent.

**Start here.** Searching is reversible; booking changes external state and needs explicit, validated execution boundaries.

**Work it through.**

1. Represent dates, passengers, constraints and price limits in structured state, clarifying missing critical fields.
2. Search and compare options, revalidate availability and total price, then obtain the required user authorization before purchase.
3. Use idempotency and a durable booking state machine; communicate confirmation or uncertain status honestly and avoid duplicate retries.

**Small example.** A flight found at one price may change before booking, so the final action must validate the current terms.

**Common trap.** Treating a search result as a reservation or reporting success before a confirmed booking identifier exists.

**Follow-up.** What if payment times out? Query transaction status or reconcile safely rather than blindly submitting another payment.

---

<a id="f189"></a>

## F189. Detect possible image copyright violations as a retrieval problem.

**Start here.** Similarity can flag candidates for review, but visual resemblance alone does not establish legal rights or infringement.

**Work it through.**

1. Index authorized reference images with metadata about ownership or permissions as available.
2. Retrieve exact and transformed matches using hashes and embeddings, then use stronger verification for likely matches.
3. Route uncertain cases to the appropriate review process, measuring retrieval recall and false flags separately from legal determinations.

**Small example.** A resized or cropped copy can defeat a byte hash while remaining retrievable by a perceptual representation.

**Common trap.** Calling every close embedding match an infringement.

**Follow-up.** What hard negatives matter? Stock images, public-domain works, logos and visually similar but independently created content.

---

<a id="f190"></a>

## F190. Classify viewers using movie completion behavior.

**Start here.** The definition of completion and the observation window determine the label before modeling begins.

**Work it through.**

1. Define eligible starts, completion thresholds, replays, partial sessions and late events.
2. Construct user/item/context features available at the intended prediction time and split by time or entity as appropriate.
3. Compare a simple baseline, inspect duration and user-segment effects and evaluate the downstream intervention rather than classification alone.

**Small example.** Watching 90 minutes means different completion fractions for a 95-minute film and a 180-minute film.

**Common trap.** Including the final watched duration as a feature when predicting completion at playback start.

**Follow-up.** What if the question is descriptive rather than predictive? Use a clear aggregation rule instead of adding an unnecessary ML model.

---
