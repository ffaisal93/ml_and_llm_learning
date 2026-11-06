# Evaluation and data quality

**Foundation explainer:** Core material · **Priority:** Very important

Very important: distinguish a good benchmark number from a dependable decision.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f061"></a>

## F061. Why can cross-entropy worsen while accuracy stays unchanged?

**Start here.** Accuracy uses the chosen class; cross-entropy also measures how much probability the model assigns to the true class.

**Work it through.**

1. Separate correct and incorrect predictions and inspect confidence changes.
2. For a true-class probability p, loss is -log(p), which grows without bound as p approaches zero.
3. Track validation loss, accuracy and calibration together; consider early stopping or calibration using a separate validation set.

**Small example.** An already-wrong example whose true-class probability falls from 0.1 to 0.001 keeps the same error count while loss grows from 2.303 to 6.908.

**Common trap.** Concluding that accuracy and confidence quality must improve or degrade together.

**Follow-up.** Can a highly accurate classifier be poorly calibrated? Yes; its confidence can systematically exceed its observed correctness.

---

<a id="f062"></a>

## F062. Bound cross-entropy when a prediction is correct or incorrect.

**Start here.** The label probability is constrained by whether it is the largest class probability, not merely by the number of classes.

**Work it through.**

1. For C classes and a correct argmax, p_true>=1/C, so 0<=loss<=log(C), subject to tie conventions.
2. For an incorrect argmax, p_true<=1/2 and loss>=log(2), with the lower bound an infimum when strict inequality is required.
3. There is no finite upper bound for an incorrect prediction as p_true approaches zero. For N examples with k errors, the mean-loss lower bound is (k/N)*log(2). If k=0 the upper bound is log(C); if k>0 there is no finite upper bound.

**Small example.** For 10 classes, a correct prediction's loss cannot exceed log(10) under an ordinary argmax decision; an incorrect prediction can have arbitrarily large loss.

**Common trap.** Giving log(C) as an upper bound on every example, including confident mistakes.

**Follow-up.** With k>0 errors in N examples, is mean loss bounded above? No; even one error can have unbounded loss.

---

<a id="f063"></a>

## F063. Choose metrics for ranking and recommendation.

**Start here.** The right metric should reflect the ordered choices users see and the value of a good recommendation.

**Work it through.**

1. For retrieval, inspect recall@K to ensure relevant items survive candidate generation.
2. For ranking, use an order-sensitive measure such as NDCG@K or a task-specific utility measure, and define relevance labels and position discounts.
3. Validate online impact with guardrails for latency, diversity, safety and long-term outcomes; correct or acknowledge exposure and selection bias.

**Small example.** Putting the only relevant item at rank 1 and rank 20 can give the same unranked recall but very different user value.

**Common trap.** Optimizing clicks without checking whether the model promotes low-quality or misleading content.

**Follow-up.** Why separate retrieval and ranking evaluation? A ranker cannot recover a relevant item that retrieval excluded.

---

<a id="f064"></a>

## F064. Design an A/B test for a new ML ranking model.

**Start here.** Randomization creates a comparison group, but the experiment must match how the product and its users interact.

**Work it through.**

1. Define a primary outcome, guardrails, exposure logging, randomization unit and minimum detectable effect before launch.
2. Check sample-ratio mismatch, contamination, novelty effects and whether observations are independent at the chosen unit.
3. Estimate treatment effects and uncertainty, follow the stopping plan, and examine meaningful predeclared slices without chasing noise.

**Small example.** If users see many recommendations, randomizing each impression can cause cross-treatment contamination; user-level randomization may fit better.

**Common trap.** Repeatedly checking ordinary fixed-horizon significance and stopping at the first favorable result.

**Follow-up.** When might a switchback experiment help? Shared marketplaces or systems with interference where treatment changes across time blocks.

---

<a id="f065"></a>

## F065. Distinguish data drift, concept drift and label delay.

**Start here.** A production metric can change because inputs changed, the input-to-label relationship changed, or the labels are not yet mature.

**Work it through.**

1. Check pipeline health, feature distributions and missingness before assuming the model changed.
2. When mature labels exist, compare conditional performance and calibration on matched slices and periods.
3. Separate delayed outcome windows from true degradation; use operational proxies cautiously and validate them against final outcomes.

**Small example.** A fraud model's observed precision may look low immediately after deployment because confirmed fraud labels arrive weeks later.

**Common trap.** Automatically retraining on a distribution alarm without checking whether labels or upstream features are broken.

**Follow-up.** Can input distributions stay stable while performance changes? Yes; concept drift can alter P(y|x) without an obvious marginal shift in x.

---

<a id="f066"></a>

## F066. Explain a regressing model to a product manager.

**Start here.** Translate a metric change into affected decisions, user impact and a concrete next action.

**Work it through.**

1. State what changed, how reliable the evidence is, which users or slices are affected and the estimated cost.
2. Check deployment differences, feature freshness, label windows and threshold changes; distinguish observations from hypotheses.
3. Recommend rollback, a limited canary or targeted repair with clear success criteria and an owner.

**Small example.** “Recall dropped from 80% to 70% on the same mature-label cohort, adding about 100 missed cases per 1000 actual positives” is more actionable than “the model drifted.”

**Common trap.** Blaming an algorithm before checking data and operational changes, or hiding uncertainty behind technical vocabulary.

**Follow-up.** What if the offline metric improved? Investigate deployment mismatch and whether the metric captures the real product objective.

---

<a id="f067"></a>

## F067. Build an LLM evaluation harness.

**Start here.** An evaluation harness is a reproducible experiment, not just a loop over prompts.

**Work it through.**

1. Version examples, prompts, model settings, scoring rules and code; keep development and final evaluation sets separate.
2. Store outputs and failure types; use deterministic checks where possible and validated human or model judgments where necessary.
3. Report per-task results, cost, latency and uncertainty. Make retries and missing outputs visible rather than silently dropping failures.

**Small example.** A model that answers 90 of 100 requests correctly and times out on 10 has 90% end-to-end success, not 100% accuracy over completed requests.

**Common trap.** Using an LLM judge's score as ground truth without measuring agreement, bias or stability.

**Follow-up.** How do you compare two versions efficiently? Paired examples and paired resampling can isolate differences better than unrelated samples.

---

<a id="f068"></a>

## F068. Check calibration and choose a decision threshold.

**Start here.** A probability estimate and the action taken from it are separate design choices.

**Work it through.**

1. Measure reliability on a validation set using bin plots and proper scoring rules such as log loss or Brier score.
2. Fit a calibration transform only on held-out predictions if needed, then choose a threshold using explicit costs or operational capacity.
3. Evaluate the entire calibrated-and-thresholded pipeline on untouched test data and important slices.

**Small example.** Under ideal calibrated probabilities and fixed costs, predict positive when p>C_FP/(C_FP+C_FN). With costs 1 and 9, the threshold is 0.1.

**Common trap.** Tuning calibration, threshold and final reported performance on the same test set.

**Follow-up.** What happens under prevalence shift? Probability calibration and the best operating point may change.

---

<a id="f069"></a>

## F069. Audit human-labeled training data before scaling it.

**Start here.** More labels help only if they carry the intended signal and match the task definition.

**Work it through.**

1. Inspect the sampling frame, instructions, class balance, duplicates, annotator disagreement and gold-label checks.
2. Separate ambiguous examples from clear errors; compare disagreement across languages, subgroups and content types.
3. Run a controlled data-quality intervention against a fixed model and evaluation set before attributing improvements to architecture.

**Small example.** Two annotators can disagree because the policy is unclear, not because one is a spammer; rewriting the rubric may help more than filtering a rater.

**Common trap.** Removing every minority label and thereby erasing legitimate edge cases or subgroup perspectives.

**Follow-up.** How would you prioritize a review budget? Combine uncertainty, business cost, diversity and suspected systematic errors.

---

<a id="f070"></a>

## F070. Explain uncertainty without confusing it with confidence.

**Start here.** A model's output probability is not automatically a trustworthy measure of every kind of uncertainty.

**Work it through.**

1. Distinguish irreducible label variation from uncertainty due to limited data or model assumptions.
2. Check calibration and out-of-distribution behavior; use ensembles, resampling or conformal methods only under their stated assumptions.
3. Connect uncertainty to a policy such as abstention or human review, evaluating both coverage and error on the retained predictions.

**Small example.** A softmax output of 0.99 on a completely unfamiliar image can still be wrong; normalization over known classes does not detect unfamiliarity by itself.

**Common trap.** Calling a narrow confidence interval protection against a badly specified data-generation model.

**Follow-up.** What tradeoff does abstention create? Better accuracy on answered cases at the cost of lower automatic coverage.

---
