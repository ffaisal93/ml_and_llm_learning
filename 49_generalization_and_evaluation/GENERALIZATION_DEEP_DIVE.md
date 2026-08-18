# Generalization and Evaluation — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

The single most common reason ML systems fail in production: the offline metric was wrong, the test set was contaminated, or the model didn't generalize the way you expected. This deep dive is about *not getting fooled by your own evaluation*.

---

## 1. Data leakage — what it is and how to spot it

**Leakage** = information from outside the training data leaks into the model, inflating offline metrics that don't transfer to deployment.

> **Saying it out loud.** Leakage is when the model gets information during training that it won't have when it's actually making predictions. The symptom is always the same — the offline numbers are spectacular and production is a disaster — and if your AUC is 0.99 on a hard problem, the right instinct is suspicion rather than celebration. The single most useful check is temporal: for every feature, ask whether that exact value would have been available at the moment of prediction, tracing the real pipeline rather than the schema. It's the most common reason a promising model dies on contact with reality.

### Common forms

**Target leakage**: a feature is a downstream consequence of the target. Examples:
- Predicting churn using a feature like "days since last login" computed *after* churn has already happened.
- Predicting fraud using a "fraud_flag" set by a manual review.
- Predicting hospital readmission using `discharge_summary` (often filled out post-readmission).

**Train-test contamination**: same record (or near-duplicate) in both train and test.
- Random split when records have temporal structure (e.g., user-level data with timestamps).
- Augmentations spanning the split.
- Duplicate rows in the source data.

**Data preprocessing leakage**: stats computed on the full data before splitting.
- Standardization with overall mean/std (must use train stats only).
- Imputation with overall median.
- Feature selection on full data.

**Temporal leakage**: using future to predict past.
- Random shuffle of time-ordered data.
- Lagged features computed across the whole dataset.

**Group leakage**: same entity (user, patient, document) on both sides of split when the entity is what you care about generalizing across.

> **Saying it out loud.** Five shapes, and it's worth being able to name them all. Target leakage is a feature that's a consequence of the outcome rather than a cause — days since last login for churn is the classic. Contamination is duplicates or near-duplicates straddling the split. Preprocessing leakage is fitting your scaler or imputer before splitting, which feels harmless and isn't. Temporal leakage is training on the future. And group leakage is the sneaky one, where the same patient or user appears on both sides so the model learns to recognize the entity instead of the pattern. Medical ML is littered with published results that collapsed on that last one.

### How to defend
- Split by **time** for time-ordered tasks; never random.
- Split by **group** (user/patient) when generalization is across entities.
- Compute preprocessing stats on **train only**, apply to test.
- Audit features: "could this feature have been computed at prediction time?"
- Check for duplicates and near-duplicates.

A clean test set is the single most valuable artifact in an ML project. Treat it like gold.

> **Saying it out loud.** The defenses are structural rather than clever: split by time when there's time structure, split by group when you need to generalize across entities, and fit every transformation inside the training fold. Wrapping preprocessing in a pipeline that cross-validation refits per fold removes an entire class of bugs for free. Then audit — walk each feature and ask whether it could genuinely have been computed at prediction time. Treat the test set as a one-use artifact; the moment you start tuning against it, it stops being an estimate of anything.

---

## 2. Calibration — does P(y=1)=0.7 mean what it says?

A model is **calibrated** if among predictions with score 0.7, about 70% are positives. Many models that have high AUC are poorly calibrated (e.g., gradient-boosted trees, deep networks with cross-entropy).

> **Saying it out loud.** Calibration asks whether the numbers mean what they say — take everything scored around 0.7 and about 70 percent should actually be positive. It's a completely separate property from ranking, and the way to prove that is a thought experiment: halve every predicted probability and your AUC is unchanged while your calibration is destroyed. Modern deep networks are badly overconfident out of the box, with typical calibration error around 10 percentage points. Whether you care depends entirely on what happens downstream — if you're sorting a feed, ignore it; if you're comparing a probability against a cost threshold, it's everything.

### Why calibration matters
- **Decision making**: thresholds depend on probabilities matching reality.
- **Risk assessment**: "model says 1% chance" had better mean ~1%.
- **Combining predictions**: bad calibration breaks averaging/ensembling.
- **Cost-sensitive learning**: cost depends on probabilities.

> **Saying it out loud.** It matters wherever the number itself enters a decision, not just the ordering. Any threshold chosen by expected cost needs real probabilities — deciding whether to block a transaction means comparing the probability of fraud times the loss against the cost of a false decline, and if your probabilities are inflated you'll set that threshold wrong. Same for ensembling: averaging miscalibrated scores from different models produces something meaningless. And in risk settings, telling a clinician one percent had better actually mean one percent.

### Measuring calibration

**Reliability diagram**: bin predictions, plot empirical positive rate per bin vs predicted score. Calibrated → diagonal.

**Expected Calibration Error (ECE)**:

$$
\mathrm{ECE} = \sum_b \frac{|B_b|}{N} |\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|
$$

Average gap between accuracy and confidence per bin.

**Brier score**: $\frac{1}{N} \sum_i (\hat{p}_i - y_i)^2$. Lower is better. Measures both calibration and resolution.

> **Saying it out loud.** The reliability diagram is the picture: bin your predictions, plot average predicted probability against actual positive rate, and look at the distance from the diagonal. Below the line means overconfident, which is where nearly all neural networks sit. Expected calibration error compresses that into one number, a bin-weighted average gap, and its weakness is worth stating — it depends on how you bin, and coarse bins hide a lot. Brier score is the other summary, mean squared error on probabilities, and it's a proper scoring rule so you can't game it by shading predictions toward the extremes.

### Calibration techniques

**Platt scaling**: fit a logistic regression on the model's logits using a held-out set. Maps $z \to \sigma(a z + b)$.

**Isotonic regression**: non-parametric monotonic mapping. More flexible than Platt; needs more data.

**Temperature scaling**: divide logits by a learned scalar $T$. Standard for calibrating deep networks (Guo et al., 2017). Doesn't change ranking → AUC unchanged, but ECE improves.

**Modern LLMs are often miscalibrated** in confidence — overconfident on what they hallucinate, underconfident in many cases. This is an active research area.

> **Saying it out loud.** Three options, distinguished mainly by how many parameters they spend. Temperature scaling is one scalar dividing the logits, so it can't overfit and it fixes the specific uniform overconfidence deep networks exhibit — Guo and colleagues showed it drops typical ECE from around 0.1 to about 0.01. Platt scaling fits a two-parameter logistic on the scores, fine for small validation sets. Isotonic fits any monotonic map, more flexible but it wants a thousand-plus samples or it degenerates into a step function. All three preserve ranking, so your AUC and any rank-based threshold survive untouched — calibration is genuinely a free add-on.

---

## 3. Distribution shift — when train ≠ deploy

Real systems face data that differs from training data. Three flavors:

| Type | What changes | Example |
|---|---|---|
| **Covariate shift** | $p(x)$ changes, $p(y\|x)$ same | New user demographic |
| **Label shift** | $p(y)$ changes, $p(x\|y)$ same | Disease prevalence shifts |
| **Concept drift** | $p(y\|x)$ changes | User preferences evolve |

> **Saying it out loud.** Three flavors, and the reason to keep them straight is that the fixes differ. Covariate shift is the inputs moving while the relationship holds — new demographics — and you can correct that by reweighting. Label shift is the base rate moving while the class-conditionals hold, like disease prevalence during an outbreak, and that has a clean analytic correction from the prior ratio, no retraining needed. Concept drift is the relationship itself changing, which is the hard one because no reweighting saves you and you need fresh labels. So diagnosing which you have determines whether you're applying a patch or rebuilding a dataset.

### Detecting shift
- Monitor input distributions: KS test, KL divergence, PSI (Population Stability Index).
- Monitor model output distributions.
- Monitor prediction-label gap if labels eventually arrive.
- For black-box detection: train a classifier to distinguish train vs production data; if AUC > 0.5+, there's shift.

> **Saying it out loud.** Monitor input distributions per feature with KS tests or PSI, but the trick worth knowing is the classifier one: label training data zero and recent production data one, train a model, and check its held-out AUC. Near 0.5 means indistinguishable and no shift; anything higher means shift, and the feature importances tell you exactly where. Per-feature tests miss shifts in the joint distribution entirely and this catches them. Two cautions: with hundreds of features you'll drown in false alarms unless you correct thresholds, and input drift isn't performance drift — performance can collapse with no visible input change.

### Mitigation
- **Importance weighting** for covariate shift: $\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}[\frac{q(x)}{p(x)} f(x)]$. Reweight training samples by $q(x)/p(x)$ — but estimating the ratio is hard.
- **Domain adaptation / DANN**: adversarial training to make features domain-invariant.
- **Continual / online learning**: retrain periodically on fresh data.
- **Test-time adaptation**: adjust BN statistics, prompt, or last-layer at test time.

> **Saying it out loud.** Importance weighting is the principled fix for covariate shift, and in practice it's limited by variance rather than by theory — if a region is rare in training and common in production, the weights explode and your effective sample size collapses. Rule of thumb: weights spanning more than about a factor of ten mean you should be collecting data, not reweighting. Domain-adversarial training is heavier machinery for learning domain-invariant features. And for most production systems the honest answer is periodic retraining on fresh data, which is boring, reliable, and what almost everyone actually does.

### Concept drift in LLMs
Knowledge cutoffs, world events, evolving language. RAG can mitigate by separating "facts" (retrievable, updateable) from "skills" (parametric, frozen).

> **Saying it out loud.** For language models the drift is in the world, not the data pipeline — facts change after the knowledge cutoff, events happen, usage shifts. The architectural response is to split what changes from what doesn't: retrieval handles facts, which are updateable without touching the model, and the frozen parameters handle skills like reasoning and language, which age much more slowly. That framing is genuinely useful in a system design interview, because it turns 'the model is out of date' from a retraining problem into an indexing problem.

---

## 4. Class imbalance

When one class is much rarer than another (fraud, click, disease).

> **Saying it out loud.** The core issue is that with a 1 percent positive rate, a model that always says no scores 99 percent accuracy while being useless — say that number out loud and the point lands immediately. So the first fix is the metric: precision and recall at the threshold you'd actually deploy, plus AUPRC, whose baseline is the positive rate itself. The second fix is the threshold, because 0.5 is an arbitrary default and the right cut comes from the relative cost of the two error types. Everything else — resampling, focal loss, class weights — is secondary, and the honest current view is that class weighting plus a well-chosen threshold matches the fancier techniques most of the time.

### Wrong solutions
- **Just look at accuracy**: 99% accuracy by predicting "no fraud" always.
- **Random oversample / undersample without thought**: changes the test distribution; don't apply to test set.

> **Saying it out loud.** Two mistakes. Reporting accuracy, which rewards the do-nothing model. And resampling the test set, which is worse because it's invisible — you report 85 percent precision on a balanced evaluation set and see 8 percent in production, because the true prevalence is a hundredth of what you measured at. Rebalance training if it helps optimization; never touch evaluation. The test set exists to estimate performance on the deployment distribution, and the moment you rebalance it, it's estimating a world that doesn't exist.

### Right solutions
- **Use the right metric**: precision-recall curve, F1, AUPRC (not accuracy or even AUROC for very rare positives).
- **Class weights in the loss**: $\mathcal{L} = -\sum_i w_{y_i} \log p_i$.
- **Focal loss**: $-(1-p_t)^\gamma \log p_t$ — down-weights easy examples (Lin et al., 2017).
- **Stratified split**: keep class ratios similar in train/test.
- **Resample only the training set**, never test.
- **Threshold tuning**: don't use 0.5 by default; pick threshold from PR curve based on cost.
- **Calibrate after rebalancing**: rebalancing distorts probabilities.

> **Saying it out loud.** Pick a metric that notices the rare class, weight the loss so the minority contributes comparable gradient, tune the threshold from the precision-recall curve using real costs, and recalibrate afterwards because anything that rebalances distorts the probabilities. Focal loss is worth understanding as a special case: it came from dense object detection where a hundred thousand easy background boxes swamp a handful of real objects, so it down-weights confident predictions specifically. If your negatives aren't overwhelmingly easy, plain class weighting does the same job with less to explain.

### Sampling strategies
- **Oversampling**: SMOTE (synthetic minority), ADASYN. Risk: amplifies noise/outliers.
- **Undersampling**: random or informed (e.g., Tomek links). Risk: throws away information.
- **Hybrid**: SMOTE-Tomek, SMOTEENN.

In modern deep learning practice, often the simplest fix (class-weighted loss + careful metric choice) works as well as fancy resampling.

> **Saying it out loud.** Oversampling with SMOTE interpolates between minority neighbors to manufacture new positives, which sounds principled and quietly amplifies any mislabeled or outlying minority point, blurring the boundary. Undersampling throws away majority data, which is cheap and wasteful. The hybrids try to get both. The current consensus is deflationary — on modern classifiers, a class-weighted loss plus threshold tuning usually matches or beats these with far less risk, and the papers showing big SMOTE gains mostly predate that. Knowing the techniques and not being impressed by them is the right posture.

---

## 5. Bias-variance and the generalization gap

**Generalization gap** = train error − test error. Large gap → overfitting.

> **Saying it out loud.** The decomposition is a diagnostic tool more than a theory: high training error means bias and you need more capacity, low training error with high test error means variance and you need more data or regularization. Those have opposite fixes, which is why misdiagnosing costs you weeks. What's changed is that the classical U-shaped curve turned out to be only the left half of the picture — past the interpolation threshold, test error comes down again, which is why enormously over-parameterized models generalize at all. That second descent is a big part of why scaling kept working when the textbook said it shouldn't.

### The classical view (bias-variance tradeoff)

$$
\mathbb{E}[(\hat{f}(x) - y)^2] = \mathrm{Bias}(\hat{f}(x))^2 + \mathrm{Var}(\hat{f}(x)) + \sigma^2
$$

Underfit (high bias) → low capacity. Overfit (high variance) → too much capacity. The "U-shaped" test error.

> **Saying it out loud.** Three terms: how wrong your model class is on average, how much your fit wobbles with different training samples, and noise you can never beat. The practical value is diagnostic rather than quantitative — you almost never compute these, you use them to decide which lever to pull. And the third term is a useful humility check, because if irreducible noise puts a ceiling at 92 percent, chasing 95 is chasing label noise.

### The modern view (double descent)

For overparameterized neural networks:
- Test error is U-shaped up to interpolation (param count = data count).
- Past interpolation, test error *decreases again* — "double descent" (Belkin et al., 2019).

Modern deep learning operates in the second descent regime, where bigger models generalize better. This contradicts classical wisdom and is part of what makes scaling laws work.

> **Saying it out loud.** Right at the interpolation threshold, where the model has just barely enough capacity to fit the training data exactly, there's essentially one way to do it and the model is forced into whatever contorted solution that is — that's the error spike. Add more capacity and there are many interpolating solutions, and gradient descent's implicit bias picks a smooth, small-norm one rather than a random one. So over-parameterization gives the optimizer room to choose well. The framing worth offering is that parameter count was never the right complexity measure; what matters is the effective complexity of the solution you actually land on.

### Implicit regularization
SGD has implicit regularization properties — it tends to find flat, generalizing minima. Adam less so (and may not generalize as well as SGD on some tasks).

> **Saying it out loud.** Among all the solutions that fit the training data equally well, SGD systematically prefers particular ones — flat basins over sharp ones, because gradient noise makes narrow minima unstable. Nobody wrote that into the loss; it falls out of the algorithm. It also explains a practical folk observation: Adam's implicit regularization is weaker, which is why vision models often generalize better with SGD plus momentum despite Adam converging faster. SAM is what you get when you stop hoping for the effect and optimize for it explicitly.

---

## 6. Cross-validation done right

**k-fold CV**: split into $k$ folds; for each fold, train on $k-1$, test on 1; average.

> **Saying it out loud.** Cross-validation is just reusing your data to get a lower-variance estimate, and everything that goes wrong with it comes from the folds not being independent in the way you assumed. Time-ordered data needs a time-based split, entity-clustered data needs group folds, imbalanced classification needs stratification. And every preprocessing step has to be refit inside each fold, which is the single largest source of quiet leakage. The other thing people skip is reporting the standard deviation across folds — the mean alone hides whether your estimate is stable or your folds disagree wildly.

### Variants
- **Stratified k-fold**: preserve class ratios. Default for classification.
- **Group k-fold**: keep groups (users, patients) entirely on one side.
- **Time-series split**: sliding or expanding window. Never use random k-fold for time series.
- **Nested CV**: outer loop for evaluation, inner loop for hyperparameters. Avoids contaminating the outer estimate with hyperparam tuning.

> **Saying it out loud.** Choose based on what independence assumption your data violates. Stratified is the classification default and costs nothing. Group k-fold when rows cluster under users or patients, so the model can't recognize the entity rather than the pattern. Time-series split when there's temporal order — always train on the past. Nested CV when you're both tuning and evaluating, so tuning can't leak into the estimate; it's 25 fits for a 5-by-5 and nobody does it in deep learning, where a single untouched holdout is the pragmatic stand-in.

### Common errors
- Hyperparameter tuning on the test set.
- Using k-fold CV on time-series data.
- Forgetting to refit preprocessing per fold (huge source of leakage).
- Not stratifying when classes are imbalanced.
- Computing fold-level metric and reporting only the mean (also report std).

> **Saying it out loud.** The one that costs the most is refitting preprocessing outside the fold, because it's invisible and it inflates every number you report. Tuning on the test set is the same problem in a more obvious costume — try a hundred configurations and the winner is partly lucky, so the reported score is biased upward and the effect grows with how many things you tried. Random k-fold on time series is the one that produces the most spectacular failures, since a model that gets to see the future looks brilliant and is worthless. And report the standard deviation, because a mean of 0.87 across folds means something different when the folds range 0.85 to 0.89 versus 0.70 to 0.99.

---

## 7. Ablations — proving an idea actually contributes

If your paper says "we added X and it improved performance," you need an ablation: a controlled experiment removing X to show the improvement is due to X.

> **Saying it out loud.** An ablation is how you show that the thing you added is the thing that helped. Change one component, hold everything else fixed, run several seeds, and match the compute budget — that last one is the step people skip, and it matters, because if your new component adds 20 percent more parameters you've conflated the idea with the extra capacity. Without ablations a result tells you a system works and nothing about why, so nobody can build on it — including you, six months later. The field has repeatedly found that headline gains came from the learning rate schedule rather than the proposed architecture.

### Good ablation design
- Hold everything else fixed.
- Vary one component at a time.
- Run multiple seeds (3+); report mean ± std.
- Match compute budget across conditions.
- Use multiple evaluation tasks if claiming generality.

> **Saying it out loud.** Vary one thing, keep everything else identical, three or more seeds with mean and spread reported, matched FLOPs, and more than one evaluation if you're claiming generality. The seeds are non-negotiable because on many benchmarks seed-to-seed variation exceeds the improvements people publish. And the hyperparameters have to be tuned equally for both conditions — comparing your carefully tuned new method against a baseline you tuned for an afternoon is the most common way honest people produce dishonest results.

### Common ablation pitfalls
- Comparing improvements that are within noise (no significance test).
- Different hyperparameters for different ablation conditions.
- Single-seed runs.
- Reporting only the best of $K$ runs ("cherry picking").
- Not keeping training cost matched (extra layers cost more compute, fair comparison should match flops).

> **Saying it out loud.** Single seeds, unmatched compute, unequal tuning effort, and reporting the best of $K$ runs rather than the mean. That last one is quietly devastating: with enough restarts, the maximum of a noisy distribution looks like a real improvement and reproduces for nobody. The general test I'd offer is whether the comparison would survive if someone else ran it with different random seeds and no knowledge of which condition was supposed to win. If you can't answer yes, you have a number, not a result.

### What "actually works" means
A good ablation answers: "if I drop this component, does performance drop *consistently across seeds and across evaluations*?" If yes, the component contributes. If only on one seed and one eval, it's noise.

> **Saying it out loud.** The bar is consistency: if you remove the component, does performance drop across seeds and across evaluations, not just in the one run you're excited about? If it only shows up on a single seed and a single benchmark, it's noise wearing a costume. Effect size matters too — an improvement that's real but a tenth of a point may not justify the complexity it adds to a system that has to be maintained. Being willing to say a result is real and not worth shipping is a senior signal.

---

## 8. Metric uncertainty — getting CIs right

Reporting "model X has accuracy 87.3%" without uncertainty is sloppy. Always report a CI.

> **Saying it out loud.** Reporting a metric without an interval is reporting one draw from a distribution as if it were a fact. The number worth carrying around is that on a 1,000-example test set, accuracy carries roughly a two-point margin either side — so anything within four points is not a distinguishable difference, and most leaderboard gaps at that scale are noise. Bootstrapping gives you the interval for any metric you like without needing a formula. Once you internalize this, half the model comparisons you've seen stop being convincing, which is the point.

### How to compute it
- **Wald** for proportions (large $n$): $\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$.
- **Wilson** for proportions (any $n$, more accurate near 0/1): use the closed-form Wilson interval.
- **Bootstrap** for any metric (especially AUC, F1): resample the test set 1000+ times, compute metric each time, take quantiles.

### Comparing models
- Don't compare two CIs visually — overlapping CIs doesn't mean no difference.
- Use **paired bootstrap** of metric *differences*. CI on the difference.
- Or use a paired test: McNemar's for binary classification, DeLong's for AUC.

> **Saying it out loud.** The mistake to avoid is eyeballing two intervals and calling overlap 'no difference' — that ignores the fact that both models are being evaluated on the same examples, so their errors are correlated. Build the interval on the difference instead, using a paired bootstrap where each resample scores both models on the identical sample. That interval is much narrower and routinely detects differences of a few tenths of a point that independent intervals would dismiss. McNemar's test does the same thing analytically for binary classifiers, and DeLong's for AUC.

### Decision rule
- Difference is "significant" if its CI excludes 0.
- Effect size matters too: a tiny but significant improvement might not be worth deploying.

> **Saying it out loud.** Zero outside the interval on the difference means the gap is real. But the second half is what separates a good answer from a rote one: real is not the same as worth it. A statistically solid two-tenths of a point may not justify the extra latency, the extra memory, or the extra thing that can break in production. The right final question is always whether the improvement clears the cost of carrying it, not just whether it clears zero.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| 99% accuracy on fraud detection — done? | Yes | No — that's the base rate; use precision/recall |
| Random k-fold for time series? | Sure | Never — temporal leakage |
| Train test split before or after preprocessing? | Whichever | Before — preprocessing on full data leaks |
| Why is AUC bad for very imbalanced? | It isn't | AUC counts negatives uniformly; prefer AUPRC |
| Calibration vs accuracy? | Same | Different — calibrated = probs match reality; accurate = right side of 0.5 |
| What does double descent mean? | Doesn't | Test error has a second descent in over-parameterized regime |
| 95% CI overlap → no difference? | Yes | No — paired test of difference is correct |

---

## 10. Eight most-asked interview questions

1. **What is data leakage and how do you prevent it?** (Forms; preprocess in train only; group/temporal splits.)
2. **You have 99% accuracy but the system performs poorly. What's wrong?** (Class imbalance; report PR/F1/AUPRC; check calibration.)
3. **What's calibration and how do you fix it?** (Reliability diagram; ECE; Platt/isotonic/temperature scaling.)
4. **Three types of distribution shift?** (Covariate, label, concept; fixes for each.)
5. **Why use stratified k-fold?** (Preserves class ratios → reduces variance of CV estimate.)
6. **You have train accuracy 95%, val 87%. Overfitting — what do you check first?** (Capacity, regularization, data size, leakage in val, calibration.)
7. **Two models with overlapping AUC CIs. Significant?** (Not necessarily — paired bootstrap of differences.)
8. **You added X and CI overlaps with baseline. Is X a contribution?** (Probably not — show effect across seeds, multiple evals.)

---

## 11. Drill plan

- For each of the four leakage types, recite: definition, example, mitigation.
- For each calibration method (Platt, isotonic, temperature), recite: when to use, what data needed.
- Implement a stratified k-fold split + bootstrap AUC CI in 50 lines of Python.
- For each shift type (covariate/label/concept), recite definition + one detection method + one mitigation.
- Practice 5 "ablation review" mini-cases: someone shows a result; ask 3 sharp questions about whether the contribution is real.

---

## 12. Further reading

- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* (2022).
- Guo et al., *On Calibration of Modern Neural Networks* (2017) — temperature scaling.
- Belkin et al., *Reconciling modern machine-learning practice and the classical bias–variance trade-off* (2019) — double descent.
- Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation* (2010).
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (2015).
