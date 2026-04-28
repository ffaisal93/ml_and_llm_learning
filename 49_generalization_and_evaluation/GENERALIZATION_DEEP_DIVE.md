# Generalization and Evaluation — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

The single most common reason ML systems fail in production: the offline metric was wrong, the test set was contaminated, or the model didn't generalize the way you expected. This deep dive is about *not getting fooled by your own evaluation*.

---

## 1. Data leakage — what it is and how to spot it

**Leakage** = information from outside the training data leaks into the model, inflating offline metrics that don't transfer to deployment.

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

### How to defend
- Split by **time** for time-ordered tasks; never random.
- Split by **group** (user/patient) when generalization is across entities.
- Compute preprocessing stats on **train only**, apply to test.
- Audit features: "could this feature have been computed at prediction time?"
- Check for duplicates and near-duplicates.

A clean test set is the single most valuable artifact in an ML project. Treat it like gold.

---

## 2. Calibration — does P(y=1)=0.7 mean what it says?

A model is **calibrated** if among predictions with score 0.7, about 70% are positives. Many models that have high AUC are poorly calibrated (e.g., gradient-boosted trees, deep networks with cross-entropy).

### Why calibration matters
- **Decision making**: thresholds depend on probabilities matching reality.
- **Risk assessment**: "model says 1% chance" had better mean ~1%.
- **Combining predictions**: bad calibration breaks averaging/ensembling.
- **Cost-sensitive learning**: cost depends on probabilities.

### Measuring calibration

**Reliability diagram**: bin predictions, plot empirical positive rate per bin vs predicted score. Calibrated → diagonal.

**Expected Calibration Error (ECE)**:

$$
\mathrm{ECE} = \sum_b \frac{|B_b|}{N} |\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|
$$

Average gap between accuracy and confidence per bin.

**Brier score**: $\frac{1}{N} \sum_i (\hat{p}_i - y_i)^2$. Lower is better. Measures both calibration and resolution.

### Calibration techniques

**Platt scaling**: fit a logistic regression on the model's logits using a held-out set. Maps $z \to \sigma(a z + b)$.

**Isotonic regression**: non-parametric monotonic mapping. More flexible than Platt; needs more data.

**Temperature scaling**: divide logits by a learned scalar $T$. Standard for calibrating deep networks (Guo et al., 2017). Doesn't change ranking → AUC unchanged, but ECE improves.

**Modern LLMs are often miscalibrated** in confidence — overconfident on what they hallucinate, underconfident in many cases. This is an active research area.

---

## 3. Distribution shift — when train ≠ deploy

Real systems face data that differs from training data. Three flavors:

| Type | What changes | Example |
|---|---|---|
| **Covariate shift** | $p(x)$ changes, $p(y\|x)$ same | New user demographic |
| **Label shift** | $p(y)$ changes, $p(x\|y)$ same | Disease prevalence shifts |
| **Concept drift** | $p(y\|x)$ changes | User preferences evolve |

### Detecting shift
- Monitor input distributions: KS test, KL divergence, PSI (Population Stability Index).
- Monitor model output distributions.
- Monitor prediction-label gap if labels eventually arrive.
- For black-box detection: train a classifier to distinguish train vs production data; if AUC > 0.5+, there's shift.

### Mitigation
- **Importance weighting** for covariate shift: $\mathbb{E}_{x \sim q}[f(x)] = \mathbb{E}_{x \sim p}[\frac{q(x)}{p(x)} f(x)]$. Reweight training samples by $q(x)/p(x)$ — but estimating the ratio is hard.
- **Domain adaptation / DANN**: adversarial training to make features domain-invariant.
- **Continual / online learning**: retrain periodically on fresh data.
- **Test-time adaptation**: adjust BN statistics, prompt, or last-layer at test time.

### Concept drift in LLMs
Knowledge cutoffs, world events, evolving language. RAG can mitigate by separating "facts" (retrievable, updateable) from "skills" (parametric, frozen).

---

## 4. Class imbalance

When one class is much rarer than another (fraud, click, disease).

### Wrong solutions
- **Just look at accuracy**: 99% accuracy by predicting "no fraud" always.
- **Random oversample / undersample without thought**: changes the test distribution; don't apply to test set.

### Right solutions
- **Use the right metric**: precision-recall curve, F1, AUPRC (not accuracy or even AUROC for very rare positives).
- **Class weights in the loss**: $\mathcal{L} = -\sum_i w_{y_i} \log p_i$.
- **Focal loss**: $-(1-p_t)^\gamma \log p_t$ — down-weights easy examples (Lin et al., 2017).
- **Stratified split**: keep class ratios similar in train/test.
- **Resample only the training set**, never test.
- **Threshold tuning**: don't use 0.5 by default; pick threshold from PR curve based on cost.
- **Calibrate after rebalancing**: rebalancing distorts probabilities.

### Sampling strategies
- **Oversampling**: SMOTE (synthetic minority), ADASYN. Risk: amplifies noise/outliers.
- **Undersampling**: random or informed (e.g., Tomek links). Risk: throws away information.
- **Hybrid**: SMOTE-Tomek, SMOTEENN.

In modern deep learning practice, often the simplest fix (class-weighted loss + careful metric choice) works as well as fancy resampling.

---

## 5. Bias-variance and the generalization gap

**Generalization gap** = train error − test error. Large gap → overfitting.

### The classical view (bias-variance tradeoff)

$$
\mathbb{E}[(\hat{f}(x) - y)^2] = \mathrm{Bias}(\hat{f}(x))^2 + \mathrm{Var}(\hat{f}(x)) + \sigma^2
$$

Underfit (high bias) → low capacity. Overfit (high variance) → too much capacity. The "U-shaped" test error.

### The modern view (double descent)

For overparameterized neural networks:
- Test error is U-shaped up to interpolation (param count = data count).
- Past interpolation, test error *decreases again* — "double descent" (Belkin et al., 2019).

Modern deep learning operates in the second descent regime, where bigger models generalize better. This contradicts classical wisdom and is part of what makes scaling laws work.

### Implicit regularization
SGD has implicit regularization properties — it tends to find flat, generalizing minima. Adam less so (and may not generalize as well as SGD on some tasks).

---

## 6. Cross-validation done right

**k-fold CV**: split into $k$ folds; for each fold, train on $k-1$, test on 1; average.

### Variants
- **Stratified k-fold**: preserve class ratios. Default for classification.
- **Group k-fold**: keep groups (users, patients) entirely on one side.
- **Time-series split**: sliding or expanding window. Never use random k-fold for time series.
- **Nested CV**: outer loop for evaluation, inner loop for hyperparameters. Avoids contaminating the outer estimate with hyperparam tuning.

### Common errors
- Hyperparameter tuning on the test set.
- Using k-fold CV on time-series data.
- Forgetting to refit preprocessing per fold (huge source of leakage).
- Not stratifying when classes are imbalanced.
- Computing fold-level metric and reporting only the mean (also report std).

---

## 7. Ablations — proving an idea actually contributes

If your paper says "we added X and it improved performance," you need an ablation: a controlled experiment removing X to show the improvement is due to X.

### Good ablation design
- Hold everything else fixed.
- Vary one component at a time.
- Run multiple seeds (3+); report mean ± std.
- Match compute budget across conditions.
- Use multiple evaluation tasks if claiming generality.

### Common ablation pitfalls
- Comparing improvements that are within noise (no significance test).
- Different hyperparameters for different ablation conditions.
- Single-seed runs.
- Reporting only the best of $K$ runs ("cherry picking").
- Not keeping training cost matched (extra layers cost more compute, fair comparison should match flops).

### What "actually works" means
A good ablation answers: "if I drop this component, does performance drop *consistently across seeds and across evaluations*?" If yes, the component contributes. If only on one seed and one eval, it's noise.

---

## 8. Metric uncertainty — getting CIs right

Reporting "model X has accuracy 87.3%" without uncertainty is sloppy. Always report a CI.

### How to compute it
- **Wald** for proportions (large $n$): $\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$.
- **Wilson** for proportions (any $n$, more accurate near 0/1): use the closed-form Wilson interval.
- **Bootstrap** for any metric (especially AUC, F1): resample the test set 1000+ times, compute metric each time, take quantiles.

### Comparing models
- Don't compare two CIs visually — overlapping CIs doesn't mean no difference.
- Use **paired bootstrap** of metric *differences*. CI on the difference.
- Or use a paired test: McNemar's for binary classification, DeLong's for AUC.

### Decision rule
- Difference is "significant" if its CI excludes 0.
- Effect size matters too: a tiny but significant improvement might not be worth deploying.

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
