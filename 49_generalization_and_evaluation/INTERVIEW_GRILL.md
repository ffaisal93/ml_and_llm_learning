# Generalization and Evaluation — Interview Grill

> 50 questions on data leakage, calibration, distribution shift, class imbalance, ablations. Drill until you can answer 35+ cold.

---

## A. Data leakage

**1. Define data leakage.**
Information from outside the training set leaking in, inflating offline metrics that don't transfer.

**2. Four types of leakage?**
Target leakage, train-test contamination, preprocessing leakage, temporal leakage. (Group leakage is a fifth special case.)

**3. Example of target leakage in churn prediction?**
Using "days since last login" as a feature when it's computed *after* churn. The future-looking feature trivially predicts churn.

**4. Why is random k-fold wrong for time series?**
Allows training on the future, predicting the past. Use time-series split (sliding/expanding window) instead.

**5. Standardize features using overall mean — leakage?**
Yes. Test data influences train preprocessing. Compute mean/std on train only.

**6. Same user in train and test — leakage?**
If predicting user-level outcomes and the model learns user-specific patterns: yes. Use group k-fold.

**7. How do you detect leakage post-hoc?**
Sky-high offline metrics that don't translate to deployment. Check feature importance for "too good to be true" features. Audit: could each feature have been computed at prediction time?

---

## B. Calibration

**8. What does calibrated mean?**
Predictions match observed frequencies: among predictions of 0.7, ~70% are positive.

**9. How do you measure calibration?**
Reliability diagram (bin predictions; plot empirical positive rate vs predicted score). ECE: average gap between confidence and accuracy across bins.

**10. ECE formula?**
$\mathrm{ECE} = \sum_b \frac{|B_b|}{N} |\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|$.

**11. Brier score?**
$\frac{1}{N}\sum (\hat{p} - y)^2$. Combined calibration + resolution.

**12. Three calibration techniques?**
Platt scaling (logistic), isotonic regression (non-parametric monotonic), temperature scaling (single scalar on logits).

**13. Why is temperature scaling popular for deep nets?**
Single scalar — minimal overfitting risk. Doesn't change ranking → AUC unchanged. Standard fix for overconfident neural networks.

**14. Does temperature scaling change AUC?**
No. Monotonic transform of scores doesn't change pairwise ordering.

**15. AUC vs calibration — different things?**
AUC measures ranking. Calibration measures whether scores are accurate probabilities. A model can have high AUC and bad calibration.

---

## C. Distribution shift

**16. Three types of distribution shift?**
Covariate shift ($p(x)$ changes), label shift ($p(y)$ changes), concept drift ($p(y|x)$ changes).

**17. Covariate shift — typical example?**
New user demographics. Input distribution shifts; the underlying relationship is the same.

**18. Label shift example?**
Disease prevalence increases during a pandemic; the conditional symptom-given-disease is unchanged.

**19. Concept drift example?**
User preferences evolving over time — same input features, but the label given those features changes ($p(y|x)$ shifts).

**20. How do you detect input drift in production?**
KS test, KL divergence, PSI between train and live distributions. Monitor input feature distributions per feature.

**21. What's PSI?**
Population Stability Index. Bin-based comparison of two distributions; PSI > 0.25 typically flagged as significant shift.

**22. Importance weighting for covariate shift?**
Reweight training samples by $q(x)/p(x)$. Hard to estimate ratio; can use density estimators or classifier-based estimates.

**23. Shift detection via classifier?**
Train a classifier to distinguish "is this from train or production?" If AUC > 0.5 + something, there's shift.

---

## D. Class imbalance

**24. Why is accuracy bad for imbalanced data?**
Predicting majority class always gives high accuracy (e.g., 99% if positive class is 1%).

**25. Right metrics for rare-class problems?**
Precision, recall, F1, AUPRC. AUROC is OK but can be misleading at extreme imbalance.

**26. AUPRC vs AUROC — when prefer AUPRC?**
Severely imbalanced data. AUPRC focuses on positive class behavior; AUROC averages across the full operating curve where the negative dominance dilutes.

**27. SMOTE — what does it do?**
Synthetic minority oversampling. Generates synthetic minority points by interpolating between minority neighbors. Risk: amplifies noise/outliers near class boundaries.

**28. Class weighting in the loss?**
Multiply per-sample loss by class-dependent weight. Standard PyTorch: `nn.CrossEntropyLoss(weight=class_weights)`.

**29. Focal loss formula?**
$-(1-p_t)^\gamma \log p_t$. Down-weights easy examples ($p_t$ near 1). $\gamma$ typically 2.

**30. Should you resample the test set?**
**No.** Resample only training set. Test set must reflect deployment distribution.

**31. After resampling, what about probabilities?**
They're distorted. Calibrate after, or apply a post-hoc shift to recover original ratios.

---

## E. Bias-variance and double descent

**32. Bias-variance decomposition?**
$\mathbb{E}[(\hat{f}(x) - y)^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$.

**33. High bias means?**
Underfitting. Model too simple. Both train and test error high.

**34. High variance means?**
Overfitting. Model too complex. Train error low, test error high.

**35. What's double descent?**
Past the interpolation threshold (params ≈ data points), test error decreases again as capacity increases. Belkin et al., 2019.

**36. Why does double descent happen?**
Modern over-parameterized models effectively select smoother interpolators. Implicit bias of optimization (SGD favors flat minima) plays a role.

**37. Implicit regularization of SGD?**
SGD tends to converge to flat minima (low Hessian eigenvalues), which generalize better empirically. Adam has weaker implicit regularization.

---

## F. Cross-validation

**38. What's nested CV?**
Outer loop: k-fold for evaluation. Inner loop: k-fold within each train fold for hyperparameter tuning. Prevents tuning from leaking into evaluation.

**39. Why do hyperparam tuning + evaluation on the same fold leak?**
You're choosing the model that does best on the eval set, biasing the eval estimate.

**40. Stratified k-fold — when?**
Classification with imbalanced classes. Preserves class ratio per fold; reduces estimator variance.

**41. Time-series CV strategy?**
Sliding window or expanding window. Always test on data later than train. Never random split.

**42. Group k-fold use case?**
When you want generalization across entities (users, patients, etc.). Each entity entirely in one fold.

---

## G. Ablations

**43. What's a good ablation?**
One component varied at a time, everything else fixed; multiple seeds; matched compute; multiple evals if claiming generality.

**44. You added X, performance improved by 0.3 points. Real?**
Need: multiple seeds, std reported, paired test or bootstrap of difference. 0.3 might be within seed noise.

**45. Why does ablation matter when papers report single numbers?**
Because single numbers without ablation can't establish *which component* drove the gain. Component might be a placebo.

---

## H. Metric uncertainty

**46. Wald CI for accuracy?**
$\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$.

**47. CI for AUC?**
Bootstrap (resample test set 1000+ times, compute AUC each time, quantile). Or DeLong's method.

**48. Two CIs overlap — does that mean no difference?**
No. Paired test on the difference is the right way. CIs of differences can exclude zero even when individual CIs overlap.

**49. Paired bootstrap procedure for model comparison?**
For each bootstrap sample, compute metric for both models on the same sample. Look at the distribution of differences. Reject "no difference" if 0 not in CI of differences.

**50. McNemar's test — when?**
Comparing two binary classifiers on the same test set. Tests if their disagreements are symmetric (same number of A-correct B-wrong vs A-wrong B-correct).

---

## Quick fire

**51.** *Best metric for fraud (rare positive)?* AUPRC + cost-sensitive threshold.
**52.** *Calibrate after rebalancing?* Yes, always.
**53.** *Split for time-series?* Time-based, never random.
**54.** *Default split for classification?* Stratified.
**55.** *Bootstrap iterations?* 1000+ typical.
**56.** *Temperature scaling change AUC?* No.
**57.** *Common shift detection metric?* PSI, KS test.
**58.** *Brier score lower = better?* Yes.
**59.** *ECE measures?* Calibration error.
**60.** *Model selection on test set?* Don't.

---

## Self-grading

If you can't answer 1-15, you'll get fooled by your own metrics. If you can't answer 16-35, you'll deploy broken systems. If you can't answer 36-50, frontier-lab evaluation rigor questions will go past you.

Aim for 40+/60 cold.
