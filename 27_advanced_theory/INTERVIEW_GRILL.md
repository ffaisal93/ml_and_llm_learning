# Advanced ML Theory — Interview Grill

> 40 questions on bias-variance, cross-validation, learning curves, AIC/BIC, ROC/PR. Drill until you can answer 28+ cold.

---

## A. Bias-variance

**1. Bias-variance decomposition?**
$\mathbb{E}[(y - \hat{f}(x))^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$.

**2. Bias definition?**
$\mathbb{E}_D[\hat{f}_D(x)] - f^*(x)$. Average error from truth.

**3. Variance definition?**
$\mathbb{E}_D[(\hat{f}_D(x) - \mathbb{E}_D[\hat{f}_D(x)])^2]$. How much predictions vary across training sets.

**4. Irreducible noise $\sigma^2$?**
$\mathbb{E}[(y - f^*(x))^2]$. Cannot be reduced by any model.

**5. High-bias signature?**
Train and val errors both high. Train-val gap small.

**6. High-variance signature?**
Train error low, val error high. Train-val gap large.

**7. Modern over-parameterized regime?**
Double descent — bias-variance trade-off doesn't follow classical U-shape.

---

## B. Cross-validation

**8. k-fold CV procedure?**
Split into $k$ folds. For each: train on $k-1$, test on 1. Average errors.

**9. Standard $k$?**
5 or 10. Compromise between bias (low for higher $k$) and variance (high for $k = n$).

**10. LOO-CV — why high variance?**
Training sets are highly correlated → predictions correlated → empirical mean has high variance.

**11. Stratified k-fold?**
Preserves class ratio per fold. Default for imbalanced classification.

**12. Group k-fold?**
Each entity (user, patient) entirely in one fold. For generalization across entities.

**13. Time-series CV?**
Sliding or expanding window. Train on past, test on future. Never random.

**14. Nested CV?**
Outer for eval, inner for hyperparameter tuning. Prevents tuning leakage.

**15. Common CV pitfalls?**
Tuning + eval same fold; preprocessing on full data; not stratifying for imbalance; random split for time-series.

**16. LOO-CV closed form for linear regression?**
$\frac{1}{n}\sum (\frac{y_i - \hat{y}_i}{1 - h_{ii}})^2$ where $h_{ii}$ is hat-matrix diagonal. Avoids retraining.

---

## C. Learning curves

**17. What does train error converging to high value mean?**
High bias. Model too simple. More data won't help much.

**18. What does big train-val gap mean?**
High variance. Overfitting. More data will help.

**19. Decision: more data vs better model?**
Plot learning curves. Big gap → more data. High train error → better model.

**20. Validation curve vs learning curve?**
Validation curve: y vs hyperparameter. Learning curve: y vs training set size.

---

## D. Information criteria

**21. AIC formula?**
$\mathrm{AIC} = 2k - 2 \log L$. Lower better.

**22. BIC formula?**
$\mathrm{BIC} = k \log n - 2 \log L$. Lower better.

**23. AIC vs BIC penalty growth?**
BIC penalty $k \log n$ grows with $n$. AIC's $2k$ stays constant. BIC selects simpler models for large $n$.

**24. AIC purpose?**
Optimal for prediction. Doesn't assume true model in candidate set.

**25. BIC purpose?**
Consistent for true model identification (when true model in candidates).

**26. When does BIC penalty exceed AIC?**
$\log n > 2$ → $n > e^2 \approx 7.4$. Almost always.

**27. Limitations of AIC/BIC?**
Need well-defined likelihood; assume correct model specification; effective $k$ unclear for regularized models.

---

## E. ROC and PR

**28. ROC axes?**
TPR (recall) vs FPR (false alarm). Threshold-free.

**29. AUROC interpretation?**
Probability random positive ranks above random negative.

**30. PR curve axes?**
Precision vs Recall. Threshold-free.

**31. AUROC vs AUPRC for imbalance?**
AUPRC much more informative. AUROC dominated by easy negatives.

**32. Choosing operating point?**
Cost-weighted: $\arg\min_\tau (c_{\mathrm{FN}} \mathrm{FN} + c_{\mathrm{FP}} \mathrm{FP})$. Or fixed recall / FP rate.

**33. F1 formula?**
$F_1 = 2PR/(P+R)$. Harmonic mean.

**34. F-beta?**
$F_\beta = (1 + \beta^2) PR / (\beta^2 P + R)$. $\beta > 1$ weights recall more.

**35. Why harmonic mean for F1?**
Penalizes imbalance: F1 = 0 if either P or R = 0. Arithmetic mean wouldn't.

---

## F. Confusion matrix

**36. Precision formula?**
$TP/(TP + FP)$. Of positive predictions, how many right.

**37. Recall (sensitivity) formula?**
$TP/(TP + FN)$. Of actual positives, how many caught.

**38. Specificity formula?**
$TN/(TN + FP)$. Of actual negatives, how many correctly negative.

**39. MCC purpose?**
Balanced metric for imbalanced classification. Range $[-1, 1]$. 0 = random.

**40. Accuracy when imbalanced?**
Misleading. 99% by predicting majority class always. Use F1, AUPRC, MCC instead.

---

## Quick fire

**41.** *Bias-variance third term?* Irreducible noise.
**42.** *Standard k-fold?* 10.
**43.** *Time-series CV?* Walk-forward.
**44.** *AIC penalty?* $2k$.
**45.** *BIC penalty?* $k \log n$.
**46.** *F1 = ?* Harmonic mean of P, R.
**47.** *Top-left of ROC?* Perfect.
**48.** *Diagonal of ROC?* Random classifier.
**49.** *PR for imbalance?* Yes — better than ROC.
**50.** *LOO-CV variance?* High.

---

## Self-grading

If you can't answer 1-15, you don't know basic theory. If you can't answer 16-30, you'll struggle on practical evaluation. If you can't answer 31-40, frontier-lab questions on classical ML rigor will go past you.

Aim for 30+/50 cold.
