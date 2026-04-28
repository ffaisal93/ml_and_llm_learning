# Advanced ML Theory — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This is the "ML theory you should actually know cold" — bias-variance with proof, cross-validation theory, learning curves, model selection (AIC/BIC), and ROC analysis. Some of this overlaps with the SLT and generalization deep dives but here we focus on the *practical decisions* these theories inform.

---

## 1. Bias-variance — the proof

For a regression model $\hat{f}$ trained on a random dataset $D$, evaluating at a fixed test point $x$:

$$
\mathbb{E}_D[(y - \hat{f}_D(x))^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2
$$

where:
- $\mathrm{Bias} = \mathbb{E}_D[\hat{f}_D(x)] - f^*(x)$ — average error of the model from truth.
- $\mathrm{Var} = \mathbb{E}_D[(\hat{f}_D(x) - \mathbb{E}_D[\hat{f}_D(x)])^2]$ — variability across training sets.
- $\sigma^2 = \mathbb{E}[(y - f^*(x))^2]$ — irreducible noise.

### Derivation

Let $\bar{f}(x) = \mathbb{E}_D[\hat{f}_D(x)]$ (average prediction across training sets).

$$
\mathbb{E}_D[(y - \hat{f}_D(x))^2] = \mathbb{E}_D[(y - \bar{f}(x) + \bar{f}(x) - \hat{f}_D(x))^2]
$$

Expanding (cross-term vanishes by definition of $\bar{f}$):

$$
= \mathbb{E}_D[(y - \bar{f}(x))^2] + \mathbb{E}_D[(\bar{f}(x) - \hat{f}_D(x))^2]
$$

The first term is bias² + noise:

$$
\mathbb{E}[(y - \bar{f}(x))^2] = (\bar{f}(x) - f^*(x))^2 + \sigma^2
$$

The second term is variance.

### Implications
- Underfit: high bias (model too simple), low variance.
- Overfit: low bias, high variance.
- Tradeoff: total error minimized at intermediate capacity.
- Modern over-parameterized regime: double descent (see SLT deep dive). Classical view doesn't apply.

---

## 2. Cross-validation

### k-fold CV

Split data into $k$ folds. For each fold: train on $k-1$, test on 1. Average the test errors.

$$
\mathrm{CV}_k = \frac{1}{k} \sum_{i=1}^k L(\hat{f}^{(-i)}, D_i)
$$

where $\hat{f}^{(-i)}$ is the model trained without fold $i$, $D_i$ is fold $i$.

### Why $k$ matters
- $k = 2$: high **bias** (each fold trains on only half the data → underestimates large-$n$ performance); low variance (folds barely overlap, estimates are nearly independent).
- $k = n$ (LOO): low bias (uses $n-1$ samples, almost all data) but **high variance** (training sets differ in only one example → estimates highly correlated).
- $k = 5$ or $10$: standard compromise between the two.

### Variants
- **Stratified k-fold**: preserve class ratios. Default for classification.
- **Group k-fold**: keep groups (users, patients) entirely on one side.
- **Time-series split**: sliding or expanding window. Never random for time series.
- **Repeated k-fold**: run k-fold multiple times with different seeds; average.
- **Nested CV**: outer for evaluation, inner for hyperparameter tuning. Avoids contamination.

### Common pitfalls
- Hyperparameter tuning + final evaluation on same fold → optimistic bias.
- Preprocessing on full data before splitting → leakage.
- Not stratifying for imbalanced classes → high CV variance.
- Random split for time-series → temporal leakage.

### LOO-CV closed forms

For linear regression:

$$
\mathrm{CV}_{\mathrm{LOO}} = \frac{1}{n} \sum_i \left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2
$$

where $h_{ii}$ is the $i$-th diagonal of the hat matrix $H = X(X^\top X)^{-1} X^\top$. Computed without retraining $n$ times.

---

## 3. Learning curves

Plot training error and validation error vs training set size $n$.

### What they tell you

**High bias (underfitting)**:
- Train error high.
- Validation error converges to train error from above.
- Gap small.
- More data won't help — model is fundamentally too simple.

**High variance (overfitting)**:
- Train error low.
- Validation error high.
- Big gap.
- More data will help (gap closes as $n$ grows).

### Decision-making
- See big gap? → more data, regularize, or simpler model.
- See high training error? → bigger model, better features, less regularization.

### Practical use
Always plot learning curves before deciding "we need more data" vs "we need a better model." Often answers it definitively.

---

## 4. Validation curves

Plot training error and validation error vs a hyperparameter (e.g., model capacity, regularization strength).

Reveals the bias-variance trade-off across hyperparameter values.

**Sweet spot**: minimum of validation error. Train error keeps improving past this; validation error rises again — overfitting.

---

## 5. Information criteria for model selection

When you can compute model likelihood, criteria let you compare models without held-out data.

### AIC (Akaike Information Criterion)

$$
\mathrm{AIC} = 2k - 2\log L
$$

where $k$ = number of parameters, $L$ = max likelihood. Lower is better.

**Derivation**: estimates the expected KL divergence between the fitted model and the true distribution. Penalty $2k$ adjusts for using the data twice (training + evaluation).

### BIC (Bayesian Information Criterion)

$$
\mathrm{BIC} = k \log n - 2\log L
$$

with $n$ = number of observations. Lower is better.

**Derivation**: large-sample approximation of the log marginal likelihood (Bayesian model evidence). Penalty $k \log n$ grows with $n$.

### AIC vs BIC
- BIC penalty grows with $n$ → BIC selects simpler models for large $n$.
- AIC: optimal for *prediction*; doesn't assume true model in candidate set.
- BIC: consistent for *true model selection* if true model is in candidate set.
- BIC > AIC penalty for $n > e^2 \approx 7.4$.

### Limitations
- Both require evaluating likelihood — only meaningful when likelihood is well-defined.
- Don't directly apply to regularized models (effective $k$ unclear).
- Assume model is correctly specified.

---

## 6. ROC and PR curves

### ROC curve
Plot True Positive Rate (TPR) vs False Positive Rate (FPR) as threshold varies.

- TPR = TP / (TP + FN) — sensitivity / recall.
- FPR = FP / (FP + TN) — fall-out.
- Top-left corner = perfect classifier.
- Diagonal = random classifier.

**AUROC** = area under ROC. Probability that a random positive ranks above a random negative.

### PR curve
Plot Precision vs Recall as threshold varies.
- Better for imbalanced (where most negatives are easy).
- AUPRC: more informative than AUROC for severe imbalance.

### Choosing operating point
- Cost-aware: $\arg\min_\tau (c_{\mathrm{FN}} \cdot \mathrm{FN}(\tau) + c_{\mathrm{FP}} \cdot \mathrm{FP}(\tau))$.
- Recall constraint: pick $\tau$ such that recall ≥ X.
- F-score optimization: $\tau^* = \arg\max F_\beta$.

### F-beta score

$$
F_\beta = (1 + \beta^2) \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
$$

$\beta = 1$: F1. $\beta > 1$: weight recall more (e.g., disease screening). $\beta < 1$: weight precision more (e.g., spam).

---

## 7. Confusion matrix and derived metrics

| | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

- **Accuracy**: $(TP + TN) / N$.
- **Precision**: $TP / (TP + FP)$ — what fraction of positive predictions were right.
- **Recall (sensitivity, TPR)**: $TP / (TP + FN)$ — what fraction of actual positives were found.
- **Specificity (TNR)**: $TN / (TN + FP)$.
- **F1**: harmonic mean of P and R.
- **MCC** (Matthews Correlation Coefficient): balanced metric for imbalanced.

### Why F1 not arithmetic mean?
Harmonic mean penalizes imbalance more — F1 = 0.5 only when both P and R = 0.5. F1 = 0 if either is 0.

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Bias-variance — what's the third term? | "Just bias and variance" | Irreducible noise $\sigma^2$ |
| Why is LOO-CV high variance? | Lots of data | Training sets are highly correlated → predictions are correlated → empirical mean has high variance |
| Why does k=10 work well? | Tradition | Empirical compromise: most data used, manageable variance |
| AIC vs BIC — same purpose? | Yes | AIC for prediction, BIC for model selection (true model in candidates) |
| AUROC vs AUPRC for imbalance? | Same | AUPRC much more informative; AUROC dominated by easy negatives |
| Time-series with k-fold? | Sure | Never — temporal leakage |
| F1 = arithmetic mean of P and R? | Yes | Harmonic mean — penalizes imbalance |

---

## 9. Eight most-asked interview questions

1. **Derive the bias-variance decomposition.** (Add and subtract $\bar{f}(x)$; expand; cross-term zero.)
2. **What's the main purpose of cross-validation?** (Estimate generalization without leaking test data.)
3. **What does a learning curve tell you?** (High bias vs high variance via train-val gap; informs "more data" vs "better model".)
4. **AIC vs BIC?** (Both penalize complexity; BIC penalty $k \log n$ grows; AIC for prediction, BIC for true-model identification.)
5. **What's wrong with AUROC for severe imbalance?** (Negatives dominate; many easy positives lift AUROC; AUPRC focuses on positives.)
6. **F1 vs accuracy?** (Accuracy misleading for imbalance; F1 is harmonic mean of P and R.)
7. **Why use stratified k-fold?** (Preserve class ratios; reduces CV variance.)
8. **What's nested CV?** (Outer for evaluation; inner for hyperparameter tuning. Prevents tuning bias in outer estimate.)

---

## 10. Drill plan

- Derive bias-variance decomposition on paper.
- For each CV variant (k-fold, stratified, group, time-series, nested), recite when used.
- Recite AIC and BIC formulas + when each.
- Sketch ROC and PR curves for: random, perfect, threshold-based binary classifier.
- For each F-score variant ($F_1, F_{0.5}, F_2$), recite when used.
- Plot a learning curve for "high bias" vs "high variance" — describe to interviewer.

---

## 11. Further reading

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* — chapters 7 (model assessment), 8 (model inference).
- Bishop, *Pattern Recognition and Machine Learning* — chapter 1 (bias-variance).
- Kohavi (1995), *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.*
- Saito & Rehmsmeier (2015), *The Precision-Recall Plot is More Informative than the ROC Plot...*
- Burnham & Anderson, *Model Selection and Multi-Model Inference* — AIC/BIC reference.
