# Advanced ML Theory — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

This is the "ML theory you should actually know cold" — bias-variance with proof, cross-validation theory, learning curves, model selection (AIC/BIC), and ROC analysis. Some of this overlaps with the SLT and generalization deep dives but here we focus on the *practical decisions* these theories inform.

---

## 1. Bias-variance — the proof

> **In plain language.** This section splits your prediction error into three separate causes. Imagine retraining your model on many different datasets drawn from the same source. Bias is how far the average of all those models sits from the truth, variance is how much they disagree with each other, and noise is the randomness in the target that no model could ever predict. The algebra below just makes that split exact.

For a regression model $\hat{f}$ trained on a random dataset $D$, evaluating at a fixed test point $x$:

$$
\mathbb{E}_D[(y - \hat{f}_D(x))^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2
$$

where:
- $\mathrm{Bias} = \mathbb{E}_D[\hat{f}_D(x)] - f^*(x)$ — average error of the model from truth.
- $\mathrm{Var} = \mathbb{E}_D[(\hat{f}_D(x) - \mathbb{E}_D[\hat{f}_D(x)])^2]$ — variability across training sets.
- $\sigma^2 = \mathbb{E}[(y - f^*(x))^2]$ — irreducible noise.

> **Saying it out loud.** Expected squared error decomposes into exactly three pieces, and every error you ever see is some mix of them. Bias is how wrong you'd be on average even after retraining on infinitely many datasets, which is the cost of your model class being too rigid. Variance is how much your predictions wobble when the training data changes. And noise is the part of the target that isn't a function of your features at all. It's diagnostic rather than decorative: high bias means get a better model, high variance means get more data or regularize, and high noise means stop, you're at the floor.

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

> **Saying it out loud.** The derivation is one trick: add and subtract the average prediction across datasets, then expand the square. The cross-term vanishes because the average prediction is, by definition, the mean, so the deviation from it has expectation zero. What's left is a term measuring how far that average sits from the truth, which further splits into bias squared plus noise, and a term measuring spread around the average, which is variance. It's three lines on a whiteboard and it's the most commonly asked derivation in this whole area, so practice the add-and-subtract step until it's automatic.

### Implications
- Underfit: high bias (model too simple), low variance.
- Overfit: low bias, high variance.
- Tradeoff: total error minimized at intermediate capacity.
- Modern over-parameterized regime: double descent (see SLT deep dive). Classical view doesn't apply.

> **Saying it out loud.** The practical reading is a dial. Turn model capacity down and you get high bias, low variance, underfitting. Turn it up and you get low bias, high variance, overfitting. Total error is minimized somewhere in the middle, which is the classical U-shaped curve. The big caveat is that modern over-parameterized networks break this picture: past the interpolation threshold, test error descends a second time, which is double descent, and it's why a model with far more parameters than data points can still generalize.

---

## 2. Cross-validation

### k-fold CV

Split data into $k$ folds. For each fold: train on $k-1$, test on 1. Average the test errors.

$$
\mathrm{CV}_k = \frac{1}{k} \sum_{i=1}^k L(\hat{f}^{(-i)}, D_i)
$$

where $\hat{f}^{(-i)}$ is the model trained without fold $i$, $D_i$ is fold $i$.

> **Saying it out loud.** k-fold cross-validation splits the data into $k$ chunks and, $k$ times over, trains on all but one and evaluates on the one held out, then averages. Every point gets used for validation exactly once, which is why it's so much more stable than a single random holdout on small data. The spread across folds also gives you a rough error bar. The cost is $k$ full training runs, which is exactly why nobody cross-validates a large language model.

### Why $k$ matters
- $k = 2$: high **bias** (each fold trains on only half the data → underestimates large-$n$ performance); low variance (folds barely overlap, estimates are nearly independent).
- $k = n$ (LOO): low bias (uses $n-1$ samples, almost all data) but **high variance** (training sets differ in only one example → estimates highly correlated).
- $k = 5$ or $10$: standard compromise between the two.

> **Saying it out loud.** Choosing $k$ is itself a bias-variance tradeoff, just about the estimator rather than the model. At $k$ equals two, each model trains on half the data, so you systematically underestimate how good the full-data model will be; that's pessimistic bias. At $k$ equals $n$, leave-one-out, each model sees almost everything so bias is tiny, but the training sets differ by a single point, so the $n$ estimates are nearly perfectly correlated and averaging them barely reduces noise. Five or ten is the empirical compromise, and ten is the usual default.

### Variants
- **Stratified k-fold**: preserve class ratios. Default for classification.
- **Group k-fold**: keep groups (users, patients) entirely on one side.
- **Time-series split**: sliding or expanding window. Never random for time series.
- **Repeated k-fold**: run k-fold multiple times with different seeds; average.
- **Nested CV**: outer for evaluation, inner for hyperparameter tuning. Avoids contamination.

> **Saying it out loud.** The variants each fix a specific way the naive split lies to you. Stratified keeps class ratios intact so an imbalanced fold doesn't produce a meaningless score. Group keeps all rows of a user or patient on one side, so you're measuring generalization to new people rather than memorization. Time-series splits forward in time so you never train on the future. Repeated k-fold averages several random splits to reduce the noise in the estimate. And nested CV separates tuning from evaluation. The question that picks the right one is always: what am I claiming this model will generalize to?

### Common pitfalls
- Hyperparameter tuning + final evaluation on same fold → optimistic bias.
- Preprocessing on full data before splitting → leakage.
- Not stratifying for imbalanced classes → high CV variance.
- Random split for time-series → temporal leakage.

> **Saying it out loud.** Four pitfalls and they all come from the same root, which is letting information from the validation data reach the training process. Tuning and evaluating on the same folds inflates your number because you selected on that data. Fitting a scaler or imputer or feature selector before splitting leaks test statistics into training. Not stratifying makes imbalanced folds meaningless. Random splitting on time series trains on the future. The universal fix is that anything which learns from data must be fit inside the fold, which is precisely what pipelines are for.

### LOO-CV closed forms

For linear regression:

$$
\mathrm{CV}_{\mathrm{LOO}} = \frac{1}{n} \sum_i \left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2
$$

where $h_{ii}$ is the $i$-th diagonal of the hat matrix $H = X(X^\top X)^{-1} X^\top$. Computed without retraining $n$ times.

> **Saying it out loud.** For linear regression you get leave-one-out for free. The leave-one-out residual is just the ordinary residual divided by one minus the corresponding diagonal of the hat matrix, so one fit gives you exact leave-one-out error with no retraining. The interpretation of $h_{ii}$ is leverage: how much a point pulls its own fitted value toward itself. High-leverage points get their residuals inflated most, which is exactly right since those are the points whose removal changes the fit the most. Generalized cross-validation extends the same shortcut to ridge and smoothers.

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

> **Saying it out loud.** A learning curve puts both training and validation error on the y-axis against training set size, and its shape tells you which problem you have. Two curves that converge early and both sit high means bias: the model can't fit even what it has, and more data changes nothing. Two curves with a large persistent gap means variance: the model memorized, and more data will close the gap. That's the whole diagnostic and it takes one plot.

### Decision-making
- See big gap? → more data, regularize, or simpler model.
- See high training error? → bigger model, better features, less regularization.

> **Saying it out loud.** The decision rule falls straight out of the shape. Big gap means buy more data, add regularization, or simplify the model. High training error means you need a bigger model, better features, or less regularization, and more data would be wasted money. The thing to check is whether the gap is still narrowing at the right edge of the plot, because a curve that's already flat tells you additional data has stopped paying.

### Practical use
Always plot learning curves before deciding "we need more data" vs "we need a better model." Often answers it definitively.

> **Saying it out loud.** The practical advice is to plot the curve before anyone argues. The question of whether we need more data or a better model comes up constantly, gets debated on intuition, and is often settled definitively by twenty minutes of retraining on subsets. Proposing that in an interview signals that you'd rather measure than opine. And it's cheap: a handful of runs at ten, twenty-five, fifty, and a hundred percent of the data usually settles it.

---

## 4. Validation curves

Plot training error and validation error vs a hyperparameter (e.g., model capacity, regularization strength).

Reveals the bias-variance trade-off across hyperparameter values.

**Sweet spot**: minimum of validation error. Train error keeps improving past this; validation error rises again — overfitting.

> **Saying it out loud.** A validation curve varies a hyperparameter instead of dataset size, so it shows you the bias-variance tradeoff directly as a function of capacity. Training error keeps falling as capacity rises; validation error falls, bottoms out, then climbs again, and that minimum is your sweet spot. The distinction from a learning curve is worth stating explicitly because people mix them up: learning curve for how much data, validation curve for how much model.

---

## 5. Information criteria for model selection

When you can compute model likelihood, criteria let you compare models without held-out data.

### AIC (Akaike Information Criterion)

$$
\mathrm{AIC} = 2k - 2\log L
$$

where $k$ = number of parameters, $L$ = max likelihood. Lower is better.

**Derivation**: estimates the expected KL divergence between the fitted model and the true distribution. Penalty $2k$ adjusts for using the data twice (training + evaluation).

> **Saying it out loud.** AIC is twice the parameter count minus twice the log-likelihood, and lower wins. The intuition is that each extra parameter has to buy you at least one unit of log-likelihood to be worth including, otherwise you're just fitting noise. It comes from an estimate of the expected KL divergence between your fitted model and the truth, so the penalty is correcting for the fact that you used the same data to fit and to score. Only differences in AIC between models mean anything, since the absolute value carries an arbitrary constant.

### BIC (Bayesian Information Criterion)

$$
\mathrm{BIC} = k \log n - 2\log L
$$

with $n$ = number of observations. Lower is better.

**Derivation**: large-sample approximation of the log marginal likelihood (Bayesian model evidence). Penalty $k \log n$ grows with $n$.

> **Saying it out loud.** BIC is the parameter count times log $n$, minus twice the log-likelihood. The structural difference is that the penalty grows with sample size, so the more data you have, the stricter BIC gets about adding parameters. It comes from a Laplace approximation to the marginal likelihood, which makes it an approximation to Bayesian model comparison with equal prior weight on each model. That's why the two criteria disagree systematically rather than randomly.

### AIC vs BIC
- BIC penalty grows with $n$ → BIC selects simpler models for large $n$.
- AIC: optimal for *prediction*; doesn't assume true model in candidate set.
- BIC: consistent for *true model selection* if true model is in candidate set.
- BIC > AIC penalty for $n > e^2 \approx 7.4$.

> **Saying it out loud.** They answer different questions, which is why they disagree. AIC targets prediction and doesn't assume the true model is among your candidates, which is realistic, and asymptotically it picks the model with the best predictive error. BIC targets identification and, if the truth really is in your list, will find it with probability going to one. Since log $n$ exceeds two once $n$ is past about seven, BIC is essentially always the stricter one and picks the smaller model. When they disagree, decide by your goal: forecasting points to AIC, explanation points to BIC.

### Limitations
- Both require evaluating likelihood — only meaningful when likelihood is well-defined.
- Don't directly apply to regularized models (effective $k$ unclear).
- Assume model is correctly specified.

> **Saying it out loud.** Three limits worth naming. Both need a genuine likelihood, so they don't apply to most modern ML, which is fit by something other than maximum likelihood. Both assume the model is correctly specified. And the parameter count is ill-defined for anything regularized, since a ridge model with a hundred coefficients doesn't have a hundred free parameters. Effective degrees of freedom, the trace of the hat matrix, patches that partially. Beyond that, just cross-validate, which is why AIC and BIC are far more common in statistics than in machine learning.

---

## 6. ROC and PR curves

### ROC curve
Plot True Positive Rate (TPR) vs False Positive Rate (FPR) as threshold varies.

- TPR = TP / (TP + FN) — sensitivity / recall.
- FPR = FP / (FP + TN) — fall-out.
- Top-left corner = perfect classifier.
- Diagonal = random classifier.

**AUROC** = area under ROC. Probability that a random positive ranks above a random negative.

> **Saying it out loud.** The ROC curve sweeps the decision threshold and plots recall against false alarm rate, so each point is one possible operating decision and the curve summarizes the ranking quality of your scores. Top-left is perfect, the diagonal is random. AUROC, the area under it, has the clean interpretation of being the probability that a random positive outranks a random negative. Its blind spot is imbalance: false positive rate has the count of negatives in its denominator, so with a million negatives you can pile up false positives and the curve barely notices.

### PR curve
Plot Precision vs Recall as threshold varies.
- Better for imbalanced (where most negatives are easy).
- AUPRC: more informative than AUROC for severe imbalance.

> **Saying it out loud.** The precision-recall curve plots precision against recall over the same threshold sweep, and both axes involve the positive class, which is what keeps it informative when positives are rare. The baseline isn't the diagonal, it's a flat line at the prevalence, so a PR-AUC of point-one is genuinely good at one percent prevalence. That's the fact people forget when comparing PR-AUC across datasets. Concretely, a model can show an AUROC of point-nine-five and still have five percent precision at any useful recall, which is the case that makes the argument.

### Choosing operating point
- Cost-aware: $\arg\min_\tau (c_{\mathrm{FN}} \cdot \mathrm{FN}(\tau) + c_{\mathrm{FP}} \cdot \mathrm{FP}(\tau))$.
- Recall constraint: pick $\tau$ such that recall ≥ X.
- F-score optimization: $\tau^* = \arg\max F_\beta$.

> **Saying it out loud.** Picking the threshold is a business decision, and the cleanest framing is expected cost: assign a price to a false negative and a false positive and minimize. In cancer screening a miss dwarfs a false alarm, so you push recall; in spam filtering a false positive loses somebody's email, so you push precision. If you can't get real costs, fall back on a hard constraint, like maximum precision at eighty percent recall, or whatever alert volume your analysts can actually handle. Say out loud that this choice belongs to the product owner, not to the model.

### F-beta score

$$
F_\beta = (1 + \beta^2) \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
$$

$\beta = 1$: F1. $\beta > 1$: weight recall more (e.g., disease screening). $\beta < 1$: weight precision more (e.g., spam).

> **Saying it out loud.** F-beta is the tunable trade between precision and recall, with beta saying how many times more you care about recall. Beta of one is plain F1, beta of two leans toward recall for something like disease screening, beta of a half leans toward precision for something like spam. It's what you use when you can articulate the asymmetry but can't put numbers on the cost of each error type. Two caveats worth adding: it depends on the threshold you chose, unlike AUROC, and it ignores true negatives entirely.

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

> **Saying it out loud.** The confusion matrix is the source of everything and each derived metric is just a different pair of cells over a different denominator. Precision divides by what you predicted positive, recall divides by what was actually positive, specificity is the same idea for negatives. Accuracy uses all four and is therefore dominated by whichever class is bigger, which is why it's useless under imbalance. MCC is the one that uses all four cells in a balanced way, effectively a correlation between predictions and labels, and it's the most honest single-number summary when classes are skewed.

### Why F1 not arithmetic mean?
Harmonic mean penalizes imbalance more — e.g. F1 = 0.5 when both P and R = 0.5. F1 = 0 if either is 0.

> **Saying it out loud.** The harmonic mean is used because it's dominated by the smaller of the two numbers, so you can't hide a terrible half behind an excellent one. Predict everything positive and recall is one while precision is the prevalence: the arithmetic mean would look respectable and F1 stays near zero. And if either precision or recall is exactly zero, F1 is zero, which is exactly the behavior you want from a summary metric.

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

> **Saying it out loud.** The compressed gotcha list: the third term in the bias-variance decomposition is irreducible noise, not something you forgot to tune. Leave-one-out is high variance because its training sets are nearly identical and so its errors are correlated. AIC is for prediction and BIC for identifying the true model. Under imbalance, PR beats ROC because ROC's false positive rate is diluted by the enormous count of easy negatives. Never k-fold a time series. And F1 is the harmonic mean, not the arithmetic one.

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
