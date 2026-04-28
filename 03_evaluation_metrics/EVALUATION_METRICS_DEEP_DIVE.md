# Evaluation Metrics: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Metrics are where most ML projects fail and most ML interviews probe. The wrong metric on the right model is worse than the right metric on a wrong model. Interviewers ask: "Your model achieves 99% accuracy on 99:1 imbalanced data — what's wrong?" If you can't answer cleanly, you can't pass.

---

## 1. The single biggest principle

**Choose your metric before you train.** Choosing it after seeing results is data leakage on the metric itself. You'll pick the metric that flatters your model.

The metric should reflect the actual decision the model is making and the cost of mistakes. Accuracy is rarely the right metric. AUROC is rarely the right metric in production. F1 is rarely the right metric for ranking. Each metric has a specific purpose and specific failure modes.

---

## 2. Classification metrics

### The confusion matrix

For binary classification:

|              | Predicted + | Predicted − |
|---           |---          |---          |
| **Actual +** | TP          | FN          |
| **Actual −** | FP          | TN          |

Almost every classification metric is some ratio of these four quantities.

### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**When it's the right metric:** balanced classes, equal cost of FP and FN, and the absolute level of all four entries is comparable.

**When it's wrong:** **almost always in real ML.** With 99:1 imbalance, predicting all-majority gets 99% accuracy. With heterogeneous costs (medical screening, fraud), accuracy hides the metric you actually care about.

**Interview test:** if a candidate's first instinct is accuracy, they don't have ML maturity.

### Precision and recall

$$
\text{Precision} = \frac{TP}{TP + FP} \qquad \text{(of those I predicted +, how many were actually +?)}
$$

$$
\text{Recall} = \frac{TP}{TP + FN} \qquad \text{(of all actual +, how many did I find?)}
$$

Precision = "don't make false alarms." Recall = "don't miss anything."

These are in tension. As you lower the threshold to predict more positives, recall goes up (you catch more actuals) but precision goes down (more false alarms). The trade-off is captured by the precision-recall curve.

**When precision matters:** spam filtering (false positive = legitimate email blocked = user pain). Recommender systems (you only show top-K, so being wrong about them hurts).

**When recall matters:** medical screening (false negative = missed cancer). Fraud detection (false negative = let the bad guy through). Search (you want all relevant docs).

### F1 score

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R} \qquad \text{(harmonic mean of precision and recall)}
$$

Harmonic mean penalizes imbalance. $F_1 = 0.5$ means *both* are around 0.5; $F_1$ stays low if one is near 0 even if the other is 1. $F_1 = 1$ only when both are 1.

**Generalized: F-beta**

$$
F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}
$$

$\beta > 1$ weights recall more (e.g., $F_2$ for medical screening). $\beta < 1$ weights precision (e.g., $F_{0.5}$ for content moderation where false flags hurt).

### Macro vs micro vs weighted average

For multi-class:

- **Macro average:** average of per-class metrics, treating all classes equally. Penalizes poor performance on rare classes.
- **Micro average:** aggregate all TP/FP/FN across classes, then compute. Dominated by majority class.
- **Weighted average:** macro, but each class weighted by its support (frequency). Compromise.

**Interview gotcha:** if asked "what's the metric?", clarify which average. The choice changes the answer dramatically on imbalanced multi-class.

### AUROC (Area Under Receiver Operating Characteristic curve)

The ROC curve plots TPR (= recall) vs FPR ($= FP/(FP+TN)$) as you vary the classification threshold. AUROC is the area under it.

**Interpretation:** $\text{AUROC} = P(\text{model ranks a random positive higher than a random negative})$. It's a **ranking** metric — measures how well the model separates classes regardless of threshold.

**Properties:**

- AUROC = 0.5: random model.
- AUROC = 1.0: perfect ranking.
- AUROC = 0: perfectly inverted (just flip predictions).
- Threshold-independent: a model whose predicted probabilities are off but whose ranking is good has high AUROC.

**When AUROC misleads:** when classes are heavily imbalanced. AUROC stays high because the absolute number of FPs is bounded by the (large) negative count. PR-AUC (Area Under Precision-Recall curve) is more honest under imbalance.

**Interview gotcha.** "When is AUROC the wrong metric?" Heavy imbalance, or when you care about a specific operating point and the model only needs to be good at that point. AUROC averages performance across all thresholds.

### PR-AUC

Area under Precision-Recall curve. Bounded by class prevalence: a random model has $\text{PR-AUC} \approx \text{class prevalence}$ (not 0.5). So compare against prevalence, not 0.5.

**When PR-AUC is right:** imbalanced classification where you care about precision at high recall.

### Log loss (cross-entropy)

$$
\text{LogLoss} = -\frac{1}{N} \sum_i \big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \big]
$$

Same as binary cross-entropy from logistic regression. **Calibration-aware:** penalizes overconfident wrong predictions much more than just-wrong predictions. The MLE-aligned metric.

**When it's right:** any time you care about probability estimates, not just rankings. Calibrated probabilities matter (medical, financial, ensembling).

**When it misleads:** when you only care about top-K rankings and the absolute probability values don't matter.

### Calibration

A model is **calibrated** if when it says "70% probability," the event happens 70% of the time. Calibration is **not** the same as accuracy or AUROC. Modern neural networks are notoriously overconfident.

**How to test:**

- **Reliability diagram:** bin predictions, plot mean predicted vs observed frequency. Should be on $y = x$.
- **Brier score:** $(1/N) \sum (p_i - y_i)^2$. Murphy's decomposition: $\text{Brier} = \text{reliability} - \text{resolution} + \text{uncertainty}$. Lower reliability = better calibration; higher resolution = bin frequencies vary more across bins; uncertainty is irreducible.
- **Expected Calibration Error (ECE):** weighted average distance between bin frequency and bin mean prediction.

**Fixes:**

- **Platt scaling:** fit $P_{\text{calibrated}} = \sigma(a \cdot \text{score} + b)$ on a held-out set.
- **Isotonic regression:** non-parametric monotonic mapping.
- **Temperature scaling:** for NN softmax — divide logits by $T > 0$ before softmax, fit $T$ on validation. Cheapest and often sufficient.

---

## 3. Regression metrics

### MSE / RMSE

$$
\text{MSE} = \frac{1}{N} \sum (y - \hat y)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}}
$$

RMSE is on the same units as $y$. MSE penalizes large errors quadratically — so a single big mistake dominates.

**When it's right:** when large errors should hurt much more (variance is critical).

**When it misleads:** when you have outliers — a single noisy point dominates. Use MAE for robustness.

### MAE (Mean Absolute Error)

$$
\text{MAE} = \frac{1}{N} \sum |y - \hat y|
$$

Linear penalty. Robust to outliers. Same units as $y$.

**When MAE wins:** when median behavior matters more than mean. Forecasting where some samples are noisy.

### R² (Coefficient of Determination)

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}
$$

How much variance is explained, relative to predicting the mean.

**Properties:**

- $R^2 = 1$: perfect.
- $R^2 = 0$: model = predict the mean.
- $R^2 < 0$: model is worse than predicting the mean.

**Interview gotcha:** $R^2$ is about variance explained, not error. It's a *relative* metric. Different datasets have different $SS_{\text{tot}}$, so $R^2$ doesn't compare across them.

### MAPE (Mean Absolute Percentage Error)

$$
\text{MAPE} = \frac{100}{N} \sum \frac{|y - \hat y|}{|y|}
$$

Scale-invariant. Familiar in business forecasting.

**Failure modes:**

- Undefined when $y = 0$.
- Asymmetric: under-predicting bounded above (can be 100%); over-predicting unbounded.
- Misleading for small $y$ (small absolute error becomes huge percentage).

**Better alternatives:** SMAPE (symmetric), MASE (Mean Absolute Scaled Error — compares to a naive baseline).

### Quantile loss

For predicting the $\tau$-th quantile:

$$
\mathcal{L}_\tau = \sum_i \max\!\big(\tau \cdot (y_i - \hat y_i),\ (\tau - 1) \cdot (y_i - \hat y_i)\big)
$$

For $\tau = 0.5$, recovers MAE (median regression). For $\tau = 0.9$, optimizes the 90th percentile prediction. Useful for delivery time estimation, demand forecasting with safety stock, etc.

---

## 4. Ranking metrics

For tasks where you produce a ranked list and care about relevance.

### Precision@k, Recall@k

$$
\text{Precision@}k = \frac{\text{relevant in top } k}{k}, \qquad \text{Recall@}k = \frac{\text{relevant in top } k}{\text{total relevant}}
$$

Used in IR, recommendation systems. Picking $k$ is the hard part.

### MAP (Mean Average Precision)

For each query, compute Average Precision (AP) = average of precision at each relevant document's rank. Then average across queries. Position-aware: missing a top-rank relevant doc hurts more than missing a low-rank one.

### NDCG (Normalized Discounted Cumulative Gain)

$$
\text{DCG@}k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
$$

$$
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k} \quad (\text{IDCG} = \text{perfect ranking's DCG})
$$

Position-discounted: relevant items at high positions count more. The $2^{\text{rel}}$ term means graded relevance (multi-level).

**Properties:**

- $\text{NDCG} \in [0, 1]$.
- More forgiving of tiny rank swaps far down the list.
- Standard in search, recommendation.

### MRR (Mean Reciprocal Rank)

$$
\text{RR} = \frac{1}{\text{rank of first correct answer}}, \qquad \text{MRR} = \text{mean RR across queries}
$$

For tasks with one correct answer (Q&A, factoid retrieval). Hard penalty for not having the answer at rank 1.

---

## 5. LLM-specific evaluation

### Perplexity

$$
\text{PPL} = \exp(\text{cross-entropy loss}) = \exp\!\left(-\frac{1}{N} \sum_i \log P(x_i \mid x_{<i})\right)
$$

How "surprised" the model is by the test data. Lower is better. Geometrically, the inverse of the average per-token probability the model assigned to the actual data.

**Properties:**

- Bounded below by $\exp(H_{\text{true}})$ where $H_{\text{true}}$ is the entropy of the true data distribution. Only equals 1 for deterministic data; for natural language $H_{\text{true}} > 0$ so the floor is strictly $> 1$.
- Bounded above by vocabulary size (if the model is uniform random over vocab, $\text{PPL} = |V|$).
- Tokenizer-dependent: different tokenizers give different PPL even on the same text. **Cannot directly compare PPL across models with different tokenizers.**

**Why it's useful:** the most natural metric for autoregressive LMs. Directly tied to the loss being optimized.

**Why it's limited:** a model with low PPL is not necessarily a good chat assistant. PPL measures how well the model predicts the next token; it doesn't measure whether the responses are helpful, factual, or safe.

### BLEU (Bilingual Evaluation Understudy)

For machine translation. Measures n-gram overlap between candidate and reference translations:

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_n w_n \log p_n\right)
$$

where $p_n$ = precision of n-grams, $\text{BP}$ = brevity penalty, $w_n$ = weights (usually uniform over $n = 1, \ldots, 4$).

**Failure modes:**

- Multiple valid translations exist; BLEU picks one as ground truth.
- Doesn't capture meaning — paraphrases score badly.
- Surface-level: cares about token overlap, not semantics.
- Replaced by COMET, BLEURT for state-of-the-art evaluation.

### ROUGE

For summarization. Recall-oriented n-gram overlap (ROUGE-N) or longest common subsequence (ROUGE-L). Same surface-level limitations as BLEU.

### Exact Match (EM) / F1 (token)

For Q&A and reading comprehension. EM = exact string match. F1 = token-level F1 between predicted and reference answers.

### Pass@k (code generation)

$$
\text{Pass@}k = \mathbb{E}\!\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
$$

where $n$ = samples generated, $c$ = number that pass tests, $k$ = number you'd actually use. It's the probability that at least one of $k$ samples passes.

For HumanEval, MBPP, etc. Standardized across the field.

### LLM-as-judge metrics

Use a stronger LLM (GPT-4) to grade outputs. Examples: AlpacaEval, MT-Bench, Arena-Hard.

**Pros:** scalable, captures quality more holistically than n-gram overlap.

**Cons:**

- Judge biases (length, style, sycophancy).
- Judge errors compound with model errors.
- Cost: large LLM API calls per evaluation.
- Sometimes systematically biased toward outputs that look like the judge's own.

**Mitigations:** average across multiple judges, length-control prompts, blinded comparisons.

### Human evaluation

Gold standard. Slow, expensive, gold standard. Common formats:

- **Side-by-side preference:** A vs B, "which is better?"
- **Likert ratings:** 1-5 score on specific axes (helpfulness, factuality).
- **Hold-out from training:** never let evaluators contribute to training data.

---

## 6. Common metric pitfalls

### Pitfall 1: data leakage between train and eval

Test set must be drawn from the **same distribution as deployment**, with no overlap with training. Common leaks:

- Time-series leakage: train on future, test on past.
- Group leakage: same user/document in train and test.
- Feature leakage: a feature that's only available after the prediction is made.

### Pitfall 2: ignoring the deployment distribution

Train metric is on training distribution. Eval metric should be on **deployment distribution**. If the production data is different (different user demographics, different time of day, drift over time), eval metrics will overstate performance.

### Pitfall 3: optimizing for the wrong proxy

A team optimizes click-through rate (CTR) for a recommendation system. The model learns to recommend clickbait. CTR goes up; user satisfaction goes down; eventually retention crashes.

The metric you train on should match (or proxy) the metric you actually care about. Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

### Pitfall 4: tail behavior

Average metrics can hide bad behavior on rare slices. A 90% accurate model might fail catastrophically on a specific demographic. Stratify your evaluation: per-language, per-region, per-user-segment.

### Pitfall 5: metric drift

What was a good metric a year ago may not be today as the task evolves. Re-validate metrics periodically against ground truth.

### Pitfall 6: false comparisons

Comparing PPL across models with different tokenizers. Comparing AUROC across datasets with different class balances. Comparing BLEU across language pairs. All meaningless without normalization.

---

## 7. Cross-validation and statistical significance

A single eval number with no error bar isn't science. Frontier-lab interviews often probe whether you understand this.

### k-fold CV

Split training into $k$ folds, train on $k-1$, evaluate on the remaining. Average across folds. Gives a CV estimate of generalization.

**For time series:** use forward-chaining CV (train on $[1, \ldots, t]$, test on $[t+1, \ldots, t+h]$) — never train on data after the test point.

### Stratified k-fold

For imbalanced classes: ensure each fold has the same class distribution as the full data. Default in sklearn for classification.

### Confidence intervals

Bootstrap resampling: compute the metric on $B$ bootstrap samples; the 2.5–97.5 percentile gives a 95% CI.

### Significance tests

- **McNemar's test** for paired classifier comparison.
- **Paired t-test** for paired regression / continuous metrics.
- **Permutation tests** for non-parametric comparisons.

### Multiple comparisons

If you're evaluating 100 hyperparameter configurations, even random noise will give you "winners" by chance. Bonferroni correction or false discovery rate (FDR) controls.

---

## 8. The 10 most-asked evaluation interview questions

1. **Why is accuracy a bad metric for imbalanced data?** Predicts all-majority and gets near-100% accuracy without learning anything.
2. **Precision vs recall — when which?** Precision when false positives hurt (spam, ads). Recall when false negatives hurt (medical, fraud).
3. **What's F1 and when is it appropriate?** Harmonic mean of P and R. Penalizes imbalance. F-$\beta$ to weight one over the other.
4. **AUROC vs PR-AUC?** AUROC ranks well across thresholds but inflates on imbalanced data; PR-AUC honest under imbalance.
5. **Calibration — what and how to test?** Predicted probabilities match observed frequencies. Test with reliability diagrams, Brier score, ECE.
6. **What's perplexity and what are its limits?** $\exp(\text{cross-entropy})$. Tokenizer-dependent — can't compare across tokenizers.
7. **What's pass@k?** Probability that $\geq 1$ of $k$ samples solves a coding problem. Standard for HumanEval-style code generation.
8. **LLM-as-judge — what biases does it have?** Length, style, sycophancy, judge-self-similarity. Mitigations: ensemble, length control, blinded.
9. **Why do you need separate train/val/test sets?** Train: learn parameters. Val: tune hyperparameters. Test: estimate deployment performance. Mixing them leaks.
10. **Goodhart's Law in evaluation?** When a metric becomes a target, it ceases to be a good measure. Pick metrics that proxy what you actually care about, not what's easy to optimize.

---

## 9. Drill plan

1. Master precision/recall/F1 derivations and trade-offs.
2. Know AUROC vs PR-AUC and when each misleads.
3. Know calibration tests (reliability diagram, Brier, ECE) and fixes (Platt, isotonic, temperature).
4. Know perplexity definition and tokenizer-dependence.
5. Know pass@k formula.
6. Drill `INTERVIEW_GRILL.md`.
