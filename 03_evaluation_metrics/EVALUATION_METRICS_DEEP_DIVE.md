# Evaluation Metrics: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Metrics are where most ML projects fail and most ML interviews probe. The wrong metric on the right model is worse than the right metric on a wrong model. Interviewers ask: "Your model achieves 99% accuracy on 99:1 imbalanced data — what's wrong?" If you can't answer cleanly, you can't pass.

---

## 1. The single biggest principle

**Choose your metric before you train.** Choosing it after seeing results is data leakage on the metric itself. You'll pick the metric that flatters your model.

The metric should reflect the actual decision the model is making and the cost of mistakes. Accuracy is rarely the right metric. AUROC is rarely the right metric in production. F1 is rarely the right metric for ranking. Each metric has a specific purpose and specific failure modes.

> **Saying it out loud.** The one rule I'd lead with is: pick your metric before you look at any results. If you choose afterwards, you'll unconsciously pick whichever one makes your model look good, and that's leakage at the level of the evaluation itself. The metric should mirror the actual decision and the actual cost of each kind of mistake — which is why accuracy is almost never right in production, and why the answer to 'what metric?' should start with a question about what happens when the model is wrong.

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

> **Saying it out loud.** Accuracy is how often you're right overall, and it's the metric most likely to flatter a useless model. With 99-to-1 imbalance, predicting the majority every time scores 99% while catching nothing — so what you're really measuring is the class balance. It's only the right metric when classes are roughly balanced and a false positive costs about the same as a false negative, which is rare. If accuracy is someone's first instinct on an imbalanced problem, that's the tell.

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

> **Saying it out loud.** Precision is 'of the things I flagged, how many were real'; recall is 'of the real things, how many did I catch'. They pull against each other, because the only way to catch more is to flag more, and flagging more means more false alarms — the threshold is the dial between them. Which one you optimize is a business question, not a modeling one: spam filtering cares about precision because blocking a real email is expensive, cancer screening cares about recall because a miss can kill someone. In practice you usually fix a floor on one and maximize the other.

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

> **Saying it out loud.** F1 is the harmonic mean of precision and recall, and the harmonic part is the whole point. If you flag everything, recall is 1 and precision is near zero — the arithmetic mean would give you a respectable 0.5 for a model that does nothing, while the harmonic mean gives you approximately zero. So F1 is only high when both are high. F-beta is the version with a dial: beta above 1 weights recall, below 1 weights precision, and stating which one you'd use forces you to name your actual cost tradeoff.

### Macro vs micro vs weighted average

For multi-class:

- **Macro average:** average of per-class metrics, treating all classes equally. Penalizes poor performance on rare classes.
- **Micro average:** aggregate all TP/FP/FN across classes, then compute. Dominated by majority class.
- **Weighted average:** macro, but each class weighted by its support (frequency). Compromise.

**Interview gotcha:** if asked "what's the metric?", clarify which average. The choice changes the answer dramatically on imbalanced multi-class.

> **Saying it out loud.** The averaging choice can change your headline number by thirty points, so always say which one you used. Macro averages the per-class scores flat, so a class with fifty examples counts as much as one with fifty thousand — that's what you want if rare classes matter. Micro pools everything first, so it's dominated by the big classes and ends up close to accuracy. The diagnostic value is in the gap: micro at 0.92 with macro at 0.55 means you're quietly failing on your rare classes.

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

> **Saying it out loud.** AUROC is a ranking metric — the probability that a random positive scores above a random negative. That's why it's threshold-independent and why 0.5 means you're guessing. It's genuinely useful for asking 'does the model order things correctly', but it lies to you under heavy imbalance: false positive rate divides by an enormous negative count, so ten thousand false alarms barely move the curve while ruining the experience of whoever has to review them. The other limitation is that it averages over every threshold, including ones you'd never deploy at.

### PR-AUC

Area under Precision-Recall curve. Bounded by class prevalence: a random model has $\text{PR-AUC} \approx \text{class prevalence}$ (not 0.5). So compare against prevalence, not 0.5.

**When PR-AUC is right:** imbalanced classification where you care about precision at high recall.

> **Saying it out loud.** PR-AUC is the honest version under imbalance, because precision never divides by the huge negative count — every false positive is felt. The critical detail people forget is the baseline: random guessing gives AUROC 0.5 always, but gives a PR-AUC equal to your positive rate, which might be 0.01. So a PR-AUC of 0.3 on a 1% prevalence problem is a thirty-fold lift over random, not a failing grade. Always report prevalence next to it or the number is uninterpretable.

### Log loss (cross-entropy)

$$
\text{LogLoss} = -\frac{1}{N} \sum_i \big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \big]
$$

Same as binary cross-entropy from logistic regression. **Calibration-aware:** penalizes overconfident wrong predictions much more than just-wrong predictions. The MLE-aligned metric.

**When it's right:** any time you care about probability estimates, not just rankings. Calibrated probabilities matter (medical, financial, ensembling).

**When it misleads:** when you only care about top-K rankings and the absolute probability values don't matter.

> **Saying it out loud.** Log loss scores you on how surprised you were by the truth, so it's the metric that actually cares about your probabilities rather than just their ordering. Being 99% confident and wrong costs you enormously more than being 60% confident and wrong, which is exactly the behavior you want when a downstream system does expected-value math on the number. It's also the thing you were already optimizing, since it's the training loss. Where it misleads is a pure ranking problem — if you only ever show a top-10 list, the absolute probabilities don't matter and log loss is measuring something you don't care about.

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

> **Saying it out loud.** Calibration is whether the model's confidence means anything: among the cases it calls 70%, roughly 70% should happen. The thing to stress is that it's completely independent of discrimination — you can have a perfect AUROC and terrible calibration, because any monotone squashing of the scores leaves the ranking untouched. Neural networks are reliably overconfident, because cross-entropy keeps rewarding pushing 0.9 toward 0.99 long after accuracy plateaus. The fix is cheap and you should name it: temperature scaling on a held-out set, one parameter, doesn't change accuracy at all.

---

## 3. Regression metrics

### MSE / RMSE

$$
\text{MSE} = \frac{1}{N} \sum (y - \hat y)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}}
$$

RMSE is on the same units as $y$. MSE penalizes large errors quadratically — so a single big mistake dominates.

**When it's right:** when large errors should hurt much more (variance is critical).

**When it misleads:** when you have outliers — a single noisy point dominates. Use MAE for robustness.

> **Saying it out loud.** Squaring means big errors dominate — one prediction that's off by ten hurts as much as a hundred that are off by one. That's what you want when a single large miss is genuinely catastrophic. It's also what you don't want when your data has outliers, because a handful of noisy points will steer the whole fit. RMSE is the same thing in interpretable units, which matters more than it sounds: 'we're typically off by 50 units' lands in a meeting, 'MSE is 2500' does not.

### MAE (Mean Absolute Error)

$$
\text{MAE} = \frac{1}{N} \sum |y - \hat y|
$$

Linear penalty. Robust to outliers. Same units as $y$.

**When MAE wins:** when median behavior matters more than mean. Forecasting where some samples are noisy.

> **Saying it out loud.** MAE treats every unit of error the same, so one huge miss counts exactly as much as its size and no more. That makes it robust to outliers, which is usually the reason people reach for it. The deeper reason is worth knowing: minimizing squared error predicts the conditional *mean*, minimizing absolute error predicts the conditional *median*. So it's not just a robustness choice, it's a choice about which summary of the distribution you want your model to output.

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

> **Saying it out loud.** $R^2$ compares you against the dumbest reasonable baseline — always predicting the average. One means perfect, zero means you've matched the baseline, and negative means you're worse than a horizontal line, which in practice signals a bug rather than a subtle modeling issue. The trap is treating it as an absolute quality score: it depends on how much variance the target had in the first place, so an $R^2$ of 0.3 on a noisy financial series can be excellent while 0.9 on a smooth one is unremarkable. It doesn't compare across datasets.

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

> **Saying it out loud.** MAPE breaks exactly where you need it. It divides by the true value, so it's undefined at zero and explodes for small values — one item with true demand of 1 contributes more than a hundred well-predicted large ones. It's also asymmetric: over-predicting can cost an unbounded percentage while under-predicting caps at 100%, so optimizing MAPE quietly biases your forecasts low. If you need a scale-free error, MASE compares you to a naive baseline and doesn't have either problem.

### Quantile loss

*In plain terms:* this is a loss that charges you a different price for guessing too high than for guessing too low. Set $\tau$ to 0.5 and the two prices are equal, which gives you the median. Set it to 0.9 and under-predicting costs nine times as much, so the model learns to sit at the 90th percentile — which is how you get prediction intervals out of a point-prediction model.

For predicting the $\tau$-th quantile:

$$
\mathcal{L}_\tau = \sum_i \max\!\big(\tau \cdot (y_i - \hat y_i),\ (\tau - 1) \cdot (y_i - \hat y_i)\big)
$$

For $\tau = 0.5$, recovers MAE (median regression). For $\tau = 0.9$, optimizes the 90th percentile prediction. Useful for delivery time estimation, demand forecasting with safety stock, etc.

> **Saying it out loud.** Quantile loss is how you get uncertainty out of a model that only makes point predictions. Because the penalty is asymmetric, the minimizer isn't the mean or the median but whatever quantile you dialed in — so fit one model at $\tau = 0.05$ and one at 0.95 and you have a prediction band. That's the right tool whenever the cost of being under is different from the cost of being over: delivery estimates, inventory with safety stock, capacity planning. And unlike a Gaussian error bar, it makes no assumption about the shape of the distribution.

---

## 4. Ranking metrics

For tasks where you produce a ranked list and care about relevance.

### Precision@k, Recall@k

$$
\text{Precision@}k = \frac{\text{relevant in top } k}{k}, \qquad \text{Recall@}k = \frac{\text{relevant in top } k}{\text{total relevant}}
$$

Used in IR, recommendation systems. Picking $k$ is the hard part.

> **Saying it out loud.** Precision-at-k and recall-at-k answer different questions and the k means different things in each. Precision@10 is 'of the ten results on the page, how many were good' — that's the user experience in search. Recall@k is 'did my candidate set even contain the right answers', which is what you measure for the retrieval stage of a pipeline, because a reranker can fix ordering but can never recover something you never retrieved. In a two-stage system you optimize recall@100 upstream and precision@10 downstream.

### MAP (Mean Average Precision)

For each query, compute Average Precision (AP) = average of precision at each relevant document's rank. Then average across queries. Position-aware: missing a top-rank relevant doc hurts more than missing a low-rank one.

> **Saying it out loud.** MAP asks how high up the list you managed to put the relevant items, averaged over queries. You walk down the ranking, and each time you hit something relevant you record the precision at that point — so a hit at rank 2 is worth far more than one at rank 50. It's the standard when a query has several correct answers and they're all equally correct. Its limitation is exactly that: relevance is binary, so it can't express 'this result is great and that one is merely okay'.

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

> **Saying it out loud.** NDCG is the ranking metric for when relevance comes in degrees. Each item contributes a gain based on its relevance grade, divided by a log of its position so the top of the list dominates, and then you normalize by the score of the perfect ordering — which is what makes queries with different numbers of relevant documents comparable. It's the industry default for search and recommendation. The thing to be careful about is the exponential in the gain: it makes NDCG very sensitive to how you assign relevance grades, so the labeling guidelines matter as much as the model.

### MRR (Mean Reciprocal Rank)

$$
\text{RR} = \frac{1}{\text{rank of first correct answer}}, \qquad \text{MRR} = \text{mean RR across queries}
$$

For tasks with one correct answer (Q&A, factoid retrieval). Hard penalty for not having the answer at rank 1.

> **Saying it out loud.** MRR only looks at where your first correct answer landed — rank one gives 1.0, rank two gives 0.5, rank ten gives 0.1. That drop-off is deliberately brutal, and it's the right shape when the user reads one answer and leaves, like a voice assistant or factoid Q&A. The limitation to name is that it's blind to everything after that first hit, so a system that puts one right answer at the top and garbage below it scores identically to one that's good all the way down. Wrong metric if the user wants comprehensive results.

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

> **Saying it out loud.** Perplexity is how many options the model was effectively choosing between at each token — perplexity 20 means it was about as uncertain as picking uniformly among 20 words. It's just the exponential of the loss you were already training on, which is why it's the natural intrinsic metric. Two limits worth naming: the floor isn't 1, it's the true entropy of the language, since the next word is genuinely uncertain even for a perfect model. And low perplexity does not make a good assistant — it measures next-token prediction, not helpfulness, factuality, or safety.

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

> **Saying it out loud.** BLEU counts how many short word sequences your translation shares with a human reference — unigrams through 4-grams, combined geometrically so you have to do decently at all lengths, times a brevity penalty so you can't cheat by being terse. It's crude and it works well enough to have run the field for twenty years. The failure mode is that it can't distinguish a good paraphrase from a bad translation: say the same thing in different words and you're punished, while fluent nonsense built from reference phrasing is rewarded. That's why serious MT eval has moved to learned metrics like COMET.

### ROUGE

For summarization. Recall-oriented n-gram overlap (ROUGE-N) or longest common subsequence (ROUGE-L). Same surface-level limitations as BLEU.

> **Saying it out loud.** ROUGE is BLEU pointed the other way — recall-oriented, for summarization, because a summary's characteristic failure is leaving something out rather than adding something. ROUGE-N counts the reference n-grams you managed to include; ROUGE-L uses the longest common subsequence, so ordering matters loosely without requiring exact contiguity. It inherits every one of BLEU's surface-level problems and adds one of its own: it rewards long summaries that copy source sentences verbatim, which is usually the opposite of what you want.

### Exact Match (EM) / F1 (token)

For Q&A and reading comprehension. EM = exact string match. F1 = token-level F1 between predicted and reference answers.

> **Saying it out loud.** Exact match is the strictest possible scoring — the string matches or it doesn't — so it's harsh in a specific way, punishing 'Paris, France' when the reference said 'Paris'. Token-level F1 softens that by measuring word overlap between your answer and the reference, which is why SQuAD-style benchmarks report both. The pair is informative: a big gap between EM and F1 means the model is finding the right region of the answer but not the exact span, which is a formatting problem rather than a comprehension one.

### Pass@k (code generation)

*In plain terms:* pass@k is 'if the model gets $k$ tries, how often does at least one work?'. The formula looks combinatorial because generating exactly $k$ samples gives a high-variance estimate, so instead you generate a larger $n$, count how many pass, and compute the probability that a random subset of size $k$ contains at least one of them.

$$
\text{Pass@}k = \mathbb{E}\!\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
$$

where $n$ = samples generated, $c$ = number that pass tests, $k$ = number you'd actually use. It's the probability that at least one of $k$ samples passes.

For HumanEval, MBPP, etc. Standardized across the field.

> **Saying it out loud.** Pass@k is the code-generation metric, and it's honest because code has an objective checker — you run the unit tests. The interesting part is comparing pass@1 with pass@10 or pass@100: pass@1 is what a user experiences, while the higher ones tell you what the model is capable of when it gets retries. A big gap means the model can solve the problem but can't tell which of its answers is right, which is a ranking failure and points you at reranking or a verifier. Just don't quote pass@100 as a product capability unless you actually have a checker at inference time.

### LLM-as-judge metrics

Use a stronger LLM (GPT-4) to grade outputs. Examples: AlpacaEval, MT-Bench, Arena-Hard.

**Pros:** scalable, captures quality more holistically than n-gram overlap.

**Cons:**

- Judge biases (length, style, sycophancy).
- Judge errors compound with model errors.
- Cost: large LLM API calls per evaluation.
- Sometimes systematically biased toward outputs that look like the judge's own.

**Mitigations:** average across multiple judges, length-control prompts, blinded comparisons.

> **Saying it out loud.** LLM-as-judge is the only thing that scales to open-ended generation, and it has known, measurable biases you should name before someone else does. Judges prefer longer answers, reward markdown and confident tone over correctness, are sycophantic toward whatever position was in the prompt, and favor their own family's outputs. There's also strong position bias in pairwise comparisons, which is why you always run both orderings and average. Treat it as a noisy rater: use a rubric, control for length, blind the comparison, and validate against human labels on a subset before you trust the number.

### Human evaluation

Gold standard. Slow, expensive, gold standard. Common formats:

- **Side-by-side preference:** A vs B, "which is better?"
- **Likert ratings:** 1-5 score on specific axes (helpfulness, factuality).
- **Hold-out from training:** never let evaluators contribute to training data.

> **Saying it out loud.** Human evaluation is the ground truth everything else is a proxy for, and it's slow and expensive enough that you use it sparingly and deliberately. Side-by-side preference is usually the most reliable format, because people are much better at 'which of these two is better' than at assigning an absolute 1-to-5 score. The things that actually determine whether the numbers mean anything are procedural: blind the raters to which system is which, randomize presentation order, measure inter-annotator agreement, and never let anyone who wrote training data grade the outputs.

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

> **Saying it out loud.** Most bad evaluations fail in one of a few standard ways, and they're all versions of 'the test set didn't look like deployment'. Leakage is the big one — training on the future, the same user appearing in both splits, or a feature that only exists after the outcome — and the tell is a number that's too good rather than too bad. Then there's optimizing a proxy until it stops proxying, which is Goodhart's law: chase click-through and you get clickbait. And averaging over slices hides the demographic where you're at 60%. The habit that catches most of it is asking, of every feature and every split, 'would I actually have this at prediction time?'

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

> **Saying it out loud.** A single number with no error bar isn't a result. Bootstrap is the honest default — resample the test set a thousand times, recompute the metric, take the 2.5th and 97.5th percentiles — and it works for weird metrics like AUROC or NDCG where the analytic formula is painful. When comparing two models on the same data, bootstrap the *difference* rather than each metric separately, because that accounts for the correlation and gives a much tighter interval. And remember multiple comparisons: test a hundred configurations at a 5% threshold and about five will look significant by pure chance.

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
6. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).
