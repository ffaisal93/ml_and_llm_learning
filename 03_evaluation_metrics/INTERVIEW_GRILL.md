# Evaluation Metrics — Interview Grill

> 50 questions on evaluation metrics. Drill until you can answer 35+ cold.

---

## A. Classification basics

**1. Define accuracy, precision, recall, F1.**
$\text{Accuracy} = (TP + TN)/(TP + TN + FP + FN)$. $\text{Precision} = TP/(TP + FP)$. $\text{Recall} = TP/(TP + FN)$. $F_1 = 2PR/(P + R)$ — harmonic mean of $P$ and $R$.

**2. Why is accuracy bad on imbalanced data?**
Predicting majority class trivially gets $1 - \text{minority\_fraction}$ accuracy. With 99:1 imbalance, predicting all-majority gets 99% accuracy without learning anything. Model could be useless yet "high accuracy."

**3. When is precision the right metric?**
When false positives hurt: spam (legitimate email blocked), recommendations (showing bad items burns trust), content moderation (false flags = censorship complaints), ads (FP = wasted budget).

**4. When is recall the right metric?**
When false negatives hurt: medical screening (missed cancer), fraud detection (let bad guy through), search recall (missed relevant docs), safety filters (missed harmful content).

**5. Why is F1 the harmonic mean and not arithmetic?**
Harmonic mean penalizes imbalance: $F_1$ stays low if either $P$ or $R$ is near 0, even if the other is 1. Arithmetic mean would give 0.5 for ($P=1, R=0$), masking the failure.

**6. What's F-beta?**
$F_\beta = (1 + \beta^2) \cdot PR / (\beta^2 \cdot P + R)$. $\beta > 1$ weights recall ($\beta=2$ = "recall is twice as important"). $\beta < 1$ weights precision. $F_1 = F_{\beta=1}$.

**7. Macro vs micro vs weighted average for multi-class?**
Macro: average per-class metrics equally — penalizes poor performance on rare classes. Micro: aggregate TP/FP/FN across classes then compute — dominated by majority. Weighted: macro weighted by class frequency.

**8. Why might macro F1 differ from micro F1?**
On imbalanced multi-class. Macro treats rare classes as equally important; micro is dominated by frequent classes. Macro $F_1 \ll$ micro $F_1$ means rare classes are being missed.

---

## B. AUROC and PR-AUC

**9. What does AUROC measure?**
Probability that the model ranks a random positive higher than a random negative. Threshold-independent ranking quality. AUROC = 0.5: random; AUROC = 1: perfect ranking.

**10. How is AUROC computed?**
Plot TPR ($=$ recall) vs FPR ($= FP/(FP+TN)$) as you sweep the classification threshold. Area under that curve. Equivalently: pairwise ranking probability.

**11. When does AUROC mislead?**
On heavily imbalanced data. The (very large) negative count keeps FPR low even with many FPs. AUROC can stay high while precision is terrible.

**12. AUROC vs PR-AUC?**
AUROC: TPR vs FPR. PR-AUC: precision vs recall. AUROC bounded below at 0.5 by random; PR-AUC bounded below at class prevalence by random. PR-AUC more honest under imbalance.

**13. When should you report PR-AUC?**
Imbalanced classification where you care about precision at high recall. Example: fraud detection with 1% fraud rate — AUROC of 0.95 sounds great but PR-AUC of 0.3 reveals the truth.

**14. What's the relationship between AUROC and the Mann-Whitney U test?**
They're equivalent. $\text{AUROC} = U / (n_{\text{pos}} \cdot n_{\text{neg}})$ for the rank-sum statistic $U$. Both measure: how often does a positive rank above a negative.

---

## C. Calibration

**15. What does calibration mean?**
Predicted probabilities match observed frequencies. If the model says "70%" and the event happens 70% of the time on those predictions, it's calibrated. Independent from accuracy or AUROC.

**16. How do you measure calibration?**
Reliability diagram (bin predictions, plot mean predicted vs observed frequency; should be y=x). Brier score (MSE between p and y). ECE (weighted average distance between bin frequency and bin mean prediction).

**17. Decompose Brier score.**
Brier = calibration + refinement (− uncertainty). Calibration = how far bin predictions are from bin frequencies. Refinement = how informative the bins are. Lower is better for both.

**18. How do you fix miscalibration?**
Platt scaling ($\sigma(a \cdot \text{score} + b)$ fit on val), isotonic regression (non-parametric monotonic), temperature scaling ($\text{logits}/T$ for softmax). Temperature is cheapest, fits one parameter, often sufficient for NN softmax.

**19. Why are deep neural networks miscalibrated?**
Overconfident due to high capacity: NN drives training cross-entropy near 0 by pushing logits to extremes, even when validation accuracy plateaus. Probabilities concentrate at 0/1 even when the model should be uncertain.

**20. What's log loss?**
$-(1/N) \sum [y \log p + (1-y) \log(1-p)]$. Same as binary cross-entropy. Calibration-aware: penalizes overconfident wrong predictions much more than just-wrong predictions. Aligned with MLE.

---

## D. Regression

**21. MSE vs MAE — when which?**
MSE: when large errors should hurt much more (variance critical). Sensitive to outliers. MAE: robust to outliers, predicts the median. Choose by what error distribution matters for your task.

**22. What does RMSE tell you that MSE doesn't?**
Same units as $y$. $\text{RMSE} = \sqrt{\text{MSE}}$. Easier to interpret in domain terms. Otherwise mathematically equivalent.

**23. What does $R^2$ of $-0.2$ mean?**
Model is worse than predicting the mean. $R^2 < 0$ happens; it means the model has *negative* explanatory power. Common bug source — should investigate immediately.

**24. Why is MAPE problematic?**
Undefined at $y = 0$. Asymmetric (under-predicting capped at 100%; over-predicting unbounded). Misleading for small $y$. Use SMAPE or MASE instead.

**25. What's quantile loss?**
$\mathcal{L}_\tau = \sum \max(\tau \cdot (y - \hat y), (\tau - 1) \cdot (y - \hat y))$. For $\tau = 0.5$, recovers MAE (median). For $\tau = 0.9$, optimizes 90th percentile. Useful for uncertainty quantification, conformal prediction, demand forecasting with safety stock.

---

## E. Ranking and IR

**26. What's MAP?**
Mean Average Precision. For each query, AP = average of precision at each relevant document's rank. Then average across queries. Position-aware: missing top-rank relevant docs hurts more.

**27. What's NDCG?**
Normalized Discounted Cumulative Gain. $\text{DCG} = \sum (2^{\text{rel}_i} - 1) / \log_2(i + 1)$. Normalized by ideal DCG. Position-discounted, handles graded relevance. Standard in search ranking.

**28. What's MRR?**
Mean Reciprocal Rank. $\text{RR} = 1/\text{rank of first correct}$. Hard penalty for not having the answer at rank 1. For tasks with one right answer (factoid Q&A).

**29. Precision@k vs Recall@k — when to choose?**
Precision@k when you only show top-k (e.g., 10 search results) and care about quality of those k. Recall@k when you care about coverage at fixed k.

---

## F. LLM-specific

**30. Define perplexity.**
`PPL = exp(−(1/N) Σ log P(x_i | x_{<i}))`. Geometric inverse of average per-token probability. Bounded below by $\exp(H_{\text{true}})$ — equals 1 only for deterministic data; for natural language the floor is strictly above 1. Bounded above by vocab size (uniform random model = `|V|`).

**31. Why can't you compare PPL across models with different tokenizers?**
PPL is per-token. Different tokenizers split text into different numbers of tokens. A model with finer tokenization gets lower PPL on the same text purely because it's predicting more tokens. **Compare per-byte or per-character likelihood instead** for cross-tokenizer comparison.

**32. What's pass@k?**
`Pass@k = E[1 − C(n−c, k)/C(n, k)]` where `n` = samples generated, `c` = pass count. Probability that at least one of `k` independent samples solves the problem. Standard for code generation.

**33. Why pass@1 vs pass@10 vs pass@100?**
Pass@1: model's first answer; mimics typical user. Pass@10/100: best-of-N capability; mimics repeated retry workflows. The gap between pass@1 and pass@10 measures how many right answers the model has but doesn't surface first.

**34. What's BLEU?**
Bilingual Evaluation Understudy. n-gram overlap between candidate and reference translations: $\text{BLEU} = \text{BP} \cdot \exp(\sum_n w_n \log p_n)$ where $w_n = 1/N$ uniformly (so weights sum to 1). Brevity penalty $\text{BP}$ discourages too-short outputs.

**35. BLEU's failure modes?**
Multiple valid translations; n-gram overlap misses paraphrases; surface-level (no semantics). COMET, BLEURT, GEMBA-MQM increasingly replace BLEU for serious MT eval.

**36. ROUGE — what and where?**
Recall-oriented n-gram overlap (ROUGE-N) or longest common subsequence (ROUGE-L). For summarization. Same surface-level limitations as BLEU.

**37. LLM-as-judge biases?**
Length (judges prefer longer outputs), style (formal/markdown formatting boosts ratings), sycophancy (prefers responses agreeing with the judge), self-similarity (prefers outputs from same model family). Mitigations: ensemble, length control, blinded comparison.

**38. What's win-rate vs Elo for LLM eval?**
Win-rate: fraction of pairwise comparisons where model A beats B. Elo: dynamic rating from many pairwise comparisons (chess-style). Used in LMSYS Chatbot Arena. Both pairwise but Elo is multi-model.

---

## G. Methodology and pitfalls

**39. Why do you need separate train/val/test?**
Train: fit parameters. Val: tune hyperparameters and early-stop. Test: estimate deployment performance (used once, never tuned against). Reusing val for test inflates estimates.

**40. What's data leakage and how do you detect it?**
A test-set feature or label is influenced by training data. Detect by: too-good-to-be-true performance, feature importance dominated by suspicious features (timestamps, IDs), random shuffling boosting metric absurdly.

**41. Time-series cross-validation?**
Forward-chaining: train on `[1..t]`, test on `[t+1..t+h]`. Never train on future and test on past. Standard k-fold leaks future into past.

**42. Stratified k-fold?**
For imbalanced classification: ensure each fold has the same class distribution as full data. Default in sklearn for classification.

**43. How do you compute confidence intervals on a metric?**
Bootstrap resampling: B bootstrap samples; metric on each; 2.5–97.5 percentile gives 95% CI. Or analytically (delta method) for simple metrics. Always report CIs for serious comparisons.

**44. Multiple comparison correction?**
If you evaluate 100 configurations, some will look "significantly" better by random noise. Bonferroni (divide α by number of tests) is conservative. False Discovery Rate (Benjamini-Hochberg) is less conservative, more practical.

**45. What's Goodhart's Law and how does it apply to ML?**
"When a measure becomes a target, it ceases to be a good measure." Once you optimize for a proxy metric, the proxy stops measuring what you wanted. Examples: optimize CTR → clickbait; optimize BLEU → translation that mimics surface but not meaning; optimize PPL → memorization not understanding.

**46. What's distribution shift and how does it affect metrics?**
Production data differs from training/eval data. Eval metrics on training distribution overstate deployment performance. Mitigations: test on held-out time period, on different user segments, monitor production metrics, recalibrate.

**47. Why stratify your evaluation?**
Average metrics hide bad behavior on slices. A 90% accurate model might fail on a specific demographic. Stratify per-language, region, segment. "Average" can be misleading; tail behavior matters.

---

## H. Quick fire

**48.** *Best metric for ranking on imbalanced data?* PR-AUC.
**49.** *Best metric for calibration?* Brier or ECE.
**50.** *Default LLM eval metric?* PPL on held-out data, plus task-specific (HumanEval pass@1, AlpacaEval win-rate, MMLU accuracy, etc.).

---

## Self-grading

If you can't answer 1-15, you don't know basic metrics. If you can't answer 16-35, you'll fall short on serious ML interviews. If you can't answer 36-50, you'll struggle with frontier-lab depth.

Aim for 35+/50 cold.
