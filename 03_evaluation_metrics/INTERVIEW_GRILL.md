# Evaluation Metrics — Interview Grill

> 50 questions on evaluation metrics. Drill until you can answer 35+ cold.

---

## A. Classification basics

**1. Define accuracy, precision, recall, F1.**
$\text{Accuracy} = (TP + TN)/(TP + TN + FP + FN)$. $\text{Precision} = TP/(TP + FP)$. $\text{Recall} = TP/(TP + FN)$. $F_1 = 2PR/(P + R)$ — harmonic mean of $P$ and $R$.

> **Saying it out loud.** Accuracy is how often you're right overall. Precision is: of the things I flagged, how many were real — so it's about the cost of crying wolf. Recall is: of the real things out there, how many did I catch — so it's about the cost of missing. F1 is the single number that forces you to be decent at both, and the way I keep them straight is that precision's denominator is what *I* predicted, recall's denominator is what actually *exists*.

**2. Why is accuracy bad on imbalanced data?**
Predicting majority class trivially gets $1 - \text{minority-fraction}$ accuracy. With 99:1 imbalance, predicting all-majority gets 99% accuracy without learning anything. Model could be useless yet "high accuracy."

> **Saying it out loud.** Because you can score 99% by refusing to do your job. If one percent of transactions are fraud, a model that says 'never fraud' for everything is 99% accurate and catches zero fraudsters — the metric is measuring the class balance, not the model. The tell is that accuracy has a floor that moves with prevalence, so a number that sounds impressive can be strictly worse than useless. On imbalanced problems I'd quote precision and recall, or PR-AUC, and treat accuracy as a distraction.

**3. When is precision the right metric?**
When false positives hurt: spam (legitimate email blocked), recommendations (showing bad items burns trust), content moderation (false flags = censorship complaints), ads (FP = wasted budget).

> **Saying it out loud.** Precision is what you want when a false alarm is expensive. Spam filtering is the classic case — sending one real job offer to the junk folder costs you far more than letting a few spam emails through. Same story for content moderation, where false flags turn into censorship complaints, and for anything where a human has to act on every alert, because a low-precision system just trains people to ignore it. The rule of thumb: if the cost of being wrong lands on someone who did nothing wrong, optimize precision.

**4. When is recall the right metric?**
When false negatives hurt: medical screening (missed cancer), fraud detection (let bad guy through), search recall (missed relevant docs), safety filters (missed harmful content).

> **Saying it out loud.** Recall is what you want when missing something is the disaster. Cancer screening is the obvious one — a false positive means an uncomfortable follow-up test, a false negative means someone dies. Fraud, security scanning, and safety filters have the same shape: the thing you missed goes on to cause the damage. Usually you set a recall floor — 'we must catch 99% of these' — and then get the best precision you can subject to that, rather than optimizing recall alone.

**5. Why is F1 the harmonic mean and not arithmetic?**
Harmonic mean penalizes imbalance: $F_1$ stays low if either $P$ or $R$ is near 0, even if the other is 1. Arithmetic mean would give 0.5 for ($P=1, R=0$), masking the failure.

> **Saying it out loud.** Because the harmonic mean refuses to be fooled by one good number. Say you flag every single transaction as fraud: recall is 1.0, precision is basically 0, and the arithmetic mean would happily report 0.5 — a passing grade for a model that does nothing. The harmonic mean gives you 0 there, because it's dominated by the smaller of the two. That's the property you want: F1 is only high when *both* are high.

**6. What's F-beta?**
$F_\beta = (1 + \beta^2) \cdot PR / (\beta^2 \cdot P + R)$. $\beta > 1$ weights recall ($\beta=2$ = "recall is twice as important"). $\beta < 1$ weights precision. $F_1 = F_{\beta=1}$.

> **Saying it out loud.** F-beta is F1 with a dial for which mistake you care about. Beta is how many times more you value recall than precision — beta of 2 means recall counts double, beta of 0.5 means precision does, and beta of 1 gives you plain F1. The useful part in an interview is that it forces you to state your business tradeoff explicitly instead of defaulting. For medical screening I'd argue for F2; for a spam filter, F0.5.

**7. Macro vs micro vs weighted average for multi-class?**
Macro: average per-class metrics equally — penalizes poor performance on rare classes. Micro: aggregate TP/FP/FN across classes then compute — dominated by majority. Weighted: macro weighted by class frequency.

> **Saying it out loud.** They differ in who gets a vote. Macro computes the metric per class and averages them flat, so a class with ten examples counts as much as one with ten thousand. Micro pools all the true positives and false positives together first, which means the big classes dominate and it ends up close to accuracy. Weighted is macro but with class frequency as the weight. Pick macro if rare classes matter to you, and say so — that choice is the actual answer to the question.

**8. Why might macro F1 differ from micro F1?**
On imbalanced multi-class. Macro treats rare classes as equally important; micro is dominated by frequent classes. Macro $F_1 \ll$ micro $F_1$ means rare classes are being missed.

> **Saying it out loud.** A big gap between them is a diagnostic, not a nuisance. If micro F1 is 0.92 and macro is 0.55, that's the model doing great on your common classes and falling over on the rare ones — micro is being carried by volume. So the moment I see that split, I go look at the per-class breakdown, because there's almost always a handful of classes at near-zero recall. Reporting only micro on an imbalanced problem is how you ship a model that silently ignores half your label set.

---

## B. AUROC and PR-AUC

**9. What does AUROC measure?**
Probability that the model ranks a random positive higher than a random negative. Threshold-independent ranking quality. AUROC = 0.5: random; AUROC = 1: perfect ranking.

> **Saying it out loud.** AUROC is a ranking score, not an accuracy score. The clean interpretation: pick a random positive and a random negative — AUROC is the probability the model gives the positive the higher score. That's why it doesn't depend on where you set your threshold, and why 0.5 means you're guessing. It answers 'does the model order things correctly', which is a different and often more useful question than 'is the model right'.

**10. How is AUROC computed?**
Plot TPR ($=$ recall) vs FPR ($= FP/(FP+TN)$) as you sweep the classification threshold. Area under that curve. Equivalently: pairwise ranking probability.

> **Saying it out loud.** You sweep the threshold from one end to the other, and at each setting you plot true positive rate against false positive rate — then take the area under that curve. At a very high threshold you predict almost nothing positive so you're at the origin; at a very low one you predict everything positive and you're at the top right. The area is equivalent to that pairwise ranking probability, which is usually the faster way to compute it. And that equivalence is exactly why AUROC is unaffected by any monotone rescaling of your scores.

**11. When does AUROC mislead?**
On heavily imbalanced data. The (very large) negative count keeps FPR low even with many FPs. AUROC can stay high while precision is terrible.

> **Saying it out loud.** AUROC lies to you when negatives massively outnumber positives. False positive rate divides by the total number of negatives, so if you have a million negatives, ten thousand false alarms barely move the FPR — the curve still looks beautiful. But those ten thousand false alarms are what a human actually has to review. So on a 1% prevalence problem you can post an AUROC of 0.97 with precision under 10%; report PR-AUC alongside it or you're hiding the problem.

**12. AUROC vs PR-AUC?**
AUROC: TPR vs FPR. PR-AUC: precision vs recall. AUROC bounded below at 0.5 by random; PR-AUC bounded below at class prevalence by random. PR-AUC more honest under imbalance.

> **Saying it out loud.** They're the same sweep, different axes. AUROC plots true positive rate against false positive rate; PR-AUC plots precision against recall, and precision never divides by the huge negative count — so it feels every false positive. The other big difference is the baseline: random guessing always gives AUROC 0.5, but random guessing gives a PR-AUC equal to your positive rate, which might be 0.01. That's why PR-AUC is the more honest number under imbalance, and why you always report prevalence next to it.

**13. When should you report PR-AUC?**
Imbalanced classification where you care about precision at high recall. Example: fraud detection with 1% fraud rate — AUROC of 0.95 sounds great but PR-AUC of 0.3 reveals the truth.

> **Saying it out loud.** Whenever positives are rare and you care about the quality of your alerts. Fraud at a 1% base rate is the canonical example: AUROC 0.95 sounds like a shipped product, but PR-AUC of 0.3 tells you that a reviewer working your top alerts will be wrong most of the time. The general rule is that if the cost is dominated by what happens to the flagged set, use PR-AUC. And always report the prevalence, because PR-AUC is only interpretable relative to that baseline.

**14. What's the relationship between AUROC and the Mann-Whitney U test?**
They're equivalent. $\text{AUROC} = U / (n_{\text{pos}} \cdot n_{\text{neg}})$ for the rank-sum statistic $U$. Both measure: how often does a positive rank above a negative.

> **Saying it out loud.** They're literally the same statistic wearing different clothes. Mann-Whitney U counts how many positive-negative pairs the positive wins; divide by the total number of pairs and you have AUROC. So an AUROC of 0.8 means the positive outranks the negative in 80% of pairs. The practical payoff is that you get a well-studied non-parametric significance test for free — you can put a p-value on 'is model A's ranking really better than B's' rather than eyeballing curves.

---

## C. Calibration

**15. What does calibration mean?**
Predicted probabilities match observed frequencies. If the model says "70%" and the event happens 70% of the time on those predictions, it's calibrated. Independent from accuracy or AUROC.

> **Saying it out loud.** Calibration is whether the model's confidence means anything. If it says 70% on a thousand cases, roughly 700 should actually happen — otherwise the number is just a score, not a probability. The key thing people miss is that this is completely separate from accuracy or AUROC: you can have perfect ranking and terrible calibration, because ranking is invariant to any monotone squashing of the scores. It matters the moment a downstream decision does arithmetic on the probability — expected value, thresholding by cost, feeding a risk model.

**16. How do you measure calibration?**
Reliability diagram (bin predictions, plot mean predicted vs observed frequency; should be y=x). Brier score (MSE between p and y). ECE (weighted average distance between bin frequency and bin mean prediction).

> **Saying it out loud.** Three tools, increasing in convenience. A reliability diagram is the visual one: bin the predictions, plot mean predicted probability against observed frequency, and a calibrated model traces the diagonal. Brier score is just mean squared error on probabilities, so it's a single number that mixes calibration and sharpness. ECE is the one people quote — the bin-size-weighted average gap between predicted and observed — and the gotcha to name is that ECE is sensitive to how many bins you choose, so it's easy to game.

**17. Decompose Brier score.**
Brier = calibration + refinement (− uncertainty). Calibration = how far bin predictions are from bin frequencies. Refinement = how informative the bins are. Lower is better for both.

> **Saying it out loud.** Brier splits into a calibration piece and a refinement piece, and both matter. Calibration is the gap between what you said and what happened within each bin — being honest. Refinement is about being *useful*: a weather model that predicts the historical base rate every day is perfectly calibrated and completely uninformative. The value of the decomposition is that it separates 'my probabilities are dishonest' from 'my probabilities are useless', and those need completely different fixes.

**18. How do you fix miscalibration?**
Platt scaling ($\sigma(a \cdot \text{score} + b)$ fit on val), isotonic regression (non-parametric monotonic), temperature scaling ($\text{logits}/T$ for softmax). Temperature is cheapest, fits one parameter, often sufficient for NN softmax.

> **Saying it out loud.** Fit a small correction on held-out data — never on the training set. Temperature scaling is the one to reach for first with a neural net: you learn a single scalar that divides the logits, and it's cheap, can't reorder anything, and usually gets you most of the way. Platt scaling fits a logistic on the score, and isotonic regression is non-parametric so it's more flexible but needs a lot more validation data or it overfits. The tradeoff to name is that temperature preserves your ranking exactly, so AUROC is untouched, while isotonic can and will change it.

**19. Why are deep neural networks miscalibrated?**
Overconfident due to high capacity: NN drives training cross-entropy near 0 by pushing logits to extremes, even when validation accuracy plateaus. Probabilities concentrate at 0/1 even when the model should be uncertain.

> **Saying it out loud.** Because modern nets have enough capacity to keep driving the loss down long after they've stopped getting more right. Cross-entropy always rewards pushing a correct prediction from 0.9 toward 0.99, so training keeps inflating logits even when validation accuracy has flattened — the model learns to be certain rather than to be better. The result is probabilities piled up at 0 and 1, so a model at 80% accuracy reports 99% confidence. Guo et al. 2017 also showed this got *worse* as networks got deeper and wider, and temperature scaling on a validation set fixes most of it for one parameter.

**20. What's log loss?**
$-(1/N) \sum [y \log p + (1-y) \log(1-p)]$. Same as binary cross-entropy. Calibration-aware: penalizes overconfident wrong predictions much more than just-wrong predictions. Aligned with MLE.

> **Saying it out loud.** Log loss is what you get when you score a probability by how surprised you were. It's the negative log of the probability you assigned to the thing that actually happened, averaged over the data — the same as binary cross-entropy, and the same thing maximum likelihood is minimizing. The reason it's worth having alongside accuracy is that it punishes confident mistakes brutally: being 99% sure and wrong costs you far more than being 60% sure and wrong. That's exactly the behavior you want when downstream decisions trust the number.

---

## D. Regression

**21. MSE vs MAE — when which?**
MSE: when large errors should hurt much more (variance critical). Sensitive to outliers. MAE: robust to outliers, predicts the median. Choose by what error distribution matters for your task.

> **Saying it out loud.** Squaring makes big errors dominate; absolute value makes them just count. So MSE is what you want when one huge miss is genuinely worse than several small ones — think inventory that either stocks out or doesn't — and MAE is what you want when your data has outliers you don't want steering the fit. There's a neat theoretical hook worth dropping: minimizing MSE predicts the conditional mean, minimizing MAE predicts the conditional median. So the choice isn't just about robustness, it's about which summary of the distribution you actually want.

**22. What does RMSE tell you that MSE doesn't?**
Same units as $y$. $\text{RMSE} = \sqrt{\text{MSE}}$. Easier to interpret in domain terms. Otherwise mathematically equivalent.

> **Saying it out loud.** RMSE is in the same units as the thing you're predicting, and that's the whole point. Saying 'MSE is 2,500 dollars-squared' means nothing to a stakeholder; saying 'we're off by about 50 dollars typically' lands immediately. Mathematically they rank models identically since the square root is monotone, so it changes nothing about optimization. It just changes whether anyone in the room understands you.

**23. What does $R^2$ of $-0.2$ mean?**
Model is worse than predicting the mean. $R^2 < 0$ happens; it means the model has *negative* explanatory power. Common bug source — should investigate immediately.

> **Saying it out loud.** Negative $R^2$ means your model is worse than a horizontal line. $R^2$ compares your errors to the errors of just predicting the mean every time, so anything below zero says you'd be better off never having built the model. In practice that's almost never a subtle modeling issue — it's a bug: a train-test mismatch, an unscaled target, a shifted index, or evaluating on a different distribution than you fit on. So the answer is 'stop tuning and go find the bug'.

**24. Why is MAPE problematic?**
Undefined at $y = 0$. Asymmetric (under-predicting capped at 100%; over-predicting unbounded). Misleading for small $y$. Use SMAPE or MASE instead.

> **Saying it out loud.** MAPE breaks exactly where you need it most. It divides by the actual value, so it's undefined at zero and explodes for small values — one item with true demand of 1 and a prediction of 3 contributes 200% and swamps everything else. It's also asymmetric: over-predicting can cost you an unbounded percentage while under-predicting caps out at 100%, so optimizing MAPE quietly biases your forecasts low. If you need a scale-free error, use MASE, which normalizes by a naive baseline and doesn't blow up.

**25. What's quantile loss?**
$\mathcal{L}_\tau = \sum \max(\tau \cdot (y - \hat y), (\tau - 1) \cdot (y - \hat y))$. For $\tau = 0.5$, recovers MAE (median). For $\tau = 0.9$, optimizes 90th percentile. Useful for uncertainty quantification, conformal prediction, demand forecasting with safety stock.

> **Saying it out loud.** Quantile loss is asymmetric on purpose — it charges you a different price for being too high than for being too low. At tau equal to 0.5 the penalties are equal and you recover MAE, so you're predicting the median; push tau to 0.9 and under-prediction costs nine times as much, so the model learns to sit at the 90th percentile. That's how you get prediction intervals out of a point-prediction model — fit tau at 0.05 and 0.95 and you have a band. It's the right tool for demand forecasting where a stock-out costs more than a little extra inventory.

---

## E. Ranking and IR

**26. What's MAP?**
Mean Average Precision. For each query, AP = average of precision at each relevant document's rank. Then average across queries. Position-aware: missing top-rank relevant docs hurts more.

> **Saying it out loud.** MAP asks 'how high up did you put the relevant stuff, on average'. For one query you walk down the ranked list and every time you hit a relevant document you record the precision at that point, then average those — so a relevant doc at rank 2 contributes much more than one at rank 50. Then you average across queries. It's binary relevance only, which is its main limitation, and it's the standard choice when a query has several correct answers.

**27. What's NDCG?**
Normalized Discounted Cumulative Gain. $\text{DCG} = \sum (2^{\text{rel}_i} - 1) / \log_2(i + 1)$. Normalized by ideal DCG. Position-discounted, handles graded relevance. Standard in search ranking.

> **Saying it out loud.** NDCG is the ranking metric that handles 'some results are more relevant than others'. Each document contributes a gain based on its relevance grade, discounted by the log of its position, so being at rank one is worth much more than rank ten. Then you divide by the score of the perfect ordering, which is what makes it comparable across queries with different numbers of relevant docs. It's the industry default in search and recommendations, and the thing to be careful about is that the exponential gain formula makes it very sensitive to how you assign relevance grades.

**28. What's MRR?**
Mean Reciprocal Rank. $\text{RR} = 1/\text{rank of first correct}$. Hard penalty for not having the answer at rank 1. For tasks with one right answer (factoid Q&A).

> **Saying it out loud.** MRR only cares about where your first correct answer landed — one over that rank. Rank one gives you 1.0, rank two gives 0.5, rank ten gives 0.1. That drop-off is brutal and deliberate: it's the right metric when the user is going to look at one answer and leave, like factoid Q&A or a voice assistant. The flip side, and the thing to name as a limitation, is that it ignores everything after the first hit, so it's the wrong choice when the user wants comprehensive results.

**29. Precision@k vs Recall@k — when to choose?**
Precision@k when you only show top-k (e.g., 10 search results) and care about quality of those k. Recall@k when you care about coverage at fixed k.

> **Saying it out loud.** It depends on whether the k is a display constraint or a budget. Precision@10 is the question 'of the ten results on the page, how many are good' — that's what the user experiences in search. Recall@k is the question 'did my candidate set capture the things worth having', which is what you measure for the retrieval stage of a pipeline, because a reranker downstream can fix ordering but can never recover something that wasn't retrieved. So in a two-stage system I'd optimize recall@100 for the retriever and precision@10 for the final ranking.

---

## F. LLM-specific

**30. Define perplexity.**
`PPL = exp(−(1/N) Σ log P(x_i | x_{<i}))`. Geometric inverse of average per-token probability. Bounded below by $\exp(H_{\text{true}})$ — equals 1 only for deterministic data; for natural language the floor is strictly above 1. Bounded above by vocab size (uniform random model = `|V|`).

> **Saying it out loud.** Perplexity is how many options the model was effectively choosing between at each token. If the perplexity is 20, it's about as confused as if it were picking uniformly from 20 words — so lower is better, and it's just the exponential of the average cross-entropy. The floor is not 1 for real language, it's the entropy of the language itself, because the next word is genuinely uncertain. The ceiling is the vocabulary size, which is what a model that has learned nothing scores.

**31. Why can't you compare PPL across models with different tokenizers?**
PPL is per-token. Different tokenizers split text into different numbers of tokens. A model with finer tokenization gets lower PPL on the same text purely because it's predicting more tokens. **Compare per-byte or per-character likelihood instead** for cross-tokenizer comparison.

> **Saying it out loud.** Because perplexity is measured per token, and tokenizers disagree about what a token is. If model A splits a sentence into 100 tokens and model B splits it into 60, model A gets to make easier, more numerous predictions — and its per-token perplexity looks better without it understanding anything more. So the comparison is measuring tokenization, not modeling. The fix is to normalize by something physical: report bits per byte or per character, which is exactly what benchmarks like the Pile use for cross-model comparison.

**32. What's pass@k?**
`Pass@k = E[1 − C(n−c, k)/C(n, k)]` where `n` = samples generated, `c` = pass count. Probability that at least one of `k` independent samples solves the problem. Standard for code generation.

> **Saying it out loud.** Pass@k asks: if we let the model try k times, what's the chance at least one attempt works. The naive way — generate k samples and check — is extremely high-variance, so the standard estimator generates a larger $n$ samples, counts how many pass, and computes the probability combinatorially. It's the default for code generation because code has an objective checker: you run the unit tests. And the k you report is a claim about the deployment setting, not a free parameter.

**33. Why pass@1 vs pass@10 vs pass@100?**
Pass@1: model's first answer; mimics typical user. Pass@10/100: best-of-N capability; mimics repeated retry workflows. The gap between pass@1 and pass@10 measures how many right answers the model has but doesn't surface first.

> **Saying it out loud.** Pass@1 is what a user actually experiences; pass@100 is what the model is capable of. The gap between them is the interesting number — it tells you the model can solve the problem but can't tell which of its answers is right, which is a ranking failure rather than a knowledge failure. That's actionable: it means reranking, self-consistency, or a verifier will buy you a lot, whereas if pass@100 is also low you need a better model. Watch out for papers quoting pass@100 as if it were a product capability — it isn't, unless you have a checker at inference time.

**34. What's BLEU?**
Bilingual Evaluation Understudy. n-gram overlap between candidate and reference translations: $\text{BLEU} = \text{BP} \cdot \exp(\sum_n w_n \log p_n)$ where $w_n = 1/N$ uniformly (so weights sum to 1). Brevity penalty $\text{BP}$ discourages too-short outputs.

> **Saying it out loud.** BLEU counts how many short word sequences your translation shares with a human reference. You compute overlap for unigrams through 4-grams, take a geometric mean so you have to do reasonably well at all of them, and multiply by a brevity penalty so the model can't cheat by outputting three safe words. It's clipped so repeating 'the' twenty times doesn't help. It's crude, but it correlated well enough with human judgment in 2002 to become the standard, and the field has been trying to leave it ever since.

**35. BLEU's failure modes?**
Multiple valid translations; n-gram overlap misses paraphrases; surface-level (no semantics). COMET, BLEURT, GEMBA-MQM increasingly replace BLEU for serious MT eval.

> **Saying it out loud.** BLEU can't tell a good paraphrase from a bad translation, because it only sees surface n-grams. Say the same thing with different words and you get punished; produce fluent nonsense that reuses reference phrasing and you get rewarded. It also only compares against a handful of references when there are dozens of valid translations. That's why serious MT eval has moved to learned metrics like COMET and BLEURT, which score in embedding space and correlate far better with humans — BLEU survives mostly as a cheap regression check.

**36. ROUGE — what and where?**
Recall-oriented n-gram overlap (ROUGE-N) or longest common subsequence (ROUGE-L). For summarization. Same surface-level limitations as BLEU.

> **Saying it out loud.** ROUGE is BLEU pointed the other way: recall-focused, for summarization. ROUGE-N counts what fraction of the reference's n-grams your summary managed to include, and ROUGE-L uses the longest common subsequence so word order matters loosely without requiring exact contiguity. The recall orientation makes sense because a summary's failure mode is leaving things out. It inherits every one of BLEU's problems though, and it has a nastier one: it rewards long summaries that copy source sentences verbatim, which is often exactly the thing you don't want.

**37. LLM-as-judge biases?**
Length (judges prefer longer outputs), style (formal/markdown formatting boosts ratings), sycophancy (prefers responses agreeing with the judge), self-similarity (prefers outputs from same model family). Mitigations: ensemble, length control, blinded comparison.

> **Saying it out loud.** LLM judges have consistent, measurable biases — treat them as a noisy rater, not an oracle. They prefer longer answers, they reward markdown and confident formatting over correctness, they're sycophantic toward whatever position was in the prompt, and they favor outputs from their own family. There's also strong position bias in pairwise comparisons, which is why you always run both orderings and average. Mitigations are straightforward — swap positions, control for length, use a rubric with explicit criteria, ensemble judges — but you should validate against human labels on a subset before you trust the number.

**38. What's win-rate vs Elo for LLM eval?**
Win-rate: fraction of pairwise comparisons where model A beats B. Elo: dynamic rating from many pairwise comparisons (chess-style). Used in LMSYS Chatbot Arena. Both pairwise but Elo is multi-model.

> **Saying it out loud.** Win-rate is a head-to-head number against one specific opponent; Elo turns lots of head-to-heads into a single ranking. The problem with win-rate is that it's only meaningful relative to the baseline you chose, so '70% win rate' means nothing without naming who you beat. Elo, the chess system, solves that by fitting a latent skill number to all pairwise outcomes at once, which is what Chatbot Arena does. The caveat worth naming is that Elo assumes a consistent transitive skill ordering, and models are genuinely non-transitive across different task types.

---

## G. Methodology and pitfalls

**39. Why do you need separate train/val/test?**
Train: fit parameters. Val: tune hyperparameters and early-stop. Test: estimate deployment performance (used once, never tuned against). Reusing val for test inflates estimates.

> **Saying it out loud.** Three sets because there are three distinct things you'd otherwise fool yourself about. Train fits parameters, validation picks hyperparameters and tells you when to stop, and test is the one number you only look at at the very end. The moment you tune against the test set even a little — try five architectures, pick the best test score — that number stops being an estimate of deployment performance and becomes an optimistic bias. In practice the discipline that matters is touching test once, and if you have to touch it repeatedly, holding out a fresh set.

**40. What's data leakage and how do you detect it?**
A test-set feature or label is influenced by training data. Detect by: too-good-to-be-true performance, feature importance dominated by suspicious features (timestamps, IDs), random shuffling boosting metric absurdly.

> **Saying it out loud.** Leakage is when information that won't exist at prediction time sneaks into training. The classic cases are subtle: a feature computed after the outcome, normalizing over the full dataset before splitting, or random-shuffling a time series so tomorrow trains on data from next week. The tell is performance that's too good — 0.99 AUC on a hard problem should make you suspicious, not happy. The way I hunt it is to look at feature importance for anything with an ID or a timestamp in it, and to ask of every feature 'would I actually have this at inference?'

**41. Time-series cross-validation?**
Forward-chaining: train on `[1..t]`, test on `[t+1..t+h]`. Never train on future and test on past. Standard k-fold leaks future into past.

> **Saying it out loud.** With time series you have to respect the arrow of time. Forward-chaining means you train on everything up to time $t$ and test on the window right after, then roll forward and repeat — so you're always predicting the future from the past, which is what deployment looks like. Standard k-fold shuffles, which means some of your training rows come from after your test rows, and the model gets to see the future. That inflates your numbers dramatically and the model then falls over in production, which is one of the most common ways real forecasting projects fail.

**42. Stratified k-fold?**
For imbalanced classification: ensure each fold has the same class distribution as full data. Default in sklearn for classification.

> **Saying it out loud.** Stratified k-fold just makes sure every fold has the same class proportions as the full dataset. It matters when the positive class is rare — with 1% positives and five folds, plain random splitting can easily give you a fold with almost none, so your per-fold metrics bounce around wildly and the mean is noisy. Stratifying removes that variance for free. It's sklearn's default for classifiers, which is why a lot of people use it without knowing they do.

**43. How do you compute confidence intervals on a metric?**
Bootstrap resampling: B bootstrap samples; metric on each; 2.5–97.5 percentile gives 95% CI. Or analytically (delta method) for simple metrics. Always report CIs for serious comparisons.

> **Saying it out loud.** Bootstrap is the honest default. You resample your test set with replacement a thousand times, recompute the metric on each resample, and take the 2.5th and 97.5th percentiles — no distributional assumptions, works for weird metrics like AUROC or NDCG where the analytic formula is painful. The reason it matters is that a lot of reported wins are inside the noise: with a 500-example test set, a two-point accuracy difference usually isn't real. If you're comparing two models on the same data, bootstrap the *difference* rather than each metric separately — that accounts for the correlation and gives you a much tighter interval.

**44. Multiple comparison correction?**
If you evaluate 100 configurations, some will look "significantly" better by random noise. Bonferroni (divide α by number of tests) is conservative. False Discovery Rate (Benjamini-Hochberg) is less conservative, more practical.

> **Saying it out loud.** If you test enough things, something will look great by luck. Run a hundred configurations at a 5% significance level and you expect about five false winners even if every model is identical. Bonferroni fixes it by dividing your threshold by the number of tests, which is correct but so conservative that you'll miss real effects. Benjamini-Hochberg is what people actually use — it controls the fraction of your 'discoveries' that are false rather than eliminating them entirely, which is the more sensible tradeoff when you're screening lots of candidates.

**45. What's Goodhart's Law and how does it apply to ML?**
"When a measure becomes a target, it ceases to be a good measure." Once you optimize for a proxy metric, the proxy stops measuring what you wanted. Examples: optimize CTR → clickbait; optimize BLEU → translation that mimics surface but not meaning; optimize PPL → memorization not understanding.

> **Saying it out loud.** Goodhart's law is 'when a measure becomes a target, it stops being a good measure', and ML is a machine for demonstrating it. The reason is that gradient descent is an extremely literal optimizer — it finds whatever exploits the proxy, and the gap between your proxy and your actual goal is where it goes. Optimize click-through and you get clickbait; optimize BLEU and you get translations that copy reference phrasing without meaning; optimize a reward model too hard and you get reward hacking. The practical defense is holding out a metric you never optimize against, and keeping a human eval in the loop as the ground truth.

**46. What's distribution shift and how does it affect metrics?**
Production data differs from training/eval data. Eval metrics on training distribution overstate deployment performance. Mitigations: test on held-out time period, on different user segments, monitor production metrics, recalibrate.

> **Saying it out loud.** Distribution shift means production data doesn't look like your eval data, so your eval number is a promise you can't keep. It shows up as covariate shift — the inputs change, like a new user demographic — or label shift, where the base rate moves, which quietly wrecks your calibrated thresholds. The standard trap is a random train-test split on time-ordered data: it tells you you're great, and then you deploy into next quarter. Evaluate on a held-out *future* time period, monitor input distributions in production, and plan to recalibrate on a schedule.

**47. Why stratify your evaluation?**
Average metrics hide bad behavior on slices. A 90% accurate model might fail on a specific demographic. Stratify per-language, region, segment. "Average" can be misleading; tail behavior matters.

> **Saying it out loud.** Because an average is a place for failures to hide. A model at 92% overall can be at 60% for one language, one age group, or one device type, and the aggregate will never show you — the bad slice is too small to move the mean. That's how fairness incidents and embarrassing launches happen, and it's also just bad engineering, since your worst slice is often your fastest-growing segment. So I'd report per-slice metrics with the worst slice called out explicitly, not just the headline number.

---

## H. Quick fire

**48.** *Best metric for ranking on imbalanced data?* PR-AUC.
**49.** *Best metric for calibration?* Brier or ECE.
**50.** *Default LLM eval metric?* PPL on held-out data, plus task-specific (HumanEval pass@1, AlpacaEval win-rate, MMLU accuracy, etc.).

---

## Self-grading

If you can't answer 1-15, you don't know basic metrics. If you can't answer 16-35, you'll fall short on serious ML interviews. If you can't answer 36-50, you'll struggle with frontier-lab depth.

Aim for 35+/50 cold.
