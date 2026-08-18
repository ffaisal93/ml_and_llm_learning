# Generalization and Evaluation — Interview Grill

> 50 questions on data leakage, calibration, distribution shift, class imbalance, ablations. Drill until you can answer 35+ cold.

---

## A. Data leakage

**1. Define data leakage.**
Information from outside the training set leaking in, inflating offline metrics that don't transfer.

> **Saying it out loud.** Leakage is when your model gets access during training to information it won't have at prediction time. The tell is always the same: the offline numbers look fantastic and production is a disaster. The useful test to describe is temporal — for every feature, ask whether its value would actually have been known at the moment you need the prediction. If the answer is no, or you're not sure, you have leakage. It's the single most common reason a promising model dies in deployment.

**2. Four types of leakage?**
Target leakage, train-test contamination, preprocessing leakage, temporal leakage. (Group leakage is a fifth special case.)

> **Saying it out loud.** Target leakage is a feature that's partly caused by the outcome. Train-test contamination is the same rows or near-duplicates appearing on both sides of the split. Preprocessing leakage is fitting your scaler or imputer on the full dataset before splitting. Temporal leakage is training on the future to predict the past. And group leakage is the sneaky one — the same user or patient appearing in both splits, so the model recognizes the entity rather than learning the pattern. Getting all five out shows you've actually debugged a pipeline.

**3. Example of target leakage in churn prediction?**
Using "days since last login" as a feature when it's computed *after* churn. The future-looking feature trivially predicts churn.

> **Saying it out loud.** The classic is a feature like days since last login, computed at analysis time rather than at prediction time. Someone who churned two months ago obviously hasn't logged in recently, so the feature basically encodes the answer — your AUC goes to 0.99 and you learn nothing. The general shape is a feature that's a consequence of the label rather than a cause. Anything derived from a support ticket, a cancellation flow, or a refund record has the same problem.

**4. Why is random k-fold wrong for time series?**
Allows training on the future, predicting the past. Use time-series split (sliding/expanding window) instead.

> **Saying it out loud.** Because a random split lets you train on Thursday and predict Tuesday, which is a situation that will never occur in production. The model gets to see the future, so it can pick up on trends and seasonality it couldn't possibly know, and your estimate comes out far too optimistic. Use an expanding or sliding window where every test fold sits strictly after its training data. The gap between random and time-based CV is often enormous — it's routine for a model to look excellent under random k-fold and be worthless under a proper temporal split.

**5. Standardize features using overall mean — leakage?**
Yes. Test data influences train preprocessing. Compute mean/std on train only.

> **Saying it out loud.** Yes, and it's the most common version because it feels harmless. If you compute the mean and standard deviation over the full dataset before splitting, information about the test set is baked into the training transformation. The effect is usually small, but it's real, and it compounds badly with anything more aggressive like target encoding or imputation. The fix is discipline rather than cleverness: fit every transformation on the training fold and apply it to the others, which is exactly what a scikit-learn Pipeline inside cross-validation enforces for you.

**6. Same user in train and test — leakage?**
If predicting user-level outcomes and the model learns user-specific patterns: yes. Use group k-fold.

> **Saying it out loud.** It depends on what you're predicting, and the right answer is to name the condition. If you're forecasting user-level outcomes and the model can learn user-specific idiosyncrasies, then having the same user on both sides means you're measuring memorization, not generalization. Group k-fold, which keeps every entity entirely within one fold, is the fix. It matters most in medical and recommendation settings, where one patient or user might contribute hundreds of rows and the model can effectively look them up.

**7. How do you detect leakage post-hoc?**
Sky-high offline metrics that don't translate to deployment. Check feature importance for "too good to be true" features. Audit: could each feature have been computed at prediction time?

> **Saying it out loud.** The first alarm is a result that's too good — an AUC of 0.99 on a problem where 0.80 would be impressive means you should go looking, not celebrate. Then check feature importance, because leakage usually concentrates in one or two dominant features. The definitive audit is the temporal one: walk each feature and ask whether it could have been computed at prediction time, using the actual data pipeline rather than the schema. And if you can, run the model against a genuinely held-out future period — production truth is the only test that can't be gamed.

---

## B. Calibration

**8. What does calibrated mean?**
Predictions match observed frequencies: among predictions of 0.7, ~70% are positive.

> **Saying it out loud.** Calibrated means the numbers mean what they say. Take everything the model scored around 0.7 and roughly 70 percent of them should actually be positive. It's a different property from being accurate — a model can rank perfectly and still be systematically overconfident. It matters whenever a downstream decision uses the probability rather than just the ordering, which is anything involving expected cost: fraud thresholds, medical triage, bidding. If you're only sorting a list, you can ignore calibration entirely.

**9. How do you measure calibration?**
Reliability diagram (bin predictions; plot empirical positive rate vs predicted score). ECE: average gap between confidence and accuracy across bins.

> **Saying it out loud.** The picture is a reliability diagram: bin the predictions, plot the average predicted probability against the actual positive rate in each bin, and see how far you are from the diagonal. Below the line means overconfident, which is where nearly all neural networks live. The single number is expected calibration error, the bin-size-weighted average gap. Worth knowing its weakness — ECE depends on how you bin, and with 10 bins you can hide a lot, so people use it as a rough signal and look at the diagram for the real story.

**10. ECE formula?**
$\mathrm{ECE} = \sum_b \frac{|B_b|}{N} |\mathrm{acc}(B_b) - \mathrm{conf}(B_b)|$.

> **Saying it out loud.** It's a weighted average of the gaps: for each bin, take the absolute difference between the average confidence and the actual accuracy, and weight by how many samples fall in that bin. So it's just the mean vertical distance from the diagonal on the reliability diagram. The number to have in your pocket is that modern deep networks routinely show ECE around 0.1 or worse before calibration — meaning confidence is off by ten percentage points on average — and temperature scaling drops that to roughly 0.01.

**11. Brier score?**
$\frac{1}{N}\sum (\hat{p} - y)^2$. Combined calibration + resolution.

> **Saying it out loud.** Brier is just mean squared error on probabilities: subtract the label from your predicted probability, square, average. What makes it useful is that it's a proper scoring rule, so it's minimized exactly when you report your true beliefs — you can't game it by shading your predictions toward the extremes. It bundles calibration and discrimination into one number, which is convenient for tracking and unhelpful for diagnosis, since a bad Brier score doesn't tell you which of the two is broken.

**12. Three calibration techniques?**
Platt scaling (logistic), isotonic regression (non-parametric monotonic), temperature scaling (single scalar on logits).

> **Saying it out loud.** Platt scaling fits a logistic regression on the model's scores — two parameters, works on small validation sets. Isotonic regression fits any monotonic mapping, which is more flexible and needs more data, maybe a thousand samples or it just overfits into a step function. Temperature scaling divides the logits by a single learned scalar, so it's one parameter and can't overfit. All three preserve ranking, so none of them change your AUC — they only move the probabilities.

**13. Why is temperature scaling popular for deep nets?**
Single scalar — minimal overfitting risk. Doesn't change ranking → AUC unchanged. Standard fix for overconfident neural networks.

> **Saying it out loud.** Because it's one parameter, so it essentially cannot overfit even on a small validation set, and it fixes the specific failure mode neural networks have. Modern networks are overconfident in a remarkably uniform way — the whole logit vector is scaled too aggressively — so one divisor corrects nearly all of it. Guo and colleagues showed it takes typical ECE from around 0.1 down to about 0.01, and it leaves the ranking untouched, so nothing else in your system changes. Best effort-to-payoff ratio in evaluation.

**14. Does temperature scaling change AUC?**
No. Monotonic transform of scores doesn't change pairwise ordering.

> **Saying it out loud.** No. Dividing every logit by the same positive number is a monotonic transformation, so the ordering of the scores is identical, and AUC depends only on the ordering. Same for Platt and isotonic, both monotonic. That's exactly why calibration is a free add-on — you can fix your probabilities after the fact without touching any ranking-based metric or any threshold you chose on rank. It's also a nice way to demonstrate you understand that AUC is a purely ordinal statistic.

**15. AUC vs calibration — different things?**
AUC measures ranking. Calibration measures whether scores are accurate probabilities. A model can have high AUC and bad calibration.

> **Saying it out loud.** Completely different, and the cleanest way to show it is with a thought experiment: take a perfectly calibrated model and halve every predicted probability. AUC is unchanged, because the ordering is identical, and calibration is now badly broken. AUC asks whether you can sort, calibration asks whether the numbers mean anything. Which one you need is set by the downstream use — ranking a feed needs AUC, deciding whether to block a transaction needs calibration, because you're comparing a probability against a cost threshold.

---

## C. Distribution shift

**16. Three types of distribution shift?**
Covariate shift ($p(x)$ changes), label shift ($p(y)$ changes), concept drift ($p(y|x)$ changes).

> **Saying it out loud.** Covariate shift is the inputs changing while the input-to-output relationship holds. Label shift is the outcome's base rate changing while the class-conditional inputs hold. Concept drift is the relationship itself changing, which is the genuinely hard one. The reason the distinction matters is that the first two are fixable by reweighting — you can correct for them without new labels — and the third can only be fixed by retraining on fresh labeled data. So diagnosing which one you have determines whether you need a quick patch or a whole new dataset.

**17. Covariate shift — typical example?**
New user demographics. Input distribution shifts; the underlying relationship is the same.

> **Saying it out loud.** A marketing push brings in a younger user base than the one you trained on. The inputs have moved, but the underlying mechanism — how a given user's behavior predicts churn — is unchanged. Your model isn't wrong, it's just being asked about a region of the input space it saw little of. The fix is importance weighting or targeted data collection, and if the new region is genuinely sparse in training, no reweighting saves you and you need new data.

**18. Label shift example?**
Disease prevalence increases during a pandemic; the conditional symptom-given-disease is unchanged.

> **Saying it out loud.** Disease prevalence jumps during an outbreak. What symptoms look like given the disease hasn't changed at all — only how common the disease is has. This one is elegantly fixable: if you can estimate the new base rate, you can adjust your predicted probabilities by the prior ratio without retraining anything. That's a genuinely useful thing to say, because it's one of the few distribution shifts with a clean analytic correction, and it applies whenever you rebalanced your training data too.

**19. Concept drift example?**
User preferences evolving over time — same input features, but the label given those features changes ($p(y|x)$ shifts).

> **Saying it out loud.** Fashion, spam, and fraud are the canonical examples — the same input features now imply a different outcome, because the world moved or an adversary adapted. A word that signaled spam last year is innocuous now because spammers stopped using it. This is the one that can't be fixed by reweighting, since the labeling function itself changed, so you need fresh labeled data and periodic retraining. In adversarial domains it's continuous, which is why fraud teams retrain on a rolling window rather than shipping a model once.

**20. How do you detect input drift in production?**
KS test, KL divergence, PSI between train and live distributions. Monitor input feature distributions per feature.

> **Saying it out loud.** Monitor each input feature's distribution against a training-time reference, using KS tests for continuous features, PSI or KL divergence for binned ones. The practical warning is alert fatigue: with 200 features and any test at a 5 percent threshold, you'll get several alarms every day from pure noise, so you need correction or sensible thresholds. And the crucial caveat to state is that input drift is not the same as performance drift — inputs can shift harmlessly, and performance can collapse with no visible input change if the concept drifted. Monitor outcomes when you can get them.

**21. What's PSI?**
Population Stability Index. Bin-based comparison of two distributions; PSI > 0.25 typically flagged as significant shift.

> **Saying it out loud.** Population Stability Index is a binned comparison of two distributions — essentially a symmetrized KL divergence with a conventional threshold table attached. The rules of thumb are under 0.1 is stable, 0.1 to 0.25 is worth a look, above 0.25 is a genuine shift. It's a credit-scoring convention rather than a deep statistical result, and its value is precisely that the thresholds are agreed on, so everyone in the organization interprets the number the same way. Watch out that it's sensitive to your binning and to empty bins.

**22. Importance weighting for covariate shift?**
Reweight training samples by $q(x)/p(x)$. Hard to estimate ratio; can use density estimators or classifier-based estimates.

> **Saying it out loud.** You reweight each training example by the ratio of its density under the new distribution to its density under the old one, so training effectively emphasizes the regions you now see. In principle that's an unbiased correction for covariate shift. In practice estimating a density ratio in high dimensions is hard, and the real killer is variance — if some region is rare in training and common in production, you get enormous weights and your effective sample size collapses. Rule of thumb: if the weight range spans more than a factor of ten or so, you should be collecting data, not reweighting.

**23. Shift detection via classifier?**
Train a classifier to distinguish "is this from train or production?" If AUC > 0.5 + something, there's shift.

> **Saying it out loud.** This is my favorite trick because it handles the multivariate case for free. Label your training data as class zero and your recent production data as class one, train a classifier, and check its AUC on a held-out split. If it can't tell them apart, AUC near 0.5, there's no meaningful shift. If it can, there is — and better, the feature importances tell you exactly which features moved. Per-feature statistical tests miss shifts in the joint distribution entirely, and this catches them.

---

## D. Class imbalance

**24. Why is accuracy bad for imbalanced data?**
Predicting majority class always gives high accuracy (e.g., 99% if positive class is 1%).

> **Saying it out loud.** Because a model that predicts the majority class every single time gets 99 percent accuracy when the positive rate is 1 percent, while being completely useless. Accuracy implicitly assumes that both error types cost the same and that the classes are balanced, and in fraud, disease, or defect detection neither holds. Say the number out loud in the interview — the 99-percent-useless model is what makes the point instantly. Then pivot to precision, recall, and the operating point that actually matters for the decision.

**25. Right metrics for rare-class problems?**
Precision, recall, F1, AUPRC. AUROC is OK but can be misleading at extreme imbalance.

> **Saying it out loud.** Precision and recall at the threshold you'd actually deploy, plus the precision-recall curve for the overall picture. AUPRC is the right summary because its baseline is the positive rate itself, so at 1 percent prevalence an AUPRC of 0.3 is genuinely good. AUROC isn't wrong but it's flattering, because a huge pile of easy negatives makes the curve look great. And the answer that shows seniority is asking what the errors cost, then picking the threshold that minimizes expected cost rather than optimizing an abstract metric.

**26. AUPRC vs AUROC — when prefer AUPRC?**
Severely imbalanced data. AUPRC focuses on positive class behavior; AUROC averages across the full operating curve where the negative dominance dilutes.

> **Saying it out loud.** Whenever the positive class is rare and you care about it specifically. The reason is that AUROC's x-axis is the false positive rate, and with a million negatives you can rack up thousands of false positives while barely moving that rate — so the curve looks fine while your precision is terrible. AUPRC uses precision, which feels every one of those false positives. The concrete illustration is a model with 0.95 AUROC and 0.20 AUPRC at 1 percent prevalence, which sounds excellent and means four out of five alerts are wrong.

**27. SMOTE — what does it do?**
Synthetic minority oversampling. Generates synthetic minority points by interpolating between minority neighbors. Risk: amplifies noise/outliers near class boundaries.

> **Saying it out loud.** SMOTE creates synthetic minority examples by interpolating between a minority point and one of its minority neighbors. The intent is to give the classifier more positive signal without simply duplicating rows. The risk is that if a minority point is actually mislabeled or sits deep inside the majority region, you're now manufacturing more of that error and blurring the boundary. Honestly, the current consensus is underwhelming — on modern classifiers, class weighting or simply adjusting the decision threshold usually matches or beats SMOTE with far less risk.

**28. Class weighting in the loss?**
Multiply per-sample loss by class-dependent weight. Standard PyTorch: `nn.CrossEntropyLoss(weight=class_weights)`.

> **Saying it out loud.** You multiply each example's loss by a weight based on its class, typically inversely proportional to class frequency, so the rare class contributes as much total gradient as the common one. It's one argument in the loss function, and there's no synthetic data and no resampling to go wrong. The thing to remember is that it distorts the output probabilities exactly like resampling does, so if you need calibrated probabilities you have to recalibrate afterwards or correct for the prior shift.

**29. Focal loss formula?**
$-(1-p_t)^\gamma \log p_t$. Down-weights easy examples ($p_t$ near 1). $\gamma$ typically 2.

> **Saying it out loud.** Focal loss multiplies cross-entropy by one minus the predicted probability raised to gamma, which shrinks the loss on examples the model already gets right. With gamma at 2, something predicted at 0.9 confidence contributes a hundredth of its usual loss. It came from dense object detection, where you have a hundred thousand background boxes per image and their aggregate easy loss swamps the handful of real objects. So it's about extreme imbalance between easy and hard examples, not just between class counts — and if your negatives aren't overwhelmingly easy, plain class weighting does the same job more simply.

**30. Should you resample the test set?**
**No.** Resample only training set. Test set must reflect deployment distribution.

> **Saying it out loud.** Absolutely not, and this one is worth being emphatic about. The test set exists to estimate performance on the real distribution, so rebalancing it means you're measuring performance in a world that doesn't exist. Rebalance training if it helps optimization; never touch evaluation. The failure mode is memorable — you report 85 percent precision on a balanced test set and see 8 percent in production, because the true prevalence is a hundredth of what you evaluated at.

**31. After resampling, what about probabilities?**
They're distorted. Calibrate after, or apply a post-hoc shift to recover original ratios.

> **Saying it out loud.** They're systematically inflated, because the model learned on a world where positives were much more common than they really are. If you resampled up to 50 percent positive from a true 1 percent, the model's outputs are wildly overconfident about the positive class. There's a clean analytic correction using the ratio of the training prior to the true prior, or you can fit a calibration map on an unresampled validation set. The point to make is that ranking survives resampling but probabilities do not, so if anything downstream reads the number rather than the order, you must fix it.

---

## E. Bias-variance and double descent

**32. Bias-variance decomposition?**
$\mathbb{E}[(\hat{f}(x) - y)^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$.

> **Saying it out loud.** Expected squared error splits into three pieces: bias squared, how wrong your model class is on average; variance, how much your fit bounces around with different training samples; and irreducible noise, which no model can beat. The value of the decomposition is diagnostic. High training error and high test error means bias, and you need a bigger model or better features. Low training error and high test error means variance, and you need more data or more regularization. Those two problems have opposite fixes, which is why misdiagnosing costs you weeks.

**33. High bias means?**
Underfitting. Model too simple. Both train and test error high.

> **Saying it out loud.** Underfitting — the model isn't expressive enough to capture the pattern, so it's wrong in the same way on every dataset. The signature is that training error is already high, so nothing on the regularization menu will help; you need more capacity, better features, or a longer training run. It's the easier problem to spot, because a model failing on data it has already seen is unambiguous.

**34. High variance means?**
Overfitting. Model too complex. Train error low, test error high.

> **Saying it out loud.** Overfitting — the model is memorizing the specific training set including its noise, so training error is near zero and test error isn't. The fixes are more data, more regularization, or less capacity, in roughly that order of preference. The reason to keep this distinct from bias is that the remedies are opposite: adding regularization to an underfitting model makes it worse, and people do it all the time because regularization feels like the safe default.

**35. What's double descent?**
Past the interpolation threshold (params ≈ data points), test error decreases again as capacity increases. Belkin et al., 2019.

> **Saying it out loud.** The classical picture says test error is U-shaped in model size — too small underfits, too big overfits. Double descent says that's only the first half. Right at the interpolation threshold, where the model has just enough capacity to fit the training data exactly, test error spikes. Push past it and error comes down again, often below the classical minimum. That's why a model with far more parameters than data points can generalize well, which the textbook curve says is impossible, and it's a big part of why scaling up kept working when theory said it shouldn't.

**36. Why does double descent happen?**
Modern over-parameterized models effectively select smoother interpolators. Implicit bias of optimization (SGD favors flat minima) plays a role.

> **Saying it out loud.** At exactly the interpolation threshold there's essentially one way to fit the data, and the model is forced into it however contorted it is — that's the spike. With more capacity there are many interpolating solutions, and gradient descent doesn't pick one at random; its implicit bias picks a smooth, small-norm one. So over-parameterization gives the optimizer room to choose a well-behaved fit rather than the only available fit. The framing worth offering is that capacity alone was never the right complexity measure — what matters is the effective complexity of the solution the optimizer actually lands on.

**37. Implicit regularization of SGD?**
SGD tends to converge to flat minima (low Hessian eigenvalues), which generalize better empirically. Adam has weaker implicit regularization.

> **Saying it out loud.** Nobody wrote it into the loss, but SGD systematically prefers certain solutions among the many that fit the data equally well — flat basins rather than sharp ones, because gradient noise makes narrow minima unstable. That preference is a big part of why heavily over-parameterized networks generalize at all. It also explains a practical folk observation: Adam has weaker implicit regularization than SGD, which is why vision models often generalize better with SGD plus momentum even though Adam converges faster. And SAM is what you get when you make the preference explicit instead of hoping for it.

---

## F. Cross-validation

**38. What's nested CV?**
Outer loop: k-fold for evaluation. Inner loop: k-fold within each train fold for hyperparameter tuning. Prevents tuning from leaking into evaluation.

> **Saying it out loud.** Two loops. The outer one splits for evaluation; inside each outer training fold, an inner loop does the hyperparameter search. The point is that the outer test folds are never seen by the tuning process, so your reported number isn't inflated by selection. It's expensive — 5 by 5 means 25 model fits — which is why nobody does it in deep learning. The honest version of the answer is that in practice you use a single untouched holdout as a cheap stand-in, and you know it's a compromise.

**39. Why do hyperparam tuning + evaluation on the same fold leak?**
You're choosing the model that does best on the eval set, biasing the eval estimate.

> **Saying it out loud.** Because picking the configuration with the best score on a set turns that set's noise into part of your selection criterion. Try 100 configurations and the winner is partly genuinely good and partly lucky on that specific data, so the reported score is optimistically biased. It's the same multiple-comparisons problem as running 100 statistical tests. The expected inflation grows with how many configurations you tried, which is why the winner of a huge sweep so reliably disappoints on the real holdout.

**40. Stratified k-fold — when?**
Classification with imbalanced classes. Preserves class ratio per fold; reduces estimator variance.

> **Saying it out loud.** Any time you're doing classification, and mandatorily when classes are imbalanced. Stratification keeps each fold's class proportions matching the whole dataset, which reduces the variance of your estimate. Without it, at 1 percent prevalence and a small dataset, some fold might contain almost no positives and its metrics become meaningless — you'd be averaging over folds that aren't measuring the same thing. It costs nothing and it's the default in most libraries for a reason.

**41. Time-series CV strategy?**
Sliding window or expanding window. Always test on data later than train. Never random split.

> **Saying it out loud.** Always train on the past and test on the future — either an expanding window that keeps accumulating history, or a sliding window of fixed length if you think older data has stopped being relevant. Never random. It's also worth adding a gap between train and test if your label takes time to materialize, otherwise you leak through the label horizon: if churn is defined over 30 days, the last 30 days of your training window contains information about the test period.

**42. Group k-fold use case?**
When you want generalization across entities (users, patients, etc.). Each entity entirely in one fold.

> **Saying it out loud.** Whenever your rows aren't independent because they cluster under some entity — multiple visits per patient, multiple sessions per user, multiple frames from one video. Group k-fold keeps every entity wholly inside one fold, so the model can't recognize the entity from training and coast at test time. Without it your cross-validation measures memorization and looks great, and deployment on genuinely new users looks nothing like it. Medical ML is full of published results that fell apart on this exact point.

---

## G. Ablations

**43. What's a good ablation?**
One component varied at a time, everything else fixed; multiple seeds; matched compute; multiple evals if claiming generality.

> **Saying it out loud.** Change one thing, hold everything else fixed, run multiple seeds, and match the compute budget. The compute point is the one people skip: if your new component adds 20 percent more parameters, comparing against the old model at the same step count means you've conflated the idea with the extra capacity. And multiple seeds are non-negotiable, because seed-to-seed variation on a lot of benchmarks is larger than the improvements people publish. If you claim generality, show it on more than one dataset.

**44. You added X, performance improved by 0.3 points. Real?**
Need: multiple seeds, std reported, paired test or bootstrap of difference. 0.3 might be within seed noise.

> **Saying it out loud.** My first question is what the seed-to-seed standard deviation is, because on most benchmarks 0.3 points is comfortably inside it. Run five seeds each way, report mean and spread, and do a paired test on the differences using the same evaluation examples. If the interval on the difference includes zero, you have nothing yet. This is the answer that separates people who have actually run experiments from people who have read about them — the instinct to ask about noise before believing the number.

**45. Why does ablation matter when papers report single numbers?**
Because single numbers without ablation can't establish *which component* drove the gain. Component might be a placebo.

> **Saying it out loud.** Because a single number tells you a system works and nothing about why. If you changed five things and got a gain, you don't know whether one of them did all the work, whether two cancel out, or whether it's noise — and the field has repeatedly found that headline gains came from the learning rate schedule rather than the proposed architecture. Without ablations you can't build on the work, only copy it wholesale. That's the practical cost: not being wrong, but being uninformative.

---

## H. Metric uncertainty

**46. Wald CI for accuracy?**
$\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$.

> **Saying it out loud.** Accuracy is a proportion, so the standard error is the square root of $p(1-p)/n$ and the interval is roughly two of those either side. The number worth carrying around is that on a 1,000-example test set at 87 percent accuracy, the interval is about plus or minus 2 points — so anything within 4 points is not a distinguishable difference. That's why leaderboard gaps of half a point on small test sets mean essentially nothing. Near zero or one, use a Wilson interval instead, since the symmetric formula runs off the end of the range.

**47. CI for AUC?**
Bootstrap (resample test set 1000+ times, compute AUC each time, quantile). Or DeLong's method.

> **Saying it out loud.** Bootstrap the test set: resample prediction-label pairs with replacement a thousand times or more, recompute AUC each time, take the 2.5th and 97.5th percentiles. DeLong's method gives you an analytic alternative specifically for AUC, but the bootstrap works for any metric you might report next, so it's the better habit. What it usually reveals is sobering — on a few hundred examples the interval is several points wide, which means most model comparisons at that scale aren't measuring anything.

**48. Two CIs overlap — does that mean no difference?**
No. Paired test on the difference is the right way. CIs of differences can exclude zero even when individual CIs overlap.

> **Saying it out loud.** No, and this is a genuinely common mistake. Overlapping intervals can still correspond to a statistically significant difference, because comparing two independent intervals ignores the fact that the two models are being evaluated on the same examples and their errors are correlated. The right move is to build a confidence interval for the difference directly, using a paired bootstrap. That interval is much narrower, and it can exclude zero even when the individual intervals overlap heavily.

**49. Paired bootstrap procedure for model comparison?**
For each bootstrap sample, compute metric for both models on the same sample. Look at the distribution of differences. Reject "no difference" if 0 not in CI of differences.

> **Saying it out loud.** Resample the test set once, evaluate both models on that same resample, record the difference, and repeat a thousand times. Keeping the sample shared is the whole point — the two models see identical examples, so their errors are correlated and the difference has far less variance than either metric alone. Then it's simple: if zero is outside the interval of differences, the gap is real. This routinely detects differences of a few tenths of a point that independent intervals would dismiss as noise.

**50. McNemar's test — when?**
Comparing two binary classifiers on the same test set. Tests if their disagreements are symmetric (same number of A-correct B-wrong vs A-wrong B-correct).

> **Saying it out loud.** When you're comparing two binary classifiers on the same test set. It ignores the cases where both models agree, which carry no information about which is better, and looks only at the disagreements: how often A is right and B wrong versus the reverse. If those two counts are roughly balanced, the models are equivalent. It's the standard paired test for classifier comparison, and the reason it's more powerful than comparing accuracies is precisely that it conditions on the agreements instead of letting them dilute the signal.

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
