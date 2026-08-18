# Advanced ML Theory — Interview Grill

> 40 questions on bias-variance, cross-validation, learning curves, AIC/BIC, ROC/PR. Drill until you can answer 28+ cold.

---

## A. Bias-variance

**1. Bias-variance decomposition?**
$\mathbb{E}[(y - \hat{f}(x))^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$.

> **Saying it out loud.** Expected squared error splits into exactly three pieces: bias squared, variance, and irreducible noise. Bias is how wrong you'd be on average even with infinite data of this model class, variance is how much your model wobbles when you resample the training set, and noise is the part nothing can fix. The reason it's worth memorizing is diagnostic: it tells you that any error you're seeing has one of three causes, and each has a different remedy. It's a decomposition for squared error specifically, and it doesn't carry over cleanly to zero-one loss.

**2. Bias definition?**
$\mathbb{E}_D[\hat{f}_D(x)] - f^*(x)$. Average error from truth.

> **Saying it out loud.** Bias is the gap between the truth and the average prediction you'd make if you retrained on many different datasets. It's error from the model class being too rigid to represent the real function, not from bad luck with data. Fit a straight line to a curve and no amount of data removes that gap. That's why more data doesn't fix a high-bias model, only a richer model does.

**3. Variance definition?**
$\mathbb{E}_D[(\hat{f}_D(x) - \mathbb{E}_D[\hat{f}_D(x)])^2]$. How much predictions vary across training sets.

> **Saying it out loud.** Variance is how much your fitted model jumps around when the training set changes. Same model class, different sample, very different predictions, that's high variance. A deep unpruned tree is the classic example: reshuffle a few points and the whole structure changes. Variance is the part that averaging fixes, which is exactly what bagging is for, and it's also the part that shrinks as you add data.

**4. Irreducible noise $\sigma^2$?**
$\mathbb{E}[(y - f^*(x))^2]$. Cannot be reduced by any model.

> **Saying it out loud.** The irreducible noise is the variance of the target given the features, that is, the part of $y$ that simply isn't a function of what you measured. If two identical patients have different outcomes, no model can predict the difference. It's the floor on your error and the honest ceiling on your performance. It also means a reported error of zero on real data almost always indicates leakage, not brilliance.

**5. High-bias signature?**
Train and val errors both high. Train-val gap small.

> **Saying it out loud.** High bias looks like both training and validation error being bad, and close together. The model isn't even fitting the data it can see, so it's not a generalization problem, it's a capacity problem. More data will not help. What helps is a bigger model, better features, less regularization, or a longer training run.

**6. High-variance signature?**
Train error low, val error high. Train-val gap large.

> **Saying it out loud.** High variance looks like great training error and much worse validation error, with a big gap between them. The model has memorized rather than generalized. That's the one case where more data genuinely helps, along with stronger regularization, more aggressive early stopping, or a simpler model. The gap itself is the diagnostic, so it's the first number to check when a model disappoints.

**7. Modern over-parameterized regime?**
Double descent — bias-variance trade-off doesn't follow classical U-shape.

> **Saying it out loud.** In the modern over-parameterized regime the classical U-shaped curve doesn't hold. As you increase model size, test error goes down, then up as you approach the interpolation threshold where the model has just enough capacity to fit the training data exactly, and then down again as you push well past it. That's double descent, and it's why an enormous network with far more parameters than data points can generalize well. The mechanism is thought to be implicit regularization: past the threshold, gradient descent selects a minimum-norm solution among the infinitely many that fit.

---

## B. Cross-validation

**8. k-fold CV procedure?**
Split into $k$ folds. For each: train on $k-1$, test on 1. Average errors.

> **Saying it out loud.** Split the data into $k$ chunks, and $k$ times train on all but one chunk and evaluate on the one you left out, then average. Every data point gets used for validation exactly once and for training $k-1$ times, which is why it beats a single holdout on small data. You also get a spread across folds, which is a rough uncertainty estimate. The cost is $k$ times the training compute.

**9. Standard $k$?**
5 or 10. Compromise between bias (low for higher $k$) and variance (high for $k = n$).

> **Saying it out loud.** Five or ten, and the reason is a bias-variance tradeoff in the estimator itself. Small $k$ means each model trains on much less data, so your error estimate is biased pessimistic. Large $k$ approaches leave-one-out, where the training sets overlap almost completely, so the fold errors are highly correlated and the average is noisy. Ten folds is the usual sweet spot, and five when compute is tight.

**10. LOO-CV — why high variance?**
Training sets are highly correlated → predictions correlated → empirical mean has high variance.

> **Saying it out loud.** With leave-one-out, every training set differs from every other by exactly one point, so the $n$ models are nearly identical and their errors are almost perfectly correlated. Averaging correlated quantities doesn't reduce variance the way averaging independent ones does, so the estimate stays noisy. It's nearly unbiased, since each model sees almost all the data, but the variance is high. It also costs $n$ trainings, which is why it's mostly used for linear models that have a closed-form shortcut.

**11. Stratified k-fold?**
Preserves class ratio per fold. Default for imbalanced classification.

> **Saying it out loud.** Stratified k-fold keeps the class proportions in each fold the same as in the full dataset. Without it, on an imbalanced problem you can get a fold with almost no positive examples, and then the metric for that fold is essentially noise. It's the default for classification in scikit-learn for exactly this reason. It's not optional at one percent positives; it's the difference between a usable estimate and a meaningless one.

**12. Group k-fold?**
Each entity (user, patient) entirely in one fold. For generalization across entities.

> **Saying it out loud.** Group k-fold keeps all rows belonging to the same entity, a user or a patient or a device, inside a single fold. Without it, the same user appears in both training and validation, so the model can memorize that user and your score reflects memorization rather than generalization to new users. Ask what unit you're actually generalizing to, and group by that. This is one of the most common and most damaging leakage bugs in applied ML.

**13. Time-series CV?**
Sliding or expanding window. Train on past, test on future. Never random.

> **Saying it out loud.** For time series you never randomize, because randomizing lets the model train on the future to predict the past. Instead you walk forward: train on everything up to some time, test on the window right after, then roll forward. Expanding window keeps all history, sliding window keeps a fixed recent span, and which you use depends on how quickly the process drifts. And you have to respect the gap between when a feature is actually available and when you'd be predicting, or you leak.

**14. Nested CV?**
Outer for eval, inner for hyperparameter tuning. Prevents tuning leakage.

> **Saying it out loud.** Nested cross-validation has two loops: the inner one picks hyperparameters and the outer one estimates performance. The reason you need it is that if you tune on the same folds you evaluate on, your reported number is optimistically biased, because you selected the configuration that happened to do best on that exact data. Nested CV keeps model selection strictly inside the training portion. It costs $k$ times $m$ trainings, which is why people skip it and quietly report inflated numbers.

**15. Common CV pitfalls?**
Tuning + eval same fold; preprocessing on full data; not stratifying for imbalance; random split for time-series.

> **Saying it out loud.** Four classics. Tuning and evaluating on the same folds, which inflates your estimate. Fitting preprocessing like a scaler or an imputer or a feature selector on the whole dataset before splitting, which leaks test statistics into training. Forgetting to stratify on imbalanced data. And random splits on time series. The rule that prevents most of them: every transformation that learns anything from data must be fit inside the fold, which is exactly what scikit-learn pipelines are for.

**16. LOO-CV closed form for linear regression?**
$\frac{1}{n}\sum (\frac{y_i - \hat{y}_i}{1 - h_{ii}})^2$ where $h_{ii}$ is hat-matrix diagonal. Avoids retraining.

> **Saying it out loud.** For linear regression you don't have to retrain $n$ times, because there's a closed form: the leave-one-out residual is the ordinary residual divided by one minus the corresponding hat-matrix diagonal. So a single fit gives you exact leave-one-out error. The intuition for $h_{ii}$ is leverage, how much a point pulls its own fitted value toward itself; high-leverage points get their residuals inflated the most, which is exactly right, since those are the points whose removal changes the fit most. It generalizes to ridge and to smoothers via generalized cross-validation.

---

## C. Learning curves

**17. What does train error converging to high value mean?**
High bias. Model too simple. More data won't help much.

> **Saying it out loud.** If the training error itself plateaus at a high value, the model can't even fit the data it has, so you're bias-limited. Adding data will do nothing, because the curve is already flat and more of the same won't change what the model class can represent. What you need is more capacity, better features, or less regularization. The recognizable shape is two curves that converge early and both sit high.

**18. What does big train-val gap mean?**
High variance. Overfitting. More data will help.

> **Saying it out loud.** A persistent gap between training and validation error means variance: the model fits what it saw and doesn't transfer. This is the case where more data actually helps, because the two curves are still converging and more samples drag validation error down toward training error. Regularization, early stopping, and simpler models help too. Look at whether the gap is still narrowing at the right edge of the plot, because that tells you whether more data is worth buying.

**19. Decision: more data vs better model?**
Plot learning curves. Big gap → more data. High train error → better model.

> **Saying it out loud.** Plot the learning curve, both errors against training set size, and read off which regime you're in. Big gap that's still closing means buy more data. High training error with the curves already converged means more data is wasted money and you need a better model or better features. That single plot converts a subjective argument into a decision, and it's a strong thing to propose in a system-design interview because it shows you'd rather measure than guess.

**20. Validation curve vs learning curve?**
Validation curve: y vs hyperparameter. Learning curve: y vs training set size.

> **Saying it out loud.** They vary different things on the x-axis. A learning curve varies training set size and tells you whether more data would help. A validation curve varies a hyperparameter, like tree depth or regularization strength, and tells you where the sweet spot for capacity is. Learning curve answers should I get more data; validation curve answers is my model too complex or too simple. People conflate them constantly.

---

## D. Information criteria

**21. AIC formula?**
$\mathrm{AIC} = 2k - 2 \log L$. Lower better.

> **Saying it out loud.** AIC is twice the number of parameters minus twice the log-likelihood, and lower is better. It's trading fit against complexity, where each extra parameter has to buy you at least one unit of log-likelihood to be worth it. It comes out of an approximation to the expected KL divergence between your fitted model and the truth. Since it's only defined up to an additive constant, only differences between models are meaningful.

**22. BIC formula?**
$\mathrm{BIC} = k \log n - 2 \log L$. Lower better.

> **Saying it out loud.** BIC is the number of parameters times the log of the sample size, minus twice the log-likelihood, and lower is better. The structural difference from AIC is that the penalty grows with $n$, so as you collect more data BIC becomes progressively stricter about adding parameters. It comes from a Laplace approximation to the marginal likelihood, so it's approximating Bayesian model selection with a uniform prior over models.

**23. AIC vs BIC penalty growth?**
BIC penalty $k \log n$ grows with $n$. AIC's $2k$ stays constant. BIC selects simpler models for large $n$.

> **Saying it out loud.** AIC's penalty is a flat two per parameter; BIC's is log $n$ per parameter, which grows without bound. So on a thousand data points, BIC charges about seven per parameter versus AIC's two, and BIC consistently picks simpler models. That's not a flaw in either, it reflects that they're answering different questions. AIC is aiming at prediction, BIC at identifying the true model.

**24. AIC purpose?**
Optimal for prediction. Doesn't assume true model in candidate set.

> **Saying it out loud.** AIC is built for prediction. It estimates which model will predict best on new data, and it explicitly does not assume the true model is among your candidates, which is realistic since it almost never is. Asymptotically it's efficient, meaning it picks the model minimizing prediction error. It's the right criterion when your goal is forecasting rather than explanation, and the cost is that it tends to select slightly over-complex models.

**25. BIC purpose?**
Consistent for true model identification (when true model in candidates).

> **Saying it out loud.** BIC is built for identification. If the true model really is in your candidate set, BIC will select it with probability approaching one as $n$ grows, which is what consistency means. That makes it the right tool when the question is which variables genuinely matter, rather than which model predicts best. The catch is the assumption that the truth is in your list, which is why BIC can be too conservative on real problems where every model is wrong.

**26. When does BIC penalty exceed AIC?**
$\log n > 2$ → $n > e^2 \approx 7.4$. Almost always.

> **Saying it out loud.** BIC's penalty exceeds AIC's as soon as log $n$ is bigger than two, so once $n$ passes about seven and a half. Which is to say, essentially always. So in practice you can just remember that BIC is the stricter criterion and picks the smaller model. When the two disagree, decide by what you're doing: prediction points to AIC, explanation points to BIC.

**27. Limitations of AIC/BIC?**
Need well-defined likelihood; assume correct model specification; effective $k$ unclear for regularized models.

> **Saying it out loud.** Three limitations. Both need a proper likelihood, so they don't apply to models fit by something other than maximum likelihood, which rules out most of modern ML. Both assume the model is correctly specified in the sense that the likelihood is right. And the parameter count $k$ is ill-defined for regularized or hierarchical models, since a ridge model with a hundred coefficients doesn't really have a hundred free parameters. The fix there is effective degrees of freedom, the trace of the hat matrix, and beyond that you're better off just cross-validating.

---

## E. ROC and PR

**28. ROC axes?**
TPR (recall) vs FPR (false alarm). Threshold-free.

> **Saying it out loud.** The ROC curve plots true positive rate against false positive rate as you sweep the decision threshold from strict to permissive. So it's recall on the vertical axis against false alarm rate on the horizontal. Every point is one possible threshold, which is why the curve summarizes the classifier's ranking rather than any single operating decision. Top-left is perfect, the diagonal is random guessing.

**29. AUROC interpretation?**
Probability random positive ranks above random negative.

> **Saying it out loud.** AUROC is the probability that a randomly chosen positive scores above a randomly chosen negative. That's a genuinely nice interpretation: it's about ranking, not about calibration or any particular threshold. It's also equal to the Mann-Whitney U statistic normalized. Its weakness is that it treats all negatives equally, so with a million negatives and a hundred positives it can look great while your top-ranked predictions are mostly wrong.

**30. PR curve axes?**
Precision vs Recall. Threshold-free.

> **Saying it out loud.** The precision-recall curve plots precision against recall as you sweep the threshold. Unlike ROC, both axes involve the positive class, which is why it stays informative under heavy imbalance. The baseline is not the diagonal, it's a horizontal line at the positive class prevalence, so a PR-AUC of point-one is excellent if your prevalence is one percent. That shifting baseline is the thing people forget when comparing PR-AUC across datasets.

**31. AUROC vs AUPRC for imbalance?**
AUPRC much more informative. AUROC dominated by easy negatives.

> **Saying it out loud.** Use PR when positives are rare. The reason is in the definitions: false positive rate has the number of negatives in its denominator, so with a million negatives, ten thousand false positives barely moves it, and ROC looks fine. Precision has false positives in its denominator directly, so it collapses immediately. At one percent prevalence a model can have an AUROC of point-nine-five and precision of five percent at any useful recall. That concrete case is what to say.

**32. Choosing operating point?**
Cost-weighted: $\arg\min_\tau (c_{\mathrm{FN}} \mathrm{FN} + c_{\mathrm{FP}} \mathrm{FP})$. Or fixed recall / FP rate.

> **Saying it out loud.** The principled way is to cost it out: assign a cost to false negatives and false positives, and pick the threshold that minimizes expected cost. In medicine a missed diagnosis dwarfs a false alarm, so you push recall; in spam filtering a false positive is a lost email, so you push precision. If you can't get real costs from the business, the practical fallback is to fix the operating constraint instead, like the highest precision achievable at eighty percent recall, or the alert budget your analysts can actually review. And say out loud that this is a business decision, not a modeling one.

**33. F1 formula?**
$F_1 = 2PR/(P+R)$. Harmonic mean.

> **Saying it out loud.** F1 is the harmonic mean of precision and recall, twice their product over their sum. It gives you one number when you need to compare models at a fixed threshold. Two caveats worth adding: it's threshold-dependent, unlike AUROC or PR-AUC, and it weights precision and recall equally, which is a real assumption and often the wrong one. It also completely ignores true negatives.

**34. F-beta?**
$F_\beta = (1 + \beta^2) PR / (\beta^2 P + R)$. $\beta > 1$ weights recall more.

> **Saying it out loud.** F-beta is the weighted version, where beta says how many times more you care about recall than precision. Beta of two weights recall more, beta of a half weights precision more, and beta of one is plain F1. It's the right thing to reach for when you can articulate the asymmetry but can't put dollar costs on the errors. The trick to remember the direction: beta greater than one favors recall, because you set beta to how much recall matters relative to precision.

**35. Why harmonic mean for F1?**
Penalizes imbalance: F1 = 0 if either P or R = 0. Arithmetic mean wouldn't.

> **Saying it out loud.** Because the harmonic mean is dominated by the smaller of the two numbers, so you can't paper over a bad half with a great one. Predict everything positive and you get recall one with precision at prevalence: the arithmetic mean would look respectable while F1 stays near zero. And if either precision or recall is exactly zero, F1 is zero, which is exactly the behavior you want. It's the same reason the harmonic mean is used for average speeds.

---

## F. Confusion matrix

**36. Precision formula?**
$TP/(TP + FP)$. Of positive predictions, how many right.

> **Saying it out loud.** Precision is true positives over everything you predicted positive: of the alarms you raised, what fraction were real. It's the metric your users feel when your system cries wolf. It says nothing about what you missed, which is why it always travels with recall. Predict a single item you're extremely sure about and you can get perfect precision while catching almost nothing.

**37. Recall (sensitivity) formula?**
$TP/(TP + FN)$. Of actual positives, how many caught.

> **Saying it out loud.** Recall, also called sensitivity or true positive rate, is true positives over all actual positives: of the things you should have caught, what fraction did you. It's the metric that matters when misses are expensive, like a cancer screen or a fraud detector. Predicting everything positive gives you perfect recall, which is why it always travels with precision. The two are traded off by moving one threshold.

**38. Specificity formula?**
$TN/(TN + FP)$. Of actual negatives, how many correctly negative.

> **Saying it out loud.** Specificity is true negatives over all actual negatives: of the things that were genuinely negative, how many you correctly left alone. It's the complement of the false positive rate, so it's the other axis of the ROC curve. Medical people use sensitivity and specificity where ML people use recall and one minus false positive rate, and being able to translate between the two vocabularies on the fly is worth a lot in a health-domain interview.

**39. MCC purpose?**
Balanced metric for imbalanced classification. Range $[-1, 1]$. 0 = random.

> **Saying it out loud.** Matthews correlation coefficient is essentially the correlation between the predicted and true labels, and unlike F1 it uses all four cells of the confusion matrix, including true negatives. It runs from minus one to one, with zero meaning random. The reason to prefer it under imbalance is that it only comes out high when the model does well on both classes, so you can't game it by getting the majority class right. It's the single most honest one-number summary of a confusion matrix, and it's underused.

**40. Accuracy when imbalanced?**
Misleading. 99% by predicting majority class always. Use F1, AUPRC, MCC instead.

> **Saying it out loud.** Accuracy is actively misleading under imbalance, because the trivial predict-the-majority model already gets ninety-nine percent when one percent of cases are positive. So a ninety-nine percent accuracy claim on rare-event data usually means the model learned nothing. Use PR-AUC for ranking quality, F1 or F-beta at your operating threshold, or MCC as a balanced single number. And the quick tell in any interview: whenever someone quotes accuracy, ask what the class balance is.

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
