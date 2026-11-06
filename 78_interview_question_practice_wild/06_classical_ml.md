# Classical ML foundations

**Foundation explainer:** Core material · **Priority:** Very important

Very important: these concepts are common follow-ups even in LLM-focused interviews.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f051"></a>

## F051. Explain logistic regression from scores to gradients.

**Start here.** Logistic regression models binary probability with a linear score passed through a sigmoid.

**Work it through.**

1. Compute z=xᵀw+b and p=sigmoid(z); p/(1-p)=exp(z), so the log odds are linear.
2. Fit weights by minimizing binary cross-entropy, preferably calculated from logits for stability.
3. The score gradient is p-y, giving weight gradient Xᵀ(p-y)/N under mean reduction. Choose the classification threshold using validation costs and metrics.

**Small example.** A logit of 0 means p=0.5; a logit of log(3) means p=0.75 and odds 3:1.

**Common trap.** Calling sigmoid a hard threshold, or assuming a threshold of 0.5 is optimal for every cost structure.

**Follow-up.** When is the linear boundary too restrictive? Feature engineering, kernels or nonlinear models may help; compare on a leakage-free split.

---

<a id="f052"></a>

## F052. Diagnose high bias versus high variance.

**Start here.** Use training and validation behavior to distinguish insufficient fitting from poor generalization.

**Work it through.**

1. If both errors remain high, check the pipeline and optimization first, then consider inadequate features or capacity.
2. If training error is low but validation error is high, check leakage-free splitting, distribution shift and overfitting.
3. Use learning curves and controlled changes: more data, regularization, simpler models or better features depending on the diagnosis.

**Small example.** Training error 1% and validation error 25% suggests a generalization problem, but a group-distribution mismatch could be the cause rather than model complexity alone.

**Common trap.** Treating the bias–variance story as a diagnosis without inspecting label noise, split structure or optimization failure.

**Follow-up.** Why can a larger model generalize better? Optimization and inductive bias can change along with capacity; simple monotonic stories are incomplete.

---

<a id="f053"></a>

## F053. Compare L1 and L2 regularization.

**Start here.** Both penalize large coefficients, but their geometry produces different coefficient patterns.

**Work it through.**

1. Add lambda*sum(abs(w)) for L1 or a stated lambda-scaled squared norm for L2.
2. L1 can produce exact zeros with suitable optimization, aiding sparse selection; L2 usually shrinks coefficients smoothly and stabilizes correlated predictors.
3. Standardize features using training statistics, tune lambda on validation data and state whether the intercept is penalized.

**Small example.** With two almost identical predictors, L1 may choose one unstably while L2 spreads weight across both.

**Common trap.** Interpreting a selected correlated feature as uniquely causal, or comparing unscaled coefficients across different units.

**Follow-up.** What is elastic net? A combination that can encourage sparsity while stabilizing correlated groups.

---

<a id="f054"></a>

## F054. Why can high accuracy hide failure on imbalanced data?

**Start here.** A model can achieve high accuracy by predicting the common class while missing nearly all important rare events.

**Work it through.**

1. Inspect class prevalence, the confusion matrix and per-class precision/recall.
2. Choose a metric connected to the decision: PR behavior for rare-event retrieval, calibration for probability-based decisions, and explicit cost curves for thresholds.
3. Try class weights, resampling or focal loss carefully, validating on the natural deployment distribution and recalibrating when needed.

**Small example.** At 1% positive prevalence, an always-negative classifier has 99% accuracy and zero recall on the positive class.

**Common trap.** Resampling the test set to a balanced distribution and then reporting its precision as if it were production precision.

**Follow-up.** Why can ROC-AUC look strong while precision is poor? False positives from the large negative population can dominate the positive predictions.

---

<a id="f055"></a>

## F055. Compare random forests and gradient boosting.

**Start here.** Random forests average diversified trees; boosting adds models sequentially to improve an objective.

**Work it through.**

1. A forest typically trains trees on bootstrapped samples with feature randomness, reducing variance through averaging.
2. Boosting fits successive updates informed by current errors or loss gradients; learning rate, tree depth and stopping determine its capacity.
3. Compare time-aware validation, calibration, inference cost and sensitivity to hyperparameters rather than selecting by reputation.

**Small example.** A hundred independent high-variance trees can become stable when their errors are weakly correlated; a hundred almost identical trees average away much less error.

**Common trap.** Saying every boosting algorithm literally trains on just the previously misclassified rows.

**Follow-up.** Why do forests often need less tuning? Their averaging is relatively robust, though feature leakage and distribution shift still matter.

---

<a id="f056"></a>

## F056. Fit a baseline on messy tabular data without leakage.

**Start here.** Build a trustworthy split and simple baseline before sophisticated feature engineering.

**Work it through.**

1. Define the target, prediction time and unit of independence, then split by time or group if required.
2. Fit imputation, scaling and category encodings on training folds only. Start with a dummy baseline and a regularized linear or small tree model.
3. Evaluate slices, missingness patterns and calibration; document features unavailable at decision time and remove them.

**Small example.** A “refund issued” field may predict churn extremely well but be recorded after the churn decision, making it unusable for a prospective model.

**Common trap.** Fitting preprocessing on all rows before cross-validation, even when labels were not directly used.

**Follow-up.** How would you handle unseen categories? An explicit unknown bucket or encoder behavior that is tested before deployment.

---

<a id="f057"></a>

## F057. Explain multicollinearity and stable least squares.

**Start here.** Correlated predictors can make coefficient estimates unstable even when predictions remain useful.

**Work it through.**

1. Write the least-squares objective ||Xw-y||²; inspect rank or singular values of X.
2. Use QR or SVD-based least squares rather than explicitly inverting XᵀX. Centering and scaling can help conditioning.
3. Use ridge if shrinkage is appropriate, and distinguish predictive stability from interpretation of individual coefficients.

**Small example.** If one column is exactly twice another, several coefficient pairs produce the same predictions. The individual coefficients are not identifiable without an extra convention.

**Common trap.** Taking a numerical inverse and treating enormous coefficients as strong evidence of real effects.

**Follow-up.** Does dropping one correlated variable always improve test prediction? No; compare held-out performance and the intended interpretation.

---

<a id="f058"></a>

## F058. How should you split data for reliable model selection?

**Start here.** The validation procedure must imitate which information will be available when the model is used.

**Work it through.**

1. Use a final held-out test set and tune only within training/validation data.
2. Use grouped splits for repeated entities and forward-looking splits for time-dependent prediction. Stratification preserves class proportions when compatible with those constraints.
3. Fit the complete preprocessing and model pipeline inside each fold; reserve nested validation for unbiased evaluation of an extensive selection procedure.

**Small example.** A patient's visits belong in the same fold if the deployment target is unseen patients; otherwise identity patterns can leak across folds.

**Common trap.** Randomly splitting temporally adjacent records and interpreting the result as a future-deployment estimate.

**Follow-up.** When is a random split reasonable? When examples are sufficiently independent and identically distributed relative to the intended use.

---

<a id="f059"></a>

## F059. Explain dropout, normalization and weight decay separately.

**Start here.** These methods affect training through different mechanisms and should not be described as one generic regularizer.

**Work it through.**

1. Dropout randomly masks activations during training with the implementation's scaling convention; ordinary inference disables the masks.
2. Normalization adjusts activation statistics, with LayerNorm using per-example feature statistics and BatchNorm usually maintaining running statistics.
3. Weight decay shrinks parameters through the optimizer update. Tune their effects with controlled ablations and inspect train/eval discrepancies.

**Small example.** With inverted dropout probability 0.5, a kept activation is doubled during training so its expectation remains unchanged.

**Common trap.** Leaving dropout on for a deterministic evaluation or assuming LayerNorm needs BatchNorm-style running means.

**Follow-up.** Can normalization also regularize? Yes, but its effect depends on the method and batch behavior; explain the actual mechanism.

---

<a id="f060"></a>

## F060. Explain double descent without claiming it is universal.

**Start here.** Prediction error can rise near an interpolation threshold and fall again as model size or sample size moves beyond it under particular conditions.

**Work it through.**

1. Create a controlled regression experiment with known data generation, train/test separation and a fixed fitting rule.
2. Sweep model dimension relative to sample count, recording train loss, test loss and the conditioning of the design matrix.
3. Repeat seeds and compare regularization strengths. Discuss noise, minimum-norm solutions and how the result depends on the experimental setup.

**Small example.** For unregularized random-feature least squares, error may peak when the number of fitted parameters approaches the number of samples.

**Common trap.** Using one noisy curve as proof that “more parameters always helps,” or changing data noise while changing model size.

**Follow-up.** What often softens the peak? Regularization, better conditioning or a different noise/data regime.

---
