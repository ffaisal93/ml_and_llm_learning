# ML Debugging — Interview Grill

> 40 questions on training failure diagnosis, NaN, leakage, drift. Drill until you can answer 28+ cold.

---

## A. Debugging tree

**1. Order of debugging layers?**
Data → pipeline → model → loss → optimizer → training loop → eval → deployment.

**2. Cheap-checks-first principle?**
Inspect data, plot losses, sanity-check shapes before custom gradient debugging.

**3. The single best 5-minute sanity check?**
Try to overfit one batch. Loss should go to ~0. If not, fundamental bug.

> **Saying it out loud.** I debug in order of how cheap the check is, not how interesting it is: data, pipeline, model, loss, optimizer, training loop, evaluation, deployment. That ordering is a base-rate argument — most training failures are a label format, a shape, or a learning rate, not a subtle gradient bug, so I look at the data and the loss curve before I open the model file. And the single highest-value five minutes is trying to overfit one batch: if a model can't drive the loss to nearly zero on thirty-two examples it has memorized, it's broken, and no amount of tuning will save it. Failure mode avoided: two days rewriting the architecture when the labels were shifted by one.

---

## B. Loss curves

**4. Flat loss curve causes?**
LR too low, frozen weights, broken graph, wrong loss function.

**5. Loss exploding (NaN)?**
LR too high, FP16 overflow, bad init, attention saturation, division by zero.

**6. Train loss low, val loss high?**
Overfitting; or train-val mismatch (preprocessing, distribution).

**7. Loss flat then sudden drop?**
Phase transition; warmup not finished.

**8. Loss decreasing then sudden spike?**
Bad batch, optimizer instability, gradient cliff, missing gradient clipping.

**9. Why plot in log-y axis?**
Reveals early dynamics; small differences visible.

> **Saying it out loud.** Loss curves have shapes and each shape has a short list of causes. Perfectly flat means gradients aren't reaching the weights — frozen parameters, a detached graph, or a learning rate near zero. Bouncing without descending means the learning rate is too high. A jump to NaN means overflow or a gradient cliff. Flat then suddenly dropping is usually warmup finishing, or a genuine phase transition. Smooth descent then a spike is almost always a single bad batch, and the fix is gradient clipping. Two habits: log y-axis, so the first few hundred steps are visible, and watch per-batch loss rather than the smoothed average, because a smoothed curve hides exactly the spike you need to see.

---

## C. Sanity checks

**10. Overfit one batch — what should happen?**
Loss → ~0. Validates loss function, gradient flow, model capacity.

**11. Tiny dataset (100 examples) — should...?**
Train accuracy → ~100%. If not, model lacks capacity or there's a bug.

**12. Inspect 5 random batches?**
Check shapes, label distributions, raw values. Many bugs visible immediately.

**13. Why check if loss is well-defined for a single example?**
A loss bug (wrong shape, wrong reduction) often shows up only when you compute it.

> **Saying it out loud.** Sanity checks are how you separate “my model is wrong” from “my code is wrong,” and you should run them before every serious training job. Overfit a single batch first: turn off shuffling, dropout, and augmentation, and drive the loss to near zero — that one test validates the loss function, gradient flow, and enough capacity all at once. Then take a hundred examples and expect nearly perfect training accuracy. Then print a few raw batches and actually look: shapes, value ranges, label distribution, and for text, decode the tokens and read them. The named failure this catches most often is a normalization or label-format mismatch that would otherwise look like a mysterious accuracy ceiling.

---

## D. NaN debugging

**14. FP16 overflow at $x > ?$**
~11. $e^{11.1} > 65504$, the FP16 max. (~88 is the FP32 threshold.)

**15. Why use BF16 over FP16?**
FP32-equivalent exponent range. No overflow at typical magnitudes.

**16. Log of 0 fix?**
Add small $\epsilon$: $\log(p + 10^{-9})$.

**17. NaN at step 0 — what?**
Bad init, bad first batch, broken data.

**18. NaN at step 5000 — what?**
Numerical instability triggered by something. Gradient clip; lower LR; restart.

**19. Detect NaN early?**
`torch.autograd.set_detect_anomaly(True)` (slow but pinpoints first NaN site).

**20. Standard gradient clip?**
1.0 by global norm.

> **Saying it out loud.** With NaNs the diagnostic question is when it appeared, because step zero and step five thousand mean different things — step zero is initialization or broken input, and thousands of steps in is an instability that finally got triggered. The usual culprits are exponentials in a low-precision format, a division by a variance that reached zero, a log of zero, and a giant gradient from an outlier batch. Here's the number: FP16 tops out at 65504, so `exp` overflows around 11, whereas FP32 holds until about 88 — which is exactly why BF16 is the modern default, same sixteen bits but FP32's exponent range, at the cost of precision you don't need. Then it's clip at global norm 1.0, drop the learning rate, and restart from the last clean checkpoint, because once the loss reads NaN the weights are already poisoned.

---

## E. Leakage

**21. Symptoms of leakage?**
Suspiciously high offline metrics; train+val both 99%, prod fails; one feature dominates importance.

**22. Common leakage type — preprocessing?**
Fitting scaler on full dataset before split.

**23. Target leakage — example?**
Using "days since last login" as feature when it includes post-churn data.

**24. Group leakage?**
Same user/patient on both sides of split when generalization is across users.

**25. Temporal leakage detection?**
Use time-based split (last $X\%$ as val). If accuracy drops a lot vs random split, there was temporal leakage.

**26. Single-feature AUC > 0.95 means?**
Suspect leakage. Audit that feature.

**27. Cross-correlation check for leakage?**
Correlate every feature with label. > 0.9 = suspicious.

> **Saying it out loud.** Leakage is any information reaching the model that it won't have at prediction time, and the symptom is a number that's too good to be true. My triggers are a single feature giving AUC above 0.95, any feature correlating above 0.9 with the label, or train and validation both near-perfect while production disappoints. The forms to name are preprocessing leakage — fitting a scaler or imputer before the split, the most common by far — target leakage, where the feature is computed after the outcome, group leakage, where the same user sits on both sides of the split, and temporal leakage, where the future predicts the past. The diagnostic for that last one: compare a random split against a time-ordered holdout, and a big drop on the time split is your answer.

---

## F. Gradient checking

**28. Numerical gradient formula?**
$(f(x + \epsilon) - f(x - \epsilon))/(2\epsilon)$.

**29. PyTorch gradient check function?**
`torch.autograd.gradcheck(func, inputs)`.

**30. Acceptable relative error?**
$\leq 10^{-5}$ typically. Higher → bug.

**31. Why central difference, not forward?**
Higher-order accuracy: $O(\epsilon^2)$ vs $O(\epsilon)$.

> **Saying it out loud.** Gradient checking proves a hand-written backward pass matches the forward pass: nudge one input by epsilon each way, take the central difference of the forward output, and compare against the analytical gradient. Central rather than forward difference because the error is $O(\epsilon^2)$ instead of $O(\epsilon)$ — the first-order terms cancel — so you get several extra digits for one extra forward pass. Compare relative error, not absolute, and treat anything under about $10^{-5}$ as fine. Two gotchas worth saying: run it in float64, because in float32 the rounding noise drowns the signal, and don't check at a kink like ReLU at zero. It costs a forward pass per parameter, so it's a one-off test on a tiny tensor, never something you leave in the training loop.

---

## G. Distribution shift

**32. Detect input drift?**
PSI, KL, KS test on input feature distributions per feature.

**33. Detect output drift?**
Compare model's output distribution train vs prod.

**34. Detect label drift?**
Compare positive rates over time.

**35. Mitigation for covariate shift?**
Importance weighting, retrain on production-like data, domain adaptation.

**36. Concept drift fix?**
Retrain on fresh data.

> **Saying it out loud.** Distribution shift is when offline was honest and the world moved. I check three things in order: the input features, the model's own output score distribution, and the label rate where labels exist — the output histogram first, because it's one plot and needs no ground truth. Then I name the type, because the fix depends on it: covariate shift means the inputs moved but the input-to-label mapping held, so importance weighting or retraining on production-like data works; concept drift means the mapping itself changed and only fresh labels help. PSI is the standard per-feature drift metric, with the working thresholds being under 0.1 stable, 0.1 to 0.25 worth watching, above 0.25 a real shift.

---

## H. Production debugging

**37. Production regression — first action?**
Roll back to last good model. Then investigate.

**38. Investigation order?**
Data quality → feature pipeline diff → model regression → infra → drift.

**39. Subgroup analysis?**
Performance by user segment. Average can hide subgroup degradation.

**40. Shadow vs canary deployment?**
Shadow: run new model, discard outputs, compare. Canary: small live traffic.

> **Saying it out loud.** In production the first action isn't debugging, it's rolling back — stop the user harm, then investigate on the artifacts you saved. After that I go in likelihood order: data quality and upstream feature pipelines first, since a silently redefined feature is the most common cause, then the model itself, then infrastructure, then genuine drift. I always look at subgroups, because an aggregate metric can hold steady while one segment collapses. And the reason you deploy behind shadow or canary is precisely this: shadow runs the new model on real traffic and throws the outputs away so you can compare with zero user risk, and canary exposes a small slice so you get real outcome data. The tradeoff is exactly that — shadow is safe but can't measure user behaviour, canary measures it but puts some users on the new model.

---

## Quick fire

**41.** *First debug step?* Sanity check.
**42.** *Overfit one batch should give?* ~0 loss.
**43.** *NaN cause #1 in transformers?* FP16 attention overflow.
**44.** *Standard grad clip?* 1.0.
**45.** *Anomaly detection in PyTorch?* `set_detect_anomaly(True)`.
**46.** *Suspicious AUC threshold?* > 0.95.
**47.** *Preprocessing leakage fix?* Fit on train only.
**48.** *Production regression — step 1?* Rollback.
**49.** *Drift metric?* PSI.
**50.** *Tiny-dataset sanity?* Should overfit to ~100%.

---

## Self-grading

If you can't answer 1-15, you can't debug ML. If you can't answer 16-30, you'll get fooled by data or numerical bugs. If you can't answer 31-40, frontier-lab debugging questions will go past you.

Aim for 30+/50 cold + ability to outline a 5-min debugging plan for any failing-training scenario.
