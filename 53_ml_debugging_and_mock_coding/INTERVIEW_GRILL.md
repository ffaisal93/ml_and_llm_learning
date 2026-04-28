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

---

## D. NaN debugging

**14. FP16 overflow at $x > ?$**
~88. $e^{88}$ exceeds FP16 max.

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
