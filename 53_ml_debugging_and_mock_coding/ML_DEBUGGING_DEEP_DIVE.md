# ML Debugging — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Senior ML interviews increasingly include "the model isn't training, debug it" — testing whether you have a systematic methodology, not just textbook knowledge. This deep dive is the methodology: how to debug failing training, NaN gradients, leakage, and production regressions.

---

## 1. The debugging tree

When something is wrong, work through layers in order. Most failures hit at one of these:

```
1. Data
2. Pipeline (preprocessing, batching, augmentation)
3. Model (architecture, init, forward pass)
4. Loss (function, label format, reduction)
5. Optimizer (LR, momentum, weight decay)
6. Training loop (gradient clip, accumulation, AMP)
7. Evaluation (metric, split, leakage)
8. Deployment (serving, infra)
```

The principle: cheap checks first. Plot losses, eyeball data, sanity-check shapes. Don't dive into custom gradient computations until you've ruled out trivial bugs.

---

## 2. Loss curve interpretation

### "Loss going down — am I done?"
No. Sanity-check:
- Train loss should decrease.
- Val loss should track or lag.
- Eventually val loss flattens or rises (overfitting).

### Common loss-curve patterns

**Loss not decreasing**:
- Bad LR (too low → no movement; too high → bouncing without progress).
- Frozen weights (forgot `requires_grad=True`).
- Disconnected gradients (broken computation graph).
- Wrong loss function or label format.
- Too-small dataset (model trivially fits but val won't improve).

**Loss exploding (NaN)**:
- LR too high.
- Numerical overflow (FP16 issue).
- Bad weight init.
- Instability in attention softmax (large logits).
- Division by zero somewhere.

**Loss decreasing but val not improving**:
- Severe overfit. More data, regularize, smaller model.
- Train-val mismatch (preprocessing, distribution).
- Leakage in train but not val (different shape of leakage).

**Loss flatlines high then jumps**:
- "Phase transition" — sometimes models break through plateau. Common in RL.
- Or warmup not finished.

**Loss decreasing then suddenly spikes**:
- Bad batch (OOD example, very long sequence, gradient cliff).
- Optimizer state went bad.
- Gradient clipping not applied.
- Re-warmup might be needed.

### Quick debugging actions
- Plot with log y-axis to see early dynamics.
- Compare loss curves across runs (changed one thing — should affect curve in expected way).
- Look at per-batch loss, not just smoothed running average.

---

## 3. Data-side debugging

### "Loss looks weird, model isn't learning"

**Sanity checks**:
- Does the data look right? Plot a few examples.
- Are labels in expected range / format?
- Are images normalized properly? (Common bug: $/255$ vs Standardization with mean/std.)
- Do tokens decode back to original text?

### Sanity check 1: overfit a single batch

Take one batch. Train on it for many steps. Loss should go to ~0.

If it can't even fit one batch:
- Wrong loss function or label format.
- Frozen layer.
- Too small a model.
- Bug in data loading.

This 5-minute test catches many issues immediately.

### Sanity check 2: tiny dataset

Take 100 examples. Train. Should overfit quickly (train accuracy ~100%).

If it doesn't, model lacks capacity or there's a fundamental bug.

### Sanity check 3: data inspection
Print 5 random batches. Look at shapes, label distributions, raw values.

If something looks off, fix that first before assuming a model issue.

---

## 4. NaN debugging

### Causes
- **FP16 overflow**: $e^x$ for $x > 88$ overflows in FP16. Use BF16 (extended exponent range) or stable softmax.
- **Division by zero**: variance estimate hits 0; output of LayerNorm; division in normalization.
- **Log of zero**: $\log p$ for $p = 0$. Add $\epsilon$ (e.g., $\log(p + 10^{-9})$).
- **Square root of negative**: numerical drift makes a "non-negative" value slightly negative.
- **Inf gradient**: explodes through layers due to bad init or large input.

### Detecting NaN early
- Loss is NaN → too late, weights already corrupted.
- Add `assert not torch.isnan(x).any()` after suspect operations.
- PyTorch has `torch.autograd.set_detect_anomaly(True)` — slow but catches first NaN site.

### Triage
- When did it appear? Step 0? Step 5000?
- Step 0: bad init or first batch issue.
- Later: optimizer instability, bad batch, explosion.

### Fix patterns
- Gradient clipping (norm 1.0).
- Lower LR.
- BF16 over FP16.
- Compute attention in higher precision.
- Add $\epsilon$ in normalizations.
- Restart from earlier checkpoint.

---

## 5. Leakage debugging

### Symptoms
- Offline metrics suspiciously high.
- Train + val accuracy both 99%, test in production poor.
- Model "feature importance" shows a feature that shouldn't exist.

### Classic forms
- **Target leakage**: feature computed using target (or post-target).
- **Train-test contamination**: same record in both splits.
- **Preprocessing leakage**: stats computed on full data.
- **Group leakage**: same user/patient on both sides.
- **Temporal leakage**: future used to predict past.

### Detection

**Suspicious AUC**:
- Train classifier with features sorted by importance; if top-1 feature alone gives AUC > 0.95, it's probably leakage.
- Train without each feature individually; one with huge drop → suspect.

**Cross-correlation check**:
- Compute correlation between every feature and the label.
- If correlation > 0.9 for any feature, audit it.

**Held-out time-period validation**:
- If you have time-stamped data, hold out the last $X\%$ as validation.
- If accuracy drops a lot relative to random split, you had temporal leakage.

### Common bug example

```python
# WRONG: scaler fit on full dataset before split
scaler = StandardScaler().fit(X)  # uses test statistics
X = scaler.transform(X)
X_train, X_test = train_test_split(X, ...)

# RIGHT: split first, fit on train only
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Gradient checking

For custom layers / losses, verify gradients numerically.

```python
def gradient_check(f, x, eps=1e-5):
    """Compare analytical gradient to numerical."""
    analytical = f.backward(x)
    numerical = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        numerical[i] = (f.forward(x_plus) - f.forward(x_minus)) / (2 * eps)
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + eps)
    return rel_error.max()
```

If rel_error > $10^{-5}$, suspect a bug.

PyTorch has `torch.autograd.gradcheck(func, inputs)`. Use it for custom autograd functions.

---

## 7. Distribution-shift debugging

### Symptoms
- Offline metrics good, online metrics bad.
- Model degrades over time.
- Subgroup performance worse than aggregate.

### Investigation
- **Compare input distributions**: train vs production. KS test, KL, PSI per feature.
- **Compare prediction distributions**: train vs production. Output histograms shifted?
- **Compare actual labels** (where available): production positive rate vs train.
- **Subgroup analysis**: performance by user segment / region / device.

### Common causes
- **Covariate shift**: new user demographics. Reweight or retrain.
- **Concept drift**: relationship $p(y|x)$ evolves. Retrain on fresh data.
- **Selection bias**: only certain populations seen offline.
- **Pipeline drift**: feature definitions changed silently.

### Mitigation
- Online retraining cadence.
- Feature monitoring with alerts.
- Shadow / canary deployment to catch regressions early.

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Loss not decreasing — first thing? | Try bigger model | Sanity-check: overfit one batch, inspect data, verify loss/label format |
| NaN appears at step 5000 — what? | Bad LR | Likely instability triggered by bad batch; gradient clip, BF16, restart |
| AUC = 0.99 — done? | Yes | Suspect leakage; check feature importance, train-test overlap |
| Model degrades in production — first check? | Retrain | Check input distribution shift first |
| Train loss low, val loss high — fix? | More layers | Overfitting: regularize, more data, smaller model |
| `loss.backward()` raised NaN — where to look? | Loss | NaN can be from any earlier op; use `set_detect_anomaly(True)` |
| Gradient clip value? | "Some number" | 1.0 typical for transformers; tune for your task |

---

## 9. Eight most-asked debugging questions

1. **Loss is not decreasing — walk through your investigation.** (Overfit one batch; verify loss; check LR; inspect data.)
2. **NaN gradient — what's your debugging process?** (When did it appear; FP16/overflow; inf in attention; gradient clip; restart from checkpoint.)
3. **Train accuracy 99%, test 60% — what's wrong?** (Severe overfit. Or leakage in train. Or distribution mismatch.)
4. **Offline AUC 0.95, online accuracy bad — what to check?** (Position bias, distribution shift, label time leakage, counterfactual issue.)
5. **Model regressed in production — investigation.** (Rollback first. Then: data, features, infra, drift.)
6. **Implement gradient checking for a custom layer.** (Numerical: $(f(x+\epsilon) - f(x-\epsilon))/(2\epsilon)$.)
7. **What does a flat loss curve mean?** (LR issue; warmup not finished; frozen weights; phase transition pending.)
8. **What sanity checks before training?** (Overfit one batch; tiny dataset to 99%; inspect data.)

---

## 10. Drill plan

- For each common loss-curve pattern (flat, exploding, val-train gap), recite cause + diagnostic + fix.
- For NaN: list 5 causes and the corresponding fix.
- For leakage: code the "fit scaler before split" bug and the corrected version.
- Practice the "overfit one batch" sanity check on a real model.
- Time yourself: 5 min to outline a debugging investigation given a vague problem.

---

## 11. Further reading

- Karpathy, *A Recipe for Training Neural Networks* (2019 blog) — the canonical practical guide.
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 11 — practical methodology.
- Smith (2018), *A disciplined approach to neural network hyperparameters.*
- Andrej Karpathy's tweets and lectures on debugging (timeless).
- *Hidden Technical Debt in ML Systems* (Sculley et al. 2015) — production-side issues.
