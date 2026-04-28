# Training Behaviors — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This deep dive complements the debugging deep dive (`53`). Where debugging is about diagnosing failures, this is about *expected behavior*: what should training look like, what's normal vs concerning, what early signals tell you about your run's trajectory? Senior ML interviews probe this to test if you've actually trained models, not just read about them.

---

## 1. The classical loss curve

A healthy training run typically shows:

1. **Brief warmup phase**: loss drops rapidly from initialization to "easy" loss.
2. **Steady descent**: loss decreases smoothly, possibly with small noise.
3. **Plateau or saturation**: loss flattens; further training has diminishing returns.
4. **Validation curve**: tracks training but with a small gap; may eventually rise (overfit).

### Concerning patterns

**Loss plateau at high value early**: under-capacity, bad LR, frozen weights.

**Loss spikes**: gradient explosion, bad batch, optimizer instability.

**Validation gap widens monotonically**: overfitting; need regularization, early stopping, more data.

**Periodic oscillation**: LR too high, edge of stability, bad scheduler.

---

## 2. Learning rate behavior

### LR is the most important hyperparameter

For a fixed architecture, LR change of 2-3× often makes the difference between divergence and a state-of-the-art model.

### LR finder (Smith 2017)

Sweep LR exponentially over a few hundred steps; plot loss. Pick LR roughly an order of magnitude below the loss-divergence point.

### Warmup

Linear ramp from 0 to peak LR over first $K$ steps. Standard for transformers (large models destabilize with sudden full LR).

Why: in early training, gradients are small and weights move dramatically per step at full LR; instability ensues.

### Decay

After warmup, LR typically decays:
- **Cosine**: smooth, popular for LLMs.
- **Linear**: simpler.
- **Step**: drop by factor at fixed steps (older convention).
- **Constant**: rare for production but used for some pre-training segments.

LLMs typically decay to ~10% of peak.

### What different LR regimes look like

**Too low**: loss decreases very slowly; underutilized compute.

**Too high**: loss bounces, possibly diverges; high gradient norms.

**Sweet spot**: smooth descent; gradient norm decreases over time.

---

## 3. Batch size effects

### What batch size affects
- **Per-step compute cost**: linear in batch size.
- **Gradient noise**: variance scales as $1/B$.
- **Effective LR**: with linear scaling rule, scale LR linearly with batch.
- **Generalization**: smaller batches sometimes generalize better (implicit regularization).
- **Memory**: linear in batch size.

### Linear scaling rule (Goyal et al. 2017)

When you scale batch size by $k$, scale LR by $k$. Maintains effective per-update step.

### Critical batch size

Beyond a certain batch size, doubling batch doesn't double effective progress (McCandlish et al. 2018). Critical batch is task-dependent; varies from 1k to millions of tokens.

### Generalization gap (Keskar et al. 2017)

Empirically, very large batches sometimes generalize worse. Hypothesis: small-batch SGD finds flatter minima.

In practice: for modern LLMs trained on web data, large batch + linear-scaled LR works fine.

---

## 4. Gradient norm and stability

### Gradient norm = signal of training health

- **Steady decrease**: training is converging.
- **Sudden spike**: instability brewing; bad batch, optimizer state issue.
- **Plateau at high value**: model isn't reducing loss but isn't diverging either.
- **Drops to ~0**: vanishing gradients (saturation, dead neurons).

### Tracking
Log gradient norm per layer / parameter group. Different layers may behave differently.

### Gradient clipping

$\nabla \to \nabla \cdot \min(1, \tau / \|\nabla\|)$. Caps norm at $\tau$ (typically 1.0 for transformers).

Prevents explosions from rare bad batches. Standard for any non-trivial training.

---

## 5. Weight magnitude evolution

### Healthy training

Weights start small (init), gradually grow as training fits the data. After training, weights have much larger magnitudes than at init.

### Pathological patterns

**Weights staying near init**: model not training (LR too low or frozen).

**Weights exploding**: instability or no weight decay.

**Layer-norm gamma exploding**: common transformer issue; some implementations clip.

**Sparse weights (lots of near-zero)**: lottery-ticket-style; or aggressive $\ell_1$.

### Weight decay role
$w \leftarrow w - \eta \cdot (\nabla + \lambda w)$. Pulls weights toward zero. Prevents unbounded growth.

In modern Adam/AdamW, decoupled from gradient (correctly implemented in AdamW).

---

## 6. Validation behavior

### Train-val gap

| Pattern | Interpretation |
|---|---|
| Both decreasing, small gap | Healthy |
| Train decreasing, val flat then up | Overfitting |
| Both flat at high value | Underfitting; LR low; capacity issue |
| Val below train | Possible: val set easier; or train has more noise (e.g., dropout) |
| Big oscillation in val | Val set too small; or training too noisy |

### Early stopping

Save model at minimum val loss. Stop if no improvement for $K$ epochs.

Best practice: also save final model; sometimes "best val" is the wrong checkpoint due to noise.

### Why train > val isn't surprising sometimes
- During training, dropout makes train loss higher (via injected noise).
- Train loss measured with augmentation; val without.
- BatchNorm statistics differ (train uses batch; val uses running mean).

---

## 7. Mixed precision training

### What it is
Forward / backward in BF16 or FP16; weights and optimizer state in FP32.

### Common pitfalls
- **FP16 overflow**: switch to BF16.
- **Loss scaling**: needed for FP16 to prevent gradient underflow. Not needed for BF16.
- **Precision mismatches**: some ops (e.g., LayerNorm) need higher precision.

### What to check
- Loss similar to FP32 baseline.
- Gradient norms similar.
- No NaN.
- 1.5–2× speedup on Volta+ GPUs.

### FP8 training
Hopper / Blackwell GPUs natively. Even more memory savings; needs careful per-tensor scale management. Frontier feature in 2024+.

---

## 8. Loss spikes and recovery

### Causes
- Bad batch (extreme outlier).
- Numerical instability (FP16 limits).
- Gradient cliff (sudden curvature increase).
- Optimizer state mismatch.

### Symptoms
- Sudden 10x loss jump.
- Possibly NaN.
- Gradient norm spikes.

### Recovery strategies
- **Skip the bad batch**: detect via threshold; don't update.
- **Restart from earlier checkpoint**: known-good state.
- **Lower LR**: temporarily, then re-warmup.
- **Gradient clip**: if not already on, enable.
- **Switch to BF16** if running FP16.

### Prevention
- Gradient clipping (1.0 is robust).
- Warmup (longer for big batch).
- BF16.
- Periodic checkpoint to enable easy rollback.

---

## 9. Overfitting timeline

### Capacity vs data

Small dataset + big model → overfits fast.

Large dataset + big model → may never finish underfit phase before training ends.

LLMs operate in over-parameterized regime: many epochs of pre-training data but loss still decreasing → reasonable to continue.

### Detecting overfit
- Val loss stops decreasing while train continues.
- Performance on held-out worse than benchmark on training data.
- Memorization tests: model can recite training samples.

### Mitigations
- Regularization (weight decay, dropout).
- Augmentation.
- Early stopping.
- Smaller model.
- More data.

---

## 10. Catastrophic forgetting

### What it is
Training on new task / data wipes out previously learned capability.

### When it shows up
- Fine-tuning on narrow task → general capability degrades.
- Mid-training on focused data → pre-training capabilities lost.
- Sequential RL tasks.

### Detection
- Periodic eval on broad benchmarks during specialized training.
- Track multiple metrics, not just target task.

### Mitigations
- **Replay**: blend mostly new data with a small replay fraction (~5–15%) of original pre-training mix.
- **EWC** (Elastic Weight Consolidation): regularize toward old weights weighted by Fisher information.
- **LoRA / adapters**: train small additional parameters; preserve base.
- **Smaller LR**: less aggressive update on existing weights.

---

## 11. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Why warmup? | Tradition | Stabilizes early training where gradients can be large; prevents divergence |
| LR too high — symptom? | Slow training | Loss bouncing, NaN, gradient explosions |
| Gradient norm dropping to 0? | Convergence | Possibly vanishing gradient; check activations + Hessian |
| Train loss > val loss — possible? | No | Yes: dropout noise, augmentation, BN stats |
| Linear scaling rule limit? | None | Critical batch size — beyond it, no further parallel speedup |
| FP16 vs BF16? | Same | BF16 has FP32 exponent range; FP16 needs loss scaling |
| Loss spike — what to do? | Restart from scratch | Restart from last good checkpoint; skip bad data; lower LR; gradient clip |

---

## 12. Eight most-asked behavior questions

1. **Walk me through what a healthy loss curve looks like.** (Warmup, descent, plateau; train + val tracking.)
2. **Why use LR warmup?** (Stabilizes early gradient instability; lets optimizer state form.)
3. **LR too high — what symptoms?** (Bouncing, divergence, NaN.)
4. **What's the linear scaling rule?** (Scale LR linearly with batch size; bounded by critical batch.)
5. **FP16 vs BF16 — when to use which?** (FP16 needs loss scaling; BF16 doesn't; BF16 default.)
6. **Loss spike — what's the playbook?** (Skip bad batch; restart from checkpoint; lower LR; clip; switch to BF16.)
7. **Catastrophic forgetting — how to mitigate?** (Replay buffer; EWC; LoRA; smaller LR.)
8. **Why might train loss exceed val loss?** (Dropout noise; augmentation; BN stats differ; val set is easier.)

---

## 13. Drill plan

- For each loss-curve pattern, recite cause + fix.
- Recite warmup duration, LR peak, decay schedule for: tiny model (1B), medium (10B), flagship (70B+).
- For each precision (FP32, FP16, BF16, FP8), recite when to use + pitfall.
- Practice talking through a single training run end-to-end (warmup, peak, decay, recovery from spike).
- Memorize critical batch size order of magnitude per task type.

---

## 14. Further reading

- Smith (2017), *Cyclical Learning Rates for Training Neural Networks.*
- Goyal et al. (2017), *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (linear scaling rule).
- Keskar et al. (2017), *On Large-Batch Training: Generalization Gap and Sharp Minima.*
- McCandlish et al. (2018), *An Empirical Model of Large-Batch Training* (critical batch size).
- Cohen et al. (2021), *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.*
- Karpathy, *A Recipe for Training Neural Networks* (2019 blog).
