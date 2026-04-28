# Training Behaviors — Interview Grill

> 35 questions on loss curves, LR, batch size, precision, spikes. Drill until you can answer 24+ cold.

---

## A. Loss curves

**1. Healthy loss curve phases?**
Warmup → steady descent → plateau / saturation.

**2. Flat early loss?**
LR too low; frozen weights; broken graph; wrong loss.

**3. Loss spikes mid-training?**
Bad batch; numerical instability; missing gradient clip.

**4. Validation gap widening?**
Overfitting — regularize, more data, early stop.

**5. Periodic oscillation in loss?**
LR too high; edge of stability; bad scheduler.

**6. Why log y-axis for loss plots?**
Reveals early dynamics; small differences visible.

---

## B. Learning rate

**7. Why warmup?**
Stabilizes early gradients; optimizer state forms; prevents divergence at full LR.

**8. Standard LLM warmup?**
~2-8k steps. Linear ramp from 0 to peak.

**9. Cosine decay common decay?**
Yes. Decay to 10% of peak typically.

**10. LR finder method?**
Sweep LR exponentially; plot loss; pick LR ~order of magnitude below divergence.

**11. LR too high symptoms?**
Loss bouncing, NaN, gradient explosions.

**12. LR too low symptoms?**
Loss decreases very slowly; underutilized compute.

---

## C. Batch size

**13. Linear scaling rule?**
Scale LR linearly with batch size. Maintains effective per-update step.

**14. Critical batch size?**
Beyond it, doubling batch doesn't double effective progress. Task-dependent.

**15. Small batch generalization advantage?**
Empirically sometimes better — implicit regularization toward flat minima.

**16. Compute cost of doubling batch?**
Linear in batch (2x compute per step) for matmul.

**17. Memory cost of doubling batch?**
Linear (2x activations); plus gradient and optimizer state if accumulating.

---

## D. Gradient norm

**18. Healthy gradient norm trajectory?**
Steady decrease over training.

**19. Sudden grad norm spike?**
Bad batch; instability; need clipping.

**20. Grad norm dropping to ~0?**
Vanishing gradient; saturation; dead neurons.

**21. Standard clip value?**
1.0 by global norm.

**22. Per-layer grad norm tracking?**
Useful for diagnosing which layers explode / vanish.

---

## E. Precision

**23. FP16 vs BF16?**
BF16 has FP32-equivalent exponent range; no loss scaling needed; safer.

**24. FP16 needs loss scaling?**
Yes. Multiply loss by large constant; divide grad before optimizer step. Prevents underflow in tiny gradients.

**25. Why stay in FP32 for master weights?**
Numerical precision for accumulated updates. BF16 weights drift over many steps.

**26. FP8 training?**
Hopper/Blackwell native. Even more memory savings; needs per-tensor scale.

**27. Mixed precision speed gain?**
1.5–2× on Volta+ for matmul-heavy workloads.

---

## F. Loss spikes and recovery

**28. Recovery strategies for loss spike?**
Skip bad batch; restart from earlier checkpoint; lower LR; switch BF16; ensure clipping enabled.

**29. Prevention for spikes?**
Gradient clipping, warmup, BF16, periodic checkpoints for fast rollback.

**30. Detect bad batch?**
Loss > some threshold (e.g., 10× recent moving average). Skip update.

---

## G. Overfitting and forgetting

**31. Overfitting signal?**
Train loss decreasing, val loss flat or increasing.

**32. Mitigations?**
Weight decay, dropout, augmentation, early stopping, smaller model, more data.

**33. Catastrophic forgetting?**
New task wipes out old capability. Common in fine-tuning.

**34. Replay buffer for forgetting?**
Blend mostly new data with a small replay fraction of original pre-training mix (typically 5–15% old / 85–95% new) so old capabilities aren't forgotten while new domain is learned.

**35. EWC (Elastic Weight Consolidation)?**
Regularize toward old weights, weighted by Fisher information. Prevents drift on important parameters.

---

## Quick fire

**36.** *Standard warmup?* 2-8k steps.
**37.** *Decay shape?* Cosine.
**38.** *Decay to?* ~10%.
**39.** *Linear scaling on batch?* Scale LR.
**40.** *FP16 needs?* Loss scaling.
**41.** *BF16 needs?* Nothing extra.
**42.** *Clip value?* 1.0.
**43.** *Critical batch?* Saturation point.
**44.** *Replay ratio for mid-training?* ~5–15% old / 85–95% new.
**45.** *Spike — first action?* Restart from checkpoint.

---

## Self-grading

If you can't answer 1-15, you don't understand training dynamics. If you can't answer 16-30, you'll get tripped up on practical questions. If you can't answer 31-45, frontier-lab interviews on training behavior will go past you.

Aim for 28+/45 cold.
