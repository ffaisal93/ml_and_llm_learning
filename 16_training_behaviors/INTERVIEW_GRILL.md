# Training Behaviors — Interview Grill

> 35 questions on loss curves, LR, batch size, precision, spikes. Drill until you can answer 24+ cold.

---

## A. Loss curves

**1. Healthy loss curve phases?**
Warmup → steady descent → plateau / saturation.

> **Saying it out loud.** Three phases, and you can spot them by eye. First a steep drop in the first few hundred steps while the model learns the trivially easy structure — token frequencies, punctuation. Then a long smooth descent that's roughly a straight line if you plot it on log axes. Then a flattening where each additional step buys less and less. If you don't see phase one, something is broken; if you never reach phase three, you're compute-limited rather than data-limited, and that's a good problem.

**2. Flat early loss?**
LR too low; frozen weights; broken graph; wrong loss.

> **Saying it out loud.** If loss is flat from step one, the model isn't learning at all, and that's almost never a subtle problem. Check the obvious things in order: learning rate too low by an order of magnitude, a layer accidentally frozen or `requires_grad` off, the loss not actually connected to the parameters, or the wrong loss function entirely. A quick sanity check is to try to overfit a single batch — if you can't drive the loss to near zero on ten examples, it's a plumbing bug, not a hyperparameter.

**3. Loss spikes mid-training?**
Bad batch; numerical instability; missing gradient clip.

> **Saying it out loud.** Loss spikes mid-run are common at scale and usually mean one of three things: a pathological batch, numerical instability in FP16, or gradient clipping that isn't actually turned on. The first response is not to restart from scratch. Roll back to the last checkpoint, skip the offending data, and make sure you're clipping at 1.0. If it keeps happening in the same place, it's the data, not the optimizer.

**4. Validation gap widening?**
Overfitting — regularize, more data, early stop.

> **Saying it out loud.** A widening train-validation gap means the model is memorising rather than generalising. The fixes in rough order of cost are: more data, more regularisation — weight decay or dropout — early stopping at the validation minimum, or a smaller model. For LLM pretraining you rarely see this because you're doing roughly one pass over web-scale data; it shows up hard in fine-tuning on a few thousand examples, which is exactly why fine-tuning runs are short.

**5. Periodic oscillation in loss?**
LR too high; edge of stability; bad scheduler.

> **Saying it out loud.** Regular oscillation in the loss means the step size is too big for the local curvature — you're bouncing back and forth across a valley instead of walking down it. That's the edge-of-stability regime, and the fix is lowering the learning rate or checking that your scheduler isn't doing something silly like restarting the cosine. It's distinguishable from ordinary batch noise because it's periodic rather than random.

**6. Why log y-axis for loss plots?**
Reveals early dynamics; small differences visible.

> **Saying it out loud.** Because loss improvements are multiplicative, not additive. On a linear axis the first thousand steps dominate the plot and the entire rest of the run looks like a flat line, so you can't see whether things are still improving. On a log axis a healthy run looks close to a straight line, which makes it obvious when the slope changes. It's a small thing, but it's how you actually catch a plateau early.

---

## B. Learning rate

**7. Why warmup?**
Stabilizes early gradients; optimizer state forms; prevents divergence at full LR.

> **Saying it out loud.** Because at initialisation the model is random and Adam hasn't got reliable estimates of gradient mean and variance yet, so the first updates at full learning rate can be enormous and throw you somewhere unrecoverable. Ramping in over a few thousand steps lets the optimizer state settle before you push. The failure mode is concrete: skip warmup on a large transformer and you'll often see the loss spike to NaN within the first thousand steps.

**8. Standard LLM warmup?**
~2-8k steps. Linear ramp from 0 to peak.

> **Saying it out loud.** Typically two to eight thousand steps, linearly from zero up to peak. The bigger the batch and the bigger the model, the longer you want it. It's usually a small fraction of total training — a percent or two — so erring long costs you almost nothing while erring short can cost you the whole run.

**9. Cosine decay common decay?**
Yes. Decay to 10% of peak typically.

> **Saying it out loud.** Cosine is the default for LLMs, decaying from peak down to about ten percent of peak by the end. The reason people like it is that it spends a long time near the peak and then anneals smoothly, and empirically that ends at a better loss than linear or step decay. The catch worth mentioning is that a cosine schedule is committed to a total step count up front, so if you decide to train longer you can't just continue — that's why constant-then-cooldown schedules have become popular.

**10. LR finder method?**
Sweep LR exponentially; plot loss; pick LR ~order of magnitude below divergence.

> **Saying it out loud.** You sweep the learning rate exponentially over a few hundred steps and plot loss against it. You'll see loss fall, hit a minimum, then blow up. You don't pick the minimum, you pick roughly an order of magnitude below where it diverges, because that minimum point is already unstable over a long run. It costs a few minutes of compute and saves you from a failed multi-day run.

**11. LR too high symptoms?**
Loss bouncing, NaN, gradient explosions.

> **Saying it out loud.** Bouncing loss that won't settle, gradient norms that spike rather than decay, and eventually NaNs. The tell that distinguishes it from a bad batch is that it's persistent rather than a single event. If you see it, halve the learning rate and re-warm rather than restarting cold.

**12. LR too low symptoms?**
Loss decreases very slowly; underutilized compute.

> **Saying it out loud.** A loss curve that descends steadily but far too slowly, and gradient norms that stay high because you're barely moving. It's insidious because nothing looks broken — the run is healthy, it's just wasting your compute budget. The way to catch it is to compare against a scaling-law expectation for your model size and token count rather than eyeballing the curve.

---

## C. Batch size

**13. Linear scaling rule?**
Scale LR linearly with batch size. Maintains effective per-update step.

> **Saying it out loud.** Double the batch, double the learning rate. The logic is that a larger batch is a lower-noise estimate of the same gradient, so you can trust it further, and this keeps your progress per token constant. It came from the ImageNet-in-an-hour work and it holds surprisingly well over a wide range. Where it breaks is past the critical batch size, and at very large learning rates where you need extra warmup to survive the transition.

**14. Critical batch size?**
Beyond it, doubling batch doesn't double effective progress. Task-dependent.

> **Saying it out loud.** It's the batch size past which making the batch bigger stops making training faster in wall-clock terms. Below it, your gradient is noise-dominated and more data per step genuinely helps; above it, the gradient is already accurate and you're just spending more GPUs for the same number of useful updates. It's task-dependent, from around a thousand tokens for easy tasks to millions for hard ones, and it grows as loss decreases. Practically, it's the hard ceiling on how much data parallelism can buy you.

**15. Small batch generalization advantage?**
Empirically sometimes better — implicit regularization toward flat minima.

> **Saying it out loud.** There's an empirical result that small-batch SGD generalises a bit better, with the usual explanation being that gradient noise acts as an implicit regulariser pushing you toward flatter minima. It's real for vision models on modest datasets. For LLM pretraining on web-scale data it mostly doesn't bite, because you're nowhere near overfitting anyway. So the honest answer is: known effect, largely irrelevant at the scale we actually train at.

**16. Compute cost of doubling batch?**
Linear in batch (2x compute per step) for matmul.

> **Saying it out loud.** Linear — double the batch and you do twice the matmul work per step, but each step covers twice the data, so per-token compute is flat. What you actually gain is hardware efficiency, since bigger matrices keep the GPU busier and amortise kernel launch overheads. That's the real reason people push batch size, not the math.

**17. Memory cost of doubling batch?**
Linear (2x activations); plus gradient and optimizer state if accumulating.

> **Saying it out loud.** Also linear, because activations dominate and there's one set of activations per example. Optimizer state and gradients don't grow with batch size, they grow with parameter count, so those are fixed. That's why gradient accumulation works: run several small batches, sum the gradients, step once, and you get large-batch behaviour at small-batch memory cost, paying only in wall-clock time.

---

## D. Gradient norm

**18. Healthy gradient norm trajectory?**
Steady decrease over training.

> **Saying it out loud.** It should drift downward and stay smooth. Early on it's large because the model is random, then it settles as the model fits the data. What you're watching for isn't the absolute value, which depends on your architecture and loss scale, but the shape — smooth is healthy, spiky is not, and flat-high means you're not making progress.

**19. Sudden grad norm spike?**
Bad batch; instability; need clipping.

> **Saying it out loud.** Almost always a bad batch — a chunk of repeated text, a weird encoding, a document that's mostly one token. It can also be an FP16 instability. The response is to make sure clipping is on so a single freak batch can't move the weights much, and if it recurs, go look at the actual data at that step. The failure you're preventing is one batch destroying a week of training.

**20. Grad norm dropping to ~0?**
Vanishing gradient; saturation; dead neurons.

> **Saying it out loud.** That's usually not convergence, that's vanishing gradients, and the distinction matters. Check whether your activations have saturated, whether a residual connection is missing, or whether a layer has effectively died. Genuine convergence shows up as a small but non-zero norm with loss also flat at a plausible value. A norm at exactly zero with loss stuck high is a bug.

**21. Standard clip value?**
1.0 by global norm.

> **Saying it out loud.** 1.0, on the global norm across all parameters, is the transformer default and it's robust. Global rather than per-parameter matters, because clipping per-tensor changes the direction of the update, whereas global clipping just shortens it. It costs essentially nothing and it's the single cheapest piece of insurance in the whole training stack.

**22. Per-layer grad norm tracking?**
Useful for diagnosing which layers explode / vanish.

> **Saying it out loud.** Because a global norm hides where the problem is. Logging per-layer lets you see that, say, the embedding layer is exploding while everything else is fine, or that the last few blocks are getting nothing. That immediately narrows a debugging session from the whole model to one module. It's a couple of lines of instrumentation that pay for themselves the first time something goes wrong.

---

## E. Precision

**23. FP16 vs BF16?**
BF16 has FP32-equivalent exponent range; no loss scaling needed; safer.

> **Saying it out loud.** Both are 16 bits, but they spend them differently. FP16 gives you more mantissa — more precision — with a narrow exponent range, so small gradients underflow to zero and you have to bolt on loss scaling. BF16 keeps FP32's full exponent range and sacrifices mantissa bits, so it just works with no scaling machinery. For training, range matters far more than precision, which is why BF16 is the default on anything Ampere or newer.

**24. FP16 needs loss scaling?**
Yes. Multiply loss by large constant; divide grad before optimizer step. Prevents underflow in tiny gradients.

> **Saying it out loud.** Yes, because gradients in the backward pass are often tiny and FP16 flushes anything below about ten to the minus eight to zero. So you multiply the loss by a big constant before backward, which scales all the gradients up into representable range, and divide it back out before the optimizer step. Dynamic loss scaling adjusts that constant automatically, backing off when it sees an overflow. It's all machinery you simply don't need with BF16, which is the argument for BF16 in one sentence.

**25. Why stay in FP32 for master weights?**
Numerical precision for accumulated updates. BF16 weights drift over many steps.

> **Saying it out loud.** Because updates are tiny relative to the weights. If a weight is around 1 and the update is around ten to the minus seven, in BF16 that update just rounds away entirely and the weight never moves — you silently stop learning. Keeping a FP32 master copy means those small updates accumulate properly, and you only cast down to 16-bit for the actual matmuls. That's the whole idea of mixed precision: compute in low precision, accumulate in high.

**26. FP8 training?**
Hopper/Blackwell native. Even more memory savings; needs per-tensor scale.

> **Saying it out loud.** FP8 is native on Hopper and Blackwell and roughly halves memory and bandwidth again versus BF16. The complication is that 8 bits has almost no dynamic range, so you need per-tensor scaling factors that get tracked and updated during training, and you typically keep sensitive layers in higher precision. It's real and shipping at frontier labs, not a research toy, but it's substantially more finicky to get right than the BF16 transition was.

**27. Mixed precision speed gain?**
1.5–2× on Volta+ for matmul-heavy workloads.

> **Saying it out loud.** Roughly 1.5 to 2x on matmul-heavy workloads on Volta and later, and you also roughly halve activation memory, which often lets you raise the batch size and gain again. The gain is smaller for workloads bottlenecked on memory bandwidth or on small kernels. The sanity check to run is that loss and gradient norms track the FP32 baseline for the first few hundred steps — if they diverge, something's being computed in the wrong precision.

---

## F. Loss spikes and recovery

**28. Recovery strategies for loss spike?**
Skip bad batch; restart from earlier checkpoint; lower LR; switch BF16; ensure clipping enabled.

> **Saying it out loud.** In order: don't restart from scratch. Roll back to the most recent good checkpoint, skip or filter the batches around the spike, drop the learning rate and re-warm into it, confirm gradient clipping is actually enabled, and if you're on FP16, move to BF16. That order is deliberate — it goes cheapest and most likely first. Restarting from scratch is the wrong answer and interviewers listen for it.

**29. Prevention for spikes?**
Gradient clipping, warmup, BF16, periodic checkpoints for fast rollback.

> **Saying it out loud.** Clip at 1.0, use BF16, warm up long enough for your batch size, and checkpoint often enough that a rollback costs an hour rather than a week. Data filtering matters too, since most spikes trace back to a pathological document. None of this is clever, which is the point — spike prevention is discipline, not insight.

**30. Detect bad batch?**
Loss > some threshold (e.g., 10× recent moving average). Skip update.

> **Saying it out loud.** Compare the batch loss to a moving average of recent losses and skip the update if it's some multiple higher — ten times is a common threshold. You can also trigger on gradient norm rather than loss. The important part is that you skip the update entirely rather than clipping it, because a genuinely pathological batch shouldn't influence the weights at all. Log which batches got skipped, because a pattern there tells you about your data pipeline.

---

## G. Overfitting and forgetting

**31. Overfitting signal?**
Train loss decreasing, val loss flat or increasing.

> **Saying it out loud.** Training loss keeps falling while validation loss flattens and then turns upward. The turning point is where you'd early-stop. Note that this only tells you something if your validation set is big enough to be stable — with a few hundred examples the curve is mostly noise and you'll chase phantoms. For generative models, the sharper test is whether the model reproduces training examples verbatim.

**32. Mitigations?**
Weight decay, dropout, augmentation, early stopping, smaller model, more data.

> **Saying it out loud.** Weight decay and dropout are the cheap first moves, then data augmentation, then early stopping at the validation minimum. Beyond that: fewer trainable parameters, which is basically why LoRA is so popular for fine-tuning, or simply more data, which is the only fix that raises the ceiling rather than lowering the variance. On small fine-tuning sets the single most effective lever is usually just training for fewer epochs.

**33. Catastrophic forgetting?**
New task wipes out old capability. Common in fine-tuning.

> **Saying it out loud.** It's when training on something new erases something old — you fine-tune on customer support transcripts and the model quietly loses its coding ability. It happens because gradient descent has no incentive to preserve behaviours that nothing in the new data is measuring. The failure mode is invisible if you only evaluate the target task, which is exactly how it gets shipped. So the operational rule is to keep a broad benchmark in the loop during specialised training.

**34. Replay buffer for forgetting?**
Blend mostly new data with a small replay fraction of original pre-training mix (typically 5–15% old / 85–95% new) so old capabilities aren't forgotten while new domain is learned.

> **Saying it out loud.** You mix a slice of the original pretraining distribution back into the new data — usually 5 to 15 percent old, the rest new. It's crude and it works remarkably well, because you're just making sure the gradient still cares about the old capabilities. It's cheaper and more reliable than the clever alternatives. The tradeoff is that replay data displaces new data, so you learn the new domain slightly slower in exchange for not losing the old one.

**35. EWC (Elastic Weight Consolidation)?**
Regularize toward old weights, weighted by Fisher information. Prevents drift on important parameters.

> **Saying it out loud.** Elastic Weight Consolidation adds a penalty for moving weights away from their old values, with the penalty weighted by how important each weight was to the old task — importance being estimated by the Fisher information. So unimportant weights are free to move and critical ones are anchored. It's elegant and it comes up in interviews, but in practice for LLMs people reach for replay data or LoRA instead, because estimating the Fisher over billions of parameters is expensive and the payoff is smaller than just mixing in old data.

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
