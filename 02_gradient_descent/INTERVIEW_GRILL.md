# Gradient Descent & Learning Rate — Interview Grill

> **How to use this file.** Cover the answer with your hand. Read the question out loud. Speak your answer out loud, completely, before checking. If your answer is more than 25% off, mark it; come back tomorrow. The answers below are interview-ready: the level of detail you should be able to produce on a whiteboard or in a Zoom screen-share, not just the gist.

---

## A. Foundations (warmup; you must get all of these)

**1. Write down the SGD update rule.**
$\theta_{t+1} = \theta_t - \eta \cdot g_t$, where $g_t = \nabla L(\theta_t; \text{batch}_t)$ is the stochastic gradient computed on a mini-batch. $\eta$ is the learning rate.

**2. What is the difference between batch, stochastic, and mini-batch gradient descent?**
Batch GD computes the gradient over the entire training set per step — expensive but exact in expectation. SGD uses a single sample per step — cheap, very noisy. Mini-batch uses $B$ samples per step (typical 32–8192) and is the practical default because it amortizes per-step overhead, gives reasonable gradient estimates, and matches accelerator hardware. The variance of the mini-batch gradient is roughly $\sigma^2 / B$, where $\sigma^2$ is the per-sample variance.

**3. What does the learning rate control?**
The step size in parameter space. For a quadratic loss with Hessian $H$, GD converges if $0 < \eta < 2/\lambda_{\max}(H)$. Above that, iterates diverge. The optimal rate for a quadratic is $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, and convergence rate is governed by the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$.

**4. What happens if your learning rate is too high?**
Updates overshoot in the sharpest curvature direction, iterates oscillate with growing amplitude, loss becomes NaN within steps to hundreds of steps. In Adam, you may also see "loss spikes" that recover — symptoms of operating past the stability edge.

**5. What happens if your learning rate is too low?**
Convergence is slow; you may run out of compute budget before reaching a useful loss. You may also fail to escape saddle points or shallow local minima. Per-layer update-to-weight ratio drops below $10^{-4}$.

**6. What's the relationship between gradient descent and Newton's method?**
Newton's method is $\theta_{t+1} = \theta_t - H_t^{-1} g_t$. It uses second-order curvature information to choose step direction *and* size. GD is the first-order approximation: $H_t^{-1} \approx \eta \cdot I$, i.e. assume the loss is locally a sphere and use a fixed scalar. Newton converges quadratically near a minimum; GD converges linearly with rate determined by $\kappa$. Adam is a coarse *diagonal Fisher* approximation: $\mathbb{E}[g g^\top]$ estimates the Fisher information matrix, which equals the Hessian only at a stationary point of negative log-likelihood. So calling Adam "diagonal Newton" is loose — strictly it's diagonal natural gradient.

---

## B. Learning rate behavior

**7. Why is no single learning rate good for the whole network?**
Because the curvature of the loss varies across layers, across parameters within a layer, and across training phases. Embedding tables and early-layer features are typically ill-conditioned compared to later layers. A global $\eta$ is necessarily a compromise — too small for flat directions, too large for sharp ones. This is exactly the problem Adam, AdamW, LARS, LAMB, and muP all attempt to fix.

**8. Why does the same model on the same data sometimes need different learning rates on different hardware?**
Because batch size depends on memory and parallelism, and the optimal LR depends on batch size. Linear scaling rule: scaling batch by $k$ requires scaling $\eta$ by $k$ (SGD) or roughly $\sqrt{k}$ (Adam) to maintain trajectory. Different hardware setups also produce slightly different numerics (fp16/bf16/fp32 mix), which can change the stable LR range.

**9. Walk me through how you'd find the right learning rate from scratch.**
LR finder (Leslie Smith): start $\eta$ very low ($10^{-7}$), multiply by ~1.1 per step, plot loss vs. $\log \eta$. Pick $\eta$ an order of magnitude below the divergence point — i.e. where loss is still descending steeply. Verify by checking update-to-weight ratio is around $10^{-3}$ per layer in the first few thousand steps. Adjust schedule from there.

**10. Why is the update-to-weight ratio ($\|\eta \cdot \text{update}\| / \|\theta\|$) a better diagnostic than the loss curve?**
Because it works per layer and reveals heterogeneity that a global loss curve hides. A loss curve might look fine while one layer is stuck at $10^{-6}$ and another is exploding at $10^{-1}$. A healthy training run has the ratio around $10^{-3}$ for most layers. Karpathy's heuristic, widely cited.

**11. What is the edge of stability phenomenon?**
**Loss spikes can be a feature, not a bug.** Cohen et al. 2021: in deep-net training, the largest Hessian eigenvalue drifts upward until it pins near $2/\eta$ — exactly the GD stability boundary. Loss bounces non-monotonically but globally descends. Classical convergence theory (smooth descent to minimum) doesn't describe real training.

**12. If your loss spikes occasionally during pretraining but training overall is healthy, what do you do?**
Don't panic. Add gradient clipping at norm 1.0 if not already present. Investigate the spike batch (sometimes a single bad sequence). Don't reflexively lower the global LR — that may actually slow you below the edge of stability and waste compute. The spike is often a feature of operating at the optimal LR. Lower LR only if spikes are getting worse over time or producing NaNs.

**13. Why do you need learning-rate warmup?**
Three reasons: (a) Adam's second-moment $\hat v$ is noisy and biased low for the first few hundred steps, so updates are oversized; (b) residual streams in transformers are not yet calibrated near init, gradients are oversized; (c) the loss landscape near random init can be pathologically sharp. Without warmup, transformers near-deterministically diverge at modern scales. Typical warmup is 0.5–5% of total steps; 2000 steps for LLM pretraining is common.

**14. What's the linear scaling rule?**
Goyal et al. 2017: when you multiply batch size by $k$, multiply $\eta$ by $k$ to maintain training dynamics — for SGD, with warmup. The intuition is that per-epoch progress depends on $\eta \cdot k$. It works on ResNets up to batch size ~8192, and breaks at very large batches (the "critical batch size"). For Adam, sqrt scaling is closer.

**15. What is the critical batch size?**
**The batch size where doubling stops paying off.** McCandlish et al. 2018: beyond this point, even with optimal LR rescaling, you don't get proportional speedups. Determined by the gradient noise scale (mean gradient magnitude / gradient covariance trace). Importantly, it *grows during training*, so the right batch size is not static. Frontier labs track this because it determines data-parallel ROI.

**16. How would you transfer a learning rate from a 1B-parameter model to a 70B-parameter model?**
Use muP (Yang & Hu 2022), which scales initialization, embedding, and per-layer learning rates so the optimal LR is width-invariant. Sweep $\eta$ cheaply on small models, scale up. Without muP, large models often need lower $\eta$, and the ratio depends on width, depth, and details of the architecture in a non-trivial way.

---

## C. Schedules

**17. What's the standard LR schedule for LLM pretraining?**
Linear warmup over ~0.5–2% of steps, then cosine decay to ~10% of peak. Some recent runs use linear or trapezoidal decay instead of cosine. Pure constant LR is rare in pretraining; pure step decay is dated.

**18. Why cosine decay specifically?**
Smooth — no sudden drops that shock the optimizer. Spends roughly half the budget at relatively high LR (where most progress happens) and gradually transitions to fine-grained polishing. Empirically dominates step decay on most large-scale tasks. Linear decay is competitive and easier to reason about for compute-budget scaling.

**19. What's `ReduceLROnPlateau` and when is it useful?**
A schedule that drops $\eta$ (typically by 10x) when validation loss has not improved for $N$ epochs/steps. Useful for fine-tuning, transfer learning, and any setting where you can't predict the optimal total schedule in advance. Less common in pretraining where compute budget is fixed and cosine works well.

**20. Should $\eta_{\min}$ at the end of cosine be zero or positive?**
Positive, typically $0.1 \cdot \eta_{\max}$. Pure zero LR at the tail provides no gradient signal and may cause optimizer state to drift. Recent practice keeps $\eta_{\min}$ non-trivial for stability.

---

## D. Optimizers

**21. What does Adam do that SGD does not?**
Tracks per-parameter first moment $m_t$ (gradient mean) and second moment $v_t$ (gradient squared mean). Updates use $\hat m_t / (\sqrt{\hat v_t} + \varepsilon)$, which gives per-parameter step sizes adapted to per-parameter gradient magnitudes. This makes Adam much more tolerant to ill-conditioned problems and removes most of the LR-tuning sensitivity that SGD has. It also incorporates momentum.

**22. Walk me through Adam with bias correction.**
**Verbal story**: "Adam = momentum (first moment) + per-parameter scaling by RMS gradient (second moment) + a small correction so the first few steps aren't biased toward zero."

**Math**:
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad &\text{(first moment / momentum)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad &\text{(second moment / RMS)} \\
\hat m_t &= m_t / (1 - \beta_1^t),\quad \hat v_t = v_t / (1 - \beta_2^t) \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \hat m_t / (\sqrt{\hat v_t} + \varepsilon)
\end{aligned}
$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$. Bias correction matters for the first few thousand steps; afterward $1 - \beta^t \approx 1$.

**23. Why does Adam need bias correction?**
The moving averages are initialized at zero. With $\beta_1 = 0.9$, $m_t$ is biased toward zero for the first ~10 steps; with $\beta_2 = 0.999$, $v_t$ is biased toward zero for the first ~1000. Without correction, the early effective LR is wrong: small $\hat v$ would inflate the update, but small $\hat m$ would deflate it. Bias correction $1/(1 - \beta^t)$ recovers the intended estimate magnitude.

**24. What does $\varepsilon$ in Adam actually do?**
Numerical floor: prevents division by zero when $\hat v_t$ is tiny, and caps the maximum effective per-parameter LR. When $\sqrt{\hat v} \ll \varepsilon$, the update is roughly $(\eta / \varepsilon) \cdot \hat m$, so dimensions with very small gradients still get sensible updates. Some recipes set $\varepsilon = 10^{-3}$ or higher (especially for embeddings) to dampen aggressive updates on rarely-touched parameters.

**25. How is AdamW different from Adam with L2 regularization?**
Naive L2 adds $\lambda \theta$ to the gradient. Adam then divides everything by $\sqrt{\hat v}$, so the L2 contribution gets weakened wherever $\hat v$ is large — i.e. regularization is no longer uniform across parameters, contrary to intent. AdamW decouples weight decay: $\theta_{t+1} = \theta_t - \eta \cdot \hat m / (\sqrt{\hat v} + \varepsilon) - \eta \cdot \lambda \cdot \theta_t$. Weight decay is now a uniform fractional shrinkage independent of gradient statistics, recovering the regularization behavior people thought they were getting.

**26. For SGD, are L2 and weight decay the same thing?**
Yes. The gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda \theta$, so SGD with explicit weight decay is identical to SGD with L2 added to the loss. They diverge only when there is preconditioning (Adam, RMSProp, etc.).

**27. What's RMSProp and how does it relate to Adam?**
RMSProp = Adam's second moment without the first. $v_t = \beta v_{t-1} + (1 - \beta) g_t^2$, then update by $\eta g_t / \sqrt{v_t}$. It adapts per-parameter LR by gradient variance but doesn't use momentum. Adam = RMSProp + momentum + bias correction. RMSProp is a useful building block to remember; Adam dominates in practice.

**28. What's Lion and when might you prefer it?**
Lion (Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*) uses $\mathrm{sign}(\text{momentum})$ instead of $\hat m / \sqrt{\hat v}$. It's roughly half the memory of Adam (no second moment) and sometimes matches or beats AdamW on language modeling. The trade-off: it's more sensitive to LR and weight decay choice; it works best at smaller LR values (3–10x smaller than Adam). Frontier-lab interviews increasingly ask about it.

**29. Why does SGD with momentum sometimes generalize better than Adam?**
Several stories, none fully settled. (a) Adam's per-parameter rescaling can overfit to the noise in $\hat v$, biasing toward sharper minima. (b) SGD's gradient noise is a stronger implicit regularizer toward flat minima. (c) Adam's effective trajectory differs from SGD's, and the geometric properties of the resulting solutions differ. Result: for many vision tasks, well-tuned SGD+momentum wins on validation. For large language models, AdamW dominates because of conditioning issues SGD can't handle.

**30. What's the "Adam generalization gap" and how is it usually mitigated?**
The empirical observation that Adam achieves equal or lower training loss than SGD but worse validation accuracy on some tasks. Mitigations: AdamW (decoupled decay), longer training, explicit regularization, switching to SGD for the final phase (the "AdamSwitch" trick).

---

## E. Stability, scaling, and noise

**31. What is gradient clipping and when do you use it?**
$g \leftarrow g \cdot \min(1, c / \|g\|)$. Caps gradient norm at $c$ (clip-by-norm; clip-by-value also exists). Use it (a) to absorb occasional gradient spikes from bad batches, (b) almost always for RNNs (which have exploding-gradient issues), (c) standard in LLM pretraining at $c = 1.0$. Different from LR: clipping targets outliers; LR scales everything.

**32. Why does gradient noise help generalization?**
Implicit regularization: SGD biases toward flat minima. Three views — escape mechanism (steep walls in sharp minima get rejected by noise), Bayesian view (SGD samples from a posterior with temperature $\eta/B$), explicit view (SGD $\approx$ GD on $L + (\eta/4B) \cdot \|\nabla L\|^2$, an extra penalty on high-gradient regions). The relevant quantity is $\eta/B$; raising batch size without raising LR removes this regularization.

**33. Why is large-batch training hard?**
Several reasons compound: (a) optimal $\eta$ scaling is non-trivial (linear for SGD, sqrt for Adam, both break at very large batches); (b) implicit regularization weakens as $\eta/B$ shrinks; (c) per-step gradient is closer to the true gradient, so the optimizer makes more deterministic progress along sharp directions, which can hurt generalization. McCandlish's critical batch size puts a hard ceiling on practical parallelism.

**34. What is the gradient noise scale?**
Roughly $\mathrm{tr}(\Sigma) / \|\mu\|^2$, where $\Sigma$ is the gradient covariance and $\mu$ is the mean gradient. Large noise scale → stochasticity dominates → small batch is fine. Small noise scale → gradient is well-estimated → larger batch helps. Critical batch size is approximately the noise scale. Importantly, noise scale grows during training, so optimal batch size grows as you train.

**35. What is muP and why does it matter for frontier labs?**
Maximal update parameterization (Yang & Hu 2022). A specific scaling of initialization, learning rate per layer, and embedding multipliers such that the optimal LR is *invariant under model width*. Practically: tune hyperparameters on a small model under muP, scale up to 70B with the same $\eta$. Without muP, hyperparameters sweep at 70B is prohibitively expensive and small-model hyperparameters don't transfer.

**36. What's the difference between LARS, LAMB, and muP?**
LARS/LAMB enforce a layer-wise trust ratio at runtime: $\eta_{\text{layer}} = \eta \cdot \|\theta_{\text{layer}}\| / \|\text{update}_{\text{layer}}\|$. They make large-batch training stable by keeping update-to-weight ratio constant per layer. muP changes the parameterization (init scales and per-layer LR factors) so that the right thing happens automatically, without runtime trust-ratio computation. muP is more elegant; LAMB is older and explicit.

---

## F. Diagnostics and debugging (stuff that wins offers)

**37. Loss is flat. Is this a learning-rate problem?**
Maybe. Check: gradient norm. If $\|g\| \to 0$, you're stuck at a critical point; consider warm restart or perturbation. If $\|g\|$ is healthy but loss isn't moving, your $\eta$ is too small. If $\|g\|$ is huge but loss isn't moving, you're oscillating across a sharp ridge — $\eta$ is too large in that direction. Per-layer update-to-weight ratio narrows it down.

**38. Loss diverges to NaN at step 1.**
Almost always one of: LR way too high, fp16 overflow in the forward pass (not optimizer-related; check activation magnitudes), bad initialization, or division-by-zero somewhere (softmax of all-$-\infty$, or $\sqrt{0}$ without $\varepsilon$). Lower LR by 10x first; if still NaN, suspect non-LR causes.

**39. Loss diverges to NaN around step 100–500.**
Classic warmup-end signature. The peak $\eta$ is too high; add or extend warmup. Also possible: a single bad batch with extreme gradient norm — gradient clipping fixes it. Also possible: Adam $\hat v$ accumulating an outlier — same fix.

**40. Loss decreases on training but eval loss increases (overfitting).**
Not directly an LR question, but interviewers often ask whether LR can fix it. Lower LR may help by under-fitting; better answers add regularization (weight decay, dropout), early stopping, or data augmentation. The cleanest answer: LR shouldn't be your first lever for overfitting.

**41. Different layers learn at very different rates. What do you do?**
Per-layer LR (LARS/LAMB or muP-style scaling). For Adam, increase $\varepsilon$ for the lagging layer; for SGD, give it a higher per-layer multiplier. In practice, "BERT layer-wise decay" (lower LR for earlier layers during fine-tuning) is a related, simpler fix.

**42. The loss spikes once and recovers. Should I lower the LR?**
Probably not as a first response. Add or tighten gradient clipping. Investigate the spike batch (sometimes a single token sequence). Edge of stability predicts that operating at the optimal $\eta$ *will* produce occasional spikes. Lower the LR only if spikes get worse over time or actually NaN.

**43. I changed batch size and now training is unstable. What's the first thing you check?**
That $\eta$ was scaled appropriately. For Adam, sqrt scaling. For SGD, linear scaling. Then check warmup length — bigger batch = need longer warmup because each step is bigger. Then check for fp16 issues if batch size change increased pre-softmax scores.

**44. How do you debug "training is fine but slow"?**
First confirm "slow" by measuring per-step time and per-token loss decrease. If per-step time is the bottleneck, it's a systems issue (data loading, GPU utilization). If per-step loss decrease is too slow, your $\eta$ is probably too low (check update-to-weight ratio) or your batch size is below the noise scale (using too many noisy gradients per parameter update).

**45. How do you decide between Adam and SGD?**
SGD with momentum if (a) you can afford to tune LR carefully, (b) the problem is well-conditioned (CNNs on vision), (c) generalization is paramount. Adam(W) if (a) the problem is ill-conditioned (transformers, embeddings), (b) you need robustness to LR choice, (c) you have heterogeneous gradients across parameters. For LLM training the answer is always AdamW or a successor (Lion, Sophia, etc.).

---

## G. Advanced and frontier topics

**46. What's Sophia, and why might it matter for LLM training?**
Sophia (Liu et al. 2023) approximates the diagonal of the Hessian using Hutchinson's estimator and uses it as a preconditioner instead of $\sqrt{\hat v}$. Empirically converges in fewer steps than Adam on language modeling — a real speedup in compute terms. Cost: extra estimator calls per step. Whether it dominates AdamW at scale is still being established.

**47. What is the relationship between learning rate and batch size in the "constant noise scale" view?**
The implicit regularization strength scales as $\eta/B$. To keep the same regularization when changing batch size, change $\eta$ proportionally — equivalently, increase $B$ and $\eta$ together to speed up training without changing implicit regularization. "Don't decay the LR, increase the batch size" (Smith et al. 2018) exploits this: instead of decaying $\eta$, increase $B$ over time, getting equivalent dynamics with better hardware utilization.

**48. What does "second-order Adam" or natural gradient buy you?**
Natural gradient uses the Fisher information matrix as a preconditioner, accounting for the geometry of the parameter space under the model's output distribution. It corresponds to "shortest distance in distribution space" rather than parameter space. K-FAC is a tractable approximation. For LLMs, full natural gradient is prohibitive; Adam's diagonal approximation is what we settle for. Sophia is one step closer to second-order at affordable cost.

**49. What's the difference between LR for embedding tables and the rest of the model?**
Embedding tables are sparse (only the rows for sampled tokens get gradient signal per step), and their $\hat v$ estimates are heavily skewed toward frequent tokens. Naive Adam can over-update rare-token embeddings (because their $\hat v$ is small) and under-update common tokens. Common fixes: per-row state, larger $\varepsilon$ for embeddings, or specialized sparse Adam. Frontier-lab pretraining recipes often have a separate LR multiplier for embeddings.

**50. Walk me through the math of why Adam is sensitive to $\beta_2$ choice.**
With $\beta_2 = 0.999$, the effective horizon of the second-moment estimate is $1/(1 - \beta_2) = 1000$ steps. If you have a gradient outlier at step $t$, its contribution to $v_t$ decays as $\beta_2^k$ after $k$ steps — half-life ~700 steps. That means a single bad batch at step 10000 still influences the update at step 10700. This is why some recipes use $\beta_2 = 0.95$ for robustness or $\beta_2 = 0.99$ for very long pretraining. Mismatched $\beta_2$ is a real source of training instability after long training runs.

---

## H. Quick-fire (say each answer in under 10 seconds)

**51.** *Default Adam betas?* $0.9, 0.999$.
**52.** *Default Adam epsilon?* $10^{-8}$.
**53.** *Default LLM clipping norm?* $1.0$.
**54.** *Typical pretraining peak LR for a 7B?* $\sim 3 \times 10^{-4}$.
**55.** *Typical fine-tuning LR for full SFT?* $\sim 2 \times 10^{-5}$.
**56.** *Typical DPO LR?* $\sim 5 \times 10^{-7}$.
**57.** *Warmup duration as fraction of training?* $0.5$–$2\%$.
**58.** *Cosine $\eta_{\min}$ as fraction of peak?* $\sim 10\%$.
**59.** *AdamW weight decay default?* $0.01$–$0.1$, often $0.1$ for LLMs.
**60.** *Update-to-weight ratio target?* $\sim 10^{-3}$ per layer.

---

## I. Self-grading

If you can't answer questions 1–10 cold, you don't know SGD. If you can't answer 11–20, you don't know modern optimization. If you can't answer 21–30, you can't pass an MLE round. If you can't answer 31–50, you'll struggle in frontier-lab applied scientist screens.

Aim for 40+ correct out of 50 before walking into an interview. Re-grill on the misses.
