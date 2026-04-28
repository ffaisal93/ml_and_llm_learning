# Learning Rate: A Frontier-Lab Interview Deep Dive

> **Scope.** This document is the single most important file in this folder for interviews. The learning rate is the hyperparameter most likely to make training succeed or fail, and it is the one interviewers use to probe whether a candidate actually understands optimization or just recites slogans. Every section below corresponds to a real follow-up question that has been asked in applied scientist / ML engineer screens.

---

## 1. What the learning rate actually is

The learning rate $\eta$ (sometimes $\alpha$ or `lr`) is the scalar that multiplies the (possibly preconditioned) gradient before it is subtracted from the parameters:

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t \qquad \text{(plain SGD)}
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot P_t \cdot g_t \qquad \text{(preconditioned: Adam, RMSProp, etc.)}
$$

That sentence sounds trivial. The non-trivial part is: **what determines a "good" $\eta$?** The answer is the geometry of the loss surface in the direction the optimizer is moving. Specifically, for a quadratic approximation of the loss with Hessian $H$, gradient descent converges if and only if:

$$
0 < \eta < \frac{2}{\lambda_{\max}(H)}
$$

where $\lambda_{\max}(H)$ is the largest eigenvalue of the Hessian — i.e. the "sharpest curvature direction." If you exceed $2/\lambda_{\max}$, the iterates oscillate and grow without bound in that direction. Below $1/\lambda_{\max}$ you converge but you may be wasting steps in flatter directions. The optimal rate for a quadratic is $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, and the convergence rate is governed by the **condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$. Big $\kappa$ means slow.

**The practical corollary:** there is no single best $\eta$. The correct learning rate depends on the curvature of the loss in the direction you're currently moving, and that curvature changes during training, varies across layers, and varies across parameters within a layer. Every modern optimizer is some attempt to approximate per-direction step sizes.

---

## 2. When the learning rate "works" vs "doesn't"

This is the question you said is hurting you. Here is the real answer.

### The learning rate "works" when:

1. **$\eta$ is below the stability threshold for the sharpest direction**, but not so far below that flat directions move imperceptibly. In a deep network this band is roughly half a decade wide for SGD and one to two decades wide for Adam.
2. **The curvature is reasonably well-matched across parameters.** For ill-conditioned problems (early-layer features, embedding tables, normalization statistics), a single global $\eta$ is wrong by orders of magnitude across parameters. Adam-family optimizers paper over this by dividing by $\sqrt{v_t}$ per parameter. SGD does not — which is why SGD on transformers without warmup or layer-wise rates routinely diverges.
3. **The schedule matches the phase of training.** Early steps need smaller updates because gradient noise dominates and the loss landscape near initialization is ill-behaved. Late steps need smaller updates because you're fine-tuning around a local optimum. A constant $\eta$ is rarely correct for both.
4. **The optimizer's noise scale matches the geometry.** SGD noise per step is roughly $\eta \cdot \sigma / \sqrt{B}$, where $B$ is batch size and $\sigma$ is gradient covariance. Too much noise pushes you off any narrow valley; too little makes you crawl.

### The learning rate "fails" with these specific signatures:

| Symptom | Likely cause | Diagnostic |
|---|---|---|
| Loss → NaN at step 1–5 | $\eta$ is way above stability; or fp16 overflow in the first matmul | Print pre-step loss; check $\|g\|$; lower $\eta$ 10x and rerun |
| Loss flat then NaN around step ~100–500 | Warmup ends and $\eta$ is too high; or Adam $\hat v$ not yet stabilized | Extend warmup; lower peak $\eta$; check $\|\text{update}\|/\|\theta\|$ |
| Loss oscillates with rising amplitude | At edge-of-stability for some direction; usually layer-specific | Per-layer gradient norms; lower $\eta$ for the offending layer |
| Loss decreases very slowly, plateaus | $\eta$ too small, or stuck near a saddle, or schedule decayed too aggressively | LR finder; check gradient norm; warm restart |
| Training loss good, eval loss bad | Could be $\eta$ too low at end (under-fit local) or too high throughout (over-shoots wide minima) | Compare schedules; check sharpness of final solution |
| "Loss spike" that recovers | Common with Adam at edge of stability; one extreme batch shifts $\hat v$ and update explodes briefly | Gradient clipping; lower peak $\eta$; investigate the spike batch |
| Different layers learning at wildly different rates | Curvature mismatch | Layer-wise LR (LARS/LAMB) or bigger $\varepsilon$ in Adam |
| Pretrained model degrades during fine-tuning | $\eta$ too high for transfer | Fine-tune $\eta$ is typically 10–100x smaller than pretrain $\eta$ |

The key meta-skill: **read the loss curve and the gradient norm together**. A flat loss with vanishing gradients means the optimizer is stuck. A flat loss with healthy gradients means $\eta$ is too small. A descending loss with growing gradient norm means you're approaching instability.

---

## 3. The single most useful diagnostic: update-to-weight ratio

If you remember one number from this entire document, remember this one. A widely-cited empirical heuristic (most prominently popularized by Karpathy's "A Recipe for Training Neural Networks" blog post and his lectures — note: this is folklore-level guidance, not a published theorem):

$$
\text{ratio} = \frac{\|\eta \cdot \text{update}_t\|}{\|\theta_t\|}
$$

Healthy training has this ratio around $10^{-3}$ per layer. If it's $10^{-1}$, your $\eta$ is way too high. If it's $10^{-6}$, your $\eta$ is way too low. This is more reliable than watching loss because it works **per layer** and reveals heterogeneity that a single loss curve hides.

This single diagnostic explains why a single global learning rate is fundamentally wrong for deep networks — the ratio differs across layers by orders of magnitude unless you adapt.

---

## 4. The LR finder (Cyclical Learning Rates, Leslie Smith)

Practical recipe to find a sensible $\eta$ without guessing:

1. Set $\eta_0$ very low (e.g. $10^{-7}$).
2. Run training, multiplying $\eta$ by a constant $> 1$ each step (e.g. by 1.1).
3. Plot loss vs. $\log \eta$.
4. Pick a learning rate roughly **one order of magnitude below the point where loss starts increasing**.

This works because the curve shows you the practical stability boundary on your data. The optimum is not at the minimum of that curve — it's well to the left of it, because you want headroom for the step that crosses a sharp ridge.

**Interview trap.** Many candidates say "pick the $\eta$ where the loss is minimum on this curve." That's wrong. Pick the $\eta$ where the loss is **steepest downward** (still decreasing strongly). The minimum point is already at the cliff edge.

---

## 5. Schedules: what they are and when each one wins

A learning rate schedule is a function $\eta(t)$ that varies the rate over training. The intuition is that the right step size depends on phase:

- **Early phase (warmup):** Adam's second-moment estimate $\hat v$ is unreliable, residual streams are not yet calibrated, and the loss surface near random initialization can be pathologically sharp. Big steps here destabilize forever.
- **Middle phase:** the optimizer is making meaningful progress; you want the highest step size that's stable.
- **Late phase:** you're polishing; smaller steps reach a tighter minimum.

### Common schedules

**Linear warmup + cosine decay** is the modern default for LLM pretraining:

$$
\eta(t) = \eta_{\max} \cdot \frac{t}{W} \qquad \text{for } t \leq W
$$

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\pi \cdot \frac{t - W}{T - W}\right)\right) \qquad \text{for } W < t \leq T
$$

Why it dominates:

- Warmup avoids the initial-instability traps above.
- Cosine smoothly decays without the sudden drops of step decay (which can shock training).
- $\eta_{\min} \approx 0.1 \cdot \eta_{\max}$ is typical, since pure zero learning rate doesn't help and may hurt the late-phase fine-tuning that the cosine tail is doing.

**Linear warmup + linear decay (a.k.a. trapezoid):** common in newer recipes. Slightly easier reasoning about "compute saved per unit of decay" and used by recent open-weight LLMs (LLaMA-style runs sometimes use cosine; some Chinchilla-style runs use linear or trapezoidal).

**Step decay:** drop $\eta$ by 10x at predetermined epochs (e.g. ResNet on ImageNet at epochs 30, 60, 90). Old style. Brutal but reliable for tasks with clean phase boundaries.

**ReduceLROnPlateau:** drop $\eta$ when validation loss has not improved for $N$ steps. Useful for fine-tuning and for tasks where you can't predict the right total budget.

**Triangular / cyclical:** sweep $\eta$ up and down repeatedly. Sometimes helps escape bad local minima. Less common in modern LLM training because it interferes with cosine-style budgets.

**Constant:** rare in pretraining, common in some online / continual settings where there is no "end of training."

### Warmup specifically — why it matters

The first few hundred steps are where most catastrophic divergences happen. The reasons compound:

1. **Adam variance estimates are noisy.** With $\beta_2 = 0.999$, the effective window is ~1000 steps. Before you have enough samples, $\hat v$ is small and biased low even with bias correction, so updates $g/\sqrt{\hat v}$ are larger than they should be.
2. **Residual streams are not calibrated.** In pre-LN transformers and even more in post-LN, the magnitude of activations early in training is far from steady state; gradients reflect that and are oversized.
3. **Batch normalization statistics** (when used) are drifting fastest at the start, which compounds with optimizer updates.
4. **Empirically**, transformer training without warmup diverges near-deterministically at modern scales.

Typical warmup is $0.5$–$5\%$ of total steps. For LLM pretraining, 2000 steps of linear warmup is a common default. Too short → instability. Too long → wasted compute at low effective LR.

---

## 6. Linear scaling rule and the LR–batch size relationship

This is one of the highest-yield interview topics in this document.

### The setup

When you train SGD on batches of size $B$, each step uses an estimate of the gradient with variance roughly $\sigma^2 / B$. If you double $B$ you halve the gradient variance per step but you're also taking half as many steps for the same data volume. Per-token progress depends on the effective step.

### The rule (Goyal et al., 2017, "Accurate, Large Minibatch SGD")

For SGD on convolutional networks: **if you scale batch size by $k$, scale the learning rate by $k$**. With warmup. This worked on ImageNet up to batch size ~8192. The reasoning is that under SGD the per-epoch progress depends on $\eta \cdot k$, so to preserve trajectory you scale $\eta$ linearly.

### The Adam version

For Adam, the correct scaling is closer to $\sqrt{k}$ for the same batch size scaling. Rough intuition: Adam already rescales by $\sqrt{\hat v}$, which is itself an estimate of the second moment of the gradient. If you change batch size, that estimate changes; the net effect is that the right $\eta$ scales sublinearly with $B$.

**Honest caveat.** The exact scaling rule for Adam has been debated; some papers find linear scaling holds at moderate batch sizes and breaks at very large batches; others find sqrt is closer. The interview-grade answer is: **for SGD use linear scaling with warmup; for Adam use sqrt scaling as a starting point and verify empirically; both rules break at very large batch sizes (above the "critical batch size").**

### Critical batch size (McCandlish et al., 2018)

Beyond a certain batch size, doubling the batch stops giving proportional speedups even with optimal $\eta$ rescaling. This batch size is determined by the **gradient noise scale**, which is roughly the ratio of gradient mean magnitude to gradient covariance. Gradient noise scale grows during training, which means the critical batch size grows during training — a key reason to not pick batch size statically based on early loss curves.

The takeaway: there is a "data-parallel sweet spot" beyond which more workers don't speed up training, regardless of how you tune $\eta$. Frontier-lab interviews ask about this.

---

## 7. Edge of stability (Cohen et al., 2021)

A more recent result that interviewers love because it overturns textbook intuition. For deep networks trained with full-batch GD, the largest Hessian eigenvalue $\lambda_{\max}$ grows during training until it stabilizes near $2/\eta$. That is, the optimizer drifts toward the **edge** of the stability region instead of staying safely below it. Once there, training continues to make progress but with non-monotonic loss — you see local oscillation but global descent.

Implications:

- Loss spikes during training are not necessarily bugs. They can be a feature of operating at the edge of stability.
- The standard convex-optimization picture ("set $\eta$ below $2/\lambda_{\max}$ and converge smoothly") is wrong for real deep learning.
- Sharper minima have lower $\lambda_{\max}$ after training, which may explain part of the relationship between sharpness and generalization.

When asked "why does my loss occasionally spike but training is fine?" the strong answer cites edge of stability.

---

## 8. Adam-specific learning rate behavior

A few details that separate strong from weak answers.

**1. Why Adam tolerates a wider LR range than SGD.** Adam normalizes per-parameter by $\sqrt{\hat v}$. This means parameters with large gradients get smaller effective updates and vice versa. The optimizer is implicitly computing per-direction step sizes, which removes most of the conditioning problem that SGD suffers from. Effective LR for parameter $i$ is roughly $\eta / \sqrt{\hat v_i}$.

**2. The role of $\varepsilon$.** $\varepsilon$ (default $10^{-8}$) is the additive constant in $\eta \cdot \hat m / (\sqrt{\hat v} + \varepsilon)$. It does two things: it prevents division by zero, and **it caps the maximum effective LR per parameter**. When $\sqrt{\hat v} \ll \varepsilon$, the update is roughly $(\eta / \varepsilon) \cdot \hat m$, so dimensions with very small gradients can still get sensible updates. Some training recipes (e.g. embedding tables) set $\varepsilon$ larger to avoid huge effective LRs on rarely-touched parameters. Some quantized-Adam variants use $\varepsilon = 10^{-3}$ deliberately.

**3. Bias correction matters early but only early.** The corrections $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$ are nearly 1 by step ~5000 with default $\beta_2 = 0.999$. Forgetting bias correction in an Adam implementation is a common interview trap that causes very different early-training behavior.

**4. $\beta_1$, $\beta_2$, and effective horizons.** $\beta_1 = 0.9$ means an effective horizon of ~10 steps for the first moment; $\beta_2 = 0.999$ is ~1000 steps for the second moment. Larger $\beta_2$ (e.g. 0.95) is sometimes used for robustness to outlier gradients but slows the optimizer's adaptation. The mismatch in horizons is intentional: you want the gradient direction to react quickly while the variance estimate stays stable.

**5. AdamW vs Adam.** This deserves its own section.

---

## 9. AdamW vs Adam — the weight decay question

If you can answer this cleanly, you're already ahead of most candidates.

**The wrong answer:** "L2 regularization adds a penalty to the loss; weight decay is a different thing." This is correct in spirit but doesn't show understanding.

**The right answer:**

Naive Adam with L2 regularization computes the gradient of $\text{loss} + (\lambda/2)\|\theta\|^2$, which adds $\lambda \theta$ to $g_t$. Then Adam preconditions everything by $1/\sqrt{\hat v}$. The effect is that for parameters with large gradient variance, the L2 penalty is divided by a large number — it's effectively *weakened*. So the regularization strength is no longer uniform across parameters.

AdamW decouples weight decay from the gradient computation:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta \cdot \lambda \cdot \theta_t
$$

Weight decay is applied directly to $\theta$, after the Adam update. Now every parameter shrinks by the same fraction $\eta \cdot \lambda$ per step regardless of its gradient statistics. This recovers the regularization behavior people *thought* they were getting from L2.

In the original "Decoupled Weight Decay Regularization" paper (Loshchilov & Hutter), AdamW substantially outperforms Adam on image classification, and it's now the default for LLM training.

**Interview trap.** "If the gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda \theta$, isn't L2 the same as weight decay?" For SGD, yes: SGD with L2 produces the same update as SGD with explicit weight decay. For Adam, no: the preconditioning changes the effective decay strength per parameter.

---

## 10. Layer-wise and per-parameter scaling: LARS, LAMB

For very large batch training, even Adam's per-parameter scaling isn't enough — you also need per-layer scaling. LARS (You et al., 2017) and LAMB (You et al., 2019) compute a layer-wise trust ratio:

$$
\text{trust} = \frac{\|\theta_{\text{layer}}\|}{\|\text{update}_{\text{layer}}\|}, \qquad \eta_{\text{effective}} = \eta_{\text{global}} \cdot \text{trust}
$$

This explicitly enforces a constant update-to-weight ratio per layer, which is exactly the diagnostic from §3. LARS made batch sizes of 32K trainable for ResNet-50; LAMB extended it to BERT pretraining.

Modern LLM training rarely uses LARS/LAMB explicitly, but the spirit lives on in muP (maximal update parameterization, Yang & Hu, 2022), which proposes initialization and learning-rate scaling that keeps update magnitudes constant across model widths. If a question goes deep into "how do you transfer hyperparameters from a small model to a 70B?", muP is the right answer.

---

## 11. Pretraining vs. fine-tuning learning rates

Different regimes, different rules.

| Regime | Typical peak $\eta$ | Schedule | Why |
|---|---|---|---|
| LLM pretraining (Adam) | $10^{-4}$ to $6 \times 10^{-4}$ | linear warmup + cosine | Large data, long horizon, high $B$ |
| LLM fine-tuning (full) | $10^{-5}$ to $5 \times 10^{-5}$ | linear warmup + linear decay | Pretrained weights are valuable; don't damage |
| LoRA / adapter fine-tuning | $10^{-4}$ to $5 \times 10^{-4}$ | sometimes constant | Only training a small subset; can be more aggressive |
| RLHF reward model | $5 \times 10^{-6}$ to $10^{-5}$ | very low | Easy to overfit; tiny dataset relative to base |
| RLHF PPO / DPO | $5 \times 10^{-7}$ to $5 \times 10^{-6}$ | constant or short warmup | Policy is fragile; KL anchor needs preserving |
| Vision SFT | $10^{-3}$ to $10^{-4}$ | step decay or cosine | Different gradient statistics |
| Diffusion training | $10^{-4}$ to $10^{-5}$ | cosine, or constant | Needs long, stable training |

**Why fine-tuning needs lower LR.** The pretrained model is at a useful basin. A large $\eta$ will move it out of that basin in a few steps. The exception is LoRA: you're adding new low-rank matrices initialized to zero, so the initial gradients live on parameters that have nothing to lose, and you can use a higher $\eta$.

**Why RLHF/DPO needs even lower LR.** The whole point is to nudge a model away from the SFT distribution while preserving capabilities. A high LR causes mode collapse, KL blowup, or reward hacking. Some recipes (DPO especially) report best results with $\eta$ as low as $5 \times 10^{-7}$.

---

## 12. Why does noise help? The implicit-regularization view

A real interview question: "Why does SGD generalize better than full-batch GD on the same loss?"

The serious answer involves implicit regularization. The noise injected by mini-batch sampling biases SGD toward **flat** minima. There are two complementary stories:

**Story 1: noise as escape mechanism.** Sharp minima have steep walls. Stochastic noise pushes the optimizer back and forth; in a sharp minimum the average loss visited is high (you keep hitting the walls), so the effective "potential" the optimizer feels is lower in flat minima.

**Story 2: SGD as approximate Bayesian inference.** Mandt et al. show that SGD is approximately sampling from a posterior over parameters, with temperature proportional to $\eta / B$. The posterior concentrates on flat minima because they have higher marginal likelihood under the noisy update.

**Story 3 (more rigorous): SGD ≈ GD on a modified loss.** Smith et al. show that SGD with learning rate $\eta$ and batch size $B$ approximately follows GD on $L(\theta) + \frac{\eta}{4B} \|\nabla L(\theta)\|^2$ — i.e. with a regularizer that penalizes regions of high gradient norm. This explicitly biases the optimizer toward flat regions.

The take-home: the noise-to-signal ratio $\eta / B$ is the relevant quantity. That's why if you scale batch size up, you generally need to scale $\eta$ up to maintain the same "implicit regularization." Going to very large batches with the same $\eta$ removes the implicit regularization and often hurts generalization — unless you compensate.

---

## 13. Gradient clipping

Closely related to learning rate; sometimes confused with it. Gradient clipping is:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{\text{clip\_norm}}{\|g\|}\right)
$$

It caps the *magnitude* of the gradient (or update), independent of $\eta$. The two main reasons to use it:

1. **Spike protection.** A single bad batch can produce a $\|g\|$ 100x typical; clipping prevents that one batch from blowing up training.
2. **RNN-specific.** Vanilla RNNs/LSTMs produce gradients that grow exponentially with sequence length (the "exploding gradient" problem). Clipping is essentially mandatory.

Clip-by-norm with $\text{clip\_norm} = 1.0$ is the LLM-pretraining default. Clip-by-value is used less. Be aware that clipping interacts with the learning rate: a very loose clip is a no-op; a very tight clip effectively reduces $\eta$.

**Interview trap.** Someone says "I have loss spikes; I'll lower $\eta$." A better first move is "I'll add gradient clipping at norm 1.0," because that targets the spike specifically without slowing the rest of training.

---

## 14. Learning-rate transfer across scales

You said you're targeting frontier-lab roles. They will ask: **how do you set the learning rate for a 70B model when you can only afford to do hyperparameter sweeps on a 1B model?**

The naive answer "use the same LR" is wrong. Bigger models have lower optimal $\eta$ because the loss landscape changes with model width and depth.

The strong answer is **muP (Yang & Hu, 2022)**: a parameterization that includes specific scalings for the initialization, embedding, and per-layer learning rates such that the optimal LR is *width-invariant*. If you tune $\eta$ on a small width-$d$ model under muP, the same $\eta$ is optimal at width $4d$, $16d$, etc.

This is why several frontier labs publicly disclose using muP: it lets them sweep $\eta$ on small models cheaply and then scale up with confidence. If you understand muP at a sketchy level, you'll be in the top decile of candidates on this topic.

---

## 15. Practical interview cheatsheet

When you walk into an interview and someone says "your training is unstable, what do you do?", here is the order of operations that signals competence:

1. **Look at the loss curve and the gradient norm together.** If gradients are vanishing, the issue is too-low LR or schedule decay; if gradients are exploding, the issue is too-high LR or no clipping.
2. **Compute the update-to-weight ratio per layer.** If it's not in the $10^{-4}$ to $10^{-2}$ range, you have an LR problem — and which layer is offending tells you whether it's global or per-parameter.
3. **Ensure warmup is in place.** "How long is your warmup?" should always be one of your first questions.
4. **Lower peak LR by 3x and rerun.** Cheap, often decisive.
5. **Add gradient clipping at norm 1.0** if not already present.
6. **Check the data.** A bad batch can masquerade as an LR problem.
7. **Check the optimizer state.** A corrupted Adam $\hat v$ from a previous bad run will sabotage everything; reset and rerun.
8. **Instrument first, theorize second.** Strong candidates print weight norms, gradient norms, and update norms by layer before adjusting hyperparameters.

If you internalize these eight steps, you will pass most learning-rate debugging questions.

---

## 16. The 12 most common interview questions about learning rate

(Brief answers; full grilling in `INTERVIEW_GRILL.md`.)

1. **What's the relationship between LR and batch size?** Linear scaling for SGD, sqrt-ish for Adam, both break at the critical batch size.
2. **Why do you need warmup?** Adam variance estimates are noisy early; residual streams are uncalibrated; the loss surface near init is sharp. Without warmup, transformers diverge.
3. **How do you choose LR?** LR finder, then schedule with cosine. Verify update-to-weight ratio is $\sim 10^{-3}$.
4. **What's edge of stability?** $\lambda_{\max}(H)$ drifts to $\sim 2/\eta$ during training; loss is non-monotonic but globally descending.
5. **Why is AdamW different from Adam with L2?** Adam's preconditioning weakens L2 for high-variance parameters; AdamW decouples decay so it applies uniformly.
6. **What does $\varepsilon$ in Adam control?** Numerical stability and a cap on the effective LR for low-gradient parameters.
7. **What is the critical batch size?** Beyond it, more parallelism stops giving speedups; it grows during training.
8. **Why does SGD generalize better than full-batch GD?** Implicit regularization via $\eta/B$ noise biases toward flat minima.
9. **Why is fine-tuning LR much lower than pretraining LR?** Pretrained weights live in a useful basin; large steps destroy that.
10. **What is muP?** A parameterization that makes the optimal LR width-invariant, enabling small-scale sweeps to transfer.
11. **What does loss spike → recovery look like?** Often edge-of-stability behavior. Use clipping; don't necessarily lower LR globally.
12. **What is the linear scaling rule?** Scale $\eta$ linearly with batch size for SGD with warmup; works up to a point, breaks at large batches.

---

## 17. Further reading (high signal-per-page)

- Goyal et al., "Accurate, Large Minibatch SGD" (2017) — linear scaling rule.
- Smith et al., "Don't Decay the Learning Rate, Increase the Batch Size" (2018).
- McCandlish et al., "An Empirical Model of Large-Batch Training" (2018) — gradient noise scale and critical batch size.
- Cohen et al., "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability" (2021).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2019).
- Yang & Hu, "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (muP, 2022).
- Smith, "A Disciplined Approach to Neural Network Hyperparameters" (LR finder, 2018).
- Mandt et al., "Stochastic Gradient Descent as Approximate Bayesian Inference" (2017).

If you internalize these papers, the learning rate stops being a mystery and becomes a measurable, debuggable quantity.
