# Optimizers — Interview Grill

> 40 questions focused on **optimizer algorithms specifically** — different angle from the LR-centric grill in `02_gradient_descent/INTERVIEW_GRILL.md`. Use both.

---

## A. Algorithmic foundations

**1. What's the relationship between optimizers and Newton's method?**
Newton uses $H_t^{-1} g_t$ as the update direction — accounting for second-order curvature. Storing $H$ is $O(d^2)$ and inverting is $O(d^3)$, infeasible at scale. Every modern optimizer is some cheap approximation: SGD = identity preconditioner; Adam/RMSProp = diagonal $1/\sqrt{\hat v}$ preconditioner approximating $\mathrm{diag}(H)^{-1/2}$; Shampoo = block-Kronecker; Sophia = stochastic Hutchinson estimate of $\mathrm{diag}(H)$.

> **Saying it out loud.** Every optimizer is a cheap stand-in for Newton's method. Newton says multiply the gradient by the inverse Hessian, which tells you not just which way is downhill but how curved the ground is in each direction. You can't do that for a billion parameters — the Hessian alone would be $10^{18}$ numbers. So everything we actually use is an approximation of that matrix: SGD pretends it's the identity, Adam keeps only the diagonal, Shampoo keeps a block structure per layer, Sophia estimates the diagonal stochastically. Framing it that way makes the whole optimizer zoo one axis instead of a list.

**2. Walk me through SGD with classical momentum.**

$$
v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Velocity $v_t$ is an exponentially-weighted sum of past gradients. $\beta = 0.9$ is standard. Effective gradient horizon is $1/(1-\beta) \approx 10$. Helps convergence in ill-conditioned valleys by averaging out perpendicular oscillations and reinforcing the persistent direction along the valley.

> **Saying it out loud.** Momentum is a running average of your recent gradients instead of just the latest one. Physically it's a ball rolling downhill — it builds speed in a consistent direction and coasts through small bumps. The reason it helps is that in a narrow valley the gradient keeps flipping side to side, and averaging cancels those oscillations while the along-the-valley component adds up. With the standard beta of 0.9 you're effectively averaging the last ten gradients, and that $1/(1-\beta)$ rule of thumb is the number to have ready.

**3. What's Nesterov momentum and why is it different?**
Computes the gradient at the *lookahead* position (where momentum will take you anyway):

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t - \eta \beta v_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

Theoretically improves convex convergence from $O(1/T)$ to $O(1/T^2)$ for smooth strongly-convex problems. Empirically often slightly better than Polyak momentum.

> **Saying it out loud.** Nesterov evaluates the gradient where momentum is *about to put you*, not where you are now. It's the difference between braking when you see the wall and braking when you hit it — you get the correction one step earlier, so overshoot is smaller. In convex theory it upgrades the rate from $O(1/T)$ to $O(1/T^2)$, which is a real theoretical gap. In deep learning the practical difference is usually small but generally non-negative, so it's a free-ish improvement rather than a game changer.

**4. Walk me through RMSProp.**

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t^2, \qquad \theta_{t+1} = \theta_t - \eta \cdot \frac{g_t}{\sqrt{v_t} + \varepsilon}
$$

Per-parameter rescaling by RMS of recent gradients. The second-moment $\mathbb{E}[g g^\top]$ is the **Fisher information matrix** (not the Hessian directly). For likelihood losses, $F = H$ only at a stationary point — so "diagonal Hessian approximation" is loose; "diagonal Fisher" is more accurate. Removes most LR-tuning sensitivity that plain SGD has.

> **Saying it out loud.** RMSProp gives every parameter its own learning rate by dividing the update by the root-mean-square of that parameter's recent gradients. So a weight with consistently huge gradients gets small steps and one with tiny gradients gets relatively bigger steps — everything ends up moving at a comparable pace. That's what makes it robust to the learning rate in a way plain SGD isn't. The precision point worth making: the running average of squared gradients is a diagonal Fisher estimate, not a Hessian estimate — they only coincide at a stationary point, so "diagonal Hessian" is loose.

**5. Walk me through Adam with bias correction.**

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad &\text{(first moment, momentum)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad &\text{(second moment, RMS)} \\
\hat m_t &= m_t / (1 - \beta_1^t) \quad &\text{(bias correction)} \\
\hat v_t &= v_t / (1 - \beta_2^t) \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \hat m_t / (\sqrt{\hat v_t} + \varepsilon)
\end{aligned}
$$

Defaults $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$. Combines momentum and adaptive per-parameter rescaling.

> **Saying it out loud.** Adam is momentum plus RMSProp. You keep two running averages: the mean of recent gradients, which gives you direction and smoothing, and the mean of squared gradients, which gives you per-parameter scale. The update is the first divided by the square root of the second. Then bias correction, because both averages start at zero and are therefore too small early on. Defaults are 0.9, 0.999, and epsilon at $10^{-8}$, and the reason Adam won is that those defaults work almost everywhere without tuning.

**6. Why is bias correction necessary?**
$m_t$ and $v_t$ initialize at zero. Without correction, the first $\sim 1/(1-\beta)$ steps have moments that are underestimates of the true running averages — biased low. For $\beta_2 = 0.999$, $\hat v$ is biased for ~1000 steps. Without correction, the early effective LR $\eta / \sqrt{\hat v}$ is too large, training often diverges. The bias correction $1/(1-\beta^t)$ exactly inverts the geometric-series discount.

> **Saying it out loud.** Because both moving averages start at zero, so early in training they're biased toward zero. That matters asymmetrically — the second moment sits in the denominator, so an underestimate makes the effective step much too big exactly when the model is most fragile. With $\beta_2$ of 0.999 the second moment is meaningfully biased for something like the first thousand steps. The correction is just dividing by $1 - \beta^t$, which exactly inverts the geometric discount and goes to 1 as $t$ grows. Skip it and runs diverge in the first few hundred steps, which is one of the classic from-scratch implementation bugs.

**7. What does $\varepsilon$ in Adam control?**
Two roles: (a) numerical floor preventing $1/\sqrt{\hat v}$ from blowing up when $\hat v \approx 0$, (b) implicit cap on per-parameter LR — when $\sqrt{\hat v} \ll \varepsilon$, the update is $(\eta/\varepsilon) \cdot \hat m$, so dimensions with very small gradients still get sensible updates. Some recipes set $\varepsilon = 10^{-3}$ for embeddings to dampen aggressive updates on rare tokens.

> **Saying it out loud.** Epsilon is the floor in the denominator, and it does two jobs. Obviously it stops you dividing by zero when a parameter has had essentially no gradient. Less obviously it caps the per-parameter learning rate: once the square root of the second moment is well below epsilon, the update is just eta over epsilon times the momentum, so tiny-gradient parameters can't get arbitrarily huge steps. That's why the value matters more than people think — some recipes deliberately raise it to $10^{-3}$ on embeddings so rare tokens don't get slammed. And a real gotcha: $\sqrt{\hat v} + \varepsilon$ and $\sqrt{\hat v + \varepsilon}$ are different, and libraries have disagreed.

**8. What if you set $\beta_2 = 0.9999$?**
The second-moment horizon grows to ~10000 steps. Pros: more robustness to outlier gradients. Cons: very slow to track changes in gradient statistics — when training transitions from warmup to the main phase, $\hat v$ lags badly. Empirically, $\beta_2 = 0.999$ is a sweet spot. $\beta_2 = 0.95$ is sometimes used for very long pretraining for the opposite reason: faster reaction.

> **Saying it out loud.** You'd be averaging squared gradients over roughly the last ten thousand steps instead of a thousand. That buys you robustness — one freak gradient barely moves the denominator, so loss spikes get damped. What it costs you is responsiveness: when the real gradient scale shifts, like coming out of warmup or hitting a new data mixture, your denominator is stale and the effective learning rate is wrong for thousands of steps. So 0.999 is the sweet spot for most runs, and people go the *other* way, to 0.95, on long pretraining when they want faster adaptation.

---

## B. AdamW vs Adam vs L2

**9. What is AdamW?**
Adam with **decoupled** weight decay. The update becomes:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta \cdot \lambda \cdot \theta_t
$$

Weight decay applied directly to $\theta$ after the Adam update, not added to the gradient.

> **Saying it out loud.** AdamW is Adam where weight decay is applied straight to the weights instead of being folded into the gradient. So the step is: do the normal Adam update, then separately shrink every parameter by learning rate times lambda. The word to use is decoupled — decay no longer passes through the adaptive denominator. It sounds like a trivial reordering, and it's worth about the difference between Adam having a reputation for generalising worse than SGD and Adam being the default for every large model trained today.

**10. Why isn't Adam-with-L2 equivalent to AdamW?**
Adam-with-L2 adds $\lambda \theta$ to the gradient: $g_t \leftarrow g_t + \lambda \theta_t$. Then $v_t$ accumulates $(g_t + \lambda \theta_t)^2$, the regularization term gets divided by $\sqrt{\hat v}$, and parameters with high gradient variance see weakened L2. Decay strength becomes non-uniform across parameters in a way nobody intends. AdamW separates decay from preconditioning; every parameter shrinks by exactly $\eta \cdot \lambda$ regardless of its gradient statistics.

> **Saying it out loud.** Because in Adam-with-L2 the decay term goes into the gradient, and then it gets divided by the square root of the second moment along with everything else. So a parameter with big noisy gradients has a big denominator, and its regularisation gets scaled *down* — exactly backwards from what you want. The result is that your decay strength varies across parameters in a way nobody chose and nobody can see. AdamW keeps decay outside the preconditioner, so every weight shrinks by exactly the same relative amount. That's Loshchilov and Hutter's paper, and it's the cleanest example of "same math, different order, very different behaviour.\"

**11. For SGD, are L2 and weight decay equivalent?**
Yes. Gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda \theta$, so SGD with explicit decay is identical to SGD with L2. They diverge only when there's preconditioning (Adam, RMSProp, K-FAC).

> **Saying it out loud.** Yes, for plain SGD they're exactly the same thing. The gradient of the L2 penalty is lambda times theta, so adding it to the gradient and subtracting it after the step give an identical update. They only come apart when there's a preconditioner in between — Adam, RMSProp, K-FAC — because then the L2 term gets rescaled and the decay term doesn't. That's the whole reason AdamW exists and SGDW doesn't need to.

**12. What's a typical AdamW weight decay value for LLMs?**
$\lambda = 0.1$ for pretraining is the modern default. $\lambda = 0.01$ is more typical for vision and smaller models. SFT and DPO usually use $0.0$ or very small ($0.001$).

> **Saying it out loud.** For LLM pretraining, 0.1 is the modern default and you'll see it in basically every published recipe. Vision and smaller models usually sit around 0.01. For fine-tuning — SFT or DPO — you generally go to zero or something tiny like 0.001, because you're doing few steps on limited data and shrinking weights just pulls you away from a checkpoint you liked. The framing: decay is for long runs where the model has time to overfit, not for short ones where you're mostly worried about drift.

**13. Why do attention layers and embeddings sometimes have different weight decay?**
Embedding parameters often see sparse gradient updates (only sampled tokens get gradient). Decay applied uniformly per step over-shrinks rare-token embeddings. Common fixes: zero weight decay on embeddings, layer-norm parameters, and biases; non-zero decay only on weight matrices.

> **Saying it out loud.** Because those parameters see fundamentally different gradient patterns. An embedding row only gets a gradient when its token actually appears, so a rare token might be updated once in ten thousand steps — but decay hits it every single step, so it just shrinks toward zero regardless of what it learned. Layer-norm gains and biases are similar: they're one-dimensional scaling knobs where shrinking them toward zero has a direct, unwanted effect on activations. So the standard recipe is decay on weight matrices only, and zero on embeddings, norms, and biases. It's a one-line change in the parameter groups and it's a very common interview detail.

---

## C. Lion, Sophia, and modern alternatives

**14. Walk me through Lion.**
Sign-based update:

$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad &\text{(interpolation)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \mathrm{sign}(c_t) - \eta \cdot \lambda \cdot \theta_t \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t \quad &\text{(momentum)}
\end{aligned}
$$

Update magnitude per parameter is exactly $\eta$ (modulo decay). No second moment, no division, no square root. Memory: one state buffer per param vs Adam's two.

> **Saying it out loud.** Lion throws away the magnitude of the gradient entirely and keeps only the sign. So every parameter moves by exactly the same amount each step, plus or minus, with only the direction coming from a momentum-smoothed gradient. That means no second moment, no square root, no division — one state buffer instead of two, which is a real memory saving at scale. It came out of a symbolic search over optimizer programs at Google, which is a nice detail to drop. The tradeoff is that every parameter takes a full-size step whether it needs one or not.

**15. Why does Lion sometimes work as well as AdamW?**
Sign normalization is an extreme form of per-parameter rescaling — like Adam's $1/\sqrt{\hat v}$ taken to the limit. When gradient magnitudes are similar across parameters (after normalization layers do their job), the normalization in Adam is doing less work than people assume; sign is "good enough" and saves memory.

> **Saying it out loud.** Because sign is just per-parameter normalisation taken to its extreme — Adam divides by the RMS to equalise step sizes, and sign equalises them completely. Once layer norms and residual connections have done their job, gradient magnitudes across parameters are already fairly uniform, so the careful scaling Adam is doing turns out to be worth less than you'd assume. Which means you can drop half the optimizer state and lose surprisingly little. The tradeoff is that Lion is less forgiving — it needs a different learning rate and decay, and it can be rougher on very small or very sparse gradients.

**16. What's the LR difference between Lion and AdamW?**
Lion's optimal $\eta$ is typically 3–10x smaller than AdamW's, because sign updates are "always full magnitude" while Adam's updates can be smaller for low-gradient parameters. Lion's optimal weight decay is typically 3x larger.

> **Saying it out loud.** Lion needs a learning rate roughly 3 to 10 times smaller than AdamW, and weight decay about 3 times larger. The reason is mechanical: Adam's update shrinks for parameters with small gradients, but Lion's is always full magnitude, so the same nominal learning rate moves things much further. And since the update never shrinks, you need more decay to keep the weight norm in check. This is the number-one reason people try Lion, see it diverge, and conclude it doesn't work — they reused their AdamW hyperparameters.

**17. What's Sophia?**
Adam-like, but uses a stochastic Hessian-diagonal estimate via Hutchinson's estimator instead of $\sqrt{\hat v}$:

$$
\hat h_t = \mathrm{clip}(\text{stoch-hutchinson-diag-H}, \rho)
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t}{\max(\gamma \cdot \hat h_t, \varepsilon)} - \eta \cdot \lambda \cdot \theta_t
$$

Hutchinson uses $\mathrm{diag}(H) \approx \mathbb{E}[v \odot Hv]$ for random $v$; $Hv$ is computed via Hessian-vector product (one extra backward pass). Reportedly converges in fewer steps than AdamW on language modeling. Cost: ~25% more compute per step.

> **Saying it out loud.** Sophia replaces Adam's squared-gradient denominator with an actual estimate of the Hessian's diagonal, obtained by Hutchinson's trick — multiply the Hessian by a random vector, which you can do with one extra backward pass, and average. Then you clip that estimate so a near-zero curvature can't produce an enormous step. So it's genuinely second-order information rather than a Fisher proxy, and the reported payoff is reaching the same loss in roughly half the steps on language modelling. The cost is about 25% more compute per step, which is what makes it a real tradeoff rather than a free win.

**18. Why isn't Sophia universally adopted?**
(a) Per-step compute cost. (b) Implementation complexity (HVP via PyTorch isn't a one-liner). (c) Public benchmarks at 70B+ scale are scarce. (d) AdamW is "good enough" — frontier labs are conservative about changing the optimizer mid-training run.

> **Saying it out loud.** A few reasons, and none of them is "it doesn't work." It's about 25% more compute per step, so the step-count win has to be large to actually pay. The Hessian-vector product is fiddly to implement and to make play nicely with distributed training and mixed precision. There's very little public evidence at 70B-plus scale. And the honest institutional reason: nobody wants to be the person who changed the optimizer on a run costing millions of dollars, so AdamW's "good enough plus battle-tested" is very hard to displace.

**19. What's Shampoo?**
Per-layer Kronecker-factored preconditioner. For an $m \times n$ weight matrix, store left factor $L_t$ ($m \times m$) and right factor $R_t$ ($n \times n$). Update:

$$
W_{t+1} = W_t - \eta \cdot L_t^{-1/4} G_t R_t^{-1/4}
$$

Memory $O(m^2 + n^2)$ per layer instead of $O(d^2)$. Empirically state-of-the-art on some tasks but adopted slowly because of implementation complexity and the cost of computing matrix inverse-roots.

> **Saying it out loud.** Shampoo keeps a preconditioner per layer, but factorises it — for a weight matrix it stores one small matrix for the rows and one for the columns instead of the full square of the parameter count. Then it preconditions the gradient on both sides. That gets you real curvature information across a layer, not just per-parameter scaling, at memory cost of $m^2 + n^2$ rather than $(mn)^2$. It's genuinely strong empirically, and adoption is slow because you have to compute inverse fourth-roots of those matrices periodically, which is numerically awkward and expensive.

**20. When would you actually pick Shampoo or K-FAC over Adam?**
Specific small-model regimes where the per-step compute overhead is acceptable, generalization is paramount, and you have engineering bandwidth. In standard LLM pretraining at scale, AdamW dominates because the implementation is battle-tested and the gains from second-order are not large enough to justify the complexity.

> **Saying it out loud.** Honestly, rarely — and saying so is the right answer. You'd consider it when the model is small enough that the per-step overhead doesn't dominate, generalisation matters more than throughput, and you have the engineering time to maintain it. At LLM pretraining scale AdamW wins not because it's the best update rule but because it's cheap, stable, and every distributed training stack already handles its state correctly. The recent Shampoo results at moderate scale are real, though, so the honest framing is "not yet default, but the gap is narrowing.\"

---

## D. Why optimizers fail and how to debug

**21. Adam diverges at step 200. What's going on and how do you fix it?**
Most likely: warmup is too short or peak $\eta$ is too high. The $\hat v$ estimate becomes unreliable when an outlier gradient hits before the variance is stable. Fix: extend warmup to 2000+ steps, lower peak $\eta$ 3x. Secondary fixes: gradient clipping at norm 1.0, increase $\beta_2$ to 0.9999 for slower variance updates.

> **Saying it out loud.** Divergence a couple hundred steps in almost always means the warmup is too short or the peak learning rate too high. Early on the second-moment estimate is built from very few samples, so one outlier gradient produces a wildly wrong denominator and a huge step, and you never recover. The fixes in order: lengthen warmup to a couple of thousand steps, drop peak learning rate by about 3x, and make sure gradient clipping at norm 1.0 is actually on. Raising $\beta_2$ helps too, since it makes the variance estimate less jumpy. If it still blows up, check bias correction is implemented — that's the classic from-scratch bug.

**22. Adam works on smaller batch but not on larger.**
LR scaling rule. For batch size scaling $k$, Adam typically needs $\sqrt{k}$ LR scaling. If you doubled batch size and kept $\eta$ constant, you may have under-scaled. Also: longer warmup is needed for larger batches because each step now has bigger effective magnitude.

> **Saying it out loud.** That's a learning-rate scaling problem, not an optimizer problem. When you increase batch size the gradient gets less noisy, so you can and should take bigger steps — for SGD the rule of thumb is linear scaling, for Adam it's closer to square-root. If you doubled the batch and kept the learning rate, you're now under-stepping and it looks like the optimizer stopped working. The other half is warmup: bigger batches mean fewer, larger steps, so you need proportionally longer warmup, and this is exactly why very-large-batch training needed tricks like LARS.

**23. Adam learns fast then plateaus.**
Schedule decayed too aggressively. Or $\hat v$ accumulated outliers and is now over-suppressing the update direction. Or the LR finder picked a value that's only good for early training. Solutions: warm restart, switch to a less aggressive schedule, or transition to SGD for the final phase.

> **Saying it out loud.** Usually the schedule, not the optimizer. Cosine decay drops the learning rate a lot in the middle of training, so what looks like "the model stopped learning" is often just "the steps got small." The second candidate is that the second moment has absorbed some large gradients and is now over-damping updates in the directions that matter. And a third: whatever learning rate you found early is frequently too small for the middle of training. So I'd check the schedule first, then consider a warm restart, and only then reach for a different optimizer.

**24. SGD with momentum is unstable on transformers.**
Expected. Transformers have ill-conditioned gradients across layers — embedding tables and FFN layers have wildly different scales. SGD's single global LR can't accommodate this. Fix: switch to AdamW. SGD+momentum without per-layer scaling is essentially never the right answer for transformers.

> **Saying it out loud.** That's expected, and the reason is that transformers are badly conditioned across layers. Embedding tables get sparse gradients, LayerNorm parameters get small dense ones, and FFN matrices get large ones — the scales differ by orders of magnitude, and SGD has exactly one global learning rate to serve all of them. Whatever value works for the FFN destroys the embeddings or leaves them frozen. Adam fixes this by giving each parameter its own scale, which is the actual reason adaptive methods took over language modelling. So the answer is: not a bug, switch to AdamW.

**25. Loss spikes occasionally with Adam at the right LR.**
Edge of stability. Common, often benign. Add gradient clipping at norm 1.0 if not already present. Don't reflexively lower LR — that may move you below the optimal operating point.

> **Saying it out loud.** Occasional loss spikes at a good learning rate are normal, and the phrase to use is edge of stability — models tend to drift toward the largest curvature they can tolerate, so you're always operating just short of instability. As long as the loss recovers within a few hundred steps, it's usually benign. The right first move is gradient clipping at norm 1.0 so a single freak batch can't do damage. What I'd resist is reflexively cutting the learning rate, because that trades a cosmetic problem for a genuinely slower run — the usual guidance is to only intervene if a spike fails to recover.

**26. Loss is fine but eval is degrading.**
Probably overfitting. Optimizer can contribute (Adam's preconditioning tends toward sharper minima), but the first move is to add regularization (weight decay, dropout, more data) rather than change optimizer.

> **Saying it out loud.** Train loss falling while eval rises is overfitting, and the optimizer is at most a contributing factor. Adam does have a documented tendency toward sharper minima that generalise slightly worse, but that's a second-order effect next to "not enough data or not enough regularisation." So the first moves are weight decay, dropout, more or more varied data, and early stopping. Changing optimizers to fix a generalisation gap is treating a symptom, and it usually costs you the training speed that made Adam attractive in the first place.

**27. After a checkpoint reload, training is unstable.**
Likely: optimizer state wasn't loaded. Adam without $m_t, v_t$ state is just Adam-from-scratch with incorrect $t$. Always serialize and restore optimizer state, including $t$.

> **Saying it out loud.** Nine times out of ten, the optimizer state wasn't saved or wasn't restored. Adam's behaviour depends on the two moment buffers and the step counter, and if you reload weights without them you've effectively restarted Adam from zero — with no bias correction applied properly, so the first steps are far too large and you get a visible loss spike. The fix is to checkpoint the optimizer state dictionary along with the weights, including $t$, and to also restore the learning-rate scheduler position. It's a boring answer, and it is genuinely the most common cause.

**28. Your team's Adam runs work; mine doesn't. What do you check?**
Optimizer state (loaded?), bias correction (correctly implemented?), $\varepsilon$ placement ($\sqrt{\hat v} + \varepsilon$ or $\sqrt{\hat v + \varepsilon}$ — different!), warmup length (matches reference?), batch size and LR scaling (compatible?), gradient clipping (in place?). The $\varepsilon$ placement is a real bug source — PyTorch and TF have differed historically.

> **Saying it out loud.** I'd work down a checklist rather than theorise. Is the optimizer state actually being loaded on resume. Is bias correction implemented, and is the step counter right. Where is epsilon — inside or outside the square root, because PyTorch and TensorFlow have differed and that changes behaviour near zero. Is warmup the same length. Is the learning rate scaled for my batch size. Is gradient clipping on. Almost every "the same code doesn't work for me" story lands on one of those, and the epsilon placement is the sneakiest because nothing errors, the run is just subtly worse.

**29. Why might LARS or LAMB show up?**
Very large batch training (>16K). Per-layer trust ratios prevent any single layer's update from being too large relative to its parameters. Mostly superseded by muP at frontier labs but appears in some published large-batch ablations.

> **Saying it out loud.** LARS and LAMB show up when the batch is enormous, over about 16 thousand, and normal scaling rules stop holding. Their trick is a per-layer trust ratio: the update is scaled so that each layer moves by a fixed fraction of its own weight norm, which stops one layer taking a huge relative step just because its gradients happen to be large. That's what made 32K-batch ResNet and BERT training work. They're mostly historical at frontier labs now — muP and better warmup schedules cover the same ground — but they're the right answer to "how do you train at extreme batch size.\"

**30. What's muP and how does it relate to optimizers?**
muP changes initialization scales and per-layer LR factors so the optimal LR is invariant under model width. Sweep LR cheaply on a small model, scale up. Doesn't replace the optimizer (you still use AdamW under muP) — it changes how parameters and learning rates are scaled across model sizes.

> **Saying it out loud.** muP is a way of parameterising a model so the best learning rate doesn't change when you make it wider. Normally you'd have to re-tune the learning rate at every scale, which is unaffordable at large sizes — so people guess, and guess wrong. muP fixes initialisation scales and per-layer learning-rate multipliers so that the optimum transfers, letting you sweep cheaply on a small model and use the result directly on the big one. It's not an optimizer, it sits underneath one — you still run AdamW. The payoff is that you get to tune the most important hyperparameter at 1/1000th the cost.

---

## E. Theoretical / advanced

**31. Why does Adam achieve lower training loss but worse test loss than SGD on some tasks?**
Adam's preconditioning biases the optimizer toward sharper minima. Several explanations: (a) per-parameter rescaling reduces SGD-style gradient noise that biases toward flat minima, (b) $1/\sqrt{\hat v}$ directs more aggressive updates toward sharper directions, (c) different effective trajectory shape. Mitigations: AdamW (helps), longer training (helps), AdamSwitch to SGD for last epochs (sometimes helps).

> **Saying it out loud.** The observation is real: Adam often reaches lower training loss but slightly worse test loss than well-tuned SGD, especially on vision. The leading explanation is about the shape of the minimum you land in — SGD's raw gradient noise pushes it toward flat basins that generalise well, and Adam's per-parameter rescaling reshapes that noise and lets it settle into sharper ones. The practical mitigations are decoupled weight decay, which closes most of the gap, training longer, and sometimes switching to SGD for the final phase. It's also worth saying this gap largely disappears for language models, where SGD isn't a viable option anyway.

**32. What's the convergence rate of SGD on convex problems?**
For smooth convex: $O(1/\sqrt{T})$ with constant LR; $O(1/T)$ with optimal LR or strong convexity. With Polyak averaging: $O(1/T)$. With Nesterov on smooth strongly-convex: $O(\exp(-c \cdot T / \sqrt{\kappa}))$. Real deep learning is non-convex so these are loose upper bounds, but they motivate why momentum and acceleration matter in theory.

> **Saying it out loud.** For smooth convex problems, plain SGD is $O(1/\sqrt{T})$ with a constant step size, and $O(1/T)$ with a well-chosen schedule or strong convexity. Nesterov acceleration gets you to $O(1/T^2)$ in the smooth convex case, and linear convergence in the strongly-convex one. I'd caveat immediately that deep learning is non-convex, so none of these are guarantees about what you'll see. What they're good for is intuition about *why* momentum helps and why conditioning matters — the rates all degrade with the condition number, which is exactly the quantity preconditioning attacks.

**33. What's the implicit regularization perspective on SGD vs. Adam?**
SGD's mini-batch noise has scale $\eta/B$, biasing toward flat minima. Adam's preconditioning rescales per parameter, changing the noise structure: noise in low-gradient parameters is amplified, noise in high-gradient parameters is suppressed. The net effect is a different (and sometimes weaker) implicit regularization than SGD.

> **Saying it out loud.** The idea is that the optimizer doesn't just find *a* minimum, it's biased toward a particular kind. SGD's mini-batch noise scales with learning rate over batch size, and that noise makes sharp minima unstable — you get shaken out of them — so it settles in flat ones, which generalise better. Adam changes the noise structure by rescaling per parameter, amplifying it where gradients are small and damping it where they're large, so the implicit bias is different and often weaker. That's the cleanest available story for why Adam sometimes generalises worse, and it's why the ratio of learning rate to batch size matters more than either number alone.

**34. Why don't we use second-order methods for deep learning?**
Storage $O(d^2)$ and inversion $O(d^3)$. For $10^9$ parameters, that's $10^{18}$ Hessian entries and $10^{27}$ inversion operations — wildly infeasible. Stochastic-Hessian approximations (Sophia, K-FAC, Shampoo) trade exactness for tractability. Even those are expensive enough that AdamW remains dominant in production.

> **Saying it out loud.** Pure cost. The Hessian for a billion parameters has $10^{18}$ entries — you can't store it, let alone invert it, which is another three orders of magnitude worse. Even if you could, it's stochastic and changing every step, so an exact inverse of a noisy Hessian isn't obviously worth much. So everything practical is an approximation: diagonal for Adam, Kronecker-factored for Shampoo and K-FAC, stochastic estimates for Sophia. And even those cost enough per step that AdamW's cheapness keeps winning — the bar isn't "converges in fewer steps," it's "converges in less wall-clock time.\"

**35. What's the natural gradient and how does it relate to optimizers?**
Natural gradient is $F^{-1} g$, where $F$ is the Fisher information matrix (expected Hessian of log-likelihood). It's the steepest descent in distribution space rather than parameter space — optimal in an information-geometric sense. K-FAC approximates $F$ block-diagonally; SGD ignores it. The relationship: under specific assumptions, RMSProp's $1/\sqrt{\mathbb{E}[g^2]} \approx 1/\sqrt{\mathrm{diag}(F)}$, giving Adam an information-geometric interpretation.

> **Saying it out loud.** Natural gradient says you should measure distance by how much the model's *output distribution* changes, not by how much the parameters change. Two networks can have very different weights and identical behaviour, so parameter space is the wrong ruler. The natural gradient is the inverse Fisher times the gradient, which is the steepest descent under that better ruler. K-FAC is a tractable block approximation of it. And the nice connection: Adam's running average of squared gradients is roughly the diagonal of the Fisher, so Adam is a very cheap, very approximate natural-gradient method — which is the elegant one-liner for this question.

**36. Why might $\varepsilon$ placement matter? $\sqrt{\hat v} + \varepsilon$ vs $\sqrt{\hat v + \varepsilon}$?**
$\sqrt{\hat v} + \varepsilon$: $\varepsilon$ is added after the square root, so it's a floor on the divisor. Standard Adam.
$\sqrt{\hat v + \varepsilon}$: $\varepsilon$ is added inside, behaves like a tiny variance prior. Almost equivalent for $\hat v \gg \varepsilon$, but different near zero.
Different libraries have used different conventions historically; PyTorch uses $\sqrt{\hat v} + \varepsilon$. Worth knowing if you're translating between codebases.

> **Saying it out loud.** Because the two behave differently exactly where it matters, near zero. With epsilon outside, it's a hard floor on the divisor, so the biggest possible step is eta over epsilon. With epsilon inside, it acts like a small variance prior and the effective floor is the square root of epsilon — for $10^{-8}$ that's $10^{-4}$, four orders of magnitude apart. So the same nominal epsilon means very different caps on the per-parameter step. PyTorch uses the outside form and that's the standard Adam. This bites people porting code between frameworks: nothing errors, the run is just quietly worse.

---

## F. Quick fire

**37.** *Default Adam betas?* $0.9, 0.999$.
**38.** *AdamW weight decay for LLM pretrain?* $0.1$.
**39.** *Lion LR vs AdamW LR?* Lion ~3–10x lower.
**40.** *Sophia per-step compute cost?* ~25% more than AdamW (one extra HVP per step).

---

## Self-grading

If you can't answer 1–10, you don't know optimizers. If you can't answer 11–20, you don't know modern LLM training. If you can't answer 21–36, you'll struggle in frontier-lab applied scientist screens. Aim for 30+/40 cold before walking in.
