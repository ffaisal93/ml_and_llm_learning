# Optimizers — Interview Grill

> 40 questions focused on **optimizer algorithms specifically** — different angle from the LR-centric grill in `02_gradient_descent/INTERVIEW_GRILL.md`. Use both.

---

## A. Algorithmic foundations

**1. What's the relationship between optimizers and Newton's method?**
Newton uses $H_t^{-1} g_t$ as the update direction — accounting for second-order curvature. Storing $H$ is $O(d^2)$ and inverting is $O(d^3)$, infeasible at scale. Every modern optimizer is some cheap approximation: SGD = identity preconditioner; Adam/RMSProp = diagonal $1/\sqrt{\hat v}$ preconditioner approximating $\operatorname{diag}(H)^{-1/2}$; Shampoo = block-Kronecker; Sophia = stochastic Hutchinson estimate of $\operatorname{diag}(H)$.

**2. Walk me through SGD with classical momentum.**

$$
v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Velocity $v_t$ is an exponentially-weighted sum of past gradients. $\beta = 0.9$ is standard. Effective gradient horizon is $1/(1-\beta) \approx 10$. Helps convergence in ill-conditioned valleys by averaging out perpendicular oscillations and reinforcing the persistent direction along the valley.

**3. What's Nesterov momentum and why is it different?**
Computes the gradient at the *lookahead* position (where momentum will take you anyway):

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t - \eta \beta v_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

Theoretically improves convex convergence from $O(1/T)$ to $O(1/T^2)$ for smooth strongly-convex problems. Empirically often slightly better than Polyak momentum.

**4. Walk me through RMSProp.**

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t^2, \qquad \theta_{t+1} = \theta_t - \eta \cdot \frac{g_t}{\sqrt{v_t} + \varepsilon}
$$

Per-parameter rescaling by RMS of recent gradients. The second-moment $\mathbb{E}[g g^\top]$ is the **Fisher information matrix** (not the Hessian directly). For likelihood losses, $F = H$ only at a stationary point — so "diagonal Hessian approximation" is loose; "diagonal Fisher" is more accurate. Removes most LR-tuning sensitivity that plain SGD has.

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

**6. Why is bias correction necessary?**
$m_t$ and $v_t$ initialize at zero. Without correction, the first $\sim 1/(1-\beta)$ steps have moments that are underestimates of the true running averages — biased low. For $\beta_2 = 0.999$, $\hat v$ is biased for ~1000 steps. Without correction, the early effective LR $\eta / \sqrt{\hat v}$ is too large, training often diverges. The bias correction $1/(1-\beta^t)$ exactly inverts the geometric-series discount.

**7. What does $\varepsilon$ in Adam control?**
Two roles: (a) numerical floor preventing $1/\sqrt{\hat v}$ from blowing up when $\hat v \approx 0$, (b) implicit cap on per-parameter LR — when $\sqrt{\hat v} \ll \varepsilon$, the update is $(\eta/\varepsilon) \cdot \hat m$, so dimensions with very small gradients still get sensible updates. Some recipes set $\varepsilon = 10^{-3}$ for embeddings to dampen aggressive updates on rare tokens.

**8. What if you set $\beta_2 = 0.9999$?**
The second-moment horizon grows to ~10000 steps. Pros: more robustness to outlier gradients. Cons: very slow to track changes in gradient statistics — when training transitions from warmup to the main phase, $\hat v$ lags badly. Empirically, $\beta_2 = 0.999$ is a sweet spot. $\beta_2 = 0.95$ is sometimes used for very long pretraining for the opposite reason: faster reaction.

---

## B. AdamW vs Adam vs L2

**9. What is AdamW?**
Adam with **decoupled** weight decay. The update becomes:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta \cdot \lambda \cdot \theta_t
$$

Weight decay applied directly to $\theta$ after the Adam update, not added to the gradient.

**10. Why isn't Adam-with-L2 equivalent to AdamW?**
Adam-with-L2 adds $\lambda \theta$ to the gradient: $g_t \leftarrow g_t + \lambda \theta_t$. Then $v_t$ accumulates $(g_t + \lambda \theta_t)^2$, the regularization term gets divided by $\sqrt{\hat v}$, and parameters with high gradient variance see weakened L2. Decay strength becomes non-uniform across parameters in a way nobody intends. AdamW separates decay from preconditioning; every parameter shrinks by exactly $\eta \cdot \lambda$ regardless of its gradient statistics.

**11. For SGD, are L2 and weight decay equivalent?**
Yes. Gradient of $(\lambda/2)\|\theta\|^2$ is $\lambda \theta$, so SGD with explicit decay is identical to SGD with L2. They diverge only when there's preconditioning (Adam, RMSProp, K-FAC).

**12. What's a typical AdamW weight decay value for LLMs?**
$\lambda = 0.1$ for pretraining is the modern default. $\lambda = 0.01$ is more typical for vision and smaller models. SFT and DPO usually use $0.0$ or very small ($0.001$).

**13. Why do attention layers and embeddings sometimes have different weight decay?**
Embedding parameters often see sparse gradient updates (only sampled tokens get gradient). Decay applied uniformly per step over-shrinks rare-token embeddings. Common fixes: zero weight decay on embeddings, layer-norm parameters, and biases; non-zero decay only on weight matrices.

---

## C. Lion, Sophia, and modern alternatives

**14. Walk me through Lion.**
Sign-based update:

$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad &\text{(interpolation)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \operatorname{sign}(c_t) - \eta \cdot \lambda \cdot \theta_t \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t \quad &\text{(momentum)}
\end{aligned}
$$

Update magnitude per parameter is exactly $\eta$ (modulo decay). No second moment, no division, no square root. Memory: one state buffer per param vs Adam's two.

**15. Why does Lion sometimes work as well as AdamW?**
Sign normalization is an extreme form of per-parameter rescaling — like Adam's $1/\sqrt{\hat v}$ taken to the limit. When gradient magnitudes are similar across parameters (after normalization layers do their job), the normalization in Adam is doing less work than people assume; sign is "good enough" and saves memory.

**16. What's the LR difference between Lion and AdamW?**
Lion's optimal $\eta$ is typically 3–10x smaller than AdamW's, because sign updates are "always full magnitude" while Adam's updates can be smaller for low-gradient parameters. Lion's optimal weight decay is typically 3x larger.

**17. What's Sophia?**
Adam-like, but uses a stochastic Hessian-diagonal estimate via Hutchinson's estimator instead of $\sqrt{\hat v}$:

$$
\hat h_t = \operatorname{clip}(\text{stoch\_hutchinson\_diag\_H}, \rho)
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t}{\max(\gamma \cdot \hat h_t, \varepsilon)} - \eta \cdot \lambda \cdot \theta_t
$$

Hutchinson uses $\operatorname{diag}(H) \approx \mathbb{E}[v \odot Hv]$ for random $v$; $Hv$ is computed via Hessian-vector product (one extra backward pass). Reportedly converges in fewer steps than AdamW on language modeling. Cost: ~25% more compute per step.

**18. Why isn't Sophia universally adopted?**
(a) Per-step compute cost. (b) Implementation complexity (HVP via PyTorch isn't a one-liner). (c) Public benchmarks at 70B+ scale are scarce. (d) AdamW is "good enough" — frontier labs are conservative about changing the optimizer mid-training run.

**19. What's Shampoo?**
Per-layer Kronecker-factored preconditioner. For an $m \times n$ weight matrix, store left factor $L_t$ ($m \times m$) and right factor $R_t$ ($n \times n$). Update:

$$
W_{t+1} = W_t - \eta \cdot L_t^{-1/4} G_t R_t^{-1/4}
$$

Memory $O(m^2 + n^2)$ per layer instead of $O(d^2)$. Empirically state-of-the-art on some tasks but adopted slowly because of implementation complexity and the cost of computing matrix inverse-roots.

**20. When would you actually pick Shampoo or K-FAC over Adam?**
Specific small-model regimes where the per-step compute overhead is acceptable, generalization is paramount, and you have engineering bandwidth. In standard LLM pretraining at scale, AdamW dominates because the implementation is battle-tested and the gains from second-order are not large enough to justify the complexity.

---

## D. Why optimizers fail and how to debug

**21. Adam diverges at step 200. What's going on and how do you fix it?**
Most likely: warmup is too short or peak $\eta$ is too high. The $\hat v$ estimate becomes unreliable when an outlier gradient hits before the variance is stable. Fix: extend warmup to 2000+ steps, lower peak $\eta$ 3x. Secondary fixes: gradient clipping at norm 1.0, increase $\beta_2$ to 0.9999 for slower variance updates.

**22. Adam works on smaller batch but not on larger.**
LR scaling rule. For batch size scaling $k$, Adam typically needs $\sqrt{k}$ LR scaling. If you doubled batch size and kept $\eta$ constant, you may have under-scaled. Also: longer warmup is needed for larger batches because each step now has bigger effective magnitude.

**23. Adam learns fast then plateaus.**
Schedule decayed too aggressively. Or $\hat v$ accumulated outliers and is now over-suppressing the update direction. Or the LR finder picked a value that's only good for early training. Solutions: warm restart, switch to a less aggressive schedule, or transition to SGD for the final phase.

**24. SGD with momentum is unstable on transformers.**
Expected. Transformers have ill-conditioned gradients across layers — embedding tables and FFN layers have wildly different scales. SGD's single global LR can't accommodate this. Fix: switch to AdamW. SGD+momentum without per-layer scaling is essentially never the right answer for transformers.

**25. Loss spikes occasionally with Adam at the right LR.**
Edge of stability. Common, often benign. Add gradient clipping at norm 1.0 if not already present. Don't reflexively lower LR — that may move you below the optimal operating point.

**26. Loss is fine but eval is degrading.**
Probably overfitting. Optimizer can contribute (Adam's preconditioning tends toward sharper minima), but the first move is to add regularization (weight decay, dropout, more data) rather than change optimizer.

**27. After a checkpoint reload, training is unstable.**
Likely: optimizer state wasn't loaded. Adam without $m_t, v_t$ state is just Adam-from-scratch with incorrect $t$. Always serialize and restore optimizer state, including $t$.

**28. Your team's Adam runs work; mine doesn't. What do you check?**
Optimizer state (loaded?), bias correction (correctly implemented?), $\varepsilon$ placement ($\sqrt{\hat v} + \varepsilon$ or $\sqrt{\hat v + \varepsilon}$ — different!), warmup length (matches reference?), batch size and LR scaling (compatible?), gradient clipping (in place?). The $\varepsilon$ placement is a real bug source — PyTorch and TF have differed historically.

**29. Why might LARS or LAMB show up?**
Very large batch training (>16K). Per-layer trust ratios prevent any single layer's update from being too large relative to its parameters. Mostly superseded by muP at frontier labs but appears in some published large-batch ablations.

**30. What's muP and how does it relate to optimizers?**
muP changes initialization scales and per-layer LR factors so the optimal LR is invariant under model width. Sweep LR cheaply on a small model, scale up. Doesn't replace the optimizer (you still use AdamW under muP) — it changes how parameters and learning rates are scaled across model sizes.

---

## E. Theoretical / advanced

**31. Why does Adam achieve lower training loss but worse test loss than SGD on some tasks?**
Adam's preconditioning biases the optimizer toward sharper minima. Several explanations: (a) per-parameter rescaling reduces SGD-style gradient noise that biases toward flat minima, (b) $1/\sqrt{\hat v}$ directs more aggressive updates toward sharper directions, (c) different effective trajectory shape. Mitigations: AdamW (helps), longer training (helps), AdamSwitch to SGD for last epochs (sometimes helps).

**32. What's the convergence rate of SGD on convex problems?**
For smooth convex: $O(1/\sqrt{T})$ with constant LR; $O(1/T)$ with optimal LR or strong convexity. With Polyak averaging: $O(1/T)$. With Nesterov on smooth strongly-convex: $O(\exp(-c \cdot T / \sqrt{\kappa}))$. Real deep learning is non-convex so these are loose upper bounds, but they motivate why momentum and acceleration matter in theory.

**33. What's the implicit regularization perspective on SGD vs. Adam?**
SGD's mini-batch noise has scale $\eta/B$, biasing toward flat minima. Adam's preconditioning rescales per parameter, changing the noise structure: noise in low-gradient parameters is amplified, noise in high-gradient parameters is suppressed. The net effect is a different (and sometimes weaker) implicit regularization than SGD.

**34. Why don't we use second-order methods for deep learning?**
Storage $O(d^2)$ and inversion $O(d^3)$. For $10^9$ parameters, that's $10^{18}$ Hessian entries and $10^{27}$ inversion operations — wildly infeasible. Stochastic-Hessian approximations (Sophia, K-FAC, Shampoo) trade exactness for tractability. Even those are expensive enough that AdamW remains dominant in production.

**35. What's the natural gradient and how does it relate to optimizers?**
Natural gradient is $F^{-1} g$, where $F$ is the Fisher information matrix (expected Hessian of log-likelihood). It's the steepest descent in distribution space rather than parameter space — optimal in an information-geometric sense. K-FAC approximates $F$ block-diagonally; SGD ignores it. The relationship: under specific assumptions, RMSProp's $1/\sqrt{\mathbb{E}[g^2]} \approx 1/\sqrt{\operatorname{diag}(F)}$, giving Adam an information-geometric interpretation.

**36. Why might $\varepsilon$ placement matter? $\sqrt{\hat v} + \varepsilon$ vs $\sqrt{\hat v + \varepsilon}$?**
$\sqrt{\hat v} + \varepsilon$: $\varepsilon$ is added after the square root, so it's a floor on the divisor. Standard Adam.
$\sqrt{\hat v + \varepsilon}$: $\varepsilon$ is added inside, behaves like a tiny variance prior. Almost equivalent for $\hat v \gg \varepsilon$, but different near zero.
Different libraries have used different conventions historically; PyTorch uses $\sqrt{\hat v} + \varepsilon$. Worth knowing if you're translating between codebases.

---

## F. Quick fire

**37.** *Default Adam betas?* $0.9, 0.999$.
**38.** *AdamW weight decay for LLM pretrain?* $0.1$.
**39.** *Lion LR vs AdamW LR?* Lion ~3–10x lower.
**40.** *Sophia per-step compute cost?* ~25% more than AdamW (one extra HVP per step).

---

## Self-grading

If you can't answer 1–10, you don't know optimizers. If you can't answer 11–20, you don't know modern LLM training. If you can't answer 21–36, you'll struggle in frontier-lab applied scientist screens. Aim for 30+/40 cold before walking in.
