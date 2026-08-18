# Whiteboard Derivations — Interview Grill

> 30 questions to verify you can do each must-master derivation cold. Drill until you can write each proof in 5 min.

---

## A. Backpropagation

**1. 2-layer MLP forward — write it.**
$z_1 = W_1 x + b_1$; $h_1 = \sigma(z_1)$; $z_2 = W_2 h_1 + b_2$; $\hat{y} = \mathrm{softmax}(z_2)$.

**2. Cross-entropy + softmax gradient at output?**
$\delta_2 = \hat{y} - y$.

**3. Backward weight gradient?**
$\nabla_{W_\ell} = \delta_\ell h_{\ell-1}^\top$.

**4. Backward error propagation?**
$\delta_\ell = (W_{\ell+1}^\top \delta_{\ell+1}) \odot \sigma'(z_\ell)$.

> **Saying it out loud (narrate while you write).** "I'll put the forward pass across the top so I can point at it, then walk backwards. At the output, cross-entropy against softmax collapses to `y-hat minus y` — I can derive that in two lines if you want, the softmax Jacobian's `y-hat_i` cancels the `1 over y-hat_i` from the loss. Then every layer repeats the same two moves: the weight gradient is this layer's delta times the previous activation transposed, and to keep going backwards I multiply by `W` transpose and gate elementwise by the activation derivative." Say the invariant out loud as you go: each `dW` must come out the same shape as its `W`, and that check catches nearly every transpose error live. The named failure mode here isn't the math, it's forgetting to divide by batch size — the gradient then scales with `n` and your effective learning rate changes every time you resize the batch.

---

## B. Attention

**5. Scaled dot-product formula?**
$\mathrm{softmax}(QK^\top/\sqrt{d_k}) V$.

**6. Why $\sqrt{d_k}$?**
Variance of $QK^\top$ is $d_k$ if $Q, K$ unit-var. Keep it 1 → no softmax saturation.

**7. Multi-head reshape order?**
$[B, L, D] \to [B, L, H, D/H] \to [B, H, L, D/H]$.

**8. Mask method?**
Add $-\infty$ before softmax.

> **Saying it out loud (narrate while you write).** "Scores first — `Q K` transpose over root `d_k`, and I'll draw that box as `L` by `L` and say what it means: row `i` is how much token `i` attends to everyone. Softmax across the row, so over keys, the last axis. Times `V`, which contracts the sequence dimension and gives me back one vector per token." Then the why, since they'll ask: a dot product in `d_k` dimensions has variance `d_k`, so unscaled scores grow like root `d_k`, the softmax saturates toward one-hot, and the gradient dies — dividing restores unit variance. Masking is additive and goes *before* the softmax, because `exp` of minus infinity is exactly zero and the remaining weights still renormalize; multiplying by zero afterwards leaves a broken denominator. The gotcha to name: mask an entire row and every entry is minus infinity, which gives zero over zero and NaN, so real implementations use a large finite negative number.

---

## C. OLS

**9. Gradient of $\frac{1}{2}\|y - Xw\|^2$?**
$X^\top(Xw - y)$.

**10. Closed form?**
$\hat{w} = (X^\top X)^{-1} X^\top y$.

**11. Hessian?**
$X^\top X$. PSD always; PD if $X$ full column rank.

**12. Geometric interpretation?**
$\hat{y}$ = projection of $y$ onto $\mathrm{Col}(X)$.

> **Saying it out loud (narrate while you write).** "Expand the squared norm, differentiate term by term, and you get `X` transpose times `Xw minus y`. Set it to zero — those are the normal equations — and solve for `w`." Then say the two things that turn a formula into understanding. "The Hessian is `X` transpose `X`, positive semi-definite for any `X` whatsoever, so this is convex and the stationary point is the global minimum; it's strictly positive definite, and therefore uniquely invertible, exactly when the columns of `X` are linearly independent. And geometrically the fit is an orthogonal projection of `y` onto the column space of `X`, which is just a restatement of the normal equations — the residual is perpendicular to every feature." Land the practical note: you never actually invert that matrix, you solve with QR or Cholesky, because collinear features make it near-singular and the inverse amplifies noise.

---

## D. Logistic regression

**13. Sigmoid derivative?**
$\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**14. BCE gradient w.r.t. logits?**
$dz = p - y$.

**15. BCE gradient w.r.t. weights?**
$\nabla_w = (p - y) x$.

**16. Hessian PSD?**
Yes: $\sum p(1-p) x x^\top$. Always PSD → loss convex.

> **Saying it out loud (narrate while you write).** "The sigmoid derivative is `p` times `one minus p` — I'll write that first because it's the pivot of everything. Differentiating binary cross-entropy with respect to `p` and combining fractions gives `p minus y` over `p times one minus p`. Multiply the two" — strike them through — "and the denominator cancels exactly, leaving `dL/dz` equals `p minus y`. Chain out to the weights and it's residual times input, same shape of answer as linear regression, which is the canonical-link property of GLMs." Then the convexity line, because it's free points: the Hessian is a sum of `p(1-p) x x` transpose, which is PSD for every dataset, so there are no local minima. And the number worth knowing: the sigmoid derivative peaks at one quarter at `z` equals zero, which is precisely why sigmoid-plus-MSE has such feeble gradients and cross-entropy doesn't.

---

## E. KL and information theory

**17. KL definition?**
$\sum p(x) \log(p(x)/q(x))$.

**18. KL non-negative — prove.**
Jensen on $-\log$. $\mathrm{KL}(p\|q) = -\sum p \log(q/p) \geq -\log \sum p (q/p) = -\log 1 = 0$.

**19. Forward vs reverse KL?**
Forward: $\mathrm{KL}(p^* \| q)$, mode-covering. Reverse: $\mathrm{KL}(q \| p^*)$, mode-seeking.

**20. MLE = forward KL?**
$\arg\max \mathbb{E}_{p^*}[\log q] = \arg\min \mathrm{KL}(p^* \| q)$ + constant.

> **Saying it out loud (narrate while you write).** "KL is the expected log-ratio under `p` — how many extra bits I pay for coding with `q` when the truth is `p`. To show it's non-negative I write minus KL as the expectation of `log q over p`, and since `log` is concave, Jensen pushes the expectation inside: that's at most `log` of the expectation, which is `log` of the sum of `q`, which is `log 1`, which is zero." That's the whole proof, three lines. "Equality only when the ratio is constant, so only when the distributions coincide." Then the asymmetry with its consequence, because that's the part that gets probed: forward KL — the MLE direction — is mass-covering, since `p` having mass where `q` has none blows the ratio up, so your model smears to cover everything. Reverse KL, what variational inference minimizes, is mode-seeking and collapses onto a single mode. Blurry versus incomplete: that's the named tradeoff.

---

## F. EM and GMM

**21. E-step in GMM?**
$\gamma_{ik} = \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) / \sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)$.

**22. M-step mean?**
$\mu_k = \sum_i \gamma_{ik} x_i / \sum_i \gamma_{ik}$.

**23. Why EM converges?**
ELBO is tight at current params after E-step; M-step maximizes ELBO; likelihood monotone non-decreasing.

> **Saying it out loud (narrate while you write).** "The chicken-and-egg problem is that cluster assignments and cluster parameters each need the other. EM alternates. E-step: compute soft responsibilities by Bayes' rule — prior times density, normalized across components. M-step: refit each Gaussian by weighted maximum likelihood using those responsibilities." Then the convergence argument, which is what's really being tested: "decompose the log-likelihood into the ELBO plus a KL term. The E-step sets `q` to the exact posterior, which kills that KL and makes the bound touch the likelihood. The M-step raises the bound. Since the bound was equal to the likelihood before the step, and it went up, the likelihood went up too — monotone and bounded, so it converges." Land it on failure modes: only to a local optimum, so initialization matters, and a component can collapse onto one point with zero variance and infinite likelihood, which is why you floor the covariance.

---

## G. SVM

**24. Primal SVM?**
$\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$.

**25. From Lagrangian, what does $\partial_w$ give?**
$w = \sum \alpha_i y_i x_i$.

**26. Support vectors?**
$\alpha_i > 0$ — points on margin or violating it.

**27. Kernel trick — what changes in dual?**
Replace $x_i^\top x_j$ with $K(x_i, x_j)$.

> **Saying it out loud (narrate while you write).** "The primal minimizes half the squared norm of `w` subject to every point clearing the margin — minimizing the norm is the same as maximizing the margin, since the margin is one over the norm. I form the Lagrangian, differentiate with respect to `w`, set it to zero, and get `w` as a weighted sum of the training points themselves. Substitute that back and the whole thing turns into a problem in the alphas where the data appears only through inner products `x_i` transpose `x_j`." That's the load-bearing observation, so say it twice: only inner products. "Which means I can replace them with any valid kernel and get a nonlinear boundary without ever computing the feature map." Then KKT: complementary slackness forces alpha to be zero for every point comfortably inside its margin, so only the points on or violating the margin have nonzero weight — the support vectors — and you could delete the rest of the training set without changing the model at all.

---

## H. RoPE, DPO, ELBO

**28. RoPE relative property?**
$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$. Inner product depends on relative position only.

**29. DPO derivation key step?**
Substitute optimal RLHF policy $\pi^* \propto \pi_{\mathrm{ref}} \exp(r/\beta)$ into Bradley-Terry; reward cancels in differences.

**30. ELBO from log-marginal?**
$\log p(x) \geq \mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z|x)]$ via Jensen on log.

> **Saying it out loud (narrate while you write).** "Three separate one-trick derivations, and each has exactly one step to remember. RoPE: rotate query and key by an angle proportional to position, then note that in the dot product the transpose of a rotation is a rotation by the negative angle, and rotations compose by adding angles — so the absolute positions subtract and only `n` minus `m` survives. That's relative position for free. DPO: the KL-regularized RLHF objective has a closed-form optimum, so you can invert it and write the reward as beta times a log policy ratio plus a partition term that depends only on the prompt — and because Bradley-Terry only ever uses reward *differences*, that partition term cancels and the reward model disappears entirely. ELBO: multiply and divide by `q` inside the intractable integral, apply Jensen to the concave log, and you get a computable lower bound whose gap to the truth is exactly the KL between your approximate posterior and the real one." Each one is a cancellation — say which thing cancels and you've said the derivation.

---

## Quick fire

**31.** *Cross-entropy + softmax gradient?* $p - y$.
**32.** *Attention scale?* $1/\sqrt{d_k}$.
**33.** *OLS Hessian?* $X^\top X$.
**34.** *Sigmoid derivative at $z=0$?* 1/4.
**35.** *KL inequality direction?* $\geq 0$.
**36.** *EM convergence?* Likelihood monotone.
**37.** *SVM support vector condition?* $\alpha > 0$.
**38.** *RoPE encoding type?* Relative.
**39.** *DPO eliminates?* Reward model.
**40.** *ELBO gap to log-likelihood?* $\mathrm{KL}(q \| p(z|x))$.

---

## Self-grading

For each of the 8 main derivations:
- 5 min cold? Pass.
- Need notes? Drill more.
- Stuck on a step? Re-read the deep dive.

Aim: all 8 derivations whiteboard-ready in 5 min each.
