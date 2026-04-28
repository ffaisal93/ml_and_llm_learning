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

---

## F. EM and GMM

**21. E-step in GMM?**
$\gamma_{ik} = \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) / \sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)$.

**22. M-step mean?**
$\mu_k = \sum_i \gamma_{ik} x_i / \sum_i \gamma_{ik}$.

**23. Why EM converges?**
ELBO is tight at current params after E-step; M-step maximizes ELBO; likelihood monotone non-decreasing.

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

---

## H. RoPE, DPO, ELBO

**28. RoPE relative property?**
$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$. Inner product depends on relative position only.

**29. DPO derivation key step?**
Substitute optimal RLHF policy $\pi^* \propto \pi_{\mathrm{ref}} \exp(r/\beta)$ into Bradley-Terry; reward cancels in differences.

**30. ELBO from log-marginal?**
$\log p(x) \geq \mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z|x)]$ via Jensen on log.

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
