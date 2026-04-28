# Information Theory — Interview Grill

> 40 questions on information theory in ML. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Define entropy.**
$H(p) = -\sum_x p(x) \log p(x)$. Average surprise / number of bits (or nats) needed to encode an outcome from $p$. Maximum at uniform distribution; minimum ($= 0$) at deterministic.

**2. State the bounds on $H(p)$.**
$0 \leq H(p) \leq \log |\mathcal{X}|$. Lower bound at deterministic distributions; upper bound at uniform.

**3. Why is $H$ concave?**
Mixing two distributions produces higher entropy than the average of their entropies. Intuitively: averaging adds uncertainty. Formally: Jensen's inequality applied to $-p \log p$.

**4. Define cross-entropy.**
$H(p, q) = -\sum_x p(x) \log q(x)$. Average code length when encoding samples from $p$ using a code optimal for $q$. Equals $H(p) + \mathrm{KL}(p \,\|\, q)$.

**5. Why is cross-entropy bounded below by entropy?**
$H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q) \geq H(p)$ because KL $\geq 0$. You can't encode samples from $p$ more efficiently than $H(p)$ (Shannon's source coding theorem).

**6. Define KL divergence.**
$\mathrm{KL}(p \,\|\, q) = \sum p(x) \log(p(x) / q(x)) = \mathbb{E}_{x \sim p}[\log p - \log q]$. Measures how $q$ differs from $p$ from $p$'s perspective.

**7. Three properties of KL.**
Non-negative ($\mathrm{KL} \geq 0$, with equality iff $p = q$). Asymmetric ($\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$). Not a metric (triangle inequality fails). Coordinate-invariant under reparameterization.

---

## B. Forward vs reverse KL

**8. What's the difference between forward and reverse KL?**
**Forward KL is mean-seeking; reverse is mode-seeking.** Forward $\mathrm{KL}(p \| q)$ penalizes $q$ being small where $p$ is large → $q$ spreads to cover all of $p$. Reverse $\mathrm{KL}(q \| p)$ penalizes $q$ being large where $p$ is small → $q$ collapses to one mode. MLE uses forward; variational inference uses reverse.

**9. Which one does MLE optimize?**
Forward KL. Minimizing cross-entropy = minimizing $\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)$. Mean-seeking — the model tries to cover all of the data distribution.

**10. Why do MLE-trained models often produce "average" outputs?**
Forward KL is mean-seeking. If the data has multiple modes (e.g., translations have multiple correct outputs), the model spreads probability across them. Sampling produces an average-looking output that may not match any single mode.

**11. When would you use reverse KL?**
Variational inference (where you want a tractable $q$ to fit the most likely mode of the posterior). Some RL methods. Knowledge distillation in some forms.

**12. Why do GANs use Jensen-Shannon?**
$\mathrm{JS} = (1/2) \mathrm{KL}(p \,\|\, M) + (1/2) \mathrm{KL}(q \,\|\, M)$ where $M = (p + q) / 2$. Symmetric, bounded $[0, \log 2]$. The original GAN (Goodfellow 2014) optimizes a JS-related objective. Provides smoother gradients than KL alone.

---

## C. Cross-entropy as ML loss

**13. Why is cross-entropy the standard ML loss?**
Three views: (a) MLE under categorical distribution (likelihood-justified); (b) Forward KL between data and model (mean-seeking); (c) Compression-optimal code length (Shannon).

**14. Cross-entropy gradient w.r.t. logits?**
**Predicted minus actual.** $\partial \mathcal{L} / \partial z = \mathrm{softmax}(z) - \mathrm{one\_hot}(y) = \hat p - y$. Same form as logistic regression — the GLM canonical-link cancellation (sigmoid/softmax derivative kills the $1/p$ from log).

**15. Why don't we use MSE for classification?**
Two reasons. (a) MLE under Bernoulli/categorical mandates cross-entropy; MSE corresponds to a different (Gaussian) generative assumption. (b) MSE+sigmoid has vanishing gradients on confidently-wrong predictions and is non-convex.

**16. Walk me through MLE = forward KL minimization.**
**One-line story**: Maximizing log-likelihood = minimizing KL from data to model. Entropy of the data is fixed, so it drops out.

**Algebra**: $\max_\theta \mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta] = \min_\theta -\mathbb{E}_{p_{\mathrm{data}}}[\log p_\theta] = \min_\theta \mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}} - \log p_\theta] - H(p_{\mathrm{data}}) = \min_\theta \mathrm{KL}(p_{\mathrm{data}} \| p_\theta) - H(p_{\mathrm{data}})$. The $H$ term doesn't depend on $\theta$, so MLE = forward KL minimization.

---

## D. Mutual information

**17. Define mutual information.**
$I(X; Y) = \mathrm{KL}(P(X, Y) \,\|\, P(X) P(Y)) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$. Multiple equivalent forms.

**18. What does MI measure?**
How much knowing $Y$ reduces uncertainty about $X$. If $X \perp Y$, MI $= 0$. If $Y$ perfectly determines $X$, $I(X; Y) = H(X)$.

**19. Properties of MI?**
Non-negative. Symmetric $I(X; Y) = I(Y; X)$. $I(X; X) = H(X)$.

**20. What's InfoNCE?**
$\mathcal{L} = -\mathbb{E}[\log \exp(f(x, y_+)) / \sum_i \exp(f(x, y_i))]$. Contrastive loss; lower bound on $I(X; Y_+)$. Used in CLIP, MoCo, SimCLR. Trains representations that have high MI with positives, low with negatives.

**21. What's the information bottleneck?**
Tishby et al. 2000. Train representations $Z$ to maximize $I(Y; Z)$ (predictive of label) while minimizing $I(X; Z)$ (compress input). Theoretical framework for learning compressed yet predictive representations.

---

## E. Conditional and joint entropy

**22. Define conditional entropy.**
$H(X \mid Y) = -\sum_{x, y} p(x, y) \log p(x \mid y)$. Average uncertainty about $X$ given known $Y$. Always between 0 and $H(X)$.

**23. Chain rule for entropy.**
$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$. Joint = marginal + conditional. Same as probability chain rule but for entropy.

**24. What's $H(Y \mid X)$ in ML?**
The irreducible "noise" any model has to contend with — the lower bound on cross-entropy loss when predicting $Y$ from $X$. If $H(Y \mid X) = 0$, the input perfectly determines the output (deterministic mapping). Otherwise, there's a fundamental limit on prediction quality.

---

## F. KL in machine learning

**25. Where does KL appear in VAE training?**
ELBO: $\log p(x) \geq \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}(q(z \mid x) \,\|\, p(z))$. The KL term penalizes the variational posterior $q$ for being far from the prior $p(z)$.

**26. Where does KL appear in RLHF?**
The objective: $\max \mathbb{E}[r] - \beta \cdot \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$. KL anchor prevents the policy from drifting too far from the SFT reference. Bounds reward hacking.

**27. Where does KL appear in distillation?**
Train student to match teacher's distribution: $\min_{\text{student}} \mathrm{KL}(p_{\text{teacher}} \,\|\, p_{\text{student}})$. Student inherits teacher's full confidence pattern, not just hard predictions.

**28. Why is the KL from the optimal RLHF policy what gives DPO?**
Closed-form solution to the RLHF objective: $\pi^* = \pi_{\text{ref}} \cdot \exp(r/\beta) / Z$. Solve for $r$ and substitute into Bradley-Terry. $Z$ cancels. Result is DPO loss. See `08_training_techniques/ALIGNMENT_DEEP_DIVE.md`.

**29. KL between two Gaussians?**
For $p = \mathcal{N}(\mu_1, \Sigma_1), q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$
\mathrm{KL}(p \,\|\, q) = \tfrac{1}{2}\!\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \operatorname{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)\right]
$$

Closed form in dimensions and means. Famous formula; sometimes asked.

---

## G. Other divergences

**30. What's the relationship between KL and total variation?**
Pinsker's inequality: $\mathrm{TV}(p, q) \leq \sqrt{\mathrm{KL}(p \,\|\, q) / 2}$. Bounds TV by KL. Used in concentration bounds and convergence proofs.

**31. What's an f-divergence?**
A family $D_f(p \,\|\, q) = \sum_x q(x) f(p(x) / q(x))$ for convex $f$ with $f(1) = 0$. KL: $f(t) = t \log t$. Reverse KL: $f(t) = -\log t$. JS, Hellinger, $\chi^2$ are also f-divergences.

**32. What's Wasserstein distance and how is it different?**
Optimal transport distance: minimum cost to "move" mass to transform $p$ into $q$, where cost is integrated over the underlying space. Considers geometry of the space (not just distribution mass). Used in WGAN, optimal transport, distribution matching. Stronger smoothness properties than KL.

**33. Why might WGAN beat vanilla GAN?**
Wasserstein gives smoother gradients than JS, especially when $p$ and $q$ have disjoint supports. Vanilla GAN's JS-based objective can saturate; WGAN's continuous Wasserstein landscape doesn't.

---

## H. Compression connections

**34. State Shannon's source coding theorem.**
The minimum average code length per symbol for a lossless code is $H(p)$. You cannot compress below entropy.

**35. What does cross-entropy tell us about compression?**
Cross-entropy $H(p, q)$ is the average code length when using a code optimal for $q$ to encode samples from $p$. Always $\geq H(p)$. Minimizing cross-entropy = finding a near-optimal code (compressor) for the data.

**36. How does this relate to LLMs?**
LLMs are compressors of their training distribution. Better LM → lower cross-entropy → better compression. Modern LLMs can compress text below traditional methods (gzip, etc.) — Deletang et al. 2023.

---

## I. Numerical and gotcha

**37. What's the log-sum-exp trick?**
For numerical stability: $\log \sum \exp(z) = \max(z) + \log \sum \exp(z - \max(z))$. Without this, large logits overflow $\exp$. Standard in softmax/cross-entropy implementations.

**38. Can KL be infinite?**
Yes. If $p(x) > 0$ but $q(x) = 0$ for some $x$, then $\mathrm{KL}(p \,\|\, q) = \infty$. (Encoding samples from $p$ with $q$'s code is impossible — $q$ assigns 0 probability to outcomes that occur.)

**39. Is the entropy of a mixture always greater than the average entropy?**
Yes (concavity). $H((p + q)/2) \geq (H(p) + H(q))/2$. Mixing increases entropy.

**40. Why do KL divergences appear in PAC-Bayes / generalization bounds?**
KL between learned posterior and prior bounds generalization error. Lower KL (posterior close to prior) = tighter generalization bound. PAC-Bayesian framework underpins much of modern generalization theory.

---

## Quick fire

**41.** *Entropy in bits if log base 2.* True.
**42.** *Entropy in nats if log base $e$.* True.
**43.** *KL is a metric?* No.
**44.** *Cross-entropy = entropy + KL.* True.
**45.** *MLE = forward KL minimization.* True.

---

## Self-grading

If you can't answer 1-15, you don't know information theory. If you can't answer 16-30, you'll struggle on RLHF/distillation interviews. If you can't answer 31-45, frontier-lab interviews will go past you.

Aim for 30+/45 cold.
