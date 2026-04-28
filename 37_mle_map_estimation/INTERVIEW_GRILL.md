# MLE and MAP Estimation — Interview Grill

> 45 questions on MLE, MAP, conjugate priors, and the connections to standard ML losses. Drill until you can answer 30+ cold.

---

## A. Likelihood basics

**1. Define likelihood and log-likelihood.**
$L(\theta) = \prod_i p(x_i|\theta)$, $\ell(\theta) = \sum_i \log p(x_i|\theta)$. Treats $\theta$ as the variable, data as fixed.

**2. Why log?**
Sums beat products numerically (no underflow). Concavity often preserved. Calculus easier.

**3. MLE definition?**
$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

**4. Why is MLE the default in ML?**
Asymptotically consistent + efficient. Reduces to standard losses (cross-entropy, MSE) under standard distributions. Simple to derive and optimize.

---

## B. Standard MLE derivations

**5. Derive MLE for Bernoulli.**
$\ell = s \log\theta + (n-s)\log(1-\theta)$. Set derivative to zero: $\hat{\theta} = s/n = \bar{x}$.

**6. Derive MLE for Gaussian (mean only, variance known).**
$\hat{\mu} = \bar{x}$.

**7. MLE for Gaussian variance?**
$\hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$. Biased — divisor should be $n-1$ for unbiased.

**8. Why is MLE for variance biased?**
$\bar{x}$ is closer to the sample than the true $\mu$. $\sum (x-\bar{x})^2 < \sum (x-\mu)^2$ on average.

**9. MLE for Poisson rate?**
$\hat{\lambda} = \bar{x}$.

**10. MLE for multinomial?**
$\hat{\theta}_k = n_k/n$ — empirical class frequency.

**11. MLE for linear regression — what loss does it correspond to?**
Squared error. $\arg\max \ell$ under Gaussian noise = $\arg\min \sum (y - w^\top x)^2$ = OLS.

**12. MLE for logistic regression — what loss?**
Cross-entropy / log loss. $\sum [y \log \sigma(w^\top x) + (1-y) \log(1-\sigma(w^\top x))]$. No closed form.

**13. Why does logistic regression have no closed-form MLE?**
The score equation is non-linear in $w$ (sigmoid). Need iterative solver: IRLS, gradient descent, Newton-Raphson.

---

## C. Asymptotic theory

**14. Asymptotic distribution of MLE?**
$\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})$ where $I$ is Fisher information.

**15. What's Fisher information?**
$I(\theta) = -\mathbb{E}[\partial^2 \ell/\partial \theta^2]$. Curvature of expected log-likelihood; measures how sharply peaked it is around true value.

**16. Why is MLE asymptotically efficient?**
Variance achieves Cramér-Rao lower bound: $1/I(\theta)$. No unbiased estimator can do better asymptotically.

**17. Invariance of MLE — what does it mean?**
$\widehat{g(\theta)} = g(\hat{\theta})$. So MLE of standard deviation = $\sqrt{\hat{\sigma}^2_{\mathrm{MLE}}}$.

**18. When does asymptotic theory fail?**
Boundary parameters (e.g., $\theta = 0$ when domain is $[0, \infty)$), non-identifiable models, infinite Fisher information, non-iid data.

---

## D. MAP

**19. Define MAP.**
$\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta p(\theta|x) = \arg\max_\theta [\log p(x|\theta) + \log p(\theta)]$.

**20. MAP vs MLE — key relationship?**
MAP = MLE + log-prior penalty.

**21. MAP equals MLE when?**
Uniform (improper) prior — log-prior is constant, has no effect.

**22. MAP vs posterior mean — same?**
No. MAP is the *mode*; posterior mean is the *expectation*. Different unless posterior is symmetric.

---

## E. Priors as regularizers

**23. Gaussian prior on weights → what regularizer?**
$\ell_2$. $\log \mathcal{N}(0, \tau^2 I) \propto -\|w\|^2/(2\tau^2)$.

**24. Show ridge regression = MAP under Gaussian prior.**
Likelihood Gaussian, prior Gaussian. $\log p(w|x,y) = -\frac{1}{2\sigma^2}\|y-Xw\|^2 - \frac{1}{2\tau^2}\|w\|^2$. Maximizing → ridge with $\lambda = \sigma^2/\tau^2$.

**25. Laplace prior → what regularizer?**
$\ell_1$. $\log \mathrm{Laplace}(0, b) \propto -|w|/b$.

**26. Why does $\ell_1$ produce sparsity?**
$\ell_1$ ball has corners at axes; optimum is often *at* a corner → some weights exactly zero. Geometrically, lasso intersects the constraint set at a corner.

**27. Why does $\ell_2$ not produce sparsity?**
$\ell_2$ ball is round → optimum is generically in the interior of an axis hyperplane → all weights non-zero.

**28. What does early stopping correspond to?**
Approximately MAP with a Gaussian prior — the early stop limits how far weights move from the (zero) initialization. Connection is exact for linear models (Friedman, Hastie & Tibshirani).

---

## F. Conjugate priors

**29. What's a conjugate prior?**
Prior whose posterior stays in the same family. Enables closed-form Bayesian updates.

**30. Conjugate of Bernoulli/Binomial?**
Beta.

**31. Conjugate of multinomial/categorical?**
Dirichlet.

**32. Conjugate of Poisson?**
Gamma.

**33. Conjugate of Gaussian (mean, variance known)?**
Gaussian.

**34. Beta-Bernoulli: prior + 5 successes / 3 failures from Beta(2, 2). What's the posterior?**
Beta(2 + 5, 2 + 3) = Beta(7, 5).

**35. Beta-Bernoulli posterior mean?**
$(\alpha + s)/(\alpha + \beta + n)$.

**36. With $\alpha = \beta = 1$, what does the posterior mean become?**
$(s+1)/(n+2)$ — Laplace's rule of succession / add-one smoothing.

**37. What's the "pseudo-count" interpretation?**
Beta($\alpha, \beta$) = $\alpha$ pseudo-successes, $\beta$ pseudo-failures. The prior acts like imaginary data.

**38. Dirichlet prior as smoothing — why does NLP use add-$\alpha$ smoothing?**
$N$-gram counts $n_w$ with Dirichlet($\alpha$) prior. Posterior probability for word $w$: $(n_w + \alpha)/(\sum_v n_v + V\alpha)$. Prevents zero probabilities for unseen tokens.

---

## G. Connections to standard ML

**39. Cross-entropy minimization equals what?**
MLE in general (negative log-likelihood). Specifically, minimizing CE = minimizing forward KL from data to model (up to data-entropy constant).

**40. Forward KL vs reverse KL — which does MLE minimize?**
Forward: $\mathrm{KL}(p^* \| p_\theta)$. Mode-covering. (VI minimizes reverse KL.)

**41. Why is squared loss the right loss for regression?**
Under Gaussian noise assumption, MLE = squared loss. Other noise models give other losses (Huber for heavy-tailed, MAE for Laplace noise).

**42. RLHF reward model — what's the MLE?**
Bradley-Terry: $p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$. MLE is logistic regression on (preferred, rejected) pairs.

**43. SFT loss = MLE of what?**
Conditional language model: $p(y|x; \theta)$. Minimize $-\sum_{(x,y)} \log p_\theta(y|x)$ = MLE.

**44. DPO loss derivation starting point?**
Substitute the optimal RLHF policy ($\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x) \exp(r/\beta)$) into the Bradley-Terry MLE, eliminating the reward — yields a closed-form classification objective on preferences.

---

## H. Subtleties

**45. Is MLE always unbiased?**
No. MLE for Gaussian variance is biased; many other MLEs are biased in finite samples.

**46. Is MAP always unbiased?**
Almost never. MAP introduces deliberate bias to reduce variance.

**47. Why might you prefer MAP over MLE?**
Small data + strong prior → MAP regularizes against overfitting. Equivalent to standard regularization.

**48. Why might you prefer Bayesian inference over MAP?**
Need uncertainty estimates, want credible intervals, decision-theoretic problems with non-symmetric loss. MAP throws away the posterior shape.

**49. When does MAP become a poor summary of the posterior?**
Multimodal posterior, highly skewed posterior, transformation-dependent (MAP is not invariant under reparameterization, but MLE is — MAP point shifts under variable change).

**50. Why is MAP not invariant under reparameterization?**
Under a transformation $\theta \to \phi = g(\theta)$, the prior density transforms by a Jacobian. The mode of $p(\phi|x)$ is generally not $g(\hat{\theta}_{\mathrm{MAP}})$.

---

## Quick fire

**51.** *MLE Bernoulli?* Sample mean.
**52.** *MLE Gaussian variance divisor?* $n$ (biased).
**53.** *Unbiased Gaussian variance divisor?* $n-1$.
**54.** *OLS = MLE under what?* Gaussian noise.
**55.** *Ridge = MAP under what?* Gaussian prior.
**56.** *Lasso = MAP under what?* Laplace prior.
**57.** *Conjugate of Bernoulli?* Beta.
**58.** *Beta($\alpha, \beta$) mean?* $\alpha/(\alpha+\beta)$.
**59.** *Beta(1,1) is?* Uniform on $[0,1]$.
**60.** *MLE achieves what bound?* Cramér-Rao.

---

## Self-grading

If you can't answer 1-15, you don't know MLE. If you can't answer 16-35, you'll struggle on every Bayesian/regularization question. If you can't answer 36-50, frontier-lab questions on RLHF/DPO/loss design will go past you.

Aim for 40+/60 cold.
