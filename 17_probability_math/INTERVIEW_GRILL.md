# Probability for ML — Interview Grill

> 50 questions on probability fundamentals, distributions, Bayes, limit theorems. Drill until you can answer 35+ cold.

---

## A. Probability basics

**1. State the three probability axioms.**
$\mathbb{P}(\Omega) = 1$. $\mathbb{P}(A) \in [0,1]$. Countable additivity for disjoint events.

**2. What's the inclusion-exclusion principle for two sets?**
$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.

**3. Define conditional probability.**
$\mathbb{P}(A|B) = \mathbb{P}(A \cap B)/\mathbb{P}(B)$ for $\mathbb{P}(B) > 0$.

**4. State Bayes' theorem.**
$\mathbb{P}(A|B) = \mathbb{P}(B|A)\mathbb{P}(A)/\mathbb{P}(B)$.

**5. Define independence vs uncorrelated.**
Independent: $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$. Uncorrelated: $\mathrm{Cov}(X,Y) = 0$. Independence ⟹ uncorrelated, but not vice versa (except for jointly Gaussian).

**6. What's the law of total probability?**
For partition $\{B_i\}$: $\mathbb{P}(A) = \sum_i \mathbb{P}(A|B_i)\mathbb{P}(B_i)$.

**7. Conditional independence — define.**
$X \perp Y | Z$ iff $p(x, y|z) = p(x|z)p(y|z)$. NOT the same as unconditional independence.

---

## B. Random variables

**8. Define expectation.**
$\mathbb{E}[X] = \sum x p(x)$ or $\int x f(x) dx$.

**9. Linearity of expectation — when does it hold?**
Always — even for dependent variables. $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$.

**10. Variance formula — two equivalent forms?**
$\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.

**11. Variance of a sum?**
$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y)$.

**12. Covariance formula?**
$\mathrm{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.

**13. Variance of $\bar{X}$ for iid samples?**
$\mathrm{Var}(\bar{X}) = \sigma^2/n$.

**14. State the law of total expectation.**
$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$ (tower property).

**15. State the law of total variance.**
$\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X|Y)] + \mathrm{Var}(\mathbb{E}[X|Y])$.

---

## C. Common distributions

**16. Bernoulli mean and variance?**
Mean $p$, variance $p(1-p)$.

**17. Binomial mean and variance?**
$np$, $np(1-p)$. Sum of $n$ iid Bernoullis.

**18. Poisson mean and variance?**
Both $\lambda$. Variance equals mean — Poisson signature.

**19. When does Binomial → Poisson?**
$n \to \infty$, $p \to 0$, $np = \lambda$ fixed. Used for rare events.

**20. Geometric mean and variance?**
Mean $1/p$, variance $(1-p)/p^2$. Number of trials until first success.

**21. Exponential mean and variance?**
$1/\lambda$, $1/\lambda^2$.

**22. Gaussian — fully specified by what?**
Mean $\mu$ and variance $\sigma^2$. (Multivariate: mean vector and covariance matrix.)

**23. What's the memoryless property?**
$\mathbb{P}(X > s + t | X > s) = \mathbb{P}(X > t)$. Only geometric (discrete) and exponential (continuous) have it.

**24. Sum of independent Gaussians?**
Gaussian. Means add, variances add.

**25. Sum of independent Poissons?**
Poisson. Rates add.

**26. Beta distribution — what does it model?**
A probability (range $[0,1]$). Conjugate prior for Bernoulli/Binomial.

**27. Gamma — what does it model?**
Positive continuous quantity. Sum of exponentials. Conjugate for Poisson rate.

---

## D. Multivariate Gaussian

**28. Density of multivariate Gaussian?**
$\mathcal{N}(x|\mu, \Sigma) = (2\pi)^{-d/2} |\Sigma|^{-1/2} \exp(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu))$.

**29. Affine transform of Gaussian?**
$AX + b \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)$.

**30. Marginal of multivariate Gaussian?**
Gaussian. Just take the corresponding subvector of $\mu$ and submatrix of $\Sigma$.

**31. Conditional of multivariate Gaussian?**
Gaussian. $X_1|X_2 = x_2 \sim \mathcal{N}(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2), \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})$.

**32. Uncorrelated jointly Gaussian = independent. True?**
Yes. This is special to Gaussians.

**33. If $X, Y$ both Gaussian individually, is $(X, Y)$ jointly Gaussian?**
Not necessarily. Marginal Gaussianity doesn't imply joint. (Counterexample: $X \sim \mathcal{N}(0,1)$, $Y = SX$ where $S = \pm 1$ randomly.)

---

## E. Limit theorems

**34. State the weak law of large numbers.**
For iid $X_i$ with finite mean $\mu$: $\bar{X}_n \to_p \mu$.

**35. State the central limit theorem.**
For iid $X_i$ with mean $\mu$, finite variance $\sigma^2$: $\sqrt{n}(\bar{X}_n - \mu) \to \mathcal{N}(0, \sigma^2)$.

**36. When does CLT fail?**
Infinite variance (heavy tails like Cauchy). Strongly dependent data without mixing conditions.

**37. CLT convergence rate?**
Berry-Esseen: $O(1/\sqrt{n})$, with constant depending on third moment. Skewed distributions need larger $n$.

**38. Why is Gaussian everywhere in stats?**
CLT — sums of many small effects approach Gaussian. So sample means, regression residuals, etc. tend to be approximately Gaussian.

---

## F. Bayes applications

**39. Disease prevalence 1%, test sensitivity 99%, specificity 99%. P(disease | positive)?**
$\mathbb{P}(D|+) = 0.99 \cdot 0.01 / (0.99 \cdot 0.01 + 0.01 \cdot 0.99) = 0.5$. Even 99% accurate tests give only 50% probability for 1% prevalence.

**40. What's the base rate fallacy?**
Ignoring prior probability when interpreting test results. The classic Bayesian error.

**41. What's naive Bayes' assumption?**
Features conditionally independent given class: $\mathbb{P}(x|c) = \prod_j \mathbb{P}(x_j|c)$.

**42. Why does naive Bayes work despite the assumption being wrong?**
Need only correct relative ordering of class probabilities; absolute values can be miscalibrated.

**43. Sequential Bayes update — what happens to posterior after multiple iid observations?**
Posterior after $n$ observations = prior × likelihood$^n$ = repeatedly applying Bayes one observation at a time.

---

## G. Calculations to do fast

**44. $X \sim \mathrm{Uniform}(0,1)$. $\mathbb{E}[X^2]$?**
$\int_0^1 x^2 dx = 1/3$.

**45. $X \sim \mathrm{Exp}(\lambda)$. $\mathbb{E}[X^2]$?**
$\mathrm{Var}(X) + \mathbb{E}[X]^2 = 1/\lambda^2 + 1/\lambda^2 = 2/\lambda^2$.

**46. $\mathbb{E}[\max(X, 0)]$ for $X \sim \mathcal{N}(0, \sigma^2)$?**
$\sigma/\sqrt{2\pi}$. (Half-normal mean.)

**47. Variance of sum of $n$ iid Bernoulli($p$)?**
$np(1-p)$.

**48. Roll a fair die until you get a 6. Expected number of rolls?**
$1/p = 6$. (Geometric distribution.)

**49. Two iid uniform $(0,1)$. $\mathbb{P}(\max > 0.5)$?**
$1 - (0.5)^2 = 0.75$. Or $\mathbb{P}(\text{both} \leq 0.5) = 0.25$.

**50. $X, Y$ iid $\mathcal{N}(0, 1)$. Distribution of $X^2 + Y^2$?**
$\chi^2_2$ = Exp(1/2). $\mathbb{E}[X^2 + Y^2] = 2$.

---

## Quick fire

**51.** *Bernoulli variance?* $p(1-p)$.
**52.** *Poisson variance equals?* Mean.
**53.** *Memoryless distributions?* Geometric, Exponential.
**54.** *Conjugate of Bernoulli?* Beta.
**55.** *CLT requires what about variance?* Finite.
**56.** *Linearity of expectation requires?* Nothing — always holds.
**57.** *Independence implies?* Uncorrelated.
**58.** *Cov = 0 implies independent?* Only for jointly Gaussian.
**59.** *95% CI z-value?* 1.96.
**60.** *Variance of sample mean of iid?* $\sigma^2/n$.

---

## Self-grading

If you can't answer 1-15, you don't know basic probability. If you can't answer 16-35, you'll get tripped up on Bayes/distribution questions. If you can't answer 36-50, frontier-lab interview probability problems will go past you.

Aim for 40+/60 cold.
