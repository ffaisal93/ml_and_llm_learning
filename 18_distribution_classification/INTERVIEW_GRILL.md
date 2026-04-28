# Distribution Classification — Interview Grill

> 40 questions on choosing distributions, exponential family, GLMs, canonical links. Drill until you can answer 28+ cold.

---

## A. Picking distributions

**1. CTR data — what distribution?**
Per-impression: Bernoulli($p$). Aggregated: Binomial($n, p$). Conjugate prior: Beta.

**2. Number of website visits per hour?**
Poisson($\lambda$) if rare, independent. If overdispersed: Negative Binomial.

**3. Time between two events?**
Exponential. (Also assumes memoryless; if hazard varies, use Weibull.)

**4. User revenue (heavy right tail)?**
Lognormal usually. Or Gamma. Be careful — the sample mean can be misleading.

**5. Time until $k$-th event?**
Gamma (sum of $k$ iid Exponentials).

**6. Probability of conversion (each user has its own rate)?**
Beta on the rates; Bernoulli on outcomes given rates.

**7. Class label out of $K$ options?**
Categorical($p_1, \ldots, p_K$).

**8. Word counts in a document?**
Multinomial (or per-word categorical). Topic mixture: Dirichlet prior.

**9. Number of trials until first success?**
Geometric.

**10. Sum of small random effects?**
Gaussian (CLT).

**11. Stock returns?**
Heavy-tailed: Student-t or Cauchy-ish. Empirically NOT Gaussian.

**12. Income distribution?**
Pareto / Lognormal. Heavy right tail.

---

## B. Distribution properties

**13. When does Binomial ≈ Poisson?**
$n$ large, $p$ small, $np = \lambda$ fixed.

**14. When does Binomial ≈ Gaussian?**
$n$ large, $p$ not near 0 or 1. CLT applies.

**15. Poisson signature?**
Variance equals mean.

**16. What's overdispersion?**
Observed variance much larger than mean (when Poisson would predict equality). Suggests Negative Binomial or hierarchical Poisson.

**17. What's underdispersion?**
Variance less than mean. Rare; can use truncated/conditional models.

**18. Memoryless distributions?**
Exponential (continuous), Geometric (discrete). Only ones.

**19. Conjugate prior table — Bernoulli?**
Beta.

**20. Conjugate prior — Poisson?**
Gamma.

**21. Conjugate prior — Multinomial?**
Dirichlet.

**22. Conjugate prior — Gaussian (mean only)?**
Gaussian.

---

## C. Exponential family

**23. Exponential family form?**
$p(x|\theta) = h(x) \exp(\eta(\theta)^\top T(x) - A(\theta))$.

**24. What's the natural parameter for Bernoulli?**
$\eta = \log\frac{p}{1-p}$ (logit).

**25. What's the natural parameter for Poisson?**
$\eta = \log \lambda$.

**26. What's the natural parameter for Gaussian (variance known)?**
$\eta = \mu/\sigma^2$.

**27. What's a sufficient statistic?**
$T(x)$ such that $p(\theta | x) = p(\theta | T(x))$ — captures all info about $\theta$ in the data.

**28. Why does exponential family give clean MLE?**
$\nabla A(\theta) = \mathbb{E}[T(X)]$. MLE matches expected sufficient statistics to empirical: $\bar{T}_{\mathrm{data}} = \mathbb{E}_\theta[T(X)]$.

**29. Why does exponential family always have a conjugate prior?**
Multiplication of likelihood by a prior of the same exponential form gives another exp-family distribution; closed-form posterior.

---

## D. GLMs

**30. Three components of a GLM?**
Random component (exp-family distribution), systematic component (linear predictor $\eta = w^\top x$), link function $g(\mu) = \eta$.

**31. What's the canonical link?**
Link function such that $\eta$ equals the natural parameter of the distribution.

**32. Canonical link for Gaussian?**
Identity. Linear regression.

**33. Canonical link for Bernoulli?**
Logit. Logistic regression.

**34. Canonical link for Multinomial?**
Multi-logit (softmax inverse). Multi-class logistic regression.

**35. Canonical link for Poisson?**
Log. Poisson regression.

**36. Why is the canonical link special?**
Score function is $\nabla \ell = \sum (y_i - \mu_i) x_i$ — clean, like OLS residuals. Asymptotic theory simplest.

**37. Logistic regression as GLM — random/systematic/link?**
Random: Bernoulli($\mu_i$). Systematic: $\eta_i = w^\top x_i$. Link: $g(\mu) = \log\frac{\mu}{1-\mu}$ (logit). Inverse link: sigmoid.

**38. Connection between cross-entropy loss and GLM?**
CE for binary classification = NLL of Bernoulli GLM. CE for multi-class = NLL of multinomial GLM with softmax canonical link.

**39. Can you do GLM with a non-canonical link?**
Yes — e.g., probit link for Bernoulli (uses Gaussian CDF instead of logit). Loses some of the clean asymptotic properties but sometimes preferred.

---

## E. Heavy tails

**40. What's a heavy-tailed distribution?**
Tail decays slower than exponential. Examples: Pareto, Cauchy, lognormal, Student-t.

**41. Why does CLT fail for Cauchy?**
Infinite variance. Sample mean of iid Cauchys is *also* Cauchy — no concentration.

**42. Pareto with $\alpha < 2$ — what's the issue?**
Infinite variance. Sample variance fluctuates wildly, doesn't stabilize.

**43. Pareto with $\alpha < 1$ — what's the issue?**
Infinite mean. Sample mean has no limit; new extremes keep dominating.

**44. How do you handle heavy-tailed data?**
Log-transform, use median/quantiles instead of mean, robust statistics, distributional models that capture tails (Student-t, Pareto).

---

## F. Practical decisions

**45. You're modeling defects per unit and see Var(defects) >> Mean(defects). What's the issue and fix?**
Overdispersion. Poisson is too restrictive. Use Negative Binomial regression.

**46. You're modeling time-to-failure of components, but failure rate increases with age (not memoryless). What distribution?**
Weibull. (Exponential = memoryless = constant hazard rate.)

**47. You want to model the probability of conversion for each user as a random variable across users.**
Beta-distributed conversion rates; Bernoulli outcomes given the rate. This is hierarchical Bayes / random-effects.

**48. Your regression target is non-negative skewed. Linear regression gives negative predictions.**
Switch to GLM with log link (Gamma or Poisson regression). Or transform target with $\log(y+1)$.

**49. Logistic regression isn't fitting well — what alternatives?**
Probit (Gaussian-CDF link), complementary log-log link, generalized additive model, neural network.

**50. You have multinomial data but suspect overdispersion across documents. What's the model?**
Dirichlet-multinomial: marginalize over document-level Dirichlet to get extra variance.

---

## Quick fire

**51.** *Variance > mean for counts → ?* Negative Binomial.
**52.** *CLT requires?* Finite variance.
**53.** *Bernoulli canonical link?* Logit.
**54.** *Poisson canonical link?* Log.
**55.** *Gaussian canonical link?* Identity.
**56.** *Memoryless continuous?* Exponential.
**57.** *Memoryless discrete?* Geometric.
**58.** *Heavy-tailed examples?* Pareto, lognormal, Cauchy, Student-t.
**59.** *Bernoulli sufficient statistic?* Sum (count of successes).
**60.** *Cross-entropy = MLE of?* Multinomial GLM (canonical link is multi-logit; softmax is its inverse).

---

## Self-grading

If you can't answer 1-15, you can't choose models intelligently. If you can't answer 16-35, you'll get tripped up on GLM/exp-family questions. If you can't answer 36-50, frontier-lab interviews on probabilistic modeling will go past you.

Aim for 40+/60 cold.
