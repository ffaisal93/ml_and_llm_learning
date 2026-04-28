# Statistical Inference — Interview Grill

> 50 questions on estimators, MLE, CIs, bootstrap, hypothesis testing, Bayesian inference. Drill until you can answer 35+ cold.

---

## A. Estimators

**1. What's an estimator?**
A function of the data that approximates an unknown parameter: $\hat{\theta} = T(X_1, \ldots, X_n)$.

**2. Define unbiased.**
$\mathbb{E}[\hat{\theta}] = \theta$ — on average across repeated samples, the estimator hits the true value.

**3. Define consistent.**
$\hat{\theta}_n \to_p \theta$ as $n \to \infty$.

**4. Unbiased vs consistent — give an example of one but not the other.**
Sample mean of one observation: unbiased, not consistent. Estimator $\hat{\theta} = X_1 + 1/n$: consistent but biased for any finite $n$.

**5. State the bias-variance decomposition for MSE.**
$\mathrm{MSE}(\hat{\theta}) = \mathrm{Bias}(\hat{\theta})^2 + \mathrm{Var}(\hat{\theta})$. Implication: biased estimators can have lower MSE than unbiased ones.

**6. What's the Cramér-Rao lower bound?**
$\mathrm{Var}(\hat{\theta}) \geq 1/I(\theta)$ where $I$ is Fisher information. Lower bound on variance for any unbiased estimator.

**7. Why $n-1$ in sample variance (Bessel's correction)?**
$\frac{1}{n}\sum(x_i - \bar{x})^2$ underestimates $\sigma^2$ because $\bar{x}$ is closer to the data than $\mu$. Dividing by $n-1$ corrects the bias.

---

## B. MLE

**8. Define MLE.**
$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \prod_i p(x_i|\theta) = \arg\max_\theta \sum_i \log p(x_i|\theta)$.

**9. Derive MLE for Bernoulli.**
$\ell(\theta) = \sum [x_i \log \theta + (1-x_i)\log(1-\theta)]$. Set $\partial \ell / \partial \theta = 0$: $\hat{\theta} = \bar{x}$.

**10. Derive MLE for Gaussian (mean and variance).**
$\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$. The MLE for variance is biased — Bessel's correction unbiases it.

**11. Why is MLE biased for variance but consistent?**
Bias is $O(1/n)$ — vanishes as $n \to \infty$. So MLE is consistent but not unbiased in finite samples.

**12. Asymptotic properties of MLE?**
Consistent, asymptotically normal: $\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I^{-1})$, asymptotically efficient (achieves CRLB).

**13. Invariance of MLE — what is it?**
If $\hat{\theta}_{\mathrm{MLE}}$ estimates $\theta$, then $g(\hat{\theta}_{\mathrm{MLE}})$ estimates $g(\theta)$. E.g., MLE of $\sigma$ is $\sqrt{\hat{\sigma}^2_{\mathrm{MLE}}}$.

**14. When does MLE fail?**
Small samples (high variance, biased), unbounded likelihood (e.g., Gaussian mixture with covariance shrinking to a point), non-identifiable models.

---

## C. Confidence intervals

**15. Define a 95% confidence interval.**
A random interval $[L, U]$ such that under repeated sampling, 95% of intervals contain $\theta$. Frequency interpretation, not "$\theta$ is in [...] with 95% probability."

**16. Wald CI formula?**
$\hat{\theta} \pm 1.96 \cdot \mathrm{SE}(\hat{\theta})$ for 95%. Relies on asymptotic normality.

**17. CI vs credible interval?**
CI is frequentist — interval random, $\theta$ fixed. Credible interval is Bayesian — interval fixed, $\theta$ has posterior probability mass. CredI supports the natural "$\theta$ in [...] with 95% probability" interpretation.

**18. How do you compute a bootstrap CI?**
Resample data with replacement $B$ times, compute $\hat{\theta}^{(b)}$ each time. CI = $[Q_{0.025}, Q_{0.975}]$ of $\{\hat{\theta}^{(b)}\}$ (percentile method).

**19. When can a CI go negative for a positive quantity?**
When CI is constructed without constraints (e.g., Wald CI for a probability close to 0 or 1). Use logit transform or bootstrap.

---

## D. Hypothesis testing

**20. State the components of a hypothesis test.**
Null $H_0$, alternative $H_1$, test statistic $T$, null distribution, rejection region, significance $\alpha$.

**21. What's a p-value?**
Probability under $H_0$ of observing a test statistic at least as extreme as the one observed. NOT $\mathbb{P}(H_0 | \mathrm{data})$.

**22. Why is "p < 0.05 means the result is true" wrong?**
$p$-value isn't $\mathbb{P}(H_0 | \mathrm{data})$. With multiple tests, $p < 0.05$ alone is meaningless. Even with one test, low $p$ is "data is unlikely under $H_0$," not "$H_0$ is unlikely."

**23. Type I vs Type II error?**
Type I: reject true $H_0$ (false positive, controlled by $\alpha$). Type II: fail to reject false $H_0$ (false negative, controlled by power $1-\beta$).

**24. What's statistical power?**
$1 - \beta = \mathbb{P}(\mathrm{reject}\, H_0 \mid H_1\, \mathrm{true})$. Depends on effect size, $n$, $\alpha$, variance.

**25. When do you use a t-test vs z-test?**
$z$-test: variance known (rare). $t$-test: variance estimated from sample (almost always).

**26. When do you use a chi-squared test?**
Goodness-of-fit, contingency tables (test of independence). Categorical data. Statistic: $\chi^2 = \sum (O - E)^2 / E$.

**27. What's a one-sided vs two-sided test?**
One-sided: $H_1: \theta > \theta_0$ (or $<$). Two-sided: $H_1: \theta \neq \theta_0$. One-sided has more power but you must commit to direction *a priori*.

**28. Paired vs unpaired t-test?**
Paired: same subjects measured twice (before/after). Unpaired: independent groups. Paired has more power because it removes between-subject variation.

---

## E. Multiple testing

**29. The multiple testing problem?**
With $m$ independent tests at $\alpha = 0.05$, FWER $\approx m\alpha$ for small $\alpha$. Run 20 tests, expect ~1 false positive even with no real effect.

**30. Bonferroni correction?**
Test each at $\alpha/m$ instead of $\alpha$. Controls FWER. Conservative; loses power.

**31. What's Benjamini-Hochberg?**
Controls false discovery rate (FDR = expected proportion of false positives among rejections). Order p-values; reject the largest $i$ for which $p_{(i)} \leq i\alpha/m$. Less conservative than Bonferroni.

**32. FWER vs FDR — when each?**
FWER: when any false positive is bad (e.g., medical diagnosis). FDR: when discovery is exploratory and some false positives are tolerable (e.g., gene expression).

**33. Where does multiple testing show up in ML?**
Hyperparameter sweeps, A/B test farms, feature selection (test each feature), subgroup analysis.

---

## F. Bootstrap

**34. What's the bootstrap?**
Resample data with replacement $B$ times to approximate the sampling distribution of an estimator. Non-parametric, simple, broadly applicable.

**35. When does bootstrap fail?**
Extreme order statistics (min/max), heavy-tailed distributions without enough data, time series (without block bootstrap), very small $n$.

**36. Bootstrap a confusion-matrix metric — how?**
Resample (predictions, labels) pairs with replacement. Compute metric on resample. Repeat 1000+ times. Quantiles of the resulting distribution give CI.

**37. What's a paired bootstrap for model comparison?**
For each bootstrap sample, compute metric for both models on the *same* sample. Look at distribution of differences. Reject "no difference" if 0 not in CI.

**38. Bagging is bootstrap of what?**
Bagging = "Bootstrap Aggregating." Train each tree on a bootstrap resample of data. Random Forests add feature subsampling.

---

## G. Bayesian inference

**39. State Bayes' theorem.**
$p(\theta | x) = p(x|\theta)p(\theta)/p(x)$.

**40. What's a conjugate prior? Example?**
A prior whose posterior stays in the same family. Beta-Bernoulli: prior $\mathrm{Beta}(\alpha, \beta)$ + $s$ successes / $n-s$ failures → posterior $\mathrm{Beta}(\alpha + s, \beta + n - s)$.

**41. Beta-Bernoulli posterior mean?**
$(\alpha + s)/(\alpha + \beta + n)$. Smoothing: prior acts like $\alpha + \beta$ "pseudo-observations."

**42. What's MAP estimation?**
$\hat{\theta}_{\mathrm{MAP}} = \arg\max p(\theta|x) = \arg\max [\log p(x|\theta) + \log p(\theta)]$. MLE + log-prior penalty.

**43. Connection between MAP and regularization?**
Gaussian prior on weights → $\ell_2$ penalty (ridge). Laplace prior → $\ell_1$ penalty (lasso). Regularization is MAP with a particular prior.

**44. What's the marginal likelihood / evidence and why does it matter?**
$p(x) = \int p(x|\theta)p(\theta)d\theta$. Used for Bayesian model comparison (Bayes factors). Hard to compute in general.

**45. MCMC vs variational inference?**
MCMC: sample from posterior; asymptotically exact, slow. VI: approximate posterior with a simpler distribution by minimizing KL; biased, fast. ML practitioners usually use VI when scale matters.

---

## H. Practical ML stats

**46. You report a model AUC of 0.85. How do you give it a CI?**
Bootstrap the test set 1000+ times; compute AUC on each; take 2.5%/97.5% quantiles.

**47. Two models: AUC 0.85 vs 0.84. Is the difference significant?**
Paired bootstrap of AUC differences. CI for difference; reject "no difference" if 0 not in CI. Or DeLong's test for AUC specifically.

**48. You run 50 A/B tests and 3 are "significant" at $p<0.05$. Are any real?**
Probably 2.5 false positives expected by chance. Apply Bonferroni ($\alpha/50 = 0.001$) or BH correction.

**49. Model accuracy = 87% on test set of 1000. CI?**
Wald: $0.87 \pm 1.96 \sqrt{0.87 \cdot 0.13 / 1000} \approx 0.87 \pm 0.021$. Or Wilson interval (better for proportions). Or bootstrap.

**50. Train accuracy 95%, test 87%. Statistically significant gap?**
Compute CIs on each. Subtract. If CIs overlap heavily, gap might be noise. Better: paired bootstrap of differences, or test on multiple test splits.

---

## Quick fire

**51.** *MLE for Bernoulli?* Sample mean.
**52.** *Bessel's correction divisor?* $n-1$.
**53.** *95% z-value?* 1.96.
**54.** *CRLB lower-bounds what?* Variance of unbiased estimator.
**55.** *Conjugate of Bernoulli?* Beta.
**56.** *Conjugate of Poisson?* Gamma.
**57.** *Conjugate of multinomial?* Dirichlet.
**58.** *Bonferroni: divide $\alpha$ by?* Number of tests $m$.
**59.** *MAP equals MLE when?* Uniform prior.
**60.** *CLT statement?* Sample mean is asymptotically Gaussian regardless of underlying distribution (with finite variance).

---

## Self-grading

If you can't answer 1-15, you don't know basic statistics. If you can't answer 16-35, you'll get tripped up on every interview that probes ML evaluation rigor. If you can't answer 36-50, frontier-lab interviews on experimental rigor will go past you.

Aim for 40+/60 cold.
