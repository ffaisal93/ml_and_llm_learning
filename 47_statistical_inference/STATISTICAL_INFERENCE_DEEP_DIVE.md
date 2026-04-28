# Statistical Inference — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Statistical inference is what separates "I trained a model and it has 87% accuracy" from "I have evidence that my model's true accuracy is 87% ± 1.2% and that's a statistically significant 0.4-point improvement over the baseline." Senior interviews probe this hard because production ML decisions hinge on it.

---

## 1. Estimators — what they are and what makes one "good"

An **estimator** is a function $\hat{\theta} = T(X_1, \ldots, X_n)$ of the data that tries to recover an unknown parameter $\theta$.

### Properties

**Unbiased**: $\mathbb{E}[\hat{\theta}] = \theta$. Sample mean is unbiased for population mean. Sample variance with $n-1$ denominator is unbiased; with $n$ it isn't (Bessel's correction).

**Consistent**: $\hat{\theta}_n \to_p \theta$ as $n \to \infty$. Most useful estimators are consistent. (Note: unbiased ≠ consistent in general; both are different properties.)

**Efficient**: minimum variance among unbiased estimators. Cramér-Rao lower bound:

$$
\mathrm{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

where $I(\theta) = -\mathbb{E}[\partial^2 \log p(X|\theta) / \partial \theta^2]$ is the Fisher information. MLE is asymptotically efficient — achieves CRLB.

**Bias-variance decomposition** for MSE:

$$
\mathrm{MSE}(\hat{\theta}) = \mathrm{Bias}(\hat{\theta})^2 + \mathrm{Var}(\hat{\theta})
$$

Important: a biased estimator with low variance can have lower MSE than an unbiased one with high variance. This is the whole point of regularization.

---

## 2. Maximum likelihood estimation

The likelihood: $L(\theta) = \prod_i p(x_i | \theta)$. The log-likelihood: $\ell(\theta) = \sum_i \log p(x_i | \theta)$.

MLE: $\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

### Properties of MLE
- **Consistent**: $\hat{\theta}_{\mathrm{MLE}} \to_p \theta_0$
- **Asymptotically normal**: $\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})$
- **Asymptotically efficient**: variance hits CRLB
- **Invariant to reparameterization**: $\widehat{g(\theta)} = g(\hat{\theta})$

### Worked examples

**Bernoulli (coin flip):** $p(x|\theta) = \theta^x (1-\theta)^{1-x}$.
$\ell(\theta) = \sum x_i \log \theta + (n - \sum x_i) \log(1-\theta)$.
$\hat{\theta}_{\mathrm{MLE}} = \bar{x}$ (sample mean).

**Gaussian (known variance):** $p(x|\mu) = \mathcal{N}(\mu, \sigma^2)$.
$\hat{\mu}_{\mathrm{MLE}} = \bar{x}$.

**Gaussian (both unknown):** $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$ — biased! Unbiased estimator uses $1/(n-1)$.

---

## 3. Confidence intervals

A $1-\alpha$ CI is a random interval $[L, U]$ with $\mathbb{P}(L \leq \theta \leq U) = 1-\alpha$ — *over repeated sampling*.

**Common misinterpretation:** "There's a 95% probability $\theta$ is in [1.2, 3.4]." Wrong (under frequentist interpretation). $\theta$ is fixed; the *interval* is random. The correct statement: "If we repeated this procedure many times, 95% of intervals would contain $\theta$."

### Wald CI (asymptotic)

For an asymptotically normal estimator:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \mathrm{SE}(\hat{\theta})
$$

with $z_{0.025} = 1.96$ for 95%. Standard error from Fisher information or sample variance.

### Bootstrap CI

When you can't compute SE analytically: resample data with replacement $B$ times, compute $\hat{\theta}^{(b)}$ for each, then take quantiles (percentile method) or use bootstrap-t.

```
for b in 1 .. B:
  sample X_b with replacement from X (size n)
  compute theta_b = T(X_b)
CI = [quantile(thetas, alpha/2), quantile(thetas, 1-alpha/2)]
```

Bootstrap is non-parametric, simple, and extremely useful in ML for things like AUC confidence intervals.

### Bayesian credible interval

The interval that contains 95% of the posterior probability mass. A *different* concept than Wald CI — and the credible interval supports the natural-language "$\theta$ is in [...] with 95% probability" interpretation, conditional on prior.

---

## 4. Hypothesis testing

Testing a claim $H_0$ vs alternative $H_1$.

### Components
- **Test statistic** $T(X)$: function of data.
- **Null distribution**: distribution of $T$ under $H_0$.
- **Rejection region**: values of $T$ where we reject $H_0$.
- **Significance level $\alpha$**: $\mathbb{P}(\text{reject} | H_0) \leq \alpha$ (Type I error).
- **Power** $1 - \beta$: $\mathbb{P}(\text{reject} | H_1)$.

### p-value

$p$-value = $\mathbb{P}(T \geq t_{\mathrm{obs}} | H_0)$ — probability of seeing data this extreme *if $H_0$ is true*.

**Common interpretation errors:**
- $p$-value is NOT $\mathbb{P}(H_0 | \text{data})$.
- A small $p$-value doesn't mean a large effect — just that the effect is unlikely under $H_0$.
- $p > 0.05$ doesn't prove $H_0$ — just lack of evidence against it.

### Standard tests

**z-test**: Gaussian, known variance. $z = (\bar{x} - \mu_0) / (\sigma/\sqrt{n})$.

**t-test**: Gaussian, unknown variance. Use sample SD; statistic follows $t_{n-1}$.

**Chi-squared**: categorical data goodness-of-fit, contingency tables. $\chi^2 = \sum (O - E)^2 / E$.

**Mann-Whitney U / Wilcoxon**: non-parametric two-sample.

**A/B test (proportions)**: binomial / two-proportion z-test.

### Type I vs Type II
- Type I (false positive): reject $H_0$ when true. Controlled by $\alpha$.
- Type II (false negative): fail to reject when $H_1$ true. $\beta$, depends on effect size, $n$, $\alpha$.

**Power analysis** picks $n$ to achieve target $1-\beta$ (typically 80%) for a minimum detectable effect.

---

## 5. Multiple testing

When you run $m$ tests at $\alpha = 0.05$, the family-wise probability of *any* false rejection grows: under independence, $1 - (1-\alpha)^m \approx m\alpha$ for small $\alpha$. With $m=20$ tests at $\alpha=0.05$, you expect 1 false positive.

### Corrections
- **Bonferroni**: use $\alpha/m$ per test. Conservative; controls family-wise error rate (FWER).
- **Holm-Bonferroni**: step-down version — less conservative.
- **Benjamini-Hochberg**: controls false discovery rate (FDR = expected proportion of false positives among rejections). Less conservative; standard in genomics, A/B testing at scale.

### When this matters in ML
- Hyperparameter search: 100 hyperparam combos → some "win" by luck.
- Many A/B tests on the same data: false positives.
- Feature selection: testing each feature for significance inflates Type I.
- Subgroup analysis ("but the model works better for users in California!") — almost always overstated without correction.

---

## 6. The bootstrap — workhorse for ML

The bootstrap (Efron 1979) lets you estimate sampling distributions when you can't derive them analytically.

**Recipe** (non-parametric bootstrap):
1. Resample $X^{(b)}$ from your data with replacement, size $n$.
2. Compute $\hat{\theta}^{(b)}$.
3. Repeat $B$ times (typically 1000–10000).
4. The empirical distribution of $\{\hat{\theta}^{(b)}\}$ approximates the sampling distribution.

**What you can do:**
- SE estimate: SD of the bootstrap distribution.
- CI: quantiles (percentile method) or bias-corrected accelerated (BCa).
- Hypothesis test: reject if observed value falls in tail.

**Bootstrap in ML practice:**
- AUC CI: bootstrap test set predictions.
- Model comparison: paired bootstrap of metric differences.
- Random forest internals: bagging *is* bootstrapping.

**Limitations:**
- Doesn't work for extreme order statistics (e.g., min/max).
- Doesn't work well for time series without block bootstrap.
- Computationally expensive for large $n$.

---

## 7. Bayesian inference

Frequentist: $\theta$ is fixed, data is random. Bayesian: $\theta$ has a probability distribution.

$$
p(\theta | x) = \frac{p(x | \theta) p(\theta)}{p(x)} \propto p(x | \theta) p(\theta)
$$

- $p(\theta)$: prior — your belief before seeing data.
- $p(x | \theta)$: likelihood — same as in MLE.
- $p(\theta | x)$: posterior — updated belief.
- $p(x) = \int p(x|\theta) p(\theta) d\theta$: marginal likelihood / evidence.

### Conjugate priors

Posterior in the same family as prior. Examples:
- Beta prior + Bernoulli likelihood → Beta posterior.
- Gamma prior + Poisson likelihood → Gamma posterior.
- Dirichlet prior + multinomial likelihood → Dirichlet posterior.
- Gaussian prior + Gaussian likelihood (known variance) → Gaussian posterior.

**Beta-Bernoulli example:** prior $\theta \sim \mathrm{Beta}(\alpha, \beta)$. After observing $s$ successes in $n$ trials: posterior $\theta | x \sim \mathrm{Beta}(\alpha + s, \beta + n - s)$. Posterior mean: $(\alpha + s)/(\alpha + \beta + n)$.

### MAP

Maximum a posteriori: $\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta p(\theta | x) = \arg\max_\theta [\log p(x|\theta) + \log p(\theta)]$.

This is exactly MLE + log-prior penalty. The penalty *is* the regularizer.

- Gaussian prior on weights → $\ell_2$ regularization (ridge).
- Laplace prior → $\ell_1$ (lasso).

### Posterior summaries
- Posterior mean: $\mathbb{E}[\theta | x]$
- Posterior median, mode (MAP)
- Credible interval: $[L, U]$ with $\mathbb{P}(\theta \in [L,U] | x) = 0.95$

### Bayesian inference in practice
- Conjugate cases: closed-form (rare beyond simple models).
- MCMC (Metropolis-Hastings, Gibbs, HMC): sample from posterior.
- Variational inference: approximate posterior with simpler distribution; minimize KL.
- Laplace approximation: Gaussian centered at MAP.

---

## 8. Common ML stats gotchas

| Mistake | Why it's wrong | Fix |
|---|---|---|
| "p > 0.05 → no effect" | Absence of evidence ≠ evidence of absence | Report effect size + CI |
| "p = 0.001 → big effect" | Small p just means precise estimate, not large | Report effect size separately |
| "Train/test gap shows generalization" | Single split is noisy | Cross-validation or bootstrap |
| "AUC = 0.85 vs 0.84 → better model" | Without CI, can be noise | Bootstrap CIs, paired tests |
| "Multiple A/B tests at $\alpha = 0.05$" | FWER blows up | Bonferroni / BH correction |
| "Use confidence interval as 'probability $\theta$ in interval'" | That's a credible interval | Be precise about interpretation |
| "MLE is always optimal" | Only asymptotically; can overfit, can be biased in finite samples | Consider MAP / regularization |
| "Bootstrap fixes any sample size problem" | Tiny $n$ → biased bootstrap | Need $n$ large enough for empirical to approximate true |

---

## 9. Eight most-asked interview questions

1. **What's the difference between a confidence interval and a credible interval?** (Frequentist vs Bayesian; "interval random vs $\theta$ random.")
2. **Derive the MLE for a Gaussian.** (Lock down log-likelihood + zero-derivative routine.)
3. **What does a p-value mean exactly?** (Probability of data this extreme under $H_0$, NOT $\mathbb{P}(H_0|\mathrm{data})$.)
4. **When would you use bootstrap?** (No analytic SE, ML metrics like AUC, paired model comparison.)
5. **What's the bias-variance tradeoff for estimators?** (MSE = bias² + variance; biased estimators can win.)
6. **Why use Bessel's correction ($n-1$)?** (Sample variance with $n$ underestimates; $n-1$ unbiases it.)
7. **What's MAP and how does it relate to regularization?** (MLE + log-prior; Gaussian prior = $\ell_2$, Laplace = $\ell_1$.)
8. **You ran 20 A/B tests, two were significant at $p < 0.05$. What do you do?** (Multiple testing — apply Bonferroni or BH correction.)

---

## 10. Drill plan

- For Bernoulli, Gaussian (both params), Poisson — derive MLE on paper. 5 minutes each.
- For Beta-Bernoulli — derive posterior. Recite posterior mean.
- Bootstrap loop in 30 lines of NumPy. AUC CI on a real dataset.
- For each common test (z, t, chi-squared, two-prop), recite: assumptions, statistic, null distribution, when to use.
- Interpret 5 different p-values and CI statements; flag the wrong ones.

---

## 11. Further reading

- Casella & Berger, *Statistical Inference* — the canonical text.
- Wasserman, *All of Statistics* — fast & broad, ML-friendly.
- Efron & Hastie, *Computer Age Statistical Inference* — bootstrap, modern methods.
- Gelman et al., *Bayesian Data Analysis* — Bayesian bible.
- xkcd 882 (jelly beans) — the canonical multiple-testing comic.
