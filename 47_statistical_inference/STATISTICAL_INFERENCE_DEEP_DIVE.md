# Statistical Inference — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Statistical inference is what separates "I trained a model and it has 87% accuracy" from "I have evidence that my model's true accuracy is 87% ± 1.2% and that's a statistically significant 0.4-point improvement over the baseline." Senior interviews probe this hard because production ML decisions hinge on it.

---

## 1. Estimators — what they are and what makes one "good"

An **estimator** is a function $\hat{\theta} = T(X_1, \ldots, X_n)$ of the data that tries to recover an unknown parameter $\theta$.

> **Saying it out loud.** An estimator is a recipe that turns data into a guess about something you can't see — the sample mean guessing the true mean, or your test accuracy guessing real-world accuracy. The move that makes everything else click is realizing the estimator is itself random: run the experiment again and you get a different number. So the whole subject is about describing that spread. Once your reported metric feels like one draw from a distribution rather than a fact, you start reporting intervals instead of point numbers, which is most of what separates a careful ML engineer from a careless one.

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

> **Saying it out loud.** Three properties and they're not the same thing. Unbiased means you're right on average across infinitely many repeats — which says nothing about the one estimate you have. Consistent means you converge to the truth as data grows, which is the property you actually want. Efficient means you have the smallest variance possible among unbiased estimators, and the Cramér-Rao bound tells you what that floor is. The punchline is the bias-variance decomposition: since squared error is bias squared plus variance, deliberately accepting some bias to cut variance often wins. That's not a hack, it's the entire justification for regularization.

---

## 2. Maximum likelihood estimation

The likelihood: $L(\theta) = \prod_i p(x_i | \theta)$. The log-likelihood: $\ell(\theta) = \sum_i \log p(x_i | \theta)$.

MLE: $\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

> **Saying it out loud.** Maximum likelihood asks which parameter value makes the data you actually observed most probable, and then picks that one. You maximize the log rather than the raw product, because multiplying thousands of small probabilities underflows to zero in floating point while summing logs doesn't. Worth stating explicitly that this is the same objective as cross-entropy — every neural network trained with cross-entropy is doing maximum likelihood, so this isn't a separate topic from deep learning, it's the same one in different notation.

### Properties of MLE
- **Consistent**: $\hat{\theta}_{\mathrm{MLE}} \to_p \theta_0$
- **Asymptotically normal**: $\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})$
- **Asymptotically efficient**: variance hits CRLB
- **Invariant to reparameterization**: $\widehat{g(\theta)} = g(\hat{\theta})$

> **Saying it out loud.** Four properties, and the last three all carry an asterisk that says 'asymptotically.' It converges to the truth, it becomes Gaussian around the truth so you can build intervals from it, it hits the efficiency floor, and it survives reparameterization so the MLE of a function is the function of the MLE. That invariance is worth flagging because unbiasedness does not have it — the square root of an unbiased variance estimate is not an unbiased standard deviation. And the asymptotics matter in practice: in small samples MLE can be meaningfully biased and its normal-approximation intervals can be junk.

### Worked examples

**Bernoulli (coin flip):** $p(x|\theta) = \theta^x (1-\theta)^{1-x}$.
$\ell(\theta) = \sum x_i \log \theta + (n - \sum x_i) \log(1-\theta)$.
$\hat{\theta}_{\mathrm{MLE}} = \bar{x}$ (sample mean).

**Gaussian (known variance):** $p(x|\mu) = \mathcal{N}(\mu, \sigma^2)$.
$\hat{\mu}_{\mathrm{MLE}} = \bar{x}$.

**Gaussian (both unknown):** $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$ — biased! Unbiased estimator uses $1/(n-1)$.

> **Saying it out loud.** Bernoulli gives you the fraction of ones, which is what your intuition said before any calculus — and that's the point, the machinery confirms the obvious answer in the obvious case. Gaussian gives you the sample mean and a variance divided by $n$, note $n$ and not $n-1$, so the maximum likelihood variance is biased low. That isn't a mistake in the derivation, it's genuinely what maximum likelihood produces, and Bessel's correction is a separate deliberate fix. Interviewers use exactly this to check whether you derived it or copied it from a textbook that had already corrected it.

---

## 3. Confidence intervals

> **In plain language.** A confidence interval is a statement about a procedure, not about the particular numbers you got. Think of the procedure as a machine that stamps out intervals: 95 percent of the intervals it produces contain the truth, but the one in your hand either does or doesn't, and you have no way to know which.

A $1-\alpha$ CI is a random interval $[L, U]$ with $\mathbb{P}(L \leq \theta \leq U) = 1-\alpha$ — *over repeated sampling*.

**Common misinterpretation:** "There's a 95% probability $\theta$ is in [1.2, 3.4]." Wrong (under frequentist interpretation). $\theta$ is fixed; the *interval* is random. The correct statement: "If we repeated this procedure many times, 95% of intervals would contain $\theta$."

> **Saying it out loud.** This is the one people get wrong, so be careful and be explicit. The 95 percent describes the procedure across repeated samples: if you ran the whole study many times, about 95 percent of the intervals you'd construct would cover the true value. It does not mean there's a 95 percent chance the parameter is inside the interval you computed — in the frequentist framing the parameter is a fixed number, so it's either in there or it isn't, and the probability is zero or one. The analogy I'd use is a ring toss: 95 percent of your throws land on the peg, but once a ring is on the ground it's either around the peg or it isn't. If you want the statement everyone actually wants to make, that's a Bayesian credible interval, and it costs you a prior.

### Wald CI (asymptotic)

For an asymptotically normal estimator:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \mathrm{SE}(\hat{\theta})
$$

with $z_{0.025} = 1.96$ for 95%. Standard error from Fisher information or sample variance.

> **Saying it out loud.** Estimate plus or minus 1.96 standard errors — the familiar interval, and it works because maximum likelihood estimates become Gaussian in large samples, so you're borrowing a normal shape for a sampling distribution you never derived. The load-bearing word is asymptotic. Near a parameter boundary or in a small sample the approximation fails, and you get visible nonsense like a confidence interval for a probability that dips below zero. Treat that as a signal, not a rounding artifact, and switch to a Wilson interval or a bootstrap.

### Bootstrap CI

When you can't compute SE analytically: resample data with replacement $B$ times, compute $\hat{\theta}^{(b)}$ for each, then take quantiles (percentile method) or use bootstrap-t.

```
for b in 1 .. B:
  sample X_b with replacement from X (size n)
  compute theta_b = T(X_b)
CI = [quantile(thetas, alpha/2), quantile(thetas, 1-alpha/2)]
```

Bootstrap is non-parametric, simple, and extremely useful in ML for things like AUC confidence intervals.

> **Saying it out loud.** The bootstrap fakes having many datasets by resampling your one dataset with replacement, thousands of times, recomputing the statistic each pass. The spread of those recomputed values approximates the sampling distribution, and you read the interval off the percentiles. The appeal is that it needs no formula for the standard error, which is why it's the default for metrics like AUC or F1 where the analytic answer is unpleasant or nonexistent. A thousand resamples is normal, ten thousand if you're going to publish the number.

### Bayesian credible interval

The interval that contains 95% of the posterior probability mass. A *different* concept than Wald CI — and the credible interval supports the natural-language "$\theta$ is in [...] with 95% probability" interpretation, conditional on prior.

> **Saying it out loud.** A credible interval is the one that means what people think a confidence interval means: given your prior and your data, there is genuinely a 95 percent probability the parameter lies in this range. The reason it can say that is that the Bayesian treats the parameter as a random quantity with a posterior distribution, so probability statements about it are well-formed. The price is the prior — you can't get that sentence for free. In practice with a lot of data and a weak prior the two intervals nearly coincide, which is why the distinction only starts to bite when your sample is small or your prior is strong.

---

## 4. Hypothesis testing

Testing a claim $H_0$ vs alternative $H_1$.

> **Saying it out loud.** A hypothesis test asks how weird your data would look in a world where nothing interesting is happening. You pick a statistic, work out its distribution under that boring world, and check whether what you observed sits far out in the tail. The engine is that null distribution — everything else is bookkeeping, and it's also where your assumptions hide, because if the null distribution is wrong then the p-value is meaningless no matter how carefully you computed it.

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

> **Saying it out loud.** A p-value is the probability of data at least this extreme assuming the null is true. The direction is everything: it's the probability of the data given the hypothesis, never the probability of the hypothesis given the data, and those can differ by orders of magnitude. If you test something implausible to begin with, a p-value of 0.04 leaves it still probably false. The two other traps are treating a small p-value as evidence of a big effect — it's evidence of a precisely measured effect, which can be tiny — and treating a large one as proof of no effect, when it usually just means you were underpowered.

### Standard tests

**z-test**: Gaussian, known variance. $z = (\bar{x} - \mu_0) / (\sigma/\sqrt{n})$.

**t-test**: Gaussian, unknown variance. Use sample SD; statistic follows $t_{n-1}$.

**Chi-squared**: categorical data goodness-of-fit, contingency tables. $\chi^2 = \sum (O - E)^2 / E$.

**Mann-Whitney U / Wilcoxon**: non-parametric two-sample.

**A/B test (proportions)**: binomial / two-proportion z-test.

> **Saying it out loud.** The choice is basically driven by what your data looks like. Continuous with estimated variance, which is essentially always, means a t-test; z-tests assume you know the true variance, which almost never happens outside textbooks. Counts in categories mean chi-squared, with the caveat that expected cell counts under about 5 break the approximation and you want Fisher's exact test. If your data is skewed or ordinal, Mann-Whitney compares ranks instead of means and doesn't need normality. And A/B tests on conversion rates are just a two-proportion comparison. Knowing which one to reach for is more valuable in an interview than being able to derive any of them.

### Type I vs Type II
- Type I (false positive): reject $H_0$ when true. Controlled by $\alpha$.
- Type II (false negative): fail to reject when $H_1$ true. $\beta$, depends on effect size, $n$, $\alpha$.

**Power analysis** picks $n$ to achieve target $1-\beta$ (typically 80%) for a minimum detectable effect.

> **Saying it out loud.** Type I is crying wolf, Type II is missing the wolf. You control the first directly by setting alpha, usually 0.05, and the second indirectly through power, which mostly means sample size. They trade off — tighten alpha and you catch fewer false alarms while missing more real effects — so which one you fear should depend on the domain. The practical point worth making is that underpowered studies are actively harmful: not only do you miss real effects, but the effects you do detect are systematically inflated, because only unusually large estimates clear the bar. Eighty percent power is the conventional target and a lot of work sits well below it.

---

## 5. Multiple testing

When you run $m$ tests at $\alpha = 0.05$, the family-wise probability of *any* false rejection grows: under independence, $1 - (1-\alpha)^m \approx m\alpha$ for small $\alpha$. With $m=20$ tests at $\alpha=0.05$, you expect 1 false positive.

> **Saying it out loud.** If each test has a one-in-twenty chance of a false alarm, running twenty tests gives you roughly a 64 percent chance of at least one, even when absolutely nothing is real. No individual test is misbehaving — the problem is looking many times and reporting the winner. In ML this is everywhere and almost never corrected: hyperparameter sweeps, subgroup breakdowns, dashboards with fifty metrics, teams running dozens of A/B tests a quarter. Quoting that 64 percent number is the fastest way to make the point land.

### Corrections
- **Bonferroni**: use $\alpha/m$ per test. Conservative; controls family-wise error rate (FWER).
- **Holm-Bonferroni**: step-down version — less conservative.
- **Benjamini-Hochberg**: controls false discovery rate (FDR = expected proportion of false positives among rejections). Less conservative; standard in genomics, A/B testing at scale.

> **Saying it out loud.** Bonferroni divides alpha by the number of tests — dead simple, guaranteed regardless of how the tests correlate, and brutally conservative once you have hundreds of tests. Holm is a strictly better version of the same idea, uniformly more powerful with the same guarantee, so there's no real reason to prefer plain Bonferroni. Benjamini-Hochberg changes the goal: instead of avoiding any false positive, it controls the expected fraction of your discoveries that are wrong. That's the right target for screening, and it's why genomics and large-scale A/B testing standardized on it.

### When this matters in ML
- Hyperparameter search: 100 hyperparam combos → some "win" by luck.
- Many A/B tests on the same data: false positives.
- Feature selection: testing each feature for significance inflates Type I.
- Subgroup analysis ("but the model works better for users in California!") — almost always overstated without correction.

> **Saying it out loud.** The one I'd lead with is the hyperparameter sweep, because everyone has been burned by it: 200 configurations evaluated on the same validation set is 200 tests, so the winner is inflated by selection and reliably disappoints on the true holdout. Subgroup analysis is the other big one — slice your users enough ways and something will look significant in California. The defense in practice isn't usually a formal correction, it's a holdout you touch exactly once, plus treating the validation-to-test drop as expected rather than as a bug.

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

> **Saying it out loud.** The bootstrap is a cheeky idea that works remarkably well: you want to know how your estimate would vary across datasets, you have one dataset, so you treat it as the population and draw from it with replacement. Do that a thousand times, recompute your metric each time, and the spread of those numbers is your sampling distribution. No formulas, no distributional assumptions, works for any statistic you can compute. The failures are specific and worth naming — extreme order statistics like the maximum, because a resample can never exceed the observed max, heavy tails, and time series unless you resample blocks to preserve correlation.

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

> **Saying it out loud.** The whole difference is what you treat as random. A frequentist says the parameter is a fixed unknown and the data is random; a Bayesian says the data is what it is and the uncertainty lives in your belief about the parameter. From that, Bayes' rule says your updated belief is the likelihood times your prior, normalized. The practical consequence is that a Bayesian can make the statement everyone wants — there's a 95 percent probability the value lies here — and the cost is having to say what you believed beforehand. Everything downstream, conjugacy, MAP, MCMC, is machinery for actually computing that posterior.

### Conjugate priors

Posterior in the same family as prior. Examples:
- Beta prior + Bernoulli likelihood → Beta posterior.
- Gamma prior + Poisson likelihood → Gamma posterior.
- Dirichlet prior + multinomial likelihood → Dirichlet posterior.
- Gaussian prior + Gaussian likelihood (known variance) → Gaussian posterior.

**Beta-Bernoulli example:** prior $\theta \sim \mathrm{Beta}(\alpha, \beta)$. After observing $s$ successes in $n$ trials: posterior $\theta | x \sim \mathrm{Beta}(\alpha + s, \beta + n - s)$. Posterior mean: $(\alpha + s)/(\alpha + \beta + n)$.

> **Saying it out loud.** A conjugate prior is one where the posterior stays in the same family, so updating becomes arithmetic instead of integration. Beta-Bernoulli is the one to know cold: you observe successes and failures and simply add them to the Beta's two parameters. The reason it matters historically is that before MCMC, conjugacy was the only route to a closed-form posterior. The reason it matters now is intuition — the posterior mean is successes plus alpha over total plus alpha plus beta, which is exactly add-one smoothing, so a prior is literally a set of imaginary observations you've already seen.

### MAP

Maximum a posteriori: $\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta p(\theta | x) = \arg\max_\theta [\log p(x|\theta) + \log p(\theta)]$.

This is exactly MLE + log-prior penalty. The penalty *is* the regularizer.

- Gaussian prior on weights → $\ell_2$ regularization (ridge).
- Laplace prior → $\ell_1$ (lasso).

> **Saying it out loud.** MAP picks the peak of the posterior, which works out to maximum likelihood plus a log-prior term. The connection worth having instantly is that this makes regularization Bayesian: an L2 penalty is a Gaussian prior on the weights, and an L1 penalty is a Laplace prior. Once you see that, lasso's sparsity stops being mysterious — the Laplace prior has a sharp spike at zero, so it genuinely believes most coefficients are exactly zero, while a Gaussian just believes they're all smallish. And the regularization strength is the prior's inverse variance, so cranking it up is literally being more confident before you look at the data.

### Posterior summaries
- Posterior mean: $\mathbb{E}[\theta | x]$
- Posterior median, mode (MAP)
- Credible interval: $[L, U]$ with $\mathbb{P}(\theta \in [L,U] | x) = 0.95$

> **Saying it out loud.** A posterior is a whole distribution, and a summary throws most of it away — so pick the summary that matches the decision. The mean minimizes squared error, the median minimizes absolute error and is robust to skew, and the mode is what MAP gives you. In high dimensions the mode can be wildly unrepresentative of where the probability mass actually is, which is the standard critique of MAP: you're reporting the tallest point of a distribution that might have almost no volume there. If you only report one number from a posterior, you've discarded the reason you did Bayesian inference in the first place.

### Bayesian inference in practice
- Conjugate cases: closed-form (rare beyond simple models).
- MCMC (Metropolis-Hastings, Gibbs, HMC): sample from posterior.
- Variational inference: approximate posterior with simpler distribution; minimize KL.
- Laplace approximation: Gaussian centered at MAP.

> **Saying it out loud.** Conjugate cases are exact and almost never apply beyond toy models. MCMC gives you asymptotically exact samples and is slow, with the added pain that deciding whether it has converged is genuinely hard. Variational inference turns the problem into optimization by fitting a simple distribution to the posterior, which is fast and scalable and biased — specifically, the standard reverse-KL objective is mode-seeking, so VI systematically underestimates posterior variance and hands you overconfident uncertainty. Laplace approximation is the cheapest option, a Gaussian fitted at the MAP. At neural-network scale you take VI or something cheaper, because MCMC over millions of parameters isn't happening.

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
