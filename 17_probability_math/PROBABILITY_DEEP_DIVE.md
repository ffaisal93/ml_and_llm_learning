# Probability for ML — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Probability is the substrate of ML. Senior interviews use probability to test whether you understand uncertainty, can do clean derivations, and can apply Bayesian reasoning under pressure. This deep dive nails the foundations.

---

## 1. Probability axioms and basic identities

A probability is a function $\mathbb{P}$ on a sample space $\Omega$ satisfying:
- $\mathbb{P}(\Omega) = 1$, $\mathbb{P}(\emptyset) = 0$.
- $\mathbb{P}(A) \in [0, 1]$ for any event $A$.
- Countable additivity: $\mathbb{P}(\bigcup A_i) = \sum \mathbb{P}(A_i)$ for disjoint $A_i$.

### Identities to know cold

- Complement: $\mathbb{P}(A^c) = 1 - \mathbb{P}(A)$.
- Inclusion-exclusion: $\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.
- Union bound: $\mathbb{P}(\bigcup A_i) \leq \sum \mathbb{P}(A_i)$.
- Conditional: $\mathbb{P}(A|B) = \mathbb{P}(A \cap B)/\mathbb{P}(B)$.
- Multiplication: $\mathbb{P}(A \cap B) = \mathbb{P}(A|B)\mathbb{P}(B)$.
- Independence: $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$ iff $A, B$ independent.
- **Law of total probability**: $\mathbb{P}(A) = \sum_i \mathbb{P}(A|B_i)\mathbb{P}(B_i)$ for partition $\{B_i\}$.
- **Bayes' theorem**: $\mathbb{P}(A|B) = \mathbb{P}(B|A)\mathbb{P}(A)/\mathbb{P}(B)$.

> **Saying it out loud.** Probability is just a way of splitting up a total of one across all the things that could happen, and the axioms only say that the pieces are non-negative and add to one. Everything else falls out of that. The two identities you actually use are the law of total probability — chop the world into cases, work out the answer within each case, and average weighted by how likely each case is — and Bayes' rule, which is how you run that backwards to update a belief after seeing evidence. If you can state those two cleanly and say what a partition is, you're fine; the rest is bookkeeping.

---

## 2. Random variables, expectations, variance

A random variable $X$ is a function $\Omega \to \mathbb{R}$ with an induced distribution.

**PMF/PDF**: $p_X(x)$ for discrete; $f_X(x)$ for continuous.

**CDF**: $F_X(x) = \mathbb{P}(X \leq x)$.

### Expectation

$$
\mathbb{E}[X] = \sum_x x p(x) \quad \text{or} \quad \int x f(x) dx
$$

**Linearity** (always — even for dependent RVs):

$$
\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]
$$

**Law of the unconscious statistician**: $\mathbb{E}[g(X)] = \sum g(x) p(x)$.

> **Saying it out loud.** Expectation is the long-run average, the centre of mass of the distribution. The one property to hold onto is linearity: the expectation of a sum is the sum of expectations, and this is true whether or not the variables are independent. That's unusually generous — almost nothing else in probability works without independence — and it's why you can compute the expected number of collisions or the expected loss over a batch without ever thinking about correlations. The trap to name is that this does *not* extend to products or to nonlinear functions; the expectation of a function is not the function of the expectation, which is Jensen's inequality.

### Variance and covariance

$$
\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

$$
\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**Variance of a sum:**

$$
\mathrm{Var}(aX + bY) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y) + 2ab \mathrm{Cov}(X, Y)
$$

For independent $X, Y$: $\mathrm{Cov} = 0$ → variance adds. (Note: $\mathrm{Cov} = 0$ does NOT imply independence in general, only for jointly Gaussian.)

> **Saying it out loud.** Variance is the average squared distance from the mean — how spread out the thing is — and covariance is the same idea for two variables, measuring whether they move together. The important asymmetry is that variances only add cleanly when the variables are independent; otherwise you pick up a cross term of twice the covariance. And the classic trap: zero covariance does not mean independence. Take a variable symmetric around zero and its square — perfectly determined by each other, zero covariance. The one place the implication does hold is jointly Gaussian variables, which is why Gaussians feel so much easier than everything else.

### Conditional expectation and variance

**Tower (law of total expectation)**:

$$
\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X | Y]]
$$

**Law of total variance**:

$$
\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X | Y)] + \mathrm{Var}(\mathbb{E}[X | Y])
$$

These are *constantly* useful in ML problems involving hierarchical or latent models (e.g., bias-variance decomposition arguments).

> **Saying it out loud.** The tower property says you can find an average by averaging within groups first and then averaging across groups, weighted by group size. Nothing surprising — the average height of a country is the average of the per-region averages weighted by population. The law of total variance is the sharper one: total spread equals the average spread within groups plus the spread of the group averages. That's exactly the bias-variance decomposition wearing different clothes, and it's the tool for any model with a latent variable or a hierarchy — say it, and you've shown you can do the derivation rather than just recite the result.

---

## 3. Common distributions — what to know

| Distribution | PMF/PDF | Mean | Variance | When |
|---|---|---|---|---|
| Bernoulli($p$) | $p^x (1-p)^{1-x}$ | $p$ | $p(1-p)$ | Binary outcome |
| Binomial($n, p$) | $\binom{n}{x} p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | Sum of Bernoullis |
| Geometric($p$) | $(1-p)^{x-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |
| Poisson($\lambda$) | $\lambda^x e^{-\lambda}/x!$ | $\lambda$ | $\lambda$ | Rare events, count data |
| Uniform($a, b$) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | No info, bounded |
| Normal($\mu, \sigma^2$) | $\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | CLT, continuous |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Time to event, memoryless |
| Gamma($k, \theta$) | $\propto x^{k-1}e^{-x/\theta}$ | $k\theta$ | $k\theta^2$ | Sum of exponentials |
| Beta($\alpha, \beta$) | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\alpha/(\alpha+\beta)$ | varies | Probability of probability |

### Key relationships
- Sum of $n$ iid Bernoulli($p$) → Binomial($n, p$).
- Limit of Binomial($n, p$) with $np = \lambda$ fixed → Poisson($\lambda$).
- Sum of independent Poissons → Poisson with summed rate.
- Sum of $k$ iid Exponential($\lambda$) → Gamma($k, 1/\lambda$).
- $\chi^2_k$ = sum of $k$ squared standard normals.
- t-distribution: ratio of standard normal to $\sqrt{\chi^2_k/k}$.

> **Saying it out loud.** The distributions all come from generating stories, and if you know the story you can rebuild the formula. Bernoulli is one coin flip; binomial is many flips counted up; geometric is how long you wait for the first success; Poisson is what binomial becomes when you have a huge number of chances each of which almost never fires, like server requests per second; exponential is the waiting time between those Poisson events; gamma is several exponential waits added together; beta is a distribution over a probability, which is why it's the natural prior for a click-through rate. The relationships matter more than the formulas, because interviewers ask things like what a binomial converges to. And the one memorable fact is that exponential and geometric are the only memoryless distributions — a lightbulb that's already burned for a year is exactly as likely to fail tomorrow as a new one.

---

## 4. The Gaussian — workhorse distribution

PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$.

### Multivariate Gaussian

*In plain language:* this is the bell curve in many dimensions. The mean vector says where the cloud of points is centred, and the covariance matrix says what shape it is — how wide along each axis and how tilted. The exponent below is just a squared distance from the centre, measured in units that account for that stretching and tilting, and the fraction in front is bookkeeping that makes the whole thing integrate to one.

$$
f(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)
$$

### Properties to memorize

- **Affine transformations**: if $X \sim \mathcal{N}(\mu, \Sigma)$, then $AX + b \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)$.
- **Marginals are Gaussian**: any marginal of a multivariate Gaussian is Gaussian.
- **Conditionals are Gaussian**: $X|Y$ where $(X,Y)$ is jointly Gaussian is Gaussian, with mean and variance computable in closed form.
- **Sum of independent Gaussians is Gaussian**.
- **Uncorrelated jointly Gaussian = independent**. (Special property — does NOT hold in general.)

### Conditioning formula

For $\begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}\right)$:

$$
X_1 | X_2 = x_2 \sim \mathcal{N}\big(\mu_1 + \Sigma_{12} \Sigma_{22}^{-1}(x_2 - \mu_2), \; \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big)
$$

This is the foundation of Gaussian processes, Kalman filters, Bayesian linear regression, and many other methods.

> **Saying it out loud.** The Gaussian is the workhorse because it's closed under almost everything you'd want to do to it. Add two independent Gaussians, you get a Gaussian; apply any linear map, you get a Gaussian; take a marginal or condition on part of it, still Gaussian. That closure is what makes Kalman filters, Gaussian processes and Bayesian linear regression tractable in closed form instead of requiring simulation. The conditioning formula is the one to be able to state, because it's literally linear regression falling out of a joint distribution. And the property people misuse is the last one — uncorrelated implies independent only when the variables are jointly Gaussian, and jointly is doing real work in that sentence.

---

## 5. Convergence and limit theorems

*In plain language:* these two theorems are the reason averaging works. The law of large numbers says an average of many samples lands on the truth; the central limit theorem says how far off it typically is, and that the error is bell-shaped. Everything below is the precise statement of those two sentences.

### Law of large numbers (LLN)

For iid $X_i$ with finite mean $\mu$:
- **Weak LLN**: $\bar{X}_n \to_p \mu$.
- **Strong LLN**: $\bar{X}_n \to_{a.s.} \mu$.

The empirical mean converges to the true mean. This is why Monte Carlo estimation works.

> **Saying it out loud.** The law of large numbers is the promise that averaging works: take enough independent samples and the sample mean converges to the true mean. That's the licence for Monte Carlo, for estimating a loss on a minibatch, for A/B testing at all. Weak versus strong is about the mode of convergence — in probability versus almost surely — and honestly the distinction rarely matters in practice, though it's worth knowing the words. The condition that does matter is a finite mean; for something like a Cauchy distribution the sample average never settles down no matter how much data you collect.

### Central limit theorem (CLT)

For iid $X_i$ with mean $\mu$ and finite variance $\sigma^2$:

$$
\sqrt{n}(\bar{X}_n - \mu) \to \mathcal{N}(0, \sigma^2)
$$

The sample mean is approximately Gaussian for large $n$, regardless of underlying distribution. This is why so many statistical tests assume normality of the sample mean.

**Caveats:**
- Need finite variance — fails for heavy-tailed distributions like Cauchy.
- Convergence rate depends on third moment; very skewed distributions need larger $n$.
- For finite samples, use $t$-distribution instead of normal for inference.

> **Saying it out loud.** The central limit theorem is why the bell curve is everywhere. Add up a lot of independent random things, and the total is approximately Gaussian no matter what the individual things looked like — that's why measurement error, heights, and estimator noise all end up bell-shaped. It's also what lets you put a confidence interval on any average without knowing the underlying distribution. The caveats are the interesting part: you need finite variance, so heavy-tailed things like Cauchy or many financial returns break it; convergence is slower for skewed distributions; and at small sample sizes you should be using a t-distribution instead of a normal. Naming the finite-variance caveat is what separates a memorised answer from an understood one.

---

## 6. Bayes' theorem — key applications

### Naive Bayes classifier

$$
\mathbb{P}(C | x) \propto \mathbb{P}(x | C) \mathbb{P}(C) = \prod_j \mathbb{P}(x_j | C) \mathbb{P}(C)
$$

The "naive" assumption: features independent given class. Surprisingly competitive baseline.

> **Saying it out loud.** Naive Bayes says: pick the class that best explains the evidence, where you score each class by its prior times how likely it makes each feature, and you pretend the features are independent given the class. The independence assumption is obviously false for text — "machine" and "learning" co-occur constantly — and yet the classifier works well anyway, because for ranking classes you only need the ordering to be right, not the probabilities. That's the interesting bit to say. The failure mode is that its confidence scores are badly calibrated, wildly overconfident, so use it for the argmax and never for the probability.

### Medical testing (the canonical interview question)

Disease prevalence $\mathbb{P}(D) = 0.01$. Test sensitivity $\mathbb{P}(+|D) = 0.95$, specificity $\mathbb{P}(-|D^c) = 0.95$.

$$
\mathbb{P}(D|+) = \frac{\mathbb{P}(+|D)\mathbb{P}(D)}{\mathbb{P}(+)} = \frac{0.95 \cdot 0.01}{0.95 \cdot 0.01 + 0.05 \cdot 0.99} \approx 0.16
$$

Even with a 95% accurate test, only 16% of positives have the disease. This is the **base rate fallacy** — and the reason rare-event detection is hard in ML.

> **Saying it out loud.** This is the base rate fallacy and it's the single most-asked probability question. A 95 percent accurate test for a disease that one person in a hundred has still means a positive result is wrong most of the time — only about 16 percent of the people who test positive actually have it. The reason is that the healthy group is ninety-nine times bigger, so 5 percent of a huge group produces more false positives than 95 percent of a tiny group produces true positives. In ML terms, this is exactly why fraud detection and rare-event classification are hard: at a 1 percent base rate, precision collapses even with an excellent model, which is why you report precision-recall rather than accuracy.

### Bayesian update

Prior $p(\theta)$ + likelihood $p(x|\theta)$ → posterior $p(\theta|x) \propto p(x|\theta)p(\theta)$. Sequential data: posterior becomes prior for next observation.

> **Saying it out loud.** A Bayesian update is just: what you believed before, multiplied by how well the new data fits, renormalised. Prior times likelihood gives posterior. The neat structural property is that it's sequential — yesterday's posterior is today's prior — so you can process data one point at a time and end up in exactly the same place as if you'd seen it all at once. That's what makes it natural for online learning and for Thompson sampling in bandits. The honest caveat is that the renormalising step is usually an intractable integral, which is why practical Bayesian methods are all about approximating it with MCMC or variational inference.

---

## 7. Joint, marginal, conditional

For two RVs $X, Y$ with joint distribution:

- **Joint PMF/PDF**: $p_{X,Y}(x, y)$.
- **Marginal**: $p_X(x) = \sum_y p_{X,Y}(x, y)$ or $\int p_{X,Y}(x, y) dy$.
- **Conditional**: $p_{X|Y}(x|y) = p_{X,Y}(x, y)/p_Y(y)$.

**Independence**: $p_{X,Y}(x, y) = p_X(x) p_Y(y)$.

**Conditional independence**: $X \perp Y | Z$ iff $p(x, y | z) = p(x|z) p(y|z)$. Different from unconditional independence.

> **Saying it out loud.** The joint distribution is the full picture; the marginal is what you get when you stop caring about one variable and sum or integrate it away; the conditional is what's left when you fix one variable to a known value. Everything else in probability is manipulating between those three. The subtle one is conditional independence, which is genuinely different from independence — two variables can be dependent overall but independent once you know a third. Shoe size and reading ability in children are correlated until you condition on age, at which point the correlation vanishes. That's the whole basis of graphical models, and it's also the assumption naive Bayes leans on.

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Cov = 0 implies independence? | Yes | Only for jointly Gaussian; in general no |
| CLT requires iid? | Yes (strict) | Identical distribution can be relaxed (Lindeberg conditions); finite variance critical |
| $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ implies independence? | Yes | Only that they're uncorrelated |
| Sum of independent variances? | Yes (always) | Yes if variance exists; for dependent must include $2\mathrm{Cov}$ |
| Memoryless property? | Geometric and exponential | Yes — specifically, the only memoryless distributions |
| Bayes' theorem requires prior to be informative? | Yes | No — uniform prior is fine; Bayes is about belief update |
| If $X, Y$ are jointly Gaussian, $X-Y$ is Gaussian? | Maybe | Yes — affine transformations of Gaussians are Gaussian |

---

## 9. Eight most-asked interview questions

1. **Walk through Bayes' theorem with the medical-testing example.** (Lock down the base-rate-fallacy intuition.)
2. **Derive CLT informally.** (Sum of zero-mean RVs scaled by $1/\sqrt{n}$ converges to Gaussian; characteristic function argument.)
3. **State the law of total expectation and law of total variance.** (Tower property; bias-variance decomposition uses this.)
4. **What's the difference between uncorrelated and independent?** (Cov = 0 vs joint = product of marginals; Gaussian is the special case where they coincide.)
5. **How do you sample from a non-uniform distribution?** (Inverse CDF, rejection sampling, MCMC; understand each.)
6. **Compute the marginal of a 2D Gaussian.** (Marginalize one variable; result is Gaussian with the corresponding marginal mean and variance.)
7. **Explain conditional independence.** (Different from independence; canonical example: causes of a common effect.)
8. **What does Poisson approximate?** (Binomial with $n$ large, $p$ small, $np = \lambda$ fixed; rare events.)

---

## 10. Drill plan

- For each common distribution: PMF/PDF, mean, variance, generating story. 1 minute each.
- Bayes' problem: medical test → recompute given different prevalence/sensitivity. Until automatic.
- Derive Gaussian conditional formula from the joint density manipulation.
- Practice 5 problems where you apply the law of total variance.
- Compute Var(sum) and Var(mean) for iid and non-iid cases.

---

## 11. Further reading

- Casella & Berger, *Statistical Inference*, ch. 1–4.
- Wasserman, *All of Statistics*, ch. 1–4.
- Pitman, *Probability* — friendly introduction.
- 3blue1brown, "But what is the Central Limit Theorem?" — beautiful visual intuition.
