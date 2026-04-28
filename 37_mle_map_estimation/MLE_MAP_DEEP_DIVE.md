# MLE and MAP Estimation — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This topic underpins almost everything in classical and modern ML. Cross-entropy, ridge/lasso, Bayesian deep learning, RLHF reward modeling — all are MLE/MAP under specific likelihoods and priors. Senior interviews probe whether you can derive these *cleanly*, not just recognize them.

---

## 1. The likelihood function

Given iid data $X_1, \ldots, X_n \sim p(\cdot | \theta)$ and a parametric family $\{p(\cdot | \theta) : \theta \in \Theta\}$:

$$
L(\theta) = \prod_{i=1}^n p(x_i | \theta), \qquad \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x_i | \theta)
$$

The likelihood treats $\theta$ as the variable and the data as fixed — opposite of how $p$ is usually written.

**Why log?** Sums are easier than products, numerically stable (no underflow), and convex programming on $\ell$ is often tractable.

**MLE**: $\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

---

## 2. Worked MLE derivations

### Bernoulli

$x_i \in \{0, 1\}$, $p(x|\theta) = \theta^x (1-\theta)^{1-x}$.

$$
\ell(\theta) = \sum_i [x_i \log \theta + (1-x_i) \log(1-\theta)] = s \log \theta + (n-s) \log(1-\theta)
$$

where $s = \sum x_i$. Setting $\partial \ell / \partial \theta = 0$:

$$
\frac{s}{\theta} - \frac{n-s}{1-\theta} = 0 \implies \hat{\theta}_{\mathrm{MLE}} = \frac{s}{n} = \bar{x}
$$

Pure intuition: MLE for Bernoulli is the empirical frequency.

### Gaussian (mean and variance unknown)

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log \sigma^2 - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2
$$

$\partial \ell/\partial \mu = 0$: $\hat{\mu} = \bar{x}$.

$\partial \ell/\partial \sigma^2 = 0$: $\hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2$.

The variance MLE has a $1/n$, not $1/(n-1)$. It's biased (too small) — Bessel's correction unbiases it.

### Multinomial

$x_i$ is a one-hot category among $K$ classes. Parameters $\theta_k$, $\sum_k \theta_k = 1$.

$$
\ell(\theta) = \sum_i \sum_k x_{i,k} \log \theta_k = \sum_k n_k \log \theta_k
$$

where $n_k = \sum_i x_{i,k}$. With Lagrangian for the simplex constraint:

$$
\hat{\theta}_k = n_k / n
$$

Empirical frequency of each category.

### Poisson

$p(x|\lambda) = e^{-\lambda} \lambda^x / x!$. $\ell(\lambda) = \sum_i [-\lambda + x_i \log \lambda - \log x_i!]$.

$\hat{\lambda}_{\mathrm{MLE}} = \bar{x}$.

### Linear regression

$y_i = w^\top x_i + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Likelihood is Gaussian:

$$
\ell(w) = -\frac{1}{2\sigma^2} \sum_i (y_i - w^\top x_i)^2 + \mathrm{const}
$$

Maximizing $\ell$ = minimizing squared error = OLS:

$$
\hat{w}_{\mathrm{MLE}} = (X^\top X)^{-1} X^\top y
$$

**Key insight: OLS *is* MLE under Gaussian noise.** The choice of squared loss isn't arbitrary — it's the negative log-likelihood of a Gaussian.

### Logistic regression

$y_i \in \{0, 1\}$, $p(y=1|x) = \sigma(w^\top x)$.

$$
\ell(w) = \sum_i [y_i \log \sigma(w^\top x_i) + (1-y_i) \log (1 - \sigma(w^\top x_i))]
$$

This is the negative cross-entropy loss (up to sign). MLE = minimize cross-entropy. No closed form; use iteratively reweighted least squares (IRLS) or gradient descent.

---

## 3. Asymptotic theory of MLE

Under regularity conditions (smooth likelihood, identifiable, true $\theta_0$ in interior of parameter space):

**Consistency**: $\hat{\theta}_n \to_p \theta_0$.

**Asymptotic normality**:

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})
$$

where $I(\theta) = -\mathbb{E}_x[\partial^2 \log p(x|\theta)/\partial \theta^2]$ is the Fisher information **per observation**. (Defining it on the joint log-likelihood $\ell = \sum_i \log p(x_i|\theta)$ would scale with $n$ and contradict the formula above; per-observation is correct.)

**Asymptotic efficiency**: variance achieves the Cramér-Rao bound.

**Invariance**: if $\hat{\theta}_{\mathrm{MLE}}$ estimates $\theta$, then $g(\hat{\theta}_{\mathrm{MLE}})$ estimates $g(\theta)$. So the MLE of standard deviation is the square root of MLE of variance.

These properties make MLE the *default* estimator in classical ML — but only asymptotically. Finite-sample MLE can be biased, can overfit, and can be unbounded.

---

## 4. Bayesian setup and MAP

Bayes' theorem applied to a parameter:

$$
p(\theta | x) = \frac{p(x|\theta) p(\theta)}{p(x)}
$$

The **MAP estimate** maximizes the posterior:

$$
\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta p(\theta | x) = \arg\max_\theta \big[\log p(x|\theta) + \log p(\theta)\big]
$$

Equivalent to MLE plus a regularizer that comes from the log-prior. Several important consequences:

### Gaussian prior → Ridge

$p(w) = \mathcal{N}(0, \tau^2 I)$. Then $\log p(w) = -\frac{1}{2\tau^2} \|w\|^2 + \mathrm{const}$.

For linear regression with Gaussian likelihood:

$$
\hat{w}_{\mathrm{MAP}} = \arg\min_w \big[ \tfrac{1}{2\sigma^2}\|y - Xw\|^2 + \tfrac{1}{2\tau^2}\|w\|^2 \big]
$$

Multiply through by $\sigma^2$: ridge regression with $\lambda = \sigma^2/\tau^2$.

### Laplace prior → Lasso

$p(w_j) \propto \exp(-|w_j|/b)$. Log-prior is $-|w|/b$ → $\ell_1$ penalty. Lasso = MAP under Laplace prior.

### Beta prior + Bernoulli → Smoothed estimate

Prior $\theta \sim \mathrm{Beta}(\alpha, \beta)$. Posterior $\theta | x \sim \mathrm{Beta}(\alpha + s, \beta + n - s)$.

MAP: mode of Beta = $\frac{\alpha + s - 1}{\alpha + \beta + n - 2}$ (when $\alpha, \beta > 1$).

Posterior mean: $\frac{\alpha + s}{\alpha + \beta + n}$ — gives a *smoothed* estimate. With $\alpha = \beta = 1$ (uniform prior), posterior mean is $(s+1)/(n+2)$ — Laplace smoothing.

This is exactly what NLP people call add-one smoothing.

---

## 5. Conjugate priors — the catalog

| Likelihood | Conjugate prior | Posterior |
|---|---|---|
| Bernoulli | Beta($\alpha, \beta$) | Beta($\alpha + s, \beta + n - s$) |
| Multinomial | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \mathbf{n}$), where $\mathbf{n} = (n_1, \ldots, n_K)$ are per-category counts |
| Poisson | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) |
| Gaussian (mean, variance known) | Gaussian | Gaussian |
| Gaussian (variance, mean known) | Inverse-Gamma | Inverse-Gamma |
| Gaussian (both unknown) | Normal-Inverse-Gamma (or Normal-Inverse-Wishart) | Same family |
| Exponential | Gamma | Gamma |

Conjugate priors give closed-form posteriors. They also yield clean intuition: hyperparameters of the prior look like *pseudo-counts* — the prior acts as if you'd seen some imaginary data before.

---

## 6. MLE vs MAP vs Bayesian — the spectrum

| Method | Output | Captures uncertainty? | Computational cost | When |
|---|---|---|---|---|
| MLE | Point estimate | No | Cheap (optimization) | Lots of data, no strong prior |
| MAP | Point estimate | No (just the mode) | Cheap (optimization with regularizer) | Want regularization with Bayesian interpretation |
| Bayesian | Full posterior | Yes | Expensive (MCMC/VI) | Need uncertainty, decision-theoretic problems |

Modern deep learning is almost entirely MLE (cross-entropy, MSE) plus MAP (weight decay, dropout). True Bayesian deep learning is a research area (Bayesian NNs, dropout-as-Bayes-approx, deep ensembles for posterior approximation).

---

## 7. Why MLE = minimum cross-entropy = minimum forward KL

For data drawn from true distribution $p^*$:

$$
\arg\max_\theta \mathbb{E}_{x \sim p^*}[\log p(x|\theta)] = \arg\min_\theta \mathbb{E}_{x \sim p^*}[-\log p(x|\theta)]
$$

The right-hand expression is the *cross-entropy* of $p^*$ relative to $p_\theta$. Equivalent:

$$
= \arg\min_\theta \mathrm{KL}(p^* \| p_\theta) + H(p^*)
$$

The $H(p^*)$ term doesn't depend on $\theta$, so MLE = minimize forward KL from $p^*$ to model. This is the "mode-covering" KL — penalizes putting low probability on regions where $p^*$ is high. (Reverse KL would be mode-seeking; that's what variational inference uses.)

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Is MLE always unbiased? | Yes | No — Gaussian variance MLE is biased; many MLEs are biased in finite samples |
| What's the relationship between MAP and regularization? | They're different | MAP = MLE + log-prior; weight decay = Gaussian prior; lasso = Laplace prior |
| What does cross-entropy minimize? | Cross-entropy | Forward KL (with constant offset $H(p^*)$) |
| MLE objective for OLS? | Minimize squared loss | MLE under Gaussian noise → minimizing squared loss |
| Why log-likelihood instead of likelihood? | Same thing | Numerics + sums vs products + matches concavity for many models |
| Why is MLE for variance biased? | It isn't | Plug-in $\bar{x}$ is closer to data than $\mu$, so $\sum(x-\bar{x})^2$ is too small |
| MAP = mean of posterior? | Yes | No, it's the *mode*. Posterior mean is a different point estimator |

---

## 9. Eight most-asked interview questions

1. **Derive MLE for a Gaussian (both parameters).** (Set partials to zero; recognize that variance MLE is biased.)
2. **Show that OLS equals MLE under Gaussian noise.** (Write log-likelihood, drop constants, recognize squared loss.)
3. **Show that ridge equals MAP under Gaussian prior.** (Write log-posterior, recognize $\ell_2$ penalty with $\lambda = \sigma^2/\tau^2$.)
4. **What's the relationship between cross-entropy and MLE?** (CE = negative log-likelihood; minimizing CE = MLE.)
5. **Bayesian smoothing for Bernoulli — derive Laplace's rule of succession.** (Beta(1,1) prior + observed data → posterior mean $(s+1)/(n+2)$.)
6. **What's a conjugate prior and why is it useful?** (Same-family posterior; closed-form updates; pseudo-count intuition.)
7. **What are the asymptotic properties of MLE?** (Consistent, asymp. normal, efficient, invariant.)
8. **MAP vs Bayesian inference?** (MAP gives a point estimate (mode); Bayesian gives full posterior + uncertainty; cost increases.)

---

## 10. Drill plan

- Derive MLE for: Bernoulli, Gaussian, Poisson, multinomial, exponential. 5 minutes each.
- Derive MAP for: linear regression with Gaussian prior (ridge), with Laplace prior (lasso).
- Derive Beta-Bernoulli posterior. Recite mean and mode.
- Recognize: OLS = MLE Gaussian, ridge = MAP Gaussian, lasso = MAP Laplace, cross-entropy = MLE general.
- Practice writing log-likelihoods cleanly without dropping constants until the end.

---

## 11. Further reading

- Murphy, *Machine Learning: A Probabilistic Perspective*, ch. 4–5.
- Bishop, *Pattern Recognition and Machine Learning*, ch. 1, 2.
- Wasserman, *All of Statistics*, ch. 9, 11.
- MacKay, *Information Theory, Inference, and Learning Algorithms* — beautiful Bayesian framing.
