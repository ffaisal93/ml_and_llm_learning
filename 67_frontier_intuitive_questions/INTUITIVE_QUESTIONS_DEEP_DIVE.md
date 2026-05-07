# Frontier Intuitive Probability / Statistics Questions — Deep Dive

> Frontier-lab research-scientist interview-grade reference for the open-ended Bayesian / probabilistic reasoning questions OpenAI / DeepMind / Anthropic ask. Built around 25 worked examples — including the canonical "two distributions, new sample, which one is it from" question — plus the underlying frameworks.

These questions are not "memorize a formula." They test whether you can **frame an open scenario in probabilistic terms, identify the right tool, and reason cleanly to an answer.** Frontier interviewers love them because they reveal depth in seconds: the answer is rarely the point; the *framing* is.

---

## Table of contents

1. The framing checklist
2. Framework 1 — Bayesian classification / hypothesis testing
3. Framework 2 — Maximum likelihood and method of moments
4. Framework 3 — Concentration and tail bounds
5. Framework 4 — KL divergence as test statistic / "distance"
6. Framework 5 — Sequential decision making and bandits
7. Framework 6 — Importance sampling, rejection sampling
8. Framework 7 — Stein's paradox and shrinkage
9. The DeepMind two-distribution question — fully worked
10. 25 worked frontier-lab questions
11. Common follow-up probes
12. Senior-level signals
13. References

---

## 1. The framing checklist

When a probabilistic-scenario question lands, ask yourself in this order:

**(a) What random variables are involved? Define them precisely.**

**(b) What are the candidate hypotheses or models?** (Two distributions? A null and an alternative? A prior over models?)

**(c) Is this a classification, an estimation, or a decision problem?**
- Classification → Bayes' rule, likelihood ratio.
- Estimation → MLE/MAP/posterior mean.
- Decision → expected loss minimization.

**(d) Do I have a prior, or am I purely frequentist?**
- Prior available → Bayesian: posterior $\propto$ likelihood $\times$ prior.
- No prior → frequentist: likelihood ratio test, confidence intervals, p-values.

**(e) What's the loss function or success metric?**
- 0-1 loss → MAP / mode of posterior.
- Squared error → posterior mean.
- Asymmetric loss → tilt the threshold.

**(f) How much data do I have? What's the appropriate level of confidence?**
- Few samples → priors and tail-bound reasoning matter most.
- Many samples → asymptotics, CLT, Fisher information.

**(g) What can I compute and what's only conceptual?** State explicitly when you'd compute numerically vs argue from principle.

This checklist is the difference between a flailing answer and a clean one. State it out loud as you start.

---

## 2. Framework 1 — Bayesian classification

The most-tested framework in frontier-lab probability questions.

### 2.1 The setup

Given hypotheses $H_1, H_2$ (e.g., "sample came from distribution $P$" vs "sample came from $Q$") and observation $x$:

$$
P(H_i \mid x) = \frac{P(x \mid H_i)\, P(H_i)}{\sum_j P(x \mid H_j)\, P(H_j)}.
$$

For the binary case:

$$
P(H_1 \mid x) = \frac{1}{1 + \frac{P(H_2)}{P(H_1)} \cdot \frac{P(x \mid H_2)}{P(x \mid H_1)}}.
$$

Decision rule under 0-1 loss: pick $H_1$ if $P(H_1 \mid x) > P(H_2 \mid x)$, equivalently:

$$
\Lambda(x) = \frac{P(x \mid H_1)}{P(x \mid H_2)} > \frac{P(H_2)}{P(H_1)}.
$$

The **likelihood ratio** $\Lambda(x)$ vs the **prior odds** ratio. The Neyman-Pearson lemma says: for a fixed false-positive rate, the likelihood ratio test is the most powerful test.

### 2.2 With multiple samples

For i.i.d. samples $x_1, ..., x_n$:

$$
\Lambda_n = \prod_{i=1}^n \frac{P(x_i \mid H_1)}{P(x_i \mid H_2)}.
$$

Better in log-space:

$$
\log \Lambda_n = \sum_{i=1}^n \log \frac{P(x_i \mid H_1)}{P(x_i \mid H_2)}.
$$

The expected log-likelihood ratio under $H_1$ is the **KL divergence**:

$$
\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right] = \mathrm{KL}(P \,\|\, Q) \ge 0.
$$

This is why two distributions with high KL are easy to distinguish; close-to-zero KL means hard.

### 2.3 Sample complexity

How many samples do you need to distinguish $H_1$ from $H_2$ with confidence $1-\delta$?

By the central limit theorem, the log-likelihood ratio's mean grows linearly in $n$ (slope = KL or $-$KL depending on the truth) and its standard deviation grows like $\sqrt{n}$. So discriminability scales as $\sqrt{n}$. Specifically:

$$
n^* \approx \frac{(z_\alpha + z_\beta)^2 \cdot \mathrm{Var}_{H_1}\!\big[\log \Lambda(x)\big]}{\big(\mathrm{KL}(P \| Q)\big)^2}
$$

(roughly; the precise formula depends on test type).

**Memorize:** distinguishing two distributions takes $O(1/\mathrm{KL}^2)$ samples.

### 2.4 Connection to Chernoff information

The exponent of the optimal Bayes error rate is the *Chernoff information*:

$$
C(P, Q) = -\min_{0 \le \lambda \le 1} \log \int p(x)^\lambda q(x)^{1-\lambda} \, dx.
$$

The optimal classifier's error rate decays as $e^{-n C(P, Q)}$. KL is a worse upper bound; Chernoff is tight.

---

## 3. Framework 2 — Maximum likelihood and method of moments

### 3.1 MLE

$\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_i p(x_i \mid \theta)$.

Properties:
- **Consistent.** $\hat\theta \to \theta_0$ as $n \to \infty$.
- **Asymptotically normal.** $\sqrt{n}(\hat\theta - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})$ where $I$ is Fisher information.
- **Asymptotically efficient.** Reaches the Cramér-Rao lower bound.
- **Sometimes biased in finite samples.** Common interview gotcha.

### 3.2 MAP

$\hat\theta_{\text{MAP}} = \arg\max_\theta \prod_i p(x_i \mid \theta) \cdot p(\theta)$.

Reduces to MLE under uniform prior. Useful when data is sparse and prior is informative.

### 3.3 Method of moments

Solve $\hat{\mu}_k(X) = \mu_k(\theta)$ for the first few moments.

- Often less efficient than MLE.
- Sometimes more robust.
- Easier when likelihood is intractable.

### 3.4 Posterior mean vs MAP

For squared-error loss, posterior mean is optimal. For 0-1 loss on continuous $\theta$, the MAP is *almost never* the optimal Bayes estimator (zero-set issue) but is a reasonable approximation when posterior is unimodal.

---

## 4. Framework 3 — Concentration and tail bounds

For "with high probability X is small" questions.

### 4.1 Markov

$P(X \ge a) \le \mathbb{E}[X] / a$ for non-negative $X$. Crude but always valid.

### 4.2 Chebyshev

$P(|X - \mu| \ge k\sigma) \le 1/k^2$. Two-sided, no distributional assumption.

### 4.3 Hoeffding

For bounded i.i.d. $X_i \in [a, b]$:

$$
P\!\left(\bigg|\frac{1}{n}\sum X_i - \mu\bigg| \ge t\right) \le 2 \exp\!\left(\frac{-2 n t^2}{(b-a)^2}\right).
$$

Sub-Gaussian tail; the workhorse for many concentration arguments.

### 4.4 Bernstein

Sharper than Hoeffding when variance is known and small. Gives sub-exponential tail.

### 4.5 Chernoff

Generalizes via moment generating function. Tightest bound from MGF.

### 4.6 When to reach for which

- "I know $X$ is bounded" → Hoeffding.
- "I know the variance" → Chebyshev (loose) / Bernstein (tight).
- "I want a numerical CI without assumption" → CLT for $n \gtrsim 30$.
- "I have a rate / count" → Poisson / Chernoff.

### 4.7 Common gotcha

Hoeffding has the *2* in the numerator; some forms have it in the denominator. Memorize the version with $-2nt^2/(b-a)^2$ in the exponent.

---

## 5. Framework 4 — KL divergence

Already invoked in §2. Three uses:

### 5.1 As "distance" (asymmetric)

$$
\mathrm{KL}(P \,\|\, Q) = \mathbb{E}_P\!\left[\log \frac{P(x)}{Q(x)}\right] \ge 0,
$$

$=0$ iff $P=Q$. Asymmetric, doesn't satisfy triangle inequality. Used as objective in distillation, alignment, regularization.

### 5.2 As Bayes-error exponent

The error rate of the optimal classifier decays at exponent $C(P, Q)$ (Chernoff), upper-bounded by KL.

### 5.3 As coding excess

If you encode $P$-distributed data with a $Q$-optimized code, expected excess bits per symbol = $\mathrm{KL}(P \,\|\, Q)$.

### 5.4 KL between two Gaussians

$$
\mathrm{KL}(\mathcal{N}(\mu_1, \Sigma_1) \,\|\, \mathcal{N}(\mu_2, \Sigma_2)) = \tfrac{1}{2}\!\left[\log \frac{|\Sigma_2|}{|\Sigma_1|} - d + \mathrm{tr}(\Sigma_2^{-1} \Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)\right].
$$

For 1D: $\mathrm{KL}(\mathcal{N}(\mu_1, \sigma^2) \| \mathcal{N}(\mu_2, \sigma^2)) = (\mu_1 - \mu_2)^2 / (2\sigma^2)$.

So distinguishing means $\mu_1, \mu_2$ at known variance $\sigma^2$ takes $n^* \propto (\mu_1 - \mu_2)^{-2} \sigma^2$ samples — a classical result.

---

## 6. Framework 5 — Sequential decision making and bandits

For "design a strategy" questions.

### 6.1 Multi-armed bandit

$K$ arms, unknown reward distributions, sequential pulls, regret = best-arm-reward minus chosen-arm-reward summed.

- **UCB:** pick arm with highest $\hat\mu + \sqrt{2 \log t / N_a}$.
- **Thompson sampling:** sample $\theta_a$ from posterior, pull $\arg\max$.
- **$\epsilon$-greedy:** simplest; doesn't achieve $O(\log T)$ regret in general.

Optimal regret is $O(\log T)$.

### 6.2 Best-arm identification

Different objective: minimize samples to confidently identify the best arm with prob $1-\delta$. Different optimal algorithms (LUCB, Track-and-Stop).

### 6.3 Connections to RL

Bandit = stateless RL. Many ideas (exploration, regret) generalize.

---

## 7. Framework 6 — Importance sampling, rejection sampling

For "estimate this expectation under a hard distribution" questions.

### 7.1 Importance sampling

To estimate $\mathbb{E}_P[f(X)]$ when $P$ is hard to sample but $Q$ is easy:

$$
\mathbb{E}_P[f(X)] = \mathbb{E}_Q\!\left[\frac{P(X)}{Q(X)} f(X)\right].
$$

Variance of the estimator is small if $Q$ is well-matched to $|f| P$. Bad if $P$ has support where $Q$ has near-zero density (heavy tails of $P/Q$).

### 7.2 Rejection sampling

Sample from $Q$, accept with probability $P(x) / (M \cdot Q(x))$ where $M = \sup P/Q$. Acceptance rate = $1/M$. Inefficient if $M$ is large.

### 7.3 In RLHF / alignment

Importance sampling is exactly how PPO computes policy gradients off-policy. The ratio $r(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ is the importance weight.

---

## 8. Framework 7 — Stein's paradox and shrinkage

A classic "intuitive" topic.

### 8.1 The result

Estimate $K \ge 3$ Gaussian means $\mu_1, ..., \mu_K$ from one observation each $x_k \sim \mathcal{N}(\mu_k, 1)$. The James-Stein estimator:

$$
\hat\mu_k^{\text{JS}} = \left(1 - \frac{K - 2}{\sum_j x_j^2}\right) x_k
$$

has *strictly lower MSE* than the obvious $\hat\mu_k = x_k$, regardless of the truth, when $K \ge 3$.

### 8.2 The intuition

The means need not be related, but *averaging across them* leverages the fact that any sample is far from the origin "by chance" with high probability in high dimensions, so shrinking toward the origin is uniformly better.

### 8.3 Connection to ML

Regularization, weight decay, and Bayesian priors are all flavors of shrinkage. The bias-variance tradeoff lives here.

---

## 9. The DeepMind two-distribution question — fully worked

The user's actual interview question:

> "You have two arrays of numbers from two distributions. A new number comes. Describe how you determine from which distribution it came from."

This is the canonical two-class classification with empirical density estimation. A clean answer walks through:

### 9.1 Set up

- **Data.** Two arrays $A = \{a_1, ..., a_n\}$ from distribution $P$, $B = \{b_1, ..., b_m\}$ from $Q$.
- **Observation.** New value $x$.
- **Question.** Decide which distribution $x$ came from.

### 9.2 Bayes formulation

$$
P(\text{from }P \mid x) = \frac{p(x)\, \pi_P}{p(x)\, \pi_P + q(x)\, \pi_Q}
$$

where $\pi_P, \pi_Q$ are priors (typically $n / (n+m)$ and $m / (n+m)$ if both arrays are samples in proportion to base rates).

Decision: classify as $P$ iff $P(\text{from }P \mid x) > 0.5$ (under 0-1 loss with equal class weights), equivalently:

$$
\Lambda(x) = \frac{p(x)}{q(x)} > \frac{\pi_Q}{\pi_P}.
$$

### 9.3 Estimating $p$ and $q$

The interesting depth — this is where the interviewer probes.

**Option 1: parametric.** Assume both are Gaussian. Estimate $\hat\mu_P, \hat\sigma_P$ from $A$ via MLE; same for $Q$. Plug into Gaussian density. Fast, low-variance, biased if assumption is wrong.

**Option 2: non-parametric (KDE).** Kernel density estimate from $A$ and from $B$. Bandwidth chosen via cross-validation or Silverman's rule. More flexible; needs more data.

**Option 3: empirical CDF + smoothing.** Compute empirical CDFs and use a smoothing kernel to estimate density. Variant of KDE.

**Option 4: discriminative.** Don't estimate $p, q$ separately; train a classifier (logistic regression, neural net) directly on $(A, 0)$ and $(B, 1)$. Output the predicted probability for $x$. Often better than density estimation (Hastie/Tibshirani: discriminative > generative when modeling assumptions are wrong).

### 9.4 Diagnostics and follow-up answers

- **Quantify confidence.** $|\log \Lambda(x)|$ measures evidence in nats. Convert to posterior probability via the Bayes formula above.
- **Sample complexity.** How many samples do you need from each side? Roughly $O(1 / \mathrm{KL}^2)$ for distinguishability plus $O(1 / \epsilon^2)$ for density estimation accuracy. KL is between $P$ and $Q$.
- **What if the new sample is in a region with no training data on either side?** The likelihoods are both ~0 estimates; the answer is "I don't know" — and a robust system flags it as out-of-distribution. This is where you mention OOD detection (Mahalanobis distance, energy score, ensemble disagreement).
- **What if the priors are unknown?** You can still compute the likelihood ratio; the prior is a multiplicative factor in the threshold.
- **What if A and B are huge but a new sample is one number?** Fast lookup: nearest-neighbor density estimation in $O(\log n)$ with a sorted array.
- **What if $P$ and $Q$ have heavy overlap?** Even the optimal Bayes classifier will have high error. Quantify via Bayes error rate, $\int \min(p, q)$.
- **What loss are you optimizing?** 0-1 loss → MAP. Asymmetric (false-A worse than false-B) → shift threshold. Multi-class extension is straightforward.
- **What if A and B are *not* independent of $x$ (covariate shift)?** Doesn't apply if $x$ is just a sample; applies if there's structured dependence (time-series, locality).

### 9.5 The 90-second oral answer

> This is binary classification: hypothesis $H_P$ that $x$ came from distribution $P$ (with array $A$ as samples) vs $H_Q$ from $Q$ (array $B$). Bayes-optimal under 0-1 loss is the likelihood ratio test: classify as $P$ if $p(x)/q(x) > \pi_Q/\pi_P$, where $\pi$'s are class priors estimated as $n/(n+m)$ and $m/(n+m)$.
>
> The interesting part is estimating $p$ and $q$. Three approaches: parametric (assume Gaussian, fit MLE — fast, biased if wrong), non-parametric KDE (more flexible, needs more data, bandwidth via cross-validation), or discriminative (train a logistic regression / neural net on combined labeled data — often better than density estimation, per the discriminative-vs-generative literature).
>
> I'd quantify confidence by $|\log \Lambda(x)|$; flag out-of-distribution if both $p(x)$ and $q(x)$ are very low; and note that sample complexity for discriminability scales as $1/\mathrm{KL}(P \| Q)^2$, so if the two distributions are very close, you need a lot of samples to be confident regardless of method.

This answer, in 90 seconds, hits: framing, Bayes rule, prior, likelihood ratio, three estimation strategies, OOD flagging, sample complexity, and KL connection. **That's a frontier-lab answer.**

---

## 10. 25 worked frontier-lab questions

Brief but enough to seed the thought.

### Q1. "Two arrays from two distributions, classify a new sample." (above, §9)

### Q2. "How many coin flips to confirm a coin is biased toward heads?"

Hypothesis test. $H_0: p = 0.5$ vs $H_1: p > 0.5$. Use the binomial / normal approximation. For detecting $p = 0.6$ at $\alpha=\beta=0.05$:

$$
n^* \approx \frac{(z_{0.05} + z_{0.05})^2 \cdot p(1-p)}{(p - 0.5)^2} = \frac{(1.645 + 1.645)^2 \cdot 0.24}{0.01} \approx 260.
$$

KL framing: $\mathrm{KL}(\text{Bern}(0.6) \| \text{Bern}(0.5)) \approx 0.0204$, so $n \sim 1/\mathrm{KL}^2 \sim 2400$ for distinguishability — order of magnitude consistency check.

### Q3. "Estimate the mean of a normal distribution given 3 samples. What's your confidence interval?"

Use $t$-distribution (small $n$): $\bar{x} \pm t_{n-1, \alpha/2} \cdot s / \sqrt{n}$. With $n=3$, $t_{2, 0.025} \approx 4.30$ — wide CI.

### Q4. "You sample $X_1, ..., X_n$ i.i.d. from a distribution with bounded variance. How concentrated is the sample mean?"

Chebyshev: $P(|\bar{X} - \mu| \ge t) \le \sigma^2 / (n t^2)$. Or CLT for $n \gtrsim 30$. Or Hoeffding if bounded support.

### Q5. "What's the variance of the sample variance for a Gaussian?"

$\mathrm{Var}(s^2) = 2\sigma^4 / (n-1)$.

### Q6. "Why can't you just use empirical CDF for likelihood?"

Empirical CDF gives $\hat F(x)$, but the density is $\hat F'(x)$ — a sum of delta functions at observations. Useless for new points. Need smoothing.

### Q7. "How would you test if a sample came from a normal distribution?"

Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, Q-Q plots, Jarque-Bera.

### Q8. "Two samples — same distribution test?"

Two-sample Kolmogorov-Smirnov. Or Mann-Whitney. Or $t$-test if assuming Gaussian. Or permutation test (most flexible).

### Q9. "Estimate KL between two empirical distributions."

KDE both, then numerically integrate; or use the $k$-NN-based estimator (Pérez-Cruz); or train a discriminator and use the bound from the GAN/$f$-divergence literature.

### Q10. "If KL between two distributions is $\epsilon$, how easy to discriminate?"

Sample complexity $\sim 1/\epsilon^2$. Bayes error rate decays as $e^{-n C(P, Q)}$ where $C \le \mathrm{KL}$.

### Q11. "Coin flip game: I flip; you guess; I pay you 2× your bet if right; how much do you bet?"

Kelly criterion: $f^* = (b p - q) / b$ where $b$ is odds ratio, $p$ is win prob, $q = 1-p$. With $b = 2, p = 0.5$: $f^* = (1 - 0.5)/2 = 0.25$ — bet 25% of bankroll.

### Q12. "$X$ uniform on [0,1]. What's $\mathbb{E}[\max(X, 0.5)]$?"

Split: with prob 0.5 max=0.5; otherwise max=$X$ with $X \in [0.5, 1]$, so $\mathbb{E}[X | X > 0.5] = 0.75$, with prob 0.5. Total: $0.5 \cdot 0.5 + 0.5 \cdot 0.75 = 0.625$.

### Q13. "Two coins; one fair, one always heads. You pick one and flip 10 times, all heads. What's $P(\text{fair})$?"

Bayes: $P(\text{fair} \mid 10H) = P(10H | \text{fair}) \cdot 0.5 / [P(10H | \text{fair}) \cdot 0.5 + P(10H | \text{biased}) \cdot 0.5] = (1/1024) \cdot 0.5 / [(1/1024) \cdot 0.5 + 1 \cdot 0.5] = 1/1025 \approx 0.001$. Almost certainly biased.

### Q14. "$X, Y$ i.i.d. uniform on $[0, 1]$. What's $\mathbb{E}[\max(X, Y)]$?"

CDF of max: $F_M(z) = z^2$. PDF: $2z$. $\mathbb{E}[M] = \int_0^1 z \cdot 2z \, dz = 2/3$.

### Q15. "How many people for 50% birthday collision probability?"

Birthday problem. $P(\text{no collision among }n) = \prod_{k=0}^{n-1} (365 - k)/365 \approx e^{-n(n-1)/(2 \cdot 365)}$. Set $\approx 0.5$ → $n \approx 23$.

### Q16. "Three doors, prize behind one, you pick door 1, host opens door 3 (no prize). Switch?"

Monty Hall. Yes — prob of prize behind door 2 is 2/3 (host's action carries information).

### Q17. "Why does Monty Hall break if host acts randomly?"

Then host's action carries less information; conditional probabilities change. You should still consider info-theoretic value of the action.

### Q18. "Estimate $\pi$ via random sampling."

Sample $(x, y) \sim U[0,1]^2$. Fraction in unit quarter circle is $\pi/4$. Multiply by 4. Variance $\propto 1/n$.

### Q19. "$X \sim \text{Exp}(\lambda)$. What's $P(X > a + b | X > a)$?"

Memorylessness: $P(X > a + b | X > a) = P(X > b)$.

### Q20. "Sum of two i.i.d. exponentials — what distribution?"

Gamma(2, $\lambda$). Sum of $k$ i.i.d. Exp($\lambda$) is Gamma($k, \lambda$).

### Q21. "Why is the median more robust than the mean?"

Median has 50% breakdown point; mean has 0%. One outlier moves the mean unboundedly, doesn't move the median.

### Q22. "Detect a change-point in a stream of values from a known distribution."

CUSUM, GLR (generalized likelihood ratio), Bayesian online change-point detection. Sequential framework — every time, compute likelihood ratio under "no change" vs "change at $t$" hypothesis; if exceeds threshold, declare change.

### Q23. "Estimate the size of a population with unique IDs from a single sample."

German tank problem. If max observed = $m$ from $n$ samples: MLE estimate $= m$, but minimum-variance unbiased estimator $= m \cdot (n+1)/n - 1 = m + m/n - 1$.

### Q24. "Two-sample mean test — but the variances differ."

Welch's $t$-test. Adjusted degrees of freedom. Non-parametric: Mann-Whitney.

### Q25. "AB test: significant at $p=0.04$, $n=10000$. Should you ship?"

Discuss: practical significance vs statistical, multiple testing, peeking, effect size, business cost of being wrong. Senior signal: don't take the p-value at face value.

---

## 11. Common follow-up probes

Frontier interviewers always probe one or two of these after your initial answer:

- **"What if your prior is wrong?"** → Bayesian sensitivity analysis. Posterior dominated by data when $n$ is large; dominated by prior when $n$ is small.
- **"What's the variance of your estimator?"** → Cramér-Rao, asymptotic variance via Fisher info.
- **"What if the distributions overlap heavily?"** → Bayes error floor; quantify via $\int \min(p, q)$.
- **"What's your sample complexity?"** → Concentration inequality + KL/Chernoff.
- **"What if you don't know the parametric family?"** → Non-parametric (KDE, $k$-NN) or discriminative.
- **"What's the asymmetric-loss version?"** → Shift threshold; minimize expected loss.
- **"How would this fail in production?"** → distribution shift, OOD, label noise, data drift.
- **"Compare your method with X."** → Bias-variance tradeoff; sample efficiency.
- **"What's the connection to information theory / KL / Fisher info?"** → Reach for the unifying theorem.
- **"Why are you confident in your estimator?"** → CI, bootstrap, robustness.

---

## 12. Senior-level signals

- **You start with the framing checklist.** Don't jump to a formula.
- **You name the framework** (Bayes / MLE / Concentration / Bandit / Importance / Stein) explicitly.
- **You quantify confidence** — $\log \Lambda$, posterior, CI, sample complexity in $1/\mathrm{KL}^2$.
- **You discuss assumptions** and what fails when they're wrong.
- **You name the connection to information theory** (KL, Fisher, Chernoff).
- **You think about OOD / failure modes**, not just the happy path.
- **You distinguish frequentist vs Bayesian** when relevant.
- **You mention the production-grade variant** (online estimation, drift detection, hypothesis testing under multiple comparisons).
- **You don't over-claim**. "Optimal under 0-1 loss with these priors" — not "optimal."
- **You can pivot from the analytical answer to a programmatic one** if asked.

---

## 13. References

- Casella & Berger, *Statistical Inference*. The standard reference.
- Cover & Thomas, *Elements of Information Theory*. KL, Fisher, Chernoff.
- Wasserman, *All of Statistics*. Concise and broad.
- Bishop, *Pattern Recognition and Machine Learning*. Bayesian flavor.
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*. Discriminative-vs-generative debate.
- Robert, *The Bayesian Choice*. Decision theory.
- Lattimore & Szepesvári, *Bandit Algorithms*. Sequential decision making.
- Berger, *Statistical Decision Theory and Bayesian Analysis*. Stein's paradox, shrinkage.
- Lehmann & Romano, *Testing Statistical Hypotheses*. Frequentist hypothesis testing.

---

## How to use this chapter

1. Read §1 (framing checklist) until automatic.
2. Memorize the seven frameworks (§2-§8) at a level where you can name them and the canonical formula on demand.
3. Drill §10 — 25 worked questions — until each has a 30-second answer.
4. Memorize §9 (the DeepMind two-distribution question) verbatim as your "model answer" template.
5. Pair with `INTERVIEW_GRILL.md` for active recall.
6. Practice out loud — these are oral exams in real interviews.

Single sentence to remember: **frame as Bayesian classification or MLE / decision / concentration, name the framework explicitly, quantify with KL or Fisher or Chernoff, discuss assumptions and OOD, and end with sample complexity.**
