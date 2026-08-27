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

> **Saying it out loud.** The thing being tested here isn't whether I know Bayes' rule, it's whether I can turn a vague scenario into a well-posed problem before I start computing. So out loud I'd go: here are the random variables, here are the competing hypotheses, is this classification or estimation or a decision, do I have a prior or am I going frequentist, what's the loss function, and how much data do I actually have. Saying that sequence aloud buys me thinking time and it *is* most of the answer. And I'd be explicit about the last item on the list — what I can actually compute versus what I'm arguing from principle — because the honest version of these answers almost always has a boundary in it, and naming that boundary reads as confidence, not evasion.

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

**Memorize:** distinguishing two *fixed* distributions takes $O(1/\mathrm{KL})$ samples.

> **Correction — read this before memorizing anything.** An earlier version of this section said $O(1/\mathrm{KL}^2)$. That is wrong, and the error is worth understanding because it is easy to make out loud. The Chernoff–Stein lemma gives the type-II error decaying as $e^{-n\,\mathrm{KL}(P\|Q)}$ at fixed type-I error, so $n^\ast \approx \log(1/\beta)/\mathrm{KL}(P\|Q)$ — linear in $1/\mathrm{KL}$. The formula above is not itself wrong, but you have to notice that for *close* distributions $\mathrm{Var}_{H_1}[\log \Lambda] \approx 2\,\mathrm{KL}$, so the $\mathrm{Var}/\mathrm{KL}^2$ collapses to $O(1/\mathrm{KL})$ too. The internal consistency check is §5.4: separating two Gaussian means needs $n^\ast \propto \sigma^2/(\mu_1-\mu_2)^2$, and $\mathrm{KL} = (\mu_1-\mu_2)^2/(2\sigma^2)$, so $n^\ast \propto 1/\mathrm{KL}$. If a quoted rate contradicts a worked example in the same document, trust the worked example.

> **Saying it out loud.** The intuition I'd give first, before any formula: each new sample adds a roughly constant number of nats of evidence, and that constant is the KL divergence — so the log-likelihood ratio drifts linearly in n at rate KL, while its noise only grows like the square root of n. Signal beats noise, so you need about one over KL samples to be confident, times a factor for how confident you want to be. That's Chernoff–Stein. I'd flag that I've seen this quoted as one over KL squared and it isn't — you can check it against the Gaussian case in about ten seconds, and I'd rather derive the sanity check on the spot than recite a rate I'm not sure of. That's the honest move here: when two things I half-remember disagree, say so and check one against the other rather than picking the confident-sounding one.

### 2.4 Connection to Chernoff information

The exponent of the optimal Bayes error rate is the *Chernoff information*:

$$
C(P, Q) = -\min_{0 \le \lambda \le 1} \log \int p(x)^\lambda q(x)^{1-\lambda} \, dx.
$$

The optimal classifier's error rate decays as $e^{-n C(P, Q)}$. KL is a worse upper bound; Chernoff is tight.

> **Saying it out loud.** Almost every "which one did this come from" question is the same object underneath: compare the likelihood ratio to the prior odds. That's it — the probability of the data under hypothesis one over its probability under hypothesis two, and you believe hypothesis one when that ratio beats how much more likely hypothesis two was to begin with. Neyman-Pearson says this test is optimal in a precise sense, so I'm not being clever, I'm using the known answer. With multiple samples you add log-likelihood ratios, which is why evidence is naturally measured in nats and why KL divergence shows up as the average rate at which evidence accumulates. The part I'd be careful not to overclaim: it's optimal *given* the two densities, and in every real version of this question I don't have the densities, I'm estimating them — so the hard part isn't the decision rule, it's everything upstream of it.

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

> **Saying it out loud.** MLE is the default because asymptotically you can't do better — it's consistent, asymptotically normal, and hits the Cramér-Rao bound, so if I have plenty of data and I trust the model family, that's the answer. MAP is the same thing with a prior, which matters exactly when data is thin. Method of moments is the fallback when the likelihood is intractable, and it's often more robust and less efficient. The caveats I'd volunteer rather than wait to be asked: MLE is only asymptotically unbiased, so at small n it can be visibly wrong — sample variance with the n divisor is the classic — and every one of those optimality properties is conditional on the model family being right. If the family is wrong, MLE efficiently converges to the wrong thing, which is worse than being inefficient.

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

> **Saying it out loud.** The selection rule is just "what do I actually know". Bounded variables and nothing else — Hoeffding. I know the variance and it's small relative to the range — Bernstein, which is sharper. I know almost nothing — Chebyshev, which is loose but never wrong. Thirty-odd samples and I want a practical interval rather than a guarantee — CLT. The distinction I'd say out loud is that concentration bounds and the CLT answer different questions: the bounds are finite-sample guarantees that hold always and are conservative, the CLT is an approximation that's usually tighter and has no guarantee attached. Which one I want depends on whether I'm making a claim or making a decision.

### 4.7 Common gotcha

Hoeffding has the *2* in the numerator; some forms have it in the denominator. Memorize the version with $-2nt^2/(b-a)^2$ in the exponent.

> **Saying it out loud.** Concentration is the toolkit for "how sure can I be that my average is near the truth", and the ladder goes from assuming almost nothing to assuming quite a lot. Markov needs only non-negativity and is nearly useless. Chebyshev needs a variance and gives you the one-over-k-squared bound. Hoeffding needs bounded support and gives you a Gaussian-shaped tail, which is why it's the workhorse. Bernstein sharpens that when the variance is small compared to the range. The framing that scores is stating what each one buys and what it costs — every step up that ladder is a stronger assumption in exchange for a tighter bound, and the interesting question in any real problem is which assumption you're actually willing to defend.

---

## 5. Framework 4 — KL divergence

Already invoked in §2. Three uses:

### 5.1 As "distance" (asymmetric)

$$
\mathrm{KL}(P \,\|\, Q) = \mathbb{E}_P\!\left[\log \frac{P(x)}{Q(x)}\right] \ge 0,
$$

$=0$ iff $P=Q$. Asymmetric, doesn't satisfy triangle inequality. Used as objective in distillation, alignment, regularization.

> **Saying it out loud.** KL is not a distance and I'd say that before using it as one. It's non-negative and zero only when the distributions match, but it's asymmetric and it violates the triangle inequality, so calling it a distance will get you into trouble. What it actually is: the expected log-likelihood ratio, which is the average evidence per sample — and equivalently the extra bits you pay for encoding data from P using a code built for Q. The asymmetry is not a defect, it's the content: KL(P||Q) blows up when Q puts near-zero mass where P has mass, which is exactly why forward and reverse KL behave so differently as training objectives.

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

> **Saying it out loud.** KL is the same quantity wearing three hats, and being able to switch between them is what makes it useful. It's the expected log-likelihood ratio, so it's the rate at which evidence accumulates per sample. It's the excess bits from coding P-data with a Q-optimised code. And it's the exponent controlling how fast a classifier's error decays — though strictly the tight exponent is Chernoff information and KL is a bound. The concrete fact worth carrying is the one-dimensional Gaussian case: KL equals the squared mean gap over twice the variance, which means separating two Gaussians costs samples proportional to variance over squared mean gap. I'd also be honest that KL is the wrong tool when the supports don't overlap — it's infinite, which is uninformative, and that's precisely why Wasserstein exists.

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

> **Saying it out loud.** Bandits are what you reach for when the question is "design a strategy" rather than "compute a quantity", and the core tension is exploration against exploitation — every pull you spend learning is a pull you didn't spend earning. UCB handles this by being optimistic in the face of uncertainty: rank arms by their upper confidence bound, so an arm you know little about gets tried. Thompson sampling does it by sampling from the posterior and acting greedily on the sample, which is simpler and often works better in practice. The number to name is that optimal regret is logarithmic in the horizon, not constant — you never stop paying for exploration, you just pay less and less. And I'd flag the distinction between minimising regret and identifying the best arm, because they're different objectives with genuinely different optimal algorithms, and interviewers ask which one you're solving.

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

> **Saying it out loud.** Both of these answer "I need expectations under a distribution I can't sample from". Importance sampling reweights: sample from something easy, weight each sample by the ratio of the target density to the proposal density, and the expectation comes out right. Rejection sampling instead throws samples away until what survives is distributed correctly. The failure mode is the same for both and it's worth naming precisely — if the proposal has thin tails where the target has mass, the weights blow up and your estimator has enormous, possibly infinite variance, and the vicious part is that it will look fine until it suddenly doesn't. Rejection sampling shows the same problem as a terrible acceptance rate, which at least fails visibly. The connection to say if you're in an RL interview: PPO's ratio of new policy to old policy is exactly an importance weight, and the clipping is exactly the variance control.

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

> **Saying it out loud.** Stein's result is genuinely counterintuitive and I'd say so rather than pretend it's obvious. Estimate three or more unrelated Gaussian means, and shrinking every estimate toward a common point beats using each observation on its own — in total squared error, always, no matter what the true means are. The means don't have to be related; that's the shocking part. The intuition that actually helps is dimensional: in high dimensions, the vector of observations is almost surely further from the origin than the true mean vector, because the noise adds length in every coordinate, so pulling everything in corrects a systematic overshoot. I'd be careful about what it does *not* say — it improves total error across all coordinates while possibly making any individual one worse, so it's a statement about aggregate loss, not about each estimate. And every regulariser you've ever used, weight decay included, is a version of it.

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

> **Saying it out loud.** This is where the question actually lives, and I'd say up front that there's no single right answer — it depends on how much data I have and how much I trust a parametric family. If both arrays look unimodal and I have a few dozen points, fit Gaussians and get the likelihood ratio in closed form; fast, low variance, and badly wrong if the shape is wrong. With more data, kernel density estimation is more flexible, and the honest caveat is that everything then depends on the bandwidth, which I'd pick by cross-validation and would not pretend is a solved choice. And the third option is to skip density estimation entirely — label array A as class zero, array B as class one, train a classifier, and read off its predicted probability, which is often better because you only need the *ratio*, and estimating a ratio is easier than estimating two densities. If forced to pick blind, I'd go discriminative, and I'd say that's a default rather than a proof.

### 9.4 Diagnostics and follow-up answers

- **Quantify confidence.** $|\log \Lambda(x)|$ measures evidence in nats. Convert to posterior probability via the Bayes formula above.
- **Sample complexity.** How many samples do you need from each side? Roughly $O(1 / \mathrm{KL})$ for distinguishability plus $O(1 / \epsilon^2)$ for density estimation accuracy. KL is between $P$ and $Q$.
- **What if the new sample is in a region with no training data on either side?** The likelihoods are both ~0 estimates; the answer is "I don't know" — and a robust system flags it as out-of-distribution. This is where you mention OOD detection (Mahalanobis distance, energy score, ensemble disagreement).
- **What if the priors are unknown?** You can still compute the likelihood ratio; the prior is a multiplicative factor in the threshold.
- **What if A and B are huge but a new sample is one number?** Fast lookup: nearest-neighbor density estimation in $O(\log n)$ with a sorted array.
- **What if $P$ and $Q$ have heavy overlap?** Even the optimal Bayes classifier will have high error. Quantify via Bayes error rate, $\int \min(p, q)$.
- **What loss are you optimizing?** 0-1 loss → MAP. Asymmetric (false-A worse than false-B) → shift threshold. Multi-class extension is straightforward.
- **What if A and B are *not* independent of $x$ (covariate shift)?** Doesn't apply if $x$ is just a sample; applies if there's structured dependence (time-series, locality).

> **Saying it out loud.** The follow-ups are where the interview is actually decided, and most of them have the same shape: what happens when the setup breaks. If the new point sits where neither array has data, both density estimates are near zero and the ratio is numerically meaningless — the correct answer is "I don't know, flag it as out-of-distribution", and saying that is worth more than producing a number. If the two distributions overlap heavily, even the optimal classifier has an error floor, which you can quote as the integral of the pointwise minimum of the two densities — so no method fixes it and more data doesn't help. If the priors are unknown, the likelihood ratio is unchanged and only the threshold moves. The senior move across all of these is refusing to give a confident answer where the data doesn't support one, and naming *what* would make you confident.

### 9.5 The 90-second oral answer

> This is binary classification: hypothesis $H_P$ that $x$ came from distribution $P$ (with array $A$ as samples) vs $H_Q$ from $Q$ (array $B$). Bayes-optimal under 0-1 loss is the likelihood ratio test: classify as $P$ if $p(x)/q(x) > \pi_Q/\pi_P$, where $\pi$'s are class priors estimated as $n/(n+m)$ and $m/(n+m)$.
>
> The interesting part is estimating $p$ and $q$. Three approaches: parametric (assume Gaussian, fit MLE — fast, biased if wrong), non-parametric KDE (more flexible, needs more data, bandwidth via cross-validation), or discriminative (train a logistic regression / neural net on combined labeled data — often better than density estimation, per the discriminative-vs-generative literature).
>
> I'd quantify confidence by $|\log \Lambda(x)|$; flag out-of-distribution if both $p(x)$ and $q(x)$ are very low; and note that sample complexity for discriminability scales as $1/\mathrm{KL}(P \| Q)$, so if the two distributions are very close, you need a lot of samples to be confident regardless of method.

This answer, in 90 seconds, hits: framing, Bayes rule, prior, likelihood ratio, three estimation strategies, OOD flagging, sample complexity, and KL connection. **That's a frontier-lab answer.**

> **Saying it out loud.** So the question is: two arrays, a new number, which array did it come from. I'd frame it as binary classification and say the optimal rule under zero-one loss is the likelihood ratio test — the density of the new point under P over its density under Q, compared against the ratio of the class priors, which I'd estimate from the array sizes. That part is textbook and takes fifteen seconds. The real content is that I don't have the densities, so I have to estimate them, and I'd lay out three options with their tradeoffs: parametric is fast and biased if the family is wrong, kernel density estimation is flexible and bandwidth-sensitive, and training a discriminative classifier often wins because it estimates the ratio directly instead of estimating two densities and dividing. Then I'd volunteer the failure modes before being asked — flag out-of-distribution points where both densities are tiny, note the Bayes error floor if the distributions overlap, and give the sample complexity as roughly one over the KL divergence. The reason that scores is that I've said what I know, what I've assumed, and where the answer stops being reliable.

---

## 10. 25 worked frontier-lab questions

Brief but enough to seed the thought.

### Q1. "Two arrays from two distributions, classify a new sample." (above, §9)

### Q2. "How many coin flips to confirm a coin is biased toward heads?"

Hypothesis test. $H_0: p = 0.5$ vs $H_1: p > 0.5$. Use the binomial / normal approximation. For detecting $p = 0.6$ at $\alpha=\beta=0.05$:

$$
n^* \approx \frac{(z_{0.05} + z_{0.05})^2 \cdot p(1-p)}{(p - 0.5)^2} = \frac{(1.645 + 1.645)^2 \cdot 0.24}{0.01} \approx 260.
$$

KL framing: $\mathrm{KL}(\text{Bern}(0.6) \| \text{Bern}(0.5)) \approx 0.0208$, and $n^\ast \approx (z_\alpha + z_\beta)^2 / (2\,\mathrm{KL}) \approx 10.8 / 0.0417 \approx 260$ — which reproduces the power calculation above exactly, as it should. (A $1/\mathrm{KL}^2$ rate would give ~2400 and disagree by an order of magnitude; that is the tell that the rate is wrong.)

> **Saying it out loud.** The framing first: this is a hypothesis test, null is a fair coin, alternative is biased toward heads, and the question is meaningless until someone says *how* biased — you can never rule out a coin at 0.5001. So I'd say "detecting a sixty-percent coin at five percent error each way needs roughly two hundred sixty flips", and note that the count scales as one over the squared deviation from a half, so a fifty-five-percent coin costs four times as many. The honest addendum is that this assumes you fixed n in advance; if you're watching the count and deciding when to stop, your error rate is much worse than nominal, which is the same peeking problem as in A/B tests.

### Q3. "Estimate the mean of a normal distribution given 3 samples. What's your confidence interval?"

Use $t$-distribution (small $n$): $\bar{x} \pm t_{n-1, \alpha/2} \cdot s / \sqrt{n}$. With $n=3$, $t_{2, 0.025} \approx 4.30$ — wide CI.

> **Saying it out loud.** With three samples you use the t-distribution, not the normal, so the interval is the sample mean plus or minus about 4.3 times the standard error — and the number to actually say out loud is that 4.3, because it's enormous compared to the 1.96 people reflexively reach for. The real answer to "what's your confidence interval" at n equals three is "very wide, and I wouldn't act on it". I'd also flag that the t-interval assumes normality, and with three points you have no way to check that assumption — so the interval is conditional on something unverifiable, which is worth saying rather than hiding.

### Q4. "You sample $X_1, ..., X_n$ i.i.d. from a distribution with bounded variance. How concentrated is the sample mean?"

Chebyshev: $P(|\bar{X} - \mu| \ge t) \le \sigma^2 / (n t^2)$. Or CLT for $n \gtrsim 30$. Or Hoeffding if bounded support.

> **Saying it out loud.** Three tools depending on what I'm allowed to assume. Chebyshev needs only the variance and gives deviation probability bounded by sigma-squared over n-t-squared — always valid, always loose. Hoeffding needs bounded support and gives an exponentially small tail, which is much tighter. CLT gives the practical interval once n is past twenty or thirty. The distinction I'd make explicit is that the first two are guarantees and the third is an approximation, so which one I want depends on whether I'm making a claim or sizing an experiment.

### Q5. "What's the variance of the sample variance for a Gaussian?"

$\mathrm{Var}(s^2) = 2\sigma^4 / (n-1)$.

> **Saying it out loud.** For a Gaussian it's two sigma to the fourth over n minus one. The part worth saying out loud is what that implies: the standard error of the variance estimate scales with the variance itself, so variance is much harder to estimate than the mean, and heavy tails make it dramatically worse because the result depends on the fourth moment. If the distribution isn't Gaussian this formula is simply wrong, and for a distribution without a finite fourth moment the sample variance has infinite variance — which is the caveat I'd volunteer rather than wait for.

### Q6. "Why can't you just use empirical CDF for likelihood?"

Empirical CDF gives $\hat F(x)$, but the density is $\hat F'(x)$ — a sum of delta functions at observations. Useless for new points. Need smoothing.

> **Saying it out loud.** Because the empirical CDF is a step function, so its derivative is a sum of point masses at the observations and zero everywhere else. That gives every new point a density of exactly zero, which makes every likelihood ratio either zero over zero or infinite. The general lesson, which is the actual answer they're after: to evaluate a likelihood at a point you've never seen, you must smooth, and smoothing means choosing a bandwidth — so there's no assumption-free way to get a density from data.

### Q7. "How would you test if a sample came from a normal distribution?"

Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov, Q-Q plots, Jarque-Bera.

> **Saying it out loud.** Named tests: Shapiro-Wilk is the most powerful for small samples, Anderson-Darling weights the tails more heavily, Kolmogorov-Smirnov is general but weak, Jarque-Bera looks at skewness and kurtosis. But the answer I'd actually lead with is a Q-Q plot, because it shows you *how* the data departs from normal rather than just refusing to reject. And the caveat that matters most: with a large enough sample every normality test rejects, because no real data is exactly normal — so the useful question is never "is it normal" but "is the departure big enough to break what I'm about to do with it".

### Q8. "Two samples — same distribution test?"

Two-sample Kolmogorov-Smirnov. Or Mann-Whitney. Or $t$-test if assuming Gaussian. Or permutation test (most flexible).

> **Saying it out loud.** Depends on what "same" means, which I'd pin down first. Two-sample Kolmogorov-Smirnov if I mean the whole distribution and I have no parametric assumption. Mann-Whitney if I care about a shift in location and want robustness. A t-test if I'm willing to assume approximate normality and only care about means. And a permutation test if I want no assumptions at all and can afford the compute — which is usually the right answer these days, since it's exact under the null by construction. The thing to say is that these test different nulls, so picking one is a modelling decision, not a technicality.

### Q9. "Estimate KL between two empirical distributions."

KDE both, then numerically integrate; or use the $k$-NN-based estimator (Pérez-Cruz); or train a discriminator and use the bound from the GAN/$f$-divergence literature.

> **Saying it out loud.** Three routes, all imperfect. Kernel-density-estimate both and integrate numerically — fine in one or two dimensions, hopeless above that. Use a k-nearest-neighbour estimator like Pérez-Cruz, which avoids explicit density estimation and handles moderate dimension better. Or train a discriminator to separate the two samples and recover the divergence from its output, which is the trick underlying f-divergence and GAN-style estimation. The honest framing is that KL estimation from samples is hard and every estimator is biased, badly so in high dimensions — so if the answer matters, I'd want error bars from a bootstrap, and I'd be suspicious of any single number.

### Q10. "If KL between two distributions is $\epsilon$, how easy to discriminate?"

Sample complexity $\sim 1/\epsilon$ (Chernoff–Stein: error $\sim e^{-n\epsilon}$, so $n \approx \log(1/\beta)/\epsilon$). Bayes error rate decays as $e^{-n C(P, Q)}$ where $C \le \mathrm{KL}$.

> **Saying it out loud.** If KL is epsilon then evidence accumulates at about epsilon nats per sample, so you need on the order of one over epsilon samples — Chernoff–Stein, error decaying like e to the minus n epsilon. Note that's one over epsilon, not one over epsilon squared; it's an easy slip and you can catch it by checking the Gaussian case. And the tight exponent for the Bayes error is actually Chernoff information, which is at most KL, so KL gives you an optimistic bound rather than the exact rate. The useful intuition is just: KL is evidence per sample, so samples needed is one over evidence per sample.

### Q11. "Coin flip game: I flip; you guess; I pay you 2× your bet if right; how much do you bet?"

Kelly criterion: $f^* = (b p - q) / b$ where $b$ is odds ratio, $p$ is win prob, $q = 1-p$. With $b = 2, p = 0.5$: $f^* = (1 - 0.5)/2 = 0.25$ — bet 25% of bankroll.

> **Saying it out loud.** Kelly says bet the fraction that maximises expected log wealth, which for these odds works out to twenty-five percent of your bankroll. But I'd immediately caveat it, because that's the real answer: Kelly is optimal for maximising long-run growth rate assuming you know p exactly, you can bet fractionally, and you're playing many rounds. Misestimate p even slightly and full Kelly is aggressively over-levered — which is why practitioners bet half-Kelly or less. So the answer is twenty-five percent in theory, materially less in practice, and the gap between those is the interesting part.

### Q12. "$X$ uniform on [0,1]. What's $\mathbb{E}[\max(X, 0.5)]$?"

Split: with prob 0.5 max=0.5; otherwise max=$X$ with $X \in [0.5, 1]$, so $\mathbb{E}[X | X > 0.5] = 0.75$, with prob 0.5. Total: $0.5 \cdot 0.5 + 0.5 \cdot 0.75 = 0.625$.

> **Saying it out loud.** Split on the event. With probability a half X is below a half, and the max is a half; with probability a half X is above, and its conditional expectation is 0.75. So the answer is 0.625. Worth saying the sanity check out loud: it must be strictly above 0.5 since we're flooring at 0.5, and strictly below the unconditional 0.75, and 0.625 sits between them.

### Q13. "Two coins; one fair, one always heads. You pick one and flip 10 times, all heads. What's $P(\text{fair})$?"

Bayes: $P(\text{fair} \mid 10H) = P(10H | \text{fair}) \cdot 0.5 / [P(10H | \text{fair}) \cdot 0.5 + P(10H | \text{biased}) \cdot 0.5] = (1/1024) \cdot 0.5 / [(1/1024) \cdot 0.5 + 1 \cdot 0.5] = 1/1025 \approx 0.001$. Almost certainly biased.

> **Saying it out loud.** Bayes with equal priors: ten heads has probability one over 1024 under the fair coin and probability one under the two-headed coin, so the posterior on fair is about one in 1025 — essentially zero. The framing worth adding is that each flip contributes one bit of evidence, so ten flips is ten bits, and the prior odds were one to one, so the posterior odds are 1024 to 1 — you can do it in your head without touching the formula. And the caveat: this is only that decisive because the alternative is *always* heads; against a coin at 0.9 the evidence per flip is much weaker.

### Q14. "$X, Y$ i.i.d. uniform on $[0, 1]$. What's $\mathbb{E}[\max(X, Y)]$?"

CDF of max: $F_M(z) = z^2$. PDF: $2z$. $\mathbb{E}[M] = \int_0^1 z \cdot 2z \, dz = 2/3$.

> **Saying it out loud.** The max of two uniforms has CDF z squared and density 2z, so the expectation is two-thirds. The generalisation is worth offering unprompted: for n uniforms the max has expectation n over n plus one, which approaches one, and the useful intuition is that the max of n draws sits about one over n from the top. That's the same fact that drives extreme value theory and, incidentally, why best-of-N sampling has diminishing returns.

### Q15. "How many people for 50% birthday collision probability?"

Birthday problem. $P(\text{no collision among }n) = \prod_{k=0}^{n-1} (365 - k)/365 \approx e^{-n(n-1)/(2 \cdot 365)}$. Set $\approx 0.5$ → $n \approx 23$.

> **Saying it out loud.** Twenty-three, which surprises people because they're answering a different question — the chance that *someone* shares a birthday, not that someone shares *your* birthday. There are n choose 2 pairs, so collisions grow quadratically while the number of days is fixed; setting n squared over two times 365 to about log 2 gives 23. The generalisation to state is the square root rule: you get a collision after roughly the square root of the number of possible values, which is exactly the birthday attack in cryptography.

### Q16. "Three doors, prize behind one, you pick door 1, host opens door 3 (no prize). Switch?"

Monty Hall. Yes — prob of prize behind door 2 is 2/3 (host's action carries information).

> **Saying it out loud.** Switch — two-thirds versus one-third. The reason isn't the counting, it's that the host's action carries information: he knows where the prize is and is constrained to open an empty door, so his choice is not random and it tells you about the door he didn't open. The cleanest way to say it out loud is that your original door had a one-third chance and nothing that happened changed it, so the remaining two-thirds has to concentrate on the one door left.

### Q17. "Why does Monty Hall break if host acts randomly?"

Then host's action carries less information; conditional probabilities change. You should still consider info-theoretic value of the action.

> **Saying it out loud.** Because the whole effect came from the host's knowledge, not from a door opening. If he opens a door at random and it happens to be empty, you've learned nothing about the two remaining doors relative to each other, so they're at fifty-fifty and switching is worthless. The general lesson, and it's the one worth stating: conditioning on an event is not the same as conditioning on a *process* — you have to know how the information was generated, not just what it was. That's the same mistake behind most selection-bias errors.

### Q18. "Estimate $\pi$ via random sampling."

Sample $(x, y) \sim U[0,1]^2$. Fraction in unit quarter circle is $\pi/4$. Multiply by 4. Variance $\propto 1/n$.

> **Saying it out loud.** Sample points uniformly in the unit square, count the fraction inside the quarter circle, multiply by four. The part I'd volunteer is the accuracy, because it's the point of the question: the error shrinks like one over the square root of n, so you need a hundred times more samples for one more digit. That's the fundamental Monte Carlo rate and it's why Monte Carlo is a terrible way to compute pi and an excellent way to compute integrals in high dimensions, where the rate doesn't depend on dimension at all.

### Q19. "$X \sim \text{Exp}(\lambda)$. What's $P(X > a + b | X > a)$?"

Memorylessness: $P(X > a + b | X > a) = P(X > b)$.

> **Saying it out loud.** It equals the probability that X exceeds b — the exponential is memoryless, so having waited a already tells you nothing. Worth adding that the exponential is the *only* continuous distribution with this property, and the geometric is its discrete counterpart. And the modelling caveat: memorylessness is a strong assumption that's often wrong in practice — machines wear out, so if failure rates rise with age the exponential is the wrong model and you want a Weibull.

### Q20. "Sum of two i.i.d. exponentials — what distribution?"

Gamma(2, $\lambda$). Sum of $k$ i.i.d. Exp($\lambda$) is Gamma($k, \lambda$).

> **Saying it out loud.** Gamma with shape 2 and the same rate; in general the sum of k i.i.d. exponentials is Gamma with shape k. The intuition to give is that this is the waiting time until the k-th event in a Poisson process, which makes the whole exponential-Gamma-Poisson family one story rather than three facts. And as k grows the Gamma becomes approximately normal, which is just the CLT showing up in a place where people don't expect it.

### Q21. "Why is the median more robust than the mean?"

Median has 50% breakdown point; mean has 0%. One outlier moves the mean unboundedly, doesn't move the median.

> **Saying it out loud.** Breakdown point: the median tolerates up to fifty percent of the data being arbitrarily corrupted, the mean tolerates zero — one point sent to infinity drags the mean to infinity. The tradeoff to say out loud rather than leave implicit is efficiency: for genuinely Gaussian data the median has about sixty-four percent the efficiency of the mean, so you're paying roughly a third more data for that robustness. Which is the right trade depends entirely on how much you believe your tails, and that's a judgement call, not a theorem.

### Q22. "Detect a change-point in a stream of values from a known distribution."

CUSUM, GLR (generalized likelihood ratio), Bayesian online change-point detection. Sequential framework — every time, compute likelihood ratio under "no change" vs "change at $t$" hypothesis; if exceeds threshold, declare change.

> **Saying it out loud.** Sequential likelihood ratio, in one form or another. CUSUM accumulates the log-likelihood ratio between change and no-change and alarms when it exceeds a threshold; the generalised likelihood ratio does the same while also maximising over the unknown change point and post-change parameter; Bayesian online change-point detection maintains a posterior over run length. The tradeoff that defines the whole problem is detection delay against false alarm rate — you can have either, and the threshold is where you choose. I'd also flag that a threshold tuned offline usually behaves differently in production because real streams are non-stationary even without a change point.

### Q23. "Estimate the size of a population with unique IDs from a single sample."

German tank problem. If max observed = $m$ from $n$ samples: MLE estimate $= m$, but minimum-variance unbiased estimator $= m \cdot (n+1)/n - 1 = m + m/n - 1$.

> **Saying it out loud.** German tank problem. The MLE is just the largest ID you've seen, which is obviously biased low — you almost certainly haven't seen the maximum. The minimum-variance unbiased estimator corrects it by adding the average gap: m plus m over n minus one. The intuition worth saying is that the observed maximum plus the typical spacing between order statistics is a better guess than the maximum itself, and this is a nice case where the MLE is clearly wrong in finite samples even though it's asymptotically fine. The assumption doing all the work is that IDs are sequential from one and your sample is uniform — break either and the estimate is worthless.

### Q24. "Two-sample mean test — but the variances differ."

Welch's $t$-test. Adjusted degrees of freedom. Non-parametric: Mann-Whitney.

> **Saying it out loud.** Welch's t-test, which doesn't assume equal variances and adjusts the degrees of freedom via the Satterthwaite approximation. The thing I'd say out loud is that Welch should really be the default even when you think the variances match, since it costs almost nothing when they do and saves you when they don't — pretesting for equal variance and then choosing is worse than just using Welch. If normality is also in doubt, Mann-Whitney or a permutation test, with the caveat that Mann-Whitney is testing stochastic dominance rather than a difference in means.

### Q25. "AB test: significant at $p=0.04$, $n=10000$. Should you ship?"

Discuss: practical significance vs statistical, multiple testing, peeking, effect size, business cost of being wrong. Senior signal: don't take the p-value at face value.

> **Saying it out loud.** I wouldn't answer yes or no from a p-value. The questions I'd ask first: what's the effect size and its confidence interval, because at n equals ten thousand a statistically significant effect can be commercially meaningless; how many metrics and variants were tested, because 0.04 across twenty tests is what you'd expect by chance; was the sample size fixed in advance or did someone stop when it went green, since peeking inflates the false positive rate substantially; and what does the interval look like at the low end, because if shipping is cheap and the lower bound is still positive, ship. The senior signal is treating 0.04 as one input to a decision under uncertainty rather than as a verdict.

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

> **Saying it out loud.** The follow-ups are all probing the same thing: do you know where your answer stops being true. So the useful reflex is to have already named your assumptions, because then each probe has a home. Prior wrong — fine, the posterior is dominated by the data when n is large and by the prior when it isn't, and I can tell you roughly where the crossover is. Distributions overlap — then there's a Bayes error floor and no method beats it, so the honest answer is that the question isn't answerable to the accuracy you want. Parametric family unknown — go non-parametric or discriminative and accept a slower rate. How does it fail in production — distribution shift, out-of-distribution inputs, and label noise, in roughly that order of likelihood. The thing that reads as senior is that none of these should surprise you, because you flagged them yourself thirty seconds earlier.

---

## 12. Senior-level signals

- **You start with the framing checklist.** Don't jump to a formula.
- **You name the framework** (Bayes / MLE / Concentration / Bandit / Importance / Stein) explicitly.
- **You quantify confidence** — $\log \Lambda$, posterior, CI, sample complexity in $1/\mathrm{KL}$.
- **You discuss assumptions** and what fails when they're wrong.
- **You name the connection to information theory** (KL, Fisher, Chernoff).
- **You think about OOD / failure modes**, not just the happy path.
- **You distinguish frequentist vs Bayesian** when relevant.
- **You mention the production-grade variant** (online estimation, drift detection, hypothesis testing under multiple comparisons).
- **You don't over-claim**. "Optimal under 0-1 loss with these priors" — not "optimal."
- **You can pivot from the analytical answer to a programmatic one** if asked.

> **Saying it out loud.** What separates a strong answer here is almost entirely about calibration rather than knowledge. You frame before you compute, you name the framework out loud so the interviewer knows where you're going, and you quantify confidence rather than just producing a point estimate. But the biggest single differentiator is not over-claiming: say "optimal under zero-one loss with these priors and this model family", not "optimal", because the caveat is the actual expertise. And when you don't know, say what would settle it — which experiment, which check, which limiting case — because "I'd want to check X" is a real answer and confident hand-waving is a failed one. These questions are designed so there's no clean answer; what's being measured is whether you can be useful anyway.

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
5. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) for active recall.
6. Practice out loud — these are oral exams in real interviews.

Single sentence to remember: **frame as Bayesian classification or MLE / decision / concentration, name the framework explicitly, quantify with KL or Fisher or Chernoff, discuss assumptions and OOD, and end with sample complexity.**
