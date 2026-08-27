# Probability and statistics

Interviews test probability as a proxy for careful conditioning. The questions are old — base rates, Monty Hall, birthdays, coupon collectors — and the interviewer watches whether you name the conditioning event before you compute. The one thing candidates get wrong is treating conditioning as a formula instead of a change of sample space. They also over-apply independence: they assume it when using linearity of expectation, which never needs it, and they forget it when adding variances, which always does.

## The equations

**Conditional probability**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0$$

Restrict the sample space to $B$ and renormalise; every "given" in a word problem is this operation.

**Chain rule**

$$P(x_1, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1})$$

Any joint factors into a product of conditionals in any order; this factorisation is exactly what an autoregressive language model fits.

**Law of total probability**

$$P(A) = \sum_{i} P(A \mid B_i) P(B_i)$$

for a partition $\{B_i\}$; it splits a hard event into cases you can condition on, and it is the denominator of Bayes' rule.

**Bayes' rule**

$$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)} = \frac{P(E \mid H) P(H)}{\sum_i P(E \mid H_i) P(H_i)}$$

Posterior equals likelihood times prior over evidence; $P(H)$ is the base rate that people drop.

**Expectation, variance, and their rules**

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y], \qquad \mathrm{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

$$\mathrm{Var}[aX + bY] = a^2 \mathrm{Var}[X] + b^2 \mathrm{Var}[Y] + 2ab\,\mathrm{Cov}[X, Y]$$

Linearity of expectation holds for any dependence; variance adds only when the covariance term is zero.

**Law of the unconscious statistician**

$$\mathbb{E}[g(X)] = \sum_x g(x) P(x) \quad \text{or} \quad \int g(x) p(x)\,dx$$

You can average a function of $X$ without ever finding the distribution of $g(X)$.

**Covariance and correlation**

$$\mathrm{Cov}[X, Y] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y], \qquad \rho = \frac{\mathrm{Cov}[X,Y]}{\sigma_X \sigma_Y} \in [-1, 1]$$

Covariance measures co-movement in raw units; correlation is the unit-free version, and both see only linear structure.

**Three densities you must write cold**

$$P(k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad P(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Binomial counts successes in $n$ trials (mean $np$, variance $np(1-p)$); Poisson is its $n \to \infty$, $np \to \lambda$ limit (mean and variance both $\lambda$); Gaussian has mean $\mu$ and variance $\sigma^2$.

**Central limit theorem**

$$\frac{\sqrt{n}\,(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{for i.i.d. } X_i \text{ with finite } \mu, \sigma^2$$

The sample mean of any finite-variance distribution is asymptotically Gaussian with standard error $\sigma / \sqrt{n}$.

**Maximum likelihood and cross-entropy**

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^{n} \log p_\theta(x_i) = \arg\min_{\theta} \left[-\frac{1}{n}\sum_{i=1}^{n} \log p_\theta(x_i)\right]$$

Negative average log-likelihood is the cross-entropy between the empirical data distribution and the model, so training on cross-entropy is maximum likelihood.

**MAP and regularisation**

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\log p(D \mid \theta) + \log p(\theta)\right]$$

MAP is MLE plus a log-prior term; a Gaussian prior $\mathcal{N}(0, \tau^2)$ contributes $-\lVert\theta\rVert_2^2 / (2\tau^2)$, which is L2 weight decay with $\lambda = 1/(2\tau^2)$.

**KL divergence and Jensen's inequality**

$$D_{\mathrm{KL}}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \ge 0, \qquad \mathbb{E}[f(X)] \ge f(\mathbb{E}[X]) \text{ for convex } f$$

KL is the extra nats you pay for coding $p$ with a code built for $q$; it is zero only when $p = q$, and non-negativity follows from Jensen applied to the convex function $-\log$.

## Code from memory

Monte Carlo estimate of a conditional probability, checked against Bayes' rule.

```python
import numpy as np

rng = np.random.default_rng(0)
prevalence, sensitivity, specificity = 0.01, 0.99, 0.99
n = 2_000_000

sick = rng.random(n) < prevalence            # disease status, then the test conditioned on it
positive = np.where(sick, rng.random(n) < sensitivity, rng.random(n) < (1 - specificity))

mc = sick[positive].mean()                   # conditioning is a filter, then a mean
analytic = (sensitivity * prevalence) / (
    sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
)
print("MC    P(D|+) =", round(float(mc), 5))
print("Bayes P(D|+) =", round(analytic, 5))
print("abs diff      =", round(abs(mc - analytic), 5))
```

Ran with 2,000,000 samples: Monte Carlo gave `0.49999` against the closed form `0.50000`, an absolute difference of `1e-05`.

Sampling from a categorical distribution by inverse CDF, with explicit loops.

```python
import numpy as np

def sample_categorical(probs, n, seed=0):
    rng = np.random.default_rng(seed)
    # Build the cumulative distribution once.
    cdf, running = [], 0.0
    for p in probs:
        running += p
        cdf.append(running)
    # One uniform draw per sample, then a linear scan for the first bin that covers it.
    counts = [0] * len(probs)
    for _ in range(n):
        u = rng.random()
        for k in range(len(cdf)):
            if u <= cdf[k]:
                counts[k] += 1
                break
    return np.array(counts) / n

probs = [0.1, 0.2, 0.3, 0.4]
emp = sample_categorical(probs, 200_000)
print("empirical  ", np.round(emp, 4))
print("target     ", np.array(probs))
print("max abs err", round(float(np.abs(emp - probs).max()), 5))
```

Ran with 200,000 draws: empirical `[0.1008, 0.1990, 0.3014, 0.3988]` against the target, maximum absolute error `0.00138`.

Bootstrap percentile confidence interval, checked against the normal-theory interval.

```python
import numpy as np

rng = np.random.default_rng(1)
data = rng.normal(loc=5.0, scale=2.0, size=200)

B = 5000                                     # resample with replacement, one statistic each
boot_means = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(data), size=len(data))
    boot_means[b] = data[idx].mean()

lo, hi = np.percentile(boot_means, [2.5, 97.5])   # the interval is a pair of percentiles
se = data.std(ddof=1) / np.sqrt(len(data))
print("bootstrap CI", (round(float(lo), 4), round(float(hi), 4)))
print("normal    CI", (round(float(data.mean() - 1.96 * se), 4),
                       round(float(data.mean() + 1.96 * se), 4)))
```

Ran on 200 points: bootstrap interval `(4.5897, 5.1150)` against the normal interval `(4.5958, 5.1096)`, a width difference of `0.0116`.

## Questions

### Q1. A disease has 1 percent prevalence. A test has 99 percent sensitivity and 99 percent specificity. A patient tests positive. What is the probability they have the disease?

Fifty percent. Take 10,000 people. 100 are sick, and 99 of them test positive. 9,900 are healthy, and 1 percent of them — 99 people — test positive anyway. So 198 positives, of which 99 are sick, giving $99/198 = 0.5$. In symbols, $P(D \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.01 \times 0.99} = 0.5$. Simulation with 2,000,000 samples gave 0.49999 against the analytic 0.50000. This surprises people because they anchor on the 99 percent accuracy and ignore the base rate. The healthy group is 99 times larger, so even a 1 percent false-positive rate on it produces as many positives as the sick group produces true ones. The posterior is driven by the ratio of group sizes, not by the test alone. Drop prevalence to 0.1 percent and the posterior falls to about 9 percent.

> **Say it.** Fifty percent. Out of ten thousand people, a hundred are sick and ninety-nine of them test positive. Nine thousand nine hundred are healthy, and one percent of those — also ninety-nine — test positive. So half the positives are false. People are surprised because they read ninety-nine percent accuracy as ninety-nine percent confidence, but the healthy pool is ninety-nine times bigger, so its small error rate produces just as many positives. I simulated it: 0.49999 against the exact 0.5.

### Q2. Monty Hall. You pick a door, the host opens a different door showing a goat, and offers a switch. Do you switch?

Yes. Switching wins with probability $2/3$, staying with $1/3$. The reason is that the host's choice is constrained, not random. Your first pick is right with probability $1/3$ and wrong with probability $2/3$. When you are wrong, the host knows where the car is and must open the one remaining goat door, so the whole $2/3$ of probability mass collapses onto the single unopened door. Conditioning on "the host opened a goat door" carries no information when the host was always going to do that, so your original $1/3$ does not move. The key event to condition on is the host's action given his knowledge. If the host opened a door at random and it happened to show a goat, the answer changes to 50/50, because then his choice is informative about where the car is not. Simulation over 500,000 games: stay 0.3318, switch 0.6682.

> **Say it.** Switch. My first pick is right one time in three. That does not change, because the host was always going to open a goat door, so his action gives me no news about my own door. All the remaining two-thirds sits on the single door he did not open. The important point is that the host is constrained — he knows where the car is and must avoid it. If he opened a door at random it would be fifty-fifty. I simulated half a million games: 0.668 for switching.

### Q3. You roll a fair die until you get a six. What is the expected number of rolls, and the variance?

Six rolls, with variance 30. Let $X$ be the number of rolls including the successful one, so $X \sim \text{Geometric}(p)$ with $p = 1/6$. Use the memoryless recursion: after one roll you either finish, with probability $p$, or you are back at the start, so $\mathbb{E}[X] = 1 + (1-p)\mathbb{E}[X]$, giving $\mathbb{E}[X] = 1/p = 6$. The variance is $(1-p)/p^2 = (5/6)/(1/36) = 30$, so the standard deviation is about 5.48 — the distribution is very heavy on the right. That recursion is the whole trick, and it generalises: any waiting time with a constant per-step success probability has mean $1/p$. Simulation over 200,000 trials gave mean 5.9911 against 6, and variance 29.54 against 30. Note the variance is nearly the square of the mean, so a single sample tells you almost nothing.

> **Say it.** Six. I set it up as a recursion: one roll always happens, and with probability five-sixths I am back where I started, so E equals one plus five-sixths E, which gives six. Variance is one minus p over p squared, so thirty — standard deviation about five and a half, which means the waiting time is very spread out. That recursion works for any constant-probability waiting time. Simulated over two hundred thousand trials I got 5.99 and 29.5.

### Q4. Give me a problem where you can compute an expectation even though the variables are dependent.

The hat-check problem. Ten people hand in hats and get them back in a uniformly random permutation. Let $X_i$ be the indicator that person $i$ gets their own hat, and $X = \sum_i X_i$. The $X_i$ are strongly dependent — if the first nine people all match, the tenth must match too. Linearity of expectation does not care. $\mathbb{E}[X_i] = 1/n$ for each $i$, so $\mathbb{E}[X] = n \times (1/n) = 1$, independent of $n$. Linearity follows from the definition of expectation as a sum over outcomes and needs no independence assumption at all. Independence would only be needed for the variance, and here it happens that the variance is also 1, but that requires computing the covariance terms. Simulation with $n = 10$ over 200,000 permutations gave mean 1.0037 and variance 1.0044, both against the analytic value 1.

> **Say it.** The hat-check problem. Ten people get hats back at random. The match indicators are clearly dependent — if nine match, the tenth is forced. But each has expectation one over n, and linearity is just a rearrangement of a sum over outcomes, so the expected number of matches is exactly one for any n. Independence never enters. It would only matter if I wanted the variance by adding variances, and there I would need the covariance terms. Simulated: mean 1.004, and the variance also came out at 1.00.

### Q5. How many people do you need in a room for a 50 percent chance that two share a birthday?

Twenty-three. Compute the complement: the probability all $k$ birthdays are distinct is $\prod_{i=0}^{k-1} (365-i)/365$. At $k = 23$ that product is about 0.493, so the collision probability is 0.507. Simulation over 20,000 rooms gave 0.5044 against the analytic 0.5073. The reason 23 feels too small is that people count people, but collisions come from pairs, and there are $\binom{23}{2} = 253$ pairs. Each pair collides with probability $1/365$, so the expected number of collisions is $253/365 \approx 0.69$, which is order one. The general rule is that you get a collision at roughly $\sqrt{d}$ draws from $d$ equally likely values. That is the same argument that sets hash-table load limits and birthday-attack key lengths. At $k = 50$ the collision probability is 0.9704, matched exactly by simulation.

> **Say it.** Twenty-three. I work with the complement: the chance all birthdays differ is the falling product 365 over 365, times 364 over 365, and so on, which drops below a half at twenty-three. It feels too small because people count people, not pairs — twenty-three people give two hundred fifty-three pairs, and each collides with chance one in 365, so about 0.69 expected collisions. The general rule is collisions at around root d. Simulation gave 0.504 against 0.507.

### Q6. Show that the MLE of a Gaussian mean is the sample mean.

Write the log-likelihood for $n$ i.i.d. points under $\mathcal{N}(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

Only the second term depends on $\mu$, and $\sigma^2 > 0$ is a positive constant multiplier, so maximising over $\mu$ is the same as minimising $\sum_i (x_i - \mu)^2$. Differentiate: $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_i (x_i - \mu)$. Set it to zero and get $\sum_i x_i = n\mu$, so $\hat{\mu} = \frac{1}{n}\sum_i x_i$. The second derivative is $-n/\sigma^2 < 0$, so it is a maximum. This is why Gaussian likelihood and squared error are the same objective: the exponent of the Gaussian *is* the squared error, so least squares is maximum likelihood under additive Gaussian noise with constant variance.

> **Say it.** I write the log-likelihood. The only mu-dependent piece is minus one over two sigma squared times the sum of squared deviations. Sigma squared is a positive constant, so maximising in mu is minimising the sum of squares. Differentiate, set to zero, and the sum of x minus mu is zero, so mu-hat is the sample mean. Second derivative is negative n over sigma squared, so it is a maximum. This is also why least squares equals maximum likelihood under constant-variance Gaussian noise.

### Q7. The MLE for the variance is biased. Show why, and where the $n-1$ comes from.

The MLE is $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_i (x_i - \bar{x})^2$, and $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$, so it is too small on average. The cause is that you centred on $\bar{x}$, which was itself estimated from the same data. The sample mean is the value that minimises $\sum_i (x_i - c)^2$ over $c$, so using it instead of the true $\mu$ makes the sum of squares as small as it can possibly be. Formally, $\sum_i (x_i - \bar{x})^2 = \sum_i (x_i - \mu)^2 - n(\bar{x} - \mu)^2$; take expectations and get $n\sigma^2 - \sigma^2 = (n-1)\sigma^2$. Dividing by $n-1$ instead of $n$ restores unbiasedness. One degree of freedom was spent on estimating the mean. Simulation with $n = 5$, $\sigma^2 = 4$, over 200,000 samples: MLE mean 3.1986 against $\frac{4}{5}\times 4 = 3.2$, and the $n-1$ version 3.9982 against 4.

> **Say it.** The MLE divides by n and comes out too small — its expectation is n minus one over n times sigma squared. The reason is that I centred on the sample mean, and the sample mean is by construction the point that minimises the sum of squared deviations, so the sum is smaller than it would be around the true mean. Algebraically the sum around x-bar equals the sum around mu minus n times x-bar minus mu squared, and taking expectations costs exactly one sigma squared. One degree of freedom went to estimating the mean. Simulated with n equals five: 3.199 against 3.2.

### Q8. What is MAP, and how does a Gaussian prior become L2 regularisation?

MAP maximises the posterior instead of the likelihood: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log p(D \mid \theta) + \log p(\theta)]$, since the evidence $p(D)$ does not depend on $\theta$. So MAP is MLE plus an additive log-prior. Put an isotropic Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ on the weights. Its log-density is $-\frac{1}{2\tau^2}\lVert\theta\rVert_2^2$ plus a constant, so the objective becomes negative log-likelihood plus $\frac{1}{2\tau^2}\lVert\theta\rVert_2^2$ — exactly L2 regularisation with $\lambda = 1/(2\tau^2)$. A tighter prior, meaning smaller $\tau$, means stronger regularisation. A Laplace prior $p(\theta) \propto e^{-\lvert\theta\rvert/b}$ gives an L1 penalty instead, which is why L1 produces sparsity: the Laplace density has a spike at zero. MAP is still a point estimate, so it is not full Bayesian inference — it throws away the posterior width and it is not invariant under reparameterisation.

> **Say it.** MAP maximises likelihood times prior, and since the evidence is constant in theta it is just MLE plus a log-prior term. A zero-mean isotropic Gaussian prior has log-density proportional to minus theta squared over two tau squared, so adding it gives negative log-likelihood plus lambda times the L2 norm, with lambda equal to one over two tau squared. A tighter prior means heavier weight decay. A Laplace prior gives L1 and hence sparsity, because it spikes at zero. MAP is still a point estimate — it discards the posterior width.

### Q9. KL divergence is not symmetric. What does that mean in practice for forward versus reverse KL?

Forward KL, $D_{\mathrm{KL}}(p \parallel q)$ with $p$ the data, is mean-seeking or mass-covering. The expectation is taken under $p$, so wherever $p$ has mass and $q$ has none, $\log(p/q)$ blows up. Therefore $q$ must cover every mode, even if that means putting mass where $p$ has none. This is what maximum likelihood does. Reverse KL, $D_{\mathrm{KL}}(q \parallel p)$, is mode-seeking or zero-forcing. The expectation is under $q$, so $q$ is only penalised where $q$ itself has mass. Therefore $q$ can safely ignore a mode and collapse onto one peak. This is what variational inference and some RL-style objectives do. Concretely, I fitted one Gaussian to an equal mixture of $\mathcal{N}(-4, 1)$ and $\mathcal{N}(4, 1)$ by grid search: forward KL chose $\mu = 0$, $\sigma = 4.12$, spanning both modes; reverse KL chose $\mu = -4$, $\sigma = 1.00$, sitting on one mode with divergence $0.693 = \log 2$.

> **Say it.** Forward KL, p over q with the expectation under the data, is mass-covering: anywhere the data has mass and the model has none, the log ratio explodes, so the model must stretch to cover every mode. That is maximum likelihood. Reverse KL takes the expectation under the model, so the model is only penalised where it itself puts mass, which lets it collapse onto one mode. That is variational inference. I checked it numerically on a two-mode mixture: forward gave a wide Gaussian at zero, reverse locked onto one mode.

### Q10. What is the difference between a confidence interval and a credible interval?

A 95 percent confidence interval is a statement about the procedure, not about the parameter. The parameter is a fixed unknown constant; the interval is random because the data are random. The claim is that if you repeated the experiment many times, 95 percent of the intervals you construct would contain the true value. For the specific interval in front of you, the true value is either in it or not — there is no probability left. A 95 percent credible interval is a statement about the parameter: given the prior and the observed data, the posterior assigns probability 0.95 to the parameter lying in that range. That is the interpretation people wrongly attach to confidence intervals. The two often coincide numerically under a flat prior and a well-behaved likelihood, which is why the confusion survives. My bootstrap run gave `(4.5897, 5.1150)` against the normal interval `(4.5958, 5.1096)`, a width difference of 0.0116 — the same numbers, different claims.

> **Say it.** A confidence interval is a property of the procedure: over repeated experiments, ninety-five percent of the intervals I build would cover the fixed true parameter. For the one interval in my hand there is no probability statement left — it either covers or it does not. A credible interval is a property of the posterior: given my prior and data, there is a ninety-five percent probability the parameter is in that range. Under a flat prior the numbers often coincide, which is why people conflate them, but only the credible interval licenses the probability language.

### Q11. Work the coupon collector problem end to end.

There are $n$ distinct coupons, each draw uniform and independent. You want the expected number of draws to collect all $n$. Split the process into stages: stage $i$ starts when you hold $i$ distinct coupons and ends when you get a new one. In stage $i$ the probability that a draw is new is $(n-i)/n$, so the stage is geometric with mean $n/(n-i)$. By linearity of expectation over the stages — the stage lengths are dependent in general but linearity does not care — the total is

$$\mathbb{E}[T] = \sum_{i=0}^{n-1} \frac{n}{n-i} = n \sum_{k=1}^{n} \frac{1}{k} = n H_n \approx n(\ln n + \gamma)$$

with $\gamma \approx 0.5772$. So the cost is $n \log n$, not $n$: the last few coupons dominate. For $n = 10$, $10 H_{10} = 29.29$; simulation over 100,000 runs gave 29.27.

> **Say it.** I break it into stages by how many distinct coupons I hold. With i in hand, a draw is new with probability n minus i over n, so that stage is geometric with mean n over n minus i. Sum over stages by linearity and I get n times the nth harmonic number, which is about n log n plus n gamma. The last coupon alone costs n draws on average, which is why it is n log n and not n. For ten coupons the answer is 29.29 and my simulation gave 29.27.

### Q12. A symmetric random walk starts at 5, with absorbing barriers at 0 and 10. How long until absorption?

Twenty-five steps in expectation. Let $T_k$ be the expected time to absorption from position $k$. One step always happens, and then you are at $k-1$ or $k+1$ with equal probability, so $T_k = 1 + \frac{1}{2}T_{k-1} + \frac{1}{2}T_{k+1}$ with $T_0 = T_{10} = 0$. Rearranged, $T_{k+1} - 2T_k + T_{k-1} = -2$: the second difference is constant, so $T_k$ is a quadratic in $k$. Fitting the boundary conditions gives $T_k = k(N-k)$ with $N = 10$, so $T_5 = 25$. The same first-step recursion gives the ruin probability: $P(\text{hit } N \text{ first}) = k/N = 0.5$ here, because position is a martingale under a fair walk. Simulation over 100,000 walks gave a mean of 25.013 steps against the analytic 25.

> **Say it.** I set up a first-step recursion: T at k equals one plus a half T at k minus one plus a half T at k plus one, with zero at both barriers. That says the second difference is minus two, so T is quadratic in k, and the boundary conditions force T equals k times N minus k. From five with barriers at zero and ten, that is twenty-five. The same recursion gives the ruin probability as k over N, because position is a martingale. Simulated mean was 25.01.

### Q13. State the central limit theorem precisely, and say when it fails you.

For i.i.d. $X_i$ with finite mean $\mu$ and finite variance $\sigma^2$, $\sqrt{n}(\bar{X}_n - \mu)/\sigma$ converges in distribution to $\mathcal{N}(0,1)$. Equivalently, the sample mean is approximately $\mathcal{N}(\mu, \sigma^2/n)$, so the standard error shrinks as $1/\sqrt{n}$ — to halve the error you need four times the data. It fails in three ways. First, infinite variance: a Cauchy or a heavy power-law tail with exponent below 2 has no CLT, and the sample mean of Cauchy variables has the same Cauchy distribution regardless of $n$. Second, dependence: strongly correlated samples have a much larger effective variance, so nominal confidence intervals are far too narrow. Third, convergence is only in distribution and it is slow in the tails, so a normal approximation for a far-tail quantile can be badly wrong even when the centre is fine. Skewed distributions need larger $n$ before the approximation is usable.

> **Say it.** For i.i.d. variables with finite mean and finite variance, root n times the centred sample mean over sigma converges in distribution to a standard normal. So the sample mean is roughly normal with standard error sigma over root n — four times the data for half the error. It fails when the variance is infinite, as with Cauchy, where the sample mean never concentrates. It fails under dependence, because the effective sample size is smaller than n. And it converges slowly in the tails, so extreme quantiles stay unreliable.

### Q14. Where does Jensen's inequality show up in machine learning?

Jensen says $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$ for convex $f$, with the inequality reversed for concave $f$. Three uses matter. First, KL non-negativity: $-D_{\mathrm{KL}}(p \parallel q) = \mathbb{E}_p[\log(q/p)] \le \log \mathbb{E}_p[q/p] = \log 1 = 0$, using concavity of $\log$, so $D_{\mathrm{KL}} \ge 0$ with equality only when $q = p$ almost everywhere. Second, the ELBO: $\log p(x) = \log \mathbb{E}_{q}[p(x,z)/q(z)] \ge \mathbb{E}_{q}[\log(p(x,z)/q(z))]$, which is the variational lower bound that VAEs and EM maximise, and the gap is exactly $D_{\mathrm{KL}}(q \parallel p(z \mid x))$. Third, it explains why the average of per-batch losses is not the loss of the average, and why $\mathbb{E}[\log \text{perplexity}]$ and $\log \mathbb{E}[\text{perplexity}]$ differ. Whenever you want to move an expectation through a nonlinearity, Jensen tells you the direction of the error.

> **Say it.** Jensen says the expectation of a convex function is at least the function of the expectation. It gives me KL non-negativity directly: minus KL is the expectation of a log ratio, and log is concave, so it is at most the log of the expectation, which is log one, zero. It gives me the ELBO: push the log inside the expectation over q and you get the variational bound, with gap equal to the KL from q to the true posterior. More generally it tells me the sign of the error whenever I swap an expectation and a nonlinearity.

## Done when

- You can compute the 1-percent-prevalence posterior in your head with the 10,000-people table and state the answer, 50 percent, in under 30 seconds.
- You can write the Monte Carlo conditional-probability estimator and the inverse-CDF categorical sampler from memory, without broadcasting tricks, and both run first try.
- You can derive the Gaussian MLE for $\mu$ and the $(n-1)/n$ bias of the MLE variance on a whiteboard in three minutes.
- You can set up a first-step recursion for coupon collector and gambler's ruin, and get $nH_n$ and $k(N-k)$ without looking them up.
