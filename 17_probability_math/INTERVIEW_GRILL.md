# Probability for ML — Interview Grill

> 50 questions on probability fundamentals, distributions, Bayes, limit theorems. Drill until you can answer 35+ cold.

---

## A. Probability basics

**1. State the three probability axioms.**
$\mathbb{P}(\Omega) = 1$. $\mathbb{P}(A) \in [0,1]$. Countable additivity for disjoint events.

> **Saying it out loud.** You've got one unit of belief and you're distributing it over everything that could happen. So: the whole sample space gets probability one, nothing gets a negative probability, and if events can't happen together their probabilities just add. That's it — every other rule in probability is derived from those three. Worth saying out loud that additivity is stated for countably many disjoint events, because that's the bit that makes measure theory necessary.

**2. What's the inclusion-exclusion principle for two sets?**
$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.

> **Saying it out loud.** If you add the probability of A and the probability of B, you've counted the overlap twice, so subtract it once. That's inclusion-exclusion. The reason it matters practically is that people default to just adding, which overestimates — and if you only need an upper bound, that's the union bound, which is fine and used constantly in learning theory.

**3. Define conditional probability.**
$\mathbb{P}(A|B) = \mathbb{P}(A \cap B)/\mathbb{P}(B)$ for $\mathbb{P}(B) > 0$.

> **Saying it out loud.** Conditioning means restricting your attention to the world where B happened, and rescaling so that world sums to one again. So you take the probability of both happening and divide by the probability of B. The thing to flag is that it's undefined when B has probability zero, which sounds pedantic until you're conditioning on a continuous variable and need densities instead.

**4. State Bayes' theorem.**
$\mathbb{P}(A|B) = \mathbb{P}(B|A)\mathbb{P}(A)/\mathbb{P}(B)$.

> **Saying it out loud.** Bayes' rule is how you flip a conditional around. You know how likely the evidence is given the hypothesis, and you want how likely the hypothesis is given the evidence, so you multiply by the prior and divide by the total probability of the evidence. In words: posterior is prior times likelihood, normalised. The whole reason it feels counterintuitive is that people drop the prior, and that's the base rate fallacy.

**5. Define independence vs uncorrelated.**
Independent: $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$. Uncorrelated: $\mathrm{Cov}(X,Y) = 0$. Independence ⟹ uncorrelated, but not vice versa (except for jointly Gaussian).

> **Saying it out loud.** Independent means knowing one tells you literally nothing about the other — the joint factorises. Uncorrelated is much weaker: it only says there's no *linear* relationship. Take a symmetric variable and its square; they're completely dependent and perfectly uncorrelated. Independence always implies uncorrelated, never the reverse, except in the jointly Gaussian case where they coincide.

**6. What's the law of total probability?**
For partition $\{B_i\}$: $\mathbb{P}(A) = \sum_i \mathbb{P}(A|B_i)\mathbb{P}(B_i)$.

> **Saying it out loud.** Split the world into cases that cover everything and don't overlap, work out the probability within each case, then average weighted by how likely each case is. That's it. It's the workhorse for anything hierarchical — you're computing the probability of a failure by conditioning on which subsystem broke. It's also the denominator in Bayes' rule, which is where most people first meet it without noticing.

**7. Conditional independence — define.**
$X \perp Y | Z$ iff $p(x, y|z) = p(x|z)p(y|z)$. NOT the same as unconditional independence.

> **Saying it out loud.** Conditional independence means that once you know Z, learning X tells you nothing more about Y. It is genuinely a different thing from plain independence — variables can be dependent overall and independent given a third one, or the reverse. Shoe size and reading ability in kids are correlated, but given age the correlation disappears. That's the entire basis of graphical models, and it's the assumption naive Bayes leans on.

---

## B. Random variables

**8. Define expectation.**
$\mathbb{E}[X] = \sum x p(x)$ or $\int x f(x) dx$.

> **Saying it out loud.** Expectation is the long-run average — the centre of mass of the distribution. Discrete, you sum value times probability; continuous, you integrate value times density. The caveat worth having ready is that it doesn't always exist: heavy-tailed things like Cauchy have no finite mean, which is why sample averages of Cauchy data never settle down.

**9. Linearity of expectation — when does it hold?**
Always — even for dependent variables. $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$.

> **Saying it out loud.** Always. That's the remarkable thing about it — the expectation of a sum is the sum of expectations whether or not the variables are independent, correlated, or wildly entangled. That's why you can compute the expected number of matches in a shuffle, or expected loss over a batch, without ever touching the joint distribution. Nothing else in probability is that forgiving, and variance definitely isn't.

**10. Variance formula — two equivalent forms?**
$\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.

> **Saying it out loud.** Variance is the average squared distance from the mean. The computational form — expectation of the square minus the square of the expectation — is the one you use in practice because it's a single pass over the data. The gotcha is that it's numerically unstable when the mean is large relative to the spread; you get catastrophic cancellation, which is why real implementations use Welford's algorithm instead.

**11. Variance of a sum?**
$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y)$.

> **Saying it out loud.** Variance of a sum is the two variances plus twice the covariance. Only when the variables are independent does the covariance vanish and variances simply add. People forget the cross term constantly, and it's exactly the term that matters in portfolio risk, in ensembling, and in why averaging correlated model predictions doesn't reduce error as much as you'd hope.

**12. Covariance formula?**
$\mathrm{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.

> **Saying it out loud.** Covariance is the average product of the deviations from the means, or equivalently expectation of the product minus the product of the expectations. Positive means they tend to move together, negative means opposite. The limitation to name is that it's scale-dependent and only sees linear relationships — that's why you usually report correlation, which is covariance divided by the two standard deviations and lives in the range minus one to one.

**13. Variance of $\bar{X}$ for iid samples?**
$\mathrm{Var}(\bar{X}) = \sigma^2/n$.

> **Saying it out loud.** Sigma squared over n. That's the single most useful formula in applied statistics: the noise in your estimate shrinks with the number of samples, but only as one over the square root of n in standard deviation terms. Practically, that's why cutting your error bar in half costs four times the data — which is exactly the arithmetic behind how long an A/B test has to run.

**14. State the law of total expectation.**
$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$ (tower property).

> **Saying it out loud.** You can average in stages. Average within each group, then average those group averages weighted by group size, and you get the overall average. The average height of a country is the population-weighted average of regional averages. It sounds trivial but it's the standard trick for computing an expectation that's hard directly by conditioning on something that makes it easy.

**15. State the law of total variance.**
$\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X|Y)] + \mathrm{Var}(\mathbb{E}[X|Y])$.

> **Saying it out loud.** Total variance splits into the average variance within groups plus the variance between the group means. That's the formal skeleton of the bias-variance decomposition, and it's also what ANOVA is doing. The practical use is when you have a latent variable — you decompose your uncertainty into how much comes from not knowing the latent state and how much from noise given the state.

---

## C. Common distributions

**16. Bernoulli mean and variance?**
Mean $p$, variance $p(1-p)$.

> **Saying it out loud.** Mean p, variance p times one minus p. The variance formula is worth remembering as a shape: it's zero at both extremes, because a coin that always lands heads has no randomness, and maximal at one half, where it equals 0.25. That maximum-at-a-half fact is why a balanced dataset carries the most information per label, and why entropy peaks there too.

**17. Binomial mean and variance?**
$np$, $np(1-p)$. Sum of $n$ iid Bernoullis.

> **Saying it out loud.** Mean np, variance np times one minus p — just n copies of the Bernoulli, since variances add for independent things. The generating story is what to lead with: n independent flips, count the successes. Almost every count-with-a-fixed-ceiling problem is binomial, and if the ceiling is huge and the rate tiny, it becomes Poisson.

**18. Poisson mean and variance?**
Both $\lambda$. Variance equals mean — Poisson signature.

> **Saying it out loud.** Both are lambda — mean and variance are the same number, which is the Poisson signature. That's diagnostically useful: if you have count data whose variance is much bigger than its mean, it isn't Poisson, it's overdispersed, and you should reach for a negative binomial. That observation is a genuinely good thing to say in an interview because it's what you'd actually do with real data.

**19. When does Binomial → Poisson?**
$n \to \infty$, $p \to 0$, $np = \lambda$ fixed. Used for rare events.

> **Saying it out loud.** When n gets large, p gets small, and their product stays fixed at lambda. Intuitively, you have a huge number of opportunities each of which almost never fires — website requests per second, typos per page, radioactive decays. The binomial's ceiling of n stops mattering because you're never near it. Rule of thumb: n above about 20 and p below about 0.05 and the approximation is already good.

**20. Geometric mean and variance?**
Mean $1/p$, variance $(1-p)/p^2$. Number of trials until first success.

> **Saying it out loud.** Mean one over p, variance one minus p over p squared. The mean is the intuitive part: if something happens one time in six, you wait six trials on average. The thing to notice is how big the variance is — with p small, the spread is roughly as large as the mean itself, so waiting times are wildly unpredictable. That's the same heavy-tail intuition behind why tail latency is so much worse than mean latency.

**21. Exponential mean and variance?**
$1/\lambda$, $1/\lambda^2$.

> **Saying it out loud.** Mean one over lambda, variance one over lambda squared — so standard deviation equals the mean. That equality is the signature of the exponential, and it means waiting times are highly variable: plenty of very short gaps and occasional very long ones. It's the continuous cousin of geometric, and like geometric it's memoryless.

**22. Gaussian — fully specified by what?**
Mean $\mu$ and variance $\sigma^2$. (Multivariate: mean vector and covariance matrix.)

> **Saying it out loud.** Two numbers, mean and variance — that's the whole distribution, nothing else to specify. In the multivariate case it's a mean vector and a covariance matrix. That minimalism is exactly why it's so tractable: every operation you care about maps to simple linear algebra on those two objects, which is what makes Kalman filters and Gaussian processes closed-form.

**23. What's the memoryless property?**
$\mathbb{P}(X > s + t | X > s) = \mathbb{P}(X > t)$. Only geometric (discrete) and exponential (continuous) have it.

> **Saying it out loud.** Memoryless means the past doesn't help you predict the future wait — given you've already waited s, the chance of waiting another t is the same as it was at the start. A bus that arrives as a Poisson process is just as far away after you've waited twenty minutes. Only geometric in discrete time and exponential in continuous time have this, and that uniqueness is the fact interviewers want. It's also why it's a bad model for anything that ages, like machine parts.

**24. Sum of independent Gaussians?**
Gaussian. Means add, variances add.

> **Saying it out loud.** Still Gaussian — means add, variances add, assuming independence. That closure property is a big part of why Gaussians dominate: you can push them through sums and linear maps forever and never leave the family. It's also the reason additive Gaussian noise is such a convenient modelling assumption; the algebra stays closed-form.

**25. Sum of independent Poissons?**
Poisson. Rates add.

> **Saying it out loud.** Poisson again, with the rates added. Intuitively obvious once you think of Poisson as counting events from independent streams — merge two streams of arrivals and you get one stream with the combined rate. It's the fact that makes Poisson processes composable, which is why queueing theory leans on it so heavily.

**26. Beta distribution — what does it model?**
A probability (range $[0,1]$). Conjugate prior for Bernoulli/Binomial.

> **Saying it out loud.** Beta lives on the interval zero to one, so it models a probability — a click-through rate, a conversion rate, a coin's bias. It's the conjugate prior for Bernoulli and binomial, which means after observing successes and failures your posterior is just another Beta with the counts added to the parameters. That's why it's the backbone of Thompson sampling in bandits: updating is literally incrementing two numbers.

**27. Gamma — what does it model?**
Positive continuous quantity. Sum of exponentials. Conjugate for Poisson rate.

> **Saying it out loud.** Gamma is for positive continuous quantities, and its story is the sum of several exponential waits — how long until the k-th event. It's also the conjugate prior for a Poisson rate, which is where it shows up in Bayesian modelling. Practically, it's the go-to prior for anything that must be positive, like a variance or a rate parameter.

---

## D. Multivariate Gaussian

**28. Density of multivariate Gaussian?**
$\mathcal{N}(x|\mu, \Sigma) = (2\pi)^{-d/2} |\Sigma|^{-1/2} \exp(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu))$.

> **Saying it out loud.** It's the bell curve in d dimensions. The exponent is a squared distance from the mean, but measured in units where the covariance stretches and rotates space — that's the Mahalanobis distance. The determinant term out front is just the volume normalisation so it integrates to one. The practical warning is that you need the covariance to be invertible, which fails the moment your features are collinear, and that's a very common real-world crash.

**29. Affine transform of Gaussian?**
$AX + b \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)$.

> **Saying it out loud.** Any linear map of a Gaussian is Gaussian: the mean goes through the map, and the covariance gets sandwiched as A sigma A transpose. That single fact is doing all the work in Kalman filters, in PCA, in Bayesian linear regression. It's also why standardising data is safe — subtracting the mean and dividing by the standard deviation is affine, so you stay Gaussian.

**30. Marginal of multivariate Gaussian?**
Gaussian. Just take the corresponding subvector of $\mu$ and submatrix of $\Sigma$.

> **Saying it out loud.** Marginals are Gaussian, and beautifully, you just read off the relevant piece of the mean vector and the corresponding block of the covariance matrix. No integration required. That's unusual — for most distributions marginalising is a hard integral — and it's a large part of why Gaussian processes are computationally feasible.

**31. Conditional of multivariate Gaussian?**
Gaussian. $X_1|X_2 = x_2 \sim \mathcal{N}(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2), \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})$.

> **Saying it out loud.** Conditioning also stays Gaussian, and the formula is worth recognising because it *is* linear regression. The conditional mean shifts by the covariance between the blocks times the inverse variance of what you observed times how surprising the observation was. And the conditional variance is the original variance minus a positive term, which says observing something can only reduce uncertainty — it never depends on the observed value itself, which is the surprising and Gaussian-specific part.

**32. Uncorrelated jointly Gaussian = independent. True?**
Yes. This is special to Gaussians.

> **Saying it out loud.** Yes, and this is genuinely special to Gaussians. In general zero covariance only rules out a linear relationship. For jointly Gaussian variables it rules out any relationship at all, because the joint density factorises the moment the off-diagonal covariance is zero. The word doing the work is *jointly* — two individually Gaussian, uncorrelated variables can still be dependent.

**33. If $X, Y$ both Gaussian individually, is $(X, Y)$ jointly Gaussian?**
Not necessarily. Marginal Gaussianity doesn't imply joint. (Counterexample: $X \sim \mathcal{N}(0,1)$, $Y = SX$ where $S = \pm 1$ randomly.)

> **Saying it out loud.** No, and this is the trap. You can have X standard normal and Y equal to X times a random sign; Y is also standard normal, and the two are uncorrelated, but they're obviously dependent — the absolute values are identical. So marginal Gaussianity plus zero correlation does not give you independence; you need joint Gaussianity, which is a strictly stronger condition. Having that counterexample ready is a good way to show you know the difference.

---

## E. Limit theorems

**34. State the weak law of large numbers.**
For iid $X_i$ with finite mean $\mu$: $\bar{X}_n \to_p \mu$.

> **Saying it out loud.** The sample average converges in probability to the true mean as you collect more independent samples. Practically: averaging works, and given enough data your estimate is arbitrarily likely to be arbitrarily close. That's the licence for Monte Carlo, for minibatch loss estimates, for A/B testing. The condition is a finite mean — with Cauchy data the average never settles anywhere.

**35. State the central limit theorem.**
For iid $X_i$ with mean $\mu$, finite variance $\sigma^2$: $\sqrt{n}(\bar{X}_n - \mu) \to \mathcal{N}(0, \sigma^2)$.

> **Saying it out loud.** Not only does the sample mean converge, but the error is approximately Gaussian, shrinking as one over root n. So you scale the deviation by root n and it converges in distribution to a normal with the original variance. That's why you can put a confidence interval on any average without knowing what distribution the data came from. The requirement is finite variance, and that's the condition that actually gets violated in practice.

**36. When does CLT fail?**
Infinite variance (heavy tails like Cauchy). Strongly dependent data without mixing conditions.

> **Saying it out loud.** Two ways. If the variance is infinite — Cauchy, or a power law with a heavy enough tail, which shows up in financial returns and network traffic — there's nothing to converge to a fixed-scale Gaussian. And if the data are strongly dependent without any mixing, the effective sample size is far below n so the theorem doesn't bite. That second one is the common practical failure: time-series data treated as iid gives you confidence intervals that are far too narrow.

**37. CLT convergence rate?**
Berry-Esseen: $O(1/\sqrt{n})$, with constant depending on third moment. Skewed distributions need larger $n$.

> **Saying it out loud.** Berry-Esseen says the error in the normal approximation shrinks like one over root n, with a constant that depends on the third absolute moment — so skewness is what slows you down. Practically, symmetric distributions converge fast, maybe n of 30 is fine, while heavily skewed ones might need thousands. That's why the old rule of thumb about n equals 30 is a bad rule: it's fine for a uniform and terrible for a lognormal.

**38. Why is Gaussian everywhere in stats?**
CLT — sums of many small effects approach Gaussian. So sample means, regression residuals, etc. tend to be approximately Gaussian.

> **Saying it out loud.** Because so many quantities are sums of many small independent effects, and the CLT says such sums go Gaussian regardless of what the pieces looked like. Measurement error, sample means, regression residuals, aggregate noise in a model — all of them get bell-shaped for that reason. The caution to add is that it applies to the *aggregate*, not to the raw data, and assuming your individual observations are Gaussian is a much stronger and often wrong claim.

---

## F. Bayes applications

**39. Disease prevalence 1%, test sensitivity 99%, specificity 99%. P(disease | positive)?**
$\mathbb{P}(D|+) = 0.99 \cdot 0.01 / (0.99 \cdot 0.01 + 0.01 \cdot 0.99) = 0.5$. Even 99% accurate tests give only 50% probability for 1% prevalence.

> **Saying it out loud.** Fifty percent, which surprises people. With a 1 percent prevalence and a 99 percent accurate test, out of ten thousand people a hundred have the disease and 99 of them test positive, while 9,900 are healthy and 99 of them also test positive. So a positive result is a coin flip. That's the base rate fallacy in one calculation, and it's exactly why rare-event classifiers have terrible precision even at high accuracy.

**40. What's the base rate fallacy?**
Ignoring prior probability when interpreting test results. The classic Bayesian error.

> **Saying it out loud.** It's forgetting how rare the thing was to begin with. People hear "99 percent accurate test" and jump to "99 percent chance I have it," but if the condition is rare, the false positives from the huge healthy population swamp the true positives. In ML terms it's why you report precision and recall rather than accuracy for imbalanced problems — a model that always predicts "no fraud" is 99.9 percent accurate and completely useless.

**41. What's naive Bayes' assumption?**
Features conditionally independent given class: $\mathbb{P}(x|c) = \prod_j \mathbb{P}(x_j|c)$.

> **Saying it out loud.** That the features are independent of each other once you know the class — so you can multiply the per-feature likelihoods instead of modelling their joint distribution. That's what makes it trainable from tiny data: you only need to estimate one number per feature per class rather than a full joint. And it's obviously false for text, where words co-occur constantly.

**42. Why does naive Bayes work despite the assumption being wrong?**
Need only correct relative ordering of class probabilities; absolute values can be miscalibrated.

> **Saying it out loud.** Because you only need the ranking of classes to be right, not the actual probabilities. Correlated features cause the same evidence to be counted multiple times, which pushes the scores toward zero and one, but it usually doesn't flip which class is on top. So classification accuracy holds up while calibration falls apart. The practical rule is: use naive Bayes for the argmax, never trust its confidence numbers.

**43. Sequential Bayes update — what happens to posterior after multiple iid observations?**
Posterior after $n$ observations = prior × likelihood$^n$ = repeatedly applying Bayes one observation at a time.

> **Saying it out loud.** You just multiply in each new likelihood, and yesterday's posterior becomes today's prior. After n iid observations the posterior is the prior times the product of the individual likelihoods, and crucially you get the same answer whether you process the data one at a time or all at once. That order-independence is what makes Bayesian methods natural for streaming and online learning.

---

## G. Calculations to do fast

**44. $X \sim \mathrm{Uniform}(0,1)$. $\mathbb{E}[X^2]$?**
$\int_0^1 x^2 dx = 1/3$.

> **Saying it out loud.** Integrate x squared from zero to one and you get one third. The quicker route to say out loud is variance plus mean squared: a uniform on zero to one has variance one twelfth and mean one half, so one twelfth plus one quarter is one third. Having both routes ready is useful, because the second one generalises to distributions you can't integrate in your head.

**45. $X \sim \mathrm{Exp}(\lambda)$. $\mathbb{E}[X^2]$?**
$\mathrm{Var}(X) + \mathbb{E}[X]^2 = 1/\lambda^2 + 1/\lambda^2 = 2/\lambda^2$.

> **Saying it out loud.** Use the identity — the second moment is the variance plus the mean squared. For an exponential both of those are one over lambda squared, so the answer is two over lambda squared. That reflex, second moment equals variance plus mean squared, is the single most useful algebraic move in a quant-style interview; you almost never actually integrate.

**46. $\mathbb{E}[\max(X, 0)]$ for $X \sim \mathcal{N}(0, \sigma^2)$?**
$\sigma/\sqrt{2\pi}$. (Half-normal mean.)

> **Saying it out loud.** Sigma over root two pi. The reasoning to say is: half the time X is negative and contributes zero, half the time it's positive, and the average of a standard normal conditional on being positive is root two over pi, so you halve that and get one over root two pi. It's the half-normal mean. It also shows up in finance as the expected payoff of an at-the-money option, which is a nice connection to mention.

**47. Variance of sum of $n$ iid Bernoulli($p$)?**
$np(1-p)$.

> **Saying it out loud.** n times p times one minus p — that's just the binomial variance, because the sum of n iid Bernoullis *is* a binomial and variances add for independent variables. The one-line justification is what they're testing: independence lets variances add. If the trials were correlated you'd need the covariance terms and the answer would be bigger.

**48. Roll a fair die until you get a 6. Expected number of rolls?**
$1/p = 6$. (Geometric distribution.)

> **Saying it out loud.** Six. It's geometric with success probability one sixth, and the mean of a geometric is one over p. The intuition to state is that if something happens one time in six, you wait six trials on average — no calculation needed. And worth adding: the variance is large, thirty, so a standard deviation of about 5.5, meaning it's entirely normal to wait fifteen rolls.

**49. Two iid uniform $(0,1)$. $\mathbb{P}(\max > 0.5)$?**
$1 - (0.5)^2 = 0.75$. Or $\mathbb{P}(\text{both} \leq 0.5) = 0.25$.

> **Saying it out loud.** Three quarters. The clean way is to flip it: the maximum exceeds 0.5 unless both are below 0.5, and that's a half times a half, so a quarter — leaving three quarters. Complementing the event is the move worth showing, because maxima and minima almost always get easier when you flip to "all of them" or "none of them."

**50. $X, Y$ iid $\mathcal{N}(0, 1)$. Distribution of $X^2 + Y^2$?**
$\chi^2_2$ = Exp(1/2). $\mathbb{E}[X^2 + Y^2] = 2$.

> **Saying it out loud.** Chi-squared with two degrees of freedom, which is the same thing as an exponential with rate one half, so the mean is two. Sum of k squared standard normals is chi-squared with k degrees of freedom, and the special coincidence at k equals two is that it collapses to an exponential. Geometrically it's the squared distance from the origin of a 2D standard Gaussian, which is why it's how you sample Gaussian points in the Box-Muller method.

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
