# Picking the Right Distribution — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

A surprisingly common interview question: "I have data of type X — what distribution should I model it with?" Or: "What does GLM canonical link mean and why does it matter?" The right answer pulls from a small set of distributions and one unifying framework — the exponential family.

---

## 1. Decision tree: what distribution fits my data?

| Data type | Generating story | Distribution | When to use |
|---|---|---|---|
| Yes/no, success/fail | Single binary trial | Bernoulli($p$) | Coin flip, click, conversion |
| Count of yes in $n$ trials | Sum of iid Bernoullis | Binomial($n, p$) | $n$ trials with same $p$ |
| Count of rare events in interval | Limit of binomial, $np \to \lambda$ | Poisson($\lambda$) | Visits per hour, defects per unit |
| Time to event | Memoryless, continuous | Exponential($\lambda$) | Inter-arrival times |
| Sum of $k$ exponential waits | Multiple events | Gamma($k, \lambda$) | $k$-th event arrival |
| Time-to-event with hazard rate | More flexible than exponential | Weibull | Survival analysis |
| Continuous, bounded | Symmetric, no info | Beta or Uniform | Probabilities, fractions |
| Continuous, unbounded | Sum of small effects | Gaussian | CLT regime, errors |
| Continuous, positive | Multiplicative noise | Lognormal | Income, gene expression, file sizes |
| Continuous, heavy-tailed | Power law | Pareto, Cauchy, Student-t | Wealth, returns, sizes |
| Categorical (one of $K$) | Single multi-way choice | Categorical($p_1, \ldots, p_K$) | Class labels |
| Counts across categories | $n$ multi-way trials | Multinomial | Word counts in document |
| Probability over simplex | Conjugate to multinomial | Dirichlet | Topic mixture in LDA |
| Multiple counts with overdispersion | Variance > mean | Negative Binomial | Web hits, sequencing reads |
| Time to event, censored | Hazard varies | Cox proportional hazards | Survival with covariates |

### How to think about it

1. **What is the support?** $\{0,1\}$? Non-negative integers? $[0, 1]$? $\mathbb{R}$? $\mathbb{R}_+$?
2. **What's the generating story?** Does my data come from "yes/no trials"? "Time until something"? "Sum of effects"?
3. **Is variance bigger or smaller than mean?** Poisson has Var = Mean. If Var > Mean, use Negative Binomial (overdispersion).
4. **Is the data heavy-tailed?** Many quantities (income, web traffic, returns) are. Gaussian dramatically underestimates extreme events.

> **Saying it out loud.** When someone asks which distribution to use, don't reach for a name — ask three questions instead. What values can the data actually take: zero-or-one, non-negative integers, a number between zero and one, or the whole real line? What's the story that generates it: independent trials, waiting for something, or lots of small effects adding up? And how does the spread compare to the average? That third one is the most useful and least used: Poisson insists variance equals mean, so if your counts have variance far above the mean, you're overdispersed and the right answer is negative binomial. Answer those three and the distribution basically picks itself.

---

## 2. Exponential family — the unifying view

*In plain language:* the exponential family is a template that a surprising number of familiar distributions fit into — Gaussian, Bernoulli, Poisson, gamma, and more. If you can rewrite a distribution in that template, a whole toolbox comes free: you instantly know what to summarise the data with, how to fit it, and what its natural prior is. The formula below is just that template, and the four symbols in it are the four slots you fill in.

A distribution is in the exponential family if its density can be written:

$$
p(x | \theta) = h(x) \exp\big(\eta(\theta)^\top T(x) - A(\theta)\big)
$$

- $\eta$: natural (canonical) parameter.
- $T(x)$: sufficient statistic.
- $A(\theta)$: log-partition / log-normalizer.
- $h(x)$: base measure.

### Why this matters

The exponential family unifies most distributions you see in practice (Bernoulli, Gaussian, Poisson, multinomial, exponential, gamma, beta, ...). They all share remarkable properties:

- **Sufficient statistics**: $T(x_1, \ldots, x_n) = \sum_i T(x_i)$. The MLE depends on data only through these sums.
- **MLE has clean form**: $\nabla A(\theta) = \mathbb{E}[T(X)] = \bar{T}$. Match expected sufficient statistics to empirical.
- **Conjugate priors exist** in closed form for any exponential-family likelihood.
- **GLMs are built on this** — see next section.
- **Variational inference and graphical models** use exp-family heavily.

### Examples in canonical form

**Bernoulli**: $p(x|p) = p^x(1-p)^{1-x} = \exp\big(x \log \tfrac{p}{1-p} + \log(1-p)\big) = \exp\big(\eta x - \log(1 + e^\eta)\big)$. Natural parameter $\eta = \log\frac{p}{1-p}$ (logit), $T(x) = x$, log-partition $A(\eta) = \log(1 + e^\eta) = -\log(1-p)$.

**Gaussian (variance known)**: $p(x|\mu) \propto \exp(\mu x/\sigma^2 - \mu^2/(2\sigma^2))$. Natural parameter $\eta = \mu/\sigma^2$, $T(x) = x$.

**Poisson**: $p(x|\lambda) \propto \exp(x \log \lambda - \lambda)$. Natural parameter $\eta = \log \lambda$, $T(x) = x$.

> **Saying it out loud.** The exponential family is the observation that most of the distributions you actually use — Gaussian, Bernoulli, Poisson, gamma, beta, multinomial — are the same shape wearing different clothes. Once you write one in that form you get several things for nothing: a sufficient statistic, meaning you can throw the raw data away and keep a couple of sums; a clean fitting rule, where the maximum likelihood estimate just matches the model's expected statistic to the empirical average; and a conjugate prior that exists in closed form. The reason to care in an ML interview is that it's the machinery underneath GLMs, underneath variational inference, and underneath why softmax plus cross-entropy is the natural pairing rather than an arbitrary one.

---

## 3. Generalized linear models (GLMs)

*In plain language:* a GLM is linear regression made flexible enough to handle outputs that aren't unbounded real numbers. You keep the linear combination of features, but you pass it through a function that squashes it into the right range — probabilities, counts, positive numbers — and you pick the noise model to match. Logistic regression and Poisson regression are just two entries in this one framework.

A GLM models data as exponential family conditional on a linear predictor:

1. **Random component**: $y_i | x_i \sim$ exponential family, with natural parameter related to mean $\mu_i$.
2. **Systematic component**: $\eta_i = w^\top x_i$.
3. **Link function**: $g(\mu_i) = \eta_i$.

The **canonical link** is the one that makes the natural parameter equal to $\eta_i$. Using it gives clean MLE — gradient is $\sum (y_i - \mu_i) x_i$ (just like linear regression's residual structure).

### Canonical link table

| Distribution | Canonical link | Inverse link | Common name |
|---|---|---|---|
| Gaussian | Identity | Identity | Linear regression |
| Bernoulli | Logit $\log\frac{p}{1-p}$ | Sigmoid | Logistic regression |
| Multinomial | Multi-logit | Softmax | Multi-class logistic regression |
| Poisson | Log | Exp | Poisson regression (count regression) |
| Gamma | Inverse ($-1/\mu$) | $-1/\eta$ | Gamma regression (in practice, log link more common) |
| Negative Binomial | (depends on dispersion; not always practical) | Log used in practice | Overdispersed count regression — log is the *conventional* link, not strictly canonical |
| Exponential | Inverse | Inverse | Hazard models |

### Why this matters in ML

- Classification with cross-entropy loss = MLE of multinomial GLM where multi-logit is the canonical link and softmax is its inverse (so the model outputs probabilities via softmax).
- Squared loss for regression = MLE of Gaussian GLM with identity link.
- The choice of activation + loss in a neural network's output layer is exactly a GLM choice.
- Generalized additive models (GAMs) extend this to non-linear features.

> **Saying it out loud.** A GLM is three pieces: a distribution for the noise, a linear combination of your features, and a link function that connects them. That's it. The link exists because a linear predictor ranges over all the real numbers while your target might be a probability or a count, so you need something to squash it into range — sigmoid for probabilities, exponential for counts. The canonical link is the one that makes the algebra collapse, and when you use it the gradient is always the same beautiful thing: the residual times the feature. The punchline that scores is that your neural network's output layer is a GLM — softmax plus cross-entropy is exactly the multinomial GLM, and squared error plus a linear output is exactly the Gaussian one.

---

## 4. Common modeling pitfalls

### Using Gaussian for everything

Default assumption in many pipelines. Wrong when:
- Data is non-negative (use lognormal, gamma).
- Data is heavy-tailed (use Student-t, Cauchy).
- Data is bounded (use beta, truncated normal).
- Data is count (use Poisson, negative binomial).

> **Saying it out loud.** The commonest modelling sin is reaching for a Gaussian by reflex. It's wrong whenever the support is wrong — a Gaussian puts probability mass on negative numbers, so it's a bad model for durations, revenue, or file sizes, all of which are strictly positive and skewed. It's also wrong when the tails are heavy, because a Gaussian says a five-sigma event basically never happens and in real data it happens all the time. The concrete failure is underestimating extremes: model latency as Gaussian and your p99 predictions will be badly optimistic, which is exactly the number that matters operationally.

### Poisson when variance > mean (overdispersion)

Poisson assumes variance = mean. Real count data often has variance >> mean. Use negative binomial instead.

> **Saying it out loud.** Overdispersion is the classic count-data trap. Poisson has a rigid constraint baked in — variance equals mean, one parameter for both — and real count data almost never obeys it, because there's usually extra variability the model doesn't see. Website hits vary by hour, sequencing reads vary by sample. The symptom is easy to check: compute the mean and the variance and if the variance is several times bigger, Poisson is wrong. The fix is negative binomial, which is Poisson with a gamma-distributed rate, giving you a second parameter to absorb the extra spread. Fitting Poisson to overdispersed data doesn't bias your point estimates much, but it makes your confidence intervals far too narrow, and that's how you ship a wrong conclusion.

### Independence assumption

Naive Bayes assumes feature independence given class. Hierarchical / sequential data violates this. GLMs assume iid given covariates — fails for time series.

> **Saying it out loud.** Independence is the assumption everyone makes and almost nobody checks. Naive Bayes assumes features are independent given the class, GLMs assume observations are independent given the covariates, and standard error formulas assume your samples are independent. Time series, repeated measurements on the same user, and anything with spatial structure all violate this. The consequence is specific and dangerous: dependence inflates your effective sample size, so your standard errors come out too small and you declare significance that isn't there. That's how underpowered A/B tests get shipped as wins.

### Using "the" distribution rather than thinking

Asking "what distribution should I use?" is usually less helpful than:
- "What's the data-generating process?"
- "What's the support?"
- "Are there extreme values? How heavy is the tail?"
- "Is variance comparable to mean?"

> **Saying it out loud.** The question "what distribution should I use" is usually the wrong question, and saying so is a good answer. The better questions are what physically generates this data, what values it can take, how heavy the tail is, and whether the variance is comparable to the mean. Get those and the distribution is a consequence rather than a choice. It also stops you from the common failure of fitting a familiar distribution and never checking — a QQ plot against your fitted model takes thirty seconds and catches most of these mistakes.

---

## 5. Heavy tails — important and overlooked

Many ML problems have heavy-tailed data (Pareto, lognormal, Cauchy). Important consequences:

- **Means are dominated by extremes**: top 1% of users contribute most of the revenue.
- **CLT convergence is slow** for heavy-tailed (or fails entirely for infinite-variance distributions like Cauchy).
- **Sample mean is unstable**; median may be more useful.
- **Log-transforming** can convert lognormal to normal → standard methods apply.

**Pareto**: $p(x) \propto x^{-(\alpha+1)}$ for $x \geq x_{\min}$. $\alpha < 2$ → infinite variance. $\alpha < 1$ → infinite mean.

**Lognormal**: $\log X \sim \mathcal{N}$. Heavy right tail. Common for incomes, sizes, times.

**Cauchy**: $p(x) \propto 1/(1+x^2)$. No mean or variance. Sample mean is just another Cauchy.

> **Saying it out loud.** Heavy tails are the thing that quietly breaks ordinary statistics, and most real business data has them. Revenue per user, web traffic, file sizes, latency — a small fraction of the population accounts for most of the total, so the sample mean is dominated by whichever extreme values happened to land in your sample and jumps around wildly between samples. The technical version is that for a Pareto with alpha below 2 the variance is infinite, so the central limit theorem gives you nothing and your confidence intervals are meaningless; below alpha of 1 even the mean is infinite. Practically: use medians and quantiles instead of means, or log-transform if it's lognormal, which turns it back into something Gaussian methods can handle. The number to end on is that at a heavy enough tail, no amount of extra data makes your average stable.

---

## 6. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| What does Poisson regression model? | Anything with counts | Counts where Var = Mean (use NegBin if overdispersed) |
| Is logistic regression a GLM? | No | Yes — Bernoulli + logit canonical link |
| What's the canonical link for a Gaussian GLM? | Sigmoid | Identity |
| What's the relationship between cross-entropy and GLMs? | Different things | CE = NLL of categorical GLM with softmax |
| Why is Gaussian everywhere? | Tradition | CLT — sums approach Gaussian under finite variance |
| Does Bayes' theorem assume iid? | Yes | No — Bayes is general; iid is an assumption about data, not Bayes |
| Can I use a continuous distribution for count data? | Sure | Bad idea unless you discretize properly — count data has support on $\mathbb{N}_0$ |

---

## 7. Eight most-asked interview questions

1. **What distribution would you use for click-through-rate data and why?** (Bernoulli per impression; Binomial for batched; Beta as conjugate prior.)
2. **You see count data with variance much larger than mean. What model?** (Negative binomial — Poisson is overdispersed here.)
3. **Walk me through GLMs and canonical links.** (Random + systematic + link; canonical = natural parameter = linear predictor.)
4. **What does it mean for cross-entropy to "match" softmax?** (Both come from the multinomial GLM; gradient is clean: $\hat{y} - y$.)
5. **You have user revenue data — what distribution?** (Lognormal usually fits well; or Gamma/heavy-tailed; sample mean can be unreliable.)
6. **What's the exponential family and why do we care?** (Unifies many distributions; closed-form sufficient statistics, MLE, conjugate priors.)
7. **How do you check if Poisson is appropriate?** (Variance ≈ Mean; if Var >> Mean, use NegBin; goodness-of-fit tests.)
8. **You can't fit your data with Gaussian. What do you check?** (Support, skewness, kurtosis, tail behavior; QQ plot vs Gaussian; consider transformations.)

---

## 8. Drill plan

- For each distribution in the decision-tree table, recite: support, generating story, mean/variance, when to use.
- For each canonical link in the GLM table, recite: distribution, inverse link, common name.
- Practice writing 3 distributions in canonical exponential-family form.
- Practice 5 "which distribution" interview problems from real domains: web traffic, financial returns, time-to-failure, conversion rates, click counts.

---

## 9. Further reading

- McCullagh & Nelder, *Generalized Linear Models* — the classic reference.
- Wasserman, *All of Statistics*, ch. 13 — fast GLM intro.
- Dobson & Barnett, *An Introduction to Generalized Linear Models* — accessible.
- Mandelbrot, *The (Mis)behavior of Markets* — heavy-tail intuition for finance.
- Clauset, Shalizi, Newman (2009), *Power-law distributions in empirical data* — how to actually test for power laws.
