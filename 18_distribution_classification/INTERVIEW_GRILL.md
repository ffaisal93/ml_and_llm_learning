# Distribution Classification — Interview Grill

> 40 questions on choosing distributions, exponential family, GLMs, canonical links. Drill until you can answer 28+ cold.

---

## A. Picking distributions

**1. CTR data — what distribution?**
Per-impression: Bernoulli($p$). Aggregated: Binomial($n, p$). Conjugate prior: Beta.

> **Saying it out loud.** Each impression is a coin flip that either converts or doesn't, so per impression it's Bernoulli. Aggregate a bunch of impressions with the same underlying rate and it's binomial. And because you're estimating a probability, the natural prior is Beta, which is conjugate, meaning updating is literally adding your successes and failures to the two parameters. That conjugacy is why Beta-Bernoulli is the standard setup for Thompson sampling and for smoothing low-traffic CTR estimates.

**2. Number of website visits per hour?**
Poisson($\lambda$) if rare, independent. If overdispersed: Negative Binomial.

> **Saying it out loud.** Poisson is the default for counts in a fixed window — lots of independent chances, each unlikely to fire. But check the assumption before you commit: Poisson forces variance to equal mean, and web traffic almost never obeys that because it has time-of-day and day-of-week structure. If your variance is several times the mean, switch to negative binomial. That check takes one line of code and it's the difference between honest and dishonest confidence intervals.

**3. Time between two events?**
Exponential. (Also assumes memoryless; if hazard varies, use Weibull.)

> **Saying it out loud.** Exponential, if the process is memoryless — meaning the chance of the next event in the next second doesn't depend on how long you've already waited. That's exactly the inter-arrival time of a Poisson process. If the hazard rate actually changes with age, which it does for machine parts and for customer churn, exponential is wrong and you want Weibull, which has a shape parameter that lets the risk rise or fall over time.

**4. User revenue (heavy right tail)?**
Lognormal usually. Or Gamma. Be careful — the sample mean can be misleading.

> **Saying it out loud.** Revenue per user is almost always lognormal or something similarly heavy-tailed — a small fraction of users generate most of the total. The warning to give is that the sample mean of heavy-tailed data is unstable: it's dominated by whichever whales happened to land in your sample, so it jumps around between samples and your confidence intervals lie to you. Report medians and quantiles alongside the mean, or model on the log scale where things are roughly Gaussian again.

**5. Time until $k$-th event?**
Gamma (sum of $k$ iid Exponentials).

> **Saying it out loud.** Gamma, because the time until the k-th event is just k independent exponential waits added together. That's the generating story, and stating it is better than reciting the density. It's also why gamma is the natural prior for a rate parameter, and why it shows up everywhere in queueing and reliability work.

**6. Probability of conversion (each user has its own rate)?**
Beta on the rates; Bernoulli on outcomes given rates.

> **Saying it out loud.** This is a hierarchical model. Each user has their own conversion rate drawn from a Beta across the population, and then each individual visit is a Bernoulli given that user's rate. Marginalising the rate out gives you a Beta-binomial, which has more variance than a plain binomial — that extra variance is exactly the user-to-user heterogeneity. It's the right structure whenever you have repeated measurements on the same entities, and ignoring it is how you get overconfident per-user estimates from three observations.

**7. Class label out of $K$ options?**
Categorical($p_1, \ldots, p_K$).

> **Saying it out loud.** Categorical — a single draw from K options with probabilities summing to one. It's the multi-way generalisation of Bernoulli. Worth naming that a softmax output layer is literally parameterising a categorical distribution, which is why cross-entropy is its negative log likelihood rather than an arbitrary choice of loss.

**8. Word counts in a document?**
Multinomial (or per-word categorical). Topic mixture: Dirichlet prior.

> **Saying it out loud.** Multinomial if you're counting words in a fixed-length document, or equivalently a categorical per word position. That's the bag-of-words model. The natural prior over the topic mixture is Dirichlet, which is the multi-dimensional Beta, and that's exactly what LDA is built on. The known weakness is that real documents are bursty — once a word appears it's much more likely to appear again — which multinomial doesn't capture, and Dirichlet-multinomial is the fix.

**9. Number of trials until first success?**
Geometric.

> **Saying it out loud.** Geometric — the number of trials until the first success. Mean is one over p, so if something works one time in six you wait six trials. The thing worth adding is that the variance is large relative to the mean, so waiting times are wildly unpredictable, and it's the discrete cousin of the exponential, sharing the memoryless property.

**10. Sum of small random effects?**
Gaussian (CLT).

> **Saying it out loud.** Gaussian, by the central limit theorem — a sum of many small independent effects goes bell-shaped regardless of what the individual effects looked like. That's why measurement error and aggregate noise are usually well modelled as Gaussian. The condition to state is finite variance; if the individual effects are heavy-tailed, the sum stays heavy-tailed and you never get the bell curve.

**11. Stock returns?**
Heavy-tailed: Student-t or Cauchy-ish. Empirically NOT Gaussian.

> **Saying it out loud.** Definitely not Gaussian — that's the whole point of the question. Empirically returns have fat tails and volatility clustering, so Student-t with a low degrees-of-freedom is the standard simple fix, and GARCH-type models handle the time-varying volatility. The concrete failure is that a Gaussian model says a five-sigma daily move happens once every few thousand years, and markets deliver several per decade. That mispricing of tail risk is the sentence that lands.

**12. Income distribution?**
Pareto / Lognormal. Heavy right tail.

> **Saying it out loud.** Lognormal through most of the range and Pareto in the upper tail — that's the classic empirical finding. The consequence is that the mean is not a good summary: a small fraction of the population holds a large share of the total, so median income and mean income tell you very different stories. Any time you see that gap, you're looking at a heavy tail and you should stop trusting the average.

---

## B. Distribution properties

**13. When does Binomial ≈ Poisson?**
$n$ large, $p$ small, $np = \lambda$ fixed.

> **Saying it out loud.** When n is big, p is small, and the product stays fixed at lambda. Intuitively you have a huge number of chances each of which basically never fires. The practical rule of thumb is n above about 20 and p below about 0.05. It's why counting rare events — defects, arrivals, mutations — is Poisson territory even though the true process is binomial.

**14. When does Binomial ≈ Gaussian?**
$n$ large, $p$ not near 0 or 1. CLT applies.

> **Saying it out loud.** When n is large and p isn't jammed against zero or one — the usual rule is that both np and n times one minus p exceed about 10. That's just the central limit theorem applied to a sum of Bernoullis. The failure case is a rare-event binomial, where the distribution stays skewed and a Gaussian approximation happily assigns probability to negative counts.

**15. Poisson signature?**
Variance equals mean.

> **Saying it out loud.** Variance equals mean — one parameter doing both jobs. That's the signature, and it's also a testable constraint, which is unusual and useful. Compute both from your data and if the variance is three or five times the mean, Poisson is the wrong model and you're looking at overdispersion.

**16. What's overdispersion?**
Observed variance much larger than mean (when Poisson would predict equality). Suggests Negative Binomial or hierarchical Poisson.

> **Saying it out loud.** Overdispersion means your counts vary more than Poisson allows. It usually happens because there's heterogeneity the model can't see — different users, different hours, different machines each with their own rate. The fix is negative binomial, which is a Poisson whose rate is itself gamma-distributed, giving you a second parameter to soak up the extra spread. The damage from ignoring it is specific: point estimates are roughly fine but standard errors come out far too small, so you overstate significance.

**17. What's underdispersion?**
Variance less than mean. Rare; can use truncated/conditional models.

> **Saying it out loud.** Variance below the mean, which is rare in practice because most real-world heterogeneity pushes the other way. When you do see it, it usually means the counts are constrained somehow — a quota, a scheduled process, a capacity limit — so the events aren't independent. Conway-Maxwell-Poisson or a truncated model handles it. Mostly it's worth knowing so you recognise it, not because you'll fit one next week.

**18. Memoryless distributions?**
Exponential (continuous), Geometric (discrete). Only ones.

> **Saying it out loud.** Exponential in continuous time and geometric in discrete time, and they're the only two. Memoryless means the wait you've already done tells you nothing about the wait remaining. That uniqueness is the fact interviewers are checking. And the practical consequence is that they're bad models for anything that ages or wears out, which is most physical systems.

**19. Conjugate prior table — Bernoulli?**
Beta.

> **Saying it out loud.** Beta. And the update is beautifully concrete: your posterior is Beta with alpha plus the number of successes and beta plus the number of failures. That's why you can run a Bayesian bandit by keeping two integers per arm. It also gives you a principled version of add-one smoothing, since a Beta(1,1) prior is exactly Laplace smoothing.

**20. Conjugate prior — Poisson?**
Gamma.

> **Saying it out loud.** Gamma. Same story as Beta-Bernoulli: the posterior is a Gamma with the shape incremented by the total count and the rate by the observation time. That's why Gamma is the standard prior for any rate parameter, and it's the ingredient that turns Poisson into negative binomial when you marginalise it out.

**21. Conjugate prior — Multinomial?**
Dirichlet.

> **Saying it out loud.** Dirichlet — the multivariate generalisation of Beta, living on the simplex where all the probabilities sum to one. Updating is again just adding the observed counts to the parameters. It's the prior over topic mixtures in LDA, and it's the formal justification for additive smoothing of word probabilities.

**22. Conjugate prior — Gaussian (mean only)?**
Gaussian.

> **Saying it out loud.** A Gaussian — with known variance, the conjugate prior for the mean is another Gaussian, and the posterior mean is a precision-weighted average of the prior mean and the sample mean. That weighting is the useful intuition: the more data you have, the more the posterior slides toward the sample mean. And it's exactly shrinkage, which is what ridge regression is doing under a Bayesian reading.

---

## C. Exponential family

**23. Exponential family form?**
$p(x|\theta) = h(x) \exp(\eta(\theta)^\top T(x) - A(\theta))$.

> **Saying it out loud.** The template is: some base function of the data, times an exponential of the natural parameter dotted with a sufficient statistic, minus a normalising term. Fill in those four slots and you're in the family. The reason to memorise the shape rather than the derivation is that once you recognise a distribution fits it, you know the sufficient statistic, the MLE rule, and the conjugate prior immediately.

**24. What's the natural parameter for Bernoulli?**
$\eta = \log\frac{p}{1-p}$ (logit).

> **Saying it out loud.** The logit — log of p over one minus p, which is the log-odds. That's not a coincidence, it's exactly why logistic regression uses that link: the linear predictor is modelling the natural parameter directly. So sigmoid isn't an arbitrary squashing function, it's the inverse of Bernoulli's natural parameterisation.

**25. What's the natural parameter for Poisson?**
$\eta = \log \lambda$.

> **Saying it out loud.** Log lambda. Which is precisely why Poisson regression uses a log link and exponentiates the linear predictor — it guarantees a positive rate and it's the canonical parameterisation. It also means your coefficients are multiplicative: a coefficient of 0.7 means the rate multiplies by about two per unit of that feature.

**26. What's the natural parameter for Gaussian (variance known)?**
$\eta = \mu/\sigma^2$.

> **Saying it out loud.** Mu over sigma squared — the mean scaled by the precision. With the variance known, the sufficient statistic is just the sum of the data, which is why the sample mean is all you ever need to fit a Gaussian mean. That's the cleanest example of sufficiency there is.

**27. What's a sufficient statistic?**
$T(x)$ such that $p(\theta | x) = p(\theta | T(x))$ — captures all info about $\theta$ in the data.

> **Saying it out loud.** A sufficient statistic is a summary of the data that contains everything relevant about the parameter — once you have it, the raw data tells you nothing more. For a Bernoulli it's the count of successes; you can throw away the order of the flips. It's a genuine data-compression statement, and in the exponential family it's always a simple sum, which is what makes streaming and distributed fitting easy.

**28. Why does exponential family give clean MLE?**
$\nabla A(\theta) = \mathbb{E}[T(X)]$. MLE matches expected sufficient statistics to empirical: $\bar{T}_{\mathrm{data}} = \mathbb{E}_\theta[T(X)]$.

> **Saying it out loud.** Because the gradient of the log-normaliser is exactly the expected sufficient statistic. So setting the derivative of the likelihood to zero gives you a rule you can state in words: choose the parameter that makes the model's expected statistic equal the empirical average. Fit a Gaussian and that says the fitted mean equals the sample mean. It's why exponential-family fitting is a moment-matching exercise rather than a numerical search.

**29. Why does exponential family always have a conjugate prior?**
Multiplication of likelihood by a prior of the same exponential form gives another exp-family distribution; closed-form posterior.

> **Saying it out loud.** Because if you pick a prior of the same exponential shape, multiplying it by the likelihood just adds the exponents, so the posterior is in the same family with updated parameters. The parameters end up being pseudo-counts you add your real counts to. That's why Bayesian updating in these models is arithmetic rather than integration — and it's the practical reason conjugacy was so important before MCMC existed.

---

## D. GLMs

**30. Three components of a GLM?**
Random component (exp-family distribution), systematic component (linear predictor $\eta = w^\top x$), link function $g(\mu) = \eta$.

> **Saying it out loud.** A distribution for the noise, a linear predictor, and a link that connects them. So you say: my target is Poisson, my linear predictor is weights dotted with features, and the log of the mean equals that predictor. Every model in this family — linear, logistic, Poisson regression — is those three choices. Naming all three explicitly is what the question is testing.

**31. What's the canonical link?**
Link function such that $\eta$ equals the natural parameter of the distribution.

> **Saying it out loud.** The canonical link is the one that makes the linear predictor equal the distribution's natural parameter. When you use it, the math simplifies dramatically — the gradient becomes just the residual times the features, the same form as ordinary least squares, and the likelihood is concave so there's a unique optimum. That's the payoff, and it's why logistic and Poisson regression are so well behaved.

**32. Canonical link for Gaussian?**
Identity. Linear regression.

> **Saying it out loud.** Identity — you just model the mean directly, and you're back to ordinary linear regression. It's worth saying explicitly, because people expect a fancy answer and the point of the GLM framework is that linear regression is the trivial member of it. Gaussian plus identity link plus maximum likelihood equals least squares.

**33. Canonical link for Bernoulli?**
Logit. Logistic regression.

> **Saying it out loud.** The logit, log-odds, and its inverse is the sigmoid. So logistic regression is: Bernoulli noise, linear predictor, logit link. And because it's canonical, the gradient is the residual times the features, which is exactly the update rule everyone memorises without knowing where it came from.

**34. Canonical link for Multinomial?**
Multi-logit (softmax inverse). Multi-class logistic regression.

> **Saying it out loud.** Multi-logit, whose inverse is the softmax. That's the whole reason softmax and cross-entropy belong together — softmax is the inverse canonical link of the multinomial, and cross-entropy is its negative log likelihood. So the gradient at the logits is predicted minus actual, which is the same clean residual form again. That fact is worth stating because it explains why that particular pairing trains stably and other pairings don't.

**35. Canonical link for Poisson?**
Log. Poisson regression.

> **Saying it out loud.** Log, with exponential as its inverse. That's what keeps the predicted rate positive, and it makes the coefficients multiplicative rather than additive. It's the default for count regression, and the caveat to attach is that it inherits Poisson's variance-equals-mean assumption, so check for overdispersion before you trust the standard errors.

**36. Why is the canonical link special?**
Score function is $\nabla \ell = \sum (y_i - \mu_i) x_i$ — clean, like OLS residuals. Asymptotic theory simplest.

> **Saying it out loud.** Because with the canonical link the score function collapses to sum of residual times feature — identical in form to ordinary least squares. That means the likelihood is concave with a unique maximum, the Fisher information is clean, and the asymptotic theory is simplest. Practically, it's why these models converge reliably with Newton's method in a handful of iterations, while non-canonical links can be fussier.

**37. Logistic regression as GLM — random/systematic/link?**
Random: Bernoulli($\mu_i$). Systematic: $\eta_i = w^\top x_i$. Link: $g(\mu) = \log\frac{\mu}{1-\mu}$ (logit). Inverse link: sigmoid.

> **Saying it out loud.** Random component: the outcome is Bernoulli with mean mu. Systematic component: the linear predictor is weights dotted with features. Link: the logit of mu equals that predictor, so mu is the sigmoid of it. Saying it in those three pieces beats saying "it's sigmoid of w dot x" because it shows you know where the loss function comes from, not just the forward pass.

**38. Connection between cross-entropy loss and GLM?**
CE for binary classification = NLL of Bernoulli GLM. CE for multi-class = NLL of multinomial GLM with softmax canonical link.

> **Saying it out loud.** They're the same thing. Cross-entropy is the negative log likelihood of the Bernoulli or multinomial model, so minimising cross-entropy *is* maximum likelihood for a GLM whose output layer is the inverse canonical link. That's why the gradient at the logits is just predicted minus actual, and why any other pairing of output activation and loss gives you a messier gradient. If someone asks why not squared error on softmax outputs, this is the answer — you break the canonical pairing and get vanishing gradients when the model is confidently wrong.

**39. Can you do GLM with a non-canonical link?**
Yes — e.g., probit link for Bernoulli (uses Gaussian CDF instead of logit). Loses some of the clean asymptotic properties but sometimes preferred.

> **Saying it out loud.** Yes. Probit uses the Gaussian CDF instead of the logistic, and complementary log-log is common for rare events with asymmetric behaviour. You lose the tidy gradient and some of the asymptotic convenience, and the fit becomes slightly harder numerically. In practice logit and probit give nearly identical predictions, so the choice is usually about interpretability — logit gives you odds ratios, probit gives you a latent-normal story.

---

## E. Heavy tails

**40. What's a heavy-tailed distribution?**
Tail decays slower than exponential. Examples: Pareto, Cauchy, lognormal, Student-t.

> **Saying it out loud.** A tail that decays slower than an exponential, which means extreme values are far more common than intuition or a Gaussian would suggest. Pareto, lognormal, Cauchy, Student-t are the standard examples. The practical consequence is that a single observation can dominate your sample mean, so the average is unstable and no amount of extra data fully fixes it.

**41. Why does CLT fail for Cauchy?**
Infinite variance. Sample mean of iid Cauchys is *also* Cauchy — no concentration.

> **Saying it out loud.** Because the Cauchy has infinite variance — the CLT's precondition fails. What's striking is what happens instead: the average of n iid Cauchys is itself exactly Cauchy, with the same spread. So averaging a million samples gives you no more precision than one sample. It's the cleanest possible counterexample to the intuition that averaging always helps.

**42. Pareto with $\alpha < 2$ — what's the issue?**
Infinite variance. Sample variance fluctuates wildly, doesn't stabilize.

> **Saying it out loud.** Infinite variance, so anything built on variance is meaningless — the sample variance doesn't converge, it just keeps jumping upward as new extremes arrive. Confidence intervals from the CLT are invalid. This isn't exotic; measured alphas for wealth, city sizes, and network traffic are often in that range. Report quantiles instead.

**43. Pareto with $\alpha < 1$ — what's the issue?**
Infinite mean. Sample mean has no limit; new extremes keep dominating.

> **Saying it out loud.** Then even the mean is infinite, and the sample average simply doesn't converge — it drifts upward forever as bigger and bigger observations show up. So asking for the average is asking a question with no answer. Any dashboard reporting a mean for such data is reporting an artifact of the sample size.

**44. How do you handle heavy-tailed data?**
Log-transform, use median/quantiles instead of mean, robust statistics, distributional models that capture tails (Student-t, Pareto).

> **Saying it out loud.** Four moves. Log-transform, which turns lognormal into Gaussian and lets standard methods work. Switch to medians and quantiles, which are stable regardless of tail weight. Use robust statistics like trimmed means if you must have a central estimate. Or model the tail explicitly with Student-t or Pareto if the extremes are what you actually care about. The one thing not to do is clip outliers and pretend it's Gaussian, because in heavy-tailed data the outliers are the signal.

---

## F. Practical decisions

**45. You're modeling defects per unit and see Var(defects) >> Mean(defects). What's the issue and fix?**
Overdispersion. Poisson is too restrictive. Use Negative Binomial regression.

> **Saying it out loud.** That's overdispersion — Poisson assumes variance equals mean, and your data says otherwise, so the model is too rigid. The likely cause is unobserved heterogeneity: different units genuinely have different defect rates. Switch to negative binomial regression, which adds a dispersion parameter. If you don't, your coefficient estimates will be roughly okay but your standard errors will be far too small and you'll call noise a result.

**46. You're modeling time-to-failure of components, but failure rate increases with age (not memoryless). What distribution?**
Weibull. (Exponential = memoryless = constant hazard rate.)

> **Saying it out loud.** Weibull. The exponential's defining property is a constant hazard rate — memorylessness — so it can't represent wear-out at all. Weibull adds a shape parameter: above one and the failure rate rises with age, below one it falls, which is the infant-mortality pattern. That single parameter is why it's the standard in reliability engineering, and naming what the shape parameter means is what makes the answer land.

**47. You want to model the probability of conversion for each user as a random variable across users.**
Beta-distributed conversion rates; Bernoulli outcomes given the rate. This is hierarchical Bayes / random-effects.

> **Saying it out loud.** That's a hierarchical or random-effects model: rates drawn from a Beta across users, Bernoulli outcomes given each user's rate. The payoff is shrinkage — a user with one conversion out of two doesn't get a 50 percent estimate, it gets pulled toward the population average in proportion to how little data you have. That's exactly the right behaviour and it's what a naive per-user rate gets catastrophically wrong on small samples.

**48. Your regression target is non-negative skewed. Linear regression gives negative predictions.**
Switch to GLM with log link (Gamma or Poisson regression). Or transform target with $\log(y+1)$.

> **Saying it out loud.** Linear regression on a non-negative skewed target is the wrong model twice over — wrong support and wrong noise assumption — which is why you're getting negative predictions. Move to a GLM with a log link, gamma for continuous positive data or Poisson for counts, so the predictions are positive by construction. Log-transforming the target also works but changes what you're estimating: you get the geometric mean back, not the arithmetic one, and that's a real difference if someone's forecasting totals.

**49. Logistic regression isn't fitting well — what alternatives?**
Probit (Gaussian-CDF link), complementary log-log link, generalized additive model, neural network.

> **Saying it out loud.** First ask whether the problem is the link or the features. Probit and complementary log-log are alternative links, and cloglog in particular is better for very imbalanced outcomes because it's asymmetric. But usually the issue is that the relationship isn't linear in the features, in which case splines or a generalised additive model or gradient boosting will do more than swapping links. The diagnostic is to plot residuals against each feature before you change anything.

**50. You have multinomial data but suspect overdispersion across documents. What's the model?**
Dirichlet-multinomial: marginalize over document-level Dirichlet to get extra variance.

> **Saying it out loud.** Dirichlet-multinomial, sometimes called a compound multinomial. You let each document have its own word-probability vector drawn from a Dirichlet, then generate counts multinomially given that vector, and marginalising the Dirichlet out gives you a distribution with more variance than a plain multinomial. That extra variance is exactly what captures burstiness — the fact that a word which appears once in a document tends to appear again. Plain multinomial systematically underestimates that, and Dirichlet-multinomial is the standard fix.

---

## Quick fire

**51.** *Variance > mean for counts → ?* Negative Binomial.
**52.** *CLT requires?* Finite variance.
**53.** *Bernoulli canonical link?* Logit.
**54.** *Poisson canonical link?* Log.
**55.** *Gaussian canonical link?* Identity.
**56.** *Memoryless continuous?* Exponential.
**57.** *Memoryless discrete?* Geometric.
**58.** *Heavy-tailed examples?* Pareto, lognormal, Cauchy, Student-t.
**59.** *Bernoulli sufficient statistic?* Sum (count of successes).
**60.** *Cross-entropy = MLE of?* Multinomial GLM (canonical link is multi-logit; softmax is its inverse).

---

## Self-grading

If you can't answer 1-15, you can't choose models intelligently. If you can't answer 16-35, you'll get tripped up on GLM/exp-family questions. If you can't answer 36-50, frontier-lab interviews on probabilistic modeling will go past you.

Aim for 40+/60 cold.
