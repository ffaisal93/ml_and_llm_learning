# Statistical Inference — Interview Grill

> 50 questions on estimators, MLE, CIs, bootstrap, hypothesis testing, Bayesian inference. Drill until you can answer 35+ cold.

---

## A. Estimators

**1. What's an estimator?**
A function of the data that approximates an unknown parameter: $\hat{\theta} = T(X_1, \ldots, X_n)$.

> **Saying it out loud.** An estimator is just a recipe that turns data into a guess about something you can't observe. The sample mean is an estimator of the true mean; a trained model's accuracy on a test set is an estimator of its accuracy in the world. The key mental move is that the estimator is random — run the experiment again with a different sample and you get a different number — so everything in this topic is about describing that spread. Once you think of your reported metric as one draw from a distribution rather than a fact, the rest of statistics follows naturally.

**2. Define unbiased.**
$\mathbb{E}[\hat{\theta}] = \theta$ — on average across repeated samples, the estimator hits the true value.

> **Saying it out loud.** Unbiased means that if you repeated the whole experiment infinitely many times, the average of your estimates would land exactly on the truth. It's a statement about the long-run average, not about the estimate you actually have — your one estimate can still be miles off. That's why unbiasedness is much less important than people assume, and why it's perfectly reasonable to prefer a biased estimator that's tightly clustered over an unbiased one that's all over the place. Ridge regression is exactly that trade.

**3. Define consistent.**
$\hat{\theta}_n \to_p \theta$ as $n \to \infty$.

> **Saying it out loud.** Consistent means your estimate converges to the truth as you collect more data — get enough samples and you're arbitrarily close, with probability approaching one. It's a large-sample guarantee, and it's the property you actually want, because it says more data eventually solves your problem. Unbiasedness says nothing about that. If I had to choose one, I'd take consistent every time, and most maximum-likelihood estimators are consistent while being slightly biased in finite samples.

**4. Unbiased vs consistent — give an example of one but not the other.**
Sample mean of one observation: unbiased, not consistent. Estimator $\hat{\theta} = X_1 + 1/n$: consistent but biased for any finite $n$.

> **Saying it out loud.** The clean example of unbiased-but-not-consistent is just using your first observation and ignoring the rest — its expectation is the true mean, so it's unbiased, but collecting a million more samples doesn't improve it one bit. Going the other way, take the sample mean and add one over $n$: it's biased at every finite sample size, but the bias shrinks to zero so it converges to the truth. That's the pair that shows the properties are genuinely independent — one is about the average of your estimates, the other is about what happens as data grows.

**5. State the bias-variance decomposition for MSE.**
$\mathrm{MSE}(\hat{\theta}) = \mathrm{Bias}(\hat{\theta})^2 + \mathrm{Var}(\hat{\theta})$. Implication: biased estimators can have lower MSE than unbiased ones.

> **Saying it out loud.** Squared error splits cleanly into bias squared plus variance — how far off you are on average, plus how much you bounce around. The reason this matters is that it licenses trading one for the other, and often a little bias buys a lot of variance reduction. That's the entire justification for regularization: ridge regression is deliberately biased and beats the unbiased least-squares fit on out-of-sample error whenever predictors are correlated or data is scarce. Unbiasedness is a constraint, and constraints cost you something.

**6. What's the Cramér-Rao lower bound?**
$\mathrm{Var}(\hat{\theta}) \geq 1/I(\theta)$ where $I$ is Fisher information. Lower bound on variance for any unbiased estimator.

> **Saying it out loud.** It says there's a floor on how precise any unbiased estimator can be, and that floor is set by how much information each observation carries about the parameter — the Fisher information. Intuitively, Fisher information measures how sharply peaked the likelihood is: a sharp peak means the data strongly pins down the parameter, a flat one means it barely constrains it. What makes it useful is that it tells you when to stop optimizing, because an estimator hitting the bound can't be beaten among unbiased estimators. And note the escape hatch — allow bias and you can go below it, which is exactly what shrinkage estimators do.

**7. Why $n-1$ in sample variance (Bessel's correction)?**
$\frac{1}{n}\sum(x_i - \bar{x})^2$ underestimates $\sigma^2$ because $\bar{x}$ is closer to the data than $\mu$. Dividing by $n-1$ corrects the bias.

> **Saying it out loud.** Because you used the data twice. You estimated the mean from the same sample, and the sample mean sits closer to your data points than the true mean does — by construction it minimizes the squared distances. So the sum of squared deviations comes out systematically too small, and dividing by $n-1$ instead of $n$ corrects exactly that. The degrees-of-freedom framing is the one to use: you spent one degree of freedom estimating the mean, so only $n-1$ remain. It matters at $n=10$ and it's irrelevant at $n=10{,}000$.

---

## B. MLE

**8. Define MLE.**
$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \prod_i p(x_i|\theta) = \arg\max_\theta \sum_i \log p(x_i|\theta)$.

> **Saying it out loud.** Maximum likelihood asks: of all possible parameter values, which one makes the data I actually saw most probable? You write down the probability of your dataset as a function of the parameter and maximize it — in practice you maximize the log, because a product of thousands of small probabilities underflows and a sum doesn't. It's worth saying out loud that this is the same objective as cross-entropy training: minimizing negative log-likelihood is what every neural network is doing, so MLE isn't a separate topic from deep learning.

**9. Derive MLE for Bernoulli.**
$\ell(\theta) = \sum [x_i \log \theta + (1-x_i)\log(1-\theta)]$. Set $\partial \ell / \partial \theta = 0$: $\hat{\theta} = \bar{x}$.

> **Saying it out loud.** Set it up, differentiate, and the answer is just the fraction of ones — which is exactly what your intuition said before you did any calculus. Nine heads out of ten flips gives you 0.9. That's the point of the derivation: it confirms that maximum likelihood recovers the obvious estimator in the obvious case. It also shows the failure mode immediately — zero successes gives you an estimate of exactly zero probability, which is nonsense for a small sample, and that's precisely what a Beta prior fixes by adding pseudo-counts.

**10. Derive MLE for Gaussian (mean and variance).**
$\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$. The MLE for variance is biased — Bessel's correction unbiases it.

> **Saying it out loud.** The mean comes out as the sample mean, no surprises. The variance comes out as the average squared deviation divided by $n$ — note, $n$, not $n-1$ — so the maximum likelihood estimate of variance is biased low. That's not a mistake in the derivation; it's what maximum likelihood genuinely gives you, and Bessel's correction is a separate deliberate adjustment. Good thing to have ready, because interviewers use it to check whether you actually derived it or memorized the formula from a textbook that already corrected it.

**11. Why is MLE biased for variance but consistent?**
Bias is $O(1/n)$ — vanishes as $n \to \infty$. So MLE is consistent but not unbiased in finite samples.

> **Saying it out loud.** The bias is a factor of $(n-1)/n$, so at $n=10$ you're 10 percent low and at $n=1000$ you're a tenth of a percent low — it shrinks to nothing as data grows, which is exactly what consistency means. So the two properties coexist without contradiction: biased at every finite $n$, converging to the truth in the limit. That's the general pattern for maximum likelihood, and it's why people care much more about consistency and asymptotic efficiency than about finite-sample unbiasedness.

**12. Asymptotic properties of MLE?**
Consistent, asymptotically normal: $\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I^{-1})$, asymptotically efficient (achieves CRLB).

> **Saying it out loud.** Three things, and they're why MLE is the default. It's consistent, so it converges to the truth. It's asymptotically normal, so you can build confidence intervals from a Gaussian approximation without knowing anything else about the distribution. And it's asymptotically efficient, meaning it achieves the Cramér-Rao bound — no unbiased estimator does better in the limit. The caveat to attach is that all three are large-sample statements, and in small samples MLE can be badly biased and its normal-approximation intervals can be junk.

**13. Invariance of MLE — what is it?**
If $\hat{\theta}_{\mathrm{MLE}}$ estimates $\theta$, then $g(\hat{\theta}_{\mathrm{MLE}})$ estimates $g(\theta)$. E.g., MLE of $\sigma$ is $\sqrt{\hat{\sigma}^2_{\mathrm{MLE}}}$.

> **Saying it out loud.** If you have the maximum likelihood estimate of a parameter, then the maximum likelihood estimate of any function of it is just that function applied to your estimate. Want the MLE of the standard deviation? Take the square root of the MLE of the variance. It's a genuinely convenient property and it's specific to maximum likelihood — unbiasedness does not survive nonlinear transformation, so the square root of an unbiased variance estimate is not an unbiased estimate of the standard deviation. That asymmetry is a nice detail to have ready.

**14. When does MLE fail?**
Small samples (high variance, biased), unbounded likelihood (e.g., Gaussian mixture with covariance shrinking to a point), non-identifiable models.

> **Saying it out loud.** Three situations. Small samples, where it's both biased and high-variance and its asymptotic guarantees haven't kicked in. Unbounded likelihoods — the classic is fitting a Gaussian mixture where one component collapses onto a single data point, sending its variance to zero and the likelihood to infinity, so the true maximum is a degenerate solution. And non-identifiable models, where different parameters give identical likelihoods, so there's no unique answer. The practical fix for all three is a prior, which is to say MAP estimation, which is to say regularization.

---

## C. Confidence intervals

**15. Define a 95% confidence interval.**
A random interval $[L, U]$ such that under repeated sampling, 95% of intervals contain $\theta$. Frequency interpretation, not "$\theta$ is in [...] with 95% probability."

> **Saying it out loud.** This is the one people get wrong, so be precise. The 95 percent is a property of the procedure, not of your interval: if you repeated the whole study many times and built an interval each time, about 95 percent of those intervals would contain the true value. It is not correct to say there's a 95 percent probability that the parameter is in the interval you computed — in the frequentist picture the parameter is a fixed number and your interval either contains it or doesn't, so that probability is zero or one. If you want the statement people actually want to make, you need a Bayesian credible interval, and you need a prior to get it.

**16. Wald CI formula?**
$\hat{\theta} \pm 1.96 \cdot \mathrm{SE}(\hat{\theta})$ for 95%. Relies on asymptotic normality.

> **Saying it out loud.** Estimate plus or minus 1.96 standard errors, which is the interval you've seen a thousand times. It works because maximum likelihood estimates are asymptotically normal, so you're borrowing a Gaussian shape for a sampling distribution you never actually computed. The word doing all the work is asymptotically — near a boundary or in a small sample the normal approximation is bad, and you get absurdities like a confidence interval for a probability that extends below zero. That's the moment to reach for a Wilson interval or a bootstrap instead.

**17. CI vs credible interval?**
CI is frequentist — interval random, $\theta$ fixed. Credible interval is Bayesian — interval fixed, $\theta$ has posterior probability mass. CredI supports the natural "$\theta$ in [...] with 95% probability" interpretation.

> **Saying it out loud.** They answer different questions and people conflate them constantly. A confidence interval treats the parameter as fixed and the interval as random — the 95 percent describes how often the procedure works across repeated experiments. A credible interval treats the parameter as random and the interval as fixed — so it genuinely does support 'there's a 95 percent probability the value is in here,' conditional on your prior. That's the sentence everyone wants, and the price of it is committing to a prior. In practice, with lots of data and a weak prior, the two intervals nearly coincide, which is why the distinction feels academic until your sample is small.

**18. How do you compute a bootstrap CI?**
Resample data with replacement $B$ times, compute $\hat{\theta}^{(b)}$ each time. CI = $[Q_{0.025}, Q_{0.975}]$ of $\{\hat{\theta}^{(b)}\}$ (percentile method).

> **Saying it out loud.** You fake having many datasets by resampling your one dataset with replacement, thousands of times, recomputing the statistic each time. That collection of recomputed values approximates the sampling distribution, and you just read off the 2.5th and 97.5th percentiles. What makes it so useful is that it needs no formula for the standard error — which is why it's the standard tool for things like AUC or F1, where the analytic sampling distribution is unpleasant. A thousand resamples is typical; ten thousand if you're reporting the numbers.

**19. When can a CI go negative for a positive quantity?**
When CI is constructed without constraints (e.g., Wald CI for a probability close to 0 or 1). Use logit transform or bootstrap.

> **Saying it out loud.** When the normal approximation runs off the end of the parameter's range. If you've got an estimated probability of 0.02 with a decent standard error, symmetric Wald bounds happily extend below zero, which is meaningless. It's telling you the sampling distribution is skewed and the Gaussian approximation doesn't hold near the boundary. The fixes are to work on a transformed scale like the log-odds and map back, or use a Wilson interval, or bootstrap. Seeing a nonsensical bound is a signal, not a rounding artifact.

---

## D. Hypothesis testing

**20. State the components of a hypothesis test.**
Null $H_0$, alternative $H_1$, test statistic $T$, null distribution, rejection region, significance $\alpha$.

> **Saying it out loud.** You need a null hypothesis, an alternative, a test statistic, that statistic's distribution assuming the null is true, and a threshold. The engine of the whole thing is the null distribution — you're asking how weird your observed statistic would be in a world where nothing is going on. Everything else is bookkeeping. It's also where the assumptions hide, because if you got the null distribution wrong, your p-value is meaningless no matter how carefully you computed it.

**21. What's a p-value?**
Probability under $H_0$ of observing a test statistic at least as extreme as the one observed. NOT $\mathbb{P}(H_0 | \mathrm{data})$.

> **Saying it out loud.** A p-value is the probability of seeing data at least as extreme as yours, assuming the null hypothesis is true. Emphasize the direction: it's the probability of the data given the null, never the probability of the null given the data. Those get confused constantly and they can differ by orders of magnitude — if you're testing a hypothesis that's implausible to begin with, a p-value of 0.04 leaves it still probably false. Getting that direction right is the single most common way this topic is tested.

**22. Why is "p < 0.05 means the result is true" wrong?**
$p$-value isn't $\mathbb{P}(H_0 | \mathrm{data})$. With multiple tests, $p < 0.05$ alone is meaningless. Even with one test, low $p$ is "data is unlikely under $H_0$," not "$H_0$ is unlikely."

> **Saying it out loud.** Two reasons. First, the p-value is about the data under the null, not about whether the null is true — flipping that conditional is just wrong. Second, 0.05 means one in twenty tests will clear the bar by pure chance, so the moment you're running many tests, or trying a few analyses until one works, the number stops meaning anything. That's the replication crisis in one sentence. The useful framing to offer is that a p-value tells you the data is surprising under the null, and how much that should move your belief depends entirely on how plausible the hypothesis was beforehand.

**23. Type I vs Type II error?**
Type I: reject true $H_0$ (false positive, controlled by $\alpha$). Type II: fail to reject false $H_0$ (false negative, controlled by power $1-\beta$).

> **Saying it out loud.** Type I is crying wolf — you declare an effect that isn't there — and you control it directly by setting alpha, typically 0.05. Type II is missing a real effect, and you control that indirectly through power, which mostly means sample size. The tension is that they trade off: make alpha stricter and you catch fewer false alarms but miss more real effects. Which one you fear more is a domain question — a medical screening test and an exploratory gene study should not use the same threshold.

**24. What's statistical power?**
$1 - \beta = \mathbb{P}(\mathrm{reject}\, H_0 \mid H_1\, \mathrm{true})$. Depends on effect size, $n$, $\alpha$, variance.

> **Saying it out loud.** Power is the probability that you detect an effect that's really there. Four things drive it: how big the effect is, how much data you have, how strict your alpha is, and how noisy your measurements are. The reason to care is that an underpowered study is worse than no study — you'll fail to find real effects, and worse, the effects you do find will be systematically overstated, since only the unusually large estimates cleared the threshold. Eighty percent power is the conventional target, and a lot of published research sits well below it.

**25. When do you use a t-test vs z-test?**
$z$-test: variance known (rare). $t$-test: variance estimated from sample (almost always).

> **Saying it out loud.** You use a z-test when you somehow know the true variance, which essentially never happens, and a t-test when you estimate it from the data, which is always. The t-distribution has heavier tails to account for the extra uncertainty in that variance estimate. Practically, past about 30 samples the two are nearly identical — at $n = 30$ the critical value is 2.04 versus 1.96 — so the distinction only bites in small samples, which is exactly when people forget it.

**26. When do you use a chi-squared test?**
Goodness-of-fit, contingency tables (test of independence). Categorical data. Statistic: $\chi^2 = \sum (O - E)^2 / E$.

> **Saying it out loud.** When your data is counts in categories. Two flavors: goodness-of-fit, asking whether observed counts match an expected distribution, and the contingency-table test, asking whether two categorical variables are independent. The statistic sums squared differences between observed and expected, scaled by expected. The assumption to remember is that expected counts should be at least around 5 per cell — below that the chi-squared approximation breaks and you want Fisher's exact test instead.

**27. What's a one-sided vs two-sided test?**
One-sided: $H_1: \theta > \theta_0$ (or $<$). Two-sided: $H_1: \theta \neq \theta_0$. One-sided has more power but you must commit to direction *a priori*.

> **Saying it out loud.** A two-sided test asks whether there's any difference; a one-sided test asks whether there's a difference in a specific direction. One-sided gives you more power for the same sample, because you concentrate your whole alpha in one tail. The catch is you must commit to the direction before seeing the data — choosing the direction afterwards is exactly the kind of p-hacking that makes results unreproducible. Practical advice: use two-sided by default, and use one-sided only when a result in the other direction would lead to the same decision as no result at all.

**28. Paired vs unpaired t-test?**
Paired: same subjects measured twice (before/after). Unpaired: independent groups. Paired has more power because it removes between-subject variation.

> **Saying it out loud.** Paired means the same subjects measured twice, so you analyze the differences and every subject serves as their own control. That removes between-subject variation, which is usually the largest noise source, so paired tests are dramatically more powerful — often needing a fraction of the sample size for the same detection ability. This maps directly onto ML: when comparing two models, evaluate both on the exact same test examples and analyze the per-example differences. Comparing independently-sampled results throws away the pairing and wastes statistical power for no reason.

---

## E. Multiple testing

**29. The multiple testing problem?**
With $m$ independent tests at $\alpha = 0.05$, FWER $\approx m\alpha$ for small $\alpha$. Run 20 tests, expect ~1 false positive even with no real effect.

> **Saying it out loud.** If each test has a 5 percent chance of a false positive, then running twenty tests gives you about a 64 percent chance of at least one false alarm even when nothing is real. The false positives aren't a bug in any individual test — each one is behaving exactly as specified — the problem is that you're looking many times and reporting the winner. It's everywhere in ML: every hyperparameter sweep, every subgroup breakdown, every dashboard with fifty metrics on it. Naming that 64 percent number is a good way to make it concrete.

**30. Bonferroni correction?**
Test each at $\alpha/m$ instead of $\alpha$. Controls FWER. Conservative; loses power.

> **Saying it out loud.** Divide your alpha by the number of tests, so twenty tests means each must clear 0.0025. It's the simplest correction and it's guaranteed to control the probability of even one false positive, regardless of how the tests correlate. The cost is power — with a thousand tests your threshold becomes absurd and you'll miss essentially every real effect. So it's right when a single false positive is genuinely unacceptable and wrong for exploratory work, where Benjamini-Hochberg is the better tool.

**31. What's Benjamini-Hochberg?**
Controls false discovery rate (FDR = expected proportion of false positives among rejections). Order p-values; reject the largest $i$ for which $p_{(i)} \leq i\alpha/m$. Less conservative than Bonferroni.

> **Saying it out loud.** BH controls the false discovery rate — the expected fraction of your declared findings that are wrong — rather than the probability of any error at all. Practically: sort your p-values, compare each to a threshold that grows with its rank, and reject everything up to the largest one that passes. The difference in mindset is what matters: Bonferroni says 'I want to be almost certain nothing here is spurious,' BH says 'I'm fine with 5 percent of my hundred findings being junk if I get to keep the other 95.' For any screening or exploratory setting, BH is the right answer.

**32. FWER vs FDR — when each?**
FWER: when any false positive is bad (e.g., medical diagnosis). FDR: when discovery is exploratory and some false positives are tolerable (e.g., gene expression).

> **Saying it out loud.** Control the family-wise error rate when a single false positive is costly — regulatory approval, a medical diagnosis, anything where you act irreversibly on one finding. Control the false discovery rate when you're screening and will follow up anyway, like scanning twenty thousand genes or a large feature set. The framing to give is that FWER protects against being wrong at all, FDR protects against being mostly wrong. Picking the wrong one costs you either every real discovery or all your credibility.

**33. Where does multiple testing show up in ML?**
Hyperparameter sweeps, A/B test farms, feature selection (test each feature), subgroup analysis.

> **Saying it out loud.** Constantly, and usually uncorrected. A hyperparameter sweep with 200 configurations is 200 tests against the validation set, so the best result is inflated by selection — that's exactly why the winner tends to disappoint on the true holdout. Same for feature selection, subgroup analysis, and any team running dozens of A/B tests a quarter. The practical defense isn't usually a formal correction, it's a clean holdout you touch once, and treating the gap between validation and test as expected rather than surprising.

---

## F. Bootstrap

**34. What's the bootstrap?**
Resample data with replacement $B$ times to approximate the sampling distribution of an estimator. Non-parametric, simple, broadly applicable.

> **Saying it out loud.** The bootstrap is a beautifully cheeky idea: you want to know how your estimate would vary across different datasets, you only have one dataset, so you treat your sample as if it were the population and draw new samples from it with replacement. Do that a thousand times, recompute your statistic each time, and the spread of those numbers approximates the real sampling distribution. It requires no formula and no distributional assumption, which is why it's the workhorse for any metric where the analytic answer is painful — AUC, F1, median, anything.

**35. When does bootstrap fail?**
Extreme order statistics (min/max), heavy-tailed distributions without enough data, time series (without block bootstrap), very small $n$.

> **Saying it out loud.** It fails when resampling can't reproduce the structure that matters. Extreme statistics like the maximum are the classic case — the bootstrap maximum can never exceed the observed maximum, so your interval is systematically wrong. Heavy tails break it because the rare huge values that dominate the estimate are either absent or over-represented in a resample. Time series break it because resampling destroys temporal correlation, which is what block bootstrap fixes. And very small $n$ breaks it because you're pretending a handful of points describes a population.

**36. Bootstrap a confusion-matrix metric — how?**
Resample (predictions, labels) pairs with replacement. Compute metric on resample. Repeat 1000+ times. Quantiles of the resulting distribution give CI.

> **Saying it out loud.** Resample your test set — the prediction-label pairs together, never separately — recompute the metric on each resample, and take the quantiles of the resulting distribution. That works uniformly for precision, recall, F1, AUC, anything, which is the whole appeal. The one detail people get wrong is resampling examples rather than pairs, which destroys the correspondence and gives you nonsense. Use at least a thousand resamples if you're reporting the interval.

**37. What's a paired bootstrap for model comparison?**
For each bootstrap sample, compute metric for both models on the *same* sample. Look at distribution of differences. Reject "no difference" if 0 not in CI.

> **Saying it out loud.** You resample the test set once and evaluate both models on that same resample, then look at the distribution of the difference. Keeping the sample shared is the entire point — the two models' errors are highly correlated because they see the same examples, and if you resample independently you throw that correlation away and your interval balloons. With pairing you can often detect differences of a fraction of a point that independent intervals would call noise. Then it's simple: if zero isn't in the interval of differences, the models differ.

**38. Bagging is bootstrap of what?**
Bagging = "Bootstrap Aggregating." Train each tree on a bootstrap resample of data. Random Forests add feature subsampling.

> **Saying it out loud.** Bagging is bootstrap aggregating — you train each model on its own bootstrap resample of the data and average the predictions. The reason it works is variance reduction: individual deep trees are unstable and overfit, but their errors are partly independent, so averaging cancels a chunk of that. Random forests push it further by also sampling features at each split, which decorrelates the trees more and helps more. Note that this only helps high-variance models — bagging a linear regression buys you almost nothing.

---

## G. Bayesian inference

**39. State Bayes' theorem.**
$p(\theta | x) = p(x|\theta)p(\theta)/p(x)$.

> **Saying it out loud.** Posterior is proportional to likelihood times prior. In words: what you believe after seeing the data equals how well the data fits each hypothesis, weighted by how plausible you thought each hypothesis was to begin with. The denominator is just a normalizing constant, and for most practical purposes you can ignore it — except when you need it for model comparison, where it becomes the hard part. The one-line version worth memorizing is that Bayes tells you how to update a belief, not how to have one.

**40. What's a conjugate prior? Example?**
A prior whose posterior stays in the same family. Beta-Bernoulli: prior $\mathrm{Beta}(\alpha, \beta)$ + $s$ successes / $n-s$ failures → posterior $\mathrm{Beta}(\alpha + s, \beta + n - s)$.

> **Saying it out loud.** A conjugate prior is one where the posterior lands in the same family as the prior, so updating is arithmetic rather than integration. Beta-Bernoulli is the example to have: start with a Beta, observe some successes and failures, and just add them to the Beta's two parameters. The reason anyone cares is history — before MCMC, conjugacy was the only way to get a posterior in closed form. Today it's mostly used for intuition and for fast online updating, since you can update a belief in one addition.

**41. Beta-Bernoulli posterior mean?**
$(\alpha + s)/(\alpha + \beta + n)$. Smoothing: prior acts like $\alpha + \beta$ "pseudo-observations."

> **Saying it out loud.** It's successes plus alpha over total plus alpha plus beta, which is exactly the smoothing formula you've already used in practice. The interpretation is that the prior acts like a set of imaginary observations you've already seen. That's why zero successes out of three doesn't give you a probability of zero — with a Beta(1,1) prior you get 0.2, which is far more sensible. Every add-one smoothing scheme in NLP is this, and it's a nice concrete way to explain what a prior does.

**42. What's MAP estimation?**
$\hat{\theta}_{\mathrm{MAP}} = \arg\max p(\theta|x) = \arg\max [\log p(x|\theta) + \log p(\theta)]$. MLE + log-prior penalty.

> **Saying it out loud.** MAP picks the parameter with the highest posterior probability, which works out to maximum likelihood plus a log-prior term. So it's MLE with an opinion — the data pulls one way and the prior pulls toward whatever you thought beforehand, with the prior's influence shrinking as data accumulates. The difference from full Bayesian inference is that MAP takes only the peak of the posterior and discards the spread, so you get a point estimate with no uncertainty, and in high dimensions the peak can be quite unrepresentative of the bulk.

**43. Connection between MAP and regularization?**
Gaussian prior on weights → $\ell_2$ penalty (ridge). Laplace prior → $\ell_1$ penalty (lasso). Regularization is MAP with a particular prior.

> **Saying it out loud.** This is the connection worth having instantly: L2 regularization is MAP with a Gaussian prior on the weights, and L1 is MAP with a Laplace prior. Once you see it, the sparsity of lasso stops being magic — a Laplace prior has a sharp spike at zero, so it genuinely believes most coefficients are exactly zero, while a Gaussian just believes they're all smallish. And the regularization strength is the prior's inverse variance, so a stronger penalty is literally a more confident prior. It's the cleanest bridge between the Bayesian and the practical ML worlds.

**44. What's the marginal likelihood / evidence and why does it matter?**
$p(x) = \int p(x|\theta)p(\theta)d\theta$. Used for Bayesian model comparison (Bayes factors). Hard to compute in general.

> **Saying it out loud.** The evidence is the probability of your data averaged over all parameter values, weighted by the prior. It matters because it's the principled way to compare models: the ratio of evidences between two models is the Bayes factor, and it has automatic Occam's razor built in — a flexible model spreads its predictions thin, so it pays a penalty unless the data really needs the flexibility. The problem is that it's an integral over the whole parameter space, and for anything nontrivial it's intractable. That's why cross-validation, which approximates the same thing, is what people actually use.

**45. MCMC vs variational inference?**
MCMC: sample from posterior; asymptotically exact, slow. VI: approximate posterior with a simpler distribution by minimizing KL; biased, fast. ML practitioners usually use VI when scale matters.

> **Saying it out loud.** MCMC builds a chain that eventually samples from the true posterior, so it's asymptotically exact and slow, and diagnosing whether it has converged is genuinely hard. Variational inference instead picks a simple family of distributions and finds the closest member by minimizing KL divergence, which turns inference into optimization — fast, scalable, and biased. The specific bias is worth knowing: the standard reverse-KL objective is mode-seeking, so VI systematically underestimates posterior variance and gives you overconfident uncertainty. At ML scale you use VI anyway, because MCMC on a neural network isn't happening.

---

## H. Practical ML stats

**46. You report a model AUC of 0.85. How do you give it a CI?**
Bootstrap the test set 1000+ times; compute AUC on each; take 2.5%/97.5% quantiles.

> **Saying it out loud.** Bootstrap the test set. Resample the prediction-label pairs with replacement a thousand times or more, recompute AUC each time, and take the 2.5th and 97.5th percentiles. There's an analytic option too — DeLong's method — but the bootstrap is more robust and works identically for any metric you might report next. The useful thing to say afterwards is that this makes the test set size vividly concrete: on 500 examples that interval will be several points wide, which usually means your leaderboard gap isn't real.

**47. Two models: AUC 0.85 vs 0.84. Is the difference significant?**
Paired bootstrap of AUC differences. CI for difference; reject "no difference" if 0 not in CI. Or DeLong's test for AUC specifically.

> **Saying it out loud.** Use a paired bootstrap: resample the test set once, evaluate both models on the same resample, and look at the distribution of the difference. The pairing matters enormously because both models see the same examples and their errors are correlated, so the difference has much less variance than either AUC alone. If zero sits inside the interval of differences, you can't claim a win. DeLong's test does the same job analytically for AUC specifically. One point of AUC on a small test set is very often noise, and this is how you find out.

**48. You run 50 A/B tests and 3 are "significant" at $p<0.05$. Are any real?**
Probably 2.5 false positives expected by chance. Apply Bonferroni ($\alpha/50 = 0.001$) or BH correction.

> **Saying it out loud.** With 50 tests at a 5 percent threshold you'd expect about 2.5 false positives from pure noise, so three hits is exactly what nothing-happening looks like. That's the answer, and it's a good one to deliver plainly. What you'd do next is apply a correction — Bonferroni takes you to 0.001, or Benjamini-Hochberg if you'd rather control the false discovery rate and keep some power — and then replicate whatever survives on fresh data. Replication is the real test; a correction just tells you how much to distrust the first pass.

**49. Model accuracy = 87% on test set of 1000. CI?**
Wald: $0.87 \pm 1.96 \sqrt{0.87 \cdot 0.13 / 1000} \approx 0.87 \pm 0.021$. Or Wilson interval (better for proportions). Or bootstrap.

> **Saying it out loud.** It's a proportion, so the standard error is the square root of $p(1-p)/n$, which here is about 1.06 percent, and the 95 percent interval is roughly 85 to 89 percent. That's a two-point spread on a thousand examples, which is the number worth internalizing — it means a model reporting 88 percent is not measurably better. For proportions near zero or one the Wilson interval behaves better than the plain Wald formula, since Wald can run off the end of the range. And when in doubt, bootstrap.

**50. Train accuracy 95%, test 87%. Statistically significant gap?**
Compute CIs on each. Subtract. If CIs overlap heavily, gap might be noise. Better: paired bootstrap of differences, or test on multiple test splits.

> **Saying it out loud.** First, it's the wrong question, and saying so is the strong answer — train and test accuracy are measuring different things, and train accuracy is optimistically biased by construction because the model fit that data. So a gap is expected, not evidence of anything by itself. If what you actually want to know is whether the model generalizes worse than some baseline, compare test-set performance against that baseline with a paired bootstrap. If you want to know whether the gap indicates overfitting, look at whether the validation curve turned upward during training, which is the diagnostic that actually answers it.

---

## Quick fire

**51.** *MLE for Bernoulli?* Sample mean.
**52.** *Bessel's correction divisor?* $n-1$.
**53.** *95% z-value?* 1.96.
**54.** *CRLB lower-bounds what?* Variance of unbiased estimator.
**55.** *Conjugate of Bernoulli?* Beta.
**56.** *Conjugate of Poisson?* Gamma.
**57.** *Conjugate of multinomial?* Dirichlet.
**58.** *Bonferroni: divide $\alpha$ by?* Number of tests $m$.
**59.** *MAP equals MLE when?* Uniform prior.
**60.** *CLT statement?* Sample mean is asymptotically Gaussian regardless of underlying distribution (with finite variance).

---

## Self-grading

If you can't answer 1-15, you don't know basic statistics. If you can't answer 16-35, you'll get tripped up on every interview that probes ML evaluation rigor. If you can't answer 36-50, frontier-lab interviews on experimental rigor will go past you.

Aim for 40+/60 cold.
