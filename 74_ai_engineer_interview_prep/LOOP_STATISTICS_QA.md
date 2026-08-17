# Statistics and Experimentation — Interview Question Bank

This is the round most ML candidates under-prepare and it eliminates people. Cover the answer, say yours out loud, then compare — reading these silently will not transfer. Answers are written the length you should actually speak them: a claim, a mechanism, and a number. Where a derivation *is* the answer, the derivation is written out. Traps flag the specific wrong answer that gets people cut.

---

## 1. Inference Fundamentals

### Q: What is a p-value?

**Answer.** A p-value is the probability of observing a test statistic at least as extreme as the one I observed, *assuming the null hypothesis is true*. It is a statement about data under an assumed model, not a statement about the hypothesis.

Formally: if $T$ is my test statistic and $t_{obs}$ my observed value, the two-sided p-value is $P(|T| \geq |t_{obs}| \mid H_0)$. The key structural fact is that under a continuous null, the p-value is uniformly distributed on $[0, 1]$. That is exactly why $\alpha = 0.05$ gives a 5% false positive rate — you reject when a $\text{Uniform}(0,1)$ draw lands below 0.05.

The four things people wrongly say it is:

1. **"The probability the null is true."** No. That is $P(H_0 \mid \text{data})$, a posterior. The p-value is $P(\text{data} \mid H_0)$. Getting these backwards is the prosecutor's fallacy. To get the posterior you need a prior and Bayes' theorem.
2. **"The probability the result is due to chance."** Same inversion, dressed up. It presupposes the null, so it cannot quantify how likely the null is.
3. **"$1 - p$ is the probability the alternative is true"** or "the probability the result replicates." Replication probability depends on true effect size and power. A result at $p = 0.05$ with 80% power replicates roughly 50% of the time, not 95%.
4. **"A small p means a large or important effect."** p conflates effect size with sample size. With $n = 10$ million, a 0.001% lift gives $p < 0.001$ and is worth nothing.

One more precision point: the p-value is computed under the *entire* null model, which includes the assumptions — random sampling, independence, the variance model, and the stopping rule. A small p-value means "the data are surprising under this model," and a violated assumption is one of the ways the model can be wrong. That is why peeking breaks p-values: the stopping rule is part of the null.

**Follow-up: "So what do I actually do with $p = 0.03$?"** → It says the data are somewhat unusual under the null. I would report it alongside the point estimate and confidence interval, because the interval carries the effect size and the precision, which is what the decision depends on. If the CI is $[0.1\%, 4.0\%]$ lift and we need 1% to justify the engineering cost, $p = 0.03$ does not settle anything — the interval overlaps "not worth it." I would also state how many metrics I tested, because $p = 0.03$ on one of 30 metrics is expected noise.

*Trap:* Saying "there's a 3% chance the null is true." That is the single most common auto-fail phrasing in this round.

---

### Q: Walk me through a hypothesis test end to end.

**Answer.** Five steps, and the discipline is that steps 1 through 4 happen before I look at the outcome data.

**1. State the hypotheses.** They must be about population parameters, not sample statistics, and they must partition the space. For an A/B test on conversion: $H_0: p_T - p_C = 0$ versus $H_1: p_T - p_C \neq 0$. I fix two-sided unless there is a pre-registered reason not to.

**2. Choose the test statistic and significance level.** The statistic is a function of the data whose distribution under $H_0$ I know. For two proportions:

$$Z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_T} + \frac{1}{n_C}\right)}}$$

where $\hat{p}$ is the pooled rate. Set $\alpha = 0.05$ and target power $1 - \beta = 0.80$, and compute the required $n$ *now*, not later.

**3. Derive the null distribution.** Under $H_0$ and with $n$ large enough for the CLT, $Z \sim N(0,1)$. If I could not appeal to a known null distribution — say the metric is a weird ratio, or $n$ is small and skewed — I would get the null distribution by permutation: shuffle the treatment labels 10,000 times, recompute the statistic each time, and use that empirical distribution.

**4. Set the decision rule.** Reject $H_0$ if $|Z| > 1.96$. Equivalently, reject if the 95% CI for the difference excludes zero. These are the same rule.

**5. Run once, compute, decide, and report the estimate.** Suppose $n_C = n_T = 20{,}000$, $\hat{p}_C = 0.100$, $\hat{p}_T = 0.107$. Pooled $\hat{p} = 0.1035$, so the standard error is $\sqrt{0.1035 \times 0.8965 \times (2/20000)} = \sqrt{9.28 \times 10^{-6}} = 0.00305$. Then $Z = 0.007 / 0.00305 = 2.30$, $p = 0.021$. The 95% CI on the absolute lift is $0.007 \pm 1.96(0.00305) = [0.0010, 0.0130]$, i.e. a 1% to 13% relative lift.

I report the interval, not just the reject/don't-reject. And I state what I checked: sample ratio mismatch, that the randomization unit matches the analysis unit, and that no one looked early.

**Follow-up: "What if the SRM check fails — you assigned 50/50 but see 50.4/49.6?"** → I stop and do not read the result. With $n = 40{,}000$, a 50.4/49.6 split is a $\chi^2$ of about 2.6, $p \approx 0.11$, so that particular split is fine. But if the SRM p-value is below roughly 0.001, the randomization or logging is broken — bot filtering applied asymmetrically, a redirect that drops slow clients in one arm, a caching layer. Any of those correlate with the outcome, so the estimate is biased by an unknown amount and no amount of statistics downstream fixes it. Debug the pipeline, then rerun.

*Trap:* Formulating $H_0$ about the sample ("the null is that the observed means are equal"). The null is always about population parameters.

---

### Q: Type I versus Type II error — define both, and tell me which one you would rather make.

**Answer.** Type I is rejecting a true null: a false positive, probability $\alpha$. Type II is failing to reject a false null: a false negative, probability $\beta$, with power $= 1 - \beta$.

The 2x2:

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I ($\alpha$) | Correct (power, $1-\beta$) |
| **Fail to reject** | Correct ($1-\alpha$) | Type II ($\beta$) |

They trade off at fixed $n$: pushing $\alpha$ from 0.05 to 0.01 moves the critical value out from 1.96 to 2.58, which shrinks false positives and grows false negatives. The only way to reduce both is more data or less variance.

Which I would rather make is entirely a business question, and the right answer names the asymmetry. Shipping a ranking change with a false positive costs a permanent regression that dilutes every future test's baseline and is expensive to detect later — I would tighten $\alpha$. But in an exploratory screen where I will run a confirmatory test on anything that passes, a false negative kills the idea permanently while a false positive just costs one more test — I would loosen $\alpha$ to 0.10 and spend the budget on power.

For a safety or trust metric the asymmetry flips hard. There the null is "the model is safe," and a Type II error ships harm. I would run those as guardrails at high $\alpha$ — I *want* to be triggered easily — and treat non-inferiority, not superiority, as the hypothesis.

**Follow-up: "Your test is underpowered and shows $p = 0.30$. Does that mean there's no effect?"** → No. Absence of evidence is not evidence of absence. With 20% power, a real effect fails to reach significance 80% of the time. What I can do is report the CI: if the 95% CI is $[-8\%, +12\%]$, I have learned essentially nothing and should say so. If it is $[-0.3\%, +0.4\%]$, I have positively established the effect is small — that is a useful negative result, and formally it is an equivalence test (TOST) against a pre-specified margin.

*Trap:* "$p > 0.05$ so the treatment has no effect." Failing to reject is not accepting.

---

### Q: What is statistical power, and what drives it?

**Answer.** Power is $P(\text{reject } H_0 \mid H_1 \text{ true at a specified effect size})$. It is not a property of a test alone — it is only defined relative to a particular alternative. "This test has 80% power" is shorthand for "80% power to detect an effect of size $\Delta$."

Four levers, and it is worth knowing the direction and magnitude of each:

1. **Effect size $\Delta$.** The dominant term, and it enters squared in the sample size formula, so halving the effect you want to catch quadruples the data you need. This is the lever you usually cannot pull, but you can sometimes pull it indirectly by targeting the test to a subpopulation where the effect is bigger.
2. **Sample size $n$.** Power grows with $\sqrt{n}$. Going from $n$ to $4n$ halves the standard error.
3. **Variance $\sigma^2$.** Reducing it is equivalent to more data and is usually the cheapest lever. Options: CUPED using pre-period data, winsorizing a heavy-tailed revenue metric at the 99th percentile, stratified assignment, or switching to a less noisy proxy metric that is validated against the real one.
4. **$\alpha$.** Raising it raises power, mechanically, by lowering the bar. Also one-sided tests have more power than two-sided at the same $\alpha$ — this is the honest reason people want one-sided tests, and the reason to be suspicious.

For a two-sample z-test, power is
$$1 - \beta = \Phi\left(\frac{\Delta}{\sigma\sqrt{2/n}} - z_{\alpha/2}\right)$$
so everything above is visible in one expression: $\Delta$ and $\sqrt{n}$ in the numerator, $\sigma$ and $z_{\alpha/2}$ working against you.

The practical failure mode I watch for is *post-hoc* power — computing power from the observed effect after the fact. It is a deterministic function of the p-value ($p = 0.05$ always gives roughly 50% observed power), so it carries no information. If someone asks "was the test underpowered?" after a null result, the right artifact is the confidence interval, not retrospective power.

**Follow-up: "You have 80% power and the test comes back significant. What's the probability the effect is real?"** → Not 95%. It depends on the prior probability that the idea works. If 10% of shipped experiment ideas have a real effect, then out of 1,000 tests: 100 real, of which 80 detected; 900 null, of which 45 falsely significant. So $80 / 125 = 64\%$ of significant results are real — the false discovery rate is 36%, not 5%. This is why teams with low hit rates and low power drown in false positives, and why the fix is raising power and prior quality, not lowering $\alpha$ alone.

---

### Q: Walk me through a sample size calculation. Use real numbers.

**Answer.** For a two-sample test with equal arms, the per-arm sample size is

$$n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{\Delta^2}$$

The 2 comes from the difference of two independent means: $\text{Var}(\bar{X}_T - \bar{X}_C) = 2\sigma^2/n$.

**Worked example.** Checkout conversion, baseline 10%. We care about a 10% relative lift, so $\Delta = 0.01$ absolute (10% → 11%). Use $\alpha = 0.05$ two-sided so $z = 1.96$, and 80% power so $z_{1-\beta} = 0.84$. For a Bernoulli metric $\sigma^2 \approx p(1-p) = 0.10 \times 0.90 = 0.09$.

$$n = \frac{2(1.96 + 0.84)^2 (0.09)}{(0.01)^2} = \frac{2(7.84)(0.09)}{0.0001} = \frac{1.4112}{0.0001} = 14{,}112$$

So about 14,100 per arm, 28,200 total. At 5,000 checkouts a day that is roughly six days — and I would round up to a full week anyway to cover the weekly cycle.

**Now make the effect realistic.** Suppose the honest expectation is a 2% relative lift, $\Delta = 0.002$:

$$n = \frac{1.4112}{(0.002)^2} = \frac{1.4112}{4\times10^{-6}} = 352{,}800 \text{ per arm}$$

Twenty-five times more data for a five-times-smaller effect. That quadratic blowup is the single most important intuition here, and it is the conversation to have with a PM *before* the test: at our traffic we can detect 10% lifts in a week and 2% lifts in five months, so if you believe the effect is 2%, we either run for five months, pool with other launches, or reduce variance.

For a continuous metric like revenue per user, swap in the actual $\sigma$ from historical data rather than a formula, and check skew — with revenue, $\sigma$ is often 3-5x the mean, which is why the same relative MDE costs far more traffic than a conversion metric.

**Follow-up: "The PM says we only have two weeks. What do you tell them?"** → I invert the formula and give them the MDE instead of arguing. With $n = 70{,}000$ per arm in two weeks: $\Delta = (1.96+0.84)\sqrt{2(0.09)/70000} = 2.8 \times 0.001604 = 0.0045$, so a 4.5% relative MDE. Then the statement is: "In two weeks we can reliably detect a 4.5% lift or bigger. If the true effect is 2%, this test will come back null about 78% of the time and we will have learned nothing — but we will be tempted to conclude the feature failed." Then offer the levers: CUPED, a higher-traffic surface, a bolder treatment, or a one-sided test if the decision is genuinely one-directional.

*Trap:* Plugging in the effect size you *hope* for rather than the smallest effect that would change the decision. That is how tests get designed to fail.

---

### Q: What does a 95% confidence interval mean — and what does it not mean?

**Answer.** It means the *procedure* covers the true parameter 95% of the time under repeated sampling. If I ran this experiment 100 times and built an interval each time, about 95 of those intervals would contain the true value. The randomness lives in the interval, not the parameter.

What it does **not** mean:

1. **"There's a 95% probability the true value is in $[0.1\%, 1.3\%]$."** Once computed, the interval either contains the parameter or it does not — the probability is 0 or 1. Under a frequentist framing the parameter is fixed and not random. The 95% is a property of the recipe. (The interval you want, where that statement *is* legitimate, is a Bayesian credible interval, and it requires a prior.)
2. **"95% of the data fall in this range."** That is a prediction interval or a tolerance interval, and it is much wider. A CI on the mean shrinks like $1/\sqrt{n}$ and goes to zero; a prediction interval converges to the width of the data distribution and does not.
3. **"95% of future sample means will fall in this interval."** Actually about 83% for a 95% CI, because the future mean has its own sampling error stacked on top of this one's.
4. **"Two CIs overlap, so the difference isn't significant."** This is wrong and common. Overlapping 95% CIs on two groups can still correspond to $p < 0.05$ on the difference — because $\text{SE}_{\text{diff}} = \sqrt{\text{SE}_1^2 + \text{SE}_2^2}$ is less than $\text{SE}_1 + \text{SE}_2$. Always test the difference directly. The converse is safe: if the CIs do *not* overlap, the difference is significant.

The duality worth stating: a 95% CI is exactly the set of null values that would not be rejected at $\alpha = 0.05$. So the CI is strictly more informative than the p-value — it contains the accept/reject decision plus the effect size plus the precision. That is the reason to lead with it.

**Follow-up: "Your CI on lift is $[-1\%, +9\%]$. What do you tell the VP?"** → "We cannot rule out a small regression, and we cannot rule out a large win. The test was underpowered for the effect size we're seeing. The point estimate is +4%, but the honest read is that we don't know the sign with confidence." Then: extend the test if the traffic is available, or if the decision is forced, note that the interval puts most of its mass on positive and the downside is bounded at about 1%, so shipping is defensible if the rollback cost is low. That is a decision-theoretic argument, and I would label it as one rather than dressing it up as statistical evidence.

---

### Q: State the Central Limit Theorem. What are the assumptions, and when does it fail?

**Answer.** For i.i.d. $X_1, \ldots, X_n$ with finite mean $\mu$ and finite variance $\sigma^2$,

$$\sqrt{n}\left(\bar{X}_n - \mu\right) \xrightarrow{d} N(0, \sigma^2)$$

equivalently $\bar{X}_n \approx N(\mu, \sigma^2/n)$ for large $n$. The remarkable part is that the limiting distribution does not depend on the shape of the underlying distribution at all — only on $\mu$ and $\sigma^2$.

Assumptions, and each one is a failure mode:

**Finite variance.** This is the one that actually bites. Cauchy has no finite mean or variance — the sample mean of $n$ Cauchy draws is itself Cauchy, with no concentration at all, no matter how large $n$ is. Pareto with tail index $\alpha < 2$ has infinite variance and the sums converge to an $\alpha$-stable law, not a normal. In practice: revenue per user, session duration, and API latency are heavy-tailed. Variance is technically finite (real data is bounded), but the *effective* convergence is slow because a handful of whales dominate the sum.

**Independence.** Correlated observations break it in the usual form. If users appear multiple times, or observations are clustered by account or session, the effective sample size is far below the row count. Under weak dependence (mixing conditions) there are CLT variants with a modified variance — for stationary sequences the asymptotic variance becomes $\sigma^2 + 2\sum_{k\ge1}\gamma_k$, the long-run variance. Under strong dependence, no CLT.

**Identical distribution.** Relaxable — the Lyapunov and Lindeberg CLTs allow non-identical variables provided no single term dominates the sum. That "no single term dominates" condition is exactly what heavy tails violate.

**When "large $n$" isn't large enough.** The rule of thumb "$n > 30$" is folklore and dangerous for skewed data. A defensible rule from Boos and Hughes-Oliver ties it to skewness: you need roughly $n > 355 \gamma^2$ for the one-sided t-test's error to be within 1 percentage point, where $\gamma$ is the population skewness. Revenue per user commonly has skewness of 5-15, which puts the requirement in the tens of thousands, not 30.

**What I do about it.** Check the metric's skewness and kurtosis on historical data. Then: winsorize or cap at the 99th or 99.9th percentile (declare the cap in advance), use a bootstrap or permutation test rather than a t-test, switch to a bounded metric (converted yes/no rather than revenue), or trim by analyzing the log — with the caveat that a test on $\log$ revenue answers a different question than a test on revenue and you cannot claim the latter from the former.

**Follow-up: "How would you actually verify the CLT holds for your metric?"** → Simulate. Take a year of historical per-user values, resample $n$ of them (the planned per-arm size) 10,000 times, and look at the distribution of the resampled mean — QQ-plot it against normal. Better: run an A/A test harness. Split historical traffic into fake treatment and control 1,000 times, run the actual analysis pipeline, and check that the p-values come out uniform and that the false positive rate is 5% at $\alpha = 0.05$. If A/A gives 8% false positives, the CLT approximation (or the independence assumption) is broken, and I would find that before it burns a real launch decision.

*Trap:* "The CLT says the data become normally distributed as $n$ grows." No — the data stay exactly as non-normal as they were. It is the *sampling distribution of the mean* that becomes normal.

---

### Q: Law of large numbers versus central limit theorem?

**Answer.** They answer different questions about the same object.

**LLN is about convergence to a point.** $\bar{X}_n \to \mu$. The weak LLN gives convergence in probability: $P(|\bar{X}_n - \mu| > \epsilon) \to 0$ for any $\epsilon > 0$. The strong LLN gives almost sure convergence: $P(\lim_n \bar{X}_n = \mu) = 1$, which is a statement about the whole trajectory settling down permanently rather than each individual $n$ being close. The weak LLN needs only finite mean; the strong LLN (Kolmogorov) also holds with just finite mean for i.i.d. data.

**CLT is about the rate and shape of that convergence.** It says the error $\bar{X}_n - \mu$ is on the order of $\sigma/\sqrt{n}$ and, once rescaled by $\sqrt{n}$, has a normal shape in the limit. So CLT is a refinement: LLN says the error vanishes, CLT says how fast and with what distribution.

Why it matters practically: LLN alone tells me my estimate is consistent, which justifies nothing about inference. CLT is what lets me build a confidence interval or a p-value, because I need the shape of the error, not just its disappearance. Every standard error I quote is a CLT statement.

The Cauchy case separates them cleanly and is a good thing to have ready: Cauchy has no finite mean, so LLN fails — the sample mean does not converge to anything. And a fortiori CLT fails. Meanwhile a Pareto with $1 < \alpha < 2$ has a finite mean but infinite variance, so LLN holds (the mean converges) but CLT fails (the fluctuations are $\alpha$-stable, not normal, and the standard error formula is meaningless). That is the clean example of LLN without CLT.

**Follow-up: "Gambler's fallacy — after 10 heads, does the LLN say tails is now more likely?"** → No. The LLN operates by dilution, not correction. The coin has no memory; the next flip is still 50/50. What happens is that the initial surplus of 10 heads becomes negligible relative to $n$: after a million flips, the proportion is $(500{,}005 + 5)/1{,}000{,}010 \approx 0.500005$. The *proportion* converges to 0.5 while the *absolute* difference between heads and tails actually grows like $\sqrt{n}$ — that is the CLT talking. Both statements are true simultaneously, and holding both is the test of whether someone understands the theorem.

---

### Q: t-test versus z-test — when do you use each?

**Answer.** The distinction is whether the population variance is known.

**z-test:** $\sigma$ is known, or $n$ is large enough that $\hat{\sigma}$ is essentially $\sigma$. Statistic $Z = (\bar{X} - \mu_0)/(\sigma/\sqrt{n}) \sim N(0,1)$.

**t-test:** $\sigma$ is estimated from the sample. Statistic $T = (\bar{X} - \mu_0)/(s/\sqrt{n}) \sim t_{n-1}$. The t-distribution has heavier tails precisely to account for the extra uncertainty in estimating $s$. Degrees of freedom $n-1$ because one is spent estimating $\bar{X}$.

In practice you essentially never know $\sigma$, so the t-test is the honest default. The reason people reach for z anyway is that the two converge fast: $t_{29}$ has critical value 2.045 versus 1.96 — about a 4% wider interval — and by $n = 100$ it is 1.984, a 1% difference. Above a few hundred observations the distinction is numerically irrelevant, which is why A/B testing at scale uses z-tests without apology.

The proportions case is the common exception where z is genuinely right: for a Bernoulli metric, the variance $p(1-p)$ is a function of the mean, so under $H_0$ you *know* the variance once you fix $p$ — nothing is being separately estimated, and the z-test is correct.

Two-sample flavors matter more than z-versus-t in practice. **Student's t** pools the variance and assumes equal variances across groups. **Welch's t** does not, using
$$T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$
with Satterthwaite-approximated df. Welch should be the default: it costs almost nothing in power when variances are equal and it saves you when they are not, and unequal group sizes with unequal variances is where pooled t goes badly wrong. In an A/B test the treatment often changes the variance, not just the mean — a feature that creates whales widens the treatment distribution — so assuming equal variances is an unforced error.

**Follow-up: "$n = 15$ per group and the data are clearly skewed. Now what?"** → The t-test's normality assumption is about the sampling distribution of the mean, and at $n = 15$ with visible skew the CLT has not rescued me. Options in order: a permutation test, which is exact under the sharp null of no effect and makes no distributional assumption; a bootstrap CI (BCa, which corrects for skew); or Mann-Whitney if I am willing to answer the stochastic-dominance question instead of the mean question. I would use a permutation test — with $n = 15$ per group there are plenty of reshuffles, it is exact, and it tests the mean difference I actually care about.

---

### Q: Paired versus unpaired test — when does pairing help, and how much?

**Answer.** Pairing applies when each observation in one condition has a natural partner in the other: the same user before and after, the same query scored by two ranking models, matched subjects. Instead of comparing two group means, you compute the within-pair difference $d_i = X_i - Y_i$ and run a one-sample t-test on $H_0: \mu_d = 0$, with $n-1$ df where $n$ is the number of *pairs*, not observations.

The payoff is variance:
$$\text{Var}(X - Y) = \sigma_X^2 + \sigma_Y^2 - 2\rho\sigma_X\sigma_Y$$
The $-2\rho\sigma_X\sigma_Y$ term is the entire point. If $\sigma_X = \sigma_Y = \sigma$, paired variance is $2\sigma^2(1-\rho)$ against $2\sigma^2$ unpaired. With $\rho = 0.8$ — typical for the same user measured twice — that is an 80% variance reduction, so the same power needs about one fifth the pairs. Pairing cancels every subject-level confounder in one stroke: heavy users are compared to themselves, so between-user variance never enters.

The cost is degrees of freedom. With $2n$ total observations, unpaired gives $2n-2$ df while paired gives $n-1$. At small $n$ that matters: with 10 pairs you have 9 df versus 18, and $t_{9} = 2.26$ against $t_{18} = 2.10$. So if $\rho$ is near zero, pairing strictly loses. The break-even is roughly $\rho > 1/(n)$ — trivially satisfied in real paired data, but worth knowing the trade exists.

In ML evaluation this is the standard mistake. When comparing two models on the same test set, the correct test is paired — model A and model B see identical examples, and per-example difficulty is a huge shared variance component. Running an unpaired test on the two accuracy vectors throws away that correlation and can easily turn a real difference into a null result. For paired binary outcomes specifically, use **McNemar's test**: it looks only at the discordant pairs (A right / B wrong versus A wrong / B right) with $\chi^2 = (b-c)^2/(b+c)$, since concordant pairs carry no information about the difference.

**Follow-up: "Can you pair in an A/B test?"** → Not on the same user for the same decision — a user is in one arm and you never observe their counterfactual. But you can get most of the benefit with pre-period data, which is exactly what CUPED does: subtract off each user's pre-experiment metric, which is the paired-differencing idea applied across arms rather than within a unit. Switchback designs also achieve a form of pairing by exposing the same market to both conditions in alternating time slices.

---

### Q: One-tailed versus two-tailed — when is a one-tailed test defensible?

**Answer.** Two-tailed splits $\alpha$ across both tails: reject if $|Z| > 1.96$. One-tailed puts all of $\alpha$ in one tail: reject if $Z > 1.645$. That lower threshold is why one-tailed is more powerful, and it is why it is abused.

A one-tailed test is defensible under exactly one condition: **an effect in the opposite direction would lead to the identical decision as no effect at all**, and you committed to this before seeing data.

Cases where that genuinely holds:

- **Non-inferiority tests.** "Is the cheaper, faster model no worse than the incumbent by more than 0.5%?" You only reject if it is meaningfully worse; a large improvement changes nothing about the ship decision. Here one-sided is correct, and the null is a margin, not zero.
- **Hard guardrails.** "Does latency regress by more than 50 ms?" An improvement is fine and requires no action.
- **Physically or logically one-directional settings.** A validity check where the reverse effect is impossible.

Cases where it does not hold, which is most A/B tests: if the new recommender tanked engagement by 5%, you would absolutely act on that — you would revert and investigate. That is a different decision from "no effect," so you need two tails.

The abuse pattern to name: running two-tailed, getting $p = 0.07$, and switching to one-tailed to get $p = 0.035$. That is p-hacking with extra steps, and the actual Type I rate is not 0.05. Related and equally bad: choosing the direction after seeing which way the estimate points, which makes the real one-sided rate 0.10.

My working rule: default two-tailed everywhere, and if a one-sided test is right, write it into the experiment doc before launch alongside the reason the opposite direction is decision-irrelevant. If someone proposes one-sided mid-test, the answer is no.

**Follow-up: "Doesn't one-tailed give you 'free' power?"** → It gives you real power — the sample size drops roughly 20% at 80% power, since $(1.645+0.84)^2/(1.96+0.84)^2 = 0.79$. But it is not free: you have traded away all ability to detect harm. You have set the power against a regression to zero. On a launch decision that is usually a bad trade, because unnoticed regressions compound into the baseline and you pay for them forever.

---

### Q: When do you use a chi-square test, and what are its assumptions?

**Answer.** Two main uses, same statistic.

**Goodness of fit:** does one categorical variable match a hypothesized distribution? Example: is traffic split 50/50 across arms as intended — an SRM check. $df = k - 1$.

**Test of independence:** are two categorical variables associated? Example: does device type relate to conversion, in a $2 \times 3$ table. $df = (r-1)(c-1)$.

The statistic in both cases:
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$
with expected counts under independence $E_{ij} = \frac{(\text{row}_i \text{ total})(\text{col}_j \text{ total})}{N}$.

**Worked SRM example.** Intended 50/50, observed 10,300 versus 9,700 out of 20,000. $E = 10{,}000$ each. $\chi^2 = 300^2/10000 + 300^2/10000 = 9 + 9 = 18$, on 1 df, so $p < 0.0001$. Randomization is broken — do not read the experiment result.

Assumptions:

1. **Independent observations.** Each unit contributes to exactly one cell. If one user generates 40 rows, this is violated and the statistic is inflated, which is the most common real-world abuse — running chi-square on events when the unit of randomization is users.
2. **Expected counts large enough.** The usual rule is all $E_{ij} \geq 5$ (some texts allow 80% of cells $\geq 5$ with none below 1). Below that, the $\chi^2$ approximation to the discrete distribution is poor; use Fisher's exact test instead.
3. **Counts, not percentages or means.** Feeding proportions into the formula changes the scale and produces nonsense.
4. The $\chi^2$ distribution is an approximation to a discrete null. Yates' continuity correction is sometimes applied to $2\times2$ tables, though it is generally considered over-conservative and modern practice prefers Fisher's exact or a permutation approach when counts are small.

Note the relationship worth citing: for a $2\times2$ table, the chi-square test of independence is algebraically equivalent to the two-proportion z-test, with $\chi^2 = Z^2$. So they always agree.

**Follow-up: "You have a $2\times2$ with a cell count of 3. What do you do?"** → Fisher's exact test. It conditions on the margins and computes the exact hypergeometric probability of tables at least as extreme, so it needs no large-sample approximation. It is conservative but valid. If I need the effect size too, I would report the odds ratio with an exact (Clopper-Pearson-style) interval, and I would say plainly that with a cell count of 3 the estimate is very imprecise regardless of which test I run.

---

### Q: What is ANOVA and when would you use it instead of multiple t-tests?

**Answer.** ANOVA tests whether $k \geq 3$ group means are all equal: $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$ against "at least one differs." It works by decomposing total variance into between-group and within-group parts:

$$SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}$$
$$F = \frac{SS_{\text{between}}/(k-1)}{SS_{\text{within}}/(N-k)} = \frac{MS_{\text{between}}}{MS_{\text{within}}}$$

Under $H_0$ both mean squares estimate $\sigma^2$, so $F \approx 1$; a real difference inflates the numerator. Under $H_0$, $F \sim F_{k-1, N-k}$.

Why not just run all pairwise t-tests? Because of multiplicity. With $k = 5$ groups there are $\binom{5}{2} = 10$ comparisons, and the family-wise error rate is $1 - 0.95^{10} = 40\%$. ANOVA gives a single omnibus test at a controlled 5%.

The honest caveat: ANOVA tells you *that* something differs, not *which*. So you follow a significant F with post-hoc comparisons that control FWER — Tukey's HSD for all pairwise comparisons, or Dunnett's if every group is compared against a single control, which is the usual A/B/n structure. Dunnett is more powerful than Tukey when you only need control-versus-each.

Assumptions: independence, normality of residuals, and homogeneity of variance across groups (check with Levene's test; Welch's ANOVA relaxes it). ANOVA is reasonably robust to normality violations at decent $n$ but sensitive to unequal variances combined with unequal group sizes.

Practical note for interviews: in a multi-arm A/B test, many teams skip the omnibus F entirely and go straight to control-versus-each-treatment with a Benjamini-Hochberg or Dunnett correction, because the omnibus null ("all arms identical") is not the question anyone is asking. Being able to say that — that the F-test is often ceremony and the decision-relevant comparison is each treatment against control — reads as practical experience.

**Follow-up: "What's the relationship between ANOVA and linear regression?"** → They are the same model. One-way ANOVA is a linear regression of the outcome on $k-1$ dummy variables for group membership, and the ANOVA F-statistic is exactly the regression's overall F-test for whether all slopes are zero. With $k=2$ groups, both reduce to the pooled two-sample t-test, and $F = t^2$. Recognizing that ANOVA, t-tests, and regression are one framework is worth saying out loud — it is what lets you move to ANCOVA (adding covariates), which is the same idea that makes CUPED work.

---

### Q: When do you need a non-parametric test, and what do you lose?

**Answer.** When the distributional assumption underlying the parametric test is not credible and cannot be rescued by sample size: small $n$ with heavy skew, ordinal data where means are not meaningful (satisfaction ratings), or outlier-dominated metrics where the mean is not the summary you want.

The standard swaps:

| Parametric | Non-parametric |
|---|---|
| Two-sample t-test | Mann-Whitney U (Wilcoxon rank-sum) |
| Paired t-test | Wilcoxon signed-rank |
| One-way ANOVA | Kruskal-Wallis |
| Pearson correlation | Spearman / Kendall's tau |
| Any of the above | Permutation test, bootstrap |

What you lose:

1. **Power, when the parametric assumption actually holds.** Mann-Whitney's asymptotic relative efficiency versus the t-test is $3/\pi \approx 0.955$ under normality — only a 5% loss, which is small. Under heavy tails it is *more* powerful, sometimes dramatically.
2. **The estimand.** This is the bigger loss and the one people miss. Mann-Whitney does **not** test equality of medians in general. It tests $P(X > Y) = 1/2$ — stochastic dominance. It only becomes a median test under the additional assumption that the two distributions have the same shape and differ by a location shift. If treatment changes the shape (which is common — features often add a heavy right tail), a significant Mann-Whitney does not license "the median went up."
3. **Interpretable effect sizes.** A rank-based statistic does not give you "+\$0.42 per user," which is what a business decision needs. You can pair it with a Hodges-Lehmann estimator, but it is extra work.

My default in an experimentation context is neither: I use a **bootstrap or permutation test on the mean**. It makes no distributional assumption, keeps the mean as the estimand (which is what maps to total revenue, since totals are means times counts), and gives an interpretable interval. That answer usually lands better than reflexively naming Mann-Whitney.

**Follow-up: "Why do people say to use the median for revenue instead of the mean?"** → Because the median is robust to whales. But it is usually the wrong business object: the company's revenue is $n \times \bar{X}$, not $n \times \text{median}$. A feature that doubles spend for the top 1% and leaves everyone else flat is enormously valuable and moves the median by exactly zero. The right move is to keep the mean and control its variance directly — cap at the 99.9th percentile with the threshold declared in advance, use CUPED, and report the capped and uncapped estimates side by side so no one can accuse you of choosing the flattering one after the fact.

---

## 2. Linear Regression as a Statistics Object

### Q: What are the assumptions of linear regression, and specifically what breaks when each is violated?

**Answer.** Five, and the useful framing is that each one buys you a *different* property — so violating one costs you something specific rather than invalidating everything.

**1. Linearity in the parameters.** $E[y \mid X] = X\beta$. Violated: the estimator is biased and inconsistent — you are estimating the best linear approximation to a nonlinear truth, and coefficients have no causal or structural meaning. Predictions are systematically wrong in a pattern, not randomly wrong. *Detect:* residuals versus fitted values shows curvature. *Fix:* add polynomial or spline terms, transform, or use interactions. Note this assumption is about linearity in $\beta$, not in $x$ — $y = \beta_0 + \beta_1 x + \beta_2 x^2$ is still a linear model.

**2. Independence of errors.** $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$. Violated: coefficients stay unbiased, but standard errors are wrong — usually far too small with positive autocorrelation or clustering, so you get spurious significance. This is the most costly violation in practice because it is invisible in the point estimate. *Detect:* Durbin-Watson for time series, residual autocorrelation plots, or just knowing your data has repeated users. *Fix:* cluster-robust standard errors at the level of dependence, mixed-effects models, or Newey-West for time series.

**3. Homoscedasticity.** $\text{Var}(\epsilon_i \mid X) = \sigma^2$, constant. Violated: coefficients still unbiased and consistent, but OLS is no longer the minimum-variance linear estimator, and the standard errors are biased in an unpredictable direction. Same failure mode as above — valid estimates, invalid inference. *Fix:* Huber-White robust standard errors (HC3 for smaller samples), or WLS if you know the variance structure.

**4. Normality of errors.** $\epsilon \sim N(0, \sigma^2)$. This is the weakest and most over-emphasized assumption. It is *not* needed for unbiasedness, consistency, or Gauss-Markov. It is needed only for exact finite-sample t and F inference. At large $n$ the CLT gives you asymptotically normal coefficient estimates regardless. Violated at small $n$: your p-values and CIs are off. *Fix:* bootstrap the coefficients, or ignore it if $n$ is in the thousands. Note the assumption is on *errors*, not on $y$ or on the predictors — nobody needs $x$ to be normal.

**5. No perfect multicollinearity.** $X$ has full column rank. Violated perfectly: $X^\top X$ is singular and $\beta$ is not identified — infinitely many solutions, and the software either errors or silently drops a column. This is the dummy variable trap: include all $k$ category indicators plus an intercept and you get exact collinearity. Violated *approximately*: everything is still unbiased, but standard errors explode and individual coefficients become unstable and sign-flippy.

The one implicitly assumed above all these, and the one that matters for causal claims: **exogeneity**, $E[\epsilon \mid X] = 0$. Violated — by omitted variables, simultaneity, or measurement error in $x$ — and the coefficients are biased and inconsistent. No amount of data fixes it. Every other assumption on this list is about efficiency or inference; this one is about whether the number means anything.

**Follow-up: "Rank those by how much you'd worry in a real project."** → Exogeneity first, by a wide margin, if I am making a causal claim — it is the only one that biases the estimate and it is untestable from the data alone. Second, independence, because clustered data is ubiquitous and it silently manufactures significance. Third, linearity. Homoscedasticity fourth, since robust standard errors are a one-line fix. Normality last — at $n > 1000$ I do not check it, and a candidate who opens with a Shapiro-Wilk test on a million rows (where it rejects for trivial deviations) is signaling they learned this from a textbook rather than a project.

---

### Q: Derive the OLS estimator from the loss function.

**Answer.** Start with the model $y = X\beta + \epsilon$, with $y \in \mathbb{R}^n$, $X \in \mathbb{R}^{n \times p}$.

**Loss.** Minimize the sum of squared residuals:
$$L(\beta) = \sum_{i=1}^n (y_i - x_i^\top\beta)^2 = (y - X\beta)^\top(y - X\beta)$$

**Expand.**
$$L(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta$$
(the two cross terms $y^\top X\beta$ and $\beta^\top X^\top y$ are equal since each is a scalar and one is the transpose of the other).

**Differentiate and set to zero.** Using $\partial(\beta^\top A\beta)/\partial\beta = 2A\beta$ for symmetric $A$, and $\partial(\beta^\top b)/\partial\beta = b$:
$$\frac{\partial L}{\partial \beta} = -2X^\top y + 2X^\top X\beta = 0$$

**Normal equations.**
$$X^\top X \hat{\beta} = X^\top y \quad \Longrightarrow \quad \hat{\beta} = (X^\top X)^{-1}X^\top y$$

**Confirm it is a minimum.** The Hessian is $2X^\top X$, which is positive semi-definite for any $X$ and positive definite when $X$ has full column rank — so the solution is unique and is a global minimum. The loss is convex, so there are no local minima to worry about.

**The geometry, which is the part worth saying.** The normal equations rearrange to $X^\top(y - X\hat{\beta}) = 0$: the residual vector is orthogonal to every column of $X$. So OLS is the orthogonal projection of $y$ onto the column space of $X$, with hat matrix $H = X(X^\top X)^{-1}X^\top$ and $\hat{y} = Hy$. $H$ is idempotent ($H^2 = H$) and symmetric, which is exactly what a projection matrix is. That is why "normal equations" — normal as in perpendicular. It also explains why adding a column can never increase the training SSE: you are projecting onto a larger subspace.

**Two properties that fall out immediately.** Unbiasedness: $E[\hat{\beta}] = (X^\top X)^{-1}X^\top E[y] = (X^\top X)^{-1}X^\top X\beta = \beta$, requiring only $E[\epsilon \mid X] = 0$. Variance: $\text{Var}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}$, requiring homoscedasticity and no autocorrelation. And if the model includes an intercept, the orthogonality condition against the column of ones forces the residuals to sum to zero.

**Follow-up: "What if $X^\top X$ isn't invertible?"** → Then the columns are linearly dependent and $\beta$ is not identified — the projection $\hat{y}$ is still unique, but the coefficients producing it are not. In practice: drop redundant columns, or use the Moore-Penrose pseudoinverse, which returns the minimum-norm solution. Ridge fixes it directly: $(X^\top X + \lambda I)$ is always invertible for $\lambda > 0$ since it adds $\lambda$ to every eigenvalue, which is why ridge was originally proposed for ill-conditioned design matrices, before anyone framed it as regularization. Numerically, you should not literally invert anyway — use a QR or SVD decomposition, which is far better conditioned.

---

### Q: State the Gauss-Markov theorem. What does it not require?

**Answer.** Under four conditions — (1) linearity in parameters, (2) strict exogeneity $E[\epsilon \mid X] = 0$, (3) homoscedasticity $\text{Var}(\epsilon_i \mid X) = \sigma^2$, (4) no autocorrelation $\text{Cov}(\epsilon_i, \epsilon_j \mid X) = 0$ — plus full column rank for identifiability, the OLS estimator is **BLUE**: the Best Linear Unbiased Estimator. "Best" means minimum variance in the class of estimators that are linear in $y$ and unbiased.

Each word in BLUE is a restriction, and the restrictions are the whole content of the theorem:

- **Linear** — only among estimators of the form $Ay$. Nonlinear estimators are not in the comparison class.
- **Unbiased** — only among unbiased estimators. Biased estimators are excluded, and this is the loophole that makes ridge and lasso useful: they are biased, so Gauss-Markov says nothing about them, and they routinely beat OLS in mean squared error.
- **Estimator of $\beta$** — a statement about coefficients, which also implies minimum-variance predictions at any $x$.

**What it does not require: normality of the errors.** This is the point of the question. Gauss-Markov holds for any error distribution with mean zero and constant finite variance. Normality is needed separately for exact t-tests and F-tests in finite samples, and for the stronger claim that OLS is the *minimum-variance unbiased* estimator among all estimators, not just the linear ones (via Cramér-Rao / Lehmann-Scheffé, since under normality OLS coincides with MLE and the sufficient statistics are complete).

Also worth stating: Gauss-Markov does not say OLS is good. If the model is misspecified — a missing confounder, so exogeneity fails — OLS is the best linear unbiased estimator of the wrong thing, or more precisely it is no longer unbiased at all. And in the presence of heteroscedasticity, OLS remains unbiased but loses the "best" property to GLS/WLS, which weights observations by inverse variance.

**Follow-up: "If OLS is BLUE, why would you ever use ridge?"** → Because minimum variance among *unbiased* estimators is not the same as minimum MSE. $\text{MSE} = \text{Bias}^2 + \text{Variance}$, and when predictors are correlated, $(X^\top X)^{-1}$ has large entries and the OLS variance is enormous. Ridge accepts a little bias to cut variance a lot. There is a theorem (Hoerl-Kennard) that for *any* $X$ and $\beta$, there exists some $\lambda > 0$ where ridge has strictly lower MSE than OLS — the improvement always exists, the only question is finding $\lambda$, which is what cross-validation is for.

---

### Q: What is $R^2$, and how does adjusted $R^2$ differ?

**Answer.** $R^2$ is the proportion of variance in $y$ explained by the model:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
It compares your model to the baseline of predicting $\bar{y}$ for everything. $R^2 = 0.30$ means you cut the squared error 30% relative to that baseline.

The structural problem: **$R^2$ never decreases when you add a predictor**, even pure noise. Adding a column expands the column space, and the projection of $y$ onto a larger subspace is at least as close. So $R^2$ cannot be used for model selection — with $p = n$ predictors of random noise you get $R^2 = 1$ exactly.

Adjusted $R^2$ penalizes parameters:
$$R^2_{adj} = 1 - \frac{SS_{res}/(n-p-1)}{SS_{tot}/(n-1)} = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$
It compares *mean* squares rather than sums, so a variable only helps if it reduces $SS_{res}$ by more than one degree of freedom's worth. It can decrease, and it can go negative.

Two things worth adding because they show judgment. First, $R^2$ is not a measure of model correctness. Anscombe's quartet is four datasets with identical $R^2 = 0.67$, identical means, variances, and regression lines, and wildly different structure — one is a perfect parabola, one is driven by a single outlier. Always plot residuals. Second, what counts as a good $R^2$ is entirely domain-dependent: $R^2 = 0.9$ in a physical system might indicate a bug or leakage, while $R^2 = 0.05$ in individual human behavior prediction can be genuinely valuable and highly significant.

And for causal work, $R^2$ is close to irrelevant. If I am estimating the effect of a price change, I care about the coefficient's point estimate and standard error and whether the identification is clean. A model with $R^2 = 0.02$ can give a precise, unbiased treatment effect if the treatment is randomized.

**Follow-up: "Should you use adjusted $R^2$ for model selection?"** → It is better than raw $R^2$ but I would not lead with it. AIC and BIC have cleaner theoretical grounding — AIC targets out-of-sample predictive loss, BIC approximates the marginal likelihood and is consistent for model recovery, with a heavier penalty ($\ln n$ versus 2 per parameter). And in practice, cross-validated error on held-out data directly measures the thing I care about and does not assume the model class is correct. Adjusted $R^2$'s penalty is fairly weak and it will still favor overfit models at large $p$.

---

### Q: What does it mean if $R^2$ is negative? What does it mean if $R^2$ is greater than 1?

**Answer.** Two very different situations, and the second one is the tell.

**$R^2 < 0$: real, and it means your model is worse than the mean.** By the formula $R^2 = 1 - SS_{res}/SS_{tot}$, negative $R^2$ means $SS_{res} > SS_{tot}$: predicting the constant $\bar{y}$ for everything would beat your model. Legitimate ways it happens:

1. **Out-of-sample evaluation.** You fit on train and compute $R^2$ on test using the *test* set's mean in the denominator. A badly overfit or drifted model easily does worse than the test mean. This is the most common case and it is not a bug — it is a diagnosis. In-sample OLS $R^2$ with an intercept can never be negative, because the intercept-only model is nested inside the fitted model.
2. **Regression through the origin.** Force the intercept to zero and the identity $SS_{tot} = SS_{reg} + SS_{res}$ breaks, and $R^2$ can go negative. This is also why $R^2$ is not comparable between with-intercept and no-intercept models — many libraries silently change the $SS_{tot}$ definition, so the numbers are not on the same scale.
3. **Coefficients not fit by OLS on this data** — a regularized fit, a model transferred from another dataset, hand-set coefficients.

So the answer to "$R^2 = -0.4$" is: the model is unusable as-is, and I would check for train/test distribution shift, leakage in the training setup, or a preprocessing mismatch between fit and predict.

**$R^2 > 1$: not possible. It is a bug.** There is no data-generating story that produces it, because $SS_{res} \geq 0$ forces $R^2 \leq 1$. So the correct answer is not to interpret it — it is to go find the defect. Where I would look, in order:

1. **Misaligned arrays.** `y_true` and `y_pred` in different orders, different lengths, or index-misaligned pandas Series that silently reindexed to NaN and then got dropped inconsistently.
2. **Wrong denominator.** Computing $SS_{tot}$ with the *training* mean while $SS_{res}$ uses test data, or using a hard-coded baseline variance from elsewhere.
3. **Sign or transform error.** Squaring residuals of already-squared values, mixing log-space and raw-space, or a metric function whose sign convention got flipped (some libraries return negated scores for maximization APIs — scikit-learn's `neg_mean_squared_error` pattern).
4. **A hand-rolled $R^2$ using the squared-correlation shortcut.** $R^2 = \text{corr}(y, \hat{y})^2$ is only equal to the standard definition for in-sample OLS with an intercept; applied elsewhere it is a different quantity, though notably it is bounded in $[0,1]$ so it would not itself produce $>1$.
5. **Weights.** Sample weights applied to one sum of squares and not the other.

The reason this exact question gets asked is that it separates people who pattern-match ("higher is better, so 1.2 is great") from people who know the algebra well enough to say "that value is impossible, so the code is wrong." The right instinct on any impossible metric value is to distrust the pipeline, not to invent an interpretation.

**Follow-up: "Your test $R^2$ is $-0.05$ and train $R^2$ is $0.85$. What's your diagnosis?"** → That gap is the signature of severe overfitting or of a train/test mismatch, and I would separate the two. Overfitting: too many parameters relative to $n$, no regularization, or a model that memorized noise — check by looking at the learning curve and by refitting with stronger regularization; if test $R^2$ climbs toward zero and above, it was variance. Mismatch: a preprocessing step fit on train and not reapplied identically at predict time (a scaler refit, a category encoding with unseen levels silently mapped to zero), temporal drift if the split is chronological, or leakage inflating the train number rather than deflating the test one. The tell between them is the *sign*: doing worse than the test mean is a strong result — random noise predictions would give roughly $R^2 = 0$, so $-0.05$ means the model is actively anti-correlated on some region, which points at a pipeline defect more than at plain variance. I would first check that the same transform pipeline runs in both paths, then check for a distribution shift between the splits, and only then tune capacity.

*Trap:* Trying to explain $R^2 > 1$ as overfitting or as "the model explains more than all the variance." Neither is a thing. It is a defect.

---

### Q: How do you detect heteroscedasticity, and what do you do about it?

**Answer.** Heteroscedasticity is non-constant error variance: $\text{Var}(\epsilon_i \mid X) = \sigma_i^2$ varying with $X$. It is the norm, not the exception, in real data — spending variance grows with income, prediction error grows with the magnitude of the target, count data has variance tied to the mean.

**What it costs.** $\hat{\beta}$ stays unbiased and consistent — this is the key point. What breaks is $\text{Var}(\hat{\beta})$: the usual formula $\sigma^2(X^\top X)^{-1}$ assumes constant variance, and under heteroscedasticity the true variance is the sandwich $(X^\top X)^{-1}X^\top \Omega X (X^\top X)^{-1}$. The conventional standard errors are wrong, typically too small in the common case where high-variance observations sit at high-leverage points, so you get t-statistics that are inflated and false significance. OLS also loses efficiency to WLS.

**Detection.**
- **Plot first.** Residuals versus fitted values. The classic signature is a funnel or cone widening to the right. Also plot residuals against each predictor and against time.
- **Breusch-Pagan test.** Regress squared residuals on the predictors and test whether they explain anything: $nR^2 \sim \chi^2_p$ under homoscedasticity. Assumes the heteroscedasticity is linear in $X$.
- **White test.** Same idea but includes squares and cross-products, so it catches nonlinear forms. More general, less powerful, and it is also a general misspecification test — a rejection could be a missing nonlinear term rather than pure heteroscedasticity.
- **Goldfeld-Quandt** if you suspect variance depends on one ordered variable: split the sorted data, compare residual variances with an F-test.

Caveat on all formal tests: at large $n$ they reject essentially always, since perfect homoscedasticity never holds. So the test answers "is there any," not "does it matter." Judge magnitude from the plot.

**Fixes, in order of what I actually reach for.**
1. **Robust (Huber-White) standard errors.** Keeps $\hat{\beta}$, replaces the variance estimate with the sandwich. One argument in every modern library. HC3 is the better small-sample variant; HC0 is fine above a few hundred rows. This is the default answer and costs nothing when errors are actually homoscedastic.
2. **Cluster-robust SEs** if the variance structure comes from grouping (users, markets, sessions), which is usually the real story.
3. **Transform the outcome.** $\log y$ or $\sqrt{y}$ often stabilizes variance when it scales with the mean — but this changes the estimand to a multiplicative one, so only if that interpretation is what you want.
4. **WLS**, weighting by $1/\hat{\sigma}_i^2$, if you can model the variance. Recovers efficiency but is sensitive to getting the variance model right.
5. **Use a GLM with the right variance function** — Poisson or negative binomial for counts, gamma for positive skewed outcomes. Often the heteroscedasticity is a symptom that the Gaussian likelihood was the wrong model.
6. **Bootstrap** the coefficients if the structure is complicated; a pairs bootstrap is naturally robust to heteroscedasticity.

**Follow-up: "Your residual plot shows a funnel. Is heteroscedasticity necessarily the problem?"** → Not necessarily. A funnel can also come from a missing nonlinear term or a missing interaction — misspecification shows up in residual plots too. And if the residual plot shows curvature *and* spread, fix the mean model first, because correcting the functional form frequently removes the apparent heteroscedasticity. Robust standard errors patch the inference on a wrong model; they do not make it right.

---

### Q: What is multicollinearity, how do you detect it, and does it always matter?

**Answer.** Multicollinearity is high linear dependence among predictors. Perfect collinearity makes $X^\top X$ singular and $\beta$ unidentified. Near-collinearity leaves it invertible but ill-conditioned.

**What it does.** It inflates coefficient variance. For predictor $j$:
$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n-1)\text{Var}(x_j)} \cdot \frac{1}{1 - R_j^2}$$
where $R_j^2$ is from regressing $x_j$ on all the other predictors. That last factor is the **variance inflation factor**:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
At $R_j^2 = 0.9$, VIF $= 10$, so the standard error is $\sqrt{10} = 3.2$ times larger than it would be with orthogonal predictors. Conventional flags are VIF $> 5$ or $> 10$, though those thresholds are rules of thumb, not theory.

**What it does not do**, and this is the part that separates answers: it does **not** bias the coefficients. $\hat{\beta}$ is still unbiased and still consistent. And it does **not** hurt predictive accuracy on data drawn from the same joint distribution — the fitted surface is fine, it is only the attribution among correlated inputs that is unstable. So the honest answer to "does it always matter" is **no**:

- If I only need predictions, and future data has the same correlation structure, I can ignore it. (If the correlation structure might change — a collinearity that holds in training but breaks in deployment — predictions degrade badly, so it is a robustness concern, not an accuracy one.)
- If I care about the coefficient on one specific variable and its collinearity is with control variables I do not need to interpret, I can ignore it. Collinearity among controls does not inflate the variance of the variable of interest unless they are correlated *with it*.
- If I am interpreting coefficients on the collinear variables themselves, it matters a lot: signs flip between samples, a variable is "insignificant" individually while the group is jointly highly significant, and coefficients change wildly when you add a row.

**Detection.** VIF per predictor; the correlation matrix (catches pairwise but misses three-way dependence, which is why VIF is better); the condition number of $X$ (ratio of largest to smallest singular value, with $>30$ flagged); and the smell test of a highly significant F-test with all individually insignificant t-tests.

**Fixes.** Drop one of the redundant variables (fine if they measure the same construct); combine them into an index or use PCA (costs interpretability); collect more data (VIF inflation is relative to $n$, so more data shrinks the absolute SE); center variables before creating interaction or polynomial terms, which removes most of the structural collinearity those induce; or use ridge, which is *designed* for this — it stabilizes the inverse and distributes weight across correlated predictors rather than picking arbitrarily. Note that lasso does the opposite: with two nearly identical predictors it picks one essentially at random, which is why elastic net exists.

**Follow-up: "You add a variable and another coefficient flips sign. What's going on?"** → Either collinearity or confounding, and telling them apart matters. If the two variables are highly correlated and both standard errors ballooned, it is collinearity — the data cannot separate their contributions and the flip is noise. If the standard errors stayed tight and the estimate moved decisively, that is omitted variable bias being corrected, and the new estimate is the better one — which is the Simpson's paradox situation. I would check VIF to distinguish, and look at whether the change is large relative to the original standard error.

---

### Q: What is omitted variable bias? Can you sign it?

**Answer.** Yes, and signing it is the answer.

Suppose the true model is
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$$
but I estimate the short regression $y = \alpha_0 + \alpha_1 x_1 + u$, leaving out $x_2$. Then $u = \beta_2 x_2 + \epsilon$, and if $x_1$ and $x_2$ are correlated, the error is correlated with the regressor — exogeneity fails. Working it out:

$$\text{plim } \hat{\alpha}_1 = \beta_1 + \beta_2 \cdot \frac{\text{Cov}(x_1, x_2)}{\text{Var}(x_1)} = \beta_1 + \beta_2 \delta$$

where $\delta$ is the slope from regressing the omitted $x_2$ on the included $x_1$. The bias is $\beta_2\delta$ — the product of (the omitted variable's effect on $y$) and (its correlation with the included variable). This gives the sign table:

| | $\text{Cov}(x_1,x_2) > 0$ | $\text{Cov}(x_1,x_2) < 0$ |
|---|---|---|
| $\beta_2 > 0$ | upward bias | downward bias |
| $\beta_2 < 0$ | downward bias | upward bias |

Two immediate consequences. **Bias vanishes if either factor is zero** — if the omitted variable does not affect $y$, or is uncorrelated with the included one. That is exactly why randomization solves the problem: randomizing $x_1$ forces $\text{Cov}(x_1, x_2) = 0$ for every conceivable $x_2$, observed or not. Randomization is a technology for setting $\delta$ to zero for all confounders simultaneously.

**Worked example.** Regress salary on years of education, omitting ability. Ability raises salary ($\beta_2 > 0$) and correlates positively with education ($\delta > 0$), so the estimated return to education is biased upward. If the true causal return is 6% per year, the naive regression might show 10%, and the extra 4 points are ability being credited to schooling. This is not fixed by more data — $n \to \infty$ converges to the biased value, which is why it is inconsistency, not just bias.

**What I do about it.** Control for the confounder if measured. If unmeasured: find an instrument (compulsory schooling laws, in the education literature), use a natural experiment / diff-in-differences, or bound the bias with a sensitivity analysis — Oster's delta or E-values quantify how strong an unobserved confounder would need to be to explain away the result. And I state the direction: even without measuring ability, I can say "this is an upper bound on the causal effect," which is often enough for the decision.

**Follow-up: "So should you just control for everything you have?"** → No, and this is where people over-correct. Controlling for a **mediator** (a variable on the causal path from $x_1$ to $y$) removes part of the effect you are trying to measure — you end up with a direct effect when you wanted the total effect. Controlling for a **collider** (a common consequence of $x_1$ and $y$) actively *creates* bias where none existed. And controlling for a post-treatment variable of any kind is dangerous. The right procedure is to draw the causal DAG and apply the backdoor criterion, not to throw every column into the regression. "Control for everything" is a specific, identifiable mistake.

---

### Q: Distinguish outliers, high-leverage points, and influential points.

**Answer.** Three different things that get conflated, and the distinction is operationally useful.

**Outlier:** a point with a large *residual* — unusual in $y$ given its $x$. Measured by studentized residuals; $|r_i| > 3$ is a common flag.

**High leverage:** a point with unusual *$x$* values, far from the centroid of the predictor space. Measured by the hat matrix diagonal $h_{ii} = x_i^\top(X^\top X)^{-1}x_i$, which satisfies $\sum_i h_{ii} = p$ and $0 \le h_{ii} \le 1$, so the average is $p/n$ and the usual flag is $h_{ii} > 2p/n$. Leverage is a property of the design only — it does not involve $y$ at all.

**Influential:** a point whose *removal* materially changes the fit. This is the intersection that matters: influence roughly requires both leverage and a large residual. Measured by **Cook's distance**,
$$D_i = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}$$
which cleanly factors into a residual term and a leverage term. Flags: $D_i > 4/n$ as a screen, or $D_i > 1$ as a strong signal. DFBETAS gives the per-coefficient version — how many standard errors coefficient $j$ moves when point $i$ is deleted — which is more useful when you care about one specific coefficient.

The three cases in words: a point far out in $x$ that sits right on the trend line has high leverage and no influence — it actually *reduces* the standard error of the slope and is a helpful observation. A point in the middle of the $x$ range with a big residual is an outlier with low influence — it inflates $\hat\sigma$ but barely moves the line, because it has no torque. A point far out in $x$ that is also off the line has high leverage *and* a large residual, and it can single-handedly determine the slope. Anscombe's fourth dataset is the pure version: eleven points, ten at $x=8$ and one at $x=19$, and the entire regression line is determined by that single point.

**What to do.** Never delete automatically. First, check if it is a data error — a sensor fault, a test account, a unit mix-up, a sentinel value like -999 or 1970-01-01. Those get fixed or removed with documentation. If the point is real, it is information: it may indicate the model is misspecified over part of the range, or that the population is a mixture. Options are to model it (add a term, add an indicator, segment), use a robust method (Huber regression, quantile regression, RANSAC), or — my preferred reporting move — run the analysis with and without it and show both. If the conclusion depends on one row out of fifty thousand, that fragility is the finding and stakeholders need to know it.

**Follow-up: "In an A/B test, a single user spends \$400,000. What do you do?"** → First verify it is real and not a test transaction or a duplicated event, and check which arm it landed in — a single point that determines the result is a red flag by itself. Assuming it is real: I would have pre-registered a capping rule, e.g. winsorize revenue at the 99.9th percentile, and applied it symmetrically to both arms. Then report the capped result as primary and the uncapped as a sensitivity check. Deciding to cap *after* seeing which arm the whale hit is p-hacking. I would also note that with tails this heavy, the mean-based test is underpowered regardless, and the durable fix is a bounded companion metric — purchase rate, or revenue-per-user capped — plus a longer horizon.

---

### Q: Why does regularization change the estimator's bias, and why is that a good trade?

**Answer.** Take ridge. The penalized loss is
$$L(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2$$
Differentiating: $-2X^\top y + 2X^\top X\beta + 2\lambda\beta = 0$, giving
$$\hat{\beta}_{ridge} = (X^\top X + \lambda I)^{-1}X^\top y$$

**Bias.** Take the expectation, using $E[y] = X\beta$:
$$E[\hat{\beta}_{ridge}] = (X^\top X + \lambda I)^{-1}X^\top X\beta$$
For $\lambda > 0$ the matrix $(X^\top X + \lambda I)^{-1}X^\top X$ is not the identity, so $E[\hat\beta] \neq \beta$. The bias is $-\lambda(X^\top X + \lambda I)^{-1}\beta$, which points toward zero — the estimate is shrunk.

**The eigenvalue picture makes it concrete.** In the basis of the eigenvectors of $X^\top X$ with eigenvalues $d_j^2$, ridge scales each OLS component by
$$\frac{d_j^2}{d_j^2 + \lambda}$$
Directions with lots of signal ($d_j^2 \gg \lambda$) are barely touched; directions where the data are nearly flat ($d_j^2 \ll \lambda$) get crushed toward zero. That is exactly right, because the low-eigenvalue directions are where OLS variance $\sigma^2/d_j^2$ blows up. Ridge is shrinking hardest precisely where OLS is least reliable.

**Variance.** $\text{Var}(\hat{\beta}_{ridge}) = \sigma^2 (X^\top X + \lambda I)^{-1}X^\top X(X^\top X + \lambda I)^{-1}$, which is strictly smaller than the OLS variance for $\lambda > 0$, in the matrix ordering sense.

**Why it is a good trade.** The relevant loss is
$$\text{MSE}(\hat\beta) = \text{Bias}^2 + \text{Variance}$$
As $\lambda$ increases from 0, bias grows from zero — quadratically at first, since it starts at a minimum — while variance falls linearly at first. So near $\lambda = 0$ the variance reduction always dominates, and there is always some $\lambda > 0$ that lowers MSE. That is the Hoerl-Kennard existence result. Gauss-Markov does not contradict this because ridge is biased and therefore outside the class the theorem is about.

**Bayesian reading, which is the cleanest one-liner.** Ridge is the MAP estimate under a Gaussian prior $\beta \sim N(0, \tau^2 I)$ with $\lambda = \sigma^2/\tau^2$. Lasso is the MAP under a Laplace prior. So the "bias" is just the prior belief that coefficients are small, and $\lambda$ is the strength of that belief — strong prior, more shrinkage, less data-driven variance.

**Follow-up: "Should you penalize the intercept?"** → No. Penalizing the intercept makes the estimator depend on the location of $y$ — shift every $y_i$ up by 100 and the fit changes in a way it should not. Standard implementations exclude it. Relatedly, ridge is not scale-invariant, so you must standardize the predictors before fitting, or a variable measured in dollars gets penalized differently than the same variable measured in cents. Forgetting to standardize is the most common practical ridge bug.


---

## 3. A/B Testing and Causal Inference

### Q: Design an A/B test end to end. We want to know if a new recommendation model increases engagement.

**Answer.** I'll go in order, and the discipline is that everything through step 6 is written down before launch.

**1. Hypothesis and decision.** State what shipping decision this test resolves. "The new model increases 7-day retained sessions per user; if it lifts by more than 1% without regressing latency or report rate, we ship." Naming the decision first prevents the test from becoming a fishing expedition.

**2. Metrics, in three tiers.** *Primary* — one metric, decided in advance: sessions per user over 7 days. One, because every additional primary metric is another chance at a false positive and another way to argue after the fact. *Secondary* — the mechanism metrics that explain *why* it moved: CTR on recommendations, dwell time, catalog diversity. These are for understanding, not for deciding. *Guardrails* — things that must not regress: p95 latency, crash rate, unsubscribes, report rate, revenue.

**3. Randomization unit.** User ID, not session and not request, because the treatment changes what a user sees over time and session-level assignment would give one user inconsistent experiences — contaminating the contrast and violating independence. Assignment by a hash of (user_id, experiment_salt) so it is deterministic, stateless, and independent of other experiments.

**4. Power and duration.** Historical $\sigma$ for sessions per user, target MDE of 1% relative, $\alpha = 0.05$, power 0.80. Plug into $n = 2(z_{\alpha/2}+z_\beta)^2\sigma^2/\Delta^2$. Then round the duration *up to whole weeks* — engagement has a strong weekly cycle and a partial week biases the comparison against whichever days it includes. Minimum one week even if power arrives sooner, because novelty effects need time to decay.

**5. Analysis plan, pre-committed.** Test statistic, variance estimator (cluster-robust at user level if the analysis unit is finer than the randomization unit), CUPED with the pre-period version of the primary metric, the winsorization rule and threshold, the segments I will look at (and that they are exploratory), and the multiple-comparison correction for secondaries. Write the stopping rule: fixed horizon, or a specific sequential procedure — not "we'll look and see."

**6. Validity checks defined up front.** SRM at $p < 0.001$; an A/A period or A/A simulation to confirm the pipeline gives uniform p-values; instrumentation parity so both arms log identically; and a check that the exposure trigger fires at the same point in both arms — logging exposure at a different moment in treatment is one of the most common silent biases.

**7. Run it.** Ramp: 1% for a day to check for crashes and SRM, then 10%, then 50/50. Do not evaluate the primary metric during ramp.

**8. Analyze and decide.** Report effect size with CI first, p-value second. Check guardrails. Check heterogeneity across a few pre-specified segments (new versus tenured, platform, geo) knowing these are exploratory and need correction. Then map to the decision stated in step 1.

**9. Post-launch.** Hold back a small long-term holdout to catch effects that decay or compound — a novelty win that reverses at 8 weeks looks like a launch success at 2 weeks.

**Follow-up: "The team wants to run this at 5% traffic to reduce risk. What do you say?"** → Risk containment is legitimate for a ramp, but 5% is not a valid *measurement* configuration if the control is 95%. The variance of the difference is dominated by the smaller arm: $\text{Var} = \sigma^2(1/n_T + 1/n_C)$, so with a 5/95 split at the same total traffic you have roughly 5.3x the variance of a 50/50 split, meaning about 2.3x the MDE. I would ramp at 5% for a day for safety, then move to 50/50 to actually measure, and be explicit that the safety ramp period is not part of the analysis window.

---

### Q: How do you choose the randomization unit?

**Answer.** Three constraints, in tension.

**1. It must be at or above the level where interference happens.** If treating one unit affects another's outcome, SUTVA is violated and the estimate is biased. In a two-sided marketplace, treating one rider changes driver availability for other riders, so rider-level randomization contaminates control. There the unit must be city-week or region.

**2. It must match user experience.** Randomizing at request level for a UI change means the interface flickers between versions, which is both a bad experience and a measurement of "inconsistency," not of the feature.

**3. Smaller units give more power.** Variance scales with the number of independent units, and coarser units mean far fewer of them. Going from 10 million users to 20 cities is a catastrophic power loss — you often cannot detect anything below a 5-10% effect with 20 clusters no matter how much per-cluster data you have.

So the rule is: **choose the coarsest unit that interference requires, and no coarser.** The ladder, from finest to coarsest: request → session → user → device/cookie → account/household → cluster (social graph community, geography) → time slice (switchback).

Specific situations worth naming:
- **Logged-out or cross-device.** Cookie-level assignment means one human can be in both arms across devices, which dilutes the effect toward zero. This is a known-and-accepted bias in web testing; quantify the dilution rather than pretending it does not exist.
- **Account-level products.** If five people share a workspace and one sees the new feature, randomize at the workspace level.
- **Social features.** Network interference; needs graph cluster randomization.
- **Pricing or supply changes.** Interference through the market; needs geo or switchback.

**The rule that gets people:** the analysis unit must be no finer than the randomization unit, or the standard errors are wrong. If you randomize by user and analyze per click, each user contributes many correlated rows and the naive standard error is badly understated. Fix by aggregating to the user first, or by using cluster-robust standard errors clustered on user, or by the delta method for the ratio metric.

**Follow-up: "How much power do you lose going from user to cluster randomization?"** → It depends on the intraclass correlation. The design effect is $1 + (m-1)\rho$, where $m$ is the average cluster size and $\rho$ the ICC. With clusters of 100 users and an ICC of only 0.05, the design effect is $1 + 99(0.05) = 5.95$ — you need six times the users for the same power. That is why cluster randomization is expensive and why you only pay it when interference genuinely demands it. It also shows the leverage: at fixed total traffic, many small clusters beat few large ones.

---

### Q: What is a minimum detectable effect and how do you pick one?

**Answer.** The MDE is the smallest true effect the test has the specified power (usually 80%) to detect at the specified $\alpha$. It is the inverse of the sample size formula:
$$\text{MDE} = (z_{1-\alpha/2} + z_{1-\beta})\sqrt{\frac{2\sigma^2}{n}}$$

The right way to pick it is **not** statistical — it is the smallest effect that would change the decision. Concretely: if the feature costs 3 engineer-months to maintain and each 1% lift in conversion is worth \$400k annually, and leadership requires a 3x return, then the break-even is roughly a 0.5% lift and the MDE should be 0.5%. Then check whether 0.5% is reachable: if not, that is a real finding — this test cannot answer the question, and you should say so before spending three weeks on it.

The common failure is picking the MDE by working backwards from available traffic and then presenting the result as if it were designed for the decision. If two weeks of traffic gives a 5% MDE and the plausible effect is 1%, the test is 90%+ likely to return null, and everyone will read null as "the feature does not work" rather than "we could not tell."

Two subtleties worth raising.

**Winner's curse.** Because you only publish effects that clear the significance bar, and the bar is roughly the MDE, statistically significant estimates from underpowered tests are systematically inflated. With 20% power, a significant result's point estimate is typically several times the true effect, and can even have the wrong sign (Gelman's Type M and Type S errors). So an underpowered significant win is not a conservative finding — it is an overstated one. Shrinkage or an empirical-Bayes correction against the historical distribution of effect sizes is the fix.

**MDE is not a promise about the effect.** "Our MDE is 1%" does not mean the effect is at least 1%; it means we can detect 1% with 80% probability if it exists. People routinely report the MDE as if it were an estimate.

**Follow-up: "How do you increase power without more traffic?"** → In order of leverage: (1) CUPED or a regression-adjusted estimator using pre-period covariates — a routine 30-50% variance reduction for user-level metrics with strong pre-period correlation; (2) pick a lower-variance primary metric — a binary "converted" rather than raw revenue, or a capped version; (3) trigger-based analysis, restricting to users who actually reached the surface being changed, which removes the never-exposed users who add pure noise and can multiply effective power severalfold; (4) stratified or blocked randomization on high-variance covariates; (5) a one-sided test if the decision is genuinely one-directional, worth about 20%; (6) accept $\alpha = 0.10$ if the cost asymmetry supports it. Trigger-based analysis and CUPED are usually the two biggest wins and neither requires anyone to compromise on rigor.

---

### Q: Why is peeking at results a problem, and by how much?

**Answer.** Because the significance threshold is calibrated for **one** decision at the end. If you check repeatedly and stop the first time $p < 0.05$, you have given yourself many chances at a false positive, and the actual Type I rate is far above 5%.

The mechanism: the running z-statistic follows a random walk that under $H_0$ wanders around zero. The fixed-horizon test asks whether it is outside $\pm 1.96$ at one specified moment. Peeking asks whether it *ever* crosses — a first-passage problem, and by the law of the iterated logarithm the random walk crosses any fixed boundary with probability 1 given unbounded time. So continuous monitoring with a fixed threshold has a Type I error rate approaching 100%.

The numbers, from Armitage's classic repeated-significance results with equally spaced looks at nominal $\alpha = 0.05$:

| Looks | Actual Type I rate |
|---|---|
| 1 | 5% |
| 2 | 8.3% |
| 3 | 10.7% |
| 5 | 14.2% |
| 10 | 19.3% |
| 20 | ~25% |
| continuous | → 100% |

So a team with a live dashboard checking daily on a two-week test has roughly a one-in-five to one-in-four chance of a false win on a pure null. That is the number to quote, because "peeking is bad" is not persuasive to a PM and "your dashboard turned a 5% error rate into 20%" is.

The asymmetric version is worse: teams peek and stop on wins, but keep running on losses hoping they turn around. That is optional stopping in one direction only, and it maximizes the bias.

**What to do instead**, since "don't look" is unrealistic when dashboards exist:
1. **Fixed horizon, committed in advance,** with the dashboard showing only guardrails and SRM during the run, not the primary metric.
2. **Group sequential designs** — pre-specify $k$ looks and spend $\alpha$ across them. O'Brien-Fleming spends very little early (making it very hard to stop at look 1) and nearly all at the end, so the final threshold is barely worse than 1.96; Pocock uses a constant, stricter nominal threshold (about 0.0158 for five looks) and stops early more readily. O'Brien-Fleming is the usual choice because it preserves final-analysis power.
3. **Always-valid / anytime-valid inference** — see the next question.

**Follow-up: "Someone shows you a result that hit significance on day 3 of a 14-day test. What do you say?"** → I would ask whether stopping at day 3 was pre-planned with an alpha-spending boundary. If not, the p-value is uninterpretable and the point estimate is inflated by the winner's curse — you stopped precisely because the random walk was at a local high, so the estimate is biased upward conditional on stopping. My recommendation would be to run the full 14 days and evaluate at the planned horizon, and to note that whether or not day 3 was significant is not evidence to weigh. If the result is genuine, the full test will confirm it at essentially no cost.

---

### Q: What is sequential testing / always-valid inference, and when would you use it?

**Answer.** It is a family of methods that give confidence sequences valid at *every* moment, so you can look continuously and stop whenever you like without inflating Type I error. The guarantee upgrades from "this interval covers the truth 95% of the time at the pre-specified $n$" to "with probability 95%, the true value is inside the interval at *all* times simultaneously."

Two main machineries:

**Group sequential (alpha spending).** Pre-specify $K$ interim analyses and an alpha-spending function $\alpha(t)$ allocating the total 5% across them. O'Brien-Fleming boundaries are wide early and converge to near-1.96 at the end; Pocock uses constant, tighter boundaries throughout. Requires knowing $K$ and the information fractions in advance. Standard in clinical trials, which is where the whole apparatus comes from — trials need early stopping for ethical reasons.

**Always-valid / e-value and mixture-martingale methods.** These give a confidence sequence with no pre-specified stopping rule at all, built on Ville's inequality applied to a nonnegative supermartingale (the likelihood-ratio process under the null). E-values and their running products (test martingales) let you combine evidence across time or studies by multiplication and remain valid under arbitrary optional stopping and optional continuation. This is the technology behind commercial "always-valid p-values" in Optimizely, Statsig, Eppo, and similar platforms.

**The cost, which you must state.** Anytime validity is not free — you pay in width. An always-valid confidence sequence is wider than a fixed-horizon CI at the same nominal level, typically 20-80% wider depending on the method and where you are relative to the planned horizon, because it is protecting against every possible stopping time. So if you truly can commit to a fixed horizon, the fixed-horizon test is more efficient and you should use it.

**When I would use each.** Fixed horizon: standard offline experiments where two weeks is fine and nobody needs to act early. Group sequential: expensive or long tests where early stopping has real value, and the look schedule is knowable. Always-valid: when the organization *will* peek regardless — a self-serve platform, or a culture where PMs watch dashboards — and when you need automated stopping rules. Also for safety guardrails, where you want to detect a regression the moment it appears; there the extra width is a fine price for the ability to halt a harmful experiment immediately.

**Follow-up: "Isn't Bayesian A/B testing immune to peeking?"** → Partly, and the claim is often overstated. A Bayesian posterior is a valid summary of the evidence given the data and prior regardless of stopping rule, because the likelihood principle means the stopping rule does not enter the likelihood. But if you define a *decision rule* — "ship when $P(\text{lift} > 0) > 0.95$" — and evaluate it continuously, its frequentist error rate is inflated by optional stopping just like a p-value's, and with a diffuse prior it degenerates to the same random-walk crossing problem. Bayesian methods control peeking damage well when the prior is genuinely informative, because the prior anchors early estimates and prevents small-$n$ noise from crossing the threshold. So the honest answer: Bayesian inference changes what the number means, and an informative prior helps a lot in practice, but it is not a free pass on false-positive control.

---

### Q: You're testing 20 metrics. How do you handle multiple comparisons?

**Answer.** With 20 independent tests at $\alpha = 0.05$, the family-wise error rate is $1 - 0.95^{20} = 0.64$. You expect one false positive per experiment and have a coin-flip-plus chance of at least one. Reporting the winner from 20 metrics without correction is guaranteed to produce spurious launches.

**Two different error rates, and the choice between them is the answer.**

**FWER — probability of *any* false positive.** Bonferroni: test each at $\alpha/m = 0.0025$. Simple, valid under any dependence, but very conservative — power collapses as $m$ grows. Holm-Bonferroni is a strictly more powerful step-down version with the identical FWER guarantee, so there is no reason to use plain Bonferroni: sort p-values ascending and compare $p_{(i)}$ to $\alpha/(m - i + 1)$, stopping at the first failure. Šidák, $1-(1-\alpha)^{1/m}$, is marginally less conservative but assumes independence.

**FDR — expected *proportion* of discoveries that are false.** Benjamini-Hochberg: sort ascending, find the largest $k$ with $p_{(k)} \leq \frac{k}{m}\alpha$, reject all up to $k$. Much more powerful, and the guarantee scales sensibly with the number of true effects. Valid under independence and under positive regression dependence (PRDS); Benjamini-Yekutieli covers arbitrary dependence at the cost of a $\ln m$ factor.

**When each.** FWER when a single false positive is expensive and the tests are few and confirmatory — the primary metric of a launch decision, a regulatory endpoint, a safety claim. FDR when you are screening many hypotheses and expect several real effects, and a few false ones are tolerable because they will be filtered downstream — a metrics dashboard with 200 metrics, feature selection, or hunting for heterogeneous treatment effects across segments.

**What I actually do in an A/B test**, which is the answer that reads as practical: I do not correct across the tiers uniformly. One **primary** metric at $\alpha = 0.05$, uncorrected, because it is one pre-registered test and the decision rests on it. **Secondary** metrics with BH, reported as supporting evidence, not as decision criteria. **Guardrails** deliberately *uncorrected* and often at a looser $\alpha$ — I want to be sensitive to harm, so I accept false alarms there; correcting guardrails would be optimizing in the wrong direction. **Segment analyses** treated as strictly exploratory and hypothesis-generating, requiring a confirmatory test before anyone acts.

The structural fix that beats any correction: pre-register the primary metric. Multiplicity is a problem created by having many chances to declare victory, and the cheapest way to control it is to only have one.

**Follow-up: "Are your 20 metrics independent?"** → Almost never — clicks, sessions, and time-on-site all move together, and they share the same users and the same randomization. Positive correlation makes Bonferroni *even more* conservative, since the true FWER is below the nominal bound. BH is valid under PRDS, which positively correlated metrics typically satisfy, so BH is the safer practical choice here. If I needed exact FWER control under this correlation structure, I would use a permutation-based max-T procedure, which learns the joint null distribution of the maximum statistic from the data and is exact regardless of dependence.

---

### Q: What are novelty and primacy effects, and how do you detect them?

**Answer.** Both are cases where the treatment effect changes over time because users are *adapting*, so a short test estimates something other than the steady-state effect.

**Novelty:** users engage with a change because it is new. A redesigned button gets clicked because it looks different, not because it is better. The effect is positive early and decays, sometimes to zero or negative. This makes short tests over-optimistic.

**Primacy (change aversion):** users are habituated to the old flow and are temporarily worse off relearning. The effect starts negative and improves. This makes short tests over-pessimistic and causes teams to kill good long-term changes.

**Detection.**
1. **Plot the daily treatment effect, not just the cumulative one.** The cumulative estimate mechanically smooths a trend into a milder average and hides the shape. A monotone decay in the daily effect is the novelty signature.
2. **Compare new users to existing users.** New users have no prior expectation, so they cannot experience novelty or primacy relative to the old design. If existing users show +5% and brand-new users show 0%, it is almost certainly novelty. If new users show +5% and existing users show 0%, primacy. This is the single most diagnostic cut.
3. **Compare first-exposure to later-exposure behavior** within treated users — effect by days-since-first-exposure.
4. **Run longer, or hold back.** The definitive answer is a long-term holdout: keep 1-5% of users in control for months after launch and measure the persistent effect.

**Adjustments.** Analyze after excluding the first week ("burn-in"), if you pre-registered that. Fit the effect as a function of exposure time and extrapolate the asymptote. For real decisions on high-stakes changes, the long-term holdout is the only thing that actually settles it.

Worth adding the general framing: novelty and primacy are special cases of the treatment effect being non-constant in time, which means the estimand itself is ambiguous. "The effect of the feature" is not well defined without specifying a horizon. Saying that explicitly — that we are estimating a 2-week effect and the decision requires a 6-month effect, and here is how we bridge the gap — is what a senior answer sounds like.

**Follow-up: "Your test shows +8% in week 1 and +1% in week 2. Do you ship?"** → I would not ship on that data, and I would not kill it either. The trend is consistent with novelty, and extrapolating suggests the steady state may be near zero. I would extend the test another two weeks to see if it stabilizes at +1% or keeps decaying, and simultaneously cut by user tenure to check the new-user signal. If new users show a flat +1% throughout while existing users decayed from +8% to +1%, I would call the true effect roughly +1% and decide on whether 1% justifies the cost. I would also check that nothing else changed between weeks — a marketing push or a holiday can produce the same pattern for a completely different reason.

---

### Q: What is network interference and how do you handle it?

**Answer.** Interference is a violation of SUTVA — specifically the "no interference" part, which says unit $i$'s outcome depends only on unit $i$'s treatment. When a treated user's behavior affects a control user, control is contaminated, and the naive difference is biased. Critically, the bias can go either direction and can even flip the sign.

Where it shows up:
- **Social products.** Treat Alice with a better sharing flow, she shares more, her control-group friends see more content and engage more. Control's outcome rises, so the measured difference *understates* the true effect.
- **Marketplaces.** Give treated riders better pricing, they book more rides, driver supply for control riders drops. Control gets worse, so the difference *overstates* the effect — you measure a redistribution as if it were creation. This is the more dangerous direction because it manufactures wins that vanish at full rollout.
- **Shared resources.** Treated users consume more cache or compute, degrading latency for everyone including control.
- **Learning systems.** Both arms feed one shared model, so treatment behavior changes the model that serves control.

**Handling it, roughly in order of cost.**

1. **Cluster randomization.** Partition the graph into communities with few cross-edges (Louvain, METIS, or balanced graph partitioning) and randomize whole clusters. Most interference becomes within-cluster and therefore inside a single arm. Cost: massive power loss via the design effect $1 + (m-1)\rho$, and residual cross-cluster edges leave residual bias.
2. **Geo / market-level randomization.** Natural clusters for marketplaces and pricing. Very few units — typically tens of markets — so you need synthetic control or diff-in-differences machinery to get any precision.
3. **Switchback / time-based randomization.** Alternate the whole market between treatment and control over time slices. Handles interference perfectly within a slice but introduces carryover across slices.
4. **Ego-cluster designs.** Randomize a focal user together with their immediate neighborhood, so a treated ego has treated friends.
5. **Exposure-based / two-stage designs.** Randomize clusters into "high saturation" and "low saturation," then randomize individuals within, which lets you *estimate* the spillover as a function of the fraction treated rather than only removing it. This is the most informative design when you can afford it.
6. **Measure it rather than remove it.** Compare a naive user-level test against a cluster-level test on the same feature; the gap estimates the interference bias, and if it is negligible you can go back to cheap user-level testing for that class of change.

**Follow-up: "How do you know if interference is a real concern before running?"** → Reason about the causal mechanism first: does one user's treatment plausibly change another's experience through a shared resource, a market, a social graph, or a shared model? If yes, it is a concern. Empirically, run the same experiment both ways once — user-randomized and cluster-randomized — and compare. Uber, LinkedIn, and Meta have published exactly this kind of calibration. It is expensive as a one-time investment and cheap thereafter, because it tells you which categories of change need clusters and which do not.

---

### Q: Explain Simpson's paradox with a worked example.

**Answer.** Simpson's paradox is when an association present in every subgroup reverses when the subgroups are pooled. It arises when a confounder is associated both with group membership and with the outcome, and is distributed unevenly across the groups.

**Worked example — a checkout redesign.**

| | Desktop | Mobile | **Overall** |
|---|---|---|---|
| **Control** | 90 / 900 = **10.0%** | 1 / 100 = **1.0%** | 91 / 1000 = **9.1%** |
| **Treatment** | 11 / 100 = **11.0%** | 18 / 900 = **2.0%** | 29 / 1000 = **2.9%** |

Treatment wins on desktop (11% vs 10%) and wins on mobile (2% vs 1%) — it doubles mobile conversion — yet loses overall by a factor of three (2.9% vs 9.1%).

The arithmetic is transparent once you see it: device type is a huge driver of conversion (desktop converts 10x better than mobile), and the arms have wildly different device mixes — control is 90% desktop, treatment is 90% mobile. The overall rate is a weighted average, and the weights differ. Control's overall number is mostly its desktop rate; treatment's is mostly its mobile rate. The comparison is desktop-versus-mobile wearing a treatment-versus-control costume.

**Which number is right?** The subgroup one, here — because the imbalance is a *pre-treatment* confounder. Device type cannot be caused by the treatment, so the correct estimate conditions on it: a stratified or weighted estimate using the population's real device mix. Pooling gives a biased answer.

**But not always.** If the stratifying variable is *post-treatment* — say the treatment changes which page users land on, and you stratify by landing page — then conditioning on it blocks part of the causal effect and the *pooled* number is the correct one. The famous kidney stone case is the confounder version (severity drives both treatment choice and success, so stratify). The rule is not "always disaggregate"; it is "draw the causal graph and decide which variables belong in the adjustment set." The paradox is purely statistical; the resolution is always causal.

**In practice, in an A/B test.** A device imbalance this extreme means randomization is broken — check SRM immediately, because with proper randomization the device mix would match. The realistic version is milder: a 51/49 mix difference producing a small pooled bias, which is exactly what stratified randomization or post-stratification / CUPED is designed to eliminate.

**Follow-up: "How do you protect against this in advance?"** → Stratified randomization on the high-impact covariates — assign within device, geo, and tenure strata so the arms are balanced by construction. If that is not feasible at assignment time, post-stratify or use regression adjustment / CUPED at analysis time, which recovers most of the benefit. And always run SRM checks not only overall but *within key segments*, because a global 50/50 split can hide compensating per-segment imbalances.

---

### Q: The primary metric is flat but a secondary metric is up 5% with $p = 0.01$. What do you do?

**Answer.** I do not ship on the secondary, and the reason is structural rather than a judgment call.

First, the statistics. If I tested 15 secondary metrics, then at $\alpha = 0.05$ I expect 0.75 false positives per experiment, and $p = 0.01$ among 15 tests has a BH-adjusted value around 0.15 — not significant once multiplicity is accounted for. So step one is: how many metrics were tested, and what does this p-value look like after correction? Very often that ends the conversation.

Second, and more important, the pre-registration point. The primary metric was chosen in advance precisely so that the decision could not be relitigated after seeing the data. Promoting a secondary metric to decision-maker post hoc converts the experiment into an exploratory analysis, and the false positive rate is whatever the analyst's search process was — unknowable. If the secondary metric were genuinely the thing we cared about, it should have been primary.

Third, what I would actually say and do:

- **Report it honestly as a hypothesis, not a result.** "The primary was flat, CI $[-0.4\%, +0.5\%]$, so we can rule out effects larger than half a percent. Metric X moved +5%, which after correction is not significant, and it is a plausible lead."
- **Check coherence.** Does the secondary movement have a mechanism consistent with the change, and do related metrics move in the same direction? A single isolated metric moving with no supporting pattern is more likely noise; a coherent cluster (CTR up, dwell up, scroll depth up) is more credible even if only one crossed the threshold.
- **Check it is not a guardrail failure in disguise** — a secondary "win" that comes from cannibalizing something else.
- **Run a confirmatory test with that metric as primary,** powered for it. That is the only way to convert the lead into evidence.

And I would push back on a framing that shows up here: "the primary was flat, so at least it didn't hurt, and the secondary is up, so ship." Flat plus a noisy secondary is not evidence of benefit, and shipping accumulates unvalidated complexity that permanently raises maintenance cost and dilutes future experiments.

**Follow-up: "The VP really wants to ship it. What's your actual recommendation?"** → I would separate the statistical claim from the business decision and let them own the second one with accurate information. Something like: "The evidence does not support a claim that this improves metric X — after correction, the result is consistent with noise. If you want to ship for strategic reasons — it unblocks the roadmap, it is already built, the downside is bounded because guardrails were clean — that is a legitimate call, and I would support it with a long-term holdout so we learn the truth within a quarter." That gives them the decision while keeping the evidentiary record honest, and the holdout means the organization is not making the same mistake next quarter.

---

### Q: What are guardrail metrics and how do you set thresholds for them?

**Answer.** Guardrails are metrics that must not regress, regardless of what the primary metric does. They exist because optimizing one metric almost always degrades something else, and without explicit guardrails those degradations accumulate invisibly across many launches.

Three kinds:

1. **Business guardrails** — revenue, retention, subscription cancellations. A recommendation change that boosts engagement by pushing clickbait may hurt long-term retention.
2. **User experience / trust guardrails** — p95 and p99 latency, crash rate, error rate, report/block rate, unsubscribes, support contacts.
3. **Validity guardrails** — sample ratio mismatch, arm-level instrumentation counts, exposure-rate parity. These are not about the feature; they test whether the experiment itself is trustworthy, and they should be checked first and block interpretation if they fail.

**Setting thresholds — the key inversion.** Guardrails are *non-inferiority* tests, not superiority tests. The question is not "did it regress significantly?" but "can we rule out a regression larger than we can tolerate?" So you specify a margin $\delta$ and test $H_0: \text{effect} \leq -\delta$ against $H_1: \text{effect} > -\delta$. In practice this is read off the confidence interval: **if the lower bound of the CI is above $-\delta$, the guardrail passes.**

This inversion has an important consequence: an underpowered test *passes* every guardrail trivially, because a wide CI never shows a significant regression. Treating "not significantly worse" as "safe" rewards low power. The CI-based non-inferiority framing fixes it — a wide CI fails to establish non-inferiority, which is the correct outcome.

I also set $\alpha$ deliberately loosely on guardrails and do **not** apply multiple-comparison corrections to them. The cost asymmetry runs the other way: a false guardrail alarm costs an investigation, a missed regression ships harm. Corrections would make me less sensitive to harm, which is backwards.

Margins come from the business, not from statistics: "latency must not regress by more than 20 ms at p95," "revenue per user must not drop more than 0.5%," "crash rate must not rise at all detectably." Write them down before launch alongside what happens if one trips — auto-halt for severe ones, review for the rest.

**Follow-up: "The primary is up 3% and one guardrail — unsubscribe rate — is up 2% with a wide CI. What do you do?"** → Wide CI means I cannot rule out a meaningful regression, so this is not a pass. Unsubscribes are also a leading indicator of a durable, hard-to-reverse loss: a churned user does not come back, while an engagement gain accrues only while they stay. I would extend the test to tighten that interval, and in parallel estimate the exchange rate — how much lifetime value does 2% more unsubscribes cost, versus what 3% more engagement is worth? If the LTV math shows the engagement gain dominates by a wide margin even at the pessimistic end of the CI, shipping is defensible. If it is close, I want the extra data before committing to something asymmetric and irreversible.

---

### Q: What is CUPED and why does it work?

**Answer.** CUPED — Controlled experiments Using Pre-Existing Data — is variance reduction using pre-experiment data. Instead of analyzing the raw metric $Y$, analyze

$$Y_{cuped} = Y - \theta(X - \bar{X})$$

where $X$ is a covariate measured *before* the experiment started (most commonly the same metric in the pre-period) and $\theta = \text{Cov}(Y,X)/\text{Var}(X)$ — which is exactly the OLS slope of $Y$ on $X$.

**Why it is unbiased.** $E[X - \bar{X}] = 0$ across the whole population and, because $X$ is pre-treatment, its distribution is identical in both arms in expectation. So subtracting it removes noise without shifting the treatment contrast: $E[Y_{cuped}^T - Y_{cuped}^C] = E[Y^T - Y^C]$.

**Why the variance drops.**
$$\text{Var}(Y_{cuped}) = \text{Var}(Y) + \theta^2\text{Var}(X) - 2\theta\text{Cov}(Y,X)$$
Minimizing over $\theta$ gives the $\theta$ above, and substituting back:
$$\text{Var}(Y_{cuped}) = \text{Var}(Y)(1 - \rho^2)$$
where $\rho = \text{corr}(Y, X)$. So the variance reduction is exactly $\rho^2$.

**Concrete numbers.** With $\rho = 0.7$, variance drops 49% — nearly halving required sample size, or equivalently letting you run the same test in half the time. Since MDE scales with $\sigma$, the MDE shrinks by $\sqrt{1-\rho^2} = \sqrt{0.51} = 0.71$, a 29% smaller detectable effect. At $\rho = 0.5$, a 25% variance cut. Below about $\rho = 0.3$ it is barely worth the pipeline complexity.

**Where the correlation comes from.** Pre-period values of the same metric are usually the best predictor: a user's sessions last month strongly predicts sessions this month. That is why CUPED works so well for user-level engagement metrics and poorly for metrics with no pre-period analogue.

**Requirements and gotchas.**
- The covariate must be strictly **pre-treatment**. Using an in-experiment covariate that treatment can affect introduces bias — this is the same mistake as controlling for a mediator.
- New users have no pre-period. Standard handling: impute zero and include a "is new user" indicator, or stratify and apply CUPED only to the returning cohort.
- $\theta$ should be estimated on **pooled** data across arms, not per-arm, to avoid introducing a difference.
- Generalizations: multiple covariates via regression adjustment (Lin's estimator, with treatment-covariate interactions, which is unbiased asymptotically and never hurts precision), or ML-predicted outcomes as the covariate (MLRATE), which can push $\rho$ higher.

The conceptual point worth stating: CUPED is just ANCOVA / regression adjustment. Regressing $Y$ on treatment plus pre-period covariates gives essentially the same estimator. Knowing that connects it to the linear regression section rather than leaving it as a piece of A/B-testing folklore.

**Follow-up: "Can you use a covariate measured during the experiment?"** → Only if it cannot possibly be affected by treatment — country, device type at signup, account age. Anything downstream of the treatment is a mediator or a collider, and adjusting for it biases the effect estimate, potentially severely. The safe rule is: if it was recorded before the assignment timestamp, it is fine; if after, you need an argument for why treatment cannot touch it, and the default should be to exclude it.

---

### Q: What is a switchback test and when do you need one?

**Answer.** A switchback randomizes *time periods* rather than units: the entire system (or an entire market) alternates between treatment and control across time slices, and you compare outcomes in treatment slices to control slices.

**When you need it.** When interference makes unit-level randomization invalid and the interference operates through a shared, time-varying resource:
- **Marketplace pricing or matching.** Every rider draws from the same driver pool, so a treated rider's booking removes supply from control. Uber, Lyft, DoorDash, and Instacart use switchbacks for dispatch and pricing changes.
- **Global system changes** with no per-user version — a shared cache policy, an infrastructure change, a market-wide inventory allocation.
- **Any change to a shared model or shared pool** where per-user isolation is impossible.

**Design choices, and each is a real trade-off.**
- **Slice length.** Short slices give more randomization units and more power; long slices reduce carryover contamination. The tension is direct: if switching to treatment pricing takes 20 minutes for supply to re-equilibrate, 30-minute slices are almost entirely carryover. Typical choices are 30 minutes to 6 hours, chosen so slice length comfortably exceeds the system's relaxation time.
- **Burn-in.** Discard the first portion of each slice while the system equilibrates, and analyze only the stable remainder. This directly addresses carryover at the cost of throwing away data.
- **Randomization scheme.** Independent coin flips per slice, or balanced/blocked designs that guarantee equal treatment exposure across time-of-day and day-of-week — important because demand is strongly cyclical and an unlucky randomization can confound treatment with rush hour.
- **Region crossing.** Switchback across (region × time) cells rather than time alone, which multiplies the number of units and greatly improves power.

**Analysis.** The effective sample size is the number of *slices*, not the number of orders, so power is usually the binding constraint. Slices are serially correlated (traffic at 5pm looks like traffic at 5pm yesterday), so you need standard errors robust to that — clustered by day, block-bootstrapped, or a model with time-of-day and day-of-week fixed effects to absorb the cycle. Ignoring serial correlation and computing naive standard errors on order-level data is the standard mistake and understates uncertainty by a lot.

**Costs.** Low power relative to user-level tests; carryover bias that is hard to fully eliminate; a possibly inconsistent user experience (a rider sees different pricing at different times, which can itself cause behavior change); and susceptibility to time-varying confounders like weather or a competitor's promotion, which is why balanced designs and long horizons matter.

**Follow-up: "How do you choose the slice length empirically?"** → Estimate the system's relaxation time from historical data: after a shock — a surge event, an outage, a pricing change — how long until the key metrics return to baseline? Set the slice length to comfortably exceed that, and validate with an A/A switchback: run the design with both arms identical and confirm the estimated effect is centered on zero with correct nominal coverage. If the A/A shows bias or inflated false positives, the slices are too short or the carryover model is wrong. Some teams also run the analysis at several burn-in lengths and check that the estimate is stable, which is a cheap sensitivity check.

---

### Q: Explain difference-in-differences. What is the key assumption?

**Answer.** DiD estimates a causal effect from observational data when a treatment hits some units at a known time and not others. It removes both time-invariant differences between groups and common shocks over time.

$$\hat{\tau}_{DiD} = (\bar{Y}_{T,post} - \bar{Y}_{T,pre}) - (\bar{Y}_{C,post} - \bar{Y}_{C,pre})$$

The treated group's change minus the control group's change. Equivalently, as a regression:
$$Y_{it} = \alpha + \beta \cdot \text{Treat}_i + \gamma \cdot \text{Post}_t + \tau (\text{Treat}_i \times \text{Post}_t) + \epsilon_{it}$$
where $\tau$, the interaction coefficient, is the DiD estimate. In panel form you use unit and time fixed effects, which generalizes the same idea.

**Worked example.** A feature launches in Canada but not the US. Canada's engagement goes from 100 to 118 (+18). The US goes from 90 to 100 (+10) over the same window. DiD $= 18 - 10 = 8$. The reasoning: 10 of Canada's 18-point gain is attributable to whatever moved both countries — seasonality, a platform-wide change — leaving 8 as the feature's effect.

**The key assumption: parallel trends.** Absent treatment, the treated group's outcome would have followed the same *trend* as control. Note what this does and does not require: the levels can differ arbitrarily (that is exactly what the design differences out), but the counterfactual trajectories must be parallel. This is fundamentally untestable, since it concerns an unobserved counterfactual.

**How to support it.**
1. **Plot pre-period trends over many periods.** Parallel pre-trends are the standard evidence. They are supportive, not proof — pre-trends can be parallel and then diverge for reasons unrelated to treatment.
2. **Event-study specification.** Estimate a separate coefficient for each period relative to treatment, normalizing the period before treatment to zero. You should see flat, near-zero coefficients pre-treatment and a jump after. This is the standard modern presentation and it displays the pre-trend evidence and the dynamic effect together.
3. **Placebo tests.** Run DiD on a fake treatment date before the real one, and on outcomes the treatment should not affect. Both should give null results.
4. **Multiple control groups** or a synthetic control — a weighted combination of untreated units constructed to match the treated unit's pre-period trajectory.

**Common threats.** Differential shocks hitting one group (a Canadian holiday); compositional changes in who is in each group over time; anticipation, where units change behavior before the official treatment date; and spillovers from treated to control units. Inference is also a known trap: with serially correlated panel data, naive standard errors are far too small (Bertrand-Duflo-Mullainathan showed placebo DiD rejecting at 45% instead of 5%), so cluster standard errors at the unit level, and with few clusters use a wild cluster bootstrap.

**Follow-up: "What if treatment rolls out to different units at different times?"** → That is staggered adoption, and the recent literature (Goodman-Bacon; Callaway and Sant'Anna; Sun and Abraham; de Chaisemartin and D'Haultfœuille) showed that the standard two-way fixed effects estimator is biased there. The reason is that TWFE implicitly uses already-treated units as controls for later-treated units, and if effects are heterogeneous or change over time, those comparisons get *negative* weights — the estimate can even have the opposite sign of every underlying effect. The fix is one of the modern estimators that only ever compares treated units to not-yet-treated ones and aggregates group-time effects with sensible weights. Knowing that TWFE-with-staggered-timing is a known failure is a strong signal you have used this method for real.

---

### Q: Explain instrumental variables. Give an example.

**Answer.** An instrument $Z$ lets you recover a causal effect of $X$ on $Y$ when $X$ is endogenous — correlated with unobserved confounders. The idea is to use only the variation in $X$ that comes from $Z$, which is by construction unconfounded.

**Three conditions:**
1. **Relevance:** $Z$ actually shifts $X$, i.e. $\text{Cov}(Z, X) \neq 0$. This one is testable — it is the first-stage F-statistic.
2. **Exclusion restriction:** $Z$ affects $Y$ *only through* $X$, with no direct path. Untestable, and it must be argued from domain knowledge.
3. **Independence / exogeneity:** $Z$ is as good as randomly assigned with respect to the unobserved confounders. Also untestable.

**Estimator.** In the simple case, the Wald estimator:
$$\hat{\tau}_{IV} = \frac{\text{Cov}(Z,Y)}{\text{Cov}(Z,X)} = \frac{\text{effect of } Z \text{ on } Y}{\text{effect of } Z \text{ on } X}$$
The general version is two-stage least squares: regress $X$ on $Z$ to get $\hat{X}$, then regress $Y$ on $\hat{X}$. (Standard errors must come from a 2SLS routine, not from naively running two OLS regressions, because the second stage must account for $\hat X$ being estimated.)

**Example I'd actually use — an encouragement design.** We want the effect of adopting a new feature on retention. We cannot randomize adoption, because we cannot force anyone to use it, and adopters differ systematically from non-adopters — they are more engaged to begin with, so the naive comparison is hopelessly confounded. So we randomize an *encouragement*: a prompt nudging users to try it. $Z$ = received the prompt (randomized, so independence holds by design). $X$ = adopted the feature. $Y$ = retention.

Numbers: adoption is 40% among un-prompted and 60% among prompted, so the first stage is 20 percentage points. Retention is 2 points higher among prompted (the intent-to-treat effect). Then
$$\hat{\tau} = \frac{0.02}{0.20} = 0.10$$
a 10 percentage point retention effect **for compliers** — users who adopt because of the prompt but would not otherwise.

**That "for compliers" is the crucial caveat.** IV with heterogeneous effects identifies the **LATE**, the local average treatment effect on compliers, and it requires a fourth assumption: **monotonicity** (no defiers — nobody adopts *because* they were not prompted). Compliers are a subpopulation you cannot even identify individually, and their treatment effect can differ from always-takers' and never-takers'. So the number does not directly answer "what if we forced everyone to adopt."

**Weak instruments.** If the first stage is weak, IV is badly biased toward the OLS estimate and standard errors are unreliable — and the bias is worse in large samples in a way that feels counterintuitive. The rule of thumb is first-stage $F > 10$; more recent work argues that is far too lenient and you may want $F$ in the tens or hundreds for reliable inference. Always report the first stage.

**Follow-up: "Give me an example where the exclusion restriction fails."** → Using distance-to-hospital as an instrument for receiving a treatment. Distance plausibly shifts treatment probability (relevance is fine), but it also correlates with rurality, income, air quality, and access to other care — all of which affect health outcomes directly. The exclusion restriction fails and the estimate absorbs those paths. Even the encouragement design can fail it: if the prompt itself reminds users the product exists and increases engagement through channels other than adoption of the specific feature, then $Z$ affects $Y$ not only through $X$, and the IV estimate is biased upward. Designing the encouragement to be as narrowly targeted as possible is how you defend the assumption.

---

### Q: What is propensity score matching and what are its limitations?

**Answer.** The propensity score is $e(x) = P(T = 1 \mid X = x)$, the probability of treatment given observed covariates. Rosenbaum and Rubin's result is that if treatment is unconfounded given $X$, it is also unconfounded given the *scalar* $e(X)$ — so a high-dimensional matching problem collapses to matching on one number. That is the whole appeal.

**Procedure.** Fit a model (logistic regression or gradient boosting) predicting treatment from pre-treatment covariates. Then match, stratify, or weight: nearest-neighbor matching with a caliper, or inverse propensity weighting with weights $1/e(x)$ for treated and $1/(1-e(x))$ for control. Then check **covariate balance** in the matched sample — standardized mean differences below 0.1 is the usual target — and only then estimate the effect on the balanced sample.

**Limitations, and this is the substance of the answer.**

1. **It only handles observed confounders.** This is the fatal one. Matching on everything you measured does nothing about what you did not measure, and the whole reason the comparison was confounded is usually something unmeasured — motivation, need, sophistication. PSM produces a table that *looks* balanced and provides false reassurance. LaLonde's famous evaluation showed observational matching estimates diverging wildly from the experimental benchmark on the same question.
2. **Model dependence.** Results shift with the propensity model specification, the matching algorithm, the caliper width, and with/without replacement. King and Nielsen argue matching on the propensity score specifically can *increase* imbalance and model dependence relative to matching on covariates directly, and recommend coarsened exact matching instead.
3. **Positivity / overlap.** You need treated and control units at every propensity value. If some units have $e(x) \approx 0.99$, there are no comparable controls, and either you drop them (changing the estimand to a possibly uninteresting subpopulation) or IPW blows up with enormous weights and huge variance. Always plot the propensity overlap.
4. **Extreme weights.** IPW variance explodes when scores approach 0 or 1. Trimming or stabilized weights help but introduce their own choices.
5. **Inference.** Standard errors that ignore the fact that the propensity score was *estimated* are wrong.

**What I would prefer.** If randomization is available, randomize — nothing here competes. If not: **doubly robust estimators** (AIPW, TMLE) that combine an outcome model and a propensity model and are consistent if *either* is correct, which is a meaningfully weaker requirement. Or a design with an actual identification argument — DiD, IV, regression discontinuity — where the assumption is about something structural rather than "we measured everything that matters." And regardless, run a **sensitivity analysis**: Rosenbaum bounds or an E-value tell you how strong an unmeasured confounder would need to be to overturn the conclusion. If the answer is "barely stronger than the confounders we already adjusted for," the result should not drive a decision.

**Follow-up: "Your matched groups are perfectly balanced on all covariates. Is the estimate now causal?"** → No. Balance on observed covariates is necessary, not sufficient. It says nothing about unobserved confounders, and balance tables can look immaculate while the key confounder is not in the dataset at all. Perfect balance is also achievable by construction on covariates you chose, which makes it a weak signal about the ones you did not. I would state the identifying assumption explicitly — "conditional on these covariates, treatment is as good as random" — argue for it substantively, and quantify how much unobserved confounding it would take to flip the sign.

---

### Q: Explain confounder, mediator, and collider. Why is controlling for a collider harmful?

**Answer.** Three distinct structures with three different adjustment rules, and getting them backwards is the most common causal error in applied work.

**Confounder** — a common cause of both treatment and outcome. $X \leftarrow C \rightarrow Y$. It opens a backdoor path creating spurious association. **You must adjust for it.** Example: in "does exercise reduce heart disease," age causes both less exercise and more heart disease.

**Mediator** — on the causal path from treatment to outcome. $X \rightarrow M \rightarrow Y$. **Do not adjust if you want the total effect.** Adjusting gives the direct effect, blocking the very mechanism you are measuring. Example: a new onboarding flow increases retention *by* increasing first-week activity. Controlling for first-week activity makes the onboarding effect look like zero — and someone will report "onboarding doesn't matter," which is exactly backwards.

**Collider** — a common *effect* of two variables. $X \rightarrow C \leftarrow Y$. **Do not adjust.** The path through a collider is naturally *blocked*; conditioning on it *opens* the path and creates association where there was none.

**Why conditioning on a collider is harmful — the intuition.** Suppose talent and interview preparation are independent in the applicant pool, and either one can get you hired: $\text{Talent} \rightarrow \text{Hired} \leftarrow \text{Prep}$. Now look only at people who *were* hired. Someone hired with low prep must have been talented. Someone hired with low talent must have prepped hard. Within the hired group, talent and prep are negatively correlated — even though they are independent in the population. Conditioning induced the association, because knowing the effect plus one cause tells you about the other cause. This is why conditioning on a collider is sometimes called "explaining away."

Real instances that show up in ML and product work:
- **Selection bias is collider bias.** Analyzing only users who installed the app conditions on installation, which is caused by many things — inducing spurious relationships among them. Any analysis on a filtered population is conditioning on whatever caused the filter.
- **Survivorship in a churn analysis.** Restricting to users still active at 90 days conditions on a collider of everything that drives retention.
- **The obesity paradox** and similar "paradoxes" in epidemiology are often collider bias from conditioning on disease status.
- **Controlling for a post-treatment variable** is the general danger zone, because such variables are frequently colliders, mediators, or both.

**The operational rule.** Draw the DAG, then apply the **backdoor criterion**: adjust for a set that blocks all backdoor paths (paths into treatment) while opening none — meaning you do not condition on colliders or their descendants, and you do not condition on mediators if you want the total effect. Notably, the correct adjustment set is *not* "everything available," and adding a variable can strictly worsen bias.

**Follow-up: "How do you know which is which from data?"** → Generally, you cannot. Confounder, mediator, and collider structures can produce identical observational correlation patterns; the distinction is causal, not statistical, and it comes from domain knowledge, temporal ordering, and theory. Temporal order is the most useful practical guardrail: a variable measured strictly *before* treatment cannot be a mediator or a collider on the treatment-outcome path, so pre-treatment variables are the safe adjustment set. That is exactly why CUPED insists on pre-period covariates. Constraint-based discovery algorithms (PC, FCI) can recover some structure — a v-structure $X \to C \leftarrow Y$ leaves a detectable conditional-independence signature — but they rest on strong assumptions and are not a substitute for knowing the domain.

---

### Q: State the difference between correlation and causation rigorously.

**Answer.** Correlation is a property of a joint distribution: $\rho(X,Y) \neq 0$ means $E[Y \mid X]$ varies with $X$ (for the linear part). It is symmetric, it is estimable from observational data, and it says nothing about what happens if you intervene.

Causation is a statement about *interventional* distributions. In Pearl's notation, $X$ causes $Y$ if $P(Y \mid do(X = x))$ differs across $x$. The distinction is between conditioning and intervening: $P(Y \mid X = x)$ asks "among units where $X$ happens to equal $x$," while $P(Y \mid do(X=x))$ asks "if we set $X$ to $x$ for everyone." In the potential-outcomes framing, the causal effect is $E[Y(1) - Y(0)]$, where $Y(1)$ and $Y(0)$ are the outcomes under treatment and control for the *same* unit — and the fundamental problem of causal inference is that you never observe both.

**Why correlation does not imply causation** — the exhaustive list of alternatives:
1. **Reverse causation.** $Y \rightarrow X$. Users of the premium tier have higher engagement, but engagement drove the upgrade.
2. **Confounding.** $C \rightarrow X$ and $C \rightarrow Y$.
3. **Selection / collider bias.** The sample was filtered on a common effect.
4. **Chance.** With enough hypotheses, spurious correlations are guaranteed.
5. **Mediation through a shared trend.** Two time series both trending produce correlation with no relationship — the classic spurious-regression problem with non-stationary series.

**What licenses a causal claim.** $E[Y(1) - Y(0)] = E[Y \mid T=1] - E[Y \mid T=0]$ requires ignorability, $(Y(0), Y(1)) \perp T$, which randomization delivers by construction — treatment is independent of potential outcomes because it was assigned by a coin. Without randomization you need conditional ignorability given measured covariates (and positivity, and SUTVA), which is an assumption you argue for rather than one you get for free.

**The point worth making that goes beyond the slogan:** causation *does* imply correlation, under mild conditions — if $X$ causes $Y$ and nothing is masking it, you will see association. So absence of correlation is weak evidence against causation, though not conclusive, since effects can cancel across subgroups (a treatment that helps half the population and hurts the other half shows zero average correlation while causing a great deal). And correlation is genuinely useful for prediction: a model does not need causality to forecast well *within the same distribution*. It needs causality the moment you intervene, or the moment the distribution shifts. That is the crisp statement of why causal structure matters for decisions and not merely for rigor.

**Follow-up: "Give me a case where a purely predictive model fails because it's not causal."** → The classic is the hospital model predicting that asthmatic pneumonia patients have *lower* mortality risk. The correlation is real in the data — but it exists because asthmatics were triaged directly to intensive care, which lowered their mortality. A model trained on that correlation and used to *decide* who gets low-intensity care would send asthmatics home and kill them. The predictive relationship was valid only under the treatment policy that generated the data; the intervention breaks it. The general lesson: any model whose output changes the policy that produced its training data is doing causal inference whether or not anyone acknowledged it.


---

## 4. Probability

### Q: A disease affects 1 in 1,000 people. A test is 99% sensitive and 99% specific. Someone tests positive. What's the probability they have the disease?

**Answer.** About 9%. Let me work it.

Let $D$ = has disease, $+$ = tests positive.
- Prior: $P(D) = 0.001$, so $P(\neg D) = 0.999$.
- Sensitivity: $P(+ \mid D) = 0.99$.
- Specificity: $P(- \mid \neg D) = 0.99$, so the false positive rate is $P(+ \mid \neg D) = 0.01$.

Bayes:
$$P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+ \mid D)P(D) + P(+ \mid \neg D)P(\neg D)}$$
$$= \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.00099 + 0.00999} = \frac{0.00099}{0.01098} = 0.0902$$

So **9.0%**. Despite a "99% accurate" test, a positive result means a 91% chance of being disease-free.

**The natural-frequency version, which is how to explain it to a non-technical stakeholder.** Take 100,000 people. 100 have the disease, and 99 of them test positive. 99,900 do not, and 1% of them — 999 people — test positive anyway. So 1,098 positives total, of which 99 are real: $99/1098 = 9\%$. The false positives swamp the true positives because the healthy pool is a thousand times larger. Stating it in counts rather than probabilities makes the base rate impossible to ignore, and there is good evidence people reason about it far more accurately that way.

**The general lesson.** With a rare condition, the test's specificity, not its sensitivity, is what determines the positive predictive value — because the false positives are drawn from an enormous population. To get PPV above 50% here you would need the false positive rate below 0.1%, ten times better.

**Where this shows up in ML.** This *is* precision. Sensitivity = recall, and $P(D \mid +)$ = precision. It is exactly why a fraud or anomaly detector with 99% accuracy on a 0.1% base rate is useless in production, and why precision-recall curves are the right tool for imbalanced problems while ROC-AUC is misleadingly flattering — ROC uses the false positive *rate*, whose denominator is the huge negative class, so it barely moves while the alert queue fills with garbage.

**Follow-up: "They test positive twice on independent tests. Now what?"** → Update again with the posterior from round one as the new prior: $P(D) = 0.0902$. Then
$$P(D \mid ++) = \frac{0.99 \times 0.0902}{0.99 \times 0.0902 + 0.01 \times 0.9098} = \frac{0.0893}{0.0893 + 0.0091} = 0.907$$
about 91%. Two positives flip the conclusion completely. The odds form makes this cleaner: the likelihood ratio is $0.99/0.01 = 99$, so each independent positive multiplies the prior odds by 99. Prior odds $1{:}999$ → $99{:}999$ ≈ 1:10 (9%) → $9801{:}999$ ≈ 10:1 (91%). The caveat is the word *independent*: if the tests share a failure mode — the same assay, the same lab, a biological cross-reactant in that individual — the second test carries far less information than 99x, and this calculation overstates the update.

*Trap:* Answering 99%. That is confusing $P(+ \mid D)$ with $P(D \mid +)$ — the same inversion as the p-value error.

---

### Q: Define conditional probability and walk me through Monty Hall.

**Answer.** $P(A \mid B) = P(A \cap B)/P(B)$ for $P(B) > 0$ — renormalizing the probability of $A$ to the world where $B$ occurred. It gives the chain rule $P(A \cap B) = P(A \mid B)P(B)$, and $A \perp B$ iff $P(A \mid B) = P(A)$.

**Monty Hall.** Three doors, a car behind one, goats behind two. You pick door 1. The host, **who knows where the car is** and **always opens a door with a goat that you did not pick**, opens door 3. Should you switch?

Yes — switching wins with probability 2/3.

The clean argument: your initial pick is right with probability 1/3, and that does not change, because the host was always going to open a goat door regardless of your choice, so his action carries no information about *your* door. The remaining 2/3 was distributed over doors 2 and 3; the host has now concentrated all of it on door 2.

By Bayes, with $C_i$ = car behind door $i$ and $H_3$ = host opens door 3, given you picked door 1:
- $P(H_3 \mid C_1) = 1/2$ — car is behind your door, host picks freely between 2 and 3.
- $P(H_3 \mid C_2) = 1$ — host is forced, since he cannot open your door or the car's.
- $P(H_3 \mid C_3) = 0$ — he never reveals the car.

$$P(C_1 \mid H_3) = \frac{(1/2)(1/3)}{(1/2)(1/3) + (1)(1/3) + 0} = \frac{1/6}{1/2} = \frac{1}{3}$$
so $P(C_2 \mid H_3) = 2/3$. Switch.

**The part that matters, and what the question is really testing:** the answer depends entirely on the host's protocol, which is where the information comes from. If the host opens a door **at random** and it happens to reveal a goat, then $P(H_3 \mid C_1) = P(H_3 \mid C_2) = 1/2$ and the posterior is 1/2 each — switching gains nothing. Same observed event, different answer, because the data-generating process differs. That is the transferable lesson: you cannot compute a likelihood without knowing the mechanism that produced the observation, which is the same reason a stopping rule matters for p-values and why "how was this sample selected" is the first question to ask about any dataset.

**Follow-up: "Boy-girl problem — a family has two children, at least one is a girl. Probability both are girls?"** → Sample space $\{GG, GB, BG, BB\}$, equally likely. Condition on at least one girl, eliminating $BB$: three equally likely outcomes remain, one of which is $GG$, so **1/3**. But — same caveat as Monty Hall — this depends on how you learned the fact. If you met one of the children at random and she is a girl, the answer is 1/2, because that observation is twice as likely under $GG$ as under a mixed family. The phrasing "at least one is a girl" is ambiguous about the sampling mechanism, and the two readings genuinely give different answers. Naming that ambiguity is a better response than confidently asserting either number.

---

### Q: Give me the expectation and variance rules you use most.

**Answer.** **Expectation is linear, unconditionally:**
$$E[aX + bY + c] = aE[X] + bE[Y] + c$$
No independence required — this is the workhorse, and it is why you can compute the expected number of matches in complicated combinatorial problems by summing indicator expectations even when the indicators are dependent.

**Variance is not linear:**
$$\text{Var}(aX + b) = a^2\text{Var}(X)$$
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$
Independence kills the covariance term. Note $\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X,Y)$ — variances *add* even for a difference when independent, which is why the two-sample SE has that factor of 2.

**Computational form:** $\text{Var}(X) = E[X^2] - (E[X])^2$.

**Products:** $E[XY] = E[X]E[Y]$ only under independence (or zero covariance). $E[g(X)] \neq g(E[X])$ in general — Jensen's inequality says $E[g(X)] \geq g(E[X])$ for convex $g$. This is a live issue: $E[1/X] \neq 1/E[X]$, so an average of per-user ratios is not the ratio of averages, and exponentiating a mean of logs gives a geometric mean, not the arithmetic one.

**Law of total expectation (tower rule):** $E[X] = E[E[X \mid Y]]$. Enormously useful for decomposing over cases.

**Law of total variance:**
$$\text{Var}(X) = E[\text{Var}(X \mid Y)] + \text{Var}(E[X \mid Y])$$
"within-group variance plus between-group variance." This is the ANOVA decomposition, and it is also the bias-variance intuition for why stratification reduces variance: stratifying removes the between-group term from your estimator's error.

**Sample mean:** for i.i.d. $X_i$, $E[\bar{X}] = \mu$ and $\text{Var}(\bar{X}) = \sigma^2/n$, giving $SE = \sigma/\sqrt{n}$ — the equation behind every sample size calculation in this document.

**Follow-up: "How do you get the variance of a ratio metric like clicks-per-impression when you randomize by user?"** → Delta method. For $R = \bar{X}/\bar{Y}$ with user-level sums $X_i$ (clicks) and $Y_i$ (impressions), a first-order Taylor expansion around $(\mu_X, \mu_Y)$ gives
$$\text{Var}(R) \approx \frac{1}{n\mu_Y^2}\left(\sigma_X^2 - 2\frac{\mu_X}{\mu_Y}\sigma_{XY} + \frac{\mu_X^2}{\mu_Y^2}\sigma_Y^2\right)$$
The point is that $n$ here is the number of *users*, not impressions, and the covariance term matters because heavy users contribute both a lot of clicks and a lot of impressions. Treating each impression as an independent observation ignores the clustering entirely and can understate the standard error by an order of magnitude — this is the single most common variance bug in ratio-metric A/B tests. The alternatives are a bootstrap resampling users, or cluster-robust standard errors clustered on user.

---

### Q: Name the common distributions and when each arises.

**Answer.** I think of these in terms of the generative mechanism that produces them.

**Bernoulli($p$)** — a single binary trial. Mean $p$, variance $p(1-p)$, maximized at $p = 0.5$. Every conversion metric.

**Binomial($n,p$)** — count of successes in $n$ independent identical trials. Mean $np$, variance $np(1-p)$.

**Poisson($\lambda$)** — count of events in a fixed interval when events are independent and the rate is constant. It is the limit of Binomial as $n \to \infty$, $p \to 0$ with $np = \lambda$ fixed, which is why it describes rare events in large populations: server errors per minute, arrivals per hour. Signature property: mean = variance = $\lambda$. When real count data has variance far exceeding the mean — which is typical, because rates vary across users — you have overdispersion and should use negative binomial instead.

**Geometric($p$)** — trials until first success. Memoryless in discrete time.

**Negative binomial** — trials until $r$-th success, but in practice used as a Poisson with a gamma-distributed rate, i.e. the overdispersed count model. Standard for user event counts.

**Exponential($\lambda$)** — waiting time between Poisson events. The unique continuous memoryless distribution: $P(T > s+t \mid T > s) = P(T > t)$. Good for time-to-event when the hazard is constant, which is often *not* true for churn (hazard usually decreases with tenure), so Weibull is the common generalization.

**Normal($\mu,\sigma^2$)** — sums and averages of many small independent effects, via CLT. Maximum entropy given a fixed mean and variance, which is the principled reason to default to it when you know only those two moments.

**Log-normal** — the *product* of many independent positive effects, since the log is a sum. Income, session duration, file sizes, latency. Right-skewed with a long tail. If your metric is log-normal, the sample mean converges slowly and the CLT caveats from earlier apply.

**Pareto / power law** — scale-free phenomena with much heavier tails: wealth, city sizes, degree distributions in social graphs. With tail index $\alpha \le 2$ the variance is infinite and standard inference breaks.

**Beta($\alpha,\beta$)** — a distribution over probabilities on $[0,1]$; the conjugate prior for Bernoulli/Binomial, which makes it the backbone of Bayesian A/B testing and Thompson sampling. The posterior after $s$ successes and $f$ failures is just Beta($\alpha+s$, $\beta+f$).

**Gamma** — sum of exponentials; conjugate prior for the Poisson rate; models positive continuous quantities with skew.

**Uniform** — maximum entropy on a bounded interval; also the distribution of a p-value under the null, and the basis of inverse-transform sampling.

**Follow-up: "Your latency data is right-skewed. Which distribution and which summary statistic?"** → Log-normal is the usual first fit for latency, though real service latency is often a mixture — a fast path plus a slow path (cache miss, retry, cold start) — so a mixture or a heavy-tailed fit may be better than any single log-normal. For the summary, I would not use the mean, and I would not use the median either as the sole number: report p50, p95, and p99. The mean is dominated by the tail and is not what users experience; the median hides the tail entirely; and the tail is exactly where the user pain and the SLO live. For inference on percentiles specifically, note that the standard errors are different from mean-based ones — bootstrap them, since the delta method for quantiles depends on the density at the quantile and is unreliable in the tail.

---

### Q: MLE versus MAP — derive one and explain the relationship.

**Answer.** **MLE** maximizes the likelihood of the data:
$$\hat{\theta}_{MLE} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$$

**MAP** maximizes the posterior:
$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \left[\log P(D\mid\theta) + \log P(\theta)\right]$$
since the evidence $P(D)$ does not depend on $\theta$. So **MAP is MLE plus a log-prior term**, and with a flat prior they coincide.

**Derivation — Bernoulli MLE.** Observe $n$ trials with $k$ successes.
$$L(p) = p^k(1-p)^{n-k}, \qquad \ell(p) = k\log p + (n-k)\log(1-p)$$
$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \implies k(1-p) = (n-k)p \implies \hat{p} = \frac{k}{n}$$
The sample proportion, as expected. Second derivative is negative, so it is a maximum.

**Now MAP with a Beta($\alpha,\beta$) prior.** The log-posterior adds $(\alpha-1)\log p + (\beta-1)\log(1-p)$, and the same algebra gives
$$\hat{p}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$
With Beta(2,2) this is $(k+1)/(n+2)$ — Laplace smoothing, derived rather than asserted. The practical payoff: with $k=0$ successes in $n=5$ trials, MLE says $\hat p = 0$, which is a confident claim that the event is impossible from five observations. MAP says $1/7 = 0.14$. The prior is regularizing against overconfidence at small $n$, and that is precisely the small-sample regime where MLE is worst.

**The general correspondence.** MAP with a Gaussian prior on regression coefficients is ridge, with $\lambda = \sigma^2/\tau^2$; MAP with a Laplace prior is lasso. So regularization *is* a prior, and the regularization strength is the inverse prior variance. That equivalence is a strong thing to state, because it links this section to the regression section.

**Properties.** MLE is consistent, asymptotically normal, and asymptotically efficient (attains the Cramér-Rao bound) under regularity conditions — but it can be biased in finite samples, the classic example being the MLE of $\sigma^2$ dividing by $n$ instead of $n-1$. Both MLE and MAP are point estimates that discard the rest of the posterior; full Bayesian inference keeps the whole distribution and integrates over it. MAP is also not parameterization-invariant — reparameterize and the mode moves, since the density picks up a Jacobian — while MLE is invariant under reparameterization. That is a genuine conceptual wart in MAP worth knowing.

**Follow-up: "When does the prior stop mattering?"** → As $n$ grows, the log-likelihood scales with $n$ while the log-prior stays fixed, so the likelihood dominates and MAP → MLE. The Bernstein-von Mises theorem formalizes it: under regularity conditions the posterior converges to a normal centered at the MLE with the inverse Fisher information as variance, regardless of the prior. The exceptions matter, though: if the prior assigns zero density to the true value, no amount of data recovers it (Cromwell's rule); and in high dimensions where $p$ grows with $n$, or in hierarchical models, the prior can continue to matter indefinitely because you never accumulate much data per parameter.

---

### Q: The birthday problem — how many people before there's a 50% chance two share a birthday?

**Answer.** 23.

Compute the complement, since "no shared birthday" is a clean product:
$$P(\text{all distinct}) = \frac{365}{365}\cdot\frac{364}{365}\cdots\frac{365-n+1}{365} = \prod_{k=0}^{n-1}\left(1 - \frac{k}{365}\right)$$
At $n = 23$ this is 0.4927, so $P(\text{at least one match}) = 0.5073$.

**The approximation, which is the part worth knowing.** Using $1-x \approx e^{-x}$:
$$P(\text{all distinct}) \approx \exp\left(-\sum_{k=0}^{n-1}\frac{k}{365}\right) = \exp\left(-\frac{n(n-1)}{730}\right)$$
Setting that to 1/2 gives $n(n-1) \approx 730\ln 2 = 506$, so $n \approx 23$. In general, for $d$ equally likely categories, you need about $1.177\sqrt{d}$ draws for a 50% collision chance — the $\sqrt{d}$ scaling is the whole insight.

**Why the intuition fails.** People compare $n$ to 365, but the relevant quantity is the number of *pairs*, $\binom{23}{2} = 253$, which is comparable to 365. Every pair is a chance at a collision, and pairs grow quadratically.

Other landmarks: $n = 57$ gives 99%; $n = 70$ gives 99.9%; and 100% requires 366 by pigeonhole.

**Why interviewers ask this.** Because the $\sqrt{d}$ scaling has direct engineering consequences. Hash collisions: with a 32-bit hash you expect collisions after about $\sqrt{2^{32}} = 65{,}536$ items, not 4 billion — which is why 32-bit IDs fail far earlier than people expect. The birthday attack on cryptographic hashes means an $n$-bit hash gives only $n/2$ bits of collision resistance, which is why MD5's 128 bits provide only 64 bits of collision security and are broken in practice. In experimentation specifically: with many experiments running off hashed user IDs, hash collisions between experiment salts are a real source of contamination, and the collision math is this exact calculation.

**Follow-up: "What if birthdays aren't uniform?"** → Any non-uniformity *increases* the collision probability. Intuitively, clustering concentrates mass and makes matches more likely; formally, the probability of all-distinct is minimized... rather, maximized under the uniform distribution, so uniform is the best case for avoiding collisions. Real birthday data is mildly non-uniform (September peaks, fewer on Feb 29 and around certain holidays), which lowers the 50% threshold slightly — but only slightly, still 23. For hash functions this is exactly why uniformity of the hash matters: a biased hash collides more than the birthday bound predicts, which is why you use a well-tested hash for experiment assignment rather than something ad hoc like taking user_id modulo 100, which correlates with however IDs were issued.

---

### Q: Coupon collector — you're collecting $n$ distinct items, one random item per purchase. How many purchases to get them all?

**Answer.** $n H_n \approx n\ln n + \gamma n + 1/2$, where $H_n$ is the $n$-th harmonic number and $\gamma \approx 0.5772$.

**Derivation, and it is a clean application of linearity of expectation.** Break the process into phases: phase $i$ is the stretch where you already have $i-1$ distinct items and are waiting for the $i$-th new one. In that phase, the probability any given purchase is new is $p_i = (n-i+1)/n$. Waiting time is geometric, so $E[T_i] = 1/p_i = n/(n-i+1)$.

By linearity — which applies even though the $T_i$ are the pieces of one dependent process:
$$E[T] = \sum_{i=1}^n \frac{n}{n-i+1} = n\sum_{k=1}^{n}\frac{1}{k} = nH_n$$

**Concrete:** $n = 50$ gives $50 \times H_{50} = 50 \times 4.499 = 225$ purchases for 50 items. Note the asymmetry that makes the problem feel unfair: the first 25 items take about $50(H_{50}-H_{25}) = 50 \times 0.69 = 35$ purchases, while the *last* item alone takes $n = 50$ purchases in expectation. Half your effort goes to the last handful of items — which is exactly the business model of collectible packs.

**Variance** is $\approx n^2\pi^2/6 \approx 1.645n^2$, so the standard deviation is about $1.28n$ — substantial relative to the mean $n\ln n$. And there is a sharp concentration result: $P(T > n\ln n + cn) \to 1 - e^{-e^{-c}}$, a Gumbel limit.

**Where it comes up in practice.** Cache warming and how long until all keys are touched; test coverage under random inputs; how long a crawler needs to visit every page under uniform sampling; and the general shape of "the long tail costs most of the budget," which is the transferable intuition — the tail of rare items dominates the completion time, and any system that must cover *all* categories pays $\ln n$ times the naive estimate.

**Follow-up: "What if items have unequal probabilities?"** → It gets substantially worse, and the expectation is dominated by the rarest item. There is no simple closed form, but by inclusion-exclusion $E[T] = \int_0^\infty \left(1 - \prod_i (1-e^{-p_i t})\right)dt$. The useful bound: the expected time is at least $1/p_{\min}$, the wait for the rarest item alone, and typically close to $(1/p_{\min})\ln(\text{something})$. Practically: if one item has probability 0.001 and the rest are common, you need on the order of 1,000 draws for that one item regardless of how easy the others are. This is exactly the dynamic behind rare-class coverage in data collection — uniform sampling to cover a long-tail class distribution is wildly inefficient, which is the argument for stratified or active sampling.

---

### Q: Joint, marginal, and conditional distributions — define them and explain how they relate.

**Answer.** **Joint** $P(X,Y)$ is the full specification of how the two vary together — everything there is to know about the pair.

**Marginal** $P(X)$ is the distribution of $X$ alone, obtained by summing or integrating out the other variable:
$$P(X=x) = \sum_y P(X=x, Y=y) \qquad \text{or} \qquad p(x) = \int p(x,y)\,dy$$
The name comes from writing row and column sums in the margins of a contingency table.

**Conditional** $P(Y \mid X)$ is the distribution of $Y$ restricted to a slice where $X$ takes a given value, renormalized:
$$P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)}$$

They relate by the **chain rule** $P(X,Y) = P(Y\mid X)P(X)$, which extends to $P(X_1,\ldots,X_n) = \prod_i P(X_i \mid X_1,\ldots,X_{i-1})$ — the factorization that underlies autoregressive language models, where each token is conditioned on the prefix.

**The direction of information matters.** From the joint you can always recover marginals and conditionals. From the marginals alone you *cannot* recover the joint — you need the dependence structure. Two joints with identical marginals can be completely different: $X, Y$ independent uniforms versus $X$ uniform with $Y = X$ have the same marginals and utterly different joints. Copulas are the formal machinery for separating marginals from dependence, and mis-modeling exactly this — assuming a Gaussian copula, so underestimating joint tail events — is the standard technical account of the 2008 CDO mispricing.

**Where it bites in practice.** Marginalizing hides the structure that matters: the marginal conversion rate can be flat while conditional rates move in opposite directions across segments — that is Simpson's paradox stated in this vocabulary. In ML, generative models learn joints and discriminative models learn conditionals $P(Y\mid X)$, which is why a discriminative classifier cannot tell you whether an input is out-of-distribution: it never modeled $P(X)$.

**Follow-up: "Given $P(Y\mid X)$ and $P(X)$, can you get $P(X\mid Y)$?"** → Yes, by Bayes: $P(X\mid Y) = P(Y\mid X)P(X)/P(Y)$, with $P(Y) = \sum_x P(Y\mid X=x)P(x)$ recovered by marginalizing the joint you just built. So $P(Y\mid X)$ plus the marginal $P(X)$ is enough to reconstruct the entire joint and therefore everything. What you cannot do is invert $P(Y\mid X)$ without $P(X)$ — that is precisely the base-rate neglect error from the medical-test problem, where knowing sensitivity tells you nothing about the posterior until you supply the prevalence.

---

### Q: Independence versus conditional independence — can you have one without the other?

**Answer.** Yes, in both directions, and that is the whole point of the question.

**Independence:** $P(X,Y) = P(X)P(Y)$.
**Conditional independence given $Z$:** $P(X,Y\mid Z) = P(X\mid Z)P(Y\mid Z)$, written $X \perp Y \mid Z$.

Neither implies the other.

**Independent but conditionally dependent.** This is the collider structure $X \to Z \leftarrow Y$. Two fair coin flips $X, Y$ are independent; let $Z = X \oplus Y$ (their XOR). Given $Z = 1$, knowing $X$ determines $Y$ exactly — perfect dependence. Conditioning on a common effect created dependence from nothing. Same structure as the talent/prep/hired example: independent causes become dependent within the selected group.

**Dependent but conditionally independent.** This is the confounder structure $X \leftarrow Z \to Y$, or the chain $X \to Z \to Y$. Ice cream sales and drowning deaths are strongly correlated; conditional on temperature, they are independent. The dependence was entirely mediated by the common cause, and conditioning on it blocks the path. This is the entire premise of adjusting for confounders — and of Naive Bayes, which *assumes* features are conditionally independent given the class even though they are marginally dependent.

**Why it matters.**
- **Causal inference.** The three DAG structures — chain, fork, collider — are distinguished exactly by their conditional independence signatures, and d-separation is the general rule that reads independencies off a graph. That is what lets constraint-based discovery algorithms learn structure from data.
- **Graphical models.** A Bayesian network's whole compression advantage is the conditional independencies it encodes; without them the joint is exponential in the number of variables.
- **Feature selection.** A feature can look useless marginally and be highly informative conditional on others (XOR is the canonical case where univariate screening fails), and conversely a feature can look predictive marginally and add nothing once correlated features are included.

**Follow-up: "Where does this bite in an A/B test?"** → Two places. First, users are not independent — they share sessions, devices, households, and social ties — but they are often approximately independent *conditional on* the cluster, which is exactly what justifies cluster-robust standard errors: model the dependence at the cluster level and treat clusters as independent. Second, in heterogeneous treatment effect analysis: treatment assignment is independent of every covariate by randomization, but if you condition on a post-treatment variable, you can induce dependence between treatment and covariates that were balanced by design — you break the very randomization you paid for. That is the collider example again, appearing in a place where people do not expect it because "we randomized" feels like it should be protective.

---

### Q: Covariance versus correlation, and what does correlation miss?

**Answer.** Covariance measures joint linear variation: $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY] - E[X]E[Y]$. Its problem is units — it is in units of $X$ times units of $Y$, so its magnitude is uninterpretable and it is not comparable across variable pairs.

Pearson correlation normalizes it:
$$\rho = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1,1]$$
Unitless and scale-invariant, so $\rho$ is unchanged by any positive linear rescaling. $\rho^2$ is the fraction of variance in one variable linearly explained by the other — the same $R^2$ as in simple regression.

**What correlation misses:**

1. **Nonlinearity.** This is the big one. Let $X \sim \text{Uniform}(-1,1)$ and $Y = X^2$. $Y$ is a deterministic function of $X$ — total dependence — and $\text{Cov}(X,Y) = E[X^3] - E[X]E[X^2] = 0$. Correlation is exactly zero. So $\rho = 0$ does **not** mean independence. The converse does hold: independence implies $\rho = 0$. (The one important exception where they coincide: for a *jointly* Gaussian pair, zero correlation does imply independence — but "each marginally Gaussian" is not enough, it must be joint.)
2. **Non-monotone and threshold relationships,** for the same reason.
3. **Outlier sensitivity.** A single extreme point can create or destroy a correlation. Anscombe's quartet: four datasets, all with $\rho = 0.816$, one of which is a perfect parabola and one of which is a vertical line plus one leverage point.
4. **Tail dependence.** Two variables can have modest overall correlation but move together almost perfectly in extreme events. This is exactly the risk-model failure mode: assets uncorrelated in normal markets crash together.
5. **Heterogeneity across subgroups.** Correlation can be positive in every subgroup and negative overall — Simpson's paradox again.
6. **Causation and direction.** $\rho$ is symmetric and says nothing about which way anything runs.
7. **Restriction of range.** Correlation is attenuated by conditioning on a narrow slice of $X$. Test scores may correlate poorly with performance *among admitted students* while correlating well in the full applicant pool — this is why selection on a variable deflates its measured predictive value, and it is a real problem in hiring analytics.

**Alternatives when you suspect these.** Spearman (Pearson on ranks) catches any monotone relationship and is robust to outliers. Kendall's tau, similar and better for small samples with ties. Mutual information, distance correlation, or HSIC capture arbitrary dependence and are zero if and only if independent — at the cost of being harder to estimate and lacking a sign. And before any of them: **plot the data**. Every failure above is obvious in a scatterplot and invisible in a single number.

**Follow-up: "Two model errors have correlation 0.9. What does that mean for ensembling?"** → It means ensembling buys you almost nothing. For two errors with equal variance $\sigma^2$, the average has variance $\frac{\sigma^2}{2}(1+\rho)$. At $\rho = 0$ that is a 50% reduction; at $\rho = 0.9$ it is $0.95\sigma^2$, a 5% reduction. This is why ensemble diversity is the entire game — bagging, feature subsampling, different architectures, different seeds all exist to push $\rho$ down. Averaging $k$ models gives $\sigma^2\left(\frac{1-\rho}{k} + \rho\right)$, so as $k \to \infty$ the variance floors at $\rho\sigma^2$: correlated error is irreducible no matter how many models you add.

---

## 5. Applied Traps

### Q: What is survivorship bias? Give an example from ML work.

**Answer.** Survivorship bias is drawing conclusions from a sample that has been filtered by an outcome-related process, while treating it as representative. It is a form of selection bias, and structurally it is collider bias: you conditioned on survival, which is a common effect of the very things you are studying.

The canonical case is Abraham Wald and the WWII bombers. Analysts examined returning aircraft, found bullet holes concentrated on wings and fuselage, and proposed armoring those. Wald pointed out the sample was planes that *made it back*. The absence of damage to engines meant planes hit in the engines did not return — so armor the engines. The data's holes were literally where the information was missing.

**Examples from ML and product work:**

- **Model evaluation on retained users.** Evaluate a recommender only on users still active at 90 days, and you have conditioned on retention, which the recommender itself influences. The model looks great on the population it did not drive away.
- **Training on accepted-only outcomes.** A credit or fraud model trained only on approved applications never observes outcomes for rejections, so the training distribution is the previous policy's accept region. Retraining on it entrenches the old policy and cannot discover it was wrong — the reject-inference problem, and the general case of feedback loops in deployed decision systems.
- **Published benchmark results.** Reported numbers survive a filter of "worth publishing," so the visible distribution of effect sizes is truncated from below. Architecture comparisons across papers are contaminated by this even before you get to unequal tuning effort.
- **Hyperparameter and architecture search.** Reporting the best of 200 runs on the validation set is survivorship on the search process. The best run's validation score is upward-biased by the maximum of a noise distribution, which is why the held-out test number is typically worse.
- **Cohort analyses.** "Users who used feature X retained better" — X-users are the ones who stuck around long enough to find X.
- **Startup and career advice generally.** Studying successful founders' habits without the failed founders who had the same habits.

**How to protect against it.** Ask, always, "what would be missing from this dataset, and would its absence correlate with the outcome?" Define the population at a point *before* the filter — analyze the full randomized cohort including churners (intent-to-treat), not the retained subset. Instrument the rejected/dropped cases where possible: hold out a small random fraction from the filter so you observe outcomes in the region your policy would exclude. That deliberate exploration is expensive but it is the only real fix for the accepted-only problem.

**Follow-up: "How does this relate to intent-to-treat analysis?"** → ITT is the direct remedy. It analyzes every unit in the arm it was randomized to, regardless of whether they complied, engaged, or churned. Per-protocol analysis — restricting to users who actually used the feature — conditions on a post-treatment behavior and reintroduces exactly the selection the randomization removed, because who engages is not random and is often caused by the treatment. ITT can understate the effect on users who actually get the treatment, which is a real limitation; the principled way to recover that is IV/CACE using assignment as an instrument, not by filtering the sample.

---

### Q: What is selection bias, and how is it different from confounding?

**Answer.** **Confounding** comes from a common *cause* of treatment and outcome: $T \leftarrow C \rightarrow Y$. It is present in the full population and adjusting for $C$ fixes it.

**Selection bias** comes from the sample being chosen in a way that depends on treatment and outcome (or their causes). Structurally it is conditioning on a *collider*: $T \rightarrow S \leftarrow Y$, where $S$ is inclusion in the sample. Adjusting for more covariates does not generally fix it, and can make it worse, because the problem is in who you are looking at, not in what you controlled for.

That difference in remedy is the practical distinction: confounding is a *variables* problem, selection is a *sample* problem. Randomization eliminates confounding entirely but does **not** eliminate selection bias — differential attrition after randomization reintroduces it, which is why an A/B test can still produce a biased estimate.

**Flavors that show up:**
- **Differential attrition.** Treatment makes low-engagement users churn faster, so at week 4 the treatment arm is a healthier subpopulation and every metric looks better. This is one of the most dangerous A/B testing failures because randomization creates a false sense of safety.
- **Nonresponse bias.** Survey respondents differ systematically from non-respondents — you measure the opinions of people who answer surveys.
- **Ascertainment / detection bias.** The treatment changes the probability an outcome is *observed*, not the outcome itself. A feature that makes reporting a bug easier increases measured bug rate with no change in bugs.
- **Undercoverage.** Sampling frame excludes part of the population — a mobile-app survey missing desktop-only users.
- **Berkson's bias.** Studying only hospitalized patients induces spurious negative associations between diseases.
- **Label availability.** In ML, labels available only for a non-random subset — feedback loops, delayed labels, only-inspected-items-are-labeled.

**Diagnosis and mitigation.** Compare the sample to known population characteristics. Check attrition rates and pre-treatment characteristics of the *retained* users by arm — if the arms differ on pre-treatment covariates among survivors while they matched at assignment, you have differential attrition. Then: analyze ITT on the full randomized set; bound the effect with worst-case imputation for missing outcomes (Manski or Lee bounds); or reweight by inverse probability of selection if you can model it and you have the covariates that drive selection. Be honest that reweighting requires the selection to be explainable by observed variables, which is the same untestable assumption as in propensity matching.

**Follow-up: "Your A/B test has 5% attrition in control and 8% in treatment. Is the result valid?"** → Not without work. Differential attrition means the two arms' remaining populations are no longer comparable and the randomization guarantee is broken for any analysis on survivors. First, is the attrition itself the finding? An extra 3 points of churn may be the most important result of the test, and I would treat it as a guardrail failure. Second, for the primary metric, I would run ITT with the missing outcomes handled explicitly — imputed at a defined value if the metric is naturally zero for churners (sessions, revenue) — and then bound: assign the missing treatment users the worst plausible outcomes and the missing control users the best, and see whether the conclusion survives. If it does, ship. If the conclusion flips inside the bounds, the test cannot answer the question and I would say so.

---

### Q: What is regression to the mean, and how does it fool people?

**Answer.** Regression to the mean is the tendency of extreme observations to be followed by less extreme ones, purely because extremeness is partly noise and noise does not repeat. If $X = \mu + \text{signal} + \text{noise}$, then selecting on a high $X$ selects partly on a high noise draw, and the next observation gets a fresh noise draw with mean zero. Formally, for a bivariate pair with correlation $\rho$ in standardized units, $E[Y \mid X = x] = \rho x$ — so any $|\rho| < 1$ pulls the prediction toward the mean, and the pull is stronger the noisier the measure.

**The classic fool.** Galton's flight instructors: pilots praised after an exceptionally good landing did worse next time, pilots criticized after a bad landing improved. The instructors concluded that criticism works and praise backfires. In fact both groups were regressing to their own mean, and the intervention did nothing. Kahneman uses this as the example of how the world is arranged to teach us that punishment works and reward does not — a lesson that is entirely a statistical artifact.

**Where it bites in applied ML and product work:**

- **Any intervention targeted at the worst performers.** Ship a fix to the 100 slowest queries, the 50 lowest-engagement users, the worst-performing stores — they will improve on their own, and the naive pre/post comparison credits the intervention with the entire regression. This is *extremely* common in "we fixed the problem accounts" analyses.
- **Post-hoc "we fixed it" claims after an anomaly.** Metrics spike, a fix ships, metrics normalize. The spike was going to normalize anyway.
- **Segment analysis in A/B tests.** Find the segment with the biggest effect, re-measure it next quarter, and the effect shrinks. The winner's curse is regression to the mean applied to selected estimates.
- **Model comparison on a leaderboard.** The top model is partly lucky on the test set; its next evaluation is typically worse.
- **Cherry-picked model checkpoints.** Selecting the best-validation checkpoint and reporting that validation score.

**The fix is always the same: a control group.** With a randomized control, both arms regress equally and the difference is unbiased. Without one, alternatives are: use a longer baseline period so the pre-measurement is less noisy; use regression discontinuity if selection is on a sharp threshold, which handles it explicitly; or explicitly model the shrinkage — empirical Bayes / James-Stein estimation shrinks each unit's estimate toward the grand mean by an amount determined by its noise, which is the principled version of "don't believe extreme small-sample estimates."

**Follow-up: "We targeted our 10,000 lowest-engagement users with a re-engagement campaign, and their engagement rose 40%. Did it work?"** → I cannot tell from that. Those users were selected for being at the bottom, so a large chunk of that 40% is regression to the mean — many of them were temporarily low, not persistently low. To measure the campaign, I would want a randomized holdout: take the 10,000 lowest, randomize 20% to receive nothing, and compare. Both groups regress identically, so the difference is the campaign effect. If no holdout exists, I would find a comparison group — users just above the selection threshold, which enables a regression discontinuity — and I would look at the historical base rate: how much did similar bottom-decile cohorts recover in past quarters with no campaign? If the historical bounce-back is 35%, the campaign is worth about 5 points, not 40.

---

### Q: What is base rate neglect and where does it show up in ML systems?

**Answer.** Base rate neglect is ignoring the prior probability of a class when interpreting evidence, focusing on how diagnostic the evidence *seems* while forgetting how rare the thing is. It is the cognitive version of the medical-test error: conflating $P(\text{evidence}\mid\text{class})$ with $P(\text{class}\mid\text{evidence})$.

**Where it shows up in ML systems:**

1. **Rare-event classifiers in production.** A fraud model with 95% recall and a 1% false positive rate, on a 0.1% fraud base rate: per 1 million transactions, 1,000 fraudulent (950 caught) and 999,000 legitimate (9,990 flagged). Precision is $950/10{,}940 = 8.7\%$ — 11 false alarms per catch. The model is genuinely good and the alert queue is unusable. Any conversation about deploying a detector that does not start with the base rate is going to end badly.
2. **Accuracy as a metric.** On a 0.1% positive rate, predicting "negative" always gives 99.9% accuracy. Accuracy is nearly uninformative under imbalance, which is why you need precision/recall, PR-AUC, or a cost-weighted metric.
3. **ROC-AUC being misleading under imbalance.** FPR's denominator is the huge negative class, so a large absolute number of false positives barely moves the ROC curve. PR curves expose it because precision's denominator is the flagged set.
4. **Threshold selection.** The optimal threshold depends on the prior and the cost ratio, and both shift over time. A model calibrated at a 2% base rate is miscalibrated when the rate drops to 0.5%, and its output probabilities need prior correction.
5. **Anomaly detection generally.** Anomalies are rare by definition, so alert fatigue is the default outcome and the design question is precision-at-a-fixed-alert-budget, not detection rate.
6. **Interpreting A/B test wins.** If only 15% of experiment ideas work, a significant result at 80% power has a false discovery rate around 27%, not 5% — the same base-rate calculation applied to the hypothesis pool.

**How to counter it.** Always state the base rate first, then compute the confusion matrix in absolute counts for realistic volume — counts make the problem visible in a way rates do not. Design around a fixed operational capacity: "we can review 500 cases a day, so what is precision at the top 500?" And calibrate probabilities and check calibration curves rather than only ranking metrics, because a well-calibrated 3% probability communicates the base rate honestly while a raw score does not.

**Follow-up: "Your model's precision is too low. What are your options?"** → Raise the threshold, trading recall for precision along the PR curve — usually the first move and often sufficient. Reframe as ranking with a fixed review budget, where precision@k is the operational metric. Add a second-stage model on the flagged set, which is cheap because the candidate volume is small. Change the base rate by segmenting: apply the model only to a subpopulation with higher prior risk, which mechanically improves precision. Add features that increase the likelihood ratio, since PPV depends on the LR times the prior odds. Or accept lower precision and cut the cost per false positive — a soft intervention like a step-up verification rather than a hard block, so a false positive costs a few seconds rather than a lost customer.

---

### Q: What is p-hacking? Name the researcher degrees of freedom.

**Answer.** P-hacking is exploiting analytic flexibility — consciously or not — to push a result under the significance threshold. It does not require dishonesty; it usually looks like a sequence of individually reasonable choices, each nudged by whether the result improves.

**The researcher degrees of freedom:**

1. **Optional stopping.** Collect data, test, collect more if not significant. Documented earlier: 5 looks turns 5% into 14%.
2. **Outlier rules chosen after the fact.** Deciding to exclude "implausible" values after seeing which exclusion helps.
3. **Metric selection.** Measuring 15 outcomes and reporting the one that moved.
4. **Subgroup mining.** Testing 20 segments and reporting the significant one. With 20 segments, expect one at $p<0.05$ under a pure null.
5. **Covariate choice.** Trying models with and without various controls and keeping the specification that produces significance — the "garden of forking paths."
6. **Transformation choice.** Raw versus log versus rank versus winsorized, picked by result.
7. **Test choice.** t-test versus Mann-Whitney versus permutation, whichever crosses.
8. **One-tailed switch.** Converting a $p=0.07$ two-tailed into $p=0.035$ one-tailed after the fact.
9. **Exclusion of "failed" conditions or time periods.** Dropping the week the result was bad because "something else was going on."
10. **HARKing** — Hypothesizing After Results are Known: writing up an exploratory finding as if it had been predicted.

The magnitude is well established: Simmons, Nelson, and Simonsohn's "False-Positive Psychology" showed that a modest combination of these flexibilities pushes the false positive rate above 60%, and they demonstrated it by "proving" that listening to a song made people younger.

**Detection.** A p-curve — the distribution of significant p-values across studies — should be right-skewed (many very small p-values) if effects are real, and flat if the null is true. A *left*-skewed p-curve with a lump just below 0.05 is the signature of p-hacking. In a single organization, an audit of experiment results showing an implausible pile-up just under 0.05 is the same evidence.

**Prevention.** Pre-registration of the primary metric, sample size, exclusions, and analysis; a fixed analysis pipeline that runs automatically so the analyst has no knobs; separating exploratory from confirmatory work explicitly and requiring a fresh confirmatory test for anything found by exploration; multiple-comparison corrections; and holdout data that is opened once. Culturally: making null results publishable internally, because most p-hacking is driven by the perception that a null result means wasted quarter.

**Follow-up: "Is exploratory analysis wrong, then?"** → Not at all — it is essential, and most good hypotheses come from it. The wrongdoing is *labeling*: presenting an exploratory finding with the inferential guarantees of a confirmatory test. The correct workflow is to explore freely, generate hypotheses, then validate on new data with a pre-registered plan. What is not acceptable is exploring and then reporting a p-value as if the hypothesis had been specified in advance, because that p-value has no defined error rate. I would report exploratory findings with effect sizes and intervals and label them explicitly as hypothesis-generating.

---

### Q: What is pre-registration and what should it contain?

**Answer.** Pre-registration is committing to the analysis plan, in a timestamped record, before the data are observed. Its function is to remove the researcher degrees of freedom by fixing every choice in advance, which is what restores the p-value's meaning — the error rate is only defined relative to a specified procedure.

**Contents, for an experiment:**
1. **Hypothesis**, stated directionally and specifically enough to be wrong.
2. **Primary metric** — exactly one — with its precise operational definition: the aggregation, the window, the denominator, the unit.
3. **Secondary and guardrail metrics**, listed exhaustively, so nothing can be added later.
4. **Randomization unit and mechanism**, plus the traffic allocation.
5. **Sample size and duration**, with the power calculation, the assumed $\sigma$, and the MDE — and the commitment to run to that horizon.
6. **Stopping rule.** Fixed horizon, or the specific sequential boundary. If interim looks are allowed, name the schedule and the alpha-spending function.
7. **Exclusion criteria** — bots, internal accounts, specific geos — defined by rules, not by inspection.
8. **Outlier and winsorization rules**, including the exact percentile.
9. **Analysis method** — the test, the variance estimator, the clustering level, whether CUPED is applied and with which covariate.
10. **Multiple comparison correction** and which family it applies to.
11. **Decision rule.** What result leads to shipping, iterating, or killing. This is the one most often skipped and it is arguably the most valuable, because it prevents the post-hoc renegotiation of what counts as success.
12. **Planned subgroup analyses**, labeled confirmatory or exploratory.

**The evidence that it works.** In clinical trials, after the FDA required pre-registration of cardiovascular trials in 2000, the share of large NHLBI trials reporting a significant benefit dropped from 57% to 8%. That is a striking natural experiment on the size of the analytic-flexibility effect, and it is a good number to have on hand.

**The practical form in industry.** A short experiment design doc reviewed before launch, ideally enforced by the platform: the experimentation tool stores the primary metric and the planned duration at creation time and displays them at analysis time, so any deviation is visible rather than silent. The enforcement matters more than the document — a doc nobody diffs against the final analysis provides no protection.

**Follow-up: "What if you need to deviate from the plan?"** → Deviating is allowed; hiding it is not. Report the pre-registered analysis as the primary result, then report the deviation separately with the reason, clearly labeled. If a genuine data problem invalidates the plan — a logging bug corrupted the primary metric — say so, and treat the revised analysis as exploratory requiring confirmation. The test is whether the reason for deviating is independent of the result: "we discovered the metric was double-counting" is legitimate; "the pre-registered test was $p=0.06$ so we tried a different one" is not. Writing down the reason at the time you deviate, before you see the new result, is what keeps that distinction honest.

---

### Q: Explain the replication crisis in one paragraph.

**Answer.** Across psychology, medicine, economics, and increasingly ML, a large fraction of published findings fail to replicate when independently repeated. The Open Science Collaboration's 2015 project re-ran 100 psychology studies: 97% of the originals reported significant results, but only 36% of replications reached significance, and the replication effect sizes averaged roughly half the originals. Amgen and Bayer reported similar or worse rates attempting to reproduce preclinical cancer biology. The causes are structural rather than fraudulent: publication bias filters for significant results so the literature is a biased sample of what was run; underpowered studies mean that the significant results which do appear are inflated by the winner's curse (Type M error) and sometimes have the wrong sign (Type S); researcher degrees of freedom let flexible analysis manufacture significance; and incentives reward novel positive findings over replication, so nobody checks. Ioannidis's 2005 argument that "most published research findings are false" follows directly from base rates — if the prior probability an arbitrary tested hypothesis is true is low, and power is low, then the majority of significant results are false discoveries even with everyone behaving honestly. In ML specifically the equivalent failures are benchmark overfitting through repeated test-set use, unequal hyperparameter tuning between the proposed method and baselines, non-reported seed variance, and comparisons against undertuned baselines — which is why several careful reproducibility studies have found that older methods, tuned properly, match much-hyped newer ones.

**Follow-up: "What does this mean for how you run experiments at a company?"** → Concretely: assume a substantial fraction of internal "wins" are false, and build the process accordingly. Power tests properly, because underpowered tests do not merely fail to detect — they produce inflated estimates when they succeed. Pre-register the primary metric. Hold back long-term holdouts so you find out which shipped wins were real, and expect the aggregate holdout effect to be well below the sum of the individual reported wins (several large companies have published exactly this gap). Replicate anything surprising or high-stakes before acting on it. Make null results cheap to report so people stop hunting. And track the organization's historical hit rate, because that prior is what determines the false discovery rate on every future test, and knowing it changes how much you should believe any single $p = 0.04$.

---

### Q: Can a statistically significant result be practically worthless? Explain the difference between statistical and practical significance.

**Answer.** Yes, routinely, and the mechanism is that the p-value confounds effect size with sample size.

$$t = \frac{\text{effect size}}{\text{standard error}} = \frac{\Delta}{\sigma/\sqrt{n}} = \frac{\Delta\sqrt{n}}{\sigma}$$

For *any* nonzero $\Delta$, however tiny, $t \to \infty$ as $n \to \infty$. Significance is guaranteed at sufficient scale. And since no two real interventions have *exactly* identical effects, the null is essentially always false, so with enough data every test eventually rejects. At that point the p-value tests sample size, not importance.

**Concrete.** A test on 50 million users finds conversion up from 10.000% to 10.012% — a 0.12% relative lift, $p = 0.001$. Highly significant. Is it worth anything? If the surface drives \$100M annually, 0.12% is \$120k — set against engineering, maintenance, and the added complexity in a codebase that every future change must navigate. Probably not worth shipping. The statistics are impeccable and the decision is no.

**Practical significance** asks whether the effect is large enough to matter, and it is defined by the domain, never by the data:
- Compare the effect to a **pre-specified threshold** — the MDE derived from the break-even calculation. If you set that threshold before the test, this conversation is already settled.
- Report the **confidence interval in business units**: "+\$120k/year, 95% CI \$40k to \$200k" is decision-ready; "$p=0.001$" is not.
- Use **standardized effect sizes** for cross-context comparison (Cohen's $d$, with 0.2/0.5/0.8 as small/medium/large conventions — noting these are arbitrary conventions, not laws).
- Weigh the **cost side**: maintenance, latency, complexity, and the opportunity cost of the next experiment on that surface.

**The mirror-image error** is just as damaging: a practically enormous effect that is not significant because the test was underpowered. A 20% lift with a CI of $[-5\%, +45\%]$ is not evidence of nothing — it is an inconclusive test of something potentially very valuable, and killing the idea on $p = 0.28$ throws away a likely win. The right response there is more data, not a verdict.

**How I present results.** Effect size with a confidence interval in business units first; the practical threshold that was agreed in advance; the p-value last, if at all. And when the interval straddles the threshold, I say explicitly that the test does not resolve the decision, and name what would — more traffic, a longer horizon, or a bolder treatment.

**Follow-up: "What's the equivalence-testing version of this?"** → When the question is "is this the same?" rather than "is it different?" — a cheaper model, a refactor, an infrastructure migration — a non-significant test is *not* evidence of equivalence, since it may just be underpowered. The correct tool is **TOST** (two one-sided tests): pre-specify an equivalence margin $\delta$, then test $H_0: \Delta \le -\delta$ and $H_0: \Delta \ge +\delta$ separately, each one-sided at $\alpha$. Rejecting both establishes $|\Delta| < \delta$. Operationally it reduces to: the 90% CI (for $\alpha=0.05$ TOST) must lie entirely within $(-\delta, +\delta)$. This flips the burden of proof correctly — you must now *demonstrate* similarity rather than merely fail to demonstrate difference, and an underpowered test correctly fails to establish equivalence rather than falsely reassuring you. It is also exactly the right frame for guardrail metrics and for model-swap decisions.
