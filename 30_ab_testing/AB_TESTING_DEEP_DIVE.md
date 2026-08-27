# A/B Testing — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

A/B testing is how product decisions actually get made. ML interviews probe this hard because every model launch eventually goes through one — and most ML teams have at least one story of an A/B test that gave the wrong answer for a subtle reason.

---

## 1. Anatomy of an A/B test

You want to know: does change $C$ (a treatment) cause a meaningful improvement in metric $M$ versus the control?

**Components:**
- **Population**: who is randomized.
- **Randomization unit**: user, session, request, etc.
- **Treatments**: control (A) vs treatment (B).
- **Outcome metric**: what you measure.
- **Sample size**: how many units in each arm.
- **Duration**: how long to run.
- **Analysis**: how to compute the test.

The point of randomization is to make the two arms comparable in expectation — eliminates confounding.

> **Saying it out loud.** An A/B test is just a fair coin plus patience. You split users at random, show one group the change, show the other group the status quo, and because assignment was random the two groups are alike in every way except the thing you changed — so any difference in the metric is caused by the change. The pieces you have to decide up front are the randomization unit, the metric, the sample size and how long you'll run. The one that trips people up is the randomization unit: if you randomize by session instead of by user, the same person can see both versions, which contaminates the comparison and shrinks your effect toward zero.

---

## 2. Hypothesis testing for A/B

**In plain terms.** All of these tests answer the same question — could this difference plausibly have come from random noise alone? They differ only in what shape the data is: yes/no outcomes, continuous numbers, or something too skewed to trust a formula on.

### Two-proportion z-test (CTR-like metrics)

Null: $p_A = p_B$. Test statistic:

$$
z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}
$$

with $\hat{p}$ the pooled rate. Reject $H_0$ at $\alpha = 0.05$ if $|z| > 1.96$.

### Welch's t-test (continuous metrics with possibly different variances)

$$
t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{s_A^2/n_A + s_B^2/n_B}}
$$

### Mann-Whitney / Wilcoxon

Non-parametric — when distributions are weird (heavy tail, ordinal). Tests stochastic dominance, not equality of means.

### Bootstrap

Always works. Resample (within each arm) thousands of times, compute the difference, take quantiles.

> **Saying it out loud.** Which test I use comes down to what the metric looks like. For a rate — click-through, conversion, anything yes-or-no — it's a two-proportion z-test, and a z-score above 1.96 is the classic five percent cutoff. For a continuous metric like time on site, it's Welch's t-test, which doesn't assume the two groups have equal variance, and there's basically no reason to use the equal-variance version. When the distribution is ugly — revenue per user, where one whale dominates — I'd bootstrap instead, resampling within each arm a few thousand times and reading the interval straight off the quantiles. Bootstrap is slower but it makes no distributional assumptions, and for heavy-tailed money metrics that's worth a lot.

---

## 3. Sample size and power

**In plain terms.** Power analysis answers one question before you run anything: how many users do I need so that if the effect is real, I'll actually see it? The inputs are how big an effect you care about, how noisy the metric is, and how often you're willing to be wrong in each direction. The output is a number of users per arm.

The most-asked A/B testing question in interviews.

### What you need
- $\alpha$: significance level (typically 0.05).
- $\beta$: false negative rate (typically 0.2 → power 0.8).
- $\sigma$ or $\hat{p}(1-\hat{p})$: variability.
- $\delta$: minimum detectable effect (MDE).

### Formula for proportions (two-sided test)

$$
n \approx \frac{2 \sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}
$$

For 80% power and 5% significance:

$$
n \approx \frac{16 \sigma^2}{\delta^2}
$$

The "16" rule of thumb: per arm, you need $\approx 16 \sigma^2 / \delta^2$ samples.

### Implications

Halving the MDE quadruples the sample size. Detecting tiny effects requires huge experiments. This is why product teams need to think hard about effect sizes:
- Big change → small experiment can detect it.
- Tiny change → may need millions of users → may not be worth the risk.

### Variance reduction

You can sometimes detect smaller effects with the same $n$ by reducing variance:
- **CUPED** (Controlled Pre-Experiment Data): regress outcome on a pre-experiment covariate; analyze residuals.
- **Stratification**: stratify randomization by a known covariate.
- **Variance reduction via baselines**: subtract pre-period baseline.

CUPED can reduce variance 30–50% in real systems. Since required $n$ scales linearly with variance, that's equivalent to ~1.4× to 2× more effective sample size for free.

> **Saying it out loud.** The whole power calculation collapses to one rule of thumb: about sixteen sigma-squared over delta-squared per arm. Sigma-squared is how noisy your metric is and delta is the smallest effect you care about, so a noisier metric or a smaller effect both cost you users. The brutal part is the square: halving the effect you want to detect quadruples the sample size. That's why detecting a 0.1 percentage-point lift on a one percent baseline takes roughly a hundred and sixty thousand users per arm, and why teams chasing tiny effects run out of traffic before they run out of ideas. If you're short on users, CUPED is the cheat code — regress out a pre-experiment covariate and you typically cut variance thirty to fifty percent, which is like getting one and a half to two times the traffic for free.

---

## 4. Common pitfalls

### Peeking / sequential testing

Looking at results before the planned end and stopping when "significant" inflates Type I error dramatically. With repeated peeks, the chance of *eventually* seeing $p < 0.05$ approaches 1.

**Fix**: use sequential / always-valid statistics (Howard et al. 2021), or commit to a fixed sample size and don't peek, or apply alpha-spending.

> **Saying it out loud.** So the reason you can't just peek at the p-value is that every look is another chance to get lucky. If you check daily and stop the moment it dips under 0.05, you're not running one test at five percent risk — you're running twenty tests and reporting the winner, and your real false-positive rate climbs toward thirty or forty percent. Keep peeking forever and you'll hit significance essentially every time, even when there's no effect at all. The fixes are to commit to a sample size and genuinely not look, or to use always-valid sequential statistics that are built to be monitored continuously. Those cost you some power in exchange for letting you stop early honestly, which is usually a trade worth making.

### Multiple testing across metrics

Run 20 metrics, one comes back significant: probably noise.

**Fix**: pre-register a small set of primary metrics, apply Bonferroni or BH for secondary, treat exploratory metrics as descriptive only.

> **Saying it out loud.** If you test twenty metrics at the five percent level, you expect one false positive by pure chance — so "one of our twenty metrics moved" is not news. The fix is to decide before the experiment which one metric decides the launch, and treat everything else as descriptive. If you genuinely need several primaries, correct for it: Bonferroni is the blunt version, dividing your threshold by the number of tests, and Benjamini-Hochberg is the less conservative one that controls the false discovery rate. The thing to say out loud is that a metric which becomes significant only after you went looking is a hypothesis, not a result.

### SUTVA violations (network effects)

Stable Unit Treatment Value Assumption: one user's outcome doesn't depend on another user's treatment.

Violated by:
- Marketplaces: a user's bid affects others' prices.
- Social platforms: treatment user's posts affect control users' feeds.
- Promotions: treatment exhausts inventory available to control.

**Fix**: cluster-randomization (whole groups assigned to one arm), geo experiments, switchback tests.

> **Saying it out loud.** SUTVA is the assumption that my outcome doesn't depend on your treatment, and it quietly breaks in exactly the places you most want to run experiments. On a social network, a treated user posts more and control users see it in their feeds, so control drifts toward treatment and your measured effect is too small. In a marketplace it's worse than dilution — if treated buyers snap up scarce inventory, control users are made actively worse off and the effect looks inflated. The fix is to randomize at the level where interference stops: whole cities, whole social clusters, or switchbacks where the entire market flips between arms over time. The cost is a huge loss of statistical power, since your effective sample size becomes the number of clusters, not the number of users.

### Sample ratio mismatch (SRM)

Random assignment should give $n_A : n_B$ matching your planned ratio. If observed split deviates significantly (chi-squared test), randomization is broken or there's a logging bug. Don't trust results.

> **Saying it out loud.** Sample ratio mismatch is the first thing I check and the one that invalidates everything downstream. If you planned a fifty-fifty split and you're seeing fifty-point-two to forty-nine-point-eight on ten million users, that tiny gap is wildly improbable by chance — a chi-squared test will scream — and it means something in your assignment or logging is broken. Usually it's a bug that drops users asymmetrically: the treatment has a slow path and impatient users bail before logging. The reason it's fatal is that the users you lost aren't random, so your two arms are no longer comparable and the metric difference could be entirely selection. When I see SRM, I don't discount the result, I discard it and go find the bug.

### Novelty / primacy effects

Users react to *change*, not the steady-state experience. Wait for the effect to stabilize (1–2 weeks for product changes).

> **Saying it out loud.** Novelty and primacy are the same problem pointing in opposite directions. Novelty is when people click the new thing because it's new, so week one looks great and the effect decays to nothing. Primacy is when people are used to the old layout, fumble the new one, and the effect looks bad at first and improves as they learn. Either way, day-one numbers are lying to you. The fix is to run one to two weeks and look at whether the effect curve has flattened, and if you want to be rigorous, compare new users — who have no habits to disrupt — against existing users. The tradeoff is calendar time, and that's the honest cost of not shipping something that was only ever good for a week.

### Selection bias / opt-in cohorts

If the population in the test isn't representative of the deployment population, results don't generalize. Don't run on power users only.

> **Saying it out loud.** If your test population isn't your launch population, your result won't survive contact with reality. Running an experiment on power users or on people who opted into a beta gives you an effect estimate for enthusiasts, and enthusiasts respond to changes differently from everyone else. The same problem shows up if the treatment itself changes who's in the sample — anything that makes users leave before they get logged. So randomize over the population you actually intend to ship to, and check that the arms match on pre-experiment covariates. The failure mode is a beta test that shows a fifteen percent lift and a full launch that shows two.

---

## 5. Effect-size metrics

Beyond $p$-values:

**Absolute lift**: $\hat{p}_B - \hat{p}_A$.

**Relative lift**: $(\hat{p}_B - \hat{p}_A)/\hat{p}_A$.

**Cohen's $d$**: standardized effect size $(\bar{X}_B - \bar{X}_A)/s$.

Always report effect size + CI. A $p < 0.001$ tells you "not noise" but not "how much it matters." With huge $n$, trivial effects can be highly significant.

> **Saying it out loud.** A p-value tells you the effect probably isn't zero; it tells you nothing about whether it's worth shipping. With ten million users you can get p below 0.001 on a lift so small no human would notice it. So I always report the effect with a confidence interval — absolute lift, relative lift, and for continuous metrics a standardized effect like Cohen's d. The interval is the useful part, because it bounds how good or bad this could plausibly be. "Plus 0.4 percent, interval from 0.1 to 0.7" is a decision; "p equals 0.03" is trivia.

---

## 6. Bayesian A/B testing

Frequentist: "Is there an effect (yes/no)?"
Bayesian: "What's the probability that B is better than A?"

### Beta-Binomial setup

For CTR with Beta($\alpha, \beta$) priors:
- Posterior $A$: $\mathrm{Beta}(\alpha + s_A, \beta + n_A - s_A)$.
- Posterior $B$: $\mathrm{Beta}(\alpha + s_B, \beta + n_B - s_B)$.

$\mathbb{P}(p_B > p_A | \mathrm{data})$ via simulation: sample posteriors, compute fraction where $p_B > p_A$.

### Advantages
- Direct probability statement.
- Natural sequential testing (posterior is always valid).
- Decision theory: weight gain by probability and cost.

### Disadvantages
- Prior choice can be controversial.
- Communication: stakeholders are used to $p$-values.

Both frameworks are valid; choice depends on org and context.

> **Saying it out loud.** Bayesian and frequentist ask different questions, and the Bayesian one is the one people actually mean. A p-value answers "how weird is this data if there's no effect," which nobody's intuition handles. A Bayesian analysis gives you the probability that B beats A, and the expected loss if you pick wrong — which is directly the thing a product manager wants. For click-through rates the math is easy: Beta priors, Binomial data, conjugate posteriors, sample from both and count how often B wins. The big practical advantage is that posteriors are always valid, so peeking isn't a sin. The costs are that someone has to defend the prior, and that most orgs' dashboards and instincts are built around p-values.

---

## 7. ML-specific A/B testing

### Recommender / ranker tests

Outcome metrics like CTR, dwell time, retention. Issues:
- **Position bias**: users click higher-ranked items more.
- **Long-term effects**: short-term CTR ≠ long-term satisfaction.
- **Holdback experiments**: keep a small population on the old model permanently to measure long-term drift.

> **Saying it out loud.** Testing a ranker is harder than testing a button, because the model changes what data you collect. Position bias means clicks reflect where you put things, not just what people want, so a naive click metric rewards any change that reshuffles the top slot. And short-term CTR is a treacherous proxy — clickbait wins on CTR and loses on retention, so I'd pair it with dwell time and week-four retention. The thing I'd argue for is a long-running holdback: keep one or two percent of users on the old model indefinitely, so you can measure cumulative drift months later. Without it you can ship twelve consecutive one-percent wins and have no idea whether the product got better overall.

### Online learning systems

If the model trains on user behavior, the test arm influences future training data. This can cause drift between treatment and control models.

> **Saying it out loud.** If the model trains on the behavior it produces, your two arms stop being comparable over time. The treatment model learns from treatment users' clicks and gets better at serving them; the control model learns from control. So you're not comparing two fixed systems, you're comparing two systems on divergent trajectories, and the gap you measure depends on how long you ran. There's also a spillover risk if both arms feed one shared training pipeline, which contaminates the control outright. The clean setup is separate training data per arm, and you accept that this makes the experiment more expensive to run.

### Counterfactual evaluation

Sometimes you don't want to ship A/B for risk reasons. Instead, **off-policy evaluation**: estimate what would have happened under the new policy from logged data of the old policy. Methods: importance sampling (IPS), doubly robust estimators.

> **Saying it out loud.** Sometimes you can't run the test — it's too risky, too slow, or legally off-limits — so you estimate what would have happened from logs. Off-policy evaluation reweights the data you already collected by how likely the new policy would have been to take each logged action; that's inverse propensity scoring. Doubly robust estimators add a model of the outcome on top, so you get the right answer if either the propensity model or the outcome model is correct. The hard requirement is that the logging policy had some randomness — if the old system was deterministic, there's nothing to reweight and no amount of cleverness recovers the counterfactual. And the variance explodes when the new policy wants to do things the old one almost never did.

### Interleaving for ranker comparison

Instead of A/B-ing entire pages, interleave items from A and B in the same page; measure which side users click more. More statistically powerful per user but harder to set up.

> **Saying it out loud.** Interleaving is a much more sensitive way to compare two rankers. Instead of showing one user ranker A and another user ranker B, you take one user and blend both rankers' results into a single list, then see which side's items they click. Because the comparison happens within a single user on a single page, you cancel out all the between-user variance, and you typically need ten to a hundred times fewer users than a standard A/B. The catch is that it only works for ranking comparisons — you can't interleave a UI change — and the interleaving policy itself has to be carefully unbiased. So it's a great screening tool: interleave to pick the candidate, then A/B the winner to measure the real business effect.

---

## 8. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Sample size for 1% baseline + 0.1pp lift @ 80% power? | "Use a calculator" | $n \approx 16 \sigma^2/\delta^2 \approx 16 \cdot 0.0099 / 10^{-6} \approx 158{,}000$ per arm. ($\sigma^2 = p(1-p) = 0.01 \cdot 0.99 = 0.0099$; $\delta = 0.001 \Rightarrow \delta^2 = 10^{-6}$.) |
| Can I peek and stop early? | Sure | Inflates Type I error; use sequential testing or commit to $n$ |
| Two-sided vs one-sided test? | Use whichever | Two-sided default; one-sided only if you commit *a priori* and direction is justified |
| Significant ($p < 0.05$) means meaningful? | Yes | Significant just means "not noise"; report effect + CI |
| Random assignment guarantees comparability? | Always | In expectation; SRM check in practice |
| Network effects — what do you do? | Ignore | Cluster randomization, geo splits, switchback |
| Holdback test? | Same as A/B | Long-running control to measure long-term effect |

> **Saying it out loud.** The gotchas cluster into a few reflexes. Someone shows me a result and I check the split ratio before anything else, because SRM means the rest is noise. I report the effect and its interval, never the p-value alone. If they peeked, I ask whether the stopping rule was fixed in advance. If the product has users interacting with each other, I ask whether randomization was at the cluster level. And I default to two-sided tests, because a one-sided test is only legitimate if you committed to the direction before seeing anything. Each of these takes ten seconds to say and each one is a place real experiments die.

---

## 9. Eight most-asked interview questions

1. **Walk me through how you'd power an A/B test.** (MDE, baseline rate, $\alpha$, $\beta$, derive $n$.)

   > **Saying it out loud.** Powering a test is four inputs and one formula. I need the baseline rate of the metric, the smallest effect that would actually change our decision, the significance level — usually five percent — and the power, usually eighty percent. Then it's roughly sixteen times the variance over the effect squared, per arm. And I'd say out loud that the MDE is a business question, not a statistical one: the right number is the smallest lift that's worth the engineering cost, and if that requires more users than we have, the honest answer is that this experiment can't be run.

2. **You ran 20 metrics, two are significant at 0.05. What do you conclude?** (Multiple testing — apply correction; pre-register primary metrics.)

   > **Saying it out loud.** Twenty metrics at the five percent level means one false positive is the expected outcome, so two significant results is roughly what pure noise looks like. My conclusion is: nothing, unless one of them was the pre-registered primary metric. If they were all exploratory, I'd treat the hits as hypotheses to test properly next time. If we genuinely care about several, I'd apply Benjamini-Hochberg rather than Bonferroni, which is less brutal when you have many metrics. The thing to resist is the story the two significant metrics will inevitably suggest.

3. **What's CUPED and why use it?** (Variance reduction via pre-experiment covariate; ~30–50% more power for free.)

   > **Saying it out loud.** CUPED is free statistical power, and it's the most practical trick in experimentation. The idea is that a lot of the variance in someone's behavior during the experiment is explained by their behavior *before* the experiment — heavy users stay heavy. So you regress the outcome on that pre-period covariate and analyze the residual, which strips out variance that has nothing to do with your treatment. It's unbiased because the covariate predates randomization and can't be affected by treatment. In real systems it cuts variance thirty to fifty percent, which is like getting one and a half to two times the traffic without spending a day more.

4. **What goes wrong with network effects?** (SUTVA violation; treatment leaks to control via shared resources.)

   > **Saying it out loud.** Network effects break the assumption that arms are independent. On a social platform, a treated user's extra posts land in control users' feeds, so control gets partly treated and your measured effect shrinks toward zero. In a marketplace it can go the other way: treatment consuming scarce supply makes control genuinely worse, which inflates the difference. Either way the number you compute isn't the number you'd get at full launch. The fix is randomizing at the cluster level — geos, communities, or switchback windows — and the price is that your effective sample size drops to the number of clusters, which can be a hundred instead of a million.

5. **You see significance after 3 days. Stop the test?** (Peeking inflates Type I error; commit to $n$ or use sequential methods.)

   > **Saying it out loud.** No, not unless we agreed on that stopping rule before we started. Seeing significance on day three is exactly what peeking produces, because early in a test the estimate bounces around a lot and will cross the threshold by chance. Stopping there means you've captured the noise at its peak, and the effect you report will be biased upward — that's the winner's curse. If we want the option to stop early, we set that up in advance with a sequential test or alpha spending. Otherwise we run to the planned sample size, and I'd note the effect will almost certainly look smaller than it does today.

6. **CI overlaps zero — null result. Anything else to report?** (Effect size + CI to bound the maximum plausible effect; "no significant effect with 95% CI [-0.3, 0.5]" is way more informative.)

   > **Saying it out loud.** A null result is still a result, and the interval is where the information lives. "No significant effect" alone is useless; "no significant effect, with a ninety-five percent interval from minus 0.3 to plus 0.5 percent" tells you the change can't possibly be worth the maintenance cost, which is an actionable finding. If instead the interval is minus two to plus three, you've learned nothing and the test was underpowered — that's a different conversation, about whether to run longer. So I'd report the interval, compare its width to the MDE we designed for, and say clearly which of those two situations we're in.

7. **Bayesian vs frequentist A/B testing — pros/cons?** (Direct probability vs $p$-value; sequential properties.)

   > **Saying it out loud.** Frequentist gives you a p-value, which answers a question nobody asked: how surprising is this data if the effect is exactly zero. Bayesian gives you the probability B is better than A, plus the expected loss from choosing wrong, which is what the decision actually needs. Bayesian also handles peeking gracefully, since a posterior is valid whenever you look at it. The downsides are that the prior is a judgment call somebody can dispute, and that most organizations run on p-values so you'll spend energy on translation. In practice they usually agree, and I'd pick based on whether the org wants a launch decision or a significance test.

8. **You run an experiment, see SRM. What do you do?** (Don't trust the result; investigate randomization/logging; rerun.)

   > **Saying it out loud.** Sample ratio mismatch means I stop and don't analyze anything. The split should match what we configured, and a significant chi-squared deviation says assignment or logging is broken — which means the two groups differ in ways that have nothing to do with the treatment. Typical causes are a redirect that drops slow users, bot filtering applied unevenly, or a bug in the assignment hash. So I'd go find the cause, fix it, and rerun. Analyzing through SRM is the single fastest way to ship a change based on a completely fictional lift.

---

## 10. Drill plan

- Compute sample size for 3 scenarios on paper: CTR baseline 1% with 10% relative lift, MAU baseline 50% with 1pp lift, revenue with 5% relative lift.
- Recite definition of: SRM, SUTVA, novelty effect, peeking, primacy effect.
- Explain CUPED in 2 minutes.
- Walk through one case where naive A/B gives wrong answer due to network effects.
- Prepare answers to: "you're an ML engineer pushing a model launch — describe the experiment plan."

---

## 11. Further reading

- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* — the canonical practitioner book.
- Deng et al., *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (2013) — CUPED.
- Howard, Ramdas, McAuliffe, Sekhon (2021), *Time-uniform, nonparametric, nonasymptotic confidence sequences*.
- Karrer et al. (2021), *Network experimentation at scale* (Facebook).
- Athey & Imbens, *The state of applied econometrics* — causal inference perspective.
