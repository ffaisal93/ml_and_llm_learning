# A/B Testing — Interview Grill

> 45 questions on experimental design, sample size, common pitfalls, and ML-specific tests. Drill until you can answer 30+ cold.

---

## A. Test design fundamentals

**1. What does randomization buy you?**
Comparability between arms in expectation. Removes confounding from observed and unobserved variables.

> **Saying it out loud.** Randomization buys you the right to say the word "caused." Because assignment is a coin flip, the two groups are alike in expectation on everything — the stuff you measured and, crucially, the stuff you didn't and couldn't. That's what separates an experiment from an observational study, where you're forever worrying about the confounder you forgot. The caveat is "in expectation": on a small sample any single randomization can be unlucky, which is why you still check that the arms balance on pre-experiment covariates before you trust the result.

**2. What is the randomization unit?**
The level at which you assign treatment: user, session, request, device. Choice depends on what's independent and where you want to measure effects.

> **Saying it out loud.** The randomization unit is the level you flip the coin at, and picking it wrong quietly ruins the test. User is the default, because a person should get a consistent experience — if you randomize by session, the same person sees both versions, their behaviors bleed together, and your effect gets diluted toward zero. Sometimes you have to go coarser: cluster or geo-level when users interact, since that's the only way to contain spillover. The tradeoff is power — coarser units mean fewer independent samples, so a geo test with fifty cities has a sample size of fifty, not fifty million.

**3. Two-sided vs one-sided test — default?**
Two-sided. One-sided only if you committed *a priori* to a direction with a strong reason.

> **Saying it out loud.** Two-sided is the default, and you should be suspicious of anyone who reaches for one-sided. A one-sided test is only legitimate if you committed to the direction before seeing any data *and* you'd genuinely take no action if the effect went the other way — which is almost never true, because a big negative result is extremely actionable. What makes it dangerous is that switching to one-sided is a free halving of your p-value, so it's the most tempting form of cheating. If you do use it, say so in the pre-registration, not in the analysis.

**4. Two-proportion z-test statistic?**
$z = (\hat{p}_A - \hat{p}_B) / \sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}$ with pooled $\hat{p}$.

> **Saying it out loud.** The z-test is just "how many standard errors apart are these two rates." You take the difference in click-through rates on top, and on the bottom the standard error computed from the pooled rate across both arms — pooled because under the null they're the same rate. If the result is bigger than about 1.96 in absolute value, that's your five percent two-sided cutoff. The thing worth remembering is that it's an approximation that relies on the normal limit, so with a very low rate and a small sample — fewer than roughly five expected successes per arm — you should use an exact test instead.

**5. Welch's t-test vs Student's t-test — when?**
Welch when variances are different across groups (default in many libraries). Student's assumes equal variances.

> **Saying it out loud.** Welch's is the one you should use, essentially always. Student's t-test assumes both groups have the same variance, and in A/B testing that's often false — a treatment that changes behavior usually changes the spread too, not just the mean. Welch drops that assumption and adjusts the degrees of freedom accordingly. The cost of using Welch when variances actually are equal is a trivial loss of power; the cost of using Student's when they aren't is a wrong error rate. That's why most libraries default to Welch, and you should let them.

**6. Mann-Whitney — why use it?**
Non-parametric. When data is skewed, ordinal, or has heavy tails. Tests stochastic dominance.

> **Saying it out loud.** Mann-Whitney is what you reach for when the data is too ugly for a t-test to be trustworthy — heavy tails, ordinal ratings, small samples with obvious skew. Instead of comparing means it compares ranks, asking whether a random draw from B tends to exceed a random draw from A. That makes it robust to outliers, which is exactly the problem with revenue-style metrics. The catch is that it tests stochastic dominance, not the mean, so a significant result doesn't tell you the average went up — and the average is usually what the business is paid on. That's why for money metrics I'd usually bootstrap the mean instead.

**7. Bootstrap for A/B — pros?**
Always works. No distributional assumption. Easy to extend to weird metrics (e.g., ratios).

> **Saying it out loud.** The bootstrap works when nothing else does, and it's conceptually simple: resample your data with replacement thousands of times, recompute the statistic each time, and read the confidence interval straight off the percentiles. No distributional assumptions, and it extends to statistics with no clean analytical variance — ratio metrics like revenue per user, or a ninety-fifth-percentile latency. The price is compute, and the real limitation is that it can't rescue you from a tiny sample or from clustered data unless you bootstrap the clusters rather than the rows.

---

## B. Sample size and power

**8. State the rule-of-thumb sample size formula for two means.**
$n \approx 2\sigma^2 (z_{\alpha/2} + z_\beta)^2 / \delta^2$. For $\alpha=0.05$, power 0.8: $n \approx 16\sigma^2/\delta^2$ per arm.

> **Saying it out loud.** The formula is two sigma-squared times the sum of two z-scores squared, all over the effect squared — but the version to have memorized is the shortcut: sixteen sigma-squared over delta-squared per arm, which is exactly the general formula at five percent significance and eighty percent power. Sigma-squared is metric variance, which for a rate is just p times one minus p. Delta is the effect you want to detect. Sixteen comes from 1.96 plus 0.84, squared, times two. If you can recite that in an interview and then plug numbers into it, you've answered the most-asked A/B question there is.

**9. Halve the MDE — what happens to $n$?**
Quadruples ($n \propto 1/\delta^2$).

> **Saying it out loud.** It quadruples, because n goes like one over delta squared. That's the single most important consequence of the power formula, and it's why chasing tiny effects gets expensive so fast — going from detecting a one percent lift to a 0.5 percent lift means four times the users and four times the calendar time. It's also the argument for variance reduction: since n scales linearly with variance, cutting variance in half with CUPED gets you the same detection ability for half the traffic, which is much cheaper than doubling the experiment.

**10. Define statistical power.**
$1 - \beta = \mathbb{P}(\mathrm{reject}\, H_0 \mid H_1)$. Probability of detecting a real effect of size $\delta$.

> **Saying it out loud.** Power is the probability that you'll actually detect an effect that's really there. Formally it's one minus beta, the chance of correctly rejecting the null when the alternative is true, and the conventional target is eighty percent. What that means in practice is sobering: at eighty percent power, a real effect gets missed one time in five. The failure mode of underpowered tests isn't just missing things — it's that when an underpowered test does hit significance, the estimated effect is systematically inflated, because only the lucky-large estimates cross the line.

**11. What's the MDE?**
Minimum detectable effect — smallest effect you have power to detect at chosen $\alpha$ and $n$.

> **Saying it out loud.** The MDE is the smallest effect your experiment can reliably detect given how many users you have. You can read the power formula in either direction: fix n and solve for delta, and that's the MDE. The important framing is that it's a business number, not a statistical one — the right MDE is the smallest lift that would actually change the decision to ship. If the MDE your traffic supports is larger than any effect you plausibly expect, the honest conclusion is that this experiment can't be run and you should look for a more sensitive metric or a coarser question.

**12. CTR baseline 5%, want to detect 0.5pp absolute lift, 80% power, two-sided 5%. Roughly how many users per arm?**
$\sigma^2 \approx 0.05 \cdot 0.95 = 0.0475$. $\delta = 0.005$. $n \approx 16 \cdot 0.0475 / 0.000025 = 30{,}400$ per arm.

> **Saying it out loud.** Plug it in. Variance is 0.05 times 0.95, about 0.0475. The effect is half a percentage point, so 0.005, and squared that's 0.000025. Sixteen times 0.0475 over 0.000025 gives about thirty thousand per arm — call it thirty thousand four hundred. So sixty thousand users total, which at any reasonable traffic level is a few days. The value of doing this out loud is that it turns "we need a big sample" into a concrete number the room can react to, and it shows you can operate the formula rather than just quote it.

**13. Why do online experiments often need millions of users?**
Tiny effect sizes. CTR lifts on the order of 0.1pp on a 5% baseline → millions per arm.

> **Saying it out loud.** Because real effects are small and the formula punishes small effects quadratically. A tenth of a percentage point on a five percent baseline sounds trivial, and it is, but at scale it's worth millions — so companies genuinely want to detect it. Run the numbers: variance about 0.0475, delta 0.001, and you need something like three-quarters of a million users per arm. That's why only large platforms can measure small effects at all, and why smaller companies should focus on changes big enough to see.

**14. What's CUPED?**
Controlled Pre-Experiment Data: regress outcome on pre-period covariate, analyze residuals. Reduces variance ~30–50%.

> **Saying it out loud.** CUPED is the best free lunch in experimentation. The insight is that a big chunk of the variation in a user's metric during the experiment is explained by that same user's behavior *before* the experiment — heavy users were already heavy. So you regress the outcome on the pre-period value and analyze the residual, which strips out variance that has nothing to do with your treatment. It's unbiased because the covariate was fixed before randomization, so treatment can't have affected it. In practice it cuts variance thirty to fifty percent, which is like getting one and a half to two times your traffic for the cost of one extra join.

**15. Stratified randomization — why?**
Reduces variance by ensuring balance on known important covariates. Like a structured form of CUPED.

> **Saying it out loud.** Stratification balances the arms on something you know matters — country, platform, user tenure — by randomizing within each stratum instead of across the whole population. That removes the chance of an unlucky split where treatment happens to get more iOS users, and it reduces variance for the same reason CUPED does: you're accounting for known structure instead of letting it bounce around. The difference is timing — stratification happens at assignment, CUPED happens at analysis — and CUPED is usually easier because you don't need to change your assignment infrastructure. You can do both.

---

## C. Common pitfalls

**16. What's peeking and why is it bad?**
Looking at results before the planned end and stopping when significant. Inflates Type I error toward 1 with infinite peeks.

> **Saying it out loud.** Peeking is checking results before the planned end and stopping when you like what you see. It's bad because each look is another opportunity to cross the threshold by chance, so with daily checks over two weeks your real false-positive rate is more like thirty percent than five. Peek infinitely and you'll hit p below 0.05 with probability approaching one, even when the true effect is exactly zero. The second harm is bias: stopping at the moment things look best means the effect you report is inflated. Anyone who tells you they only peeked and didn't act on it is describing a rule they'd have broken if the number had been better.

**17. How to allow safe early stopping?**
Sequential analysis (Wald's SPRT, group sequential designs, alpha spending). Or always-valid $p$-values (mSPRT, e-values).

> **Saying it out loud.** If you want to stop early, you have to plan for it up front. Group sequential designs give you a small number of scheduled interim looks with adjusted, stricter thresholds at each — alpha spending, essentially rationing your five percent across the checkpoints. Always-valid p-values and confidence sequences go further and let you look whenever you want, because the guarantee holds uniformly over time. Both are legitimate; the cost is power, so if the effect turns out to be small you'll need somewhat more users than a fixed-horizon test would. That's the honest trade: the option to stop early isn't free, you pay for it in sample size.

**18. SUTVA — what is it and what violates it?**
Stable Unit Treatment Value Assumption. Each unit's outcome doesn't depend on others' assignments. Violated by marketplaces, social platforms, capacity constraints.

> **Saying it out loud.** SUTVA says my outcome depends only on my treatment, not on yours. It's the assumption underneath every simple A/B comparison, and it fails wherever users share something. In a marketplace they share inventory, so treatment buyers taking supply makes control worse. On a social platform they share feeds, so treatment users' content leaks into control. In an ad auction they share the auction itself. The consequence is that your measured difference is wrong in a direction you can't easily sign — dilution if treatment leaks into control, inflation if treatment actively harms control.

**19. Cluster randomization — why?**
When SUTVA fails at the user level, randomize at a higher level (groups, geographies) to keep interference within clusters.

> **Saying it out loud.** Cluster randomization means assigning whole groups — cities, communities, schools — to one arm, so the interference happens inside a cluster where both sides are in the same condition. That restores the comparison SUTVA broke. The price is brutal and worth naming: your effective sample size becomes the number of clusters, not the number of users, so a hundred-city test has n equal to a hundred. Your standard errors blow up accordingly, and you have to analyze at the cluster level or use a mixed model. So you only do it when interference is real, because otherwise you're throwing away most of your statistical power for nothing.

**20. What's a switchback test?**
For two-sided marketplace experiments, alternate treatments by time periods (e.g., one hour each) across the entire population. Eliminates network effects.

> **Saying it out loud.** A switchback flips the entire market between treatment and control over time — everyone gets version A for an hour, then everyone gets B for an hour, and you compare the periods. It's the standard answer for two-sided marketplaces like ride-hailing, where a pricing or dispatch change affects the whole supply pool and you simply can't have two systems running side by side. The tradeoffs are that you inherit time-of-day and day-of-week confounding, which you handle with randomized window ordering, and carryover effects, where the previous period's state bleeds into the next one — that's why windows have to be long enough for the market to settle.

**21. SRM — what is it and how do you check?**
Sample Ratio Mismatch. Observed split doesn't match planned split. Chi-squared test on counts. If significant, randomization is broken — don't trust the test.

> **Saying it out loud.** Sample ratio mismatch is when the observed split doesn't match the planned one. You check it with a chi-squared test on the raw assignment counts, and because you usually have millions of units it's extremely sensitive — a fifty-point-two versus forty-nine-point-eight split can be significant. If it fires, the randomization or the logging is broken, and I stop right there. The reason it's fatal is that whatever mechanism dropped users almost certainly dropped them non-randomly, so the arms are no longer comparable and the effect you measure could be pure selection. It's the first thing I check and the only check that can invalidate everything by itself.

**22. What's the novelty effect?**
Users react to *change* itself. Effect size shifts after initial exposure. Run experiments long enough for steady state (typically 1–2+ weeks).

> **Saying it out loud.** Novelty is people reacting to the fact that something changed rather than to what it changed into. A new UI gets extra clicks because it's unfamiliar and interesting, and that fades within days or a couple of weeks. The mirror image is primacy, where users are annoyed by the disruption and the metric dips before recovering. Either way, if you read the effect on day two you're reading a transient. So run one to two weeks, plot the effect over time to see whether it's flattened, and if you want to be rigorous, look at new users separately since they have no habits to disturb.

**23. Multiple metrics — what to do?**
Pre-register a small primary set; apply Bonferroni or BH correction across them. Treat exploratory metrics as descriptive.

> **Saying it out loud.** With multiple metrics your five percent error rate is per-metric, not overall, so twenty metrics means one false hit is the expected outcome. The discipline is to pre-register exactly one primary metric that decides the launch, plus a short list of guardrails, and treat everything else as exploratory description. If you truly have several primaries, correct — Bonferroni divides your alpha by the number of tests and is very conservative, Benjamini-Hochberg controls the false discovery rate and is the better choice when you have many correlated metrics. What you don't do is go hunting after the fact and report what you found.

**24. You see one significant metric out of 20. What do you conclude?**
Likely false positive ($\sim 1$ expected by chance). Apply correction or treat as exploratory.

> **Saying it out loud.** I conclude nothing. At a five percent threshold across twenty metrics, one significant result is precisely what you expect from pure noise, so this is the null outcome dressed up. Unless that metric was the pre-registered primary, I'd treat it as a hypothesis worth testing properly rather than a finding. And I'd resist the story — there's always a plausible mechanism you can invent after the fact for whichever metric happened to move. If it matters, run a confirmatory test with that metric named in advance.

---

## D. Effect-size reporting

**25. Why report effect size + CI, not just $p$-value?**
$p$-value tells you "not noise." Effect size tells you "by how much" — what actually matters for product decisions. With huge $n$, trivial effects can be significant.

> **Saying it out loud.** A p-value only tells you the effect probably isn't exactly zero, and with millions of users almost nothing is exactly zero. The effect size and its interval tell you whether it's worth anything. I've seen p below 0.001 on a lift so small the engineering cost of maintaining the feature exceeds its value. The interval is the part that drives decisions, because it bounds the plausible range: an interval from plus 0.1 to plus 0.4 percent says small but real, while minus two to plus three says we learned nothing. Report effect, interval, and then the p-value if anyone still cares.

**26. Cohen's $d$?**
Standardized effect: $(\bar{X}_B - \bar{X}_A)/s$. Rule of thumb: 0.2 small, 0.5 medium, 0.8 large.

> **Saying it out loud.** Cohen's d is the effect expressed in standard deviations — the difference in means divided by the pooled standard deviation — which makes it comparable across metrics with different units. The conventional anchors are 0.2 small, 0.5 medium, 0.8 large. It's useful for describing behavioral effects, but be careful in product settings: online experiments routinely produce d values around 0.01 that are enormously valuable in revenue terms. So the rule-of-thumb labels come from psychology, not from business, and I'd report the raw lift alongside it.

**27. Absolute vs relative lift — which to report?**
Both. Absolute for low baselines (1pp lift on 1% is huge); relative for higher baselines.

> **Saying it out loud.** Report both, because each one lies in a different direction. Absolute lift is what you need at low baselines — going from one percent to two percent is only one percentage point but it's a doubling. Relative lift is what you need at high baselines, where a percentage point on a fifty percent metric is barely anything. Relative also travels better across segments with different baselines. The failure mode is quoting only relative lift on a tiny baseline, which is how a change that affected four hundred users becomes "a hundred percent improvement" in a deck.

**28. CI of difference is $[-0.3\%, +1.0\%]$. What can you say?**
Cannot reject null (CI includes 0). True effect is somewhere in this range with 95% confidence; could be anywhere from slightly negative to moderately positive. Decide based on minimum interesting effect.

> **Saying it out loud.** That interval straddles zero, so I can't reject the null — but I've still learned something. The honest statement is that the true effect is probably between a 0.3 percent decline and a one percent gain, so if our minimum interesting effect was, say, one and a half percent, we've effectively ruled out the case for shipping. If our minimum interesting effect was 0.2 percent, the test just didn't have the power to answer the question and I'd want to run longer or reduce variance. That's the useful move: compare the interval width against the MDE you designed for, and say which of those two situations you're in.

---

## E. Bayesian A/B

**29. Bayesian A/B for two CTRs?**
Beta priors → Beta posteriors after observing data. Sample posteriors and compute $\mathbb{P}(p_B > p_A | \mathrm{data})$ by simulation.

> **Saying it out loud.** For click-through rates the Bayesian setup is genuinely easy because Beta and Binomial are conjugate. Start with a Beta prior on each arm's rate, add successes and failures to the two parameters, and you've got the exact posterior — no MCMC needed. Then draw a hundred thousand samples from each posterior and count the fraction where B beats A; that's your probability B is better. You can compute expected loss the same way, which is the number that actually supports a decision. With reasonable traffic the prior barely matters, and you can say so to defuse the usual objection.

**30. Advantages of Bayesian framing?**
Direct probability statements ("70% chance B is better"). Sequential analysis is natural. Easier business communication.

> **Saying it out loud.** The big advantage is that it answers the question people are actually asking. "There's a ninety-three percent chance B is better and the expected loss from shipping it is 0.02 percent" is a sentence a product manager can act on; "we reject the null at p equals 0.04" isn't. Sequential monitoring is also natural, since a posterior is valid whenever you look at it — no peeking penalty. And you can fold in a cost function directly, deciding by expected value rather than by a significance threshold that has nothing to do with your business.

**31. Disadvantages?**
Prior choice. Stakeholders may prefer $p$-values. Computational cost for non-conjugate cases.

> **Saying it out loud.** The main objection is the prior — someone has to choose it, and a skeptical stakeholder can always argue you chose one that flattered your result. In practice with large samples the prior washes out, so this is more of a political problem than a statistical one, but it's real. Second, communication: most organizations have years of dashboards and habits built around p-values, so you'll spend energy translating. Third, non-conjugate models need sampling, which costs compute. And the peeking freedom is easy to oversell — you still shouldn't stop the instant probability crosses ninety-five percent, because you'll systematically overestimate the effect.

---

## F. ML-specific tests

**32. Position bias in ranker A/B?**
Higher positions get more clicks regardless of relevance. Naive metric like CTR doesn't isolate ranker quality.

> **Saying it out loud.** Position bias means people click the top result partly because it's on top, not because it's the best answer. That contaminates any click-based comparison of two rankers: a change that just shuffles items can move CTR without improving relevance at all. It also poisons your training data, since tomorrow's model learns from clicks shaped by today's ranking. The standard fixes are inverse propensity weighting with position propensities estimated from small randomized swaps, or interleaving, which sidesteps the problem by putting both rankers on the same page. And I'd never evaluate a ranker on raw CTR alone without saying this.

**33. Interleaving — what is it?**
Mix items from rankers A and B on a single result page; track which side users click. More powerful per user than full A/B.

> **Saying it out loud.** Interleaving mixes the two rankers' results into one list and watches which side gets the clicks. Because the comparison happens within a single user on a single page, all the between-user variance cancels — the same person is the control and the treatment — and you typically need ten to a hundred times fewer users than a conventional A/B. It's also much less exposed to position bias if the interleaving policy is designed properly, like team-draft interleaving. The limits are that it only works for ranking comparisons and it gives you a relative preference, not a business metric. So the pattern is: interleave to choose, A/B to quantify.

**34. Holdback test?**
Permanent (or long-running) control arm to measure long-term effects of model changes. Catches drift that short A/B misses.

> **Saying it out loud.** A holdback is a control arm you never turn off. You keep one or two percent of users on the old model or old experience indefinitely, so months later you can still measure the cumulative effect of everything you shipped. The reason you need it is that individual two-week tests each show a small win, but they can't see slow drift — retention erosion, feed narrowing, ad-load fatigue — and twelve consecutive one-percent wins can add up to a product that got worse. The cost is real: you're deliberately withholding improvements from some users, and you have to keep the old code path alive.

**35. Online learning system A/B — what's tricky?**
Treatment arm trains on its own user behavior; control trains on its own. Models drift apart over time. Effect mixes "the new architecture" with "the new training data."

> **Saying it out loud.** The tricky part is that the arms train on their own data, so they diverge in ways that aren't the change you're testing. The treatment model learns from treatment users' behavior and gets progressively better tuned to them; control does the same. So the gap you measure grows over time and depends on how long you ran, which makes it hard to say what you'd get at full launch. There's also contamination risk if both arms feed a shared training pipeline, which corrupts the control. The clean setup is separate training data per arm, and you should say explicitly whether you're measuring the architecture, the data, or both.

**36. Counterfactual / off-policy evaluation — when?**
When you can't safely run live A/B. Use logged data + propensity scores (IPS) or doubly robust estimators to estimate what would have happened.

> **Saying it out loud.** Off-policy evaluation is what you do when running the test is too risky, too slow, or not allowed — a pricing policy you can't ethically randomize, or a hundred candidate models you can't test one at a time. You use logged data from the old policy and reweight it to estimate what the new policy would have earned. The hard requirement is that the old policy had some randomness: if it was deterministic, some actions have zero logging probability and no estimator can recover them. So this is a screening tool for narrowing candidates, not a replacement for the A/B you eventually run on the winner.

**37. IPS estimator?**
$\hat{V}(\pi_{\mathrm{new}}) = \frac{1}{N}\sum_i \frac{\pi_{\mathrm{new}}(a_i|x_i)}{\pi_{\mathrm{old}}(a_i|x_i)} r_i$. Reweight rewards by policy ratio. High variance for big policy changes.

> **Saying it out loud.** IPS reweights each logged reward by the ratio of how likely the new policy was to take that action versus how likely the old one was. If the new policy would have done what the old one did, weight one; if it strongly prefers something the old policy rarely tried, that rare event gets a huge weight. That's unbiased, which is the appeal — and it's also the problem, because a few enormous weights make the variance explode when the two policies differ a lot. Fixes are weight clipping, which trades a little bias for a lot of variance, or a doubly robust estimator that adds a reward model and stays correct if either component is right.

---

## G. Communication and decision

**38. You ran an A/B. Result: control 5.0% CTR, treatment 5.05%, $p$ = 0.04. Ship?**
Depends on cost of treatment, business context, secondary metrics, novelty. Significance ≠ ship-worthy. 0.05pp absolute lift may be tiny.

> **Saying it out loud.** A 0.05 percentage-point lift on a five percent baseline is a one percent relative improvement, and p equals 0.04 means it's just barely distinguishable from noise — so I'd want the confidence interval before saying anything. It probably runs from nearly zero to something like a tenth of a point, meaning the true effect could be negligible. Then the question stops being statistical: what does it cost to build, maintain and carry this forever, and did any guardrail move? At a large enough platform, one percent relative CTR is worth shipping. At a small one, it isn't worth the code. Significance is an input to that decision, not the decision.

**39. Treatment looks great on primary metric, worse on a secondary "guardrail" metric. What do you do?**
Don't ship by default. Investigate the guardrail decline. Negative effects on user retention or engagement matter even if primary metric improves.

> **Saying it out loud.** Default answer is don't ship, and then go understand the guardrail. Guardrails exist precisely because it's easy to move the primary metric by doing something harmful — more notifications lift engagement and raise unsubscribes. So I'd first check whether the guardrail move is real or just noise across many metrics, then look for the mechanism. If it's real, the question is whether the tradeoff is acceptable, which is a business decision that needs to be made explicitly and by the right people, not quietly resolved by whoever's writing the analysis. Shipping over a guardrail without a conversation is how you get a metric that takes a year to recover.

**40. Treatment improves overall but hurts a specific user segment. Ship?**
Depends. Equity considerations matter — sometimes you ship; sometimes you fix the segment-specific regression first.

> **Saying it out loud.** It depends on which segment and how badly. First I'd check whether the segment effect is real or just the multiple-comparisons artifact you get from slicing twenty ways — a subgroup finding needs a much higher bar. If it's real, the questions are how big the harm is, how large the group is, and whether it's a protected or otherwise sensitive population, because a small average gain funded by real harm to one group is usually not acceptable. Often the right move is to ship with a carve-out or fix the regression first. And I'd always look for the mechanism, since a segment regression is usually telling you something true about the model.

---

## H. Subtleties

**41. Why does SUTVA matter for ad auctions?**
One advertiser's bid affects others' costs. Treatment users in an ad system can't be analyzed in isolation.

> **Saying it out loud.** Ad auctions violate SUTVA by construction, because price is set by competition. If your treatment makes some advertisers bid more, everyone else's costs go up — control advertisers are affected by treatment advertisers' assignment, which is exactly what SUTVA forbids. So you can't split advertisers into arms and compare their costs naively; you'll measure a mix of the real effect and the auction reshuffling. The usual approaches are budget-split tests, where each advertiser's budget is divided between arms, or market-level randomization over regions or time. And you'd expect the naive estimate to overstate the effect, because part of the treatment gain is just taken from control.

**42. Network effects in social platforms?**
A treatment user posts content; their control friends see it; control behavior shifts. Cluster by social graph community to limit leakage.

> **Saying it out loud.** On a social platform, treatment and control aren't separate worlds. If the treatment makes users post more, their friends in the control group see more content and behave differently, so control drifts toward treatment and your measured difference is too small — that's dilution. Sometimes it goes the other way and the effect gets amplified through the network. The fix is to randomize over graph communities rather than individuals, using clustering to keep most edges inside a cluster. It never fully works, because the graph has no clean boundaries, so you also want an estimate of how much leakage remains.

**43. Why use bootstrapping for ratio metrics (e.g., revenue per user)?**
Variance is hard to derive analytically (variance of a ratio is messy). Bootstrap is robust.

> **Saying it out loud.** Revenue per user is a ratio, and ratios don't have nice closed-form variances — the delta method gets messy and the assumptions are shaky when the denominator varies. Worse, revenue is heavily skewed, so a handful of whales dominate the mean and normal approximations behave badly. Bootstrapping sidesteps both problems: resample users, recompute the ratio, read the interval off the percentiles. The one thing to get right is resampling at the randomization unit — resample users, not transactions — otherwise you understate the variance and get intervals that are far too narrow.

**44. Two A/Bs at the same time — interaction?**
Typically OK if independently randomized (factorial design); each test reads through the noise of the other. But if treatments interact (one's effect depends on the other), you need explicit interaction analysis.

> **Saying it out loud.** Usually fine, as long as the two are randomized independently — that's a factorial design, and each experiment sees the other as noise that's balanced across its own arms. The expected effect on each estimate is zero, at a small cost in variance. It becomes a problem when treatments genuinely interact, like two features that both add banners to the same page, where the combination is worse than either alone. In that case you need to analyze the interaction term explicitly, which requires enough traffic in all four cells. Most experimentation platforms handle this with mutual exclusion groups for tests known to conflict.

**45. Shipping decision when CI = [+0.1%, +0.4%]?**
Effect is positive with high confidence, but small. Compare to deployment cost / risk. If cheap to ship, do it. If risky, may not be worth.

> **Saying it out loud.** That interval says the effect is positive with high confidence but small — somewhere between a tenth and four tenths of a percent. So the statistics have done their job and the decision is now about cost. If shipping is cheap, the code is already written, and there's no ongoing maintenance burden, take it: small compounding wins are how products improve. If it means carrying a second code path forever or taking on real operational risk, a 0.2 percent expected gain probably doesn't cover it. I'd also check that it's above whatever minimum interesting effect we set before the test, because that's the number we agreed would justify shipping.

---

## Quick fire

**46.** *Power = ?* $1 - \beta$.
**47.** *Default $\alpha$?* 0.05.
**48.** *Default power?* 0.8.
**49.** *Halve MDE → $n$ multiplies by?* 4.
**50.** *Peeking inflates which error?* Type I.
**51.** *SRM detected — what next?* Investigate; don't trust result.
**52.** *CUPED reduces what?* Variance.
**53.** *Switchback used for?* Marketplace experiments.
**54.** *IPS = ?* Inverse Propensity Scoring.
**55.** *Novelty effect direction?* Initial spike, then decay.

---

## Self-grading

If you can't answer 1-15, you can't run an A/B test. If you can't answer 16-30, you'll get fooled by your own results. If you can't answer 31-45, frontier-lab and big-tech experimentation interviews will go past you.

Aim for 35+/55 cold.
