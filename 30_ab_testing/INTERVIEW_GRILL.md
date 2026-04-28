# A/B Testing — Interview Grill

> 45 questions on experimental design, sample size, common pitfalls, and ML-specific tests. Drill until you can answer 30+ cold.

---

## A. Test design fundamentals

**1. What does randomization buy you?**
Comparability between arms in expectation. Removes confounding from observed and unobserved variables.

**2. What is the randomization unit?**
The level at which you assign treatment: user, session, request, device. Choice depends on what's independent and where you want to measure effects.

**3. Two-sided vs one-sided test — default?**
Two-sided. One-sided only if you committed *a priori* to a direction with a strong reason.

**4. Two-proportion z-test statistic?**
$z = (\hat{p}_A - \hat{p}_B) / \sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}$ with pooled $\hat{p}$.

**5. Welch's t-test vs Student's t-test — when?**
Welch when variances are different across groups (default in many libraries). Student's assumes equal variances.

**6. Mann-Whitney — why use it?**
Non-parametric. When data is skewed, ordinal, or has heavy tails. Tests stochastic dominance.

**7. Bootstrap for A/B — pros?**
Always works. No distributional assumption. Easy to extend to weird metrics (e.g., ratios).

---

## B. Sample size and power

**8. State the rule-of-thumb sample size formula for two means.**
$n \approx 2\sigma^2 (z_{\alpha/2} + z_\beta)^2 / \delta^2$. For $\alpha=0.05$, power 0.8: $n \approx 16\sigma^2/\delta^2$ per arm.

**9. Halve the MDE — what happens to $n$?**
Quadruples ($n \propto 1/\delta^2$).

**10. Define statistical power.**
$1 - \beta = \mathbb{P}(\mathrm{reject}\, H_0 \mid H_1)$. Probability of detecting a real effect of size $\delta$.

**11. What's the MDE?**
Minimum detectable effect — smallest effect you have power to detect at chosen $\alpha$ and $n$.

**12. CTR baseline 5%, want to detect 0.5pp absolute lift, 80% power, two-sided 5%. Roughly how many users per arm?**
$\sigma^2 \approx 0.05 \cdot 0.95 = 0.0475$. $\delta = 0.005$. $n \approx 16 \cdot 0.0475 / 0.000025 = 30{,}400$ per arm.

**13. Why do online experiments often need millions of users?**
Tiny effect sizes. CTR lifts on the order of 0.1pp on a 5% baseline → millions per arm.

**14. What's CUPED?**
Controlled Pre-Experiment Data: regress outcome on pre-period covariate, analyze residuals. Reduces variance ~30–50%.

**15. Stratified randomization — why?**
Reduces variance by ensuring balance on known important covariates. Like a structured form of CUPED.

---

## C. Common pitfalls

**16. What's peeking and why is it bad?**
Looking at results before the planned end and stopping when significant. Inflates Type I error toward 1 with infinite peeks.

**17. How to allow safe early stopping?**
Sequential analysis (Wald's SPRT, group sequential designs, alpha spending). Or always-valid $p$-values (mSPRT, e-values).

**18. SUTVA — what is it and what violates it?**
Stable Unit Treatment Value Assumption. Each unit's outcome doesn't depend on others' assignments. Violated by marketplaces, social platforms, capacity constraints.

**19. Cluster randomization — why?**
When SUTVA fails at the user level, randomize at a higher level (groups, geographies) to keep interference within clusters.

**20. What's a switchback test?**
For two-sided marketplace experiments, alternate treatments by time periods (e.g., one hour each) across the entire population. Eliminates network effects.

**21. SRM — what is it and how do you check?**
Sample Ratio Mismatch. Observed split doesn't match planned split. Chi-squared test on counts. If significant, randomization is broken — don't trust the test.

**22. What's the novelty effect?**
Users react to *change* itself. Effect size shifts after initial exposure. Run experiments long enough for steady state (typically 1–2+ weeks).

**23. Multiple metrics — what to do?**
Pre-register a small primary set; apply Bonferroni or BH correction across them. Treat exploratory metrics as descriptive.

**24. You see one significant metric out of 20. What do you conclude?**
Likely false positive ($\sim 1$ expected by chance). Apply correction or treat as exploratory.

---

## D. Effect-size reporting

**25. Why report effect size + CI, not just $p$-value?**
$p$-value tells you "not noise." Effect size tells you "by how much" — what actually matters for product decisions. With huge $n$, trivial effects can be significant.

**26. Cohen's $d$?**
Standardized effect: $(\bar{X}_B - \bar{X}_A)/s$. Rule of thumb: 0.2 small, 0.5 medium, 0.8 large.

**27. Absolute vs relative lift — which to report?**
Both. Absolute for low baselines (1pp lift on 1% is huge); relative for higher baselines.

**28. CI of difference is $[-0.3\%, +1.0\%]$. What can you say?**
Cannot reject null (CI includes 0). True effect is somewhere in this range with 95% confidence; could be anywhere from slightly negative to moderately positive. Decide based on minimum interesting effect.

---

## E. Bayesian A/B

**29. Bayesian A/B for two CTRs?**
Beta priors → Beta posteriors after observing data. Sample posteriors and compute $\mathbb{P}(p_B > p_A | \mathrm{data})$ by simulation.

**30. Advantages of Bayesian framing?**
Direct probability statements ("70% chance B is better"). Sequential analysis is natural. Easier business communication.

**31. Disadvantages?**
Prior choice. Stakeholders may prefer $p$-values. Computational cost for non-conjugate cases.

---

## F. ML-specific tests

**32. Position bias in ranker A/B?**
Higher positions get more clicks regardless of relevance. Naive metric like CTR doesn't isolate ranker quality.

**33. Interleaving — what is it?**
Mix items from rankers A and B on a single result page; track which side users click. More powerful per user than full A/B.

**34. Holdback test?**
Permanent (or long-running) control arm to measure long-term effects of model changes. Catches drift that short A/B misses.

**35. Online learning system A/B — what's tricky?**
Treatment arm trains on its own user behavior; control trains on its own. Models drift apart over time. Effect mixes "the new architecture" with "the new training data."

**36. Counterfactual / off-policy evaluation — when?**
When you can't safely run live A/B. Use logged data + propensity scores (IPS) or doubly robust estimators to estimate what would have happened.

**37. IPS estimator?**
$\hat{V}(\pi_{\mathrm{new}}) = \frac{1}{N}\sum_i \frac{\pi_{\mathrm{new}}(a_i|x_i)}{\pi_{\mathrm{old}}(a_i|x_i)} r_i$. Reweight rewards by policy ratio. High variance for big policy changes.

---

## G. Communication and decision

**38. You ran an A/B. Result: control 5.0% CTR, treatment 5.05%, $p$ = 0.04. Ship?**
Depends on cost of treatment, business context, secondary metrics, novelty. Significance ≠ ship-worthy. 0.05pp absolute lift may be tiny.

**39. Treatment looks great on primary metric, worse on a secondary "guardrail" metric. What do you do?**
Don't ship by default. Investigate the guardrail decline. Negative effects on user retention or engagement matter even if primary metric improves.

**40. Treatment improves overall but hurts a specific user segment. Ship?**
Depends. Equity considerations matter — sometimes you ship; sometimes you fix the segment-specific regression first.

---

## H. Subtleties

**41. Why does SUTVA matter for ad auctions?**
One advertiser's bid affects others' costs. Treatment users in an ad system can't be analyzed in isolation.

**42. Network effects in social platforms?**
A treatment user posts content; their control friends see it; control behavior shifts. Cluster by social graph community to limit leakage.

**43. Why use bootstrapping for ratio metrics (e.g., revenue per user)?**
Variance is hard to derive analytically (variance of a ratio is messy). Bootstrap is robust.

**44. Two A/Bs at the same time — interaction?**
Typically OK if independently randomized (factorial design); each test reads through the noise of the other. But if treatments interact (one's effect depends on the other), you need explicit interaction analysis.

**45. Shipping decision when CI = [+0.1%, +0.4%]?**
Effect is positive with high confidence, but small. Compare to deployment cost / risk. If cheap to ship, do it. If risky, may not be worth.

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
