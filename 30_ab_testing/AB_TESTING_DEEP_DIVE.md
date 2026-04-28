# A/B Testing — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

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

---

## 2. Hypothesis testing for A/B

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

---

## 3. Sample size and power

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

---

## 4. Common pitfalls

### Peeking / sequential testing

Looking at results before the planned end and stopping when "significant" inflates Type I error dramatically. With repeated peeks, the chance of *eventually* seeing $p < 0.05$ approaches 1.

**Fix**: use sequential / always-valid statistics (Howard et al. 2021), or commit to a fixed sample size and don't peek, or apply alpha-spending.

### Multiple testing across metrics

Run 20 metrics, one comes back significant: probably noise.

**Fix**: pre-register a small set of primary metrics, apply Bonferroni or BH for secondary, treat exploratory metrics as descriptive only.

### SUTVA violations (network effects)

Stable Unit Treatment Value Assumption: one user's outcome doesn't depend on another user's treatment.

Violated by:
- Marketplaces: a user's bid affects others' prices.
- Social platforms: treatment user's posts affect control users' feeds.
- Promotions: treatment exhausts inventory available to control.

**Fix**: cluster-randomization (whole groups assigned to one arm), geo experiments, switchback tests.

### Sample ratio mismatch (SRM)

Random assignment should give $n_A : n_B$ matching your planned ratio. If observed split deviates significantly (chi-squared test), randomization is broken or there's a logging bug. Don't trust results.

### Novelty / primacy effects

Users react to *change*, not the steady-state experience. Wait for the effect to stabilize (1–2 weeks for product changes).

### Selection bias / opt-in cohorts

If the population in the test isn't representative of the deployment population, results don't generalize. Don't run on power users only.

---

## 5. Effect-size metrics

Beyond $p$-values:

**Absolute lift**: $\hat{p}_B - \hat{p}_A$.

**Relative lift**: $(\hat{p}_B - \hat{p}_A)/\hat{p}_A$.

**Cohen's $d$**: standardized effect size $(\bar{X}_B - \bar{X}_A)/s$.

Always report effect size + CI. A $p < 0.001$ tells you "not noise" but not "how much it matters." With huge $n$, trivial effects can be highly significant.

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

---

## 7. ML-specific A/B testing

### Recommender / ranker tests

Outcome metrics like CTR, dwell time, retention. Issues:
- **Position bias**: users click higher-ranked items more.
- **Long-term effects**: short-term CTR ≠ long-term satisfaction.
- **Holdback experiments**: keep a small population on the old model permanently to measure long-term drift.

### Online learning systems

If the model trains on user behavior, the test arm influences future training data. This can cause drift between treatment and control models.

### Counterfactual evaluation

Sometimes you don't want to ship A/B for risk reasons. Instead, **off-policy evaluation**: estimate what would have happened under the new policy from logged data of the old policy. Methods: importance sampling (IPS), doubly robust estimators.

### Interleaving for ranker comparison

Instead of A/B-ing entire pages, interleave items from A and B in the same page; measure which side users click more. More statistically powerful per user but harder to set up.

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

---

## 9. Eight most-asked interview questions

1. **Walk me through how you'd power an A/B test.** (MDE, baseline rate, $\alpha$, $\beta$, derive $n$.)
2. **You ran 20 metrics, two are significant at 0.05. What do you conclude?** (Multiple testing — apply correction; pre-register primary metrics.)
3. **What's CUPED and why use it?** (Variance reduction via pre-experiment covariate; ~30–50% more power for free.)
4. **What goes wrong with network effects?** (SUTVA violation; treatment leaks to control via shared resources.)
5. **You see significance after 3 days. Stop the test?** (Peeking inflates Type I error; commit to $n$ or use sequential methods.)
6. **CI overlaps zero — null result. Anything else to report?** (Effect size + CI to bound the maximum plausible effect; "no significant effect with 95% CI [-0.3, 0.5]" is way more informative.)
7. **Bayesian vs frequentist A/B testing — pros/cons?** (Direct probability vs $p$-value; sequential properties.)
8. **You run an experiment, see SRM. What do you do?** (Don't trust the result; investigate randomization/logging; rerun.)

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
