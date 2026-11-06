# Regression and experiment depth

<span class="practice-page"></span>

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: understand what least-squares and experiment estimates actually establish.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f131"></a>

## F131. Pool two disjoint ordinary least-squares datasets correctly.

**Start here.** The pooled solution depends on sufficient statistics, not a simple average of coefficients.

**Work it through.**

1. For each dataset retain S_i=X_iᵀX_i and t_i=X_iᵀy_i using the same feature and intercept conventions.
2. Sum S=S1+S2 and t=t1+t2, then solve S*beta=t when well-posed, or use stable factorization-based alternatives.
3. Inspect conditioning and rank; for raw data, QR/SVD avoids unnecessary normal-equation instability.

**Small example.** One dataset with 10 rows and another with 1000 generally should not contribute equally by averaging their coefficient vectors.

**Common trap.** Averaging coefficients without their design information or ignoring different feature encodings.

**Follow-up.** What additional statistic helps estimate residual error? The summed yᵀy and sample count, under the same model convention.

---

<a id="f132"></a>

## F132. Add a feature to an existing least-squares regression efficiently.

**Start here.** Only the part of the new feature unexplained by current features can explain additional response variation.

**Work it through.**

1. Residualize the new column z against X to get z_perp, and residualize y to get y_perp.
2. If z_perp has nonzero squared norm, its incremental coefficient is z_perpᵀy_perp/(z_perpᵀz_perp).
3. Update old coefficients consistently using the projection of z onto X, or update a QR factorization directly.

**Small example.** If z is an exact copy of an existing column, z_perp=0 and it provides no new identifiable direction.

**Common trap.** Dividing by a tiny residual norm without checking conditioning.

**Follow-up.** Why is this related to partial regression? Both isolate a predictor's contribution after controlling for other columns.

---

<a id="f133"></a>

## F133. Why can regression fit independent Gaussian noise?

**Start here.** Training fit measures how well a model projects onto its fitted feature space, not whether a real predictive relationship exists.

**Work it through.**

1. Assume a fixed rank-r design with an intercept and an independent isotropic Gaussian response.
2. After removing the mean, projection onto r-1 fitted directions captures expected fraction (r-1)/(n-1) of centered response energy.
3. Thus expected training R² is (r-1)/(n-1) under these assumptions; evaluate on independent data to reveal lack of predictive signal.

**Small example.** With n=101 and r=21 including the intercept, expected training R² is 20/100=0.2 despite independence.

**Common trap.** Applying this formula to arbitrary dependent designs, no-intercept conventions or held-out R² without re-deriving assumptions.

**Follow-up.** What happens near full rank? Training interpolation can look perfect while test prediction is unstable.

---

<a id="f134"></a>

## F134. Fit a one-dimensional two-segment constant regressor.

**Start here.** After sorting the predictor, every valid split has a left mean and a right mean; prefix sums make evaluation cheap.

**Work it through.**

1. Sort (x,y) pairs and consider boundaries only between distinct x values if the model uses a threshold on x.
2. Use prefix sums of y and y² to calculate each side's SSE as sum(y²)-sum(y)²/count.
3. Choose the smallest total SSE under a stated tie rule; return a threshold and two fitted means, handling empty or unsplittable data explicitly.

**Small example.** For y=[1,1,9,9] at increasing x, the middle split yields zero SSE with means 1 and 9.

**Common trap.** Splitting equal x values into different prediction regions when a single threshold cannot reproduce that assignment.

**Follow-up.** Complexity? O(N log N) sorting, then O(N) split scanning.

---

<a id="f135"></a>

## F135. Fit a multi-target tabular regressor.

**Start here.** Shared inputs do not guarantee that every target benefits from the same representation or error scale.

**Work it through.**

1. Define per-target labels, missing-label masks, horizons and leakage-free splits.
2. Train independent baselines, then compare a joint model with a weighted sum of target losses.
3. Normalize or weight target scales deliberately and report each target's metric plus the joint business objective.

**Small example.** A target measured in dollars can dominate an unweighted squared-error objective over a second target measured in fractions.

**Common trap.** Reporting only average error while one target improves and the other becomes unusable.

**Follow-up.** When can joint training hurt? Negative transfer occurs when shared capacity or gradients conflict across tasks.

---

<a id="f136"></a>

## F136. Derive ridge regression and its effect on conditioning.

**Start here.** Ridge penalizes large weights and reduces sensitivity along weakly identified feature directions.

**Work it through.**

1. For objective ||Xw-y||²+lambda*||w||², set its gradient to zero.
2. Solve (XᵀX+lambda*I)w=Xᵀy, keeping intercept treatment explicit.
3. Use a numerical solve rather than an explicit inverse, and tune lambda using held-out data.

**Small example.** A zero singular value becomes regularized by a positive lambda, making the penalized system invertible for the penalized directions.

**Common trap.** Using a lambda from an objective averaged by N without adjusting for a summed-loss convention.

**Follow-up.** Does improved conditioning remove bias? No; ridge deliberately trades coefficient bias against variance and stability.

---

<a id="f137"></a>

## F137. Separate correlation, prediction and causation.

**Start here.** A useful predictive association need not describe the effect of intervening on a feature.

**Work it through.**

1. State whether the goal is forecasting, explanation or estimating an intervention effect.
2. List plausible confounders and timing relationships; check selection and measurement mechanisms.
3. Use randomization or a defensible causal identification strategy for causal claims, with explicit assumptions and sensitivity checks.

**Small example.** Umbrella use predicts rain but making people carry umbrellas does not cause rain.

**Common trap.** Interpreting a regression coefficient as an intervention effect solely because other recorded columns were included.

**Follow-up.** Can a noncausal predictor still be useful? Yes for prediction, subject to availability, stability and policy constraints.

---

<a id="f138"></a>

## F138. Design a switchback experiment for a shared marketplace.

**Start here.** Time-block treatment assignment can reduce simultaneous cross-treatment interference, but introduces time effects and carryover concerns.

**Work it through.**

1. Define the interacting unit, such as a region-time block, and randomize a balanced treatment schedule.
2. Choose block length and washout periods based on system response and persistence, not convenience alone.
3. Analyze at the assignment unit with time and region structure, checking carryover and enough independent blocks.

**Small example.** Alternating dispatch policies within one shared fleet by individual rider can contaminate both arms; region-time assignments may fit better.

**Common trap.** Treating thousands of rides inside one time block as thousands of independent randomized units.

**Follow-up.** What if effects persist beyond a block? Increase washout, change design or model carryover under justified assumptions.

---

<a id="f139"></a>

## F139. Interpret a confidence interval that includes zero.

**Start here.** The interval describes uncertainty under a sampling procedure; it does not prove no meaningful effect exists.

**Work it through.**

1. State the estimated effect and interval in practical units.
2. Compare the entire interval with the smallest meaningful benefit and acceptable harm, not just zero.
3. Check power, design and uncertainty assumptions before deciding to launch, gather more data or reject the change.

**Small example.** An estimated lift of 0.2 percentage points with interval [-0.1,0.5] allows both a small loss and a useful gain.

**Common trap.** Saying “not significant” means the models are equivalent or that the effect is exactly zero.

**Follow-up.** How do you establish equivalence? Use a prespecified equivalence margin and an appropriate test or interval decision rule.

---

<a id="f140"></a>

## F140. Design a credit-model business evaluation without confusing accuracy and profit.

**Start here.** The business result depends on decisions, exposure, costs and realized outcomes, not only ranking discrimination.

**Work it through.**

1. Decompose expected revenue, default loss, funding cost, operating cost and acceptance policy under explicit assumptions.
2. Compare models on time-valid cohorts and use calibration and threshold policies appropriate to the decision.
3. Assess uncertainty, delayed outcomes and relevant operational constraints; avoid presenting an unvalidated toy model as a deployment recommendation.

**Small example.** A higher approval rate can raise revenue and defaults simultaneously; the net effect depends on amounts and loss severity.

**Common trap.** Assuming a small AUC gain translates directly to the same percentage increase in profit.

**Follow-up.** Why evaluate policy and model together? Changing the threshold changes who is accepted and which outcomes become observable.

---
