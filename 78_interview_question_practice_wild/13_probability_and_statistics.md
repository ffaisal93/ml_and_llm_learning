# Probability and statistics

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: state the random variables and assumptions before calculating.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f121"></a>

## F121. Generate Gaussian noise using only uniform random numbers.

**Start here.** The Box–Muller transformation converts two independent uniforms into two independent standard Gaussian draws.

**Work it through.**

1. Draw U1 in (0,1) and U2 in [0,1), avoiding log(0).
2. Set R=sqrt(-2*log(U1)) and theta=2*pi*U2; return R*cos(theta) and R*sin(theta).
3. For mean mu and standard deviation sigma, transform each z to mu+sigma*z. Test sample moments and distribution shape rather than exact draws.

**Small example.** With U1=exp(-0.5) and U2=0, the pair is (1,0).

**Common trap.** Using the same uniform twice or treating good sample moments alone as proof of independence.

**Follow-up.** Why does the formula work? Transforming a radially symmetric Gaussian into polar coordinates yields an exponential law for squared radius.

---

<a id="f122"></a>

## F122. Find E[1/N] when N is the trial of the first six.

**Start here.** This is the expectation of a reciprocal geometric random variable, not the reciprocal of its expectation.

**Work it through.**

1. Let p=1/6 and q=1-p, so P(N=n)=p*q^(n-1).
2. Sum E[1/N]=(p/q)*sum(q^n/n) from n=1 to infinity.
3. Use sum(q^n/n)=-log(1-q)=-log(p), giving -p*log(p)/q.

**Small example.** For a fair die, E[1/N]=log(6)/5, approximately 0.35835; 1/E[N]=1/6 is different.

**Common trap.** Moving the reciprocal outside the expectation.

**Follow-up.** What is the p=1 boundary? N=1 surely, so the expectation is 1, obtained by the appropriate limit.

---

<a id="f123"></a>

## F123. Model one umbrella shared between home and work.

**Start here.** Track whether the umbrella is with the commuter before a trip; carrying it depends on rain.

**Work it through.**

1. Assume independent rain probability p for each trip and no carrying in dry weather. Define state W=with umbrella and A=away.
2. From W, rain leads to W at the next departure and dry weather to A. From A, either weather leads to W next because the destination has the umbrella.
3. For 0<p<1, stationary P(A)=(1-p)/(2-p), so wet-trip probability is p*(1-p)/(2-p).

**Small example.** At p=0.5, P(A)=1/3 and the wet-trip probability is 1/6.

**Common trap.** Assuming rain and umbrella availability are independent without modeling how past trips move the umbrella.

**Follow-up.** What changes if the commuter always carries it? Availability becomes certain after the initial condition is resolved.

---

<a id="f124"></a>

## F124. Map a random direction to a hit on a vertical screen.

**Start here.** A uniform angle does not produce a uniform hit location on a screen.

**Work it through.**

1. Specify a point distance a from the screen and a direction angle theta uniform on (-pi/2,pi/2) toward it.
2. Geometry gives signed displacement along the screen X=a*tan(theta).
3. Use change of variables to obtain f_X(x)=a/[pi*(a²+x²)], a Cauchy density. Simulate via uniform angles while avoiding endpoints.

**Small example.** For a=1, P(|X|<=1)=1/2, but the ordinary mean and variance do not exist.

**Common trap.** Using a sample mean as a stable estimator of a nonexistent population mean.

**Follow-up.** Why are extreme hits common? Angles nearly parallel to the screen create arbitrarily large displacement.

---

<a id="f125"></a>

## F125. Use Bayes' rule for a rare-event detector.

**Start here.** A high true-positive rate does not by itself imply that positive alerts are usually correct.

**Work it through.**

1. Write prevalence p, sensitivity s and false-positive rate f.
2. Compute P(event|alert)=s*p/[s*p+f*(1-p)].
3. Check operational alert volume and expected true/false counts before choosing thresholds.

**Small example.** With prevalence 1%, sensitivity 90% and false-positive rate 5%, precision is 0.009/(0.009+0.0495), about 15.4%.

**Common trap.** Confusing P(alert|event) with P(event|alert).

**Follow-up.** What if prevalence rises? Precision rises under unchanged sensitivity and false-positive rate, though real conditional rates may also change.

---

<a id="f126"></a>

## F126. Estimate a confidence interval for a paired model comparison.

**Start here.** The same test examples induce correlated errors, so preserve pairing when estimating uncertainty in their difference.

**Work it through.**

1. For each example, record the two model outcomes or a paired score difference.
2. Resample independent units as pairs, using clusters when examples share users or tasks.
3. Compute the metric difference in each replicate and summarize uncertainty; explain bootstrap assumptions and limitations for very small or rare-event samples.

**Small example.** If the same difficult examples fool both models, treating the two accuracy estimates as independent discards useful covariance information.

**Common trap.** Resampling individual events when the independent unit is a user or session.

**Follow-up.** What if the final metric is nonlinear? Recompute that metric on each paired bootstrap sample rather than averaging inappropriate per-example surrogates.

---

<a id="f127"></a>

## F127. Compute the expected minimum of uniform samples.

**Start here.** Use a survival probability to avoid an unnecessarily complicated joint density.

**Work it through.**

1. Let M=min(U1,...,Un), with independent Ui uniform on [0,1].
2. P(M>t)=(1-t)^n for t in [0,1].
3. Integrate E[M]=integral_0^1 P(M>t)dt=1/(n+1).

**Small example.** For three independent draws, the expected minimum is 1/4, even though each draw's mean is 1/2.

**Common trap.** Applying the independent-product rule when the draws are dependent.

**Follow-up.** What is the expected maximum? n/(n+1), by symmetry or its distribution function t^n.

---

<a id="f128"></a>

## F128. A customer gets two coffee cups; what is the probability both beat the other customer's two?

**Start here.** Under four independent continuous draws, all relative orderings are equally likely and there are no ties.

**Work it through.**

1. Label one customer's cups A1,A2 and the other's B1,B2, with “beat” meaning larger quality score.
2. The positions occupied by the two A cups are uniformly distributed among choose(4,2)=6 possibilities.
3. Exactly one placement puts both A cups in the top two positions, so the probability is 1/6.

**Small example.** The reverse event, both B cups beating both A cups, also has probability 1/6; the remaining 2/3 are mixed orderings.

**Common trap.** Treating four pairwise comparisons as independent.

**Follow-up.** What if the quality distribution has ties? Define tie handling and recompute; continuity was essential to the simple argument.

---

<a id="f129"></a>

## F129. Use linearity of expectation for random-graph edges.

**Start here.** You can sum expected indicator values without requiring all edge events to be independent.

**Work it through.**

1. Define I_e=1 if an allowed edge exists and zero otherwise.
2. If each of m allowed edges appears with probability p, E[number_of_edges]=sum E[I_e]=m*p.
3. For a simple undirected graph on n vertices, m=n*(n-1)/2. Dependence matters when computing variance or other nonlinear quantities.

**Small example.** With 10 vertices and p=0.2, there are 45 possible edges and an expected 9 realized edges.

**Common trap.** Demanding independence for linearity of expectation, or forgetting to divide by two for undirected pairs.

**Follow-up.** How would degree expectations work? Sum the n-1 incident edge indicators, giving (n-1)*p under equal marginals.

---

<a id="f130"></a>

## F130. Design a Monte Carlo estimate and quantify its error.

**Start here.** An estimate is useful only with its sampling assumptions, uncertainty and computational cost.

**Work it through.**

1. Generate independent draws under the stated model and average the quantity of interest.
2. Estimate standard error as sample standard deviation divided by sqrt(number_of_draws), when finite-variance conditions apply.
3. Use confidence intervals or suitable alternatives, validate against analytic special cases and assess model error separately from simulation error.

**Small example.** Reducing Monte Carlo standard error by a factor of two generally requires four times as many independent samples.

**Common trap.** Reporting a tiny simulation standard error as proof that the underlying financial or scientific model is correct.

**Follow-up.** What if the distribution has infinite variance? Ordinary square-root-N error reasoning may fail.

---
