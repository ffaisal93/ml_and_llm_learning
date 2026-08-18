# Frontier Intuitive Probability / Statistics — Interview Grill

> 100+ active-recall questions calibrated for OpenAI / DeepMind / Anthropic research-scientist rounds. Each is a 60-second oral exam answer.
> Pair with `INTUITIVE_QUESTIONS_DEEP_DIVE.md`.

---

## Section A — Framing checklist (Q1–7)

1. List the 7 framing checklist items in order.

   > **Saying it out loud.** Variables, hypotheses, problem type, prior or not, loss function, sample size, and what's computable versus conceptual. Said as a sentence: define the random variables precisely, name the competing hypotheses, decide whether this is classification, estimation or a decision, say whether I have a prior, state the loss, check how much data there is, and flag which parts I'd compute numerically versus argue from principle. The reason to say it aloud rather than think it is that it buys time and it demonstrates the thing being tested, which is framing rather than arithmetic.

2. When does a problem call for Bayesian vs frequentist framing?

   > **Saying it out loud.** Bayesian when I have a genuine prior worth using — small sample, domain knowledge, or a sequential setting where today's posterior is tomorrow's prior. Frequentist when the prior would be made up, when I need a guarantee that doesn't depend on someone's belief, or when I'm reporting to people who'll object to the prior. The honest answer is that with a lot of data they agree, so the choice only matters when data is thin — which is exactly when the prior is doing real work and therefore when you have to defend it. I'd rather state both and say which I'd act on.

3. When is a problem a classification vs estimation vs decision?

   > **Saying it out loud.** Classification when the answer is one of a discrete set of hypotheses — use Bayes and the likelihood ratio. Estimation when the answer is a number or parameter — MLE, MAP, or a posterior. Decision when there's an action with consequences attached — minimise expected loss, which is a superset of the other two. The useful observation is that classification and estimation are both decision problems with particular loss functions bolted on, so if the question mentions costs or asymmetric consequences at all, go straight to expected loss and don't detour through a point estimate.

4. Why is it important to state the loss function before computing?

   > **Saying it out loud.** Because the loss determines the answer, not just the presentation. Under zero-one loss the optimal estimate is the posterior mode; under squared error it's the posterior mean; under absolute error it's the median — three different numbers from the same posterior. And under asymmetric loss the decision threshold moves away from a half, sometimes a long way. So a question like "which distribution did it come from" is genuinely ambiguous until someone says what it costs to be wrong in each direction, and asking that is a better move than guessing.

5. What's the difference between MAP and posterior mean as point estimates? Under what loss is each optimal?

   > **Saying it out loud.** MAP is the mode of the posterior, optimal under zero-one loss. Posterior mean is the expectation, optimal under squared-error loss. They coincide when the posterior is symmetric and unimodal, and can be far apart when it's skewed or multimodal — for a strongly skewed posterior the mode can be in a region carrying very little total probability. There's also a technical wrinkle worth mentioning: for a continuous parameter, zero-one loss is degenerate since any single point has probability zero, so MAP is really the limit of a small-interval loss rather than exactly optimal.

6. What does "asymptotic" mean and when do you reach for it?

   > **Saying it out loud.** Asymptotic means "in the limit as the sample size goes to infinity", and it's what licenses most of the standard toolkit — MLE consistency and normality, the CLT, Wald intervals. I reach for it when n is comfortably large relative to the number of parameters, and I stop trusting it when n is small, when the parameter is near a boundary, or when the distribution has heavy tails so convergence is slow. The honest framing is that asymptotic results are approximations whose error you usually can't quantify without more work — Berry-Esseen gives you a rate, but in practice at n equals ten you should say "this is a rough guide" rather than quote a confidence level.

7. State a single-sentence summary of the framing approach.

   > **Saying it out loud.** One sentence: frame the scenario as a well-posed inference problem — variables, hypotheses, prior, loss — name the framework you're using out loud, compute or bound what you can, and be explicit about which assumptions the answer rests on and where it stops being reliable.


## Section B — Bayesian classification (Q8–18)

8. Sketch Bayes' rule with priors and likelihoods.

   > **Saying it out loud.** Posterior over a hypothesis is proportional to likelihood times prior, normalised by summing that product over all hypotheses. Said as odds, which is the version I'd actually use: posterior odds equal prior odds times the likelihood ratio. That form is better for talking because multiplication becomes addition in logs, so evidence just accumulates, and because the normalising constant disappears.

9. Define the likelihood ratio and the prior odds.

   > **Saying it out loud.** The likelihood ratio is the probability of the observed data under hypothesis one divided by its probability under hypothesis two — how much this observation favours one over the other. The prior odds is the ratio of the two prior probabilities — how much you favoured one before looking. The decision rule is just comparing them: believe hypothesis one when the evidence ratio beats the prior odds against it. Everything in Bayesian classification is that one comparison, dressed differently.

10. State the Neyman-Pearson lemma.

    > **Saying it out loud.** Neyman-Pearson: among all tests with false-positive rate at most alpha, the likelihood ratio test has the highest power — that is, the lowest false-negative rate. So thresholding the likelihood ratio isn't a heuristic, it's provably the best you can do. The conditions matter: it's for testing two *simple* hypotheses, both fully specified. Once either side is composite — "the mean is greater than zero" — you need generalised likelihood ratio tests and the guarantee weakens.

11. Why is the LRT optimal at fixed false-positive rate?

    > **Saying it out loud.** The intuition is a knapsack argument. You have a budget of false positives, and you want to spend it on the regions of the sample space that buy the most power per unit of budget spent. The likelihood ratio is exactly the "power per unit false-positive cost" of including a region in your rejection set, so ranking by it and taking greedily until the budget is exhausted is optimal. That's the whole proof, and it's worth being able to give in that form because it shows you understand it rather than remembering it.

12. For i.i.d. samples, how does the log-likelihood ratio scale with $n$?

    > **Saying it out loud.** It's a sum of i.i.d. terms, so its mean grows linearly in n at rate KL, and its standard deviation grows like the square root of n. That means the signal-to-noise ratio grows like the square root of n, which is why discriminability improves but slowly, and why the whole thing behaves like a random walk with drift. Under the other hypothesis the drift reverses sign — with rate the reverse KL, which is a different number.

13. What's the expected log-likelihood ratio under $H_1$? (Hint: it's a familiar quantity.)

    > **Saying it out loud.** It's the KL divergence from P to Q. That's not a coincidence but a definition unpacked: KL is defined as the expected log ratio, so "average evidence per sample" and "KL divergence" are literally the same quantity. The practical reading is that KL is measured in nats of evidence per observation, which is the single most useful way to hold it in your head.

14. State the sample complexity formula for distinguishability.

    > **Saying it out loud.** The clean statement is Chernoff–Stein: at a fixed type-one error, the type-two error decays like e to the minus n times KL, so the number of samples you need is about the log of one over your target error rate, divided by KL. So it's linear in one over KL. The Gaussian special case is the sanity check — separating two means costs samples proportional to variance over squared mean gap, and KL there is squared mean gap over twice the variance, so the two agree.

15. Why is sample complexity $O(1/\mathrm{KL}^2)$ rather than $O(1/\mathrm{KL})$?

    > **Saying it out loud.** I'd push back on the premise, politely. It isn't one over KL squared — it's one over KL. Chernoff–Stein gives error decaying as e to the minus n KL, so samples scale linearly in one over KL. The place the squared version comes from is a normal-approximation power formula with variance over squared mean-gap, and people forget that for close distributions the variance of the log-likelihood ratio is itself about twice KL, so the ratio collapses back to one over KL. The fastest check is the Gaussian case: n proportional to variance over squared mean gap, and KL is squared mean gap over twice the variance, so n is proportional to one over KL. If a remembered rate contradicts a worked example, trust the example.

16. What's Chernoff information? How is it related to Bayes error rate?

    > **Saying it out loud.** Chernoff information is the exponent governing how fast the optimal Bayes error decays with sample size — it's the negative log of the minimum over lambda between zero and one of the integral of p to the lambda times q to the one minus lambda. Bayes error decays like e to the minus n times that. The relationship to KL is that Chernoff is the tight exponent and KL is a bound on it, so quoting KL is optimistic. The intuition for the lambda optimisation is that you're finding the tilted distribution sitting between P and Q where the two error types trade off best.

17. Walk through the Bayes error rate formula $\int \min(p, q)$.

    > **Saying it out loud.** The Bayes error rate with equal priors is the integral of the pointwise minimum of the two densities, and the picture is the overlapping area under two curves. The reason it's a minimum is that at every point you predict whichever hypothesis is more likely there, so your error at that point is the smaller of the two densities. What makes it worth quoting in an interview is that it's a floor — no classifier, no amount of data, no cleverness gets below it. So if someone asks how to improve accuracy and the overlap is large, the honest answer is "measure something else", not "try a better model".

18. Two-class question — under asymmetric loss, how does the threshold shift?

    > **Saying it out loud.** You shift the threshold by the ratio of the costs. Under zero-one loss you compare the likelihood ratio to the prior odds; under asymmetric loss you compare it to the prior odds times the ratio of the two misclassification costs. So if a false negative costs ten times a false positive, the threshold moves by a factor of ten and you accept many more false positives to avoid the expensive error. The general statement is that you're minimising expected loss, and the threshold is wherever the two expected costs cross — which is also exactly what moving along an ROC curve means.


## Section C — MLE, MAP, method of moments (Q19–27)

19. State the MLE objective.

    > **Saying it out loud.** Maximise the likelihood of the observed data over the parameter — in practice, maximise the sum of log densities, since logs turn the product into a sum and avoid underflow. The framing worth adding is that maximising log-likelihood is equivalent to minimising the KL divergence from the empirical distribution to your model family, which is why MLE and cross-entropy training are the same procedure.

20. Three asymptotic properties of MLE.

    > **Saying it out loud.** Consistency — it converges to the true parameter as n grows. Asymptotic normality — the scaled error converges to a Gaussian with covariance the inverse Fisher information. And asymptotic efficiency — that variance attains the Cramér-Rao lower bound, so nothing does better in the limit. All three are conditional on regularity assumptions and on the model family containing the truth, which is the qualification I'd say out loud rather than leave implied.

21. What's Fisher information?

    > **Saying it out loud.** Fisher information is the expected squared derivative of the log-likelihood with respect to the parameter, or equivalently minus the expected second derivative — the curvature of the log-likelihood at the truth. The intuition is that sharp curvature means the likelihood falls away fast as you move off the true parameter, so the data pins the parameter down tightly. High information means low achievable variance, and it's additive over independent samples, which is why variance scales like one over n.

22. What's the Cramér-Rao lower bound?

    > **Saying it out loud.** The Cramér-Rao bound says any unbiased estimator's variance is at least the inverse of the Fisher information. So it's a floor on how well you can possibly estimate, given the model. Two caveats worth giving: it applies to unbiased estimators, and biased estimators can beat it in mean squared error — James-Stein being the famous case — and it requires regularity conditions that fail when the support depends on the parameter, like the uniform on zero to theta.

23. When is MLE biased in finite samples? Give an example.

    > **Saying it out loud.** Routinely — MLE is only asymptotically unbiased. The standard example is the Gaussian variance: the MLE divides by n, but the unbiased estimator divides by n minus one, because you spent a degree of freedom estimating the mean. The other classic is the uniform on zero to theta, where the MLE is the sample maximum, which is biased low by construction since you can't observe above the truth. The general reason is that MLE is invariant under reparameterisation and unbiasedness isn't, so the two properties are fundamentally in tension.

24. Compare MLE with MAP — when are they the same?

    > **Saying it out loud.** MAP maximises likelihood times prior, so it reduces exactly to MLE when the prior is uniform over the parameter space — or more loosely, whenever the likelihood dominates, which means large n. The relationship worth naming is that a Gaussian prior on parameters makes MAP identical to L2-regularised MLE, and a Laplace prior gives you L1 — so weight decay is a MAP estimate with a Gaussian prior, which is the connection interviewers like.

25. Why is MAP not the optimal Bayesian estimator under squared loss?

    > **Saying it out loud.** Because under squared-error loss the optimal Bayesian estimator is the posterior *mean*, and MAP is the *mode*. Those coincide only for symmetric unimodal posteriors. For a skewed posterior the mode can sit in a narrow spike carrying little probability mass while the bulk of the distribution is elsewhere, and the mean tracks the bulk. There's also the reparameterisation issue: the mode isn't invariant to a change of variables, since the density picks up a Jacobian, whereas the mean of a well-defined quantity is.

26. When does method of moments beat MLE?

    > **Saying it out loud.** When the likelihood is intractable or expensive but the moments have closed forms — that's the main case, and it's why method of moments survives in things like fitting heavy-tailed distributions or in generalised-method-of-moments econometrics. It's also sometimes more robust, since it only depends on a few summary statistics rather than the full likelihood shape, and it's a good initialiser for an iterative MLE. The cost is efficiency: it generally has higher asymptotic variance than MLE, so you're trading statistical efficiency for computational or robustness reasons, and I'd say which one I'm buying.

27. What does "asymptotically efficient" mean?

    > **Saying it out loud.** It means the estimator's asymptotic variance attains the Cramér-Rao lower bound — as n grows, no other consistent estimator does better. It's a limiting statement, so it says nothing about finite samples, and an asymptotically efficient estimator can be badly behaved at small n. It's also conditional on the model being correctly specified: under misspecification MLE converges to the parameter minimising KL to the truth, which is a well-defined thing but not the truth, and the efficiency guarantee no longer means what you want it to.


## Section D — Concentration and tail bounds (Q28–36)

28. State Markov's inequality.

    > **Saying it out loud.** For a non-negative random variable, the probability of exceeding a is at most the mean divided by a. It's the weakest useful bound and it assumes almost nothing, which is exactly why it's the building block for everything else — Chebyshev is Markov applied to the squared deviation, and Chernoff is Markov applied to the exponential moment. Worth noting it's tight in the worst case, so it can't be improved without more assumptions.

29. State Chebyshev's inequality.

    > **Saying it out loud.** The probability of being at least k standard deviations from the mean is at most one over k squared. It needs only that the variance exists, no distributional shape at all, which is its whole appeal. The price is looseness — at two sigma it gives twenty-five percent where a Gaussian gives five — so it's for guarantees rather than for reporting intervals.

30. State Hoeffding's inequality (precise form).

    > **Saying it out loud.** For n independent variables bounded in an interval of width b minus a, the probability that the sample mean deviates from its expectation by at least t is at most two times the exponential of minus two n t squared over the width squared. Two things to be careful about out loud: the two in the numerator of the exponent, which is the form to memorise, and that it needs *bounded* variables, not just finite variance. It's the workhorse because it gives a sub-Gaussian tail with essentially no assumptions beyond boundedness.

31. When does Hoeffding apply but Bernstein doesn't?

    > **Saying it out loud.** Hoeffding needs boundedness and nothing else, so it works when you have no idea what the variance is — which is common when you only know a range. Bernstein needs a variance bound as well. So the case where Hoeffding applies and Bernstein doesn't is precisely when the variables are bounded but you have no usable variance estimate. Practically, if you can bound the variance you should, because the gain when variance is small relative to range is large.

32. When does Bernstein give a sharper bound than Hoeffding?

    > **Saying it out loud.** When the variance is much smaller than the range. Hoeffding only knows the range, so it implicitly assumes worst-case variance — variables sitting at the endpoints. Bernstein uses the actual variance and interpolates between a Gaussian tail for small deviations and an exponential tail for large ones. The concrete case is rare events: a Bernoulli with p equal to a thousandth has range one but variance about a thousandth, so Hoeffding is off by orders of magnitude and Bernstein is nearly tight.

33. What's the moment generating function and why does it matter for Chernoff?

    > **Saying it out loud.** The moment generating function is the expectation of e to the t X, and it encodes all the moments in one object. It matters because the Chernoff method is: apply Markov's inequality to the exponentiated variable, which turns a tail probability into an MGF bound, then optimise over t to make the bound as tight as possible. That optimisation is what gives exponentially decaying tails rather than polynomial ones. The catch is that heavy-tailed distributions have no finite MGF anywhere, so the whole approach collapses — which is exactly why heavy tails are hard.

34. When does CLT apply?

    > **Saying it out loud.** For i.i.d. variables with finite variance, the scaled sample mean converges in distribution to a Gaussian, regardless of the underlying shape. The conditions people forget: it needs finite variance, so it fails for Cauchy or a Pareto with a small exponent, and it's a statement about convergence in distribution, not about the tails at any finite n. Practically it kicks in around twenty or thirty samples for well-behaved distributions and much later for skewed or heavy-tailed ones — and the tails converge slowest, which is unfortunate since tails are usually what you care about.

35. CLT — what's the rate of convergence (Berry-Esseen)?

    > **Saying it out loud.** Berry-Esseen: the maximum difference between the true CDF of the standardised mean and the normal CDF is bounded by a constant times the third absolute central moment over the cube of the standard deviation, all divided by the square root of n. So the error shrinks like one over root n. The useful reading is that convergence is slow, and the constant depends on skewness — heavily skewed distributions need much larger n before the normal approximation is honest. This is the quantitative answer to "is thirty samples enough", and the answer is "it depends on the third moment".

36. For binary outcomes with $n=200$, $p=0.5$, what's the 95% CI half-width?

    > **Saying it out loud.** Standard error is the square root of p times one minus p over n, which is the square root of 0.25 over 200, about 0.035. Times 1.96 gives a half-width of roughly seven percentage points. The rule of thumb worth carrying is that for a proportion near a half, the ninety-five percent half-width is about one over the square root of n — one over root two hundred is about 0.07, matching. That rule makes you fast at sanity-checking survey and A/B numbers in your head.


## Section E — KL divergence and information theory (Q37–46)

37. Define KL divergence and state two key properties.

    > **Saying it out loud.** KL from P to Q is the expectation under P of the log ratio of P to Q. Two key properties: it's non-negative and zero only when the distributions are equal — that's Gibbs' inequality — and it's asymmetric, so it's not a metric and it doesn't satisfy the triangle inequality. A third worth having: it's infinite whenever P puts mass where Q puts none, which is why it behaves badly for distributions on disjoint supports.

38. Why is KL asymmetric?

    > **Saying it out loud.** Because the expectation is taken under one distribution and not the other, so the two directions weight disagreements in different places. KL from P to Q penalises Q for being small where P has mass; KL from Q to P penalises P for being small where Q has mass. That asymmetry is the content, not a defect: it's why forward KL is mean-seeking and covers all the modes, while reverse KL is mode-seeking and collapses onto one — the difference between a maximum-likelihood fit and a variational approximation.

39. Sketch KL between two univariate Gaussians with same variance.

    > **Saying it out loud.** With shared variance it's the squared difference of the means divided by twice the variance. So it's essentially the squared effect size, or half the squared z-distance between them. That's the single most useful KL fact to have memorised, because it turns any Gaussian separation question into an evidence-per-sample number instantly, and it's the sanity check that catches wrong sample-complexity rates.

40. Why does KL matter for distinguishability?

    > **Saying it out loud.** Because KL is the expected log-likelihood ratio, which is exactly the evidence one sample provides on average. So samples needed is roughly one over KL, times a factor for how confident you want to be — that's Chernoff–Stein. Two distributions with large KL separate in a handful of samples; two with KL near zero need enormous numbers no matter what test you use, because the information simply isn't there per observation. It converts "how different are these" into "how many samples", which is the reason it keeps showing up.

41. State the relationship between KL and Bayes error exponent.

    > **Saying it out loud.** The Bayes error decays exponentially in n with exponent the Chernoff information, and KL upper-bounds that exponent. In the one-sided setting where you fix the type-one error, Chernoff–Stein gives the type-two exponent as exactly KL. So the relationship is: KL is the exponent for asymmetric testing, Chernoff information is the exponent for symmetric Bayes error, and Chernoff is at most KL. Quoting KL for the symmetric case is optimistic, which is worth flagging rather than glossing.

42. What's the Fano inequality?

    > **Saying it out loud.** Fano's inequality lower-bounds the probability of error in guessing a random variable from an observation, in terms of the conditional entropy — roughly, error probability is at least the conditional entropy minus one, over the log of the alphabet size. It's the standard tool for proving that some estimation problem is impossible below a certain sample size: you show the mutual information is small, so the conditional entropy is large, so any method must err. It's the information-theoretic converse to a concentration bound — bounds say what you can do, Fano says what nobody can.

43. What's mutual information in one sentence?

    > **Saying it out loud.** Mutual information is how much knowing one variable reduces your uncertainty about the other — formally, the KL divergence between the joint distribution and the product of the marginals. It's zero exactly when they're independent, it's symmetric unlike KL itself, and it's measured in bits or nats of shared information.

44. Why is KL the "natural" loss in maximum-likelihood / VAE / diffusion?

    > **Saying it out loud.** Because maximising likelihood is the same thing as minimising KL from the empirical data distribution to the model. Once you write down "make the model assign high probability to the data", you've written down cross-entropy, which is KL plus the data's own entropy — and that entropy doesn't depend on the parameters, so minimising one minimises the other. Diffusion and VAEs get there through the ELBO, which is a KL-based bound on the log evidence. The unifying statement: KL isn't a design choice in these methods, it's what "fit the data" means once you commit to probabilistic modelling.

45. Why is reverse-KL different from forward-KL in posterior approximation?

    > **Saying it out loud.** Reverse KL — expectation under the approximation — punishes putting mass where the true posterior has none, so the approximation retreats to a single mode and underestimates uncertainty. That's what standard variational inference optimises, and it's why VI is famously overconfident. Forward KL punishes missing mass where the posterior has some, so it spreads out and covers all the modes, at the cost of putting mass in the gaps between them. Neither is right in general; the choice depends on whether the cost of missing a mode exceeds the cost of hallucinating one, and I'd say which failure I'd rather have for the problem at hand.

46. KL as coding excess — explain.

    > **Saying it out loud.** If the data really comes from P but you built your code optimally for Q, KL from P to Q is the expected number of extra bits per symbol you pay. That's the cleanest intuition for why it's non-negative — a mismatched code can never beat the matched one — and for why it's asymmetric, since the penalty depends on which distribution is generating the symbols you're paying for. It also explains the infinity case: if P generates a symbol Q assigned zero probability, your code has no codeword for it and the cost is unbounded.


## Section F — Sequential decision / bandits (Q47–53)

47. Define the multi-armed bandit problem.

    > **Saying it out loud.** K arms with unknown reward distributions, you pull one per round, observe only that arm's reward, and want to maximise total reward — equivalently minimise regret, which is the cumulative gap between always pulling the best arm and what you actually got. The defining feature is partial feedback: you only learn about what you tried, so information and reward come from the same action. That's the exploration-exploitation tension in its purest form, which is why it's the standard model for it.

48. What's UCB and what regret does it achieve?

    > **Saying it out loud.** UCB picks the arm with the highest upper confidence bound — the empirical mean plus a term that grows with the log of time and shrinks with the number of pulls of that arm. It's optimism in the face of uncertainty: an arm is attractive either because it looks good or because you haven't tried it much. It achieves logarithmic regret in the horizon, which is optimal up to constants by the Lai-Robbins lower bound. Worth naming that the constant depends on the gaps between arms — closely-matched arms are expensive to separate, which is the same KL story again.

49. What's Thompson sampling?

    > **Saying it out loud.** Thompson sampling maintains a posterior over each arm's reward, draws one sample from each arm's posterior, and pulls the argmax of the samples. So exploration comes for free from posterior uncertainty rather than from an explicit bonus. It's older than UCB, achieves the same optimal logarithmic regret, and in practice usually beats UCB empirically while being simpler to implement. The cost is that you need a posterior, so you need a likelihood model — which is easy for Bernoulli or Gaussian arms and awkward otherwise.

50. Why doesn't $\epsilon$-greedy achieve $O(\log T)$ regret in general?

    > **Saying it out loud.** Because with a fixed epsilon you keep exploring at a constant rate forever, so you pull suboptimal arms a constant fraction of the time and regret grows linearly rather than logarithmically. You can fix it by decaying epsilon like one over t, which does recover logarithmic regret, but you have to tune the decay rate to the gaps, which you don't know. The deeper problem is that epsilon-greedy explores uniformly at random — it spends as much on an arm it's already ruled out as on a promising one — whereas UCB and Thompson direct exploration toward arms that could plausibly still be best.

51. Distinguish regret minimization from best-arm identification.

    > **Saying it out loud.** Regret minimisation is about total reward earned while learning, so you must exploit as you go and you never fully stop caring about the cost of a bad pull. Best-arm identification is pure exploration: you have a budget or a confidence target, the pulls during the experiment are free, and you only care about naming the right arm at the end. They're genuinely different objectives with different optimal algorithms — UCB and Thompson for the first, LUCB and Track-and-Stop for the second — and an algorithm optimal for one is provably not optimal for the other. In practice the distinction is whether you're running a live system or an offline experiment.

52. What's the Track-and-Stop algorithm for?

    > **Saying it out loud.** Track-and-Stop is for fixed-confidence best-arm identification: identify the best arm with probability at least one minus delta, using as few samples as possible. It works by computing the optimal sampling proportions from the current estimates — which arms deserve what fraction of your budget, derived from the information-theoretic lower bound — then tracking those proportions, and stopping when a generalised likelihood ratio statistic crosses a threshold. It's notable for being asymptotically optimal, matching the lower bound rather than just achieving the right rate.

53. How does bandit theory connect to RLHF?

    > **Saying it out loud.** Several ways. Best-of-N sampling with a reward model is literally a pure-exploration bandit over candidate responses. Exploration-exploitation is the core tension in RL fine-tuning — the KL penalty is a device for controlling how far you explore from a known-good policy. Online preference collection is a bandit problem: which comparisons to ask humans for, given a labelling budget, is best-arm identification. And bandits are the stateless special case of RL generally, so the regret and exploration intuitions carry directly.


## Section G — Importance and rejection sampling (Q54–58)

54. State the importance-sampling identity.

    > **Saying it out loud.** The expectation of f under P equals the expectation under Q of f times the ratio of P to Q. So you sample from the easy distribution and reweight by the density ratio. The only requirement is that Q covers the support of P wherever f is non-zero — if Q assigns zero density somewhere P doesn't, the identity breaks and the estimator is biased in a way no amount of sampling reveals.

55. When does importance sampling have high variance?

    > **Saying it out loud.** When the ratio of P to Q has heavy tails — meaning Q is much smaller than P somewhere that matters. Then a rare sample gets an enormous weight and dominates the whole estimate, so the variance is huge and possibly infinite. The vicious part is that it looks fine until it doesn't: you get a stable-looking estimate for a thousand samples and then one draw moves it by an order of magnitude. The practical diagnostic is effective sample size — if a few weights carry most of the total, your effective n is tiny regardless of how many samples you drew.

56. Why does importance sampling appear in PPO?

    > **Saying it out loud.** Because PPO reuses data collected under an older policy to update the current one, which is exactly an off-policy correction. The ratio of the new policy's probability to the old policy's probability for the same action is the importance weight, and the objective is the advantage times that ratio. The clipping exists precisely because of the high-variance failure mode — when the ratio gets large, the estimator becomes unreliable, so PPO refuses to take credit for improvements beyond the clip range. So PPO's central trick is a variance-control device for importance sampling, which is a nice thing to be able to say.

57. Walk through rejection sampling.

    > **Saying it out loud.** Find a constant M such that M times the proposal density dominates the target everywhere. Sample a point from the proposal, then accept it with probability equal to the target density over M times the proposal density at that point; otherwise discard and try again. What survives is distributed exactly according to the target — it's exact, not approximate, which is its advantage over MCMC. The costs are that you need the dominating constant, and you throw work away.

58. When is rejection sampling impractical (acceptance rate)?

    > **Saying it out loud.** When M is large, because the acceptance rate is exactly one over M — so a mismatch between proposal and target costs you directly in wasted samples. And M grows brutally with dimension: even a modest per-dimension mismatch compounds exponentially, so in high dimensions the acceptance rate becomes astronomically small and rejection sampling is unusable. That's the general lesson worth stating — rejection sampling is exact and elegant in low dimensions and hopeless in high ones, which is why MCMC exists despite being approximate.


## Section H — Stein and shrinkage (Q59–62)

59. State the James-Stein result.

    > **Saying it out loud.** For three or more independent Gaussian means estimated from one observation each, the James-Stein estimator — shrink every observation toward the origin by a factor depending on the total squared magnitude — has strictly lower total mean squared error than using the observations themselves, for every possible set of true means. So the obvious estimator is inadmissible in three or more dimensions. It fails for one or two dimensions, which is itself striking — the dimension threshold is real, not an artefact.

60. Why is James-Stein "paradoxical"?

    > **Saying it out loud.** Because the means can be completely unrelated — the price of tea, the mass of a proton, and a batting average — and you still do better by letting each estimate be influenced by the others. That seems to violate the idea that independent problems should be solved independently. The resolution is that the guarantee is about *total* squared error summed across coordinates, not about any single one — you can make an individual estimate worse while improving the sum. And the geometric intuition is that in high dimensions noise almost always makes the observation vector longer than the true mean vector, so shrinking corrects a systematic overshoot.

61. How does shrinkage relate to Bayesian priors?

    > **Saying it out loud.** Shrinkage is what a prior does. A Gaussian prior centred at zero produces a posterior mean that is the observation scaled down toward zero, which is exactly the shape of James-Stein — and empirical Bayes makes the connection precise, since James-Stein is very close to estimating the prior's variance from the data itself and then applying the Bayes rule. So the paradox becomes less paradoxical: you're borrowing strength across coordinates to estimate the shared prior, which is a legitimate thing to do even when the parameters themselves are unrelated.

62. How does this connect to weight decay in deep learning?

    > **Saying it out loud.** Weight decay is L2 regularisation, which is MAP estimation with a zero-centred Gaussian prior on the weights, which is shrinkage. So the same mathematics: you accept some bias to buy a larger reduction in variance, and in high dimensions that trade is almost always favourable — which is the deep learning version of the James-Stein dimension condition. Dropout, early stopping, and small initialisations are all doing something similar by different means. The framing that scores is that regularisation isn't a hack for finite data, it's the estimation-theoretically correct thing to do in high dimensions.


## Section I — The two-distribution scenario, fully drilled (Q63–75)

63. State the question in one sentence.

    > **Saying it out loud.** You have two arrays of numbers, each drawn from a different unknown distribution; a new number arrives; decide which distribution it came from — and say how confident you are.

64. What's the Bayes-optimal decision rule?

    > **Saying it out loud.** Under zero-one loss, classify to P when the likelihood ratio — the density of x under P over its density under Q — exceeds the ratio of the prior probabilities the other way round. Priors are typically estimated from the array sizes if the arrays are proportional to base rates, and I'd say that assumption out loud because it often isn't true. This is the Neyman-Pearson optimal rule, so the decision rule isn't the hard part.

65. Three approaches to estimating $p(x)$ and $q(x)$ from arrays.

    > **Saying it out loud.** Parametric — assume a family, say Gaussian, fit by MLE to each array, and evaluate the densities in closed form. Non-parametric — kernel density estimation on each array, with the bandwidth chosen by cross-validation. Discriminative — skip the densities entirely, label the arrays as two classes, train a classifier, and use its predicted probability, which is a direct estimate of the quantity you actually need. I'd add that the third is often the best answer precisely because you only need the *ratio*, and estimating a ratio is easier than estimating two things and dividing.

66. Tradeoff between parametric (Gaussian) vs KDE.

    > **Saying it out loud.** Parametric is low variance and fast, works with very few points, and is badly biased if the family is wrong — and with a small sample you can't check whether it's wrong, which is the uncomfortable part. KDE makes no shape assumption so it's asymptotically consistent for any density, but it needs far more data, it degrades quickly in more than a couple of dimensions, and everything depends on the bandwidth. It's the bias-variance tradeoff in its cleanest form. My practical rule: with a few dozen points and unimodal-looking data, go parametric and say so; with hundreds and any hint of multimodality, go KDE or discriminative.

67. When is discriminative classification (logistic regression on combined data) better than generative?

    > **Saying it out loud.** When the generative model is misspecified, which is most of the time. If you assume Gaussians and the truth is skewed or bimodal, you get systematically wrong densities and therefore a wrong ratio; a discriminative model only has to learn the decision boundary, which is a much smaller thing to get right. The classic counterpoint, from Ng and Jordan, is that generative models converge faster in the small-sample regime — naive Bayes beats logistic regression when data is scarce, even though logistic regression wins asymptotically. So the honest answer is: discriminative when you have enough data or doubt your model, generative when data is scarce and the assumption is defensible.

68. How do you quantify confidence in the classification of a new sample?

    > **Saying it out loud.** The absolute log-likelihood ratio, which is the evidence in nats — and I'd convert it to a posterior probability with Bayes' rule so it's interpretable. But I'd give a second answer too, because the first is conditional on the densities being right: I'd bootstrap the whole pipeline, resampling the arrays, refitting, and looking at the spread of the resulting posteriors. That captures estimation uncertainty, which the plug-in likelihood ratio silently ignores. If those two disagree — a confident likelihood ratio with a wide bootstrap spread — the honest answer is the wide one.

69. What if both $p(x)$ and $q(x)$ are tiny — how do you handle?

    > **Saying it out loud.** Then the ratio is numerically meaningless and I would refuse to classify. Both densities near zero means the point is outside the region where either array informs me, so a likelihood ratio there is the quotient of two extrapolations, and small errors in either can flip the answer entirely. The right behaviour is to flag it as out of distribution and abstain — via Mahalanobis distance, a k-nearest-neighbour distance to both arrays, an energy score, or ensemble disagreement. Saying "I don't know, and here's how I detect that case" is a stronger answer than producing a number, and in production it's the difference between a system that fails loudly and one that fails silently.

70. Sample complexity scaling: $1/\mathrm{KL}(P\|Q)^2$ — derive the intuition.

    > **Saying it out loud.** I'd correct the premise first: it's one over KL, not one over KL squared. The intuition is a random walk with drift — each sample adds on average KL nats of evidence, so after n samples the log-likelihood ratio has drifted about n times KL, while its random fluctuation is only about the square root of n times its per-sample standard deviation. You're confident once the drift dominates the noise, and setting n times KL against root n gives n on the order of one over KL. Chernoff–Stein states it exactly: error decays like e to the minus n KL. The check that catches the squared version in ten seconds is the Gaussian case — n proportional to variance over squared mean gap, and KL is squared mean gap over twice the variance.

71. What if priors $\pi_P, \pi_Q$ are unknown?

    > **Saying it out loud.** The likelihood ratio doesn't change at all — priors only enter through the threshold. So you can still report the evidence in nats, which is prior-free, and let whoever has the priors set the cutoff. If you must choose, the array sizes are a reasonable estimate *if* the arrays were sampled proportionally to base rates, which is an assumption I'd state rather than assume. If they're a convenience sample — say, equal numbers collected deliberately — using their ratio as the prior is simply wrong. And if the priors are genuinely unknown, the minimax answer is to report the ratio and the sensitivity of the decision to the prior.

72. What if the loss is asymmetric?

    > **Saying it out loud.** Then the threshold moves by the ratio of the misclassification costs — you classify to P when the likelihood ratio exceeds the prior odds times the cost ratio. Concretely, if calling something P when it's really Q costs ten times the reverse, you demand ten times more evidence before saying P. What I'd add is that this is the same thing as sliding along an ROC curve, so if you have the curve you can pick the operating point graphically, and if you don't know the costs you can at least show the interviewer the frontier and say the choice is theirs, not the statistics'.

73. KDE bandwidth — how do you pick it?

    > **Saying it out loud.** Cross-validation is the principled answer — leave-one-out likelihood, choose the bandwidth maximising held-out log-density. Silverman's rule is the fast default if you need a number now. But the substantive point is that bandwidth *is* the bias-variance dial: too small and the estimate is a spiky mess that overfits the sample, too large and you smooth away real structure and the two distributions look artificially similar. So it's not a technicality you can wave through — the answer to "which distribution did it come from" can genuinely flip with the bandwidth, and I'd check sensitivity rather than report one number.

74. What's Silverman's rule of thumb?

    > **Saying it out loud.** Silverman's rule sets the bandwidth to about 1.06 times the sample standard deviation times n to the power minus one fifth, often with the standard deviation replaced by the smaller of it and the interquartile range over 1.34 for robustness. It's derived by minimising asymptotic mean integrated squared error *assuming the true density is Gaussian*, which is the caveat that matters: on multimodal or heavy-tailed data it systematically oversmooths, and it will happily merge two real modes into one. It's a starting point for cross-validation, not a substitute for it.

75. Walk me through the 90-second oral answer end to end.

    > **Saying it out loud.** Frame it as binary classification and give the optimal rule: likelihood ratio against prior odds, Neyman-Pearson. Say the priors come from the array sizes if the arrays reflect base rates. Then spend most of the time on the real content — estimating the densities — laying out parametric, KDE, and discriminative with the tradeoff for each, and saying I'd default to discriminative because you only need the ratio. Then volunteer the failure modes before being asked: flag out-of-distribution points where both densities are tiny and abstain, note the Bayes error floor if the distributions overlap heavily since no method beats it, and give sample complexity as roughly one over KL. Close by naming the assumptions the whole answer rests on. Ninety seconds, and the part that scores is the last thirty.


## Section J — Brain-teaser style (Q76–95)

76. Coin flip: 10 heads in a row. $P(\text{biased})$ given prior $0.5$ on bias?

    > **Saying it out loud.** With a fifty-fifty prior between a fair coin and a two-headed one, ten heads gives posterior odds of 1024 to one for the biased coin, so the probability it's fair is about a tenth of a percent. Do it in odds: each head is one bit of evidence, ten heads is ten bits, prior odds were even, so posterior odds are two to the tenth. The caveat worth adding: this is only that decisive because the alternative is a coin that *always* lands heads — against a coin at 0.9 you'd have far less evidence, and against a continuous prior over bias the answer depends entirely on how much prior mass sits near one.

77. Two arrays of size $n$ from continuous distributions. New point. Decide source.

    > **Saying it out loud.** Likelihood ratio test with the densities estimated from the arrays — parametric if I trust a family, KDE if I have data and don't, or a discriminative classifier which is usually my default since it estimates the ratio directly. Report the evidence as a log-likelihood ratio and convert to a posterior with priors from the array sizes, stating that assumption. Then the honest part: abstain if the point lies where neither array has support, and note that if the distributions overlap heavily there's a Bayes error floor no method gets under.

78. Birthday problem — formula and answer for 50%.

    > **Saying it out loud.** About twenty-three people for a fifty percent chance. The formula is the product over k of one minus k over 365, approximated by the exponential of minus n times n minus one over 730. It surprises people because they answer a different question — the chance someone shares *your* birthday, which needs about 253 people. The generalisation to give is the square root rule: collisions appear after roughly the square root of the number of possible values, which is exactly the birthday attack in cryptography.

79. Monty Hall — and why it breaks under random host.

    > **Saying it out loud.** Switch: two thirds versus one third. The mechanism is that the host knows where the prize is and is constrained to open an empty door, so his choice carries information about the door he didn't open. Under a random host who sometimes reveals the prize, conditioning on the case where he happened to miss it leaves both remaining doors at fifty-fifty and switching is worthless. The general lesson worth stating: you have to condition on the *process* that generated the information, not just on the observed fact — which is the same error behind most selection-bias mistakes.

80. $X, Y$ uniform $[0,1]$ — compute $\mathbb{E}[\max(X, Y)]$.

    > **Saying it out loud.** Two thirds. The max of two uniforms has CDF z squared, hence density 2z, and integrating z times 2z from zero to one gives two thirds. Generalise unprompted: the max of n uniforms has expectation n over n plus one, so it approaches one at a rate of about one over n — which is the same diminishing-returns curve that governs best-of-N sampling.

81. $X \sim \text{Exp}(\lambda)$ — what's $P(X > a+b | X > a)$?

    > **Saying it out loud.** It equals the probability that X exceeds b — memorylessness. Having already waited a tells you nothing about the remaining wait. The exponential is the only continuous distribution with that property, and the geometric is its discrete twin. The modelling caveat I'd add: memorylessness is a strong and often false assumption, since real components wear out, so if the hazard rate rises with age you want a Weibull instead.

82. Sum of $k$ i.i.d. exponentials — what distribution?

    > **Saying it out loud.** Gamma with shape k and the same rate parameter. The intuition that unifies it: this is the waiting time until the k-th arrival in a Poisson process, which ties exponential, Gamma and Poisson into one story rather than three memorised facts. And as k grows the Gamma tends to a normal, which is just the central limit theorem showing up somewhere people don't expect it.

83. Why is median more robust than mean?

    > **Saying it out loud.** Breakdown point. The median tolerates up to half the data being arbitrarily corrupted; the mean tolerates none, since one point sent to infinity takes the mean with it. The tradeoff to name is efficiency: on genuinely Gaussian data the median has about sixty-four percent of the mean's efficiency, so robustness costs you roughly a third more data. Which side to be on depends on how much you believe your tails, and that's a judgement about the data-generating process, not a theorem.

84. Estimate $\pi$ via Monte Carlo.

    > **Saying it out loud.** Sample uniformly in the unit square, count the fraction inside the quarter circle, multiply by four. The part worth volunteering is the accuracy: error shrinks like one over the square root of n, so each extra decimal digit costs a hundred times more samples — which makes this a terrible way to compute pi. The reason Monte Carlo matters anyway is that this rate is independent of dimension, so for high-dimensional integrals it beats every deterministic quadrature scheme.

85. Detect a change-point in a Gaussian stream — algorithm?

    > **Saying it out loud.** Sequential likelihood ratio in some form. CUSUM accumulates the log-likelihood ratio of change versus no-change and alarms at a threshold; generalised likelihood ratio also maximises over the unknown change point and post-change parameter; Bayesian online change-point detection keeps a posterior over run length. The tradeoff that defines the problem is detection delay against false alarm rate, and the threshold is where you pick your point on that curve. The practical caveat: thresholds tuned offline usually misbehave in production because real streams drift even without a change point.

86. German tank problem — MLE and MVUE.

    > **Saying it out loud.** MLE is the largest serial number observed, which is obviously biased low since you almost certainly haven't seen the true maximum. The minimum-variance unbiased estimator adds back the average gap between order statistics: m plus m over n, minus one. The intuition is that the observed max plus one typical spacing is a better guess than the max alone. It's a nice case where MLE is clearly wrong in finite samples despite being asymptotically fine — and the assumptions doing all the work are that IDs run sequentially from one and your sample is uniform.

87. Welch's $t$-test — when?

    > **Saying it out loud.** When the two groups may have different variances — and honestly, as the default, since Welch costs essentially nothing when variances are equal and saves you when they're not. It adjusts the degrees of freedom via the Satterthwaite approximation. Worth saying that pretesting for equal variances and then choosing a test is worse than just always using Welch, because the pretest's own error rate contaminates the final one. If normality is also doubtful, go to Mann-Whitney or a permutation test, noting that Mann-Whitney tests stochastic dominance rather than a difference of means.

88. AB test: $p=0.04, n=10000$ — should you ship?

    > **Saying it out loud.** I wouldn't answer from the p-value. First: what's the effect size and its confidence interval, because at ten thousand samples a significant effect can be commercially irrelevant. How many metrics and variants were tested, since 0.04 among twenty tests is what chance produces. Was n fixed in advance or did someone stop when it turned green, because peeking substantially inflates the false-positive rate. And what's the cost of shipping a null change versus missing a real one — if shipping is cheap and the interval's lower bound is still positive, ship. The senior signal is treating the p-value as one input to a decision under uncertainty rather than as a verdict.

89. Power calculation: detect $p=0.6$ vs $p=0.5$ at 5% Type-I, 5% Type-II — sample size?

    > **Saying it out loud.** About 260 flips. The formula is the squared sum of the two z-values, times p times one minus p, over the squared difference — so 3.29 squared times 0.24 over 0.01. The scaling to state is that it goes as one over the squared effect size, so detecting a fifty-five percent coin instead of a sixty needs four times as many flips. And the assumption doing the work is that n was fixed in advance.

90. Variance of sample variance for Gaussian — formula?

    > **Saying it out loud.** Two sigma to the fourth over n minus one, for a Gaussian. What that says is that the precision of a variance estimate scales with the variance itself, so variance is much harder to estimate than the mean. And the general formula depends on the fourth moment, so for heavy-tailed data it's far worse — for a distribution without a finite fourth moment, the sample variance has infinite variance.

91. Estimate the mean from 3 samples — what's the CI?

    > **Saying it out loud.** Sample mean plus or minus about 4.3 times the standard error, using the t-distribution with two degrees of freedom. The number to say aloud is that 4.3, because it's more than double the 1.96 people reach for reflexively. The real answer is that the interval is very wide and I wouldn't act on it, and that it's conditional on normality — which with three points you have no way to check.

92. Empirical CDF vs density estimation — what's the gotcha?

    > **Saying it out loud.** The empirical CDF is a step function, so its derivative is a sum of point masses: density zero everywhere you haven't observed, infinite where you have. So you can't evaluate a likelihood at a new point, which is exactly what a likelihood ratio needs. The lesson is that getting a density from data requires smoothing, and smoothing requires a bandwidth — there's no assumption-free route from samples to a usable density.

93. Test if a sample is normal — three methods.

    > **Saying it out loud.** Shapiro-Wilk, which is the most powerful for small samples; Anderson-Darling, which weights the tails more; Kolmogorov-Smirnov, general but weak. But I'd lead with a Q-Q plot, because it shows *how* the data departs rather than just whether a test rejects. And the caveat that matters most: with a large enough sample every normality test rejects, since no real data is exactly normal — so the useful question is whether the departure is big enough to break whatever you're about to do.

94. Two-sample distribution test — Kolmogorov-Smirnov vs Mann-Whitney vs $t$-test.

    > **Saying it out loud.** They test different nulls, so the choice is a modelling decision. Kolmogorov-Smirnov tests whether the whole distributions match, no assumptions, but it's not very powerful and it's most sensitive near the median. Mann-Whitney tests stochastic dominance and is robust, good when you care about a shift. A t-test tests means and assumes approximate normality, most powerful when that holds. And a permutation test assumes nothing and is exact under the null by construction, which makes it the right default when you can afford the compute.

95. Estimate KL between two empirical distributions — three methods.

    > **Saying it out loud.** KDE both and integrate numerically, which works in one or two dimensions and not above. A k-nearest-neighbour estimator like Pérez-Cruz, which skips explicit densities and handles moderate dimension better. Or train a discriminator between the two samples and recover the divergence from its outputs, the f-divergence trick. The honest framing is that KL estimation from samples is hard, every estimator is biased and badly so in high dimensions, so I'd want bootstrap error bars and would distrust any single number.


## Section K — Common follow-up probes (Q96–105)

96. "What if your prior is wrong?"

    > **Saying it out loud.** Then it matters most exactly when you have least data. With large n the likelihood swamps the prior and the posterior is essentially the same whatever you assumed; with small n the prior can dominate the answer entirely. The right response is sensitivity analysis — recompute under a few plausible priors and see whether the decision changes. If it does, that's the finding, and the honest report is "the answer depends on an assumption I can't justify from the data" rather than picking one and presenting a number.

97. "What's the variance of your estimator?"

    > **Saying it out loud.** For an MLE, asymptotically the inverse Fisher information over n, and I can report a confidence interval from that. For anything more complicated, or at small n where the asymptotics are shaky, I'd bootstrap — resample the data, rerun the whole pipeline, and look at the spread. The point worth making is that the bootstrap captures uncertainty from every stage including the estimation choices, whereas the plug-in asymptotic variance only captures the last step and silently assumes everything upstream was right.

98. "What if the distributions overlap heavily?"

    > **Saying it out loud.** Then there's a Bayes error floor — the integral of the pointwise minimum of the two densities — and no classifier, no model, and no additional data gets below it. So the correct response to "can you improve accuracy" is no, not with these features. What you can do is get more informative measurements, use multiple samples per decision rather than one, or change the objective to abstain on ambiguous cases. Saying "this is information-theoretically impossible" and quantifying it is a much stronger answer than proposing a better model.

99. "What's your sample complexity?"

    > **Saying it out loud.** It has two parts, which is worth separating. Distinguishing the two distributions costs on the order of one over KL samples — Chernoff–Stein, evidence accumulating at KL nats per sample. Estimating the densities well enough to compute that ratio costs on the order of one over epsilon squared for accuracy epsilon, and much worse in high dimensions because of the curse of dimensionality in nonparametric estimation. Usually the second dominates, which is the practically important point: it isn't the testing that's expensive, it's the modelling.

100. "What if you don't know the parametric family?"

     > **Saying it out loud.** Go non-parametric or discriminative. KDE or k-nearest-neighbours if I want density estimates and have enough data; a flexible classifier if I only need the ratio, which is usually the case. The cost is a slower convergence rate — non-parametric methods have rates that degrade with dimension, so you're paying in sample size for the removed assumption. I'd also try to salvage something: even if I don't know the family, I might be willing to assume unimodality or a shared shape, and a semi-parametric method that uses that is better than assuming nothing.

101. "What if the loss is asymmetric?"

     > **Saying it out loud.** Shift the threshold by the ratio of the costs — demand more evidence before making the expensive error. Formally you're minimising expected loss, and the threshold sits where the two expected costs cross, which is the likelihood ratio equalling the prior odds times the cost ratio. If the costs aren't given, I'd show the ROC frontier and say the operating point is a business decision, not a statistical one — and that's the honest answer rather than quietly assuming symmetry.

102. "How would this fail in production?"

     > **Saying it out loud.** In roughly this order of likelihood: distribution shift, where the arrays are stale and the incoming data has moved; out-of-distribution inputs, where the point lies outside both training regions and the ratio is meaningless; label noise, where the arrays weren't as cleanly separated as claimed; and feedback loops, where acting on the classification changes what data you see next. The monitoring answer is that I'd track the input distribution, not just the accuracy, because accuracy is only measurable if you get labels back and shift shows up in the inputs first.

103. "Why are you confident in your estimator?"

     > **Saying it out loud.** I'd say I'm not, unconditionally — I'd give the confidence interval and say what it's conditional on. Then the concrete supports: an asymptotic interval from Fisher information if the sample is large, a bootstrap if it isn't or if the pipeline is complex, and a sensitivity check across the assumptions that could move the answer. If those disagree, I'd report the widest. The move that reads as senior here is not defending the estimate — it's naming the specific check that would change my mind.

104. "Compare with another method — bias-variance trade-off?"

     > **Saying it out loud.** Almost always the same axis: the more assumptions a method makes, the lower its variance and the higher its bias if the assumptions are wrong. Parametric versus KDE, generative versus discriminative, mean versus median, MLE versus method of moments — all the same trade. So the answer depends on sample size and on how much I trust the assumption, and with small n the biased method usually wins. What I'd avoid is claiming one dominates: the honest framing is which regime you're in, and I'd say what evidence would tell me I'm in the other one.

105. "Connection to information theory?"

     > **Saying it out loud.** Usually KL, and usually through the same door: KL is the expected log-likelihood ratio, so it's evidence per sample, so it sets sample complexity. From there, Fisher information is the local curvature version — the second-order approximation to KL for nearby parameters — which gives you the Cramér-Rao bound. Chernoff information is the tight version of the error exponent that KL bounds. And mutual information plus Fano gives you the converse, what nobody can do. Being able to move between those four is the unifying thread across nearly all of these questions.


## Quick fire (Q106–125)

106. One line: Bayes' rule.

     > **Saying it out loud.** Posterior is proportional to likelihood times prior — or in the form I'd actually use, posterior odds equal prior odds times the likelihood ratio. The odds form is better because evidence adds in logs and the normalising constant vanishes.

107. One line: likelihood ratio test.

     > **Saying it out loud.** Threshold the ratio of the data's probability under one hypothesis to its probability under the other. By Neyman-Pearson it's the most powerful test at any fixed false-positive rate, for two fully specified hypotheses.

108. One line: Neyman-Pearson lemma.

     > **Saying it out loud.** Among all tests with false-positive rate at most alpha, the likelihood ratio test maximises power. It applies to two simple hypotheses; once either side is composite you're in generalised-likelihood-ratio territory and the optimality guarantee weakens.

109. One line: KL between two Gaussians.

     > **Saying it out loud.** With equal variances it's the squared difference of means over twice the variance — essentially half the squared effect size. It's the single most useful KL fact to hold, because it converts any Gaussian separation into evidence per sample instantly.

110. One line: Cramér-Rao bound.

     > **Saying it out loud.** Any unbiased estimator's variance is at least the inverse Fisher information. Two caveats: it's for unbiased estimators, and biased ones can beat it in mean squared error — James-Stein does — and it needs regularity conditions that fail when the support depends on the parameter.

111. One line: Hoeffding inequality.

     > **Saying it out loud.** For bounded independent variables, the sample mean's deviation probability is at most twice the exponential of minus two n t squared over the squared range. Memorise the two in the numerator. It needs boundedness, not merely finite variance.

112. One line: CLT.

     > **Saying it out loud.** Sums of i.i.d. finite-variance variables become Gaussian after scaling, whatever the underlying shape. It needs finite variance, so it fails for Cauchy, and convergence is slowest in the tails — which is unfortunate, since tails are usually what you care about.

113. One line: UCB.

     > **Saying it out loud.** Pick the arm with the highest empirical mean plus an uncertainty bonus that grows with log time and shrinks with the number of pulls — optimism in the face of uncertainty. Achieves logarithmic regret, which is optimal up to constants by Lai-Robbins.

114. One line: Thompson sampling.

     > **Saying it out loud.** Sample once from each arm's posterior and pull the argmax of those samples, so exploration comes free from posterior uncertainty. Same optimal logarithmic regret as UCB, usually better in practice, and simpler — but you need a likelihood model to have a posterior at all.

115. One line: importance sampling.

     > **Saying it out loud.** Estimate an expectation under a hard distribution by sampling from an easy one and reweighting by the density ratio. It fails when the ratio has heavy tails, giving enormous or infinite variance — and it looks fine right up until it doesn't, so check effective sample size.

116. One line: James-Stein.

     > **Saying it out loud.** For three or more Gaussian means, shrinking all estimates toward a common point beats the raw observations in total squared error, always, even if the means are unrelated. It's about the aggregate, not each coordinate — and it's why regularisation is correct rather than merely convenient.

117. One line: Chernoff information.

     > **Saying it out loud.** The exponent governing how fast the optimal Bayes error decays with n — the negative log of the minimised tilted overlap integral between the two densities. It's the tight exponent, and KL bounds it, so quoting KL for symmetric error is optimistic.

118. One line: Bayes error rate.

     > **Saying it out loud.** The lowest error any classifier can achieve — with equal priors, the integral of the pointwise minimum of the two densities. It's a floor, so if it's high the answer isn't a better model, it's better features or an abstain option.

119. One line: empirical CDF.

     > **Saying it out loud.** A step function jumping by one over n at each observation — a consistent estimate of the CDF. Its derivative is point masses, so it gives you no usable density at new points, which is why likelihood-based methods need smoothing.

120. One line: KDE.

     > **Saying it out loud.** Kernel density estimation: place a smooth bump at each observation and add them up. Assumption-free about shape, but everything depends on the bandwidth, which is the bias-variance dial, and it degrades badly beyond a couple of dimensions.

121. One line: Welch's $t$-test.

     > **Saying it out loud.** A two-sample t-test that doesn't assume equal variances, with degrees of freedom adjusted by Satterthwaite. It should be the default, since it costs almost nothing when variances match and pretesting-then-choosing is worse than just using it.

122. One line: power of a test.

     > **Saying it out loud.** One minus the type-two error rate — the probability of detecting an effect of a given size if it's really there. The thing to say is that power is only defined relative to a specific effect size, so "is this test powerful" is unanswerable until someone says what they want to detect.

123. One line: change-point detection.

     > **Saying it out loud.** Sequentially accumulate the log-likelihood ratio of change versus no change and alarm at a threshold — CUSUM, or the generalised version if the post-change parameter is unknown. The threshold trades detection delay against false alarm rate, and there's no way to have both.

124. One line: German tank problem.

     > **Saying it out loud.** Estimate a population size from serial numbers: MLE is the observed maximum, which is biased low, and the unbiased estimator adds back the average gap — max plus max over n, minus one. It rests on IDs being sequential and the sample uniform.

125. One line: discriminative vs generative classification.

     > **Saying it out loud.** Generative models the densities of each class and applies Bayes; discriminative models the boundary or the class probability directly. Discriminative usually wins asymptotically and under misspecification; generative converges faster with little data, per Ng and Jordan. The choice is a bias-variance call keyed to sample size.


---

## Self-grading

- 110+ correct: ready for frontier-lab probability rounds.
- 80–109: re-read framework sections (§2–§8) and the worked examples (§10).
- 50–79: re-read full deep dive then redo.
- <50: spend three days drilling the deep dive.

## 5-day drill plan

- **Day 1:** §1 (framing) + §2 (Bayesian classification). Drill A, B.
- **Day 2:** §3 (MLE) + §4 (concentration). Drill C, D.
- **Day 3:** §5 (KL) + §6 (bandits) + §7 (importance) + §8 (Stein). Drill E, F, G, H.
- **Day 4:** §9 (two-distribution scenario, memorize the 90-second answer) + §10 (25 worked questions). Drill I, J.
- **Day 5:** §11 (follow-up probes) + §12 (senior signals) + Quick fire. Whiteboard 5 random questions end-to-end out loud.
