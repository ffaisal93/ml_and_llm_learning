# MLE and MAP Estimation — Interview Grill

> 45 questions on MLE, MAP, conjugate priors, and the connections to standard ML losses. Drill until you can answer 30+ cold.

---

## A. Likelihood basics

**1. Define likelihood and log-likelihood.**
$L(\theta) = \prod_i p(x_i|\theta)$, $\ell(\theta) = \sum_i \log p(x_i|\theta)$. Treats $\theta$ as the variable, data as fixed.

> **Saying it out loud.** The likelihood is the probability of the data you actually saw, viewed as a function of the parameters rather than of the data. That flip is the whole idea: the data is fixed, it already happened, and you're asking which parameter values would have made it most plausible. The log-likelihood is just the log of that, which turns the product over data points into a sum. And it's worth being precise that the likelihood is not a probability distribution over the parameters — it doesn't integrate to one — which is exactly the gap that Bayesian inference fills in with a prior.

**2. Why log?**
Sums beat products numerically (no underflow). Concavity often preserved. Calculus easier.

> **Saying it out loud.** Three reasons, and the first is purely practical. Multiplying ten thousand probabilities each around 0.01 underflows to exactly zero in floating point, while summing their logs is perfectly stable — that alone is enough. Second, the log turns products into sums, so derivatives become sums of simple terms instead of an ugly product rule. Third, log is monotone increasing, so the argmax is unchanged — you get all of this for free. And for the exponential family, taking the log makes the objective concave, which means one global optimum instead of many.

**3. MLE definition?**
$\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

> **Saying it out loud.** MLE picks the parameter values that make the observed data most probable. That's it — you write down the probability of your dataset as a function of the parameters and you find the peak. The appealing thing is how little you have to assume beyond your model: no prior, no subjective input. The corresponding weakness is that with little data the peak can be in a silly place — three coin flips all heads gives you an MLE of a hundred percent heads, which is the classic illustration of why priors exist.

**4. Why is MLE the default in ML?**
Asymptotically consistent + efficient. Reduces to standard losses (cross-entropy, MSE) under standard distributions. Simple to derive and optimize.

> **Saying it out loud.** Because it's principled and it keeps reducing to the losses we already use. Asymptotically it's consistent, meaning it converges to the truth, and efficient, meaning no unbiased estimator does better — so you're not leaving anything on the table with enough data. And every standard loss falls out of it: Gaussian noise gives you squared error, categorical gives you cross-entropy, Poisson gives you Poisson loss. So instead of picking a loss by taste, you state your assumption about the noise and the loss is determined. The failure mode is small samples, where MLE overfits, which is exactly what MAP and regularization fix.

---

## B. Standard MLE derivations

**5. Derive MLE for Bernoulli.**
$\ell = s \log\theta + (n-s)\log(1-\theta)$. Set derivative to zero: $\hat{\theta} = s/n = \bar{x}$.

> **Saying it out loud.** You write down the log-likelihood — successes times log theta plus failures times log one-minus-theta — take the derivative, set it to zero, and out falls the sample proportion. Successes over trials. That's satisfying because it's what anyone would have guessed, and it's a good sanity check that the machinery isn't producing something exotic. The failure mode to name is the small-sample case: zero successes in five trials gives you an MLE of exactly zero probability, which is almost certainly wrong and is why smoothing exists.

**6. Derive MLE for Gaussian (mean only, variance known).**
$\hat{\mu} = \bar{x}$.

> **Saying it out loud.** The sample mean. Take the log of the Gaussian density, and the only part that involves mu is the negative sum of squared deviations, so maximizing the likelihood is minimizing squared error — differentiate, set to zero, and you get the average. That's the cleanest possible illustration of why squared loss and Gaussian assumptions are the same statement, and it's the one-line answer to "why do we use MSE for regression."

**7. MLE for Gaussian variance?**
$\hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$. Biased — divisor should be $n-1$ for unbiased.

> **Saying it out loud.** The MLE for variance is the average squared deviation from the sample mean, dividing by n. And it's biased low — the unbiased version divides by n minus one, which is Bessel's correction. Worth noting that MLE is biased here even though it's the maximum likelihood answer, because likelihood and unbiasedness are simply different criteria. In practice with large n the difference is negligible, which is why nobody worries about it in deep learning.

**8. Why is MLE for variance biased?**
$\bar{x}$ is closer to the sample than the true $\mu$. $\sum (x-\bar{x})^2 < \sum (x-\mu)^2$ on average.

> **Saying it out loud.** Because you estimated the mean from the same data. The sample mean is, by construction, the point that minimizes the sum of squared deviations for your particular sample — so the deviations from it are systematically smaller than the deviations from the true mean would be. You've spent one degree of freedom estimating the center, and dividing by n minus one instead of n gives it back. The intuition to say out loud is: the sample mean is pulled toward whatever data you happened to draw, so distances from it look artificially short.

**9. MLE for Poisson rate?**
$\hat{\lambda} = \bar{x}$.

> **Saying it out loud.** The sample mean again. Poisson has a single parameter that's simultaneously the mean and the variance, and the MLE turns out to be the average count. There's a nice pattern here worth naming: for Bernoulli, Gaussian mean, and Poisson, the MLE is the sample average — which isn't a coincidence, it's a property of the exponential family, where the MLE always matches the model's expected sufficient statistics to the empirical ones.

**10. MLE for multinomial?**
$\hat{\theta}_k = n_k/n$ — empirical class frequency.

> **Saying it out loud.** Empirical frequencies — the count of each category divided by the total. You derive it with a Lagrange multiplier to enforce that the probabilities sum to one, and the answer is exactly what you'd have written down without any math. Same failure mode as Bernoulli, magnified: any category you never observed gets probability zero, which in language modeling means one unseen word makes an entire sentence impossible. That's the direct route to Laplace smoothing, which is the Dirichlet-prior version of this estimate.

**11. MLE for linear regression — what loss does it correspond to?**
Squared error. $\arg\max \ell$ under Gaussian noise = $\arg\min \sum (y - w^\top x)^2$ = OLS.

> **Saying it out loud.** Squared error, and the derivation is worth having ready. Assume the target equals a linear function of the inputs plus Gaussian noise. Write the log-likelihood: the only term involving the weights is the negative sum of squared residuals divided by twice the variance. So maximizing likelihood is minimizing squared error, and ordinary least squares is MLE under Gaussian noise. The point of saying it this way is that squared error isn't an arbitrary convention — it's a claim about your noise, which also tells you when it's the wrong choice, namely when the noise is heavy-tailed and you should be using Huber or absolute error instead.

**12. MLE for logistic regression — what loss?**
Cross-entropy / log loss. $\sum [y \log \sigma(w^\top x) + (1-y) \log(1-\sigma(w^\top x))]$. No closed form.

> **Saying it out loud.** Cross-entropy — the same thing as log loss or negative log-likelihood under a Bernoulli. You assume each label is a coin flip whose probability is the sigmoid of a linear function, write down the likelihood, and take logs, and you get the familiar y-log-p plus one-minus-y-log-one-minus-p. That's the whole derivation, and it's the reason cross-entropy is mandated rather than chosen. Unlike linear regression, there's no closed form, so you solve it iteratively.

**13. Why does logistic regression have no closed-form MLE?**
The score equation is non-linear in $w$ (sigmoid). Need iterative solver: IRLS, gradient descent, Newton-Raphson.

> **Saying it out loud.** Because the sigmoid makes the score equation non-linear in the weights. In linear regression the derivative is linear, so setting it to zero gives you a system you can solve directly — the normal equations. Here, setting the derivative to zero gives you an equation with the weights trapped inside a sigmoid, and there's no algebraic way out. So you iterate: Newton-Raphson, which for this problem is iteratively reweighted least squares, or plain gradient descent at scale. The good news is that the objective is concave, so there's a unique global optimum and any decent solver finds it.

---

## C. Asymptotic theory

**14. Asymptotic distribution of MLE?**
$\sqrt{n}(\hat{\theta} - \theta_0) \to \mathcal{N}(0, I(\theta_0)^{-1})$ where $I$ is Fisher information.

> **Saying it out loud.** As you get more data, the MLE becomes normally distributed around the true value, with a variance that shrinks like one over n and whose shape is the inverse Fisher information. That's the result underpinning basically all classical statistics — it's where standard errors, confidence intervals and Wald tests come from. The practical reading is that you get error bars on your estimate for free, without any bootstrapping, provided the sample is large enough and you're not on a boundary.

**15. What's Fisher information?**
$I(\theta) = -\mathbb{E}[\partial^2 \ell/\partial \theta^2]$. Curvature of expected log-likelihood; measures how sharply peaked it is around true value.

> **Saying it out loud.** Fisher information measures how sharply the likelihood peaks around the truth — formally, the expected curvature of the log-likelihood. High information means a narrow peak, so the data pins the parameter down tightly; low information means a flat likelihood and a lot of parameter values look equally good. Its inverse is the variance of the MLE, which is the direct link to precision. It's also the metric that natural gradient descent uses, which is a nice bridge to make in an ML interview.

**16. Why is MLE asymptotically efficient?**
Variance achieves Cramér-Rao lower bound: $1/I(\theta)$. No unbiased estimator can do better asymptotically.

> **Saying it out loud.** Because its asymptotic variance exactly hits the Cramér-Rao lower bound, which is the theoretical floor on the variance of any unbiased estimator. So asymptotically you cannot do better — that's what efficiency means. The two caveats to attach are that it's asymptotic, so it says nothing about small samples where MLE can be badly biased, and that it's restricted to unbiased estimators — a biased estimator like ridge regression can absolutely beat MLE in mean squared error by trading a little bias for a lot of variance.

**17. Invariance of MLE — what does it mean?**
$\widehat{g(\theta)} = g(\hat{\theta})$. So MLE of standard deviation = $\sqrt{\hat{\sigma}^2_{\mathrm{MLE}}}$.

> **Saying it out loud.** Invariance means the MLE of a function of a parameter is that function of the MLE. If you've estimated the variance, the MLE of the standard deviation is just the square root of it — no re-derivation needed. That's a genuinely convenient property and it's not shared by MAP or by unbiasedness: an unbiased estimator of variance does not give you an unbiased estimator of standard deviation. It's also the cleanest contrast to draw with MAP, which is not reparameterization-invariant because the prior density picks up a Jacobian.

**18. When does asymptotic theory fail?**
Boundary parameters (e.g., $\theta = 0$ when domain is $[0, \infty)$), non-identifiable models, infinite Fisher information, non-iid data.

> **Saying it out loud.** The theory needs the true parameter to be in the interior of the parameter space, so it breaks at boundaries — estimating a variance that's genuinely zero, or a mixture weight at zero, gives you a non-normal sampling distribution and invalid p-values. It also fails when the model isn't identifiable, so different parameters give the same likelihood, which is the norm in neural networks. And it assumes independent, identically distributed data, which time series and clustered data violate. The practical version: for any modern deep model, none of this applies, which is why we bootstrap or use held-out data instead.

---

## D. MAP

**19. Define MAP.**
$\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta p(\theta|x) = \arg\max_\theta [\log p(x|\theta) + \log p(\theta)]$.

> **Saying it out loud.** MAP finds the parameter with the highest posterior probability — the most likely explanation after combining what the data says with what you believed beforehand. Mechanically it's just MLE with an extra term: log-likelihood plus log-prior. That single extra term is the whole difference, and it's why MAP and regularization turn out to be the same thing wearing different clothes.

**20. MAP vs MLE — key relationship?**
MAP = MLE + log-prior penalty.

> **Saying it out loud.** MAP is MLE plus a penalty, and the penalty is the log of your prior. That's the one sentence to remember, because it immediately gives you the ML translation: every regularizer you use is a prior in disguise. Weight decay is a Gaussian prior, L1 is a Laplace prior. And it tells you what the regularization strength means — it's the ratio of noise variance to prior variance, so tuning lambda is really stating how much you trust the data relative to your prior belief.

**21. MAP equals MLE when?**
Uniform (improper) prior — log-prior is constant, has no effect.

> **Saying it out loud.** When the prior is uniform — flat over the parameter space. Then the log-prior is a constant, it doesn't depend on theta, and it drops out of the argmax. So MLE is MAP with a completely non-committal prior, which is a nice way to see MLE as a special case rather than a separate philosophy. The catch is that a uniform prior over an unbounded space isn't a proper distribution and doesn't integrate to one, and it isn't invariant under reparameterization — flat in theta isn't flat in log theta.

**22. MAP vs posterior mean — same?**
No. MAP is the *mode*; posterior mean is the *expectation*. Different unless posterior is symmetric.

> **Saying it out loud.** No, and the difference matters. MAP is the mode of the posterior, the single highest point; the posterior mean is its average. They coincide only when the posterior is symmetric, like a Gaussian. For a skewed posterior they can be far apart, and the mean is often the better summary — it's the estimate that minimizes expected squared error, whereas the mode minimizes a zero-one loss that rarely reflects what you care about. The extreme case is a posterior with a tall narrow spike and a broad hump: MAP sits on the spike while nearly all the probability mass lives elsewhere.

---

## E. Priors as regularizers

**23. Gaussian prior on weights → what regularizer?**
$\ell_2$. $\log \mathcal{N}(0, \tau^2 I) \propto -\|w\|^2/(2\tau^2)$.

> **Saying it out loud.** A Gaussian prior centered at zero gives you L2 regularization — weight decay. Take the log of a Gaussian density and you get the negative squared norm over twice the variance, and that's exactly the ridge penalty. The tightness of the prior sets the strength: a narrow prior means a strong belief that weights are near zero, so a large lambda. That's the sentence that makes weight decay stop feeling arbitrary — you're saying you believe, before seeing any data, that the weights are small.

**24. Show ridge regression = MAP under Gaussian prior.**
Likelihood Gaussian, prior Gaussian. $\log p(w|x,y) = -\frac{1}{2\sigma^2}\|y-Xw\|^2 - \frac{1}{2\tau^2}\|w\|^2$. Maximizing → ridge with $\lambda = \sigma^2/\tau^2$.

> **Saying it out loud.** Write down the posterior: Gaussian likelihood for the residuals times Gaussian prior on the weights. Take logs, and you have a negative squared residual term scaled by one over the noise variance, plus a negative squared weight-norm term scaled by one over the prior variance. Maximizing that is minimizing squared error plus lambda times the squared norm — ridge regression, with lambda equal to the ratio of noise variance to prior variance. The interpretation to volunteer is that lambda is a signal-to-noise ratio: noisy data or a confident prior both push you toward more shrinkage.

**25. Laplace prior → what regularizer?**
$\ell_1$. $\log \mathrm{Laplace}(0, b) \propto -|w|/b$.

> **Saying it out loud.** A Laplace prior gives you L1 — lasso. The Laplace density decays like the exponential of negative absolute value, so its log is a negative absolute-value penalty. The reason this produces different behavior from Gaussian is the shape at zero: Laplace has a sharp spike there, encoding a genuine belief that many weights are exactly zero, whereas a Gaussian is smooth at zero and merely thinks they're small. That distinction between "small" and "exactly zero" is the whole sparsity story.

**26. Why does $\ell_1$ produce sparsity?**
$\ell_1$ ball has corners at axes; optimum is often *at* a corner → some weights exactly zero. Geometrically, lasso intersects the constraint set at a corner.

> **Saying it out loud.** Geometrically, the L1 constraint region is a diamond with corners on the axes, and the corners stick out. As you inflate the loss contours until they touch the constraint region, they'll usually touch at a corner — and a corner is a point where some coordinates are exactly zero. The calculus version is that the absolute value has a constant-size gradient all the way in to zero, so it keeps pushing a small weight until it hits zero and then holds it there, whereas the squared penalty's gradient vanishes as the weight shrinks. Both stories land in the same place: L1 gives you exact zeros, so it does feature selection as a side effect.

**27. Why does $\ell_2$ not produce sparsity?**
$\ell_2$ ball is round → optimum is generically in the interior of an axis hyperplane → all weights non-zero.

> **Saying it out loud.** Because the L2 constraint region is a sphere — perfectly round, no corners — so the point where the loss contours first touch it is generically off the axes, meaning every coordinate is small but nonzero. The gradient story is the same: the derivative of the squared penalty is proportional to the weight itself, so as a weight approaches zero the pressure to shrink it further vanishes and it converges to something tiny rather than exactly zero. Practically that means ridge shrinks everything smoothly and lasso selects, which is why elastic net exists to get some of both.

**28. What does early stopping correspond to?**
Approximately MAP with a Gaussian prior — the early stop limits how far weights move from the (zero) initialization. Connection is exact for linear models (Friedman, Hastie & Tibshirani).

> **Saying it out loud.** Early stopping acts like an L2 penalty. If you start at zero and take gradient steps, stopping early means the weights haven't traveled far from the origin, which is exactly what a Gaussian prior centered at zero would enforce. For linear models with gradient descent you can make this precise — there's a correspondence between the number of steps and a specific lambda, with more steps meaning weaker regularization. In deep networks it's only an analogy, but a useful one, and it explains why early stopping and weight decay are partly redundant, and why the epoch count is a hyperparameter you should tune rather than a budget you exhaust.

---

## F. Conjugate priors

**29. What's a conjugate prior?**
Prior whose posterior stays in the same family. Enables closed-form Bayesian updates.

> **Saying it out loud.** A conjugate prior is one where, after you multiply by the likelihood, the posterior comes out in the same family as the prior. Beta prior plus Binomial data gives you a Beta posterior. That's an enormous practical convenience, because updating becomes arithmetic on the parameters instead of numerical integration — no MCMC, no approximation. The cost is flexibility: you're restricted to whatever shapes the conjugate family can express, so if your real prior belief is bimodal you can't represent it. That's why conjugacy dominated pre-computational Bayes and matters less now, except that it still makes analysis clean and it's why the Beta-Binomial shows up in every bandit algorithm.

**30. Conjugate of Bernoulli/Binomial?**
Beta.

> **Saying it out loud.** Beta. It lives on the interval from zero to one, which is exactly where a probability lives, and it has two shape parameters that behave like counts of prior successes and failures. Updating is addition: add your observed successes to the first parameter and failures to the second. That's the pair behind Thompson sampling and behind every Bayesian A/B test on click-through rate, so it's worth knowing cold.

**31. Conjugate of multinomial/categorical?**
Dirichlet.

> **Saying it out loud.** Dirichlet — it's the multivariate generalization of Beta, a distribution over probability vectors that sum to one. Updating is the same trick: add your observed counts to the prior parameters. It's the formal justification for add-alpha smoothing in language models, and it's the prior over topic distributions in latent Dirichlet allocation, which is where the name comes from. A symmetric Dirichlet with parameter below one favors sparse distributions, which is exactly why LDA produces documents about a few topics rather than a little of everything.

**32. Conjugate of Poisson?**
Gamma.

> **Saying it out loud.** Gamma. Poisson counts events, Gamma is the conjugate prior over the rate, and updating is again just addition — add the total observed count to the shape parameter and the number of observation periods to the rate parameter. The resulting predictive distribution is negative binomial, which is why negative binomial regression is the standard tool for overdispersed count data where Poisson's variance-equals-mean assumption fails.

**33. Conjugate of Gaussian (mean, variance known)?**
Gaussian.

> **Saying it out loud.** Gaussian is its own conjugate for the mean. The posterior mean comes out as a precision-weighted average of the prior mean and the sample mean — whichever one you're more certain about pulls harder. That's a beautifully intuitive result and it's the same shrinkage formula that shows up in Kalman filters and in empirical Bayes. If the variance is also unknown, the conjugate becomes normal-inverse-gamma, which is worth mentioning as the more realistic case.

**34. Beta-Bernoulli: prior + 5 successes / 3 failures from Beta(2, 2). What's the posterior?**
Beta(2 + 5, 2 + 3) = Beta(7, 5).

> **Saying it out loud.** Beta(7, 5). You just add: two prior successes plus five observed successes gives seven, two prior failures plus three observed gives five. That's the whole update, and the fact that it's arithmetic rather than integration is the entire selling point of conjugacy. Posterior mean is seven over twelve, about 0.58 — pulled down from the raw data's five out of eight, 0.625, by the prior's insistence on being closer to a half.

**35. Beta-Bernoulli posterior mean?**
$(\alpha + s)/(\alpha + \beta + n)$.

> **Saying it out loud.** Prior successes plus observed successes, over the total of everything — prior pseudo-counts plus real trials. What makes it worth stating is what it does at the extremes: with no data it's the prior mean, and as the number of trials grows it converges to the empirical rate, because the fixed pseudo-counts get swamped. So it's an automatic shrinkage estimator that trusts the data more and more as data accumulates, which is exactly the behavior you'd design by hand.

**36. With $\alpha = \beta = 1$, what does the posterior mean become?**
$(s+1)/(n+2)$ — Laplace's rule of succession / add-one smoothing.

> **Saying it out loud.** You get successes plus one over trials plus two — Laplace's rule of succession. That's the classic answer to "the sun has risen n times, what's the probability it rises tomorrow," and it's the same add-one smoothing used in Naive Bayes. The key behavior is that it never returns exactly zero or exactly one: zero successes in five trials gives one-seventh rather than zero, which is much more sensible than the MLE. Beta(1,1) is the uniform distribution, so this is what a completely non-committal prior does.

**37. What's the "pseudo-count" interpretation?**
Beta($\alpha, \beta$) = $\alpha$ pseudo-successes, $\beta$ pseudo-failures. The prior acts like imaginary data.

> **Saying it out loud.** The prior parameters behave exactly like data you never actually observed. A Beta(2,2) prior is you saying "pretend I already saw two successes and two failures" — which is why the update is plain addition. This is by far the most useful way to communicate priors to non-statisticians, because you can ask "how many imaginary coin flips is your belief worth?" and get a real answer. It also tells you immediately how much a prior matters: two pseudo-counts against ten thousand observations is nothing, against five observations it's decisive.

**38. Dirichlet prior as smoothing — why does NLP use add-$\alpha$ smoothing?**
$N$-gram counts $n_w$ with Dirichlet($\alpha$) prior. Posterior probability for word $w$: $(n_w + \alpha)/(\sum_v n_v + V\alpha)$. Prevents zero probabilities for unseen tokens.

> **Saying it out loud.** Add-alpha smoothing is just the posterior mean under a Dirichlet prior — it's not a hack, it's Bayes. You add alpha to every word count and alpha times the vocabulary size to the denominator, which is exactly the Dirichlet posterior mean. Framing it that way tells you what alpha means: it's how many imaginary occurrences of every word you're assuming, so alpha equals one is a fairly strong claim on a fifty-thousand-word vocabulary, which is why smaller values usually work better. And it explains why smoothing is necessary at all — without it, one unseen n-gram sets the probability of the entire sentence to zero.

---

## G. Connections to standard ML

**39. Cross-entropy minimization equals what?**
MLE in general (negative log-likelihood). Specifically, minimizing CE = minimizing forward KL from data to model (up to data-entropy constant).

> **Saying it out loud.** Minimizing cross-entropy is maximum likelihood, full stop — cross-entropy against a one-hot label is literally the negative log-probability the model assigned to the right answer. The information-theoretic version is that it's the forward KL divergence from the data distribution to the model, offset by the data's entropy, which doesn't depend on your parameters. Both framings tell you the same thing: cross-entropy isn't a design choice, it's what maximum likelihood looks like for a categorical output.

**40. Forward KL vs reverse KL — which does MLE minimize?**
Forward: $\mathrm{KL}(p^* \| p_\theta)$. Mode-covering. (VI minimizes reverse KL.)

> **Saying it out loud.** MLE minimizes forward KL — data first, model second — which is mode-covering. The model gets heavily penalized for assigning near-zero probability anywhere the data actually appears, so it spreads out to cover everything, even the empty space between modes. That's the direct explanation for why maximum-likelihood generative models produce blurry images and bland text. Variational inference goes the other way with reverse KL, which is mode-seeking: it picks one mode and fits it tightly, at the cost of ignoring the others.

**41. Why is squared loss the right loss for regression?**
Under Gaussian noise assumption, MLE = squared loss. Other noise models give other losses (Huber for heavy-tailed, MAE for Laplace noise).

> **Saying it out loud.** It isn't inherently right — it's right if your noise is Gaussian. Squared loss is the negative log-likelihood under additive Gaussian noise, so choosing it is asserting that your errors are symmetric and light-tailed. When that's false, the loss is wrong: with heavy-tailed noise or outliers, squared error lets a single bad point dominate the fit, and you should use Huber, which is quadratic near zero and linear in the tails, or absolute error, which corresponds to Laplace noise and estimates the median. The framing that scores is: state your noise model and the loss is determined, rather than picking a loss by habit.

**42. RLHF reward model — what's the MLE?**
Bradley-Terry: $p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$. MLE is logistic regression on (preferred, rejected) pairs.

> **Saying it out loud.** The reward model is trained by maximum likelihood under the Bradley-Terry model, which says the probability that a human prefers one response over another is the sigmoid of the difference in their rewards. Take the negative log of that over your preference pairs and you have the training loss — which is literally logistic regression on reward differences. The consequence worth naming is that only differences are identified: the reward scale is arbitrary up to a shift, which is why raw reward values are meaningless and only comparisons matter.

**43. SFT loss = MLE of what?**
Conditional language model: $p(y|x; \theta)$. Minimize $-\sum_{(x,y)} \log p_\theta(y|x)$ = MLE.

> **Saying it out loud.** Supervised fine-tuning is maximum likelihood on a conditional language model — minimize the negative log-probability of each target token given the prompt and the preceding tokens. It's exactly pretraining, just on curated pairs and usually with the loss masked over the prompt so you only train on the response. That's why SFT inherits pretraining's characteristic weakness: being forward KL, it's mode-covering, so the model learns to hedge across all the demonstrated behaviors rather than committing to the best one — which is a decent part of the argument for doing preference optimization afterward.

**44. DPO loss derivation starting point?**
Substitute the optimal RLHF policy ($\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x) \exp(r/\beta)$) into the Bradley-Terry MLE, eliminating the reward — yields a closed-form classification objective on preferences.

> **Saying it out loud.** Start with the KL-regularized RL objective and solve it exactly — the optimal policy is the reference policy reweighted by the exponential of reward over beta, normalized. Now invert that: the reward equals beta times the log-ratio of optimal policy to reference, plus a partition function that depends only on the prompt. Substitute that into the Bradley-Terry likelihood, and because Bradley-Terry only sees the difference of two rewards on the same prompt, the partition function cancels exactly. What's left is a loss you can compute directly from the policy's log-probabilities. So DPO's claim is that the reward model was never needed — the policy is implicitly one.

---

## H. Subtleties

**45. Is MLE always unbiased?**
No. MLE for Gaussian variance is biased; many other MLEs are biased in finite samples.

> **Saying it out loud.** No. The most famous counterexample is the variance of a Gaussian, where the MLE systematically underestimates because you estimated the mean from the same data. Plenty of others are biased in finite samples too. What MLE guarantees is asymptotic behavior — consistency and efficiency as n grows — not unbiasedness at any particular n. And that's fine, because unbiasedness isn't actually the goal: mean squared error is, and a biased estimator with lower variance often wins, which is the entire justification for regularization.

**46. Is MAP always unbiased?**
Almost never. MAP introduces deliberate bias to reduce variance.

> **Saying it out loud.** Almost never, and deliberately so. The whole point of a prior is to pull your estimate toward it, which introduces bias by construction. That's a feature: you're trading bias for a reduction in variance, and when data is scarce that trade is strongly favorable in mean squared error. Ridge regression is the canonical example — biased, and better than unbiased OLS on almost any real problem with correlated features. The bias shrinks as data accumulates, since the likelihood eventually swamps the prior.

**47. Why might you prefer MAP over MLE?**
Small data + strong prior → MAP regularizes against overfitting. Equivalent to standard regularization.

> **Saying it out loud.** When data is scarce and you actually know something. With five coin flips, MLE will happily tell you the coin is a hundred percent heads, and any sane prior fixes that. More generally, MAP is regularization: if you're using weight decay you're already doing MAP with a Gaussian prior, whether you call it that or not. The tradeoff is that a wrong prior biases you toward a wrong answer and, unlike overfitting, that bias doesn't go away by looking at the training data — you find it on the held-out set.

**48. Why might you prefer Bayesian inference over MAP?**
Need uncertainty estimates, want credible intervals, decision-theoretic problems with non-symmetric loss. MAP throws away the posterior shape.

> **Saying it out loud.** Because MAP collapses the entire posterior to a single point and throws away everything about your uncertainty. If you need credible intervals, or you're making a decision with asymmetric costs, or you want to know whether the model is confident, that shape is exactly what you need. There's also a subtler problem: the mode can sit somewhere with almost no probability mass around it, especially in high dimensions where the bulk of the volume is far from the peak. The cost of going fully Bayesian is compute — MCMC or variational inference instead of one optimization — which is why MAP is the practical default and full Bayes is reserved for when uncertainty is the product.

**49. When does MAP become a poor summary of the posterior?**
Multimodal posterior, highly skewed posterior, transformation-dependent (MAP is not invariant under reparameterization, but MLE is — MAP point shifts under variable change).

> **Saying it out loud.** MAP is a bad summary whenever the posterior isn't roughly unimodal and symmetric. With two modes, the mode picks one and pretends the other doesn't exist. With a heavy skew, the mode can be far from where most of the probability lives. And in high dimensions the mode is often in a thin, atypical region — a Gaussian's density peaks at the center but almost all of its mass is in a shell far from it, so the most probable point is nothing like a typical sample. The other structural complaint is that MAP moves under reparameterization, so which point you call "most probable" depends on your choice of coordinates.

**50. Why is MAP not invariant under reparameterization?**
Under a transformation $\theta \to \phi = g(\theta)$, the prior density transforms by a Jacobian. The mode of $p(\phi|x)$ is generally not $g(\hat{\theta}_{\mathrm{MAP}})$.

> **Saying it out loud.** Because a density isn't a probability — it changes when you change variables. Reparameterize theta into some function of it, and the prior density picks up a Jacobian factor, which shifts the location of the peak. So the MAP estimate for the standard deviation is not the square root of the MAP estimate for the variance, which is unsettling if you thought MAP was giving you "the answer." MLE doesn't have this problem, because the likelihood is a function of theta with no measure attached, so it transforms cleanly. The posterior *mean* is also not invariant, but the posterior distribution itself is — it's only the act of picking a single point that introduces the dependence on coordinates.

---

## Quick fire

**51.** *MLE Bernoulli?* Sample mean.
**52.** *MLE Gaussian variance divisor?* $n$ (biased).
**53.** *Unbiased Gaussian variance divisor?* $n-1$.
**54.** *OLS = MLE under what?* Gaussian noise.
**55.** *Ridge = MAP under what?* Gaussian prior.
**56.** *Lasso = MAP under what?* Laplace prior.
**57.** *Conjugate of Bernoulli?* Beta.
**58.** *Beta($\alpha, \beta$) mean?* $\alpha/(\alpha+\beta)$.
**59.** *Beta(1,1) is?* Uniform on $[0,1]$.
**60.** *MLE achieves what bound?* Cramér-Rao.

---

## Self-grading

If you can't answer 1-15, you don't know MLE. If you can't answer 16-35, you'll struggle on every Bayesian/regularization question. If you can't answer 36-50, frontier-lab questions on RLHF/DPO/loss design will go past you.

Aim for 40+/60 cold.
