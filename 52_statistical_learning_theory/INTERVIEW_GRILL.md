# Statistical Learning Theory — Interview Grill

> 40 questions on ERM, PAC, VC, Rademacher, bias-variance, double descent. Drill until you can answer 28+ cold.

---

## A. Empirical risk minimization

**1. Define population risk.**
$R(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f(x), y)]$ — expected loss on the true distribution.

**2. Define empirical risk.**
$\hat{R}_n(f) = \frac{1}{n}\sum_i \ell(f(x_i), y_i)$ — average loss on training sample.

**3. ERM definition?**
$\hat{f}_n = \arg\min_{f \in \mathcal{F}} \hat{R}_n(f)$.

**4. Approximation vs estimation error?**
Approximation: gap between best in $\mathcal{F}$ and true optimum. Estimation: gap between ERM solution and best in $\mathcal{F}$. Bias-variance in formal terms.

**5. Why does ERM fail for too-large $\mathcal{F}$?**
Many functions interpolate the training data with very different test behavior. Empirical winner overfits.

> **Saying it out loud.** ERM is the whole of practical machine learning in one line: you cannot measure error on the true distribution, so you minimize error on the sample you have. The theory is about when that substitution is safe. It splits the damage into two pieces — approximation error, meaning the best function you're even allowed to pick is imperfect, and estimation error, meaning the one you actually picked was chosen using noisy data. Widen the model class and the first shrinks while the second grows, which is bias and variance wearing formal clothes. Named failure mode: in a class rich enough to interpolate, many functions have identical zero training error and wildly different test error, so ERM alone can't tell them apart.

---

## B. PAC learning

*In plain language:* PAC learning turns “how much data do I need?” into a number. You accept two kinds of imperfection up front — being within $\epsilon$ of the best achievable error rather than exactly at it, and being allowed to fail on a $\delta$ fraction of unlucky training draws. Probably, approximately, correct.

**6. State PAC learnability.**
Algorithm returns $\hat{f}$ such that $\mathbb{P}(R(\hat{f}) - R^* \leq \epsilon) \geq 1 - \delta$, with sample complexity $n(\epsilon, \delta)$ polynomial.

**7. Sample complexity for finite $\mathcal{F}$ realizable?**
$n \geq (\log|\mathcal{F}| + \log(1/\delta))/\epsilon$.

**8. Sample complexity agnostic case?**
$n \geq (\log|\mathcal{F}| + \log(1/\delta))/\epsilon^2$. Slower rate.

**9. Why log $|\mathcal{F}|$ and not $|\mathcal{F}|$?**
Union bound over $\mathcal{F}$ gives $|\mathcal{F}|$, log appears via Hoeffding's exponential concentration → log of count.

**10. Realizable vs agnostic?**
Realizable: some $f \in \mathcal{F}$ has zero error. Agnostic: best $f$ has positive error.

> **Saying it out loud.** PAC learning gives you a sample-complexity number instead of a feeling. The promise is two-sided: with probability at least $1-\delta$ over the draw of the training set, your hypothesis is within $\epsilon$ of the best one available. For a finite hypothesis class the count enters only through its logarithm — that's a union bound over hypotheses meeting Hoeffding's exponentially small tail, and the exponential and the log cancel each other out — so a thousand times more candidates costs about seven times more data. The number worth quoting is the rate change: realizable needs $1/\epsilon$ samples, agnostic needs $1/\epsilon^2$, so once no hypothesis is perfect, halving your error tolerance quadruples the data you need.

---

## C. VC dimension

*In plain language:* VC dimension measures how flexible a family of models is. Ask how many points it can label in every possible way: if for $d$ points you can name any pattern of pluses and minuses and the family always has a function that fits it, its capacity is at least $d$. It's a count of effective degrees of freedom, not a measure of how good the model is.

**11. Define VC dimension.**
Size of largest set shattered by $\mathcal{F}$ — i.e., for which $\mathcal{F}$ realizes every binary labeling.

**12. VC of linear classifiers in $\mathbb{R}^d$?**
$d + 1$.

**13. VC of axis-aligned rectangles in $\mathbb{R}^2$?**
$4$.

**14. VC of decision stumps?**
1D axis-aligned threshold: VC = 2 (a stump can shatter any 2 points but not 3 collinear). Over $d$ binary features (axis-aligned thresholds), VC = $\Theta(\log d)$.

**15. VC bound on generalization gap?**
$O(\sqrt{(\mathrm{VC} + \log(1/\delta))/n})$. Shrinks as $1/\sqrt{n}$ for fixed VC.

**16. Why is VC bound vacuous for deep nets?**
VC dim of a deep net is enormous (exponential in some parameters). Bound says "you might be wildly overfitting" — but empirically you're not.

> **Saying it out loud.** VC dimension asks the largest number of points a model class can label in absolutely every possible way. Concretely: a line in the plane can split any three points however you like, but four points arranged as an XOR beat it — so linear classifiers in $d$ dimensions have VC dimension $d+1$, which is satisfyingly close to their parameter count. The payoff is a distribution-free generalization bound where the gap shrinks like the square root of VC over $n$. And the honest closing line is the failure mode: for a modern deep network the VC dimension is so enormous that the bound guarantees a gap of at most something larger than 1 — literally true, completely vacuous, and the reason margin- and norm-based measures exist.

---

## D. Rademacher complexity

*In plain language:* Rademacher complexity measures capacity by seeing how well a model family can fit pure noise. Throw away the real labels, replace them with random plus-ones and minus-ones, and ask how strongly the best function in the class can correlate with that garbage. Fits noise perfectly, scores near 1; can't chase it at all, scores near 0.

**17. Rademacher complexity intuition?**
How well can $\mathcal{F}$ fit random binary labels (Rademacher variables)? Larger = more capacity.

**18. Rademacher generalization bound?**
$R(f) \leq \hat{R}_n(f) + 2 \mathfrak{R}_n(\mathcal{F}) + O(\sqrt{\log(1/\delta)/n})$.

**19. Rademacher of linear classifiers with $\|w\| \leq B$?**
$O(B/\sqrt{n})$. Depends on norm, not dimension!

**20. Why is Rademacher tighter than VC?**
Distribution-aware. VC is worst case over all distributions; Rademacher uses the actual training sample.

**21. Margin-based bounds — what's the idea?**
Replace VC dim with norm-times-margin terms. Tighter for trained networks (Bartlett-Foster-Telgarsky).

> **Saying it out loud.** Rademacher complexity measures capacity empirically: replace your labels with coin flips and see how well the model class can still fit them. If it can fit noise, its low training error means nothing, and that quantity shows up directly — doubled — in the generalization bound. Two advantages over VC dimension. It's distribution-aware, computed on the data you actually have rather than the worst case anyone could hand you. And it's sensitive to scale: a linear class with weight norm bounded by $B$ has complexity $O(B/\sqrt{n})$ with no dependence on dimension at all. That's the punchline, and it's what margin bounds run with — the size of the weights controls capacity more than the number of them.

---

## E. Bias-variance and double descent

**22. State the bias-variance trade-off.**
More capacity → less approximation error, more estimation error. U-shaped test error. Find sweet spot.

**23. What's double descent?**
Test error has *second* descent in over-parameterized regime (params ≫ data).

**24. Where's the double descent peak?**
At interpolation threshold (params ≈ data), test error spikes. Past that, decreases.

**25. Why does double descent happen?**
Implicit regularization (SGD finds particular interpolators), margin-based bounds, structure of overparameterized loss landscape.

**26. Lottery ticket hypothesis?**
Dense networks contain sparse subnetworks ("winning tickets") that, retrained from same init, match dense performance. Frankle & Carbin 2018.

**27. NTK — what is it?**
Neural Tangent Kernel. In infinite-width limit, deep networks behave like a kernel method with a specific kernel. Provides theoretical handle on generalization.

> **Saying it out loud.** Classically, capacity trades off against noise-chasing: too little and you underfit, too much and you fit the noise, giving the U-shaped test curve. Double descent says that U is only the left half of the picture. Test error peaks exactly at the interpolation threshold, where the model has just barely enough capacity to hit every training point and no freedom left to be sensible — then, as you keep growing, it falls again, often below the classical minimum. The explanations rhyme: among the many functions that fit the data perfectly, SGD implicitly selects a low-norm, flat one, the lottery-ticket view says a good sparse subnetwork was in there all along, and NTK says infinitely wide nets are just kernel machines. The number to state: the worst place to be is a model sized right at your training-set size.

---

## F. No-free-lunch and inductive bias

**28. State no-free-lunch.**
Averaged over all possible data distributions, all learning algorithms have the same expected performance.

**29. What does NFL imply?**
ML works because of *bias toward useful structure* in real data. Without inductive bias, no algorithm is universally better.

**30. Examples of inductive bias?**
Convolutions: locality and translation equivariance. Attention: content-based mixing. RNN: sequential. MLP: smooth. GBDT: hierarchical splits.

**31. Why do CNNs work for images?**
Inductive bias matches structure of natural images: local features, translation invariance, hierarchy.

**32. Why don't CNNs work as well for tabular?**
Tabular features don't have local spatial structure. GBDT inductive bias (axis-aligned splits) matches better.

> **Saying it out loud.** No free lunch says that averaged over all possible target functions, every learning algorithm does equally well — including random guessing. The load-bearing assumption is the word “all”: it puts uniform weight on every conceivable function, and real data is nowhere near uniform. Images are local and translation-invariant, language is compositional, tabular data is a pile of independently-scaled columns with no geometry. That's why learning works at all: an algorithm's inductive bias is a bet about structure, and it pays when the bet is right. Named tradeoff: bias buys you sample efficiency and costs you generality, which is exactly why CNNs beat MLPs on images with far less data and gradient-boosted trees still beat both on tabular.

---

## G. Regularization

**33. Regularization as inductive bias — explain.**
Regularizer adds preference for some functions over others. Equivalent to a prior in the Bayesian sense.

**34. $\ell_2$ regularization corresponds to which prior?**
Gaussian on weights.

**35. $\ell_1$ regularization corresponds to which prior?**
Laplace on weights → sparsity.

**36. Why does early stopping regularize?**
GD started from small weights stays close to them; small effective norm → regularization. Analogous to $\ell_2$.

**37. Data augmentation as regularization?**
Encodes invariance — model must be robust to specified transformations. Implicit prior.

> **Saying it out loud.** Regularization is you telling the model which functions you'd prefer before it sees the data, so in Bayesian language every regularizer is a prior — $\ell_2$ is a Gaussian prior on the weights, $\ell_1$ is a Laplace prior, which has that spike at zero and is why it produces genuine sparsity rather than just small weights. Data augmentation is a prior about invariance. Early stopping is subtler: gradient descent started near zero doesn't travel far, so cutting training short keeps the effective norm small, which is $\ell_2$ by another route. The point worth making is that in the over-parameterized regime a regularizer isn't shrinking capacity, it's *selecting which interpolating solution you land on* — and if you don't choose one explicitly, the optimizer chooses for you.

---

## H. Modern bounds

**38. PAC-Bayes idea?**
Bound generalization gap by $\mathrm{KL}(\mathrm{posterior} \| \mathrm{prior})$. Trained model = posterior; init = prior. Empirically gives nonvacuous bounds for deep nets.

**39. Stability-based generalization?**
If algorithm output is stable to small training set changes, it generalizes. SGD is approximately stable.

**40. Compression-based bounds?**
If a trained network compresses to few effective parameters, that's the relevant capacity. Lottery-ticket flavor.

> **Saying it out loud.** These are the bounds people actually reach for once VC and Rademacher go vacuous, and they share a strategy: stop measuring the hypothesis class and start measuring the trained model. PAC-Bayes bounds the gap by the KL divergence between the weight distribution you ended at and the one you started from — how far training had to move you — and that's the route to the first non-vacuous numerical bounds for real networks. Compression bounds say the relevant capacity is how small you could squeeze the trained net without hurting it. Stability bounds say if swapping one training example barely changes the output, the model can't be memorizing, and SGD with a small step size is provably about that stable. The tradeoff to name: each of these is tighter precisely because it depends on the algorithm and the data, so you trade distribution-free universality for a bound that means something.

---

## Quick fire

**41.** *VC linear classifier in $\mathbb{R}^d$?* $d+1$.
**42.** *Rademacher of linear with norm $B$?* $O(B/\sqrt{n})$.
**43.** *Sample complexity rate, agnostic?* $1/\epsilon^2$.
**44.** *NFL — implication?* Need inductive bias.
**45.** *Double descent location?* At interpolation.
**46.** *Lottery ticket?* Sparse subnetwork matching dense performance.
**47.** *NTK regime?* Infinite width.
**48.** *PAC stands for?* Probably Approximately Correct.
**49.** *VC for over-parameterized nets?* Vacuous bounds.
**50.** *Inductive bias of CNN?* Translation equivariance + locality.

---

## Self-grading

If you can't answer 1-15, you don't know SLT basics. If you can't answer 16-30, you'll struggle on capacity / generalization questions. If you can't answer 31-45, frontier-lab theory questions will go past you.

Aim for 30+/50 cold.
