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

---

## B. PAC learning

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

---

## C. VC dimension

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

---

## D. Rademacher complexity

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

---

## H. Modern bounds

**38. PAC-Bayes idea?**
Bound generalization gap by $\mathrm{KL}(\mathrm{posterior} \| \mathrm{prior})$. Trained model = posterior; init = prior. Empirically gives nonvacuous bounds for deep nets.

**39. Stability-based generalization?**
If algorithm output is stable to small training set changes, it generalizes. SGD is approximately stable.

**40. Compression-based bounds?**
If a trained network compresses to few effective parameters, that's the relevant capacity. Lottery-ticket flavor.

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
