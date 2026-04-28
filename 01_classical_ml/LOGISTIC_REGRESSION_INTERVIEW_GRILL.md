# Logistic Regression — Interview Grill

> 50 brutal questions on logistic regression. The point is not whether you can recognize the answer — it's whether you can produce it cleanly under interview pressure. Cover answers, speak yours out loud, then check.

---

## A. Foundations (must-have)

**1. What does logistic regression model?**
The conditional probability of a binary label given features. Specifically, it assumes the **log-odds** of the positive class are linear in the features:

$$
\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = w^\top x + b
$$

Equivalently, $P(y=1 \mid x) = \sigma(w^\top x + b)$.

**2. What's the assumption of logistic regression?**
The log-odds of the positive class are linear in the features. **Not** "the data is linearly separable" or "features are normally distributed" — those are not assumptions of logistic regression.

**3. Why "regression" if it does classification?**
Historical naming. The model regresses the log-odds (a real number) on the features, even though the goal is classification. The output is a probability that you threshold to get a class label.

**4. Why sigmoid?**
You need to squash a real number into $[0, 1]$ — sigmoid is the natural smooth choice. The deeper reason: it's the canonical inverse-link of the Bernoulli GLM, which gives the cleanest gradient: $(\sigma - y)\,x$ — "predicted minus actual, times input."

**5. Derive the loss function from MLE.**
Bernoulli likelihood: $L = \prod_i p_i^{y_i} (1-p_i)^{1-y_i}$. Negative log-likelihood: $-\sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]$. This is binary cross-entropy. It's not a design choice; it's what likelihood under Bernoulli mandates.

**6. Why not MSE?**
Two reasons. (a) MLE under Bernoulli gives cross-entropy, not MSE — different generative assumption. (b) MSE composed with sigmoid is non-convex and has vanishing gradients when the model is confidently wrong ($\sigma'(z) \to 0$ for large $|z|$). Cross-entropy is convex and gradient-friendly.

**7. Compute the gradient.**
$\nabla_w \mathcal{L} = X^\top (\sigma(Xw + b) - y)$. Form: input matrix transposed times residual. Same form as linear regression because both are GLMs with their canonical link.

**8. Compute the Hessian.**
$H = X^\top \operatorname{diag}(\sigma(z)(1 - \sigma(z))) X$. Positive semi-definite, so the loss is convex. Strictly positive definite if $X$ has full column rank and no point is perfectly classified.

**9. Is the loss convex?**
Yes. Hessian is PSD as a weighted Gram matrix $X^\top D X$ with $D$ diagonal, positive entries. Strict convexity needs $X$ full rank and $0 < \sigma(z) < 1$ for all data points (so no perfectly-separable case).

**10. What's the decision boundary?**
The hyperplane $w^\top x + b = 0$. Linear in the input space, regardless of threshold. Changing the threshold shifts the boundary parallel to itself.

---

## B. Behavior, gotchas, edge cases

**11. What happens if the data is perfectly linearly separable?**
MLE has no finite solution. Weights diverge to infinity (you can always increase likelihood by scaling $w$ by a larger constant). In practice, you must add regularization (L2 is standard) — without it, the optimizer never converges and predicted probabilities collapse to 0 or 1.

**12. Logistic regression gives probability 0.99 — should I trust it?**
Depends on calibration. A well-fit logistic regression on representative data is usually well-calibrated. But check: bin the predictions and see what fraction of "0.99-confident" predictions are actually correct. If it's ~99%, trust it. If it's 80%, the model is overconfident — recalibrate (Platt or isotonic) or retrain.

**13. What does coefficient $w_j = 2.0$ mean?**
A one-unit increase in $x_j$, holding all other features constant, increases the log-odds of the positive class by 2.0 — i.e. multiplies the odds by $e^2 \approx 7.4$. The change in probability depends on the baseline.

**14. What does $w_j = -0.5$ mean for a binary feature?**
Switching $x_j$ from 0 to 1 decreases the log-odds by 0.5, multiplying the odds by $e^{-0.5} \approx 0.61$. The presence of that binary feature is associated with lower odds of $y=1$, all else equal.

**15. My coefficient is huge but the feature looks unimportant. What's wrong?**
Multicollinearity. When two features are nearly redundant, the Hessian becomes nearly singular and individual coefficients explode (one big positive offset by a big negative on the correlated partner). Predictions stay fine; coefficients aren't interpretable. Use L2 or drop one feature.

**16. Multicollinearity diagnostic?**
Variance inflation factor (VIF). $\text{VIF} = 1/(1 - R^2)$ for that feature regressed on the others. VIF > 10 is a warning. Also: condition number of $X^\top X$.

**17. What does L2 do to the optimization?**
Adds $(\lambda/2) \|w\|^2$ to the loss. Hessian becomes $X^\top D X + \lambda I$, which is strictly positive definite for any $\lambda > 0$, even if features are collinear or data are separable. Guarantees a unique finite solution.

**18. Why does L1 give sparse weights?**
Geometric: L1's level sets are diamonds with corners at the axes. The penalized minimum often lies at a corner, which corresponds to a coordinate being exactly zero. L2's level sets are circles — no corners, no exact zeros.

**19. When would you choose L1 over L2?**
When you suspect most features are irrelevant and want feature selection. When interpretability requires a sparse model. When $d \gg N$.

**20. Logistic regression on an imbalanced dataset (99:1)?**
First, don't use accuracy as the metric — it'll show 99% by predicting all-majority. Use AUPRC, F1, or balanced accuracy. Adjust the threshold (away from 0.5). Optionally use class weights or resampling. The model itself isn't broken; default decisions are.

**21. How would you choose a threshold for an imbalanced problem?**
Define the cost trade-off: false positives vs false negatives, business value. Plot precision/recall vs threshold. Pick the operating point. Commonly: maximize F1 (if you want balance) or fix recall and minimize FPR (if you want guaranteed sensitivity).

---

## C. Beyond binary

**22. Generalize to K classes — what's the model?**
Multinomial logistic regression / softmax:

$$
P(y = k \mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}
$$

$K$ weight vectors with one redundant degree of freedom (subtracting a constant from all $w_k$ doesn't change probabilities). Often parameterized with $w_K = 0$ reference class.

**23. Show that binary logistic regression is the K=2 case of softmax.**
With two classes:

$$
P(y=1 \mid x) = \frac{e^{w_1^\top x}}{e^{w_0^\top x} + e^{w_1^\top x}} = \frac{1}{1 + e^{-(w_1 - w_0)^\top x}} = \sigma\!\big((w_1 - w_0)^\top x\big)
$$

The single weight in binary LR is the difference $w_1 - w_0$.

**24. Loss function for multinomial?**
Categorical cross-entropy: $-\sum_i \sum_k \mathbf{1}[y_i = k] \log P(y = k \mid x_i) = -\sum_i \log P(y = y_i \mid x_i)$. MLE under categorical distribution.

**25. Are softmax outputs reliable probabilities?**
Mathematically yes (they're a valid distribution). Practically, in logistic regression on tabular data — usually yes if the model is well-specified. In **deep neural networks** — usually no, they're poorly calibrated despite high accuracy. Temperature scaling or post-hoc calibration is standard.

---

## D. Deeper theory

**26. Why is the gradient $(\sigma - y)\,x$ and not something messier?**
**It's literally "predicted minus actual, times input"** — the cleanest possible form. Same shape as linear-regression gradient. This is the GLM canonical-link beauty: for *any* GLM with its canonical link (Gaussian+identity, Bernoulli+logit, Poisson+log), $\partial \text{NLL}/\partial w = X^\top (\hat\mu - y)$. The mess from sigmoid's derivative cancels with the mess from CE's $1/p$ exactly.

**27. What is Iteratively Reweighted Least Squares (IRLS)?**
Newton's method applied to logistic regression. Each iteration solves a weighted least squares problem with weights $\sigma(z)(1-\sigma(z))$. Converges quadratically; typically 5-10 iterations. Used by classical statistics packages (R's `glm`, `statsmodels`).

**28. Why does sklearn use L-BFGS by default for logistic regression?**
L-BFGS is a quasi-Newton method that approximates the Hessian without storing it explicitly. It's nearly as fast as IRLS for medium-sized problems, scales better to large $d$, and is more robust numerically. For $d$ in the millions, neither IRLS nor L-BFGS are great; SGD is used.

**29. What's the relationship between logistic regression and Naive Bayes?**
Both can produce the same functional form ($\sigma$ of linear predictor) under specific conditional-feature assumptions (Gaussian features with shared covariance for NB). The key difference is fitting: NB is generative, fitting $P(x \mid y) P(y)$ separately; LR is discriminative, fitting $P(y \mid x)$ directly.

**30. When does Naive Bayes beat logistic regression?**
Small data. Ng & Jordan (2001) showed: with infinite data, LR dominates; with finite data, NB can win because of lower variance. NB converges faster to its (biased) limit. Common in small-data text classification.

**31. What's the relationship to SVM?**
Both are linear classifiers. LR uses logistic loss $\log(1 + e^{-y \cdot z})$; SVM uses hinge loss $\max(0, 1 - y \cdot z)$. Hinge is exactly zero outside the margin (sparse contributions); logistic is smooth everywhere. LR gives calibrated probabilities; SVM gives only a score (Platt-scaled for probs).

**32. Show that softmax + cross-entropy gradient is `softmax − one-hot`.**
Let $s_i = e^{w_i^\top x} / \sum_j e^{w_j^\top x}$. Loss $\mathcal{L} = -\log s_y$. After working through the softmax derivative, the result is $\partial \mathcal{L}/\partial z_k = s_k - \mathbf{1}[k = y]$, so $\partial \mathcal{L}/\partial w_k = (s_k - \mathbf{1}[k = y]) \cdot x$. This generalizes the binary $(\sigma - y)\,x$ gradient.

**33. What's the maximum entropy interpretation of logistic regression?**
**It's the least-assuming distribution that fits what we've seen.** Formally: among all conditional distributions $P(y|x)$ that match the empirical feature-label moments, logistic regression is the one with maximum entropy — most uniform, fewest extra assumptions. This is dual to the GLM/canonical-link view.

**34. Explain calibration and how to test it.**
Calibration = predicted probabilities match observed frequencies. Test with reliability diagrams (bin predictions, plot mean predicted vs observed frequency), Brier score (MSE between probs and outcomes), and ECE (weighted bin error).

**35. How do you fix miscalibration?**
Platt scaling: fit a 1D logistic regression $P_{\text{calibrated}} = \sigma(a \cdot \text{score} + b)$ on a held-out set. Isotonic regression: non-parametric monotonic mapping. Temperature scaling (NN-specific): divide logits by $T > 0$ before softmax, fit $T$ on validation.

---

## E. Connections to deep learning

**36. Logistic regression is a single-layer neural network — explain.**
It's $\sigma(w^\top x + b)$. That's exactly one fully-connected layer with one output neuron and sigmoid activation. The cross-entropy loss is exactly the BCE loss in PyTorch. Multinomial LR with softmax is exactly the standard final layer of a multi-class neural classifier.

**37. What does this imply for the final layer of any classifier NN?**
It's logistic / multinomial logistic regression on top of learned features. Everything that affects logistic regression — calibration issues, separability, regularization — applies to that final layer. The "deep" part is feature learning; the "classification" part is unchanged.

**38. Why do deep networks miscalibrate but logistic regression doesn't?**
NNs overfit confidence. With high capacity and many parameters, NNs can drive training cross-entropy near zero by pushing logits to extreme values, even when validation accuracy plateaus. Result: predicted probabilities concentrate at 0 and 1 even when the model is uncertain. LR's lower capacity makes it harder to overfit confidence the same way.

**39. Is dropout useful in logistic regression?**
Generally not. Dropout's value is in deep networks where it prevents co-adaptation of hidden units. LR has no hidden units; there's nothing to drop out. Use L2 or L1 for regularization.

---

## F. Practical engineering

**40. How do you decide which features matter?**
For a fitted L2 model: standardize features first, then compare absolute coefficient magnitudes (only meaningful if scales are comparable). For L1 model: zero coefficients are explicitly excluded. Better: permutation importance — shuffle a feature, measure performance drop.

**41. Should you standardize features for logistic regression?**
**Yes, if you're using regularization**, because L1/L2 penalties depend on coefficient magnitude, which depends on feature scale. Without standardization, regularization unfairly penalizes high-magnitude features. Without regularization, the math doesn't care about scale (only convergence speed for SGD).

**42. How do you handle missing features?**
Imputation (mean, median, model-based) is standard. Or treat missingness as an indicator: add a binary "is_missing" feature. Don't drop rows unless missingness is rare and missing-completely-at-random.

**43. Categorical features — how?**
One-hot encode (drop one level as reference, or include all if you have a regularization penalty to prevent the redundancy from causing infinite weights). For high-cardinality (e.g. zip codes), use target encoding, hash trick, or embeddings. Beware of leak in target encoding.

**44. Online updates — how does logistic regression handle streaming data?**
SGD update per new sample is $w \leftarrow w + \eta \cdot (y - \sigma(w^\top x)) \cdot x$. Single-pass online learning is principled (it's MLE in the streaming regime). Strong choice for streaming applications. Common in ad ranking, online recommendation.

**45. Latency requirements — when is logistic regression preferred?**
Sub-millisecond inference budgets (real-time bidding, online recommendation). Inference is $O(d)$ — a single dot product. No matrix multiplications, no GPU needed. Often the only feasible model for tight latency budgets.

**46. Interpretability — when does LR win over a tree?**
Regulated industries (credit, insurance, healthcare): you must explain individual predictions. LR's coefficients give clean, additive explanations. Trees are *locally* interpretable but *globally* messy. LR's calibration also matters here — credit scoring needs reliable probabilities.

---

## G. Probit and other GLMs

**47. What's probit regression?**
GLM with the probit link $\Phi^{-1}$ (inverse standard normal CDF) instead of logit. Functional form: $P(y=1 \mid x) = \Phi(w^\top x + b)$. Used in econometrics; rarely in ML practice.

**48. Logit vs probit — does it matter?**
Empirically, almost never. Both are S-shaped, both are between 0 and 1, predictions agree on most data within rescaling. Logit dominates in ML because the gradient is cleaner and the canonical-link beauty applies.

**49. Given a Poisson outcome, what's the GLM?**
Poisson regression with log link: $\log(\lambda) = w^\top x + b$. The canonical link for the Poisson is the log. Same $X^\top (\hat\mu - y)$ gradient form. Used for count data (clicks, events, etc.).

**50. Walk me through the canonical-link beauty.**
For exponential-family distributions, the negative log-likelihood with canonical link gives $\partial \text{NLL}/\partial w = X^\top (\hat\mu - y)$. The Hessian is $X^\top \operatorname{diag}(V(\hat\mu)) X$ where $V$ is the variance function. This is why linear, logistic, and Poisson regressions all share the form "$X^\top \cdot \text{residual}$" — they're all GLMs with their canonical link. This is one of the deepest unifying results in classical statistics.

---

## Quick-fire (under 10 seconds each)

**51.** *Loss for binary LR?* Binary cross-entropy.
**52.** *Default sklearn solver?* L-BFGS.
**53.** *Default regularization?* L2 with $C = 1.0$.
**54.** *What happens at perfect separation without regularization?* Weights diverge.
**55.** *What's $e^\beta$?* Odds ratio for unit feature change.
**56.** *Calibrated probabilities?* Usually yes for LR.
**57.** *L1 produces?* Sparse weights.
**58.** *Multinomial loss?* Categorical cross-entropy.
**59.** *Connection to softmax?* LR is binary case.
**60.** *Connection to neural net?* Single-layer with sigmoid.

---

## Self-grading

If you can't answer 1–10 cold, you don't know logistic regression. If you can't answer 11–25, you don't have the depth to defend it in interviews. If you can't answer 26–50, you'll struggle when an interviewer goes deeper than the textbook.

Aim for 40+/60 cold before any classical-ML interview.
