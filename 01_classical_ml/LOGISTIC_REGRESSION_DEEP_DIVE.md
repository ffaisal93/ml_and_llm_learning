# Logistic Regression: A Frontier-Lab Interview Deep Dive

> **Why this document exists.** Logistic regression is the simplest model that has the richest theoretical structure: linear log-odds, MLE, calibration, regularization geometry, link functions, exponential family, Fisher information, multinomial extension, and direct connections to deep learning. Interviewers use it to probe whether you understand classification at first principles or just know how to call `LogisticRegression()`. A surprising number of senior-level offers turn on whether the candidate can answer five hard logistic-regression questions cleanly. The questions you said are hurting you live in this document.

---

## 1. What logistic regression actually models

Despite the name, logistic regression is not regression on labels — it is **a model for the conditional probability of a binary label given features**:

$$
P(y = 1 \mid x) = \sigma(w^\top x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Equivalently, it assumes the **log-odds** (logit) of $y = 1$ is a linear function of the features:

$$
\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = w^\top x + b
$$

This single assumption — that the **log-odds are linear in the features** — is the entire content of the model. Every statement about logistic regression's strengths, weaknesses, decision boundaries, and interview gotchas follows from this assumption.

**Interview test:** if someone asks you "what's the assumption of logistic regression?", do not say "it's a linear classifier." Say "the log-odds of the positive class are linear in the features." That's the precise statement.

---

## 2. Why log-odds?

Why do we transform $P$ to $\log(P/(1-P))$ before assuming linearity?

- **Range matching.** $P \in [0, 1]$ is bounded. $P/(1-P) \in [0, \infty)$. $\log(P/(1-P)) \in (-\infty, \infty)$. The log-odds match the range of a linear function $w^\top x + b$. We don't have to constrain weights to keep predictions in $[0, 1]$.
- **Symmetry.** $\operatorname{logit}(P) = -\operatorname{logit}(1-P)$. Swapping the class labels just flips the sign of the weights.
- **Connection to exponential families.** The Bernoulli distribution belongs to the exponential family with **natural parameter** equal to the log-odds. Logistic regression is, in this sense, "the natural" model for binary outcomes — analogous to linear regression as "the natural" model for Gaussian outcomes.

This last point is the deepest. Generalized linear models (GLMs) use a "link function" to connect the linear predictor $w^\top x + b$ to the conditional mean of the response. The **canonical link** for the Bernoulli is the logit. That's why the gradient is so clean (see §6).

---

## 3. Decision boundary: why is it linear?

The decision boundary is the set of points where the model is indifferent between classes:

$$
P(y=1 \mid x) = 0.5 \iff \sigma(w^\top x + b) = 0.5 \iff w^\top x + b = 0
$$

This is the equation of a hyperplane in $x$-space. So **the decision boundary is always linear in the input features**, no matter what threshold you choose (changing the threshold just shifts the boundary parallel to itself).

**This is the model's superpower and its limitation.** Logistic regression cannot represent any non-linear boundary in the input space. To capture non-linear patterns, you must engineer features (polynomials, interactions, basis functions, kernels) or use a different model.

**Interview gotcha.** Someone asks: "Can logistic regression separate non-linearly-separable data?" The strict answer is no — the model has a linear boundary in input space. The pragmatic answer is yes — you can engineer feature transforms ($x_1^2$, $x_1 \cdot x_2$, etc.) such that the data becomes linearly separable in the transformed space. But the model itself is still linear. This is exactly what kernel methods make explicit.

---

## 4. Why cross-entropy and not MSE?

This is one of the highest-frequency interview questions in classical ML. The strong answer is two-part: (a) MLE under Bernoulli gives cross-entropy as the negative log-likelihood, and (b) MSE with sigmoid is **non-convex** and has vanishing gradients.

### Part (a): MLE derivation

Treat each $y_i \in \{0, 1\}$ as drawn from $\operatorname{Bernoulli}(p_i)$ where $p_i = \sigma(w^\top x_i + b)$. The likelihood of the dataset is:

$$
L(w, b) = \prod_i p_i^{y_i} \cdot (1 - p_i)^{1 - y_i}
$$

Take the negative log:

$$
-\log L = -\sum_i \big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \big]
$$

This **is** the binary cross-entropy loss. Cross-entropy is not a design choice; it's what the data's likelihood assigns. Any other loss would correspond to a different (incorrect) generative assumption.

### Part (b): MSE with sigmoid is broken

Using $L = \frac{1}{N} \sum_i (\sigma(w^\top x_i) - y_i)^2$ — the obvious "regression-style" loss applied to classification — has two pathological properties:

**Non-convex.** The composition of MSE with sigmoid is non-convex in $w$. There can be multiple local minima, and gradient descent may get stuck. Cross-entropy on top of sigmoid is convex (more on this in §10).

**Vanishing gradients.** The MSE gradient with respect to $w$ is $2(\sigma(z) - y) \cdot \sigma'(z) \cdot x$, where $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. When the model is confidently wrong (say $\sigma(z) \approx 0.99$ but $y = 0$), $\sigma'(z) \approx 0.01$ is tiny, and the gradient is tiny. The model can't recover quickly from a confident wrong prediction. Cross-entropy's gradient (see §6) doesn't have this problem.

So: cross-entropy is correct by likelihood, convex, and gradient-friendly. MSE with sigmoid is wrong by likelihood, non-convex, and gradient-vanishing. There's no defensible reason to ever use MSE with sigmoid for classification.

---

## 5. The MLE derivation in full

You should be able to whiteboard this:

$$
\mathcal{L} = -\sum_i \big[ y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \big], \qquad z_i = w^\top x_i + b
$$

Useful identities:

$$
\log \sigma(z) = -\log(1 + e^{-z}), \qquad \log(1 - \sigma(z)) = -z - \log(1 + e^{-z})
$$

Substituting:

$$
\mathcal{L} = \sum_i \big[ -y_i z_i + \log(1 + e^{z_i}) \big]
$$

This is the "softplus" form, more numerically stable than implementing it directly.

**Gradient:**

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \sigma(z_i) - y_i
$$

$$
\nabla_w \mathcal{L} = \sum_i (\sigma(z_i) - y_i)\, x_i = X^\top (\sigma(Xw) - y)
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_i (\sigma(z_i) - y_i)
$$

**Hessian:**

$$
H = X^\top \operatorname{diag}\!\big(\sigma(z_i)(1 - \sigma(z_i))\big)\, X
$$

This is positive semi-definite, so the loss is convex.

A clean whiteboard derivation of these three things — the loss, the gradient, and the Hessian — is a very strong signal in interviews. The whole derivation is six lines.

---

## 6. Why the gradient is $(\sigma - y)\,x$ — the canonical link beauty

The gradient of the loss w.r.t. weights simplifies astonishingly:

$$
\nabla_w \mathcal{L} = X^\top (\sigma(Xw + b) - y)
$$

This is $X^\top (\hat y - y)$. Compare with linear regression:

$$
\nabla_w \mathcal{L} = X^\top (Xw + b - y) = X^\top (\hat y - y) \qquad \text{(same form!)}
$$

Both have the form "input matrix transposed times residual." This is **not a coincidence**. It happens because both are GLMs with their canonical link function:

- Gaussian distribution + identity link = linear regression.
- Bernoulli distribution + logit link = logistic regression.

For canonical-link GLMs, the gradient of the negative log-likelihood w.r.t. weights is always $X^\top (\hat\mu - y)$. This is one of the most beautiful results in classical statistics, and it's why these two models are so closely related despite solving different problems.

**Interview reward:** if you mention the canonical-link beauty when asked "why is the gradient so clean?", you stand out.

---

## 7. Newton's method and IRLS

Logistic regression has a non-trivial Hessian, which means Newton's method is feasible and fast. The Newton update is:

$$
w_{t+1} = w_t - H^{-1}\, g = w_t - \big(X^\top \operatorname{diag}(\sigma(1-\sigma))\, X\big)^{-1}\, X^\top (\sigma - y)
$$

This is **iteratively reweighted least squares** (IRLS): each iteration is a weighted least-squares problem where the weights are $\sigma(1-\sigma)$ for each sample. Convergence is typically quadratic — IRLS converges in 5–10 iterations for well-conditioned problems.

**When to use IRLS over SGD.** For small-to-medium datasets with $d < 10^4$, IRLS dominates SGD: fewer iterations, no LR to tune, deterministic. For large-scale problems IRLS becomes infeasible because of the $O(d^2)$ Hessian inversion, and SGD/L-BFGS are used instead. `sklearn.LogisticRegression` defaults to L-BFGS for this reason.

**Interview question:** "Why is logistic regression typically fit with second-order methods while neural networks aren't?" Answer: because logistic regression's Hessian has tractable structure (it's $X^\top D X$) and modest dimension; neural networks have $O(d^2)$ storage problems for $d$ in the millions or billions.

---

## 8. Convexity: why MLE is well-behaved

Cross-entropy with sigmoid (or, equivalently, the negative log-likelihood of Bernoulli + logit) is **convex** in $w$. Specifically:

$$
H = X^\top \operatorname{diag}\!\big(\sigma(z)(1 - \sigma(z))\big)\, X
$$

$\sigma(z)(1 - \sigma(z))$ is positive (it's a variance — Bernoulli with parameter $\sigma(z)$). So $H$ is a positive semi-definite weighted Gram matrix. Hessian PSD $\Rightarrow$ convex loss $\Rightarrow$ unique global minimum $\Rightarrow$ no local minima $\Rightarrow$ any optimization method converges to the same answer.

**Caveat.** $H$ is positive *semi*-definite, not strictly positive definite. If features are perfectly collinear or the data are perfectly separable (see §9), $H$ has a null space and the minimum is not unique.

**Interview question:** "Is logistic regression's loss strictly convex?" Strict convexity requires $H \succ 0$. This holds when $X^\top X \succ 0$, i.e. $X$ has full column rank, **and** $0 < \sigma(z_i) < 1$ for all $i$, i.e. no point is predicted with absolute certainty. In particular, if the data are linearly separable, $\sigma(z_i)$ can approach 0 or 1 for all training points and $H$ becomes singular at infinity.

---

## 9. Linear separability and the divergence problem

This is one of the most elegant gotchas in classical ML. **If the training data is linearly separable, MLE for logistic regression has no finite solution.**

Why? If you can find $(w, b)$ such that $w^\top x_i + b > 0$ for all positives and $< 0$ for all negatives, then scaling $(w, b)$ by $c \to \infty$ makes $\sigma(c \cdot z_i)$ approach 1 for positives and 0 for negatives. The likelihood becomes 1; the negative log-likelihood approaches 0; the optimum is at infinity. Numerically, the optimizer's weights blow up.

**This is why you should always include regularization for logistic regression.** L2 regularization adds $\frac{\lambda}{2} \|w\|^2$ to the loss, ensuring the optimum is finite even when the data are separable. This is also why `sklearn.LogisticRegression` defaults to L2 with $C = 1.0$.

**Interview question:** "What happens if you fit logistic regression to perfectly separable data with no regularization?" The strong answer: weights diverge, the optimizer never converges, predicted probabilities become extreme (0 or 1), and the model is useless. The cure is regularization or simpler models (which won't separate the data).

---

## 10. The connection to maximum entropy

A surprising and beautiful framing. Among all probability distributions $P(y \mid x)$ that:

1. Are valid probabilities (sum to 1, non-negative),
2. Match the empirical statistics of the data: $\mathbb{E}[y \cdot x] = \frac{1}{N} \sum_i y_i x_i$,

**the one with maximum entropy is exactly logistic regression.** This is sometimes called the "MaxEnt" or principle-of-maximum-entropy framing. It says: given the constraints in the data, logistic regression makes the *fewest additional assumptions* about the conditional distribution.

This is why logistic regression often works well in NLP and information retrieval — it's the entropy-maximizing distribution given linear constraints on features, which is a desirable property when you don't want to impose more structure than the data justifies.

---

## 11. Multinomial / softmax: the multiclass extension

For $K > 2$ classes:

$$
P(y = k \mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}} \qquad \text{(softmax)}
$$

This is $K$ linear functions, one per class, normalized by the softmax. **There is one redundant degree of freedom**: subtracting a constant from all $w_k$ doesn't change the probabilities. So in practice you can fix $w_K = 0$ (one class as the reference) without loss of generality, giving $K-1$ independent parameter vectors.

**Interview question:** "What's the relationship between logistic regression and softmax?" Logistic regression is the $K=2$ case of softmax. With two classes:

$$
P(y=1 \mid x) = \frac{e^{w_1^\top x}}{e^{w_0^\top x} + e^{w_1^\top x}}
= \frac{1}{1 + e^{-(w_1 - w_0)^\top x}}
= \sigma\!\big((w_1 - w_0)^\top x\big)
$$

The single weight vector $w$ in binary logistic regression is $w_1 - w_0$ from the multinomial parameterization.

**Interview gotcha.** "Are softmax outputs probabilities?" They sum to 1 and are non-negative, so technically yes. But **they are very poorly calibrated** in deep networks. A model that outputs $\operatorname{softmax} = [0.95, 0.05]$ is often wrong much more than 5% of the time. The probabilities are valid as relative scores; they may not be reliable as absolute probabilities. Calibration techniques (temperature scaling, Platt scaling) fix this.

---

## 12. Generative vs discriminative: logistic regression vs Naive Bayes

**Naive Bayes** (a generative classifier) models the joint distribution $P(x, y) = P(x \mid y) P(y)$ and uses Bayes' rule to derive $P(y \mid x)$. Under Gaussian features with shared covariance, the resulting $P(y \mid x)$ is exactly **logistic regression** in form — same sigmoid-of-linear structure. The difference is in the *fitting procedure*:

- **Naive Bayes** fits $P(x \mid y)$ and $P(y)$ separately by counting, then applies Bayes' rule.
- **Logistic regression** fits $P(y \mid x)$ directly by MLE.

**The trade-off.** Naive Bayes is biased (its model is wrong unless features are conditionally independent given $y$) but has lower variance — converges to its (wrong) limit fast. Logistic regression is unbiased (it's the right model in the limit) but has higher variance — needs more data.

**Ng & Jordan (2001)** showed: with infinite data, logistic regression dominates naive Bayes. With finite data, naive Bayes can win because of variance. Naive Bayes is often a strong baseline for small-data text classification for this reason.

**Interview question:** "When does Naive Bayes beat logistic regression?" Answer: small data, especially with high-dimensional features, where the bias of NB is offset by lower variance. Spam filtering on small training sets is a classic example.

---

## 13. The connection to SVM

Both logistic regression and (linear, hinge-loss) SVM are linear classifiers — they share the form $\operatorname{sign}(w^\top x + b)$. Where they differ:

- **Loss function.** Logistic regression: $\log(1 + e^{-y \cdot z})$. SVM: $\max(0, 1 - y \cdot z)$ (hinge).
- **Behavior on confidently-correct points.** Logistic regression keeps applying gradient ($\sigma - y$ is small but nonzero for correct, confident points). SVM's hinge loss is exactly zero for points outside the margin — those points contribute nothing to the gradient.
- **Probabilistic output.** Logistic regression gives calibrated (or close to it) probabilities. SVM gives only a score; you need Platt scaling to get probabilities.
- **Sensitivity to class balance.** Logistic regression's loss treats all examples equally; SVM's hinge loss is dominated by support vectors. SVM is more robust to class imbalance in some sense.
- **Kernels.** SVMs naturally extend to kernels via dual formulation; logistic regression's kernel extension exists but is less common in practice.

**Interview question:** "Loss-wise, what's the relationship?" Both are upper bounds on 0-1 loss. Hinge is sharper at the margin; logistic is smoother everywhere. Smoother loss $\to$ easier optimization $\to$ why logistic regression often wins in practice for non-margin-based reasons.

---

## 14. Coefficients: what does $w_j = 0.5$ actually mean?

This is the most basic but most mishandled interview question in classical ML.

$w_j = 0.5$ means: holding all other features fixed, **a one-unit increase in $x_j$ increases the log-odds of the positive class by 0.5**. Equivalently, it multiplies the odds by $e^{0.5} \approx 1.65$.

Notice what it does *not* mean:

- It does **not** mean the probability increases by 0.5.
- It does **not** mean the probability increases by 50%.
- It does **not** mean a multiplicative effect on the probability.

The coefficient is on the **log-odds scale**. The corresponding effect on probability depends on the base probability (it's biggest near 0.5, smallest near 0 or 1).

**Practical interpretation.** $e^{w_j}$ is the **odds ratio** for a one-unit increase in $x_j$. That's the quantity that translates directly to clinical risk reasoning, marketing decisions, etc.

**Interview question:** "Logistic regression coefficient for 'age' is 0.04. What does it mean?" Strong answer: each year of age increases the log-odds of the positive outcome by 0.04, multiplying the odds by $e^{0.04} \approx 1.04$, i.e. about a 4% relative increase in odds per year. The effect on probability depends on baseline.

---

## 15. Calibration: are probabilities reliable?

A model is **calibrated** if when it says "probability 0.7," the event happens 70% of the time. Calibration is **not** the same as accuracy; a model can be highly accurate but poorly calibrated.

### How to test calibration

**Reliability diagram.** Bin predictions into deciles. For each bin, compute the average predicted probability vs. the actual frequency of positives. Plot. A perfectly calibrated model is on the $y = x$ line.

**Brier score.** Mean squared error between predicted probabilities and outcomes:

$$
\text{Brier} = \frac{1}{N} \sum_i (p_i - y_i)^2
$$

Decomposes as $\text{calibration error} + \text{refinement error}$. Lower is better.

**Expected Calibration Error (ECE).** Weighted average distance between bin frequencies and bin probabilities:

$$
\text{ECE} = \sum_b \frac{|\text{bin}_b|}{N} \, \big| \operatorname{freq}(\text{bin}_b) - \overline{p}(\text{bin}_b) \big|
$$

### Why calibration matters

- **Medical / financial decisions.** Threshold-based decisions need accurate probabilities, not just rankings.
- **Cost-sensitive prediction.** Expected cost is $(1-p) \cdot \text{FP\_cost} + p \cdot \text{FN\_cost}$; needs reliable $p$.
- **Ensembling.** Combining multiple models requires comparable confidences.

### Calibration of logistic regression

**Logistic regression is usually well-calibrated** if the model is reasonably specified. This is one reason it's still used in heavily regulated industries (insurance, credit) — interpretability + calibration. Modern neural networks are notoriously poorly calibrated despite high accuracy; this is part of why people add temperature scaling on top.

**Interview question:** "How do you check calibration?" Reliability diagrams, Brier score, ECE. "How do you fix miscalibration?" Platt scaling, isotonic regression, temperature scaling.

---

## 16. Class imbalance: what changes

Logistic regression with cross-entropy treats all examples equally. With heavy imbalance (say 99:1), the optimizer essentially focuses on getting the majority class right and ignores the minority. The model still produces probabilities, but the *threshold* for converting probability to label is no longer 0.5.

### Three legitimate fixes

**1. Adjust the decision threshold.** The model's calibration may be fine; the default threshold of 0.5 is wrong. Choose a threshold based on the desired precision/recall trade-off. **This is usually the right first move.**

**2. Class weights / loss reweighting.**

$$
\mathcal{L} = -\frac{1}{N} \sum_i w_{y_i} \cdot \big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \big]
$$

Up-weight the minority class. This is a soft form of resampling. `sklearn` exposes this as `class_weight='balanced'`.

**3. Resampling.** Oversample the minority (SMOTE or simple replication) or undersample the majority. This is more aggressive and can introduce bias if not done carefully.

### What does NOT help

**Synthetic feature engineering.** Adding "is_minority_class" or similar to the features is a leak.

**Using accuracy as the metric.** With 99:1 imbalance, predicting all-majority gets 99% accuracy. Use AUROC, AUPRC, F1, or balanced accuracy instead.

**Interview question:** "Your model achieves 99% accuracy on 99:1 imbalanced data. What's wrong?" The trap. "It's predicting all-majority. Switch to AUPRC or F1." Then discuss thresholding and class weights.

---

## 17. Regularization: L1 vs L2 geometry

### L2 (ridge)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|w\|^2
$$

L2 shrinks weights toward zero proportionally. Geometrically, L2's level sets are circles (in 2D) — smooth, isotropic. The penalized minimum lies on the contour line of the loss tangent to a circle of constant $\|w\|^2$. **Coefficients are shrunk but rarely exactly zero.** L2 produces dense weights.

### L1 (lasso)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \|w\|_1
$$

L1's level sets are diamonds. The penalized minimum often lies at a *corner* of the diamond, where one or more coefficients are exactly zero. **L1 produces sparse weights** — automatic feature selection.

### When to use which

- **L2 default.** Shrinks all coefficients smoothly; well-conditioned numerically; doesn't destroy signal.
- **L1 for feature selection.** When you suspect most features are irrelevant and want the model to discover which ones matter.
- **Elastic net.** $\alpha \|w\|_1 + (1 - \alpha)\,\frac{\|w\|^2}{2}$ — combines both. Good when features are correlated (L1 alone arbitrarily picks one of a correlated pair; elastic net smooths this).

**Interview gotcha.** "L1 gives sparse solutions because the gradient of $|w|$ doesn't vanish at zero." Almost right. The deeper reason is geometric: the diamond has corners where the projection lands, and corners coincide with sparse weight vectors.

---

## 18. Multicollinearity

If two features are highly correlated, logistic regression's coefficient estimates become unstable: small changes in data flip large amounts of weight between the two features. The Hessian becomes nearly singular, standard errors blow up, and individual coefficient interpretations become unreliable.

**Symptoms:**

- Unreasonable coefficient signs (e.g., a feature you'd expect to be positive comes out negative).
- Huge standard errors on coefficients that "should" matter.
- Predictions are stable, but coefficients aren't.

**Diagnostics:**

- Variance inflation factor (VIF). VIF $> 10$ is a warning sign.
- Condition number of $X^\top X$.

**Fixes:**

- Drop redundant features.
- Use L2 regularization (which mathematically eliminates the singularity).
- Use PCA or other dimensionality reduction.

**Interview question:** "Two of my features are correlated; what happens?" The model's predictions are fine. The individual coefficients are not interpretable. Use L2 or drop one feature.

---

## 19. Probit vs logit

Logistic regression uses the logit link: $P = \sigma(z)$. **Probit regression** uses the probit link: $P = \Phi(z)$, where $\Phi$ is the standard normal CDF.

The two are extremely similar:

- Both are monotonic, S-shaped, between 0 and 1.
- The probit has slightly lighter tails than logit.
- They're approximately related by a scale factor: $\Phi(x) \approx \sigma(1.6 x)$ near the center.

**When does probit appear?**

- Statistics literature, especially in econometrics (the Heckman selection model and similar).
- Cases where Gaussian latent-variable interpretation matters (probit comes from $y^* = w^\top x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,1)$, $y = \mathbf{1}[y^* > 0]$).

**In ML practice:** logit dominates because (a) the gradient is cleaner, (b) numerical stability is better, (c) the canonical-link beauty applies. Mention probit's existence and move on.

---

## 20. Logistic regression as a one-layer neural network

This is the cleanest framing for connecting classical ML to deep learning.

$$
P(y=1 \mid x) = \sigma(w^\top x + b)
$$

is exactly **a single neuron with sigmoid activation**. The loss (cross-entropy) is exactly **the loss used in the final layer of binary classifiers**. When you train a deep neural network for binary classification, you are training a hierarchy of feature extractors that feed into a logistic regression in the final layer.

This means:

- Everything that breaks logistic regression breaks the final layer of a NN classifier (separability, miscalibration, threshold choice).
- The "softmax + cross-entropy" output layer of multi-class NNs is **multinomial logistic regression** on top of learned features.
- Logistic regression is the *natural baseline* against which any classifier should be benchmarked. If a fancy NN doesn't beat well-tuned logistic regression with reasonable feature engineering, the NN is overfitting.

**Interview question:** "What's the relationship between logistic regression and neural networks?" Strong answer: logistic regression is the special case of a NN with no hidden layers. The final layer of any binary classifier NN is logistic regression on the learned representation. Multinomial logistic regression is the same for multi-class.

---

## 21. Practical deployment and serving

A few things real-world MLE interviews probe:

**1. Coefficient stability over time.** Logistic regression coefficients can drift if the data distribution drifts. Monitor the calibration on holdout data; recalibrate or retrain when reliability diagrams degrade.

**2. Online learning.** Logistic regression admits efficient online updates (it's a GLM). One pass of SGD per new sample is exact in the limit. This makes it a strong choice for streaming applications where retraining is expensive.

**3. Interpretability for compliance.** In credit, insurance, healthcare, the model must be explainable. Logistic regression's coefficients are directly interpretable; this is why it's still the primary model in regulated industries despite the existence of better black-box alternatives.

**4. Latency.** Logistic regression inference is $O(d)$ per prediction — a single dot product. For real-time bidding (sub-millisecond budgets), it's often the only feasible option.

---

## 22. Common interview traps (cheatsheet)

| Trap | Strong answer |
|---|---|
| "It's a regression" | Despite the name, it models conditional probability of a binary outcome via the logit link. |
| "Why sigmoid?" | Logit is the canonical link for Bernoulli; gradient simplifies; range-matched to linear predictor; max-entropy interpretation. |
| "Is it linear or non-linear?" | Linear in the log-odds; the boundary in input space is a hyperplane. Non-linear patterns require feature engineering. |
| "Why CE not MSE?" | MLE under Bernoulli gives CE. MSE+sigmoid is non-convex and has vanishing gradients on confidently-wrong predictions. |
| "What if data is separable?" | Weights diverge; MLE has no finite solution. Always regularize. |
| "Coefficient interpretation?" | Log-odds change per unit feature change; $e^{w_j}$ is the odds ratio. Never "probability change". |
| "What does multicollinearity do?" | Predictions OK; coefficient interpretations unreliable. Use L2 or drop features. |
| "Calibration?" | Logistic regression is usually well-calibrated. Check via reliability diagram, Brier, ECE. Calibrate via Platt or isotonic. |
| "Imbalance?" | Adjust threshold first; class weights / resampling second. Don't use accuracy. |
| "L1 vs L2?" | L1 = sparse (corners of diamond); L2 = shrinkage (smooth). Elastic net for correlated features. |
| "Connection to NN?" | Single-layer NN with sigmoid; the output layer of any binary classifier. |
| "Connection to softmax?" | Binary case of multinomial logistic regression. |
| "Connection to NB?" | Same functional form under Gaussian conditional features; LR = discriminative, NB = generative. NB wins on small data. |
| "Connection to SVM?" | Both linear classifiers; logistic loss is smooth, hinge loss is sharper at margin. |

---

## 23. Recommended drill plan

1. **Whiteboard the MLE derivation** end-to-end (loss → gradient → Hessian) until you can do it in 4 minutes without notes.
2. **State and defend the linear-log-odds assumption** in 60 seconds.
3. **Explain why the gradient is $(\sigma - y)\,x$** including the canonical-link beauty.
4. **Defend cross-entropy over MSE** in two complementary ways (likelihood and convexity / vanishing gradients).
5. **Explain separability and divergence** plus the regularization fix.
6. **Explain the connection to softmax, NN, NB, SVM** — one sentence each.
7. **Run through `LOGISTIC_REGRESSION_INTERVIEW_GRILL.md`** until 40+/50 cold.

---

## 24. Further reading

- Ng & Jordan, "On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes" (2001).
- Murphy, *Probabilistic Machine Learning: An Introduction*, Chapter 10.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 4.
- Bishop, *Pattern Recognition and Machine Learning*, Chapter 4 (logistic regression and IRLS).
- Friedman et al., "Regularization Paths for Generalized Linear Models via Coordinate Descent" (2010) — the algorithm behind `glmnet`.

If you internalize this document, logistic regression stops being a simple model you "already know" and becomes a window into the entire mathematical structure of supervised classification.
