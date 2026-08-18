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

> **Saying it out loud.** It models the probability of the positive class, not the class itself. The trick is that a probability lives between 0 and 1, which is awkward to fit a line to, so instead you model the log-odds — the log of the ratio of the two probabilities — and *that* is what you assume is linear in the features. Log-odds run from minus infinity to plus infinity, so a linear function is a perfectly sensible thing to put there. Then you push it back through a sigmoid to get a probability out.

**2. What's the assumption of logistic regression?**
The log-odds of the positive class are linear in the features. **Not** "the data is linearly separable" or "features are normally distributed" — those are not assumptions of logistic regression.

> **Saying it out loud.** The only real assumption is that the log-odds are linear in your features. That's it. The things people wrongly list — that the data has to be linearly separable, that features are normally distributed, that errors are Gaussian — are none of them required, and separability is actually a *problem* rather than a requirement. The one thing you do need beyond linearity is that observations are independent, and that's where a lot of real-world misuse happens, with repeated measurements on the same user.

**3. Why "regression" if it does classification?**
Historical naming. The model regresses the log-odds (a real number) on the features, even though the goal is classification. The output is a probability that you threshold to get a class label.

> **Saying it out loud.** It's a naming accident from statistics. The model really is doing a regression — it fits a linear function to a continuous quantity, the log-odds — it's just that afterwards you squash that number into a probability and threshold it to get a class. So it's a regression model being used as a classifier. Worth saying because it signals you understand the model outputs a probability, and that the classification step is a decision you make on top, not part of the model.

**4. Why sigmoid?**
You need to squash a real number into $[0, 1]$ — sigmoid is the natural smooth choice. The deeper reason: it's the canonical inverse-link of the Bernoulli GLM, which gives the cleanest gradient: $(\sigma - y)\,x$ — "predicted minus actual, times input."

> **Saying it out loud.** Because you need something that maps any real number into a valid probability, smoothly and monotonically, and sigmoid is the natural one. But the better answer is that it isn't an arbitrary choice — if you say 'the log-odds are linear', then algebraically inverting that statement *gives* you the sigmoid. It's the canonical inverse-link for a Bernoulli outcome, and the payoff is the gradient: everything ugly cancels and you're left with predicted minus actual, times the input.

**5. Derive the loss function from MLE.**
Bernoulli likelihood: $L = \prod_i p_i^{y_i} (1-p_i)^{1-y_i}$. Negative log-likelihood: $-\sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]$. This is binary cross-entropy. It's not a design choice; it's what likelihood under Bernoulli mandates.

> **Saying it out loud.** You don't pick the loss, you derive it. Assume each label is a coin flip with probability $p_i$ from your model, write down the probability of the labels you actually observed, and that's a product of $p_i$ for the ones and $1-p_i$ for the zeros. Take the log to turn the product into a sum, flip the sign because optimizers minimize, and what falls out is exactly binary cross-entropy. So cross-entropy isn't a design decision — it's what maximum likelihood under a Bernoulli hands you.

**6. Why not MSE?**
Two reasons. (a) MLE under Bernoulli gives cross-entropy, not MSE — different generative assumption. (b) MSE composed with sigmoid is non-convex and has vanishing gradients when the model is confidently wrong ($\sigma'(z) \to 0$ for large $|z|$). Cross-entropy is convex and gradient-friendly.

> **Saying it out loud.** Two reasons, and the second one is the one interviewers want. First, MLE under a Bernoulli gives you cross-entropy, so MSE is answering a question about Gaussian noise that doesn't apply to a 0/1 label. Second, and more practically, MSE composed with a sigmoid is non-convex, and worse, its gradient contains a $\sigma'(z)$ factor that goes to zero when the model is confidently wrong. So the very examples you most need to learn from produce almost no gradient — the model gets stuck being confidently wrong. Cross-entropy's gradient is linear in the error and doesn't have that failure mode.

**7. Compute the gradient.**
$\nabla_w \mathcal{L} = X^\top (\sigma(Xw + b) - y)$. Form: input matrix transposed times residual. Same form as linear regression because both are GLMs with their canonical link.

> **Saying it out loud.** The gradient is the data matrix transposed times the residual — predicted probability minus actual label. That's it, and it's worth pausing on how clean that is: it's the exact same form as linear regression's gradient, just with a sigmoid inside the prediction. The reason is that the sigmoid's derivative and the cross-entropy's derivative are reciprocals and cancel exactly. If you ever derive a gradient for a GLM and it doesn't come out as 'design matrix transpose times residual', you've made an algebra mistake.

**8. Compute the Hessian.**
$H = X^\top \mathrm{diag}(\sigma(z)(1 - \sigma(z))) X$. Positive semi-definite, so the loss is convex. Strictly positive definite if $X$ has full column rank and no point is perfectly classified.

> **Saying it out loud.** The Hessian is $X^\top D X$, where $D$ is diagonal with entries $p(1-p)$ — the variance of a Bernoulli at that point's predicted probability. What's nice is the interpretation: $p(1-p)$ is largest at $p = 0.5$ and vanishes as $p$ goes to 0 or 1, so points the model is confident about contribute almost no curvature. That's a sandwich of a positive diagonal between $X^\top$ and $X$, which is automatically positive semi-definite — and that's your proof of convexity in one line.

**9. Is the loss convex?**
Yes. Hessian is PSD as a weighted Gram matrix $X^\top D X$ with $D$ diagonal, positive entries. Strict convexity needs $X$ full rank and $0 < \sigma(z) < 1$ for all data points (so no perfectly-separable case).

> **Saying it out loud.** Yes, and it's a one-line proof. The Hessian is $X$-transpose times a positive diagonal times $X$, and anything of that form is positive semi-definite, so the loss is convex — which means no local minima, any optimizer that converges converges to the global answer. Strict convexity needs a bit more: $X$ full column rank, so no collinear features, and no point predicted at exactly 0 or 1, which is the separable case. That last condition is why perfect separation breaks the model — you lose strict convexity and the optimum runs off to infinity.

**10. What's the decision boundary?**
The hyperplane $w^\top x + b = 0$. Linear in the input space, regardless of threshold. Changing the threshold shifts the boundary parallel to itself.

> **Saying it out loud.** It's a hyperplane — the set of points where $w^\top x + b$ equals zero, which is exactly where the predicted probability is 0.5. The thing that surprises people is that changing your threshold doesn't bend the boundary, it just slides the same flat hyperplane sideways, because a threshold on the probability is equivalent to a threshold on a linear function. So logistic regression is a linear classifier no matter what threshold you pick. If you want a curved boundary you have to put the curvature in the features.

---

## B. Behavior, gotchas, edge cases

**11. What happens if the data is perfectly linearly separable?**
MLE has no finite solution. Weights diverge to infinity (you can always increase likelihood by scaling $w$ by a larger constant). In practice, you must add regularization (L2 is standard) — without it, the optimizer never converges and predicted probabilities collapse to 0 or 1.

> **Saying it out loud.** The fit blows up — there's no finite maximum-likelihood solution. Intuitively, if a hyperplane separates the classes perfectly, then doubling all the weights keeps the same boundary but makes every prediction more confident, which strictly increases the likelihood. So the optimizer keeps scaling the weights forever, and you get coefficients in the hundreds and probabilities pinned at 0 and 1. It's a common gotcha with high-dimensional or one-hot-heavy data, and the fix is any amount of L2 — that adds a $\lambda I$ to the Hessian and pins the solution down.

**12. Logistic regression gives probability 0.99 — should I trust it?**
Depends on calibration. A well-fit logistic regression on representative data is usually well-calibrated. But check: bin the predictions and see what fraction of "0.99-confident" predictions are actually correct. If it's ~99%, trust it. If it's 80%, the model is overconfident — recalibrate (Platt or isotonic) or retrain.

> **Saying it out loud.** Only if you've checked calibration, and that's a five-minute check. Take all the predictions near 0.99, look at what fraction of them were actually positive, and if it's around 99% then the number means what it says. Logistic regression on representative data is usually pretty well calibrated, precisely because cross-entropy is a proper scoring rule and the model doesn't have the capacity to overfit its confidence the way a deep net does. But if your training data was resampled to fix class imbalance, the probabilities are systematically shifted and you have to correct the intercept.

**13. What does coefficient $w_j = 2.0$ mean?**
A one-unit increase in $x_j$, holding all other features constant, increases the log-odds of the positive class by 2.0 — i.e. multiplies the odds by $e^2 \approx 7.4$. The change in probability depends on the baseline.

> **Saying it out loud.** A coefficient is a change in log-odds, which nobody can picture, so translate it to an odds ratio: $e^2$ is about 7.4, so a one-unit increase in that feature multiplies the odds of the positive class by roughly seven and a half, holding everything else fixed. The catch worth naming is that odds are not probability — going from odds of 0.01 to 0.074 barely moves the probability, while going from 1 to 7.4 moves it from 50% to 88%. So the same coefficient means very different things in probability terms depending on where you start.

**14. What does $w_j = -0.5$ mean for a binary feature?**
Switching $x_j$ from 0 to 1 decreases the log-odds by 0.5, multiplying the odds by $e^{-0.5} \approx 0.61$. The presence of that binary feature is associated with lower odds of $y=1$, all else equal.

> **Saying it out loud.** For a binary feature it's the cleanest possible reading: having the feature versus not having it multiplies the odds by $e^{-0.5}$, about 0.61 — so roughly a 40% reduction in odds, all else equal. Being able to say 'multiplies the odds by 0.6' rather than 'decreases log-odds by 0.5' is what makes logistic regression usable in a regulated setting. The two words doing all the work are 'all else equal' — if that feature is correlated with something else in the model, the coefficient is not the effect of flipping it in the real world.

**15. My coefficient is huge but the feature looks unimportant. What's wrong?**
Multicollinearity. When two features are nearly redundant, the Hessian becomes nearly singular and individual coefficients explode (one big positive offset by a big negative on the correlated partner). Predictions stay fine; coefficients aren't interpretable. Use L2 or drop one feature.

> **Saying it out loud.** That's almost always multicollinearity. When two features carry nearly the same information, the model can't tell which one deserves the credit, so it finds a solution with a huge positive weight on one and a huge negative weight on its partner that mostly cancel. The Hessian is nearly singular along that direction, so the loss is basically flat there and the coefficients are free to wander. Key point: the *predictions* are still fine — it's the interpretation that's destroyed. Check VIF, then either drop one feature or add L2, which picks a unique sensible solution.

**16. Multicollinearity diagnostic?**
Variance inflation factor (VIF). $\text{VIF} = 1/(1 - R^2)$ for that feature regressed on the others. VIF > 10 is a warning. Also: condition number of $X^\top X$.

> **Saying it out loud.** Variance inflation factor. You regress each feature on all the other features, and VIF is one over one minus that $R^2$ — so a feature that the others can predict at $R^2$ of 0.9 gets a VIF of 10. The rule of thumb is that above 5 you look and above 10 you act. The other quick check is the condition number of $X^\top X$; if it's huge, some direction in feature space has almost no variance and your coefficients along it are noise.

**17. What does L2 do to the optimization?**
Adds $(\lambda/2) \|w\|^2$ to the loss. Hessian becomes $X^\top D X + \lambda I$, which is strictly positive definite for any $\lambda > 0$, even if features are collinear or data are separable. Guarantees a unique finite solution.

> **Saying it out loud.** L2 makes the problem strictly convex, which is the real point. It adds $\lambda I$ to the Hessian, so even if your features are collinear or your data is perfectly separable — the two cases where the plain solution is undefined or infinite — you now have a unique finite minimum. That's why sklearn regularizes by default and why people are surprised when they can't reproduce the unpenalized textbook answer. The cost is bias: your coefficients are systematically shrunk toward zero, which you trade for a massive reduction in variance.

**18. Why does L1 give sparse weights?**
Geometric: L1's level sets are diamonds with corners at the axes. The penalized minimum often lies at a corner, which corresponds to a coordinate being exactly zero. L2's level sets are circles — no corners, no exact zeros.

> **Saying it out loud.** It's the corners. Think of it as minimizing the loss subject to a budget on the weights — L2's budget region is a sphere and L1's is a diamond with sharp points sitting exactly on the axes. The loss contours expanding outward are much more likely to first touch a pointy corner than a smooth surface, and a corner is a point where some coordinates are exactly zero. The calculus version is that L1's gradient has constant magnitude all the way down to zero, so it keeps pushing a small coefficient to exactly zero, while L2's gradient shrinks as the coefficient does and never quite gets there.

**19. When would you choose L1 over L2?**
When you suspect most features are irrelevant and want feature selection. When interpretability requires a sparse model. When $d \gg N$.

> **Saying it out loud.** L1 when you believe most of your features are noise and you want the model to tell you which ones aren't — it does feature selection as part of fitting. It's especially natural when you have more features than examples, since you need something to pick a subset anyway. L2 when you think most features carry a bit of signal and features are correlated, because L1 handles correlated groups badly — it arbitrarily picks one and zeroes the rest, and which one it picks is unstable across resamples. If you want both, that's elastic net.

**20. Logistic regression on an imbalanced dataset (99:1)?**
First, don't use accuracy as the metric — it'll show 99% by predicting all-majority. Use AUPRC, F1, or balanced accuracy. Adjust the threshold (away from 0.5). Optionally use class weights or resampling. The model itself isn't broken; default decisions are.

> **Saying it out loud.** The model isn't broken; your defaults are. Accuracy is useless at 99:1 because predicting the majority scores 99%, so switch to PR-AUC or recall at a fixed precision. The 0.5 threshold is also just a convention — it's the right cutoff only if false positives and false negatives cost the same, which they never do in an imbalanced problem — so tune it on validation. Class weights or resampling are optional, and the thing to remember if you use them is that they distort your predicted probabilities, so you have to recalibrate before anyone trusts the numbers.

**21. How would you choose a threshold for an imbalanced problem?**
Define the cost trade-off: false positives vs false negatives, business value. Plot precision/recall vs threshold. Pick the operating point. Commonly: maximize F1 (if you want balance) or fix recall and minimize FPR (if you want guaranteed sensitivity).

> **Saying it out loud.** By working backwards from what a mistake costs. If a missed fraud costs a hundred times what a false alarm costs, that ratio directly implies where to put the cutoff — you're just doing expected value. In practice I'd sweep the threshold, plot precision and recall against it, and pick the point that satisfies the actual constraint, which is usually something like 'we can review 500 alerts a day' or 'we must catch 95% of cases'. And I'd pick it on validation data, not test, and re-check it after deployment because the base rate drifts.

---

## C. Beyond binary

**22. Generalize to K classes — what's the model?**
Multinomial logistic regression / softmax:

$$
P(y = k \mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}
$$

$K$ weight vectors with one redundant degree of freedom (subtracting a constant from all $w_k$ doesn't change probabilities). Often parameterized with $w_K = 0$ reference class.

> **Saying it out loud.** You go from one weight vector to $K$ of them, one per class, and replace the sigmoid with a softmax — exponentiate each class's score and divide by the sum so they add to one. That's multinomial logistic regression, and it's exactly the last layer of every neural classifier you've ever trained. One subtlety worth mentioning: the parameterization is redundant, because adding the same constant to every class's score changes nothing, so there's one degree of freedom too many. Statisticians fix a reference class at zero; ML just leaves it and lets regularization pick.

**23. Show that binary logistic regression is the K=2 case of softmax.**
With two classes:

$$
P(y=1 \mid x) = \frac{e^{w_1^\top x}}{e^{w_0^\top x} + e^{w_1^\top x}} = \frac{1}{1 + e^{-(w_1 - w_0)^\top x}} = \sigma\!\big((w_1 - w_0)^\top x\big)
$$

The single weight in binary LR is the difference $w_1 - w_0$.

> **Saying it out loud.** Write out the softmax for two classes and divide the top and bottom by the numerator of class one. Everything collapses to one over one plus $e$ to the minus the *difference* of the two weight vectors — which is exactly the sigmoid. So binary logistic regression's single weight vector isn't $w_1$, it's $w_1 - w_0$. That's the concrete version of the redundancy point: with $K$ classes you only ever need $K-1$ weight vectors, and the binary case is the special case where that's one.

**24. Loss function for multinomial?**
Categorical cross-entropy: $-\sum_i \sum_k \mathbf{1}[y_i = k] \log P(y = k \mid x_i) = -\sum_i \log P(y = y_i \mid x_i)$. MLE under categorical distribution.

> **Saying it out loud.** Categorical cross-entropy, which sounds fancy but is just the negative log of the probability you assigned to the correct class. The sum over classes with an indicator is only there for notational tidiness — every term but the true one is multiplied by zero. So you're penalized purely by how surprised you were by the right answer, and being 1% confident in the truth costs you a lot more than being 40% confident. And just like the binary case, it's not a choice: it's what maximum likelihood under a categorical distribution gives you.

**25. Are softmax outputs reliable probabilities?**
Mathematically yes (they're a valid distribution). Practically, in logistic regression on tabular data — usually yes if the model is well-specified. In **deep neural networks** — usually no, they're poorly calibrated despite high accuracy. Temperature scaling or post-hoc calibration is standard.

> **Saying it out loud.** They're valid probabilities in the sense that they're non-negative and sum to one, but that's not the same as being *trustworthy*. For a logistic regression on tabular data with a reasonably specified model, they're usually well calibrated. For a deep network they're usually badly overconfident — a model at 80% accuracy will happily report 99% — because high capacity lets it keep pushing logits apart long after accuracy plateaus. Fix is temperature scaling on a held-out set: one parameter, doesn't change accuracy at all, and typically cuts calibration error by most of the way.

---

## D. Deeper theory

**26. Why is the gradient $(\sigma - y)\,x$ and not something messier?**
**It's literally "predicted minus actual, times input"** — the cleanest possible form. Same shape as linear-regression gradient. This is the GLM canonical-link beauty: for *any* GLM with its canonical link (Gaussian+identity, Bernoulli+logit, Poisson+log), $\partial \text{NLL}/\partial w = X^\top (\hat\mu - y)$. The mess from sigmoid's derivative cancels with the mess from CE's $1/p$ exactly.

> **Saying it out loud.** Because two messes cancel exactly. Cross-entropy's derivative brings down a $1/p$, and the sigmoid's derivative brings up a $p(1-p)$, and those multiply out to leave you with just the error term. That isn't luck — it's what happens for any exponential-family distribution when you use its canonical link, which is why Gaussian with identity link, Bernoulli with logit, and Poisson with log all give you 'design matrix transpose times residual'. It's the same reason least squares and logistic regression feel like the same algorithm.

**27. What is Iteratively Reweighted Least Squares (IRLS)?**
Newton's method applied to logistic regression. Each iteration solves a weighted least squares problem with weights $\sigma(z)(1-\sigma(z))$. Converges quadratically; typically 5-10 iterations. Used by classical statistics packages (R's `glm`, `statsmodels`).

> **Saying it out loud.** IRLS is just Newton's method for logistic regression, dressed up. Each iteration, you build a local quadratic approximation of the loss, and because the Hessian is $X^\top D X$, solving for the Newton step turns out to be exactly a weighted least squares problem — where the weights are $p(1-p)$, the Bernoulli variance. So each step downweights the points the model is already confident about and focuses on the ones near the boundary, then you refit and reweight. Newton converges quadratically, so it's typically 5 to 10 iterations, which is why R's glm feels instant.

**28. Why does sklearn use L-BFGS by default for logistic regression?**
L-BFGS is a quasi-Newton method that approximates the Hessian without storing it explicitly. It's nearly as fast as IRLS for medium-sized problems, scales better to large $d$, and is more robust numerically. For $d$ in the millions, neither IRLS nor L-BFGS are great; SGD is used.

> **Saying it out loud.** Because L-BFGS scales and IRLS doesn't. IRLS needs to form and solve with the full $d \times d$ Hessian every iteration, which is fine at 50 features and impossible at 50,000. L-BFGS builds an implicit curvature approximation from the last dozen gradient differences, so it never stores a matrix, and in practice it converges in a comparable number of iterations for well-conditioned problems. Past a few million features you drop to SGD or SAGA, which is exactly why sklearn also ships those solvers.

**29. What's the relationship between logistic regression and Naive Bayes?**
Both can produce the same functional form ($\sigma$ of linear predictor) under specific conditional-feature assumptions (Gaussian features with shared covariance for NB). The key difference is fitting: NB is generative, fitting $P(x \mid y) P(y)$ separately; LR is discriminative, fitting $P(y \mid x)$ directly.

> **Saying it out loud.** They can end up with the same functional form but they get there completely differently. Naive Bayes is generative — it models how the features are distributed within each class and then applies Bayes' rule — and under Gaussian features with shared covariance, that produces a sigmoid of a linear function, exactly logistic regression's form. Logistic regression is discriminative: it skips modeling the features entirely and fits the boundary directly. The consequence is that logistic regression doesn't care whether the naive independence assumption is true, which is why it's usually more accurate given enough data.

**30. When does Naive Bayes beat logistic regression?**
Small data. Ng & Jordan (2001) showed: with infinite data, LR dominates; with finite data, NB can win because of lower variance. NB converges faster to its (biased) limit. Common in small-data text classification.

> **Saying it out loud.** When you're data-starved. Ng and Jordan's result is the cleanest statement of it: the generative model has higher asymptotic error because its independence assumption is wrong, but it converges to that error much faster — roughly in log-$d$ examples versus $d$ for the discriminative one. So there's a crossover, and below it Naive Bayes wins. That's the bias-variance tradeoff at the level of entire model families, and in practice it shows up in text classification with a few hundred labeled documents.

**31. What's the relationship to SVM?**
Both are linear classifiers. LR uses logistic loss $\log(1 + e^{-y \cdot z})$; SVM uses hinge loss $\max(0, 1 - y \cdot z)$. Hinge is exactly zero outside the margin (sparse contributions); logistic is smooth everywhere. LR gives calibrated probabilities; SVM gives only a score (Platt-scaled for probs).

> **Saying it out loud.** Same linear model, different loss, and the loss is the whole story. Hinge loss goes to exactly zero once a point is on the right side of the margin, so correctly-classified points stop contributing at all — that's what makes SVMs depend only on the support vectors. Logistic loss never quite reaches zero, so every point keeps exerting a little pressure forever, which is why it needs regularization on separable data and the SVM doesn't. Practical consequence: logistic regression gives you calibrated probabilities out of the box, while an SVM's decision value is just a score you have to Platt-scale before you can interpret it.

**32. Show that softmax + cross-entropy gradient is `softmax − one-hot`.**
Let $s_i = e^{w_i^\top x} / \sum_j e^{w_j^\top x}$. Loss $\mathcal{L} = -\log s_y$. After working through the softmax derivative, the result is $\partial \mathcal{L}/\partial z_k = s_k - \mathbf{1}[k = y]$, so $\partial \mathcal{L}/\partial w_k = (s_k - \mathbf{1}[k = y]) \cdot x$. This generalizes the binary $(\sigma - y)\,x$ gradient.

> **Saying it out loud.** The answer is beautifully simple: the gradient with respect to the logits is the softmax output minus the one-hot label. So if the true class is 3, you subtract one from the third entry and leave everything else — every wrong class gets pushed down in proportion to how much probability it grabbed, and the right one gets pushed up by how much it was missing. Then the gradient with respect to the weights is that vector times the input. It's the exact multi-class generalization of $(\sigma - y)x$, and it's why the softmax and cross-entropy are fused into one op in every framework.

**33. What's the maximum entropy interpretation of logistic regression?**
**It's the least-assuming distribution that fits what we've seen.** Formally: among all conditional distributions $P(y|x)$ that match the empirical feature-label moments, logistic regression is the one with maximum entropy — most uniform, fewest extra assumptions. This is dual to the GLM/canonical-link view.

> **Saying it out loud.** Suppose all you're willing to commit to is that the model should match certain averages you observed in the data. There are infinitely many distributions that do that, so which one do you pick? The maximum-entropy principle says pick the most spread-out, least-committed one — the one that assumes nothing beyond your constraints. Do that math and logistic regression falls out exactly. It's a satisfying answer because it means the sigmoid isn't a convenient squashing function someone picked; it's forced on you by refusing to assume anything extra.

**34. Explain calibration and how to test it.**
Calibration = predicted probabilities match observed frequencies. Test with reliability diagrams (bin predictions, plot mean predicted vs observed frequency), Brier score (MSE between probs and outcomes), and ECE (weighted bin error).

> **Saying it out loud.** Calibration means the number means what it says: among predictions of 0.7, about 70% should turn out positive. The visual check is a reliability diagram — bin the predictions, plot the average prediction against the actual rate, and a calibrated model sits on the diagonal. For a single number there's Brier score, which is MSE on probabilities, or ECE, which is the average bin-wise gap. Worth knowing that calibration is independent of discrimination: you can have a perfect AUROC and terrible calibration, because AUROC only sees the ordering.

**35. How do you fix miscalibration?**
Platt scaling: fit a 1D logistic regression $P_{\text{calibrated}} = \sigma(a \cdot \text{score} + b)$ on a held-out set. Isotonic regression: non-parametric monotonic mapping. Temperature scaling (NN-specific): divide logits by $T > 0$ before softmax, fit $T$ on validation.

> **Saying it out loud.** Fit a small correction on held-out data — never on the data you trained on. Platt scaling fits a one-dimensional logistic regression on top of your scores, so two parameters; temperature scaling is the neural-net version with just one parameter dividing the logits; isotonic regression is non-parametric and fits any monotone shape. Temperature and Platt can't change your ranking, so AUROC is untouched; isotonic can, and it needs a lot more validation data or it overfits the bins. Start with temperature — it's one parameter and it usually gets you most of the way.

---

## E. Connections to deep learning

**36. Logistic regression is a single-layer neural network — explain.**
It's $\sigma(w^\top x + b)$. That's exactly one fully-connected layer with one output neuron and sigmoid activation. The cross-entropy loss is exactly the BCE loss in PyTorch. Multinomial LR with softmax is exactly the standard final layer of a multi-class neural classifier.

> **Saying it out loud.** Take a neural network, delete all the hidden layers, and what's left is logistic regression. One fully-connected layer, one output, sigmoid on the end, binary cross-entropy loss — that's literally `nn.Linear(d, 1)` plus `BCEWithLogitsLoss`. Nothing about the math changes; the only thing deep learning adds is layers of learned features underneath. It's a useful framing because it means everything you know about logistic regression applies unchanged to the last layer of any classifier.

**37. What does this imply for the final layer of any classifier NN?**
It's logistic / multinomial logistic regression on top of learned features. Everything that affects logistic regression — calibration issues, separability, regularization — applies to that final layer. The "deep" part is feature learning; the "classification" part is unchanged.

> **Saying it out loud.** It means the classification part of a deep classifier is logistic regression, and only the feature extraction is new. So every property carries over: the loss is the same, the gradient is still predicted-minus-actual, separability in the learned feature space causes the same weight blow-up, and the same calibration questions apply. That's actually a useful way to reason about deep models — the network learns a representation in which the classes are nearly linearly separable, and then does the simplest possible thing on top. It also explains why fine-tuning just the last layer works so often.

**38. Why do deep networks miscalibrate but logistic regression doesn't?**
NNs overfit confidence. With high capacity and many parameters, NNs can drive training cross-entropy near zero by pushing logits to extreme values, even when validation accuracy plateaus. Result: predicted probabilities concentrate at 0 and 1 even when the model is uncertain. LR's lower capacity makes it harder to overfit confidence the same way.

> **Saying it out loud.** Capacity. Cross-entropy always rewards moving a correct prediction from 0.9 to 0.99, and a big network has enough parameters to keep doing that long after validation accuracy has flattened, so it learns confidence rather than correctness. Logistic regression simply can't — with $d$ parameters and a linear boundary, it hits the limit of how well it can fit and stops, so its probabilities stay honest. Guo et al. showed the miscalibration gets worse as networks get wider and deeper, and that a single temperature parameter fixes most of it.

**39. Is dropout useful in logistic regression?**
Generally not. Dropout's value is in deep networks where it prevents co-adaptation of hidden units. LR has no hidden units; there's nothing to drop out. Use L2 or L1 for regularization.

> **Saying it out loud.** No — there's nothing to drop out. Dropout works by preventing hidden units from co-adapting, and logistic regression has no hidden units; dropping input features instead just injects noise into a convex problem, which is roughly an inefficient version of L2. So use L2 for correlated features, L1 if you want sparsity. It's a good question to be asked because the temptation is to answer 'sure, regularization is regularization', and the point is that dropout's mechanism is specific to depth.

---

## F. Practical engineering

**40. How do you decide which features matter?**
For a fitted L2 model: standardize features first, then compare absolute coefficient magnitudes (only meaningful if scales are comparable). For L1 model: zero coefficients are explicitly excluded. Better: permutation importance — shuffle a feature, measure performance drop.

> **Saying it out loud.** Standardize first, otherwise coefficient magnitudes are just telling you about units — a feature measured in millimeters gets a coefficient a thousand times smaller than the same feature in meters. After standardizing, absolute coefficient size is a reasonable first pass, and with L1 the zeros tell you directly what got dropped. But I'd trust permutation importance more: shuffle one column, see how much the metric degrades, because that measures contribution to actual predictions rather than to a possibly-collinear parameterization. And with correlated features, no coefficient-based method is reliable — the credit gets split arbitrarily.

**41. Should you standardize features for logistic regression?**
**Yes, if you're using regularization**, because L1/L2 penalties depend on coefficient magnitude, which depends on feature scale. Without standardization, regularization unfairly penalizes high-magnitude features. Without regularization, the math doesn't care about scale (only convergence speed for SGD).

> **Saying it out loud.** Yes, if you're regularizing, and that's the whole condition. L1 and L2 penalize coefficient magnitude, and a coefficient's magnitude depends on the units of its feature — so if income is in dollars and age is in years, the penalty falls almost entirely on age. Standardizing puts every feature on the same footing so the penalty is fair. Without regularization the fit is invariant to scaling and it only matters for how fast the optimizer converges, but since sklearn regularizes by default, in practice the answer is just 'yes, always'.

**42. How do you handle missing features?**
Imputation (mean, median, model-based) is standard. Or treat missingness as an indicator: add a binary "is_missing" feature. Don't drop rows unless missingness is rare and missing-completely-at-random.

> **Saying it out loud.** Impute, and add an indicator for whether it was missing. The imputation keeps the row usable; the indicator lets the model learn that missingness itself is predictive, which it very often is — a blank income field in a loan application is not random. Dropping rows is only safe if missingness is rare and truly unrelated to the outcome, and that's rarer than people assume. The critical engineering detail is to fit the imputation on training data only and apply the same values at inference, otherwise you've leaked test statistics into the model.

**43. Categorical features — how?**
One-hot encode (drop one level as reference, or include all if you have a regularization penalty to prevent the redundancy from causing infinite weights). For high-cardinality (e.g. zip codes), use target encoding, hash trick, or embeddings. Beware of leak in target encoding.

> **Saying it out loud.** One-hot for low cardinality, and drop one level as the reference — otherwise the dummy columns sum to the intercept and you've built collinearity into the design on purpose. If you're regularizing you can get away with keeping them all, since the penalty resolves the ambiguity. For something like zip codes with thousands of levels, one-hot is hopeless and you go to target encoding, hashing, or a learned embedding. And target encoding is a leakage trap — you have to compute it out-of-fold, or the model memorizes the answer through the encoding.

**44. Online updates — how does logistic regression handle streaming data?**
SGD update per new sample is $w \leftarrow w + \eta \cdot (y - \sigma(w^\top x)) \cdot x$. Single-pass online learning is principled (it's MLE in the streaming regime). Strong choice for streaming applications. Common in ad ranking, online recommendation.

> **Saying it out loud.** It's about as online-friendly as a model gets. The SGD update is just the learning rate times the error times the feature vector — one sample, one dot product, one weight update, no state to keep. For sparse features you only touch the coordinates that are non-zero, so an update is microseconds even with millions of features. That's why it was the workhorse for ad click prediction for years, usually with a per-coordinate adaptive learning rate like FTRL, which also gets you sparsity for free.

**45. Latency requirements — when is logistic regression preferred?**
Sub-millisecond inference budgets (real-time bidding, online recommendation). Inference is $O(d)$ — a single dot product. No matrix multiplications, no GPU needed. Often the only feasible model for tight latency budgets.

> **Saying it out loud.** When your latency budget is sub-millisecond and you're serving on CPU. Inference is a single dot product — order $d$ operations, no matrix multiply, no GPU, no batching required — so you can do it inside a real-time bidding auction that has to complete in ten milliseconds end to end. There's also no cold start, no memory footprint to speak of, and it's trivially parallel. The tradeoff is that you're paying for that speed with a linear boundary, which you compensate for by putting the cleverness into feature engineering.

**46. Interpretability — when does LR win over a tree?**
Regulated industries (credit, insurance, healthcare): you must explain individual predictions. LR's coefficients give clean, additive explanations. Trees are *locally* interpretable but *globally* messy. LR's calibration also matters here — credit scoring needs reliable probabilities.

> **Saying it out loud.** When someone has a legal right to an explanation. In credit and insurance you have to be able to tell an applicant which factors drove the decision, and a coefficient times a feature is an explanation that survives a regulator — it's additive, it's stable, and it's the same explanation for everyone. Trees give you a local decision path but no coherent global story, and ensembles give you nothing without a post-hoc tool like SHAP, which is an approximation you now also have to defend. The other half is calibration: credit scoring needs an actual probability of default, not a ranking, and logistic regression gives you that natively.

---

## G. Probit and other GLMs

**47. What's probit regression?**
GLM with the probit link $\Phi^{-1}$ (inverse standard normal CDF) instead of logit. Functional form: $P(y=1 \mid x) = \Phi(w^\top x + b)$. Used in econometrics; rarely in ML practice.

> **Saying it out loud.** Probit is the same idea with a different squashing function — instead of the logistic curve you use the normal CDF. The motivating story is nicer in economics: imagine there's a latent continuous utility with Gaussian noise, and you observe the outcome only when it crosses zero. That gives you probit exactly. It's essentially interchangeable with logit in practice, and it's common in econometrics for that latent-variable interpretation.

**48. Logit vs probit — does it matter?**
Empirically, almost never. Both are S-shaped, both are between 0 and 1, predictions agree on most data within rescaling. Logit dominates in ML because the gradient is cleaner and the canonical-link beauty applies.

> **Saying it out loud.** Almost never. The two curves are nearly identical after rescaling — logistic coefficients are roughly 1.6 times the probit ones — and predictions differ meaningfully only far out in the tails where you have almost no data anyway. So the choice is about convention and convenience, not fit. ML uses logit because the gradient is clean, the coefficients read as log-odds ratios, and it's the canonical link; econometrics keeps probit where the latent Gaussian story matters.

**49. Given a Poisson outcome, what's the GLM?**
Poisson regression with log link: $\log(\lambda) = w^\top x + b$. The canonical link for the Poisson is the log. Same $X^\top (\hat\mu - y)$ gradient form. Used for count data (clicks, events, etc.).

> **Saying it out loud.** Poisson regression with a log link — you model the log of the rate as linear in the features. The log link is doing real work: it keeps the predicted rate positive no matter what the linear part does, and it makes the coefficients multiplicative, so $e^\beta$ is a rate ratio. It's the right model for counts — clicks, arrivals, defects. And since log is the canonical link for the Poisson, you get the same $X^\top(\hat\mu - y)$ gradient. The failure mode to name is overdispersion: real count data usually has variance bigger than its mean, which Poisson forbids, so you check and switch to negative binomial.

**50. Walk me through the canonical-link beauty.**
For exponential-family distributions, the negative log-likelihood with canonical link gives $\partial \text{NLL}/\partial w = X^\top (\hat\mu - y)$. The Hessian is $X^\top \mathrm{diag}(V(\hat\mu)) X$ where $V$ is the variance function. This is why linear, logistic, and Poisson regressions all share the form "$X^\top \cdot \text{residual}$" — they're all GLMs with their canonical link. This is one of the deepest unifying results in classical statistics.

> **Saying it out loud.** Here's the unifying idea. Pick any exponential-family distribution for your outcome — Gaussian, Bernoulli, Poisson — use its canonical link, and the gradient of the negative log-likelihood is always the design matrix transposed times the residual. Same formula, three completely different-looking models. The Hessian is always $X^\top$ times a diagonal of the distribution's variance function times $X$, which is why they're all convex and why IRLS works for all of them. It's worth knowing because it turns a pile of separately-memorized models into one theorem plus a table of links.

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
