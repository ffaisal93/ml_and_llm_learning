# Classical ML

Classical ML is the warm-up round, and it is where most candidates lose points quietly. The interviewer is not checking that you know what a random forest is. He is checking that you can derive a gradient on a whiteboard, say why a loss is convex, and name the exact term that a change moves. The common failure is answering with names instead of mechanism: saying "L1 gives sparsity" without the geometry, or "boosting reduces bias" without saying what boosting fits at each step.

## The equations

**Mean squared error.**

$$\mathcal{L}_{\text{MSE}}(w) = \frac{1}{n}\sum_{i=1}^{n}\left(w^{\top}x_i - y_i\right)^2$$

Here $x_i \in \mathbb{R}^d$ is one feature vector, $y_i$ its real-valued target, $w$ the weights, and $n$ the number of examples; MSE gives you a smooth convex objective whose minimiser is the conditional mean $\mathbb{E}[y \mid x]$.

**Normal equation.**

$$w^{\star} = (X^{\top}X)^{-1}X^{\top}y$$

$X$ is the $n \times d$ design matrix and $y$ the $n$-vector of targets; this solves linear regression in one step, and it exists only when $X^{\top}X$ is invertible, so it fails on collinear or wide data.

**Gradient of MSE.**

$$\nabla_w \mathcal{L}_{\text{MSE}} = \frac{2}{n}X^{\top}(Xw - y)$$

The term $Xw - y$ is the residual vector, so the gradient is the feature matrix transposed against the residuals; setting it to zero recovers the normal equation.

**Sigmoid and the logistic model.**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad p(y=1 \mid x) = \sigma(w^{\top}x + b)$$

$z = w^{\top}x + b$ is the logit, and $\sigma$ squashes it into $(0,1)$; the inverse is the log-odds, $z = \log\frac{p}{1-p}$, which is why logistic regression is a linear model in log-odds space.

**Log-loss and its gradient.**

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big], \qquad \nabla_w \mathcal{L} = \frac{1}{n}X^{\top}(p - y)$$

$p_i = \sigma(w^{\top}x_i + b)$ is the predicted probability; the gradient is again features times residual, because the sigmoid derivative cancels exactly against the log-loss denominator.

**Bias-variance decomposition.**

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{variance}} + \sigma^2$$

The expectation is over training sets; bias is the error of the average model, variance is how much the model moves when the training set changes, and $\sigma^2$ is irreducible label noise you can never remove.

**L2 and L1 penalties.**

$$\mathcal{L}_{\text{ridge}} = \mathcal{L} + \lambda\|w\|_2^2 = \mathcal{L} + \lambda\sum_j w_j^2, \qquad \mathcal{L}_{\text{lasso}} = \mathcal{L} + \lambda\sum_j |w_j|$$

$\lambda$ sets the strength; L2 has gradient $2\lambda w$ which shrinks proportionally and never reaches zero, while L1 has constant subgradient $\lambda\,\text{sign}(w_j)$ which drives small weights exactly to zero.

**Softmax and cross-entropy.**

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \mathcal{L} = -\sum_{k=1}^{K} y_k \log p_k, \qquad \frac{\partial \mathcal{L}}{\partial z_k} = p_k - y_k$$

$z$ is the vector of $K$ logits and $y$ the one-hot label; the gradient with respect to the logits is again predicted minus true, which is why softmax plus cross-entropy is the standard classification head.

**Entropy and Gini for splits.**

$$H(S) = -\sum_{c} p_c \log_2 p_c, \qquad G(S) = 1 - \sum_{c} p_c^2, \qquad \text{IG} = H(S) - \sum_{v} \frac{|S_v|}{|S|}H(S_v)$$

$p_c$ is the fraction of class $c$ in node $S$ and $S_v$ a child produced by a candidate split; information gain is the impurity you remove, and Gini is a cheaper quadratic proxy for entropy that ranks splits almost identically.

**k-means objective.**

$$J = \sum_{i=1}^{n}\sum_{j=1}^{k} r_{ij}\,\|x_i - \mu_j\|_2^2, \qquad \mu_j = \frac{\sum_i r_{ij} x_i}{\sum_i r_{ij}}$$

$r_{ij}$ is 1 when point $i$ belongs to cluster $j$ and 0 otherwise; the assign step minimises $J$ over $r$ with $\mu$ fixed and the update step minimises over $\mu$ with $r$ fixed, so $J$ decreases monotonically to a local minimum.

## Code from memory

Linear regression by gradient descent, checked against the closed-form normal equation.

```python
import numpy as np

def linreg_gd(X, y, lr=0.1, steps=500):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        pred = X @ w + b
        err = pred - y                      # residual
        gw = (2.0 / n) * (X.T @ err)        # dMSE/dw
        gb = (2.0 / n) * err.sum()          # dMSE/db
        w -= lr * gw; b -= lr * gb
    return w, b

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([1.5, -2.0, 0.5]) + 0.3 + 0.05 * rng.normal(size=200)

w, b = linreg_gd(X, y)
Xa = np.hstack([X, np.ones((200, 1))])                 # augmented design
w_ne = np.linalg.solve(Xa.T @ Xa, Xa.T @ y)            # normal equation
print("gd    ", np.round(np.append(w, b), 4))
print("normal", np.round(w_ne, 4))
```

Output: both lines print `[ 1.4955 -2.0048  0.5027  0.2982]`, so gradient descent matches the normal equation to four decimals.

Logistic regression with the log-loss gradient, checked against `sklearn.linear_model.LogisticRegression` with regularisation effectively off.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logreg_gd(X, y, lr=0.5, steps=20000):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        err = p - y                     # same residual form as linear regression
        w -= lr * (X.T @ err) / n
        b -= lr * err.sum() / n
    return w, b

def log_loss(X, y, w, b):
    p = np.clip(sigmoid(X @ w + b), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

rng = np.random.default_rng(1)
X = rng.normal(size=(300, 2))
z = X @ np.array([2.0, -1.0]) + 0.5
y = (rng.random(300) < sigmoid(z)).astype(float)       # noisy labels, not separable

w, b = logreg_gd(X, y)
sk = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
print("mine   ", np.round(w, 3), round(b, 3), round(log_loss(X, y, w, b), 5))
print("sklearn", np.round(sk.coef_[0], 3), round(sk.intercept_[0], 3))
```

Output: mine gives `[1.829 -0.781] 0.401` with log-loss `0.47182`, sklearn gives `[1.829 -0.78] 0.401`, so they agree to three decimals. Note the labels must be non-separable, because on separable data the unregularised weights diverge and no finite optimum exists.

k-means with explicit assign and update loops, checked against `sklearn.cluster.KMeans` inertia.

```python
import numpy as np

def kmeans(X, k, steps=25, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(steps):
        # assign: each point to its nearest centroid
        for i in range(n):
            best, bd = 0, np.inf
            for j in range(k):
                dist = np.sum((X[i] - C[j]) ** 2)
                if dist < bd:
                    best, bd = j, dist
            labels[i] = best
        # update: each centroid to the mean of its members
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                C[j] = members.mean(axis=0)
    inertia = sum(np.sum((X[i] - C[labels[i]]) ** 2) for i in range(n))
    return C, labels, inertia

rng = np.random.default_rng(3)
X = np.vstack([rng.normal(m, 0.5, size=(60, 2)) for m in ([0, 0], [4, 0], [2, 4])])
C, lab, J = kmeans(X, 3)
print("objective J =", round(J, 3))
```

Output: `objective J = 92.783`, and `sklearn.cluster.KMeans(3, n_init=10, random_state=0).inertia_` is also `92.783`, so the two reach the same local optimum on this data.

## Questions

### Q1. Why does the logistic-regression gradient have the same form as linear regression's?

Because the sigmoid derivative cancels against the log-loss denominator. For one example, log-loss is $-y\log p - (1-y)\log(1-p)$ with $p = \sigma(z)$ and $z = w^{\top}x$. The derivative of the loss with respect to $p$ is $\frac{p-y}{p(1-p)}$. The derivative of the sigmoid is $\sigma'(z) = p(1-p)$. Multiply them by the chain rule and the $p(1-p)$ factors cancel exactly, leaving $\frac{\partial \mathcal{L}}{\partial z} = p - y$. Then $\frac{\partial \mathcal{L}}{\partial w} = (p-y)x$. Linear regression with MSE gives $\frac{\partial \mathcal{L}}{\partial w} = 2(\hat{y}-y)x$, the same shape. This is not a coincidence: both are generalised linear models with a canonical link, and for any GLM with its canonical link the gradient is always features times residual. Softmax with cross-entropy gives $p_k - y_k$ for the same reason.

> **Say it.** Both are generalised linear models with the canonical link, so both gradients are features times residual. Concretely, log-loss differentiated with respect to the probability gives $p-y$ over $p(1-p)$, and the sigmoid derivative is exactly $p(1-p)$, so the two cancel and you are left with $p - y$ at the logit. Multiply by $x$ and you get the same update shape as least squares. Softmax and cross-entropy do the same thing in the multiclass case, giving predicted minus one-hot.

### Q2. Why do we not use MSE for classification?

Two separate reasons. First, convexity. MSE composed with the sigmoid, $(\sigma(w^{\top}x) - y)^2$, is not convex in $w$, so gradient descent can stop at a local minimum. Log-loss composed with the sigmoid is convex in $w$, so any stationary point is global. Second, the gradient. With MSE the chain rule gives $2(p-y)\,\sigma'(z)\,x = 2(p-y)p(1-p)x$. When the model is confidently wrong, say $p \approx 0.999$ and $y = 0$, the factor $p(1-p) \approx 0.001$ crushes the gradient to almost nothing, so the worst-predicted examples produce the smallest updates. Log-loss removes that factor entirely and gives $(p-y)x$, so a confidently wrong example produces a gradient of magnitude near one. Log-loss is also the correct likelihood for Bernoulli labels, so its minimiser is the true conditional probability.

> **Say it.** Two reasons. MSE through a sigmoid is non-convex in the weights, so you can get stuck; log-loss is convex. More practically, the MSE gradient carries an extra sigmoid-derivative factor $p(1-p)$. If the model says 0.999 and the truth is zero, that factor is about 0.001, so the most wrong example gives you almost no gradient and learning stalls. Log-loss cancels that factor and the gradient is just $p - y$, so confident errors push hardest. Log-loss is also the Bernoulli likelihood, so it estimates calibrated probabilities.

### Q3. L1 versus L2: what is the difference, and why does L1 give exact zeros?

L2 adds $\lambda\|w\|_2^2$ with gradient $2\lambda w$. That force is proportional to the weight, so it shrinks large weights hard and small weights barely at all, and it never reaches zero in finite time. L1 adds $\lambda\sum_j|w_j|$ with subgradient $\lambda\,\text{sign}(w_j)$. That force is constant in magnitude regardless of how small the weight is, so once the data gradient on a coordinate falls below $\lambda$, the penalty pins that coordinate at exactly zero. Geometrically: constrained optimisation puts the solution where a loss contour first touches the constraint set. The L2 ball is round, so the contact point is almost never on an axis. The L1 ball is a cross-polytope with corners on the axes, and corners are where the contours touch with high probability, and a corner means most coordinates equal zero. So L2 does dense shrinkage, L1 does feature selection. Elastic net combines both.

> **Say it.** L2's gradient is $2\lambda w$, proportional to the weight, so it shrinks everything smoothly but never to exactly zero. L1's subgradient is $\lambda$ times the sign, constant in size, so once the data gradient on a coordinate drops below $\lambda$ that coordinate is pinned at exactly zero. Geometrically, the L2 constraint region is a ball and the L1 region is a diamond with corners on the axes. Elliptical loss contours almost never touch a sphere on an axis, but they very often touch a diamond at a corner, and a corner has zeros in it. So L1 selects features.

### Q4. Explain bias-variance in words. What actually moves each term?

Bias is the error of the average model over training sets: the part your hypothesis class cannot express no matter how much data you get. Variance is how much the fitted model moves when you resample the training set. The third term, $\sigma^2$, is label noise and nothing you do touches it. What lowers bias: a richer model class, more features, deeper trees, more parameters, longer training, weaker regularisation. What lowers variance: more training data, stronger regularisation, fewer features, shallower trees, averaging over decorrelated models such as bagging, early stopping. The two conflict because most knobs move them in opposite directions, which is why the test error curve is U-shaped in model complexity. Two practical notes. More data lowers variance but does not lower bias. And in the overparameterised deep-learning regime the classical U-curve does not hold, because very large models can have both low bias and low variance.

> **Say it.** Bias is the error of the average model over training sets, so it is what your model class cannot express. Variance is how much the fitted model moves when the training set changes. Noise is irreducible. More capacity, more features and less regularisation lower bias. More data, stronger regularisation, fewer features and averaging decorrelated models lower variance. More data lowers variance but never bias, which is the part people forget. And the classical U-shaped curve is a small-model story; very large modern models can be low in both.

### Q5. Generative versus discriminative: what is the difference and when do you pick each?

A discriminative model learns $p(y \mid x)$ directly, or just the decision boundary. Logistic regression, SVMs, and neural network classifiers are discriminative. A generative model learns the joint $p(x, y)$, usually as $p(x \mid y)p(y)$, then applies Bayes' rule to get $p(y \mid x)$. Naive Bayes, Gaussian discriminant analysis, and HMMs are generative. Discriminative models generally reach lower asymptotic error, because they spend all capacity on the boundary and make no assumption about how $x$ is distributed. Generative models converge faster in the low-data regime, because the modelling assumption acts as a strong prior, and they give you extra abilities: sampling new $x$, handling missing features by marginalising, and scoring how unusual an input is for outlier detection. The classic result is that naive Bayes beats logistic regression on small $n$ and loses as $n$ grows.

> **Say it.** Discriminative models learn $p(y \mid x)$ or just the boundary; generative models learn $p(x \mid y)$ and the prior, then invert with Bayes. Discriminative usually wins asymptotically, because it spends all its capacity on the boundary and assumes nothing about how the inputs are distributed. Generative wins with little data, because the modelling assumption acts as a prior, and it gives you sampling, marginalising over missing features, and outlier scores for free. Naive Bayes against logistic regression is the textbook pair: naive Bayes leads on small data and loses as data grows.

### Q6. Naive Bayes assumes conditional independence, which is false. Why does it still work?

Because classification only needs the argmax to be right, not the probabilities. Naive Bayes scores $\log p(y) + \sum_j \log p(x_j \mid y)$. When features are correlated, it counts the same evidence several times, so the posterior is pushed towards zero or one and becomes badly calibrated. However, over-counting usually scales the scores of all classes in a correlated way, so the ranking of classes often survives even though the magnitudes do not. Bag-of-words text is the standard example: words are clearly dependent, but the class with the most supporting words is still usually the correct one. Two consequences you should state. First, never trust naive Bayes probabilities; use them only for ranking or recalibrate them. Second, it breaks when the dependence is asymmetric across classes, because then the over-counting favours one class and flips the argmax. It also has very low variance, which is why it does well on tiny datasets.

> **Say it.** Because classification needs the argmax, not the probability. The independence assumption makes it double-count correlated evidence, so the posteriors get pushed to zero or one and calibration is terrible. But the over-counting tends to affect classes in a correlated way, so the ordering of classes usually survives. Text with bag-of-words is the standard case. So use it for ranking or the label, never for a probability you will act on, and recalibrate if you need one. It also has very low variance, which is why it is strong on small datasets.

### Q7. Bagging versus boosting: what does each reduce, and how?

Bagging reduces variance. You train $M$ models on bootstrap resamples and average them. If each model has variance $\sigma^2$ and average pairwise correlation $\rho$, the average has variance $\rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$. So averaging only helps to the extent the models are decorrelated, and the $\rho\sigma^2$ floor is why random forests also subsample features at each split: that lowers $\rho$. The base learners are deliberately low-bias and high-variance, which means deep unpruned trees. Bagging is embarrassingly parallel. Boosting reduces bias. You fit models sequentially, and each new model fits the residual, or gradient of the loss, left by the current ensemble: $F_m(x) = F_{m-1}(x) + \eta h_m(x)$. Base learners are deliberately high-bias, which means shallow trees of depth 3 to 6. Boosting is sequential and can overfit if you use too many rounds, so you control it with the learning rate $\eta$, tree depth, and early stopping.

> **Say it.** Bagging cuts variance by averaging models trained on bootstrap samples. The variance of the average is $\rho\sigma^2$ plus $(1-\rho)\sigma^2$ over $M$, so it only helps if the models are decorrelated, which is exactly why random forests also subsample features per split. Base learners are deep, low-bias, high-variance trees, and it parallelises. Boosting cuts bias by fitting models sequentially to the residual or gradient of the current ensemble, with shallow high-bias trees and a small learning rate. Boosting can overfit with too many rounds, so you early-stop it.

### Q8. How does a decision tree pick a split?

Greedily and locally. At each node it enumerates candidate splits: for each feature, each threshold between sorted adjacent values. For each candidate it computes the weighted impurity of the two children and picks the split with the largest impurity reduction, $\text{IG} = H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v)$, where $H$ is entropy $-\sum_c p_c \log_2 p_c$ or Gini $1 - \sum_c p_c^2$. For regression it uses variance reduction, which is the same thing with squared error as the impurity. It never revisits an earlier split, so the tree is greedy and not globally optimal; finding the optimal tree is NP-hard. Growth stops on max depth, minimum samples per leaf, or zero gain, and you then prune with cost-complexity pruning that penalises leaf count by $\alpha|T|$. Two known biases: impurity gain favours high-cardinality features, because many candidate thresholds give more chances to fit noise, and axis-aligned splits handle diagonal boundaries badly.

> **Say it.** Greedily. At each node it tries every feature and every threshold between sorted values, computes the weighted impurity of the children, and takes the split with the largest reduction. Impurity is entropy or Gini for classification, variance for regression. It never goes back and revises an earlier split, because the optimal tree is NP-hard, so trees are greedy. It stops on depth, minimum leaf size or zero gain, then you cost-complexity prune. Watch two biases: gain favours high-cardinality features, and axis-aligned splits are bad at diagonal boundaries.

### Q9. Explain the SVM margin and the kernel trick.

The SVM finds the hyperplane $w^{\top}x + b = 0$ that maximises the distance to the closest points. With labels in $\{-1, +1\}$ the functional constraint is $y_i(w^{\top}x_i + b) \ge 1$, the geometric margin is $2/\|w\|$, so maximising the margin means minimising $\frac{1}{2}\|w\|^2$ subject to those constraints. Soft margin adds slack: minimise $\frac{1}{2}\|w\|^2 + C\sum_i \xi_i$, where $C$ trades margin width against violations. The dual depends on the data only through inner products $x_i^{\top}x_j$, and the solution is $w = \sum_i \alpha_i y_i x_i$ with $\alpha_i$ nonzero only for support vectors. The kernel trick replaces every inner product with $K(x_i, x_j) = \phi(x_i)^{\top}\phi(x_j)$. You get the geometry of a high or infinite dimensional feature space without ever computing $\phi$. The RBF kernel $K = \exp(-\gamma\|x_i - x_j\|^2)$ corresponds to an infinite-dimensional space. Cost: training is roughly quadratic to cubic in $n$, so kernels do not scale to millions of rows.

> **Say it.** The SVM maximises the geometric margin $2$ over the norm of $w$, which means minimising half the squared norm subject to every point having functional margin at least one. Soft margin adds slack variables with penalty $C$. The dual touches the data only through inner products, so you swap each inner product for a kernel and get the geometry of a huge feature space without ever forming the features. RBF corresponds to an infinite-dimensional space. The catch is that training scales roughly quadratically to cubically in the number of points.

### Q10. What is the curse of dimensionality and what does it do to distance-based methods?

As dimension $d$ grows, the volume of the space grows exponentially, so any fixed sample becomes sparse. Two concrete effects. First, to keep a fixed fraction $r$ of the data inside a neighbourhood you need side length $r^{1/d}$, so at $d = 100$ capturing one percent of the data needs side length $0.01^{0.01} \approx 0.955$, which is almost the whole range: the "local" neighbourhood is not local. Second, distance concentration. For many distributions the ratio $\frac{\max_j\|x - x_j\| - \min_j\|x - x_j\|}{\min_j\|x - x_j\|}$ goes to zero as $d$ grows, so the nearest and farthest neighbours become nearly equidistant and "nearest neighbour" stops carrying information. This breaks k-NN, k-means, and RBF kernels, all of which rely on distance ranking. Fixes: reduce dimension with PCA or a learned embedding, use cosine similarity on normalised vectors which is better behaved for sparse text, or use models that select features rather than weighting all of them.

> **Say it.** Volume grows exponentially with dimension, so data becomes sparse and two things break. A neighbourhood holding one percent of the data in a hundred dimensions needs about 95 percent of the range per side, so it is not local at all. And distances concentrate: nearest and farthest points become nearly the same distance away, so the ranking that k-NN, k-means and RBF kernels depend on carries almost no signal. The fixes are dimensionality reduction, learned embeddings, cosine similarity for sparse high-dimensional text, and models that select features instead of using all of them.

### Q11. How do you handle class imbalance, and why is accuracy the wrong metric?

Accuracy is wrong because the majority-class baseline already scores well. At one percent positives, predicting "negative" always gives 99 percent accuracy and zero recall, so accuracy cannot distinguish a useless model from a good one. Use precision, recall, F1, and PR-AUC instead, and report the positive-class numbers. On the modelling side, first ask whether the imbalance is real or a sampling artefact. Then, in rough order of what I try: set class weights in the loss, which is equivalent to reweighting the gradient and needs no data change; move the decision threshold using a precision-recall curve rather than accepting 0.5; resample, with undersampling the majority when you have plenty of data and oversampling or SMOTE when positives are very few. Note that resampling shifts the predicted probabilities away from the true base rate, so recalibrate afterwards if you need probabilities. If positives are extremely rare, treat it as anomaly detection instead.

> **Say it.** Accuracy is wrong because the trivial majority classifier already wins it: at one percent positives, always saying negative scores 99 percent with zero recall. So report precision, recall, F1 and PR-AUC on the positive class. To fix the model I start with class weights in the loss, then tune the threshold off the precision-recall curve rather than using 0.5, then resample if I still need to. Remember resampling distorts predicted probabilities away from the true base rate, so recalibrate afterwards. Below roughly a thousandth positive rate, treat it as anomaly detection.

### Q12. How would you explain regularisation as a prior?

Maximum a posteriori estimation maximises $\log p(w \mid D) = \log p(D \mid w) + \log p(w) + \text{const}$. The first term is the negative training loss. The second term is the prior, and it is exactly the penalty. Put a zero-mean Gaussian prior $w_j \sim \mathcal{N}(0, \tau^2)$ on each weight and its log density is $-\frac{w_j^2}{2\tau^2}$ plus a constant, so the penalty is $\frac{1}{2\tau^2}\|w\|_2^2$: that is ridge, with $\lambda = \frac{1}{2\tau^2}$. Put a zero-mean Laplace prior $p(w_j) \propto e^{-|w_j|/b}$ and the log density is $-|w_j|/b$, so the penalty is $\frac{1}{b}\|w\|_1$: that is lasso. The correspondence tells you two useful things. A wide prior, large $\tau$, means small $\lambda$ and weak regularisation. And the Laplace prior has a sharp peak at zero and heavy tails, which is exactly why lasso produces exact zeros while still allowing a few large weights.

> **Say it.** MAP estimation maximises log-likelihood plus log-prior, and the log-prior is the penalty term. A zero-mean Gaussian prior on the weights has log density proportional to minus $w$ squared, so it gives you ridge with $\lambda$ equal to one over twice the prior variance. A Laplace prior gives minus the absolute value, so it gives you lasso. That explains the behaviour: a wide prior means weak regularisation, and the Laplace prior is spiked at zero with heavy tails, which is exactly why lasso zeros most weights but still permits a few big ones.

## Done when

- You can write the log-loss gradient derivation on a whiteboard in under three minutes, showing the $p(1-p)$ cancellation explicitly.
- You can code linear regression, logistic regression and k-means from memory in NumPy, with no reference, and each runs first try.
- You can draw the L1 diamond and L2 ball and say in one sentence why the diamond's corner produces exact zeros.
- You can state, for any of bagging, boosting, more data, and stronger regularisation, which of bias or variance it moves and in which direction.
