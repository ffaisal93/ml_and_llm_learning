# The breadth round: machine learning

A breadth round is twenty short questions in forty minutes. The interviewer wants the equation, the one distinction that matters, and the reason. He does not want a lecture. Every answer below is written for about a minute of speech — roughly 120 to 140 spoken words. The shape is the same every time: the direct answer, then the equation or the mechanism, then one sentence of consequence, then stop. Silence after a correct answer is not a failure. It is an invitation for the interviewer to pick the follow-up he cares about, and that is the conversation you want. A candidate who talks for four minutes on question one fails the round even when every sentence is correct.

## Models and objectives

### Q1. Give the equations for a generative and a discriminative model and the difference between them.

A discriminative model learns the conditional directly:

$$P(y \mid x)$$

A generative model learns the joint, usually by factorising it:

$$P(x, y) = P(x \mid y)\,P(y), \qquad P(y \mid x) = \frac{P(x \mid y)P(y)}{\sum_{y'} P(x \mid y')P(y')}$$

Here $x$ is the input and $y$ the label. Discriminative examples: logistic regression, SVM, most neural nets. Generative examples: naive Bayes, Gaussian mixture models, LDA, and modern language models. The difference is what you model. Discriminative spends all its capacity on the decision boundary, so it usually wins on accuracy with enough data. Generative models the data itself, so it can sample new $x$, handle missing features, and needs less data when its assumptions hold.

**With numbers.** Two classes, spam at prior 0.3, and one feature with three values.

| Feature value | $P(x, \text{spam})$ | $P(x, \text{ham})$ | $P(x)$ | $P(\text{spam} \mid x)$ |
|---|---|---|---|---|
| a | 0.240 | 0.070 | 0.310 | 0.774 |
| b | 0.057 | 0.616 | 0.673 | 0.085 |
| c | 0.003 | 0.014 | 0.017 | 0.176 |

The discriminative model gives me the last column only. The joint gives the $P(x)$ column as well, so I can see that value c appears in 1.7 percent of messages and flag it as an outlier, and I can draw a fresh example by sampling $y$ then $x$. That one extra column is the whole difference.

### Q2. When would you prefer the generative model?

Three cases. First, small data: the generative model's assumptions act like a prior, so it reaches its asymptotic error faster. Naive Bayes beats logistic regression at small $n$, then loses as $n$ grows. Second, when I need to generate or impute: only a model of $P(x \mid y)$ can fill in a missing feature or draw a new sample. Third, anomaly detection, where I have almost no labels of the rare class, so I model $P(x)$ for normal data and flag low-likelihood points. Otherwise I take the discriminative model, because it does not waste capacity modelling the input distribution.

### Q3. Write linear regression and its closed-form solution.

The model is

$$\hat{y} = X w, \qquad \mathcal{L}(w) = \|Xw - y\|_2^2$$

where $X$ is the $n \times d$ design matrix, $y$ the target vector, $w$ the weights. Setting the gradient $2X^{\top}(Xw - y)$ to zero gives the normal equation:

$$w^{\star} = (X^{\top}X)^{-1}X^{\top}y$$

Geometrically this projects $y$ onto the column space of $X$, so the residual is orthogonal to every feature. It exists only when $X^{\top}X$ is invertible, which fails on collinear or wide data. Cost is about $O(nd^2 + d^3)$, so for large $d$ I use gradient descent or a QR solve instead of inverting.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The objective.** I write the squared error as a norm, $\|Xw - y\|_2^2$. It is a quadratic in $w$, therefore it has one stationary point and that point is the minimum.
2. **The gradient.** Differentiating gives $2X^{\top}(Xw - y)$. I set it to zero, which gives $X^{\top}Xw = X^{\top}y$, the normal equation. Then I invert $X^{\top}X$ to isolate $w$.
3. **The reading.** The zero condition says $X^{\top}(Xw - y) = 0$, so every column of $X$ is orthogonal to the residual. That is the geometric statement: the fit is the projection of $y$ onto the column space.

### Q4. Give me five things about linear regression and what each one means.

The five assumptions of ordinary least squares:

| Assumption | Meaning | What breaks if it fails |
|---|---|---|
| Linearity | $\mathbb{E}[y \mid x]$ is linear in the parameters | Biased predictions; curved residual plot |
| Independent errors | Residuals are uncorrelated across rows | Coefficients still unbiased, but standard errors too small |
| Homoscedasticity | Error variance is constant across $x$ | OLS is no longer the lowest-variance estimator; tests mislead |
| Normality of errors | Residuals are Gaussian | Point estimates fine, but small-sample p-values and intervals are wrong |
| No perfect multicollinearity | Columns of $X$ are linearly independent | $X^{\top}X$ is singular, so coefficients are unidentifiable and unstable |

The one that bites in practice is multicollinearity, because it does not hurt predictions but makes individual coefficients meaningless.

**With numbers.** I fit two features that correlate at 0.99994, because the second is the first plus a little noise. Then I nudge the labels slightly and refit.

| Fit | $w_1$ | $w_2$ | $w_1 + w_2$ | Largest prediction change |
|---|---|---|---|---|
| A | $-3.18$ | $6.19$ | $3.01$ | — |
| B | $-3.61$ | $6.63$ | $3.02$ | $0.013$ |

Each coefficient moved by about 0.44, which is 14 percent, from a tiny change in the labels. The sum stayed at 3.01 and no prediction moved by more than 0.013. So multicollinearity destroys the individual coefficients while leaving the fit itself intact, which is why it hurts inference and not prediction.

### Q5. Which of those assumptions matter for prediction and which only for inference?

Linearity and multicollinearity matter for prediction. If the true relationship is curved, the model is biased everywhere. If features are collinear, the fitted weights swing wildly with tiny data changes. Independence, homoscedasticity, and normality mostly matter for inference: they set the standard errors, the confidence intervals, and the p-values. If I only need good predictions and I measure error on held-out data, I can ignore them. If I am reporting that a coefficient is significant, I cannot, because correlated or heteroscedastic errors make the significance test overconfident.

### Q6. Explain logistic regression.

I model the probability with a sigmoid on a linear score:

$$p(y=1 \mid x) = \sigma(w^{\top}x + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

I fit it by minimising log-loss, which is the negative log-likelihood of the Bernoulli model. The gradient is $X^{\top}(p - y)/n$, features times residual, same shape as linear regression. Inverting the sigmoid gives the log-odds:

$$\log\frac{p}{1-p} = w^{\top}x + b$$

So logistic regression is linear in log-odds, and each weight $w_j$ is the change in log-odds per unit of feature $j$. There is no closed form, so I use gradient descent or Newton's method.

**Walk the derivation.** Three steps, and I say them in this order.

1. **The link.** The linear score $w^{\top}x + b$ can be any real number, however a probability must sit between 0 and 1. So I pass the score through the sigmoid, which squashes it into that range.
2. **The likelihood.** Each label is 0 or 1, therefore each point is a Bernoulli trial. I write both outcomes as one expression, $p^{y}(1-p)^{1-y}$, which equals $p$ when $y=1$ and $1-p$ when $y=0$. Multiplying it across all points gives the likelihood of the whole dataset.
3. **The loss.** A product of many probabilities underflows to zero in floating point. So I take the log, which turns the product into a sum, and I flip the sign, which turns maximising likelihood into minimising loss. That sum is binary cross-entropy.

**With numbers.** Say the model predicts heads with $p = 0.8$.

| Outcome | $p^{y}(1-p)^{1-y}$ | Loss $-\log(\cdot)$ |
|---|---|---|
| Heads, $y=1$ | $0.8^{1} \times 0.2^{0} = 0.8$ | $0.22$ |
| Tails, $y=0$ | $0.8^{0} \times 0.2^{1} = 0.2$ | $1.61$ |

A confident right answer costs almost nothing. The same confidence on a wrong answer costs seven times more, and the penalty grows without bound as the predicted probability approaches the wrong end. That asymmetry is why log-loss produces a calibrated model and why accuracy does not.

### Q7. Why is the logistic decision boundary linear?

I predict class 1 when $p > 0.5$. The sigmoid is monotone and $\sigma(0) = 0.5$, so $p > 0.5$ exactly when $w^{\top}x + b > 0$. That condition is a hyperplane in feature space, so the boundary is linear by construction. The sigmoid only bends how confidence changes as you move away from the plane; it never bends the plane. To get a curved boundary I have to change the features, not the link function, by adding polynomial terms or a kernel. The same argument holds for softmax: the boundary between two classes is where their logits are equal, which is again linear.

### Q8. Explain naive Bayes and why it works despite a false assumption.

It is generative. I model

$$P(y \mid x) \propto P(y)\prod_{j=1}^{d} P(x_j \mid y)$$

The assumption is that features are conditionally independent given the class, which is almost always false, for example in text where words co-occur. It still works because classification only needs the correct argmax, not calibrated probabilities. Correlated features double-count evidence, so the posterior is pushed toward 0 or 1, but the ranking of classes often survives. Therefore accuracy stays good while the probabilities are badly overconfident. It also trains in one pass and needs very little data, so it is a strong baseline for text.

**With numbers.** Take equal priors, so the prior odds are 1, and one feature whose likelihood ratio for class 1 is 3.

| Feature copies | Posterior odds | $P(y=1 \mid x)$ |
|---|---|---|
| One | $3$ | $0.75$ |
| Two identical copies | $9$ | $0.90$ |
| Three identical copies | $27$ | $0.964$ |

Duplicating one piece of evidence pushes the probability from 0.75 to 0.96, so the confidence is badly wrong. The argmax is class 1 in every row. That is the whole story: correlation moves the number and not the decision.

### Q9. LDA versus logistic regression.

Both give a linear boundary, but they get there differently. LDA is generative: it assumes each class is Gaussian with a shared covariance $\Sigma$, then applies Bayes' rule, which makes the log-odds linear. Logistic regression models that same log-odds directly and fits it by maximum likelihood, assuming nothing about $P(x)$. So LDA is more efficient when the Gaussian and shared-covariance assumptions hold, especially with few samples or well-separated classes where logistic regression's weights diverge. Logistic regression is more robust when they do not hold, and it handles outliers and non-Gaussian features better. LDA also gives you a supervised projection for free.

### Q10. Explain the SVM, the margin, and the hinge loss.

The SVM finds the hyperplane with the largest margin to the nearest points. The hard-margin problem is

$$\min_w \tfrac{1}{2}\|w\|_2^2 \quad \text{subject to} \quad y_i(w^{\top}x_i + b) \ge 1$$

with margin width $2/\|w\|$. Real data is not separable, so the soft-margin form uses hinge loss:

$$\mathcal{L} = \sum_i \max\big(0,\; 1 - y_i(w^{\top}x_i + b)\big) + \lambda\|w\|_2^2$$

Hinge loss is zero once a point is correctly classified past the margin, so only the support vectors, the points on or inside the margin, affect the solution. That is what makes the SVM sparse in examples.

**With numbers.** Say $w = (3, 4)$, so $\|w\| = 5$ and the margin width is $2/5 = 0.4$.

| Point, $y_i(w^{\top}x_i + b)$ | Hinge loss | Role |
|---|---|---|
| $1.2$ | $0$ | Outside the margin, ignored |
| $0.6$ | $0.4$ | Inside the margin, a support vector |
| $-0.3$ | $1.3$ | Misclassified, a support vector |

The first point contributes nothing to the gradient, so I could delete it and get the same hyperplane. Only the points that score below 1 shape the solution, and that is what sparse in examples means.

### Q11. What is the kernel trick?

The SVM's dual depends on the data only through inner products $x_i^{\top}x_j$. So I replace that inner product with a kernel $K(x_i, x_j)$, which is an inner product in some higher-dimensional feature space $\phi$:

$$K(x_i, x_j) = \phi(x_i)^{\top}\phi(x_j)$$

I never compute $\phi$ itself, which may be infinite-dimensional. The RBF kernel $K = \exp(-\gamma\|x_i - x_j\|^2)$ is the common choice. This buys a nonlinear boundary at the cost of a linear one. The price is scaling: the kernel matrix is $n \times n$, so kernel SVMs get impractical past roughly a hundred thousand rows.

### Q12. Explain k-nearest neighbours and its tradeoff.

To predict, I find the $k$ closest training points under some distance and take their majority vote or mean. There is no training step, which is why it is called lazy learning: all the work happens at query time, costing $O(nd)$ per prediction without an index. Small $k$ gives low bias and high variance, so the boundary is jagged; large $k$ smooths it and raises bias. It needs scaled features, because distance is dominated by whatever feature has the largest units. It also degrades badly in high dimensions, because distances concentrate and "nearest" stops meaning "similar".

### Q13. How does a decision tree choose a split?

It is greedy. At each node it tries every feature and every threshold, and picks the split that most reduces impurity. For classification the criteria are

$$H(S) = -\sum_c p_c \log_2 p_c, \qquad G(S) = 1 - \sum_c p_c^2$$

and it maximises information gain, the parent impurity minus the weighted average child impurity. Gini is cheaper because it avoids logarithms, and it ranks splits almost identically to entropy. For regression the criterion is variance reduction, that is squared error. The split is chosen locally, so a tree never backtracks, and that greediness is why single trees are high-variance and need pruning or an ensemble.

**With numbers.** A node holds 100 rows, 50 positive and 50 negative. A candidate split makes two children of 50 rows, each 40 to 10.

| Measure | Parent | Each child | Gain |
|---|---|---|---|
| Entropy | $1.000$ | $0.722$ | $0.278$ |
| Gini | $0.500$ | $0.320$ | $0.180$ |

Both children have the same impurity here, so the weighted average is just the child value. Entropy and Gini disagree on the size of the gain but agree that this split is an improvement, which is why the choice between them almost never changes the tree.

### Q14. Why do trees not need feature scaling?

A split is a test of the form $x_j < t$. Any monotone transform of $x_j$ maps thresholds to thresholds, so the set of reachable partitions is unchanged, and the tree finds the same structure. Therefore scaling, log transforms, and rank transforms do nothing for a tree. Distance-based and gradient-based models are the opposite: k-NN, SVM, PCA, k-means, and any penalised linear model all depend on the units, because distance or the penalty compares features to each other. That is the clean rule I use: if the model compares features across dimensions, scale them; if it splits one feature at a time, do not bother.

## Regularisation and the bias-variance picture

### Q15. Explain the bias-variance tradeoff, deeper.

For squared error, expected test error at a point decomposes as

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{variance}} + \sigma^2$$

The expectation is over training sets drawn from the same distribution. Bias is the error of the average model, so it measures wrong assumptions. Variance is how much the fitted model moves when the training set changes. $\sigma^2$ is label noise and no model removes it. Capacity moves the two in opposite directions: a deeper tree or more features lowers bias and raises variance. The tradeoff exists because I only have one finite sample.

**Walk the derivation.** Three steps, and I say them in this order.

1. **Add and subtract the mean model.** Write the error as $y - \hat{f}$, then insert $\mathbb{E}[\hat{f}]$ and subtract it again. That splits the error into how far the average model is from the truth, plus how far this model is from the average model.
2. **Square it.** The cross term contains $\hat{f} - \mathbb{E}[\hat{f}]$, whose expectation is zero by construction, so the cross term vanishes. Only the two squares survive.
3. **Add the label noise.** The target itself is $f(x)$ plus independent noise, so its variance $\sigma^2$ adds on. Nothing in the model touches it, therefore it is the floor.

**With numbers.** Twenty noisy points from a smooth curve, fitted with polynomials of rising degree.

| Degree | Train RMSE | Validation RMSE |
|---|---|---|
| 1 | $0.467$ | $0.459$ |
| 3 | $0.252$ | $0.197$ |
| 9 | $0.165$ | $2.19$ |

Train error falls at every step, because more capacity always fits the sample better. Validation error turns up hard between degree 3 and degree 9. That turn is variance overtaking bias, and it is only visible on held-out data.

### Q16. What actually moves each term?

Bias goes down when I add capacity: more features, higher polynomial degree, deeper trees, less pruning, weaker penalty. Variance goes down when I add constraint or data: stronger regularisation, fewer parameters, more training rows, averaging over models, early stopping. Two levers are special. More data lowers variance without raising bias, which is why it is the only free improvement. Bagging lowers variance without touching bias, because averaging does not change the expected model. Boosting is the mirror image: it lowers bias by adding terms sequentially and pays in variance. Everything else is a genuine trade.

### Q17. Does the tradeoff always hold, given double descent?

The decomposition is an identity, so it always holds. What is not a law is the U-shaped curve. In heavily overparameterised models, test error falls, rises to a peak at the interpolation threshold where parameters roughly equal samples, then falls again as capacity grows further. The reason is that beyond that threshold many solutions fit the data exactly, and the optimiser's implicit bias picks a low-norm one, so effective complexity drops even as parameter count rises. So I say: the tradeoff is about effective complexity, not parameter count, and parameter count stopped being a good proxy for it.

### Q18. Explain regularisation, deeper.

Regularisation is any change that raises training error to lower test error. The explicit form adds a penalty to the objective:

$$\hat{w} = \arg\min_w \; \mathcal{L}(w) + \lambda\, \Omega(w)$$

$\Omega$ measures complexity and $\lambda$ sets how much I care. This is a constrained problem in disguise: by Lagrange duality, penalising $\|w\|$ is the same as optimising inside a ball of some radius. The effect is always the same trade, more bias for less variance, so $\lambda$ is a dial on the decomposition. Implicit forms do the same thing without a penalty term: early stopping, dropout, data augmentation, small batches, and even the optimiser's preference for low-norm solutions.

### Q19. L1 versus L2 — give the penalties and the difference.

$$\mathcal{L}_{\text{ridge}} = \mathcal{L} + \lambda\sum_j w_j^2, \qquad \mathcal{L}_{\text{lasso}} = \mathcal{L} + \lambda\sum_j |w_j|$$

L2 has gradient $2\lambda w_j$, which shrinks in proportion to the weight, so it gets smaller as the weight gets smaller and never reaches zero. L1 has subgradient $\lambda\,\text{sign}(w_j)$, a constant pull that does not weaken near zero, so it pushes small weights exactly to zero and gives a sparse model. Therefore I use L2 when all features carry a little signal and I want stability under correlation, and L1 when I want selection and a model I can read.

### Q20. Give the geometric reason L1 gives exact zeros.

Think of the constrained form: minimise the loss subject to $\|w\|_1 \le t$ or $\|w\|_2 \le t$. The solution is where the elliptical loss contours first touch the constraint region. The L1 region is a diamond with corners on the axes; the L2 region is a smooth ball. A corner sticks out, and a corner lies on an axis, which means some coordinates are exactly zero. An expanding ellipse is very likely to touch a corner first. The ball has no corners, so the touch point almost surely has every coordinate non-zero. Sparsity comes from the non-differentiable corners, not from the penalty being smaller.

### Q21. What prior does L1 regularisation correspond to, and what prior does L2 correspond to?

L2 is a Gaussian prior on the weights, L1 is a Laplace prior. The derivation: MAP estimation maximises the posterior, so

$$\hat{w} = \arg\max_w \big[\log P(D \mid w) + \log P(w)\big]$$

which is log-likelihood plus log-prior. A zero-mean Gaussian $P(w_j) \propto \exp(-w_j^2/2\tau^2)$ has log-density $-w_j^2/2\tau^2$, a squared term, which is exactly ridge with $\lambda = 1/2\tau^2$. A Laplace $P(w_j) \propto \exp(-|w_j|/b)$ has log-density $-|w_j|/b$, an absolute-value term, which is lasso. So regularisation strength is inverse prior width: large $\lambda$ means a narrow prior tightly concentrated at zero.

**Walk the derivation.** Three steps, and I say them in this order.

1. **MAP is likelihood plus prior.** Bayes gives $P(w \mid D) \propto P(D \mid w)P(w)$. I take the log, so the product becomes a sum: log-likelihood plus log-prior. Maximising that sum is MAP.
2. **A Gaussian prior gives L2.** Its log-density is $-w_j^2/2\tau^2$. That is a squared term in $w$, so flipping the sign to make it a loss gives exactly the ridge penalty.
3. **A Laplace prior gives L1.** Its log-density is $-|w_j|/b$. That is an absolute value, so the same flip gives exactly the lasso penalty. The prior shape picks the penalty shape.

**With numbers.** One coefficient whose unpenalised value is $0.4$, with an orthonormal design so each weight is solved on its own.

| Penalty strength | Ridge, $w/(1+\lambda)$ | Lasso, soft threshold |
|---|---|---|
| $\lambda = 0.2$ | $0.333$ | $0.200$ |
| $\lambda = 0.5$ | $0.267$ | $0.000$ |

Ridge shrinks the weight by a fraction, so it never arrives at zero however hard I push. Lasso subtracts a fixed amount, so once the penalty exceeds the gradient at 0.4 the weight is exactly zero. That is selection versus shrinkage in two rows.

### Q22. Why does the Laplace prior give sparsity but the Gaussian does not?

Look at the density at zero. The Laplace has a sharp peak there, so it puts a lot of prior mass very near zero and its log-density has a kink. That kink is the constant-size gradient that survives all the way to the origin, so the MAP solution sits exactly at zero for any weak feature. The Gaussian is flat and smooth at zero, so its pull vanishes as the weight vanishes and no coordinate is ever driven fully to zero. One caution worth saying: the Bayesian posterior mean under a Laplace prior is not sparse. Only the MAP point estimate is.

### Q23. What is elastic net and when do you use it?

It adds both penalties:

$$\mathcal{L} + \lambda_1\sum_j |w_j| + \lambda_2\sum_j w_j^2$$

I use it when features are correlated and I still want selection. Pure lasso behaves badly there: given a group of correlated features it picks one almost arbitrarily and zeroes the rest, and that choice flips with a small data change. The L2 term makes the objective strictly convex, which shares the weight across the correlated group and stabilises the selection. Elastic net can also select more than $n$ features, which lasso cannot when $d > n$. The cost is a second hyperparameter to tune.

### Q24. Give ridge in closed form and say why the diagonal term helps.

$$w^{\star} = (X^{\top}X + \lambda I)^{-1}X^{\top}y$$

$X^{\top}X$ is symmetric positive semi-definite, so its eigenvalues are non-negative but can be zero or tiny when features are collinear or when $d > n$. Adding $\lambda I$ shifts every eigenvalue up by $\lambda$, so the matrix becomes positive definite and always invertible. The condition number goes from $\sigma_{\max}/\sigma_{\min}$ to $(\sigma_{\max}+\lambda)/(\sigma_{\min}+\lambda)$, which is smaller. So ridge both regularises and conditions. In the SVD view it shrinks each direction by $\sigma^2/(\sigma^2+\lambda)$, so low-variance directions are damped hardest.

**With numbers.** Take a two-feature problem where $X^{\top}X$ has eigenvalues 10 and 0.01.

| $\lambda$ | Eigenvalues | Condition number |
|---|---|---|
| $0$ | $10,\; 0.01$ | $1000$ |
| $0.1$ | $10.1,\; 0.11$ | $91.8$ |
| $1.0$ | $11,\; 1.01$ | $10.9$ |

A shift of 0.1 cuts the condition number by a factor of eleven, because it is huge relative to 0.01 and negligible relative to 10. The same asymmetry appears in the shrinkage factor $\sigma^2/(\sigma^2+\lambda)$, which is 0.99 for the strong direction and 0.09 for the weak one. Ridge leaves the well-determined directions alone and crushes the badly determined ones.

### Q25. Why is regularisation described as a bias-variance instrument?

Because $\lambda$ moves along the decomposition directly. At $\lambda = 0$ I get the unconstrained fit: lowest bias, highest variance. As $\lambda$ grows I shrink weights toward zero, which makes the fitted model less sensitive to the particular training sample, so variance falls, and it also pulls the average model away from the truth, so bias rises. At $\lambda \to \infty$ every weight is zero, giving a constant predictor: zero variance, maximum bias. Test error is the sum, so it is U-shaped in $\lambda$ and I find the bottom by cross-validation. That is the whole mechanism.

### Q26. How is early stopping regularisation?

I stop training when validation loss stops improving, before the training loss bottoms out. For a linear model with gradient descent from zero, this is close to ridge with an explicit correspondence: each step moves the weights along the gradient, and directions with small curvature are learned slowly, so stopping early leaves them near zero, which is what the ridge shrinkage factor does. Fewer steps therefore means a smaller effective weight norm and lower effective capacity. It is cheap because it needs no extra hyperparameter search: the number of epochs is the regularisation strength, and I read it off the validation curve.

### Q27. What does dropout approximate?

During training I zero each unit independently with probability $p$, then rescale by $1/(1-p)$ so the expected activation is unchanged. At test time I use the full network. Two readings. First, it samples a different thinned sub-network each step, so training approximates an exponentially large ensemble and the test-time full network approximates averaging them. Second, it stops co-adaptation: no unit can rely on any particular other unit being present, so features must be individually useful, which is a redundancy pressure. For a linear model with squared error, dropout is provably equivalent to L2 on scaled weights. Typical $p$ is 0.1 to 0.5.

**With numbers.** Take a hidden layer of 100 units with $p = 0.5$. Each unit is kept with probability 0.5 and then scaled by $1/(1-p) = 2$, so a unit that outputs 1.0 contributes 2.0 half the time and 0 half the time, and the expectation is 1.0, unchanged. The number of distinct thinned sub-networks is $2^{100}$, which is about $1.3 \times 10^{30}$. That count is why dropout is described as an ensemble I could never train explicitly.

### Q28. Why is data augmentation regularisation?

Because it encodes an invariance I already believe. If I know a flipped or slightly rotated image has the same label, augmentation forces the model to give both the same output, which removes solutions that would have used the orientation. That shrinks the effective hypothesis space, which is exactly what a penalty does, so variance falls. It differs from a penalty in that it also acts like extra data: the model sees more distinct inputs per epoch, so it lowers variance without raising bias much. The risk is picking an invariance that is false, for example flipping digits, which then injects real bias.

### Q29. Weight decay versus L2 in adaptive optimisers.

With plain SGD they are the same thing: adding $\lambda\|w\|^2$ to the loss produces the gradient term $2\lambda w$, which is identical to shrinking $w$ by a constant fraction each step. With Adam they are not. Adam divides every gradient component by its running root-mean-square, so the L2 term gets divided too. Weights with large historical gradients are then decayed less, which is backwards. AdamW fixes this by removing the penalty from the loss and applying the decay directly to the weights after the adaptive step:

$$w \leftarrow w - \eta\,\hat{g} - \eta\lambda w$$

That decoupling is why AdamW is the default for transformers.

**With numbers.** Set the decay to 0.01 and take two weights, both at 1.0, whose gradient root-mean-square histories are 10 and 0.1.

| Gradient RMS | L2 in the loss, effective pull | AdamW, decoupled |
|---|---|---|
| $10$ | $0.001$ | $0.01$ |
| $0.1$ | $0.100$ | $0.01$ |

Putting the penalty in the loss makes the pull vary by a factor of 100 across the two weights, and it is weakest exactly where gradients are largest. AdamW applies 0.01 to both. That is the bug and the fix in one table.

## Ensembles

### Q30. Explain bagging and why averaging reduces variance.

Bagging trains $B$ copies of the same model on bootstrap resamples, then averages the predictions. The argument is the variance of a mean. If each model has variance $\sigma^2$ and pairwise correlation $\rho$, the average has variance

$$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

The second term vanishes as $B$ grows, but the first does not. So averaging only helps to the extent the models are decorrelated, and $\rho$ is the ceiling. Bias is unchanged, because the expected prediction of the average equals the expected prediction of one model. Therefore bagging suits high-variance, low-bias learners: deep unpruned trees, not linear models.

**Walk the derivation.** Three steps, and I say them in this order.

1. **Write the variance of a sum.** For $B$ predictors the variance of the sum is the sum of the variances plus every covariance, that is $B\sigma^2 + B(B-1)\rho\sigma^2$.
2. **Divide by $B$ squared.** The average is the sum over $B$, so its variance is that expression over $B^2$. Collecting terms gives $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$.
3. **Take $B$ large.** The second term goes to zero, however the first term has no $B$ in it. So correlation, not the number of models, sets the floor.

**With numbers.** Fix $\sigma^2 = 1$ and vary the correlation.

| Correlation | $B = 10$ | $B = 100$ | Floor |
|---|---|---|---|
| $\rho = 0$ | $0.100$ | $0.010$ | $0$ |
| $\rho = 0.5$ | $0.550$ | $0.505$ | $0.5$ |
| $\rho = 0.9$ | $0.910$ | $0.901$ | $0.9$ |

At $\rho = 0.9$, going from ten trees to a hundred buys a variance drop of 0.009, which is nothing. The whole gain sits in decorrelation, which is exactly what a random forest goes after.

### Q31. What do random forests add?

They attack the $\rho$ term. Bootstrapping alone leaves trees highly correlated, because one or two strong features dominate the top split in every tree. So at each split a random forest considers only a random subset of features, typically $\sqrt{d}$ for classification and $d/3$ for regression. That forces some trees to split on weaker features, which decorrelates them and drops the floor $\rho\sigma^2$. Each individual tree gets slightly worse and the ensemble gets better. The forest also gives out-of-bag error for free, since each tree can be scored on the roughly one third of rows its bootstrap sample missed.

### Q32. How is boosting different in what it reduces?

Boosting fits models sequentially, each one on what the previous ones got wrong, then adds them:

$$F_M(x) = \sum_{m=1}^{M} \nu\, h_m(x)$$

Each new term reduces the training error of the running sum, so the ensemble mainly reduces bias. That is why the base learners are weak and high-bias: stumps or trees of depth three to six. Bagging is the opposite, parallel and variance-reducing, so it uses deep low-bias trees. The consequence is that boosting can overfit if $M$ is too large, while bagging essentially cannot, so I pick $M$ by early stopping on a validation set.

### Q33. How does gradient boosting work?

It is gradient descent in function space. Start with a constant $F_0$, usually the mean. At step $m$, compute for each training point the negative gradient of the loss with respect to the current prediction:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

Fit a small regression tree $h_m$ to those values, then update

$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$

with learning rate $\nu$, typically 0.01 to 0.1. For squared loss the negative gradient is exactly the residual $y_i - F_{m-1}(x_i)$, which is why the plain version is described as fitting residuals. Any differentiable loss works because only the gradient changes.

**Walk the derivation.** Four steps, and I say them in this order.

1. **Treat the prediction as the variable.** I ask which direction in prediction space lowers the loss fastest. That direction is the negative gradient of the loss with respect to $F(x_i)$, evaluated at each training point.
2. **Fit a learner to it.** The negative gradient is only defined at the training points, so I fit a small tree to those values. The tree is how I extend the step to the rest of the input space.
3. **Note the special case.** For squared loss the negative gradient is $y_i - F(x_i)$, the plain residual. That is why the standard explanation says "fit the residual".
4. **Take a small step.** I add $\nu h_m$ rather than $h_m$, then recompute the gradients and repeat.

**With numbers.** Three points with targets 2, 4 and 6, learning rate $\nu = 0.5$, and a stump each round.

| Round | Predictions | Residuals | Squared error |
|---|---|---|---|
| 0 | $4,\; 4,\; 4$ | $-2,\; 0,\; 2$ | $8.00$ |
| 1 | $3.5,\; 3.5,\; 5$ | $-1.5,\; 0.5,\; 1$ | $3.50$ |
| 2 | $2.75,\; 3.875,\; 5.375$ | $-0.75,\; 0.125,\; 0.625$ | $0.97$ |

Error falls from 8 to 0.97 in two rounds, and no single stump got close on its own. Each round only had to explain what was left over.

### Q34. What does the shrinkage factor do?

$\nu$ scales every tree's contribution before it is added. It is regularisation: small steps mean no single tree can dominate, so the ensemble explores more directions and generalises better. The tradeoff is straightforward, since halving $\nu$ roughly doubles the number of trees needed for the same training fit. So $\nu$ and $M$ trade off directly, and the standard recipe is to set $\nu$ small, around 0.05, then choose $M$ by early stopping. Two other regularisers sit alongside it: row subsampling per tree, which makes it stochastic gradient boosting, and a limit on tree depth or leaf count.

### Q35. How does AdaBoost work and how does it relate?

AdaBoost keeps a weight on each training example. It fits a weak learner, computes its weighted error rate $\varepsilon_m$, gives it a vote

$$\alpha_m = \tfrac{1}{2}\log\frac{1 - \varepsilon_m}{\varepsilon_m}$$

then multiplies the weights of misclassified points up and correct points down, and renormalises. The final prediction is the sign of $\sum_m \alpha_m h_m(x)$. The relationship: AdaBoost is exactly gradient boosting with exponential loss $e^{-y F(x)}$, where the example weights are the gradients. That also explains its weakness, because exponential loss punishes outliers extremely hard, so AdaBoost is sensitive to label noise.

**With numbers.** A weak learner gets 30 percent of the weighted mass wrong.

| Quantity | Value |
|---|---|
| $\varepsilon_m$ | $0.30$ |
| $\alpha_m = \tfrac{1}{2}\log\frac{1-\varepsilon}{\varepsilon}$ | $0.424$ |
| Weight multiplier if wrong, $e^{\alpha}$ | $1.53$ |
| Weight multiplier if right, $e^{-\alpha}$ | $0.65$ |

The ratio between the two multipliers is 2.33, so a missed point becomes 2.33 times more important relative to a hit, every round. Apply that to a mislabelled point for fifty rounds and it dominates the training set, which is the noise sensitivity in one number.

### Q36. What did XGBoost and LightGBM actually change?

XGBoost changed the objective and the engineering. It uses a second-order Taylor expansion of the loss, so each split uses gradient and Hessian rather than gradient alone, which gives a closed-form optimal leaf value. It adds explicit L1 and L2 penalties on leaf weights and a penalty per leaf, plus column subsampling, sparsity-aware handling of missing values, and cache-aware histogram splitting. LightGBM changed the tree growth and the candidate search: leaf-wise growth instead of level-wise, which reaches lower loss per leaf, gradient-based one-side sampling that keeps large-gradient rows, and bundling of mutually exclusive sparse features. LightGBM is usually faster; accuracy is comparable.

### Q37. Bagging versus boosting, in one table.

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel, independent models | Sequential, each fits previous errors |
| Data per model | Bootstrap resample | Full data, reweighted or gradient targets |
| Base learner | Deep, low bias, high variance | Weak, high bias, low variance |
| Reduces | Variance | Bias |
| Overfits with more models | Essentially no | Yes, so early-stop on $M$ |
| Combination | Equal-weight average or vote | Weighted sum with learning rate |
| Noise sensitivity | Robust | Sensitive, it chases mislabelled points |

The row that decides the design is which term you reduce, because it fixes the base learner depth.

### Q38. What is stacking?

I train several different base models, then train a second-level model, the meta-learner, on their predictions as features. The critical detail is how those features are made: they must be out-of-fold predictions. I split the training data into $k$ folds, and each base model predicts a fold only when it was trained on the others. If I use in-sample predictions instead, a base model that memorised the training set looks perfect, the meta-learner trusts it completely, and the stack collapses at test time. The meta-learner is usually something simple like regularised logistic regression, to avoid overfitting a small prediction matrix.

### Q39. When does an ensemble not help?

When the members are correlated, which is the $\rho\sigma^2$ floor again. Averaging ten models trained with the same architecture, the same features, and the same seed range gains almost nothing. Diversity is the whole product, so it must come from somewhere: different algorithms, different feature subsets, different resamples, or different objectives. It also does not help when the model is already high-bias and underfitting, since averaging identical underfits gives the same underfit. And it costs: $B$ times the inference latency and memory, plus loss of interpretability, which sometimes rules it out regardless of accuracy.

## Evaluation and validation

### Q40. Explain cross-validation.

I split the training data into $k$ folds, train on $k-1$ and evaluate on the held-out one, then rotate and average the $k$ scores. It gives a lower-variance estimate of generalisation error than a single split, because every row is used for validation exactly once. $k=5$ or $k=10$ is standard. Small $k$ means each model sees less data, so the estimate is pessimistically biased; large $k$ means the training sets overlap heavily, so the fold scores are correlated and cost rises linearly. Leave-one-out is the extreme: nearly unbiased, high variance, and usually too expensive.

**With numbers.** I fixed a 500-row dataset and one model, then re-ran the whole cross-validation sixty times with different random fold assignments.

| Folds | Mean score | Standard deviation of the estimate |
|---|---|---|
| $k = 5$ | $0.834$ | $0.0140$ |
| $k = 10$ | $0.836$ | $0.0125$ |

Doubling the work moved the standard deviation from 0.014 to 0.013, which is about 11 percent, and moved the mean by 0.002. So the honest statement is that ten folds cost twice as much for a small gain, and neither number is precise enough to justify picking a model that wins by half a point.

### Q41. When is k-fold the wrong thing to do?

Two cases. With time series, random folds let the model train on the future and predict the past, which leaks and gives a score you can never reproduce in production. I use forward-chaining instead: train on everything before time $t$, validate on the window after it, roll forward. With grouped data, where several rows come from the same patient, user, or document, random folds put the same group in train and validation, so the model recognises the group rather than the pattern. I use grouped k-fold, keyed on the group ID. With heavy class imbalance I also stratify, so every fold has the same class ratio.

### Q42. Define precision, recall, and F1.

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,PR}{P + R}$$

Precision answers: of the things I flagged, how many were right. Recall answers: of the things I should have flagged, how many did I catch. F1 is their harmonic mean, which is low unless both are decent, so it will not let a model score well by being extreme on one. The choice is a cost question. For a spam filter I want precision, because a false positive deletes real mail. For cancer screening I want recall, because a miss is far more expensive than a re-test.

**With numbers.** A thousand cases, sixty of them positive.

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | $TP = 40$ | $FN = 20$ |
| **Actually negative** | $FP = 10$ | $TN = 930$ |

Precision is $40/50 = 0.80$. Recall is $40/60 = 0.667$. F1 is $2(0.8)(0.667)/1.467 = 0.727$. Accuracy is $970/1000 = 0.97$, which sounds excellent and hides the fact that I missed a third of the positives. That gap between 0.97 and 0.727 is the reason I quote the three numbers and not the one.

### Q43. ROC-AUC versus PR-AUC.

ROC plots true positive rate against false positive rate over all thresholds; AUC is the probability a random positive is ranked above a random negative. PR plots precision against recall. The difference is the denominator. False positive rate divides by the number of negatives, which is huge under imbalance, so a large absolute number of false positives barely moves the ROC curve and the AUC still looks good. Precision divides by predicted positives, so it feels every false positive. Therefore I use PR-AUC when positives are rare and I care about them, and ROC-AUC when the classes are roughly balanced or I care about both errors equally.

**With numbers.** One model, one score distribution, two prevalences. I scored positives and negatives from separated Gaussians, then measured both curves.

| Setting | Prevalence | ROC-AUC | PR-AUC | Precision at a fixed cut |
|---|---|---|---|---|
| Balanced | $0.5$ | $0.855$ | $0.853$ | $0.81$ |
| One in a thousand | $0.001$ | $0.851$ | $0.023$ | $0.004$ |

The ranking never changed, so ROC-AUC moved by 0.004. PR-AUC fell by a factor of 37, and precision at the same cut fell from 81 percent to under half a percent. The negatives that flooded in are invisible to a false positive rate and are the entire story for anyone who has to read the flagged cases.

### Q44. What is calibration and how do you check it?

A model is calibrated when its stated probabilities match observed frequencies: of the cases it calls 0.7, about 70 percent should be positive. I check it with a reliability diagram, binning predictions and plotting mean predicted probability against observed rate, and I summarise it with expected calibration error, the weighted average gap across bins. Ranking metrics like AUC are invariant to any monotone transform, so a model can rank perfectly and be badly calibrated. Fixes are Platt scaling, which fits a sigmoid on a held-out set, and isotonic regression, which is non-parametric and more flexible but needs more data.

**With numbers.** Take the bucket of cases the model scores near 0.9. Suppose there are 200 of them and 120 turn out positive, so the observed rate is 0.60. The gap is 0.30, and if that bucket is a tenth of the data it contributes $0.1 \times 0.30 = 0.03$ to expected calibration error. The ranking inside the bucket may still be perfect, so AUC does not notice any of this. If I act on 0.9 as a real probability, for example by pricing a decision on it, I am wrong by 50 percent in relative terms.

### Q45. Why is accuracy a bad metric under class imbalance?

Because the majority-class baseline already scores high. With 1 percent positives, a model that always says "negative" is 99 percent accurate and useless, so accuracy cannot distinguish it from a model that works. Accuracy weights every error equally, and under imbalance the errors I care about are exactly the rare ones. So I use precision, recall, F1, or PR-AUC on the minority class, and I look at the full confusion matrix. If the two error types have different costs, I say so explicitly and choose the operating threshold by expected cost, not by the default 0.5.

**With numbers.** Ten thousand rows with 100 positives, so prevalence is 1 percent. A model that always predicts negative gets 9900 of 10000 right, that is 99 percent accuracy, with recall 0, precision undefined, and F1 0. A useful model that catches 70 of the 100 positives at the cost of 200 false positives scores 97.7 percent accuracy, which is lower. Accuracy ranks the useless model above the useful one, and that single inversion is the whole argument.

### Q46. How do you diagnose overfitting versus underfitting from curves?

I plot training and validation error against training set size or epochs. Underfitting: both errors are high and close together, and they flatten early. The model lacks capacity or the features lack signal, so I add capacity, add features, or train longer. Overfitting: training error is low and still falling while validation error is much higher and starting to rise. The gap is the variance. I add data, add regularisation, or reduce capacity. The useful third read is the trajectory. If validation error is still dropping when the data runs out, more data will help; if it has flattened, more data will not.

### Q47. What is data leakage? Give two concrete examples.

Leakage is any information in training that will not be available at prediction time, which makes validation scores optimistic and production performance collapse. First example: fitting the scaler, the imputer, or the feature selector on the full dataset before splitting. The validation fold's mean and variance then influenced the training transform, so the score is contaminated. The fix is to fit every transform inside the training fold, using a pipeline. Second example: a target-derived feature, such as using "number of days until account closed" to predict churn, or an ID that correlates with the label because rows were sorted by outcome.

### Q48. Why is the test set touched once?

Because every time I look at it and make a decision, I am fitting to it. Choosing between twenty models by test score means the winner is partly the model that got lucky on that particular sample, so the reported number is optimistically biased and no longer an estimate of generalisation. That is multiple-comparison overfitting on a single held-out set. So the split is three ways: train fits parameters, validation selects hyperparameters and architecture and stopping point, and test is opened once at the end to report a number. If I do reuse the test set, the honest thing is to state how many times.

### Q49. How do you choose a classification threshold?

Not by defaulting to 0.5, since that is only right when classes are balanced and errors cost the same. I pick it on the validation set against the objective. If I have costs, I choose the threshold that minimises expected cost, $C_{FP}\cdot FP + C_{FN}\cdot FN$. If I have a service constraint, such as reviewers who can only handle a hundred cases a day, I set the threshold at that capacity and report the recall it delivers. If I only need a balanced summary, I take the point that maximises F1. Then I check the threshold still holds on the test set, because the base rate drifts.

**With numbers.** A calibrated model, prevalence 10 percent, a missed positive costing ten and a false alarm costing one.

| Threshold | False positives | False negatives | Cost per 1000 rows |
|---|---|---|---|
| $0.05$ | $107641$ | $1681$ | $622$ |
| $0.091$ | $69596$ | $4621$ | $579$ |
| $0.30$ | $5108$ | $17085$ | $880$ |
| $0.50$ | $188$ | $19798$ | $991$ |

The minimum sits at $C_{FP}/(C_{FP}+C_{FN}) = 1/11 = 0.091$, exactly as the theory says. The default 0.5 costs 71 percent more than the right threshold, on the same model with no retraining.

## Features and data

### Q50. Standardisation versus normalisation, and which models need it.

Standardisation subtracts the mean and divides by the standard deviation, $z = (x - \mu)/\sigma$, giving mean zero and unit variance with no bounded range. Normalisation, usually min-max, maps to $[0,1]$ by $(x - \min)/(\max - \min)$, which preserves shape but is destroyed by a single outlier. I standardise by default, and use min-max only when I need a bounded input, such as pixel values. Models that need it: anything distance-based (k-NN, k-means, SVM, PCA) and anything penalised (ridge, lasso), because the penalty compares weights across features. Models that do not: trees and all tree ensembles.

**With numbers.** Two features, one ranging over zero to one and one ranging over zero to a hundred thousand.

| Data | Largest eigenvalue of $X^{\top}X$ | Condition number | Largest stable learning rate |
|---|---|---|---|
| Raw | $6.4 \times 10^{11}$ | $2.3 \times 10^{10}$ | $3 \times 10^{-12}$ |
| Standardised | $217$ | $1.2$ | $0.009$ |

On the raw data any step big enough to move the small feature blows up the large one, so the usable learning rate collapses by nine orders of magnitude and the loss surface is a knife-edge ravine. Standardising makes the two curvatures almost equal, and gradient descent then goes straight down.

### Q51. One-hot versus target encoding.

One-hot makes a binary column per category. It is safe and assumes no ordering, but it explodes dimension on high-cardinality features like postcodes, which hurts linear models and makes tree splits weak because each column is mostly zero. Target encoding replaces the category with a statistic of the label within it, usually the mean, so one column carries the signal and cardinality stops mattering. The danger is leakage: a category seen once gets that row's own label as its feature, and the model reads the answer. The fix is out-of-fold encoding plus smoothing toward the global mean in proportion to how rare the category is.

**With numbers.** Global positive rate 0.10, smoothing weight 20, and the smoothed value is $(k\bar{y}_c + 20 \times 0.10)/(k + 20)$ for a category seen $k$ times.

| Category | Count | Raw category mean | Smoothed value |
|---|---|---|---|
| Rare | $3$ | $1.00$ | $0.217$ |
| Common | $500$ | $0.40$ | $0.389$ |

The rare category was positive every time it appeared, so the raw encoding hands the model a 1.0 that is really three coin flips. Smoothing pulls it to 0.217, close to the global rate, while the common category barely moves. The prior weight decides how much evidence a category must show before I believe it.

### Q52. How do you handle missing values?

First I ask why they are missing, because the mechanism decides the fix. If missing at random, simple imputation is fine: median for numeric, a "missing" category for categorical, and I always add a binary indicator column so the model can use the fact of absence. If the missingness itself is informative, for example an unfilled income field on a loan form, that indicator is often the strongest feature. If a column is mostly empty, I drop it. Two rules: fit the imputer on the training fold only, otherwise it leaks, and prefer models with native handling, since XGBoost and LightGBM learn a default direction per split.

### Q53. Explain the curse of dimensionality.

As dimension $d$ grows, the volume of the space grows exponentially, so any fixed sample becomes sparse. Concretely, to keep the same density I need exponentially more points. Two consequences bite. Distances concentrate: the ratio of the nearest to the farthest neighbour distance goes to 1, so "nearest" stops carrying information, which breaks k-NN, k-means, and RBF kernels. And almost all volume sits near the boundary of the region, so most query points are extrapolations rather than interpolations. The escape is that real data usually lies on a much lower-dimensional manifold, which is what PCA, embeddings, and feature selection exploit.

**With numbers.** Take the unit hypercube and ask what fraction of its volume lies within 0.1 of the surface. The inner core is a cube of side 0.8, so the shell fraction is $1 - 0.8^d$.

| Dimension | Core volume $0.8^d$ | Shell fraction |
|---|---|---|
| $d = 2$ | $0.640$ | $36$ percent |
| $d = 10$ | $0.107$ | $89$ percent |
| $d = 100$ | $2 \times 10^{-10}$ | over $99.99$ percent |

At a hundred dimensions essentially every point is near a face. So a new query point is almost certainly outside the region my training points surround, and the model is extrapolating rather than interpolating, whatever the sample size says.

### Q54. Explain PCA in one breath.

PCA finds the orthogonal directions of maximum variance. I centre the data, then take the eigenvectors of the covariance matrix $\frac{1}{n}X^{\top}X$, or equivalently the right singular vectors of $X$, ordered by eigenvalue. Projecting onto the top $k$ gives the rank-$k$ reconstruction with the lowest squared error. I choose $k$ by explained variance ratio, often 90 or 95 percent. Two cautions: it is unsupervised, so a low-variance direction it discards may be exactly the one that separates the classes; and it needs scaled inputs, or the feature with the biggest units becomes component one by unit choice alone.

**With numbers.** Suppose the covariance eigenvalues come out as 6, 3, 0.8 and 0.2, which sum to 10.

| Components kept | Cumulative variance |
|---|---|
| 1 | $60$ percent |
| 2 | $90$ percent |
| 3 | $98$ percent |

Two components out of four hold 90 percent of the variance, so I would keep two and halve the dimension. The caution is that the discarded fourth direction holds 2 percent of the variance and may still be the one that separates the classes, because nothing in this calculation ever looked at the label.

### Q55. What do you do about class imbalance, and what does each cost?

| Remedy | Cost |
|---|---|
| Oversample the minority | Duplicates rows, so it invites overfitting on the copies |
| Undersample the majority | Throws away real data and information |
| SMOTE, synthetic interpolation | Can create points in regions where no class exists, blurring the boundary |
| Class weights in the loss | Free and usually first choice, but it distorts the output probabilities |
| Threshold tuning only | Keeps the model honest; needs a calibrated ranking to work |

I start with class weights plus threshold tuning, because they change no data. Every resampling method must be applied inside the training fold, never before the split.

### Q56. What families of feature selection are there?

Three. Filters score each feature against the target independently, using correlation, mutual information, or a chi-squared test. They are fast and model-free, but they miss interactions and keep redundant features. Wrappers search subsets by actually training the model, such as recursive feature elimination or forward selection. They respect interactions but cost many fits and can overfit the selection set. Embedded methods select during fitting: lasso zeroing coefficients, or tree-based importances. They are the best value, since selection comes free with the fit. Whichever I use, the selection must happen inside the cross-validation loop, or the score leaks.

### Q57. Are tree feature importances trustworthy?

Not on their own. The default impurity-based importance is biased toward high-cardinality and continuous features, because those give more candidate split points and therefore more chances to reduce impurity by luck. It also splits credit arbitrarily among correlated features, so a genuinely important feature can look weak because a twin absorbed the splits. Permutation importance on held-out data is better, since it measures the actual drop in score when one column is shuffled, though correlated features still mask each other. SHAP values give consistent per-prediction attributions and are what I use when someone needs to act on the ranking.

### Q58. When does feature engineering still matter?

On tabular data, which is where gradient boosting still beats deep models. Trees can only cut axis-aligned, so they cannot express a ratio or a difference compactly. Giving them price per square metre, or the gap between two dates, replaces a staircase of splits with one clean feature. Aggregations over a group and time-since-last-event features are the two that repay effort most often. On images, audio, and text it matters much less, because the network learns the representation and hand-built features usually lose. The other thing it always buys is a model I can explain to whoever has to sign off on it.

## Optimisation basics

### Q59. Explain gradient descent, and batch versus stochastic versus mini-batch.

The update is

$$w_{t+1} = w_t - \eta\,\nabla_w \mathcal{L}(w_t)$$

where $\eta$ is the learning rate. Full-batch uses every example per step: the gradient is exact and the path is smooth, but each step costs a pass over the data and it sticks in flat regions. Stochastic uses one example: very cheap and very noisy, and the noise helps escape poor regions but it never settles without a decaying rate. Mini-batch, typically 32 to 512 examples, is the practical middle. It gives a low-variance gradient estimate and uses the hardware efficiently, because a batch is one matrix multiply.

**With numbers.** The standard deviation of a batch gradient falls as $1/\sqrt{B}$.

| Batch size | Gradient standard deviation | Steps per epoch of 10000 rows |
|---|---|---|
| $32$ | $0.177$ | $313$ |
| $512$ | $0.043$ | $20$ |

Sixteen times the batch buys four times less noise, not sixteen. Meanwhile the number of updates per epoch drops by the full sixteen. That square-root return is why mini-batch beats full-batch: past a certain size I pay linearly in compute for a square-root improvement in gradient quality.

### Q60. What does the learning rate actually do?

It scales the step. Too small and training is slow and can stall in a flat region within the epoch budget. Too large and the step overshoots the curvature, so the loss oscillates or diverges to NaN. The usable ceiling is set by curvature: for a quadratic with largest eigenvalue $L$, gradient descent diverges above $\eta = 2/L$. That is why the standard recipe is a warm-up, then decay, usually cosine or step. Warm-up avoids a huge early step while the gradients are badly scaled, and decay lets the model settle instead of bouncing around the minimum. It is the hyperparameter I tune first.

**With numbers.** A one-dimensional quadratic with curvature $L = 10$, so the update multiplies the weight by $1 - \eta L$ each step. Start at $w = 1$.

| Learning rate | Factor $1 - \eta L$ | First four iterates |
|---|---|---|
| $0.05$ | $0.5$ | $0.5,\; 0.25,\; 0.125,\; 0.063$ |
| $0.15$ | $-0.5$ | $-0.5,\; 0.25,\; -0.125,\; 0.063$ |
| $0.25$ | $-1.5$ | $-1.5,\; 2.25,\; -3.375,\; 5.06$ |

The threshold is exactly $\eta = 2/L = 0.2$, where the factor is $-1$ and the iterate bounces between 1 and $-1$ forever. Just above it the weight grows by 50 percent per step and reaches NaN quickly. So the ceiling is set by curvature, not by the loss value.

### Q61. Convex versus non-convex, and why it matters.

A function is convex when the line between any two points on it lies above it, equivalently when its Hessian is positive semi-definite everywhere. For a convex problem every local minimum is global, so gradient descent with a sensible rate finds the answer and the initialisation does not matter. Linear regression, logistic regression, SVM, and lasso are all convex. Neural networks are not, so there are many minima, the result depends on the seed, and there is no certificate that I found the best one. In practice that is tolerable, because in high dimension most local minima found by SGD have similar loss.

### Q62. Why is logistic loss convex but squared error on a sigmoid is not?

Logistic loss on the logit $z = w^{\top}x$ is $\log(1 + e^{-yz})$, whose second derivative in $z$ is $\sigma(z)(1-\sigma(z)) > 0$, so it is convex in $z$; and $z$ is linear in $w$, and convexity survives composition with a linear map. Therefore it is convex in $w$. Squared error on a sigmoid, $(\sigma(z) - y)^2$, is not: the sigmoid is convex on one side of zero and concave on the other, so the composition has regions of negative curvature and multiple minima. It is also badly behaved for learning, because the gradient carries a $\sigma'(z)$ factor that vanishes when the prediction is confidently wrong.

**Walk the derivation.** Three steps, and I say them in this order.

1. **Check curvature in the logit.** The second derivative of $\log(1 + e^{-yz})$ with respect to $z$ is $\sigma(z)(1-\sigma(z))$, which is strictly positive everywhere. So the loss is convex in $z$.
2. **Push it back to the weights.** The logit $z = w^{\top}x$ is linear in $w$, and composing a convex function with a linear map preserves convexity. Therefore the loss is convex in $w$, which is what I actually optimise.
3. **Repeat for the sigmoid case.** In $(\sigma(z) - y)^2$ the sigmoid itself is convex for $z < 0$ and concave for $z > 0$. That sign change survives the squaring, so the composition has regions of negative curvature and step 2 no longer applies.

### Q63. What does momentum do?

It accumulates an exponentially weighted average of past gradients and steps along that:

$$v_t = \beta v_{t-1} + \nabla \mathcal{L}(w_t), \qquad w_{t+1} = w_t - \eta v_t$$

with $\beta$ typically 0.9. In a ravine, a valley that is steep across and shallow along, plain gradient descent zig-zags across the steep direction. The oscillating components cancel in the average while the consistent along-valley component accumulates, so momentum damps the zig-zag and accelerates the useful direction. The effective step size along a consistent gradient is about $1/(1-\beta)$ times larger, which is ten at $\beta = 0.9$. It also carries the iterate through small flat spots.

**With numbers.** Take $\beta = 0.9$ and a constant gradient of 1, so the velocity is $v_t = \beta v_{t-1} + 1$.

| Step | Velocity |
|---|---|
| $1$ | $1.00$ |
| $10$ | $6.51$ |
| $50$ | $9.95$ |
| Limit | $10.00$ |

The step size along a persistent direction reaches ten times the plain gradient step, and it takes about twenty-two steps to get most of the way there. That ramp is why momentum needs a warm-up period to be worth anything, and why the components that keep flipping sign never build up at all.

### Q64. Explain Adam in one breath.

Adam keeps two running averages per parameter: the mean of the gradient $m_t$ and the mean of its square $v_t$, with decay rates $\beta_1 = 0.9$ and $\beta_2 = 0.999$. Both are bias-corrected because they start at zero. The update is

$$w_{t+1} = w_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

So it is momentum in the numerator and a per-parameter learning rate in the denominator: parameters with historically large gradients take smaller steps. That makes it robust to badly scaled features and sparse gradients, which is why it is the default for transformers. The costs are two extra states per parameter and slightly worse generalisation than tuned SGD on vision tasks.

**With numbers.** Look at the very first step, $t = 1$, with a gradient of 0.1. Then $m_1 = 0.1 \times 0.1 = 0.01$ and $v_1 = 0.001 \times 0.01 = 10^{-5}$. Uncorrected, the ratio $m_1/\sqrt{v_1}$ is 3.16, so the first step is over three times the intended size. Bias correction divides by $1 - \beta_1 = 0.1$ and $1 - \beta_2 = 0.001$, giving $\hat{m}_1 = 0.1$ and $\hat{v}_1 = 0.01$, whose ratio is exactly 1. So the corrected first step is exactly $\eta$, which is the point of the correction.

### Q65. Local minima or saddle points — which is the real problem in high dimensions?

Saddle points. At a critical point in $d$ dimensions, the Hessian has $d$ eigenvalues, and a local minimum requires every one to be positive. Under any roughly symmetric assumption about their signs, that becomes exponentially unlikely as $d$ grows, so almost all critical points are saddles, and the bad ones are surrounded by long plateaus where the gradient is tiny. The practical answer is that this is why gradient noise helps: SGD's stochastic gradient is almost never exactly zero in the escape direction, so it drifts off the saddle, and momentum carries it across the plateau.

**With numbers.** Assume each Hessian eigenvalue is positive or negative with equal chance and independently. Then the probability that a critical point is a genuine local minimum is $2^{-d}$.

| Dimension | Probability all eigenvalues are positive |
|---|---|
| $d = 1$ | $0.5$ |
| $d = 10$ | $0.001$ |
| $d = 100$ | $8 \times 10^{-31}$ |

The assumption is crude, however the scaling is the point. In any real network $d$ is in the millions, so a critical point being a minimum is astronomically unlikely, and anything with a near-zero gradient that I hit is almost certainly a saddle.

### Q66. What activation functions do you know, and what are the benefits and weaknesses of each?

| Activation | Form | Benefit | Weakness |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | Bounded, reads as a probability | Saturates both sides, so gradients vanish; output not zero-centred |
| Tanh | $\tanh z$ | Zero-centred, so updates are better conditioned | Still saturates, still vanishes in deep stacks |
| ReLU | $\max(0,z)$ | No saturation for $z>0$, cheap, sparse | Dead units: a unit stuck negative gets zero gradient forever |
| Leaky ReLU | $\max(\alpha z, z)$ | Small negative slope keeps dead units alive | Extra hyperparameter, small gain |
| GELU | $z\,\Phi(z)$ | Smooth, non-monotone near zero, standard in transformers | More expensive than ReLU |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ | Turns logits into a distribution | Output layer only, not a hidden unit |

ReLU is the row that matters, because removing saturation on the positive side is what made deep networks trainable.

**With numbers.** The sigmoid derivative is $\sigma(1-\sigma)$, which peaks at 0.25 when $\sigma = 0.5$ and falls away on both sides. Backprop multiplies one such factor per layer, so through ten sigmoid layers the best possible gradient scaling is $0.25^{10}$, about $9.5 \times 10^{-7}$. That is the best case, at the most favourable input; a saturated unit makes it far worse. ReLU's derivative is exactly 1 on the positive side, so the same product is 1. That contrast is the whole reason deep stacks became trainable.

### Q67. Why does a network need a non-linear activation at all?

Because a stack of linear layers is one linear layer. If every layer computes $W_i h$, then the composition is $W_L \cdots W_1 x$, which is a single matrix, so depth buys nothing at all and the model can only draw a hyperplane. The non-linearity between layers is what lets the composition build regions that a single affine map cannot. With ReLU specifically, the network is piecewise linear, and the number of linear regions it can express grows exponentially in depth, which is the formal statement of why deep beats wide on the same parameter budget.

### Q68. How do you pick a batch size?

Start with the largest that fits in memory and keeps the accelerator busy, then check generalisation. Large batches give a lower-variance gradient and better throughput per step, but fewer steps per epoch and less gradient noise, and that noise is part of what regularises the model. So very large batches often generalise slightly worse unless I compensate. The usual compensation is to scale the learning rate with the batch size, linearly as a first approximation, with a warm-up so the early large steps do not blow up. If memory is the binding constraint, gradient accumulation gives the same effective batch across several passes.

## The ten they ask most

1. [Give the equations for a generative and a discriminative model and the difference between them.](#q1-give-the-equations-for-a-generative-and-a-discriminative-model-and-the-difference-between-them)
2. [Give me five things about linear regression and what each one means.](#q4-give-me-five-things-about-linear-regression-and-what-each-one-means)
3. [Explain the bias-variance tradeoff, deeper.](#q15-explain-the-bias-variance-tradeoff-deeper)
4. [Explain regularisation, deeper.](#q18-explain-regularisation-deeper)
5. [What prior does L1 regularisation correspond to, and what prior does L2 correspond to?](#q21-what-prior-does-l1-regularisation-correspond-to-and-what-prior-does-l2-correspond-to)
6. [How does gradient boosting work?](#q33-how-does-gradient-boosting-work)
7. [What activation functions do you know, and what are the benefits and weaknesses of each?](#q66-what-activation-functions-do-you-know-and-what-are-the-benefits-and-weaknesses-of-each)
8. [Explain logistic regression.](#q6-explain-logistic-regression)
9. [What is data leakage? Give two concrete examples.](#q47-what-is-data-leakage-give-two-concrete-examples)
10. [ROC-AUC versus PR-AUC.](#q43-roc-auc-versus-pr-auc)
