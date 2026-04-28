# Tree-Based Methods — Interview Grill

> 50 questions on trees, RF, GBDT. Drill until you can answer 35+ cold.

---

## A. Decision trees

**1. How does a decision tree split?**
Greedily: at each node, evaluate all candidate features × thresholds, pick the split that maximizes impurity reduction (information gain or variance reduction). Recurse on each child.

**2. Define Gini impurity.**
$\operatorname{Gini}(S) = 1 - \sum_c p_c^2$ where $p_c$ is the fraction of class $c$ at node $S$. Equals the probability that two random samples from $S$ have different labels.

**3. Define entropy.**
$H(S) = -\sum_c p_c \log p_c$. Information-theoretic uncertainty of the label distribution.

**4. Information gain?**
$\text{IG}(S, \text{split}) = H(S) - \sum_i (|S_i|/|S|) H(S_i)$. Reduction in entropy from splitting.

**5. Gini vs entropy in practice?**
Almost identical results. Gini is faster (no log). CART (sklearn default) uses Gini; ID3/C4.5 use entropy. Pick Gini.

**6. Regression tree splits?**
Minimize variance: $\text{Var}(S) = (1/|S|) \sum_{i \in S} (y_i - \bar y_S)^2$. Equivalent to MSE. Split picks threshold that minimizes weighted child variance.

**7. Why do trees overfit?**
Without depth limits, a tree grows until each leaf has one sample → perfect train fit, terrible test. Single deep tree has high variance.

**8. Standard regularizations for trees?**
`max_depth` (cap depth), `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, post-pruning (cost-complexity).

**9. What's cost-complexity pruning?**
Grow a deep tree, then prune: minimize $\text{loss}(T) + \alpha \cdot |T|$ where $|T|$ is leaf count. Tunes $\alpha$ via CV.

**10. Are decision trees stable?**
No. Small data perturbations can produce very different trees. High variance. This is exactly why ensembles (RF, GBDT) help.

---

## B. Random forests

**11. What is a random forest?**
Bagging applied to decision trees, plus feature randomization at each split. Each tree trained on bootstrap sample; each split considers $\sqrt{d}$ random features. Average predictions (or vote).

**12. Two sources of randomness in RF?**
Bootstrap sampling (different data per tree) and feature subsampling (different splits available). Both decorrelate trees.

**13. Why decorrelate trees?**
Variance of an average of correlated RVs: $\rho \sigma^2 + (1-\rho)\sigma^2/B$. The first term is irreducible — lower correlation $\rho$ → lower asymptotic variance.

**14. Typical max_features in RF?**
$\sqrt{d}$ for classification, $d/3$ for regression. Common defaults; tunable.

**15. What's out-of-bag (OOB) estimate?**
Each bootstrap sample leaves out ~37% of data ($1 - (1 - 1/N)^N \to 1/e$). Average predictions on those held-out samples gives a free CV-like estimate.

**16. Does RF overfit?**
Less than single trees — bagging reduces variance. But still possible if individual trees are too deep. With enough trees and deep individual trees, RF can memorize noise.

**17. Pros of RF?**
Robust to hyperparameters, parallel training, OOB estimate built-in, handles mixed features. Strong baseline.

**18. Cons of RF?**
Slower than well-tuned GBDT at same quality. Bigger model size. Worse extrapolation than linear models.

---

## C. Gradient boosting

**19. What's the core idea of gradient boosting?**
Sequential ensemble. Each new tree fits the **residuals** (negative gradient of the loss) of the current ensemble.

**20. For squared error, what are the residuals?**
$r_i = -\partial L/\partial \hat y_i = y_i - \hat y_i$. Just the standard residual. So GBDT with MSE literally fits residuals.

**21. For other losses?**
Pseudo-residuals: $r_i = -\partial L / \partial \hat y_i$ at current prediction. For logistic loss: $r_i = y_i - \sigma(\hat y_i)$. Different per loss but framework is general.

**22. Why "gradient" boosting?**
Functional gradient descent in the space of functions. Each tree is a step in the negative gradient direction of the loss.

**23. What's the role of $\eta$ (learning rate / shrinkage)?**
Scale each tree's contribution: $f_m = f_{m-1} + \eta \gamma_m h_m$. Smaller $\eta$ + more trees = better generalization. Typical: 0.01–0.1.

**24. Why does GBDT often beat RF?**
Lower bias (each tree corrects errors), better signal extraction, more tunable. RF only reduces variance through bagging.

**25. Why is GBDT sequential?**
Each tree depends on the residuals of the previous ensemble. Cannot parallelize across trees. Within-tree splitting is parallelizable.

**26. Stochastic gradient boosting?**
Subsample data per tree (typically 0.5–0.8). Reduces variance, regularizes, slightly faster. Friedman 2002 extension.

---

## D. XGBoost specifics

**27. What's XGBoost's regularized objective?**

$$
\mathcal{L}(\phi) = \sum_i \ell(y_i, \hat y_i) + \sum_k \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \|w\|^2
$$

Penalizes leaf count $T$ and leaf weight magnitude. Standard GBDT has no explicit regularization.

**28. What's XGBoost's second-order trick?**
Taylor-expand the loss around current prediction. Use both gradient $g_i = \partial \ell / \partial \hat y$ and Hessian $h_i = \partial^2 \ell / \partial \hat y^2$. Newton-style update.

**29. Optimal leaf weight in XGBoost?**

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

Closed-form. Includes the L2 regularization $\lambda$.

**30. Optimal objective for a given tree structure?**

$$
\mathcal{L}^* = -\tfrac{1}{2} \sum_j \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

This is the **gain** used to score candidate splits. Includes second-order curvature info, which beats Friedman's first-order GB.

**31. What does `min_child_weight` control in XGBoost?**
Minimum sum of $h_i$ (Hessians) in a leaf. For squared error, $h_i = 1$, so it's min samples per leaf. For logistic, $h_i = p(1-p)$ — *largest* at $p = 0.5$ (uncertain points) and small for confident predictions. So the threshold effectively requires enough *uncertain* points per leaf, preventing splits driven by a few high-curvature outliers.

**32. Default $\eta$ in XGBoost?**
0.3 historically; modern recommendation 0.05–0.1 with more trees. Smaller LR + more trees = better generalization.

**33. How does XGBoost handle missing values?**
Sparsity-aware split finding: at each split, learn a default direction (left or right) for missing values. No imputation needed.

---

## E. LightGBM and CatBoost

**34. LightGBM's main innovations?**
(a) Leaf-wise tree growth (split highest-gain leaf vs level-wise). (b) GOSS — gradient-based one-side sampling. (c) EFB — exclusive feature bundling for sparse high-dim data. Faster than XGBoost at equal quality.

**35. Why is leaf-wise growth faster?**
Trees converge with fewer total nodes. But: deeper trees on critical regions; can overfit. `num_leaves` parameter caps it.

**36. CatBoost's main innovations?**
(a) Native categorical handling via ordered target statistics. (b) Symmetric (oblivious) trees. (c) Ordered boosting to avoid target leakage in residual computation.

**37. What's "ordered target statistics"?**
Permute the data; encode each example using only earlier examples in the permutation. Avoids leakage that naive target encoding causes.

**38. Symmetric trees — what and why?**
Each level uses the same split. Makes inference much faster (one matmul-like op per tree). Mild quality cost. CatBoost's choice.

**39. When does CatBoost win?**
High-cardinality categorical features. Where target encoding leakage would hurt naive XGBoost/LightGBM workflows.

---

## F. Categorical features and missing values

**40. One-hot vs target encoding for trees?**
One-hot: standard for low cardinality. Wasteful for trees (each binary feature splittable once). Target encoding: replace category with mean target. Risk: leakage if not done out-of-fold.

**41. Why is target encoding leaky?**
Using full-dataset target stats incorporates labels into features. Out-of-fold target encoding (compute from data outside the current row's fold) fixes it.

**42. How does LightGBM handle categorical features?**
Splits by partitioning categories into two groups. Tractable via a sorting trick (sort categories by mean gradient).

**43. How do tree models handle missing values?**
XGBoost/LightGBM: learn default direction per split. CatBoost: missing as a category. Trees handle this natively — a real advantage over NN/LR which need imputation.

---

## G. Hyperparameter tuning

**44. Most important XGBoost hyperparameters?**
`learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `lambda`, `gamma`. Tune with grid/random search or Bayesian optimization.

**45. Practical tuning order?**
(1) `n_estimators` and `learning_rate` together (early stopping). (2) Tree complexity (`max_depth`, `min_child_weight`). (3) Stochasticity (`subsample`, `colsample`). (4) Regularization (`lambda`, `gamma`).

**46. What's early stopping?**
Stop adding trees when validation metric stops improving. Standard in XGBoost/LightGBM. Avoids overfitting and saves compute.

**47. Default tree depth for GBDT?**
4–8 is typical. Deeper = more variance, more capacity per tree. Shallow trees with many iterations (with $\eta$ small) usually generalize best.

---

## H. Subtleties and gotchas

**48. Why don't trees extrapolate?**
Trees predict by averaging training labels in each leaf. For new inputs outside training feature ranges, prediction is bounded by training observations. Linear models extrapolate naturally; trees don't.

**49. Trees vs NN on tabular data — why trees still win?**
Tabular has heterogeneous features, non-smooth dependencies, sparsity, few samples per interaction. Trees handle all naturally. NN often need extensive feature engineering and regularization to compete.

**50. When would you NOT use trees?**
Sequential/temporal data with rich structure (use RNNs/transformers). Image/audio (use CNNs/transformers). Very large data with feature interactions where deep tabular pretraining helps. When inference latency must be sub-millisecond and the model must be tiny (linear models).

---

## Quick fire

**51.** *XGBoost paper?* Chen & Guestrin 2016.
**52.** *LightGBM paper?* Ke et al. 2017.
**53.** *CatBoost paper?* Prokhorenkova et al. 2018.
**54.** *Default RF max_features for classification?* $\sqrt{d}$.
**55.** *Default learning rate for GBDT?* 0.05–0.1.

---

## Self-grading

If you can't answer 1-15, you don't know trees. If you can't answer 16-35, you'll struggle on tabular ML interviews. If you can't answer 36-50, frontier-lab interviews on tabular methods will go past you.

Aim for 35+/50 cold.
