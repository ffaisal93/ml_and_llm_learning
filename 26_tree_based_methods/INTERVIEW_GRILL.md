# Tree-Based Methods — Interview Grill

> 50 questions on trees, RF, GBDT. Drill until you can answer 35+ cold.

---

## A. Decision trees

**1. How does a decision tree split?**
Greedily: at each node, evaluate all candidate features × thresholds, pick the split that maximizes impurity reduction (information gain or variance reduction). Recurse on each child.

> **Saying it out loud.** A tree splits greedily and locally. At each node it tries every feature and every candidate threshold, scores each split by how much it purifies the two children, takes the best one, and then recurses without ever reconsidering. It's greedy, so it never looks ahead and it isn't the globally optimal tree, which is fine because finding that is NP-hard. The practical consequence is that trees are unstable: a slightly different sample can flip an early split and change the entire structure below it.

**2. Define Gini impurity.**
$\mathrm{Gini}(S) = 1 - \sum_c p_c^2$ where $p_c$ is the fraction of class $c$ at node $S$. Equals the probability that two random samples from $S$ have different labels.

> **Saying it out loud.** Gini impurity is the chance that if you grab two random samples from a node and compare their labels, they disagree. All one class gives you zero; a fifty-fifty split of two classes gives you a half. So it's a direct measure of how mixed a node is. It's the sklearn default because it's cheap, no logarithms involved, and it gives essentially the same trees as entropy.

**3. Define entropy.**
$H(S) = -\sum_c p_c \log p_c$. Information-theoretic uncertainty of the label distribution.

> **Saying it out loud.** Entropy is the information-theoretic version of the same question: how many bits do you need on average to encode the label of a sample from this node. Pure node, zero bits, zero entropy. Fifty-fifty binary split, exactly one bit. It's the impurity measure behind ID3 and C4.5, and its maximum for $k$ classes is $\log k$.

**4. Information gain?**
$\text{IG}(S, \text{split}) = H(S) - \sum_i (|S_i|/|S|) H(S_i)$. Reduction in entropy from splitting.

> **Saying it out loud.** Information gain is the entropy before the split minus the weighted average entropy of the children, so it's the uncertainty you removed by asking that question. The weighting by child size matters, because peeling off three samples into a pure leaf is not the same achievement as cleanly separating half the data. There's a known bias here: raw information gain favors high-cardinality features, since splitting on an ID column makes every child pure, which is exactly why C4.5 uses gain ratio to normalize it away.

**5. Gini vs entropy in practice?**
Almost identical results. Gini is faster (no log). CART (sklearn default) uses Gini; ID3/C4.5 use entropy. Pick Gini.

> **Saying it out loud.** In practice they give nearly identical trees, and there's no real accuracy argument for either. Gini is somewhat faster because it avoids logarithms, and it's the CART and sklearn default; entropy comes from the ID3 and C4.5 lineage. They only diverge on strongly imbalanced nodes, where entropy is a bit more sensitive. So use Gini and spend your tuning budget on depth and the number of trees instead.

**6. Regression tree splits?**
Minimize variance: $\text{Var}(S) = (1/|S|) \sum_{i \in S} (y_i - \bar y_S)^2$. Equivalent to MSE. Split picks threshold that minimizes weighted child variance.

> **Saying it out loud.** For regression you replace impurity with variance: pick the threshold that minimizes the size-weighted variance of the two children, which is exactly minimizing squared error since each leaf predicts its own mean. Reducing variance and reducing MSE are the same objective here. The consequence worth naming is that a squared-error criterion is highly sensitive to outliers, which is why you'd switch to an absolute-error or Huber criterion when your target has heavy tails.

**7. Why do trees overfit?**
Without depth limits, a tree grows until each leaf has one sample → perfect train fit, terrible test. Single deep tree has high variance.

> **Saying it out loud.** Because nothing stops them. Let a tree grow and it keeps splitting until every leaf holds a single training point, at which point it fits the training data perfectly and has memorized the noise. It's a fully non-parametric model that adds capacity for free as it goes. That's the extreme high-variance end of the bias-variance tradeoff, and the entire ensembling literature exists to bring that variance back down.

**8. Standard regularizations for trees?**
`max_depth` (cap depth), `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, post-pruning (cost-complexity).

> **Saying it out loud.** Two kinds: limits during growth and pruning after. During growth you cap max_depth, require a minimum number of samples to split a node or to sit in a leaf, and require a minimum impurity decrease per split. After growth you can prune back with cost-complexity pruning. In practice max_depth plus min_samples_leaf gets you most of the way, and min_samples_leaf is the one that actually protects you from leaves built on three points.

**9. What's cost-complexity pruning?**
Grow a deep tree, then prune: minimize $\text{loss}(T) + \alpha \cdot |T|$ where $|T|$ is leaf count. Tunes $\alpha$ via CV.

> **Saying it out loud.** Cost-complexity pruning grows the tree out fully, then trims it back by minimizing the training loss plus alpha times the number of leaves. As you turn alpha up you get a nested sequence of progressively smaller trees, and you cross-validate to pick where to stop. It's better than depth limits in principle because it prunes based on what actually earned its keep, rather than cutting every branch at the same level. In practice most people just use max_depth because it's one parameter and boosting has other regularizers anyway.

**10. Are decision trees stable?**
No. Small data perturbations can produce very different trees. High variance. This is exactly why ensembles (RF, GBDT) help.

> **Saying it out loud.** Not at all, and this is the defining weakness. Change a handful of training points and the top split can flip to a different feature, which reshuffles everything below it and gives you a visually completely different tree. That's high variance in the technical sense. It also undercuts the interpretability claim, since a structure that changes when you resample isn't a stable explanation. And it's exactly why bagging works so well here: averaging helps most when the base learner is unstable.

---

## B. Random forests

**11. What is a random forest?**
Bagging applied to decision trees, plus feature randomization at each split. Each tree trained on bootstrap sample; each split considers $\sqrt{d}$ random features. Average predictions (or vote).

> **Saying it out loud.** A random forest is bagging plus feature randomness. You train many deep trees, each on a bootstrap resample of the data, and at every split each tree only gets to consider a random subset of features, typically the square root of the total. Then you average the predictions or vote. The individual trees are deliberately overfit and high-variance, and the averaging is what pulls the variance down.

**12. Two sources of randomness in RF?**
Bootstrap sampling (different data per tree) and feature subsampling (different splits available). Both decorrelate trees.

> **Saying it out loud.** Two independent sources. Bootstrap resampling means each tree sees a different sample of rows, and feature subsampling means each split only gets to consider a random handful of columns. The second one is what makes it a random forest rather than plain bagged trees. It matters because without it, a single dominant feature would sit at the top of every tree and they'd all be nearly identical, and averaging near-identical models buys you nothing.

**13. Why decorrelate trees?**
Variance of an average of correlated RVs: $\rho \sigma^2 + (1-\rho)\sigma^2/B$. The first term is irreducible — lower correlation $\rho$ → lower asymptotic variance.

> **Saying it out loud.** The variance of an average of $B$ correlated variables is $\rho\sigma^2$ plus $(1-\rho)\sigma^2/B$. Look at what happens as $B$ goes to infinity: the second term vanishes but the first doesn't. So no matter how many trees you add, you're floored at $\rho\sigma^2$, and the only way to lower that floor is to make the trees less correlated. That single equation is the entire justification for feature subsampling, and it also tells you why adding trees past a few hundred stops helping.

**14. Typical max_features in RF?**
$\sqrt{d}$ for classification, $d/3$ for regression. Common defaults; tunable.

> **Saying it out loud.** Square root of the number of features for classification, and about a third of them for regression. Those are just empirically good defaults and they're worth tuning. The direction of the knob is what matters: fewer features per split means more decorrelated, higher-bias trees, more means stronger but more similar trees. If your data has a lot of noise features, lowering it usually helps, because it gives the informative ones a chance to be considered.

**15. What's out-of-bag (OOB) estimate?**
Each bootstrap sample leaves out ~37% of data ($1 - (1 - 1/N)^N \to 1/e$). Average predictions on those held-out samples gives a free CV-like estimate.

> **Saying it out loud.** Every bootstrap sample leaves out about thirty-seven percent of the data, because the probability of never picking a given row in $N$ draws goes to one over $e$. So for each row you can average the predictions of just those trees that never saw it, and you get a validation estimate for free without holding anything out. That's a genuine convenience of random forests that boosting doesn't offer. The caveat is that OOB is slightly pessimistic, since each prediction uses only about a third of the forest.

**16. Does RF overfit?**
Less than single trees — bagging reduces variance. But still possible if individual trees are too deep. With enough trees and deep individual trees, RF can memorize noise.

> **Saying it out loud.** Less than a single tree, but it's a myth that it can't. Bagging reduces variance and doesn't touch bias, so with very deep trees and enough of them, the forest can absolutely fit noise, especially on small datasets. What is true is that adding more trees doesn't cause overfitting, since it's just averaging, so unlike boosting, you never need early stopping on tree count. The knob that controls overfitting is per-tree depth and leaf size.

**17. Pros of RF?**
Robust to hyperparameters, parallel training, OOB estimate built-in, handles mixed features. Strong baseline.

> **Saying it out loud.** Random forests are forgiving. Defaults usually work, hyperparameters barely matter compared to boosting, training is embarrassingly parallel across trees, and you get a free validation estimate from out-of-bag samples. They handle mixed numeric and categorical features and don't need scaling. That's why the right move on a new tabular dataset is to fit a random forest first, in a couple of lines, and treat it as the number everything else has to beat.

**18. Cons of RF?**
Slower than well-tuned GBDT at same quality. Bigger model size. Worse extrapolation than linear models.

> **Saying it out loud.** A well-tuned boosted model usually beats it, because boosting reduces bias while bagging only reduces variance. Random forests also need many deep trees, so the model is large and inference is slower than a shallow boosted ensemble. And like all trees, they can't extrapolate beyond the range of the training targets. So the honest positioning: random forest is the great baseline, gradient boosting is the model you ship.

---

## C. Gradient boosting

**19. What's the core idea of gradient boosting?**
Sequential ensemble. Each new tree fits the **residuals** (negative gradient of the loss) of the current ensemble.

> **Saying it out loud.** Gradient boosting builds an ensemble one tree at a time, where each new tree is trained to fix what the current ensemble is getting wrong. Concretely, you compute the negative gradient of the loss at every training point, which for squared error is just the residual, and you fit the next tree to that. Then you add it, scaled down by a learning rate, and repeat. It's sequential and it reduces bias, which is exactly the complement to bagging.

**20. For squared error, what are the residuals?**
$r_i = -\partial L/\partial \hat y_i = y_i - \hat y_i$. Just the standard residual. So GBDT with MSE literally fits residuals.

> **Saying it out loud.** For squared error the negative gradient is literally $y_i$ minus the current prediction, the plain residual. So gradient boosting with MSE is just repeatedly fitting a tree to what's left over, which is where the name residual fitting comes from. It's the case worth deriving out loud because it makes the general framework obvious: for other losses, the residual gets replaced by whatever the negative gradient happens to be.

**21. For other losses?**
Pseudo-residuals: $r_i = -\partial L / \partial \hat y_i$ at current prediction. For logistic loss: $r_i = y_i - \sigma(\hat y_i)$. Different per loss but framework is general.

> **Saying it out loud.** For a general loss you fit each tree to the pseudo-residuals, which are the negative gradients of the loss with respect to the current predictions. For logistic loss that comes out as the label minus the predicted probability, which is beautifully similar to the squared-error case. That's the generality of the framework: swap the loss, recompute one derivative, everything else stays the same. It's why you can boost against ranking objectives, quantile losses, or a custom business metric as long as it's differentiable.

**22. Why "gradient" boosting?**
Functional gradient descent in the space of functions. Each tree is a step in the negative gradient direction of the loss.

> **Saying it out loud.** Because it's gradient descent performed in function space rather than parameter space. Your model is a function, the loss is a functional of it, and the direction of steepest descent is the negative gradient evaluated at your training points. A tree can't be that gradient exactly, so you fit a tree to approximate it and take a small step in that direction. The learning rate is exactly the step size, which is why boosting's hyperparameters feel so much like an optimizer's.

**23. What's the role of $\eta$ (learning rate / shrinkage)?**
Scale each tree's contribution: $f_m = f_{m-1} + \eta \gamma_m h_m$. Smaller $\eta$ + more trees = better generalization. Typical: 0.01–0.1.

> **Saying it out loud.** Eta is the step size: each tree's contribution gets multiplied by it before being added. Smaller eta means each tree matters less, which requires more trees but consistently generalizes better, because you're approaching the solution in many small careful steps rather than a few big greedy ones. Typical values are point-zero-one to point-one. And the key relationship is that eta and the number of trees trade off inversely, so halving the learning rate means roughly doubling the trees.

**24. Why does GBDT often beat RF?**
Lower bias (each tree corrects errors), better signal extraction, more tunable. RF only reduces variance through bagging.

> **Saying it out loud.** Because boosting attacks bias and bagging only attacks variance. Each boosted tree is explicitly trained on what the ensemble is still getting wrong, so the ensemble keeps improving in a directed way, while a random forest just averages many independent guesses. Boosting also has far more knobs, so it rewards tuning. The flip side is the cost: boosting is sensitive to hyperparameters and can overfit if you don't early-stop, whereas a random forest is nearly foolproof.

**25. Why is GBDT sequential?**
Each tree depends on the residuals of the previous ensemble. Cannot parallelize across trees. Within-tree splitting is parallelizable.

> **Saying it out loud.** Because tree $m$ is fit to the residuals left by trees one through $m$ minus one, so you can't build it until the previous one exists. That's an inherent data dependency, not an implementation limitation. What does parallelize is the work inside a single tree, finding the best split across features and bins, which is what XGBoost and LightGBM exploit heavily. So boosting scales on a single machine with many cores, it just doesn't scale by throwing trees at separate workers the way a random forest does.

**26. Stochastic gradient boosting?**
Subsample data per tree (typically 0.5–0.8). Reduces variance, regularizes, slightly faster. Friedman 2002 extension.

> **Saying it out loud.** Stochastic gradient boosting fits each tree on a random subsample of the rows, typically half to eighty percent, rather than all of them. That injects the same kind of decorrelation bagging uses, so it regularizes and reduces variance, and as a side effect each iteration is faster. It's Friedman's 2002 addition to his own algorithm and it's a genuinely free win. Combine it with column subsampling per tree and you have XGBoost's two main stochastic regularizers.

---

## D. XGBoost specifics

**27. What's XGBoost's regularized objective?**

$$
\mathcal{L}(\phi) = \sum_i \ell(y_i, \hat y_i) + \sum_k \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \|w\|^2
$$

Penalizes leaf count $T$ and leaf weight magnitude. Standard GBDT has no explicit regularization.

> **Saying it out loud.** XGBoost's objective adds an explicit penalty to the standard boosting loss: gamma times the number of leaves plus L2 on the leaf weights. That's the real conceptual departure from classic gradient boosting, which had no regularization term in its objective and relied entirely on heuristics like depth limits. Here the penalty is inside the split-scoring math, so a split only happens if its gain exceeds gamma. That's why XGBoost prunes as a natural consequence of optimization rather than as a post-processing step.

**28. What's XGBoost's second-order trick?**
Taylor-expand the loss around current prediction. Use both gradient $g_i = \partial \ell / \partial \hat y$ and Hessian $h_i = \partial^2 \ell / \partial \hat y^2$. Newton-style update.

> **Saying it out loud.** XGBoost Taylor-expands the loss to second order around the current prediction, so it uses both the gradient and the Hessian at each data point instead of just the gradient. That makes each step a Newton step rather than a plain gradient step, which converges faster and uses curvature to size the update. The concrete payoff is that leaf values and split gains both get closed-form expressions. And it's why min_child_weight in XGBoost is a sum of Hessians rather than a count of rows.

**29. Optimal leaf weight in XGBoost?**

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

Closed-form. Includes the L2 regularization $\lambda$.

> **Saying it out loud.** The optimal leaf value is minus the sum of the gradients in that leaf, divided by the sum of the Hessians plus lambda. It's a Newton step: gradient over curvature, with the regularizer sitting in the denominator so it shrinks the leaf toward zero. Notice what lambda does: leaves with few points have small Hessian sums, so lambda dominates and their weights get pulled hard toward zero. That's automatic shrinkage of exactly the leaves you should distrust.

**30. Optimal objective for a given tree structure?**

$$
\mathcal{L}^* = -\tfrac{1}{2} \sum_j \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

This is the **gain** used to score candidate splits. Includes second-order curvature info, which beats Friedman's first-order GB.

> **Saying it out loud.** Plug those optimal leaf weights back in and you get a closed-form score for any tree structure: a sum over leaves of the squared gradient sum over the Hessian sum plus lambda, minus gamma per leaf. That number is the gain, and it's what every candidate split is scored by. It beats Friedman's original first-order approach because it accounts for curvature, so it knows the difference between a big gradient on a confident region and a big gradient on an uncertain one. And gamma appearing in the gain is what lets a split be rejected outright for not paying for itself.

**31. What does `min_child_weight` control in XGBoost?**
Minimum sum of $h_i$ (Hessians) in a leaf. For squared error, $h_i = 1$, so it's min samples per leaf. For logistic, $h_i = p(1-p)$ — *largest* at $p = 0.5$ (uncertain points) and small for confident predictions. So the threshold effectively requires enough *uncertain* points per leaf, preventing splits driven by a few high-curvature outliers.

> **Saying it out loud.** min_child_weight is a floor on the sum of Hessians in a leaf, not a count of rows. For squared error the Hessian is one per point, so it reduces to minimum samples per leaf. For logistic loss the Hessian is $p(1-p)$, which peaks at a half and is nearly zero for confident predictions, so the constraint effectively demands that a leaf contain enough genuinely uncertain points. That's a smarter regularizer than a raw count, because it stops you from carving off leaves justified by a handful of confidently-classified outliers.

**32. Default $\eta$ in XGBoost?**
0.3 historically; modern recommendation 0.05–0.1 with more trees. Smaller LR + more trees = better generalization.

> **Saying it out loud.** The historical default is point-three, which is far too high for good results; the modern practice is point-zero-five to point-one with many more trees and early stopping. The tradeoff is pure compute versus quality: smaller learning rate needs proportionally more trees and more training time, and it reliably generalizes better. A sensible workflow is to tune everything else at point-one for speed and then do a final run at point-zero-three with early stopping.

**33. How does XGBoost handle missing values?**
Sparsity-aware split finding: at each split, learn a default direction (left or right) for missing values. No imputation needed.

> **Saying it out loud.** XGBoost learns a default direction per split. During training it tries sending all the missing values left and all of them right, scores both, and stores whichever won as part of the node. So missingness itself becomes a learned signal rather than something you have to impute away. That's a real advantage over linear models and neural nets, and it matters most when the data is missing not at random, where the fact that a field is blank genuinely carries information.

---

## E. LightGBM and CatBoost

**34. LightGBM's main innovations?**
(a) Leaf-wise tree growth (split highest-gain leaf vs level-wise). (b) GOSS — gradient-based one-side sampling. (c) EFB — exclusive feature bundling for sparse high-dim data. Faster than XGBoost at equal quality.

> **Saying it out loud.** Three things. Leaf-wise growth, splitting whichever leaf in the whole tree offers the biggest gain instead of expanding level by level. GOSS, which keeps all the large-gradient examples and randomly subsamples the small-gradient ones with a reweighting, so you train on less data without biasing the estimate. And EFB, which bundles mutually exclusive sparse features into single columns, shrinking the effective feature count on one-hot-heavy data. Net result is several times faster than XGBoost at the same accuracy, which is why it's the default choice on large datasets.

**35. Why is leaf-wise growth faster?**
Trees converge with fewer total nodes. But: deeper trees on critical regions; can overfit. `num_leaves` parameter caps it.

> **Saying it out loud.** Because level-wise growth spends splits on parts of the tree that don't need them, while leaf-wise always spends the next split where the gain is largest, so it reaches a given loss with far fewer nodes. The cost is that it grows deep, unbalanced trees that can chase noise, especially on small datasets. That's what num_leaves is for, and you should also set max_depth on small data. The rule of thumb: keep num_leaves below two to the max_depth or you're not really constraining anything.

**36. CatBoost's main innovations?**
(a) Native categorical handling via ordered target statistics. (b) Symmetric (oblivious) trees. (c) Ordered boosting to avoid target leakage in residual computation.

> **Saying it out loud.** Three. Native categorical handling with ordered target statistics, so you don't have to one-hot or hand-roll target encoding. Symmetric or oblivious trees, where every node at a level uses the same split. And ordered boosting, which computes each example's residual using a model trained only on examples that came earlier in a random permutation, avoiding a subtle target leakage that all standard boosting has. That third one is the most intellectually interesting and the least well known.

**37. What's "ordered target statistics"?**
Permute the data; encode each example using only earlier examples in the permutation. Avoids leakage that naive target encoding causes.

> **Saying it out loud.** Ordered target statistics is leak-free target encoding. You take a random permutation of the data, and to encode a category for a given row you only use the rows before it in that permutation. That way a row's own label never contributes to its own feature value, which is exactly the leakage that makes naive target encoding look great in training and fall apart in production. CatBoost averages over several permutations to reduce the variance this introduces. It's the same idea as out-of-fold encoding, done more finely.

**38. Symmetric trees — what and why?**
Each level uses the same split. Makes inference much faster (one matmul-like op per tree). Mild quality cost. CatBoost's choice.

> **Saying it out loud.** In a symmetric or oblivious tree, every node at the same depth uses the identical feature and threshold, so the whole tree is really just a sequence of yes-no questions and the leaf index is a binary number you assemble. That makes inference extremely fast and branch-free, essentially an index computation instead of a traversal. The cost is expressiveness, since each tree is weaker, and you compensate with more trees. It's a deliberate trade of a little accuracy for a lot of prediction speed, which is why CatBoost is attractive for low-latency serving.

**39. When does CatBoost win?**
High-cardinality categorical features. Where target encoding leakage would hurt naive XGBoost/LightGBM workflows.

> **Saying it out loud.** CatBoost wins when you have high-cardinality categorical features, things like zip code, product ID, or user ID, where one-hot explodes and naive target encoding leaks. Its ordered statistics handle those correctly by construction, so you get a strong model without building a careful out-of-fold encoding pipeline yourself. On purely numeric data it's usually a wash against LightGBM and often slower to train. So pick it for the data type, not as a general default.

---

## F. Categorical features and missing values

**40. One-hot vs target encoding for trees?**
One-hot: standard for low cardinality. Wasteful for trees (each binary feature splittable once). Target encoding: replace category with mean target. Risk: leakage if not done out-of-fold.

> **Saying it out loud.** One-hot is fine for low cardinality, say under about ten categories, but it's genuinely bad for trees at high cardinality, because each binary column can only ever be split once and the signal gets scattered across many weak features. Target encoding replaces the category with the mean target for that category, which is compact and powerful, and dangerous, because done naively it leaks the label into the feature. If you use it, compute it out of fold, always. Or use LightGBM's or CatBoost's native categorical handling and skip the problem.

**41. Why is target encoding leaky?**
Using full-dataset target stats incorporates labels into features. Out-of-fold target encoding (compute from data outside the current row's fold) fixes it.

> **Saying it out loud.** Because a row's own label goes into the statistic used as that row's feature. On a rare category with three examples, the encoded value essentially is the label, so the tree learns to read the answer off the feature and training accuracy looks fantastic. Then in production the category is new or the mean differs and the whole thing collapses. The fix is to compute the encoding from data outside the current row's fold, plus smoothing toward the global mean for rare categories.

**42. How does LightGBM handle categorical features?**
Splits by partitioning categories into two groups. Tractable via a sorting trick (sort categories by mean gradient).

> **Saying it out loud.** LightGBM can split a categorical feature by partitioning its categories into two groups, rather than testing one category at a time. Finding the best partition is exponential in general, but there's a classic result that sorting the categories by their mean gradient and then scanning for the best split point along that order gives the optimum for the standard criterion. So it's linear after a sort. The practical caution is overfitting on high-cardinality columns, which is what min_data_per_group and cat_smooth exist to control.

**43. How do tree models handle missing values?**
XGBoost/LightGBM: learn default direction per split. CatBoost: missing as a category. Trees handle this natively — a real advantage over NN/LR which need imputation.

> **Saying it out loud.** Trees handle missing values natively, which is one of their underrated advantages. XGBoost and LightGBM learn a default direction per split, trying both and keeping the better one, so missingness becomes a learned routing decision. CatBoost can treat it as its own category. No imputation anywhere in the pipeline. That matters most when data is missing not at random, since imputing a mean would destroy the information that the field was blank in the first place.

---

## G. Hyperparameter tuning

**44. Most important XGBoost hyperparameters?**
`learning_rate`, `n_estimators`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `lambda`, `gamma`. Tune with grid/random search or Bayesian optimization.

> **Saying it out loud.** The ones that matter most are learning_rate and n_estimators together, then tree complexity through max_depth and min_child_weight, then the stochastic ones, subsample and colsample_bytree, then the explicit regularizers lambda and gamma. Beyond those, returns fall off quickly. The single highest-leverage move isn't tuning at all, it's using early stopping on a validation set so n_estimators tunes itself.

**45. Practical tuning order?**
(1) `n_estimators` and `learning_rate` together (early stopping). (2) Tree complexity (`max_depth`, `min_child_weight`). (3) Stochasticity (`subsample`, `colsample`). (4) Regularization (`lambda`, `gamma`).

> **Saying it out loud.** Fix a moderate learning rate and let early stopping determine the number of trees, so those two are handled first and together. Then tune tree complexity, depth and min_child_weight, which is where most of the remaining gain is. Then the subsampling rates. Then the explicit L2 and gamma penalties, which usually move things least. Finally, drop the learning rate and retrain for the final model. Working in that order means you're always tuning the parameter with the biggest effect first.

**46. What's early stopping?**
Stop adding trees when validation metric stops improving. Standard in XGBoost/LightGBM. Avoids overfitting and saves compute.

> **Saying it out loud.** Early stopping means you monitor a validation metric during training and stop adding trees once it hasn't improved for some number of rounds, then keep the best iteration. It's how the number of trees gets chosen in practice, and it saves you both overfitting and compute. Fifty rounds of patience is a common setting. The trap is using your test set as the early-stopping set, which quietly leaks and makes your reported number optimistic.

**47. Default tree depth for GBDT?**
4–8 is typical. Deeper = more variance, more capacity per tree. Shallow trees with many iterations (with $\eta$ small) usually generalize best.

> **Saying it out loud.** Four to eight for gradient boosting, which is much shallower than the trees in a random forest. The reason is that boosting only needs each tree to be a weak learner correcting a small error; depth is what controls the order of feature interactions each tree can capture, so depth six means up to six-way interactions. Combining shallow trees with a small learning rate and many iterations reliably generalizes best. If you find yourself needing depth twelve, you probably have a feature engineering problem instead.

---

## H. Subtleties and gotchas

**48. Why don't trees extrapolate?**
Trees predict by averaging training labels in each leaf. For new inputs outside training feature ranges, prediction is bounded by training observations. Linear models extrapolate naturally; trees don't.

> **Saying it out loud.** Because a tree's prediction is always an average of training labels sitting in some leaf, so it's mathematically incapable of producing a value outside the range it saw during training. Feed it a house twice the size of anything in the data and it returns whatever the largest-house leaf returns. A linear model would keep extrapolating the trend. That's a hard constraint to remember for time series with trend, where a boosted model will simply flatline at the last observed level.

**49. Trees vs NN on tabular data — why trees still win?**
Tabular has heterogeneous features, non-smooth dependencies, sparsity, few samples per interaction. Trees handle all naturally. NN often need extensive feature engineering and regularization to compete.

> **Saying it out loud.** Because tabular data has exactly the properties trees are built for and neural nets are not. Features are heterogeneous, some categorical, some skewed, on wildly different scales, and trees don't care about scale at all. The true functions are often piecewise and non-smooth, with hard thresholds, which trees represent natively while MLPs have an inductive bias toward smoothness. Trees also ignore uninformative features rather than being distracted by them. The benchmarks bear it out, with Grinsztajn and colleagues in 2022 showing boosted trees still ahead on medium-sized tabular data, and the gap only closes at very large sample sizes.

**50. When would you NOT use trees?**
Sequential/temporal data with rich structure (use RNNs/transformers). Image/audio (use CNNs/transformers). Very large data with feature interactions where deep tabular pretraining helps. When inference latency must be sub-millisecond and the model must be tiny (linear models).

> **Saying it out loud.** Don't use trees where the structure of the input is the point: sequences, text, images, audio, all of which have spatial or temporal structure that trees have to laboriously rediscover through hand-built features. Don't use them when you need smooth extrapolation outside the training range. And don't use them when you need a sub-millisecond, few-kilobyte model, where a logistic regression wins on operational grounds. The dividing line to name is representation learning: if the features have to be learned from raw input, use a network; if you already have meaningful columns, use trees.

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
