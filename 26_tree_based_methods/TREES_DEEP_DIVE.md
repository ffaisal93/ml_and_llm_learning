# Tree-Based Methods: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Decision trees, random forests, and gradient-boosted trees are still the dominant baseline for tabular data. Frontier-lab interviews probe split criteria, why GBDT beats RF on most tabular problems, the math behind XGBoost's regularized objective, and the deep gotchas (categorical handling, missing values, leakage). Strong candidates can derive entropy/Gini, explain why GBDT optimizes a Newton-style update, and compare XGBoost/LightGBM/CatBoost design choices.

---

## 1. The hierarchy

| Method | Core idea | Strength | Weakness |
|---|---|---|---|
| Decision tree | Recursively split feature space | Interpretable, handles non-linear | High variance, overfits |
| Random forest | Bag of decorrelated trees | Robust, parallel, low variance | Slower than boosting per quality unit |
| Gradient boosting | Sequential trees fitting residuals | State-of-the-art on tabular | Slower training, sequential |
| XGBoost / LightGBM / CatBoost | Optimized GBDT implementations | Fast, regularized, production-grade | Hyperparameter-heavy |

**Tabular ML has stayed remarkably tree-dominated.** Even with deep tabular models (TabNet, FT-Transformer), gradient boosting wins more often than not. This is one of the few areas where classical ML still beats deep learning at scale.

> **Saying it out loud.** The whole family is one idea plus two ways of fixing it. A single decision tree is easy to read and hopelessly high-variance. Bagging fixes the variance by averaging lots of decorrelated trees, which is a random forest. Boosting fixes the bias instead, by growing trees one at a time where each one corrects what the ensemble still gets wrong, and XGBoost, LightGBM, and CatBoost are engineered versions of that. The headline fact worth stating: on tabular data, boosted trees still beat deep learning most of the time, which is one of the last places where classical ML is genuinely ahead.

---

## 2. Decision trees: the foundation

A decision tree partitions the feature space recursively, choosing splits that reduce impurity.

### Splitting criteria

> **In plain language.** A split criterion is just a way of scoring how mixed up the labels are in a group. A group that's all one class scores zero; a fifty-fifty group scores worst. The tree tries every possible split, computes how much the mixing drops, and takes the best one. The formulas below are two different ways of measuring that mixing.

For classification, two main impurity measures:

**Gini impurity:**

$$
\mathrm{Gini}(S) = 1 - \sum_{c} p_c^2
$$

where $p_c$ is the fraction of class $c$ at node $S$. Equals the probability that two random samples from the node have different labels.

**Entropy:**

$$
H(S) = -\sum_c p_c \log p_c
$$

**Information gain** = entropy reduction:

$$
\text{IG}(S, \text{split}) = H(S) - \sum_{i} \frac{|S_i|}{|S|} H(S_i)
$$

For regression, splits minimize **variance reduction** (equivalent to MSE):

$$
\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar y_S)^2
$$

> **Saying it out loud.** A split criterion measures how mixed up a node's labels are, and the tree picks whichever split reduces that mixing the most. Gini is the probability that two random samples from the node have different labels; entropy is the same idea in bits. For regression there's no notion of class purity, so you use variance instead, which is identical to minimizing squared error, since each leaf predicts its own mean. The one bias to name: raw information gain favors high-cardinality features, because splitting on something like an ID makes every child perfectly pure while learning nothing.

### Gini vs entropy

In practice, Gini and entropy give almost identical trees. Gini is faster (no log). CART (sklearn default) uses Gini; ID3/C4.5 use entropy. **Don't worry about which to pick — pick Gini and move on.**

> **Saying it out loud.** Gini and entropy give effectively the same trees, so this is a non-decision. Gini is a bit faster because it avoids logarithms, and it's the CART and sklearn default, while entropy belongs to the ID3 and C4.5 lineage. They only diverge slightly on strongly imbalanced nodes. So pick Gini and spend the time you saved on depth, leaf size, and the number of trees, which actually matter.

### Greedy splitting

At each node, exhaustively evaluate every feature × every threshold. Pick the split that minimizes weighted child impurity. Greedy → not globally optimal (NP-hard in general) but works well in practice.

> **Saying it out loud.** Splitting is exhaustive locally and greedy globally: at each node you try every feature and every threshold, take the best, and never revisit it. Finding the globally optimal tree is NP-hard, so nobody does. Greedy works well in practice, but it's the reason a tree can miss an interaction that requires two individually-useless splits in sequence. It's also the source of instability, since flipping an early split reshuffles everything below it.

### Stopping criteria

- Max depth.
- Minimum samples per leaf (`min_samples_leaf`).
- Minimum samples to split (`min_samples_split`).
- Minimum impurity decrease.

These prevent overfitting. Without them, trees grow until each leaf has one sample (perfect train fit, terrible test).

> **Saying it out loud.** Stopping criteria are what stand between you and a tree with one sample per leaf. Cap the depth, require a minimum number of samples to attempt a split, require a minimum number in each resulting leaf, and require a minimum improvement in impurity. Without any of these, the tree grows until training error is exactly zero and test error is terrible. The one I'd actually set first is min_samples_leaf, because it directly prevents leaves whose prediction is based on two or three points.

### Pruning

**Pre-pruning:** stop early via the criteria above.

**Post-pruning:** grow a full tree, then collapse subtrees whose removal doesn't hurt validation performance much. Cost-complexity pruning (CART): minimize $\text{loss} + \alpha \cdot |T|$ where $|T|$ = number of leaves.

> **Saying it out loud.** Two kinds of pruning and they differ in when you make the decision. Pre-pruning stops growth using the criteria above, which is cheap but short-sighted, since a split that looks useless might enable a great one below it. Post-pruning grows the tree out fully and then collapses subtrees that didn't earn their keep, scored as training loss plus alpha times the leaf count, with alpha chosen by cross-validation. Post-pruning is better in principle; in practice most people just set max_depth because boosting brings its own regularizers anyway.

### Why decision trees overfit

A single deep tree memorizes the training data. The leaves are pure on training but represent tiny, noisy regions on test. **High variance, low bias.** Ensembles (RF, GBDT) fix this.

> **Saying it out loud.** A deep tree memorizes. Its leaves are perfectly pure on the training data, but each one covers a tiny sliver of feature space defined by a handful of noisy points, so on new data it's guessing. That's the textbook high-variance, low-bias regime. The important consequence is that instability is exactly what makes averaging effective, which is why bagging helps trees far more than it helps a linear model.

### Split scoring in code (whiteboardable in 1 minute)

```python
def gini(y):
    """Gini impurity: 1 - sum p_c^2 over class probabilities."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - (p ** 2).sum()

def entropy(y):
    """Entropy: - sum p_c log2 p_c."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p + 1e-12)).sum()

def info_gain(y, y_left, y_right, criterion=gini):
    """Reduction in impurity after splitting y into (y_left, y_right)."""
    n, n_l, n_r = len(y), len(y_left), len(y_right)
    return criterion(y) - (n_l / n) * criterion(y_left) - (n_r / n) * criterion(y_right)
```

What to say while writing: "Gini is the prob of misclassifying if we labeled randomly by class freq. Information gain is the entropy drop after split, weighted by child sizes. Trees pick the split maximizing IG — cheap, axis-aligned, greedy."

> **Saying it out loud.** If they ask you to code this, the whole thing is three tiny functions. Gini is one minus the sum of squared class proportions, entropy is minus the sum of $p$ log $p$, and information gain is the parent's impurity minus the size-weighted average of the children's. Say the size weighting out loud, because that's the part candidates drop, and without it you'd happily split off a single pure sample and call it progress. Then note that the search is over every feature and threshold, which is why the naive implementation is $O(n d)$ per node and why real libraries bin the features into histograms.

---

## 3. Random forests

Bagging applied to decision trees, with feature randomization.

### The recipe

1. Bootstrap sample the data ($N$ samples with replacement).
2. Train a deep tree on the sample.
3. **At each split**, consider only a random subset of $\sqrt{d}$ features (classification) or $d/3$ features (regression).
4. Repeat for $B$ trees.
5. Predict by averaging (regression) or majority vote (classification).

> **Saying it out loud.** The recipe is five steps. Bootstrap a sample of the rows, train a deep tree on it, restrict each split to a random subset of features, repeat for a few hundred trees, and average or vote. The two deliberate injections of randomness are the point: you're trying to make the trees disagree, because averaging models that agree buys you nothing. And note the trees are intentionally grown deep and left unpruned, since the averaging is what handles the variance.

### Why two sources of randomness?

**Bootstrap** decorrelates trees by training on different data. **Feature subsampling** further decorrelates by forcing different splits — without it, dominant features would always be picked first, producing similar trees.

The whole point: **decorrelation reduces variance**. The variance of an average of $B$ identically distributed but correlated random variables with correlation $\rho$ is:

$$
\mathrm{Var}\!\left(\frac{1}{B}\sum_b X_b\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

The second term decays as $1/B$, but the first is irreducible. Lower $\rho$ → lower asymptotic variance.

> **Saying it out loud.** This equation is the whole theory of the random forest in one line. Averaging $B$ correlated predictors gives you variance $\rho\sigma^2$ plus $(1-\rho)\sigma^2/B$: the second term vanishes as you add trees, the first one does not. So there's a floor set entirely by how correlated your trees are, which is why feature subsampling matters as much as bagging, and why adding trees past a few hundred stops helping. Without feature randomness, one dominant feature would sit at the top of every single tree and $\rho$ would be near one.

### Out-of-bag (OOB) estimate

Each bootstrap leaves out ~37% of data ($1 - (1 - 1/N)^N \to 1/e$). Average those out-of-bag predictions per sample for a free held-out estimate of generalization error. **Built-in CV without splitting.**

> **Saying it out loud.** Every bootstrap sample misses about thirty-seven percent of the rows, because the chance of never drawing a given row in $N$ draws converges to one over $e$. So for each row you can average the predictions of only the trees that never saw it, and that gives you a validation estimate for free with no held-out split. It's a genuinely nice property that boosting doesn't have. The caveat: it's a little pessimistic, because each row's prediction comes from only a third of the forest.

### Hyperparameters

- $B$ (n_estimators): more is better, plateaus eventually. 100–1000 typical.
- max_depth: deeper trees = more variance, more capacity. Typical: unlimited or 20–30.
- max_features: $\sqrt{d}$ for classification, $d/3$ for regression.
- min_samples_leaf: 1–5.

> **Saying it out loud.** Random forest hyperparameters barely matter, which is the point of the model. More trees is always at least as good, just slower, and it plateaus somewhere in the hundreds. Depth is usually left unlimited because the averaging handles variance. max_features is the one worth touching, since it's the knob that trades tree strength against tree correlation, and lowering it often helps when you have lots of noise features. Contrast that with boosting, where getting the learning rate wrong can cost you several points.

### When to use

Strong baseline for tabular tasks. Robust to hyperparameters. Parallelizes trivially. Slower than well-tuned GBDT in quality per compute unit.

> **Saying it out loud.** Use a random forest as your first model on any tabular problem, because it takes two lines, the defaults work, it trains in parallel, and it gives you a free out-of-bag error estimate. Treat that number as the bar everything else has to clear. Where it loses is quality per unit of compute against a tuned boosting model, and it can't extrapolate outside the range of the training targets. Baseline with a forest, ship with boosting.

---

## 4. Gradient boosting: the core idea

**Sequential ensemble that fits residuals.** Each new tree corrects the errors of the existing ensemble.

### The general algorithm (Friedman 2001)

For loss $L(y, \hat y)$:

1. Initialize $f_0(x) = \arg\min_c \sum_i L(y_i, c)$ (constant prediction).
2. For $m = 1, 2, \ldots, M$:
   - Compute pseudo-residuals: $r_{im} = -\partial L(y_i, f(x_i)) / \partial f(x_i)$ at $f = f_{m-1}$.
   - Fit a tree $h_m(x)$ to the residuals.
   - Find the optimal step size: $\gamma_m = \arg\min_\gamma \sum_i L(y_i, f_{m-1}(x_i) + \gamma h_m(x_i))$.
   - Update: $f_m(x) = f_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$.

$\eta$ = **learning rate** (a.k.a. shrinkage). Smaller $\eta$ + more trees = better generalization.

> **Saying it out loud.** Friedman's algorithm is four lines and worth reciting. Start with a constant prediction. Then repeatedly: compute the negative gradient of the loss at every training point, fit a tree to those pseudo-residuals, find the step size, and add the tree scaled by the learning rate. The learning rate is doing real work, because taking many small steps generalizes much better than a few greedy ones. The practical relationship to name is that learning rate and number of trees trade off inversely, so halving one means roughly doubling the other.

### Why "gradient" boosting?

The pseudo-residual is the negative gradient of the loss w.r.t. the current prediction. Fitting a tree to it is **functional gradient descent** in the space of functions.

For squared-error loss $L = \frac{1}{2}(y - \hat y)^2$:

$$
r_i = -\frac{\partial L}{\partial \hat y} = y - \hat y
$$

so we literally fit the residual.

For other losses (logistic, Huber, etc.), the pseudo-residuals are different but the framework is the same.

> **Saying it out loud.** It's called gradient boosting because it's gradient descent performed in function space rather than parameter space. The thing you're optimizing is a function, the steepest-descent direction is the negative gradient of the loss evaluated at your training points, and since you can't represent that exactly, you fit a tree to approximate it and step that way. For squared error the gradient is literally the residual, which is why the intuition is fit the leftovers. Swap in logistic loss and the residual becomes the label minus the predicted probability, and nothing else about the algorithm changes.

### Why GBDT often beats RF

- **Lower bias**: trees correct each other's errors. RF's trees are independently-trained averages.
- **Better signal extraction**: each tree adds incremental refinement.
- **More tunable**: $\eta$ + tree depth + regularization give fine control.

The trade-off: GBDT is sequential (can't parallelize trees). RF is embarrassingly parallel. So RF is preferred when training time dominates.

> **Saying it out loud.** Boosting usually beats bagging because it attacks bias while bagging only attacks variance. Each new tree is trained specifically on what the ensemble is still getting wrong, so progress is directed rather than an average of independent guesses, and boosting gives you far more knobs to tune. The tradeoff is training time and fragility: boosting is inherently sequential across trees and it will overfit if you don't early-stop, while a random forest is embarrassingly parallel and nearly impossible to misuse.

---

## 5. XGBoost: regularized GBDT

Chen & Guestrin 2016. The standard production GBDT for tabular data. Three innovations make it dominant:

### 1. Regularized objective

XGBoost optimizes:

$$
\mathcal{L}(\phi) = \sum_i \ell(y_i, \hat y_i) + \sum_k \Omega(f_k)
$$

where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ penalizes the number of leaves $T$ and the magnitude of leaf weights $w$. Standard GBDT has no explicit regularization; XGBoost adds it.

> **Saying it out loud.** XGBoost's real conceptual contribution is putting regularization inside the objective. Classic gradient boosting had a loss and some heuristics like depth caps; XGBoost adds gamma times the number of leaves plus L2 on the leaf values directly into what's being minimized. Because the penalty is in the objective, it flows into the split-scoring formula, so a split simply doesn't happen unless its gain exceeds gamma. Pruning becomes a consequence of the math instead of a separate post-processing pass.

### 2. Newton-style updates (second-order info)

At each iteration, approximate the loss via second-order Taylor expansion around $\hat y_i^{(t-1)}$:

$$
\mathcal{L}^{(t)} \approx \sum_i \!\left[\ell(y_i, \hat y_i^{(t-1)}) + g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2\right] + \Omega(f_t)
$$

where $g_i = \partial_{\hat y} \ell$, $h_i = \partial^2_{\hat y} \ell$. Closed-form optimal leaf weight given a tree structure:

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

Optimal objective for a given tree:

$$
\mathcal{L}^{(t)}(q) = -\frac{1}{2} \sum_j \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

This is the **gain** used to score candidate splits — it accounts for both first and second derivatives of the loss, like Newton's method.

> **Saying it out loud.** The second-order trick is to Taylor-expand the loss to two terms around the current prediction, so each iteration uses both the gradient and the curvature at every point, making it a Newton step rather than a plain gradient step. The payoff is that both the optimal leaf value and the score of a candidate split come out in closed form: leaf weight is minus the gradient sum over the Hessian sum plus lambda, and gain is built from those same quantities. Notice that lambda sits in the denominator, so leaves with few points get shrunk hard toward zero automatically. That's better than Friedman's first-order version because it knows the difference between a big gradient in a confident region and one in an uncertain region.

### 3. System-level optimizations

- **Approximate split finding** with histogram-based binning (avoids sorting per feature per split).
- **Cache-aware access patterns**.
- **Parallel column subsampling** within each tree.
- **Sparsity-aware split finding** for missing values.

These engineering wins make XGBoost orders of magnitude faster than naive GBDT.

> **Saying it out loud.** A lot of XGBoost's dominance is engineering rather than math. Histogram binning replaces sorting every feature at every node with bucketing into a couple hundred bins, which turns split finding from $O(n \log n)$ per feature into a linear scan. Cache-aware memory layouts, parallel column subsampling, and sparsity-aware handling of missing values do the rest. These are the differences between a research implementation and something that trains on ten million rows in minutes. It's worth naming the histogram trick specifically, because that's the one that gave the order-of-magnitude speedup.

### Hyperparameters that matter

- `n_estimators`, `learning_rate` ($\eta$): the classic tension. Start with small $\eta$ = 0.1, n_estimators = 1000, and tune.
- `max_depth`: 4–8 typical. Deeper = more bias-variance complexity.
- `min_child_weight`: minimum sum of $h_i$ in a leaf. Prevents overfitting.
- `subsample`, `colsample_bytree`: stochastic GBDT (à la Stochastic GB), also reduces variance.
- `lambda`, `gamma`: L2 + leaf count regularization.

> **Saying it out loud.** The hyperparameters that matter, in order: learning rate paired with the number of trees, which you handle by fixing a smallish rate and letting early stopping choose the count. Then depth and min_child_weight, which control per-tree complexity and are where most of the remaining gain lives. Then subsample and colsample, which add stochastic regularization for free. Then lambda and gamma, which usually move things least. If you only do one thing, use early stopping, because it tunes the single most important parameter for you.

---

## 6. LightGBM and CatBoost: the variations

### LightGBM (Microsoft, 2017)

- **Leaf-wise tree growth**: split the leaf with highest gain (vs level-wise growth in XGBoost). Faster convergence; deeper trees on critical regions; needs `num_leaves` cap to prevent overfitting.
- **GOSS (Gradient-based One-Side Sampling)**: keep all high-gradient samples + sample low-gradient ones. Faster training with minimal accuracy loss.
- **EFB (Exclusive Feature Bundling)**: bundle mutually-exclusive sparse features into one. Memory and time savings on sparse high-dim data.
- Faster than XGBoost at similar quality on most tabular datasets.


> **Saying it out loud.** LightGBM's three ideas are all about doing less work per unit of accuracy. Leaf-wise growth splits whichever leaf in the entire tree offers the biggest gain, so it reaches a given loss with far fewer nodes than XGBoost's level-wise growth, at the cost of deep unbalanced trees that need num_leaves to rein them in. GOSS keeps every large-gradient example and subsamples the small-gradient ones with a correction, so you train on a fraction of the data without biasing the estimate. And EFB bundles mutually exclusive sparse columns together, which shrinks the effective feature count on one-hot-heavy data. Net effect is several times faster training at the same accuracy.
### CatBoost (Yandex, 2017)

- **Native categorical handling**: target statistics with permutation-based bias correction. Avoids leakage that naive target encoding causes.
- **Symmetric (oblivious) trees**: same split at every node of a level. Faster inference; mild quality cost.
- **Ordered boosting**: avoids target leakage in residual computation by using a permutation order.
- Particularly good with high-cardinality categoricals.

> **Saying it out loud.** CatBoost's three ideas are all about categorical data and leakage. Ordered target statistics encode a category using only examples earlier in a random permutation, so a row's own label can never leak into its own feature. Oblivious trees use the same split at every node of a level, which makes inference nearly branch-free and very fast, at a small cost in per-tree expressiveness. And ordered boosting applies the same permutation logic to residual computation, fixing a subtle bias that all standard boosting quietly has. It's the natural pick when your data is full of high-cardinality categoricals.

### Choosing among them

- **XGBoost**: most mature, best ecosystem, default choice.
- **LightGBM**: fastest training, often equals or slightly beats XGBoost.
- **CatBoost**: best for high-cardinality categoricals; minimal preprocessing.

For most tabular tasks: try LightGBM first; if quality is the bottleneck, try XGBoost; if categorical features dominate, try CatBoost.

> **Saying it out loud.** The practical selection rule is short. LightGBM first, because it's the fastest and usually matches XGBoost on quality. XGBoost when you want the most mature ecosystem or you're squeezing out the last bit of accuracy. CatBoost when high-cardinality categorical features dominate and you'd otherwise be hand-building an out-of-fold target encoding. Honestly, with a decent tuning budget all three land within noise of each other on most datasets, so the differentiator is training speed and how much preprocessing you avoid.

---

## 7. Categorical feature handling

A real interview probe.

### One-hot encoding
Standard for low-cardinality categoricals. For tree models, this is sometimes wrong:
- **Splits become inefficient.** A binary feature can only be split once usefully.
- **High cardinality blows up.** "Zip code" becomes 40K columns.

> **Saying it out loud.** One-hot encoding is the safe default at low cardinality and a real problem for trees at high cardinality. The reason is structural: a binary column can only be usefully split once, so the signal from a category gets scattered across many weak features that each carry a sliver of it, and the greedy split search never finds them worth choosing. Then zip code turns into forty thousand columns and memory becomes the issue too. Under about ten categories, one-hot is fine and simple; above that, use native categorical handling or out-of-fold target encoding.

### Target encoding (mean encoding)
Replace category with the mean target value:
$$
\text{enc}(c) = \frac{\sum_{i : \text{cat}_i = c} y_i}{|\{i : \text{cat}_i = c\}|}
$$

**Leakage risk:** if you target-encode using the full dataset, you leak labels into features. Cross-fold target encoding (compute from out-of-fold data) is the proper version.

> **Saying it out loud.** Target encoding replaces a category with the average target for that category, which is compact and powerful and quietly dangerous. The danger is that a row's own label contributes to its own feature value, so for a rare category with three examples the feature essentially is the answer, and the model learns to read it. Training accuracy looks amazing and production falls apart. The fix is to compute the encoding out of fold, plus smoothing rare categories toward the global mean, and if you take one rule from this section it's that target encoding computed on the full dataset is always a bug.

### CatBoost's ordered target statistics
Permute the data, encode each example using only earlier examples in the permutation. Avoids leakage by construction.

> **Saying it out loud.** CatBoost's ordered target statistics is leak-free target encoding done at the row level rather than the fold level. You take a random permutation and encode each row using only the rows that came before it, so a row's label can never influence its own feature. It's the same principle as out-of-fold encoding but finer-grained, and CatBoost averages over several permutations to reduce the variance that introduces. That's the single feature that makes CatBoost worth reaching for on categorical-heavy data.

### LightGBM's native handling
Splits categorical features by partitioning categories into two groups. Tractable due to a sorting trick.

> **Saying it out loud.** LightGBM can split a categorical feature by partitioning its categories into two groups instead of testing them one at a time. Searching all partitions is exponential, but there's a classic result that sorting categories by their mean gradient and scanning for the best cut point along that order gives the optimal partition for the usual criterion. So it's linear after a sort. The caution is that this is powerful enough to overfit high-cardinality columns, which is what min_data_per_group and cat_smooth exist to control.

---

## 8. Missing value handling

### XGBoost / LightGBM
At each split, learn a default direction (left or right) for missing values. Missing values are routed there. This handles missingness automatically — **no imputation needed**.

> **Saying it out loud.** XGBoost and LightGBM learn a default direction for missing values at every split. During training they try routing all the missing values left, then all right, score both, and store the winner as part of the node. So missingness becomes a learned routing decision rather than something you have to impute away, and the model can exploit the fact that a field being blank is itself informative. That's a real advantage over linear models and neural nets, which need you to fill in a value and thereby destroy that signal.

### CatBoost
Treats missing as a separate category.

> **Saying it out loud.** CatBoost handles it by treating missing as just another category, which fits naturally with its categorical machinery. It's a slightly different philosophy from learning a per-split direction: instead of routing decisions, missingness gets its own identity in the encoding. Both approaches beat imputation. The thing to know is that neither one requires you to make up a number, which is the source of most preprocessing bugs.

### Why this matters
Most tabular real-world data has missing values. Tree-based methods handle this well; logistic regression and NN often need explicit imputation. **A common reason GBDT beats NN on tabular: fewer preprocessing pitfalls.**

> **Saying it out loud.** This matters more than it sounds because real tabular data is full of holes, and how you fill them is a modeling decision most people make accidentally. Impute with the column mean and you've asserted that a missing value is an average value, which is usually false and sometimes badly false, since data is often missing for a reason correlated with the target. Trees sidestep the whole question. It's one of the concrete, unglamorous reasons boosted trees beat neural nets on tabular data: fewer preprocessing decisions to get wrong.

---

## 9. Why tree-based methods dominate tabular

### Tabular data has different structure than images/text

- Heterogeneous features (numeric + categorical).
- Non-smooth dependencies (sharp thresholds).
- Sparse, high-dimensional.
- Few samples per feature interaction.

Trees handle all of this naturally:
- Splits handle heterogeneity (each split type-aware).
- Splits are non-smooth by definition.
- Sparsity-aware algorithms (XGBoost) handle high-dim sparse data.
- Trees can capture interactions explicitly (deep trees) or via boosting depth.

> **Saying it out loud.** Tabular data has properties that happen to match what trees do and mismatch what neural nets assume. Features are heterogeneous, different units, different scales, some categorical, and trees are scale-invariant so they don't care at all. The true functions are frequently non-smooth, with hard thresholds like an age cutoff, which a split represents exactly and an MLP can only approximate with a lot of capacity. Trees also ignore irrelevant features rather than being distracted by them. That combination, rather than any single factor, is why they hold up.

### Where deep tabular models win

- Very large datasets (1M+ rows) with many feature interactions.
- Tasks needing transfer learning across tabular tasks.
- When pre-trained tabular models exist (rare but growing — TabPFN, TabTransformer).

> **Saying it out loud.** Deep tabular models win in the corners. Very large datasets, past a million rows or so, where a network has enough data to learn interactions that a tree would need enormous depth for. Anything requiring transfer across tasks, where you want a pretrained representation and trees offer nothing. And situations where tabular columns sit alongside text or images, so you want one model handling all modalities. TabPFN is the interesting recent exception in the opposite direction, doing in-context learning on tiny datasets, but it's capped at around a thousand rows.

### Empirical reality (as of 2024-2025)

GBDT (LightGBM/XGBoost/CatBoost) wins ~70% of tabular benchmarks. Deep tabular models win the rest. The gap is closing slowly.

> **Saying it out loud.** The empirical state of play as of the last couple of years is that boosted trees win roughly seventy percent of tabular benchmarks, and the gap closes slowly rather than abruptly. Grinsztajn and colleagues in 2022 is the paper to cite, and their key finding was that the advantage comes from inductive bias, not from data volume: trees handle non-smooth targets and irrelevant features better. The honest framing in an interview is that this is an empirical regularity with a plausible explanation, not a law, and it's been narrowing.

---

## 10. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why not just use a single deep tree?" | High variance — overfits training data. Ensembles (RF, GBDT) reduce variance. |
| "RF vs GBDT — which is better?" | GBDT usually wins on quality; RF wins on training speed and parallelism. For production tabular, GBDT. |
| "Why does GBDT use the gradient?" | Functional gradient descent: each tree fits the negative gradient of the loss. Generalizes from MSE residuals to any differentiable loss. |
| "What's XGBoost's second-order trick?" | Newton-style: use both first ($g$) and second ($h$) derivative of the loss; closed-form leaf weight. Better than Friedman's first-order GB. |
| "How do trees handle missing values?" | XGBoost/LightGBM: learn a default direction per split. No imputation needed. |
| "Why is target encoding risky?" | Leaks labels into features unless done out-of-fold. CatBoost's ordered TS fixes it. |
| "Why is GBDT sequential?" | Each tree depends on the residuals of the previous ensemble. Cannot parallelize across trees (only within tree-building). |
| "Tree-based vs NN on tabular?" | GBDT wins ~70% of tabular benchmarks. NN wins on very large data with feature interactions, or with pretrained tabular models. |
| "Why limit tree depth in GBDT?" | Shallow trees (depth 4-8) capture low-order interactions; ensemble adds capacity via depth in $M$ (number of trees). |
| "What's the role of $\eta$ (learning rate)?" | Shrinks each tree's contribution. Smaller $\eta$ + more trees = better generalization (analogous to small LR in SGD). |

> **Saying it out loud.** The compressed version of the gotchas: a single deep tree is high-variance, which is why you ensemble. Boosting usually wins on quality, bagging on training speed. The gradient framing is what lets you boost against any differentiable loss, not just squared error. XGBoost's second-order trick is a Newton step that yields closed-form leaf weights. Trees handle missing values natively by learning a default direction. And target encoding leaks unless it's out of fold, which is probably the most common real-world bug in this entire area.

---

## 11. The 10 most-asked tree interview questions

1. **Walk me through Gini and entropy.** Both impurity measures; pick split with max impurity reduction. Gini = $1 - \sum p_c^2$; entropy = $-\sum p_c \log p_c$. Almost identical in practice.
2. **How does gradient boosting work?** Sequential trees fit pseudo-residuals (negative gradients of loss). Each tree corrects the previous ensemble.
3. **XGBoost vs random forest?** XGBoost = boosting (sequential, lower bias). RF = bagging (parallel, lower variance). XGBoost usually wins on quality.
4. **What's XGBoost's regularized objective?** $\sum \ell + \sum \Omega(f_k)$ where $\Omega = \gamma T + \tfrac{1}{2}\lambda \|w\|^2$. Penalizes leaf count and weight magnitude.
5. **What's the second-order trick?** Newton-style update using both gradient and Hessian. Closed-form leaf weight $-\sum g / (\sum h + \lambda)$.
6. **LightGBM vs XGBoost?** LightGBM: leaf-wise growth (faster), GOSS sampling, EFB feature bundling. Often faster at equal quality.
7. **CatBoost — what's special?** Native categorical handling via ordered target statistics. Symmetric trees. Best on high-cardinality categoricals.
8. **How do trees handle missing values?** Learn default direction per split. No imputation needed.
9. **Why does target encoding leak?** Using full-dataset target stats to encode features leaks labels. Use out-of-fold encoding.
10. **Why GBDT on tabular but not NN?** Tabular has heterogeneous features, non-smooth dependencies, sparsity, few samples. Trees handle all naturally.

---

## 12. Drill plan

1. Master Gini/entropy and information gain.
2. Walk through gradient boosting end-to-end (residuals → tree → step size).
3. Whiteboard XGBoost's second-order optimal leaf weight.
4. Compare RF/XGBoost/LightGBM/CatBoost.
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

---

## 13. Further reading

- Breiman, "Random Forests" (2001).
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001).
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016).
- Ke et al., "LightGBM" (2017).
- Prokhorenkova et al., "CatBoost" (2018).
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Chapters 9–10, 15.
