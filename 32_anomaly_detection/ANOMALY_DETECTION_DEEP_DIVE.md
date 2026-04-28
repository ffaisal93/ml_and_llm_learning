# Anomaly Detection — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Anomaly detection comes up in interviews because every production system needs it (fraud, system monitoring, quality control, security). What separates good answers is knowing *which* method matches *which* problem structure — not just listing names.

---

## 1. The fundamental setup

You have data $\mathcal{D} = \{x_1, \ldots, x_n\}$ from "normal" distribution $p_{\mathrm{normal}}$. New point $x^*$: is it from $p_{\mathrm{normal}}$ or something else?

### Three problem variants

**Unsupervised**: only normal data. Most realistic. Methods estimate $p_{\mathrm{normal}}$ or its support; flag low-likelihood / outside-support points.

**Semi-supervised**: training data is mostly normal but contains some anomalies (and you don't know which). Or you have a small set of labeled anomalies.

**Supervised**: labeled normal vs anomaly examples. Just classification with imbalanced classes.

In practice, most production systems are unsupervised (anomalies are rare and hand-labeling is expensive).

---

## 2. Statistical methods

### Z-score / Modified Z-score

For each feature: $z_i = (x_i - \mu)/\sigma$. Flag $|z_i| > 3$ as outliers (3-sigma rule).

**When**: roughly Gaussian per-feature, low-dim.

**Modified z-score** uses median absolute deviation (MAD): more robust to outliers themselves.

$$
M_i = 0.6745 (x_i - \mathrm{median}) / \mathrm{MAD}
$$

Threshold $|M| > 3.5$.

### IQR rule

Flag $x_i < Q_1 - 1.5 \mathrm{IQR}$ or $> Q_3 + 1.5 \mathrm{IQR}$. Box-plot rule.

**When**: skewed univariate data; quick exploratory.

### Mahalanobis distance

For multivariate Gaussian:

$$
D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

Accounts for correlations between features. Flag high $D_M$.

**When**: roughly Gaussian multivariate; small dimensionality.

### Limitations
- Assume specific distributional form (often Gaussian).
- Don't handle non-linear structure.
- Per-feature z-scores miss multivariate anomalies.

---

## 3. Density-based methods

### KDE (Kernel Density Estimation)

Estimate $p_{\mathrm{normal}}(x)$ via Gaussian kernels around each training point. Flag $x^*$ with low $\hat{p}(x^*)$.

**When**: low-dim, moderate $n$. Curse of dimensionality kills it for $d > 10$.

### LOF (Local Outlier Factor)

For each $x$, compare its local density to densities of its $k$-nearest neighbors. Anomaly: lower density than neighbors.

$$
\mathrm{LOF}(x) = \frac{\frac{1}{k}\sum_{x' \in N_k(x)} \mathrm{lrd}(x')}{\mathrm{lrd}(x)}
$$

where $\mathrm{lrd}$ is local reachability density. LOF > 1 typically anomalous.

**When**: data has varying density across regions (clusters of different tightness).

### DBSCAN-as-anomaly-detector

Points labeled noise (not in any cluster) are anomalies. Side-effect of clustering.

---

## 4. Distance-based: nearest neighbor

Flag points whose distance to $k$-th nearest neighbor is large. Simple, parameter-light (just $k$).

Variants:
- Distance to $k$-th NN.
- Average distance to $k$ NN.
- Sum of distances to $k$ NN.

**When**: anomalies are "far from everyone." Fails when normal data has natural varying density.

---

## 5. Isolation Forest

The 2008 standard for tabular anomaly detection. Liu, Ting, Zhou.

### Idea
Anomalies are "few and different" — they should be easy to isolate by random splits.

Build random tree by:
1. Pick a random feature.
2. Pick a random split value in [min, max] of that feature.
3. Recurse until leaves contain single points.

Anomalies isolate quickly (short path from root). Normals require many splits to reach a single-point leaf (long path).

### Anomaly score

$$
s(x, n) = 2^{-E(h(x))/c(n)}
$$

with $E(h(x))$ = expected path length, $c(n)$ = average path length normalization. Score near 1 = anomaly; near 0.5 = normal.

### Why it works
- No distance metric needed → robust in high dim.
- Sub-linear training (random subsamples).
- Easy to parallelize.
- No distributional assumption.

### Why it shipped
Strong baseline; cheap; sklearn implementation; minimal hyperparameters. Default for production tabular anomaly detection.

---

## 6. One-Class SVM

Learn a decision boundary around normal data; flag points outside.

### Formulation

$$
\min_{w, \rho, \xi} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_i \xi_i
$$

subject to $w^\top \phi(x_i) \geq \rho - \xi_i$, $\xi_i \geq 0$.

In feature space (via kernel $\phi$), find a hyperplane that separates the data from origin with maximal margin. Points below the hyperplane → anomaly.

$\nu$: upper bound on training error fraction; lower bound on support vector fraction.

### When
- Roughly globular normal data.
- RBF kernel for non-linear boundaries.
- Small to moderate $n$ (kernel methods don't scale).

### Variant: SVDD (Support Vector Data Description)
Find smallest sphere enclosing normal data. Same idea, different formulation.

---

## 7. Reconstruction-based: autoencoders

Train autoencoder on normal data; minimize reconstruction error. At test: high reconstruction error → anomaly.

$$
\mathrm{anomaly}(x) = \|x - g(f(x))\|^2
$$

### Why it works
- AE learns to reconstruct normal patterns.
- Anomalies don't fit the learned manifold → poor reconstruction.

### Variants
- **Vanilla AE**: standard.
- **Denoising AE**: corrupt input, reconstruct clean. Stronger generalization.
- **VAE**: probabilistic; can use likelihood as anomaly score.
- **Convolutional AE**: for images.
- **Sequence AE**: for time series.

### Strengths
- Scales to high-dim (images, sequences).
- Captures non-linear structure.
- Same recipe across modalities.

### Weakness
- Can over-reconstruct anomalies if model is too powerful (regularize!).
- Threshold tuning is empirical.

---

## 8. Density-ratio and PU learning

### Density ratio

Estimate $r(x) = p_{\mathrm{anomaly}}(x) / p_{\mathrm{normal}}(x)$ directly via classifier. Train binary classifier to distinguish "normal" data from "all data" or "anomaly" if some labels available.

### PU (Positive-Unlabeled) learning

Treat normal as "positive"; unlabeled data may contain anomalies. Specialized methods (e.g., nnPU) train classifier under this asymmetry.

### LLM / foundation-model anomaly detection
Modern approach: use pretrained embedding model (CLIP for images, sentence encoders for text). Compute distance from "normal" centroid in embedding space. Surprisingly effective.

---

## 9. Time-series anomalies

Time series anomalies have temporal structure that simple methods miss.

### Types
- **Point anomaly**: single value out of pattern.
- **Contextual anomaly**: value normal globally but anomalous given recent context (e.g., 0°C in summer).
- **Collective anomaly**: pattern of values jointly anomalous (e.g., flat line where there should be variation).

### Methods
- **STL decomposition + outlier on residuals**: decompose into trend + seasonal + residual; flag outlier residuals.
- **ARIMA / SARIMAX prediction**: flag values far from model prediction.
- **LSTM / transformer prediction**: flag high prediction error.
- **Spectral residuals**: signal processing approach (used by Twitter / Microsoft).
- **Matrix profile**: cross-correlation of subsequences. Anomaly = low similarity to all other subsequences.

---

## 10. Evaluation challenges

The hardest part. Anomalies are rare → labels expensive → evaluation noisy.

### Metrics
- **Precision @ k**: fraction of top-$k$ flagged that are real.
- **AUPRC**: precision-recall curve area. Standard for imbalanced.
- **Recall @ false-alarm rate**: how many anomalies caught at fixed false positive budget.
- **F1**: only meaningful at fixed threshold.

### Why AUC is misleading
With 1% anomalies, even a poor classifier has high AUC (mostly negatives). AUPRC much more informative.

### Threshold tuning
- Cost-aware: cost of false positive vs false negative often very asymmetric.
- Operating point: tune to acceptable false alarm rate.
- Adaptive: thresholds may need to change as data drifts.

### Realistic evaluation
- Hand-labeled subset (small, expensive, gold).
- Synthetic anomalies (inject known patterns).
- Production validation: monitor flagged-but-correct rate.

---

## 11. Choosing a method

| Setting | Method |
|---|---|
| Low-dim Gaussian-ish | Mahalanobis |
| Univariate with skew | IQR or modified z-score |
| Tabular, varied scales | Isolation Forest (default) |
| Local density variation | LOF |
| High-dim images | Convolutional AE / VAE |
| Sequential / time-series | LSTM / matrix profile |
| Have some labels | Treat as imbalanced classification |
| Modern foundation models | Embedding distance from normal centroid |

The "boring" choices (Isolation Forest, embedding distance) usually win at start. More sophisticated methods help when you understand exactly which assumption matters.

---

## 12. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Z-score for high-dim? | Yes | Per-feature z-scores miss multivariate anomalies; use Mahalanobis or non-statistical methods |
| Isolation Forest is what kind of method? | Density | Tree-based; isolates anomalies via random partitioning, not density |
| Why does AE-based AD work? | Magic | AE learns to reconstruct normal; anomalies don't fit learned manifold |
| AUC is fine for AD? | Yes | Misleading for severe imbalance; use AUPRC |
| OC-SVM for high-dim? | Yes | Doesn't scale well; kernel methods are $O(n^2)$ memory |
| What's the cold-start in AD? | Doesn't apply | New data type / new region of feature space; model has no normal examples |
| Precision = 0.99, recall = 0.05 — good? | Maybe | Catching 5% of anomalies is bad; tune to higher recall, accept more false positives |

---

## 13. Eight most-asked interview questions

1. **You suspect anomalies in your production logs. Walk through your approach.** (Frame: unsupervised vs labeled; method choice; evaluation; threshold tuning.)
2. **Compare Isolation Forest, OC-SVM, autoencoder.** (Tree-based vs kernel-based vs reconstruction-based; trade-offs; when each.)
3. **Why is AUC misleading for anomaly detection?** (Severe class imbalance; AUPRC better.)
4. **How does Isolation Forest work?** (Random splits; anomalies isolate quickly → short path; score from path length.)
5. **Time-series anomaly detection — what's special?** (Temporal context; point vs contextual vs collective; decomposition methods.)
6. **You have only normal data — what methods?** (Unsupervised: density estimation, IF, OC-SVM, AE; not labeled classification.)
7. **What's the cost asymmetry in fraud / health anomaly?** (False negative usually much more expensive than false positive.)
8. **How do you evaluate without much labeled data?** (Hand-labeled subset; synthetic injection; production validation rate.)

---

## 14. Drill plan

- For each method, recite: assumption, when to use, evaluation strength, common failure.
- Walk through Isolation Forest score derivation in 3 minutes.
- Recite 3 time-series anomaly types with examples.
- For "why is AUC misleading?" — recite full reasoning + AUPRC alternative.
- Practice 2 case studies: log anomaly, fraud detection — design end-to-end answer.

---

## 15. Further reading

- Liu, Ting, Zhou (2008), *Isolation Forest.*
- Schölkopf et al. (2001), *Estimating the Support of a High-Dimensional Distribution* (One-Class SVM).
- Breunig, Kriegel, Ng, Sander (2000), *LOF: Identifying Density-Based Local Outliers.*
- Chandola, Banerjee, Kumar (2009), *Anomaly Detection: A Survey* — comprehensive.
- Sakurada & Yairi (2014), *Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction.*
- Pang, Shen, Cao, Hengel (2021), *Deep Learning for Anomaly Detection: A Review.*
