# Clustering: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Clustering is the canonical unsupervised learning task. Interviewers probe: K-means assumptions and failure modes, why DBSCAN handles non-convex shapes, how GMM relates to K-means, the EM algorithm, evaluation challenges. Strong candidates can derive K-means as coordinate descent on a quadratic objective and explain when each method fits.

---

## 1. The clustering taxonomy

| Method | Approach | Strength | Weakness |
|---|---|---|---|
| **K-means** | Partition into $K$ spherical clusters minimizing intra-cluster variance | Fast, scalable, simple | Requires $K$ known; assumes spherical clusters |
| **Gaussian Mixture (GMM)** | Soft K-means with covariance per cluster | Captures elliptical clusters; soft assignments | Requires $K$ known; sensitive to init |
| **Hierarchical** | Agglomerative tree of merges | No need to specify $K$ a priori; produces dendrogram | $O(N^2)$–$O(N^3)$ memory/compute |
| **DBSCAN** | Density-based: clusters = dense regions | Handles arbitrary shapes; finds outliers | Sensitive to $\varepsilon$; struggles with varying density |
| **Spectral** | Cluster via eigendecomposition of similarity graph | Handles non-convex shapes; theoretically principled | $O(N^3)$ eigendecomposition |
| **HDBSCAN** | Hierarchical density-based | DBSCAN without $\varepsilon$ tuning; varying density | Complex implementation |

There's no universally best clustering method. The right choice depends on cluster shape, density, scale, and whether $K$ is known.

---

## 2. K-means

The workhorse. Most-asked clustering algorithm in interviews.

### The algorithm

Given $K$ clusters and data $\{x_i\}$:

1. Initialize centroids $\mu_1, \ldots, \mu_K$ (e.g., k-means++).
2. **Assignment step**: each point joins the nearest centroid: $c_i = \arg\min_k \|x_i - \mu_k\|^2$.
3. **Update step**: each centroid moves to the mean of its assigned points: $\mu_k = (1/|C_k|) \sum_{i \in C_k} x_i$.
4. Repeat 2–3 until assignments don't change.

### The objective

K-means minimizes within-cluster sum of squares (WCSS):

$$
\mathcal{L}(\mu, c) = \sum_i \|x_i - \mu_{c_i}\|^2
$$

### Why it converges

Both steps decrease the objective:
- Assignment step: reassigning to nearest centroid can only decrease per-point distances (or keep equal).
- Update step: setting centroid to the mean is the closed-form optimum given the assignments.

Since the objective is bounded below and decreases monotonically, K-means converges (to a local minimum, not necessarily global).

### K-means as coordinate descent

K-means is **coordinate descent** on the WCSS objective: alternately optimize over $c$ (assignments) holding $\mu$ fixed, and over $\mu$ holding $c$ fixed. Both subproblems have closed-form solutions. This is why it converges and why it's so fast.

### Initialization: k-means++

Random initialization can produce bad local minima. **k-means++** (Arthur & Vassilvitskii 2007) initializes centroids spread out:

1. Pick first centroid uniformly at random.
2. For each subsequent centroid, pick a point $x$ with probability proportional to $\min_k \|x - \mu_k\|^2$ — far from existing centroids.

Provides $O(\log K)$-approximation guarantees and dramatically improves convergence empirically. **Default in sklearn.**

### Choosing K

- **Elbow method**: plot WCSS vs $K$; find the "elbow" where additional clusters give diminishing returns. Ad-hoc.
- **Silhouette score**: average of $(b - a) / \max(a, b)$ where $a$ = mean distance to own cluster, $b$ = mean distance to nearest other cluster. Range $[-1, 1]$. Higher = better clustering.
- **Gap statistic**: compare WCSS to expected WCSS under a reference null distribution. More principled.
- **Domain knowledge**: often the best answer.

### K-means failure modes

- **Non-spherical clusters**: K-means uses Euclidean distance, prefers spherical clusters. Fails on elongated, curved, or nested clusters.
- **Different cluster sizes**: K-means tends to balance cluster sizes (centroid is "pulled" toward more data).
- **Different cluster densities**: high-density clusters dominate; low-density ones may be split.
- **Outliers**: pull centroids toward them.
- **Local minima**: bad initialization → wrong clustering. Mitigate with k-means++ and multiple restarts.

### Mini-batch K-means

For large data: sample mini-batches; update centroids incrementally. Trades some quality for scalability. Used for clustering millions of samples.

---

## 3. Gaussian Mixture Models (GMM)

**K-means with covariance.** Each cluster is a Gaussian; data is a weighted mixture.

### The model

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

Parameters: mixture weights $\pi_k$, means $\mu_k$, covariances $\Sigma_k$. Soft assignments: each point belongs partially to each cluster.

### EM algorithm for GMM

**E-step**: compute posterior responsibilities (soft assignments):

$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

**M-step**: update parameters using the responsibilities as weights:

$$
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \qquad \Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i \gamma_{ik}}, \qquad \pi_k = \frac{\sum_i \gamma_{ik}}{N}
$$

Iterate until convergence. EM monotonically increases the log-likelihood.

### Why EM? Why not just MLE?

The MLE for a mixture has no closed form (the log-sum is intractable). EM is a tractable alternative that monotonically increases a lower bound on the log-likelihood (the ELBO).

### K-means as a degenerate GMM

If $\Sigma_k = \sigma^2 I$ for all $k$, mixing weights $\pi_k = 1/K$ are equal, and $\sigma \to 0$, GMM's soft assignments become hard (the closest cluster gets $\gamma = 1$), and EM reduces to K-means. So **K-means is GMM with shared spherical covariance, equal mixing weights, and hard assignments**.

### Covariance choices

- **Spherical**: $\Sigma_k = \sigma_k^2 I$ — like K-means with per-cluster scale.
- **Diagonal**: $\Sigma_k = \mathrm{diag}(\sigma_{k,1}^2, \ldots, \sigma_{k,d}^2)$ — axis-aligned ellipses.
- **Full**: arbitrary $\Sigma_k$ — full ellipsoidal clusters. Most expressive; needs most data per cluster to estimate reliably.

### When GMM beats K-means

- Elliptical (non-spherical) clusters.
- Soft assignments are useful (uncertainty quantification).
- Probabilistic interpretation needed.

### Failure modes

- **Singular covariances**: a cluster with very few points can shrink $\Sigma$ to near-zero, blowing up likelihood. Fix: regularization, minimum eigenvalue constraints.
- **Local minima**: like K-means, EM converges to a local optimum.

---

## 4. DBSCAN

Density-Based Spatial Clustering of Applications with Noise. Ester et al. 1996.

### The idea

Clusters are dense regions of points; sparse regions are noise. Two parameters:

- $\varepsilon$: radius for neighborhood.
- `min_samples`: minimum points in $\varepsilon$-neighborhood for a "core" point.

### Definitions

- **Core point**: has $\geq$ `min_samples` neighbors within $\varepsilon$.
- **Border point**: not a core point, but in the $\varepsilon$-neighborhood of one.
- **Noise**: neither core nor border.
- **Density-connected**: chain of core points within $\varepsilon$ of each other.

A cluster = maximal set of density-connected points.

### The algorithm

For each unvisited point:
1. If core: start a new cluster; add all density-connected points (BFS/DFS).
2. If border: assign to a neighboring cluster (or noise if no core neighbors).
3. If noise: leave unassigned.

### Strengths

- **Arbitrary shapes**: can find non-convex clusters (concentric circles, S-curves, etc.).
- **Noise detection**: outliers explicitly identified.
- **No need to specify $K$**: discovers number of clusters from data.

### Weaknesses

- **Sensitive to $\varepsilon$**: too small → many noise points; too large → clusters merge.
- **Varying density**: a single $\varepsilon$ doesn't fit clusters with different densities.
- **High dimensions**: distances become uniform; $\varepsilon$ becomes meaningless. Curse of dimensionality.

### Choosing $\varepsilon$

K-distance plot: for each point, compute distance to its $k$-th nearest neighbor; sort; plot. The "knee" is a good $\varepsilon$. With `min_samples = k`.

### HDBSCAN

Hierarchical DBSCAN. Removes the $\varepsilon$ parameter by computing cluster stability across all density levels. Better for varying-density data. Slower but more robust.

---

## 5. Hierarchical clustering

Build a tree of clusters by merging or splitting.

### Agglomerative (bottom-up)

1. Start with each point as its own cluster.
2. Merge the two closest clusters.
3. Repeat until one cluster remains.

Result: a **dendrogram**. Cut at any height to get a clustering with that many clusters.

### Linkage criteria

How to measure distance between clusters:
- **Single linkage**: min distance between any pair. Produces "chaining" — long, thin clusters.
- **Complete linkage**: max distance between any pair. Produces compact, spherical clusters.
- **Average linkage**: mean distance between pairs. Compromise.
- **Ward's linkage**: minimize within-cluster variance increase. Most common; produces well-separated clusters.

### Pros

- No need to specify $K$ in advance — examine the dendrogram.
- Hierarchy is interpretable.
- Deterministic (given linkage and distance).

### Cons

- $O(N^2)$ memory (distance matrix), $O(N^3)$ naive algorithm. Limits to $N \sim 10^4$.
- Greedy: early bad merges propagate.
- Sensitive to noise.

### Divisive (top-down)

Less common. Start with one cluster; recursively split.

---

## 6. Spectral clustering

Cluster using the eigenstructure of a similarity graph.

### The recipe

1. Build similarity graph $W$ (e.g., Gaussian kernel of distances).
2. Compute graph Laplacian $L = D - W$ (or normalized).
3. Eigendecompose $L$; take bottom $K$ eigenvectors.
4. Cluster the eigenvectors (typically with K-means).

### Why it works

The eigenvectors of $L$ correspond to "smooth" functions on the graph. The first $K$ eigenvectors approximately indicate cluster membership. Especially good for non-convex shapes.

### Pros

- Handles non-convex clusters (where K-means fails).
- Theoretically grounded (graph Laplacian theory).

### Cons

- $O(N^3)$ eigendecomposition. Hard at scale.
- Choice of similarity function and number of nearest neighbors matters.

---

## 7. Evaluation of clustering

Hard, because there's no ground truth.

### Internal metrics

Use only the data and the clustering, no labels.

**Silhouette coefficient**: $(b - a)/\max(a, b)$. Range $[-1, 1]$. Higher = better separation.

**Davies-Bouldin index**: average of cluster-pair similarities. Lower = better.

**Calinski-Harabasz index**: ratio of between-cluster to within-cluster variance. Higher = better.

### External metrics

Require ground-truth labels (when available).

**Adjusted Rand Index (ARI)**: counts pairs that are in the same/different clusters in both predictions and labels, adjusted for chance. Range $[-1, 1]$.

**Normalized Mutual Information (NMI)**: $I(C; Y) / \sqrt{H(C) H(Y)}$. Information-theoretic; $[0, 1]$.

**V-measure**: harmonic mean of homogeneity (each cluster contains samples of one class) and completeness (each class is in one cluster).

### Why this is hard

Clustering is task-dependent: the "right" clustering depends on what you'll do with it. Internal metrics measure compactness/separation but may not align with downstream utility. **Best practice**: evaluate on a downstream task, not just clustering metrics.

---

## 8. The curse of dimensionality

In high-dimensional spaces, all pairwise distances become similar. Clustering relies on distance-based grouping, so high-dim data is hard.

### Symptoms

- All clusters look "equally far" from any query.
- Density (DBSCAN) becomes meaningless.
- K-means converges to weird, near-uniform partitions.

### Mitigations

- **Dimensionality reduction first**: PCA, UMAP, or autoencoder embeddings. Then cluster in the reduced space.
- **Domain-specific kernels**: cosine for text, perceptual distances for images.
- **Use the right method**: GMM on PCA-reduced embeddings is a strong default.

---

## 9. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why does K-means converge?" | Both steps decrease the WCSS; objective is bounded below; coordinate descent on a quadratic. Converges to a local min. |
| "K-means vs GMM?" | K-means: hard assignments, spherical clusters. GMM: soft assignments via EM, elliptical clusters. K-means is a degenerate GMM (shared spherical $\Sigma$, $\sigma \to 0$). |
| "Why use k-means++?" | Spread initial centroids; $O(\log K)$-approximation; avoids bad local minima from random init. |
| "How do you choose $K$?" | Elbow on WCSS, silhouette, gap statistic, or domain knowledge. There's no universal answer. |
| "When does DBSCAN beat K-means?" | Non-convex/arbitrary-shape clusters, when noise/outliers must be detected, when $K$ is unknown. |
| "DBSCAN's main weakness?" | Sensitive to $\varepsilon$. Varying-density clusters need different $\varepsilon$ — single value can't fit both. HDBSCAN fixes this. |
| "Why does spectral clustering work on non-convex shapes?" | Operates on graph eigenstructure, not Euclidean distance. Captures connectivity rather than centroid distance. |
| "How do you evaluate clustering without labels?" | Silhouette, Davies-Bouldin, Calinski-Harabasz. None is a perfect substitute for downstream task evaluation. |
| "Curse of dimensionality?" | High-dim distances are uniform; clustering breaks down. Reduce dimensionality first or use domain-aware similarity. |

---

## 10. The 8 most-asked clustering interview questions

1. **Walk me through K-means.** Initialize → assign points to nearest centroid → update centroid to cluster mean → repeat. Coordinate descent on WCSS.
2. **Why does K-means converge?** Both steps decrease objective; bounded below; coordinate descent.
3. **K-means vs GMM?** Hard vs soft assignments; spherical vs elliptical clusters; K-means is a degenerate GMM.
4. **Walk through EM for GMM.** E-step: posterior responsibilities. M-step: weighted MLE updates of $\mu, \Sigma, \pi$. Iterate.
5. **DBSCAN — how does it work?** Core points (dense neighborhoods) + density-connected expansion. Discovers clusters and noise.
6. **Spectral clustering?** Eigendecompose graph Laplacian; bottom $K$ eigenvectors are cluster indicators; cluster the eigenvectors.
7. **How do you choose $K$?** Elbow method, silhouette, gap statistic, or domain knowledge.
8. **How do you evaluate clustering?** Internal (silhouette, DB, CH) without labels; external (ARI, NMI) with ground truth; downstream task is best.

---

## 11. Drill plan

1. Master K-means as coordinate descent on WCSS.
2. Walk through GMM EM end-to-end.
3. Know DBSCAN's core/border/noise classification.
4. Know spectral clustering's graph Laplacian basis.
5. Drill `INTERVIEW_GRILL.md`.

---

## 12. Further reading

- Lloyd, "Least Squares Quantization in PCM" (K-means, 1957/1982).
- Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (2007).
- Dempster, Laird, Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm" (1977).
- Ester et al., "DBSCAN" (1996).
- von Luxburg, "A Tutorial on Spectral Clustering" (2007).
- McInnes & Healy, "HDBSCAN" (2017).
