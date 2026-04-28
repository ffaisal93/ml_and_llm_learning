# Clustering — Interview Grill

> 40 questions on clustering algorithms. Drill until you can answer 28+ cold.

---

## A. K-means

**1. Walk me through K-means.**
Initialize $K$ centroids. Repeat: (1) assign each point to nearest centroid; (2) update each centroid to the mean of its assigned points. Stop when assignments don't change.

**2. What objective does K-means minimize?**
Within-cluster sum of squares: $\mathcal{L} = \sum_i \|x_i - \mu_{c_i}\|^2$. K-means is coordinate descent on this objective.

**3. Why does K-means converge?**
Both steps decrease the objective: assignment minimizes per-point distance; centroid update is the closed-form mean. Bounded below by 0 → monotonic decrease → convergence to local min.

**4. Does K-means find the global optimum?**
No. Local minimum only (the WCSS objective is non-convex). Different initializations give different results.

**5. What's k-means++?**
Smarter initialization (Arthur & Vassilvitskii 2007). Pick first centroid randomly, then each next centroid with probability $\propto \min_k \|x - \mu_k\|^2$ — far from existing centroids. Provides $O(\log K)$-approximation guarantees and dramatically improves convergence.

**6. How do you choose $K$?**
Elbow method (WCSS plateau), silhouette score, gap statistic, or domain knowledge. There's no universal answer.

**7. K-means failure modes?**
Non-spherical clusters (assumes Euclidean distance). Different cluster sizes (centroid pulled toward majority). Different densities. Outliers (centroid drifts). Bad init → local minimum.

**8. How does K-means handle outliers?**
Poorly. Outliers pull centroids toward them. Mitigations: K-medoids (use medians), pre-filter outliers, robust variants.

**9. Mini-batch K-means?**
Sample mini-batches; update centroids incrementally (running mean). Trades some quality for scalability. Used for $N > 10^6$.

**10. What if $K$ is too large vs too small?**
Too large: clusters split unnecessarily; centroids over-fit local noise. Too small: distinct concepts merged; clusters become amorphous. Pick $K$ via elbow/silhouette.

---

## B. Gaussian Mixture Models

**11. What's a GMM?**
$p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$. Each cluster is a Gaussian; data is a weighted mixture. Soft assignments.

**12. K-means vs GMM relationship?**
K-means is GMM with shared spherical covariance $\Sigma_k = \sigma^2 I$, equal mixing weights $\pi_k = 1/K$, and $\sigma \to 0$. Soft assignments become hard; EM reduces to K-means.

**13. Walk me through EM for GMM.**

E-step: posterior responsibilities

$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

M-step: weighted MLE updates

$$
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \qquad \Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i \gamma_{ik}}, \qquad \pi_k = \frac{\sum_i \gamma_{ik}}{N}
$$

Iterate until convergence.

**14. Why EM and not direct MLE?**
Mixture log-likelihood has no closed form (log of a sum). EM provides a tractable lower bound (the ELBO) that monotonically increases. Direct MLE is non-trivial.

**15. Why does EM converge?**
Each E-step constructs a tight lower bound at current params. Each M-step maximizes that bound, increasing the true likelihood. Bounded above → convergence (to a local max).

**16. Covariance choices in GMM?**
Spherical ($\Sigma = \sigma^2 I$): like K-means with scale per cluster. Diagonal: axis-aligned ellipses. Full: arbitrary ellipsoids — most expressive but needs more data.

**17. GMM failure modes?**
Singular covariances (cluster shrinks to a point, likelihood blows up — fix with regularization). Local minima (bad init). Wrong $K$.

**18. Why use GMM over K-means?**
Soft assignments (uncertainty), elliptical clusters, probabilistic interpretation. Use K-means when speed matters and clusters are roughly spherical.

---

## C. DBSCAN

**19. What's DBSCAN?**
Density-based clustering. Two parameters: $\varepsilon$ (radius), `min_samples` (density threshold). Core points have $\geq$ min_samples neighbors within $\varepsilon$. Clusters = connected components of core points + reachable border points.

**20. Core, border, noise — define them.**
Core: $\geq$ min_samples neighbors within $\varepsilon$. Border: not core, but in $\varepsilon$-neighborhood of a core point. Noise: neither core nor border.

**21. Why does DBSCAN handle non-convex shapes?**
Connectivity-based, not centroid-based. Two points are in the same cluster if connected through a chain of dense neighborhoods, regardless of overall shape.

**22. DBSCAN strengths?**
Arbitrary cluster shapes. Noise detection (outliers explicitly identified). No need to specify $K$.

**23. DBSCAN weaknesses?**
Sensitive to $\varepsilon$. Varying density (single $\varepsilon$ doesn't fit clusters with different densities). Curse of dimensionality (distances become uniform in high-dim).

**24. How do you choose $\varepsilon$?**
K-distance plot: for each point, distance to its $k$-th nearest neighbor; sort; plot. Find the "knee" — that's $\varepsilon$. With `min_samples = k`.

**25. What's HDBSCAN?**
Hierarchical DBSCAN. No $\varepsilon$ parameter — computes cluster stability across all density levels. Better for varying-density data. Slower but more robust.

---

## D. Hierarchical clustering

**26. What's agglomerative clustering?**
Bottom-up: start with each point as own cluster; merge closest pair iteratively until one cluster remains. Produces a dendrogram.

**27. Linkage criteria?**
Single (min distance, "chaining"), complete (max distance, compact clusters), average, Ward (minimize variance increase). Ward most common in practice.

**28. Pros of hierarchical?**
No need to specify $K$ in advance — examine dendrogram. Hierarchy is interpretable.

**29. Cons?**
$O(N^2)$ memory, $O(N^3)$ naive — limits to $N \sim 10^4$. Greedy: bad early merges propagate.

**30. Single vs complete linkage?**
Single: chaining — produces long thin clusters; sensitive to noise. Complete: compact clusters; can split natural elongated clusters. Ward is usually the default.

---

## E. Spectral clustering

**31. What's spectral clustering?**
Build similarity graph $W$. Compute Laplacian $L = D - W$. Eigendecompose; take bottom $K$ eigenvectors. Cluster the eigenvectors (typically with K-means).

**32. Why does it handle non-convex shapes?**
Operates on graph connectivity, not Euclidean distances directly. Two points are in the same cluster if connected in the similarity graph, regardless of overall shape.

**33. Cons of spectral clustering?**
$O(N^3)$ eigendecomposition. Hard to scale beyond $N \sim 10^4$. Sensitive to similarity graph construction (kernel choice, k-NN parameter).

**34. Spectral vs DBSCAN?**
Both handle non-convex shapes. Spectral: principled (graph theory), needs $K$ specified. DBSCAN: density-based, finds $K$ automatically, more sensitive to parameters.

---

## F. Evaluation

**35. Internal evaluation metrics?**
Silhouette ($(b-a)/\max(a,b)$, range $[-1, 1]$). Davies-Bouldin (lower = better). Calinski-Harabasz (between/within variance ratio, higher = better). Use when no ground truth.

**36. External evaluation metrics?**
Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), V-measure (homogeneity + completeness). Require ground-truth labels.

**37. Why is clustering evaluation hard?**
No ground truth in unsupervised setting. Internal metrics reward compactness/separation but may not align with downstream utility. Best: evaluate on a downstream task.

---

## G. Subtleties

**38. Curse of dimensionality for clustering?**
In high-dim, all distances become similar — clusters indistinguishable. K-means converges to weird partitions; DBSCAN density meaningless. Mitigate: dimensionality reduction first, or use domain-aware similarity.

**39. Online clustering?**
For streaming data: incremental K-means (mini-batch), online GMM. Don't store all data; update model as data flows.

**40. Soft vs hard clustering?**
Hard: each point in exactly one cluster (K-means, DBSCAN). Soft: each point has a probability per cluster (GMM, fuzzy K-means). Soft is more informative when boundaries are uncertain.

---

## Quick fire

**41.** *K-means objective?* WCSS — within-cluster sum of squares.
**42.** *K-means++ contribution?* Spread initial centroids.
**43.** *DBSCAN parameters?* $\varepsilon$ and min_samples.
**44.** *EM monotonicity property?* Likelihood is non-decreasing.
**45.** *Default linkage?* scipy `linkage` requires `method` (no default). sklearn `AgglomerativeClustering` defaults to Ward.

---

## Self-grading

If you can't answer 1-15, you don't know clustering. If you can't answer 16-30, you'll struggle on classical ML interviews. If you can't answer 31-45, frontier-lab interviews on unsupervised learning will go past you.

Aim for 28+/45 cold.
