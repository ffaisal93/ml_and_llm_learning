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

> **Saying it out loud.** There's no best clustering algorithm, and saying that first is the right move. What you actually pick depends on three things: what shape you think the clusters are, whether you know how many there are, and how much data you've got. K-means if you expect roughly round blobs and know K, and it scales to millions of points. GMM if the blobs are stretched or you want soft memberships. DBSCAN or HDBSCAN if the shapes are weird and you want outliers flagged. Spectral if the structure is about connectivity rather than distance, and hierarchical if you want a tree — but both of those choke past about ten thousand points because of the cubic cost.

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

*In plain language:* this formula is the score K-means is trying to make small. For every point, measure how far it is from the centre of the cluster it was put in, square that distance, and add it all up. A good clustering is one where points sit close to their own centre — that's all "within-cluster sum of squares" means.

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

> **Saying it out loud.** K-means is two steps repeated: assign every point to its nearest centre, then move each centre to the average of the points assigned to it. That's it. The reason it always terminates is that both steps can only reduce the same quantity — the total squared distance from points to their own centre — and that quantity can't go below zero, so it has to stop. Formally it's coordinate descent: you're minimising one objective by alternately optimising the assignments with the centres frozen, and the centres with the assignments frozen, and both subproblems have exact closed-form answers. The catch to state is that converging isn't the same as converging to the right answer — you land in a local minimum, not the global one.

### Initialization: k-means++

Random initialization can produce bad local minima. **k-means++** (Arthur & Vassilvitskii 2007) initializes centroids spread out:

1. Pick first centroid uniformly at random.
2. For each subsequent centroid, pick a point $x$ with probability proportional to $\min_k \|x - \mu_k\|^2$ — far from existing centroids.

Provides $O(\log K)$-approximation guarantees and dramatically improves convergence empirically. **Default in sklearn.**

> **Saying it out loud.** Random initialisation is genuinely bad for K-means — drop two centres in the same natural cluster and they'll fight over it forever while a real cluster elsewhere goes unrepresented. k-means++ fixes it by seeding greedily: pick the first centre at random, then pick each next one with probability proportional to its squared distance from the nearest existing centre, so far-away points are strongly favoured. It costs one extra pass over the data and it comes with a provable guarantee — you land within a factor of order log K of optimal in expectation. It's the default in scikit-learn and there's basically no reason not to use it, plus a handful of random restarts on top.

### Choosing K

- **Elbow method**: plot WCSS vs $K$; find the "elbow" where additional clusters give diminishing returns. Ad-hoc.
- **Silhouette score**: average of $(b - a) / \max(a, b)$ where $a$ = mean distance to own cluster, $b$ = mean distance to nearest other cluster. Range $[-1, 1]$. Higher = better clustering.
- **Gap statistic**: compare WCSS to expected WCSS under a reference null distribution. More principled.
- **Domain knowledge**: often the best answer.

> **Saying it out loud.** There's no principled way to choose K, and admitting that is better than pretending. The elbow method plots the objective against K and looks for where it stops dropping sharply, but real elbows are often ambiguous. Silhouette is better because it actually measures whether points are closer to their own cluster than to the next nearest one, and it's on a fixed scale from minus one to one so you can compare across K. The gap statistic is the most principled — it compares your objective against what you'd get on random data with no structure. But honestly the best answer is usually domain knowledge or downstream utility: if the clusters are going to become customer segments, five segments a marketing team can act on beats seventeen that score marginally better.

### K-means failure modes

- **Non-spherical clusters**: K-means uses Euclidean distance, prefers spherical clusters. Fails on elongated, curved, or nested clusters.
- **Different cluster sizes**: K-means tends to balance cluster sizes (centroid is "pulled" toward more data).
- **Different cluster densities**: high-density clusters dominate; low-density ones may be split.
- **Outliers**: pull centroids toward them.
- **Local minima**: bad initialization → wrong clustering. Mitigate with k-means++ and multiple restarts.

> **Saying it out loud.** K-means fails in a specific and predictable way: it assumes clusters are round, roughly the same size, and roughly the same density, because all it can do is draw straight-line boundaries equidistant between centres. So give it two long parallel stripes and it'll slice them crosswise; give it a small tight cluster next to a big diffuse one and it'll steal points from the big one. Outliers drag centres toward themselves because it's minimising squared distance, which weights far points heavily. And bad initialisation gives you a bad local optimum. The one-sentence version to say is: K-means draws straight-line boundaries at the midpoints between centres, so anything that isn't well described by that will be clustered wrong.

### Mini-batch K-means

For large data: sample mini-batches; update centroids incrementally. Trades some quality for scalability. Used for clustering millions of samples.

> **Saying it out loud.** Mini-batch K-means is what you use when the data doesn't fit in memory. Instead of touching every point each iteration, you sample a batch, assign it, and nudge the centres with a decaying learning rate. It converges to a slightly worse objective than full K-means but it's orders of magnitude faster, and on millions of points that difference is what makes the job possible at all. The tradeoff to name is quality versus scale, and in practice the quality gap is small enough that mini-batch is the default above a few hundred thousand points.

---

## 3. Gaussian Mixture Models (GMM)

*In plain language:* a mixture model says your data came from several different bell-shaped blobs, and you don't know which blob produced which point. Fitting it means simultaneously guessing the blobs and guessing the assignments, which is why it's done by alternating between the two. Each blob gets a centre, a shape, and a weight saying how much of the data it accounts for.

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

> **Saying it out loud.** A Gaussian mixture is the soft version of K-means. Rather than each point belonging to exactly one cluster, it gets a probability of belonging to each — the responsibility — and clusters can be stretched ellipses rather than circles because each one carries its own covariance matrix. You fit it with EM, which alternates: given the current blobs, compute how responsible each blob is for each point; then given those responsibilities, refit each blob as a weighted average. The reason you can't just do maximum likelihood directly is that the log of a sum doesn't decompose, so there's no closed-form solution — EM sidesteps that by optimising a lower bound instead, and it's guaranteed to increase the likelihood every iteration.

### K-means as a degenerate GMM

If $\Sigma_k = \sigma^2 I$ for all $k$, mixing weights $\pi_k = 1/K$ are equal, and $\sigma \to 0$, GMM's soft assignments become hard (the closest cluster gets $\gamma = 1$), and EM reduces to K-means. So **K-means is GMM with shared spherical covariance, equal mixing weights, and hard assignments**.

> **Saying it out loud.** Here's the connection that interviewers love: K-means is just a GMM with the dials pinned. Force every cluster to share the same spherical covariance, force the mixing weights equal, and then shrink the variance toward zero — the soft responsibilities collapse to hard zero-or-one assignments, and EM becomes exactly the K-means assign-and-update loop. So they're not two unrelated algorithms, they're the same algorithm with different amounts of flexibility. Saying that cleanly tells the interviewer you understand both rather than having memorised two recipes.

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

> **Saying it out loud.** The covariance choice is a straight bias-variance trade and it's worth framing that way. Spherical means one number per cluster, cheap and stable but only round blobs. Diagonal means one number per dimension, so you get axis-aligned ellipses. Full covariance can represent any tilted ellipse but needs on the order of d-squared parameters per cluster, so in high dimensions you'll be estimating thousands of numbers from a handful of points. The failure mode that actually bites is singular covariance: a cluster that captures two or three points can shrink to a spike, the likelihood goes to infinity, and the fit blows up. The fix is a small ridge added to the diagonal, and every real implementation does this by default.

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

> **Saying it out loud.** DBSCAN throws out the idea of a cluster centre entirely and defines clusters by density instead: a point is a core point if it has enough neighbours within a given radius, and clusters are the connected chains of core points. Anything not reachable from a core point is labelled noise rather than forced into a cluster. That gives you two things K-means can't do — arbitrary shapes, like two concentric rings or an S-curve, and explicit outlier detection. And you don't have to specify the number of clusters. The price is that you've traded one hard parameter for another: the radius is fiddly, and because it's a single global value, data with clusters of genuinely different densities cannot be handled — tune the radius for the dense cluster and the sparse one becomes noise.

### Choosing $\varepsilon$

K-distance plot: for each point, compute distance to its $k$-th nearest neighbor; sort; plot. The "knee" is a good $\varepsilon$. With `min_samples = k`.

### HDBSCAN

Hierarchical DBSCAN. Removes the $\varepsilon$ parameter by computing cluster stability across all density levels. Better for varying-density data. Slower but more robust.

> **Saying it out loud.** The practical way to set the radius is the k-distance plot: for every point, measure the distance to its k-th nearest neighbour, sort those values, and plot them. The curve stays flat and then bends sharply upward, and the knee is where points stop being in dense regions — that's your radius. It's the same eyeballing exercise as the elbow method and it's about as reliable. If you'd rather not do it at all, HDBSCAN runs DBSCAN across every density level at once and keeps the clusters that persist longest, which removes the parameter and handles varying density. It's slower and harder to implement, but it's the better default for messy real data.

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

> **Saying it out loud.** The linkage rule is the whole personality of hierarchical clustering. Single linkage measures cluster distance by the closest pair, which lets it follow thin winding shapes but also causes chaining — one bridge of noise points welds two real clusters together. Complete linkage uses the farthest pair, so it insists on compact balls and will happily split an elongated cluster. Average is the compromise, and Ward's merges whichever pair increases within-cluster variance least, which is essentially the K-means objective done greedily and is why it's the usual default. The point to make is that you're not choosing a distance, you're choosing a bias about cluster shape.

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

> **Saying it out loud.** Hierarchical clustering builds a tree instead of a flat partition: start with every point its own cluster, repeatedly merge the two closest, and record the order. You get a dendrogram, and you cut it at whatever height gives you the number of clusters you want — so you don't have to commit to K up front, and the nesting itself is often the interesting output, as in taxonomy or document hierarchies. It's also fully deterministic, unlike K-means. The hard limit is cost: you need the full pairwise distance matrix, which is N-squared memory and up to N-cubed time, so you're capped around ten thousand points. And merges are greedy and irreversible — one bad early merge is baked in forever.

---

## 6. Spectral clustering

*In plain language:* instead of measuring straight-line distance, spectral clustering builds a graph where nearby points are connected, then asks where you'd cut that graph to separate it into pieces with the fewest severed connections. The eigenvector machinery below is just an efficient, relaxed way of finding that cut. The point is that it groups by connectivity rather than by proximity to a centre.

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

> **Saying it out loud.** Spectral clustering works when the clusters are defined by connectivity rather than compactness — the textbook case is two concentric circles, where K-means fails completely because the centres coincide but spectral gets it instantly. You build a similarity graph, take the Laplacian, and use its smallest eigenvectors as a new set of coordinates in which the clusters *are* round, then run K-means there. So it's really a change of representation followed by ordinary clustering. The two costs are that the eigendecomposition is cubic in the number of points, and that the result is quite sensitive to how you build the graph — the kernel width or the neighbour count. Get that wrong and the graph is either disconnected or fully connected and you learn nothing.

---

## 7. Evaluation of clustering

Hard, because there's no ground truth.

### Internal metrics

Use only the data and the clustering, no labels.

**Silhouette coefficient**: $(b - a)/\max(a, b)$. Range $[-1, 1]$. Higher = better separation.

**Davies-Bouldin index**: average of cluster-pair similarities. Lower = better.

**Calinski-Harabasz index**: ratio of between-cluster to within-cluster variance. Higher = better.

> **Saying it out loud.** Internal metrics score a clustering using only the data, no labels — they're all measuring some version of "tight inside, far apart outside." Silhouette is the one to know: for each point, compare its average distance to its own cluster against its average distance to the nearest other cluster, normalise, and average over everything, giving a number from minus one to one. Davies-Bouldin and Calinski-Harabasz are the same instinct with different arithmetic. The catch worth stating is that all of them are biased toward round, well-separated clusters, so they'll systematically rate a K-means solution above a correct DBSCAN one on non-convex data. They measure geometry, not correctness.

### External metrics

Require ground-truth labels (when available).

**Adjusted Rand Index (ARI)**: counts pairs that are in the same/different clusters in both predictions and labels, adjusted for chance. Range $[-1, 1]$.

**Normalized Mutual Information (NMI)**: $I(C; Y) / \sqrt{H(C) H(Y)}$. Information-theoretic; $[0, 1]$.

**V-measure**: harmonic mean of homogeneity (each cluster contains samples of one class) and completeness (each class is in one cluster).

### Why this is hard

Clustering is task-dependent: the "right" clustering depends on what you'll do with it. Internal metrics measure compactness/separation but may not align with downstream utility. **Best practice**: evaluate on a downstream task, not just clustering metrics.

> **Saying it out loud.** The honest answer to "how do you evaluate clustering" is that it's genuinely hard, because there's no ground truth and no single right partition — the correct clustering depends on what you're going to do with it. If you happen to have labels you can use Adjusted Rand Index or normalised mutual information, both corrected so that random clusterings score around zero rather than something misleadingly positive. If you don't, internal metrics give you a rough geometric sanity check and nothing more. The answer that actually scores is: evaluate on the downstream task. If these clusters feed a recommender, measure the recommender; a clustering with a mediocre silhouette that lifts click-through is the better clustering.

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

> **Saying it out loud.** In high dimensions, distance stops meaning anything, and clustering is built entirely on distance. The reason is that as you add dimensions, the gap between the nearest and farthest neighbour shrinks relative to the typical distance — everything ends up roughly equidistant from everything else. So DBSCAN's density becomes meaningless, K-means produces near-arbitrary partitions, and your silhouette scores hover near zero no matter what you do. The fix is to not cluster in the raw space: reduce first with PCA or UMAP or a learned embedding, or use a similarity that's meaningful for your domain, like cosine for text. The strong default to name is a GMM or K-means on top of a reduced embedding, typically 10 to 50 dimensions, rather than on the raw thousands.

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
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

---

## 12. Further reading

- Lloyd, "Least Squares Quantization in PCM" (K-means, 1957/1982).
- Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (2007).
- Dempster, Laird, Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm" (1977).
- Ester et al., "DBSCAN" (1996).
- von Luxburg, "A Tutorial on Spectral Clustering" (2007).
- McInnes & Healy, "HDBSCAN" (2017).
