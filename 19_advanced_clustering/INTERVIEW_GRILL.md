# Clustering — Interview Grill

> 40 questions on clustering algorithms. Drill until you can answer 28+ cold.

---

## A. K-means

**1. Walk me through K-means.**
Initialize $K$ centroids. Repeat: (1) assign each point to nearest centroid; (2) update each centroid to the mean of its assigned points. Stop when assignments don't change.

> **Saying it out loud.** Pick K starting points as your cluster centres. Then repeat two steps until nothing changes: assign every data point to whichever centre is nearest, then move each centre to the average of the points that chose it. That's the whole algorithm — it's about five lines of code. The intuition is that you're letting the centres and the memberships negotiate until they agree.

**2. What objective does K-means minimize?**
Within-cluster sum of squares: $\mathcal{L} = \sum_i \|x_i - \mu_{c_i}\|^2$. K-means is coordinate descent on this objective.

> **Saying it out loud.** It minimises the total squared distance from each point to its own cluster's centre — within-cluster sum of squares. Both steps of the algorithm are exactly minimising that same quantity, one variable at a time, which is why it's coordinate descent. That framing is what interviewers want, because it immediately explains why it converges and why it only finds a local optimum.

**3. Why does K-means converge?**
Both steps decrease the objective: assignment minimizes per-point distance; centroid update is the closed-form mean. Bounded below by 0 → monotonic decrease → convergence to local min.

> **Saying it out loud.** Because both steps can only push the same number down, and that number can't go below zero. Reassigning a point to a nearer centre reduces its contribution; recomputing a centre as the mean is provably the best possible centre for those points. Monotone decrease plus a lower bound means it has to stop, and since assignments are discrete, it stops in finitely many steps. What it converges *to* is a local minimum, and that distinction is the follow-up they're waiting for.

**4. Does K-means find the global optimum?**
No. Local minimum only (the WCSS objective is non-convex). Different initializations give different results.

> **Saying it out loud.** No — the objective is non-convex and the algorithm is greedy, so you get a local minimum that depends entirely on where you started. Finding the true optimum is NP-hard in general. The practical mitigation is k-means++ initialisation plus several random restarts, taking the best objective, and scikit-learn does ten restarts by default for exactly this reason.

**5. What's k-means++?**
Smarter initialization (Arthur & Vassilvitskii 2007). Pick first centroid randomly, then each next centroid with probability $\propto \min_k \|x - \mu_k\|^2$ — far from existing centroids. Provides $O(\log K)$-approximation guarantees and dramatically improves convergence.

> **Saying it out loud.** It's a smarter way to place the initial centres. First one at random, then each subsequent one sampled with probability proportional to its squared distance from the nearest centre already chosen — so far-flung points are heavily favoured and your centres start spread out. It costs one extra pass over the data and comes with a guarantee: expected objective within a factor of order log K of optimal. It's the default everywhere and there's no good reason to skip it.

**6. How do you choose $K$?**
Elbow method (WCSS plateau), silhouette score, gap statistic, or domain knowledge. There's no universal answer.

> **Saying it out loud.** There's no principled universal answer, and saying that up front is better than pretending. Elbow on the objective is the quick eyeball, silhouette is better because it's on a fixed comparable scale, and the gap statistic is the most principled because it compares against structureless reference data. But in practice domain knowledge usually wins — if these become customer segments, the number a team can actually act on matters more than a marginal metric gain.

**7. K-means failure modes?**
Non-spherical clusters (assumes Euclidean distance). Different cluster sizes (centroid pulled toward majority). Different densities. Outliers (centroid drifts). Bad init → local minimum.

> **Saying it out loud.** It draws straight-line boundaries halfway between centres, so anything that shape can't describe goes wrong. Elongated or curved or nested clusters get sliced apart. Clusters of very different sizes get rebalanced, because a big cluster pulls its centre and steals points from a small neighbour. Different densities cause the same problem. And outliers drag centres because squared distance weights far points heavily. That single sentence about straight-line midpoint boundaries covers most of the list.

**8. How does K-means handle outliers?**
Poorly. Outliers pull centroids toward them. Mitigations: K-medoids (use medians), pre-filter outliers, robust variants.

> **Saying it out loud.** Badly. It minimises squared distance, so a single far-away point contributes enormously and drags its centre toward it, distorting the whole cluster. Options are K-medoids, which uses actual data points as centres and an absolute-distance objective so it's much more robust, or just filtering outliers first. The general principle to name is that squared error is not robust — that's true here, in linear regression, and everywhere else.

**9. Mini-batch K-means?**
Sample mini-batches; update centroids incrementally (running mean). Trades some quality for scalability. Used for $N > 10^6$.

> **Saying it out loud.** Instead of touching every point each iteration you sample a batch, assign it, and nudge the centres with a decaying step size. It converges to a slightly worse objective but runs orders of magnitude faster, which is what makes clustering millions of points feasible at all. The tradeoff is quality against scale, and above roughly a million points it's not really a choice.

**10. What if $K$ is too large vs too small?**
Too large: clusters split unnecessarily; centroids over-fit local noise. Too small: distinct concepts merged; clusters become amorphous. Pick $K$ via elbow/silhouette.

> **Saying it out loud.** Too large and you split real clusters and start modelling noise — in the limit K equals N and the objective is zero and completely meaningless, which is why you can't just minimise the objective over K. Too small and you merge genuinely distinct groups into an amorphous blob. That degenerate case is worth mentioning, because it explains why elbow and silhouette exist at all: the raw objective always improves with more clusters.

---

## B. Gaussian Mixture Models

**11. What's a GMM?**
$p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$. Each cluster is a Gaussian; data is a weighted mixture. Soft assignments.

> **Saying it out loud.** A Gaussian mixture says the data came from several bell-shaped blobs and you don't know which one produced each point. Each blob has a centre, a covariance describing its shape and tilt, and a weight for how much of the data it accounts for. Because it's a real probability model you get soft memberships — a point can be 70 percent one cluster and 30 percent another — and you can score the likelihood of new data, which K-means can't do.

**12. K-means vs GMM relationship?**
K-means is GMM with shared spherical covariance $\Sigma_k = \sigma^2 I$, equal mixing weights $\pi_k = 1/K$, and $\sigma \to 0$. Soft assignments become hard; EM reduces to K-means.

> **Saying it out loud.** K-means is a GMM with everything locked down. Force all clusters to share one spherical covariance, force equal mixing weights, then let the variance shrink toward zero — the soft responsibilities snap to hard zero-or-one and EM becomes exactly the assign-and-update loop. So they aren't rival algorithms, they're the same algorithm at different levels of flexibility. Saying that connection cleanly is one of the highest-value sentences in a clustering interview.

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

> **Saying it out loud.** Two alternating steps, same rhythm as K-means but soft. The E-step asks: given the blobs as they currently stand, what fraction of responsibility does each blob have for each point? That's just the density of that blob at that point, normalised across blobs. The M-step then refits each blob as a responsibility-weighted mean and covariance, and sets its weight to its total share of responsibility. Repeat until the likelihood stops improving. The thing to say is that it's K-means with fractional membership instead of committed membership.

**14. Why EM and not direct MLE?**
Mixture log-likelihood has no closed form (log of a sum). EM provides a tractable lower bound (the ELBO) that monotonically increases. Direct MLE is non-trivial.

> **Saying it out loud.** Because the mixture log-likelihood has a log of a sum inside it, and that doesn't break apart into anything you can solve in closed form — take derivatives and you get a coupled mess. EM sidesteps it by introducing the unknown assignments as latent variables, which turns the log-of-a-sum into a sum-of-logs that you *can* solve. You're optimising a lower bound rather than the thing itself, and the bound is tight at the current parameters, which is what makes the guarantee work.

**15. Why does EM converge?**
Each E-step constructs a tight lower bound at current params. Each M-step maximizes that bound, increasing the true likelihood. Bounded above → convergence (to a local max).

> **Saying it out loud.** Because the E-step builds a lower bound on the log-likelihood that touches it exactly at the current parameters, and the M-step maximises that bound. So the new parameters are at least as good on the bound, and since the bound touched the true likelihood, the true likelihood can only go up. Monotone increase plus an upper bound gives convergence — to a local maximum, and to a saddle point in bad cases. That tight-bound-then-maximise structure is the same idea as the ELBO in variational inference, and mentioning that connection lands well.

**16. Covariance choices in GMM?**
Spherical ($\Sigma = \sigma^2 I$): like K-means with scale per cluster. Diagonal: axis-aligned ellipses. Full: arbitrary ellipsoids — most expressive but needs more data.

> **Saying it out loud.** It's a bias-variance dial. Spherical is one parameter per cluster — stable but only round blobs. Diagonal gives you one per dimension, so axis-aligned ellipses. Full covariance gives you any tilted ellipse but costs on the order of d-squared parameters per cluster, so in high dimensions you're estimating thousands of numbers from very few points and it overfits. The rule of thumb is that you want many more points per cluster than dimensions before full covariance is safe.

**17. GMM failure modes?**
Singular covariances (cluster shrinks to a point, likelihood blows up — fix with regularization). Local minima (bad init). Wrong $K$.

> **Saying it out loud.** The distinctive one is singular covariance: a cluster latches onto two or three points, its covariance collapses toward zero, and the likelihood shoots to infinity — the fit is technically optimal and completely useless. That's a genuine degeneracy in the objective, not a numerical accident. Every implementation adds a small ridge to the diagonal to prevent it. Beyond that, EM has local optima like K-means, so multiple restarts matter, and you still have to choose K, usually by BIC.

**18. Why use GMM over K-means?**
Soft assignments (uncertainty), elliptical clusters, probabilistic interpretation. Use K-means when speed matters and clusters are roughly spherical.

> **Saying it out loud.** Use GMM when the clusters are stretched or tilted, when you want a probability rather than a hard label, or when you want to score how likely new data is under the model — which makes it an anomaly detector for free. Use K-means when you have a lot of data, the clusters are roughly round, and you want speed. The tradeoff to name is that GMM has far more parameters per cluster, so it needs more data and is more prone to degenerate fits.

---

## C. DBSCAN

**19. What's DBSCAN?**
Density-based clustering. Two parameters: $\varepsilon$ (radius), `min_samples` (density threshold). Core points have $\geq$ min_samples neighbors within $\varepsilon$. Clusters = connected components of core points + reachable border points.

> **Saying it out loud.** DBSCAN defines clusters by density rather than by centres. A point is a core point if it has at least min_samples neighbours within a radius epsilon; clusters are the connected chains of such core points plus the sparse points hanging off their edges; and anything left over is explicitly labelled noise. So it discovers how many clusters there are, and it refuses to force every point into one — those two properties are what people come to it for.

**20. Core, border, noise — define them.**
Core: $\geq$ min_samples neighbors within $\varepsilon$. Border: not core, but in $\varepsilon$-neighborhood of a core point. Noise: neither core nor border.

> **Saying it out loud.** A core point is dense: it has at least min_samples other points within the radius. A border point isn't dense itself but sits inside some core point's neighbourhood, so it gets pulled into that cluster. Noise is neither — too far from everything. The subtlety worth mentioning is that a border point reachable from two clusters gets assigned by whichever is processed first, so DBSCAN isn't fully deterministic at the boundaries.

**21. Why does DBSCAN handle non-convex shapes?**
Connectivity-based, not centroid-based. Two points are in the same cluster if connected through a chain of dense neighborhoods, regardless of overall shape.

> **Saying it out loud.** Because it grows clusters by connectivity, not by distance to a centre. Two points on opposite ends of a long S-curve end up together as long as there's an unbroken chain of dense neighbourhoods between them, even though they're far apart in a straight line. That's exactly what K-means can't express, because K-means can only draw straight boundaries between centres. Concentric circles are the canonical demo: K-means fails completely, DBSCAN gets it instantly.

**22. DBSCAN strengths?**
Arbitrary cluster shapes. Noise detection (outliers explicitly identified). No need to specify $K$.

> **Saying it out loud.** Three things. Arbitrary shapes, because it follows connectivity rather than assuming round blobs. Explicit noise handling, so outliers get labelled rather than forced into a cluster and distorting it. And it works out the number of clusters itself instead of making you specify it. For messy real-world data with genuine outliers, those three together often beat K-means outright.

**23. DBSCAN weaknesses?**
Sensitive to $\varepsilon$. Varying density (single $\varepsilon$ doesn't fit clusters with different densities). Curse of dimensionality (distances become uniform in high-dim).

> **Saying it out loud.** One global radius is the core weakness — it's a single density threshold applied everywhere, so if one cluster is tight and another is diffuse, no single value works and the sparse cluster gets classified as noise. It's also finicky to tune, and it degrades badly in high dimensions where all distances converge and the notion of a dense neighbourhood stops meaning anything. HDBSCAN fixes the varying-density part; nothing fixes high dimensions except reducing them first.

**24. How do you choose $\varepsilon$?**
K-distance plot: for each point, distance to its $k$-th nearest neighbor; sort; plot. Find the "knee" — that's $\varepsilon$. With `min_samples = k`.

> **Saying it out loud.** The k-distance plot. Compute each point's distance to its k-th nearest neighbour, sort those distances, and plot them — the curve is flat where points are in dense regions and bends sharply upward where they aren't. The knee is your epsilon, and you set min_samples to that same k. It's the same eyeball exercise as the elbow method, with the same weakness that the knee isn't always obvious.

**25. What's HDBSCAN?**
Hierarchical DBSCAN. No $\varepsilon$ parameter — computes cluster stability across all density levels. Better for varying-density data. Slower but more robust.

> **Saying it out loud.** HDBSCAN runs DBSCAN at every density level at once and builds a hierarchy, then keeps the clusters that persist across the widest range of levels — the stable ones. That removes the radius parameter entirely and lets different clusters have genuinely different densities, which is the biggest limitation of plain DBSCAN. It's slower and more complex, but for exploratory work on messy data it's the better default.

---

## D. Hierarchical clustering

**26. What's agglomerative clustering?**
Bottom-up: start with each point as own cluster; merge closest pair iteratively until one cluster remains. Produces a dendrogram.

> **Saying it out loud.** Start with every point as its own cluster, repeatedly merge the two closest clusters, and record the whole sequence. What you get is a dendrogram — a tree — and you cut it at whatever height gives the number of clusters you want. So you don't commit to K up front, and often the tree structure itself is the deliverable, as in taxonomies or document organisation.

**27. Linkage criteria?**
Single (min distance, "chaining"), complete (max distance, compact clusters), average, Ward (minimize variance increase). Ward most common in practice.

> **Saying it out loud.** Single linkage measures cluster distance by the closest pair, complete by the farthest pair, average by the mean over pairs, and Ward by how much within-cluster variance the merge would add. Those aren't just formulas, they're assumptions about shape: single follows thin winding structures, complete insists on compact balls, Ward is essentially greedy K-means and is the usual default. Choosing linkage is choosing a bias about what a cluster looks like.

**28. Pros of hierarchical?**
No need to specify $K$ in advance — examine dendrogram. Hierarchy is interpretable.

> **Saying it out loud.** You don't have to pick K in advance — you build the whole tree once and cut it wherever you like, even at several heights to see structure at different granularities. It's also deterministic, unlike K-means, so you get the same answer every run. And in many domains the hierarchy is itself the answer people want, not the flat partition.

**29. Cons?**
$O(N^2)$ memory, $O(N^3)$ naive — limits to $N \sim 10^4$. Greedy: bad early merges propagate.

> **Saying it out loud.** Cost. You need the full pairwise distance matrix, which is N-squared memory, and the naive algorithm is N-cubed time, so you're realistically capped around ten thousand points. Merges are also greedy and irreversible — a bad early merge stays in the tree forever, and there's no repair step. And it's sensitive to noise, especially with single linkage, where one stray point can bridge two clusters.

**30. Single vs complete linkage?**
Single: chaining — produces long thin clusters; sensitive to noise. Complete: compact clusters; can split natural elongated clusters. Ward is usually the default.

> **Saying it out loud.** Single linkage merges on the closest pair, so it can chain — a thin bridge of noise points welds two genuinely separate clusters into one, which is its notorious failure mode. Complete linkage uses the farthest pair, so it demands compactness and will happily chop a long thin cluster in half. Ward sits in between and is usually right, which is why it's the default. If your clusters are elongated, single is right in principle but fragile; if they're blobs, complete or Ward.

---

## E. Spectral clustering

**31. What's spectral clustering?**
Build similarity graph $W$. Compute Laplacian $L = D - W$. Eigendecompose; take bottom $K$ eigenvectors. Cluster the eigenvectors (typically with K-means).

> **Saying it out loud.** You build a graph where nearby points are connected, take its Laplacian — degree matrix minus adjacency — and compute the eigenvectors with the smallest eigenvalues. Those eigenvectors give you new coordinates for each point, and in those coordinates the clusters are round, so you finish with ordinary K-means. So it's really a change of representation followed by a simple clustering, and the eigen-step is a relaxed version of finding the minimum graph cut.

**32. Why does it handle non-convex shapes?**
Operates on graph connectivity, not Euclidean distances directly. Two points are in the same cluster if connected in the similarity graph, regardless of overall shape.

> **Saying it out loud.** Because it groups by connectivity in the graph rather than by straight-line distance. Two points at opposite ends of a spiral are connected through a chain of neighbours, so the graph says they belong together even though Euclidean distance says otherwise. The concentric-circles example is the one to cite: the two rings have the same centre so K-means cannot separate them, but they're disconnected in the neighbourhood graph so spectral separates them trivially.

**33. Cons of spectral clustering?**
$O(N^3)$ eigendecomposition. Hard to scale beyond $N \sim 10^4$. Sensitive to similarity graph construction (kernel choice, k-NN parameter).

> **Saying it out loud.** The eigendecomposition is cubic in the number of points, so past roughly ten thousand you need approximations like Nyström. And the result depends heavily on how you built the similarity graph — the kernel bandwidth or the number of neighbours. Set it too large and the graph is fully connected and structureless, too small and it fragments into disconnected pieces. You also still have to specify K, so it doesn't buy you that.

**34. Spectral vs DBSCAN?**
Both handle non-convex shapes. Spectral: principled (graph theory), needs $K$ specified. DBSCAN: density-based, finds $K$ automatically, more sensitive to parameters.

> **Saying it out loud.** Both handle non-convex shapes, which is the shared selling point. Spectral needs you to specify K and is grounded in graph-cut theory, and it's cleaner when clusters are balanced and the graph is well built. DBSCAN discovers the number of clusters itself and labels outliers, which spectral won't do — spectral forces every point into a cluster. In practice DBSCAN scales better and is the more common choice for messy data; spectral is for when you have modest N and genuine graph structure.

---

## F. Evaluation

**35. Internal evaluation metrics?**
Silhouette ($(b-a)/\max(a,b)$, range $[-1, 1]$). Davies-Bouldin (lower = better). Calinski-Harabasz (between/within variance ratio, higher = better). Use when no ground truth.

> **Saying it out loud.** Internal metrics use only the data and your clustering — no labels. Silhouette is the one to know: for each point, compare its average distance to its own cluster against its average distance to the nearest other cluster, normalise to the range minus one to one, and average. Davies-Bouldin and Calinski-Harabasz are the same instinct with different arithmetic. The caveat to state is that they all reward compact round clusters, so they'll systematically prefer a K-means answer over a correct density-based one on non-convex data.

**36. External evaluation metrics?**
Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), V-measure (homogeneity + completeness). Require ground-truth labels.

> **Saying it out loud.** External metrics compare your clusters against known labels. Adjusted Rand Index counts how often pairs of points are grouped consistently between the two partitions, corrected for chance so a random clustering scores about zero rather than something spuriously positive. Normalised mutual information measures shared information between the partitions. V-measure balances homogeneity and completeness. The word doing the work in ARI is *adjusted* — plain Rand index looks impressively high even for random assignments.

**37. Why is clustering evaluation hard?**
No ground truth in unsupervised setting. Internal metrics reward compactness/separation but may not align with downstream utility. Best: evaluate on a downstream task.

> **Saying it out loud.** Because there is no right answer to compare against — clustering is unsupervised by definition, and different valid clusterings exist depending on what you care about. Group the same customers by geography or by spending and both are correct. Internal metrics measure geometry, not usefulness, so they can rank a bad clustering above a good one. The answer that scores is: evaluate on the downstream task. If the clusters feed a recommender or a routing rule, measure that — a mediocre silhouette that lifts the business metric is the better clustering.

---

## G. Subtleties

**38. Curse of dimensionality for clustering?**
In high-dim, all distances become similar — clusters indistinguishable. K-means converges to weird partitions; DBSCAN density meaningless. Mitigate: dimensionality reduction first, or use domain-aware similarity.

> **Saying it out loud.** In high dimensions everything becomes roughly equidistant from everything else — the gap between the nearest and farthest neighbour shrinks relative to the typical distance — and clustering is built entirely on distance. So K-means gives near-arbitrary partitions, DBSCAN's density threshold stops meaning anything, and your silhouette scores sit near zero regardless. The fix isn't a cleverer algorithm, it's not clustering in the raw space: reduce with PCA or UMAP or a learned embedding down to something like 10 to 50 dimensions, or use a domain-appropriate similarity like cosine for text.

**39. Online clustering?**
For streaming data: incremental K-means (mini-batch), online GMM. Don't store all data; update model as data flows.

> **Saying it out loud.** For streaming data you can't store everything or re-run from scratch, so you update incrementally: mini-batch K-means nudges centres with each arriving batch, and online EM does the same for a mixture. The hard part isn't the update rule, it's drift — the underlying distribution changes and your old clusters stop being right, so you need a way to spawn new clusters and retire dead ones. That's what algorithms like BIRCH and CluStream are for. The failure mode to name is stale centres that quietly stop matching current data.

**40. Soft vs hard clustering?**
Hard: each point in exactly one cluster (K-means, DBSCAN). Soft: each point has a probability per cluster (GMM, fuzzy K-means). Soft is more informative when boundaries are uncertain.

> **Saying it out loud.** Hard clustering commits each point to exactly one cluster — K-means, DBSCAN, hierarchical. Soft clustering gives each point a distribution over clusters, which GMM and fuzzy c-means do. Soft is more honest when a point genuinely sits between two groups, and the responsibility numbers themselves are useful: a point with 50-50 membership is flagging that your model is uncertain there. The cost is more parameters and more computation, and eventually you often have to take the argmax anyway to make a decision.

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
