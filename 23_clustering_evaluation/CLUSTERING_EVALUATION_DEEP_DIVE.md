# Clustering Evaluation — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

Clustering evaluation is uniquely hard because there's often no ground truth. This deep dive covers internal metrics (don't need labels), external metrics (require labels), choosing $K$, stability analysis, and the practical principle that downstream task performance trumps any clustering metric.

---

## 1. Internal evaluation metrics

Use only the data and the cluster assignments — no labels needed.

### Silhouette score

> **In plain language.** Silhouette answers one question per point: am I in the right cluster? It compares how far you are from your own clustermates against how far you are from the nearest rival cluster. If your own cluster is closer, the score is positive; if the rival is closer, it goes negative.

For each point $i$:
- $a(i)$ = mean intra-cluster distance.
- $b(i)$ = mean distance to nearest *other* cluster.
- $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$.

Range $[-1, 1]$. Mean over all points = silhouette score.

**Interpretation**:
- $\sim 1$: well-separated clusters.
- $\sim 0$: ambiguous, points on cluster boundaries.
- $< 0$: misclassified (closer to other cluster than own).

**When**: convex, well-separated clusters; meaningful Euclidean distances.

> **Saying it out loud.** Silhouette asks, for every single point, whether it looks like it's in the right cluster. You measure the average distance to your own clustermates and the average distance to the members of the nearest other cluster, and you take the difference, normalized so it lands between minus one and one. Near one means the clusters are cleanly separated, near zero means points are sitting on the boundary, and negative literally means the point is closer to a different cluster than to its own. The bit people get wrong in interviews is the range: it's minus one to one, not zero to one.

### Davies-Bouldin index

$$
\mathrm{DB} = \frac{1}{K} \sum_i \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
$$

where $\sigma_i$ = average distance of cluster-$i$ points to centroid $c_i$.

Lower = better. Penalizes clusters that are spread out and close to others.

> **Saying it out loud.** Davies-Bouldin asks, for each cluster, who's its worst neighbor. For every pair you take how spread out the two clusters are, divided by how far apart their centers are, and each cluster gets scored by its worst partner, then you average. Lower is better, which is the opposite convention from silhouette and an easy thing to fumble. Its bias is toward round clusters with clear centroids, so it agrees with k-means almost by construction.

### Calinski-Harabasz (variance ratio)

$$
\mathrm{CH} = \frac{\mathrm{tr}(B)/(K-1)}{\mathrm{tr}(W)/(N-K)}
$$

where $B$ = between-cluster scatter matrix, $W$ = within-cluster scatter. Higher = better.

Like F-statistic for clustering. Strong when clusters are roughly equal-sized and globular.

> **Saying it out loud.** Calinski-Harabasz is essentially an F-statistic for clustering: how much variance sits between clusters versus how much sits inside them, corrected for the number of clusters and points. Higher is better. It's fast because it only needs centroids, no pairwise distances, so it scales far better than silhouette. Its weakness is that it tends to keep climbing as you add clusters on many datasets, so it can push you toward a larger K than is meaningful.

### Dunn index

$$
\mathrm{Dunn} = \frac{\min_{i \neq j} d(c_i, c_j)}{\max_k \mathrm{diam}(c_k)}
$$

Ratio of minimum inter-cluster distance to maximum intra-cluster diameter. Higher = better. Sensitive to outliers.

> **Saying it out loud.** Dunn takes the worst case on both ends: the smallest gap between any two clusters divided by the largest diameter of any single cluster. High Dunn means even your closest clusters are far apart relative to your widest cluster. Because it's built from minima and maxima rather than averages, it's extremely sensitive to outliers, and one stray point can crater the score. That fragility is why it's less used in practice than silhouette.

### Limitations of internal metrics
- Reward compactness + separation, but those don't always match what the task needs.
- Globular bias: K-means + silhouette favor spherical clusters even when data has elongated shape.
- Can't tell you the "right" number of clusters in any absolute sense.

> **Saying it out loud.** The catch with all internal metrics is that they define good in terms of geometry, specifically compact and well-separated, and your task might not care about that. If your true clusters are long and curved, like two interleaved crescents, silhouette will tell you they're terrible even though they're exactly right. And no internal metric can tell you the correct number of clusters in any absolute sense, because that's not a well-posed question without a notion of purpose. The honest summary: internal metrics measure whether your clusters look like the metric's idea of clusters.

---

## 2. External evaluation metrics (with ground truth labels)

When you have ground-truth class labels, you can compare cluster assignments to them.

### Adjusted Rand Index (ARI)

> **In plain language.** The Rand index counts pairs of points instead of looking at individual labels. For every pair, ask whether your clustering and the ground truth agree on whether those two belong together. The adjusted version subtracts off how much agreement you'd get by pure chance, so random clustering scores zero instead of something misleadingly high.

Rand Index: fraction of pairs (i, j) that are clustered consistently (both same cluster + same class, or both different):

$$
\mathrm{RI} = \frac{TP + TN}{\binom{N}{2}}
$$

ARI corrects for chance agreement:

$$
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{1 - \mathbb{E}[\mathrm{RI}]}
$$

Range $[-1, 1]$. 1 = perfect; 0 = chance; negative = worse than chance.

> **Saying it out loud.** The Rand index counts pairs: take every pair of points and ask whether your clustering and the true labels agree about whether they belong together. The problem is that raw Rand looks great even for random assignments, because most pairs are in different clusters no matter what. The adjusted version subtracts the expected agreement under chance and rescales, so random clustering scores zero and perfect scores one, and you can go negative if you do worse than random. Always report ARI, not RI, because raw RI over zero-point-seven can be pure noise.

### Normalized Mutual Information (NMI)

Mutual information between clustering $C$ and labels $L$:

$$
I(C; L) = \sum_{c, l} P(c, l) \log \frac{P(c, l)}{P(c) P(l)}
$$

Normalized:

$$
\mathrm{NMI}(C, L) = \frac{2 I(C; L)}{H(C) + H(L)}
$$

Range $[0, 1]$. Symmetric. Doesn't penalize having more or fewer clusters than classes.

> **Saying it out loud.** NMI treats the clustering and the labels as two random variables and asks how much knowing one tells you about the other, normalized by their entropies so the answer lands between zero and one. It's symmetric, so it doesn't care which one you call the truth. The key behavioral difference from ARI is that plain NMI doesn't punish you for splitting classes into extra clusters, so it drifts upward as K increases. If you're comparing across different numbers of clusters, use adjusted mutual information or ARI, not raw NMI.

### V-measure

Harmonic mean of homogeneity and completeness:
- **Homogeneity**: each cluster contains only one class. $h = 1 - H(L|C)/H(L)$.
- **Completeness**: each class is contained in one cluster. $c = 1 - H(C|L)/H(C)$.
- **V-measure**: $V_\beta = (1 + \beta) h c / (\beta h + c)$.

> **Saying it out loud.** V-measure splits cluster quality into two things that pull against each other. Homogeneity asks whether each cluster contains only one true class, which you can max out by making every point its own cluster. Completeness asks whether each class stays inside one cluster, which you max out by lumping everything into one cluster. V-measure is the harmonic mean, so it's the F1 of clustering, and reporting both halves separately is far more informative than reporting the combined number.

### Purity

$$
\mathrm{Purity} = \frac{1}{N} \sum_k \max_l |C_k \cap L_l|
$$

For each cluster, count majority label. Sum / $N$. Simple but biased toward many small clusters.

> **Saying it out loud.** Purity is the simple one: for each cluster, look at its most common true label and count how many points carry it, then sum over clusters and divide by N. Easy to explain to a stakeholder. The fatal flaw is that it's trivially maximized by making every point its own cluster, which gives you a purity of exactly one and zero information. So purity alone is never an answer; you need completeness or a K penalty alongside it.

### Pairwise F-measure
Compute precision/recall over pairs (do they belong to the same cluster, same class).

> **Saying it out loud.** Pairwise F-measure is the same pair-counting idea as Rand, but framed as precision and recall. Precision is the fraction of pairs you put together that really belong together; recall is the fraction of pairs that belong together which you actually grouped. Then take the harmonic mean. The reason to prefer it over ARI sometimes is that the precision and recall halves tell you which direction you're failing in, whether you're over-splitting or over-merging, which a single ARI number hides.

---

## 3. Choosing the number of clusters $K$

A famously underdetermined problem.

### Elbow method

Plot WCSS (within-cluster sum of squares) vs $K$. Look for "elbow" where adding clusters stops helping much.

Issues: subjective; often no clear elbow; for very different cluster sizes can mislead.

> **Saying it out loud.** The elbow method plots within-cluster sum of squares against K and looks for the bend where adding clusters stops paying off. The reason it's shaky is that the curve always decreases, so you're eyeballing a rate of change, and on real data there's often no clear bend at all. Two people look at the same plot and pick different K. It's fine as a first look and it should never be your only justification.

### Silhouette method

Compute silhouette for various $K$. Pick $K$ with maximum.

More principled than elbow. Doesn't always give a clear answer.

> **Saying it out loud.** The silhouette method just computes the silhouette score across a range of K and takes the peak. It's more principled than the elbow because you're maximizing an actual quantity rather than eyeballing a curve. But it still frequently gives a flat, ambiguous profile, and it carries silhouette's built-in preference for round clusters. There's also a cost issue: silhouette needs pairwise distances, so it's quadratic and gets slow past tens of thousands of points.

### Gap statistic (Tibshirani et al. 2001)

Compare WCSS to WCSS expected under uniform reference distribution. Pick $K$ where gap is largest.

$$
\mathrm{Gap}(K) = \mathbb{E}[\log W_K^{\mathrm{ref}}] - \log W_K
$$

Statistically grounded. Computationally expensive (need many reference samplings).

> **Saying it out loud.** The gap statistic is the statistically honest version: compare your within-cluster scatter at each K against what you'd get on random uniform data with the same bounding shape, and pick the K where the gap between them is largest. What makes it better than the elbow is that it has a built-in null hypothesis, so it can actually tell you that K equals one, meaning there's no cluster structure at all. Very few methods can say that. The cost is compute, since you need many reference datasets clustered at every K.

### Stability-based

Run clustering on bootstrap subsamples; compare assignments. Stable $K$ → consistent clusters across samples.

> **Saying it out loud.** Stability selection says a real cluster structure should survive perturbation. You bootstrap or subsample your data, rerun clustering, and measure how similar the assignments are, typically with ARI. If K equals four gives you consistent partitions across resamples and K equals seven gives you a different answer every time, that's evidence for four. It's my favorite practical method because it directly tests the thing you care about, which is whether the structure is real rather than an artifact of one particular sample.

### Information criteria
For mixture models (GMM): BIC, AIC.

> **Saying it out loud.** For model-based clustering like a Gaussian mixture you have an actual likelihood, so you can use BIC or AIC and let the penalty term handle the number of components. BIC penalizes parameters more heavily, so it tends to pick fewer, more conservative components, while AIC leans toward more. That's a genuine advantage of mixture models over k-means: model selection becomes a standard statistical problem instead of a heuristic. The caveat is that the criterion is only valid if the Gaussian assumption roughly holds.

### Practical answer
- Start with domain knowledge if available.
- Try multiple $K$; visualize.
- Validate downstream task — clustering isn't an end, it's a means.

> **Saying it out loud.** The real answer is that K is usually decided by what you're going to do with the clusters. If marketing can run four campaigns, K is four, and no gap statistic overrules that. So use domain knowledge first, try a range and actually look at the clusters, and validate against the downstream task whenever there is one. Clustering is almost never the deliverable, it's a step, so the metric that matters lives one step later.

---

## 4. Stability analysis

Beyond just picking $K$: are clusters meaningful or just an artifact of the algorithm + initialization?

### Bootstrap stability
- Resample data; rerun clustering.
- Measure consistency: ARI between bootstrap clustering and original.
- Stable clusters: high ARI across resamples.

> **Saying it out loud.** Bootstrap stability is the cleanest sanity check for whether your clusters exist. Resample the data with replacement, cluster it again, and compare the new assignments to the original with ARI on the shared points. Repeat that dozens of times and look at the distribution. Consistently high ARI, say above eight-tenths, means the structure is robust; scores bouncing around the middle mean you're clustering noise and shouldn't present it.

### Initialization stability
- Re-run K-means with different seeds.
- Variance in resulting clusters → solution depends on init.

> **Saying it out loud.** Initialization stability is the narrower version, aimed at the algorithm rather than the data. Rerun k-means with different random seeds on the same data and see how much the partition moves. K-means minimizes a non-convex objective, so different starts land in different local minima, which is exactly why k-means++ initialization and best-of-ten-restarts are the defaults. If your clusters change substantially across seeds on fixed data, you don't have clusters, you have an optimization artifact.

### Visualization
- Project to 2D (PCA, t-SNE, UMAP).
- Visually inspect cluster structure.

If clusters wildly different across runs / bootstraps, your "clusters" may be noise.

> **Saying it out loud.** The cheapest check of all is to look at the thing. Project down to two dimensions with PCA or UMAP, color by cluster assignment, and see whether it looks like anything. If the colors are interleaved confetti, no metric is going to save you. Just remember the direction of inference: UMAP separation is weak evidence for real clusters since UMAP manufactures visual separation, but interleaving is strong evidence against. And the summary line is that clusters which change wildly across runs and resamples are noise, no matter how good the silhouette score looks.

---

## 5. Common pitfalls

### Comparing across algorithms with different $K$
Internal metrics depend on $K$. Different algorithms returning different $K$ can't be fairly compared.

> **Saying it out loud.** You can't compare internal metric values across algorithms that returned different numbers of clusters, because these metrics are systematically biased by K. Calinski-Harabasz tends to reward more clusters, purity definitely does, and silhouette has its own preferences. So DBSCAN finding six clusters and k-means finding three cannot be compared by reading off two silhouette scores. Either fix K and compare, or compare through a downstream task that doesn't care how many clusters you used.

### Using internal metric to choose $K$ for the wrong algorithm
Silhouette favors compact, well-separated clusters. K-means produces those by construction. Selecting $K$ via silhouette + K-means is partly tautological.

> **Saying it out loud.** Choosing K by maximizing silhouette while clustering with k-means is close to circular reasoning. Silhouette rewards compact, well-separated, roughly spherical clusters, and that's precisely the geometry k-means produces no matter what the data looks like. So you're partly measuring how well k-means did k-means. The fix is to evaluate with a metric whose assumptions differ from your algorithm's, or better, to validate with stability or a downstream task instead.

### Ignoring outliers
Some clustering methods (DBSCAN) explicitly mark outliers; others (K-means) absorb them. Affects all metrics.

> **Saying it out loud.** Outliers get handled completely differently across algorithms and that quietly wrecks metric comparisons. DBSCAN labels them as noise and excludes them; k-means has to assign every point to something, so outliers get absorbed and drag centroids around. If you then compute silhouette on both, DBSCAN looks better partly because it was allowed to throw away the hard points. Decide up front how you're treating noise, and score both methods on the same set of points.

### Forgetting evaluation has hyperparameters
"Cluster quality" metric can favor specific cluster shapes. Match metric to expected cluster geometry.

> **Saying it out loud.** It's easy to forget that your evaluation metric encodes assumptions just as much as your algorithm does. Silhouette and Davies-Bouldin assume compact round clusters, density-based validity indices assume something else entirely, and the choice of distance function is a modeling decision hiding inside the metric. Pick a metric that matches the cluster geometry you actually expect. Otherwise you'll rank a correct clustering below a wrong one, and the metric will look authoritative while doing it.

### Not validating downstream
If clustering is for a downstream task (segmentation, anomaly detection, recommendation), evaluate via that task's success metric — not clustering metrics.

> **Saying it out loud.** The principle that overrides everything above: if the clustering feeds a downstream task, evaluate on that task. Customer segments should be judged by whether campaigns targeted at them convert better, anomaly clusters by whether the flagged items were actually bad. A segmentation with a mediocre silhouette that lifts conversion is better than a beautiful silhouette nobody can act on. Clustering is a means, so the metric that decides is one layer downstream.

---

## 6. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Silhouette range? | $[0, 1]$ | $[-1, 1]$ — negative when cluster assignment is wrong |
| ARI vs NMI? | Same | ARI: pair-based, corrects for chance. NMI: information-theoretic, doesn't penalize K mismatch |
| How to choose $K$? | Elbow | Multiple methods; ultimately downstream-task validation |
| Internal metric guarantees clustering quality? | Yes | No — favors specific geometries; matches algorithm bias |
| Purity seems high — done? | Yes | Trivially high with many small clusters; check completeness too |
| ARI = 0.5 — good? | Yes | Depends; far from random (0) but far from perfect (1); context matters |
| Run K-means once and trust? | Sure | Init-sensitive; use k-means++ + multiple runs + best-of |

> **Saying it out loud.** The gotchas that catch people most: silhouette runs from minus one to one, not zero to one, and negative means the point is in the wrong cluster. ARI is pair-based and chance-corrected while NMI is information-theoretic and doesn't punish K mismatch. High purity means nothing on its own because more clusters always raise it. And never trust a single k-means run, since the objective is non-convex, so use k-means++ with multiple restarts and keep the best.

---

## 7. Eight most-asked interview questions

1. **What metrics evaluate clustering without labels?** (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn.)
2. **What metrics need labels?** (ARI, NMI, V-measure, purity, pairwise F.)
3. **Why is choosing $K$ hard?** (Underdetermined; no objective best; methods give different answers.)
4. **Walk me through silhouette.** ($a$ vs $b$; range $[-1, 1]$; meaningful for compact globular clusters.)
5. **ARI vs NMI?** (ARI pair-based with chance correction; NMI info-theoretic; different intuitions.)
6. **Why does internal metric favor K-means style clusters?** (Both reward compactness + separation; tautological.)
7. **How would you sanity-check a clustering result?** (Visualization, bootstrap stability, downstream task validation.)
8. **When clustering doesn't match labels, what's wrong?** (Could be: labels noisy, clustering uses different similarity, labels don't reflect natural clusters.)

---

## 8. Drill plan

- For each internal metric, recite formula + when it works.
- For each external metric, recite formula + interpretation.
- Recite 3 methods to choose $K$ + their failure modes.
- Sketch how bootstrap stability validates a clustering.
- Practice 2 cases: customer segmentation, image clustering — describe full evaluation strategy.

---

## 9. Further reading

- Halkidi, Batistakis, Vazirgiannis (2001), *On Clustering Validation Techniques.*
- Vinh, Epps, Bailey (2010), *Information Theoretic Measures for Clusterings Comparison.* — NMI variants.
- Tibshirani, Walther, Hastie (2001), *Estimating the number of clusters in a data set via the gap statistic.*
- Hubert & Arabie (1985), *Comparing partitions* — original ARI paper.
