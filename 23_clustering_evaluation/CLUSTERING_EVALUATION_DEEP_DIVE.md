# Clustering Evaluation — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Clustering evaluation is uniquely hard because there's often no ground truth. This deep dive covers internal metrics (don't need labels), external metrics (require labels), choosing $K$, stability analysis, and the practical principle that downstream task performance trumps any clustering metric.

---

## 1. Internal evaluation metrics

Use only the data and the cluster assignments — no labels needed.

### Silhouette score

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

### Davies-Bouldin index

$$
\mathrm{DB} = \frac{1}{K} \sum_i \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
$$

where $\sigma_i$ = average distance of cluster-$i$ points to centroid $c_i$.

Lower = better. Penalizes clusters that are spread out and close to others.

### Calinski-Harabasz (variance ratio)

$$
\mathrm{CH} = \frac{\mathrm{tr}(B)/(K-1)}{\mathrm{tr}(W)/(N-K)}
$$

where $B$ = between-cluster scatter matrix, $W$ = within-cluster scatter. Higher = better.

Like F-statistic for clustering. Strong when clusters are roughly equal-sized and globular.

### Dunn index

$$
\mathrm{Dunn} = \frac{\min_{i \neq j} d(c_i, c_j)}{\max_k \mathrm{diam}(c_k)}
$$

Ratio of minimum inter-cluster distance to maximum intra-cluster diameter. Higher = better. Sensitive to outliers.

### Limitations of internal metrics
- Reward compactness + separation, but those don't always match what the task needs.
- Globular bias: K-means + silhouette favor spherical clusters even when data has elongated shape.
- Can't tell you the "right" number of clusters in any absolute sense.

---

## 2. External evaluation metrics (with ground truth labels)

When you have ground-truth class labels, you can compare cluster assignments to them.

### Adjusted Rand Index (ARI)

Rand Index: fraction of pairs (i, j) that are clustered consistently (both same cluster + same class, or both different):

$$
\mathrm{RI} = \frac{TP + TN}{\binom{N}{2}}
$$

ARI corrects for chance agreement:

$$
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{1 - \mathbb{E}[\mathrm{RI}]}
$$

Range $[-1, 1]$. 1 = perfect; 0 = chance; negative = worse than chance.

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

### V-measure

Harmonic mean of homogeneity and completeness:
- **Homogeneity**: each cluster contains only one class. $h = 1 - H(L|C)/H(L)$.
- **Completeness**: each class is contained in one cluster. $c = 1 - H(C|L)/H(C)$.
- **V-measure**: $V_\beta = (1 + \beta) h c / (\beta h + c)$.

### Purity

$$
\mathrm{Purity} = \frac{1}{N} \sum_k \max_l |C_k \cap L_l|
$$

For each cluster, count majority label. Sum / $N$. Simple but biased toward many small clusters.

### Pairwise F-measure
Compute precision/recall over pairs (do they belong to the same cluster, same class).

---

## 3. Choosing the number of clusters $K$

A famously underdetermined problem.

### Elbow method

Plot WCSS (within-cluster sum of squares) vs $K$. Look for "elbow" where adding clusters stops helping much.

Issues: subjective; often no clear elbow; for very different cluster sizes can mislead.

### Silhouette method

Compute silhouette for various $K$. Pick $K$ with maximum.

More principled than elbow. Doesn't always give a clear answer.

### Gap statistic (Tibshirani et al. 2001)

Compare WCSS to WCSS expected under uniform reference distribution. Pick $K$ where gap is largest.

$$
\mathrm{Gap}(K) = \mathbb{E}[\log W_K^{\mathrm{ref}}] - \log W_K
$$

Statistically grounded. Computationally expensive (need many reference samplings).

### Stability-based

Run clustering on bootstrap subsamples; compare assignments. Stable $K$ → consistent clusters across samples.

### Information criteria
For mixture models (GMM): BIC, AIC.

### Practical answer
- Start with domain knowledge if available.
- Try multiple $K$; visualize.
- Validate downstream task — clustering isn't an end, it's a means.

---

## 4. Stability analysis

Beyond just picking $K$: are clusters meaningful or just an artifact of the algorithm + initialization?

### Bootstrap stability
- Resample data; rerun clustering.
- Measure consistency: ARI between bootstrap clustering and original.
- Stable clusters: high ARI across resamples.

### Initialization stability
- Re-run K-means with different seeds.
- Variance in resulting clusters → solution depends on init.

### Visualization
- Project to 2D (PCA, t-SNE, UMAP).
- Visually inspect cluster structure.

If clusters wildly different across runs / bootstraps, your "clusters" may be noise.

---

## 5. Common pitfalls

### Comparing across algorithms with different $K$
Internal metrics depend on $K$. Different algorithms returning different $K$ can't be fairly compared.

### Using internal metric to choose $K$ for the wrong algorithm
Silhouette favors compact, well-separated clusters. K-means produces those by construction. Selecting $K$ via silhouette + K-means is partly tautological.

### Ignoring outliers
Some clustering methods (DBSCAN) explicitly mark outliers; others (K-means) absorb them. Affects all metrics.

### Forgetting evaluation has hyperparameters
"Cluster quality" metric can favor specific cluster shapes. Match metric to expected cluster geometry.

### Not validating downstream
If clustering is for a downstream task (segmentation, anomaly detection, recommendation), evaluate via that task's success metric — not clustering metrics.

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
