# Clustering Evaluation — Interview Grill

> 35 questions on internal/external metrics, choosing K, stability. Drill until you can answer 24+ cold.

---

## A. Internal metrics

**1. Silhouette score formula?**
$s(i) = (b - a)/\max(a, b)$. $a$ intra-cluster mean dist, $b$ nearest-other-cluster mean dist.

**2. Silhouette range?**
$[-1, 1]$. Negative = misclassified.

**3. Davies-Bouldin intuition?**
Average over clusters of (spread + spread of nearest other) / distance to nearest other. Lower = better.

**4. Calinski-Harabasz intuition?**
Variance ratio: between-cluster / within-cluster. Higher = better.

**5. Dunn index?**
Min inter-cluster distance / max intra-cluster diameter. Higher = better. Sensitive to outliers.

**6. Why do internal metrics favor globular clusters?**
They reward compactness + separation — the structure K-means produces. Tautological with K-means.

**7. Internal metric range?**
Silhouette: $[-1,1]$. DB: $[0, \infty)$ lower better. CH: $[0, \infty)$ higher better. Dunn: $[0, \infty)$ higher better.

---

## B. External metrics

**8. Adjusted Rand Index range?**
$[-1, 1]$. 1 perfect; 0 chance; negative worse than chance.

**9. ARI core idea?**
Pair-based: fraction of pairs consistently classified (same vs different), corrected for chance.

**10. NMI definition?**
Mutual information / mean entropy. $[0, 1]$.

**11. NMI vs ARI — main difference?**
NMI: doesn't penalize having more / fewer clusters than classes. ARI: pair-based, more sensitive to cardinality.

**12. V-measure components?**
Homogeneity (each cluster = 1 class) + completeness (each class = 1 cluster). Harmonic mean.

**13. Purity formula?**
$\frac{1}{N} \sum_k \max_l |C_k \cap L_l|$. Majority label per cluster.

**14. Purity bias?**
Trivially high with many small clusters. Always check completeness.

**15. Pairwise F-measure?**
Precision/recall over pairs (same cluster, same class). Pairwise version of standard F1.

---

## C. Choosing K

**16. Elbow method?**
Plot WCSS vs $K$. Look for "elbow" (kink). Subjective.

**17. Issue with elbow?**
Often no clear elbow; varies with cluster sizes; subjective.

**18. Silhouette method for K?**
Compute silhouette for various $K$; pick max.

**19. Gap statistic?**
Compare WCSS to expected under uniform reference. Pick $K$ where gap is largest. Statistically grounded; expensive.

**20. Stability-based K selection?**
Bootstrap data, rerun clustering. Pick $K$ with most consistent assignments across bootstraps.

**21. BIC for K (in GMM)?**
Yes — likelihood-based information criterion. Picks $K$ balancing fit and complexity.

**22. Should K equal number of true classes?**
Not necessarily. Classes may not match natural cluster structure.

---

## D. Stability and validation

**23. Bootstrap stability procedure?**
Resample data, rerun clustering, compute ARI between bootstrap and original. High ARI → stable.

**24. Initialization stability?**
Run K-means with different seeds. High variance → init-sensitive solution.

**25. Visualization tools for clustering?**
PCA, t-SNE, UMAP for 2D projection. Visually inspect.

**26. Why does visualization help?**
Catches obvious failures (one giant cluster + many tiny; clusters that aren't separable).

**27. Downstream task validation?**
If clustering serves a use case (segmentation, anomaly), evaluate via that task. The most reliable validation.

---

## E. Common pitfalls

**28. Comparing different algorithms with internal metrics?**
Often unfair — different $K$, different cluster shapes. Watch out.

**29. Internal metric for K-means + silhouette = good?**
Tautological. Both favor compact globular clusters.

**30. Ignoring outliers?**
DBSCAN flags them; K-means absorbs. Affects all metrics differently.

**31. Trusting one run?**
K-means is init-sensitive. Use k-means++ + multiple runs + report best.

**32. Reporting only mean metric?**
Report variance across seeds / bootstraps. Single number misleads.

---

## F. Advanced

**33. Cluster validity in high-dim?**
Curse of dimensionality: distances become uniform. Internal metrics break down. Use dimensionality reduction first.

**34. Soft clustering evaluation?**
Soft (GMM) needs different metrics: NLL on held-out, soft V-measure, etc.

**35. Hierarchical clustering evaluation?**
Cophenetic correlation: correlation between original distances and dendrogram distances. Higher = dendrogram preserves geometry.

---

## Quick fire

**36.** *Silhouette range?* $[-1, 1]$.
**37.** *ARI chance value?* 0.
**38.** *NMI range?* $[0, 1]$.
**39.** *DB lower or higher better?* Lower.
**40.** *CH lower or higher?* Higher.
**41.** *Gap statistic compares to?* Uniform reference.
**42.** *Best K choice strategy?* Multiple methods + downstream validation.
**43.** *V-measure components?* Homogeneity + completeness.
**44.** *Purity bias?* High with many small clusters.
**45.** *Stability test?* Bootstrap + ARI.

---

## Self-grading

If you can't answer 1-15, you don't know clustering metrics. If you can't answer 16-30, you'll struggle on K-selection / validation. If you can't answer 31-45, frontier-lab questions on rigorous unsupervised eval will go past you.

Aim for 28+/45 cold.
