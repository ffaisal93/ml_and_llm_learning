# Clustering Evaluation — Interview Grill

> 35 questions on internal/external metrics, choosing K, stability. Drill until you can answer 24+ cold.

---

## A. Internal metrics

**1. Silhouette score formula?**
$s(i) = (b - a)/\max(a, b)$. $a$ intra-cluster mean dist, $b$ nearest-other-cluster mean dist.

> **Saying it out loud.** For each point you compute two numbers: how far it is on average from its own clustermates, and how far it is on average from the members of the nearest other cluster. Silhouette is the difference between those, divided by whichever is bigger so it stays bounded. Then you average over all points. It's a per-point diagnostic first and a global score second, which is why the per-point plot is more useful than the mean.

**2. Silhouette range?**
$[-1, 1]$. Negative = misclassified.

> **Saying it out loud.** Minus one to one, and the negative half is the part people forget. A negative silhouette means the point is on average closer to a different cluster than to its own, which is a concrete statement that it's assigned wrong. Zero means it's sitting on a boundary. If an interviewer asks the range and you say zero to one, that's an immediate tell that you've only ever read the score off a library and never looked at the formula.

**3. Davies-Bouldin intuition?**
Average over clusters of (spread + spread of nearest other) / distance to nearest other. Lower = better.

> **Saying it out loud.** Davies-Bouldin asks each cluster about its worst neighbor. For every pair you take the sum of how spread out both clusters are, divide by the distance between their centers, and each cluster keeps its worst such ratio; then you average across clusters. Lower is better, which flips the convention from silhouette and is an easy thing to mix up. It's cheap because it only needs centroids and average radii, no pairwise distance matrix.

**4. Calinski-Harabasz intuition?**
Variance ratio: between-cluster / within-cluster. Higher = better.

> **Saying it out loud.** Calinski-Harabasz is the F-statistic idea applied to clustering: variance between clusters divided by variance within them, with degrees-of-freedom corrections for K and N. Higher is better. It's fast, since it only touches centroids rather than all pairs, which makes it the practical choice on large datasets where silhouette's quadratic cost hurts. The known weakness is that it often increases with K, so it can nudge you toward too many clusters.

**5. Dunn index?**
Min inter-cluster distance / max intra-cluster diameter. Higher = better. Sensitive to outliers.

> **Saying it out loud.** Dunn is the worst-case-over-worst-case metric: the smallest distance between any two clusters divided by the largest diameter of any single cluster. High Dunn means even your tightest pair is well separated relative to your baggiest cluster. Because it's built from a min over a max, one outlier can dominate the whole score. That fragility is why you see it in textbooks more than in practice.

**6. Why do internal metrics favor globular clusters?**
They reward compactness + separation — the structure K-means produces. Tautological with K-means.

> **Saying it out loud.** Because compactness and separation are the definition these metrics use, and compact separated blobs are exactly what k-means produces regardless of what the data looks like. So scoring a k-means result with silhouette partly measures how k-means-shaped your k-means output is. On two interleaved crescents, the correct clustering scores badly and a wrong spherical split scores well. The fix is to validate with stability or a downstream task, whose assumptions don't line up with your algorithm's.

**7. Internal metric range?**
Silhouette: $[-1,1]$. DB: $[0, \infty)$ lower better. CH: $[0, \infty)$ higher better. Dunn: $[0, \infty)$ higher better.

> **Saying it out loud.** Silhouette runs minus one to one with higher better. Davies-Bouldin is zero upward with lower better. Calinski-Harabasz is zero upward with higher better and unbounded, so its absolute value means nothing, only comparisons at fixed data. Dunn is zero upward with higher better. The two that trip people up are silhouette's negative half and Davies-Bouldin's inverted direction.

---

## B. External metrics

**8. Adjusted Rand Index range?**
$[-1, 1]$. 1 perfect; 0 chance; negative worse than chance.

> **Saying it out loud.** Minus one to one. One is a perfect match with the ground truth, zero is what you'd expect from random assignment, and negative means you've done worse than chance, which usually indicates something systematically inverted. The chance correction is the whole point, because unadjusted Rand sits around seven- or eight-tenths for random clusterings and looks deceptively good.

**9. ARI core idea?**
Pair-based: fraction of pairs consistently classified (same vs different), corrected for chance.

> **Saying it out loud.** ARI counts pairs, not points, which neatly sidesteps the fact that cluster labels are arbitrary. For every pair of points you ask whether your clustering and the ground truth agree about whether they belong together, then you subtract off the agreement you'd expect by chance and rescale so perfect is one. That's why you don't need to solve a matching problem between cluster IDs and class IDs. Always report the adjusted version, since raw Rand is inflated by all the trivially-different pairs.

**10. NMI definition?**
Mutual information / mean entropy. $[0, 1]$.

> **Saying it out loud.** NMI is the mutual information between your cluster assignment and the true labels, divided by something like the average of the two entropies to bring it into zero-to-one. Intuitively it's how many bits knowing the cluster tells you about the class. It's symmetric, so it makes no assumption about which side is truth. The thing to add is that plain NMI creeps upward as you add clusters, so use adjusted mutual information when comparing across different K.

**11. NMI vs ARI — main difference?**
NMI: doesn't penalize having more / fewer clusters than classes. ARI: pair-based, more sensitive to cardinality.

> **Saying it out loud.** The practical difference is how they respond to a mismatch in the number of clusters. NMI doesn't really penalize you for splitting one true class into three clusters, so it drifts up with K, while ARI's pair counting makes over-splitting visibly costly. So if you're comparing methods that produced different numbers of clusters, ARI or adjusted mutual information is the fairer choice. They also disagree on skewed cluster sizes, with NMI more forgiving of tiny clusters.

**12. V-measure components?**
Homogeneity (each cluster = 1 class) + completeness (each class = 1 cluster). Harmonic mean.

> **Saying it out loud.** V-measure splits into two halves that fight each other. Homogeneity asks whether each cluster contains only one class, which you can trivially max out by giving every point its own cluster. Completeness asks whether each class is kept together in one cluster, which you max out by putting everything in one cluster. V-measure is their harmonic mean, so it's the F1 of clustering, and reporting the two halves separately tells you far more than the combined number.

**13. Purity formula?**
$\frac{1}{N} \sum_k \max_l |C_k \cap L_l|$. Majority label per cluster.

> **Saying it out loud.** For each cluster you find the most common true label and count how many points carry it, add those counts up, and divide by N. It's the fraction of points that would be correct if you labeled every cluster by its majority class. It's the easiest metric to explain to a non-technical stakeholder. And it's the easiest to game, which is the next question.

**14. Purity bias?**
Trivially high with many small clusters. Always check completeness.

> **Saying it out loud.** Purity goes to one if you make every point its own cluster, so it rewards over-splitting without limit. That means a high purity number alone tells you nothing. Always pair it with completeness, or with the number of clusters, or just use ARI, which has the chance correction built in. If someone reports ninety-five percent purity, the first question is how many clusters they used.

**15. Pairwise F-measure?**
Precision/recall over pairs (same cluster, same class). Pairwise version of standard F1.

> **Saying it out loud.** Pairwise F-measure recasts pair agreement as precision and recall. Precision is, of the pairs you grouped together, what fraction really belong together; recall is, of the pairs that truly belong together, what fraction did you group. Then harmonic-mean them. The advantage over a single ARI number is diagnostic: low precision means you're over-merging, low recall means you're over-splitting, and knowing which one tells you how to adjust K.

---

## C. Choosing K

**16. Elbow method?**
Plot WCSS vs $K$. Look for "elbow" (kink). Subjective.

> **Saying it out loud.** Plot within-cluster sum of squares against K and look for the kink where the curve stops dropping steeply. The logic is that up to the true K each new cluster genuinely splits something, and after that you're just cutting real clusters in half. It's the most commonly taught method and the least reliable one, because the curve is monotone and you're eyeballing a second derivative.

**17. Issue with elbow?**
Often no clear elbow; varies with cluster sizes; subjective.

> **Saying it out loud.** Two problems. Often there is no clear elbow at all, just a smooth curve, and two people will pick different K from the same plot. And when your true clusters differ a lot in size or density, the curve is dominated by the big ones and the bend gets smeared out. It's a reasonable first look and a bad final justification. If you cite the elbow in an interview, immediately follow with a second method.

**18. Silhouette method for K?**
Compute silhouette for various $K$; pick max.

> **Saying it out loud.** Compute silhouette across a range of K and take the maximum. It's better than the elbow because you're maximizing an actual quantity instead of eyeballing a bend. Two caveats: silhouette bakes in a preference for round clusters, so it's partly tautological with k-means, and it needs a pairwise distance matrix, which is quadratic and slow past tens of thousands of points.

**19. Gap statistic?**
Compare WCSS to expected under uniform reference. Pick $K$ where gap is largest. Statistically grounded; expensive.

> **Saying it out loud.** The gap statistic compares your within-cluster scatter at each K against what you'd get from random uniform data with the same overall extent, and you pick the K where your data beats the reference by the most. What makes it special is the built-in null, so it can actually return K equals one, meaning there's no cluster structure at all. Almost no other method can say that. The cost is compute: you're clustering many reference datasets at every K, so it's expensive.

**20. Stability-based K selection?**
Bootstrap data, rerun clustering. Pick $K$ with most consistent assignments across bootstraps.

> **Saying it out loud.** Stability selection says a real K should give you the same partition on slightly different data. You bootstrap or subsample, rerun the clustering, and measure agreement with ARI, then pick the K whose agreement is highest across resamples. It's testing the thing you actually care about, which is whether the structure is a property of the population or an artifact of this sample. Watch out for the degenerate case: K equals two is often trivially the most stable, so you compare against a null.

**21. BIC for K (in GMM)?**
Yes — likelihood-based information criterion. Picks $K$ balancing fit and complexity.

> **Saying it out loud.** Yes, for mixture models you have a real likelihood, so BIC works: log-likelihood minus a penalty proportional to the number of parameters times log N. That turns choosing K into ordinary model selection instead of a heuristic. BIC penalizes complexity harder than AIC, so it picks fewer components and is the usual choice here. The requirement is that the model be roughly right; if your data isn't Gaussian, BIC will happily pile on components to paper over the mismatch.

**22. Should K equal number of true classes?**
Not necessarily. Classes may not match natural cluster structure.

> **Saying it out loud.** Not necessarily, and this comes up more than people expect. Your labels reflect what a human decided to categorize by, and the natural geometric structure of the data may split along something entirely different: photographs might cluster by lighting rather than by object class. So a clustering that disagrees with your labels might be finding something real that your labels don't encode. Before concluding the clustering failed, look at what the clusters actually correspond to.

---

## D. Stability and validation

**23. Bootstrap stability procedure?**
Resample data, rerun clustering, compute ARI between bootstrap and original. High ARI → stable.

> **Saying it out loud.** Resample the dataset with replacement, cluster the resample, and compute ARI between the new assignment and the original one on the points they share. Repeat that thirty or a hundred times and look at the whole distribution, not just the mean. Consistently high agreement, roughly above eight-tenths, means the structure is real; scores scattered around the middle mean you're clustering noise. This is the single most useful sanity check in unsupervised work.

**24. Initialization stability?**
Run K-means with different seeds. High variance → init-sensitive solution.

> **Saying it out loud.** Rerun k-means with different random seeds on the same fixed data and see how much the partition changes. K-means minimizes a non-convex objective, so different starting points land in different local minima; that's why k-means++ initialization and best-of-ten restarts are library defaults. If the clusters move a lot across seeds with the data held constant, you're looking at an optimization artifact, not structure. Report the variance across seeds, not just the best run.

**25. Visualization tools for clustering?**
PCA, t-SNE, UMAP for 2D projection. Visually inspect.

> **Saying it out loud.** PCA, t-SNE, or UMAP down to two dimensions, colored by cluster assignment. PCA is the honest one because it's a linear projection you can reason about, while t-SNE and UMAP produce prettier pictures that manufacture apparent separation. So use PCA to check whether clusters are genuinely separated and UMAP to explore. And a common workflow is PCA down to fifty dimensions first, then UMAP to two.

**26. Why does visualization help?**
Catches obvious failures (one giant cluster + many tiny; clusters that aren't separable).

> **Saying it out loud.** Because your eyes catch failure modes no scalar metric surfaces. One giant cluster plus a scatter of tiny ones, clusters that are completely interleaved, an outlier sitting alone as its own cluster: all of those can coexist with a respectable silhouette score. Thirty seconds of looking saves hours. Just be careful about the direction of inference, since visual separation in UMAP is weak evidence for structure but visual interleaving is strong evidence against it.

**27. Downstream task validation?**
If clustering serves a use case (segmentation, anomaly), evaluate via that task. The most reliable validation.

> **Saying it out loud.** If the clustering exists to serve something, that something is the evaluation. Customer segments get judged by whether campaigns aimed at them convert better, anomaly clusters by whether the flagged items were genuinely bad, compression clusters by downstream accuracy. A segmentation with mediocre silhouette that lifts conversion beats a gorgeous silhouette nobody can act on. This is the most reliable validation because it's the only one measuring the thing you actually want.

---

## E. Common pitfalls

**28. Comparing different algorithms with internal metrics?**
Often unfair — different $K$, different cluster shapes. Watch out.

> **Saying it out loud.** It's usually unfair, because internal metrics are biased by the number of clusters and by cluster shape. DBSCAN returning six clusters and marking outliers as noise cannot be compared to k-means returning three by reading off two silhouette scores; you'd be comparing K effects and outlier handling, not quality. Either hold K fixed and compare, or compare through a downstream task. If you must compare directly, at least score both on the same set of points.

**29. Internal metric for K-means + silhouette = good?**
Tautological. Both favor compact globular clusters.

> **Saying it out loud.** It's circular. Silhouette rewards compact well-separated round clusters, and that's the geometry k-means imposes whether or not the data has it. So you're largely measuring how successfully k-means did what k-means does. Any K you pick this way inherits that bias. Use a metric with different assumptions, or validate by stability or a downstream task instead.

**30. Ignoring outliers?**
DBSCAN flags them; K-means absorbs. Affects all metrics differently.

> **Saying it out loud.** The algorithms treat outliers so differently that it silently corrupts comparisons. DBSCAN labels them noise and drops them; k-means has to assign every point somewhere, so outliers get absorbed and drag centroids around. Score both with silhouette and DBSCAN looks better partly because it was allowed to discard the hard cases. Decide your noise policy first and evaluate both methods on the same point set.

**31. Trusting one run?**
K-means is init-sensitive. Use k-means++ + multiple runs + report best.

> **Saying it out loud.** K-means has a non-convex objective and a random start, so a single run is a sample from a distribution of local minima, not the answer. Use k-means++ initialization, which spreads out the initial centers, run it ten or more times, and keep the lowest inertia. That's what scikit-learn does by default with n_init. Trusting one run is how you end up presenting a clustering nobody can reproduce.

**32. Reporting only mean metric?**
Report variance across seeds / bootstraps. Single number misleads.

> **Saying it out loud.** A single mean hides whether the result is reliable. Report the spread across seeds and across bootstrap resamples, because a silhouette of point-five with a standard deviation of point-zero-two is a completely different claim than point-five with a standard deviation of point-two. The second one means your clustering is unstable and the number is meaningless. Error bars on unsupervised results are rare in practice and they're a strong signal of rigor in an interview.

---

## F. Advanced

**33. Cluster validity in high-dim?**
Curse of dimensionality: distances become uniform. Internal metrics break down. Use dimensionality reduction first.

> **Saying it out loud.** In high dimensions distances concentrate: the nearest and farthest points end up nearly equidistant, so any metric built on distance ratios stops discriminating. Silhouette and Dunn both quietly degrade toward uninformative values. The standard remedy is to reduce dimensions first, PCA to fifty or so, or use a learned embedding, and cluster there. Also consider cosine distance, which holds up better than Euclidean on high-dimensional sparse data like text.

**34. Soft clustering evaluation?**
Soft (GMM) needs different metrics: NLL on held-out, soft V-measure, etc.

> **Saying it out loud.** Soft clustering gives you a probability per cluster per point, so a hard-assignment metric throws away exactly the information you paid for. Held-out log-likelihood is the natural metric for a Gaussian mixture, since it's a real probabilistic model and you can just score unseen data. Beyond that there are soft versions of V-measure and NMI that use the responsibilities. And the practical extra check is calibration: are the points the model says it's unsure about actually the ambiguous ones?

**35. Hierarchical clustering evaluation?**
Cophenetic correlation: correlation between original distances and dendrogram distances. Higher = dendrogram preserves geometry.

> **Saying it out loud.** For hierarchical clustering the object you're evaluating is the whole dendrogram, not a single partition. Cophenetic correlation measures how well it preserves the original geometry: for each pair of points, take the height at which they first get merged, and correlate those merge heights against the original pairwise distances. High correlation means the tree faithfully represents the distance structure. It's also the standard way to compare linkage criteria, and average linkage usually wins on cophenetic correlation while Ward usually wins on producing clusters people find useful.

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
