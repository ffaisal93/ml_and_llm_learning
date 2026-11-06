# Clustering and retrieval depth

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: choose geometry, supervision and evaluation together.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f141"></a>

## F141. Choose the number of k-means clusters.

**Start here.** There is rarely one objectively correct k; the useful resolution depends on geometry and the downstream task.

**Work it through.**

1. Compare objective curves across k, using several initialization seeds.
2. Inspect cluster stability, silhouette behavior, sizes and representative examples, and standardize features if appropriate.
3. Validate usefulness for the real decision; do not choose a visually neat elbow without checking whether clusters are meaningful.

**Small example.** Increasing k always weakly improves the optimal within-cluster SSE, so lowest training SSE alone selects overly large k.

**Common trap.** Treating cluster labels as natural ground-truth classes.

**Follow-up.** What if clusters have different densities or non-spherical shapes? Consider other methods rather than forcing k-means geometry.

---

<a id="f142"></a>

## F142. Implement the EM steps of a Gaussian mixture model.

**Start here.** EM alternates estimating soft cluster membership and fitting weighted Gaussian components.

**Work it through.**

1. In the E step, compute log component densities plus log mixture weights and normalize with log-sum-exp.
2. In the M step, update component weights, weighted means and covariances from responsibilities.
3. Regularize covariance matrices, handle nearly empty components and stop using likelihood changes plus a maximum iteration cap.

**Small example.** A point between two overlapping components can have responsibilities [0.4,0.6] instead of one hard assignment.

**Common trap.** Allowing a covariance to collapse to zero around one point, making likelihood singular.

**Follow-up.** How does this differ from k-means? Membership is soft and covariance geometry can vary, depending on the chosen GMM family.

---

<a id="f143"></a>

## F143. Cluster image patches rather than full images.

**Start here.** Each patch is an observation whose representation and position must be defined explicitly.

**Work it through.**

1. Extract non-overlapping or overlapping patches under a stated boundary policy.
2. Flatten raw intensities or compute normalized descriptors, fit clustering on training patches and retain the index-to-image-position map.
3. Reconstruct or visualize cluster assignments to inspect what the representation actually grouped.

**Small example.** A 16-by-16 grayscale image split into 4-by-4 patches gives 16 observations with 16 intensity features each.

**Common trap.** Mixing patch identity with pixel order or clustering brightness differences when texture was the desired signal.

**Follow-up.** Why might descriptors outperform raw pixels? They can reduce sensitivity to nuisance variation such as lighting or small shifts.

---

<a id="f144"></a>

## F144. Explain PCA from centered data to reconstruction.

**Start here.** PCA finds orthogonal directions that capture the greatest variance under squared reconstruction error.

**Work it through.**

1. Fit a training-data mean and center X; use SVD X=U*S*Vᵀ.
2. Project onto the first k right singular vectors and reconstruct with the same mean and directions.
3. Choose k using validation needs or explained variance; scale features first only when that matches the intended geometry.

**Small example.** If almost all variation lies along a diagonal line in 2D, one principal component can reconstruct points approximately on that line.

**Common trap.** Fitting PCA on all train and test data before evaluating a model.

**Follow-up.** What does variance preservation not guarantee? That the retained directions are best for predicting the target.

---

<a id="f145"></a>

## F145. Scale dimensionality reduction to large or sparse data.

**Start here.** Choose a factorization that fits the data's size and storage structure rather than forming a dense covariance matrix automatically.

**Work it through.**

1. Use randomized/truncated SVD when only a small number of leading components is needed.
2. Use incremental PCA for bounded-memory dense batches, with its approximation and batch effects understood.
3. For sparse count matrices, truncated SVD can avoid densification from centering; explain that this changes the centered-PCA problem.

**Small example.** A million-by-100000 sparse matrix may be manageable in sparse form but impossible to materialize as a dense array.

**Common trap.** Calling uncentered truncated SVD identical to centered PCA.

**Follow-up.** When would random projection fit? When approximate distance preservation is enough and interpretability is less important.

---

<a id="f146"></a>

## F146. Interpret clustering completeness and homogeneity.

**Start here.** External clustering metrics compare assignments with labels but reward different behaviors.

**Work it through.**

1. Homogeneity asks whether each predicted cluster contains one label; completeness asks whether all members of a label stay in one cluster.
2. Inspect both, plus a combined measure or task-specific evaluation, under the same data and label conventions.
3. Remember that the reference labels may not represent the only useful clustering structure.

**Small example.** Putting every observation in one cluster gives perfect completeness but generally poor homogeneity for multiple classes.

**Common trap.** Choosing a clustering algorithm based on one metric without inspecting degenerate solutions.

**Follow-up.** What does splitting every sample into its own cluster do? It gives perfect homogeneity but typically poor completeness.

---

<a id="f147"></a>

## F147. Build bag-of-words similarity search from scratch.

**Start here.** Represent documents using a shared vocabulary, then compare normalized vectors under a clearly defined weighting scheme.

**Work it through.**

1. Tokenize consistently and construct term counts or TF-IDF using corpus statistics fitted on the intended index.
2. Transform the query with the same vocabulary and inverse-document-frequency convention.
3. Use cosine similarity, handling zero vectors explicitly, and rank ties deterministically.

**Small example.** Two documents with proportional count vectors have cosine similarity 1 despite different lengths.

**Common trap.** Fitting a different vocabulary to each document or dividing by zero for an empty/out-of-vocabulary query.

**Follow-up.** What information is lost? Word order and much semantic structure, though exact rare terms can remain valuable.

---

<a id="f148"></a>

## F148. Choose hard negatives for a retrieval model.

**Start here.** Useful negatives are challenging non-matches, not mislabeled positives selected for looking similar.

**Work it through.**

1. Start with in-batch or random negatives as a baseline.
2. Mine high-scoring incorrect candidates, filter known duplicates and alternative valid answers, and control the negative mixture.
3. Evaluate on unseen queries and inspect whether the model learns real distinctions or shortcuts from mining artifacts.

**Small example.** A near-duplicate product listing may be a valid substitute rather than a true negative, depending on the task.

**Common trap.** Increasing difficulty without checking label correctness, thereby teaching the model to repel legitimate matches.

**Follow-up.** Why refresh mined negatives? The model's failure boundary changes during training.

---

<a id="f149"></a>

## F149. Create stratified samples from class buckets.

**Start here.** Sampling rules must specify desired class proportions, replacement and rounding behavior.

**Work it through.**

1. Compute class quotas from target proportions and requested size using a deterministic remainder-allocation rule.
2. Sample within buckets using a seeded generator, respecting available counts or explicitly allowing replacement.
3. Preserve or record inclusion probabilities if the sample will be used for population estimation.

**Small example.** For 10 examples at proportions [0.55,0.45], quotas cannot both be exact; a stated tie/rounding rule must allocate the final unit.

**Common trap.** Assuming an intentionally balanced sample has the original population prevalence.

**Follow-up.** When do you need weights? When estimating population quantities from unequal-probability samples.

---

<a id="f150"></a>

## F150. Choose between exact and approximate nearest-neighbor search.

**Start here.** Approximation trades some retrieval accuracy for faster or smaller search at scale.

**Work it through.**

1. Benchmark exact search as a correctness reference on a representative subset.
2. Choose an index and tune its search effort against recall, latency, memory and update frequency.
3. Evaluate downstream answer or recommendation quality, because a small recall loss can disproportionately affect rare important items.

**Small example.** An index with 95% nearest-neighbor recall may still fail badly if its misses concentrate on a critical query slice.

**Common trap.** Reporting only average query latency without index build cost, freshness or recall.

**Follow-up.** What changes when embeddings are updated? The index and stored vectors must be compatible with the new representation version.

---
