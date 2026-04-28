# Dimensionality Reduction — Interview Grill

> 45 questions on PCA, t-SNE, UMAP, autoencoders, ICA, NMF. Drill until you can answer 30+ cold.

---

## A. PCA fundamentals

**1. What is PCA solving?**
Find orthogonal directions of maximum variance in the data. Equivalent: find the rank-$k$ linear projection that minimizes reconstruction error in $\ell_2$.

**2. Derive the first principal component.**
Center the data. Maximize $w^\top \Sigma w$ subject to $\|w\| = 1$, where $\Sigma = \frac{1}{n} X^\top X$. Lagrangian gives $\Sigma w = \lambda w$ — $w$ is the top eigenvector of $\Sigma$, $\lambda$ is its variance.

**3. What does the variance maximization view say about top-$k$ components?**
The top-$k$ eigenvectors of $\Sigma$ give the rank-$k$ subspace that captures the most variance. This is a $k$-dimensional generalization: $\arg\max_{W^\top W = I} \mathrm{tr}(W^\top \Sigma W)$.

**4. PCA via SVD — write it down.**
Let $X$ be centered ($n \times d$). SVD: $X = U S V^\top$. Then $\Sigma = \frac{1}{n} V S^2 V^\top$. Right singular vectors $V$ = principal directions; $S^2/n$ = eigenvalues (variances); $US$ = scores (data in PC space).

**5. Why use SVD instead of eigendecomposition of $\Sigma$?**
Numerically more stable. Forming $X^\top X$ squares the condition number. Also faster when $n \ll d$ or $d \ll n$ (skinny SVD).

**6. Eckart-Young theorem — what does it say?**
The truncated SVD $X_k = U_k S_k V_k^\top$ is the best rank-$k$ approximation to $X$ in both Frobenius and operator norm. PCA inherits this — top-$k$ PCs minimize reconstruction error.

**7. When does PCA fail?**
Non-linear manifolds (Swiss roll), data where directions of high variance are uninteresting (e.g., dominant noise), heavy-tailed distributions, when interpretability requires sparsity.

**8. Why do we center the data?**
Without centering, the top PC just points toward the mean. PCA is about variance around the mean, not about the absolute position.

**9. Should we standardize (scale) features before PCA?**
If features are on different scales (e.g., one is dollars, another is age), yes — otherwise the high-variance feature dominates. Use correlation matrix instead of covariance, or standardize.

**10. How do you choose $k$?**
Cumulative explained variance (e.g., 95%), scree plot elbow, cross-validation on a downstream task, parallel analysis (compare to noise eigenvalues).

---

## B. PCA subtleties

**11. PCA in high-dim (d > n)?**
Covariance matrix is $d \times d$ but rank at most $n - 1$. Compute $X X^\top$ ($n \times n$) instead — eigenvectors of this give the same PCs via $v = X^\top u / \sqrt{\lambda}$. This is the kernel trick for the linear kernel.

**12. Is PCA convex?**
The variance maximization objective is non-concave in general, but the constrained problem has a closed-form solution (eigendecomposition). So we always find the global optimum.

**13. PCA vs LDA?**
PCA is unsupervised — maximizes variance. LDA is supervised — maximizes between-class / within-class variance ratio (Fisher criterion). LDA explicitly uses labels.

**14. PCA vs autoencoder?**
Linear autoencoder with $\ell_2$ loss = PCA (the encoder weights span the same subspace). Deep AEs add non-linearity and capture non-linear manifolds.

**15. What's a "loadings" interpretation in PCA?**
Each PC is a linear combination of original features. The coefficients (entries of $V$) are the loadings — they tell you how much each feature contributes to that PC.

**16. Probabilistic PCA?**
Tipping & Bishop. Generative model: $x = Wz + \mu + \epsilon$ where $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$. As $\sigma \to 0$, recovers classical PCA. Lets you handle missing data via EM.

**17. Sparse PCA?**
Add an $\ell_1$ penalty on the loadings. Each PC depends on only a few features → more interpretable. Loses orthogonality.

**18. Robust PCA?**
Decompose $X = L + S$ where $L$ is low-rank and $S$ is sparse (outliers). Solved via convex optimization (PCP — Principal Component Pursuit). Useful when classical PCA is corrupted by gross errors.

---

## C. Kernel PCA

**19. What's kernel PCA?**
Map $x$ to a high-dim feature space $\phi(x)$ via a kernel $K(x, y) = \phi(x)^\top \phi(y)$, then do PCA there implicitly. Eigendecompose the centered kernel matrix.

**20. Why use kernel PCA over PCA?**
For non-linear structure. RBF kernel can capture curved manifolds (PCA can only find flat subspaces).

**21. What's the catch with kernel PCA?**
$K$ is $n \times n$ — eigendecomposition is $O(n^3)$. Doesn't scale. Also no straightforward "inverse transform" (pre-image problem).

---

## D. t-SNE

**22. What does t-SNE optimize?**
KL divergence between two pairwise-similarity distributions. High-dim: Gaussian-like $p_{ij}$ from input distances; low-dim: Student-t (heavy-tailed) $q_{ij}$. Minimize $\mathrm{KL}(P \| Q)$.

**23. Why Student-t in low dim?**
Solves the "crowding problem." Moderate distances in high-dim need to map to large distances in low-dim. Heavy tails of Student-t allow this without large gradient penalties.

**24. What's perplexity in t-SNE?**
Effective number of neighbors per point. Sets the bandwidth $\sigma_i$ of the Gaussian for each point $i$. Typical: 5–50.

**25. What does perplexity control?**
Local vs global structure. Low perplexity → focuses on small neighborhoods; high → smooths out, preserves more global structure (somewhat).

**26. Why is t-SNE non-deterministic?**
Random initialization and stochastic gradient descent. Different runs give different layouts. Setting a seed helps reproduce.

**27. Can you interpret cluster sizes/distances in t-SNE?**
**No.** t-SNE preserves local neighborhoods, not global geometry. Cluster sizes and inter-cluster distances are not meaningful. This is a famous misuse.

**28. Does t-SNE have an `inverse_transform`?**
No — and no `transform` either. You can't embed new points without re-running. (UMAP fixes this.)

**29. Computational cost of t-SNE?**
Naive: $O(n^2)$ — pairwise similarities. Barnes-Hut t-SNE: $O(n \log n)$ via tree approximation. Still slow for $n > 10^5$.

---

## E. UMAP

**30. What's UMAP at a high level?**
Uniform Manifold Approximation and Projection. Builds a fuzzy simplicial complex from k-NN graphs in high-dim, then optimizes a low-dim graph to match. Cross-entropy loss between fuzzy graph memberships.

**31. UMAP vs t-SNE?**
UMAP is faster, more deterministic (with fixed seed), preserves more global structure, has a `transform` method for new points, and scales better. t-SNE often gives nicer-looking local clusters but is slower.

**32. Key UMAP hyperparameters?**
`n_neighbors` (local vs global tradeoff, like perplexity), `min_dist` (how tightly points pack in low-dim), `metric` (distance function in high-dim).

**33. Can you interpret UMAP cluster distances?**
Better than t-SNE but still fragile. Inter-cluster distances are *somewhat* meaningful but heavily depend on `n_neighbors` and `min_dist`. Don't over-interpret.

**34. UMAP for supervised dim reduction?**
Pass labels in fit. UMAP uses them to bias the graph construction toward separating classes — a supervised embedding.

---

## F. Autoencoders

**35. What's a standard autoencoder?**
Encoder $f_\phi(x) \to z$ (bottleneck), decoder $g_\theta(z) \to \hat{x}$. Train to minimize reconstruction loss $\|x - \hat{x}\|^2$. The bottleneck forces a compressed representation.

**36. Linear AE vs PCA?**
Identical subspace (up to rotation). The encoder spans the same $k$-dim subspace as the top-$k$ PCs.

**37. What's a denoising autoencoder?**
Corrupt input ($\tilde{x} = x + \epsilon$ or mask-out), train AE to reconstruct clean $x$ from $\tilde{x}$. Forces the model to learn robust features. Conceptual ancestor of MLM (BERT).

**38. What's a VAE?**
Variational autoencoder. Encoder outputs distribution $q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$. Loss = reconstruction + $\mathrm{KL}(q_\phi \| p(z))$ where $p(z) = \mathcal{N}(0, I)$. Generates new samples by drawing $z \sim p(z)$ and decoding.

**39. Why the KL term in VAE?**
It regularizes the latent space to match the prior $p(z)$, so we can sample from $p(z)$ and get coherent decodes. Without it, the encoder collapses to a delta — autoencoder, not generative.

**40. What's posterior collapse in VAE?**
Decoder ignores $z$; encoder outputs the prior. Common with strong autoregressive decoders (the decoder doesn't need $z$). Fixes: KL annealing, free bits, weakening the decoder.

**41. Why are deep AEs better than PCA on images?**
Convolutional layers exploit spatial structure; non-linearities capture the manifold of natural images, which is far from a linear subspace.

---

## G. ICA and NMF

**42. What's ICA?**
Independent Component Analysis. Find a linear transformation $s = Wx$ such that components of $s$ are statistically *independent* (not just uncorrelated). Used for blind source separation (e.g., cocktail party).

**43. ICA vs PCA?**
PCA: orthogonal components, maximize variance, components uncorrelated. ICA: components statistically independent (stronger), not necessarily orthogonal. Requires non-Gaussianity.

**44. Why does ICA require non-Gaussian sources?**
If sources are all Gaussian, any orthogonal rotation of them is also a valid solution — the problem is unidentifiable. Non-Gaussianity (kurtosis, negentropy) breaks the symmetry.

**45. What's NMF?**
Non-negative Matrix Factorization. Factor $X \approx WH$ with $W, H \geq 0$. Used when data is non-negative (text counts, images, audio spectrograms). Parts-based representation.

**46. NMF vs PCA?**
Both factorize $X \approx LR$. NMF constrains non-negativity, giving additive ("parts") representations. PCA gives signed components (subtractive). NMF is interpretable for topic modeling.

---

## H. Practical decisions

**47. I want to visualize 100k points in 2D — what do I use?**
UMAP. Faster than t-SNE, has transform method, scales better, more deterministic.

**48. I want to compress 1M images for retrieval — what do I use?**
Trained autoencoder (or even better, a pretrained model's embeddings + product quantization). PCA only as a quick baseline.

**49. I want interpretable topics from a document-term matrix — what do I use?**
NMF or LDA (Latent Dirichlet Allocation). NMF gives sparse, parts-based topics; LDA is probabilistic.

**50. I want to remove noise from EEG signals — what do I use?**
ICA. The brain signals are statistically independent sources mixed in the recording.

---

## I. Subtleties and gotchas

**51. What's the curse of dimensionality for DR methods?**
In very high dim, all distances become similar — $k$-NN graphs (used by t-SNE, UMAP) become unreliable. Often pre-reduce with PCA first (e.g., PCA → 50 dims → UMAP → 2 dims).

**52. Can PCA be used for compression?**
Yes — store $U_k S_k$ (scores) and $V_k$ (loadings). Reconstruct with $X_k = U_k S_k V_k^\top$. Used in image compression (JPEG-style ideas), genomics, etc.

**53. What's the relationship between PCA and spectral clustering?**
Spectral clustering = eigendecomposition of a graph Laplacian, then K-means on eigenvectors. PCA = eigendecomposition of covariance. Both are spectral methods on a kernel/affinity matrix.

**54. PCA with missing data?**
Naive PCA fails. Use probabilistic PCA + EM, or matrix completion (low-rank methods). Or impute first (mean, k-NN) then PCA.

**55. Whitening — what is it and why?**
Project to PC space and scale by $1/\sqrt{\lambda_i}$ — output has identity covariance. Removes scale and correlation. Used as preprocessing (e.g., in ICA).

---

## Quick fire

**56.** *PCA top component is the eigenvector of?* The covariance matrix.
**57.** *Best rank-k approximation of a matrix?* Truncated SVD (Eckart-Young).
**58.** *t-SNE divergence?* KL between high-dim and low-dim joint.
**59.** *UMAP loss?* Cross-entropy of fuzzy graph memberships.
**60.** *Linear AE = ?* PCA.
**61.** *ICA needs?* Non-Gaussian sources.
**62.** *NMF constraint?* Non-negativity.
**63.** *VAE second loss term?* KL to prior $p(z)$.

---

## Self-grading

If you can't answer 1-15, you don't know PCA. If you can't answer 16-30, you'll struggle on practical DR / visualization questions. If you can't answer 31-45, frontier-lab generative-model and representation-learning interviews will go past you.

Aim for 35+/63 cold. Below 25 → re-read the deep-dive.
