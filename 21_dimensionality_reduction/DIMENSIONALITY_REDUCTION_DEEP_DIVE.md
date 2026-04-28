# Dimensionality Reduction: A Frontier-Lab Interview Deep Dive

> **Why this exists.** PCA is the most-asked unsupervised method in interviews. But "modern" dimensionality reduction (t-SNE, UMAP, autoencoders) involves different assumptions and trade-offs. Strong candidates can derive PCA from variance maximization, explain what t-SNE actually optimizes, and know when each method is the right tool.

---

## 1. The map

| Method | Goal | Linear? | Preserves | Best for |
|---|---|---|---|---|
| **PCA** | Maximize variance in low-dim | Yes | Global structure | Linear projection, decorrelation |
| **Kernel PCA** | PCA in kernel feature space | No | Global, in feature space | Non-linear linear projection |
| **t-SNE** | Preserve local neighborhoods | No | Local clusters | Visualization (2D/3D) |
| **UMAP** | Preserve manifold structure | No | Local + some global | Visualization + downstream features |
| **Autoencoder** | Learned compression | No | What loss optimizes | Custom representations |
| **ICA** | Independent components | No (for sources) | Statistical independence | Source separation |
| **NMF** | Non-negative parts | Yes (for non-negative) | Additive decomposition | Topic modeling, parts |

These differ in what they preserve. PCA preserves variance; t-SNE preserves local distances; UMAP preserves neighborhoods + manifold topology; autoencoders preserve whatever the loss specifies. **The "right" method depends on what structure you care about.**

---

## 2. PCA from first principles

The most-asked dimensionality-reduction method.

### The intuition

Find directions in the input space along which the data varies most. Project onto those directions; throw away the rest.

### The setup

Centered data $X \in \mathbb{R}^{N \times d}$ (subtract column means). Goal: find unit vector $u$ that maximizes the variance of projections:

$$
\max_{\|u\| = 1} \mathrm{Var}(X u) = \max_{\|u\| = 1} u^\top \Sigma u, \qquad \Sigma = \frac{1}{N} X^\top X
$$

### The solution

By Lagrange multipliers, the maximum is attained when $u$ is the **top eigenvector of $\Sigma$**:

$$
\Sigma u = \lambda u
$$

The largest eigenvalue $\lambda$ equals the variance along $u$. Subsequent components are the next eigenvectors, all orthogonal.

### Equivalently: SVD

$X = U \Sigma_{\text{SVD}} V^\top$ (singular value decomposition). The columns of $V$ are the principal components; singular values are square roots of eigenvalues of $X^\top X$.

For numerical stability, **always compute PCA via SVD**, not via eigendecomposition of $X^\top X$ (which can lose precision for ill-conditioned data).

### Reconstruction error view

PCA also minimizes reconstruction error: project to $k$ dim, project back, minimize $\sum_i \|x_i - \hat x_i\|^2$. This gives the same components as variance maximization. **The two views are equivalent** — a beautiful classical result (Eckart-Young theorem).

### How many components?

- **Cumulative explained variance**: pick $k$ such that $\sum_{i=1}^k \lambda_i / \sum_i \lambda_i \geq$ threshold (e.g., 95%).
- **Elbow on scree plot**: plot $\lambda_i$ vs $i$; find the elbow.
- **Cross-validation**: in downstream task, find $k$ that maximizes performance.

### Assumptions of PCA

- **Linearity**: only finds linear projections. Non-linear structure (curved manifolds) won't be captured.
- **Orthogonality**: components are forced orthogonal. Real underlying factors may not be.
- **Variance ≈ importance**: high variance directions are kept. But high variance doesn't always mean important — sometimes noise dominates variance.

### When PCA fails

- Data on curved manifolds (e.g., MNIST is on a low-dim manifold but PCA needs ~50+ components to capture it).
- When important directions have low variance (e.g., signal hidden under high-variance noise).
- When data has multiple uncorrelated subspaces.

### PCA in 8 lines (the canonical interview ask)

```python
def pca(X, k):
    """Return (X_reduced [N, k], components [k, d], explained_var [k])."""
    X_centered = X - X.mean(axis=0)              # always center first
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:k]                           # top-k principal directions
    X_reduced = X_centered @ components.T         # = U[:, :k] * S[:k]
    explained_var = (S[:k] ** 2) / (X.shape[0] - 1)
    return X_reduced, components, explained_var
```

Notes you'd say out loud: (1) centering is critical, (2) SVD is more stable than eigendecomposing $X^\top X$, (3) `Vt` rows are the eigenvectors of the covariance, (4) explained-variance = singular-values squared / $(n-1)$.

---

## 3. Kernel PCA

Apply PCA in the implicit feature space of a kernel.

### How

Don't compute $\phi(x)$ explicitly. Compute the kernel matrix $K_{ij} = k(x_i, x_j)$. Center it ($K \to K - 1_N K - K 1_N + 1_N K 1_N$). Eigendecompose; project onto top eigenvectors.

### When to use

- Data lies on a non-linear manifold.
- You want PCA-like decomposition but with a non-linear feature map.

### Cons

- $O(N^2)$ memory, $O(N^3)$ training. Like all kernel methods.
- Mostly superseded by t-SNE / UMAP for visualization, autoencoders for feature learning.

---

## 4. t-SNE

t-Distributed Stochastic Neighbor Embedding (van der Maaten & Hinton 2008). The most popular method for **visualization** of high-dim data.

### The objective

Match high-dim and low-dim **conditional probability distributions** of being neighbors.

In high dim, define probability that point $j$ is a neighbor of $i$:

$$
p_{j \mid i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)}
$$

with $\sigma_i$ chosen so that the perplexity (effective number of neighbors) is a target value (5–50 typical).

In low dim (the embedding), use a Student-t distribution with 1 degree of freedom (Cauchy in 1D — heavier tails than Gaussian; kernel $\propto (1 + \|y_i - y_j\|^2)^{-1}$):

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

Minimize KL divergence between joint distributions:

$$
\mathcal{L} = \mathrm{KL}(P \,\|\, Q) = \sum_{i, j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

### Why Student-t in low dim?

The "crowding problem": Gaussian neighbors in high-dim need to map to lots of room in low-dim, but low-dim has less room. Student-t's heavy tails let distant pairs spread out farther, alleviating crowding.

### Properties

- **Preserves local structure**: nearby points in high-dim end up nearby in low-dim.
- **Doesn't preserve global structure**: distances between clusters in t-SNE are not meaningful.
- **Slow**: $O(N^2)$. Barnes-Hut acceleration makes it $O(N \log N)$. Still hard at $N > 10^5$.

### Hyperparameters

- **Perplexity**: 5–50 typical. Higher = more global structure preserved.
- **Learning rate**: 100–1000 typical.
- **Iterations**: 1000+.
- **Initialization**: PCA initialization helps stability.

### Failure modes

- Different runs give different embeddings (random init).
- Cluster sizes / distances in t-SNE are NOT proportional to anything meaningful.
- Outliers can be misplaced.
- Doesn't generalize: no transformation function for new data.

---

## 5. UMAP

Uniform Manifold Approximation and Projection (McInnes et al. 2018). The modern alternative to t-SNE.

### Core idea

Approximate the data manifold as a fuzzy simplicial complex; find a low-dim embedding that has the same fuzzy structure.

### vs t-SNE

| Aspect | t-SNE | UMAP |
|---|---|---|
| Speed | $O(N^2)$, slow | $O(N \log N)$, faster |
| Local structure | Excellent | Excellent |
| Global structure | Poor | Better |
| Stability across runs | Variable | More stable |
| Generalizability | None (no transform) | Has transform method for new points |
| Hyperparameters | Perplexity | n_neighbors, min_dist |

### When to use UMAP

- Visualization of high-dim data: UMAP usually beats t-SNE on quality and speed.
- Need to embed new points after fitting: UMAP supports this; t-SNE doesn't natively.
- Manifold learning for downstream features.

### Hyperparameters

- `n_neighbors`: how many neighbors to consider in the manifold approximation. 15 typical. Higher = more global; lower = more local.
- `min_dist`: minimum distance between points in the embedding. Smaller = tighter clusters; larger = more spread.

---

## 6. Autoencoders

Neural network approach. Encoder $\phi: \mathbb{R}^d \to \mathbb{R}^k$ and decoder $\psi: \mathbb{R}^k \to \mathbb{R}^d$ trained to reconstruct:

$$
\mathcal{L}(x, \psi(\phi(x)))
$$

The bottleneck $z = \phi(x)$ is the low-dim representation.

### Variants

- **Vanilla AE**: simple encoder-decoder MLP/CNN. Reconstruction loss (MSE or cross-entropy).
- **Sparse AE**: add L1 penalty on $z$ to encourage sparse activations.
- **Denoising AE**: corrupt input; reconstruct clean. Learns robust features.
- **VAE**: variational; $z$ is a distribution, not point. Generative.
- **Contractive AE**: penalize large Jacobians of encoder.

### Pros

- Learn task-specific features (with appropriate loss).
- Non-linear by default.
- Generalizes to new data.
- Scales to large datasets (mini-batch SGD).

### Cons

- Requires training.
- Less interpretable than PCA.
- Can be unstable; needs regularization to avoid trivial solutions.

### When to use AE over PCA

- Non-linear structure in data.
- Large datasets where SGD scaling matters.
- When the encoded space needs to be useful for downstream tasks (not just variance preservation).

---

## 7. ICA — Independent Component Analysis

Different goal: find directions where projections are **statistically independent**, not just uncorrelated (PCA only requires uncorrelated, not independent).

### When ICA matters

The classic example: blind source separation. Mix audio sources additively; ICA recovers original sources.

### Pros

- Recovers underlying generative factors when they are independent.

### Cons

- Less commonly used in modern ML; more useful in signal processing.

---

## 8. NMF — Non-negative Matrix Factorization

For non-negative data: factorize $X \approx W H$ with $W, H \geq 0$.

### When NMF wins

- Non-negative data (counts, intensities).
- Want **additive parts** decomposition (no subtractions).
- Topic modeling on document-term matrices: components are interpretable as topics.

### Cons

- Optimization is non-convex; multiple local minima.
- Less common in modern deep learning era.

---

## 9. Choosing the right method

| If you want... | Use... |
|---|---|
| Linear projection, decorrelation, baseline features | PCA |
| Visualization (2D/3D) of high-dim data | UMAP (or t-SNE) |
| Non-linear features for downstream tasks | Autoencoder |
| Non-linear PCA, theoretical principles | Kernel PCA |
| Source separation (independent factors) | ICA |
| Interpretable additive parts on non-negative data | NMF |
| Manifold learning preserving topology | UMAP |
| Specific feature learning at scale | Self-supervised contrastive (CLIP, SimCLR) |

### Modern deep learning lens

For most modern ML applications, dimensionality reduction is **implicit**: a deep network's hidden representations are learned task-specific embeddings. Explicit DR methods (PCA, t-SNE, UMAP) are mostly used for visualization, exploratory analysis, or as preprocessing for legacy pipelines.

---

## 10. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Derive PCA." | Maximize $u^\top \Sigma u$ subject to $\|u\|=1$. Lagrange → top eigenvector of $\Sigma$. Subsequent components: subsequent eigenvectors. |
| "PCA via SVD vs eigendecomposition?" | SVD on $X$ directly. More numerically stable. |
| "Why standardize before PCA?" | PCA is sensitive to feature scales. High-variance features dominate. Standardizing puts all on equal footing. |
| "t-SNE vs PCA?" | PCA: linear, global, fast. t-SNE: non-linear, local, slow. Different goals. |
| "Why use Student-t in low-dim t-SNE?" | Heavy tails alleviate the crowding problem — distant pairs can spread in low-dim. |
| "UMAP vs t-SNE?" | UMAP: faster, more stable, better global structure, has transform method. Modern default for visualization. |
| "When does PCA fail?" | Non-linear manifolds; signal in low-variance directions; multiple uncorrelated subspaces. |
| "PCA reconstruction error?" | Eckart-Young theorem: top-$k$ SVD truncation minimizes Frobenius reconstruction error. |
| "Autoencoder vs PCA?" | AE: non-linear, learned, scales with SGD. PCA: linear, closed-form, principled. AE > PCA for non-linear data with enough samples. |
| "How do you pick $k$ for PCA?" | Cumulative variance threshold (95%), elbow on scree plot, or downstream task CV. |

---

## 11. The 8 most-asked DR interview questions

1. **Walk me through PCA derivation.** Maximize variance $u^\top \Sigma u$; top eigenvector via Lagrange. Equivalent to SVD of $X$.
2. **PCA via SVD vs covariance eigendecomposition?** SVD more numerically stable; same result.
3. **What's t-SNE doing?** Match high-dim and low-dim neighborhood probability distributions; KL divergence loss; Student-t in low-dim avoids crowding.
4. **t-SNE vs UMAP?** UMAP is faster, more stable, preserves more global structure, has transform method. Modern default.
5. **Why use Student-t in t-SNE?** Heavy tails handle the crowding problem in low-dim.
6. **Autoencoder vs PCA?** AE: non-linear, learned, scales. PCA: linear, closed-form. AE wins on non-linear data.
7. **How to choose $k$?** Cumulative explained variance (95%), scree plot elbow, or CV on downstream task.
8. **When does PCA fail?** Non-linear manifolds; signal in low-variance directions; non-orthogonal underlying factors.

---

## 12. Drill plan

1. Whiteboard PCA derivation (variance max + Lagrange).
2. State the SVD decomposition and connection to PCA.
3. Know t-SNE's KL objective at sketchy level.
4. Compare UMAP vs t-SNE on speed and structure preservation.
5. Drill `INTERVIEW_GRILL.md`.

---

## 13. Further reading

- Hotelling, "Analysis of a complex of statistical variables into principal components" (1933) — original PCA.
- Eckart & Young, "The approximation of one matrix by another of lower rank" (1936) — SVD reconstruction.
- van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008).
- McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018).
- Lee & Seung, "Algorithms for Non-negative Matrix Factorization" (2001).
- Hyvärinen, "Independent Component Analysis" (2001).
