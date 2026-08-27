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

> **Saying it out loud.** Every one of these methods throws away dimensions, so the only real question is what it decides to keep. PCA keeps the directions where the data spreads out most. t-SNE and UMAP keep who your nearest neighbors are and cheerfully distort everything else. Autoencoders keep whatever your loss function told them to keep. So when someone asks which one to use, the honest answer is another question: what structure do you actually care about, global geometry or local neighborhoods?

---

## 2. PCA from first principles

The most-asked dimensionality-reduction method.

### The intuition

Find directions in the input space along which the data varies most. Project onto those directions; throw away the rest.

> **Saying it out loud.** The intuition for PCA is that most real data doesn't fill up its space, it lies in a thin pancake tilted at some angle. PCA finds the direction the pancake is widest along, then the next widest perpendicular to that, and so on. You keep the first few directions and drop the rest, because those dropped directions barely had any spread in them anyway. The gamble baked in is that spread means information, and that's exactly where PCA goes wrong when your signal is quiet and your noise is loud.

### The setup

> **In plain language.** This is where we write down what PCA is actually asking for, as a maximization problem. In words: find a single direction in the input space such that when you shadow every data point onto that line, the shadows are as spread out as possible. The matrix $\Sigma$ below is just the covariance of your centered data, and $u^\top \Sigma u$ is the variance of the data measured along direction $u$.

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

> **Saying it out loud.** Here's the punchline: you don't need calculus tricks to remember this, the answer is just the top eigenvector of the covariance matrix. You set up "maximize variance along $u$, subject to $u$ being unit length," you attach a Lagrange multiplier, take the derivative, and the condition you get out is literally the eigenvector equation. And the multiplier itself turns out to be the variance in that direction, so the eigenvalues rank your components for free. The second component is the next eigenvector, which is orthogonal by construction, and that forced orthogonality is the assumption that bites you when the real underlying factors are correlated.

### Equivalently: SVD

> **In plain language.** This says you can get the same principal components without ever building the covariance matrix. The SVD splits your data matrix into three pieces, and one of them, the rows of $V^\top$, are exactly the principal directions. It's the same answer by a numerically safer route.

$X = U \Sigma_{\text{SVD}} V^\top$ (singular value decomposition). The columns of $V$ are the principal components; singular values are square roots of eigenvalues of $X^\top X$.

For numerical stability, **always compute PCA via SVD**, not via eigendecomposition of $X^\top X$ (which can lose precision for ill-conditioned data).

> **Saying it out loud.** In practice you never eigendecompose the covariance matrix, you just run SVD on the centered data directly. The reason is that forming $X^\top X$ squares the condition number, so you lose about half your digits of precision before you've computed anything. SVD gets you the same components without ever forming that product. It's a one-line change in code and it's the kind of detail that tells an interviewer you've actually run this on real data.

### Reconstruction error view

PCA also minimizes reconstruction error: project to $k$ dim, project back, minimize $\sum_i \|x_i - \hat x_i\|^2$. This gives the same components as variance maximization. **The two views are equivalent** — a beautiful classical result (Eckart-Young theorem).

> **Saying it out loud.** There's a second story about PCA that gives you the identical answer. Instead of asking which directions have the most spread, ask which $k$-dimensional subspace lets you squash the data down and pop it back up with the smallest total error. Those two questions have the same solution, which is the Eckart-Young theorem, and it's why PCA feels like it's doing compression and feature extraction at the same time. The number to remember: the reconstruction error you're left with is exactly the sum of the eigenvalues you threw away.

### How many components?

- **Cumulative explained variance**: pick $k$ such that $\sum_{i=1}^k \lambda_i / \sum_i \lambda_i \geq$ threshold (e.g., 95%).
- **Elbow on scree plot**: plot $\lambda_i$ vs $i$; find the elbow.
- **Cross-validation**: in downstream task, find $k$ that maximizes performance.

> **Saying it out loud.** Picking $k$ is a judgment call and you should say that out loud rather than pretending there's a formula. The three standard moves are: keep enough components to explain some fraction of the variance, ninety-five percent being the usual choice; look at the scree plot and find the elbow where eigenvalues flatten out; or ignore both and just cross-validate $k$ against whatever downstream task you actually care about. The third one is the honest answer, because explaining ninety-five percent of the variance and being useful for your classifier are different objectives.

### Assumptions of PCA

- **Linearity**: only finds linear projections. Non-linear structure (curved manifolds) won't be captured.
- **Orthogonality**: components are forced orthogonal. Real underlying factors may not be.
- **Variance ≈ importance**: high variance directions are kept. But high variance doesn't always mean important — sometimes noise dominates variance.

> **Saying it out loud.** PCA is making three assumptions and it's worth naming all three. It only looks for straight-line projections, so anything curved is out of reach. It forces components to be perpendicular to each other, even if the true factors generating your data aren't. And it treats high variance as high importance, which is the one that actually burns people. If you have a sensor with a units bug reading in thousands while everything else is between zero and one, that sensor is your first principal component regardless of whether it means anything.

### When PCA fails

- Data on curved manifolds (e.g., MNIST is on a low-dim manifold but PCA needs ~50+ components to capture it).
- When important directions have low variance (e.g., signal hidden under high-variance noise).
- When data has multiple uncorrelated subspaces.

> **Saying it out loud.** PCA fails in three recognizable situations. Curved manifolds, where the data lives on a rolled-up sheet and no flat projection can unroll it: MNIST digits sit on a low-dimensional manifold but PCA still needs fifty-odd components to describe them decently. Cases where the interesting signal is small and the noise is loud, so the directions you keep are pure noise. And data made of several unrelated subspaces, where one global set of axes just doesn't describe any of them well. The tell for the second one is that your top components look like nothing but they explain most of the variance.

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

> **Saying it out loud.** If they ask you to code PCA, the whole thing is four real lines and the first one is the one people forget. Center the data by subtracting the column means, run SVD, take the top $k$ rows of $V^\top$ as your components, and project. If you skip centering, your first component just points at the mean of your data and you've wasted a dimension measuring where the cloud sits instead of how it's shaped. Explained variance is the singular values squared divided by $n-1$.

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

> **Saying it out loud.** Kernel PCA is PCA run in a space you never actually build. The trick is that PCA only ever needs inner products between points, so you swap in a kernel function that computes what the inner product would be in some huge nonlinear feature space, and you get nonlinear components out of a linear method. The catch is the one every kernel method has: you're building an $N \times N$ matrix and eigendecomposing it, so you're at $O(N^2)$ memory and $O(N^3)$ time. That's why for anything past a few tens of thousands of points people reach for UMAP or an autoencoder instead.

---

## 4. t-SNE

t-Distributed Stochastic Neighbor Embedding (van der Maaten & Hinton 2008). The most popular method for **visualization** of high-dim data.

### The objective

> **In plain language.** This section writes down what t-SNE is optimizing. The idea is to turn "who is near whom" into a probability distribution in the original high-dimensional space, build the same kind of distribution in your two-dimensional picture, and then move the points around until the two distributions agree. All the notation below is just those two distributions and the measure of disagreement between them.

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

> **Saying it out loud.** t-SNE turns the whole problem into matching two probability distributions. In the original space, for each point, you write down how likely every other point is to be its neighbor, using a Gaussian whose width is tuned per point so each one has the same effective number of neighbors, which is what perplexity means. Then you do the same thing in the 2D picture and shove the points around until the two sets of neighbor probabilities line up, measured by KL divergence. The important consequence of using KL in that direction is that it punishes you hard for putting true neighbors far apart, and barely punishes you for putting strangers close together, which is exactly why local structure survives and global structure doesn't.

### Why Student-t in low dim?

The "crowding problem": Gaussian neighbors in high-dim need to map to lots of room in low-dim, but low-dim has less room. Student-t's heavy tails let distant pairs spread out farther, alleviating crowding.

> **Saying it out loud.** The Student-t in the low-dimensional space is fixing the crowding problem, and here's the crowding problem in one sentence: in high dimensions a point can have far more roughly-equidistant neighbors than there is room for in a 2D plane. If you used a Gaussian in both spaces, everything would get crushed into the middle. The Student-t has fat tails, so a moderately-far pair in 2D still gets decent probability mass, which lets clusters push apart and leave gaps. That's the whole reason the picture separates into islands instead of one blob.

### Properties

- **Preserves local structure**: nearby points in high-dim end up nearby in low-dim.
- **Doesn't preserve global structure**: distances between clusters in t-SNE are not meaningful.
- **Slow**: $O(N^2)$. Barnes-Hut acceleration makes it $O(N \log N)$. Still hard at $N > 10^5$.

> **Saying it out loud.** The single most important thing to say about a t-SNE plot is what you're not allowed to read off it. Local structure is trustworthy: points that look like neighbors really were neighbors. Global structure is not: the distance between two clusters, and the size of a cluster, mean essentially nothing. And it's slow, naively $O(N^2)$, or $O(N \log N)$ with Barnes-Hut, which still gets painful past about a hundred thousand points. If you present a t-SNE plot and claim cluster A is more similar to B than to C because it's closer, you've made the classic mistake.

### Hyperparameters

- **Perplexity**: 5–50 typical. Higher = more global structure preserved.
- **Learning rate**: 100–1000 typical.
- **Iterations**: 1000+.
- **Initialization**: PCA initialization helps stability.

> **Saying it out loud.** Perplexity is the knob that matters and it's basically "how many neighbors count as your neighborhood," typically five to fifty. Turn it up and you get more global structure and blurrier clusters; turn it down and you get tight little islands that may be pure noise. Learning rate in the hundreds and at least a thousand iterations are the usual defaults, and initializing from PCA instead of random makes runs far more reproducible. The honest practice is to run it at three different perplexities and only trust the structure that survives all three.

### Failure modes

- Different runs give different embeddings (random init).
- Cluster sizes / distances in t-SNE are NOT proportional to anything meaningful.
- Outliers can be misplaced.
- Doesn't generalize: no transformation function for new data.

> **Saying it out loud.** The failure modes of t-SNE are all about over-reading the picture. Two runs with different random seeds give visibly different layouts, cluster sizes and inter-cluster distances carry no meaning, outliers get flung to arbitrary places, and there's no fitted transform, so a new data point means rerunning the whole thing. That last one is the practical killer: t-SNE is an exploratory picture, not a preprocessing step you can put in a production pipeline.

---

## 5. UMAP

Uniform Manifold Approximation and Projection (McInnes et al. 2018). The modern alternative to t-SNE.

### Core idea

Approximate the data manifold as a fuzzy simplicial complex; find a low-dim embedding that has the same fuzzy structure.

> **Saying it out loud.** UMAP's story is that your data lies on some curved surface, a manifold, and it builds a fuzzy graph of who's connected to whom on that surface, then lays out a low-dimensional graph with the same connectivity. Practically it's a k-nearest-neighbor graph with soft edge weights, optimized with a cross-entropy-flavored loss and negative sampling. That negative sampling is why it's fast, roughly $O(N \log N)$, instead of comparing every pair. The theory is heavier than t-SNE's but the thing you're doing is more intuitive: preserve the neighbor graph.

### vs t-SNE

| Aspect | t-SNE | UMAP |
|---|---|---|
| Speed | $O(N^2)$, slow | $O(N \log N)$, faster |
| Local structure | Excellent | Excellent |
| Global structure | Poor | Better |
| Stability across runs | Variable | More stable |
| Generalizability | None (no transform) | Has transform method for new points |
| Hyperparameters | Perplexity | n_neighbors, min_dist |

> **Saying it out loud.** Compared with t-SNE, UMAP wins on almost every practical axis. It's substantially faster, it's more stable run to run, it retains more meaningful global structure, and crucially it gives you a transform method so you can embed new points without refitting. t-SNE's local structure is just as good, so it isn't strictly worse, but there's rarely a reason to choose it today. The one honest caution: UMAP's better global structure is better, not correct, so inter-cluster distances still shouldn't be quoted as measurements.

### When to use UMAP

- Visualization of high-dim data: UMAP usually beats t-SNE on quality and speed.
- Need to embed new points after fitting: UMAP supports this; t-SNE doesn't natively.
- Manifold learning for downstream features.

> **Saying it out loud.** Reach for UMAP when you want to look at high-dimensional data in two dimensions, when you need to embed new points later without refitting the whole thing, or when you want the low-dimensional coordinates as actual features for a downstream model. That last use case is where it beats t-SNE decisively, because t-SNE has no transform at all. The tradeoff to name is that UMAP's embeddings are still not distance-preserving, so using them as features can quietly destroy information your model needed.

### Hyperparameters

- `n_neighbors`: how many neighbors to consider in the manifold approximation. 15 typical. Higher = more global; lower = more local.
- `min_dist`: minimum distance between points in the embedding. Smaller = tighter clusters; larger = more spread.

> **Saying it out loud.** There are really only two knobs. n_neighbors controls how much of the data each point looks at when building the graph, with fifteen as the default: crank it up and you get more global shape, dial it down and you get fine local detail. min_dist controls how tightly points are allowed to pack in the output: small values give dense clumps that look great for finding clusters, large values spread things out and preserve more of the topology. If you want visually clean clusters use a small min_dist, but understand you've made the clusters look more separated than they are.

---

## 6. Autoencoders

Neural network approach. Encoder $\phi: \mathbb{R}^d \to \mathbb{R}^k$ and decoder $\psi: \mathbb{R}^k \to \mathbb{R}^d$ trained to reconstruct:

$$
\mathcal{L}(x, \psi(\phi(x)))
$$

The bottleneck $z = \phi(x)$ is the low-dim representation.

> **Saying it out loud.** An autoencoder is a network that's trained to copy its input to its output, through a bottleneck too narrow to just pass the data through. Because the middle layer is small, the network is forced to learn a compressed code that keeps whatever's needed to reconstruct. That bottleneck vector is your low-dimensional representation. Nice fact for interviews: a linear autoencoder with squared-error loss recovers the same subspace PCA does, so PCA is literally the linear special case.

### Variants

- **Vanilla AE**: simple encoder-decoder MLP/CNN. Reconstruction loss (MSE or cross-entropy).
- **Sparse AE**: add L1 penalty on $z$ to encourage sparse activations.
- **Denoising AE**: corrupt input; reconstruct clean. Learns robust features.
- **VAE**: variational; $z$ is a distribution, not point. Generative.
- **Contractive AE**: penalize large Jacobians of encoder.

> **Saying it out loud.** The variants are all about what you add on top of plain reconstruction. Sparse autoencoders penalize the code with an L1 term so only a few units fire, which is exactly the trick behind the sparse autoencoders people use for interpretability today. Denoising autoencoders corrupt the input and ask for the clean version, which stops the network from learning an identity shortcut. Contractive ones penalize the encoder's Jacobian so nearby inputs map to nearby codes. And VAEs make the code a distribution rather than a point, which is what turns it into a generative model you can sample from.

### Pros

- Learn task-specific features (with appropriate loss).
- Non-linear by default.
- Generalizes to new data.
- Scales to large datasets (mini-batch SGD).

### Cons

- Requires training.
- Less interpretable than PCA.
- Can be unstable; needs regularization to avoid trivial solutions.

> **Saying it out loud.** The case for an autoencoder over PCA is that it's nonlinear, it learns whatever your loss defines as important, it generalizes to new points, and it trains by minibatch SGD so it scales to datasets that would never fit in a covariance matrix. The case against is everything that comes with training a network: you need data and hyperparameters and time, the latent axes mean nothing individually so you lose PCA's interpretability, and without regularization the thing can find degenerate solutions that reconstruct perfectly and represent nothing. PCA has a closed-form answer; an autoencoder has a training run that might not converge.

### When to use AE over PCA

- Non-linear structure in data.
- Large datasets where SGD scaling matters.
- When the encoded space needs to be useful for downstream tasks (not just variance preservation).

> **Saying it out loud.** Use an autoencoder over PCA when the structure is genuinely nonlinear, when your dataset is too big for a closed-form decomposition, or when the code has to be good for a downstream task rather than just good at reconstructing. If your data is roughly linear and you have a few thousand samples, PCA will match it and take one second. The rule of thumb worth saying: try PCA first as your baseline, and only reach for the autoencoder when PCA's reconstruction error at your target dimension is unacceptable.

---

## 7. ICA — Independent Component Analysis

Different goal: find directions where projections are **statistically independent**, not just uncorrelated (PCA only requires uncorrelated, not independent).

### When ICA matters

The classic example: blind source separation. Mix audio sources additively; ICA recovers original sources.

### Pros

- Recovers underlying generative factors when they are independent.

### Cons

- Less commonly used in modern ML; more useful in signal processing.

> **Saying it out loud.** ICA is after a different thing than PCA. PCA gives you uncorrelated components; ICA gives you statistically independent ones, which is a much stronger condition, and it gets there by finding directions whose projections look as non-Gaussian as possible. The classic picture is the cocktail party: several microphones each pick up a mix of several speakers, and ICA pulls the individual voices back out. It's still the standard tool in EEG and audio, and the limitation to name is that it can't recover the order or the scale of the sources, only the directions.

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

> **Saying it out loud.** NMF factorizes your data into two matrices that are both non-negative, and that constraint is doing all the interpretive work. Because nothing can be subtracted, components have to combine additively, so you get parts rather than cancellation: run it on faces and you get noses and eyebrows, run it on a document-term matrix and you get topics. PCA on the same data gives components with negative entries that nobody can read. The cost is that the optimization is non-convex, so different initializations give different answers, and you have to pick the number of components yourself.

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

> **Saying it out loud.** The way to pick is to start from what you want, not from what's fashionable. Linear baseline features or decorrelation, that's PCA. A picture of high-dimensional data, that's UMAP. Nonlinear features at scale, an autoencoder. Interpretable additive parts on count data, NMF. Recovering independent sources, ICA. And the meta-point worth making: in modern deep learning, dimensionality reduction is mostly implicit, since a network's hidden layer is already a learned low-dimensional embedding, so explicit methods now live mainly in exploration and visualization.

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

> **Saying it out loud.** If I had to name the two that come up most: derive PCA by maximizing $u^\top \Sigma u$ under a unit-norm constraint and landing on the top eigenvector, and know why you standardize first, which is that PCA is scale-sensitive so a feature measured in dollars will dominate one measured in fractions. After that it's the Student-t crowding answer for t-SNE and the Eckart-Young result for reconstruction. The single most common miss is forgetting to mention centering.

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
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

---

## 13. Further reading

- Hotelling, "Analysis of a complex of statistical variables into principal components" (1933) — original PCA.
- Eckart & Young, "The approximation of one matrix by another of lower rank" (1936) — SVD reconstruction.
- van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008).
- McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" (2018).
- Lee & Seung, "Algorithms for Non-negative Matrix Factorization" (2001).
- Hyvärinen, "Independent Component Analysis" (2001).
