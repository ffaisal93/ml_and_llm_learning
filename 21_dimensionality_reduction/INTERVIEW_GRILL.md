# Dimensionality Reduction — Interview Grill

> 45 questions on PCA, t-SNE, UMAP, autoencoders, ICA, NMF. Drill until you can answer 30+ cold.

---

## A. PCA fundamentals

**1. What is PCA solving?**
Find orthogonal directions of maximum variance in the data. Equivalent: find the rank-$k$ linear projection that minimizes reconstruction error in $\ell_2$.

> **Saying it out loud.** PCA is looking for the directions your data actually spreads out along, and keeping only those. There's a second way to say the same thing: it's finding the flat $k$-dimensional slab that sits closest to your data cloud, so squashing onto it and popping back out loses the least. Those two descriptions give literally the same answer. The assumption you're buying into is that variance equals information, which is fine until your loudest direction is a broken sensor.

**2. Derive the first principal component.**
Center the data. Maximize $w^\top \Sigma w$ subject to $\|w\| = 1$, where $\Sigma = \frac{1}{n} X^\top X$. Lagrangian gives $\Sigma w = \lambda w$ — $w$ is the top eigenvector of $\Sigma$, $\lambda$ is its variance.

> **Saying it out loud.** Center the data first, then ask for the unit vector $w$ that makes the variance of the projections as large as possible. Write that as maximize $w^\top \Sigma w$ subject to $w$ having length one, attach a Lagrange multiplier, differentiate, and what falls out is $\Sigma w = \lambda w$. So the answer is just the top eigenvector of the covariance matrix, and the multiplier $\lambda$ is the variance you captured. If you forget the centering step, the first component ends up pointing at the mean instead of describing the shape.

**3. What does the variance maximization view say about top-$k$ components?**
The top-$k$ eigenvectors of $\Sigma$ give the rank-$k$ subspace that captures the most variance. This is a $k$-dimensional generalization: $\arg\max_{W^\top W = I} \mathrm{tr}(W^\top \Sigma W)$.

> **Saying it out loud.** It generalizes cleanly: for $k$ dimensions you want the orthonormal set of directions maximizing the total variance captured, which is the trace of $W^\top \Sigma W$. The answer is the top $k$ eigenvectors, and the neat part is that greedily taking them one at a time gives the same result as optimizing all $k$ jointly. That's not true of most subset-selection problems, and it's why PCA is nested: your top-two solution is your top-one solution plus one more direction.

**4. PCA via SVD — write it down.**
Let $X$ be centered ($n \times d$). SVD: $X = U S V^\top$. Then $\Sigma = \frac{1}{n} V S^2 V^\top$. Right singular vectors $V$ = principal directions; $S^2/n$ = eigenvalues (variances); $US$ = scores (data in PC space).

> **Saying it out loud.** Take the centered data matrix and factor it as $X = U S V^\top$. Then the right singular vectors, the columns of $V$, are your principal directions, the squared singular values over $n$ are the variances, and $US$ gives you the data already projected into component space. You never form the covariance matrix at all. The one-liner: principal components are the right singular vectors of the centered data.

**5. Why use SVD instead of eigendecomposition of $\Sigma$?**
Numerically more stable. Forming $X^\top X$ squares the condition number. Also faster when $n \ll d$ or $d \ll n$ (skinny SVD).

> **Saying it out loud.** Because building $X^\top X$ squares the condition number, and squaring a condition number means throwing away half your significant digits before the eigensolver even starts. SVD works on $X$ directly and never takes that hit. It's also cheaper when the matrix is very skinny or very wide, since you can take the thin SVD. On badly-conditioned data the difference isn't cosmetic, you can get components that are numerically garbage.

**6. Eckart-Young theorem — what does it say?**
The truncated SVD $X_k = U_k S_k V_k^\top$ is the best rank-$k$ approximation to $X$ in both Frobenius and operator norm. PCA inherits this — top-$k$ PCs minimize reconstruction error.

> **Saying it out loud.** Eckart-Young says that if you want the best rank-$k$ approximation to a matrix, you just truncate its SVD, and that's optimal in both the Frobenius and the spectral norm. That's a strong statement, because in general low-rank approximation problems are hard, and this one has a closed form. PCA inherits it directly, which is why the variance-maximizing components are also the reconstruction-error-minimizing ones. The error you're left with is exactly the sum of the singular values you dropped, squared.

**7. When does PCA fail?**
Non-linear manifolds (Swiss roll), data where directions of high variance are uninteresting (e.g., dominant noise), heavy-tailed distributions, when interpretability requires sparsity.

> **Saying it out loud.** PCA fails whenever "flat and high-variance" isn't the right description of your structure. Curved manifolds like a Swiss roll, where no linear projection can unroll the sheet. Data where the dominant variance is noise, so your top components are the noise. Heavy-tailed data, where a few outliers drag the whole covariance around. And cases where you need interpretability, since every component is a dense mix of all your features. The Swiss roll is the standard picture for the first one and it's worth naming.

**8. Why do we center the data?**
Without centering, the top PC just points toward the mean. PCA is about variance around the mean, not about the absolute position.

> **Saying it out loud.** Because PCA is about the shape of the cloud, not where the cloud sits. If you skip centering, the first component will point roughly toward the mean vector, and you've spent one whole dimension encoding an offset you already knew. Centering makes $X^\top X$ the actual covariance rather than a second-moment matrix. This is the single most common bug in a hand-rolled PCA implementation.

**9. Should we standardize (scale) features before PCA?**
If features are on different scales (e.g., one is dollars, another is age), yes — otherwise the high-variance feature dominates. Use correlation matrix instead of covariance, or standardize.

> **Saying it out loud.** Usually yes, if your features live on different scales. PCA measures variance in whatever units you give it, so a feature in dollars and a feature in years aren't comparable, and the dollar column will dominate purely because its numbers are bigger. Standardizing, which is the same as running PCA on the correlation matrix rather than the covariance, puts everything on equal footing. The exception is when your features share meaningful units, like pixel intensities, where rescaling would destroy real information.

**10. How do you choose $k$?**
Cumulative explained variance (e.g., 95%), scree plot elbow, cross-validation on a downstream task, parallel analysis (compare to noise eigenvalues).

> **Saying it out loud.** Four options and you should say they're all heuristics. Cumulative explained variance to some threshold, ninety-five percent being conventional. The elbow of the scree plot, where the eigenvalues flatten. Cross-validation on whatever downstream task you actually care about, which is the honest one. And parallel analysis, where you compare your eigenvalues to those you'd get from pure noise of the same shape and keep only the ones that stand out. Naming parallel analysis is a nice signal because most candidates only know the first two.

---

## B. PCA subtleties

**11. PCA in high-dim (d > n)?**
Covariance matrix is $d \times d$ but rank at most $n - 1$. Compute $X X^\top$ ($n \times n$) instead — eigenvectors of this give the same PCs via $v = X^\top u / \sqrt{\lambda}$. This is the kernel trick for the linear kernel.

> **Saying it out loud.** When you have more features than samples, the covariance matrix is huge but its rank is at most $n-1$, so most of it is zeros in disguise. Instead of eigendecomposing the $d \times d$ covariance, eigendecompose the $n \times n$ Gram matrix $XX^\top$ and map the eigenvectors back with $v = X^\top u / \sqrt{\lambda}$. Same components, dramatically less compute. That's the kernel trick with a linear kernel, and it's what makes PCA on gene-expression data with twenty thousand genes and two hundred patients tractable.

**12. Is PCA convex?**
The variance maximization objective is non-concave in general, but the constrained problem has a closed-form solution (eigendecomposition). So we always find the global optimum.

> **Saying it out loud.** Technically the objective isn't concave, so it's not a convex program, but that doesn't matter because the constrained problem has a closed-form solution via eigendecomposition. You always land on the global optimum, every time, no initialization and no local minima. That's the thing that separates PCA from an autoencoder or from NMF, both of which can and do get stuck. Determinism up to sign flips is a real practical advantage.

**13. PCA vs LDA?**
PCA is unsupervised — maximizes variance. LDA is supervised — maximizes between-class / within-class variance ratio (Fisher criterion). LDA explicitly uses labels.

> **Saying it out loud.** PCA doesn't know your labels and LDA does. PCA finds directions of maximum total variance; LDA finds directions that maximize between-class spread relative to within-class spread, which is the Fisher criterion. So for a classification problem LDA usually gives you a much better low-dimensional view. The constraint on LDA is that it can only give you at most number-of-classes-minus-one dimensions, so with two classes you get exactly one axis.

**14. PCA vs autoencoder?**
Linear autoencoder with $\ell_2$ loss = PCA (the encoder weights span the same subspace). Deep AEs add non-linearity and capture non-linear manifolds.

> **Saying it out loud.** A linear autoencoder with squared-error loss recovers the same subspace PCA does, though the individual axes may be rotated within it. Add nonlinear activations and depth and you get something strictly more expressive that can follow curved manifolds. So PCA is the linear special case of an autoencoder. The tradeoff is that PCA has an exact closed-form solution while the autoencoder has a training run with hyperparameters and no guarantee of finding the optimum.

**15. What's a "loadings" interpretation in PCA?**
Each PC is a linear combination of original features. The coefficients (entries of $V$) are the loadings — they tell you how much each feature contributes to that PC.

> **Saying it out loud.** Loadings are just the recipe: each principal component is a weighted sum of your original features, and the loadings are those weights, the entries of $V$. If a component has a big loading on income and a big loading on education, you can start telling a story about what it measures. That's PCA's interpretability advantage over an autoencoder. The catch is that loadings are dense, every feature contributes a bit, which is exactly the problem sparse PCA exists to solve.

**16. Probabilistic PCA?**
Tipping & Bishop. Generative model: $x = Wz + \mu + \epsilon$ where $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$. As $\sigma \to 0$, recovers classical PCA. Lets you handle missing data via EM.

> **Saying it out loud.** Probabilistic PCA turns PCA into an actual generative model: you assume a low-dimensional Gaussian latent, map it up through a linear map, and add isotropic Gaussian noise. Fit it by maximum likelihood and as the noise variance goes to zero you recover classical PCA exactly. The payoff is everything a probabilistic model gives you: you can fit it with EM, handle missing entries naturally, get a likelihood, and put it inside a mixture. That missing-data capability is the main reason to reach for it.

**17. Sparse PCA?**
Add an $\ell_1$ penalty on the loadings. Each PC depends on only a few features → more interpretable. Loses orthogonality.

> **Saying it out loud.** Sparse PCA adds an L1 penalty on the loadings so each component only uses a handful of features. Now you can say "component three is these five genes" instead of "component three is a weighted blend of twenty thousand genes," which is a completely different conversation with a domain expert. The price is real: you lose exact orthogonality between components and the closed-form solution, so you're back to iterative optimization. You're trading mathematical tidiness for something a human can read.

**18. Robust PCA?**
Decompose $X = L + S$ where $L$ is low-rank and $S$ is sparse (outliers). Solved via convex optimization (PCP — Principal Component Pursuit). Useful when classical PCA is corrupted by gross errors.

> **Saying it out loud.** Robust PCA splits the matrix into a low-rank part plus a sparse part, where the sparse part absorbs the outliers. It's solved as a convex program called Principal Component Pursuit, minimizing nuclear norm plus L1. The reason you'd want it is that ordinary PCA is squared-error based, so a handful of gross errors can swing every component. The textbook demo is surveillance video, where the low-rank part is the static background and the sparse part is the people walking through.

---

## C. Kernel PCA

**19. What's kernel PCA?**
Map $x$ to a high-dim feature space $\phi(x)$ via a kernel $K(x, y) = \phi(x)^\top \phi(y)$, then do PCA there implicitly. Eigendecompose the centered kernel matrix.

> **Saying it out loud.** Kernel PCA is PCA done in a feature space you never build. Since PCA only needs inner products, you replace them with a kernel function that computes the inner product in some implicit high-dimensional space, then eigendecompose the centered kernel matrix. Out come nonlinear components from a linear method. The centering step is easy to forget and it has to happen in feature space, not on the raw data.

**20. Why use kernel PCA over PCA?**
For non-linear structure. RBF kernel can capture curved manifolds (PCA can only find flat subspaces).

> **Saying it out loud.** Because PCA can only find flat subspaces, and lots of real structure is curved. An RBF kernel effectively lets you bend the projection, so a dataset shaped like concentric rings, which PCA cannot separate at all, becomes separable in kernel PCA's first couple of components. You choose the nonlinearity by choosing the kernel. The tradeoff is you've now got a kernel and a bandwidth to tune, with no clean way to cross-validate an unsupervised objective.

**21. What's the catch with kernel PCA?**
$K$ is $n \times n$ — eigendecomposition is $O(n^3)$. Doesn't scale. Also no straightforward "inverse transform" (pre-image problem).

> **Saying it out loud.** The catch is the same one every kernel method has: the kernel matrix is $n$ by $n$, so you're at quadratic memory and cubic time, and it stops being practical somewhere in the tens of thousands of points. There's a second, less obvious catch, which is the pre-image problem: you can project a point down but there's no exact way to map back to input space, because not every point in feature space corresponds to a real input. So no clean inverse transform, and that kills a lot of denoising use cases.

---

## D. t-SNE

**22. What does t-SNE optimize?**
KL divergence between two pairwise-similarity distributions. High-dim: Gaussian-like $p_{ij}$ from input distances; low-dim: Student-t (heavy-tailed) $q_{ij}$. Minimize $\mathrm{KL}(P \| Q)$.

> **Saying it out loud.** t-SNE minimizes the KL divergence between two distributions over pairs of points. In the original space you define a probability that $j$ is a neighbor of $i$ using a Gaussian, and in the 2D map you define the same thing using a heavy-tailed Student-t, then you move the 2D points until those two distributions match. The direction of the KL matters a lot: $\mathrm{KL}(P \| Q)$ heavily penalizes putting true neighbors far apart, but hardly penalizes putting non-neighbors close together. That asymmetry is exactly why local structure is preserved and global structure isn't.

**23. Why Student-t in low dim?**
Solves the "crowding problem." Moderate distances in high-dim need to map to large distances in low-dim. Heavy tails of Student-t allow this without large gradient penalties.

> **Saying it out loud.** It's fixing the crowding problem. In high dimensions a point can have many neighbors at roughly the same distance, and in two dimensions there simply isn't enough room to place them all at that distance. Using a heavy-tailed Student-t in the low-dimensional space means a moderately-separated pair still gets reasonable probability, so clusters can push apart instead of getting crushed together. Concretely, the heavy tails also keep the repulsive gradients from exploding for far-apart pairs.

**24. What's perplexity in t-SNE?**
Effective number of neighbors per point. Sets the bandwidth $\sigma_i$ of the Gaussian for each point $i$. Typical: 5–50.

> **Saying it out loud.** Perplexity is roughly how many neighbors each point pays attention to. Mechanically, for every point t-SNE binary-searches a Gaussian bandwidth so that the entropy of that point's neighbor distribution hits your target, and perplexity is two to that entropy. That adaptive bandwidth is why t-SNE handles regions of different density gracefully. Typical values are five to fifty, and the thing to say is that it's an effective neighbor count, not a hard $k$.

**25. What does perplexity control?**
Local vs global structure. Low perplexity → focuses on small neighborhoods; high → smooths out, preserves more global structure (somewhat).

> **Saying it out loud.** Perplexity is the local-versus-global dial. Set it low and each point only sees a few neighbors, so you get lots of small tight clusters, some of which are pure noise. Set it high and neighborhoods overlap more, so the layout smooths out and retains somewhat more global arrangement. The practical advice is to run several perplexities and only believe structure that appears in all of them, because you can manufacture convincing-looking clusters just by picking perplexity three.

**26. Why is t-SNE non-deterministic?**
Random initialization and stochastic gradient descent. Different runs give different layouts. Setting a seed helps reproduce.

> **Saying it out loud.** Because it starts from a random initialization and optimizes stochastically, so different seeds land in different local minima and the picture looks different every time. Fixing the seed makes a run reproducible, and initializing from PCA instead of randomly makes runs much more consistent with each other and preserves more global structure. This is a genuine problem for anyone using t-SNE plots as evidence. If your conclusion changes when the seed changes, it wasn't a conclusion.

**27. Can you interpret cluster sizes/distances in t-SNE?**

> **Saying it out loud.** No, and this is the most famous misuse of the method. t-SNE only tries to preserve who's near whom, so the size of a blob and the gap between two blobs carry essentially no information; the algorithm will happily inflate a tight cluster and shrink a diffuse one. Two clusters sitting far apart on the plot are not more different than two sitting close. If you need distances to mean something, you need a different tool or you need to go measure in the original space.
**No.** t-SNE preserves local neighborhoods, not global geometry. Cluster sizes and inter-cluster distances are not meaningful. This is a famous misuse.

**28. Does t-SNE have an `inverse_transform`?**
No — and no `transform` either. You can't embed new points without re-running. (UMAP fixes this.)

> **Saying it out loud.** No inverse transform, and no forward transform either. t-SNE optimizes the positions of the specific points you gave it, so there's no learned function to apply to a new point; adding one data point means rerunning the whole optimization. That makes it useless as a preprocessing step in a production pipeline. UMAP fixes exactly this, which is a big part of why it took over.

**29. Computational cost of t-SNE?**
Naive: $O(n^2)$ — pairwise similarities. Barnes-Hut t-SNE: $O(n \log n)$ via tree approximation. Still slow for $n > 10^5$.

> **Saying it out loud.** Naively it's quadratic, because you compute similarities between all pairs. Barnes-Hut t-SNE brings that down to $O(n \log n)$ by approximating the repulsive forces from distant groups of points with a single summary, the same trick used in n-body simulations. Even so, past about a hundred thousand points it's painful. That practical ceiling is the main reason people switch to UMAP on large datasets.

---

## E. UMAP

**30. What's UMAP at a high level?**
Uniform Manifold Approximation and Projection. Builds a fuzzy simplicial complex from k-NN graphs in high-dim, then optimizes a low-dim graph to match. Cross-entropy loss between fuzzy graph memberships.

> **Saying it out loud.** UMAP builds a k-nearest-neighbor graph with fuzzy edge weights, treating that graph as an approximation of the manifold your data lives on, then optimizes a low-dimensional graph to have the same fuzzy connectivity. The loss is a cross-entropy between the two sets of edge memberships, optimized with negative sampling, which is what makes it fast. The mental model to give is: preserve the neighbor graph. The theory it's derived from is heavy but the algorithm is basically a force-directed graph layout.

**31. UMAP vs t-SNE?**
UMAP is faster, more deterministic (with fixed seed), preserves more global structure, has a `transform` method for new points, and scales better. t-SNE often gives nicer-looking local clusters but is slower.

> **Saying it out loud.** UMAP is faster, more reproducible with a fixed seed, keeps more global structure, scales better, and gives you a transform method so you can embed new points. t-SNE is still very good at tight local clusters, so it isn't strictly worse, but there's rarely a strong reason to prefer it now. The default recommendation is UMAP. The one caveat: UMAP's better global structure is better, not correct, so don't start quoting inter-cluster distances.

**32. Key UMAP hyperparameters?**
`n_neighbors` (local vs global tradeoff, like perplexity), `min_dist` (how tightly points pack in low-dim), `metric` (distance function in high-dim).

> **Saying it out loud.** Three that matter. n_neighbors is the local-versus-global dial, the analogue of perplexity, defaulting around fifteen. min_dist controls how tightly points are allowed to clump in the output, so small values give crisp visually-separated clusters and large values spread things out. And metric, the distance function used in the original space, which is easy to forget but matters a lot for text embeddings where you want cosine, not Euclidean. Setting min_dist very small makes your clusters look more separated than they really are.

**33. Can you interpret UMAP cluster distances?**
Better than t-SNE but still fragile. Inter-cluster distances are *somewhat* meaningful but heavily depend on `n_neighbors` and `min_dist`. Don't over-interpret.

> **Saying it out loud.** Somewhat, but don't lean on it. UMAP does keep more global structure than t-SNE, so the coarse arrangement of clusters carries a bit of signal, but it shifts noticeably when you change n_neighbors or min_dist. Anything that moves when you change a hyperparameter is not a measurement. The safe phrasing in an interview: relative positions are a hypothesis to check in the original space, not a result.

**34. UMAP for supervised dim reduction?**
Pass labels in fit. UMAP uses them to bias the graph construction toward separating classes — a supervised embedding.

> **Saying it out loud.** You can hand UMAP the labels at fit time and it uses them to bias the graph, pulling same-class points together and pushing different classes apart. The result is a supervised embedding that separates classes much more cleanly than the unsupervised version. The trap is that this makes for very persuasive-looking pictures which are partly just your labels drawn back at you, so you can't use a supervised UMAP plot as evidence that your classes are separable. There's also a middle setting with a target weight, letting you blend supervised and unsupervised structure.

---

## F. Autoencoders

**35. What's a standard autoencoder?**
Encoder $f_\phi(x) \to z$ (bottleneck), decoder $g_\theta(z) \to \hat{x}$. Train to minimize reconstruction loss $\|x - \hat{x}\|^2$. The bottleneck forces a compressed representation.

> **Saying it out loud.** An autoencoder is a network trained to reproduce its own input through a bottleneck that's too narrow to just pass everything through. The encoder squeezes down to a code, the decoder expands back out, and the loss is reconstruction error. Because the bottleneck is narrow, the network has to learn what's worth keeping. If the bottleneck isn't actually a constraint, the network learns the identity function and you've learned nothing, which is the failure mode to name.

**36. Linear AE vs PCA?**
Identical subspace (up to rotation). The encoder spans the same $k$-dim subspace as the top-$k$ PCs.

> **Saying it out loud.** A linear autoencoder with squared-error loss finds the same subspace as PCA, though not necessarily the same individual axes, since any rotation within the subspace reconstructs equally well. So PCA is the linear autoencoder, plus an ordering and orthogonality that the autoencoder doesn't bother enforcing. Practically, if your autoencoder with linear activations isn't matching PCA's reconstruction error, you have a training bug.

**37. What's a denoising autoencoder?**
Corrupt input ($\tilde{x} = x + \epsilon$ or mask-out), train AE to reconstruct clean $x$ from $\tilde{x}$. Forces the model to learn robust features. Conceptual ancestor of MLM (BERT).

> **Saying it out loud.** A denoising autoencoder gets a corrupted input, by noise or masking, and has to output the clean original. That corruption is what stops it from learning a shortcut identity map, because the identity is now the wrong answer. It has to learn the structure of the data well enough to fill in what's missing. And that's exactly BERT's masked language modeling, a denoising autoencoder over tokens, which is a nice lineage to point out.

**38. What's a VAE?**
Variational autoencoder. Encoder outputs distribution $q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$. Loss = reconstruction + $\mathrm{KL}(q_\phi \| p(z))$ where $p(z) = \mathcal{N}(0, I)$. Generates new samples by drawing $z \sim p(z)$ and decoding.

> **Saying it out loud.** A VAE makes the code a distribution instead of a point: the encoder outputs a mean and a variance, you sample from it, and the decoder reconstructs. The loss has two terms, reconstruction plus a KL that pulls the encoder's distribution toward a standard normal prior. Because the latent space is now shaped like the prior, you can sample a random $z$ and decode it into something plausible, which a plain autoencoder cannot do. The reparameterization trick, writing the sample as mean plus sigma times noise, is what makes it differentiable.

**39. Why the KL term in VAE?**
It regularizes the latent space to match the prior $p(z)$, so we can sample from $p(z)$ and get coherent decodes. Without it, the encoder collapses to a delta — autoencoder, not generative.

> **Saying it out loud.** The KL term is what makes the latent space samplable. Without it, the encoder is free to scatter each input to its own tiny isolated region, and the space between those regions decodes to garbage, so you can't draw a random point and get anything coherent. The KL pulls all the per-input distributions toward a shared standard normal so the space is filled in and continuous. The tension is direct: crank the KL weight up and samples get smooth but blurry, turn it down and reconstructions get sharp but the model stops being generative. That's the beta in beta-VAE.

**40. What's posterior collapse in VAE?**
Decoder ignores $z$; encoder outputs the prior. Common with strong autoregressive decoders (the decoder doesn't need $z$). Fixes: KL annealing, free bits, weakening the decoder.

> **Saying it out loud.** Posterior collapse is when the decoder learns to ignore the latent entirely and the encoder gives up and just outputs the prior, so the KL term goes to zero and $z$ carries no information. It happens when the decoder is powerful enough to model the data on its own, which is why it's endemic with autoregressive text decoders. The standard fixes are KL annealing, where you ramp the KL weight up from zero during training, free bits, which stops penalizing KL below a floor per dimension, or just weakening the decoder. The tell is a KL term that decays to essentially zero early in training.

**41. Why are deep AEs better than PCA on images?**
Convolutional layers exploit spatial structure; non-linearities capture the manifold of natural images, which is far from a linear subspace.

> **Saying it out loud.** Because images are nowhere near a linear subspace. Shift an image one pixel to the right and it's basically the same picture, but in pixel space it's a long way away, and PCA has no way to know that. Convolutional encoders build in translation structure and the nonlinearities let the model follow the curved manifold of natural images. Concretely, PCA on faces gives you those ghostly blurred eigenfaces, while a conv autoencoder at the same code size gives you something recognizable.

---

## G. ICA and NMF

**42. What's ICA?**
Independent Component Analysis. Find a linear transformation $s = Wx$ such that components of $s$ are statistically *independent* (not just uncorrelated). Used for blind source separation (e.g., cocktail party).

> **Saying it out loud.** ICA looks for a linear transformation whose output components are statistically independent, which is much stronger than the uncorrelatedness PCA gives you. It gets there by making the projections as non-Gaussian as possible, using kurtosis or negentropy as the objective. The classic application is the cocktail party problem: several microphones each hear a mixture of several speakers, and ICA separates the voices. The limitation to name is that you can't recover the ordering or the scaling of the sources.

**43. ICA vs PCA?**
PCA: orthogonal components, maximize variance, components uncorrelated. ICA: components statistically independent (stronger), not necessarily orthogonal. Requires non-Gaussianity.

> **Saying it out loud.** PCA gives you orthogonal components that are uncorrelated and ordered by variance. ICA gives you components that are statistically independent, which implies uncorrelated but goes much further, and they don't have to be orthogonal or come in any order. PCA is about compression, ICA is about unmixing. And ICA needs the sources to be non-Gaussian, whereas PCA doesn't care about the distribution at all.

**44. Why does ICA require non-Gaussian sources?**
If sources are all Gaussian, any orthogonal rotation of them is also a valid solution — the problem is unidentifiable. Non-Gaussianity (kurtosis, negentropy) breaks the symmetry.

> **Saying it out loud.** Because a sum of independent Gaussians is Gaussian, and a Gaussian is rotationally symmetric, so if all your sources are Gaussian then any rotation of the solution fits the data exactly as well. The problem is unidentifiable, full stop, and no amount of data helps. Non-Gaussianity breaks that symmetry, which is why ICA algorithms maximize kurtosis or negentropy. The corollary worth stating: at most one of your sources is allowed to be Gaussian.

**45. What's NMF?**
Non-negative Matrix Factorization. Factor $X \approx WH$ with $W, H \geq 0$. Used when data is non-negative (text counts, images, audio spectrograms). Parts-based representation.

> **Saying it out loud.** NMF factors your matrix into two matrices with no negative entries anywhere. That constraint is the whole point: with nothing allowed to cancel, components have to add up to the data, so you get a parts-based decomposition. It applies when your data is naturally non-negative, like word counts, pixel intensities, or audio spectrograms. On a document-term matrix, the components read directly as topics.

**46. NMF vs PCA?**
Both factorize $X \approx LR$. NMF constrains non-negativity, giving additive ("parts") representations. PCA gives signed components (subtractive). NMF is interpretable for topic modeling.

> **Saying it out loud.** Both are low-rank factorizations, but PCA lets entries be negative so components combine by cancellation, while NMF forbids that so components combine additively. That's why NMF components are interpretable as parts or topics and PCA components usually aren't. The costs of NMF are that the optimization is non-convex, so results depend on initialization, and there's no nested structure, meaning your ten-component solution isn't your five-component solution plus five more.

---

## H. Practical decisions

**47. I want to visualize 100k points in 2D — what do I use?**
UMAP. Faster than t-SNE, has transform method, scales better, more deterministic.

> **Saying it out loud.** UMAP. At a hundred thousand points, t-SNE even with Barnes-Hut is slow enough to be annoying, and UMAP handles it comfortably. You also get a transform method so you can project new points later, and much more stability across runs. The one thing I'd add is to run PCA down to about fifty dimensions first, which both speeds it up and makes the nearest-neighbor graph more reliable.

**48. I want to compress 1M images for retrieval — what do I use?**
Trained autoencoder (or even better, a pretrained model's embeddings + product quantization). PCA only as a quick baseline.

> **Saying it out loud.** Don't train an autoencoder from scratch, take embeddings from a pretrained vision model and then compress those. If you need to go smaller, product quantization on top gets you to a handful of bytes per image with fast approximate search. PCA is worth doing as a baseline because it takes minutes and tells you how much of the problem is linear. The reason to prefer pretrained embeddings is that reconstruction loss optimizes for pixels, and pixel fidelity is not what makes retrieval good.

**49. I want interpretable topics from a document-term matrix — what do I use?**
NMF or LDA (Latent Dirichlet Allocation). NMF gives sparse, parts-based topics; LDA is probabilistic.

> **Saying it out loud.** NMF or LDA. NMF gives you sparse, parts-based topics out of the document-term matrix, usually on TF-IDF, and it's fast and deterministic enough to iterate on. LDA is the probabilistic version with a proper generative story and Dirichlet priors, which gives you per-document topic distributions that sum to one. In practice NMF is often the better first move because it's quicker and the topics are frequently more coherent on short documents. Either way you pick the number of topics yourself, and topic coherence is the metric to check.

**50. I want to remove noise from EEG signals — what do I use?**
ICA. The brain signals are statistically independent sources mixed in the recording.

> **Saying it out loud.** ICA. The premise fits perfectly: EEG electrodes each record a linear mixture of underlying sources, and artifacts like eye blinks and heartbeat are statistically independent from brain activity. You run ICA, look at the components, throw out the ones that are clearly blinks, and reconstruct. This is standard practice in neuroscience pipelines, and the manual step of identifying which components are artifacts is the part that hasn't been automated away.

---

## I. Subtleties and gotchas

**51. What's the curse of dimensionality for DR methods?**
In very high dim, all distances become similar — $k$-NN graphs (used by t-SNE, UMAP) become unreliable. Often pre-reduce with PCA first (e.g., PCA → 50 dims → UMAP → 2 dims).

> **Saying it out loud.** In very high dimensions everything is roughly equidistant from everything else, so the nearest-neighbor graphs that t-SNE and UMAP depend on become nearly meaningless. Your neighbors aren't really your neighbors, they're just whichever points won a coin flip. The standard fix is to run PCA down to around fifty dimensions first and then apply UMAP or t-SNE, which both denoises the distances and speeds everything up. That two-stage PCA-then-UMAP pipeline is what practitioners actually do.

**52. Can PCA be used for compression?**
Yes — store $U_k S_k$ (scores) and $V_k$ (loadings). Reconstruct with $X_k = U_k S_k V_k^\top$. Used in image compression (JPEG-style ideas), genomics, etc.

> **Saying it out loud.** Yes, and it's a decent lossy compressor for correlated data. You keep the scores and the loadings, so instead of $n \times d$ numbers you store $k(n+d)$, and reconstruct by multiplying them back. It only wins when $k$ is much smaller than $d$, which requires the features to be genuinely correlated. It's the same family of ideas as the DCT in JPEG, though JPEG uses a fixed basis rather than one learned from your data.

**53. What's the relationship between PCA and spectral clustering?**
Spectral clustering = eigendecomposition of a graph Laplacian, then K-means on eigenvectors. PCA = eigendecomposition of covariance. Both are spectral methods on a kernel/affinity matrix.

> **Saying it out loud.** Both are spectral methods, they just eigendecompose different matrices. PCA takes the eigenvectors of the covariance; spectral clustering takes the eigenvectors of a graph Laplacian built from an affinity matrix, then runs k-means on them. So both are answering "what are the dominant modes of this matrix," one for a covariance and one for a graph. The practical difference is that spectral clustering can separate non-convex shapes like two interlocking crescents, which PCA plus k-means cannot.

**54. PCA with missing data?**
Naive PCA fails. Use probabilistic PCA + EM, or matrix completion (low-rank methods). Or impute first (mean, k-NN) then PCA.

> **Saying it out loud.** Plain PCA has no notion of a missing entry, so it just fails. Your options are probabilistic PCA fit with EM, which handles missingness natively as part of the model, low-rank matrix completion, which is the same idea Netflix-style recommenders use, or imputing first with a mean or k-NN fill and then running ordinary PCA. Imputing first is the quick answer and it biases your covariance toward whatever you filled in. If a lot of data is missing, EM-based PPCA is the principled choice.

**55. Whitening — what is it and why?**
Project to PC space and scale by $1/\sqrt{\lambda_i}$ — output has identity covariance. Removes scale and correlation. Used as preprocessing (e.g., in ICA).

> **Saying it out loud.** Whitening means projecting into PC space and then dividing each component by the square root of its eigenvalue, so every direction ends up with unit variance and the covariance becomes the identity. You've removed both correlation and scale, leaving only the shape. It's the standard preprocessing step before ICA, because once the data is whitened the remaining unmixing is just a rotation. The danger is that dividing by tiny eigenvalues massively amplifies noise directions, which is why you regularize by adding a small epsilon.

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
