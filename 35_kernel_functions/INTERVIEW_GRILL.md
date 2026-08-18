# Kernel Methods — Interview Grill

> 35 questions on kernels and kernel SVM. Drill until you can answer 25+ cold.

---

## A. Foundations

**In plain terms.** A kernel is a similarity score between two data points that secretly equals a dot product in a much richer space of features. You get all the power of that richer space — curved decision boundaries, feature interactions — while only ever computing a cheap formula on the original data.

**1. What's the kernel trick?**
A kernel function $k(x, x')$ computes an inner product in some (possibly infinite-dimensional) feature space without ever computing the features explicitly. Replace $x^\top x'$ with $k(x, x')$ in any algorithm that only accesses data via inner products.

> **Saying it out loud.** The kernel trick is a shortcut that lets you work in a huge feature space without ever going there. Here's the analogy I like: imagine you want to know how similar two songs are in terms of every possible combination of their notes. You could write out that gigantic list of combinations for each song and compare them — expensive — or you could find a formula that gives you the same similarity number directly from the two songs. That formula is the kernel. So any algorithm that only ever touches your data through dot products can be made non-linear for free, just by swapping the dot product for a kernel. The cost you pay is that you never see the features, so you lose interpretability, and you have to compute similarities between all pairs of points, which is quadratic.

**2. State Mercer's theorem.**
A function $k(x, x')$ is a valid kernel iff: (a) symmetric ($k(x, x') = k(x', x)$), (b) positive semi-definite (the kernel matrix $K_{ij} = k(x_i, x_j)$ has all eigenvalues $\geq 0$ for any finite set). Equivalently: there exists a feature map $\phi$ with $k(x, x') = \phi(x)^\top \phi(x')$.

> **Saying it out loud.** Mercer's theorem tells you which similarity functions are legal. Two conditions: it has to be symmetric, and the matrix of pairwise similarities has to be positive semi-definite, meaning no negative eigenvalues, for any set of points you pick. If those hold, the theorem guarantees there exists some feature space where your kernel is a genuine dot product — you never have to construct that space, you just get to know it exists. That's what licenses the whole trick. The practical way to use it is that you rarely check from scratch; you build new kernels out of known-valid ones, since sums and products of valid kernels stay valid.

**3. Why does PSD matter?**
PSD guarantees the kernel corresponds to a real inner product in some feature space. Non-PSD "kernels" don't correspond to any feature space — methods built on them lose their theoretical grounding (though sometimes work empirically, like sigmoid).

> **Saying it out loud.** Positive semi-definiteness is what guarantees you're actually computing a dot product in some space rather than an arbitrary similarity score. That matters because everything downstream rests on it: the SVM's optimization is convex only if the kernel matrix is PSD, so without it you can have multiple local optima and the solver can fail to converge. You also lose the geometric interpretation — the notion of margin stops meaning anything. The famous exception is the sigmoid kernel, which isn't PSD for all parameter settings and yet people used it anyway because it sometimes worked, which is a good illustration of theory versus practice.

**4. Why is the kernel trick useful?**
Allows non-linear methods that operate in high-dimensional feature spaces without explicit computation. Linear method + kernel = non-linear method.

> **Saying it out loud.** Because it turns any linear algorithm into a non-linear one with a one-line change. Linear methods are well understood, convex, and fast, but they can only draw straight boundaries. The kernel trick lets you keep all that machinery while the boundary curves arbitrarily in the original space, because it's straight in an implicit high-dimensional space. The classic picture is two concentric rings of points — impossible to separate with a line in two dimensions, trivial once you add the squared radius as a third dimension. The tradeoff is that cost now scales with the number of data points rather than the number of features, which is exactly backwards for big data.

---

## B. Specific kernels

**5. Linear kernel?**
$k(x, x') = x^\top x'$. Just dot product. Equivalent to no kernel — used as a baseline.

> **Saying it out loud.** The linear kernel is just the ordinary dot product — no transformation at all, so kernel SVM with a linear kernel is plain linear SVM. It's not useless, though: it's the right choice when your data is already high-dimensional and roughly linearly separable, which is exactly the case for text with bag-of-words features. It's also dramatically faster, since you can solve in the primal with cost linear in the number of samples rather than quadratic. My rule is: try linear first, and only reach for RBF if linear underfits.

**6. Polynomial kernel?**
$k(x, x') = (x^\top x' + c)^d$. Implicit feature space contains all monomials up to degree $d$. For $d = 2$: includes products like $x_1 x_2$.

> **Saying it out loud.** The polynomial kernel takes the dot product, adds a constant, and raises it to a power. What that buys you implicitly is every product of features up to that degree — for degree two you get all the pairwise interaction terms, so the model can learn "this matters only when that is also high." The constant controls the balance between high-order and low-order terms. In practice degree two or three is what people use, because higher degrees make the kernel values explode numerically and the model overfit badly. It's the natural choice when you believe feature interactions matter and you want to bound how complex they get.

**7. RBF (Gaussian) kernel?**
$k(x, x') = \exp(-\gamma \|x - x'\|^2)$. Most popular. Infinite-dimensional implicit feature space. $\gamma$ controls bandwidth.

> **Saying it out loud.** The RBF kernel measures similarity as a bump that decays with distance — two points close together get a value near one, far apart get near zero. Gamma sets how fast that decay happens, so it's an inverse width: small gamma means each point influences a wide region and you get a smooth boundary, large gamma means each point only speaks for its immediate neighborhood and the boundary gets wiggly. It's the default non-linear kernel because it makes almost no assumption about the shape of the boundary and it's universal — it can approximate any decision boundary given enough data. The failure mode is that with gamma too large you effectively memorize the training set.

**8. Why is RBF infinite-dimensional?**
Factor $\exp(-\gamma\|x-x'\|^2) = \exp(-\gamma\|x\|^2)\exp(-\gamma\|x'\|^2)\exp(2\gamma\, x^\top x')$. Taylor-expand the cross term: $\exp(2\gamma\, x^\top x') = \sum_{n=0}^\infty (2\gamma\, x^\top x')^n / n!$. Each $(x^\top x')^n$ equals an inner product of all degree-$n$ polynomial features of $x$ and $x'$ — so RBF is an inner product over polynomial features of *all* degrees, hence infinite-dim.

> **Saying it out loud.** Take the RBF formula and factor it into terms depending on each point separately, times an exponential of their dot product. Now Taylor-expand that last exponential and you get an infinite sum of powers of the dot product. Each power corresponds to polynomial features of that degree — so RBF is secretly a polynomial kernel of *every* degree at once, with weights that decay factorially. That's the derivation, and it's why RBF's implicit feature space is infinite-dimensional. The reassuring part is that you never compute any of it: the regularization means the effective complexity is controlled by gamma and C, not by the dimension.

**9. Cosine kernel?**
$k(x, x') = (x^\top x') / (\|x\| \|x'\|)$. Normalized inner product. Used in NLP/IR.

> **Saying it out loud.** The cosine kernel is the dot product divided by both magnitudes, so it measures the angle between two vectors and ignores their lengths entirely. That's exactly what you want in text and information retrieval, where a long document and a short document about the same topic should count as similar — length shouldn't be the signal. It's the same thing as using a linear kernel on L2-normalized inputs, which is often how it's implemented. It's the standard similarity for embedding search today, which is a nice bridge to make.

**10. Sigmoid kernel?**
$k(x, x') = \tanh(\alpha x^\top x' + c)$. Inspired by NN. Not always PSD, but used in practice.

> **Saying it out loud.** The sigmoid kernel applies a tanh to a scaled dot product, and it was proposed because it makes an SVM look like a two-layer neural network. Its real claim to fame is as a cautionary tale: it isn't positive semi-definite for all parameter values, so the optimization isn't guaranteed convex and the theory doesn't apply — yet it sometimes performed fine, which is why it stuck around. In practice I wouldn't use it; RBF does the job with better guarantees, and if you actually want a neural network, train a neural network.

---

## C. Kernel SVM

**In plain terms.** An SVM finds the boundary that leaves the widest possible gap between the two classes. The equations below are two views of the same problem: the primal solves for the boundary directly, the dual solves for a weight on each training point — and the dual is the version you can plug a kernel into.

**11. SVM primal formulation?**

$$
\min_w \tfrac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b))
$$

Hinge loss + L2 penalty. $C$ controls margin softness.

> **Saying it out loud.** The primal is the version you'd write down naturally: minimize the squared norm of the weights, which maximizes the margin, plus C times the hinge loss for points that violate it. The two pieces trade off directly — the norm term wants a wide, simple boundary, the hinge term wants every point correctly classified with room to spare. C is the exchange rate between them. It's also worth noticing this is just L2-regularized hinge loss, which puts SVM in the same family as regularized logistic regression, differing only in the loss shape.

**12. SVM dual formulation?**

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^\top x_j)
$$

subject to $0 \leq \alpha_i \leq C, \sum_i \alpha_i y_i = 0$. Decision function: $f(x) = \sum_i \alpha_i y_i (x_i^\top x) + b$.

> **Saying it out loud.** The dual rewrites the same problem in terms of one weight per training point instead of one per feature. Two things make it interesting. First, the data appears only inside dot products between pairs of points — which is precisely what lets you swap in a kernel. Second, the solution turns out to be sparse: most of those per-point weights come out exactly zero, and the ones that don't are the support vectors. The constraint that C bounds each weight from above is what implements the soft margin, capping how much influence any single point can have.

**13. Why does the dual enable kernels?**
The dual only accesses data through inner products $x_i^\top x_j$. Replace with $k(x_i, x_j)$ — kernel SVM. Primal can't be kernelized directly because $w$ would live in the implicit feature space (infinite-dim for RBF).

> **Saying it out loud.** Because the dual only ever touches the data through pairwise dot products, so you can replace each one with a kernel evaluation and everything else goes through unchanged. The primal can't do that, because the primal explicitly carries a weight vector living in the feature space — and for an RBF kernel that space is infinite-dimensional, so you literally cannot store it. That's the whole reason the dual matters, and it's the crispest way to answer why we bother with duality at all. The cost is that the dual has one variable per data point, which is why kernel SVMs scale with the dataset rather than the dimension.

**14. What are support vectors?**
Training points with non-zero $\alpha_i$ in the dual. Geometrically: points on or inside the margin. Decision function depends only on them.

> **Saying it out loud.** Support vectors are the training points that actually determine the boundary — the ones sitting on the margin or violating it. Everything else has a dual weight of exactly zero and contributes nothing. Geometrically, they're the points closest to the frontier between classes, so they're the hard cases. The consequence people find surprising is that you could delete every non-support-vector from the training set, retrain, and get the identical model.

**15. Why are support vectors interesting?**
Sparsity: most $\alpha_i = 0$, so the model only "remembers" support vectors. Inference: $O(|\text{SV}|)$. Also: support vectors are robust — adding non-SV points doesn't change the model.

> **Saying it out loud.** Two reasons. Practically, sparsity means inference costs scale with the number of support vectors rather than the whole training set, which is what makes SVM usable at prediction time. Conceptually, it means the model is defined entirely by the hard examples near the boundary, which is a nice form of robustness — adding more easy points changes nothing. The flip side, and the thing to name as a failure mode, is that if a large fraction of your points become support vectors, that's a signal you're overfitting or your kernel parameters are wrong.

**16. What does $C$ control in SVM?**
Soft-margin parameter. Large $C$: hard margin, less regularization, can overfit. Small $C$: more slack allowed, more regularization, more support vectors.

> **Saying it out loud.** C is how much you care about classifying training points correctly versus keeping the boundary simple. Large C says every mistake is expensive, so the boundary contorts to fit the training data — hard margin, low regularization, overfitting risk. Small C tolerates violations, giving you a wide, smooth margin and more support vectors. The rule of thumb worth stating is that C is inversely a regularization strength, and you search it on a log scale from about ten to the minus three up to ten to the three, because the sensible value spans orders of magnitude.

**17. What does $\gamma$ in RBF SVM control?**
Bandwidth. Small $\gamma$: smooth decision boundary, low complexity. Large $\gamma$: wiggly boundary, high complexity, overfitting risk.

> **Saying it out loud.** Gamma is the reach of each training point. Small gamma means each point's influence extends far, so the boundary is smooth and the model is close to linear — underfitting territory. Large gamma means influence is local, so the boundary can wrap tightly around individual points, and at the extreme you get islands around every training example, which is memorization. The way I'd frame the interaction: gamma controls the complexity of the function class and C controls how hard you push to fit the data, so they can compensate for each other, which is exactly why you have to search them jointly rather than one at a time.

**18. How do you tune SVM hyperparameters?**
Grid search over $C$ (log-scale, e.g., $10^{-3}$ to $10^3$) and $\gamma$ (also log-scale). Use cross-validation. SVMs are notoriously sensitive to these choices.

> **Saying it out loud.** Grid search on a log scale for both, with cross-validation, and jointly rather than one at a time — because C and gamma interact, tuning one at a fixed value of the other will land you in the wrong place. The usual ranges are ten to the minus three through ten to the three for C, and something around one over the number of features times the variance as a starting scale for gamma. And you must standardize the features first, because gamma's meaning depends on the scale of the distances. SVMs are notoriously sensitive here — the difference between a well-tuned and badly-tuned RBF SVM can be twenty points of accuracy.

---

## D. Other kernel methods

**19. Kernel ridge regression?**
$\hat f(x) = \sum_i \alpha_i k(x_i, x)$ with $\alpha = (K + \lambda I)^{-1} y$. Closed-form. Kernel version of linear ridge.

> **Saying it out loud.** Kernel ridge regression is ridge regression with the kernel trick, and its appeal is that the solution is closed-form: invert the kernel matrix plus lambda times the identity, multiply by the targets. No iteration, no local optima. The lambda term does double duty — it regularizes and it makes the matrix invertible. The catch is that inverting an N-by-N matrix costs N cubed, so past ten thousand points or so you need approximations like Nyström or random features. Unlike SVM it isn't sparse: every training point gets a non-zero weight, so inference costs scale with the entire training set.

**20. What's a Gaussian process?**
Bayesian extension of kernel ridge. $f \sim \mathrm{GP}(0, k)$. Posterior given data is Gaussian with mean $\mu(x) = k(x, X)(K + \sigma^2 I)^{-1} y$ and variance from kernel structure. Provides uncertainty estimates.

> **Saying it out loud.** A Gaussian process is the Bayesian version of kernel regression: instead of learning one function, you put a distribution over functions and condition it on your data. The kernel plays the role of the prior, encoding how smooth you think the function is and how quickly correlations decay with distance. The payoff is that predictions come with honest uncertainty — the variance grows automatically in regions where you have no data, which is exactly what you need for Bayesian optimization and active learning. The cost is the same N-cubed inversion, which is why GPs are the right tool below about ten thousand points and not above.

**21. Kernel PCA?**
PCA in feature space via the kernel trick. Compute kernel matrix; eigendecompose; extract top components. Useful for non-linear dimensionality reduction.

> **Saying it out loud.** Kernel PCA runs PCA in the implicit feature space, so instead of finding straight directions of maximum variance, it finds curved ones. Mechanically you center the kernel matrix and eigendecompose it, and the components come out as combinations of kernel evaluations. It's genuinely useful for non-linear dimensionality reduction — it can unroll a curved manifold that ordinary PCA would flatten wrong. The awkward part is the pre-image problem: you can project into the component space but mapping back to the original space isn't well defined, so reconstruction is approximate at best.

**22. Kernel k-means?**
K-means in feature space. Equivalent to spectral clustering for some kernels. Cluster centers are linear combinations of $\phi(x_i)$.

> **Saying it out loud.** Kernel k-means runs the same alternating algorithm but computes distances in feature space, which lets it find clusters that aren't spherical blobs — concentric rings, for example. You never form the centroids explicitly, since they live in the implicit space; you express distances entirely through kernel values. It's closely related to spectral clustering, and for certain kernel choices they're equivalent, which is a nice connection to volunteer. The costs are the usual ones: quadratic memory for the kernel matrix, plus you still have to pick k and you're still sensitive to initialization.

---

## E. Theory and RKHS

**23. What's an RKHS?**
Reproducing Kernel Hilbert Space. A function space $\mathcal{H}_k$ where each kernel evaluation $k(\cdot, x)$ is a function in $\mathcal{H}_k$, and inner product $\langle f, k(\cdot, x) \rangle = f(x)$ — the reproducing property.

> **Saying it out loud.** An RKHS is the function space that a kernel implicitly defines — the set of functions you can build as combinations of kernel evaluations. The property it's named for is the reproducing property: evaluating a function at a point is the same as taking its inner product with the kernel centered at that point. That sounds like a technicality but it's the engine of the whole theory, because it means point evaluation is a continuous operation, which is exactly what lets you reason about function values with inner-product tools.

**24. Why does RKHS matter?**
Provides a unified theoretical framework for kernel methods. Functions learned by SVM, kernel ridge, GP all live in the RKHS. Regularization with $\|f\|_{\mathcal{H}_k}^2$ explains why kernel methods don't overfit despite operating in infinite-dim spaces.

> **Saying it out loud.** RKHS is what explains why kernel methods don't blow up despite operating in infinite-dimensional spaces. The answer is that you're not searching all of that space — you're penalizing the RKHS norm, which is a smoothness measure, so the effective complexity is controlled by regularization rather than by dimension. That's a genuinely satisfying resolution to what looks like a paradox, and it's the same idea that later showed up in explaining why overparameterized neural networks generalize. It also unifies the field: SVM, kernel ridge and Gaussian processes are all solving regularized problems in the same space with different loss functions.

**25. What's the representer theorem?**
For many regularized kernel methods, the optimal solution has the form $f^*(x) = \sum_i \alpha_i k(x_i, x)$ — i.e., a linear combination of kernel evaluations at training points. This is why kernel methods are tractable: the solution is always finite-dimensional.

> **Saying it out loud.** The representer theorem is the result that makes kernel methods computable at all. It says that for a broad class of regularized problems, the optimal function — even though it lives in a possibly infinite-dimensional space — can always be written as a finite weighted sum of kernel evaluations at your training points. So an infinite-dimensional optimization collapses to finding N numbers. That's why you never have to touch the feature map. The condition is that the regularizer be an increasing function of the RKHS norm, which covers essentially every method people actually use.

---

## F. Kernels vs deep learning

**26. Why did kernels lose to deep learning?**
$O(N^2)$ memory and $O(N^2)$–$O(N^3)$ training. Fixed kernels (no representation learning). Deep learning scales linearly and learns features end-to-end.

> **Saying it out loud.** Scale, in two ways. Computationally, kernel methods need the pairwise kernel matrix, which is quadratic in memory and up to cubic in training time, so a million points is out of reach where a neural network just does more gradient steps. Statistically, the kernel is fixed — you choose RBF or polynomial in advance, and the model can't learn a better notion of similarity from the data. Deep learning learns the representation, which is exactly what matters for images and text where raw pixel or token distance is meaningless. That's the crisp version: kernels have a fixed similarity function and linear-in-features cost, deep nets learn similarity and scale linearly in data.

**27. Where do kernels still win?**
Small data ($N < 10^4$). Bayesian uncertainty (GPs). Theoretical analysis (NTK). Tabular tasks where SVM-RBF is the right capacity.

> **Saying it out loud.** Three places. Small data — under about ten thousand points — where kernel methods are competitive, need almost no tuning of architecture, and train in seconds. Uncertainty quantification, where Gaussian processes remain the cleanest tool and underpin Bayesian optimization for hyperparameter search. And theory, where the neural tangent kernel gives you a tractable model of what infinitely wide networks do. There's a fourth practical one: on modest tabular problems, an RBF SVM is often within a point of a neural net and takes a tenth of the effort, though gradient-boosted trees usually beat both.

**28. What's the Neural Tangent Kernel (NTK)?**
Jacot et al. 2018. In the infinite-width limit, deep NNs behave as kernel ridge regression with the NTK: $k_{\text{NTK}}(x, x') = \mathbb{E}[\langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle]$. Bridges kernels and deep learning theoretically.

> **Saying it out loud.** The NTK says that in the limit of infinite width, a neural network trained by gradient descent behaves exactly like kernel regression with a specific fixed kernel — one defined by the inner product of the network's gradients at initialization. The reason that's remarkable is that it makes deep learning analytically tractable: training becomes a convex problem with a closed-form solution, and you can prove convergence and generalization results. The catch, which you should state, is that in this regime the features never change — the network doesn't learn representations, which is precisely the thing that makes real deep learning work.

**29. NTK in practice?**
Theoretical lens. Real NNs at finite width and after training don't behave purely as NTK — feature learning happens. NTK is useful for theory, less for practice.

> **Saying it out loud.** It's a theoretical lens, not a practical method. The NTK regime requires the parameters to barely move from initialization, which happens at infinite width but not at real widths — real networks change their internal representations substantially during training, and that feature learning is where most of the performance comes from. Empirically, finite networks outperform their NTK equivalents on hard tasks, and the gap grows with task difficulty. So the honest framing is that NTK is a valuable model of a limiting case that tells us what deep learning would be if it didn't learn features, which usefully isolates how much feature learning is worth.

---

## G. Connection to attention

**30. Is attention a kernel?**
Yes, conceptually. Attention computes $\sum_j (\exp(q_i^\top k_j / \sqrt{d_k}) / Z) v_j$ — a query attends to keys via a kernel-like similarity, then weighted-averages values. The "kernel" $\exp(q^\top k / \sqrt{d_k})$ is **learned** via $W_Q, W_K$.

> **Saying it out loud.** Yes, and it's a clean way to describe attention. Each query computes a similarity with every key, softmaxes those into weights, and returns a weighted average of the values — that's kernel-weighted regression, with the exponential of the scaled dot product playing the role of the kernel. The framing makes attention feel much less mysterious: it's a lookup where similar things contribute more, which is exactly what kernel smoothing does. The important difference is that the similarity function isn't chosen by you; it's parameterized by the query and key projection matrices and learned from data.

**31. What's the difference between attention and classical kernels?**
Classical kernels are fixed (you choose RBF, polynomial, etc.). Attention is **learned** — the similarity function depends on $W_Q, W_K$ which the model trains. This makes attention task-adaptive, classical kernels not.

> **Saying it out loud.** Classical kernels are fixed and chosen by a human before training — you pick RBF, you pick gamma, and that's the notion of similarity forever. Attention learns its similarity: the query and key matrices are trained, so the model figures out for itself which dimensions of the representation should count as "similar" for this task, and different heads learn different notions simultaneously. That's the whole advantage, and it's the same story as hand-crafted features versus learned features one level up. Attention's kernel is also asymmetric and not necessarily positive semi-definite, so it isn't a Mercer kernel in the strict sense.

**32. Implications of viewing attention as kernel?**
Theoretical unification. Attention can be analyzed using kernel-method tools. Recent research uses kernel theory to understand transformer behavior. Frontier-lab interview-relevant.

> **Saying it out loud.** Mostly it buys you analysis tools. If attention is kernel smoothing, you can bring RKHS theory, generalization bounds and approximation results to bear on transformers, which is otherwise hard. It also directly motivated linear attention: if the softmax kernel is what makes attention quadratic, replace it with a kernel that factorizes — random feature maps, as in Performer — and you get linear-time attention. That's a real practical payoff from a theoretical reframing, which is why it's worth being able to make the connection out loud.

---

## H. Subtleties

**33. Why does kernel SVM need to scale features?**
RBF and polynomial kernels are sensitive to feature scales. $\|x - x'\|^2$ depends on raw feature magnitudes. Standardize or normalize features before fitting.

> **Saying it out loud.** Because the kernel is computed from distances, and distances are dominated by whatever feature happens to have the largest units. If one feature is income in dollars and another is age in years, the income differences are thousands of times larger, so the RBF kernel essentially ignores age entirely. A single gamma applies to all features, so there's no way for the model to compensate. Standardize to zero mean and unit variance, or min-max scale, before fitting — and fit the scaler on the training set only. This is the single most common reason a kernel SVM performs badly for no apparent reason.

**34. Curse of dimensionality for kernels?**
In high-dim spaces, distances become uniform (all points become equidistant). RBF kernel can degenerate. Kernels work best at moderate dimensionality with enough data per dimension.

> **Saying it out loud.** In high dimensions, the distance between any two random points concentrates — everything ends up roughly equidistant, so the ratio between the nearest and farthest neighbor approaches one. That guts the RBF kernel, since its whole job is to distinguish near from far: all the kernel values collapse toward a single value and the similarity carries no information. The practical symptom is that either everything looks similar or, with large gamma, nothing does. The mitigations are dimensionality reduction first, feature selection, or using a kernel with a milder distance dependence — but past a few hundred meaningful dimensions, learned representations are the better answer.

**35. Can you combine kernels?**
Yes. Sum, product, weighted combination, etc. of valid kernels are valid kernels. "Multiple kernel learning" optimizes the combination weights.

> **Saying it out loud.** Yes, and it's one of the most useful facts about kernels. Sums of valid kernels are valid, products are valid, and multiplying by a positive constant is valid — so you can build compound kernels that express structure, like a periodic kernel times an RBF for something seasonal with drift. Gaussian process practitioners do this constantly as a way of encoding domain knowledge. Multiple kernel learning goes further and learns the combination weights from data, typically with a sparsity penalty so the model picks which kernels matter. The tradeoff is more hyperparameters and more compute for usually modest gains, which is why it stayed niche.

---

## Quick fire

**36.** *RBF formula?* $\exp(-\gamma \|x - x'\|^2)$.
**37.** *Polynomial degree typical?* 2 or 3.
**38.** *SVM dual scales as?* $O(N^2)$ memory, $O(N^2)$–$O(N^3)$ time.
**39.** *Mercer's conditions?* Symmetric + PSD.
**40.** *Connection to attention?* Attention = learned kernel.

---

## Self-grading

If you can't answer 1-15, you don't know kernels. If you can't answer 16-30, you'll struggle on classical ML interviews involving SVM. If you can't answer 31-40, frontier-lab interviews on kernel-attention connections will go past you.

Aim for 25+/40 cold.
