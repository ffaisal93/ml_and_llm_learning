# Kernel Methods — Interview Grill

> 35 questions on kernels and kernel SVM. Drill until you can answer 25+ cold.

---

## A. Foundations

**1. What's the kernel trick?**
A kernel function $k(x, x')$ computes an inner product in some (possibly infinite-dimensional) feature space without ever computing the features explicitly. Replace $x^\top x'$ with $k(x, x')$ in any algorithm that only accesses data via inner products.

**2. State Mercer's theorem.**
A function $k(x, x')$ is a valid kernel iff: (a) symmetric ($k(x, x') = k(x', x)$), (b) positive semi-definite (the kernel matrix $K_{ij} = k(x_i, x_j)$ has all eigenvalues $\geq 0$ for any finite set). Equivalently: there exists a feature map $\phi$ with $k(x, x') = \phi(x)^\top \phi(x')$.

**3. Why does PSD matter?**
PSD guarantees the kernel corresponds to a real inner product in some feature space. Non-PSD "kernels" don't correspond to any feature space — methods built on them lose their theoretical grounding (though sometimes work empirically, like sigmoid).

**4. Why is the kernel trick useful?**
Allows non-linear methods that operate in high-dimensional feature spaces without explicit computation. Linear method + kernel = non-linear method.

---

## B. Specific kernels

**5. Linear kernel?**
$k(x, x') = x^\top x'$. Just dot product. Equivalent to no kernel — used as a baseline.

**6. Polynomial kernel?**
$k(x, x') = (x^\top x' + c)^d$. Implicit feature space contains all monomials up to degree $d$. For $d = 2$: includes products like $x_1 x_2$.

**7. RBF (Gaussian) kernel?**
$k(x, x') = \exp(-\gamma \|x - x'\|^2)$. Most popular. Infinite-dimensional implicit feature space. $\gamma$ controls bandwidth.

**8. Why is RBF infinite-dimensional?**
Factor $\exp(-\gamma\|x-x'\|^2) = \exp(-\gamma\|x\|^2)\exp(-\gamma\|x'\|^2)\exp(2\gamma\, x^\top x')$. Taylor-expand the cross term: $\exp(2\gamma\, x^\top x') = \sum_{n=0}^\infty (2\gamma\, x^\top x')^n / n!$. Each $(x^\top x')^n$ equals an inner product of all degree-$n$ polynomial features of $x$ and $x'$ — so RBF is an inner product over polynomial features of *all* degrees, hence infinite-dim.

**9. Cosine kernel?**
$k(x, x') = (x^\top x') / (\|x\| \|x'\|)$. Normalized inner product. Used in NLP/IR.

**10. Sigmoid kernel?**
$k(x, x') = \tanh(\alpha x^\top x' + c)$. Inspired by NN. Not always PSD, but used in practice.

---

## C. Kernel SVM

**11. SVM primal formulation?**

$$
\min_w \tfrac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b))
$$

Hinge loss + L2 penalty. $C$ controls margin softness.

**12. SVM dual formulation?**

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^\top x_j)
$$

subject to $0 \leq \alpha_i \leq C, \sum_i \alpha_i y_i = 0$. Decision function: $f(x) = \sum_i \alpha_i y_i (x_i^\top x) + b$.

**13. Why does the dual enable kernels?**
The dual only accesses data through inner products $x_i^\top x_j$. Replace with $k(x_i, x_j)$ — kernel SVM. Primal can't be kernelized directly because $w$ would live in the implicit feature space (infinite-dim for RBF).

**14. What are support vectors?**
Training points with non-zero $\alpha_i$ in the dual. Geometrically: points on or inside the margin. Decision function depends only on them.

**15. Why are support vectors interesting?**
Sparsity: most $\alpha_i = 0$, so the model only "remembers" support vectors. Inference: $O(|\text{SV}|)$. Also: support vectors are robust — adding non-SV points doesn't change the model.

**16. What does $C$ control in SVM?**
Soft-margin parameter. Large $C$: hard margin, less regularization, can overfit. Small $C$: more slack allowed, more regularization, more support vectors.

**17. What does $\gamma$ in RBF SVM control?**
Bandwidth. Small $\gamma$: smooth decision boundary, low complexity. Large $\gamma$: wiggly boundary, high complexity, overfitting risk.

**18. How do you tune SVM hyperparameters?**
Grid search over $C$ (log-scale, e.g., $10^{-3}$ to $10^3$) and $\gamma$ (also log-scale). Use cross-validation. SVMs are notoriously sensitive to these choices.

---

## D. Other kernel methods

**19. Kernel ridge regression?**
$\hat f(x) = \sum_i \alpha_i k(x_i, x)$ with $\alpha = (K + \lambda I)^{-1} y$. Closed-form. Kernel version of linear ridge.

**20. What's a Gaussian process?**
Bayesian extension of kernel ridge. $f \sim \mathrm{GP}(0, k)$. Posterior given data is Gaussian with mean $\mu(x) = k(x, X)(K + \sigma^2 I)^{-1} y$ and variance from kernel structure. Provides uncertainty estimates.

**21. Kernel PCA?**
PCA in feature space via the kernel trick. Compute kernel matrix; eigendecompose; extract top components. Useful for non-linear dimensionality reduction.

**22. Kernel k-means?**
K-means in feature space. Equivalent to spectral clustering for some kernels. Cluster centers are linear combinations of $\phi(x_i)$.

---

## E. Theory and RKHS

**23. What's an RKHS?**
Reproducing Kernel Hilbert Space. A function space $\mathcal{H}_k$ where each kernel evaluation $k(\cdot, x)$ is a function in $\mathcal{H}_k$, and inner product $\langle f, k(\cdot, x) \rangle = f(x)$ — the reproducing property.

**24. Why does RKHS matter?**
Provides a unified theoretical framework for kernel methods. Functions learned by SVM, kernel ridge, GP all live in the RKHS. Regularization with $\|f\|_{\mathcal{H}_k}^2$ explains why kernel methods don't overfit despite operating in infinite-dim spaces.

**25. What's the representer theorem?**
For many regularized kernel methods, the optimal solution has the form $f^*(x) = \sum_i \alpha_i k(x_i, x)$ — i.e., a linear combination of kernel evaluations at training points. This is why kernel methods are tractable: the solution is always finite-dimensional.

---

## F. Kernels vs deep learning

**26. Why did kernels lose to deep learning?**
$O(N^2)$ memory and $O(N^2)$–$O(N^3)$ training. Fixed kernels (no representation learning). Deep learning scales linearly and learns features end-to-end.

**27. Where do kernels still win?**
Small data ($N < 10^4$). Bayesian uncertainty (GPs). Theoretical analysis (NTK). Tabular tasks where SVM-RBF is the right capacity.

**28. What's the Neural Tangent Kernel (NTK)?**
Jacot et al. 2018. In the infinite-width limit, deep NNs behave as kernel ridge regression with the NTK: $k_{\text{NTK}}(x, x') = \mathbb{E}[\langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle]$. Bridges kernels and deep learning theoretically.

**29. NTK in practice?**
Theoretical lens. Real NNs at finite width and after training don't behave purely as NTK — feature learning happens. NTK is useful for theory, less for practice.

---

## G. Connection to attention

**30. Is attention a kernel?**
Yes, conceptually. Attention computes $\sum_j (\exp(q_i^\top k_j / \sqrt{d_k}) / Z) v_j$ — a query attends to keys via a kernel-like similarity, then weighted-averages values. The "kernel" $\exp(q^\top k / \sqrt{d_k})$ is **learned** via $W_Q, W_K$.

**31. What's the difference between attention and classical kernels?**
Classical kernels are fixed (you choose RBF, polynomial, etc.). Attention is **learned** — the similarity function depends on $W_Q, W_K$ which the model trains. This makes attention task-adaptive, classical kernels not.

**32. Implications of viewing attention as kernel?**
Theoretical unification. Attention can be analyzed using kernel-method tools. Recent research uses kernel theory to understand transformer behavior. Frontier-lab interview-relevant.

---

## H. Subtleties

**33. Why does kernel SVM need to scale features?**
RBF and polynomial kernels are sensitive to feature scales. $\|x - x'\|^2$ depends on raw feature magnitudes. Standardize or normalize features before fitting.

**34. Curse of dimensionality for kernels?**
In high-dim spaces, distances become uniform (all points become equidistant). RBF kernel can degenerate. Kernels work best at moderate dimensionality with enough data per dimension.

**35. Can you combine kernels?**
Yes. Sum, product, weighted combination, etc. of valid kernels are valid kernels. "Multiple kernel learning" optimizes the combination weights.

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
