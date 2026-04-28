# Kernel Methods: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Kernels are still asked in classical ML interviews, especially for SVM-related questions. The kernel trick is one of the most beautiful results in classical ML and a common interview probe ("why does the dual formulation enable kernels?"). Understanding kernels also clarifies what attention is doing — the attention $QK^\top$ is essentially a learned kernel.

---

## 1. The kernel trick in one sentence

> A kernel function $k(x, x')$ computes an inner product in a (possibly infinite-dimensional) feature space without ever computing the features explicitly.

That sentence is the entire content of kernel methods. Once you have it, everything else follows.

---

## 2. Why kernels exist

### The motivation

Linear methods (linear regression, logistic regression, linear SVM) can only learn linear decision boundaries. Real data is often non-linear. Two options:

1. **Hand-engineer features.** Add $x_1^2, x_1 x_2, \log x_3$, etc. Then apply a linear method.
2. **Use kernels.** Implicitly map $x \to \phi(x)$ in some high-dimensional space, then apply the linear method there.

Kernels become useful when the feature mapping $\phi$ is high-dimensional or infinite-dimensional — explicit computation of $\phi(x)$ is intractable, but $\phi(x)^\top \phi(x') = k(x, x')$ might be cheap.

### The trick

Many algorithms can be written so they only ever access the data through inner products $x_i^\top x_j$. Replace $x_i^\top x_j$ with $k(x_i, x_j)$ and you have a non-linear version of the algorithm operating in $\phi$-space — without ever computing $\phi$.

This is **the kernel trick**.

---

## 3. Common kernels

### Linear kernel

$$
k(x, x') = x^\top x'
$$

Just the dot product. No mapping. Equivalent to no kernel.

### Polynomial kernel

$$
k(x, x') = (x^\top x' + c)^d
$$

Implicitly maps to a feature space containing all monomials up to degree $d$. For $d = 2$: includes products like $x_1 x_2, x_1^2$.

### RBF (Gaussian) kernel

$$
k(x, x') = \exp\!\left(-\gamma \|x - x'\|^2\right)
$$

The most popular kernel. Implicitly maps to an **infinite-dimensional** feature space. $\gamma$ controls bandwidth: small $\gamma$ → smooth, large $\gamma$ → wiggly.

### Sigmoid kernel

$$
k(x, x') = \tanh(\alpha\, x^\top x' + c)
$$

Inspired by neural networks. Not always positive semi-definite (Mercer's condition might fail) but used in practice.

### Cosine kernel

$$
k(x, x') = \frac{x^\top x'}{\|x\| \|x'\|}
$$

Normalized inner product. Used heavily in NLP/IR.

---

## 4. Mercer's theorem

For a function $k(x, x')$ to be a **valid kernel** (i.e., correspond to some inner product in some feature space):

1. **Symmetric:** $k(x, x') = k(x', x)$.
2. **Positive semi-definite:** for any finite set $\{x_1, \ldots, x_n\}$, the kernel matrix $K_{ij} = k(x_i, x_j)$ has all eigenvalues $\geq 0$.

Equivalently: there exists a feature map $\phi$ such that $k(x, x') = \phi(x)^\top \phi(x')$.

This is **Mercer's theorem**. It's why we can use kernels without ever computing $\phi$ — the PSD condition guarantees an implicit feature space exists.

---

## 5. Kernel SVM: the canonical application

Linear SVM optimizes:

$$
\min_w \tfrac{1}{2} \|w\|^2 + C \sum_i \max(0, 1 - y_i (w^\top x_i + b))
$$

The **dual formulation** (using Lagrange multipliers $\alpha_i$):

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^\top x_j)
$$

subject to $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$. Decision function:

$$
f(x) = \sum_i \alpha_i y_i (x_i^\top x) + b
$$

**The dual only accesses data through inner products.** Replace $x_i^\top x_j$ with $k(x_i, x_j)$:

$$
f(x) = \sum_i \alpha_i y_i k(x_i, x) + b
$$

Boom — non-linear SVM in the kernel feature space, computed without ever materializing $\phi$.

### Support vectors

Most $\alpha_i$ end up at 0 in the dual. The non-zero $\alpha_i$ correspond to **support vectors** — points on or inside the margin. Decision function only depends on support vectors:

$$
f(x) = \sum_{i \in \text{SV}} \alpha_i y_i k(x_i, x) + b
$$

Cost: $O(|\text{SV}| \cdot \text{kernel eval})$ per prediction.

---

## 6. RKHS (Reproducing Kernel Hilbert Space)

The advanced framing.

For any valid kernel, there exists a Hilbert space $\mathcal{H}_k$ of functions where:

1. Each kernel evaluation $k(\cdot, x)$ is itself a function in $\mathcal{H}_k$.
2. Inner product in $\mathcal{H}_k$ is given by the kernel: $\langle f, k(\cdot, x) \rangle = f(x)$ — the **reproducing property**.

Functions learned by kernel methods (SVM, kernel ridge, GP) live in $\mathcal{H}_k$ — they're linear combinations of kernel evaluations.

**Why it matters:** RKHS provides a unified theoretical framework. The "regularization with $\|f\|_{\mathcal{H}_k}^2$" view explains why SVM with RBF kernel doesn't overfit despite operating in infinite dimensions.

---

## 7. Other kernel methods

Beyond SVM, the kernel trick applies to many algorithms:

### Kernel ridge regression

$$
\hat f(x) = \sum_i \alpha_i k(x_i, x), \qquad \alpha = (K + \lambda I)^{-1} y
$$

Closed-form. Just like linear ridge but in kernel space.

### Gaussian processes

Bayesian extension. $f \sim \mathrm{GP}(0, k)$. Posterior given data is also Gaussian, with mean $\mu(x) = k(x, X)(K + \sigma^2 I)^{-1} y$ and variance $\sigma^2(x) = k(x, x) - k(x, X)(K + \sigma^2 I)^{-1} k(X, x)$. Provides uncertainty estimates.

### Kernel PCA

PCA in feature space via the kernel trick. Useful for non-linear dimensionality reduction.

### Kernel k-means

K-means in feature space. Equivalent to spectral clustering for some kernels.

---

## 8. Why kernels lost to deep learning

Kernel methods dominated 1995–2010. Then deep learning won. Why?

### Computational scaling
Kernel methods are typically $O(N^2)$ memory (full kernel matrix) and $O(N^2)$ to $O(N^3)$ training. For $N > 10^5$, infeasible. Deep learning scales linearly in data size.

### Representation learning
Kernels are **fixed**: you choose RBF, polynomial, etc. Deep learning **learns the representation** end-to-end. For images, text, audio, the right "feature space" is task-specific, and deep learning discovers it.

### Hyperparameter tuning
Kernel choice + bandwidth + regularization is hard to tune. Deep learning hyperparameters are easier to navigate at scale.

### Where kernels still win
- Small data ($N < 10^4$).
- Tabular data where SVM with RBF beats logistic regression but you don't have the data for deep learning.
- Bayesian uncertainty (Gaussian processes).
- Theoretical analysis (NTK).

---

## 9. The Neural Tangent Kernel (NTK) — connection to deep learning

Jacot et al. 2018. In the infinite-width limit of a deep neural network with appropriate scaling, the network's training behavior is exactly described by a **kernel** — the NTK.

$$
k_{\text{NTK}}(x, x') = \mathbb{E}\!\left[\langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle\right]
$$

i.e., inner product of the gradients of the network's output with respect to its parameters, in the limit of infinite width.

### Why it matters

- Provides a theoretical framework for understanding what NNs are doing.
- The NTK is fixed at initialization — doesn't change during training (in the infinite-width limit). Training is equivalent to kernel ridge regression with the NTK.
- Explains why over-parameterized NNs generalize: under NTK dynamics, gradient descent finds the minimum-norm solution in $\mathcal{H}_{\text{NTK}}$.

### Limitations

The NTK theory describes wide networks at initialization. Real NNs at modest width or after substantial training don't behave purely as NTK — feature learning happens. So NTK is a useful theoretical lens but doesn't fully explain deep learning's success.

---

## 10. Connection to attention

**The attention mechanism is essentially a learned kernel.**

Attention computes:

$$
\mathrm{attention}(Q, K, V)_i = \sum_j \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^\top k_{j'} / \sqrt{d_k})} v_j
$$

Compare to kernel ridge regression's prediction:

$$
\hat f(x) = \sum_i \alpha_i k(x_i, x)
$$

Attention is similar — query attends to keys via a kernel-like similarity (dot product), then weighted-averages values.

**The key difference:** attention's "kernel" $\exp(q^\top k / \sqrt{d_k})$ is **learned** via $W_Q, W_K$. Classical kernels are fixed. This is why attention is so powerful — it learns the right similarity per task.

This connection is increasingly invoked in research (e.g., Tsai et al., "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel"). Frontier-lab interview-relevant.

---

## 11. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "What's the kernel trick?" | Replace $x^\top x'$ with $k(x, x')$ in any algorithm that only accesses data via inner products. Operates in implicit high-dim feature space without computing it. |
| "What kernels are valid?" | Mercer's theorem: symmetric and positive semi-definite. PSD ⟹ implicit feature space exists. |
| "Why is RBF infinite-dimensional?" | Taylor-expand $\exp(-\gamma\|x - x'\|^2)$; the polynomial expansion has infinitely many terms, each corresponding to a feature dimension. |
| "Why does SVM work with the kernel trick?" | The dual formulation only uses inner products. Replace with kernel; non-linear SVM. |
| "What are support vectors?" | Training points with $\alpha_i > 0$ in the dual — points on or inside the margin. Decision function depends only on them. |
| "Why did kernels lose to deep learning?" | $O(N^2)$ scaling; fixed kernels can't learn task-specific representations; deep learning learns features. |
| "Is attention a kernel?" | Yes, conceptually. Attention is a learned kernel via $W_Q, W_K$. The connection unifies classical kernels and modern transformers. |
| "When are kernels still useful?" | Small data, Bayesian uncertainty (GPs), tabular tasks below NN scale, theoretical analysis (NTK). |

---

## 12. The 8 most-asked kernel interview questions

1. **What's the kernel trick?** Replace inner products with kernel evaluations to operate in implicit feature space.
2. **Mercer's theorem?** Symmetric + PSD ⟹ valid kernel.
3. **RBF kernel?** $\exp(-\gamma \|x - x'\|^2)$. Infinite-dimensional implicit feature space. Most popular.
4. **Why does kernel SVM use the dual?** Dual formulation accesses data only via inner products → kernel trick applies.
5. **Support vectors?** Non-zero $\alpha_i$ in the dual; decision function depends only on these.
6. **Why kernels lost to deep learning?** $O(N^2)$ scaling, fixed kernels, no representation learning.
7. **What's the NTK?** Infinite-width NN behaves as kernel ridge regression with the NTK. Theoretical bridge.
8. **Connection to attention?** Attention is a learned kernel — query-key dot product is a similarity function the model learns.

---

## 13. Drill plan

1. State the kernel trick precisely.
2. Memorize RBF kernel + Mercer's conditions.
3. Walk through SVM dual derivation.
4. Cite the NTK connection at sketchy level.
5. Drill `INTERVIEW_GRILL.md`.

---

## 14. Further reading

- Schölkopf & Smola, *Learning with Kernels* (2002) — the textbook.
- Hastie, Tibshirani, Friedman, *Elements of Statistical Learning*, Chapter 12.
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006).
- Jacot et al., "Neural Tangent Kernel" (2018).
- Tsai et al., "Transformer Dissection: Attention as a Kernel" (2019).
