# ML & LLM Interview Q&A: 144 Questions

Comprehensive interview questions and answers for ML/LLM coding interviews.

## Table of Contents
1. [Classical ML](#classical-ml)
2. [LLM Fundamentals](#llm-fundamentals)
3. [LLM Inference](#llm-inference)
4. [Training Techniques](#training-techniques)
5. [Optimization](#optimization)
6. [Regularization](#regularization)
7. [Bias & Variance](#bias-variance)
8. [Information Theory](#information-theory)
9. [Discriminative vs Generative Models](#discriminative-vs-generative-models)
10. [Kernel Functions](#kernel-functions)
11. [NLP Basics](#nlp-basics)
12. [MLE and MAP Estimation](#mle-and-map-estimation)
13. [Multimodal Models and Embeddings](#multimodal-models-and-embeddings)
14. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
15. [Linear and Logistic Regression Derivations](#linear-and-logistic-regression-derivations)
16. [RAG Retrieval Methods](#rag-retrieval-methods)
17. [NLP Problems: Standard Solution Procedures](#nlp-problems-standard-solution-procedures)
18. [Foundation Models: Evolution from BERT to GPT-4](#foundation-models-evolution-from-bert-to-gpt-4)
19. [Multimodal Integration and World Models](#multimodal-integration-and-world-models)
20. [GPT Implementation, Training, and Decoding](#gpt-implementation-training-and-decoding)
21. [Prompt Tuning and Prefix Tuning](#prompt-tuning-and-prefix-tuning)
22. [Diffusion Models](#diffusion-models)
23. [Perplexity and Related Concepts](#perplexity-and-related-concepts)
24. [Causal Attention](#causal-attention)
25. [Advanced Attention Mechanisms (GQA, Paged Attention)](#advanced-attention-mechanisms-gqa-paged-attention)
26. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
27. [State Space Models (SSM)](#state-space-models-ssm)
28. [Classical ML: Trees, Ensembles, and Dimensionality](#classical-ml-trees-ensembles-and-dimensionality)
29. [Evaluation and Data Discipline](#evaluation-and-data-discipline)
30. [Training Fundamentals](#training-fundamentals)
31. [Modern LLM Systems](#modern-llm-systems)

---

## Classical ML

### Q1: Implement linear regression from scratch.

**Answer:**
```python
class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = X.dot(self.weights) + self.bias
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias
```

**Key Points:**
- Gradient descent: Update weights using gradients
- Cost function: MSE = mean((y_pred - y)²)
- Gradients: dw = X.T @ (y_pred - y) / n, db = mean(y_pred - y)

**Walkthrough — what the algorithm is actually doing.**

Linear regression assumes the target is a weighted sum of the features plus a constant offset, so the model is $\hat{y} = Xw + b$, where $X$ is the $n \times d$ design matrix ($n$ rows of data, $d$ features per row), $w$ is the length-$d$ vector of weights, and $b$ is a single scalar called the bias or intercept. "Fitting" means choosing $w$ and $b$ so the predictions are as close as possible to the observed targets, where "close" is measured by the mean squared error (MSE), the average of the squared gaps:

$$J(w, b) = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2$$

The squares matter for two reasons: they make every error positive so overshoots and undershoots do not cancel, and they make $J$ a smooth convex bowl in $(w, b)$, which means there is exactly one minimum and gradient descent cannot get stuck anywhere else.

Gradient descent is the method used here. The gradient $\nabla J$ is the vector of partial derivatives; it points in the direction of steepest *increase* of the loss, so stepping in the *negative* gradient direction reduces the loss. Doing that repeatedly with a small step size walks downhill to the bottom of the bowl.

**Where the gradients come from.** Write the residual for row $i$ as $e_i = \hat{y}_i - y_i$. Differentiating the MSE with respect to weight $j$ and applying the chain rule (the outer derivative of $e_i^2$ is $2e_i$, the inner derivative of $\hat{y}_i = \sum_j x_{ij}w_j + b$ with respect to $w_j$ is $x_{ij}$):

$$\frac{\partial J}{\partial w_j} = \frac{2}{n}\sum_{i=1}^{n} e_i\, x_{ij}, \qquad \frac{\partial J}{\partial b} = \frac{2}{n}\sum_{i=1}^{n} e_i$$

Stacked over all $j$, the first expression is exactly $\frac{2}{n}X^\top e$. The code drops the factor of 2 because it is a constant that just rescales the learning rate — a common and harmless simplification. Notice what the weight gradient says intuitively: a feature gets a large gradient when it is large *and* correlated with the direction we are currently getting wrong. The bias gradient is just the average residual, which nudges the whole prediction line up or down.

**Line by line.** `__init__` stores the two hyperparameters — `lr`, the step size, and `n_iter`, how many full passes to take — and leaves the parameters as `None` because they cannot be sized until we see the data. In `fit`, `X.shape` gives us `n_samples` and `n_features`, and the weights are initialised to a zero vector of length `n_features` with the bias at zero; for a convex problem the starting point does not affect the final answer, only how long it takes to get there. Inside the loop, `y_pred = X.dot(self.weights) + self.bias` is one matrix-vector product that computes all $n$ predictions at once. `y_pred - y` is the residual vector $e$. `X.T.dot(y_pred - y)` contracts that residual against every feature column simultaneously, giving all $d$ partial derivatives in one operation, and dividing by `n_samples` turns the sum into a mean so the learning rate does not have to be retuned when the dataset size changes. `np.sum(y_pred - y) / n_samples` is the bias gradient. The two subtraction lines take the downhill step. `predict` simply reapplies the learned affine map. This is full-batch gradient descent: every iteration touches every row, which is exact but slow on large data; swapping in a random subset per step turns it into stochastic (mini-batch) gradient descent.

**One practical caveat.** Because the same `lr` is used for every weight, features on wildly different scales converge at wildly different rates and the loss surface becomes a long narrow valley that gradient descent zigzags down. Standardising each feature to zero mean and unit variance before fitting fixes this. If `lr` is too large the updates overshoot the minimum and the loss diverges to `nan`; if too small it crawls.

**Follow-up:** *Why iterate at all when linear regression has a closed-form solution?* The normal equation $w = (X^\top X)^{-1} X^\top y$ gives the exact optimum in one shot, and for a few thousand features it is the better choice. But it costs roughly $O(d^3)$ to invert (or factorise) a $d \times d$ matrix and requires $X^\top X$ to be invertible, which fails under exact multicollinearity. Gradient descent costs $O(nd)$ per step, streams over data that does not fit in memory, extends unchanged to models with no closed form, and degrades gracefully when features are collinear.

> **Why the interviewer asks this.** It is the cheapest possible check that you can go from a loss function to its gradient to working vectorised code without reaching for a library.

> **Saying it out loud.** "The model's just a weighted sum of the features plus an intercept, and I'm scoring it with mean squared error. I initialise the weights at zero, and then each iteration I predict, take the residual — prediction minus truth — and push the weights in the direction that shrinks it. The weight gradient is X transpose times the residual over n, and the bias gradient is just the mean residual. It's a convex problem, so as long as my learning rate is sane I'll land at the global optimum. In practice I'd standardise the features first, otherwise one big-scale column dominates the step size."

---

### Q2: What's the difference between linear and logistic regression?

**Answer:**

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| **Output** | Continuous values | Probabilities (0-1) |
| **Activation** | None (linear) | Sigmoid |
| **Cost Function** | MSE | Log loss (cross-entropy) |
| **Use Case** | Regression | Classification |
| **Gradient** | Linear | Non-linear (sigmoid derivative) |

**Key Difference:**
- Linear: y = w*x + b
- Logistic: p = sigmoid(w*x + b), then classify p > 0.5

**The mechanism behind the table.**

The two models share a linear core, $z = w^\top x + b$. Linear regression stops there and treats $z$ as the prediction. Logistic regression pushes $z$ through the sigmoid (also called the logistic function),

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

which is a smooth S-shaped squash from the whole real line into the open interval $(0, 1)$, so the output can be read as a probability. Inverting it shows what the linear part really means: $z = \log\frac{p}{1-p}$, the *log-odds* (the log of the ratio of the probability of the event to the probability of its complement). So logistic regression is a linear model in log-odds space, not in probability space. A one-unit increase in a feature adds a fixed amount to the log-odds, which multiplies the odds by a constant factor $e^{w_j}$ — that is the standard way to interpret a logistic coefficient.

**Why log loss and not MSE — the part follow-ups probe.** There are two independent reasons.

The first is optimisation geometry. With MSE the logistic objective $\sum_i (\sigma(z_i) - y_i)^2$ is *not convex* in $w$, because the sigmoid's own curvature flips sign at $z = 0$; the composition can have multiple local minima and flat plateaus, so gradient descent's answer depends on where it started. Log loss, $-\sum_i [\,y_i \log p_i + (1-y_i)\log(1-p_i)\,]$, is convex in $w$, so there is a single optimum.

The second is gradient behaviour, and it is easy to see numerically. Differentiating MSE through the sigmoid gives a factor of $\sigma'(z) = p(1-p)$ in the gradient. Suppose the true label is $y = 1$ but the model is confidently wrong with $p = 0.001$ (so $z \approx -6.9$). Then $\sigma' = 0.001 \times 0.999 \approx 0.000999$, and the MSE gradient with respect to $z$ is $2(p - y)\,p(1-p) = 2(-0.999)(0.000999) \approx -0.002$ — nearly zero. The model is as wrong as it can be and the update is vanishingly small; it is stuck. Under log loss the $p(1-p)$ term cancels exactly against the derivative of the log, leaving the strikingly simple

$$\frac{\partial \mathcal{L}}{\partial z} = p - y$$

which for the same case is $0.001 - 1 = -0.999$: a full-strength correction. The size of the update is proportional to how wrong the probability is, which is exactly what you want. (The same cancellation is why softmax-with-cross-entropy is implemented as a single fused op in every framework.)

**Follow-up:** *Can you use logistic regression on more than two classes?* Yes — replace the sigmoid with the softmax, $p_k = e^{z_k} / \sum_j e^{z_j}$, one linear score per class, and the loss becomes multiclass cross-entropy. The gradient keeps the same $p - y$ form with $y$ as a one-hot vector. This is multinomial logistic regression, also known as softmax regression, and it is precisely the output layer of almost every classification neural network.

> **Why the interviewer asks this.** The surface answer is memorised by everyone; the real question underneath is whether you know why the loss function changes, not just that it does.

> **Saying it out loud.** "They share the same linear core — weights dot features plus a bias. Linear regression uses that number directly as the prediction; logistic regression squashes it through a sigmoid so it comes out as a probability, which means the linear part is really modelling log-odds. And you have to switch the loss too. If you put MSE on top of a sigmoid you lose convexity, and worse, when the model's confidently wrong the sigmoid derivative kills the gradient so it can't recover. With log loss all that cancels and the gradient is just predicted minus actual — big error, big update."

---

### Q3: Explain KNN algorithm.

**Answer:**
K-Nearest Neighbors is a lazy learning algorithm:

**Algorithm:**
1. Store all training data (no training phase)
2. For new point, find k nearest neighbors
3. For classification: Majority vote
4. For regression: Average of neighbors

**Distance Metric:**
- Usually Euclidean: √(Σ(xi - yi)²)
- Can use Manhattan, cosine, etc.

**K Value:**
- Small k: More sensitive to noise (high variance)
- Large k: Smoother decision boundary (high bias)
- Rule of thumb: k = √n

**Time Complexity:**
- Training: O(1) - just store data
- Prediction: O(n) - compare to all points

**Walkthrough — the idea and the code.**

K-Nearest Neighbors makes one assumption and nothing else: points that are close together in feature space tend to have similar labels. So there is no model to fit — no weights, no loss function, no gradient. "Training" is memorising the dataset, which is why it is called a *lazy* learner (all the work is deferred to prediction time) and a *non-parametric* method (the number of things it remembers grows with the data rather than being fixed in advance).

To predict for a new point $x$, compute the distance from $x$ to every stored training point, sort, take the $k$ closest, and let them vote. For classification the prediction is the most common label among those $k$; for regression it is their mean.

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # "Training" is just storage - nothing is learned.
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # Squared Euclidean distance from every test point to every train point.
        # Shape: (n_test, 1, n_features) - (1, n_train, n_features) -> (n_test, n_train)
        d2 = ((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2)
        # argpartition puts the k smallest in front in O(n) instead of O(n log n).
        idx = np.argpartition(d2, kth=self.k - 1, axis=1)[:, :self.k]
        neigh = self.y_train[idx]
        return np.array([Counter(row).most_common(1)[0][0] for row in neigh])
```

Reading it in order: `fit` casts and stores, and that is genuinely all it does. In `predict`, the broadcasting trick `X[:, None, :] - self.X_train[None, :, :]` creates an array of every coordinate-wise difference between every test point and every training point, squaring and summing along the feature axis gives the squared distance matrix. We use *squared* distance rather than taking the square root because the square root is monotonic — it does not change which neighbours are closest — and skipping it saves time. `np.argpartition` is used instead of `argsort` because we only need to know *which* $k$ are smallest, not their internal order, and partitioning is linear rather than $n\log n$. Indexing `y_train` by those positions gives the neighbours' labels, and `Counter(...).most_common(1)` performs the majority vote. For regression the last line becomes `neigh.mean(axis=1)`.

**Why $k$ controls the bias-variance tradeoff.** With $k = 1$ every training point is classified perfectly by itself, and the decision boundary wraps tightly around individual points — including mislabelled ones. That is high variance: reshuffle the training set and the boundary moves a lot. As $k$ grows, each prediction averages more neighbours, noise cancels, and the boundary smooths; push $k$ to $n$ and the model always predicts the global majority class, which is maximum bias. Choose $k$ by cross-validation, and prefer odd $k$ for binary problems so votes cannot tie.

**Two failure modes worth naming.** First, scaling: Euclidean distance sums squared differences across features, so a feature measured in the thousands (say, income) drowns out one measured in single digits (say, number of bedrooms). Standardise before using KNN, always. Second, the curse of dimensionality: as dimension $d$ grows, the ratio between the distance to the nearest and the farthest point converges to 1, so "nearest" stops carrying information. KNN is strong in low dimensions and weak in high ones unless you reduce dimensionality first.

**Follow-up:** *How do you make prediction faster than $O(nd)$ per query?* Build a spatial index — a KD-tree or ball tree — which prunes whole regions and gives roughly $O(\log n)$ queries in low dimensions, though both degrade to brute force past about 20 dimensions. Beyond that, use approximate nearest neighbour methods (HNSW graphs, IVF or product quantisation as in FAISS), which trade an exact guarantee for orders-of-magnitude speedups; this is the same machinery behind modern vector databases for embedding retrieval.

> **Why the interviewer asks this.** It is a quick probe of whether you understand that some models pay their cost at training time and others at inference time, and whether you remember that distance-based methods need feature scaling.

> **Saying it out loud.** "KNN doesn't really train — it just stores the data. At prediction time you measure the distance from the new point to everything you stored, grab the k closest, and take a majority vote, or an average if it's regression. Small k gives you a jagged boundary that chases noise; big k smooths everything out until you're basically predicting the majority class. Two gotchas I'd always mention: you have to scale your features, because otherwise whichever column has the biggest units owns the distance, and it falls apart in high dimensions because everything ends up roughly equidistant."

---

### Q4: How does K-means clustering work?

**Answer:**

**Algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

**Convergence:**
- Centroids don't change
- Or max iterations reached

**Initialization:**
- Random: Can get poor results
- K-means++: Better initialization

**Limitations:**
- Assumes spherical clusters
- Need to specify k
- Sensitive to initialization

**Walkthrough — what the algorithm is optimising and why it stops.**

K-means partitions $n$ points into $k$ groups by minimising the within-cluster sum of squares, also called inertia:

$$J = \sum_{j=1}^{k} \sum_{x \in C_j} \lVert x - \mu_j \rVert^2$$

where $C_j$ is the set of points assigned to cluster $j$ and $\mu_j$ is that cluster's centroid (its mean vector). Minimising this jointly over both the assignments and the centroids is NP-hard, so the standard algorithm — Lloyd's algorithm — alternates between optimising one while holding the other fixed. That is the whole trick, and it is why each of the two steps in the loop is not arbitrary but is the *exact* minimiser of $J$ given the other:

Holding centroids fixed, the assignment that minimises $J$ is obviously "put each point with its nearest centroid" — each point's contribution is minimised independently. Holding assignments fixed, the point $\mu$ minimising $\sum_{x \in C_j} \lVert x - \mu \rVert^2$ is the arithmetic mean of $C_j$ (set the derivative to zero and solve). Since each step can only decrease $J$ and there are finitely many possible assignments, $J$ is monotonically non-increasing and the algorithm must terminate — but only at a *local* minimum, which is exactly why initialisation matters.

```python
import numpy as np

def kmeans(X, k, n_iter=100, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # --- k-means++ seeding: spread the initial centroids out ---
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(2), axis=1)
        probs = d2 / d2.sum()                      # far-from-everything points are likelier
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)

    for _ in range(n_iter):
        # Assignment step: nearest centroid for every point
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        # Update step: centroid = mean of its members (keep old center if empty)
        new_centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                                else centers[j] for j in range(k)])
        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers

    inertia = ((X - centers[labels]) ** 2).sum()
    return centers, labels, inertia
```

Step by step: the seeding block implements **k-means++**, which picks the first centre uniformly at random and then picks each subsequent centre with probability proportional to its squared distance from the nearest already-chosen centre. That biases the initial centres to be far apart, which avoids the classic failure where two centres land inside the same true cluster and split it while merging two others; it also comes with a proof that the expected inertia is within $O(\log k)$ of optimal. Inside the main loop, the distance matrix is built by the same broadcasting pattern as KNN, `argmin` along the centroid axis performs the assignment, and the list comprehension recomputes each centroid as the mean of its members — with a guard, because a cluster can end up empty and `mean` of an empty slice is `nan`. The convergence check compares how far the centres moved; when that is below tolerance nothing more will change. `inertia` is the objective value, useful for comparing runs.

**Practical consequences of the objective.** Because $J$ uses squared Euclidean distance and each cluster is summarised by a single mean, K-means implicitly assumes clusters that are roughly spherical, similarly sized, and similarly dense; it will happily slice a long thin cluster in half or merge two crescents. Squared distance also makes it sensitive to outliers, since one far-away point drags a centroid toward it (K-medoids, which uses actual data points as centres, is the robust alternative). And since it depends on a local optimum, standard practice is `n_init` restarts keeping the lowest-inertia run — scikit-learn does this by default.

**Choosing $k$.** Inertia decreases monotonically with $k$ (at $k = n$ it is zero), so you cannot just minimise it. The elbow method plots inertia against $k$ and looks for the bend where extra clusters stop buying much; the silhouette score, which compares each point's mean distance to its own cluster against its mean distance to the nearest other cluster, gives a value in $[-1, 1]$ that can be maximised directly and is usually the more defensible choice.

**Follow-up:** *What is the time complexity?* Each iteration is $O(nkd)$ — every point against every centroid in $d$ dimensions — for $t$ iterations and $r$ restarts, so $O(tnkd \cdot r)$ overall. It is linear in the number of points, which is why K-means scales to large data where hierarchical clustering ($O(n^2)$ memory at least) does not.

> **Why the interviewer asks this.** They want to hear that the two steps are coordinate descent on a specific objective, not a heuristic someone made up — and that you know why it only ever finds a local optimum.

> **Saying it out loud.** "K-means is trying to minimise the total squared distance from each point to its cluster's centre. It alternates two steps: assign every point to the nearest centre, then move each centre to the mean of the points that picked it. Each step is the exact best move given the other one, so the objective only goes down and it has to converge — but only to a local optimum, which is why you use k-means++ seeding and multiple restarts. It assumes round, similarly-sized clusters, so if the real shapes are elongated or nested, DBSCAN or a Gaussian mixture is a better fit."

---

## LLM Fundamentals

### Q5: Explain the transformer architecture.

**Answer:**

**Components:**
1. **Embedding Layer**: Token → Dense vectors
2. **Position Encoding**: Add position info
3. **Transformer Blocks** (N layers):
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Layer Normalization
   - Residual Connections
4. **Output Layer**: Project to vocabulary

**Key Innovation:**
- Self-attention: Relate all positions
- Parallel processing: All positions at once
- Long-range dependencies: No RNN limitations

**How the pieces fit together.**

The clearest mental model of a transformer is the **residual stream**. Every token carries a vector of width $d_{\text{model}}$ from the bottom of the network to the top, and each sublayer *reads* from that vector, computes something, and *adds* the result back. That is what the residual connection means: the block computes $x \leftarrow x + \text{Sublayer}(\text{LayerNorm}(x))$, not $x \leftarrow \text{Sublayer}(x)$. Because the update is additive, gradients flow to the bottom layer along an unobstructed path (the derivative of $x + f(x)$ has an identity term in it), which is what makes stacking 100 blocks trainable at all.

Within a block the two sublayers have complementary jobs. **Attention moves information between token positions** — it is the only operation in the whole architecture that lets position $i$ see position $j$. The **feed-forward network (FFN) processes each position independently**, applying the same two-layer MLP to every token separately: it projects up to an inner width (classically $4 \times d_{\text{model}}$), applies a nonlinearity such as GELU or SwiGLU, and projects back down. Attention mixes across tokens; the FFN adds nonlinear capacity within a token. Note that the FFN holds roughly two-thirds of the parameters in a standard block ($8d^2$ versus attention's $4d^2$), which is why it is the usual target for mixture-of-experts sparsification.

**Layer normalisation** rescales each token's vector to zero mean and unit variance across its features (then applies a learned gain and bias), which keeps activation magnitudes stable as depth grows. The original 2017 paper put it *after* the residual addition (post-LN); essentially every modern LLM puts it *before* the sublayer (pre-LN), because post-LN needs a learning-rate warmup to train deep stacks without diverging while pre-LN is stable out of the box. Many current models further simplify to RMSNorm, which divides by the root-mean-square and skips the mean subtraction and the bias.

**Position encoding** is needed because attention is permutation-equivariant: with no positional signal, "dog bites man" and "man bites dog" produce identical sets of representations. The original design added fixed sinusoids of geometrically spaced frequencies to the embeddings; current models overwhelmingly use RoPE (rotary position embedding), which rotates the query and key vectors by an angle proportional to position so that the attention score depends only on the *relative* offset between two tokens — that relative property is what makes context-length extension via frequency scaling possible.

**Three configurations.** The original paper is an encoder-decoder, built for translation: a bidirectional encoder reads the source, and a decoder generates the target while cross-attending to the encoder's output. **Encoder-only** models (BERT) keep bidirectional attention and are trained by masked-token prediction; they are for understanding tasks like classification and retrieval, and cannot generate autoregressively. **Decoder-only** models (GPT, Llama, Claude) use causal masking — position $i$ may attend only to positions $\le i$ — and are trained to predict the next token; this is the dominant design for LLMs today because a single next-token objective scales cleanly and covers generation, classification, and everything else via prompting.

**Follow-up:** *Where does the quadratic cost come from, and what is done about it?* The attention score matrix is $n \times n$ for sequence length $n$, so compute is $O(n^2 d)$ and, naively, memory is $O(n^2)$ too. FlashAttention removes the memory term by tiling the computation and never materialising the full matrix — it is exact, just IO-aware. The compute term is attacked by sparse or sliding-window attention (attend only to a local neighbourhood plus a few global tokens) and by linear-attention or state-space alternatives such as Mamba, which trade some expressivity for $O(n)$ scaling.

> **Why the interviewer asks this.** It is an open door: they want to see which level of detail you naturally reach for, and whether you can say what each component is *for* rather than just listing the diagram top to bottom.

> **Saying it out loud.** "The way I picture it, every token carries a vector up through the stack, and each layer reads that vector, computes something, and adds its result back — that's the residual stream. Inside a block there are two jobs. Attention is the only thing that moves information *between* positions. The feed-forward net works on each token on its own and gives you the nonlinear capacity — it's also where most of the parameters live. LayerNorm keeps the scale under control, and you need positional information injected somewhere because attention on its own has no idea what order the tokens came in. Modern LLMs are decoder-only with a causal mask, pre-norm, and rotary positions."

---

### Q6: How does self-attention work?

**Answer:**

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Steps:**
1. Compute Q, K, V from input
2. Compute attention scores: Q @ K^T
3. Scale by √d_k (prevent large values)
4. Softmax to get attention weights
5. Apply weights to V

**Why it works:**
- Query asks "what am I looking for?"
- Key answers "what information do I have?"
- Value is "the actual information"
- Attention weights show relevance

**The mechanism in full: two circuits, QK and OV.**

Start from the projections. Each token's residual-stream vector $x_i \in \mathbb{R}^{d_{\text{model}}}$ is multiplied by three learned matrices to give a query $q_i = W_Q x_i$, a key $k_i = W_K x_i$, and a value $v_i = W_V x_i$. The useful way to read this is that attention factors into two independent circuits:

- The **QK circuit** decides *where to look*. The score between positions $i$ and $j$ is $q_i \cdot k_j = x_i^\top W_Q^\top W_K x_j$, so only the product $W_Q^\top W_K$ matters — a single bilinear form saying "how much does a token like me want to read from a token like that?" This circuit produces the attention pattern and nothing else; it never touches the content that gets copied.
- The **OV circuit** decides *what to move*. Once the weights are fixed, the output is $\sum_j \alpha_{ij} W_O W_V x_j$, so only the product $W_O W_V$ matters — a linear map saying "if I read from a token, what do I write into my residual stream?"

Splitting it this way explains attention's real function: it is a *soft, content-addressable lookup*. The QK circuit computes addresses, the softmax turns them into a normalised mixing weight, and the OV circuit is the payload. A concrete example is the induction head found in real models: the QK circuit matches the current token against earlier occurrences of the same token, and the OV circuit copies whatever followed it last time, which is how models do in-context pattern completion.

**Why divide by $\sqrt{d_k}$ — with numbers.** Suppose the entries of $q$ and $k$ are roughly independent with mean 0 and variance 1. Their dot product is a sum of $d_k$ such products, so it has variance $d_k$ and typical magnitude $\sqrt{d_k}$. With $d_k = 64$ that is around $\pm 8$; with $d_k = 128$, around $\pm 11$. Feed scores that large into a softmax and it saturates — one weight goes to essentially 1 and the rest to essentially 0. Concretely, softmax over $[8, 0]$ gives $[0.99966, 0.00034]$, and the gradient of the softmax is proportional to $p(1-p)$, so at $p = 0.99966$ the gradient is about $3.4 \times 10^{-4}$: the attention pattern freezes and stops learning. Dividing by $\sqrt{d_k}$ renormalises the scores to unit variance regardless of head width, so the softmax stays in its responsive range. Softmax over $[1, 0]$ gives $[0.73, 0.27]$ with a healthy gradient.

**A small worked example.** Take $d_k = 2$ and three tokens with $q_2 = [1, 0]$ and keys $k_1 = [1, 0]$, $k_2 = [0, 1]$, $k_3 = [1, 1]$. The raw scores for query 2 are $q_2 \cdot k_j = [1, 0, 1]$. Scaling by $\sqrt{2} \approx 1.414$ gives $[0.707, 0, 0.707]$. Exponentiating gives $[2.028, 1.0, 2.028]$, which sums to $5.056$, so the attention weights are $[0.401, 0.198, 0.401]$. The output for position 2 is $0.401 v_1 + 0.198 v_2 + 0.401 v_3$ — a convex combination of the value vectors, weighted by query-key similarity. Every row of the attention matrix is a probability distribution over positions in exactly this way.

**Causal masking.** In a decoder, position $i$ must not see the future, or the model could cheat at next-token prediction. This is implemented by adding $-\infty$ (in practice a large negative number like $-10^9$) to all scores where $j > i$ *before* the softmax, so those weights come out as exactly zero after exponentiation. Doing it pre-softmax rather than zeroing afterwards matters, because it keeps the remaining weights correctly normalised to sum to one.

**Follow-up:** *Why do we need three separate projections — why not just use $x$ itself?* Because $x \cdot x$ is maximised by $x$ itself, so unprojected attention would collapse to every token attending mostly to itself, and the score would be forced to be symmetric — position $i$ would attend to $j$ exactly as much as $j$ attends to $i$. Separate $W_Q$ and $W_K$ break that symmetry and let "what I'm looking for" differ from "what I advertise". A separate $W_V$ then decouples relevance from content, so a token can be highly relevant as an address while contributing something entirely different as a payload.

> **Why the interviewer asks this.** They want to know whether "query, key, value" is a phrase you repeat or a structure you can decompose — the split between where attention looks and what it copies is the tell.

> **Saying it out loud.** "Each token projects into three vectors — a query, a key, and a value. The query is what this token is looking for, the key is what each other token advertises, and the dot product between them scores relevance. You divide by root d-k because otherwise the dot products grow with the head dimension, the softmax saturates, and the gradient dies. Then softmax turns those scores into weights that sum to one, and the output is a weighted average of the value vectors. So really there are two separate circuits: query-key decides *where* you look, and value-output decides *what* gets copied back into your representation. In a decoder you mask out the future before the softmax so a token can't see what comes after it."

---

### Q7: What is multi-head attention?

**Answer:**

**Concept:**
- Instead of one attention, use multiple "heads"
- Each head learns different relationships
- Concatenate all heads, then project

**Why Multiple Heads:**
- Different heads attend to different aspects
- Example: One head for syntax, one for semantics
- More expressive than single head

**Implementation:**
1. Split d_model into num_heads × d_k
2. Each head has its own Q, K, V
3. Compute attention for each head
4. Concatenate outputs
5. Final projection

**The dimension arithmetic, concretely.**

Multi-head attention does not add parameters over single-head attention of the same width — it *partitions* them. With $d_{\text{model}} = 512$ and $h = 8$ heads, each head gets $d_k = d_{\text{model}} / h = 64$. In practice you keep one big $512 \times 512$ matrix for $W_Q$ (and likewise $W_K$, $W_V$), project once, then reshape the result from $(\text{batch}, n, 512)$ to $(\text{batch}, n, 8, 64)$ and transpose to $(\text{batch}, 8, n, 64)$ so the eight heads become a batch dimension. Every head runs the scaled-dot-product attention of Q6 independently on its own 64-dimensional slice, producing $(\text{batch}, 8, n, 64)$. Transposing back and reshaping concatenates the heads into $(\text{batch}, n, 512)$, and the output projection $W_O$ (also $512 \times 512$) mixes them before the result is added to the residual stream. Total parameters: $4 d_{\text{model}}^2$, identical to one head of full width.

**Why splitting helps rather than hurts.** A single head produces one attention distribution per query position — it must commit to one weighted average. That is a hard constraint: a pronoun resolving its antecedent and a verb finding its subject are different lookups that a single softmax cannot perform at once, because probability mass spent on one is taken from the other. Eight heads give eight independent distributions whose results are summed into the residual stream, so the block can perform several distinct retrievals in parallel. The cost is that each head sees a lower-rank slice of the space, which is a real limitation — this is why very small $d_k$ (below about 32) tends to hurt, and why head count and head dimension are tuned together rather than head count alone being maximised.

**Follow-up:** *What are MQA and GQA, and why do they exist?* The KV cache at inference stores one key and one value vector per head per token, so its size scales with head count — and at generation time the arithmetic is memory-bandwidth-bound, meaning the GPU spends most of its time reading that cache rather than doing math. **Multi-query attention (MQA)** keeps $h$ separate query heads but shares a *single* key/value head across all of them, shrinking the cache by a factor of $h$; it is fast but measurably degrades quality. **Grouped-query attention (GQA)** is the compromise now used by most open models: heads are split into $g$ groups (say 8 groups over 64 query heads) with one KV head per group, recovering most of the quality at most of the speed. Nothing about the query side changes; only the number of distinct K and V projections does.

> **Why the interviewer asks this.** The giveaway answer is "heads learn different things", which is true but unfalsifiable; they are checking whether you know the dimensions split rather than multiply, and ideally that you know why inference pushed the field toward GQA.

> **Saying it out loud.** "Multi-head attention splits the model dimension across heads rather than duplicating it — 512 dimensions and 8 heads means each head works in 64. So the parameter count is the same as one wide head, but instead of one attention distribution you get eight running in parallel, and you concatenate them and pass them through an output projection. The reason that matters is that a single softmax has to spend its probability mass in one place; with several heads the layer can do several different lookups at once. The modern wrinkle is that KV cache size scales with head count at inference, so people share key and value heads across groups of query heads — that's grouped-query attention."

---

## LLM Inference

### Q8: How does KV caching work?

**Answer:**

**Problem:**
- Autoregressive generation: Generate token by token
- Each token needs attention to all previous tokens
- Without cache: Recompute attention for all tokens each step

**Solution:**
- Cache K and V matrices for previous tokens
- New token: Only compute Q, reuse cached K/V
- Append new K/V to cache

**Example:**
```
Step 1: Token 1 → Compute Q1, K1, V1, cache K1, V1
Step 2: Token 2 → Compute Q2, K2, V2
        → Attention: Q2 @ [K1, K2]^T, use [V1, V2]
        → Cache: [K1, K2], [V1, V2]
Step 3: Token 3 → Compute Q3, K3, V3
        → Attention: Q3 @ [K1, K2, K3]^T, use [V1, V2, V3]
        → Cache: [K1, K2, K3], [V1, V2, V3]
```

**Speedup:** 10-100x for generation

**Why it works at all, and why only K and V.**

The reason caching is even possible is causal masking. In a decoder, position $j$ attends only to positions $\le j$, so once token $j$'s key and value vectors are computed they are *final* — no later token can change them, because nothing later flows into them. That is a property of the mask, not an approximation; KV caching is mathematically exact, producing bit-comparable outputs (up to floating-point reduction order), not a speed-for-quality trade.

Why keys and values but not queries: at each generation step you have exactly one new token, so you need exactly one new query — the queries of earlier tokens are never used again, because their outputs were already computed and consumed. But that single new query must be scored against *every* previous key and must read from *every* previous value. So K and V accumulate; Q does not.

**What this does to the cost.** Without a cache, generating token $t$ means a full forward pass over $t$ tokens, which is $O(t^2 d)$ of attention work, and generating a whole sequence of length $n$ costs $O(n^3 d)$. With the cache, each step is one query against $t$ keys, so $O(td)$ per step and $O(n^2 d)$ overall — a factor of $n$ saved. This is also why LLM inference has two distinct phases with very different characteristics: **prefill**, where the whole prompt is processed in one parallel pass and the GPU is compute-bound, and **decode**, where one token is produced at a time and the GPU is memory-bandwidth-bound because it must stream the entire cache and all model weights per token.

**The cost is memory, and the formula is worth knowing.**

$$\text{cache bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{seq len} \times \text{batch} \times \text{bytes per element}$$

The leading 2 is for K and V. Take a 7B-class model with 32 layers, 32 KV heads, head dimension 128, in FP16 (2 bytes): that is $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes per token, or about 0.5 MB. At 4,096 tokens of context that is roughly 2 GB for a *single* sequence — and it scales linearly with batch size, so 16 concurrent requests at that length is about 32 GB, comfortably more than the model weights themselves. This is the single biggest constraint on serving throughput, and it is what motivates GQA (fewer KV heads), KV-cache quantisation to INT8 or FP8, and PagedAttention in vLLM, which stores the cache in fixed-size non-contiguous blocks like OS virtual memory so that fragmentation and over-reservation stop wasting the majority of GPU memory.

**Follow-up:** *What breaks if the prompt changes at the front?* Everything downstream of the change, because each cached key depends on all preceding tokens through the attention of earlier layers. This is why prefix caching works — a shared system prompt at the *start* can be computed once and reused across requests — but appending to the beginning of a prompt invalidates the whole cache. It is a concrete reason to put stable content first and variable content last in prompt templates.

> **Why the interviewer asks this.** It separates people who have read about transformers from people who have served them; the memory formula and the prefill/decode distinction are what a serving engineer actually reasons about.

> **Saying it out loud.** "Because of the causal mask, once you've computed a token's key and value they never change — nothing later can affect them. So instead of recomputing the whole sequence every step, you keep K and V around and just compute the query for the new token. You don't cache queries because old queries are never used again. It turns generation from cubic in sequence length to quadratic. The catch is memory: the cache is roughly half a megabyte per token on a 7B model, so at long context and decent batch sizes it gets bigger than the weights — that's why people use grouped-query attention, quantise the cache, and use paged allocators like vLLM."

---

### Q9: What is quantization and why use it?

**Answer:**

**Quantization:** Reduce model precision
- FP32 → FP16: 2x smaller, 2x faster
- FP16 → INT8: 2x smaller, 2x faster
- INT8 → INT4: 2x smaller, 2x faster

**Why:**
- **Memory**: Smaller models fit in memory
- **Speed**: Faster computation
- **Cost**: Lower inference cost

**Trade-off:**
- Accuracy may decrease slightly
- Need calibration for INT8/INT4

**Process:**
1. Find min/max of weights
2. Calculate scale factor
3. Quantize to integer range
4. Store scale for dequantization

**The mechanism: an affine map between floats and integers.**

Quantisation replaces a high-precision tensor with low-precision integers plus a small amount of metadata to reconstruct approximate floats. The standard *asymmetric* (affine) scheme picks a scale $s$ and a zero-point $z$ so that

$$q = \text{round}\!\left(\frac{x}{s}\right) + z, \qquad \hat{x} = s\,(q - z)$$

with $s = (x_{\max} - x_{\min}) / (q_{\max} - q_{\min})$ and $z$ chosen so that real zero maps exactly to an integer (which matters, because padding and ReLU outputs produce a lot of exact zeros and you do not want them to drift). The *symmetric* variant fixes $z = 0$ and uses $s = \max|x| / q_{\max}$; it is cheaper because the dequantisation has no offset term, and it is the usual choice for weights, which are roughly zero-centred.

**A worked example.** Suppose a weight block ranges over $[-0.8, 1.2]$ and we quantise to INT8, which covers $[-128, 127]$. Asymmetrically, $s = (1.2 - (-0.8)) / 255 = 0.00784$. A weight of $0.5$ maps to $\text{round}(0.5 / 0.00784) = 64$ (plus the zero-point offset), and dequantising gives $64 \times 0.00784 = 0.5018$ — an error of about 0.0018, roughly half a step of $s$. That bounded round-off is the entire accuracy cost. Now note what happens with one outlier: if a single weight in the block were $12.0$, the scale would jump to $0.05$ and every ordinary weight would carry 27 times more error. Outliers, not average precision, are what actually break quantisation — which is why modern methods quantise in small **groups** (say 64 or 128 weights sharing one scale) rather than per-tensor, and why LLM.int8() keeps a handful of outlier channels in FP16 while quantising the rest.

**Why the speedup is bigger than the arithmetic suggests.** During single-token decoding the GPU is memory-bandwidth-bound: it must read every weight from HBM to produce one token, and the matrix multiplies are small. Halving the bytes per weight therefore roughly halves the time per token even if the arithmetic itself runs at the same rate. This is why weight-only quantisation — store INT4, dequantise to FP16 in the kernel, multiply in FP16 — is so popular for LLM serving: it captures the bandwidth win without needing integer matmul support or activation calibration.

**The families worth naming.** *Post-training quantisation (PTQ)* converts an already-trained model, optionally using a few hundred calibration samples to set activation ranges; GPTQ (which uses second-order information to compensate rounding error weight by weight) and AWQ (which scales up the channels the activations actually depend on before quantising) are the standard 4-bit PTQ methods. *Quantisation-aware training (QAT)* simulates rounding during training with a straight-through estimator for the gradient, costing a training run but retaining more accuracy at very low bit widths. QLoRA is the hybrid people actually use for fine-tuning: freeze a 4-bit NF4 base model and train small LoRA adapters in higher precision on top.

**The honest trade-off.** Going FP16 to INT8 is typically near-lossless on quality benchmarks. INT4 with good grouping loses a little, and it loses it unevenly — long-chain reasoning, arithmetic, and rarely-seen languages degrade before general fluency does, so a perplexity check alone can look fine while a task benchmark drops. Below 4 bits quality falls off sharply without specialised methods.

**Follow-up:** *Why not just quantise activations too?* You can, and INT8 activations are what let you use integer tensor cores for a genuine compute win. But activations depend on the input, so their range must be estimated from calibration data and can be exceeded at run time, and transformer activations contain systematic large-magnitude outlier channels that make per-tensor activation scales very lossy. That is exactly the problem LLM.int8() and SmoothQuant (which migrates the difficulty from activations into weights by rescaling) were designed to solve.

> **Why the interviewer asks this.** They want to know if you can reason about the deployment constraint — memory bandwidth — rather than reciting bit widths.

> **Saying it out loud.** "Quantisation is storing weights as low-bit integers plus a scale, so you can reconstruct an approximate float. The accuracy cost is just rounding error, bounded by half a step of the scale — the thing that actually hurts is outliers, because one huge weight blows up the scale for everything sharing it. That's why people quantise in small groups instead of per-tensor. And the reason it's such a big win for LLMs is that decoding is memory-bandwidth-bound: you're reading every weight from memory to emit one token, so halving the bytes roughly halves your latency even before you touch the arithmetic. INT8 is basically free, INT4 with good grouping is usually acceptable, and below that you need real work."

---

### Q10: Explain top-p (nucleus) sampling.

**Answer:**

**Algorithm:**
1. Sort tokens by probability (descending)
2. Compute cumulative probability
3. Find smallest set where cum_prob >= p
4. Sample from this "nucleus"
5. Renormalize probabilities

**Why it works:**
- Adaptive: Number of tokens varies
- High probability tokens: Always included
- Low probability tokens: Excluded
- Better than top-k (fixed size)

**Example:**
```
Probabilities: [0.5, 0.3, 0.1, 0.05, 0.03, ...]
Cumulative:    [0.5, 0.8, 0.9, 0.95, 0.98, ...]
Top-p=0.9: Nucleus = first 3 tokens (cum_prob = 0.9)
```

**Why the adaptive set size is the whole point.**

The problem top-p solves is that a language model's next-token distribution has wildly varying shape from step to step. After "the capital of France is" the distribution is nearly a spike — one token holds most of the mass. After "she opened the door and saw a" it is broad, with hundreds of plausible continuations. Top-k with a fixed $k$ handles neither well: at the spike it admits $k-1$ tokens that should have been ruled out, and at the broad step it truncates hundreds of legitimate options. Top-p instead fixes the *probability mass* to keep and lets the *count* float, so the nucleus is 1 token in the first case and several hundred in the second. That is the entire argument for it.

**A worked example.** Take probabilities $[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]$ over six tokens, with $p = 0.9$. The cumulative sums are $[0.5, 0.8, 0.9, 0.95, 0.98, 1.0]$. The smallest prefix reaching 0.9 is the first three, so the nucleus is $\{0.5, 0.3, 0.1\}$, summing to 0.9, and after renormalising (dividing by 0.9) we sample from $[0.556, 0.333, 0.111]$. Now take the peaked case $[0.95, 0.02, 0.015, 0.01, 0.005]$: the first token alone already reaches 0.95 which is $\ge 0.9$, so the nucleus is a single token and generation is effectively greedy at that step. Same $p$, completely different set size — which is exactly what you want. (Note the standard convention: the nucleus is the smallest prefix whose cumulative probability is *at least* $p$, so the kept mass is always $\ge p$, never less.)

**How it composes with temperature.** Temperature $T$ rescales the logits before the softmax, $p_i \propto \exp(z_i / T)$. Below 1 it sharpens the distribution and above 1 it flattens it. Order matters: temperature is applied *first*, then top-k/top-p truncation, then renormalisation. So raising temperature does not only make sampling more random — it also enlarges the nucleus, because flattening the distribution means more tokens are needed to accumulate mass $p$. The two knobs interact, which is why tuning both at once tends to be confusing and why most practitioners fix one (commonly $T = 1$ with $p \approx 0.9$–$0.95$, or $p = 1$ with a tuned temperature).

**What it is actually fixing.** The tail of a softmax over a 100k-token vocabulary contains tens of thousands of tokens each with tiny probability, but their *sum* can be a few percent. Sample long enough and you will draw from that tail, and one bad token conditions everything after it — the model has no way to take it back, and degeneration into incoherence follows. Truncation sampling exists because the model's tail is less trustworthy than its head, not because the tail has zero mass.

**Follow-up:** *When would you not want to sample at all?* Whenever there is a single correct answer and diversity is a liability: extraction, classification, structured or JSON output, most tool-call arguments, and any evaluation you want to be reproducible. There greedy decoding ($T \to 0$, equivalently top-k of 1) is right. Sampling is for open-ended generation where you want variety across runs. A common middle ground for reasoning tasks is to sample several times at moderate temperature and take a majority vote over final answers — self-consistency — which uses diversity as a search strategy rather than as an end in itself.

> **Why the interviewer asks this.** Decoding parameters are the knobs everyone touches and few can explain; being able to say precisely why the adaptive cutoff beats a fixed one is a small but reliable signal.

> **Saying it out loud.** "Top-p keeps the smallest set of tokens whose probabilities add up to p — say 0.9 — and samples from that after renormalising. The point is that the set size adapts. When the model's confident, that's one or two tokens and you're basically greedy; when it's genuinely uncertain, it might be hundreds. Fixed top-k can't do that: it lets junk in when the model is sure and cuts off good options when it isn't. And it composes with temperature — temperature is applied first, so cranking it up doesn't just add randomness, it also widens the nucleus."

---

## Training Techniques

### Q11: Explain RLHF (Reinforcement Learning from Human Feedback).

**Answer:**

**Pipeline:**
1. **Supervised Fine-tuning**: Train on human demonstrations
2. **Reward Model**: Train on human preferences (chosen vs rejected)
3. **RL Optimization**: Use PPO to optimize policy with reward model

**Why RLHF:**
- Align models with human preferences
- Make models helpful, harmless, honest
- Improve response quality

**Challenges:**
- Need human feedback data
- Reward model training
- RL optimization complexity

**Filling in the mechanism at each stage.**

*Stage 1, supervised fine-tuning (SFT).* Start from a pretrained base model, which can only continue text, and train it on human-written prompt-response pairs with ordinary next-token cross-entropy. This does not teach new knowledge; it teaches *format* — that a question should be answered rather than continued with more questions. It also matters technically, because the RL stage needs a starting policy that already produces plausible outputs; RL from a raw base model would spend its entire budget rediscovering the response format.

*Stage 2, the reward model (RM).* Humans are unreliable at assigning absolute scores ("how good is this answer out of 10?") but quite reliable at comparisons ("which of these two is better?"), so the data collected is preference pairs. The reward model is usually the SFT model with the token-prediction head replaced by a scalar head, trained with the Bradley-Terry loss

$$\mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

where $y_w$ is the preferred ("won") response, $y_l$ the rejected one, and $\sigma$ the sigmoid. Note what this objective does and does not pin down: it constrains *differences* in reward, so the scale is arbitrary and the absolute value of a reward is meaningless — only comparisons within a prompt are trustworthy. The model implicitly assumes preferences follow the Bradley-Terry model, that the probability a human prefers $y_w$ is $\sigma(r_w - r_l)$.

*Stage 3, RL optimisation.* The policy generates responses, the RM scores them, and PPO updates the policy to raise expected reward while a KL penalty against the frozen SFT model keeps it from wandering. The full per-token objective is

$$R(x, y) = r_\phi(x, y) - \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

The reward is *sparse and terminal* — one scalar for the whole response — which is why a value model is needed to spread credit back over the tokens, and why this stage is the fragile one.

**Why the reward model is the weak link.** It is trained on a finite sample of on-distribution responses, but the policy immediately starts producing responses the RM never saw. Since the policy is explicitly optimising the RM's output, it will find and exploit whatever regions the RM overestimates — this is Goodhart's law in its purest engineering form. Empirically, measured true quality rises, peaks, and then *falls* while the RM's score keeps climbing. The standard mitigations are the KL penalty (Q17), early stopping on the KL budget rather than on RM score, and periodically collecting fresh preferences on the current policy's outputs, which is what "iterated RLHF" means.

**Follow-up:** *What replaced parts of this pipeline?* Two things, mostly. RLAIF / Constitutional AI replaces human labels with model-generated preferences guided by a written set of principles, which makes the preference data far cheaper to scale. And for anything with a checkable answer, RLVR — reinforcement learning from verifiable rewards — replaces the learned reward model with a program that checks correctness, which removes the hackable component entirely. DPO (Q12) removes the RL loop while keeping human preferences.

> **Why the interviewer asks this.** They are checking that you know why there are three stages rather than one, and ideally that you can name which stage actually breaks in practice.

> **Saying it out loud.** "It's three stages. First you fine-tune on human-written demonstrations, which mostly teaches the model what a response is supposed to look like. Then you collect preference pairs — people are much better at saying which of two answers is better than at scoring one in isolation — and train a reward model on those comparisons. Then you run RL, usually PPO, to push the policy toward higher reward, with a KL penalty back to the SFT model so it doesn't drift off. The fragile bit is the reward model: the policy is actively searching for its blind spots, so true quality tends to peak and then decline while the reward score keeps going up."

---

### Q12: What is DPO and how does it differ from RLHF?

**Answer:**

**DPO (Direct Preference Optimization):**
- Directly optimizes policy to prefer chosen over rejected
- No reward model needed
- Uses reference model instead

**Key Difference:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Yes | No |
| **Reference Model** | Used in RL | Used directly |
| **Complexity** | High | Lower |
| **Flexibility** | More | Less |

**DPO Loss:**
```
Loss = -log(σ(β * (log π_chosen - log π_rejected - log π_ref_chosen + log π_ref_rejected)))
```

Where σ is sigmoid, β is temperature.

**Where the DPO loss comes from — the derivation that makes it click.**

The KL-constrained RLHF objective has a known closed-form optimum. If you maximise expected reward minus $\beta$ times KL to the reference, the optimal policy is

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

that is, the reference policy reweighted by exponentiated reward. This is not usable directly because the partition function $Z(x)$ sums over all possible responses. But rearrange it to solve for the reward instead:

$$r(x, y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Now substitute this into the Bradley-Terry preference likelihood $\sigma(r(x, y_w) - r(x, y_l))$. Because the two responses share the same prompt $x$, the intractable $\beta \log Z(x)$ term appears in both and **cancels exactly**. What is left is a loss over the policy alone:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

That is the formula in the box above, now with a reason. The insight is that the language model *is* its own reward model — the log-ratio to the reference is an implicit reward — so you never have to instantiate a separate one, and you never have to sample from the policy during training. DPO is a supervised loss on a fixed dataset of pairs.

**What the gradient does.** Differentiating gives a weight of $\sigma(\hat{r}_l - \hat{r}_w)$ on each pair: the update is large when the implicit reward currently ranks the pair *wrongly* and small when it already ranks it correctly. So DPO automatically focuses on the examples it has not yet learned, which is the same self-limiting behaviour that makes the logistic loss well-behaved.

**Where DPO is genuinely worse.** The comparison table is right that DPO is less flexible, and the mechanism is worth stating: DPO trains on a *fixed, off-policy* dataset, so it never sees the responses the current policy actually generates. RLHF's online loop keeps collecting fresh samples and scoring them, which is how it discovers and corrects new failure modes. A known DPO pathology follows from the loss: nothing constrains the *absolute* likelihood of the chosen response, only the gap, so the optimiser can and often does reduce the probability of both responses while widening the difference — pushing mass onto entirely unseen text. Practical fixes include adding an SFT term on the chosen responses (this is roughly what RPO/CPO do), and running iterative DPO where you periodically sample from the current policy, label those samples, and retrain.

**Follow-up:** *What does $\beta$ control in DPO?* It is the same KL strength as in RLHF, but it enters as the inverse temperature of the implicit reward. Small $\beta$ (say 0.01) means the policy is allowed to move far from the reference and fits preferences aggressively; large $\beta$ (0.5) keeps it tethered. Typical values are 0.1 to 0.5. Because the reference model appears explicitly in the loss, its log-probabilities can be precomputed once for the whole dataset, so DPO needs only one model in memory during training rather than PPO's three or four.

> **Why the interviewer asks this.** DPO is the standard alignment method most teams actually run; knowing that it comes from analytically solving the RLHF objective — rather than being a separately invented heuristic — is what distinguishes reading the paper from reading a blog summary.

> **Saying it out loud.** "DPO comes from noticing that the KL-constrained RLHF objective has a closed-form optimal policy, and if you invert it you can write the reward in terms of the policy's log-ratio against the reference. Plug that into the preference likelihood and the intractable normalising term cancels, because both responses share a prompt. So you end up with a plain supervised loss on preference pairs — no reward model, no sampling, no RL loop. The cost is that it's off-policy: it only ever sees the pairs in your dataset, so it can't discover new failure modes the way an online loop does. And there's a known quirk where it lowers the probability of the chosen response too, as long as the gap widens."

---

### Q13: Explain PPO (Proximal Policy Optimization) in detail. Why is it used in RLHF?

**Answer:**

**What is PPO?**
PPO is a policy gradient algorithm that prevents large policy updates by clipping the objective function.

**Mathematical Formulation:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s) (importance sampling ratio)
- A: Advantage estimate
- ε: Clipping parameter (typically 0.1-0.3)
```

**Why Clipping?**
- Prevents large updates that can destabilize training
- Policy changes gradually (more stable)
- Can reuse same data multiple times (sample efficient)

**Why PPO in RLHF:**
1. **Stability**: Language models are sensitive - need stable updates
2. **Sample efficiency**: Human feedback is expensive - reuse data
3. **KL constraint**: Keeps policy close to reference
4. **Proven**: Works well in practice (ChatGPT, Claude)

**PPO Algorithm:**
1. Collect trajectories with current policy
2. Compute advantages A(s,a)
3. For K epochs:
   - Compute ratio r(θ) = π_θ / π_θ_old
   - Compute clipped objective
   - Update policy
4. Update old policy

**Why the ratio and the clip are shaped the way they are.**

Vanilla policy gradient computes $\nabla_\theta \mathbb{E}[R]$ from samples drawn by the *current* policy, so as soon as you take one gradient step the data is stale and must be thrown away. That is intolerable when each sample requires generating a full response from a large language model. Importance sampling fixes it: you can estimate an expectation under $\pi_\theta$ using samples from $\pi_{\theta_{old}}$ by reweighting each sample by $\rho = \pi_\theta(a \mid s) / \pi_{\theta_{old}}(a \mid s)$. That is where the ratio in the objective comes from — it is what licenses multiple epochs over the same batch.

The problem is that importance sampling has unbounded variance: if the policy moves far from the sampler, some ratios blow up and a single sample dominates the gradient. TRPO handled this with a hard KL trust-region constraint and a second-order solve. PPO's contribution is to get almost the same effect with a first-order trick — clip the ratio and take the pessimistic branch:

$$L^{CLIP} = \mathbb{E}\big[\min\big(\rho A,\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon)A\big)\big]$$

**Read the four cases and it becomes obvious.** With $\epsilon = 0.2$: if the advantage $A > 0$ (the action was better than expected) and $\rho$ has already grown past $1.2$, the clipped branch is flat, so the gradient is zero — you have already increased this action's probability enough for one batch, stop. If $A > 0$ and $\rho < 1$, no clipping applies and you get the full gradient. If $A < 0$ and $\rho$ has fallen below $0.8$, again flat, stop pushing it down. If $A < 0$ and $\rho > 1$, the unclipped term is *more* negative, and the `min` selects it — so a bad action whose probability accidentally increased still gets a full corrective gradient. The `min` is what makes the bound pessimistic rather than merely bounded: it only ever removes incentive to move further, never removes a correction.

**The advantage, and why RLHF needs a value model.** $A(s, a) = Q(s, a) - V(s)$ asks "was this action better than the average action from this state", and subtracting the state-value baseline $V$ removes variance without introducing bias. PPO estimates it with GAE (generalised advantage estimation), an exponentially weighted average over $n$-step temporal-difference errors controlled by $\lambda$, which trades bias against variance. In RLHF the reward arrives only at the end of the response, so the value model's job is to spread that single scalar back across hundreds of token-level decisions. This is precisely the component GRPO removes.

**The full RLHF-PPO loop in memory terms.** Four models are live: the policy being trained, a frozen reference for the KL term, the reward model, and the value model. That is the practical reason PPO-based RLHF is hard to run, and the reason both DPO (drop the RL loop) and GRPO (drop the critic) found adoption.

**Follow-up:** *What typically goes wrong when PPO training destabilises?* Watch three numbers. If the KL to the reference climbs steadily, the policy is drifting and reward hacking usually follows — most implementations use an adaptive $\beta$ that increases when KL exceeds a target. If the clip fraction (the share of tokens hitting the clip boundary) rises above roughly 20-30%, the policy is moving too fast per batch, so lower the learning rate or take fewer epochs per batch. If entropy collapses, the policy has converged to a narrow set of phrasings — mode collapse — and an entropy bonus or a stronger KL term is the usual response.

> **Why the interviewer asks this.** They want to see whether you can explain the clip as a variance-control mechanism for importance sampling, rather than as "it stops big updates".

> **Saying it out loud.** "PPO exists so you can reuse a batch of samples for more than one gradient step. You do that with an importance ratio — new policy probability over old — but that ratio has nasty variance if the policy moves too far, so PPO clips it to a narrow band around one and takes the pessimistic branch with a min. The effect is that once you've pushed a good action's probability up by twenty percent, the gradient goes flat for that batch, but if a bad action's probability went up you still get the full correction. In RLHF you also need a value model, because the reward only shows up at the end of the response and something has to spread that credit across all the tokens."

---

### Q14: What is GRPO (Group Relative Policy Optimization)? When is it useful?

**Answer:**

**What is GRPO?**
GRPO (Shao et al., *DeepSeekMath*, 2024; the algorithm behind DeepSeek-R1) is a variant of PPO that **removes the learned value network**. The "group" in the name is a group of sampled responses to the *same prompt*, not a group of users. For each prompt, the policy samples $G$ completions (typically 8-64), each is scored, and the group's own reward statistics serve as the baseline that PPO would otherwise get from a critic.

**Mathematical Formulation:**
For a prompt $q$, sample outputs $o_1, \dots, o_G \sim \pi_{\theta_{old}}(\cdot \mid q)$ with rewards $r_1, \dots, r_G$. The advantage for output $i$ is the reward standardised within the group:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

and this scalar is assigned to every token of that output. The objective is then the usual PPO clipped surrogate with a KL term:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\min\big(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\big)\right] - \beta\, D_{KL}(\pi_\theta \parallel \pi_{ref})$$

where $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})$ is the per-token importance ratio.

**Why GRPO?**
- **No critic**: PPO needs a value model roughly the size of the policy, so it holds four models in memory (policy, reference, reward, value). Dropping the critic cuts memory and removes a second network that itself has to be trained and can be badly calibrated on long generations.
- **Natural fit for verifiable rewards**: When the reward is a rule-based checker — does the maths answer match, do the unit tests pass — you get a clean scalar per sample and no reward model is needed either, leaving just policy and reference.
- **Comparison is the signal**: The advantage says "was this attempt better or worse than my other attempts at the same problem", which is exactly the credit-assignment question, and it needs no learned estimate of expected return.

**Use Cases:**
- Reasoning with verifiable answers: mathematics, competitive programming, unit-test-checked code — its original and strongest domain.
- Any setting where sampling several completions is cheap relative to training a critic.

**Example:**
- Prompt: a maths problem. Sample 8 chains of thought; 3 reach the right answer, 5 do not.
- Rewards are $[1, 1, 1, 0, 0, 0, 0, 0]$, so mean $= 0.375$ and std $\approx 0.484$.
- Advantages are $\approx +1.29$ for the three correct chains and $\approx -0.77$ for the five wrong ones, pushing probability mass toward whatever the successful chains did — with no value network anywhere in the loop.

> **Note on a correction.** An earlier version of this answer described GRPO as optimising across *demographic or user groups with different preferences*. That is not what GRPO is. The "group" is a group of sampled completions for one prompt, and the method's purpose is to eliminate the value network. The text above has been corrected.

**Why standardising within the group is the right baseline.**

Any policy-gradient method needs a baseline subtracted from the reward, because $\mathbb{E}[\nabla \log \pi \cdot b] = 0$ for any $b$ that does not depend on the action — so subtracting a baseline reduces variance without introducing bias. PPO learns that baseline with a value network. GRPO observes that if you sample $G$ completions from the *same* prompt, their mean reward is already an unbiased, zero-cost estimate of that prompt's expected return. No network, no training, no calibration drift. Dividing by the group standard deviation additionally normalises the advantage scale across prompts, so an easy prompt where everything scores 0.9-1.0 and a hard prompt where everything scores 0.0-0.1 contribute comparably sized gradients.

That normalisation is also GRPO's main known bias. If every completion in a group gets the same reward — all correct or all wrong — the standard deviation is zero, the advantage is undefined or zero, and the prompt contributes nothing; those prompts are simply wasted compute. And dividing by the standard deviation systematically upweights prompts where the group happened to disagree, which is a difficulty bias rather than a learning signal. Later variants (Dr. GRPO and the loss-normalisation fixes in DAPO) drop the standard-deviation divisor and change the token-length normalisation for exactly these reasons.

**How this pairs with verifiable rewards.** GRPO became prominent because it composes so cleanly with rule-based rewards. If your reward is "does the final answer match" or "do the tests pass", you have removed the reward model; GRPO removes the value model; and you are left with a policy and a frozen reference. That configuration — sample many attempts, check them, upweight the ones that worked — is what produced the long chain-of-thought behaviour in DeepSeek-R1, where the model learned to backtrack and re-derive without ever being shown a demonstration of doing so.

**Follow-up:** *What is the practical cost?* Compute at sampling time. You are generating $G$ full completions per prompt instead of one, so a group size of 16 means 16 times the generation work per prompt — and generation, not the gradient step, dominates RL wall-clock for LLMs. The trade you are making is critic memory and critic instability in exchange for sampling throughput, which is a good trade when generation is well-optimised and a bad one when it is not. Group size is a real hyperparameter: too small and the baseline is noisy, too large and you are paying for redundant samples.

> **Why the interviewer asks this.** It is a recency check with substance: GRPO is only useful to know if you can say what it removes and what it costs, not just that DeepSeek used it.

> **Saying it out loud.** "GRPO is PPO with the critic deleted. Instead of learning a value function to tell you what reward to expect from a prompt, you sample a whole group of completions for that prompt — say sixteen — and use the group's own mean reward as the baseline, standardised by the group's standard deviation. So the advantage is just 'did this attempt beat my other attempts at the same problem'. That kills one of the four models you'd otherwise hold in memory, and it pairs beautifully with verifiable rewards like 'did the tests pass', because then you don't need a reward model either. The cost is generation: you're producing sixteen completions per prompt instead of one."

---

### Q15: What are the main challenges in RL alignment? How do you address them?

**Answer:**

**Challenge 1: Reward Hacking**
- **Problem**: Model finds ways to maximize reward that don't align with intent
- **Solution**: Careful reward design, KL penalty, monitoring

**Challenge 2: Distribution Shift**
- **Problem**: Policy changes, but reward model trained on old distribution
- **Solution**: Retrain reward model periodically, regularization

**Challenge 3: Mode Collapse**
- **Problem**: Policy collapses to single response pattern
- **Solution**: KL penalty, entropy bonus, diverse training data

**Challenge 4: Instability**
- **Problem**: Training can be unstable
- **Solution**: PPO clipping, gradient clipping, learning rate scheduling

**Challenge 5: Human Feedback Quality**
- **Problem**: Inconsistent or biased feedback
- **Solution**: Multiple annotators, quality control, bias detection

**The common root.** Four of the five challenges above are the same phenomenon seen from different angles: you are optimising a *proxy* (a learned reward model) rather than the thing you care about (human judgement), using a policy that actively searches the proxy's input space. Reward hacking is the proxy being exploited; distribution shift is the proxy being evaluated off its training distribution; mode collapse is the policy concentrating on the narrow region where the proxy scores highest; instability is the optimisation running away in that direction. Framing it as one problem rather than five is what makes the mitigations cohere — the KL penalty appears in three of the five rows because it directly bounds how far into unmeasured territory the policy is allowed to go.

**A number worth having.** Gao, Schulman and Hilton's scaling-law study of reward-model overoptimisation found that true reward, plotted against the square root of the KL divergence from the initial policy, rises and then falls in a predictable arc, with the peak moving further out as the reward model gets larger and is trained on more data. The practical implication is concrete: KL distance is the right x-axis for deciding when to stop, and "stop when the RM score plateaus" is the wrong criterion because the RM score keeps rising past the point where quality starts dropping.

**On feedback quality specifically.** The failure is not only noise but *systematic* bias, and the biases are known: annotators reliably prefer longer responses, more confident phrasing, and more formatting (lists, bold headers) independent of content. Length bias is severe enough that some RLHF runs produce models that are longer and no better, and length-controlled evaluation exists precisely because of it. Mitigations are measurement-first — check the correlation between reward and response length, and if it is high, either debias the reward (subtract a length term) or resample the preference data to decorrelate them.

**Follow-up:** *How would you tell reward hacking apart from genuine improvement?* Use a held-out signal the policy is not being optimised against. Concretely: a separate reward model trained on different data or with a different seed (if scores diverge between the two, at least one is being gamed), pairwise human evaluation on a fresh sample, and task benchmarks the RM never saw. Add cheap behavioural monitors for the known degenerate modes — mean response length, refusal rate, n-gram repetition, entropy — because hacking usually shows up as a sharp change in one of those before it shows up in an aggregate score.

> **Why the interviewer asks this.** The list is easy to memorise; they are listening for whether you understand that these are symptoms of proxy optimisation and can name what you would actually monitor.

> **Saying it out loud.** "Almost all of these are one problem wearing different hats: you're optimising a learned proxy for human judgement, and the policy is actively hunting for places where that proxy is wrong. So true quality goes up, peaks, and then falls while the reward score keeps climbing. The main lever is the KL penalty back to the reference model, because it bounds how far you can get from the region where the reward model was actually trained. Beyond that I'd hold out a second reward model and some real human eval as an unhacked signal, and I'd watch for the obvious tells — response length creeping up, entropy collapsing, refusal rate spiking."

---

### Q16: How do you prevent reward hacking in RLHF?

**Answer:**

**What is Reward Hacking?**
Model finds unintended ways to maximize reward (e.g., always says "I can't answer").

**Prevention:**
1. **Careful reward design**: Multiple signals, penalize hacks
2. **Regularization**: KL penalty prevents extreme behaviors
3. **Reward model robustness**: Diverse training, bias detection
4. **Monitoring**: Track patterns, detect anomalies
5. **Constrained optimization**: Hard/soft constraints
6. **Iterative refinement**: Identify hacks, refine reward

**Why this is hard rather than just fiddly.** Reward hacking is not a bug in a particular reward model; it is what optimisation *does* to any imperfect proxy. Goodhart's law in its sharpest form: the reward model and human judgement agree on the distribution where preferences were collected, and the optimiser's job is to leave that distribution. So the goal is never "build an unhackable reward" — it is to make the gap costly to reach and to notice when it has been reached.

**Concrete hacks that actually show up.** Naming these is more convincing in an interview than abstractions. *Length inflation*: annotators mildly prefer longer answers, so the policy learns to pad, and reward rises with word count. *Sycophancy*: agreeing with the user's stated view scores well, so the model stops contradicting false premises. *Formatting theatre*: bullet points, bold headers and a confident summary paragraph score well independent of content. *Hedged non-answers*: on hard prompts, a fluent refusal outscores a wrong attempt, so the model learns to refuse more. *Confident fabrication*: certainty is rewarded, uncertainty is not, so calibration degrades. Each is a case of the reward model having learned a *correlate* of quality rather than quality.

**The one mechanism to explain properly.** Of the six prevention items listed above, the KL penalty is the load-bearing one, and the reason is stated precisely in Q17: it caps how far the policy can travel from the region where the reward model's judgements were actually validated. Reward-model ensembles are the second most useful, because a hack usually exploits an idiosyncrasy of one model's error surface; taking the *minimum* (pessimistic) score across an ensemble, rather than the mean, penalises responses that any member distrusts and measurably delays overoptimisation.

**Follow-up:** *What if the reward is a verifier rather than a learned model?* It is harder to hack but not immune, and the failures are different in kind: unit tests get special-cased (`if input == test_case_1: return 42`), maths answers get asserted without a valid derivation, and any reward that only checks the final answer will accept a correct answer reached by an incoherent chain. So the mitigation shifts from KL-style constraints to test coverage, hidden held-out tests, and process-level checking of the reasoning rather than only the outcome.

> **Why the interviewer asks this.** Anyone can list mitigations; they want to hear that you expect hacking by default and have a detection story, not just a prevention story.

> **Saying it out loud.** "I'd start by saying reward hacking isn't a bug you fix, it's what optimisation does to any imperfect proxy — the policy's whole job is to find where your reward model is wrong. The concrete versions are pretty consistent: answers get longer, more sycophantic, more heavily formatted, and more prone to a confident hedge instead of a real attempt. The main defence is the KL penalty, which keeps the policy near the distribution the reward model was actually trained on. After that I'd ensemble a couple of reward models and take the pessimistic score, and I'd monitor length, entropy and refusal rate, because hacking normally shows up there before it shows up in any aggregate metric."

---

### Q17: Explain the KL penalty in RLHF. Why is it important?

**Answer:**

**What is KL Penalty?**
KL divergence measures how different policy is from reference. Penalty prevents large deviations.

**Mathematical Formulation:**
```
KL(π_θ || π_ref) = E[log(π_θ(a|s) / π_ref(a|s))]

In practice:
KL_penalty = β * (log π_θ - log π_ref)
```

**Why Important:**
1. **Prevents mode collapse**: Keeps policy diverse
2. **Prevents reward hacking**: Constrains to reasonable behaviors
3. **Maintains capabilities**: Preserves SFT capabilities
4. **Stability**: Prevents large policy changes
5. **Trust region**: Policy can't deviate too far

**How to Choose β:**
- Too small: Policy can deviate too much
- Too large: Policy can't learn
- Typical: β = 0.1-0.5

**What the penalty is doing, geometrically.** KL divergence $D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ measures how much the trained policy's distribution has moved from the frozen reference. Adding $-\beta D_{KL}$ to the reward turns unconstrained reward maximisation into a *trust region*: the policy may buy reward, but it pays in distance, and the exchange rate is $\beta$. The reason this is the right currency is that the reward model's judgements are only validated near the reference distribution — that is where the preference data was collected. KL distance is therefore a direct proxy for "how far outside my measurement is this".

**How it is actually computed.** You cannot evaluate the true expectation over all sequences, so implementations use the per-token log-ratio of the sampled tokens, $\log \pi_\theta(y_t \mid \cdot) - \log \pi_{\text{ref}}(y_t \mid \cdot)$, as a single-sample estimator and subtract $\beta$ times it from the reward at each token. That naive estimator is unbiased but high-variance and can go negative, which is confusing to read on a dashboard; the common fix is the low-variance estimator $\hat{k}_3 = (\rho - 1) - \log \rho$ where $\rho = \pi_{\text{ref}} / \pi_\theta$, which is always non-negative and much less noisy. Note also the implementation choice: adding the penalty *into the reward* (so it flows through the advantage and the value function) behaves differently from adding it as a separate loss term, and the two are not equivalent — the original RLHF papers put it in the reward.

**A small worked example of what $\beta$ buys.** Suppose at some token the reference gives probability 0.10 to the token the policy sampled and the policy now gives it 0.60. The log-ratio is $\log(0.6/0.1) = 1.79$. With $\beta = 0.1$ the penalty is $0.179$ reward units at that token; with $\beta = 0.5$ it is $0.90$. If the reward model's scale is such that a typical good-versus-bad gap is around 1 unit, then at $\beta = 0.5$ this single-token deviation has consumed most of the available reward and the policy will not make it unless the payoff is genuinely large — whereas at $\beta = 0.1$ it is nearly free. That is the whole tuning intuition, and it also explains why $\beta$ cannot be transferred between runs without checking the reward model's scale.

**Adaptive $\beta$.** Because the right $\beta$ depends on that scale, most implementations do not fix it. They set a *target* KL and use a controller that raises $\beta$ when measured KL exceeds the target and lowers it when it falls below. This makes the hyperparameter you tune ("how far am I willing to drift") interpretable in a way a raw coefficient is not.

**Follow-up:** *Which direction of KL is used, and does it matter?* RLHF uses the forward-from-the-policy form $D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$, which is *mode-seeking*: it heavily punishes the policy for putting mass where the reference puts almost none, but does not punish it for abandoning modes the reference covered. That asymmetry is convenient — it blocks the policy from inventing wholly new behaviour — but it is also part of why RLHF reduces output diversity, since dropping modes is cheap under this direction. The reverse direction would be mass-covering and would preserve diversity, but it would require sampling from the reference, which is more expensive.

> **Why the interviewer asks this.** It is the one hyperparameter that decides whether an RLHF run works, so knowing what it trades off — and that it is usually adaptive — is a strong signal of hands-on experience.

> **Saying it out loud.** "The KL penalty measures how far the policy has drifted from the frozen reference model and charges the policy for that drift. The reason it matters is that your reward model was only ever validated near the reference distribution — the preference data came from there — so KL distance is basically a measure of how far outside your measurements you've wandered. Too small a beta and the policy runs off and hacks the reward; too large and it can't learn anything. In practice people don't fix beta at all — they set a target KL and let a controller adjust beta to hit it, which makes the knob you're tuning something you can actually reason about."

---

See `08_training_techniques/rl_alignment_qa.md` for even more detailed answers!

---

## Optimization

### Q18: Explain the Adam optimizer.

**Answer:**

**Adam = Adaptive Moment Estimation**

**Components:**
1. **First moment (m)**: Exponential moving average of gradients (momentum)
2. **Second moment (v)**: Exponential moving average of squared gradients (variance)
3. **Bias correction**: Fix initial bias

**Update Rule:**
```
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * g_t²
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
```

**Why it works:**
- Adaptive learning rates per parameter
- Momentum for smooth updates
- Second moment adapts to gradient variance

**Default hyperparameters:**
- β1 = 0.9 (momentum)
- β2 = 0.999 (variance)
- α = 0.001 (learning rate)

**Reading the update rule as two separate ideas glued together.**

Adam is momentum and RMSProp stacked, plus a correction for how they start.

*The first moment is momentum.* $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ is an exponential moving average of past gradients, which with $\beta_1 = 0.9$ has an effective window of about $1/(1-\beta_1) = 10$ steps. Averaging helps because minibatch gradients are noisy estimates of the true gradient: components that point consistently in one direction survive the average, components that flip sign cancel. In a long narrow ravine — the typical shape of a neural loss surface — this damps the oscillation across the valley and accumulates speed along it.

*The second moment is a per-parameter learning rate.* $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ tracks the average *squared* gradient with a much longer window, about 1000 steps at $\beta_2 = 0.999$. Dividing the step by $\sqrt{\hat{v}_t}$ means a parameter whose gradients have been consistently large takes small steps and one whose gradients have been small takes large ones. The effective step size becomes roughly $\alpha \cdot \text{mean}(g) / \text{RMS}(g)$, which is close to a signal-to-noise ratio: parameters with a consistent gradient direction move at nearly the full learning rate, parameters whose gradient is mostly noise barely move. This is why Adam works out of the box on problems — like transformers, with embeddings whose gradients are extremely sparse and layer norms whose gradients are not — where a single global learning rate would be badly wrong for most parameters.

*Bias correction is not cosmetic.* Both averages start at zero, so early on they are biased toward zero. Concretely: at $t = 1$, $v_1 = (1 - 0.999) g_1^2 = 0.001 g_1^2$, so $\sqrt{v_1} \approx 0.032 |g_1|$ — about 30 times too small, which would make the first step about 30 times too large and can blow the model up immediately. The correction divides by $1 - \beta_2^t$, which at $t=1$ is exactly $0.001$, restoring $\hat{v}_1 = g_1^2$. The correction decays to nothing as $\beta^t \to 0$, so it only matters for roughly the first few thousand steps — precisely the fragile part of training. (Even with it, transformers usually still need learning-rate warmup, because the *variance* of Adam's update, not just its mean, is large when $v$ is estimated from few samples.)

*Epsilon* sits inside the square root's denominator to stop division by zero, but it also quietly sets a floor: with $\epsilon = 10^{-8}$, any parameter whose gradient RMS falls below that gets a step proportional to $g/\epsilon$ rather than a normalised one. Raising $\epsilon$ to $10^{-6}$ or $10^{-4}$ is a standard stability fix for large-model training.

**Worked micro-example.** Take a parameter with a steady gradient $g = 0.1$ every step, $\alpha = 0.001$. After a few steps $\hat{m} \approx 0.1$ and $\hat{v} \approx 0.01$, so $\sqrt{\hat{v}} = 0.1$ and the update is $0.001 \times 0.1/0.1 = 0.001$ — the full learning rate. Now a parameter whose gradient alternates $+0.1, -0.1$: $\hat{m} \approx 0$ but $\hat{v} \approx 0.01$ still, so the update is near zero. Same gradient magnitude, opposite treatment, based purely on consistency.

**Follow-up:** *Why does Adam use so much memory?* It stores $m$ and $v$ per parameter, so with FP32 master weights the optimiser state is 8 bytes per parameter on top of the 4 bytes of weights and 4 of gradients — roughly 16 bytes per parameter in total, meaning a 7B model needs about 112 GB just to hold training state before activations. This is exactly what ZeRO shards across devices, what 8-bit Adam quantises, and what memory-light optimisers like Adafactor (which factorises $v$ into row and column statistics) and Lion (which keeps only momentum) are trying to avoid.

> **Why the interviewer asks this.** Everyone uses Adam; few can say what the second moment is doing or why the bias correction exists, and the bias-correction question in particular has a crisp right answer.

> **Saying it out loud.** "Adam is two ideas together. The first moment is momentum — an exponential average of the gradients, so consistent directions build up and noisy ones cancel. The second moment is an average of squared gradients, and dividing by its square root gives every parameter its own effective learning rate; roughly you're stepping by the signal-to-noise ratio of that parameter's gradient. The bias correction is there because both averages start at zero, so early on they're way too small — at step one the second moment is a thousandth of its true value, which would make your first step enormous. And the cost is memory: two extra values per parameter."

---

### Q19: What's the difference between Adam and AdamW?

**Answer:**

**Adam:** Weight decay applied to gradients
**AdamW:** Weight decay decoupled (applied separately)

**Why AdamW:**
- Better weight decay
- Improved generalization
- More principled

**Difference:**
```python
# Adam: weight decay in gradient
gradient = gradient + weight_decay * params

# AdamW: weight decay separate
params = params - lr * (adam_update + weight_decay * params)
```

**Why decoupling actually changes the answer.**

L2 regularisation and weight decay are the same thing for plain SGD and *different* things for Adam — that is the whole content of the AdamW paper, and it is worth being able to show why.

With L2 regularisation you add $\frac{\lambda}{2}\lVert w \rVert^2$ to the loss, so the gradient becomes $g + \lambda w$. Feed that into Adam and the penalty term goes through the same second-moment normalisation as everything else: the update is

$$\Delta w = -\alpha \frac{\hat{m}(g + \lambda w)}{\sqrt{\hat{v}(g + \lambda w)} + \epsilon}$$

Because the denominator is large for parameters with large historical gradients, the decay those parameters receive is *divided down*. The consequence is backwards: parameters with big, consistent gradients — the ones doing the most work and most at risk of growing large — get the *least* regularisation, while parameters with tiny gradients get the most. The decay strength ends up coupled to the gradient history, which is not what anyone intends when they set a weight decay value.

AdamW removes the penalty from the gradient entirely and applies it directly to the weights after the adaptive step:

$$w \leftarrow w - \alpha\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w\right)$$

Now every parameter shrinks by the same relative amount per step regardless of its gradient statistics, which is what "weight decay" is supposed to mean. A practical side effect: the two hyperparameters decouple, so tuning the learning rate no longer silently changes the effective regularisation, which is why AdamW's hyperparameters transfer between runs far better.

**Numeric illustration.** Two parameters, both at $w = 1.0$, both with $\lambda = 0.01$. Parameter A has had a gradient RMS around $1.0$; parameter B around $0.01$. Under Adam-with-L2 the decay contributions are roughly $\lambda w / \sqrt{\hat{v}}$, so A gets $0.01/1.0 = 0.01$ and B gets $0.01/0.01 = 1.0$ — a hundredfold difference in regularisation strength that nobody asked for. Under AdamW both get exactly $\alpha \lambda w$.

**Practical note.** Because the decay is now applied to every parameter uniformly, you generally want to *exclude* biases, LayerNorm gains and (often) embeddings from it — decaying a normalisation gain toward zero has no regularising interpretation and just fights the layer. Every serious training script has a parameter-group split doing exactly this. Typical values are $\lambda = 0.1$ for large transformer pretraining and $0.01$ for fine-tuning; note these are much larger than classic L2 values, another consequence of the rescaling.

**Follow-up:** *If AdamW is strictly better, why does `torch.optim.Adam` still have a `weight_decay` argument?* Because it implements the coupled L2 form, which is what the original Adam paper described and what a lot of older code depends on for reproducibility. `Adam(weight_decay=0.01)` and `AdamW(weight_decay=0.01)` do genuinely different things — a real source of silent discrepancies when porting a config between codebases.

> **Why the interviewer asks this.** It looks like trivia but has a real mechanism behind it, and the answer reveals whether you have ever debugged why a regularisation setting did not behave as expected.

> **Saying it out loud.** "For plain SGD, adding an L2 term to the loss and decaying the weights directly are the same operation. For Adam they aren't, because the L2 term goes through the second-moment normalisation along with the real gradient. That means parameters with big gradient histories get their decay divided down and end up barely regularised, while quiet parameters get hammered — the opposite of what you want. AdamW just applies the decay straight to the weights after the adaptive step, so everything shrinks by the same proportion. It also decouples the two hyperparameters, so changing your learning rate stops secretly changing your regularisation."

---

## Regularization

### Q20: Explain L1 vs L2 regularization.

**Answer:**

**L1 (Lasso):**
- Penalty: λ * Σ|w|
- Effect: Many weights become exactly 0
- Use: Feature selection, sparsity
- Gradient: Constant (λ * sign(w))

**L2 (Ridge):**
- Penalty: λ * Σw²
- Effect: Weights shrink toward 0 (but not 0)
- Use: Generalization, most common
- Gradient: Linear (2λ * w)

**When to use:**
- L1: Want feature selection, interpretability
- L2: Want generalization, standard choice

**Elastic Net:** Combines both

**Why L1 gives exact zeros and L2 does not — the gradient argument.**

This is the part an interviewer will push on, and the cleanest explanation is about what happens near zero. Consider a single weight $w$ with data-fit gradient $g$ at the optimum of the unregularised loss.

Under L2, the penalty gradient is $2\lambda w$. As $w \to 0$ that gradient also $\to 0$, so the shrinkage force vanishes exactly where you would need it to be strongest. The weight settles wherever the data-fit gradient balances $2\lambda w$, which for any nonzero $g$ is a nonzero $w$. Solve the one-dimensional case and you get proportional shrinkage: $\hat{w}_{\text{ridge}} = \hat{w}_{\text{OLS}} / (1 + \lambda)$ — everything scaled down by a constant factor, nothing set to zero.

Under L1, the penalty gradient is $\lambda \cdot \text{sign}(w)$, which has magnitude $\lambda$ *no matter how small $w$ is*. So there is a constant force pushing toward zero that does not weaken on approach. If the data-fit gradient at $w = 0$ has magnitude less than $\lambda$, it cannot overcome that force and the weight is pinned at exactly zero. The closed form for the orthonormal case is the soft-threshold operator

$$\hat{w}_{\text{lasso}} = \text{sign}(\hat{w}_{\text{OLS}}) \cdot \max\big(|\hat{w}_{\text{OLS}}| - \lambda,\ 0\big)$$

which says it plainly: subtract $\lambda$ from the magnitude, and if that goes negative, clamp to zero.

**A worked number.** With $\lambda = 0.5$ and OLS estimates $[3.0, 0.4, -0.3]$: L1 gives $[2.5, 0, 0]$ — the two small coefficients are eliminated and the large one is shifted. L2 with the same $\lambda$ gives $[2.0, 0.267, -0.2]$ — everything shrunk by a third, nothing removed. Same regularisation strength, completely different structure in the answer.

**The geometric picture, stated properly.** The constraint region for L1 is a diamond (a cross-polytope) whose corners lie *on* the axes; for L2 it is a sphere with no corners. The solution is where the elliptical contours of the squared-error loss first touch the constraint region, and an ellipse touching a diamond will generically touch at a corner — a corner being exactly a point where some coordinates are zero. A sphere has no preferred points, so contact happens at a generic location with all coordinates nonzero. This picture and the soft-threshold formula are two views of the same fact.

**Consequences you should mention.** L1 is not differentiable at zero, so plain gradient descent cannot land exactly on zero; solvers use coordinate descent, proximal gradient (ISTA/FISTA), or LARS. L1 handles correlated features badly — given two nearly identical predictors it picks one essentially arbitrarily and zeroes the other, and which one it picks is unstable across resamples, whereas L2 splits the weight between them. Elastic net, $\lambda_1\lVert w\rVert_1 + \lambda_2 \lVert w \rVert_2^2$, exists specifically to keep sparsity while making correlated groups enter or leave together.

**Follow-up:** *Should you regularise the intercept?* No. The intercept is not a feature weight; shrinking it biases predictions toward zero output rather than toward a simpler function, and it makes the fit depend on where you happened to centre the target. Standard implementations exclude it — and for the same reason you should standardise features before regularising, since otherwise the penalty is applied in units that depend on arbitrary measurement scales.

> **Why the interviewer asks this.** The "L1 gives sparsity" fact is universal knowledge; the constant-gradient-near-zero explanation is not, and it is a clean test of whether you think in terms of what the optimiser actually does.

> **Saying it out loud.** "Both add a penalty on the size of the weights, but the shape of the penalty changes the outcome. L2's gradient is proportional to the weight, so as a weight gets small the pressure on it gets small too — it shrinks toward zero and never arrives. L1's gradient is a constant lambda times the sign, so the push toward zero is just as strong at 0.001 as at 10. If the data's pull on that weight is weaker than lambda, it gets pinned at exactly zero. That's why L1 does feature selection and L2 doesn't. Geometrically it's the diamond-versus-circle picture — the diamond has corners on the axes and the loss contours tend to hit a corner."

---

### Q21: How does dropout work?

**Answer:**

**During Training:**
1. Randomly set some activations to 0 (with probability p)
2. Scale remaining activations by 1/(1-p)
3. This prevents co-adaptation

**During Inference:**
- No dropout
- Scale all activations by (1-p) to maintain expected value

**Why it works:**
- Prevents neurons from co-adapting
- Forces model to be robust
- Acts as ensemble of sub-networks

**Common rates:**
- Input layer: 0.1-0.2
- Hidden layers: 0.5
- Output layer: Usually no dropout

**The scaling detail, made precise.**

There are two equivalent implementations and it is worth knowing which one frameworks use, because interviewers ask. *Classic dropout* zeroes activations with probability $p$ at training time and does nothing else, then at test time multiplies every activation by $(1-p)$ to match the expected value. *Inverted dropout* — what PyTorch, TensorFlow and everyone else actually implements — does the scaling at training time instead: zero with probability $p$, then divide the survivors by $(1-p)$. Test time is then a pure no-op, which is what you want, because inference should not need to know the training configuration.

The arithmetic: if a unit has value $a$ and survives with probability $1-p$, then $\mathbb{E}[\text{output}] = (1-p) \cdot \frac{a}{1-p} + p \cdot 0 = a$. The expectation is preserved exactly, so the layers downstream see inputs with unchanged mean magnitude and do not have to be retuned between training and inference. (The description in the answer above states the classic form; note that in real frameworks the $1/(1-p)$ scaling happens during training and inference does nothing.)

**Why it regularises, in three compatible framings.** *Co-adaptation:* a unit cannot rely on a specific other unit being present, because that other unit vanishes half the time, so features must be individually useful rather than useful only in a fixed committee. *Implicit ensembling:* a network with $n$ droppable units defines $2^n$ subnetworks sharing weights; training samples one per minibatch and inference with the scaled weights approximates averaging over all of them, which is a very cheap ensemble. *Noise injection:* dropout is multiplicative Bernoulli noise on activations, and it can be shown that for a linear model with squared loss it reduces exactly to an L2 penalty scaled by each feature's magnitude — an adaptive weight decay.

**Where the "0.5 for hidden layers" advice no longer holds.** That rate comes from 2012-era fully connected vision networks. Modern practice is much lower and much more targeted: convolutional layers use little or no dropout (spatial correlation means zeroing individual pixels removes little information — use DropBlock or SpatialDropout if you want it), and transformers use around 0.1 on attention weights and on the residual branches. Large-scale language model *pretraining* frequently uses dropout of 0 entirely, because with a corpus far larger than the parameter count the model is not in the overfitting regime and dropout just slows convergence; it comes back for fine-tuning on small datasets.

**A real interaction to know.** Dropout and batch normalisation conflict. Dropout changes the variance of activations between training and inference, while batch norm has stored running statistics estimated under the training-time (noisy) distribution, so the two disagree at test time and accuracy drops. The usual resolutions are to put dropout only after all batch-norm layers, or to drop one of the two — which is part of why modern architectures with layer norm and heavy data augmentation often use very little dropout.

**Follow-up:** *What is Monte Carlo dropout?* Leave dropout *on* at inference and run the same input through several times, then look at the spread of the predictions. Gal and Ghahramani showed this approximates Bayesian inference in a deep Gaussian process, so the variance across runs is a usable uncertainty estimate. It is one of the cheapest ways to get calibrated uncertainty out of a network that was not designed for it — at the cost of several forward passes per prediction.

> **Why the interviewer asks this.** The scaling factor is the detail that separates people who have read the paper from people who have used the API, and the "why 1/(1-p)" question has an exact answer.

> **Saying it out loud.** "During training you randomly zero out activations with probability p, and you scale the survivors up by one over one minus p so the expected value going into the next layer doesn't change. Then at inference you do nothing at all — that's inverted dropout, which is what every framework actually implements. The reason it helps is that no unit can depend on any specific other unit being there, so you don't get fragile co-adapted groups of features; another way to say it is that you're training an exponential ensemble of subnetworks that share weights. The old advice of 0.5 is out of date though — transformers use about 0.1, and large-scale pretraining often uses none at all."

---

## Bias & Variance

### Q22: Explain bias-variance tradeoff.

**Answer:**

**Bias:** Error from oversimplifying
- High bias = Underfitting
- Model too simple

**Variance:** Error from sensitivity to training data
- High variance = Overfitting
- Model too complex

**Tradeoff:** Can't minimize both
- Simple model: High bias, low variance
- Complex model: Low bias, high variance
- Goal: Find balance

**Diagnosis:**
- High bias: High train error, high test error (similar)
- High variance: Low train error, high test error (gap)

**Solutions:**
- High bias: More complex model, better features
- High variance: More data, regularization, simpler model

**The decomposition the whole thing rests on.**

For squared error, the expected test error at a point $x$ — expectation taken over random draws of the training set — decomposes exactly into three terms:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

Read the terms carefully, because the definitions matter. **Bias** is how far the *average* model — averaged over training sets you might have drawn — is from the truth; it is a property of the model class, not of any one fit. **Variance** is how much a particular fit bounces around that average when the training set changes. **Irreducible error** $\sigma^2$ is label noise; no model reduces it, and it is why 100% accuracy is usually not the target. The reason there is a tradeoff at all is that the two controllable terms respond in opposite directions to model flexibility: more capacity lets the average fit track the truth (bias down) but also lets each individual fit chase the noise in its particular sample (variance up).

**Making it concrete.** Fit polynomials to points from a sine curve plus noise. Degree 1: every training set gives roughly the same straight line, so variance is tiny, but no line is close to a sine, so bias is large. Degree 15: each training set gives a wildly different wiggly curve that passes through its own points exactly, so bias is near zero and variance is enormous. Degree 3-4 is where the sum is minimised.

**How the standard fixes map onto the terms.** More training data reduces variance and does nothing to bias — which is why adding data does not fix an underfitting model. Regularisation *adds* bias deliberately to buy a larger reduction in variance. Bagging (random forests) averages many high-variance, low-bias trees, and because averaging $B$ roughly-independent estimators divides variance by up to $B$ while leaving bias unchanged, it attacks variance specifically; feature subsampling exists to decorrelate the trees so that division is closer to real. Boosting works the other way: it fits shallow, high-bias stumps sequentially, each correcting the previous residuals, so it reduces bias — which is why boosted models can overfit with too many rounds while random forests largely do not.

**One honest caveat worth raising.** The classical U-shaped curve is not the whole story for modern overparameterised networks. Past the interpolation threshold — where the model has enough capacity to fit the training data exactly — test error can *decrease again* with further capacity, a phenomenon called double descent, observed in both model size and training time. So "more parameters means more variance" is a reliable rule for classical models and an unreliable one for large neural networks, where implicit regularisation from the optimiser changes the picture.

**Follow-up:** *You have 8% training error and 10% test error, and human performance is 1%. What do you do?* The gap is only 2 points, so variance is not the problem; the 7-point gap between human performance and training error is bias. Adding data or regularising will not help. Increase capacity, train longer, engineer better features, or check whether the input actually contains the information needed. The diagnostic move is always to compare training error against an irreducible-error estimate first, and only then look at the train-test gap.

> **Why the interviewer asks this.** It is really a debugging question in disguise — they want to know whether, shown two error numbers, you would reach for the right fix.

> **Saying it out loud.** "If you decompose squared test error, it splits into bias squared, variance, and irreducible noise. Bias is how far the average model over all possible training sets is from the truth — that's about the model class being too rigid. Variance is how much your particular fit moves around when you resample the data. Simple models are stable but wrong; complex ones fit their own sample beautifully and don't generalise. The reason I care is diagnosis: if training error is already high, that's bias, and more data won't help — I need capacity. If training error is low and test error is much higher, that's variance, and data or regularisation will help."

---

## Information Theory

### Q23: Explain entropy. What does it measure?

**Answer:**

**Entropy** measures uncertainty/randomness in a probability distribution.

**Formula:** H(X) = -Σ p(x) * log₂(p(x))

**Interpretation:**
- High entropy: High uncertainty (uniform distribution)
- Low entropy: Low uncertainty (concentrated distribution)
- Zero entropy: Deterministic (one outcome has prob=1)

**Example:**
- Fair coin: H = 1 bit (maximum uncertainty)
- Biased coin (90/10): H ≈ 0.47 bits (less uncertainty)
- Deterministic: H = 0 bits (no uncertainty)

**Use Cases:**
- Decision trees: Information gain
- Compression: Lower bound on code length
- Feature selection: High entropy = more informative

**Where the formula comes from.** Entropy is not an arbitrary functional; it is forced by requiring that a measure of information be additive over independent events. The information content, or *surprisal*, of an outcome with probability $p$ is $-\log p$: it is zero for a certain event, grows as the event gets rarer, and — critically — the surprisal of two independent events is $-\log(p_1 p_2) = -\log p_1 - \log p_2$, so it adds. Only the logarithm turns multiplication of probabilities into addition. Entropy is then just the *expected* surprisal, $H(X) = \mathbb{E}[-\log p(X)]$, which is what the formula says.

**The operational meaning is the one to give in an interview.** Shannon's source coding theorem says $H(X)$ in bits is the minimum average number of bits per symbol needed to encode messages from $X$ — no scheme does better, and schemes exist that get arbitrarily close. So entropy is not a metaphor for uncertainty; it is a compression bound. That is also the crispest way to explain the fair-coin answer: one bit per flip, and you cannot do better. For the 90/10 coin, $H = -0.9\log_2 0.9 - 0.1\log_2 0.1 = 0.137 + 0.332 = 0.469$ bits — meaning a long sequence of such flips can be compressed to under half a bit per flip, which run-length encoding does in practice.

**Bounds and units.** For $n$ outcomes, $0 \le H(X) \le \log n$, with the maximum at the uniform distribution and the minimum when one outcome is certain (using the convention $0\log 0 = 0$). The base of the logarithm is only a unit choice: base 2 gives bits, base $e$ gives nats. Machine learning code uses nats because $\log$ is the natural log; multiply by $\log_2 e \approx 1.443$ to convert.

**The connection to language models.** Perplexity, the standard LM metric, is exactly $2^{H}$ (or $e^{H}$ in nats) where $H$ is the cross-entropy per token. It is interpretable as an *effective vocabulary size*: a perplexity of 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 tokens. This is why perplexity 10 versus 20 is a much bigger improvement than the numbers suggest — it is one full bit of information per token.

**Follow-up:** *Why is entropy used for decision-tree splits?* A split is scored by information gain, $H(\text{parent}) - \sum_k \frac{n_k}{n} H(\text{child}_k)$: how much uncertainty about the label the split removes. It is always non-negative (conditioning cannot increase entropy on average), and it is zero exactly when the feature is independent of the label. Note the known failure mode: raw information gain favours high-cardinality features, since splitting on a unique ID gives pure children and maximal gain while generalising not at all — which is why C4.5 uses gain *ratio*, normalising by the entropy of the split itself.

> **Why the interviewer asks this.** They usually want the compression interpretation, because it is the one that shows you understand entropy as a quantity with units rather than a vague synonym for disorder.

> **Saying it out loud.** "Entropy is the expected surprise of a distribution — surprise being minus log p, which is the only form that makes independent events add up. The concrete meaning is compression: entropy in bits is the smallest average number of bits per symbol you can encode the source in. A fair coin is one bit and you can't do better; a ninety-ten coin is under half a bit because you can exploit the skew. It's maximised by the uniform distribution and zero when one outcome is certain. In language modelling it's the same quantity — perplexity is just two to the entropy, so you can read it as an effective vocabulary size."

---

### Q24: What is cross-entropy? Why is it used as a loss function?

**Answer:**

**Cross-Entropy:** H(P, Q) = -Σ p(x) * log(q(x))

**Why good loss function:**
1. **Penalizes confident wrong predictions**: -log(0.1) = 3.32 (large penalty)
2. **Encourages calibrated probabilities**: Mathematically well-founded
3. **Always ≥ entropy**: H(P, Q) ≥ H(P), equal when Q = P
4. **Smooth gradients**: Well-behaved optimization

**Use Cases:**
- Classification (most common loss)
- Language modeling
- Any probabilistic prediction

**Three derivations that arrive at the same loss.**

*From coding.* $H(P, Q) = -\sum_x p(x)\log q(x)$ is the average number of bits used if you build an optimal code for $Q$ but the data actually comes from $P$. It exceeds the true entropy $H(P)$ by exactly $D_{KL}(P \parallel Q)$ — the wasted bits from being wrong. Since $H(P)$ is fixed by the data, **minimising cross-entropy is identical to minimising KL divergence from the true distribution to the model.** That is the cleanest statement of what training a classifier does.

*From maximum likelihood.* The probability the model assigns to the whole dataset is $\prod_i q(y_i \mid x_i)$. Taking logs turns the product into a sum, and negating turns maximisation into minimisation: $-\sum_i \log q(y_i \mid x_i)$. With one-hot labels this is exactly the cross-entropy. So cross-entropy loss is not a heuristic scoring function — it *is* the negative log-likelihood, and every guarantee that maximum likelihood carries comes with it.

*From calibration.* Cross-entropy is a *strictly proper scoring rule*, meaning its expected value is uniquely minimised when the model reports its true beliefs. If the real probability of the positive class is 0.7 and you report 0.9 to look decisive, your expected loss goes up. Accuracy is not proper — reporting 0.99 and reporting 0.51 score identically — which is precisely why you train on cross-entropy and only report accuracy.

**Worked numbers on the "penalises confident wrong predictions" claim.** With the true label being class 1: predicting $q = 0.9$ costs $-\log 0.9 = 0.105$ nats; $q = 0.5$ costs $0.693$; $q = 0.1$ costs $2.303$; $q = 0.01$ costs $4.605$. The loss grows without bound as $q \to 0$, so a single confidently wrong prediction can dominate a whole batch. That unboundedness is a feature during training — it produces the large gradient that fixes the mistake — but it is also why numerically you must never compute $\log(\text{softmax}(z))$ separately (a rounded-to-zero probability gives $-\infty$) and instead use the fused `log_softmax` / `cross_entropy` op, which subtracts the max logit before exponentiating.

**Follow-up:** *What if the classes are heavily imbalanced?* Plain cross-entropy is dominated by the majority class simply because there are more of its terms. Class weighting multiplies each term by $w_c$ (often inversely proportional to class frequency) to rebalance. Focal loss goes further and multiplies by $(1 - q_y)^\gamma$, which shrinks the contribution of already-well-classified examples so training focuses on the hard ones — it was designed for dense object detection where the background class outnumbers objects by a thousand to one. Both keep the log-likelihood structure and just reweight it.

> **Why the interviewer asks this.** The good answer connects cross-entropy to KL divergence and to maximum likelihood, showing you see one object rather than three unrelated formulas.

> **Saying it out loud.** "Cross-entropy is the expected number of bits you pay for encoding data from P using a code built for Q, and it beats the true entropy by exactly the KL divergence — so minimising it is minimising KL from the data distribution to your model. It's also just the negative log-likelihood, so you get all the maximum-likelihood guarantees for free. And it's a proper scoring rule, which means you minimise it by reporting your honest probabilities, not by being overconfident. The practical property is that the loss goes to infinity as your predicted probability for the true class goes to zero, so confidently wrong predictions produce big corrective gradients."

---

### Q25: Explain KL divergence. Why is it asymmetric?

**Answer:**

**KL Divergence:** KL(P || Q) = Σ p(x) * log(p(x) / q(x))

**Why asymmetric:**
- KL(P || Q): "How surprised when we expect Q but get P?"
- KL(Q || P): "How surprised when we expect P but get Q?"
- Different interpretations → different values

**Properties:**
- KL(P || Q) ≥ 0
- KL(P || Q) = 0 if and only if P = Q
- Not a metric (doesn't satisfy triangle inequality)

**Use Cases:**
- RLHF: KL penalty
- VAEs: KL between posterior and prior
- Model comparison
- Regularization

**Where the asymmetry actually bites: mode-seeking versus mass-covering.**

The formula $D_{KL}(P \parallel Q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ is an expectation **under $P$**. That single fact drives everything. Only the regions where $p(x)$ is large contribute; regions where $p(x) = 0$ contribute nothing no matter what $Q$ does there. And if $q(x) \to 0$ where $p(x) > 0$, the log blows up and the divergence goes to infinity. In one sentence: $D_{KL}(P \parallel Q)$ punishes $Q$ for *failing to cover* $P$, and is indifferent to $Q$ putting mass where $P$ has none.

Now fit a single Gaussian $Q$ to a bimodal $P$ and the two directions give visibly different answers:

- **Forward KL, $D_{KL}(P \parallel Q)$ — mass-covering / zero-avoiding.** $Q$ must be nonzero everywhere $P$ is, or it takes an infinite penalty. So the fitted Gaussian stretches wide to span *both* modes, placing most of its mass in the empty valley between them. This is the direction minimised by maximum likelihood, which is why a maximum-likelihood language model will assign some probability to almost everything.
- **Reverse KL, $D_{KL}(Q \parallel P)$ — mode-seeking / zero-forcing.** Now the expectation is under $Q$, so $Q$ is punished for putting mass where $P$ is small, but not for ignoring a mode entirely. The fitted Gaussian collapses onto *one* mode and abandons the other. This is the direction in the VAE's ELBO and in variational inference generally, and it is the standard explanation for posterior-variance underestimation in VI and for mode collapse in some distillation setups.

**A numeric example.** Let $P = [0.5, 0.5, 0]$ and $Q = [0.9, 0.05, 0.05]$ over three outcomes. Forward: $0.5\log(0.5/0.9) + 0.5\log(0.5/0.05) + 0 = -0.294 + 1.151 = 0.857$ nats. Reverse: $0.9\log(0.9/0.5) + 0.05\log(0.05/0.5) + 0.05\log(0.05/0)= \infty$, because $Q$ puts mass on an outcome $P$ rules out. Same pair of distributions, radically different numbers — which is also a concrete demonstration that KL is not a distance and cannot be used where symmetry is assumed.

**Why it is non-negative.** By Jensen's inequality applied to the convex function $-\log$: $-\sum p \log(q/p) \ge -\log\sum p \cdot (q/p) = -\log \sum q = -\log 1 = 0$, with equality only when $q = p$ everywhere. This result is Gibbs' inequality, and it is what guarantees cross-entropy is always at least entropy.

**Follow-up:** *If KL is infinite whenever supports mismatch, how is it used in practice?* By construction, mostly. In RLHF the reference and policy are the same architecture over the same vocabulary with softmax outputs, so every token has nonzero probability under both and KL is finite. In VAEs both terms are Gaussians with full support and the KL has a closed form. Where support genuinely mismatches — comparing empirical samples from a generator against real data, the original GAN setting — KL and JS both become useless (constant or infinite, giving no gradient), which is exactly the motivation for the Wasserstein distance, which stays finite and differentiable for disjoint supports.

> **Why the interviewer asks this.** "It's asymmetric" is the memorised answer; the follow-up is always "so what?", and mode-seeking versus mass-covering is the answer that shows you know when the choice of direction changes your model's behaviour.

> **Saying it out loud.** "KL is an expectation under the first argument, and that's the whole reason it's asymmetric. Forward KL, P given Q, only looks at places where P has mass, and it goes to infinity if your model puts near-zero probability there — so it's mass-covering, it forces your model to spread out and cover everything. That's what maximum likelihood minimises. Reverse KL is the opposite: it punishes you for putting mass where P doesn't have any, but it's perfectly happy to ignore a whole mode. So it's mode-seeking, and that's what variational inference uses — which is exactly why VI tends to underestimate posterior variance."

---

### Q26: What is mutual information? How is it used in feature selection?

**Answer:**

**Mutual Information:** I(X; Y) = H(X) + H(Y) - H(X, Y)

**Interpretation:**
- I(X; Y) = 0: X and Y independent
- I(X; Y) > 0: X and Y dependent
- I(X; Y) = H(X): X completely determines Y

**Feature Selection:**
1. Compute I(X_i; Y) for each feature
2. Select features with high MI
3. High MI = feature is informative about target

**Why it works:**
- Captures non-linear relationships (unlike correlation)
- Removes irrelevant features (MI ≈ 0)
- Information-theoretic foundation

**Three equivalent readings of the same quantity.** $I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$. The first says "how much does knowing $Y$ reduce my uncertainty about $X$"; the second is the same statement with the roles swapped, which is why MI is symmetric; the third is the inclusion-exclusion form given above. A fourth reading is the most useful theoretically: $I(X; Y) = D_{KL}\big(P(X, Y) \parallel P(X)P(Y)\big)$ — mutual information is the KL divergence between the true joint and the joint you would have if they were independent. That immediately gives you $I \ge 0$ (KL is non-negative) and $I = 0$ exactly under independence.

**One correction to the note above.** The line "$I(X; Y) = H(X)$: X completely determines Y" has it backwards. $I(X;Y) = H(X)$ means $H(X \mid Y) = 0$, i.e. knowing $Y$ leaves no uncertainty about $X$ — so $Y$ determines $X$. In general $I(X;Y) \le \min(H(X), H(Y))$, and it equals $H(X)$ precisely when $X$ is a deterministic function of $Y$.

**Worked example against correlation.** Let $X$ be uniform on $[-1, 1]$ and $Y = X^2$. Pearson correlation is exactly 0 — the relationship is symmetric about zero, so the linear term cancels — yet $Y$ is a deterministic function of $X$, so $I(X; Y) = H(Y) > 0$, in fact maximal for that pair. A correlation-based filter discards this feature; a mutual-information filter keeps it. This is the single best example to have ready, because it makes "captures non-linear relationships" concrete instead of asserted.

**How it is actually estimated, and why that matters.** For discrete variables you count joint and marginal frequencies and plug in — but the plug-in estimator is *biased upward*, and the bias grows with the number of bins relative to the sample size. In the limit, a feature with as many distinct values as there are samples gets maximal apparent MI while carrying no signal at all, which is the same high-cardinality trap that afflicts decision-tree information gain. For continuous variables, binning is crude and bin-width-sensitive; the standard alternative is the Kraskov (KSG) $k$-nearest-neighbour estimator, which is what `sklearn.feature_selection.mutual_info_regression` uses. Normalised variants — dividing by $\min(H(X), H(Y))$ or by $\sqrt{H(X)H(Y)}$, or the adjusted mutual information — put values on a comparable $[0,1]$ scale and partially correct the cardinality bias.

**The real limitation of MI feature selection.** Ranking features by individual $I(X_i; Y)$ is a *univariate filter*: it evaluates each feature in isolation. That misses both directions of interaction. It keeps redundant features — ten copies of the same predictor all score high and all get selected. And it discards features that are only useful jointly: in an XOR relationship, $I(X_1; Y) = I(X_2; Y) = 0$ while $I(X_1, X_2; Y)$ is maximal, so a univariate filter throws away both of the only useful features. The fix is a criterion that accounts for the already-selected set, such as mMRM (maximum relevance, minimum redundancy), which selects to maximise $I(X_i; Y) - \frac{1}{|S|}\sum_{j \in S} I(X_i; X_j)$.

**Follow-up:** *When would you use MI over a model-based importance measure?* When you want something model-agnostic and cheap as a first-pass filter over thousands of candidate features, or when you need a dependence measure that is invariant to monotone reparameterisation of the features. When you can afford it, wrapper or embedded methods — permutation importance, L1 paths, tree-based importances — usually select better because they account for feature interactions and for the specific model you intend to deploy.

> **Why the interviewer asks this.** They want to see whether you know MI's failure modes — estimation bias and the univariate blind spot — not just that it beats correlation on nonlinear data.

> **Saying it out loud.** "Mutual information is how much knowing one variable reduces your uncertainty about the other. My favourite way to state it is that it's the KL divergence between the true joint distribution and the product of the marginals — so it's literally measuring how far the two are from independent. The reason people reach for it over correlation is that correlation only sees linear structure: if Y equals X squared with X centred at zero, correlation is exactly zero but mutual information is maximal. The catch is that estimating it is biased upward with lots of bins, and ranking features one at a time misses interactions — an XOR pair scores zero individually and everything jointly."

---

### Q27: Compare Gini impurity and entropy. When would you use each?

**Answer:**

**Gini:** Gini = 1 - Σ p_i² (faster, no log)
**Entropy:** H = -Σ p_i * log(p_i) (more theoretical)

**Comparison:**
- **Gini**: Faster computation, used in CART
- **Entropy**: More theoretical, used in ID3/C4.5
- **In practice**: Both work similarly, results usually very similar

**When to use:**
- **Gini**: When speed matters, CART algorithm
- **Entropy**: When you need information-theoretic interpretation

**Why they behave alike — they are the same curve to first order.** Both are concave functions of the class proportions, both are zero for a pure node and maximal at the uniform distribution, and both reward splits that produce purer children. For binary classification with positive-class fraction $p$: Gini is $2p(1-p)$, entropy is $-p\log_2 p - (1-p)\log_2(1-p)$. At $p = 0.5$ Gini peaks at 0.5 and entropy at 1.0; halve the entropy and the two curves agree to within about 0.03 everywhere on $[0,1]$. In fact the first two terms of the Taylor expansion of entropy about $p = 0.5$ give exactly the Gini form, which is the formal reason "results are usually very similar" — they are not similar by coincidence.

**A worked comparison.** A node with 80 positives and 20 negatives has Gini $= 1 - (0.8^2 + 0.2^2) = 0.32$ and entropy $= 0.722$ bits. Split it into a pure child of 60 positives and an impure child of 20 positives / 20 negatives. Weighted Gini afterwards is $0.6 \times 0 + 0.4 \times 0.5 = 0.20$, a reduction of 0.12. Weighted entropy is $0.6 \times 0 + 0.4 \times 1.0 = 0.40$, a reduction of 0.322 bits. Different units, same ranking — and that is the general pattern: published comparisons find the two criteria disagree on the chosen split in only a small minority of cases, and the resulting trees differ in accuracy by amounts well inside cross-validation noise.

**The one real difference.** Entropy's $-\log p$ term goes to infinity as $p \to 0$, so entropy is steeper near the pure ends of the range, which makes it slightly more willing to isolate a small pure subgroup. Gini, being quadratic, is flatter there and has a mild preference for splits that produce one large, mostly-pure child — it can be shown to favour balanced, high-purity partitions. In practice this is a second-order effect. The speed argument for Gini (no logarithm) was material in the 1980s and is largely irrelevant now that logs are a single instruction and split search is dominated by sorting.

**What matters far more.** Neither criterion addresses the actual failure mode of impurity-based splitting: bias toward high-cardinality features. A feature with many distinct values can carve out pure children by memorising, so it wins on both Gini and entropy while generalising not at all. C4.5's gain ratio normalises information gain by the entropy of the split itself; CART restricts to binary splits, which limits the damage. If you are choosing between Gini and entropy you are tuning a knob that barely moves; if you are ignoring cardinality bias, tree depth, minimum leaf size, and the number of trees, you are ignoring the knobs that do.

**Follow-up:** *What about regression trees?* Impurity is replaced by variance (equivalently, mean squared error within the node): score a split by the weighted reduction in variance of the target. Mean absolute error is the robust alternative when outliers matter, at the cost of being slower to compute since the optimal constant becomes the median rather than the mean. Gradient boosting generalises this further by fitting each tree to the gradient of an arbitrary differentiable loss, so the splitting criterion is derived from the loss rather than fixed.

> **Why the interviewer asks this.** Often to see whether you will claim a meaningful difference where there is not one — the strong answer is that they are nearly the same function and the real hyperparameters are elsewhere.

> **Saying it out loud.** "They're measuring the same thing and they're almost the same curve — if you Taylor-expand entropy around a half you get the Gini form back. Both are zero on a pure node and maximal when the classes are balanced, and in practice they pick the same split the overwhelming majority of the time. Entropy is a bit steeper near purity so it's slightly keener to peel off a small clean group; Gini is a bit cheaper because there's no logarithm, though that hardly matters now. Honestly, if someone's tuning Gini versus entropy they're spending effort on the wrong knob — depth, minimum leaf size and high-cardinality bias matter far more."

---

### Q28: What is Jensen-Shannon divergence? How does it differ from KL?

**Answer:**

**JS Divergence:** JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
Where M = 0.5 * (P + Q)

**Key Differences:**
- **Symmetric**: JS(P || Q) = JS(Q || P) (KL is asymmetric)
- **Bounded**: JS ∈ [0, 1] (KL can be infinite)
- **Metric**: Satisfies triangle inequality (KL doesn't)
- **Stable**: More stable when distributions very different

**Use Cases:**
- GANs: Measure distance between distributions
- Model comparison: When you need symmetric distance
- When KL is unstable

**One precision fix on the claims above, then the mechanism.** JS divergence itself is bounded by $\log 2$ (that is 1 when measured in bits, $0.693$ in nats) and is *not* a metric — it fails the triangle inequality. What is a true metric is its square root, $\sqrt{JS}$, sometimes called the Jensen-Shannon distance; that is the quantity `scipy.spatial.distance.jensenshannon` returns. So the row above should read: bounded by $\log 2$, and metric after taking the square root.

**Why the mixture fixes KL's blowup.** The failure mode of KL is that $D_{KL}(P \parallel Q) = \infty$ whenever $Q$ assigns zero probability somewhere $P$ does not. JS compares each distribution not to the other but to the midpoint $M = \tfrac{1}{2}(P + Q)$, and $M$ is nonzero wherever *either* is nonzero. So the ratio $p/m$ can never exceed 2, the log never exceeds $\log 2$, and the whole quantity stays finite even for completely disjoint supports. Take the extreme case: $P$ and $Q$ with no overlap at all. Then wherever $p > 0$ we have $m = p/2$, so $D_{KL}(P \parallel M) = \sum p \log 2 = \log 2$, and likewise for $Q$, giving $JS = \log 2$ — the maximum. Contrast $D_{KL}(P \parallel Q) = \infty$ for the same pair.

**A worked example.** With $P = [0.5, 0.5, 0]$ and $Q = [0, 0.5, 0.5]$: $M = [0.25, 0.5, 0.25]$. Then $D_{KL}(P \parallel M) = 0.5\log 2 + 0.5\log 1 = 0.347$ nats and by symmetry $D_{KL}(Q \parallel M) = 0.347$, so $JS = 0.347$ nats (0.5 bits) — a finite, interpretable number, where forward KL between $P$ and $Q$ is infinite in both directions.

**The GAN connection, and its punchline.** Goodfellow's original analysis showed that with an optimal discriminator, the generator's objective reduces to minimising $2\,JS(P_{\text{data}} \parallel P_{\text{gen}}) - \log 4$. That looks like good news until you notice the consequence of the boundedness argument above: early in training the generator's output distribution and the real data distribution lie on nearly disjoint low-dimensional manifolds, so JS sits pinned at its maximum $\log 2$ and its *gradient is essentially zero*. The generator gets no useful signal — this is the vanishing-gradient explanation for GAN training instability. Wasserstein GAN replaced JS with the earth-mover distance precisely because it varies smoothly with how far apart two disjoint distributions are, rather than saturating.

**Follow-up:** *When would you still prefer KL?* Whenever the asymmetry is the point. Maximum-likelihood training is forward KL and you want its mass-covering behaviour; variational inference is reverse KL and you want its mode-seeking behaviour; the RLHF penalty is one-directional because "how far has the policy moved from the reference" is inherently a directed question. Reach for JS when you need a symmetric, bounded comparison — measuring drift between two data distributions in monitoring, comparing two clusterings, or any case where you would otherwise be embarrassed to report a value of infinity.

> **Why the interviewer asks this.** Usually to get to the GAN story: knowing that JS saturates on disjoint supports, and that this is why WGAN exists, is what the question is really reaching for.

> **Saying it out loud.** "JS is the symmetrised version of KL — you build the midpoint mixture of the two distributions and average the KL from each one to that midpoint. The reason that helps is that the mixture has mass wherever either distribution does, so you can never divide by zero and the value is capped at log 2. KL between disjoint distributions is just infinity, which is useless. The catch is that capped behaviour: if two distributions barely overlap, JS is pinned at its maximum and the gradient is flat. That's the standard explanation for why GANs were so hard to train and why Wasserstein distance replaced it."

---

See `33_information_theory/interview_qa.md` for even more detailed answers!

---

## Discriminative vs Generative Models

### Q29: Explain the difference between discriminative and generative models.

**Answer:**

**Discriminative Models:**
- Learn P(Y|X) directly - the conditional probability of Y given X
- Focus on finding the decision boundary between classes
- Don't model the data distribution
- Examples: Logistic Regression, SVM, Neural Networks, Decision Trees

**Generative Models:**
- Learn P(X, Y) = P(X|Y) * P(Y) - the joint probability distribution
- Model how data is generated
- Can generate new data samples
- Examples: Naive Bayes, GMM, GANs, VAEs, Language Models

**Key Differences:**

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| **What they learn** | P(Y\|X) | P(X, Y) |
| **Can generate data** | No | Yes |
| **Data efficiency** | More efficient | Less efficient |
| **Complexity** | Simpler | More complex |
| **Use case** | Prediction | Generation + Prediction |

**When to use:**
- **Discriminative**: When you only need predictions, have limited data
- **Generative**: When you need to generate data, have missing data, want to understand distribution

**The distinction restated so it survives a follow-up.**

The real dividing line is *what the model spends its capacity on*. A discriminative model only ever has to answer "given this input, which label?", so it can ignore everything about the input that does not separate classes. A generative model has to account for the input itself — $P(X \mid Y)$ requires describing what the data looks like — which is strictly more information and usually a much harder estimation problem.

**The canonical worked contrast: naive Bayes versus logistic regression.** These two are a *generative-discriminative pair*: with the same features and a matched parametric form, they define the same hypothesis class of linear decision boundaries but fit it differently. Naive Bayes estimates $P(X \mid Y)$ and $P(Y)$ by counting, assuming features are conditionally independent given the label, then applies Bayes' rule. Logistic regression parameterises $P(Y \mid X)$ directly and maximises conditional likelihood. Ng and Jordan's result on this pair is the fact worth citing: naive Bayes converges to its (higher) asymptotic error much faster — in $O(\log d)$ samples versus $O(d)$ for logistic regression — so **the generative model wins on small data and the discriminative model wins once there is enough data**, with a crossover point. That is the mechanism behind the "data efficiency" row in the table, and it is more useful than the row itself: the generative model's assumptions act as a strong prior, which helps when data is scarce and hurts when the assumptions are wrong and data is plentiful.

**A caution about modern usage.** The table's "generative models are more complex" is a statement about classical estimation, and today's usage of the word has drifted. A modern autoregressive language model is generative in the strict sense — it models $P(X)$, factorised as $\prod_t P(x_t \mid x_{<t})$ — and it is spectacularly effective, not handicapped. What changed is that modelling $P(X)$ over text turned out to be a rich enough task that solving it produces general capability, and that conditioning ($P(\text{answer} \mid \text{question})$) can be obtained from the same model by prompting. So "generative is less data efficient for classification" is still true when comparing matched pairs on a fixed task, and simultaneously the most capable classifiers available today are generative models used zero-shot. Both are true; they are answers to different questions.

**Follow-up:** *Which family handles missing features better, and why?* Generative. Because it models the joint, a missing feature can be marginalised out — integrate $P(X, Y)$ over the unobserved variable and carry on with a properly normalised answer. A discriminative model has no distribution over $X$ to integrate against, so it must impute the missing value first and then hope the imputation error does not move the decision boundary. Anomaly detection is the same argument: a generative model gives you $P(X)$ directly, so "this input is unlike anything I was trained on" is a quantity it can report, whereas a discriminative model will confidently assign a class to pure noise.

> **Why the interviewer asks this.** It is a check on whether you can name a consequence of the distinction — data efficiency, missing values, anomaly detection — rather than only restating the two formulas.

> **Saying it out loud.** "A discriminative model learns the conditional — given the input, what's the label — so it only has to model what separates the classes. A generative model learns the joint, which means it has to describe what the data itself looks like. That's strictly more work, so with limited data the generative model's assumptions act as a prior and it does better, but once you have plenty of data the discriminative model wins because it isn't wasting capacity on the input distribution. Naive Bayes versus logistic regression is the classic pair. The nice side effect of going generative is that you can sample, marginalise over missing features, and score how unusual an input is."

---

### Q30: What are the assumptions of linear regression?

**Answer:**

**1. Linearity:** Relationship between X and Y is linear
- Check: Plot residuals vs predicted (should be random)
- Fix: Add polynomial features, transformations

**2. Independence of Errors:** Errors are independent
- Check: Durbin-Watson test, plot residuals vs time
- Fix: Use time series models for temporal data

**3. Homoscedasticity:** Constant variance of errors
- Check: Plot residuals vs predicted (look for funnel)
- Fix: Weighted least squares, transformations

**4. Normality of Errors:** Errors are normally distributed
- Check: Q-Q plot, Shapiro-Wilk test
- Fix: Transformations (less critical for large samples)

**5. No Multicollinearity:** Features not highly correlated
- Check: Correlation matrix, VIF
- Fix: Remove correlated features, regularization

**What happens if violated:**
- Poor predictions, wrong standard errors, unreliable tests
- See `34_discriminative_generative/model_assumptions_detailed.md` for detailed explanations

**Which assumptions matter for what — the distinction that makes this answer good.**

The five assumptions are not equally important, and they do not all protect the same thing. Split them:

*Needed for the coefficient estimates to be unbiased:* linearity and exogeneity (errors uncorrelated with the predictors). If these fail, $\hat{\beta}$ is estimating the wrong thing and no amount of data fixes it. This is the serious one — omitted variable bias lives here.

*Needed for the standard errors, p-values and confidence intervals to be correct:* homoscedasticity and independence of errors. If these fail, $\hat{\beta}$ is still unbiased, but your uncertainty estimates are wrong — typically too small, so you declare significance that is not there. The fix is often not to change the model but to change the variance estimator: heteroscedasticity-consistent (Huber-White, `HC3`) standard errors, or clustered standard errors for grouped data.

*Needed only for exact small-sample inference:* normality of errors. By the central limit theorem the sampling distribution of $\hat{\beta}$ is approximately normal for large $n$ regardless of the error distribution, so this assumption fades as $n$ grows. Note also what it is *not*: nothing requires $X$ or $y$ to be normally distributed — only the residuals, and only for inference.

*Not an assumption of the model at all:* no multicollinearity. Perfect collinearity makes $X^\top X$ singular so the solution is not unique; severe-but-imperfect collinearity leaves predictions and $R^2$ completely fine and only inflates the variance of individual coefficients. So if you care about prediction, collinearity is close to a non-issue; if you care about interpreting a specific coefficient, it is fatal. The variance inflation factor $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ is from regressing feature $j$ on the others, quantifies it — VIF of 10 means that coefficient's standard error is $\sqrt{10} \approx 3.2$ times larger than it would be with uncorrelated features.

**Why residual plots are the single most useful diagnostic.** Plot residuals against fitted values and you can see three assumptions at once: curvature indicates a linearity violation, a funnel shape indicates heteroscedasticity, and clusters or drift indicate dependence. It is worth remembering Anscombe's quartet here — four datasets with identical means, variances, correlations and regression lines but completely different structure, visible instantly in a plot and invisible in the summary statistics.

**Follow-up:** *Do these assumptions apply to ridge and lasso?* The estimation assumptions do not carry over cleanly, because both are *deliberately biased* estimators — the whole point of regularisation is to trade bias for variance, so unbiasedness was never on offer. Linearity still matters for the model to be right. Inference is genuinely harder: the standard $t$-tests do not apply to lasso coefficients, since the selection step is data-dependent, which is why post-selection inference is its own research area. In practice people using regularised regression for prediction check linearity and residual structure and skip the inference machinery entirely.

> **Why the interviewer asks this.** Listing five assumptions is memorisation; saying which ones break unbiasedness versus which only break the standard errors is understanding.

> **Saying it out loud.** "I'd group them by what they protect. Linearity and having errors uncorrelated with your predictors are what make the coefficients unbiased — if those fail, your estimates are wrong and more data won't save you. Homoscedasticity and independent errors don't bias the coefficients, they bias your standard errors, so you'd get significance you haven't earned; often the fix is just robust or clustered standard errors rather than a different model. Normality only matters for small-sample inference and washes out with the central limit theorem. And multicollinearity isn't really a model assumption at all — it doesn't hurt prediction, it just makes individual coefficients unstable and uninterpretable."

---

### Q31: What are the assumptions of logistic regression?

**Answer:**

**1. Binary Outcome:** Y must be binary (0 or 1)
- Fix: Use multinomial logistic for multi-class

**2. Linearity of Log-Odds:** Log-odds is linear in X
- Check: Box-Tidwell test
- Fix: Add polynomial features, interactions

**3. Independence:** Observations are independent
- Fix: Use mixed-effects models for correlated data

**4. No Multicollinearity:** Features not highly correlated
- Same as linear regression

**5. Large Sample Size:** Need sufficient data
- Rule of thumb: 10-20 observations per feature
- Fix: Collect more data, reduce features

**Differences from linear regression:**
- No normality assumption (errors are binary)
- No homoscedasticity (variance = p(1-p))
- Probability is sigmoid (non-linear), not linear

**Two things the list above understates.**

*Linearity of the log-odds is the assumption that actually does the work.* The model is $\log\frac{p}{1-p} = w^\top x + b$, so it is a linear model in log-odds space. On the *probability* scale the same relationship is S-shaped, which is why "logistic regression is a linear model" confuses people — it is linear in the link, not in the output. The practical consequence: a genuinely non-monotone relationship (risk high at both low and high values of a feature, as with blood pressure or dosage) cannot be captured, and no amount of data will reveal it. You have to add the quadratic term, spline the feature, or bin it.

*Complete separation is a failure mode worth naming.* If some hyperplane perfectly separates the classes, the likelihood is maximised by pushing the coefficients to infinity — every step further increases the fitted probabilities toward 0 and 1 and keeps reducing the loss. There is no finite MLE. Symptoms are enormous coefficients, standard errors in the thousands, and non-convergence warnings. This happens routinely with small samples, rare outcomes, or a feature that encodes the label. The standard remedies are exactly the regularisers you would use anyway — an L2 penalty makes the objective strictly convex and guarantees a finite solution, which is why `sklearn`'s `LogisticRegression` applies L2 by default — or Firth's penalised likelihood in a statistics setting.

**On the "large sample size" rule.** The real constraint is the number of events, not the number of rows. The classical rule of thumb is 10-20 *events* per predictor variable (EPV): with 1,000 rows but only 30 positives, you can support two or three predictors, not fifty. This is why rare-outcome problems are hard even with large datasets.

**Follow-up:** *How do you read a fitted coefficient?* $w_j$ is the change in log-odds per unit change in feature $j$, holding the others fixed, so $e^{w_j}$ is the *odds ratio* — a multiplicative effect on the odds. A coefficient of 0.7 means $e^{0.7} \approx 2$, so the odds double per unit. Two cautions worth voicing: odds ratios are not risk ratios and diverge badly when the base rate is high (doubling odds from 0.9 raises probability only from 0.47 to 0.64), and the effect on *probability* is not constant — it is largest near $p = 0.5$ and near zero out in the tails, because that is where the sigmoid is steep.

> **Why the interviewer asks this.** Mostly to see whether you know what "linear" refers to in logistic regression, and whether you have ever met separation in a real fit.

> **Saying it out loud.** "The load-bearing assumption is that the log-odds are linear in the features — not the probability, the log-odds. So the relationship on the probability scale is S-shaped, and anything genuinely non-monotone, like risk being high at both extremes of a measurement, just can't be represented without adding terms. You also need independent observations and you need enough events, not just enough rows — the rule of thumb is ten to twenty positive cases per predictor. And the one that bites in practice is complete separation: if a feature perfectly splits the classes the coefficients run off to infinity and the fit doesn't converge. A little L2 fixes it, which is why sklearn regularises by default."

---

### Q32: What are the assumptions of SVM?

**Answer:**

**1. Separable Data:** Data should be (nearly) separable
- Hard-margin: Must be linearly separable
- Soft-margin: Most points separable, some violations OK
- Fix: Use kernel, allow margin violations

**2. Feature Scaling:** Features must be scaled
- **Critical!** SVM is very sensitive to scales
- Fix: Always use StandardScaler or MinMaxScaler

**3. Appropriate Kernel:** Kernel should match data structure
- Linear: Linearly separable data
- RBF: Non-linear, local structure
- Polynomial: Polynomial relationships
- Fix: Try different kernels, use cross-validation

**4. Balanced Classes:** Can be sensitive to imbalance
- Fix: Use class weights, SMOTE, cost-sensitive learning

**What SVM doesn't assume:**
- Normal distributions
- Linear relationships (with kernels)
- Large sample size

**Why feature scaling is not a preference but a consequence of the objective.**

The linear SVM maximises the margin, and the margin is $2/\lVert w \rVert$ measured in Euclidean distance in feature space. So the geometry of the answer depends directly on the units of your features. If income is in dollars (range $10^4$–$10^5$) and age is in years (range 20–80), a one-unit change in age is geometrically negligible next to a one-unit change in income, so the maximum-margin hyperplane will be almost entirely determined by income regardless of which feature is actually informative. The RBF kernel makes it worse: $\exp(-\gamma \lVert x - y\rVert^2)$ is dominated by whichever feature has the largest numeric spread, so the other features effectively vanish from the similarity computation. This is why the answer above marks scaling "critical" — it is not a tuning nicety, it is required for the objective to mean what you intend.

**The regularisation parameter C, which the list omits.** Soft-margin SVM minimises $\frac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i$, where $\xi_i$ are slack variables measuring how far each point violates its margin. $C$ is the exchange rate between a wide margin and few violations. Small $C$ means violations are cheap, so the margin grows, more points end up as support vectors, and the model is smoother and higher-bias. Large $C$ means violations are expensive, so the boundary contorts to classify training points correctly — low bias, high variance, and in the limit $C \to \infty$ you recover the hard-margin SVM, which does not exist for non-separable data. $C$ and $\gamma$ interact strongly, which is why they are always tuned jointly on a 2-D grid rather than one at a time.

**What "support vector" means and why it matters.** Only the points on or inside the margin have nonzero dual coefficients; the solution depends on those alone. Delete every other training point and refit and you get the identical boundary. That is a genuinely unusual property — it makes SVM robust to far-away outliers of the *correct* class, but it also means a single mislabelled point sitting inside the opposite class's region becomes a support vector and can move the boundary substantially, especially with large $C$.

**Where the "assumptions" framing is loose.** SVM is not a probabilistic model, so it has no likelihood and therefore no distributional assumptions in the sense that linear regression has them. What it has are *requirements for the geometry to be meaningful* (scaling), *a modelling choice that must match the data* (the kernel), and *a known sensitivity* (class imbalance, because the margin term does not know that one class matters more; `class_weight='balanced'` rescales $C$ per class to fix it). It is worth saying this explicitly in an interview, because it shows you know why the question is phrased differently for SVM than for regression.

**Follow-up:** *SVM does not output probabilities — what do you do if you need them?* The decision function is a signed distance from the hyperplane, not a probability. Platt scaling fits a one-dimensional logistic regression to those distances on held-out data, which is what `probability=True` triggers in scikit-learn — note it runs an internal cross-validation, so it is slow and can produce probabilities that disagree with `predict`. Isotonic regression is the non-parametric alternative and fits better when you have enough calibration data. If probabilities are central to your problem, though, logistic regression or a gradient-boosted model is usually the better starting point.

> **Why the interviewer asks this.** Scaling is the answer they are fishing for; the good candidate explains *why* the objective demands it rather than just asserting it.

> **Saying it out loud.** "The main one is feature scaling, and it's not optional — the SVM maximises a margin measured as Euclidean distance, so whichever feature has the biggest numeric range dominates the geometry. With an RBF kernel that's even more true, because the distance in the exponent is driven by the largest-scale feature. Beyond that it's really about choices rather than assumptions: pick a kernel that matches the structure, tune C and gamma together because they interact, and watch out for class imbalance since the margin doesn't know one class matters more. And it's not probabilistic, so if you need calibrated probabilities you have to bolt on Platt scaling afterwards."

---

### Q33: Explain Bayes' theorem in detail.

**Answer:**

**Mathematical Formulation:**
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**Components:**
- **Prior P(A)**: Belief about A before seeing evidence
- **Likelihood P(B|A)**: Probability of evidence B given A
- **Evidence P(B)**: Total probability of B
- **Posterior P(A|B)**: Updated belief about A after seeing B

**Why it matters:**
- Updates beliefs with evidence
- Foundation of Bayesian statistics
- Used in Naive Bayes, spam detection, medical diagnosis

**Example:**
- Disease prevalence: 1% (prior)
- Test accuracy: 95% (likelihood)
- Positive test → Only 16% chance of disease!
- Why? False positives from large healthy population

**Use Cases:**
- Naive Bayes classifier
- Spam detection
- Medical diagnosis
- Recommendation systems

See `34_discriminative_generative/bayes_theorem_detailed.md` for comprehensive explanation!

**Working the medical example all the way through, because the number is the point.**

Let $D$ be "has the disease" and $+$ be "tests positive". Given prevalence $P(D) = 0.01$, sensitivity $P(+ \mid D) = 0.95$, and specificity $P(- \mid \neg D) = 0.95$ (so the false-positive rate is $P(+ \mid \neg D) = 0.05$):

$$P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+ \mid D)P(D) + P(+ \mid \neg D)P(\neg D)} = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} = \frac{0.0095}{0.0590} = 0.161$$

The intuition is easier in counts. Test 10,000 people: 100 have the disease and 95 of them test positive; 9,900 do not and 5% of them — 495 people — test positive anyway. So 590 positive tests, of which 95 are real: 16%. The false positives outnumber the true positives five to one purely because the healthy group is 99 times larger. This is base rate neglect, and the counting version is the one to use in an interview because it makes the result feel obvious rather than paradoxical.

**The odds form, which is faster and more illuminating.** Divide the posterior for $D$ by the posterior for $\neg D$ and the evidence term cancels:

$$\underbrace{\frac{P(D \mid +)}{P(\neg D \mid +)}}_{\text{posterior odds}} = \underbrace{\frac{P(D)}{P(\neg D)}}_{\text{prior odds}} \times \underbrace{\frac{P(+ \mid D)}{P(+ \mid \neg D)}}_{\text{likelihood ratio}}$$

Here: prior odds $1{:}99$, likelihood ratio $0.95/0.05 = 19$, so posterior odds $19{:}99$, which is $19/118 = 0.161$. Same answer with no denominator to compute. This form also makes sequential updating trivial — a second independent positive test multiplies by 19 again, giving odds $361{:}99$, or 78% — and it shows exactly what evidence *is*: a multiplier on the odds, with strength given by the likelihood ratio.

**Why the denominator is where the work is.** $P(B)$ is computed by the law of total probability, $\sum_A P(B \mid A)P(A)$, summing over every hypothesis. For a handful of discrete hypotheses this is arithmetic. For continuous parameters it becomes an integral over the whole parameter space that is usually intractable, and that single fact is the reason Bayesian computation exists as a field: MCMC samples from the posterior without ever evaluating the denominator (Metropolis-Hastings only needs ratios, in which it cancels), and variational inference sidesteps it by optimising a bound instead.

**How naive Bayes uses this.** Classify by $\arg\max_y P(y)\prod_j P(x_j \mid y)$ — the denominator is dropped because it is the same for every class and cannot change the argmax. The "naive" part is the product: assuming features are conditionally independent given the class. That assumption is essentially always false in text (words are correlated), yet the classifier works well, because the argmax only needs the class *ranking* to be right, not the probabilities. The probabilities themselves come out wildly overconfident — typically saturated at 0 or 1 — which is why naive Bayes should not be trusted for calibrated output.

**Follow-up:** *What happens to the posterior as evidence accumulates?* Under mild conditions the likelihood dominates and the posterior concentrates on the truth regardless of the prior — this is the Bernstein-von Mises theorem, and it is why prior choice matters most when data is scarce. The important exception is a prior that assigns exactly zero probability to some hypothesis: multiplying by zero stays zero forever, so no amount of evidence can recover it. Cromwell's rule — never assign a prior of exactly 0 or 1 to anything you are not logically certain of — is the practical statement, and it is the same reason Laplace smoothing (Q40) exists in naive Bayes.

> **Why the interviewer asks this.** The disease example is a base-rate test, and they want to see whether you can produce the 16% and explain it in counts rather than just recite the formula.

> **Saying it out loud.** "Bayes' theorem is just how you update a belief when evidence arrives — posterior is proportional to likelihood times prior. The example I always use is the medical test: one percent prevalence, ninety-five percent accurate, and a positive result only means about a sixteen percent chance you're sick. The easiest way to see why is in counts. Out of ten thousand people, a hundred are sick and ninety-five of them test positive, but of the ninety-nine hundred healthy people, five percent — that's four hundred and ninety-five — also test positive. So the false positives swamp the true ones, purely because the healthy group is so much bigger. The odds form is even quicker: prior odds times the likelihood ratio."

---

See `34_discriminative_generative/model_assumptions_detailed.md` for detailed assumption explanations!

---

## Kernel Functions

### Q34: What is a kernel function? Explain the kernel trick.

**Answer:**

**Kernel Function:**
Kernel K(x, y) computes dot product in high-dimensional space without explicitly computing the transformation.

**Formula:** K(x, y) = φ(x) · φ(y)

**Kernel Trick:**
- **Problem**: Transform data to high dimensions (expensive)
- **Solution**: Use kernel to compute dot product directly (cheap)
- **Benefit**: Get high-dimensional features without computing them

**Example:**
Polynomial kernel (degree=2):
- Without trick: Compute [x₁, x₂, x₁², x₂², x₁x₂, ...] (8 dimensions)
- With trick: Just compute (x · y)² (same result, much faster!)

**Why it works:**
Algorithms like SVM only need dot products, not features themselves.

**Doing the polynomial expansion explicitly, since that is what makes the trick concrete.**

Take two-dimensional inputs $x = (x_1, x_2)$ and $y = (y_1, y_2)$ and expand the degree-2 kernel:

$$(x \cdot y)^2 = (x_1y_1 + x_2y_2)^2 = x_1^2y_1^2 + 2x_1x_2y_1y_2 + x_2^2y_2^2 = \phi(x)\cdot\phi(y)$$

with $\phi(x) = (x_1^2,\ \sqrt{2}\,x_1x_2,\ x_2^2)$. So a 3-dimensional feature map falls out, and the $\sqrt{2}$ on the cross term is not decorative — it is exactly what makes the dot product match. The inhomogeneous version $(x\cdot y + 1)^2$ gives six features, adding $\sqrt{2}x_1, \sqrt{2}x_2, 1$, which is where a count of "8" for the degree-2 map would come from only in three input dimensions. (In general, degree $d$ over $n$ input dimensions gives $\binom{n+d}{d}$ features, so a degree-4 kernel on 100 features corresponds to about 4.6 million explicit dimensions — computed by one dot product and one exponentiation.)

The RBF kernel makes the point unanswerably: $\exp(-\gamma\lVert x-y\rVert^2)$ corresponds to an *infinite-dimensional* feature map (expand the exponential as a power series and every polynomial degree appears). You could never write $\phi(x)$ down, yet the kernel evaluates in a few floating-point operations.

**Why "algorithms only need dot products" is not a coincidence.** The SVM's dual formulation is
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j\,(x_i\cdot x_j)$$
and prediction is $f(x) = \sum_i \alpha_i y_i (x_i \cdot x) + b$. The inputs appear *only* inside dot products, in both training and prediction. So replacing each $x_i \cdot x_j$ with $K(x_i, x_j)$ gives you the same algorithm operating in the feature space, without $\phi$ ever being formed. The Representer Theorem is the general statement: for a broad class of regularised problems, the optimal solution lies in the span of the training points, so it can always be written using kernel evaluations alone. The same substitution turns PCA into kernel PCA, ridge regression into kernel ridge regression, and so on.

**What makes a function a valid kernel.** Mercer's condition: $K$ must be symmetric and positive semi-definite, meaning the Gram matrix $K_{ij} = K(x_i, x_j)$ has no negative eigenvalues for any finite set of points. That is exactly the condition under which some $\phi$ exists with $K(x,y) = \phi(x)\cdot\phi(y)$, and it is also what keeps the SVM's dual problem convex. Kernels compose: sums, positive scalings, and products of valid kernels are valid, which is how structured kernels for text, graphs and time series get built.

**The cost you are paying.** The trick converts an $O(n \cdot D)$ problem in explicit feature dimension $D$ into an $O(n^2)$ problem in the number of samples, because the Gram matrix is $n \times n$. That is a spectacular win when $D \gg n$ and a disaster when $n$ is large — kernel SVMs are impractical past roughly $10^5$ samples, since the matrix alone would need tens of gigabytes. Random Fourier features invert the trick for exactly this reason: they approximate the RBF kernel with an explicit low-dimensional random map, trading a little accuracy for linear-time training.

**Follow-up:** *Is the kernel trick relevant in deep learning?* Directly, rarely — neural networks learn their feature map rather than fixing it, which is the whole advantage. Theoretically, very much so: an infinitely wide network trained by gradient descent behaves as kernel regression under the Neural Tangent Kernel, which is one of the main tools for analysing why overparameterised networks generalise. And attention itself is often described as a kernel: $\text{softmax}(q\cdot k)$ is a similarity function determining how much each value contributes, and linear-attention methods work precisely by replacing that softmax kernel with a factorisable feature map to escape the quadratic cost.

> **Why the interviewer asks this.** They want to see the explicit expansion, or at least that you know one exists — being able to write $\phi$ for the degree-2 case is the difference between having understood the trick and having heard of it.

> **Saying it out loud.** "A kernel is a function that gives you the dot product of two points in some high-dimensional feature space without ever building that space. The concrete version: take x dot y squared in two dimensions and expand it — you get x-one squared, root-two x-one x-two, and x-two squared, so it's exactly the dot product of a three-dimensional feature map. The RBF kernel corresponds to an infinite-dimensional map, and you still evaluate it with one subtraction and one exponential. It works because algorithms like SVMs only ever touch the data through dot products, in training and at prediction time, so you just swap in the kernel. The price is that you're now working with an n-by-n Gram matrix, so it doesn't scale past a hundred thousand samples or so."

---

### Q35: Explain different types of kernels. When would you use each?

**Answer:**

**Linear Kernel:** K(x, y) = x · y
- **Use when**: Linearly separable data, high-dimensional data
- **Example**: Text classification with TF-IDF

**Polynomial Kernel:** K(x, y) = (γx·y + r)^d
- **Use when**: Polynomial relationships, moderate non-linearity
- **Example**: Circular boundaries (degree=2)

**RBF Kernel:** K(x, y) = exp(-γ||x-y||²)
- **Use when**: Non-linear problems (default choice)
- **Example**: Complex boundaries, most common kernel

**Sigmoid Kernel:** K(x, y) = tanh(γx·y + r)
- **Use when**: Rarely (RBF is usually better)

**Selection:**
1. Try linear first
2. If fails, use RBF
3. If RBF overfits, try polynomial

**What each kernel is really assuming.** A kernel is a similarity function, and choosing one is choosing a prior over what "similar" means — so the right framing is not "which is most powerful" but "which notion of similarity matches my data".

The **linear kernel** says similarity is alignment of directions: two documents are similar if they share weighted vocabulary. It is the right choice when the data is already high-dimensional enough to be separable as it is, which is why it is standard for TF-IDF text — with 50,000 features and 5,000 documents, the data is almost certainly linearly separable and any extra flexibility only buys overfitting. It is also the only kernel whose weights you can inspect directly, since $w = \sum_i \alpha_i y_i x_i$ lives in the original feature space; every other kernel gives you a solution expressed in terms of support vectors instead.

The **polynomial kernel** says similarity comes from *conjunctions* of features up to degree $d$ — it explicitly constructs interaction terms. That is a good match when you believe the label depends on products of features (a pixel pair, a feature interaction in tabular data). Its practical weakness is numerical: $(\gamma x \cdot y + r)^d$ either explodes or vanishes as $d$ grows, so it is finicky to tune and rarely worth going past degree 3.

The **RBF (Gaussian) kernel** says similarity decays with distance, full stop. This is a *local* prior: nearby points should share labels and distant points are uninformative, with $\gamma$ setting the scale of "nearby". Because it can approximate any continuous decision boundary given enough support vectors, it is the sensible default when you have no structural belief about the data. Note it is stationary — $K$ depends only on $x - y$, not on where in the space you are — so it assumes the same length scale applies everywhere.

The **sigmoid kernel** $\tanh(\gamma x\cdot y + r)$ is worth knowing mainly for the reason to avoid it: it is *not positive semi-definite* for most parameter settings, so it violates Mercer's condition, the dual problem stops being convex, and the solver may return something that is not a global optimum. Its historical appeal was a loose analogy to a two-layer neural network. RBF dominates it on essentially every benchmark.

**A decision rule with actual numbers.** Compare the number of features $d$ to the number of samples $n$. If $d \gtrsim n$ (text, genomics, any wide sparse problem), use linear — the data is likely already separable and a nonlinear kernel adds variance for nothing. If $n \gg d$ (a few dozen dense features, tens of thousands of rows), the data probably needs curvature, so use RBF. If $n$ is very large, above $10^5$, drop kernels altogether: the $O(n^2)$ Gram matrix makes them infeasible, so use a linear SVM with a solver like LIBLINEAR, random Fourier features, or a gradient-boosted tree ensemble.

**Follow-up:** *Can you build a kernel for non-vector data?* Yes, and this is one of the kernel method's real advantages. Because a kernel only needs to be a symmetric positive semi-definite similarity, you can define one directly on strings (the string kernel counts shared subsequences), on graphs (the Weisfeiler-Lehman kernel compares iteratively refined neighbourhood labels), or on sets and trees. That lets you run SVM, PCA or ridge regression on objects that have no natural vector representation at all — historically a large part of why kernel methods mattered in bioinformatics and NLP before deep learning learned representations instead.

> **Why the interviewer asks this.** To hear whether you pick kernels by reasoning about the data's shape, or by working down a fixed list.

> **Saying it out loud.** "I think of a kernel as a statement about what 'similar' means, so picking one is picking a prior. Linear says similarity is directional alignment, and it's right when your data's already high-dimensional — text with TF-IDF is nearly always linearly separable, so anything fancier just overfits. Polynomial builds explicit feature interactions, which helps if you think the label depends on products of features, but it's numerically awkward past degree three. RBF says similarity decays with distance, which is a local prior and a sensible default when you've got no strong structural belief. Sigmoid isn't even positive semi-definite for most settings, so the optimisation isn't guaranteed convex — I'd skip it."

---

### Q36: Explain RBF kernel in detail. How does gamma affect it?

**Answer:**

**RBF Kernel:** K(x, y) = exp(-γ||x-y||²)

**What it does:**
Measures similarity based on distance. Close points → high similarity, far points → low similarity.

**Gamma Effect:**
- **Low γ (0.001)**: Wide kernel → Simpler boundary, risk underfitting
- **Medium γ (0.1-1.0)**: Balanced → Good starting point
- **High γ (10.0)**: Narrow kernel → Complex boundary, risk overfitting

**Visual:**
Each point creates a "bump". Low gamma = wide bumps (simple), high gamma = narrow bumps (complex).

**Tuning:**
Start with γ = 1/(n_features * variance), then grid search.

**Gamma as an inverse length scale, with numbers.**

Write the kernel as $\exp(-\lVert x - y\rVert^2 / 2\sigma^2)$ and you can read off $\gamma = 1/(2\sigma^2)$: gamma is an inverse squared length scale, so $\sigma = 1/\sqrt{2\gamma}$ is the distance over which similarity meaningfully decays. That conversion turns an abstract hyperparameter into something you can sanity-check against your data.

Take two points at Euclidean distance 1 after standardisation:

| $\gamma$ | $K$ at distance 1 | $K$ at distance 2 | $K$ at distance 3 | effective $\sigma$ |
|---|---|---|---|---|
| 0.01 | 0.990 | 0.961 | 0.914 | 7.07 |
| 0.1 | 0.905 | 0.670 | 0.407 | 2.24 |
| 1.0 | 0.368 | 0.018 | 0.00012 | 0.71 |
| 10.0 | 0.000045 | ~0 | ~0 | 0.22 |

At $\gamma = 0.01$ every point in a standardised dataset is similar to every other point — the kernel matrix is nearly all ones, the model has almost no ability to distinguish anything, and it underfits toward a constant. At $\gamma = 10$ a point is similar only to itself: the Gram matrix approaches the identity, every training point becomes its own support vector, training accuracy hits 100%, and test accuracy collapses. The useful range sits where $K$ at typical inter-point distances is neither near 0 nor near 1 — which is exactly what the default heuristic targets.

**The scikit-learn defaults, decoded.** `gamma='scale'` sets $\gamma = 1/(d \cdot \text{Var}(X))$ where $d$ is the number of features. The reasoning: for standardised data the expected squared distance between two random points is about $2d\,\text{Var}(X)$, so this choice puts the exponent at roughly $-2$ for a typical pair, landing $K$ near $0.14$ — squarely in the responsive region and, crucially, invariant to how many features you have and what scale they are on. The older `gamma='auto'` used $1/d$, which ignores variance and is why it was replaced. Either way, grid search on a logarithmic scale around that value, jointly with $C$.

**How gamma and C interact, which is what people get wrong.** They are not independent knobs, because both control effective complexity. Large $\gamma$ with large $C$ is the classic overfitting corner: narrow bumps *and* an insistence that every training point be classified correctly, producing islands of one class around individual points. Small $\gamma$ with small $C$ underfits from both directions. There are also compensating diagonals — moderate $\gamma$ with large $C$ can behave much like large $\gamma$ with moderate $C$ — which is precisely why a joint 2-D grid finds good regions that two sequential 1-D searches miss.

**Follow-up:** *Why is the RBF kernel called universal?* Because the function class it induces is dense in the space of continuous functions on a compact domain: with enough support vectors and a suitable $\gamma$, an RBF SVM can approximate any continuous decision boundary to arbitrary accuracy. That is the theoretical reason it is the default choice. It also explains why regularisation is not optional here — a model class that can represent anything will represent your noise if you let it, so $C$ and $\gamma$ are doing all the work of controlling capacity.

> **Why the interviewer asks this.** Gamma is the hyperparameter people tune blindly; being able to convert it into a distance scale and predict what happens at the extremes shows you understand what you are tuning.

> **Saying it out loud.** "The RBF kernel is similarity that decays with distance, and gamma is an inverse length scale — it sets how far away two points can be and still count as similar. If gamma's tiny, everything looks similar to everything, the kernel matrix is basically all ones and you underfit to a constant. If gamma's huge, every point is only similar to itself, the matrix goes to the identity, and you memorise the training set. So I want gamma where the similarity between typical pairs of points is somewhere in the middle, which is exactly what sklearn's 'scale' default is doing — one over features times variance. And I'd always tune it jointly with C on a log grid, because both control complexity and they trade off against each other."

---

### Q37: How do you choose the right kernel?

**Answer:**

**Decision Process:**
1. **Try linear first**: Fast, interpretable
2. **If fails, try RBF**: Default for non-linear
3. **If RBF overfits, try polynomial**: Less flexible
4. **Never use sigmoid**: RBF is better

**Parameter Tuning:**
- **RBF**: Tune gamma [0.001, 0.01, 0.1, 1.0, 10.0] and C [0.1, 1, 10, 100, 1000]
- **Polynomial**: Start with degree=2, gamma=1.0
- **Use cross-validation**: Compare different kernels and parameters

**Key Points:**
- Always scale features before SVM
- Linear often works for high-dimensional data
- RBF is most common for non-linear problems

**The rule behind the sequence.** "Try linear first" is not just about speed — it is a statement about the bias-variance tradeoff and about what your data's shape implies. A linear SVM in $d$ dimensions can shatter at most $d+1$ points, so its capacity is bounded by the feature count; an RBF SVM has effectively unbounded capacity. When $d$ is comparable to or larger than $n$, the linear model already has enough capacity to separate the data, and adding more can only add variance. That is why the linear-versus-RBF decision is well predicted by the ratio $d/n$ rather than by trial and error.

**A concrete protocol.** Standardise the features (fitting the scaler on the training fold only, inside the cross-validation loop, or you leak test statistics into training). Fit a linear SVM across $C \in \{0.01, 0.1, 1, 10, 100\}$ and record cross-validated performance. Then fit RBF over the outer product of that $C$ grid with $\gamma \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ — 25 combinations, on a log scale because these parameters act multiplicatively. Compare the best of each. If RBF beats linear by less than the standard error across folds, take linear: it is faster, it gives interpretable weights, and it has fewer ways to fail on new data. Then refine with a finer grid around the winning region, or use random search, which finds good regions with fewer evaluations when only one of the two parameters actually matters.

**Diagnosing rather than guessing.** The learning curve tells you which direction to move. If training and validation error are both high and close together, you are underfitting — a more expressive kernel or a larger $C$ will help. If training error is near zero and validation error is much higher, you are overfitting, and the answer is a simpler kernel, smaller $C$, smaller $\gamma$, or more data. Reading this off a curve is faster and more reliable than expanding the grid.

**The honest caveat.** Before running any of this, ask whether an SVM is the right model. On tabular data of moderate size, gradient-boosted trees (XGBoost, LightGBM, CatBoost) typically beat a tuned kernel SVM while needing far less preprocessing — no scaling required, categorical features handled natively, missing values handled natively, and training that is near-linear in $n$ rather than quadratic. Kernel SVMs remain genuinely competitive in the small-$n$, high-$d$ regime (a few hundred samples, thousands of features, as in some biological assays), where trees struggle and the margin-maximisation prior is a real advantage. Saying this unprompted is usually a stronger signal than any amount of grid-search detail.

**Follow-up:** *What if you cannot tell which kernel matches the structure?* Use multiple kernel learning, which learns a convex combination $\sum_m \beta_m K_m$ of candidate kernels jointly with the classifier, letting the data weight them. In practice it is rarely worth the complexity for a small kernel set — cross-validating over the individual kernels usually gets you the same answer more cheaply — but it is the principled response, and it is genuinely useful when your kernels encode different *data sources* (one for text, one for images, one for a graph) that must be combined.

> **Why the interviewer asks this.** They want a decision procedure grounded in something — the $d/n$ ratio, learning curves — rather than "try them all and see".

> **Saying it out loud.** "I start with the shape of the data. If features are comparable to or more numerous than samples — text, genomics, anything wide and sparse — linear is almost always right, because the data's likely separable already and anything more flexible just adds variance. If I've got a few dense features and lots of rows, I'd reach for RBF and grid search C and gamma together on a log scale, because they interact. Then I'd look at the learning curve rather than guessing: both errors high and close means underfitting, big train-test gap means overfitting. And honestly, if it's tabular data of any real size, I'd check gradient-boosted trees first — they usually beat a tuned kernel SVM and need far less preprocessing."

---

See `35_kernel_functions/interview_qa.md` for even more detailed answers!

---

## NLP Basics

### Q38: Explain TF-IDF. How does it work?

**Answer:**

**TF-IDF** measures how important a word is to a document in a collection.

**Components:**

**Term Frequency (TF):**
- TF(t, d) = count(t, d) / |d|
- How often word appears in document
- Higher TF = more important to document

**Inverse Document Frequency (IDF):**
- IDF(t, D) = log(N / |{d : t ∈ d}|)
- How rare word is across documents
- Common words (appear in many docs) → low IDF
- Rare words (appear in few docs) → high IDF

**TF-IDF:**
- TF-IDF(t, d) = TF(t, d) × IDF(t, D)
- High TF-IDF: Word appears often in this document (high TF) but rarely in others (high IDF)
- Identifies characteristic words for each document

**Example:**
- "algorithm" in Python tutorial: High TF (appears often) + High IDF (rare word) → High TF-IDF
- "the" in any document: High TF but Low IDF (common word) → Lower TF-IDF

**Use Cases:**
- Text classification (feature extraction)
- Search engines (ranking)
- Information retrieval

**Where the log in IDF comes from, and a worked calculation.**

The logarithm is not cosmetic. Raw inverse document frequency $N/df_t$ is a ratio that grows without bound: a word appearing in 1 document out of a million would get 1,000,000 times the weight of a word appearing in every document, which is far too aggressive — it would let a single typo dominate a document's representation. Taking the log compresses this into a difference of magnitudes, so a word appearing in 1% of documents gets roughly twice the weight of one appearing in 10%, not ten times. There is also an information-theoretic reading: $\log(N/df_t) = -\log P(t \in d)$ is exactly the *surprisal* of seeing the term in a randomly chosen document, so IDF is measuring how informative the term's presence is, in the sense of Q23.

Compute it on a small corpus of $N = 1000$ documents, for a document of 100 words:

| term | count in doc | TF | $df$ | $\log(N/df)$ | TF-IDF |
|---|---|---|---|---|---|
| "the" | 7 | 0.07 | 1000 | $\log(1) = 0$ | 0.000 |
| "model" | 5 | 0.05 | 100 | $\log(10) = 2.303$ | 0.115 |
| "quantisation" | 3 | 0.03 | 10 | $\log(100) = 4.605$ | 0.138 |

Note what happened to "the": appearing in every document gives $\log(1) = 0$, which zeroes it out entirely — TF-IDF performs stopword removal automatically, without a stopword list. Note also that "quantisation" outscores "model" despite appearing less often, because rarity beat frequency. That interaction is the entire point of the metric.

**The smoothing you will see in real implementations.** A term in the vocabulary but absent from the corpus would give division by zero, so scikit-learn uses $\text{idf}(t) = \log\frac{1+N}{1+df_t} + 1$. The trailing $+1$ matters: it stops a term appearing in every document from being zeroed out completely, on the grounds that it may still be worth a little. There are also variants for TF — sublinear scaling $1 + \log(\text{count})$ is common, on the reasoning that a word appearing 20 times is not 20 times more relevant than one appearing once. BM25, the standard ranking function in search engines, extends the same idea with a saturating TF term and explicit document-length normalisation, and it consistently beats plain TF-IDF for retrieval.

**Why L2 normalisation is applied afterwards.** Without it, a long document has larger counts and therefore a longer vector, so it would score higher against any query purely because of length. Normalising each document vector to unit length means cosine similarity — the dot product of two normalised vectors — measures only the *angle*, that is, the relative composition of terms. This is why cosine similarity, not Euclidean distance, is the standard metric for TF-IDF vectors.

**Follow-up:** *Why would you still use TF-IDF now that embeddings exist?* Because it is exact on rare terms, and that is the failure mode of dense retrieval. Embedding models compress meaning into a few hundred dimensions, which works well for paraphrase and topical similarity but blurs exact tokens — a product code, an error string, a surname, a rare acronym. TF-IDF and BM25 match those exactly by construction. It is also fast, needs no training or GPU, and is fully interpretable: you can point at which terms produced a score. Modern retrieval systems therefore run *hybrid* search, combining BM25 and dense scores (often via reciprocal rank fusion), because the two fail on complementary queries.

> **Why the interviewer asks this.** It is a cheap probe of whether you can explain a weighting scheme's design choices — especially the log — rather than reciting the formula.

> **Saying it out loud.** "TF-IDF weights a term by how often it shows up in this document, times how rare it is across the collection. The rarity part is a log, and that matters — without it a word appearing once in a million documents would get a million times the weight and completely dominate. With the log, one appearing in one percent of docs gets about twice the weight of one in ten percent. The nice side effect is that a word in every document gets log of one, which is zero, so stopwords fall out automatically without a stoplist. And you normalise the vectors so that long documents don't score higher just for being long."

---

### Q39: What are n-grams? Explain n-gram language models.

**Answer:**

**N-grams** are contiguous sequences of n items (words) from text.

**Types:**
- **Unigram (1-gram)**: Single words ["machine", "learning"]
- **Bigram (2-gram)**: Pairs ["machine learning", "learning is"]
- **Trigram (3-gram)**: Triplets ["machine learning is"]

**N-gram Language Model:**

**Bigram Model:**
```
P(w₁, w₂, ..., wₙ) ≈ P(w₁) × P(w₂|w₁) × P(w₃|w₂) × ... × P(wₙ|wₙ₋₁)

Where:
P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
```

**Why N-grams:**
- **Unigram**: Simple but ignores word order
- **Bigram**: Captures local dependencies
- **Higher n**: More context but needs more data

**Trade-offs:**
- Higher n: More context, better predictions
- But: Needs more data (exponential growth), sparse data problem

**Use Cases:**
- Language modeling (predict next word)
- Text generation
- Spell checking

**The Markov assumption is what is actually being assumed.**

The exact chain rule for a sentence is $P(w_1, \dots, w_n) = \prod_t P(w_t \mid w_1, \dots, w_{t-1})$ — every word conditioned on the entire history, with no approximation. That is unestimable, because almost every long history occurs at most once in any corpus. An n-gram model makes the $(n-1)$-order **Markov assumption**: that the history can be truncated,

$$P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$$

so a bigram model claims one word of context is enough. The approximate sign in the formula above is doing all the work, and everything that is wrong with n-gram models follows from it: "The man who was standing by the door that I mentioned earlier ___" needs a subject-verb agreement decision that is 12 words away, and no fixed $n$ you can estimate will reach it.

**Why you cannot just raise $n$ — the sparsity arithmetic.** With a vocabulary of $V = 50{,}000$, the parameter count is $V^n$: 2.5 billion bigrams, $1.25 \times 10^{14}$ trigrams, $6\times10^{18}$ 4-grams. A corpus of a billion tokens contains at most a billion distinct 4-grams, so the overwhelming majority of the table is zero — not "rare", but never observed. And each zero is fatal under the product formula, since one zero factor sends the entire sentence probability to zero. This is the sparsity wall, and it is why smoothing (Q40) is not a refinement but a requirement, and why practical n-gram models topped out around $n = 5$ with heavy smoothing.

**A worked count.** Corpus: "the cat sat", "the cat ran", "the dog sat". Then $c(\text{the}) = 3$, $c(\text{the cat}) = 2$, $c(\text{the dog}) = 1$, so $P(\text{cat} \mid \text{the}) = 2/3$ and $P(\text{dog} \mid \text{the}) = 1/3$. Also $c(\text{cat}) = 2$, $c(\text{cat sat}) = 1$, giving $P(\text{sat} \mid \text{cat}) = 1/2$. The probability of "the cat sat" under the bigram model is $P(\text{the}) \times \frac{2}{3} \times \frac{1}{2}$. Notice that "the dog ran" — a perfectly good sentence — gets probability zero, because that bigram was never observed. These MLE estimates are just normalised counts, which is what makes n-gram models trivially fast to train and impossible to generalise.

**How this connects to what replaced it.** A neural language model attacks exactly this problem by representing each word as a dense vector, so "dog" and "cat" occupy nearby points and evidence about one informs the other — counts cannot share statistical strength between words, embeddings can. A transformer goes further and drops the fixed window entirely: self-attention conditions on the whole context, so the Markov assumption disappears rather than being loosened. Seeing n-gram models as "the chain rule plus a truncation you cannot afford" makes it obvious what each successor removed.

**Follow-up:** *Are n-grams still used?* Yes, in places where their weaknesses do not matter. They are extremely fast, need no GPU, and are exactly interpretable, so they still appear in production spell-checking and autocomplete, in the statistical components of some machine-translation and speech systems, in KenLM for shallow fusion during ASR decoding, and pervasively as *features* — character n-grams remain a strong, cheap baseline for language identification and authorship attribution. They are also the mechanism inside BLEU and ROUGE (Q42, Q43), so understanding them is not optional even in an all-neural pipeline.

> **Why the interviewer asks this.** It tests whether you can state the Markov assumption precisely and connect its failure to why neural language models exist.

> **Saying it out loud.** "An n-gram is just a contiguous run of n words, and an n-gram language model factorises sentence probability with a Markov assumption — you truncate the history to the last n minus one words and estimate each conditional by counting. The trouble is the counting doesn't scale: with a fifty-thousand-word vocabulary there are ten to the fourteen possible trigrams, so almost all of them are zero in any corpus, and a single zero kills the whole sentence probability. That's why you need smoothing. And it fundamentally can't handle long-range dependencies — agreement across a dozen words is just out of reach. Neural models fixed both: embeddings let similar words share evidence, and attention drops the fixed window entirely."

---

### Q40: What is Laplace smoothing? Why is it needed?

**Answer:**

**Laplace Smoothing (Add-k):**
Handles zero probability problem in n-gram models.

**Problem:**
- Unseen n-grams have P = 0
- Product of probabilities becomes 0
- Model can't handle unseen text

**Solution:**
```
P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + k) / (count(wᵢ₋₁) + k*V)

Where:
- k: Smoothing parameter (usually 1)
- V: Vocabulary size
```

**Effect:**
- **Seen n-grams**: Slightly lower probability
- **Unseen n-grams**: Non-zero probability (fixed!)
- **Redistributes**: Probability from seen to unseen

**Example:**
Training: "the cat", "the dog"
Test: "the bird" (unseen)

- Without smoothing: P(bird|the) = 0 (problem!)
- With smoothing: P(bird|the) = 1/5 = 0.2 (fixed!)

**Why needed:**
- Prevents zeros
- Allows generalization to unseen text
- Essential for language models

**Reading the formula as redistribution, with the example completed.**

Add-$k$ smoothing pretends you saw every possible continuation $k$ extra times. The numerator gains $k$, and the denominator gains $kV$ because there are $V$ possible next words each receiving $k$ phantom counts — that is what keeps the distribution summing to one, and it is the part people forget. The example above assumes a vocabulary of $V = 3$ ("cat", "dog", "bird") with $c(\text{the}) = 2$, giving $P(\text{bird} \mid \text{the}) = (0+1)/(2+3) = 1/5$. Worth completing the picture: the seen continuations drop from $1/2$ each to $(1+1)/(2+3) = 2/5$. So the three probabilities are $2/5, 2/5, 1/5$, summing to 1 — the unseen event was funded by taking mass from the observed ones.

**Why add-one is usually too much.** The mass moved to unseen events is $kV/(c + kV)$, and $V$ is large. With a realistic $V = 50{,}000$ and a context seen 10 times, add-one gives the observed continuations $(c+1)/(10 + 50{,}000)$ — the denominator is 5,000 times the actual count, so a continuation observed 5 times out of 10 goes from probability 0.5 to about 0.00012. Essentially *all* the probability mass has been handed to word types never seen in this context. That is not a small correction; it is a destruction of the estimate. Hence $k < 1$ (add-$\alpha$ or Lidstone smoothing, with $\alpha$ around 0.01-0.1 tuned on held-out data), and hence the fact that better methods exist.

**What better methods do differently.** The deeper flaw in add-$k$ is that it treats all unseen continuations as *equally* likely, which is obviously wrong: after "the", an unseen "aardvark" and an unseen "problem" should not get the same probability. The fixes all bring in information from lower-order models. *Backoff* falls back to the $(n-1)$-gram estimate when the $n$-gram count is zero. *Interpolation* always mixes orders, $\lambda_1 P(w_t \mid w_{t-2}, w_{t-1}) + \lambda_2 P(w_t \mid w_{t-1}) + \lambda_3 P(w_t)$, with the $\lambda$s tuned on held-out data. *Good-Turing* estimates the total mass of unseen events from the count of things seen exactly once, which is an elegant and surprisingly accurate trick. *Kneser-Ney*, the best of the classical methods, subtracts a fixed discount from every observed count and — its key insight — backs off not to how *frequent* a word is but to how many distinct contexts it appears in. That is why it correctly gives "Francisco" a low backoff probability despite it being common: it almost only ever follows "San", so it is a poor guess in a novel context.

**The Bayesian reading.** Add-$k$ is exactly the posterior mean under a symmetric Dirichlet prior with concentration $k$ over the multinomial — add-one is a uniform prior, Laplace's original "rule of succession". This is worth saying because it reframes the hyperparameter: $k$ is a *pseudocount*, your prior strength in units of observations, so choosing $k=1$ against a real count of 10 is asserting a prior worth $V$ observations against 10 real ones. Stated that way, its excessiveness is obvious.

**Follow-up:** *Do modern neural language models need smoothing?* Not in this form, because a softmax over the vocabulary assigns strictly positive probability to every token by construction — $e^{z}$ is never zero — so zero probabilities cannot occur and there is nothing to fix. The analogous concern is overconfidence, and the analogous tool is *label smoothing*: train against a target of $1-\epsilon$ for the correct token and $\epsilon/(V-1)$ spread over the rest, which prevents the logits from growing without bound and improves calibration. Same instinct — do not let a model assert certainty — implemented on the target side rather than the count side.

> **Why the interviewer asks this.** The formula is easy; knowing *why* the denominator gains $kV$ and why add-one is too blunt at realistic vocabulary sizes is the discriminating part.

> **Saying it out loud.** "Smoothing exists because a single unseen n-gram gives you probability zero, and since you're multiplying probabilities that zeroes out the whole sentence. Add-k pretends you saw every possible continuation k extra times — the numerator gets k and the denominator gets k times the vocabulary size, so it still sums to one. The problem is that with a fifty-thousand-word vocabulary, add-one shoves nearly all your probability mass onto words you've never seen in that context, which wrecks the estimate. So you use a much smaller k, or better, something like Kneser-Ney that backs off to lower-order counts — and its clever bit is backing off to how many distinct contexts a word appears in, not how often it appears."

---

### Q41: Explain the Bayesian interpretation of L1/L2 regularization.

**Answer:**

**L2 Regularization = Gaussian Prior:**
- **Frequentist**: Loss = MSE + λ||w||²
- **Bayesian**: Prior w ~ N(0, 1/λ)
- **Interpretation**: Parameters normally distributed around 0
- **Effect**: Shrinks all parameters toward 0 (smooth, no sparsity)

**L1 Regularization = Laplace Prior:**
- **Frequentist**: Loss = MSE + λ||w||₁
- **Bayesian**: Prior w ~ Laplace(0, 1/λ)
- **Interpretation**: Parameters Laplace distributed (sharp peak at 0)
- **Effect**: Shrinks parameters to exactly 0 (sparse, feature selection)

**Key Differences:**
- **L2 (Gaussian)**: Smooth bell curve, no sparsity
- **L1 (Laplace)**: Sharp peak at 0, creates sparsity

**Why it matters:**
- Helps choose right regularization
- Understand why L1 creates sparsity
- Interpret regularization strength (λ = prior variance)

**Use:**
- **L2**: Prevent overfitting, all features relevant
- **L1**: Feature selection, many irrelevant features

**The derivation, which takes four lines and makes the correspondence exact.**

Maximum a posteriori (MAP) estimation maximises the posterior $P(w \mid D) \propto P(D \mid w) P(w)$. Take the negative log to turn it into a minimisation:

$$\hat{w}_{\text{MAP}} = \arg\min_w \big[-\log P(D \mid w) - \log P(w)\big]$$

The first term is the ordinary loss — for Gaussian noise it is the sum of squared errors. The second term is the penalty, and it is entirely determined by the prior. Substitute a zero-mean Gaussian prior $P(w_j) \propto \exp(-w_j^2 / 2\tau^2)$ and $-\log P(w) = \frac{1}{2\tau^2}\sum_j w_j^2 + \text{const}$ — that is L2, with $\lambda = 1/(2\tau^2)$. Substitute a Laplace prior $P(w_j) \propto \exp(-|w_j|/b)$ and $-\log P(w) = \frac{1}{b}\sum_j |w_j| + \text{const}$ — that is L1, with $\lambda = 1/b$. So the correspondence is not an analogy; the regulariser *is* the negative log prior, and the regularisation strength *is* the inverse prior width.

That relation, $\lambda \propto 1/\tau^2$, is the useful takeaway: a strong penalty is a narrow, confident prior that the weights are near zero; a weak penalty is a wide, vague one. Setting $\lambda \to 0$ is a flat prior, which recovers maximum likelihood — regularisation and priors are the same knob.

**Why the shapes produce different behaviour, in prior terms.** Both densities peak at zero, but the Laplace density has a *kink* there — it is not differentiable at 0, and it is much more sharply peaked, with correspondingly fatter tails. Compare the two at equal variance: the Laplace prior puts more mass very near zero *and* more mass far out, and less in the middle range. That is precisely the belief "most coefficients are essentially zero, but a few are genuinely large", which is a sparsity prior. The Gaussian's smooth quadratic peak says "all coefficients are smallish and none are exactly zero", so its MAP solution has no reason to land on an axis. The kink at zero is what makes the MAP estimate stick there, which is the same fact as the constant-gradient argument in Q20, seen from the probabilistic side.

**One caveat that shows depth.** The MAP estimate under a Laplace prior is sparse, but the *posterior mean* under the same prior is not — the posterior assigns zero probability to any weight being exactly zero, since it is a continuous distribution. So "L1 gives sparsity" is a property of the MAP point estimate specifically, not of Bayesian inference with a Laplace prior. Genuinely Bayesian sparsity requires a spike-and-slab prior (a point mass at zero mixed with a broad distribution) or a continuous approximation to it such as the horseshoe. This distinction between MAP and full posterior is a good thing to be able to draw.

**Follow-up:** *What is the Bayesian view of early stopping and dropout?* Early stopping is an implicit regulariser: starting from small initialisation and halting before convergence keeps the weights near the origin, which for a linear model can be shown to be approximately equivalent to L2 with a $\lambda$ that decreases as training continues. Dropout has a stronger version of the correspondence — Gal and Ghahramani showed that a network trained with dropout is performing approximate variational inference in a deep Gaussian process, which is exactly why keeping dropout on at test time (Q21's follow-up) gives a usable posterior sample. The general pattern is that most regularisers can be read as priors, and reading them that way tells you what belief you are actually encoding.

> **Why the interviewer asks this.** It tests whether you can move between the optimisation and probabilistic framings of the same object — a good proxy for having real statistical grounding rather than recipe knowledge.

> **Saying it out loud.** "If you do MAP estimation instead of maximum likelihood, you're minimising negative log-likelihood plus negative log-prior — and that second term is exactly your regulariser. Put a Gaussian prior on the weights and the negative log is a sum of squares, so that's L2. Put a Laplace prior on and you get a sum of absolute values, so that's L1. The regularisation strength is the inverse of the prior's width, so a big lambda is just a narrow, confident prior that the weights are near zero. And the reason L1 gives sparsity falls out of the shape — the Laplace density has a kink at zero rather than a smooth peak, so the MAP estimate actually sticks there."

---

See `36_nlp_basics/regularization_priors.md` for comprehensive explanation!

---

### Q42: Explain BLEU score. How is it calculated?

**Answer:**

**BLEU** (Bilingual Evaluation Understudy) measures quality of machine translation or text generation.

**Components:**

**1. N-gram Precision:**
- Precision for n=1,2,3,4 (unigram, bigram, trigram, 4-gram)
- p_n = (matching n-grams) / (total n-grams in candidate)
- Clipped: Count capped at reference count

**2. Brevity Penalty (BP):**
- Penalizes short translations
- BP = 1 if candidate > reference length
- BP = exp(1 - ref_len/cand_len) otherwise

**3. BLEU Formula:**
```
BLEU = BP * exp(Σ w_n * log(p_n))

Where:
- w_n: Weights (usually [0.25, 0.25, 0.25, 0.25])
- p_n: n-gram precisions
```

**Range:** 0 to 1 (higher is better)

**Interpretation:**
- 1.0: Perfect match
- 0.5-0.7: Good translation
- <0.3: Poor translation

**Limitations:**
- Doesn't consider meaning (only n-gram overlap)
- Doesn't handle synonyms well
- Favors shorter translations (even with BP)

**Walking a full calculation, because BLEU is where people's understanding stops at the formula.**

Reference: *"the cat is on the mat"*. Candidate: *"the cat the cat on the mat"* (7 tokens).

*Unigram precision with clipping.* Candidate unigram counts: the$\times$3, cat$\times$2, on$\times$1, mat$\times$1. Reference counts: the$\times$2, cat$\times$1, is$\times$1, on$\times$1, mat$\times$1. Clipping caps each candidate count at the reference count, so "the" contributes $\min(3,2)=2$ and "cat" contributes $\min(2,1)=1$, plus 1 each for "on" and "mat". Total matched $=5$, out of 7 candidate unigrams, so $p_1 = 5/7 = 0.714$.

*Clipping is the mechanism worth explaining.* Without it, a candidate of just *"the the the the"* would score $p_1 = 4/4 = 1.0$ against this reference — perfect precision from pure repetition. Clipping caps the credit at how many times the word actually appears in the reference, so that candidate scores $2/4 = 0.5$. This is what stops the metric from being trivially gamed by repeating high-frequency words.

*Bigram precision.* Candidate bigrams: "the cat", "cat the", "the cat", "cat on", "on the", "the mat" (6 total). Reference bigrams: "the cat", "cat is", "is on", "on the", "the mat". Matches with clipping: "the cat" appears twice in the candidate but once in the reference, so it contributes 1; "on the" contributes 1; "the mat" contributes 1. So $p_2 = 3/6 = 0.5$. Note how the higher-order precisions are what actually penalise the scrambled word order — unigram precision barely noticed.

*Brevity penalty.* Candidate length 7, reference length 6. Since $c > r$, $BP = 1$; no penalty. If the candidate had been just *"the cat"* ($c=2$), $BP = e^{1 - 6/2} = e^{-2} = 0.135$, cutting the score by a factor of seven.

*Combining.* BLEU uses the *geometric* mean of the $p_n$, and that choice matters: a geometric mean is zero if any single term is zero. So a candidate with no matching 4-grams scores exactly zero regardless of how good its unigram precision is. This is why BLEU on a single short sentence is often 0 and why the metric is designed to be computed at the *corpus* level, aggregating the numerators and denominators over all sentences before dividing. Sentence-level BLEU needs smoothing (adding a small count to zero n-gram matches) to be usable at all.

**Why BLEU uses precision rather than recall.** Recall is ill-defined with multiple valid references — a translation cannot be expected to contain all n-grams of all references. Precision plus a brevity penalty is the workaround: precision stops you adding junk, and BP stops you from gaming precision by saying almost nothing. So the answer's note that BLEU "favours shorter translations even with BP" is best stated as: precision alone rewards brevity, and BP is the correction, which is approximately but not perfectly calibrated.

**Follow-up:** *What are the practical gotchas when reporting BLEU?* That BLEU numbers are not comparable across papers unless the tokenisation, casing, and number of references match — a difference in tokenisation alone can move the score by several points. This is exactly why SacreBLEU exists: it takes detokenised text and applies a fixed internal tokenisation, and it emits a version signature so a score can be reproduced. On modern systems, learned metrics such as COMET and BLEURT correlate substantially better with human judgement because they compare meaning via pretrained representations rather than surface n-grams, and chrF (character n-gram F-score) is a better lexical metric for morphologically rich languages. BLEU persists mainly as a cheap, deterministic, universally understood baseline.

> **Why the interviewer asks this.** Clipping and the geometric mean are the two details that separate someone who has implemented BLEU from someone who has called it.

> **Saying it out loud.** "BLEU is n-gram precision from one to four, combined with a geometric mean, times a brevity penalty. The two details that matter are clipping and the geometric mean. Clipping caps how much credit a repeated word can earn at its count in the reference — otherwise outputting 'the the the the' would score perfect unigram precision. And the geometric mean means if any order has zero matches, the whole score is zero, which is why sentence-level BLEU is usually zero and you're meant to compute it over a whole corpus. It's precision-based because recall doesn't make sense with multiple valid references, and the brevity penalty is there to stop you gaming precision by saying almost nothing."

---

### Q43: Explain ROUGE score. What are ROUGE-1, ROUGE-2, ROUGE-L?

**Answer:**

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference text.

**ROUGE-1 (Unigram):**
- Measures word overlap
- ROUGE-1 = (overlapping words) / (words in reference)
- Focus: Content words

**ROUGE-2 (Bigram):**
- Measures bigram overlap
- ROUGE-2 = (overlapping bigrams) / (bigrams in reference)
- Focus: Word order and phrases

**ROUGE-L (Longest Common Subsequence):**
- Measures LCS overlap
- ROUGE-L = LCS(candidate, reference) / length(reference)
- Focus: Sentence structure and order
- LCS: Longest sequence appearing in both (not necessarily contiguous)

**Returns:** Precision, Recall, F1

**Use Cases:**
- **Summarization**: Primary metric
- **Text generation**: Secondary metric
- **ROUGE-1**: Content coverage
- **ROUGE-2**: Phrase matching
- **ROUGE-L**: Structure similarity

**Comparison to BLEU:**
- **BLEU**: Precision-oriented (translation)
- **ROUGE**: Recall-oriented (summarization)
- **ROUGE-L**: Better for order-sensitive tasks

**A worked example, and the LCS detail that ROUGE-L turns on.**

Reference: *"the cat sat on the mat"*. Candidate: *"the cat was on the mat"*.

*ROUGE-1.* Overlapping unigrams: the$\times$2, cat, on, mat $=5$. Reference has 6 unigrams, candidate has 6, so recall $=5/6=0.833$, precision $=5/6=0.833$, F1 $=0.833$.

*ROUGE-2.* Reference bigrams: "the cat", "cat sat", "sat on", "on the", "the mat" (5). Candidate bigrams: "the cat", "cat was", "was on", "on the", "the mat" (5). Overlap: "the cat", "on the", "the mat" $=3$. Recall $=3/5=0.6$, precision $=3/5=0.6$. Notice the single substituted word destroyed two bigrams while costing only one unigram — bigram scores fall roughly twice as fast, which is why ROUGE-2 is always much lower than ROUGE-1 and why the two numbers are not comparable to each other.

*ROUGE-L.* The longest common subsequence is "the cat on the mat", length 5 — subsequences need not be contiguous, which is the whole point: ROUGE-L rewards preserved *order* without demanding adjacency, so an inserted or substituted word costs you that word but not the surrounding structure. Recall $=5/6$, precision $=5/6$, and the reported figure is the F-measure. A detail worth knowing: ROUGE-L as originally defined uses an F-measure weighted by $\beta$ strongly favouring recall, and there are two variants — sentence-level LCS, and ROUGE-Lsum, which computes LCS per sentence and aggregates. Summarisation papers usually report ROUGE-Lsum, and the two differ enough that mixing them up invalidates a comparison.

**Why recall-oriented, and what it costs.** In summarisation the question is "did the summary cover the important content of the source", which is a recall question; in translation the question is "is what you produced correct", which is precision. Hence the split with BLEU. But pure recall is trivially gamed by producing a long summary, which is why every modern report uses the F1 variants and why summarisation evaluations must control for length — a system that writes longer summaries will show higher ROUGE recall while being no better.

**The limitation that matters most.** ROUGE counts surface n-gram overlap, so it cannot see paraphrase. *"The film was excellent"* and *"the movie was superb"* share almost no unigrams and would score near zero against each other despite being the same statement. That is a serious problem now that abstractive summarisers genuinely paraphrase — ROUGE systematically undervalues good abstractive output and overvalues extractive output that copies phrases verbatim. Two consequences follow: a purely extractive baseline (say, "take the first three sentences", the notorious Lead-3 baseline on news) is hard to beat on ROUGE while being obviously worse to read, and reported ROUGE gains often do not survive human evaluation.

**Follow-up:** *What would you use instead?* BERTScore matches tokens via contextual embeddings and cosine similarity, so paraphrase is credited; it correlates far better with human judgement while staying reference-based. For faithfulness specifically — whether the summary states anything the source does not — n-gram overlap is the wrong tool entirely, and the standard approaches are entailment-based (does the source entail each summary sentence) or QA-based (generate questions from the summary and check the source answers them the same way). Increasingly the practical choice is LLM-as-judge with a rubric, which correlates well but introduces its own biases toward length and toward the judge model's own style. The honest answer in an interview is that you report ROUGE for comparability with prior work and something else for the decision you are actually making.

> **Why the interviewer asks this.** Usually to reach the limitation: knowing that ROUGE cannot see paraphrase, and that this biases evaluation toward extractive systems, is the substantive part.

> **Saying it out loud.** "ROUGE measures overlap between a generated summary and a reference. ROUGE-1 is unigrams, ROUGE-2 is bigrams — and bigrams drop much faster, because one substituted word breaks two bigrams but only one unigram. ROUGE-L uses the longest common subsequence, which doesn't have to be contiguous, so it rewards keeping the right order without demanding exact adjacency. It's recall-oriented because in summarisation the question is coverage, whereas BLEU is precision-oriented because in translation the question is correctness. The big weakness is that it's pure surface overlap — 'the film was excellent' and 'the movie was superb' score near zero against each other — so it quietly punishes good paraphrasing and rewards copying."

---

### Q44: How do you handle large database schemas in NL2Code?

**Answer:**

**Problem:** Large schemas (thousands of tables/columns) don't fit in context window.

**Solutions:**

**1. Schema Pruning:**
- **Relevance scoring**: Score tables/columns by relevance to query
  - TF-IDF similarity
  - Embedding similarity (BERT)
  - Keyword matching
- **Top-K selection**: Select top-K most relevant elements
- **Hierarchical**: Prune at table level, then column level

**2. Schema Encoding:**
- **Hierarchical encoding**: Encode at different levels
- **Graph neural networks**: Model schema as graph
- **Separate encoding**: Encode schema separately, combine later

**3. Two-Stage Approach:**
- **Stage 1**: Schema selection (which tables/columns needed)
- **Stage 2**: Code generation (given selected schema)

**4. Retrieval-Augmented:**
- **Retrieve relevant schema**: Use retrieval to find relevant parts
- **Dynamic context**: Add retrieved schema to context
- **Iterative**: Refine selection based on generation

**Standard Procedure:**
```
Query → Schema Pruning → Schema Encoding → Code Generation → Code
```

**Example:**
- Query: "Find customers who bought products in 2023"
- Pruned schema: customers, orders, products tables + relevant columns
- Generated SQL: SELECT with JOINs on relevant tables

**Best Practices:**
- Index schemas for fast retrieval
- Add schema descriptions
- Handle schema versioning
- Validate generated code

**Why this is a retrieval problem before it is a generation problem.**

The framing that makes the answer coherent: with a thousand tables, the model's difficulty is not writing SQL, it is *deciding which four tables the question is about*. Published error analyses on cross-domain text-to-SQL consistently find schema linking — connecting phrases in the question to the right columns and tables — is the dominant error source, ahead of SQL syntax or join logic. So most of the engineering effort belongs in the selection stage, and the accuracy ceiling of the whole system is set by the recall of that stage: if pruning drops a needed table, no amount of generation quality recovers it. That asymmetry dictates the tuning target — **optimise the pruning stage for recall, not precision**, and let the generator discard the extras. Retrieving 30 candidate tables of which 4 are needed is fine; retrieving 5 of which one needed table is missing is fatal.

**What "relevance scoring" actually has to handle.** Naive embedding similarity between the question and a column name fails on the cases that matter, because schema names are rarely natural language: `cust_dob`, `t_ord_hdr`, `flg_actv`. Three things help concretely. First, *verbalise the schema* — turn each table into a sentence ("Table orders: one row per customer order, with columns order_id, customer_id, order_date, total_amount") using column comments and any data dictionary, and embed that rather than the raw identifier. Second, *include sample values*, because a question mentioning "California" links to a column only if you know that column contains state names; indexing distinct values of low-cardinality string columns is one of the highest-leverage additions available. Third, *combine lexical and semantic retrieval*, since exact matches on identifier fragments are precisely what dense embeddings blur.

**Foreign keys change the retrieval problem.** Tables are not independent documents — selecting `orders` and `products` without the `order_items` join table produces SQL that cannot be written. So after scoring, expand the selected set along foreign-key edges to include any table on a join path between selected tables. This is graph closure, not ranking, and it is a common omission: the join table often has no lexical or semantic overlap with the question at all and will never be retrieved on its own merits.

**Making the two-stage approach robust.** The failure mode of a hard two-stage pipeline is that stage-1 errors are unrecoverable. Two mitigations: keep a generous candidate set as described above, and add a *validation-and-repair loop* — run `EXPLAIN` or execute the generated query against the database, and feed any error back to the model with the message and possibly extra schema. A large share of errors are mechanically detectable (unknown column, ambiguous reference, type mismatch) and a single repair round fixes many of them. Execution-guided decoding, where you filter candidate queries by whether they run and return a non-empty result, is the stronger version of the same idea.

**Follow-up:** *How do you evaluate an NL2SQL system?* Not by string match against a reference query — there are many correct SQL statements for one question, differing in join order, aliasing, or subquery versus CTE. The standard metric is *execution accuracy*: run both the predicted and reference query against a real database and compare result sets. For that to be meaningful the test data must exercise the distinctions you care about, since an underpopulated table can make a wrong query return the same rows as a right one — which is what test-suite accuracy addresses, by running against several databases chosen to distinguish semantically different queries. Alongside that, track schema-linking recall separately, because it tells you which stage to fix.

> **Why the interviewer asks this.** It is a systems-design question wearing an NLP costume; they want to see you decompose it into retrieval, generation, and validation with a clear view of where the errors concentrate.

> **Saying it out loud.** "The core insight is that with a big schema the hard part isn't writing SQL, it's working out which four tables out of a thousand the question is even about — schema linking is where most of the errors are. So it's really a retrieval problem first. I'd verbalise each table into a description using comments and sample values, index that, and retrieve with a hybrid of lexical and embedding search. Then I'd expand along foreign keys, because the join table you need often has nothing in common with the question. I'd tune that stage for recall rather than precision, since an extra table is cheap and a missing one is fatal. And I'd close the loop by executing the query and feeding errors back for a repair round."

---

### Q45: What are the standard procedures for different NLP tasks?

**Answer:**

**1. Text Classification:**
- Preprocess → Feature extraction (TF-IDF/embeddings) → Model → Evaluate
- Metrics: Accuracy, F1-score

**2. NER:**
- BIO tagging → Embeddings → Sequence labeling (CRF/BiLSTM) → Extract entities
- Metrics: F1 per entity type

**3. Question Answering:**
- Encode question+context → Attention → Extract answer span
- Metrics: EM, F1

**4. Machine Translation:**
- Parallel corpus → Tokenization → Seq2Seq/Transformer → Beam search
- Metrics: BLEU, METEOR

**5. Summarization:**
- **Extractive**: Sentence ranking → Select top sentences
- **Abstractive**: Encode → Generate summary
- Metrics: ROUGE-1/2/L

**6. NL2Code:**
- Query → Schema pruning → Schema encoding → Code generation
- Metrics: CodeBLEU, Execution accuracy

**General Pipeline:**
```
Text → Preprocessing → Feature Extraction → Model → Output → Evaluation
```

**Key Points:**
- Start with simple baselines
- Use pre-trained models when possible
- Evaluate with task-specific metrics
- Handle domain-specific challenges

**The unifying observation.** Every pipeline in the list above is the same three moves — represent the text, predict a structure, score against a reference — and what changed over the last decade is only *how much of it is learned*. The classical era engineered the representation (TF-IDF, hand-built features) and used a small task-specific model. The pretrain-then-finetune era learned the representation once and attached a small task head. The current era often replaces the whole middle with a prompt. Being able to say which era a given approach belongs to, and when to pick each, is more useful than the list itself.

**When to pick which, concretely.** *Prompt a general model* when you have no labelled data, when the task is fluid or one-off, or when you need it working this week; expect the highest per-item cost and latency. *Fine-tune a small encoder* (a BERT-family model, 100M-400M parameters) when the task is fixed and high-volume: with a few thousand labels it will typically match or beat a large prompted model on a narrow classification or tagging task, at a thousandth the inference cost and single-digit milliseconds of latency. *Fine-tune a generative model with LoRA* when the output is free-form but the style or format is specific. The economics matter more than the accuracy in most production decisions, and interviewers notice when a candidate raises them unprompted.

**Two things the list omits that dominate real projects.** First, *the label set is the hard part*. For NER the difficulty is almost never the model — it is deciding whether a product name inside a company name is one entity or two, and getting annotators to apply that consistently; inter-annotator agreement below about 0.8 Cohen's kappa means the ceiling on your model is already set by the noise. Second, *evaluation design outranks model choice*. A random train-test split leaks when documents come from the same source or share near-duplicates, which is endemic in scraped corpora, and it produces the classic result of a model that scores 0.95 offline and fails on deployment. Splitting by document, by time, or by source is the fix, and choosing the split is a modelling decision, not a bookkeeping one.

**Metric selection, since the list gives metrics without caveats.** Accuracy is misleading under class imbalance — 99% accuracy on a 1%-positive problem is achieved by predicting "no" — so use per-class F1, and be explicit about macro (unweighted class average, which surfaces poor performance on rare classes) versus micro (which is dominated by the frequent ones). NER F1 should be computed on whole spans, not tokens, because getting three of four tokens of an entity right is not three-quarters correct. And for any generation task, remember Q42 and Q43: the automatic metric is a proxy, and if the decision matters, a small human evaluation on a couple of hundred examples will tell you more than a decimal point of ROUGE.

**Follow-up:** *What is the first thing you do on a new NLP task?* Build the dumbest possible end-to-end baseline and get it evaluated — majority class, keyword rules, or TF-IDF with logistic regression — before touching a transformer. It takes under an hour and it does three things: it establishes the score anyone must beat, it forces the evaluation harness to exist early, and it frequently exposes that the task is nearly solvable by a keyword, or that the labels are too noisy to learn from at all. Both discoveries are much cheaper to make on day one than after a week of fine-tuning.

> **Why the interviewer asks this.** Breadth check — they want to know whether you have a mental map of the field and can choose an approach for a task you have not seen before.

> **Saying it out loud.** "They're all the same three steps really — represent the text, predict some structure, score it against a reference — and what's changed is how much of that is learned rather than engineered. So the question I actually ask is which tool fits: if I've got no labels and need something working now, I prompt a general model. If it's a fixed high-volume classification task, I'd fine-tune a small encoder, because with a few thousand labels it'll match a big model at a fraction of the cost and latency. And whatever the task, I'd build a dumb baseline first — TF-IDF and logistic regression — because it forces the evaluation harness to exist and it quite often reveals that the labels are too noisy to learn from anyway."

---

See `36_nlp_basics/nlp_tasks_and_solutions.md` for detailed procedures!

---

## MLE and MAP Estimation

### Q46: Derive MLE for a coin flip (Bernoulli distribution).

**Answer:**

**Setup:** n flips, k heads, model P(heads) = θ

**Likelihood:** L(θ) = θᵏ × (1-θ)ⁿ⁻ᵏ

**Log-likelihood:** log L(θ) = k log θ + (n-k) log(1-θ)

**Derivative:** d/dθ [log L(θ)] = k/θ - (n-k)/(1-θ)

**Set to zero:** k/θ = (n-k)/(1-θ) → θ = k/n

**Result:** θ̂_MLE = k/n (observed proportion)

**Intuition:** MLE is simply the proportion of heads observed!

**Full walkthrough, one step at a time.**

The phrase *maximum likelihood* names the recipe exactly: write down the probability of the data you actually saw as a function of the unknown parameter, then pick the parameter value that makes that number as large as possible. Nothing else is going on. Here is every step with a word on why the move is allowed.

**Step 0 — the model.** Each flip is a Bernoulli random variable: it is $1$ (heads) with probability $\theta$ and $0$ (tails) with probability $1-\theta$. *Bernoulli* just means "a single yes/no trial with a fixed success probability." We assume the flips are i.i.d. — *independent and identically distributed*, meaning no flip influences another and every flip uses the same $\theta$. That assumption is what lets us multiply the per-flip probabilities in the next step.

**Step 1 — the likelihood.** For one flip with outcome $x_i \in \{0,1\}$, a compact way to write the probability is $\theta^{x_i}(1-\theta)^{1-x_i}$; plug in $x_i=1$ and you get $\theta$, plug in $x_i=0$ and you get $1-\theta$. Independence lets us multiply across flips:

$$L(\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^{k}(1-\theta)^{n-k}, \qquad k=\sum_i x_i .$$

Notice the data enters only through $k$ and $n$. In statistical language $k$ is a *sufficient statistic* — the order of the flips carries no information about $\theta$.

**Step 2 — take logs.** $\log$ is strictly increasing, so whatever maximizes $L$ also maximizes $\log L$. That is the entire justification, and it is why the move is legal rather than merely convenient:

$$\ell(\theta) = \log L(\theta) = k\log\theta + (n-k)\log(1-\theta).$$

**Step 3 — differentiate.** Using $\frac{d}{d\theta}\log\theta = 1/\theta$ and the chain rule on $\log(1-\theta)$, whose inner derivative is $-1$:

$$\ell'(\theta) = \frac{k}{\theta} - \frac{n-k}{1-\theta}.$$

**Step 4 — set to zero and solve.** Setting $\ell'(\theta)=0$ and cross-multiplying by $\theta(1-\theta)$, which is strictly positive for $\theta \in (0,1)$ so it cannot introduce or destroy a root:

$$k(1-\theta) = (n-k)\theta \;\Longrightarrow\; k - k\theta = n\theta - k\theta \;\Longrightarrow\; k = n\theta \;\Longrightarrow\; \hat\theta = \frac{k}{n}.$$

**Step 5 — check it is a maximum, not a minimum.** A stationary point is only a maximum if the function curves downward there. The second derivative is

$$\ell''(\theta) = -\frac{k}{\theta^{2}} - \frac{n-k}{(1-\theta)^{2}} < 0$$

for every $\theta$ in $(0,1)$ whenever $0<k<n$, so $\ell$ is strictly concave and the stationary point is the unique global maximum. This step is the one candidates skip and interviewers notice.

**Edge cases.** If $k=0$ the log-likelihood is $n\log(1-\theta)$, which increases as $\theta$ falls, so the maximum sits at the boundary $\hat\theta=0$; symmetrically $k=n$ gives $\hat\theta=1$. The derivative-equals-zero argument does not apply at a boundary. This is exactly the pathology that a prior fixes — see Q51.

**Worked numbers.** Ten flips, three heads: $\hat\theta = 3/10 = 0.3$. Three flips, three heads: $\hat\theta = 1.0$, i.e. the model now claims the coin *never* lands tails. That confident nonsense from three data points is the standard motivation for MAP.

**Follow-up:** *What is the variance of this estimator?* Since $k \sim \text{Binomial}(n,\theta)$ has variance $n\theta(1-\theta)$, the estimator $\hat\theta=k/n$ has variance $\theta(1-\theta)/n$ — it shrinks like $1/n$, and it is largest at $\theta=0.5$, which is the intuitive statement that a fair coin is the hardest one to pin down.

> **Why the interviewer asks this.** It is the smallest possible derivation that still has all the moving parts, so it reveals whether you actually manipulate likelihoods or merely recite the answer $k/n$.

> **Saying it out loud.** I write down the probability of exactly the data I saw as a function of theta — that's theta to the k times one-minus-theta to the n-minus-k. Then I take the log, because log turns that product into a sum and doesn't move the maximum, since log is increasing. Differentiate, set it to zero, and it collapses to theta equals k over n — just the observed fraction of heads. I'd also check the second derivative is negative so I know it's a max, and flag that if I saw three heads out of three, MLE says the coin never lands tails, which is where a prior starts earning its keep.

---

### Q47: Derive MLE for linear regression.

**Answer:**

**Setup:** y = Xw + ε, where ε ~ N(0, σ²)

**Likelihood:** L(w) ∝ exp(-||y - Xw||²/(2σ²))

**Log-likelihood:** log L(w) = -||y - Xw||²/(2σ²) + constant

**Maximize:** argmax_w log L(w) = argmin_w ||y - Xw||²

**Derivative:** ∂/∂w [||y - Xw||²] = -2Xᵀ(y - Xw) = 0

**Result:** ŵ_MLE = (XᵀX)⁻¹Xᵀy (Ordinary Least Squares!)

**Key Insight:** MLE for linear regression with Gaussian noise = OLS!

**Full walkthrough, one step at a time.**

**Step 0 — what the model actually claims.** Writing $y = Xw + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,\sigma^2 I)$ says three separate things: the mean of $y$ given $x$ is linear in $w$; the noise around that mean is Gaussian; and the noise is *homoscedastic and uncorrelated* — same variance $\sigma^2$ at every point, and no correlation between rows. That last part is what makes the joint density a product over rows.

**Step 1 — the likelihood.** The density of one observation is the Gaussian density evaluated at the residual:

$$p(y_i \mid x_i, w) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\!\left(-\frac{(y_i - x_i^{\top}w)^{2}}{2\sigma^{2}}\right).$$

Multiplying over $i$ (independence again) and collecting the exponents:

$$L(w) = (2\pi\sigma^{2})^{-n/2}\exp\!\left(-\frac{\lVert y - Xw\rVert^{2}}{2\sigma^{2}}\right).$$

**Step 2 — log, and drop constants.** Taking logs turns the exponential into the thing in the exponent:

$$\ell(w) = -\frac{n}{2}\log(2\pi\sigma^{2}) - \frac{1}{2\sigma^{2}}\lVert y - Xw\rVert^{2}.$$

The first term does not contain $w$, so it cannot change where the maximum in $w$ sits; drop it. The remaining factor $1/(2\sigma^2)$ is a positive constant, and scaling an objective by a positive constant does not move its argmax. So

$$\arg\max_w \ell(w) = \arg\min_w \lVert y - Xw\rVert^{2}.$$

That single line is the whole "Gaussian noise implies least squares" result. Least squares is not a taste in loss functions; it is the maximum-likelihood consequence of assuming Gaussian errors.

**Step 3 — expand before differentiating.** Write the squared norm as an inner product and expand:

$$\lVert y - Xw\rVert^{2} = (y-Xw)^{\top}(y-Xw) = y^{\top}y - 2w^{\top}X^{\top}y + w^{\top}X^{\top}Xw,$$

using $y^\top X w = w^\top X^\top y$ because a scalar equals its own transpose.

**Step 4 — differentiate term by term.** The two vector-calculus identities you need are $\nabla_w (w^{\top}a) = a$ and $\nabla_w (w^{\top}Aw) = 2Aw$ when $A$ is symmetric — and $X^{\top}X$ is symmetric by construction. So

$$\nabla_w \lVert y - Xw\rVert^{2} = -2X^{\top}y + 2X^{\top}Xw = -2X^{\top}(y - Xw).$$

**Step 5 — set to zero.** $X^{\top}(y-Xw)=0$ is the *normal equation*: the residual vector is orthogonal to every column of $X$. Geometrically the fitted values are the orthogonal projection of $y$ onto the column space of $X$, which is the picture worth carrying into the interview. Solving,

$$X^{\top}Xw = X^{\top}y \;\Longrightarrow\; \hat w_{\text{MLE}} = (X^{\top}X)^{-1}X^{\top}y.$$

**Step 6 — when is this valid, and is it a minimum?** The inverse exists only if $X^{\top}X$ is invertible, i.e. $X$ has full column rank — no exactly collinear features and at least as many rows as columns. The Hessian is $2X^{\top}X$, which is positive semi-definite always and positive definite exactly under that full-rank condition, so the stationary point is the unique global minimum. When rank is deficient there are infinitely many optima, and the standard fixes are the pseudo-inverse or a ridge penalty (Q49).

**A note on $\sigma^2$.** It dropped out of the $w$ solution but it is still a parameter. Maximizing $\ell$ over $\sigma^2$ gives $\hat\sigma^{2} = \lVert y - X\hat w\rVert^{2}/n$ — the mean squared residual. That is the *biased* variance estimate; the familiar $n-p$ denominator comes from an unbiasedness correction, not from MLE. Interviewers like this detail because it shows you know MLE is not automatically unbiased.

> **Why the interviewer asks this.** They want to see that you can connect a probabilistic assumption to a loss function, because that link is what lets you invent a loss for a new problem instead of guessing.

> **Saying it out loud.** If I assume the noise is Gaussian with constant variance, the likelihood is a product of Gaussians, and its log is just minus the sum of squared residuals over two sigma squared, plus constants. The constants don't move the argmax, so maximizing likelihood is literally minimizing squared error — least squares isn't a stylistic choice, it falls out of the Gaussian assumption. Expand the quadratic, take the gradient, set it to zero, and you get the normal equation X-transpose times the residual equals zero, which geometrically says the residual is orthogonal to the column space. Solve and you get X-transpose-X inverse X-transpose y, valid as long as X has full column rank.

---

### Q48: Explain the connection between MLE and MAP.

**Answer:**

**MLE:** θ̂_MLE = argmax_θ log P(D|θ)

**MAP:** θ̂_MAP = argmax_θ [log P(D|θ) + log P(θ)] = MLE + log(prior)

**Relationship:** MAP = MLE + Prior

**When same:**
- Uniform prior → MAP = MLE
- Large dataset → MAP ≈ MLE

**When different:**
- Small dataset → Prior has more influence
- Strong prior → MAP pulled toward prior mean

**Regularization:**
- L2 (Ridge) = MAP with Gaussian prior
- L1 (Lasso) = MAP with Laplace prior

**Where the prior comes from: Bayes' rule in one line.**

The bridge between the two estimators is Bayes' rule, $P(\theta \mid D) = P(D\mid\theta)P(\theta)/P(D)$. Take logs of both sides:

$$\log P(\theta\mid D) = \underbrace{\log P(D\mid\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{log-prior}} - \underbrace{\log P(D)}_{\text{constant in }\theta}.$$

The evidence term $\log P(D)$ does not depend on $\theta$, so it cannot move the argmax and we discard it. What is left is exactly "MLE plus a log-prior," which is why MAP looks like a regularized MLE. *MAP* stands for maximum a posteriori: the mode — the highest point — of the posterior distribution.

**Why regularizers are log-priors, concretely.** A zero-mean Gaussian prior $w \sim \mathcal{N}(0, \tau^{2}I)$ has log-density $-\lVert w\rVert^{2}/(2\tau^{2}) + \text{const}$. Subtracting that from the negative log-likelihood is an L2 penalty — ridge. A Laplace prior $p(w_j) \propto \exp(-|w_j|/b)$ has log-density $-|w_j|/b$, giving an L1 penalty — lasso. The Laplace density is sharply peaked at zero with heavier tails than a Gaussian, which is the probabilistic reason lasso pushes coefficients exactly to zero while ridge only shrinks them: the L1 penalty has a non-zero-width kink at the origin, so zero is a genuine optimum for a whole range of data, whereas the Gaussian's smooth parabola has zero gradient at the origin and never pins a coefficient there.

**Why "large dataset makes them agree" is more than a slogan.** The log-likelihood is a sum of $n$ terms and therefore grows linearly in $n$; the log-prior is a single fixed term that does not grow at all. So the prior's share of the objective falls like $1/n$. Under mild regularity conditions this is the Bernstein–von Mises phenomenon: the posterior concentrates on the true parameter and becomes asymptotically Gaussian regardless of which (positive, smooth) prior you started from. The practical reading is that priors are leverage on small data and rounding error on large data.

**Follow-up:** *Is MAP a Bayesian method?* Only half-heartedly. It uses a prior, which is Bayesian, but it reports a single point — the posterior mode — and throws away the uncertainty, which is not. It is also not invariant under reparameterization: the mode of a density changes if you transform the parameter (because the Jacobian reshapes the density), whereas the posterior *mean* and full posterior transform sensibly. Fully Bayesian inference integrates over the posterior rather than maximizing it.

> **Why the interviewer asks this.** They are checking whether you see regularization as a principled modelling statement rather than a knob you turn until validation loss drops.

> **Saying it out loud.** Start from Bayes' rule and take logs: the log-posterior is the log-likelihood plus the log-prior, minus the evidence, which is constant in theta so it drops. So MAP is just MLE with an extra term, and that extra term is your regularizer. A Gaussian prior gives you the L2 penalty — that's ridge — and a Laplace prior gives you L1, which is lasso. And because the likelihood term grows with n while the prior stays fixed, the prior matters a lot on small data and washes out on big data.

---

### Q49: Derive MAP for linear regression with Gaussian prior (Ridge).

**Answer:**

**Setup:** y = Xw + ε, prior w ~ N(0, σ²_prior I)

**Posterior:** log P(w|D) = -||y - Xw||²/(2σ²) - ||w||²/(2σ²_prior)

**Derivative:** ∂/∂w [log P(w|D)] = -1/σ² × Xᵀ(y - Xw) - 1/σ²_prior × w = 0

**Result:** ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy where λ = σ²/σ²_prior

**Key Insight:** MAP with Gaussian prior = Ridge regression (L2 regularization)!

**Full walkthrough, one step at a time.**

**Step 0 — set up both pieces.** The likelihood is the same Gaussian-noise model as Q47. The prior says each weight is drawn independently from $\mathcal{N}(0,\tau^{2})$, written jointly as $w \sim \mathcal{N}(0,\tau^{2}I)$, where $\tau^{2}$ is the prior variance (the answer above calls it $\sigma^{2}_{\text{prior}}$). A small $\tau^{2}$ is a confident statement that the weights are near zero; a large $\tau^{2}$ is an agnostic one.

**Step 1 — write the log-posterior.** Dropping every term free of $w$:

$$\log P(w\mid D) = -\frac{1}{2\sigma^{2}}\lVert y - Xw\rVert^{2} \;-\; \frac{1}{2\tau^{2}}\lVert w\rVert^{2} \;+\; \text{const}.$$

The first term comes from the Gaussian likelihood, the second from the Gaussian prior; both are just "quadratic in the exponent."

**Step 2 — flip the sign and rescale.** Maximizing that is minimizing its negative, and multiplying by the positive constant $2\sigma^{2}$ leaves the argmin alone:

$$\hat w_{\text{MAP}} = \arg\min_w \; \lVert y - Xw\rVert^{2} + \lambda\lVert w\rVert^{2}, \qquad \lambda = \frac{\sigma^{2}}{\tau^{2}}.$$

That $\lambda$ is worth pausing on: it is a *ratio of variances*, noise over prior. Noisy data (large $\sigma^2$) or a confident prior (small $\tau^2$) both push $\lambda$ up and shrink the weights harder. This is the sentence that makes ridge stop feeling arbitrary.

**Step 3 — differentiate.** Using the same two identities as Q47 plus $\nabla_w \lVert w\rVert^{2} = 2w$:

$$-2X^{\top}(y - Xw) + 2\lambda w = 0.$$

**Step 4 — collect terms in $w$.** Distribute and move things across:

$$-X^{\top}y + X^{\top}Xw + \lambda w = 0 \;\Longrightarrow\; (X^{\top}X + \lambda I)\,w = X^{\top}y \;\Longrightarrow\; \hat w_{\text{MAP}} = (X^{\top}X + \lambda I)^{-1}X^{\top}y.$$

**Step 5 — why this is strictly better conditioned.** $X^{\top}X$ is symmetric positive semi-definite, so its eigenvalues are $\ge 0$; adding $\lambda I$ shifts every eigenvalue up by $\lambda$, making the matrix positive definite and therefore invertible *even when $X$ is rank deficient or has more columns than rows*. The condition number, the ratio of largest to smallest eigenvalue, drops from $\lambda_{\max}/\lambda_{\min}$ to $(\lambda_{\max}+\lambda)/(\lambda_{\min}+\lambda)$. This is the numerical reason ridge is used on collinear data even by people who do not care about Bayes.

**What ridge does in the singular-value basis.** If $X = U\Sigma V^{\top}$ is the singular value decomposition, then OLS divides each component by $s_j$ while ridge divides by $s_j + \lambda/s_j$ — equivalently it multiplies the OLS coefficient in direction $j$ by the shrinkage factor $s_j^{2}/(s_j^{2}+\lambda)$. Directions with large singular values (well-determined by the data) are barely touched; directions with tiny singular values (where the data says almost nothing) are crushed toward zero. That is *exactly* the behaviour you would want a prior to have, and seeing it stated this way is usually what separates a good answer from a great one.

**Follow-up:** *Should the intercept be penalized?* No. Penalizing it makes the fit depend on where you happen to have put the origin of $y$, so in practice you centre the response and the features and leave the intercept out of the penalty. Features should also be standardized first, since an L2 penalty is not scale invariant — measuring a feature in metres versus kilometres changes how hard it gets shrunk.

> **Why the interviewer asks this.** It tests whether you can carry a derivation through with a second term added, and whether you understand $\lambda$ as a signal-to-prior ratio rather than a hyperparameter with no meaning.

> **Saying it out loud.** I add a zero-mean Gaussian prior on the weights, so the log-posterior is the Gaussian log-likelihood minus a term proportional to the squared norm of w. Flip the sign and you're minimizing squared error plus lambda times w-squared — that's ridge, and lambda comes out as the noise variance over the prior variance, which is a nice interpretation: noisier data or a tighter prior means more shrinkage. Take the gradient, set to zero, and you get X-transpose-X plus lambda-I, inverse, times X-transpose y. The lambda-I is also why ridge is numerically safe — it bumps every eigenvalue up, so the matrix is invertible even when the features are collinear.

---

### Q50: Why do we use log-likelihood instead of likelihood?

**Answer:**

**Reasons:**
1. **Numerical stability**: Products of small probabilities → underflow, sums are stable
2. **Mathematical convenience**: Products become sums, derivatives easier
3. **Monotonicity**: Maximizing log L(θ) = maximizing L(θ)
4. **Additive properties**: Can combine log-likelihoods easily

**Example:**
- Likelihood: 0.1 × 0.1 × 0.1 = 0.001 (very small!)
- Log-likelihood: log(0.1) + log(0.1) + log(0.1) ≈ -6.91 (manageable)

**How bad is the underflow, exactly?**

This is not a theoretical worry. A standard 64-bit float underflows to exactly zero below roughly $10^{-308}$. A language model scoring a 1,000-token document at an average per-token probability of $0.05$ produces a likelihood of $0.05^{1000} \approx 10^{-1301}$ — the product is zero in floating point long before you finish, and once it is zero every gradient is zero and the run is dead. The log-likelihood of the same document is $1000 \times \log 0.05 \approx -3000$, which is a perfectly ordinary number. Any time you see a sum of log-probabilities in a codebase, this is why.

**The other three reasons, spelled out.**

*Products become sums, and sums differentiate independently.* The derivative of a product of $n$ terms needs the product rule and produces $n$ terms each containing all the others; the derivative of a sum is just the sum of derivatives. For an i.i.d. dataset this is the difference between a tractable gradient and an unusable one.

*The maximizer is unchanged because $\log$ is strictly increasing.* If $L(\theta_1) > L(\theta_2)$ then $\log L(\theta_1) > \log L(\theta_2)$, so the ordering of every pair of candidate parameters is preserved and therefore so is the argmax. Note this is about the *location* of the maximum: the maximum *value* obviously changes.

*Concavity often appears only after the log.* The likelihood surface for many standard models is not concave, but the log-likelihood is — Bernoulli, Gaussian, Poisson and the rest of the exponential family all have concave log-likelihoods in their natural parameters. Concavity is what guarantees a unique optimum and makes gradient ascent reliable, so taking the log can turn a hard optimization into an easy one.

**The trick you will actually reach for: log-sum-exp.** When you need $\log \sum_i \exp(z_i)$ — which is what a softmax denominator is — computing the exponentials directly overflows for large $z_i$. The stable form subtracts the maximum first:

$$\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i - m}, \qquad m = \max_i z_i .$$

Every exponent is now $\le 0$, so every term is in $(0,1]$ and at least one equals exactly $1$; nothing overflows and the largest term cannot underflow. This identity is what `torch.logsumexp` and every framework's `log_softmax` implement, and it is why you should call `log_softmax` rather than `log(softmax(x))`.

**Follow-up:** *Does taking the log change the estimate's variance or bias?* No — it is the same estimator, since it is the same argmax. What changes is the numerical path you take to find it.

> **Saying it out loud.** Three reasons, really. Numerically, likelihoods are products of thousands of numbers below one, so they underflow to zero in floating point — a thousand tokens at probability 0.05 gives you ten to the minus thirteen hundred, which is just zero to a computer, whereas the log is about minus three thousand and totally fine. Mathematically, log turns products into sums, so gradients decompose per data point. And it's free, because log is monotonic, so the maximizer doesn't move. The same idea shows up in log-sum-exp, where you subtract the max before exponentiating so nothing overflows.

---

### Q51: What's the difference between MLE and MAP in practice?

**Answer:**

**MLE:**
- Frequentist approach
- No prior (or uniform)
- Use: Large dataset, no prior knowledge
- Example: θ̂ = k/n

**MAP:**
- Bayesian approach
- Informative prior
- Use: Small dataset, have prior knowledge, need regularization
- Example: θ̂ = (k+α-1)/(n+α+β-2) with Beta prior

**Practical:**
- **Small data**: MLE can be extreme, MAP more reasonable
- **Regularization**: MAP provides natural regularization
- **Computation**: Similar complexity

**A worked Bernoulli/Beta example, end to end.**

This is the example to have ready, because it makes every abstract claim above concrete in about thirty seconds of arithmetic.

The *Beta distribution* is a distribution over a probability — a density on the interval $[0,1]$ — with two shape parameters $\alpha,\beta>0$ and density $p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$. It is the *conjugate prior* for the Bernoulli likelihood, meaning that if the prior is Beta then the posterior is Beta too, so the update is arithmetic rather than integration. Multiply prior by likelihood:

$$p(\theta \mid D) \;\propto\; \underbrace{\theta^{k}(1-\theta)^{n-k}}_{\text{likelihood}} \cdot \underbrace{\theta^{\alpha-1}(1-\theta)^{\beta-1}}_{\text{prior}} \;=\; \theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1},$$

which is $\text{Beta}(k+\alpha,\; n-k+\beta)$ by inspection. The update rule is therefore "add your heads to $\alpha$ and your tails to $\beta$," which is why $\alpha$ and $\beta$ are read as *pseudo-counts*: a $\text{Beta}(3,3)$ prior behaves like having already seen 2 heads and 2 tails before the experiment started (the "minus one" convention comes from the exponents).

MAP takes the *mode* of that posterior. Differentiating $\log p(\theta\mid D) = (k+\alpha-1)\log\theta + (n-k+\beta-1)\log(1-\theta)$ and setting to zero repeats the Q46 algebra exactly, giving

$$\hat\theta_{\text{MAP}} = \frac{k+\alpha-1}{n+\alpha+\beta-2}, \qquad \mathbb{E}[\theta\mid D] = \frac{k+\alpha}{n+\alpha+\beta}.$$

Note that the mode and the posterior mean are *different numbers*; MAP is the mode. Now the arithmetic, with a $\text{Beta}(3,3)$ prior — symmetric, centred on $0.5$, mildly confident:

| Data | MLE $k/n$ | MAP (mode) | Posterior mean |
|---|---|---|---|
| 3 heads in 10 | 0.300 | 0.357 | 0.375 |
| 3 heads in 3 | **1.000** | 0.714 | 0.667 |
| 30 heads in 100 | 0.300 | 0.308 | 0.311 |

Read the three rows in order and you have the whole lesson. In row one the prior pulls the estimate a little toward $0.5$. In row two MLE gives the absurd answer that tails is impossible, while MAP gives a cautious $0.71$ — this is the failure mode that makes MAP worth the trouble, and it is the same failure that add-one (Laplace) smoothing fixes in n-gram language models and naive Bayes, which is precisely a MAP estimate under a $\text{Beta}(2,2)$ / Dirichlet prior. In row three the data has ten times the weight and the two estimators have nearly converged, which is the $1/n$ washout from Q48 showing up numerically.

*(Verified numerically: with $\alpha=\beta=3$, $(3+2)/(10+4)=0.357$ and $(3+2)/(3+4)=0.714$.)*

**One more practical difference worth naming.** MLE has an asymptotic guarantee — under regularity conditions it is consistent and asymptotically efficient, meaning it attains the lowest possible variance for large $n$. MAP trades a little bias for a large variance reduction on small $n$. That is the bias–variance tradeoff appearing in an estimation-theory costume, and saying it that way tends to land well.

**Follow-up:** *How do you choose $\alpha$ and $\beta$ in practice?* Either from domain knowledge expressed as pseudo-counts ("I'd be surprised by a click-through rate outside 1–5%, so pick a Beta with that bulk"), or empirically by fitting the prior to the pooled distribution across many similar items — that is empirical Bayes, and it is how per-item conversion rates get shrunk toward the population rate in production ranking systems.

> **Why the interviewer asks this.** The interesting answer is not the definitions but the small-sample failure of MLE, so this question is really asking whether you have ever been burned by an estimate computed from four data points.

> **Saying it out loud.** Practically, MLE is the pure data answer and MAP nudges it toward a prior. The example I like: flip a coin three times, get three heads, and MLE says the probability of tails is exactly zero — which is nonsense, and it'll blow up anything downstream that takes a log. Put a Beta-three-three prior on it and MAP says about 0.71, which is a sane answer. With a hundred flips the two agree to within a percent, because the likelihood grows with n and the prior doesn't. It's the same idea as add-one smoothing in n-gram models — that's literally a MAP estimate.

---

See `37_mle_map_estimation/mle_map_derivations.md` for complete derivations!
See `37_mle_map_estimation/interview_qa.md` for more detailed answers!

---

## Multimodal Models and Embeddings

### Q52: Explain CLIP. How does it work?

**Answer:**

**CLIP** (Contrastive Language-Image Pre-training) learns to align text and images in a shared embedding space.

**Architecture:**
- **Image Encoder**: ViT or ResNet → image embeddings
- **Text Encoder**: Transformer → text embeddings
- **Contrastive Learning**: Align matching pairs

**Training:**
1. Collect 400M text-image pairs from web
2. Encode images and texts to same space
3. Contrastive loss: maximize similarity of matching pairs, minimize non-matching
4. Large batch size (32K) for many negatives

**Key Insight:**
Instead of predicting exact labels, predict which text matches which image.

**Zero-Shot Transfer:**
- Create text prompts: "a photo of a cat"
- Find most similar image
- Works on new tasks without fine-tuning!

**Results:**
- Matches supervised models on many tasks
- More robust to distribution shifts
- Strong image-text retrieval

**The mechanism, in more detail.**

*Contrastive learning* means training on relative comparisons rather than absolute labels: the model is never told "this is a cat," only "this image goes with this caption and not with those other 32,767 captions." Here is how one CLIP training step actually runs.

Take a batch of $N$ image–caption pairs. Encode all images into vectors $\{I_1,\dots,I_N\}$ and all captions into $\{T_1,\dots,T_N\}$, project both into a shared dimension, and **L2-normalize** every vector so it lies on the unit sphere. Normalization matters: once vectors have unit length, the dot product *is* the cosine similarity, so the loss cannot be gamed by simply making embeddings longer.

Now form the $N \times N$ matrix $S_{ij} = (I_i \cdot T_j)/\tau$, where $\tau$ is a *temperature* that controls how sharp the resulting softmax is. In CLIP $\tau$ is learned rather than fixed — the model is parameterized by $\log(1/\tau)$ and it is clipped to stop it running away. The diagonal entries are the true pairs. Apply a cross-entropy loss across each row (given image $i$, which caption is right?) and across each column (given caption $j$, which image is right?), and average the two. In code that is two `cross_entropy(logits, arange(N))` calls, one on the matrix and one on its transpose — it is genuinely about five lines.

**Why the batch size is part of the algorithm, not a tuning detail.** Every non-diagonal entry is a negative example, so a batch of $N$ supplies $N-1$ negatives per anchor for free. The difficulty of the task — and hence how much signal each step carries — scales with $N$. At 32,768 the model must pick the right caption out of tens of thousands, which forces genuinely fine-grained representations. This is also why CLIP-style training needs either very large accelerators or distributed tricks that gather embeddings across devices before computing the loss.

**Why zero-shot transfer works at all.** Classification is recast as retrieval. You never train a classifier head; you write one sentence per class ("a photo of a {label}"), encode those sentences once, and label an image by nearest caption. The class set is therefore just a list of strings you can change at inference time. *Prompt ensembling* — averaging the embeddings of several templates per class — reliably adds a point or two, because it averages out the quirks of any single phrasing.

**What CLIP is bad at, which is the follow-up you should expect.** It is weak at counting, at spatial relations ("the cup left of the laptop"), and at binding attributes to the right object ("a red cube and a blue sphere" versus the swap) — a well-documented bag-of-words tendency, because a contrastive objective over web captions rarely needs compositional structure to pick the right caption. It also inherits web-scale social biases, and its zero-shot accuracy is sensitive to prompt wording. Contrastive image-text pretraining remains the backbone of most vision-language systems as of 2026, though production systems now typically add a generative captioning or matching objective on top rather than using the contrastive loss alone — *this is a fast-moving area; check the current state of the art before quoting specifics.*

**Follow-up:** *Why not just train a supervised classifier on 400M images?* Because you would first need 400M consistent labels from a fixed taxonomy, which does not exist and would not transfer. Natural-language supervision is both cheaper to collect and richer — the caption "a chest X-ray showing pneumonia" carries structure that the integer class ID 37 does not.

> **Why the interviewer asks this.** CLIP is the cleanest example of turning supervision into a retrieval problem, so the question tests whether you understand contrastive objectives rather than whether you have memorised an architecture diagram.

> **Saying it out loud.** CLIP trains two encoders, one for images and one for text, so that matching pairs land close together in a shared space. You take a batch of image-caption pairs, encode everything, normalize to unit length so dot product is cosine similarity, and build an N-by-N similarity matrix. The diagonal is the true pairs, everything off-diagonal is a negative, and you just do cross-entropy across rows and across columns. Big batches matter because they're where the negatives come from. The payoff is zero-shot classification — you turn class names into sentences like "a photo of a cat," embed them, and pick the nearest one, so you can change your label set at inference time without retraining.

---

### Q53: How do you train Word2Vec?

**Answer:**

**Word2Vec Skip-gram:**

**Architecture:**
- Input: One-hot vector for center word
- Hidden: Embedding layer (V × d)
- Output: Softmax over vocabulary (predict context words)

**Training:**
1. **Create pairs**: For each word, create (center, context) pairs from window
2. **Forward pass**: Embed center word, predict context words
3. **Loss**: -log P(context | center)
4. **Negative sampling**: Instead of softmax over all V words, sample k negatives
   - Binary classification: positive (context) vs negative
   - Much faster!

**Loss with Negative Sampling:**
```
Loss = -log σ(v_context · v_center) - Σ log σ(-v_neg · v_center)
```

**Training Details:**
- Data: Billions of words
- Window size: 5-10 words
- Embedding dim: 100-300
- Negative samples: 5-20
- Training: Hours to days

**Result:**
- Dense, low-dimensional embeddings
- Captures semantic relationships
- "King - Man + Woman ≈ Queen"

**Walkthrough of the actual computation.**

The one-hot-times-matrix framing in the answer above is how the paper draws it, but it is worth knowing what really happens in code: multiplying a one-hot vector by an embedding matrix is just a row lookup, so the "input layer" is an array index, not a matrix multiply. There are two embedding tables — an *input* (centre) table $V$ and an *output* (context) table $U$, both of shape (vocabulary size $\times$ dimension). Most implementations keep $V$ at the end and discard $U$, though summing or concatenating the two sometimes helps.

**Step 1 — build the training pairs.** Slide a window over the corpus. For "the quick brown fox jumps" with window 2 and centre "brown," you emit (brown, the), (brown, quick), (brown, fox), (brown, jumps). Word2vec actually samples the window size uniformly from $1$ to the maximum for each centre word, which has the effect of weighting nearby words more heavily without any extra machinery.

**Step 2 — subsample frequent words.** Before pair generation, each token is discarded with probability $1 - \sqrt{t/f(w)}$ where $f(w)$ is the word's corpus frequency and $t \approx 10^{-5}$. This throws away most instances of "the" and "of," which both speeds training and improves quality, because a co-occurrence with "the" carries almost no information.

**Step 3 — the negative sampling loss.** The full softmax over a vocabulary of, say, 1M words costs a 1M-way normalization per pair, which is fatal. *Negative sampling* replaces "which of the 1M words is the context?" with "is this pair real or fake?" — a set of binary logistic regressions. For one true pair $(c,o)$ and $k$ sampled fake contexts $n_1..n_k$:

$$\mathcal{L} = -\log\sigma(u_o^{\top} v_c) - \sum_{j=1}^{k}\log\sigma(-u_{n_j}^{\top} v_c).$$

Reading it in words: push the dot product of the true pair up (first term), push the dot products of $k$ random pairs down (second term). Cost per example drops from $O(V)$ to $O(k)$ with $k$ of 5–20.

**Step 4 — sample the negatives from a flattened unigram distribution.** Negatives are drawn with probability proportional to $f(w)^{3/4}$, not $f(w)$. Raising to the three-quarter power flattens the distribution, so rare words get sampled somewhat more often than their raw frequency would suggest and common words somewhat less. It is an empirical choice that measurably helps.

**Step 5 — update.** Plain SGD with a linearly decaying learning rate; each example touches only the centre row of $V$ and $k+1$ rows of $U$, so updates are extremely sparse and cheap.

**Why the analogy trick works, mechanically.** "King − Man + Woman ≈ Queen" is not magic. The objective drives $v_c \cdot u_o$ toward roughly the pointwise mutual information between the two words, so differences of vectors encode ratios of co-occurrence probabilities. The vector "king minus man" isolates whatever co-occurrence pattern distinguishes royalty-with-male-context from male-context, and adding "woman" re-applies it. Worth mentioning the caveats too: the standard evaluation excludes the three input words from the nearest-neighbour search, and if you do not exclude them the answer is frequently just "king" again. That detail signals you have actually run the code.

**Follow-up:** *Skip-gram or CBOW?* CBOW predicts the centre word from the averaged context, is several times faster, and is better on frequent words. Skip-gram predicts each context word from the centre, sees each rare word many times as an anchor, and is better on small corpora and rare words. Skip-gram with negative sampling is the usual default.

> **Saying it out loud.** Skip-gram slides a window over the corpus and, for each centre word, tries to predict the words around it. The catch is that a softmax over a million-word vocabulary per training pair is way too expensive, so instead you use negative sampling — reframe it as a binary question: is this centre-context pair real, or did I make it up? You push the dot product up for the real pair and down for maybe five to twenty random fake ones, and now each update costs almost nothing. There are a couple of tricks that matter in practice: subsampling frequent words like "the," and drawing negatives from the frequency distribution raised to the three-quarters power so rare words show up a bit more.

---

### Q54: How does GloVe differ from Word2Vec?

**Answer:**

**Word2Vec:**
- Uses local context (windows)
- Predicts context from center word
- Local statistics

**GloVe:**
- Uses global co-occurrence matrix
- Preserves co-occurrence ratios
- Global statistics

**GloVe Objective:**
```
w_i · w_j + b_i + b_j ≈ log(X_ij)

Where X_ij = co-occurrence count
```

**Training:**
1. Build co-occurrence matrix from entire corpus
2. Weighted least squares to preserve ratios
3. More efficient than Word2Vec

**Key Insight:**
Preserves ratios: P(solid|ice) / P(solid|steam) ≈ P(gas|ice) / P(gas|steam)

**Comparison:**
- **Word2Vec**: Local, window-based
- **GloVe**: Global, matrix-based
- **Performance**: Often similar, GloVe sometimes better

**What the GloVe objective actually is, and where it comes from.**

The line $w_i \cdot w_j + b_i + b_j \approx \log X_{ij}$ is the model; the loss that fits it is a *weighted* least-squares problem:

$$J = \sum_{i,j:\,X_{ij}>0} f(X_{ij})\left(w_i^{\top}\tilde w_j + b_i + \tilde b_j - \log X_{ij}\right)^{2},$$

with the weighting function

$$f(x) = \begin{cases}(x/x_{\max})^{\alpha} & x < x_{\max}\\ 1 & \text{otherwise}\end{cases}, \qquad x_{\max}=100,\ \alpha=0.75 .$$

Three details in there each solve a specific problem. The sum runs only over non-zero entries, so the cost is proportional to the number of observed co-occurrences rather than $V^{2}$ — the matrix is extremely sparse. The weight $f$ rising from zero stops rare, noisy co-occurrences from dominating a squared loss, and its cap at $x_{\max}$ stops "the" from dominating instead. And the two bias terms absorb each word's overall frequency, so the dot product is left to model the *interaction* rather than the marginals. There are separate centre and context vectors $w$ and $\tilde w$; the released vectors are their sum, which averages out initialization noise.

**Why ratios, and why that forces a log.** The paper's starting observation is that raw co-occurrence probabilities are less meaningful than their ratios. $P(\text{solid}\mid\text{ice})/P(\text{solid}\mid\text{steam})$ is large, the same ratio for "gas" is small, and for a word related to both ("water") or neither ("fashion") it is near one — so the ratio is what isolates meaning. If you want vector differences to correspond to ratios, i.e. $F(w_i - w_j, \tilde w_k) = P_{ik}/P_{jk}$, and you want $F$ to turn a difference of vectors into a ratio of scalars, the exponential is essentially forced, which inverts to the log form above. Bringing this up shows you know GloVe is derived rather than guessed.

**The honest modern verdict.** The word2vec/GloVe distinction matters much less than either camp claimed: Levy and Goldberg showed skip-gram with negative sampling is implicitly factorizing a shifted PMI matrix, so both methods are matrix factorizations of co-occurrence statistics with different weightings and different optimizers. Tuned carefully, they land within noise of each other on most benchmarks. Both are also *static* embeddings — one vector per word type, so "bank" gets a single vector blending the river and the money sense — which is the limitation that contextual models removed in 2018 and the reason neither is a first choice for new work today.

**Follow-up:** *Which trains faster?* GloVe's per-epoch cost is low because it works over the pre-aggregated co-occurrence matrix, but building that matrix is a full corpus pass with substantial memory. Word2vec streams the corpus with almost no memory and parallelizes trivially via asynchronous SGD. On a very large corpus word2vec is usually the more practical choice.

> **Why the interviewer asks this.** They want to hear "global versus local statistics," but the answer that stands out explains that both are ultimately factorizing the same co-occurrence information.

> **Saying it out loud.** Word2vec is local — it slides a window and learns from individual context pairs. GloVe first builds the whole co-occurrence matrix, then fits vectors so that the dot product plus two bias terms matches the log of the co-occurrence count, using a weighted least-squares loss that downweights both very rare and very frequent pairs. The motivating idea is that ratios of co-occurrence probabilities are what carry meaning. In practice they perform about the same, and there's a nice result showing skip-gram is implicitly factorizing a shifted PMI matrix, so they're closer relatives than they look. Both are static — one vector per word regardless of context — which is exactly what BERT and friends fixed.

---

### Q55: Explain the evolution of NLP embeddings.

**Answer:**

**Timeline:**

**1. TF-IDF (1970s):**
- Statistical weighting
- Sparse, high-dimensional
- No semantic understanding

**2. N-grams (1980s-1990s):**
- Sequence modeling
- Count-based probabilities
- Local context only

**3. Word2Vec (2013):**
- Neural embeddings
- Dense, low-dimensional
- Semantic relationships
- Fixed embeddings (no context)

**4. GloVe (2014):**
- Global co-occurrence
- Matrix factorization
- Better than Word2Vec on some tasks

**5. Contextual Embeddings (2018+):**
- **ELMo**: Bidirectional LSTM
- **BERT**: Transformer, bidirectional
- **GPT**: Transformer, unidirectional
- Context-dependent embeddings

**6. Modern LLMs (2020+):**
- Large-scale language models
- Multimodal (CLIP, GPT-4V)
- Instruction tuning, RLHF

**Key Evolution:**
- Sparse → Dense
- Local → Global → Contextual
- Fixed → Context-dependent
- Single modality → Multimodal

**The thread that connects the whole timeline.**

The one sentence that ties these six stages together is the *distributional hypothesis*: a word's meaning is characterized by the company it keeps (Firth, 1957). Every method on this list is a different answer to "how do I compress a word's co-occurrence statistics?" — TF-IDF stores them raw and sparse, word2vec and GloVe factorize them into a few hundred dimensions, and contextual models compute them on the fly per occurrence. Saying that out loud reframes a list of names into a single idea with a progression, which is what the question is really after.

**What each transition actually bought you.**

*Sparse to dense (TF-IDF → word2vec).* A TF-IDF vector has one dimension per vocabulary word and is almost entirely zeros, so "car" and "automobile" are exactly orthogonal — cosine similarity zero, no matter how interchangeably they are used. Dense embeddings put them in nearby directions because they appear in similar contexts. The cost is interpretability: a TF-IDF dimension means "the word *insulin*," a word2vec dimension means nothing you can name.

*Static to contextual (word2vec → ELMo/BERT).* A static embedding must average all senses of a word into one vector, so "bank" sits somewhere between rivers and money and is a good representation of neither. Contextual models run the whole sentence through the network and emit a different vector for each occurrence. ELMo did this with stacked bidirectional LSTMs, concatenating a left-to-right and a right-to-left pass; BERT did it with a Transformer trained by *masked language modelling* — hide 15% of tokens and predict them from both sides at once, which gives genuinely joint bidirectional conditioning rather than two independently-trained directions stitched together.

*Contextual to generative-and-general (BERT → GPT-family).* BERT gives you good representations that still need a task-specific head and a fine-tuning run. Decoder-only models fold the task itself into the input as text, so one frozen model handles many tasks.

**One correction worth making to the timeline.** TF-IDF and n-grams are listed as if superseded, and they are not. BM25, the direct descendant of TF-IDF, is still a competitive retrieval baseline in 2026 and is half of every serious hybrid search stack (Q68–Q70). The honest framing is "added to the toolbox," not "replaced."

**Where embeddings sit today.** Modern text embedding models are Transformer encoders — often initialised from a decoder-only LLM — trained contrastively on hundreds of millions of query–document pairs with hard negatives, typically producing 384 to 3072 dimensions, and frequently supporting *Matryoshka* representations where you can truncate the vector to a shorter prefix and keep most of the quality, which is a real cost lever when you are storing hundreds of millions of vectors. *This layer of the stack turns over every few months; treat any specific model name or leaderboard position as a snapshot and verify before quoting it.*

> **Saying it out loud.** The thread running through all of it is the distributional hypothesis — you know a word by the company it keeps — and each generation is a better way to compress that co-occurrence information. TF-IDF stores it raw and sparse, so "car" and "automobile" are literally orthogonal. Word2vec and GloVe squash it into a few hundred dense dimensions so similar words land near each other. Then ELMo and BERT made it contextual, so "bank" gets a different vector in a river sentence than in a finance sentence. And now embeddings are contrastively trained Transformer encoders. The one thing I'd push back on is calling the old stuff obsolete — BM25 is a direct TF-IDF descendant and it's still in production everywhere.

---

### Q56: How do you evaluate multimodal models?

**Answer:**

**1. Zero-Shot Image Classification:**
- Create text prompts for classes
- Find most similar prompt
- Measure accuracy

**2. Image-Text Retrieval:**
- Text → Image: Find images matching query
- Image → Text: Find captions matching image
- Metrics: Recall@K, Median Rank

**3. Visual Question Answering:**
- Answer questions about images
- Requires reasoning
- Accuracy by question type

**4. Robustness:**
- Distribution shifts
- Adversarial examples
- Natural variations

**5. Bias Evaluation:**
- Gender, racial, cultural biases
- Fairness across groups

**6. Few-Shot Learning:**
- Learn from few examples
- Transfer learning capability

**Best Practices:**
- Multiple metrics
- Diverse datasets
- Human evaluation when possible
- Error analysis
- Bias testing

**Making the metrics concrete.**

*Recall@K* is the fraction of queries for which the correct item appears anywhere in the top $K$ results. For image–text retrieval the convention is to report R@1, R@5 and R@10 in both directions, because a model can be strong at text-to-image and weak at image-to-text and a single number hides that. *Median rank* is the median position of the correct item across queries; it is more robust to a handful of catastrophic queries than the mean rank, which a single result at rank 4000 can destroy.

**The evaluation trap specific to multimodal models: contamination.** These models are pretrained on hundreds of millions of web image–text pairs, and the standard benchmarks — COCO, Flickr30k, and the popular VQA sets — are also on the web. "Zero-shot" is only zero-shot with respect to the *task format*, not necessarily with respect to the *data*. Bringing this up unprompted is a strong signal. The mitigations are near-duplicate detection between the pretraining corpus and the eval set (perceptual hashing or embedding-space nearest neighbours), and evaluating on held-out sets built after the pretraining cutoff.

**Two more axes the list above misses.**

*Prompt sensitivity.* A CLIP-style model's zero-shot accuracy can swing several points depending on whether you write "cat," "a photo of a cat," or "a photo of a cat, a type of pet." A single-number accuracy therefore does not describe the model, it describes the model plus one prompt. Report a mean and spread over a template set, or use prompt ensembling and say so.

*Hallucination and grounding for generative multimodal models.* For a model that produces text about an image, retrieval metrics do not apply at all. What you want is whether the described objects are actually present — object-hallucination probes such as asking yes/no existence questions about objects that are and are not in the image — and whether claims are attributable to image regions. This is a different failure mode from a classification error and needs its own measurement.

**A practical evaluation protocol.** Pick one held-out retrieval set for representation quality, one classification set for zero-shot transfer, one adversarial or distribution-shifted set for robustness, and a small human-rated set of a few hundred examples for anything generative. Freeze the prompts, version the eval set, and track all of it over time; the failure to catch in production is a slow drift, and you cannot see drift without a fixed yardstick.

**Follow-up:** *Your zero-shot accuracy dropped after a model update but retrieval R@1 improved. What now?* Usually a prompt-distribution mismatch rather than a genuine regression: the new model's text encoder responds differently to your class templates. Re-tune the templates or use prompt ensembling before concluding the model got worse.

> **Why the interviewer asks this.** Evaluation is where multimodal projects quietly fail, so this question separates people who have shipped one from people who have read the papers.

> **Saying it out loud.** I'd split it into three buckets. Retrieval quality — Recall at 1, 5 and 10 in both directions, image-to-text and text-to-image, plus median rank, because a model can be lopsided. Downstream transfer — zero-shot classification on held-out sets, VQA if it's generative. And robustness and fairness — distribution shift, and bias across demographic groups. The thing I'd raise unprompted is contamination: these models are trained on web-scale image-text data and the benchmarks are on the web too, so "zero-shot" means zero-shot on the task format, not necessarily on the data. I'd also report prompt sensitivity, because zero-shot numbers move several points just from rewording the template.

---

See `38_multimodal_and_embeddings/` for detailed explanations!

---

## RAG (Retrieval-Augmented Generation)

### Q57: Design a RAG system. What are the key components?

**Answer:**

**RAG Components:**

**1. Document Ingestion:**
- Load documents (PDF, DOCX, HTML, etc.)
- Extract text, metadata
- Preprocess and clean

**2. Chunking:**
- Split documents into chunks
- Strategies: fixed-size, sentence-based, semantic
- Overlap between chunks (10-20%)

**3. Embedding Generation:**
- Generate embeddings for chunks
- Use embedding models (sentence-transformers, OpenAI)
- Store in vector database

**4. Query Processing:**
- Process user query
- Generate query embedding
- Query expansion/rewriting

**5. Retrieval:**
- Vector similarity search
- Hybrid search (dense + sparse)
- Metadata filtering
- Top-K retrieval

**6. Re-ranking (Optional):**
- Cross-encoder for accuracy
- Re-rank top results
- Better precision

**7. Context Assembly:**
- Select top-K chunks
- Order by relevance
- Fit in context window

**8. Generation:**
- LLM with context
- Prompt engineering
- Generate answer

**9. Post-processing:**
- Extract answer
- Generate citations
- Validate answer

**Pipeline:**
```
Query → Embedding → Retrieval → Re-ranking → Context → Generation → Answer
```

**Walkthrough: what happens to one query, end to end.**

The component list above is the *what*; here is the *when*, which is what a design interview is actually probing. The system has two loops that run on completely different clocks, and saying so early is the single best structural move you can make.

**The offline loop (indexing), run on ingest or on a schedule.** Documents arrive; you parse them (PDFs are the hard case — layout, tables, and scanned pages all need different handling); you chunk them (Q63); you embed each chunk with an *embedding model*, which is a network mapping text to a fixed-length vector such that semantically similar text lands nearby; you write vectors plus the original text plus metadata into a vector index. The metadata is not optional — chunk ID, source document, section, page, timestamp, and access-control tags all need to travel with the vector, because filtering and citation both depend on them later, and retrofitting them means a full re-index.

**The online loop (serving), run per query in a few hundred milliseconds.** Rewrite the query if needed (resolving "what about last year?" against conversation history is a real and commonly-skipped step); embed it with *the same model* used for indexing — different models produce incompatible spaces, and mixing them silently returns garbage rather than erroring; run an approximate nearest neighbour search for the top 50–100 candidates alongside a BM25 keyword search; fuse the two lists; re-rank the fused top ~50 with a cross-encoder down to the top 5; assemble those into a prompt with explicit source markers; call the LLM; then post-process to attach citations and check for unsupported claims.

**Why the two-stage retrieve-then-rerank split exists.** A *bi-encoder* embeds query and document separately, so document vectors can be computed once, offline, and searched in milliseconds — but the query never gets to interact with the document text, so precision suffers. A *cross-encoder* concatenates query and document and runs them jointly through a Transformer, which is far more accurate because every query token can attend to every document token, but it costs a full forward pass *per candidate* and cannot be precomputed. Scoring a million documents with a cross-encoder is impossible; scoring fifty is a few tens of milliseconds. So you use the cheap model for recall and the expensive one for precision. That is the entire architectural argument, and it generalizes to almost every large-scale ranking system.

**A latency budget to have in your pocket.** Roughly: query embedding 5–20 ms, ANN search 5–50 ms depending on index and filter selectivity, cross-encoder re-rank of 50 candidates 20–100 ms on a GPU, and then generation, which dominates everything at hundreds of milliseconds to several seconds and scales with output length, not input length. The consequence worth stating: if you need to cut latency, cut *output tokens* first — trimming retrieval is optimizing the wrong term.

**Cost, and where it actually goes.** Embedding a corpus is a one-time cost and is cheap: at roughly \$0.02 per million tokens for a small commercial embedding model as of August 2026, a 100-million-token corpus costs about \$2 to embed once. Generation is the recurring cost and is orders of magnitude larger, because you pay per query and every retrieved chunk is input tokens you are billed for. Vector storage is the third line item and is easy to underestimate: one million chunks at 1536 dimensions in float32 is about 6 GB of raw vectors before index overhead, which is why dimension reduction, product quantization, or Matryoshka truncation to 256–512 dimensions is standard at scale. *All prices here are time-sensitive — verify current rates before using them in a real design document.*

**Follow-up:** *How do you handle document updates and deletions?* Key chunks by a stable `(document_id, chunk_index)` and store a content hash. On re-ingest, re-embed only chunks whose hash changed, and delete by `document_id` prefix. Also check that your vector store supports real deletes rather than tombstones only — some index types (HNSW in particular) mark deleted entries and need periodic compaction, and a system that never compacts slowly fills with ghosts.

> **Why the interviewer asks this.** It is a system-design question wearing an ML costume; they are watching for whether you separate offline indexing from online serving and whether you think about staleness, cost, and failure modes rather than reciting a pipeline diagram.

> **Saying it out loud.** I'd split it into two loops that run on totally different clocks. Offline, you parse documents, chunk them, embed each chunk, and store the vectors with metadata like source, section, and access tags — metadata you'll wish you had later for filtering and citations. Online, per query, you embed the query with the same model, do a fast approximate nearest-neighbour search plus a BM25 keyword search, fuse the two lists, re-rank the top fifty with a cross-encoder down to about five, stuff those into the prompt with source markers, and generate. The reason for the two retrieval stages is that a bi-encoder is cheap and can be precomputed but imprecise, while a cross-encoder is accurate but costs a forward pass per candidate — so cheap for recall, expensive for precision.

---

### Q58: How do you improve RAG retrieval accuracy?

**Answer:**

**1. Better Chunking:**
- Semantic chunking (respect boundaries)
- Hierarchical chunking
- Multi-granularity chunks
- Overlapping chunks

**2. Better Embeddings:**
- Domain fine-tuning
- Hybrid embeddings (dense + sparse)
- Multi-vector embeddings
- Query-specific embeddings

**3. Hybrid Search:**
- Dense retrieval (semantic)
- Sparse retrieval (BM25, keywords)
- Weighted combination
- Better coverage

**4. Re-ranking:**
- Cross-encoder (more accurate)
- Learning-to-rank
- Multi-stage retrieval
- Better precision

**5. Query Expansion:**
- Synonym expansion
- Related terms
- Query rewriting
- Multi-query generation

**6. Metadata Filtering:**
- Filter by document type
- Filter by date, source
- Improve precision

**7. Multi-Stage Retrieval:**
- Stage 1: Coarse (ANN, top-100)
- Stage 2: Re-rank (top-10)
- Stage 3: Fine-grained (top-5)

**How to actually do this, in priority order.**

The list above is a menu; an interviewer wants a method. The method is: measure first, because retrieval failures split into two kinds with opposite fixes, and treating them the same is how teams waste a quarter.

**Step 1 — separate recall failures from ranking failures.** Build a small evaluation set of 100–200 real queries with the correct chunk labelled. Then for each query check whether the right chunk is anywhere in the top 100. If it is not, you have a *recall* problem and no amount of re-ranking will help — re-ranking can only reorder what retrieval already found. If it is in the top 100 but not the top 5, you have a *ranking* problem and a cross-encoder will likely fix it in an afternoon. This single diagnostic is the highest-value thing in this answer.

**Step 2 — fix recall problems at the source.** Recall failures usually trace to one of four causes. Chunks are too small, so the answer is split across a boundary — fix with larger chunks or overlap. Chunks are too large, so the relevant sentence is diluted by 900 irrelevant tokens and the embedding drifts — fix with smaller chunks or by embedding a summary while retrieving the full chunk. Vocabulary mismatch, where the user says "cannot log in" and the document says "authentication failure" — fix with hybrid search so the dense side handles paraphrase, or with query expansion. Or the information genuinely is not in the corpus, which is an ingestion bug, not a retrieval one, and is worth ruling out before anything else.

**Step 3 — add a cross-encoder re-ranker.** This is reliably the best effort-to-improvement ratio in the whole list. Retrieve 50–100 candidates cheaply, score each jointly with the query, keep the top 5. Expect a solid jump in precision@5 for tens of milliseconds of GPU time.

**Two techniques worth naming explicitly, since they come up constantly.**

*HyDE* (Hypothetical Document Embeddings) has the LLM write a plausible fake answer to the query, then embeds *that* and searches with it. It works because a query and its answer often look nothing alike in embedding space — "how do I reset my password?" versus a paragraph of instructions — whereas a hypothetical answer looks a lot like a real one. It costs an extra LLM call, so it suits low-QPS, high-value queries.

*Multi-query and RAG-fusion.* Have the LLM produce three or four paraphrases of the query, retrieve for each, and fuse the result lists with Reciprocal Rank Fusion (see Q69). This covers vocabulary variation cheaply and is one of the few techniques that helps recall rather than just ranking.

*Parent-document / small-to-big retrieval* deserves a mention too: index small, precise chunks so matching is sharp, but return the surrounding parent section to the LLM so the context is complete. It resolves the "small chunks retrieve better, large chunks answer better" tension directly, and it is what most mature production systems end up doing.

**Follow-up:** *When is fine-tuning the embedding model worth it?* When your domain vocabulary is genuinely unlike the pretraining distribution — legal citations, clinical abbreviations, internal product codenames — and you have or can mine a few thousand query–positive pairs, for instance from click logs or by having an LLM generate questions from each chunk. Below a few thousand pairs you will usually get more from a re-ranker. Also budget for the fact that changing the embedding model means re-embedding the entire corpus.

> **Why the interviewer asks this.** They want to know whether you debug retrieval with measurement or by stacking techniques hopefully.

> **Saying it out loud.** First thing I'd do is figure out which failure I have. Build a small labelled set and check whether the right chunk is in the top hundred. If it isn't, that's a recall problem and re-ranking can't save you — you fix chunking, add hybrid search for vocabulary mismatch, or generate query paraphrases. If it is in the top hundred but not the top five, that's a ranking problem and a cross-encoder re-ranker usually fixes it in an afternoon. Beyond that, the two things I reach for most are hybrid dense-plus-BM25, because keyword search catches exact IDs and product names that embeddings fumble, and small-to-big retrieval, where you index tight chunks for precision but hand the model the surrounding section so it has enough to answer.

---

### Q59: How do you handle context window limits in RAG?

**Answer:**

**1. Priority-Based Selection:**
- Sort by relevance score
- Take top-K until context full
- Truncate if needed

**2. Summarization:**
- Summarize chunks that don't fit
- Hierarchical summarization
- Preserve key information

**3. Chunk Merging:**
- Merge related chunks
- Remove redundancy
- Create coherent context

**4. Dynamic Context:**
- Adaptive chunk selection
- Iterative retrieval
- Expand if needed

**5. Long-Context Models:**
- Use models with larger context (32K, 100K+)
- More expensive but better
- Less truncation

**Best Practice:**
- Prioritize by relevance
- Summarize overflow
- Use appropriate context size

**The part that changes the answer: long context is not the same as good context.**

The list above treats "use a long-context model" as the easy escape hatch, and in 2026 context windows of 200K to 1M tokens are widely available, with some models advertising more. But two effects mean you should not simply dump everything in.

*Lost in the middle.* Model accuracy on retrieving a fact from a long context is strongly position-dependent: highest when the relevant text is near the beginning or the end, measurably worse when it is buried in the middle. This is a robust, repeatedly-reproduced finding. The practical consequence is an ordering rule — put your best chunk first and your second-best last, and let the mediocre ones fill the middle. That is a free accuracy gain and costs one line of code.

*Precision falls as you add context.* Every irrelevant chunk you include is a distractor the model must ignore, and models are imperfect at ignoring. Adding chunks 6 through 20 often *lowers* answer accuracy even though recall went up. Feeding fewer, better chunks generally beats feeding more.

There is also the blunt economic point: input tokens are billed, and attention cost grows quadratically in sequence length (Q89), so a 100K-token prompt is both expensive and slow. *Context window sizes and prices move constantly — check current model specs rather than trusting a number written down months ago.*

**A concrete assembly algorithm.** Given a token budget $B$ for context, reserve room for the system prompt, the question, and the expected answer first — a common bug is budgeting only the input and then getting truncated mid-generation. Then walk the re-ranked chunks in score order, adding each if it fits, and stopping when the budget is spent or the relevance score falls below a floor. That floor matters: a chunk scoring 0.2 is more likely to mislead than to help, so it is better to send fewer chunks than to fill the window because you can. Finally, deduplicate — overlapping chunks from adjacent regions of the same document waste budget on repeated sentences.

**When summarization is and is not appropriate.** Compressing overflow chunks with an LLM adds a call, adds latency, and adds a place for facts to get mangled. It earns its keep for genuinely long single documents where the answer needs the whole arc — "summarize the legal argument across these 40 pages." It is the wrong tool for factual lookup, where losing the exact number or date is the one failure you cannot tolerate. *Contextual compression* — extracting only the query-relevant sentences from each chunk rather than abstractively rewriting it — is the safer middle ground, because it can drop text but cannot invent it.

**Follow-up:** *How do you pick the number of chunks $K$?* Empirically, by sweeping $K$ against end-to-end answer accuracy on your eval set, not by picking a round number. The curve almost always rises then falls, and the peak is often lower than people expect — frequently 3 to 5 chunks rather than 10 or 20.

> **Saying it out loud.** The first move is to stop thinking of it as a fitting problem and start thinking of it as a selection problem — you want the fewest chunks that contain the answer, not the most chunks that fit. I'd re-rank, take the top three to five, dedupe overlapping text, and reserve budget for the system prompt and the answer itself, which people forget. One trick that's basically free: put the strongest chunk first and the second strongest last, because models reliably attend better to the beginning and end than the middle. And I'd push back on just using a million-token window — accuracy often goes down when you add mediocre chunks, because each one is a distractor, and you're paying for every token.

---

### Q60: How do you prevent hallucination in RAG?

**Answer:**

**1. Prompt Engineering:**
```
"Answer ONLY based on the provided context.
If the answer is not in the context, say 'I don't know'."
```

**2. Answer Validation:**
- Check if answer supported by context
- Extract supporting sentences
- Confidence scoring

**3. Citation Generation:**
- Link answer to source chunks
- Show supporting evidence
- Enable fact-checking

**4. Confidence Scoring:**
- Model confidence in answer
- Retrieval confidence
- Combined confidence score

**5. Answer Extraction:**
- Extract answer from context
- Don't generate new information
- Use extractive QA models

**6. Post-Processing:**
- Validate answer against context
- Check for contradictions
- Flag uncertain answers

**Naming the failure precisely, because the fix depends on which one it is.**

"Hallucination" in RAG covers three distinct bugs and they have different remedies. Distinguishing them is most of the answer.

*Retrieval failure.* The right chunk was never retrieved, so the model answered from parametric memory — what it absorbed during pretraining — instead of from your documents. Nothing in the generation stage fixes this; it is a Q58 problem.

*Unfaithful generation.* The correct chunk was in context and the model still said something the chunk does not support — a subtle number change, a merged fact, an over-confident generalization. This is what prompt discipline, citation requirements, and post-hoc verification target.

*Unanswerable question answered anyway.* The corpus genuinely does not contain the answer and the model produces something plausible rather than abstaining. This is the one that needs an explicit escape hatch and an explicit reward for using it.

**A prompt pattern that measurably helps.** Beyond "answer only from the context," three specific instructions do real work: require every sentence to carry a source marker like `[chunk_3]`, so an unsupported claim has nowhere to hide; tell the model to quote the supporting span verbatim before stating the conclusion, which forces it to locate evidence rather than reconstruct it from memory; and make "the provided context does not contain this information" an explicitly sanctioned, named output rather than an implicit option. Models abstain far more readily when abstention is presented as a correct answer rather than a failure.

**Verification, mechanically.** After generation, split the answer into atomic claims and check each against the retrieved context. Three implementations, in increasing cost and accuracy: string or n-gram overlap between claim and context (fast, catches only blatant invention); a *natural language inference* model, which classifies a (premise, hypothesis) pair as entail / contradict / neutral, run with the chunk as premise and the claim as hypothesis (this is the standard approach and a small NLI model is cheap); or an LLM-as-judge call asking whether each claim is supported. Anything scoring below threshold gets stripped or flagged. This is essentially what the *faithfulness* metric in RAG evaluation frameworks computes, so the same machinery serves as both a guardrail and a metric.

**A calibration caveat worth raising.** Token-level model confidence is a poor proxy for factual correctness — models are frequently fluent and confident while wrong, and log-probabilities measure fluency more than truth. If you want a usable confidence signal, derive it from *retrieval* scores and from *self-consistency* (sample the answer several times at non-zero temperature and check whether the samples agree on the load-bearing facts) rather than from the generator's own probabilities.

**Follow-up:** *Does strict grounding hurt anything?* Yes, and you should say so. A tightly grounded model refuses more often, including on questions it could have answered correctly by combining context with common sense. There is a real precision/recall dial here, and where you set it depends on domain: in clinical or legal settings a false abstention is far cheaper than a false claim; in an internal help bot the reverse may hold.

> **Why the interviewer asks this.** Hallucination is the reason RAG systems fail in production, so they want to hear a layered defence rather than a single prompt trick.

> **Saying it out loud.** I'd first separate three different bugs, because they have different fixes. If the right chunk was never retrieved, the model falls back on what it memorized during pretraining — that's a retrieval bug, not a generation bug. If the chunk was there and the model still misstated it, that's unfaithfulness, and that's where prompting, forced citations, and post-hoc checking help. And if the answer just isn't in the corpus, you need "I don't know" to be an explicitly allowed answer, not an implicit one — models abstain a lot more when you name abstention as a valid output. For verification I'd split the answer into atomic claims and run a small entailment model against the retrieved chunks, flagging anything that isn't entailed.

---

### Q61: How do you evaluate a RAG system?

**Answer:**

**Retrieval Metrics:**
- **Precision@K**: Precision of top-K
- **Recall@K**: Recall of top-K
- **MRR**: Mean reciprocal rank
- **MAP**: Mean average precision
- **NDCG@K**: Normalized discounted cumulative gain

**Generation Metrics:**
- **BLEU**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Semantic similarity
- **Answer accuracy**: Correctness

**End-to-End Metrics:**
- **Answer relevance**: Is answer relevant?
- **Answer correctness**: Is answer correct?
- **Answer completeness**: Is answer complete?
- **Citation quality**: Are citations correct?

**Best Practices:**
- Use multiple metrics
- Combine automated + human evaluation
- Monitor in production
- Task-specific evaluation

**Splitting the evaluation so a failure points at a component.**

The metric list above is correct but flat. What makes it usable is organizing it so a bad number tells you which stage to fix. Evaluate retrieval and generation separately, then end to end.

**Retrieval, with the metrics defined.** *Precision@K* is the fraction of the returned $K$ that are relevant; *Recall@K* is the fraction of all relevant chunks that made it into the top $K$. *MRR* — mean reciprocal rank — averages $1/\text{rank of the first relevant result}$, so it cares only about how quickly you find one good answer and is the right metric when a single chunk suffices. *NDCG@K* — normalized discounted cumulative gain — sums graded relevance discounted by $1/\log_2(\text{rank}+1)$ and divides by the best achievable value, so it handles multi-level relevance and rewards putting the best item first. Use MRR for lookup-style questions, NDCG when several chunks contribute and their order matters.

**Generation: why BLEU and ROUGE are near-useless here, and what to use instead.** BLEU and ROUGE measure n-gram overlap with a reference answer. For RAG that is the wrong measurement twice over: a correct answer phrased differently scores badly, and a fluent hallucination that reuses the reference's vocabulary scores well. The field has largely moved to reference-free, LLM-judged decompositions, which are worth knowing by name because they will come up:

- **Faithfulness** — of the claims in the answer, what fraction are supported by the retrieved context? This is the hallucination metric.
- **Answer relevance** — does the answer address the question that was asked, rather than a neighbouring one?
- **Context precision** — of the retrieved chunks, what fraction were actually needed?
- **Context recall** — of the facts in the ground-truth answer, what fraction were present in the retrieved context?

The last two are diagnostic gold: low context recall means fix retrieval, high context recall with low faithfulness means fix generation or prompting.

**How to get a test set without a labelling budget.** Have an LLM read each chunk and generate a question answerable from it alone; the source chunk becomes the ground-truth retrieval label and the chunk text the reference answer. This bootstraps hundreds of examples in an hour. State the limitation too, because a good interviewer will press: synthetic questions are easier and more literal than real user questions, they rarely span multiple chunks, and they inherit the generator's biases. Use synthetic data to catch regressions, and a smaller hand-curated set drawn from real query logs for absolute judgements.

**Beware LLM-as-judge artefacts.** Judges show position bias (favouring the first option shown), verbosity bias (favouring longer answers), and self-preference (favouring text from the same model family). Mitigate by randomizing presentation order, calibrating the judge against a few hundred human labels before trusting it, and — where the stakes justify it — using a judge from a different model family than the generator.

**Follow-up:** *What do you monitor in production, where there are no labels?* Proxies: retrieval score distributions (a drift downward means queries are moving away from your corpus), abstention rate, citation coverage, answer length, latency percentiles, and explicit user feedback. Sample a small fraction of live traffic for offline human or judge review — a hundred conversations a week catches most regressions long before a user complains.

> **Why the interviewer asks this.** Anyone can build a RAG demo; knowing how to tell whether it is getting better is the part that requires having done it.

> **Saying it out loud.** I'd evaluate the two halves separately so a bad number tells me where to look. For retrieval, recall at K and NDCG against a set of queries with the right chunk labelled. For generation, I'd skip BLEU and ROUGE — n-gram overlap punishes correct paraphrases and rewards fluent hallucinations that reuse the reference's words. Instead I'd use the RAGAS-style breakdown: faithfulness, which asks what fraction of the answer's claims are supported by the retrieved context; answer relevance; and context precision and recall. Context recall low means fix retrieval; context recall fine but faithfulness low means fix the prompt or the model. For a test set I'd bootstrap with LLM-generated questions per chunk, but I'd keep a hand-curated set from real logs too, because synthetic questions are always easier than real ones.

---

### Q62: What are common RAG challenges and solutions?

**Answer:**

**1. Chunking Strategy:**
- **Challenge**: How to split documents
- **Solution**: Semantic chunking, hierarchical, overlap

**2. Embedding Quality:**
- **Challenge**: Domain-specific semantics
- **Solution**: Fine-tuning, hybrid embeddings

**3. Retrieval Accuracy:**
- **Challenge**: Retrieved chunks not relevant
- **Solution**: Multi-stage retrieval, re-ranking, hybrid search

**4. Context Window Limits:**
- **Challenge**: Too many chunks, can't fit
- **Solution**: Priority selection, summarization, long-context models

**5. Hallucination:**
- **Challenge**: Model generates wrong info
- **Solution**: Prompt engineering, citations, validation

**6. Scalability:**
- **Challenge**: Large document sets
- **Solution**: ANN search, distributed systems, caching

**7. Cost:**
- **Challenge**: High API costs
- **Solution**: Self-hosted models, caching, batch processing

**Three challenges the list misses, which are usually the ones that bite in production.**

*Access control and multi-tenancy.* If different users may see different documents, retrieval must filter by permission *before* or *during* the search, not after. Filtering after means the ranking was computed over documents the user cannot see, so results become sparse and inconsistent — and if any snippet leaks into a prompt, you have a data breach rather than a quality bug. Most vector databases support pre-filtered ANN search; check that yours does it natively and understand the recall cost, because aggressive filters can force the index to search a much larger fraction of the graph and latency degrades sharply as filters get selective.

*Freshness and staleness.* A vector index is a cache of your documents and it goes stale like any cache. You need change detection on the source, incremental re-embedding keyed by a content hash, real deletes, and — critically — a plan for what happens when you change embedding models, which invalidates every vector in the store. Budget a full re-index as a routine operation rather than an emergency.

*Conflicting sources.* Real corpora contain a 2023 policy document and its 2026 replacement, both retrievable and both plausible. The model has no way to know which wins. The fixes are metadata-driven: attach effective dates and version numbers at ingest, prefer recent documents in ranking, and surface the conflict to the user ("two sources disagree; the current policy says...") rather than silently picking one. This one comes up in almost every real deployment and almost never in tutorials.

**On cost, with numbers.** The cost line above says "high API costs" without saying where they go. Concretely, embedding is one-time and small — on the order of \$0.02 per million tokens for a small commercial model as of August 2026, so a large corpus costs single-digit dollars to embed. Generation is per-query and dominates: if each query sends 4,000 input tokens of retrieved context and produces 300 output tokens, your per-query cost is set almost entirely by the model tier and the context size. The three levers that actually move the bill are, in order: send fewer retrieved tokens; cache aggressively (both exact-match response caching for repeated queries and provider-side prompt caching for a stable system prompt, which can cut input cost substantially); and route easy queries to a smaller model, reserving the frontier model for hard ones. *These figures are time-sensitive — verify current pricing before quoting it.*

**Follow-up:** *What is the single most common cause of a disappointing RAG system?* Chunking and ingestion quality, not model choice. Badly parsed PDFs, tables flattened into word salad, headers and footers repeated into every chunk, and boilerplate that dominates the embedding will cap your ceiling no matter how good the retriever is. Reading fifty random chunks by eye is an unglamorous hour that finds more bugs than a week of hyperparameter tuning.

> **Saying it out loud.** The textbook list is chunking, embedding quality, retrieval accuracy, context limits, hallucination, scale, and cost. The ones that actually bite in production, though, are a bit different. Access control — you have to filter by permission during retrieval, not after, or your ranking is computed over documents the user can't see. Staleness — the index is a cache, and it needs change detection, real deletes, and a plan for the day you swap embedding models and invalidate everything. And conflicting sources, where an old policy and its replacement are both retrievable and equally plausible; you fix that with effective dates in metadata and recency in ranking. And honestly the most common cause of a bad RAG system is just bad ingestion — mangled PDFs and flattened tables — not the model.

---

See `39_rag_retrieval_augmented_generation/` for detailed implementations!

---

### Q63: Explain different chunking strategies. When to use each?

**Answer:**

**1. Fixed-Size Chunking:**
- Split into fixed-size chunks (e.g., 512 chars)
- Overlap between chunks (10-20%)
- **Use**: Simple documents, prototyping, uniform content
- **Pros**: Simple, fast
- **Cons**: Breaks sentences, no semantic awareness

**2. Sentence-Based Chunking:**
- Split on sentence boundaries
- Group sentences into chunks
- **Use**: Narrative text, general documents, production (common default)
- **Pros**: Respects boundaries, better coherence
- **Cons**: Sentence splitting can be imperfect

**3. Paragraph-Based Chunking:**
- Split on paragraph boundaries
- **Use**: Structured documents, academic papers, long-form
- **Pros**: Preserves structure, natural units
- **Cons**: Variable sizes, may be too large/small

**4. Semantic Chunking:**
- Use embeddings to find semantic boundaries
- Split when semantic shift detected
- **Use**: High accuracy needs, topic-based documents
- **Pros**: Best semantic coherence, optimal retrieval
- **Cons**: Slower, requires embeddings, higher cost

**5. Recursive Chunking:**
- Hierarchical splitting (paragraphs → sentences → words)
- **Use**: General-purpose, variable structure, production (LangChain default)
- **Pros**: Robust, handles any structure
- **Cons**: More complex, can be slow

**6. Sliding Window:**
- Fixed-size with stride (overlap)
- **Use**: Sequential data, code, long documents
- **Pros**: Preserves context, good for sequential
- **Cons**: Many chunks, redundancy

**7. Token-Based:**
- Split by token count (not characters)
- **Use**: LLM systems, accurate sizing, cost optimization
- **Pros**: Accurate for LLMs, precise control
- **Cons**: Requires tokenizer, model-specific

**8. Hierarchical:**
- Multi-level (document → section → paragraph)
- **Use**: Complex documents, academic papers
- **Pros**: Preserves structure, multi-level retrieval
- **Cons**: Complex, more storage

**9. Content-Aware:**
- Different strategy per content type (code, tables, text)
- **Use**: Mixed content, technical docs, research papers
- **Pros**: Optimal per type, handles complexity
- **Cons**: Very complex, requires detection

**10. Metadata-Enriched:**
- Chunks with rich metadata (section, page, etc.)
- **Use**: Structured docs, citations, filtering
- **Pros**: Rich context, better filtering
- **Cons**: More storage, complex

**Best Practices:**
- **Start**: Sentence-based or recursive
- **Upgrade**: Semantic if accuracy critical
- **Overlap**: 10-20% of chunk size
- **Size**: 256-1024 tokens (512 common)
- **Test**: Evaluate retrieval accuracy

**How to actually choose, rather than picking from the menu.**

The decision reduces to one tension. Small chunks retrieve precisely, because the embedding of 100 tokens about one topic is a sharp point in the space, but they answer badly, because the surrounding sentence that resolves a pronoun or supplies the units is missing. Large chunks answer well and retrieve poorly, because averaging 1,500 tokens of mixed content produces a vague vector that is close to everything and specific to nothing. Every strategy on the list is an attempt to get one without paying for the other.

**The move that resolves it: decouple the retrieval unit from the generation unit.** Index small (a sentence, a paragraph, or an LLM-written summary), retrieve on that, then hand the model the *parent* — the surrounding section or full document. This is variously called parent-document retrieval, small-to-big, or sentence-window retrieval, and it is what most mature systems converge on. Mentioning it moves the answer from "I know the list" to "I have built one."

**Overlap, concretely.** With 512-token chunks and 15% overlap you carry about 77 tokens from the previous chunk into the next, so a fact straddling a boundary appears whole in at least one chunk. The cost is storage and near-duplicate results, which is why you deduplicate at assembly time (Q59).

**Structure beats heuristics whenever structure exists.** Markdown headers, HTML sections, docstrings, and function boundaries are ground truth about where topics begin and end, and they are free. Split on them first, and only fall back to length-based splitting inside an oversized section. Recursive character splitting is exactly this idea generalized: try paragraph breaks, then sentence breaks, then whitespace, then raw characters, descending only when a piece is still too big.

**On semantic chunking, an honest caveat.** It sounds obviously better — embed each sentence, cut where consecutive similarity drops — and in published comparisons it wins less often and by less than expected, while costing an embedding call per sentence at ingest. It is worth trying, but "we use semantic chunking because it is more principled" is not a result. Measure it against recursive splitting on your own data before adopting it.

**Content types that break naive chunking, which is where real corpora live.** Tables must never be split across chunks, and a table separated from its caption and column headers is unusable — serialize each row with its headers, or keep the table whole and attach a text summary for embedding. Code should be split on function or class boundaries with the imports and enclosing signature carried along. Long lists and procedures should keep their numbered steps together, since half a procedure is worse than none. And headers, footers, and navigation boilerplate should be stripped before chunking; if the same 40-token footer appears in every chunk, it contributes to every embedding and pulls the whole corpus toward a common, useless direction.

**Follow-up:** *How do you evaluate a chunking strategy without rebuilding everything?* Fix a labelled query set, then measure recall@50 for each candidate strategy — it isolates chunking from ranking, since re-ranking cannot recover a chunk that retrieval never surfaced. Re-indexing a few thousand documents under three strategies is an afternoon, and the winner is usually visible immediately.

> **Why the interviewer asks this.** Chunking is where most of a RAG system's quality is actually decided, so a candidate who treats it as a preprocessing detail has probably not debugged one.

> **Saying it out loud.** The whole thing comes down to one tradeoff: small chunks retrieve well but answer badly, and big chunks answer well but retrieve badly. Small chunks give you a sharp embedding; big chunks average into mush. So my default is recursive splitting that respects structure first — headers, paragraphs, then sentences — at maybe five hundred tokens with ten or fifteen percent overlap. But the real fix is to decouple the two: index small units for precision, then hand the model the surrounding parent section so it has enough to actually answer. And I'd treat tables and code specially — never split a table from its header row, and split code on function boundaries — because that's where naive chunking silently destroys the corpus.

---

See `39_rag_retrieval_augmented_generation/chunking_strategies.md` for complete guide!

---

## Linear and Logistic Regression Derivations

### Q64: Derive linear regression from first principles. Explain intuitively.

**Answer:**

**Goal:** Find line y = wx + b that best fits data

**Step 1: Define Error**
- For each point: error = yᵢ - (wxᵢ + b)
- Actual value minus predicted value

**Step 2: Cost Function (MSE)**
- Sum of squared errors: MSE = (1/n) Σ (yᵢ - wxᵢ - b)²
- Why squared? Always positive, penalizes large errors, differentiable

**Step 3: Minimize**
- Take derivatives, set to zero
- ∂MSE/∂b = 0 → b = ȳ - wx̄ (line passes through center!)
- ∂MSE/∂w = 0 → w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

**Final Solution:**
```
w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²  (covariance / variance)
b = ȳ - wx̄
```

**Intuition:**
- Slope = how much y changes per unit x (covariance/variance)
- Intercept = mean y - slope × mean x
- Line passes through center of data (x̄, ȳ)

**Matrix Form:**
```
w = (XᵀX)⁻¹Xᵀy  (normal equation)
```

**Why it works:**
- Minimizes sum of squared errors (geometrically optimal)
- Projects y onto column space of X
- Unique solution (if data not degenerate)

**The derivative steps, fully worked.**

The answer above jumps from "take derivatives, set to zero" straight to the solution. Here is what happens in between, because that gap is exactly what a whiteboard interview asks you to fill.

Write the cost as a sum rather than an average — dividing by $n$ scales the whole function by a positive constant and cannot move the minimizer, so dropping it is legal and keeps the algebra clean:

$$J(w,b) = \sum_{i=1}^{n} (y_i - wx_i - b)^{2}.$$

**Differentiating with respect to $b$.** Treat $w$ as fixed. Each term is a square of something whose derivative with respect to $b$ is $-1$, so by the chain rule:

$$\frac{\partial J}{\partial b} = \sum_i 2(y_i - wx_i - b)(-1) = -2\sum_i (y_i - wx_i - b).$$

Set to zero and divide by $-2$: $\sum_i y_i - w\sum_i x_i - nb = 0$. Divide through by $n$ and use $\bar y = \frac1n\sum y_i$:

$$\bar y - w\bar x - b = 0 \;\Longrightarrow\; b = \bar y - w\bar x .$$

The interpretation is worth stating: the fitted line passes exactly through the centroid $(\bar x,\bar y)$, for *any* slope. That is why the residuals always sum to zero when the model has an intercept.

**Differentiating with respect to $w$.** The inner derivative is now $-x_i$:

$$\frac{\partial J}{\partial w} = -2\sum_i x_i(y_i - wx_i - b) = 0 .$$

**Substituting $b$ to eliminate it.** Replace $b$ with $\bar y - w\bar x$:

$$\sum_i x_i\big(y_i - \bar y - w(x_i - \bar x)\big) = 0 \;\Longrightarrow\; \sum_i x_i(y_i - \bar y) = w\sum_i x_i(x_i - \bar x).$$

**The centering step that makes it recognizable.** The useful identity here is that $\sum_i (x_i - \bar x) = 0$, so subtracting a constant multiple of it changes nothing: $\sum_i x_i(y_i-\bar y) = \sum_i (x_i - \bar x)(y_i - \bar y)$, and likewise $\sum_i x_i(x_i-\bar x) = \sum_i (x_i - \bar x)^{2}$. Substituting:

$$w = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^{2}} = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)} .$$

**Confirm it is a minimum.** $\partial^{2}J/\partial w^{2} = 2\sum x_i^{2} > 0$, $\partial^{2}J/\partial b^{2} = 2n > 0$, and the determinant of the $2\times2$ Hessian is $4n\sum(x_i-\bar x)^{2} > 0$ whenever the $x_i$ are not all identical. Positive definite Hessian, so it is the unique global minimum — and the degenerate case (all $x_i$ equal) is exactly the case where no slope is identifiable, which the algebra reports as a zero denominator.

**A relationship worth having ready.** Since $\operatorname{Corr}(x,y) = \operatorname{Cov}(x,y)/(s_x s_y)$, the slope is $w = r \cdot s_y/s_x$. If you standardize both variables the slope *is* the correlation. That is a one-line answer to "what is the relationship between regression and correlation?"

**Why squared error and not absolute error.** Three separate reasons, and they are different: squared error is differentiable everywhere while $|e|$ has a kink at zero; it has a closed-form solution while absolute error requires linear programming; and it is the maximum-likelihood loss under Gaussian noise (Q47). The cost is sensitivity to outliers, since a residual of 10 contributes 100 while a residual of 1 contributes 1. Minimizing absolute error gives you the conditional *median* instead of the conditional *mean*, which is the robust alternative and is what quantile regression generalizes.

**Numerically verified.** On 40 simulated points with true slope 2.5 and intercept 1.3, the covariance-over-variance formula gives slope 2.4754, intercept 1.3888, and solving the normal equation $(X^{\top}X)^{-1}X^{\top}y$ gives the identical pair to ten decimal places — as it must, since they are the same equations written differently.

**Follow-up:** *When would you not use the normal equation?* When $p$ is large, since forming and inverting $X^{\top}X$ costs $O(np^{2} + p^{3})$ and becomes impractical past a few thousand features; when $X^{\top}X$ is ill-conditioned, where you should use a QR or SVD-based solver instead of an explicit inverse for numerical stability; or when the data does not fit in memory, where SGD is the answer. In practice you should essentially never compute a matrix inverse explicitly — `np.linalg.solve` or `lstsq` is both faster and more stable than `inv(A) @ b`.

> **Why the interviewer asks this.** Linear regression is the smallest model where you can demonstrate the full loop — assumption, loss, gradient, closed form, geometry — so it is a proxy for whether you can derive anything at all.

> **Saying it out loud.** You start with a line and define the error at each point as actual minus predicted. Square it, because squaring is differentiable, penalizes big misses more, and — this is the real reason — it's the maximum-likelihood loss if you assume Gaussian noise. Sum over points, then take partial derivatives with respect to slope and intercept and set both to zero. The intercept equation immediately tells you the line goes through the mean of x and the mean of y. Substitute that back in and the slope comes out as covariance over variance — which reads as "how much do x and y move together, normalized by how much x moves on its own." Geometrically you're projecting y onto the column space of X, so the residual ends up orthogonal to every feature.

---

### Q65: Derive logistic regression. Why sigmoid function?

**Answer:**

**Problem:** Need probabilities (0 to 1), not continuous values

**Step 1: Log Odds**
- Odds = P / (1-P) (can be any positive number)
- Log odds = log(P / (1-P)) (can be any real number)
- Model: log(P / (1-P)) = wx + b (linear!)

**Step 2: Solve for P**
- P / (1-P) = e^(wx + b)
- P = e^(wx + b) / (1 + e^(wx + b))
- P = 1 / (1 + e^(-(wx + b))) = σ(wx + b)

**Why Sigmoid?**
- Bounded: Always between 0 and 1
- Smooth: Differentiable everywhere
- S-shaped: Good for probabilities
- When wx + b → -∞: P → 0
- When wx + b = 0: P = 0.5
- When wx + b → +∞: P → 1

**Step 3: Likelihood**
- P(y|x) = σ(wx + b)^y × (1 - σ(wx + b))^(1-y)
- Likelihood = ∏ᵢ P(yᵢ|xᵢ)

**Step 4: Cost Function (Cross-Entropy)**
- Maximize likelihood = Minimize negative log-likelihood
- J(w, b) = -Σ [y log σ(wx + b) + (1-y) log(1 - σ(wx + b))]

**Step 5: Gradient**
- ∂J/∂w = Σ [σ(wx + b) - y] × x
- ∂J/∂b = Σ [σ(wx + b) - y]
- Error = predicted - actual

**Update:**
```
w = w - α × Σ [σ(wx + b) - y] × x
b = b - α × Σ [σ(wx + b) - y]
```

**Decision Boundary:**
- P(y=1|x) = 0.5 when wx + b = 0
- Line wx + b = 0 separates classes

**Why Cross-Entropy?**
- Optimal for classification (information theory)
- Convex (guaranteed global minimum)
- Works well with probabilities

**The gradient derivation, with the step everyone skips.**

The answer above states $\partial J/\partial w = \sum(\sigma(wx+b) - y)x$ without showing why the sigmoid derivative and the log both vanish so cleanly. That cancellation is the point of the question, so here it is in full.

**Fact 1 — the sigmoid derivative.** With $\sigma(z) = 1/(1+e^{-z})$, apply the quotient (or chain) rule to $(1+e^{-z})^{-1}$:

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^{2}} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\big(1-\sigma(z)\big),$$

using $\frac{e^{-z}}{1+e^{-z}} = \frac{(1+e^{-z}) - 1}{1+e^{-z}} = 1 - \sigma(z)$. This identity is why logistic regression is cheap: you already have $\sigma(z)$ from the forward pass, so the derivative costs one multiply.

**Fact 2 — differentiate the loss for one example.** Let $p = \sigma(z)$ with $z = w^{\top}x + b$, and let the per-example loss be $J = -[y\log p + (1-y)\log(1-p)]$. Differentiate with respect to $p$ first:

$$\frac{\partial J}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{-y(1-p) + p(1-y)}{p(1-p)} = \frac{p - y}{p(1-p)}.$$

**Fact 3 — chain through $z$, and watch the cancellation.**

$$\frac{\partial J}{\partial z} = \frac{\partial J}{\partial p}\cdot\frac{\partial p}{\partial z} = \frac{p-y}{p(1-p)}\cdot p(1-p) = p - y .$$

The $p(1-p)$ factors cancel exactly. Then $\partial z/\partial w = x$ gives

$$\frac{\partial J}{\partial w} = (p - y)\,x, \qquad \frac{\partial J}{\partial b} = p - y,$$

and summing over the dataset reproduces the answer above. In matrix form, $\nabla_w J = X^{\top}(\sigma(Xw) - y)$.

**Why this cancellation is the whole argument for cross-entropy.** Try squared error instead: $J = \tfrac12(p-y)^{2}$ gives $\partial J/\partial z = (p-y)\,p(1-p)$, and that extra $p(1-p)$ factor goes to zero whenever $p$ is near 0 or 1. So a confidently *wrong* prediction — $p = 0.999$ when $y=0$ — produces a gradient of about $0.001$ and the model barely moves. Cross-entropy's gradient in that case is $0.999$: proportional to the error, with no saturation. That is the real reason for the loss choice, and it is a strictly better answer than "cross-entropy is information-theoretically optimal."

**On convexity, precisely.** The Hessian of the summed cross-entropy is $\nabla^{2}J = X^{\top}SX$ where $S = \operatorname{diag}\big(p_i(1-p_i)\big)$. Every $p_i(1-p_i) > 0$, so for any vector $v$, $v^{\top}X^{\top}SXv = \lVert S^{1/2}Xv\rVert^{2} \ge 0$ — positive semi-definite, hence convex, hence no local minima. It is strictly convex when $X$ has full column rank. Note also that there is no closed-form solution, because setting $X^{\top}(\sigma(Xw)-y)=0$ leaves $w$ inside a nonlinear function; you solve it iteratively with gradient descent or Newton's method (the latter, applied here, is exactly iteratively reweighted least squares).

**The failure mode to mention: separable data.** If a hyperplane perfectly separates the classes, the likelihood can always be increased by scaling $w$ up — pushing every $p_i$ closer to 0 or 1 — so the maximum is at infinity and the weights diverge. Any L2 penalty fixes this by bounding $\lVert w\rVert$, which is one more reason regularized logistic regression is the default in every library.

**Verified numerically.** For a 50-example, 3-feature problem, the analytic gradient $X^{\top}(\sigma(Xw)-y)$ and a central-difference numerical gradient agree to within $3\times10^{-10}$.

**Follow-up:** *Why the logit link specifically, and not probit?* The logit is the *canonical link* for the Bernoulli family, which is what makes the gradient come out as the clean "prediction minus target" above and guarantees convexity. Probit (the Gaussian CDF) fits almost identically in practice — the curves differ mainly in the tails — but loses the closed-form gradient elegance and the log-odds interpretation that makes coefficients readable as odds ratios.

> **Why the interviewer asks this.** The $p(1-p)$ cancellation is the moment where loss choice, activation choice and gradient behaviour all connect, so it is a compact test of whether you understand backprop or merely use it.

> **Saying it out loud.** I start from the constraint that I need an output between zero and one. A linear model can output anything, so instead of modelling the probability directly I model the log-odds, which does range over all the reals — set log p over one-minus-p equal to w-x plus b. Solve that for p and the sigmoid just falls out; it's not an arbitrary choice, it's the inverse of the logit. Then for the loss I write the Bernoulli likelihood and take negative log, which gives cross-entropy. The nice part is the gradient: the derivative of sigmoid is p times one-minus-p, and the derivative of the log loss has p times one-minus-p in the denominator, so they cancel exactly and you're left with just prediction minus target times x. That cancellation is why cross-entropy beats squared error here — with squared error that factor survives, so a confidently wrong prediction gets an almost zero gradient and the model can't correct itself.

---

### Q66: Why can't we use linear regression for classification?

**Answer:**

**Problems:**

**1. Output Range:**
- Linear regression: Output can be any real number (-∞, +∞)
- Classification: Need probabilities [0, 1]
- Linear regression can give negative values or > 1

**2. Interpretation:**
- Linear regression output doesn't represent probability
- Can't interpret as "80% chance of class 1"

**3. Loss Function:**
- MSE not optimal for classification
- Doesn't penalize misclassifications appropriately
- Can get stuck in local minima

**4. Decision Boundary:**
- Linear regression: Threshold at 0.5 (arbitrary)
- No probabilistic interpretation
- Doesn't work well for imbalanced classes

**Solution:**
- Use logistic regression (sigmoid)
- Output is probability [0, 1]
- Cross-entropy loss (optimal for classification)
- Probabilistic interpretation

**When Linear Regression Works for Classification:**
- Binary classification with balanced classes
- When you just need a threshold
- But logistic regression is almost always better

**The mechanism behind "MSE is not optimal," which is the part worth knowing.**

The list above is right, but points 3 and 4 assert without explaining, and a follow-up will find that out. Two concrete mechanisms:

*The gradient vanishes exactly where you need it most.* Suppose you use a linear output with squared error and threshold at 0.5. A point of class 0 predicted at $-3$ is already correctly classified, yet it contributes a residual of 9 to the loss and a large gradient — the model spends capacity dragging an already-correct point toward zero, rotating the boundary to accommodate it. This is the real content of "linear regression is sensitive to outliers in classification": a single far-away but correctly-classified point can shift the decision boundary enough to misclassify points near it. Logistic regression's loss is near zero for confidently correct points, so they exert almost no pull.

*The claim about local minima is not right, and you should correct it.* Squared error with a *linear* output is convex in $w$ — it is ordinary least squares — so there are no local minima. The genuine problems are the ones above (wrong loss shape, no probabilistic meaning, outlier sensitivity), not optimization getting stuck. If you pair squared error with a *sigmoid* output, then you do get a non-convex surface with flat saturated regions, which is a real optimization problem — but that is a different model from plain linear regression. Being precise about which combination has which problem is a strong signal.

**One nuance that makes the "when it works" section honest.** Least-squares classification is not merely a hack: for two classes with equal covariance, the least-squares fit to $\pm1$ targets produces the *same* decision direction as Fisher's linear discriminant, up to scale. So it can work respectably on well-behaved balanced binary problems. It degrades badly for more than two classes, though, where one-hot targets with least squares produce the classic *masking* problem — an intermediate class can be dominated by its neighbours and never predicted at all for any input.

**Follow-up:** *Isn't logistic regression also just a linear boundary?* Yes — the decision boundary $w^{\top}x + b = 0$ is a hyperplane in both cases. The difference is not the shape of the boundary but the loss used to place it and the calibrated probability you get out. If you need a nonlinear boundary you need features (polynomial, splines, kernels) or a different model class; switching from linear to logistic regression does not buy you one.

> **Saying it out loud.** Two real problems. First, the output isn't a probability — it can be negative or above one, so you can't threshold it meaningfully or feed it to anything that expects a probability. Second, and this one matters more, squared error keeps pushing on points that are already correct. A class-zero point predicted at minus three is right, but it still contributes a residual of nine, so the model rotates the boundary to reduce it and can misclassify points near the boundary as a result. Logistic regression's loss goes to almost nothing for confidently correct points, so they stop pulling. One thing I'd correct in the usual answer: least squares with a linear output is still convex, so "local minima" isn't the issue — the issue is the loss is shaped wrong for the task.

---

### Q67: Explain the relationship between linear and logistic regression.

**Answer:**

**Similarities:**
- Both use linear combination: wx + b
- Both learn weights w and bias b
- Both use gradient descent (or closed-form for linear)

**Differences:**

**Linear Regression:**
- Output: Continuous values (-∞, +∞)
- Model: y = wx + b
- Cost: MSE (sum of squared errors)
- Solution: Closed-form (normal equation) or gradient descent

**Logistic Regression:**
- Output: Probabilities [0, 1]
- Model: P(y=1|x) = σ(wx + b)
- Cost: Cross-entropy (negative log-likelihood)
- Solution: Gradient descent (no closed-form)

**Key Insight:**
- Logistic regression = Linear regression + sigmoid
- Log odds are linear: log(P/(1-P)) = wx + b
- Probabilities are sigmoid: P = σ(wx + b)

**Visual:**
```
Linear:     y = wx + b  (straight line)
            ↓
Logistic:   P = σ(wx + b)  (sigmoid curve)
```

**Connection:**
- Both model linear relationships
- Linear: Direct relationship
- Logistic: Linear in log-odds space

**The unifying frame: generalized linear models.**

The cleanest way to say what these two share is that both are *generalized linear models* — GLMs. A GLM has three parts: a linear predictor $\eta = w^{\top}x + b$; a distribution for $y$ from the exponential family; and a *link function* $g$ connecting them by $g(\mathbb{E}[y\mid x]) = \eta$. Linear regression is the GLM with a Gaussian response and the identity link, $\mathbb{E}[y] = \eta$. Logistic regression is the GLM with a Bernoulli response and the logit link, $\log\frac{p}{1-p} = \eta$. Poisson regression is the same recipe with a count response and a log link. Naming this frame turns "logistic is linear plus a sigmoid" into a principle that also tells you what to do when the target is a count, a rate, or a duration.

**Why both gradients look identical, which is not a coincidence.** For linear regression, $\nabla_w J = \sum(\hat y_i - y_i)x_i$. For logistic regression, $\nabla_w J = \sum(p_i - y_i)x_i$. Same form: prediction minus target, times the input. This holds for every GLM fitted with its *canonical link* — the link that makes the exponential-family natural parameter equal the linear predictor. It is a general theorem, not a lucky cancellation repeated twice, and it is why the same optimizer code works across all of them.

**Where the analogy breaks.** Three places worth naming, since "logistic = linear + sigmoid" is a useful slogan that hides real differences.

*Closed form.* Linear regression has one because its gradient is linear in $w$. Logistic does not, because $w$ sits inside a nonlinear $\sigma$. You iterate — gradient descent, L-BFGS, or Newton's method, which here is exactly iteratively reweighted least squares: at each step you solve a weighted least-squares problem with weights $p_i(1-p_i)$, so logistic regression is literally repeated linear regression.

*Existence of the optimum.* Linear regression always has a finite solution given full rank. Logistic regression's optimum runs off to infinity on perfectly separable data (Q65), which is why the default in every serious library is regularized.

*Interpretation of coefficients.* In linear regression $w_j$ is "the change in $y$ per unit change in $x_j$," in the units of $y$. In logistic regression $w_j$ is a change in *log-odds*, so $e^{w_j}$ is an *odds ratio*: a coefficient of $0.7$ means the odds multiply by about $2$ per unit increase. People routinely misreport this as a change in probability, which is wrong — the effect on probability depends on where on the sigmoid you are, and is largest near $p=0.5$ and nearly nil in the tails.

**Follow-up:** *So could I fit logistic regression by running linear regression on the log-odds of the labels?* No, and it is a good trap. The labels are 0 and 1, so their empirical log-odds are $\pm\infty$. You would need grouped data with observed proportions in each group, which is the classical "logit transform then weighted least squares" approach, and it fails as soon as a group has a proportion of exactly 0 or 1. Maximum likelihood on the individual observations avoids the whole problem.

> **Why the interviewer asks this.** They are probing whether you have a general framework or two memorised special cases; the GLM answer demonstrates the former in one sentence.

> **Saying it out loud.** Both are generalized linear models — same linear predictor w-x plus b, different assumption about the response and a different link. Linear regression assumes Gaussian noise with an identity link, so the prediction is the linear predictor directly. Logistic assumes Bernoulli with a logit link, so the linear predictor is the log-odds and you sigmoid it to get a probability. That's also why both gradients come out as prediction minus target times x — that's a general property of canonical links, not a coincidence. Where they differ: linear has a closed form because the gradient is linear in w; logistic doesn't, so you iterate. And the coefficients mean different things — in logistic, exponentiating a coefficient gives you an odds ratio, not a change in probability.

---

See `01_classical_ml/linear_regression_derivation.md` and `01_classical_ml/logistic_regression_derivation.md` for complete derivations!

---

## RAG Retrieval Methods

### Q68: Explain BM25. How does it differ from TF-IDF?

**Answer:**

**BM25** (Best Matching 25) is industry-standard sparse retrieval, improving upon TF-IDF.

**BM25 Formula:**
```
BM25(t, d) = IDF(t) × (f(t, d) × (k₁ + 1)) / (f(t, d) + k₁ × (1 - b + b × |d|/avgdl))
```

**Key Improvements:**

**1. Term Frequency Saturation:**
- TF-IDF: Linear (10x → score = 10, 20x → score = 20)
- BM25: Saturates (10x → 8.5, 20x → 9.2)
- Prevents one term from dominating

**2. Document Length Normalization:**
- Normalizes by document length
- Prevents bias toward long documents

**3. Better IDF:**
- BM25 IDF: log((N - df + 0.5) / (df + 0.5))
- More robust

**Use Cases:**
- Keyword-based search
- Production systems (Elasticsearch, Lucene)
- Better than TF-IDF in most cases

**Reading the formula piece by piece.**

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\frac{|d|}{\text{avgdl}}\right)}$$

*$f(t,d)$* is the raw count of term $t$ in document $d$. *$|d|$* is the document length in tokens and *avgdl* the mean length across the collection. *$k_1$* controls how fast term frequency saturates (typical value 1.2, sometimes 1.5) and *$b$* controls how strongly length normalization applies, from 0 (none) to 1 (full), with 0.75 the standard default.

**The saturation term, with real numbers.** Set $b=0.75$ and $k_1=1.2$ and consider a document of average length, so the bracket equals 1 and the factor is $f(k_1+1)/(f+k_1)$. Then:

| $f$ | 1 | 2 | 5 | 10 | 20 | 100 |
|---|---|---|---|---|---|---|
| TF factor | 1.000 | 1.375 | 1.774 | 1.964 | 2.075 | 2.174 |

Two things are visible. Going from 1 to 2 occurrences buys 0.375; going from 20 to 100 buys 0.099. And the factor is bounded above by $k_1 + 1 = 2.2$, approached but never reached. That ceiling is the mathematically important point: *no single term can ever dominate a BM25 score by repetition*, which is precisely the keyword-stuffing attack that linear TF-IDF is vulnerable to.

*A correction to the illustrative figures in the answer above.* The numbers "10x → 8.5, 20x → 9.2" do not correspond to any standard $(k_1,b)$ setting — the term-frequency factor is capped at $k_1+1$, so it cannot reach 8.5. Read them as a schematic illustration of "sublinear growth," not as computed BM25 values; the table just above gives the actual figures for the standard parameters.

**The length normalization term, with real numbers.** With the same parameters, an IDF of 6.91 (a term in 1,000 of 1,000,000 documents) and $f=3$: an average-length document scores $6.91 \times 1.571 = 10.85$, while a document four times the average length scores $6.91 \times 0.956 = 6.61$. Three mentions in a short focused page is stronger evidence of aboutness than three mentions in a long rambling one, and $b$ is the dial for how much you believe that.

**The IDF change, and why the $+1$ matters.** BM25's probabilistic IDF is $\log\frac{N - \text{df} + 0.5}{\text{df} + 0.5}$. Note that this goes *negative* when a term appears in more than half the documents — which arguably makes sense (a term in 90% of documents is anti-evidence) but breaks scoring in practice, since a document could improve its score by dropping a common term. Real implementations, Lucene included, use $\log\left(1 + \frac{N-\text{df}+0.5}{\text{df}+0.5}\right)$, which is always positive. Knowing that the textbook formula and the shipped formula differ here is a nice detail.

*(All figures above computed directly from the formula with $k_1=1.2$, $b=0.75$, $N=10^{6}$, $\text{df}=1000$.)*

**Follow-up:** *Why is BM25 still competitive in 2026?* Because it has zero training cost, no domain adaptation problem, is fully interpretable ("this document ranked high because it contains these three rare query terms"), handles rare exact strings — error codes, part numbers, proper nouns — that embeddings blur together, and inverts trivially to a sparse index. Its weakness is exactly one thing: it cannot match a query to a document that uses different words for the same idea. Which is why hybrid search exists (Q69).

> **Why the interviewer asks this.** BM25 is a rare case of a hand-designed formula where every term has a defensible justification, so explaining it tests whether you can read a model rather than just call one.

> **Saying it out loud.** BM25 is TF-IDF with two fixes. First, term frequency saturates instead of growing linearly — going from one occurrence to two helps a lot, twenty to a hundred barely helps at all, and the whole factor is capped at k1 plus one, so no amount of keyword stuffing lets one term run away with the score. Second, it normalizes by document length relative to the collection average, so three mentions in a short page counts for more than three mentions in a long one, with a parameter b controlling how aggressively. The defaults are k1 around 1.2 and b around 0.75 and they're remarkably robust. It's still a strong baseline because it needs no training and nails exact matches like error codes and part numbers that embeddings tend to smear together.

---

### Q69: Explain hybrid search in RAG. How do you combine sparse and dense?

**Answer:**

**Hybrid Search** combines BM25 (sparse) + Dense (embeddings).

**Why:**
- BM25: Exact matches, keywords
- Dense: Semantic similarity
- Neither perfect alone → Combine!

**How:**
1. Retrieve from both (top-K each)
2. Normalize scores to [0, 1]
3. Combine: Final = α × BM25 + (1-α) × Dense
4. Re-rank by combined score

**Weight Selection:**
- α = 0.7: More BM25 (keyword-heavy)
- α = 0.5: Balanced (default)
- α = 0.3: More dense (semantic)

**Best Practice:**
- Normalize before combining
- Tune α on validation set
- Use for production systems

**Why score normalization is harder than it sounds, and what people do instead.**

The weighted-sum recipe above has a real problem: BM25 scores are unbounded and their scale depends on the query — a rare three-term query produces much larger scores than a common one-term query — while cosine similarities live in $[-1,1]$ and, for a good embedding model, are usually squeezed into a narrow band like $[0.6, 0.9]$. Min-max normalizing *within a single result list* makes the top result 1.0 and the bottom 0.0 regardless of whether that list was excellent or useless, so a query where dense retrieval found nothing still contributes a confident-looking 1.0. That is why weighted score fusion needs care and per-query calibration.

**Reciprocal Rank Fusion, the method most systems actually use.** RRF throws the scores away and fuses *ranks*:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}, \qquad k \approx 60 .$$

Because only ranks enter, no normalization is needed and the two systems' incomparable score scales stop mattering. The constant $k$ damps the influence of the very top ranks: without it, rank 1 would be worth twice rank 2, which over-rewards a single retriever's confident-but-wrong first result. A document appearing at a decent rank in *both* lists beats a document ranked first in one and absent from the other, which is exactly the consensus behaviour you want.

**A worked example.** BM25 returns [d3, d1, d7, d2]; dense returns [d7, d9, d1, d4]. With $k=60$: d7 scores $\frac{1}{63}+\frac{1}{61}=0.03227$, d1 scores $\frac{1}{61}+\frac{1}{63}=0.03200$, and d3 — ranked first by BM25 but absent from the dense list — scores only $\frac{1}{61}=0.01639$. So the fused order is d7, d1, d3, and the two documents both systems liked outrank the one only BM25 found, despite that one having been rank 1. *(Computed directly; d7 edges d1 because it holds ranks 3 and 1 versus d1's 2 and 3.)*

**Why the combination beats either part — the actual mechanism.** The two retrievers fail on *disjoint* query types. BM25 fails on paraphrase: query "can't sign in," document "authentication failure," zero term overlap, zero score. Dense fails on rare exact strings: an error code like `ERR_2049`, a part number, a person's surname, or a newly-coined product name is either out of vocabulary or embedded near other alphanumeric noise. Because the failures are uncorrelated, the union of the two candidate sets has substantially higher recall than either alone, and that recall gain is what the re-ranker then converts into precision. Ensembling only helps when the members fail differently, and here they demonstrably do.

**Follow-up:** *Do you need to run both retrievers on every query?* Not necessarily. Query routing — classify the query as keyword-like (contains quoted strings, IDs, code identifiers) versus natural-language and weight accordingly — saves latency. But hybrid on everything is the safer default, since the classifier becomes one more thing that can be wrong, and the cost of BM25 is small compared to the generation call that follows.

> **Saying it out loud.** Hybrid means running BM25 and dense retrieval in parallel and merging the two result lists. The reason it works is that they fail on different queries — BM25 whiffs on paraphrase, where the user says "can't sign in" and the doc says "authentication failure," and dense whiffs on rare exact strings like error codes or part numbers that embeddings smear together. Uncorrelated failures are exactly when ensembling pays. For merging I'd default to Reciprocal Rank Fusion rather than a weighted score sum, because BM25 scores are unbounded and query-dependent while cosine similarities sit in a narrow band, so normalizing them onto a common scale is genuinely fiddly. RRF just uses ranks — one over sixty-plus-rank, summed — so the scale problem disappears, and documents both retrievers liked naturally float to the top.

---

### Q70: When to use BM25 vs Dense vs Hybrid?

**Answer:**

**BM25:**
- Keyword queries, exact matching
- Fast, interpretable
- Start here

**Dense:**
- Semantic queries, synonyms
- Related concepts
- When embeddings available

**Hybrid:**
- Production systems
- Mixed query types
- Best overall performance
- Industry standard

**Recommendation:**
- Start: BM25
- Add: Dense if semantic needed
- Production: Hybrid

**Choosing by query shape, with the reasoning attached.**

The right way to answer this is not a preference ordering but a mapping from *query characteristics* to method, because that is what you would actually reason about on a real system.

**Reach for BM25 when** queries carry rare, exact tokens whose identity matters — error codes, SKUs, legal citations, API names, surnames, chemical formulae. Also when the corpus is small or highly specialized so there is no good embedding model for it, when you need to explain to a user or an auditor *why* a document ranked where it did, when the corpus updates constantly and re-embedding is a burden, or when you simply have no GPU. Its cost profile is unbeatable: an inverted index, no model, milliseconds.

**Reach for dense when** users describe what they want rather than naming it, when the corpus and the queries use different vocabularies for the same concepts, when you need cross-lingual matching (a multilingual embedding model retrieves German documents for an English query, which BM25 cannot do at all), or when queries are long and conversational so exact-term overlap is diluted.

**Reach for hybrid when** you have real users, because real query logs contain both kinds and usually in the same session. This is the production default and the answer expected for a design question.

**The honest caveats.** Hybrid is not free: two indexes to build, keep in sync, and monitor; a fusion step with parameters to tune; and roughly double the retrieval infrastructure. On a genuinely homogeneous workload — say, a semantic search over support articles where users never type identifiers — dense alone may match hybrid at half the operational cost. Measure before assuming.

**Where re-ranking sits in this decision.** Any of the three retrieval modes can feed a cross-encoder re-ranker, and in most published comparisons adding a re-ranker to BM25 alone beats dense retrieval without one. If you are choosing where to spend one unit of effort, a re-ranker is frequently the better buy than a second retriever.

**Follow-up:** *What is the fastest useful thing to try first on a new corpus?* BM25, on the same day, with no training. It gives you a baseline number, and — more usefully — reading its failures tells you exactly what dense retrieval would need to fix, which is a far better guide than starting with the fanciest option and having nothing to compare against.

> **Why the interviewer asks this.** They want evidence you pick methods from problem characteristics rather than from recency, and the tell is whether you can name a case where the simple method wins.

> **Saying it out loud.** I'd pick based on what the queries look like. If they contain exact tokens that matter — error codes, part numbers, API names — BM25, because embeddings blur those together and BM25 nails them. If people describe what they want in their own words and the corpus uses different vocabulary, dense, because that's the paraphrase gap embeddings close. If it's real users, hybrid, because real query logs contain both in the same session. And I'd start with BM25 on day one regardless, because it needs no training and its failures tell you precisely what the dense side would have to fix. One thing worth saying: adding a cross-encoder re-ranker on top of plain BM25 often beats dense retrieval with no re-ranker, so if I only had budget for one addition, it'd be the re-ranker.

---

See `39_rag_retrieval_augmented_generation/retrieval_methods.md` for detailed explanations!

---

## NLP Problems: Standard Solution Procedures

### Q71: What's the standard procedure for text classification?

**Answer:**

**Phase 1: Data Preparation**
- Collect labeled data, handle class imbalance
- Preprocessing: Lowercase, remove special chars, tokenize
- Split: Train/Validation/Test

**Phase 2: Feature Extraction**
- **Small data**: TF-IDF + Naive Bayes/SVM
- **Medium data**: Word embeddings + LSTM/CNN
- **Large data**: Fine-tuned BERT

**Phase 3: Model Selection**
- < 10K: TF-IDF + SVM
- 10K-100K: Embeddings + Neural or XGBoost
- > 100K: Fine-tuned BERT

**Phase 4: Training**
- Traditional ML: Hyperparameter tuning
- Neural: Adam optimizer, dropout, early stopping
- BERT: Learning rate 2e-5, 3-5 epochs

**Phase 5: Evaluation**
- Metrics: Accuracy, F1, Precision, Recall
- Multi-class: Macro/Micro F1
- Confusion matrix for analysis

**Phase 6: Deployment**
- API endpoint
- Monitoring, drift detection
- A/B testing

**What the phase list leaves out, and what to say first.**

Before any of the six phases, spend a sentence on the two questions that determine everything downstream: what exactly is a label here, and what does a mistake cost? Multi-class (one label per document) and multi-label (any number of labels per document) need different output layers — softmax versus per-class sigmoid — different losses, and different metrics, and choosing wrong is a rewrite rather than a tweak. And an asymmetric error cost, say fraud detection where a miss costs a thousand times a false alarm, means accuracy is meaningless and your operating threshold is a business decision rather than 0.5.

**Handling class imbalance, since the answer above names it without solving it.** Four levers, in the order I would try them. Move the decision threshold — train normally, then pick the threshold on a validation set to hit your target precision or recall; this is free and often sufficient, and people skip straight past it. Weight the loss by inverse class frequency, which is one argument in most libraries. Resample, oversampling the minority or undersampling the majority, remembering to resample only the training split — resampling before the split leaks and produces a beautiful, fictional validation score. And for extreme imbalance use *focal loss*, which multiplies the cross-entropy of each example by $(1-p_t)^{\gamma}$ where $p_t$ is the predicted probability of the true class, so easy well-classified examples contribute almost nothing and the gradient concentrates on hard ones.

**On the data-size thresholds.** The "under 10K use TF-IDF and SVM, over 100K fine-tune BERT" guidance is a reasonable starting heuristic, but it is worth saying out loud that transfer learning has shifted those boundaries down: fine-tuning a pretrained encoder frequently wins at 1,000 labelled examples, sometimes at a few hundred, precisely because the model arrives already knowing the language and only has to learn the task. And in 2026 the realistic first baseline for a new classification task is often a zero-shot or few-shot LLM prompt, which needs no labelled data at all — useful both as a baseline and as a way to generate labels to distil into a small, cheap model you can actually serve. *This tradeoff shifts with model cost; re-check it rather than assuming.*

**One preprocessing correction.** "Lowercase, remove special characters, remove stopwords" is inherited from the TF-IDF era and is actively harmful for a pretrained Transformer. Those models were pretrained on natural text with casing and punctuation intact, and their subword tokenizers depend on it — casing distinguishes "Apple" from "apple," punctuation carries sentence structure, and stopwords carry syntax. For a Transformer, do essentially nothing beyond normalizing whitespace and stripping markup. Heavy preprocessing belongs with bag-of-words models.

**Follow-up:** *Your test accuracy is 94% but production performance is poor. Where do you look first?* Label leakage and distribution shift, in that order. Leakage: some feature (a template phrase, an ID, a timestamp artefact) correlates with the label in your dataset and not in the wild — check by looking at what the model attends to on its most confident predictions. Shift: your training data was sampled from a different period, source, or user population than production. Both are far more common than the model being wrong.

> **Saying it out loud.** Before touching a model I'd pin down two things: is it single-label or multi-label, because that changes the output layer and the metrics, and what does a mistake cost, because that sets the threshold. Then the pipeline is the usual one — split first, so nothing leaks; establish a dumb baseline like TF-IDF plus logistic regression so you know what "good" means; then fine-tune a pretrained encoder, which these days wins even at a thousand labelled examples. For imbalance, I'd move the threshold before I did anything clever, then loss weighting, then resampling — and only resample the training split. And I'd skip the old lowercase-and-strip-stopwords routine for a Transformer; those models want the casing and punctuation they were pretrained on.

---

### Q72: How do you solve NER? What's the standard approach?

**Answer:**

**Phase 1: Data Format**
- BIO tagging: B-PER, I-PER, O
- Label each token

**Phase 2: Features**
- Word features: Current, previous, next word
- Context: Surrounding words, position
- Embeddings: Word + character-level

**Phase 3: Model**
- **CRF**: Traditional, interpretable
- **BiLSTM-CRF**: Better performance
- **Fine-tuned BERT**: State-of-the-art

**Phase 4: Training**
- CRF: Maximum likelihood, L-BFGS
- BiLSTM-CRF: Adam, dropout 0.5
- BERT: Learning rate 3e-5, token classification

**Phase 5: Evaluation**
- Entity-level F1 (exact match)
- Token-level F1
- Per entity type

**Phase 6: Challenges**
- OOV words: Character embeddings, subword
- Nested entities: Multi-label, span-based
- Ambiguity: Context, larger window

**Why NER is a sequence labelling problem and not just per-token classification.**

*NER* — named entity recognition — assigns each token a label saying whether it starts, continues, or falls outside an entity of some type. The BIO scheme spells that out: `B-PER` begins a person, `I-PER` continues one, `O` is outside any entity. So "Barack Obama visited Paris" becomes `B-PER I-PER O B-LOC`. The reason for separate B and I tags is adjacency: without them, "Barack Obama Michelle Obama" would be indistinguishable from one four-token person and two two-token people.

The crucial structural point is that the labels are *not independent*. `I-PER` cannot legally follow `O` or `B-LOC`, because a continuation must continue something of the same type. A plain per-token softmax has no way to know this and will happily emit `O I-PER`, an ill-formed sequence you then have to patch up in post-processing.

**What the CRF layer actually does.** A *conditional random field* on top of the encoder adds a learned transition matrix $A$, where $A_{ij}$ scores moving from label $i$ to label $j$. The score of a whole label sequence is the sum of per-token emission scores from the encoder plus the transition scores between consecutive labels, and training maximizes the log-probability of the true sequence normalized over *all* sequences — computed efficiently with the forward algorithm, since enumerating $L^{n}$ sequences is impossible. At inference, Viterbi decoding finds the highest-scoring valid sequence in $O(nL^{2})$ by dynamic programming. The net effect is that impossible transitions learn large negative scores and the model stops emitting malformed output.

Worth knowing the modern nuance: with a strong pretrained encoder, the CRF layer's benefit shrinks considerably, because the Transformer's context already captures most of the constraint. Many production systems now use plain token classification plus a cheap post-processing pass to repair invalid transitions. So "always add a CRF" is out of date; "add a CRF when the tag set is large and structured, or the encoder is weak" is current.

**The subword alignment problem, which is the actual implementation pitfall.** BERT-family tokenizers split words into subwords, so "Washington" might become `Wash ##ing ##ton` — three tokens for one word-level label. The standard convention is to label the first subword of each word and set the rest to $-100$, the index PyTorch's cross-entropy ignores, so they contribute no loss. At prediction time you take the first subword's label as the word's label. Getting this wrong is the single most common source of a NER model that trains fine and scores terribly, and mentioning it signals implementation experience.

**Why entity-level F1 is the metric, not token accuracy.** Token accuracy is inflated by the `O` class, which is typically 85–95% of tokens — predicting `O` everywhere scores 90% and finds nothing. Entity-level F1 counts a prediction correct only when both the span boundaries and the type match exactly, which is what a downstream consumer actually needs. Partial matches score zero, which is strict but right: half a person's name is not a usable extraction.

**Follow-up:** *Nested entities — "Bank of America" contains "America" as a location.* Flat BIO cannot represent overlap at all. The standard alternatives are span-based classification (enumerate candidate spans up to some length and classify each independently, which naturally allows overlap) or a machine-reading-comprehension formulation where you ask one question per entity type and extract all matching spans. Both cost more compute; span enumeration is $O(n^{2})$ in sentence length.

> **Why the interviewer asks this.** NER is the standard example of structured prediction, so the question is really "do you understand that the outputs are dependent on each other?"

> **Saying it out loud.** NER is sequence labelling — you tag every token with BIO tags like B-PER, I-PER, O — and the key point is that the labels aren't independent. I-PER can't follow O, so you either add a CRF layer that learns transition scores and decodes with Viterbi, or, with a strong pretrained encoder, use plain token classification and repair invalid transitions in post-processing, which is what a lot of production systems do now. The implementation detail that trips people up is subword alignment: the tokenizer splits words into pieces, so you label the first subword of each word and mask the rest with minus one hundred so they don't contribute loss. And I'd evaluate with entity-level F1, exact span and type match — token accuracy is useless because ninety percent of tokens are O.

---

### Q73: What's the standard procedure for question answering?

**Answer:**

**Phase 1: Data Format**
- SQuAD: Context + Question → Answer span
- Extractive: Start/end positions

**Phase 2: Model**
- **BERT-based**: Standard approach
  - Input: [CLS] question [SEP] context [SEP]
  - Two heads: Start position, End position
  - Fine-tune BERT

**Phase 3: Training**
- Load pre-trained BERT
- Add QA head (start/end logits)
- Loss: Start loss + End loss
- Learning rate: 3e-5, batch 16-32, 2-3 epochs

**Phase 4: Long Contexts**
- Sliding window
- Hierarchical (paragraph ranking)
- Long-context models

**Phase 5: Evaluation**
- EM (Exact Match)
- F1 (token overlap)
- Per question type

**Phase 6: Production**
- Retrieval for open-domain
- Re-ranking
- Ensemble models

**How the span-prediction head actually works.**

This is the mechanical part worth walking through, because "two heads: start and end" is where the answer above stops and the follow-up begins.

Feed `[CLS] question [SEP] context [SEP]` through the encoder to get one vector $h_i$ per token. Learn two vectors, $s$ and $e$, each of size $d$. The start logit for token $i$ is the dot product $s^{\top}h_i$, and the end logit is $e^{\top}h_i$. Softmax each over the sequence to get $P_{\text{start}}(i)$ and $P_{\text{end}}(j)$, and train with cross-entropy against the true start and end indices, averaging the two losses. That is the entire QA head — two vectors, roughly $2d$ parameters on top of the encoder.

At inference you want the best *span*, not the best start and the best end independently, since the argmax start and argmax end can be inconsistent (end before start, or a 400-token span). The standard decode: take the top $n$ start candidates and the top $n$ end candidates (typically $n=20$), form all pairs, discard any with $j < i$ or $j - i + 1 >$ max answer length or with either index in the question rather than the context, and pick the surviving pair with the highest $P_{\text{start}}(i) \cdot P_{\text{end}}(j)$. Then map token indices back to character offsets in the original text — the tokenizer's offset mapping is what makes this possible, and getting it wrong yields answers that are correct but sliced one character short.

**Handling unanswerable questions.** SQuAD 2.0 added questions with no answer in the passage, which is closer to reality. The standard trick is to let the span point at the `[CLS]` token to mean "no answer." At inference you compare the best real span's score against the null score $s^{\top}h_{\text{[CLS]}} + e^{\top}h_{\text{[CLS]}}$, and abstain when the null score wins by more than a threshold tuned on the validation set. That threshold is a real dial: it trades false answers against false abstentions, and where you set it is a product decision.

**The sliding window, in detail.** For a context longer than the model's limit, split it into overlapping windows — a common setting is a 384-token stride with 128 tokens of overlap — run each window with the same question prepended, and take the highest-scoring span across all windows. The overlap exists so that an answer straddling a window boundary appears whole in at least one window. Scores across windows are comparable because they come from the same softmax-normalized head, though they are only loosely calibrated, which is one reason very long contexts favour a retrieve-then-read pipeline over brute-force windowing.

**Extractive versus generative, which is the 2026 framing.** Everything above is *extractive* QA: the answer must be a contiguous substring of the passage. That is a genuine feature — the answer is guaranteed grounded, it cannot hallucinate, and it comes with exact character offsets for highlighting. Generative QA with an LLM handles multi-span answers, questions requiring synthesis across sentences, and reformulation into natural phrasing, at the cost of that grounding guarantee. Modern production systems mostly generate, then apply the verification machinery from Q60 to recover some of what extraction gave for free. Say which one you mean; interviewers notice when a candidate conflates them.

**Why EM and F1 both.** *Exact match* is a strict string equality after normalization (lowercase, strip articles and punctuation). *F1* is computed over the bag of tokens shared between prediction and gold, so "in 1969" against "1969" scores EM 0 but F1 0.67. Reporting both tells you whether errors are boundary sloppiness or genuine misunderstanding, and the gap between them is itself informative.

**Follow-up:** *Why not just fine-tune the model to output the answer text directly?* You can, and generative models do. But extractive spans give you free provenance — you know exactly where the answer came from — which matters enormously in any setting where a human has to check the work.

> **Saying it out loud.** For extractive QA you feed CLS, question, SEP, context, SEP through an encoder and put two tiny heads on top: one vector that scores each token as a possible answer start, one that scores it as a possible end. Softmax each over the sequence, train with cross-entropy against the true indices. At inference you don't just take the best start and best end separately, because they can be inconsistent — you take the top twenty of each, form valid pairs where end comes after start and the span isn't too long, and pick the highest product of probabilities. For unanswerable questions you let the span point at the CLS token and compare that null score against the best real span. And for long passages you use overlapping sliding windows so an answer straddling a boundary shows up intact in at least one.

---

### Q74: How do you build a machine translation system?

**Answer:**

**Phase 1: Data**
- Parallel corpus (millions of pairs)
- High-quality translations
- Domain match if possible

**Phase 2: Preprocessing**
- Sentence segmentation
- Subword tokenization (BPE, SentencePiece)
- Handle rare words

**Phase 3: Model**
- **Transformer**: State-of-the-art
- Encoder-Decoder architecture
- Multi-head attention

**Phase 4: Training**
- Pre-train on large corpus (optional)
- Fine-tune on translation data
- Learning rate: 1e-4, warmup
- Decoding: Beam search with length penalty

**Phase 5: Evaluation**
- BLEU score (primary)
- METEOR, human evaluation

**Phase 6: Production**
- Multilingual models
- Transfer learning for low-resource
- Back-translation for data augmentation

**Why subword tokenization is the load-bearing choice.**

Of everything in that phase list, subword tokenization is the one that most determines whether a translation system works, so it is worth being able to explain rather than name.

Word-level vocabularies fail on translation for a specific reason: morphologically rich languages (Finnish, Turkish, German compounds) generate effectively unbounded vocabularies, so any fixed word list produces a flood of unknown tokens, and an unknown token in the source is information the decoder can never recover. Character-level avoids that but makes sequences four to five times longer, which given attention's quadratic cost (Q89) is expensive, and it forces the model to relearn word structure from scratch.

*Byte Pair Encoding* splits the difference. Start with a vocabulary of individual characters. Count all adjacent symbol pairs in the corpus, merge the most frequent pair into a new symbol, and record the merge. Repeat until the vocabulary reaches the target size, typically 32K per language or 32K–64K shared. The learned result is that frequent words survive as single tokens while rare words decompose into meaningful pieces — "unhappiness" into "un", "happi", "ness" — so the model can handle a word it has never seen by composing parts it has. SentencePiece does the same thing but treats the input as a raw byte or Unicode stream including spaces (encoded as `▁`), so it needs no language-specific pre-tokenizer and detokenizes losslessly — which matters for languages like Japanese and Thai that do not delimit words with spaces at all. Byte-level BPE goes further and guarantees no token is ever truly unknown, since every input is representable as bytes.

For translation specifically, a *shared* source-target vocabulary is standard, because it lets the model copy names, numbers and technical terms across languages directly and enables tying the embedding matrices.

**Beam search, since "beam search with length penalty" hides the mechanics.** Greedy decoding picks the highest-probability token at each step and can be trapped by a locally attractive choice that dooms the rest of the sentence. Beam search keeps the $k$ highest-scoring partial sequences (beam width 4–5 is standard for translation), extends each by every possible next token, and re-prunes to $k$. Because sequence log-probability is a sum of negative numbers, longer sequences score worse automatically, so beam search left alone produces truncated output — hence a length penalty dividing the score by $((5+|Y|)/6)^{\alpha}$ with $\alpha$ around 0.6, the Google NMT formulation. Worth adding that beam search is standard for translation and *not* for open-ended generation, where it produces bland, repetitive text: translation has a roughly correct answer, so searching for the highest-probability output is appropriate, while open-ended text has many good answers and the highest-probability one is generic.

**On BLEU, with the caveat.** BLEU is a modified n-gram precision against one or more references, with a brevity penalty to stop the model gaming precision by emitting three confident words. It correlates with human judgement well enough for tracking a single system over time and badly enough that cross-system comparisons are unreliable, especially between systems of different types — and it is sensitive to tokenization, which is why sacreBLEU (which fixes the tokenization) is the reportable form. COMET and other learned neural metrics correlate substantially better with human judgement and are the current default for serious evaluation. *Metric practice here moves; check what the current WMT findings recommend before committing to one.*

**Follow-up:** *What do you do for a low-resource language pair with 50K sentence pairs?* Do not train from scratch. Start from a large pretrained multilingual model and fine-tune, so the language pair benefits from transfer across the other hundred languages. Then augment with back-translation — train a reverse-direction model, use it to translate abundant monolingual target-language text into synthetic source text, and add those pairs to training. It works because the synthetic noise sits on the *source* side where the encoder is more robust, while the target side stays fluent human text.

> **Why the interviewer asks this.** Machine translation is the original sequence-to-sequence task, so it is a natural vehicle for probing tokenization, decoding and evaluation all at once.

> **Saying it out loud.** The pieces are parallel data, subword tokenization, an encoder-decoder Transformer, and beam search decoding. The part I'd emphasize is tokenization — word-level vocabularies drown in unknowns for morphologically rich languages, character-level makes sequences far too long, so you use BPE or SentencePiece, which keeps frequent words whole and splits rare ones into reusable pieces. Usually a shared source-target vocabulary, so names and numbers copy across for free. For decoding, beam search with a length penalty, because without the penalty the sum of log-probs favours short output and you get truncated sentences. And I'd report sacreBLEU for comparability but lean on a learned metric like COMET, because BLEU correlates with human judgement well enough to track one system over time and poorly enough to mislead across systems.

---

### Q75: What's the standard approach for text summarization?

**Answer:**

**Two Types:**

**Extractive:**
1. Score sentences (position, TF-IDF, similarity)
2. Select top-K sentences
3. Order by original position
4. Methods: TextRank, BERT-based scoring

**Abstractive:**
1. **Model**: Fine-tuned BART/T5
2. **Training**: Encoder-Decoder, max source 1024, target 128
3. **Generation**: Beam search, length penalty, repetition penalty
4. **Post-processing**: Remove repetition, fix grammar

**Evaluation:**
- ROUGE-1/2/L (primary)
- BLEU, human evaluation

**Challenges:**
- Long documents: Hierarchical encoding
- Factual consistency: Fact checking
- Repetition: Repetition penalty

**Extractive versus abstractive, and why the choice is a risk decision.**

The real distinction is not technique but guarantee. Extractive summarization copies sentences verbatim, so it *cannot* state a fact the source does not contain — factual consistency is structural, not hoped for. Abstractive summarization generates new text, so it reads far better and can compress across sentences, but it can and does invent. In regulated or high-stakes settings that guarantee is often worth more than the fluency, and saying so shows you are thinking about deployment rather than benchmarks.

**How TextRank works, since it is named without explanation.** Build a graph with one node per sentence and edge weights equal to the similarity between sentence pairs — originally token overlap normalized by length, now more often embedding cosine. Run PageRank over that graph: each sentence's score is a damped sum of its neighbours' scores weighted by edge strength, iterated to convergence. Sentences that are similar to many other well-connected sentences score highest, which operationalizes "central to what this document is about." It needs no training data at all, which makes it a genuinely useful baseline. Follow it with a redundancy filter — *maximal marginal relevance*, which greedily picks the next sentence maximizing relevance minus $\lambda$ times maximum similarity to what you have already selected — or your summary will be three paraphrases of the same central idea.

**Factual consistency, since "fact checking" is listed without a method.** The measurable approaches are worth naming. Entailment-based: split the summary into claims and run an NLI model with the source as premise, scoring the fraction entailed — the same machinery as RAG faithfulness in Q60. QA-based (QAGS/FEISQA family): generate questions from the summary, answer them against both the summary and the source, and compare the answers; disagreement localizes the hallucination to a specific fact. Both correlate with human judgement far better than ROUGE, which is essentially blind to factuality — a summary can hit high ROUGE while swapping a name or negating a claim, because the n-grams still overlap.

**Handling long documents.** Three approaches with different tradeoffs. *Hierarchical*: summarize each section, then summarize the summaries — cheap and parallel, but cross-section connections are lost at the first level. *Refine / rolling*: carry a running summary and update it chunk by chunk — preserves narrative order, but errors compound and later chunks get disproportionate influence. *Long-context model in one pass* — simplest and now often feasible, but subject to the lost-in-the-middle position effect from Q59, so material buried in the middle of a very long document is systematically under-represented in the summary. Knowing that failure mode is more useful than knowing the three names.

**On evaluation.** ROUGE-1, ROUGE-2 and ROUGE-L measure unigram, bigram, and longest-common-subsequence overlap with a reference summary. They are recall-oriented and they reward copying the reference's wording, which means they systematically favour extractive systems and penalize good paraphrase. Report them for continuity with the literature, but pair them with a factual-consistency score and a small human or LLM-judged relevance rating, because ROUGE alone will happily rank a fluent, wrong summary above a clumsy, correct one.

**Follow-up:** *How do you control summary length?* For abstractive models, three levers: a `max_length` / `min_length` constraint at decode time, which is blunt and can cut mid-sentence; length control in the prompt or as a control token seen during training, which is softer and more natural; and a length penalty in beam search, which biases rather than enforces. For extractive, it is just how many sentences you select, which is the one case where the control is exact.

> **Saying it out loud.** Two families. Extractive picks sentences out of the source — something like TextRank, which builds a sentence similarity graph and runs PageRank on it to find the central sentences, then a redundancy filter so you don't return three paraphrases of the same point. The nice property is that it physically can't hallucinate, because every word came from the source. Abstractive generates new text with a fine-tuned encoder-decoder or an LLM; it reads much better and can compress across sentences, but it can invent facts. So the choice is really a risk decision. On evaluation I'd report ROUGE for continuity but not trust it — it's blind to factuality, so a summary that swaps a name still scores well. I'd add an entailment-based consistency check and a small human-rated set.

---

See `36_nlp_basics/nlp_problems_detailed.md` for complete procedures for all NLP problems!

---

## Foundation Models: Evolution from BERT to GPT-4

### Q76: How did we evolve from BERT to modern foundation models like GPT-4?

**Answer:**

**Phase 1: BERT (2018)**
- **Bidirectional**: Reads text both directions
- **Encoder-only**: Good for understanding
- **Pre-training + Fine-tuning**: Train on large corpus, fine-tune on tasks
- **Limitation**: Can't generate text, needs fine-tuning per task

**Phase 2: GPT-2 (2019)**
- **Generative**: Can generate coherent text
- **Decoder-only**: Autoregressive (left-to-right)
- **Zero-shot**: No fine-tuning for some tasks
- **Limitation**: Unidirectional, limited context

**Phase 3: GPT-3 (2020)**
- **Massive scale**: 175B parameters
- **In-context learning**: Few-shot without gradient updates
- **Scaling laws**: Performance ∝ (Size)^α × (Data)^β
- **Emergent abilities**: Arithmetic, code, reasoning at scale
- **Impact**: Proved scaling works, foundation for modern LLMs

**Phase 4: InstructGPT (2021)**
- **RLHF**: Reinforcement Learning from Human Feedback
- **Alignment**: Align model with human preferences
- **Instruction following**: Better at following instructions
- **Impact**: Foundation for ChatGPT, RLHF becomes standard

**Phase 5: ChatGPT (2022)**
- **Conversational**: Natural dialogue interface
- **RLHF**: Aligned with human preferences
- **Multi-turn**: Maintains context
- **Impact**: Viral adoption, paradigm shift to assistants

**Phase 6: GPT-4 (2023)**
- **Multimodal**: Text + images
- **Better reasoning**: Improved logical reasoning
- **Large context**: 8K → 32K → 128K tokens
- **State-of-the-art**: Best performance on many tasks

**Key Paradigm Shifts:**
1. Task-specific → General (single model for many tasks)
2. Fine-tuning → Prompting (in-context learning)
3. Understanding → Generation (decoder architectures)
4. Supervised → Self-supervised (pre-training)
5. Capability → Alignment (RLHF, safety)

**A date correction and a currency note.**

Two things to fix in the timeline above. *InstructGPT is dated 2021; the models and the paper both landed in early 2022* — OpenAI released the InstructGPT models in January 2022 and the paper (*Training language models to follow instructions with human feedback*) in March 2022. The underlying human-preference work goes back to 2017–2020, which is probably where the 2021 came from, but the milestone itself belongs in 2022.

Second, the timeline stops at GPT-4 in 2023. Since then the field has moved through several further phases that any 2026 interviewer will expect you to at least gesture at: instruction-tuned open-weight model families becoming genuinely competitive; mixture-of-experts architectures becoming standard for frontier-scale models, where only a fraction of parameters activate per token so capacity grows without proportional inference cost; natively multimodal training (audio, image and video in the same model from the start rather than a vision encoder bolted on); context windows moving from 128K to the million-token range; and explicit *reasoning* models trained to spend extra compute at inference time producing intermediate reasoning before answering — which shifted the field's attention from training-time scaling to *inference-time* scaling. *Everything in this paragraph is time-sensitive and specific model names date within months; treat it as a direction of travel rather than a fact sheet, and verify current specifics.*

**The mechanism behind the biggest jump, stated as one idea.** If you have to compress the whole evolution into a single sentence, it is this: **every NLP task was reformulated as text prediction.** BERT still needed a new output head and a new fine-tuning run per task. GPT-2's contribution was demonstrating that translation, summarization and question answering all fit inside "predict the next token" if you simply write the task into the input. Once that reframing holds, scale becomes the lever, because improving one objective improves every task at once, and you no longer need labelled data per task. Everything after — in-context learning, instruction tuning, RLHF, tool use — is a refinement of that single move.

**Why decoder-only won, mechanically.** Three reasons that are worth separating. Training efficiency: a decoder gets a prediction target at every one of $n$ positions, whereas BERT's masked language modelling only supervises the 15% of positions that were masked, so per token of data the decoder extracts several times more signal. Generality: generation subsumes understanding — you can classify by generating a label — but the reverse does not hold. And the pretrain/fine-tune mismatch: BERT sees `[MASK]` tokens during pretraining that never occur at fine-tuning time, whereas a decoder's training and inference objectives are identical.

**What was actually lost.** Be honest about the tradeoff, because it makes the answer credible. Causal attention means each token only sees its left context, which is strictly less information than BERT's bidirectional view for pure *encoding* tasks. That is why encoder models remain the right tool for embeddings, retrieval, re-ranking and high-throughput classification, and why BERT-family models are still running in enormous volume in production in 2026 — they are smaller, faster, cheaper, and better at producing a single vector representing a whole text. "BERT was superseded" is a headline; "BERT moved to a different part of the stack" is the accurate version.

**Follow-up:** *What ended the pure-scaling era?* Chinchilla (2022) showed the big models of the day were badly undertrained for their size — the compute-optimal ratio is roughly 20 training tokens per parameter, so GPT-3's 175B parameters on 300B tokens was far off optimal. That redirected effort from parameter count to data quality and quantity. Since then the binding constraints have shifted again toward high-quality data availability and toward inference cost, which is what motivates mixture-of-experts, distillation, and spending compute at inference time instead of training time.

> **Why the interviewer asks this.** They want to know whether you can tell a causal story with mechanisms, or only recite a sequence of model names and parameter counts.

> **Saying it out loud.** The single move that mattered was reframing every task as next-token prediction. BERT was bidirectional and great at understanding, but you needed a new head and a fine-tuning run per task. GPT-2 showed that if you just write the task into the input as text, one model does translation, summarization, and QA without task-specific training. Once that holds, scale is the lever, because improving one objective improves everything at once. GPT-3 proved few-shot prompting works at scale, InstructGPT and RLHF in 2022 made models actually follow instructions rather than just continue text, and GPT-4 added multimodality and much better reasoning. Since then the interesting shift has been from scaling training compute to scaling inference compute — models that think longer before answering.

---

### Q77: What are scaling laws? How do they explain the success of large models?

**Answer:**

**Neural Scaling Laws:**
```
Performance = f(Model Size, Data Size, Compute)

Performance improves predictably with:
- Model size (parameters)
- Training data size
- Compute budget
```

**Key Findings:**

**1. Power Law Relationship:**
- Performance ∝ (Model Size)^α
- α ≈ 0.076 (diminishing returns but still improves)
- Larger models = better performance

**2. Data Scaling:**
- Larger models need more data
- Optimal data size ∝ Model size
- More data = better performance

**3. Compute Scaling:**
- More compute = better performance
- Optimal allocation: Scale model, data, compute together

**4. Predictable Improvements:**
- Can predict performance before training
- Helps with planning and resource allocation

**Implications:**
- **Bigger is better**: Larger models perform better
- **Data matters**: Need more data for larger models
- **Massive compute**: Requires huge compute budgets
- **Predictable**: Can estimate performance

**Why It Matters:**
- Explains why GPT-3 succeeded
- Guides model development
- Shows path to better models
- Justifies investment in scale

**What the law actually says, with the exponents made precise.**

The formula in the answer is loosely stated in a way that inverts the meaning, so it is worth writing carefully. Scaling laws are about *loss*, and loss goes **down** as a power law:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076,$$

where $N$ is non-embedding parameter count and $L$ is cross-entropy loss in nats per token (Kaplan et al., 2020). So "performance $\propto$ size$^{\alpha}$" should read "**loss** $\propto$ size$^{-0.076}$." The exponent being small is the important part: because $0.076$ is tiny, you need roughly a $10\times$ increase in parameters to cut loss by a factor of $10^{0.076} \approx 1.19$, i.e. about 16%. Scaling works, and it is brutally expensive per unit of improvement. Analogous power laws hold in dataset size $D$ with $\alpha_D \approx 0.095$ and in compute $C$ with $\alpha_C \approx 0.050$, and the three are only independent while the other two are not the bottleneck.

**Chinchilla, which changed the practical conclusion and is missing above.** Kaplan's original analysis suggested that given more compute you should mostly grow the model. Hoffmann et al. (2022) redid the experiment more carefully and found $N$ and $D$ should scale roughly *equally*: compute-optimal training uses about **20 tokens per parameter**. By that rule GPT-3 (175B parameters, ~300B tokens, under 2 tokens per parameter) was drastically undertrained — a 70B model trained on 1.4T tokens (Chinchilla) beat it while being 2.5× smaller and far cheaper to serve. This is the single most important correction to the naive "bigger is better" reading and it belongs in any answer to this question.

Note the further practical twist: compute-*optimal* is about training cost only. If you will serve a model to millions of users, inference cost dominates total cost, so it is rational to train a *smaller* model on *far more* tokens than Chinchilla-optimal — well past the point of diminishing training returns — because you pay the training bill once and the inference bill forever. That is why recent small models are trained on token counts hundreds of times their parameter count.

**Why the curve stays straight, and where it must bend.** There is no complete theory. The leading intuitions are that natural language itself has power-law structure (Zipf), and that larger models can resolve finer-grained sub-distributions of the data. The honest position is that scaling laws are a robust *empirical* regularity across many orders of magnitude, not a derived law. And they must break somewhere: the loss cannot fall below the *irreducible entropy* of the text — genuinely unpredictable content, so the fitted forms include an additive constant $L_\infty$ — and high-quality human text is a finite resource, which is the constraint now driving synthetic data, multi-epoch training, and multimodal data.

**Why they matter operationally.** The practical payoff is not philosophical, it is budgetary: fit the curve on small runs — a handful of models spanning two or three orders of magnitude of compute — and extrapolate to predict the loss of a run you have not yet paid for. That is how a training run costing tens of millions of dollars gets approved by people who need a forecast rather than a hope. It also lets you choose hyperparameters (batch size, learning rate) at small scale and transfer them.

**Follow-up:** *Does lower loss actually mean a better product?* Only loosely. Loss is next-token cross-entropy averaged over a pretraining distribution, and downstream benchmark scores are a noisy, sometimes discontinuous function of it (Q80). A model can improve on loss while getting worse on the thing you care about, and post-training — instruction tuning, preference optimization — moves downstream quality substantially without moving pretraining loss at all.

> **Why the interviewer asks this.** They are checking whether "scale it up" is a slogan for you or a quantitative statement you can reason about, and the Chinchilla correction is the tell.

> **Saying it out loud.** Scaling laws say that test loss falls as a power law in model size, data size, and compute — smooth, straight lines on a log-log plot over many orders of magnitude. The exponent is small, about 0.076 for parameters, which means ten times the model gets you roughly a sixteen percent loss reduction; it works, but it's expensive per unit of improvement. The important refinement is Chinchilla — the original analysis said grow the model, and Chinchilla showed you should grow parameters and data together, roughly twenty tokens per parameter. By that standard GPT-3 was badly undertrained. The practical value is that you fit the curve on small cheap runs and extrapolate, which is how you justify a nine-figure training run before spending the money.

---

### Q78: What is in-context learning? How does it differ from fine-tuning?

**Answer:**

**In-Context Learning:**
- Model learns from examples in the prompt
- No gradient updates
- No weight changes
- Same model for all tasks

**Types:**

**Zero-shot:**
```
"Translate to French: hello →"
Model generates: "bonjour"
```

**One-shot:**
```
"Translate to French: hello → bonjour, cat →"
Model generates: "chat"
```

**Few-shot:**
```
"Translate to French: hello → bonjour, cat → chat, dog →"
Model generates: "chien"
```

**Fine-tuning:**
- Update model weights
- Task-specific model
- Requires labeled data
- Gradient updates
- Different model per task

**Key Differences:**

| Aspect | In-Context Learning | Fine-tuning |
|--------|---------------------|-------------|
| **Weight Updates** | No | Yes |
| **Data** | Examples in prompt | Labeled dataset |
| **Model** | Same for all tasks | Different per task |
| **Flexibility** | Easy to change | Need retraining |
| **Performance** | Good for many tasks | Best for specific task |

**Why In-Context Learning Works:**
- Large models have seen similar patterns
- Can generalize from examples
- Emergent ability at scale
- Flexible and efficient

**When to Use:**
- **In-context**: Quick prototyping, many tasks, no labeled data
- **Fine-tuning**: Best performance, specific task, have labeled data

**Why it works, beyond "the model has seen similar patterns."**

That explanation is not wrong but it is not a mechanism, and a good interviewer will push. Three more substantive accounts, worth knowing because none is fully settled:

*Task location rather than task learning.* On this view the model already contains many capabilities from pretraining, and the prompt's examples act as a *selector* that identifies which distribution to condition on, not as training data. The strongest evidence is the striking finding that replacing the labels in few-shot examples with *random* labels often barely hurts performance — which is impossible if the model were learning the input-output mapping from them. What the examples mainly convey is the input distribution, the label space, and the output format.

*Implicit gradient descent.* Several analyses show that a Transformer's forward pass over in-context examples can implement something formally equivalent to a gradient-descent step on a linear model, with the attention mechanism playing the role of the update. Toy models trained on synthetic regression tasks do learn algorithms of this shape. Whether large trained-on-text models actually do this is contested.

*Induction heads.* A concrete, mechanistically-verified circuit: pairs of attention heads that detect a repeated pattern earlier in the context and copy what followed it. They form abruptly during training, and their formation coincides with a visible bump in in-context learning ability. This is the most solid mechanistic evidence available and is the answer to give if pressed for something specific.

The honest summary: in-context learning is a real, measurable phenomenon whose mechanism is partly understood and actively researched. Saying that is stronger than asserting a single explanation.

**Practical facts that make an answer credible.**

Ordering matters — the same examples in a different order can move accuracy by tens of points on some tasks, with a bias toward the label of the final example. Format matters more than correctness, per the random-label result. The gains from more examples saturate quickly, typically by 8 to 32, and long-context models have not changed that as much as expected. And in-context learning re-processes the examples on *every* call, so a system serving a million queries a day pays for the demonstrations a million times — which is precisely the economic argument for eventually distilling into a fine-tuned model.

**The middle ground the table omits.** In 2026 the choice is not binary. *Parameter-efficient fine-tuning*, above all LoRA — freezing the base weights and training a pair of low-rank matrices whose product is added to selected weight matrices — trains well under 1% of the parameters, produces adapters of a few megabytes, and lets you serve many task-specific adapters against one shared base model. That collapses the "different model per task" cost that makes the fine-tuning column look expensive. There is also *prompt caching*, where a provider caches the encoded prefix so repeated demonstrations cost a fraction of full input price, which pushes in the other direction. The real decision is a cost-per-query and quality curve, not a philosophy.

**Follow-up:** *When should you definitely fine-tune?* When you need a specific output format reliably every time; when the task needs more demonstrations than fit comfortably in context; when per-query latency and cost matter at volume and you want to stop paying for a long prompt; or when you need a small model to match a large one on one narrow task, which distillation from the large model's outputs does well.

> **Why the interviewer asks this.** It is the fastest way to find out whether you have deployed LLMs or only used them, because everyone knows the definitions and only practitioners know the ordering effects and the cost math.

> **Saying it out loud.** In-context learning is when you put examples in the prompt and the model adapts, with no weight updates at all — same frozen model for every task. Fine-tuning changes the weights. The mechanism is more interesting than it first looks: there's a well-known result that replacing the labels in your few-shot examples with random ones barely hurts, which means the examples are mostly conveying the format and the label space, not teaching the mapping. So it's closer to locating a capability the model already has than to learning. Practically, the order of examples matters more than people expect, gains saturate around eight to thirty-two examples, and you re-pay for those tokens on every single call — which is the real argument for eventually fine-tuning, ideally with LoRA so you're training under one percent of the parameters.

---

### Q79: Explain RLHF (Reinforcement Learning from Human Feedback). Why is it important?

**Answer:**

**RLHF** aligns language models with human preferences using reinforcement learning.

**Three Steps:**

**Step 1: Supervised Fine-tuning (SFT)**
```
1. Collect human-written prompts and responses
2. Fine-tune base model (GPT-3) on this data
3. Model learns to follow instructions
```

**Step 2: Reward Modeling**
```
1. Collect comparisons: Which response is better?
2. Train reward model to predict human preferences
3. Reward model scores: response_A > response_B
```

**Step 3: Reinforcement Learning (PPO)**
```
1. Generate responses from SFT model
2. Score with reward model
3. Update model to maximize reward
4. Use PPO (Proximal Policy Optimization)
```

**Why Important:**

**1. Alignment:**
- Model behavior ≠ model capability
- Need to align with human values
- Helpful, harmless, honest

**2. Better User Experience:**
- Follows instructions better
- More helpful responses
- Admits mistakes
- Refuses harmful requests

**3. Safety:**
- Can make models safer
- Reduces harmful outputs
- Better control

**Impact:**
- Foundation for ChatGPT
- Standard practice for alignment
- Drives research in alignment
- Better user experience

**Challenges:**
- Expensive (human feedback)
- Subjective (different preferences)
- Can be gamed (reward hacking)
- Ongoing research

**Why reinforcement learning at all, rather than more supervised fine-tuning?**

This is the question behind the question, and it has a clean answer. Supervised fine-tuning teaches by imitation: it needs a human to *write* an ideal response, and it can only ever pull the model toward that one response. But for most prompts there is no single ideal answer, writing good answers is slow and expensive, and — critically — humans are far better at *comparing* two answers than at authoring one from scratch. Preference comparisons are cheaper, more reliable, and more consistent between annotators. RLHF exists to convert that cheap comparison signal into a training signal, which requires a reward model because comparisons are not differentiable targets.

**The reward model, concretely.** Take the SFT model, strip the token-prediction head, and attach a scalar head that outputs one number per (prompt, response) pair. Train on pairs where a human said $y_w$ (winner) beats $y_l$ (loser), using the Bradley–Terry loss:

$$\mathcal{L}_{\text{RM}} = -\log\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big).$$

Reading it: push the winner's score above the loser's, with a sigmoid so the pressure eases once the gap is comfortable. Only score *differences* are meaningful — the absolute scale is arbitrary — which is why reward values across different reward models are not comparable.

**The PPO objective, with the term everyone forgets.** The optimized objective is

$$\mathbb{E}\big[r_\theta(x,y)\big] - \beta\,\mathrm{KL}\!\left(\pi_{\text{RL}}(y\mid x)\,\|\,\pi_{\text{SFT}}(y\mid x)\right).$$

The second term — a KL penalty against the SFT model — is not a detail, it is what makes the whole thing work. The reward model is only accurate near the distribution it was trained on; push the policy far from that and it finds adversarial text that scores enormously well and is gibberish to a human. That is *reward hacking*, and without the KL leash it happens quickly and reliably. Sometimes the failure is subtler and therefore worse: the model discovers that long, hedged, flattering answers score highly, and you get verbosity and sycophancy rather than obvious nonsense.

**DPO, which the answer above predates.** *Direct Preference Optimization* (2023) showed that the constrained RL objective above has a closed-form optimal policy, which can be rearranged so that the reward model is expressed in terms of the policy itself. The result is that you can optimize preferences with a simple supervised-style loss on preference pairs, with no separate reward model and no RL loop:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right).$$

It is dramatically simpler to implement and tune, and it is now the default for most open-weight alignment work, with a family of variants (IPO, KTO, ORPO, and online/iterative versions) addressing its limitations. Frontier labs still use online RL methods too, partly because online generation lets the model explore beyond a fixed preference dataset. *Alignment methodology is one of the fastest-moving parts of the field — verify what is current before asserting a default.*

Also worth naming: *Constitutional AI / RLAIF*, where the preference labels come from a model judging against a written set of principles instead of from humans, which removes the human-labelling bottleneck and makes the value judgements auditable as text.

**What RLHF does not fix.** It aligns the model to *the preferences of the annotator pool you hired*, which is a specific and non-neutral set of people. It optimizes for what raters approve of, which is not the same as what is true — hence the well-documented tendency toward sycophancy, since agreeing with the user is reliably rated highly. And it can degrade raw capability slightly, the "alignment tax." These limits are the interesting part of the answer.

**Follow-up:** *What is reward hacking, in one concrete example?* The classic pattern: raters mildly prefer longer, more thorough answers, so the reward model learns length as a proxy for quality, and the policy learns to pad. You catch it by tracking response length alongside reward during training — a reward curve rising in lockstep with length is the signature — and by holding out a human evaluation that the reward model never sees.

> **Why the interviewer asks this.** RLHF is where capability turns into a usable product, so understanding it signals that you think about deployment and not only about training loss.

> **Saying it out loud.** RLHF is three stages. First supervised fine-tuning on human-written demonstrations so the model follows instructions at all. Then a reward model: show humans two responses, ask which is better, and train a model to predict that preference — because people are much better at comparing two answers than at writing the perfect one. Then reinforcement learning against that reward, with a KL penalty keeping the policy close to where it started. That KL term is the crucial bit — without it the model finds text that scores enormously well on the reward model and is garbage to a human, which is reward hacking. Worth adding that DPO has largely replaced the RL loop in open-weight work: it turns out you can optimize the same objective directly on preference pairs with no reward model at all.

---

### Q80: What are emergent abilities? Give examples.

**Answer:**

**Emergent Abilities:**
- Abilities that appear only at large scale
- Not present in smaller models
- Unexpected capabilities
- Emerge from scale, not explicit training

**Examples:**

**1. Arithmetic:**
- Small models: Can't do math
- Large models: Can do arithmetic (not explicitly trained)
- Example: "What is 123 × 456?" → Correct answer

**2. Code Generation:**
- Small models: Can't write code
- Large models: Can generate working code
- Example: "Write Python function to sort list" → Working code

**3. Few-shot Learning:**
- Small models: Need many examples
- Large models: Learn from few examples
- Example: 1-2 examples sufficient

**4. Reasoning:**
- Small models: Limited reasoning
- Large models: Some logical reasoning
- Example: Multi-step problem solving

**5. Instruction Following:**
- Small models: Don't follow instructions well
- Large models: Better at following instructions
- Example: Complex multi-step instructions

**Why Important:**
- Shows scale matters
- Unexpected capabilities
- Hard to predict what will emerge
- Justifies investment in scale

**Implications:**
- Can't predict all capabilities
- Need to test large models
- Emergent abilities are powerful
- Safety concerns (unexpected behaviors)

**The strong counterargument you should raise yourself.**

Emergence is the one topic in this section where reciting the standard answer is a weaker response than complicating it. In 2023, *Are Emergent Abilities of Large Language Models a Mirage?* (Schaeffer et al.) argued that many reported emergent jumps are artefacts of the **metric**, not the model. The argument is precise and worth being able to reproduce.

Take multi-digit arithmetic scored by exact match. Exact match on a 5-token answer is roughly the joint probability of getting every token right, so if per-token accuracy improves smoothly from 0.70 to 0.95, exact match goes from $0.70^{5} = 0.17$ to $0.95^{5} = 0.77$ — the underlying capability moved smoothly, but a metric that multiplies per-token probabilities converts that into a curve that looks flat then explodes. Swap to a continuous metric such as token edit distance or the log-probability of the correct answer, and the same models show smooth, predictable improvement with no discontinuity at all. Discontinuous, all-or-nothing metrics manufacture discontinuous-looking capabilities.

There are secondary artefacts too: small test sets make near-zero scores indistinguishable from zero, and log-spaced model sizes make a smooth curve look like a step because you sampled it three times.

**What survives the critique.** Not everything. Even granting the metric argument, two things remain real and useful. First, *for a fixed metric that you care about*, capability really is unusable below some scale and usable above it, and that threshold is not currently predictable from the loss curve — which is an operational problem regardless of whether the underlying capability is smooth. Second, some phase-transition-like phenomena have been observed with continuous metrics and mechanistic explanations, notably the abrupt formation of induction heads during training (Q78) and grokking, where generalization appears long after memorization. So the defensible position is: *loss scales smoothly and predictably; downstream capability on discrete metrics can appear abruptly, sometimes genuinely and sometimes as a measurement artefact.*

**Why the specific examples above need care.** "Large models can do arithmetic" is a fragile claim — models are unreliable at multi-digit multiplication and their accuracy falls off sharply with digit count unless they use chain-of-thought or a calculator tool. What genuinely emerged with scale is *chain-of-thought prompting itself*: asking a small model to reason step by step does not help and often hurts, while asking a sufficiently large model to do so produces large gains. That is a cleaner and better-documented example than raw arithmetic. Similarly, instruction following is better described as a product of instruction tuning and RLHF than as something that emerged from scale alone.

**Follow-up:** *Why does this matter for safety?* Because it bears on predictability. If capabilities genuinely appear without warning, you cannot certify a model's behaviour before training it, and evaluation must happen after the fact. If the apparent unpredictability is largely a metric artefact, then better-designed continuous evaluations could forecast capabilities in advance — which would be a substantially better world to be in. That is the practical stake in what looks like an academic dispute.

> **Why the interviewer asks this.** It is a soft test for whether you read critically; a candidate who volunteers the mirage critique is signalling that they track the literature rather than the headlines.

> **Saying it out loud.** The standard story is that some abilities just don't exist below a certain scale and then appear — chain-of-thought reasoning is the cleanest example, since asking a small model to think step by step doesn't help and can hurt, while asking a big one helps a lot. But I'd raise the counterargument, because it's a good one. There's a well-known paper arguing a lot of these jumps are metric artefacts: if you score with exact match on a five-token answer, per-token accuracy going smoothly from seventy to ninety-five percent turns into exact-match going from seventeen to seventy-seven, which looks like a cliff. Switch to a continuous metric and the curve is smooth. What survives is the operational point — for the metric you actually care about, capability can be unusable at one scale and usable at the next, and we can't currently predict where that happens from the loss curve.

---

### Q81: How do modern foundation models differ from BERT?

**Answer:**

**Architecture:**

**BERT:**
- Encoder-only
- Bidirectional
- Good for understanding
- Can't generate

**Modern Foundation Models (GPT-4, etc.):**
- Decoder-only (or encoder-decoder)
- Unidirectional (for generation)
- Good for generation
- Can do understanding (with prompting)

**Training:**

**BERT:**
- Masked language modeling
- Next sentence prediction
- ~3B tokens
- Task-specific fine-tuning

**Modern:**
- Next token prediction
- Trillions of tokens
- RLHF for alignment
- In-context learning

**Capabilities:**

**BERT:**
- Understanding tasks
- Classification, NER, QA
- Needs fine-tuning per task
- Task-specific models

**Modern:**
- Generation + understanding
- Many tasks with one model
- In-context learning
- General-purpose

**Scale:**

**BERT:**
- 110M-340M parameters
- Small datasets
- Moderate compute

**Modern:**
- 175B+ parameters
- Trillions of tokens
- Massive compute

**Usage:**

**BERT:**
- Fine-tune for specific task
- Different model per task
- Requires labeled data

**Modern:**
- Prompt with examples
- Same model for all tasks
- No labeled data needed (few-shot)

**Key Differences Summary:**

| Aspect | BERT | Modern Foundation Models |
|--------|------|------------------------|
| **Architecture** | Encoder | Decoder |
| **Direction** | Bidirectional | Unidirectional |
| **Primary Use** | Understanding | Generation |
| **Scale** | 110M-340M | 175B+ |
| **Training** | MLM + NSP | Next token + RLHF |
| **Usage** | Fine-tuning | In-context learning |
| **Tasks** | Task-specific | General-purpose |

**Correcting the framing: BERT was not replaced, it was relocated.**

The comparison table is accurate but its implicit story — old versus new — is misleading, and correcting it is the most valuable thing you can add here. BERT-family encoders are running at enormous volume in production in 2026, because for a large class of jobs they are strictly the better tool.

*Embeddings and retrieval.* Producing one vector for a passage is exactly what a bidirectional encoder is built for, and it is what every dense retriever and re-ranker in Q57–Q70 uses. A decoder's causal mask means early tokens cannot see later ones, which is a real handicap for summarizing a whole text into one vector; decoder-derived embedding models exist and work, but they typically need architectural surgery (removing the causal mask, or bidirectional adaptation) to get there.

*High-throughput classification.* Spam filtering, content moderation triage, intent routing, and toxicity scoring at millions of requests per hour. A 100M-parameter encoder runs in single-digit milliseconds on a CPU. Routing that traffic through a frontier LLM would cost orders of magnitude more per request and add latency for no accuracy gain on a task with abundant labels.

*Token-level tasks.* NER and other span extraction (Q72) benefit directly from bidirectional context — knowing what comes *after* a token genuinely helps classify it — and produce structured output that is awkward to coax reliably out of free-form generation.

The right summary is that the field bifurcated: encoders for *representation*, decoders for *generation*, with each dominating a different part of the stack.

**Two technical corrections to the table.**

*"Unidirectional" understates modern architecture.* Frontier models are decoder-only with causal attention, yes, but the table's "Decoder" row hides a lot: rotary or ALiBi positional encodings instead of learned absolute positions, RMSNorm instead of LayerNorm, pre-normalization instead of post, SwiGLU instead of ReLU feed-forward blocks, grouped-query or multi-query attention to shrink the KV cache, and mixture-of-experts for sparse capacity. "Decoder-only Transformer" describes the family, not the design.

*NSP was a mistake.* BERT's next-sentence-prediction objective was shown by RoBERTa to be useless or mildly harmful, and essentially every successor dropped it. Listing it as a defining feature of BERT is technically right and historically a footnote about what did not work.

**One more axis the table misses: what you can inspect.** A fine-tuned BERT is a fixed function you can evaluate exhaustively, version, and audit — you know precisely what it was trained on and its behaviour does not change unless you change it. A hosted frontier model can be updated underneath you, behaves differently with a reworded prompt, and cannot be exhaustively characterized. For regulated deployments that difference frequently decides the architecture regardless of accuracy.

**Follow-up:** *If you had to pick one model for a new text classification task today, which?* I would prompt an LLM first to get a zero-shot baseline and to generate labels cheaply, then distil into a fine-tuned encoder for serving. That gets the LLM's quality with the encoder's cost and latency, and it is the pattern most teams converge on once they see the inference bill.

> **Saying it out loud.** Architecturally, BERT is an encoder with bidirectional attention trained by masking tokens, so it's built for understanding and can't generate. Modern foundation models are decoders with causal attention trained on next-token prediction, so they generate, and understanding comes along for free through prompting. Scale differs by three orders of magnitude, and modern models add RLHF on top. But I'd push back on the framing that BERT was replaced — it moved. Every dense retriever and re-ranker in a RAG stack is an encoder, because producing one vector for a passage is exactly what bidirectional attention is good at. And for high-volume classification, a hundred-million-parameter encoder runs in milliseconds on CPU where an LLM call costs orders of magnitude more. The field split: encoders for representation, decoders for generation.

---

See `38_multimodal_and_embeddings/foundation_models_evolution.md` for complete evolution story!

---

## Multimodal Integration and World Models

### Q82: How do you integrate triplet data (knowledge graphs) into foundation models?

**Answer:**

**Triplet Data:**
- Format: (Subject, Relation, Object)
- Example: ("Einstein", "born_in", "Germany")
- Represents structured knowledge

**Integration Strategies:**

**1. Direct Encoding:**
- Convert triplet to text: "Einstein [born_in] Germany"
- Add to training corpus
- Model learns relationships

**2. Knowledge Graph Embedding:**
- Pre-train embeddings (TransE, TransR)
- Learn entity and relation embeddings
- Integrate into language model

**3. Structured Prompting:**
```
"Given: (Einstein, born_in, Germany)
Question: Where was Einstein born?
Answer: Germany"
```

**4. Multi-Task Learning:**
- Language modeling + triplet prediction
- Joint training on all tasks

**Processing Pipeline:**
1. Data collection (Wikidata, Freebase)
2. Data cleaning (remove duplicates, validate)
3. Format conversion (triplet → text)
4. Integration (mixed corpus or separate objective)

**Best Practice:**
- Mix 20% triplet-derived text with 80% natural text
- Use knowledge graph embeddings for better reasoning
- Fine-tune on domain-specific triplets

**Why this is harder than it looks, and what the real tradeoff is.**

A *knowledge graph* is a set of facts stored as (subject, relation, object) triples — structured, exact, and easy to update. A language model stores knowledge as distributed weights — fuzzy, approximate, and expensive to update. Integration means bridging two representations with opposite properties, and the central question is where the knowledge should *live*.

**Baking it into the weights (options 1, 2 and 4 above).** Verbalize triples into sentences and pretrain or fine-tune on them, or align KG embeddings with the language model's space. The advantage is that the knowledge becomes available with no retrieval step at inference. The disadvantages are severe and worth naming: updating one fact requires retraining or some form of model editing; the model may still hallucinate around the fact because nothing enforces the constraint at generation time; and verbalized triples are stilted text that can degrade fluency if you mix in too much — which is why the "20% triple-derived text" guidance in the answer above should be treated as a rough upper bound rather than a target, and tuned by measuring both factual accuracy *and* general language quality, since it is easy to buy the first with the second.

**Keeping it external (option 3, structured prompting) is usually the better default.** Retrieve the relevant subgraph at query time and put it in the context. Facts update instantly by writing to the graph, you get provenance for free, and there is no training run. This is RAG with a graph instead of a text index, and it is where most production systems land. The retrieval step is the interesting engineering: you need entity linking to map surface strings in the query to graph nodes, then a neighbourhood expansion of one or two hops, then serialization of the subgraph into text the model can read. Two hops from a well-connected entity can pull in thousands of triples, so you need relevance filtering — usually by relation type or by embedding similarity to the query — or you blow the context window on irrelevant edges.

**The technique worth naming: GraphRAG.** Rather than assuming a pre-existing knowledge graph, extract entities and relations from your document corpus with an LLM, build a graph, cluster it into communities, and pre-generate a summary per community. Then a query can be answered either locally (relevant entities and their neighbourhoods) or globally (over the community summaries). The point of the global path is that it answers questions ordinary vector RAG structurally cannot — "what are the main themes across this corpus?" has no single chunk that contains the answer, so top-K similarity retrieval will always fail on it. The cost is a substantial LLM-driven indexing pass, so it suits corpora that are queried far more often than they change.

**How the KG embedding methods work, briefly.** TransE models a relation as a translation in vector space, training so that $h + r \approx t$ for true triples and not for corrupted ones. It is elegant but cannot represent one-to-many relations properly — if (US, city, NYC) and (US, city, LA) both hold, then NYC and LA are forced to the same point. TransR and successors add relation-specific projection matrices to fix that. Mention the limitation, not just the name.

**Follow-up:** *How do you handle a conflict between the graph and the model's parametric knowledge?* Instruct the model explicitly that the provided facts take precedence over what it believes, and verify afterwards that the answer's entities actually appear in the supplied subgraph. Models do not reliably defer to context on their own, especially when the context contradicts something strongly represented in the weights — this is a measured effect, not a theoretical worry.

> **Why the interviewer asks this.** It is really a question about where knowledge should live — weights versus retrieval — and whether you understand the update, provenance and cost consequences of that choice.

> **Saying it out loud.** The real question is where you want the knowledge to live. You can bake triples into the weights by verbalizing them into sentences and training on them, which means no retrieval at inference — but then updating one fact means retraining, and there's nothing stopping the model hallucinating around it. Or you keep the graph external and retrieve the relevant subgraph at query time, which is basically RAG with a graph instead of a text index. That's what I'd default to: facts update instantly, you get provenance for free, and no training run. The engineering is entity linking plus a one- or two-hop neighbourhood expansion, with filtering, because two hops off a popular node pulls in thousands of edges. And GraphRAG is worth knowing — you build the graph from your corpus with an LLM and pre-summarize communities, which lets you answer corpus-wide questions that vector search structurally can't.

---

### Q83: How do you integrate past conversation history into LLMs?

**Answer:**

**Challenges:**
- Limited context window (2K-32K tokens)
- Need long-term memory
- User personalization

**Integration Strategies:**

**1. Context Window Extension:**
- Store history in external memory
- Retrieve relevant history
- Concatenate to context

**2. Memory-Augmented Models:**
- Main model: Processes current input
- Memory bank: Stores conversation history
- Attention: Attend to relevant history

**3. Hierarchical Encoding:**
- Level 1: Individual messages
- Level 2: Conversation turns
- Level 3: Conversation sessions
- Level 4: User profile

**4. RAG for History:**
- Store conversations in vector DB
- Retrieve relevant history
- Add to context

**Processing:**
1. Data collection (chat logs, transcripts)
2. Data cleaning (remove PII, anonymize)
3. History segmentation (turns, sessions)
4. Feature extraction (sentiment, intent, entities)
5. Integration (conversation modeling or user embedding)

**Best Practice:**
- Keep last 10-20 turns in context
- Use retrieval for older history
- Learn user embeddings for personalization

**A currency note and a sharper framing.**

The stated "2K–32K tokens" context limit is out of date: by 2026 windows of 200K to 1M tokens are common. But that does not dissolve the problem, and explaining why is the strongest part of an answer here. Three reasons memory is still an architecture problem rather than a solved one: cost, since you pay for input tokens on every turn and resending a 200-turn history each time is quadratic in total spend across the conversation; latency and the lost-in-the-middle effect (Q59), so a fact stated 300 turns ago and buried mid-context is unreliably recalled; and persistence, since a context window lasts one session while a user expects the system to remember their preferences next month. That third one no context window ever solves.

**A concrete tiered memory design, which is what to sketch.**

*Working memory* — the last $N$ turns verbatim, where $N$ is set by budget rather than principle. Recency is cheap and disproportionately useful.

*Episodic memory* — older turns embedded and stored in a vector index, retrieved by relevance to the current message. The subtlety worth mentioning: embed at the level of a *topic segment* rather than a single message, because "yes, that one" is a useless retrieval unit on its own; and store each segment with a timestamp so recency can be blended into ranking.

*Semantic memory / user profile* — durable extracted facts ("prefers Python," "works in oncology," "is allergic to shellfish"), written as structured records rather than raw text. These are the things that should survive indefinitely and be injected into every session. The extraction step is an LLM call over recent conversation asking what is worth remembering, run asynchronously so it does not sit on the response path.

*Rolling summary* — a compressed running narrative of the conversation so far, updated periodically, so the model has the arc even when the verbatim turns have been evicted.

**The hard parts, which is where the answer earns its keep.** Contradiction handling: the user said "I use Java" in March and "I use Go" in September, and both are now in memory. You need a write path that updates or supersedes rather than only appends, with timestamps deciding precedence. Forgetting: memory that only grows becomes retrieval noise, so you need eviction by age and by access frequency. And privacy, which is not a footnote — a persistent memory store contains everything a user ever said, so you need PII detection at write time, explicit user visibility into what is stored, deletion that actually deletes (including from any derived index), and a defensible retention policy. Raising privacy unprompted is a strong signal in this question because the failure mode is a headline rather than a metric.

**Follow-up:** *How do you evaluate a memory system?* Construct multi-session test conversations where a fact is established early and required much later, and measure recall at varying distances. Then measure the failure modes separately: false memories (facts asserted that were never stated), stale memories (superseded facts recalled as current), and intrusion (memories retrieved into an unrelated conversation, which users find unsettling even when the fact is correct).

> **Saying it out loud.** Even with a million-token window this doesn't go away, because you pay for every input token on every turn, models are unreliable at recalling things buried in the middle of a long context, and a window lasts one session while users expect you to remember them next month. So I'd build tiers. Last few turns verbatim, because recency is cheap and useful. Older turns embedded into a vector store and retrieved by relevance — segmented by topic, not by message, because "yes, that one" is a useless retrieval unit. A rolling summary for the overall arc. And a structured user profile of durable facts, extracted asynchronously and injected into every session. The hard parts are contradiction — the user said Java in March and Go in September, so writes have to supersede, not just append — forgetting, so memory doesn't become noise, and privacy, because that store ends up holding everything the user ever said.

---

### Q84: What is a world model? How do you build one for LLMs?

**Answer:**

**World Model:**
- Internal representation of how world works
- Predicts future states
- Understands cause and effect
- Enables planning and reasoning

**Key Components:**

**1. State Representation:**
- Entities and properties
- Relationships
- Temporal information
- Methods: Symbolic, embedding, graph

**2. Transition Model:**
- Predicts next state given current state and action
- Types: Deterministic, stochastic, learned
- Training: Neural network on state-action-next_state tuples

**3. Observation Model:**
- Maps world state to observations
- Handles partial observability
- Models what we can observe

**4. Reward Model:**
- Defines what's good/bad
- Guides learning
- Types: Task-specific, shaped, learned

**5. Planning:**
- Use world model to find good actions
- Methods: Model-based RL, tree search, MPC

**Integration with LLMs:**
```
LLM → World Model Interface → World Model → Planning → Actions
```

**Training:**
1. Learn world model from data
2. Integrate with LLM
3. Joint training end-to-end

**Where the term comes from, and what the real debate is.**

The phrase comes from model-based reinforcement learning, where a *world model* is a learned simulator: given the current state and a proposed action, it predicts the next state and the reward. Its value is that an agent can plan by rolling the simulator forward — trying thousands of imagined action sequences and picking the best — instead of taking thousands of expensive or dangerous real actions. Ha and Schmidhuber's "World Models" and the Dreamer line of work are the canonical references, and MuZero is the clearest demonstration that learning the model and planning in it can beat planning in a known one.

**The live question for LLMs: do they already have one?** This is what the interviewer is actually curious about, and the honest answer is "partially, and it is contested."

*Evidence for.* Probing studies find internal representations of things the model was never explicitly taught — the state of a board in a game from move sequences alone, spatial and temporal coordinates for entities, and other latent structure recoverable by linear probes. That is more than surface statistics, and it is reasonably strong evidence of some internal model of the domain being described.

*Evidence against.* LLM predictions violate physical and causal constraints in ways a real simulator would not; performance degrades sharply on counterfactual variants of familiar tasks (arithmetic in base 9, chess from a slightly non-standard opening), which suggests reliance on memorized patterns rather than a general model that transfers; and the models are notoriously weak at multi-step planning that requires backtracking, which is exactly what a usable world model should make easy.

The defensible position is that LLMs learn something like a world model *of text-describable regularities*, learned from a passive corpus, which is not the same as a grounded causal model of physical dynamics — and that the difference shows up precisely where you would predict, on counterfactuals and long-horizon planning.

**The deepest obstacle, which is worth naming.** A world model learned only from observation cannot in general distinguish correlation from causation, because *intervention* is what identifies causal structure and passive text contains descriptions of interventions rather than interventions themselves. This is the standard Pearl argument and it is the strongest theoretical reason to expect that scaling passive prediction alone will not produce a fully causal world model. It is also why embodiment, tool use, and agents that actually act and observe consequences are the direction of travel — an agent that calls an API and sees the result is performing an intervention, however small.

**A practical version, because "build a world model for an LLM" needs to become concrete.** A deployable approximation: give the model an explicit external state representation (a structured document, a database, a game state), require it to propose actions against that state, execute them in a real or simulated environment, feed the observed result back, and let the model revise. The environment supplies the transition dynamics the model lacks, and the LLM supplies the priors and the language interface. That is the architecture behind most agentic systems in 2026, and describing it grounds an otherwise abstract question. *Agent architectures are moving fast; treat any specific framework as a snapshot.*

**Follow-up:** *Why not just train on video?* Video gives you passive physical dynamics, which is real progress on the observation side and is why video generation models are increasingly discussed as world models. But it is still observational: you see what happened, not what would have happened had the agent acted differently. Interaction is the missing ingredient, and it is expensive to collect.

> **Why the interviewer asks this.** This is a research-taste question — they want to see whether you can hold a genuinely open problem carefully, with evidence on both sides, rather than picking a slogan.

> **Saying it out loud.** A world model is a learned simulator: give it a state and an action, it predicts the next state and the reward, so an agent can plan by rolling it forward in imagination instead of acting in the real world. The interesting question is whether LLMs already have one. There's decent evidence they partly do — probes find internal representations of things like board state or spatial layout that were never explicitly supervised. But they also fail on counterfactual variants of familiar tasks, like arithmetic in a different base, which suggests a lot of pattern matching rather than a general model. The deep obstacle is that you can't identify causation from passive observation alone — you need intervention — which is the main argument that scaling text prediction won't get you there and why agents that act and see consequences matter.

---

### Q85: What are the future directions of LLMs? What's the path to AGI?

**Answer:**

**Future Directions:**

**1. General Intelligence:**
- Human-level intelligence
- General problem solving
- Transfer learning
- Few-shot adaptation

**2. World Understanding:**
- Understand how world works
- Predict consequences
- Plan actions
- Reason about causality

**3. Continual Learning:**
- Learn from new data continuously
- Don't forget old knowledge
- Adapt to new domains

**4. Embodied Intelligence:**
- Interact with physical world
- Learn from experience
- Understand physics

**Key Research Areas:**

**Scaling:**
- Efficient scaling
- Better architectures
- Sparse models

**Multimodality:**
- All modalities
- Unified representation

**Reasoning:**
- Strong logical reasoning
- Causal reasoning
- Mathematical reasoning

**Planning:**
- Long-term planning
- Hierarchical planning

**Memory:**
- Long-term memory
- Episodic memory
- Semantic memory

**AGI Architecture Vision:**
```
1. Perception Module (multimodal input)
2. World Model (state, transition, planning)
3. Memory System (episodic, semantic, working)
4. Reasoning Engine (logical, causal, analogical)
5. Action Module (text, tools, physical)
6. Learning System (continual, meta-learning)
```

**Path to AGI:**
- Multimodal integration
- World models
- Strong reasoning
- Long-term planning
- Long-term memory
- Continual learning

**Reading this question correctly.**

Nobody knows the path to AGI, and an interviewer asking this is not checking whether you do. They are checking three things: whether you can discuss speculative material without either dismissing it or overclaiming; whether you know what the current *concrete* bottlenecks are, as opposed to the aspirational bullet list; and whether you can attach evidence to a prediction. The list above is a fine map of aspirations. What follows is the part that makes an answer sound like it came from someone working in the field.

**The bottlenecks that are actually binding right now.**

*Data.* High-quality human-written text is finite and much of it is already used. This has pushed the field toward synthetic data, multiple epochs over curated corpora, and non-text modalities. The open risk is model collapse — training on model output degrading quality across generations — which appears controllable with careful mixing of real data but is a genuine constraint rather than a hypothetical.

*Inference cost.* Capability is increasingly purchased at inference time rather than training time: reasoning models spend more compute per query, agents make many model calls per task. That flips the economics, since inference cost scales with usage while training cost is amortized. Mixture-of-experts, distillation, quantization, speculative decoding and caching are all responses to this single pressure, and it is probably the most consequential shift since Chinchilla.

*Continual learning.* Models are frozen at a cutoff. Fine-tuning on new data causes *catastrophic forgetting* — new gradients overwrite the weights encoding old capabilities — and there is still no method that reliably adds knowledge without degrading something else. This is why the industry routes around the problem with retrieval and long context rather than solving it, and it is a real, unglamorous open problem.

*Reliability and long-horizon execution.* An agent with 95% per-step reliability completes a 20-step task about 36% of the time, since $0.95^{20} \approx 0.358$. Long-horizon autonomy therefore needs either very high per-step reliability or robust error detection and recovery, and this compounding is the concrete reason agents demo well and deploy badly. It is a much more useful thing to say than "reasoning needs to improve."

*Evaluation.* We increasingly cannot measure what we are building. Benchmarks saturate, leak into training data, and fail to predict real task performance. Without trustworthy measurement, "progress toward AGI" is not a claim anyone can check — which is itself part of why the question has no rigorous answer.

**On the AGI architecture diagram above.** It is a reasonable functional decomposition and it is worth noting that it closely resembles classical cognitive architectures (SOAR, ACT-R) from decades ago. The lesson from that history is that the hard part was never drawing the boxes; it was that each box turned out to be an open research problem, and that the interfaces between them were harder still. Also worth saying: the recent trend has run the other way — end-to-end learned systems have repeatedly beaten hand-designed modular ones, which is the Bitter Lesson, so a diagram of six hand-specified modules should be held loosely.

*Everything in this answer is time-sensitive and reflects the state of the field as of mid-2026. Treat it as a snapshot; the bottlenecks change faster than the aspirations do.*

**Follow-up:** *What would change your mind that we are close?* Something concrete and falsifiable: a model that reliably completes multi-day, multi-tool tasks with a real error-recovery loop; genuine continual learning without forgetting; or robust performance on counterfactual variants of tasks it has mastered, which would suggest general mechanisms rather than memorized patterns. Naming a falsifiable criterion is what separates a considered view from an opinion.

> **Why the interviewer asks this.** It is a calibration test, not a knowledge test — they are watching how you handle a question where the honest answer is uncertainty.

> **Saying it out loud.** I'd be honest that nobody knows, and talk about what's actually blocking things instead. Four bottlenecks I'd name. Data — high-quality human text is finite and mostly used, which is what's driving synthetic data. Inference cost, because capability is increasingly bought at inference time now, and unlike training that scales with every user. Continual learning, which is genuinely unsolved — fine-tuning on new data causes catastrophic forgetting, so the industry routes around it with retrieval rather than fixing it. And reliability compounding: an agent that's ninety-five percent reliable per step finishes a twenty-step task about a third of the time, which is exactly why agents demo well and deploy badly. On the architecture diagrams with perception, memory, reasoning, and planning modules — those look a lot like cognitive architectures from the eighties, and the lesson there was that drawing the boxes was never the hard part.

---

See `38_multimodal_and_embeddings/multimodal_integration_and_world_models.md` for complete details!

---

## GPT Implementation, Training, and Decoding

### Q86: Implement a complete GPT model from scratch. What are all the components?

**Answer:**

**Complete GPT consists of:**

**1. Token Embedding:**
- Converts token indices to dense vectors
- Learned embeddings, shape (vocab_size, d_model)

**2. Positional Encoding:**
- Adds position information to embeddings
- Sinusoidal or learned positional embeddings
- Shape: (max_seq_len, d_model)

**3. Multi-Head Attention:**
- Self-attention mechanism
- Multiple heads (typically 12-96)
- Each head: Q, K, V projections → attention → concatenate
- Complexity: O(n²d)

**4. Feed-Forward Network:**
- Two linear layers with ReLU
- Expands then contracts: d_model → d_ff → d_model
- Applied position-wise

**5. Transformer Block:**
- Multi-head attention + Feed-forward
- Residual connections + Layer normalization
- Dropout for regularization

**6. Stack of Transformer Blocks:**
- Multiple blocks (typically 12-96 layers)
- Each block refines representations

**7. Final Layer Norm:**
- Normalizes final representations

**8. Output Projection:**
- Maps to vocabulary size
- Produces logits for next token prediction

**See `04_transformers/gpt_complete.py` for complete implementation!**

**Narrative walkthrough: building it in the order you would write it.**

The component list above is a parts inventory. What an interviewer wants is the assembly order and the reason for each piece, told so that someone could reconstruct the code from the description. Here is that narrative, with shapes tracked throughout. Take a batch of $B$ sequences of length $T$, model width $d$, and $h$ attention heads.

**1. Turn token IDs into vectors.** The input is an integer tensor of shape $(B,T)$. An embedding table of shape (vocab, $d$) is looked up row-wise to give $(B,T,d)$. This is a lookup, not a matrix multiply — the one-hot formulation you see in papers is mathematically equivalent and computationally wasteful.

**2. Add position.** Self-attention is *permutation equivariant*: shuffle the input tokens and the outputs shuffle identically, because attention is a weighted sum over a set with no notion of order. Without positional information "dog bites man" and "man bites dog" are literally the same input. GPT-2 used a learned position embedding table of shape (max_len, $d$), added to the token embeddings. Modern models mostly use rotary embeddings instead, which rotate the query and key vectors by an angle proportional to position so that attention scores depend on *relative* distance — this extrapolates to longer sequences far better than a learned absolute table, which has no entry at all for a position it never saw in training.

**3. Causal self-attention, the core.** Project the $(B,T,d)$ input three times to get queries, keys and values. In practice one linear layer of width $3d$ then split, because a single large matmul is faster than three small ones. Reshape each to $(B,h,T,d/h)$ so every head works in its own subspace. Compute $QK^{\top}$ to get $(B,h,T,T)$ — the score of every token against every other. Divide by $\sqrt{d/h}$: the dot product of two vectors with unit-variance entries has variance equal to the dimension, so without this scaling the logits grow with head size, the softmax saturates, and gradients vanish. Then apply the causal mask by setting all positions where key index exceeds query index to $-\infty$, so softmax assigns them exactly zero — this is what makes position $t$ unable to see position $t+1$, which is what makes it valid to compute the loss at every position simultaneously. Softmax over the last dimension, multiply by $V$ to get $(B,h,T,d/h)$, transpose and reshape back to $(B,T,d)$, and apply a final output projection that lets the heads' results mix.

**4. The feed-forward block.** Two linear layers with a nonlinearity between, expanding $d \to 4d \to d$. It is applied independently at every position — attention moves information *between* positions, the feed-forward network processes each position *on its own*, and that division of labour is the cleanest way to describe a Transformer block. Note that the answer above says ReLU; GPT-2 and successors actually use GELU, and modern models typically use SwiGLU, a gated variant. The $4\times$ expansion is convention rather than derivation, and this block holds roughly two-thirds of the model's parameters.

**5. Residual connections and normalization.** Each sub-block computes `x = x + sublayer(norm(x))`. Two separate ideas here. The residual gives gradients a path straight from the loss to every layer via the identity term, which is what makes 96-layer training feasible at all — it is also useful to think of the residual stream as a shared bus that each layer reads from and writes to. The normalization stabilizes activation scale. GPT-2 moved the norm *before* the sublayer (pre-norm) rather than after, which is the reason deep Transformers train without a delicate learning-rate warmup schedule; post-norm as in the original paper is noticeably harder to train deep.

**6. Stack, normalize, project out.** Repeat the block $L$ times, apply a final layer norm, then a linear map from $d$ to vocabulary size producing logits $(B,T,V)$. That output matrix is very commonly *tied* to the input embedding table — the same weights, transposed — which saves a large number of parameters (vocab $\times d$ can be a substantial fraction of a small model) and typically improves quality.

**Verified implementation.** The following runs and trains. On a synthetic copy task — learning to continue a repeating sequence — this exact code drives cross-entropy from $\ln 17 = 2.83$ (the uniform-guessing baseline for a 17-token vocabulary) to $0.0000$ in 300 AdamW steps, and then generates the correct continuation. That end-to-end sanity check, overfitting one tiny batch before touching real data, is the first thing to do with any model implementation.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv  = nn.Linear(d_model, 3 * d_model)   # one matmul, then split
        self.proj = nn.Linear(d_model, d_model)       # lets heads mix
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("mask", mask)            # not a parameter

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d_head): every head in its own subspace
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B,h,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)                                  # masked -> exactly 0
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.mlp  = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                  nn.Linear(4 * d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm: norm inside the residual branch
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_head=4, n_layer=2, max_len=32):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight            # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))              # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

**Parameter count, so the shapes stop being abstract.** Per block: attention has $4d^{2}$ (three projections plus the output projection) and the feed-forward has $8d^{2}$ (two layers of $d\times 4d$), so about $12d^{2}$ per block, times $L$ blocks, plus $V\times d$ for embeddings. For GPT-2 small ($d=768$, $L=12$, $V=50257$) that is $12 \times 768^{2} \times 12 \approx 85\text{M}$ plus $\approx 39\text{M}$ of embeddings — the familiar 124M. Being able to do this arithmetic live is a strong signal.

**Follow-up:** *What would you change to make this production-grade?* Replace the hand-written attention with `F.scaled_dot_product_attention` so it dispatches to FlashAttention; swap learned positions for rotary; use RMSNorm instead of LayerNorm; add a KV cache for generation (Q88); add dropout; and use grouped-query attention to shrink that cache. None of these change what the model computes conceptually, and all of them matter for cost.

> **Why the interviewer asks this.** Anyone can call a library; writing attention by hand is the standard check that you know what the shapes are and why the mask and the scaling factor exist.

> **Saying it out loud.** I'd build it bottom up. Token IDs go through an embedding table, and you add positional information — because attention is permutation-equivariant, so without it "dog bites man" and "man bites dog" are the same input. Then the core block: project to queries, keys, and values, split across heads, take Q times K transposed to score every token against every other, divide by the square root of the head dimension so the softmax doesn't saturate, mask out the future with negative infinity, softmax, and multiply by V. Then a position-wise feed-forward that expands to four times the width and back. The mental model is that attention moves information between positions and the MLP processes each position alone. Wrap both in residual connections with pre-normalization, stack it however many times, final layer norm, and project to vocabulary size — usually tying that matrix to the input embeddings.

---

### Q87: How is GPT trained? Explain the training process in detail.

**Answer:**

**Training Objective:**
- Next token prediction (language modeling)
- Given tokens [t₁, t₂, ..., tₙ], predict [t₂, t₃, ..., tₙ₊₁]
- Autoregressive: each token depends on all previous tokens

**Training Process:**

**1. Data Preparation:**
- Large text corpora (books, web, articles)
- Tokenization (BPE, SentencePiece)
- Batching and padding to fixed length

**2. Forward Pass:**
- Token embeddings + positional encoding
- Pass through transformer blocks
- Output logits for each position

**3. Loss Function:**
- Cross-entropy loss
- L = -(1/n) Σ log P(tᵢ | t₁, ..., tᵢ₋₁)
- Compares predicted distribution to true next token

**4. Backward Pass:**
- Compute gradients via backpropagation
- Gradient clipping (max_norm=1.0)
- Update parameters with optimizer (Adam/AdamW)

**5. Training Details:**
- Learning rate: 3e-4 to 1e-4
- Learning rate scheduling (warmup + decay)
- Dropout for regularization
- Weight initialization (normal, std=0.02)

**Key Insight:**
- Massive scale: GPT-3 trained on trillions of tokens
- Self-supervised: no labels needed, just text
- Learns language patterns, syntax, semantics, reasoning

**See `04_transformers/gpt_training_decoding.md` for complete details!**

**Walkthrough: what one training step actually does.**

The five phases above are correct; here is the mechanical detail that turns them into something you could implement.

**The shift, precisely.** You do not run the model $T$ times to predict $T$ tokens. You take a chunk of $T+1$ tokens, feed positions $0..T-1$ as input and use positions $1..T$ as targets — literally `x = chunk[:-1]; y = chunk[1:]`. One forward pass produces logits at every position, and because the causal mask guarantees position $t$ never saw position $t+1$, all $T$ predictions are simultaneously valid. This is the efficiency that makes language model pretraining possible at all, and it is why decoder training extracts far more signal per token than BERT's masked objective, which supervises only the masked 15%.

**The loss, with units.** Cross-entropy averaged over all positions and all sequences in the batch. Its natural unit is nats per token; divide by $\ln 2$ for bits per token, and exponentiate for *perplexity*, the effective number of equally-likely choices the model is deciding among. A useful anchor: a uniform model over a 50,257-token vocabulary has perplexity 50,257 and loss $\ln(50257) = 10.8$; a good modern model on general web text is in the low single digits of perplexity. If your loss does not start near $\ln(\text{vocab size})$ on step one, your initialization or your data pipeline is broken — that is a five-second diagnostic worth knowing.

**Packing, and the detail everyone gets wrong.** The answer above says "batching and padding to fixed length." For pretraining you generally do not pad; you concatenate the whole corpus into one long token stream and slice fixed-length windows out of it, so no compute is wasted on padding tokens. The subtlety is that documents then bleed into each other — the model is asked to predict the first token of document $k+1$ from the tail of document $k$, which is noise. Fixes are to insert an end-of-text token at boundaries (cheap, mostly sufficient) or to use a block-diagonal attention mask that prevents cross-document attention (correct, slightly more work). For *fine-tuning* on instruction data you do pad, and there you must mask the padding out of the loss and usually mask the prompt tokens too, so the model is scored only on the response.

**The optimizer settings, with reasons.** AdamW rather than Adam, because decoupling weight decay from the adaptive scaling is what makes decay behave as intended. $\beta_2 = 0.95$ rather than the 0.999 default, since language-model gradients are noisy and the shorter second-moment window adapts faster. No weight decay on biases, layer-norm parameters, or embeddings — decaying a normalization gain toward zero is meaningless. Gradient clipping at global norm 1.0, which catches the loss spikes that come from a pathological batch and is cheap insurance. And a learning-rate schedule of linear warmup over a few hundred to a few thousand steps followed by cosine decay to about 10% of peak: the warmup exists because Adam's second-moment estimate is unreliable in the first steps, so a full-size step early can knock the model into a bad region it never recovers from.

**Batch size in tokens, not sequences.** Large-model training uses batches of hundreds of thousands to millions of *tokens*, assembled with gradient accumulation across many steps and many devices. Reporting batch size in sequences is meaningless unless the sequence length is also given, and interviewers notice the difference.

**Memory, which is where real training runs die.** Per parameter in mixed precision you store roughly: 2 bytes for the fp16/bf16 weight, 4 bytes for the fp32 master copy, and 8 bytes for Adam's two moment estimates — around 14–16 bytes per parameter before activations. A 7B model therefore needs on the order of 100 GB just for optimizer state, which is why sharding (ZeRO/FSDP) exists, distributing optimizer state, gradients and parameters across devices. Activation memory scales with batch $\times$ length $\times$ width $\times$ layers and is traded against compute by gradient checkpointing, which discards intermediate activations and recomputes them in the backward pass for roughly 30% extra compute. Being able to say why a model that "fits in memory" still fails to train is a strong practical signal.

**Verified.** The model in Q86 trained with exactly this recipe — AdamW, gradient clipping at 1.0, shifted targets — drops from the uniform baseline of $\ln 17 = 2.83$ to a loss of $0.0000$ on a memorization task in 300 steps, confirming the loop is wired correctly.

**Follow-up:** *Your loss suddenly spikes at step 40,000 and never recovers. What happened?* Most often a bad data batch (a long run of repeated or corrupted tokens) combined with an optimizer state that then goes bad. The standard practice is to checkpoint frequently, and on a spike, roll back to the last good checkpoint and skip the offending data shard. Lowering the learning rate or clipping more aggressively reduces the frequency. Loss spikes are common enough at scale that skip-and-resume is an ordinary operational procedure, not an emergency.

> **Why the interviewer asks this.** The interesting content is not the objective — everyone knows it is next-token prediction — but whether you know what makes a large run actually complete.

> **Saying it out loud.** The objective is just next-token prediction, but the mechanical bit worth stating is that you don't run the model once per token. You take a chunk, feed all but the last token as input and all but the first as targets, and one forward pass gives you a prediction at every position — the causal mask is what makes that legitimate, because position t never saw position t plus one. Loss is cross-entropy averaged over positions; exponentiate it and you get perplexity. On the practical side: AdamW with beta-two around 0.95, gradient clipping at norm one, and linear warmup then cosine decay, because Adam's second-moment estimate is unreliable in the first few hundred steps and a full-size step early can wreck the run. And I'd concatenate documents into one stream rather than padding, with an end-of-text token at the boundaries so the model isn't asked to predict one document from another.

---

### Q88: How does GPT decode/generate text? Explain the decoding process.

**Answer:**

**Autoregressive Generation:**
- Generate one token at a time
- Each token depends on all previous tokens
- Start with prompt, generate until stop condition

**Decoding Process:**

**1. Initial Prompt:**
- User provides starting text
- Tokenized into sequence [p₁, p₂, ..., pₖ]

**2. Forward Pass:**
- Process prompt through model
- Get logits for next token
- Shape: (vocab_size,) - scores for each token

**3. Convert to Probabilities:**
- Apply softmax: P(t) = exp(logit_t) / Σ exp(logit_i)
- Temperature scaling: P(t) = softmax(logits / T)
  - T=1.0: original distribution
  - T>1.0: more random (higher diversity)
  - T<1.0: more deterministic (lower diversity)

**4. Sample Token:**
- **Greedy**: Always pick highest probability
- **Sampling**: Random sample from distribution
- **Top-k**: Sample from top-k tokens
- **Top-p (Nucleus)**: Sample from tokens with cumulative prob > p

**5. Append and Repeat:**
- Append sampled token to sequence
- Process new sequence again
- Repeat until stop condition

**6. Stop Conditions:**
- Maximum length reached
- End-of-sequence token generated
- Specific stop sequence

**Causal Masking:**
- During decoding, mask future positions
- Upper triangular matrix: -inf for future, 0 for past
- Ensures model only sees previous tokens

**See `04_transformers/gpt_training_decoding.md` for complete details!**

**The missing piece: the KV cache, and why generation is memory-bound.**

The description above is correct but omits the single most important implementation fact about decoding, which is that a naive implementation is quadratically wasteful and nobody does it.

Re-read step 5: "append sampled token, process new sequence again." Done literally, generating token $n$ requires a forward pass over all $n$ tokens, so producing $N$ tokens costs $O(N^{2})$ passes worth of work. But the causal mask means the keys and values for tokens $1..n-1$ do not change when you append token $n$ — nothing later can alter an earlier token's representation. So you cache them. With a *KV cache*, each new step computes $Q$, $K$ and $V$ for the single new token only, appends its $K$ and $V$ to the cache, and attends over the cached history. The per-step cost drops from $O(n)$ to $O(1)$ in sequence passes, which is the difference between a usable system and a toy.

This reshapes the whole performance picture, and it is why serving engineers talk the way they do:

*Two distinct phases.* **Prefill** processes the entire prompt in one parallel pass — compute-bound, and it sets time-to-first-token. **Decode** produces one token per pass — *memory-bandwidth-bound*, because each step reads the entire weight matrix from memory to do a tiny amount of arithmetic on a single token. This is why decode throughput tracks memory bandwidth rather than FLOPs, why batching many requests together helps so much (the weights are read once and amortized across the batch), and why speculative decoding works — a small draft model proposes several tokens and the large model verifies them in one parallel pass, converting several memory-bound steps into one.

*The cache is large.* Its size is $2 \times L \times T \times d \times \text{bytes}$ per sequence — two for K and V, times layers, times sequence length, times width. For a 7B-class model at 32K context this reaches several gigabytes *per concurrent request*, which is usually what limits how many users you can serve on one GPU, not the model weights. This is the direct motivation for multi-query and grouped-query attention, which share key and value heads across query heads and shrink the cache by a large factor, and for KV cache quantization and paged attention.

**Temperature and top-p, with numbers.** For logits $[3,2,1,0]$:

| $T$ | probabilities |
|---|---|
| 0.5 | 0.865, 0.117, 0.016, 0.002 |
| 1.0 | 0.644, 0.237, 0.087, 0.032 |
| 2.0 | 0.455, 0.276, 0.167, 0.102 |

Low temperature sharpens toward the argmax (as $T \to 0$ it becomes greedy); high temperature flattens toward uniform. At $T=1$ the sorted cumulative probabilities are 0.644, 0.881, 0.968, 1.0, so nucleus sampling with $p=0.9$ keeps 3 tokens — the smallest set whose mass reaches 0.9. *(Computed directly.)*

**Why top-p usually beats top-k.** A fixed $k$ ignores the shape of the distribution. After "the capital of France is" the distribution is nearly a spike, and $k=50$ admits 49 tokens that are all wrong; after "she opened the door and saw a" the distribution is genuinely flat and $k=50$ truncates plausible continuations. Top-p adapts: it keeps few tokens when the model is confident and many when it is not. The deeper justification for truncation at all is that the aggregated tail of a 50,000-token vocabulary holds meaningful probability mass made of individually terrible tokens, and sampling from an untruncated distribution eventually draws one — after which the model must continue from its own mistake, which is how a generation derails.

**Repetition, and the honest caveat.** Greedy and beam decoding on open-ended text produce degenerate repetition loops, which is the core motivation for sampling. The common patches are a repetition penalty (divide the logits of already-generated tokens), a frequency or presence penalty, and n-gram blocking (forbid any n-gram that has already appeared). All are blunt: n-gram blocking will happily prevent a legitimately repeated name or a code identifier that must recur. Worth naming the tradeoff rather than presenting the penalties as free.

**Verified sampling implementation.** The following runs against the Q86 model and produces the correct continuation with both top-k and top-p:

```python
@torch.no_grad()
def generate(model, idx, n_new, temperature=1.0, top_k=None, top_p=None):
    for _ in range(n_new):
        logits, _ = model(idx[:, -model.max_len:])   # crop to context window
        logits = logits[:, -1, :] / temperature      # only the last position matters
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            s, si = torch.sort(logits, descending=True)
            probs = s.softmax(-1)
            drop = probs.cumsum(-1) - probs > top_p  # keep the token that crosses p
            s = s.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, si, s)
        idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
    return idx
```

Note the `cumsum - probs > top_p` rather than `cumsum > top_p`: it keeps the token that *crosses* the threshold, so the retained mass is at least $p$ rather than just under it. Off-by-one here silently changes your sampling distribution.

**Follow-up:** *When would you use greedy or beam search instead of sampling?* When there is a roughly correct answer and you want the most probable one: translation, structured extraction, code completion against a spec, or anything where you will parse the output. Use sampling for open-ended generation, where the highest-probability output is reliably bland and repetitive. Also note that greedy is not deterministic in practice across batch sizes or hardware, because floating-point reduction order changes — a fact that surprises people debugging reproducibility.

> **Why the interviewer asks this.** Decoding is where most people's understanding stops at "it predicts the next token," so it is a good place to find out who has actually served a model.

> **Saying it out loud.** Generation is autoregressive — one token at a time, each conditioned on everything before it. The implementation detail that matters most is the KV cache: naively you'd re-run the model over the whole sequence for every new token, which is quadratic, but the causal mask means earlier tokens' keys and values never change, so you cache them and each step only processes the one new token. That splits serving into two phases with completely different characteristics — prefill, which processes the prompt in parallel and is compute-bound, and decode, which is memory-bandwidth-bound because you read all the weights to produce one token. That's why batching helps so much and why speculative decoding works. For sampling, temperature sharpens or flattens the distribution, and I'd default to nucleus sampling over top-k, because a fixed k ignores the shape — after "the capital of France is" you want one token, and after "she opened the door and saw a" you want many.

---

### Q89: What is the complexity of attention? Explain O(n²d) and different attention types.

**Answer:**

**Standard Self-Attention: O(n²d)**

**Why O(n²d)?**
- Compute QK^T: (n, d) @ (d, n) → (n, n) matrix
- Each element: dot product of d-dimensional vectors
- n² elements × d operations = O(n²d)
- Apply to V: (n, n) @ (n, d) → O(n²d)
- Total: O(n²d) time, O(n²) space

**The n² term:**
- Attention matrix: n×n (all pairs of tokens)
- Each token attends to all other tokens
- Quadratic in sequence length

**The d term:**
- Model dimension (typically 768-12288)
- Vector operations scale with dimension

**Multi-Head Attention:**
- Still O(n²d) overall
- Divides d into h heads (d/h each)
- h heads × O(n²d/h) = O(n²d)
- Can parallelize across heads

**Linear Attention: O(nd²)**
- Reformulates: (QK^T)V = Q(K^T V)
- Compute K^T V first: O(nd²)
- Then Q @ (K^T V): O(nd²)
- Faster when n >> d

**Sparse Attention: O(n√n d) or O(n log n d)**
- Only attends to subset of tokens
- Local window + global tokens
- Reduces n² to n√n or n log n

**Flash Attention: O(n²d) time, O(n) space**
- Same computation, block-wise
- Doesn't store full attention matrix
- Memory efficient

**See `05_attention_mechanisms/attention_complexity.md` for complete analysis!**

**Getting the accounting exactly right.**

The derivation above is correct but leaves out the term that decides everything in practice. A full attention layer has two kinds of cost:

- The **projections** — computing $Q$, $K$, $V$ and the output projection — are four matmuls of $(n,d)\times(d,d)$, so $O(nd^{2})$.
- The **attention itself** — $QK^{\top}$ then $\text{softmax}\cdot V$ — is $O(n^{2}d)$.

Total $O(n^{2}d + nd^{2})$. Which term dominates is just the ratio $n/d$, and that single observation explains a lot of otherwise confusing behaviour:

| $n$ | $n^{2}d$ | $nd^{2}$ | dominant |
|---|---|---|---|
| 512 | $1.07\times10^{9}$ | $8.59\times10^{9}$ | projections |
| 1,024 | $4.29\times10^{9}$ | $1.72\times10^{10}$ | projections |
| 4,096 | $6.87\times10^{10}$ | $6.87\times10^{10}$ | equal |
| 16,384 | $1.10\times10^{12}$ | $2.75\times10^{11}$ | attention (4×) |
| 131,072 | $7.04\times10^{13}$ | $2.20\times10^{12}$ | attention (32×) |

*(Computed with $d=4096$.)* The crossover sits exactly at $n=d$. So at the sequence lengths typical of BERT-era training, the quadratic term was not the bottleneck at all and the feed-forward and projection matmuls dominated — which is why "attention is quadratic and therefore the problem" was somewhat overstated for years, and why it became genuinely urgent only once contexts pushed past ten thousand tokens.

**FlashAttention deserves a sharper explanation than the answer above gives.** Saying "same computation, block-wise, memory efficient" undersells it. The real point is that attention is *memory-bandwidth-bound*, not compute-bound: the $n\times n$ score matrix must be written to high-bandwidth memory, read back for the softmax, written again, and read again for the multiply by $V$. At $n=131{,}072$ that matrix alone is about 34 GB in fp16 per head-batch — it does not fit anywhere sensible. FlashAttention tiles the computation so that blocks of Q, K and V are loaded into fast on-chip SRAM, computes the partial attention there, and combines results using the *online softmax* trick — an incremental formulation that maintains a running maximum and running sum so the normalization can be corrected as new blocks arrive, without ever materializing a full row. Memory drops from $O(n^{2})$ to $O(n)$ and, because far less data crosses the memory bus, wall-clock time improves substantially even though the FLOP count is unchanged. It is exact, not an approximation — which is why it is now the default everywhere and why the approximate-attention literature lost much of its motivation.

**A caveat on linear attention.** $O(nd^{2})$ via the associativity $(QK^{\top})V = Q(K^{\top}V)$ is real, but it requires replacing the softmax with a kernel feature map $\phi$, since softmax does not factorize — you compute $\phi(Q)(\phi(K)^{\top}V)$. That changes what the model computes, and quality has historically lagged full attention, particularly on tasks needing precise retrieval from context. It also flips the constant: $d^{2}$ with $d=4096$ is large, so linear attention only wins for genuinely long sequences. The modern descendants of this idea — state-space models and linear-recurrent architectures — are more competitive, and hybrid designs that interleave a few full-attention layers among many linear ones are a common compromise. *This area is moving quickly; verify the current state before asserting what is standard.*

**The generation-time picture is different, and is worth adding.** All the above concerns training or prefill. With a KV cache during decode (Q88), generating one token costs $O(nd)$ attention against the cached history plus $O(d^{2})$ of projections — linear in context, not quadratic. The binding constraint at decode time is *memory*: the KV cache is $O(nLd)$ and it is what limits concurrency on a serving GPU. So "attention is quadratic" is a training and prefill statement; at decode the problem is the cache. Distinguishing the two is the detail that marks someone who has profiled a real system.

**Follow-up:** *If attention were free, would context be unlimited?* No. Two other limits bind. The KV cache still grows linearly and still exhausts memory. And model *quality* over long context does not scale with the window — the lost-in-the-middle effect (Q59) means retrieval accuracy from deep context degrades regardless of how cheaply you computed the attention. Longer windows are an infrastructure achievement that outruns the modelling.

> **Why the interviewer asks this.** Complexity questions separate people who have read the paper from people who have profiled a model, and the tell is whether you mention the $nd^{2}$ term and the memory bandwidth.

> **Saying it out loud.** Attention itself is n-squared times d, because you build an n-by-n score matrix and each entry is a d-dimensional dot product. But the projections — Q, K, V and the output — are n times d-squared, and which one dominates depends on whether n is bigger or smaller than d. At a sequence length of 512 with a model width of 4096, the projections actually dominate; the crossover is right at n equals d. That's why the quadratic term only became urgent once contexts got long. The other thing I'd say is that attention is memory-bandwidth-bound, not compute-bound — you write the whole score matrix out to memory and read it back. That's what FlashAttention fixes: it tiles the computation into on-chip memory and uses an online softmax so it never materializes the full matrix. Same math, exact, but memory goes from n-squared to n and it's much faster in wall-clock terms.

---

See `04_transformers/gpt_complete.py` for complete GPT implementation!
See `04_transformers/gpt_training_decoding.md` for training and decoding details!
See `05_attention_mechanisms/attention_complexity.md` for complexity analysis!

---

## Prompt Tuning and Prefix Tuning

### Q90: What is prompt tuning? How does it work?

**Answer:**

**Prompt Tuning:**
- Parameter-efficient fine-tuning method
- Adds trainable "soft prompts" (continuous embeddings) to input
- Keeps entire pre-trained model frozen
- Only trains prompt embeddings (typically 20-100 tokens)

**How It Works:**
1. Prepend trainable prompt embeddings to input
2. Pass [prompt; input] through frozen model
3. Only update prompt embeddings during training
4. Prompt learns to encode task-specific information

**Mathematical Formulation:**
```
E_input = Embedding(x)  # Input embeddings
P = [p₁, ..., pₚ]  # Trainable prompt (p tokens)
E_combined = [P; E_input]  # Concatenate
output = Model_θ(E_combined)  # Model frozen
# Only P is updated: P ← P - α∇P
```

**Parameters:**
- Trainable: p × d_model (e.g., 20 × 768 = 15,360)
- Efficiency: 0.01% of model parameters
- Storage: Only prompt embeddings per task

**Advantages:**
- Extremely parameter-efficient
- Simple implementation
- Fast training
- Preserves pre-trained knowledge
- Enables multi-task deployment

**Working Through the Parameter Count:**

The trainable size is just $p \times d_{model}$, where $p$ is the number of soft-prompt tokens and $d_{model}$ is the hidden width. For a GPT-2 small backbone, $20 \times 768 = 15{,}360$ floats. Against 124M backbone parameters that is $15{,}360 / 124{,}000{,}000 \approx 0.012\%$. Note what this number does *not* depend on: the number of layers, or the vocabulary size. Prompt tuning is the only PEFT method whose cost is flat in model depth, which is why the per-task storage stays in the tens of kilobytes even for a 70B model (a 70B model with $d_{model} = 8192$ and 20 tokens is 163,840 floats, roughly 320 KB in fp16).

**Why the Learning Rate Is So Large:**

Soft prompts are typically trained with learning rates around $0.1$ to $0.5$ — a thousand times higher than the $10^{-5}$ used for full fine-tuning. The reason is that gradients reach the prompt only through the frozen stack, so the signal arriving at $P$ is small, and there is no risk of damaging pretrained weights because there are none in the optimizer. A candidate who quotes `lr=3e-5` for prompt tuning usually has not actually run it.

**The Scale Caveat:**

The original result (Lester et al., 2021) is that prompt tuning only *closes the gap* with full fine-tuning at large scale — around 10B parameters and up. On a sub-1B model it typically underperforms LoRA noticeably. So "prompt tuning is as good as fine-tuning" is a claim with a size condition attached, and interviewers like to check whether you know it.

**Follow-up:** *Does the soft prompt consume context window?* Yes. The $p$ prompt tokens occupy positions in the sequence, so a 20-token prompt costs 20 tokens of usable context at every forward pass and adds $O(p \cdot n)$ extra attention work. LoRA has no such cost.

> **Why the interviewer asks this.** It separates people who have read the PEFT papers from people who have only used the word "prompt" to mean text they typed into a chat box.

> **Saying it out loud.** Prompt tuning freezes the whole model and learns a handful of fake token embeddings that you glue onto the front of every input. They're not real words — they're just vectors in embedding space that gradient descent found useful for the task. You end up training something like fifteen thousand numbers instead of a hundred and twenty million, so each task is a tiny file you can swap in. The catch is that it really only matches full fine-tuning once the base model is big, roughly ten billion parameters and up.

---

### Q91: What is prefix tuning? How does it differ from prompt tuning?

**Answer:**

**Prefix Tuning:**
- Similar to prompt tuning but adds parameters at every layer
- Adds trainable "prefix" key-value pairs at each transformer layer
- More expressive than prompt tuning
- Still parameter-efficient

**Key Differences:**

**1. Where Parameters Are Added:**
- **Prompt tuning**: Only at input layer
- **Prefix tuning**: At every transformer layer
- **Impact**: Prefix influences model at multiple levels

**2. What's Added:**
- **Prompt tuning**: Prompt embeddings (input)
- **Prefix tuning**: Prefix keys and values (attention)
- **Impact**: Prefix directly modifies attention computation

**3. Parameters:**
- **Prompt tuning**: p × d_model
- **Prefix tuning**: L × p × 2d_model (for K and V)
- **Example**: 12 layers, 20 tokens, 768 dim
  - Prompt: 15,360 parameters
  - Prefix: ~368,640 parameters
  - Still much less than full model

**4. Performance:**
- **Prompt tuning**: Good for simple tasks
- **Prefix tuning**: Often matches full fine-tuning
- **Trade-off**: More parameters for better performance

**Mathematical Formulation:**
```
At each layer l:
K_l = [P_l^K; K_l]  # Add prefix keys
V_l = [P_l^V; V_l]  # Add prefix values
Q_l unchanged
Attention_l = softmax(Q_l K_l^T) V_l
```

**Checking the Parameter Arithmetic:**

The formula $L \times p \times 2 \times d_{model}$ with $L = 12$ layers, $p = 20$ prefix tokens and $d_{model} = 768$ gives $12 \times 20 \times 2 \times 768 = 368{,}640$. The factor of 2 is because you store one prefix for keys and one for values. That is 24 times more than prompt tuning's 15,360 for the same 20 tokens — the multiplier is exactly $2L$.

**What "Prepending to K and V" Actually Means:**

At layer $l$, ordinary self-attention over a sequence of length $n$ computes $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{n \times d}$, $V \in \mathbb{R}^{n \times d}$. Prefix tuning concatenates $p$ learned rows on top of $K$ and $V$ only, giving $K' \in \mathbb{R}^{(n+p) \times d}$ and $V' \in \mathbb{R}^{(n+p) \times d}$, while $Q$ stays at $n$ rows. The attention matrix becomes $n \times (n+p)$ instead of $n \times n$: every real token gets $p$ extra things it may attend to, but the prefix positions themselves never produce outputs. That asymmetry is the whole trick — the prefix is pure memory, never a query.

**Why the Reparameterization Exists:**

Training the prefix matrices directly is unstable; the loss is very sensitive to the learning rate and often diverges. The paper's fix is to learn a smaller matrix $P' \in \mathbb{R}^{p \times d'}$ and pass it through an MLP to produce the real prefixes, then throw the MLP away after training and keep only the expanded $P$. So the deployed artifact is still just $2Ld_{model}p$ numbers; the extra machinery is a training-time crutch.

**Follow-up:** *Is prefix tuning the same as P-tuning v2?* Effectively yes for the mechanism — P-tuning v2 is deep prompt tuning that injects trainable key/value prefixes at every layer, which is prefix tuning without the reparameterization MLP, retuned for classification and sequence labelling.

> **Saying it out loud.** Prompt tuning only touches the input embeddings, so the model's deeper layers have to be steered indirectly. Prefix tuning pushes learned vectors into the keys and values at every single layer, so it can nudge the computation all the way up the stack. You pay about two-L times more parameters — twenty-four times for a twelve-layer model — but you get much closer to full fine-tuning on hard tasks. The prefix is attended to, but it never attends to anything itself.

---

### Q92: Compare prompt tuning, prefix tuning, LoRA, and full fine-tuning.

**Answer:**

**Parameter Efficiency:**

| Method | Parameters | Example (GPT-2) | Efficiency |
|--------|-----------|-----------------|------------|
| **Full Fine-tuning** | 100% | 125M | 1x |
| **LoRA** | 0.1-1% | 125K-1.25M | 100-1000x |
| **Prefix Tuning** | 0.3% | ~368K | ~340x |
| **Prompt Tuning** | 0.01% | ~15K | ~8000x |

**Performance:**

**Full Fine-tuning:**
- Best performance
- Risk of catastrophic forgetting
- Requires most resources

**LoRA:**
- Near full fine-tuning performance
- Best balance of efficiency and performance
- Most popular in practice

**Prefix Tuning:**
- Very good performance (often matches full fine-tuning)
- More expressive than prompt tuning
- Good for complex tasks

**Prompt Tuning:**
- Good performance
- Sufficient for many tasks
- Maximum efficiency

**Use Cases:**

- **Full Fine-tuning**: Maximum performance, single task
- **LoRA**: Best balance, most common
- **Prefix Tuning**: Complex tasks, good performance
- **Prompt Tuning**: Simple tasks, maximum efficiency

**Recommendation:**
- Start with prompt tuning (simplest)
- If insufficient, try prefix tuning
- For best balance, use LoRA
- Full fine-tuning only if needed

**Reading the Table:**

The middle column is the same GPT-2-sized model (roughly 125M parameters) under each method, so the numbers are directly comparable. Full fine-tuning updates all 125M. LoRA at rank $r$ on the query and value projections of $L$ layers costs $2 \times L \times 2 \times r \times d_{model}$ parameters — for $L=12$, $r=8$, $d=768$ that is $2 \times 12 \times 2 \times 8 \times 768 = 294{,}912$, about $0.24\%$. Prefix tuning's 368K and prompt tuning's 15K come from the two previous questions. The "Efficiency" column is just $125\text{M}$ divided by the trainable count, so it is a storage-and-optimizer-state ratio, not a speed ratio.

**The Distinction the Table Hides:**

Trainable-parameter count is *not* the same as training cost. All four methods still run a full forward and backward pass through the whole network — you save optimizer state and gradient memory, not FLOPs. Adam keeps two moments per trainable parameter, so full fine-tuning of a 7B model needs roughly $7\text{B} \times (4 + 4 + 4) = 84$ GB just for fp32 weights plus moments, while LoRA at 0.1% needs about 84 MB for the same states. That memory difference, not compute, is why PEFT lets you fine-tune on one GPU.

**The One Property Only LoRA Has:**

LoRA's update is $W + \frac{\alpha}{r}BA$, a plain additive change to a weight matrix, so it can be *merged* into the base weights after training and adds exactly zero inference latency. Prompt and prefix tuning both lengthen the sequence, so they cost extra attention work at every token forever. If the interviewer asks "which of these is free at inference time," the answer is LoRA and only LoRA.

**Follow-up:** *Which would you actually reach for in 2026?* LoRA or QLoRA, essentially always — it dominates on the quality-per-parameter curve, merges away at inference, and has the most mature tooling. Prompt and prefix tuning are worth knowing mainly because they are asked about and because prompt tuning's flat-in-depth cost is genuinely appealing for serving hundreds of tasks off one frozen backbone.

> **Why the interviewer asks this.** They want to see whether you can rank methods on more than one axis at once — parameters, quality, and inference cost do not agree, and the interesting answer lives in the disagreement.

> **Saying it out loud.** All four of these do the same job — adapt a pretrained model to a task — but they trade off differently. Full fine-tuning is best and most expensive. LoRA gets you within noise of it for about a tenth of a percent of the parameters, and crucially you can merge it back into the weights so inference costs nothing extra. Prefix tuning is close behind, prompt tuning is the cheapest and works best on very large models. In practice I'd default to LoRA and only move off it for a specific reason.

---

### Q93: How do you initialize prompt/prefix embeddings?

**Answer:**

**Initialization Strategies:**

**1. Random Initialization:**
```
P ~ N(0, 0.02²)  # Small random values
```
- Simple, unbiased
- May require more training

**2. Vocabulary-Based:**
```
Sample random tokens from vocabulary
Use their embeddings as initial prompt
```
- Starts with semantic information
- Often works better than random

**3. Task-Specific:**
```
Use embeddings from task-related tokens
E.g., sentiment: "sentiment", "positive", "negative"
```
- Better starting point
- Faster convergence

**4. Reparameterization (Prefix):**
```
Learn in smaller space (d_model/2)
Project up to full dimension
```
- More stable training
- Used in prefix tuning

**Best Practices:**
- **Prompt tuning**: Vocabulary-based initialization
- **Prefix tuning**: Reparameterization + random
- **Experiment**: Try different strategies
- **Use domain knowledge**: When available

**Why Initialization Matters More Here Than Usual:**

In full fine-tuning, initialization barely matters because you start from pretrained weights and take small steps. In prompt tuning you are creating brand-new vectors in a space the model has strong opinions about — the embedding matrix occupies a particular region, with a typical norm and a particular anisotropic shape. Random $\mathcal{N}(0, 0.02^2)$ vectors can land far outside that region, so the frozen model treats them as out-of-distribution garbage and gradients are weak until they wander back. Sampling real vocabulary embeddings starts you inside the manifold the model already understands.

**Concretely, Vocabulary-Based Initialization:**

Take the $p$ most frequent tokens in the vocabulary (or tokens from the task's label words), look up their rows in the frozen embedding table, and copy those as the initial $P$. Lester et al. found that on smaller models this beats random initialization by several points; on very large models the gap closes, because a big model can recover from a bad start.

**Class-Label Initialization for Classification:**

For a sentiment task with labels "positive" and "negative", initializing some prompt slots with the embeddings of those exact words gives the model a head start on the output distribution. This is closer to converting a hand-written discrete prompt into a soft one, and it is the strongest option when the task has natural verbalizer words.

> **Saying it out loud.** The thing to remember is that a soft prompt lives in the same space as real token embeddings, so if you initialize it with pure Gaussian noise you're handing the frozen model vectors that look like nothing it's ever seen. Copying embeddings of actual vocabulary tokens — ideally words related to the task, like the label words for a classification problem — starts you somewhere the model already understands. It matters a lot on small models and less on huge ones.

---

### Q94: What is the optimal prompt/prefix length?

**Answer:**

**Typical Ranges:**
- **Prompt tuning**: 20-100 tokens (commonly 20-50)
- **Prefix tuning**: 10-50 tokens (commonly 10-20 per layer)

**Selection Factors:**

**1. Task Complexity:**
- Simple tasks: 20 tokens sufficient
- Complex tasks: 50-100 tokens needed
- Rule: More complex → longer prompt/prefix

**2. Dataset Size:**
- Large datasets: Can support longer prompts
- Small datasets: Shorter prompts (avoid overfitting)

**3. Model Size:**
- Larger models: Can utilize longer prompts
- Smaller models: Shorter prompts sufficient

**Selection Process:**
1. Start with moderate length (20-30 tokens)
2. Try different lengths: [10, 20, 50, 100]
3. Evaluate on validation set
4. Choose best performing length

**Empirical Finding:**
- Performance improves with length up to a point
- Then plateaus (diminishing returns)
- Sweet spot: 20-50 tokens for most tasks

**What Length Actually Buys You:**

The soft prompt is the entire capacity of the method — it is the only thing that can encode the task. So length is the capacity knob, exactly like rank $r$ in LoRA. With $d_{model} = 768$, a 20-token prompt has 15,360 degrees of freedom and a 100-token prompt has 76,800. The empirical curve is steep from 1 to about 20 tokens, then close to flat: Lester et al. report that going beyond 20 gives little on a 10B+ model, while smaller models keep improving out to 100.

**The Costs of Going Long:**

Every prompt token is a real sequence position. With prompt length $p$ and input length $n$, the attention cost goes from $O(n^2)$ to $O((n+p)^2)$, and the prompt eats $p$ tokens of your context budget on every single example. At $p = 100$ on a 512-token context you have given up nearly 20% of the window before the user has said anything. That is the practical ceiling, not overfitting.

**How to Actually Pick It:**

Sweep $p \in \{1, 5, 20, 50, 100\}$ on a validation set, plot score against $p$, and take the smallest $p$ within noise of the best. This is a one-dimensional search over a cheap-to-train model, so it is one of the few hyperparameter sweeps that is genuinely affordable.

**Follow-up:** *Does prefix tuning need the same length?* No, it can be shorter — 10 to 20 is typical — because it gets $2L$ times more parameters out of each token. Capacity per prefix token is much higher, so you need fewer of them.

> **Saying it out loud.** Prompt length is the capacity dial, same idea as rank in LoRA. You get most of the benefit by about twenty tokens and it flattens out after that, so people usually land somewhere between twenty and fifty. The reason not to just crank it to a hundred is that every prompt token is a real position in the sequence — it costs you context window and attention compute on every forward pass. I'd sweep a few values on validation and take the smallest one that's within noise of the best.

---

### Q95: Implement prompt tuning from scratch. Show the key code.

**Answer:**

**Key Implementation:**

```python
class PromptTuning(nn.Module):
    def __init__(self, model, prompt_length=20):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        self.d_model = model.config.n_embd
        
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
        
        # Trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, self.d_model) * 0.02
        )
    
    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        
        # Input embeddings
        input_emb = self.model.transformer.wte(input_ids)
        
        # Expand prompt for batch
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: [prompt; input]
        combined = torch.cat([prompt, input_emb], dim=1)
        
        # Forward through frozen model
        outputs = self.model.transformer(inputs_embeds=combined)
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        return logits

# Training
optimizer = torch.optim.Adam([prompt_model.prompt_embeddings], lr=0.3)
# Only prompt_embeddings are updated
```

**Key Points:**
- Freeze all model parameters
- Only prompt_embeddings requires gradients
- Simple concatenation at input
- Extremely parameter-efficient

**See `25_adapters_lora/prompt_prefix_code.py` for complete implementation!**

**Walking Through the Code:**

The constructor does three things. It records `d_model` from the backbone config so the prompt vectors have the right width. It sets `requires_grad = False` on every backbone parameter, which is what makes the model frozen — without this line the optimizer would still only see the prompt, but the backward pass would waste time computing and storing gradients for 124M weights. And it registers `prompt_embeddings` as an `nn.Parameter` of shape `(prompt_length, d_model)`, scaled by `0.02` to match GPT-2's embedding initialization scale.

In `forward`, `wte` is GPT-2's word token embedding table, so `input_emb` has shape `(batch, seq_len, d_model)`. The prompt is `(prompt_length, d_model)`; `unsqueeze(0)` makes it `(1, prompt_length, d_model)` and `expand` broadcasts it to `(batch, prompt_length, d_model)` without copying memory. The `torch.cat` along `dim=1` produces `(batch, prompt_length + seq_len, d_model)`, which is handed to the transformer as `inputs_embeds` rather than `input_ids` — that is the only way to feed vectors that do not correspond to any real token.

**Two Things This Snippet Leaves Out (worth saying in an interview):**

First, the output logits now cover `prompt_length + seq_len` positions, so when you compute the loss you must slice off the first `prompt_length` positions before comparing to labels — otherwise you are training the model to predict your prompt. Second, if you pass an `attention_mask` for padding you must left-pad it with `prompt_length` ones, or the model will mask out its own prompt. Both are the classic bugs in hand-rolled prompt tuning.

**Verified Shape Arithmetic:**

```python
import torch
B, S, P, D = 4, 16, 20, 768
input_emb = torch.zeros(B, S, D)
prompt = torch.zeros(P, D).unsqueeze(0).expand(B, -1, -1)
combined = torch.cat([prompt, input_emb], dim=1)
print(combined.shape)          # torch.Size([4, 36, 768])
print(P * D)                   # 15360 trainable parameters
logits = torch.zeros(B, P + S, 50257)
print(logits[:, P:, :].shape)  # torch.Size([4, 16, 50257]) -- slice before loss
```

**Follow-up:** *Why `lr=0.3` in the optimizer line?* Because the prompt is the only trainable tensor and its gradients, arriving through a deep frozen stack, are tiny. Prompt tuning is routinely run at learning rates between $0.1$ and $0.5$; at $10^{-5}$ it would essentially not move.

> **Why the interviewer asks this.** Asking for code exposes whether you have actually run the method — the loss-slicing and attention-mask details only surface when something has gone wrong for you once.

> **Saying it out loud.** The whole implementation is about fifteen lines. You freeze every backbone parameter, create one trainable tensor shaped prompt-length by hidden-size, embed the input normally, and concatenate the prompt in front along the sequence dimension. Then you feed it in as `inputs_embeds` instead of `input_ids`, because these vectors don't correspond to any real token. The two bugs everyone hits are forgetting to slice the prompt positions off the logits before computing loss, and forgetting to pad the attention mask.

---

See `25_adapters_lora/prompt_prefix_tuning.md` for detailed theory!
See `25_adapters_lora/prompt_prefix_code.py` for complete code!
See `25_adapters_lora/prompt_prefix_qa.md` for comprehensive Q&A!

---

## Diffusion Models

### Q96: What are diffusion models? How do they work?

**Answer:**

**Diffusion Models:**
- Generative models that learn to reverse a gradual noising process
- Work by iteratively removing noise from data, starting from pure noise
- State-of-the-art results in image generation (DALL-E, Stable Diffusion)

**How They Work:**

**1. Forward Process (Fixed):**
- Gradually add Gaussian noise to data
- q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
- After T steps, data becomes pure noise

**2. Reverse Process (Learned):**
- Learn to remove noise step by step
- p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
- Neural network predicts how to denoise

**3. Training:**
- Predict the noise that was added
- Loss: L = E[||ε - ε_θ(x_t, t)||²]

**4. Generation:**
- Start from pure noise x_T ~ N(0, I)
- Iteratively apply reverse process: x_T → x_{T-1} → ... → x_0

**Key Insight:**
- Break down complex generation into many simple denoising steps
- Each step only removes small amount of noise
- Much easier to learn than generating directly

**The One Equation That Makes Training Cheap:**

Written literally, the forward process is a chain: to get $x_t$ you would apply the noising step $t$ times. But because each step is Gaussian and the composition of Gaussians is Gaussian, the chain collapses into a closed form. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$. Then

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t)I\right)$$

which you sample as $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. This is the single most important fact about diffusion training: you can jump to *any* timestep in one line, so a training step never simulates the chain. Without it, training would cost $O(T)$ per example.

**Worked Numbers:**

With a linear schedule $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$ steps, $\bar{\alpha}_{1000} \approx 4 \times 10^{-5}$, so $\sqrt{\bar{\alpha}_T} \approx 0.0063$ — the original image contributes essentially nothing and $x_T$ is indistinguishable from pure noise. At $t = 100$, $\bar{\alpha}_{100} \approx 0.90$, so $x_{100}$ is still about 95% signal by amplitude. That spread is why the model must be told $t$: the same network handles "barely noisy" and "pure static," and only the timestep embedding tells it which regime it is in.

**Why Predict Noise Instead of the Image:**

Predicting $\epsilon$ rather than $x_0$ makes the target have unit variance at every timestep, so the loss is comparably scaled across $t$ and the network does not have to learn a $t$-dependent output magnitude. It is a variance-reduction trick, and it is why the standard objective is $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ rather than a reconstruction loss.

**Follow-up:** *Why so many steps?* Each reverse step is only approximately Gaussian, and the approximation is good only when $\beta_t$ is small. Many small steps keep the Gaussian assumption valid. Deterministic samplers like DDIM exploit the same trained model with 20-50 steps by taking a non-Markovian path, which is how production image models sample fast.

> **Why the interviewer asks this.** Diffusion is the one modern generative family whose training objective looks nothing like next-token prediction, so it is a clean test of whether you understand a model you did not memorize.

> **Saying it out loud.** A diffusion model learns to undo noise. You take real data and gradually corrupt it with Gaussian noise until, after a thousand steps, it's pure static — and that direction is fixed, no learning involved. Then you train a network to look at a noisy sample plus the timestep and guess what noise was added. To generate, you start from static and denoise step by step. The clever bit is that you can jump straight to any noise level in closed form, so training one example is one forward pass, not a thousand.

---

### Q97: How do you train a diffusion model?

**Answer:**

**Training Algorithm:**

**1. Setup:**
- Define variance schedule β_t (linear or cosine)
- Precompute α_t, ᾱ_t for efficiency

**2. Training Loop:**
```
For each batch:
  a. Sample data: x_0 ~ q(x_0)
  b. Sample timestep: t ~ Uniform({1, 2, ..., T})
  c. Sample noise: ε ~ N(0, I)
  d. Create noisy data: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
  e. Predict noise: ε_pred = ε_θ(x_t, t)
  f. Compute loss: L = ||ε - ε_pred||²
  g. Update: θ ← θ - α∇_θ L
```

**Best Practices:**
- Learning rate: 1e-4 to 1e-3
- Use learning rate scheduling (cosine annealing)
- Gradient clipping (norm = 1.0)
- Monitor loss and generate samples during training

**Variance Schedule:**
- Linear: β_t = (β_max - β_min) * (t/T) + β_min
- Cosine: ᾱ_t = cos²(π/2 * (t/T)) (often better)

**Reading the Training Loop Line by Line:**

Steps (a) through (d) are just "make one training example." You draw a clean sample $x_0$, draw a *random* timestep $t$ uniformly from $1..T$ — this is important, each example in a batch gets a different noise level, so the batch covers the whole schedule — draw fresh noise $\epsilon$, and form $x_t$ with the closed-form jump from the previous question. Steps (e) through (g) are an ordinary regression: the network sees $(x_t, t)$ and must output $\epsilon$, and you take an MSE gradient step. Nothing here iterates over $T$; the loop body is $O(1)$ in the number of diffusion steps.

**Why the Cosine Schedule Usually Wins:**

The linear schedule destroys information too early: $\bar{\alpha}_t$ drops fast, so a large fraction of the $T$ timesteps are spent on samples that are already essentially pure noise and carry no learning signal. Nichol and Dhariwal's cosine schedule keeps $\bar{\alpha}_t$ near 1 for longer and decays smoothly near the end, spending more of the budget on the noise levels where the model actually has something to learn. The practical effect is better sample quality at the same $T$, especially at low resolution.

**Two Practical Details the Answer Omits:**

Use an exponential moving average of the weights for sampling — EMA with decay around $0.9999$ is close to universal in diffusion training and the difference in sample quality is large, not marginal. And condition on $t$ with sinusoidal position embeddings passed into every residual block (usually via FiLM-style scale-and-shift), not just concatenated at the input; the network needs the timestep at every depth.

**Follow-up:** *How do you make it conditional, e.g. text-to-image?* Add the conditioning signal $c$ to the network and train $\epsilon_\theta(x_t, t, c)$, dropping $c$ at random on maybe 10% of examples. At sampling time you extrapolate between the conditional and unconditional prediction — classifier-free guidance, $\hat{\epsilon} = \epsilon_\theta(x_t,t,\emptyset) + w(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\emptyset))$ — which trades diversity for prompt adherence as $w$ grows.

> **Saying it out loud.** Training is surprisingly plain. For each example you pick a random timestep, sample some noise, use the closed-form formula to build the noisy version in one shot, and ask the network to predict the noise you just added. It's mean-squared error regression. The parts people forget are that the timestep has to be fed into every block, not just the input, and that you should keep an EMA copy of the weights for sampling — that one alone makes a visible difference.

---

### Q98: What are discrete diffusion models? How do they work for NLP?

**Answer:**

**The Challenge:**
- Standard diffusion works on continuous data (images)
- Text is discrete (tokens), need adaptation

**Discrete Forward Process:**

Instead of Gaussian noise, use transition matrix:
```
q(x_t | x_{t-1}) = Categorical(x_t; Q_t x_{t-1})
```

**Common Approaches:**

**1. Absorbing State:**
- Have special [MASK] token
- At each step, tokens transition to [MASK] with probability β_t
- After T steps, all tokens become [MASK]

**2. Uniform Transition:**
- Tokens can transition to any other token uniformly

**Discrete Reverse Process:**

Learn to predict original token:
```
p_θ(x_{t-1} | x_t) = Categorical(x_{t-1}; p_θ(x_t, t))
```

**Advantages for NLP:**
- Non-autoregressive (can generate in parallel)
- Better for editing tasks (text inpainting)
- More flexible control

**What the Transition Matrix Means:**

For continuous data, "add noise" means adding a Gaussian. For tokens there is no meaningful way to add $0.3$ to the word "cat," so corruption has to be a *random substitution*. Represent a token as a one-hot vector $x \in \{0,1\}^{|V|}$; then $Q_t$ is a $|V| \times |V|$ column-stochastic matrix and $Q_t x$ is the distribution over what the token becomes at step $t$. The absorbing-state variant sets $Q_t$ so that a token stays put with probability $1-\beta_t$ and jumps to `[MASK]` with probability $\beta_t$; `[MASK]` never leaves, which is why it is called absorbing.

**Worked Example:**

Take a 10-token sentence and a schedule where the cumulative mask probability reaches 1 at $t=T$. At $t$ with $\bar{\beta}_t = 0.3$, about 3 of the 10 tokens are `[MASK]` and the model must fill them in given the other 7. At $\bar{\beta}_t = 0.9$, 9 tokens are masked and the task is nearly unconditional generation. Absorbing-state discrete diffusion is therefore *BERT-style masked prediction with a randomized, annealed mask ratio* — that framing is the single most useful sentence to have ready, and it makes the connection to modern masked diffusion language models obvious.

**Why Uniform Transitions Are Worse in Practice:**

With uniform corruption, a token can become any other token, so at inference the model cannot tell which positions are corrupted — it must decide both *what* is wrong and *how* to fix it. The absorbing variant marks corruption explicitly with `[MASK]`, which is a much easier learning problem and is what most working discrete-diffusion text models use.

**Follow-up:** *How does generation actually decide which tokens to unmask?* Typically confidence-based: the model predicts a distribution for every masked position, and at each reverse step you commit the $k$ positions where it is most confident and re-mask the rest. That is why these models can trade quality for latency by changing the number of steps — fewer steps means committing more tokens per step.

> **Why the interviewer asks this.** It checks whether you can transfer a continuous-domain idea to a discrete one rather than just reciting the image-diffusion recipe.

> **Saying it out loud.** Text is discrete, so you can't add Gaussian noise to it. Instead the forward process randomly replaces tokens — and the version that works best replaces them with a mask token that can never change back, so after enough steps everything is masked. The reverse process predicts the original tokens. If that sounds like BERT, it basically is, except the mask ratio is randomized and annealed, and generation runs it repeatedly, unmasking the most confident positions each round.

---

### Q99: What are use cases of diffusion models in NLP?

**Answer:**

**1. Non-Autoregressive Text Generation:**
- Generate all tokens in parallel
- Faster than autoregressive models
- Better for controlled generation

**2. Text Inpainting:**
- Fill in masked tokens
- Edit specific parts of text
- Example: "The [MASK] sat on the [MASK]" → "The cat sat on the mat"

**3. Text-to-Image:**
- DALL-E, Stable Diffusion
- Generate images from text descriptions
- Multimodal understanding

**4. Text Editing:**
- Style transfer
- Paraphrasing
- Rewriting with constraints

**5. Controllable Generation:**
- Generate with specific attributes
- Control length, style, topic
- More flexible than autoregressive

**Industry Examples:**
- DALL-E: Text-to-image generation
- Stable Diffusion: Open-source text-to-image
- Research: Non-autoregressive text generation

> **Correction to the list above.** DALL-E 1 (2021) was *not* a diffusion model — it was an autoregressive transformer over discrete VQ-VAE image tokens. Diffusion entered the DALL-E line with DALL-E 2 (unCLIP, 2022) and continued in DALL-E 3. Stable Diffusion is correctly described: it is a latent diffusion model, meaning the diffusion runs in a compressed VAE latent space (roughly $64 \times 64$ for a $512 \times 512$ image) rather than pixel space, which is a 48x reduction in the number of dimensions being denoised and is the reason it runs on consumer hardware.

**Where the Text Cases Actually Stand (as of 2026 — this is the fastest-moving item in this section):**

Diffusion-style *masked* language models have moved from research curiosity to shipped products in the last two years, marketed on latency: because they decode many tokens per network call instead of one, they can post very high tokens-per-second on the same hardware. They remain a minority of deployed text generation, and autoregressive decoding still dominates for open-ended quality and for anything needing a long, coherent chain of reasoning. Treat any specific ranking here as having a short shelf life.

**The Structural Reason Diffusion Fits Editing:**

An autoregressive model conditions only on the left. To rewrite the middle of a document it must either regenerate everything downstream or be trained with a special infilling objective. A diffusion model conditions on *everything unmasked*, in both directions, at every step — so "fill this hole given the surrounding text" is not a special case, it is the native operation. That is why the honest use cases for text diffusion are infilling, constrained rewriting, and parallel decoding, rather than "replacing GPT."

> **Saying it out loud.** For images, diffusion won outright — Stable Diffusion and the later DALL-E models are all diffusion, though the original DALL-E actually wasn't. For text it's a narrower story. The genuine advantage is that diffusion conditions on both sides at once, so filling in a gap or rewriting the middle of a paragraph is the natural operation rather than a special case. And because it decodes many tokens per pass, it can be very fast. But autoregressive models still own general-purpose text generation.

---

### Q100: How do you evaluate diffusion models?

**Answer:**

**For Images:**

**1. FID (Frechet Inception Distance):**
- Measures quality and diversity
- Lower is better
- Compares feature distributions

**2. IS (Inception Score):**
- Measures quality and diversity
- Higher is better (typically 1-10)

**3. Reconstruction Error:**
- Test if model can recover original
- Lower is better

**For Text:**

**1. BLEU Score:**
- Measures n-gram overlap with reference
- Higher is better (0-1)

**2. Perplexity:**
- Measures how well model predicts tokens
- Lower is better

**3. Diversity Metrics:**
- Distinct-n: Ratio of unique n-grams
- Self-BLEU: Average BLEU between samples
- Higher distinct = more diverse

**Diffusion-Specific:**

**1. Denoising Accuracy:**
- Test accuracy at each timestep
- Measures how well model denoises

**2. Sample Quality:**
- Visual inspection (for images)
- Human evaluation (for text)

> **Correction on Inception Score range.** IS is not "typically 1-10." It is bounded below by 1 and above by the number of classes in the classifier — 1000 for ImageNet-trained Inception. Real ImageNet generative models score in the tens to low hundreds; real ImageNet data itself scores around 233. A "1-10" band would describe only a badly broken model.

**How FID Is Actually Computed:**

Push $N$ real images and $N$ generated images through an Inception-v3 network and take the 2048-dimensional pool3 activations. Fit a Gaussian to each set — mean $\mu$ and covariance $\Sigma$ — and compute the Frechet distance between them:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$$

Two things follow that interviewers probe. FID is *biased by sample count* — it decreases as $N$ grows, so numbers computed with 10k samples are not comparable to numbers computed with 50k, and 50k is the convention. And because it compares a single Gaussian fit, it is insensitive to some failure modes and sensitive to image preprocessing (resizing, JPEG) in ways that make cross-paper comparison unreliable unless the exact pipeline matches.

**Why Perplexity Is Awkward for Diffusion Text:**

A diffusion language model does not factorize the sequence probability left to right, so it has no exact per-token likelihood. What you can compute is a variational bound on the negative log-likelihood — the ELBO — and exponentiate that, which gives an *upper bound* on perplexity, not perplexity itself. Comparing that number against an autoregressive model's exact perplexity is comparing a bound to a value, and it stacks the deck against the diffusion model. Say this if asked; it is a common gotcha.

**Follow-up:** *What single number should you report for a text diffusion model?* There isn't one. Report a generative quality measure (human preference or a task metric), a diversity measure such as distinct-$n$ or self-BLEU, and the number of denoising steps used — quality and steps trade off directly, so a quality number without a step count is meaningless.

> **Saying it out loud.** For images the workhorse is FID, which compares Inception feature statistics between real and generated sets — lower is better, and you have to hold the sample count fixed at fifty thousand or the numbers aren't comparable. For text it's messier, because a diffusion model doesn't give you an exact likelihood, only a bound, so quoting its perplexity next to GPT's isn't apples to apples. And whatever you report, report the number of denoising steps with it, because that's the quality-versus-speed dial.

---

### Q101: Compare diffusion models with autoregressive models (GPT) for text generation.

**Answer:**

**Generation Process:**

**Autoregressive (GPT):**
- Generate left-to-right, one token at a time
- Sequential: t₁ → t₂ → t₃ → ...

**Diffusion:**
- Generate all tokens in parallel (discrete diffusion)
- Iteratively refine all tokens together

**Advantages:**

**Autoregressive:**
- Faster single-pass generation
- Simpler implementation
- Better for long sequences
- More established for text

**Diffusion:**
- Non-autoregressive (parallel)
- Better for editing tasks
- More flexible control
- Can edit specific parts

**When to Use:**

**Autoregressive:**
- Standard text generation
- Long sequences
- When speed is important

**Diffusion:**
- Text editing/inpainting
- Controlled generation
- When need parallel generation

**Current State:**
- Autoregressive (GPT) dominates text generation
- Diffusion better for images
- Discrete diffusion promising for text

**Putting Real Numbers on "Faster":**

The comparison is not steps versus steps, it is *network evaluations* versus tokens. Generating $n$ tokens autoregressively costs $n$ forward passes, each over a growing sequence but cheap per pass thanks to the KV cache. A diffusion model generating the same $n$ tokens costs $S$ forward passes over the *full* length, where $S$ is the number of denoising steps. So diffusion wins exactly when $S < n$, and by roughly the factor $n/S$. For $n = 512$ tokens and $S = 32$ steps that is a 16x advantage in passes — real, and it is why the parallel-decoding pitch is credible. For a short 20-token answer with $S = 32$, diffusion is *slower*. The crossover is the whole story.

**The Cost Diffusion Pays:**

Within one denoising step, all positions are predicted independently given the current state. If "New" and "York" are both masked and each is individually likely, the model can commit "New" and "Delhi" in the same step, because nothing coordinates them. Autoregressive decoding never has this problem — every token is conditioned on all previous tokens. This is the same conditional-independence failure that plagued non-autoregressive machine translation, and it is why practical diffusion decoders commit only a few high-confidence tokens per step, which pushes $S$ back up toward $n$ and eats the speed advantage.

**One Real Asymmetry in Controllability:**

Constraints like "this sequence must contain these five words" or "leave characters 40 through 80 untouched" are trivial for diffusion — clamp those positions and never unmask them. For an autoregressive model the same constraint requires beam search with lookahead heuristics or a specially trained infilling objective. If the interviewer pushes on "why would anyone use text diffusion," this is the strongest honest answer.

**Follow-up:** *Can you combine them?* Yes, and this is where the field is going: block-wise or semi-autoregressive schemes decode a chunk of tokens by diffusion, condition on it, and move to the next chunk. You get intra-block parallelism with inter-block causal conditioning, and you can keep a KV cache across blocks.

> **Why the interviewer asks this.** They want a quantified trade-off — steps versus tokens — not a preference between two architectures.

> **Saying it out loud.** The headline is that autoregressive models decode one token per forward pass and diffusion decodes the whole sequence per pass but needs many passes. So diffusion is faster only when your step count is smaller than your output length — for five hundred tokens in thirty steps that's a real win, for a twenty-token answer it's a loss. The cost is that tokens within a step are predicted independently, so the model can produce locally-fine, globally-inconsistent phrases. Autoregressive still owns general text; diffusion's genuine edge is infilling and hard constraints.

---

See `40_diffusion_models/diffusion_theory.md` for complete theory!
See `40_diffusion_models/diffusion_code.py` for continuous diffusion!
See `40_diffusion_models/nlp_diffusion.py` for discrete diffusion!
See `40_diffusion_models/training_diffusion.py` for training procedures!
See `40_diffusion_models/evaluation_diffusion.py` for evaluation methods!
See `40_diffusion_models/diffusion_qa.md` for comprehensive Q&A!

---

## Perplexity and Related Concepts

### Q102: What is perplexity? How is it computed?

**Answer:**

**Perplexity:**
- Metric that measures how well a probability model predicts a sample
- Defined as exponentiated average negative log-likelihood
- Lower perplexity = better model

**Mathematical Definition:**

```
PP(W) = exp(-(1/N) * Σ log P(w_i | context))
```

Where:
- W = (w₁, w₂, ..., wₙ) is a sequence of tokens
- P(w_i | context) is probability assigned by model
- N is number of tokens

**Intuitive Understanding:**
- Perplexity = k means model is as uncertain as uniform choice among k options
- If PP = 10, model thinks there are 10 equally likely next tokens
- Lower perplexity = model is more confident = better predictions

**Computation:**

**1. Get Model Predictions:**
```python
logits = model(input_ids)  # (batch, seq_len, vocab_size)
probs = softmax(logits, dim=-1)
```

**2. Get True Token Probabilities:**
```python
true_token_probs = probs[range(batch), range(seq_len), true_tokens]
```

**3. Compute Perplexity:**
```python
nll = -log(true_token_probs).mean()  # Negative log-likelihood
perplexity = exp(nll)
```

**Connection to Cross-Entropy:**
- Cross-entropy loss = average negative log-likelihood
- Perplexity = exp(cross-entropy_loss)
- Minimizing loss = minimizing perplexity

**A Worked Example You Can Do in Your Head:**

Suppose a model assigns the true next token probabilities $0.5$, $0.25$, $0.125$, $0.125$ over a four-token sequence. The negative log-likelihoods in base 2 are $1, 2, 3, 3$ bits, averaging $2.25$ bits, so perplexity is $2^{2.25} \approx 4.76$. Read that as: on this sequence the model was, on average, about as uncertain as picking uniformly among 4.76 options. Notice the mean is over *log* probabilities, not probabilities — perplexity is the inverse geometric mean of the per-token probabilities, $\left(\prod_i P(w_i)\right)^{-1/N}$, which is why a single token with probability near zero can dominate the whole number.

**The Geometric-Mean Consequence:**

Because it is a geometric mean, one catastrophic token is not averaged away. If a 100-token sequence has 99 tokens at $P = 0.5$ and one at $P = 10^{-6}$, the average NLL in bits is $(99 \times 1 + 19.93)/100 = 1.189$, giving perplexity $2.28$ instead of $2.0$ — a 14% degradation from a single token. That sensitivity is exactly why perplexity is a good training signal (it punishes confident mistakes hard) and a fragile evaluation metric.

**The Line in the Code That Matters:**

The snippet `probs[range(batch), range(seq_len), true_tokens]` is schematic rather than runnable — with two parallel `range` objects NumPy-style advanced indexing would require them to broadcast, and in practice you use `gather` along the vocabulary axis, as the fuller code in Q105 does. The idea is right: for each position, pick out the probability the model assigned to the token that actually occurred.

**Follow-up:** *Is perplexity computed on the loss you already have?* Yes — if you train with mean cross-entropy in nats, perplexity is `exp(loss)` and nothing else needs computing. A training loss of $2.5$ is a perplexity of $12.2$.

> **Saying it out loud.** Perplexity is just the exponential of your average cross-entropy loss, so it's the same number your training loop already prints, on a different scale. The interpretation is "how many equally-likely options was the model effectively choosing between." Because it's built from a mean of log probabilities, it's really a geometric mean of the token probabilities — which means one token the model was confidently wrong about can wreck the score for a whole sequence.

---

### Q103: What does perplexity mean? How do you interpret it?

**Answer:**

**Interpretation:**

**Perplexity = k means:**
- Model is as uncertain as if it had to choose uniformly among k options
- On average, model thinks there are k equally likely next tokens

**Examples:**

**Perplexity = 1:**
- Model is perfectly certain
- Always predicts one token with probability 1
- Unrealistic for real language

**Perplexity = 10:**
- Model is as uncertain as uniform choice among 10 tokens
- Reasonable for a good language model
- Better than random (which would be vocabulary size)

**Perplexity = 100:**
- Model is very uncertain
- As confused as uniform choice among 100 tokens
- Indicates poor model or difficult task

**Perplexity = Vocabulary Size:**
- Model is as bad as random guessing
- Worst case scenario

**Typical Values:**

**For Language Models:**
- GPT-2 (small): ~30-50 on WikiText-103
- GPT-2 (large): ~15-25 on WikiText-103
- GPT-3: ~10-20 on various datasets
- State-of-the-art: < 10 on some datasets

**For Different Tasks:**
- Simple tasks: Lower perplexity (5-20)
- Complex tasks: Higher perplexity (20-100)
- Domain-specific: Varies widely

**Connection to Entropy:**
- Perplexity = 2^H (where H is cross-entropy in bits)
- Entropy measures uncertainty in bits
- Perplexity measures uncertainty in "effective vocabulary size"

**The Comparability Trap:**

The quoted numbers — GPT-2 small around 37 on WikiText-103, the larger variants in the high teens to low twenties — are only meaningful because everyone evaluates them the same way. Perplexity is *per token*, and different models use different tokenizers. A model with a large vocabulary spends fewer tokens on the same text, so each token carries more information and its per-token perplexity is higher, even if it models the text better. Comparing a byte-level model's perplexity to a 50k-BPE model's perplexity is meaningless.

The fix is to renormalize to a fixed unit. If your model needs $N_{tok}$ tokens to cover $N_{word}$ words, word-level perplexity is

$$\text{PPL}_{word} = \exp\!\left(\frac{N_{tok}}{N_{word}} \cdot \log \text{PPL}_{tok}\right)$$

Bits per byte does the same thing with bytes as the denominator, and is the standard for cross-tokenizer comparison.

**Worked Number:**

Suppose a BPE model reports token perplexity $12$ over 1.3 tokens per word. Then $\log 12 = 2.485$ nats per token, $\times 1.3 = 3.23$ nats per word, so word perplexity is $e^{3.23} \approx 25.3$ — more than double the token number. Two models can be reported at "perplexity 12" and differ by a wide margin once you put them on the same unit.

**The Ceiling Is Not the Vocabulary Size:**

The claim that a random model scores perplexity equal to vocabulary size is right only for a *uniform* random model. A unigram model that just knows token frequencies already scores far below $|V|$ — on English text, something in the hundreds against a 50k vocabulary. So "as bad as random" in practice means a few hundred, not 50,000, and a model at perplexity 500 is not at chance, it is worse than a bigram counter.

**Follow-up:** *Can perplexity go below 1?* No. Perplexity is $\exp$ of a non-negative quantity (average NLL), so its floor is exactly 1, reached only when the model assigns probability 1 to every observed token.

> **Why the interviewer asks this.** Anyone can recite the formula; the interpretation question checks whether you know that a perplexity number without its tokenizer and dataset attached is not information.

> **Saying it out loud.** Perplexity of ten means the model was about as uncertain as choosing uniformly among ten options, and lower is better with one as the floor. The thing I'd flag is that you can't compare perplexities across models with different tokenizers or different test sets — a bigger vocabulary means fewer, more informative tokens and a higher-looking per-token number. If I actually need to compare two models, I convert to bits per byte or per-word perplexity first.

---

### Q104: How is perplexity related to entropy and cross-entropy?

**Answer:**

**Connection to Entropy:**

**Entropy:**
```
H(X) = -Σ P(x) * log P(x)
```

**Perplexity:**
```
PP = 2^H(X)  (for base-2 log)
PP = exp(H(X))  (for natural log)
```

**Intuition:**
- Entropy: uncertainty in bits
- Perplexity: uncertainty in "effective vocabulary size"
- If entropy = log₂(10) ≈ 3.32 bits, perplexity = 2^3.32 ≈ 10

**Connection to Cross-Entropy:**

**Cross-Entropy:**
```
H(P, Q) = -Σ P(x) * log Q(x)
```

**For Language Models:**
```
H = -(1/N) * Σ log P(w_i | context)
```

**Perplexity:**
```
PP = exp(H)
```

**Key Insight:**
- Cross-entropy loss = average negative log-likelihood
- Perplexity = exp(cross-entropy_loss)
- Minimizing cross-entropy = minimizing perplexity
- They are equivalent objectives

**Training:**
- When training language models, we minimize cross-entropy
- This is equivalent to minimizing perplexity
- Lower loss = lower perplexity = better model

**Bits per Token:**
- BPT = log₂(PP) = H (in bits)
- Lower BPT = lower perplexity = better model
- More interpretable for some applications

**Where the Exponential Comes From:**

The connection is not a coincidence, it is a definition unwound. Entropy $H$ in bits is the average number of yes/no questions needed to identify the outcome. If you need $H$ bits, you are distinguishing among $2^H$ equally likely possibilities. Perplexity is defined as that count: $\text{PPL} = 2^{H}$ in bits, $e^{H}$ in nats. So perplexity and entropy are the same quantity in different units — "3.32 bits of uncertainty" and "effectively 10 choices" are the same sentence.

**Why It Is Cross-Entropy and Not Entropy:**

The true distribution $P$ over language is unknown, so you cannot compute $H(P)$. What you can compute is $H(P, Q) = -\mathbb{E}_{x \sim P}[\log Q(x)]$, estimated by averaging $-\log Q$ over held-out samples that are drawn from $P$. And because $H(P,Q) = H(P) + D_{KL}(P \| Q) \ge H(P)$, model perplexity is always an upper bound on the true entropy of the language, with the gap being exactly the KL divergence between the language and your model. Driving perplexity down is literally driving $D_{KL}(P \| Q)$ down, since $H(P)$ is a constant you cannot touch. That sentence is the best single answer to "why do we minimize cross-entropy."

**Unit Conversion, Concretely:**

Frameworks report loss in nats (natural log). $\text{PPL} = e^{\text{loss}}$; bits per token is $\text{loss}/\ln 2$. A loss of $2.5$ nats is $2.5/0.693 = 3.61$ bits per token and a perplexity of $12.18$. Going the other way, Shannon's classic estimate of English at roughly $1$ bit per character corresponds to a per-character perplexity of $2$.

**Follow-up:** *If two models have the same perplexity, are they the same model?* No. Perplexity is one scalar summarizing an average over a distribution; two models can agree on the mean log-likelihood while disagreeing badly on individual examples, on calibration, and on generation quality.

> **Saying it out loud.** They're the same thing in different units. Entropy counts your uncertainty in bits; perplexity exponentiates that to give an effective number of choices, so three point three bits and ten options are the same statement. In practice you never have the true distribution, so you compute cross-entropy against held-out data, and that's always at least the true entropy — the excess is exactly the KL divergence between the language and your model. So minimizing loss is literally minimizing that divergence.

---

### Q105: How do you compute perplexity for a language model? Show the code.

**Answer:**

**Step-by-Step Algorithm:**

**1. Get Model Predictions:**
```python
logits = model(input_ids)  # (batch, seq_len, vocab_size)
```

**2. Get Log Probabilities:**
```python
log_probs = F.log_softmax(logits, dim=-1)
```

**3. Get True Token Log Probabilities:**
```python
batch_size, seq_len = targets.shape
indices = targets.unsqueeze(-1)  # (batch, seq, 1)
true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
```

**4. Compute Average Negative Log-Likelihood:**
```python
if mask is not None:
    nll = -(true_token_log_probs * mask).sum() / mask.sum()
else:
    nll = -true_token_log_probs.mean()
```

**5. Compute Perplexity:**
```python
perplexity = torch.exp(nll).item()
```

**Complete Function:**

```python
def perplexity_from_logits(logits, targets, mask=None):
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get true token log probabilities
    indices = targets.unsqueeze(-1)
    true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
    
    # Average negative log-likelihood
    if mask is not None:
        nll = -(true_token_log_probs * mask).sum() / mask.sum()
    else:
        nll = -true_token_log_probs.mean()
    
    # Perplexity
    return torch.exp(nll).item()
```

**For Language Model Evaluation:**

```python
def language_model_perplexity(model, dataloader, device='cpu'):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids).logits
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # Compute perplexity
            pp = perplexity_from_logits(shift_logits, shift_labels)
            
            # Accumulate
            batch_tokens = shift_labels.numel()
            total_nll += np.log(pp) * batch_tokens
            total_tokens += batch_tokens
    
    # Average perplexity
    avg_pp = np.exp(total_nll / total_tokens)
    return avg_pp
```

**See `03_evaluation_metrics/perplexity_code.py` for complete implementation!**

**Why `gather` and Not Fancy Indexing:**

`log_probs` has shape `(batch, seq_len, vocab)` and `targets` has shape `(batch, seq_len)`. `targets.unsqueeze(-1)` makes it `(batch, seq_len, 1)`, and `gather(dim=-1, index=...)` picks, at every `(batch, position)` slot, the single vocabulary entry named by the target. The result is `(batch, seq_len, 1)`, and `squeeze(-1)` drops back to `(batch, seq_len)`. This is the vectorized version of "for each position, look up the log-probability of the token that actually occurred," and it does it without materializing anything vocabulary-sized beyond the log-softmax itself.

**Why the Shift by One:**

`shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]` implement next-token prediction. Position $i$ of the logits predicts token $i+1$, so the last logit has no target and the first label has no predictor. Forgetting this shift is the single most common perplexity bug and it produces a suspiciously low number, because the model appears to be predicting the token it was just shown.

**A Real Bug in the Aggregation Function:**

The `language_model_perplexity` loop calls `perplexity_from_logits` without passing a mask, so padding tokens are counted as real tokens. If your batches are padded, every pad position contributes its (usually very confident, because pad is easy) log-probability to the average and the reported perplexity is biased low. The `batch_tokens = shift_labels.numel()` line has the same problem: it counts padded slots. The fix is to build a mask from the label tensor (`shift_labels != pad_id`), pass it in, and accumulate `mask.sum()` rather than `numel()`. Round-tripping through `np.log(pp)` to recover the NLL is mathematically harmless but pointlessly lossy — returning the summed NLL and the token count directly is cleaner.

**Runnable Check (executed):**

```python
import torch, torch.nn.functional as F, math

torch.manual_seed(0)
V = 7
logits = torch.zeros(1, 4, V)
# force known probabilities 0.5, 0.25, 0.125, 0.125 on the target tokens
targets = torch.tensor([[0, 1, 2, 3]])
for i, p in enumerate([0.5, 0.25, 0.125, 0.125]):
    rest = (1 - p) / (V - 1)
    row = torch.full((V,), math.log(rest))
    row[targets[0, i]] = math.log(p)
    logits[0, i] = row

log_probs = F.log_softmax(logits, dim=-1)
nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
print(float(nll))                 # 1.5596 nats
print(float(torch.exp(nll)))      # 4.7568  -> matches 2**2.25 from Q97
print(2 ** 2.25)                  # 4.7568
```

**The Long-Document Subtlety:**

If a document is longer than the context window you must chunk it, and naive non-overlapping chunks penalize the model unfairly: the first token of every chunk is predicted with no context at all. The standard fix is a strided sliding window — advance by, say, 512 tokens through a 1024-token window and score only the newly revealed 512 — which roughly halves compute cost relative to a stride of 1 while giving every scored token real context. Reported perplexities move by several points depending on the stride, so it must be stated alongside the number.

**Follow-up:** *Why not just average the per-batch perplexities?* Because perplexity is exponential in the mean loss, and the mean of exponentials is not the exponential of the mean (Jensen's inequality) — averaging perplexities overestimates. Accumulate total NLL and total token count, then exponentiate once at the end.

> **Why the interviewer asks this.** Perplexity code is short enough to write on a whiteboard and has three classic bugs, so it is an efficient correctness probe.

> **Saying it out loud.** The computation is: log-softmax the logits, gather the log-probability of the token that actually occurred at each position, average those, negate, and exponentiate. Three things trip people up. You have to shift by one so position i predicts token i plus one. You have to mask out padding, or pad tokens flatter your number. And you have to accumulate total loss and total tokens across the whole dataset and exponentiate once at the very end, not average per-batch perplexities — the exponential doesn't commute with averaging.

---

### Q106: What are the limitations of perplexity? When should you use other metrics?

**Answer:**

**Limitations:**

**1. Not Always Correlates with Quality:**
- Lower perplexity doesn't always mean better text
- Can overfit to training data
- May not reflect human judgment
- Need other metrics (BLEU, ROUGE, human eval)

**2. Dataset Dependent:**
- Perplexity varies by dataset
- Can't compare across different datasets
- Need same preprocessing
- Fair comparison requires same setup

**3. Vocabulary Size Matters:**
- Larger vocabulary = higher baseline perplexity
- Need to account for vocabulary size
- Normalized perplexity helps
- Compare models with similar vocabularies

**4. Sequence Length:**
- Longer sequences = more stable estimate
- Shorter sequences = more variable
- Need sufficient data for reliable estimate

**5. Task-Specific:**
- Good perplexity doesn't guarantee good performance on downstream tasks
- May not reflect task-specific quality
- Need task-specific metrics

**When to Use Other Metrics:**

**1. Text Generation:**
- Use BLEU, ROUGE for quality
- Use diversity metrics (distinct-n)
- Use human evaluation
- Perplexity as one of many metrics

**2. Machine Translation:**
- Use BLEU as primary metric
- Use METEOR, TER
- Perplexity for model selection

**3. Summarization:**
- Use ROUGE as primary metric
- Use BLEU, METEOR
- Perplexity for training monitoring

**4. Question Answering:**
- Use EM (Exact Match), F1
- Use BLEU for generation quality
- Perplexity less relevant

**Best Practices:**
- Use perplexity for model selection during training
- Combine with task-specific metrics
- Don't rely only on perplexity
- Consider context and task requirements

**The Sharpest Version of "Doesn't Correlate with Quality":**

Perplexity measures how well a model assigns probability to *text that already exists*. Generation quality depends on what the model produces when it is sampling from itself, which is a different distribution — the model's own outputs are not drawn from the test set. These come apart concretely: greedy decoding from a low-perplexity model produces repetitive, degenerate text, while nucleus sampling from the same model produces much better text and *higher* measured perplexity on its own outputs. Holtzman et al. made exactly this point — human text has moderate, variable per-token likelihood, and text optimized to be maximally likely does not look human.

Perplexity is also blind to everything the loss does not see. An RLHF-tuned chat model typically has *worse* perplexity on raw web text than its base model, and is dramatically more useful. If you are evaluating an aligned model, perplexity is close to the wrong instrument.

**When It Is Genuinely the Right Metric:**

Pretraining loss curves, tokenizer and data-mixture ablations, scaling-law fits, quantization damage, and any comparison of two checkpoints of the same architecture on the same tokenizer and same test set. In all of these you are asking "did the model's density estimate improve," which is exactly what perplexity answers, and it has the enormous practical virtue of needing no references, no annotators, and no decoding.

**What to Reach for Instead, by Failure Mode:**

If you care about factuality, perplexity cannot help — use task accuracy or a grounded QA benchmark. If you care about instruction following, use pairwise human or model-judged preferences. If you care about repetition and degeneration, use distinct-$n$ and repetition rate. If you care about calibration, use expected calibration error, not perplexity, since a model can have good average likelihood and badly miscalibrated confidence.

**Follow-up:** *A team reports their new model has 20% lower perplexity. What do you ask?* Same tokenizer? Same test set, and was it decontaminated against the training data? Same evaluation stride and context length? Same handling of padding and end-of-document tokens? Any one of those can produce a 20% swing with no modelling improvement at all.

> **Why the interviewer asks this.** Perplexity is the metric people quote most and interrogate least, so it is an efficient probe for whether a candidate evaluates critically or just reports.

> **Saying it out loud.** Perplexity tells you how well the model assigns probability to text that already exists, which isn't the same as how good the text it generates is. The classic example is that maximizing likelihood at decode time gives you repetitive, robotic output — human text isn't maximally likely. It's also not comparable across tokenizers or test sets, and an RLHF'd chat model usually has worse perplexity than its base model while being far more useful. I'd use it for pretraining ablations and checkpoint comparisons, and reach for task metrics or human preference for anything user-facing.

---

See `03_evaluation_metrics/perplexity_detailed.md` for complete theory!
See `03_evaluation_metrics/perplexity_code.py` for complete code!
See `33_information_theory/information_theory.py` for entropy implementation!

---

## Causal Attention

### Q107: Explain causal attention. What does the code `np.tril(np.ones((seq_len, seq_len)))` do?

**Answer:**

**Causal Attention:**
- Masks future positions to enforce autoregressive property
- Each position can only attend to itself and previous positions
- Critical for GPT-style models (autoregressive generation)

**The Code:**
```python
mask = np.tril(np.ones((seq_len, seq_len)))
```

**Step-by-Step:**

**1. `np.ones((seq_len, seq_len))`:**
- Creates matrix of all 1s
- Shape: (seq_len, seq_len)
- Example for seq_len=4:
```
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

**2. `np.tril()`:**
- Takes lower triangular part
- Sets everything above diagonal to 0
- Keeps everything on and below diagonal as is
- Result:
```
[[1, 0, 0, 0],   ← Position 0: can only see itself
 [1, 1, 0, 0],   ← Position 1: can see 0, 1
 [1, 1, 1, 0],   ← Position 2: can see 0, 1, 2
 [1, 1, 1, 1]]   ← Position 3: can see all (0, 1, 2, 3)
```

**3. Application:**
- Mask applied to attention scores
- `scores[mask == 0] = -∞` (future positions)
- After softmax: Future positions get 0 attention weight
- Result: Each position only attends to past and current

**Why Lower Triangular?**
- Lower triangular = can attend to positions ≤ current (past + current)
- Upper triangular = wrong (would allow future, block past)
- This enforces causal constraint for autoregressive generation

**See `05_attention_mechanisms/causal_attention_detailed.md` for complete explanation!**

**Why It Is a Mask and Not Just "Don't Compute Those":**

You could imagine skipping the upper-triangular entries entirely, and fused kernels like FlashAttention do exactly that. But in a plain implementation the scores are produced by one dense matmul `Q @ K.T`, which computes all $n^2$ entries whether you want them or not. The mask is applied afterwards as an additive $-\infty$ so that the softmax — which normalizes across each row — assigns those entries zero weight. The mask is a correctness device, not an efficiency device; the efficiency version requires a kernel that never materializes the blocked tiles.

**The Convention Trap:**

Note that `np.tril` produces a *keep* mask: 1 means allowed. PyTorch's `torch.triu(torch.ones(n, n), diagonal=1)` produces the complementary *block* mask: 1 means forbidden. Both appear constantly and they are inverses of each other, so `masked_fill(mask == 0, -inf)` and `masked_fill(mask == 1, -inf)` are both correct code depending on which convention produced the mask. Getting this backwards gives you a model that can see only the future, which trains to a suspiciously low loss and generates nonsense.

**Runnable Version (executed):**

```python
import numpy as np

seq_len = 4
keep = np.tril(np.ones((seq_len, seq_len)))
scores = np.array([[2.3, 9.9, 9.9, 9.9],
                   [1.8, 2.1, 9.9, 9.9],
                   [1.2, 1.7, 2.0, 9.9],
                   [0.9, 1.4, 1.6, 2.2]])
masked = np.where(keep == 1, scores, -np.inf)
w = np.exp(masked - masked.max(axis=1, keepdims=True))
w = w / w.sum(axis=1, keepdims=True)
print(np.round(w, 3))
# [[1.    0.    0.    0.   ]
#  [0.426 0.574 0.    0.   ]
#  [0.205 0.338 0.457 0.   ]
#  [0.12  0.198 0.242 0.44 ]]  <- rows sum to 1, upper triangle exactly 0
print(w.sum(axis=1))  # [1. 1. 1. 1.]
```

Note that the deliberately large 9.9 scores in the upper triangle have zero influence: masking happens before the exponential, so the forbidden entries cannot leak in no matter how large they are.

**One Numerical Gotcha:**

Use a large negative finite number (or `-torch.inf` with care) rather than computing `exp` of a genuine `-inf` inside an unstable softmax. If an entire row is masked — which happens with padded sequences where a query position has no valid keys — `softmax` over all $-\infty$ produces `NaN`, and the `NaN` then propagates through the whole batch. This is a real production bug, not a theoretical one.

**Follow-up:** *Why is the mask usually a registered buffer, not a parameter?* Because it is fixed, has no gradient, and depends only on the maximum sequence length. Registering it as a buffer means it moves to the GPU with `.to(device)` and is built once rather than reallocated every forward pass.

> **Saying it out loud.** `np.tril` takes the lower triangle of an all-ones matrix, so row i has ones in columns zero through i and zeros after that. Read a row as "which positions may I look at" — position two can see zero, one, and itself, and nothing beyond. You add negative infinity wherever the mask is zero, then softmax, so the future positions get exactly zero weight and each row still sums to one. The trap is the convention: some libraries hand you the opposite mask, where one means blocked.

---

### Q108: Why do we need causal attention? What happens without it?

**Answer:**

**Why We Need It:**

**Autoregressive Constraint:**
- In autoregressive generation, tokens are generated left-to-right
- When generating token at position i, only tokens 0...i-1 exist
- Future tokens (i+1, i+2, ...) don't exist yet
- Model should only use information from past and current tokens

**What Happens Without Causal Mask:**

**During Training:**
- Model sees full sequence: [token_0, token_1, ..., token_n]
- Without mask: Each position can attend to ALL positions (including future)
- Model learns to use future tokens for prediction

**During Inference:**
- Generate one token at a time
- At step i, only have [token_0, ..., token_{i-1}]
- Future tokens don't exist
- But model was trained to use future tokens!

**Result:**
- Training and inference mismatch
- Model behavior inconsistent
- Poor generation quality

**With Causal Mask:**
- Training: Each position only sees past/current (matches inference)
- Inference: Each position only sees past/current (matches training)
- Consistent behavior → good generation

**Example:**
- Without mask: Position 1 can see position 2 (future) during training
- With mask: Position 1 cannot see position 2 (future) during training
- This matches inference where position 2 doesn't exist yet

**Concretely, What "Learning to Cheat" Looks Like:**

Without the mask, the fastest way to reduce loss at position $i$ is to attend to position $i+1$ and copy it, because position $i+1$'s embedding *is* the answer. A single attention head can learn this in a handful of steps. Training loss collapses toward zero — you will see per-token loss in the $10^{-3}$ range, far below the entropy of language, which is the tell. Then at generation time position $i+1$ does not exist, the head attends to padding or to the last real token, and the output is incoherent. The symptom pair — implausibly low training loss, garbage generation — is the classic signature of a missing causal mask, and interviewers ask it as a debugging question.

**Why Not Just Train One Position at a Time?**

You could avoid the mask by feeding the model prefix $x_{1:i}$ and training only on $x_{i+1}$, for each $i$ separately. That is correct but costs $n$ forward passes per sequence. The causal mask is what lets a single forward pass over a length-$n$ sequence produce $n$ training signals simultaneously — every position predicts its successor, all in parallel. Teacher forcing plus causal masking is the reason transformer pretraining is affordable at all; it is an $n$-fold efficiency win, not just a correctness patch.

**Where You Deliberately Don't Want It:**

BERT-style encoders use bidirectional attention on purpose, because they are not generating — a classifier or a token tagger benefits from seeing both sides, and the training objective (masked LM) supplies the information-hiding instead. Encoder-decoder models are mixed: the encoder is bidirectional, the decoder self-attention is causal, and the cross-attention from decoder to encoder is unmasked. So "always use a causal mask" is wrong; the rule is "mask exactly when the inference-time information set is a prefix."

**Follow-up:** *During incremental generation with a KV cache, do you still need the mask?* Not for the new token: at step $t$ the cache contains only positions $1..t-1$, so there is nothing in the future to mask — the constraint is enforced by what exists in the cache. You still need it during the prefill pass over the prompt, which processes all prompt positions at once.

> **Why the interviewer asks this.** Framed as a debugging question — implausibly low training loss with incoherent generation — it is one of the most common real failures in hand-written transformer code.

> **Saying it out loud.** Without the mask, every position can see the token that comes after it, and the easiest way to predict the next token is to just copy it. So training loss falls through the floor and the model learns nothing useful, and then at inference the future isn't there and it produces garbage. The mask is what makes training match inference. It's also what lets one forward pass give you a training signal at every position at once, instead of running the model separately for each prefix.

---

### Q109: How does the causal mask work mathematically?

**Answer:**

**Mathematical Formulation:**

**Standard Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Causal Attention:**
```
Attention(Q, K, V) = softmax((QK^T / √d_k) + M) V
```

Where M is the causal mask:
```
M[i, j] = {
    0   if j ≤ i  (can attend to past/current)
    -∞  if j > i  (cannot attend to future)
}
```

**Step-by-Step:**

**1. Compute Attention Scores:**
```
scores = Q @ K.T / √d_k  # Shape: (seq_len, seq_len)
```

**2. Apply Causal Mask:**
```
masked_scores = scores + M
# Where M[i, j] = -∞ if j > i (future positions)
```

**3. Softmax:**
```
attention_weights = softmax(masked_scores)
```

**What Happens:**
- Future positions: scores = -∞ → softmax(-∞) = 0
- Past/current positions: scores = original → softmax(original) = normal weights

**Result:**
- Each position gets attention weights that sum to 1
- Future positions always have 0 weight
- Past/current positions have non-zero weights

**Example for seq_len=4:**

**Mask Matrix:**
```
M = [[0,  -∞, -∞, -∞],
     [0,  0,  -∞, -∞],
     [0,  0,  0,  -∞],
     [0,  0,  0,  0]]
```

**After Adding Mask:**
```
scores = [[2.3, -∞,  -∞,  -∞],
          [1.8, 2.1, -∞,  -∞],
          [1.2, 1.7, 2.0, -∞],
          [0.9, 1.4, 1.6, 2.2]]
```

**After Softmax:**
```
weights = [[1.0, 0.0, 0.0, 0.0],   ← Position 0: 100% to itself
           [0.4, 0.6, 0.0, 0.0],   ← Position 1: 40% to 0, 60% to 1
           [0.2, 0.3, 0.5, 0.0],   ← Position 2: distributed, 0% to future
           [0.1, 0.2, 0.3, 0.4]]   ← Position 3: distributed across all
```

**Why Additive $-\infty$ Rather Than Multiplicative Zeroing:**

You could multiply the post-softmax weights by the 0/1 mask instead, but then the rows no longer sum to 1 and you would have to renormalize — and worse, the softmax denominator would still have included the future terms, so the surviving weights would be wrong before renormalization. Adding $-\infty$ *before* the softmax removes those terms from the denominator itself, so the normalization is exactly over the allowed set. Formally, for row $i$:

$$\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k \le i} \exp(s_{ik})} \quad \text{for } j \le i, \qquad \alpha_{ij} = 0 \text{ for } j > i$$

**Verifying the Worked Example (executed):**

The illustrative weights in the answer above are rounded for readability; the exact softmax of the given scores is:

```python
import numpy as np
s = np.array([[2.3, -np.inf, -np.inf, -np.inf],
              [1.8, 2.1, -np.inf, -np.inf],
              [1.2, 1.7, 2.0, -np.inf],
              [0.9, 1.4, 1.6, 2.2]])
e = np.exp(s - s.max(1, keepdims=True))
print(np.round(e / e.sum(1, keepdims=True), 3))
# [[1.    0.    0.    0.   ]
#  [0.426 0.574 0.    0.   ]
#  [0.205 0.338 0.457 0.   ]
#  [0.12  0.198 0.242 0.44 ]]
```

So row 1 is $(0.426, 0.574)$ rather than the rounded $(0.4, 0.6)$, and row 3 is $(0.120, 0.198, 0.242, 0.440)$. Every row still sums to exactly 1 and the upper triangle is exactly 0, which is the property that matters.

**The Structural Consequence:**

Row $i$ of the attention matrix has $i+1$ non-zero entries, so the number of live entries is $\sum_{i=0}^{n-1}(i+1) = n(n+1)/2$ — just over half of $n^2$. For $n = 4096$ that is 8.4M live entries versus 16.8M computed, so a naive masked implementation throws away almost exactly half its attention FLOPs. FlashAttention with a causal flag skips entire blocks above the diagonal and recovers close to that full 2x. This is a good number to have ready when someone asks what causal masking costs.

**Follow-up:** *Does the mask change the gradient?* Yes, and cleanly: since the masked entries are removed from the softmax denominator, their gradient is exactly zero, so no gradient ever flows to a future position's key or value through a masked edge. Nothing special is needed in the backward pass.

> **Saying it out loud.** You add a matrix to the raw scores where allowed positions get zero and future positions get negative infinity, then softmax. Because the negative infinities exponentiate to zero, they drop out of the denominator too, so each row normalizes over exactly the positions it's allowed to see. Doing it multiplicatively after the softmax would be wrong, because the future terms would already be in the denominator. And note the causal structure means only about half the score matrix is ever live, which is what the causal-aware fast kernels exploit.

---

See `05_attention_mechanisms/causal_attention_detailed.md` for complete theory!
See `05_attention_mechanisms/causal_attention_code.py` for visualization!

---

## Advanced Attention Mechanisms (GQA, Paged Attention)

### Q110: What is Group Query Attention (GQA)? How does it differ from Multi-Head Attention?

**Answer:**

**Group Query Attention (GQA):**
- Groups heads and shares K, V within each group
- Middle ground between MHA and MQA
- Reduces KV cache memory while maintaining quality

**Key Differences:**

**Multi-Head Attention (MHA):**
- Each head has separate Q, K, V
- KV Cache: num_heads × seq_len × (d_k + d_v)
- Parameters: 3 × num_heads × d_model²

**Group Query Attention (GQA):**
- Heads grouped, K, V shared within each group
- Q separate per head (like MHA)
- KV Cache: num_groups × seq_len × (d_k + d_v)
- Parameters: num_heads × d_model² + 2 × num_groups × d_model²

**Example: 32 heads, 8 groups**
- MHA: 32 × seq_len × (d_k + d_v) KV cache
- GQA: 8 × seq_len × (d_k + d_v) KV cache
- Reduction: 4× in KV cache memory

**Why It Works:**
- Queries need to be different (capture different aspects)
- Keys and values can be shared within groups
- Maintains most of MHA's expressiveness
- Significant memory reduction

**When to Use:**
- Production inference (recommended)
- Need efficiency but maintain quality
- Best balance between MHA and MQA

**Exact Shapes, One Layer:**

Let $d_{model} = 8192$, $n_h = 64$ query heads, $d_{head} = 128$ (so $n_h \cdot d_{head} = d_{model}$), and $n_{kv} = 8$ groups — these are Llama-2-70B's numbers. Then per layer:

- $W_Q \in \mathbb{R}^{8192 \times 8192}$, producing $Q$ of shape `(batch, 64, seq, 128)`
- $W_K, W_V \in \mathbb{R}^{8192 \times 1024}$, producing $K, V$ of shape `(batch, 8, seq, 128)`

Before the score matmul you `repeat_interleave` the KV heads by $n_h / n_{kv} = 8$, expanding `(batch, 8, seq, 128)` to `(batch, 64, seq, 128)`. Every group of 8 consecutive query heads then shares one physical key/value tensor. Crucially the expansion is a *view-level* broadcast at compute time — you never store the 64-head version in the cache, and that is where the saving comes from.

**The Memory Number That Actually Matters:**

KV cache bytes per token $= 2 \times n_{layers} \times n_{kv} \times d_{head} \times \text{bytes per element}$. For Llama-2-70B at fp16 with 80 layers:

| Scheme | $n_{kv}$ | Per token | 4096-token sequence | 32 concurrent sequences |
|---|---|---|---|---|
| MHA | 64 | 2.50 MiB | 10.0 GiB | 320 GiB |
| GQA | 8 | 320 KiB | 1.25 GiB | 40 GiB |
| MQA | 1 | 40 KiB | 0.16 GiB | 5 GiB |

The MHA row is the point: 32 concurrent 4k-token sequences would need 320 GiB of KV cache alone — more than four H100s, before the 140 GiB of weights. GQA brings that to 40 GiB and makes the deployment possible. This table is the single most useful thing to be able to reconstruct in an interview, because it turns "GQA saves memory" into "GQA is the difference between serving 4 requests and serving 32."

**Why It Doesn't Hurt Quality Much:**

The GQA paper's argument is empirical: they take a trained MHA checkpoint, mean-pool the key and value projections within each group to initialize the smaller matrices, and "uptrain" for about 5% of the original pretraining compute. The recovered model sits very close to MHA quality and far above MQA. The intuition is that queries carry the head-specific "what am I looking for," while keys and values are closer to a shared content representation, so heads can share a lookup table without collapsing into each other.

**Follow-up:** *What is GQA with $n_{kv} = n_h$, and with $n_{kv} = 1$?* Exactly MHA and exactly MQA respectively — GQA is the one-parameter family that interpolates between them, which is why it superseded both as the default.

> **Why the interviewer asks this.** GQA is in essentially every open-weights model shipped since 2023, so not knowing it signals you have not looked inside a modern config file.

> **Saying it out loud.** In normal multi-head attention every head has its own keys and values, so the KV cache scales with the number of heads. Grouped-query attention keeps all the query heads separate but has groups of them share one set of keys and values — Llama-2 70B has sixty-four query heads and eight KV groups, so the cache is eight times smaller. In real terms that's the difference between ten gigabytes and one and a quarter per four-thousand-token sequence. Quality barely moves, because the head-specific information mostly lives in the queries.

---

### Q111: What is Multi-Query Attention (MQA)? How does it reduce memory?

**Answer:**

**Multi-Query Attention (MQA):**
- Shares K and V across ALL heads
- Only Q is separate per head
- Maximum memory reduction

**Key Difference:**

**MHA:**
```
Head 1: Q_1, K_1, V_1
Head 2: Q_2, K_2, V_2
...
Head h: Q_h, K_h, V_h
```

**MQA:**
```
Head 1: Q_1, K_shared, V_shared
Head 2: Q_2, K_shared, V_shared
...
Head h: Q_h, K_shared, V_shared
```

**Memory Reduction:**

**KV Cache:**
- MHA: num_heads × seq_len × (d_k + d_v)
- MQA: 1 × seq_len × (d_k + d_v) (shared, not per head!)
- Reduction: num_heads× (e.g., 32× for 32 heads)

**Parameters:**
- MHA: 3 × num_heads × d_model²
- MQA: num_heads × d_model² + 2 × d_model²
- Reduction: From 3×num_heads to (num_heads + 2)

**Example: 32 heads, seq_len=2048, d_k=128**
- MHA KV Cache: 32 × 2048 × 256 = 16.8M values
- MQA KV Cache: 1 × 2048 × 256 = 0.5M values
- Reduction: 32× (16.8M → 0.5M)

**Why It Works:**
- Queries represent "what am I looking for?" (different per head)
- Keys represent "what information do I have?" (can be shared)
- Values represent "what is the information?" (can be shared)
- Same information, different queries → similar quality

**Trade-offs:**
- Maximum memory reduction
- Slight quality loss compared to MHA
- Still achieves good quality
- Used when maximum efficiency needed

**Checking the Arithmetic in the Example:**

$32 \times 2048 \times 256 = 16{,}777{,}216$ values, and $1 \times 2048 \times 256 = 524{,}288$ — so 16.8M and 0.52M are right, and the ratio is exactly 32, the head count. Two caveats about how these numbers are presented, worth stating so you are not caught out: this counts *one layer* and *one sequence*, and the $256$ is $d_k + d_v = 128 + 128$, i.e. keys and values together. Multiply by the layer count and the batch size for the real figure, and by 2 bytes for fp16. For a 32-layer model that 16.8M becomes $16.8\text{M} \times 32 \times 2 = 1.07$ GB per sequence under MHA and 34 MB under MQA.

**Why Memory Bandwidth, Not Capacity, Is the Real Win:**

Autoregressive decoding is memory-bandwidth-bound, not compute-bound. At each generated token you must *read the entire KV cache* to compute attention, and the arithmetic per byte read is tiny. So decoding time is roughly (bytes of cache) / (HBM bandwidth). Cutting the cache 32x cuts the bytes you stream per token by 32x, which is why MQA's original motivation (Shazeer, 2019) was decoding *speed*, not fitting in memory. Candidates who say "MQA saves memory" get half credit; the full answer is that it removes the dominant term in decoder latency.

**Where the Quality Loss Comes From:**

With one shared key/value set, all heads compute their scores against the same content representation and differ only in how they project the query. Heads lose the ability to specialize their *retrieval basis* — you get 32 different questions asked of one index rather than 32 different indexes. The measured effect is a small but consistent perplexity increase and, more visibly, degradation on long-context retrieval tasks. GQA exists precisely because the drop from 32 KV heads to 1 is not smooth: most of the quality is recovered by going back up to 8.

**Follow-up:** *What replaced MQA at the frontier?* GQA for most open-weights models, and for the very largest context windows, multi-head latent attention (MLA, introduced in DeepSeek-V2) — which compresses K and V into a shared low-rank latent that is cached instead, reaching MQA-like cache sizes with quality closer to full MHA. If you are asked "what's newer than GQA," MLA is the answer to name.

> **Why the interviewer asks this.** The good answer mentions memory *bandwidth* rather than capacity, which is the tell for someone who has profiled a decoder.

> **Saying it out loud.** Multi-query attention takes it to the limit — all the heads share a single set of keys and values, so the cache is as small as it gets, thirty-two times smaller for a thirty-two-head model. The reason that matters isn't really capacity, it's bandwidth: generating each token means reading the whole cache from memory, and decoding is bandwidth-bound, so a smaller cache is directly a faster decode. The downside is that heads can no longer specialize what they retrieve against, and you see it on long-context recall. That's why grouped-query, with something like eight groups, became the compromise everyone settled on.

---

### Q112: What is Paged Attention? How does it improve memory efficiency?

**Answer:**

**Paged Attention:**
- Memory-efficient KV cache management
- Manages cache in non-contiguous pages (blocks)
- Similar to virtual memory in operating systems
- Core innovation behind vLLM

**The Problem: Memory Fragmentation**

**Standard KV Cache:**
- Store K, V for each sequence in contiguous memory
- Variable-length sequences → memory fragmentation
- When sequence finishes, memory freed but fragmented
- Cannot reuse efficiently → waste

**Example:**
```
Sequence 1: [12 tokens, finished] → 12 tokens freed
Sequence 2: [16 tokens, still generating]
New sequence needs 20 tokens → Cannot use the 12 freed tokens (fragmented)
```

**Paged Attention Solution:**

**1. Page Structure:**
- Divide KV cache into fixed-size pages (blocks)
- Each page stores K, V for block_size tokens (e.g., 16 tokens)
- Pages can be non-contiguous in memory

**2. Memory Management:**
- Maintain pool of free pages
- Allocate pages on-demand
- Return pages to pool when sequence finishes
- Pages can be reused immediately

**3. Benefits:**
- No memory fragmentation
- Efficient memory reuse
- Can handle variable-length sequences
- Better GPU memory utilization (95%+ vs ~70%)

**Example:**
- block_size = 16 tokens
- Sequence of 25 tokens: needs 2 pages (32 tokens allocated)
- Waste: Only 7 tokens (within last page)
- Much better than standard (could waste 50%+)

**Memory Efficiency:**
- Standard: ~70% utilization (due to fragmentation)
- Paged: ~95%+ utilization
- Enables serving more sequences with same memory

**See `05_attention_mechanisms/advanced_attention_mechanisms.md` for complete details!**

**Naming the Two Kinds of Waste:**

Classical KV-cache allocators reserve a contiguous buffer sized to the *maximum* possible sequence length, because you cannot know in advance how long a generation will be. That produces three separate losses. Internal fragmentation is the unused tail of a reserved buffer — reserve 2048 slots, generate 200 tokens, waste 1848. Reservation waste is space held for tokens that will be generated later but is idle now. External fragmentation is free memory that exists but is not contiguous enough to satisfy a new request. The vLLM paper measured that existing systems wasted 60-80% of KV memory to these three combined; PagedAttention's own waste is bounded by *at most one block per sequence*, which at block size 16 and typical sequence lengths is under 4%.

**Worked Example with the Block Size:**

With `block_size = 16`, a 25-token sequence occupies $\lceil 25/16 \rceil = 2$ blocks holding 32 slots, so 7 slots are idle — 21.9% waste for this short sequence, and it shrinks as the sequence grows: a 1000-token sequence uses 63 blocks (1008 slots) and wastes 8 slots, 0.8%. The bound is always "less than one block," which is why the *average* is small even though a short sequence can look bad. Smaller blocks reduce waste but add block-table lookup overhead and hurt kernel efficiency; 16 is the usual compromise.

**The Part the Answer Above Leaves Out — Sharing:**

Because the mapping from logical token positions to physical blocks goes through a per-sequence block table, two sequences can *point at the same physical block*. That enables two things the contiguous design cannot do at all. Parallel sampling and beam search: $k$ candidates from one prompt share the prompt's blocks with a reference count, and copy-on-write only the block being appended to, so $k$ beams cost roughly one prompt's memory instead of $k$. And prefix caching: a long shared system prompt is stored once and reused across every request in the fleet. In practice the sharing wins are as large as the fragmentation wins, and mentioning them is what distinguishes a real answer from a paraphrase of the abstract.

**What It Costs:**

Attention can no longer be one contiguous `Q @ K.T`; the kernel must gather K and V through the block table, which is why PagedAttention is a custom CUDA kernel rather than a memory-allocator change. There is a small per-token indirection cost, repaid many times over by the higher batch size the saved memory allows — vLLM reported 2-4x throughput over the then-current systems at the same latency.

> **Note on currency (2026).** Paged KV cache is now standard across serving stacks rather than a vLLM-specific feature, and it is routinely combined with prefix caching, chunked prefill, continuous batching, and KV quantization. Specific throughput multipliers from the 2023 paper are baselines against 2023 systems — quote them as historical, not as current speedups.

**Follow-up:** *Does paged attention reduce the total KV cache size?* No. It stores the same number of key/value vectors; it changes *where* they live so that near-100% of the reserved pool is usable and blocks can be shared. Reducing the size per token is GQA/MQA/MLA's job, and the two are complementary — production stacks use both.

> **Why the interviewer asks this.** It is a systems question wearing a modelling costume, and it separates people who have deployed an LLM from people who have only trained one.

> **Saying it out loud.** Paged attention borrows virtual memory from operating systems. Instead of reserving one contiguous slab of KV cache per request, sized for the worst case, you chop the cache into fixed blocks of about sixteen tokens and hand out blocks on demand through a per-sequence page table. Wasted space drops to at most one partial block per sequence instead of most of a reservation. The part people forget is sharing — because it's a page table, two sequences can point at the same physical block, so beam search and a shared system prompt cost memory once instead of once per request.

---

### Q113: Compare MHA, GQA, and MQA. When should you use each?

**Answer:**

**Comparison Table:**

| Aspect | MHA | GQA | MQA |
|-------|-----|-----|-----|
| **Q Projections** | num_heads | num_heads | num_heads |
| **K Projections** | num_heads | num_groups | 1 |
| **V Projections** | num_heads | num_groups | 1 |
| **KV Cache** | num_heads × seq_len × (d_k + d_v) | num_groups × seq_len × (d_k + d_v) | seq_len × (d_k + d_v) |
| **Quality** | Best | Very Good | Good |
| **Memory** | Highest | Medium | Lowest |
| **Use Case** | Training, research | Production (recommended) | Maximum efficiency |

**Example: 32 heads, 8 groups, seq_len=2048**

**MHA:**
- KV Cache: 32 × 2048 × 256 = 16.8M values
- Quality: Best
- Use: Training, maximum quality needed

**GQA:**
- KV Cache: 8 × 2048 × 256 = 4.2M values (4× reduction)
- Quality: Very Good (minimal loss)
- Use: Production inference (recommended)

**MQA:**
- KV Cache: 1 × 2048 × 256 = 0.5M values (32× reduction)
- Quality: Good (slight loss)
- Use: Maximum efficiency needed

**When to Use:**

**MHA:**
- Training: Maximum quality
- Research: Need best performance
- When: Have resources, quality is priority

**GQA:**
- Production inference: Best balance
- Recommended default
- When: Need efficiency but maintain quality

**MQA:**
- Maximum efficiency needed
- Quality loss acceptable
- When: Resource-constrained, high throughput

**Paged Attention:**
- Can be used with any of above
- Production serving (vLLM)
- When: Need efficient memory management

**How to Read the Table:**

Every row is per layer, per sequence. The "Q Projections" row is identical across all three because none of these schemes touches the query side — that is the defining property of the family. The cache rows differ only in the multiplier, $n_h$ versus $n_{kv}$ versus 1, so the whole comparison collapses to a single number: how many distinct key/value head sets do you keep. The parameter savings are real but secondary; on a 70B model, shrinking $W_K$ and $W_V$ from $8192 \times 8192$ to $8192 \times 1024$ saves about $2 \times 80 \times (8192 \times 7168) \approx 9.4$B parameters, meaningful but not the reason anyone adopts GQA.

**Rebuilding the Example from First Principles:**

Take 32 heads, $d_k = d_v = 128$, `seq_len = 2048`, 32 layers, fp16, batch of 1:

- MHA: $32 \times 2048 \times 256 = 16.8$M values per layer $\to \times 32$ layers $\times 2$ bytes $= 1.07$ GB
- GQA-8: $8 \times 2048 \times 256 = 4.19$M per layer $\to 268$ MB
- MQA: $1 \times 2048 \times 256 = 0.52$M per layer $\to 34$ MB

On an 80 GB accelerator holding a 14 GB fp16 7B model, you have roughly 66 GB for cache: about 61 concurrent sequences under MHA, 246 under GQA-8, and 1900 under MQA. Throughput on a serving system is close to linear in concurrency until you saturate compute, so this is directly a throughput table.

**The Decision Rule in One Line:**

Use GQA with $n_{kv}$ between 4 and 8 unless you have a specific reason not to. MHA only if you are training something small where the cache never binds, or replicating a paper. MQA only when the cache is still the binding constraint after GQA — extreme context lengths or extreme batch sizes — and you have measured that the long-context recall loss is acceptable for your task.

**One Correction Worth Making to "MHA: Training":**

The table's suggestion that MHA is for training is a bit misleading. GQA is used *during pretraining* in every modern model that ships with it — Llama 2/3, Mistral, Qwen, Gemma — not bolted on afterwards. The uptraining procedure from the GQA paper is for converting existing MHA checkpoints; if you are training from scratch today you choose GQA at the start.

**Follow-up:** *Does GQA slow down training?* Essentially no. Training is compute-bound and processes the whole sequence in parallel with no cache, so the KV savings do not help, and the `repeat_interleave` broadcast adds negligible cost. GQA is close to free at training time and decisive at inference time, which is why it is a pure win.

> **Saying it out loud.** All three keep separate query heads; the only thing that changes is how many distinct key-value sets you cache. Multi-head keeps one per head, multi-query keeps exactly one, and grouped-query picks something in between, usually four or eight. For a thirty-two-head model that's a one-times, thirty-two-times, or four-times reduction in cache, and since cache size sets how many requests you can batch, it's really a throughput dial. I'd default to grouped-query with eight groups, and I'd note that paged attention is orthogonal — it manages the cache, it doesn't shrink it.

---

See `05_attention_mechanisms/advanced_attention_mechanisms.md` for complete theory!
See `05_attention_mechanisms/advanced_attention_code.py` for complete code!

---

## Mixture of Experts (MoE)

### Q114: What is Mixture of Experts? How does it work?

**Answer:**

**Mixture of Experts (MoE):**
- Architecture with multiple expert networks
- Router decides which experts to activate
- Only subset of experts process each input
- Enables models with trillions of parameters

**How It Works:**

**1. Multiple Experts:**
- 8-128 feed-forward networks
- Each expert is independent
- All experts have same architecture

**2. Router:**
- Takes input, outputs expert scores
- Computes probability distribution
- Selects top-k experts with highest scores

**3. Sparse Activation:**
- Only k experts activated per token (typically k=1 or 2)
- Most experts remain inactive
- Reduces computation significantly

**4. Weighted Combination:**
- Process through selected experts
- Weighted combination of outputs

**Efficiency:**
- Total parameters: num_experts × params_per_expert
- Active parameters: k × params_per_expert
- Example: 8 experts, k=2 → 4× reduction in computation

**What Is and Is Not Replaced:**

An important detail the summary skips: in a modern LLM the experts replace *the feed-forward block only*. Attention, layer norms and embeddings stay dense and shared. This matters because the FFN is roughly two thirds of a transformer's parameters, so replicating it 8x does not make the model 8x bigger. It also explains why MoE is applied where it is: the FFN is a per-token, position-independent function, so routing each token to a different copy is coherent, whereas routing tokens to different attention blocks would break the mixing that attention exists to do.

**The Router, Concretely:**

The router is a single linear layer $W_r \in \mathbb{R}^{d_{model} \times E}$ — for $d_{model} = 4096$ and $E = 8$ experts that is 32,768 parameters, utterly negligible. For each token's hidden state $x$:

$$g = \mathrm{softmax}(W_r^\top x) \in \mathbb{R}^{E}, \qquad \mathcal{T} = \mathrm{top}\text{-}k(g), \qquad y = \sum_{i \in \mathcal{T}} \frac{g_i}{\sum_{j \in \mathcal{T}} g_j} E_i(x)$$

Two things to notice. Routing is per *token*, not per sequence — the tokens of one sentence scatter across many experts, and the assignment changes at every layer, so a token's path through a 32-layer top-2 MoE is one of $\binom{8}{2}^{32}$ possible routes. And the gate weights $g_i$ multiply the expert outputs, which is the only path by which gradient reaches $W_r$ at all: `top-k` is not differentiable, so the router learns solely through the magnitude of the weights on the experts it *did* select.

**Worked Shapes for Mixtral-8x7B:**

$d_{model} = 4096$, $d_{ff} = 14336$, SwiGLU (three matrices: gate, up, down), 32 layers, 8 experts, top-2. Per expert per layer: $3 \times 4096 \times 14336 \approx 176$M parameters. All 8 experts, all 32 layers: $176\text{M} \times 8 \times 32 \approx 45.1$B. Add GQA attention ($\approx 1.3$B) and embeddings ($\approx 0.26$B) and you land at roughly 46.7B total. Active per token: $176\text{M} \times 2 \times 32 \approx 11.3$B of FFN plus the shared 1.3B attention $\approx 12.9$B.

**Follow-up:** *Why top-2 rather than top-1?* With $k=1$ the gate weight is always 1 after renormalization, so the router gets almost no useful gradient and training is unstable — Switch Transformer needed extra tricks to make $k=1$ work. With $k=2$ the two weights compete, giving a real gradient signal, at double the FFN compute. Most production models use $k=2$; some very sparse designs use $k=8$ out of 256 fine-grained experts.

> **Why the interviewer asks this.** Nearly every frontier model shipped since 2024 is sparse, so MoE has moved from exotic to table stakes.

> **Saying it out loud.** A mixture-of-experts layer replaces one feed-forward block with several copies, plus a tiny linear router that scores each of them for every token. You keep the top one or two, run only those, and combine their outputs weighted by the router's scores. So the model holds a lot of parameters but only touches a slice of them per token. The routing is per token and per layer, not per sentence, so words in the same sentence take completely different paths through the network.

---

### Q115: How does MoE reduce computation? Compare with dense models.

**Answer:**

**Dense Model:**
- All parameters used for every input
- Computation: O(d_model²) per token
- Example: 7B parameters, all active

**MoE Model:**
- Total: num_experts × params_per_expert
- Active: k × params_per_expert
- Computation: O(k × d_model²) per token

**Example: Mixtral-8x7B**
- 8 experts × 7B = 56B total parameters
- k=2 → 2 × 7B = 14B active per token
- Computation: Only 14B parameters (not 56B!)

**Reduction:**
- Computation: (num_experts / k)× reduction
- 8 experts, k=2 → 4× reduction
- But total parameters: 8× more

**Trade-off:**
- More parameters (memory)
- Less computation (speed)
- Best of both worlds

> **Correction to the Mixtral numbers above.** Mixtral-8x7B is **not** $8 \times 7\text{B} = 56$B total with 14B active. The name is misleading: only the feed-forward blocks are replicated, while attention, embeddings and norms are shared across experts. The published figures are **46.7B total parameters and about 12.9B active per token**. The arithmetic in the previous question reconstructs both. Getting this right matters in an interview because the "8x7 = 56" mistake is exactly what someone who has only read the model name would say.

**Where the Savings Actually Land:**

Per token, FLOPs scale with $k$ (experts used), while memory scales with $E$ (experts stored). For Mixtral: FLOPs are those of a ~12.9B dense model, memory is that of a ~46.7B dense model. So the honest framing is that MoE buys you the *quality* of a large model at the *compute* of a small one, and you pay in VRAM. On an 80 GB accelerator, Mixtral at fp16 needs 93 GB and does not fit on one card — the compute saving is real but you needed two GPUs to get it.

**Why the Compute Saving Is Less Than the Ratio Suggests:**

The naive claim is $E/k = 4\times$ compute reduction versus a dense model with the same total parameters. Three things erode it. Attention is unchanged and is a large share of FLOPs at long context, so total speedup is well under the FFN-only ratio. Sparse dispatch means each expert receives a scattered subset of tokens, so you pay a gather/scatter and get worse matmul shapes than one big dense GEMM. And at scale experts live on different devices, so every MoE layer contains two all-to-all collectives (dispatch and combine) whose cost is network-bound and does not shrink with $k$. Real end-to-end MoE speedups at matched quality are meaningful but nowhere near $E/k$.

**The Scaling-Law Framing:**

The useful way to state the trade is: at a fixed training FLOP budget, a sparse model reaches a lower loss than a dense one, because it has more parameters to store knowledge in while activating the same number per token. That is why the frontier went sparse. The cost is that inference memory, not inference compute, becomes the binding constraint — which is precisely the constraint that MoE-specific serving work (expert offloading, expert-parallel routing, expert caching) exists to attack.

**Follow-up:** *Is Mixtral as good as a dense 46.7B model?* No — roughly, a sparse model behaves like a dense model of size near the geometric mean of its total and active parameters. Mixtral benchmarks around or above a dense 13B and competitively with much larger dense models on many tasks, but it is not a 47B dense model with 13B's compute for free.

> **Saying it out loud.** The saving is in compute per token, not in memory. Mixtral holds about forty-seven billion parameters but only runs about thirteen billion of them for any given token, so you get the compute cost of a small model and the memory footprint of a large one. And notice the name is misleading — eight times seven isn't fifty-six, because only the feed-forward blocks are duplicated and everything else is shared. The other thing I'd flag is that the real speedup is well under the four-times you'd predict, because attention is unchanged and the all-to-all communication between experts isn't free.

---

### Q116: What is load balancing in MoE? Why is it important?

**Answer:**

**Load Balancing Problem:**
- Without balancing, router might always select same experts
- Some experts never used (waste)
- Others overloaded (bottleneck)
- Expert collapse: Only few experts ever used

**Solution: Load Balancing Loss**
```
L_balance = (1/num_experts) * sum(load_i)²
```

Where load_i is fraction of tokens routed to expert i.

**Goal:**
- Minimize variance of expert usage
- Distribute tokens evenly
- All experts used roughly equally

**Why Important:**
- Without: Experts 0-2 always used, 3-7 never used
- With: All experts used equally
- Better parameter utilization
- Prevents expert collapse

> **Correction to the loss formula above.** As written, $L = \frac{1}{E}\sum_i \text{load}_i^2$ cannot train the router: $\text{load}_i$ is a *count* of tokens routed to expert $i$, produced by a `top-k` operation, and counts have no gradient. The Switch Transformer formulation fixes this by pairing the count with the router's soft probability:

$$L_{aux} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ (non-differentiable, treated as a constant) and $P_i$ is the *mean router probability* assigned to expert $i$ over the batch (differentiable). The gradient flows through $P_i$, and multiplying by $f_i$ scales the penalty by how overloaded that expert already is: if expert 3 is taking 40% of the tokens, the loss pushes down $P_3$ hard. The factor $E$ makes the minimum value 1 at perfect balance regardless of expert count, and $\alpha$ is typically $0.01$.

**Worked Numbers:**

With $E = 8$ and perfect balance, $f_i = P_i = 1/8$, so $L_{aux} = 8 \times 8 \times \frac{1}{64} = 1.0$. Now suppose one expert takes half the tokens and the router agrees ($f_1 = P_1 = 0.5$) while the other seven split the rest evenly ($f_i = P_i = 1/14 \approx 0.0714$): $L_{aux} = 8 \times (0.25 + 7 \times 0.0051) = 8 \times 0.2857 = 2.29$. Balanced gives 1.0, collapsed gives 2.29, and the maximum, total collapse onto one expert, is $E = 8$. So the auxiliary loss ranges over $[1, E]$ and you can read the number directly as "how many times worse than balanced am I."

**Why Imbalance Is a Systems Problem, Not Just a Quality Problem:**

Each expert is given a fixed *capacity* — `capacity = capacity_factor × tokens_per_batch / E` — because the dispatch buffers must be statically shaped for the all-to-all collective. Tokens routed to an expert that is already full are **dropped**: they skip the FFN entirely and pass through on the residual connection only. So imbalance does not just underuse parameters; it silently deletes computation for real tokens. A capacity factor of 1.25 is common, which means you are provisioning 25% headroom and still dropping tokens whenever routing is skewed. And under expert parallelism the step time is set by the *slowest* expert's device, so a 2x-overloaded expert doubles your step time no matter how idle the other seven GPUs are.

**Two Refinements Worth Naming (current practice, 2026):**

Router z-loss, $L_z = \frac{1}{N}\sum_n (\log \sum_i e^{s_{ni}})^2$, penalizes large router logits and is what actually stabilizes large MoE training in bf16 — imbalance is not the only pathology, logit blowup is the other. And *auxiliary-loss-free* balancing, introduced with DeepSeek-V3, drops the aux loss entirely in favour of a per-expert bias added to the routing scores, nudged up or down after each step based on observed load. Because the bias affects only selection and not the gate weights used in the output, it balances load without adding a gradient term that fights the language-modelling objective — which is the standing complaint against auxiliary losses. Expect this to be the answer an up-to-date interviewer is listening for.

**Follow-up:** *What is expert collapse and why does it happen?* It is a rich-get-richer feedback loop: an expert that receives slightly more tokens early trains slightly faster, becomes slightly better, so the router scores it higher, so it receives more tokens. Without an explicit counterweight the system converges to a handful of used experts, and the rest are dead parameters occupying VRAM.

> **Why the interviewer asks this.** Load balancing is where MoE stops being an architecture diagram and becomes a distributed-systems problem, so it is a fast way to tell whether you have trained one or read about one.

> **Saying it out loud.** Left alone, routers collapse — an expert that gets a few more tokens early trains faster, gets scored higher, and takes even more, until most experts are dead weight. So you add an auxiliary loss that pushes the routing probabilities toward uniform, weighted by how overloaded each expert already is. It matters more than it sounds, because experts have a fixed capacity and tokens routed to a full expert are just dropped, and because under expert parallelism your step time is set by the busiest GPU. The newer approach, from DeepSeek-V3, skips the extra loss and just nudges a per-expert bias based on observed load.

---

See `41_mixture_of_experts/moe_theory.md` for complete theory!
See `41_mixture_of_experts/moe_code.py` for complete code!
See `41_mixture_of_experts/moe_qa.md` for comprehensive Q&A!

---

## State Space Models (SSM)

### Q117: What are State Space Models? How do they work?

**Answer:**

**State Space Models (SSMs):**
- Sequence models using hidden state
- Process sequences with linear recurrence
- O(n) complexity (vs O(n²) for transformers)
- Better for very long sequences

**How They Work:**

**1. Hidden State:**
- Maintain state h[k] that evolves over time
- State captures information from all previous inputs
- Updated at each step

**2. State Evolution:**
```
h[k+1] = A_d h[k] + B_d u[k]  # State update
y[k] = C_d h[k] + D_d u[k]    # Output
```

**3. Linear Recurrence:**
- Each step: O(1) computation
- Total: O(n) for sequence of length n
- Much faster than attention: O(n²)

**Key Insight:**
- State summarizes past information
- Don't need to attend to all previous tokens
- More efficient than attention

**Where the Discrete Equations Come From:**

The equations above are the *discretized* form. The underlying object is a continuous linear ODE borrowed from control theory:

$$h'(t) = A\,h(t) + B\,u(t), \qquad y(t) = C\,h(t) + D\,u(t)$$

To run it on a token sequence you discretize with a step size $\Delta$, usually zero-order hold, which gives $\bar{A} = \exp(\Delta A)$ and $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B$. Those bars are why the discrete recurrence uses $A_d, B_d$. The reason to care is that $\Delta$ becomes a learnable parameter with a real interpretation — it is the model's timescale, controlling how fast the state forgets. A large $\Delta$ means "pay attention to the current input"; a small $\Delta$ means "ignore this input and hold the state."

**The Two Modes — This Is the Whole Trick:**

Because the recurrence is linear and (in a classical SSM) time-invariant, it can be unrolled into a convolution. Substituting repeatedly gives $y = \bar{K} * u$ with kernel

$$\bar{K} = \left(C\bar{B},\ C\bar{A}\bar{B},\ C\bar{A}^2\bar{B},\ \dots,\ C\bar{A}^{n-1}\bar{B}\right)$$

So the same parameters give you two computational forms: a *recurrent* mode, $O(1)$ time and $O(1)$ memory per step, ideal for autoregressive generation; and a *convolutional* mode, computable by FFT in $O(n \log n)$ and fully parallel over the sequence, ideal for training. You train with the convolution and generate with the recurrence. Without that duality an SSM would be as slow to train as an RNN, and nobody would use it.

**The Fixed-Size State Is the Point and the Limitation:**

A transformer's KV cache grows with sequence length — at step $t$ it holds $t$ key/value pairs. An SSM's state is a fixed $N$-dimensional vector (typically $N = 16$ per channel) regardless of whether you have seen 100 tokens or 100,000. Constant memory during generation is the headline benefit. The flip side is that everything the model remembers must be compressed into those $N$ numbers, so exact recall of an arbitrary earlier token is impossible in a way it simply is not for attention.

**Why $A$ Cannot Be Random:**

Naively initializing $A$ randomly produces a model that fails badly — the state either explodes or forgets within a few steps. S4's contribution was to initialize $A$ with a HiPPO matrix, derived so that the state maintains an optimal polynomial-basis approximation of the *entire input history*. The structure of $A$ is doing the memory work, and this is the single most common gap in a candidate's answer: they describe the recurrence and omit that a generic linear recurrence does not work.

**Follow-up:** *How is this different from an LSTM?* An LSTM's recurrence is nonlinear (gates applied to the hidden state), so it cannot be unrolled into a convolution or a parallel scan and must be trained strictly sequentially. SSMs deliberately keep the state update *linear* in $h$, which is what buys parallel training. Nonlinearity is reintroduced between layers instead of inside the recurrence.

> **Why the interviewer asks this.** Whether you can explain the recurrent/convolutional duality is a clean test of whether you understand SSMs or have only memorized "linear instead of quadratic."

> **Saying it out loud.** A state space model keeps a fixed-size hidden state and updates it linearly at each step — new state is A times old state plus B times the input, and the output reads off through C. Because that update is linear and time-invariant, you can unroll it into a convolution and train the whole sequence in parallel with an FFT, then switch to the recurrent form for generation, where it's constant time and constant memory per token. The catch is that the memory is a fixed-size vector, so it compresses history rather than storing it, and the A matrix has to be initialized with special structure or it doesn't learn at all.

---

### Q118: What is Mamba? How does it differ from standard SSMs?

**Answer:**

**Mamba:**
- Selective State Space Model
- Makes parameters input-dependent
- More expressive than fixed SSMs
- State-of-the-art for long sequences

**Key Difference:**

**Standard SSM:**
```
h[k+1] = A h[k] + B u[k]  # Fixed A, B
y[k] = C h[k]             # Fixed C
```

**Mamba (Selective):**
```
B[k] = Linear_B(u[k])  # Input-dependent B
C[k] = Linear_C(u[k])  # Input-dependent C
h[k+1] = A h[k] + B[k] u[k]
y[k] = C[k] h[k]
```

**Why This Works:**
- Different inputs need different transitions
- B[k] controls how input affects state
- C[k] controls what to extract
- More expressive while maintaining O(n) complexity

**The Parameter the Summary Above Omits — and It Is the Important One:**

Mamba makes $B$, $C$ **and $\Delta$** input-dependent. $\Delta$ is the discretization step, and since $\bar{A} = \exp(\Delta A)$, making $\Delta$ a function of the input makes the *forget rate itself* content-dependent. Large $\Delta_k$ for a token means "this matters — reset toward it"; small $\Delta_k$ means "filler, hold the state." That is the actual selection mechanism, and it is the direct analogue of an LSTM forget gate reintroduced in a form that still admits parallel training. Naming only $B$ and $C$ is a half answer.

**What Selectivity Breaks:**

The moment $B$, $C$, $\Delta$ depend on the input, the system stops being time-invariant — $\bar{A}_k$ differs at every position — and the convolutional form from the previous question *no longer exists*. There is no fixed kernel to FFT. This is the central engineering problem Mamba had to solve, and the solution is a hardware-aware **parallel scan** (an associative scan over the sequence, $O(n)$ work and $O(\log n)$ depth), plus kernel fusion that keeps the expanded state in SRAM and recomputes it in the backward pass rather than writing $(B, L, D, N)$-shaped intermediates to HBM. The paper's speed comes as much from that memory-movement design as from the architecture.

**Why Selectivity Was Necessary:**

A time-invariant SSM applies the same filter to every token, so it cannot do content-based reasoning — it cannot decide to remember *this* name and discard *that* adjective. The paper's diagnostic tasks make this concrete: selective copying (copy tokens while ignoring randomly interspersed noise) and induction heads (see "A B ... A", predict "B") are unsolvable for S4 and solved by Mamba, because both require the recurrence to condition on content.

**Concrete Shapes:**

For $d_{model} = 2048$ with the usual expansion factor 2, the inner dimension $D = 4096$ and the state size is $N = 16$ per channel, so the materialized state is $4096 \times 16 = 65{,}536$ values per layer per sequence — constant in sequence length. Compare that with a transformer layer's KV cache at 32k context: $2 \times 8 \times 128 \times 32768 \approx 67$M values under GQA, a factor of roughly 1000. That ratio is the entire commercial argument for SSMs.

> **Note on currency (2026).** Mamba-2 recast the selective SSM as "structured state space duality," showing the scan is equivalent to a form of masked linear attention and letting it reuse matmul-shaped kernels for a large speedup; a further iteration, Mamba-3, has since continued the line. Verify which generation a paper or job description means before quoting specifics.

**Follow-up:** *If selectivity is just gating, why not use an LSTM?* Because Mamba's state update remains *linear in $h$* — the input-dependence enters through coefficients, not through a nonlinearity applied to the state. Linearity in $h$ is exactly what makes the recurrence associative, and associativity is what makes the parallel scan possible. An LSTM's gate multiplies a sigmoid of the previous state, which destroys that property.

> **Saying it out loud.** A classical state space model applies the same fixed filter to every token, so it can't decide that one word is worth remembering and another isn't. Mamba makes the input matrix, the output matrix, and — most importantly — the step size depend on the current token, so the forget rate becomes content-aware. That's basically a gate, but expressed in a way that keeps the recurrence linear. The price is that you lose the convolutional training shortcut, so they wrote a hardware-aware parallel scan that keeps everything in fast on-chip memory instead.

---

### Q119: Compare SSMs (Mamba) with Transformers. When to use each?

**Answer:**

**Complexity:**

| Aspect | Transformer | SSM (Mamba) |
|--------|-------------|-------------|
| **Time** | O(n²d) | O(nd) |
| **Space** | O(n²) | O(nd) |
| **Scaling** | Quadratic | Linear |

**When to Use:**

**Transformers:**
- Short-medium sequences (< 8K tokens)
- Need maximum quality
- Established architecture

**SSMs (Mamba):**
- Very long sequences (> 8K tokens)
- Need efficiency
- Sequences of length 100K+

**Crossover:**
- < 2K: Transformers faster
- > 8K: SSMs faster
- > 100K: SSMs much better

**Reading the Complexity Table Properly:**

The $O(n^2 d)$ for transformers is the attention term only; the feed-forward blocks are $O(n d^2)$ and dominate until $n \approx d$. For $d = 4096$, attention is not the largest cost until sequences exceed a few thousand tokens, which is why short-context transformers feel fine. At $n = 128\text{k}$ the picture inverts completely. So the honest statement is that attention's quadratic term is irrelevant at small $n$ and decisive at large $n$, and "SSMs are linear" only cashes out past that crossover.

> **Caution on the crossover numbers.** The "< 2K transformers faster, > 8K SSMs faster" thresholds in the answer above are indicative, not physical constants. They depend on hidden size, whether FlashAttention or a fused scan kernel is used, GPU memory bandwidth, batch size, and whether you are measuring training or decoding. Treat them as an order of magnitude and say so — an interviewer will respect "it depends on the kernel and the hardware, roughly a few thousand tokens" more than a confident wrong number.

**The Inference Asymmetry Is Bigger Than the Training One:**

Per generated token, a transformer must read a KV cache that grows linearly with context, so decoding cost per token *increases* as the sequence lengthens. An SSM reads a fixed-size state, so its per-token cost is flat forever. That means the gap is not a constant factor, it widens without bound: at 1M tokens of context an SSM still decodes at its 100-token speed. For streaming or very-long-document workloads this, not training FLOPs, is the reason to care.

**The Real Weakness, Stated Honestly:**

Because the state is a fixed-size compression, SSMs are measurably worse at tasks requiring *exact* retrieval from context — copying a long string verbatim, needle-in-a-haystack lookups, following a many-shot in-context pattern. Attention has, in effect, lossless random access to every previous token; an SSM has a lossy summary. This is not a training deficiency, it is an information-theoretic consequence of the architecture, and it is the honest answer to "why hasn't Mamba replaced transformers."

> **Note on where the field landed (2026 — the most time-sensitive claim in this section).** The practical answer is neither pure architecture but **hybrids**: stacks that are mostly SSM layers with a small fraction of full-attention layers interleaved, typically something like one attention layer per six or eight SSM layers. Jamba, Nemotron-H, Falcon-H1 and Zamba all follow this pattern, and the empirical finding is consistent — a few attention layers recover almost all of the retrieval ability while the SSM layers keep most of the memory and throughput advantage. If asked "SSM or transformer," the strongest current answer is "hybrid, and here is why the ratio is what it is." Specific model names and ratios in this area date quickly; check current releases before quoting them.

**Follow-up:** *Can you convert a trained transformer into an SSM?* Yes, approximately — distillation approaches initialize SSM layers from the attention weights of a pretrained transformer and fine-tune on a small fraction of the original data, which is far cheaper than pretraining a Mamba model from scratch. Quality lands below the teacher but above training the same SSM from scratch on the same budget.

> **Why the interviewer asks this.** They want to hear a trade-off with a named failure mode, not architecture cheerleading — "SSMs are linear so they're better" is the answer that fails.

> **Saying it out loud.** Transformers cost quadratic time in sequence length and their KV cache grows with context, so per-token decoding gets slower the longer you go. State space models are linear and keep a fixed-size state, so decoding cost is flat no matter how long the context. The catch is that a fixed state is a lossy summary, so they're noticeably worse at exact recall — copying a string, finding a needle in a haystack. Which is why almost nobody ships a pure one; the winning pattern right now is hybrids, mostly SSM layers with an attention layer sprinkled in every six or eight, which recovers the retrieval and keeps most of the speed.

---

See `42_state_space_models/ssm_theory.md` for complete theory!
See `42_state_space_models/ssm_code.py` for complete code!
See `42_state_space_models/ssm_qa.md` for comprehensive Q&A!

## Classical ML: Trees, Ensembles, and Dimensionality

### Q120: How does a decision tree decide where to split?

**Answer:**

A decision tree is a model that repeatedly partitions the feature space with axis-aligned cuts of the form "feature $j$ is less than threshold $t$." Training is greedy: at every node the algorithm enumerates candidate splits, scores each one by how much it reduces an *impurity* measure, and takes the best. Impurity is a number that is zero when a node contains one class only and maximal when the classes are evenly mixed.

The two standard impurity measures for classification. Let $p_k$ be the fraction of samples at a node belonging to class $k$.

$$\text{Gini}(S) = 1 - \sum_k p_k^2 \qquad\qquad H(S) = -\sum_k p_k \log_2 p_k$$

Gini is the probability that two samples drawn at random from the node have different labels. Entropy $H$ is the expected number of bits needed to encode a label drawn from the node. Both peak at even mixtures — for two classes, Gini peaks at $0.5$ and entropy at $1$ bit.

A split sends $n_L$ samples left and $n_R$ right out of $n$ total. Its score is the drop in impurity, weighted by how many samples land in each child:

$$\Delta = I(\text{parent}) - \frac{n_L}{n} I(\text{left}) - \frac{n_R}{n} I(\text{right})$$

With entropy this quantity is called **information gain**. The weighting matters — without it, a split that peels off a single pure sample would look fantastic.

**Worked arithmetic.** Ten samples, five positive and five negative, so the parent has $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 0.5$ and $H = 1.0$ bit. Compare two candidate splits.

*Split A* — left gets 4 positive and 1 negative, right gets 1 positive and 4 negative.
*Split B* — left gets 5 positive and 2 negative, right gets 0 positive and 3 negative (a pure leaf).

Running the arithmetic:

```python
def gini(counts):
    n = sum(counts)
    p = [c / n for c in counts]
    return 1 - sum(x * x for x in p)
```

| | left | right | weighted Gini | Gini gain | weighted entropy | info gain |
|---|---|---|---|---|---|---|
| Split A | 0.3200 | 0.3200 | 0.3200 | **0.1800** | 0.7219 | **0.2781** |
| Split B | 0.4082 | 0.0000 | 0.2857 | **0.2143** | 0.6042 | **0.3958** |

Verified output from the script:

```
parent gini 0.5 ent 1.0
A L [4, 1] R [1, 4] weighted gini 0.32   gini gain 0.18   weighted ent 0.7219 info gain 0.2781
B L [5, 2] R [0, 3] weighted gini 0.2857 gini gain 0.2143 weighted ent 0.6042 info gain 0.3958
```

Split B wins on both criteria, even though its left child is *less* pure than either of A's children, because it manufactures one perfectly pure node and pure nodes are terminal — no further work needed on those three samples.

Gini and entropy almost always pick the same split. Entropy is slightly more sensitive to changes near the pure end because $\log$ blows up there, so it has a mild preference for splits that carve off pure regions; Gini is cheaper because it avoids logarithms. In practice the choice is worth less than one hyperparameter tick of `max_depth`. For regression trees the impurity is variance (equivalently, sum of squared errors), and the same weighted-drop formula applies.

Two details that interviewers probe. First, the search over thresholds is done by sorting each feature and sweeping the split point, updating class counts incrementally, so scoring all $n-1$ thresholds for one feature costs $O(n)$ after an $O(n \log n)$ sort — not $O(n^2)$. Second, information gain is biased toward high-cardinality features: a customer-ID column splits every node perfectly and gains a full bit, while generalizing not at all. The classical fix is **gain ratio**, which divides information gain by the entropy of the split itself (the "split information"), penalizing splits with many small branches.

**Follow-up:** *Why not just optimize accuracy at each split?* → Misclassification rate is piecewise-linear in $p$, so it is often flat: a split can move probability mass in a genuinely useful direction and produce exactly zero change in accuracy, giving the greedy search no gradient to follow. Gini and entropy are strictly concave, so almost any purifying split registers as an improvement.

> **Why the interviewer asks this.** They want to know whether you understand trees as an optimization procedure with a concrete objective, or only as a diagram you have seen in slides.

> **Saying it out loud.** "At each node the tree tries every feature and every threshold, and scores each candidate by how much it drops impurity — Gini or entropy — weighted by how many samples go to each side. Then it takes the best one and recurses. Gini's the probability two random samples in the node disagree; entropy's the bits you need to encode the label. They basically always pick the same split, and Gini's cheaper because there's no log. The reason we don't just use accuracy is that accuracy is flat over big regions, so the greedy search gets no signal."

---

### Q121: Bagging vs boosting — what is actually different, and why does each reduce error?

**Answer:**

Both are ensembles, but they attack different halves of the bias-variance decomposition and they have opposite dependency structures.

**Bagging** (bootstrap aggregating) trains $B$ models *independently*, each on a bootstrap resample — a sample of $n$ rows drawn with replacement from the $n$ training rows — and averages their predictions. Because the models are independent given the data, training is embarrassingly parallel. Each base model is deliberately low-bias and high-variance: a fully grown, unpruned tree. Averaging then knocks the variance down.

**Boosting** trains $M$ models *sequentially*, each one fit to the errors the current ensemble is still making, and adds them up with a small step size. Each base model is deliberately high-bias — a stump or depth-3 tree — and the sequence of corrections drives bias down. Training cannot be parallelized across rounds, because round $m$ needs the predictions of rounds $1 \ldots m-1$.

**The variance decomposition — the heart of the answer.** Suppose you average $B$ predictors, each with variance $\sigma^2$, with pairwise correlation $\rho$ between any two of them. Variance of the average:

$$\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} f_b(x)\right) = \frac{1}{B^2}\left[B\sigma^2 + B(B-1)\rho\sigma^2\right] = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Read that formula carefully, because it explains the entire design of random forests. The second term vanishes as $B \to \infty$ — more trees always helps, and never hurts, which is why "number of trees" is not really a regularization knob. But the first term, $\rho\sigma^2$, does **not** depend on $B$ at all. It is a floor. Adding trees past the point where $\frac{1-\rho}{B}\sigma^2$ is small buys you nothing.

Numerically, with $\sigma^2 = 1$ and $\rho = 0.5$:

```
B=    1 Var=1.0000
B=    5 Var=0.6000
B=   10 Var=0.5500
B=   50 Var=0.5100
B= 1000 Var=0.5005
limit -> 0.5
```

An empirical simulation with 200,000 draws and $B = 50$ gives measured single-model variance $0.9947$, measured pairwise correlation $0.4993$, and measured variance of the average $0.5088$ against the predicted $0.5100$. The formula holds.

So the only way to get below the floor is to **reduce $\rho$**. That is precisely what a random forest adds on top of bagging: at every split, consider only a random subset of features (`max_features`, classically $\sqrt{d}$ for classification). This makes individual trees slightly worse — $\sigma^2$ goes up a little — in exchange for a large drop in $\rho$, and the product $\rho\sigma^2$ falls. Bagging alone leaves trees highly correlated because one dominant feature gets chosen at the root of nearly every tree.

**Why boosting reduces error, in contrast.** Boosting is doing stagewise gradient descent in function space (see Q122). Each round fits the residual, so the ensemble's *bias* shrinks monotonically on the training set. Variance is controlled indirectly, by the learning rate and by keeping the base learners weak. The consequence: boosting **can** overfit if you keep adding rounds, and the number of rounds is a genuine regularization hyperparameter that must be tuned with early stopping. This is the single sharpest practical difference to state — more trees never hurts a random forest, more rounds absolutely can hurt a boosted model.

| | Bagging / random forest | Boosting |
|---|---|---|
| Base learner | deep, low-bias, high-variance | shallow, high-bias, low-variance |
| Fitted to | bootstrap resample of data | residuals / gradients of current ensemble |
| Dependency | independent, parallel | sequential |
| Primarily reduces | variance | bias |
| More models | never hurts | can overfit — tune with early stopping |
| Noisy labels | robust | sensitive (chases the noise) |
| Tuning | forgiving | needs care (lr, depth, rounds interact) |

**Follow-up:** *Is boosting immune to variance reduction?* → No. Stochastic gradient boosting subsamples rows (`subsample`) and columns (`colsample_bytree`) per round, which decorrelates the trees and gives boosting a bagging-like variance benefit on top of its bias reduction. This is why `subsample=0.8` is a near-universal default.

> **Why the interviewer asks this.** The bias-variance framing is the standard test of whether a candidate reasons about generalization from first principles, and the $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ formula is the specific thing that separates people who have derived it from people who have memorized "bagging reduces variance."

> **Saying it out loud.** "Bagging trains models independently on bootstrap samples and averages them — that kills variance. Boosting trains them sequentially, each one fitting the previous ensemble's residuals — that kills bias. The formula I keep in my head is that averaging $B$ correlated predictors gives you rho-sigma-squared plus one-minus-rho over B, times sigma squared. The second term goes to zero with more trees, but the first term is a floor set by how correlated the trees are. That's the whole reason random forests randomize the features at each split — you're buying a lower rho. And the practical difference: more trees never hurts a forest, but more boosting rounds definitely can, so you early-stop."

---

### Q122: How does gradient boosting work, step by step?

**Answer:**

Gradient boosting is gradient descent, but the thing being descended is not a parameter vector — it is the function itself. That reframing is the whole idea and it is what the interviewer wants to hear.

Ordinary gradient descent updates parameters: $\theta \leftarrow \theta - \eta \nabla_\theta L$. Gradient boosting updates the *predictions*: it treats the vector of model outputs $F(x_1), \ldots, F(x_n)$ as the free variables and takes a step in the direction $-\partial L / \partial F$. The catch is that a step in prediction-space only tells you how to move on the $n$ training points, and you need a function defined everywhere. So you fit a regression tree to the negative gradient and use that tree as your step direction. That's it.

**The algorithm.**

1. Initialize with the constant that minimizes the loss: $F_0(x) = \arg\min_c \sum_i L(y_i, c)$. For squared error that is the mean of $y$; for log loss it is the log-odds of the base rate.
2. For $m = 1 \ldots M$:
   a. Compute the negative gradient (the "pseudo-residual") for every training point: $r_{im} = -\left[\partial L(y_i, F(x_i)) / \partial F(x_i)\right]_{F = F_{m-1}}$.
   b. Fit a regression tree $h_m$ to the pairs $(x_i, r_{im})$ — note it is always a *regression* tree, even for classification, because you are regressing on gradients.
   c. Optionally, re-solve for the optimal constant in each leaf under the true loss (a line search per leaf) rather than using the tree's own mean.
   d. Update: $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$, where $\eta$ is the learning rate, typically $0.01$ to $0.1$.
3. Output $F_M$.

For squared error $L = \tfrac{1}{2}(y - F)^2$, the negative gradient is exactly $y - F$ — the ordinary residual. That is why the textbook explanation "each tree fits the residuals" is right for regression and misleading in general. For log loss with $p = \sigma(F)$, the negative gradient is $y - p$, the residual in probability space. The gradient framing is what lets you swap in Huber loss, quantile loss, or a ranking objective without changing any other machinery.

**Worked example, executed.** Six points, $x = 1 \ldots 6$, $y = [2, 3, 5, 9, 12, 14]$, depth-1 trees (stumps), learning rate $0.5$:

```
F0 = 7.5   MSE = 20.25
round 1: residuals=[-5.5 -4.5 -2.5  1.5  4.5  6.5] split x<3.5, leaves=(-4.167, 4.167)
         -> F=[5.417 5.417 5.417 9.583 9.583 9.583]   MSE=7.2292
round 2: residuals=[-3.417 -2.417 -0.417 -0.583  2.417  4.417] split x<4.5, leaves=(-1.708, 3.417)
         -> F=[4.562 4.562 4.562 8.729 11.292 11.292]  MSE=2.8516
round 3: residuals=[-2.562 -1.562  0.438  0.271  0.708  2.708] split x<2.5, leaves=(-2.062, 1.031)
         -> F=[3.531 3.531 5.078 9.245 11.807 11.807]  MSE=1.2563
```

MSE falls $20.25 \to 7.23 \to 2.85 \to 1.26$. Notice the residuals shrink but do not vanish, because $\eta = 0.5$ deliberately takes half-steps. Setting $\eta = 1$ would fit the training data faster and generalize worse — shrinkage is regularization, and it is why lowering the learning rate always requires raising the number of rounds to compensate.

**What XGBoost adds to this skeleton.** It takes a second-order view. Write the loss to second order around the current prediction with gradients $g_i$ and Hessians $h_i$, add an explicit penalty $\gamma T + \tfrac{1}{2}\lambda\sum_j w_j^2$ for $T$ leaves with weights $w_j$, and the optimal weight for leaf $j$ falls out in closed form:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \text{gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

That gain expression is the split criterion — it replaces Gini/entropy entirely. The $\gamma$ term means a split with positive but small gain is rejected outright, which is pre-pruning built into the objective rather than bolted on.

**Follow-up:** *Learning rate 0.01 with 5000 trees, or 0.1 with 500?* → Roughly the same fit, but the slower rate usually generalizes a little better and costs 10x the training and inference time. Pick based on your latency budget; tune rounds by early stopping on a validation set at whichever rate you chose.

> **Why the interviewer asks this.** "Fits the residuals" is the memorized answer. "Gradient descent in function space, and residuals are just what the gradient happens to equal under squared error" is the understood answer, and it immediately predicts how to handle any other loss.

> **Saying it out loud.** "Gradient boosting is gradient descent where the parameters are the predictions themselves. You start with a constant, compute the negative gradient of the loss at every training point, fit a small regression tree to those gradients, and add it in with a small learning rate. For squared error the gradient is literally the residual, which is why people say it fits residuals — but for log loss it's y minus p, and the general framing is what lets you plug in any differentiable loss. XGBoost extends it to second order, so the leaf values and the split gain both come out in closed form from the gradients and Hessians."

---

### Q123: XGBoost vs LightGBM vs CatBoost — when does the choice actually matter?

**Answer:**

Start with the honest headline: on most tabular problems, all three land within noise of each other once tuned, and the differences that matter in practice are training speed, categorical handling, and small-data robustness — not accuracy. An interviewer who has shipped models will respect that framing more than a claim that one is uniformly best.

The real differences are in three mechanisms.

**Tree growth policy.** XGBoost grows **level-wise** by default: it expands every node at depth $d$ before touching depth $d+1$. This produces balanced trees and makes `max_depth` a meaningful, well-behaved regularizer. LightGBM grows **leaf-wise**: it always splits whichever leaf in the whole tree offers the largest loss reduction. For a fixed number of leaves, leaf-wise reaches lower training loss, because it spends its budget where the error is. It also overfits far more readily on small datasets, since it will happily drive a single branch very deep to isolate a handful of rows. The control knob is therefore `num_leaves` plus `min_data_in_leaf`, not `max_depth`. The classic LightGBM failure is leaving `num_leaves` at its default of 31 with only a few thousand training rows and wondering why validation loss diverges. XGBoost can be switched to leaf-wise with `grow_policy='lossguide'`.

**Split-finding.** LightGBM's speed comes from two tricks. GOSS (gradient-based one-side sampling) keeps all rows with large gradients — the ones the model is currently wrong about — and randomly subsamples the small-gradient rows, reweighting them to keep the gain estimate unbiased. EFB (exclusive feature bundling) packs mutually-exclusive sparse features, such as the columns of a one-hot encoding, into a single dense feature, cutting effective dimensionality. Combined with histogram binning of continuous features (XGBoost has this too, via `tree_method='hist'`, which is now its default), LightGBM is typically the fastest of the three on wide, large datasets.

**Categorical features.** This is the axis where the choice genuinely matters. XGBoost historically required you to encode categoricals yourself (one-hot for low cardinality, target encoding for high); it now has native support via `enable_categorical=True` with `tree_method='hist'`. LightGBM accepts a categorical feature list and partitions category sets directly. CatBoost is built around the problem: it uses **ordered target statistics**, which encode a category by the mean target of previous rows only under a random permutation, never including the current row. That last detail is the point — naive target encoding leaks the row's own label into its own feature and produces validation scores that collapse in production. CatBoost's ordering scheme is a principled fix, and it also builds combinations of categorical features automatically. If your data is a handful of numeric columns and a pile of high-cardinality categoricals — user IDs, merchant IDs, zip codes — CatBoost is the default worth trying first, and it typically needs the least tuning to get a good first number.

CatBoost's other distinctive choice is **oblivious trees**: every node at a given depth uses the same split condition, so the tree is a full binary decision table. This is a strong regularizer and makes inference extremely fast, because scoring is an index computation rather than a branchy traversal.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Growth | level-wise (default) | leaf-wise | oblivious / symmetric |
| Speed on large wide data | good | fastest | moderate |
| Categoricals | native (hist mode), or encode yourself | native, category-set splits | ordered target statistics, best in class |
| Small-data overfitting | most forgiving | least forgiving | forgiving |
| Inference latency | good | good | fastest (oblivious trees) |
| Default-out-of-the-box quality | needs tuning | needs tuning | strongest |

Practical decision rule: LightGBM when training time on a large dataset is the bottleneck; CatBoost when categorical cardinality is high or you want a strong baseline with minimal tuning; XGBoost when you want the most battle-tested, best-documented option with the widest deployment tooling and you have the budget to tune it. Do not spend a week choosing — spend it on features, which will move the metric more.

**Follow-up:** *Which hyperparameters actually matter?* → In rough order: learning rate paired with number of rounds (set by early stopping), tree complexity (`max_depth` for XGBoost/CatBoost, `num_leaves` and `min_data_in_leaf` for LightGBM), then row and column subsampling. The regularization terms `lambda`, `alpha`, and `gamma` are worth a coarse sweep and rarely the difference between a good and a bad model.

> **Why the interviewer asks this.** They are checking for shipping experience. The tell is whether you name mechanisms — leaf-wise growth, ordered target statistics — or just recite "LightGBM is faster."

> **Saying it out loud.** "Honestly, tuned, they're usually within noise of each other on accuracy, so I pick on other grounds. LightGBM grows leaf-wise and uses gradient-based sampling, so it's the fastest on big wide data — but it overfits small data unless you pull num_leaves down. CatBoost uses ordered target statistics for categoricals, which avoids the target leakage you get from naive target encoding, so it's my first pick when there's a lot of high-cardinality categorical data. XGBoost is the safest, most documented default. The real gains are in features, not in which of the three I pick."

---

### Q124: Explain PCA and derive it. Why SVD rather than eigendecomposition of the covariance matrix?

**Answer:**

Principal component analysis finds an orthogonal set of directions in feature space, ordered so that the first captures the most variance in the data, the second captures the most variance among directions orthogonal to the first, and so on. Projecting onto the top $k$ directions gives the best rank-$k$ linear approximation of the data in the least-squares sense.

**Derivation.** Let $X \in \mathbb{R}^{n \times d}$ be the data matrix with the column means already subtracted — centering is not optional, and skipping it makes the first component point at the mean rather than at the direction of variation. The sample covariance is $C = \frac{1}{n-1}X^\top X$.

We want the unit vector $w$ maximizing the variance of the projection $Xw$:

$$\max_{w} \; w^\top C w \quad \text{subject to} \quad \|w\|_2 = 1$$

Form the Lagrangian $\mathcal{L} = w^\top C w - \lambda(w^\top w - 1)$ and set the gradient to zero:

$$\frac{\partial \mathcal{L}}{\partial w} = 2Cw - 2\lambda w = 0 \;\Longrightarrow\; Cw = \lambda w$$

So the stationary points are exactly the eigenvectors of $C$, and at such a point the objective value is $w^\top C w = w^\top \lambda w = \lambda$. The maximum is therefore the eigenvector with the largest eigenvalue, and the variance it explains *is* that eigenvalue. Repeating the argument under the added constraint of orthogonality to the components already found gives the rest, in descending eigenvalue order. Because $C$ is real symmetric and positive semi-definite, the eigenvalues are real and non-negative and the eigenvectors can be chosen orthonormal — so the components form a genuine orthonormal basis.

**The SVD connection.** The singular value decomposition writes $X = U \Sigma V^\top$ with $U$ and $V$ orthonormal and $\Sigma$ diagonal with non-negative entries. Substitute:

$$X^\top X = V\Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top \;\Longrightarrow\; C = \frac{1}{n-1}X^\top X = V \left(\frac{\Sigma^2}{n-1}\right) V^\top$$

That is an eigendecomposition of $C$. So the right singular vectors $V$ *are* the principal components, and the eigenvalues are $\sigma_i^2/(n-1)$. Verified on 500 samples of 6 correlated features:

```
eig vals : [2.6746298e+01 1.4516113e+01 8.4506620e+00 2.4447220e+00 1.2687790e+00 2.6360e-03]
S^2/(n-1): [2.6746298e+01 1.4516113e+01 8.4506620e+00 2.4447220e+00 1.2687790e+00 2.6360e-03]
max |diff|: 1.24e-14
|cos| between components: [1. 1. 1. 1. 1. 1.]
```

Identical to machine precision, and each component matches up to sign (eigenvector sign is arbitrary — a fact that trips people up when comparing PCA runs).

**Why SVD is the right implementation.** Three reasons, and the first is the one that earns the point.

*Numerical conditioning.* Forming $X^\top X$ squares the condition number: $\kappa(X^\top X) = \kappa(X)^2$. Every digit of precision you had in $X$, you lose two of in $X^\top X$. On a small ill-conditioned example the measured numbers are $\kappa(X) = 1.15 \times 10^{8}$ and $\kappa(X^\top X) = 7.4 \times 10^{15}$ — a matrix that is merely awkward becomes numerically singular in double precision. SVD operates on $X$ directly and never forms the product, so it keeps the better conditioning. Small eigenvalues, which are exactly the ones that tell you the intrinsic dimensionality, are the first casualties of the squaring.

*Cost when $d \gg n$.* If you have 200 samples of 20,000 genes, $C$ is $20000 \times 20000$ — 3.2 GB in float64 and expensive to decompose — while the data matrix is tiny. SVD costs $O(\min(n,d)^2\max(n,d))$ and never materializes the big covariance.

*Truncation.* Randomized and truncated SVD compute only the top $k$ singular triplets in roughly $O(ndk)$, which is what you actually want when $k = 50$ out of $d = 20000$.

Two practical notes. Standardize (not just center) when features have different units, because PCA maximizes raw variance and a feature measured in millimetres will dominate the same feature measured in metres. And PCA is unsupervised — it optimizes variance, not label separability — so the discarded low-variance direction can be the only one carrying the signal. When the goal is class separation, LDA optimizes the right thing.

**Follow-up:** *How do you pick $k$?* → Cumulative explained variance ratio $\sum_{i\le k}\lambda_i / \sum_i \lambda_i$ against a threshold like 95%, the elbow of the scree plot, or — best when PCA feeds a supervised model — cross-validated downstream performance, treating $k$ as an ordinary hyperparameter.

> **Why the interviewer asks this.** The Lagrangian derivation shows you can do constrained optimization; the SVD-versus-covariance question separates people who have implemented PCA from people who have called `.fit()`.

> **Saying it out loud.** "PCA finds orthogonal directions of maximum variance. You center the data, maximize w-transpose-C-w subject to w having unit norm, and the Lagrangian immediately gives you C-w equals lambda-w — so the components are eigenvectors of the covariance and the variance explained is the eigenvalue. In practice you use SVD on the centered data instead of eigendecomposing the covariance, because forming X-transpose-X squares the condition number, so you lose precision exactly on the small eigenvalues you care about. It's also much cheaper when you have way more features than samples, since you never build the d-by-d covariance at all."

---

### Q125: When do tree ensembles still beat deep learning?

**Answer:**

On tabular data with heterogeneous columns, gradient-boosted trees remain the default and frequently the winner. This is not nostalgia; it follows from concrete properties of the data and the inductive biases of the two model families.

**Heterogeneous, non-smooth features.** Neural networks have a smoothness prior — they build predictions from compositions of smooth functions, and they are biased toward solutions that vary gently over the input space. Tabular targets are often genuinely non-smooth: risk jumps at a credit-score threshold, price jumps at a category boundary. Trees represent a step function natively with one split; an MLP has to spend capacity approximating that step and will round its corners. This is the central argument in the 2022 Grinsztajn, Oyallon and Varoquaux benchmark study "Why do tree-based models still outperform deep learning on tabular data?", which also isolates two other causes: neural nets are hurt much more by uninformative features, and MLPs are rotationally invariant while real tabular data is not — the columns have individual meaning, and a model that treats an arbitrary rotation of the features as equivalent is throwing away that structure. Trees are the opposite: axis-aligned by construction.

**Scale of data.** Deep learning's advantage comes from learning representations, which requires enough data to learn them. With a few thousand to a few hundred thousand rows — the size of most business datasets — boosted trees win comfortably. Neural nets need the sample counts that only images, text, and audio naturally provide.

**Mixed types and missing values.** LightGBM and XGBoost route missing values down a learned default branch, treating missingness as information, with no imputation step. Trees are invariant to any monotone transform of a feature, so no scaling, no log transforms, no outlier clipping. A neural net needs all of that pipeline, and each stage is a chance to introduce leakage or a train/serve skew bug.

**Operational reasons that matter more than people admit.** Boosted trees train in minutes on CPU, so you can iterate on features many times a day. They give you stable, cheap feature importances and SHAP values that regulators and product managers accept. Inference is sub-millisecond on CPU with no accelerator. And there is no learning-rate schedule, no warmup, no batch-size interaction, no divergence at 3 a.m.

**Where deep learning does win on tabular-adjacent problems.** When there is genuine unstructured content in a column — free-text descriptions, images, sequences of events — an embedding model beats any feature you can hand-craft. When you need multi-task or transfer learning across related targets. When the data has strong relational structure that a graph network can exploit. And in the increasingly common hybrid: use a neural encoder to embed the text and categorical fields, then feed those embeddings, plus the raw numeric columns, into a gradient-boosted model. That hybrid is usually the right architecture when you have both kinds of signal.

It is worth flagging that this is an active area — transformer-style tabular architectures (FT-Transformer, TabPFN and its successors) have narrowed the gap and TabPFN-class models are genuinely strong on very small datasets, where a single forward pass of a pretrained model beats a fitted GBM. **Time-sensitive claim:** the exact frontier of "deep learning has caught up on tabular data" moves every year; the durable part of the answer is *why* trees have the better inductive bias for heterogeneous columns, not a leaderboard position.

**Follow-up:** *Would you ever ensemble the two?* → Yes, and it usually helps a little, because their errors are decorrelated — the neural net is smooth where the tree is piecewise-constant. Blend with weights fit on a held-out set. Whether the gain justifies maintaining two training pipelines is a separate, usually negative, judgment.

> **Why the interviewer asks this.** They want to see technical judgment rather than fashion-following, and specifically whether you can articulate inductive bias as the reason rather than saying "trees just work better on tables."

> **Saying it out loud.** "On tabular data, boosted trees usually still win, and it's about inductive bias. Neural nets have a smoothness prior and they're rotationally invariant, but tabular columns have individual meaning and the targets often have hard thresholds. A tree gets a step function with one split; an MLP has to approximate it. Trees also handle missing values and mixed scales natively, train in minutes on CPU, and give you SHAP values people trust. I'd reach for deep learning when there's actual unstructured content — text or images in a column — and often the best answer is a hybrid: embed the text with a neural model, then feed those embeddings into the GBM alongside the numeric columns."

---

### Q126: What is the curse of dimensionality, concretely?

**Answer:**

The curse of dimensionality is the collection of ways that geometric intuition built in two or three dimensions becomes actively wrong in high dimensions. The abstract statement — "data becomes sparse" — is not convincing on its own. Numbers are.

**1. Volume concentrates in the shell.** Take the unit hypercube $[0,1]^d$ and ask what fraction of its volume lies within $0.1$ of some face. The interior cube $[0.1, 0.9]^d$ has volume $0.8^d$, so the shell fraction is $1 - 0.8^d$:

```
d=  1  interior=8.000e-01  shell=20.0%
d=  2  interior=6.400e-01  shell=36.0%
d=  3  interior=5.120e-01  shell=48.8%
d= 10  interior=1.074e-01  shell=89.3%
d= 50  interior=1.427e-05  shell=99.9986%
d=100  interior=2.037e-10  shell=99.99999998%
```

In 100 dimensions essentially every point is near a boundary. There is no "middle" of a high-dimensional dataset. Every prediction is an extrapolation in some coordinate.

**2. Distances stop discriminating.** Draw 1000 uniform points in $[0,1]^d$ and a query point, and look at the spread between nearest and farthest neighbour relative to the nearest:

```
d=    1  min=0.000  max=0.986  (max-min)/min = 2574.66
d=    2  min=0.017  max=1.128  (max-min)/min =   66.06
d=   10  min=0.453  max=1.942  (max-min)/min =    3.28
d=  100  min=3.343  max=4.786  (max-min)/min =    0.43
d= 1000  min=12.128 max=13.548 (max-min)/min =    0.12
```

At $d=1000$, the farthest of a thousand points is only 12% farther away than the nearest. This is the formal result of Beyer et al. (1999): under broad conditions the ratio of max to min distance converges to 1. Every method whose core operation is "find the closest thing" — $k$-NN, kernel methods with an RBF kernel, DBSCAN, cosine retrieval over raw high-dimensional features — degrades toward meaninglessness, because "closest" stops being a distinguished status.

**3. Sample requirements explode.** To cover $[0,1]^d$ with a grid of resolution $0.1$ per axis you need $10^d$ cells: 10 in one dimension, 100,000 in five, and $10^{10}$ in ten. To hold local density constant while adding a dimension, you multiply your dataset by 10. This is why non-parametric methods — which need enough neighbours within a small radius to estimate a local average — have convergence rates that degrade as $O(n^{-2/(2+d)})$ and become useless past a modest $d$.

**4. The inscribed ball vanishes.** The ball inscribed in the unit cube, touching every face, occupies:

```
d=  2 : 7.854e-01 of the cube
d=  5 : 1.645e-01
d= 10 : 2.490e-03
d= 20 : 2.461e-08
d= 50 : 1.537e-28
```

In 50 dimensions the inscribed ball is $10^{-28}$ of the cube it fits inside. All the volume is in the corners — of which there are $2^{50}$. This is why a Gaussian in high dimensions does not concentrate at its mode: its mass lives in a thin annulus at radius $\approx\sigma\sqrt{d}$, so the most likely single point is one almost no sample ever lands near.

**Why anything works at all.** The redeeming fact is the manifold hypothesis: real data of nominal dimension $d$ typically lies on or near a manifold of much lower intrinsic dimension. A $224\times224\times3$ image lives in $\mathbb{R}^{150528}$, but natural images occupy a vanishingly thin sliver of that space. The curse applies to the ambient dimension; learning algorithms succeed by discovering the intrinsic one. That is exactly what PCA, autoencoders, and the hidden layers of any deep network are doing.

**Practical consequences to state.** Prefer models with strong structural priors in high dimensions — linear models with $\ell_1$, or trees, which only ever look at one axis at a time. Distance-based methods need dimensionality reduction first. Regularization stops being optional. And be suspicious of an RBF-kernel SVM or a $k$-NN baseline on raw 1000-dimensional features; it is probably measuring noise.

**Follow-up:** *Why do embeddings work if they are 768- or 1536-dimensional?* → Because they are learned to place semantically similar items close together, so the data occupies a low-dimensional structure inside that space rather than filling it uniformly. The curse is a statement about uniformly-filled space. Cosine similarity over trained embeddings works; cosine similarity over 768 random features does not.

> **Why the interviewer asks this.** Everyone can say "data gets sparse." Having the shell fraction, the distance-ratio collapse, and the $10^d$ sampling number ready shows you have actually internalized the geometry, and it is precisely the reasoning you need to debug a nearest-neighbour system that quietly stopped working.

> **Saying it out loud.** "The number that makes it real for me: in a hundred-dimensional unit cube, 99.99999998% of the volume is within 0.1 of a face. There's no interior. And if you drop a thousand random points in a thousand dimensions, the farthest one is only about 12% farther than the nearest — so 'nearest neighbour' stops meaning anything, and everything distance-based falls over. The reason ML works at all is the manifold hypothesis: real data sits on a much lower-dimensional surface inside that huge space, and the whole job of representation learning is finding it."

---
## Evaluation and Data Discipline

### Q127: Walk me through precision, recall, F1, ROC-AUC and PR-AUC — and when each is the right choice.

**Answer:**

Everything starts from the confusion matrix. For a binary classifier at a fixed decision threshold, TP is a positive correctly called positive, FP a negative wrongly called positive, FN a positive missed, TN a negative correctly rejected.

$$\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad \text{FPR} = \frac{FP}{FP+TN}$$

**Precision** answers "of the things I flagged, what fraction were real?" — it is the quality of your alerts, and its denominator is what you predicted. **Recall** (also sensitivity, or true positive rate) answers "of the real things, what fraction did I catch?" — its denominator is the ground truth. They trade off through the threshold: lower it and recall rises while precision falls.

The choice between them is a business question, not a statistical one, and the way to answer it in an interview is to name the asymmetric cost. If a false negative means a missed cancer diagnosis and a false positive means one extra biopsy, you optimize recall. If a false positive means wrongly blocking a paying customer's transaction and a false negative means absorbing the fraud loss, you weigh precision against the actual dollar amounts. Say the costs out loud — that is the signal the interviewer is listening for.

**F1** is the harmonic mean, $F_1 = 2PR/(P+R)$. The harmonic mean, not the arithmetic mean, because it punishes imbalance: precision $1.0$ and recall $0.0$ gives arithmetic mean $0.5$ but F1 exactly $0$. Use $F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$ when you want to weight recall $\beta$ times as much as precision — $F_2$ favours recall, $F_{0.5}$ favours precision. F1's real weakness is that it hides which side you are failing on and it ignores true negatives entirely, so always show the underlying precision and recall too.

**ROC-AUC** sweeps the threshold and plots TPR against FPR. Its value has a clean probabilistic meaning: it is the probability that a randomly chosen positive is scored above a randomly chosen negative. So it measures *ranking quality*, is threshold-independent, and is invariant to prevalence — the same model scored on a 50/50 sample and a 1-in-1000 sample gets the same ROC-AUC. Random guessing is 0.5.

**PR-AUC** (average precision) sweeps the same thresholds and plots precision against recall. It is *not* prevalence-invariant: its baseline for a random model is the positive class rate. On a dataset with 0.48% positives, measured:

```
positives: 961   prevalence: 0.004805
sep=1.5: ROC-AUC=0.8635  PR-AUC=0.0578   baseline PR-AUC=0.0048
sep=2.5: ROC-AUC=0.9611  PR-AUC=0.3544   baseline PR-AUC=0.0048
```

The second model is genuinely 12x better than random on PR-AUC and looks nearly perfect on ROC-AUC. The PR number is the one that tells you what the alert queue will feel like.

Decision rule, stated compactly. Use ROC-AUC when the classes are roughly balanced and you care about overall ranking, or when you need a metric that is comparable across populations with different base rates. Use PR-AUC when positives are rare and you only care about performance on the positive class. Use precision/recall at a *specific operating point* whenever the system has a fixed capacity — "we can review 500 alerts a day, so report precision@500" is almost always the metric the business actually has.

One warning worth volunteering: accuracy on imbalanced data is worthless. With 1,000 positives among 1,001,000 rows, always predicting negative scores $0.999001$ accuracy — 99.9% — while catching nothing.

**Follow-up:** *How do you choose the threshold?* → Not by leaving it at 0.5. Pick it on a validation set by maximizing expected utility with your real cost matrix, or by pinning the constraint the business actually has — a precision floor, a recall floor, or an alert-volume cap. Then monitor it, since the right threshold drifts as prevalence changes.

> **Why the interviewer asks this.** This is the metric-literacy screen. The candidates who pass do not just define the terms — they connect the choice to a cost asymmetry and mention the operating point.

> **Saying it out loud.** "Precision is 'of what I flagged, how much was real'; recall is 'of what's real, how much did I catch.' Which one matters is a cost question — missing a cancer versus one extra biopsy pushes you to recall; blocking good customers pushes you to precision. F1's the harmonic mean so it punishes being lopsided. ROC-AUC is the probability a random positive outranks a random negative, and it's threshold-free and prevalence-invariant. PR-AUC is the one I trust when positives are rare, because its baseline is the base rate. And in production I usually report precision at whatever alert volume the review team can actually handle."

---

### Q128: Why does ROC-AUC mislead on imbalanced data?

**Answer:**

Because of the denominator in the false positive rate. $\text{FPR} = FP/(FP+TN)$, and when negatives massively outnumber positives, $TN$ is enormous, so the denominator is essentially the total negative count and is nearly constant. That makes FPR insensitive: an alarming number of false positives produces a tiny, reassuring FPR.

**Real numbers.** A million negatives, a thousand positives — a 0.1% prevalence, which is realistic for fraud, ad clicks, or rare disease. Your model flags 10,000 cases and catches 900 of the 1,000 positives:

```
TP=900  FP=9100  FN=100  TN=990900
precision = 0.0900   recall = 0.9000   FPR = 0.0091   F1 = 0.1636
```

The ROC curve sees recall $0.90$ at FPR $0.0091$ — a point deep in the upper-left corner, the picture of an excellent classifier. The precision-recall curve sees recall $0.90$ at precision $0.09$. Both describe the same model, and both are correct.

Now translate. Precision $0.09$ means that of every 100 alerts your analysts open, 91 are false alarms. Nine thousand one hundred wasted investigations to find nine hundred real cases. That is the number the operations team lives with, and ROC-AUC never showed it to you.

The formal reason: precision has $FP$ compared against $TP$, both of which are small numbers of the same order, so it stays sensitive to changes in $FP$. FPR compares $FP$ against $TN$, which is huge, so it is anaesthetized. Concretely, adding 9,100 false positives moved FPR from 0 to 0.0091 — visually indistinguishable from the axis — while it moved precision from 1.0 to 0.09, which is the entire dynamic range of that metric.

There is a second, subtler failure. ROC-AUC is prevalence-invariant, which is sometimes a feature and here is a bug. A model with ROC-AUC $0.96$ has the same ROC-AUC whether you evaluate it at 50% prevalence or 0.1%, but its precision at a fixed recall changes by orders of magnitude between those two worlds. If you validated on a rebalanced sample and deployed to the real base rate, ROC-AUC will report no problem at all while precision collapses. PR-AUC, because it moves with prevalence, would have warned you.

What to do instead. Report PR-AUC (average precision) as the headline for rare-positive problems, always alongside the baseline, which equals the prevalence — quoting "PR-AUC 0.35" without saying the baseline is 0.005 is meaningless. Report precision and recall at the actual operating point. And if you have a capacity constraint, report precision@k for the k you can actually process.

A fair caveat, since a good interviewer may push here: ROC-AUC is not *wrong*, it answers a different question. If your use case genuinely is "rank these and I will consume the whole ranking," or you need to compare a model across sites with different base rates, ROC-AUC is the appropriate summary. The mistake is using it as the sole headline number for a rare-event detector.

**Follow-up:** *Is PR-AUC comparable across datasets?* → No, and that is the price of its sensitivity. Because the baseline is the prevalence, a PR-AUC of 0.30 at 1% prevalence is a far stronger model than 0.30 at 20%. Always report the baseline, or report the lift over it.

> **Why the interviewer asks this.** It is the fastest way to find out whether someone has actually deployed a rare-event model, because everyone who has, has been burned by exactly this.

> **Saying it out loud.** "It's the FPR denominator. FPR is false positives over all negatives, and when negatives outnumber positives a thousand to one, that denominator is huge, so a mountain of false positives still looks like a tiny FPR. Concrete case: a million negatives, a thousand positives, you catch 900 with 9,100 false alarms. FPR is 0.009 — beautiful ROC curve. But precision is 0.09, so 91 out of every 100 alerts your analysts open are junk. Same model, and the PR curve is the one that told you the truth."

---

### Q129: What is calibration, and why doesn't accuracy imply it?

**Answer:**

A model is **calibrated** when its predicted probabilities match observed frequencies: among all the cases where it says 0.7, about 70% should actually be positive. Formally, $P(y=1 \mid \hat{p} = p) = p$ for all $p$.

Calibration and discrimination are orthogonal properties, and this is the crux. **Discrimination** is whether the model ranks positives above negatives — that is what accuracy, ROC-AUC, and F1 measure. **Calibration** is whether the numbers mean anything as probabilities. A model can be perfect at one and terrible at the other.

The clean demonstration: apply any strictly increasing transform to the predicted probabilities. The ranking is untouched, so every ranking metric is identical, but the numbers are now wrong. Measured on 100,000 samples where the true probability was known:

```
calibrated        : AUC=0.8342  Brier=0.1662
p^3 (same ranking): AUC=0.8342  Brier=0.2439
sqrt(p)           : AUC=0.8342  Brier=0.1990
```

ROC-AUC is identical to four decimals across all three because monotone transforms cannot change a ranking. The Brier score — mean squared error of the probabilities — degrades by 47% for the cubed version. Its reliability table:

```
bin [0.0,0.1)  n=46444  mean_pred=0.025  actual=0.232
bin [0.1,0.2)  n=12053  mean_pred=0.146  actual=0.532
bin [0.4,0.5)  n= 5809  mean_pred=0.449  actual=0.768
bin [0.9,1.0)  n= 3425  mean_pred=0.950  actual=0.984
```

The model says 0.146 and the event happens 53% of the time. It is systematically, massively underconfident — and every accuracy-style metric would tell you the model is fine.

**Why you should care.** Any time a probability is an input to a downstream decision rather than just a sort key, calibration is the thing that matters. Expected-value calculations — "block if $p \times \text{loss} > \text{friction cost}$" — are wrong if $p$ is wrong, no matter how good the ranking is. Risk aggregation across a portfolio needs probabilities that sum correctly. Thresholds set on one population transfer to another only if the numbers are meaningful. And any human consuming "83% likely" is entitled to have that mean something.

**How to measure it.** The **reliability diagram** is the primary tool: bin predictions, plot mean predicted probability against observed frequency per bin, and compare to the diagonal. **Expected Calibration Error** is the weighted average absolute gap, $\text{ECE} = \sum_b \frac{n_b}{n}\left|\text{acc}_b - \text{conf}_b\right|$; it is a useful scalar but sensitive to binning choices, so quote the diagram too. The **Brier score** and **log loss** are *proper scoring rules*, meaning they are uniquely minimized by reporting your true beliefs — they capture calibration and discrimination together, which is why "Brier got worse but AUC didn't move" is a clean calibration diagnosis.

**How to fix it.** Fit a post-hoc mapping on a held-out calibration set — never the training set, or you will just relearn the training fit. **Platt scaling** fits a one-dimensional logistic regression on the model's scores; it is parametric, works with a few hundred points, and assumes a sigmoid-shaped distortion. **Isotonic regression** fits any monotone step function; it is more flexible, needs thousands of points, and can overfit on small sets. For neural networks, **temperature scaling** — divide the logits by a single learned scalar $T$ before the softmax — is the standard, because it fixes miscalibration with exactly one parameter and provably cannot change the argmax, so accuracy is untouched.

**Who is miscalibrated, and how.** Modern deep networks are systematically *overconfident*, a finding from Guo et al. (2017); the cause is that they are trained to near-zero loss on the training set and keep pushing logits apart after the errors are gone. Naive Bayes is overconfident because its independence assumption multiplies correlated evidence as if it were independent. Boosted trees are typically overconfident at the extremes. Random forests are usually *under*confident near 0 and 1, because averaging many trees rarely produces a unanimous vote. Logistic regression trained with log loss on well-specified features is close to calibrated by construction, since log loss is a proper scoring rule and the model is directly optimizing it.

**Follow-up:** *Does class-rebalancing affect calibration?* → It destroys it. Oversampling to 50/50 shifts the model's implicit prior, so predicted probabilities come out roughly at the rebalanced rate rather than the true one. See Q132 for the measured effect and the prior-correction formula.

> **Why the interviewer asks this.** It separates people who treat model outputs as scores from people who treat them as probabilities, and the latter is what you need to build any system that makes a decision with expected value.

> **Saying it out loud.** "Calibration means when the model says 0.7, it's right about 70% of the time. It's completely separate from accuracy — if I cube every probability, the ranking is identical, so AUC doesn't move at all, but the numbers are now badly wrong. I measure it with a reliability diagram and Brier score, and fix it post-hoc on a held-out set with Platt scaling or isotonic, or temperature scaling for a neural net since that's one parameter and can't change the argmax. It matters any time the probability feeds a decision — expected-value thresholds are garbage if the probability is garbage, however good the ranking is."

---

### Q130: Name every form of data leakage you can think of, and how you would detect each.

**Answer:**

Data leakage is any situation where information unavailable at prediction time influences training. Its signature is a validation score that is too good and a production score that is much worse. It is the single most common cause of a model that works in the notebook and fails on deployment, and interviewers ask it because catching leakage is most of what separates a careful practitioner from a careless one.

**1. Target leakage — a feature that encodes the answer.** A `discount_applied_after_refund` column in a churn model, a `days_in_ICU` column in a mortality model, an `account_closed_date` in a default model. The feature exists in the historical table but is only populated *after* the outcome. *Detection:* any single feature with implausibly high importance or high univariate AUC deserves an audit; the real test is to ask, for each column, "at the moment I need this prediction, does this value exist yet?" Build a feature dictionary with an as-of timestamp per column and enforce it.

**2. Train-test contamination through preprocessing.** Fitting a scaler, imputer, PCA, target encoder, or feature selector on the full dataset before splitting. The test set's statistics have then influenced the transformation. *Detection:* the fix and the test are the same — put every transformation inside a `sklearn.Pipeline` and cross-validate the pipeline, not the model. If your score drops when you do that, you had leakage.

**3. Temporal leakage.** Random splitting of time-ordered data, so the model trains on the future and predicts the past. Also, using a feature computed over a window that extends past the prediction time — a 30-day rolling average centered on today. *Detection:* split by time and compare to the random-split score; a large gap is diagnostic. Check that every aggregation window is strictly backward-looking.

**4. Group leakage.** The same entity appearing in both train and test: the same patient with multiple visits, the same user with multiple sessions, the same document in multiple chunks. The model memorizes the entity rather than learning the pattern. *Detection:* count the intersection of entity IDs between splits — it should be zero. Use `GroupKFold`.

**5. Duplicate and near-duplicate rows.** Exact duplicates split across train and test are trivially memorized. Near-duplicates — the same article reposted, augmented copies of an image — are worse because they evade an exact-match check. *Detection:* hash-based dedup for exact, then MinHash / SimHash or embedding cosine similarity above a threshold for near-duplicates, run *across* the split boundary.

**6. Leakage through the target encoding of a categorical.** Computing the mean target per category on the full training set means each row's own label contributes to its own feature. High-cardinality categories, where a category has one or two rows, are almost pure label. *Detection:* the effect is invisible in cross-validation if the encoding was fitted outside the fold. Use out-of-fold or ordered target encoding (Q123) and compare.

**7. Label leakage through data collection.** The labels were produced by a process correlated with a feature — for example, cases were only labelled positive if a human reviewed them, and reviews were triggered by a rule that uses one of your features. The model relearns the triggering rule. *Detection:* interrogate how labels were generated, not just what they are. This is an interview question in itself and the answer is always "go ask the person who built the labelling pipeline."

**8. Leakage via row order or an index artifact.** IDs assigned sequentially by time or by outcome, so the row index itself predicts the label. Data sorted by class. *Detection:* check the correlation of the index with the target — a genuinely alarming number of public datasets fail this.

**9. Hyperparameter and selection leakage.** Tuning hyperparameters or selecting features against the test set, repeatedly. Each glance at the test set leaks a little information, and after fifty experiments the test estimate is optimistic. *Detection:* structurally impossible to detect after the fact — prevent it with a three-way split, or nested cross-validation, and a genuinely untouched holdout opened once.

**10. Leakage through external data joined by key.** Joining an enriched table that was itself built with knowledge of the outcome period — a "customer lifetime value" column computed over all of history. *Detection:* trace the provenance of every joined table to its computation date.

**The universal detection heuristics.** First, a suspiciously good score is a bug report, not a result — 0.99 AUC on a hard problem means you have leakage until proven otherwise. Second, ablate: drop the single most important feature and see whether performance collapses to plausible. Third, the as-of-time audit: for every feature, state the timestamp at which its value becomes known, and confirm it precedes the prediction time. Fourth, the ultimate test — build a temporally held-out set from a period after all your development data, and score it once.

**Follow-up:** *You find leakage after the model shipped. What now?* → Quantify first: retrain without the leaky feature and measure the honest performance, since the deployed model may still be net-positive. Then decide whether to roll back or to keep it running while the fix is built, based on that honest number against the incumbent. And write the as-of-time check into the feature store so the class of bug cannot recur.

> **Why the interviewer asks this.** Leakage is the highest-frequency real-world ML failure, and the breadth of your list is a direct proxy for how many datasets you have personally been burned by.

> **Saying it out loud.** "Leakage is anything in training that wouldn't be available at prediction time. The big families are target leakage — a column that's only populated after the outcome — preprocessing fitted before the split, temporal leakage from random-splitting time series, and group leakage where the same user or patient appears on both sides. My standing rule is that a suspiciously good score is a bug report. And the check I run on every column is: at the moment I need this prediction, does this value exist yet? If I can't answer that with a timestamp, I don't trust the feature."

---

### Q131: How do you set up cross-validation when rows are grouped, or ordered in time?

**Answer:**

The assumption behind ordinary $k$-fold cross-validation is that rows are exchangeable — independent and identically distributed, so any partition is as good as any other. Grouped data violates independence; time-ordered data violates both independence and the premise that the future is predictable from a randomly-chosen subset of the past. Applying plain $k$-fold to either produces an optimistic estimate, sometimes wildly so.

**Grouped data.** If several rows share a latent entity — multiple visits by one patient, multiple sessions by one user, multiple chunks from one document, multiple photos of one product — those rows are correlated. Random splitting puts some of an entity's rows in train and some in test, and the model can score well by memorizing the entity rather than learning the signal. Use `GroupKFold`, which guarantees no group is split across folds, or `StratifiedGroupKFold` when you also need class balance preserved.

The correct grouping key is the one that matches how the model will be used. If it will see brand-new users, group by user. If it will see new sessions from known users, random splitting within a user is actually the honest setup. Ask what the deployment population is, and let that pick the key — this is the part interviewers are testing.

Two practical wrinkles. Group sizes are usually skewed, so folds end up unbalanced in row count; check that no fold is dominated by one whale. And when there are multiple candidate grouping keys — a patient belongs to a hospital, which belongs to a region — pick the coarsest level at which you need generalization; if the model will be deployed to a new hospital, group by hospital, not patient.

**Time-ordered data.** Two rules. Never train on data that comes after your validation data. And respect the gap between when a feature is known and when the label is known.

The standard scheme is **forward-chaining** (expanding window), `TimeSeriesSplit`:

```
fold 1: train [1..100]  test [101..120]
fold 2: train [1..120]  test [121..140]
fold 3: train [1..140]  test [141..160]
```

Every test block is strictly after its training block, and the training set grows, which mirrors production where you retrain on everything you have. The **rolling window** variant fixes the training length instead of expanding it — train on `[21..120]`, then `[41..140]` — which is the right choice when the relationship drifts and old data is actively misleading. Compare the two empirically: if the rolling window wins, you have measured concept drift, which is a useful finding to report on its own.

**The gap — the detail that distinguishes a good answer.** If your label takes 30 days to materialize (did the customer churn within 30 days?), then at the moment you would have trained the model you did not yet know the labels for the last 30 days of your training window. Training right up to the test boundary is leakage. Insert a **purge** of at least the label horizon between train and test, and if features use backward-looking windows, add an **embargo** after the test block too, so training rows just after the test period cannot see into it through their own rolling windows. This is `PurgedGroupTimeSeriesSplit` in the quantitative finance literature (López de Prado), and it is exactly the right machinery whenever labels have a lag.

**Grouped and temporal at once**, which is the common real case — many users, each with events over time. Split by time globally so no fold sees the future, and if you also need generalization to unseen users, additionally hold out a set of user IDs. Do not group-split by user *within* a time-random split; that fixes one leak and leaves the other.

**Other structure to watch for.** Spatial autocorrelation needs blocked spatial CV, not random points, or nearby training pixels leak into test pixels. Nested hierarchies need the outer level as the group. And any dataset with a "session" or "batch" column recorded by the collection process usually has a batch effect worth grouping on.

**Follow-up:** *How many folds, and does that change here?* → For time series the number of folds is governed by how much data you can afford to withhold from the first training window, not by the usual bias-variance argument; you often end up with 3 to 5. Also note the folds are not exchangeable — later folds have more training data and different market conditions — so quote the per-fold scores, not just the mean, and look at the trend.

> **Why the interviewer asks this.** Nearly every real dataset has group or time structure, and defaulting to `KFold` is the most common way a candidate produces a number that is quietly meaningless.

> **Saying it out loud.** "Plain k-fold assumes rows are exchangeable, and grouped or time-ordered data isn't. For groups — same patient, same user, same document — I use GroupKFold, and I pick the grouping key by asking what the deployment population looks like: if we'll see brand-new users, group by user. For time series it's forward chaining, train on the past, test on the future, and the detail people miss is the purge — if the label takes 30 days to materialize, you have to leave a 30-day gap before the test block, otherwise you're training on labels you wouldn't have had."

---

### Q132: How do you handle class imbalance, and what does each fix cost you?

**Answer:**

First, the answer an interviewer most wants to hear: check whether you have a problem at all. Imbalance is not intrinsically harmful. What harms you is *too few positive examples in absolute terms* and *an evaluation metric or loss that ignores the minority class*. A hundred thousand positives out of ten million is a 1% rate and a perfectly learnable problem. Two hundred positives out of twenty thousand is the same 1% and a genuinely hard problem — and the difficulty is the two hundred, not the ratio. So diagnose before you treat.

The interventions, and the cost of each.

**Do nothing to the data; change the metric and the threshold.** Train normally with log loss, evaluate with PR-AUC, and move the decision threshold to the operating point your business wants. *Cost:* none. This is the correct first move and it solves most cases, because the usual complaint ("the model predicts everything as negative") is a threshold artifact at 0.5, not a training failure. It preserves calibration exactly.

**Class weights / cost-sensitive loss.** Weight the minority class up in the loss, e.g. `class_weight='balanced'` or `scale_pos_weight` in XGBoost. *Cost:* it is mathematically a reweighting of the objective, so the model no longer estimates $P(y=1\mid x)$ — it estimates a tilted version. Calibration breaks. It also amplifies label noise on the minority class by the same weight, so a mislabelled positive now costs you 100x. Gradient variance rises.

**Random oversampling of the minority.** Duplicate minority rows. *Cost:* exact duplicates give the model the opportunity to memorize them, so overfitting risk rises, and training time grows with the dataset. Calibration breaks.

**Random undersampling of the majority.** Throw away majority rows. *Cost:* you are discarding real data, which is the most expensive thing you own; variance rises because the effective sample is smaller. Its virtue is speed, and it works well inside an ensemble — train several models on different majority subsamples and average, which recovers the discarded information (EasyEnsemble / BalancedBagging).

**SMOTE and variants.** Synthesize new minority points by interpolating between a minority point and one of its $k$ nearest minority neighbours. *Cost:* the interpolation assumes the region between two minority points is minority, which is false near a class boundary and false in high dimensions (see Q126 — "between" is not a well-behaved concept there). It performs badly with categorical features (SMOTE-NC patches this crudely) and it can generate points inside the majority region, actively creating label noise. Empirically SMOTE often fails to beat plain class weighting on tabular data, and it is worth saying so rather than reciting it as a best practice.

**Get more positives.** Targeted labelling, active learning on high-uncertainty cases, or relaxing the positive definition to a related, more frequent proxy event. *Cost:* time and money — and it is usually the highest-return option by a wide margin.

**The point that ties them together: every resampling and reweighting method breaks calibration.** They all change the effective class prior the model is trained under, so the output probabilities come out near the manipulated rate rather than the true one. Measured, on data with a 1.69% true positive rate:

```
plain      : AUC=0.8446  Brier=0.015641  mean_pred=0.0172  actual=0.0170
oversampled: AUC=0.8447  Brier=0.165720  mean_pred=0.3328  actual=0.0170
corrected  : AUC=0.8447  Brier=0.015645  mean_pred=0.0174
```

Read that carefully. Oversampling to 50/50 changed ROC-AUC by $0.0001$ — it did not improve discrimination at all — while the Brier score got **ten times worse** and the mean predicted probability went from a correct 1.72% to 33%. The model now says "one in three" about events that happen one in sixty.

The correction, when you have resampled from a true prior $\pi$ to a training prior $\tau$:

$$p_{\text{corrected}} = \frac{p\,\frac{\pi}{\tau}}{p\,\frac{\pi}{\tau} + (1-p)\frac{1-\pi}{1-\tau}}$$

Applying it recovered Brier $0.015645$ against the uncorrected $0.165720$ — back to the plain model's $0.015641$. Equivalently, for a logistic model, just subtract $\log\frac{\tau/(1-\tau)}{\pi/(1-\pi)}$ from the logit. Or skip the whole detour and recalibrate on a held-out set with the *true* class distribution, which is more robust and handles boosted trees too.

**Follow-up:** *What would you actually do first on a 1%-positive fraud problem?* → Train unmodified with log loss, evaluate with PR-AUC and precision at the alert volume the review team can handle, and tune only the threshold. If positives are scarce in absolute count, spend effort on getting more labels before touching the sampler. Reach for class weights only if the loss is genuinely being swamped, and recalibrate afterward.

> **Why the interviewer asks this.** SMOTE is the cached answer and it is often the wrong one. What they want is someone who diagnoses first, knows that thresholds fix most of it, and knows that every resampling fix silently costs you probability calibration.

> **Saying it out loud.** "First I check whether imbalance is actually the problem — what hurts is having few positives in absolute terms, not the ratio. Usually the complaint is 'it predicts everything negative,' and that's just a threshold at 0.5, so I move the threshold and evaluate with PR-AUC and I'm done. If I do need more, class weights before SMOTE — SMOTE interpolates between minority points and that assumption falls apart near the boundary and in high dimensions. And the thing everyone forgets: all of these break calibration. I measured it — oversampling to 50/50 left AUC unchanged to four decimals but pushed the mean predicted probability from 1.7% to 33%. If you need real probabilities you have to correct the prior or recalibrate afterward."

---

### Q133: What makes a good feature, and what is feature engineering doing that a deep model cannot?

**Answer:**

A good feature has four properties, and it is worth naming all four because candidates usually name only the first.

*Predictive* — it carries signal about the target, conditional on the features you already have. Marginal information is what counts; a feature with high univariate correlation that is a linear combination of two existing columns adds nothing. *Available at prediction time* — with the same value it will have in production, which is the leakage question from Q130. *Stable* — its distribution and its relationship to the target do not drift out from under you, and it is not going to be silently redefined by an upstream team. *Cheap enough* — computable inside the latency budget, from data that exists at serving time.

The fourth one is where production systems actually die. A feature that requires a join against a table refreshed nightly cannot serve a real-time request. A feature computed differently in the training SQL than in the serving code is train/serve skew, and it is the reason feature stores exist.

**What feature engineering does that a deep model cannot.**

*It injects information the model does not have.* This is the big one and it is not a matter of capacity. If you compute "distance from this transaction to the customer's usual location," you have brought in geography and a notion of usual that is nowhere in the raw columns. No amount of depth extracts external knowledge from data that does not contain it. Domain features are a channel for information, not a substitute for capacity.

*It supplies the right inductive bias cheaply.* A network can in principle learn that the ratio of two columns matters — universal approximation guarantees it in the limit — but it needs enough data to discover the ratio, and division is a hard function for a ReLU stack to represent. Handing it `debt/income` directly costs one line and saves a large amount of data. Same for cyclic encodings: `hour` as an integer tells a tree that 23 and 0 are far apart; `(sin(2πh/24), cos(2πh/24))` encodes that midnight is adjacent to 11pm. That is a fact about clocks, and you know it and the model does not.

*It aggregates across rows the model never sees together.* Any model consumes one row at a time. "Number of transactions by this card in the last hour," "user's average session length over 30 days," "count of distinct merchants this week" are cross-row aggregates. Unless you build a sequence or graph model, these are structurally invisible to the architecture and are usually the strongest features in fraud, churn, and recommendation systems.

*It regularizes by reducing dimensionality with knowledge.* Bucketing a continuous feature at a known clinical or regulatory threshold builds in a step that would otherwise cost data to learn.

**Where feature engineering loses.** On unstructured data — pixels, raw audio, text — hand-crafted features (SIFT, HOG, MFCCs, n-gram counts) were comprehensively beaten by learned representations, and that is not coming back. The reason is that the useful features there are hierarchical compositions with no compact human description; a person can write down "debt over income" but cannot write down "the third-level texture detector that fires on fur." When the useful representation is inexpressible in words and you have enough data, learn it. When the useful representation is a fact you already know, type it in.

**Practical process.** Start from the domain question — ask a fraud analyst what they look at, and encode that. Then the mechanical families: ratios and differences between related quantities, time deltas since the last event, rolling aggregates at several windows, counts and distinct-counts, deviation from an entity's own baseline, and interactions between a categorical and a numeric. Validate each with out-of-fold performance and permutation importance, not with training-set gain, which is biased toward high-cardinality features. And check drift on every feature you ship, because a stable feature is worth more than a slightly stronger unstable one.

**Follow-up:** *How do you know whether a new feature is actually helping?* → Cross-validated performance with and without it, on the split scheme that matches deployment, and a permutation importance computed out-of-fold. If the gain is within the fold-to-fold standard deviation, it is not real. And check its importance is not concentrated on a handful of rows, which is the fingerprint of leakage.

> **Why the interviewer asks this.** "Deep learning does feature engineering for you" is a common half-truth. The full picture — that engineering injects external information and cross-row structure that no architecture can conjure from a single row — is the mark of someone who has shipped a tabular model.

> **Saying it out loud.** "A good feature is predictive given what you already have, available at prediction time with the same value it'll have in production, stable, and cheap enough to serve. What feature engineering does that a deep model can't is inject information that isn't in the data — a distance from the customer's usual location brings in geography that no amount of depth would find — and aggregate across rows, since the model only ever sees one row. On images and text, learned representations won and that's settled. On tabular data, the domain features are still where the wins are."

---
## Training Fundamentals

### Q134: Batch norm vs layer norm — mechanism, and why transformers use layer norm.

**Answer:**

Both normalize activations to zero mean and unit variance and then apply a learned scale and shift, $y = \gamma\hat{x} + \beta$. The entire difference is **which axis the statistics are computed over**, and every downstream consequence follows from that one choice.

For an activation tensor of shape (batch $N$, features $D$):

**Batch normalization** computes one mean and one variance *per feature, across the batch*:

$$\mu_j = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, \qquad \sigma_j^2 = \frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2, \qquad \hat{x}_{ij} = \frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}$$

**Layer normalization** computes one mean and one variance *per sample, across the features*:

$$\mu_i = \frac{1}{D}\sum_{j=1}^{D} x_{ij}, \qquad \sigma_i^2 = \frac{1}{D}\sum_{j=1}^{D}(x_{ij}-\mu_i)^2$$

On a 4x3 example with wildly different feature scales, the difference is visible immediately:

```
x =
 [[ 1.126  4.736 10.320]
  [ 1.105  3.929 10.181]
  [ 2.304  6.894  9.648]
  [-0.265  3.753 10.021]]

BatchNorm  -> column means [0, 0, 0]        column stds [1, 1, 1]
LayerNorm  -> row means    [0, 0, 0, 0]     row stds    [1, 1, 1, 1]
```

BatchNorm makes each *column* standard; LayerNorm makes each *row* standard.

**The consequences.** BatchNorm's statistics depend on the other examples in the batch, and that single fact creates all of its problems.

*It behaves differently at train and test time.* During training it uses the batch statistics; at inference there is no batch, so it uses running averages accumulated during training. Train and inference are therefore computing different functions, and a mismatch between the running estimates and the deployment distribution is a classic silent-degradation bug. LayerNorm is identical at train and test — no running statistics, no mode switch.

*It degrades with small batches.* With batch size 2 the mean and variance estimates are extremely noisy, and normalizing by a noisy variance injects noise into every activation. Anything memory-hungry enough to require batch size 1 or 2 — large models, high-resolution segmentation, video — is where BatchNorm falls apart, which is why GroupNorm exists.

*It couples examples in a batch.* Examples influence each other's predictions, which breaks the independence assumption that some algorithms need and interferes with contrastive learning, reinforcement learning, and any setting where you care about a per-example output being a function of that example alone.

*It is awkward under distributed training.* Correct batch statistics across data-parallel workers require a synchronizing all-reduce every BatchNorm layer (SyncBN), which is a communication cost per layer.

**Why transformers use layer norm — the sequence-length argument.** This is the specific reason and the one to lead with. A transformer batch has shape (batch, sequence length, model dimension), and sequences have *different lengths*, so they are padded. BatchNorm would compute per-feature statistics over the batch-and-time axes, which means the statistics depend on how much padding is in the batch and on how long the other sequences happen to be. Two identical sentences batched with different neighbours would normalize differently. That is unacceptable.

LayerNorm normalizes over the model dimension only, so each token's normalization depends on that token's own $d_{\text{model}}$ activations and nothing else. It is exactly invariant to batch composition, sequence length, and padding. It also works identically during autoregressive generation, where the effective batch is one token at a time and BatchNorm would be meaningless.

Additional reasons that reinforce the choice: transformers are trained with very large or very small effective batch sizes depending on the hardware and gradient accumulation, and LayerNorm is indifferent; and the activations at a given position vary enormously in scale, which per-token normalization handles naturally.

**What normalization actually does.** The original "internal covariate shift" explanation has been substantially undermined — Santurkar et al. (2018) showed that injecting noise *after* BatchNorm, deliberately restoring covariate shift, still leaves the benefit intact. The better-supported account is that normalization smooths the loss landscape, bounding the gradient magnitudes and making the effective Lipschitz constant smaller, which permits larger learning rates and makes optimization less sensitive to initialization. Worth knowing, because a good interviewer may specifically probe whether you still believe the original story.

**Pre-norm vs post-norm**, which is where this question usually goes next. The original transformer put LayerNorm *after* the residual add: $x + \text{Sublayer}(x)$, then normalize. Modern models normalize *before* the sublayer: $x + \text{Sublayer}(\text{LN}(x))$. Pre-norm leaves a clean, unnormalized identity path from input to output, so gradients flow to early layers without passing through any normalization, and deep models train stably without a warmup schedule. Post-norm sometimes reaches marginally better final quality but is notoriously fragile past a few dozen layers. Essentially every large model since roughly GPT-2 is pre-norm, and many now use **RMSNorm**, which drops the mean-subtraction and rescales by the root mean square only — $x/\sqrt{\frac{1}{D}\sum x_j^2} \cdot \gamma$ — because the re-centering turns out to contribute little and removing it saves compute.

**Follow-up:** *When would you still use BatchNorm?* → Convolutional vision models with reasonable batch sizes, where it remains excellent and where its regularizing noise is a genuine benefit. Its statistics also fold into the preceding convolution's weights at inference, making it free at serving time — a real advantage LayerNorm does not have.

> **Why the interviewer asks this.** It is the cleanest test of whether you understand a mechanism or have memorized a table. The follow-up "why not BatchNorm in a transformer" has one right answer, and it is about variable sequence length.

> **Saying it out loud.** "Same operation, different axis. BatchNorm normalizes each feature across the batch; LayerNorm normalizes each sample across its features. Everything else follows from that — BatchNorm's statistics depend on the other examples, so it needs running averages at inference, it breaks with small batches, and it needs syncing across GPUs. Transformers use LayerNorm because sequences have different lengths and get padded, so batch statistics would depend on what else happened to be in the batch and how much padding there was. LayerNorm only looks at one token's own hidden vector, so it's invariant to all of that, and it works the same during generation when you're doing one token at a time."

---

### Q135: What causes vanishing and exploding gradients, and what actually fixes each?

**Answer:**

They come from the same mechanism — repeated multiplication during backpropagation — but they have genuinely different fixes, and conflating them is the mistake this question is designed to catch.

**The mechanism.** Backprop through $L$ layers multiplies Jacobians:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{\ell=1}^{L} \frac{\partial h_\ell}{\partial h_{\ell-1}}, \qquad \frac{\partial h_\ell}{\partial h_{\ell-1}} = W_\ell^\top \operatorname{diag}(\phi'(z_\ell))$$

If the typical singular value of each factor is $s$, the gradient magnitude scales as $s^L$. Anything other than $s \approx 1$ is an exponential in depth. Measured, propagating a gradient back through 50 random layers of width 100 with weights scaled by a gain factor:

```
gain=0.5 : ||grad|| after 10 layers=7.487e-04, 30=7.887e-10, 50=5.402e-16
gain=1.0 : ||grad|| after 10 layers=1.023e+00, 30=8.485e-01, 50=1.207e+00
gain=2.0 : ||grad|| after 10 layers=7.621e+02, 30=5.530e+08, 50=3.822e+14
```

A factor-of-two error in the initialization scale is the difference between a gradient of $10^{-16}$ and one of $10^{14}$. Nothing else in training is this sensitive.

**Vanishing gradients — causes.** Saturating activations are the historical cause: $\sigma'(z) = \sigma(z)(1-\sigma(z))$ has a maximum of exactly $0.25$ at $z=0$, so even in the best case ten sigmoid layers multiply the gradient by at most $0.25^{10} = 9.5\times 10^{-7}$, and thirty layers by $8.7\times10^{-19}$. That is not an edge case, that is the *best* case. Tanh is better ($\tanh'(0)=1$) but still saturates. The second cause is initialization with weights too small. The third, in RNNs, is repeated multiplication by the same recurrent matrix over hundreds of timesteps, where the effective depth is the sequence length.

**Vanishing gradients — fixes.** These are structural.

*Non-saturating activations.* ReLU has derivative exactly 1 on the positive side, so it does not attenuate. This was the single biggest unlock. GELU and SiLU are the modern smooth variants.

*Residual connections.* $h_\ell = h_{\ell-1} + F(h_{\ell-1})$ makes the Jacobian $I + \partial F/\partial h$, so the identity term guarantees a path along which the gradient reaches the input undiminished regardless of depth. This is why 100-layer networks became trainable, and it is the most important item on the list.

*Variance-preserving initialization.* He initialization, $\mathcal{N}(0, 2/n_{\text{in}})$, is derived to keep activation variance constant through ReLU layers (the factor 2 compensates for ReLU zeroing half the inputs); Xavier/Glorot, $\mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$, is the tanh equivalent. These target exactly the $s\approx1$ condition above.

*Normalization layers.* Batch/layer norm rescale activations back to unit variance at every layer, which prevents the compounding drift.

*Gating, for recurrent models.* The LSTM cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ has derivative $f_t$ with respect to $c_{t-1}$; when the forget gate is near 1 the gradient passes through unattenuated. It is a residual connection in time.

**Exploding gradients — causes.** Weights initialized or grown too large; recurrent matrices with spectral radius greater than 1; a learning rate high enough to push the model into a sharp region where the loss curvature is extreme; and occasionally a single pathological batch — a duplicated document, corrupted labels, a numerical overflow in a loss term.

**Exploding gradients — fixes.** These are largely *runtime* interventions, and that is the contrast the interviewer is after.

*Gradient clipping* is the primary tool and has no vanishing-gradient counterpart. Clip by global norm: compute $\|g\|_2$ over all parameters and, if it exceeds a threshold $c$, rescale $g \leftarrow g \cdot c/\|g\|$. Clipping by global norm preserves the *direction* of the update, which is why it is preferred over element-wise value clipping, which distorts it. A threshold of 1.0 is standard for LLM training. Note this is a hard cap applied after the fact — it does not prevent the explosion, it survives it.

*Lower the learning rate*, and use warmup so the early, poorly-conditioned phase does not take a huge step (Q136).

*Careful initialization and normalization*, which overlap with the vanishing fixes because they both target $s\approx1$.

*Fix the data.* An explosion localized to one step is frequently a bad batch. Log the offending batch indices and inspect them before reaching for a hyperparameter.

**The asymmetry, stated plainly.** Vanishing gradients are a *structural* problem — the architecture does not permit signal to reach the early layers — and are fixed by changing the architecture: residuals, ReLU, normalization. Exploding gradients are a *dynamics* problem — the optimizer took too big a step — and are fixed at runtime by clipping and by learning-rate control. Gradient clipping does nothing whatsoever for vanishing gradients, and adding residual connections does not prevent an explosion. Say that sentence.

**Diagnosis.** Log the gradient norm per layer, every step. Vanishing shows as early-layer norms orders of magnitude below late-layer norms, with early-layer weights barely moving from initialization. Exploding shows as a spike in the global norm, usually followed by a loss spike or a NaN. If you are already clipping, log the *pre-clip* norm and the fraction of steps that get clipped — a clip rate that climbs from 1% to 40% is your early warning.

**Follow-up:** *Do transformers still have this problem?* → The vanishing side is largely solved by pre-norm residual architecture. Exploding is very much alive at scale: large-model training runs clip at every step as a matter of course, and loss spikes from gradient explosions are one of the main operational hazards of a long run (Q137).

> **Why the interviewer asks this.** Many candidates give one merged answer — "use ReLU, batch norm, and clipping" — without knowing which fixes which. Being able to separate the structural problem from the dynamics problem is the whole point.

> **Saying it out loud.** "Both come from backprop multiplying Jacobians layer after layer, so anything not close to one becomes exponential in depth. Vanishing is structural — sigmoid's derivative maxes out at 0.25, so ten layers gets you 1e-6 in the best case — and you fix it structurally, with ReLU, residual connections, proper He init, and normalization. Exploding is a dynamics problem, and you fix it at runtime with gradient clipping by global norm and a lower learning rate. That's the distinction I'd emphasize: clipping does absolutely nothing for vanishing, and residual connections don't stop an explosion."

---

### Q136: Why do we need learning-rate warmup, specifically for Adam?

**Answer:**

Warmup means starting at a learning rate near zero and ramping it linearly (or otherwise) to the target over the first few hundred to few thousand steps, before the main decay schedule begins. It is nearly universal in transformer training, and the reason is specific to how Adam estimates its second moment.

**The Adam update.** With $\beta_1 = 0.9$, $\beta_2 = 0.999$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

The step size in each coordinate is $\eta \cdot \hat{m}_t/\sqrt{\hat{v}_t}$ — the gradient divided by its own root-mean-square. Adam is adaptive precisely because of that division.

**The problem is that $\hat{v}_t$ is a terrible estimate early on.** It is an exponential moving average with $\beta_2 = 0.999$, which has an effective averaging window of about $1/(1-\beta_2) = 1000$ steps. At step 10 it has seen ten samples of $g^2$ and is being asked to report their long-run mean. The bias correction fixes the *expectation* but does nothing about the *variance*, and the variance is the whole issue: the update divides by $\sqrt{\hat{v}_t}$, so when $\hat{v}_t$ happens to come in small, the step is enormous.

The pathological case is step 1. Then $m_1 = (1-\beta_1)g_1$ and $v_1 = (1-\beta_2)g_1^2$; after bias correction $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, so the update is exactly $\eta \cdot g_1/|g_1| = \pm\eta$. **Every parameter moves by the full learning rate, with only the sign of a single noisy gradient sample deciding the direction.** Simulating a pure-noise coordinate whose true gradient mean is zero, and measuring the magnitude of Adam's update over time:

```
step     1: E|update|=1.0000  sd=0.0000  p99=1.0000
step     2: E|update|=0.6340  sd=0.3155  p99=1.0012
step     5: E|update|=0.3804  sd=0.2448  p99=0.9199
step    10: E|update|=0.2686  sd=0.1906  p99=0.7664
step    50: E|update|=0.1865  sd=0.1371  p99=0.5808
step  1000: E|update|=0.1811  sd=0.1374  p99=0.5921
```

The steady-state update magnitude for a pure-noise coordinate is about $0.18\eta$. At step 1 it is $1.0\eta$ — a **5.5x overshoot**, in a direction that is pure noise. By step 50 it has converged. That gap, over the first tens to hundreds of steps, is exactly what warmup exists to cover.

This is the argument formalized by RAdam (Liu et al., 2020): the variance of the adaptive learning rate is unbounded in the first few steps, and warmup acts as an implicit variance-reduction heuristic. RAdam instead rectifies the term analytically and turns off adaptivity until the variance estimate is trustworthy, which makes it work without warmup — a useful thing to be able to name.

**Two reinforcing reasons that are not Adam-specific.** At initialization the parameters are random, so the loss surface is poorly conditioned and the gradients are large and uninformative; a big step early moves the model into a bad basin it may never leave. And large-batch training uses a large target learning rate to compensate for the reduced gradient noise, which makes the early-step overshoot correspondingly more destructive — warmup is essentially mandatory once batch size gets large, and its length typically scales with batch size.

**Practical parameters.** Linear warmup over roughly 1% to 5% of total steps, commonly 500 to 4,000 for a large model, then cosine decay to about 10% of peak. Transformers are the standard case; the original paper's $\sqrt{d_{\text{model}}}$-scaled schedule with 4,000 warmup steps was for post-norm, which is genuinely untrainable without it. Pre-norm architectures need warmup much less — that is one of the main reasons pre-norm won — but essentially everyone still uses it, because it is nearly free and removes a whole class of failure.

**Follow-up:** *Would warmup help plain SGD?* → Somewhat, for the poor-conditioning-at-init reason and especially with large batches, but far less, because SGD's step is proportional to the gradient magnitude rather than normalized by it. When SGD's early gradients are noisy the steps are just noisy; when Adam's are noisy the division by a badly-estimated $\sqrt{\hat{v}}$ makes them noisy *and* full-sized.

> **Why the interviewer asks this.** "Because everyone does it" is the failing answer. The variance of $\hat{v}$ at small $t$ is the real one, and it demonstrates you have read the update rule rather than just called the optimizer.

> **Saying it out loud.** "It's about the second-moment estimate. Adam divides the gradient by the square root of a running average of squared gradients, and with beta-2 at 0.999 that average needs about a thousand steps to settle. Early on it's high-variance, and when it comes in small you take a huge step. At step one it's degenerate — the bias-corrected update is exactly plus-or-minus the full learning rate, no matter what the gradient was. I simulated it: for a pure-noise coordinate, the steady-state update is about 0.18 times the learning rate, and at step one it's 1.0. So warmup just keeps the learning rate small until that estimate is trustworthy."

---

### Q137: What does a loss spike mean and what do you do about it?

**Answer:**

A loss spike is a sudden jump in training loss — often by several nats, sometimes to NaN — after a period of stable descent. In small-scale training it is a curiosity. In large-model pretraining it is one of the main operational hazards, because a single unrecovered spike can waste days of accelerator time.

**What it means, in order of how often it is the cause.**

*A bad batch.* By far the most common. A batch containing corrupted text, a document of repeated tokens, mislabelled data, or an extreme outlier produces a huge gradient, the optimizer takes a large step, and the model is damaged. The tell is that the spike is tied to a specific step, and re-running from a checkpoint with a different data order does not reproduce it.

*Gradient explosion.* The gradient norm blows up (Q135), the step is enormous, and the model is thrown out of its basin. The tell is the pre-clip gradient norm spiking one or a few steps before the loss does — which is why you log pre-clip norm, not post-clip.

*Learning rate too high for the current curvature.* Training enters a sharper region and the step size that was fine yesterday is now past the stability edge. The tell is repeated spikes at roughly regular intervals, and a spike right after a warmup ends or a schedule changes.

*Numerical precision.* In fp16, activations or attention logits overflow to `inf` and produce NaN. The loss-scaling machinery normally catches this, but a persistent overflow means the scale is being repeatedly halved. bf16 has the same exponent range as fp32 and largely eliminates this class of failure; if you are training in fp16 at scale, this is your prime suspect.

*Adam state pathology.* If a parameter's gradient has been near zero for a long stretch, its $v$ accumulator decays toward zero; when a real gradient finally arrives, dividing by a tiny $\sqrt{v}$ produces an enormous step. This is a known contributor to spikes in large runs, and it is why some practitioners lower $\beta_2$ to $0.95$ for large-model training — the shorter window is more responsive and less prone to a stale, tiny $v$. The $\epsilon$ value matters here too.

*Attention logit growth.* In large transformers, the pre-softmax logits can grow without bound during training, driving the softmax toward one-hot, gradients toward zero, and the numerics toward the edge. QK-normalization — applying a norm to the query and key vectors before the dot product — is the standard mitigation and is now common in large model training recipes.

**What to do, in the order you should do it.**

*First, does it recover?* Many spikes self-heal within a few hundred steps and cost you nothing but nerves. Watch before you act. A spike that recovers to the previous trajectory needs no intervention.

*If it does not recover, roll back and skip.* The standard playbook: restore the most recent checkpoint from before the spike, skip the batches that produced it (typically the surrounding few hundred), and resume. This is not a hack — it is documented practice in the PaLM and OPT training reports, and it works because the cause is usually data-specific. Crucially, restore the *optimizer state* along with the weights, since damaged $m$ and $v$ will re-cause the problem.

*Then investigate the skipped batches.* If they contain garbage, fix the data pipeline; you have found a real bug and it will recur.

*Structural mitigations, if spikes are frequent.* Lower the peak learning rate or extend warmup. Tighten gradient clipping. Switch fp16 to bf16. Lower $\beta_2$ from $0.999$ to $0.95$. Add QK-norm or z-loss (a small penalty on the log-partition function of the softmax, which keeps logits from drifting). Improve data cleaning and shuffling so that similar documents do not cluster in a batch.

**What not to do.** Do not simply continue and hope — an unrecovered spike frequently means the model has lost capability it will spend many steps re-earning, and sometimes never recovers. Do not diagnose without the pre-clip gradient norm; post-clip norm is capped by construction and shows you nothing.

**Monitoring that makes this tractable.** Per-step: loss, pre-clip global gradient norm, clip rate, learning rate, and — if using fp16 — the loss scale. Per-N-steps: per-layer gradient norms, weight norms, and the max absolute activation. Checkpoint often enough that rolling back costs an acceptable amount of compute; for a long run that usually means every few hundred to few thousand steps. Retain the exact data ordering, and the RNG seed, so a rollback is reproducible and you can identify which batches to skip.

**Follow-up:** *Is a validation loss spike different from a training loss spike?* → Yes, and importantly so. A training spike is an optimization event. Validation loss rising while training loss keeps falling is overfitting, and the response is regularization or early stopping, not a rollback. Validation loss rising in step with a training spike is the optimization event showing through — treat it as one incident.

> **Why the interviewer asks this.** It is an operations question dressed as a theory question. Anyone who has babysat a real training run has the rollback-and-skip playbook and the pre-clip-gradient-norm habit; anyone who has not, does not.

> **Saying it out loud.** "Usually it's a bad batch — corrupted text or a document of repeated tokens — producing a huge gradient and a step that damages the model. Second most common is straightforward gradient explosion, and the way you tell is that the pre-clip gradient norm spikes a step or two before the loss does, which is why you log pre-clip and not post-clip. First thing I do is watch: a lot of spikes self-heal in a few hundred steps. If it doesn't recover, roll back to the last checkpoint including the optimizer state, skip the offending batches, and resume — that's the standard playbook from the PaLM and OPT reports. Then go look at what was in those batches, because it's usually a real data bug."

---
## Modern LLM Systems

### Q138: What does FlashAttention actually do?

**Answer:**

Start by killing the common wrong answer, because interviewers ask this question specifically to see whether you have it. **The wrong answer is "FlashAttention is an efficient approximation of attention that reduces the quadratic complexity."** It is not an approximation and it does not reduce FLOPs. FlashAttention computes *exactly* the same output as standard attention, bit-for-bit equivalent up to floating-point reassociation, and it performs the same $O(n^2 d)$ arithmetic. What it reduces is **memory traffic** — the number of reads and writes between the GPU's high-bandwidth memory (HBM) and its on-chip SRAM.

**Why memory traffic is the bottleneck.** A modern GPU has roughly two orders of magnitude more arithmetic throughput than memory bandwidth. Attention is a *memory-bound* operation: the arithmetic per byte moved is low, so the matrix units sit idle waiting for data. Standard attention makes this worse than necessary by materializing intermediates in HBM:

```
S = Q K^T          write n x n to HBM
P = softmax(S)     read n x n, write n x n
O = P V            read n x n
```

Three round trips of an $n\times n$ matrix per head. The size of that matrix is the problem:

```
seq=1024:   n^2 fp16 per head = 0.0021 GB;  x32 heads =    0.07 GB
seq=8192:   n^2 fp16 per head = 0.1342 GB;  x32 heads =    4.29 GB
seq=131072: n^2 fp16 per head = 34.36 GB;   x32 heads = 1099.51 GB
```

At 128K context the attention matrices alone would require a terabyte. That is why long context was impossible before this, and it is a memory problem, not a compute problem.

**What FlashAttention does instead.** It tiles $Q$, $K$, and $V$ into blocks that fit in SRAM, and computes the output for a block of queries by streaming through blocks of keys and values, accumulating the result — never writing the $n \times n$ matrix anywhere. Two techniques make this possible.

*Online softmax.* Softmax normally needs the full row before it can normalize. The online formulation keeps a running maximum $m$ and a running sum of exponentials $\ell$, and when a new block arrives with its own $m_{\text{new}}$, rescales the accumulated output by $e^{m_{\text{old}} - m_{\text{new}}}$ before adding the new contribution. This is algebraically exact — it is the same numerically-stable max-subtraction trick everyone already uses, applied incrementally. This is the mathematical core of the method.

*Recomputation instead of storage in the backward pass.* Rather than saving the $n\times n$ attention matrix for backprop, FlashAttention saves only the per-row softmax statistics $(m, \ell)$ — which are $O(n)$ — and recomputes the attention blocks on the fly. It spends *extra* FLOPs to avoid memory traffic, which is exactly backwards from ordinary optimization intuition and exactly right on this hardware.

**The results.** Memory goes from $O(n^2)$ to $O(n)$ in sequence length. HBM accesses drop from $O(n^2 d)$ to roughly $O(n^2 d^2/M)$ where $M$ is the SRAM size — still quadratic in $n$, note, but divided by a large constant. The reported speedups are around 2-4x wall-clock for attention with substantially reduced memory. Crucially, **the output is exact**, so there is no accuracy trade-off to evaluate, no hyperparameter, and nothing to validate — you turn it on and the model is the same model. That is the property that made it universal, whereas approximate attention methods (Linformer, Performer, sparse patterns) required you to accept a quality cost and mostly did not stick.

The lineage: FlashAttention-2 improved the work partitioning across thread blocks and warps and cut non-matmul FLOPs; FlashAttention-3 targeted Hopper-generation hardware with asynchrony and FP8 support. **Time-sensitive:** the specific speedup multiples and the current version number date quickly and are hardware-specific — quote the mechanism, not a benchmark figure.

**What it is not.** It does not reduce the asymptotic $O(n^2)$ compute. It does not change the model, the weights, or the output. It is not an alternative to sparse or linear attention — those change the math; this changes the memory schedule. It is orthogonal to and composable with multi-query and grouped-query attention, which reduce the KV cache rather than the attention computation.

**Follow-up:** *If it does not reduce FLOPs, why is it faster?* → Because attention was never FLOP-limited. The matrix units were stalled waiting on HBM. Removing the stalls raises achieved utilization, so the same arithmetic finishes sooner. This is why the backward pass profitably recomputes: extra FLOPs are cheap, extra memory traffic is not.

> **Why the interviewer asks this.** It is a precise filter for whether you understand the memory hierarchy of a GPU or are pattern-matching on "efficient attention." Saying "it's exact and it doesn't reduce FLOPs" in the first sentence is the strongest possible opening.

> **Saying it out loud.** "The thing people get wrong is calling it an approximation — it's exact, and it doesn't reduce FLOPs at all. What it reduces is memory traffic between HBM and on-chip SRAM. Standard attention writes the full n-by-n matrix out to HBM and reads it back twice, and at long context that matrix is enormous. FlashAttention tiles the computation so blocks fit in SRAM and uses an online softmax with a running max and running sum, so it never materializes the matrix. In the backward pass it even recomputes attention rather than storing it — spending extra FLOPs to save memory traffic, because attention was memory-bound, not compute-bound."

Sources: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

---

### Q139: Explain speculative decoding, including why the output distribution is provably unchanged.

**Answer:**

Autoregressive generation is memory-bound in exactly the way attention is. Producing one token requires reading every weight of the model from HBM, and the arithmetic done with those weights — a batch of one token — is trivial. The GPU is idle most of the time. If you could process several tokens per weight-read, you would get them almost for free.

**The algorithm.** Keep two models: a small fast **draft** model $q$ and the large **target** model $p$ whose output distribution you must reproduce exactly.

1. The draft model autoregressively generates $k$ candidate tokens $x_1 \ldots x_k$ (typically $k = 4$ to $8$). This is cheap because the draft is small.
2. The target model runs **one forward pass over all $k$ candidates in parallel**, which costs about the same as generating one token, because that single pass is bounded by reading the weights, not by the arithmetic. This yields $p(\cdot \mid \text{prefix}, x_{<i})$ for every position $i$.
3. Walk left to right. Accept token $x_i$ with probability $\min\!\left(1, \frac{p(x_i)}{q(x_i)}\right)$.
4. On the first rejection at position $i$, discard $x_i$ and everything after it, and sample a replacement from the **residual distribution** $p'(x) \propto \max(0,\; p(x) - q(x))$.
5. If all $k$ are accepted, the target's own distribution at position $k+1$ gives you one extra free token.

So each round produces between 1 and $k+1$ tokens for roughly the cost of one target forward pass.

**The correctness proof.** This is the part that separates a real answer from a sketch. Claim: the token emitted at each position is distributed exactly as $p$.

Let $\alpha(x) = \min(1, p(x)/q(x))$. The probability that token $x$ is drawn from the draft *and* accepted is

$$q(x)\,\alpha(x) = q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) = \min\big(q(x),\, p(x)\big)$$

The total acceptance probability is therefore $\beta = \sum_x \min(p(x), q(x))$, which is $1 - \text{TV}(p,q)$, one minus the total variation distance. On rejection — probability $1-\beta$ — we sample from the residual $p'(x) = \frac{\max(0,\,p(x)-q(x))}{\sum_{x'}\max(0,\,p(x')-q(x'))}$, and the normalizer is exactly $1-\beta$ because $\sum_x \max(0, p-q) = \sum_x (p - \min(p,q)) = 1 - \beta$.

Put the two paths together:

$$P(\text{emit } x) = \underbrace{\min(p(x), q(x))}_{\text{accepted}} + \underbrace{(1-\beta)\cdot\frac{\max(0,\,p(x)-q(x))}{1-\beta}}_{\text{rejected, resampled}} = \min(p,q) + \max(0, p-q)$$

And $\min(a,b) + \max(0, a-b) = a$ for any reals — if $p \ge q$ it is $q + (p-q) = p$; if $p < q$ it is $p + 0 = p$. So $P(\text{emit } x) = p(x)$. **Exactly the target distribution, for any draft model whatsoever.** The draft's quality affects only the *speed*, never the *output distribution* — a deliberately terrible draft model gives you no speedup and a still-perfectly-correct sample.

Verified empirically over 4,000,000 draws with a random 6-token vocabulary:

```
target p : [0.23676 0.10028 0.01523 0.00614 0.30230 0.33928]
empirical: [0.23671 0.10031 0.01531 0.00618 0.30212 0.33936]
max abs err: 1.77e-4
acceptance rate: 0.51380    theoretical sum min(p,q) = 0.51391
```

Both the distribution and the predicted acceptance rate $\sum_x\min(p,q)$ match.

**Speedup arithmetic.** With per-token acceptance probability $\alpha$ (assumed independent), the expected number of tokens per round is a truncated geometric sum:

$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{k+1}}{1-\alpha}$$

```
alpha=0.7 k=4: 2.773      alpha=0.8 k=4: 3.362      alpha=0.9 k=4: 4.095
alpha=0.7 k=8: 3.199      alpha=0.8 k=8: 4.329      alpha=0.9 k=8: 6.126
```

Note the diminishing returns in $k$: at $\alpha = 0.7$, going from $k=4$ to $k=8$ buys only $2.77 \to 3.20$ while doubling the draft cost. The optimal $k$ falls as $\alpha$ falls. Real end-to-end speedups are typically 2-3x, less than these numbers suggest because the draft model's own time and the verification overhead are not free.

**When it works and when it does not.** It works when the draft agrees with the target often — which means on predictable text, code boilerplate, and formatting, and much less on genuinely hard reasoning tokens. It requires the draft to share the target's tokenizer. It hurts throughput under high batch load, because at large batch sizes the GPU is already compute-saturated and the wasted draft work is a real cost; speculative decoding is a *latency* optimization for low-batch serving, not a throughput one. That trade-off is the practical point worth volunteering.

Variants worth naming: **Medusa** attaches extra prediction heads to the target model itself instead of using a separate draft; **EAGLE** drafts in feature space rather than token space for higher acceptance; **self-speculation** uses a subset of the target's own layers as the draft; and n-gram or prompt-lookup drafting simply copies from the prompt, which works remarkably well for summarization and code editing where output overlaps input.

**Follow-up:** *Does this work with greedy decoding?* → Yes, and it is simpler: accept a draft token if it equals the target's argmax, reject at the first mismatch and take the target's token. The same "output is unchanged" guarantee holds trivially. The general rejection-sampling scheme is what extends it to temperature sampling, top-p, and any other sampler applied consistently to both models.

> **Why the interviewer asks this.** Anyone can describe draft-and-verify. The acceptance rule and the residual distribution are what make it lossless, and being able to show $\min(p,q)+\max(0,p-q)=p$ on a whiteboard is a decisive answer.

> **Saying it out loud.** "Decoding is memory-bound — you read all the weights to produce one token — so a small draft model proposes maybe five tokens and the big model verifies all of them in one forward pass, which costs about the same as producing one. You accept each draft token with probability min of one and p over q, and on the first rejection you resample from the normalized positive part of p minus q. The reason that's exactly lossless is that accepting gives you min(p,q) and the rejection branch gives you max(0, p minus q), and those two always sum to p. So the draft model only affects your speed, never your output distribution. I verified it on four million samples — matches to about one part in ten thousand."

Sources: [Fast Inference from Transformers via Speculative Decoding](https://openreview.net/pdf?id=C9NEblP8vS), [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

---

### Q140: How does knowledge distillation work, and when is it the right call?

**Answer:**

Knowledge distillation trains a small **student** model to reproduce the behaviour of a large **teacher**, rather than training it from scratch on hard labels. The core insight, from Hinton, Vinyals and Dean (2015), is that the teacher's full probability distribution carries far more information than the correct answer alone.

**The mechanism.** A one-hot label tells the student "this is a 7." The teacher's distribution says "0.90 seven, 0.07 one, 0.02 nine, 0.001 cat" — which additionally encodes that sevens resemble ones, somewhat resemble nines, and are nothing like cats. Hinton called this the **dark knowledge**: a learned similarity structure over the output space that the hard label throws away. It also acts as a per-example difficulty signal, since an ambiguous example gets a high-entropy teacher distribution and is therefore softly weighted down.

The problem is that a well-trained teacher's distribution is nearly one-hot, so the dark knowledge is buried in probabilities of $10^{-6}$ and contributes nothing to the gradient. The fix is **temperature**: divide the logits by $T > 1$ before the softmax,

$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

which flattens the distribution and surfaces the relative ordering of the non-target classes. The student is trained with the same temperature and the combined loss

$$\mathcal{L} = \alpha\, T^2 \cdot \mathrm{KL}\!\left(p_{\text{teacher}}^{(T)} \,\|\, p_{\text{student}}^{(T)}\right) + (1-\alpha)\,\mathrm{CE}(y, p_{\text{student}}^{(1)})$$

The $T^2$ factor is not decoration: softening the distribution scales the gradients of the KL term by roughly $1/T^2$, so multiplying by $T^2$ keeps the two loss terms comparably weighted as you tune $T$. Typical values are $T$ between 2 and 5 and $\alpha$ around 0.5 to 0.9. Forgetting the $T^2$ is a common bug that makes $T$ appear to have no effect.

**Variants.** *Response distillation* matches output distributions, as above. *Feature distillation* additionally matches intermediate hidden states, usually through a learned projection to reconcile widths — this is what DistilBERT and TinyBERT do, and it transfers more signal than logits alone. *Attention distillation* matches attention maps. *Sequence-level distillation*, the dominant form for generative LLMs, simply has the teacher generate a large corpus of outputs and fine-tunes the student on them with ordinary cross-entropy — this is what "training on synthetic data from a bigger model" means, and it is by far the most commonly deployed variant today. *Self-distillation*, where student and teacher are the same architecture, still improves accuracy, which is a strong hint that the regularization effect matters independently of compression.

**When it is the right call.** When you have a latency or cost budget that the large model cannot meet, and you have a large pool of unlabelled in-domain data — distillation needs inputs, not labels, so unlabelled data is enough and the teacher supplies the targets. When you need to specialize: a 7B student distilled from a frontier model on a *single* task routinely matches or beats the teacher on that task while being an order of magnitude cheaper, because it does not have to be good at everything. When you need on-device or edge deployment. And when you already have an expensive ensemble in production and want one model with most of its quality.

**When it is not.** When you do not have enough unlabelled data covering the deployment distribution — the student only learns the teacher's behaviour where you query it, so coverage gaps become silent failures. When you need broad general capability, since capacity is a real constraint and a small student cannot absorb a frontier model's full range. When quantization or pruning would meet your budget more cheaply — those need no training run and no data, so try them first and reach for distillation when they are not enough. And when the teacher's licence forbids using its outputs to train a competing model, which is a genuine and frequently overlooked constraint in the API era.

**Practical notes.** Distill on the *deployment* distribution, not on a generic corpus — the student inherits the teacher's behaviour exactly where you sampled. Distillation compresses biases and hallucinations faithfully along with the capability, so the student inherits the teacher's failure modes. Combine with quantization for compounding gains. And evaluate the student on your task metrics, not on the KL divergence, which can look excellent while the behaviour you care about has degraded.

**Follow-up:** *Can a student beat its teacher?* → On a narrow task, routinely — specializing frees capacity that the teacher spends on generality, and the teacher's soft targets act as a regularizer. Across general capability, essentially never; capacity binds.

> **Why the interviewer asks this.** Distillation is the standard answer to "make this cheaper," so they want to know if you understand *why* soft targets help rather than just that they do — and whether you know that the modern generative version is mostly "fine-tune on teacher-generated text."

> **Saying it out loud.** "You train a small model to match a big model's output distribution instead of the hard labels. The reason that works is the soft distribution carries what Hinton called dark knowledge — it tells you a seven looks a bit like a one and nothing like a cat, which one-hot labels throw away. You raise the softmax temperature to surface that structure and scale the loss by T-squared so the gradient magnitudes stay comparable. For LLMs the version people actually run is sequence-level: have the teacher generate a big corpus and fine-tune the student on it. It's the right call when you've got a latency budget and lots of unlabelled in-domain data — but I'd try quantization first, since it needs no training run at all."

---

### Q141: What is chain-of-thought, when does it help, and when does it hurt?

**Answer:**

Chain-of-thought (CoT) prompting is asking a model to produce intermediate reasoning steps before its final answer, instead of emitting the answer directly. In its original zero-shot form it is literally the phrase "Let's think step by step"; in the few-shot form you provide exemplars that show worked reasoning.

**Why it works — the mechanical explanation, which is the one to give.** A transformer performs a fixed amount of computation per token: a fixed number of layers, each with a fixed width. If a problem requires more sequential computation than the depth of the network provides, the model cannot do it in one forward pass, full stop. Generating intermediate tokens is how a transformer buys additional serial compute — each generated token gets its own full forward pass, and it can attend to the tokens produced before it. Chain-of-thought converts a depth-limited problem into a length-unlimited one, effectively turning the model into a machine with a scratchpad. That framing also predicts the empirical finding that even semantically meaningless filler tokens can help slightly on some tasks, and it explains why CoT does essentially nothing on tasks that were already within a single forward pass.

A secondary effect: intermediate steps condition subsequent generation, so having written "the total is 47" makes the model far more likely to use 47 consistently downstream than if it had to hold it implicitly.

**When it helps.** Multi-step arithmetic and word problems. Symbolic manipulation and logic. Multi-hop questions that require chaining retrieved facts. Planning and code generation, where laying out the structure first improves the output. Anything where a human would need a scratchpad is the reliable heuristic. The original result was also strongly *emergent with scale* — small models get little benefit or are actively harmed, because they generate plausible-looking reasoning that is wrong, and then commit to it.

**When it hurts.** This is the more interesting half and where the good answers separate.

*Tasks where deliberation degrades human performance too.* This is a real, documented result: "Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse" (Liu et al., 2024) draws on the psychology literature on verbal overshadowing and shows that on tasks in that family — implicit statistical learning, facial recognition, classifying data containing exceptions — CoT causes substantial drops. The intuition is that these tasks depend on holistic pattern recognition, and forcing an explicit verbal account of the decision disrupts it. Naming that paper is a strong move in an interview.

*Simple factual retrieval.* If the answer is a single lookup, reasoning adds latency, cost, and the opportunity to talk yourself out of a correct answer.

*Latency- and cost-sensitive paths.* CoT multiplies output tokens, and output tokens dominate both latency and cost in most APIs. A 10x token increase for a 3% accuracy gain is a bad trade in a real-time product.

*When it manufactures false confidence.* CoT produces reasoning that reads as rigorous whether or not it is. Worse, the stated reasoning is not necessarily the *actual* reason for the answer — models have been shown to produce a plausible chain that rationalizes an answer determined by a bias in the prompt, without mentioning that bias. Treating a chain of thought as a faithful explanation is a genuine safety error, and interviewers at labs care about this specifically.

*When errors compound.* Each step is a chance to go wrong, and an early arithmetic slip propagates confidently to the end. Self-consistency — sampling several chains and majority-voting the final answers — is the standard mitigation and works because independent errors diverge while correct reasoning converges.

**The 2026 context, which you should flag as time-sensitive.** Explicit CoT prompting has been substantially absorbed into models themselves: reasoning models trained with reinforcement learning to produce long internal chains before answering do this natively, and telling such a model to "think step by step" is redundant or mildly harmful. The durable content of this answer is the *mechanism* — intermediate tokens buy serial compute — and the failure modes, both of which apply equally to a model's internal reasoning. What dates quickly is the prompting advice.

**Follow-up:** *How do you know whether the reasoning is faithful?* → You largely cannot from the text. Perturbation tests help: change a step in the chain and see whether the answer changes as it should; inject a biasing cue and check whether the chain mentions it. Faithfulness of reasoning traces is an open research problem, and the honest answer is to treat the chain as an artifact that improves accuracy, not as an explanation you can audit.

> **Why the interviewer asks this.** Almost everyone can define chain-of-thought. Far fewer can explain it as serial-computation-through-token-generation, and fewer still know that it measurably *hurts* on a characterizable class of tasks.

> **Saying it out loud.** "A transformer does a fixed amount of computation per token, so if a problem needs more sequential steps than the model has layers, it just can't do it in one pass. Generating intermediate tokens buys serial compute — each token gets its own forward pass and can attend to everything before it. That's why it helps on anything where a human would want a scratchpad. Where it hurts is tasks that are holistic rather than deliberate — there's a nice paper showing CoT degrades performance on exactly the tasks where overthinking hurts humans too. And I'd push back on treating the chain as an explanation. Models will produce a clean-looking chain that rationalizes an answer they reached for a different reason entirely."

Sources: [Mind Your Step (by Step): Chain-of-Thought can Reduce Performance on Tasks where Thinking Makes Humans Worse](https://arxiv.org/abs/2410.21333)

---

### Q142: What is an agent, and how is it actually different from a chatbot with tools?

**Answer:**

The distinction is **who controls the loop**, and that is the whole answer. Everything else follows.

In a chatbot with tools — sometimes called single-turn function calling — the *application* controls the flow. The user asks something, the model may emit a tool call, the application executes it, feeds the result back, and the model produces a reply. The number of steps is fixed or shallowly bounded, the control flow is written in your code, and every turn returns to the user. It is a request-response system with a model in the middle.

In an agent, the *model* controls the flow. It is placed in a loop: observe the current state, decide what to do next, act, observe the result, repeat, and decide for itself when the task is done. The number of iterations is not known in advance, the sequence of tools is not predetermined, and control does not return to the user between steps. The application supplies tools, a goal, and guardrails; the model supplies the plan.

That shift has four concrete consequences, and naming them is what makes the answer more than a definition.

**Autonomy over the number of steps.** Because the model decides when to stop, the cost and latency of a request are unbounded a priori. This is the source of most of the operational difficulty (Q143).

**State that accumulates across steps.** An agent must carry forward what it has learned — files read, results computed, hypotheses ruled out — across many turns. Managing that state against a finite context window is the central engineering problem of agent building, and it is why compaction, summarization, and external memory (scratchpad files, vector stores) are standard rather than optional.

**Error recovery is the model's job.** A chatbot's failed tool call is handled by your error-handling code. An agent sees the error message as an observation and must decide whether to retry, try a different tool, or give up. That means an agent's competence is bounded as much by its ability to interpret failures as by its ability to plan.

**Actions have consequences the user did not individually approve.** A chatbot with tools acts once, visibly. An agent may take fifty actions, some of them writing to systems, before returning. This is why permission models, allow-lists of side-effecting tools, and human-in-the-loop checkpoints on irreversible actions are architectural requirements rather than nice-to-haves.

**The spectrum, and why a purist definition is wrong.** In practice these are not two categories but a continuum, and it is worth saying so: fixed prompt chain, then a router that picks one of $n$ paths, then a bounded loop with a step cap, then an open-ended loop, then multi-agent systems where one agent delegates to others. Each step up the ladder buys flexibility and costs predictability, cost control, and testability.

**The genuinely important corollary: prefer the least agentic thing that solves the problem.** If your task is "extract fields from an invoice," that is a prompt, not an agent — an agent will cost 20x as much, take 30 seconds instead of 2, and fail in more interesting ways. Agents earn their overhead when the sequence of steps genuinely cannot be known in advance: debugging, open-ended research, and any task where step $n$ depends on what step $n-1$ discovered. Interviewers ask this question partly to see whether you reach for the most complex architecture by default.

**Follow-up:** *What about multi-agent systems?* → They add a second axis — delegation between models — and are worth it mainly when subtasks are genuinely parallel, or need isolated context windows so one subtask's clutter does not pollute another's. They also multiply the failure modes: error compounding across handoffs and information loss at the interface between agents are well-documented problems, and a single agent with good tools beats a poorly-decomposed multi-agent system most of the time.

> **Why the interviewer asks this.** "Agent" is the most overloaded word in the field right now, and the question is really testing for architectural judgment: can you say what the actual technical difference is, and do you know when *not* to build one?

> **Saying it out loud.** "The difference is who's driving the loop. With a chatbot that has tools, my application code controls the flow — model calls a tool, I execute it, hand back the result, done. With an agent, the model decides what to do next and when it's finished, so the number of steps isn't known up front. That gets you flexibility for tasks where you genuinely can't script the sequence, like debugging. But it costs you predictability, cost control, and testability, so my default is to build the least agentic thing that solves the problem. If the steps are knowable, write them down — don't make the model rediscover them every time."

---

### Q143: How do you stop an agent loop from running forever or costing unbounded money?

**Answer:**

An agent decides for itself when to stop, so "when to stop" is a property of your system, not of the model. Every production agent needs layered limits, and the right answer here is a list of independent mechanisms — because any single one can be defeated by a sufficiently confused model.

**Hard limits, which are non-negotiable.** A maximum iteration count, checked in your loop, not requested in the prompt. A wall-clock deadline. A token budget covering both input and output across the whole run. A tool-call count, and a per-tool count so one flaky API cannot be hammered. All of these must be enforced in the orchestration code, because a model asked to "use at most 10 steps" will exceed it. Choose the numbers from the observed distribution of successful runs — if 95% of successful runs finish in 12 steps, cap at 25, not 200. A cap far above the real distribution is not a safety limit, it is an expensive way to fail.

**Cost limits, stated in money.** Track spend per run and per user or tenant, and hard-stop at a ceiling. Two ceilings are useful: a per-run cap that kills the individual run, and a rolling per-hour account-level cap that catches a bug that has spawned a thousand runs — this is the one that actually saves you, because a per-run cap of \$2 does nothing when something is launching runs in a tight loop. Alert on the rate of spend, not just the total, so you find out during the incident rather than on the invoice.

**Loop detection, which is more useful than a step cap.** The characteristic agent failure is not an infinite variety of actions but the *same* action repeated. Detect it directly: hash each (tool, normalized arguments) pair and flag exact repeats; flag a cycle where the last $n$ actions repeat a previous window; detect a no-progress condition where state has not changed across several steps. On detection, do not just kill the run — inject an observation telling the model it is repeating itself and to try something different. That often recovers, and it is much better than failing.

**Progress requirements.** Require the agent to state a goal and check off subgoals; if no subgoal has been completed in $k$ steps, escalate. This turns "am I looping?" into a measurable condition rather than a heuristic.

**Structural limits on the loop's shape.** Cap recursion depth for sub-agents, and cap the total number of sub-agents spawned across a run — an agent that can spawn agents is a fork bomb waiting for a bad prompt. Cap retries per tool with exponential backoff, and after the cap return a *terminal* error the model cannot retry, rather than the same transient error again.

**Graceful degradation, which matters for user experience.** When a limit is hit, do not just error out. Return the partial work with an explicit statement of what was completed and what was not, so the user gets something and can decide whether to continue. An agent that burned \$4 and returns nothing is worse than one that burned \$4 and returns three of five findings.

**The observability that makes any of this tunable.** Log every step: the action, arguments, result, token counts, cost, and latency. Trace the whole run with a correlation ID. Then plot the distribution of steps-to-completion for successful and failed runs — the gap between those two distributions is where your caps belong. Without this you are guessing.

**The design principle underneath all of it.** Bound at multiple independent levels, because each mechanism has a blind spot: a step cap does not stop one enormous tool result from blowing your context and cost; a token cap does not stop a fast loop of cheap calls from hammering an external API; loop detection does not catch an agent making steady, varied, useless progress. And put every limit in the harness, never in the prompt. A prompt is a request; a harness limit is a guarantee.

**Follow-up:** *What about an agent that is making progress but far too slowly?* → That is the hardest case, because no single check fires. Use a budget-aware prompt — tell the agent how much of its budget remains and instruct it to prioritize and summarize as it approaches the limit — plus a checkpoint at, say, 50% budget where the agent must report progress and either continue or hand back. Human-in-the-loop escalation at that checkpoint is the reliable answer for expensive long-running tasks.

> **Why the interviewer asks this.** This is the question that reveals whether you have run an agent in production or only demoed one. Everyone who has shipped one has a story about a loop and a bill.

> **Saying it out loud.** "Layered limits, all enforced in the harness and never in the prompt, because a prompt is a request and a harness limit is a guarantee. Max steps, wall-clock deadline, token budget, and a dollar cap — and I want two dollar caps, one per run and one rolling per-hour on the account, because a two-dollar per-run cap does nothing when a bug is spawning a thousand runs. Then loop detection, which is more useful than the step cap in practice: hash the tool plus normalized arguments and catch exact repeats, and when you catch one, tell the agent it's repeating itself rather than just killing the run — it usually recovers. And I set the caps from the observed distribution of successful runs, not from a round number."

---

### Q144: What is the single biggest practical failure mode of agents in production?

**Answer:**

**Compounding error over a long horizon.** Every other failure mode is either a special case of it or much easier to fix.

**The arithmetic that makes it vivid.** If each step of an agent succeeds independently with probability $p$, the probability of completing $n$ steps is $p^n$. At $p = 0.95$ — which is a genuinely good per-step reliability, better than most tool-calling setups achieve — the probability of a clean 20-step run is $0.95^{20} = 0.358$. At 50 steps, $0.95^{50} = 0.077$. A model that is right 95% of the time fails two out of three moderately long tasks.

This is why agent demos work and agent products do not. A demo is five steps: $0.95^5 = 0.77$, and if you run it twice you get a good take. A real task is forty steps, and the same components deliver under 13%.

**Why it is worse than the arithmetic suggests.** Three effects make real agents fall below the independence bound.

*Errors are not independent, they are absorbing.* A wrong belief entering the context stays there, and every subsequent step conditions on it. An agent that misreads a config file does not merely have one bad step — it has a corrupted premise that poisons all remaining reasoning. The failures correlate, so the effective $p$ drops as the run proceeds.

*Recovery requires recognizing failure, which is the weak capability.* Models are noticeably better at doing things than at noticing they did the wrong thing. A tool that returns an empty result, a silently truncated file, or a plausible-but-wrong value often gets treated as success. The agent then builds confidently on sand. This is the specific gap that makes long-horizon autonomy hard.

*Context degradation.* Over many steps the context fills with tool outputs, errors, and retries. The signal-to-noise ratio falls, the original instruction recedes, and the model starts attending to the clutter — drifting from the goal, re-doing completed work, or losing a constraint stated at the top. Compaction helps and also loses information, so it trades one failure for another.

**The visible symptoms, all downstream of this.** Getting stuck in a repeated tool call. Confidently reporting success on a task that was not done — the worst one, because it is silent. Drifting from the original objective. Cost blowing up as the agent flails.

**What actually mitigates it.** Not "a better model," though that helps; the fixes are architectural.

*Shorten the horizon.* Decompose into subtasks that each complete in a handful of steps, with a verified checkpoint between them. Reliability then multiplies over the number of *subtasks*, not the number of steps, and each subtask starts with a clean context. This is the single highest-leverage change.

*Verify at every step, mechanically.* Do not trust the model to notice failure — run the tests, check the exit code, validate the schema, re-read the file you claimed to write. Ground truth from the environment beats the model's self-assessment every time. A tool that returns a structured success/failure signal is worth far more than one that returns prose.

*Make failure loud.* Tools should error explicitly rather than return empty or truncated results that look like data. Half the silent-failure problem is tools that fail quietly.

*Constrain the action space.* Fewer, better-designed, more clearly-documented tools produce far higher per-step reliability than a large undifferentiated set. Every additional similar tool is another chance to pick the wrong one.

*Checkpoint and make actions reversible.* Snapshot state so a bad branch can be rolled back rather than repaired.

*Keep a human at the irreversible steps.* Full autonomy is the wrong target for most products; the achievable target is an agent that does 90% of the work and asks before it does anything it cannot undo.

**Time-sensitive.** The per-step reliability of frontier models is improving, which shifts the viable horizon outward year over year — the length of task an agent can complete unaided has been roughly doubling on a period of months by some measurements. The *structure* of the argument does not change: whatever $p$ is, $p^n$ still decays, so the engineering response is always to shorten $n$ and verify externally rather than to wait for $p$.

**Follow-up:** *Which is worse, an agent that fails loudly or one that gives up?* → Giving up is far better. A loud failure is a retry; a confident wrong report is a wrong result that enters a downstream system and is discovered much later, by someone who trusted it. Design the reward and the prompt so that "I could not complete this, here is what I did and where I stopped" is an acceptable outcome, because otherwise you are training the agent to fabricate completion.

> **Why the interviewer asks this.** Answering "hallucination" marks you as someone who has read about agents. Answering "compounding error over long horizons, here's the $p^n$ math, and here's why you fix it by shortening the horizon and verifying externally" marks you as someone who has shipped one.

> **Saying it out loud.** "Compounding error over a long horizon. If each step is 95% reliable — which is honestly better than most tool setups get — then twenty steps is 0.95 to the twentieth, about 36%. Fifty steps is under 8%. That's why demos work and products don't. And it's worse than that because errors aren't independent: one wrong belief lands in the context and poisons everything after it, and models are much better at doing things than at noticing they did the wrong thing. So the fix isn't a better model, it's architecture — decompose into short subtasks with verified checkpoints, and verify with the environment rather than asking the model whether it succeeded."

---
