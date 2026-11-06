# Core ML coding: understand it, implement it, check it

<span class="practice-page"></span>

**Foundation material · Very important**

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

Each answer below explains the task, works through the idea, then gives code and a small example. Read the example before trying to memorize the implementation.

The NumPy examples all use this import:

```python
import numpy as np
```

Rows represent examples; columns represent features or classes. The implementations use `float64` for clear numerical checks. Inputs are finite unless a section explicitly allows another value. Shapes, label rules and tie policies are stated with each task.

All functions are also available in [one runnable Python file](core_ml_readable.py). Most examples need only NumPy. F047 includes a separate PyTorch version for training with automatic differentiation. Nothing downloads a dataset.

## Jump to a question

- [F041: Softmax and log-softmax](#f041)
- [F042: LayerNorm forward and backward](#f042)
- [F043: MLP and backpropagation](#f043)
- [F044: K-means](#f044)
- [F045: K-nearest neighbors](#f045)
- [F046: Pairwise Euclidean distances](#f046)
- [F047: Symmetric InfoNCE](#f047)
- [F048: Binary focal loss](#f048)
- [F049: Calibration error](#f049)
- [F050: Noisy annotation cleaning](#f050)

<a id="f041"></a>

## F041. Implement stable softmax and log-softmax

### What are we trying to do?

A classifier produces raw scores, called **logits**. Turn them into probabilities that are nonnegative and sum to one. For logits shaped `(batch_size, class_count)`, normalize each row independently. The functions also accept a single vector and always normalize the final axis.

For class $j$,

$$
p_j = \frac{e^{x_j}}{\sum_k e^{x_k}}.
$$

The problem is numerical: `exp(1000)` overflows even though the final probabilities are perfectly ordinary.

### Why does subtracting the maximum work?

Subtract the same number $m$ from every score. The factor $e^{-m}$ cancels from numerator and denominator, leaving the probabilities unchanged. Choose $m=\max_j x_j$, so every shifted score is at most zero and its exponential is at most one.

For `[1000, 1001]`, the shifted scores are `[-1, 0]`. Their exponentials are approximately `[0.3679, 1]`; dividing by their sum gives `[0.2689, 0.7311]`.

Log-softmax returns **log probabilities**, calculated directly:

$$
\log p_j=(x_j-m)-\log\sum_k e^{x_k-m}.
$$

This matters because a very small probability can round to zero. Taking its logarithm afterward produces negative infinity, while the direct log-softmax can still return a useful finite value.

### Readable implementation

`keepdims=True` preserves a one-column result for row maxima and totals. NumPy can then subtract or divide that value across the corresponding row.

```python
def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    row_totals = np.sum(exponentials, axis=-1, keepdims=True)
    return exponentials / row_totals


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_row_totals = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_row_totals
```

### Try it

```python
scores = np.array([[1000.0, 1001.0]])
print(np.round(softmax(scores), 4))       # [[0.2689 0.7311]]
print(np.round(log_softmax(scores), 4))   # [[-1.3133 -0.3133]]
assert np.allclose(softmax(scores).sum(axis=-1), 1.0)
assert np.allclose(softmax(scores), softmax(scores + 5000))

# The first probability underflows, but its log probability remains finite.
print(log_softmax(np.array([-1000.0, 0.0])))  # [-1000.     0.]
```

**What to watch for:** these functions expect finite scores and at least one class. Attention masks require a separate policy for negative-infinity scores, especially an all-masked row. Do not silently treat such a row as a valid distribution.

**Interview follow-up:** the softmax Jacobian is $\operatorname{diag}(p)-pp^\top$. With ordinary one-hot cross-entropy, the logit gradient simplifies to $p-y$ before any batch averaging.

<a id="f042"></a>

## F042. Implement LayerNorm forward and backward

### What are we trying to do?

Normalize **each example across its own features**, then apply a learned scale and bias. Let `x` have shape `(N, D)`, and let `gamma` and `beta` each have shape `(D,)`. Return an output shaped `(N, D)` and save intermediate values for backward.

For one row,

$$
\mu=\frac1D\sum_jx_j,\qquad
v=\frac1D\sum_j(x_j-\mu)^2,\qquad
\hat x_j=\frac{x_j-\mu}{\sqrt{v+\epsilon}},\qquad
y_j=\gamma_j\hat x_j+\beta_j.
$$

For `[1, 2, 3, 4]`, the mean is 2.5 and the population variance is 1.25. Subtract the mean, divide by approximately $\sqrt{1.25}$, and obtain `[-1.3416, -0.4472, 0.4472, 1.3416]` when the scale is one and bias is zero.

### Why is backward more than multiplying by the scale?

Changing an input feature changes its own centered value **and** the row's mean and variance. All three effects must reach the gradient.

Let $G=\partial L/\partial Y$ be the gradient arriving from the next operation, and let $U=G\odot\gamma$. With row-wise averages denoted by an overline, the combined derivative is

$$
\frac{\partial L}{\partial X}
=\frac{1}{\sqrt{v+\epsilon}}
\left(U-\overline U-\hat X\odot\overline{U\odot\hat X}\right).
$$

The first term is the direct path. The second corrects for the changing mean; the third corrects for the changing variance. Scale and bias are reused by every row, so their gradients **sum across rows**:

$$
\frac{\partial L}{\partial\gamma_j}=\sum_i G_{ij}\hat x_{ij},\qquad
\frac{\partial L}{\partial\beta_j}=\sum_i G_{ij}.
$$

### Readable implementation

```python
def layer_norm_forward(x, gamma, beta, epsilon=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must be a matrix with at least one feature")
    if gamma.shape != (x.shape[1],) or beta.shape != gamma.shape:
        raise ValueError("gamma and beta need one value per feature")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    row_mean = np.mean(x, axis=1, keepdims=True)
    centered = x - row_mean
    row_variance = np.mean(centered ** 2, axis=1, keepdims=True)
    inverse_std = 1.0 / np.sqrt(row_variance + epsilon)
    normalized = centered * inverse_std
    output = gamma * normalized + beta

    # Save values needed for the backward pass. Do not modify them in place.
    cache = (normalized, inverse_std, gamma.copy())
    return output, cache


def layer_norm_backward(output_gradient, cache):
    normalized, inverse_std, gamma = cache
    output_gradient = np.asarray(output_gradient, dtype=np.float64)
    if output_gradient.shape != normalized.shape:
        raise ValueError("output_gradient must have the same shape as x")

    # Scale and bias are shared across rows, so their gradients SUM over rows.
    gamma_gradient = np.sum(output_gradient * normalized, axis=0)
    beta_gradient = np.sum(output_gradient, axis=0)

    # First move backward through y = gamma * normalized + beta.
    normalized_gradient = output_gradient * gamma
    mean_gradient = np.mean(normalized_gradient, axis=1, keepdims=True)
    mean_gradient_times_input = np.mean(
        normalized_gradient * normalized, axis=1, keepdims=True
    )
    # Account for the direct path, centering, and variance normalization.
    input_gradient = inverse_std * (
        normalized_gradient - mean_gradient
        - normalized * mean_gradient_times_input
    )
    return input_gradient, gamma_gradient, beta_gradient
```

### Try it

```python
x = np.array([[1.0, 2.0, 3.0, 4.0]])
gamma = np.ones(4)
beta = np.zeros(4)
output, cache = layer_norm_forward(x, gamma, beta)
print(np.round(output, 4))  # [[-1.3416 -0.4472  0.4472  1.3416]]

# Imagine that the next operation cares only about output feature 0.
output_gradient = np.array([[1.0, 0.0, 0.0, 0.0]])
dx, dgamma, dbeta = layer_norm_backward(output_gradient, cache)
assert dx.shape == x.shape
assert dgamma.shape == gamma.shape
assert np.array_equal(dbeta, np.array([1.0, 0.0, 0.0, 0.0]))
```

**How to verify the gradient:** define the scalar objective `sum(output * output_gradient)`. Perturb one input or parameter by a small $h$, recompute the objective on both sides, and compare the result with the corresponding analytical derivative:

$$
\frac{\partial L}{\partial w}\approx\frac{L(w+h)-L(w-h)}{2h}.
$$

The accompanying tests do this independently for `x`, `gamma` and `beta`. A constant row normalizes to zero, so its output equals `beta`; its input gradient need not be zero for an arbitrary upstream gradient. Use centered deviations to calculate variance, and keep epsilon **inside** the square root.

<a id="f043"></a>

## F043. Write a two-hidden-layer MLP and backpropagation

### What are we trying to do?

Build a small **regression** network using NumPy and calculate its parameter gradients without automatic differentiation. Inputs have shape `(N, input_size)` and targets have shape `(N, output_size)`. There are two ReLU hidden layers and a linear output layer.

The forward computation is

$$
Z_1=XW_1+b_1,\quad A_1=\max(Z_1,0),\qquad
Z_2=A_1W_2+b_2,\quad A_2=\max(Z_2,0),\qquad
\hat Y=A_2W_3+b_3.
$$

Define mean squared error over **all** $NC$ output entries:

$$
L=\frac{1}{NC}\sum_{i,j}(\hat Y_{ij}-Y_{ij})^2,
\qquad \frac{\partial L}{\partial\hat Y}=\frac{2(\hat Y-Y)}{NC}.
$$

### How do we move backward?

For any dense layer `output = input @ weight + bias`, the incoming gradient gives three rules:

| Quantity | Rule |
|---|---|
| Weight gradient | `input.T @ output_gradient` |
| Bias gradient | `output_gradient.sum(axis=0)` |
| Gradient passed to the previous layer | `output_gradient @ weight.T` |

For ReLU, multiply by `(preactivation > 0)`: positive inputs pass the gradient; negative inputs block it. Apply these rules from the last layer toward the first. The normalization by $NC$ happens once at the output, not again at each layer.

### Readable implementation

The initialization helper uses random weights so hidden units can learn different features. `mlp_loss_and_gradients` calculates gradients but does **not** update the parameters; that prevents an early update from corrupting a later backward step.

```python
def initialize_mlp(input_size, hidden1, hidden2, output_size, seed=0):
    random = np.random.default_rng(seed)
    return {
        "W1": random.normal(0, np.sqrt(2 / input_size), (input_size, hidden1)),
        "b1": np.zeros(hidden1),
        "W2": random.normal(0, np.sqrt(2 / hidden1), (hidden1, hidden2)),
        "b2": np.zeros(hidden2),
        "W3": random.normal(0, np.sqrt(1 / hidden2), (hidden2, output_size)),
        "b3": np.zeros(output_size),
    }


def mlp_loss_and_gradients(x, targets, parameters):
    x = np.asarray(x, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Forward: affine layer, ReLU, affine layer, ReLU, affine output.
    z1 = x @ W1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2
    a2 = np.maximum(z2, 0)
    predictions = a2 @ W3 + b3
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("targets must match the nonempty prediction matrix")
    loss = np.mean((predictions - targets) ** 2)

    # Backward: reverse the exact sequence of forward operations.
    prediction_gradient = 2 * (predictions - targets) / targets.size
    gradients = {}
    gradients["W3"] = a2.T @ prediction_gradient
    gradients["b3"] = np.sum(prediction_gradient, axis=0)

    a2_gradient = prediction_gradient @ W3.T
    z2_gradient = a2_gradient * (z2 > 0)
    gradients["W2"] = a1.T @ z2_gradient
    gradients["b2"] = np.sum(z2_gradient, axis=0)

    a1_gradient = z2_gradient @ W2.T
    z1_gradient = a1_gradient * (z1 > 0)
    gradients["W1"] = x.T @ z1_gradient
    gradients["b1"] = np.sum(z1_gradient, axis=0)
    return float(loss), predictions, gradients
```

### Trace one example, then make an update

Use one input, one unit in each hidden layer, and all weights equal to one. Input 2 travels through the network unchanged, so the prediction is 2. If the target is 1, the loss is 1 and each weight gradient is 4.

```python
parameters = {
    "W1": np.array([[1.0]]), "b1": np.zeros(1),
    "W2": np.array([[1.0]]), "b2": np.zeros(1),
    "W3": np.array([[1.0]]), "b3": np.zeros(1),
}
x = np.array([[2.0]])
targets = np.array([[1.0]])
loss, predictions, gradients = mlp_loss_and_gradients(x, targets, parameters)
print(loss, predictions)       # 1.0 [[2.]]
print(gradients["W1"])        # [[4.]]

learning_rate = 0.01
for name in parameters:
    parameters[name] = parameters[name] - learning_rate * gradients[name]
new_loss, _, _ = mlp_loss_and_gradients(x, targets, parameters)
assert new_loss < loss
```

**What to watch for:** never update `W3` before computing the gradient sent through `W3.T`. Bias gradients sum over the batch because every example uses the same bias. Verify all six parameter gradients with finite differences, using points away from a ReLU kink at zero. For classification, change the output objective and its derivative; do not assume this regression loss is cross-entropy.

<a id="f044"></a>

## F044. Implement k-means with convergence and empty clusters

### What are we trying to do?

Given `N` points with `D` features, partition them into `K` groups represented by centers. Return `centers` shaped `(K, D)` and one cluster index per point. We minimize the sum of squared distances to the assigned centers:

$$
J=\sum_{i=1}^{N}\|x_i-c_{a_i}\|_2^2.
$$

There are no class labels. K-means discovers groups from geometry.

### Why alternate two steps?

If the centers are fixed, each point should choose the nearest center. If assignments are fixed, the mean of a group's points minimizes that group's squared-distance sum. Repeat those two updates until the centers move very little, or an iteration limit is reached.

For `[0, 1, 9, 10]`, a natural split is `[0, 1]` and `[9, 10]`. Their means are 0.5 and 9.5. If a center receives no points, its mean is undefined. This implementation keeps the previous center in that case, an explicit policy that avoids producing NaNs.

### Readable implementation

The inner helper calculates one column of distances per center. Keeping that loop visible makes the assignment step easy to inspect. Squared distances are enough: taking a square root would not change which center is nearest.

```python
def k_means(points, cluster_count, max_iterations=100, tolerance=1e-6, seed=0):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0 or points.shape[1] == 0:
        raise ValueError("points must be a nonempty (N, D) matrix")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if not 1 <= cluster_count <= len(points):
        raise ValueError("cluster_count must be between 1 and N")
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("need positive iteration limit and nonnegative tolerance")
    random = np.random.default_rng(seed)
    initial_indices = random.choice(len(points), cluster_count, replace=False)
    centers = points[initial_indices].copy()

    def assign_to_centers(current_centers):
        squared_distances = np.empty((len(points), cluster_count))
        for cluster in range(cluster_count):
            differences = points - current_centers[cluster]
            squared_distances[:, cluster] = np.sum(differences ** 2, axis=1)
        # Ties go to the first center index.
        return np.argmin(squared_distances, axis=1)

    for _ in range(max_iterations):
        labels = assign_to_centers(centers)
        new_centers = centers.copy()
        for cluster in range(cluster_count):
            members = points[labels == cluster]
            if len(members) > 0:
                new_centers[cluster] = np.mean(members, axis=0)
            # Empty cluster: keep its previous center rather than compute NaN.

        center_movement = np.linalg.norm(new_centers - centers, axis=1)
        centers = new_centers
        if np.max(center_movement) <= tolerance:
            break

    # Labels must refer to the RETURNED centers, even at the iteration limit.
    labels = assign_to_centers(centers)
    return centers, labels
```

### Try it

```python
points = np.array([[0.0], [1.0], [9.0], [10.0]])
centers, labels = k_means(points, cluster_count=2, seed=0)
print(np.sort(centers[:, 0]))   # [0.5 9.5]
assert labels[0] == labels[1]
assert labels[2] == labels[3]
assert labels[0] != labels[2]
```

**What to watch for:** cluster numbers have no inherent meaning; swapping center indices gives the same partition. Labels must be recalculated against the final returned centers, especially when stopping at the iteration limit. Distinct initial row indices do not guarantee distinct point values, so duplicate data can still produce empty clusters.

One assignment/update iteration costs about $O(NKD)$ arithmetic. Random initialization can reach a poor local solution; multiple restarts or k-means++ can help. Feature scaling matters because large numeric units dominate Euclidean distance.

<a id="f045"></a>

## F045. Implement k-nearest-neighbor classification

### What are we trying to do?

Unlike k-means, KNN uses **labeled training examples**. For each query, find the `k` nearest training points and predict their most common label. Training points have shape `(N, D)`, labels have shape `(N,)`, and queries have shape `(M, D)`; the output has one label per query.

For a query near points labeled `[0, 0, 1]`, a three-neighbor vote predicts 0. There are no learned weight matrices in this baseline: it stores the examples and does the distance work at prediction time.

### What decisions must be explicit?

Use squared Euclidean distance. Choose an integer `neighbor_count` between 1 and the training size. If distances tie, preserve the original training-row order. If votes tie, choose the smallest label under the label type's sorting order. These are deterministic conventions, rather than claims that one tied class is more likely to be correct.

### Readable implementation

```python
def knn_predict(training_points, training_labels, query_points, neighbor_count=3):
    training_points = np.asarray(training_points, dtype=np.float64)
    training_labels = np.asarray(training_labels)
    query_points = np.asarray(query_points, dtype=np.float64)
    if training_points.ndim != 2 or query_points.ndim != 2:
        raise ValueError("training_points and query_points must be matrices")
    if training_points.shape[1] != query_points.shape[1]:
        raise ValueError("training and query feature counts must match")
    if training_labels.shape != (len(training_points),):
        raise ValueError("need one label per training point")
    if not 1 <= neighbor_count <= len(training_points):
        raise ValueError("neighbor_count must be between 1 and training size")

    predictions = []
    for query in query_points:
        differences = training_points - query
        squared_distances = np.sum(differences ** 2, axis=1)
        # Stable sorting resolves equal distances by original training-row order.
        neighbor_indices = np.argsort(squared_distances, kind="stable")[:neighbor_count]
        neighbor_labels = training_labels[neighbor_indices]
        classes, vote_counts = np.unique(neighbor_labels, return_counts=True)
        # np.unique sorts classes; np.argmax selects the first maximum.
        winner = classes[np.argmax(vote_counts)]
        predictions.append(winner)
    return np.asarray(predictions, dtype=training_labels.dtype)
```

### Try it

```python
training_points = np.array([[0.0], [1.0], [9.0], [10.0]])
training_labels = np.array([0, 0, 1, 1])
queries = np.array([[0.2], [9.2]])
predictions = knn_predict(training_points, training_labels, queries, neighbor_count=3)
print(predictions)  # [0 1]
```

For query 0.2, the nearest three training values are 0, 1 and 9. Their labels are 0, 0 and 1, so class 0 wins. For query 9.2, the nearest three are 9, 10 and 1, so class 1 wins.

**What to watch for:** fit scaling transformations using training data, then reuse them on queries. Otherwise a feature measured in thousands can overwhelm a useful feature measured between zero and one. When evaluating on training examples, exclude a point from its own neighbor set if your evaluation intends leave-one-out prediction.

This readable version fully sorts distances: per query it uses $O(ND)$ for distance calculation plus $O(N\log N)$ for sorting. Partial selection can reduce sorting work, but preserving a precise tie policy then takes care. Do not describe the full-sort implementation as linear-time neighbor selection.

<a id="f046"></a>

## F046. Compute pairwise Euclidean distances

### What are we trying to do?

Given `x` shaped `(N, D)` and `y` shaped `(M, D)`, return **every cross-pair distance**, shaped `(N, M)`:

$$
d_{ij}=\sqrt{\sum_{r=1}^{D}(x_{ir}-y_{jr})^2}.
$$

This is not a comparison of corresponding rows. There can be different numbers of rows in the two inputs.

### Start with the direct interpretation

`x[:, None, :]` has shape `(N, 1, D)`, while `y[None, :, :]` has shape `(1, M, D)`. Subtracting them broadcasts to `(N, M, D)`: one feature-difference vector for each pair. Square those differences, sum over features, then take the square root.

```python
def pairwise_euclidean(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("expected matrices (N, D) and (M, D)")
    differences = x[:, None, :] - y[None, :, :]  # (N, M, D)
    squared_distances = np.sum(differences ** 2, axis=-1)  # (N, M)
    return np.sqrt(squared_distances)
```

### Try it

```python
x = np.array([[0.0, 0.0], [3.0, 4.0]])
y = np.array([[0.0, 0.0], [0.0, 4.0]])
print(pairwise_euclidean(x, y))
# [[0. 4.]
#  [5. 3.]]
```

The bottom-left entry is the familiar 3–4–5 triangle: the distance from `[3, 4]` to `[0, 0]` is 5. The bottom-right entry is 3 because only the first coordinate differs.

### How can we avoid the large difference tensor?

Expand the square:

$$
\|x_i-y_j\|^2=\|x_i\|^2+\|y_j\|^2-2x_i^\top y_j.
$$

This version avoids storing `(N, M, D)` differences. It still needs the `(N, M)` output and matrix-product intermediates.

```python
def pairwise_euclidean_fast(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("expected matrices (N, D) and (M, D)")
    x_squared_norms = np.sum(x ** 2, axis=1, keepdims=True)  # (N, 1)
    y_squared_norms = np.sum(y ** 2, axis=1)[None, :]       # (1, M)
    squared_distances = x_squared_norms + y_squared_norms - 2 * (x @ y.T)
    squared_distances = np.maximum(squared_distances, 0)  # roundoff protection
    return np.sqrt(squared_distances)
```

```python
assert np.allclose(pairwise_euclidean_fast(x, y), pairwise_euclidean(x, y))
```

**What to watch for:** roundoff can make a mathematically zero squared distance slightly negative, so clamp before taking the square root. However, clamping does not repair all cancellation error for nearly identical, large-offset vectors; direct differences can be more accurate there. For large inputs, compute blocks, or retain only the nearest candidates if the full distance matrix is unnecessary. Both formulas use $O(NMD)$ arithmetic; the direct temporary uses $O(NMD)$ memory, while the algebraic pairwise arrays use $O(NM)$.

<a id="f047"></a>

## F047. Implement symmetric InfoNCE loss

### What are we trying to do?

You have matching query/key pairs—for example, image embeddings and their corresponding caption embeddings. `queries[i]` matches `keys[i]`. Both arrays have shape `(B, D)`, with nonzero rows. Make the correct pair score higher than the other candidates in its batch.

Normalize each embedding to unit length, calculate cosine similarities, and divide by a positive temperature $\tau$:

$$
S_{ij}=\frac{\tilde q_i^\top\tilde k_j}{\tau},\qquad
L_{q\to k}=-\frac1B\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}.
$$

Row $i$ asks “which key matches query $i$?” The correct class index is $i$. To ask the reverse question, transpose the score matrix. **Symmetric** InfoNCE averages the two cross-entropies.

### Readable NumPy implementation

This uses `log_softmax` from F041 above. It calculates the loss for inspection; NumPy does not automatically backpropagate into an embedding model.

```python
def symmetric_info_nce(queries, keys, temperature=0.1):
    queries = np.asarray(queries, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    if queries.ndim != 2 or queries.shape != keys.shape or len(queries) == 0:
        raise ValueError("queries and keys must have the same nonempty (B, D) shape")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    key_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    if np.any(query_norms == 0) or np.any(key_norms == 0):
        raise ValueError("zero vectors have no defined cosine direction")
    normalized_queries = queries / query_norms
    normalized_keys = keys / key_norms
    scores = normalized_queries @ normalized_keys.T / temperature
    positive_indices = np.arange(len(queries))
    # Uses log_softmax from F041: normalize across possible matches in each row.
    query_to_key = -np.mean(log_softmax(scores)[positive_indices, positive_indices])
    key_to_query = -np.mean(log_softmax(scores.T)[positive_indices, positive_indices])
    return float((query_to_key + key_to_query) / 2)
```

### Trace a two-pair example

```python
queries = np.array([[1.0, 0.0], [0.0, 1.0]])
keys = queries.copy()
loss = symmetric_info_nce(queries, keys, temperature=1.0)
print(round(loss, 6))  # 0.313262
```

Here the score matrix is `[[1, 0], [0, 1]]`. Each correct candidate gets probability $e/(e+1)\approx0.7311$. Its negative log probability is about 0.313262. The transpose gives the same loss in this particular example.

### PyTorch version for training

Keep embeddings as tensors so gradients can flow into both embedding branches. Do not convert them to NumPy inside a training loss.

```python
def symmetric_info_nce_torch(queries, keys, temperature=0.1):
    import torch
    from torch.nn import functional as F

    if queries.ndim != 2 or queries.shape != keys.shape or len(queries) == 0:
        raise ValueError("queries and keys must have the same nonempty (B, D) shape")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if (queries.norm(dim=1) == 0).any() or (keys.norm(dim=1) == 0).any():
        raise ValueError("zero vectors have no defined cosine direction")
    queries = F.normalize(queries, dim=1)
    keys = F.normalize(keys, dim=1)
    scores = queries @ keys.T / temperature
    targets = torch.arange(len(queries), device=queries.device)
    return (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2
```

```python
import torch

query_tensor = torch.tensor([[1.0, 0.2], [0.1, 1.0]], requires_grad=True)
key_tensor = torch.tensor([[0.9, 0.0], [0.0, 1.1]], requires_grad=True)
loss = symmetric_info_nce_torch(query_tensor, key_tensor, temperature=0.5)
loss.backward()
assert query_tensor.grad is not None
assert key_tensor.grad is not None
```

**What to watch for:** with one pair, there are no negatives, so this objective is zero even for unhelpful embeddings. The diagonal-target version assumes the other batch entries are negatives; repeated captions or alternative valid matches can violate that assumption. A small temperature makes the distribution sharper, but it is not a substitute for correct pairs.

<a id="f048"></a>

## F048. Implement stable binary focal loss

### What are we trying to do?

For each binary example, take a raw logit $z$ and a label $y\in\{0,1\}$. Ordinary binary cross-entropy penalizes incorrect predictions. Focal loss multiplies it by a factor that **shrinks the contribution of easy examples**:

$$
L=\alpha_y(1-p_t)^\gamma(-\log p_t),
$$

where $p_t$ is the probability assigned to the true label and $\gamma\ge0$. Here `alpha=None` means no class weighting; otherwise positives receive weight `alpha` and negatives receive `1-alpha`. Return the mean over examples.

### Why use logits and stable operations?

For a positive label, define the signed logit as $z$; for a negative label, use $-z$. In both cases, binary cross-entropy is

$$
\operatorname{BCE}=\log(1+e^{-z_{\rm signed}}).
$$

`np.logaddexp(0, -signed_logits)` evaluates that expression without an overflowing exponential. Since $\log p_t=-\operatorname{BCE}$, calculate $1-p_t$ with `-expm1(-BCE)`. This avoids subtracting two nearly equal numbers when an example is very easy.

### Readable implementation

```python
def binary_focal_loss(logits, labels, gamma=2.0, alpha=None):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.shape != labels.shape or logits.size == 0:
        raise ValueError("logits and labels must have matching nonempty shapes")
    if not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("need finite logits and binary labels")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1, or None")

    # Positive label: signed_logit=z. Negative label: signed_logit=-z.
    signed_logits = np.where(labels == 1, logits, -logits)
    binary_cross_entropy = np.logaddexp(0.0, -signed_logits)
    # p_true = exp(-BCE). expm1 computes exp(x)-1 accurately near zero.
    one_minus_p_true = -np.expm1(-binary_cross_entropy)
    focal_weights = one_minus_p_true ** gamma
    losses = focal_weights * binary_cross_entropy
    if alpha is not None:
        class_weights = np.where(labels == 1, alpha, 1 - alpha)
        losses = class_weights * losses
    return float(np.mean(losses))
```

### Compare an uncertain example with an easy example

```python
uncertain = binary_focal_loss([0.0], [1], gamma=2, alpha=None)
easy = binary_focal_loss([5.0], [1], gamma=2, alpha=None)
print(round(uncertain, 6))   # 0.173287
assert easy < uncertain

# With gamma=0 and no alpha weighting, focal loss becomes ordinary BCE.
ordinary_bce = binary_focal_loss([0.0], [1], gamma=0, alpha=None)
assert np.isclose(ordinary_bce, np.log(2))
```

For logit zero, the true-class probability is 0.5. BCE is $\log2\approx0.6931$ and the focal factor is $(1-0.5)^2=0.25$, giving about 0.1733. For logit 5 with a positive label, the example is already easy, so focal weighting reduces its contribution much further.

**What to watch for:** pass logits, not sigmoid probabilities. `gamma=0` with a non-`None` alpha still applies class weighting. This is a NumPy forward reference; a training implementation must use differentiable tensor operations. Hard mislabeled examples can receive disproportionate emphasis, so focal loss does not automatically improve accuracy or calibration.

<a id="f049"></a>

## F049. Calculate average calibration error from confidence bins

### What are we trying to do?

A model is well calibrated when predictions made with confidence 0.8 are correct about 80% of the time. Here each input log confidence refers to the model's **predicted class**. A separate `correct` array says whether that prediction was right.

Recover confidence with `exp(log_confidence)`. Divide `[0, 1]` into equal-width bins, then compare each nonempty bin's average confidence with its observed accuracy. This exercise defines **ACE** as the unweighted average of those bin gaps:

$$
\operatorname{ACE}=\frac{1}{|\mathcal B_+|}
\sum_{b\in\mathcal B_+}\left|\operatorname{accuracy}(b)-\operatorname{confidence}(b)\right|.
$$

Empty bins are omitted. A confidence of exactly 1 belongs to the final bin. The convention for no examples is a returned value of zero. Other calibration metrics use different binning or population weights, so state the definition rather than relying on the acronym alone.

### Readable implementation

Return both the error and the bin details so you can inspect how the answer was calculated. Negative infinity is allowed as a log confidence and means probability zero.

```python
def calibration_error(log_confidences, correct, bin_count=10):
    log_confidences = np.asarray(log_confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if log_confidences.ndim != 1 or log_confidences.shape != correct.shape:
        raise ValueError("need equally sized 1D arrays")
    if not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer")
    if np.isnan(log_confidences).any() or np.any(log_confidences > 0):
        raise ValueError("log confidences must be <= 0; -inf means confidence 0")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("correct must contain only 0 and 1")
    if len(correct) == 0:
        return 0.0, []

    confidences = np.exp(log_confidences)
    bin_indices = np.minimum((confidences * bin_count).astype(int), bin_count - 1)
    report = []
    for bin_index in range(bin_count):
        members = bin_indices == bin_index
        count = int(np.sum(members))
        if count == 0:
            continue
        mean_confidence = float(np.mean(confidences[members]))
        accuracy = float(np.mean(correct[members]))
        gap = abs(accuracy - mean_confidence)
        report.append({"bin": bin_index, "count": count,
                       "confidence": mean_confidence, "accuracy": accuracy, "gap": gap})
    average_error = float(np.mean([row["gap"] for row in report]))
    return average_error, report
```

### Trace the bins

```python
confidences = np.array([0.9, 0.95, 1.0, 0.5, 0.55])
correct = np.array([1, 1, 1, 0, 1])
error, report = calibration_error(np.log(confidences), correct, bin_count=5)
print(round(error, 4))  # 0.0375
for row in report:
    print(row["count"], round(row["accuracy"], 3),
          round(row["confidence"], 3), round(row["gap"], 3))
# 2 0.5 0.525 0.025
# 3 1.0 0.95 0.05
```

The two nonempty bins have gaps 0.025 and 0.05, so ACE is their average, 0.0375. A sample-weighted ECE would instead be $(2\times0.025+3\times0.05)/5=0.04$. The distinction matters even though both numbers describe calibration.

**What to watch for:** classification accuracy alone cannot tell you whether confidence estimates are calibrated. Tiny bins can have noisy accuracy estimates, so inspect counts and evaluate on independent data. Do not count an empty bin as evidence of perfect calibration.

<a id="f050"></a>

## F050. Clean noisy annotations using rater agreement

### What are we trying to do?

Several annotators label the same items. Use their agreement as a **screening signal** to identify annotations worth reviewing or retaining. Each record is `(item_id, rater_id, binary_label)`. A rater may vote at most once on an item.

Return three things: the original majority label per item, a reliability report per rater, and retained annotation records. A tied item has consensus `None`; that does not silently become label 0.

### Why exclude the rater from their own reference vote?

Suppose we score an annotator against a majority that includes their own vote. That vote can influence the very reference used to judge them. The code below uses **leave-one-rater-out agreement**: when evaluating Alice's vote, build the reference label from everyone except Alice.

If the remaining votes tie, skip that item for Alice's score. Only retain a rater when enough comparison items are resolved and their agreement meets the chosen threshold. These thresholds are example policies, not universal quality standards.

### Readable implementation

The helper `majority_label` returns 0, 1 or `None`. `defaultdict(list)` simply gives each new item an initially empty list of votes.

```python
from collections import defaultdict
```

```python
def majority_label(votes):
    if len(votes) == 0 or sum(votes) * 2 == len(votes):
        return None  # no votes or a tie: there is no resolved label
    return int(sum(votes) * 2 > len(votes))


def clean_annotations(annotations, minimum_agreement=0.8, minimum_items=3):
    if not 0 <= minimum_agreement <= 1 or minimum_items < 1:
        raise ValueError("invalid agreement threshold or minimum item count")
    annotations = list(annotations)
    by_item = defaultdict(list)
    raters = set()
    seen = set()
    for item, rater, label in annotations:
        if label not in (0, 1) or (item, rater) in seen:
            raise ValueError("need one binary vote per (item, rater)")
        seen.add((item, rater))
        by_item[item].append((rater, label))
        raters.add(rater)

    consensus = {}
    matches = defaultdict(int)
    resolved_counts = defaultdict(int)
    for item, votes in by_item.items():
        consensus[item] = majority_label([label for _, label in votes])
        for rater, label in votes:
            # Leave THIS rater out of the vote used to assess their agreement.
            other_votes = [other_label for other, other_label in votes if other != rater]
            reference = majority_label(other_votes)
            if reference is not None:
                resolved_counts[rater] += 1
                matches[rater] += int(label == reference)

    reliability = {}
    retained_raters = set()
    for rater in sorted(raters):
        count = resolved_counts[rater]
        agreement = matches[rater] / count if count else None
        keep = count >= minimum_items and agreement >= minimum_agreement
        reliability[rater] = {"resolved_items": count, "agreement": agreement, "keep": keep}
        if keep:
            retained_raters.add(rater)
    cleaned = [row for row in annotations if row[1] in retained_raters]
    return consensus, reliability, cleaned
```

### Trace a four-rater example

Three raters agree on every item, and a fourth disagrees on every item. Each agreeing rater can be assessed against two other agreeing votes and one disagreeing vote, so their leave-one-out reference is resolved.

```python
annotations = []
for item, majority in [("item1", 1), ("item2", 0), ("item3", 1)]:
    annotations.extend([
        (item, "Alice", majority),
        (item, "Bob", majority),
        (item, "Chen", majority),
        (item, "Dana", 1 - majority),
    ])
consensus, reliability, cleaned = clean_annotations(
    annotations, minimum_agreement=0.8, minimum_items=3
)
print(reliability["Alice"])  # resolved_items=3, agreement=1.0, keep=True
print(reliability["Dana"])   # resolved_items=3, agreement=0.0, keep=False
assert len(cleaned) == 9
```

For `item1`, Alice is compared with `[1, 1, 0]`, whose majority is 1, so she agrees. Dana is compared with `[1, 1, 1]`, so she disagrees. The same pattern holds on all three items.

**Why this is not proof of correctness:** Alice, Bob and Chen could share the same mistaken guideline. Leave-one-out scoring reduces self-inclusion, but it does not remove correlated majority bias. With only three raters and a 2-to-1 split, an agreeing rater's two peers may tie; the code then records no resolved comparison for that item, rather than manufacturing certainty.

`cleaned` retains the selected raters' original votes. If you need one training label per item, aggregate those retained votes again and keep ties unresolved. Before treating filtering as an improvement, train otherwise comparable baseline and filtered-data models and evaluate them on an **independent gold-labeled set**. Inspect subgroup effects and review disagreement; a minority annotation is not automatically wrong.

## Run the checks

The [test file](test_core_ml_readable.py) checks numerical examples, softmax invariance, LayerNorm and MLP gradients, deterministic neighbor ties, empty clusters, contrastive symmetry, focal-loss limits, calibration bins and annotation ties. It also executes the examples on this page. The differentiable InfoNCE check needs PyTorch.

From this chapter's directory, with NumPy installed:

```bash
python -m unittest test_core_ml_readable -v
```

If PyTorch is absent, only the PyTorch-specific checks and example are skipped. The remaining examples run locally without an internet connection.
