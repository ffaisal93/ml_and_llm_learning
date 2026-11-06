"""Readable numerical implementations for F041–F050.

NumPy functions use float64 so the tiny examples and gradient checks are easy
to inspect. InfoNCE also has a differentiable PyTorch implementation.
"""
from collections import defaultdict
import numpy as np


# F041: rows are examples; the final axis contains class scores.
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


# F042: x has shape (batch_size, feature_count).
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


# F043: a two-hidden-layer REGRESSION network with mean squared error.
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


# F044: keeping the assignment loop explicit makes the algorithm easy to trace.
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


# F045: labels may be numbers or strings, provided they can be sorted together.
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


# F046: start with direct differences; optimize only after understanding shapes.
def pairwise_euclidean(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("expected matrices (N, D) and (M, D)")
    differences = x[:, None, :] - y[None, :, :]  # (N, M, D)
    squared_distances = np.sum(differences ** 2, axis=-1)  # (N, M)
    return np.sqrt(squared_distances)


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


# F047: forward numerical reference; matching examples share a row index.
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


# F048: inputs are LOGITS, not probabilities; return a mean over examples.
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


# F049: equal-width confidence bins, equal weight per NONEMPTY bin.
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


# F050: each annotation is (item_id, rater_id, binary_label).
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
