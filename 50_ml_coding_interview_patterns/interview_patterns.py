"""
Compact ML coding interview patterns.
These functions are intentionally short and easy to reconstruct.
"""
from typing import Iterable, List

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def masked_softmax(logits: np.ndarray, mask: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    mask should contain 1 for valid positions and 0 for masked positions.
    """
    masked_logits = np.where(mask == 1, logits, -1e9)
    return stable_softmax(masked_logits, axis=axis)


def causal_mask(seq_len: int) -> np.ndarray:
    """Lower-triangular mask for autoregressive attention."""
    return np.tril(np.ones((seq_len, seq_len), dtype=int))


def pairwise_squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Squared distances between every row of X and every row of C.

    X: (n_samples, d)
    C: (n_centers, d)
    Returns: (n_samples, n_centers)
    """
    x_norm = np.sum(X**2, axis=1, keepdims=True)
    c_norm = np.sum(C**2, axis=1, keepdims=True).T
    return x_norm + c_norm - 2.0 * X @ C.T


def batch_iterator(data: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    """Yield contiguous mini-batches."""
    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]


def pad_sequences(sequences: List[List[int]], pad_value: int = 0) -> np.ndarray:
    """Pad ragged token sequences into a dense matrix."""
    max_len = max(len(seq) for seq in sequences)
    output = np.full((len(sequences), max_len), pad_value, dtype=int)
    for i, seq in enumerate(sequences):
        output[i, : len(seq)] = seq
    return output


def top_k_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k logits, set the rest to a large negative value.
    """
    if k >= len(logits):
        return logits.copy()
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits >= threshold, logits, -1e9)


def top_p_logits(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Keep the smallest set of tokens whose cumulative probability reaches p.
    """
    probs = stable_softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    keep_sorted = cumulative <= p
    if not np.any(keep_sorted):
        keep_sorted[0] = True
    else:
        first_exceed = np.argmax(cumulative > p)
        keep_sorted[first_exceed] = True

    keep_mask = np.zeros_like(logits, dtype=bool)
    keep_mask[sorted_idx[keep_sorted]] = True
    return np.where(keep_mask, logits, -1e9)


def kmeans_step(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    One k-means update:
    1. assign each point to nearest center
    2. recompute centers as cluster means
    """
    distances = pairwise_squared_distances(X, centers)
    labels = np.argmin(distances, axis=1)
    new_centers = centers.copy()

    for k in range(len(centers)):
        mask = labels == k
        if np.any(mask):
            new_centers[k] = np.mean(X[mask], axis=0)

    return new_centers


if __name__ == "__main__":
    print("ML Coding Interview Patterns")
    print("=" * 60)

    logits = np.array([2.0, 1.0, 0.1, -1.0])
    print("Stable softmax:", stable_softmax(logits).round(4))
    print("Top-k logits:", top_k_logits(logits, k=2))
    print("Top-p logits:", top_p_logits(logits, p=0.8))

    mask = causal_mask(4)
    print("\nCausal mask:")
    print(mask)

    scores = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.5, 0.5, 2.5, 3.5],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    print("\nMasked softmax with causal mask:")
    print(masked_softmax(scores, mask).round(4))

    X = np.array([[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 2.8]])
    centers = np.array([[0.0, 0.1], [3.1, 3.1]])
    print("\nPairwise squared distances:")
    print(pairwise_squared_distances(X, centers).round(4))
    print("Updated centers:")
    print(kmeans_step(X, centers).round(4))

    sequences = [[1, 2, 3], [4, 5], [6]]
    print("\nPadded sequences:")
    print(pad_sequences(sequences, pad_value=0))
