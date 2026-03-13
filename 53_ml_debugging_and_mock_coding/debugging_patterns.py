"""
Compact debugging helpers and patterns for ML interviews.
"""
from typing import Dict, Tuple

import numpy as np


def check_binary_labels(y: np.ndarray) -> bool:
    """Return True if labels look binary {0, 1}."""
    unique = set(np.unique(y).tolist())
    return unique.issubset({0, 1})


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe log."""
    return np.log(np.clip(x, eps, None))


def clip_gradients(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    Clip a vector gradient to a maximum L2 norm.
    """
    norm = np.linalg.norm(grad)
    if norm <= max_norm or norm == 0.0:
        return grad
    return grad * (max_norm / norm)


def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Binary accuracy from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(np.mean(y_true == y_pred))


def inspect_training_step(
    logits: np.ndarray, probs: np.ndarray, grad: np.ndarray
) -> Dict[str, float]:
    """
    Quick scalar checks for a training step.
    """
    return {
        "logits_abs_max": float(np.max(np.abs(logits))),
        "probs_min": float(np.min(probs)),
        "probs_max": float(np.max(probs)),
        "grad_norm": float(np.linalg.norm(grad)),
        "has_nan_logits": float(np.isnan(logits).any()),
        "has_nan_probs": float(np.isnan(probs).any()),
        "has_nan_grad": float(np.isnan(grad).any()),
    }


def leakage_check_same_rows(X_train: np.ndarray, X_test: np.ndarray) -> int:
    """
    Count exact duplicate rows across train and test.
    """
    train_rows = {tuple(row.tolist()) for row in X_train}
    return sum(tuple(row.tolist()) in train_rows for row in X_test)


if __name__ == "__main__":
    print("ML Debugging Patterns")
    print("=" * 60)

    y = np.array([0, 1, 1, 0, 1])
    print(f"Labels look binary: {check_binary_labels(y)}")

    grad = np.array([3.0, 4.0])
    print(f"Original grad norm: {np.linalg.norm(grad):.4f}")
    clipped = clip_gradients(grad, max_norm=2.0)
    print(f"Clipped grad: {clipped.round(4)}")
    print(f"Clipped grad norm: {np.linalg.norm(clipped):.4f}")

    logits = np.array([12.0, -11.0, 0.5])
    probs = np.array([0.9999, 0.0001, 0.5])
    print(inspect_training_step(logits, probs, grad))

    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8], [3, 4], [9, 1]])
    print(f"Duplicate rows across train/test: {leakage_check_same_rows(X_train, X_test)}")
