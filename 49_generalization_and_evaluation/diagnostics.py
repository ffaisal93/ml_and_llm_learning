"""
Generalization, evaluation, and diagnostics boilerplate.
Small utilities for interview coding and experiment sanity checks.
"""
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Binary confusion matrix as counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall, and F1 for binary classification."""
    cm = confusion_matrix_binary(y_true, y_pred)
    total = max(len(y_true), 1)
    precision = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
    recall = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (cm["tp"] + cm["tn"]) / total
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected calibration error for binary probabilities.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < n_bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        avg_confidence = np.mean(y_prob[mask])
        avg_accuracy = np.mean(y_true[mask])
        ece += np.sum(mask) / n * abs(avg_accuracy - avg_confidence)

    return float(ece)


def bootstrap_metric_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a metric computed from y_true and y_pred.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        estimates.append(metric_fn(y_true[idx], y_pred[idx]))

    alpha = 1.0 - confidence
    lower = np.percentile(estimates, 100.0 * alpha / 2.0)
    upper = np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def slice_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, slices: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Accuracy for user-provided slices.

    slices maps slice names to boolean masks.
    """
    result = {}
    for name, mask in slices.items():
        if np.sum(mask) == 0:
            result[name] = float("nan")
        else:
            result[name] = float(np.mean(y_true[mask] == y_pred[mask]))
    return result


def ablation_deltas(scores: Dict[str, float], baseline_name: str) -> Dict[str, float]:
    """
    Compare every score to a named baseline.
    """
    baseline = scores[baseline_name]
    return {name: float(score - baseline) for name, score in scores.items()}


if __name__ == "__main__":
    print("Generalization and Evaluation Boilerplate")
    print("=" * 60)

    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.92, 0.40, 0.81, 0.76, 0.33, 0.22, 0.64, 0.55, 0.71, 0.10])
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix_binary(y_true, y_pred)
    metrics = classification_metrics(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)

    print(f"Confusion matrix: {cm}")
    print(f"Metrics: {metrics}")
    print(f"ECE: {ece:.4f}")

    acc_ci = bootstrap_metric_confidence_interval(
        y_true,
        y_pred,
        metric_fn=lambda a, b: float(np.mean(a == b)),
        n_bootstrap=2000,
    )
    print(f"Bootstrap CI for accuracy: {acc_ci}")

    lengths = np.array([20, 10, 50, 60, 8, 7, 45, 12, 55, 9])
    slices = {
        "short_inputs": lengths < 15,
        "long_inputs": lengths >= 15,
    }
    print(f"Slice accuracy: {slice_accuracy(y_true, y_pred, slices)}")

    scores = {
        "baseline": 0.821,
        "add_more_data": 0.834,
        "better_decoder": 0.829,
        "remove_regularizer": 0.809,
    }
    print(f"Ablation deltas: {ablation_deltas(scores, baseline_name='baseline')}")
