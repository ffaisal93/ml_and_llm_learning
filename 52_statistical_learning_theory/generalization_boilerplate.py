"""
Small utilities for statistical learning theory intuition.
"""
from typing import Callable, Tuple

import numpy as np


def empirical_risk(y_true: np.ndarray, y_pred: np.ndarray, loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """Average loss on observed data."""
    return float(np.mean(loss_fn(y_true, y_pred)))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Elementwise squared error."""
    return (y_true - y_pred) ** 2


def generalization_gap(train_loss: float, test_loss: float) -> float:
    """Difference between test and train performance."""
    return float(test_loss - train_loss)


def train_validation_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple shuffled train/validation split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1.0 - val_fraction))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def l2_regularized_linear_objective(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_l2: float
) -> float:
    """
    Average MSE + L2 penalty.
    """
    preds = X @ w + b
    mse = np.mean((preds - y) ** 2)
    penalty = lambda_l2 * np.sum(w**2)
    return float(mse + penalty)


def hoeffding_confidence_radius(n: int, delta: float, value_range: float = 1.0) -> float:
    """
    Hoeffding-style radius for averages of bounded variables.

    If losses are in [a, b], use value_range = b - a.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return float(value_range * np.sqrt(np.log(2.0 / delta) / (2.0 * n)))


if __name__ == "__main__":
    print("Statistical Learning Theory Boilerplate")
    print("=" * 60)

    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    yhat_train = np.array([1.1, 2.1, 2.9, 3.8])
    y_test = np.array([1.5, 2.5, 3.5, 4.5])
    yhat_test = np.array([1.1, 2.0, 2.9, 3.6])

    train_loss = empirical_risk(y_train, yhat_train, mse_loss)
    test_loss = empirical_risk(y_test, yhat_test, mse_loss)
    print(f"Train loss: {train_loss:.4f}")
    print(f"Test loss:  {test_loss:.4f}")
    print(f"Generalization gap: {generalization_gap(train_loss, test_loss):.4f}")

    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
    y = np.array([1.0, 1.2, 2.5, 2.8, 4.1])
    X_train, X_val, y_train_split, y_val_split = train_validation_split(X, y, val_fraction=0.4)
    print(f"\nTrain split shape: {X_train.shape}, Validation split shape: {X_val.shape}")

    w = np.array([0.2, 0.4])
    b = 0.1
    obj = l2_regularized_linear_objective(X_train, y_train_split, w, b, lambda_l2=0.01)
    print(f"L2-regularized objective: {obj:.4f}")

    radius = hoeffding_confidence_radius(n=1000, delta=0.05, value_range=1.0)
    print(f"Hoeffding confidence radius (n=1000, delta=0.05): {radius:.4f}")
