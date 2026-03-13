"""
Optimization and matrix calculus boilerplate.
Interview-friendly code with direct formulas.
"""
from typing import Callable, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with max subtraction for numerical stability."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Average binary cross-entropy."""
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(loss))


def linear_regression_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> Tuple[np.ndarray, float]:
    """
    Gradients of average MSE loss for linear regression.
    """
    n = X.shape[0]
    preds = X @ w + b
    errors = preds - y
    grad_w = (X.T @ errors) * (2.0 / n)
    grad_b = float(2.0 * np.mean(errors))
    return grad_w, grad_b


def logistic_regression_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Gradients of average BCE loss for logistic regression.

    Key result:
        grad_w = X^T (p - y) / n
        grad_b = mean(p - y)
    """
    logits = X @ w + b
    probs = sigmoid(logits)
    errors = probs - y
    grad_w = (X.T @ errors) / X.shape[0]
    grad_b = float(np.mean(errors))
    return grad_w, grad_b, probs


def logistic_regression_step(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float = 0.1
) -> Tuple[np.ndarray, float, float]:
    """
    One gradient descent step for logistic regression.

    Returns:
        new_w, new_b, loss_before_step
    """
    grad_w, grad_b, probs = logistic_regression_gradients(X, y, w, b)
    loss = binary_cross_entropy(y, probs)
    w = w - lr * grad_w
    b = b - lr * grad_b
    return w, b, loss


def numerical_gradient_1d(f: Callable[[float], float], x: float, eps: float = 1e-5) -> float:
    """Central-difference numerical gradient for a scalar function."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def quadratic_objective(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """
    f(w) = 0.5 * w^T A w - b^T w
    """
    return float(0.5 * w.T @ A @ w - b.T @ w)


def quadratic_gradient(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gradient of 0.5 * w^T A w - b^T w."""
    return A @ w - b


def gradient_descent_quadratic(
    A: np.ndarray, b: np.ndarray, lr: float = 0.1, steps: int = 50
) -> np.ndarray:
    """Simple gradient descent on a quadratic objective."""
    w = np.zeros(A.shape[0])
    for _ in range(steps):
        w -= lr * quadratic_gradient(w, A, b)
    return w


def condition_number(A: np.ndarray) -> float:
    """
    Condition number based on singular values.
    Larger means worse conditioning.
    """
    singular_values = np.linalg.svd(A, compute_uv=False)
    return float(singular_values[0] / singular_values[-1])


if __name__ == "__main__":
    print("Optimization and Matrix Calculus Boilerplate")
    print("=" * 60)

    # Logistic regression demo
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [3.0, 3.5],
            [3.2, 4.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0])
    w = np.zeros(X.shape[1])
    b = 0.0

    print("Logistic Regression Steps")
    for step in range(5):
        w, b, loss = logistic_regression_step(X, y, w, b, lr=0.5)
        print(f"step={step:02d} loss={loss:.4f} w={w.round(4)} b={b:.4f}")

    # Numerical gradient check
    print("\nNumerical Gradient Check")
    f = lambda x: x**2 + 3.0 * x
    x0 = 2.0
    grad_numeric = numerical_gradient_1d(f, x0)
    grad_exact = 2.0 * x0 + 3.0
    print(f"numeric gradient: {grad_numeric:.6f}")
    print(f"exact gradient:   {grad_exact:.6f}")

    # Quadratic optimization
    print("\nQuadratic Optimization")
    A = np.array([[4.0, 0.0], [0.0, 1.0]])
    b_vec = np.array([8.0, 2.0])
    w_star = gradient_descent_quadratic(A, b_vec, lr=0.2, steps=50)
    print(f"Approximate optimum: {w_star.round(4)}")
    print(f"Condition number of A: {condition_number(A):.4f}")

    # Stable softmax
    print("\nStable Softmax")
    logits = np.array([1000.0, 1001.0, 999.0])
    print(stable_softmax(logits).round(6))
