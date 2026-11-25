import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1 + ez)
    return out


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def dense_layer_forward(X, W, b):
    return sigmoid(X @ W + b)


def dense_layer_backward(dY, X, W, b):
    dZ = dY * sigmoid_derivative(X @ W + b)
    return dZ @ W.T, X.T @ dZ, dZ.sum(axis=0)


X = np.array([[0.5, 0.2]])
W = np.array([[0.3, 0.8], [0.5, 0.2]])
b = np.array([0.1, 0.4])
Y = dense_layer_forward(X, W, b)
assert np.allclose(Y, [[0.5866175789, 0.698465216]])
assert np.allclose(sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
rng = np.random.default_rng(2)
X = rng.normal(size=(3, 2))
W = rng.normal(size=(2, 4))
b = rng.normal(size=4)
G = rng.normal(size=(3, 4))
grads = dense_layer_backward(G, X, W, b)
eps = 1e-6
worst = 0.0
for value, grad in zip([X, W, b], grads):
    for index in np.ndindex(value.shape):
        old = value[index]
        value[index] = old + eps
        plus = (dense_layer_forward(X, W, b) * G).sum()
        value[index] = old - eps
        minus = (dense_layer_forward(X, W, b) * G).sum()
        value[index] = old
        error = abs((plus - minus) / (2 * eps) - grad[index])
        worst = max(worst, error)
        assert error < 1e-7
zero = dense_layer_backward(np.zeros_like(G), X, W, b)
assert all(np.all(g == 0) for g in zero)
print(f"W147: forward {Y}; all X/W/b finite differences passed, max error {worst:.2g}")
