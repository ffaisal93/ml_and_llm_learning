import numpy as np


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        shifted = z - np.max(z, axis=axis, keepdims=True)
        e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        shifted = z - np.max(z, axis=axis, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def validate(logits, y):
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("nonempty 2-D logits required")
    if y.shape != (len(logits),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("one integer label per row required")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("label outside class range")
    return logits, y


def cross_entropy(logits, y):
    logits, y = validate(logits, y)
    return float(-log_softmax(logits)[np.arange(len(y)), y].mean())


class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(size=(n_in, n_hidden)) * np.sqrt(2 / n_in),
            "b1": np.zeros(n_hidden, dtype=np.float64),
            "W2": rng.normal(size=(n_hidden, n_out)) / np.sqrt(n_hidden),
            "b2": np.zeros(n_out, dtype=np.float64),
        }

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)
        return h @ p["W2"] + p["b2"], (X, z1, h)

    def loss_and_grads(self, X, y):
        logits, (X, z1, h) = self.forward(X)
        loss = cross_entropy(logits, y)
        y = np.asarray(y)
        g = softmax(logits)
        g[np.arange(len(y)), y] -= 1
        g /= len(y)
        hidden_grad = (g @ self.params["W2"].T) * (z1 > 0)
        grads = {
            "W2": h.T @ g,
            "b2": g.sum(axis=0),
            "W1": X.T @ hidden_grad,
            "b1": hidden_grad.sum(axis=0),
        }
        return loss, grads

    def step(self, grads, lr):
        for name, value in self.params.items():
            value -= lr * grads[name]

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    def fit(self, X, y, lr=0.5, epochs=500):
        losses = []
        for _ in range(epochs):
            loss, grads = self.loss_and_grads(X, y)
            losses.append(loss)
            self.step(grads, lr)
        return losses


def check():
    assert np.allclose(
        softmax([1000, 1001, 1002]), [0.09003057, 0.24472847, 0.66524096]
    )
    assert np.allclose(softmax([-1000, -1000]), [0.5, 0.5])
    assert np.allclose(log_softmax([0, -1000]), [0, -1000])
    assert cross_entropy([[0, 1000]], [0]) == 1000
    assert np.allclose(softmax(np.ones((2, 3, 4)), axis=1).sum(axis=1), 1)
    for a, b in [
        ([], []),
        ([[1, 2]], [-1]),
        ([[1, 2]], [2]),
        ([[1, 2]], [0.5]),
        ([[1, 2]], []),
        ([1, 2], [0]),
    ]:
        try:
            cross_entropy(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    rng = np.random.default_rng(42)
    X, y = rng.normal(size=(4, 3)), np.array([0, 1, 0, 1])
    net = TwoLayerNet(3, 4, 2, seed=9)
    _, grads = net.loss_and_grads(X, y)
    eps, worst = 1e-6, 0.0
    for name, p in net.params.items():
        numeric = np.empty_like(p)
        for index in np.ndindex(p.shape):
            old = p[index]
            p[index] = old + eps
            plus = cross_entropy(net.forward(X)[0], y)
            p[index] = old - eps
            minus = cross_entropy(net.forward(X)[0], y)
            p[index] = old
            numeric[index] = (plus - minus) / (2 * eps)
        error = np.linalg.norm(numeric - grads[name]) / max(
            1e-12, np.linalg.norm(numeric) + np.linalg.norm(grads[name])
        )
        worst = max(worst, error)
        assert error < 1e-6, (name, error)
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    net = TwoLayerNet(2, 16, 2, seed=0)
    losses = net.fit(X, y)
    assert losses[-1] < 0.03 and np.array_equal(net.predict(X), y)
    print(
        f"W112: gradient relative error {worst:.2g}; XOR loss {losses[-1]:.4f}; predictions {net.predict(X)}"
    )


if __name__ == "__main__":
    check()
