import math
import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma, beta = np.asarray(gamma), np.asarray(beta)
    if (
        x.ndim != 2
        or gamma.shape != x.shape[1:]
        or beta.shape != gamma.shape
        or eps <= 0
    ):
        raise ValueError("invalid shapes or epsilon")
    centered = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    normalized = centered * inv
    return normalized * gamma + beta, (normalized, inv, gamma)


def layernorm_backward(dout, cache):
    normalized, inv, gamma = cache
    dout = np.asarray(dout)
    g = dout * gamma
    dx = inv * (
        g
        - g.mean(axis=-1, keepdims=True)
        - normalized * (g * normalized).mean(axis=-1, keepdims=True)
    )
    return dx, (dout * normalized).sum(axis=0), dout.sum(axis=0)


rng = np.random.default_rng(2)
x = rng.normal(size=(2, 4))
gamma = rng.normal(size=4)
beta = rng.normal(size=4)
g = rng.normal(size=x.shape)
y, cache = layernorm_forward(x, gamma, beta)
grads = layernorm_backward(g, cache)
for a, grad in zip([x, gamma, beta], grads):
    for index in np.ndindex(a.shape):
        old = a[index]
        a[index] = old + 1e-6
        plus = (layernorm_forward(x, gamma, beta)[0] * g).sum()
        a[index] = old - 1e-6
        minus = (layernorm_forward(x, gamma, beta)[0] * g).sum()
        a[index] = old
        assert abs((plus - minus) / 2e-6 - grad[index]) < 1e-5
assert np.allclose(layernorm_forward(x + 1e8, gamma, beta)[0], y, atol=1e-7)
assert np.allclose(
    layernorm_forward(np.ones((2, 4)), gamma, beta)[0], np.broadcast_to(beta, (2, 4))
)
assert np.allclose(
    layernorm_backward(
        np.ones((1, 4)), layernorm_forward([[1, 2, 3, 4]], np.ones(4), np.zeros(4))[1]
    )[0],
    0,
    atol=1e-12,
)

print("W007: all exact-contract behavior checks passed")
