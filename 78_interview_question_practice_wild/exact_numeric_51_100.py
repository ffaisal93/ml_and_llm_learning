"""Exact NumPy contracts for W072, W076 and W079."""

import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered**2).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(variance + eps) * gamma + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def patch_embed(images, w, b, patch_size):
    B, C, H, W = images.shape
    P = patch_size
    if P <= 0 or H % P or W % P:
        raise ValueError("Patch size must divide both image dimensions")
    # B,C,patch-row,pixel-row,patch-col,pixel-col
    patches = images.reshape(B, C, H // P, P, W // P, P)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)  # B,grid-H,grid-W,C,P,P
    flat = patches.reshape(B, (H // P) * (W // P), C * P * P)
    return flat @ w + b


def multi_head_attention(x, w_qkv, b_qkv, w_out, b_out, num_heads):
    B, N, D = x.shape
    if num_heads <= 0 or D % num_heads:
        raise ValueError("Heads must divide embedding dimension")
    h, d = num_heads, D // num_heads
    qkv = x @ w_qkv + b_qkv
    q, k, v = [
        part.reshape(B, N, h, d).transpose(0, 2, 1, 3)
        for part in np.split(qkv, 3, axis=-1)
    ]
    weights = softmax((q @ k.swapaxes(-1, -2)) / np.sqrt(d), axis=-1)
    combined = (weights @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
    return combined @ w_out + b_out


def encoder_block(x, p, num_heads):
    h = layer_norm(x, p["ln1_g"], p["ln1_b"])
    x = x + multi_head_attention(
        h, p["qkv_w"], p["qkv_b"], p["proj_w"], p["proj_b"], num_heads
    )
    h = layer_norm(x, p["ln2_g"], p["ln2_b"])
    return x + gelu(h @ p["fc1_w"] + p["fc1_b"]) @ p["fc2_w"] + p["fc2_b"]


def vit_forward(images, params, patch_size, num_heads):
    tokens = patch_embed(images, params["patch_w"], params["patch_b"], patch_size)
    B, N, D = tokens.shape
    cls = np.broadcast_to(params["cls"], (B, 1, D))
    x = np.concatenate([cls, tokens], axis=1) + params["pos"]
    for block in params["blocks"]:
        x = encoder_block(x, block, num_heads)
    x = layer_norm(x, params["ln_g"], params["ln_b"])
    return x[:, 0] @ params["head_w"] + params["head_b"]


def ace_calc(logprobs, accs, m_bins=15):
    logprobs, accs = np.asarray(logprobs), np.asarray(accs)
    if logprobs.ndim != 1 or accs.shape != logprobs.shape or not 1 <= m_bins <= 1000:
        raise ValueError("Expected matching vectors and valid bin count")
    if (
        np.any(logprobs > 0)
        or np.any(~np.isfinite(logprobs))
        or not np.all((accs == 0) | (accs == 1))
    ):
        raise ValueError("Expected finite log confidences <=0 and binary correctness")
    if logprobs.size == 0:
        return 0.0
    confidences = np.exp(logprobs)
    bins = np.minimum((confidences * m_bins).astype(int), m_bins - 1)
    counts = np.bincount(bins, minlength=m_bins)
    conf_sum = np.bincount(bins, weights=confidences, minlength=m_bins)
    acc_sum = np.bincount(bins, weights=accs, minlength=m_bins)
    occupied = counts > 0
    return float(
        np.mean(np.abs(acc_sum[occupied] - conf_sum[occupied]) / counts[occupied])
    )


class MLP:
    def __init__(self, input_dim=10, h1=64, h2=32, output_dim=1, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.standard_normal((input_dim, h1)) * 0.01,
            "b1": np.zeros(h1),
            "W2": rng.standard_normal((h1, h2)) * 0.01,
            "b2": np.zeros(h2),
            "W3": rng.standard_normal((h2, output_dim)) * 0.01,
            "b3": np.zeros(output_dim),
        }

    def _forward(self, x):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        return a2 @ p["W3"] + p["b3"], (z1, a1, z2, a2)

    def forward(self, x):
        return self._forward(x)[0]

    def loss(self, x, y):
        prediction = self.forward(x)
        if prediction.shape != y.shape or y.size == 0:
            raise ValueError("Targets must match nonempty predictions")
        return float(np.mean((prediction - y) ** 2))

    def backward(self, x, y):
        p = self.params
        pred, (z1, a1, z2, a2) = self._forward(x)
        if y.shape != pred.shape or y.size == 0:
            raise ValueError("Targets must match nonempty predictions")
        d3 = 2 * (pred - y) / y.size
        dz2 = (d3 @ p["W3"].T) * (z2 > 0)
        dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
        return {
            "W3": a2.T @ d3,
            "b3": d3.sum(axis=0),
            "W2": a1.T @ dz2,
            "b2": dz2.sum(axis=0),
            "W1": x.T @ dz1,
            "b1": dz1.sum(axis=0),
        }

    def step(self, grads, lr):
        for name in self.params:
            self.params[name] = self.params[name] - lr * grads[name]
