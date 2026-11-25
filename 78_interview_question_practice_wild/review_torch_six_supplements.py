"""Optional PyTorch studies for W002/W003/W004/W007/W008/W147.

Primary NumPy/manual submissions remain unchanged. No library attention modules.
Tensors use a shared floating dtype/device (float64 for the numerical checks).
"""

import math
import torch


def attention_torch(X, Wq, Wk, Wv, mask=None):
    """W002: (n,dm) and projection weights -> (n,dv); True means allowed."""
    if X.ndim != 2 or any((W.ndim != 2 for W in (Wq, Wk, Wv))):
        raise ValueError("X and weights must be matrices")
    n, dm = X.shape
    if (
        n < 1
        or any((W.shape[0] != dm for W in (Wq, Wk, Wv)))
        or Wq.shape[1] != Wk.shape[1]
        or (min(Wq.shape[1], Wv.shape[1]) < 1)
    ):
        raise ValueError("incompatible projection shapes")
    q, k, v = (X @ Wq, X @ Wk, X @ Wv)
    scores = q @ k.T / math.sqrt(q.shape[1])
    if mask is not None:
        allowed = torch.as_tensor(mask, dtype=torch.bool, device=X.device)
        if allowed.shape != (n, n) or not bool(allowed.any(dim=-1).all()):
            raise ValueError("mask must be (n,n) with a key in every row")
        scores = scores.masked_fill(~allowed, -torch.inf)
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    weights = shifted.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ v


def multi_head_attention_torch(Q, K, V, num_heads, mask=None):
    """W003: consecutive supplied features, no learned/output projections."""
    if Q.ndim != 3 or Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError("matching (B,T,D) tensors required")
    B, T, D = Q.shape
    if min(B, T, D) < 1 or type(num_heads) is not int or num_heads < 1 or D % num_heads:
        raise ValueError("positive heads must divide D")
    d = D // num_heads
    q, k, v = [a.reshape(B, T, num_heads, d).transpose(1, 2) for a in (Q, K, V)]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    if mask is not None:
        allowed = torch.as_tensor(mask, dtype=torch.bool, device=Q.device)
        try:
            allowed = torch.broadcast_to(allowed, scores.shape)
        except RuntimeError as exc:
            raise ValueError("mask cannot broadcast to (B,H,T,T)") from exc
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("all-masked query")
        scores = scores.masked_fill(~allowed, -torch.inf)
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    probabilities = shifted.exp()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return (probabilities @ v).transpose(1, 2).reshape(B, T, D)


def grouped_query_attention_torch(Q, K, V):
    """W004: contiguous groups share KV; no masks and no repeated KV storage."""
    if any((a.ndim != 4 for a in (Q, K, V))):
        raise ValueError("expected (B,H,T,D) tensors")
    B, Hq, Tq, D = Q.shape
    if K.shape != V.shape or K.shape[0] != B or K.shape[-1] != D:
        raise ValueError("incompatible KV tensors")
    Hkv, Tk = K.shape[1:3]
    if min(B, Hq, Tq, D, Hkv, Tk) < 1 or Hq % Hkv:
        raise ValueError("positive dimensions; Hq divisible by Hkv")
    group = Hq // Hkv
    grouped_q = Q.reshape(B, Hkv, group, Tq, D)
    shared_k = K.unsqueeze(2)
    shared_v = V.unsqueeze(2)
    scores = grouped_q @ shared_k.transpose(-1, -2) / math.sqrt(D)
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    weights = shifted.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (weights @ shared_v).reshape(B, Hq, Tq, D)


def layernorm_forward_torch(x, gamma, beta, eps=1e-05):
    """W007: feature mean/biased variance, epsilon inside square root."""
    if (
        x.ndim != 2
        or gamma.shape != x.shape[1:]
        or beta.shape != gamma.shape
        or (eps <= 0)
    ):
        raise ValueError("invalid shapes or epsilon")
    centered = x - x.mean(dim=-1, keepdim=True)
    inv_std = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    xhat = centered * inv_std
    return (xhat * gamma + beta, (xhat, inv_std, gamma))


def layernorm_backward_torch(dout, cache):
    """Same manual derivative as NumPy, expressed with tensor operations."""
    xhat, inv_std, gamma = cache
    g = dout * gamma
    dx = inv_std * (
        g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True)
    )
    return (dx, (dout * xhat).sum(dim=0), dout.sum(dim=0))


def masked_cross_entropy_torch(logits, labels, mask, epsilon=0.0):
    """W008: uniform-over-all-C smoothing and mean over selected examples."""
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("logits must be (N,C) with C>0")
    N, C = logits.shape
    labels = torch.as_tensor(labels, device=logits.device)
    mask = torch.as_tensor(mask, device=logits.device)
    if labels.shape != (N,) or mask.shape != (N,):
        raise ValueError("labels and mask must have shape (N,)")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if labels.dtype not in integer_types or bool(((labels < 0) | (labels >= C)).any()):
        raise ValueError("integer class indices required")
    if not 0 <= epsilon < 1 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("invalid epsilon or binary mask")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("finite logits required")
    active = mask.bool()
    if not bool(active.any()):
        return (logits * 0).sum()
    z = logits[active]
    shifted = z - z.amax(dim=-1, keepdim=True)
    logp = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    target = labels[active].long()
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    per_example = (1 - epsilon) * nll - epsilon * logp.mean(dim=-1)
    return per_example.mean()


def sigmoid_torch(z):
    """W147: stable sigmoid; derivative takes logits, not probabilities."""
    return torch.sigmoid(z)


def sigmoid_derivative_torch(z):
    s = sigmoid_torch(z)
    return s * (1 - s)


def dense_layer_forward_torch(X, W, b):
    """W147: X:(N,D), W:(D,C), b:(C,); return sigmoid(X@W+b)."""
    return sigmoid_torch(X @ W + b)


def dense_layer_backward_torch(dY, X, W, b):
    dZ = dY * sigmoid_derivative_torch(X @ W + b)
    return (dZ @ W.T, X.T @ dZ, dZ.sum(dim=0))
