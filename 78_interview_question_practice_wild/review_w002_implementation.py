import math
import numpy as np


def attention_qkv(q, k, v, mask=None):
    """Single-head attention. True mask entries mean allowed. Empty rows rejected."""
    q, k, v = [np.asarray(a, dtype=np.float64) for a in (q, k, v)]
    if any(a.ndim != 2 for a in (q, k, v)):
        raise ValueError("q,k,v must be matrices")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("incompatible attention shapes")
    if q.shape[1] == 0 or k.shape[0] == 0:
        raise ValueError("empty head dimension or key sequence")
    if not all(np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("inputs must be finite")
    scores = q @ k.T / math.sqrt(q.shape[1])
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != scores.shape or not mask.any(axis=-1).all():
            raise ValueError("invalid mask or all-masked row")
        scores = np.where(mask, scores, -np.inf)
    z = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(z)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def attention(x, wq, wk, wv, mask=None):
    x = np.asarray(x, dtype=np.float64)
    return attention_qkv(x @ wq, x @ wk, x @ wv, mask)


X=np.eye(2);W=np.eye(2);V=np.array([[1.,2.],[3.,4.]])
y=attention(X,W,W,V)
assert np.allclose(y,[[1.6604769,2.6604769],[2.3395231,3.3395231]])
assert np.allclose(attention(X,W,W,V,np.eye(2,dtype=bool)),V)
causal=np.tril(np.ones((2,2),dtype=bool));changed=X.copy();changed[1]*=100
assert np.allclose(attention(X,W,W,V,causal)[0],attention(changed,W,W,V,causal)[0])
assert np.isfinite(attention(np.eye(2)*1000,W,W,V)).all()

print('W002: all exact-contract behavior checks passed')
