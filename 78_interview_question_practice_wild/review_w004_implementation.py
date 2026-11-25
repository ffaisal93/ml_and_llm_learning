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


def grouped_query_attention(q, k, v):
    """Reference loops preserve the contiguous head mapping without repeating KV."""
    q, k, v = [np.asarray(a, dtype=np.float64) for a in (q, k, v)]
    if any(a.ndim != 4 for a in (q, k, v)):
        raise ValueError("expected (B,H,T,D) tensors")
    b, hq, tq, d = q.shape
    if k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d:
        raise ValueError("incompatible KV tensors")
    hkv = k.shape[1]
    if hkv < 1 or hq < 1 or hq % hkv:
        raise ValueError("Hq must be divisible by positive Hkv")
    out = np.empty_like(q)
    group = hq // hkv
    for batch in range(b):
        for h in range(hq):
            kh = h // group
            out[batch, h] = attention_qkv(q[batch, h], k[batch, kh], v[batch, kh])
    return out


q=np.zeros((1,4,2,1));k=np.zeros((1,2,3,1));v=np.array([[[[2.],[2.],[2.]],[[9.],[9.],[9.]]]])
y=grouped_query_attention(q,k,v)
assert y.shape==(1,4,2,1) and np.all(y[0,:2]==2) and np.all(y[0,2:]==9)
rng=np.random.default_rng(7);q=rng.normal(size=(2,6,3,4));k=rng.normal(size=(2,2,5,4));v=rng.normal(size=k.shape)
kr=np.repeat(k,3,axis=1);vr=np.repeat(v,3,axis=1);scores=q@kr.swapaxes(-1,-2)/2
p=np.exp(scores-scores.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
assert np.allclose(grouped_query_attention(q,k,v),p@vr)

print('W004: all exact-contract behavior checks passed')
