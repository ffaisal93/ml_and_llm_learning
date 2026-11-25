import math
import numpy as np


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("expected non-empty finite logits")
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis=axis))


def sample_top_k(logits, k, rng=None, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or not 1 <= k <= len(x) or temperature <= 0:
        raise ValueError("invalid logits, k or temperature")
    if not np.isfinite(x).all():
        raise ValueError("logits must be finite")
    rng = np.random.default_rng() if rng is None else rng
    # Stable descending order makes equal-logit behavior explicit.
    ids = np.argsort(-x, kind="stable")[:k]
    p = softmax(x[ids] / temperature)
    return int(rng.choice(ids, p=p))


rng=np.random.default_rng(7)
draws=[sample_top_k([1.,2.,.5,4.],2,rng=rng) for _ in range(5000)]
assert set(draws)=={1,3} and abs(draws.count(3)/len(draws)-1/(1+np.exp(-2)))<.02
assert sample_top_k([3.,3.,1.],1,rng=rng)==0
assert set(sample_top_k([0.,0.,0.],3,rng=rng) for _ in range(100))=={0,1,2}

print('W010: all exact-contract behavior checks passed')
