import math
import numpy as np


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("expected non-empty finite logits")
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def masked_cross_entropy(logits, labels, mask, epsilon=0.0):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must be (N,C), C > 0")
    n, c = logits.shape
    if labels.shape != (n,) or mask.shape != (n,):
        raise ValueError("labels and mask must have shape (N,)")
    if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= c)):
        raise ValueError("labels must be integer class indices")
    if not 0 <= epsilon < 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("invalid smoothing or binary mask")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    active = mask.astype(bool)
    if not active.any():
        return 0.0
    lp = log_softmax(logits[active])
    targets = labels[active]
    losses = -(1 - epsilon) * lp[np.arange(len(targets)), targets]
    losses -= epsilon * lp.mean(axis=-1)
    return float(losses.mean())


assert np.isclose(masked_cross_entropy(np.zeros((3,4)),[0,1,2],[1,1,1],.2),np.log(4))
assert masked_cross_entropy([[1000,-1000]],[1],[1],0)==2000
assert masked_cross_entropy([[0,1000]],[0],[0],.1)==0
assert np.isclose(masked_cross_entropy([[2,0],[0,2]],[0,0],[1,0],.2),np.log1p(np.exp(-2))+.2)

print('W008: all exact-contract behavior checks passed')
