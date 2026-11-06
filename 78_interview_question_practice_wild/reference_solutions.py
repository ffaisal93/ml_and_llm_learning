"""Small, readable interview reference implementations; Python 3.9+ and NumPy.

These are teaching implementations, not production kernels. Finite inputs are
assumed unless a function explicitly supports negative-infinity masks.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np


# ANCHOR: probabilities
def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("expected non-empty finite logits")
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis=axis))


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
# ANCHOR_END: probabilities


# ANCHOR: attention
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


def cached_attention_step(q_new, k_new, v_new, cache=None):
    """One-token, one-head teaching cache. q/k are already position-adjusted.

    This tests attention equivalence, not a complete multi-layer GPT. Appending
    with concatenate copies memory; production caches preallocate or use blocks.
    """
    if len(q_new) != 1 or len(k_new) != 1 or len(v_new) != 1:
        raise ValueError("this reference handles one new token")
    if cache is None:
        keys, values = np.array(k_new, copy=True), np.array(v_new, copy=True)
    else:
        keys = np.concatenate([cache[0], k_new], axis=0)
        values = np.concatenate([cache[1], v_new], axis=0)
    # The current query may see the entire prefix, including its current key.
    return attention_qkv(q_new, keys, values), (keys, values)
# ANCHOR_END: attention


# ANCHOR: sampling
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


def nucleus_candidates(logits, p, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or not 0 < p <= 1 or temperature <= 0:
        raise ValueError("invalid logits, p or temperature")
    probs = softmax(x / temperature)
    order = np.argsort(-probs, kind="stable")
    count = min(len(order), int(np.searchsorted(np.cumsum(probs[order]), p)) + 1)
    ids = order[:max(1, count)]
    selected = probs[ids]
    return ids, selected / selected.sum()


def sample_top_p(logits, p, rng=None, temperature=1.0):
    rng = np.random.default_rng() if rng is None else rng
    ids, probs = nucleus_candidates(logits, p, temperature)
    return int(rng.choice(ids, p=probs))


def beam_search(next_log_probs, start, eos, beam_width, max_new_tokens):
    """No length normalization. Callback returns a finite 1D log-probability array.

    Finished hypotheses remain in the beam with unchanged scores. This is a
    readable baseline; width pruning is approximate search, not exhaustive MAP.
    """
    if beam_width < 1 or max_new_tokens < 0:
        raise ValueError("invalid search limits")
    beams = [(tuple(start), 0.0, False)]
    for _ in range(max_new_tokens):
        candidates = []
        for tokens, score, done in beams:
            if done:
                candidates.append((tokens, score, True))
                continue
            lp = np.asarray(next_log_probs(tokens), dtype=float)
            if lp.ndim != 1 or not len(lp) or not np.isfinite(lp).all():
                raise ValueError("callback must return finite log probabilities")
            for token in np.argsort(-lp, kind="stable")[:beam_width]:
                candidates.append((tokens + (int(token),), score + float(lp[token]), token == eos))
        beams = sorted(candidates, key=lambda x: (-x[1], x[0]))[:beam_width]
        if all(done for _, _, done in beams):
            break
    best = max(beams, key=lambda x: x[1])
    return list(best[0]), best[1]
# ANCHOR_END: sampling


# ANCHOR: entropy
def compute_entropy_blockwise(logits, block_size):
    """Two scalar scans; supports len/index-only lazy inputs and negative infinity."""
    n = len(logits)
    if n == 0 or block_size <= 0:
        raise ValueError("empty input or non-positive block size")
    maximum = -math.inf
    for start in range(0, n, block_size):
        for i in range(start, min(start + block_size, n)):
            x = float(logits[i])
            if math.isnan(x) or x == math.inf:
                raise ValueError("invalid logit")
            maximum = max(maximum, x)
    if maximum == -math.inf:
        raise ValueError("at least one logit must be finite")
    z_sum, weighted_shift = 0.0, 0.0
    for start in range(0, n, block_size):
        for i in range(start, min(start + block_size, n)):
            x = float(logits[i])
            if x == -math.inf:
                continue
            shift = x - maximum
            weight = math.exp(shift)
            z_sum += weight
            # Avoid 0 * -inf after subtraction overflows for extreme finite inputs.
            if weight:
                weighted_shift += weight * shift
    return math.log(z_sum) - weighted_shift / z_sum
# ANCHOR_END: entropy


# ANCHOR: layernorm
def layernorm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma, beta = np.asarray(gamma), np.asarray(beta)
    if x.ndim != 2 or gamma.shape != x.shape[1:] or beta.shape != gamma.shape or eps <= 0:
        raise ValueError("invalid shapes or epsilon")
    centered = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    normalized = centered * inv
    return normalized * gamma + beta, (normalized, inv, gamma)


def layernorm_backward(dout, cache):
    normalized, inv, gamma = cache
    dout = np.asarray(dout)
    g = dout * gamma
    dx = inv * (g - g.mean(axis=-1, keepdims=True)
                - normalized * (g * normalized).mean(axis=-1, keepdims=True))
    return dx, (dout * normalized).sum(axis=0), dout.sum(axis=0)
# ANCHOR_END: layernorm


# ANCHOR: mlp
def mlp_loss_and_gradients(x, y, params):
    """Three affine layers, two ReLUs; MSE averaged over all target elements."""
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(z2, 0)
    prediction = a2 @ w3 + b3
    residual = prediction - y
    loss = float(np.mean(residual ** 2))
    g3 = 2 * residual / residual.size
    dw3, db3 = a2.T @ g3, g3.sum(axis=0)
    g2 = (g3 @ w3.T) * (z2 > 0)
    dw2, db2 = a1.T @ g2, g2.sum(axis=0)
    g1 = (g2 @ w2.T) * (z1 > 0)
    dw1, db1 = x.T @ g1, g1.sum(axis=0)
    return loss, [dw1, db1, dw2, db2, dw3, db3]
# ANCHOR_END: mlp


# ANCHOR: clustering
def kmeans(x, k, seed=0, max_iter=100, tol=1e-6):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not 1 <= k <= len(x) or max_iter < 1 or tol < 0:
        raise ValueError("invalid data or k-means configuration")
    if not np.isfinite(x).all():
        raise ValueError("data must be finite")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members):
                updated[j] = members.mean(axis=0)
        shift = np.linalg.norm(updated - centers)
        centers = updated
        if shift <= tol:
            break
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = distances.argmin(axis=1)
    return centers, labels


def knn_predict(x_train, y_train, query, k):
    x_train, query = np.asarray(x_train), np.asarray(query)
    y_train = np.asarray(y_train)
    if x_train.ndim != 2 or query.shape != x_train.shape[1:] or not 1 <= k <= len(x_train):
        raise ValueError("invalid KNN shapes or k")
    if y_train.shape != (len(x_train),):
        raise ValueError("invalid label shape")
    ids = np.argsort(((x_train - query) ** 2).sum(axis=1), kind="stable")[:k]
    labels, counts = np.unique(y_train[ids], return_counts=True)
    # np.unique sorts labels; equal votes choose the smallest comparable label.
    return labels[counts.argmax()]
# ANCHOR_END: clustering


# ANCHOR: metrics
def ace_calc(logprobs, accs, m_bins=15):
    lp, acc = np.asarray(logprobs), np.asarray(accs)
    if lp.ndim != 1 or lp.shape != acc.shape or m_bins < 1:
        raise ValueError("invalid calibration inputs")
    if np.isnan(lp).any() or (lp > 0).any() or not np.isin(acc, [0, 1]).all():
        raise ValueError("invalid confidence or correctness")
    if len(lp) == 0:
        return 0.0
    confidence = np.exp(lp)
    bins = np.minimum((confidence * m_bins).astype(int), m_bins - 1)
    counts = np.bincount(bins, minlength=m_bins)
    good = counts > 0
    bin_conf = np.bincount(bins, weights=confidence, minlength=m_bins)[good] / counts[good]
    bin_acc = np.bincount(bins, weights=acc, minlength=m_bins)[good] / counts[good]
    return float(np.abs(bin_conf - bin_acc).mean())


def binary_focal_loss(logits, labels, gamma=2.0, alpha=None):
    z, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    if z.shape != y.shape or not z.size or not np.isin(y, [0, 1]).all() or gamma < 0:
        raise ValueError("invalid focal-loss inputs")
    if not np.isfinite(z).all() or (alpha is not None and not 0 <= alpha <= 1):
        raise ValueError("invalid logits or alpha")
    bce = np.logaddexp(0, z) - y * z
    modulation = (-np.expm1(-bce)) ** gamma
    weights = 1 if alpha is None else np.where(y == 1, alpha, 1 - alpha)
    return float(np.mean(weights * modulation * bce))


def info_nce_loss(q, k, tau=0.07):
    """NumPy forward reference only; use equivalent differentiable tensor ops to train."""
    q, k = np.asarray(q, dtype=float), np.asarray(k, dtype=float)
    if q.ndim != 2 or q.shape != k.shape or not len(q) or tau <= 0:
        raise ValueError("invalid contrastive inputs")
    qn, kn = np.linalg.norm(q, axis=1, keepdims=True), np.linalg.norm(k, axis=1, keepdims=True)
    if (qn == 0).any() or (kn == 0).any():
        raise ValueError("zero-norm embedding")
    logits = (q / qn) @ (k / kn).T / tau
    ids = np.arange(len(q))
    return float(-0.5 * (log_softmax(logits)[ids, ids].mean()
                         + log_softmax(logits.T)[ids, ids].mean()))
# ANCHOR_END: metrics


# ANCHOR: batching
class DynamicBatcher:
    """Pure scheduling core; no threads, clock or production retry policy."""
    def __init__(self, model, b_max, t_max):
        if b_max < 1 or t_max < 0:
            raise ValueError("invalid batching limits")
        self.model, self.b_max, self.t_max = model, b_max, t_max
        self.queue, self.next_id = deque(), 0

    def submit(self, x, now):
        request_id = self.next_id
        self.next_id += 1
        self.queue.append((request_id, x, now))
        return request_id

    def next_deadline(self):
        return None if not self.queue else self.queue[0][2] + self.t_max

    def should_dispatch(self, now):
        return bool(self.queue) and (len(self.queue) >= self.b_max or now >= self.next_deadline())

    def dispatch(self, now):
        if not self.should_dispatch(now):
            return []
        batch = [self.queue.popleft() for _ in range(min(self.b_max, len(self.queue)))]
        outputs = list(self.model([item[1] for item in batch]))
        if len(outputs) != len(batch):
            raise ValueError("model returned the wrong number of outputs")
        return [(item[0], result) for item, result in zip(batch, outputs)]
# ANCHOR_END: batching


# ANCHOR: packing
@dataclass
class PackedSample:
    input_ids: list
    loss_mask: list
    position_ids: list


def pack(examples, seq_len, sep, pad_id):
    """Stable first-fit decreasing; at most quadratic placement plus token copying.

    Position reset and loss masks alone DO NOT isolate examples in attention.
    This is a data-layout helper, not a full training collator.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    layouts = []
    for prompt, answer in examples:
        ids = list(prompt) + list(answer) + [sep]
        if len(ids) <= seq_len:
            layouts.append((ids, [0] * len(prompt) + [1] * (len(answer) + 1)))
    layouts.sort(key=lambda row: -len(row[0]))  # Python sort preserves input-order ties.
    samples = []
    for ids, mask in layouts:
        target = next((s for s in samples if len(s.input_ids) + len(ids) <= seq_len), None)
        if target is None:
            target = PackedSample([], [], [])
            samples.append(target)
        target.input_ids.extend(ids)
        target.loss_mask.extend(mask)
        target.position_ids.extend(range(len(ids)))
    for sample in samples:
        padding = seq_len - len(sample.input_ids)
        sample.input_ids.extend([pad_id] * padding)
        sample.loss_mask.extend([0] * padding)
        sample.position_ids.extend([0] * padding)
    return samples
# ANCHOR_END: packing
