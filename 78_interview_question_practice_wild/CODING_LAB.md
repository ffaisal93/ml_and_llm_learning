# Coding lab

Write your own solution first. These are small readable references that prioritize an explicit contract over peak performance. They are not complete production services or optimized GPU kernels. The NumPy contrastive loss is a forward numerical reference; an actual training implementation needs equivalent differentiable tensor operations.

## Run the checks

From the repository root, with Python 3.9+ and NumPy already available:

```bash
python3 -m unittest discover -s 78_interview_question_practice_wild -p 'test_*.py' -v
```

The 22 numerical checks cover independent worked examples, causality, cache-prefix equivalence, GQA head mapping, finite-difference gradients for LayerNorm and the MLP, lazy-input entropy, empty clusters, calibration conventions, sampling support, batch deadlines, request/result mapping and packed-example layout. There are also five executable [SQL checks](SQL_LAB.md), for 27 checks in the initial labs. The additional algorithms and bank-coverage checks add more tests. These checks do not establish production performance or benchmark parity.

The implementations live in [reference_solutions.py](reference_solutions.py); the runnable checks are in [test_reference_solutions.py](test_reference_solutions.py). The snippets below are the same code, so you can also read them offline in this page.

## Practice map

| Questions | Implementation | Main invariant |
|---|---|---|
| F022, F041 | Stable probabilities and masked loss | Shift invariance; loss averages included examples only |
| F011–F013 | Attention, GQA and caching | Correct key axis, head mapping and prefix equivalence |
| F014–F016 | Sampling and beam search | Correct vocabulary ids, boundary token and EOS behavior |
| F017 | Blockwise entropy | No slices, iteration or growing temporary buffers |
| F042, F107 | LayerNorm | Centered variance; analytic gradients match finite differences |
| F043 | MLP | Chain rule and one consistent mean-loss reduction |
| F044–F045 | Clustering and KNN | Returned labels match returned centers; ties are specified |
| F047–F049 | Contrastive loss, focal loss and calibration | Objective and metric conventions are explicit |
| F032, F040 | Dynamic batch scheduling | One model call per batch; FIFO ids and inclusive deadline |
| F023 | SFT packing | Stable placement, correct masks and position resets |

## Stable probabilities and masked loss

Time is O(N*C) and intermediate array space is O(N*C) for the loss. This is a straightforward teaching implementation, not an in-place or fused kernel. The mask selects rows; labels still follow the declared valid-class contract.

```python
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
```

## Attention, shared KV heads and a toy cache

Single-head attention takes O(Tq*Tk*d) arithmetic and materializes O(Tq*Tk) scores. The GQA loops make sharing explicit. The one-head cache example requires already position-adjusted Q/K and proves attention equivalence only; a full GPT also needs per-layer caches, output projections, normalization and generation logic. Concatenating on each step copies the cache and is deliberately not a production allocation strategy.

```python
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
```

## Sampling and beam search

The reference top-k and top-p code fully sorts the vocabulary, costing O(V log V). Partial selection can improve top-k selection to expected O(V), with extra work to define ties. Beam search uses raw summed log probabilities, no length normalization, and retains finished beams without extending them. Callbacks return finite log probabilities; broader masking and constraint handling are outside this minimal reference.

```python
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
```

## Constant-memory blockwise entropy

The helper makes two indexed scans and keeps scalar statistics only: O(N) time and O(1) auxiliary memory. Block size bounds each inner scan; no block copies are made. Negative infinity has zero probability and is explicitly skipped in the weighted sum.

```python
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
```

## LayerNorm forward and backward

The reference uses float64 and population variance over each row. Forward and backward take O(N*D) time and array space. The cache stores normalized values, inverse standard deviations and gamma; do not mutate cached arrays before backward. Finite differences check x, gamma and beta.

```python
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
```

## Two-hidden-layer MLP gradients

This is an ordinary ReLU regression network with MSE averaged over all target elements, not just rows. All gradients are computed before any parameter update. Matrix multiplications dominate runtime; stored intermediate activations determine training memory. The test avoids ReLU nondifferentiable points.

```python
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
```

## K-means and KNN

K-means retains the old center for an empty cluster and recomputes final labels. Its direct distance tensor uses O(N*K*D) temporary space, which should be blocked for large data. KNN sorts all N distances: O(N*D+N log N) per query and O(N) distance/selection storage. Labels must be comparable for the stated smallest-label tie rule.

```python
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
```

## Calibration and numerical losses

ACE averages non-empty bin gaps equally rather than weighting by bin population. Focal loss at gamma=0 reduces to BCE, with optional alpha weighting. InfoNCE normalizes nonzero embeddings and averages the two retrieval directions; the NumPy function cannot be used directly with autograd.

```python
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
```

## Deterministic dynamic batcher

Submit and deadline checks are O(1); dispatch is O(batch size) excluding model cost. Storage is O(queued requests). now must be nondecreasing under the exercise contract. This core removes items before calling the model: a production version needs a durable in-flight state and explicit failure/retry completion policy so exceptions do not lose work.

```python
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
```

## Stable first-fit SFT packing

The stable descending sort costs O(E log E). Searching existing bins can cost O(E²) in the worst case; copying and padding cost the size of the returned sequences. Oversize examples are skipped by this exercise contract. Position resets and loss masks do not isolate attention: use segment-aware masking or separate examples when required, and align the target mask after shifting.

```python
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
```

## What to change after you can write the baseline

Change one contract at a time: add multi-head masks, variable-length caches, tied scores, duplicate positives or a model-failure path. State the new behavior before editing code. The smallest counterexample usually teaches more than another large random batch.
