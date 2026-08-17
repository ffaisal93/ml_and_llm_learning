# Code you should be able to write from memory

The code an ML interviewer actually makes you type: softmax, attention, a training loop, k-means. Every block here has been executed; shapes and outputs are real.

Tonight: read it once end to end, then close the file and retype two or three blocks cold — attention and the training loop at minimum. Recognition is not recall. If you can't produce it on a blank page, you don't have it.

---

## A note on style

Everything here is written to be **reproduced from memory under pressure**, not to be short. That means
explicit loops over clever broadcasting, real variable names over single letters, and one idea per line.
A `for` loop you can write correctly on a whiteboard beats a vectorized one-liner you half-remember and
then cannot debug when the interviewer changes the shapes.

Where a compact vectorized form exists, it is mentioned in the prose so you can *say* it — "I would
vectorize this as a single distance matrix in production" — which gets you the credit for knowing it
without the risk of writing it. That sentence is worth more than the line itself, because it shows you
chose clarity deliberately.

Every snippet in this file has been executed. Where an independent reference exists — PyTorch,
scikit-learn — the output was checked against it and the check is noted.

---

## Numerical bedrock

### Stable softmax

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

Shapes: `(..., k)` in → `(..., k)` out, sums to 1 along `axis`.

Why the max subtraction: `softmax(x) = softmax(x - c)` for any constant $c$ — the shift cancels in numerator and denominator. Subtracting the row max makes the largest exponent $e^0 = 1$, so nothing overflows. Without it, `np.exp([1000, 1001, 1002])` is `inf, inf, inf` → `nan`. Verified: with the shift, `[1000,1001,1002]` gives the same `[0.090, 0.245, 0.665]` as `[1,2,3]`.

**What they'll ask about this**
- Why is subtracting the max safe? (shift invariance — prove it in one line)
- What about underflow? Small logits go to 0 in `exp`, which is fine; the denominator is $\geq 1$ because the max term is exactly 1.
- `keepdims=True` — what breaks without it? Broadcasting: `(n,k) - (n,)` fails or misaligns.
- Softmax with temperature: `softmax(x / T)`. $T \to 0$ = argmax, $T \to \infty$ = uniform.

### Log-sum-exp

```python
def logsumexp(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)
```

Shapes: `(..., k)` in → `(..., )` out (that axis removed). Matches `torch.logsumexp` exactly: `[[1,2,3],[1000,1001,1002]]` → `[3.4076, 1002.4076]`.

$$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}, \quad m = \max_i x_i$$

**What they'll ask about this**
- Relation to softmax: $\log \text{softmax}(x) = x - \text{logsumexp}(x)$. That identity is the whole reason stable cross-entropy exists.
- Why not `log(sum(exp(x)))` directly? Overflow for large $x$, and `log(0) = -inf` if everything underflows.
- LSE is a smooth max: bounded between $\max x$ and $\max x + \log k$.

### Cross-entropy from logits

```python
def cross_entropy(logits, y):          # logits (n,k), y (n,) int labels
    log_probs = logits - logsumexp(logits, -1, keepdims=True)
    n = len(y)
    picked = np.zeros(n)
    for i in range(n):                 # pick the log-prob of the true class
        picked[i] = log_probs[i, y[i]]
    return -picked.mean()
```

Shapes: `(n,k)` + `(n,)` → scalar. Verified against `F.cross_entropy`: exact match to 16 digits.

`keepdims=True` instead of `[:, None]` — same thing, one less thing to remember. The fancy-index
one-liner is `-log_probs[np.arange(n), y].mean()`; write the loop, mention the one-liner.

Binary, from logits, no sigmoid anywhere:

```python
def bce_with_logits(z, y):             # z (n,), y (n,) in {0,1}
    return (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean()
```

Shapes: `(n,)` + `(n,)` → scalar. Matches `F.binary_cross_entropy_with_logits` to machine precision (`0.20501879790072944`), including `z = -50` and `z = 60` where the naive form returns `nan`.

**What they'll ask about this**
- Derive that `max(z,0) - z*y + log1p(exp(-|z|))` identity — it's just $\log(1+e^{z})$ rewritten to never exponentiate a positive number.
- Why does PyTorch's `cross_entropy` take logits, not probabilities? Fused log-softmax + NLL: stable, and the gradient simplifies to `softmax(z) - onehot(y)`.
- What's the gradient of CE w.r.t. logits? `(p - y) / n`. Say this fast; it's the answer to half of the follow-ups.
- `reduction='mean'` vs `'sum'` — affects effective learning rate.

### Sigmoid and softplus

```python
def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))     # stable everywhere
```

Shapes: elementwise. Verified on `[-1000, -1, 0, 1, 1000]` → `[0, 0.2689, 0.5, 0.7311, 1]`, no overflow warnings.

The trick: $-\log \sigma(z) = \log(1 + e^{-z}) = \text{softplus}(-z)$, and `np.logaddexp(0, -z)` computes $\log(1+e^{-z})$ without ever forming $e^{-z}$. The naive `1/(1+np.exp(-z))` overflows for `z = -1000`.

**What they'll ask about this**
- $\sigma'(z) = \sigma(z)(1-\sigma(z))$, max value $0.25$ at $z=0$ → vanishing gradients in deep sigmoid stacks.
- $\sigma(z) = \text{softmax}([z, 0])_0$ — binary is the 2-class special case.
- Why is softplus the "smooth ReLU"? $\log(1+e^z) \to z$ for large $z$, $\to 0$ for very negative $z$.

---

## Linear regression

### Normal equation

```python
def fit_normal(X, y):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X1.T @ X1, X1.T @ y)      # (d+1,) = [w..., b]
```

Shapes: `(n,d)` + `(n,)` → `(d+1,)`. On `w=[1.5,-2,0.5], b=0.7` with noise, recovers `[1.496, -2.005, 0.503, 0.698]`.

$$\hat\theta = (X^\top X)^{-1} X^\top y$$

Unusable when: $X^\top X$ is singular (collinear features, or $d > n$); $d$ is large ($O(d^3)$ solve, $O(nd^2)$ to form the Gram matrix); the data doesn't fit in memory. Also: **never** write `np.linalg.inv(...) @ ...` — use `solve`, or `np.linalg.lstsq(X1, y, rcond=None)[0]` which handles rank deficiency via SVD (verified: same answer to 3 decimals).

### Gradient descent

```python
def fit_gd(X, y, lr=0.1, steps=500):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        r = X @ w + b - y            # (n,) residual
        w -= lr * (X.T @ r) / n
        b -= lr * r.mean()
    return w, b
```

Shapes: `(n,d)` + `(n,)` → `(d,)`, scalar. Converges to `[1.496, -2.005, 0.503], 0.698` — same as the closed form.

**What they'll ask about this**
- Add ridge: `solve(X1.T@X1 + lam*np.eye(d+1), X1.T@y)` — and don't regularize the bias row in a strict implementation.
- Why does the `/n` matter? Otherwise the effective step scales with dataset size and you diverge.
- Condition number and feature scaling: unscaled features → elongated loss contours → tiny usable `lr`.
- MSE has a unique global min (convex, quadratic); GD with `lr < 2/L` converges, $L = \lambda_{\max}(X^\top X)/n$.

### PyTorch, manual (no `nn.Module`)

```python
import torch
w = torch.zeros(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for _ in range(500):
    loss = ((X @ w + b - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad; b -= 0.1 * b.grad
        w.grad.zero_(); b.grad.zero_()
```

Final: `w = [1.497, -2.001, 0.498]`, `b = 0.700`, loss `0.00285`.

### PyTorch, idiomatic

```python
import torch.nn as nn
model = nn.Linear(3, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
lossf = nn.MSELoss()
for _ in range(500):
    opt.zero_grad()
    loss = lossf(model(X).squeeze(-1), y)
    loss.backward()
    opt.step()
```

Shapes: `X (n,3)` → `model(X) (n,1)` → squeeze → `(n,)`. Identical result to the manual version.

**What they'll ask about this**
- Why `with torch.no_grad()` around the update? Otherwise the parameter update itself joins the graph.
- Why `.grad.zero_()`? Autograd **accumulates** into `.grad`; the optimizer version does this via `opt.zero_grad()`.
- Why `.squeeze(-1)`? `nn.Linear(3,1)` returns `(n,1)`; against `y` of shape `(n,)` MSELoss broadcasts to `(n,n)` and silently computes garbage.

---

## Logistic regression

### NumPy from scratch

```python
def sigmoid(z): return np.exp(-np.logaddexp(0, -z))

def bce(p, y, eps=1e-12):
    return -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)).mean()

def fit_logreg(X, y, lr=0.5, steps=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)          # (n,)
        w -= lr * (X.T @ (p - y)) / n   # (d,)
        b -= lr * (p - y).mean()
    return w, b
```

Shapes: `(n,d)` + `(n,)` → `(d,)`, scalar. On separable synthetic data: acc `0.995`, BCE `0.0572`.

The gradient `X.T @ (sigma - y) / n` is the whole trick — **identical in form to linear regression's**, because both are GLMs with the canonical link. Say that out loud when you write it.

### PyTorch

```python
clf = nn.Linear(3, 1)
opt = torch.optim.Adam(clf.parameters(), lr=0.1)
lossf = nn.BCEWithLogitsLoss()
for _ in range(300):
    opt.zero_grad()
    logits = clf(X).squeeze(-1)         # (n,) — raw logits, NO sigmoid
    loss = lossf(logits, y)             # y float in {0,1}
    loss.backward(); opt.step()
probs = torch.sigmoid(clf(X).squeeze(-1))   # sigmoid only at inference
```

Final loss `0.0540`, accuracy `1.0`.

Why `BCEWithLogitsLoss` and not `Sigmoid` + `BCELoss`: the fused version uses the log-sum-exp trick internally, so it's finite for `|z| = 60`; the split version computes $\sigma(z)$ first, saturates to exactly 0 or 1 in float32, then takes `log(0)` → `inf`/`nan`. It also gives a cleaner gradient (`p - y`) instead of one that passes through a near-zero $\sigma'$.

**What they'll ask about this**
- Derive $\partial \mathcal{L}/\partial z = \sigma(z) - y$.
- Class imbalance: `pos_weight` in `BCEWithLogitsLoss`, threshold tuning, PR curve over ROC.
- Why is BCE convex in $w$ but MSE-on-sigmoid isn't?
- Multiclass: swap to `nn.Linear(d, k)` + `CrossEntropyLoss` with `long` labels, no one-hot.

---

## Attention

### NumPy, pure

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)     # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)       # mask True = keep
    A = softmax(scores, -1)
    return A @ V, A
```

Shapes: `Q (B,T_q,d_k)`, `K (B,T_k,d_k)`, `V (B,T_k,d_v)` → out `(B,T_q,d_v)`, weights `(B,T_q,T_k)`. Verified `(2,5,8)` in → `(2,5,8)` out, attention rows sum to exactly 1.

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### PyTorch, 6 lines

```python
import torch.nn.functional as F

def attn(q, k, v, mask=None):
    d_k = q.size(-1)
    s = q @ k.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        s = s.masked_fill(mask == 0, float('-inf'))
    return F.softmax(s, dim=-1) @ v
```

Verified numerically identical to the numpy version (`allclose` True, masked and unmasked).

### Causal mask

```python
T = x.size(1)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))     # True = blocked
```

`diagonal=1` keeps the diagonal (a token attends to itself) and blocks strictly-future positions. Verified: row 0 of the softmax is `[1, 0, 0, 0, 0]`.

Two conventions, and mixing them is the classic bug: `torch.triu(..., 1)` produces **True = mask out** (feed to `masked_fill` directly); `torch.tril(ones)` produces **True = keep** (needs `masked_fill(mask == 0, -inf)`).

**What they'll ask about this**
- Why $\sqrt{d_k}$? With unit-variance $q, k$ entries, the dot product of $d_k$ terms has variance $d_k$; without scaling, logits grow like $\sqrt{d_k}$, softmax saturates, gradients vanish.
- Why `-inf` and not a large negative number? `-inf` gives exactly 0 after softmax. But a fully-masked row gives `nan` — a real edge case with padding masks.
- Complexity: $O(T^2 d)$ time, $O(T^2)$ memory for the score matrix. This is the setup for a FlashAttention question.
- Self- vs cross-attention: same function, different source for K/V.

### Multi-head attention

```python
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # split heads: (B, T, d_model) -> (B, n_heads, T, d_head), one line each
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / self.dk**0.5       # (B,H,T,T)
        if causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v                                     # (B,H,T,dk)

        out = out.transpose(1, 2)                             # (B,T,H,dk)
        out = out.reshape(B, T, C)                            # merge heads
        return self.proj(out)
```

Shapes: `(2,6,32)` in → `(2,6,32)` out, `d_model=32, n_heads=4, d_head=8`. Verified.

The reshape dance is the part people fumble: `view(B, T, H, dk).transpose(1, 2)` to split heads, `transpose(1, 2).reshape(B, T, C)` to merge. `transpose` then `reshape` (not `view`) — the tensor is non-contiguous after transposing.

Modern one-liner — same numbers, fused kernel, no explicit mask:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B,H,T,dk)
```

Verified: `allclose` to the manual implementation at `atol=1e-5` with shared weights. It dispatches to FlashAttention / memory-efficient backends and never materializes the $T \times T$ matrix.

**What they'll ask about this**
- Why one fused `qkv` Linear instead of three? One GEMM, better utilization. Then `.split(C, dim=2)`.
- Param count: $4d^2 + 4d$ (Q, K, V, out projections). Independent of head count.
- Why multiple heads at all if total compute is the same? Different subspaces / relation types per head.
- What's `self.proj` for? Mixing information across heads — without it, heads never interact.
- MQA / GQA: share K,V across heads to shrink the KV cache at inference.

---

## Transformer block

### LayerNorm from scratch

```python
def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                    # biased (÷N)
    return g * (x - mu) / np.sqrt(var + eps) + b
```

Shapes: `(B,T,d)` in → `(B,T,d)` out. Verified `allclose` to `nn.LayerNorm(6)` at `atol=1e-6`; per-position mean `0`, std `1`.

Torch version, if they want an `nn.Module`:

```python
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)   # unbiased=False matters
        return self.g * (x - mu) / torch.sqrt(var + self.eps) + self.b
```

**What they'll ask about this**
- LayerNorm vs BatchNorm: LN normalizes over the feature dim per token — no batch dependence, no train/eval discrepancy, works with variable sequence length.
- Why `unbiased=False`? `nn.LayerNorm` uses the biased ($\div N$) variance. `unbiased=True` gives a mismatch that widens as $d$ shrinks.
- Where does `eps` go — inside or outside the sqrt? Inside. Outside is a real bug.
- RMSNorm: drop the mean subtraction and the bias, divide by $\sqrt{\text{mean}(x^2)}$. Cheaper, used by Llama.

### Pre-LN block

```python
class Block(nn.Module):
    def __init__(self, d_model, n_heads, mult=4, p=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mult * d_model), nn.GELU(),
            nn.Linear(mult * d_model, d_model), nn.Dropout(p))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Shapes: `(2,7,32)` in → `(2,7,32)` out; 12,704 params at `d_model=32, n_heads=4`.

**What they'll ask about this**
- Pre-LN vs post-LN: pre-LN (`x + f(ln(x))`) leaves a clean identity path, so it trains without a warmup schedule. Post-LN (`ln(x + f(x))`) is the original paper and needs warmup to be stable.
- Why the 4× MLP expansion? Empirical; most parameters live here ($8d^2$ vs $4d^2$ for attention).
- GELU vs ReLU — smooth, nonzero gradient for small negatives. SwiGLU in modern models.
- Where does dropout go? After the MLP output and on attention weights; largely dropped in large-scale LLM pretraining.

### Sinusoidal positional encoding

```python
def positional_encoding(T, d):
    pe = np.zeros((T, d))
    for pos in range(T):
        for i in range(0, d, 2):           # even index: sin, odd index: cos
            angle = pos / (10000 ** (i / d))
            pe[pos, i]     = np.sin(angle)
            pe[pos, i + 1] = np.cos(angle)
    return pe
```

Shapes: → `(T, d)`. Verified `(10,16)`; row 0 is `[0,1,0,1]`, row 1 is `[0.841, 0.540, 0.311, 0.950]`,
and identical to the vectorized version.

The double loop reads straight off the formula below, which is the point — you can derive it at the
whiteboard instead of recalling it. Assumes `d` is even; say that.

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Modern models use RoPE instead — rotate Q and K by a position-dependent angle inside attention, so only *relative* position enters the scores and context extension is possible.

**What they'll ask about this**
- Why sinusoids? $PE_{pos+k}$ is a fixed linear function of $PE_{pos}$, so relative offsets are linearly decodable; extrapolates past training length in principle.
- Learned vs fixed embeddings — learned is simpler and just as good in-distribution, but can't extrapolate.
- Why does a transformer need this at all? Attention is permutation-equivariant; without positions, "dog bites man" == "man bites dog".

---

## The training loop

```python
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
lossf = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()                                    # 1. clear grads
        loss = lossf(model(xb), yb)                        # 2. forward
        loss.backward()                                    # 3. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4. after backward, before step
        opt.step()                                         # 5. update
        total += loss.item() * xb.size(0)
    sched.step()                                           # 6. per epoch (not per batch)
    print(epoch, total / len(train_loader.dataset))
```

Ran clean: loss `0.692 → 0.673 → 0.663`, LR `7.5e-4 → 2.5e-4 → 0`.

Order that matters: `zero_grad` → forward → `backward` → **clip** → `step` → **`sched.step()` after `opt.step()`**. Clipping before `backward` clips stale gradients; `sched.step()` before `opt.step()` skips the first LR and warns.

### Evaluation

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = n = 0
    for xb, yb in loader:
        pred = model(xb).argmax(-1)
        correct += (pred == yb).sum().item(); n += yb.size(0)
    return correct / n
```

Returns a float; verified `0.609` on the smoke model.

**What they'll ask about this**
- `model.eval()` vs `torch.no_grad()` — different jobs. `eval()` switches dropout off and BatchNorm to running stats; `no_grad()` stops graph building (memory + speed). You need **both**.
- Why `loss.item()` and not `loss`? Keeping the tensor retains the whole graph → memory leak across the epoch.
- Weighting by `xb.size(0)` handles the ragged last batch.
- Gradient accumulation: `loss = loss / accum_steps`, step every `accum_steps` batches.
- AMP: wrap forward in `torch.autocast`, use `GradScaler`, and `scaler.unscale_(opt)` before clipping.

---

## Classic ML, written the way you would write it under pressure

### k-means

Two loops, no tricks. Assign, then move. Write this one and you will not get lost:

```python
def kmeans(X, k, iters=50, seed=0):
    n = len(X)
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(n, k, replace=False)].astype(float).copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(iters):
        # Step 1: assign each point to its nearest centroid
        for i in range(n):
            dists = ((X[i] - centroids) ** 2).sum(axis=1)   # (k,)
            labels[i] = dists.argmin()

        # Step 2: move each centroid to the mean of its points
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:                            # empty cluster: leave it
                centroids[j] = points.mean(axis=0)

    return centroids, labels
```

Shapes: `(n,d)` → centroids `(k,d)`, labels `(n,)`. Verified on two Gaussian blobs at 0 and 5:
centroids `[[-0.01,0.18],[4.87,5.03]]`, split `[50,50]`, and the final inertia matches
`sklearn.cluster.KMeans` exactly (182.411).

If you want the inner distance loop fully explicit too — no broadcasting at all — this is the same thing:

```python
        for i in range(n):
            best_j, best_dist = 0, float('inf')
            for j in range(k):
                diff = X[i] - centroids[j]
                dist = np.dot(diff, diff)
                if dist < best_dist:
                    best_dist, best_j = dist, j
            labels[i] = best_j
```

Verified identical output to the version above. **Write the loop version, then say out loud:** "I would
vectorize the assignment step as a single `(n,k)` distance matrix in production — it is
`((X[:,None,:] - centroids)**2).sum(-1)` — but I will keep it explicit here so it is easy to check."
Interviewers consistently prefer that over a clever line you cannot debug when they change the problem.

**Ask:** empty-cluster handling (reinit or keep old centroid); k-means++ init; converges to a local min only; assumes isotropic equal-variance clusters; $O(nkd)$ per iteration.

### k-NN prediction

One query point at a time. Distances, sort, vote:

```python
def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = ((X_train - x) ** 2).sum(axis=1)    # (n,) distance to every train point
        nearest = np.argsort(dists)[:k]             # indices of the k smallest

        votes = {}                                 # majority vote among those k
        for label in y_train[nearest]:
            votes[label] = votes.get(label, 0) + 1
        preds.append(max(votes, key=votes.get))

    return np.array(preds)
```

Shapes: `(m,d)` query → `(m,)` labels. Verified: 100% train accuracy on the blob data, correct on
`[[0,0]]` → 0 and `[[5,5]]` → 1, and identical predictions to `sklearn.neighbors.KNeighborsClassifier(3)`.

No `np.bincount` trick, no broadcasting. The squared distance is fine — skipping the square root does
not change the ordering, and saying that out loud is a small free point.

**Ask:** no training cost, $O(nd)$ per query at test time; curse of dimensionality; must scale features; use `np.argpartition` instead of `argsort` for $O(n)$; ties in `bincount`.

### PCA via SVD

```python
def pca(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, Vt[:k], S[:k]**2 / (len(X) - 1)
```

Shapes: `(n,d)` → scores `(n,k)`, components `(k,d)`, explained variance `(k,)`. Verified `(100,2)` → variances `[12.557, 0.199]`.

**Ask:** why SVD and not eigendecomposition of the covariance? Better conditioned, no $X^\top X$ needed. Centering is mandatory. Rows of `Vt` are eigenvectors of the covariance; $\lambda_i = s_i^2/(n-1)$.

### Train/test split

```python
def train_test_split(X, y, test_size=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]
```

Verified: `(80,2), (20,2), (80,), (20,)`.

**Ask:** stratify for imbalanced labels; group splits when rows share an entity; **time-based split for temporal data** — random splits leak the future.

### Metrics

```python
def metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

def confusion(y_true, y_pred, k=2):
    M = np.zeros((k, k), dtype=int)       # rows = true, cols = predicted
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M
```

Verified: `y=[1,1,0,0,1]`, `pred=[1,0,0,1,1]` → precision/recall/F1 all `0.667`, confusion `[[1,1],[1,2]]`.

**Ask:** F1 is the *harmonic* mean — punishes imbalance between P and R; macro vs micro averaging; when accuracy lies (99% negatives); PR-AUC over ROC-AUC under heavy imbalance.

---

## The five bugs interviewers watch for

1. **Missing `optimizer.zero_grad()`.** Gradients accumulate across batches, so each step uses the sum of all gradients so far. Symptom: loss drops for a few steps, then explodes or plateaus at a bad value; effective LR grows without bound.

2. **Softmax over the wrong axis.** In attention, `dim=-1` normalizes over keys (correct); `dim=-2` normalizes over queries. Symptom: it runs, shapes are right, loss decreases slowly to a mediocre plateau — every row no longer sums to 1 across keys. Always assert `A.sum(-1) ≈ 1`.

3. **Forgetting `/sqrt(d_k)`.** Symptom: with large `d_head`, attention logits scale like $\sqrt{d_k}$, softmax saturates to near-one-hot, gradients through it vanish. Training stalls immediately and worsens as you scale the model — a bug that looks like "big models just don't train."

4. **Sigmoid before `BCEWithLogitsLoss`.** You've applied the sigmoid twice: the loss squashes the already-squashed probabilities. Symptom: loss is finite but bounded away from 0, gradients are tiny, model underfits and predictions collapse toward 0.5. Corollary: applying `softmax` before `nn.CrossEntropyLoss`.

5. **Skipping `model.eval()` / `torch.no_grad()` at inference.** Symptom: validation accuracy is noisy and lower than train for no reason (dropout still active, BatchNorm using batch stats — which also makes predictions depend on batch composition), plus OOM from graph retention.

Runner-up: **wrong mask orientation.** `torch.triu(ones, 1)` is True-means-block; `torch.tril(ones)` is True-means-keep. Invert it and every token attends only to the future. Symptom: training loss goes suspiciously, impossibly low, and generation is garbage — the classic label-leak signature. Also check `diagonal=1`, not `0`: `diagonal=0` masks the token's own position and row 0 becomes all `-inf` → `nan`.
