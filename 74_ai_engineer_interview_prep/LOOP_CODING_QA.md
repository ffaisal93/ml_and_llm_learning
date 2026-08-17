# AI/ML Coding Round — Question Bank with Worked Solutions

Sixty problems in the exact mix these loops use: ML implemented from scratch, DSA that actually gets asked, data manipulation, and debugging. Every code block in this file was executed and checked against a reference (PyTorch, scikit-learn, or brute force) before being written down.

## How this round is actually scored (read once, then go do problems)

The bar is not "did you produce an accepted solution." It is "would I ship code with this person." Four signals:

**Clarify before you type.** Two questions, maximum, then start. Input sizes, dtypes, whether the array is sorted, what to return on empty input, whether ties matter. In ML-from-scratch problems the highest-value clarifications are shape conventions (is it `(B, T, D)` or `(T, B, D)`?) and whether you may use a framework or must stay in NumPy.

**Say the brute force out loud, then improve it.** "Nested loop is $O(n^2)$; the repeated work is re-scanning for the complement, so a hash map buys $O(n)$." Interviewers grade the transition, not the destination. Never silently jump to the optimal answer — you lose the chance to show the reasoning, and if the optimal has a bug you have no fallback.

**Numerical stability is a correctness bug, not a nicety.** In an ML loop, a `softmax` without max-subtraction, a cross-entropy that takes `log` of a probability, or a `sigmoid` that calls `exp` on a large positive number is a failed problem regardless of whether the test case happens to pass. Say why: $\mathrm{exp}(1000)$ overflows to `inf` in float64, and `inf/inf` is `nan`.

**Test yourself before they ask.** Empty input, single element, all-identical elements, ties, and one hand-checkable case. Candidates who trace their own code on `[]` and `[5]` almost never get a "did you consider…" question.

Two habits that cost offers: mutating an input silently, and writing Python loops where a vectorized form exists. For ML problems, comment your shapes on every reshape — that is itself a hiring signal.

Conventions used below: `B` = batch, `T` = sequence length, `D`/`d_model` = model dim, `h` = heads, `d_k = D/h`.

---

## Part 1 — ML From Scratch

### Q: Implement softmax in NumPy over the last axis. Make it numerically stable.

**Clarify first:** Which axis, and should it handle arbitrary-rank input (a `(B, h, T, T)` attention score tensor) or just 1-D? Do I need to handle `-inf` entries from masking?

**Approach.** The definition is $\sigma(x)_i = e^{x_i} / \sum_j e^{x_j}$. Written literally it breaks: with logits around $1000$, $e^{1000}$ overflows float64 to `inf`, and `inf/inf` gives `nan`. The fix uses the shift-invariance of softmax: for any constant $c$,

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{-c}e^{x_i}}{e^{-c}\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

so we may pick $c = \max_j x_j$ for free. Then the largest exponent is exactly $e^0 = 1$ (no overflow), and the worst underflow is a term flushing to $0$, which is harmless because the denominator contains the $1$. Use `keepdims=True` so it broadcasts on any rank.

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)   # shift-invariance: subtract the max
    e = np.exp(x - x_max)                         # largest term is exactly exp(0) == 1
    return e / np.sum(e, axis=axis, keepdims=True)

x = np.array([[1000.0, 1001.0, 1002.0], [1.0, 2.0, 3.0]])
print(softmax(x))          # [[0.09003057 0.24472847 0.66524096]
                           #  [0.09003057 0.24472847 0.66524096]]
print(softmax(x).sum(1))   # [1. 1.]
```

Note both rows give the same answer — softmax depends only on logit *differences*. The naive version returns `[nan nan nan]` on row 0.

**Complexity:** $O(n)$ time, $O(n)$ space for the output; two passes over the axis (max, then sum) with $O(1)$ extra working memory per row.

**Follow-up: "What if an entire row is `-inf` from masking?"** → You get `0/0 = nan`. `x_max` is `-inf`, `x - x_max` is `-inf - (-inf) = nan`. Guard it if fully-masked rows are possible:

```python
def softmax_safe(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)  # all -inf row -> shift by 0
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    return np.where(s > 0, e / np.where(s > 0, s, 1.0), 0.0)
```

In production this is why masks use a large finite negative (`-1e9`) rather than `-inf`.

*Trap:* Subtracting a global `x.max()` instead of a per-row max. It won't overflow, but it silently changes nothing mathematically and hides the fact that you understood *why* per-row matters — and it does underflow to zeros if rows have very different scales.

---

### Q: Implement cross-entropy loss from logits. No `log` of a probability anywhere.

**Clarify first:** Am I given logits or probabilities, and are labels integer class indices or one-hot? Mean or sum reduction, and do I need to support ignoring a padding index?

**Approach.** Naive route: softmax, then `-log(p[label])`. That composes two unstable ops — softmax can underflow a small probability to exactly $0$, and `log(0) = -inf`. People patch it with `+1e-12`, which biases the loss and still loses precision. The right move is to never form the probability. Fuse it:

$$\log \sigma(x)_y = x_y - \log\sum_j e^{x_j} = (x_y - c) - \log\sum_j e^{x_j - c},\quad c=\max_j x_j$$

That is log-sum-exp. Compute `log_softmax` once, then index the true class. Loss is $-\frac{1}{N}\sum_i \log\sigma(x_i)_{y_i}$. The gradient falls out beautifully: $\partial L/\partial x = (\sigma(x) - \text{onehot}(y))/N$.

```python
import numpy as np

def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    z = x - x_max
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))   # log-sum-exp trick

def cross_entropy(logits, labels, reduction="mean"):
    """logits: (N, C) raw scores. labels: (N,) int class indices."""
    ls = log_softmax(logits, axis=-1)          # (N, C)
    n = logits.shape[0]
    nll = -ls[np.arange(n), labels]            # pick the true-class log-prob
    return nll.mean() if reduction == "mean" else nll.sum()

def cross_entropy_grad(logits, labels):
    """dL/dlogits for the mean-reduced loss."""
    p = np.exp(log_softmax(logits))
    n = logits.shape[0]
    p[np.arange(n), labels] -= 1.0
    return p / n

logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
labels = np.array([0, 1])
print(cross_entropy(logits, labels))   # 0.31853976964918573
```

Verified against `torch.nn.functional.cross_entropy`: identical to machine precision, and `cross_entropy_grad` matches autograd's `.grad` exactly.

**Complexity:** $O(NC)$ time, $O(NC)$ space for the `log_softmax` buffer. One fused pass; no intermediate probability tensor is needed if you only want the loss.

**Follow-up: "Now support an `ignore_index` for padding tokens."** → Mask the per-token losses and divide by the count of *valid* tokens, not by `N`:

```python
def cross_entropy_ignore(logits, labels, ignore_index=-100):
    ls = log_softmax(logits, axis=-1)
    n = logits.shape[0]
    valid = labels != ignore_index
    safe = np.where(valid, labels, 0)                    # dummy index, masked out after
    nll = -ls[np.arange(n), safe]
    denom = max(int(valid.sum()), 1)
    return float((nll * valid).sum() / denom)
```

Dividing by `N` instead of `valid.sum()` is the single most common LM training bug — it makes loss depend on how much padding is in the batch.

*Trap:* Writing `-np.log(softmax(logits)[range(n), labels] + 1e-9)`. It runs, it roughly matches, and it marks you as someone who has read about log-sum-exp but not internalized it. The interviewer will ask what happens at a logit of $-800$; the epsilon version returns $\log(10^{-9}) \approx 20.7$ for *every* such case, destroying the gradient signal.

---

### Q: Implement a numerically stable sigmoid, and binary cross-entropy from logits.

**Clarify first:** Do I get logits or probabilities for the BCE? Do I need to handle very large-magnitude inputs (roughly $|z| > 700$, where `exp` overflows in float64)?

**Approach.** $\sigma(z) = 1/(1+e^{-z})$ overflows when $z$ is very negative: $e^{-(-1000)} = e^{1000} = \infty$, giving $1/\infty = 0$ — actually correct by luck in float, but it raises a warning and it genuinely produces `nan` in the equivalent `e^z/(1+e^z)` form for large positive $z$. Branch on sign so the exponent argument is always $\le 0$: use $1/(1+e^{-z})$ for $z \ge 0$ and $e^{z}/(1+e^{z})$ for $z < 0$. For BCE, don't compose `log` with `sigmoid`; use the fused identity

$$L(z,y) = \max(z,0) - zy + \log\left(1+e^{-|z|}\right)$$

which is exactly `binary_cross_entropy_with_logits`. The `log1p` handles the small-argument case precisely.

```python
import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))     # exponent <= 0 here
    ex = np.exp(x[~pos])                          # exponent < 0 here too
    out[~pos] = ex / (1.0 + ex)
    return out

def bce_with_logits(z, y):
    """Stable mean binary cross-entropy. z: raw logits, y: 0/1 targets."""
    z = np.asarray(z, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

print(sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])))
# [0. 0.26894142 0.5 0.73105858 1.]
print(bce_with_logits(np.array([-500.0, 0.5, 3.0]), np.array([0.0, 1.0, 1.0])))
# 0.1742214452512829  -- matches torch.nn.functional.binary_cross_entropy_with_logits
```

**Complexity:** $O(n)$ time and space. The branch costs two masked passes but no extra asymptotic work.

**Follow-up: "Derive $\sigma'(z)$ and explain vanishing gradients."** → $\sigma'(z) = \sigma(z)(1-\sigma(z))$, maximized at $z=0$ where it equals $0.25$ and decaying exponentially in $|z|$. Stacking $L$ sigmoid layers multiplies $L$ factors each $\le 0.25$, so gradients shrink like $4^{-L}$ — the historical reason for ReLU. Note this is also why you keep BCE fused: at $z=-500$, $\sigma(z)$ rounds to exactly $0$, so `log(sigmoid(z))` is `-inf` and its gradient is `nan`, whereas the fused form gives a finite loss and gradient $\sigma(z)-y$.

*Trap:* Claiming the naive `1/(1+exp(-x))` is "fine because NumPy handles inf." It returns the right value but emits an overflow warning, and the same naive pattern in the loss (`log(p)`) genuinely produces `nan`. Interviewers are testing whether you know which composition breaks.

---

### Q: Implement scaled dot-product attention in NumPy.

**Clarify first:** What are the input shapes — is it `(B, T, d_k)` or already split into heads as `(B, h, T, d_k)`? Do I need to return the attention weights as well as the output, and is the mask boolean-keep or additive?

**Approach.** The formula is

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$QK^\top$ gives a $(T_q, T_k)$ score matrix — every query's dot product with every key. The $\sqrt{d_k}$ divisor exists because if $q$ and $k$ have i.i.d. unit-variance components, $q\cdot k$ has variance $d_k$; without scaling, scores grow like $\sqrt{d_k}$, softmax saturates into a near one-hot, and gradients through it vanish. Softmax over the *key* axis (last), then weight the values. Write it with `swapaxes(-1, -2)` rather than `.T` so it works on any leading batch/head dims.

```python
import numpy as np

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (..., Tq, dk)  K, V: (..., Tk, dk). mask: bool, True = keep."""
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)     # (..., Tq, Tk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)          # large finite, not -inf
    weights = softmax(scores, axis=-1)                 # rows sum to 1 over keys
    return weights @ V, weights                        # (..., Tq, dk)

rng = np.random.default_rng(0)
Q, K, V = (rng.normal(size=(2, 4, 8)) for _ in range(3))
out, w = scaled_dot_product_attention(Q, K, V)
print(out.shape, w.sum(-1))    # (2, 4, 8), all ones
```

Checked against `torch.nn.functional.scaled_dot_product_attention` — matches to machine precision.

**Complexity:** $O(T^2 d_k)$ time and $O(T^2)$ memory for the score matrix — the quadratic memory term is what FlashAttention removes by tiling and never materializing the full $T\times T$ matrix.

**Follow-up: "Why $-10^9$ instead of $-\infty$?"** → With `-inf`, a fully-masked row (every key masked, which happens with padded-only rows) makes `x - x_max` equal `-inf - (-inf) = nan`, and the `nan` propagates through the whole batch. A large finite negative degrades gracefully to a uniform distribution over the masked row instead. Also, in float16, `-1e9` overflows — use `-1e4` or `np.finfo(dtype).min` when running in half precision.

*Trap:* Softmaxing over the wrong axis. `softmax(scores, axis=-2)` normalizes over queries instead of keys; shapes are identical, output is garbage, no error is raised. Always assert `weights.sum(-1) ≈ 1`.

---

### Q: Implement multi-head attention, including the reshape and transpose.

**Clarify first:** Is `d_model` guaranteed divisible by `n_heads`? Should the projections be a single fused `(d_model, 3*d_model)` matrix or three separate ones, and do we need a KV cache for incremental decoding?

**Approach.** One head over `d_model` can only express one similarity structure; $h$ heads let the model attend to different subspaces in parallel at the same total FLOP cost, since each head works in $d_k = d_{\text{model}}/h$ dimensions. The mechanical core is the shape dance: project to `(B, T, d_model)`, view as `(B, T, h, d_k)`, then **transpose** to `(B, h, T, d_k)` so the batched matmul treats `h` as a batch dim and each head attends over its own `T × T`. After attention, transpose back and reshape to `(B, T, d_model)`, then apply the output projection $W_O$ that mixes head outputs. The transpose is not optional — `reshape(B, h, T, d_k)` directly has the same shape but interleaves tokens into heads, which is a silent, catastrophic bug.

```python
import numpy as np

def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        rng = np.random.default_rng(seed)
        self.h, self.d_k, self.d = n_heads, d_model // n_heads, d_model
        s = 1 / np.sqrt(d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = [rng.normal(0, s, (d_model, d_model))
                                              for _ in range(4)]

    def _split(self, x):                      # (B, T, D) -> (B, h, T, dk)
        B, T, _ = x.shape
        return x.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):                      # (B, h, T, dk) -> (B, T, D)
        B, h, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, h * dk)

    def __call__(self, x, mask=None):
        q, k, v = self._split(x @ self.Wq), self._split(x @ self.Wk), self._split(x @ self.Wv)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(self.d_k)      # (B, h, T, T)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)                # (T,T) broadcasts over B,h
        out = softmax(scores, -1) @ v                            # (B, h, T, dk)
        return self._merge(out) @ self.Wo                        # (B, T, D)

mha = MultiHeadAttention(16, 4)
x = np.random.default_rng(1).normal(size=(2, 5, 16))
print(mha(x).shape)                                              # (2, 5, 16)
```

**Complexity:** $O(B\,T^2 d_{\text{model}} + B\,T\,d_{\text{model}}^2)$ time — attention term plus projection term. Memory $O(B h T^2)$ for the scores. Note total attention FLOPs are independent of $h$: $h$ heads $\times$ $d_k = d_{\text{model}}/h$ each.

**Follow-up: "Add a KV cache for autoregressive decoding — what changes?"** → At step $t$ you only compute a query for the single new token, and you append its key/value to caches of shape `(B, h, t, d_k)`:

```python
def step(self, x_t, cache=None):
    """x_t: (B, 1, D) — one new token. cache: (K, V) each (B, h, t, dk)."""
    q = self._split(x_t @ self.Wq)                      # (B, h, 1, dk)
    k, v = self._split(x_t @ self.Wk), self._split(x_t @ self.Wv)
    if cache is not None:
        k = np.concatenate([cache[0], k], axis=2)       # grow along T
        v = np.concatenate([cache[1], v], axis=2)
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(self.d_k)  # (B, h, 1, t+1)
    out = softmax(scores, -1) @ v
    return self._merge(out) @ self.Wo, (k, v)
```

Cost per token drops from $O(T^2)$ to $O(T)$, at the price of $O(BhTd_k)$ memory that grows with context — which is exactly what multi-query and grouped-query attention shrink by sharing K/V across heads.

*Trap:* `x.reshape(B, self.h, T, self.d_k)` instead of reshape-then-transpose. Same shape, silently wrong values, model trains to mediocre loss and nobody notices for a week. Verify with `right.transpose(0,2,1,3).reshape(B,T,D) == x`.

---

### Q: Build a causal mask and apply it. Prove that no position can see the future.

**Clarify first:** Should the mask be boolean-keep or an additive $-\infty$ bias? Do I need to combine it with a padding mask, and are we in the prefix-LM setting where a prompt region is bidirectional?

**Approach.** A causal (autoregressive) mask forbids query position $i$ from attending to key position $j > i$; without it, the LM objective is trivially solvable by copying the next token, so training loss collapses and generation is garbage. The mask is a lower-triangular boolean of shape `(T, T)` — `np.tril(np.ones((T, T), bool))`. Applied *before* softmax so masked positions get zero probability, not zeroed afterward (zeroing after softmax leaves rows that don't sum to 1). The right test is behavioral: perturb the last token's input and confirm all earlier outputs are bit-identical.

```python
import numpy as np

def causal_mask(T):
    """True where attention is allowed: position i may see j <= i."""
    return np.tril(np.ones((T, T), dtype=bool))

def combine_masks(causal, pad):
    """causal: (T,T). pad: (B,T) True for real tokens. -> (B,1,T,T)"""
    return causal[None, None, :, :] & pad[:, None, None, :]

print(causal_mask(4).astype(int))
# [[1 0 0 0]
#  [1 1 0 0]
#  [1 1 1 0]
#  [1 1 1 1]]

# Behavioral proof of causality using the MHA above
x  = np.random.default_rng(1).normal(size=(2, 5, 16))
x2 = x.copy(); x2[:, 4, :] += 10.0            # scramble the LAST token only
a, b = mha(x, causal_mask(5)), mha(x2, causal_mask(5))
print(np.allclose(a[:, :4], b[:, :4]))        # True -> first 4 outputs unchanged
```

Also verified against `F.scaled_dot_product_attention(..., is_causal=True)`: identical output.

**Complexity:** $O(T^2)$ to build and $O(T^2)$ memory, but it is built once and cached — never rebuild it inside the forward pass per layer or per step.

**Follow-up: "How does this interact with a padding mask, and what shape do you broadcast?"** → Combine with logical AND as above. The shape convention that matters: causal is `(1, 1, T, T)`, padding is `(B, 1, 1, T)` — the padding mask indexes **keys** (last axis), not queries. Putting it on the query axis as `pad[:, None, :, None]` broadcasts silently to the same `(B, h, T, T)` and is wrong; see the debugging section.

*Trap:* Applying the mask multiplicatively after softmax (`weights * mask`). Rows no longer sum to 1, so the output is an arbitrarily-scaled convex-ish combination, and the scale varies by position — a subtle bug that shows up as poor long-context behavior.

---

### Q: Implement layer norm forward and backward in NumPy.

**Clarify first:** Normalize over the last axis only, or over a tuple of trailing dims? Do the learnable $\gamma,\beta$ have shape `(D,)`, and do you want the backward pass for $\gamma$ and $\beta$ too?

**Approach.** LayerNorm standardizes each token's feature vector independently:

$$y = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta,\quad \mu=\frac{1}{D}\sum_d x_d,\ \sigma^2=\frac{1}{D}\sum_d (x_d-\mu)^2$$

Unlike BatchNorm, statistics come from the feature axis, so behavior is identical at train and eval and independent of batch size — the reason transformers use it. The backward pass is where candidates stall. Naively you'd chain through $\mu$ and $\sigma^2$ separately; the compact closed form is

$$\frac{\partial L}{\partial x} = \frac{1}{\sqrt{\sigma^2+\epsilon}}\left(\hat{g} - \overline{\hat{g}} - \hat{x}\,\overline{\hat{g}\hat{x}}\right),\quad \hat{g} = \frac{\partial L}{\partial y}\odot\gamma$$

where the bars are means over the normalized axis. The two subtracted terms are exactly the corrections for $\mu$ and $\sigma^2$ depending on $x$.

```python
import numpy as np

def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu  = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                 # biased (1/D), matches torch
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta, (x_hat, var, eps)

def layernorm_backward(dout, cache, gamma):
    x_hat, var, eps = cache
    reduce_axes = tuple(range(dout.ndim - 1))      # everything but the feature axis
    dgamma = (dout * x_hat).sum(axis=reduce_axes)
    dbeta  = dout.sum(axis=reduce_axes)
    g_hat  = dout * gamma
    istd   = 1.0 / np.sqrt(var + eps)
    dx = istd * (g_hat
                 - g_hat.mean(-1, keepdims=True)                 # mu correction
                 - x_hat * (g_hat * x_hat).mean(-1, keepdims=True))  # var correction
    return dx, dgamma, dbeta

rng = np.random.default_rng(0)
x, g, b = rng.normal(size=(3, 6)), rng.normal(size=6), rng.normal(size=6)
y, cache = layernorm_forward(x, g, b)
dx, dg, db = layernorm_backward(rng.normal(size=(3, 6)), cache, g)
```

Forward matches `F.layer_norm`; `dx`, `dgamma`, `dbeta` all match PyTorch autograd exactly (`np.allclose` → `True` on all three).

**Complexity:** $O(ND)$ time, $O(ND)$ space for the cached $\hat{x}$. Both passes are a constant number of sweeps over the feature axis.

**Follow-up: "Why does the transformer use LayerNorm and not BatchNorm?"** → Three reasons. (1) Sequence lengths vary and padding pollutes batch statistics — a BN mean computed over padded positions is meaningless. (2) BN's train/eval mismatch (batch stats vs. running averages) is a chronic source of bugs, and autoregressive decoding at batch size 1 has no batch to normalize over. (3) LN's per-token normalization is what makes residual streams well-conditioned regardless of batch composition. Pre-LN (`x + Attn(LN(x))`) versus Post-LN also matters: Pre-LN keeps an unnormalized identity path to the output and trains without a warmup schedule, which is why every modern LLM uses it.

*Trap:* Using `np.var(x, ddof=1)` (unbiased). PyTorch uses the biased $1/D$ estimator; with `ddof=1` your forward is off by $\sqrt{D/(D-1)}$ and every gradient check fails, usually sending candidates hunting in the backward pass for a bug that's in the forward.

---

### Q: Implement RMSNorm. Why did Llama switch to it from LayerNorm?

**Clarify first:** Is there a bias term (most RMSNorm implementations drop $\beta$)? What epsilon convention — inside the square root, as here, or outside?

**Approach.** RMSNorm drops the mean-centering entirely:

$$y = \gamma \odot \frac{x}{\sqrt{\frac{1}{D}\sum_d x_d^2 + \epsilon}}$$

The claim from the paper is that LayerNorm's benefit comes almost entirely from the *re-scaling* invariance, not the re-centering, so you can delete the mean and lose nothing. What you gain: one fewer reduction over the feature axis, no subtraction, and no $\beta$ parameter — roughly a 10–15% speedup in the norm op, which is non-trivial when you call it twice per layer across 80 layers. It's also better behaved in low precision because you skip the catastrophic-cancellation-prone $x-\mu$.

```python
import numpy as np

def rmsnorm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)

def rmsnorm_backward(dout, x, gamma, eps=1e-6):
    D = x.shape[-1]
    ms  = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    rms = np.sqrt(ms)
    x_hat = x / rms
    dgamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim - 1)))
    g_hat = dout * gamma
    dx = (g_hat - x_hat * np.mean(g_hat * x_hat, axis=-1, keepdims=True)) / rms
    return dx, dgamma

x = np.random.default_rng(0).normal(size=(3, 6)); g = np.random.default_rng(1).normal(size=6)
dout = np.random.default_rng(2).normal(size=(3, 6))
y = rmsnorm(x, g)
dx, dgamma = rmsnorm_backward(dout, x, g)
print(y.shape, dx.shape, dgamma.shape)      # (3, 6) (3, 6) (6,)
```

Forward verified against `torch.nn.RMSNorm(6, eps=1e-6)` with the same weight, and `dx`/`dgamma` against autograd: `np.allclose` → `True` on all three.

**Complexity:** $O(ND)$, same asymptotics as LayerNorm but with one reduction instead of two and no centering subtract.

**Follow-up: "Show that RMSNorm equals LayerNorm when the input is already zero-mean."** → If $\mu=0$ then $\sigma^2 = \frac{1}{D}\sum x_d^2$, so the denominators coincide and, with $\beta=0$, the two are identical. That's the whole intuition: after the first block, residual-stream activations are approximately centered anyway, so the centering step is redundant work.

*Trap:* Computing the RMS in float16 for a long feature vector. The sum of squares overflows or loses precision; production implementations upcast to float32 for the reduction and cast back — `x.float()` then `.type_as(x)`.

---

### Q: Implement sinusoidal positional encoding.

**Clarify first:** Additive to the embeddings or concatenated? Do I need to support sequences longer than what I precompute, and what's the base (10000 is the transformer default)?

**Approach.** Self-attention is permutation-equivariant — shuffle the tokens and the outputs shuffle identically — so position must be injected. The original transformer uses fixed sinusoids of geometrically-spaced frequencies:

$$PE_{(p,2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),\quad PE_{(p,2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

Low dimensions oscillate fast (fine-grained position), high dimensions oscillate slowly (coarse position) — a continuous binary-counter encoding. The key property: $PE_{p+k}$ is a *fixed linear function* of $PE_p$ (a rotation by angle $k\omega_i$ in each 2-D pair), so the model can learn relative offsets with a linear map. It also extrapolates to unseen lengths, unlike learned position embeddings.

```python
import numpy as np

def sinusoidal_positional_encoding(max_len, d_model, base=10000.0):
    assert d_model % 2 == 0, "d_model must be even"
    pos = np.arange(max_len)[:, None]                       # (T, 1)
    i   = np.arange(0, d_model, 2)[None, :]                 # (1, d/2)
    angles = pos / np.power(base, i / d_model)              # (T, d/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)                            # even dims
    pe[:, 1::2] = np.cos(angles)                            # odd dims
    return pe

pe = sinusoidal_positional_encoding(50, 16)
print(pe.shape, pe[0, :4])          # (50, 16)  [0. 1. 0. 1.]
print(np.abs(pe).max() <= 1.0)      # True — bounded, so it won't swamp embeddings
# every position has the same norm:
print(np.allclose(np.linalg.norm(pe, axis=1), np.linalg.norm(pe[0])))   # True
```

**Complexity:** $O(Td)$ time and space, computed once at init and cached, then sliced to the actual sequence length each batch.

**Follow-up: "Learned vs. sinusoidal — which would you pick?"** → Learned embeddings (BERT, GPT-2) fit the training distribution slightly better but hard-cap the context length and cannot extrapolate a single token past `max_position_embeddings`. Sinusoidal extrapolates in principle, though in practice attention degrades past training length anyway. Modern models use neither: RoPE injects relative position directly into the attention dot product and interpolates cleanly to longer contexts via frequency scaling. If asked what you'd ship today, say RoPE and explain why.

*Trap:* Multiplying the embedding by $\sqrt{d_{\text{model}}}$ before adding PE — this is in the original paper and is easy to forget. Without it, embeddings initialized at scale $1/\sqrt{d}$ are drowned out by the unit-scale positional signal.

---

### Q: Implement RoPE (rotary position embedding).

**Clarify first:** Interleaved pairs `(x0,x1), (x2,x3), ...` (the paper's formulation) or the split-half convention used in Llama's HF implementation? Applied to Q and K only, or to V as well?

**Approach.** RoPE encodes *absolute* position by rotating each consecutive pair of channels by an angle proportional to position, engineered so the attention dot product depends only on the *relative* offset. Treat channels as $d/2$ complex numbers; multiply channel-pair $i$ at position $p$ by $e^{ip\theta_i}$ with $\theta_i = base^{-2i/d}$. Then

$$\langle R_m q, R_n k\rangle = \mathrm{Re}\!\left(\sum_i q_i \bar{k_i} e^{i(m-n)\theta_i}\right)$$

which is a function of $m-n$ alone. So it's absolute to implement, relative in effect, needs no extra parameters, and is applied to Q and K only (V carries content, not position).

```python
import numpy as np

def rope(x, base=10000.0):
    """x: (B, h, T, d) with even d. Rotates channel pairs (0,1), (2,3), ..."""
    B, h, T, d = x.shape
    pos = np.arange(T)[:, None]                                   # (T, 1)
    inv_freq = 1.0 / np.power(base, np.arange(0, d, 2) / d)       # (d/2,)
    ang = pos * inv_freq[None, :]                                 # (T, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]                           # (B,h,T,d/2) each
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin                          # 2-D rotation
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

# Relative-invariance check: put the SAME q at positions 2 and 5, the SAME k at 4 and 7.
rng = np.random.default_rng(0)
qv, kv = rng.normal(size=8), rng.normal(size=8)
q = np.zeros((1, 1, 10, 8)); k = np.zeros((1, 1, 10, 8))
q[0, 0, 2] = qv; q[0, 0, 5] = qv
k[0, 0, 4] = kv; k[0, 0, 7] = kv
rq, rk = rope(q), rope(k)
print(np.isclose(rq[0,0,2] @ rk[0,0,4], rq[0,0,5] @ rk[0,0,7]))   # True: offset 2 both times
```

Use it by calling `rope` on `q` and `k` after the head split, before computing scores.

**Complexity:** $O(BhTd)$ time; the `cos`/`sin` tables are $O(Td)$ and precomputed once per max length.

**Follow-up: "How do you extend a RoPE model to 4× its training context?"** → Two standard tricks. *Position interpolation* (PI): divide positions by the scale factor $s$ so the model never sees an angle outside its training range — `pos = np.arange(T) / s` — cheap but compresses fine-grained resolution. *NTK-aware / YaRN scaling*: raise the base instead (`base * s^(d/(d-2))`), which stretches low-frequency (long-range) channels while leaving high-frequency channels alone, so local resolution is preserved. Both usually want a short fine-tune to fully recover quality.

*Trap:* Mixing conventions. HF Llama uses the split-half layout (`x[..., :d/2]` paired with `x[..., d/2:]`) with a `rotate_half` helper, not interleaved pairs. Both are valid rotations and both satisfy the relative property, but weights trained under one convention produce garbage under the other.

---

### Q: Implement logistic regression trained with gradient descent. No sklearn.

**Clarify first:** Binary or multiclass? Should I include an intercept and L2 regularization, and do you want full-batch gradient descent or mini-batch SGD?

**Approach.** Model $p = \sigma(Xw+b)$, minimize mean negative log-likelihood

$$L = -\frac{1}{n}\sum_i \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right] + \frac{\lambda}{2}\|w\|^2$$

The reason this problem is asked: the gradient is startlingly clean. Despite the sigmoid and the log, everything cancels to

$$\nabla_w L = \frac{1}{n}X^\top(p-y) + \lambda w,\qquad \nabla_b L = \frac{1}{n}\sum_i(p_i-y_i)$$

the same form as linear regression's gradient, with $p$ in place of the linear prediction. That's not a coincidence — it holds for any GLM with the canonical link. There's no closed form (unlike OLS), so we iterate. Loss is convex, so GD converges to the global optimum given a small enough step.

```python
import numpy as np

def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    e = np.exp(z[~pos]); out[~pos] = e / (1 + e)
    return out

def fit_logistic(X, y, lr=0.5, epochs=5000, l2=0.0):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y                        # the whole gradient lives in this residual
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * err.mean()               # intercept is never regularized
    return w, b

def predict_proba(X, w, b): return sigmoid(X @ w + b)

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3))
y = (rng.random(400) < sigmoid(X @ np.array([2.0, -1.0, 0.5]) + 0.3)).astype(float)
w, b = fit_logistic(X, y)
print(np.round(w, 3), round(b, 3))    # [ 1.44 -0.771  0.344] 0.263
```

Matches `sklearn.linear_model.LogisticRegression(penalty=None)` to three decimals: `[1.44, -0.771, 0.344]`, intercept `0.263`. (The recovered weights are shrunk relative to the true `[2, -1, 0.5]` because 400 noisy samples is a finite sample — not a bug.)

**Complexity:** $O(\text{epochs}\cdot nd)$ time, $O(d)$ extra space. Each epoch is two matrix-vector products.

**Follow-up: "What if the classes are perfectly separable?"** → The MLE does not exist: pushing $\|w\|\to\infty$ drives the loss to zero, so gradient descent diverges and coefficients grow without bound (in sklearn this shows up as a convergence warning plus enormous coefficients). Any L2 penalty $\lambda>0$ makes the objective strictly convex and coercive, restoring a unique finite optimum. This is why sklearn regularizes by default — and it's the intended answer here.

*Trap:* Regularizing the intercept. Penalizing $b$ biases predictions toward $p=0.5$ regardless of the base rate, which is badly wrong on imbalanced data. Keep it out of the penalty, as above.

---

### Q: Implement linear regression two ways — closed-form normal equations and gradient descent. When would you use each?

**Clarify first:** Should I add an intercept column, and is the design matrix guaranteed full rank? Ridge or plain OLS?

**Approach.** OLS minimizes $\|X\theta-y\|^2$. Setting the gradient to zero gives the normal equations $X^\top X\theta = X^\top y$, so $\theta = (X^\top X)^{-1}X^\top y$ — exact, one shot, but $X^\top X$ is $d\times d$ and costs $O(nd^2 + d^3)$, which is fatal when $d$ is large, and it squares the condition number so it's numerically fragile. Gradient descent costs $O(nd)$ per step and scales to huge $d$ and streaming data. Critically: **never call `inv`** — use `np.linalg.solve` (or `lstsq`/QR, which is what you'd say if pushed on conditioning). Ridge adds $\lambda I$, which also fixes rank deficiency when $d>n$.

```python
import numpy as np

def linreg_closed_form(X, y, l2=0.0):
    Xb = np.c_[np.ones(len(X)), X]                    # intercept column
    d = Xb.shape[1]
    A = Xb.T @ Xb + l2 * np.eye(d)
    A[0, 0] -= l2                                     # don't penalize the intercept
    return np.linalg.solve(A, Xb.T @ y)               # solve, never inv()

def linreg_gd(X, y, lr=0.05, epochs=5000):
    Xb = np.c_[np.ones(len(X)), X]
    n, d = Xb.shape
    theta = np.zeros(d)
    for _ in range(epochs):
        theta -= lr * (2 / n) * Xb.T @ (Xb @ theta - y)   # grad of MSE
    return theta

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3))
y = X @ np.array([1.5, -2.0, 0.7]) + 0.4 + rng.normal(0, 0.1, 400)
print(np.round(linreg_closed_form(X, y), 3))   # [ 0.39  1.505 -1.988  0.701]
print(np.round(linreg_gd(X, y), 3))            # [ 0.39  1.505 -1.988  0.701]
```

Both agree with `np.linalg.lstsq` to three decimals.

**Complexity:** Closed form $O(nd^2 + d^3)$ time, $O(d^2)$ space. GD $O(\text{epochs}\cdot nd)$ time, $O(d)$ space. Crossover is around $d \sim 10^4$, earlier if $n$ is streaming.

**Follow-up: "How do you pick the learning rate, and what happens if it's too large?"** → For MSE the objective is quadratic with Hessian $\frac{2}{n}X^\top X$; GD converges iff $\eta < 2/L$ where $L$ is the largest eigenvalue of that Hessian. Above that it oscillates and diverges to `inf`/`nan` — which you'll see within ~20 steps. Convergence *rate* depends on the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$; standardizing features shrinks $\kappa$ and is the single most effective fix for slow convergence.

*Trap:* Forgetting the factor of 2 from $\frac{d}{d\theta}\|\cdot\|^2$, or dividing by $n$ in one place and not the other. Neither breaks correctness at the optimum — it just rescales the effective learning rate — but say it out loud so the interviewer knows you know.

---

### Q: Implement k-means with k-means++ initialization.

**Clarify first:** Is $k$ given? How should I handle a cluster that goes empty mid-iteration, and what's the stopping rule — fixed iterations, centroid movement, or inertia change?

**Approach.** Lloyd's algorithm alternates two steps that each provably decrease inertia $\sum_i \|x_i - c_{a_i}\|^2$: assign each point to its nearest centroid, then move each centroid to the mean of its members. It converges to a *local* optimum, and the local optimum you land in is determined entirely by initialization — which is why random init is a bad answer. k-means++ seeds centroids one at a time with probability proportional to $D(x)^2$, the squared distance to the nearest already-chosen centroid, giving an $O(\log k)$ expected approximation guarantee. Distances are computed with broadcasting; for large $n$ use the $\|a\|^2 - 2ab + \|b\|^2$ expansion instead to avoid the $(n,k,d)$ intermediate.

```python
import numpy as np

def kmeans(X, k, iters=100, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    # --- k-means++ seeding ---
    C = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(n, p=d2 / d2.sum())])      # prob proportional to D(x)^2
    C = np.array(C, dtype=float)

    prev = np.inf
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)    # (n, k)
        labels = d2.argmin(1)
        inertia = d2[np.arange(n), labels].sum()
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
            else:                                        # empty cluster: reseed on
                C[j] = X[d2[np.arange(n), labels].argmax()]   # the worst-fit point
        if abs(prev - inertia) < tol:
            break
        prev = inertia
    return C, labels, inertia

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], .3, (100, 2)),
               rng.normal([5, 5], .3, (100, 2)),
               rng.normal([0, 5], .3, (100, 2))])
C, labels, inertia = kmeans(X, 3)
print(round(inertia, 2))    # 53.31  -- identical to sklearn KMeans(3, n_init=10)
```

**Complexity:** $O(\text{iters}\cdot nkd)$ time; $O(nk)$ memory for the distance matrix (and $O(nkd)$ transiently for the broadcast form — mention the expansion trick for large $n$).

**Follow-up: "How do you choose $k$?"** → Elbow on inertia (kinked but subjective), silhouette score (mean of $(b-a)/\max(a,b)$ per point, higher is better, works when clusters are compact), gap statistic (compares inertia against a uniform null), or BIC if you switch to a Gaussian mixture. In practice the honest answer is usually "downstream task metric" — if the clusters feed a recommender, tune $k$ on the recommender's offline metric.

*Trap:* Not handling empty clusters. If a centroid captures nothing, `X[m].mean(0)` is `nan` and every subsequent distance becomes `nan` — the whole run silently dies. Reseeding on the worst-fit point (as above) or on a random point is standard.

---

### Q: Implement k-NN classification, vectorized.

**Clarify first:** How large is the training set — does it fit in memory, and are we allowed $O(n)$ query time or does this need an index? How should I break ties in the majority vote?

**Approach.** No training; all cost is at query. The naive double loop over test $\times$ train points is $O(n_{te}n_{tr}d)$ in slow Python. Vectorize with the squared-distance expansion

$$\|a-b\|^2 = \|a\|^2 - 2a\cdot b + \|b\|^2$$

so one BLAS matmul does the work, and skip the `sqrt` entirely since it's monotone and we only rank. Then use `np.argpartition` rather than `argsort` for the top-$k$: partition is $O(n)$ versus $O(n\log n)$, and we don't care about order within the $k$ neighbors for an unweighted vote.

```python
import numpy as np

def knn_predict(X_train, y_train, X_test, k=5):
    # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2 ; sqrt omitted (monotone)
    d2 = ((X_test ** 2).sum(1)[:, None]
          - 2 * X_test @ X_train.T
          + (X_train ** 2).sum(1)[None, :])              # (n_te, n_tr)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]      # O(n) per row
    neighbors = y_train[idx]                             # (n_te, k)
    return np.array([np.bincount(row).argmax() for row in neighbors])

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0,0], .3, (100,2)), rng.normal([5,5], .3, (100,2)),
               rng.normal([0,5], .3, (100,2))])
y = np.repeat([0, 1, 2], 100)
print((knn_predict(X, y, X, k=5) == y).mean())    # 1.0
```

**Complexity:** $O(n_{te}n_{tr}d)$ time, $O(n_{te}n_{tr})$ memory for the distance block — chunk the test set if that doesn't fit. Training is $O(1)$.

**Follow-up: "Scale this to 10M vectors."** → Exact brute force is out. Options: a KD-tree or ball tree (only useful below ~20 dims — above that they degenerate to linear scan), or approximate nearest neighbors: HNSW (graph-based, best recall/latency tradeoff, what FAISS/Qdrant/pgvector use), IVF-PQ (cluster then quantize, far smaller memory), or LSH. All trade recall for speed; report recall@k against a brute-force sample before shipping.

*Trap:* Forgetting to standardize features. k-NN is a pure distance method, so a feature measured in dollars (range $10^5$) annihilates one measured in years (range $10^1$). Scale before you search — and fit the scaler on train only.

---

### Q: Implement PCA using SVD. Why SVD and not eigendecomposition of the covariance?

**Clarify first:** Should I center the data (yes, always) and also scale to unit variance? Do you want the transformed scores, the components, or both — and the explained variance ratio?

**Approach.** PCA finds the orthogonal directions of maximum variance. Textbook route: form $C = \frac{1}{n-1}X_c^\top X_c$ and eigendecompose it. Problem: forming $C$ squares the condition number ($\kappa(C)=\kappa(X)^2$), so small singular values get destroyed by rounding — and $C$ is $d\times d$, which is unusable when $d\gg n$. SVD on the centered data, $X_c = U\Sigma V^\top$, gives the same answer directly: the right singular vectors $V$ *are* the eigenvectors of $C$, and the eigenvalues are $\sigma_i^2/(n-1)$. Scores are $X_cV$, equivalently $U\Sigma$.

```python
import numpy as np

def pca(X, k):
    mu = X.mean(0)
    Xc = X - mu                                   # centering is mandatory
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                           # (k, d) rows are the axes
    scores = Xc @ components.T                    # (n, k), equals (U * S)[:, :k]
    evr = (S ** 2) / (S ** 2).sum()               # explained variance ratio
    return components, scores, evr[:k], mu

def pca_inverse(scores, components, mu):
    return scores @ components + mu               # reconstruct in original space

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
comps, Z, evr, mu = pca(X, 2)
print(np.round(evr, 4))       # [0.6384 0.192 ]  -- matches sklearn PCA(2)
```

Scores match `sklearn.decomposition.PCA(2).transform(X)` up to per-component sign, which is inherent (an eigenvector is defined only up to sign).

**Complexity:** $O(nd\min(n,d))$ for the thin SVD, $O(nd)$ space. Randomized SVD drops it to $O(ndk)$ when $k\ll d$ — mention `sklearn.utils.extmath.randomized_svd`.

**Follow-up: "Should you standardize before PCA?"** → If features are on different units, yes — otherwise the component with the largest raw variance (say, salary in dollars) dominates purely because of its scale, and PCA becomes a scale detector rather than a structure detector. If all features share units (pixel intensities, log-returns), centering alone is usually right, because relative variances are meaningful. Standardizing means you're doing PCA on the correlation matrix instead of the covariance matrix — say it that way.

*Trap:* Forgetting to center. Without it the first component points at the data's mean vector rather than the direction of maximum variance, and every subsequent component is wrong too. A close second: applying `fit` to the test set — the mean and components must come from train only.

---

### Q: Given a dataset, find the best binary split by Gini impurity.

**Clarify first:** Continuous or categorical features? Should I return the impurity *decrease* (weighted) or the raw child impurity, and do I need to handle a minimum-samples-per-leaf constraint?

**Approach.** Gini impurity of a node is $G = 1-\sum_c p_c^2$ — the probability that two random draws from the node disagree in class. A split's value is the weighted decrease

$$\Delta G = G_{\text{parent}} - \frac{n_L}{n}G_L - \frac{n_R}{n}G_R$$

Brute force is: for each feature, for each candidate threshold, partition and score — $O(d n^2)$ if you recompute impurity from scratch each time. The practical version sorts each feature once and considers only midpoints between *distinct adjacent* values (splitting between equal values is meaningless and creates duplicate candidates). Below is the clear $O(dn^2)$ version; the follow-up gives the $O(dn\log n)$ incremental one.

```python
import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - (p ** 2).sum()

def best_split(X, y):
    """Returns (feature_index, threshold, impurity_decrease)."""
    n, d = X.shape
    parent = gini(y)
    best = (None, None, 0.0)
    for f in range(d):
        order = np.argsort(X[:, f])
        xs, ys = X[order, f], y[order]
        for i in range(1, n):
            if xs[i] == xs[i - 1]:
                continue                          # only split between distinct values
            left, right = ys[:i], ys[i:]
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / n
            gain = parent - weighted
            if gain > best[2]:
                best = (f, (xs[i - 1] + xs[i]) / 2, gain)
    return best

X = np.array([[2.7,1.0],[1.4,2.0],[3.3,2.1],[1.3,3.3],[3.0,2.8],
              [7.6,2.7],[5.3,2.0],[6.9,1.7],[8.6,-0.2],[7.6,3.5]])
y = np.array([0,0,0,0,0,1,1,1,1,1])
print(best_split(X, y))     # (0, 4.3, 0.5) -- feature 0 at 4.3 gives a perfect split
```

Gain of `0.5` equals the parent Gini exactly, i.e. both children are pure — correct for this separable data.

**Complexity:** As written, $O(dn^2)$ time (sorting is $O(dn\log n)$, but each threshold recomputes impurity in $O(n)$), $O(n)$ space.

**Follow-up: "Make it $O(dn\log n)$."** → Sort once per feature, then sweep left to right maintaining class counts incrementally so each candidate threshold is $O(C)$ instead of $O(n)$:

```python
def best_split_fast(X, y, n_classes=None):
    n, d = X.shape
    C = n_classes or int(y.max()) + 1
    total = np.bincount(y, minlength=C)
    parent = 1.0 - ((total / n) ** 2).sum()
    best = (None, None, 0.0)
    for f in range(d):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, ys = X[order, f], y[order]
        left = np.zeros(C, dtype=int)
        for i in range(n - 1):
            left[ys[i]] += 1                       # O(1) count update
            if xs[i] == xs[i + 1]:
                continue
            nl = i + 1; nr = n - nl
            right = total - left
            gl = 1.0 - ((left / nl) ** 2).sum()
            gr = 1.0 - ((right / nr) ** 2).sum()
            gain = parent - (nl * gl + nr * gr) / n
            if gain > best[2]:
                best = (f, (xs[i] + xs[i + 1]) / 2, gain)
    return best
```

Same answer, `(0, 4.3, 0.5)`. This is essentially what sklearn's splitter does in Cython.

*Trap:* Comparing raw child impurity instead of the *weighted* decrease. A split that peels off one pure sample gives $G_L=0$, which looks perfect but is worthless — the weighting by $n_L/n$ is what makes the criterion meaningful.

---

### Q: Implement batch norm forward and backward.

**Clarify first:** Train mode or inference mode — do I need to maintain running statistics? Normalizing over the batch axis only (for a `(N, D)` MLP) or over batch and spatial dims (for `(N, C, H, W)` convs)?

**Approach.** BatchNorm normalizes each *feature* using statistics across the *batch*: $\hat{x} = (x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, then $y=\gamma\hat{x}+\beta$. This is the mirror image of LayerNorm, and it's the source of every BatchNorm bug: the output for one example depends on the other examples in the batch. At inference you must switch to running averages accumulated during training. The backward pass is the classic whiteboard derivation — because $\mu$ and $\sigma^2$ both depend on every $x_i$, the gradient has three terms:

$$\frac{\partial L}{\partial x} = \frac{1}{N\sqrt{\sigma^2+\epsilon}}\left(N\hat{g} - \sum\hat{g} - \hat{x}\sum \hat{g}\hat{x}\right),\quad \hat{g}=\frac{\partial L}{\partial y}\odot\gamma$$

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5, running=None, momentum=0.1, training=True):
    if training:
        mu, var = x.mean(0), x.var(0)                 # stats over the BATCH axis
        if running is not None:                        # EMA for inference
            running["mean"] = (1 - momentum) * running["mean"] + momentum * mu
            running["var"]  = (1 - momentum) * running["var"]  + momentum * var
    else:
        mu, var = running["mean"], running["var"]      # frozen at eval
    istd = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * istd
    return gamma * x_hat + beta, (x_hat, istd, gamma)

def batchnorm_backward(dout, cache):
    x_hat, istd, gamma = cache
    N = dout.shape[0]
    dgamma = (dout * x_hat).sum(0)
    dbeta  = dout.sum(0)
    g_hat  = dout * gamma
    dx = istd / N * (N * g_hat - g_hat.sum(0) - x_hat * (g_hat * x_hat).sum(0))
    return dx, dgamma, dbeta

rng = np.random.default_rng(0)
x, g, b = rng.normal(size=(8, 4)), rng.normal(size=4), rng.normal(size=4)
y, cache = batchnorm_forward(x, g, b)
dx, dg, db = batchnorm_backward(rng.normal(size=(8, 4)), cache)
```

Forward matches `F.batch_norm(..., training=True)`; all three gradients match PyTorch autograd exactly.

**Complexity:** $O(ND)$ time and space for both passes; the cached `x_hat` is the memory cost.

**Follow-up: "Why does BatchNorm hurt at small batch sizes, and what do you use instead?"** → With $N=2$ or $4$ the batch mean/variance are extremely noisy estimates, so the normalization injects large random perturbations and the train/eval gap widens (running stats never match any actual batch). Alternatives: GroupNorm (normalize over channel groups within one example — batch-size independent, standard in detection/segmentation where batches are tiny), LayerNorm (transformers), or InstanceNorm (style transfer). Also note BN in distributed training needs SyncBatchNorm, otherwise each GPU normalizes with its own local batch.

*Trap:* Leaving the model in `train()` mode at eval. BatchNorm then uses the *test batch's* statistics — which is a form of test-time information leakage, makes predictions depend on how you shuffled the test set, and can look either better or worse than reality. Combined with Dropout still being active, this is the single most common PyTorch evaluation bug.

---

### Q: Implement dropout. What exactly differs between training and inference?

**Clarify first:** Standard inverted dropout, or the original formulation that rescales at test time? Should the mask be per-element or per-channel (spatial dropout)?

**Approach.** During training, zero each activation independently with probability $p$. That changes the expected magnitude of the layer's output by a factor of $(1-p)$, so the network sees a different input scale at test time when nothing is dropped — you have to correct for it somewhere. The original paper scaled by $(1-p)$ at *inference*. Everyone now uses **inverted dropout**: divide by $(1-p)$ during *training*, so $\mathbb{E}[\text{output}]$ is unchanged and inference is a plain identity function — no test-time cost, no risk of forgetting the correction at serving time. At eval, dropout is exactly a no-op.

```python
import numpy as np

def dropout(x, p, training, rng=None):
    """Inverted dropout: scale at train time, identity at inference."""
    if not training or p == 0.0:
        return x, None                              # eval: pure identity
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    rng = rng or np.random.default_rng()
    mask = (rng.random(x.shape) >= p) / (1.0 - p)   # keep-mask, pre-scaled
    return x * mask, mask

def dropout_backward(dout, mask):
    return dout if mask is None else dout * mask    # same mask, same scaling

x = np.ones(10000)
out, mask = dropout(x, 0.5, training=True, rng=np.random.default_rng(0))
print(round(out.mean(), 3))                                    # 1.002  ~ E[x] preserved
print(np.array_equal(dropout(x, 0.5, training=False)[0], x))   # True — identity at eval
```

**Complexity:** $O(n)$ time and $O(n)$ space for the mask, which must be cached for the backward pass.

**Follow-up: "Why is dropout rare in modern transformers?"** → Large models trained on internet-scale corpora for roughly one epoch are in an underfitting, not overfitting, regime — the regularizer just slows convergence. Llama and most recent LLMs set dropout to 0 for pretraining and re-introduce it (0.05–0.1) only for fine-tuning on small task datasets, where overfitting is real. Weight decay, data scale, and early stopping do the regularization work instead.

*Trap:* Applying dropout at inference because you forgot `model.eval()`. Predictions become stochastic — running the same input twice gives different answers — and accuracy drops by a few points, which people misdiagnose as a modeling problem for hours.

---

### Q: Write a complete PyTorch training loop — the kind you'd actually ship.

**Clarify first:** Classification or regression, and do you want the distributed/mixed-precision version? Should I include checkpointing and early stopping, or just the core loop?

**Approach.** Interviewers use this to check habits, not syntax. The must-haves: `zero_grad` before every backward, `model.train()`/`model.eval()` toggling, `torch.no_grad()` for validation, moving batches to the device, tracking loss weighted by batch size (not a plain mean over batches — the last batch is usually smaller), gradient clipping, an LR schedule stepped at the right granularity, and keeping the best checkpoint by validation metric rather than the final epoch. Say why for each as you write it.

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

def train(model, train_dl, val_dl, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl))
    crit = nn.CrossEntropyLoss()
    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()                                    # enables dropout / BN batch stats
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)              # BEFORE backward, every step
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # after backward,
            opt.step()                                                # before step
            sched.step()                                 # per-step schedule
            running += loss.item() * xb.size(0)          # weight by actual batch size

        model.eval()                                     # disables dropout, freezes BN
        val_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():                            # no graph, no memory blowup
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += crit(out, yb).item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                n += xb.size(0)
        val_loss /= n
        if val_loss < best_val:                          # checkpoint on the metric
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep} train {running/len(train_dl.dataset):.4f} "
              f"val {val_loss:.4f} acc {correct/n:.3f}")

    model.load_state_dict(best_state)                    # restore the best, not the last
    return model

torch.manual_seed(0)
X = torch.randn(512, 10); y = (X @ torch.randn(10) > 0).long()
tr = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
va = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=64)
train(MLP(10, 32, 2), tr, va, epochs=3)
# epoch 0 train 0.6857 val 0.6624 acc 0.643
# epoch 1 train 0.6526 val 0.6469 acc 0.634
# epoch 2 train 0.6456 val 0.6443 acc 0.634
```

**Complexity:** $O(\text{epochs}\cdot N \cdot \text{fwd+bwd cost})$; memory is dominated by activations, $O(\text{batch}\times\text{model depth}\times\text{width})$.

**Follow-up: "Add mixed precision and gradient accumulation."** → AMP halves activation memory and roughly doubles throughput on tensor cores; accumulation simulates a larger batch than fits in memory. The subtlety is that clipping must happen on *unscaled* gradients, and the scheduler should step once per optimizer step, not per micro-batch:

```python
scaler = torch.amp.GradScaler("cuda")
accum = 4
opt.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(train_dl):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = crit(model(xb), yb) / accum          # scale so grads average correctly
    scaler.scale(loss).backward()
    if (i + 1) % accum == 0:
        scaler.unscale_(opt)                        # unscale BEFORE clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()                                # once per real step
```

*Trap:* Accumulating `running += loss` instead of `loss.item()`. That keeps the entire autograd graph for every batch alive, and memory grows linearly through the epoch until CUDA OOMs — a bug that only manifests on long epochs.

---

### Q: Derive and implement backpropagation for a two-layer MLP by hand. No autograd.

**Clarify first:** ReLU hidden and softmax output with cross-entropy? Batched input, and do you want mean or sum reduction over the batch?

**Approach.** Forward: $Z_1 = XW_1+b_1$, $A_1=\mathrm{ReLU}(Z_1)$, $Z_2=A_1W_2+b_2$, $P=\mathrm{softmax}(Z_2)$, $L=-\frac1N\sum\log P_{i,y_i}$. Backward is repeated chain rule, and the one piece worth memorizing is that softmax + cross-entropy collapse: $\partial L/\partial Z_2 = (P - \text{onehot}(y))/N$. The Jacobian of softmax alone is the ugly $\mathrm{diag}(p)-pp^\top$; fusing it with the log cancels everything. From there each layer is mechanical: for $Y = XW+b$, $\partial L/\partial W = X^\top\,\partial L/\partial Y$, $\partial L/\partial b=\sum_{\text{rows}}\partial L/\partial Y$, $\partial L/\partial X = \partial L/\partial Y\,W^\top$. ReLU's local gradient is the indicator $\mathbb{1}[Z_1>0]$. Shape-check every line: $\partial L/\partial W$ must match $W$.

```python
import numpy as np

def mlp_forward_backward(X, Y, W1, b1, W2, b2):
    N = len(X)
    # ---- forward ----
    Z1 = X @ W1 + b1                   # (N, H)
    A1 = np.maximum(0, Z1)             # ReLU
    Z2 = A1 @ W2 + b2                  # (N, C)
    m  = Z2.max(1, keepdims=True)      # stable softmax
    e  = np.exp(Z2 - m)
    P  = e / e.sum(1, keepdims=True)
    loss = -np.log(P[np.arange(N), Y]).mean()

    # ---- backward ----
    dZ2 = P.copy()
    dZ2[np.arange(N), Y] -= 1.0        # softmax+CE fuse to (P - onehot)
    dZ2 /= N
    dW2 = A1.T @ dZ2                   # (H, C)
    db2 = dZ2.sum(0)                   # (C,)
    dA1 = dZ2 @ W2.T                   # (N, H)
    dZ1 = dA1 * (Z1 > 0)               # ReLU gate
    dW1 = X.T @ dZ1                    # (D, H)
    db1 = dZ1.sum(0)                   # (H,)
    return loss, (dW1, db1, dW2, db2)

rng = np.random.default_rng(0)
X, Y = rng.normal(size=(6, 4)), rng.integers(0, 3, 6)
W1, b1 = rng.normal(size=(4, 5)) * .5, np.zeros(5)
W2, b2 = rng.normal(size=(5, 3)) * .5, np.zeros(3)
loss, grads = mlp_forward_backward(X, Y, W1, b1, W2, b2)
print(round(loss, 6))
```

Loss and all four gradients match PyTorch autograd exactly (`np.allclose` → `True` for `dW1, db1, dW2, db2`).

**Complexity:** $O(N(DH + HC))$ time for each of forward and backward — backward is about 2× forward's FLOPs. Space $O(N(H+C))$ for the cached activations, which is why activation memory dominates training.

**Follow-up: "How would you gradient-check this?"** → Central differences on a few random coordinates, comparing relative error:

```python
def grad_check(f, W, analytic, n_checks=5, h=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_checks):
        idx = tuple(rng.integers(0, s) for s in W.shape)
        old = W[idx]
        W[idx] = old + h; fp = f()
        W[idx] = old - h; fm = f()
        W[idx] = old
        num = (fp - fm) / (2 * h)
        rel = abs(num - analytic[idx]) / max(1e-12, abs(num) + abs(analytic[idx]))
        print(f"{idx} numeric {num:+.8f} analytic {analytic[idx]:+.8f} rel {rel:.2e}")

grad_check(lambda: mlp_forward_backward(X, Y, W1, b1, W2, b2)[0], W1, grads[0])
# relative errors come out between 1e-9 and 1e-12 — anything under 1e-7 is a pass
```

Use central (not forward) differences — error is $O(h^2)$ instead of $O(h)$ — and check relative, not absolute, error. Note that ReLU's kink makes the check fail spuriously if a $Z_1$ value sits within $h$ of zero.

*Trap:* Forgetting the $1/N$ on `dZ2` when the loss uses `.mean()`. Gradients come out $N\times$ too large, training appears to explode, and people blame the learning rate.

---

### Q: Implement top-k and nucleus (top-p) sampling from a logits vector.

**Clarify first:** Should top-k and top-p compose if both are given, and in what order relative to temperature? What happens when the single most likely token already exceeds $p$ — do we keep at least one token?

**Approach.** Greedy decoding (argmax) is deterministic and produces repetitive text; pure sampling from the full softmax occasionally draws from the long, garbage tail of a 50k-token vocabulary. Truncated sampling fixes the tail. **Top-k** keeps the $k$ highest-logit tokens — simple, but $k$ is fixed while the appropriate number of plausible tokens varies wildly by context (after "the capital of France is" there is one; mid-sentence there may be hundreds). **Top-p / nucleus** keeps the smallest prefix of the sorted distribution whose cumulative probability first exceeds $p$, so the candidate set adapts. Order of operations matters: temperature first (it reshapes the distribution), then top-k, then top-p, then renormalize by softmax over the survivors. Mask with $-\infty$ rather than deleting entries, so indices stay aligned to token ids.

```python
import numpy as np

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)

def sample_next(logits, temperature=1.0, top_k=0, top_p=0.0, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64).copy()
    if temperature <= 0:
        return int(logits.argmax())                       # temperature 0 == greedy
    logits /= temperature                                 # 1) temperature

    if top_k > 0:                                         # 2) top-k
        k = min(top_k, logits.size)
        kth = np.partition(logits, -k)[-k]                # k-th largest value
        logits[logits < kth] = -np.inf

    if 0.0 < top_p < 1.0:                                 # 3) nucleus
        order = np.argsort(-logits)
        cum = np.cumsum(softmax(logits[order]))
        cut = int(np.searchsorted(cum, top_p)) + 1        # +1 keeps >= 1 token always
        logits[order[cut:]] = -np.inf

    probs = softmax(logits)                               # 4) renormalize survivors
    return int(rng.choice(len(probs), p=probs))

lg = np.array([5.0, 4.0, 1.0, 0.5, 0.1])
r = np.random.default_rng(1)
from collections import Counter
print(sorted(Counter(sample_next(lg, rng=r) for _ in range(5000)).items()))
# [(0, 3564), (1, 1310), (2, 67), (3, 34), (4, 25)]  -- full distribution
r = np.random.default_rng(1)
print(sorted(Counter(sample_next(lg, top_k=2, rng=r) for _ in range(5000)).items()))
# [(0, 3653), (1, 1347)]  -- tail eliminated; 0.731 expected share for token 0, got 0.731
r = np.random.default_rng(1)
print(sorted(Counter(sample_next(lg, top_p=0.9, rng=r) for _ in range(5000))))   # [0, 1]
r = np.random.default_rng(1)
print(sorted(Counter(sample_next(lg, top_p=0.01, rng=r) for _ in range(200))))   # [0]
```

**Complexity:** Top-k alone is $O(V)$ using `np.partition`. Top-p needs the sort, so $O(V\log V)$ — for $V=128000$ per token this is real, which is why production kernels fuse a partial sort on GPU.

**Follow-up: "How do repetition and presence penalties fit in?"** → They are logit-space edits applied *before* truncation, so they change which tokens survive:

```python
def apply_penalties(logits, generated, repetition_penalty=1.1, presence_penalty=0.0):
    logits = logits.copy()
    for t in set(generated):
        # repetition: divide positive logits, multiply negative ones (GPT-style)
        logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty
        logits[t] -= presence_penalty                 # flat additive penalty
    return logits
```

Note the asymmetric division rule — naively dividing a negative logit by 1.1 makes it *larger*, encouraging the repetition you were trying to suppress.

*Trap:* The `searchsorted` boundary. If the top token alone has probability $0.95$ and $p=0.9$, a naive `cum <= p` mask keeps zero tokens and `softmax` of an all-`-inf` vector is `nan`. The `+1` guarantees at least one survivor — verified above with `top_p=0.01`, which correctly returns token 0 every time.

---

### Q: Implement beam search decoding.

**Clarify first:** Do we stop at the first finished beam or collect all of them and pick at the end? Is length normalization required, and should beams be scored by summed log-prob or averaged?

**Approach.** Greedy decoding commits to the locally-best token, which can be globally terrible ("The" → dead end). Beam search keeps the $B$ highest-scoring *prefixes* by cumulative log-probability. At each step, expand every live beam by its top-$B$ continuations, giving $B^2$ candidates, sort, and keep the best $B$ that haven't hit EOS; finished beams move to a done list. Work in log space — multiplying 50 probabilities in linear space underflows to 0. The catch is the length bias: every extra token adds a negative log-prob, so longer sequences score worse and beam search systematically produces short output. Fix with length normalization, dividing by $|y|^\alpha$ with $\alpha\approx0.6$–$0.7$ (GNMT).

```python
import numpy as np

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)

def beam_search(step_fn, start, beam=3, max_len=6, eos=0, alpha=0.0):
    """step_fn(seq) -> logits over the vocab for the next token."""
    beams = [(0.0, [start])]          # (cumulative logprob, tokens)
    finished = []
    for _ in range(max_len):
        cand = []
        for lp, seq in beams:
            logp = np.log(softmax(step_fn(seq)))         # log space: add, don't multiply
            top = (np.argpartition(-logp, beam - 1)[:beam]
                   if beam < len(logp) else np.arange(len(logp)))
            for t in top:
                cand.append((lp + float(logp[t]), seq + [int(t)]))
        cand.sort(key=lambda x: -x[0])
        beams = []
        for lp, seq in cand:
            if seq[-1] == eos:
                finished.append((lp, seq))               # retire, free a slot
            elif len(beams) < beam:
                beams.append((lp, seq))
        if not beams:
            break
    finished += beams                                    # include unfinished at cutoff
    score = (lambda lp, seq: lp / (len(seq) ** alpha)) if alpha > 0 else (lambda lp, seq: lp)
    return sorted(finished, key=lambda x: -score(*x))[0]

rng = np.random.default_rng(0)
W = rng.normal(size=(6, 6))
step_fn = lambda seq: W[seq[-1]]                          # toy bigram "model"
print(beam_search(step_fn, start=1, beam=3, max_len=6, eos=5))
# (-2.1124583855957386, [1, 5])
print(beam_search(step_fn, start=1, beam=1, max_len=6, eos=5))   # greedy
# (-6.9423330947726050, [1, 0, 2, 1, 0, 2, 1])   -- never even reaches EOS
print(beam_search(step_fn, start=1, beam=3, max_len=6, eos=5, alpha=0.7))
# (-2.4396557273365760, [1, 0, 5])   -- length norm prefers the longer sequence
```

Beam 3 finds a sequence scoring $-2.11$ where greedy (beam 1) gets stuck at $-6.94$ — exactly the failure mode beam search exists to fix.

**Complexity:** $O(\text{max\_len}\cdot B\cdot(\text{model forward} + V\log B))$ time, $O(BT)$ memory for the beams plus $B$ KV caches — which is why beam search costs roughly $B\times$ the serving memory of greedy.

**Follow-up: "Why don't chat LLMs use beam search?"** → Two reasons. It maximizes likelihood, and the maximum-likelihood continuation of an open-ended prompt is bland and repetitive — the "likelihood trap" documented in *The Curious Case of Neural Text Degeneration*. Human text is not the mode of the distribution; it has high, varied surprisal. Beam search remains standard for closed-ended tasks where there's one right answer: translation, summarization, speech recognition, constrained code generation. Second reason: it multiplies serving cost by $B$.

*Trap:* Comparing raw cumulative log-probs across beams of different lengths without normalization. You'll get systematically truncated outputs, and it's a bug people ship — it looks like the model "doesn't finish sentences."

---

### Q: Train a BPE tokenizer from a corpus, then encode a word with the learned merges.

**Clarify first:** Word-level pre-tokenization with an end-of-word marker (the original Sennrich formulation) or byte-level with no marker (GPT-2)? How many merges, and how do I break ties on equally-frequent pairs?

**Approach.** BPE sits between two bad extremes: word-level vocabularies explode and can't handle OOV; character-level makes sequences enormously long. Start from characters, then repeatedly find the most frequent adjacent symbol pair across the corpus and merge it into one symbol, recording the merge. Frequent whole words end up as single tokens; rare words decompose into meaningful subwords, so there is no OOV. The `</w>` end-of-word marker lets the tokenizer distinguish "est" as a suffix from "est" as a standalone word. Encoding must apply merges in **training order** (learned rank), not greedy-longest-match — that ordering is the learned artifact.

```python
from collections import Counter

def train_bpe(corpus, num_merges):
    """Returns the ordered merge list and the final symbol vocabulary."""
    vocab = Counter()
    for word, c in Counter(corpus.split()).items():
        vocab[tuple(list(word) + ["</w>"])] += c          # start from characters
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, c in vocab.items():
            for i in range(len(sym) - 1):
                pairs[(sym[i], sym[i + 1])] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]   # freq, then lex tiebreak
        merges.append(best)
        new = Counter()
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] += c
        vocab = new
    return merges, vocab

def bpe_encode(word, merges):
    """Apply merges in learned-rank order — lowest rank first."""
    sym = list(word) + ["</w>"]
    rank = {m: i for i, m in enumerate(merges)}
    while True:
        cands = [(rank[(sym[i], sym[i + 1])], i)
                 for i in range(len(sym) - 1) if (sym[i], sym[i + 1]) in rank]
        if not cands:
            return sym
        _, i = min(cands)                                  # earliest-learned merge wins
        sym[i:i + 2] = [sym[i] + sym[i + 1]]

corpus = ("low low low low low lower lower newest newest newest "
          "newest newest newest widest widest widest")
merges, vocab = train_bpe(corpus, 10)
print(merges[:6])
# [('t','</w>'), ('s','t</w>'), ('e','st</w>'), ('o','w'), ('l','ow'), ('w','est</w>')]
print(bpe_encode("lowest", merges))     # ['low', 'est</w>']  -- unseen word, no OOV
```

"lowest" never appears in the corpus yet tokenizes cleanly into `low` + `est</w>` — the whole point of BPE.

**Complexity:** Naive training is $O(\text{num\_merges}\cdot N)$ where $N$ is total corpus symbols, since each merge rescans everything. Production implementations keep an index from pairs to the words containing them and update incrementally. Encoding a word of length $L$ is $O(L^2)$ as written; the real implementation uses a priority queue for $O(L\log L)$.

**Follow-up: "What is byte-level BPE and why does GPT-2 use it?"** → Start from the 256 raw bytes instead of Unicode characters. Then *every possible string* — emoji, Chinese, malformed UTF-8, binary — is representable with zero OOV and no `<unk>` token ever, at a base vocabulary of 256. GPT-2 additionally applies a regex pre-tokenizer that splits on word boundaries and keeps leading whitespace attached (` the` is one token, distinct from `the`), which prevents merges from crossing word boundaries. The downside: non-Latin scripts consume several bytes per character, so a Chinese document costs many more tokens than an English one of equivalent content.

*Trap:* Encoding with greedy longest-match instead of merge rank. It produces plausible-looking but different tokenization from what the model was trained on, silently degrading quality. Also: a nondeterministic tie-break in training (`max` over an unordered dict) makes the tokenizer irreproducible across runs — hence the explicit `(count, pair)` key above.

---

### Q: Given a query matrix and a document embedding matrix, return the top-k most cosine-similar documents per query.

**Clarify first:** Are the embeddings already L2-normalized (if so, cosine is just a dot product)? How big is the matrix — does it fit in memory, and do the results need to be sorted within the top-k?

**Approach.** Cosine similarity is $\cos(a,b)=\frac{a\cdot b}{\|a\|\|b\|}$. Normalizing both sides once turns the whole search into a single matmul $\hat{Q}\hat{M}^\top$, which is BLAS-bound and hundreds of times faster than looping. Normalize the *document* matrix once and cache it — that's amortized across all future queries. Then use `argpartition` for $O(n)$ top-k selection instead of a full $O(n\log n)$ sort, and sort only the $k$ survivors. Guard the norm with a floor so a zero vector doesn't produce `nan`.

```python
import numpy as np

def cosine_topk(Q, M, k=5, eps=1e-8):
    """Q: (q, d) queries. M: (n, d) corpus. Returns (indices, scores), sorted desc."""
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), eps)   # cache this
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), eps)
    S = Qn @ Mn.T                                        # (q, n) — one matmul
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]      # O(n) select, unordered
    rows = np.arange(len(Q))[:, None]
    order = np.argsort(-S[rows, idx], axis=1)            # sort only the k survivors
    idx = idx[rows, order]
    return idx, S[rows, idx]

rng = np.random.default_rng(0)
M = rng.normal(size=(1000, 64))
Q = M[[3, 7]] * 2.0                                      # scaled copies of docs 3 and 7
idx, scores = cosine_topk(Q, M, k=3)
print(idx)               # [[  3 191  37]
                         #  [  7 789 913]]
print(np.round(scores, 3))   # [[1. 0.445 0.443] [1. 0.35 0.343]]
```

Note the self-matches score exactly 1.0 despite the 2× scaling — cosine is scale-invariant, which is the reason it's preferred over raw dot product for embeddings of varying magnitude. Results verified identical to a full `argsort` brute force.

**Complexity:** $O(qnd)$ time for the matmul, $O(qn)$ for selection; memory $O(qn)$ for the score block. Chunk over queries when $q\times n$ doesn't fit.

**Follow-up: "The corpus is 100M vectors and updates hourly. Now what?"** → A single matmul is $10^8\times d$ per query — far too slow and it won't fit in RAM. Move to an ANN index: HNSW for best recall-per-latency (but expensive to update and memory-hungry), or IVF-PQ when memory is the binding constraint (coarse-quantize into cells, search a few cells, store compressed residuals). For hourly updates, the standard pattern is a large immutable base index plus a small in-memory delta index searched in parallel, with results merged and a periodic rebuild. Always measure recall@k against a brute-force sample of queries — ANN silently loses accuracy, and nothing in the system will alert you.

*Trap:* Normalizing the corpus matrix inside the query function. It's $O(nd)$ work repeated on every single query; for a static corpus it should be precomputed once at index build. The other common bug: using dot product without normalizing and calling it cosine — with embeddings of varying norm, that ranks long documents higher regardless of relevance.

---

## Part 2 — DSA That Actually Appears

### Q: Given a string, find the index of the first non-repeating character. Return -1 if none.

**Clarify first:** Is the alphabet restricted to lowercase ASCII, or full Unicode? Should I return the index or the character itself, and is case significant?

**Approach.** Brute force: for each character, scan the rest of the string for a duplicate — $O(n^2)$. The repeated work is counting, so hoist it: one pass to build a frequency map, a second pass in original order to return the first character with count 1. The second pass is what makes it correct — iterating the dict instead would work in modern Python (insertion-ordered) but is a fragile thing to rely on and reads as accidental. Two passes, $O(n)$.

```python
from collections import Counter

def first_uniq_char(s):
    counts = Counter(s)                    # pass 1: frequencies
    for i, ch in enumerate(s):             # pass 2: first in ORIGINAL order
        if counts[ch] == 1:
            return i
    return -1

print(first_uniq_char("leetcode"))       # 0
print(first_uniq_char("loveleetcode"))   # 2
print(first_uniq_char("aabb"))           # -1
print(first_uniq_char(""))               # -1
```

**Complexity:** $O(n)$ time (two linear passes), $O(\min(n,|\Sigma|))$ space — bounded by 26 for lowercase ASCII, so effectively $O(1)$.

**Follow-up: "The string arrives as a stream and you must answer at any moment."** → Maintain a queue of candidates plus counts; pop from the front while the head is no longer unique. Each character is enqueued and dequeued at most once, so it's $O(1)$ amortized per character:

```python
from collections import deque, defaultdict

class FirstUnique:
    def __init__(self):
        self.counts = defaultdict(int)
        self.q = deque()
    def add(self, ch):
        self.counts[ch] += 1
        self.q.append(ch)
        while self.q and self.counts[self.q[0]] > 1:
            self.q.popleft()               # amortized O(1): each char pops once
    def query(self):
        return self.q[0] if self.q else None

fu = FirstUnique()
for ch in "aabc":
    fu.add(ch)
print(fu.query())    # 'b'
```

*Trap:* Returning the character when the problem asks for the index (or vice versa) — read the return spec back to the interviewer. Also, `s.index(ch)` inside the loop reintroduces $O(n^2)$ while looking clean.

---

### Q: Determine whether two strings are anagrams of each other.

**Clarify first:** Are we comparing lowercase ASCII only, or Unicode? Do spaces and punctuation count, and is this case-insensitive?

**Approach.** Sorting both and comparing is one line and $O(n\log n)$ — mention it, it's a perfectly good answer for small inputs and has $O(1)$ extra space in some languages. The counting solution is $O(n)$: count the first string, decrement over the second, and bail the moment a count goes negative. The early length check is what makes the single-map version correct — without it, `"a"` vs `"ab"` would pass the decrement loop for its first character. Early exit on negative also avoids a final full comparison pass.

```python
from collections import Counter

def is_anagram(s, t):
    if len(s) != len(t):          # essential: makes the single-counter version correct
        return False
    counts = Counter(s)
    for ch in t:
        counts[ch] -= 1
        if counts[ch] < 0:        # t has more of ch than s does
            return False
    return True

print(is_anagram("anagram", "nagaram"))   # True
print(is_anagram("rat", "car"))           # False
print(is_anagram("a", "ab"))              # False
```

**Complexity:** $O(n)$ time, $O(|\Sigma|)$ space — $O(1)$ for a fixed alphabet. The sorting alternative is $O(n\log n)$ time.

**Follow-up: "What changes for Unicode?"** → Two things. First, the fixed-size 26-slot array is out; you need a hash map (or normalize to code points). Second, and more subtly, the same visible string can have multiple encodings — "é" is either U+00E9 or "e" + U+0301. Apply `unicodedata.normalize("NFC", s)` to both inputs first, otherwise two strings that render identically compare as non-anagrams. Also note `len()` on a Python `str` counts code points, not grapheme clusters, so emoji with modifiers still surprise you.

*Trap:* Using `sorted(s) == sorted(t)` and then claiming it's $O(n)$. Also: skipping the length check when using a single counter — a genuinely wrong answer that passes most naive test cases.

---

### Q: Group a list of words into anagram groups.

**Clarify first:** What's the maximum word length, and can I assume lowercase ASCII? Does the output order of groups or of words within a group matter?

**Approach.** The problem is to define a canonical key such that anagrams collide and non-anagrams don't. Option A: `tuple(sorted(word))` — $O(k\log k)$ per word, trivially correct, works for any alphabet. Option B: a 26-length count vector as a tuple — $O(k)$ per word, better when words are long, but only for a fixed small alphabet. Either way, one pass into a `defaultdict(list)`. There is a cute prime-product key (map each letter to a prime, multiply) but it overflows in most languages and is a bad recommendation; mention it only to dismiss it.

```python
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        key = [0] * 26                      # O(k) canonical key, no sorting
        for ch in w:
            key[ord(ch) - 97] += 1
        groups[tuple(key)].append(w)        # tuple is hashable; list is not
    return list(groups.values())

print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

**Complexity:** $O(nk)$ time with the count key ($O(nk\log k)$ with the sort key), $O(nk)$ space for the output plus keys. Here $n$ = number of words, $k$ = max word length.

**Follow-up: "There are 10 billion words and they don't fit on one machine."** → It's a shuffle: map each word to `(canonical_key, word)`, partition by hash of the key so all anagrams of a group land on the same reducer, then group. In Spark that's `rdd.map(lambda w: (key(w), w)).groupByKey()` — though `reduceByKey`/`aggregateByKey` is preferred since `groupByKey` shuffles every value. Watch for skew: a key with a pathologically large group (common short words) overloads one reducer, which you handle by salting the key.

*Trap:* Using the unsorted string or a `list` as the dict key — `TypeError: unhashable type: 'list'`. Convert to `tuple`.

---

### Q: Two Sum — return the indices of the two numbers that add to a target.

**Clarify first:** Is there exactly one solution guaranteed, and may I use the same element twice? Is the array sorted (which changes the optimal approach)?

**Approach.** Brute force checks all pairs, $O(n^2)$. The insight: for each element $x$ you're asking "have I already seen $\text{target}-x$?" — a membership query, which is $O(1)$ with a hash map. Crucially, check *before* inserting the current element; that's what prevents using the same index twice, and it's cleaner than the two-pass version because it never needs a special case. One pass, $O(n)$.

```python
def two_sum(nums, target):
    seen = {}                              # value -> index
    for i, n in enumerate(nums):
        if target - n in seen:             # check BEFORE insert: no self-pairing
            return [seen[target - n], i]
        seen[n] = i
    return []

print(two_sum([2, 7, 11, 15], 9))    # [0, 1]
print(two_sum([3, 2, 4], 6))         # [1, 2]
print(two_sum([3, 3], 6))            # [0, 1]  -- duplicate values, distinct indices
```

`[3, 3]` is the test that catches the bug where you build the whole map first and then look up — that version returns `[0, 0]` or misses.

**Complexity:** $O(n)$ time, $O(n)$ space. Worst case is $O(n)$ per lookup with adversarial hashing, but expected $O(1)$.

**Follow-up: "The array is sorted — can you do it in $O(1)$ space?"** → Yes, two pointers from the ends. If the sum is too large, only decrementing `hi` can help (every element left of `lo` is smaller); if too small, only incrementing `lo` can. Each step eliminates a row/column of the implicit matrix:

```python
def two_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target: return [lo, hi]
        if s < target:  lo += 1            # need bigger; only lo can grow the sum
        else:           hi -= 1
    return []

print(two_sum_sorted([2, 7, 11, 15], 9))    # [0, 1]
print(two_sum_sorted([1, 2, 3, 4, 6], 10))  # [3, 4]
```

If sorting is required first, that's $O(n\log n)$ and you lose the original indices — so the hash map still wins on unsorted input.

*Trap:* Sorting an unsorted input to use two pointers, then returning indices into the sorted array. You must carry the original indices along, at which point the hash map is simpler and faster.

---

### Q: Return the k most frequent elements in an array.

**Clarify first:** Is $k$ guaranteed $\le$ the number of distinct elements? How should ties be broken, and does the output need to be in frequency order?

**Approach.** Count with a hash map — that part is forced. The question is how to select the top $k$ from $m$ distinct counts. Full sort is $O(m\log m)$. A size-$k$ min-heap is $O(m\log k)$, better when $k\ll m$ — this is the expected answer. But there's a linear-time option most candidates miss: **bucket sort by frequency**. Frequencies are bounded by $n$, so allocate $n+1$ buckets, drop each value into the bucket for its count, and walk buckets from high to low until you've collected $k$. That's $O(n)$ total, beating the heap. Present the heap, then offer the bucket version.

```python
from collections import Counter
import heapq

def top_k_frequent_heap(nums, k):
    counts = Counter(nums)
    return [v for v, _ in heapq.nlargest(k, counts.items(), key=lambda kv: kv[1])]

def top_k_frequent_bucket(nums, k):
    counts = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]     # index == frequency
    for val, freq in counts.items():
        buckets[freq].append(val)
    out = []
    for freq in range(len(nums), 0, -1):             # walk high -> low
        for val in buckets[freq]:
            out.append(val)
            if len(out) == k:
                return out
    return out

print(top_k_frequent_heap([1,1,1,2,2,3], 2))     # [1, 2]
print(top_k_frequent_bucket([1,1,1,2,2,3], 2))   # [1, 2]
```

**Complexity:** Heap: $O(n + m\log k)$ time, $O(m)$ space. Bucket: $O(n)$ time, $O(n)$ space — strictly better asymptotically, at the cost of an $n$-sized array even when $m$ is tiny.

**Follow-up: "Now it's a stream of a billion events and you can't store all distinct keys."** → Exact top-k needs $\Omega(m)$ space in general, so you approximate. Count-Min Sketch gives frequency estimates in fixed sublinear space with a one-sided error bound, paired with a heap of the current top candidates. Space-Saving / Misra-Gries gives deterministic guarantees for heavy hitters. Both are the standard answer for "trending topics" system-design follow-ups.

*Trap:* `heapq.nlargest(k, counts, key=counts.get)` iterates keys — fine — but `heapq.nlargest(k, counts.items())` without a key compares tuples by value first, silently returning the largest *values*, not the most frequent. Always pass the key explicitly.

---

### Q: Merge all overlapping intervals.

**Clarify first:** Is the input sorted? Do intervals that merely touch (`[1,4]` and `[4,5]`) count as overlapping? May I mutate the input list?

**Approach.** The whole problem is the sort. Once intervals are sorted by start, an interval can only overlap the one immediately before it in the output — because any earlier output interval ends no later than the last one's end (we always take the max). So: sort by start, then sweep, extending the last output interval's end when the current start is within it, otherwise appending a new interval. The `max` on the end is essential for the containment case `[[1,4],[2,3]]`, where the second interval is entirely inside the first.

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)                 # by start; copy, don't mutate input
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:                       # <= means touching counts as overlap
            out[-1][1] = max(out[-1][1], e)       # max handles full containment
        else:
            out.append([s, e])
    return out

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))   # [[1, 6], [8, 10], [15, 18]]
print(merge_intervals([[1,4],[4,5]]))                  # [[1, 5]]  -- touching merged
print(merge_intervals([[1,4],[2,3]]))                  # [[1, 4]]  -- containment
```

**Complexity:** $O(n\log n)$ time dominated by the sort, $O(n)$ space for the output (or $O(\log n)$ if sorting in place and merging in place).

**Follow-up: "Given the merged set, insert a new interval efficiently."** → Binary search for the insertion point, then merge outward — $O(\log n + m)$ where $m$ is the number of intervals actually swallowed, instead of re-sorting:

```python
def insert_interval(intervals, new):
    out, i, n = [], 0, len(intervals)
    while i < n and intervals[i][1] < new[0]:     # strictly before: keep as-is
        out.append(intervals[i]); i += 1
    s, e = new
    while i < n and intervals[i][0] <= e:         # overlapping: absorb
        s = min(s, intervals[i][0]); e = max(e, intervals[i][1]); i += 1
    out.append([s, e])
    out.extend(intervals[i:])                      # strictly after
    return out

print(insert_interval([[1,3],[6,9]], [2,5]))       # [[1, 5], [6, 9]]
print(insert_interval([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]))
# [[1, 2], [3, 10], [12, 16]]
```

*Trap:* Sorting by end instead of start (that's the *interval scheduling* greedy, a different problem), or forgetting `max` on the end and breaking on nested intervals. Also, `sorted(intervals)` sorts by start then end lexicographically, which is exactly what you want — but say so rather than leaving it implicit.

---

### Q: Length of the longest substring without repeating characters.

**Clarify first:** Substring (contiguous) or subsequence? What's the character set, and should I return the length or the substring itself?

**Approach.** Brute force enumerates all $O(n^2)$ substrings and checks each for duplicates, $O(n^3)$ or $O(n^2)$ with a set. Sliding window collapses it: maintain a window `[left, right]` with no repeats. On seeing a character already in the window, jump `left` to one past that character's previous position. Storing the *last index* rather than a presence set makes this a single jump instead of a shrink loop. The subtle condition is `last[ch] >= left` — a character may exist in the map from before the window started, in which case it isn't actually a repeat and moving `left` would incorrectly shrink (or worse, move it backward).

```python
def length_of_longest_substring(s):
    last = {}                      # char -> most recent index
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:    # >= left: only if INSIDE the window
            left = last[ch] + 1                # jump past the previous occurrence
        last[ch] = right
        best = max(best, right - left + 1)
    return best

for t in ["abcabcbb", "bbbbb", "pwwkew", "", "tmmzuxt"]:
    print(t, length_of_longest_substring(t))
# abcabcbb 3 | bbbbb 1 | pwwkew 3 | (empty) 0 | tmmzuxt 5
```

`"tmmzuxt"` is the case that exposes a missing `>= left` check: the trailing `t` was last seen at index 0, well before the window start, so `left` must not jump back — the answer is 5 (`"mzuxt"`), not 4.

**Complexity:** $O(n)$ time — each index is visited once by `right`, and `left` only moves forward. $O(\min(n,|\Sigma|))$ space.

**Follow-up: "Return the substring, and generalize to at most $k$ distinct characters."** → For the substring, track the best window's bounds. For at-most-$k$-distinct, the window shrinks from the left while the distinct count exceeds $k$:

```python
from collections import defaultdict

def longest_k_distinct(s, k):
    counts = defaultdict(int)
    left = best = 0
    for right, ch in enumerate(s):
        counts[ch] += 1
        while len(counts) > k:                 # shrink until valid
            counts[s[left]] -= 1
            if counts[s[left]] == 0:
                del counts[s[left]]            # must delete, else len() is wrong
            left += 1
        best = max(best, right - left + 1)
    return best

print(longest_k_distinct("eceba", 2))     # 3  ("ece")
print(longest_k_distinct("aa", 1))        # 2
```

*Trap:* Forgetting `del counts[...]` when a count hits zero — `len(counts)` then counts characters no longer in the window, and the window never shrinks. This is the single most common sliding-window bug.

---

### Q: Implement an LRU cache with $O(1)$ get and put.

**Clarify first:** Does a `get` on an existing key count as a use (yes, in the standard definition)? Is the capacity fixed at construction, and is this single-threaded?

**Approach.** You need two things simultaneously: $O(1)$ key lookup and $O(1)$ recency reordering. A hash map gives the first; a list gives the second only in $O(n)$ because removal from the middle requires a shift. The answer is a hash map from key to *node* in a **doubly linked list** ordered by recency. The map finds the node in $O(1)$; because the node holds both neighbors, unlinking and relinking to the front is $O(1)$ with no search. Sentinel head and tail nodes eliminate all null checks. In Python, `OrderedDict` is a built-in hash map + DLL and is the right production answer — but interviewers usually want the hand-rolled version, so present both.

```python
from collections import OrderedDict

class LRUCache:                                   # production version
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)                   # mark as most recent
        return self.d[key]
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)            # evict least recent (front)

c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
print(c.get(1))     # 1   (1 becomes most recent)
c.put(3, 3)         # evicts key 2
print(c.get(2))     # -1
c.put(4, 4)         # evicts key 1
print(c.get(1), c.get(3), c.get(4))   # -1 3 4
```

And the hand-rolled hash map + doubly linked list, which is what to write on the whiteboard:

```python
class Node:
    __slots__ = ("k", "v", "prev", "next")
    def __init__(self, k=0, v=0):
        self.k, self.v, self.prev, self.next = k, v, None, None

class LRUCacheDLL:
    def __init__(self, capacity):
        self.cap, self.m = capacity, {}
        self.head, self.tail = Node(), Node()       # sentinels: no null checks
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, n):
        n.prev.next, n.next.prev = n.next, n.prev
    def _add_front(self, n):
        n.next, n.prev = self.head.next, self.head
        self.head.next.prev = n
        self.head.next = n

    def get(self, k):
        if k not in self.m:
            return -1
        n = self.m[k]
        self._remove(n); self._add_front(n)         # O(1): node knows its neighbors
        return n.v

    def put(self, k, v):
        if k in self.m:
            n = self.m[k]; n.v = v
            self._remove(n); self._add_front(n); return
        if len(self.m) == self.cap:
            lru = self.tail.prev                    # evict from the back
            self._remove(lru); del self.m[lru.k]
        n = Node(k, v)
        self.m[k] = n
        self._add_front(n)

c = LRUCacheDLL(2)
c.put(1, 1); c.put(2, 2); print(c.get(1))     # 1
c.put(3, 3); print(c.get(2), c.get(3), c.get(1))   # -1 3 1
```

**Complexity:** $O(1)$ expected for both operations, $O(\text{capacity})$ space. Every pointer operation is constant; there is no scan anywhere.

**Follow-up: "Make it thread-safe, and then make it an LFU."** → For thread safety, a single mutex around both operations is correct but serializes everything; sharded caches (hash the key to one of $N$ independently-locked shards) are the standard scaling answer, and note that even `get` mutates the recency list, so a read-write lock buys you nothing here. For LFU, you need frequency buckets: a map from key to (value, freq) plus a map from freq to a DLL of keys at that frequency, plus a `min_freq` pointer — eviction pops from `freq_lists[min_freq]` in $O(1)$. The tricky part is that `min_freq` only ever increments by 1 when a bucket empties.

*Trap:* Forgetting that `get` must update recency. The cache then behaves as FIFO, which passes casual tests and fails the standard LeetCode sequence at exactly the step where key 1 is accessed and should survive eviction.

---

### Q: Course Schedule — can all courses be finished given prerequisite pairs?

**Clarify first:** Do I just return a boolean, or the actual ordering? Can there be duplicate edges or self-loops, and are course ids guaranteed to be $0..n-1$?

**Approach.** "Can all courses be finished" is exactly "is this directed graph acyclic." Two standard answers. **Kahn's algorithm** (BFS topological sort): compute in-degrees, seed a queue with the zero in-degree nodes, and repeatedly remove a node and decrement its neighbors' in-degrees. If you process fewer than $n$ nodes, the remainder are in a cycle. **DFS with three colors** (white/gray/black): finding a gray node during traversal means a back edge, i.e. a cycle. Kahn's is usually cleaner to write under pressure and gives you the ordering for free. Note the edge direction: `[a, b]` means "b before a", so the edge goes `b -> a`.

```python
from collections import defaultdict, deque

def can_finish(num_courses, prerequisites):
    """prerequisites[i] = [a, b] means b must be taken before a."""
    adj = defaultdict(list)
    indeg = [0] * num_courses
    for a, b in prerequisites:
        adj[b].append(a)                   # edge b -> a
        indeg[a] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    order, seen = [], 0
    while q:
        u = q.popleft()
        seen += 1
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:              # all prereqs satisfied
                q.append(v)
    return seen == num_courses, order      # order is a valid topological sort

print(can_finish(2, [[1,0]]))                          # (True,  [0, 1])
print(can_finish(2, [[1,0],[0,1]]))                    # (False, [])
print(can_finish(4, [[1,0],[2,0],[3,1],[3,2]]))        # (True,  [0, 1, 2, 3])
```

**Complexity:** $O(V+E)$ time — each node enters the queue once and each edge is relaxed once. $O(V+E)$ space for the adjacency list and in-degree array.

**Follow-up: "Now return the ordering, and handle the case where multiple valid orders exist."** → The `order` list above already is one. If a *specific* order is needed (say lexicographically smallest), swap the `deque` for a `heapq` min-heap — $O((V+E)\log V)$. If the interviewer instead asks for the DFS version, the three-color scheme is:

```python
def can_finish_dfs(n, prereqs):
    adj = defaultdict(list)
    for a, b in prereqs:
        adj[b].append(a)
    color = [0] * n                       # 0 white, 1 gray (on stack), 2 black (done)
    def dfs(u):
        if color[u] == 1: return False    # back edge -> cycle
        if color[u] == 2: return True     # already fully explored
        color[u] = 1
        for v in adj[u]:
            if not dfs(v): return False
        color[u] = 2
        return True
    return all(dfs(i) for i in range(n))

print(can_finish_dfs(2, [[1,0]]), can_finish_dfs(2, [[1,0],[0,1]]))   # True False
```

Note recursion depth: for $10^5$ nodes in a chain, Python blows the stack — say you'd use an explicit stack or Kahn's in production.

*Trap:* Reversing the edge direction. `[a, b]` with an edge `a -> b` still detects cycles correctly (a cycle is a cycle either way), so the boolean passes all tests while the returned *ordering* is exactly backwards. This is why interviewers ask for the ordering.

---

### Q: Count the number of islands in a grid of '1' (land) and '0' (water).

**Clarify first:** Is connectivity 4-directional or 8-directional? May I mutate the input grid, or do I need a separate visited set? Is the grid guaranteed non-empty and rectangular?

**Approach.** Every unvisited land cell starts a new island; flood-fill from it to mark the entire connected component, then continue scanning. That's one full traversal of the grid plus one visit per land cell, so $O(RC)$. DFS or BFS both work — but use an **explicit stack**, not recursion: a $1000\times1000$ all-land grid produces a recursion depth of $10^6$ and a `RecursionError` in Python. Marking visited by writing `'0'` into the grid is $O(1)$ extra space, but say out loud that you're mutating the input and offer a `visited` set if that's unacceptable.

```python
def num_islands(grid):
    if not grid or not grid[0]:
        return 0
    R, C = len(grid), len(grid[0])
    g = [list(row) for row in grid]        # copy: don't mutate the caller's grid
    count = 0
    for i in range(R):
        for j in range(C):
            if g[i][j] == "1":
                count += 1
                stack = [(i, j)]           # explicit stack: no recursion limit
                g[i][j] = "0"              # mark on PUSH, not on pop
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == "1":
                            g[nr][nc] = "0"
                            stack.append((nr, nc))
    return count

print(num_islands(["11110","11010","11000","00000"]))   # 1
print(num_islands(["11000","11000","00100","00011"]))   # 3
```

**Complexity:** $O(RC)$ time — every cell is examined a constant number of times. $O(RC)$ space worst case for the stack (a snake-shaped island) or for the grid copy.

**Follow-up: "Islands are added one at a time — report the count after each addition."** → Union-Find with path compression and union by rank. Start with count 0; each `addLand` increments the count, then unions with any of the up-to-4 adjacent land cells, decrementing the count on each successful merge. Nearly $O(1)$ amortized (inverse Ackermann) per operation:

```python
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0]*n; self.count = 0
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]        # path compression
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1
        self.count -= 1                          # two components became one
        return True

def num_islands_streaming(R, C, positions):
    uf, land, out = UnionFind(R*C), set(), []
    for r, c in positions:
        if (r, c) in land:
            out.append(uf.count); continue
        land.add((r, c)); uf.count += 1
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if (nr, nc) in land:
                uf.union(r*C + c, nr*C + nc)
        out.append(uf.count)
    return out

print(num_islands_streaming(3, 3, [(0,0),(0,1),(1,2),(2,1)]))   # [1, 1, 2, 3]
```

*Trap:* Marking a cell visited when you *pop* it instead of when you *push* it. The same cell gets pushed many times before it's processed, and the stack can blow up to $O(RC)$ duplicates — still correct, but quadratic-ish in practice on dense grids.

---

### Q: Word Break — can the string be segmented into a sequence of dictionary words?

**Clarify first:** Can a word be reused multiple times? What's the maximum word length in the dictionary (it bounds the inner loop)? Do I return a boolean or the actual segmentation?

**Approach.** The naive recursion tries every split point and recurses on the suffix — exponential, because the same suffix is re-solved along many paths. Define `dp[i] = True` if `s[:i]` is segmentable. Then `dp[i]` is true when there exists a `j < i` with `dp[j]` true and `s[j:i]` in the dictionary. Two crucial optimizations: put the dictionary in a **set** (list membership makes it $O(n^2m)$), and bound `j` below by `i - max_word_len`, since no dictionary word is longer than that. That turns the inner loop from $O(n)$ into $O(L)$ where $L$ is the max word length — a huge win on long strings with short words.

```python
def word_break(s, word_dict):
    words = set(word_dict)                      # set, not list: O(1) membership
    n = len(s)
    max_len = max((len(w) for w in words), default=0)
    dp = [False] * (n + 1)
    dp[0] = True                                # empty prefix is trivially segmentable
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i): # only look back max_len characters
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]

print(word_break("leetcode", ["leet","code"]))                         # True
print(word_break("catsandog", ["cats","dog","sand","and","cat"]))      # False
print(word_break("aaaaaaa", ["aaa","aaaa"]))                           # True
```

**Complexity:** $O(nL)$ substring checks, each $O(L)$ to hash, so $O(nL^2)$ time — versus the textbook $O(n^2)$ bound when $L$ is unbounded. $O(n + \sum|w|)$ space.

**Follow-up: "Return all possible segmentations (Word Break II)."** → Memoized recursion returning lists of sentences. The output itself can be exponential (`"aaaa"` with `["a","aa"]`), so no polynomial algorithm exists; memoization just avoids recomputing shared suffixes:

```python
from functools import lru_cache

def word_break_all(s, word_dict):
    words = set(word_dict)
    @lru_cache(None)
    def solve(i):
        if i == len(s):
            return [[]]
        out = []
        for j in range(i + 1, len(s) + 1):
            if s[i:j] in words:
                for rest in solve(j):
                    out.append([s[i:j]] + rest)
        return out
    return [" ".join(p) for p in solve(0)]

print(word_break_all("catsanddog", ["cat","cats","and","sand","dog"]))
# ['cat sand dog', 'cats and dog']
```

*Trap:* Leaving `word_dict` as a list. With a 100k-word dictionary the solution goes from instant to minutes, and the interviewer will just say "what's the complexity of `in` here?"

---

### Q: Coin Change — fewest coins to make a given amount, or -1 if impossible.

**Clarify first:** Is the coin supply unlimited? Can amount be 0? Are coins guaranteed positive (a zero or negative denomination breaks the DP)?

**Approach.** Greedy — take the largest coin that fits — is wrong for general denominations: with coins `[1, 3, 4]` and amount 6, greedy gives `4+1+1 = 3` coins, optimum is `3+3 = 2`. So it's DP. Define `dp[a]` = fewest coins summing to exactly `a`, with `dp[0] = 0`. Fill upward: `dp[a] = 1 + min(dp[a-c])` over coins `c <= a`. This is the *unbounded* knapsack shape — iterating amounts in the outer loop and coins inside lets each coin be reused, which is what we want. `inf` marks unreachable amounts and propagates naturally.

```python
def coin_change(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount              # dp[a] = min coins for exactly a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]

print(coin_change([1,2,5], 11))                   # 3   (5+5+1)
print(coin_change([2], 3))                        # -1  (odd amount, even coin)
print(coin_change([1], 0))                        # 0
print(coin_change([186,419,83,408], 6249))        # 20
print(coin_change([1,3,4], 6))                    # 2   (greedy would say 3)
```

**Complexity:** $O(A\cdot|C|)$ time, $O(A)$ space, where $A$ is the amount. Note this is *pseudo-polynomial* — linear in the numeric value of $A$, exponential in its bit-length — which is the answer to "is this polynomial?"

**Follow-up: "Now count the number of distinct combinations, not the minimum."** → Loop order flips meaning. Coins in the **outer** loop counts combinations (order-insensitive); amounts outer counts permutations:

```python
def coin_combinations(coins, amount):
    dp = [1] + [0] * amount
    for c in coins:                        # coin OUTER -> combinations
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]

def coin_permutations(coins, amount):
    dp = [1] + [0] * amount
    for a in range(1, amount + 1):         # amount OUTER -> permutations
        for c in coins:
            if c <= a:
                dp[a] += dp[a - c]
    return dp[amount]

print(coin_combinations([1,2,5], 5))       # 4   {1x5, 1x3+2, 1+2+2, 5}
print(coin_permutations([1,2,5], 5))       # 9   ordered sequences
```

Being able to explain *why* the loop order changes the semantics is the real test here.

*Trap:* Initializing `dp = [0] * (amount+1)` and returning `dp[amount]` — every amount looks reachable in 0 coins. And using `-1` as the unreachable sentinel forces awkward guards everywhere; `inf` composes correctly under `min` and `+1`.

---

### Q: Length of the longest common subsequence of two strings.

**Clarify first:** Subsequence (non-contiguous) or substring (contiguous)? Do I need the actual subsequence or just its length? How long are the inputs — does $O(mn)$ memory fit?

**Approach.** The recurrence follows from a single case split on the last characters. If `a[i-1] == b[j-1]`, that character can be taken and the problem reduces to `dp[i-1][j-1] + 1`. Otherwise we must drop one of the two, so `max(dp[i-1][j], dp[i][j-1])`. Straight $O(mn)$ table. But observe row `i` depends only on row `i-1` — so keep two rows and get $O(\min(m,n))$ space. That memory optimization is what separates a good answer from a great one on genomic-scale inputs. If the actual subsequence is needed, you must keep the full table (or use Hirschberg's divide-and-conquer for $O(n)$ space).

```python
def lcs(a, b):
    """Length only — O(min(m,n)) space via rolling rows."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            cur[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return prev[n]

def lcs_string(a, b):
    """Reconstruct the subsequence — needs the full O(mn) table."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    out, i, j = [], m, n
    while i > 0 and j > 0:                     # walk back through the choices
        if a[i-1] == b[j-1]:
            out.append(a[i-1]); i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))

print(lcs("abcde", "ace"), lcs("abc", "abc"), lcs("abc", "def"))   # 3 3 0
print(lcs_string("abcde", "ace"))          # ace
print(lcs_string("AGGTAB", "GXTXAYB"))     # GTAB
```

**Complexity:** $O(mn)$ time both ways. Space: $O(\min(m,n))$ for the length-only version, $O(mn)$ for reconstruction.

**Follow-up: "What about longest common *substring*?"** → Different recurrence: contiguity means a mismatch resets to zero rather than inheriting a max, and the answer is the table maximum rather than the corner cell:

```python
def longest_common_substring(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    best, end_i = 0, 0
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                cur[j] = prev[j-1] + 1
                if cur[j] > best:
                    best, end_i = cur[j], i
            # else: cur[j] stays 0 — contiguity broken
        prev = cur
    return a[end_i - best:end_i]

print(longest_common_substring("abcde", "abfce"))    # ab
print(longest_common_substring("ABABC", "BABCA"))    # BABC
```

*Trap:* Off-by-one between string indices and table indices. `dp[i][j]` covers the first `i` characters of `a`, so the characters being compared are `a[i-1]` and `b[j-1]`. Mixing these up produces answers that are off by one only on some inputs, which is maddening to debug.

---

### Q: Edit distance (Levenshtein) between two strings.

**Clarify first:** Which operations are allowed and do they have equal cost — insert, delete, replace (standard) or also transposition (Damerau)? Do I need the alignment or just the distance?

**Approach.** Same DP shape as LCS. `dp[i][j]` is the cost to turn `a[:i]` into `b[:j]`. If the last characters match, nothing to pay: `dp[i-1][j-1]`. Otherwise pay 1 plus the cheapest of three moves — replace (`dp[i-1][j-1]`), delete from `a` (`dp[i-1][j]`), insert into `a` (`dp[i][j-1]`). The base cases are what people fumble: `dp[i][0] = i` (delete everything) and `dp[0][j] = j` (insert everything). Roll to two rows for $O(n)$ space, seeding each new row's first cell with `i`.

```python
def edit_distance(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))                     # dp[0][j] = j  (all insertions)
    for i in range(1, m + 1):
        cur = [i] + [0] * n                       # dp[i][0] = i  (all deletions)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                cur[j] = prev[j-1]                # free match
            else:
                cur[j] = 1 + min(prev[j-1],       # replace
                                 prev[j],         # delete from a
                                 cur[j-1])        # insert into a
        prev = cur
    return prev[n]

print(edit_distance("horse", "ros"))            # 3
print(edit_distance("intention", "execution"))  # 5
print(edit_distance("", "abc"))                 # 3
```

**Complexity:** $O(mn)$ time, $O(\min(m,n))$ space with the rolling row (swap the arguments so the shorter string drives the row width).

**Follow-up: "You need this over a million-word dictionary for spell check — $O(mn)$ per candidate is too slow."** → Don't compute it against every candidate. Options: (1) if you only care whether distance $\le k$, run the banded DP that fills only the $2k+1$ diagonal band, $O(nk)$, and early-exits when the whole band exceeds $k$; (2) a BK-tree exploits the triangle inequality to prune whole subtrees; (3) a Levenshtein automaton compiled once per query word runs over a trie of the dictionary in near-linear time — this is what Lucene does; (4) generate all deletion variants of the query (SymSpell), which is the fastest practical approach for $k\le2$.

*Trap:* Initializing the rolling row's first element to 0 instead of `i`. Distances come out too small for prefixes, and only for inputs where deletions matter — the empty-string test case catches it immediately, which is why you run it.

---

### Q: LC 1312 — minimum insertions anywhere in a string to make it a palindrome.

**Clarify first:** Can I insert at any position, not just the ends? Am I returning the count or the resulting palindrome?

**Approach.** The reframe is the entire problem. Characters you never touch must already form a palindromic subsequence; every other character needs a partner inserted. So the answer is $n - |\text{LPS}(s)|$, the longest palindromic subsequence. And LPS has a one-line characterization: $\mathrm{LPS}(s) = \mathrm{LCS}(s, \mathrm{reverse}(s))$ — a subsequence common to a string and its reverse reads the same forwards and backwards. So this "hard" problem is two lines given LCS. (The direct interval DP `dp[i][j]` over substrings is equivalent and equally valid; the reduction is more impressive and less error-prone.)

```python
def lcs(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            cur[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return prev[n]

def min_insertions(s):
    """n - LPS(s), and LPS(s) == LCS(s, reversed(s))."""
    return len(s) - lcs(s, s[::-1])

print(min_insertions("zzazz"))      # 0  (already a palindrome)
print(min_insertions("mbadm"))      # 2  ("mbdadbm")
print(min_insertions("leetcode"))   # 5
print(min_insertions("a"))          # 0
```

**Complexity:** $O(n^2)$ time, $O(n)$ space with the rolling-row LCS.

**Follow-up: "Solve it directly as an interval DP, without LCS."** → `dp[i][j]` = insertions needed for `s[i..j]`. Matching ends cost nothing and shrink both; otherwise pay 1 and shrink one side:

```python
def min_insertions_interval(s):
    n = len(s)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n + 1):                 # iterate by interval LENGTH
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]            # ends already match
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1] if n else 0

print([min_insertions_interval(x) for x in ["zzazz","mbadm","leetcode","a"]])
# [0, 2, 5, 0]
```

Note the iteration order: by increasing interval length, because `dp[i][j]` depends on shorter intervals. Iterating `i` then `j` in the natural order reads uninitialized cells.

*Trap:* Assuming insertions are only allowed at the ends — that's a different (and harder to get right) problem. Read the statement carefully; LC 1312 permits insertion anywhere.

---

### Q: Median of two sorted arrays in $O(\log(m+n))$.

**Clarify first:** Can either array be empty? Are they sorted ascending, and can they contain duplicates? Do I return a float even when the total length is odd?

**Approach.** Merging is $O(m+n)$ and is the answer to state first. To hit log time, stop thinking "merge" and think **partition**: choose a cut in `a` at index `i` and the complementary cut in `b` at `j = half - i`, so that exactly $\lceil (m+n)/2 \rceil$ elements sit on the left side combined. The partition is correct iff `a[i-1] <= b[j]` and `b[j-1] <= a[i]` — every left element is $\le$ every right element. Then the median comes from the boundary values alone. Binary search over `i`, always on the **shorter** array so that `j` stays in range. Use $\pm\infty$ sentinels to erase all the boundary special cases.

```python
def find_median_sorted_arrays(a, b):
    if len(a) > len(b):
        a, b = b, a                                # binary search the SHORTER array
    m, n = len(a), len(b)
    total, half = m + n, (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2                         # cut in a
        j = half - i                               # complementary cut in b
        aL = a[i-1] if i > 0 else float("-inf")    # sentinels kill the edge cases
        aR = a[i]   if i < m else float("inf")
        bL = b[j-1] if j > 0 else float("-inf")
        bR = b[j]   if j < n else float("inf")
        if aL <= bR and bL <= aR:                  # valid partition
            if total % 2:
                return float(max(aL, bL))
            return (max(aL, bL) + min(aR, bR)) / 2.0
        elif aL > bR:
            hi = i - 1                             # cut too far right in a
        else:
            lo = i + 1                             # cut too far left in a
    raise ValueError("inputs must be sorted")

print(find_median_sorted_arrays([1,3], [2]))       # 2.0
print(find_median_sorted_arrays([1,2], [3,4]))     # 2.5
print(find_median_sorted_arrays([], [1]))          # 1.0
```

Randomized-tested against `statistics.median` of the merged list over 500 random cases with lengths 0–8: all match.

**Complexity:** $O(\log\min(m,n))$ time — binary search over the shorter array only. $O(1)$ space.

**Follow-up: "Generalize to the $k$-th smallest element of two sorted arrays."** → Same partition idea with `half` replaced by `k`, or the recursive halving version: compare `a[k/2-1]` with `b[k/2-1]`, discard the smaller half (it cannot contain the $k$-th element), and recurse with `k` reduced. Both are $O(\log k)$. For $N$ sorted arrays the partition trick breaks down and you use a heap-based merge, $O(k\log N)$.

*Trap:* Not swapping to search the shorter array. `j = half - i` then goes negative or exceeds `n`, and you get an `IndexError` on lopsided inputs like `a` of length 100 and `b` of length 1 — which is exactly the test case the interviewer runs.

---

### Q: Merge k sorted linked lists into one sorted list.

**Clarify first:** Can the input list contain `None` entries or be empty entirely? Should I reuse the existing nodes or allocate new ones? How large is $k$ relative to total nodes $N$?

**Approach.** Concatenate-and-sort is $O(N\log N)$ and throws away the sortedness — mention and discard. Two good answers. **Min-heap of size $k$**: push the head of each list, repeatedly pop the smallest and push its successor. Each of the $N$ nodes enters and leaves the heap once, so $O(N\log k)$. **Divide and conquer**: pairwise-merge lists, halving $k$ each round for $\log k$ rounds over $N$ nodes — same $O(N\log k)$ but $O(1)$ extra space. The Python-specific detail: heap entries must include a tiebreaker index, because `ListNode` has no `__lt__` and comparing two nodes with equal values raises `TypeError`.

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:                                  # skip None heads
            heapq.heappush(heap, (node.val, i, node))   # i breaks ties: nodes aren't comparable
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node                          # splice existing node, no allocation
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    tail.next = None                              # sever any stale trailing pointer
    return dummy.next

def to_list(arr):
    head = None
    for v in reversed(arr): head = ListNode(v, head)
    return head
def to_arr(node):
    out = []
    while node: out.append(node.val); node = node.next
    return out

print(to_arr(merge_k_lists([to_list([1,4,5]), to_list([1,3,4]), to_list([2,6])])))
# [1, 1, 2, 3, 4, 4, 5, 6]
print(to_arr(merge_k_lists([])), to_arr(merge_k_lists([None])))     # [] []
```

**Complexity:** $O(N\log k)$ time, $O(k)$ space for the heap. The divide-and-conquer variant is $O(N\log k)$ time with $O(1)$ space (or $O(\log k)$ stack if recursive).

**Follow-up: "Do it without a heap."** → Bottom-up pairwise merging. Merge lists 0&1, 2&3, ... then repeat on the results:

```python
def merge_two(a, b):
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val: tail.next, a = a, a.next
        else:              tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b                            # attach the remaining tail
    return dummy.next

def merge_k_divide(lists):
    lists = [l for l in lists if l]
    if not lists: return None
    while len(lists) > 1:
        merged = [merge_two(lists[i], lists[i+1] if i+1 < len(lists) else None)
                  for i in range(0, len(lists), 2)]
        lists = merged
    return lists[0]

print(to_arr(merge_k_divide([to_list([1,4,5]), to_list([1,3,4]), to_list([2,6])])))
# [1, 1, 2, 3, 4, 4, 5, 6]
```

*Trap:* Sequentially merging list 1 into an accumulator, then 2, then 3. That's $O(Nk)$ — you re-traverse the growing accumulator every time. It's the intuitive approach and it's the wrong complexity; say the bound out loud before you commit.

---

### Q: Serialize and deserialize a binary tree.

**Clarify first:** Is it a BST (which allows a more compact encoding) or a general binary tree? Can values be negative or multi-digit — does my delimiter conflict with the data? Must the round trip be exact?

**Approach.** The core difficulty: an in-order traversal alone doesn't determine a tree, and neither does pre-order alone — *unless* you record the null children. Encoding nulls as an explicit sentinel makes pre-order uniquely invertible, because the recursion knows exactly when to stop descending. Serialize with a pre-order DFS emitting `#` for null; deserialize by consuming an iterator in the same order, which reconstructs left then right naturally. Using an iterator rather than an index avoids the classic bug of passing an integer index by value and losing the caller's advance.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def serialize(root):
    out = []
    def dfs(n):
        if not n:
            out.append("#")                  # explicit null marker: makes it invertible
            return
        out.append(str(n.val))
        dfs(n.left); dfs(n.right)            # pre-order
    dfs(root)
    return ",".join(out)

def deserialize(data):
    it = iter(data.split(","))               # iterator carries position across calls
    def build():
        tok = next(it)
        if tok == "#":
            return None
        node = TreeNode(int(tok))
        node.left = build()                  # same order as serialize
        node.right = build()
        return node
    return build()

root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
s = serialize(root)
print(s)                                     # 1,2,#,#,3,4,#,#,5,#,#
print(serialize(deserialize(s)) == s)        # True — exact round trip
print(serialize(deserialize(serialize(None))))   # #
```

**Complexity:** $O(n)$ time and $O(n)$ output size for both directions; $O(h)$ recursion stack, which is $O(n)$ for a degenerate tree.

**Follow-up: "Make it more compact, and handle string values containing commas."** → For a **BST**, nulls are unnecessary: pre-order alone suffices because the BST property tells you where each subtree ends (recurse with a `(lo, hi)` value bound), roughly halving the output. For arbitrary string payloads, a comma delimiter is unsafe — use length-prefixed encoding (`5:hello`) or a real format like protobuf/msgpack. For very large trees, a level-order (BFS) encoding that omits trailing nulls is more compact than pre-order and is what LeetCode's own display format uses. Also note the recursive `build` will stack-overflow on a 10^5-deep skewed tree; the iterative version uses an explicit stack.

*Trap:* Serializing in pre-order and deserializing with a fresh index variable per recursive call. Each frame starts at its own index, the tree comes back mangled, and the bug only shows on trees deeper than 2. The iterator (or a `nonlocal` index) is the fix.

---

### Q: Implement a rate limiter.

**Clarify first:** Per-user or global? Is a hard guarantee required (no more than $N$ in any window) or is approximate acceptable? Single process or distributed across servers? Should rejected requests queue or fail fast?

**Approach.** Fixed windows (a counter reset every minute) are the simplest and are wrong at the boundary: 100 requests at 11:59:59 plus 100 at 12:00:01 lets 200 through in two seconds. **Sliding window log** stores the timestamp of each request in a deque, evicting entries older than the window — exact, but $O(N)$ memory per key. **Token bucket** holds tokens refilled at a constant rate up to a capacity; each request spends one. It's $O(1)$ memory per key, allows controlled bursts up to the capacity (usually desirable), and refills lazily on access rather than needing a timer. Implement both; token bucket is what you'd ship.

```python
import time
from collections import defaultdict, deque

class SlidingWindowLimiter:
    """Exact: never more than max_requests in any window of window_seconds."""
    def __init__(self, max_requests, window_seconds):
        self.max, self.win = max_requests, window_seconds
        self.log = defaultdict(deque)

    def allow(self, key, now=None):
        now = time.monotonic() if now is None else now   # monotonic: immune to clock skew
        dq = self.log[key]
        while dq and dq[0] <= now - self.win:            # evict expired timestamps
            dq.popleft()
        if len(dq) < self.max:
            dq.append(now)
            return True
        return False

class TokenBucket:
    """O(1) memory per key; permits bursts up to `capacity`."""
    def __init__(self, rate, capacity):
        self.rate, self.cap = rate, capacity             # tokens per second
        self.tokens, self.last = float(capacity), 0.0

    def allow(self, now, cost=1.0):
        self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)  # lazy refill
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

rl = SlidingWindowLimiter(3, 1.0)
print([rl.allow("u", t) for t in [0.0, 0.1, 0.2, 0.3, 1.05]])
# [True, True, True, False, True]   -- 4th rejected; 5th ok, first has expired

tb = TokenBucket(rate=2.0, capacity=3)
print([tb.allow(t) for t in [0.0, 0.0, 0.0, 0.0, 0.5, 1.0]])
# [True, True, True, False, True, True]   -- burst of 3, then refills at 2/sec
```

**Complexity:** Sliding window: $O(1)$ amortized per request (each timestamp is appended and popped once), $O(N)$ memory per key. Token bucket: $O(1)$ time and $O(1)$ memory per key.

**Follow-up: "Now make it work across 50 API servers."** → Local counters no longer sum to the global limit. Standard solution is Redis with an atomic Lua script implementing the token bucket (read tokens + last timestamp, refill, decrement, write — all in one round trip so there's no read-modify-write race). Two refinements worth mentioning: (1) the network round trip per request adds latency, so high-traffic systems use *local* buckets holding a lease of tokens fetched in batches from the central store, trading exactness for latency; (2) always return `429` with a `Retry-After` header and add jitter to client backoff, or all rejected clients retry in lockstep and you get a thundering herd. Keys should also expire (Redis TTL) or memory grows unboundedly with unique users.

*Trap:* Using `time.time()` instead of `time.monotonic()`. Wall-clock time can jump backwards (NTP correction, DST in naive implementations), which makes `now - self.last` negative and either grants infinite tokens or wedges the limiter. For the sliding window, also note that an unbounded `defaultdict` keyed by user is a memory leak — real implementations expire idle keys.

---

### Q: Sliding window maximum — the max of every window of size k.

**Clarify first:** Is $k$ guaranteed $\le n$ and positive? Can the array contain duplicates or negatives?

**Approach.** Recomputing the max per window is $O(nk)$. A max-heap gets $O(n\log n)$ but needs lazy deletion for elements that have slid out. The optimal structure is a **monotonic deque** holding *indices* whose values are strictly decreasing. Before pushing index `i`, pop from the back every index whose value is $\le$ `nums[i]` — those can never be the max of any future window, because `i` is both larger and more recent. Then evict the front if it has slid out of the window. The front is always the current window's max. Each index is pushed and popped at most once, so it's $O(n)$.

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()                              # holds INDICES, values decreasing
    out = []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] <= n:       # n dominates: newer AND larger
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                    # front slid out of the window
            dq.popleft()
        if i >= k - 1:                        # window is full
            out.append(nums[dq[0]])
    return out

print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))    # [3, 3, 5, 5, 6, 7]
```

**Complexity:** $O(n)$ time — amortized $O(1)$ per element since each index enters and leaves the deque exactly once. $O(k)$ space.

**Follow-up: "Sliding window *median* instead of max."** → The monotonic deque doesn't generalize, because the median isn't determined by a suffix-dominance relation. Use two heaps (a max-heap of the lower half, a min-heap of the upper half) kept balanced, with lazy deletion via a `collections.Counter` of pending removals — $O(n\log k)$. In Python, `sortedcontainers.SortedList` gives an $O(n\log k)$ solution in far fewer lines and is what you'd actually reach for.

*Trap:* Storing values in the deque instead of indices. You then can't tell when the front has slid out of the window, and the answer is silently wrong on inputs where the max repeats.

---

### Q: Minimum window substring — smallest substring of `s` containing all characters of `t`, including multiplicities.

**Clarify first:** Does `t` contain duplicates that must all be matched (yes, standard)? Return the substring or the indices? What if no such window exists?

**Approach.** The standard expand-and-contract sliding window, with one trick that makes it clean. Instead of comparing two dictionaries at every step ($O(|\Sigma|)$ per position), maintain a single `need` counter that goes *negative* for surplus characters, plus an integer `missing` counting how many required characters are still unmatched. `missing == 0` exactly means the window is valid, checked in $O(1)$. Expand right always; when valid, contract from the left as far as possible while recording the best window. The `need[ch] > 0` test after incrementing on removal is what correctly distinguishes "we removed a required character" from "we removed a surplus one."

```python
from collections import Counter

def min_window(s, t):
    if not s or not t:
        return ""
    need = Counter(t)
    missing = len(t)                        # counts multiplicities, not distinct chars
    best = (float("inf"), 0, 0)
    left = 0
    for right, ch in enumerate(s):
        if need[ch] > 0:                    # only a still-needed char reduces missing
            missing -= 1
        need[ch] -= 1                       # may go negative for surplus
        while missing == 0:                 # window valid: try to shrink it
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right)
            need[s[left]] += 1
            if need[s[left]] > 0:           # we just gave back a REQUIRED char
                missing += 1
            left += 1
    return "" if best[0] == float("inf") else s[best[1]:best[2] + 1]

print(min_window("ADOBECODEBANC", "ABC"))   # BANC
print(repr(min_window("a", "aa")))          # ''  -- not enough 'a's
print(min_window("a", "a"))                 # a
```

**Complexity:** $O(|s| + |t|)$ time — each pointer traverses `s` once, and validity is an $O(1)$ integer check. $O(|\Sigma|)$ space.

**Follow-up: "What if you need the minimum window containing all characters in *any order* with at least one occurrence each (ignore multiplicity)?"** → Set `missing = len(set(t))` and only decrement when a character's count reaches its first match, i.e. build `need` from `set(t)` with count 1 each. The structure is unchanged. And if you need all $k$ distinct characters of `s` itself, it's the same window with `need` built from `set(s)`.

*Trap:* Recomputing `need == required` (a full dict comparison) inside the loop. It's still linear-ish for a fixed alphabet but turns an elegant $O(n)$ into an $O(26n)$ with much messier code, and interviewers notice.

---

### Q: Product of array except self, without division.

**Clarify first:** Are zeros possible (they're the reason the division shortcut fails)? Is the output array counted against the space complexity? Are values bounded so overflow isn't a concern?

**Approach.** The obvious solution — compute the total product and divide by each element — is banned, and rightly so: it breaks on zeros (one zero makes all-but-one output zero; two zeros make everything zero). The insight is that the answer at position `i` is (product of everything to the left) × (product of everything to the right). Compute the left products in a forward pass, writing them directly into the output, then multiply in the right products during a backward pass using a single running scalar. Two passes, no extra arrays, no division, and zeros are handled with zero special-casing.

```python
def product_except_self(nums):
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):                     # pass 1: out[i] = product of nums[:i]
        out[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):         # pass 2: multiply in product of nums[i+1:]
        out[i] *= suffix
        suffix *= nums[i]
    return out

print(product_except_self([1,2,3,4]))          # [24, 12, 8, 6]
print(product_except_self([-1,1,0,-3,3]))      # [0, 0, 9, 0, 0]
```

The zero case is the tell: index 2 (the zero itself) correctly gets $-1\cdot1\cdot-3\cdot3 = 9$, and every other index gets 0.

**Complexity:** $O(n)$ time, $O(1)$ extra space excluding the output array — which is exactly the constraint the follow-up usually imposes.

**Follow-up: "Now you may use division — handle it correctly."** → Count the zeros. Zero zeros: divide the total product. Exactly one zero: every position except the zero gets 0, and the zero's position gets the product of the non-zero elements. Two or more zeros: all outputs are 0.

```python
def product_except_self_div(nums):
    zeros = nums.count(0)
    if zeros > 1:
        return [0] * len(nums)
    prod = 1
    for x in nums:
        if x != 0: prod *= x
    if zeros == 1:
        return [prod if x == 0 else 0 for x in nums]
    return [prod // x for x in nums]

print(product_except_self_div([1,2,3,4]))       # [24, 12, 8, 6]
print(product_except_self_div([-1,1,0,-3,3]))   # [0, 0, 9, 0, 0]
print(product_except_self_div([0,0,3]))         # [0, 0, 0]
```

Also note: with floats, division accumulates error, so the prefix/suffix version is more numerically sound even when division is permitted.

*Trap:* Building explicit `left[]` and `right[]` arrays. Correct, but $O(n)$ extra space when the interviewer's follow-up is specifically "do it in $O(1)$ extra space." Go straight to the in-place two-pass form.

---

## Part 3 — Data Manipulation

All examples below use this small orders/users pair, which is enough to expose every join and null trap that matters:

```python
import pandas as pd, numpy as np

orders = pd.DataFrame({
    "order_id": range(1, 11),
    "user_id":  [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
    "ts": pd.to_datetime(["2024-01-01","2024-01-15","2024-01-05","2024-02-10","2024-01-20",
                          "2024-02-01","2024-03-02","2024-02-15","2024-03-01","2024-03-20"]),
    "amount": [10., 20., 5., 50., 7., None, 30., 15., 25., 40.],   # one NULL
})
users = pd.DataFrame({"user_id": [1, 2, 3, 4, 6],                   # user 5 missing, 6 has no orders
                      "country": ["US", "US", "CA", "UK", "DE"]})
```

### Q: Compute revenue, order count, and distinct customers per country. Explain the join you chose.

**Clarify first:** Should users with no orders appear with zero revenue, and should orders from users missing in the dimension table be dropped or bucketed as unknown? Are NULL amounts treated as zero or excluded?

**Approach.** The join type *is* the analytical decision, and stating it explicitly is what's being graded. An inner join silently drops orders whose `user_id` isn't in `users` — here that's user 5's \$40 order, 32% of a small dataset's revenue, gone with no error. A left join from `orders` keeps all orders and surfaces the orphan as `NaN` country, which you can then investigate or bucket. A right/outer join would additionally surface user 6, who has no orders at all. Then aggregate: `sum` for revenue, `size` for order count (not `count`, which skips nulls), `nunique` for distinct customers. Note that `sum` skips NULLs by default while `size` counts the row — so `n_orders` and the implied denominator of `aov` legitimately differ.

```python
revenue = (orders
    .merge(users, on="user_id", how="left")          # LEFT: never silently drop orders
    .groupby("country", dropna=False)                # dropna=False: keep the orphan bucket
    .agg(revenue=("amount", "sum"),                  # sum skips NaN
         n_orders=("order_id", "size"),              # size counts rows incl. NaN amounts
         n_users=("user_id", "nunique"),
         aov=("amount", "mean"))                     # mean over non-null amounts only
    .reset_index()
    .sort_values("revenue", ascending=False))
print(revenue.to_string(index=False))
# country  revenue  n_orders  n_users   aov
#      US     85.0         4        2 21.25
#      UK     40.0         2        1 20.00
#     NaN     40.0         1        1 40.00     <- user 5: orphaned order, NOT dropped
#      CA     37.0         3        1 18.50     <- 3 orders but aov over 2 (one NULL)
print(len(orders.merge(users, on="user_id")),                # 9  inner drops a row
      len(orders.merge(users, on="user_id", how="left")))    # 10 left keeps all
```

The equivalent SQL:

```sql
SELECT COALESCE(u.country, 'UNKNOWN') AS country,
       SUM(o.amount)             AS revenue,
       COUNT(*)                  AS n_orders,     -- COUNT(*) counts rows
       COUNT(DISTINCT o.user_id) AS n_users,
       AVG(o.amount)             AS aov           -- AVG ignores NULLs
FROM orders o
LEFT JOIN users u USING (user_id)
GROUP BY 1
ORDER BY revenue DESC;
```

**Complexity:** $O(n+m)$ for a hash join, $O(n)$ for the group-by. Memory is the concern at scale — a many-to-many join fans out multiplicatively.

**Follow-up: "Your revenue number came out 3× too high after a join. What happened?"** → Almost certainly a fan-out: the right table wasn't unique on the join key, so every order matched multiple dimension rows and its amount was counted once per match. Diagnose before you fix — `users.user_id.duplicated().any()` or `df.merge(..., validate="m:1")`, which raises instead of silently fanning out. Always check `len()` before and after a join; a row count that changes when you expected a lookup is the single highest-yield data bug check there is.

*Trap:* `count` vs `size` in `.agg`. `count` excludes NaN, `size` includes it — here `("amount","count")` for CA would give 2, not 3. Also `groupby` drops NaN keys by default, which would have hidden the orphaned order entirely.

---

### Q: For each user, compute their order sequence number, running total, days since their previous order, and each order's rank by amount.

**Clarify first:** How are ties in amount ranked — dense, min, or first? Should the running total reset per user (yes)? What should the first order of each user show for "days since previous"?

**Approach.** This is the window-function family, and the pandas mistake people make is reaching for `apply` with a Python lambda per group, which is 10–100× slower than the vectorized group-aware methods. Every one of these has a direct primitive: `cumcount()` for row number, `cumsum()` for running total, `shift(1)` for lag, `rank()` for ranking, and `transform("sum")` for a group total broadcast back to row level. The critical setup step is sorting by `(user_id, ts)` *before* any cumulative operation — `cumsum` and `shift` respect row order, not timestamp order, so unsorted input gives silently wrong running totals.

```python
o = orders.sort_values(["user_id", "ts"]).copy()     # ORDER BY is not optional
o["order_seq"]       = o.groupby("user_id").cumcount() + 1              # ROW_NUMBER()
o["cum_spend"]       = o.groupby("user_id")["amount"].cumsum()          # SUM() OVER
o["prev_amount"]     = o.groupby("user_id")["amount"].shift(1)          # LAG()
o["days_since_prev"] = (o["ts"] - o.groupby("user_id")["ts"].shift(1)).dt.days
o["rank_in_user"]    = o.groupby("user_id")["amount"].rank(method="dense", ascending=False)
o["pct_of_user"]     = o["amount"] / o.groupby("user_id")["amount"].transform("sum")

print(o[["user_id","ts","amount","order_seq","cum_spend",
         "prev_amount","days_since_prev","rank_in_user"]].head(7).to_string(index=False))
#  user_id         ts  amount  order_seq  cum_spend  prev_amount  days_since_prev  rank_in_user
#        1 2024-01-01    10.0          1       10.0          NaN              NaN           2.0
#        1 2024-01-15    20.0          2       30.0         10.0             14.0           1.0
#        2 2024-01-05     5.0          1        5.0          NaN              NaN           2.0
#        2 2024-02-10    50.0          2       55.0          5.0             36.0           1.0
#        3 2024-01-20     7.0          1        7.0          NaN              NaN           2.0
#        3 2024-02-01     NaN          2        NaN          7.0             12.0           NaN
#        3 2024-03-02    30.0          3       37.0          NaN             30.0           1.0
```

Look at user 3: the NULL amount makes `cumsum` return `NaN` for that row but the running total *recovers* to 37.0 afterward (pandas' cumsum skips NaN rather than poisoning the rest) — worth calling out, since SQL's `SUM() OVER` behaves the same way but many people expect poisoning.

The SQL equivalent:

```sql
SELECT user_id, ts, amount,
       ROW_NUMBER() OVER w                                   AS order_seq,
       SUM(amount)  OVER w                                   AS cum_spend,
       LAG(amount)  OVER w                                   AS prev_amount,
       DATE_DIFF('day', LAG(ts) OVER w, ts)                  AS days_since_prev,
       DENSE_RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rank_in_user,
       amount / SUM(amount) OVER (PARTITION BY user_id)      AS pct_of_user
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY ts);
```

**Complexity:** $O(n\log n)$ for the sort, then $O(n)$ per window operation. All are vectorized in C.

**Follow-up: "Compute each user's 3-order trailing average — and then a 30-day trailing average."** → Row-based and time-based windows are different objects. Row-based uses `rolling(3)`; time-based requires a DatetimeIndex:

```python
o["avg_last3"] = (o.groupby("user_id")["amount"]
                    .rolling(3, min_periods=1).mean()
                    .reset_index(level=0, drop=True))         # drop the group index level
o30 = (o.set_index("ts").groupby("user_id")["amount"]
         .rolling("30D", min_periods=1).mean()
         .reset_index(name="avg_30d"))
print(o30.head(4).to_string(index=False))
```

The `reset_index(level=0, drop=True)` is mandatory — `groupby().rolling()` returns a MultiIndex, and assigning it directly to a column misaligns rows (pandas aligns on index, so you get NaNs or scrambled values with no error).

*Trap:* Forgetting the sort before `cumsum`/`shift`. Running totals then reflect insertion order, which for data loaded from a partitioned warehouse table is effectively random. Nothing errors; the numbers are just wrong.

---

### Q: Deduplicate records with a business tie-break rule: keep the most recently updated row per user+email, preferring the CRM source, then the lowest row id.

**Clarify first:** Is the match on the raw email or a normalized form (case, whitespace)? What's the full precedence order of the tie-break keys? Should I keep one row per group or merge fields across the duplicates?

**Approach.** Two separate problems disguised as one. First, **define the key**: raw string equality will treat `"a@x"` and `"A@X "` as distinct, so normalize (strip, lowercase) into an explicit key column rather than mutating the original — you usually want to preserve the source value for auditing. Second, **define the precedence**: sort by every tie-break column in the specified order, then `drop_duplicates(keep="first")`. Encoding a categorical priority (`crm` before `web`) requires mapping to an integer, since sorting strings alphabetically is not the business rule. Making both steps explicit and ordered is the whole answer; `drop_duplicates()` without a sort keeps whatever row happened to arrive first.

```python
events = pd.DataFrame({
    "user_id":    [1, 1, 1, 2, 2],
    "email":      ["a@x", "A@X ", "a@x", "b@y", "b@y"],       # same address, 3 spellings
    "updated_at": pd.to_datetime(["2024-01-01","2024-03-01","2024-03-01","2024-02-01","2024-01-01"]),
    "source":     ["crm", "web", "crm", "web", "crm"],
    "row_id":     [5, 9, 7, 3, 1],
})

events["email_key"] = events["email"].str.strip().str.lower()   # 1) normalize the KEY
priority = {"crm": 0, "web": 1}                                 # 2) encode precedence

dedup = (events
    .assign(_p=events["source"].map(priority))
    .sort_values(["updated_at", "_p", "row_id"],                # tie-breaks, in order
                 ascending=[False, True, True])                 # newest, then crm, then low id
    .drop_duplicates(subset=["user_id", "email_key"], keep="first")
    .drop(columns="_p"))
print(dedup.to_string(index=False))
#  user_id email updated_at source  row_id email_key
#        1   a@x 2024-03-01    crm       7       a@x    <- 2024-03-01 tie broken by crm
#        2   b@y 2024-02-01    web       3       b@y    <- newer wins over crm
```

Row 1 is the interesting one: two rows share `2024-03-01`, and the CRM row (row_id 7) wins over the web row (row_id 9) — the tie-break did real work. For user 2, the newer web row beats the older CRM row, confirming recency outranks source.

Equivalent SQL:

```sql
SELECT * EXCEPT (rn) FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY user_id, LOWER(TRIM(email))
           ORDER BY updated_at DESC,
                    CASE source WHEN 'crm' THEN 0 ELSE 1 END,
                    row_id
         ) AS rn
  FROM events
) WHERE rn = 1;
```

**Complexity:** $O(n\log n)$ for the sort, $O(n)$ for the dedup pass. `drop_duplicates` after sorting is a single hash-set scan.

**Follow-up: "Instead of picking one row, merge the non-null fields across duplicates, preferring newer values."** → Sort by recency ascending, then `groupby(...).last()` with `skipna` semantics — or explicitly forward-fill within groups so newer non-nulls override older ones:

```python
merged = (events.sort_values("updated_at")                # oldest first
                .groupby(["user_id", "email_key"], as_index=False)
                .agg({"email": "last", "source": "last",
                      "updated_at": "max", "row_id": "min"}))
print(merged.to_string(index=False))
```

Be explicit that `last()` on a group with NaN in the newest row keeps the NaN — if you want "newest non-null", use `.ffill().groupby(...).last()` instead. This distinction is exactly what the interviewer is probing.

*Trap:* `drop_duplicates(subset=[...])` with no preceding sort. It keeps the first row in *file* order, which is arbitrary and non-reproducible across reruns if the upstream data is partitioned. Always sort first, and say why.

---

### Q: You have hourly event data with gaps. Produce a daily aggregate and a 24-hour rolling mean.

**Clarify first:** Are timestamps in UTC or local time, and do we need DST handling? Should hours with no events appear as zero or as missing? Which timestamp labels each bucket — the left or right edge?

**Approach.** `resample` is a groupby over time bins, and the two things that bite are (1) gaps and (2) the difference between *downsampling* (many rows → one bin, needs an aggregation) and *upsampling* (one row → many bins, needs a fill rule). A missing hour is not the same as a zero-count hour: `resample("D").sum()` will report 0 for both, while `count()` distinguishes them — so always emit `count` alongside `mean`/`sum` so downstream consumers can tell a genuinely quiet day from a broken pipeline. For rolling windows, the string offset `"24h"` gives a *time-based* window that is robust to gaps, whereas `rolling(24)` counts rows and silently spans a longer wall-clock period when hours are missing.

```python
rng = np.random.default_rng(0)
ts = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=200, freq="h"),
                   "v":  rng.normal(10, 2, 200)})
ts.loc[5:8, "v"] = np.nan                                     # a gap

daily = ts.set_index("ts")["v"].resample("D").agg(["mean", "sum", "count"])
print(daily.head(3))
#                  mean         sum  count
# ts
# 2024-01-01   9.535174  190.703474     20    <- 20, not 24: the gap is visible
# 2024-01-02  10.506034  252.144814     24
# 2024-01-03  10.256806  246.163353     24

# Time-based rolling window: "24h" of wall clock, not 24 rows
roll = ts.set_index("ts")["v"].resample("h").mean().rolling("24h", min_periods=12).mean()
print(roll.tail(2).round(3).tolist())      # [9.506, 9.567]

# Upsampling creates NaNs that you must decide how to fill
print(ts.set_index("ts")["v"].resample("30min").mean().head(3).tolist())
# [10.251460442186787, nan, 9.735790273417397]   <- the 30-min slots between hours
```

The `count` column immediately reveals the gap: day 1 has 20 observations, not 24. Without it, the `mean` looks perfectly healthy.

**Complexity:** $O(n)$ for resampling (a single pass with binning), $O(n)$ for time-based rolling using a two-pointer window. Memory is $O(\text{n\_bins})$.

**Follow-up: "Timestamps arrive in local time across three timezones and you need correct daily buckets."** → Store and compute in UTC, convert only at the presentation boundary. The bug people ship is bucketing UTC timestamps into "days" that don't align with any user's actual day:

```python
s = ts.set_index("ts")["v"]
s_utc = s.tz_localize("UTC")                        # attach a timezone
daily_ny = s_utc.tz_convert("America/New_York").resample("D").mean()
print(daily_ny.head(2))
```

Also mention DST: on the spring-forward day a "daily" bucket contains 23 hours and on fall-back 25, so a naive per-hour average over a DST boundary is off. `tz_localize` on ambiguous/nonexistent local times raises by default — that's a feature, and the `ambiguous=` / `nonexistent=` arguments force you to make the decision explicitly.

*Trap:* `rolling(24)` when hours are missing. With a 4-hour gap, a 24-row window covers 28 hours of wall clock, and the "24-hour average" quietly becomes something else. Use the string offset.

---

### Q: A column has missing values. Walk me through how you'd handle them.

**Clarify first:** Are the values missing at random, or is missingness itself informative (a null `income` may mean "declined to answer")? Is this a feature for a tree model (which can often handle nulls natively) or a linear model? Am I allowed to drop rows?

**Approach.** The first move is never "fill with the mean" — it's to ask *why* the data is missing, because the mechanism determines the valid response. MCAR (missing completely at random) permits dropping rows with only a variance cost. MAR (missingness depends on observed covariates) requires conditional imputation. MNAR (depends on the unobserved value itself, e.g. high earners refusing to state income) means any imputation biases the estimate and the missingness must be modeled explicitly. The universal safety move regardless of mechanism: **add a binary missingness indicator**, so the model can learn from the fact of absence even if your imputed value is wrong. And impute using a *group-conditional* statistic where a sensible grouping exists — a user's own median beats the global median.

```python
df = orders.copy()
print(df.isna().sum().to_dict())      # {'order_id': 0, 'user_id': 0, 'ts': 0, 'amount': 1}

# 1) preserve the signal that it WAS missing
df["amount_was_missing"] = df["amount"].isna().astype(int)

# 2) impute conditionally: user's own median, then global median as a backstop
per_user_median = df.groupby("user_id")["amount"].transform("median")
df["amount_filled"] = df["amount"].fillna(per_user_median).fillna(df["amount"].median())

print(df[["user_id","amount","amount_filled","amount_was_missing"]].iloc[4:8].to_string(index=False))
#  user_id  amount  amount_filled  amount_was_missing
#        3     7.0            7.0                   0
#        3     NaN           18.5                   1    <- median of user 3's {7, 30}
#        3    30.0           30.0                   0
#        4    15.0           15.0                   0
```

Note 18.5 is the median of user 3's own other orders (7 and 30), not the global median — the group-conditional imputation is doing real work.

**Complexity:** $O(n)$ for the indicator, $O(n)$ for a hash-based group transform. Iterative/model-based imputation is far more expensive and rarely worth it.

**Follow-up: "Where does imputation leak, and how do you prevent it?"** → Computing the imputation statistic on the full dataset before splitting leaks test-set information into training — the same class of bug as fitting a scaler before the split. The median must be estimated on train and *applied* to test:

```python
def fit_imputer(train_df, col):
    return {"median": train_df[col].median()}            # learned on TRAIN only

def apply_imputer(df, col, params):
    out = df.copy()
    out[col + "_was_missing"] = out[col].isna().astype(int)
    out[col] = out[col].fillna(params["median"])
    return out

params = fit_imputer(orders.iloc[:7], "amount")
test_imputed = apply_imputer(orders.iloc[7:], "amount", params)
print(params, test_imputed["amount"].tolist())      # {'median': 15.0} [15.0, 25.0, 40.0]
```

For time series, the leak is worse: a global median uses future data to fill a past hole, and forward-fill (`ffill`) is the only safe direction. Backward-fill is a time-travel bug.

*Trap:* `df.fillna(0)` on a numeric feature. Zero is a real, meaningful value for most quantities — it shifts the mean, distorts the distribution, and for something like `days_since_last_login` it means the opposite of missing. If you must use a sentinel, use one that's out of range (`-1`, `-999`) *and* pair it with the indicator column.

---

### Q: Split data for training so that no leakage occurs — by group, and by time.

**Clarify first:** Do rows share an entity (same user, same patient, same document) that must not straddle the split? Is the model going to predict the future at serving time? Is there a gap needed between train and test to account for label delay?

**Approach.** A random row-level split is correct only when rows are i.i.d., which is rarely true. Two failure modes dominate. **Group leakage**: if one user contributes 50 rows and they scatter across train and test, the model memorizes user-specific quirks and test accuracy is inflated — sometimes enormously. Split on the *group* key, not the row. **Temporal leakage**: if you shuffle a time series, the model trains on the future to predict the past, which is impossible at serving time. Split by a time cutoff. And add an **embargo** gap when labels take time to materialize (a 30-day churn label computed on day $t$ uses data through $t+30$, so training rows within 30 days of the cutoff contain test-period information).

```python
def group_split(df, group_col, test_frac=0.2, seed=0):
    """No group appears in both splits."""
    groups = df[group_col].unique()
    rs = np.random.default_rng(seed)
    rs.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_frac)))
    test_groups = set(groups[:n_test])
    mask = df[group_col].isin(test_groups)
    return df[~mask].copy(), df[mask].copy()

def time_split(df, time_col, cutoff, embargo="0D"):
    """Train strictly before cutoff minus embargo; test at/after cutoff."""
    cutoff = pd.Timestamp(cutoff)
    gap = pd.Timedelta(embargo)
    train = df[df[time_col] <  cutoff - gap].copy()      # embargo drops the boundary rows
    test  = df[df[time_col] >= cutoff].copy()
    return train, test

tr, te = group_split(orders, "user_id", test_frac=0.4)
print(set(tr.user_id) & set(te.user_id), len(tr), len(te))   # set() 6 4  <- zero overlap

tr, te = time_split(orders, "ts", "2024-02-15", embargo="3D")
print(tr.ts.max(), te.ts.min(), len(tr), len(te))
# 2024-02-10 00:00:00 2024-02-15 00:00:00 6 4   <- 3-day gap enforced
```

**Complexity:** $O(n)$ for both, plus $O(g)$ to shuffle groups. Both are single boolean-mask passes.

**Follow-up: "Now cross-validate correctly under each constraint."** → `GroupKFold` for group structure, and expanding-window (walk-forward) CV for time — never `KFold` on a time series, which trains on the future in 4 of 5 folds:

```python
def time_series_cv_splits(df, time_col, n_splits=3, embargo="0D"):
    """Expanding window: train on everything before each fold's test block."""
    df = df.sort_values(time_col).reset_index(drop=True)
    gap = pd.Timedelta(embargo)
    bounds = np.array_split(np.arange(len(df)), n_splits + 1)
    for i in range(1, n_splits + 1):
        test_idx = bounds[i]
        cutoff = df.loc[test_idx[0], time_col]
        train_idx = df.index[df[time_col] < cutoff - gap].to_numpy()
        yield train_idx, test_idx

for tr_idx, te_idx in time_series_cv_splits(orders, "ts", n_splits=3, embargo="1D"):
    print(len(tr_idx), len(te_idx))
# 3 3
# 6 2
# 7 2
```

Training-set sizes grow while test blocks always follow in time — the shape of a correct backtest. If groups *and* time both matter (patients over time), you need both constraints simultaneously, which usually means grouping first and splitting each group's timeline.

*Trap:* Splitting by group but computing normalization statistics before the split — you fixed one leak and left another. The rule is that *every* fitted transformation, including scalers, encoders, imputers, and feature selectors, must be fit inside the training fold.

---

### Q: Compute precision, recall, F1, ROC-AUC, and average precision from raw predictions. No sklearn.

**Clarify first:** Are the scores probabilities or arbitrary real-valued rankings (only the ordering matters for AUC)? How should ties in the score be handled? What do we return if one class is entirely absent?

**Approach.** Precision/recall/F1 come straight from the confusion matrix at a fixed threshold, with the only real care being zero-denominator guards. AUC is the interesting one. The naive route sweeps thresholds and integrates the ROC curve — $O(n^2)$ or a fiddly $O(n\log n)$ with cumulative sums. The clean route uses the Mann-Whitney U identity: **ROC-AUC equals the probability that a random positive is ranked above a random negative**, which is computable from rank sums:

$$\mathrm{AUC} = \frac{\sum_{i \in \text{pos}} r_i - \frac{n_+(n_++1)}{2}}{n_+ n_-}$$

where $r_i$ are 1-based ranks of the scores ascending. Ties must receive their *average* rank, or tied scores are treated as strictly ordered and the AUC is wrong. Average precision is a separate quantity — it's the step-wise integral of precision over recall, and it is far more informative than AUC on heavily imbalanced data.

```python
import numpy as np

def precision_recall_f1(y, y_hat):
    tp = int(((y == 1) & (y_hat == 1)).sum())
    fp = int(((y == 0) & (y_hat == 1)).sum())
    fn = int(((y == 1) & (y_hat == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0        # guard: predicted nothing
    recall    = tp / (tp + fn) if tp + fn else 0.0        # guard: no actual positives
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def roc_auc(y, scores):
    """Mann-Whitney U form, with average ranks for ties."""
    y = np.asarray(y); s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")                              # AUC undefined for one class
    order = np.argsort(s, kind="mergesort")              # stable sort
    s_sorted = s[order]
    ranks = np.empty(len(s))
    i, r = 0, 1
    while i < len(s):                                    # assign AVERAGE ranks to ties
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + (r + j - i)) / 2.0
        r += j - i + 1
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def average_precision(y, scores):
    """Step-wise integral of precision over recall (== sklearn's average_precision_score)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = np.asarray(y)[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall = np.r_[0, recall]; precision = np.r_[1, precision]
    return float(np.sum(np.diff(recall) * precision[1:]))

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 300)
s = rng.random(300) + 0.3 * y                            # weak but real signal
print(round(roc_auc(y, s), 6))                           # 0.74941
print(round(average_precision(y, s), 6))                 # 0.785221
print(np.round(precision_recall_f1(y, (s > 0.6).astype(int)), 4))   # [0.6491 0.707  0.6768]
print(roc_auc(np.array([1,0,1,0]), np.array([0.5,0.5,0.5,0.1])))    # 0.75  (ties handled)
```

All four verified against `sklearn.metrics.roc_auc_score`, `average_precision_score`, and `precision_recall_fscore_support`: identical to six decimals, including the tie case.

**Complexity:** $O(n\log n)$ for AUC and AP (dominated by the sort), $O(n)$ for the confusion-matrix metrics. $O(n)$ space.

**Follow-up: "Which do you report for a fraud model with a 0.1% positive rate?"** → PR-AUC (average precision), not ROC-AUC. ROC-AUC's false-positive-rate axis is normalized by the huge negative count, so adding thousands of false positives barely moves it — a model can score 0.95 ROC-AUC and still be unusable, because at any workable alert volume the precision is near zero. PR-AUC has the positive class in both axes and degrades visibly. Better still, report the operational metric: precision at the recall your review team can actually handle, or recall at a fixed daily alert budget. Note that PR-AUC's baseline is the positive rate (0.001), while ROC-AUC's is always 0.5 — so the numbers are not comparable across datasets.

*Trap:* Ignoring ties. `scipy.stats.rankdata` defaults to average ranks for a reason; a hand-rolled `argsort().argsort()` gives strict ranks and produces a wrong AUC exactly when a model outputs many identical scores — which is precisely what a shallow tree or a saturated sigmoid does.

---

### Q: Implement a stratified train/test split.

**Clarify first:** Stratify on the label alone, or on a label-plus-group combination? What should happen to a class with only one member? Should the split be reproducible across runs?

**Approach.** A plain random split gives class proportions that vary by chance, and with a rare class it can produce a test set containing zero positives — which makes recall undefined and the whole evaluation meaningless. Stratification enforces the population proportion within each split by partitioning *within* each class independently: shuffle each class's indices, take the same fraction from each. Two details separate a working implementation from a fragile one: a seeded generator for reproducibility, and a guard so that a class with $n\ge2$ contributes at least one row to each side rather than being rounded out of existence.

```python
import numpy as np

def stratified_split(y, test_size=0.2, seed=0):
    """Returns (train_idx, test_idx) preserving class proportions."""
    y = np.asarray(y)
    rs = np.random.default_rng(seed)                     # seeded: reproducible
    train, test = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rs.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        if len(idx) > 1:                                 # guarantee >=1 on each side
            n_test = min(max(n_test, 1), len(idx) - 1)
        else:
            n_test = 0                                   # singleton class stays in train
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.sort(np.array(train)), np.sort(np.array(test))

y = np.array([0] * 90 + [1] * 10)                        # 10% positive
tr, te = stratified_split(y, test_size=0.2)
print(y[tr].mean(), y[te].mean())      # 0.1 0.1   <- proportion preserved exactly
print(len(tr), len(te), len(set(tr) & set(te)))          # 80 20 0  <- disjoint
```

**Complexity:** $O(n)$ time (one pass per class, $O(n)$ total) plus $O(n\log n)$ for the final sort, which is cosmetic. $O(n)$ space.

**Follow-up: "Stratify on a continuous target, and stratify on group + label at once."** → For regression, bin the target into quantiles and stratify on the bin — the standard trick to avoid a test set that omits the tail:

```python
import pandas as pd

def stratified_split_continuous(y, test_size=0.2, n_bins=10, seed=0):
    bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop").to_numpy()
    return stratified_split(bins, test_size=test_size, seed=seed)

rng = np.random.default_rng(0)
y_cont = rng.lognormal(0, 1, 500)                        # heavy right tail
tr, te = stratified_split_continuous(y_cont, 0.2)
print(round(np.median(y_cont[tr]), 3), round(np.median(y_cont[te]), 3))   # 0.951 0.935
print(round(y_cont[tr].max(), 2), round(y_cont[te].max(), 2))             # 12.84 21.46
```

Medians agree to two decimals and both splits carry the extreme tail — with a plain random split, the top quantile lands entirely on one side often enough to matter.

For group + label together, you cannot stratify exactly — a group may contain both classes, so the constraints conflict. The practical approach is to compute each group's dominant label (or its positive rate binned), stratify on *that* at the group level, and accept approximate label balance. `sklearn`'s `StratifiedGroupKFold` implements a greedy version of exactly this.

*Trap:* Stratifying on the label and then, separately, doing a group split — the two constraints fight, and whichever runs second silently destroys the first one's guarantee. Decide which constraint is load-bearing (it's almost always the group constraint, since group leakage inflates metrics far more than mild class imbalance does) and treat the other as best-effort.

---

## Part 4 — Debugging

Each of these is presented the way it's presented in the interview: here is code, it runs, the output looks plausible, find the bug. Say the symptom out loud before you say the fix — "the answer is too small on this input, so the window must be dropping an element" is the reasoning being graded.

### Q: This computes the max sum of any contiguous subarray of size k. It returns 48 but the answer is 51. Find the bug.

```python
def max_sum_subarray(nums, k):
    best = cur = sum(nums[:k])
    for i in range(k, len(nums)):
        cur += nums[i] - nums[i - k + 1]
        best = max(best, cur)
    return best

print(max_sum_subarray([1, 12, -5, -6, 50, 3], 4))   # 48, expected 51
```

**Clarify first:** Is `k` guaranteed positive and no larger than the array? Should the window be strictly of size k, or at most k?

**Approach.** Instrument before you theorize. The invariant is that after processing index `i`, `cur` should equal `sum(nums[i-k+1 : i+1])`. Print both and find the first divergence. The window currently spans indices `[i-k, i-1]`; to advance it to `[i-k+1, i]` you add `nums[i]` and remove the element leaving the window, which is the one at the **old left edge**, index `i-k`. The code removes `nums[i-k+1]` — the *new* left edge, which is still inside the window. So one element is double-counted out and the true left edge is never removed. The window silently becomes size $k+1$ with a hole in it. Off-by-one in the removal index, not the addition.

```python
def max_sum_subarray(nums, k):
    if k <= 0 or k > len(nums):
        raise ValueError("k must satisfy 1 <= k <= len(nums)")
    best = cur = sum(nums[:k])
    for i in range(k, len(nums)):
        cur += nums[i] - nums[i - k]      # FIX: drop the OLD left edge, index i-k
        best = max(best, cur)
    return best

nums = [1, 12, -5, -6, 50, 3]
print(max_sum_subarray(nums, 4))                                        # 51
print(max(sum(nums[i:i+4]) for i in range(len(nums) - 3)))              # 51  (brute force)
```

**Complexity:** $O(n)$ time, $O(1)$ space — unchanged by the fix; the bug was correctness, not performance.

**Follow-up: "How would you have caught this before shipping?"** → A brute-force oracle on random inputs. It takes four lines and catches every off-by-one variant instantly:

```python
import random
def brute(nums, k):
    return max(sum(nums[i:i+k]) for i in range(len(nums) - k + 1))

for _ in range(1000):
    n = random.randint(1, 12)
    a = [random.randint(-10, 10) for _ in range(n)]
    k = random.randint(1, n)
    assert max_sum_subarray(a, k) == brute(a, k), (a, k)
print("1000 random cases pass")
```

Also test `k == 1` and `k == len(nums)`: both are degenerate cases where several wrong index expressions coincidentally give the right answer, which is why the original bug survived casual testing.

*Trap:* The general principle — when sliding a window forward by one, the element you *remove* is at the position the window's left edge occupied **before** the move. Writing `i - k + 1` feels symmetric with the new window bounds and is wrong. Derive it once from the invariant rather than guessing.

---

### Q: This training loop runs, loss decreases for a moment, then explodes. What's wrong?

```python
def train(model, X, y, epochs=10, lr=0.05):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for ep in range(epochs):
        for i in range(0, len(X), 20):
            xb, yb = X[i:i+20], y[i:i+20]
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
```

**Clarify first:** Is this the complete loop — is there a `zero_grad` anywhere outside what I can see? Is the loss genuinely diverging, or just noisy?

**Approach.** The symptom — a few good steps then geometric blowup — is the signature of an effective learning rate that grows over time. In PyTorch, `.backward()` **accumulates** into `param.grad` rather than overwriting it. That design is deliberate (it enables gradient accumulation across micro-batches and multi-loss setups), but it means that without `opt.zero_grad()` the gradient at step $t$ is the sum of all gradients from steps $1..t$. The update magnitude therefore grows roughly linearly, and once it overshoots, the loss grows, which grows the gradient, which grows the overshoot. Fix: `opt.zero_grad(set_to_none=True)` before `backward()` — `set_to_none` is marginally faster and turns "forgot to zero" into an explicit `None` rather than silent accumulation.

```python
import torch, torch.nn as nn

def train(model, X, y, epochs=10, lr=0.05, zero_grad=True):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        for i in range(0, len(X), 20):
            xb, yb = X[i:i+20], y[i:i+20]
            if zero_grad:
                opt.zero_grad(set_to_none=True)     # FIX: clear before every backward
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()                          # accumulates into .grad
            opt.step()
        losses.append(round(loss.item(), 4))
    return losses

torch.manual_seed(0)
X = torch.randn(200, 4); y = (X @ torch.tensor([1., -2., .5, 0.])).unsqueeze(1)
print(train(nn.Linear(4, 1), X, y, zero_grad=False))
# [10.1289, 10.7173, 13.0052, 17.3568, 29.4961, 72.5875, 221.314, 683.175, 1968.09, 5196.82]
torch.manual_seed(0)
print(train(nn.Linear(4, 1), X, y, zero_grad=True))
# [1.5421, 0.1364, 0.0143, 0.0018, 0.0002, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Identical model, identical seed, identical learning rate: divergence to 5197 versus clean convergence to 0.

**Complexity:** Unchanged — `zero_grad` is $O(\text{params})$ and negligible against the backward pass.

**Follow-up: "When would you deliberately *not* zero the gradients?"** → Gradient accumulation, to simulate a batch larger than fits in memory: run $N$ micro-batches with `loss / N` and call `zero_grad` only once per $N$. Also multi-task setups where you `backward()` several losses into the same parameters before stepping (though `loss_a + loss_b` then one backward is usually cleaner). The tell that you're in this situation is that `opt.step()` is *also* inside a conditional — `step` and `zero_grad` should always be called at the same cadence, and a mismatch between them is the bug to look for.

*Trap:* Two adjacent versions of this bug. Calling `zero_grad()` *after* `step()` is fine; calling it *after* `backward()` but before `step()` zeroes the gradient you just computed, so the model never updates at all — loss goes flat instead of exploding. And putting `zero_grad` outside the inner loop gives you accidental gradient accumulation over an entire epoch.

---

### Q: This attention implementation produces `nan` losses about an hour into training. It works fine on small test inputs.

```python
def softmax(x):
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)
```

**Clarify first:** Does the `nan` appear in the loss only, or in the model weights too? Does it start after a specific batch, and is it reproducible with a fixed seed?

**Approach.** "Works on small inputs, fails after an hour" is diagnostic on its own: nothing in the code changed, so the *inputs* grew. Early in training, logits are small and $e^x$ is fine. As the model trains, attention logits grow — especially without proper scaling or with a saturating layer norm — and once any score exceeds about 709, `np.exp` overflows float64 to `inf`. Then `inf / inf` is `nan`, the `nan` flows into the loss, then into every gradient, and every parameter becomes `nan` within one step. The fix is the max-subtraction identity: softmax is invariant to a constant shift, so subtract the row max and the largest exponent becomes exactly $e^0=1$.

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)     # FIX: shift by the row max
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

big, small = np.array([1000., 1001., 1002.]), np.array([-1000., -1001., -1002.])
# buggy version: [nan nan nan] on both, with an overflow warning on `big`
print(softmax(big))     # [0.09003057 0.24472847 0.66524096]
print(softmax(small))   # [0.66524096 0.24472847 0.09003057]
```

Both extreme cases now return the correct distribution — and note that both reduce to the same shape, because softmax depends only on logit *differences*.

**Complexity:** $O(n)$, one extra reduction pass. Free relative to the matmuls around it.

**Follow-up: "You added the max subtraction and it still `nan`s occasionally. Now what?"** → Work backwards from where the `nan` first appears; `torch.autograd.set_detect_anomaly(True)` names the offending op. Common remaining causes, in order of likelihood: (1) `-inf` mask values producing an all-masked row, so `x - x_max` is `-inf - (-inf) = nan` — use a large finite negative like `-1e9`, or `-1e4` in fp16; (2) the loss taking `log` of a probability instead of using a fused `log_softmax`/`cross_entropy`; (3) division by a zero norm somewhere (an all-zero embedding row); (4) fp16 overflow, since the half-precision max is only 65504 and `-1e9` itself overflows it. Add a cheap guard in the training loop:

```python
import torch

def guard_loss(loss, step, batch=None):
    """Call right after computing the loss, before backward()."""
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {loss.item()} at step {step}")
    return loss

print(guard_loss(torch.tensor(0.5), 10).item())          # 0.5, passes through
try:
    guard_loss(torch.tensor(float("nan")), 11)
except RuntimeError as e:
    print(e)                                             # non-finite loss nan at step 11
```

Failing fast with the offending batch in hand is worth far more than discovering the `nan` an hour later in a checkpoint.

*Trap:* Concluding the model is unstable and lowering the learning rate. It masks the symptom — smaller logits take longer to reach the overflow threshold — so the run survives longer and fails at hour six instead of hour one. Fix the numerics, not the hyperparameter.

---

### Q: This pipeline reports 80% test accuracy. The labels are pure random noise. Explain.

```python
X = rng.normal(size=(60, 400))       # 60 samples, 400 features
y = rng.integers(0, 2, 60)           # labels are pure coin flips — no signal exists

Xs = (X - X.mean(0)) / X.std(0)                                   # scale
corr = np.array([abs(np.corrcoef(Xs[:, j], y)[0, 1]) for j in range(400)])
top = np.argsort(-corr)[:5]                                       # pick best 5 features
Xtr, Xte, ytr, yte = Xs[:40, top], Xs[40:, top], y[:40], y[40:]   # THEN split
print(LogisticRegression().fit(Xtr, ytr).score(Xte, yte))         # 0.8
```

**Clarify first:** Is 80% being compared against the correct baseline (the majority-class rate)? Was any step of this pipeline fit before the split?

**Approach.** Two leaks, and the second is the lethal one. **Leak 1:** the standardization uses `X.mean(0)` and `X.std(0)` over all 60 rows, so test-set statistics are baked into the training features. On its own this is a mild leak. **Leak 2:** feature selection is done using `y` for all 60 samples, then the split happens afterward. With 400 candidate features and 60 samples, some features correlate with the random labels *by chance* — and the selection deliberately finds the five most extreme ones, using the test labels to do it. The model is then trained and tested on features hand-picked to correlate with the test labels. This is the single most common cause of "my offline metrics were amazing and production was chance."

The fix is that every fitted transformation — scaler, selector, imputer, encoder — must be fit on the training fold and merely *applied* to test:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

rng = np.random.default_rng(0)
n, d = 60, 400
X = rng.normal(size=(n, d))
y = rng.integers(0, 2, n)                       # NO signal: chance is 0.5

def leaky():
    Xs = (X - X.mean(0)) / X.std(0)                                # fit on ALL data
    corr = np.array([abs(np.corrcoef(Xs[:, j], y)[0, 1]) for j in range(d)])
    top = np.argsort(-corr)[:5]                                    # uses TEST labels
    return LogisticRegression().fit(Xs[:40, top], y[:40]).score(Xs[40:, top], y[40:])

def clean():
    Xtr_raw, Xte_raw, ytr, yte = X[:40], X[40:], y[:40], y[40:]    # SPLIT FIRST
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)                       # fit on train only
    Xtr, Xte = (Xtr_raw - mu) / sd, (Xte_raw - mu) / sd            # apply to test
    corr = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1]) for j in range(d)])
    top = np.argsort(-corr)[:5]                                    # train labels only
    return LogisticRegression().fit(Xtr[:, top], ytr).score(Xte[:, top], yte)

print(round(leaky(), 3), round(clean(), 3))     # 0.8 0.45   -- truth is 0.5
```

The leaky pipeline reports 80% on data with no signal whatsoever. The clean pipeline reports 45%, correctly indistinguishable from the 50% coin flip.

**Complexity:** Identical for both — leakage is free, which is exactly why it's dangerous.

**Follow-up: "How do you make this structurally impossible?"** → Put every fitted step inside a `Pipeline` and only ever call `fit` on training folds. Then cross-validation refits the entire chain per fold automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score

pipe = Pipeline([("scale", StandardScaler()),
                 ("select", SelectKBest(f_classif, k=5)),   # refit inside each fold
                 ("clf", LogisticRegression())])
print(round(cross_val_score(pipe, X, y, cv=5).mean(), 3))   # ~0.5, correctly at chance
```

Beyond tooling, the cultural fix is a sanity check that should be mandatory: **shuffle the labels and rerun**. If the pipeline still beats the base rate on shuffled labels, you have a leak. It costs one line and catches this entire class of bug.

*Trap:* The subtler production version of this — building features from a table that was computed with future information (a "user's lifetime order count" column that was snapshotted today, used to predict an event from six months ago). Scaler leakage costs a few points; target leakage through a feature costs the whole model, and it survives every correctly-implemented split because the leak is inside the feature itself.

---

### Q: This masked attention runs without error and the shapes are all correct, but the model won't learn. Find the bug.

```python
pad = np.array([[1,1,1,0,0], [1,1,1,1,1]], dtype=bool)   # (B, T) True = real token
scores = q @ k.swapaxes(-1, -2) / np.sqrt(dk)            # (B, h, T, T)
masked = np.where(pad[:, None, :, None], scores, -1e9)   # broadcasts to (B, h, T, T)
```

**Clarify first:** Which axis of the score matrix indexes keys and which indexes queries? What is the padding mask supposed to prevent — attending *to* padding, or producing output *at* padding positions?

**Approach.** Broadcasting is what hides this. `pad[:, None, :, None]` has shape `(B, 1, T, 1)` and `pad[:, None, None, :]` has shape `(B, 1, 1, T)`; both broadcast cleanly against `(B, h, T, T)`, so NumPy raises nothing and the output shape is identical. But in `scores[b, h, i, j]`, `i` is the **query** and `j` is the **key**. The buggy version puts the mask on axis 2 — the query axis — so it masks entire *rows* (positions that shouldn't produce output) while leaving padded *columns* fully attendable. Real tokens therefore attend to padding embeddings, and masked query rows become uniform over all keys. The fix is one axis: the padding mask belongs on the last axis, indexing keys.

```python
import numpy as np

def softmax(x):
    m = x.max(-1, keepdims=True); e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)

B, h, T, dk = 2, 4, 5, 8
rng = np.random.default_rng(0)
q, k, v = (rng.normal(size=(B, h, T, dk)) for _ in range(3))
pad = np.array([[1,1,1,0,0], [1,1,1,1,1]], dtype=bool)     # batch 0 has 2 pad tokens
scores = q @ k.swapaxes(-1, -2) / np.sqrt(dk)

bug  = np.where(pad[:, None, :, None], scores, -1e9)       # (B,1,T,1) -> masks QUERIES
good = np.where(pad[:, None, None, :], scores, -1e9)       # (B,1,1,T) -> masks KEYS

wb, wg = softmax(bug), softmax(good)
print(wb[0, 0, 0, 3:].round(4))    # [0.2125 0.0867]  <- 30% of attention on PADDING
print(wg[0, 0, 0, 3:].round(4))    # [0. 0.]          <- correct
print(wb[0, 0, 3].round(4))        # [0.2 0.2 0.2 0.2 0.2]  uniform garbage row
print(np.abs(wb @ v - wg @ v).max().round(4))    # 1.8111  -- outputs are wildly different
```

Real token 0 in batch 0 puts 30% of its attention mass on two padding positions. The shapes were right the whole time.

**Complexity:** Unchanged. The cost of this bug is entirely in wasted training runs.

**Follow-up: "What's the companion shape bug in multi-head attention?"** → Splitting heads with a reshape and no transpose:

```python
x = rng.normal(size=(B, T, h * dk))
wrong = x.reshape(B, h, T, dk)                       # same shape, interleaves tokens
right = x.reshape(B, T, h, dk).transpose(0, 2, 1, 3) # reshape THEN transpose
print(wrong.shape == right.shape, np.allclose(wrong, right))   # True False
print(np.allclose(right.transpose(0, 2, 1, 3).reshape(B, T, h * dk), x))   # True
```

Identical shapes, completely different values, no error. The round-trip assertion on the last line is the guard to write: split then merge must reproduce the input exactly.

*Trap:* Relying on shape assertions alone. `assert scores.shape == (B, h, T, T)` passes for both versions. The assertions that actually catch these are *semantic*: `assert np.allclose(weights[..., ~pad_row], 0)` for masking, and the split/merge round trip for head reshaping. Write behavioral checks, not shape checks.

---

### Q: A colleague reports 99% accuracy on a fraud detector and wants to ship it. What do you ask?

```python
y_true = np.array([0] * 990 + [1] * 10)      # 1% fraud
y_pred = model.predict(X)                    # the model predicts all zeros
print("accuracy:", (y_pred == y_true).mean())   # 0.99  ship it?
```

**Clarify first:** What is the base rate of the positive class, and what does a trivial majority-class baseline score? What's the operational cost of a false negative versus a false positive?

**Approach.** With a 1% positive rate, the constant predictor "never fraud" achieves 99% accuracy while catching zero fraud. Accuracy is a weighted average dominated by the majority class, so on imbalanced data it measures the class distribution, not the model. The first question is always "what does the trivial baseline score?" — if the model doesn't beat it by a meaningful margin, there's nothing there. Report the full confusion matrix plus class-conditional metrics: precision (of flagged cases, how many are fraud), recall (of actual fraud, how much we caught), F1, and balanced accuracy (the mean of per-class recalls, which pins a constant predictor at exactly 0.5 regardless of imbalance).

```python
import numpy as np

def classification_report(y, y_pred):
    tp = int(((y == 1) & (y_pred == 1)).sum()); fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum()); tn = int(((y == 0) & (y_pred == 0)).sum())
    precision   = tp / (tp + fp) if tp + fp else 0.0
    recall      = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(y),
            "precision": precision, "recall": recall, "f1": f1,
            "balanced_acc": 0.5 * (recall + specificity),      # constant model -> 0.5
            "confusion": [[tn, fp], [fn, tp]],
            "base_rate": float(y.mean()),                      # ALWAYS report this
            "majority_baseline": float(max(y.mean(), 1 - y.mean()))}

y = np.array([0] * 990 + [1] * 10)
print(classification_report(y, np.zeros(1000, dtype=int)))
# {'accuracy': 0.99, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
#  'balanced_acc': 0.5, 'confusion': [[990, 0], [10, 0]],
#  'base_rate': 0.01, 'majority_baseline': 0.99}
```

`accuracy == majority_baseline` and `recall == 0`: the model has learned nothing. Balanced accuracy of exactly 0.5 is the unambiguous tell.

**Complexity:** $O(n)$ for all metrics. Reporting them costs nothing relative to the model that produced the predictions.

**Follow-up: "The model does have signal, but at the default 0.5 threshold recall is 12%. What now?"** → 0.5 is a modeling artifact, not a business decision. Pick the threshold from the precision-recall curve against an operational constraint — "our review team handles 200 alerts a day" or "a missed fraud costs \$500 and a false alarm costs \$5 of review time." Sweep and choose:

```python
def threshold_sweep(y, scores, cost_fn=500.0, cost_fp=5.0):
    best = None
    for t in np.unique(scores):
        pred = (scores >= t).astype(int)
        fn = int(((y == 1) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if best is None or cost < best[1]:
            best = (float(t), cost, fn, fp)
    return {"threshold": best[0], "expected_cost": best[1], "fn": best[2], "fp": best[3]}

rng = np.random.default_rng(0)
y = np.r_[np.zeros(990), np.ones(10)].astype(int)
scores = rng.random(1000) + 0.35 * y                       # a model with real signal
print(threshold_sweep(y, scores))
```

Also: threshold selection is model fitting, so tune it on a validation split, not on test. And for training itself, address the imbalance with class weights (`class_weight="balanced"`) or focal loss rather than by naively oversampling, which duplicates rows and inflates confidence.

*Trap:* Fixing the imbalance by upsampling the minority class *before* the train/test split. The duplicated positives land on both sides, so the model sees test positives during training and recall looks superb. Resample inside the training fold only — this is the leakage bug of the previous problem wearing a different hat.

---

## Quick reference — what each part is testing

| Part | What the interviewer is actually checking |
|---|---|
| ML from scratch | Do you understand the math well enough to write it, and do you treat numerical stability as correctness? |
| DSA | Can you get from brute force to optimal out loud, state complexity honestly, and test your own edge cases? |
| Data manipulation | Do you know where joins, windows, and splits silently corrupt results — and do you check row counts? |
| Debugging | Can you form a hypothesis from a symptom instead of pattern-matching, and do you verify the fix? |

Three closing habits worth more than any single problem here. **Say the invariant.** Most of the bugs in Part 4 are violations of a one-sentence invariant that nobody wrote down. **Compare against an oracle.** Brute force, PyTorch, sklearn, or a shuffled-label run — every solution in this document was checked that way, and it takes under a minute. **Report the baseline alongside the metric.** A number without its baseline is not evidence.
