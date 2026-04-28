# ML Coding Interview Patterns — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

The "live coding" round of ML interviews tests whether you can implement the right thing under pressure — fast, correct, idiomatic. This deep dive covers the canonical patterns: numerical stability, attention internals, sampling, batched operations, training-loop boilerplate. Master these and you'll be fluent on most ML coding prompts.

---

## 1. Numerical stability — softmax and log-sum-exp

The most-asked stability pattern.

### The problem

Naive softmax: $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$.

If $x_i = 1000$, $e^{1000}$ overflows.

### The fix

Subtract the max:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

Mathematically identical (cancels in numerator and denominator). Numerically: every exponent is $\leq 0$.

### Code

```python
def stable_softmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)
```

### Log-sum-exp

For log-probabilities:

$$
\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}
$$

```python
def logsumexp(x, axis=-1, keepdims=False):
    x_max = x.max(axis=axis, keepdims=True)
    out = x_max + np.log(np.exp(x - x_max).sum(axis=axis, keepdims=True))
    return out if keepdims else out.squeeze(axis)
```

Used in cross-entropy loss to combine softmax + log without overflow.

### Why interviewers love it
- Tests numerical-stability awareness.
- Quick to write but easy to get wrong.
- Gateway to harder questions (FlashAttention works on this principle at scale).

---

## 2. Cross-entropy loss

For one-hot $y$ and logits $z$:

$$
\mathcal{L} = -\sum_c y_c \log p_c = -z_y + \log \sum_c e^{z_c}
$$

The right-hand form combines softmax + log in a numerically stable single step.

```python
def cross_entropy(logits, labels):
    # logits: [B, C], labels: [B] (class indices)
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -log_probs[np.arange(len(labels)), labels].mean()
```

### Common interview gotcha
Don't write `softmax → log → loss`. Combine into log-softmax. PyTorch's `nn.CrossEntropyLoss` takes raw logits for this reason.

---

## 3. Attention from scratch

The canonical "implement scaled dot-product attention" prompt.

### Code

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, n_heads, seq_len, d_head]
    mask: [seq_len, seq_len], 0/1 or bool (True = visible)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # [B, H, L, L]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)  # numerically stable in PyTorch
    return attn @ V  # [B, H, L, d_head]
```

### Causal masking

```python
def causal_mask(L):
    return torch.tril(torch.ones(L, L)).bool()  # [L, L], True for visible positions
```

### Multi-head

```python
def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads):
    """
    x: [B, L, D]; W_q, W_k, W_v: [D, D]; W_o: [D, D]
    Returns: [B, L, D]
    """
    B, L, D = x.shape
    d_head = D // n_heads
    # Project + reshape to [B, L, H, d_head] then transpose to [B, H, L, d_head]
    Q = (x @ W_q).reshape(B, L, n_heads, d_head).transpose(1, 2)
    K = (x @ W_k).reshape(B, L, n_heads, d_head).transpose(1, 2)
    V = (x @ W_v).reshape(B, L, n_heads, d_head).transpose(1, 2)
    # Attention runs per-head in parallel; mask broadcasts [L, L] -> [B, H, L, L]
    out = attention(Q, K, V, mask=causal_mask(L))            # [B, H, L, d_head]
    # Concatenate heads back: [B, H, L, d_head] -> [B, L, D]
    out = out.transpose(1, 2).reshape(B, L, D)
    return out @ W_o                                          # [B, L, D]
```

### Common interview gotchas
- Forgetting $\sqrt{d_k}$ scaling.
- Wrong dimension for softmax (should be over keys, last dim).
- Mask: $-\infty$ before softmax, NOT 0 multiply after.
- Multi-head reshape order matters.

---

## 4. Sampling techniques

### Greedy
```python
def greedy(logits):
    return logits.argmax(-1)
```

### Temperature
```python
def temperature_sample(logits, T):
    probs = stable_softmax(logits / T)
    return np.random.choice(len(probs), p=probs)
```

### Top-k

```python
def top_k_sample(logits, k):
    # zero out all but top-k logits
    top_k_idx = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, -np.inf)
    mask[top_k_idx] = logits[top_k_idx]
    probs = stable_softmax(mask)
    return np.random.choice(len(probs), p=probs)
```

### Top-p (nucleus)

```python
def top_p_sample(logits, p):
    """Nucleus sampling: keep smallest set of tokens whose cumulative prob >= p."""
    probs = stable_softmax(logits)
    order = np.argsort(probs)[::-1]              # indices, high → low prob
    cumprobs = np.cumsum(probs[order])
    # Smallest k such that cumprobs[k-1] >= p (boolean → first True)
    keep = cumprobs <= p
    keep[np.argmax(cumprobs >= p)] = True        # ensure we include the threshold-crossing token
    nucleus = order[keep]
    # Renormalize over the nucleus and sample
    nucleus_probs = probs[nucleus] / probs[nucleus].sum()
    return np.random.choice(nucleus, p=nucleus_probs)
```

### Common gotchas
- Top-k of `k=1` should equal greedy.
- Top-p with $p=1$ should equal full sampling.
- Nucleus selects the *smallest* set summing to ≥ $p$, not exactly $p$.

---

## 5. Beam search

Maintain top-$B$ hypotheses; expand each by one step; keep top-$B$ overall.

```python
def beam_search(model, start_token, beam_size=5, max_len=50, eos_token=None):
    beams = [([start_token], 0.0)]  # (sequence list, cumulative log_prob)
    finished = []

    for _ in range(max_len):
        all_candidates = []
        for seq, score in beams:
            if eos_token is not None and seq[-1] == eos_token:
                finished.append((seq, score))
                continue
            log_probs = model(seq)  # log-softmax over vocab
            # Pre-prune: take top-B per beam to avoid O(B*V)
            top_b_idx = np.argpartition(log_probs, -beam_size)[-beam_size:]
            for token in top_b_idx:
                all_candidates.append((seq + [int(token)], score + log_probs[token]))

        if not all_candidates:
            break
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

    finished.extend(beams)
    # Length-normalized score for final selection
    alpha = 0.6
    finished.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
    return finished[0][0]
```

### Length normalization
Long sequences get worse log-prob just by being longer. Common fix: divide by length to the power $\alpha$.

```python
score / (len(seq) ** alpha)
```

### Why beam search loses to sampling for LLMs
Beam search produces deterministic, repetitive, low-entropy outputs. Sampling (top-p, temperature) is the modern default for open-ended generation.

---

## 6. K-means update

```python
def kmeans(X, k, max_iter=100):
    # Initialize centroids randomly from data
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)]

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # [N, K]
        labels = dists.argmin(axis=1)

        # Update centroids to mean of assigned points (handle empty cluster: re-init from random point)
        new_centroids = np.empty_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                new_centroids[c] = X[np.random.randint(n)]   # avoid NaN from empty mean

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids
```

### Common gotchas
- Forgetting to handle empty clusters.
- Wrong axis in norm (should be over feature dim).
- K-means++ initialization is better than uniform random.

---

## 7. Padding and batching

Variable-length sequences need padding for batched matmul.

### Padding

```python
def pad_batch(sequences, pad_value=0):
    max_len = max(len(s) for s in sequences)
    return np.array([list(s) + [pad_value] * (max_len - len(s)) for s in sequences])
```

### Attention mask for padding

```python
def attention_mask(sequences, pad_id=0):
    # 1 where valid, 0 where padding
    return np.array([[1 if t != pad_id else 0 for t in s] for s in sequences])
```

### Combining causal + padding mask

```python
def combined_mask(L, padding_mask):
    causal = torch.tril(torch.ones(L, L)).bool()
    return causal & padding_mask[:, None, :]  # [B, L, L]
```

---

## 8. Vectorized cosine similarity

For retrieval / semantic search.

```python
def cosine_sim_matrix(Q, K, eps=1e-8):
    # Q: [B, D], K: [N, D] -> returns [B, N] cosine similarities
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + eps)   # eps avoids /0
    K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + eps)
    return Q_norm @ K_norm.T
```

### Common gotchas
- Normalize each vector independently (not the whole matrix).
- Handle zero vectors (avoid division by zero).
- For sparse vectors, use scipy.sparse to avoid materializing.

---

## 9. Logistic regression from scratch

```python
def sigmoid(z):
    # Stable: clip extreme values
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression(X, y, lr=0.01, n_iter=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    
    for _ in range(n_iter):
        z = X @ w + b
        p = sigmoid(z)
        gradient_w = X.T @ (p - y) / n
        gradient_b = (p - y).mean()
        w -= lr * gradient_w
        b -= lr * gradient_b
    
    return w, b
```

### Common gotchas
- Sigmoid overflow for large negative inputs (clip).
- Regularization: add $\lambda w$ to gradient for $\ell_2$.
- Multi-class: use softmax + cross-entropy instead.

---

## 10. Backpropagation from scratch (1-hidden-layer MLP)

```python
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1            # [N, H]
    h1 = np.maximum(0, z1)       # ReLU
    z2 = h1 @ W2 + b2            # [N, C]
    return z1, h1, z2

def backward(X, y, z1, h1, z2, W2):
    n = X.shape[0]
    
    # Softmax + cross-entropy: dz2 = (p - y) / n
    p = stable_softmax(z2)
    y_onehot = np.eye(z2.shape[1])[y]
    dz2 = (p - y_onehot) / n     # [N, C]
    
    dW2 = h1.T @ dz2              # [H, C]
    db2 = dz2.sum(0)              # [C]
    
    dh1 = dz2 @ W2.T              # [N, H]
    dz1 = dh1 * (z1 > 0)          # ReLU derivative
    
    dW1 = X.T @ dz1               # [D, H]
    db1 = dz1.sum(0)              # [H]
    
    return dW1, db1, dW2, db2
```

### Tips
- Cross-entropy + softmax gradient simplifies to $p - y$.
- ReLU derivative: 1 where $z > 0$, else 0.
- Batch dimension: divide by $n$ for mean loss; sum biases over batch.

---

## 11. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Softmax overflow fix? | Just compute it | Subtract max for stability |
| Combining softmax + cross-entropy? | Two steps | Use log-softmax + NLL in one step (numerically stable) |
| Attention mask: how applied? | Multiply by 0 | Add $-\infty$ before softmax (so masked positions contribute 0 after softmax) |
| Top-p selects how many tokens? | Exactly hits $p$ | Smallest set summing to $\geq p$ |
| Beam search vs sampling? | Beam better | Beam = repetitive deterministic; sampling preferred for open-ended |
| Cosine similarity normalize what? | Whole matrix | Each row independently |
| K-means empty cluster? | Just ignore | Re-initialize from a random point or mean |

---

## 12. Eight most-asked coding questions

1. **Implement stable softmax.** (Subtract max; combine with log for log-softmax.)
2. **Implement scaled dot-product attention.** ($\sqrt{d_k}$, mask via $-\infty$, softmax over last dim.)
3. **Implement top-p (nucleus) sampling.** (Sort, cumulative, threshold, sample from set.)
4. **Implement K-means.** (Init, assign, update; handle empty clusters.)
5. **Implement logistic regression with gradient descent.** (Sigmoid, BCE gradient.)
6. **Implement backprop for a 2-layer MLP.** (Chain rule; cross-entropy + softmax simplification.)
7. **Implement beam search.** (Top-$B$ hypotheses; length normalization.)
8. **Implement batched cosine similarity.** (Per-row normalize; matmul.)

---

## 13. Drill plan

- For each of the 8 questions, code from scratch in 5-10 minutes.
- For each, recite 2 numerical-stability gotchas.
- Test cases:
  - Softmax with one large value (e.g., 1000).
  - Attention with all-zero mask.
  - Top-p with uniform vs peaked distribution.
  - K-means with $k > n$ (degenerate case).

Keep practicing until you can write idiomatic code without looking up syntax.

---

## 14. Further reading

- Karpathy's *neural networks from scratch* video — backprop + autograd.
- *The Annotated Transformer* (Harvard NLP) — attention from scratch in clean PyTorch.
- *minGPT* (Karpathy) — minimal GPT implementation.
- HuggingFace Transformers source — see how production attention/sampling/beam are implemented.
