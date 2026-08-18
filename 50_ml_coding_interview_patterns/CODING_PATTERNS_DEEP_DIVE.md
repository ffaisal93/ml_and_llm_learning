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

> **Saying it out loud.** The trick is one line: subtract the largest logit before you exponentiate. It doesn't change the answer at all, because that constant cancels top and bottom, but it means every exponent you actually compute is zero or negative, so the biggest thing you ever hand to `exp` is 1. Without it, a logit around 1000 overflows to infinity and the whole row comes back NaN. So the named failure mode is overflow-to-NaN in the forward pass — and the same trick, done blockwise, is exactly what makes FlashAttention's online softmax possible.

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

> **Saying it out loud.** Cross-entropy on logits is just the negative log-probability of the correct class, and the clean way to write it is minus the true logit plus log-sum-exp over all of them. I'd never compute softmax and then take a log — that's two chances to lose precision, and if a probability underflows to zero the log is minus infinity. Fold them into log-softmax and it's one stable pass. That's exactly why PyTorch's `nn.CrossEntropyLoss` wants raw logits: handing it softmax output is the classic double-softmax bug that quietly flattens your gradients.

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

> **Saying it out loud.** Attention is a soft dictionary lookup. Every query dots against every key to get a score, you divide by the square root of the head dimension, softmax those scores into weights, and return that weighted average of the values. The scaling matters because dot products of $d$-dimensional vectors grow like $\sqrt{d}$, and without it the softmax saturates and the gradients vanish. Masking is additive — set forbidden scores to $-\infty$ *before* the softmax rather than multiplying by zero after, or the weights never renormalize. Multi-head is just that computation run in parallel on $H$ slices of size $d_{\text{model}}/H$; the failure mode there is a reshape that interleaves heads with positions, which trains silently and badly.

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

> **Saying it out loud.** All of these are just ways of reshaping the distribution before you draw from it. Temperature divides the logits — below 1 sharpens toward greedy, above 1 flattens toward uniform. Top-k keeps a fixed number of candidates; top-p keeps the smallest set whose probabilities sum to at least $p$, and that's the real difference: top-k is blind to how peaked the distribution is, while top-p adapts, taking one token when the model is confident and fifty when it isn't. The tradeoff is diversity versus coherence, and the classic failure is a large k on an already-peaked distribution, which lets tail garbage in.

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

> **Saying it out loud.** Beam search keeps the $B$ best partial sequences instead of just one, expands each by a step, then prunes back to $B$ — greedy search with a wider frontier. You need length normalization because every extra token adds another negative log-probability, so without dividing by length to the power $\alpha$, about 0.6 in practice, the search always prefers to stop early. The tradeoff is that a bigger beam buys higher likelihood but not better text. That's the named failure mode for open-ended generation: the highest-probability output is bland and repetitive, which is why modern LLMs sample with top-p instead.

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

> **Saying it out loud.** K-means alternates two steps until nothing moves: assign every point to its nearest centroid, then move each centroid to the mean of the points it just got. It's coordinate descent on within-cluster squared distance, so it always converges — but only to a local optimum, which is why initialization matters and why K-means++, spreading the initial seeds apart, is the standard fix. The bug I'd call out while coding is the empty cluster: no assigned points means a divide-by-zero mean and NaN centroids, so you re-seed from a random point. And it assumes roughly spherical, similarly sized clusters; elongated ones it will happily cut in half.

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

> **Saying it out loud.** You pad because matmuls need rectangles, and then you mask because those pad tokens are fake and must not influence anything. Two different masks get combined: the causal mask, which stops position $i$ from seeing the future, and the padding mask, which stops anything from attending to filler — you AND them together. Forget the padding mask and the pad positions still receive attention weight, so a sentence's representation depends on who else is in its batch. The failure mode is nondeterministic eval metrics. Practical tradeoff: bucketing similar lengths together cuts a lot of waste, since a batch padded to its longest sequence can easily be half padding.

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

> **Saying it out loud.** Cosine similarity is just a dot product after making every vector unit length, so it measures angle and ignores magnitude. Batched, that's normalize the rows of both matrices once, then a single matmul gives you the whole similarity matrix — much faster than looping. Two things to get right: normalize each row independently, not the matrix as a whole, and add an epsilon in the denominator so a zero vector doesn't produce NaN. The tradeoff against a raw dot product is that cosine throws away the norm, which in embeddings often encodes confidence or frequency — usually what you want for retrieval, often not what you want when popularity is signal.

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

> **Saying it out loud.** Logistic regression puts a sigmoid on a linear score and trains it with log loss, and the nice part is the gradient: it's just $X^\top(p - y)/n$, the same shape of expression you get from linear regression with squared error. The loss is convex, so gradient descent finds the global optimum — there's no local-minimum story here. The failure mode worth naming is perfectly separable data: the weights run off to infinity chasing ever-more-confident predictions, which is exactly what $\ell_2$ regularization exists to stop. Numerically, guard the sigmoid, because `exp` of a large positive argument overflows.

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

> **Saying it out loud.** Backprop is the chain rule applied in reverse order, reusing everything you cached on the way forward. You start at the loss and hand each layer the gradient with respect to its output; the layer turns that into a gradient for its own weights and one to pass further back. The one simplification worth memorizing is that softmax plus cross-entropy collapses to $p - y$ — all the messy Jacobian terms cancel, which is another reason you fuse them. ReLU's derivative is just the mask of where the pre-activation was positive. The classic bug is forgetting to divide by batch size, so your effective learning rate scales with $n$ and training blows up the moment you raise the batch.

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

> **Saying it out loud.** If I compress this table into one habit: in ML coding rounds the wrong answers are almost never about algorithms, they're about numerics and axes. Subtract the max before exponentiating, add $-\infty$ instead of multiplying by zero, normalize per row rather than per matrix, and handle the degenerate case — the empty cluster, the zero vector, the length-one sequence. Say the gotcha out loud as you write the line, because interviewers score the awareness at least as much as the fix. The single most common one is feeding softmax outputs into a loss that already applies log-softmax internally.

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

> **Saying it out loud.** These eight cover most of what actually gets asked, and they share a skeleton: get the shapes right, get the reduction axis right, then say the stability caveat aloud. When I get one, I narrate the plan first — inputs and shapes, then the three or four real lines, then the edge case — because five silent minutes reads as stuck even when the code is fine. Budget is roughly ten minutes each, and I'd rather ship a correct loop and mention the vectorized version than half-finish clever broadcasting. The tradeoff to state explicitly is readability versus speed: interviewers accept a slow correct answer, never a fast wrong one.

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
