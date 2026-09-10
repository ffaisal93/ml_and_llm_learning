# Attention mechanisms: from alignment to efficient long context

Attention is not one linear sequence of replacements. It is a design space with independent axes:

| Axis | Question | Choices |
|---|---|---|
| Role | Where do $Q,K,V$ come from? | self, causal self, cross |
| Heads | How much do heads share? | MHA, MQA, GQA, MLA |
| Connectivity | Which token pairs interact? | dense, window, local-global, random, LSH |
| Approximation | Is softmax attention exact? | Linformer, Nyströmformer, Performer |
| Kernel | How is the same math executed? | standard, FlashAttention |
| Serving memory | How is the KV cache allocated? | contiguous, PagedAttention |

A real model combines choices. A decoder might use **causal + sliding-window + GQA + FlashAttention**,
then store its KV cache with **PagedAttention**. Calling FlashAttention or PagedAttention a replacement
for self-attention is a category error: they change execution and storage, not who attends to whom.

```text
Learn alignment:       additive -> dot product -> scaled dot product
Change the role:       self -> causal self / cross
Add representation:   one head -> MHA
Shrink decode state:   MHA -> MQA -> GQA -> MLA
Reduce token pairs:    window -> Longformer / BigBird / Reformer
Approximate the matrix: Linformer / Nyströmformer / Performer
Keep exact math fast:  FlashAttention
Manage serving memory: PagedAttention
```

## Notation, baseline, and interview code setup

Let $n_q$ and $n_k$ be query and key lengths, $d$ the head width, $h$ the query-head count, $h_{kv}$
the KV-head count, and $w$ a local window. Dense attention forms an $n_q\times n_k$ score matrix:

$$
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right)V.
$$

The snippets below favor clarity over production kernels and assume NumPy plus this stable softmax:

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

## 1. Bahdanau additive attention — learn the alignment score

**Algorithm.** For decoder query $q$ and encoder key $k_j$,

$$e_j=v_a^\top\tanh(W_q q+W_k k_j),\quad
\alpha=\operatorname{softmax}(e),\quad c=\sum_j\alpha_jv_j.$$

**Intuition.** A small neural network learns whether a decoder state and source state match.

**Good.** Flexible when query and key representations differ; produced interpretable translation
alignments. **Bad.** The scoring MLP is harder to batch than one matrix multiplication.

**Complexity.** $O(n_qn_kd_a)$ time and $O(n_qn_k)$ score memory.

```python
def additive_attention(q, K, V, Wq, Wk, va):
    # q:(d,), K:(n,d), Wq/Wk:(d,a), va:(a,)
    scores = np.tanh(q @ Wq + K @ Wk) @ va
    return softmax(scores) @ V
```

**Led to:** replace the learned scoring network with a fast dot product.

## 2. Luong dot-product attention — make alignment a matrix multiply

**Algorithm.** $e_j=q^\top k_j$; a general variant uses $q^\top Wk_j$.

**Intuition.** If queries and keys live in a compatible space, similarity is their dot product.

**Good.** Simple, accelerator-friendly, and easy to vectorize. **Bad.** Unscaled dot products grow with
$d$, pushing softmax into saturation.

**Complexity.** $O(n_qn_kd)$ time and $O(n_qn_k)$ score memory.

```python
def dot_attention(Q, K, V):
    return softmax(Q @ K.T) @ V
```

**Led to:** control the variance of the dot product as head width grows.

## 3. Scaled dot-product attention — stabilize softmax

**Algorithm.** If query and key coordinates have unit variance, $q^\top k$ has variance proportional to $d$. Divide by
$\sqrt d$ so score scale remains roughly constant:

$$A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad O=AV.$$

**Intuition.** Wider vectors create larger accidental dot products; scaling keeps softmax temperature
comparable across head widths.

**Good.** Fast dense matmuls with stable gradients. **Bad.** Every query still compares with every key.

**Complexity.** $O(n_qn_kd)$ time; $O(n_qn_k)$ naive score memory.

```python
def scaled_dot_attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores) @ V
```

**Led to:** choose where $Q,K,V$ originate and what positions may interact.

## 4. Self-attention — let tokens exchange information

**Algorithm.** Project one sequence three ways: $Q=XW_Q$, $K=XW_K$, $V=XW_V$.

**Intuition.** Each token asks a question, advertises what it contains, and supplies information to
matching queries.

**Good.** All token pairs communicate in one layer and training is parallel. **Bad.** Dense self-attention
is quadratic in sequence length and has no order information without positional encoding.

**Complexity.** $O(n^2d)$ attention time, $O(n^2)$ naive score memory.

```python
def self_attention(X, Wq, Wk, Wv):
    return scaled_dot_attention(X @ Wq, X @ Wk, X @ Wv)
```

**Led to:** masks for autoregression and separate sources for encoder-decoder interaction.

## 5. Causal self-attention — hide the future

**Algorithm.** Add $-\infty$ above the diagonal before softmax:

$$M_{ij}=0\text{ if }j\le i,\qquad M_{ij}=-\infty\text{ otherwise}.$$

**Intuition.** Position $i$ may read only tokens already generated.

**Good.** Trains all next-token predictions in parallel without leakage. **Bad.** Information flows only
left-to-right; autoregressive inference is sequential.

**Complexity.** Training remains $O(n^2d)$. With a KV cache, decode step $t$ attends over $t$ stored keys.

```python
def causal_attention(Q, K, V):
    n = Q.shape[0]
    allowed = np.tril(np.ones((n, n), dtype=bool))
    return scaled_dot_attention(Q, K, V, allowed)
```

**Led to:** cache past keys and values during generation; later sections reduce and manage that cache.

## 6. Cross-attention — query another representation

**Algorithm.** $Q=X_{decoder}W_Q$ while $K,V$ come from encoder or modality features.

**Intuition.** The decoder asks which source words, image patches, or retrieved features matter now.

**Good.** Clean separation of source and target; source KV is computed once and reused during decode.
**Bad.** Adds an attention block and a second representation pipeline.

**Complexity.** $O(n_{target}n_{source}d)$ time and $O(n_{target}n_{source})$ score memory.

```python
def cross_attention(X_query, X_source, Wq, Wk, Wv):
    Q = X_query @ Wq
    K, V = X_source @ Wk, X_source @ Wv
    return scaled_dot_attention(Q, K, V)
```

**Led to:** multiple heads so different relations can be represented simultaneously.

## 7. Multi-head attention (MHA) — attend in several subspaces

**Algorithm.** Split model width $D$ into $h$ heads of width $d_h=D/h$, run attention per head, concatenate, and
project:

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Intuition.** Each head asks a different low-dimensional question, then the output projection combines
the answers.

**Good.** Heads can specialize in position, syntax, retrieval, or other relations. **Bad.** Every head
stores its own K and V, making the decode cache $2nD$ values per layer.

**Complexity.** Still $O(n^2D)$ time; KV cache $O(nD)$ values/layer.

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, h):
    n, D = X.shape; dh = D // h
    split = lambda z: z.reshape(n, h, dh).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(dh))
    H = (A @ V).transpose(1, 0, 2).reshape(n, D)
    return H @ Wo
```

**Led to:** share or compress KV heads because cache bandwidth limits decode throughput.

## 8. Multi-query attention (MQA) — share one KV head

**Algorithm.** Keep $h$ query heads but use one shared key head and one shared value head.

**Intuition.** Ask many different questions of one shared memory.

**Good.** Reduces KV cache and KV bandwidth by a factor of $h$. **Bad.** One memory view can reduce quality
when heads genuinely need different key/value projections.

**Complexity.** Attention compute remains $O(n^2D)$; KV cache falls from $O(nD)$ to $O(nd_h)$.

```python
def mqa(Q, K_shared, V_shared):
    # Q:(h,n,dh), shared K/V:(n,dh)
    scores = np.einsum("hqd,kd->hqk", Q, K_shared)
    return softmax(scores / np.sqrt(Q.shape[-1])) @ V_shared
```

**Led to:** use a small number of KV groups to recover expressivity.

## 9. Grouped-query attention (GQA) — compromise between MHA and MQA

**Algorithm.** Each of $h_{kv}$ KV heads is shared by $h/h_{kv}$ query heads. MHA is $h_{kv}=h$; MQA is $h_{kv}=1$.

**Intuition.** Give small groups of query heads their own memory view instead of choosing between one
view for everyone and one view per head.

**Good.** Most of MQA's cache saving with a smaller quality tradeoff. **Bad.** Group count is an
architecture choice and converting an MHA checkpoint requires uptraining or careful pooling.

**Complexity.** Dense compute remains $O(n^2D)$; KV cache is smaller than MHA by $h/h_{kv}$.

```python
def gqa(Q, K_groups, V_groups):
    # Q:(h,n,dh), K/V:(hkv,n,dh)
    h, hkv = Q.shape[0], K_groups.shape[0]
    group = np.arange(h) // (h // hkv)
    K, V = K_groups[group], V_groups[group]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
    return softmax(scores) @ V
```

**Led to:** compress KV content rather than sharing a full KV head.

## 10. Multi-head latent attention (MLA) — cache a compressed latent

**Algorithm sketch.** Project each token to $c=xW_{down}$ with $d_c\ll D$, cache $c$, then derive the
key/value representations needed by attention from that latent.

**Intuition.** Store a compressed sufficient representation of each token and reconstruct the views
needed by the heads, trading extra arithmetic for fewer cache bytes.

**Good.** Large KV-cache reduction while retaining head-specific structure. **Bad.** More architectural
complexity; positional components and projection absorption must be handled carefully, and compression
trades memory for compute.

**Complexity.** Dense attention is still quadratic; cached content is $O(nd_c)$ rather than $O(nD)$.

```python
def latent_kv(X, W_down, W_key_up, W_value_up):
    C = X @ W_down                 # cache this: (n, dc)
    K = C @ W_key_up               # reconstruct/project when needed
    V = C @ W_value_up
    return C, K, V
```

**Led to:** an orthogonal problem—reduce the number of token pairs for very long contexts.

## 11. Sliding-window attention — keep only local edges

**Algorithm.** Token $i$ attends only within distance $w$. In a causal model it sees $[i-w+1,i]$.

**Intuition.** Most language dependencies are local, so spend attention edges nearby and let depth move
summaries across windows.

**Good.** Linear scaling in $n$ for fixed $w$ and a bounded decode cache. **Bad.** Exact information
outside the window must travel through layers; theoretical reach $Lw$ is not exact one-hop recall.

**Complexity.** $O(nwd)$ time and $O(nw)$ score storage; causal KV can be capped at $w$ tokens.

```python
def sliding_attention(Q, K, V, w, causal=True):
    n = Q.shape[0]; i, j = np.ogrid[:n, :n]
    allowed = np.abs(i - j) < w
    if causal: allowed &= (j <= i)
    return scaled_dot_attention(Q, K, V, allowed)
```

**Led to:** add a few global routes so distant information communicates in one hop.

## 12. Longformer — local windows plus designated global tokens

**Algorithm.** Most tokens use a sliding window. Selected task tokens attend to all positions and all
positions attend to them.

**Intuition.** Local edges process text; global tokens act as broadcast hubs.

**Good.** Linear in sequence length for fixed window/global count; strong for long-document encoding.
**Bad.** Global-token selection is task-specific, and local tokens still lack arbitrary direct edges.

**Complexity.** $O(n(w+g)d)$ time and $O(n(w+g))$ attention storage for $g$ globals.

```python
def longformer_mask(n, w, global_ids):
    i, j = np.ogrid[:n, :n]
    allowed = np.abs(i - j) < w
    allowed[:, global_ids] = True   # everyone reads globals
    allowed[global_ids, :] = True   # globals read everyone
    return allowed
```

**Led to:** add random edges for short paths without hand-selecting every global route.

## 13. BigBird — local, global, and random sparse edges

**Algorithm.** Union three patterns: a local window, global tokens, and $r$ random keys per query.

**Intuition.** Local edges capture nearby structure, hubs broadcast important state, and random edges
create short paths between otherwise distant regions.

**Good.** Linear sparsity with short graph paths and theoretical expressivity results. **Bad.** Irregular
sparsity is difficult to turn into real GPU speed, and random connectivity can miss task-specific pairs.

**Complexity.** $O(n(w+g+r)d)$ time and $O(n(w+g+r))$ stored scores.

```python
def bigbird_mask(n, w, global_ids, r, seed=0):
    allowed = longformer_mask(n, w, global_ids)
    rng = np.random.default_rng(seed)
    for i in range(n):
        allowed[i, rng.choice(n, size=min(r, n), replace=False)] = True
    return allowed
```

**Led to:** choose sparse neighbors by content rather than position or randomness.

## 14. Reformer LSH attention — bucket similar content

**Algorithm.** Hash normalized queries/keys with random projections, sort by bucket, and attend within
the same or neighboring buckets, often using multiple hash rounds.

**Intuition.** If two tokens would have high similarity, try to place them in the same bucket before
paying for a dot product.

**Good.** Content-based sparse lookup and approximately $O(n\log n)$ sorting behavior. **Bad.** Hash
collisions, bucket boundaries, multiple rounds, and sorting complicate quality and hardware efficiency.

**Complexity.** Commonly described as $O(n\log n)$ time/memory for fixed bucket size and hash rounds.

```python
def lsh_mask(X, R):
    # R:(bits,d); equal sign patterns share a bucket
    bits = (X @ R.T) > 0
    ids = bits @ (1 << np.arange(bits.shape[1]))
    return ids[:, None] == ids[None, :]
```

**Led to:** instead of selecting edges, approximate the dense attention matrix with low rank.

## 15. Linformer — project sequence length to a small rank

**Algorithm.** Project keys and values from length $n$ to $k\ll n$ using learned matrices $E,F\in\mathbb{R}^{k\times n}$:

$$K'=EK,\qquad V'=FV,\qquad O=\operatorname{softmax}(QK'^\top/\sqrt d)V'.$$

**Intuition.** Replace $n$ token slots with $k$ learned sequence summaries before attention.

**Good.** Simple low-rank path with linear dependence on $n$ when $k$ is fixed. **Bad.** Fixed sequence
projections complicate variable length and autoregressive caching; quality depends on low-rank structure.

**Complexity.** $O(nkd)$ time and $O(nk)$ attention storage.

```python
def linformer_attention(Q, K, V, E, F):
    K_small, V_small = E @ K, F @ V
    return scaled_dot_attention(Q, K_small, V_small)
```

**Led to:** derive low-rank landmarks from the sequence rather than fixed projection matrices.

## 16. Nyströmformer — reconstruct attention from landmarks

**Algorithm.** Choose $m\ll n$ landmark queries/keys and approximate the full softmax matrix using three smaller
attention matrices and a pseudoinverse.

**Intuition.** Describe how ordinary tokens relate to a small landmark set, solve relationships among
the landmarks, then reconstruct the full interaction.

**Good.** Data-dependent low-rank approximation; linear in $n$ for fixed landmarks. **Bad.** Landmark
quality and pseudoinverse approximation affect stability and accuracy.

**Complexity.** Roughly $O(nmd+nm^2+m^3)$ in the direct sketch; memory $O(nm+m^2)$.

```python
def nystrom_attention(Q, K, V, Q_land, K_land):
    scale = np.sqrt(Q.shape[-1])
    A = softmax(Q @ K_land.T / scale)
    B = softmax(Q_land @ K_land.T / scale)
    C = softmax(Q_land @ K.T / scale)
    return A @ np.linalg.pinv(B) @ C @ V
```

**Led to:** use a kernel feature map so multiplication can be reassociated exactly for that kernel.

## 17. Linear attention — summarize keys and values first

**Algorithm.** For a nonnegative feature map $\phi$,

$$
O=\frac{\phi(Q)(\phi(K)^\top V)}
{\phi(Q)(\phi(K)^\top\mathbf 1)}.
$$

Compute $\phi(K)^\top V$ before multiplying by queries; no $n\times n$ matrix is formed.

**Intuition.** Compress the entire key/value history into a fixed-size sufficient statistic that every
query can read.

**Good.** Linear sequence scaling and a recurrent causal state. **Bad.** Replacing softmax changes the
attention kernel and can weaken exact recall and in-context learning.

**Complexity.** $O(nrd_v)$ time and $O(rd_v)$ recurrent state for feature width $r$.

```python
def linear_attention(Q, K, V):
    phi = lambda x: np.where(x > 0, x, np.exp(x) - 1) + 1
    Qf, Kf = phi(Q), phi(K)
    KV = Kf.T @ V
    normalizer = Qf @ Kf.sum(axis=0)
    return (Qf @ KV) / normalizer[:, None]
```

**Led to:** approximate the actual softmax kernel with principled random features.

## 18. Performer/FAVOR+ — random features for softmax attention

**Algorithm.** Performer chooses positive random features $\phi$ so
$\phi(q)^\top\phi(k)\approx\exp(q^\top k)$, then uses the linear-attention association above.

**Intuition.** Approximate the exponential softmax kernel with a finite random feature vector, turning
pairwise attention into two linear passes.

**Good.** Linear in sequence length with an approximation tied to softmax. **Bad.** Random-feature
variance requires enough features and careful numerical stabilization; approximation errors affect
sharp retrieval patterns.

**Complexity.** $O(nrd)$ time and $O(nr)$ feature storage, with $r$ random features.

```python
def positive_random_features(X, R):
    # Simplified FAVOR-style map; production code stabilizes exponent ranges.
    projection = X @ R.T
    norm = 0.5 * np.sum(X * X, axis=-1, keepdims=True)
    return np.exp(projection - norm) / np.sqrt(R.shape[0])

def performer_attention(Q, K, V, R):
    Qf, Kf = positive_random_features(Q, R), positive_random_features(K, R)
    return (Qf @ (Kf.T @ V)) / (Qf @ Kf.sum(0))[:, None]
```

**Led to:** in practice, preserve exact softmax and attack memory traffic with a better kernel.

## 19. FlashAttention — exact attention without the $n^2$ HBM traffic

**Algorithm.** Tile Q, K, and V into on-chip memory. Maintain an online softmax maximum, denominator,
and weighted output; rescale previous partial results whenever the running maximum changes.

**Intuition.** Never write the large score matrix to slow HBM: finish one tile while it is in fast SRAM,
retain only sufficient softmax statistics, and move on.

**Good.** Exact softmax attention, linear auxiliary memory, and much less high-bandwidth-memory traffic.
**Bad.** Arithmetic remains quadratic; the kernel is hardware-specific and substantially harder than
the interview sketch.

**Complexity.** $O(n^2d)$ FLOPs—unchanged—but $O(nd)$ rather than $O(n^2)$ materialized memory.

```python
def flash_attention_one_query(q, K, V, block=64):
    m, ell = -np.inf, 0.0
    out = np.zeros(V.shape[1])
    for start in range(0, len(K), block):
        Kb, Vb = K[start:start+block], V[start:start+block]
        scores = q @ Kb.T / np.sqrt(q.size)
        new_m = max(m, scores.max())
        old_scale, p = np.exp(m - new_m), np.exp(scores - new_m)
        out = old_scale * out + p @ Vb
        ell = old_scale * ell + p.sum()
        m = new_m
    return out / ell
```

**Led to:** during generation the next bottleneck is not the score matrix but the growing KV cache.

## 20. PagedAttention — virtual memory for the KV cache

**Algorithm.** Split physical KV memory into fixed-token blocks. Each request owns a block table mapping
logical token blocks to arbitrary physical blocks. Allocate on demand, free on completion, and reference
count blocks for shared prefixes and copy-on-write branching.

**Intuition.** It is virtual memory for K and V: logical continuity without physical contiguity.

**Good.** Removes external fragmentation and worst-case preallocation; enables prefix sharing and larger
continuous batches. **Bad.** It does not shrink useful KV data or attention FLOPs, and indirection makes
the serving kernel and scheduler more complex.

**Complexity.** Attention math is unchanged. Allocation is $O(1)$ per new block; internal waste is at
most one partially filled block per active sequence.

```python
class PageTable:
    def __init__(self, num_blocks, block_size=16):
        self.free = list(range(num_blocks))
        self.tables, self.block_size = {}, block_size

    def append_token(self, request_id, position):
        table = self.tables.setdefault(request_id, [])
        if position % self.block_size == 0:
            table.append(self.free.pop())
        return table[position // self.block_size], position % self.block_size

    def finish(self, request_id):
        self.free.extend(self.tables.pop(request_id))
```

**Led to:** continuous batching, prefix caching/RadixAttention, chunked prefill, cache quantization, and
eviction policies. These compose with—not replace—the attention mechanism.

## Complexity and memory comparison

| Method | Exact softmax? | Time | Materialized attention/cache | Main compromise |
|---|---:|---:|---:|---|
| Dense scaled attention | Yes | $O(n^2d)$ | $O(n^2)$ scores | Quadratic length |
| MQA/GQA/MLA | Yes | $O(n^2D)$ | Smaller KV cache | Sharing/compression |
| Sliding window | On chosen edges | $O(nwd)$ | $O(nw)$ | Weak distant recall |
| Longformer | On chosen edges | $O(n(w+g)d)$ | $O(n(w+g))$ | Chosen global tokens |
| BigBird | On chosen edges | $O(n(w+g+r)d)$ | Same sparse order | Irregular random edges |
| Reformer | On LSH buckets | about $O(n\log n)$ | subquadratic | Hash/bucket error |
| Linformer | No | $O(nkd)$ | $O(nk)$ | Fixed low-rank projection |
| Nyströmformer | No | roughly $O(nmd+nm^2)$ | $O(nm)$ | Landmark approximation |
| Performer | No | $O(nrd)$ | $O(nr)$ | Random-feature variance |
| FlashAttention | Yes | $O(n^2d)$ | $O(nd)$ auxiliary | Same quadratic FLOPs |
| PagedAttention | Yes | unchanged | Efficient block KV layout | Does not shrink true KV |

## The code to memorize first

If an interviewer gives ten minutes, write only this order:

1. Stable softmax.
2. Scaled dot-product attention with an optional Boolean mask.
3. Causal mask using `np.tril`.
4. Multi-head reshape: `(n, D) -> (h, n, dh)`.
5. One efficient variant: sliding-window mask or linear reassociation.

The named architectures are usually explanation questions. Do not try to reproduce a production
FlashAttention or PagedAttention kernel on a whiteboard; write the online-softmax loop or page-table
allocator and explain what production adds.

## Interview questions

### Q1. Why is attention quadratic?

Every one of $n$ queries is compared with all $n$ keys, producing an $n\times n$ score matrix. Each
comparison costs $O(d)$, so dense attention costs $O(n^2d)$ arithmetic and the naive implementation
stores $O(n^2)$ scores. Projections and FFNs have different terms; name the score matrix specifically.

### Q2. Longformer versus BigBird?

Both keep local windows and global tokens. BigBird additionally uses random sparse edges, providing
short paths and theoretical expressivity results. Longformer is simpler and task-global-token driven;
BigBird has richer connectivity but more irregular execution.

### Q3. Linformer versus Performer?

Linformer assumes low rank along sequence length and learns projections that compress K and V from $n$
positions to $k$. Performer approximates the softmax kernel with positive random features and
reassociates the products. Both can be linear in $n$, but their approximation errors come from different
assumptions.

### Q4. FlashAttention versus linear attention?

FlashAttention computes exact softmax with the same quadratic FLOPs but avoids materializing the full
score matrix in HBM. Linear attention changes or approximates the kernel so sequence-time complexity can
be linear, paying a potential quality cost. One optimizes IO; the other changes the algorithmic work.

### Q5. FlashAttention versus PagedAttention?

FlashAttention tiles the attention computation, especially valuable for prefill/training. PagedAttention
allocates the decode KV cache in blocks so variable-length requests do not waste contiguous reservations.
Neither changes the model's attention probabilities; one optimizes kernel IO, the other serving memory.

### Q6. MHA, MQA, GQA, and MLA in one answer?

MHA gives every query head its own K and V. MQA shares one K/V pair across all query heads. GQA uses a
small number of KV groups, balancing quality and cache size. MLA stores a low-rank latent per token and
derives the required K/V structure from it. Their main inference distinction is KV-cache bytes and
bandwidth, not asymptotic dense-attention FLOPs.

### Q7. Why can theoretical sparsity fail to produce GPU speed?

GPUs are optimized for large regular dense matmuls. Irregular gather/scatter, sorting, small blocks, and
poor occupancy can erase the saved FLOPs. Always distinguish asymptotic operations from realized
wall-clock performance and memory traffic.

## Done when

- You can classify a named method by role, head sharing, connectivity, approximation, kernel, or cache.
- You can write scaled, causal, cross, and multi-head attention from memory.
- You can derive KV-cache savings for MQA/GQA and explain MLA's compression tradeoff.
- You can compare Longformer, BigBird, Reformer, Linformer, Nyströmformer, and Performer without mixing
  their assumptions.
- You can say, without hesitation, that FlashAttention is exact and PagedAttention is a memory manager.
