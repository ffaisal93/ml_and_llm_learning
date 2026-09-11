# Attention mechanisms: from alignment to efficient long context

This chapter should be read as a story, not as a catalogue of names. We first learn one operation:
**a query compares itself with keys, softmax turns the comparison into weights, and those weights mix
the values**. Everything later changes one part of that sentence.

Attention is not one linear sequence of replacements. After scaled dot-product attention, the history
branches because researchers were solving different problems:

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

## The family tree: what changed, and why

```text
                         SCALED DOT-PRODUCT ATTENTION
                                      |
        +-----------------------------+-----------------------------+
        |                |                 |              |         |
   Who supplies      How heads        Which token      Approximate  Execute/store
    Q, K, V?         share KV?        pairs meet?      the matrix?  exact attention?
        |                |                 |              |         |
 self / causal /    MHA -> MQA ->     window ->        Linformer    FlashAttention
 cross attention     GQA -> MLA       Longformer       Nyström      PagedAttention
                                       BigBird         Performer
                                       Reformer
```

This is the map to keep in your head. **Self, causal, and cross-attention change the source or legal
visibility of tokens. MQA, GQA, and MLA change the KV cache. Longformer, BigBird, and Reformer change
connectivity. Linformer, Nyströmformer, and Performer approximate dense attention. FlashAttention and
PagedAttention preserve the mathematical result while changing execution or storage.**

When answering in an interview, use the same five-sentence shape every time:

1. State the bottleneck in the previous design.
2. State the new idea in one sentence.
3. Walk through what happens to one query token.
4. Give time/memory complexity and the main benefit.
5. Name the quality or systems cost, then connect to the next method.

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

**Why was this needed?** Older encoder-decoder models put the full source sentence into one fixed-size
vector. Important details could be lost, especially in long sentences. Bahdanau attention lets the
decoder look at all source states again. At each output step, it gives more weight to the useful states.

**Algorithm.** For decoder query $q$ and encoder key $k_j$,

$$e_j=v_a^\top\tanh(W_q q+W_k k_j),\quad
\alpha=\operatorname{softmax}(e),\quad c=\sum_j\alpha_jv_j.$$

**Intuition.** A small neural network learns whether a decoder state and source state match.

**Good.** Flexible when query and key representations differ; produced interpretable translation
alignments. **Bad.** The scoring MLP is harder to batch than one matrix multiplication.

**Complexity.** $O(n_qn_kd_a)$ time and $O(n_qn_k)$ score memory.

**Working procedure.** Take the current decoder state as the query. Project it once. Project every
encoder state as a key. Add the projected query to each projected key, apply `tanh`, and use a learned
vector to turn each result into one score. Softmax the scores, then compute the weighted sum of values.
The important picture is: **a tiny neural network judges every query-key pair**.

> **Interview paragraph.** Bahdanau attention was introduced for encoder-decoder alignment. For each
> decoder state, I compare it with every encoder state using a small learned MLP, normalize those scores,
> and take a weighted sum of encoder values. It is flexible because query and key dimensions need not
> already form a good dot-product space, but the pairwise MLP is slower than a batched matrix multiply.

**Code memory.** Broadcast `q @ Wq` across all rows of `K @ Wk`; everything else is softmax and a
weighted sum.

```python
def additive_attention(q, K, V, Wq, Wk, va):
    # q:(d,), K:(n,d), Wq/Wk:(d,a), va:(a,)
    scores = np.tanh(q @ Wq + K @ Wk) @ va
    return softmax(scores) @ V
```

**Led to:** replace the learned scoring network with a fast dot product.

## 2. Luong dot-product attention — make alignment a matrix multiply

**Why was this needed?** Bahdanau attention uses a small neural network to score each query-key pair.
This scoring method is flexible, but it has many operations. Luong attention uses a dot product instead.
The model keeps dynamic alignment and can compute many scores with one fast matrix multiplication.

**Algorithm.** $e_j=q^\top k_j$; a general variant uses $q^\top Wk_j$.

**Intuition.** If queries and keys live in a compatible space, similarity is their dot product.

**Good.** Simple, accelerator-friendly, and easy to vectorize. **Bad.** Unscaled dot products grow with
$d$, pushing softmax into saturation.

**Complexity.** $O(n_qn_kd)$ time and $O(n_qn_k)$ score memory.

**Working procedure.** Put every query in one matrix and every key in another. `Q @ K.T` computes all
pair scores at once. Softmax each query's row, then multiply by `V`. Compared with additive attention,
the alignment network disappears; the similarity function is now just a dot product.

> **Interview paragraph.** Luong attention keeps the encoder-decoder alignment idea but replaces the
> learned pairwise scoring MLP with a dot product. That turns all comparisons into one efficient matrix
> multiplication. The weakness is scale: as vector width grows, raw dot products grow and softmax
> becomes overly sharp, which is exactly why scaled dot-product attention divides by square root of the
> key dimension.

**Code memory.** The whole algorithm is `softmax(Q @ K.T) @ V`.

```python
def dot_attention(Q, K, V):
    return softmax(Q @ K.T) @ V
```

**Led to:** control the variance of the dot product as head width grows.

## 3. Scaled dot-product attention — stabilize softmax

**Why was this needed?** Dot products become large when the vectors become wide. Large scores make
softmax too sharp. Then most positions get almost no gradient. Division by $\sqrt d$ keeps the scores in
a safer range and makes training more stable.

**Algorithm.** If query and key coordinates have unit variance, $q^\top k$ has variance proportional to $d$. Divide by
$\sqrt d$ so score scale remains roughly constant:

$$A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad O=AV.$$

**Intuition.** Wider vectors create larger accidental dot products; scaling keeps softmax temperature
comparable across head widths.

**Good.** Fast dense matmuls with stable gradients. **Bad.** Every query still compares with every key.

**Complexity.** $O(n_qn_kd)$ time; $O(n_qn_k)$ naive score memory.

**Working procedure.** Compute all query-key dot products, divide by $\sqrt d$, apply any legal-position
mask, normalize each query row with softmax, and use the resulting probabilities to average values. One
query's output is therefore not a chosen token; it is a weighted mixture of information from allowed
tokens.

> **Interview paragraph.** Scaled dot-product attention has four steps: compare queries with keys,
> divide by square root of head width, mask illegal positions, and softmax before mixing values. The
> scale matters because an unscaled dot product has variance proportional to width and would saturate
> softmax. This becomes the reusable core; self, causal, cross, and multi-head attention mainly change
> where the tensors come from or how they are shaped.

**Code memory.** Memorize four lines: scores, scale, optional mask, `softmax(scores) @ V`.

```python
def scaled_dot_attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores) @ V
```

**Led to:** choose where $Q,K,V$ originate and what positions may interact.

## 4. Self-attention — let tokens exchange information

**Why was this needed?** Cross-sequence attention lets a decoder read an encoded source. It does not
explain how tokens in one sequence should build context together. Self-attention lets each token read
other tokens in the same sequence. The model can now build contextual token representations in parallel.

**Algorithm.** Project one sequence three ways: $Q=XW_Q$, $K=XW_K$, $V=XW_V$.

**Intuition.** Each token asks a question, advertises what it contains, and supplies information to
matching queries.

**Good.** All token pairs communicate in one layer and training is parallel. **Bad.** Dense self-attention
is quadratic in sequence length and has no order information without positional encoding.

**Complexity.** $O(n^2d)$ attention time, $O(n^2)$ naive score memory.

**Working procedure.** Start with token matrix `X`. Create three learned views: queries describe what
each token wants, keys describe what each token offers, and values contain what will actually be copied.
Run scaled dot-product attention. Every output row now contains a context-aware version of that token.

> **Interview paragraph.** Self-attention means Q, K, and V all come from the same sequence through
> different projections. Each token compares its query with every key and mixes the matching values, so
> all tokens exchange information in one layer. It is parallel and gives one-hop global communication,
> but it is quadratic in sequence length and still needs positional information because the operation
> alone does not know token order.

**Code memory.** Self-attention is the core helper called with `X @ Wq`, `X @ Wk`, and `X @ Wv`.

```python
def self_attention(X, Wq, Wk, Wv):
    return scaled_dot_attention(X @ Wq, X @ Wk, X @ Wv)
```

**Led to:** masks for autoregression and separate sources for encoder-decoder interaction.

## 5. Causal self-attention — hide the future

**Why was this needed?** Normal self-attention lets a token read future tokens. This leaks the correct
answer during next-token training. Future tokens are also not available during generation. A causal
mask blocks the future, so training follows the same information rule as generation.

**Algorithm.** Add $-\infty$ above the diagonal before softmax:

$$M_{ij}=0\text{ if }j\le i,\qquad M_{ij}=-\infty\text{ otherwise}.$$

**Intuition.** Position $i$ may read only tokens already generated.

**Good.** Trains all next-token predictions in parallel without leakage. **Bad.** Information flows only
left-to-right; autoregressive inference is sequential.

**Complexity.** Training remains $O(n^2d)$. With a KV cache, decode step $t$ attends over $t$ stored keys.

**Working procedure.** Build the ordinary self-attention score matrix. Before softmax, replace every
entry above the diagonal with negative infinity. Softmax converts those positions to zero probability.
During training all rows are computed in parallel; during generation only the newest row is needed
because future tokens do not exist.

> **Interview paragraph.** Causal attention is self-attention with a triangular visibility rule. Token
> $i$ can read positions up to $i$ but not the future. Adding negative infinity above the diagonal makes
> future softmax weights exactly zero, which lets us train all next-token predictions in parallel
> without leakage. At inference, we cache old keys and values and compute attention only for the new
> query, so the cache grows one token per step.

**Code memory.** Make the legal mask with `np.tril`; do not write a second attention implementation.

```python
def causal_attention(Q, K, V):
    n = Q.shape[0]
    allowed = np.tril(np.ones((n, n), dtype=bool))
    return scaled_dot_attention(Q, K, V, allowed)
```

**Led to:** cache past keys and values during generation; later sections reduce and manage that cache.

## 6. Cross-attention — query another representation

**Why was this needed?** Self-attention only reads from the same sequence. Some tasks need one stream to
read another stream. For example, generated text can read source text or image features. Cross-attention
uses queries from the active stream and uses keys and values from the other stream.

**Algorithm.** $Q=X_{decoder}W_Q$ while $K,V$ come from encoder or modality features.

**Intuition.** The decoder asks which source words, image patches, or retrieved features matter now.

**Good.** Clean separation of source and target; source KV is computed once and reused during decode.
**Bad.** Adds an attention block and a second representation pipeline.

**Complexity.** $O(n_{target}n_{source}d)$ time and $O(n_{target}n_{source})$ score memory.

**Working procedure.** Project decoder or target states into queries. Project the fixed source sequence
into keys and values. Each target query scores every source key and retrieves a weighted mixture of
source values. During autoregressive decoding, source K and V are computed once and reused.

> **Interview paragraph.** Cross-attention uses queries from one stream and keys and values from
> another. A decoder token can ask which encoder word, image patch, or retrieved feature matters for its
> next computation. It is ideal when source and target have distinct roles, and source KV is reusable
> across decoding. The cost is an extra block and $n_{target}n_{source}$ pair comparisons.

**Code memory.** The core attention call is unchanged; only the sources of Q versus K/V differ.

```python
def cross_attention(X_query, X_source, Wq, Wk, Wv):
    Q = X_query @ Wq
    K, V = X_source @ Wk, X_source @ Wv
    return scaled_dot_attention(Q, K, V)
```

**Led to:** multiple heads so different relations can be represented simultaneously.

## 7. Multi-head attention (MHA) — attend in several subspaces

**Why was this needed?** One attention map must learn all useful relationships. This is difficult when
the model must track position, syntax, identity, and long-range links at the same time. MHA gives the
model several smaller attention spaces. Each head can learn a different type of relationship.

**Algorithm.** Split model width $D$ into $h$ heads of width $d_h=D/h$, run attention per head, concatenate, and
project:

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Intuition.** Each head asks a different low-dimensional question, then the output projection combines
the answers.

**Good.** Heads can specialize in position, syntax, retrieval, or other relations. **Bad.** Every head
stores its own K and V, making the decode cache $2nD$ values per layer.

**Complexity.** Still $O(n^2D)$ time; KV cache $O(nD)$ values/layer.

**Working procedure.** Project the token matrix into Q, K, and V as usual. Reshape each projection from
`(n, D)` into `h` smaller matrices of shape `(n, dh)`. Each head independently performs scaled
dot-product attention, so one head may focus on nearby syntax while another finds a distant reference.
Concatenate the `h` answers back to width `D`, then use `Wo` to mix information across heads.

> **Interview paragraph.** Multi-head attention runs several smaller attentions in parallel. We split
> the model dimension across heads, let every head learn its own Q, K, and V projections, concatenate
> the results, and apply an output projection. This gives the model several relation types at once
> without changing the asymptotic compute. Its serving weakness is that every head stores its own keys
> and values, so the KV cache becomes expensive during generation.

**Code memory.** Remember the shape dance: `reshape -> transpose -> attend -> transpose -> reshape`.

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

**Why was this needed?** MHA stores separate keys and values for every head and every past token. During
generation, this KV cache becomes large. Reading it can limit generation speed. MQA keeps many query
heads but gives them one shared K/V memory, so the cache is much smaller.

**Algorithm.** Keep $h$ query heads but use one shared key head and one shared value head.

**Intuition.** Ask many different questions of one shared memory.

**Good.** Reduces KV cache and KV bandwidth by a factor of $h$. **Bad.** One memory view can reduce quality
when heads genuinely need different key/value projections.

**Complexity.** Attention compute remains $O(n^2D)$; KV cache falls from $O(nD)$ to $O(nd_h)$.

**Working procedure.** Keep all `h` query heads, because different heads should still ask different
questions. Create only one key matrix and one value matrix, however, and let every query head read that
same memory. The output still contains one result per query head; only the stored K/V representation is
shared. During autoregressive decoding this cuts both cache size and the bytes read for each new token.

> **Interview paragraph.** Multi-query attention keeps many query heads but shares one key head and one
> value head. I remember it as “many questions, one memory.” Compared with MHA it reduces KV-cache
> storage and bandwidth by roughly the number of heads, which can greatly improve decoding throughput.
> The tradeoff is that all heads must use the same memory view, which can reduce model quality.

**Code memory.** Q has a head dimension; shared K and V do not. The `einsum` broadcasts that one memory
across all query heads.

```python
def mqa(Q, K_shared, V_shared):
    # Q:(h,n,dh), shared K/V:(n,dh)
    scores = np.einsum("hqd,kd->hqk", Q, K_shared)
    return softmax(scores / np.sqrt(Q.shape[-1])) @ V_shared
```

**Led to:** use a small number of KV groups to recover expressivity.

## 9. Grouped-query attention (GQA) — compromise between MHA and MQA

**Why was this needed?** MQA saves memory, but one shared K/V view can reduce quality. MHA keeps full
quality but uses a large cache. GQA gives each small group of query heads one K/V head. It keeps several
memory views and still saves much of the cache space.

**Algorithm.** Each of $h_{kv}$ KV heads is shared by $h/h_{kv}$ query heads. MHA is $h_{kv}=h$; MQA is $h_{kv}=1$.

**Intuition.** Give small groups of query heads their own memory view instead of choosing between one
view for everyone and one view per head.

**Good.** Most of MQA's cache saving with a smaller quality tradeoff. **Bad.** Group count is an
architecture choice and converting an MHA checkpoint requires uptraining or careful pooling.

**Complexity.** Dense compute remains $O(n^2D)$; KV cache is smaller than MHA by $h/h_{kv}$.

**Working procedure.** Divide the `h` query heads into `hkv` groups. All query heads in one group share
one key head and one value head, while different groups keep different memories. At the extremes,
`hkv=h` recreates MHA and `hkv=1` recreates MQA. Choosing a value between them gives most of MQA's
memory saving without forcing every query head to share exactly the same K/V representation.

> **Interview paragraph.** Grouped-query attention is the middle point between MHA and MQA. Query heads
> are partitioned into groups, and each group shares one K/V head. That reduces the cache by
> $h/h_{kv}$ while preserving more K/V diversity than MQA. This is often the practical quality-versus-
> serving-efficiency compromise.

**Code memory.** Build a query-head-to-KV-group lookup, then index K and V with it. After that, the
attention calculation is ordinary batched attention.

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

**Why was this needed?** GQA saves cache space by sharing full K/V heads. This sharing can still remove
useful head-specific information. MLA stores one small latent vector for each token instead. The model
uses this latent vector to make the key and value views that it needs.

**Algorithm sketch.** Project each token to $c=xW_{down}$ with $d_c\ll D$, cache $c$, then derive the
key/value representations needed by attention from that latent.

**Intuition.** Store a compressed sufficient representation of each token and reconstruct the views
needed by the heads, trading extra arithmetic for fewer cache bytes.

**Good.** Large KV-cache reduction while retaining head-specific structure. **Bad.** More architectural
complexity; positional components and projection absorption must be handled carefully, and compression
trades memory for compute.

**Complexity.** Dense attention is still quadratic; cached content is $O(nd_c)$ rather than $O(nD)$.

**Working procedure.** Instead of storing full key and value vectors for every previous token, first
down-project each token into a much smaller latent vector `C`. Cache that latent once. The model then
derives the key and value views needed by its heads from `C`, paying extra projection work to read fewer
bytes from memory. Real MLA implementations carefully separate or absorb projections and handle the
positional component; the short code below shows only the central compression idea.

> **Interview paragraph.** MLA attacks the KV-cache problem by compression rather than sharing. Each
> token is stored as a low-dimensional latent, and head-specific key/value information is derived from
> that latent when attention runs. Dense attention is still quadratic, but the persistent cache is much
> smaller. So MLA trades additional reconstruction arithmetic and architectural complexity for lower
> memory capacity and bandwidth requirements.

**Code memory.** `C = X @ W_down` is the important line: `C` is what survives in the cache. The two
up-projections only illustrate how key and value views can be obtained from it.

```python
def latent_kv(X, W_down, W_key_up, W_value_up):
    C = X @ W_down                 # cache this: (n, dc)
    K = C @ W_key_up               # reconstruct/project when needed
    V = C @ W_value_up
    return C, K, V
```

**Led to:** an orthogonal problem—reduce the number of token pairs for very long contexts.

## 11. Sliding-window attention — keep only local edges

**Why was this needed?** MQA, GQA, and MLA reduce KV-cache size. They do not remove the $n^2$ comparisons
in dense attention. These comparisons are too costly for very long sequences. Sliding-window attention
computes only nearby comparisons because many language relationships are local.

**Algorithm.** Token $i$ attends only within distance $w$. In a causal model it sees $[i-w+1,i]$.

**Intuition.** Most language dependencies are local, so spend attention edges nearby and let depth move
summaries across windows.

**Good.** Linear scaling in $n$ for fixed $w$ and a bounded decode cache. **Bad.** Exact information
outside the window must travel through layers; theoretical reach $Lw$ is not exact one-hop recall.

**Complexity.** $O(nwd)$ time and $O(nw)$ score storage; causal KV can be capped at $w$ tokens.

**Working procedure.** Compute attention exactly as before, but change which query-key pairs are legal.
For query position `i`, keep only keys within `w` positions; in a causal model also remove keys to the
right of `i`. A simple teaching implementation still constructs the full mask, but an efficient kernel
computes only those local pairs. In causal generation, old K/V entries can be overwritten with a ring
buffer once they fall outside the window.

> **Interview paragraph.** Sliding-window attention assumes most useful interactions are local. Each
> token attends to only $w$ nearby tokens, reducing work from $O(n^2d)$ to $O(nwd)$ and allowing a
> bounded causal cache. Attention inside the window is exact, not approximate. The weakness is that a
> distant fact cannot be reached in one hop; it must be carried through multiple layers or tokens.

**Code memory.** Keep the attention helper and replace only its mask: `abs(i-j) < w`, plus `j <= i` for
causal attention.

```python
def sliding_attention(Q, K, V, w, causal=True):
    n = Q.shape[0]; i, j = np.ogrid[:n, :n]
    allowed = np.abs(i - j) < w
    if causal: allowed &= (j <= i)
    return scaled_dot_attention(Q, K, V, allowed)
```

**Led to:** add a few global routes so distant information communicates in one hop.

## 12. Longformer — local windows plus designated global tokens

**Why was this needed?** A sliding window cannot directly reach a distant token. Information must pass
through many layers to move across a long document. Longformer adds a small set of global tokens. These
tokens give distant parts of the document a short communication path.

**Algorithm.** Most tokens use a sliding window. Selected task tokens attend to all positions and all
positions attend to them.

**Intuition.** Local edges process text; global tokens act as broadcast hubs.

**Good.** Linear in sequence length for fixed window/global count; strong for long-document encoding.
**Bad.** Global-token selection is task-specific, and local tokens still lack arbitrary direct edges.

**Complexity.** $O(n(w+g)d)$ time and $O(n(w+g))$ attention storage for $g$ globals.

**Working procedure.** Start with the sliding-window visibility mask. Mark a small set of positions as
global—often a classification token or task-selected tokens. Then open both directions: every local
token may read the global tokens, and every global token may read the entire sequence. Local processing
stays cheap, while a global token can collect information from anywhere and broadcast it back.

> **Interview paragraph.** Longformer adds a few global communication hubs to sliding-window attention.
> Ordinary tokens use local windows, while selected global tokens attend everywhere and are visible
> everywhere. This preserves near-linear scaling and restores one-hop long-range communication through
> the hubs. Its main design question is which tokens deserve global attention for the task.

**Code memory.** Create the window mask, then make the global columns and global rows `True`.

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

**Why was this needed?** Longformer depends on the choice of global tokens. A useful distant link can
still be missed. BigBird adds random links to the local and global links. The random links create more
short paths between distant parts of the sequence.

**Algorithm.** Union three patterns: a local window, global tokens, and $r$ random keys per query.

**Intuition.** Local edges capture nearby structure, hubs broadcast important state, and random edges
create short paths between otherwise distant regions.

**Good.** Linear sparsity with short graph paths and theoretical expressivity results. **Bad.** Irregular
sparsity is difficult to turn into real GPU speed, and random connectivity can miss task-specific pairs.

**Complexity.** $O(n(w+g+r)d)$ time and $O(n(w+g+r))$ stored scores.

**Working procedure.** Begin with Longformer's local and global edges. For every query, additionally
choose `r` key positions at random and allow those pairs. The union supplies three behaviors: windows
handle nearby structure, globals carry task-wide state, and random links shorten paths between distant
regions that have no chosen hub. Production layouts make this pattern more structured than the tiny
mask example.

> **Interview paragraph.** BigBird combines local, global, and random sparse attention. Local edges
> capture neighborhood structure, global tokens provide stable hubs, and random edges create short
> paths across the sequence. For fixed numbers of each edge type, cost is linear in sequence length.
> The downside is that irregular sparse patterns can be difficult to execute efficiently on GPUs.

**Code memory.** Reuse the Longformer mask and turn `r` additional random positions on in every row.

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

**Why was this needed?** Local, global, and random rules choose links without first checking token
content. They can compare unrelated tokens and miss two similar distant tokens. Reformer groups tokens
by content similarity. Attention then runs mainly inside each group.

**Algorithm.** Hash normalized queries/keys with random projections, sort by bucket, and attend within
the same or neighboring buckets, often using multiple hash rounds.

**Intuition.** If two tokens would have high similarity, try to place them in the same bucket before
paying for a dot product.

**Good.** Content-based sparse lookup and approximately $O(n\log n)$ sorting behavior. **Bad.** Hash
collisions, bucket boundaries, multiple rounds, and sorting complicate quality and hardware efficiency.

**Complexity.** Commonly described as $O(n\log n)$ time/memory for fixed bucket size and hash rounds.

**Working procedure.** Project every token onto several random directions and record only the sign of
each projection. Tokens with the same sign pattern receive the same bucket ID. Sort or group tokens by
that ID, then compute dense attention only inside each small bucket and sometimes its neighbor. Repeat
with several hashes so a useful pair missed by one partition has another chance to meet.

> **Interview paragraph.** Reformer chooses sparse connections by content. Locality-sensitive hashing
> places vectors that are likely to be similar into the same bucket, and attention is computed within
> buckets rather than across all token pairs. This gives roughly $O(n\log n)$ behavior, but it is an
> approximation: collisions, boundary misses, repeated hashing, and sorting complicate both quality
> and hardware execution.

**Code memory.** The sketch is `random projection -> sign bits -> bucket ID -> equality mask`.

```python
def lsh_mask(X, R):
    # R:(bits,d); equal sign patterns share a bucket
    bits = (X @ R.T) > 0
    ids = bits @ (1 << np.arange(bits.shape[1]))
    return ids[:, None] == ids[None, :]
```

**Led to:** instead of selecting edges, approximate the dense attention matrix with low rank.

## 15. Linformer — project sequence length to a small rank

**Why was this needed?** Sparse methods select only some token pairs. Their irregular connection
patterns can be hard to run efficiently. Linformer uses a different idea. It assumes that a small set of
learned summaries can represent the keys and values, so each query has fewer items to compare.

**Algorithm.** Project keys and values from length $n$ to $k\ll n$ using learned matrices $E,F\in\mathbb{R}^{k\times n}$:

$$K'=EK,\qquad V'=FV,\qquad O=\operatorname{softmax}(QK'^\top/\sqrt d)V'.$$

**Intuition.** Replace $n$ token slots with $k$ learned sequence summaries before attention.

**Good.** Simple low-rank path with linear dependence on $n$ when $k$ is fixed. **Bad.** Fixed sequence
projections complicate variable length and autoregressive caching; quality depends on low-rank structure.

**Complexity.** $O(nkd)$ time and $O(nk)$ attention storage.

**Working procedure.** Leave the queries at length `n`. Use learned matrices `E` and `F` to mix the `n`
key and value rows into only `k` summary rows. Each original query now compares with those `k` key
summaries and mixes the corresponding value summaries using ordinary scaled attention. The saving
comes from replacing an `n by n` score matrix with an `n by k` one.

> **Interview paragraph.** Linformer assumes the sequence dimension of attention is approximately low
> rank. It learns projections that compress K and V from $n$ positions to $k$ summaries, then performs
> normal attention from every query to those summaries. With fixed $k$, complexity is linear in $n$.
> Fixed learned length projections and autoregressive updates are its main practical difficulties.

**Code memory.** Add only two lines before the baseline helper: `K_small = E @ K` and
`V_small = F @ V`.

```python
def linformer_attention(Q, K, V, E, F):
    K_small, V_small = E @ K, F @ V
    return scaled_dot_attention(Q, K_small, V_small)
```

**Led to:** derive low-rank landmarks from the sequence rather than fixed projection matrices.

## 16. Nyströmformer — reconstruct attention from landmarks

**Why was this needed?** Linformer uses fixed learned projections to make its summaries. These
projections can be difficult to use with different sequence lengths. They also do not choose summaries
from the current input. Nyströmformer creates landmarks from the input and uses them as routes between
all tokens. This avoids a direct comparison for every token pair.

**Algorithm.** Choose $m\ll n$ landmark queries/keys and approximate the full softmax matrix using three smaller
attention matrices and a pseudoinverse.

**Intuition.** Describe how ordinary tokens relate to a small landmark set, solve relationships among
the landmarks, then reconstruct the full interaction.

**Good.** Data-dependent low-rank approximation; linear in $n$ for fixed landmarks. **Bad.** Landmark
quality and pseudoinverse approximation affect stability and accuracy.

**Complexity.** Roughly $O(nmd+nm^2+m^3)$ in the direct sketch; memory $O(nm+m^2)$.

**Working procedure.** Select `m` representative landmark queries and keys, often by averaging chunks
of the sequence. Build three maps: every query to landmark keys (`A`), landmark queries to landmark keys
(`B`), and landmark queries to every key (`C`). The product `A @ pinv(B) @ C` reconstructs an
approximation of the full attention matrix, which then mixes V.

> **Interview paragraph.** Nyströmformer replaces all token-to-token interactions with routes through a
> small set of data-derived landmarks. Token-to-landmark and landmark-to-token matrices, corrected by a
> pseudoinverse of landmark-to-landmark attention, approximate the full matrix. It is linear in sequence
> length for fixed landmark count, but accuracy depends on representative landmarks and a stable inverse.

**Code memory.** Remember the three matrices in order: `A` gets into landmarks, `pinv(B)` corrects the
landmark space, and `C` gets back to all tokens.

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

**Why was this needed?** Landmark methods still build an approximation of the attention matrix. They can
also need an expensive or unstable matrix inverse. Linear attention avoids the full matrix. It first
makes one summary of the keys and values, then lets every query read that summary.

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

**Working procedure.** Apply a positive feature map `phi` to queries and keys. First combine all
transformed keys and values into the fixed-size summary `Kf.T @ V`, and separately sum the transformed
keys for normalization. Each query reads those two summaries instead of comparing with every key. In a
causal model, update both summaries as each new key/value arrives, so the past becomes a recurrent state.

> **Interview paragraph.** Linear attention removes the quadratic matrix by using a kernel feature map
> and reassociating multiplication. It summarizes the key/value history first, then answers every query
> from that summary, giving linear sequence scaling and a fixed-size causal state. The cost is semantic:
> a generic feature map replaces softmax attention and may weaken sharp retrieval or exact recall.

**Code memory.** The decisive line is `KV = Kf.T @ V`; queries multiply this summary afterward. Do not
forget the key-sum denominator.

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

**Why was this needed?** Basic linear attention changes the softmax rule. This can reduce the model's
ability to retrieve one exact item. Performer tries to keep behavior that is closer to softmax. It uses
random features to approximate the softmax kernel and then uses the linear-attention calculation.

**Algorithm.** Performer chooses positive random features $\phi$ so
$\phi(q)^\top\phi(k)\approx\exp(q^\top k)$, then uses the linear-attention association above.

**Intuition.** Approximate the exponential softmax kernel with a finite random feature vector, turning
pairwise attention into two linear passes.

**Good.** Linear in sequence length with an approximation tied to softmax. **Bad.** Random-feature
variance requires enough features and careful numerical stabilization; approximation errors affect
sharp retrieval patterns.

**Complexity.** $O(nrd)$ time and $O(nr)$ feature storage, with $r$ random features.

**Working procedure.** Draw a fixed random projection matrix `R`. Map every query and key into `r`
positive features whose inner product estimates the exponential softmax kernel. Once those features
exist, run the same summary-first calculation as linear attention: form `Kf.T @ V`, form the key sum,
and let each transformed query read them. More features reduce approximation variance but add work.

> **Interview paragraph.** Performer is a principled version of linear attention that uses FAVOR+
> random features to approximate the exponential kernel behind softmax. It then reassociates the
> computation so no $n\times n$ matrix is formed, giving $O(nrd)$ cost. Unlike FlashAttention, its
> output is approximate; feature count, variance, and numerical stabilization control the tradeoff.

**Code memory.** Memorize two pieces: the positive random-feature map, followed by exactly the same
`Kf.T @ V` and normalization pattern used in linear attention.

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

**Why was this needed?** Sparse and approximate methods can lose quality. They can also be slow on real
hardware when their memory access is irregular. Standard exact attention writes a very large score
matrix to slow memory. FlashAttention computes the same result in small blocks and does not store that
full matrix.

**Algorithm.** Tile Q, K, and V into on-chip memory. Maintain an online softmax maximum, denominator,
and weighted output; rescale previous partial results whenever the running maximum changes.

**Intuition.** Never write the large score matrix to slow HBM: finish one tile while it is in fast SRAM,
retain only sufficient softmax statistics, and move on.

**Good.** Exact softmax attention, linear auxiliary memory, and much less high-bandwidth-memory traffic.
**Bad.** Arithmetic remains quadratic; the kernel is hardware-specific and substantially harder than
the interview sketch.

**Complexity.** $O(n^2d)$ FLOPs—unchanged—but $O(nd)$ rather than $O(n^2)$ materialized memory.

**Working procedure.** Take a small block of queries and stream through K/V blocks. For each query,
keep only three running quantities: the largest score seen, the softmax denominator, and the weighted
value sum. When a later block contains a larger score, rescale the old denominator and output into the
new numerical scale before adding the new block. Discard each score tile after use; the final division
produces the same answer as dense softmax. The backward pass recomputes tiles instead of storing them.

> **Interview paragraph.** FlashAttention does not change attention mathematically. It computes exact
> softmax in tiles, keeps intermediate scores in fast on-chip memory, and maintains online-softmax
> statistics so the full score matrix is never written to HBM. The FLOP count remains quadratic, but
> memory traffic and auxiliary storage fall sharply, making exact attention much faster in practice.

**Code memory.** For each tile update `m`, `ell`, and `out`; if `m` changes, rescale the old `ell` and
`out` before adding the new exponentials.

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

**Why was this needed?** FlashAttention improves the attention calculation. It does not solve KV-cache
allocation for many requests with unknown output lengths. Large reserved buffers waste memory, and
different request lengths cause gaps. PagedAttention allocates small KV blocks only when they are needed.

**Algorithm.** Split physical KV memory into fixed-token blocks. Each request owns a block table mapping
logical token blocks to arbitrary physical blocks. Allocate on demand, free on completion, and reference
count blocks for shared prefixes and copy-on-write branching.

**Intuition.** It is virtual memory for K and V: logical continuity without physical contiguity.

**Good.** Removes external fragmentation and worst-case preallocation; enables prefix sharing and larger
continuous batches. **Bad.** It does not shrink useful KV data or attention FLOPs, and indirection makes
the serving kernel and scheduler more complex.

**Complexity.** Attention math is unchanged. Allocation is $O(1)$ per new block; internal waste is at
most one partially filled block per active sequence.

**Working procedure.** Divide the KV-memory pool into equal physical blocks. Give each request a small
table whose entries say where its first, second, and later logical KV blocks physically live. Allocate a
new physical block only when the current one fills. To read token position `p`, use integer division to
find its table entry and modulo to find the offset inside that block. When a request finishes, return its
blocks to the free pool; reference counts and copy-on-write can safely share prompt prefixes.

> **Interview paragraph.** PagedAttention applies virtual-memory ideas to the KV cache. A request sees a
> logically continuous cache, but fixed-size blocks may live anywhere in physical memory and are
> allocated on demand. This removes fragmentation and large worst-case reservations and enables prefix
> sharing and larger continuous batches. It does not reduce the true KV data or attention FLOPs; it
> improves how serving memory is managed.

**Code memory.** On a block boundary, allocate one page. For lookup, use
`table[position // block_size]` and `position % block_size`; on finish, return all pages.

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

## The whole evolution as one interview story

Start with **Bahdanau attention**: a learned network scores how well one decoder query matches each
encoder state. **Luong** simplifies the scorer to a dot product, and **scaled dot-product attention**
divides by $\sqrt d$ so large vectors do not saturate softmax. From that stable core, **self-attention**
changes Q, K, and V to come from one sequence, **causal attention** adds a future-blocking mask, and
**cross-attention** keeps Q in one stream while K/V come from another. **MHA** then adds several relation
subspaces.

After MHA, the story branches because later methods solve different bottlenecks:

1. **Decode cache:** MQA shares one K/V head, GQA shares within groups, and MLA caches a compressed
   latent. The score matrix is still dense; what changes is the stored memory per past token.
2. **Too many token pairs:** sliding windows keep local pairs, Longformer adds global hubs, BigBird adds
   random shortcuts, and Reformer chooses sparse neighbors by content hashing.
3. **Approximate the dense matrix:** Linformer compresses the sequence axis, Nyströmformer routes
   through landmarks, and Performer approximates the softmax kernel with random features.
4. **Execute exact attention better:** FlashAttention keeps the same answer but tiles the computation
   to avoid HBM traffic. PagedAttention then manages the growing inference KV cache in non-contiguous
   blocks. Flash changes the kernel; Paged changes the cache allocator.

> **90-second interview answer.** Attention began as a way for a decoder to align with encoder states.
> Bahdanau used a learned additive scorer, Luong replaced it with a dot product, and Transformers added
> scaling for stable softmax. Self-, causal-, and cross-attention are then different choices of where
> Q/K/V come from and which positions are visible, while multi-head attention adds parallel relation
> subspaces. Later variants do not form one straight replacement chain. MQA, GQA, and MLA reduce the
> generation KV cache. Sliding window, Longformer, BigBird, and Reformer sparsify which token pairs meet.
> Linformer, Nyströmformer, and Performer approximate dense attention in different ways. FlashAttention
> preserves exact attention but executes it with tiling and online softmax, while PagedAttention manages
> cached K/V blocks efficiently during serving. I classify a new method by asking whether it changes
> roles, connectivity, approximation, exact execution, or cache storage.

### How the code changes from the baseline

```text
scaled attention  = softmax(Q @ K.T / sqrt(d)) @ V
self / cross      = change where Q, K, and V come from
causal            = add a triangular mask
MHA               = reshape into heads; run the same attention
MQA / GQA          = change only the number of K/V heads
MLA                = cache a down-projected latent instead of full K/V
sliding / Longformer / BigBird / Reformer = change the legal-pair mask
Linformer          = shorten K/V before baseline attention
Nyströmformer      = reconstruct scores through landmarks
linear / Performer = summarize K and V before queries arrive
FlashAttention     = same exact formula, evaluated tile by tile
PagedAttention     = same attention, different physical addresses for cached K/V
```

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
