# Transformers and attention

This is the highest-yield page in the workbook. Expect to write attention from memory on a whiteboard, then defend every shape and every constant. The one thing candidates get wrong is arithmetic: they can describe attention but cannot say how many parameters a block has, how many FLOPs a token costs, or how large the KV cache gets at 8k context. Know the mechanism and the numbers. Say "the $T \times T$ score matrix" when asked what makes attention quadratic — name the object, not the concept.

## The equations

**Scaled dot-product attention**

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$Q$ is $T \times d_k$, $K$ is $S \times d_k$, $V$ is $S \times d_v$; the $\sqrt{d_k}$ divisor is variance control, because a dot product of two independent zero-mean unit-variance $d_k$-vectors has variance $d_k$, and dividing by $\sqrt{d_k}$ returns it to 1 so the softmax does not saturate.

**Multi-head attention and its shapes**

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)W^O, \qquad \mathrm{head}_h = \mathrm{Attention}(XW_h^Q, XW_h^K, XW_h^V)$$

With $X$ of shape $(B, T, d)$ and $d_h = d/H$, each projection reshapes $(B, T, d) \to (B, T, H, d_h) \to (B, H, T, d_h)$, the scores are $(B, H, T, T)$, and the concatenation transposes back to $(B, T, d)$ before $W^O \in \mathbb{R}^{d \times d}$.

**Causal mask**

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}, \qquad A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)$$

Add $-\infty$ to future positions before the softmax so their exponentials are exactly zero and every row still sums to 1.

**Feed-forward block**

$$\mathrm{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{4d \times d},\ W_2 \in \mathbb{R}^{d \times 4d}$$

A position-wise two-layer MLP with expansion factor 4 and a GELU or SwiGLU nonlinearity $\phi$; it is where most of the parameters and most of the per-token compute sit.

**Residual connections and layer norm placement**

$$\text{post-norm: } x \leftarrow \mathrm{LN}(x + \mathrm{Sublayer}(x)), \qquad \text{pre-norm: } x \leftarrow x + \mathrm{Sublayer}(\mathrm{LN}(x))$$

Pre-norm keeps a clean identity path from input to output, so gradients survive depth without a warm-up schedule; that is why every modern decoder uses it.

**Layer normalisation**

$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu, \sigma^2 \text{ over the } d \text{ feature dimension}$$

Normalises per token across features, so it is independent of batch size and of sequence length; RMSNorm drops the mean subtraction and $\beta$.

**Parameter count of one block**

$$P_{\text{block}} = \underbrace{4d^2}_{\text{attention } Q,K,V,O} + \underbrace{2 \cdot m \cdot d^2}_{\text{FFN, } m = 4} = 12d^2 \ \text{(plus } 9d \text{ of biases and norms)}$$

With $m = 4$ the FFN holds $8d^2$ and attention holds $4d^2$, so two thirds of a block is the FFN.

**Total model parameters**

$$N \approx 12 L d^2 + V d$$

$L$ layers of $12d^2$ plus a $V \times d$ embedding table; positional and output-head terms are small or tied.

**FLOPs per token**

$$\text{forward} \approx 2N, \qquad \text{forward + backward} \approx 6N, \qquad \text{attention extra} \approx 4LTd$$

Every parameter is one multiply and one add per token in the forward pass, and the backward pass costs twice the forward; the attention term is separate because it does not involve parameters.

**KV cache size**

$$\text{bytes} = 2 \cdot L \cdot T \cdot d_{\text{kv}} \cdot b \cdot B$$

The 2 is $K$ and $V$, $L$ is layers, $T$ is context length, $d_{\text{kv}}$ is $n_{\text{kv}} d_h$, $b$ is bytes per element, $B$ is batch; it grows linearly in $T$ and in batch, unlike the weights.

**Cross-attention**

$$Q = Y W^Q,\quad K = H W^K,\quad V = H W^V, \qquad \mathrm{CrossAttn}(Y, H) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{pad}}\right)V$$

$Y \in \mathbb{R}^{T_{dec} \times d}$ is the decoder state and $H \in \mathbb{R}^{T_{enc} \times d}$ is the encoder output, so $Q \in \mathbb{R}^{T_{dec} \times d}$ while $K, V \in \mathbb{R}^{T_{enc} \times d}$; the score matrix is therefore $T_{dec} \times T_{enc}$ and rectangular, not square, and the output is back to $T_{dec} \times d$.

**The three attention types and their masks**

$$\underbrace{\mathrm{Attn}(H, H)}_{\text{encoder self, no mask}}, \qquad \underbrace{\mathrm{Attn}(Y, Y) + M_{\text{causal}}}_{\text{decoder self, } T_{dec} \times T_{dec}}, \qquad \underbrace{\mathrm{Attn}(Y, H) + M_{\text{pad}}}_{\text{cross, } T_{dec} \times T_{enc}}$$

Encoder self-attention is bidirectional and square over the source; decoder self-attention is square over the target and causally masked; cross-attention is rectangular, has no causal mask because the whole source is known, and carries only a padding mask over the encoder axis.

**Encoder-decoder KV cache**

$$K_{\text{enc}}, V_{\text{enc}} \ \text{computed once: } T_{enc} \times d_{\text{kv}} \ \text{fixed}, \qquad K_{\text{dec}}, V_{\text{dec}} \ \text{grow to } t \times d_{\text{kv}} \ \text{at step } t$$

The encoder keys and values depend only on the source, so you project them once and reuse them at every decode step; decoder self-attention keys and values append one row per generated token, so only that half of the cache grows.

## Code from memory

Scaled dot-product attention with a causal mask, NumPy only, checked against torch.

```python
import numpy as np, torch, torch.nn.functional as F

def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)      # subtract the max for numerical stability
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def causal_attention(Q, K, V):
    T, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)            # (T, T) scaled scores
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)  # block the future BEFORE the softmax
    A = softmax(scores, axis=-1)
    return A @ V, A

rng = np.random.default_rng(0)
T, d = 6, 8
Q, K, V = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
out, A = causal_attention(Q, K, V)

ref = F.scaled_dot_product_attention(
    *[torch.tensor(x)[None, None] for x in (Q, K, V)], is_causal=True)[0, 0].numpy()
print("max abs diff vs torch:", float(np.abs(out - ref).max()))
print("row sums of A:", np.round(A.sum(axis=1), 6))
```

Ran it: maximum absolute difference against `torch.nn.functional.scaled_dot_product_attention` was `1.39e-16`, and every attention row summed to exactly 1.

Multi-head attention with the reshape and transpose written out, one head at a time.

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    T, d = X.shape
    dh = d // n_heads
    # project, split d into (n_heads, dh), then move heads to the front
    Q = (X @ Wq).reshape(T, n_heads, dh).transpose(1, 0, 2)   # (H, T, dh)
    K = (X @ Wk).reshape(T, n_heads, dh).transpose(1, 0, 2)
    V = (X @ Wv).reshape(T, n_heads, dh).transpose(1, 0, 2)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    heads = []
    for h in range(n_heads):                                  # one head at a time, no broadcasting
        s = Q[h] @ K[h].T / np.sqrt(dh)
        heads.append(softmax(np.where(mask, -np.inf, s)) @ V[h])
    # concatenate heads back to (T, d) and mix with the output projection
    return np.concatenate(heads, axis=-1) @ Wo

rng = np.random.default_rng(1)
T, d, H = 5, 12, 3
X = rng.normal(size=(T, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
print(multi_head_attention(X, Wq, Wk, Wv, Wo, H).shape)
```

Ran against a torch reference that reshapes to $(B, H, T, d_h)$ and calls `scaled_dot_product_attention` with `is_causal=True`: maximum absolute difference `1.11e-16`. Note the head split is over the *last* dimension, so head $h$ owns feature columns $h d_h$ to $(h+1)d_h$.

A full pre-norm transformer block forward pass in PyTorch.

```python
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads, mult=4):
        super().__init__()
        self.h = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)     # one fused projection
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x):
        B, T, d = x.shape
        # pre-norm attention sublayer with a causal mask
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        shape = lambda z: z.view(B, T, self.h, d // self.h).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(
            shape(q), shape(k), shape(v), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))   # residual 1
        # pre-norm feed-forward sublayer
        return x + self.ff(self.ln2(x))                          # residual 2

blk = Block(64, 8)
print(blk(torch.randn(2, 10, 64)).shape)
print(sum(p.numel() for p in blk.parameters()), 12 * 64**2 + 9 * 64)
```

Ran it: output shape `(2, 10, 64)`, and the parameter count was `49728`, matching $12d^2 + 9d$ exactly at $d = 64$.

Cross-attention from scratch, NumPy only, with a padding mask on the encoder side.

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def cross_attention(dec_state, enc_out, Wq, Wk, Wv, src_pad_mask):
    T_dec, d = dec_state.shape
    T_enc = enc_out.shape[0]
    Q = dec_state @ Wq                       # queries from the DECODER: (T_dec, d)
    K = enc_out @ Wk                         # keys and values from the ENCODER: (T_enc, d)
    V = enc_out @ Wv
    scores = Q @ K.T / np.sqrt(d)            # rectangular: (T_dec, T_enc), no causal mask
    scores = np.where(src_pad_mask[None, :], -1e9, scores)   # pad mask on the encoder axis only
    A = softmax(scores)
    return A @ V, A

rng = np.random.default_rng(0)
T_dec, T_enc, d = 4, 7, 16
dec_state = rng.normal(size=(T_dec, d))
enc_out = rng.normal(size=(T_enc, d))
Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
src_pad_mask = np.zeros(T_enc, dtype=bool); src_pad_mask[5:] = True   # last 2 source slots are padding

ctx, A = cross_attention(dec_state, enc_out, Wq, Wk, Wv, src_pad_mask)
print("attention matrix shape (T_dec, T_enc):", A.shape)
print("context shape (T_dec, d):", ctx.shape)
print("weight on padded encoder positions:", np.abs(A[:, src_pad_mask]).max())
print("row sums:", np.round(A.sum(-1), 6))
```

Ran it:

```
attention matrix shape (T_dec, T_enc): (4, 7)
context shape (T_dec, d): (4, 16)
weight on padded encoder positions: 0.0
row sums: [1. 1. 1. 1.]
```

The matrix is 4 by 7, not square, and the two padded source positions get exactly zero weight from all four decoder positions while every row still sums to 1.

A minimal encoder-decoder transformer forward pass and greedy decode in PyTorch.

```python
import torch, torch.nn as nn
sdpa = torch.nn.functional.scaled_dot_product_attention

class EncoderBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.o(sdpa(self.q(h), self.k(h), self.v(h)))     # bidirectional, NO mask
        return x + self.ff(self.ln2(x))

class DecoderBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d) for _ in range(3))
        self.sq, self.sk, self.sv, self.so = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.cq, self.ck, self.cv, self.co = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, y, enc):
        h = self.ln1(y)
        y = y + self.so(sdpa(self.sq(h), self.sk(h), self.sv(h), is_causal=True))   # causal self-attn
        h = self.ln2(y)
        y = y + self.co(sdpa(self.cq(h), self.ck(enc), self.cv(enc)))               # cross-attn, no causal mask
        return y + self.ff(self.ln3(y))

class Seq2Seq(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.se, self.te = nn.Embedding(V, d), nn.Embedding(V, d)
        self.enc, self.dec = EncoderBlock(d), DecoderBlock(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, src, tgt):
        enc = self.enc(self.se(src))
        return self.head(self.dec(self.te(tgt), enc)), enc

torch.manual_seed(0)
V, d = 32, 24
model = Seq2Seq(V, d).eval()
src = torch.randint(0, V, (1, 7))
tgt = torch.randint(0, V, (1, 5))
with torch.no_grad():
    logits, enc = model(src, tgt)
print("src", tuple(src.shape), "-> encoder out", tuple(enc.shape))
print("tgt", tuple(tgt.shape), "-> logits", tuple(logits.shape))

with torch.no_grad():                       # greedy decode from a start token
    ys = torch.tensor([[1]])
    for _ in range(4):
        nxt = model(src, ys)[0][:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
print("greedy decoded ids:", ys.tolist()[0], "| final ys shape", tuple(ys.shape))

with torch.no_grad():                       # causality check: perturb the LAST target token
    tgt2 = tgt.clone(); tgt2[0, -1] = (tgt2[0, -1] + 1) % V
    l2 = model(src, tgt2)[0]
same = torch.allclose(logits[:, :-1], l2[:, :-1], atol=1e-6)
assert same, "causality violated"
print("logits at positions 0..T-2 unchanged by a future token:", same,
      "| max diff", float((logits[:, :-1] - l2[:, :-1]).abs().max()))
```

Ran it:

```
src (1, 7) -> encoder out (1, 7, 24)
tgt (1, 5) -> logits (1, 5, 32)
greedy decoded ids: [1, 17, 24, 27, 28] | final ys shape (1, 5)
logits at positions 0..T-2 unchanged by a future token: True | max diff 0.0
```

The source of length 7 becomes an encoder output of shape $(1, 7, 24)$, the target of length 5 becomes logits of shape $(1, 5, 32)$, and the cross-attention inside the decoder block is $5 \times 7$. Changing the last target token left every earlier logit bit-identical, maximum difference exactly `0.0`, so the causal mask holds. Note that the greedy loop re-runs the encoder each step here for brevity; a real implementation computes the encoder output once outside the loop.

## Questions

### Q1. Why do we divide the attention scores by the square root of the head dimension?

To keep the softmax out of saturation. Suppose the entries of $q$ and $k$ are independent, zero-mean, unit-variance. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ independent terms, each with mean 0 and variance $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. So $\mathrm{Var}[q \cdot k] = d_k$ and the standard deviation is $\sqrt{d_k}$. With $d_k = 128$ the raw scores would have standard deviation about 11.3, so the largest logit in a row sits many units above the rest. The softmax then puts nearly all mass on one key and its gradient goes to zero, because $\partial \mathrm{softmax}/\partial z$ scales as $p(1-p)$. Dividing by $\sqrt{d_k}$ makes the score variance 1, independent of head size. I verified this: for $d = 64$ the simulated variance of $q \cdot k$ was 63.77 against 64, and after scaling it was 0.996; for $d = 128$ it was 128.09 and 1.0007.

> **Say it.** The score is a dot product of two d-dimensional vectors. If the entries are independent with unit variance, that sum has variance d, so the scores have standard deviation root d — about eleven at head dimension one twenty-eight. Logits that spread out saturate the softmax: one key takes all the mass and the gradient, which scales like p times one minus p, vanishes. Dividing by root d restores unit score variance for any head size. I checked it numerically: variance 128 before scaling, 1.00 after.

### Q2. What does multi-head attention buy over a single head of the same total width?

Multiple independent softmaxes. A single head produces one attention distribution per query, so it can only compute one weighted average of values. Splitting $d$ into $H$ heads of size $d_h = d/H$ gives $H$ separate score matrices and $H$ separate distributions, so one head can attend to the previous token, another to the matching bracket, another to the subject of the sentence, all at the same time and all in one layer. The parameter count and the FLOPs are essentially unchanged, because $H$ heads of size $d/H$ have the same total projection size as one head of size $d$. The cost is that each head has a lower-rank view: with $d_h = 64$ the score matrix $QK^\top$ is rank at most 64. So there is a real tradeoff, and very many very small heads hurt. The output projection $W^O$ then mixes the heads, which is what makes the concatenation more than bookkeeping.

> **Say it.** One head can only form one weighted average per query. Splitting the width into H heads gives H independent attention patterns in the same layer at the same parameter cost — one head can track the previous token, another the matching bracket, another agreement. The output projection mixes them afterwards, so it is not just concatenation. The cost is that each head's score matrix is rank-limited by the head dimension, so too many tiny heads is worse, not better.

### Q3. Why is attention quadratic in sequence length, and which matrix is responsible?

The score matrix $S = QK^\top / \sqrt{d_k}$ is $T \times T$ per head per layer. That single object is the whole answer. Computing it costs $O(T^2 d)$ FLOPs, storing it costs $O(HT^2)$ memory, and the second matmul $AV$ costs another $O(T^2 d)$. Everything else in the block — the projections and the FFN — is linear in $T$, because it is applied position-wise. Concretely, for a model with $d = 4096$, $L = 32$ at $T = 8192$: the parameter-driven cost is $2N \approx 1.32 \times 10^{10}$ FLOPs per token, and the attention term $4LTd \approx 4.29 \times 10^{9}$, so attention is 24.6 percent of the forward cost at 8k and it keeps growing linearly in $T$ from there. At short context it is negligible; at long context it dominates. Sparse, linear, and sliding-window attention all attack exactly this matrix by never forming all $T^2$ entries.

> **Say it.** The T-by-T score matrix Q K transpose, per head per layer. Building it is order T squared d, and the AV product is another T squared d. Everything else in the block is position-wise, so it is linear in T. For a four-thousand-width, thirty-two-layer model at eight-k context, attention is about a quarter of the per-token FLOPs and its share grows linearly. Every efficient-attention method is an attempt to avoid materialising that one matrix.

### Q4. How is the causal mask implemented, and why negative infinity before the softmax rather than zeroing after?

Build a boolean upper-triangular mask above the diagonal and add $-\infty$ to those score entries, then apply the softmax. In code: `scores = np.where(np.triu(np.ones((T,T),bool), k=1), -np.inf, scores)`. The exponential of $-\infty$ is exactly 0, so the future contributes nothing, and the softmax denominator sums only over allowed positions, so each row still sums to 1. If instead you softmax first and then zero the future entries, the denominator already includes the future scores, so the surviving weights sum to less than 1 and are scaled by an amount that depends on the future tokens. That leaks information from the future into the present and breaks the autoregressive factorisation. In practice a large negative constant such as `-1e9` is used instead of true $-\infty$, because $-\infty$ times a zero-valued attention weight produces NaN in some backward passes. My masked implementation matched torch to `1.39e-16` and all rows summed to 1.

> **Say it.** I add negative infinity to the upper triangle of the scores and then softmax. Exp of negative infinity is exactly zero, and critically the denominator only sums over allowed positions, so each row still normalises to one. If I softmaxed first and zeroed afterwards, the denominator would still contain the future scores, so the remaining weights would depend on future tokens — an information leak that breaks the causal factorisation. In practice I use minus one e nine rather than true infinity to avoid NaNs in the backward pass.

### Q5. Encoder-only, decoder-only, encoder-decoder. What is each for?

Encoder-only, such as BERT, uses bidirectional attention with no mask, so every token sees every other token. That is right for understanding tasks — classification, retrieval embeddings, token labelling — and it cannot generate, because it has no causal factorisation and is trained by masked-token prediction. Decoder-only, such as GPT and Llama, uses a causal mask and next-token prediction, so it models $\prod_t P(x_t \mid x_{<t})$ directly. It generates, and because the task is the same at every position it gets a training signal from every token, which makes it the most compute-efficient pretraining objective. Encoder-decoder, such as T5, encodes the source bidirectionally and decodes causally with cross-attention into the encoder states. That fits genuine sequence-to-sequence tasks like translation, where the source is fully known and the target is generated. Decoder-only has largely won for general models, because concatenating source and target into one stream gets most of the benefit with one set of weights.

> **Say it.** Encoder-only is bidirectional with no mask — good for classification and embeddings, cannot generate. Decoder-only is causal next-token prediction, which gives a loss at every position and generates directly, so it is the most compute-efficient objective and it is what general models use. Encoder-decoder encodes the source bidirectionally and cross-attends from a causal decoder, which suits translation where the whole source is known up front. Decoder-only won mostly because you can just concatenate source and target into one stream.

### Q6. What does the KV cache store, how big does it get, and why does it dominate memory at long context?

It stores the key and value vectors for every past token, at every layer, so that generating token $t+1$ does not recompute attention over the whole prefix. Only $K$ and $V$ are cached, never $Q$, because the new token supplies its own single query. The size is $2 \cdot L \cdot T \cdot d_{\text{kv}} \cdot b \cdot B$ bytes. Take $d = 4096$, $L = 32$, fp16 so $b = 2$, batch 1, full multi-head so $d_{\text{kv}} = d$. Then per token it is $2 \times 32 \times 4096 \times 2 = 524{,}288$ bytes, that is 512 KiB per token, and at $T = 8192$ that is exactly 4.0 GiB. The weights of that model are about 6.57 billion parameters, so 13.1 GB in fp16 — fixed. The cache is not fixed: it scales linearly in both context length and batch size, so at long context and any real batch it passes the weights and becomes the binding constraint on how many concurrent requests fit.

> **Say it.** It caches the key and value vectors of every past token at every layer, so each new token attends over the prefix without recomputing it. Queries are never cached. Size is two — for K and V — times layers, times tokens, times the KV width, times bytes per element, times batch. For a four-thousand-wide thirty-two-layer model in fp16 that is five hundred twelve kibibytes per token, so four gibibytes at eight-k context for a single sequence. Weights are fixed; the cache grows with context and batch, so it becomes the limit.

### Q7. Explain MQA and GQA and the tradeoff.

Multi-query attention keeps $H$ query heads but a single shared key head and value head. Grouped-query attention is the interpolation: $H$ query heads share $G$ key-value heads, with $G$ typically 4 or 8. The queries are unaffected, so the compute is nearly unchanged; what shrinks is $d_{\text{kv}} = G d_h$ in the cache formula. Using the same model as before — $d = 4096$, 32 heads of 128, $L = 32$, fp16 — full multi-head costs 512 KiB per token, and GQA with 8 key-value groups costs $2 \times 32 \times 1024 \times 2 = 128$ KiB per token, a 4-fold reduction. MQA with one group would be 32-fold. The cost is quality: fewer distinct key-value subspaces means less expressive attention, and MQA measurably degrades on some tasks. GQA at 8 groups is the usual compromise because it captures most of the memory saving with a small quality loss. It also speeds decoding, which is memory-bandwidth bound.

> **Say it.** Multi-query keeps all the query heads but shares one key-value head; grouped-query shares a small number of key-value heads, usually four or eight, across all query heads. Compute barely changes; the KV cache shrinks by the head-to-group ratio. On a thirty-two-head model, eight groups cut the cache from five hundred twelve to one hundred twenty-eight kibibytes per token. The cost is expressiveness — fewer distinct key-value subspaces. Full MQA loses noticeable quality, so eight-group GQA is the usual compromise, and it also helps because decoding is bandwidth-bound.

### Q8. Why is the feed-forward layer usually 4 times the model dimension?

It is empirical, not derived. The FFN is the only place with a position-wise nonlinearity, so it is where per-token feature computation happens; attention only mixes across positions and is linear given the weights. Widening the hidden layer increases capacity, and 4 was the value in the original transformer that gave good quality per parameter, so it stuck. The arithmetic matters: the FFN holds $2 \times 4 d^2 = 8d^2$ parameters against attention's $4d^2$, so with $m = 4$ two thirds of every block is the FFN. Raising $m$ therefore buys capacity at a steep parameter cost, and lowering it starves the nonlinear computation. Models using SwiGLU need three matrices instead of two, so they set $m \approx 8/3$ to keep the same total, which is where the odd hidden sizes in Llama-style models come from. Treat 4 as a well-tested default, not a theorem.

> **Say it.** It is empirical. The feed-forward layer is the only position-wise nonlinearity, so it is where per-token computation happens, while attention just mixes positions. Four was the original choice and it held up on quality per parameter. The arithmetic is the point: at four times width the FFN has eight d squared parameters against attention's four, so two thirds of a block is FFN. SwiGLU uses three matrices instead of two, so those models use about eight-thirds to keep the total the same. I would call it a default, not a theorem.

### Q9. What happens if you remove the residual connections?

Deep transformers stop training. The residual makes each sublayer compute $x + f(x)$, so the Jacobian is $I + \partial f / \partial x$. Backpropagating through $L$ layers multiplies $L$ such Jacobians, and the identity term keeps a direct path with gain 1 from the loss to every layer. Without it the gradient is a product of $L$ sublayer Jacobians, whose singular values are generically not 1, so the gradient norm decays or explodes geometrically in depth. A 2-layer model without residuals still trains; a 32-layer one does not. There is a second effect specific to attention: attention output is a convex combination of value vectors, so repeated attention without a residual drives all token representations toward each other — token uniformity, or rank collapse. The residual injects the token's own representation back at every layer, which preserves the distinction between positions. Pre-norm placement matters for the same reason: it keeps the identity path free of any normalisation.

> **Say it.** Deep models stop training. Each sublayer computes x plus f of x, so the Jacobian is identity plus something, and backprop through L layers keeps a gain-one path to every layer. Without residuals the gradient is a product of L Jacobians and decays or explodes geometrically. There is also a representational effect: attention output is a convex combination of values, so stacking attention without residuals pushes all token vectors together — rank collapse. The residual re-injects each token's own representation. Pre-norm keeps that identity path clean, which is why it needs no warm-up.

### Q10. How would you extend a trained model's context length?

Positions are the problem, not the architecture — attention itself is length-agnostic. With RoPE the standard route is to change the rotation frequencies rather than retrain from scratch. Position interpolation divides the position index by a factor $s$, so positions $0..sT$ map into the range the model already saw; it works but it compresses high-frequency detail and costs a little short-context quality. NTK-aware scaling instead raises the RoPE base $\theta$ from 10,000 to a larger value, which stretches the low-frequency dimensions a lot and the high-frequency ones barely, preserving local resolution. YaRN combines interpolation with a per-frequency schedule and a logit temperature. Any of these needs a short fine-tune on long sequences, typically a small fraction of pretraining tokens. Then you must pay the serving cost: the KV cache is linear in $T$, so going from 8k to 128k multiplies it by 16, from 4.0 GiB to 64 GiB per sequence on the model above. Also check that your training data actually contains long-range dependencies.

> **Say it.** Attention has no length limit; the position encoding does. With RoPE I would rescale the frequencies — either interpolate the position index down into the trained range, or raise the base from ten thousand, which stretches the low frequencies while leaving local resolution intact. Then a short fine-tune on long sequences. After that it is a serving problem: the KV cache is linear in context, so eight-k to one-twenty-eight-k is sixteen times the cache, four gibibytes to sixty-four. And the long data has to contain real long-range dependencies.

### Q11. Explain flash attention. What does it change asymptotically and what does it not?

Flash attention is an IO-aware, tiled implementation of exactly the same mathematical function. Standard attention writes the full $T \times T$ score matrix and the full attention matrix to high-bandwidth memory, then reads them back, so it is bound by memory traffic rather than by arithmetic. Flash attention loads blocks of $Q$, $K$, and $V$ into on-chip SRAM, computes each block of scores there, and accumulates the output using the online softmax trick, which carries a running maximum and a running normaliser so the softmax can be updated incrementally without ever holding a full row. Memory drops from $O(T^2)$ to $O(T)$, and that is a real asymptotic change. The FLOPs do not change: it is still $O(T^2 d)$. For the backward pass it recomputes the scores instead of storing them, trading extra FLOPs for far less memory traffic, and still comes out several times faster in wall-clock time. The outputs are numerically equivalent to a standard implementation up to floating-point reassociation.

> **Say it.** It is the same function, computed with better memory movement. Standard attention writes the whole T-by-T matrix to HBM and reads it back, so it is bandwidth-bound. Flash tiles Q, K and V into on-chip SRAM and accumulates the output with an online softmax that carries a running max and normaliser, so the full matrix is never materialised. Memory goes from order T squared to order T — a genuine asymptotic win. FLOPs stay order T squared d; it does not change the asymptotic compute, only the constant and the memory.

### Q12. Where do the parameters actually live in a transformer? Do the arithmetic.

Per block: attention holds $W^Q, W^K, W^V, W^O$, each $d \times d$, so $4d^2$. The FFN holds $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$, so $8d^2$. Total $12d^2$ per block, of which two thirds is the FFN. Layer norms contribute $4d$ and biases a further $5d$, which is negligible. I verified this in PyTorch: a block with $d = 64$, 8 heads reported 49,728 parameters, exactly $12d^2 + 9d$. Scaling up with $d = 4096$, $L = 32$, $V = 32{,}000$: each block is $12 \times 4096^2 = 201.3$ million, times 32 layers is 6.44 billion, plus an embedding table of $32{,}000 \times 4096 = 131$ million, giving 6.57 billion total. So the embedding is 2 percent and the blocks are 98 percent, and inside the blocks the FFN is two thirds. In small models with large vocabularies the embedding fraction is much larger, which is why tied input and output embeddings matter there.

> **Say it.** Four d squared in attention for Q, K, V and O, and eight d squared in the feed-forward at four-times expansion, so twelve d squared per block, two thirds of it FFN. Norms and biases are order d and negligible. At width four thousand ninety-six that is two hundred one million per block; thirty-two layers is six point four billion, plus a hundred thirty-one million of embeddings for a thirty-two-thousand vocabulary — six point five seven billion total, so embeddings are two percent. I checked the formula in code and it matched exactly.

### Q13. How many FLOPs does it take to train a model? Where does the 6N come from?

About $6N$ FLOPs per parameter-token, so total training compute is roughly $6ND$ for $N$ parameters and $D$ training tokens. The forward pass is $2N$: every weight participates in one multiply and one add per token, and a matmul of a $d \times d$ weight against one token vector is $d^2$ multiplies and $d^2$ adds. The backward pass is $4N$, because it computes two things — the gradient with respect to the inputs and the gradient with respect to the weights — and each costs about as much as the forward matmul. So $2N + 4N = 6N$. This excludes the attention score computation, which adds $4LTd$ per token in the forward pass and is separate because it involves no parameters; at $d = 4096$, $L = 32$, $T = 8192$ that extra term is $4.3 \times 10^9$ against $2N = 1.32 \times 10^{10}$, so about 25 percent. Inference generation is only $2N$ per token, plus the cache read, and it is bandwidth-bound rather than compute-bound.

> **Say it.** Six N D. Forward is two N, because each weight does one multiply and one add per token. Backward is four N, because it computes both the gradient to the inputs and the gradient to the weights, and each of those costs about a forward pass. Two plus four is six. That excludes attention scores, which are parameter-free and add four L T d per token — about a quarter of the total at eight-k context on a six-billion model. Inference is only two N per token and is bandwidth-bound.

### Q14. Where does layer norm go, and why does the choice matter?

Post-norm normalises after the residual add: $x \leftarrow \mathrm{LN}(x + f(x))$. Pre-norm normalises the sublayer input and leaves the residual path untouched: $x \leftarrow x + f(\mathrm{LN}(x))$. The difference is whether the identity path passes through a normalisation. In post-norm it does, so the gradient reaching layer $\ell$ is multiplied by the Jacobian of every later layer norm, and deep post-norm models need a learning-rate warm-up and careful initialisation or they diverge. In pre-norm the identity path is clean addition, so gradients reach every layer at gain 1 and training is stable to great depth with no warm-up. The cost of pre-norm is that activations grow across depth, since nothing rescales the residual stream, so a final layer norm before the output head is required. Modern decoders use pre-norm with RMSNorm, which drops the mean subtraction and the bias and is slightly cheaper with no measured quality loss.

> **Say it.** Post-norm normalises after the residual add, so the identity path goes through a normalisation and the gradient picks up every later layer norm's Jacobian — that is why post-norm needs warm-up and careful initialisation at depth. Pre-norm normalises only the sublayer input, so the residual path is pure addition and gradients reach every layer at gain one. The cost is that the residual stream grows with depth, so you need a final norm before the output head. Modern decoders use pre-norm with RMSNorm.

### Q15. Encoder-decoder or decoder-only? Why did the field mostly move to decoder-only?

An encoder-decoder has two stacks: an encoder that reads the source bidirectionally with no mask, and a decoder that runs causal self-attention, then cross-attends into the encoder output, then a feed-forward. A decoder-only model has one stack with a causal mask over a single concatenated stream. Encoder-decoder suits tasks where the source is fully known and distinct from the target: translation, summarisation, speech transcription. Be honest about the comparison. A decoder-only model with the source in the prompt does sequence-to-sequence perfectly well, and there is no proven capability gap. The argument for decoder-only is training simplicity and scaling: one stack, one objective, a loss at every token, no separate mask logic, and it consumes any raw text corpus. Encoder-decoder gets a loss only on target tokens and needs paired data or a span-corruption objective. So decoder-only won on engineering economics, and encoder-decoder persists where the source is long, fixed, and decoded from repeatedly.

> **Say it.** Encoder-decoder is two stacks: a bidirectional encoder over the source, and a causal decoder that cross-attends into it. Decoder-only is one causal stack over source and target concatenated. I would not claim a capability gap — a decoder-only model with the source in the prompt does seq2seq fine. The case for decoder-only is training economics: one objective, a loss at every token, any raw corpus, no separate mask logic. Encoder-decoder survives in translation and speech, where the source is fixed and you decode from it many times.

### Q16. How does cross-attention differ from self-attention mechanically?

Only in where the three projections read from. In self-attention $Q$, $K$ and $V$ are all projections of the same tensor. In cross-attention the queries come from the decoder state $Y$ and the keys and values come from the encoder output $H$: $Q = YW^Q$, $K = HW^K$, $V = HW^V$. That makes the shapes asymmetric. $Q$ is $T_{dec} \times d$, $K$ and $V$ are $T_{enc} \times d$, so $QK^\top$ is $T_{dec} \times T_{enc}$ — rectangular whenever the source and target lengths differ. The output is $A V$, which is $T_{dec} \times d$, so the decoder stream keeps its own length. The softmax still runs over the last axis, so each decoder position forms one distribution over source positions. My code printed a 4 by 7 matrix for 4 decoder and 7 encoder positions. Everything else — the $\sqrt{d_k}$ scaling, multi-head splitting, the output projection — is unchanged.

> **Say it.** Same operation, different sources. Queries come from the decoder, keys and values from the encoder output. So Q is T-decoder by d, K and V are T-encoder by d, and the score matrix is T-decoder by T-encoder — rectangular, because the source and target lengths are independent. The softmax is still over the last axis, so each target position gets one distribution over source positions, and the result is back to T-decoder by d. In my code it printed four by seven. Scaling, heads and the output projection are identical.

### Q17. What does the encoder actually contribute that a decoder-only prompt does not?

Two concrete things. First, bidirectional context over the source. Every source token is encoded while seeing tokens to its right as well as its left, so a word whose sense is fixed later in the sentence is represented correctly the first time. In a decoder-only prompt the source tokens are causally masked, so each source token sees only its own left context. Second, a fixed source representation. The encoder output depends only on the source, so you compute $K_{\text{enc}}$ and $V_{\text{enc}}$ once and reuse them at every decode step; only the decoder self-attention cache grows, by one row per generated token. In a decoder-only model the source occupies KV cache that grows with the target too, and the whole prompt must be prefilled per request. For a long source decoded many times — beam search, multiple candidates, reranking — that reuse is a real inference saving. Neither is a capability the decoder-only model lacks; both are efficiency and representation quality.

> **Say it.** Bidirectionality over the source, and a source representation computed once. Encoder tokens see their right context, so ambiguity resolved later in the sentence is captured immediately; in a decoder-only prompt the source is causally masked. And the encoder output depends only on the source, so its keys and values are projected once and reused at every decode step, while only the decoder self-attention cache grows. For a long source you decode from repeatedly — beam search, reranking — that is a genuine saving. It is efficiency and representation quality, not capability.

### Q18. State the masking rules for all three attention types, and the bug from getting cross-attention wrong.

Encoder self-attention: no causal mask, because the full source is available; a padding mask over the source only, so padded slots get $-\infty$. Decoder self-attention: a causal mask, upper triangle set to $-\infty$, plus a target padding mask during training. Cross-attention: no causal mask, because every target position may look at every source position, plus a padding mask over the encoder axis. The mask shapes follow the score shapes: $T_{enc} \times T_{enc}$, $T_{dec} \times T_{dec}$, $T_{dec} \times T_{enc}$. The bug is applying a causal mask in cross-attention. Then decoder position $i$ can only see source positions $j \le i$, so the model silently learns a monotonic, truncated alignment. Target position 0 sees one source token; the tail of the source is invisible unless the target is at least as long as the source. It does not crash, because the mask is broadcastable when the lengths happen to match, and loss still falls. It shows up as translations that drop the end of the sentence.

> **Say it.** Encoder self-attention: no causal mask, source padding only. Decoder self-attention: causal mask plus target padding. Cross-attention: no causal mask, padding on the encoder axis only. The shapes are T-enc square, T-dec square, and T-dec by T-enc. The classic bug is reusing the causal mask in cross-attention. Then target position i sees only source positions up to i, so you get a forced monotonic alignment and the end of the source is never attended to. It does not crash and the loss still falls — it shows up as truncated translations.

## Done when

- You can write causal scaled dot-product attention in NumPy from memory in under five minutes and it matches `torch.nn.functional.scaled_dot_product_attention` to floating-point precision.
- You can state the $(B, T, H, d_h) \to (B, H, T, d_h)$ reshape and transpose for multi-head attention without hesitating about which axis the head split takes.
- You can compute a block's parameter count as $12d^2$, a model's as $12Ld^2 + Vd$, and the KV cache as $2LTd_{\text{kv}}b$ for a stated model, in your head, in under a minute.
- You can derive the $\sqrt{d_k}$ scaling from the variance of a dot product and say why $-\infty$ goes in before the softmax, not zeros after.
- You can draw an encoder-decoder block in under two minutes, name the three attention types with their mask rules and score shapes ($T_{enc} \times T_{enc}$, $T_{dec} \times T_{dec}$, $T_{dec} \times T_{enc}$), and say which side of the KV cache is fixed and which grows during decoding.
