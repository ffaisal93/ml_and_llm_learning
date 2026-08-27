# Transformers — Interview Grill

> 60 brutal questions on the transformer architecture. Drill until you can answer 40+ cold.

---

## A. Core architecture

**1. What does a transformer block do?**
Two operations: attention (mixes information across positions) and FFN (per-token non-linear computation), each wrapped in a residual connection and preceded by LayerNorm in modern (pre-LN) transformers. That's the entire architecture; everything else elaborates these two.

> **Saying it out loud.** A transformer block only does two things, and everything else is decoration. It mixes information *between* tokens with attention, then it thinks about each token *on its own* with the feed-forward net. Both are wrapped in residual connections so the original signal always has a clean path through, and both get a LayerNorm on the way in. If you stack that block ninety times you have a modern LLM — the interesting part is that there is no third mechanism hiding anywhere.

**2. Walk me through scaled dot-product attention.**
Project input $X$ into queries $Q = X W_Q$, keys $K = X W_K$, values $V = X W_V$. Compute $\text{scores} = Q K^\top / \sqrt{d_k}$, apply softmax row-wise to get attention weights, multiply by $V$ to get output. Each output position is a weighted average of all values, where the weights come from query-key compatibility.

> **Saying it out loud.** Attention is a soft lookup table. Every token writes a query saying what it's looking for, a key advertising what it has, and a value that's the actual content it will hand over. You dot every query against every key to get a compatibility score, divide by $\sqrt{d_k}$, softmax the scores into weights that sum to one, and then each token's output is just a weighted average of everybody's values. So a token ends up being a blend of the tokens it cared about most.

**3. Why divide by $\sqrt{d_k}$?**
**Without it, scores get huge in high dim, softmax saturates, gradients die.** The math: dot product of two $d_k$-dim unit-variance vectors has variance $d_k$, so entries grow as $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ keeps score variance $O(1)$ regardless of dimension → softmax stays in its linear regime, gradients flow.

> **Saying it out loud.** Without the scaling, softmax saturates and training dies. Here's the intuition: a dot product of two $d_k$-dimensional random vectors is a sum of $d_k$ little products, so its spread grows like $\sqrt{d_k}$. At $d_k = 128$ that's scores in the tens, and softmax of scores in the tens is basically a one-hot — one weight at 1.0, everything else at zero, and a gradient of essentially zero everywhere. Dividing by $\sqrt{d_k}$ pins the variance back near 1 no matter how wide you go, so the softmax stays in the region where it actually has a gradient.

**4. Why softmax and not sigmoid?**
Softmax produces a **probability distribution** over keys: non-negative weights summing to 1. This makes $\mathrm{softmax}(\text{scores}) \cdot V$ a proper convex combination — a weighted average. Sigmoid would give independent gates per key, not a normalized mixture, and you'd need ad-hoc renormalization. Softmax also sharpens: largest score dominates exponentially.

> **Saying it out loud.** Softmax because we want a weighted average, not a bunch of independent volume knobs. The weights come out non-negative and summing to one, which makes the output a genuine convex combination of the values — the token can't blow up its own magnitude just by attending to lots of things. Sigmoid would give each key its own independent gate, so attending to twenty tokens gives you twenty times the output scale, and you'd have to bolt on a normalization anyway. Softmax also has a nice side effect: because it exponentiates, the winner dominates, so attention can be genuinely selective when it needs to be.

**5. What's the computational complexity of attention?**
$O(N^2 \cdot d)$ time and $O(N^2)$ memory for the attention matrix. The $N^2$ is the limiting factor for long contexts. FlashAttention reduces memory access but not FLOPs; linear attention variants try to reduce FLOPs at some quality cost.

> **Saying it out loud.** It's quadratic in sequence length: $O(N^2 d)$ time and $O(N^2)$ memory for the score matrix. That's because every token compares itself to every other token — 1,000 tokens is a million comparisons, 100,000 tokens is ten billion. That $N^2$ is the whole reason long context is hard and the reason half of modern systems work exists. FlashAttention makes the memory side behave without changing the FLOPs; linear attention actually cuts the FLOPs but usually gives up some quality to do it.

**6. What's multi-head attention and why?**
Split $d$ into $h$ heads of size $d_h = d/h$. Run attention separately in each head; concatenate outputs; project. Same total parameters as single-head but allows the model to attend to multiple patterns simultaneously (e.g., one head for syntax, another for coreference). Empirically, ~5–10 heads provide most of the benefit.

> **Saying it out loud.** Multi-head is just running several smaller attentions in parallel instead of one big one. You split the $d$-dimensional space into $h$ chunks of size $d/h$, each does its own queries, keys, and its own softmax, then you concatenate and project. The point is that one softmax can only really commit to one pattern at a time — if a token needs to look at its syntactic head *and* at the last mention of the same entity, a single head has to pick. Same parameter count, more patterns at once, and empirically you get most of the benefit by the time you have five to ten heads.

**7. What does each attention head learn?**
Empirical analyses show some heads learn interpretable functions (positional, syntactic dependency, coreference) but most are not cleanly interpretable. Ablating individual heads usually costs little, suggesting heavy redundancy. Voita et al., Clark et al. for canonical analyses.

> **Saying it out loud.** Honestly, most heads are not interpretable, and you should say that out loud rather than overclaim. A handful do have clean stories — attend to the previous token, attend to your syntactic head, attend to the earlier mention of this entity. But the majority look like noise to us, and the tell is that you can ablate individual heads and quality barely moves, which says the network is heavily redundant. So the honest framing is: heads specialize somewhat, redundantly, and the interpretable ones are the exception rather than the rule.

**8. What does the FFN do?**
Per-token non-linear computation. $\mathrm{FFN}(x) = W_2 \cdot \mathrm{activation}(W_1 \cdot x)$. The 4× expansion ($W_1: d \to 4d$, $W_2: 4d \to d$) is the standard ratio. Provides non-linearity (attention is linear in values) and acts as a key-value memory holding factual knowledge. Holds 2/3+ of all transformer parameters.

> **Saying it out loud.** The FFN is where the model actually stores what it knows. Attention moves information around, but attention is linear in the values — without the FFN, stacking blocks would collapse into one big linear map. The FFN takes each token separately, blows it up 4x, applies a non-linearity, and squeezes it back, which acts a lot like a key-value memory: the first matrix matches patterns, the second one writes out the associated fact. And it's not a side dish — two-thirds or more of all the parameters in a transformer live there.

**9. Why 4× expansion in the FFN?**
Empirical choice from the original paper. Larger $d_{\text{ff}}/d$ improves quality up to a point; 4× is the sweet spot for vanilla FFN. Some modern architectures use 8/3 with SwiGLU activation (Llama). Going much larger increases parameters without proportional benefit.

> **Saying it out loud.** Four is empirical, not derived — it came out of the original paper and stuck because it kept working. The tradeoff is that widening the FFN buys you quality with diminishing returns while costing parameters linearly, and 4x is roughly where the curve bends. Modern models complicate it: SwiGLU uses three matrices instead of two, so LLaMA-style models drop the ratio to 8/3 to keep the parameter count in the same place. If someone pushes you, the honest answer is that it's a well-tested default, not a law.

**10. What's SwiGLU and why is it used?**
Gated activation: $\mathrm{SwiGLU}(x) = (\mathrm{Swish}(x W_1) \odot x W_2) W_3$. Adds a gating term that empirically outperforms ReLU/GELU for LLMs. Triples the FFN parameter count compared to single-matrix FFN, so the inner dimension is reduced (often to $8/3 \cdot d$) to keep total parameters comparable.

> **Saying it out loud.** SwiGLU is a gate bolted onto the FFN. Instead of one path through the non-linearity, you compute two projections, run one through a Swish, and multiply them elementwise — so one branch decides how much of the other branch gets through. It consistently beats plain ReLU or GELU by a small but real margin on language modeling, which is why every recent open model uses it. The catch is that it's three matrices instead of two, so you shrink the inner dimension to about $8/3 \cdot d$ to keep the parameter count honest.

**11. What's the role of $W_O$ in multi-head attention?**
The output projection. After $h$ heads each produce a $d_h$-dim output, you concatenate to get a $d$-dim vector, then project with $W_O: d \to d$. $W_O$ allows the model to mix information across heads — without it, each head's output would be confined to its own subspace.

> **Saying it out loud.** $W_O$ is what lets the heads talk to each other. Each head writes into its own little $d_h$-sized slot of the concatenated vector, so without an output projection, head three's findings would be permanently stuck in dimensions 192 through 255 and nothing downstream could combine them. $W_O$ is a full $d \times d$ mix that reads across all the slots at once. It's easy to dismiss it as a formality, but the interpretability people treat $W_O V$ as the real 'what does this head write to the residual stream' operator.

**12. Why are residual connections critical?**
Gradient flow. Without them, gradients through depth multiply across layers and vanish. With them, the gradient w.r.t. early layers includes an identity term (the residual passes the gradient through unchanged), making vanishing-through-depth impossible. Enables stacking 100+ layers.

> **Saying it out loud.** Residuals exist so gradients survive depth. Without them, the gradient reaching layer one is a product of ninety Jacobians, and a product of ninety things that are each a bit less than one is effectively zero — training just stalls. With a residual, the derivative through each block is identity-plus-something, so there's always a path where the gradient passes through untouched. That's the whole reason we can train hundred-layer networks at all; take residuals out and even a modestly deep transformer becomes untrainable.

**13. What's the "residual stream" view?**
Each token's representation flows through layers as a stream. Each block reads from the stream (via LayerNorm + projections), computes something, and writes back via the residual $+$. Layers communicate by reading and writing to this shared stream — like a bus in computer architecture. This is the modern interpretability framing.

> **Saying it out loud.** Picture the residual stream as a shared bus running down the length of the model, one per token. Each block doesn't transform the stream — it *reads* a normalized copy of it, computes something, and adds its result back on. So layers communicate by writing messages into a common space and later layers reading them out, which is why you can meaningfully ask 'what did head 7 in layer 12 write?' It's the framing behind logit lens, activation patching, and most of modern interpretability, and it's a much better mental model than 'each layer transforms the input'.

---

## B. Normalization placement

**14. Pre-LN vs post-LN?**
Post-LN (original, 2017): $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Norm after residual.
Pre-LN (modern): $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. Norm before sublayer; residual is unnormed.
Pre-LN trains stably without elaborate warmup; post-LN does not at modern scales. Every modern LLM uses pre-LN or RMSNorm.

> **Saying it out loud.** The difference is whether the normalization sits inside the residual path or outside it. Post-LN, the original, adds the sublayer output and then normalizes the whole sum — so the clean residual highway gets squashed at every block. Pre-LN normalizes the input to the sublayer instead and leaves the residual untouched all the way down. Post-LN needs careful warmup and still gets flaky at scale, so every modern LLM is pre-LN or an RMSNorm variant of it.

**15. Why does pre-LN train more stably?**
**Pre-LN keeps a clean signal flowing through the residual stream; post-LN keeps re-normalizing it and amplifies any wobble.** Mechanically: in pre-LN, the sublayer reads a normed input but writes back to the unnormed residual — the residual path's gradient flows unchanged. Post-LN renormalizes the stream every block, which can amplify small perturbations into instability.

> **Saying it out loud.** Pre-LN keeps the highway clean; post-LN puts a toll booth on it every block. Concretely, in pre-LN the gradient can flow from the loss straight down the residual path with nothing multiplying it, whereas post-LN re-normalizes the stream after every addition and small perturbations get re-amplified layer after layer. The practical symptom is loss spikes and divergence in post-LN at large depth and large learning rate, which people used to paper over with long warmup schedules. The cost of pre-LN, worth naming, is that the residual stream's magnitude grows with depth, so you need a final norm before the unembedding.

**16. What's RMSNorm and why is it used?**
LayerNorm: $(x - \mu) / \sigma$. RMSNorm: $x / \mathrm{RMS}(x)$ where $\mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2)}$. Just unit-variance normalization, no mean centering. ~30% cheaper (one fewer reduction). Empirically as good as LayerNorm. Used in LLaMA, Gemma, Mistral, etc.

> **Saying it out loud.** RMSNorm is LayerNorm with the mean subtraction thrown away. You just divide by the root-mean-square of the vector and scale by a learned gain — no centering, no mean statistic. It turns out the centering was doing nothing measurable for quality, and dropping it saves you a reduction pass, so it's roughly 30% cheaper in a place the model hits hundreds of times per forward. That's why LLaMA, Mistral, and Gemma all use it: same quality, free speed.

**17. Why not BatchNorm in transformers?**
BN normalizes across the batch dimension, which is bad for sequences: (a) different sequence lengths in a batch, (b) at inference time you may want to process single sequences without batch statistics, (c) the running statistics lag during training. LayerNorm normalizes per-token, sidestepping all of these.

> **Saying it out loud.** BatchNorm ties every example's normalization to whatever else happens to be in the batch, and that's a disaster for sequences. Your batch has sequences of different lengths, so the statistics are computed over a ragged mess of real tokens and padding; at inference you might be serving a single sequence with no batch at all; and the running averages lag whenever the distribution shifts. LayerNorm normalizes across the feature dimension of one token, so it's completely independent of batch composition and behaves identically at train and test time. The rule of thumb: BatchNorm for fixed-size images, LayerNorm for variable-length sequences.

---

## C. Positional information

**18. Why do transformers need positional encoding?**
Pure attention is permutation-equivariant: shuffle the input tokens and the output is shuffled the same way. So attention has no notion of order. Positional encoding injects the position information that the architecture otherwise lacks.

> **Saying it out loud.** Attention has no idea what order anything is in. If you shuffle the tokens, the attention outputs shuffle right along with them — the mechanism is a set operation, not a sequence operation, because dot products don't care where their operands sat. So 'dog bites man' and 'man bites dog' would be literally identical to the model without extra help. Positional encoding is the patch: you inject the position into the representation so the model can tell where things are.

**19. What's sinusoidal positional encoding?**
The original method. Add deterministic sinusoidal vectors per position: $\mathrm{PE}(t, 2i) = \sin(t / 10000^{2i/d})$, $\mathrm{PE}(t, 2i+1) = \cos(\cdots)$. Properties: same encoding regardless of training, in principle extrapolates to longer sequences than training (in practice, mediocre extrapolation).

> **Saying it out loud.** The original trick was to give every position a fixed fingerprint made of sines and cosines at geometrically spaced frequencies. Low-frequency dimensions change slowly so they encode roughly where you are in the document; high-frequency ones flip fast so they distinguish neighbors. The elegant claim was that because $\sin(a+b)$ decomposes, relative offsets become linear functions of the encoding, so in principle it extrapolates past the training length. In practice extrapolation is mediocre, which is exactly why nobody uses it anymore.

**20. What's learned positional encoding?**
Treat position as a categorical feature; learn an embedding per position. Used in BERT, GPT-2. Simple, works well within max-position seen in training, **does not extrapolate** beyond.

> **Saying it out loud.** Learned positional encoding is the lazy version and it works fine — you just make an embedding table indexed by position and add it in, exactly like a token embedding. BERT and GPT-2 both do this. The upside is zero cleverness required and it fits whatever weird positional structure the data has. The fatal downside is that position 4,097 has no row in the table, so the model doesn't just degrade past its training length, it's completely undefined.

**21. What's RoPE?**
Rotary Position Embedding (Su et al. 2021). Encode position by **rotating** $Q$ and $K$ by an angle proportional to position before computing the dot product. The dot product $Q \cdot K$ then depends on the **relative position** (position difference), not absolute positions. Standard in modern LLMs (LLaMA, Mistral, GPT-J, etc.).

> **Saying it out loud.** RoPE encodes position by rotating instead of adding. You take each query and key, chop them into 2D pairs, and rotate each pair by an angle proportional to the token's position before you do the dot product. The magic is that when you dot a query rotated by $m$ against a key rotated by $n$, the rotations partially cancel and what's left depends only on $m - n$ — so absolute position goes in but *relative* position comes out. That's why it's the default in LLaMA, Mistral, Qwen, and basically everything modern, and why context-extension tricks like YaRN work by messing with the rotation frequencies.

**22. What's ALiBi?**
Attention with Linear Biases (Press et al. 2021). Add a per-head bias to attention scores that linearly penalizes attending to distant positions: $\text{scores} \mathrel{-}= m_h \cdot |i - j|$, where $m_h$ is a head-specific slope. No positional embeddings at all. Extrapolates well to longer sequences. Used in BLOOM and others.

> **Saying it out loud.** ALiBi throws out positional embeddings entirely and just penalizes distance in the scores. Before the softmax you subtract a slope times the gap between the two positions, with a different slope per head — so some heads look mostly local and others tolerate long range. Because it's a bias on the score and not a learned embedding, nothing breaks when you go past the training length; it just keeps subtracting more. It extrapolates beautifully, which is why BLOOM used it, but the tradeoff is that the built-in recency bias makes exact long-range recall harder.

**23. Why does RoPE extrapolate better than learned PE?**
RoPE encodes **relative** positions, so it doesn't matter what absolute position you've seen at training. Learned PE only knows positions seen in training. When you go beyond, learned PE is undefined; RoPE just keeps rotating. (RoPE still has practical limits — frequencies trained at short range may not generalize. YaRN and dynamic NTK scaling extend this.)

> **Saying it out loud.** Because RoPE never memorizes an absolute position — it only ever produces a function of the *gap*. Learned PE has a lookup table, and past the last row there's simply nothing there, so quality falls off a cliff rather than degrading. RoPE just keeps rotating, so position 10,000 is a perfectly well-defined thing. Caveat worth saying so you don't sound naive: RoPE still degrades past training length, because the low-frequency dimensions never completed a full rotation during training, and that's exactly the hole YaRN and NTK-aware scaling patch.

---

## D. Encoder vs decoder

**24. Encoder vs decoder vs encoder-decoder?**
Encoder: bidirectional attention, no mask, contextualizes input (BERT, embeddings).
Decoder: causal attention, autoregressive generation (GPT, modern LLMs).
Encoder-decoder: encoder processes source, decoder generates target with cross-attention to encoder (T5, original transformer for translation).

> **Saying it out loud.** It's a question of who can see whom. An encoder is bidirectional — every token sees every other token — which is great for understanding but useless for generating, since you'd be reading the answer. A decoder is causal: token $i$ only sees tokens up to $i$, which is exactly what you need to predict the next one. Encoder-decoder splits the job: encode the source bidirectionally, then generate the target causally while cross-attending to the encoded source, which is the classic translation setup.

**25. Why are modern LLMs decoder-only?**
Simpler architecture (one tower not two), one objective (next-token), naturally extends to in-context learning. Empirically scales better than encoder-decoder for general-purpose generation. The bidirectional encoder objective (masked LM) doesn't extend cleanly to long contexts and generation.

> **Saying it out loud.** Decoder-only won because it's one tower, one objective, and it scales. You train on next-token prediction over raw text, which is essentially unlimited data, and the same model that predicts text turns out to do translation, summarization, and Q&A once you just put the instruction in the context. Encoder-decoder needs paired data to really shine and has two sets of weights to tune. The deeper reason: masked language modeling gives you great representations but no way to sample token by token, and generation is what turned out to matter.

**26. What's the causal mask?**
A lower-triangular matrix $M$ with $0$ on/below diagonal and $-\infty$ above, added to the attention scores: $\text{scores} = Q K^\top / \sqrt{d_k} + M$. The $-\infty$ entries become $0$ after softmax, so position $i$ cannot attend to position $j > i$. Implements autoregressive constraint without changing the attention algorithm.

> **Saying it out loud.** The causal mask is how you stop a token from reading the future. You build a matrix that's zero on and below the diagonal and negative infinity above it, and just add it to the scores before the softmax. Negative infinity exponentiates to zero, so those positions get exactly zero attention weight — no change to the algorithm, just a constant added in. The beautiful part is you can then train on the whole sequence in one parallel pass and get $N$ next-token predictions for the price of one forward.

**27. What's cross-attention?**
The mechanism in encoder-decoder models. Decoder queries: $Q = \text{decoder-state} \cdot W_Q$. Encoder keys/values: $K, V = \text{encoder-output} \cdot W_K, W_V$. The decoder attends to the encoder output. Pure decoder LLMs don't have cross-attention; they handle inputs via in-context.

> **Saying it out loud.** Cross-attention is attention where the queries and the keys come from different places. In a translation model, the decoder's current state produces the queries, and the encoder's output of the source sentence produces the keys and values — so the decoder is asking 'which source words matter for what I'm writing right now?'. That's what gives you the classic word-alignment heatmaps. Pure decoder LLMs skip it entirely: they just put the source in the context window and let ordinary self-attention do the same job.

**28. What's masked language modeling (MLM)?**
BERT's pretraining objective. Mask 15% of tokens; train the model to predict them from bidirectional context. Bidirectional → encoder. Doesn't directly enable generation; the model learns rich representations but can't be sampled token-by-token.

> **Saying it out loud.** MLM is fill-in-the-blank. You corrupt about 15% of the tokens and train the model to recover them using context from both sides, which is why it produces such strong representations — every token gets to see the full sentence. That's BERT, and it's why BERT is still a great encoder for classification and retrieval. The limitation is fundamental: it only trains on 15% of positions, so it's sample-inefficient, and there's no natural way to sample text one token at a time from it.

**29. What's causal/autoregressive language modeling (CLM)?**
GPT's pretraining. Predict each token from preceding context only. Causal mask in attention. Naturally generates: sample token, append, repeat. The dominant pretraining objective in modern LLMs.

> **Saying it out loud.** CLM is just 'predict the next word', with a causal mask so you can't cheat. Every single position contributes a training signal, which makes it far more sample-efficient than masked LM's 15%. And generation falls straight out of the objective — sample a token, stick it on the end, run again. That alignment between what you train and what you deploy is a big part of why every frontier model uses it.

---

## E. Subtleties and gotchas

**30. What's weight tying?**
Sharing the embedding matrix between input embeddings and output unembedding. Saves parameters (one of the largest weight tensors), and empirically helps (the input/output spaces are dual). Standard in many LLMs but not all (GPT-2 ties; some recent open-source models don't).

> **Saying it out loud.** Weight tying means the input embedding matrix and the output unembedding matrix are literally the same tensor. The justification is that both are asking about the same relationship between tokens and vectors, just in opposite directions, so sharing them is a sensible inductive bias. Practically it's a big parameter saving — with a 128K vocab and $d$ of 4,096 that matrix is half a billion parameters. It's not universal though: GPT-2 ties, and several recent models deliberately untie because at large scale the extra capacity is worth more than the savings.

**31. What's the maximum context length determined by?**
At training: the longest sequence in the training data, plus positional encoding range. At inference: KV cache memory. The architecture itself doesn't impose a hard limit; it's compute, memory, and positional encoding that do. RoPE/ALiBi extend the practical range; brute KV memory limits the rest.

> **Saying it out loud.** Nothing in the architecture sets a max context — attention will happily run on a million tokens. What actually limits you is three things: the longest sequences you trained on, whether your positional scheme still makes sense past that, and KV cache memory at inference. RoPE and ALiBi handle the second one, and context-extension methods stretch it further. In practice the binding constraint is the third: KV cache grows linearly with length, so at some point one long request eats an entire GPU's memory.

**32. What's flash attention's contribution?**
I/O-aware tiled attention with online softmax. Same FLOPs as standard attention, but avoids materializing the $N \times N$ matrix in HBM. 2–4x wall-clock speedup on long sequences. See [`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`](../06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md) for details.

> **Saying it out loud.** FlashAttention's contribution is not fewer FLOPs — it's fewer trips to memory. The naive implementation writes the whole $N \times N$ score matrix out to HBM and reads it back, and at long sequences that memory traffic, not the arithmetic, is what you're waiting on. FlashAttention tiles the computation so the blocks fit in on-chip SRAM and uses an online softmax to accumulate the result without ever materializing the full matrix. Same math, 2 to 4x faster in wall clock, and it drops the memory from quadratic to linear — which is why long context became practical.

**33. Why do transformers use tied positional encoding for QK but not V?**
RoPE rotates $Q$ and $K$ (so the dot product depends on relative position) but does not rotate $V$ (which is the content being mixed). Mixing positional information into $V$ would unnecessarily entangle position with content; rotating only $Q$ and $K$ cleanly separates the two roles.

> **Saying it out loud.** Because $Q$ and $K$ decide *where* to look, and $V$ is *what* you get — you only want position in the first job. RoPE rotates the queries and keys so their dot product encodes relative distance, which is exactly the routing decision. If you also rotated $V$, you'd be smearing positional information into the content that gets copied forward, so a token's meaning would depend on where it happened to sit. Keeping them separate means the model can say 'look three tokens back' and 'copy that word' as two independent facts.

**34. What's the attention sink phenomenon?**
Empirical observation: in long-context transformers, the first few tokens get disproportionate attention from many heads. They act as "sinks" that absorb attention mass. Removing them (when streaming) breaks the model. Mitigation: keep them in the KV cache (StreamingLLM, sink tokens).

> **Saying it out loud.** Attention sinks are the model's junk drawer. In a trained transformer, the first token or two — often just the BOS — soaks up a huge share of attention from many heads, and it's not because that token is meaningful. It's because softmax forces the weights to sum to one, so a head that has nothing it wants to look at has to put its mass somewhere, and it dumps it on a token everyone can see. The practical consequence is nasty: StreamingLLM showed that if you evict those first tokens from the KV cache when sliding a window, the model falls apart — so you pin them.

**35. What's a "logit lens"?**
Interpretability tool: project intermediate-layer activations through the unembedding matrix to read out token probabilities at intermediate layers. Reveals how predictions sharpen across depth — early layers give vague predictions, late layers sharpen.

> **Saying it out loud.** Logit lens is the trick of asking 'what would the model say if we stopped here?'. You take the residual stream at layer 15, skip the remaining layers, and shove it straight through the final unembedding to get a token distribution. What you see is predictions forming gradually — early layers give something vague and frequency-driven, middle layers get the right topic, late layers commit. It's crude, since intermediate representations aren't strictly in the same basis as the final one, but it's a surprisingly informative first look at where a computation happens.

**36. What does an attention pattern look like for a "copy head"?**
A nearly-diagonal attention pattern shifted by a fixed offset. Position $t$ attends primarily to position $t - k$ for some $k$. Useful for copying tokens from a previous position (e.g., variable names in code, repeated structure in lists).

> **Saying it out loud.** A copy head has an attention pattern that looks like a diagonal stripe shifted off the main diagonal. Every position attends to whatever is $k$ tokens back, more or less regardless of content. That's the mechanism behind reproducing repeated structure — closing a bracket, repeating a variable name, continuing a list format. It's also the building block for induction heads, which pair a previous-token head with a match-and-copy head to do in-context learning.

---

## F. Computational and scaling

**37. Where do parameters live in a transformer?**
Roughly: 2/3 to 3/4 in FFN matrices (4d expansion × 2 matrices = $8d^2$ per layer); attention has $Q, K, V, O$ projections ($4 d^2$ per layer). Embeddings ($\text{vocab} \times d$) and unembeddings can be substantial. Layer norms: trivial. So FFN dominates.

> **Saying it out loud.** Roughly two-thirds to three-quarters of the weights are in the FFNs. Per layer, attention has four $d \times d$ projections — that's $4d^2$ — while a vanilla FFN has two matrices at 4x expansion, which is $8d^2$, so it's a 2:1 split before you even count SwiGLU's third matrix. Embeddings and unembeddings matter at small scale, where a 128K vocab times $d$ can rival several layers, but they get diluted as depth grows. LayerNorms are a rounding error. The punchline that matters for inference: when you're memory-bound during decode, you're mostly streaming FFN weights.

**38. Compute breakdown for a forward pass?**
For a transformer with $L$ layers, sequence $N$, dim $d$:

- Attention: $O(L \cdot N^2 \cdot d)$
- FFN: $O(L \cdot N \cdot d^2)$ (per-token, scales with hidden $4d$)
- Embeddings: $O(N \cdot d)$

For $N \gg d$, attention dominates; for $N \ll d$, FFN dominates. With vanilla 2-matmul FFN at $4d$ expansion, attention FLOPs $\sim 4N^2 d$ and FFN FLOPs $\sim 8 N d^2$ → crossover at $N \approx 2d$. With SwiGLU's 3 matmuls scaled to match params, the constants shift but the same-order conclusion holds: crossover scales as $O(d)$, on the order of $2d$–$4d$.

> **Saying it out loud.** Attention scales like $N^2 d$ and the FFN scales like $N d^2$, so which one dominates depends entirely on whether your sequence is longer or shorter than your model width. With a vanilla 4x FFN, the crossover lands around $N \approx 2d$ — so for a model with $d = 4096$, short prompts are FFN-bound and anything past about 8K tokens is attention-bound. This is why people's intuition about 'attention is the expensive part' is only true at long context; at 512 tokens the FFN is eating most of your FLOPs. SwiGLU shifts the constants but not the conclusion.

**39. Memory breakdown for a forward pass?**
Activations: $O(L \cdot N \cdot d)$ for stream + intermediate FFN states + attention scores $O(N^2)$ per layer.
Weights: roughly $12 d^2 L + d V$ ($V$ = vocab size) for a vanilla transformer.
KV cache (during decode): $2 L \cdot n_{\text{kv-heads}} \cdot d_{\text{head}} \cdot N$ per sequence.

> **Saying it out loud.** Three buckets, and they scale differently. Weights are about $12 d^2$ per layer and don't care about sequence length. Activations are $O(L N d)$ for the stream, plus the attention scores which are $O(N^2)$ per layer unless you use FlashAttention, which is why the naive implementation OOMs on long sequences. And at inference the KV cache is $2 L \cdot n_{kv} \cdot d_h \cdot N$ per sequence, which grows with every token you generate — that's the one that decides how many users you can serve at once.

**40. What are scaling laws?**
Kaplan et al. 2020, Hoffmann et al. 2022 (Chinchilla): loss scales as a power law in compute, parameters, and tokens. Chinchilla optimal: ~20 tokens per parameter (Llama-2 increased this further). Scaling laws guide compute allocation between model size and training data.

> **Saying it out loud.** Scaling laws say loss falls off as a smooth power law in compute, parameters, and data — no cliffs, no plateaus, over many orders of magnitude. The practical payoff is that you can run small experiments and predict what a run 1000x bigger will do, which is how anyone justifies a nine-figure training budget. Chinchilla's version added the allocation rule: for a fixed compute budget, you want roughly 20 tokens per parameter. Modern practice deliberately overshoots that — LLaMA trains way past compute-optimal — because inference cost matters and a smaller, longer-trained model is cheaper to serve forever.

**41. Chinchilla vs Kaplan?**
Kaplan suggested overparameterized + undertrained models. Chinchilla showed those models are suboptimal: at fixed compute, smaller models trained on more tokens beat larger models trained on fewer. The "Chinchilla-optimal" frontier rebalanced the ML community's training recipes.

> **Saying it out loud.** Kaplan said make the model big; Chinchilla said make it well-fed. Kaplan's 2020 scaling laws implied that at a fixed budget you should spend most of it on parameters and relatively little on tokens, which is why GPT-3 was 175B parameters trained on only 300B tokens. Chinchilla redid the experiments with proper learning-rate schedules and found the opposite balance — a 70B model on 1.4T tokens beat the 280B model on the same compute. The rule of thumb became 20 tokens per parameter, and the whole field's recipes changed within a year.

---

## G. Design choices and ablations

**42. Why use multi-head and not single-head?**
Empirically, parallel attention to multiple patterns helps. Theoretically, multiple heads form a richer function class than a single head with $h \times d_h = d$ dimensions, because each head has its own softmax (heads can't share normalization).

> **Saying it out loud.** Because one softmax can only pay attention to one thing. If you had a single head over the full $d$ dimensions, the model would have exactly one distribution over positions and would have to average incompatible needs together — 'look at the subject' and 'look at the previous token' would blur into a compromise that serves neither. Split into heads and each gets its own normalization, so they can commit to different patterns independently. Same parameters, strictly richer function class, and the cost is that each head works in a narrower $d/h$-dim subspace, which is why very large $h$ eventually hurts.

**43. Why not normalize Q and K explicitly (instead of scaling by $\sqrt{d_k}$)?**
Some recent work does (QK-norm). Empirically helps stability for very large models. The original paper's $\sqrt{d_k}$ is the simpler version that works for moderate scales.

> **Saying it out loud.** You can, and people increasingly do — that's QK-norm, where you L2-normalize the queries and keys before the dot product. It helps stability at very large scale, because $\sqrt{d_k}$ only fixes the variance under an assumption of unit-scale inputs, and during training the QK projections can drift and blow the logits up anyway. The reason it wasn't in the original paper is that the simple scalar divide is cheaper and works fine at moderate size. If you're training something huge and seeing attention-logit explosions, QK-norm is the standard fix.

**44. Why don't we make $d_k$ and $d_v$ different?**
We could; the original paper does have them as separate hyperparameters. In practice, almost all implementations use $d_k = d_v = d / h$. Empirically, no clear win from making them different.

> **Saying it out loud.** You can have them differ — the original paper actually treats $d_k$ and $d_v$ as separate hyperparameters — but essentially nobody does. The reasoning is that $d_k$ controls how expressive the matching is and $d_v$ controls how much content each head can carry, and there's no empirical evidence that unbalancing them buys anything. Setting both to $d/h$ also makes the concatenation land exactly at $d$, which keeps the implementation clean. It's one of those places where the default is default because tuning it is a waste of a compute budget.

**45. What happens if you remove the FFN?**
Catastrophic. Pure-attention transformers are essentially linear (attention is linear in V). Cannot represent non-linear functions, cannot store factual knowledge well. Some "linear transformer" variants reintroduce non-linearity elsewhere; they are not standard.

> **Saying it out loud.** Remove the FFN and the model collapses to something close to linear. Attention is a weighted average of values, and a weighted average is a linear operation on $V$ — the only non-linearity is in how the weights are computed, which isn't enough. So you lose the ability to represent non-linear functions, and more concretely you lose the place where factual knowledge is stored, since the FFN is the key-value memory. Empirically, an attention-only transformer can still do copying and induction, which is a genuinely interesting result, but it can't do the knowledge-heavy stuff at all.

**46. What happens if you remove the residual connections?**
Training fails for deep networks. Gradients vanish through depth. Even a few layers becomes hard to train. Residuals are not optional.

> **Saying it out loud.** Training just fails. The gradient reaching early layers becomes a product of many Jacobians, and unless each of those has a spectral norm right at one, the product either vanishes or explodes — vanishing is the usual outcome. Even at six or eight layers you'll see it; at fifty it's hopeless. There's no clever initialization that fully saves you, which is why residuals show up in essentially every deep architecture since 2015, not just transformers.

**47. What happens if you remove LayerNorm?**
Activations grow uncontrolled across depth (the residual stream accumulates layer outputs). Training becomes very unstable, often diverges. In Pre-LN, removing LayerNorm makes the model unable to limit the read magnitude from the stream.

> **Saying it out loud.** The residual stream grows without bound and training goes unstable. Every block adds its output onto the stream, so magnitudes accumulate with depth, and if nothing renormalizes on the way in, later layers are reading inputs many times larger than the ones early layers saw. The softmax and the activations then sit in saturated regions and you get loss spikes or outright divergence. In pre-LN specifically, the LayerNorm is what caps how big the input to each sublayer can be, so removing it means the model loses control of its own read magnitude.

---

## H. Comparisons

**48. Transformer vs RNN?**
Transformer wins on parallelism (no sequential bottleneck), gradient flow (one attention layer reaches anything), and scaling. RNN wins on inference memory (constant per step vs growing KV cache) and on data efficiency at small scales.

> **Saying it out loud.** Transformers won on parallelism, not on elegance. An RNN has to process token 500 after token 499, so training time scales with sequence length no matter how many GPUs you own; a transformer sees the whole sequence at once. Attention also gives any token a one-hop path to any other token, whereas an RNN has to carry information through hundreds of sequential updates and it gets forgotten. What the RNN wins is inference: constant memory per step versus a KV cache that grows linearly, which is exactly the pressure that made Mamba and linear attention interesting again.

**49. Transformer vs CNN?**
CNNs have local connectivity and translation equivariance (good biases for vision). Transformers have weaker biases but better scaling. ViT showed transformers can match/beat CNNs on vision at sufficient scale.

> **Saying it out loud.** CNNs bake in the assumptions that nearby pixels relate and that a cat is a cat wherever it appears; transformers assume almost nothing and learn the structure from data. That means CNNs win at small data scale — the built-in bias is doing free work — while transformers overtake them once there's enough data to learn better structure than we could have designed. ViT is the clean demonstration: worse than ResNets on ImageNet alone, better once pretrained on 300M images. The general lesson is that weak biases plus scale beat strong biases plus scarcity.

**50. Transformer vs SSM (Mamba)?**
SSMs reintroduce sequential processing in a parallelizable form (convolutional view). $O(N)$ sequence complexity vs transformer's $O(N^2)$. Empirically competitive at smaller scale; whether they match transformers at frontier scale is open. Hybrid models (combining attention layers and SSM layers) are an active area.

> **Saying it out loud.** SSMs are the serious attempt to get rid of the quadratic. They process the sequence with a recurrence, but one structured so it can be computed in parallel during training, giving $O(N)$ instead of $O(N^2)$ and a constant-size state at inference instead of a growing KV cache. The catch is the constant-size state: a transformer can go back and read any token exactly, while an SSM has compressed everything into a fixed vector, so exact recall and in-context copying are where they lag. That's why the practical answer today is hybrids — a few attention layers for recall, SSM layers for everything else.

---

## I. Quick-fire

**51.** *Original paper?* Vaswani et al. 2017.
**52.** *Standard activation in original FFN?* ReLU.
**53.** *Modern activation in LLM FFN?* SwiGLU.
**54.** *Standard FFN expansion ratio?* 4× (or 8/3 with SwiGLU).
**55.** *Pre-LN or post-LN in modern LLMs?* Pre-LN (or RMSNorm variant).
**56.** *Most common positional encoding in modern LLMs?* RoPE.
**57.** *Encoder-only flagship?* BERT.
**58.** *Decoder-only flagship?* GPT family, LLaMA, etc.
**59.** *Encoder-decoder flagship?* T5.
**60.** *Attention complexity?* $O(N^2 d)$ compute, $O(N^2)$ memory naively.

---

## Self-grading

If you can't answer 1–10, you don't know transformers. If you can't answer 11–25, you can't pass an LLM-focused MLE round. If you can't answer 26–50, frontier-lab applied scientist screens will expose gaps.

Aim for 40+/60 cold before any architecture interview.
