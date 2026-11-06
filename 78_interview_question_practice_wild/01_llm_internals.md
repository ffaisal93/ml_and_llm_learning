# LLM internals

**Foundation explainer:** Core material · **Priority:** Very important

Very important: explain the computation before naming an architecture.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f001"></a>

## F001. Why are Q, K and V separate projections?

**Start here.** Attention is a learned lookup: a query says what to find, a key says what can be matched, and a value carries what to copy.

**Work it through.**

1. Write Q=XWq, K=XWk and V=XWv, with X shaped (B,T,d_model).
2. Compute scores QKᵀ, normalize each query's row over keys, then mix the value vectors.
3. Separate Q and K allow asymmetric relevance. Separate V lets the features used for matching differ from the features transmitted; the output projection mixes heads.

**Small example.** In “The dog chased the ball because it moved,” a pronoun's query may match an object's key while its value carries object attributes. The three vectors serve different jobs even though they start from the same token state.

**Common trap.** Calling Q, K and V three different input sequences: self-attention projects the same input; cross-attention uses a different source for K and V.

**Follow-up.** What restriction appears in the unmasked score matrix if Wq and Wk are tied? The raw pairwise scores become symmetric for the same input.

---

<a id="f002"></a>

## F002. Why divide attention logits by the square root of the head dimension?

**Start here.** It controls the initial scale of the dot products so softmax is less likely to become nearly one-hot before learning.

**Work it through.**

1. Assume independent, zero-mean, unit-variance query and key coordinates. Each product has variance one.
2. A sum of d_k such products has variance d_k and standard deviation sqrt(d_k).
3. Divide by sqrt(d_k), then subtract the row maximum before exponentiation. The divisor controls scale; max subtraction prevents overflow. These are different mechanisms.

**Small example.** With d_k=64, the unscaled score has standard deviation 8 under these assumptions. Scaling reduces it to 1; dividing by 64 would make the scores unnecessarily small.

**Common trap.** Claiming the unit-variance argument is an exact law after training, when Q and K can be correlated and differently scaled.

**Follow-up.** Could learned temperature or QK normalization change this calculation? Yes; state how the effective score scale is controlled.

---

<a id="f003"></a>

## F003. Trace every shape in multi-head self-attention.

**Start here.** Heads are parallel attention computations over smaller feature spaces; they do not each attend to a disjoint piece of the sequence.

**Work it through.**

1. Project X:(B,T,d) into Q,K,V:(B,T,H*d_h). Reshape and transpose to (B,H,T,d_h).
2. Multiply Q by K transposed on its last two axes to obtain (B,H,T,T). Softmax over the final key axis.
3. Multiply by V to obtain (B,H,T,d_h), transpose to (B,T,H,d_h), concatenate to (B,T,H*d_h), and apply Wo to return to (B,T,d).

**Small example.** For B=2, T=3, H=4, d_h=8, each score tensor has shape (2,4,3,3); concatenating produces (2,3,32).

**Common trap.** Reshaping without the transpose, which silently mixes token and head dimensions while preserving the total element count.

**Follow-up.** What changes for cross-attention? The score matrix becomes (B,H,T_query,T_key), which can be rectangular.

---

<a id="f004"></a>

## F004. Where does the causal mask go, and why?

**Start here.** A decoder must predict using only tokens already available at that position. The mask prevents future-token information from entering the weighted sum.

**Work it through.**

1. For query position i, allow key positions j<=i, including the current input token.
2. Set forbidden scores to negative infinity before softmax; their exponentials become zero and the allowed probabilities are renormalized.
3. Combine causality with padding and any segment restrictions. Ensure each valid query has at least one allowed key or explicitly define zero output for an empty row.

**Small example.** For three tokens the allowed rows are [1,0,0], [1,1,0], [1,1,1]. Changing token 3 must not change the output for token 1.

**Common trap.** Zeroing scores before softmax: exp(0)=1, so a forbidden position still receives probability mass.

**Follow-up.** Why can loss masking alone not isolate packed examples? Loss masking changes which targets train the model, not which tokens it can attend to.

---

<a id="f005"></a>

## F005. Why do Transformers need positional information, and what does RoPE do?

**Start here.** Without position information, ordinary self-attention cannot reliably distinguish token order. RoPE puts position-dependent rotations into queries and keys.

**Work it through.**

1. Split each head vector into coordinate pairs and rotate each pair by its position times a frequency.
2. Use rotated Q and K in the attention dot product. The relative rotation depends on the difference between their positions.
3. At cached decoding time, continue the absolute position offset from the prefix. Long-context extension still needs validation; relative structure alone does not guarantee extrapolation.

**Small example.** For a coordinate pair (1,0), a 90-degree rotation gives (0,1), preserving length. Applying the same rotation to both q and k leaves their mutual dot product unchanged.

**Common trap.** Adding RoPE to token embeddings, or restarting decode positions at zero every step.

**Follow-up.** What is cached with RoPE? Typically already-rotated keys and ordinary values; the exact representation depends on the implementation.

---

<a id="f006"></a>

## F006. Compare pre-norm, post-norm, LayerNorm and RMSNorm.

**Start here.** Normalization controls activation scale; its placement also changes how gradients travel through a deep residual stack.

**Work it through.**

1. Post-norm computes LN(x+F(x)); pre-norm computes x+F(LN(x)). Pre-norm leaves an explicit identity route through the residual addition.
2. LayerNorm subtracts each token's feature mean and divides by sqrt(feature variance+epsilon), then applies learned scale and usually bias.
3. RMSNorm divides by sqrt(mean(x²)+epsilon), usually with scale but no centering. Compare stability and quality empirically; neither architecture removes the need to tune training.

**Small example.** For x=[2,2], LayerNorm's normalized part is [0,0]; RMS normalization gives approximately [1,1]. A learned affine term can change either output.

**Common trap.** Normalizing across the sequence or batch when implementing per-token LayerNorm; using sample variance with denominator D-1.

**Follow-up.** Does pre-norm eliminate warmup? No. Stability still depends on initialization, depth, precision and optimization settings.

---

<a id="f007"></a>

## F007. Why are residual connections and the feed-forward network both needed?

**Start here.** Attention moves information between positions; the feed-forward network transforms the features at each position. Residual additions preserve an easy path for useful existing information.

**Work it through.**

1. An attention sublayer creates context-dependent mixtures of token values.
2. A position-wise nonlinear MLP expands the feature dimension, applies an activation or gate, and projects back. Its weights are shared across positions, not across all layers by default.
3. Add each sublayer's result to its input. Without nonlinearities, stacked affine feed-forward layers collapse into one affine map; attention's own input-dependent mixing remains nonlinear.

**Small example.** With d=8 and hidden width 32, an ordinary two-matrix FFN has 8*32+32*8=512 weights before biases. A three-matrix gated FFN has a different count.

**Common trap.** Saying the FFN is identical to attention, or assuming all models use exactly a 4d intermediate width.

**Follow-up.** Why does a residual branch need scale management at large depth? Repeated additions can grow activation variance.

---

<a id="f008"></a>

## F008. What makes attention quadratic, and what does FlashAttention change?

**Start here.** Dense attention compares every query with every key. FlashAttention computes the same attention operation with a different memory-access schedule.

**Work it through.**

1. At sequence length T, the logical score matrix has T² entries per head; forming QKᵀ and multiplying by V cost O(T²*d_h).
2. A naive implementation materializes scores and probabilities in high-bandwidth memory, creating heavy memory traffic.
3. Tiled exact attention maintains rowwise normalization statistics and avoids storing the full matrix in that memory. Arithmetic remains quadratic for full attention; practical memory use and throughput improve.

**Small example.** Doubling T from 1024 to 2048 quadruples the number of pairwise scores. A tiled kernel can avoid storing those scores all at once, but still processes the interactions.

**Common trap.** Calling FlashAttention sparse attention or claiming it changes full attention's asymptotic arithmetic to linear.

**Follow-up.** Why can two exact implementations differ slightly? Floating-point reduction order and precision differ.

---

<a id="f009"></a>

## F009. How do you estimate model parameters and KV-cache memory?

**Start here.** Count stored tensors separately. Model weights are mostly fixed with sequence length; the KV cache grows with active tokens and requests.

**Work it through.**

1. For ordinary equal-width multi-head attention, Q,K,V,O contribute approximately 4d² weights. A two-matrix FFN with hidden width m contributes 2dm.
2. Sum the actual layer tensors, embeddings and output head, accounting for tied weights and architecture-specific projections.
3. Cache bytes are 2*B*L*T*H_kv*d_h*bytes_per_element for equal-length sequences. The factor two is keys plus values; add allocator and metadata overhead separately.

**Small example.** For B=1, L=32, T=4096, H_kv=8, d_h=128 and two-byte elements, the cache is 536870912 bytes, or 512 MiB.

**Common trap.** Using query-head count in the cache formula for GQA, or silently treating GB and GiB as the same unit.

**Follow-up.** How does ragged batching change the estimate? Replace B*T with the sum of live sequence lengths, then account for block rounding.

---

<a id="f010"></a>

## F010. Why is a next-token model trained with shifted targets?

**Start here.** At position t, the model sees tokens through t and predicts token t+1. The target shift makes the training task match autoregressive generation.

**Work it through.**

1. Given tokens [x0,x1,x2,x3], use inputs [x0,x1,x2] and targets [x1,x2,x3], unless the library performs this shift internally.
2. Apply a causal mask to prevent access to later input positions. Use cross-entropy over vocabulary logits without an extra softmax first.
3. For instruction tuning, mask unwanted target positions after aligning the shift. Divide by the count of supervised targets, not padded sequence length.

**Small example.** If a prompt ends at input position 3 and the first answer is token 4, the prediction at position 3 is the first supervised answer target.

**Common trap.** Shifting twice because both a data collator and the model's loss already shift labels, or supervising the wrong prompt boundary.

**Follow-up.** How do you test leakage? Change future tokens and confirm that earlier logits are unchanged.

---
