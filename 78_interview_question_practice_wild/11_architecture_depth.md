# Architecture depth

**Foundation explainer:** Further depth · **Priority:** Important — second pass

Important: second-pass depth after the core internals are comfortable.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f101"></a>

## F101. Implement the forward structure of a Vision Transformer.

**Start here.** A ViT turns image patches into a sequence processed by Transformer blocks.

**Work it through.**

1. Split an image into non-overlapping patches, requiring compatible image dimensions or an explicit padding policy.
2. Flatten each patch, project it into the model dimension and add position information and any classification token.
3. Run normalization, attention and FFN blocks, then use the specified pooling/head. Trace every reshape to preserve patch order.

**Small example.** A 32-by-32 RGB image with 8-by-8 patches gives 16 patch tokens of 192 input features each, before projection.

**Common trap.** Mixing channels and patch positions during reshape or silently discarding image borders.

**Follow-up.** How does smaller patch size affect attention cost? Token count rises, and dense attention grows quadratically in that count.

---

<a id="f102"></a>

## F102. Derive a manual attention backward pass.

**Start here.** Backpropagate through value mixing, rowwise softmax and the scaled score product.

**Work it through.**

1. For O=P@V and upstream G, compute dV=Pᵀ@G and dP=G@Vᵀ.
2. For each softmax row, dS=P*(dP-sum(dP*P)); zero forbidden-score gradients.
3. For S=QKᵀ/sqrt(d), use dQ=dS@K/sqrt(d) and dK=dSᵀ@Q/sqrt(d), then chain through projections.

**Small example.** Every row of dS sums approximately to zero because softmax is invariant to adding a row constant.

**Common trap.** Materializing an enormous softmax Jacobian or forgetting the scale in both Q and K gradients.

**Follow-up.** How would you validate it? Compare each input and projection gradient with finite differences on a tiny masked example.

---

<a id="f103"></a>

## F103. Implement rotary position embeddings on coordinate pairs.

**Start here.** RoPE rotates each two-dimensional pair using a position-dependent angle while preserving pair length.

**Work it through.**

1. Choose the pair layout and frequencies explicitly; adjacent-pair and split-half layouts require matching weight conventions.
2. For each pair (a,b), compute (a*cos(theta)-b*sin(theta),a*sin(theta)+b*cos(theta)).
3. Apply to Q and K using the correct absolute positions, including cache offsets; leave V unchanged for ordinary RoPE.

**Small example.** Rotating (3,4) preserves squared length 25 for any angle, up to rounding.

**Common trap.** Mixing incompatible pair layouts when loading a model or rotating old cached keys a second time.

**Follow-up.** What test checks relative structure? Compare dot(R_m*q,R_n*k) with dot(q,R_(n-m)*k).

---

<a id="f104"></a>

## F104. When can input and output embeddings share weights?

**Start here.** Weight tying reuses the token representation matrix for vocabulary scoring when dimensions are compatible.

**Work it through.**

1. Let E:(V,d) map token ids to hidden vectors.
2. For hidden h:(d,), output logits can be h@Eᵀ, optionally with a separate bias.
3. Count E once, ensure gradients accumulate from both uses and compare quality against an untied head under matched settings.

**Small example.** With V=10000 and d=256, tying can avoid an additional 2560000-weight output matrix.

**Common trap.** Counting the shared matrix twice or assuming tying is mandatory in every model.

**Follow-up.** What if hidden and embedding dimensions differ? Add a compatible projection or leave the matrices untied.

---

<a id="f105"></a>

## F105. Why is the common 6ND training-FLOP estimate approximate?

**Start here.** It counts dominant parameter multiplications for a forward-and-backward pass, not every operation in a real training run.

**Work it through.**

1. Define N as the relevant dense parameter count and D as training tokens.
2. A rough forward cost is 2ND and backward adds about twice that, giving 6ND.
3. Account separately for sequence-dependent attention, recomputation, embeddings, sparsity, communication and optimizer overhead when important.

**Small example.** Doubling both N and D gives roughly four times the parameter-matmul work under the approximation.

**Common trap.** Using total MoE parameters as active dense compute per token or calling FLOPs the same as wall-clock cost.

**Follow-up.** How does activation checkpointing change the estimate? It saves activation memory by recomputing some forward work during backward.

---

<a id="f106"></a>

## F106. Evaluate a longer-context extension.

**Start here.** Accepting a longer input is not evidence that the model uses distant evidence correctly.

**Work it through.**

1. Measure quality across sequence lengths, relevant-evidence positions, distractor counts and multi-hop tasks.
2. Compare with the original context configuration and retrieval baselines under matched output budgets.
3. Track latency, cache memory and failure slices; test real workloads alongside synthetic needle tasks.

**Small example.** A model may retrieve one exact phrase at 64000 tokens yet fail to combine two distant facts reliably.

**Common trap.** Reporting only maximum accepted length or extrapolating from a single synthetic benchmark.

**Follow-up.** Why vary evidence position? Models can show strong beginning/end biases and weaker use of middle context.

---

<a id="f107"></a>

## F107. Optimize a tokenizer without changing its semantics.

**Start here.** An efficient encoder must preserve the specified tokenization, including tie rules and boundaries.

**Work it through.**

1. First implement and test a simple reference over small strings and special tokens.
2. For fixed-vocabulary longest matching, consider a trie; for ranked BPE merges, use a representation that updates neighboring merge candidates efficiently.
3. Benchmark normalization, Unicode handling and merge operations separately, comparing exact token ids against the reference.

**Small example.** Changing a merge tie-break can alter token ids even if the decoded text is identical.

**Common trap.** Replacing a BPE encoder with greedy longest-vocabulary matching and assuming the results always agree.

**Follow-up.** What must decoding guarantee? The chosen normalization and byte/token conventions determine whether exact text round trips are possible.

---

<a id="f108"></a>

## F108. Explain convolution, pooling and residual CNN blocks.

**Start here.** Convolution reuses local filters across space; residual blocks make it easier to learn changes to a carried representation.

**Work it through.**

1. Specify channels, kernel, stride and padding, then calculate output height and width.
2. Apply learned local filters, nonlinearities and optional normalization; use pooling or strided operations to reduce resolution.
3. Add a shortcut only when shapes match, or use a projection. Explain the resulting receptive field and information loss.

**Small example.** A 3-by-3 convolution with stride 1 and padding 1 preserves spatial size, while changing channel count through the filter bank.

**Common trap.** Adding residual tensors with incompatible channel or spatial dimensions.

**Follow-up.** Why use global average pooling? It aggregates spatial responses with fewer head parameters, but discards explicit location detail.

---

<a id="f109"></a>

## F109. Compare ReLU, GELU and gated activations.

**Start here.** Activation choices alter gradient flow and representation, but must be compared under a fair parameter and compute budget.

**Work it through.**

1. ReLU returns max(0,x), with zero gradient on negative inputs under the usual convention.
2. GELU smoothly weights inputs by a Gaussian-CDF-based gate; SiLU uses x*sigmoid(x).
3. Gated FFNs use one learned branch to gate another, often with an extra projection. Count those extra weights when comparing models.

**Small example.** A three-matrix gated FFN with hidden width m has about 3dm weights, versus 2dm for an ordinary two-matrix FFN.

**Common trap.** Calling GELU just a hard threshold or claiming one activation always wins regardless of architecture and training.

**Follow-up.** What is a dead ReLU? A unit stuck in a region where its gradient is zero for the observed training inputs.

---

<a id="f110"></a>

## F110. Combine padding, causal and segment attention masks.

**Start here.** Each mask expresses a different restriction; a token may attend only where all applicable restrictions allow it.

**Work it through.**

1. Create a key-validity mask, causal position relation and, for isolated packing, same-segment relation.
2. Combine them with logical AND under a clearly stated True-means-allowed convention.
3. Apply before softmax and define padded-query outputs separately so all-masked rows do not create NaNs.

**Small example.** Two examples sharing a packed sequence may be causal in global index order yet still require a same-segment mask to remain independent.

**Common trap.** Assuming every library uses the same boolean-mask polarity or using query padding as if it were key padding.

**Follow-up.** How do positions differ during cached decode? Query indices need their absolute cache offset when constructing causality.

---
