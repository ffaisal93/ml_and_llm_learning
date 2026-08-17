# ML Depth Round — Interview Question Bank

The ML depth round is the one where an interviewer picks a topic you claimed on your résumé and drills straight down until they hit bedrock — usually three or four "why" levels past the answer you rehearsed. The questions below are written the way interviewers actually phrase them, and each answer is written the way a strong candidate actually talks: prose, not bullets, with the math spelled out where the math *is* the answer. Use it by covering the answer, saying your version out loud, then uncovering and diffing — the gap between "I recognize this" and "I can say this" is exactly what the round measures. Follow-ups are included because the first answer is never where the round ends. Where a claim is genuinely contested or has shifted recently, the answer says so instead of asserting it flatly.

---

## 1. Attention and Transformers

### Q: Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention?

**Answer.** It's a variance-control argument, and the specific number $\sqrt{d_k}$ falls out of it exactly. Take a query vector $q$ and a key vector $k$, both in $\mathbb{R}^{d_k}$. Model their components at initialization as independent, zero-mean, unit-variance random variables. The unnormalized score is the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. Each term $q_i k_i$ has mean $\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ by independence, and variance $\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. Summing $d_k$ independent terms adds variances, so

$$\operatorname{Var}(q \cdot k) = d_k, \qquad \text{std} = \sqrt{d_k}.$$

So the logits going into the softmax have a standard deviation that grows like $\sqrt{d_k}$. For $d_k = 128$ that's roughly $\pm 11$ on typical logits, and softmax over logits with that spread is essentially an argmax — one entry gets probability ~1 and the rest get ~0.

Now the part people leave out: why that's *bad*. It's not aesthetics, it's gradients. For softmax output $p$, the Jacobian is $\operatorname{diag}(p) - pp^\top$. When $p$ is nearly one-hot, every entry of that Jacobian is nearly zero — $p_i(1-p_i) \to 0$. So the gradient flowing back through the attention weights to $Q$ and $K$ vanishes, and the model can't learn *where* to attend; it's frozen into whatever routing the random initialization gave it. Dividing by $\sqrt{d_k}$ renormalizes the logits back to unit variance, which keeps the softmax in a regime where the distribution is soft and the Jacobian has real magnitude.

The reason it's $\sqrt{d_k}$ and not $d_k$ is that we're normalizing a standard deviation, not a variance. And note the scaling uses $d_k$, the *per-head* key dimension, not the model dimension $d_{\text{model}}$ — with $h$ heads and $d_k = d_{\text{model}}/h$, that's the dimension the sum is actually taken over.

**Follow-up: "That argument assumes unit-variance inputs. Does it still hold after training?"** → Strictly, no — it's an initialization-time argument. Once trained, $Q$ and $K$ are correlated and their entry variances aren't 1, so the true logit scale is empirical, not $\sqrt{d_k}$. The scaling still matters because it gets you through early training without the softmax saturating, and because it makes the initialization behavior independent of $d_k$ so you can change head dimension without retuning the learning rate. In practice people do observe logit growth during training — that's exactly why QK-norm (applying RMSNorm to $Q$ and $K$ before the dot product) has become common in large models: it re-imposes the bounded-logit property at every step instead of only at step zero.

**Follow-up: "So could you just fold the $1/\sqrt{d_k}$ into the weight initialization of $W_Q$?"** → Yes, mathematically — scaling $W_Q$ by $d_k^{-1/4}$ and $W_K$ by $d_k^{-1/4}$ gives the same forward pass at init. But it's worse in practice: the weights are trainable, so the optimizer will drift them away from that scale, and weight decay pulls on them, whereas an explicit constant divisor is a fixed property of the architecture. Keeping it explicit also decouples the scaling from initialization scheme choices.

*Trap:* Saying "for numerical stability" and stopping. Softmax is already numerically stabilized by subtracting the max; overflow isn't the issue. The issue is softmax saturation killing gradients.

### Q: What do Q, K, and V actually represent? Why three projections instead of one?

**Answer.** Think of attention as a soft, differentiable dictionary lookup. Every token emits a *key*, which advertises "here is what I am, as something you could look for," and a *value*, which is "here is what you get if you look at me." Every token also emits a *query*: "here is what I'm looking for." The compatibility between a query and a key is a dot product, softmaxed into a distribution over positions, and the output is the value vector averaged under that distribution.

The reason you need query and key to be *separate* projections is that relevance is asymmetric. If you used the same projection for both — attention scores $x_i^\top W^\top W x_j$ — the score matrix would be symmetric in $i$ and $j$ up to the mask. But "the pronoun *it* should attend to the noun *dog*" is not the same relation as "the noun *dog* should attend to the pronoun *it*." Separate $W_Q$ and $W_K$ give you a general bilinear form $x_i^\top W_Q^\top W_K x_j$, which can represent any asymmetric relation of rank up to $d_k$.

The reason value is a *third* projection is that what makes a token findable is not the same as what's useful about it once found. The key for the token "Paris" might encode "I am a capital-city proper noun in subject position" — the features that make a query match — while the value encodes the semantic content you actually want to move into the residual stream. Tying them would force one representation to do both jobs, and would mean any change to how a token is retrieved changes what gets retrieved.

There's a useful factorization here: attention has two low-rank matrices, $W_{QK} = W_Q^\top W_K$, which determines the *attention pattern* (where information moves), and $W_{OV} = W_O W_V$, which determines *what* is written when it moves. These are functionally independent circuits — the QK circuit reads the residual stream to decide routing, the OV circuit reads it to decide content. That's the cleanest mental model, and it's the one that lets you reason about what a head is doing when you go look at one.

**Follow-up: "Why is $W_O$ there at all? Isn't $V$ already the output?"** → Two reasons. Dimensionally, with $h$ heads each producing $d_v = d_{\text{model}}/h$, you concatenate to $d_{\text{model}}$ and $W_O$ mixes across heads — without it each head would write to a fixed disjoint slice of the residual stream and couldn't compose. Functionally, $W_O$ lets each head choose which subspace of the residual stream to *write into*, independently of which subspace it read from. That read-subspace/write-subspace independence is what lets heads in different layers chain together.

*Trap:* Saying V is "the actual content" and K is "a compressed version of it." They're different linear views of the same vector, and neither is a compression of the other — $W_V$ is often the same rank as $W_K$.

### Q: Why multiple attention heads instead of one big one with the same total dimension?

**Answer.** The parameter count is identical — $h$ heads of dimension $d/h$ has the same $W_Q, W_K, W_V$ parameter count as one head of dimension $d$. What changes is the number of *independent softmaxes*.

A single attention head computes one probability distribution per query position. That distribution is a convex combination, so the output is a weighted average of value vectors. Averaging is lossy: if a token genuinely needs to gather the syntactic subject *and* the coreferent antecedent *and* the nearest number, one distribution has to split mass across all three, and the output is a blend that may not be decodable back into its parts. With $h$ heads you get $h$ distributions, each free to be sharply peaked on a different thing, and $W_O$ combines them additively rather than by averaging.

The second, more mechanical reason is rank. In one head with $d_k = d_{\text{model}}$, the QK circuit $W_Q^\top W_K$ is a full-rank $d \times d$ bilinear form — expressive, but you've spent all your capacity on one relation. Splitting into $h$ heads gives you $h$ separate rank-$d/h$ bilinear forms. That's strictly less expressive in a naive sense but far better matched to the actual structure of language, where you want many cheap specialized relations rather than one expensive general one. Empirically that trade lands well on the useful side.

There's a caveat worth volunteering: heads are much less cleanly specialized than the 2017-era "one head does syntax, one does coreference" story suggests. Michel et al. showed you can prune a large fraction of heads at inference with little loss, and many heads are redundant or degenerate (attending to the BOS token as a no-op — the "attention sink" behavior). So the honest version is: multiple heads give you multiple routing patterns per layer, that's clearly valuable, but the individual heads are not tidy interpretable modules in general.

**Follow-up: "If many heads are prunable, why not just train with fewer?"** → Because you don't know in advance which ones will matter, and the redundancy appears to help optimization — it's a lottery-ticket-ish effect where having many candidate routing patterns early makes it likely some become useful. Training a small-head-count model from scratch generally underperforms pruning a large-head-count one down to the same size. That said, modern large models have moved toward fewer, wider heads relative to parameter count than the original scaling suggested, partly for exactly this reason and partly for KV cache cost.

### Q: Walk me through MHA, MQA, and GQA. What problem is GQA solving?

**Answer.** The problem is the KV cache, and it's an inference-memory-bandwidth problem, not a training problem.

At autoregressive decode time, you generate one token per forward pass, but each new token must attend to every previous token. Rather than recompute keys and values for the whole prefix each step, you cache them. Cache size for multi-head attention is

$$2 \times L \times n_{\text{heads}} \times d_{\text{head}} \times s \times b \times \text{bytes},$$

where the 2 is K and V, $L$ is layers, $s$ is sequence length, $b$ is batch. Plug in something Llama-2-70B-shaped — 80 layers, 64 heads, head dim 128, fp16 — and you get about 2.5 MB per token per sequence, so 32k of context is around 80 GB. That doesn't fit, and worse, every decode step must *read the entire cache from HBM*. Decode is memory-bandwidth-bound, so cache size translates almost linearly into tokens-per-second.

Multi-query attention is the extreme fix: keep $n$ query heads but have all of them share a *single* K head and a single V head. Cache shrinks by a factor of $n_{\text{heads}}$ — that 80 GB becomes about 1.2 GB. The cost is quality: you've collapsed all the key/value diversity into one subspace, and MQA models measurably degrade, plus they're less stable to train.

Grouped-query attention is the interpolation. Partition the $n$ query heads into $g$ groups; each group shares one K/V head. $g = n$ recovers MHA, $g = 1$ recovers MQA. In practice $g = 8$ is the common choice — Llama-2-70B and Llama-3 use 8 KV heads. That gives an 8× cache reduction for near-MHA quality. The empirical finding from the GQA paper is that quality degrades very gently as you reduce $g$ until you get near 1, where it falls off — so there's a sweet spot, and 8 is roughly where it sits for large models.

One nice practical detail: you can *uptrain* an existing MHA checkpoint into GQA by mean-pooling the K and V projections within each group and then training on about 5% of the original token budget. You don't have to pretrain from scratch.

**Follow-up: "Why $g = 8$ specifically? Is there something principled there?"** → It's mostly empirical, but there's a systems reason it's a convenient number: it matches typical tensor-parallel degree. With 8-way TP, one KV head per GPU means no KV replication across devices and no extra communication. So the choice is a coincidence of "quality is fine here" and "this maps cleanly onto an 8-GPU node."

**Follow-up: "How does Multi-head Latent Attention differ from GQA?"** → MLA, from DeepSeek-V2, attacks the same cache problem differently: instead of *sharing* KV heads, it caches a single low-rank latent vector per token and reconstructs per-head K and V from it with learned up-projections. The cache is the latent, which is much smaller than even GQA's, and because the reconstruction is per-head you retain more head diversity than GQA does. The reported claim is better quality than MHA at a smaller cache than GQA; the catch is that it's incompatible with RoPE as-is, so they carry a small extra "decoupled" RoPE-carrying dimension alongside the latent.

*Trap:* Saying GQA reduces FLOPs. Query-side compute is unchanged; you still compute $n$ query heads and $n$ attention patterns. It reduces cache memory and the bandwidth to read it, which is what actually binds at decode.

### Q: Compare positional encoding schemes — sinusoidal, learned absolute, RoPE, ALiBi. Why did RoPE win?

**Answer.** Attention is permutation-equivariant: $\text{softmax}(QK^\top)V$ is invariant to reordering tokens if you reorder consistently, so with no positional signal a transformer sees a bag of tokens. Everything here is about injecting order.

*Sinusoidal absolute* (original transformer): add a fixed vector $PE_{pos}$ to the embedding, with $PE_{pos,2i} = \sin(pos/10000^{2i/d})$ and cosine for odd indices. The pitch was that relative position is linearly recoverable, since $PE_{pos+k}$ is a linear function of $PE_{pos}$ — but the model has to learn to exploit that, and it's fighting the fact that position is being added into the same vector space as semantics.

*Learned absolute* (BERT, GPT-2): a lookup table of position embeddings. Simple, works fine in-distribution, but has a hard cap at the trained max length and generalizes to longer contexts not at all.

*RoPE*: instead of adding anything, rotate. Split each query and key into $d/2$ two-dimensional pairs and rotate pair $i$ at position $m$ by angle $m\theta_i$, with $\theta_i = 10000^{-2i/d}$. The key property is that the dot product between a rotated query at $m$ and a rotated key at $n$ depends only on $m - n$:

$$\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle.$$

So you get exact relative position *for free inside the existing dot product*, with no extra parameters, no added bias term, and no interference with the residual stream — the rotation is norm-preserving, so it doesn't perturb magnitudes.

*ALiBi*: skip encoding entirely, just add a linear distance penalty to the attention logits: $q_i^\top k_j - m_h(i-j)$, with a per-head slope $m_h$. Extremely simple, and it extrapolates to longer sequences out of the box.

Why RoPE won: it hits the sweet spot. It's relative (which is what actually generalizes), it's parameter-free, it doesn't consume residual-stream capacity, it's cheap, and crucially it's *extendable* — you can stretch the context of a trained RoPE model post hoc by manipulating $\theta$. Position interpolation (divide positions by a factor), NTK-aware scaling, and YaRN all exploit the frequency structure to take a 4k model to 32k or 128k with a small amount of fine-tuning. Nothing else offers that. ALiBi extrapolates but does so by *decaying* attention with distance, which means it doesn't really use far context — its long-range performance is more "gracefully ignores" than "attends correctly." Learned absolute has no path forward at all.

**Follow-up: "Naively, RoPE should extrapolate — it's relative. Why doesn't it?"** → Because the low-frequency dimensions haven't completed a full period within the training length. A dimension with $\theta_i$ small has period $2\pi/\theta_i$ far longer than the training context, so at positions beyond training the model sees rotation angles it has literally never observed for those dimensions — out-of-distribution inputs, not interpolated ones. The high-frequency dimensions wrap around many times and are fine. That's exactly the insight behind NTK-aware scaling and YaRN: don't interpolate all frequencies uniformly, leave high frequencies alone (they've seen full periods) and interpolate the low ones (they haven't).

**Follow-up: "What's the base $\theta$ = 10000 doing, and why do long-context models raise it?"** → It sets the spread of frequencies across dimensions. Raising the base to e.g. 500,000 (Llama-3) lengthens all periods, so more dimensions complete less than a full rotation within a given context — effectively reserving more of the spectrum for long-range distinctions and reducing the OOD problem at long positions. The trade is coarser resolution at short distances, which matters less than you'd think because short-range structure is also carried by local content.

*Trap:* Describing RoPE as "adding rotary embeddings to the input." It's applied to $Q$ and $K$ only, inside each attention layer, every layer — not to $V$ and not to the residual stream. Applying it to $V$ would break the relative-position property.

### Q: How does causal masking actually work, and where in the computation does it go?

**Answer.** You compute the full score matrix $S = QK^\top/\sqrt{d_k}$, which is $s \times s$, then add a mask $M$ where $M_{ij} = 0$ if $j \le i$ and $-\infty$ if $j > i$, *before* the softmax. Softmax of $-\infty$ is exactly 0, so position $i$ puts zero probability on any future position. Then $\text{softmax}(S + M)V$.

Two implementation details worth knowing. First, it must be pre-softmax. If you masked after the softmax by zeroing entries, the remaining probabilities wouldn't sum to 1 and you'd need to renormalize — and you'd have already leaked future information into the denominator. Pre-softmax masking gets the normalization right automatically because the exponential of $-\infty$ contributes nothing to the sum. Second, in practice you use a large negative finite number rather than literal $-\infty$ (typically the dtype minimum or around $-10^9$), because $-\infty$ times 0 in some autograd paths produces NaN.

The reason this matters conceptually is that it's what lets you train on all positions in parallel. Without a mask you'd have to run the model once per prefix. With it, one forward pass over a length-$s$ sequence gives you $s$ training signals — the prediction at every position, each conditioned only on its own prefix. That parallelism over the sequence dimension is the entire reason decoder-only transformers are trainable at scale; RNNs can't do it because their recurrence is sequential.

At inference with a KV cache, the mask mostly disappears: when you're decoding token $t$ you have one query attending to $t$ cached keys, all of which are in the past by construction, so there's nothing to mask. The mask only reappears for the prefill pass over the prompt, and for batched decoding where you need padding masks.

**Follow-up: "In FlashAttention the full $s \times s$ score matrix is never materialized. How does masking work there?"** → Per tile. FlashAttention iterates over blocks of keys for each block of queries, and applies the mask to the block's scores in SRAM before the block's softmax contribution is accumulated. Better still, blocks entirely above the diagonal can be *skipped* — they'd be all $-\infty$ — which is why causal FlashAttention is roughly 2× faster than non-causal, matching the fact that you only need half the score matrix.

**Follow-up: "You're packing multiple documents into one 8k sequence. What goes wrong?"** → Without a document mask, tokens in document 2 attend to document 1, which is both a semantic contamination and a subtle train/test mismatch versus inference where documents arrive alone. The fix is block-diagonal masking — a mask that is causal *within* each document and $-\infty$ across boundaries — plus resetting position IDs per document so RoPE doesn't see one giant sequence. FlashAttention's variable-length API supports exactly this via cumulative sequence-length offsets.

### Q: Why LayerNorm instead of BatchNorm in transformers?

**Answer.** Several reasons, and the sequence-length one is the decisive one.

BatchNorm normalizes each feature across the batch: for feature $j$, subtract the batch mean and divide by the batch std computed over all examples. In a transformer the "batch" for a given feature includes every token position of every sequence, and sequences have different lengths. So the statistics for a feature depend on how many tokens happened to be in the batch at each position, which is a nuisance variable. Positions near the end of the max length are covered by only a few unpadded sequences, so their statistics are estimated from tiny samples and are wildly noisy.

Second, and worse for a decoder: BatchNorm at training time uses batch statistics, which means the representation of token $i$ in sequence A depends on tokens in sequence B. That's a dependency across examples, and in a causal model it's a route by which information can flow in ways the mask doesn't control. It also makes the model's output on one example depend on what else was in the batch, which is bad at inference.

Third, the train/inference mismatch. BatchNorm keeps running averages of mean and variance for inference. In an autoregressive model generating token by token, the distribution of activations shifts with position and with generation length, so those running stats — estimated on training-time distributions — are systematically wrong.

LayerNorm sidesteps all of it: it normalizes across the *feature* dimension, per token, independently. $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ with $\mu, \sigma$ computed over the $d_{\text{model}}$ features of that single token. No batch dependence, no sequence-length dependence, identical behavior at train and inference, works with batch size 1.

Modern models mostly use RMSNorm instead: $\text{RMS}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$ — drop the mean subtraction and the bias. It's cheaper (no mean pass, fewer reductions), and Zhang and Sennrich's finding, which has held up, is that the re-centering does essentially nothing; the re-scaling is what matters. Llama, T5, Gemma, and most current models use RMSNorm.

**Follow-up: "Why does normalization help at all? Is it internal covariate shift?"** → That was the original story and it's largely been discredited — Santurkar et al. showed you can inject noise after BatchNorm to deliberately *increase* covariate shift and still get the training benefit. The better explanation is loss-landscape smoothing: normalization makes the loss and its gradients more Lipschitz, so larger, more stable steps are possible. There's also a scale-invariance effect — because the output is invariant to the scale of the incoming weights, the effective learning rate auto-adjusts, which is a real and mechanically clean benefit.

*Trap:* Saying "BatchNorm doesn't work with variable-length sequences" and stopping there. That's true but you can pad and mask; the deeper reasons are the cross-example dependency and the train/inference statistics mismatch in autoregressive generation.

### Q: Pre-norm versus post-norm. Why did pre-norm become the default?

**Answer.** Post-norm is the original: $x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))$ — normalize *after* the residual add. Pre-norm is $x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$ — normalize the input to the sublayer, and leave the residual path untouched.

The difference is what happens to the residual path. In pre-norm, there is a completely unnormalized, unscaled identity path from the embedding all the way to the output. Gradients flow along it with no attenuation at all — the derivative of $x_{l+1}$ with respect to $x_l$ contains an exact identity term. In post-norm, the LayerNorm sits *on* the residual path, so every layer's gradient gets multiplied by the LayerNorm Jacobian on the way back. Those Jacobians have norm typically less than one, and multiplying $L$ of them together makes deep post-norm networks have exponentially attenuated gradients at early layers.

The practical consequence is warmup. Post-norm transformers essentially cannot be trained without a careful learning-rate warmup — Xiong et al. showed that post-norm gradients at initialization scale like $O(d\sqrt{\ln d})$ at the output layer regardless of depth, and the mismatch across layers means a large initial LR blows up. Pre-norm gradients are well-behaved at init, so pre-norm models train with much less warmup, tolerate larger learning rates, and simply don't diverge as easily. At 100+ layers post-norm is basically untrainable without extra tricks.

The cost of pre-norm is real though, and it's worth mentioning because it's the follow-up: the residual stream grows in magnitude with depth, since each layer adds to it and nothing rescales it. By late layers the residual is large relative to what any individual sublayer contributes, so later layers have diminishing marginal influence — the "representation collapse" issue. This is part of why very deep pre-norm models show less benefit per added layer than you'd hope, and it's why you always put a final LayerNorm before the output head.

So the honest summary: pre-norm won because it's trainable at scale, and the field accepted a modest expressivity cost for that. The current state is more nuanced — there are hybrids. Gemma-2 and Grok use "sandwich" norm (normalize both before and after each sublayer, with the second one inside the residual branch), and DeepSpeed-style DeepNorm rescales the residual to make post-norm trainable at 1000 layers. So it's not settled that pre-norm is optimal, only that it's robust.

**Follow-up: "Why exactly does post-norm need warmup and pre-norm doesn't?"** → With post-norm the gradient magnitude is very uneven across layers at initialization, so a single global LR is either too big for the layers with large gradients or too small for the rest. Warmup starts small enough that no layer explodes, and by the time the LR is large the network has reached a region where the gradients have equilibrated. Pre-norm's gradients are roughly depth-independent at init, so a single LR works from step one. Pre-norm models still typically use *some* warmup, but that's for Adam's second-moment estimate, which is a separate issue.

*Trap:* Claiming pre-norm is strictly better. Post-norm, when it trains, often reaches slightly better final quality — the original BERT and the strongest translation models were post-norm. The win for pre-norm is stability at scale, not final loss.

### Q: Encoder-only, decoder-only, encoder-decoder. When would you pick each in 2026?

**Answer.** The architectural distinction is about masking and cross-attention. Encoder-only (BERT) is bidirectional: every token attends to every other, trained with masked-language modeling. Decoder-only (GPT) is causal: token $i$ sees only $\le i$, trained with next-token prediction. Encoder-decoder (T5, original transformer) has a bidirectional encoder over the input, a causal decoder over the output, and cross-attention layers where decoder queries attend to encoder keys and values.

Where each is right. Encoder-only still wins for *pure representation* tasks at scale: dense retrieval embeddings, reranking, token classification, high-throughput classification where you need one forward pass and a fixed-size output. Bidirectional context is genuinely better for encoding a fixed input, and these models are 100× smaller and cheaper than an LLM doing the same job. ModernBERT and similar 2024–25 refreshes brought the architecture up to date (RoPE, FlashAttention, 8k context) and they remain the right tool. That said, the frontier of retrieval embeddings has partly shifted to decoder-based models with the causal mask removed or with last-token pooling, so this is contested.

Decoder-only dominates general-purpose generation, and the reasons are more practical than theoretical. Training is maximally efficient: every position gives you a supervised signal, versus MLM which only trains on the ~15% masked tokens. The architecture is uniform, so scaling and parallelism engineering is simpler. It handles arbitrary tasks via prompting since input and output live in one stream. And KV caching makes incremental decoding cheap.

Encoder-decoder makes sense when input and output are structurally distinct and the input is fixed: translation, summarization of a long fixed document, speech-to-text. You get a full bidirectional read of the input and you encode it *once* regardless of output length. For a long input and long output that's a real efficiency win over re-attending to the input causally. It's also better where you want an information bottleneck between input and output.

In 2026 the practical answer skews heavily decoder-only for anything generative, encoder-only for embeddings and cheap classification, and encoder-decoder in specific domains — it never disappeared in speech and translation.

**Follow-up: "Why is MLM less sample-efficient than causal LM, quantitatively?"** → BERT masks 15% of tokens and computes loss only on those, so per forward pass you get 0.15 signals per token versus 1.0 for causal LM — roughly a 6–7× difference in gradient signal per unit of compute. You can raise the mask rate, but past ~40% you destroy too much context for the task to be learnable. There's also a pretrain/finetune mismatch: `[MASK]` never appears at inference. ELECTRA's replaced-token detection was a direct attempt to fix the efficiency problem by getting a loss at every position, and it worked.

*Trap:* "Decoder-only is better because bidirectional attention isn't needed." Bidirectional attention *is* better for encoding — that's not the reason. Decoder-only won on training efficiency, engineering uniformity, and task generality.

### Q: Why is the FFN hidden dimension typically 4× the model dimension?

**Answer.** Start with what the FFN is for. Attention moves information between positions but is, per position, a linear map on values. The FFN is where per-token nonlinear computation happens. The clean interpretation from the interpretability literature is key-value memory: the first matrix $W_1 \in \mathbb{R}^{d \times 4d}$ has $4d$ rows acting as pattern detectors — each row dotted with the residual stream produces an activation, so it's "does this token's representation match key $i$" — and $W_2 \in \mathbb{R}^{4d \times d}$ has $4d$ rows that are the values written back when the corresponding key fires. So the hidden width is literally the number of memory slots.

The 4× is empirical, from the original transformer, and it has stuck because it sits near a broad optimum. Wider gives more memory slots but the FFN already dominates parameter count — $8d^2$ for the FFN versus $4d^2$ for attention, so two-thirds of the non-embedding parameters — and quality per parameter falls off. Narrower and you underfit the per-token computation. Ablations have consistently shown the curve is fairly flat between roughly 2.5× and 6×, so 4 is a convenient round number in a shallow basin rather than a magic constant.

The important modern wrinkle: with gated activations the ratio changed. SwiGLU, which most current models use, computes $\text{FFN}(x) = (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ — three matrices instead of two. To hold parameter count constant, you scale the hidden dimension by $2/3$, giving $\frac{8}{3}d \approx 2.67d$. Llama uses exactly this, usually rounded to a multiple of 256 for hardware alignment. So if someone asks "why 4×" about a Llama-shaped model, the right answer includes "it isn't 4× anymore, it's about 2.67× because of the gate, chosen to preserve the parameter budget."

There's also a superposition argument for why wide matters: models represent far more features than they have dimensions, packed in near-orthogonal directions. The FFN's expansion gives room to temporarily unpack those features into a higher-dimensional, more nearly-one-feature-per-neuron space, do the computation, and repack. Under that view the expansion ratio is a bet on how much superposition you need to resolve.

**Follow-up: "Why does SwiGLU beat plain ReLU or GELU?"** → The gate makes it multiplicative rather than purely additive, so a single layer can express input-dependent gating — "let this feature through only when that other feature is present" — which a pointwise nonlinearity cannot do without depth. Noam Shazeer's own paper on it famously ends by attributing the success to "divine benevolence," which is a candid admission that there's no rigorous theory. Empirically the gain is small but consistent, roughly a fraction of a percent of loss at fixed parameters, and it's free at inference.

### Q: Explain the residual stream. Why does that framing matter?

**Answer.** In a pre-norm transformer, the value flowing from embedding to unembedding is never transformed in place — every layer only *adds* to it:

$$x_{l+1} = x_l + \text{Attn}(\text{LN}(x_l)) + \text{FFN}(\text{LN}(x_l')).$$

So the final representation is literally $x_0 + \sum_l (\text{attn contribution}_l + \text{ffn contribution}_l)$. The stream is a shared communication bus of dimension $d_{\text{model}}$ that every layer reads from and writes to.

Why this framing is load-bearing: it means the transformer's computation is *additive and decomposable*. Each head reads a subspace via $W_{QK}$ and $W_{OV}$, and writes into a subspace determined by $W_O$. Because the operations compose by addition, you can meaningfully ask "what did layer 7 head 3 contribute to this logit" and get an answer — that's the whole basis for logit lens, activation patching, and direct logit attribution. It's why we can talk about induction heads as a *circuit*: a previous-token head in layer $k$ writes a signal into the stream, and an induction head in layer $k+m$ reads exactly that signal. They compose through the stream without ever touching each other directly.

It also explains a few practical phenomena. The stream's norm grows with depth in pre-norm models, because layers keep adding; that's why you need a final LN before the head. It explains why the model is robust to deleting or reordering some layers — each is a small additive correction, not a required transformation. It explains superposition: with $d_{\text{model}}$ dimensions and far more features to represent, features get packed into near-orthogonal directions, and different layers implicitly negotiate which directions they use. And it explains why LoRA works so well on attention output projections: you're adding a low-rank correction to what a component writes into the stream.

The metaphor I'd give in an interview is a shared whiteboard. Nobody erases; everyone writes. Later layers read what earlier ones wrote, but they have to find it among everything else, which is exactly what the QK circuit is for.

**Follow-up: "Post-norm doesn't have a clean residual stream. Does the framing break?"** → It weakens considerably. In post-norm the LN sits on the main path, so the stream is rescaled at every layer and the contributions of early layers are progressively squashed and mixed. You can't decompose the final logits into a clean sum of layer contributions. That's one underrated reason interpretability work concentrated on pre-norm models.

### Q: What's the time and memory complexity of attention, and what does FlashAttention actually change?

**Answer.** Standard attention over sequence length $s$ and model dimension $d$: computing $QK^\top$ is $O(s^2 d)$ FLOPs, softmax is $O(s^2)$, and multiplying by $V$ is another $O(s^2 d)$. So $O(s^2 d)$ time. Memory is the killer — the score matrix is $s \times s$ per head, so $O(s^2)$ memory per head, and you have to keep it for the backward pass.

Now, what FlashAttention changes, and this is where most candidates get it wrong: **it does not reduce FLOPs.** It computes exactly the same attention, bit-for-bit equivalent up to floating-point reassociation. It's *not* an approximation — that distinguishes it from Linformer, Performer, sparse attention, and the rest of the efficient-attention literature, essentially all of which trade accuracy for speed and essentially none of which are used in frontier models.

What FlashAttention reduces is **HBM traffic**. The observation is that attention on modern GPUs is memory-bandwidth-bound, not compute-bound: an A100 does ~312 TFLOPS of fp16 but only ~2 TB/s of HBM bandwidth, and SRAM is ~19 TB/s but only ~20 MB. Naive attention writes the $s \times s$ score matrix to HBM, reads it back for softmax, writes the softmax result, reads it back for the $V$ multiply — several $O(s^2)$ round trips to the slowest memory in the system.

FlashAttention fixes this with two techniques. **Tiling**: split $Q$, $K$, $V$ into blocks that fit in SRAM, and for each query block iterate over key blocks, accumulating the output in SRAM. **Online softmax** makes this correct — you maintain a running max $m$ and running sum $\ell$, and when a new block arrives with a larger max, you rescale the accumulated output by $e^{m_{\text{old}} - m_{\text{new}}}$ before adding the new contribution. That's the Milakov–Gimelshein trick, and it means you never need the full row of scores at once. **Recomputation**: for the backward pass, instead of storing the $s\times s$ attention matrix, store only the softmax normalization statistics ($O(s)$) and recompute the scores on the fly. That's more FLOPs than the naive backward, and it's still faster, which tells you exactly how bandwidth-bound the operation was.

Result: HBM accesses drop from $O(s^2)$ to $O(s^2 d^2 / M)$ where $M$ is SRAM size, memory goes from $O(s^2)$ to $O(s)$, and wall-clock is 2–4× faster. FlashAttention-2 improved work partitioning and reduced non-matmul FLOPs for roughly another 2×; FlashAttention-3 added Hopper-specific asynchrony and FP8.

**Follow-up: "So attention is still $O(s^2)$. How do we serve 1M-token contexts?"** → Several things stack. Memory is now $O(s)$, so the quadratic *memory* wall is gone and only time is quadratic — and prefill is compute-bound and parallel, so it's tolerable. Beyond that: GQA/MLA shrink the KV cache, ring attention shards the sequence across devices, and increasingly models use hybrid architectures — sliding-window attention in most layers with a few full-attention layers (Mistral, Gemma-2, Character.AI's setup), or interleaved SSM/attention layers (Jamba). So the practical answer is "we made the constant tiny and made most layers sub-quadratic," not "we solved the quadratic."

**Follow-up: "Why did the sub-quadratic approximate attention methods not win?"** → Two reasons. Their asymptotic win only materializes at sequence lengths where the constant factors stop dominating, and FlashAttention pushed that crossover point way out. And the quality loss is real but shows up in exactly the hard-to-measure places — long-range retrieval, needle-in-a-haystack — so it doesn't show in perplexity but does show in use. Tri Dao's own framing was that people were optimizing FLOPs when the bottleneck was memory movement, and once that was fixed the motivation for approximation largely evaporated.

*Trap:* "FlashAttention makes attention linear" or "reduces complexity to $O(s\log s)$." It's exact and still quadratic in time. Only *memory* becomes linear.

### Q: Why do transformers need an FFN at all? Attention already mixes information.

**Answer.** Because attention is, per output position, a *linear* function of the values. Yes, the attention weights are computed by a nonlinear softmax, but conditional on the weights, the output is $\sum_j \alpha_{ij} v_j$ — a convex combination of linear projections of the inputs. If you stacked attention layers with no FFN, you would still get some nonlinearity through the softmax's dependence on the input, but the *content* pathway would be a composition of linear maps and weighted averages.

There's a formal result here worth citing: Dong, Cordonnier, and Loukas showed that pure self-attention networks without skip connections and FFNs converge doubly exponentially with depth to a rank-1 matrix — every token's representation collapses to the same vector. They call it "token uniformity" or rank collapse. The FFN and the skip connections are precisely what counteract it: skip connections preserve rank by carrying the input forward, and the FFN's nonlinearity prevents the geometric contraction.

Functionally, the division of labor is: attention decides *what information to gather from where* — it's the routing mechanism, and it's the only part of the transformer that moves information across positions. The FFN does *per-token processing* — it's the only part that applies a nonlinear function of a single position's representation, and it holds most of the parameters and, on the evidence, most of the factual knowledge. The ROME and knowledge-editing work localizes factual recall to mid-layer FFNs specifically, and Geva et al.'s key-value memory framing gives a mechanism for why.

You can also just check the numbers: the FFN is $8d^2$ parameters per layer versus $4d^2$ for attention. Two-thirds of the model is FFN. If it were doing nothing you'd have noticed.

**Follow-up: "Would a transformer with no FFN but twice the attention layers work?"** → Substantially worse. It's been ablated repeatedly; the model loses most of its ability to store facts and its per-token computation, and the rank-collapse pressure gets stronger. The residual stream keeps it from fully collapsing but the capacity loss is severe. Interestingly the reverse — an FFN-heavy, attention-light model — degrades more gracefully for short contexts and badly for anything requiring long-range composition, which is a clean confirmation of the routing/processing split.
---

## 2. Training Dynamics and Optimization

### Q: Vanishing and exploding gradients — what causes each, and what actually fixes each?

**Answer.** Both come from the same place: backprop through $L$ layers multiplies $L$ Jacobians, and the product of many matrices tends to either shrink or grow geometrically. If the typical singular value of the per-layer Jacobian is $\sigma$, the gradient at the input scales roughly like $\sigma^L$. For $\sigma < 1$ that's exponential decay; for $\sigma > 1$ exponential growth. There's no stable middle unless you engineer one.

For **vanishing**, historically the dominant cause was saturating activations. Sigmoid has derivative at most $0.25$, so even in the best case you lose a factor of 4 per layer — ten layers gives you $10^{-6}$. Tanh maxes at 1 but saturates for large inputs. What actually fixed it, in rough order of importance:

1. **Non-saturating activations** — ReLU has derivative exactly 1 on the positive side, so it doesn't attenuate. GELU/SwiGLU are smooth versions of the same idea.
2. **Residual connections** — this is the big one for depth. $\partial x_{l+1}/\partial x_l = I + \partial F/\partial x_l$. The identity term means the gradient has a path with *no* multiplication at all; the product over layers is $\prod(I + J_l)$, which expands into a sum containing the pure identity path. Gradient reaches layer 1 regardless of depth. This is why ResNets went from 20 layers to 150 and why transformers go to 100+.
3. **Normalization** — keeps activations in the non-saturating regime and makes the Jacobian's scale roughly depth-independent.
4. **Careful init** — He init sets $\text{Var}(W) = 2/n_{\text{in}}$ so ReLU layers preserve variance forward; Xavier does the same for tanh. This buys you a good starting $\sigma \approx 1$; it doesn't hold during training, which is why you need the others.

For **exploding**, the causes are different: large weights, high learning rate, a bad batch, or numerical issues in the loss. The fix is different too — you don't fix exploding gradients architecturally, you *clip* them. Global norm clipping: compute $g = \|\nabla\|_2$ over all parameters concatenated, and if $g > c$, scale the whole gradient by $c/g$. The key property is that it preserves direction and only rescales magnitude, so you don't distort the update, you just bound its size. Typical $c$ is 1.0 for LLM training. Note that clipping is essentially universal in transformer training and essentially absent from ResNet training — that's because language data has heavy-tailed batch-to-batch variation (a batch with weird tokens produces a huge gradient) in a way that ImageNet doesn't.

RNNs are the pathological case for both, because the *same* weight matrix is applied at every timestep, so the Jacobian product is $W^T$ — a literal matrix power, and the spectral radius decides everything. LSTMs fix vanishing with an additive cell-state path (the same trick as residuals) and gates; they still need clipping for exploding.

**Follow-up: "You're clipping at 1.0 and 30% of steps hit the clip. Is that bad?"** → Not necessarily bad, but it means your clip threshold is now effectively a learning-rate cap and you've lost the intended "only intervene on outliers" semantics — you're doing normalized SGD most of the time. I'd look at a histogram of pre-clip gradient norms. If there's a clean bulk plus a thin tail, raise the threshold to sit above the bulk. If the whole distribution has drifted up, that's a signal the LR is too high or something upstream is wrong. Watching the clip *fraction* over training is one of the cheapest useful diagnostics you can log.

*Trap:* Saying "use ReLU" as the fix for vanishing gradients in a deep network. ReLU alone gets you maybe 20-30 layers. Residual connections are the thing that actually broke the depth barrier.

### Q: Why does Adam work so much better than SGD for transformers, when SGD with momentum wins on ResNets?

**Answer.** This is genuinely one of the better open questions in optimization, and I'd give the leading explanations rather than pretend it's settled.

The mechanical difference: Adam maintains per-parameter first and second moment estimates and takes steps of size $\eta \cdot \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$. Because it divides by the RMS of recent gradients, the step size for each parameter is roughly *scale-invariant* — a parameter with tiny gradients gets the same effective step as one with huge gradients.

Why that matters more for transformers:

**Heterogeneous gradient scales.** A transformer has embeddings, attention projections, LayerNorm gains, and FFN weights, and their gradient magnitudes differ by orders of magnitude. Embeddings in particular get extremely sparse gradients — a rare token's embedding row receives a nonzero gradient in maybe one batch in a thousand, and SGD with a global LR will effectively never train it. Adam's normalization means a rare token's embedding gets a full-size step whenever it does appear. A ResNet's parameters are far more homogeneous.

**Heavy-tailed gradient noise.** Zhang et al. (2020) showed the gradient noise distribution in language tasks is heavy-tailed (roughly $\alpha$-stable with $\alpha < 2$), whereas in vision it's closer to Gaussian. Under heavy-tailed noise, SGD's convergence guarantees degrade badly, while adaptive methods with clipping remain well-behaved. This is probably the most convincing single explanation and it's data-driven, not architecture-driven — which predicts that SGD should also struggle on a CNN trained on text-like data, and it does.

**Loss landscape conditioning.** Transformer loss surfaces have much worse conditioning — the Hessian eigenvalue spectrum is more spread out. Adam's per-parameter scaling is a crude diagonal preconditioner, which helps exactly here. Related recent work (Kunstner et al.) argues class imbalance is the key driver: language has a Zipfian token distribution, so the loss is dominated by frequent tokens, and SGD makes almost no progress on the rare-token part of the loss while Adam does.

The cost is memory: Adam stores $m$ and $v$ per parameter, so 8 bytes per parameter in fp32 on top of the weights. For a 70B model that's 560 GB of optimizer state, which is why ZeRO sharding and 8-bit optimizers exist.

I'd also flag that this isn't a permanent verdict. Shampoo, SOAP, and Muon — which use structural (matrix-level) preconditioning rather than diagonal — have been beating AdamW on transformer training in recent benchmarks, and Muon in particular has been used in production-scale runs. So the field is actively moving off "Adam is just what you use."

**Follow-up: "What does $\epsilon$ do in Adam and how should you set it?"** → It prevents division by zero and, more importantly, bounds the maximum effective step when $\hat v$ is tiny. It is *not* a negligible constant — for parameters with consistently small gradients, $\epsilon$ dominates the denominator and Adam degenerates toward SGD for those parameters. Default $10^{-8}$; large-model training often uses $10^{-8}$ to $10^{-6}$, and raising it is a known stabilization lever when you're getting loss spikes. In bf16 you have to be careful because $\hat v$ can underflow, which is why optimizer states are usually kept in fp32 regardless of the compute dtype.

### Q: AdamW versus Adam. Why does decoupling weight decay matter?

**Answer.** In vanilla Adam, "weight decay" is implemented as L2 regularization: you add $\lambda \theta$ to the gradient before the moment updates. That means the decay term flows through the adaptive normalization. The update becomes

$$\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}, \quad \text{where } m,v \text{ are built from } (g_t + \lambda\theta_t).$$

The problem: the decay contribution gets divided by $\sqrt{\hat v_t}$, which is parameter-specific. So a parameter with large historical gradients gets its decay *shrunk*, and a parameter with small gradients gets its decay *amplified*. That's exactly backwards from what you want — the weights with big gradients are the ones actively doing work and could use regularization, while the ones with small gradients are already near-dead and you're crushing them further. The effective regularization strength becomes an uncontrolled function of the gradient history.

AdamW decouples it:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t\right),$$

with the decay applied directly to the parameters, outside the adaptive term. Now every parameter shrinks by the same multiplicative factor per step, which is what "weight decay" is supposed to mean, and the regularization strength is a clean hyperparameter you can tune independently of the learning rate.

The practical payoff Loshchilov and Hutter demonstrated is that AdamW *decouples the hyperparameter search*. With Adam+L2, changing the LR changes the effective decay, so the optimal $(\eta, \lambda)$ pair sits on a curved ridge and you have to search jointly. With AdamW the optimal values are much closer to independent, so you can tune them separately — a big practical saving. And AdamW closed most of the generalization gap to SGD+momentum on image tasks, which was the original motivating puzzle.

Note that $\eta$ still multiplies $\lambda\theta$ in the standard formulation, so LR schedule and decay are not *fully* decoupled — decay anneals with the LR. Some implementations make it fully independent. Also, in practice you exclude biases, LayerNorm gains, and often embeddings from decay: those parameters aren't the ones causing overfitting, and decaying a LayerNorm gain toward zero actively fights the normalization.

**Follow-up: "In LLM pretraining you're doing one epoch over trillions of tokens. There's no overfitting. Why decay at all?"** → It's not acting as a regularizer there — it's acting as an optimization aid. Decay keeps weight norms bounded, which for scale-invariant parts of the network keeps the *effective* learning rate from decaying as norms grow, and it prevents the slow blow-up of logit magnitudes that causes late-training instability. There's also evidence it improves the loss directly at fixed compute, i.e. it's doing something to the trajectory rather than to the generalization gap. Typical value is 0.1 for LLM pretraining, which is much larger than the 1e-4 you'd see in vision.

*Trap:* Saying "AdamW applies decay to the weights, Adam applies it to the gradient, they're the same thing." They're only the same for plain SGD, where the two are algebraically identical. Adaptivity is exactly what breaks the equivalence.

### Q: Why does Adam need learning-rate warmup?

**Answer.** The core reason is that Adam's second-moment estimate $\hat v_t$ is unreliable at the start, and an unreliable denominator produces enormous, badly-directed steps.

Concretely: $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ with $v_0 = 0$. Bias correction divides by $(1 - \beta_2^t)$, which corrects the *expectation* but not the *variance*. At $t=1$, $\hat v_1 = g_1^2$ exactly, so the update is $\eta \cdot g_1/|g_1| = \pm\eta$ — a full-magnitude step in a direction determined by a single minibatch. At small $t$, $\hat v_t$ is estimated from a handful of samples, so its variance is huge, and since it appears in the *denominator*, an unluckily small estimate produces an enormous step. With $\beta_2 = 0.999$ the effective averaging window is ~1000 steps, so you need on the order of a thousand steps before the estimate settles. That's why warmup lengths of 1000–10000 steps are typical, and why they're roughly independent of dataset size.

That's the Liu et al. (RAdam) analysis, and their proposed alternative — rectify the variance analytically and skip warmup — works, which is decent evidence the diagnosis is right.

There are two other contributing factors worth mentioning. First, at initialization the model's predictions are near-uniform and the loss surface is steep and poorly conditioned; large early steps can push you into a region you never recover from — this is architecture-related and applies to SGD too, though less severely. Second, for post-norm transformers the gradient scale is very uneven across layers at init, so a global LR that's fine for one layer is catastrophic for another; warmup gives layers time to equilibrate. Pre-norm reduces but doesn't eliminate this.

The empirical signature is clean: no warmup on a large transformer typically gives you either an immediate divergence or an early loss spike the model spends thousands of steps recovering from.

**Follow-up: "How long should warmup be, and does it scale with anything?"** → Rules of thumb: a few thousand steps, or roughly 1% of total steps for a long run, or about $2/(1-\beta_2)$ steps if you follow the variance argument. It scales with $\beta_2$ (longer averaging window needs longer warmup) and with batch size (bigger batches mean fewer steps for the same tokens, so you need proportionally more warmup as a fraction). It does *not* scale much with model size, though very large models are more fragile so people err long. Lowering $\beta_2$ to 0.95 — common in LLM pretraining — is partly a way to shorten the required warmup and react faster to distribution shifts in the data.

### Q: Walk me through learning-rate schedules. Why cosine, and what's replacing it?

**Answer.** The standard recipe is linear warmup to a peak, then cosine decay to some floor:

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right).$$

Typically decaying to 10% of peak, over a horizon $T$ set to the planned total steps.

Why decay at all: early in training you want large steps to traverse the landscape quickly; late in training you want small steps to settle into a minimum rather than bouncing around it. The SGD convergence theory requires $\sum\eta_t = \infty$ and $\sum\eta_t^2 < \infty$, which forces decay. More practically, the gradient noise floor means that at a fixed LR you converge to a ball around the optimum whose radius scales with $\eta$; shrinking $\eta$ shrinks the ball.

Why *cosine* specifically over linear or step decay: it decays slowly at first (keeping the high-LR exploration phase long), then quickly through the middle, then slowly again at the end (a long fine-tuning tail). Empirically it beats step decay and slightly beats linear on most LLM runs. There's no strong theory; it's a well-shaped curve that people validated repeatedly.

The big practical problem with cosine: **you must know $T$ in advance.** If you stop early, you're at a high LR and your loss is much worse than a run that was scheduled to end there. If you want to continue training past $T$, there's no clean way to resume. That's a serious constraint when you're not sure how many tokens you'll train on.

What's replacing it: **warmup-stable-decay** (WSD, also called trapezoidal), used by MiniCPM, DeepSeek, and others. Warm up, hold at a constant peak LR for the bulk of training, then decay sharply over the last ~10% of steps. This matches cosine's final loss, and it has two big advantages — you can take a checkpoint from the stable phase at *any* point and anneal it briefly to get a fully-trained model, so you get a family of models from one run; and you can extend training indefinitely because the stable phase has no horizon. There's also a striking empirical finding that the loss drops sharply during the decay phase, which is still not well explained but is very reproducible. Related, **schedule-free** optimizers (Defazio et al.) get comparable results with no schedule at all by using an averaging scheme, and won a recent optimization benchmark.

So: cosine is still the default, WSD is the direction the field is moving, and if you're running a training job where the end date is uncertain, WSD is the right choice today.

**Follow-up: "Why decay to 10% rather than to zero?"** → Partly superstition, partly the observation that the last bit of decay contributes little and a nonzero floor keeps some ability to escape if the model is in a bad spot. In practice models that decay fully to zero do fine, and Chinchilla-style runs decayed to 10× below peak. The more important finding is that the *final* LR value matters much less than making sure you actually completed the decay before evaluating — comparing a mid-cosine checkpoint against a fully-decayed one is a common and badly misleading mistake.

### Q: Explain the relationship between batch size and learning rate. What's the linear scaling rule and when does it break?

**Answer.** The intuition: the gradient of a minibatch of size $B$ is an average of $B$ per-example gradients, so its variance scales like $1/B$. Doubling $B$ halves the noise, which means you can afford a larger step without the noise dominating.

**Linear scaling rule** (Goyal et al., the "1 hour ImageNet" paper): multiply LR by $k$ when you multiply batch size by $k$. The justification is that $k$ SGD steps with batch $B$ and LR $\eta$ approximately equal one step with batch $kB$ and LR $k\eta$ — *provided the gradient doesn't change much across those $k$ steps*. That proviso is exactly where it breaks. They combined it with a warmup precisely because at initialization the weights move fast and the assumption fails.

**Square-root scaling** is the alternative, and it's the right one for adaptive optimizers. The argument is that you want to hold the gradient *noise scale* constant: the SNR of the update goes like $\sqrt{B}$, so LR should go like $\sqrt{B}$. Empirically, SGD tends to follow linear and Adam tends to follow square-root, though this is fuzzy and the correct answer depends on regime. Recent muP-flavored work suggests the right framing is that there's an optimal LR that rises with $B$ and saturates.

Where it breaks generally: past a **critical batch size**, increasing $B$ stops buying you anything. McCandlish et al.'s gradient-noise-scale paper formalizes this — define $B_{\text{crit}} \approx \text{tr}(\Sigma)/(g^\top H g / |g|^2 \cdot \ldots)$, roughly the ratio of gradient variance to gradient magnitude. Below $B_{\text{crit}}$, doubling the batch roughly halves the number of steps needed (perfect scaling, pure time win). Above it, doubling the batch barely reduces step count and you're just burning compute. Importantly, $B_{\text{crit}}$ *grows during training* — as the loss falls, the gradient shrinks relative to its noise, so larger batches become useful later. That's the justification for batch-size ramps, which several frontier training runs use.

The second failure mode is generalization. Keskar et al.'s large-batch generalization gap — large batches converge to "sharp" minima that generalize worse. This is real but partly an artifact of holding the number of *epochs* rather than *steps* fixed; with LR scaling and enough steps, much of the gap closes. For single-epoch LLM pretraining it's not really a concern.

**Follow-up: "You have 8 GPUs and want an effective batch of 1M tokens but can only fit 8k per GPU. What do you do and what changes?"** → Gradient accumulation: run 16 micro-batches per GPU, accumulate gradients, then step. The LR should be set for the *effective* batch of 1M, not the micro-batch. Things that break if you're careless: BatchNorm statistics would be computed per micro-batch (a non-issue for transformers since they use LayerNorm); the loss must be scaled by $1/N_{\text{accum}}$ or you'll effectively multiply your LR by $N$; and if sequences have different token counts you need to normalize by total tokens, not by micro-batch count, or short micro-batches get overweighted. Also disable gradient sync on all but the last micro-step (`no_sync` in DDP) or you pay $N$× the communication.

*Trap:* Saying "bigger batch always trains faster." It reduces *steps*, not *compute*. Past the critical batch size you're paying linearly more FLOPs for sublinear step reduction — it's a wall-clock win only if you have idle parallel hardware.

### Q: Explain mixed-precision training. Why is loss scaling needed, and is it still needed with bf16?

**Answer.** Mixed precision means storing and computing most things in 16 bits while keeping a master copy of weights and the optimizer state in fp32. The wins are roughly 2× memory for activations, 2× memory bandwidth, and access to tensor cores which are several times faster on 16-bit matmuls.

The formats matter. **fp16** is 1 sign / 5 exponent / 10 mantissa — good precision, terrible range: max ~65504, and the smallest normal is ~$6\times10^{-5}$. **bf16** is 1 / 8 / 7 — it has the *same exponent range as fp32* but only 7 mantissa bits, so much worse precision, far better range.

**Loss scaling** exists because of fp16's range problem, specifically on the *gradient* side. Activation gradients in a deep network are frequently in the $10^{-8}$ to $10^{-10}$ range. In fp16 those underflow to zero — the gradient just disappears. The fix: multiply the loss by a large constant $S$ before calling backward. By linearity, every gradient in the graph is scaled by $S$, moving them into fp16's representable range. Then, before the optimizer step, divide the gradients by $S$ (in fp32) and proceed. Nothing about the math changes; you've just shifted the exponent.

In practice you use **dynamic loss scaling**: start with a large $S$ (say $2^{16}$), and if any gradient becomes inf or NaN, skip that step and halve $S$; if you go $N$ steps (say 2000) with no overflow, double $S$. This tracks the changing gradient magnitude over training automatically.

**Is it still needed with bf16?** No — and that's the main reason bf16 took over for large-model training. bf16's exponent range matches fp32, so gradients that would underflow in fp16 are representable, and activations that would overflow are too. You just don't need the machinery. The price is 7 mantissa bits (about 2-3 decimal digits), which turns out to be fine for the forward/backward matmuls because the tensor cores accumulate in fp32 anyway, and because gradient noise dominates rounding noise.

What still must be fp32 regardless of format: the master weights (a 16-bit weight can't represent a tiny update — if the update is more than ~$2^{-11}$ smaller than the weight, adding it is a no-op and training stalls), the optimizer moments, the softmax and its normalization sums, the loss reduction, and typically the LayerNorm statistics. That's the "mixed" part.

fp8 training is now real (H100/Blackwell) for the matmuls specifically, with per-tensor or per-block scaling factors — DeepSeek-V3 trained largely in fp8. That reintroduces range problems and hence scaling machinery, so in a sense loss scaling has come back one level down.

**Follow-up: "Your fp16 run has loss going to NaN around step 4000. Walk me through debugging."** → First check whether the loss scaler is thrashing — if $S$ has collapsed to something tiny and you're skipping most steps, you have persistent overflow, not a scaler-tuning problem. Then find *where*: register hooks and log per-tensor max magnitude, or bisect by casting suspect modules to fp32. The usual culprits are the attention logits (large before softmax — fix with QK-norm or fp32 softmax), the LayerNorm variance (fp32 it), and the final logits over a large vocab. Check the data loader for a corrupted batch by re-running the exact step. If it reproduces at exactly the same step, it's data or a determinism bug; if it moves, it's numerical. The pragmatic fix nine times out of ten is switch to bf16.

### Q: You're 40,000 steps into a pretraining run and the loss spikes. What's happening and what do you do?

**Answer.** Loss spikes are the characteristic failure mode of large-scale pretraining and essentially every large run has them. First distinguish two shapes: a spike that recovers on its own over a few hundred steps (annoying but survivable), versus one that plateaus at a high value or goes to NaN (the run is dead, the model has entered a degenerate state).

Diagnosis, in the order I'd actually check:

**Data.** The single most common cause. A batch containing a pathological document — repeated tokens, a base64 blob, a corrupted encoding, a huge run of whitespace — produces an enormous gradient. Test: reload the exact batch at the spike step and inspect it. If the spike reproduces deterministically at that step across restarts, it's data.

**Attention logit growth.** The well-documented mechanism (PaLM, and the "attention entropy collapse" work): attention logits grow slowly through training until the softmax saturates, entropy collapses to near-zero, the head becomes a hard argmax, gradients vanish through it, and the model destabilizes. Diagnostic: log max attention logit and attention entropy per layer. Fix: QK-norm, which bounds the logits by construction and has become standard for exactly this reason.

**Optimizer state staleness.** If the gradient distribution shifts (new data domain, a curriculum boundary), Adam's $\hat v$ is estimated from the old distribution, the denominator is wrong, and you get a huge step. Lower $\beta_2$ to 0.95 makes this less likely.

**Numerics.** Covered above — fp16 overflow, an underflowing $\epsilon$.

**Immediate remediation**, which is what interviewers actually want to hear: rewind to a checkpoint 100–500 steps *before* the spike, skip or reshuffle the data batches in that window, and resume. PaLM did exactly this and reported that resuming with different data skipped the spike, which is strong evidence the spikes were data-triggered rather than a fundamental instability at that point in the loss landscape. If it happens repeatedly, then make a structural change: lower the peak LR, add or tighten gradient clipping, add QK-norm, raise Adam $\epsilon$, or check for a hardware fault (a flaky GPU producing silent data corruption is real and shows up exactly like this).

The thing I'd emphasize is that you should have the *instrumentation* to answer this before it happens: gradient norm, per-layer gradient norms, attention entropy, activation max, loss-scale value, and a way to reproduce any batch by step index. Debugging a spike without those is guesswork.

**Follow-up: "Why do larger models spike more?"** → Several compounding reasons: more layers means more opportunity for one bad component to poison the residual stream; larger batches mean any single pathological document has been diluted less than you'd hope because the gradient is heavy-tailed; longer training means more chances; and low-precision numerics have less headroom at large activation magnitudes. There's also evidence that at scale the loss landscape has sharper regions the trajectory passes near. Empirically the frequency of spikes rises with parameter count, which is why the stabilization tricks (QK-norm, z-loss on the logits, careful init scaling) are essentially mandatory above ~10B.

### Q: Why do very deep networks train at all? Naively they shouldn't.

**Answer.** The naive worry is right: a plain 56-layer CNN has strictly *higher training error* than a 20-layer one, which He et al. pointed out is absurd — the deeper net can express the shallower one by making the extra layers identity, so it should do at least as well. That it doesn't is an optimization failure, not a capacity failure. That observation is the entire motivation for ResNets.

Why residual connections fix it, at three levels:

**Gradient flow.** $\partial x_{l+1}/\partial x_l = I + \partial F_l/\partial x_l$, so the backward product $\prod_l (I + J_l)$ expands to $I + \sum_l J_l + \sum_{l<m}J_lJ_m + \ldots$. That leading $I$ is an unattenuated path from loss to every layer. Depth no longer multiplies gradients down to nothing.

**Identity is now the easy default.** In a plain net, expressing identity requires the layer to learn a specific weight matrix. In a residual net, identity is what you get when $F_l = 0$ — and $F_l \approx 0$ is where initialization puts you. So the network starts as approximately the identity function and layers learn *deviations* from it. Adding a layer can't hurt, because "do nothing" is free.

**Ensemble / effective depth.** Veit et al. argued residual networks behave like an ensemble of exponentially many paths of varying length, and that gradients are dominated by the *short* paths — in a 110-layer ResNet, the effective depth contributing gradient is more like 10-30. So you're not really training a 110-layer network in the pathological sense; you're training a large collection of shallow ones that share weights.

Two other pieces matter. **Normalization** keeps per-layer Jacobians near unit scale and, in combination with residuals, makes the whole thing scale-stable. **Overparameterization** is doing something real too — the NTK/lazy-training line of work shows that in sufficiently wide networks the loss surface near initialization is close to convex in function space, so gradient descent reliably finds a global minimum. The width-depth interaction is why wide-and-deep works and narrow-and-deep doesn't.

For transformers specifically, add to this: they're pre-norm (see the earlier question), and initialization is often scaled by depth — e.g. scaling residual-branch output projections by $1/\sqrt{2L}$ (GPT-2) so that the residual stream's variance doesn't grow linearly with depth. Without that, deep transformers start with a residual stream so large that individual layers can't influence it.

*Trap:* Attributing it entirely to "skip connections give gradient highways." True but incomplete — the more important framing is that residuals change what the *optimization problem* is, making identity the default and layers learn perturbations.

### Q: What is gradient accumulation and what does it cost you?

**Answer.** Gradient accumulation lets you simulate a large batch on hardware that can't hold one. You run $N$ forward/backward passes on micro-batches of size $b$, summing gradients into the `.grad` buffers without stepping, then take one optimizer step and zero the gradients. Effective batch size is $N \cdot b \cdot (\text{data parallel degree})$.

Mathematically it is *exactly* equivalent to a batch of size $Nb$ — same gradient, provided you handle the loss normalization correctly. Which is the thing that goes wrong: if your loss is a mean over the micro-batch, you must divide each micro-batch loss by $N$ before backward, or your accumulated gradient is $N$× too large and you've silently multiplied your learning rate by $N$. And if micro-batches have different numbers of non-padding tokens, dividing by $N$ is *still* wrong — you need to weight by token count, i.e. sum the per-token losses across all micro-batches and divide by the total token count. This bug was widespread enough that it was found and publicly fixed in several popular fine-tuning libraries in late 2024, where it was producing measurably worse models. Worth knowing.

What it costs:

**Wall-clock.** Nothing is parallelized across micro-batches — you're serializing what a bigger machine would do at once. You get the *statistical* benefit of a large batch but none of the *speed* benefit, since the whole point of large batches on real hardware is more parallel work per step.

**Communication, if done naively.** In DDP, gradients all-reduce at every backward by default, so $N$ accumulation steps means $N$ all-reduces for one update. Wrap all but the last micro-step in `model.no_sync()` and you do one. This is a large real speedup, not a micro-optimization.

**Nothing on memory beyond one micro-batch**, which is the point — activations are freed after each micro-backward. The gradient buffer is full-model-size regardless.

Note it composes with, and is distinct from, gradient *checkpointing* (recompute activations instead of storing them, trading ~30% extra compute for large activation-memory savings). People confuse the names; accumulation is about batch size, checkpointing is about activation memory. They're commonly used together.

**Follow-up: "Does it interact badly with anything?"** → BatchNorm (statistics computed per micro-batch, so it's genuinely not equivalent — a reason transformers' use of LayerNorm is convenient). Anything with a per-step schedule needs to be stepped per *optimizer* step, not per micro-batch, which includes the LR scheduler and any EMA. And gradient clipping must be applied to the *accumulated* gradient right before the step, not per micro-batch — clipping each micro-gradient separately changes the result and is a common bug.
---

## 3. Generalization and Evaluation

### Q: State the bias-variance decomposition precisely. Derive it.

**Answer.** Setup: true relationship $y = f(x) + \epsilon$ with $\mathbb{E}[\epsilon]=0$, $\text{Var}(\epsilon)=\sigma^2$. We fit $\hat f$ on a random training set $D$; the randomness is over $D$. Fix a test point $x_0$ and ask for the expected squared error.

$$\mathbb{E}_{D,\epsilon}\left[(y - \hat f_D(x_0))^2\right].$$

Write $\bar f(x_0) = \mathbb{E}_D[\hat f_D(x_0)]$, the average prediction over training sets. Insert and subtract:

$$y - \hat f_D = \underbrace{\epsilon}_{\text{noise}} + \underbrace{(f - \bar f)}_{\text{bias}} + \underbrace{(\bar f - \hat f_D)}_{\text{variance}}.$$

Square and take expectations. All three cross terms vanish: $\epsilon$ is independent of $D$ and mean-zero, so any cross term with $\epsilon$ is zero; and $\mathbb{E}_D[\bar f - \hat f_D] = 0$ by definition of $\bar f$, while $(f - \bar f)$ is a constant with respect to $D$, so that cross term vanishes too. Left with

$$\mathbb{E}[(y - \hat f)^2] = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{(f(x_0) - \bar f(x_0))^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D[(\hat f_D(x_0) - \bar f(x_0))^2]}_{\text{variance}}.$$

Interpretation: **bias** is how far the average model is from the truth — systematic error from the hypothesis class being too restrictive. **Variance** is how much the model moves when you resample the training data — sensitivity to the particular sample. **Irreducible error** is the noise floor; no model beats it.

Three caveats I'd volunteer, because they're where the follow-ups go. First, this decomposition is specific to **squared loss**. For 0–1 loss there is no clean additive decomposition — Domingos and others have proposed versions, but the terms interact multiplicatively and bias can *reduce* error in some regimes. For cross-entropy there's a related decomposition via the Bregman divergence framing, but it isn't the familiar three-term one. Second, the expectation is over training sets, which is a thought experiment — you have one training set, so you can't measure these directly without bootstrapping. Third, the classic U-shaped "bias-variance tradeoff" curve is *not* a theorem; it's an empirical pattern for classical models, and double descent shows it fails for overparameterized ones.

**Follow-up: "Give me the canonical high-bias and high-variance models and how to fix each."** → High bias: linear regression on nonlinear data, or a depth-1 tree. Fix by adding capacity — more features, interactions, a bigger model, less regularization. Symptom is high training error *and* high test error, close together. High variance: an unpruned decision tree, or 1-NN. Fix with more data, regularization, bagging, or feature reduction. Symptom is near-zero training error with a big gap to test error. The diagnostic is the learning curve: if train and validation error converge to a high value, you're bias-limited and more data won't help; if there's a persistent gap that's still narrowing, you're variance-limited and more data will.

### Q: What is double descent and why doesn't it break the bias-variance tradeoff?

**Answer.** Plot test error against model capacity. The classical picture is a U: error falls as you reduce bias, then rises as variance takes over, with the minimum at some intermediate complexity. Double descent says: keep going. Past the **interpolation threshold** — where the model has just enough capacity to fit the training data exactly, roughly $\text{params} \approx \text{examples}$ — test error *peaks*, and then it *falls again*, often below the classical minimum.

So the curve is: descend, ascend to a peak at the interpolation threshold, descend again into the overparameterized regime. Belkin et al. (2019) named it; it appears in linear regression, random features, random forests, and deep nets alike, so it's not a deep-learning quirk.

Why the peak: right at the interpolation threshold there is essentially *one* set of parameters that fits the data exactly, and the model has no freedom to choose a nice one. That unique solution is wildly sensitive to noise — variance blows up. Just below the threshold you can't fit the data but you're stable; just above, you can fit it but only barely, in the most contorted way possible.

Why the second descent: once you're comfortably overparameterized, there are *infinitely many* interpolating solutions, and now the choice among them matters. Gradient descent has an implicit bias toward low-norm / low-complexity solutions — for linear models trained with GD from zero, you provably converge to the minimum-$\ell_2$-norm interpolator. So more capacity gives the optimizer more room to pick a *smoother* interpolator, and smoothness generalizes.

The key reconciliation: it doesn't break the tradeoff, it breaks the assumption that **parameter count is the right measure of complexity**. The tradeoff is a statement about effective complexity versus error, and it's fine. What's wrong is the mapping from "number of parameters" to "effective complexity" — in the overparameterized regime, adding parameters with an implicitly-regularizing optimizer *decreases* effective complexity, because it lets you find a lower-norm function. If you plotted against the norm of the learned function instead of parameter count, you'd see something much closer to the classical U.

There's also **epoch-wise** double descent (test error goes down, up, down as you train longer at fixed size) and **sample-wise non-monotonicity** (more data can temporarily *hurt*, because it moves the interpolation threshold to where your model size sits). Nakkiran et al. mapped all three and showed that sufficient explicit regularization can remove the peak entirely — which is a useful practical takeaway: if you're near the threshold and seeing weird behavior, either get much bigger or regularize harder, but don't sit at the peak.

*Trap:* Saying double descent means "overfitting doesn't exist for big models." It exists; it's just that the classical capacity-based intuition for *when* it happens is wrong. A big model trained many epochs on tiny data still memorizes and generalizes badly.

### Q: Compare regularization methods by mechanism — L2, dropout, early stopping, data augmentation.

**Answer.** They're all "reduce effective capacity," but the mechanisms are genuinely different and it's worth being precise.

**L2 / weight decay.** Adds $\lambda\|w\|^2$ to the loss, so the gradient gets $2\lambda w$ and weights shrink toward zero each step. Mechanism: it's a Gaussian prior on the weights — MAP estimation with $w \sim \mathcal{N}(0, \tau^2 I)$ gives exactly the L2 penalty with $\lambda = \sigma^2/(2\tau^2)$. Geometrically, in the eigenbasis of the Hessian it shrinks the component along eigendirection $i$ by $\lambda_i/(\lambda_i + \alpha)$ — so directions the loss doesn't care about (small $\lambda_i$) get crushed, directions it does care about are barely touched. That's the cleanest way to say what it does. Contrast with **L1**, which produces *sparsity*: the constraint region is a diamond with corners on the axes, so the optimum tends to land on a corner where some coordinates are exactly zero. L2's spherical constraint has no corners, so it shrinks but never zeroes.

**Dropout.** Randomly zero each unit with probability $p$ during training, scale by $1/(1-p)$ so expectations match at inference. Three mechanisms are usually cited and all have support: (a) it's an approximate *ensemble* over $2^n$ sub-networks with shared weights, and inference with the full net approximates the geometric-mean prediction of the ensemble; (b) it prevents **co-adaptation** — a unit can't rely on a specific other unit being present, so features must be individually useful, which is the mechanism Hinton emphasized; (c) it's equivalent to a form of adaptive noise injection, and for linear regression Gaussian dropout is *exactly* equivalent to L2 on a rescaled parameterization. Worth noting dropout has largely fallen out of use in LLM pretraining — with one epoch over trillions of tokens you're not overfitting, and dropout just slows convergence. It's still used in fine-tuning and in small-data regimes.

**Early stopping.** Stop when validation loss stops improving. Mechanism: it limits how far the parameters can travel from initialization. For a quadratic loss with gradient descent, stopping after $t$ steps gives you a shrinkage factor of $(1 - (1-\eta\lambda_i)^t)$ along eigendirection $i$ — which is *the same shape* as L2's shrinkage, with $t \approx 1/(\eta\alpha)$. So early stopping is provably equivalent to L2 in the quadratic case. It's the cheapest regularizer (it saves compute rather than costing it) and it's adaptive, but it couples regularization strength to your optimization schedule, which is untidy.

**Data augmentation.** Expands the training set with label-preserving transformations. Mechanism is different in kind from the other three: instead of restricting the hypothesis class, it *injects prior knowledge about invariances* into the data. Telling the model "a rotated cat is a cat" is information the other three cannot supply. That's why it's usually the highest-leverage regularizer when you know the right invariances, and why it's much harder in NLP — most text transformations aren't reliably meaning-preserving. In the limit, augmentation approximates training on the true data distribution rather than shrinking capacity.

**Follow-up: "Rank them for a 7B LLM being fine-tuned on 10k examples."** → Data-side interventions first: more and better data, and if you can't get it, careful synthetic augmentation. Then early stopping — with 10k examples you'll overfit within 2-3 epochs and the validation curve will tell you exactly when. Then weight decay, modestly. Then LoRA itself, which is a strong implicit regularizer because it caps the rank of the update — that's often the single most effective knob. Dropout last; it's usually already at 0 and turning it on rarely helps much in this regime. I'd also add label smoothing to that list, which is arguably more useful than dropout here.

### Q: What does overfitting look like in an LLM, given that pretraining is typically single-epoch?

**Answer.** It looks different at each stage, and that's the interesting part.

**Pretraining**, one epoch over trillions of tokens: classical overfitting basically doesn't occur, because you never see an example twice — training loss and held-out loss track each other almost exactly. What you get instead is **memorization**: the model verbatim-reproduces sequences that appeared in training, especially ones that appeared many times (duplicated documents, license texts, common code snippets, and — the thing that matters — PII and copyrighted text). Carlini et al. showed extraction is possible and scales with model size, duplication count, and prompt length. That's overfitting in the sense of memorizing individual examples, but it coexists with excellent generalization, which is why the classical framing doesn't transfer. The measurement is extraction rate or n-gram overlap with the training corpus, not a train/val gap. This is a big part of why deduplication is a standard pretraining step.

If you do multiple epochs, the picture changes: Muennighoff et al.'s "Scaling Data-Constrained Language Models" found up to ~4 epochs is nearly as good as fresh data, and beyond ~16 epochs the returns go to zero and you start seeing real degradation. So the epoch threshold is known and it's higher than people assume.

**Fine-tuning** is where you see textbook overfitting, and fast — a few thousand examples and a 7B model will fit them in 2-3 epochs. Symptoms: validation loss turns up sharply while training loss keeps falling; the model starts reproducing training-set phrasing verbatim; it collapses onto stylistic tics of the fine-tuning data (the "as an AI language model" pattern is exactly this); and **catastrophic forgetting** — capabilities not represented in the fine-tuning set degrade, which a validation set drawn from the same distribution as the fine-tuning data will completely fail to detect. That last point is the one worth making: you need *out-of-distribution* evals (a general benchmark suite) alongside your in-distribution validation loss, or you'll ship a model that's better at your task and worse at everything else.

**RLHF** has its own version: **reward hacking**, which is overfitting to the reward model rather than to data. The policy finds inputs where the reward model is wrong — excessive length, sycophancy, formatting patterns the RM likes — and exploits them. True quality peaks and then declines while measured reward keeps climbing. The KL penalty to the reference model is the main defense.

**Follow-up: "How would you actually detect pretraining contamination of a benchmark?"** → Several complementary methods. N-gram overlap between benchmark items and the training corpus (13-gram is a common threshold) — the direct method, requires corpus access. Comparing loss on the benchmark's canonical ordering versus a shuffled ordering of the same items — a contaminated model has memorized the specific order and shows lower perplexity on the original. Comparing performance on a benchmark against a freshly-authored version of the same task (the GSM1k approach), where a large gap indicates contamination. And checking whether the model can complete a benchmark item given only its first half. None is conclusive alone; the fresh-benchmark comparison is the most convincing.

### Q: Walk me through train/validation/test discipline. What's the failure mode people actually hit?

**Answer.** The three-way split exists because each set answers a different question. **Train** fits parameters. **Validation** selects hyperparameters and architecture and decides when to stop. **Test** estimates generalization *once*, at the end.

The reason test must be touched once: every time you look at test performance and change something in response, you're doing optimization against it, and its estimate of generalization becomes biased upward. This is the same phenomenon as multiple-hypothesis testing. If you evaluate 100 model variants on the test set and pick the best, the winner's test score is inflated by roughly the standard error times the expected maximum of 100 draws — on a 1000-example test set with 85% accuracy, standard error is about 1.1%, and the best-of-100 selection bias is on the order of 2-3 points. That's larger than most reported improvements in papers.

**The failure mode people actually hit** isn't the textbook one of literally training on test. It's *gradual* test-set contamination through iteration: you build a model, check test, it's 84%, you try something else, check test, 86%, ship. Nobody wrote a training loop over the test set, but the human in the loop performed the same optimization by hand over months. The community-scale version of this is why ImageNet and CIFAR-10 accuracy numbers were inflated — the Recht et al. "Do ImageNet Classifiers Generalize to ImageNet?" work built fresh test sets and found consistent accuracy drops of 3-15 points, though notably the *ranking* of models was preserved, which suggests adaptive overfitting was less severe than feared.

The other big failure mode is **splitting on the wrong unit**. Random row-level splits are wrong whenever rows are not independent: multiple records per user, multiple frames per video, multiple sentences per document, time series. If the same user appears in train and test, you're measuring memorization of that user. Split by the unit of generalization — the thing that will be new at inference time.

Practical additions I'd mention: for time series, split temporally and never randomly, because a random split lets the model see the future. For small datasets, use k-fold cross-validation for hyperparameter selection but *still* hold out a final test set — nested CV if you're being rigorous. Keep a "dev-test" set that you're allowed to look at repeatedly and a true holdout you touch once. And report a confidence interval on the test metric, so you don't chase differences inside the noise.

*Trap:* Using the validation set as the reported final number. It's been optimized against; it's not an unbiased estimate of anything.

### Q: What forms does data leakage take?

**Answer.** Leakage is any case where information available at training time would not be available at prediction time. It's the single most common cause of "great offline metrics, useless in production." The taxonomy:

**Target leakage.** A feature is a proxy for, or a consequence of, the label. Classic: predicting hospital readmission with a `discharge_medication` field that's only populated after the outcome is known. Or predicting churn with `cancellation_reason`. The tell is a single feature with implausibly high importance and near-perfect performance. The fix is a temporal audit: for every feature, ask "at the moment of prediction, would this value exist and would it have this value?"

**Train-test contamination via preprocessing.** Fitting a scaler, imputer, PCA, or vocabulary on the *full* dataset before splitting. The test set's statistics have leaked into the transformation. Subtle but real — normalizing with the global mean means test examples influenced their own preprocessing. The fix is `fit` on train only, `transform` on both, enforced by putting everything in a pipeline object so it can't be done wrong.

**Temporal leakage.** Random splitting on time-ordered data, so the model trains on the future and predicts the past. Also feature-level: computing a rolling aggregate with a window that includes the target period, or joining a dimension table that reflects its *current* state rather than its state at event time. Slowly-changing-dimension handling is exactly this problem.

**Group leakage.** Splitting rows when the natural unit is a group — the same patient's scans in train and test, the same author's documents, augmented copies of the same image. You measure recognition of the group, not generalization.

**Duplicate leakage.** Near-duplicates spanning the split. Extremely common in scraped web data and image datasets. Requires explicit near-dup detection (MinHash/LSH for text, perceptual hashing for images), not exact matching.

**Benchmark contamination**, the LLM-era version: the eval set was in the pretraining corpus. Covered above.

**Leakage through the label pipeline.** If labels were generated by a heuristic that uses a feature you also feed the model, the model learns the heuristic, not the phenomenon.

How I'd catch these in practice: be suspicious of any result that's much better than you expected — that's the strongest signal. Check feature importances for a single dominant feature. Check performance degradation over time in backtesting; leakage often shows as a suspiciously flat curve. And ask, feature by feature, the counterfactual question about what's knowable at prediction time. That last one catches most of it and requires no tooling.

### Q: How do you handle class imbalance? Walk me through your decision process.

**Answer.** First question I'd ask: is imbalance actually the problem? Imbalance per se isn't a pathology — if you have 1% positives and 10 million examples, you have 100,000 positives, which is plenty, and a well-calibrated model will do fine. The problems arise from (a) *absolute* scarcity of the minority class, (b) an evaluation metric that ignores the minority class, and (c) a decision threshold picked without regard to the cost asymmetry. Those need different fixes.

**Fix the metric first.** Accuracy on a 99/1 split is worthless — predict-all-negative gets 99%. Use precision/recall on the minority class, and use **AUPRC** rather than AUROC as the summary, because AUROC's false-positive rate has the large negative class in the denominator so it's insensitive to exactly the errors you care about. Whatever you use, the metric should reflect the actual costs.

**Fix the threshold, not the model.** This is the most underrated move. Train normally with proper cross-entropy, get well-calibrated probabilities, then choose the operating threshold to optimize your actual objective — expected cost, or recall subject to a precision floor. Most "imbalance problems" are threshold problems in disguise. A model that outputs 0.03 for positives and 0.001 for negatives is a fine model; you just shouldn't threshold at 0.5.

**Then, if you still need it, resampling or reweighting.** Class weights in the loss, $w_c \propto 1/n_c$, are the simplest and don't change the data. Oversampling the minority (duplicating, or SMOTE which interpolates between minority neighbors) works but risks overfitting the minority and, importantly, **destroys calibration** — you've changed the prior, so predicted probabilities no longer match reality and you need to correct them back. Undersampling the majority discards data, which is fine when you have millions of negatives and wasteful otherwise. **Focal loss**, $-(1-p_t)^\gamma \log p_t$, down-weights easy examples so the gradient concentrates on hard ones; it came from dense object detection where the imbalance is ~1000:1 and it's the right tool for extreme cases.

For genuinely extreme imbalance — fraud at $10^{-5}$ — I'd consider reframing as **anomaly detection** and modeling only the majority class, since there may not be enough positives to characterize a "positive class" at all.

**Follow-up: "You reweighted and your probabilities are now miscalibrated. Fix it."** → If you oversampled the minority by factor $k$ (or equivalently reweighted), the model estimates $p'$ under the resampled prior. Convert back with the prior-correction formula: $p = \frac{p'/k}{p'/k + (1-p')}$, generalizing to $p = \frac{p' \pi/\pi'}{p'\pi/\pi' + (1-p')(1-\pi)/(1-\pi')}$ for true prior $\pi$ and training prior $\pi'$. Or just fit Platt scaling / isotonic regression on a validation set drawn from the *true* distribution, which handles it empirically and fixes other miscalibration at the same time. The second is what I'd actually do.

*Trap:* Applying SMOTE before the train/test split. You then have synthetic points in test that were interpolated from train points — direct leakage, and it inflates results dramatically. SMOTE goes inside the CV loop, on the training fold only.

### Q: Precision, recall, F1, AUROC, AUPRC. When is each the right metric?

**Answer.** Definitions first, then the decision rule.

Precision $= TP/(TP+FP)$: of the things I flagged, what fraction were right. Recall (sensitivity, TPR) $= TP/(TP+FN)$: of the things that were positive, what fraction did I catch. F1 is their harmonic mean, $2PR/(P+R)$ — harmonic because it penalizes imbalance between them; you can't get a good F1 by maxing one and tanking the other.

Precision and recall trade off through the threshold. Lower the threshold, catch more positives (recall up), flag more junk (precision down).

**Which to optimize is a question about costs, not about statistics.** If a false negative is expensive and a false positive is cheap — cancer screening, where a missed tumor is fatal and a false alarm is a follow-up scan — optimize recall, possibly with a precision floor. If a false positive is expensive — an automated account ban, a spam filter that eats legitimate mail — optimize precision. If they're comparable, F1. If they're comparable but not equal, use $F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$, where $\beta > 1$ weights recall; $\beta=2$ says recall is twice as important.

**AUROC** integrates TPR against FPR over all thresholds. Its clean interpretation: the probability that a randomly chosen positive gets a higher score than a randomly chosen negative. It's threshold-free, so it measures *ranking* quality, and it's invariant to the class prior — which is both its strength and its trap.

**AUPRC** integrates precision against recall. Not prior-invariant: the baseline for a random classifier is the positive rate $\pi$, whereas AUROC's baseline is always 0.5.

**The decision rule for AUROC vs AUPRC:** use AUPRC when positives are rare *and* you care about performance on positives. The reason is in the denominators. FPR $= FP/(FP+TN)$ has the huge negative count in the denominator, so going from 100 to 1000 false positives barely moves FPR when you have a million negatives — AUROC will look excellent while your precision is 10%. Precision $= TP/(TP+FP)$ has no $TN$ term at all, so it registers that change fully. That's the whole story: AUROC is insensitive to false positives when negatives dominate; AUPRC isn't.

Use AUROC when classes are roughly balanced, when you want a prior-invariant measure to compare across datasets with different base rates, or when you genuinely care about both classes symmetrically.

One more thing I'd add: for a deployed system, none of these is really the metric. The metric is expected utility at your chosen operating point — dollars, lives, latency. The threshold-free summaries are for model *selection*; the deployed number should be cost at threshold.

**Follow-up: "Model A has AUROC 0.95 / AUPRC 0.30, Model B has 0.92 / 0.45. Which ships?"** → B, almost certainly, if positives are rare and you care about them — which the AUPRC gap implies. B ranks positives better in the high-precision region where you'd actually operate; A's AUROC advantage is coming from correctly ordering things deep in the negative bulk, which doesn't affect any threshold you'd deploy. I'd confirm by plotting both PR curves and looking specifically at the recall range you'd operate in, since a single AUPRC number can hide a curve that's better at high recall and worse at low.

### Q: What is calibration, why doesn't high accuracy imply it, and how do you fix it?

**Answer.** A model is calibrated if its stated probabilities are empirically correct: among all predictions with confidence 0.7, exactly 70% are right. Formally, $P(Y=1 \mid \hat p = p) = p$ for all $p$.

**Accuracy and calibration are orthogonal**, and the cleanest way to see it is with two examples. Take a balanced binary problem where the truth is 60/40 and features are uninformative beyond that. A model that always outputs 0.6 for the majority class is *perfectly calibrated* and has 60% accuracy. A model that outputs 0.99 whenever it's right and 0.99 whenever it's wrong can have 90% accuracy and be badly miscalibrated. Accuracy only depends on which side of the threshold you land; calibration depends on the *value* of the probability. Any monotone transformation of your scores leaves accuracy and AUROC unchanged while arbitrarily destroying calibration.

**Modern networks are systematically overconfident.** Guo et al. (2017) documented that as networks got deeper and more accurate, they got *worse* calibrated — LeNet was well calibrated, ResNet was badly overconfident. The causes: training to near-zero training loss pushes softmax outputs toward one-hot, since cross-entropy keeps rewarding higher confidence on already-correct examples with no counterweight; and capacity plus lack of regularization lets the model fit the training labels with certainty they don't deserve.

**Measurement.** Expected Calibration Error: bin predictions by confidence into $M$ bins, and compute $\text{ECE} = \sum_m \frac{|B_m|}{n}\left|\text{acc}(B_m) - \text{conf}(B_m)\right|$. Plot a reliability diagram — accuracy versus confidence per bin — where the diagonal is perfect. ECE is bin-count-sensitive and can be gamed, so I'd report the diagram alongside, and consider adaptive binning or a debiased estimator.

**Fixes.** **Temperature scaling** is the default and it's remarkably good: learn a single scalar $T$ on a validation set, output $\text{softmax}(z/T)$, minimizing NLL. $T>1$ softens. One parameter, so it can't overfit, and because it's monotone it *cannot change accuracy at all* — you get calibration for free. **Platt scaling** ($\sigma(az+b)$) is the two-parameter binary version. **Isotonic regression** is nonparametric and more flexible, so it can fix non-monotone miscalibration, but it needs more validation data and can overfit. **Label smoothing** and **mixup** improve calibration during training. **Ensembles** and MC-dropout give better-calibrated uncertainty for free as a side effect.

For LLMs specifically: base models are surprisingly well calibrated on multiple-choice, and RLHF *degrades* calibration substantially — the OpenAI GPT-4 system card showed exactly this, with the pre-RLHF model near the diagonal and the post-RLHF one confidently wrong. That's a real and underappreciated cost of preference tuning.

**Follow-up: "Temperature scaling on the val set — anything to watch for?"** → The validation set must come from the deployment distribution; calibration is distribution-specific and a temperature fit on in-domain data will be wrong under shift, usually still overconfident. It also only fixes *average* miscalibration — if the model is overconfident on one subgroup and underconfident on another, a single scalar can't fix both and ECE will look fine while both groups are wrong. Check calibration per subgroup. And it needs enough validation data to estimate reliably, though with one parameter that's a low bar — a few thousand examples is plenty.

### Q: What's a proper scoring rule and why should you care?

**Answer.** A scoring rule $S(p, y)$ scores a predicted distribution $p$ against an observed outcome $y$. It's **proper** if the expected score is optimized by reporting your true belief: if the true probability is $q$, then $\mathbb{E}_{y\sim q}[S(p,y)]$ is optimized at $p = q$. It's **strictly proper** if $p=q$ is the *unique* optimum.

Why you should care: a proper scoring rule is one you cannot game. If your loss is strictly proper, the only way to minimize it is to be honestly calibrated and sharp. If it isn't proper, the model has an incentive to lie, and it will.

The two you use: **log loss / cross-entropy**, $-\log p_y$, and **Brier score**, $\sum_k (p_k - y_k)^2$. Both strictly proper. Quick proof for log loss in the binary case: expected score is $-[q\log p + (1-q)\log(1-p)]$; differentiate with respect to $p$, get $-q/p + (1-q)/(1-p)$, set to zero, and $p = q$ falls out. Second derivative is positive, so it's a minimum, and it's unique.

The classic **improper** example is accuracy, and this is the practically important case. Suppose the true probability of an event is 0.7. To maximize expected accuracy you should predict "yes" with probability 1 — reporting 0.7 gets you an expected accuracy of $0.7\cdot0.7 + 0.3\cdot0.3 = 0.58$, while reporting 1.0 gets you 0.7. So accuracy *actively rewards overconfidence*. Any metric that's a function of thresholded predictions has this property. Mean absolute error on probabilities is likewise improper — it's optimized by reporting 0 or 1, whichever is more likely.

The practical consequence: if you train or select models by accuracy, you should expect miscalibration, and you shouldn't be surprised when the probabilities are useless downstream. If you need probabilities — for expected-value decisions, for cost-sensitive thresholding, for feeding into another system — you must train and select on a proper rule.

The other thing worth knowing is the **decomposition**. Brier score decomposes as reliability − resolution + uncertainty: reliability is calibration error (lower better), resolution is how much your predictions vary across cases (higher better), uncertainty is the irreducible base-rate variance. So a proper score rewards both being calibrated *and* being sharp. A model that always outputs the base rate is perfectly calibrated but has zero resolution and a mediocre Brier score — which is exactly right, because it's useless. That's the sense in which proper scoring rules capture what you actually want in one number.

**Follow-up: "Log loss or Brier?"** → Log loss is unbounded — a single confident-and-wrong prediction ($p \to 0$ on the true class) sends it to infinity, so it punishes catastrophic overconfidence extremely hard, which is usually what you want and is why it's the training loss. Brier is bounded in $[0,2]$ and therefore more robust to outliers and easier to interpret, and it decomposes cleanly. Log loss's gradient is also better behaved for training — it's what gives you the clean $(p - y)$ form. I'd train on log loss and report both, plus a reliability diagram.
---

## 4. LLM-Specific

### Q: What does next-token prediction actually teach a model, and what does it fail to teach?

**Answer.** The objective is to maximize $\sum_t \log P(x_t \mid x_{<t})$ — that's it, one loss, no task labels. What makes it powerful is that it's a *density estimation* objective on a corpus that happens to contain the residue of human reasoning. To predict the next token well on arbitrary text you have to model whatever generated that text. Predicting the last token of "the murderer was revealed to be the" requires having tracked the plot. Predicting the output of a code snippet requires simulating it. Ilya Sutskever's framing — compression implies understanding — is the strong version of this, and I'd say it's directionally right: the objective forces the model to build whatever internal machinery reduces surprise, and for human-generated text that machinery includes syntax, world knowledge, arithmetic, some theory of mind, and style.

Concretely it teaches: grammar and long-range syntactic structure, factual associations (Paris/France, encoded largely in mid-layer FFNs), in-context pattern completion including induction (the mechanism behind few-shot prompting), a compressed world model sufficient for plausible continuation, and the ability to *represent* many personas and registers, since the corpus contains all of them.

What it does *not* teach:

**Which behavior you want.** The model learns the distribution over all continuations, including unhelpful, false, and harmful ones. It has no notion that being helpful is preferred — it will happily continue a bad answer if the context suggests a bad answer is coming. That's the entire motivation for instruction tuning and preference optimization. Base models aren't misaligned so much as *unaimed*.

**Truth as distinct from plausibility.** The objective rewards matching the training distribution, and the training distribution contains falsehoods, contested claims, and fiction. Nothing in the loss distinguishes "this is what a document would say next" from "this is true." Hallucination is the objective working correctly.

**Calibration about its own knowledge.** The model is trained to always produce a next token, never to decline. There's no gradient signal that says "you don't know this." Expressions of uncertainty are learned as *style* from text where humans hedged, not from any internal uncertainty estimate.

**Planning and backtracking.** Generation is left-to-right and committed — the model can't revise token 3 after realizing at token 40 that it was wrong. Chain-of-thought is a partial workaround (externalize intermediate state into the context so later tokens can condition on it) but it doesn't give real search. This is a genuine architectural limitation of the sampling procedure, not just of the objective.

**Anything not in the data**, obviously — and also anything that humans do but don't *write down*, which is a lot of tacit skill.

**Follow-up: "There's an argument that next-token prediction can't reach human-level reasoning because errors compound. Is that right?"** → The argument (LeCun's version) is that if each token has error probability $\epsilon$ and errors are independent, the probability of a correct $n$-token sequence is $(1-\epsilon)^n$, which decays exponentially. The flaw is the independence assumption — the model conditions on what it already generated, so it can recover, and there's clear empirical evidence of self-correction within a generation. The stronger version of the critique is about *teacher forcing*: training always conditions on ground-truth prefixes, so the model never learns to recover from its own mistakes, which is a genuine distribution shift (exposure bias, the same problem DAgger addresses in imitation learning). That's a real limitation, and it's part of why RL on model-generated rollouts helps — it trains on the model's own distribution.

### Q: Why cross-entropy for language modeling? Why not something else?

**Answer.** Several independent justifications converge on it, which is a good sign.

**It's the MLE.** Maximizing the likelihood of the corpus under the model is $\max_\theta \prod_t P_\theta(x_t|x_{<t})$; take logs and negate, and you get exactly the cross-entropy loss. So it's the maximum-likelihood estimator, with all its asymptotic properties — consistency, and asymptotic efficiency under correct specification.

**It's KL divergence to the data distribution.** $H(p, q) = H(p) + D_{KL}(p \| q)$, and $H(p)$ is a constant with respect to your parameters. So minimizing cross-entropy is exactly minimizing $D_{KL}(p_{\text{data}} \| p_\theta)$. That direction of KL is *mode-covering* / zero-avoiding: it heavily penalizes assigning near-zero probability to anything that occurs, which for a general-purpose model is the behavior you want. (Reverse KL would be mode-seeking and would give you a model that produces a narrow slice of fluent text — which, incidentally, is roughly what heavy RLHF with a reverse-KL penalty does to diversity.)

**It's a strictly proper scoring rule**, so the optimum is the honest conditional distribution and the model has no incentive to distort probabilities.

**It has a clean information-theoretic reading.** The loss in nats is the average number of nats needed to encode the next token under the model. Divide by $\ln 2$ and it's bits-per-token — literally a compression rate. That's what makes perplexity meaningful and what grounds the "LLMs are compressors" framing.

**Its gradient is beautifully simple.** With softmax outputs, $\partial L/\partial z_i = p_i - y_i$. The logit gradient is just prediction minus target. No activation derivative appears — which is the key point in contrast with MSE.

Why not the alternatives: **MSE** on the softmax output has a vanishing-gradient problem (see the next question) and is not a natural fit for categorical data. **Ranking losses** only constrain relative order, so you get no usable probabilities and no compression interpretation. **Direct sequence-level objectives** like BLEU aren't differentiable and require RL. **Unlikelihood training** and contrastive variants are additive fixes for specific pathologies (repetition), not replacements.

The one genuine complaint about cross-entropy is that it's *token-level* and *equally weighted*: predicting a function's return value and predicting a comma contribute the same to the loss, and the model spends most of its capacity on the easy high-frequency mass. That's a real mismatch with what we care about, and it motivates things like selective language modeling (weighting tokens by learnability) — but nothing has displaced plain cross-entropy at scale.

### Q: Why not use MSE for classification?

**Answer.** Three reasons, and the gradient one is the sharpest.

**Gradients vanish exactly when you need them.** With a sigmoid output $p = \sigma(z)$ and MSE loss $L = (p-y)^2$, the chain rule gives $\partial L/\partial z = 2(p-y)\cdot\sigma'(z) = 2(p-y)p(1-p)$. Now consider a confidently wrong prediction: $y=1$, $p = 0.01$. The error term $(p-y)$ is large at $-0.99$, but $\sigma' = 0.01 \times 0.99 \approx 0.0099$, so the gradient is about $-0.0196$ — tiny. The model is maximally wrong and learns almost nothing. With cross-entropy, $\partial L/\partial z = p - y = -0.99$, fifty times larger. The $\sigma'$ factor cancels against the $1/p$ from the log. That cancellation is the whole reason cross-entropy is the right loss for a sigmoid or softmax output: it's designed so that the saturating nonlinearity's derivative disappears from the gradient.

**Non-convexity.** For logistic regression, MSE composed with the sigmoid is *not* convex in the weights — it has local minima and saddle points. Cross-entropy composed with the sigmoid *is* convex (proof in the classical section). For a linear model that's the difference between a guaranteed global optimum and a solver that can get stuck.

**Wrong probabilistic model.** MSE is the negative log-likelihood of a Gaussian with fixed variance. A binary outcome isn't Gaussian; it's Bernoulli, whose NLL *is* cross-entropy. So using MSE means you've assumed a noise model that's plainly wrong — homoscedastic Gaussian noise on a variable that takes two values. It also treats class indices as if they were numbers on a line: with classes {cat=0, dog=1, bird=2}, MSE says predicting "dog" when the answer is "bird" is a smaller error than predicting "cat," which is meaningless. And MSE is improper in the sense that it penalizes confident-correct and confident-wrong roughly symmetrically, while cross-entropy's unbounded penalty for confident-wrong is usually what you want.

Worth being fair though: MSE for classification *does* work in some regimes. There's a line of work (Hui and Belkin) showing that with proper rescaling, squared loss matches or beats cross-entropy on several NLP and vision benchmarks. So it's not that it can't work — it's that the default configuration has bad gradients and bad conditioning, and cross-entropy has none of those problems and better theory.

*Trap:* Saying "MSE is for regression, cross-entropy is for classification" as if it's a rule. That's the conclusion, not the reason. The reason is the gradient cancellation and the likelihood model.

### Q: Define perplexity. What's its relationship to loss, and what are its limitations?

**Answer.** Perplexity is the exponentiated average negative log-likelihood per token:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right) = \exp(\mathcal{L}_{\text{CE}}).$$

So it's just $e^{\text{loss}}$ when the loss is in nats. Loss 2.0 → PPL 7.4. Loss 3.0 → PPL 20.1. That exponential relationship matters for reading progress: a drop of 0.1 in loss at loss 2.0 is a perplexity drop of about 0.8, while the same 0.1 at loss 5.0 is a drop of about 15. Perplexity exaggerates improvements at high loss and compresses them at low loss, which is one reason people report loss during training and perplexity for comparison.

The interpretation: perplexity is the **effective branching factor** — the model is as uncertain as if it were choosing uniformly among PPL equally-likely options at each step. A perplexity of 20 means "as confused as picking uniformly from 20 candidates." Uniform over a 50k vocabulary gives PPL 50,000, which is the ceiling for a useless model.

Also: $\log_2(\text{PPL}) = $ bits per token, the compression rate. That's the cleanest way to compare across setups.

**Limitations**, and this is what the question is really about:

**It is not comparable across tokenizers.** This is the big one. Perplexity is per *token*, so a model with a larger vocabulary uses fewer tokens for the same text and each token carries more information — its per-token perplexity will be higher for identical modeling quality. Comparing a Llama-tokenizer model's PPL to a GPT-2-tokenizer model's PPL is meaningless. The fix is **bits per byte** or **bits per character**: $\text{BPB} = \frac{\mathcal{L}_{\text{total, bits}}}{N_{\text{bytes}}}$, which is tokenizer-invariant and lets you compare anything.

**It is not comparable across datasets.** Perplexity on Wikipedia and on Reddit are different numbers about different things.

**It's weakly correlated with what you care about.** Perplexity is dominated by the high-frequency mass — function words, whitespace, common continuations. A model can improve its perplexity meaningfully by getting better at predicting punctuation while getting no better at reasoning. Empirically, perplexity tracks downstream performance well *within* a family and *across scale*, but poorly across architectures and training recipes, and it barely responds to instruction tuning or RLHF, which visibly change model usefulness (and typically make perplexity slightly *worse*).

**It's undefined or misleading for models that aren't trained on plain likelihood** — a heavily RLHF'd model's perplexity on natural text goes up because its output distribution has been deliberately narrowed.

**Follow-up: "How do you compute perplexity on a document longer than the context window?"** → Sliding window with a stride. The naive approach — chop into disjoint context-length chunks — is unfair, because the first tokens of each chunk have no context and get bad predictions, inflating perplexity. The standard fix is a strided window: slide by $s < L$, and only score the last $s$ tokens of each window, so every scored token has at least $L - s$ tokens of context. Smaller stride is more accurate and more expensive; the limit $s=1$ is exact and costs $N$ forward passes.

### Q: Explain BPE. Why does tokenization cause the character-counting and arithmetic failures?

**Answer.** BPE starts with a base vocabulary (bytes, in byte-level BPE, so 256 symbols and no out-of-vocabulary possible ever). Then it repeatedly finds the most frequent adjacent symbol pair in the training corpus and merges it into a new symbol, recording the merge in an ordered list. Do this $k$ times and you have a vocabulary of $256 + k$ tokens. At encode time you apply the merge rules in the order they were learned. The result is that frequent sequences become single tokens and rare ones stay fragmented — a learned, frequency-driven compression.

Why it exists: character-level vocabularies give sequences that are 4-5× longer, and attention is quadratic, so you'd pay enormously. Word-level vocabularies can't handle unseen words, misspellings, or morphology, and the tail is infinite. Subword is the compromise: bounded vocabulary, no OOV, reasonable sequence length. Typical vocabularies are 32k-256k, and they've been growing (Llama-3 went to 128k, partly for multilingual efficiency).

**Why character counting fails.** The model does not see characters. "strawberry" might tokenize as `str|aw|berry` — three opaque IDs. Asking "how many r's" requires the model to know the spelling of each token, which is information not present in the input representation at all. It has to have *learned* the character composition of each token from indirect evidence in training data (spelling-out examples, hyphenation, typos), which it does imperfectly. So the failure isn't a reasoning failure, it's an information-access failure — it's like asking someone to count the letters in a word they've only ever heard, never seen. Notably, if you space out the letters — `s t r a w b e r r y`, which tokenizes per character — the models get it right, which is a clean confirmation of the diagnosis.

**Why arithmetic fails.** Number tokenization is erratic. In many tokenizers, "1234" might be one token, "1235" might be two, and "12345" might split as `123|45` — the boundaries are determined by corpus frequency, not by place value. So the same digit occupies different positions within different tokens, and there's no consistent representation of "the hundreds digit." Addition requires digit-aligned operations, which the tokenization actively obscures. The evidence for this diagnosis is strong: forcing single-digit tokenization, or right-to-left grouping in consistent 3-digit chunks, substantially improves arithmetic. Llama-3 splits numbers into groups of up to 3 digits partly for this reason. There's also good work showing that adding explicit digit-position embeddings (Abacus embeddings) gets length generalization on addition from a few digits to a hundred.

Other tokenization artifacts worth knowing: the **SolidGoldMagikarp** class of glitch tokens (tokens present in the tokenizer's training corpus but absent from the model's training data, so their embeddings are essentially untrained and produce bizarre behavior); leading-space sensitivity, where `" the"` and `"the"` are different tokens and prompts that end mid-word put the model off-distribution; and the multilingual tax, where languages underrepresented in tokenizer training need 2-5× more tokens for the same content, making them proportionally more expensive and effectively shrinking their context window.

**Follow-up: "So why not go tokenizer-free?"** → People are trying. ByT5 works on raw bytes and is robust to noise but much slower. The more promising direction is **dynamic/learned patching** — the Byte Latent Transformer (BLT) segments bytes into patches based on next-byte entropy, so predictable regions get large patches and surprising regions get small ones, allocating compute where it's needed. Reported results are competitive with token-based models at scale with better robustness on character-level tasks. It's not settled and BPE remains the default, but this is the most credible line of attack.

### Q: What did the Chinchilla paper actually say? And what's the difference between compute-optimal and inference-optimal?

**Answer.** Context: Kaplan et al. (2020) found that for a fixed compute budget you should scale parameters much faster than data — their recommendation implied models like GPT-3 (175B params, 300B tokens) were correctly proportioned. Hoffmann et al. (2022) redid the analysis more carefully and found the opposite: parameters and data should scale roughly **equally**.

Their headline: for compute budget $C \approx 6ND$ (with $N$ params, $D$ tokens, and 6 FLOPs per parameter per token for forward+backward), the optimal allocation has $N \propto C^{0.5}$ and $D \propto C^{0.5}$, giving the rule of thumb **~20 tokens per parameter**. They demonstrated it by training Chinchilla — 70B params on 1.4T tokens — with the *same compute* as Gopher (280B params on 300B tokens), and Chinchilla beat Gopher on essentially everything. So the previous generation of models was substantially undertrained.

The parametric form they fit:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

with $E$ the irreducible entropy of natural text, and the two terms the cost of finite parameters and finite data.

**A correction worth mentioning**, because it's the kind of thing that separates people who read the paper from people who read the summary: Epoch AI's 2024 replication attempt reconstructed the data from the paper's figures and found the third estimation approach (the parametric fit) was mis-estimated — the reported model fit the data poorly and the confidence intervals were implausibly tight for ~400 data points. Their corrected exponents are $\alpha \approx 0.35$, $\beta \approx 0.37$, versus the paper's lower values. Importantly, the corrected fit is *more* consistent with the paper's other two methods and still implies the ~20 tokens/parameter rule. So the headline survived; one of the three derivations didn't.

**Compute-optimal vs inference-optimal.** Chinchilla optimizes *training* compute only. That's the wrong objective for a model you're going to serve to millions of users, because inference cost scales with $N$ and is paid on every request forever. If you expect to serve $D_{\text{inf}}$ tokens over a model's lifetime, the total-cost-optimal point is a *smaller* model trained on *more* data than Chinchilla says. Sardana et al. formalized this and showed the optimal token/parameter ratio can be an order of magnitude above 20 for heavily-served models.

This is exactly what the field did. Llama-3 8B was trained on 15T tokens — roughly 1,875 tokens per parameter, almost 100× Chinchilla-optimal. It's badly "compute-inefficient" in Chinchilla's terms and completely correct as an engineering decision, because the training cost is amortized over billions of inference requests. So the modern practice is: use Chinchilla to understand the shape of the tradeoff, then deliberately overtrain small models because inference dominates lifetime cost.

**Follow-up: "Where does the $C = 6ND$ come from?"** → Forward pass costs roughly $2N$ FLOPs per token — each parameter participates in one multiply and one add. The backward pass costs about twice the forward, because you compute gradients with respect to both the inputs and the weights, so $4N$. Total $6N$ per token, times $D$ tokens. It ignores attention's $O(s^2)$ term, which is a good approximation when $d_{\text{model}} \gg s$ and a bad one for very long contexts — at 100k context the attention term is no longer negligible and the estimate breaks down.

### Q: Are emergent abilities real?

**Answer.** The claim (Wei et al., 2022) was that certain capabilities are absent in smaller models and present in larger ones, appearing sharply and unpredictably at some scale threshold rather than improving smoothly — multi-digit arithmetic, word unscrambling, some BIG-Bench tasks. The framing was that this is qualitatively different from smooth scaling laws and makes capabilities hard to forecast, which has obvious safety implications.

The main critique (Schaeffer, Miranda, Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?", NeurIPS 2023) is that the sharpness is largely **an artifact of the metric**, not a property of the model. The argument: many emergence-exhibiting tasks are scored with *discontinuous* metrics — exact string match, or multi-token accuracy where all $n$ tokens must be right. If per-token accuracy improves smoothly, then exact-match accuracy $\approx p^n$ improves like a smooth function raised to a power, which looks flat and then suddenly rises. Swap in a continuous metric on the same model outputs — token edit distance, or Brier score, or per-token log-likelihood — and the curve becomes smooth and predictable.

Their supporting evidence is strong: they showed emergence appears and disappears depending on the metric on the *same* model family and outputs; they showed emergence is rare on metrics that are continuous; and, most convincingly, they *induced* apparent emergence in vision autoencoders (where nobody claims emergence) purely by choosing a discontinuous metric. That's close to a demonstration.

Where I'd land, honestly: the metric critique is correct and important, and it does dissolve *most* claimed instances. But it doesn't dissolve the practical concern. Two responses stand up. First, for many real applications the discontinuous metric is the one that matters — if a task requires all 12 digits of an answer to be right, "smoothly improving per-token accuracy" doesn't help you, and from a deployment perspective the capability really does appear suddenly. Second, some phenomena look genuinely phase-transition-like at the mechanism level, not just the metric level — the formation of induction heads happens over a narrow window of training and coincides with a visible bump in the loss curve, and grokking shows sharp transitions in a continuous metric. So "capabilities sometimes appear abruptly" survives; "capabilities appear abruptly in a way that violates smooth scaling of the underlying quantity" mostly doesn't.

The correct practical takeaway is the one both camps agree on: if you want to *forecast* capabilities, measure them with a continuous proper-scoring metric, because that's what extrapolates. Reserve the discontinuous metric for deciding whether to ship.

*Trap:* Treating this as settled in either direction. Stating flatly "emergence is a mirage" is as overconfident as stating it's real — the honest answer names the artifact, accepts most of it, and identifies what remains.

### Q: SFT, RLHF, and DPO. Give me the actual objectives and explain what each buys you.

**Answer.** **SFT** is plain supervised learning on demonstrations: pairs $(x, y)$ of prompt and desired response, minimizing $-\sum_t \log \pi_\theta(y_t \mid x, y_{<t})$ — cross-entropy, exactly like pretraining but on curated data. It teaches format, instruction-following, and register. Its limit is that it can only teach you to imitate the demonstrations you have; it has no way to express "this response is better than that one," and human demonstrations are expensive and cap you at demonstrator quality.

**RLHF** has three stages.

1. SFT to get $\pi_{\text{SFT}}$, which also serves as $\pi_{\text{ref}}$.
2. Train a **reward model** on preference pairs. Given $(x, y_w, y_l)$ where $y_w$ is preferred, fit under the Bradley-Terry model $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$, so the loss is
$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\right].$$
The RM is typically the SFT model with the LM head swapped for a scalar head.
3. **PPO** against that reward with a KL penalty:
$$\max_\theta \ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\Big[r_\phi(x,y) - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)\Big].$$
In practice the KL is estimated per-token and folded into the reward as $r_t - \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$.

What RLHF buys: preference data is far cheaper to collect than demonstrations (judging is easier than writing), and crucially the policy can *exceed* the demonstrators, because it explores and gets scored rather than imitating. The cost is enormous complexity — four models in memory (policy, reference, reward, value), unstable optimization, many hyperparameters.

**DPO** eliminates the RM and the RL loop. The derivation: the KL-constrained objective above has a closed-form optimum, $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$. Invert it: $r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$. Substitute that into the Bradley-Terry loss; the partition function $Z(x)$ is the same for $y_w$ and $y_l$ and cancels in the difference. You're left with a loss purely in terms of the policy:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].$$

The elegant part: **the language model is secretly its own reward model.** You train with a simple classification-style loss, two models in memory instead of four, no sampling loop, no value function.

The tradeoff, which is where interviewers push: DPO is **off-policy** — it only ever sees the fixed preference dataset, never its own generations. RLHF's online sampling lets the policy discover and get feedback on responses nobody in the dataset produced. Empirically DPO is easier and often matches RLHF on standard benchmarks, but the best results at the frontier still tend to come from online methods, and there's a family of fixes (online DPO, iterative DPO, IPO, KTO, ORPO) trying to close the gap. As of now this is genuinely contested — DPO is the default for most practitioners because of the complexity difference, and online RL is what frontier labs use.

**Follow-up: "Why does DPO need a reference model at all if there's no KL term in the loss?"** → The KL constraint is *baked into* the loss — the $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ ratios are the implicit reward, and $\beta$ is exactly the KL coefficient. Without $\pi_{\text{ref}}$ you'd have a pure likelihood-ratio objective that would push the chosen response's probability up without bound and the rejected one's down without bound, degenerating immediately. The reference term anchors it. ORPO and SimPO are attempts to drop the reference model, using a length-normalized log-probability margin instead, which saves memory at some cost in stability.

**Follow-up: "What's a known failure mode specific to DPO?"** → The gradient can decrease the probability of the *chosen* response as long as it decreases the rejected one faster — the loss only cares about the margin. In practice you often see $\log\pi(y_w)$ falling over training, which means the model is getting worse at producing good responses while its preference ranking improves. Fixes include adding an SFT term on the chosen responses (as in RPO/CPO) or length normalization. Worth knowing because it's easy to miss if you only log the DPO loss and the implicit accuracy.

### Q: What is reward hacking and what do you do about it?

**Answer.** Reward hacking is when the policy finds inputs where the reward model is wrong and exploits them, so measured reward climbs while true quality stalls or falls. It's Goodhart's law with a gradient: the reward model is a *proxy* trained on a finite sample, and optimizing hard against a proxy takes you into the region where the proxy and the target diverge.

The mechanism is specific. The RM was trained on responses from the SFT distribution. As the policy moves away from that distribution, it produces responses the RM has never scored, and the RM's predictions there are extrapolations — often confidently wrong. So the policy gets a gradient toward whatever the RM happens to over-score off-distribution.

Documented manifestations: **length bias** is the canonical one — human annotators prefer longer responses on average, RMs learn "longer = better," and unchecked RLHF produces verbose padding. Verbosity correlates with reward strongly enough that it accounts for much of the apparent improvement, which is why AlpacaEval added a length-controlled variant. **Sycophancy** — agreeing with the user's stated position regardless of correctness, because annotators rate agreement highly. **Formatting exploits** — bullet points, headers, hedging preambles the RM likes. **Confident fabrication** — an authoritative-sounding wrong answer beats a hedged right one on annotator preference. And in coding/agentic RL, the sharpest version: writing code that special-cases the test rather than solving the problem, or modifying the test itself.

Gao, Schulman, and Hilton quantified the scaling: true reward as a function of the KL distance from the initial policy follows roughly $R(d) = d(\alpha - \beta d)$ where $d = \sqrt{\mathrm{KL}}$ — it rises, peaks, then falls. That's the overoptimization curve, and it's remarkably clean. Larger RMs push the peak further out and higher, but they don't eliminate it.

**Mitigations**, roughly in order of value:

- **KL penalty** to the reference model — the main structural defense, discussed below. Tune $\beta$ by watching the KL/reward curve rather than by taste.
- **Early stopping on true quality**, not on reward. You need an independent evaluation — held-out human preference or a different judge — because reward will keep going up.
- **RM ensembles**, and taking a conservative (min or lower-confidence-bound) aggregate. Attacks that fool one RM often don't fool all of them, so the pessimistic aggregate is much harder to exploit.
- **Iterative retraining** — collect fresh preference data on the *current* policy's outputs and retrain the RM. This directly attacks the distribution-shift mechanism and is what makes online RLHF loops work.
- **Explicit debiasing** — length-normalize the reward, or regress out length.
- **Rule-based / verifiable rewards** where possible. If correctness is checkable (math answers, unit tests), use the checker as the reward; it can't be fooled the way a learned RM can. This is the RLVR direction and it's the main reason reasoning-model RL has been more robust than preference RL. It's not immune — reward hacking on unit tests is real — but the surface is much smaller.

**Follow-up: "How do you actually detect it during a run?"** → Log reward and KL together and look for the point where reward keeps rising while KL accelerates — that's the signature. Track response length as a direct proxy. Hold out a small set of prompts and periodically get a human or a strong independent judge to rate outputs, and compare that curve to the RM curve; divergence is the definition of the problem. And read the samples. Reward hacking is usually blindingly obvious in the text and completely invisible in the metrics, and the failure mode of teams that get burned by it is that nobody read the outputs.

### Q: What exactly is the KL penalty doing in RLHF, and why is a reference model needed?

**Answer.** The objective is $\mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$. Three things it does, and they're distinct.

**It's a trust region on the proxy.** The reward model is only accurate near the distribution it was trained on. KL bounds how far the policy can drift, keeping it in the region where the reward is a meaningful estimate of quality. Without it, the policy walks straight into the RM's blind spots — and it does so *fast*, because those regions have the steepest apparent reward gradient. Empirically, unconstrained RLHF collapses within hundreds of steps to degenerate text that the RM scores very highly.

**It preserves capabilities.** The pretrained/SFT model has broad competence that the preference data doesn't cover. RL only optimizes what's rewarded; the KL term is what keeps everything else from being destroyed. It's a regularizer against catastrophic forgetting, and it's why the reference model is the SFT checkpoint and not, say, the base model.

**It prevents distributional collapse.** Reward maximization is mode-seeking — the optimal policy under a pure reward objective is deterministic, putting all mass on the single highest-reward response. That would give you a model that answers everything identically. KL keeps entropy up and preserves diversity.

**Why a reference model specifically**, rather than an entropy bonus or a norm penalty: an entropy bonus keeps the distribution spread out but doesn't anchor it anywhere — you can be high-entropy over garbage. A weight-space penalty is a poor proxy because small weight changes can cause large behavior changes and vice versa. KL to a reference is a penalty in *function* space, on the actual output distribution, measured on the actual prompts you care about. That's the right geometry. Mathematically, the KL-constrained optimum $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$ shows the structure directly: it's an exponential tilt of the reference distribution, with $\beta$ controlling the tilt strength. $\beta\to\infty$ gives you the reference back; $\beta\to0$ gives you argmax over reward.

Practical details: the KL is usually estimated per-token from the sampled trajectory rather than computed exactly, using $\log\frac{\pi_\theta(y_t)}{\pi_{\text{ref}}(y_t)}$ — an unbiased but high-variance estimator; the $k3$ estimator $(\rho - 1 - \log\rho)$ where $\rho = \pi_{\text{ref}}/\pi_\theta$ is lower-variance and always non-negative, and is now standard. $\beta$ is typically in the range 0.01–0.1 and is the single most important RLHF hyperparameter. Some implementations use adaptive $\beta$, targeting a fixed KL budget. The cost is that the reference model must stay in memory and be run on every sample — one of the four models RLHF needs.

**Follow-up: "What if you set $\beta$ too high?"** → The policy barely moves and RLHF does nothing — you've spent enormous compute to reproduce the SFT model. The failure is quiet, which makes it worse than the low-$\beta$ failure. The right way to tune is to plot true quality against achieved KL divergence, find where quality peaks, and set $\beta$ so you land there. That's exactly the overoptimization curve from the previous question, used as a tuning instrument.

### Q: Explain LoRA. Why does a low-rank update work at all?

**Answer.** Instead of updating $W \in \mathbb{R}^{d\times k}$ directly, freeze it and learn a low-rank correction:

$$W' = W + \Delta W = W + \frac{\alpha}{r}BA, \qquad B \in \mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll \min(d,k).$$

$A$ is initialized from a small Gaussian and $B$ is initialized to **zero**, so $\Delta W = 0$ at the start and the model is exactly the pretrained model at step 0 — no perturbation, no warmup shock. The scaling $\alpha/r$ exists so that changing $r$ doesn't require retuning the learning rate; $\alpha$ is usually set to $r$ or $2r$.

Parameter count goes from $dk$ to $r(d+k)$. For $d=k=4096$ and $r=8$: 16.8M down to 65k, a 256× reduction. Memory savings are larger than that in practice because the optimizer states (Adam's $m$ and $v$, 8 bytes/param) only exist for the trainable parameters — which is the dominant memory term in fine-tuning. Activation memory is unchanged, and you still need the frozen base weights in memory.

**Why low rank works.** The core claim, from Aghajanyan et al. before LoRA, is that fine-tuning has a low *intrinsic dimension*: you can reparameterize fine-tuning into a randomly-oriented subspace of a few thousand dimensions and still reach ~90% of full fine-tuning performance on GLUE. The interpretation is that pretraining has already learned the features; adaptation is a matter of *reweighting and recombining* existing features for a task, not learning new ones. That's inherently a low-rank operation. Consistent with this, LoRA's measured performance degrades gracefully as $r$ falls, and $r=8$ or 16 is usually enough for style/format adaptation — while tasks that require genuinely new knowledge need much higher rank, or full fine-tuning.

Other properties that matter practically: **zero inference latency**, because you can merge $BA$ into $W$ after training — unlike adapters, which add layers and add latency. **Composability** — you can hot-swap many LoRAs against one base model, which is what makes multi-tenant serving of hundreds of fine-tunes on one set of weights possible (S-LoRA, punica). And it's a **strong regularizer** — the rank constraint limits how far you can move, which is exactly what you want with a few thousand training examples.

Which matrices to apply it to: the original paper did $W_Q$ and $W_V$ only. Current practice applies it to all linear layers including the FFN, which works better at the same total parameter budget. QLoRA adds 4-bit NF4 quantization of the frozen base plus double quantization and paged optimizers, enabling 65B fine-tuning on a single 48GB GPU with essentially no quality loss relative to 16-bit LoRA.

**Follow-up: "Why initialize $B$ to zero and not both to zero?"** → If both were zero, the gradient of the product would be zero for both — $\partial(BA)/\partial A = B^\top = 0$ and vice versa — so nothing would ever move. It's a saddle point. You need exactly one factor nonzero to break symmetry, and putting the zero on $B$ (the output side) is what guarantees $\Delta W = 0$ initially.

**Follow-up: "When does LoRA clearly underperform full fine-tuning?"** → When you're teaching genuinely new knowledge or a new domain rather than adapting behavior — continued pretraining on a new language, or on a large corpus of domain text. Biderman et al.'s "LoRA Learns Less and Forgets Less" quantified this: on code and math continued-pretraining, LoRA lagged full fine-tuning substantially, but it also preserved base-model capabilities much better. So it's a real tradeoff, not a free lunch: the rank constraint that regularizes you is the same one that limits what you can learn. Rule of thumb: adaptation, use LoRA; knowledge injection at scale, use full fine-tuning.

### Q: What breaks when you quantize a model, and when?

**Answer.** Start with what quantization is: map high-precision weights (and sometimes activations) to low-bit integers with a scale and zero-point, $q = \text{round}(x/s) + z$. The error is roughly uniform in $[-s/2, s/2]$, so it's the *scale* — set by the range of the values in the group — that determines damage.

**The main thing that breaks: activation outliers.** This is the key empirical finding (Dettmers et al., LLM.int8()). Beyond about 6.7B parameters, transformers develop systematic outlier features — a small number of dimensions in the residual stream (often fewer than 1% of dimensions, and consistently the *same* dimensions across tokens) with magnitudes 20-100× the rest. Because a quantization scale must cover the max, one outlier forces a scale so coarse that every normal value collapses into a couple of levels. That's why naive int8 works fine on small models and falls off a cliff at scale — it's an emergent problem, not a gradual one. The fixes all target it: LLM.int8() keeps outlier dimensions in fp16 and does mixed decomposition; SmoothQuant migrates the difficulty from activations to weights by rescaling ($X\text{diag}(s)^{-1}\cdot\text{diag}(s)W$, which is mathematically equivalent and much friendlier to quantize); AWQ identifies salient weight channels using activation statistics and protects them.

**Weights vs activations.** Weight-only quantization is much easier — weights are well-behaved and roughly Gaussian, and 4-bit weight-only (GPTQ, AWQ, NF4) is essentially free in quality for most models. Activation quantization is where the outlier problem bites. So W4A16 is routine; W8A8 needs SmoothQuant-style tricks; W4A4 is research.

**What breaks first, in terms of capability.** Not perplexity — perplexity is remarkably robust and will look fine well past the point where the model is degraded. The things that go first are: long-context retrieval and instruction-following at long range; multi-step reasoning and arithmetic (errors compound across a chain, so a small per-step degradation becomes a large end-to-end one); low-resource languages; and calibration. This is a general and important point — **evaluate quantization on downstream tasks, especially compositional ones, not on perplexity.**

**Rough thresholds** (contested, and model-dependent): 8-bit is essentially lossless. 4-bit weight-only with a good method (AWQ/GPTQ, group size 128) costs a small but measurable amount — often under a point on most benchmarks. 3-bit is a noticeable drop. 2-bit requires specialized methods (QuIP#, AQLM) and still degrades. There's a scaling-law framing here (Dettmers and Zettlemoyer) that 4-bit is roughly the accuracy-per-bit optimum — for a fixed memory budget you're better off with a bigger model at 4-bit than a smaller one at 8-bit. That finding has held up reasonably well.

**Two more effects worth knowing.** First, models trained on far more tokens than Chinchilla-optimal appear *harder* to quantize — the intuition is that more training packs more information into the same weights, leaving less redundancy to discard. That's a real concern given the industry's move to heavy overtraining. Second, quantization-aware training (or QAT-lite via distillation) recovers most of the loss at low bit-widths, at the cost of needing a training run.

**Follow-up: "Why do outlier features exist at all?"** → The leading explanation connects them to attention sinks: models learn to dump attention mass on a small number of tokens (often BOS) as a no-op when a head has nothing useful to do, and producing a near-one-hot attention pattern requires very large key/query values in specific dimensions. There's also evidence they encode a "do not modify" signal that LayerNorm interacts with. Notably, architectural changes that remove the need for a sink — like adding a learnable no-op slot to the softmax denominator ("softmax-off-by-one" / attention with a null token) — reduce outliers substantially, which is decent evidence for the mechanism.

### Q: Do the KV cache math for a model I give you. 70B, 80 layers, 64 heads, head dim 128, GQA with 8 KV heads, fp16, 32k context, batch 16.

**Answer.** The formula:

$$\text{bytes} = 2 \times L \times n_{kv} \times d_{\text{head}} \times s \times b \times \text{bytes per value}$$

The leading 2 is for K and V separately.

Per token per sequence, per layer: $2 \times 8 \times 128 \times 2\text{ bytes} = 4096$ bytes = 4 KB.
Across 80 layers: $4\text{ KB} \times 80 = 320$ KB per token per sequence.
At 32k context: $320\text{ KB} \times 32768 \approx 10.5$ GB per sequence.
At batch 16: $\approx 164$ GB.

For comparison, without GQA (64 KV heads) that per-token figure would be $8\times$ larger — 2.56 MB per token, 84 GB per sequence, 1.34 TB at batch 16. Which is the whole reason GQA exists.

Two things I'd flag immediately. First, ~164 GB of cache versus ~140 GB of fp16 weights — **the cache is larger than the model**. That's the normal situation for long-context serving and it's why cache management, not weight loading, is the hard part of inference infrastructure. Second, this bounds your throughput directly: decode is memory-bandwidth-bound, so each decode step must stream the whole cache from HBM. On an 8×H100 node with ~3.35 TB/s each, roughly 27 TB/s aggregate, reading 164 GB takes about 6 ms — so you're capped near 160 steps/sec for that batch regardless of how fast the GPUs compute.

What you do about it, in order of impact:
- **Quantize the cache** to fp8 or int8. Halves it to ~82 GB, and the quality cost is small since KV values are better-behaved than activations generally. This is standard now.
- **PagedAttention** (vLLM): store the cache in fixed-size blocks with an indirection table instead of contiguous per-sequence buffers. This doesn't shrink the cache but eliminates the internal and external fragmentation from pre-allocating for max length, which in practice wastes 60-80% of cache memory. Big real-world win, plus it enables copy-on-write sharing of common prefixes.
- **Prefix caching** for shared system prompts — one copy across all requests that share it.
- **Sliding-window attention** in most layers, so those layers only cache $w$ tokens instead of $s$.
- **MLA** if you're designing the architecture, which compresses further than GQA.

**Follow-up: "Why is decode memory-bound but prefill compute-bound?"** → Arithmetic intensity. In prefill you process $s$ tokens at once, so each weight you load from HBM is used for $s$ token-computations — the matmuls are large and the FLOP-to-byte ratio is high. In decode you process one token per sequence, so you load the entire weight matrix to do a single matrix-vector product per sequence: arithmetic intensity is roughly the batch size. Modern GPUs need an intensity of a few hundred to saturate compute, so unless your batch is enormous, decode sits far on the memory-bound side of the roofline. This is precisely why batching helps throughput so much at decode and barely at all at prefill, and why speculative decoding works — it converts memory-bound single-token steps into compute-bound multi-token verification.

### Q: How does speculative decoding work, and why is the output distribution exactly correct?

**Answer.** The setup: a small fast draft model $q$ and the large target model $p$. Each round, the draft generates $\gamma$ tokens autoregressively (cheap). Then the target model scores **all $\gamma+1$ positions in a single forward pass** — this is the whole trick, because that pass costs almost the same as generating one token, since decode is memory-bandwidth-bound and you're reading the same weights either way. Then you accept or reject the drafted tokens.

The acceptance rule (Leviathan et al. / Chen et al.), for each drafted token $x$ in order:

1. Sample $u \sim U(0,1)$. **Accept** if $u < \min\left(1, \frac{p(x)}{q(x)}\right)$.
2. If rejected, **resample** from the residual distribution
$$p'(x) = \frac{\max\big(0,\ p(x) - q(x)\big)}{\sum_{x'}\max\big(0,\ p(x') - q(x')\big)}$$
and discard all remaining drafted tokens.
3. If all $\gamma$ are accepted, you additionally get a free token from the target's own distribution at the last position — so a fully-accepted round yields $\gamma+1$ tokens.

**Why the output distribution is exactly $p$.** This is modified rejection sampling, and the proof is a short case analysis. Fix a token $x$. There are two disjoint ways to emit it: accept it from the draft, or reject and resample it.

$$P(\text{emit } x) = \underbrace{q(x)\min\left(1,\frac{p(x)}{q(x)}\right)}_{\text{accepted}} + \underbrace{P(\text{reject})\cdot p'(x)}_{\text{resampled}}.$$

The first term is $\min(q(x), p(x))$. For the second, the total rejection probability is
$$P(\text{reject}) = \sum_{x'} q(x')\left[1 - \min\left(1,\tfrac{p(x')}{q(x')}\right)\right] = \sum_{x'}\big(q(x') - \min(q(x'),p(x'))\big) = 1 - \sum_{x'}\min(q(x'),p(x')),$$
and the normalizer of $p'$ is $\sum_{x'}\max(0, p(x')-q(x')) = \sum_{x'}\big(p(x') - \min(p(x'),q(x'))\big) = 1 - \sum_{x'}\min(p(x'),q(x'))$ — the *same quantity*. So they cancel, and the second term is exactly $\max(0, p(x)-q(x))$.

Adding: $\min(q(x),p(x)) + \max(0, p(x)-q(x))$. If $p(x)\le q(x)$ this is $p(x) + 0 = p(x)$. If $p(x) > q(x)$ this is $q(x) + p(x)-q(x) = p(x)$. Either way, **exactly $p(x)$**. No approximation, no temperature caveat, no distributional drift. That's the property that makes speculative decoding a free lunch rather than a quality tradeoff.

**Speedup.** If $\alpha$ is the per-token acceptance rate, expected tokens per round is $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, and the speedup is that divided by the cost of one target pass plus $\gamma$ draft passes. Typical real numbers: 2-3× for a well-matched draft. It's bounded by acceptance rate, so the draft must be *distributionally similar*, not just fast — which is why draft models are usually distilled from or trained alongside the target.

Variants worth naming: **Medusa** adds extra decoding heads to the target model itself, avoiding a separate draft model. **EAGLE** drafts in feature space rather than token space and reuses the target's hidden states, achieving much higher acceptance. **Lookahead decoding** and prompt-lookup use n-gram matching against the context as a zero-cost draft, which works surprisingly well for tasks with a lot of copying (summarization, code editing).

**Follow-up: "Does this help throughput or just latency?"** → Primarily latency. At batch size 1 it's a clear win because you're deeply memory-bound and have spare FLOPs. As batch size grows, the target model's forward pass becomes compute-bound and the extra verification FLOPs are no longer free — at high batch you're spending real compute on tokens you may reject, and speculative decoding can *reduce* throughput. So it's the right tool for interactive low-batch serving and the wrong one for maximizing tokens/sec on a saturated batch server. Good serving stacks make it adaptive on current batch occupancy.

### Q: Explain MoE routing. Why is load balancing needed, and how is it done?

**Answer.** A Mixture-of-Experts layer replaces the FFN with $N$ parallel FFNs ("experts") plus a small router. For each token, the router computes $h(x) = W_r x$, takes a softmax, and selects the top-$k$ experts (usually $k=1$ or 2, sometimes 8 in fine-grained designs). The output is the weighted sum of those experts' outputs, $\sum_{i\in\text{top-}k} g_i(x)\,E_i(x)$ with $g$ the normalized gate values.

The point is **decoupling parameters from FLOPs**. A model with 8 experts and top-2 routing has roughly 8× the FFN parameters but only 2× the FFN compute per token. Parameters store knowledge; FLOPs cost money. Mixtral 8x7B has 47B total parameters and activates ~13B per token. That's the entire value proposition, and it's why nearly every frontier model is now believed to be MoE.

**Why load balancing is needed.** The router is trained by gradient descent alongside everything else, and it has a nasty positive feedback loop: an expert that is chosen slightly more often gets more gradient, becomes better, and is therefore chosen more often. Left alone, the router collapses onto a small number of experts and the rest are never trained — you've paid for $N$ experts and are using two. This is **routing collapse**, and it is the default outcome without intervention, not an edge case.

There's also a hard systems constraint: experts are distributed across devices, and each has a fixed **capacity** — the number of tokens it can process in a batch, set as $\text{capacity factor}\times \frac{\text{tokens}}{N}$. Tokens routed to a full expert are **dropped** (they skip the FFN and pass through via the residual). So imbalance directly costs you quality *and* wastes the idle devices, since every device waits for the most loaded one.

**How it's done.** The standard **auxiliary load-balancing loss** (Switch Transformer):

$$\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^{N} f_i \cdot P_i,$$

where $f_i$ is the *fraction of tokens* dispatched to expert $i$ (non-differentiable, a count) and $P_i$ is the *average router probability* assigned to expert $i$ (differentiable). The product is minimized when both are uniform at $1/N$, giving $\mathcal{L}_{\text{aux}} = \alpha$. The clever part is that gradients flow through $P_i$ while $f_i$ acts as a scaling coefficient — so an over-subscribed expert gets its router probability pushed down proportionally to how over-subscribed it is. $\alpha$ is typically 0.01: large enough to balance, small enough not to dominate the LM loss.

Other mechanisms: **capacity factor** above 1 (1.25 is common) gives slack. **Noisy top-k gating** (the original Shazeer approach) adds tunable Gaussian noise to the router logits to encourage exploration. **Expert choice** routing inverts the problem — each expert picks its top-$c$ tokens instead of each token picking experts — which makes balance perfect by construction, at the cost of some tokens getting zero experts and a causality issue that makes it awkward for autoregressive decoding. **Auxiliary-loss-free balancing** (DeepSeek-V3) adds a per-expert learned bias to the routing scores, adjusted online to equalize load, avoiding the gradient interference an auxiliary loss introduces — this is a notable recent shift, since the aux loss does measurably hurt the LM objective. And **shared experts** (DeepSeek) always route every token through one or two common experts, so general knowledge lives there and the routed experts can specialize.

**Follow-up: "Are MoE experts actually specialized by topic?"** → Mostly not, which surprises people. The Mixtral paper looked for domain specialization and found little — routing correlates far more with *syntax and token identity* than with semantics, and consecutive tokens are routed to the same expert more often than chance. Fine-grained MoE designs with many small experts show more evidence of meaningful specialization. So the honest answer is that "experts" is a misleading name; they're better thought of as a learned sparse parameterization than as a committee of specialists.

*Trap:* Saying MoE reduces memory. It increases total parameters substantially and all experts must be resident (or paged) — MoE trades memory for compute, not the other way round. It's a win only when you're compute- or latency-bound and have memory to spare.
---

## 5. Classical ML That Still Gets Asked

### Q: Derive logistic regression from scratch and prove the loss is convex.

**Answer.** **Model.** We want $P(y=1|x)$. A linear function $w^\top x$ ranges over all of $\mathbb{R}$, so squash it with the logistic function: $p = \sigma(z) = \frac{1}{1+e^{-z}}$, $z = w^\top x + b$. The motivation isn't arbitrary — if you assume the log-odds are linear, $\log\frac{p}{1-p} = w^\top x$, then solving for $p$ gives you exactly the sigmoid. And that log-odds-linear assumption is what you get from any two class-conditional densities in the exponential family with shared dispersion (e.g. two Gaussians with a common covariance), so logistic regression is the discriminative counterpart to a broad family of generative models.

**Likelihood.** Each observation is Bernoulli: $P(y|x) = p^y(1-p)^{1-y}$. Over $n$ i.i.d. observations, the log-likelihood is $\sum_i [y_i\log p_i + (1-y_i)\log(1-p_i)]$. Negate to get the loss:

$$\mathcal{L}(w) = -\sum_{i=1}^{n}\Big[y_i\log\sigma(z_i) + (1-y_i)\log\big(1-\sigma(z_i)\big)\Big].$$

**Gradient.** Use $\sigma'(z) = \sigma(z)(1-\sigma(z))$. Differentiating one term with respect to $z_i$:

$$\frac{\partial \ell_i}{\partial z_i} = -\left[\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i}\right]p_i(1-p_i) = -\big[y_i(1-p_i) - (1-y_i)p_i\big] = p_i - y_i.$$

So $\nabla_w \mathcal{L} = \sum_i (p_i - y_i)x_i = X^\top(p - y)$. The gradient is the design matrix times the residual — structurally identical to linear regression, which is the elegant part. There is no closed-form solution because $p$ depends nonlinearly on $w$, so you solve it iteratively (Newton's method here is IRLS).

**Convexity.** Differentiate again. $\frac{\partial p_i}{\partial w} = p_i(1-p_i)x_i$, so

$$H = \nabla^2_w \mathcal{L} = \sum_{i=1}^n p_i(1-p_i)\,x_i x_i^\top = X^\top S X, \qquad S = \operatorname{diag}\big(p_i(1-p_i)\big).$$

Now the PSD argument. Take any $v \in \mathbb{R}^d$:

$$v^\top H v = \sum_i p_i(1-p_i)\,v^\top x_i x_i^\top v = \sum_i p_i(1-p_i)\,(x_i^\top v)^2.$$

Every term is a product of $p_i(1-p_i)$, which is strictly positive because $\sigma$ maps into the open interval $(0,1)$, and $(x_i^\top v)^2$, which is a square and hence $\ge 0$. So the sum is $\ge 0$ for every $v$, meaning $H \succeq 0$ everywhere. A twice-differentiable function with a PSD Hessian on a convex domain is convex. Therefore any stationary point is a global minimum and gradient descent cannot get stuck in a bad local optimum.

Two refinements worth adding. It's convex but not *strictly* convex in general — $H$ is singular if $X$ is rank-deficient ($X^\top S X$ has rank at most $\text{rank}(X)$), so the solution needn't be unique. And on **perfectly separable** data, the MLE doesn't exist: you can always increase the likelihood by scaling $w$ up, driving $p_i \to y_i$ and the loss toward zero while $\|w\|\to\infty$. Adding L2 regularization makes the objective strictly convex (the Hessian becomes $X^\top S X + 2\lambda I \succ 0$) and guarantees a unique finite solution, which is why regularized logistic regression is the practical default.

**Follow-up: "Why is separable data a problem in practice if the loss still goes down?"** → The weights diverge, so the model becomes arbitrarily confident — every prediction goes to 0 or 1 — and the probabilities become useless even though classification accuracy is perfect. Numerically the optimizer never converges and you get overflow warnings. It's especially common with high-dimensional sparse features (bag-of-words with more features than documents is almost always separable), which is why sklearn's `LogisticRegression` applies L2 by default and people are often surprised to learn it's regularized out of the box.

### Q: Explain SVMs — the margin intuition and the kernel trick.

**Answer.** **Margin.** For linearly separable data there are infinitely many separating hyperplanes. The SVM picks the one that maximizes the distance to the nearest points of either class. The intuition is robustness: a decision boundary sitting far from all data is least likely to be flipped by a small perturbation of the data, and there's a real generalization bound behind it — the VC dimension of large-margin separators is bounded in terms of the margin and the data radius, independent of the ambient dimension, which is why SVMs work in very high dimensions.

Setup: with labels $y_i \in \{-1,+1\}$, the functional margin is $y_i(w^\top x_i + b)$ and the geometric margin is that over $\|w\|$. Fix the scale so the closest points have functional margin exactly 1; then the geometric margin is $1/\|w\|$ and maximizing it is minimizing $\|w\|^2$:

$$\min_{w,b}\ \tfrac{1}{2}\|w\|^2 \quad \text{s.t.}\quad y_i(w^\top x_i + b)\ge 1\ \ \forall i.$$

That's a convex QP with a unique solution. The soft-margin version adds slacks: $\min \frac12\|w\|^2 + C\sum_i\xi_i$ subject to $y_i(w^\top x_i+b)\ge 1-\xi_i$, $\xi_i\ge0$. Small $C$ means a wide margin with more violations (more regularization); large $C$ means fit the training data hard. Equivalently, the whole thing is $\min \sum_i \max(0, 1-y_i f(x_i)) + \lambda\|w\|^2$ — **hinge loss plus L2**, which is the useful way to compare it to logistic regression: same regularizer, different loss, and hinge's flat region past margin 1 is what produces sparsity in the support vectors.

**The dual and the kernel trick.** Form the Lagrangian and eliminate $w$; you get

$$\max_\alpha \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\,\langle x_i, x_j\rangle \quad\text{s.t.}\quad 0\le\alpha_i\le C,\ \sum_i \alpha_i y_i = 0,$$

with $w = \sum_i \alpha_i y_i x_i$. Two crucial features: the data appears **only through inner products** $\langle x_i, x_j\rangle$, and by the KKT complementary-slackness conditions $\alpha_i = 0$ for every point strictly outside the margin — so only the **support vectors** (points on or inside the margin) have nonzero $\alpha$ and the solution depends on nothing else.

The kernel trick: since only inner products appear, replace $\langle x_i,x_j\rangle$ with $K(x_i,x_j) = \langle\phi(x_i),\phi(x_j)\rangle$ for some feature map $\phi$ into a higher-dimensional space. You never compute $\phi$ — you only ever need the kernel value. Mercer's theorem says any symmetric PSD kernel corresponds to *some* valid inner product in *some* Hilbert space, so you can design kernels directly. The polynomial kernel $(x^\top z + c)^d$ corresponds to all monomials up to degree $d$; the RBF kernel $\exp(-\gamma\|x-z\|^2)$ corresponds to an *infinite*-dimensional feature space (expand the exponential and you get all polynomial degrees). That's the striking part — you're fitting a linear classifier in an infinite-dimensional space and paying $O(n^2)$ kernel evaluations, not infinite cost.

Why SVMs faded: training is $O(n^2)$ to $O(n^3)$ and prediction cost scales with the number of support vectors, so they don't handle millions of examples; and deep nets *learn* the feature map instead of requiring you to choose a kernel, which turns out to matter enormously for perception. They're still excellent for small-to-medium tabular problems with a few thousand examples, and for anything where you have a domain-specific kernel (string kernels, graph kernels).

**Follow-up: "How do you choose $\gamma$ in the RBF kernel?"** → $\gamma$ controls the width of the similarity bump — $K \to 0$ once $\|x-z\| \gg 1/\sqrt\gamma$. Large $\gamma$ means each support vector influences only its immediate neighborhood, so the boundary becomes extremely wiggly and you overfit (in the limit, every point is its own support vector and you've built a 1-NN classifier). Small $\gamma$ makes the kernel nearly constant and the model underfits toward linear. The `scale` heuristic, $\gamma = 1/(d\cdot\text{Var}(X))$, is a reasonable default; beyond that, grid-search $\gamma$ and $C$ jointly on a log scale, because they interact — both control effective complexity and the good region is a diagonal band, not a rectangle. And standardize your features first, since RBF uses Euclidean distance and is completely at the mercy of feature scaling.

### Q: Bias-variance in trees versus boosting. Why do bagging and boosting work differently?

**Answer.** A deep decision tree is the archetypal **high-variance, low-bias** model. It can fit any training set to zero error given enough depth, so bias is essentially zero, but it's brutally unstable — change one training point near a high split and the entire tree below it reorganizes. Two bootstrap samples from the same data give you visibly different trees.

**Bagging attacks variance.** Train $B$ trees on bootstrap resamples and average. If the individual trees each have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the average is

$$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$

The second term vanishes as $B\to\infty$; the first doesn't. So the ceiling on bagging's benefit is set entirely by how *correlated* the base models are. Bias is unchanged — averaging unbiased-ish models keeps them unbiased-ish. That formula also explains **random forests** precisely: by restricting each split to a random subset of $m$ features (typically $\sqrt{d}$ for classification), you *decorrelate* the trees, lowering $\rho$. You pay a small amount of bias per tree, because each tree is sometimes forced to split on a worse feature, and you buy a large reduction in the irreducible $\rho\sigma^2$ term. That's the entire design.

So: bagging uses deep, fully-grown, low-bias/high-variance trees, and the averaging kills the variance. It's embarrassingly parallel, hard to overfit by adding trees (more trees monotonically helps, it just saturates), and robust to hyperparameters.

**Boosting attacks bias.** It uses *weak* learners — stumps or depth-3 trees, which are high-bias and low-variance — and fits them **sequentially**, each one correcting the errors of the ensemble so far. AdaBoost reweights misclassified examples upward; gradient boosting fits each new tree to the negative gradient of the loss with respect to the current predictions. Either way the ensemble's bias falls monotonically as you add trees, because you keep adding capacity aimed exactly at the current residual error.

The consequence is the opposite failure mode: boosting **can and will overfit** if you add too many trees, because you're continuously reducing bias with no averaging to control variance. Number of trees is a regularization hyperparameter you must tune with early stopping — unlike random forests where more is simply better. The other regularizers are the learning rate (shrinkage, $\nu \approx 0.01$–$0.1$), tree depth, subsampling of rows and columns, and explicit L1/L2 on leaf weights.

Practical summary: random forests are the robust default that works with almost no tuning; gradient boosting reaches higher accuracy but requires tuning and early stopping; and gradient boosting (XGBoost/LightGBM/CatBoost) remains the state of the art on tabular data, where neural nets still don't reliably win.

### Q: How does gradient boosting actually work? Give me the algorithm.

**Answer.** The framing that makes it click: gradient boosting is **gradient descent in function space**. In ordinary gradient descent you update parameters, $\theta \leftarrow \theta - \eta\nabla_\theta L$. Here you update the *function itself*, $F \leftarrow F - \eta\nabla_F L$ — and because you can't represent an arbitrary function, you fit a regression tree to approximate the negative gradient and take a step in that direction.

The algorithm (Friedman's gradient boosting machine):

1. Initialize with a constant: $F_0(x) = \arg\min_\gamma \sum_i L(y_i,\gamma)$. For squared loss that's the mean; for log loss it's the log-odds of the base rate.
2. For $m = 1,\dots,M$:
   a. Compute **pseudo-residuals** — the negative gradient of the loss with respect to the current prediction, per example:
   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}.$$
   b. Fit a regression tree $h_m$ to the pairs $(x_i, r_{im})$ — note it's always a *regression* tree fit with squared error, regardless of the outer loss.
   c. For each leaf $j$ of that tree, compute the optimal constant by line search on the actual loss:
   $$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L\big(y_i,\ F_{m-1}(x_i) + \gamma\big).$$
   d. Update with shrinkage: $F_m(x) = F_{m-1}(x) + \nu\sum_j \gamma_{jm}\mathbb{1}[x\in R_{jm}]$.
3. Output $F_M$.

The key insight in step (a): **the residual you fit is loss-specific.** For squared loss, $\partial L/\partial F = F - y$, so the pseudo-residual is literally $y_i - F(x_i)$ — the ordinary residual, which is why the intuitive "fit the errors" story works for regression. For log loss with $F$ as the log-odds, the pseudo-residual is $y_i - p_i$. For absolute error it's $\text{sign}(y_i - F(x_i))$, which is why L1 boosting is robust to outliers — a wildly wrong point contributes a gradient of magnitude 1, not a huge one. So the whole framework generalizes to any differentiable loss just by changing what you compute in step (a), which is the real contribution.

**Shrinkage** $\nu$ matters more than people expect: taking small steps and using more trees consistently beats taking full steps with fewer, and Friedman's empirical finding — $\nu \le 0.1$ with correspondingly more trees — has held up. It's a regularizer in the same family as early stopping.

**XGBoost's contributions** on top of this: it uses a second-order Taylor expansion of the loss (both gradient $g_i$ and Hessian $h_i$), which gives a closed-form optimal leaf weight $w_j^* = -\frac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda}$ and a principled split-gain criterion derived from the same expansion; explicit L1/L2 regularization on the leaf weights in the objective; a sparsity-aware split finder that learns a default direction for missing values; and heavy systems engineering (column blocks, cache-aware access, approximate quantile sketching). LightGBM adds histogram binning, leaf-wise rather than level-wise growth, and GOSS/EFB for speed. CatBoost adds ordered target statistics for categorical features, which fixes a subtle target-leakage problem in naive target encoding.

**Follow-up: "Why fit a regression tree to the gradient rather than solving for the best tree directly?"** → Because finding the tree that directly minimizes the loss is intractable — it's a combinatorial search over structures with a non-decomposable objective. The gradient step turns it into a standard least-squares regression problem, which greedy tree induction solves efficiently. It's the same reason you use gradients anywhere: replacing a hard global optimization with a sequence of easy local ones.

### Q: Derive PCA.

**Answer.** Two derivations, and they give the same answer, which is the interesting bit.

**Setup.** Center the data: $X \in \mathbb{R}^{n\times d}$ with column means removed. Centering is not optional — without it the first component points at the mean, not at the direction of variation.

**Derivation 1: maximize variance.** Find the unit vector $w$ such that the projections $Xw$ have maximum variance. Since the data is centered, the projected variance is $\frac{1}{n}\|Xw\|^2 = w^\top \Sigma w$ with $\Sigma = \frac{1}{n}X^\top X$ the covariance matrix. We need the constraint $\|w\| = 1$, or the objective is unbounded. Lagrangian:

$$\mathcal{L} = w^\top\Sigma w - \lambda(w^\top w - 1).$$

Set the derivative to zero: $2\Sigma w - 2\lambda w = 0$, so

$$\Sigma w = \lambda w.$$

That's an eigenvalue equation — the stationary points are exactly the eigenvectors of the covariance matrix. And substituting back, the variance captured is $w^\top\Sigma w = w^\top\lambda w = \lambda$. So the eigenvalue *is* the variance along that direction, and to maximize it you take the eigenvector with the largest eigenvalue. Subsequent components follow by the same argument with an added orthogonality constraint, giving the eigenvectors in descending eigenvalue order. Since $\Sigma$ is real symmetric, the spectral theorem guarantees a full orthonormal eigenbasis with real non-negative eigenvalues.

**Derivation 2: minimize reconstruction error.** Find the rank-$k$ subspace with projection matrix $P = WW^\top$ ($W$ orthonormal, $d\times k$) minimizing $\sum_i \|x_i - WW^\top x_i\|^2$. Expand: $\|x_i\|^2 - \|W^\top x_i\|^2$. The first term is fixed, so minimizing reconstruction error is *exactly* maximizing $\sum_i\|W^\top x_i\|^2$ — the projected variance. Same problem, same solution. This equivalence is worth stating because it explains why PCA is simultaneously the best linear compressor and the maximum-variance projection; those are not obviously the same goal.

**In practice you use the SVD**, not the eigendecomposition of $\Sigma$. Write $X = U\Sigma_{\text{sv}} V^\top$. Then $X^\top X = V\Sigma_{\text{sv}}^2 V^\top$, so the right singular vectors $V$ are the principal directions and the eigenvalues are $\sigma_i^2/n$. Two reasons this is better: you never form $X^\top X$, which squares the condition number and loses precision, and it's $O(nd\min(n,d))$ rather than requiring an explicit $d\times d$ matrix, which matters when $d$ is huge.

**Choosing $k$:** cumulative explained variance ratio $\sum_{i\le k}\lambda_i / \sum_i \lambda_i$ (pick 95%, say), the scree-plot elbow, or a principled criterion like Minka's automatic dimensionality selection.

**Caveats I'd volunteer.** PCA is scale-dependent, so standardize features first unless they're already in comparable units — otherwise the component with the largest units dominates. It finds directions of maximum *variance*, which need not be the directions that are useful for your downstream task; if you want discriminative directions, LDA is the supervised analogue. It's linear, so it can't unroll a curved manifold — that's what kernel PCA, t-SNE, UMAP, and autoencoders are for. And components are orthogonal by construction, which sometimes forces uninterpretable mixtures; if you want parts-based interpretable factors, NMF or sparse dictionary learning may be more appropriate.

### Q: When do linear models beat deep learning?

**Answer.** More often than people expect, and it's worth being concrete about the conditions.

**Small data relative to feature count.** With a few hundred to a few thousand examples, a deep net has enough capacity to memorize and not enough signal to regularize. A regularized linear model has strong inductive bias and degrades gracefully. The rough boundary is data-dependent, but under ~10k examples on tabular data, linear or gradient-boosted models are usually the right call.

**Genuinely linear or near-linear structure.** If the underlying relationship is close to additive in your features, a linear model is correctly specified and a neural net is spending capacity discovering something you already knew. Adding a few interaction terms or splines covers most of the remaining gap.

**High-dimensional sparse features.** Text classification with bag-of-words, click prediction with hashed categorical crosses — millions of sparse features, and linear models with L1/L2 are extremely strong here. This is why logistic regression dominated ad CTR prediction for years and why linear models remain competitive baselines on text classification.

**When you need interpretability or auditability.** A coefficient is a coefficient — it has a sign, a magnitude, a confidence interval, and it can be shown to a regulator. In credit scoring, insurance underwriting, and clinical risk models, this is often a hard requirement, not a preference. Post-hoc explanations of a neural net (SHAP, LIME) are approximations of a model you don't understand, and they're contestable in a way that a coefficient isn't.

**When you need calibrated probabilities out of the box.** Logistic regression is trained by maximum likelihood on the correct likelihood and tends to be well calibrated without post-processing.

**Latency, cost, and operational simplicity.** A dot product is microseconds and runs anywhere. A model with no GPU dependency, no framework version drift, and a closed-form or convex fit is much cheaper to own. For a high-QPS ranking system, the linear model may be the only thing that fits the latency budget.

**Non-stationary data requiring frequent retraining.** Convex problems retrain in seconds with reproducible results; deep nets have run-to-run variance and need monitoring.

The one I'd emphasize is **tabular data generally**. As of now, gradient-boosted trees still beat neural networks on most tabular benchmarks — Grinsztajn et al.'s systematic comparison attributes this to trees' robustness to uninformative features, their invariance to feature scaling and monotone transformations, and their bias toward the axis-aligned, non-smooth decision boundaries that tabular targets actually have. Various deep-tabular architectures (TabNet, FT-Transformer, TabPFN) have claimed wins, and TabPFN in particular is genuinely impressive on very small datasets, but the general verdict hasn't flipped.

Where deep learning is clearly correct: perceptual data with strong local structure (images, audio), sequences with long-range dependencies (language), any case where you need representation learning rather than fixed features, and anywhere transfer from a large pretrained model is available. Those cover a lot of ground, but they aren't everything.

### Q: MLE versus MAP. What's the relationship, and when does it matter?

**Answer.** **MLE** maximizes the likelihood of the data given the parameters:

$$\hat\theta_{\text{MLE}} = \arg\max_\theta P(D\mid\theta) = \arg\max_\theta \sum_i \log P(x_i\mid\theta).$$

**MAP** maximizes the posterior, which by Bayes is proportional to likelihood times prior:

$$\hat\theta_{\text{MAP}} = \arg\max_\theta P(\theta\mid D) = \arg\max_\theta \big[\log P(D\mid\theta) + \log P(\theta)\big],$$

since the evidence $P(D)$ doesn't depend on $\theta$.

**The relationship you should be able to state instantly: MAP is MLE plus a regularizer, and the prior *is* the regularizer.** Two cases:

- Gaussian prior $\theta\sim\mathcal{N}(0,\tau^2 I)$: $\log P(\theta) = -\frac{\|\theta\|^2}{2\tau^2} + \text{const}$, so MAP = MLE $-\ \lambda\|\theta\|_2^2$ with $\lambda = 1/(2\tau^2)$. **L2 regularization is a Gaussian prior.** Tighter prior (smaller $\tau$) means stronger regularization.
- Laplace prior $P(\theta)\propto e^{-|\theta|/b}$: MAP = MLE $-\ \lambda\|\theta\|_1$. **L1 is a Laplace prior**, and its spike at zero is exactly why it induces sparsity.
- Uniform (improper) prior: $\log P(\theta)$ is constant, and MAP reduces to MLE. So MLE is MAP with a flat prior.

**When it matters.** With little data, the prior dominates and MAP is much better behaved — MLE will happily produce degenerate estimates. Canonical example: estimating a coin's bias from 3 heads out of 3 flips. MLE says $p=1$, which assigns probability zero to ever seeing a tail. MAP with a Beta(2,2) prior gives $(3+1)/(3+2) = 0.8$, which is sane. Laplace smoothing in naive Bayes is exactly MAP with a Dirichlet prior, and it's there to prevent one unseen word from zeroing out an entire document's probability. As $n\to\infty$ the likelihood term grows like $n$ while the prior term stays constant, so MAP converges to MLE — the prior washes out, which is the Bernstein-von Mises phenomenon.

**What both share, and what they miss.** Both are **point estimates**, which means neither gives you uncertainty. The fully Bayesian thing is to keep the whole posterior and integrate it out when predicting: $P(y|x,D) = \int P(y|x,\theta)P(\theta|D)d\theta$. That's the posterior predictive, and it's what actually propagates parameter uncertainty into predictions. MAP is a crude summary of it — the mode — and the mode can be badly unrepresentative in high dimensions, where the posterior mass concentrates far from the mode (the "typical set" issue). That's a real limitation, not a technicality: in high dimensions almost none of the probability mass is near the peak.

One more subtlety worth knowing: **MAP is not invariant to reparameterization.** If you reparameterize $\theta \to g(\theta)$, the MLE transforms correctly (the mode of the likelihood follows the map, because likelihood is not a density in $\theta$) but the MAP does not, because a change of variables introduces a Jacobian factor into the prior density and shifts the mode. So the MAP estimate depends on how you chose to write down your parameters, which is philosophically uncomfortable. The posterior mean and the full posterior don't have this problem for the corresponding transformed quantities.

**Follow-up: "Give me the deep learning connection."** → Weight decay is MAP with a Gaussian prior on weights, as above. Early stopping is an implicit prior favoring parameters near initialization. Dropout has a variational-Bayesian interpretation (Gal and Ghahramani) under which MC-dropout at test time approximates the posterior predictive, which is why averaging several dropout-enabled forward passes gives usable uncertainty estimates. And deep ensembles, which just train $N$ models from different seeds and average, are a crude but empirically excellent approximation to the posterior predictive — they consistently beat more principled variational methods on calibration under distribution shift, which is a slightly embarrassing but robust finding.

### Q: What's the difference between generative and discriminative models, and when would you prefer each?

**Answer.** A **discriminative** model learns $P(y|x)$ directly — logistic regression, SVMs, neural classifiers, random forests. A **generative** model learns the joint $P(x,y)$, usually by factoring it as $P(x|y)P(y)$, and then gets $P(y|x)$ by Bayes' rule — naive Bayes, Gaussian discriminant analysis, HMMs, and in the modern sense, language models and diffusion models.

The classic analysis is Ng and Jordan (2001), comparing naive Bayes to logistic regression, which form a "generative-discriminative pair" — they have the same parametric form for $P(y|x)$ but fit it differently. Their result: the generative model has a **higher asymptotic error** (because its class-conditional independence assumption is wrong, so it's asymptotically biased) but it **converges to that error much faster** — $O(\log d)$ examples versus $O(d)$ for logistic regression. So naive Bayes wins in the small-data regime and logistic regression overtakes it once you have enough data. That crossover is the whole story, and it's the answer to "when would you prefer each" for the classical case: little data or many features relative to examples, go generative; plenty of data, go discriminative.

The general principle behind that: generative models solve a *harder* problem (model the full distribution of $x$, which is high-dimensional and mostly irrelevant to the decision boundary) and therefore need stronger assumptions, but those assumptions act as a regularizer. Vapnik's dictum applies — don't solve a more general problem than you need to as an intermediate step.

Other reasons to go generative even with plenty of data: you can **sample** new data; you can handle **missing features** naturally by marginalizing; you get **anomaly detection** for free from low $P(x)$; you can add a new class by fitting only that class's conditional, without retraining anything else; and you can handle semi-supervised learning, because unlabeled $x$ still informs $P(x)$.

The modern twist worth mentioning: the discriminative-wins conclusion was drawn in a world of small models and small data, and large language models have inverted it in a specific way. A generative language model trained on the whole internet, then prompted or lightly fine-tuned, beats task-specific discriminative classifiers on most NLP tasks — not because the generative framing is statistically better for classification, but because modeling $P(x)$ over a colossal corpus is an extraordinarily effective way to *learn representations*, and those transfer. So the generative objective won as a pretraining strategy while the discriminative objective is still what you'd use if you had a fixed task and fixed labeled data and nothing else.

*Trap:* Calling GANs and VAEs "generative" in the same sense as naive Bayes without noting the difference — most modern deep generative models learn $P(x)$ unconditionally or $P(x|c)$, and are used for synthesis rather than for classification via Bayes' rule.

### Q: Explain the curse of dimensionality concretely.

**Answer.** It's a family of related geometric facts about high-dimensional spaces, all of which make intuition from 2D and 3D actively misleading. The concrete ones:

**Volume concentrates at the boundary.** In a $d$-dimensional unit ball, the fraction of volume within the outer shell of thickness $\epsilon$ is $1 - (1-\epsilon)^d$. For $d=100$ and $\epsilon=0.05$, that's $1 - 0.95^{100} \approx 99.4\%$. Essentially all the volume is in a thin skin. So "typical" points are near the surface, and the notion of a dense interior evaporates.

**Sampling density collapses exponentially.** To cover the unit cube at a resolution of 0.1 per axis you need $10^d$ cells. In 1D that's 10 samples; in 10D it's $10^{10}$. Any method that relies on having neighbors nearby — kNN, kernel density estimation, local regression, decision trees at depth — needs a sample size that grows exponentially with dimension. This is the version that bites in practice.

**Distances concentrate.** For i.i.d. features, the ratio of the farthest to the nearest neighbor distance tends to 1 as $d\to\infty$:
$$\frac{\max_i \|x_i - q\| - \min_i\|x_i-q\|}{\min_i\|x_i-q\|} \to 0.$$
Every point is roughly equidistant from every other. That makes "nearest neighbor" nearly meaningless, and it undermines any algorithm whose core operation is comparing distances — kNN, k-means, DBSCAN, and distance-based anomaly detection all degrade.

**Everything is nearly orthogonal.** Two random unit vectors in $\mathbb{R}^d$ have expected inner product 0 with standard deviation $1/\sqrt d$. So in high dimensions random directions are almost always nearly perpendicular. This one is a *feature*, not a bug, for representation learning: it's why you can pack exponentially many nearly-orthogonal directions into $d$ dimensions (Johnson-Lindenstrauss), which is the basis for random projections, hashing, and the superposition story for how neural networks represent more features than they have dimensions.

**Why ML works anyway:** the **manifold hypothesis**. Real data doesn't fill its ambient space — natural images live on a manifold of far lower intrinsic dimension than the pixel count, and the same is true of text embeddings. The effective dimension that governs sample complexity is the intrinsic one, not the ambient one. That's why a 224×224×3 image (150,528 ambient dimensions) can be classified from a million examples when the curse would say you need astronomically more. Deep networks are, on this view, machines for finding coordinates on that manifold.

**Follow-up: "Practical consequences for something like a vector database?"** → Exact nearest-neighbor search degrades to brute force — tree-based indexes (KD-trees, ball trees) lose to linear scan above roughly 20 dimensions, because the pruning that makes them fast requires distance gaps that concentration destroys. That's why production systems use *approximate* methods with different structure: HNSW (navigable small-world graphs), IVF with product quantization, or LSH. It's also why cosine similarity is preferred to Euclidean for embeddings — normalizing removes the magnitude dimension, and angular distance concentrates less badly on data that lies on a manifold. And it's why dimensionality reduction before indexing (PCA down to 128-256 dims, or Matryoshka embeddings trained to be truncatable) is standard practice rather than an optimization.
