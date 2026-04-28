# Language Modeling Losses — Interview Grill

> 35 questions on LM pretraining objectives. Drill until you can answer 25+ cold.

---

## A. CLM (Causal Language Modeling)

**1. What's the CLM loss?**
$\mathcal{L} = -\sum_t \log P(\text{token}_t \mid \text{tokens}_{<t}; \theta)$. Cross-entropy on next-token prediction at every position. Equivalent to maximizing autoregressively factored joint probability of the sequence.

**2. Why is CLM the dominant LLM objective?**
(a) Direct generation — training task matches inference task. (b) Computationally efficient — $N$ losses per forward pass. (c) Scales as a power law in compute/parameters/data. (d) Natural in-context learning. (e) Single architecture for everything.

**3. Walk me through CLM loss computation in code.**
```
logits[:, :-1, :]   # predictions at positions 0..N-2
targets[:, 1:]       # tokens at positions 1..N-1
loss = cross_entropy(logits.view(-1, V), targets.view(-1))
```
Shift-by-one: at position t, predict token at t+1.

**4. How do you handle padding in CLM loss?**
Mask out padding positions: `mask = tokens != pad_id; loss = (per_token_loss * mask).sum() / mask.sum()`. Or use PyTorch's `ignore_index = -100` for padding.

**5. How do you mask the prompt during SFT?**
Set the loss for prompt tokens to 0 (or replace target with `-100`). Only response tokens contribute to the loss. The prompt is "given context"; only the response is "what we want to predict."

**6. Why can't CLM directly use bidirectional context?**
Bidirectional attention would let the model see the answer when predicting it (trivial). The causal mask enforces that position $t$ can only see positions $\leq t$, making the next-token prediction meaningful.

**7. How does CLM enable in-context learning?**
Trained to continue patterns from any prefix. With scale, the model develops induction heads (Olsson et al. 2022) that copy tokens after prefix matches. Few-shot prompts work because the model continues the demonstrated pattern.

---

## B. MLM (Masked Language Modeling)

**8. What's MLM?**
BERT's objective. Mask 15% of tokens; predict them from bidirectional context. Loss is cross-entropy on masked positions only.

**9. Walk me through BERT's masking strategy.**
For 15% of selected tokens: 80% replaced with `[MASK]`, 10% replaced with random token, 10% kept unchanged. Mitigates train-test mismatch (model can't assume `[MASK]` always means "predict me").

**10. Why is MLM less sample-efficient than CLM?**
Only 15% of tokens contribute to the loss. The other 85% is "wasted" forward-pass compute. CLM has $N$ losses per forward pass.

**11. Why does MLM not directly support generation?**
Trained to fill the middle, not extend the end. To generate, you'd need to autoregressively mask one position and fill it — slow and unnatural. Modern LLMs need generation; they use CLM.

**12. Where does MLM still win?**
Encoder models for embeddings, classification, NER. Sentence-BERT, BGE, E5 — all use MLM-style pretraining. Bidirectional context produces richer per-token representations.

**13. MLM's train-test mismatch?**
`[MASK]` appears in training but not at inference. The 80/10/10 mix mitigates but doesn't fully eliminate this. ELECTRA's replaced-token detection sidesteps the issue.

---

## C. NSP and ELECTRA

**14. What was NSP?**
Next Sentence Prediction. BERT trained with an auxiliary task: predict whether sentence B follows sentence A. 50% positive pairs (consecutive sentences); 50% negative (random pairs).

**15. Why was NSP removed?**
RoBERTa (Liu et al. 2019) showed empirically: NSP doesn't help, removing it improves downstream performance. The task is too easy (random sentences are trivially distinguishable from continuation) — minimal useful signal.

**16. What's ELECTRA?**
Clark et al. 2020. Replaced-token detection. A small generator fills in masked tokens; a larger discriminator predicts which tokens were replaced. Loss is binary classification per token.

**17. Why is ELECTRA more sample-efficient than MLM?**
Every token contributes to the loss (binary classification at every position), not just 15%. ~4x more sample-efficient than MLM. Matches BERT performance with ~25% the compute.

**18. Where is ELECTRA-style pretraining used?**
Some efficient encoder models. Doesn't directly support generation, so not relevant for LLMs. ELECTRA-style ideas are sometimes incorporated into hybrid objectives.

---

## D. Span corruption and PrefixLM

**19. What's span corruption?**
T5/BART objective. Mask contiguous spans of tokens (~3 tokens average); replace each span with a sentinel; encoder-decoder predicts the masked spans as output. Encoder is bidirectional; decoder is causal.

**20. Walk me through span corruption with an example.**
Input: "The <X> jumped over the <Y> dog". Target: "<X> quick brown fox <Y> lazy <eos>". Each `<X>` etc. is a sentinel token marking a masked span. Decoder generates the spans in order.

**21. Pros and cons of span corruption?**
Pros: efficient (whole spans contribute), captures phrase-level semantics, encoder-decoder architecture flexible. Cons: encoder-decoder is heavier than decoder-only; doesn't directly enable free-form generation as cleanly as CLM. Modern LLMs prefer CLM.

**22. What's PrefixLM?**
Hybrid attention pattern: bidirectional over the prefix; causal over the target. Used in T5.1.1, GLM. Theoretically combines benefits of bidirectional encoding and autoregressive generation.

**23. Why didn't PrefixLM dominate?**
Implementation complexity (different attention in different parts). Decoder-only with in-context demonstrations achieves similar effects with less complexity. PrefixLM never took off at frontier scale.

---

## E. Modern variants

**24. What's Mixture-of-Denoisers (UL2)?**
Tay et al. 2022. Combines multiple denoising objectives: R-denoising (regular spans), S-denoising (sequential prefix-LM), X-denoising (extreme corruption). Model learns multiple skills. Research-prominent; not adopted at frontier scale.

**25. What's multi-token prediction?**
Gloeckle et al. 2024, used in DeepSeek-V3. Predict the next $k$ tokens at each position using $k$ separate prediction heads. Denser signal per token. Auxiliary heads enable speculative decoding without a separate draft model.

**26. What's the role of contrastive losses?**
Used for embedding models (Sentence-BERT, BGE, E5): pull similar sentences together in embedding space, push different ones apart. Different paradigm from generative LM losses; supports retrieval and semantic search.

---

## F. Cross-entropy specifics

**27. Why cross-entropy as the LM loss?**
MLE under categorical distribution. The log-likelihood of the data given the model factorizes as $\sum_t \log P(\text{token}_t \mid \text{context})$. Negative gives cross-entropy. It's not a design choice; it's what likelihood mandates.

**28. What's perplexity?**
$\text{PPL} = \exp(\text{cross-entropy})$. Geometric inverse of average per-token probability. Bounded below by $\exp(\text{true entropy})$ (perfect LM $\approx 1$); bounded above by $|V|$ (uniform random model = vocab size).

**29. Why can't you compare PPL across tokenizers?**
PPL is per-token. Different tokenizers split text into different numbers of tokens. A tokenizer with finer splits gets lower PPL on the same text purely from having more easy predictions. Compare per-byte/per-character likelihood for fair comparison.

**30. Cross-entropy gradient w.r.t. logits?**
$\nabla \mathcal{L} / \nabla z = \mathrm{softmax}(z) - \text{one\_hot}(\text{target})$. Same form as logistic regression. Clean because softmax is the canonical link function for the categorical distribution (GLM theory).

---

## G. Implementation gotchas

**31. What's the log-sum-exp trick and why?**
For numerical stability when computing softmax: $\log \sum_v \exp(z_v) = \max(z) + \log \sum_v \exp(z_v - \max(z))$. Without it, large logits would overflow `exp`. Standard in all softmax/cross-entropy implementations.

**32. Why is `F.cross_entropy` better than `softmax + log + nll_loss`?**
PyTorch's `F.cross_entropy` combines log-softmax with negative log-likelihood in one numerically stable operation. Computing softmax first then taking log can lose precision via overflow/underflow.

**33. How do you handle very large vocabularies efficiently?**
Sampled softmax (during training): only compute softmax over a sampled subset of the vocab. Hierarchical softmax: tree-structured factorization. Adaptive softmax: cluster vocab by frequency. For modern LLMs, full softmax is feasible and standard.

---

## H. Advanced and frontier

**34. What's z-loss / output normalization regularization?**
Add $\alpha \cdot (\log Z(x))^2$ to the loss, where $Z$ is the partition function. Prevents the model from learning extremely large logits (which can cause instability). Used in some LLM pretraining recipes (PaLM, GPT-3 likely).

**35. What's auxiliary loss in MoE training?**
For Mixture-of-Experts models: an auxiliary loss to encourage balanced expert utilization (so all experts get used roughly equally). Without it, the router collapses to using a few experts. See `41_mixture_of_experts/`.

---

## Quick fire

**36.** *Default mask ratio in BERT?* 15%.
**37.** *NSP removed in?* RoBERTa.
**38.** *T5 objective?* Span corruption.
**39.** *ELECTRA paper?* Clark et al. 2020.
**40.** *Modern LLM default?* CLM (next-token prediction).

---

## Self-grading

If you can't answer 1-10, you don't know LM losses. If you can't answer 11-25, you'll struggle on architecture interviews. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 25+/40 cold.
