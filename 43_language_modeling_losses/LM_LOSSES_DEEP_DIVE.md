# Language Modeling Losses: A Frontier-Lab Interview Deep Dive

> **Why this exists.** The choice of pretraining objective shapes everything downstream — what the model can do, how it scales, what tasks it's good at. Interviewers ask: "Why CLM not MLM for modern LLMs?", "What's span corruption?", "Why did NSP get removed from BERT?". This document covers each major LM objective, why it was proposed, and why most have been replaced by next-token prediction.

---

## 1. The big picture

Every language modeling objective tries to teach a model to predict masked or future tokens from context. The differences:

| Objective | Mask pattern | Direction | Used by | Status |
|---|---|---|---|---|
| **CLM** (Causal LM) | All future tokens | Causal | GPT, LLaMA, modern LLMs | **Dominant** |
| **MLM** (Masked LM) | 15% random | Bidirectional | BERT, RoBERTa | Encoder models |
| **NSP** (Next Sentence) | Sentence pairs | — | BERT (removed in RoBERTa) | Deprecated |
| **Span corruption** | Contiguous spans | Encoder-decoder | T5, BART | Niche |
| **PrefixLM** | Prefix bidirectional, target causal | Hybrid | T5.1.1, GLM | Niche |
| **MoD** (Mixture-of-Denoisers) | Mixed patterns | Hybrid | UL2 | Research |
| **ELECTRA** | Replaced-token detection | Bidirectional | ELECTRA | Encoder models |

Modern LLM pretraining is overwhelmingly **CLM (next-token prediction)** — for reasons we'll see.

---

## 2. CLM (Causal Language Modeling)

The objective:

$$
\mathcal{L} = -\sum_t \log P(\text{token}_t \mid \text{tokens}_{<t}; \theta)
$$

For every position $t$, predict the next token from all previous tokens. Equivalent to maximizing the joint probability of the sequence factored autoregressively.

### Why CLM is the dominant objective for modern LLMs

**1. Direct generation.** The training task — predict the next token from context — is exactly what generation requires. No mismatch between training and inference.

**2. Computationally efficient.** Every position contributes a loss term in parallel. One forward pass gives $N$ next-token-prediction tasks. Combined with the causal mask, you get $N$ losses for the cost of one forward pass.

**3. Scales beautifully.** Loss decreases as a power law in compute, parameters, and data (Kaplan, Chinchilla). No saturating regime in sight.

**4. Naturally supports in-context learning.** Few-shot prompts work because the model is trained to predict whatever continues the prefix; demonstrations in the prompt shape what's likely.

**5. Same architecture for everything.** No need for separate encoder/decoder. One transformer with a causal mask.

### CLM's limitations

- **Bidirectional context not used.** The model only ever sees the past. For tasks where future context matters (some embedding tasks, some classification), encoder-style bidirectional models win.
- **Wasteful for some tasks.** Predicting easy tokens (function words, common patterns) doesn't teach much. ELECTRA-style objectives can extract more signal per token.

But for general-purpose LLMs, CLM has won the field.

---

## 3. MLM (Masked Language Modeling)

BERT's pretraining objective. Randomly mask 15% of tokens; predict them from bidirectional context.

For 15% of selected tokens: 80% replaced with `[MASK]`, 10% replaced with random token, 10% kept unchanged. Loss:

$$
\mathcal{L} = -\sum_{t \in \text{masked}} \log P(\text{token}_t \mid \text{rest of sequence}; \theta)
$$

### Why MLM has bidirectional advantages

The model sees both left and right context when predicting a masked token. This produces richer representations for tasks like classification, NER, embeddings. BERT-style models still dominate sentence/document embedding leaderboards.

### Why MLM doesn't generate
The model is trained to fill in the middle, not extend the end. To generate, you'd need to autoregressively mask one position at a time and fill it — slow and unnatural. **Modern LLMs need to generate**, so they use CLM.

### MLM's other limitations

- **Train-test mismatch.** `[MASK]` tokens appear during training but not at inference, so the model sees something different. The 80/10/10 mix mitigates but doesn't eliminate this.
- **Inefficient.** Only 15% of tokens contribute to the loss; the rest is "wasted" forward-pass compute.
- **Doesn't learn long-range generation patterns.** The mask covers a token; surrounding tokens give strong signal. The model never has to predict from scratch.

### Where MLM still wins

- Embedding models (BERT, sentence-BERT, BGE, etc.).
- Classification and NER on rich representations.
- Some retrieval models.

Encoder-only LLMs are not gone; they're just niche compared to decoder-only LLMs.

---

## 4. NSP (Next Sentence Prediction) — and why it died

Original BERT included an auxiliary objective: given two sentences, predict if sentence B follows sentence A in the original text.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}, \qquad \mathcal{L}_{\text{NSP}} = -\log P(\text{is-next} \mid \text{sentence}_A, \text{sentence}_B)
$$

50% of training pairs were "next sentence"; 50% were random.

### Why NSP was thought to help
Was supposed to capture sentence-level relationships, useful for downstream tasks like NLI (Natural Language Inference) and QA.

### Why it was removed
RoBERTa (Liu et al. 2019) showed: **NSP doesn't help**. Removing it improves downstream performance. The task is too easy (random sentences are trivially distinguishable from continuation) and provides almost no useful signal.

Modern encoder LMs don't use NSP. Some models (XLNet, ALBERT) propose alternative sentence-level objectives but none have caught on.

### Lesson
"This auxiliary objective sounds reasonable" is not a good reason to add it. Validate empirically.

---

## 5. Span corruption (T5, BART)

Mask contiguous spans of tokens (~3 tokens average); replace each span with a sentinel; predict the masked spans as the output.

```
Input:  "The <X> jumped over the <Y> dog"
Target: "<X> quick brown fox <Y> lazy <eos>"
```

T5's flagship objective. Encoder-decoder architecture.

### Pros
- More efficient than MLM (whole spans contribute, not single tokens).
- Better captures phrase-level semantics.
- Works well with encoder-decoder.

### Cons
- Encoder-decoder is heavier than decoder-only.
- Span corruption doesn't directly support free-form generation as cleanly as CLM.
- Mostly superseded by CLM for general-purpose LLMs.

### Status
T5 family still used (especially Flan-T5 for fine-tuning). Span corruption variants in some research (UL2's R-denoising). Not the modern default.

---

## 6. PrefixLM (T5.1.1, GLM)

A hybrid: bidirectional attention over the **prefix**; causal attention for the **target** to be generated.

```
For input "Translate to French: The cat":
  - "Translate to French: The cat" gets bidirectional attention
  - The model generates the French translation autoregressively
  - During generation: causal mask within the target portion
```

### Why this design
Combines benefits: bidirectional encoding for the prompt (rich representations) + autoregressive generation for the output. Theoretically should beat both pure encoder and pure decoder for some tasks.

### Why it didn't dominate
Implementation complexity (different attention patterns in different parts). Decoder-only with in-context demonstrations achieves similar effects with less complexity. PrefixLM never really took off at frontier scale.

### Status
T5.1.1, GLM, some research models. Not the modern default for general-purpose LLMs.

---

## 7. Mixture-of-Denoisers (UL2)

Tay et al. 2022 (Google). Combines multiple denoising objectives:
- **R-denoising:** regular span corruption.
- **S-denoising:** sequential prefix-LM (long prefix, short target).
- **X-denoising:** extreme corruption (large span lengths, small spans).

The model is trained on a mix; learns to handle different denoising types.

### Why this idea
Each denoiser teaches different skills: short masks teach local pattern; long masks teach long-range dependencies; extreme corruption teaches imagination/generation.

### Status
Research-prominent (UL2, Flan-UL2). Not adopted at frontier scale by major labs.

---

## 8. ELECTRA: replaced token detection

Clark et al. 2020. Different from masked-then-predict:

1. A small **generator** model fills in masked tokens (like BERT).
2. A larger **discriminator** model is trained to detect which tokens were replaced (binary classification per token).

The discriminator $D(t)$ outputs the probability that token $t$ was replaced. Trained with binary cross-entropy over all positions:

$$
\mathcal{L} = -\,\mathbb{E}\!\left[\sum_t \big(\mathbf{1}[\text{token}_t \text{ replaced}] \log D(t) + \mathbf{1}[\text{token}_t \text{ original}] \log(1 - D(t))\big)\right]
$$

### Why this idea
Signal per token: every position contributes to the loss (binary classification), not just 15%. Sample efficiency is much higher than MLM. ELECTRA matches BERT performance with ~25% the compute.

### Status
ELECTRA-style discriminative pretraining is used in some efficient encoder models. Doesn't directly support generation; not relevant for LLMs.

---

## 9. The CLM loss in detail

For an autoregressive decoder transformer:

```python
def clm_loss(logits, tokens):
    """
    logits: shape [batch, seq_len, vocab_size]
    tokens: shape [batch, seq_len]
    
    For position t, predict tokens[t+1] from logits[t].
    The "shift by one": logits[:-1] vs tokens[1:].
    """
    # Shift: predict next token at each position
    pred_logits = logits[:, :-1, :]      # [batch, seq-1, vocab]
    pred_targets = tokens[:, 1:]          # [batch, seq-1]
    
    # Per-position cross-entropy
    loss = cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        pred_targets.reshape(-1),
        reduction="mean"
    )
    return loss
```

### The "shift by one" gotcha
A common interview question: "Why does CLM training use `logits[:-1]` and `targets[1:]`?" Because at position `t`, the model's logits should predict the token at position `t+1`. The shift aligns predictions with their targets.

### Padding handling
Ignore padding positions in the loss:
```python
mask = tokens != pad_token_id
loss = (per_token_loss * mask).sum() / mask.sum()
```
Otherwise the model gets noisy gradients on padding tokens.

### Loss masking for SFT
For supervised fine-tuning on (prompt, response) pairs: only compute loss on response tokens, not prompt tokens. The prompt is "given," only the response is "predicted." Masking the prompt portion is a critical detail.

---

## 10. Why CLM produces in-context learning

This is one of the deepest results in modern ML, frequently asked in interviews.

When trained on next-token prediction over diverse data, the model implicitly learns to:
1. Recognize patterns in the prefix.
2. Continue those patterns in the suffix.

If the prefix contains demonstrations like:
```
Q: 2 + 2 = ?  A: 4
Q: 3 + 5 = ?  A: 8
Q: 7 + 1 = ?  A:
```

The model continues the pattern (the demonstration sets up the rule; the model applies it to the new question). This **emerges from CLM training** with sufficient scale and data; there's no explicit ICL objective.

The mechanism (Olsson et al. 2022): induction heads learn to copy patterns from earlier in the context when prefix matches recur. With enough scale, induction heads enable robust ICL.

---

## 11. Auxiliary objectives sometimes added to LLMs

Beyond pure CLM, some recipes add:
- **Multi-token prediction** (Gloeckle et al. 2024, used in DeepSeek-V3): predict the next `k` tokens at each position with `k` separate heads. Better signal density per token; faster speculative decoding via the auxiliary heads.
- **Contrastive losses** for embedding training (e.g., E5, BGE).
- **Code-specific objectives** for code models (in-filling losses, fix-the-bug objectives).

These are mostly research-stage at frontier scale. Pure CLM remains the workhorse.

---

## 12. Loss function math

For any of these objectives, the per-position loss is cross-entropy:

$$
\mathcal{L} = -\log p(\text{true token} \mid \text{model}) = -\log \mathrm{softmax}(\text{logits})[\text{true idx}] = -\!\left(z_{\text{true}} - \log \sum_v \exp(z_v)\right)
$$

The $\log \sum_v \exp(z_v)$ is the **log-sum-exp** (LSE) — numerically computed with the standard

$$
\mathrm{LSE}(z) = \max(z) + \log \sum_v \exp(z_v - \max(z))
$$

trick to avoid overflow.

Cross-entropy is the same loss as binary cross-entropy in logistic regression generalized to $V$ classes. Same gradient form: $\mathrm{softmax}(\text{logits}) - \text{one-hot}(\text{target})$.

---

## 13. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why CLM not MLM?" | CLM directly enables generation; MLM doesn't. CLM gets `N` losses per forward pass; MLM only ~15% of positions. |
| "Why was NSP removed?" | Empirically didn't help (RoBERTa). Task too easy; provides minimal useful signal. |
| "What's span corruption?" | Mask contiguous spans; predict in encoder-decoder architecture. Used in T5, BART. |
| "Why does the loss use shift-by-one?" | At position `t`, the prediction is for token `t+1`. Align logits[:-1] with targets[1:]. |
| "How do you mask the prompt during SFT?" | Compute loss only on response tokens; mask prompt tokens with -100 (PyTorch ignores this). |
| "What's ELECTRA's contribution?" | Replaced-token detection — every token contributes to loss, not just 15%. ~4x more sample efficient than MLM. |
| "What's the mixture-of-denoisers idea?" | Combine multiple denoising objectives (different mask patterns) so the model learns multiple skills. UL2. |
| "How does in-context learning emerge from CLM?" | The model learns to continue patterns from the prefix; with scale, induction heads form, enabling robust ICL. |

---

## 14. The 8 most-asked LM-loss interview questions

1. **What's the CLM loss?** Sum of cross-entropy on next-token predictions across positions.
2. **CLM vs MLM?** CLM autoregressive, MLM bidirectional. CLM: generation, scaling, ICL. MLM: representations, embeddings.
3. **Why was NSP removed?** Empirically useless; RoBERTa showed it hurts.
4. **What's span corruption?** Mask spans, encoder-decoder predicts them. T5.
5. **What's ELECTRA?** Replaced-token detection; every token contributes to loss.
6. **Why does CLM enable ICL?** Trained to continue patterns; induction heads emerge with scale.
7. **What's the shift-by-one in CLM?** Logits at position `t` predict token at `t+1`; align logits[:-1] with targets[1:].
8. **What auxiliary objectives are used in modern LLMs?** Multi-token prediction (DeepSeek-V3), contrastive (embeddings), code-specific.

---

## 15. Drill plan

1. Whiteboard CLM loss with the shift-by-one detail.
2. Compare CLM vs MLM (sample efficiency, generation, representations).
3. Know the failure of NSP empirically.
4. Be able to discuss ICL emergence from CLM at a sketchy level.
5. Drill `INTERVIEW_GRILL.md`.
