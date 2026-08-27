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

> **Saying it out loud.** Every one of these objectives is a variation on one idea: hide part of the text and make the model reconstruct it. What differs is which part you hide and which direction the model gets to look. Hide the future and look only left, that's next-token prediction and you get a generator. Hide scattered words and look both ways, that's BERT and you get a great encoder that can't write. The whole history of this table is the field discovering that the simplest option — predict the next token — scales better than every clever alternative, so everything else ended up either niche or dead.

---

## 2. CLM (Causal Language Modeling)

> **In plain language.** This is next-token prediction, the objective behind GPT and every modern chat model. The formula below just says: at every position, take the probability the model assigned to the token that actually came next, take its log, add them up, and flip the sign.

The objective:

$$
\mathcal{L} = -\sum_t \log P(\text{token}_t \mid \text{tokens}_{<t}; \theta)
$$

For every position $t$, predict the next token from all previous tokens. Equivalent to maximizing the joint probability of the sequence factored autoregressively.

> **Saying it out loud.** The objective is: at every position, guess the next token, and get penalized by how much probability you failed to put on the right one. The sum over positions looks imposing but it comes from something simple — the probability of a whole sentence is the probability of the first word, times the second given the first, and so on, and taking logs turns that chain of multiplications into a sum. So there's really only one prediction problem here, repeated at every position. The practical upshot is density: a single forward pass over a 4,000-token sequence gives you 4,000 supervised examples, which is why this objective is so cheap per unit of learning.

### Why CLM is the dominant objective for modern LLMs

**1. Direct generation.** The training task — predict the next token from context — is exactly what generation requires. No mismatch between training and inference.

**2. Computationally efficient.** Every position contributes a loss term in parallel. One forward pass gives $N$ next-token-prediction tasks. Combined with the causal mask, you get $N$ losses for the cost of one forward pass.

**3. Scales beautifully.** Loss decreases as a power law in compute, parameters, and data (Kaplan, Chinchilla). No saturating regime in sight.

**4. Naturally supports in-context learning.** Few-shot prompts work because the model is trained to predict whatever continues the prefix; demonstrations in the prompt shape what's likely.

**5. Same architecture for everything.** No need for separate encoder/decoder. One transformer with a causal mask.

> **Saying it out loud.** The one-sentence version is that CLM won because it's the only objective where training and deployment are the same task. Everything else on the list has some gap — you train the model to fill blanks and then ask it to write, and you have to bridge that somehow. On top of that CLM extracts a loss from every token instead of 15 percent of them, works with a single stack of causal layers rather than an encoder plus decoder, and gives you in-context learning as a free side effect. And the clincher for labs is that it scales as a clean power law, so you can predict the final loss of a hundred-million-dollar run from small pilot runs — no other objective has that much evidence behind its scaling curve.

### CLM's limitations

- **Bidirectional context not used.** The model only ever sees the past. For tasks where future context matters (some embedding tasks, some classification), encoder-style bidirectional models win.
- **Wasteful for some tasks.** Predicting easy tokens (function words, common patterns) doesn't teach much. ELECTRA-style objectives can extract more signal per token.

But for general-purpose LLMs, CLM has won the field.

> **Saying it out loud.** The real cost is that the model only ever looks left. When your goal is to represent a sentence rather than continue it, that's a genuine handicap — the meaning of a word often depends on what comes after it, and a causal model has to guess. That's why a 110M-parameter BERT still beats much bigger decoder models on embedding benchmarks. The second issue is that most tokens are easy — punctuation, function words, the second half of a common phrase — so a lot of your compute goes into predictions that teach almost nothing, which is exactly the inefficiency ELECTRA and multi-token prediction try to attack.

---

## 3. MLM (Masked Language Modeling)

BERT's pretraining objective. Randomly mask 15% of tokens; predict them from bidirectional context.

For 15% of selected tokens: 80% replaced with `[MASK]`, 10% replaced with random token, 10% kept unchanged. Loss:

$$
\mathcal{L} = -\sum_{t \in \text{masked}} \log P(\text{token}_t \mid \text{rest of sequence}; \theta)
$$

> **Saying it out loud.** Masked language modeling is fill-in-the-blank. You hide about 15 percent of the words and the model recovers them using context from both sides, which is a fundamentally different job from continuing text. That two-way view is why BERT-style models produce such good per-token representations — when you're encoding a word, seeing what follows it helps as much as seeing what preceded it. The 80/10/10 detail exists because the MASK token never appears at inference, so BERT sometimes substitutes a random word or leaves the word alone, forcing the model to keep a useful representation everywhere. The structural cost is that you pay for a full forward pass and only harvest a gradient from 15 percent of the positions.

### Why MLM has bidirectional advantages

The model sees both left and right context when predicting a masked token. This produces richer representations for tasks like classification, NER, embeddings. BERT-style models still dominate sentence/document embedding leaderboards.

> **Saying it out loud.** Think about the word 'bank' — whether it's a river bank or a financial one might only be settled by the word three positions later. A causal model has to hedge; a bidirectional one just looks. That's why the good retrieval and classification encoders are all masked-language-model pretrained, and why they haven't been displaced by LLMs despite being a thousand times smaller. The concrete number worth having is that a 110M BERT-family embedding model still competes at the top of MTEB against models with fifty times the parameters.

### Why MLM doesn't generate
The model is trained to fill in the middle, not extend the end. To generate, you'd need to autoregressively mask one position at a time and fill it — slow and unnatural. **Modern LLMs need to generate**, so they use CLM.

> **Saying it out loud.** Because it was trained to patch holes in an otherwise-complete sentence, and generation means the entire right side is missing — a situation it never saw in training. You could fake it by appending a MASK, predicting it, appending another, and re-running, but each step costs a full bidirectional forward pass because there's no KV cache to reuse when every representation depends on every other. So you get quadratic cost per token plus an out-of-distribution input. That's not a tuning problem, it's structural, and it's the single reason the field moved to decoder-only.

### MLM's other limitations

- **Train-test mismatch.** `[MASK]` tokens appear during training but not at inference, so the model sees something different. The 80/10/10 mix mitigates but doesn't eliminate this.
- **Inefficient.** Only 15% of tokens contribute to the loss; the rest is "wasted" forward-pass compute.
- **Doesn't learn long-range generation patterns.** The mask covers a token; surrounding tokens give strong signal. The model never has to predict from scratch.

> **Saying it out loud.** The biggest one is sample efficiency: you're paying for the whole sequence and learning from a seventh of it. And you can't just crank the mask rate — go much past 15 percent and there isn't enough surviving context to make the prediction solvable, so quality drops. There's also a subtler issue, that filling a single-token hole surrounded by intact text is often trivially easy, so the gradient signal is weak even where it exists. Span corruption and ELECTRA are both direct responses to that tension between signal density and task difficulty.

### Where MLM still wins

- Embedding models (BERT, sentence-BERT, BGE, etc.).
- Classification and NER on rich representations.
- Some retrieval models.

Encoder-only LLMs are not gone; they're just niche compared to decoder-only LLMs.

> **Saying it out loud.** Anywhere the deliverable is a vector rather than a sentence. Embeddings for search, document classification, entity extraction — these all want a rich representation of text you already have, not a continuation of it. And the economics are decisive: a small encoder runs in milliseconds on a CPU, while calling an LLM to embed every document in a corpus is absurd. So encoders aren't obsolete, they just serve a different part of the stack, usually as the retrieval layer feeding a generative model.

---

## 4. NSP (Next Sentence Prediction) — and why it died

Original BERT included an auxiliary objective: given two sentences, predict if sentence B follows sentence A in the original text.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}, \qquad \mathcal{L}_{\text{NSP}} = -\log P(\text{is-next} \mid \text{sentence}_A, \text{sentence}_B)
$$

50% of training pairs were "next sentence"; 50% were random.

> **Saying it out loud.** NSP was BERT's second objective: show two segments, predict whether B actually followed A, with half the pairs randomly swapped in. The motivation was reasonable — the target tasks were things like entailment and question answering, which involve reasoning across two pieces of text, and masked prediction alone is very local. It's cheap too, just a binary head on the CLS token. It's a good lesson in why plausible ideas need ablations, because it turned out to contribute nothing.

### Why NSP was thought to help
Was supposed to capture sentence-level relationships, useful for downstream tasks like NLI (Natural Language Inference) and QA.

### Why it was removed
RoBERTa (Liu et al. 2019) showed: **NSP doesn't help**. Removing it improves downstream performance. The task is too easy (random sentences are trivially distinguishable from continuation) and provides almost no useful signal.

Modern encoder LMs don't use NSP. Some models (XLNet, ALBERT) propose alternative sentence-level objectives but none have caught on.

> **Saying it out loud.** RoBERTa ablated it and found removing NSP made downstream results better, not worse. The reason is that the task has a shortcut: a random negative sentence almost always comes from a different document, so the model can answer purely by noticing the topic changed, without learning anything about how sentences connect. Once a task is solvable by a shortcut it stops producing useful gradient. ALBERT's fix is instructive — sentence-order prediction uses two genuinely consecutive sentences and just swaps them, so topic overlap can't leak the answer, and that version does help a little.

### Lesson
"This auxiliary objective sounds reasonable" is not a good reason to add it. Validate empirically.

> **Saying it out loud.** The lesson is that 'this auxiliary objective sounds reasonable' is worth nothing without an ablation. NSP survived in the most-cited paper in modern NLP for a year before anyone checked, and when they checked it was actively harmful. The general failure mode is an auxiliary task with an unintended shortcut — the model solves it the cheap way and you learn nothing while spending capacity. Whenever you add a term to a loss, the first experiment should be turning it back off.

---

## 5. Span corruption (T5, BART)

Mask contiguous spans of tokens (~3 tokens average); replace each span with a sentinel; predict the masked spans as the output.

```
Input:  "The <X> jumped over the <Y> dog"
Target: "<X> quick brown fox <Y> lazy <eos>"
```

T5's flagship objective. Encoder-decoder architecture.

> **Saying it out loud.** Span corruption is masked language modeling that hides whole phrases instead of scattered single words. T5 drops spans averaging about three tokens, replaces each with one sentinel marker, and has the decoder emit only the missing pieces. The point is difficulty — recovering a single word from intact context is often trivial, but recovering 'quick brown fox' takes real understanding of the sentence. It also compresses the target nicely, since you never regenerate the parts that weren't corrupted.

### Pros
- More efficient than MLM (whole spans contribute, not single tokens).
- Better captures phrase-level semantics.
- Works well with encoder-decoder.

### Cons
- Encoder-decoder is heavier than decoder-only.
- Span corruption doesn't directly support free-form generation as cleanly as CLM.
- Mostly superseded by CLM for general-purpose LLMs.

> **Saying it out loud.** The tradeoffs are architectural more than conceptual. An encoder-decoder is roughly twice the parameters for the same depth and adds cross-attention to serve, so you're paying more for the same capacity. And free-form open-ended generation isn't what span corruption trains — the model learns to produce short fillers tagged by sentinel, not to keep writing indefinitely. For a purpose-built task like summarization that's fine, and T5 is still excellent there. For a general assistant, one causal stack doing next-token prediction is simpler and scales better, which is why the field moved.

### Status
T5 family still used (especially Flan-T5 for fine-tuning). Span corruption variants in some research (UL2's R-denoising). Not the modern default.

> **Saying it out loud.** ELECTRA lives on in efficient encoders and nowhere else, for one structural reason: a model whose output head says real-or-fake per token has no way to generate. That was fine when the field ran on encoders plus fine-tuning, and it became disqualifying the moment everyone wanted a model that writes. It's the clearest case in this document of a genuinely better objective losing on requirements rather than merit.

---

## 6. PrefixLM (T5.1.1, GLM)

A hybrid: bidirectional attention over the **prefix**; causal attention for the **target** to be generated.

```
For input "Translate to French: The cat":
  - "Translate to French: The cat" gets bidirectional attention
  - The model generates the French translation autoregressively
  - During generation: causal mask within the target portion
```

> **Saying it out loud.** PrefixLM is the obvious compromise: read the prompt with full bidirectional attention, then generate the answer causally, all in a single stack. The logic is hard to argue with — you already have the entire prompt, so why force the model to read it left to right? It never dominated for a very unglamorous reason: the attention mask now depends on where the prefix ends, which varies per example, and that fights every optimized attention kernel and complicates KV caching. Meanwhile plain causal models got so good at using prompts that the theoretical gain shrank with scale, so you'd be taking on real infrastructure pain for a benefit nobody has demonstrated at frontier scale.

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

> **Saying it out loud.** UL2's pitch is to stop choosing one corruption scheme and train on several, with a mode token telling the model which game it's playing. Short spans teach local structure, long prefixes teach generation, and extreme corruption forces the model to invent rather than interpolate. At inference you can pick the mode that fits your task. It got a lot of research attention and zero frontier adoption, which fits the pattern across this whole document — sophisticated multi-objective schemes keep losing to plain next-token prediction with more data, because the simple objective scales more predictably and the complexity has to earn its keep at every step.

### Why this idea
Each denoiser teaches different skills: short masks teach local pattern; long masks teach long-range dependencies; extreme corruption teaches imagination/generation.

> **Saying it out loud.** The core argument is signal per FLOP, and the negatives being plausible rather than random is what makes it work. Because the fake tokens come from a trained generator, the model has to make genuinely fine distinctions rather than spotting obvious nonsense. The published result is matching BERT's downstream quality at roughly a quarter of the compute. The cost is a more complicated training setup — two models whose relative sizes you have to tune, with the generator typically a quarter to a half the discriminator's size, and making the generator too strong actually hurts.

### Status
Research-prominent (UL2, Flan-UL2). Not adopted at frontier scale by major labs.

---

## 8. ELECTRA: replaced token detection

> **In plain language.** ELECTRA changes the game from fill-in-the-blank to spot-the-fake. A small helper model quietly swaps some words for plausible substitutes, and the main model has to flag which words were tampered with. The formula below is just standard binary cross-entropy applied at every position.

Clark et al. 2020. Different from masked-then-predict:

1. A small **generator** model fills in masked tokens (like BERT).
2. A larger **discriminator** model is trained to detect which tokens were replaced (binary classification per token).

The discriminator $D(t)$ outputs the probability that token $t$ was replaced. Trained with binary cross-entropy over all positions:

$$
\mathcal{L} = -\,\mathbb{E}\!\left[\sum_t \big(\mathbf{1}[\text{token}_t \text{ replaced}] \log D(t) + \mathbf{1}[\text{token}_t \text{ original}] \log(1 - D(t))\big)\right]
$$

> **Saying it out loud.** ELECTRA replaces fill-in-the-blank with spot-the-fake. A small generator fills the masked positions with plausible guesses, and the main model looks at every token in the result and says real or replaced. That's a binary decision at every position instead of a vocabulary-sized softmax at 15 percent of positions, so you extract about seven times more signal from the same forward pass. It's GAN-shaped but not adversarial — the generator is trained on ordinary masked language modeling, because you can't backpropagate through discrete token sampling. You keep the discriminator and throw the generator away.

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

> **Saying it out loud.** All the code is doing is lining up predictions with targets. The model emits a distribution at every position, but the prediction at position $t$ is about position $t+1$, so you drop the last logit and the first token and then run one flat cross-entropy over everything. The bug everyone hits once is forgetting that shift, and the symptom is memorable — training loss collapses toward zero in a few hundred steps because the model just learned to copy the current token, and you don't find out until generation is nonsense. If a loss curve looks too good, check the shift before anything else.

### The "shift by one" gotcha
A common interview question: "Why does CLM training use `logits[:-1]` and `targets[1:]`?" Because at position `t`, the model's logits should predict the token at position `t+1`. The shift aligns predictions with their targets.

### Padding handling
Ignore padding positions in the loss:
```python
mask = tokens != pad_token_id
loss = (per_token_loss * mask).sum() / mask.sum()
```
Otherwise the model gets noisy gradients on padding tokens.

> **Saying it out loud.** Mask padding out, and pay attention to the denominator. If padding enters the loss you're teaching the model to predict filler, and worse, your reported loss depends on how much padding happened to land in the batch, so runs stop being comparable. In PyTorch the clean route is setting padded targets to $-100$, which cross-entropy skips by default. If you build your own mask, divide by the number of real tokens rather than the tensor size — dividing by the wrong count silently rescales your gradient with batch composition.

### Loss masking for SFT
For supervised fine-tuning on (prompt, response) pairs: only compute loss on response tokens, not prompt tokens. The prompt is "given," only the response is "predicted." Masking the prompt portion is a critical detail.

> **Saying it out loud.** During fine-tuning you only train on the response, never the prompt. The prompt is supplied by the user at inference, so spending gradient on learning to generate it is wasted at best. Skip this and you get a specific, recognizable failure: the model starts writing its own user turns, or tacks a hallucinated follow-up question onto its answers. It matters most when prompts are long and responses are short, because then most of your gradient is going into the half you don't care about.

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

> **Saying it out loud.** The surprising thing is that nobody trained for this. Next-token prediction over a huge messy corpus turns out to be secretly a course in pattern continuation, because the internet is full of lists, translations, and question-answer pairs where the right continuation depends on the format established above it. A few-shot prompt is just a prefix with a very obvious pattern. Mechanistically, Olsson and colleagues found induction heads — circuits that locate where a token appeared earlier and copy whatever followed it — and those heads form abruptly during training, showing up as a visible bump in the loss curve. That's the cleanest example we have of a capability emerging from an objective that never asked for it.

---

## 11. Auxiliary objectives sometimes added to LLMs

Beyond pure CLM, some recipes add:
- **Multi-token prediction** (Gloeckle et al. 2024, used in DeepSeek-V3): predict the next `k` tokens at each position with `k` separate heads. Better signal density per token; faster speculative decoding via the auxiliary heads.
- **Contrastive losses** for embedding training (e.g., E5, BGE).
- **Code-specific objectives** for code models (in-filling losses, fix-the-bug objectives).

These are mostly research-stage at frontier scale. Pure CLM remains the workhorse.

> **Saying it out loud.** The most interesting one right now is multi-token prediction: predict the next $k$ tokens from each position with $k$ small heads, on the theory that pure next-token prediction is myopic and lets the model coast on local grammar without planning. DeepSeek-V3 uses it and reports better quality at equal compute. The kicker is practical rather than theoretical — those extra heads are a built-in draft model, so you get speculative decoding roughly doubling inference speed without training a separate drafter. Everything else on the list is domain-specific: contrastive terms if you want embeddings, fill-in-the-middle if you want a code model that can complete inside a function.

---

## 12. Loss function math

> **In plain language.** This section is the algebra behind the one loss everything uses. It rewrites cross-entropy in terms of raw logits, points out the one term that can overflow, and gives the standard fix.

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

> **Saying it out loud.** Written out in logits, the loss is just the true token's logit minus the log-sum-exp of all of them, negated. The log-sum-exp term is where the numerical danger lives — exponentiating a logit of a few hundred gives you infinity and everything downstream turns to NaN. The fix is to subtract the largest logit first, which changes nothing mathematically since it cancels between numerator and denominator, but guarantees the biggest exponent is zero. This matters concretely in mixed precision, where fp16 overflows above about 65,000, meaning a logit past roughly 11 would kill your run. And the gradient stays beautifully simple through all of it: predicted probability minus the one-hot target.

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
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).
