# Mock Interview Loops

Each loop is designed to feel like a real technical interview segment.

Use a timer. Do not look things up while answering.

---

## Loop 1: Theory + Follow-Ups

### Prompt

Explain why logistic regression uses cross-entropy instead of MSE.

### Expected strong answer

You should connect:
- Bernoulli likelihood
- sigmoid output as probability
- MLE leading to cross-entropy
- better gradient behavior than MSE for classification

### Follow-ups

1. Derive the gradient with respect to the logits.
2. Why does the gradient simplify to `p - y`?
3. When might MSE still appear in classification work?

> **Saying it out loud.** Cross-entropy isn't a preference, it's what falls out of the model. If I say the label is a coin flip whose bias is the sigmoid of my score, then writing down the likelihood of the data and taking its negative log *is* cross-entropy — MSE would correspond to assuming Gaussian noise on a variable that's actually zero or one. The practical reason matters even more: with MSE, the sigmoid's derivative multiplies into the gradient, so a confidently wrong prediction sits in the flat tail and gets almost no gradient, and learning stalls. With cross-entropy those terms cancel and you're left with prediction minus label, so the more wrong you are the harder you're pushed. Named failure mode: MSE plus sigmoid gives you vanishing gradients exactly on the examples you most need to fix.

---

## Loop 2: Probability / Statistics

### Prompt

You have two arrays from two distributions and a new scalar value. How do you decide which source it most likely came from?

### Expected strong answer

You should cover:
- likelihood comparison
- priors
- Gaussian plug-in classification if assumptions are acceptable
- KDE fallback if distribution family is unknown
- confidence / ambiguity if overlap is high

### Follow-ups

1. What if both distributions have the same mean?
2. What if one class is much more common?
3. What if you only have a few samples?

> **Saying it out loud.** This is Bayes' rule in disguise, and I'd say that first. Fit a density to each array, evaluate the new point under both, weight each by how common that source is, and take the bigger posterior. If a Gaussian is defensible I just need a mean and a variance per group; if the samples look skewed or multimodal I'd switch to a kernel density estimate. The three follow-ups all probe the same instinct: equal means still classify fine because densities differ in the tails, an imbalanced prior can override a modest likelihood difference, and with only a handful of samples my density estimate is the weak link. That last one is the tradeoff to name — with small $n$ the parametric fit is biased but stable, while KDE is flexible and hopelessly noisy.

---

## Loop 3: Coding

### Prompt

Implement masked softmax for attention.

### Expectations

You should:
- clarify mask convention
- write a stable softmax
- use the correct axis
- mention complexity

### Follow-ups

1. How would you make it causal?
2. What bug would produce NaNs here?
3. What shape errors are common?

> **Saying it out loud.** I'd narrate this before writing a line: masked softmax means I compute the scores, set the disallowed entries to negative infinity, then softmax over the key axis — so masked positions exponentiate to exactly zero and the surviving weights still sum to one. Doing it the other way round, multiplying by zero after the softmax, leaves the denominator wrong and the rows no longer normalized. I'd state the mask convention out loud first, because half the bugs here are True-means-keep versus True-means-mask. Causal is just a lower-triangular mask. And the named failure mode: mask an entire row, every entry becomes negative infinity, softmax gives zero over zero, and you get NaN — which is why implementations use a large negative number like -1e9 rather than literal infinity.

---

## Loop 4: Debugging

### Prompt

A training loop suddenly starts returning NaN losses after a few hundred steps. Walk through your debugging plan.

### Expected strong answer

You should cover:
- inspect data and labels
- check learning rate and schedule
- inspect activation/gradient ranges
- check `log`, `exp`, division, normalization
- clip gradients if needed
- isolate the exact step where instability begins

### Follow-ups

1. What if the issue only appears in mixed precision?
2. What if train is fine but validation is NaN?
3. What if this only happens on one GPU rank?

> **Saying it out loud.** NaN after a few hundred steps tells me it's not initialization, it's an instability that finally got triggered — so my plan is to localize in time first, then in the graph. I'd log gradient norms and activation ranges every step, find the last clean step, and dump that batch, because a single outlier example or a very long sequence is the most common trigger. Then I check the usual arithmetic suspects: log of zero, exp overflow, division by a variance that collapsed, a normalization with no epsilon. Fixes in order of cheapness: clip gradients at global norm 1.0, lower the learning rate, extend warmup, restart from the last good checkpoint. The follow-ups are diagnostic gold — if it only happens in mixed precision it's FP16 range and BF16 fixes it, and if it only happens on one rank it's a data-sharding or all-reduce problem, not a math problem.

---

## Loop 5: Research Judgment

### Prompt

A new method improves perplexity but hurts exact match on downstream QA. How do you reason about that?

### Expected strong answer

You should discuss:
- training objective vs downstream metric mismatch
- calibration and decoding effects
- domain mismatch
- answer-format sensitivity
- slice analysis and error analysis

### Follow-ups

1. What ablations would you run next?
2. What if the gain only appears on one seed?
3. What if retrieval quality improved at the same time?

> **Saying it out loud.** My first line is that perplexity and exact match aren't the same objective, so this is a mismatch to explain, not a contradiction to resolve. Perplexity averages surprise over every token including all the boilerplate; exact match is a brittle all-or-nothing check on one span. The cheapest hypothesis is formatting — the model now wraps the answer in a sentence and scores zero on a string comparison — so I'd look at a fuzzy metric like token F1 first, and if F1 held while exact match fell, that's the answer. Otherwise I'd suspect the model got better calibrated and hedgier, lowering average surprise while blurring the decisive token. Named failure mode: optimizing a proxy metric while the target metric regresses, which is why you always eval both and slice the errors.

---

## Loop 6: Large-Scale Systems

### Prompt

How would you fit a larger LLM training run when you are running out of memory?

### Expected strong answer

You should discuss:
- lower batch size + gradient accumulation
- mixed precision
- activation checkpointing
- optimizer state sharding
- FSDP / ZeRO intuition
- sequence length trade-offs

### Follow-ups

1. What do you lose with checkpointing?
2. Why does Adam consume so much memory?
3. How does longer context affect memory?

> **Saying it out loud.** Memory in a training run comes in four buckets — parameters, gradients, optimizer state, and activations — and the fix depends on which one is blowing up. Adam is the surprising one: it keeps two extra tensors per parameter, so with FP32 master weights you're at roughly sixteen bytes per parameter before you've stored a single activation, which is why a one-billion-parameter model wants about sixteen gigabytes just to sit there. Activations scale with batch times sequence length, so gradient accumulation gets you the same effective batch on less memory, and activation checkpointing throws activations away and recomputes them in the backward pass. Sharding via ZeRO or FSDP splits optimizer state, gradients, and eventually parameters across ranks. The tradeoffs to name are concrete: checkpointing costs roughly 30 percent more compute, accumulation costs wall-clock time proportional to the number of micro-steps, and sharding costs communication bandwidth.

---

## Loop 7: Paper Critique

### Prompt

A paper claims a strong improvement on one benchmark. What do you need to see before you believe it?

### Expected strong answer

You should ask for:
- strong baseline
- same data and compute controls
- multiple seeds
- ablations
- slice metrics
- failure cases

### Follow-ups

1. What if the benchmark is saturated?
2. What if the paper uses a proprietary internal dataset?
3. What if the improvement is only 0.2 points?

> **Saying it out loud.** Before I believe a single-benchmark claim I want to know that the comparison was fair and the number was stable. Fair means the baseline was actually tuned and both sides got the same data and compute — a lot of reported gains are just a better-tuned method beating a lazily-run baseline. Stable means multiple seeds with a spread, because seed variance routinely exceeds the size of the improvement being claimed. Then I want to know where the gain came from: ablations to isolate the component, and slice metrics, because an aggregate win can hide a regression on the hard subset. And I'd ask about contamination, since on a public benchmark the test set may well be in the pretraining data. The number to name: a 0.2-point improvement without error bars is not evidence of anything.

---

## Loop 8: End-to-End Mixed Loop

### Prompt

Design and defend a small RAG experiment for factual QA.

### Expected strong answer

You should cover:
- baseline retriever/generator
- chunking choice
- retrieval metrics and answer metrics
- ablations
- failure taxonomy
- confidence and evaluation slices

### Follow-ups

1. How do you know whether failure is retrieval-side or generation-side?
2. Why might better Recall@10 not improve final answers?
3. What would you optimize first under latency constraints?

> **Saying it out loud.** I'd design this so that when it fails I can tell *which half* failed, because that's the whole difficulty with RAG. So I measure the two stages separately: retrieval with recall-at-k and whether the gold passage is present, generation with answer accuracy given the gold passage handed to it directly. The difference between those two tells me whether to fix the retriever or the reader. Chunking is the choice I'd defend most carefully — small chunks retrieve precisely but sever context, large chunks preserve context but dilute the embedding, and roughly 200 to 500 tokens with overlap is the usual starting point. Then a failure taxonomy: not retrieved, retrieved but ranked too low, retrieved but ignored, and generated beyond the evidence. The tradeoff to state under latency pressure is that increasing k reliably improves recall and reliably adds distractors, so end-to-end accuracy peaks at a smaller k than recall alone suggests.

