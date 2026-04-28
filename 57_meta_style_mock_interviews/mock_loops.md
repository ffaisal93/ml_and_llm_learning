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
