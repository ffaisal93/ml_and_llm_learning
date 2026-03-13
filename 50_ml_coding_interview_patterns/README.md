# Topic 50: ML Coding Interview Patterns

## What You'll Learn

This topic is about writing correct code quickly.

That is different from writing a polished library.

You will learn small, reusable patterns for:
- numerical stability
- masking
- batching and padding
- vectorized distance computation
- top-k and top-p filtering
- simple beam search logic
- k-means update steps

## Why This Topic Exists

Many candidates understand the idea but still struggle in coding rounds because they:
- forget edge cases
- write too much code
- lose track of tensor shapes
- do not know the standard stable version of a function

The goal here is to give you compact templates you can rewrite from memory.

## Coding Round Strategy

When the interviewer gives a coding task:

1. Restate the input and output shapes
2. Start with the simplest correct version
3. Add the main numerical-stability fix
4. Mention runtime and memory cost
5. Handle one or two obvious edge cases

That is usually better than trying to impress with clever code.

## Core Patterns

### 1. Stable Softmax

Never write:

`np.exp(x) / np.sum(np.exp(x))`

Write:

`x_shifted = x - max(x)`

Then exponentiate.

Reason:
- avoids overflow
- same mathematical result

### 2. Masked Softmax

This shows up constantly in attention.

Pattern:
- set masked positions to a very negative number
- softmax over the full row

That ensures masked positions get probability near zero.

### 3. Pairwise Distances

For KNN, k-means, clustering, or retrieval, avoid Python loops if possible.

Vectorized squared distances:

`||x - c||^2 = ||x||^2 + ||c||^2 - 2 x dot c`

This is one of the most useful coding-round tricks.

### 4. Top-k / Top-p Filtering

LLM interviews often ask about decoding.

You should know:
- greedy: take argmax
- top-k: keep only highest-k logits
- top-p: keep smallest set whose cumulative probability exceeds threshold

### 5. Beam Search Skeleton

You do not need a full production decoder. You need the loop structure:

1. keep active beams
2. expand with next-token candidates
3. update cumulative scores
4. keep best `beam_size`

### 6. Padding and Batch Iteration

Simple utilities like `pad_sequences` and `batch_iterator` are surprisingly useful in coding rounds.

They also show that you understand how real training code is wired together.

## Boilerplate Code

See [interview_patterns.py](/Users/faisal/Projects/ml_and_llm_learning/50_ml_coding_interview_patterns/interview_patterns.py) for compact implementations of:

- Stable softmax
- Masked softmax
- Causal mask
- Pairwise squared distances
- Batch iterator
- Sequence padding
- Top-k logits filter
- Top-p logits filter
- One k-means update step

These are meant to be memorized as patterns, not copied blindly.

## What to Practice Saying Out Loud

1. Why do we subtract the max in softmax?
2. How does a causal mask enforce autoregressive generation?
3. Why are vectorized distance formulas faster than nested loops?
4. What is the difference between top-k and top-p?
5. In beam search, why do we track cumulative log probabilities instead of raw probabilities?

## Interview Advice

- Prefer short helper functions over one giant function
- Say shapes while you code
- If you are unsure, write the simplest loop version first, then vectorize
- Mention numerical stability before the interviewer has to ask

## Next Steps

After this topic:
- Use Topic 51 for LLM research interview prep, evaluation, ablations, and paper discussion
