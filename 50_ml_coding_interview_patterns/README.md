# Topic 50: ML Coding Interview Patterns

> 🔥 **For interviews, read these first:**
> - **`CODING_PATTERNS_DEEP_DIVE.md`** — frontier-lab deep dive: stable softmax + log-sum-exp, scaled dot-product attention with masking, multi-head, top-k/top-p sampling, beam search, K-means, padding/masking, vectorized cosine similarity, logistic regression, backprop from scratch.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions + 8 must-code patterns to drill.

## What You'll Learn

This topic is about writing correct ML code quickly under interview pressure.

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

Many candidates understand the algorithmic idea but still struggle in coding rounds because they:
- forget edge cases
- write too much code
- lose track of tensor shapes
- miss the standard numerically stable form

The goal here is to give you compact templates you can rewrite from memory.

## Core Intuition

Most ML coding rounds are not testing whether you can invent a new algorithm.

They are testing whether you know the small implementation patterns that keep common ML code:
- correct
- numerically stable
- shape-safe
- easy to explain

If you remember the invariant, the code becomes much easier to write.

Examples:
- softmax should produce a valid probability distribution without overflow
- masking should make invalid positions contribute zero probability
- beam search should compare partial sequences fairly
- batching should preserve alignment between tokens, masks, and labels

The real skill is to remember the invariant, then write the shortest code that preserves it.

## Coding Round Strategy

When the interviewer gives a coding task:

1. Restate the input and output shapes.
2. Start with the simplest correct version.
3. Add the main numerical-stability fix.
4. Mention runtime and memory cost.
5. Handle one or two obvious edge cases.

That is usually better than trying to impress with clever code.

## Technical Details Interviewers Often Want

### Stable Softmax

Softmax is:

`softmax(x_i) = exp(x_i) / sum_j exp(x_j)`

The problem is overflow when some logits are large.

So the stable version is:

`softmax(x) = exp(x - max(x)) / sum exp(x - max(x))`

Subtracting the same constant from every logit does not change the final probabilities, so this is mathematically equivalent but much more stable.

### Masked Softmax

The main idea is:
- valid positions should compete with each other
- invalid positions should get probability `0`

The standard pattern is:
- fill masked positions with a very negative number
- apply softmax over the full row

That works well, but there is one important edge case: if every position in a row is masked, the row can become undefined. In real systems you often need an explicit guard for fully masked rows.

### Pairwise Distances

For data matrix `X` with shape `(n, d)` and centers `C` with shape `(k, d)`, the vectorized squared-distance matrix has shape `(n, k)`.

The useful identity is:

`||x - c||^2 = ||x||^2 + ||c||^2 - 2 x^T c`

This avoids nested Python loops and is usually much faster. Because of floating-point rounding, it can produce tiny negative values, so clipping to at least `0` is a reasonable defensive step.

### Top-k and Top-p

Top-k keeps a fixed number of candidates.

Top-p keeps the smallest set whose cumulative probability exceeds threshold `p`.

That means:
- top-k gives a fixed-width shortlist
- top-p adapts to confidence

The common interview bug is to threshold raw probability values instead of cumulative sorted probability mass. That is not nucleus sampling.

### Beam Search Scores

Beam search should usually accumulate log probabilities, not raw probabilities.

Why:
- multiplying many probabilities underflows quickly
- adding log probabilities is stable
- log space preserves ranking because `log` is monotonic

A natural follow-up is length bias. Shorter sequences can be unfairly favored, so some beam-search variants use length normalization.

## Core Patterns

### 1. Stable Softmax

Never write:

`np.exp(x) / np.sum(np.exp(x))`

Write:

`x_shifted = x - max(x)`

Then exponentiate.

Reason:
- avoids overflow
- gives the same mathematical result

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
- top-p: keep the smallest set whose cumulative probability exceeds threshold

### 5. Beam Search Skeleton

You do not need a full production decoder. You need the loop structure:

1. Keep active beams.
2. Expand with next-token candidates.
3. Update cumulative scores.
4. Keep the best `beam_size`.

### 6. Padding and Batch Iteration

Simple utilities like `pad_sequences` and `batch_iterator` are surprisingly useful in coding rounds.

They also show that you understand how real training code is wired together.

## Common Failure Modes

### 1. Softmax Overflow or NaNs

This usually happens when you exponentiate raw logits directly.

The fix is to subtract the maximum before exponentiating and to think about fully masked rows.

### 2. Wrong Mask Semantics

Candidates often know masking matters but still get one of these wrong:
- wrong mask orientation
- broadcasting over the wrong axis
- mixing up whether `True` means "keep" or "mask out"

This is one of the most common attention bugs in interviews.

### 3. Incorrect Top-p Logic

Another common mistake is to keep tokens whose individual probabilities exceed `p`.

Top-p is based on cumulative mass after sorting, not individual probability thresholds.

### 4. Beam Search Using Raw Probabilities

Multiplying probabilities across many decoding steps quickly causes underflow.

If a candidate does not move to log space, that often signals they know the concept but not the stable implementation.

### 5. Empty Clusters in k-means

A k-means update can produce a cluster with no assigned points.

If you take the mean of an empty set without handling it, the code breaks or silently returns garbage.

### 6. Padding Mismatch

Candidates often pad sequences correctly but forget to update:
- attention masks
- sequence lengths
- labels aligned with padding

That causes downstream bugs even when the padding helper itself looks fine.

## Edge Cases and Follow-Up Questions

### What if every position is masked?

You need a defined behavior.

Possible approaches:
- return zeros
- skip the row
- unmask a fallback position if the task requires one valid target

The important point is to notice that naive masked softmax can fail here.

### What if `k` is larger than vocabulary size in top-k?

Clamp `k` to vocabulary size.

Say that explicitly instead of assuming the input is always valid.

### What if `p` in top-p is tiny or exactly `1`?

If `p` is tiny, you usually still want to keep at least one token.

If `p = 1`, top-p should effectively keep all tokens.

### What if two distances are equal?

For nearest-neighbor style logic, ties need a policy:
- deterministic tie-break
- first occurrence
- break by label frequency

The interviewer usually cares more that you notice the ambiguity than which policy you choose.

### What if a batch contains all-padding examples?

That can break reductions, attention masks, and loss computation.

In real systems you often filter empty sequences early or explicitly skip those rows during reduction.

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

The code is intentionally short. That makes it good interview boilerplate, but it is still simplified compared with production code. In a real codebase you would usually add stronger input validation and more explicit tests for corner cases.

## What to Practice Saying Out Loud

1. Why do we subtract the max in softmax?
2. How does a causal mask enforce autoregressive generation?
3. Why are vectorized distance formulas faster than nested loops?
4. What is the difference between top-k and top-p?
5. In beam search, why do we track cumulative log probabilities instead of raw probabilities?
6. What breaks if an attention row is fully masked?
7. Why can top-p return a different number of tokens for different contexts?
8. What do you do when a k-means cluster gets no assigned points?

## Interview Advice

- Prefer short helper functions over one giant function.
- Say shapes while you code.
- If you are unsure, write the simplest loop version first, then vectorize.
- Mention numerical stability before the interviewer has to ask.
- If your code depends on a tie-break or fallback policy, say that policy explicitly.

## Next Steps

After this topic:
- Use Topic 51 for LLM research interview prep, evaluation, ablations, and paper discussion.
