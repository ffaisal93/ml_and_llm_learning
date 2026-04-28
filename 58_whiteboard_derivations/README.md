# Topic 58: Whiteboard Derivations

> 🔥 **For interviews, read these first:**
> - **`WHITEBOARD_DERIVATIONS_DEEP_DIVE.md`** — meta-collection of 13 must-master derivations: backprop, attention, OLS, logistic gradient, KL, EM, PCA via SVD, SVM dual, RoPE rotation, DPO, ELBO, bias-variance, information gain. Each with step-by-step proof + cross-references.
> - **`INTERVIEW_GRILL.md`** — 40 active-recall questions verifying you can do each derivation cold.

## What You'll Learn

This topic is for derivations you should be able to do cleanly from memory in an interview.

You will practice:
- logistic regression gradient
- softmax plus cross-entropy gradient
- Bernoulli and Gaussian MLE
- sample variance `n` vs `n - 1`
- attention score shapes
- bias-variance decomposition intuition
- confidence interval structure

## Why This Matters

In theory-heavy interviews, the interviewer is rarely asking for a full formal proof.

They are usually checking:
- can you start from first principles
- can you keep the algebra organized
- can you track assumptions and shapes
- can you explain each step out loud

## Core Intuition

Most whiteboard derivations in interviews are not really about memorizing a final formula.

They are about whether you can move from:
- model definition
- objective
- derivative or estimator
- interpretable final form

under time pressure and without losing the thread.

The strongest derivation answers are structured.

They do not jump directly to the result.

They keep a visible chain:
- what is given
- what is being optimized
- which rule is applied
- what the final expression means

## Files in This Topic

- [derivations.md](/Users/faisal/Projects/ml_and_llm_learning/58_whiteboard_derivations/derivations.md): step-by-step derivations
- [memory_skeletons.md](/Users/faisal/Projects/ml_and_llm_learning/58_whiteboard_derivations/memory_skeletons.md): compact memory aids

## Technical Details Interviewers Often Want

### Start from the Objective

Candidates often know the final gradient but cannot reconstruct it.

A stronger approach is:
- write the likelihood or loss
- simplify if needed
- differentiate carefully
- interpret the result

For logistic regression, for example, the interviewer often cares that you can show why the gradient becomes prediction minus target times input.

### Keep Shapes Visible

For attention and neural-network derivations, shapes are part of the derivation.

If you cannot say what the shape of the score matrix or gradient is, the interviewer may doubt whether you really understand the operation.

### Explain the Meaning of the Result

Do not stop at the algebra.

Examples of good interpretive endings:
- "This gradient says we push the prediction toward the label."
- "This estimator equals the sample mean because the Gaussian log-likelihood reduces to squared error."
- "Using `n - 1` corrects the downward bias in sample variance estimation."

That final interpretive sentence often matters as much as the derivation itself.

## Common Failure Modes

### 1. Jumping to a Memorized Result

If the interviewer asks for a derivation and you only state the answer, it is hard for them to judge whether you could recover it under pressure.

### 2. Losing the Sign

This happens often in log-likelihood and cross-entropy derivations.

Always track whether you are maximizing likelihood or minimizing negative log-likelihood.

### 3. Misusing the Chain Rule

This is one of the most common issues in softmax, sigmoid, and neural-network gradients.

### 4. Ignoring Dimensions

An algebraic expression that has the wrong shape is usually a sign that the derivation went off track.

### 5. Stopping Before Interpretation

Even a correct derivation can sound incomplete if you do not explain what the expression means.

## Edge Cases and Follow-Up Questions

### What if you forget one algebraic step?

State the step you do remember and continue from the objective rather than freezing.

### What if the interviewer asks for intuition instead of algebra?

Give the intuition first, then say you can derive it formally if they want.

### What if the derivation depends on an assumption?

Say the assumption explicitly, such as:
- i.i.d. samples
- Gaussian noise
- differentiability
- full-rank covariance if relevant

### What if the interviewer asks why `n - 1` appears?

Do not just say "unbiased estimator."

Explain that the sample mean is estimated from the same data, which removes one degree of freedom and causes the naive `1/n` estimator to underestimate variance on average.

## How to Practice

1. Cover the answer.
2. Re-derive it on paper.
3. Say each step out loud.
4. Compare with the full derivation.

## Standard Pattern

When deriving under pressure:

1. Write the model.
2. Write the objective.
3. Differentiate the outermost term first.
4. Apply chain rule carefully.
5. Simplify.
6. Check dimensions or interpretation.

## What to Practice Saying Out Loud

1. What is the objective I am differentiating?
2. Which assumption makes this estimator form valid?
3. What shape should this gradient or tensor have?
4. What does the final expression mean in plain English?
5. If I forget the final formula, can I rebuild it from first principles?
