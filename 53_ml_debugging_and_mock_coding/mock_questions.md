# Mock Coding and Debugging Questions

These are designed for timed practice. Try to answer each in 10 to 20 minutes.

## Timed Coding

### 1. Logistic Regression

Implement binary logistic regression with:
- sigmoid
- binary cross-entropy
- one gradient descent step

What the interviewer is testing:
- vectorization
- stability
- loss/gradient correctness

### 2. K-Means One Iteration

Given points and current centers:
- assign each point to nearest center
- recompute means

What the interviewer is testing:
- distance computation
- cluster updates
- edge cases for empty clusters

### 3. Attention Mask

Implement masked softmax for attention.

What the interviewer is testing:
- correct masking convention
- softmax axis
- numerical stability

### 4. Top-p Sampling

Given logits and threshold `p`:
- convert to probabilities
- sort by probability
- keep the smallest set whose cumulative mass reaches `p`

What the interviewer is testing:
- sorting
- cumulative probability logic
- corner cases

## Debugging

### 5. Loss Is NaN

Your training loop starts returning NaN after a few iterations.

Explain your debugging order.

Expected discussion:
- check learning rate
- check log/division operations
- inspect activations and gradients
- check normalization and masking
- clip gradients if needed

### 6. Validation Accuracy Is Too Good

You see 99.8% validation accuracy on a hard real-world problem.

Explain what is suspicious and how you would verify it.

Expected discussion:
- leakage
- duplicates
- future information
- preprocessing fit on all data
- label leakage

### 7. Transformer Output Looks Wrong

Your attention implementation runs, but the output is nonsense.

Expected checks:
- shape of Q, K, V
- transpose placement
- mask orientation
- scale by `sqrt(d_k)`
- softmax axis

### 8. Model Does Not Learn

Loss barely changes for 1,000 steps.

Expected checks:
- gradients zero or tiny
- optimizer step missing
- frozen parameters
- bad initialization
- wrong target type or shape

## Research-Oriented Debugging

### 9. Benchmark Improves Only on One Seed

Your method beats baseline on one seed but not others.

What is the right conclusion?

Expected answer:
- do not claim robust improvement yet
- report mean and variance across seeds
- inspect whether the gain is real or fragile

### 10. New Retriever Improves Recall@10 but Hurts End-to-End QA

How can that happen?

Expected answer:
- retrieval metric and generation metric are not identical
- retrieved context may be noisy or poorly ordered
- context packing may hurt answer synthesis
- the model may ignore retrieved text
