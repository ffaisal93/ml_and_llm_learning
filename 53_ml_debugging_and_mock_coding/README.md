# Topic 53: ML Debugging and Mock Coding

> 🔥 **For interviews, read these first:**
> - **`ML_DEBUGGING_DEEP_DIVE.md`** — frontier-lab deep dive: 8-layer debugging tree, loss-curve interpretation, sanity checks (overfit one batch, tiny dataset), NaN debugging (FP16/log-of-zero/anomaly detection), leakage detection, gradient checking, distribution-shift investigation.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

## What You'll Learn

This topic is for the part of interviews where something is broken and you need to reason quickly.

You will learn:
- how to debug training loops
- how to check shapes and masks
- how to catch unstable softmax or log operations
- how to reason about exploding or vanishing gradients
- how to debug evaluation bugs and leakage
- how to structure timed coding answers

## Why This Matters

Interview coding is rarely just:

"Write clean code from scratch."

Very often it is:
- "This model is not learning. What would you check?"
- "This attention code gives NaNs. Why?"
- "This metric looks too good. What is suspicious?"

That is debugging, not just implementation.

## Core Intuition

Debugging is mostly the process of shrinking the space of possible mistakes.

Weak answers jump randomly between hypotheses.

Strong answers eliminate categories of failure in a disciplined order.

Most ML bugs fall into a few buckets:
- the data is wrong
- the target is wrong
- the objective is wrong
- the shapes are wrong
- the numerics are unstable
- the optimization step is not doing what you think

That means a good debugging interview answer is not a long list of guesses.

It is a short ordered procedure that rules out the highest-probability failures first.

## Debugging Mindset

Use this order:

1. Check the data.
2. Check the target.
3. Check tensor shapes.
4. Check loss definition.
5. Check scale and numerical stability.
6. Check whether parameters are actually updating.

This sequence avoids random guessing.

## Technical Details Interviewers Often Want

### Data and Label Checks

Before touching the model, verify that the data pipeline is sane:
- input values are in the expected range
- labels correspond to the right examples
- train and evaluation splits are truly separate
- preprocessing at evaluation time matches training-time preprocessing

An interviewer often wants to see that you do not blame the optimizer before checking whether the target itself is corrupted.

### Loss and Activation Compatibility

This is one of the most common silent bugs.

Examples:
- applying softmax before a loss that already expects logits
- using mean-squared error for classification without good reason
- mismatching binary labels with multiclass output format

The key point is that the code can run without crashing and still learn the wrong thing.

### Gradient Flow

When a model is not learning, a strong answer is to inspect:
- whether gradients are zero
- whether gradients are `NaN`
- whether parameters change after `optimizer.step()`
- whether the intended parameters are actually included in the optimizer

This is more informative than saying "maybe the learning rate is wrong" and stopping there.

### Attention Debugging

For attention, four checks solve many issues:
- `Q`, `K`, `V` shapes
- whether the score matrix has the expected shape
- whether the mask broadcasts to the score shape
- whether softmax is applied over the key dimension

If any of those is wrong, the model may still run but produce meaningless attention patterns.

### Numerical Stability

NaNs usually come from a small set of operations:
- exponentials on large values
- logarithms of zero
- divisions by tiny denominators
- half-precision overflow
- invalid normalization constants

A good answer names the operation class, not just the symptom.

## Common Interview Bugs

### 1. Loss Does Not Decrease

Possible causes:
- learning rate too high or too low
- wrong labels
- no gradient flow
- optimizer not stepping
- output activation mismatched with loss

### 2. NaNs During Training

Common reasons:
- `log(0)`
- `exp(large_number)`
- division by zero
- exploding gradients
- invalid normalization

### 3. Accuracy Is Suspiciously High

Check:
- train/test leakage
- duplicates across splits
- label leakage in features
- preprocessing fit on all data

### 4. Attention Is Wrong

Check:
- mask orientation
- broadcasting shape
- scaling by `sqrt(d_k)`
- softmax axis

## Common Failure Modes

### 1. The Model Appears to Train but the Metric Is Broken

Sometimes the loss goes down because the code optimizes something real, but the evaluation metric is computed incorrectly.

Examples:
- thresholding logits incorrectly
- averaging over padded tokens
- mixing micro and macro averaging unintentionally

### 2. Leakage Hidden in Preprocessing

A classic example is fitting normalization, vocabulary construction, PCA, or imputation on the full dataset before splitting.

This can make evaluation look unrealistically strong.

### 3. Training/Eval Mode Bugs

BatchNorm and dropout behave differently in training and evaluation.

If the mode is wrong, metrics can swing dramatically even though the model code itself looks unchanged.

### 4. Parameters Not Updating

Common reasons:
- frozen parameters
- missing parameter group in the optimizer
- `zero_grad` or `step` used incorrectly
- gradient accumulation logic applied incorrectly

### 5. Shape Bugs That Broadcast Silently

This is particularly dangerous in NumPy and PyTorch because the code may run and produce outputs of the wrong meaning without raising an exception.

## Edge Cases and Follow-Up Questions

### What if the training loss decreases but validation quality is flat?

That suggests:
- overfitting
- train/eval distribution mismatch
- wrong validation metric
- label leakage in training but not validation

Do not assume optimization is the only issue.

### What if NaNs happen only in mixed precision?

Then the likely problem is not the abstract model architecture.

It is probably one of:
- reduced numeric range
- unstable gradient scaling
- half-precision overflow in activations or logits

### What if attention code runs but outputs are nonsense?

That often means a semantic shape bug:
- wrong transpose
- wrong softmax axis
- bad mask broadcast
- mixing batch and head dimensions

### What if accuracy is high but examples look obviously wrong?

Then inspect the evaluation setup itself:
- label encoding
- thresholding
- class imbalance
- leakage
- duplicate examples

### What if only one class is ever predicted?

Possible causes include:
- severe class imbalance
- threshold issue
- collapsed logits
- bad bias initialization
- loss-weighting problem

## Timed Coding Strategy

When the problem starts, do this:

1. State assumptions.
2. Write a simple correct version.
3. Mention edge cases.
4. Improve stability or efficiency.
5. Give runtime.

That pattern is reliable under pressure.

## Files in This Topic

- [debugging_patterns.py](/Users/faisal/Projects/ml_and_llm_learning/53_ml_debugging_and_mock_coding/debugging_patterns.py): small bug patterns and checks
- [mock_questions.md](/Users/faisal/Projects/ml_and_llm_learning/53_ml_debugging_and_mock_coding/mock_questions.md): timed coding and debugging prompts

These files are intentionally small and repeatable. The point is to make your debugging procedure easy to recall in an interview, not to build a large debugging framework.

## What to Practice Saying Out Loud

1. If loss is flat, what are your first five checks?
2. If accuracy is 99.9%, why might that be wrong?
3. If attention output is nonsense, which tensor shapes would you inspect first?
4. Why does clipping help with exploding gradients?
5. Why can the wrong loss/activation pair silently break learning?
6. Why can a model look stable in training but fail only at evaluation time?
7. What is the difference between a numerical bug and a statistical bug in model performance?
8. Which checks would you do before changing the model architecture?
