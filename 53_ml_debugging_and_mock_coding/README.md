# Topic 53: ML Debugging and Mock Coding

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

## Debugging Mindset

Use this order:

1. Check the data
2. Check the target
3. Check tensor shapes
4. Check loss definition
5. Check scale and numerical stability
6. Check whether parameters are actually updating

This sequence avoids random guessing.

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

## Timed Coding Strategy

When the problem starts, do this:

1. State assumptions
2. Write a simple correct version
3. Mention edge cases
4. Improve stability or efficiency
5. Give runtime

That pattern is reliable under pressure.

## Files in This Topic

- [debugging_patterns.py](/Users/faisal/Projects/ml_and_llm_learning/53_ml_debugging_and_mock_coding/debugging_patterns.py): small bug patterns and checks
- [mock_questions.md](/Users/faisal/Projects/ml_and_llm_learning/53_ml_debugging_and_mock_coding/mock_questions.md): timed coding and debugging prompts

## What to Practice Saying Out Loud

1. If loss is flat, what are your first five checks?
2. If accuracy is 99.9%, why might that be wrong?
3. If attention output is nonsense, which tensor shapes would you inspect first?
4. Why does clipping help with exploding gradients?
5. Why can the wrong loss/activation pair silently break learning?
