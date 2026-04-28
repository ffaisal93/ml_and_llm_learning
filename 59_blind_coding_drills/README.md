# Topic 59: Blind Coding Drills

## What You'll Learn

This topic is for coding from memory without search, autocomplete, or notes.

You will practice:
- writing short correct implementations from scratch
- stating runtime and edge cases
- handling interruptions
- recovering after small mistakes

## Why This Matters

Knowing how code works is not the same as being able to produce it under pressure.

You need reps where:
- you cannot search
- you cannot copy from earlier files
- you must explain while writing

## Core Intuition

Blind coding drills are about reducing dependence on comfort tools.

In an interview, you may lose:
- autocomplete
- syntax memory
- confidence after one small mistake
- your usual debugging rhythm

The goal of this topic is to make standard implementations feel familiar enough that you can still write them when your working memory is busy.

This is why the drills are short.

You are practicing retrieval under pressure, not full project development.

## Files in This Topic

- [drills.md](/Users/faisal/Projects/ml_and_llm_learning/59_blind_coding_drills/drills.md): timed prompts
- [blind_drill_picker.py](/Users/faisal/Projects/ml_and_llm_learning/59_blind_coding_drills/blind_drill_picker.py): random drill selector

## Technical Details Interviewers Often Want

### State Shapes and Runtime While Coding

In ML interviews, code quality is only part of the evaluation.

Interviewers often want to hear:
- expected input shapes
- output shapes
- runtime complexity
- memory trade-offs

Saying those while coding makes your thinking easier to follow.

### Write the Simplest Correct Version First

A loop-based implementation that is correct is usually better than a broken vectorized implementation.

Once the baseline is correct, you can improve:
- numerical stability
- vectorization
- edge-case handling

### Use Small Helpers

Short helper functions often improve correctness under pressure.

Examples:
- `stable_softmax`
- `pad_sequences`
- `top_k_filter`

Those helpers reduce mental load and make it easier to explain the code.

## Common Failure Modes

### 1. Over-Optimizing Too Early

Candidates sometimes jump into advanced vectorization before confirming the semantics.

That often causes shape or indexing bugs.

### 2. Forgetting Edge Cases

Common misses include:
- empty input
- one-element input
- ties
- all-padding rows
- `k` larger than the number of items

### 3. Writing Code Without a Verbal Plan

If you start typing before stating the function contract, it is easier to drift into the wrong problem.

### 4. Panic After a Small Syntax Error

One small bug can cause candidates to abandon an otherwise good approach.

These drills are meant to reduce that reaction.

## Edge Cases and Follow-Up Questions

### What if you run out of time?

Finish the core logic, then explain the missing edge-case handling verbally.

That is usually better than leaving the main path incomplete.

### What if the interviewer asks for a vectorized version after you wrote loops?

Explain the loop version first, then show the vectorized identity you would use.

### What if you are unsure about a library call?

State the intended behavior and write the surrounding logic clearly.

Interviewers often care more about the algorithm than exact library trivia.

## Rules

For each drill:

1. Set a timer.
2. Close notes.
3. Do not open related files.
4. Code from memory.
5. Explain the solution out loud.
6. Review against the repo only after finishing.

## Target Skill

The goal is not perfect code elegance.

The goal is that your first answer is correct enough, stable enough, and clearly reasoned.

## What to Practice Saying Out Loud

1. What are the input and output shapes of this function?
2. What is the simplest correct implementation I can write first?
3. What edge case is most likely to break this code?
4. What is the runtime and where is the bottleneck?
5. If I had two more minutes, what improvement would I add?
