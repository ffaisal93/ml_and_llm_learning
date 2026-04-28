# README Quality Standard

This file defines what a strong topic README should contain in this repository.

The problem this standard solves is simple:

Many topic READMEs explain the headline concept, but do not go far enough on:
- intuition
- technical detail
- common mistakes
- edge cases
- follow-up interview questions

The goal is that each topic should feel like a compact teaching note plus interview-prep guide, not just a summary with code pasted into it.

## What a Good Topic README Must Include

### 1. Plain-English Intuition

Before formulas, answer:
- what problem is this solving?
- why do we need it?
- what breaks without it?

If a reader cannot explain the intuition after reading the README, the explanation is not complete enough.

### 2. Technical Core

A strong README should explain:
- the key mathematical object
- the main algorithmic steps
- the runtime or memory trade-off if relevant
- the assumptions hidden in the method

The point is not to include every theorem. The point is to explain the real technical mechanism cleanly.

### 3. When to Use It / When Not to Use It

Good interview answers are comparative.

A README should usually answer:
- when is this method a good fit?
- when is it a bad fit?
- what is the common alternative?

### 4. Common Failure Modes

Every important topic should have a section on what can go wrong.

Examples:
- class imbalance breaking accuracy
- numerical instability in softmax
- train/test leakage in preprocessing
- quantization hurting model quality
- weak baselines invalidating a research claim

### 5. Edge Cases

A README should explicitly discuss edge cases that interviewers like to ask about.

Examples:
- what if a class is very rare?
- what if two distributions overlap heavily?
- what if variance is zero?
- what if context length doubles?
- what if the metric improves but task quality does not?

### 6. Follow-Up Interview Questions

Each README should contain a short section like:

"What the interviewer may ask next"

This should cover 3-6 follow-up questions that naturally arise after the main explanation.

### 7. Pressure-Friendly Explanation

The explanation should not only be correct. It should also be sayable in an interview.

That means:
- short paragraphs
- explicit assumptions
- clean step-by-step structure
- easy transition from intuition to math

### 8. Boilerplate Code Context

If code is included, the README should explain:
- what the code is demonstrating
- what simplifications it makes
- what part is most important to remember

The code should not appear without surrounding guidance.

## Recommended README Structure

1. What You'll Learn
2. Why This Matters
3. Core Intuition
4. Technical Details
5. When to Use It
6. Common Failure Modes
7. Edge Cases and Follow-Ups
8. Boilerplate Code
9. What to Practice Saying Out Loud
10. Exercises / Next Steps

## Writing Style Rules

- Prefer short paragraphs over long bullet dumps
- Explain the mechanism, not just the label
- Use examples whenever the concept is abstract
- State hidden assumptions explicitly
- Include at least one edge-case discussion for interview-heavy topics
- If a comparison matters, make the comparison explicit

## Minimal Upgrade Rule

If a README is too short to rewrite fully in one pass, add at least:
- one intuition section
- one common-mistakes section
- one follow-up-questions section

That alone usually makes the document far more useful.
