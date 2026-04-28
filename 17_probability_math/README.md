# Topic 17: Probability Math Q&A

> 🔥 **For interviews, read these first:**
> - **`PROBABILITY_DEEP_DIVE.md`** — frontier-lab deep dive: axioms, conditional probability, Bayes' theorem (with base-rate fallacy), expectations and variance (linearity, total expectation, total variance), common distributions, multivariate Gaussian (marginals/conditionals), LLN/CLT.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic covers common probability questions for interviews:
- Bayes' theorem
- Conditional probability
- Expected value
- Variance
- Common distributions
- Interview questions with solutions

## Why We Need This

### Interview Importance
- **Common questions**: Probability is fundamental
- **Math skills**: Shows mathematical maturity
- **Problem-solving**: Many ML problems use probability

### Real-World Application
- **ML theory**: Probability is foundation
- **Bayesian methods**: Used in many models
- **Uncertainty**: Quantifying uncertainty

## Core Intuition

Probability questions show up in ML interviews because they test whether you can reason under uncertainty.

In many cases, the interviewer is not trying to see whether you memorized a formula.

They are testing whether you can:
- define the event clearly
- distinguish conditional from unconditional probability
- reason about uncertainty step by step

### Bayes' Theorem

Bayes' theorem matters because it lets you reverse a conditional probability.

That is useful when you know:
- how likely evidence is under a hypothesis

and want:
- how likely the hypothesis is after seeing the evidence

### Expectation

Expectation is the average value you would get in the long run.

It is not the most likely outcome. It is the probability-weighted average outcome.

### Variance

Variance measures spread around the mean.

Two random variables can have the same mean but very different variance, which means:
- same average behavior
- very different uncertainty

## Technical Details Interviewers Often Want

### Conditional Probability

One of the most common mistakes is reversing a conditional.

`P(A | B)` and `P(B | A)` are usually not the same.

That is exactly why Bayes' theorem matters.

### Base Rate Matters

In medical test or fraud examples, candidates often ignore the base rate.

Even if a test is very accurate:
- a rare event can still have a low posterior probability after a positive result

This is one of the highest-value interview insights in probability.

### Independence

Independence means:

`P(A and B) = P(A) * P(B)`

It does **not** just mean the events "feel unrelated."

You should be ready to say independence is a mathematical condition, not a vague intuition.

## Common Failure Modes

- confusing `P(A | B)` with `P(B | A)`
- ignoring base rates in Bayes problems
- treating expectation as the most likely outcome
- forgetting that variance depends on spread, not just scale
- assuming events are independent without justification

## Edge Cases and Follow-Up Questions

1. Why can a positive test still imply a low posterior probability of disease?
2. Why is expectation not necessarily an achievable outcome?
3. What is the difference between independence and mutual exclusivity?
4. Why can two variables have the same mean but different variance?
5. Why do interviewers like Bayes questions so much?

## What to Practice Saying Out Loud

1. Why Bayes' theorem is really about updating belief after evidence
2. Why base rate matters in real-world probability problems
3. Why expectation and variance capture different aspects of uncertainty

## Common Interview Questions

### Q1: Bayes' Theorem

**Question**: Given P(A|B), P(B), P(A), find P(B|A)

**Answer**:
```
P(B|A) = P(A|B) × P(B) / P(A)
```

**Example**:
- P(disease|positive test) = 0.95
- P(positive test) = 0.1
- P(disease) = 0.01
- Find: P(disease|positive test)

```
P(disease|positive) = P(positive|disease) × P(disease) / P(positive)
                    = 0.95 × 0.01 / 0.1
                    = 0.095
```

### Q2: Expected Value

**Question**: What's the expected value of rolling a die?

**Answer**:
```
E[X] = Σ x × P(x)
     = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = 3.5
```

### Q3: Variance

**Question**: What's the variance of rolling a die?

**Answer**:
```
Var(X) = E[X²] - (E[X])²
       = (1²+2²+3²+4²+5²+6²)/6 - 3.5²
       = 91/6 - 12.25
       = 2.92
```

### Q4: Conditional Probability

**Question**: In a deck of 52 cards, what's P(ace | face card)?

**Answer**:
```
P(ace | face card) = P(ace and face card) / P(face card)
                   = 0 / (12/52)  # Ace is not a face card
                   = 0
```

### Q5: Independence

**Question**: Are two coin flips independent?

**Answer**: Yes
```
P(A and B) = P(A) × P(B)
P(heads and heads) = 0.5 × 0.5 = 0.25
```

## Code Implementation

See `probability_qa.py` for implementations.

## Exercises

1. Solve Bayes' theorem problems
2. Calculate expected values
3. Compute variances
4. Work with conditional probabilities

## Next Steps

- **Topic 18**: Distribution classification
- **Topic 19**: Advanced clustering
