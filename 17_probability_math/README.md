# Topic 17: Probability Math Q&A

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

