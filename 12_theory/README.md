# Topic 12: Comprehensive Theory

## What You'll Learn

This topic provides comprehensive theory on:
- Classical ML theory
- LLM theory
- LLM inference theory
- Bias-variance tradeoff
- Regularization theory
- Optimization theory

## Why We Need This

### Interview Importance
- **Theory questions**: "Explain bias-variance tradeoff"
- **Deep understanding**: Theory helps answer "why"
- **Problem-solving**: Theory guides solutions

### Real-World Application
- **Decision-making**: Theory helps choose approaches
- **Debugging**: Understand why things work/don't work
- **Innovation**: Build on theoretical foundations

## Key Theory Topics

### 1. Bias-Variance Tradeoff

**Bias**: Error from oversimplifying model
- High bias = Underfitting
- Low bias = Model can learn complex patterns

**Variance**: Error from model sensitivity to training data
- High variance = Overfitting
- Low variance = Model generalizes well

**Tradeoff**: Can't minimize both simultaneously
- Simple model: High bias, low variance
- Complex model: Low bias, high variance
- Goal: Find balance

### 2. Regularization Theory

**Why Regularization Works:**
- Prevents overfitting
- Improves generalization
- Controls model complexity

**L1 vs L2:**
- L1: Promotes sparsity (feature selection)
- L2: Shrinks weights (smoother)

### 3. Optimization Theory

**Convex vs Non-convex:**
- Convex: One global minimum
- Non-convex: Multiple local minima
- Deep learning: Non-convex optimization

**Gradient Descent:**
- Converges to local minimum
- Learning rate critical
- Momentum helps escape local minima

### 4. LLM Theory

**Transformer Architecture:**
- Self-attention: Relate all positions
- Position encoding: Add position info
- Layer normalization: Stabilize training

**Attention Mechanism:**
- Query-Key-Value paradigm
- Scaled dot-product
- Multi-head for different subspaces

**Generation:**
- Autoregressive: One token at a time
- KV caching: Avoid recomputation
- Sampling: Control randomness

## Detailed Explanations

See individual theory files for detailed explanations.

## Exercises

1. Derive gradient formulas
2. Prove bias-variance decomposition
3. Analyze convergence rates
4. Understand attention complexity

## Next Steps

- **Topic 13**: Interview Q&A
- Review all topics

