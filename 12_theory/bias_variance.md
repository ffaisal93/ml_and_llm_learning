# Bias-Variance Tradeoff: Detailed Theory

## The Fundamental Tradeoff

**Bias-Variance Decomposition:**

```
Total Error = Bias² + Variance + Irreducible Error
```

### Components

**1. Bias**
- Error from oversimplifying the model
- High bias = Model too simple
- Example: Linear model for non-linear data

**2. Variance**
- Error from model sensitivity to training data
- High variance = Model overfits
- Example: Complex model memorizes training data

**3. Irreducible Error**
- Error inherent in the problem
- Cannot be reduced
- Example: Noise in data

## Visual Understanding

```
Simple Model (High Bias, Low Variance):
- Consistent predictions
- But consistently wrong
- Underfitting

Complex Model (Low Bias, High Variance):
- Can fit training data perfectly
- But predictions vary a lot
- Overfitting

Optimal Model (Balanced):
- Good fit to data
- Generalizes well
- Right complexity
```

## Mathematical Formulation

For a model f(x) predicting target y:

**Expected Prediction Error:**
```
E[(y - f(x))²] = Bias² + Var(f(x)) + σ²
```

Where:
- **Bias²**: (E[f(x)] - E[y])²
- **Variance**: E[(f(x) - E[f(x)])²]
- **σ²**: Irreducible error

## How to Reduce Bias

1. **More complex model**: Add features, layers
2. **Better features**: More informative inputs
3. **Longer training**: Let model learn more

## How to Reduce Variance

1. **More data**: Larger training set
2. **Regularization**: L1, L2, dropout
3. **Simpler model**: Reduce complexity
4. **Ensemble**: Average multiple models

## Practical Implications

- **Underfitting (High Bias)**: Increase model complexity
- **Overfitting (High Variance)**: Add regularization, more data
- **Goal**: Find sweet spot

## Interview Questions

**Q: Explain bias-variance tradeoff**
**A**: [See above - comprehensive explanation]

**Q: How do you diagnose high bias vs high variance?**
**A**: 
- High bias: High training error, high test error (both similar)
- High variance: Low training error, high test error (gap between them)

