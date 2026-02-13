# Topic 11: Regularization

## What You'll Learn

This topic teaches you regularization techniques:
- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Dropout
- Early stopping
- Theory and implementations

## Why We Need This

### Interview Importance
- **Common question**: "Explain L1 vs L2 regularization"
- **Understanding**: Prevent overfitting
- **Trade-offs**: Bias-variance tradeoff

### Real-World Application
- **Overfitting**: Models overfit without regularization
- **Generalization**: Regularization improves generalization
- **Feature selection**: L1 can select features

## Industry Use Cases

### 1. **L2 Regularization**
**Use Case**: Most common
- Prevents large weights
- Improves generalization
- Default in many frameworks

### 2. **L1 Regularization**
**Use Case**: Feature selection
- Sparse models
- Feature selection
- Interpretability

### 3. **Dropout**
**Use Case**: Neural networks
- Prevents co-adaptation
- Improves generalization
- Standard in deep learning

## Industry-Standard Boilerplate Code

### L1 Regularization (Lasso)

```python
"""
L1 Regularization (Lasso)
Adds |weights| to loss
Promotes sparsity (many weights become 0)
"""
import numpy as np

def l1_regularization_loss(weights: np.ndarray, lambda_reg: float) -> float:
    """
    L1 Regularization: lambda * sum(|w|)
    
    Effect: Many weights become exactly 0 (sparsity)
    Use: Feature selection, interpretability
    """
    return lambda_reg * np.sum(np.abs(weights))

def l1_gradient(weights: np.ndarray, lambda_reg: float) -> np.ndarray:
    """Gradient of L1 regularization"""
    return lambda_reg * np.sign(weights)
```

### L2 Regularization (Ridge)

```python
"""
L2 Regularization (Ridge)
Adds weights^2 to loss
Prevents large weights
"""
import numpy as np

def l2_regularization_loss(weights: np.ndarray, lambda_reg: float) -> float:
    """
    L2 Regularization: lambda * sum(w^2)
    
    Effect: Shrinks weights toward 0
    Use: Most common, improves generalization
    """
    return lambda_reg * np.sum(weights**2)

def l2_gradient(weights: np.ndarray, lambda_reg: float) -> np.ndarray:
    """Gradient of L2 regularization"""
    return 2 * lambda_reg * weights
```

### Dropout

```python
"""
Dropout from Scratch
Randomly set some activations to 0 during training
"""
import numpy as np

def dropout(x: np.ndarray, dropout_rate: float, training: bool = True) -> np.ndarray:
    """
    Dropout: Randomly zero out activations
    
    Args:
        x: Input activations
        dropout_rate: Probability of dropping (0.0 to 1.0)
        training: If False, no dropout (scale by 1-dropout_rate)
    """
    if not training:
        return x * (1 - dropout_rate)
    
    # Create mask
    mask = np.random.binomial(1, 1 - dropout_rate, x.shape)
    
    # Apply mask and scale
    return x * mask / (1 - dropout_rate)
```

## Theory

### L1 vs L2

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Penalty** | |w| | w² |
| **Effect** | Sparsity (weights → 0) | Shrinking (weights → small) |
| **Use** | Feature selection | Generalization |
| **Gradient** | Constant | Linear |

### Bias-Variance Tradeoff
- **No regularization**: Low bias, high variance (overfitting)
- **Too much regularization**: High bias, low variance (underfitting)
- **Right amount**: Balance

## Exercises

1. Implement L1/L2 in linear regression
2. Compare with/without regularization
3. Implement dropout
4. Tune regularization strength

## Next Steps

- **Topic 12**: Comprehensive theory
- **Topic 13**: Interview Q&A

