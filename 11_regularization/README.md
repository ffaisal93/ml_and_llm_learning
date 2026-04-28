# Topic 11: Regularization

> 🔥 **For interviews, read these first:**
> - **`REGULARIZATION_DEEP_DIVE.md`** — frontier-lab interview deep dive: bias-variance trade-off, L1/L2 geometry and Bayesian priors, dropout (3 stories), early stopping ≈ L2, MixUp/CutMix, label smoothing, SAM, implicit regularization of SGD, why modern LLMs use no dropout.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

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

## Core Intuition

Regularization is about controlling how the model fits the data, not just adding a penalty term mechanically.

The deeper idea is:
- models can fit patterns that are real
- models can also fit noise, shortcuts, or accidental correlations

Regularization pushes learning toward more stable solutions.

### L2 Regularization

L2 discourages very large weights.

Intuition:
- if a solution needs extreme parameter values to fit the data, it may be too brittle
- smaller weights usually correspond to smoother functions

### L1 Regularization

L1 encourages sparsity.

That makes it useful when:
- many features may be irrelevant
- interpretability matters
- you want the model to rely on a smaller subset of features

### Dropout

Dropout randomly removes activations during training.

The core intuition is:
- the network should not rely too heavily on any one hidden pathway
- multiple redundant, more robust pathways are encouraged

### Early Stopping

Early stopping is also regularization.

It works because:
- later optimization steps may fit noise more aggressively
- stopping at the right point can reduce overfitting

## Technical Details Interviewers Often Want

### L1 vs L2 Difference

This is a very common question.

- **L1** can drive weights exactly to zero
- **L2** usually shrinks weights continuously but not exactly to zero

That is why L1 is associated with feature selection.

### Why Dropout Uses Scaling

During training, units are dropped randomly.

To keep expected activation magnitude consistent, dropout implementations usually scale activations appropriately. Otherwise train-time and test-time behavior would not match.

### Regularization Is an Inductive Bias

This is a stronger interview answer than "it prevents overfitting."

Regularization says:
- prefer simpler or more stable explanations
- prefer smaller weights
- prefer less co-adaptation
- prefer solutions that transfer better

## Common Failure Modes

- too much regularization causing underfitting
- using dropout mechanically where it does not help much
- confusing L2 regularization with all forms of weight decay in adaptive optimizers
- claiming regularization always improves test performance
- forgetting that data augmentation is also a form of regularization

## Edge Cases and Follow-Up Questions

1. Why can L1 produce sparse solutions?
2. Why is L2 often the default regularizer?
3. Why can too much regularization hurt performance?
4. Why is early stopping considered regularization?
5. Why may dropout help in some networks more than others?

## What to Practice Saying Out Loud

1. Why regularization is really about inductive bias
2. The conceptual difference between L1, L2, dropout, and early stopping
3. Why preventing overfitting is not the same as blindly increasing regularization

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
