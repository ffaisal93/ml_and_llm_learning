# Topic 10: Optimizers

## What You'll Learn

This topic teaches you optimization algorithms:
- SGD (Stochastic Gradient Descent)
- Momentum
- Adam
- AdamW
- RMSprop
- Theory and implementations

## Why We Need This

### Interview Importance
- **Common question**: "Explain Adam optimizer"
- **Implementation**: "Implement Adam from scratch"
- **Understanding**: Know how optimizers work

### Real-World Application
- **Training neural networks**: All use optimizers
- **Convergence**: Right optimizer = faster training
- **Default choice**: Adam is default for most cases

## Industry Use Cases

### 1. **SGD**
**Use Case**: Simple problems, small models
- Basic optimization
- When you need simplicity

### 2. **Adam**
**Use Case**: Default for most deep learning
- Adaptive learning rates
- Works well out of the box
- Most common in practice

### 3. **AdamW**
**Use Case**: Better weight decay
- Improved over Adam
- Better generalization
- Modern default

## Industry-Standard Boilerplate Code

### SGD with Momentum

```python
"""
SGD with Momentum
Adds momentum term to smooth updates
"""
import numpy as np

class SGDWithMomentum:
    """
    SGD with Momentum
    v_t = beta * v_{t-1} + gradient
    params = params - lr * v_t
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Update velocity (momentum)
        self.velocity = self.momentum * self.velocity + gradients
        
        # Update parameters
        params -= self.learning_rate * self.velocity
        
        return params
```

### Adam Optimizer

```python
"""
Adam Optimizer from Scratch
Interview question: "Implement Adam optimizer"
"""
import numpy as np

class Adam:
    """
    Adam: Adaptive Moment Estimation
    Combines momentum (first moment) and RMSprop (second moment)
    """
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (momentum)
        self.v = None  # Second moment (variance)
        self.t = 0     # Time step
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Adam"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first moment (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment (variance)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

## Theory

### Optimizer Comparison

| Optimizer | Learning Rate | Momentum | Adaptive | Use Case |
|-----------|--------------|----------|----------|----------|
| SGD | Fixed | No | No | Simple |
| SGD+Momentum | Fixed | Yes | No | Better than SGD |
| Adam | Adaptive | Yes | Yes | **Default** |
| AdamW | Adaptive | Yes | Yes | Better weight decay |

### Why Adam Works
- **Adaptive learning rates**: Different rates for different parameters
- **Momentum**: Smooths updates
- **Second moment**: Adapts to gradient variance
- **Bias correction**: Fixes initial bias

## Exercises

1. Implement all optimizers
2. Compare convergence
3. Tune hyperparameters
4. Visualize optimization paths

## Next Steps

- **Topic 11**: Regularization
- **Topic 12**: Theory

