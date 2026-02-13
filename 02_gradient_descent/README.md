# Topic 2: Gradient Descent Variants

## What You'll Learn

This topic teaches you different gradient descent algorithms:
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Momentum
- Adam optimizer
- Theory and trade-offs

## Why We Need This

### Interview Importance
- **Common question**: "Explain different gradient descent variants"
- **Implementation**: "Implement SGD from scratch"
- **Trade-offs**: Understanding when to use which

### Real-World Application
- **Training neural networks**: All use gradient descent
- **Optimization**: Choosing right optimizer matters
- **Efficiency**: Different variants have different speeds

## Industry Use Cases

### 1. **Batch Gradient Descent**
**Use Case**: Small datasets, exact gradients
- Simple models
- When you need exact gradient

### 2. **Stochastic Gradient Descent (SGD)**
**Use Case**: Large datasets, online learning
- Deep learning
- Real-time updates

### 3. **Mini-Batch Gradient Descent**
**Use Case**: Most common in practice
- Neural networks
- Balance between speed and accuracy

### 4. **Adam Optimizer**
**Use Case**: Default choice for many
- Deep learning
- Adaptive learning rates

## Industry-Standard Boilerplate Code

### Batch Gradient Descent

```python
"""
Batch Gradient Descent
Uses entire dataset for each update
"""
import numpy as np

class BatchGradientDescent:
    """
    Batch GD: Update using all training examples
    Pros: Stable, exact gradient
    Cons: Slow for large datasets
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray, model):
        """Train using batch gradient descent"""
        n_samples = len(X)
        cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass (all samples)
            predictions = model.predict(X)
            
            # Compute gradient (all samples)
            gradient = self._compute_gradient(X, y, predictions, model)
            
            # Update parameters
            model.update_parameters(gradient, self.learning_rate)
            
            # Compute cost
            cost = self._compute_cost(y, predictions)
            cost_history.append(cost)
        
        return cost_history
    
    def _compute_gradient(self, X, y, predictions, model):
        """Compute gradient using all samples"""
        n_samples = len(X)
        error = predictions - y
        gradient = (1/n_samples) * X.T.dot(error)
        return gradient
    
    def _compute_cost(self, y, predictions):
        """Compute MSE cost"""
        return np.mean((y - predictions)**2)
```

### Stochastic Gradient Descent (SGD)

```python
"""
Stochastic Gradient Descent
Updates using one sample at a time
"""
import numpy as np

class StochasticGradientDescent:
    """
    SGD: Update using one training example at a time
    Pros: Fast, can escape local minima
    Cons: Noisy updates, less stable
    """
    
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
    
    def fit(self, X: np.ndarray, y: np.ndarray, model):
        """Train using SGD"""
        n_samples = len(X)
        cost_history = []
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            # Process each sample
            for i in range(n_samples):
                x_i = X_shuffled[i:i+1]  # Keep 2D
                y_i = y_shuffled[i]
                
                # Forward pass (single sample)
                prediction = model.predict(x_i)
                
                # Compute gradient (single sample)
                gradient = self._compute_gradient(x_i, y_i, prediction, model)
                
                # Update parameters
                model.update_parameters(gradient, self.learning_rate)
                
                # Accumulate cost
                epoch_cost += (prediction - y_i)**2
            
            cost_history.append(epoch_cost / n_samples)
        
        return cost_history
    
    def _compute_gradient(self, x, y, prediction, model):
        """Compute gradient for single sample"""
        error = prediction - y
        gradient = x.T.dot(error)
        return gradient
```

### Mini-Batch Gradient Descent

```python
"""
Mini-Batch Gradient Descent
Updates using small batches
Most common in practice
"""
import numpy as np

class MiniBatchGradientDescent:
    """
    Mini-batch GD: Update using small batches
    Pros: Balance between speed and stability
    Cons: Need to tune batch size
    """
    
    def __init__(self, learning_rate: float = 0.01, 
                 batch_size: int = 32, n_epochs: int = 10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    def fit(self, X: np.ndarray, y: np.ndarray, model):
        """Train using mini-batch gradient descent"""
        n_samples = len(X)
        cost_history = []
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            n_batches = n_samples // self.batch_size
            
            # Process in batches
            for i in range(0, n_samples, self.batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass (batch)
                predictions = model.predict(X_batch)
                
                # Compute gradient (batch)
                gradient = self._compute_gradient(X_batch, y_batch, 
                                                  predictions, model)
                
                # Update parameters
                model.update_parameters(gradient, self.learning_rate)
                
                # Accumulate cost
                batch_cost = np.mean((predictions - y_batch)**2)
                epoch_cost += batch_cost
            
            cost_history.append(epoch_cost / n_batches)
        
        return cost_history
    
    def _compute_gradient(self, X_batch, y_batch, predictions, model):
        """Compute gradient for batch"""
        batch_size = len(X_batch)
        error = predictions - y_batch
        gradient = (1/batch_size) * X_batch.T.dot(error)
        return gradient
```

### Adam Optimizer

```python
"""
Adam Optimizer
Adaptive learning rate optimizer
Most popular for deep learning
"""
import numpy as np

class Adam:
    """
    Adam: Adaptive Moment Estimation
    Combines momentum and adaptive learning rates
    """
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (momentum)
        self.v = None  # Second moment (RMSprop)
        self.t = 0     # Time step
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Adam"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

## Theory

### Batch vs SGD vs Mini-Batch

| Method | Batch Size | Speed | Stability | Memory |
|--------|-----------|-------|-----------|--------|
| Batch | All | Slow | Very Stable | High |
| SGD | 1 | Fast | Noisy | Low |
| Mini-Batch | 32-256 | Medium | Stable | Medium |

### When to Use Which

- **Batch GD**: Small datasets, need exact gradient
- **SGD**: Large datasets, online learning
- **Mini-Batch**: Most common, best balance
- **Adam**: Deep learning, default choice

## Exercises

1. Implement momentum SGD
2. Compare convergence rates
3. Tune batch size
4. Implement learning rate decay

## Next Steps

- **Topic 3**: Evaluation metrics
- **Topic 4**: Transformers

