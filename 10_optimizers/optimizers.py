"""
Optimizers from Scratch
Interview question: "Implement Adam optimizer"

Mathematical Formulation (Adam):
1. First moment (momentum): m_t = β₁ × m_{t-1} + (1-β₁) × g_t
2. Second moment (variance): v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
3. Bias correction: m̂_t = m_t / (1 - β₁^t), v̂_t = v_t / (1 - β₂^t)
4. Update: θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)

Why it works:
- Adaptive learning rates per parameter
- Momentum smooths updates
- Second moment adapts to gradient variance
- Bias correction fixes initial bias
"""
import numpy as np

class SGD:
    """Stochastic Gradient Descent"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Simple SGD update"""
        params -= self.learning_rate * gradients
        return params

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
        """Update with momentum"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity + gradients
        params -= self.learning_rate * self.velocity
        return params

class RMSprop:
    """
    RMSprop: Adaptive learning rate
    v_t = beta * v_{t-1} + (1-beta) * gradient^2
    params = params - lr * gradient / sqrt(v_t + eps)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = None
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update with RMSprop"""
        if self.v is None:
            self.v = np.zeros_like(params)
        
        self.v = self.beta * self.v + (1 - self.beta) * (gradients**2)
        params -= self.learning_rate * gradients / (np.sqrt(self.v) + self.epsilon)
        return params

class Adam:
    """
    Adam: Adaptive Moment Estimation
    Combines momentum (first moment) and RMSprop (second moment)
    Most popular optimizer
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
        
        # Bias correction (important!)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params

class AdamW:
    """
    AdamW: Adam with decoupled weight decay
    Better weight decay than Adam
    """
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 weight_decay: float = 0.01, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update with AdamW"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update moments (same as Adam)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Adam update
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Decoupled weight decay (applied separately)
        params -= self.learning_rate * self.weight_decay * params
        
        return params


# Usage Example
if __name__ == "__main__":
    print("Optimizers Comparison")
    print("=" * 60)
    
    # Simple optimization problem: minimize x^2
    x_sgd = np.array([5.0])
    x_momentum = np.array([5.0])
    x_adam = np.array([5.0])
    
    sgd = SGD(learning_rate=0.1)
    momentum = SGDWithMomentum(learning_rate=0.1, momentum=0.9)
    adam = Adam(learning_rate=0.1)
    
    print("Minimizing f(x) = x^2 (optimal x = 0)")
    print(f"Initial x: {x_sgd[0]}")
    print()
    
    for step in range(10):
        # Gradient of x^2 is 2*x
        grad = 2 * x_sgd
        
        x_sgd = sgd.update(x_sgd, grad)
        x_momentum = momentum.update(x_momentum, grad)
        x_adam = adam.update(x_adam, grad)
        
        if step % 2 == 0:
            print(f"Step {step}:")
            print(f"  SGD: {x_sgd[0]:.4f}")
            print(f"  Momentum: {x_momentum[0]:.4f}")
            print(f"  Adam: {x_adam[0]:.4f}")

