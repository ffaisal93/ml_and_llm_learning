"""
Linear Regression from Scratch - Pure Python/NumPy Version
Interview question: "Implement linear regression without using sklearn"

Two versions:
1. Pure Python/NumPy (this file)
2. PyTorch version (see linear_regression_torch.py)

Mathematical Formulation:
- Model: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w^T x + b
- Cost Function: MSE = (1/n) Σ(y_pred - y_true)²
- Gradient Descent:
  - ∂MSE/∂w = (1/n) X^T (y_pred - y_true)
  - ∂MSE/∂b = (1/n) Σ(y_pred - y_true)
  - Update: w = w - α × ∂MSE/∂w, b = b - α × ∂MSE/∂b

For detailed derivation with intuitive explanations, see:
- linear_regression_derivation.md (complete step-by-step derivation)
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression: y = w*x + b
    Industry standard implementation pattern
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using gradient descent
        
        Args:
            X: Features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost (MSE)
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)
    
    # Train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Predict
    predictions = model.predict(X)
    
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"R² Score: {model.score(X, y):.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X, predictions, 'r-', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Linear Regression Fit')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost History')
    
    plt.tight_layout()
    plt.savefig('linear_regression.png')
    print("Plot saved to linear_regression.png")

