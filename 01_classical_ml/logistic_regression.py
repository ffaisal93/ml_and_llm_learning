"""
Logistic Regression from Scratch - Pure Python/NumPy Version
Interview question: "Implement logistic regression"

Two versions:
1. Pure Python/NumPy (this file)
2. PyTorch version (see logistic_regression_torch.py)
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression: P(y=1|x) = 1 / (1 + exp(-(w*x + b)))
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        # Clip to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost (log loss)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            cost = -(1/n_samples) * np.sum(
                y * np.log(y_pred + epsilon) + 
                (1 - y) * np.log(1 - y_pred + epsilon)
            )
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class"""
        linear_model = X.dot(self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        linear_model = X.dot(self.weights) + self.bias
        return self._sigmoid(linear_model)


# Usage Example
if __name__ == "__main__":
    # Generate sample data (two classes)
    np.random.seed(42)
    n_samples = 100
    
    # Class 0
    X0 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
    y0 = np.zeros(n_samples//2)
    
    # Class 1
    X1 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    y1 = np.ones(n_samples//2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # Predict
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.6)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.6)
    
    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Log Loss)')
    plt.title('Cost History')
    
    plt.tight_layout()
    plt.savefig('logistic_regression.png')
    print("Plot saved to logistic_regression.png")

