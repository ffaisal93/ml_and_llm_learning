# Topic 1: Classical ML Algorithms

## What You'll Learn

This topic teaches you to implement classical ML algorithms from scratch:
- Linear regression
- Logistic regression
- K-Nearest Neighbors (KNN)
- K-Means clustering
- Theory and intuition behind each

## Why We Need This

### Interview Importance
- **Common interview questions**: "Implement linear regression from scratch"
- **Foundation knowledge**: Understanding basics is crucial
- **Problem-solving**: Shows you understand fundamentals

### Real-World Application
- **Baseline models**: Simple models often work well
- **Understanding**: Helps understand complex models
- **Debugging**: Know what's happening under the hood

## Industry Use Cases

### 1. **Linear Regression**
**Use Case**: Predicting continuous values
- House price prediction
- Sales forecasting
- Risk assessment

### 2. **Logistic Regression**
**Use Case**: Binary classification
- Spam detection
- Customer churn prediction
- Medical diagnosis

### 3. **KNN**
**Use Case**: Classification and regression
- Recommendation systems
- Anomaly detection
- Pattern recognition

### 4. **K-Means**
**Use Case**: Clustering
- Customer segmentation
- Image compression
- Data preprocessing

## Industry-Standard Boilerplate Code

### Linear Regression (From Scratch)

```python
"""
Linear Regression from Scratch
Interview question: "Implement linear regression without using sklearn"
"""
import numpy as np
from typing import Tuple

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
```

### Linear Regression (PyTorch)

```python
"""
Linear Regression using PyTorch
Very simple PyTorch version
"""
import torch
import torch.nn as nn

class LinearRegressionTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Input size 1, output size 1
    
    def forward(self, x):
        return self.linear(x)

# Usage
model = LinearRegressionTorch()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    predictions = model(X)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Logistic Regression (Pure Python/NumPy)

```python
"""
Logistic Regression from Scratch
Interview question: "Implement logistic regression"
"""
import numpy as np

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
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
    
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
            cost = -(1/n_samples) * np.sum(
                y * np.log(y_pred + 1e-15) + 
                (1 - y) * np.log(1 - y_pred + 1e-15)
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
```

### K-Nearest Neighbors (From Scratch)

```python
"""
K-Nearest Neighbors from Scratch
Interview question: "Implement KNN"
"""
import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data (lazy learning)"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _get_neighbors(self, x: np.ndarray) -> list:
        """Get k nearest neighbors"""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Get k nearest
        neighbors = [label for _, label in distances[:self.k]]
        return neighbors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for each sample"""
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            # Majority vote
            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
```

### K-Means Clustering (From Scratch)

```python
"""
K-Means Clustering from Scratch
Interview question: "Implement K-means"
"""
import numpy as np

class KMeans:
    """
    K-Means clustering algorithm
    """
    
    def __init__(self, k: int = 3, max_iters: int = 100, random_state: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def _initialize_centroids(self, X: np.ndarray):
        """Initialize centroids randomly"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray):
        """Update centroids based on cluster assignments"""
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = cluster_points.mean(axis=0)
    
    def fit(self, X: np.ndarray):
        """Fit K-means to data"""
        self._initialize_centroids(X)
        
        for _ in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            self._update_centroids(X, labels)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        self.labels = labels
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster for new data"""
        return self._assign_clusters(X)
```

## Theory

### Linear Regression
- **Goal**: Find line that best fits data
- **Cost function**: Mean Squared Error (MSE)
- **Optimization**: Gradient descent
- **Assumptions**: Linear relationship, independence, homoscedasticity

### Logistic Regression
- **Goal**: Predict probability of binary outcome
- **Activation**: Sigmoid function
- **Cost function**: Log loss (cross-entropy)
- **Optimization**: Gradient descent

### KNN
- **Lazy learning**: No training phase
- **Distance metric**: Usually Euclidean
- **K value**: Trade-off between bias and variance
- **Time complexity**: O(n) for prediction

### K-Means
- **Goal**: Partition data into k clusters
- **Algorithm**: Iterative optimization
- **Initialization**: Important for convergence
- **Limitations**: Assumes spherical clusters

## Exercises

1. Implement linear regression with regularization
2. Add early stopping to logistic regression
3. Implement weighted KNN
4. Add k-means++ initialization

## Next Steps

- **Topic 2**: Gradient descent variants
- **Topic 3**: Evaluation metrics

