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

## Detailed Derivations

**Complete mathematical derivations with intuitive explanations:**
- **Linear Regression**: Step-by-step derivation from first principles
  - Why least squares?
  - Normal equation derivation
  - Geometric intuition
  - Gradient descent alternative
  - See `linear_regression_derivation.md`

- **Logistic Regression**: Complete derivation with intuitive explanations
  - Why sigmoid function?
  - Log odds transformation
  - Cross-entropy loss derivation
  - Gradient descent
  - Decision boundary
  - See `logistic_regression_derivation.md`

### 🔥 Logistic Regression — Interview-Grade Files (READ THESE FOR INTERVIEWS)

The base derivation file is fine for understanding. But if you're prepping for applied scientist or ML engineer interviews, the bar is much higher. The two files below cover the deep, intuitive, gotcha-style questions that interviewers actually ask:

- **`LOGISTIC_REGRESSION_DEEP_DIVE.md`** — Frontier-lab interview deep dive: linear log-odds assumption, MLE derivation, why CE not MSE (likelihood + convexity), separability and divergence, IRLS, calibration, multicollinearity, L1/L2 geometry, max entropy, connections to softmax/NB/SVM/neural networks. **Read this whole document before any classical-ML interview.**
- **`LOGISTIC_REGRESSION_INTERVIEW_GRILL.md`** — 60 active-recall questions with strong answers. Drill until you can answer 40+ cold.

## Core Intuition

Classical ML is still high-value interview material because it exposes the basic logic that more complex models build on.

### Linear Regression

Linear regression tries to fit a straight-line or linear relationship between input features and a continuous target.

Easy way to think about it:
- each feature gets a weight
- the prediction is a weighted sum plus a bias
- learning means adjusting those weights to reduce squared error

Why least squares?
- large mistakes should hurt more than small mistakes
- the math becomes clean and differentiable
- the closed-form solution and gradient solution are both easy to explain

### Logistic Regression

Logistic regression is not "linear regression but for labels."

The key difference is that it models probability:

`P(y = 1 | x) = sigmoid(w^T x + b)`

Why sigmoid?
- it maps any real number to `[0, 1]`
- it gives a probabilistic interpretation
- it connects naturally to Bernoulli likelihood and cross-entropy

### KNN

KNN does not learn parameters in the usual sense.

Instead:
- store the dataset
- compare a new point to stored points
- let nearby points vote or average

This is useful in interviews because it helps explain the idea of local similarity very directly.

### K-Means

K-means is trying to summarize a dataset using a small number of centroids.

The loop is simple:
1. assign each point to the nearest centroid
2. move each centroid to the mean of the assigned points
3. repeat

The main intuition is that it alternates between:
- deciding cluster membership
- refining cluster representatives

## Technical Details Interviewers Often Care About

### Linear Regression

- Objective: minimize mean squared error
- Gradients: `dw = X^T (y_pred - y) / n`, `db = mean(y_pred - y)`
- Closed-form solution exists for small well-behaved problems
- Sensitive to collinearity and outliers

### Logistic Regression

- Uses sigmoid plus cross-entropy
- Decision boundary is linear in feature space
- The gradient simplifies nicely because BCE with sigmoid gives `p - y`
- Still a linear classifier even though the output is probabilistic

### KNN

- Prediction cost is high because you compare with stored points
- Distance metric choice matters
- Feature scaling matters a lot
- Small `k` means high variance; large `k` means high bias

### K-Means

- Assumes roughly spherical clusters
- Sensitive to initialization
- Uses Euclidean-style geometry
- Empty clusters are a real edge case in implementation

## Common Failure Modes

### Linear / Logistic Regression

- unscaled features slow or distort optimization
- collinear features make weights unstable
- outliers can dominate the fit
- using the wrong loss for the task

### KNN

- bad feature scaling breaks nearest-neighbor behavior
- large datasets make prediction slow
- irrelevant features pollute distance

### K-Means

- non-spherical clusters are handled poorly
- wrong number of clusters
- bad initialization leads to poor local minima
- empty clusters during updates

## What the Interviewer May Ask Next

1. Why does logistic regression use cross-entropy instead of MSE?
2. Why is KNN called a lazy learner?
3. Why does feature scaling matter for KNN and K-means?
4. When would linear regression beat a larger nonlinear model?
5. What assumptions make linear regression a reasonable model?

## What to Practice Saying Out Loud

1. Why is logistic regression still a linear classifier?
2. Why does K-means converge, and what does it converge to?
3. Why can KNN be strong as a baseline but weak at scale?
4. What is the geometric interpretation of linear regression?

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
