"""
K-Nearest Neighbors from Scratch - Pure Python Version
Interview question: "Implement KNN"

Simple pure Python implementation (can use NumPy for arrays)
"""
import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier
    Lazy learning algorithm - no training phase
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
            most_common = Counter(neighbors).most_common(1)[0][0] ## explain the code line by line:
            # Counter is a class that counts the number of times each element appears in a list
            # most_common is a method that returns the most common element in a list
            # most_common(1) returns the most common element in a list
            # [0][0] returns the most common element in a list
            # most_common(1)[0][0] returns the most common element in a list

            predictions.append(most_common)
        return np.array(predictions)


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Class 0
    X0 = np.random.randn(n_samples//2, 2) + np.array([-1, -1])
    y0 = np.zeros(n_samples//2)
    
    # Class 1
    X1 = np.random.randn(n_samples//2, 2) + np.array([1, 1])
    y1 = np.ones(n_samples//2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train and predict
    model = KNN(k=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.4f}")

