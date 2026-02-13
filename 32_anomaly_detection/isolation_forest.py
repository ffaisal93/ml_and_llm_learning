"""
Isolation Forest from Scratch
Detailed implementation with explanations
"""
import numpy as np
from typing import List, Tuple
import random

class IsolationTree:
    """
    Single Isolation Tree
    
    How it works:
    1. Randomly select feature and split value
    2. Split data: left if < split, right if >= split
    3. Recursively build subtrees
    4. Stop when max depth reached or only one sample
    
    Key insight: Anomalies are isolated quickly (short path)
    """
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.root = None
        self.size = 0  # Number of samples in this tree
    
    def build_tree(self, X: np.ndarray, current_depth: int = 0):
        """
        Recursively build isolation tree
        
        Detailed explanation:
        - Random feature selection: Ensures anomalies are isolated quickly
        - Random split value: Between min and max of feature
        - Binary split: Left if value < split, right otherwise
        - Stop condition: Max depth or only one sample
        """
        n_samples, n_features = X.shape
        
        # Stop conditions
        if current_depth >= self.max_depth or n_samples <= 1:
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': current_depth
            }
        
        # Randomly select feature
        feature_idx = random.randint(0, n_features - 1)
        feature_values = X[:, feature_idx]
        
        # Randomly select split value
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        
        # If all values are same, make leaf
        if min_val == max_val:
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': current_depth
            }
        
        split_value = random.uniform(min_val, max_val)
        
        # Split data
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        # If one side is empty, make leaf
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': current_depth
            }
        
        # Recursively build subtrees
        node = {
            'type': 'internal',
            'feature': feature_idx,
            'split_value': split_value,
            'depth': current_depth,
            'left': self.build_tree(X[left_mask], current_depth + 1),
            'right': self.build_tree(X[right_mask], current_depth + 1)
        }
        
        return node
    
    def path_length(self, x: np.ndarray, node: dict, current_depth: int = 0) -> float:
        """
        Compute path length for a sample
        
        Path length = number of edges from root to leaf
        
        Why this matters:
        - Normal points: Long path (many splits needed)
        - Anomalies: Short path (few splits needed)
        """
        if node['type'] == 'leaf':
            # Adjust for unsuccessful search
            # If leaf has size > 1, average path length is higher
            if node['size'] > 1:
                return current_depth + self._c(node['size'])
            else:
                return current_depth
        
        # Traverse tree
        feature_idx = node['feature']
        split_value = node['split_value']
        
        if x[feature_idx] < split_value:
            return self.path_length(x, node['left'], current_depth + 1)
        else:
            return self.path_length(x, node['right'], current_depth + 1)
    
    def _c(self, n: int) -> float:
        """
        Normalization constant c(n)
        
        Formula: c(n) = 2H(n-1) - 2(n-1)/n
        
        Where H(n) is harmonic number:
        H(n) = 1 + 1/2 + 1/3 + ... + 1/n ≈ ln(n) + γ
        
        This adjusts path length for different tree sizes
        """
        if n <= 1:
            return 0
        
        # Harmonic number approximation
        H = sum(1.0 / i for i in range(1, n))
        
        return 2 * H - 2 * (n - 1) / n


class IsolationForest:
    """
    Isolation Forest: Ensemble of isolation trees
    
    How it works:
    1. Build multiple isolation trees (each on random subset)
    2. For each sample, compute average path length across all trees
    3. Compute anomaly score: s(x) = 2^(-E(h(x)) / c(n))
    4. High score (≈1) = anomaly, Low score (≈0) = normal
    
    Mathematical foundation:
    - Anomalies are easier to isolate → shorter path lengths
    - Normal points are harder to isolate → longer path lengths
    - Score formula: s = 2^(-avg_path_length / normalization)
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 max_depth: int = 10, contamination: float = 0.1):
        """
        Parameters:
        - n_estimators: Number of trees (more = more stable)
        - max_samples: Samples per tree (smaller = faster, less stable)
        - max_depth: Max tree depth (log2 of max_samples is typical)
        - contamination: Expected proportion of anomalies (for threshold)
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.contamination = contamination
        self.trees = []
        self.n_samples = None
    
    def fit(self, X: np.ndarray):
        """
        Train Isolation Forest
        
        Steps:
        1. Store number of samples (for normalization)
        2. For each tree:
           - Randomly sample max_samples
           - Build isolation tree
        3. Store all trees
        """
        self.n_samples = X.shape[0]
        
        # Adjust max_depth if not set
        if self.max_depth is None:
            self.max_depth = int(np.ceil(np.log2(self.max_samples)))
        
        # Build trees
        for i in range(self.n_estimators):
            # Random sample
            if self.max_samples >= self.n_samples:
                sample_indices = np.arange(self.n_samples)
            else:
                sample_indices = np.random.choice(
                    self.n_samples, self.max_samples, replace=False
                )
            
            X_sample = X[sample_indices]
            
            # Build tree
            tree = IsolationTree(max_depth=self.max_depth)
            tree.root = tree.build_tree(X_sample)
            tree.size = len(X_sample)
            self.trees.append(tree)
    
    def _anomaly_score(self, path_length: float) -> float:
        """
        Compute anomaly score
        
        Formula: s(x, n) = 2^(-E(h(x)) / c(n))
        
        Where:
        - E(h(x)): Average path length (path_length parameter)
        - c(n): Normalization constant
        
        Interpretation:
        - s ≈ 1: Anomaly (short path, easy to isolate)
        - s ≈ 0.5: Borderline
        - s ≈ 0: Normal (long path, hard to isolate)
        """
        # Normalization constant
        c_n = self._c(self.n_samples)
        
        # Anomaly score
        score = 2 ** (-path_length / c_n)
        
        return score
    
    def _c(self, n: int) -> float:
        """Normalization constant (same as in IsolationTree)"""
        if n <= 1:
            return 0
        H = sum(1.0 / i for i in range(1, n))
        return 2 * H - 2 * (n - 1) / n
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores
        
        Returns: Anomaly scores (0-1, higher = more anomalous)
        """
        scores = []
        
        for x in X:
            # Compute average path length across all trees
            path_lengths = []
            for tree in self.trees:
                path_length = tree.path_length(x, tree.root)
                path_lengths.append(path_length)
            
            avg_path_length = np.mean(path_lengths)
            
            # Compute anomaly score
            score = self._anomaly_score(avg_path_length)
            scores.append(score)
        
        return np.array(scores)
    
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels (anomaly or not)
        
        Uses contamination to set threshold
        """
        scores = self.predict(X)
        
        # Set threshold based on contamination
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return (scores >= threshold).astype(int)


# Usage Example
if __name__ == "__main__":
    print("Isolation Forest from Scratch")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    
    # Normal data: 2D Gaussian
    normal_data = np.random.randn(1000, 2)
    
    # Anomalies: Far from normal data
    anomalies = np.array([
        [5, 5],
        [-5, -5],
        [6, -4],
        [-4, 6]
    ])
    
    X = np.vstack([normal_data, anomalies])
    y = np.array([0] * 1000 + [1] * 4)  # Labels (0=normal, 1=anomaly)
    
    print(f"Data: {len(normal_data)} normal, {len(anomalies)} anomalies")
    print()
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=256,
        max_depth=10,
        contamination=0.004  # 4 anomalies out of 1004 samples
    )
    iso_forest.fit(X)
    
    # Predict
    scores = iso_forest.predict(X)
    labels = iso_forest.predict_labels(X)
    
    print("\nResults:")
    print(f"  Anomaly scores for anomalies: {scores[-4:]}")
    print(f"  Average score for normal: {np.mean(scores[:-4]):.4f}")
    print(f"  Average score for anomalies: {np.mean(scores[-4:]):.4f}")
    print()
    
    # Check detection
    detected = np.sum(labels[-4:] == 1)
    print(f"  Detected {detected}/{len(anomalies)} anomalies")
    
    false_positives = np.sum(labels[:-4] == 1)
    print(f"  False positives: {false_positives}/{len(normal_data)}")

