"""
Decision Tree from Scratch
Simple implementation
"""
import numpy as np
from collections import Counter

class DecisionTree:
    """
    Decision Tree Classifier
    
    How it's learned:
    1. Start with all data at root
    2. For each feature, find best split (minimize impurity)
    3. Split data based on best feature
    4. Recursively build left and right subtrees
    5. Stop when: max_depth, min_samples_split, or pure node
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y: np.ndarray) -> float:
        """
        Gini Impurity: Measure of node impurity
        
        Gini = 1 - Σ(p_i)²
        where p_i is proportion of class i
        
        Range: 0 (pure) to 1 (impure)
        """
        if len(y) == 0:
            return 0.0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions**2)
    
    def entropy(self, y: np.ndarray) -> float:
        """
        Entropy: Another impurity measure
        
        Entropy = -Σ(p_i × log2(p_i))
        """
        if len(y) == 0:
            return 0.0
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]  # Remove zeros
        return -np.sum(proportions * np.log2(proportions))
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find best feature and threshold to split
        
        Tries all features and thresholds
        Returns split that minimizes weighted impurity
        """
        best_impurity = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            # Try different thresholds (unique values)
            values = np.unique(X[:, feature_idx])
            for threshold in values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Compute weighted impurity
                left_impurity = self.gini_impurity(y[left_mask])
                right_impurity = self.gini_impurity(y[right_mask])
                weighted_impurity = (np.sum(left_mask) * left_impurity + 
                                   np.sum(right_mask) * right_impurity) / len(y)
                
                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_impurity
    
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively build decision tree
        
        Stopping conditions:
        1. Max depth reached
        2. Too few samples
        3. Pure node (all same class)
        """
        # Stopping conditions
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Return majority class
            return Counter(y).most_common(1)[0][0]
        
        # Find best split
        feature, threshold, impurity = self.find_best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        node = {
            'feature': feature,
            'threshold': threshold,
            'impurity': impurity,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train decision tree"""
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x: np.ndarray, node) -> int:
        """Predict single sample by traversing tree"""
        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self.predict_one(x, node['left'])
            else:
                return self.predict_one(x, node['right'])
        else:
            return node  # Leaf node: return class
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for all samples"""
        return np.array([self.predict_one(x, self.tree) for x in X])


class RandomForest:
    """
    Random Forest: Ensemble of decision trees
    
    How it's learned:
    1. Create bootstrap samples (random sampling with replacement)
    2. For each sample, train tree on random feature subset
    3. Final prediction: Majority vote (classification) or average (regression)
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 max_features: int = None, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        Create bootstrap sample (random sampling with replacement)
        
        Same size as original, but some samples repeated, some missing
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest"""
        if self.max_features is None:
            # Default: sqrt of number of features
            self.max_features = int(np.sqrt(X.shape[1]))
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self.bootstrap_sample(X, y)
            
            # Random feature subset
            feature_indices = np.random.choice(
                X.shape[1], self.max_features, replace=False
            )
            X_boot = X_boot[:, feature_indices]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append((tree, feature_indices))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote"""
        predictions = []
        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions).T
        return np.array([Counter(p).most_common(1)[0][0] for p in predictions])


# Usage Example
if __name__ == "__main__":
    print("Decision Tree and Random Forest")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    # Two classes, two features
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    
    # Split train/test
    split_idx = n_samples // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Decision Tree
    print("Decision Tree:")
    dt = DecisionTree(max_depth=5, min_samples_split=2)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = np.mean(dt_pred == y_test)
    print(f"  Accuracy: {dt_accuracy:.4f}")
    print()
    
    # Random Forest
    print("Random Forest:")
    rf = RandomForest(n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = np.mean(rf_pred == y_test)
    print(f"  Accuracy: {rf_accuracy:.4f}")
    print(f"  Number of trees: {len(rf.trees)}")

