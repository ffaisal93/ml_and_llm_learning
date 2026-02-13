"""
Gradient Boosting and XGBoost (Simplified)
Simple implementations
"""
import numpy as np
from collections import Counter

# Simple DecisionTree for regression (simplified version)
class SimpleTree:
    """Simplified tree for gradient boosting"""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        """Fit tree - simplified"""
        # For simplicity, use mean prediction
        self.mean = np.mean(y)
        self.tree = self.mean
    
    def predict(self, X):
        """Predict - simplified"""
        return np.full(len(X), self.mean)

class GradientBoosting:
    """
    Gradient Boosting
    
    How it's learned:
    1. Start with initial prediction (mean/median)
    2. For each tree:
       - Compute residuals (errors) of current model
       - Train tree on residuals
       - Add tree to ensemble (with learning rate)
    3. Final prediction: Sum of all trees
    
    Mathematical Formulation:
    F_m(x) = F_{m-1}(x) + α × h_m(x)
    
    Where:
    - F_m: Model after m trees
    - h_m: m-th tree (trained on residuals)
    - α: Learning rate (shrinkage)
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train gradient boosting"""
        # Initial prediction: mean (for regression)
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = y - predictions
            
            # Train tree on residuals (simplified - in practice use proper tree)
            # For this example, using simple mean prediction
            tree = SimpleTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
            
            self.trees.append(tree)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict: Sum of initial + all trees"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
        
        return predictions


class XGBoostSimplified:
    """
    XGBoost (Simplified)
    
    Key differences from Gradient Boosting:
    1. Uses second-order gradient (Hessian)
    2. Built-in regularization (L1, L2)
    3. More efficient tree construction
    4. Handles missing values
    
    Objective Function:
    Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
    
    Where:
    - L: Loss function
    - Ω: Regularization term (L1 + L2)
    - f_k: k-th tree
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 reg_lambda: float = 1.0, reg_alpha: float = 0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda  # L2 regularization
        self.reg_alpha = reg_alpha    # L1 regularization
        self.trees = []
        self.initial_prediction = None
    
    def compute_gradients(self, y: np.ndarray, predictions: np.ndarray):
        """
        Compute gradients and hessians
        
        For squared error loss:
        - Gradient: -2(y - pred)
        - Hessian: 2 (constant)
        """
        gradients = -2 * (y - predictions)
        hessians = np.full(len(y), 2.0)
        return gradients, hessians
    
    def find_best_split_xgb(self, X: np.ndarray, gradients: np.ndarray,
                           hessians: np.ndarray):
        """
        Find best split using XGBoost gain formula
        
        Gain = 1/2 [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ
        
        Where:
        - GL, GR: Sum of gradients in left/right
        - HL, HR: Sum of hessians in left/right
        - λ: L2 regularization
        - γ: Minimum gain (not implemented here)
        """
        best_gain = float('-inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            values = np.unique(X[:, feature_idx])
            for threshold in values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Sum gradients and hessians
                GL = np.sum(gradients[left_mask])
                GR = np.sum(gradients[right_mask])
                HL = np.sum(hessians[left_mask])
                HR = np.sum(hessians[right_mask])
                
                # XGBoost gain formula
                gain = (0.5 * (GL**2 / (HL + self.reg_lambda) +
                              GR**2 / (HR + self.reg_lambda) -
                              (GL + GR)**2 / (HL + HR + self.reg_lambda)))
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost"""
        # Initial prediction: mean
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute gradients and hessians
            gradients, hessians = self.compute_gradients(y, predictions)
            
            # Train tree on gradients (simplified - would use XGBoost split)
            # For simplicity, using simple tree
            tree = SimpleTree(max_depth=self.max_depth)
            tree.fit(X, -gradients)  # Negative gradient as target
            
            # Update predictions
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
            
            self.trees.append(tree)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
        
        return predictions


# Pruning Functions

def pre_pruning_example():
    """
    Pre-pruning: Stop before tree is fully grown
    
    Parameters:
    - max_depth: Maximum tree depth
    - min_samples_split: Minimum samples to split node
    - min_samples_leaf: Minimum samples in leaf
    """
    print("Pre-pruning Parameters:")
    print("  - max_depth: Stop at this depth")
    print("  - min_samples_split: Need this many samples to split")
    print("  - min_samples_leaf: Each leaf needs this many samples")
    print("  - max_features: Consider only this many features per split")

def post_pruning_example():
    """
    Post-pruning: Grow full tree, then remove branches
    
    Cost-complexity pruning:
    - Remove branches that don't improve validation error
    - Balance tree complexity vs accuracy
    """
    print("Post-pruning:")
    print("  1. Grow full tree")
    print("  2. Evaluate each subtree on validation set")
    print("  3. Remove branches that don't help")
    print("  4. Stop when validation error increases")


# Usage Example
if __name__ == "__main__":
    print("Gradient Boosting and XGBoost")
    print("=" * 60)
    
    # Generate sample regression data
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    y = X[:, 0]**2 + X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Split
    split_idx = n_samples // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Gradient Boosting
    print("Gradient Boosting:")
    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_mse = np.mean((gb_pred - y_test)**2)
    print(f"  MSE: {gb_mse:.4f}")
    print(f"  Number of trees: {len(gb.trees)}")
    print()
    
    # XGBoost
    print("XGBoost (Simplified):")
    xgb = XGBoostSimplified(n_estimators=50, learning_rate=0.1, 
                           max_depth=3, reg_lambda=1.0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mse = np.mean((xgb_pred - y_test)**2)
    print(f"  MSE: {xgb_mse:.4f}")
    print(f"  Number of trees: {len(xgb.trees)}")
    print()
    
    # Pruning
    print("Pruning:")
    pre_pruning_example()
    print()
    post_pruning_example()

