# Topic 26: Tree-Based Methods

> 🔥 **For interviews, read these first:**
> - **`TREES_DEEP_DIVE.md`** — frontier-lab interview deep dive: Gini/entropy splits, RF decorrelation math, gradient boosting derivation, XGBoost regularized objective + second-order trick, LightGBM/CatBoost innovations, categorical handling, missing values.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions.

## What You'll Learn

This topic teaches you tree-based algorithms:
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- How they're learned
- Pruning techniques
- When to use what

## Why We Need This

### Interview Importance
- **Common question**: "How does random forest work?"
- **Practical knowledge**: Used in many companies
- **Understanding**: Foundation for ensemble methods

### Real-World Application
- **Tabular data**: Best for structured data
- **Feature importance**: Interpretable
- **Production**: Used in many ML systems

## Industry Use Cases

### 1. **Decision Trees**
**Use Case**: Simple, interpretable models
- Rule-based systems
- Feature selection
- Baseline models

### 2. **Random Forest**
**Use Case**: Robust, general-purpose
- Default choice for tabular data
- Feature importance
- Handles missing values

### 3. **Gradient Boosting**
**Use Case**: High performance
- Kaggle competitions
- Production systems
- When you need best accuracy

### 4. **XGBoost**
**Use Case**: Optimized gradient boosting
- Fast and efficient
- Handles large datasets
- Industry standard

## Core Intuition

Tree-based methods are powerful because they model nonlinear decision rules without requiring heavy feature preprocessing.

They are especially strong on tabular data because they can naturally handle:
- nonlinear feature interactions
- thresholds
- mixed feature importance

### Decision Trees

A decision tree recursively asks simple threshold questions.

Intuition:
- split the space into regions
- make each region more pure or more predictable

### Random Forest

Random forest improves a single tree by averaging many de-correlated trees.

The key idea is variance reduction:
- one tree is unstable
- many randomized trees averaged together are much more stable

### Gradient Boosting

Gradient boosting improves predictions sequentially.

Each new tree tries to correct what the current ensemble is still getting wrong.

That is why boosting often gets very strong accuracy but can overfit if pushed too hard.

## Technical Details Interviewers Often Want

### Why Trees Overfit Easily

Deep trees can carve the feature space into very specific regions.

That gives low training error, but it can memorize noise.

### Why Random Forest Helps

Bootstrap sampling plus random feature subsets make trees less correlated.

Averaging less-correlated estimators is what reduces variance.

### Why Boosting Is Different from Bagging

This is a classic interview question.

- **Bagging / Random Forest**: train many models independently, then average
- **Boosting**: train sequentially so later models fix earlier errors

### XGBoost Advantage

XGBoost is not just "gradient boosting but faster."

Important points:
- second-order optimization information
- regularization
- system-level efficiency
- handling of sparse or missing data

## Common Failure Modes

- growing trees too deep and overfitting
- assuming random forests and boosting solve the same problem in the same way
- using boosting without enough tuning on noisy data
- over-interpreting feature importance without care

## Edge Cases and Follow-Up Questions

1. Why are trees so strong on tabular data?
2. Why is a single decision tree unstable?
3. Why does random forest reduce variance?
4. Why can gradient boosting outperform random forest but overfit more easily?
5. What is the conceptual difference between bagging and boosting?

## What to Practice Saying Out Loud

1. Why tree methods model nonlinear interactions well
2. Why random forest and boosting are fundamentally different ensemble ideas
3. Why XGBoost became so dominant in tabular ML

## Theory

### Decision Tree Learning

**Algorithm (ID3/CART):**
1. Start with root node (all data)
2. For each feature, find best split (maximize information gain)
3. Split data based on best feature
4. Recursively build left and right subtrees
5. Stop when: max depth reached, min samples, or pure node

**Splitting Criterion:**
- **Classification**: Gini impurity or entropy
- **Regression**: MSE (Mean Squared Error)

**Gini Impurity:**
```
Gini = 1 - Σ(p_i)²
where p_i is proportion of class i
```

**Information Gain:**
```
Gain = Entropy(parent) - Σ(|child_i|/|parent|) × Entropy(child_i)
```

### Random Forest

**Concept:**
- Ensemble of decision trees
- Each tree trained on random subset of data (bootstrap)
- Each split uses random subset of features
- Final prediction: Average (regression) or majority vote (classification)

**Why it works:**
- Reduces overfitting (variance reduction)
- More robust than single tree
- Handles missing values

**Parameters:**
- `n_estimators`: Number of trees (100-1000)
- `max_depth`: Max depth of trees
- `max_features`: Features to consider per split (sqrt(n_features))
- `min_samples_split`: Min samples to split node

### Gradient Boosting

**Concept:**
- Sequentially add trees
- Each tree corrects errors of previous trees
- Train on residuals (errors) of previous model

**Algorithm:**
1. Start with initial prediction (mean/median)
2. For each tree:
   - Compute residuals (errors)
   - Train tree on residuals
   - Add tree to ensemble (with learning rate)
3. Final prediction: Sum of all trees

**Mathematical Formulation:**
```
F_m(x) = F_{m-1}(x) + α × h_m(x)

Where:
- F_m: Model after m trees
- h_m: m-th tree
- α: Learning rate (shrinkage)
```

**Parameters:**
- `n_estimators`: Number of trees (100-1000)
- `learning_rate`: Shrinkage factor (0.01-0.3)
- `max_depth`: Tree depth (3-8)
- `subsample`: Fraction of data per tree (0.8-1.0)

### XGBoost

**Concept:**
- Optimized gradient boosting
- Uses second-order gradient (Hessian)
- Regularization (L1, L2)
- Parallel tree construction

**Key Differences from Gradient Boosting:**
- Uses second-order approximation
- Built-in regularization
- Handles missing values
- More efficient

**Objective Function:**
```
Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)

Where:
- L: Loss function
- Ω: Regularization term
- f_k: k-th tree
```

## Industry-Standard Boilerplate Code

### Decision Tree (Simplified)

```python
"""
Decision Tree from Scratch
"""
import numpy as np

class DecisionTree:
    """
    Simple decision tree
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y: np.ndarray) -> float:
        """Gini impurity for classification"""
        if len(y) == 0:
            return 0.0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions**2)
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find best feature and threshold to split"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            # Try different thresholds
            values = np.unique(X[:, feature_idx])
            for threshold in values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Compute weighted Gini
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])
                weighted_gini = (np.sum(left_mask) * left_gini + 
                                np.sum(right_mask) * right_gini) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build tree"""
        # Stopping conditions
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return np.bincount(y).argmax()  # Return majority class
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        node = {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train tree"""
        self.tree = self.build_tree(X, y)
    
    def predict_one(self, x: np.ndarray, node) -> int:
        """Predict single sample"""
        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self.predict_one(x, node['left'])
            else:
                return self.predict_one(x, node['right'])
        else:
            return node
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        return np.array([self.predict_one(x, self.tree) for x in X])
```

### Random Forest (Simplified)

```python
"""
Random Forest from Scratch
"""
import numpy as np
from collections import Counter

class RandomForest:
    """
    Random Forest: Ensemble of decision trees
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5,
                 max_features: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    def bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest"""
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self.bootstrap_sample(X, y)
            
            # Random feature subset
            feature_indices = np.random.choice(
                X.shape[1], self.max_features, replace=False
            )
            X_boot = X_boot[:, feature_indices]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth)
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
```

## Pruning

### Pre-pruning (Early Stopping)
- Stop before tree is fully grown
- Parameters: max_depth, min_samples_split, min_samples_leaf

### Post-pruning
- Grow full tree, then remove branches
- Cost-complexity pruning
- Remove branches that don't improve validation error

## When to Use What

| Method | Use When | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **Decision Tree** | Need interpretability | Simple, interpretable | Overfits easily |
| **Random Forest** | General purpose | Robust, handles missing values | Less interpretable |
| **Gradient Boosting** | Need best accuracy | High performance | Can overfit, slower |
| **XGBoost** | Large datasets | Fast, efficient | More complex |

## Exercises

1. Implement decision tree
2. Implement random forest
3. Compare with/without pruning
4. Tune hyperparameters

## Next Steps

- Review all tree methods
- Practice implementations
