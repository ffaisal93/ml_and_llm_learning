"""
Learning Curves: Detailed Implementation
Plot training and validation error vs sample size
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def plot_learning_curves(X: np.ndarray, y: np.ndarray, model, 
                        train_sizes: np.ndarray = None):
    """
    Plot learning curves for a model
    
    Learning curves show:
    - Training error vs sample size
    - Validation error vs sample size
    
    Interpretation:
    - Large gap: High variance (overfitting)
    - Both high: High bias (underfitting)
    - Small gap, both low: Good fit
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Compute learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, 
        scoring='accuracy', n_jobs=-1
    )
    
    # Compute mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Interpretation
    final_gap = train_mean[-1] - val_mean[-1]
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    
    print("\nLearning Curve Interpretation:")
    print(f"  Final training score: {final_train:.4f}")
    print(f"  Final validation score: {final_val:.4f}")
    print(f"  Gap: {final_gap:.4f}")
    
    if final_gap > 0.1:
        print("  → High Variance (Overfitting): Large gap between curves")
        print("    Solution: More data, regularization, simpler model")
    elif final_train < 0.7 and final_val < 0.7:
        print("  → High Bias (Underfitting): Both scores are low")
        print("    Solution: More complex model, better features")
    else:
        print("  → Good Fit: Small gap, both scores are good")
    
    return train_sizes_abs, train_mean, val_mean


def diagnose_model_with_learning_curves(X: np.ndarray, y: np.ndarray):
    """
    Diagnose model problems using learning curves
    
    Compares simple model (high bias) vs complex model (high variance)
    """
    print("Model Diagnosis with Learning Curves")
    print("=" * 60)
    
    # Simple model (might have high bias)
    print("\n1. Simple Model (Logistic Regression):")
    simple_model = LogisticRegression(max_iter=1000)
    plot_learning_curves(X, y, simple_model)
    
    # Complex model (might have high variance)
    print("\n2. Complex Model (Deep Decision Tree):")
    complex_model = DecisionTreeClassifier(max_depth=20)
    plot_learning_curves(X, y, complex_model)
    
    print("\nCompare the two:")
    print("  - Simple model: Both curves low? → High bias")
    print("  - Complex model: Large gap? → High variance")


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=10, n_redundant=10,
                              random_state=42)
    
    print("Learning Curves Example")
    print("=" * 60)
    
    # Example: Model with potential overfitting
    model = DecisionTreeClassifier(max_depth=10)
    train_sizes, train_scores, val_scores = plot_learning_curves(X, y, model)
    
    print("\nWhat to look for:")
    print("  1. Gap between curves: Indicates overfitting")
    print("  2. Both curves low: Indicates underfitting")
    print("  3. Curves converging: Good fit")
    print("  4. Validation still improving: Need more data")

