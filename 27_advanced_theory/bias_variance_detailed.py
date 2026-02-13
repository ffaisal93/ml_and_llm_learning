"""
Bias-Variance Tradeoff: Detailed Theory and Code
"""
import numpy as np
import matplotlib.pyplot as plt

def bias_variance_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute bias and variance components
    
    Mathematical Formulation:
    E[(y - f(x))²] = Bias² + Variance + Irreducible Error
    
    Where:
    - Bias² = (E[f(x)] - E[y])²
    - Variance = E[(f(x) - E[f(x)])²]
    - Irreducible Error = σ² (noise in data)
    """
    # Bias: Average prediction - true value
    bias = np.mean(y_pred) - np.mean(y_true)
    bias_squared = bias**2
    
    # Variance: Variance of predictions
    variance = np.var(y_pred)
    
    # Total error
    total_error = np.mean((y_true - y_pred)**2)
    
    # Irreducible error (estimated from residuals)
    irreducible_error = total_error - bias_squared - variance
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'irreducible_error': max(0, irreducible_error),
        'total_error': total_error
    }

def diagnose_model(train_error: float, test_error: float) -> str:
    """
    Diagnose model based on training and test error
    
    High Bias (Underfitting):
    - High train error
    - High test error
    - Similar train and test error
    
    High Variance (Overfitting):
    - Low train error
    - High test error
    - Large gap between train and test
    """
    gap = test_error - train_error
    
    if train_error > 0.3 and test_error > 0.3:
        if abs(gap) < 0.1:
            return "High Bias (Underfitting) - Model too simple"
        else:
            return "Both High Bias and Variance"
    elif train_error < 0.1 and test_error > 0.3:
        return "High Variance (Overfitting) - Model too complex"
    elif train_error < 0.1 and test_error < 0.2:
        return "Good Fit - Balanced model"
    else:
        return "Needs investigation"

def solutions_for_bias_variance(problem: str) -> list:
    """
    Solutions for bias-variance problems
    """
    solutions = {
        'high_bias': [
            "Increase model complexity (more layers, features)",
            "Add more features (domain knowledge)",
            "Train longer (more epochs)",
            "Reduce regularization",
            "Use ensemble methods"
        ],
        'high_variance': [
            "Get more training data",
            "Add regularization (L1, L2, dropout)",
            "Simplify model (fewer layers, features)",
            "Use ensemble methods (averaging)",
            "Early stopping",
            "Cross-validation for hyperparameter tuning"
        ]
    }
    
    if 'bias' in problem.lower():
        return solutions['high_bias']
    elif 'variance' in problem.lower():
        return solutions['high_variance']
    else:
        return solutions['high_bias'] + solutions['high_variance']


# Usage Example
if __name__ == "__main__":
    print("Bias-Variance Tradeoff")
    print("=" * 60)
    
    # Example: High bias scenario
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred_high_bias = np.array([2.5, 2.5, 2.5, 2.5, 2.5])  # Simple model
    
    result = bias_variance_decomposition(y_true, y_pred_high_bias)
    print("High Bias Example:")
    print(f"  Bias²: {result['bias_squared']:.4f}")
    print(f"  Variance: {result['variance']:.4f}")
    print(f"  Total Error: {result['total_error']:.4f}")
    print()
    
    # Example: High variance scenario
    y_pred_high_variance = np.array([0.5, 2.2, 3.1, 4.3, 5.5])  # Overfits
    
    result2 = bias_variance_decomposition(y_true, y_pred_high_variance)
    print("High Variance Example:")
    print(f"  Bias²: {result2['bias_squared']:.4f}")
    print(f"  Variance: {result2['variance']:.4f}")
    print(f"  Total Error: {result2['total_error']:.4f}")
    print()
    
    # Diagnosis
    train_error = 0.05
    test_error = 0.35
    diagnosis = diagnose_model(train_error, test_error)
    print(f"Diagnosis: {diagnosis}")
    print("\nSolutions:")
    for solution in solutions_for_bias_variance(diagnosis):
        print(f"  - {solution}")

