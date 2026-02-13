"""
SVM with Different Kernels: Complete Example
Shows how to use different kernels in practice
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_nonlinear_data():
    """
    Create non-linearly separable data (concentric circles)
    """
    np.random.seed(42)
    n_samples = 200
    
    # Inner circle (class 0)
    angles = np.random.uniform(0, 2*np.pi, n_samples//2)
    radii = np.random.uniform(0, 1, n_samples//2)
    x_inner = radii * np.cos(angles)
    y_inner = radii * np.sin(angles)
    inner = np.column_stack([x_inner, y_inner])
    labels_inner = np.zeros(n_samples//2)
    
    # Outer circle (class 1)
    angles = np.random.uniform(0, 2*np.pi, n_samples//2)
    radii = np.random.uniform(2, 3, n_samples//2)
    x_outer = radii * np.cos(angles)
    y_outer = radii * np.sin(angles)
    outer = np.column_stack([x_outer, y_outer])
    labels_outer = np.ones(n_samples//2)
    
    X = np.vstack([inner, outer])
    y = np.hstack([labels_inner, labels_outer])
    
    return X, y

def compare_kernels_on_data():
    """
    Compare different kernels on the same dataset
    """
    print("Comparing Kernels on Non-Linear Data")
    print("=" * 60)
    
    # Create data
    X, y = create_nonlinear_data()
    
    # Scale features (important for SVM!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Try different kernels
    kernels = {
        'Linear': {'kernel': 'linear', 'C': 1.0},
        'Polynomial (degree=2)': {'kernel': 'poly', 'degree': 2, 'C': 1.0},
        'Polynomial (degree=3)': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        'RBF (gamma=0.1)': {'kernel': 'rbf', 'gamma': 0.1, 'C': 1.0},
        'RBF (gamma=1.0)': {'kernel': 'rbf', 'gamma': 1.0, 'C': 1.0},
        'RBF (gamma=10.0)': {'kernel': 'rbf', 'gamma': 10.0, 'C': 1.0},
    }
    
    results = {}
    
    for name, params in kernels.items():
        svm = SVC(**params)
        svm.fit(X_train, y_train)
        
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        n_support = len(svm.support_vectors_)
        
        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_support': n_support
        }
    
    # Print results
    print("\nResults:")
    print(f"{'Kernel':<25} {'Train Acc':<12} {'Test Acc':<12} {'Support Vectors':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<25} {result['train_acc']:<12.4f} {result['test_acc']:<12.4f} {result['n_support']:<15}")
    
    print("\nObservations:")
    print("  - Linear kernel: Fails (can't separate circles)")
    print("  - Polynomial: Works (can handle quadratic boundaries)")
    print("  - RBF: Works best (flexible boundaries)")
    print("  - Higher gamma: More support vectors (more complex)")

def kernel_parameter_tuning():
    """
    Show how to tune kernel parameters
    """
    print("\nKernel Parameter Tuning")
    print("=" * 60)
    
    X, y = create_nonlinear_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Tune gamma for RBF
    print("\nTuning RBF Gamma:")
    gammas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    print(f"{'Gamma':<10} {'Train Acc':<12} {'Test Acc':<12} {'Support Vectors':<15}")
    print("-" * 50)
    
    for gamma in gammas:
        svm = SVC(kernel='rbf', gamma=gamma, C=1.0)
        svm.fit(X_train, y_train)
        
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        n_support = len(svm.support_vectors_)
        
        print(f"{gamma:<10.3f} {train_acc:<12.4f} {test_acc:<12.4f} {n_support:<15}")
    
    print("\nInterpretation:")
    print("  - Low gamma (0.001): Underfitting (too simple)")
    print("  - Medium gamma (0.1-1.0): Good balance")
    print("  - High gamma (10.0): Overfitting (too complex, many support vectors)")

def kernel_trick_visualization():
    """
    Explain kernel trick with example
    """
    print("\nKernel Trick: How It Works")
    print("=" * 60)
    
    print("""
Example: Polynomial Kernel (degree=2)

Original space: x = [x₁, x₂]

Without kernel trick:
  Transform to: φ(x) = [x₁, x₂, x₁², x₂², √2x₁x₂, √2x₁, √2x₂, 1]
  Compute: φ(x) · φ(y) (expensive, 8 dimensions)

With kernel trick:
  Just compute: K(x, y) = (x · y)²
  Same result, but only 2 dimensions!

Why it works:
  φ(x) · φ(y) = (x₁y₁ + x₂y₂)²
              = x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂² + ...
  
  This equals: (x · y)² = (x₁y₁ + x₂y₂)²
  
  So we get the same result without computing high-dimensional features!

Benefit:
  - Original: O(d²) computation (d = dimension of φ(x))
  - Kernel: O(d) computation (d = dimension of x)
  - Much faster!
    """)

# Usage
if __name__ == "__main__":
    print("SVM with Kernels: Practical Examples")
    print("=" * 60)
    
    compare_kernels_on_data()
    kernel_parameter_tuning()
    kernel_trick_visualization()

