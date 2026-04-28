# Topic 35: Kernel Functions

> 🔥 **For interviews, read these first:**
> - **`KERNELS_DEEP_DIVE.md`** — frontier-lab interview deep dive: kernel trick, Mercer's theorem, RBF/polynomial, SVM dual derivation, RKHS, NTK, attention-as-kernel.
> - **`INTERVIEW_GRILL.md`** — 40 active-recall questions.

## What You'll Learn

This topic covers kernel functions in detail:
- What kernels are and why we need them
- Linear kernel
- Polynomial kernel
- RBF (Radial Basis Function) kernel
- Sigmoid kernel
- How kernels work mathematically
- When to use each kernel
- Kernel trick explained
- Simple implementations with examples

## Why We Need This

### Interview Importance
- **Common questions**: "Explain kernel trick", "Which kernel to use?"
- **SVM knowledge**: Kernels are essential for SVM
- **Understanding**: Shows deep ML knowledge

### Real-World Application
- **SVM**: Kernels make SVM powerful
- **Non-linear problems**: Kernels enable non-linear classification
- **Feature engineering**: Kernels implicitly create features

## Core Intuition

Kernels let linear algorithms behave as if they were operating in a richer feature space without explicitly constructing that space.

That is the essence of the kernel trick.

### Why This Matters

If a problem is not linearly separable in the original space, a nonlinear feature mapping may make it separable.

Kernels let you get that benefit by computing inner products in the transformed space directly.

### Different Kernels Imply Different Similarity Notions

- linear kernel: ordinary dot-product similarity
- polynomial kernel: similarity with interaction structure
- RBF kernel: locality-based similarity

So kernel choice is really an inductive-bias choice.

## Technical Details Interviewers Often Want

### Why the Kernel Trick Works

Many learning algorithms only need dot products between examples.

If you can replace `x^T y` with `K(x, y) = φ(x)^T φ(y)`, you can behave as if the model used the transformed feature space without building it explicitly.

### RBF Gamma Trade-Off

This is one of the most common kernel follow-ups.

- high gamma -> very local influence, more complex boundary
- low gamma -> smoother, broader similarity, simpler boundary

### Linear Kernel Can Still Be Strong

Candidates often over-romanticize nonlinear kernels.

Linear kernels work very well when:
- feature dimension is already high
- representation is already informative
- overfitting risk matters

## Common Failure Modes

- choosing kernels without considering data structure
- using RBF as a default without thinking about gamma
- thinking kernels are only about SVM rather than about similarity in transformed space
- forgetting that nonlinear flexibility can overfit

## Edge Cases and Follow-Up Questions

1. Why is the kernel trick computationally valuable?
2. Why can linear kernels still work very well in high-dimensional settings?
3. What does gamma do in the RBF kernel?
4. Why can polynomial kernels overfit at high degree?
5. Why is kernel choice really a similarity assumption?

## What to Practice Saying Out Loud

1. The kernel trick in one clean sentence
2. Why RBF creates local similarity behavior
3. Why kernel selection is an inductive-bias decision

## Detailed Theory

### What is a Kernel Function?

**Definition:**
A kernel function K(x, y) computes the dot product of two vectors in a high-dimensional feature space without explicitly computing the transformation. It's a way to measure similarity between data points in a transformed space.

**Mathematical Formulation:**
```
K(x, y) = φ(x) · φ(y)

Where:
- φ(x): Transformation to high-dimensional space
- K(x, y): Kernel function (dot product in transformed space)
- We never compute φ(x) explicitly (kernel trick)
```

**Why Kernels Matter:**
- **Non-linearity**: Enable non-linear decision boundaries
- **Efficiency**: Don't need to compute high-dimensional features
- **Flexibility**: Can use infinite-dimensional spaces
- **Power**: Make linear algorithms work on non-linear data

**The Kernel Trick:**
Instead of transforming data to high dimensions (expensive), we use a kernel function that computes the dot product in that space directly (cheap). This is the "kernel trick" - we get the benefit of high-dimensional features without the computational cost.

### Linear Kernel

**Mathematical Formulation:**
```
K(x, y) = x · y = x^T y

This is just the standard dot product.
```

**What it does:**
The linear kernel computes the standard dot product between two vectors. It assumes the data is linearly separable (or nearly so) in the original feature space.

**When to use:**
- **Linearly separable data**: Data can be separated by a straight line/plane
- **High-dimensional data**: When you already have many features
- **Baseline**: Start here, try others if it doesn't work
- **Interpretability**: Linear kernel is more interpretable

**Example:**
If you have text data with many features (TF-IDF vectors), linear kernel often works well because the high dimensionality already provides separation.

**Advantages:**
- Fast (simple computation)
- Interpretable (linear decision boundary)
- Less prone to overfitting
- Works well with many features

**Disadvantages:**
- Can't handle non-linear relationships
- Fails if data is not linearly separable

**Code Example:**
```python
def linear_kernel(x, y):
    """Linear kernel: K(x, y) = x · y"""
    return np.dot(x, y)

# Example
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
similarity = linear_kernel(x, y)  # 1*4 + 2*5 + 3*6 = 32
```

### Polynomial Kernel

**Mathematical Formulation:**
```
K(x, y) = (γ * x^T y + r)^d

Where:
- γ (gamma): Coefficient (controls influence of higher-order terms)
- r (coef0): Constant term (bias)
- d (degree): Polynomial degree
```

**What it does:**
The polynomial kernel computes the dot product in a polynomial feature space. It creates features like x₁², x₂², x₁x₂, etc., without explicitly computing them.

**How it works:**
For degree d=2, the kernel implicitly creates features like:
- Original: x₁, x₂
- Implicit: x₁, x₂, x₁², x₂², x₁x₂, √2x₁, √2x₂, ...

The kernel computes the dot product in this expanded space efficiently.

**When to use:**
- **Polynomial relationships**: When relationship is polynomial
- **Moderate non-linearity**: Not too complex, not too simple
- **When you know degree**: If you know the relationship is quadratic, cubic, etc.

**Parameters:**
- **degree (d)**: Higher = more complex, but risk of overfitting
  - d=2: Quadratic (common)
  - d=3: Cubic
  - d>3: Rarely used (overfitting risk)
- **gamma (γ)**: Controls influence of higher-order terms
  - Higher γ: More influence of polynomial terms
  - Lower γ: More like linear kernel
- **coef0 (r)**: Bias term
  - Higher r: More influence of lower-degree terms

**Advantages:**
- Captures polynomial relationships
- More flexible than linear
- Interpretable (polynomial degree)

**Disadvantages:**
- Can overfit with high degree
- Computationally more expensive than linear
- Need to tune degree parameter

**Example:**
If you have data where y = x₁² + x₂² (circular boundary), polynomial kernel with degree=2 can separate it, but linear kernel cannot.

**Code Example:**
```python
def polynomial_kernel(x, y, degree=2, gamma=1.0, coef0=0.0):
    """Polynomial kernel: K(x, y) = (γ * x^T y + r)^d"""
    return (gamma * np.dot(x, y) + coef0) ** degree

# Example
x = np.array([1, 2])
y = np.array([3, 4])
# For degree=2: (1*3 + 2*4 + 0)^2 = 11^2 = 121
similarity = polynomial_kernel(x, y, degree=2)  # 121
```

### RBF (Radial Basis Function) Kernel

**Mathematical Formulation:**
```
K(x, y) = exp(-γ * ||x - y||²)

Where:
- ||x - y||²: Squared Euclidean distance
- γ (gamma): Controls the "width" of the kernel
- Also called: Gaussian kernel (when γ = 1/(2σ²))
```

**What it does:**
The RBF kernel measures similarity based on distance. Points that are close together have high similarity (close to 1), points that are far apart have low similarity (close to 0). It creates an infinite-dimensional feature space.

**How it works:**
The RBF kernel implicitly maps data to an infinite-dimensional space where:
- Similar points (close in original space) are close
- Different points (far in original space) are far apart
- Creates local neighborhoods around each point

**Visual intuition:**
Imagine each data point creates a "bump" (Gaussian) in the feature space. The kernel value is high when two points are in the same bump (close together), and low when they're in different bumps (far apart).

**When to use:**
- **Non-linear problems**: When data is not linearly separable
- **Local structure**: When similar points should be close
- **Default choice**: Often works well as default for non-linear problems
- **Complex boundaries**: Can create very complex decision boundaries

**Parameters:**
- **gamma (γ)**: Controls kernel width
  - **High γ**: Narrow kernel (small influence radius)
    - Each point only influences nearby points
    - More complex boundaries (risk of overfitting)
    - More support vectors
  - **Low γ**: Wide kernel (large influence radius)
    - Each point influences many points
    - Simpler boundaries (risk of underfitting)
    - Fewer support vectors
  - **Rule of thumb**: γ = 1 / (n_features * variance)

**Advantages:**
- Very flexible (can handle complex boundaries)
- Works well for most non-linear problems
- Only one parameter to tune (gamma)
- Creates smooth decision boundaries

**Disadvantages:**
- Can overfit with high gamma
- Computationally more expensive
- Less interpretable
- Sensitive to gamma parameter

**Example:**
If you have data arranged in concentric circles (inner circle = class 0, outer circle = class 1), RBF kernel can separate them, but linear and polynomial kernels cannot.

**Code Example:**
```python
def rbf_kernel(x, y, gamma=1.0):
    """RBF kernel: K(x, y) = exp(-γ * ||x - y||²)"""
    distance_squared = np.sum((x - y) ** 2)
    return np.exp(-gamma * distance_squared)

# Example
x = np.array([0, 0])
y = np.array([1, 1])
# Distance = √2, distance² = 2
# For gamma=1: exp(-1 * 2) = exp(-2) ≈ 0.135
similarity = rbf_kernel(x, y, gamma=1.0)  # ≈ 0.135 (not very similar)

# Same point
similarity_same = rbf_kernel(x, x, gamma=1.0)  # 1.0 (identical)
```

### Sigmoid Kernel

**Mathematical Formulation:**
```
K(x, y) = tanh(γ * x^T y + r)

Where:
- γ (gamma): Scaling parameter
- r (coef0): Bias term
- tanh: Hyperbolic tangent function
```

**What it does:**
The sigmoid kernel is similar to a neural network activation function. It's less commonly used than RBF or polynomial kernels.

**When to use:**
- **Neural network-like behavior**: When you want sigmoid-like activation
- **Rarely used**: RBF is usually better
- **Specific use cases**: Some specific problems where it works well

**Advantages:**
- Similar to neural network
- Can work for some non-linear problems

**Disadvantages:**
- Less stable than RBF
- Not always positive definite (can cause issues)
- Rarely the best choice

**Note:** Sigmoid kernel is rarely used in practice. RBF is usually preferred for non-linear problems.

## Kernel Selection Guide

### Decision Tree

**Step 1: Try Linear Kernel**
- Fast, interpretable
- If it works, use it!
- Good for high-dimensional data

**Step 2: If Linear Fails, Try RBF**
- Most common for non-linear problems
- Tune gamma parameter
- Usually works well

**Step 3: If RBF Overfits, Try Polynomial**
- Less flexible than RBF
- More interpretable
- Try degree=2 or 3

**Step 4: Never Use Sigmoid**
- Unless you have specific reason
- RBF is almost always better

### Parameter Tuning

**For RBF Kernel:**
- **Gamma**: 
  - Too high: Overfitting (narrow kernel)
  - Too low: Underfitting (wide kernel)
  - Try: 0.001, 0.01, 0.1, 1.0, 10.0
- **C (regularization)**:
  - Higher C: Less regularization (more complex boundary)
  - Lower C: More regularization (simpler boundary)
  - Try: 0.1, 1, 10, 100, 1000

**For Polynomial Kernel:**
- **Degree**: Start with 2, try 3 if needed
- **Gamma**: Usually 1.0 or scale with 1/n_features
- **Coef0**: Usually 0.0

## Industry Use Cases

### 1. **Text Classification**
- **Kernel**: Linear (high-dimensional TF-IDF vectors)
- **Why**: Text data is already high-dimensional, linear often works

### 2. **Image Classification**
- **Kernel**: RBF (non-linear patterns)
- **Why**: Images have complex non-linear patterns

### 3. **Bioinformatics**
- **Kernel**: RBF or polynomial
- **Why**: Complex biological relationships

### 4. **Time Series**
- **Kernel**: RBF
- **Why**: Non-linear temporal patterns

## Exercises

1. Implement all kernel functions
2. Compare kernels on same dataset
3. Tune kernel parameters
4. Visualize decision boundaries

## Next Steps

- Use kernels in SVM
- Understand kernel trick deeply
- Choose right kernel for your problem
