# Kernel Functions: Detailed Examples

## Example 1: Linearly Separable Data

**Problem:**
Two classes that can be separated by a straight line.

**Data:**
- Class 0: Points with x₁ + x₂ < 0
- Class 1: Points with x₁ + x₂ > 0

**Which kernel?**
- **Linear kernel**: Perfect! Data is linearly separable
- **RBF kernel**: Works but unnecessary (overkill)
- **Polynomial kernel**: Works but unnecessary

**Why linear works:**
The decision boundary is a line: x₁ + x₂ = 0. Linear kernel can find this line directly.

**Code:**
```python
# Linear kernel is perfect
svm = SVC(kernel='linear')
svm.fit(X, y)  # Will find the line x₁ + x₂ = 0
```

## Example 2: Circular Boundary (Concentric Circles)

**Problem:**
Two classes arranged in concentric circles:
- Class 0: Inner circle (radius < 2)
- Class 1: Outer circle (radius > 2)

**Data:**
- Points where x₁² + x₂² < 4 → Class 0
- Points where x₁² + x₂² > 4 → Class 1

**Which kernel?**
- **Linear kernel**: Fails! Can't separate circles with a line
- **RBF kernel**: Perfect! Can create circular boundaries
- **Polynomial kernel (degree=2)**: Works! Can handle quadratic boundaries

**Why RBF works:**
RBF kernel creates local neighborhoods. Points near the origin (inner circle) are similar to each other, points far from origin (outer circle) are similar to each other, but inner and outer are different.

**Visual:**
```
    Class 1 (outer)
  • • • • • • • • •
  •               •
• •     Class 0   • •
• •     (inner)   • •
  •               •
  • • • • • • • • •
    Class 1 (outer)
```

**Code:**
```python
# RBF kernel works
svm = SVC(kernel='rbf', gamma=1.0)
svm.fit(X, y)  # Creates circular decision boundary

# Polynomial kernel (degree=2) also works
svm = SVC(kernel='poly', degree=2)
svm.fit(X, y)  # Can handle quadratic boundaries
```

## Example 3: XOR Problem

**Problem:**
Classic XOR problem:
- (0, 0) → Class 0
- (0, 1) → Class 1
- (1, 0) → Class 1
- (1, 1) → Class 0

**Which kernel?**
- **Linear kernel**: Fails! XOR is not linearly separable
- **RBF kernel**: Works! Can create complex boundaries
- **Polynomial kernel (degree=2)**: Works! Can handle XOR

**Why linear fails:**
No single line can separate the classes. You need a non-linear boundary.

**Why RBF/Polynomial work:**
They can create non-linear boundaries that separate the classes.

**Code:**
```python
# Linear fails
svm_linear = SVC(kernel='linear')
# Won't work well

# RBF works
svm_rbf = SVC(kernel='rbf', gamma=1.0)
svm_rbf.fit(X, y)  # Works!

# Polynomial works
svm_poly = SVC(kernel='poly', degree=2)
svm_poly.fit(X, y)  # Works!
```

## Example 4: Text Classification

**Problem:**
Classify documents (spam/not spam) using TF-IDF features.

**Data:**
- High-dimensional (thousands of features)
- Sparse (most features are 0)
- Often linearly separable in high dimensions

**Which kernel?**
- **Linear kernel**: Usually best! High-dimensional data is often linearly separable
- **RBF kernel**: Can work but often overfits
- **Polynomial kernel**: Rarely needed

**Why linear works:**
In high dimensions, data is often linearly separable (curse of dimensionality works in our favor here). Linear kernel is fast and works well.

**Code:**
```python
# Linear kernel is best for text
svm = SVC(kernel='linear', C=1.0)
svm.fit(tfidf_features, labels)
```

## Example 5: Image Classification (Small Images)

**Problem:**
Classify small images (e.g., 32x32 pixels = 1024 features).

**Data:**
- High-dimensional but structured
- Non-linear patterns (edges, textures)

**Which kernel?**
- **Linear kernel**: Might work if features are good
- **RBF kernel**: Usually better (captures non-linear patterns)
- **Polynomial kernel**: Can work but RBF usually better

**Why RBF often better:**
Images have complex non-linear patterns. RBF kernel can capture these better than linear.

**Code:**
```python
# RBF kernel for images
svm = SVC(kernel='rbf', gamma=0.001, C=10.0)
svm.fit(image_features, labels)

# Tune gamma: too high = overfitting, too low = underfitting
```

## Example 6: High-Dimensional Sparse Data

**Problem:**
Data with many features (e.g., 10,000 features) but most are 0 (sparse).

**Which kernel?**
- **Linear kernel**: Best choice! Sparse high-dimensional data is often linearly separable
- **RBF kernel**: Can be slow and overfit
- **Polynomial kernel**: Usually not needed

**Why linear:**
- Fast computation (sparse dot products are efficient)
- High-dimensional spaces often allow linear separation
- Less prone to overfitting

**Code:**
```python
# Linear kernel for sparse high-dimensional data
svm = SVC(kernel='linear', C=1.0)
svm.fit(sparse_features, labels)
```

## Parameter Tuning Examples

### RBF Kernel: Gamma Selection

**Low Gamma (γ = 0.001):**
- Wide kernel (large influence radius)
- Simpler boundaries
- Risk: Underfitting
- Use when: Data has smooth, simple patterns

**Medium Gamma (γ = 0.1):**
- Moderate kernel width
- Balanced complexity
- Good starting point
- Use when: Not sure, start here

**High Gamma (γ = 10.0):**
- Narrow kernel (small influence radius)
- Complex boundaries
- Risk: Overfitting
- Use when: Data has very complex patterns

**How to choose:**
1. Start with γ = 1 / (n_features * variance)
2. Try grid search: [0.001, 0.01, 0.1, 1.0, 10.0]
3. Use cross-validation to select best

### Polynomial Kernel: Degree Selection

**Degree = 1:**
- Same as linear kernel
- Use when: Data is linear

**Degree = 2:**
- Quadratic features
- Most common
- Use when: Moderate non-linearity

**Degree = 3:**
- Cubic features
- More complex
- Risk: Overfitting
- Use when: Strong non-linearity

**Degree > 3:**
- Rarely used
- High overfitting risk
- Avoid unless necessary

## Decision Tree for Kernel Selection

```
Start
  ↓
Is data linearly separable?
  ├─ Yes → Use Linear Kernel
  │
  └─ No → Try RBF Kernel
           ↓
        Does it overfit?
          ├─ No → Use RBF Kernel
          │
          └─ Yes → Try Polynomial Kernel (degree=2)
                     ↓
                  Does it work?
                    ├─ Yes → Use Polynomial Kernel
                    │
                    └─ No → Tune RBF parameters or use different model
```

## Common Mistakes

**Mistake 1: Always using RBF**
- Linear kernel is often better for high-dimensional data
- Always try linear first

**Mistake 2: Not scaling features**
- SVM is sensitive to feature scales
- Always scale before using kernels (especially RBF)

**Mistake 3: Gamma too high**
- Causes overfitting
- Start with lower gamma, increase if needed

**Mistake 4: Ignoring linear kernel**
- Linear kernel is fast and interpretable
- Don't skip it!

## Summary Table

| Kernel | Use When | Parameters | Pros | Cons |
|--------|----------|------------|------|------|
| **Linear** | Linearly separable, high-dim | C only | Fast, interpretable | Can't handle non-linear |
| **Polynomial** | Polynomial relationships | degree, gamma, coef0 | Captures polynomials | Can overfit, need to tune degree |
| **RBF** | Non-linear (default) | gamma, C | Very flexible, one param | Can overfit, less interpretable |
| **Sigmoid** | Rarely | gamma, coef0 | Neural-like | Unstable, rarely best |

## Key Takeaways

1. **Start with linear**: It's fast and often works
2. **Use RBF for non-linear**: Most common choice
3. **Scale features**: Critical for SVM
4. **Tune parameters**: Gamma and C matter a lot
5. **Avoid sigmoid**: RBF is almost always better

