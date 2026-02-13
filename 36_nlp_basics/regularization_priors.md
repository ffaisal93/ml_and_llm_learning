# Bayesian Interpretation of L1/L2 Regularization

## Overview

Regularization in frequentist machine learning corresponds to placing priors on parameters in Bayesian machine learning. Understanding this connection helps explain why regularization works and how to choose regularization strength.

## L2 Regularization (Ridge) = Gaussian Prior

### Frequentist Formulation

**Loss Function:**
```
L(w) = MSE + λ * ||w||²
     = (y - Xw)² + λ * Σ wᵢ²

Minimize: (y - Xw)² + λ||w||²
```

**Gradient:**
```
∂L/∂w = -2X^T(y - Xw) + 2λw = 0
→ w = (X^T X + λI)^(-1) X^T y
```

### Bayesian Interpretation

**Prior Distribution:**
```
w ~ N(0, σ²_prior I)

Where:
- Mean: 0 (parameters centered at 0)
- Variance: σ²_prior = 1/λ
- Higher λ → Smaller variance → Parameters closer to 0
```

**Posterior Distribution:**
```
P(w|data) ∝ P(data|w) * P(w)
          ∝ exp(-MSE/2σ²) * exp(-||w||²/2σ²_prior)
          ∝ exp(-(MSE + λ||w||²)/2)

Where λ = σ²/σ²_prior
```

**Maximum A Posteriori (MAP) Estimation:**
```
MAP = argmax_w P(w|data)
    = argmax_w log P(w|data)
    = argmin_w (MSE + λ||w||²)

This is exactly the regularized loss function!
```

### Detailed Explanation

**What the Gaussian Prior Means:**

The Gaussian prior N(0, 1/λ) assumes that:
1. **Parameters are normally distributed** around 0
2. **Most parameters should be small** (high probability near 0)
3. **Few parameters should be large** (low probability far from 0)
4. **Symmetric**: Positive and negative values equally likely

**Why it Works:**

**1. Prevents Overfitting:**
- Without regularization: Model can learn large parameters to fit noise
- With L2: Large parameters are penalized (low prior probability)
- Model prefers smaller parameters (higher prior probability)

**2. Smooth Shrinkage:**
- All parameters shrunk toward 0
- But rarely exactly 0 (Gaussian has no sharp peak)
- Smooth, continuous shrinkage

**3. Handles Multicollinearity:**
- When features are correlated, coefficients can be unstable
- L2 regularization stabilizes by shrinking toward 0
- Reduces variance of estimates

**Effect of λ (Regularization Strength):**

- **λ = 0**: No regularization (MLE)
  - Parameters can be any value
  - Risk of overfitting
  
- **λ small (0.01)**: Weak regularization
  - Parameters can be moderately large
  - Slight shrinkage
  
- **λ medium (1.0)**: Moderate regularization
  - Parameters shrunk toward 0
  - Balanced bias-variance tradeoff
  
- **λ large (100)**: Strong regularization
  - Parameters very close to 0
  - Risk of underfitting

**Mathematical Connection:**
```
λ = σ² / σ²_prior

Where:
- σ²: Noise variance in data
- σ²_prior: Prior variance of parameters
- Higher λ: More confidence in prior (smaller σ²_prior)
- Lower λ: Less confidence in prior (larger σ²_prior)
```

## L1 Regularization (Lasso) = Laplace Prior

### Frequentist Formulation

**Loss Function:**
```
L(w) = MSE + λ * ||w||₁
     = (y - Xw)² + λ * Σ |wᵢ|

Minimize: (y - Xw)² + λ||w||₁
```

### Bayesian Interpretation

**Prior Distribution:**
```
w ~ Laplace(0, b)

Where:
- Mean: 0
- Scale: b = 1/λ
- PDF: f(w) = (1/2b) * exp(-|w|/b)
```

**Posterior Distribution:**
```
P(w|data) ∝ P(data|w) * P(w)
          ∝ exp(-MSE/2σ²) * exp(-λ||w||₁)
          ∝ exp(-(MSE + λ||w||₁)/2)
```

**MAP Estimation:**
```
MAP = argmin_w (MSE + λ||w||₁)

Again, exactly the regularized loss!
```

### Detailed Explanation

**What the Laplace Prior Means:**

The Laplace prior assumes that:
1. **Parameters come from Laplace distribution** centered at 0
2. **Most parameters should be exactly 0** (sharp peak at 0)
3. **Few parameters should be non-zero** (fat tails)
4. **Sparse**: Most features irrelevant, few relevant

**Why it Works:**

**1. Feature Selection:**
- Laplace distribution has sharp peak at 0
- High probability mass at exactly 0
- Many parameters set to exactly 0
- Automatically selects important features

**2. Sparse Solutions:**
- Unlike L2, L1 can set parameters to exactly 0
- Useful when you have many irrelevant features
- Reduces model complexity

**3. Handles High Dimensions:**
- When p > n (more features than samples)
- L1 can still work (sparse solutions)
- L2 might not be as effective

**Effect of λ:**

- **λ = 0**: No regularization
  - All features used
  - Risk of overfitting
  
- **λ small**: Weak regularization
  - Few parameters set to 0
  - Most features kept
  
- **λ medium**: Moderate regularization
  - Many parameters set to 0
  - Feature selection active
  
- **λ large**: Strong regularization
  - Most parameters set to 0
  - Very sparse model
  - Risk of underfitting

## Comparison: L1 vs L2 Priors

### Distribution Shapes

**Gaussian (L2):**
```
PDF: f(w) = (1/√(2πσ²)) * exp(-w²/2σ²)

Shape:
     /\\
    /  \\
   /    \\
  /      \\
  
Smooth bell curve, no sharp peak
```

**Laplace (L1):**
```
PDF: f(w) = (1/2b) * exp(-|w|/b)

Shape:
     /|\\
    / | \\
   /  |  \\
  /   |   \\
  
Sharp peak at 0, fat tails
```

### Key Differences

| Aspect | L2 (Gaussian) | L1 (Laplace) |
|--------|---------------|--------------|
| **Distribution** | Normal (bell curve) | Laplace (double exponential) |
| **Peak at 0** | Smooth | Sharp |
| **Tails** | Thin (exponential decay) | Fat (slower decay) |
| **Sparsity** | No (rarely exactly 0) | Yes (many exactly 0) |
| **Shrinkage** | Smooth, continuous | Sharp, discontinuous |
| **Feature Selection** | No | Yes |
| **Use Case** | Prevent overfitting | Feature selection + overfitting |

### Why L1 Creates Sparsity

**Mathematical Reason:**

The L1 penalty |w| is not differentiable at 0. This creates a "corner" in the optimization landscape. When the optimal solution is at this corner, the parameter is exactly 0.

**Geometric Intuition:**

- **L2 constraint**: Circle (||w||² ≤ t)
  - Smooth boundary
  - Optimal solution rarely on boundary
  - Parameters rarely exactly 0

- **L1 constraint**: Diamond (||w||₁ ≤ t)
  - Sharp corners at axes
  - Optimal solution often at corners
  - Parameters often exactly 0

**Visual:**
```
L2 (circle):        L1 (diamond):
     •                 •   •
    • •               •     •
   •   •             •   •   •
    • •               •     •
     •                 •   •

Smooth              Sharp corners
```

## Elastic Net: Combining L1 and L2

**Formulation:**
```
L(w) = MSE + λ₁||w||₁ + λ₂||w||²
```

**Bayesian Interpretation:**
```
Prior: Combination of Laplace and Gaussian
P(w) ∝ exp(-λ₁||w||₁) * exp(-λ₂||w||²)
```

**Why Use Both:**
- **L1**: Feature selection (sparsity)
- **L2**: Stability (prevents correlated features from having very different coefficients)
- **Combined**: Best of both worlds

## Practical Implications

### Choosing Regularization Type

**Use L2 (Ridge) when:**
- All features might be relevant
- Features are correlated
- You want smooth shrinkage
- Interpretability less important

**Use L1 (Lasso) when:**
- Many irrelevant features
- Need feature selection
- Want sparse model
- High-dimensional data (p > n)

**Use Elastic Net when:**
- Want both sparsity and stability
- Features are correlated
- Need feature selection but also stability

### Choosing λ (Regularization Strength)

**Methods:**
1. **Cross-validation**: Try different λ, choose best
2. **Grid search**: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
3. **Bayesian approach**: Treat λ as hyperparameter, use hyperprior

**Interpretation:**
- **Higher λ**: Stronger prior belief that parameters are small
- **Lower λ**: Weaker prior, more trust in data
- **Optimal λ**: Balances bias and variance

## Summary

**Key Insights:**
1. **L2 = Gaussian prior**: Assumes parameters normally distributed around 0
2. **L1 = Laplace prior**: Assumes parameters Laplace distributed (sparse)
3. **Regularization = Prior belief**: λ controls strength of prior
4. **MAP = Regularized MLE**: Maximum a posteriori equals regularized maximum likelihood
5. **Choose based on sparsity need**: L1 for feature selection, L2 for smooth shrinkage

Understanding the Bayesian interpretation helps:
- Choose right regularization type
- Interpret regularization strength
- Understand why regularization works
- Connect frequentist and Bayesian approaches

