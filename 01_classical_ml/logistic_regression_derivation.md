# Logistic Regression: Detailed Derivation with Intuitive Explanations

## Overview

This document provides a complete, intuitive derivation of logistic regression from first principles. We'll explain why we need it, how it works, and derive everything step-by-step.

## Intuitive Understanding

### Why Logistic Regression?

**Problem with Linear Regression for Classification:**
- Linear regression: y = wx + b (can be any value)
- Classification: y should be 0 or 1 (probability)
- Linear regression can give negative values or > 1 (not probabilities!)

**Example:**
- Predict: Will it rain? (yes/no)
- Linear regression: Might give 1.5 or -0.3 (not valid probabilities)
- Need: Values between 0 and 1 (probabilities)

**Solution:**
- Use sigmoid function to squash output to [0, 1]
- This is logistic regression!

### What is Logistic Regression?

**Simple Explanation:**
Logistic regression predicts probabilities (0 to 1) instead of continuous values.

**Formula:**
```
P(y=1|x) = 1 / (1 + e^(-(wx + b)))

Where:
- P(y=1|x): Probability that y=1 given x
- w, b: Parameters to learn
- e: Euler's number (~2.718)
```

**Visual:**
```
Probability
  1.0 |        *---* (sigmoid curve)
      |      *
      |    *
      |  *
  0.5 | *
      |*
  0.0 |*_________________ x
      
      S-shaped curve (sigmoid)
```

**Key Property:**
- Always between 0 and 1
- Smooth, differentiable
- S-shaped (sigmoid)

---

## Step-by-Step Derivation

### Step 1: The Problem

**Given:**
- Data: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- yᵢ ∈ {0, 1} (binary classification)
- Goal: Predict P(y=1|x) for new x

**Why not use linear regression?**
- Linear: ŷ = wx + b (can be < 0 or > 1)
- Need: P(y=1|x) ∈ [0, 1]

### Step 2: The Odds Ratio

**Intuition:**
Instead of probability, think about "odds"

**Odds:**
```
Odds = P / (1 - P)

Example:
- P = 0.8 → Odds = 0.8 / 0.2 = 4 (4:1 in favor)
- P = 0.5 → Odds = 0.5 / 0.5 = 1 (1:1, even)
- P = 0.2 → Odds = 0.2 / 0.8 = 0.25 (1:4 against)
```

**Why odds?**
- Odds can be any positive number (0 to ∞)
- Can model with linear function!

**Log Odds (Logit):**
```
log(Odds) = log(P / (1 - P))

Why log?
- Odds: [0, ∞)
- Log odds: (-∞, ∞) (unbounded!)
- Can use linear function: log(Odds) = wx + b
```

**Visual:**
```
Probability → Odds → Log Odds
    0.1    →  0.11 →  -2.2
    0.5    →  1.0  →   0.0
    0.9    →  9.0  →   2.2
```

### Step 3: The Model

**Start with log odds:**
```
log(P / (1 - P)) = wx + b
```

**Solve for P:**
```
P / (1 - P) = e^(wx + b)

P = (1 - P) × e^(wx + b)

P = e^(wx + b) - P × e^(wx + b)

P + P × e^(wx + b) = e^(wx + b)

P(1 + e^(wx + b)) = e^(wx + b)

P = e^(wx + b) / (1 + e^(wx + b))

P = 1 / (1 + e^(-(wx + b)))
```

**This is the sigmoid function!**

**Notation:**
```
σ(z) = 1 / (1 + e^(-z))  (sigmoid)

So: P(y=1|x) = σ(wx + b)
```

### Step 4: Why Sigmoid?

**Properties of sigmoid:**
1. **Bounded**: Always between 0 and 1
2. **Smooth**: Differentiable everywhere
3. **Symmetric**: σ(-z) = 1 - σ(z)
4. **Monotonic**: Increases with z

**Visual:**
```
σ(z) = 1 / (1 + e^(-z))

z → -∞: σ(z) → 0
z = 0:  σ(z) = 0.5
z → +∞: σ(z) → 1
```

**Why this shape?**
- When wx + b is very negative → P ≈ 0 (very unlikely)
- When wx + b = 0 → P = 0.5 (even chance)
- When wx + b is very positive → P ≈ 1 (very likely)

### Step 5: The Likelihood Function

**For binary classification:**
- If y = 1: Want P(y=1|x) to be high
- If y = 0: Want P(y=0|x) = 1 - P(y=1|x) to be high

**Combined:**
```
P(y|x) = P(y=1|x)^y × (1 - P(y=1|x))^(1-y)

Why?
- If y = 1: P(y=1|x)^1 × (1 - P(y=1|x))^0 = P(y=1|x) ✓
- If y = 0: P(y=1|x)^0 × (1 - P(y=1|x))^1 = 1 - P(y=1|x) ✓
```

**For all data points (assuming independence):**
```
L(w, b) = ∏ᵢ P(yᵢ|xᵢ)
        = ∏ᵢ [P(y=1|xᵢ)^yᵢ × (1 - P(y=1|xᵢ))^(1-yᵢ)]
```

**This is the likelihood function!**

**Intuition:**
- Likelihood = probability of observing data given parameters
- Higher likelihood = better fit
- Want to maximize likelihood

### Step 6: Log-Likelihood

**Why log?**
- Products become sums (easier!)
- More numerically stable
- Same maximum (log is monotonic)

**Log-likelihood:**
```
log L(w, b) = log ∏ᵢ [P(y=1|xᵢ)^yᵢ × (1 - P(y=1|xᵢ))^(1-yᵢ)]

           = Σᵢ log [P(y=1|xᵢ)^yᵢ × (1 - P(y=1|xᵢ))^(1-yᵢ)]

           = Σᵢ [yᵢ log P(y=1|xᵢ) + (1-yᵢ) log(1 - P(y=1|xᵢ))]
```

**Substitute P(y=1|xᵢ) = σ(wxᵢ + b):**
```
log L(w, b) = Σᵢ [yᵢ log σ(wxᵢ + b) + (1-yᵢ) log(1 - σ(wxᵢ + b))]
```

**This is what we want to maximize!**

### Step 7: Cost Function (Negative Log-Likelihood)

**Convention:**
- Maximize likelihood = Minimize negative log-likelihood
- This is the cost function

**Cost function:**
```
J(w, b) = -log L(w, b)
        = -Σᵢ [yᵢ log σ(wxᵢ + b) + (1-yᵢ) log(1 - σ(wxᵢ + b))]
```

**This is cross-entropy loss!**

**Intuition:**
- If y = 1: Want σ(wx + b) close to 1
  - Cost = -log(σ(wx + b))
  - If σ(wx + b) = 1 → cost = 0 ✓
  - If σ(wx + b) = 0.1 → cost = -log(0.1) = 2.3 (high cost!)
  
- If y = 0: Want σ(wx + b) close to 0
  - Cost = -log(1 - σ(wx + b))
  - If σ(wx + b) = 0 → cost = 0 ✓
  - If σ(wx + b) = 0.9 → cost = -log(0.1) = 2.3 (high cost!)

**Visual:**
```
Cost
  |
  |\     (y=0)
  | \   /
  |  \ /
  |   *
  |  / \
  | /   \  (y=1)
  |/     \
  +------- Probability
  0       1
```

### Step 8: Gradient Descent

**Goal:** Minimize J(w, b)

**Method:** Gradient descent

**Gradient with respect to w:**
```
∂J/∂w = ∂/∂w [-Σᵢ [yᵢ log σ(wxᵢ + b) + (1-yᵢ) log(1 - σ(wxᵢ + b))]]

      = -Σᵢ [yᵢ × (1/σ(wxᵢ + b)) × ∂σ/∂w + (1-yᵢ) × (1/(1-σ(wxᵢ + b))) × ∂(1-σ)/∂w]
```

**Key derivative:**
```
∂σ(z)/∂z = σ(z)(1 - σ(z))

Why?
σ(z) = 1 / (1 + e^(-z))

∂σ/∂z = e^(-z) / (1 + e^(-z))²
      = [1 / (1 + e^(-z))] × [e^(-z) / (1 + e^(-z))]
      = σ(z) × [1 - σ(z)]
```

**Continue gradient:**
```
∂J/∂w = -Σᵢ [yᵢ × (1/σ(wxᵢ + b)) × σ(wxᵢ + b)(1 - σ(wxᵢ + b)) × xᵢ
        + (1-yᵢ) × (1/(1-σ(wxᵢ + b))) × (-σ(wxᵢ + b)(1 - σ(wxᵢ + b))) × xᵢ]

      = -Σᵢ [yᵢ × (1 - σ(wxᵢ + b)) × xᵢ - (1-yᵢ) × σ(wxᵢ + b) × xᵢ]

      = -Σᵢ [yᵢxᵢ - yᵢσ(wxᵢ + b)xᵢ - (1-yᵢ)σ(wxᵢ + b)xᵢ]

      = -Σᵢ [yᵢxᵢ - σ(wxᵢ + b)xᵢ]

      = Σᵢ [σ(wxᵢ + b) - yᵢ] × xᵢ
```

**Similarly for b:**
```
∂J/∂b = Σᵢ [σ(wxᵢ + b) - yᵢ]
```

**Update rules:**
```
w = w - α × ∂J/∂w
  = w - α × Σᵢ [σ(wxᵢ + b) - yᵢ] × xᵢ

b = b - α × ∂J/∂b
  = b - α × Σᵢ [σ(wxᵢ + b) - yᵢ]
```

**Intuition:**
- Error = σ(wx + b) - y (predicted - actual)
- If error > 0: Predicted too high → decrease w, b
- If error < 0: Predicted too low → increase w, b
- Update proportional to error and input x

---

## Matrix Formulation

### Setup

**Multiple features:**
```
X = [1  x₁₁  x₁₂  ...  x₁ₚ]    (design matrix with bias column)
    [1  x₂₁  x₂₂  ...  x₂ₚ]
    [...]
    [1  xₙ₁  xₙ₂  ...  xₙₚ]

w = [w₀]    (weights including bias)
    [w₁]
    [...]
    [wₚ]

y = [y₁]    (targets)
    [y₂]
    [...]
    [yₙ]
```

**Model:**
```
P(y=1|X) = σ(Xw)
```

**Cost:**
```
J(w) = -Σᵢ [yᵢ log σ(Xᵢw) + (1-yᵢ) log(1 - σ(Xᵢw))]
```

**Gradient:**
```
∂J/∂w = Xᵀ(σ(Xw) - y)
```

**Update:**
```
w = w - α × Xᵀ(σ(Xw) - y)
```

---

## Why Cross-Entropy Loss?

### Information Theory Perspective

**Cross-entropy:**
- Measures difference between two distributions
- True distribution: [y, 1-y] (one-hot)
- Predicted distribution: [σ(wx+b), 1-σ(wx+b)]

**Minimizing cross-entropy:**
- Makes predicted distribution close to true
- Equivalent to maximum likelihood
- Information-theoretically optimal

### Comparison to Other Losses

**Mean Squared Error (MSE):**
```
MSE = (1/n) Σ (y - σ(wx + b))²
```
**Problem:**
- Not convex for classification
- Can get stuck in local minima
- Doesn't work well for probabilities

**Cross-Entropy:**
```
CE = -Σ [y log σ(wx + b) + (1-y) log(1 - σ(wx + b))]
```
**Advantages:**
- Convex (guaranteed global minimum)
- Works well with probabilities
- Information-theoretically motivated

---

## Decision Boundary

### How Classification Works

**After training, we have:**
```
P(y=1|x) = σ(wx + b)
```

**Decision rule:**
```
If P(y=1|x) > 0.5: Predict y = 1
If P(y=1|x) < 0.5: Predict y = 0
```

**What does P(y=1|x) = 0.5 mean?**
```
0.5 = 1 / (1 + e^(-(wx + b)))

1 + e^(-(wx + b)) = 2

e^(-(wx + b)) = 1

-(wx + b) = 0

wx + b = 0
```

**Decision boundary:**
- Line: wx + b = 0
- Points on line: P = 0.5 (uncertain)
- Points above line: P > 0.5 (predict class 1)
- Points below line: P < 0.5 (predict class 0)

**Visual:**
```
Class 1: *  *  *
         |
         |  (decision boundary: wx + b = 0)
         |
Class 0: *  *  *
```

---

## Assumptions

### Key Assumptions

**1. Linearity in log odds:**
- log(P/(1-P)) = wx + b is linear
- **Breaks when:** Non-linear relationships (use polynomial features)

**2. Independence:**
- Observations are independent
- **Breaks when:** Time series, repeated measurements

**3. No multicollinearity:**
- Features not highly correlated
- **Breaks when:** Redundant features

**4. Large sample size:**
- Need sufficient data for stable estimates
- **Breaks when:** Small datasets

### What Happens When Assumptions Break?

**Non-linearity:**
- Can't capture curved decision boundaries
- Solution: Polynomial features, feature engineering

**Multicollinearity:**
- Unstable coefficients
- Solution: Regularization, feature selection

---

## Regularization

### Why Regularize?

**Problem:**
- Overfitting with many features
- Unstable coefficients
- Poor generalization

**Solution:**
- Add penalty to cost function
- Shrink coefficients toward zero

### L2 Regularization (Ridge)

**Cost function:**
```
J(w) = -log L(w) + λ||w||²
```

**Gradient:**
```
∂J/∂w = Xᵀ(σ(Xw) - y) + 2λw
```

**Effect:**
- Shrinks all coefficients
- Prevents overfitting
- More stable

### L1 Regularization (Lasso)

**Cost function:**
```
J(w) = -log L(w) + λ||w||₁
```

**Effect:**
- Sets some coefficients to exactly 0
- Feature selection
- Sparse model

---

## Summary

**Key Insights:**

1. **Problem**: Need probabilities (0 to 1), not continuous values
2. **Solution**: Use sigmoid to map linear function to [0, 1]
3. **Derivation**: Start with log odds, solve for probability
4. **Cost**: Cross-entropy loss (negative log-likelihood)
5. **Optimization**: Gradient descent (no closed-form solution)

**Formulas:**
```
Model: P(y=1|x) = σ(wx + b) = 1 / (1 + e^(-(wx + b)))

Cost: J(w, b) = -Σ [y log σ(wx + b) + (1-y) log(1 - σ(wx + b))]

Gradient: ∂J/∂w = Σ [σ(wx + b) - y] × x
          ∂J/∂b = Σ [σ(wx + b) - y]

Update: w = w - α × gradient
```

**Why it works:**
- Sigmoid ensures probabilities in [0, 1]
- Cross-entropy is optimal for classification
- Gradient descent finds optimal parameters
- Decision boundary is linear in feature space

**Intuition:**
- Log odds are linear → probabilities are sigmoid
- Maximize likelihood → minimize cross-entropy
- Gradient points to steepest increase → move opposite direction

