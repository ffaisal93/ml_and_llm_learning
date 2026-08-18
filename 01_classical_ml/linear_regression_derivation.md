# Linear Regression: Detailed Derivation with Intuitive Explanations

## Overview

This document provides a complete, intuitive derivation of linear regression from first principles. We'll build understanding step-by-step, explaining not just the math, but why it works.

## Intuitive Understanding

### What is Linear Regression?

**Simple Explanation:**
Linear regression finds the best straight line that fits your data points.

**Example:**
- You have data: (house size, house price)
- You want to predict: Given a new house size, what's the price?
- Linear regression finds: price = a × size + b (a line!)

**Visual:**
```
Price ($)
  |
  |     * (data point)
  |  *
  | *
  |*_________________ (fitted line)
  |
  +------------------- Size (sq ft)
```

### Why "Linear"?

**Linear means:**
- Relationship is a straight line (not curved)
- Formula: y = mx + b (or y = w₁x + w₀)
- One unit change in x → constant change in y

**Example:**
- If price = 100 × size + 50000
- Increasing size by 1 sq ft → price increases by \$100 (constant!)

> **Saying it out loud.** Linear regression fits the best straight line through your data — and 'best' has a specific meaning we have to pin down. If you're predicting house price from square footage, the model is just price equals some slope times size plus an intercept, and fitting means choosing those two numbers. The word linear means the effect is constant: every extra square foot adds the same amount, whether you're going from 800 to 801 or 3000 to 3001. That constant-effect assumption is the model's whole strength and its whole limitation.

---

## Mathematical Setup

### Problem Statement

**Given:**
- Data points: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- Model: y = w₁x + w₀ (or y = wx + b)
- Goal: Find w₁ and w₀ that best fit the data

**What does "best fit" mean?**
- Line should be close to all data points
- Minimize the distance between line and points
- This distance is called "error" or "residual"

### Visual Intuition

**Error for one point:**
```
     * (data point)
     |
     | error = distance from point to line
     |
  ---*--- (fitted line)
```

**Total error:**
- Sum of all errors (squared, to avoid cancellation)
- Want to minimize this total error

> **Saying it out loud.** Best fit means the line that's closest to all the points at once, and we measure closeness with the vertical gap between each point and the line — the residual. Notice it's the *vertical* distance, not the perpendicular one, because we're predicting $y$ from $x$ and only errors in $y$ count. Then we add those errors up into one number to minimize. The whole derivation is just deciding how to combine them and then doing calculus.

---

## Step-by-Step Derivation

### Step 1: Define the Model

**Linear Model:**
```
ŷᵢ = w₁xᵢ + w₀

Where:
- ŷᵢ: Predicted value for point i
- xᵢ: Input value for point i
- w₁: Slope (weight)
- w₀: Intercept (bias)
```

**Why this form?**
- Simple: Just two parameters
- Interpretable: w₁ = how much y changes per unit x
- Flexible: Can fit any linear relationship

### Step 2: Define the Error (Residual)

**For each data point:**
```
Error for point i = yᵢ - ŷᵢ
                  = yᵢ - (w₁xᵢ + w₀)
```

**Why yᵢ - ŷᵢ?**
- yᵢ = actual value (what we observed)
- ŷᵢ = predicted value (what model says)
- Error = difference between actual and predicted

**Visual:**
```
Actual point: (xᵢ, yᵢ) = (5, 10)
Predicted: ŷᵢ = 2×5 + 1 = 11
Error: 10 - 11 = -1 (predicted too high by 1)
```

### Step 3: Define the Cost Function

**Problem:** We have errors for each point. How do we combine them?

**Option 1: Sum of errors**
```
Σ (yᵢ - ŷᵢ)
```
**Problem:** Positive and negative errors cancel out!
- Point 1: error = +5
- Point 2: error = -5
- Sum = 0 (looks perfect, but it's not!)

**Option 2: Sum of absolute errors**
```
Σ |yᵢ - ŷᵢ|
```
**Problem:** Not differentiable everywhere (hard to optimize)

**Option 3: Sum of squared errors (MSE)**
```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
    = (1/n) Σ (yᵢ - w₁xᵢ - w₀)²
```

**Why squared?**
1. **Always positive**: No cancellation
2. **Penalizes large errors**: Squaring makes big errors much worse
3. **Differentiable**: Smooth function, easy to optimize
4. **Mathematically nice**: Leads to closed-form solution

**Visual Intuition:**
```
Small error: 1² = 1
Large error: 5² = 25 (much worse!)

This encourages model to avoid large errors.
```

> **Saying it out loud.** We square the errors, and the reasons are worth listing because interviewers ask. Plain summing is useless since a plus-five and a minus-five cancel and a terrible line scores zero. Absolute value fixes that but has a kink at zero, so it isn't differentiable and there's no clean formula. Squaring is smooth everywhere, always positive, and it makes big errors hurt disproportionately — one error of 5 costs as much as twenty-five errors of 1. That last property is a real modeling choice, and it's why a single outlier can drag the whole line.

### Step 4: The Optimization Problem

**Goal:** Find w₁ and w₀ that minimize MSE

**Mathematically:**
```
minimize: MSE(w₁, w₀) = (1/n) Σ (yᵢ - w₁xᵢ - w₀)²
```

**How to solve?**
- Take derivatives with respect to w₁ and w₀
- Set derivatives to zero
- Solve for w₁ and w₀

**Why derivatives?**
- Derivative = slope of function
- At minimum, slope = 0
- So we find where derivative = 0

### Step 5: Take Derivatives

*In plain terms:* we're looking for the bottom of a bowl. The cost is a smooth function of the two parameters, and at the lowest point the surface is flat in every direction — so we differentiate with respect to each parameter, set both derivatives to zero, and solve. The algebra below is long but it's only ever the chain rule plus collecting terms.

**Cost function:**
```
MSE = (1/n) Σ (yᵢ - w₁xᵢ - w₀)²
```

**Derivative with respect to w₀ (intercept):**

**Step-by-step:**
```
∂MSE/∂w₀ = ∂/∂w₀ [(1/n) Σ (yᵢ - w₁xᵢ - w₀)²]

         = (1/n) Σ ∂/∂w₀ (yᵢ - w₁xᵢ - w₀)²

         = (1/n) Σ 2(yᵢ - w₁xᵢ - w₀) × (-1)

         = -(2/n) Σ (yᵢ - w₁xᵢ - w₀)

         = -(2/n) [Σ yᵢ - w₁Σ xᵢ - nw₀]
```

**Set to zero:**
```
-(2/n) [Σ yᵢ - w₁Σ xᵢ - nw₀] = 0

Σ yᵢ - w₁Σ xᵢ - nw₀ = 0

nw₀ = Σ yᵢ - w₁Σ xᵢ

w₀ = (1/n)Σ yᵢ - w₁(1/n)Σ xᵢ

w₀ = ȳ - w₁x̄
```

**Where:**
- ȳ = (1/n)Σ yᵢ (mean of y)
- x̄ = (1/n)Σ xᵢ (mean of x)

**Intuition:**
- Intercept = mean of y - slope × mean of x
- Line passes through point (x̄, ȳ)!
- This makes sense: line should go through center of data

**Derivative with respect to w₁ (slope):**

**Step-by-step:**
```
∂MSE/∂w₁ = ∂/∂w₁ [(1/n) Σ (yᵢ - w₁xᵢ - w₀)²]

         = (1/n) Σ ∂/∂w₁ (yᵢ - w₁xᵢ - w₀)²

         = (1/n) Σ 2(yᵢ - w₁xᵢ - w₀) × (-xᵢ)

         = -(2/n) Σ xᵢ(yᵢ - w₁xᵢ - w₀)

         = -(2/n) [Σ xᵢyᵢ - w₁Σ xᵢ² - w₀Σ xᵢ]
```

**Set to zero:**
```
-(2/n) [Σ xᵢyᵢ - w₁Σ xᵢ² - w₀Σ xᵢ] = 0

Σ xᵢyᵢ - w₁Σ xᵢ² - w₀Σ xᵢ = 0
```

**Substitute w₀ = ȳ - w₁x̄:**
```
Σ xᵢyᵢ - w₁Σ xᵢ² - (ȳ - w₁x̄)Σ xᵢ = 0

Σ xᵢyᵢ - w₁Σ xᵢ² - ȳΣ xᵢ + w₁x̄Σ xᵢ = 0

Σ xᵢyᵢ - ȳΣ xᵢ = w₁(Σ xᵢ² - x̄Σ xᵢ)

w₁ = (Σ xᵢyᵢ - ȳΣ xᵢ) / (Σ xᵢ² - x̄Σ xᵢ)
```

**Simplify:**
Note that:
- Σ xᵢ = nx̄
- Σ xᵢ² - x̄Σ xᵢ = Σ xᵢ² - nx̄² = Σ(xᵢ - x̄)²
- Σ xᵢyᵢ - ȳΣ xᵢ = Σ xᵢyᵢ - nȳx̄ = Σ(xᵢ - x̄)(yᵢ - ȳ)

**Final formula:**
```
w₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
```

**Intuition:**
- Numerator: Covariance (how x and y vary together)
- Denominator: Variance of x (how x varies)
- Slope = covariance / variance
- If x and y vary together → positive slope
- If x increases, y increases → positive slope

> **Saying it out loud.** Setting the derivative with respect to the intercept to zero gives something you can picture: the fitted line always passes through the average point, the mean of $x$ and the mean of $y$. And setting the slope's derivative to zero gives slope equals covariance over variance — how much $x$ and $y$ move together, divided by how much $x$ moves on its own. So the whole fit is: anchor the line at the center of the data, then tilt it by however much the two variables co-vary. If $x$ barely varies, that denominator is tiny and the slope estimate is wildly unstable, which is the seed of every multicollinearity problem later.

### Step 6: Final Solution

**Optimal parameters:**
```
w₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

w₀ = ȳ - w₁x̄
```

**This is the closed-form solution!**

**Why it works:**
- Minimizes sum of squared errors
- Unique solution (if data not all same x)
- Best possible line (in least squares sense)

> **Saying it out loud.** The closed-form answer is two lines: slope is covariance over variance, intercept is the mean of $y$ minus slope times the mean of $x$. What's remarkable is that there's no iteration and no learning rate — the minimum has an exact formula, because the cost is a quadratic bowl with exactly one bottom. That's a luxury almost no other model has. The one degenerate case to name is all your $x$ values being identical, which makes the denominator zero — there's no unique line through a vertical stack of points.

---

## Matrix Formulation (Multiple Features)

### Setup

**Multiple features:**
- Input: x = [x₁, x₂, ..., xₚ] (p features)
- Model: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₚxₚ

**Matrix notation:**
```
X = [1  x₁₁  x₁₂  ...  x₁ₚ]    (design matrix)
    [1  x₂₁  x₂₂  ...  x₂ₚ]
    [...]
    [1  xₙ₁  xₙ₂  ...  xₙₚ]

w = [w₀]    (weights)
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
ŷ = Xw
```

**Cost function:**
```
MSE = (1/n) ||y - Xw||²
    = (1/n) (y - Xw)ᵀ(y - Xw)
```

### Derivation

*In plain terms:* this is the same 'differentiate and set to zero' move as before, just written for many features at once. The vector $w$ holds all the weights, the matrix $X$ holds all the data, and the cost is the squared length of the residual vector. Everything below is one matrix derivative and one rearrangement.

**Expand:**
```
MSE = (1/n) (yᵀy - 2yᵀXw + wᵀXᵀXw)
```

**Take derivative:**
```
∂MSE/∂w = (1/n) (-2Xᵀy + 2XᵀXw)
        = (2/n) (XᵀXw - Xᵀy)
```

**Set to zero:**
```
(2/n) (XᵀXw - Xᵀy) = 0

XᵀXw = Xᵀy

w = (XᵀX)⁻¹Xᵀy
```

**This is the normal equation!**

**Intuition:**
- XᵀX: Correlation matrix (how features relate)
- Xᵀy: Correlation between features and target
- (XᵀX)⁻¹: Invert to solve for weights
- Result: Optimal weights that minimize error

> **Saying it out loud.** With many features the same derivation collapses into one equation: $X^\top X w = X^\top y$, the normal equations, so $w$ is $(X^\top X)^{-1} X^\top y$. Read it as: $X^\top X$ captures how the features relate to each other, $X^\top y$ captures how each feature relates to the target, and inverting the first undoes the double-counting caused by correlated features. The failure mode is right there in the formula — if two features are perfectly correlated, that matrix isn't invertible and there's no unique answer. That's why ridge regression, which adds a small amount to the diagonal, fixes collinearity so neatly.

---

## Why This Works: Geometric Intuition

### Projection Interpretation

*In plain terms:* think of the vector of all your observed $y$ values as a single point in $n$-dimensional space. Your model can only reach points that are combinations of the feature columns — a flat subspace inside that space. Least squares picks the point in that subspace closest to $y$, which is exactly the perpendicular projection.

**Linear regression = projection onto column space of X**

**Visual:**
```
y (actual vector)
  |
  |  /
  | /  error = y - ŷ
  |/
  *---ŷ (projection onto column space)
  |
  Column space of X
```

**What this means:**
- Find point in column space closest to y
- This point is ŷ = Xw
- Error is minimized (perpendicular distance)

### Why Least Squares?

**Geometric reason:**
- Perpendicular distance is shortest
- This is exactly what squared error measures
- Pythagorean theorem: distance² = sum of squared differences

> **Saying it out loud.** There's a geometric picture that makes least squares feel inevitable. Put all $n$ of your observed $y$ values into a single vector — one point in $n$-dimensional space. Your model can only produce combinations of the feature columns, which form a flat subspace, and it almost certainly doesn't contain your $y$. So you pick the closest point in that subspace, which is the perpendicular projection, and 'perpendicular' is precisely what setting the derivative to zero enforces: the residual ends up orthogonal to every feature. That's also why the residuals carry no remaining linear information about your features — you've squeezed all of it out.

---

## Assumptions and When They Break

### Assumptions

**1. Linearity:**
- Relationship is linear
- **Breaks when:** Curved relationships (use polynomial regression)

**2. Independence:**
- Errors are independent
- **Breaks when:** Time series, repeated measurements

**3. Homoscedasticity:**
- Constant variance of errors
- **Breaks when:** Variance changes with x (use weighted regression)

**4. Normality:**
- Errors are normally distributed
- **Breaks when:** Outliers, skewed errors

**5. No multicollinearity:**
- Features not highly correlated
- **Breaks when:** Redundant features (use regularization)

### What Happens When Assumptions Break?

**Non-linearity:**
- Model can't capture relationship
- Solution: Polynomial features, transformations

**Heteroscedasticity:**
- Predictions less reliable in some regions
- Solution: Weighted least squares

**Outliers:**
- One bad point can skew entire line
- Solution: Robust regression, remove outliers

> **Saying it out loud.** Five assumptions, and only the first one is about whether the predictions are right. Linearity is the real modeling assumption — break it and the model is simply wrong. The other four — independent errors, constant error variance, normal errors, no collinearity — mostly affect whether your *standard errors and p-values* mean anything, not whether the fit is any good. That distinction is a great thing to say out loud, because most people list all five as if they were equally fatal. The one that bites hardest in practice is independence, which repeated measurements on the same user quietly violate.

---

## Gradient Descent Alternative

### Why Gradient Descent?

**Normal equation problems:**
- (XᵀX)⁻¹ expensive to compute (O(p³))
- Doesn't work if XᵀX not invertible
  - Happens with perfect multicollinearity (one feature is linear combination of others)
  - Happens with duplicate/redundant features or dummy variable trap (all one-hot columns + intercept)
  - Happens when features > samples (p > n), so full column rank is impossible
  - Practical fixes: drop collinear columns, use pseudo-inverse `X⁺` (`np.linalg.pinv`), or use Ridge regression
- Doesn't scale to very large datasets

**Gradient descent:**
- Iterative method
- Works for any dataset size
- Can handle non-invertible cases

### Algorithm

**Initialize:** w = random or zeros

**Repeat until convergence:**
```
w = w - α × ∂MSE/∂w

Where:
- α: Learning rate
- ∂MSE/∂w: Gradient (direction of steepest increase)
- We move opposite to gradient (steepest decrease)
```

**Gradient:**
```
∂MSE/∂w = (2/n) Xᵀ(Xw - y)
```

**Update rule:**
```
w = w - α × (2/n) Xᵀ(Xw - y)
```

**Intuition:**
- Compute gradient (slope)
- Move in direction that decreases error
- Small steps (learning rate)
- Eventually reach minimum

**Visual:**
```
Error
  |
  |    /\
  |   /  \
  |  /    \
  | /      \
  |/        \
  +---------- w
  Start    Minimum
```

> **Saying it out loud.** There's an exact formula for the answer, so why iterate? Because the formula requires inverting a $p \times p$ matrix, which costs cubic time in the number of features and needs the matrix to be invertible in the first place — and with more features than samples, or duplicate columns, it isn't. Gradient descent sidesteps both: each step is just a matrix-vector product, it streams over data in mini-batches, and it degrades gracefully when the problem is degenerate. The rule of thumb is normal equations up to a few thousand features, gradient descent beyond that — and gradient descent is the only option once you're in deep-learning territory.

---

## Summary

**Key Insights:**

1. **Goal**: Find line that minimizes sum of squared errors
2. **Solution**: Closed-form (normal equation) or gradient descent
3. **Intuition**: Line passes through center, slope = covariance/variance
4. **Geometric**: Projection onto column space
5. **Assumptions**: Linearity, independence, constant variance, normality

**Formulas:**
```
Simple: w₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
        w₀ = ȳ - w₁x̄

Matrix: w = (XᵀX)⁻¹Xᵀy
```

**Why it works:**
- Minimizes squared error (geometrically optimal)
- Unique solution (if data not degenerate)
- Interpretable (slope, intercept have meaning)

