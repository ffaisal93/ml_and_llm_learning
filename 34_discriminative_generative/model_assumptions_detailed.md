# Model Assumptions: Detailed Explanations

## Why Assumptions Matter

Every machine learning model makes assumptions about the data. These assumptions are necessary for the model to work correctly. When assumptions are violated, the model may:
- Give incorrect predictions
- Have wrong confidence intervals
- Fail to generalize
- Produce biased estimates

Understanding assumptions helps you:
- Choose the right model for your data
- Diagnose why models fail
- Fix problems when assumptions are violated
- Interpret results correctly

## Linear Regression Assumptions

### Assumption 1: Linearity

**What it means:**
The relationship between independent variables X and the dependent variable Y is linear. This means that the effect of each feature on the target is constant - a one-unit change in X always results in the same change in Y, regardless of the current value of X.

**Mathematical formulation:**
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

The relationship is linear in the parameters β. This doesn't mean the relationship in the original space must be linear - you can use polynomial features (X², X³) or transformations (log X) to create linear relationships.

**Why this assumption exists:**
Linear regression is based on the method of least squares, which assumes a linear relationship. If the true relationship is non-linear, the model will systematically mispredict - it will consistently overestimate or underestimate depending on the region of the input space.

**What happens if violated:**
- Poor predictions (systematic errors)
- Low R² (model doesn't fit well)
- Residuals show patterns (not random)

**How to check:**
1. Plot residuals vs predicted values - should be random scatter
2. Plot residuals vs each feature - should be random
3. If you see curves or patterns, relationship is non-linear

**How to fix:**
- Add polynomial features (X², X³, interactions)
- Use transformations (log, sqrt)
- Use non-linear models (neural networks, decision trees)

**Example:**
If true relationship is Y = X², linear regression Y = β₀ + β₁X will fail. But Y = β₀ + β₁X + β₂X² will work (polynomial regression).

### Assumption 2: Independence of Errors

**What it means:**
The errors (residuals) ε are independent of each other. The error for one observation doesn't depend on or correlate with the error for another observation. This means there's no systematic relationship between errors.

**Why this assumption exists:**
The statistical theory behind linear regression (standard errors, confidence intervals, hypothesis tests) assumes independent errors. When errors are correlated, the effective sample size is smaller than the actual sample size, leading to incorrect uncertainty estimates.

**What happens if violated:**
- Standard errors are wrong (usually too small)
- Confidence intervals are too narrow
- Hypothesis tests give wrong p-values
- Model appears more certain than it should be

**Common violations:**
- **Time series data**: Today's error depends on yesterday's
- **Repeated measurements**: Same person measured multiple times
- **Clustered data**: Observations from same group are similar
- **Spatial data**: Nearby locations have similar errors

**How to check:**
1. **Durbin-Watson test**: For time series (value should be ~2)
2. **Plot residuals vs time/order**: Look for patterns
3. **Autocorrelation function**: Check for correlation at different lags

**How to fix:**
- **Time series**: Use time series models (ARIMA, LSTM)
- **Repeated measures**: Use mixed-effects models
- **Clustered data**: Use hierarchical models or cluster-robust standard errors
- **Spatial data**: Use spatial regression models

### Assumption 3: Homoscedasticity (Constant Variance)

**What it means:**
The variance of the errors is constant across all values of X. This means that the spread of residuals should be the same whether X is small or large, whether the prediction is high or low.

**Why this assumption exists:**
When variance is constant, all observations contribute equally to the model. When variance changes (heteroscedasticity), some observations are more reliable than others, but the model treats them the same.

**What happens if violated:**
- Coefficient estimates are still unbiased (correct on average)
- But standard errors are wrong
- Confidence intervals are incorrect
- Hypothesis tests are unreliable
- Some predictions are more uncertain than others (but model doesn't know this)

**How to check:**
1. **Plot residuals vs predicted values**: Look for funnel shape
   - Funnel opening right: Variance increases with prediction
   - Funnel opening left: Variance decreases with prediction
2. **Breusch-Pagan test**: Statistical test for heteroscedasticity
3. **White test**: Another test for heteroscedasticity

**How to fix:**
- **Weighted least squares**: Weight observations by inverse variance
- **Transformations**: Log transformation often helps
- **Robust standard errors**: Use heteroscedasticity-robust standard errors
- **Generalized least squares**: Model the variance structure

**Example:**
If predicting income, variance might increase with income level (richer people have more variable incomes). This violates homoscedasticity.

### Assumption 4: Normality of Errors

**What it means:**
The errors ε follow a normal distribution with mean 0. For any value of X, the errors should be normally distributed around 0.

**Why this assumption exists:**
Needed for:
- Hypothesis testing (t-tests, F-tests)
- Confidence intervals
- Prediction intervals

However, for **large samples** (n > 30), this is less critical due to Central Limit Theorem.

**What happens if violated:**
- For **large samples**: Usually OK (CLT applies)
- For **small samples**: Confidence intervals and hypothesis tests may be wrong
- Prediction intervals may be incorrect

**How to check:**
1. **Q-Q plot**: Points should fall on straight line
2. **Histogram of residuals**: Should look bell-shaped
3. **Shapiro-Wilk test**: Statistical test for normality
4. **Kolmogorov-Smirnov test**: Another normality test

**How to fix:**
- **Large samples**: Often not necessary (CLT)
- **Transformations**: Log, Box-Cox transformations
- **Non-parametric methods**: Don't assume normality
- **Robust methods**: Less sensitive to non-normality

**Note:** This is often the least critical assumption, especially with large samples.

### Assumption 5: No Multicollinearity

**What it means:**
The independent variables are not highly correlated with each other. If X₁ and X₂ are highly correlated, it's difficult to separate their individual effects on Y.

**Why this assumption exists:**
When features are highly correlated:
- Coefficients become unstable (small data changes → large coefficient changes)
- Standard errors become large (uncertainty increases)
- Hard to interpret individual coefficients
- Model may overfit

**What happens if violated:**
- Coefficients are still unbiased, but:
- Large standard errors (high uncertainty)
- Coefficients can have wrong signs
- Unstable estimates (small data changes → large coefficient changes)
- Hard to interpret: "What's the effect of X₁?" (Can't separate from X₂)

**How to check:**
1. **Correlation matrix**: Look for high correlations (>0.8)
2. **Variance Inflation Factor (VIF)**: 
   - VIF = 1: No multicollinearity
   - VIF > 5: Moderate multicollinearity
   - VIF > 10: High multicollinearity
3. **Eigenvalues of correlation matrix**: Small eigenvalues indicate multicollinearity

**How to fix:**
- **Remove correlated features**: Keep one, remove others
- **Principal Component Analysis (PCA)**: Create uncorrelated features
- **Regularization (Ridge, Lasso)**: Shrinks coefficients, reduces impact
- **Domain knowledge**: Combine correlated features into one

**Example:**
If you have "height in cm" and "height in inches", they're perfectly correlated (multicollinearity). Remove one.

## Logistic Regression Assumptions

### Assumption 1: Binary Outcome

**What it means:**
The dependent variable Y must be binary (0 or 1). Logistic regression is specifically designed for binary classification.

**Why this assumption exists:**
The logistic function (sigmoid) maps any real number to [0, 1], which is perfect for binary probabilities. For multi-class problems, you need extensions (multinomial logistic regression, one-vs-rest).

**What happens if violated:**
- Model won't work for multi-class directly
- Need to use extensions or different models

**How to fix:**
- **Multinomial logistic regression**: For 3+ classes
- **One-vs-rest**: Train binary classifier for each class
- **Softmax regression**: Generalization to multi-class

### Assumption 2: Linearity of Log-Odds

**What it means:**
The relationship between independent variables and the **log-odds** of the outcome is linear. This is different from linear regression - here linearity is in log-odds space, not probability space.

**Mathematical formulation:**
```
log(P(Y=1|X) / P(Y=0|X)) = β₀ + β₁X₁ + ... + βₙXₙ

This is the logit (log-odds), which is linear in X.
The probability itself is non-linear (sigmoid curve):
P(Y=1|X) = 1 / (1 + exp(-(β₀ + β₁X₁ + ... + βₙXₙ)))
```

**Why this assumption exists:**
Logistic regression models the log-odds as linear. The probability curve is always S-shaped (sigmoid), which is appropriate for binary outcomes, but the log-odds should be linear in the features.

**What happens if violated:**
- Poor predictions
- Low accuracy
- Model doesn't capture true relationship

**How to check:**
- **Box-Tidwell test**: Tests linearity of log-odds
- **Plot log-odds vs features**: Should be linear
- **Residual analysis**: Deviance residuals should be random

**How to fix:**
- **Polynomial features**: Add X², X³ terms
- **Interactions**: Add X₁ * X₂ terms
- **Splines**: Non-linear transformations
- **Non-linear models**: Neural networks, decision trees

### Assumption 3: Independence of Observations

**What it means:**
Each observation is independent. Similar to linear regression, errors should be independent.

**Why this assumption exists:**
The maximum likelihood estimation in logistic regression assumes independent observations. Correlated observations reduce the effective sample size.

**What happens if violated:**
- Standard errors are wrong
- Confidence intervals incorrect
- Hypothesis tests unreliable

**Common violations:**
- Repeated measurements (same person multiple times)
- Clustered data (observations from same group)
- Time series (temporal correlation)

**How to fix:**
- **Mixed-effects models**: Account for clustering
- **Generalized Estimating Equations (GEE)**: Handle correlated data
- **Cluster-robust standard errors**: Adjust for clustering

### Assumption 4: No Multicollinearity

**What it means:**
Independent variables should not be highly correlated. Same as linear regression.

**Impact:**
- Unstable coefficients
- Large standard errors
- Hard to interpret

**How to check and fix:**
Same as linear regression (correlation matrix, VIF, remove features, regularization).

### Assumption 5: Large Sample Size

**What it means:**
Logistic regression works best with large sample sizes, especially when you have many features or rare events (imbalanced classes).

**Why this assumption exists:**
Maximum likelihood estimation requires sufficient data. With small samples:
- Estimates can be biased
- Standard errors unreliable
- Model may not converge

**Rule of thumb:**
- At least 10-20 observations per feature
- For rare events: Need many more observations
- Minimum 100-200 observations total

**What happens if violated:**
- Biased estimates
- Unreliable standard errors
- Model may not converge
- Poor predictions

**How to fix:**
- **Collect more data**: Best solution
- **Reduce features**: Fewer features need less data
- **Regularization**: Helps with small samples
- **Simplify model**: Use fewer parameters

## SVM Assumptions

### Assumption 1: Separable or Nearly Separable Data

**What it means:**
For hard-margin SVM, data must be linearly separable (can draw a line/plane that perfectly separates classes). For soft-margin SVM, data should be nearly separable (most points can be separated with small margin violations).

**Why this assumption exists:**
SVM tries to find the maximum margin separator. If data is not separable, hard-margin SVM has no solution. Soft-margin SVM handles this with slack variables (allows some misclassification), but performance degrades if too many points violate the margin.

**What happens if violated:**
- **Hard-margin**: No solution (algorithm fails)
- **Soft-margin**: Many support vectors, poor generalization
- Low accuracy

**How to check:**
- Visualize data (if 2D)
- Check if classes overlap significantly
- Try linear SVM - if fails, data not separable

**How to fix:**
- **Use soft-margin**: Allow some misclassification (parameter C)
- **Use kernel**: Transform to higher dimension where data is separable
- **Preprocess data**: Remove outliers, balance classes

### Assumption 2: Feature Scaling

**What it means:**
SVM is very sensitive to feature scales. Features should be normalized (mean=0, std=1) or standardized before training.

**Why this assumption exists:**
SVM tries to maximize the margin. The margin is computed using distances, which depend on feature scales. If one feature has much larger values (e.g., age in years vs income in dollars), it will dominate the margin calculation, and the model might ignore important but small-scale features.

**What happens if violated:**
- Model performance degrades significantly
- Some features ignored (those with small scales)
- Margin calculation dominated by large-scale features
- Poor generalization

**How to check:**
- Look at feature ranges: If very different, need scaling
- Check feature means and standard deviations

**How to fix:**
- **StandardScaler**: (x - mean) / std (most common)
- **MinMaxScaler**: (x - min) / (max - min) (scales to [0,1])
- **Always scale before SVM**: This is critical!

**Example:**
If you have age (0-100) and income (0-100000), income will dominate. Scale both to same range.

### Assumption 3: Appropriate Kernel

**What it means:**
The choice of kernel (linear, polynomial, RBF) should match the structure of your data.

**Why this assumption exists:**
Different kernels make different assumptions:
- **Linear kernel**: Assumes data is (or can be) linearly separated
- **RBF kernel**: Assumes data has local structure (similar points are close)
- **Polynomial kernel**: Assumes polynomial relationships

Wrong kernel choice means the model can't capture the true relationship.

**What happens if violated:**
- Poor performance
- Model can't learn the pattern
- Low accuracy

**How to choose:**
- **Linear kernel**: Start here, use if data is linearly separable
- **RBF kernel**: Most common, works for most non-linear problems
- **Polynomial kernel**: When you know relationship is polynomial
- **Try multiple**: Compare performance

**How to check:**
- Try different kernels
- Use cross-validation to compare
- Visualize decision boundary (if 2D)

### Assumption 4: Balanced Classes (for classification)

**What it means:**
SVM can be sensitive to class imbalance, especially with certain kernels.

**Why this assumption exists:**
SVM tries to maximize margin. With imbalanced classes, the margin might be determined by the majority class, and the model might ignore the minority class.

**What happens if violated:**
- Model focuses on majority class
- Poor performance on minority class
- Low recall for minority class

**How to check:**
- Check class distribution
- Look at per-class performance metrics

**How to fix:**
- **Class weights**: Give more weight to minority class
- **SMOTE**: Oversample minority class
- **Undersample majority**: Reduce majority class
- **Cost-sensitive learning**: Penalize misclassifying minority more

## What Models Don't Assume

It's also important to know what models **don't** assume:

**Linear/Logistic Regression don't assume:**
- Features are normally distributed (only errors need to be normal for linear regression)
- Features are independent (only errors need to be independent)
- Linear relationships in original space (can use transformations)

**SVM doesn't assume:**
- Normal distributions
- Linear relationships (with kernels)
- Specific data distribution
- Large sample size (works with small samples)

## Summary

Understanding assumptions helps you:
1. **Choose the right model**: Match model assumptions to your data
2. **Diagnose problems**: When model fails, check assumptions
3. **Fix issues**: Know how to address assumption violations
4. **Interpret results**: Understand what results mean

Always check assumptions before and after modeling!

