# Topic 34: Discriminative vs Generative Models

## What You'll Learn

This topic covers fundamental ML model types:
- Discriminative Models (what they are, how they work)
- Generative Models (what they are, how they work)
- Key differences and when to use each
- Assumptions of different models (Linear Regression, Logistic Regression, SVM)
- Bayes' Theorem (detailed explanation)
- Simple implementations

## Why We Need This

### Interview Importance
- **Common questions**: "Explain discriminative vs generative models"
- **Fundamental knowledge**: Shows deep ML understanding
- **Model selection**: Know which type to use

### Real-World Application
- **Model choice**: Different problems need different model types
- **Understanding**: Know what assumptions models make
- **Debugging**: Understand why models fail

## Detailed Theory

### Discriminative Models

**What are Discriminative Models?**

Discriminative models learn the boundary or decision function that directly separates different classes or predicts the target variable. They model the conditional probability P(Y|X), which means "what is the probability of Y given X?" These models focus on learning the decision boundary between classes rather than modeling the underlying data distribution.

**How they work:**

Discriminative models take input features X and directly learn a function that maps to the output Y. They don't care about how the data was generated - they only care about finding the best way to distinguish between different classes or predict the target value. The model learns patterns in the data that are useful for making predictions, without needing to understand the full probability distribution of the data.

**Key characteristics:**

Discriminative models are typically simpler and more direct. They focus on what matters for prediction: the relationship between inputs and outputs. Because they don't need to model the entire data distribution, they often require less data and can be more efficient. However, they can't generate new data samples because they don't learn the data distribution.

**Examples:**
- **Logistic Regression**: Models P(Y|X) directly using a sigmoid function
- **Support Vector Machines (SVM)**: Find the optimal hyperplane that separates classes
- **Neural Networks**: Learn complex non-linear decision boundaries
- **Decision Trees**: Learn hierarchical decision rules
- **Linear Regression**: Models E[Y|X] (expected value of Y given X)

**When to use:**
- When you only need predictions, not data generation
- When you have limited data (more data-efficient)
- When the decision boundary is what matters
- Classification and regression tasks

### Generative Models

**What are Generative Models?**

Generative models learn the joint probability distribution P(X, Y) of both inputs and outputs. This means they learn how the data is generated - they understand the underlying data distribution. Once trained, generative models can generate new data samples that look like the training data, because they've learned the probability distribution that the data comes from.

**How they work:**

Generative models learn P(X, Y) = P(X|Y) * P(Y). This means they learn:
1. The class prior P(Y): How likely each class is overall
2. The class-conditional distribution P(X|Y): What the data looks like for each class

To make predictions, they use Bayes' theorem to compute P(Y|X) = P(X|Y) * P(Y) / P(X). This is more complex than discriminative models, but it gives them the ability to generate new data.

**Key characteristics:**

Generative models are more powerful in the sense that they can generate new data, but they're also more complex. They need to learn the full data distribution, which requires more data and more sophisticated modeling. However, this extra complexity gives them capabilities that discriminative models don't have: they can generate new samples, handle missing data better, and provide more interpretable results in some cases.

**Examples:**
- **Naive Bayes**: Assumes features are independent given class
- **Gaussian Mixture Models (GMM)**: Models data as mixture of Gaussians
- **Generative Adversarial Networks (GANs)**: Generate new images/data
- **Variational Autoencoders (VAEs)**: Generate and reconstruct data
- **Language Models (GPT)**: Generate new text

**When to use:**
- When you need to generate new data
- When you have missing data (can impute)
- When you need to understand data distribution
- When you have sufficient data
- Anomaly detection (learn normal distribution)

### Key Differences

**Mathematical Difference:**

Discriminative models learn P(Y|X) directly. They find a function f such that Y ≈ f(X). Generative models learn P(X, Y) = P(X|Y) * P(Y), then use Bayes' theorem to compute P(Y|X) = P(X|Y) * P(Y) / P(X).

**Practical Differences:**

Discriminative models are typically simpler, faster to train, and require less data. They focus on what's needed for prediction. Generative models are more complex, can generate data, and handle missing data better, but require more data and computation.

**When to choose:**

Choose **discriminative** when: You only need predictions, have limited data, or need fast training. Choose **generative** when: You need to generate data, have missing data, want to understand data distribution, or have sufficient data.

## Model Assumptions

### Linear Regression Assumptions

**Assumption 1: Linearity**
The relationship between independent variables X and dependent variable Y is linear. This means that a unit change in X results in a constant change in Y, regardless of the current value of X. Mathematically, this means Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε, where the relationship is linear in the parameters β.

**Why this matters:** If the true relationship is non-linear (e.g., Y = X²), linear regression will perform poorly. The model assumes that the effect of each feature is constant across all values of that feature.

**How to check:** Plot residuals vs predicted values. If there's a pattern (curve), the relationship might be non-linear.

**Assumption 2: Independence of Errors**
The errors (residuals) ε are independent of each other. This means that the error for one observation doesn't depend on the error for another observation. In other words, there's no correlation between errors.

**Why this matters:** If errors are correlated (e.g., time series data where today's error depends on yesterday's), the model's standard errors will be wrong, leading to incorrect confidence intervals and hypothesis tests.

**How to check:** Durbin-Watson test for time series, or plot residuals vs time/order.

**Assumption 3: Homoscedasticity (Constant Variance)**
The variance of errors is constant across all values of X. This means that the spread of residuals should be the same whether X is small or large.

**Why this matters:** If variance changes (heteroscedasticity), the model's estimates are still unbiased, but standard errors are wrong. This affects confidence intervals and hypothesis tests.

**How to check:** Plot residuals vs predicted values. Look for a funnel shape (variance increasing/decreasing).

**Assumption 4: Normality of Errors**
The errors ε are normally distributed with mean 0. This means that for any value of X, the errors should follow a normal distribution centered at 0.

**Why this matters:** Needed for hypothesis testing and confidence intervals. However, for large samples, this is less critical due to Central Limit Theorem.

**How to check:** Q-Q plot, histogram of residuals, Shapiro-Wilk test.

**Assumption 5: No Multicollinearity**
The independent variables are not highly correlated with each other. If X₁ and X₂ are highly correlated, it's hard to separate their individual effects.

**Why this matters:** High multicollinearity makes coefficient estimates unstable. Small changes in data can cause large changes in coefficients. It also makes it hard to interpret individual coefficients.

**How to check:** Correlation matrix, Variance Inflation Factor (VIF). VIF > 10 indicates high multicollinearity.

**What happens if assumptions are violated:**
- **Non-linearity**: Poor predictions, use polynomial features or non-linear models
- **Correlated errors**: Wrong standard errors, use time series models
- **Heteroscedasticity**: Wrong standard errors, use weighted least squares
- **Non-normal errors**: Wrong confidence intervals (but OK for large samples)
- **Multicollinearity**: Unstable coefficients, remove correlated features or use regularization

### Logistic Regression Assumptions

**Assumption 1: Binary Outcome**
The dependent variable Y is binary (0 or 1). Logistic regression is designed for binary classification.

**Why this matters:** If you have more than 2 classes, you need multinomial logistic regression or one-vs-rest approach.

**Assumption 2: Linearity of Log-Odds**
The relationship between independent variables and the log-odds of the outcome is linear. This is different from linear regression - here we assume linearity in log-odds space, not in the probability space.

**Mathematical formulation:** log(P(Y=1|X) / P(Y=0|X)) = β₀ + β₁X₁ + ... + βₙXₙ

This means that the log-odds (logit) is linear in X, but the probability itself is non-linear (sigmoid curve).

**Why this matters:** If the relationship between X and log-odds is non-linear, the model will perform poorly. However, the probability curve is always S-shaped (sigmoid), which is appropriate for binary outcomes.

**How to check:** Plot log-odds vs X (if you can compute it), or use Box-Tidwell test.

**Assumption 3: Independence of Observations**
Each observation is independent. Similar to linear regression, errors should be independent.

**Why this matters:** Violations (e.g., repeated measurements, clustered data) lead to wrong standard errors.

**Assumption 4: No Multicollinearity**
Independent variables should not be highly correlated. Same as linear regression.

**Assumption 5: Large Sample Size**
Logistic regression works best with large sample sizes, especially when you have many features or rare events.

**Why this matters:** Maximum likelihood estimation in logistic regression requires sufficient data. With small samples, estimates can be biased.

**What's different from linear regression:**
- **No normality assumption**: Errors don't need to be normal (they're binary)
- **No homoscedasticity**: Variance is determined by the mean (p(1-p))
- **Non-linear probability**: Probability is sigmoid, not linear

### SVM Assumptions

**Assumption 1: Separable or Nearly Separable Data**
For hard-margin SVM, data must be linearly separable. For soft-margin SVM, data should be nearly separable (most points can be separated with a small margin violation).

**Why this matters:** If data is not separable, hard-margin SVM has no solution. Soft-margin SVM handles this with slack variables, but performance degrades if too many points are misclassified.

**Assumption 2: Feature Scaling**
SVM is sensitive to feature scales. Features should be normalized or standardized.

**Why this matters:** SVM tries to maximize the margin. If one feature has much larger values than others, it will dominate the margin calculation. Without scaling, the model might ignore important but small-scale features.

**How to handle:** Always scale features before SVM (StandardScaler, MinMaxScaler).

**Assumption 3: Appropriate Kernel**
The choice of kernel (linear, polynomial, RBF) should match the data structure.

**Why this matters:** 
- **Linear kernel**: Assumes data is linearly separable (or nearly so)
- **RBF kernel**: Assumes data has local structure (similar points are close)
- **Polynomial kernel**: Assumes polynomial relationships

Wrong kernel choice leads to poor performance.

**Assumption 4: Balanced Classes (for classification)**
SVM can be sensitive to class imbalance, especially with certain kernels.

**Why this matters:** With imbalanced classes, SVM might focus on the majority class and ignore the minority class.

**How to handle:** Use class weights, SMOTE, or other balancing techniques.

**What SVM doesn't assume:**
- **No distributional assumptions**: Unlike linear/logistic regression, SVM doesn't assume normal distributions
- **No linearity in original space**: With kernels, can handle non-linear relationships
- **Robust to outliers**: Margin-based approach is less sensitive to outliers than least squares

## Bayes' Theorem

### Detailed Explanation

**What is Bayes' Theorem?**

Bayes' theorem is a fundamental principle in probability theory that describes how to update our beliefs about an event based on new evidence. It provides a mathematical way to combine prior knowledge with observed data to arrive at a posterior (updated) probability.

**Mathematical Formulation:**

```
P(A|B) = P(B|A) * P(A) / P(B)

Where:
- P(A|B): Posterior probability (probability of A given B)
- P(B|A): Likelihood (probability of B given A)
- P(A): Prior probability (probability of A before seeing B)
- P(B): Evidence (probability of B, normalization constant)
```

**Detailed Interpretation:**

**Prior P(A):** This is what we believe about A before we see any evidence. It represents our initial knowledge or assumptions. For example, if we're testing for a disease, the prior is the base rate of the disease in the population.

**Likelihood P(B|A):** This is the probability of observing evidence B if A is true. It tells us how likely we are to see this evidence given that our hypothesis A is correct. For example, if someone has the disease, how likely are they to test positive?

**Evidence P(B):** This is the total probability of observing B, regardless of whether A is true or not. It's a normalization constant that ensures probabilities sum to 1. It can be computed as P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A).

**Posterior P(A|B):** This is our updated belief about A after seeing evidence B. It combines our prior knowledge with the new evidence to give us a revised probability.

**Why Bayes' Theorem Matters:**

Bayes' theorem is the foundation of Bayesian statistics and many machine learning algorithms. It allows us to:
1. **Update beliefs**: Start with prior knowledge, update with evidence
2. **Handle uncertainty**: Quantify uncertainty in a principled way
3. **Combine information**: Integrate multiple sources of information
4. **Make decisions**: Use posterior probabilities for decision-making

**Example: Medical Diagnosis**

Suppose a disease affects 1% of the population (prior: P(disease) = 0.01). A test for the disease is 95% accurate (likelihood: P(positive|disease) = 0.95, P(positive|no disease) = 0.05).

If someone tests positive, what's the probability they have the disease?

**Using Bayes' Theorem:**
```
P(disease|positive) = P(positive|disease) * P(disease) / P(positive)

P(positive) = P(positive|disease) * P(disease) + P(positive|no disease) * P(no disease)
            = 0.95 * 0.01 + 0.05 * 0.99
            = 0.0095 + 0.0495
            = 0.059

P(disease|positive) = 0.95 * 0.01 / 0.059
                    = 0.0095 / 0.059
                    ≈ 0.161 (16.1%)
```

**Key Insight:** Even with a 95% accurate test, if the disease is rare (1%), a positive test only means 16% chance of having the disease! This is because false positives from the large healthy population (99%) outweigh true positives from the small diseased population (1%).

**Use Cases in ML:**
- **Naive Bayes**: Classification using Bayes' theorem
- **Bayesian Neural Networks**: Uncertainty quantification
- **Bayesian Optimization**: Hyperparameter tuning
- **Spam Detection**: Update probability of spam given email features
- **Recommendation Systems**: Update user preferences given behavior

## Industry-Standard Boilerplate Code

See `models_comparison.py` for implementations.

## Exercises

1. Implement Naive Bayes (generative)
2. Compare with Logistic Regression (discriminative)
3. Test model assumptions
4. Apply Bayes' theorem to real problems

## Next Steps

- Use this knowledge to choose right model type
- Understand why models fail (assumption violations)
- Apply Bayes' theorem in practice

