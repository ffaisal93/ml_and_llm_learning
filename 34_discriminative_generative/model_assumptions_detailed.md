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

## Naive Bayes Assumptions

Naive Bayes is a family of classifiers. The distribution assumption depends on the variant:

- **Gaussian Naive Bayes:** continuous features follow a Gaussian distribution within each class.
- **Multinomial Naive Bayes:** features are non-negative counts, such as word counts.
- **Bernoulli Naive Bayes:** features are binary events, such as whether a word is present.

### Assumption 1: Conditional Independence

**What it means:**
Features are independent after the class is known:

$$
P(x_1,\ldots,x_d\mid y)=\prod_{j=1}^{d}P(x_j\mid y).
$$

For example, a spam model acts as if the words "free" and "offer" are independent after it knows
whether the email is spam.

**Why this assumption exists:**
The full joint distribution needs too much data to estimate. The independence assumption changes one
large probability into several small probabilities that are easy to estimate.

**What happens if violated:**
- Predictions can still be good.
- Probability estimates can be too confident.
- Repeated or strongly correlated features can be counted more than once.

**How to check:**
- Measure feature correlation within each class.
- Compare probability calibration with accuracy.
- Remove one of two nearly duplicate features and compare validation results.

**How to fix:**
- Remove or combine duplicate features.
- Use feature selection.
- Calibrate probabilities.
- Use logistic regression or a tree model when feature interactions are important.

### Assumption 2: The Likelihood Matches the Feature Type

**What it means:**
The selected Naive Bayes variant must match the data. Gaussian Naive Bayes expects continuous values.
Multinomial Naive Bayes expects counts or non-negative count-like values. Bernoulli Naive Bayes expects
binary features.

**What happens if violated:**
- Estimated likelihoods do not describe the data.
- Class probabilities become misleading.
- Prediction quality can fall.

**How to check and fix:**
- Plot continuous features by class before using Gaussian Naive Bayes.
- Use Multinomial Naive Bayes for token counts or TF values that are non-negative.
- Convert presence/absence data to 0 and 1 for Bernoulli Naive Bayes.
- Choose another likelihood when the data has a different form.

### Assumption 3: Training Data Represents Class Priors

**What it means:**
The class frequencies in training should represent deployment, unless priors are set separately.

**What happens if violated:**
The model starts with the wrong belief about how common each class is. This changes posterior
probabilities and can move the decision boundary.

**How to fix:**
- Set class priors from deployment data.
- Re-estimate priors when the class mix changes.
- Evaluate calibration as well as accuracy.

### Assumption 4: Unseen Events Need Smoothing

**What it means:**
An event that never appears in training gets probability zero in a raw count model. One zero factor
makes the full class likelihood zero.

**How to fix:**
Use Laplace or additive smoothing. Tune the smoothing value with validation data.

**Interview summary:**
Naive Bayes assumes conditional feature independence and a likelihood that matches the feature type.
It is fast and works well with sparse, high-dimensional data. Correlated features mainly hurt probability
calibration because the model can count the same evidence more than once.

## k-Nearest Neighbors (k-NN) Assumptions

### Assumption 1: Nearby Points Have Similar Targets

**What it means:**
k-NN predicts from nearby training examples. It assumes that points close under the selected distance
metric usually have the same class or a similar target value.

**Why this assumption exists:**
k-NN does not learn an explicit formula. Its full prediction rule is local similarity.

**What happens if violated:**
- Neighbors do not provide useful evidence.
- The decision boundary becomes noisy.
- Accuracy can be close to random even with many examples.

**How to check and fix:**
- Compare labels among each point's nearest neighbors.
- Select a distance metric that matches the data.
- Learn an embedding before k-NN when raw features do not express similarity.

### Assumption 2: Feature Scales Are Comparable

**What it means:**
Distance-based models give more influence to features with larger numeric ranges.

**Example:**
An income feature measured in dollars can dominate an age feature measured in years, even when age is
more useful.

**How to fix:**
- Standardize numeric features.
- Normalize vectors when cosine similarity is appropriate.
- Fit the scaler on training data only.

### Assumption 3: Irrelevant Dimensions Are Limited

**What it means:**
In high dimensions, distances often become less informative. Many irrelevant features can make the
nearest point almost as far away as other points. This is the curse of dimensionality.

**What happens if violated:**
- k-NN needs much more data.
- Predictions become unstable.
- Search becomes slow and memory use stays high.

**How to fix:**
- Remove irrelevant features.
- Use PCA or another dimensionality-reduction method.
- Use a learned representation.
- Validate a cosine or task-specific distance metric.

### Assumption 4: Local Density Is Sufficient

**What it means:**
The training set must contain enough examples near a new point. k-NN cannot safely extrapolate far
beyond observed data.

**How to check and fix:**
- Inspect distances to the nearest neighbors.
- Flag predictions when all neighbors are far away.
- Collect more data in sparse regions.
- Use distance-weighted voting so closer neighbors have more influence.

### Assumption 5: The Value of k Matches the Noise Level

**What it means:**
A small `k` gives a flexible but noisy model. A large `k` gives a smooth model but can erase small local
patterns.

**How to fix:**
Select `k` with cross-validation. Use an odd `k` for binary classification to reduce ties.

**Interview summary:**
k-NN assumes that the distance metric represents real similarity and that nearby training points have
similar targets. Scaling, irrelevant dimensions, sparse regions, and the choice of `k` determine whether
its local voting rule works.

## k-Means Clustering Assumptions

### Assumption 1: Euclidean Distance Represents Similarity

**What it means:**
k-means assigns each point to the nearest mean and minimizes within-cluster squared Euclidean distance:

$$
\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|_2^2.
$$

Features must therefore be numeric, and Euclidean distance must have a useful meaning.

**How to fix:**
- Standardize features.
- Encode categories carefully or use a method designed for categorical data.
- Use another distance-based method when Euclidean distance is not suitable.

### Assumption 2: Clusters Are Compact and Approximately Spherical

**What it means:**
Each cluster is represented by one center. k-means works best when clusters form compact groups around
their means.

**What happens if violated:**
k-means can split a curved cluster, combine nearby elongated clusters, or create a boundary through a
low-density region.

**How to check and fix:**
- Visualize low-dimensional projections.
- Compare with DBSCAN for non-spherical clusters.
- Compare with a Gaussian mixture when elliptical clusters are expected.

### Assumption 3: Clusters Have Similar Size and Spread

**What it means:**
The nearest-center rule works best when clusters have similar variance and density.

**What happens if violated:**
A large diffuse cluster can absorb a small cluster. A dense cluster can be split while a sparse cluster
is merged with another group.

**How to fix:**
- Try a Gaussian mixture with separate covariance matrices.
- Try DBSCAN or HDBSCAN when density differs strongly.
- Evaluate each cluster, not only the total k-means objective.

### Assumption 4: The Number of Clusters k Is Known

**What it means:**
k-means always returns exactly `k` non-empty groups when training succeeds. It cannot discover the
correct number by itself.

**How to check and fix:**
- Use domain knowledge first.
- Compare the elbow curve, silhouette score, and cluster stability.
- Check whether clusters are useful for the real task.

### Assumption 5: Outliers Are Limited

**What it means:**
The mean is sensitive to extreme values. A small number of outliers can pull a centroid away from the
main group.

**How to fix:**
- Inspect and handle outliers before clustering.
- Use robust scaling.
- Try k-medoids or a density-based method.
- Run k-means with several initializations because it can stop at a local optimum.

**Interview summary:**
k-means assumes that Euclidean distance is meaningful and that clusters are compact, roughly spherical,
and similar in spread. It also requires `k` in advance and is sensitive to scaling, outliers, and
initialization.

## DBSCAN Assumptions

### Assumption 1: Clusters Are Dense Regions Separated by Sparse Regions

**What it means:**
DBSCAN does not search for cluster centers. It joins points that have enough nearby neighbors and marks
isolated points as noise.

**Why this assumption exists:**
The algorithm defines a cluster through density connectivity. It can find curved shapes because it does
not require spherical clusters.

**What happens if violated:**
Clusters that overlap in density can merge. A cluster that is not dense enough can be labeled as noise.

### Assumption 2: One Density Scale Fits the Data

**What it means:**
The radius `eps` and minimum-neighbor value `min_samples` define one density threshold for the full
dataset. Standard DBSCAN works best when clusters have similar density.

**What happens if violated:**
- A large `eps` merges dense clusters.
- A small `eps` breaks sparse clusters into noise.
- One parameter setting cannot recover all variable-density clusters.

**How to fix:**
- Use a k-distance plot to choose `eps`.
- Validate `min_samples` based on dimension and expected noise.
- Use OPTICS or HDBSCAN when density changes strongly.

### Assumption 3: Distance and Feature Scale Are Meaningful

**What it means:**
The neighborhood definition is only useful when the distance metric matches the problem. Large-scale
features otherwise control which points count as neighbors.

**How to fix:**
- Standardize numeric features.
- Select a metric for the data type.
- Reduce irrelevant dimensions.
- Use a learned embedding for complex objects such as text or images.

### Assumption 4: The Dimension Is Not Too High

**What it means:**
In high dimensions, neighborhoods become sparse and distances become less distinct. It becomes hard to
select a useful `eps`.

**How to fix:**
- Apply feature selection or dimensionality reduction first.
- Check whether distance concentration occurs.
- Use clustering methods designed for the representation and data size.

**Interview summary:**
DBSCAN assumes that clusters are connected high-density regions separated by low-density space. It can
find irregular shapes and noise without a known cluster count. It struggles with high dimensions,
poorly scaled features, and clusters with very different densities.

## Gaussian Mixture Model (GMM) Clustering Assumptions

### Assumption 1: Data Comes From a Mixture of Gaussian Components

**What it means:**
GMM models the density as:

$$
p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x\mid\mu_k,\Sigma_k).
$$

Each hidden component has a mean, covariance matrix, and mixture weight. A point receives a probability
of belonging to each component instead of only one hard label.

**What happens if violated:**
A Gaussian component can be a poor description of a skewed, heavy-tailed, curved, or multi-modal group.
The model may use several Gaussian components to represent one real cluster.

**How to check and fix:**
- Inspect component projections and residual shape.
- Compare held-out log-likelihood.
- Transform skewed features.
- Use a different density model when Gaussian components are not suitable.

### Assumption 2: The Covariance Form Matches Cluster Shape

**What it means:**
The covariance setting controls the shapes that a component can represent:

- **Spherical:** one variance per component; round clusters.
- **Diagonal:** different variance per feature; axis-aligned clusters.
- **Tied:** all components share one covariance matrix.
- **Full:** each component can have a rotated elliptical shape.

**What happens if violated:**
A restricted covariance can underfit cluster shape. A full covariance can overfit or become singular
when data is limited.

**How to fix:**
Compare covariance types with validation likelihood, BIC, or AIC. Add covariance regularization when
matrices are nearly singular.

### Assumption 3: The Number of Components Is Selected

**What it means:**
Like k-means, a GMM needs the number of components `K`. A component is a density term and does not always
equal one real-world cluster.

**How to check and fix:**
- Compare BIC and AIC across candidate values.
- Check stability across random starts.
- Use domain meaning when converting components into business groups.

### Assumption 4: Samples Are Independent and Representative

**What it means:**
Standard maximum-likelihood fitting treats rows as independent samples from one stable mixture.
Time dependence, repeated entities, or sampling bias can produce misleading components.

**How to fix:**
- Split data by time, person, or group when needed.
- Use a time-series or hierarchical mixture for dependent data.
- Make training data match the deployment population.

### Assumption 5: Optimization Finds a Useful Solution

**What it means:**
The EM algorithm can stop at a local optimum. Results can depend on initialization. A component can also
collapse around very few points.

**How to fix:**
- Use several random initializations.
- Initialize means with k-means.
- Regularize covariance matrices.
- Check component weights and convergence warnings.

**Interview summary:**
A GMM assumes that the data density can be represented by Gaussian components. Unlike k-means, it gives
soft memberships and can model elliptical clusters through covariance matrices. It still needs a
component count and is sensitive to covariance choice, initialization, outliers, and non-Gaussian data.

## Multilayer Perceptron (MLP) Assumptions

An MLP makes fewer distribution assumptions than linear regression or GMMs. It still has important data,
model, and training conditions.

### Assumption 1: Training Data Represents Deployment Data

**What it means:**
The input-target relationship should remain reasonably stable from training to deployment. Training
examples should cover the cases that the model will see later.

**What happens if violated:**
- Performance falls under distribution shift.
- Predictions are unreliable outside the training range.
- Spurious training patterns can fail in production.

**How to check and fix:**
- Use time-based or group-based validation when appropriate.
- Monitor feature and prediction drift.
- Collect data from missing deployment cases.
- Retrain or adapt the model when the process changes.

### Assumption 2: Inputs Contain Enough Signal

**What it means:**
An MLP can learn complex functions, but it cannot recover target information that is absent from the
features. It also cannot determine causality from correlation alone.

**How to check and fix:**
- Compare with a simple baseline.
- Use domain knowledge to add useful features.
- Run feature ablations.
- Remove leakage that is available only during training.

### Assumption 3: Numeric Inputs Are Prepared Correctly

**What it means:**
MLPs train best when numeric features have comparable scales and missing values are handled explicitly.
Categorical inputs need a valid encoding or learned embedding.

**What happens if violated:**
- Optimization becomes slow or unstable.
- Large-scale features can dominate gradients.
- Missing values can produce invalid outputs.

**How to fix:**
- Standardize continuous features.
- Fit preprocessing on training data only.
- Impute missing values and add missing-value indicators when useful.
- Use one-hot encoding or embeddings for categorical features.

### Assumption 4: The Architecture Has Suitable Capacity

**What it means:**
The network must be large enough to learn the pattern but not so flexible that it only memorizes the
training set.

**What happens if violated:**
- Too little capacity causes underfitting.
- Too much capacity with too little data causes overfitting.
- Poor activation or depth choices can make gradients unstable.

**How to fix:**
- Compare training and validation learning curves.
- Tune width and depth.
- Use regularization, dropout, early stopping, or more data.
- Use ReLU-family activations and suitable weight initialization as a strong default.

### Assumption 5: The Output Layer and Loss Match the Task

**What it means:**
The final activation and loss define what the model learns:

- Regression: linear output with MSE or another suitable regression loss.
- Binary classification: one logit with binary cross-entropy.
- Single-label multiclass classification: class logits with cross-entropy.
- Multi-label classification: one binary logit per label.

**What happens if violated:**
The output can have the wrong range, probabilities may not sum correctly, or training can optimize the
wrong objective.

### Assumption 6: Optimization Can Reach a Useful Solution

**What it means:**
MLP training is non-convex. The result depends on initialization, learning rate, optimizer, batch size,
and gradient quality.

**How to check and fix:**
- Plot training and validation loss.
- Check for exploding, vanishing, or invalid gradients.
- Tune the learning rate before making the model much larger.
- Use normalization, good initialization, gradient clipping, and multiple seeds when needed.

**Interview summary:**
An MLP does not require linear relationships or normally distributed features. It assumes that training
data represents deployment, inputs contain predictive signal, preprocessing is suitable, and the network
has enough data and capacity. Its loss, output layer, and optimization setup must also match the task.

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

**Naive Bayes doesn't require:**
- Features to be independent before the class is known
- A linear decision boundary
- Large dense feature vectors

**k-NN doesn't assume:**
- A fixed parametric data distribution
- A linear decision boundary
- A known global function form

**k-means doesn't assume:**
- Labeled examples
- Gaussian probability estimates
- A linear decision boundary

**DBSCAN doesn't assume:**
- Spherical clusters
- A known number of clusters
- Every point belongs to a cluster

**GMM doesn't assume:**
- Hard cluster membership
- Spherical clusters when full covariance is used
- Equal component sizes

**MLPs don't assume:**
- Linear input-target relationships
- Normally distributed features
- Independent input features

## Summary

Understanding assumptions helps you:
1. **Choose the right model**: Match model assumptions to your data
2. **Diagnose problems**: When model fails, check assumptions
3. **Fix issues**: Know how to address assumption violations
4. **Interpret results**: Understand what results mean

Always check assumptions before and after modeling!
