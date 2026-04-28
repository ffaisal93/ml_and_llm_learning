# Topic 27: Advanced ML Theory

> 🔥 **For interviews, read these first:**
> - **`ADVANCED_THEORY_DEEP_DIVE.md`** — frontier-lab deep dive: bias-variance decomposition with proof, cross-validation theory (k-fold/stratified/group/time-series/nested with LOO closed form), learning curves (high bias vs high variance signatures), AIC vs BIC, ROC/PR curves with cost-aware operating points, F-beta scores.
> - **`INTERVIEW_GRILL.md`** — 50 active-recall questions.

## What You'll Learn

This topic covers advanced theory needed for research scientist interviews:
- Bias-Variance Tradeoff (detailed)
- Overfitting vs Underfitting
- Cross-Validation
- Learning Curves
- Model Selection
- Hyperparameter Tuning
- Feature Engineering Theory
- Data Leakage

## Why We Need This

### Interview Importance
- **Common questions**: "Explain bias-variance tradeoff"
- **Deep understanding**: Shows theoretical maturity
- **Problem-solving**: Theory guides solutions

### Real-World Application
- **Model selection**: Choose right model complexity
- **Debugging**: Understand why models fail
- **Optimization**: Improve model performance

## Core Intuition

Advanced theory questions are usually about diagnosis.

The interviewer often wants to know whether you can look at a modeling situation and identify:
- bias problem
- variance problem
- leakage problem
- data-size problem
- validation-design problem

This topic is therefore not just "more theory." It is the theory you use to reason about failures.

## Technical Details Interviewers Often Want

### Cross-Validation Is About Estimation Reliability

Cross-validation is not just a ritual for model selection.

Its purpose is to reduce dependence on one arbitrary split and give a more stable estimate of out-of-sample performance.

### Learning Curves Diagnose the Bottleneck

Learning curves help answer:
- do I need more data?
- do I need a stronger model?
- am I overfitting?

That makes them one of the most useful theory-to-practice tools in interviews.

### Feature Engineering Can Dominate Model Choice

This is a very practical interview insight:
- a better feature representation can beat a more complicated model
- bad features can make strong models look weak

## Common Failure Modes

- using cross-validation carelessly with leakage
- interpreting validation improvements without checking variance across folds
- assuming more complexity always reduces bias in a helpful way
- using learning curves without understanding what the gap means
- treating feature engineering as secondary to model choice

## Edge Cases and Follow-Up Questions

1. Why can cross-validation still be misleading if preprocessing leaks?
2. Why are learning curves so useful for deciding whether more data will help?
3. Why can a simple model with better features beat a more complex model?
4. Why is model selection really about expected generalization, not training fit?
5. Why can hyperparameter tuning itself overfit the validation process?

## What to Practice Saying Out Loud

1. How to diagnose bias vs variance from train and validation behavior
2. Why cross-validation helps and what it does not fix
3. Why leakage can invalidate even careful model selection

## Detailed Theory

### Bias-Variance Decomposition

**Mathematical Formulation:**
```
Expected Prediction Error = Bias² + Variance + Irreducible Error

Where:
- Bias²: (E[f(x)] - E[y])²  (Error from oversimplification)
- Variance: E[(f(x) - E[f(x)])²]  (Error from sensitivity to data)
- Irreducible Error: σ²  (Inherent noise, cannot be reduced)
```

**Detailed Explanation:**

**Bias (Bias²):**
Bias measures how much the average prediction differs from the true value. It represents the error from oversimplifying the model. 

**Why it happens:** When your model is too simple (e.g., linear model for non-linear data), it cannot capture the true underlying pattern. The model makes consistent errors - it's consistently wrong in the same way.

**Example:** If the true relationship is y = x², but you use a linear model y = ax + b, the model will consistently underestimate for large |x| and overestimate for small |x|. This systematic error is bias.

**How to reduce:** Use a more complex model (add polynomial features, use neural networks), add more relevant features, or train longer to let the model learn more complex patterns.

**Variance:**
Variance measures how much predictions vary when trained on different datasets. It represents the model's sensitivity to the specific training data.

**Why it happens:** When your model is too complex, it memorizes the training data instead of learning the general pattern. Small changes in training data lead to very different models and predictions.

**Example:** A decision tree with unlimited depth will create different trees for slightly different training sets, leading to very different predictions for the same input. This instability is variance.

**How to reduce:** Get more training data (reduces the impact of specific samples), use regularization (L1, L2, dropout), simplify the model (reduce depth, fewer features), or use ensemble methods (averaging reduces variance).

**Irreducible Error:**
This is the inherent noise in the data that cannot be reduced by any model. It represents the fundamental uncertainty in the problem.

**Example:** In predicting house prices, even with perfect features, there's always some randomness (e.g., buyer preferences, timing). This noise is irreducible.

**Visual Understanding:**
- **High Bias (Underfitting)**: Model too simple, consistently wrong. Like trying to hit a target but always missing in the same direction.
- **High Variance (Overfitting)**: Model too complex, predictions vary a lot. Like trying to hit a target but shots are scattered everywhere.
- **Optimal**: Balance between bias and variance. Shots are centered on target with reasonable spread.

**How to Diagnose:**
- **High Bias**: High training error AND high test error (both similar, around 30-40%). Model is too simple for the problem.
- **High Variance**: Low training error (5-10%) BUT high test error (30-40%). Large gap indicates overfitting.

**Solutions:**
- **Reduce Bias**: More complex model (add layers, polynomial features), better features (domain knowledge), longer training (more epochs), reduce regularization
- **Reduce Variance**: More data (most effective), regularization (L1, L2, dropout), simpler model (fewer layers, fewer features), ensemble methods (averaging), early stopping

### Cross-Validation

**K-Fold Cross-Validation:**

**Detailed Process:**
1. Split data into k equal-sized folds (typically k=5 or k=10)
2. For each fold i (i = 1 to k):
   - Use fold i as validation set
   - Use remaining k-1 folds as training set
   - Train model and evaluate on validation set
   - Record validation score
3. Average all k validation scores to get final performance estimate

**Example with k=5:**
- Fold 1: Train on folds 2,3,4,5 → Validate on fold 1 → Score₁
- Fold 2: Train on folds 1,3,4,5 → Validate on fold 2 → Score₂
- Fold 3: Train on folds 1,2,4,5 → Validate on fold 3 → Score₃
- Fold 4: Train on folds 1,2,3,5 → Validate on fold 4 → Score₄
- Fold 5: Train on folds 1,2,3,4 → Validate on fold 5 → Score₅
- Final score = (Score₁ + Score₂ + Score₃ + Score₄ + Score₅) / 5

**Why it works:**
- **Uses all data**: Every sample is used for both training and validation (just not at the same time)
- **More reliable**: Average of k estimates is more stable than single train/test split
- **Reduces variance**: Different train/test splits give different scores; averaging reduces this variance
- **Better estimate**: Gives you confidence interval for model performance

**When to use:**
- Limited data: Can't afford to hold out large test set
- Model selection: Compare different models fairly
- Hyperparameter tuning: Find best hyperparameters
- Performance estimation: Get reliable estimate of model performance

**Stratified K-Fold:**
- **What it does**: Maintains the same class distribution in each fold as in the full dataset
- **Why important**: For imbalanced datasets, regular k-fold might put all minority class in one fold
- **Example**: If 10% of data is class 1, each fold should have ~10% class 1
- **When to use**: Classification problems with imbalanced classes

**Leave-One-Out Cross-Validation (LOOCV):**
- Special case where k = n (number of samples)
- Train on n-1 samples, validate on 1 sample
- Very expensive but uses maximum data
- Use when dataset is very small (< 100 samples)

### Learning Curves

**What they show:**
Learning curves plot model performance (error or accuracy) as a function of training set size. You plot two curves:
- **Training error**: How well model fits training data
- **Validation error**: How well model generalizes to new data

**How to create:**
1. Start with small training set (e.g., 10 samples)
2. Train model and record training error and validation error
3. Increase training set size (20, 50, 100, 200, ...)
4. Repeat training and recording
5. Plot both errors vs training set size

**Detailed Interpretation:**

**Case 1: High Variance (Overfitting)**
- **Training error**: Starts low and stays low (model fits training data well)
- **Validation error**: Starts high, decreases but remains much higher than training error
- **Gap**: Large gap between training and validation error
- **What it means**: Model memorizes training data but doesn't generalize
- **Solution**: More data (gap will decrease), regularization, simpler model

**Case 2: High Bias (Underfitting)**
- **Training error**: Starts high and decreases slowly, but remains high
- **Validation error**: Similar to training error (small gap)
- **Both converge**: Both errors converge to high value
- **What it means**: Model is too simple to capture the pattern
- **Solution**: More complex model, better features, longer training

**Case 3: Good Fit**
- **Training error**: Decreases and stabilizes at low value
- **Validation error**: Decreases and converges close to training error
- **Small gap**: Small, stable gap between curves
- **What it means**: Model has learned the pattern and generalizes well
- **Action**: Model is good, can deploy

**Case 4: Need More Data**
- **Training error**: Decreasing
- **Validation error**: Still decreasing (hasn't converged)
- **Gap**: Gap is decreasing as you add more data
- **What it means**: More data will help
- **Solution**: Collect more training data

**Why learning curves are useful:**
- **Diagnose problems**: Quickly see if you have bias or variance problem
- **Data efficiency**: See if more data will help
- **Model selection**: Compare different models
- **Resource planning**: Decide if worth collecting more data

### Feature Engineering Theory

**Why it matters:**
Feature engineering is often more important than the choice of algorithm. A simple model with great features often outperforms a complex model with poor features. This is because features determine what patterns the model can learn.

**The Feature Engineering Process:**

**Step 1: Understanding the Domain**
Before creating features, you need to understand the problem domain. What factors actually influence the target variable? For example:
- **House prices**: Location, size, age, neighborhood quality
- **Customer churn**: Usage patterns, payment history, support interactions
- **Fraud detection**: Transaction patterns, user behavior, velocity

**Step 2: Creating Features**

**Encoding Categorical Variables:**
- **One-hot encoding**: Each category becomes a binary feature. Use when categories are nominal (no order). Example: Color [Red, Blue, Green] → [1,0,0], [0,1,0], [0,0,1]
- **Label encoding**: Assign numbers to categories. Use when categories are ordinal (have order). Example: Size [Small, Medium, Large] → [0, 1, 2]
- **Target encoding**: Encode categories by their average target value. More powerful but can cause overfitting if not done carefully.

**Scaling Numerical Features:**
- **Standardization (Z-score)**: (x - mean) / std. Transforms to mean=0, std=1. Use when features have different scales and you're using distance-based algorithms (KNN, SVM, neural networks).
- **Normalization (Min-Max)**: (x - min) / (max - min). Transforms to range [0, 1]. Use when you need bounded features.
- **Why scale**: Algorithms like neural networks, SVM, KNN are sensitive to feature scales. Without scaling, features with larger ranges dominate.

**Polynomial Features:**
Create interactions and powers of features. Example: If you have x₁ and x₂, create x₁², x₂², x₁×x₂. This allows linear models to learn non-linear patterns. However, be careful: number of features grows exponentially (curse of dimensionality).

**Binning:**
Convert continuous features to categorical by grouping values into bins. Example: Age [0-18, 19-35, 36-50, 51+]. Useful when relationship is non-linear or you want to handle outliers.

**Time Features:**
Extract information from timestamps:
- **Cyclical**: Day of week (0-6), hour of day (0-23), month (1-12)
- **Time since**: Days since last event, time since account creation
- **Seasonality**: Is it holiday? Is it weekend?
- **Why important**: Many patterns are time-dependent (e.g., sales higher on weekends)

**Interaction Features:**
Combine multiple features. Example: If price and quality both matter, create price × quality. This captures relationships that individual features can't.

**Step 3: Feature Selection**
Not all features are useful. Some might be:
- **Irrelevant**: Don't relate to target
- **Redundant**: Highly correlated with other features
- **Noisy**: Add more noise than signal

**Methods:**
- **Correlation analysis**: Remove features highly correlated with others
- **Feature importance**: Use tree-based models to see which features matter
- **Univariate selection**: Test each feature individually
- **Recursive feature elimination**: Iteratively remove least important features

**Step 4: Validation**
Always validate features on validation set, not training set. Features that work on training might not generalize.

**Common Pitfalls:**
- **Data leakage**: Using information that won't be available at prediction time
- **Overfitting**: Creating too many features leads to overfitting
- **Ignoring domain knowledge**: Not using expert knowledge about what matters

### Data Leakage

**What is Data Leakage:**
Data leakage occurs when information from the future (or from the test set) is used to make predictions. This makes your model look good during training but fail in production.

**Types of Data Leakage:**

**1. Target Leakage:**
Using features that wouldn't be available at prediction time because they're a result of the target variable.

**Example 1:** Predicting if someone will default on a loan, and using "number of late payments" as a feature. But late payments only happen AFTER someone defaults, so this information won't be available when making the prediction.

**Example 2:** Predicting customer churn and using "cancellation request date" as a feature. If someone has requested cancellation, they're already churning - this is the target, not a feature.

**How to detect:** Ask yourself: "Would I have this information when making the prediction?" If the answer is no, it's target leakage.

**How to prevent:** Remove features that are direct results of the target. Use only features that are available BEFORE the event you're predicting.

**2. Train-Test Leakage:**
Information from the test set leaks into the training process. This happens when you do preprocessing (like scaling, imputation) on the full dataset before splitting.

**Example:** Computing mean and std for standardization using the full dataset (including test set), then splitting. The test set statistics influence training, which is leakage.

**How it happens:**
```python
# WRONG - Data leakage!
scaler = StandardScaler()
scaler.fit(X_full)  # Includes test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CORRECT - No leakage
scaler = StandardScaler()
scaler.fit(X_train)  # Only training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**How to prevent:** Always split data first, then do all preprocessing (scaling, imputation, feature engineering) using only training data. Apply the same transformations to test data.

**3. Temporal Leakage:**
Using future information to predict the past. This is common in time series problems.

**Example:** Predicting stock price on day 10 using information from day 11. In reality, you can only use information up to day 10.

**How to prevent:** For time series, always use time-based splitting. Training data should be from earlier time periods, test data from later periods. Never shuffle time series data randomly.

**4. Preprocessing Leakage:**
Computing statistics (mean, median, etc.) on full dataset before splitting.

**Example:** Computing mean to fill missing values using full dataset. This leaks test set information into training.

**How to prevent:** Compute statistics only on training set, then apply to test set.

**How to Detect Data Leakage:**

**Red Flags:**
- **Unrealistically high performance**: If your model performs too well (e.g., 99% accuracy), suspect leakage
- **Perfect correlation**: If a feature has perfect correlation with target, it might be leakage
- **Feature importance**: If a feature has extremely high importance, investigate if it's leakage
- **Production failure**: Model works great in validation but fails in production - classic sign of leakage

**Detection Methods:**
1. **Domain knowledge**: Understand what information is available when
2. **Feature analysis**: Check if features are available at prediction time
3. **Correlation analysis**: Very high correlation might indicate leakage
4. **Ablation study**: Remove suspicious features and see if performance drops dramatically

**Best Practices:**
1. **Split first**: Always split data before any preprocessing
2. **Time-based split**: For time series, use temporal splitting
3. **Pipeline**: Use scikit-learn Pipeline to ensure proper order
4. **Validation**: Use cross-validation correctly (fit on fold, transform on fold)
5. **Documentation**: Document when each feature becomes available
6. **Review**: Have someone review your features for leakage

## Exercises

1. Derive bias-variance decomposition
2. Implement k-fold cross-validation
3. Plot learning curves
4. Detect data leakage

## Next Steps

- **Topic 28**: Business use cases
- **Topic 29**: System design for ML
