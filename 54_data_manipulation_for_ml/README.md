# Topic 54: Data Manipulation for ML

## What You'll Learn

Some ML interviews still include practical data wrangling.

This topic focuses on the patterns you are most likely to need quickly:

- filtering rows
- handling missing values
- groupby aggregations
- joins and merges
- feature normalization
- one-hot encoding
- train/test leakage concerns in preprocessing

## Why This Matters

A lot of candidates prepare only model code and then get slowed down by:
- basic pandas manipulation
- feature engineering tables
- aggregations over users, sessions, or time windows

For research and applied interviews, this can matter a lot because messy data is often where real work starts.

## Core Intuition

### 1. Data Manipulation Is Part of Modeling

If your feature table is wrong, the model is wrong.

This means:
- row-level correctness matters
- time logic matters
- grouping logic matters
- leakage can happen before training even begins

### 2. Most Interview Tasks Are Simple but Easy to Mess Up

Typical prompts:
- compute per-user statistics
- fill missing values
- merge labels with features
- create normalized columns
- aggregate by day or by category

The difficulty is usually not deep math. It is avoiding silent mistakes.

### 3. Leakage Risk

Very common issue:

- compute global mean using all rows
- then use it in the training pipeline

That is wrong if the test split influenced the transform.

Correct principle:
- fit preprocessing on training data
- apply the learned transform to validation/test

## Common Failure Modes

### 1. Wrong Join Semantics

A merge can silently:
- drop rows
- duplicate rows
- misalign labels and features

That can completely change the dataset before modeling starts.

### 2. Aggregating Over the Wrong Time Window

This is a common source of leakage.

If you compute a user feature using events that occur after the prediction timestamp, the model gets future information it should not have.

### 3. Filling Missing Values Without Thinking About Meaning

A missing value can mean:
- zero
- unknown
- not applicable
- sensor failure

Treating all missing values the same can erase useful signal or create misleading artifacts.

### 4. Fitting Preprocessing on the Full Dataset

Normalization, imputation, encoding, and PCA should usually be fit on training data only.

Otherwise evaluation is contaminated.

## Edge Cases and Follow-Up Questions

### What if a left join creates duplicate rows?

That usually means the join key is not unique on the right-hand side.

You should inspect key cardinality before trusting the result.

### What if a category appears only at test time?

Then one-hot encoding needs a policy for unseen categories.

A good answer is to mention an "unknown" bucket or a learned fallback.

### What if missingness itself is predictive?

Then the missing-value indicator can be useful as its own feature.

### What if normalization is done per batch instead of from training statistics?

Then the feature meaning can drift across splits, and evaluation may become inconsistent.

## Files in This Topic

- [data_manipulation.py](/Users/faisal/Projects/ml_and_llm_learning/54_data_manipulation_for_ml/data_manipulation.py): compact pandas and NumPy patterns

## What to Practice Saying Out Loud

1. Why can preprocessing cause leakage?
2. How do you aggregate per entity without duplicating rows incorrectly?
3. When would you use a left join versus an inner join?
4. Why should normalization parameters come from training data only?
5. How do you handle missing values without hiding a signal?
