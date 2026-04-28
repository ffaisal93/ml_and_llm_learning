# Blind Coding Drills

Use these with a timer and no notes.

---

## 15-Minute Drills

### 1. Stable Softmax

Implement stable softmax for a 1D NumPy array.

Must mention:
- subtract max
- normalization
- overflow prevention

### 2. Binary Logistic Regression Step

Implement:
- sigmoid
- BCE
- one gradient descent step

Must mention:
- shape of `X`
- why gradient becomes `p - y`

### 3. Causal Mask

Implement a causal mask of shape `(n, n)`.

Must mention:
- lower-triangular mask
- why future tokens are blocked

### 4. Top-k Filtering

Given logits, keep only the top `k` entries and set the rest to a very negative number.

Must mention:
- edge case if `k >= vocab_size`

---

## 20-Minute Drills

### 5. Masked Softmax

Implement masked softmax over the last dimension.

Must mention:
- mask convention
- very negative fill
- stable softmax

### 6. K-Means One Iteration

Given points and current centers:
- assign cluster labels
- recompute centers

Must mention:
- empty cluster handling
- runtime

### 7. Decision Tree Best Split

Given `X` and labels `y`, find the best threshold over one feature using Gini impurity.

Must mention:
- weighted impurity
- skipping invalid splits

### 8. Pairwise Squared Distances

Implement vectorized squared distance between every row in `X` and every row in `C`.

Must mention:
- shape of result
- why vectorization is faster

---

## 30-Minute Drills

### 9. Attention From Scratch

Implement:
- scaled dot-product attention
- optional mask
- return attention weights

Must mention:
- score shape
- `sqrt(d_k)` scaling
- softmax axis

### 10. Beam Search Skeleton

Write a simple beam search loop for token generation.

Must mention:
- cumulative log probabilities
- beam pruning
- stopping condition

### 11. Bootstrap Confidence Interval

Implement percentile bootstrap for a user-specified statistic.

Must mention:
- sampling with replacement
- number of bootstrap samples
- percentile endpoints

### 12. Data Leakage Check

Given train and test tables:
- count duplicates across splits
- explain one more leakage pattern

Must mention:
- why duplicate overlap invalidates evaluation

---

## Review Rule

After each drill, ask:
- Was the code correct?
- Was the explanation structured?
- Did I state runtime and edge cases?
- Did I freeze on a small syntax issue?
