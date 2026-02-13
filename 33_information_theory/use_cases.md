# Information Theory Metrics: Use Cases

## When to Use Each Metric

### Entropy

**Use When:**
- Building decision trees (ID3, C4.5)
- Measuring uncertainty in distributions
- Information gain calculations
- Feature selection (high entropy = more informative)

**Example:**
```python
# Decision tree split selection
parent_entropy = entropy(parent_labels)
left_entropy = entropy(left_labels)
right_entropy = entropy(right_labels)

# Information gain
info_gain = parent_entropy - (|left|/|parent| * left_entropy + 
                              |right|/|parent| * right_entropy)
# Choose split with maximum information gain
```

### Cross-Entropy

**Use When:**
- Classification loss function (most common)
- Language modeling (next token prediction)
- Any probabilistic prediction task
- Comparing true vs predicted distributions

**Example:**
```python
# Classification
true_labels = [0, 1, 2]  # Class indices
pred_probs = model(input)  # [batch, n_classes] probabilities
loss = cross_entropy_loss(pred_probs, true_labels)
```

### KL Divergence

**Use When:**
- RLHF (KL penalty to keep policy close to reference)
- VAEs (KL between posterior and prior)
- Model comparison
- Regularization (prevent overfitting)
- Distribution matching

**Example:**
```python
# RLHF KL penalty
policy_logprobs = policy_model(input)
reference_logprobs = reference_model(input)
kl_penalty = beta * kl_divergence(policy_logprobs, reference_logprobs)
loss = policy_loss + kl_penalty
```

### Mutual Information

**Use When:**
- Feature selection (select informative features)
- Information bottleneck (compress while preserving info)
- Clustering evaluation
- Dimensionality reduction
- Understanding feature relationships

**Example:**
```python
# Feature selection
for feature in features:
    mi = mutual_information(feature, target)
    if mi > threshold:
        selected_features.append(feature)
```

### Gini Impurity

**Use When:**
- Decision trees (CART algorithm)
- Classification impurity measure
- When you need faster computation than entropy
- Binary classification

**Example:**
```python
# Decision tree split
gini_parent = gini_impurity(parent_labels)
gini_left = gini_impurity(left_labels)
gini_right = gini_impurity(right_labels)

# Gini gain
gini_gain = gini_parent - (|left|/|parent| * gini_left + 
                          |right|/|parent| * gini_right)
```

### Jensen-Shannon Divergence

**Use When:**
- GANs (measure distance between real and generated)
- Model comparison (when you need symmetric distance)
- Clustering (when KL is unstable)
- When distributions might not overlap (KL can be infinite)

**Example:**
```python
# GAN training
real_dist = real_data_distribution
generated_dist = generator_output_distribution
distance = jensen_shannon_divergence(real_dist, generated_dist)
# Minimize distance to make generated data similar to real
```

## Quick Reference Table

| Metric | Primary Use | Key Property |
|--------|-------------|--------------|
| **Entropy** | Decision trees, uncertainty | H(X) ≥ 0, max when uniform |
| **Cross-Entropy** | Classification loss | H(P,Q) ≥ H(P), = H(P) when Q=P |
| **KL Divergence** | RLHF, VAEs, regularization | Asymmetric, KL(P\|Q) ≥ 0 |
| **Mutual Information** | Feature selection | I(X;Y) = 0 if independent |
| **Gini** | Decision trees (CART) | Faster than entropy |
| **JS Divergence** | GANs, symmetric distance | Symmetric, bounded [0,1] |

## Common Patterns

### Pattern 1: Decision Tree Split Selection
```python
# Use entropy or Gini
def choose_best_split(X, y):
    best_gain = -float('inf')
    for feature, threshold in candidate_splits:
        left, right = split(X, y, feature, threshold)
        gain = information_gain(y, left, right)  # Uses entropy
        # or
        gain = gini_gain(y, left, right)  # Uses Gini
        if gain > best_gain:
            best_gain = gain
            best_split = (feature, threshold)
```

### Pattern 2: Classification Loss
```python
# Always use cross-entropy
loss = nn.CrossEntropyLoss()(predictions, true_labels)
```

### Pattern 3: RLHF Regularization
```python
# Use KL divergence
kl_penalty = beta * kl_divergence(policy_dist, reference_dist)
total_loss = policy_loss + kl_penalty
```

### Pattern 4: Feature Selection
```python
# Use mutual information
for feature in features:
    mi = mutual_information(feature, target)
    if mi > threshold:
        selected.append(feature)
```

