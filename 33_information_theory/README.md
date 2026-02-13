# Topic 33: Information Theory & Probability Metrics

## What You'll Learn

This topic covers essential information theory and probability concepts:
- Entropy
- Cross-Entropy
- KL Divergence
- Mutual Information
- Gini Impurity
- Jensen-Shannon Divergence
- Detailed mathematical explanations
- Simple implementations
- When to use each

## Why We Need This

### Interview Importance
- **Common questions**: "Explain KL divergence", "What is entropy?"
- **Mathematical foundation**: Shows deep understanding
- **Practical knowledge**: Used in many ML algorithms

### Real-World Application
- **Loss functions**: Cross-entropy for classification
- **Regularization**: KL divergence in RLHF, VAEs
- **Feature selection**: Mutual information
- **Decision trees**: Gini impurity, entropy

## Detailed Theory

### 1. Entropy

**What is Entropy?**
Entropy measures the uncertainty or randomness in a probability distribution. Higher entropy = more uncertainty.

**Mathematical Formulation:**
```
H(X) = -Σ p(x) * log₂(p(x))

Where:
- X: Random variable
- p(x): Probability of outcome x
- Sum over all possible outcomes
```

**Interpretation:**
- **High entropy**: Uniform distribution (maximum uncertainty)
- **Low entropy**: Concentrated distribution (low uncertainty)
- **Zero entropy**: Deterministic (one outcome has probability 1)

**Example:**
- Fair coin: H = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1 bit (maximum uncertainty)
- Biased coin (90% heads): H = -0.9*log₂(0.9) - 0.1*log₂(0.1) ≈ 0.47 bits (less uncertainty)

**Properties:**
- H(X) ≥ 0 (always non-negative)
- Maximum when uniform distribution
- Minimum (0) when deterministic

**Use Cases:**
- Decision trees: Choose splits that maximize information gain (reduce entropy)
- Compression: Entropy is lower bound on average code length
- Feature selection: Features with high entropy are more informative

### 2. Cross-Entropy

**What is Cross-Entropy?**
Cross-entropy measures the average number of bits needed to encode events from distribution P using a code optimized for distribution Q.

**Mathematical Formulation:**
```
H(P, Q) = -Σ p(x) * log(q(x))

Where:
- P: True distribution
- Q: Predicted/approximated distribution
- p(x): True probability
- q(x): Predicted probability
```

**Interpretation:**
- Measures how different Q is from P
- Always ≥ H(P) (entropy of true distribution)
- Equal to H(P) when Q = P
- Used as loss function in classification

**Why it's a good loss function:**
- Penalizes confident wrong predictions heavily
- Encourages calibrated probabilities
- Mathematically well-founded

**Use Cases:**
- **Classification**: Cross-entropy loss (most common)
- **Language modeling**: Next token prediction
- **Any probabilistic prediction**: When you have true vs predicted distributions

### 3. KL Divergence (Kullback-Leibler Divergence)

**What is KL Divergence?**
KL divergence measures how different two probability distributions are. It's the "distance" between distributions (though not a true metric).

**Mathematical Formulation:**
```
KL(P || Q) = Σ p(x) * log(p(x) / q(x))
           = Σ p(x) * log(p(x)) - Σ p(x) * log(q(x))
           = H(P, Q) - H(P)

Where:
- P: True/reference distribution
- Q: Approximated distribution
```

**Interpretation:**
- **KL(P || Q) = 0**: Distributions are identical
- **KL(P || Q) > 0**: Distributions are different
- **Not symmetric**: KL(P || Q) ≠ KL(Q || P) in general
- **Not a metric**: Doesn't satisfy triangle inequality

**Properties:**
- KL(P || Q) ≥ 0 (always non-negative)
- KL(P || Q) = 0 if and only if P = Q
- Asymmetric: KL(P || Q) ≠ KL(Q || P)

**Use Cases:**
- **RLHF**: KL penalty to keep policy close to reference
- **VAEs**: KL divergence between posterior and prior
- **Model comparison**: Compare different models
- **Regularization**: Prevent overfitting

**Why asymmetric?**
- KL(P || Q): "How surprised are we when we expect Q but get P?"
- KL(Q || P): "How surprised are we when we expect P but get Q?"
- Different interpretations, different values

### 4. Mutual Information

**What is Mutual Information?**
Mutual information measures how much information one random variable gives about another. It's the reduction in uncertainty about Y when we know X.

**Mathematical Formulation:**
```
I(X; Y) = H(X) - H(X | Y)
        = H(Y) - H(Y | X)
        = H(X) + H(Y) - H(X, Y)

Where:
- H(X): Entropy of X
- H(X | Y): Conditional entropy of X given Y
- H(X, Y): Joint entropy
```

**Interpretation:**
- **I(X; Y) = 0**: X and Y are independent (no information shared)
- **I(X; Y) > 0**: X and Y are dependent (share information)
- **I(X; Y) = H(X)**: X completely determines Y
- **Symmetric**: I(X; Y) = I(Y; X)

**Use Cases:**
- **Feature selection**: Select features with high mutual information with target
- **Information bottleneck**: Compress while preserving information
- **Clustering**: Measure cluster quality
- **Dimensionality reduction**: Preserve mutual information

**Example:**
- If X and Y are independent: I(X; Y) = 0
- If Y = X: I(X; Y) = H(X) (maximum)
- If Y = f(X) (deterministic function): I(X; Y) = H(Y)

### 5. Gini Impurity

**What is Gini Impurity?**
Gini impurity measures the probability of misclassifying a randomly chosen element if it were labeled according to the distribution of classes.

**Mathematical Formulation:**
```
Gini = 1 - Σ p_i²

Where:
- p_i: Proportion of class i
- Sum over all classes
```

**Interpretation:**
- **Gini = 0**: Pure node (all same class)
- **Gini = 1 - 1/k**: Maximum impurity for k classes (uniform distribution)
- **Range**: [0, 1-1/k] for k classes
- **For binary**: Range [0, 0.5]

**Properties:**
- Minimum (0) when pure (one class)
- Maximum when uniform distribution
- Similar to entropy but different formula

**Gini vs Entropy:**
- **Gini**: Gini = 1 - Σ p_i² (faster to compute)
- **Entropy**: H = -Σ p_i * log(p_i) (more information-theoretic)
- **In practice**: Both work similarly for decision trees
- **Gini**: Slightly faster, more sensitive to class probability changes
- **Entropy**: More theoretically grounded

**Use Cases:**
- **Decision trees**: CART algorithm uses Gini
- **Classification**: Measure node impurity
- **Feature selection**: Choose splits that minimize Gini

### 6. Jensen-Shannon Divergence

**What is JS Divergence?**
JS divergence is a symmetric version of KL divergence. It's a true metric (satisfies triangle inequality).

**Mathematical Formulation:**
```
JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

Where:
- M = 0.5 * (P + Q) (average distribution)
```

**Properties:**
- **Symmetric**: JS(P || Q) = JS(Q || P)
- **Bounded**: JS(P || Q) ∈ [0, 1] (when using log base 2)
- **Metric**: Satisfies triangle inequality
- **Smooth**: More stable than KL divergence

**Use Cases:**
- **GANs**: Measure distance between real and generated distributions
- **Model comparison**: When you need symmetric distance
- **Clustering**: Measure cluster separation

## Industry-Standard Boilerplate Code

See `information_theory.py` for complete implementations.

## Exercises

1. Implement all metrics from scratch
2. Compare Gini vs Entropy
3. Compute mutual information for feature selection
4. Use KL divergence for regularization

## Next Steps

- Use these in decision trees, neural networks, RLHF
- Reference when implementing algorithms

