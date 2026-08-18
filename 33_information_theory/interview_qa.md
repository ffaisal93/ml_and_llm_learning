# Information Theory Interview Q&A

## Q1: Explain entropy. What does it measure?

**In plain terms.** Entropy is a single number for "how unpredictable is this?" Zero means you already know the answer; the maximum means every outcome is equally likely. Everything below is just that idea written as a formula and applied to coins.

**Answer:**

**Entropy** measures the uncertainty or randomness in a probability distribution.

**Mathematical Definition:**
```
H(X) = -Σ p(x) * log₂(p(x))
```

**Detailed Explanation:**

**What it measures:**
- **High entropy**: High uncertainty, uniform distribution
- **Low entropy**: Low uncertainty, concentrated distribution
- **Zero entropy**: No uncertainty, deterministic (one outcome has probability 1)

**Example:**
- **Fair coin**: p(heads) = 0.5, p(tails) = 0.5
  - H = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1 bit
  - Maximum uncertainty for binary variable
  
- **Biased coin**: p(heads) = 0.9, p(tails) = 0.1
  - H = -0.9*log₂(0.9) - 0.1*log₂(0.1) ≈ 0.47 bits
  - Less uncertainty (we're more confident it's heads)

- **Deterministic**: p(heads) = 1.0, p(tails) = 0.0
  - H = 0 bits
  - No uncertainty (always heads)

**Properties:**
1. **Non-negative**: H(X) ≥ 0
2. **Maximum for uniform**: H(X) ≤ log₂(n) for n outcomes
3. **Zero for deterministic**: H(X) = 0 when one outcome has probability 1

**Use Cases:**
- **Decision trees**: Choose splits that maximize information gain (reduce entropy)
- **Compression**: Entropy is lower bound on average code length
- **Feature selection**: Features with high entropy are more informative

> **Saying it out loud.** Entropy is average surprise. A fair coin gives you exactly one bit, because you genuinely don't know and one yes-or-no answer settles it. A coin that lands heads ninety percent of the time gives you only about 0.47 bits, because you'd guess heads and mostly be right — there's less to learn. And a two-headed coin gives you zero, because there's no news in the outcome at all. The reason this matters practically is that entropy is the floor on compression — Shannon proved you can't encode a source in fewer bits than its entropy — and it's the split criterion in decision trees, where information gain is just the entropy you removed.

---

## Q2: What is cross-entropy? Why is it used as a loss function?

**Answer:**

**Cross-Entropy** measures the average number of bits needed to encode events from distribution P using a code optimized for distribution Q.

**Mathematical Definition:**
```
H(P, Q) = -Σ p(x) * log(q(x))

Where:
- P: True distribution
- Q: Predicted distribution
```

**Why it's a good loss function:**

**1. Penalizes confident wrong predictions:**
- If true class is A (p=1.0) and the model assigns A high probability (q=0.9)
- Loss = -log₂(0.9) ≈ 0.15 bits (small penalty)
- But if the model assigns A low probability (q=0.1), i.e. it confidently predicted some other class
- Loss = -log₂(0.1) ≈ 3.32 bits (large penalty)
- **Encourages calibrated probabilities**

**2. Mathematically well-founded:**
- Related to maximum likelihood estimation
- Minimizing cross-entropy = maximizing likelihood

**3. Gradient properties:**
- Smooth gradients (no discontinuities)
- Well-behaved optimization

**4. Always ≥ entropy:**
- H(P, Q) ≥ H(P)
- Equal when Q = P (perfect prediction)
- Measures how far Q is from P

**Example:**
```
True distribution: [1.0, 0.0, 0.0]  (class 0)
Perfect prediction: [1.0, 0.0, 0.0] → Cross-entropy = 0
Good prediction: [0.8, 0.1, 0.1] → Cross-entropy ≈ 0.32 bits
Bad prediction: [0.1, 0.8, 0.1] → Cross-entropy ≈ 3.32 bits
```

**Use Cases:**
- **Classification**: Most common loss function
- **Language modeling**: Next token prediction
- **Any probabilistic prediction**: When comparing true vs predicted distributions

> **Saying it out loud.** Cross-entropy asks "how surprised was the model by the right answer?" and makes that the loss. If the model gave the true class ninety percent, the penalty is tiny; if it gave the true class ten percent, the penalty is more than twenty times bigger, because the log blows up as the probability goes to zero. That asymmetric punishment is exactly what you want — it makes confident mistakes expensive and pushes toward honest, calibrated probabilities. It's also not an arbitrary choice: minimizing cross-entropy is maximum likelihood under a categorical distribution, and it equals the data's entropy plus the KL divergence to your model. The number that anchors it: guessing uniformly over a thousand classes gives you a loss of about 6.9 nats, so that's your "model has learned nothing" baseline.

---

## Q3: Explain KL divergence. Why is it asymmetric?

**In plain terms.** KL divergence is the penalty for holding the wrong beliefs. If the world behaves like P and you planned for Q, KL says how much that mistake costs you, measured in bits. It is not a distance — being wrong about P in terms of Q is a different cost from the reverse.

**Answer:**

**KL Divergence** measures how different two probability distributions are.

**Mathematical Definition:**
```
KL(P || Q) = Σ p(x) * log(p(x) / q(x))
           = H(P, Q) - H(P)

Where:
- P: True/reference distribution
- Q: Approximated distribution
```

**Why it's asymmetric:**

**KL(P || Q) vs KL(Q || P):**

**KL(P || Q)**: "How surprised are we when we expect Q but get P?"
- Measures how well Q approximates P
- Penalizes when Q assigns low probability to events that P assigns high probability
- Example: If P(x) = 0.9 but Q(x) = 0.1, KL is large

**KL(Q || P)**: "How surprised are we when we expect P but get Q?"
- Measures how well P approximates Q
- Different interpretation, different value

**Example:**
```
P = [0.5, 0.5]  (uniform)
Q = [0.9, 0.1]  (biased)

KL(P || Q) = 0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1) ≈ 0.51 nats
KL(Q || P) = 0.9*ln(0.9/0.5) + 0.1*ln(0.1/0.5) ≈ 0.37 nats

→ Not equal (asymmetric)
```

**Properties:**
1. **Non-negative**: KL(P || Q) ≥ 0
2. **Zero when equal**: KL(P || Q) = 0 if and only if P = Q
3. **Asymmetric**: KL(P || Q) ≠ KL(Q || P)
4. **Not a metric**: Doesn't satisfy triangle inequality

**Use Cases:**
- **RLHF**: KL penalty to keep policy close to reference
- **VAEs**: KL between posterior and prior
- **Model comparison**: Compare different models
- **Regularization**: Prevent overfitting

> **Saying it out loud.** KL divergence is the extra cost of using the wrong distribution — the bits you waste by planning for Q when the world runs on P. It's asymmetric because the expectation is taken under one of the two distributions, so whichever one you put first is the one whose regions of high probability get weighted. Concretely: putting near-zero probability where the truth has lots of mass is catastrophic and can even be infinite, but putting mass where the truth is thin is merely wasteful. That asymmetry is why direction matters — forward KL makes your model cover everything and produce bland averages, reverse KL makes it lock onto one mode and be confidently narrow. And it's why calling KL a distance in an interview will get you corrected: it fails symmetry and the triangle inequality both.

---

## Q4: What is mutual information? How is it used in feature selection?

**Answer:**

**Mutual Information** measures how much information one random variable gives about another.

**Mathematical Definition:**
```
I(X; Y) = H(X) - H(X | Y)
        = H(Y) - H(Y | X)
        = H(X) + H(Y) - H(X, Y)
```

**Interpretation:**
- **I(X; Y) = 0**: X and Y are independent (no information shared)
- **I(X; Y) > 0**: X and Y are dependent (share information)
- **I(X; Y) = H(X)**: Y completely determines X (no uncertainty about X remains)
- **Symmetric**: I(X; Y) = I(Y; X)

**Example:**
- **Independent**: X = coin flip, Y = another coin flip
  - I(X; Y) = 0 (no information shared)
  
- **Dependent**: X = weather, Y = umbrella usage
  - I(X; Y) > 0 (weather gives information about umbrella)
  
- **Deterministic**: Y = X (same variable)
  - I(X; Y) = H(X) (maximum information)

**Feature Selection with Mutual Information:**

**Algorithm:**
1. Compute I(X_i; Y) for each feature X_i
2. Select features with high mutual information
3. High MI = feature is informative about target

**Why it works:**
- Features with high MI are strongly related to target
- Removes irrelevant features (MI ≈ 0)
- Captures non-linear relationships (unlike correlation)

**Example:**
```python
# Features and target
X1 = [1, 2, 3, 4, 5]  # High MI with Y
X2 = [1, 1, 1, 1, 1]  # Low MI (constant)
Y = [2, 4, 6, 8, 10]  # Y = 2*X1

# MI(X1, Y) is high (X1 determines Y)
# MI(X2, Y) is low (X2 is constant, no information)
```

**Use Cases:**
- **Feature selection**: Select informative features
- **Information bottleneck**: Compress while preserving information
- **Clustering**: Measure cluster quality
- **Dimensionality reduction**: Preserve mutual information

> **Saying it out loud.** Mutual information asks how much knowing one thing tells you about another. Zero means completely independent — knowing the weather tells you nothing about the coin flip. Higher means one variable erases uncertainty about the other, and the maximum is when one determines the other entirely, at which point mutual information equals that variable's whole entropy. For feature selection, this is better than correlation because correlation only sees straight lines: a feature that's a perfect parabola of the target has zero correlation and high mutual information. The tradeoff to name is that mutual information is univariate as usually applied — it scores each feature against the target independently, so it happily keeps two redundant features and misses combinations that only matter jointly.

---

## Q5: Compare Gini impurity and entropy. When would you use each?

**Answer:**

**Gini Impurity:**
```
Gini = 1 - Σ p_i²
```

**Entropy:**
```
H = -Σ p_i * log(p_i)
```

**Comparison:**

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Formula** | 1 - Σ p_i² | -Σ p_i * log(p_i) |
| **Computation** | Faster (no log) | Slower (needs log) |
| **Range (binary)** | [0, 0.5] | [0, 1] |
| **Sensitivity** | More sensitive | Less sensitive |
| **Theoretical** | Empirical | Information-theoretic |

**When to Use Gini:**
- **Decision trees (CART)**: Faster computation
- **Large datasets**: Speed matters
- **Binary classification**: Simple and effective

**When to Use Entropy:**
- **Decision trees (ID3, C4.5)**: More theoretically grounded
- **Information gain**: Directly related to entropy
- **When you need information-theoretic interpretation**

**In Practice:**
- **Both work similarly**: Results are usually very similar
- **Gini slightly faster**: No logarithm computation
- **Entropy more standard**: Better theoretical foundation
- **Choice often doesn't matter**: Both give similar splits

**Example:**
```
Distribution: [0.5, 0.5]
Gini = 1 - (0.5² + 0.5²) = 0.5
Entropy = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1.0

Distribution: [0.9, 0.1]
Gini = 1 - (0.9² + 0.1²) = 0.18
Entropy = -0.9*log₂(0.9) - 0.1*log₂(0.1) ≈ 0.47
```

**Key Insight:**
- Both measure impurity/uncertainty
- Both are minimized when pure (one class)
- Both are maximized when uniform
- Gini is faster, entropy is more standard

> **Saying it out loud.** Gini and entropy are two ways of asking the same question — how mixed up are the labels in this node — and in practice they almost never disagree about which split to take. Entropy uses logarithms, which makes it slightly slower but gives it the information-theoretic story: information gain is literally the uncertainty you removed. Gini is one minus the sum of squared probabilities, which has a neat reading of its own: it's the probability you'd misclassify a random item if you labeled it by randomly drawing from the node's class distribution. The honest answer is that the choice is not where your model's accuracy comes from — depth, number of trees, and feature quality matter far more, and studies find the two criteria disagree on well under five percent of splits.

---

## Q6: What is Jensen-Shannon divergence? How does it differ from KL divergence?

**Answer:**

**Jensen-Shannon Divergence** is a symmetric version of KL divergence.

**Mathematical Definition:**
```
JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

Where:
M = 0.5 * (P + Q)  (average distribution)
```

**Key Differences from KL:**

| Property | KL Divergence | JS Divergence |
|----------|---------------|---------------|
| **Symmetric** | No | Yes |
| **Bounded** | No (can be ∞) | Yes [0, 1] |
| **Metric** | No | √JS is a metric; JS itself does not satisfy the triangle inequality |
| **Stability** | Can be unstable | More stable |

**Why JS Divergence?**

**1. Symmetry:**
- JS(P || Q) = JS(Q || P)
- More intuitive for comparing distributions
- No need to choose "reference" distribution

**2. Bounded:**
- JS(P || Q) ∈ [0, 1] (when using log base 2)
- Easier to interpret
- KL can be infinite when distributions don't overlap

**3. Metric (via its square root):**
- √JS satisfies the triangle inequality, so it is a true metric
- JS itself is bounded and symmetric but not a metric
- KL is neither symmetric nor a metric

**4. Stability:**
- More stable when distributions are very different
- KL can explode when Q assigns 0 probability to events P assigns high probability

**Use Cases:**
- **GANs**: Measure distance between real and generated distributions
- **Model comparison**: When you need symmetric distance
- **Clustering**: Measure cluster separation
- **When KL is unstable**: Use JS as more stable alternative

**Example:**
```
P = [0.5, 0.5]
Q = [0.9, 0.1]

KL(P || Q) ≈ 0.51 nats
KL(Q || P) ≈ 0.37 nats  (different!)

JS(P || Q) = JS(Q || P) ≈ 0.10 nats  (symmetric)
```

> **Saying it out loud.** Jensen-Shannon is the symmetric cousin of KL: instead of comparing P to Q, you compare each of them to their average. That fixes the two things that make KL awkward — it's symmetric, so there's no "reference" distribution to choose, and it's bounded by log two, so it never blows up to infinity when the supports don't overlap. Its square root is even a genuine metric, satisfying the triangle inequality. The catch is that its boundedness is also its weakness: when two distributions don't overlap at all, JS is pegged at its maximum with a flat gradient, which is exactly the vanishing-gradient failure that made early GANs so hard to train and motivated the switch to Wasserstein distance.

---

## Q7: How do you use these metrics in practice?

**Answer:**

**Entropy:**
- **Decision trees**: Information gain = H(parent) - weighted H(children)
- **Feature selection**: High entropy features are more informative
- **Compression**: Lower bound on code length

**Cross-Entropy:**
- **Classification loss**: Most common loss function
- **Language modeling**: Next token prediction
- **Any probabilistic model**: When comparing true vs predicted

**KL Divergence:**
- **RLHF**: KL penalty to keep policy close to reference
- **VAEs**: KL between posterior and prior
- **Regularization**: Prevent overfitting
- **Model comparison**: Compare different models

**Mutual Information:**
- **Feature selection**: Select features with high MI with target
- **Information bottleneck**: Compress while preserving information
- **Clustering**: Measure cluster quality

**Gini Impurity:**
- **Decision trees (CART)**: Measure node impurity
- **Classification**: Alternative to entropy (faster)

**JS Divergence:**
- **GANs**: Measure distance between distributions
- **Model comparison**: When you need symmetric metric
- **Clustering**: When KL is unstable

> **Saying it out loud.** The way I keep these straight is by what each one is for. Entropy is a property of one distribution — how uncertain it is — and it's what decision trees minimize when they split. Cross-entropy compares your prediction to the truth and is the loss you actually train on. KL is the same comparison with the data's own entropy removed, which makes it the right tool when you want a regularizer rather than a loss — the KL anchor in RLHF, the prior term in a VAE. Mutual information is about pairs of variables and is the feature-selection and representation-learning tool. And Gini is just a cheaper entropy for trees. If someone asks which to use, the answer is almost always determined by whether you're comparing one distribution to itself, to another, or to a joint.

---

## Summary

All these metrics are fundamental to machine learning:
- **Entropy**: Uncertainty measure
- **Cross-Entropy**: Classification loss
- **KL Divergence**: Distribution distance (asymmetric)
- **Mutual Information**: Information shared between variables
- **Gini**: Misclassification probability
- **JS Divergence**: Symmetric distribution distance

Understanding these is crucial for:
- Decision trees
- Neural networks
- RLHF/DPO
- Feature selection
- Model evaluation

