# ML & LLM Interview Q&A: 100+ Questions

Comprehensive interview questions and answers for ML/LLM coding interviews.

## Table of Contents
1. [Classical ML](#classical-ml)
2. [LLM Fundamentals](#llm-fundamentals)
3. [LLM Inference](#llm-inference)
4. [Training Techniques](#training-techniques)
5. [Optimization](#optimization)
6. [Regularization](#regularization)
7. [Bias & Variance](#bias--variance)

---

## Classical ML

### Q1: Implement linear regression from scratch.

**Answer:**
```python
class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            y_pred = X.dot(self.weights) + self.bias
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias
```

**Key Points:**
- Gradient descent: Update weights using gradients
- Cost function: MSE = mean((y_pred - y)²)
- Gradients: dw = X.T @ (y_pred - y) / n, db = mean(y_pred - y)

---

### Q2: What's the difference between linear and logistic regression?

**Answer:**

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|---------------------|
| **Output** | Continuous values | Probabilities (0-1) |
| **Activation** | None (linear) | Sigmoid |
| **Cost Function** | MSE | Log loss (cross-entropy) |
| **Use Case** | Regression | Classification |
| **Gradient** | Linear | Non-linear (sigmoid derivative) |

**Key Difference:**
- Linear: y = w*x + b
- Logistic: p = sigmoid(w*x + b), then classify p > 0.5

---

### Q3: Explain KNN algorithm.

**Answer:**
K-Nearest Neighbors is a lazy learning algorithm:

**Algorithm:**
1. Store all training data (no training phase)
2. For new point, find k nearest neighbors
3. For classification: Majority vote
4. For regression: Average of neighbors

**Distance Metric:**
- Usually Euclidean: √(Σ(xi - yi)²)
- Can use Manhattan, cosine, etc.

**K Value:**
- Small k: More sensitive to noise (high variance)
- Large k: Smoother decision boundary (high bias)
- Rule of thumb: k = √n

**Time Complexity:**
- Training: O(1) - just store data
- Prediction: O(n) - compare to all points

---

### Q4: How does K-means clustering work?

**Answer:**

**Algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat steps 2-3 until convergence

**Convergence:**
- Centroids don't change
- Or max iterations reached

**Initialization:**
- Random: Can get poor results
- K-means++: Better initialization

**Limitations:**
- Assumes spherical clusters
- Need to specify k
- Sensitive to initialization

---

## LLM Fundamentals

### Q5: Explain the transformer architecture.

**Answer:**

**Components:**
1. **Embedding Layer**: Token → Dense vectors
2. **Position Encoding**: Add position info
3. **Transformer Blocks** (N layers):
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Layer Normalization
   - Residual Connections
4. **Output Layer**: Project to vocabulary

**Key Innovation:**
- Self-attention: Relate all positions
- Parallel processing: All positions at once
- Long-range dependencies: No RNN limitations

---

### Q6: How does self-attention work?

**Answer:**

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Steps:**
1. Compute Q, K, V from input
2. Compute attention scores: Q @ K^T
3. Scale by √d_k (prevent large values)
4. Softmax to get attention weights
5. Apply weights to V

**Why it works:**
- Query asks "what am I looking for?"
- Key answers "what information do I have?"
- Value is "the actual information"
- Attention weights show relevance

---

### Q7: What is multi-head attention?

**Answer:**

**Concept:**
- Instead of one attention, use multiple "heads"
- Each head learns different relationships
- Concatenate all heads, then project

**Why Multiple Heads:**
- Different heads attend to different aspects
- Example: One head for syntax, one for semantics
- More expressive than single head

**Implementation:**
1. Split d_model into num_heads × d_k
2. Each head has its own Q, K, V
3. Compute attention for each head
4. Concatenate outputs
5. Final projection

---

## LLM Inference

### Q8: How does KV caching work?

**Answer:**

**Problem:**
- Autoregressive generation: Generate token by token
- Each token needs attention to all previous tokens
- Without cache: Recompute attention for all tokens each step

**Solution:**
- Cache K and V matrices for previous tokens
- New token: Only compute Q, reuse cached K/V
- Append new K/V to cache

**Example:**
```
Step 1: Token 1 → Compute Q1, K1, V1, cache K1, V1
Step 2: Token 2 → Compute Q2, K2, V2
        → Attention: Q2 @ [K1, K2]^T, use [V1, V2]
        → Cache: [K1, K2], [V1, V2]
Step 3: Token 3 → Compute Q3, K3, V3
        → Attention: Q3 @ [K1, K2, K3]^T, use [V1, V2, V3]
        → Cache: [K1, K2, K3], [V1, V2, V3]
```

**Speedup:** 10-100x for generation

---

### Q9: What is quantization and why use it?

**Answer:**

**Quantization:** Reduce model precision
- FP32 → FP16: 2x smaller, 2x faster
- FP16 → INT8: 2x smaller, 2x faster
- INT8 → INT4: 2x smaller, 2x faster

**Why:**
- **Memory**: Smaller models fit in memory
- **Speed**: Faster computation
- **Cost**: Lower inference cost

**Trade-off:**
- Accuracy may decrease slightly
- Need calibration for INT8/INT4

**Process:**
1. Find min/max of weights
2. Calculate scale factor
3. Quantize to integer range
4. Store scale for dequantization

---

### Q10: Explain top-p (nucleus) sampling.

**Answer:**

**Algorithm:**
1. Sort tokens by probability (descending)
2. Compute cumulative probability
3. Find smallest set where cum_prob >= p
4. Sample from this "nucleus"
5. Renormalize probabilities

**Why it works:**
- Adaptive: Number of tokens varies
- High probability tokens: Always included
- Low probability tokens: Excluded
- Better than top-k (fixed size)

**Example:**
```
Probabilities: [0.5, 0.3, 0.1, 0.05, 0.03, ...]
Cumulative:    [0.5, 0.8, 0.9, 0.95, 0.98, ...]
Top-p=0.9: Nucleus = first 3 tokens (cum_prob = 0.9)
```

---

## Training Techniques

### Q11: Explain RLHF (Reinforcement Learning from Human Feedback).

**Answer:**

**Pipeline:**
1. **Supervised Fine-tuning**: Train on human demonstrations
2. **Reward Model**: Train on human preferences (chosen vs rejected)
3. **RL Optimization**: Use PPO to optimize policy with reward model

**Why RLHF:**
- Align models with human preferences
- Make models helpful, harmless, honest
- Improve response quality

**Challenges:**
- Need human feedback data
- Reward model training
- RL optimization complexity

---

### Q12: What is DPO and how does it differ from RLHF?

**Answer:**

**DPO (Direct Preference Optimization):**
- Directly optimizes policy to prefer chosen over rejected
- No reward model needed
- Uses reference model instead

**Key Difference:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Yes | No |
| **Reference Model** | Used in RL | Used directly |
| **Complexity** | High | Lower |
| **Flexibility** | More | Less |

**DPO Loss:**
```
Loss = -log(σ(β * (log π_chosen - log π_rejected - log π_ref_chosen + log π_ref_rejected)))
```

Where σ is sigmoid, β is temperature.

---

### Q13: Explain PPO (Proximal Policy Optimization) in detail. Why is it used in RLHF?

**Answer:**

**What is PPO?**
PPO is a policy gradient algorithm that prevents large policy updates by clipping the objective function.

**Mathematical Formulation:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s) (importance sampling ratio)
- A: Advantage estimate
- ε: Clipping parameter (typically 0.1-0.3)
```

**Why Clipping?**
- Prevents large updates that can destabilize training
- Policy changes gradually (more stable)
- Can reuse same data multiple times (sample efficient)

**Why PPO in RLHF:**
1. **Stability**: Language models are sensitive - need stable updates
2. **Sample efficiency**: Human feedback is expensive - reuse data
3. **KL constraint**: Keeps policy close to reference
4. **Proven**: Works well in practice (ChatGPT, Claude)

**PPO Algorithm:**
1. Collect trajectories with current policy
2. Compute advantages A(s,a)
3. For K epochs:
   - Compute ratio r(θ) = π_θ / π_θ_old
   - Compute clipped objective
   - Update policy
4. Update old policy

---

### Q14: What is GRPO (Group Relative Policy Optimization)? When is it useful?

**Answer:**

**What is GRPO?**
GRPO extends PPO to handle multiple groups with different preferences. Optimizes relative to group baseline, not absolute reward.

**Mathematical Formulation:**
```
L_GRPO = -E[r(θ) * (R_group - R_baseline)] + β * KL(π_θ || π_ref)

Where:
- R_group: Reward for specific group
- R_baseline: Average reward across all groups
- r(θ): Importance sampling ratio
```

**Why GRPO?**
- **Multiple preferences**: Different user groups have different preferences
- **Relative optimization**: Optimize to be better than baseline
- **Fairness**: Ensures all groups improve relative to average

**Use Cases:**
- Different age groups, regions, cultures
- Different use cases (coding, writing, analysis)
- Different skill levels (beginners vs experts)

**Example:**
- Group A: Prefer concise responses
- Group B: Prefer detailed responses
- GRPO optimizes to be better than baseline for each group

---

### Q15: What are the main challenges in RL alignment? How do you address them?

**Answer:**

**Challenge 1: Reward Hacking**
- **Problem**: Model finds ways to maximize reward that don't align with intent
- **Solution**: Careful reward design, KL penalty, monitoring

**Challenge 2: Distribution Shift**
- **Problem**: Policy changes, but reward model trained on old distribution
- **Solution**: Retrain reward model periodically, regularization

**Challenge 3: Mode Collapse**
- **Problem**: Policy collapses to single response pattern
- **Solution**: KL penalty, entropy bonus, diverse training data

**Challenge 4: Instability**
- **Problem**: Training can be unstable
- **Solution**: PPO clipping, gradient clipping, learning rate scheduling

**Challenge 5: Human Feedback Quality**
- **Problem**: Inconsistent or biased feedback
- **Solution**: Multiple annotators, quality control, bias detection

---

### Q16: How do you prevent reward hacking in RLHF?

**Answer:**

**What is Reward Hacking?**
Model finds unintended ways to maximize reward (e.g., always says "I can't answer").

**Prevention:**
1. **Careful reward design**: Multiple signals, penalize hacks
2. **Regularization**: KL penalty prevents extreme behaviors
3. **Reward model robustness**: Diverse training, bias detection
4. **Monitoring**: Track patterns, detect anomalies
5. **Constrained optimization**: Hard/soft constraints
6. **Iterative refinement**: Identify hacks, refine reward

---

### Q17: Explain the KL penalty in RLHF. Why is it important?

**Answer:**

**What is KL Penalty?**
KL divergence measures how different policy is from reference. Penalty prevents large deviations.

**Mathematical Formulation:**
```
KL(π_θ || π_ref) = E[log(π_θ(a|s) / π_ref(a|s))]

In practice:
KL_penalty = β * (log π_θ - log π_ref)
```

**Why Important:**
1. **Prevents mode collapse**: Keeps policy diverse
2. **Prevents reward hacking**: Constrains to reasonable behaviors
3. **Maintains capabilities**: Preserves SFT capabilities
4. **Stability**: Prevents large policy changes
5. **Trust region**: Policy can't deviate too far

**How to Choose β:**
- Too small: Policy can deviate too much
- Too large: Policy can't learn
- Typical: β = 0.1-0.5

---

See `08_training_techniques/rl_alignment_qa.md` for even more detailed answers!

---

## Optimization

### Q13: Explain the Adam optimizer.

**Answer:**

**Adam = Adaptive Moment Estimation**

**Components:**
1. **First moment (m)**: Exponential moving average of gradients (momentum)
2. **Second moment (v)**: Exponential moving average of squared gradients (variance)
3. **Bias correction**: Fix initial bias

**Update Rule:**
```
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * g_t²
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
```

**Why it works:**
- Adaptive learning rates per parameter
- Momentum for smooth updates
- Second moment adapts to gradient variance

**Default hyperparameters:**
- β1 = 0.9 (momentum)
- β2 = 0.999 (variance)
- α = 0.001 (learning rate)

---

### Q14: What's the difference between Adam and AdamW?

**Answer:**

**Adam:** Weight decay applied to gradients
**AdamW:** Weight decay decoupled (applied separately)

**Why AdamW:**
- Better weight decay
- Improved generalization
- More principled

**Difference:**
```python
# Adam: weight decay in gradient
gradient = gradient + weight_decay * params

# AdamW: weight decay separate
params = params - lr * (adam_update + weight_decay * params)
```

---

## Regularization

### Q15: Explain L1 vs L2 regularization.

**Answer:**

**L1 (Lasso):**
- Penalty: λ * Σ|w|
- Effect: Many weights become exactly 0
- Use: Feature selection, sparsity
- Gradient: Constant (λ * sign(w))

**L2 (Ridge):**
- Penalty: λ * Σw²
- Effect: Weights shrink toward 0 (but not 0)
- Use: Generalization, most common
- Gradient: Linear (2λ * w)

**When to use:**
- L1: Want feature selection, interpretability
- L2: Want generalization, standard choice

**Elastic Net:** Combines both

---

### Q16: How does dropout work?

**Answer:**

**During Training:**
1. Randomly set some activations to 0 (with probability p)
2. Scale remaining activations by 1/(1-p)
3. This prevents co-adaptation

**During Inference:**
- No dropout
- Scale all activations by (1-p) to maintain expected value

**Why it works:**
- Prevents neurons from co-adapting
- Forces model to be robust
- Acts as ensemble of sub-networks

**Common rates:**
- Input layer: 0.1-0.2
- Hidden layers: 0.5
- Output layer: Usually no dropout

---

## Bias & Variance

### Q17: Explain bias-variance tradeoff.

**Answer:**

**Bias:** Error from oversimplifying
- High bias = Underfitting
- Model too simple

**Variance:** Error from sensitivity to training data
- High variance = Overfitting
- Model too complex

**Tradeoff:** Can't minimize both
- Simple model: High bias, low variance
- Complex model: Low bias, high variance
- Goal: Find balance

**Diagnosis:**
- High bias: High train error, high test error (similar)
- High variance: Low train error, high test error (gap)

**Solutions:**
- High bias: More complex model, better features
- High variance: More data, regularization, simpler model

---

## Information Theory

### Q18: Explain entropy. What does it measure?

**Answer:**

**Entropy** measures uncertainty/randomness in a probability distribution.

**Formula:** H(X) = -Σ p(x) * log₂(p(x))

**Interpretation:**
- High entropy: High uncertainty (uniform distribution)
- Low entropy: Low uncertainty (concentrated distribution)
- Zero entropy: Deterministic (one outcome has prob=1)

**Example:**
- Fair coin: H = 1 bit (maximum uncertainty)
- Biased coin (90/10): H ≈ 0.47 bits (less uncertainty)
- Deterministic: H = 0 bits (no uncertainty)

**Use Cases:**
- Decision trees: Information gain
- Compression: Lower bound on code length
- Feature selection: High entropy = more informative

---

### Q19: What is cross-entropy? Why is it used as a loss function?

**Answer:**

**Cross-Entropy:** H(P, Q) = -Σ p(x) * log(q(x))

**Why good loss function:**
1. **Penalizes confident wrong predictions**: -log(0.1) = 3.32 (large penalty)
2. **Encourages calibrated probabilities**: Mathematically well-founded
3. **Always ≥ entropy**: H(P, Q) ≥ H(P), equal when Q = P
4. **Smooth gradients**: Well-behaved optimization

**Use Cases:**
- Classification (most common loss)
- Language modeling
- Any probabilistic prediction

---

### Q20: Explain KL divergence. Why is it asymmetric?

**Answer:**

**KL Divergence:** KL(P || Q) = Σ p(x) * log(p(x) / q(x))

**Why asymmetric:**
- KL(P || Q): "How surprised when we expect Q but get P?"
- KL(Q || P): "How surprised when we expect P but get Q?"
- Different interpretations → different values

**Properties:**
- KL(P || Q) ≥ 0
- KL(P || Q) = 0 if and only if P = Q
- Not a metric (doesn't satisfy triangle inequality)

**Use Cases:**
- RLHF: KL penalty
- VAEs: KL between posterior and prior
- Model comparison
- Regularization

---

### Q21: What is mutual information? How is it used in feature selection?

**Answer:**

**Mutual Information:** I(X; Y) = H(X) + H(Y) - H(X, Y)

**Interpretation:**
- I(X; Y) = 0: X and Y independent
- I(X; Y) > 0: X and Y dependent
- I(X; Y) = H(X): X completely determines Y

**Feature Selection:**
1. Compute I(X_i; Y) for each feature
2. Select features with high MI
3. High MI = feature is informative about target

**Why it works:**
- Captures non-linear relationships (unlike correlation)
- Removes irrelevant features (MI ≈ 0)
- Information-theoretic foundation

---

### Q22: Compare Gini impurity and entropy. When would you use each?

**Answer:**

**Gini:** Gini = 1 - Σ p_i² (faster, no log)
**Entropy:** H = -Σ p_i * log(p_i) (more theoretical)

**Comparison:**
- **Gini**: Faster computation, used in CART
- **Entropy**: More theoretical, used in ID3/C4.5
- **In practice**: Both work similarly, results usually very similar

**When to use:**
- **Gini**: When speed matters, CART algorithm
- **Entropy**: When you need information-theoretic interpretation

---

### Q23: What is Jensen-Shannon divergence? How does it differ from KL?

**Answer:**

**JS Divergence:** JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
Where M = 0.5 * (P + Q)

**Key Differences:**
- **Symmetric**: JS(P || Q) = JS(Q || P) (KL is asymmetric)
- **Bounded**: JS ∈ [0, 1] (KL can be infinite)
- **Metric**: Satisfies triangle inequality (KL doesn't)
- **Stable**: More stable when distributions very different

**Use Cases:**
- GANs: Measure distance between distributions
- Model comparison: When you need symmetric distance
- When KL is unstable

---

See `33_information_theory/interview_qa.md` for even more detailed answers!

---

## Discriminative vs Generative Models

### Q24: Explain the difference between discriminative and generative models.

**Answer:**

**Discriminative Models:**
- Learn P(Y|X) directly - the conditional probability of Y given X
- Focus on finding the decision boundary between classes
- Don't model the data distribution
- Examples: Logistic Regression, SVM, Neural Networks, Decision Trees

**Generative Models:**
- Learn P(X, Y) = P(X|Y) * P(Y) - the joint probability distribution
- Model how data is generated
- Can generate new data samples
- Examples: Naive Bayes, GMM, GANs, VAEs, Language Models

**Key Differences:**

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| **What they learn** | P(Y\|X) | P(X, Y) |
| **Can generate data** | No | Yes |
| **Data efficiency** | More efficient | Less efficient |
| **Complexity** | Simpler | More complex |
| **Use case** | Prediction | Generation + Prediction |

**When to use:**
- **Discriminative**: When you only need predictions, have limited data
- **Generative**: When you need to generate data, have missing data, want to understand distribution

---

### Q25: What are the assumptions of linear regression?

**Answer:**

**1. Linearity:** Relationship between X and Y is linear
- Check: Plot residuals vs predicted (should be random)
- Fix: Add polynomial features, transformations

**2. Independence of Errors:** Errors are independent
- Check: Durbin-Watson test, plot residuals vs time
- Fix: Use time series models for temporal data

**3. Homoscedasticity:** Constant variance of errors
- Check: Plot residuals vs predicted (look for funnel)
- Fix: Weighted least squares, transformations

**4. Normality of Errors:** Errors are normally distributed
- Check: Q-Q plot, Shapiro-Wilk test
- Fix: Transformations (less critical for large samples)

**5. No Multicollinearity:** Features not highly correlated
- Check: Correlation matrix, VIF
- Fix: Remove correlated features, regularization

**What happens if violated:**
- Poor predictions, wrong standard errors, unreliable tests
- See `34_discriminative_generative/model_assumptions_detailed.md` for detailed explanations

---

### Q26: What are the assumptions of logistic regression?

**Answer:**

**1. Binary Outcome:** Y must be binary (0 or 1)
- Fix: Use multinomial logistic for multi-class

**2. Linearity of Log-Odds:** Log-odds is linear in X
- Check: Box-Tidwell test
- Fix: Add polynomial features, interactions

**3. Independence:** Observations are independent
- Fix: Use mixed-effects models for correlated data

**4. No Multicollinearity:** Features not highly correlated
- Same as linear regression

**5. Large Sample Size:** Need sufficient data
- Rule of thumb: 10-20 observations per feature
- Fix: Collect more data, reduce features

**Differences from linear regression:**
- No normality assumption (errors are binary)
- No homoscedasticity (variance = p(1-p))
- Probability is sigmoid (non-linear), not linear

---

### Q27: What are the assumptions of SVM?

**Answer:**

**1. Separable Data:** Data should be (nearly) separable
- Hard-margin: Must be linearly separable
- Soft-margin: Most points separable, some violations OK
- Fix: Use kernel, allow margin violations

**2. Feature Scaling:** Features must be scaled
- **Critical!** SVM is very sensitive to scales
- Fix: Always use StandardScaler or MinMaxScaler

**3. Appropriate Kernel:** Kernel should match data structure
- Linear: Linearly separable data
- RBF: Non-linear, local structure
- Polynomial: Polynomial relationships
- Fix: Try different kernels, use cross-validation

**4. Balanced Classes:** Can be sensitive to imbalance
- Fix: Use class weights, SMOTE, cost-sensitive learning

**What SVM doesn't assume:**
- Normal distributions
- Linear relationships (with kernels)
- Large sample size

---

### Q28: Explain Bayes' theorem in detail.

**Answer:**

**Mathematical Formulation:**
```
P(A|B) = P(B|A) * P(A) / P(B)
```

**Components:**
- **Prior P(A)**: Belief about A before seeing evidence
- **Likelihood P(B|A)**: Probability of evidence B given A
- **Evidence P(B)**: Total probability of B
- **Posterior P(A|B)**: Updated belief about A after seeing B

**Why it matters:**
- Updates beliefs with evidence
- Foundation of Bayesian statistics
- Used in Naive Bayes, spam detection, medical diagnosis

**Example:**
- Disease prevalence: 1% (prior)
- Test accuracy: 95% (likelihood)
- Positive test → Only 16% chance of disease!
- Why? False positives from large healthy population

**Use Cases:**
- Naive Bayes classifier
- Spam detection
- Medical diagnosis
- Recommendation systems

See `34_discriminative_generative/bayes_theorem_detailed.md` for comprehensive explanation!

---

See `34_discriminative_generative/model_assumptions_detailed.md` for detailed assumption explanations!

---

## Kernel Functions

### Q29: What is a kernel function? Explain the kernel trick.

**Answer:**

**Kernel Function:**
Kernel K(x, y) computes dot product in high-dimensional space without explicitly computing the transformation.

**Formula:** K(x, y) = φ(x) · φ(y)

**Kernel Trick:**
- **Problem**: Transform data to high dimensions (expensive)
- **Solution**: Use kernel to compute dot product directly (cheap)
- **Benefit**: Get high-dimensional features without computing them

**Example:**
Polynomial kernel (degree=2):
- Without trick: Compute [x₁, x₂, x₁², x₂², x₁x₂, ...] (8 dimensions)
- With trick: Just compute (x · y)² (same result, much faster!)

**Why it works:**
Algorithms like SVM only need dot products, not features themselves.

---

### Q30: Explain different types of kernels. When would you use each?

**Answer:**

**Linear Kernel:** K(x, y) = x · y
- **Use when**: Linearly separable data, high-dimensional data
- **Example**: Text classification with TF-IDF

**Polynomial Kernel:** K(x, y) = (γx·y + r)^d
- **Use when**: Polynomial relationships, moderate non-linearity
- **Example**: Circular boundaries (degree=2)

**RBF Kernel:** K(x, y) = exp(-γ||x-y||²)
- **Use when**: Non-linear problems (default choice)
- **Example**: Complex boundaries, most common kernel

**Sigmoid Kernel:** K(x, y) = tanh(γx·y + r)
- **Use when**: Rarely (RBF is usually better)

**Selection:**
1. Try linear first
2. If fails, use RBF
3. If RBF overfits, try polynomial

---

### Q31: Explain RBF kernel in detail. How does gamma affect it?

**Answer:**

**RBF Kernel:** K(x, y) = exp(-γ||x-y||²)

**What it does:**
Measures similarity based on distance. Close points → high similarity, far points → low similarity.

**Gamma Effect:**
- **Low γ (0.001)**: Wide kernel → Simpler boundary, risk underfitting
- **Medium γ (0.1-1.0)**: Balanced → Good starting point
- **High γ (10.0)**: Narrow kernel → Complex boundary, risk overfitting

**Visual:**
Each point creates a "bump". Low gamma = wide bumps (simple), high gamma = narrow bumps (complex).

**Tuning:**
Start with γ = 1/(n_features * variance), then grid search.

---

### Q32: How do you choose the right kernel?

**Answer:**

**Decision Process:**
1. **Try linear first**: Fast, interpretable
2. **If fails, try RBF**: Default for non-linear
3. **If RBF overfits, try polynomial**: Less flexible
4. **Never use sigmoid**: RBF is better

**Parameter Tuning:**
- **RBF**: Tune gamma [0.001, 0.01, 0.1, 1.0, 10.0] and C [0.1, 1, 10, 100, 1000]
- **Polynomial**: Start with degree=2, gamma=1.0
- **Use cross-validation**: Compare different kernels and parameters

**Key Points:**
- Always scale features before SVM
- Linear often works for high-dimensional data
- RBF is most common for non-linear problems

---

See `35_kernel_functions/interview_qa.md` for even more detailed answers!

---

## NLP Basics

### Q33: Explain TF-IDF. How does it work?

**Answer:**

**TF-IDF** measures how important a word is to a document in a collection.

**Components:**

**Term Frequency (TF):**
- TF(t, d) = count(t, d) / |d|
- How often word appears in document
- Higher TF = more important to document

**Inverse Document Frequency (IDF):**
- IDF(t, D) = log(N / |{d : t ∈ d}|)
- How rare word is across documents
- Common words (appear in many docs) → low IDF
- Rare words (appear in few docs) → high IDF

**TF-IDF:**
- TF-IDF(t, d) = TF(t, d) × IDF(t, D)
- High TF-IDF: Word appears often in this document (high TF) but rarely in others (high IDF)
- Identifies characteristic words for each document

**Example:**
- "algorithm" in Python tutorial: High TF (appears often) + High IDF (rare word) → High TF-IDF
- "the" in any document: High TF but Low IDF (common word) → Lower TF-IDF

**Use Cases:**
- Text classification (feature extraction)
- Search engines (ranking)
- Information retrieval

---

### Q34: What are n-grams? Explain n-gram language models.

**Answer:**

**N-grams** are contiguous sequences of n items (words) from text.

**Types:**
- **Unigram (1-gram)**: Single words ["machine", "learning"]
- **Bigram (2-gram)**: Pairs ["machine learning", "learning is"]
- **Trigram (3-gram)**: Triplets ["machine learning is"]

**N-gram Language Model:**

**Bigram Model:**
```
P(w₁, w₂, ..., wₙ) ≈ P(w₁) × P(w₂|w₁) × P(w₃|w₂) × ... × P(wₙ|wₙ₋₁)

Where:
P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
```

**Why N-grams:**
- **Unigram**: Simple but ignores word order
- **Bigram**: Captures local dependencies
- **Higher n**: More context but needs more data

**Trade-offs:**
- Higher n: More context, better predictions
- But: Needs more data (exponential growth), sparse data problem

**Use Cases:**
- Language modeling (predict next word)
- Text generation
- Spell checking

---

### Q35: What is Laplace smoothing? Why is it needed?

**Answer:**

**Laplace Smoothing (Add-k):**
Handles zero probability problem in n-gram models.

**Problem:**
- Unseen n-grams have P = 0
- Product of probabilities becomes 0
- Model can't handle unseen text

**Solution:**
```
P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + k) / (count(wᵢ₋₁) + k*V)

Where:
- k: Smoothing parameter (usually 1)
- V: Vocabulary size
```

**Effect:**
- **Seen n-grams**: Slightly lower probability
- **Unseen n-grams**: Non-zero probability (fixed!)
- **Redistributes**: Probability from seen to unseen

**Example:**
Training: "the cat", "the dog"
Test: "the bird" (unseen)

- Without smoothing: P(bird|the) = 0 (problem!)
- With smoothing: P(bird|the) = 1/5 = 0.2 (fixed!)

**Why needed:**
- Prevents zeros
- Allows generalization to unseen text
- Essential for language models

---

### Q36: Explain the Bayesian interpretation of L1/L2 regularization.

**Answer:**

**L2 Regularization = Gaussian Prior:**
- **Frequentist**: Loss = MSE + λ||w||²
- **Bayesian**: Prior w ~ N(0, 1/λ)
- **Interpretation**: Parameters normally distributed around 0
- **Effect**: Shrinks all parameters toward 0 (smooth, no sparsity)

**L1 Regularization = Laplace Prior:**
- **Frequentist**: Loss = MSE + λ||w||₁
- **Bayesian**: Prior w ~ Laplace(0, 1/λ)
- **Interpretation**: Parameters Laplace distributed (sharp peak at 0)
- **Effect**: Shrinks parameters to exactly 0 (sparse, feature selection)

**Key Differences:**
- **L2 (Gaussian)**: Smooth bell curve, no sparsity
- **L1 (Laplace)**: Sharp peak at 0, creates sparsity

**Why it matters:**
- Helps choose right regularization
- Understand why L1 creates sparsity
- Interpret regularization strength (λ = prior variance)

**Use:**
- **L2**: Prevent overfitting, all features relevant
- **L1**: Feature selection, many irrelevant features

---

See `36_nlp_basics/regularization_priors.md` for comprehensive explanation!

---

### Q37: Explain BLEU score. How is it calculated?

**Answer:**

**BLEU** (Bilingual Evaluation Understudy) measures quality of machine translation or text generation.

**Components:**

**1. N-gram Precision:**
- Precision for n=1,2,3,4 (unigram, bigram, trigram, 4-gram)
- p_n = (matching n-grams) / (total n-grams in candidate)
- Clipped: Count capped at reference count

**2. Brevity Penalty (BP):**
- Penalizes short translations
- BP = 1 if candidate > reference length
- BP = exp(1 - ref_len/cand_len) otherwise

**3. BLEU Formula:**
```
BLEU = BP * exp(Σ w_n * log(p_n))

Where:
- w_n: Weights (usually [0.25, 0.25, 0.25, 0.25])
- p_n: n-gram precisions
```

**Range:** 0 to 1 (higher is better)

**Interpretation:**
- 1.0: Perfect match
- 0.5-0.7: Good translation
- <0.3: Poor translation

**Limitations:**
- Doesn't consider meaning (only n-gram overlap)
- Doesn't handle synonyms well
- Favors shorter translations (even with BP)

---

### Q38: Explain ROUGE score. What are ROUGE-1, ROUGE-2, ROUGE-L?

**Answer:**

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference text.

**ROUGE-1 (Unigram):**
- Measures word overlap
- ROUGE-1 = (overlapping words) / (words in reference)
- Focus: Content words

**ROUGE-2 (Bigram):**
- Measures bigram overlap
- ROUGE-2 = (overlapping bigrams) / (bigrams in reference)
- Focus: Word order and phrases

**ROUGE-L (Longest Common Subsequence):**
- Measures LCS overlap
- ROUGE-L = LCS(candidate, reference) / length(reference)
- Focus: Sentence structure and order
- LCS: Longest sequence appearing in both (not necessarily contiguous)

**Returns:** Precision, Recall, F1

**Use Cases:**
- **Summarization**: Primary metric
- **Text generation**: Secondary metric
- **ROUGE-1**: Content coverage
- **ROUGE-2**: Phrase matching
- **ROUGE-L**: Structure similarity

**Comparison to BLEU:**
- **BLEU**: Precision-oriented (translation)
- **ROUGE**: Recall-oriented (summarization)
- **ROUGE-L**: Better for order-sensitive tasks

---

### Q39: How do you handle large database schemas in NL2Code?

**Answer:**

**Problem:** Large schemas (thousands of tables/columns) don't fit in context window.

**Solutions:**

**1. Schema Pruning:**
- **Relevance scoring**: Score tables/columns by relevance to query
  - TF-IDF similarity
  - Embedding similarity (BERT)
  - Keyword matching
- **Top-K selection**: Select top-K most relevant elements
- **Hierarchical**: Prune at table level, then column level

**2. Schema Encoding:**
- **Hierarchical encoding**: Encode at different levels
- **Graph neural networks**: Model schema as graph
- **Separate encoding**: Encode schema separately, combine later

**3. Two-Stage Approach:**
- **Stage 1**: Schema selection (which tables/columns needed)
- **Stage 2**: Code generation (given selected schema)

**4. Retrieval-Augmented:**
- **Retrieve relevant schema**: Use retrieval to find relevant parts
- **Dynamic context**: Add retrieved schema to context
- **Iterative**: Refine selection based on generation

**Standard Procedure:**
```
Query → Schema Pruning → Schema Encoding → Code Generation → Code
```

**Example:**
- Query: "Find customers who bought products in 2023"
- Pruned schema: customers, orders, products tables + relevant columns
- Generated SQL: SELECT with JOINs on relevant tables

**Best Practices:**
- Index schemas for fast retrieval
- Add schema descriptions
- Handle schema versioning
- Validate generated code

---

### Q40: What are the standard procedures for different NLP tasks?

**Answer:**

**1. Text Classification:**
- Preprocess → Feature extraction (TF-IDF/embeddings) → Model → Evaluate
- Metrics: Accuracy, F1-score

**2. NER:**
- BIO tagging → Embeddings → Sequence labeling (CRF/BiLSTM) → Extract entities
- Metrics: F1 per entity type

**3. Question Answering:**
- Encode question+context → Attention → Extract answer span
- Metrics: EM, F1

**4. Machine Translation:**
- Parallel corpus → Tokenization → Seq2Seq/Transformer → Beam search
- Metrics: BLEU, METEOR

**5. Summarization:**
- **Extractive**: Sentence ranking → Select top sentences
- **Abstractive**: Encode → Generate summary
- Metrics: ROUGE-1/2/L

**6. NL2Code:**
- Query → Schema pruning → Schema encoding → Code generation
- Metrics: CodeBLEU, Execution accuracy

**General Pipeline:**
```
Text → Preprocessing → Feature Extraction → Model → Output → Evaluation
```

**Key Points:**
- Start with simple baselines
- Use pre-trained models when possible
- Evaluate with task-specific metrics
- Handle domain-specific challenges

---

See `36_nlp_basics/nlp_tasks_and_solutions.md` for detailed procedures!

---

## MLE and MAP Estimation

### Q41: Derive MLE for a coin flip (Bernoulli distribution).

**Answer:**

**Setup:** n flips, k heads, model P(heads) = θ

**Likelihood:** L(θ) = θᵏ × (1-θ)ⁿ⁻ᵏ

**Log-likelihood:** log L(θ) = k log θ + (n-k) log(1-θ)

**Derivative:** d/dθ [log L(θ)] = k/θ - (n-k)/(1-θ)

**Set to zero:** k/θ = (n-k)/(1-θ) → θ = k/n

**Result:** θ̂_MLE = k/n (observed proportion)

**Intuition:** MLE is simply the proportion of heads observed!

---

### Q42: Derive MLE for linear regression.

**Answer:**

**Setup:** y = Xw + ε, where ε ~ N(0, σ²)

**Likelihood:** L(w) ∝ exp(-||y - Xw||²/(2σ²))

**Log-likelihood:** log L(w) = -||y - Xw||²/(2σ²) + constant

**Maximize:** argmax_w log L(w) = argmin_w ||y - Xw||²

**Derivative:** ∂/∂w [||y - Xw||²] = -2Xᵀ(y - Xw) = 0

**Result:** ŵ_MLE = (XᵀX)⁻¹Xᵀy (Ordinary Least Squares!)

**Key Insight:** MLE for linear regression with Gaussian noise = OLS!

---

### Q43: Explain the connection between MLE and MAP.

**Answer:**

**MLE:** θ̂_MLE = argmax_θ log P(D|θ)

**MAP:** θ̂_MAP = argmax_θ [log P(D|θ) + log P(θ)] = MLE + log(prior)

**Relationship:** MAP = MLE + Prior

**When same:**
- Uniform prior → MAP = MLE
- Large dataset → MAP ≈ MLE

**When different:**
- Small dataset → Prior has more influence
- Strong prior → MAP pulled toward prior mean

**Regularization:**
- L2 (Ridge) = MAP with Gaussian prior
- L1 (Lasso) = MAP with Laplace prior

---

### Q44: Derive MAP for linear regression with Gaussian prior (Ridge).

**Answer:**

**Setup:** y = Xw + ε, prior w ~ N(0, σ²_prior I)

**Posterior:** log P(w|D) = -||y - Xw||²/(2σ²) - ||w||²/(2σ²_prior)

**Derivative:** ∂/∂w [log P(w|D)] = -1/σ² × Xᵀ(y - Xw) - 1/σ²_prior × w = 0

**Result:** ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy where λ = σ²/σ²_prior

**Key Insight:** MAP with Gaussian prior = Ridge regression (L2 regularization)!

---

### Q45: Why do we use log-likelihood instead of likelihood?

**Answer:**

**Reasons:**
1. **Numerical stability**: Products of small probabilities → underflow, sums are stable
2. **Mathematical convenience**: Products become sums, derivatives easier
3. **Monotonicity**: Maximizing log L(θ) = maximizing L(θ)
4. **Additive properties**: Can combine log-likelihoods easily

**Example:**
- Likelihood: 0.1 × 0.1 × 0.1 = 0.001 (very small!)
- Log-likelihood: log(0.1) + log(0.1) + log(0.1) ≈ -6.91 (manageable)

---

### Q46: What's the difference between MLE and MAP in practice?

**Answer:**

**MLE:**
- Frequentist approach
- No prior (or uniform)
- Use: Large dataset, no prior knowledge
- Example: θ̂ = k/n

**MAP:**
- Bayesian approach
- Informative prior
- Use: Small dataset, have prior knowledge, need regularization
- Example: θ̂ = (k+α-1)/(n+α+β-2) with Beta prior

**Practical:**
- **Small data**: MLE can be extreme, MAP more reasonable
- **Regularization**: MAP provides natural regularization
- **Computation**: Similar complexity

---

See `37_mle_map_estimation/mle_map_derivations.md` for complete derivations!
See `37_mle_map_estimation/interview_qa.md` for more detailed answers!

---

## Multimodal Models and Embeddings

### Q47: Explain CLIP. How does it work?

**Answer:**

**CLIP** (Contrastive Language-Image Pre-training) learns to align text and images in a shared embedding space.

**Architecture:**
- **Image Encoder**: ViT or ResNet → image embeddings
- **Text Encoder**: Transformer → text embeddings
- **Contrastive Learning**: Align matching pairs

**Training:**
1. Collect 400M text-image pairs from web
2. Encode images and texts to same space
3. Contrastive loss: maximize similarity of matching pairs, minimize non-matching
4. Large batch size (32K) for many negatives

**Key Insight:**
Instead of predicting exact labels, predict which text matches which image.

**Zero-Shot Transfer:**
- Create text prompts: "a photo of a cat"
- Find most similar image
- Works on new tasks without fine-tuning!

**Results:**
- Matches supervised models on many tasks
- More robust to distribution shifts
- Strong image-text retrieval

---

### Q48: How do you train Word2Vec?

**Answer:**

**Word2Vec Skip-gram:**

**Architecture:**
- Input: One-hot vector for center word
- Hidden: Embedding layer (V × d)
- Output: Softmax over vocabulary (predict context words)

**Training:**
1. **Create pairs**: For each word, create (center, context) pairs from window
2. **Forward pass**: Embed center word, predict context words
3. **Loss**: -log P(context | center)
4. **Negative sampling**: Instead of softmax over all V words, sample k negatives
   - Binary classification: positive (context) vs negative
   - Much faster!

**Loss with Negative Sampling:**
```
Loss = -log σ(v_context · v_center) - Σ log σ(-v_neg · v_center)
```

**Training Details:**
- Data: Billions of words
- Window size: 5-10 words
- Embedding dim: 100-300
- Negative samples: 5-20
- Training: Hours to days

**Result:**
- Dense, low-dimensional embeddings
- Captures semantic relationships
- "King - Man + Woman ≈ Queen"

---

### Q49: How does GloVe differ from Word2Vec?

**Answer:**

**Word2Vec:**
- Uses local context (windows)
- Predicts context from center word
- Local statistics

**GloVe:**
- Uses global co-occurrence matrix
- Preserves co-occurrence ratios
- Global statistics

**GloVe Objective:**
```
w_i · w_j + b_i + b_j ≈ log(X_ij)

Where X_ij = co-occurrence count
```

**Training:**
1. Build co-occurrence matrix from entire corpus
2. Weighted least squares to preserve ratios
3. More efficient than Word2Vec

**Key Insight:**
Preserves ratios: P(solid|ice) / P(solid|steam) ≈ P(gas|ice) / P(gas|steam)

**Comparison:**
- **Word2Vec**: Local, window-based
- **GloVe**: Global, matrix-based
- **Performance**: Often similar, GloVe sometimes better

---

### Q50: Explain the evolution of NLP embeddings.

**Answer:**

**Timeline:**

**1. TF-IDF (1970s):**
- Statistical weighting
- Sparse, high-dimensional
- No semantic understanding

**2. N-grams (1980s-1990s):**
- Sequence modeling
- Count-based probabilities
- Local context only

**3. Word2Vec (2013):**
- Neural embeddings
- Dense, low-dimensional
- Semantic relationships
- Fixed embeddings (no context)

**4. GloVe (2014):**
- Global co-occurrence
- Matrix factorization
- Better than Word2Vec on some tasks

**5. Contextual Embeddings (2018+):**
- **ELMo**: Bidirectional LSTM
- **BERT**: Transformer, bidirectional
- **GPT**: Transformer, unidirectional
- Context-dependent embeddings

**6. Modern LLMs (2020+):**
- Large-scale language models
- Multimodal (CLIP, GPT-4V)
- Instruction tuning, RLHF

**Key Evolution:**
- Sparse → Dense
- Local → Global → Contextual
- Fixed → Context-dependent
- Single modality → Multimodal

---

### Q51: How do you evaluate multimodal models?

**Answer:**

**1. Zero-Shot Image Classification:**
- Create text prompts for classes
- Find most similar prompt
- Measure accuracy

**2. Image-Text Retrieval:**
- Text → Image: Find images matching query
- Image → Text: Find captions matching image
- Metrics: Recall@K, Median Rank

**3. Visual Question Answering:**
- Answer questions about images
- Requires reasoning
- Accuracy by question type

**4. Robustness:**
- Distribution shifts
- Adversarial examples
- Natural variations

**5. Bias Evaluation:**
- Gender, racial, cultural biases
- Fairness across groups

**6. Few-Shot Learning:**
- Learn from few examples
- Transfer learning capability

**Best Practices:**
- Multiple metrics
- Diverse datasets
- Human evaluation when possible
- Error analysis
- Bias testing

---

See `38_multimodal_and_embeddings/` for detailed explanations!

---

## RAG (Retrieval-Augmented Generation)

### Q52: Design a RAG system. What are the key components?

**Answer:**

**RAG Components:**

**1. Document Ingestion:**
- Load documents (PDF, DOCX, HTML, etc.)
- Extract text, metadata
- Preprocess and clean

**2. Chunking:**
- Split documents into chunks
- Strategies: fixed-size, sentence-based, semantic
- Overlap between chunks (10-20%)

**3. Embedding Generation:**
- Generate embeddings for chunks
- Use embedding models (sentence-transformers, OpenAI)
- Store in vector database

**4. Query Processing:**
- Process user query
- Generate query embedding
- Query expansion/rewriting

**5. Retrieval:**
- Vector similarity search
- Hybrid search (dense + sparse)
- Metadata filtering
- Top-K retrieval

**6. Re-ranking (Optional):**
- Cross-encoder for accuracy
- Re-rank top results
- Better precision

**7. Context Assembly:**
- Select top-K chunks
- Order by relevance
- Fit in context window

**8. Generation:**
- LLM with context
- Prompt engineering
- Generate answer

**9. Post-processing:**
- Extract answer
- Generate citations
- Validate answer

**Pipeline:**
```
Query → Embedding → Retrieval → Re-ranking → Context → Generation → Answer
```

---

### Q53: How do you improve RAG retrieval accuracy?

**Answer:**

**1. Better Chunking:**
- Semantic chunking (respect boundaries)
- Hierarchical chunking
- Multi-granularity chunks
- Overlapping chunks

**2. Better Embeddings:**
- Domain fine-tuning
- Hybrid embeddings (dense + sparse)
- Multi-vector embeddings
- Query-specific embeddings

**3. Hybrid Search:**
- Dense retrieval (semantic)
- Sparse retrieval (BM25, keywords)
- Weighted combination
- Better coverage

**4. Re-ranking:**
- Cross-encoder (more accurate)
- Learning-to-rank
- Multi-stage retrieval
- Better precision

**5. Query Expansion:**
- Synonym expansion
- Related terms
- Query rewriting
- Multi-query generation

**6. Metadata Filtering:**
- Filter by document type
- Filter by date, source
- Improve precision

**7. Multi-Stage Retrieval:**
- Stage 1: Coarse (ANN, top-100)
- Stage 2: Re-rank (top-10)
- Stage 3: Fine-grained (top-5)

---

### Q54: How do you handle context window limits in RAG?

**Answer:**

**1. Priority-Based Selection:**
- Sort by relevance score
- Take top-K until context full
- Truncate if needed

**2. Summarization:**
- Summarize chunks that don't fit
- Hierarchical summarization
- Preserve key information

**3. Chunk Merging:**
- Merge related chunks
- Remove redundancy
- Create coherent context

**4. Dynamic Context:**
- Adaptive chunk selection
- Iterative retrieval
- Expand if needed

**5. Long-Context Models:**
- Use models with larger context (32K, 100K+)
- More expensive but better
- Less truncation

**Best Practice:**
- Prioritize by relevance
- Summarize overflow
- Use appropriate context size

---

### Q55: How do you prevent hallucination in RAG?

**Answer:**

**1. Prompt Engineering:**
```
"Answer ONLY based on the provided context.
If the answer is not in the context, say 'I don't know'."
```

**2. Answer Validation:**
- Check if answer supported by context
- Extract supporting sentences
- Confidence scoring

**3. Citation Generation:**
- Link answer to source chunks
- Show supporting evidence
- Enable fact-checking

**4. Confidence Scoring:**
- Model confidence in answer
- Retrieval confidence
- Combined confidence score

**5. Answer Extraction:**
- Extract answer from context
- Don't generate new information
- Use extractive QA models

**6. Post-Processing:**
- Validate answer against context
- Check for contradictions
- Flag uncertain answers

---

### Q56: How do you evaluate a RAG system?

**Answer:**

**Retrieval Metrics:**
- **Precision@K**: Precision of top-K
- **Recall@K**: Recall of top-K
- **MRR**: Mean reciprocal rank
- **MAP**: Mean average precision
- **NDCG@K**: Normalized discounted cumulative gain

**Generation Metrics:**
- **BLEU**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Semantic similarity
- **Answer accuracy**: Correctness

**End-to-End Metrics:**
- **Answer relevance**: Is answer relevant?
- **Answer correctness**: Is answer correct?
- **Answer completeness**: Is answer complete?
- **Citation quality**: Are citations correct?

**Best Practices:**
- Use multiple metrics
- Combine automated + human evaluation
- Monitor in production
- Task-specific evaluation

---

### Q57: What are common RAG challenges and solutions?

**Answer:**

**1. Chunking Strategy:**
- **Challenge**: How to split documents
- **Solution**: Semantic chunking, hierarchical, overlap

**2. Embedding Quality:**
- **Challenge**: Domain-specific semantics
- **Solution**: Fine-tuning, hybrid embeddings

**3. Retrieval Accuracy:**
- **Challenge**: Retrieved chunks not relevant
- **Solution**: Multi-stage retrieval, re-ranking, hybrid search

**4. Context Window Limits:**
- **Challenge**: Too many chunks, can't fit
- **Solution**: Priority selection, summarization, long-context models

**5. Hallucination:**
- **Challenge**: Model generates wrong info
- **Solution**: Prompt engineering, citations, validation

**6. Scalability:**
- **Challenge**: Large document sets
- **Solution**: ANN search, distributed systems, caching

**7. Cost:**
- **Challenge**: High API costs
- **Solution**: Self-hosted models, caching, batch processing

---

See `39_rag_retrieval_augmented_generation/` for detailed implementations!

---

### Q58: Explain different chunking strategies. When to use each?

**Answer:**

**1. Fixed-Size Chunking:**
- Split into fixed-size chunks (e.g., 512 chars)
- Overlap between chunks (10-20%)
- **Use**: Simple documents, prototyping, uniform content
- **Pros**: Simple, fast
- **Cons**: Breaks sentences, no semantic awareness

**2. Sentence-Based Chunking:**
- Split on sentence boundaries
- Group sentences into chunks
- **Use**: Narrative text, general documents, production (common default)
- **Pros**: Respects boundaries, better coherence
- **Cons**: Sentence splitting can be imperfect

**3. Paragraph-Based Chunking:**
- Split on paragraph boundaries
- **Use**: Structured documents, academic papers, long-form
- **Pros**: Preserves structure, natural units
- **Cons**: Variable sizes, may be too large/small

**4. Semantic Chunking:**
- Use embeddings to find semantic boundaries
- Split when semantic shift detected
- **Use**: High accuracy needs, topic-based documents
- **Pros**: Best semantic coherence, optimal retrieval
- **Cons**: Slower, requires embeddings, higher cost

**5. Recursive Chunking:**
- Hierarchical splitting (paragraphs → sentences → words)
- **Use**: General-purpose, variable structure, production (LangChain default)
- **Pros**: Robust, handles any structure
- **Cons**: More complex, can be slow

**6. Sliding Window:**
- Fixed-size with stride (overlap)
- **Use**: Sequential data, code, long documents
- **Pros**: Preserves context, good for sequential
- **Cons**: Many chunks, redundancy

**7. Token-Based:**
- Split by token count (not characters)
- **Use**: LLM systems, accurate sizing, cost optimization
- **Pros**: Accurate for LLMs, precise control
- **Cons**: Requires tokenizer, model-specific

**8. Hierarchical:**
- Multi-level (document → section → paragraph)
- **Use**: Complex documents, academic papers
- **Pros**: Preserves structure, multi-level retrieval
- **Cons**: Complex, more storage

**9. Content-Aware:**
- Different strategy per content type (code, tables, text)
- **Use**: Mixed content, technical docs, research papers
- **Pros**: Optimal per type, handles complexity
- **Cons**: Very complex, requires detection

**10. Metadata-Enriched:**
- Chunks with rich metadata (section, page, etc.)
- **Use**: Structured docs, citations, filtering
- **Pros**: Rich context, better filtering
- **Cons**: More storage, complex

**Best Practices:**
- **Start**: Sentence-based or recursive
- **Upgrade**: Semantic if accuracy critical
- **Overlap**: 10-20% of chunk size
- **Size**: 256-1024 tokens (512 common)
- **Test**: Evaluate retrieval accuracy

---

See `39_rag_retrieval_augmented_generation/chunking_strategies.md` for complete guide!

---

## Linear and Logistic Regression Derivations

### Q59: Derive linear regression from first principles. Explain intuitively.

**Answer:**

**Goal:** Find line y = wx + b that best fits data

**Step 1: Define Error**
- For each point: error = yᵢ - (wxᵢ + b)
- Actual value minus predicted value

**Step 2: Cost Function (MSE)**
- Sum of squared errors: MSE = (1/n) Σ (yᵢ - wxᵢ - b)²
- Why squared? Always positive, penalizes large errors, differentiable

**Step 3: Minimize**
- Take derivatives, set to zero
- ∂MSE/∂b = 0 → b = ȳ - wx̄ (line passes through center!)
- ∂MSE/∂w = 0 → w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

**Final Solution:**
```
w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²  (covariance / variance)
b = ȳ - wx̄
```

**Intuition:**
- Slope = how much y changes per unit x (covariance/variance)
- Intercept = mean y - slope × mean x
- Line passes through center of data (x̄, ȳ)

**Matrix Form:**
```
w = (XᵀX)⁻¹Xᵀy  (normal equation)
```

**Why it works:**
- Minimizes sum of squared errors (geometrically optimal)
- Projects y onto column space of X
- Unique solution (if data not degenerate)

---

### Q60: Derive logistic regression. Why sigmoid function?

**Answer:**

**Problem:** Need probabilities (0 to 1), not continuous values

**Step 1: Log Odds**
- Odds = P / (1-P) (can be any positive number)
- Log odds = log(P / (1-P)) (can be any real number)
- Model: log(P / (1-P)) = wx + b (linear!)

**Step 2: Solve for P**
- P / (1-P) = e^(wx + b)
- P = e^(wx + b) / (1 + e^(wx + b))
- P = 1 / (1 + e^(-(wx + b))) = σ(wx + b)

**Why Sigmoid?**
- Bounded: Always between 0 and 1
- Smooth: Differentiable everywhere
- S-shaped: Good for probabilities
- When wx + b → -∞: P → 0
- When wx + b = 0: P = 0.5
- When wx + b → +∞: P → 1

**Step 3: Likelihood**
- P(y|x) = σ(wx + b)^y × (1 - σ(wx + b))^(1-y)
- Likelihood = ∏ᵢ P(yᵢ|xᵢ)

**Step 4: Cost Function (Cross-Entropy)**
- Maximize likelihood = Minimize negative log-likelihood
- J(w, b) = -Σ [y log σ(wx + b) + (1-y) log(1 - σ(wx + b))]

**Step 5: Gradient**
- ∂J/∂w = Σ [σ(wx + b) - y] × x
- ∂J/∂b = Σ [σ(wx + b) - y]
- Error = predicted - actual

**Update:**
```
w = w - α × Σ [σ(wx + b) - y] × x
b = b - α × Σ [σ(wx + b) - y]
```

**Decision Boundary:**
- P(y=1|x) = 0.5 when wx + b = 0
- Line wx + b = 0 separates classes

**Why Cross-Entropy?**
- Optimal for classification (information theory)
- Convex (guaranteed global minimum)
- Works well with probabilities

---

### Q61: Why can't we use linear regression for classification?

**Answer:**

**Problems:**

**1. Output Range:**
- Linear regression: Output can be any real number (-∞, +∞)
- Classification: Need probabilities [0, 1]
- Linear regression can give negative values or > 1

**2. Interpretation:**
- Linear regression output doesn't represent probability
- Can't interpret as "80% chance of class 1"

**3. Loss Function:**
- MSE not optimal for classification
- Doesn't penalize misclassifications appropriately
- Can get stuck in local minima

**4. Decision Boundary:**
- Linear regression: Threshold at 0.5 (arbitrary)
- No probabilistic interpretation
- Doesn't work well for imbalanced classes

**Solution:**
- Use logistic regression (sigmoid)
- Output is probability [0, 1]
- Cross-entropy loss (optimal for classification)
- Probabilistic interpretation

**When Linear Regression Works for Classification:**
- Binary classification with balanced classes
- When you just need a threshold
- But logistic regression is almost always better

---

### Q62: Explain the relationship between linear and logistic regression.

**Answer:**

**Similarities:**
- Both use linear combination: wx + b
- Both learn weights w and bias b
- Both use gradient descent (or closed-form for linear)

**Differences:**

**Linear Regression:**
- Output: Continuous values (-∞, +∞)
- Model: y = wx + b
- Cost: MSE (sum of squared errors)
- Solution: Closed-form (normal equation) or gradient descent

**Logistic Regression:**
- Output: Probabilities [0, 1]
- Model: P(y=1|x) = σ(wx + b)
- Cost: Cross-entropy (negative log-likelihood)
- Solution: Gradient descent (no closed-form)

**Key Insight:**
- Logistic regression = Linear regression + sigmoid
- Log odds are linear: log(P/(1-P)) = wx + b
- Probabilities are sigmoid: P = σ(wx + b)

**Visual:**
```
Linear:     y = wx + b  (straight line)
            ↓
Logistic:   P = σ(wx + b)  (sigmoid curve)
```

**Connection:**
- Both model linear relationships
- Linear: Direct relationship
- Logistic: Linear in log-odds space

---

See `01_classical_ml/linear_regression_derivation.md` and `01_classical_ml/logistic_regression_derivation.md` for complete derivations!

---

## RAG Retrieval Methods

### Q63: Explain BM25. How does it differ from TF-IDF?

**Answer:**

**BM25** (Best Matching 25) is industry-standard sparse retrieval, improving upon TF-IDF.

**BM25 Formula:**
```
BM25(t, d) = IDF(t) × (f(t, d) × (k₁ + 1)) / (f(t, d) + k₁ × (1 - b + b × |d|/avgdl))
```

**Key Improvements:**

**1. Term Frequency Saturation:**
- TF-IDF: Linear (10x → score = 10, 20x → score = 20)
- BM25: Saturates (10x → 8.5, 20x → 9.2)
- Prevents one term from dominating

**2. Document Length Normalization:**
- Normalizes by document length
- Prevents bias toward long documents

**3. Better IDF:**
- BM25 IDF: log((N - df + 0.5) / (df + 0.5))
- More robust

**Use Cases:**
- Keyword-based search
- Production systems (Elasticsearch, Lucene)
- Better than TF-IDF in most cases

---

### Q64: Explain hybrid search in RAG. How do you combine sparse and dense?

**Answer:**

**Hybrid Search** combines BM25 (sparse) + Dense (embeddings).

**Why:**
- BM25: Exact matches, keywords
- Dense: Semantic similarity
- Neither perfect alone → Combine!

**How:**
1. Retrieve from both (top-K each)
2. Normalize scores to [0, 1]
3. Combine: Final = α × BM25 + (1-α) × Dense
4. Re-rank by combined score

**Weight Selection:**
- α = 0.7: More BM25 (keyword-heavy)
- α = 0.5: Balanced (default)
- α = 0.3: More dense (semantic)

**Best Practice:**
- Normalize before combining
- Tune α on validation set
- Use for production systems

---

### Q65: When to use BM25 vs Dense vs Hybrid?

**Answer:**

**BM25:**
- Keyword queries, exact matching
- Fast, interpretable
- Start here

**Dense:**
- Semantic queries, synonyms
- Related concepts
- When embeddings available

**Hybrid:**
- Production systems
- Mixed query types
- Best overall performance
- Industry standard

**Recommendation:**
- Start: BM25
- Add: Dense if semantic needed
- Production: Hybrid

---

See `39_rag_retrieval_augmented_generation/retrieval_methods.md` for detailed explanations!

---

## NLP Problems: Standard Solution Procedures

### Q66: What's the standard procedure for text classification?

**Answer:**

**Phase 1: Data Preparation**
- Collect labeled data, handle class imbalance
- Preprocessing: Lowercase, remove special chars, tokenize
- Split: Train/Validation/Test

**Phase 2: Feature Extraction**
- **Small data**: TF-IDF + Naive Bayes/SVM
- **Medium data**: Word embeddings + LSTM/CNN
- **Large data**: Fine-tuned BERT

**Phase 3: Model Selection**
- < 10K: TF-IDF + SVM
- 10K-100K: Embeddings + Neural or XGBoost
- > 100K: Fine-tuned BERT

**Phase 4: Training**
- Traditional ML: Hyperparameter tuning
- Neural: Adam optimizer, dropout, early stopping
- BERT: Learning rate 2e-5, 3-5 epochs

**Phase 5: Evaluation**
- Metrics: Accuracy, F1, Precision, Recall
- Multi-class: Macro/Micro F1
- Confusion matrix for analysis

**Phase 6: Deployment**
- API endpoint
- Monitoring, drift detection
- A/B testing

---

### Q67: How do you solve NER? What's the standard approach?

**Answer:**

**Phase 1: Data Format**
- BIO tagging: B-PER, I-PER, O
- Label each token

**Phase 2: Features**
- Word features: Current, previous, next word
- Context: Surrounding words, position
- Embeddings: Word + character-level

**Phase 3: Model**
- **CRF**: Traditional, interpretable
- **BiLSTM-CRF**: Better performance
- **Fine-tuned BERT**: State-of-the-art

**Phase 4: Training**
- CRF: Maximum likelihood, L-BFGS
- BiLSTM-CRF: Adam, dropout 0.5
- BERT: Learning rate 3e-5, token classification

**Phase 5: Evaluation**
- Entity-level F1 (exact match)
- Token-level F1
- Per entity type

**Phase 6: Challenges**
- OOV words: Character embeddings, subword
- Nested entities: Multi-label, span-based
- Ambiguity: Context, larger window

---

### Q68: What's the standard procedure for question answering?

**Answer:**

**Phase 1: Data Format**
- SQuAD: Context + Question → Answer span
- Extractive: Start/end positions

**Phase 2: Model**
- **BERT-based**: Standard approach
  - Input: [CLS] question [SEP] context [SEP]
  - Two heads: Start position, End position
  - Fine-tune BERT

**Phase 3: Training**
- Load pre-trained BERT
- Add QA head (start/end logits)
- Loss: Start loss + End loss
- Learning rate: 3e-5, batch 16-32, 2-3 epochs

**Phase 4: Long Contexts**
- Sliding window
- Hierarchical (paragraph ranking)
- Long-context models

**Phase 5: Evaluation**
- EM (Exact Match)
- F1 (token overlap)
- Per question type

**Phase 6: Production**
- Retrieval for open-domain
- Re-ranking
- Ensemble models

---

### Q69: How do you build a machine translation system?

**Answer:**

**Phase 1: Data**
- Parallel corpus (millions of pairs)
- High-quality translations
- Domain match if possible

**Phase 2: Preprocessing**
- Sentence segmentation
- Subword tokenization (BPE, SentencePiece)
- Handle rare words

**Phase 3: Model**
- **Transformer**: State-of-the-art
- Encoder-Decoder architecture
- Multi-head attention

**Phase 4: Training**
- Pre-train on large corpus (optional)
- Fine-tune on translation data
- Learning rate: 1e-4, warmup
- Decoding: Beam search with length penalty

**Phase 5: Evaluation**
- BLEU score (primary)
- METEOR, human evaluation

**Phase 6: Production**
- Multilingual models
- Transfer learning for low-resource
- Back-translation for data augmentation

---

### Q70: What's the standard approach for text summarization?

**Answer:**

**Two Types:**

**Extractive:**
1. Score sentences (position, TF-IDF, similarity)
2. Select top-K sentences
3. Order by original position
4. Methods: TextRank, BERT-based scoring

**Abstractive:**
1. **Model**: Fine-tuned BART/T5
2. **Training**: Encoder-Decoder, max source 1024, target 128
3. **Generation**: Beam search, length penalty, repetition penalty
4. **Post-processing**: Remove repetition, fix grammar

**Evaluation:**
- ROUGE-1/2/L (primary)
- BLEU, human evaluation

**Challenges:**
- Long documents: Hierarchical encoding
- Factual consistency: Fact checking
- Repetition: Repetition penalty

---

See `36_nlp_basics/nlp_problems_detailed.md` for complete procedures for all NLP problems!

---

## Foundation Models: Evolution from BERT to GPT-4

### Q71: How did we evolve from BERT to modern foundation models like GPT-4?

**Answer:**

**Phase 1: BERT (2018)**
- **Bidirectional**: Reads text both directions
- **Encoder-only**: Good for understanding
- **Pre-training + Fine-tuning**: Train on large corpus, fine-tune on tasks
- **Limitation**: Can't generate text, needs fine-tuning per task

**Phase 2: GPT-2 (2019)**
- **Generative**: Can generate coherent text
- **Decoder-only**: Autoregressive (left-to-right)
- **Zero-shot**: No fine-tuning for some tasks
- **Limitation**: Unidirectional, limited context

**Phase 3: GPT-3 (2020)**
- **Massive scale**: 175B parameters
- **In-context learning**: Few-shot without gradient updates
- **Scaling laws**: Performance ∝ (Size)^α × (Data)^β
- **Emergent abilities**: Arithmetic, code, reasoning at scale
- **Impact**: Proved scaling works, foundation for modern LLMs

**Phase 4: InstructGPT (2021)**
- **RLHF**: Reinforcement Learning from Human Feedback
- **Alignment**: Align model with human preferences
- **Instruction following**: Better at following instructions
- **Impact**: Foundation for ChatGPT, RLHF becomes standard

**Phase 5: ChatGPT (2022)**
- **Conversational**: Natural dialogue interface
- **RLHF**: Aligned with human preferences
- **Multi-turn**: Maintains context
- **Impact**: Viral adoption, paradigm shift to assistants

**Phase 6: GPT-4 (2023)**
- **Multimodal**: Text + images
- **Better reasoning**: Improved logical reasoning
- **Large context**: 8K → 32K → 128K tokens
- **State-of-the-art**: Best performance on many tasks

**Key Paradigm Shifts:**
1. Task-specific → General (single model for many tasks)
2. Fine-tuning → Prompting (in-context learning)
3. Understanding → Generation (decoder architectures)
4. Supervised → Self-supervised (pre-training)
5. Capability → Alignment (RLHF, safety)

---

### Q72: What are scaling laws? How do they explain the success of large models?

**Answer:**

**Neural Scaling Laws:**
```
Performance = f(Model Size, Data Size, Compute)

Performance improves predictably with:
- Model size (parameters)
- Training data size
- Compute budget
```

**Key Findings:**

**1. Power Law Relationship:**
- Performance ∝ (Model Size)^α
- α ≈ 0.076 (diminishing returns but still improves)
- Larger models = better performance

**2. Data Scaling:**
- Larger models need more data
- Optimal data size ∝ Model size
- More data = better performance

**3. Compute Scaling:**
- More compute = better performance
- Optimal allocation: Scale model, data, compute together

**4. Predictable Improvements:**
- Can predict performance before training
- Helps with planning and resource allocation

**Implications:**
- **Bigger is better**: Larger models perform better
- **Data matters**: Need more data for larger models
- **Massive compute**: Requires huge compute budgets
- **Predictable**: Can estimate performance

**Why It Matters:**
- Explains why GPT-3 succeeded
- Guides model development
- Shows path to better models
- Justifies investment in scale

---

### Q73: What is in-context learning? How does it differ from fine-tuning?

**Answer:**

**In-Context Learning:**
- Model learns from examples in the prompt
- No gradient updates
- No weight changes
- Same model for all tasks

**Types:**

**Zero-shot:**
```
"Translate to French: hello →"
Model generates: "bonjour"
```

**One-shot:**
```
"Translate to French: hello → bonjour, cat →"
Model generates: "chat"
```

**Few-shot:**
```
"Translate to French: hello → bonjour, cat → chat, dog →"
Model generates: "chien"
```

**Fine-tuning:**
- Update model weights
- Task-specific model
- Requires labeled data
- Gradient updates
- Different model per task

**Key Differences:**

| Aspect | In-Context Learning | Fine-tuning |
|--------|---------------------|-------------|
| **Weight Updates** | No | Yes |
| **Data** | Examples in prompt | Labeled dataset |
| **Model** | Same for all tasks | Different per task |
| **Flexibility** | Easy to change | Need retraining |
| **Performance** | Good for many tasks | Best for specific task |

**Why In-Context Learning Works:**
- Large models have seen similar patterns
- Can generalize from examples
- Emergent ability at scale
- Flexible and efficient

**When to Use:**
- **In-context**: Quick prototyping, many tasks, no labeled data
- **Fine-tuning**: Best performance, specific task, have labeled data

---

### Q74: Explain RLHF (Reinforcement Learning from Human Feedback). Why is it important?

**Answer:**

**RLHF** aligns language models with human preferences using reinforcement learning.

**Three Steps:**

**Step 1: Supervised Fine-tuning (SFT)**
```
1. Collect human-written prompts and responses
2. Fine-tune base model (GPT-3) on this data
3. Model learns to follow instructions
```

**Step 2: Reward Modeling**
```
1. Collect comparisons: Which response is better?
2. Train reward model to predict human preferences
3. Reward model scores: response_A > response_B
```

**Step 3: Reinforcement Learning (PPO)**
```
1. Generate responses from SFT model
2. Score with reward model
3. Update model to maximize reward
4. Use PPO (Proximal Policy Optimization)
```

**Why Important:**

**1. Alignment:**
- Model behavior ≠ model capability
- Need to align with human values
- Helpful, harmless, honest

**2. Better User Experience:**
- Follows instructions better
- More helpful responses
- Admits mistakes
- Refuses harmful requests

**3. Safety:**
- Can make models safer
- Reduces harmful outputs
- Better control

**Impact:**
- Foundation for ChatGPT
- Standard practice for alignment
- Drives research in alignment
- Better user experience

**Challenges:**
- Expensive (human feedback)
- Subjective (different preferences)
- Can be gamed (reward hacking)
- Ongoing research

---

### Q75: What are emergent abilities? Give examples.

**Answer:**

**Emergent Abilities:**
- Abilities that appear only at large scale
- Not present in smaller models
- Unexpected capabilities
- Emerge from scale, not explicit training

**Examples:**

**1. Arithmetic:**
- Small models: Can't do math
- Large models: Can do arithmetic (not explicitly trained)
- Example: "What is 123 × 456?" → Correct answer

**2. Code Generation:**
- Small models: Can't write code
- Large models: Can generate working code
- Example: "Write Python function to sort list" → Working code

**3. Few-shot Learning:**
- Small models: Need many examples
- Large models: Learn from few examples
- Example: 1-2 examples sufficient

**4. Reasoning:**
- Small models: Limited reasoning
- Large models: Some logical reasoning
- Example: Multi-step problem solving

**5. Instruction Following:**
- Small models: Don't follow instructions well
- Large models: Better at following instructions
- Example: Complex multi-step instructions

**Why Important:**
- Shows scale matters
- Unexpected capabilities
- Hard to predict what will emerge
- Justifies investment in scale

**Implications:**
- Can't predict all capabilities
- Need to test large models
- Emergent abilities are powerful
- Safety concerns (unexpected behaviors)

---

### Q76: How do modern foundation models differ from BERT?

**Answer:**

**Architecture:**

**BERT:**
- Encoder-only
- Bidirectional
- Good for understanding
- Can't generate

**Modern Foundation Models (GPT-4, etc.):**
- Decoder-only (or encoder-decoder)
- Unidirectional (for generation)
- Good for generation
- Can do understanding (with prompting)

**Training:**

**BERT:**
- Masked language modeling
- Next sentence prediction
- ~3B tokens
- Task-specific fine-tuning

**Modern:**
- Next token prediction
- Trillions of tokens
- RLHF for alignment
- In-context learning

**Capabilities:**

**BERT:**
- Understanding tasks
- Classification, NER, QA
- Needs fine-tuning per task
- Task-specific models

**Modern:**
- Generation + understanding
- Many tasks with one model
- In-context learning
- General-purpose

**Scale:**

**BERT:**
- 110M-340M parameters
- Small datasets
- Moderate compute

**Modern:**
- 175B+ parameters
- Trillions of tokens
- Massive compute

**Usage:**

**BERT:**
- Fine-tune for specific task
- Different model per task
- Requires labeled data

**Modern:**
- Prompt with examples
- Same model for all tasks
- No labeled data needed (few-shot)

**Key Differences Summary:**

| Aspect | BERT | Modern Foundation Models |
|--------|------|------------------------|
| **Architecture** | Encoder | Decoder |
| **Direction** | Bidirectional | Unidirectional |
| **Primary Use** | Understanding | Generation |
| **Scale** | 110M-340M | 175B+ |
| **Training** | MLM + NSP | Next token + RLHF |
| **Usage** | Fine-tuning | In-context learning |
| **Tasks** | Task-specific | General-purpose |

---

See `38_multimodal_and_embeddings/foundation_models_evolution.md` for complete evolution story!

---

## Multimodal Integration and World Models

### Q77: How do you integrate triplet data (knowledge graphs) into foundation models?

**Answer:**

**Triplet Data:**
- Format: (Subject, Relation, Object)
- Example: ("Einstein", "born_in", "Germany")
- Represents structured knowledge

**Integration Strategies:**

**1. Direct Encoding:**
- Convert triplet to text: "Einstein [born_in] Germany"
- Add to training corpus
- Model learns relationships

**2. Knowledge Graph Embedding:**
- Pre-train embeddings (TransE, TransR)
- Learn entity and relation embeddings
- Integrate into language model

**3. Structured Prompting:**
```
"Given: (Einstein, born_in, Germany)
Question: Where was Einstein born?
Answer: Germany"
```

**4. Multi-Task Learning:**
- Language modeling + triplet prediction
- Joint training on all tasks

**Processing Pipeline:**
1. Data collection (Wikidata, Freebase)
2. Data cleaning (remove duplicates, validate)
3. Format conversion (triplet → text)
4. Integration (mixed corpus or separate objective)

**Best Practice:**
- Mix 20% triplet-derived text with 80% natural text
- Use knowledge graph embeddings for better reasoning
- Fine-tune on domain-specific triplets

---

### Q78: How do you integrate past conversation history into LLMs?

**Answer:**

**Challenges:**
- Limited context window (2K-32K tokens)
- Need long-term memory
- User personalization

**Integration Strategies:**

**1. Context Window Extension:**
- Store history in external memory
- Retrieve relevant history
- Concatenate to context

**2. Memory-Augmented Models:**
- Main model: Processes current input
- Memory bank: Stores conversation history
- Attention: Attend to relevant history

**3. Hierarchical Encoding:**
- Level 1: Individual messages
- Level 2: Conversation turns
- Level 3: Conversation sessions
- Level 4: User profile

**4. RAG for History:**
- Store conversations in vector DB
- Retrieve relevant history
- Add to context

**Processing:**
1. Data collection (chat logs, transcripts)
2. Data cleaning (remove PII, anonymize)
3. History segmentation (turns, sessions)
4. Feature extraction (sentiment, intent, entities)
5. Integration (conversation modeling or user embedding)

**Best Practice:**
- Keep last 10-20 turns in context
- Use retrieval for older history
- Learn user embeddings for personalization

---

### Q79: What is a world model? How do you build one for LLMs?

**Answer:**

**World Model:**
- Internal representation of how world works
- Predicts future states
- Understands cause and effect
- Enables planning and reasoning

**Key Components:**

**1. State Representation:**
- Entities and properties
- Relationships
- Temporal information
- Methods: Symbolic, embedding, graph

**2. Transition Model:**
- Predicts next state given current state and action
- Types: Deterministic, stochastic, learned
- Training: Neural network on state-action-next_state tuples

**3. Observation Model:**
- Maps world state to observations
- Handles partial observability
- Models what we can observe

**4. Reward Model:**
- Defines what's good/bad
- Guides learning
- Types: Task-specific, shaped, learned

**5. Planning:**
- Use world model to find good actions
- Methods: Model-based RL, tree search, MPC

**Integration with LLMs:**
```
LLM → World Model Interface → World Model → Planning → Actions
```

**Training:**
1. Learn world model from data
2. Integrate with LLM
3. Joint training end-to-end

---

### Q80: What are the future directions of LLMs? What's the path to AGI?

**Answer:**

**Future Directions:**

**1. General Intelligence:**
- Human-level intelligence
- General problem solving
- Transfer learning
- Few-shot adaptation

**2. World Understanding:**
- Understand how world works
- Predict consequences
- Plan actions
- Reason about causality

**3. Continual Learning:**
- Learn from new data continuously
- Don't forget old knowledge
- Adapt to new domains

**4. Embodied Intelligence:**
- Interact with physical world
- Learn from experience
- Understand physics

**Key Research Areas:**

**Scaling:**
- Efficient scaling
- Better architectures
- Sparse models

**Multimodality:**
- All modalities
- Unified representation

**Reasoning:**
- Strong logical reasoning
- Causal reasoning
- Mathematical reasoning

**Planning:**
- Long-term planning
- Hierarchical planning

**Memory:**
- Long-term memory
- Episodic memory
- Semantic memory

**AGI Architecture Vision:**
```
1. Perception Module (multimodal input)
2. World Model (state, transition, planning)
3. Memory System (episodic, semantic, working)
4. Reasoning Engine (logical, causal, analogical)
5. Action Module (text, tools, physical)
6. Learning System (continual, meta-learning)
```

**Path to AGI:**
- Multimodal integration
- World models
- Strong reasoning
- Long-term planning
- Long-term memory
- Continual learning

---

See `38_multimodal_and_embeddings/multimodal_integration_and_world_models.md` for complete details!

---

## GPT Implementation, Training, and Decoding

### Q81: Implement a complete GPT model from scratch. What are all the components?

**Answer:**

**Complete GPT consists of:**

**1. Token Embedding:**
- Converts token indices to dense vectors
- Learned embeddings, shape (vocab_size, d_model)

**2. Positional Encoding:**
- Adds position information to embeddings
- Sinusoidal or learned positional embeddings
- Shape: (max_seq_len, d_model)

**3. Multi-Head Attention:**
- Self-attention mechanism
- Multiple heads (typically 12-96)
- Each head: Q, K, V projections → attention → concatenate
- Complexity: O(n²d)

**4. Feed-Forward Network:**
- Two linear layers with ReLU
- Expands then contracts: d_model → d_ff → d_model
- Applied position-wise

**5. Transformer Block:**
- Multi-head attention + Feed-forward
- Residual connections + Layer normalization
- Dropout for regularization

**6. Stack of Transformer Blocks:**
- Multiple blocks (typically 12-96 layers)
- Each block refines representations

**7. Final Layer Norm:**
- Normalizes final representations

**8. Output Projection:**
- Maps to vocabulary size
- Produces logits for next token prediction

**See `04_transformers/gpt_complete.py` for complete implementation!**

---

### Q82: How is GPT trained? Explain the training process in detail.

**Answer:**

**Training Objective:**
- Next token prediction (language modeling)
- Given tokens [t₁, t₂, ..., tₙ], predict [t₂, t₃, ..., tₙ₊₁]
- Autoregressive: each token depends on all previous tokens

**Training Process:**

**1. Data Preparation:**
- Large text corpora (books, web, articles)
- Tokenization (BPE, SentencePiece)
- Batching and padding to fixed length

**2. Forward Pass:**
- Token embeddings + positional encoding
- Pass through transformer blocks
- Output logits for each position

**3. Loss Function:**
- Cross-entropy loss
- L = -(1/n) Σ log P(tᵢ | t₁, ..., tᵢ₋₁)
- Compares predicted distribution to true next token

**4. Backward Pass:**
- Compute gradients via backpropagation
- Gradient clipping (max_norm=1.0)
- Update parameters with optimizer (Adam/AdamW)

**5. Training Details:**
- Learning rate: 3e-4 to 1e-4
- Learning rate scheduling (warmup + decay)
- Dropout for regularization
- Weight initialization (normal, std=0.02)

**Key Insight:**
- Massive scale: GPT-3 trained on trillions of tokens
- Self-supervised: no labels needed, just text
- Learns language patterns, syntax, semantics, reasoning

**See `04_transformers/gpt_training_decoding.md` for complete details!**

---

### Q83: How does GPT decode/generate text? Explain the decoding process.

**Answer:**

**Autoregressive Generation:**
- Generate one token at a time
- Each token depends on all previous tokens
- Start with prompt, generate until stop condition

**Decoding Process:**

**1. Initial Prompt:**
- User provides starting text
- Tokenized into sequence [p₁, p₂, ..., pₖ]

**2. Forward Pass:**
- Process prompt through model
- Get logits for next token
- Shape: (vocab_size,) - scores for each token

**3. Convert to Probabilities:**
- Apply softmax: P(t) = exp(logit_t) / Σ exp(logit_i)
- Temperature scaling: P(t) = softmax(logits / T)
  - T=1.0: original distribution
  - T>1.0: more random (higher diversity)
  - T<1.0: more deterministic (lower diversity)

**4. Sample Token:**
- **Greedy**: Always pick highest probability
- **Sampling**: Random sample from distribution
- **Top-k**: Sample from top-k tokens
- **Top-p (Nucleus)**: Sample from tokens with cumulative prob > p

**5. Append and Repeat:**
- Append sampled token to sequence
- Process new sequence again
- Repeat until stop condition

**6. Stop Conditions:**
- Maximum length reached
- End-of-sequence token generated
- Specific stop sequence

**Causal Masking:**
- During decoding, mask future positions
- Upper triangular matrix: -inf for future, 0 for past
- Ensures model only sees previous tokens

**See `04_transformers/gpt_training_decoding.md` for complete details!**

---

### Q84: What is the complexity of attention? Explain O(n²d) and different attention types.

**Answer:**

**Standard Self-Attention: O(n²d)**

**Why O(n²d)?**
- Compute QK^T: (n, d) @ (d, n) → (n, n) matrix
- Each element: dot product of d-dimensional vectors
- n² elements × d operations = O(n²d)
- Apply to V: (n, n) @ (n, d) → O(n²d)
- Total: O(n²d) time, O(n²) space

**The n² term:**
- Attention matrix: n×n (all pairs of tokens)
- Each token attends to all other tokens
- Quadratic in sequence length

**The d term:**
- Model dimension (typically 768-12288)
- Vector operations scale with dimension

**Multi-Head Attention:**
- Still O(n²d) overall
- Divides d into h heads (d/h each)
- h heads × O(n²d/h) = O(n²d)
- Can parallelize across heads

**Linear Attention: O(nd²)**
- Reformulates: (QK^T)V = Q(K^T V)
- Compute K^T V first: O(nd²)
- Then Q @ (K^T V): O(nd²)
- Faster when n >> d

**Sparse Attention: O(n√n d) or O(n log n d)**
- Only attends to subset of tokens
- Local window + global tokens
- Reduces n² to n√n or n log n

**Flash Attention: O(n²d) time, O(n) space**
- Same computation, block-wise
- Doesn't store full attention matrix
- Memory efficient

**See `05_attention_mechanisms/attention_complexity.md` for complete analysis!**

---

See `04_transformers/gpt_complete.py` for complete GPT implementation!
See `04_transformers/gpt_training_decoding.md` for training and decoding details!
See `05_attention_mechanisms/attention_complexity.md` for complexity analysis!

---

## Prompt Tuning and Prefix Tuning

### Q85: What is prompt tuning? How does it work?

**Answer:**

**Prompt Tuning:**
- Parameter-efficient fine-tuning method
- Adds trainable "soft prompts" (continuous embeddings) to input
- Keeps entire pre-trained model frozen
- Only trains prompt embeddings (typically 20-100 tokens)

**How It Works:**
1. Prepend trainable prompt embeddings to input
2. Pass [prompt; input] through frozen model
3. Only update prompt embeddings during training
4. Prompt learns to encode task-specific information

**Mathematical Formulation:**
```
E_input = Embedding(x)  # Input embeddings
P = [p₁, ..., pₚ]  # Trainable prompt (p tokens)
E_combined = [P; E_input]  # Concatenate
output = Model_θ(E_combined)  # Model frozen
# Only P is updated: P ← P - α∇P
```

**Parameters:**
- Trainable: p × d_model (e.g., 20 × 768 = 15,360)
- Efficiency: 0.01% of model parameters
- Storage: Only prompt embeddings per task

**Advantages:**
- Extremely parameter-efficient
- Simple implementation
- Fast training
- Preserves pre-trained knowledge
- Enables multi-task deployment

---

### Q86: What is prefix tuning? How does it differ from prompt tuning?

**Answer:**

**Prefix Tuning:**
- Similar to prompt tuning but adds parameters at every layer
- Adds trainable "prefix" key-value pairs at each transformer layer
- More expressive than prompt tuning
- Still parameter-efficient

**Key Differences:**

**1. Where Parameters Are Added:**
- **Prompt tuning**: Only at input layer
- **Prefix tuning**: At every transformer layer
- **Impact**: Prefix influences model at multiple levels

**2. What's Added:**
- **Prompt tuning**: Prompt embeddings (input)
- **Prefix tuning**: Prefix keys and values (attention)
- **Impact**: Prefix directly modifies attention computation

**3. Parameters:**
- **Prompt tuning**: p × d_model
- **Prefix tuning**: L × p × 2d_model (for K and V)
- **Example**: 12 layers, 20 tokens, 768 dim
  - Prompt: 15,360 parameters
  - Prefix: ~368,640 parameters
  - Still much less than full model

**4. Performance:**
- **Prompt tuning**: Good for simple tasks
- **Prefix tuning**: Often matches full fine-tuning
- **Trade-off**: More parameters for better performance

**Mathematical Formulation:**
```
At each layer l:
K_l = [P_l^K; K_l]  # Add prefix keys
V_l = [P_l^V; V_l]  # Add prefix values
Q_l unchanged
Attention_l = softmax(Q_l K_l^T) V_l
```

---

### Q87: Compare prompt tuning, prefix tuning, LoRA, and full fine-tuning.

**Answer:**

**Parameter Efficiency:**

| Method | Parameters | Example (GPT-2) | Efficiency |
|--------|-----------|-----------------|------------|
| **Full Fine-tuning** | 100% | 125M | 1x |
| **LoRA** | 0.1-1% | 125K-1.25M | 100-1000x |
| **Prefix Tuning** | 0.3% | ~368K | ~340x |
| **Prompt Tuning** | 0.01% | ~15K | ~8000x |

**Performance:**

**Full Fine-tuning:**
- Best performance
- Risk of catastrophic forgetting
- Requires most resources

**LoRA:**
- Near full fine-tuning performance
- Best balance of efficiency and performance
- Most popular in practice

**Prefix Tuning:**
- Very good performance (often matches full fine-tuning)
- More expressive than prompt tuning
- Good for complex tasks

**Prompt Tuning:**
- Good performance
- Sufficient for many tasks
- Maximum efficiency

**Use Cases:**

- **Full Fine-tuning**: Maximum performance, single task
- **LoRA**: Best balance, most common
- **Prefix Tuning**: Complex tasks, good performance
- **Prompt Tuning**: Simple tasks, maximum efficiency

**Recommendation:**
- Start with prompt tuning (simplest)
- If insufficient, try prefix tuning
- For best balance, use LoRA
- Full fine-tuning only if needed

---

### Q88: How do you initialize prompt/prefix embeddings?

**Answer:**

**Initialization Strategies:**

**1. Random Initialization:**
```
P ~ N(0, 0.02²)  # Small random values
```
- Simple, unbiased
- May require more training

**2. Vocabulary-Based:**
```
Sample random tokens from vocabulary
Use their embeddings as initial prompt
```
- Starts with semantic information
- Often works better than random

**3. Task-Specific:**
```
Use embeddings from task-related tokens
E.g., sentiment: "sentiment", "positive", "negative"
```
- Better starting point
- Faster convergence

**4. Reparameterization (Prefix):**
```
Learn in smaller space (d_model/2)
Project up to full dimension
```
- More stable training
- Used in prefix tuning

**Best Practices:**
- **Prompt tuning**: Vocabulary-based initialization
- **Prefix tuning**: Reparameterization + random
- **Experiment**: Try different strategies
- **Use domain knowledge**: When available

---

### Q89: What is the optimal prompt/prefix length?

**Answer:**

**Typical Ranges:**
- **Prompt tuning**: 20-100 tokens (commonly 20-50)
- **Prefix tuning**: 10-50 tokens (commonly 10-20 per layer)

**Selection Factors:**

**1. Task Complexity:**
- Simple tasks: 20 tokens sufficient
- Complex tasks: 50-100 tokens needed
- Rule: More complex → longer prompt/prefix

**2. Dataset Size:**
- Large datasets: Can support longer prompts
- Small datasets: Shorter prompts (avoid overfitting)

**3. Model Size:**
- Larger models: Can utilize longer prompts
- Smaller models: Shorter prompts sufficient

**Selection Process:**
1. Start with moderate length (20-30 tokens)
2. Try different lengths: [10, 20, 50, 100]
3. Evaluate on validation set
4. Choose best performing length

**Empirical Finding:**
- Performance improves with length up to a point
- Then plateaus (diminishing returns)
- Sweet spot: 20-50 tokens for most tasks

---

### Q90: Implement prompt tuning from scratch. Show the key code.

**Answer:**

**Key Implementation:**

```python
class PromptTuning(nn.Module):
    def __init__(self, model, prompt_length=20):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        self.d_model = model.config.n_embd
        
        # Freeze model
        for param in model.parameters():
            param.requires_grad = False
        
        # Trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, self.d_model) * 0.02
        )
    
    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        
        # Input embeddings
        input_emb = self.model.transformer.wte(input_ids)
        
        # Expand prompt for batch
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: [prompt; input]
        combined = torch.cat([prompt, input_emb], dim=1)
        
        # Forward through frozen model
        outputs = self.model.transformer(inputs_embeds=combined)
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        return logits

# Training
optimizer = torch.optim.Adam([prompt_model.prompt_embeddings], lr=0.3)
# Only prompt_embeddings are updated
```

**Key Points:**
- Freeze all model parameters
- Only prompt_embeddings requires gradients
- Simple concatenation at input
- Extremely parameter-efficient

**See `25_adapters_lora/prompt_prefix_code.py` for complete implementation!**

---

See `25_adapters_lora/prompt_prefix_tuning.md` for detailed theory!
See `25_adapters_lora/prompt_prefix_code.py` for complete code!
See `25_adapters_lora/prompt_prefix_qa.md` for comprehensive Q&A!

---

## Diffusion Models

### Q91: What are diffusion models? How do they work?

**Answer:**

**Diffusion Models:**
- Generative models that learn to reverse a gradual noising process
- Work by iteratively removing noise from data, starting from pure noise
- State-of-the-art results in image generation (DALL-E, Stable Diffusion)

**How They Work:**

**1. Forward Process (Fixed):**
- Gradually add Gaussian noise to data
- q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
- After T steps, data becomes pure noise

**2. Reverse Process (Learned):**
- Learn to remove noise step by step
- p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
- Neural network predicts how to denoise

**3. Training:**
- Predict the noise that was added
- Loss: L = E[||ε - ε_θ(x_t, t)||²]

**4. Generation:**
- Start from pure noise x_T ~ N(0, I)
- Iteratively apply reverse process: x_T → x_{T-1} → ... → x_0

**Key Insight:**
- Break down complex generation into many simple denoising steps
- Each step only removes small amount of noise
- Much easier to learn than generating directly

---

### Q92: How do you train a diffusion model?

**Answer:**

**Training Algorithm:**

**1. Setup:**
- Define variance schedule β_t (linear or cosine)
- Precompute α_t, ᾱ_t for efficiency

**2. Training Loop:**
```
For each batch:
  a. Sample data: x_0 ~ q(x_0)
  b. Sample timestep: t ~ Uniform({1, 2, ..., T})
  c. Sample noise: ε ~ N(0, I)
  d. Create noisy data: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
  e. Predict noise: ε_pred = ε_θ(x_t, t)
  f. Compute loss: L = ||ε - ε_pred||²
  g. Update: θ ← θ - α∇_θ L
```

**Best Practices:**
- Learning rate: 1e-4 to 1e-3
- Use learning rate scheduling (cosine annealing)
- Gradient clipping (norm = 1.0)
- Monitor loss and generate samples during training

**Variance Schedule:**
- Linear: β_t = (β_max - β_min) * (t/T) + β_min
- Cosine: ᾱ_t = cos²(π/2 * (t/T)) (often better)

---

### Q93: What are discrete diffusion models? How do they work for NLP?

**Answer:**

**The Challenge:**
- Standard diffusion works on continuous data (images)
- Text is discrete (tokens), need adaptation

**Discrete Forward Process:**

Instead of Gaussian noise, use transition matrix:
```
q(x_t | x_{t-1}) = Categorical(x_t; Q_t x_{t-1})
```

**Common Approaches:**

**1. Absorbing State:**
- Have special [MASK] token
- At each step, tokens transition to [MASK] with probability β_t
- After T steps, all tokens become [MASK]

**2. Uniform Transition:**
- Tokens can transition to any other token uniformly

**Discrete Reverse Process:**

Learn to predict original token:
```
p_θ(x_{t-1} | x_t) = Categorical(x_{t-1}; p_θ(x_t, t))
```

**Advantages for NLP:**
- Non-autoregressive (can generate in parallel)
- Better for editing tasks (text inpainting)
- More flexible control

---

### Q94: What are use cases of diffusion models in NLP?

**Answer:**

**1. Non-Autoregressive Text Generation:**
- Generate all tokens in parallel
- Faster than autoregressive models
- Better for controlled generation

**2. Text Inpainting:**
- Fill in masked tokens
- Edit specific parts of text
- Example: "The [MASK] sat on the [MASK]" → "The cat sat on the mat"

**3. Text-to-Image:**
- DALL-E, Stable Diffusion
- Generate images from text descriptions
- Multimodal understanding

**4. Text Editing:**
- Style transfer
- Paraphrasing
- Rewriting with constraints

**5. Controllable Generation:**
- Generate with specific attributes
- Control length, style, topic
- More flexible than autoregressive

**Industry Examples:**
- DALL-E: Text-to-image generation
- Stable Diffusion: Open-source text-to-image
- Research: Non-autoregressive text generation

---

### Q95: How do you evaluate diffusion models?

**Answer:**

**For Images:**

**1. FID (Frechet Inception Distance):**
- Measures quality and diversity
- Lower is better
- Compares feature distributions

**2. IS (Inception Score):**
- Measures quality and diversity
- Higher is better (typically 1-10)

**3. Reconstruction Error:**
- Test if model can recover original
- Lower is better

**For Text:**

**1. BLEU Score:**
- Measures n-gram overlap with reference
- Higher is better (0-1)

**2. Perplexity:**
- Measures how well model predicts tokens
- Lower is better

**3. Diversity Metrics:**
- Distinct-n: Ratio of unique n-grams
- Self-BLEU: Average BLEU between samples
- Higher distinct = more diverse

**Diffusion-Specific:**

**1. Denoising Accuracy:**
- Test accuracy at each timestep
- Measures how well model denoises

**2. Sample Quality:**
- Visual inspection (for images)
- Human evaluation (for text)

---

### Q96: Compare diffusion models with autoregressive models (GPT) for text generation.

**Answer:**

**Generation Process:**

**Autoregressive (GPT):**
- Generate left-to-right, one token at a time
- Sequential: t₁ → t₂ → t₃ → ...

**Diffusion:**
- Generate all tokens in parallel (discrete diffusion)
- Iteratively refine all tokens together

**Advantages:**

**Autoregressive:**
- Faster single-pass generation
- Simpler implementation
- Better for long sequences
- More established for text

**Diffusion:**
- Non-autoregressive (parallel)
- Better for editing tasks
- More flexible control
- Can edit specific parts

**When to Use:**

**Autoregressive:**
- Standard text generation
- Long sequences
- When speed is important

**Diffusion:**
- Text editing/inpainting
- Controlled generation
- When need parallel generation

**Current State:**
- Autoregressive (GPT) dominates text generation
- Diffusion better for images
- Discrete diffusion promising for text

---

See `40_diffusion_models/diffusion_theory.md` for complete theory!
See `40_diffusion_models/diffusion_code.py` for continuous diffusion!
See `40_diffusion_models/nlp_diffusion.py` for discrete diffusion!
See `40_diffusion_models/training_diffusion.py` for training procedures!
See `40_diffusion_models/evaluation_diffusion.py` for evaluation methods!
See `40_diffusion_models/diffusion_qa.md` for comprehensive Q&A!

---

## Perplexity and Related Concepts

### Q97: What is perplexity? How is it computed?

**Answer:**

**Perplexity:**
- Metric that measures how well a probability model predicts a sample
- Defined as exponentiated average negative log-likelihood
- Lower perplexity = better model

**Mathematical Definition:**

```
PP(W) = exp(-(1/N) * Σ log P(w_i | context))
```

Where:
- W = (w₁, w₂, ..., wₙ) is a sequence of tokens
- P(w_i | context) is probability assigned by model
- N is number of tokens

**Intuitive Understanding:**
- Perplexity = k means model is as uncertain as uniform choice among k options
- If PP = 10, model thinks there are 10 equally likely next tokens
- Lower perplexity = model is more confident = better predictions

**Computation:**

**1. Get Model Predictions:**
```python
logits = model(input_ids)  # (batch, seq_len, vocab_size)
probs = softmax(logits, dim=-1)
```

**2. Get True Token Probabilities:**
```python
true_token_probs = probs[range(batch), range(seq_len), true_tokens]
```

**3. Compute Perplexity:**
```python
nll = -log(true_token_probs).mean()  # Negative log-likelihood
perplexity = exp(nll)
```

**Connection to Cross-Entropy:**
- Cross-entropy loss = average negative log-likelihood
- Perplexity = exp(cross-entropy_loss)
- Minimizing loss = minimizing perplexity

---

### Q98: What does perplexity mean? How do you interpret it?

**Answer:**

**Interpretation:**

**Perplexity = k means:**
- Model is as uncertain as if it had to choose uniformly among k options
- On average, model thinks there are k equally likely next tokens

**Examples:**

**Perplexity = 1:**
- Model is perfectly certain
- Always predicts one token with probability 1
- Unrealistic for real language

**Perplexity = 10:**
- Model is as uncertain as uniform choice among 10 tokens
- Reasonable for a good language model
- Better than random (which would be vocabulary size)

**Perplexity = 100:**
- Model is very uncertain
- As confused as uniform choice among 100 tokens
- Indicates poor model or difficult task

**Perplexity = Vocabulary Size:**
- Model is as bad as random guessing
- Worst case scenario

**Typical Values:**

**For Language Models:**
- GPT-2 (small): ~30-50 on WikiText-103
- GPT-2 (large): ~15-25 on WikiText-103
- GPT-3: ~10-20 on various datasets
- State-of-the-art: < 10 on some datasets

**For Different Tasks:**
- Simple tasks: Lower perplexity (5-20)
- Complex tasks: Higher perplexity (20-100)
- Domain-specific: Varies widely

**Connection to Entropy:**
- Perplexity = 2^H (where H is cross-entropy in bits)
- Entropy measures uncertainty in bits
- Perplexity measures uncertainty in "effective vocabulary size"

---

### Q99: How is perplexity related to entropy and cross-entropy?

**Answer:**

**Connection to Entropy:**

**Entropy:**
```
H(X) = -Σ P(x) * log P(x)
```

**Perplexity:**
```
PP = 2^H(X)  (for base-2 log)
PP = exp(H(X))  (for natural log)
```

**Intuition:**
- Entropy: uncertainty in bits
- Perplexity: uncertainty in "effective vocabulary size"
- If entropy = log₂(10) ≈ 3.32 bits, perplexity = 2^3.32 ≈ 10

**Connection to Cross-Entropy:**

**Cross-Entropy:**
```
H(P, Q) = -Σ P(x) * log Q(x)
```

**For Language Models:**
```
H = -(1/N) * Σ log P(w_i | context)
```

**Perplexity:**
```
PP = exp(H)
```

**Key Insight:**
- Cross-entropy loss = average negative log-likelihood
- Perplexity = exp(cross-entropy_loss)
- Minimizing cross-entropy = minimizing perplexity
- They are equivalent objectives

**Training:**
- When training language models, we minimize cross-entropy
- This is equivalent to minimizing perplexity
- Lower loss = lower perplexity = better model

**Bits per Token:**
- BPT = log₂(PP) = H (in bits)
- Lower BPT = lower perplexity = better model
- More interpretable for some applications

---

### Q100: How do you compute perplexity for a language model? Show the code.

**Answer:**

**Step-by-Step Algorithm:**

**1. Get Model Predictions:**
```python
logits = model(input_ids)  # (batch, seq_len, vocab_size)
```

**2. Get Log Probabilities:**
```python
log_probs = F.log_softmax(logits, dim=-1)
```

**3. Get True Token Log Probabilities:**
```python
batch_size, seq_len = targets.shape
indices = targets.unsqueeze(-1)  # (batch, seq, 1)
true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
```

**4. Compute Average Negative Log-Likelihood:**
```python
if mask is not None:
    nll = -(true_token_log_probs * mask).sum() / mask.sum()
else:
    nll = -true_token_log_probs.mean()
```

**5. Compute Perplexity:**
```python
perplexity = torch.exp(nll).item()
```

**Complete Function:**

```python
def perplexity_from_logits(logits, targets, mask=None):
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get true token log probabilities
    indices = targets.unsqueeze(-1)
    true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
    
    # Average negative log-likelihood
    if mask is not None:
        nll = -(true_token_log_probs * mask).sum() / mask.sum()
    else:
        nll = -true_token_log_probs.mean()
    
    # Perplexity
    return torch.exp(nll).item()
```

**For Language Model Evaluation:**

```python
def language_model_perplexity(model, dataloader, device='cpu'):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids).logits
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # Compute perplexity
            pp = perplexity_from_logits(shift_logits, shift_labels)
            
            # Accumulate
            batch_tokens = shift_labels.numel()
            total_nll += np.log(pp) * batch_tokens
            total_tokens += batch_tokens
    
    # Average perplexity
    avg_pp = np.exp(total_nll / total_tokens)
    return avg_pp
```

**See `03_evaluation_metrics/perplexity_code.py` for complete implementation!**

---

### Q101: What are the limitations of perplexity? When should you use other metrics?

**Answer:**

**Limitations:**

**1. Not Always Correlates with Quality:**
- Lower perplexity doesn't always mean better text
- Can overfit to training data
- May not reflect human judgment
- Need other metrics (BLEU, ROUGE, human eval)

**2. Dataset Dependent:**
- Perplexity varies by dataset
- Can't compare across different datasets
- Need same preprocessing
- Fair comparison requires same setup

**3. Vocabulary Size Matters:**
- Larger vocabulary = higher baseline perplexity
- Need to account for vocabulary size
- Normalized perplexity helps
- Compare models with similar vocabularies

**4. Sequence Length:**
- Longer sequences = more stable estimate
- Shorter sequences = more variable
- Need sufficient data for reliable estimate

**5. Task-Specific:**
- Good perplexity doesn't guarantee good performance on downstream tasks
- May not reflect task-specific quality
- Need task-specific metrics

**When to Use Other Metrics:**

**1. Text Generation:**
- Use BLEU, ROUGE for quality
- Use diversity metrics (distinct-n)
- Use human evaluation
- Perplexity as one of many metrics

**2. Machine Translation:**
- Use BLEU as primary metric
- Use METEOR, TER
- Perplexity for model selection

**3. Summarization:**
- Use ROUGE as primary metric
- Use BLEU, METEOR
- Perplexity for training monitoring

**4. Question Answering:**
- Use EM (Exact Match), F1
- Use BLEU for generation quality
- Perplexity less relevant

**Best Practices:**
- Use perplexity for model selection during training
- Combine with task-specific metrics
- Don't rely only on perplexity
- Consider context and task requirements

---

See `03_evaluation_metrics/perplexity_detailed.md` for complete theory!
See `03_evaluation_metrics/perplexity_code.py` for complete code!
See `33_information_theory/information_theory.py` for entropy implementation!

---

## Causal Attention

### Q102: Explain causal attention. What does the code `np.tril(np.ones((seq_len, seq_len)))` do?

**Answer:**

**Causal Attention:**
- Masks future positions to enforce autoregressive property
- Each position can only attend to itself and previous positions
- Critical for GPT-style models (autoregressive generation)

**The Code:**
```python
mask = np.tril(np.ones((seq_len, seq_len)))
```

**Step-by-Step:**

**1. `np.ones((seq_len, seq_len))`:**
- Creates matrix of all 1s
- Shape: (seq_len, seq_len)
- Example for seq_len=4:
```
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

**2. `np.tril()`:**
- Takes lower triangular part
- Sets everything above diagonal to 0
- Keeps everything on and below diagonal as is
- Result:
```
[[1, 0, 0, 0],   ← Position 0: can only see itself
 [1, 1, 0, 0],   ← Position 1: can see 0, 1
 [1, 1, 1, 0],   ← Position 2: can see 0, 1, 2
 [1, 1, 1, 1]]   ← Position 3: can see all (0, 1, 2, 3)
```

**3. Application:**
- Mask applied to attention scores
- `scores[mask == 0] = -∞` (future positions)
- After softmax: Future positions get 0 attention weight
- Result: Each position only attends to past and current

**Why Lower Triangular?**
- Lower triangular = can attend to positions ≤ current (past + current)
- Upper triangular = wrong (would allow future, block past)
- This enforces causal constraint for autoregressive generation

**See `05_attention_mechanisms/causal_attention_detailed.md` for complete explanation!**

---

### Q103: Why do we need causal attention? What happens without it?

**Answer:**

**Why We Need It:**

**Autoregressive Constraint:**
- In autoregressive generation, tokens are generated left-to-right
- When generating token at position i, only tokens 0...i-1 exist
- Future tokens (i+1, i+2, ...) don't exist yet
- Model should only use information from past and current tokens

**What Happens Without Causal Mask:**

**During Training:**
- Model sees full sequence: [token_0, token_1, ..., token_n]
- Without mask: Each position can attend to ALL positions (including future)
- Model learns to use future tokens for prediction

**During Inference:**
- Generate one token at a time
- At step i, only have [token_0, ..., token_{i-1}]
- Future tokens don't exist
- But model was trained to use future tokens!

**Result:**
- Training and inference mismatch
- Model behavior inconsistent
- Poor generation quality

**With Causal Mask:**
- Training: Each position only sees past/current (matches inference)
- Inference: Each position only sees past/current (matches training)
- Consistent behavior → good generation

**Example:**
- Without mask: Position 1 can see position 2 (future) during training
- With mask: Position 1 cannot see position 2 (future) during training
- This matches inference where position 2 doesn't exist yet

---

### Q104: How does the causal mask work mathematically?

**Answer:**

**Mathematical Formulation:**

**Standard Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Causal Attention:**
```
Attention(Q, K, V) = softmax((QK^T / √d_k) + M) V
```

Where M is the causal mask:
```
M[i, j] = {
    0   if j ≤ i  (can attend to past/current)
    -∞  if j > i  (cannot attend to future)
}
```

**Step-by-Step:**

**1. Compute Attention Scores:**
```
scores = Q @ K.T / √d_k  # Shape: (seq_len, seq_len)
```

**2. Apply Causal Mask:**
```
masked_scores = scores + M
# Where M[i, j] = -∞ if j > i (future positions)
```

**3. Softmax:**
```
attention_weights = softmax(masked_scores)
```

**What Happens:**
- Future positions: scores = -∞ → softmax(-∞) = 0
- Past/current positions: scores = original → softmax(original) = normal weights

**Result:**
- Each position gets attention weights that sum to 1
- Future positions always have 0 weight
- Past/current positions have non-zero weights

**Example for seq_len=4:**

**Mask Matrix:**
```
M = [[0,  -∞, -∞, -∞],
     [0,  0,  -∞, -∞],
     [0,  0,  0,  -∞],
     [0,  0,  0,  0]]
```

**After Adding Mask:**
```
scores = [[2.3, -∞,  -∞,  -∞],
          [1.8, 2.1, -∞,  -∞],
          [1.2, 1.7, 2.0, -∞],
          [0.9, 1.4, 1.6, 2.2]]
```

**After Softmax:**
```
weights = [[1.0, 0.0, 0.0, 0.0],   ← Position 0: 100% to itself
           [0.4, 0.6, 0.0, 0.0],   ← Position 1: 40% to 0, 60% to 1
           [0.2, 0.3, 0.5, 0.0],   ← Position 2: distributed, 0% to future
           [0.1, 0.2, 0.3, 0.4]]   ← Position 3: distributed across all
```

---

See `05_attention_mechanisms/causal_attention_detailed.md` for complete theory!
See `05_attention_mechanisms/causal_attention_code.py` for visualization!

---

## Advanced Attention Mechanisms (GQA, Paged Attention)

### Q105: What is Group Query Attention (GQA)? How does it differ from Multi-Head Attention?

**Answer:**

**Group Query Attention (GQA):**
- Groups heads and shares K, V within each group
- Middle ground between MHA and MQA
- Reduces KV cache memory while maintaining quality

**Key Differences:**

**Multi-Head Attention (MHA):**
- Each head has separate Q, K, V
- KV Cache: num_heads × seq_len × (d_k + d_v)
- Parameters: 3 × num_heads × d_model²

**Group Query Attention (GQA):**
- Heads grouped, K, V shared within each group
- Q separate per head (like MHA)
- KV Cache: num_groups × seq_len × (d_k + d_v)
- Parameters: num_heads × d_model² + 2 × num_groups × d_model²

**Example: 32 heads, 8 groups**
- MHA: 32 × seq_len × (d_k + d_v) KV cache
- GQA: 8 × seq_len × (d_k + d_v) KV cache
- Reduction: 4× in KV cache memory

**Why It Works:**
- Queries need to be different (capture different aspects)
- Keys and values can be shared within groups
- Maintains most of MHA's expressiveness
- Significant memory reduction

**When to Use:**
- Production inference (recommended)
- Need efficiency but maintain quality
- Best balance between MHA and MQA

---

### Q106: What is Multi-Query Attention (MQA)? How does it reduce memory?

**Answer:**

**Multi-Query Attention (MQA):**
- Shares K and V across ALL heads
- Only Q is separate per head
- Maximum memory reduction

**Key Difference:**

**MHA:**
```
Head 1: Q_1, K_1, V_1
Head 2: Q_2, K_2, V_2
...
Head h: Q_h, K_h, V_h
```

**MQA:**
```
Head 1: Q_1, K_shared, V_shared
Head 2: Q_2, K_shared, V_shared
...
Head h: Q_h, K_shared, V_shared
```

**Memory Reduction:**

**KV Cache:**
- MHA: num_heads × seq_len × (d_k + d_v)
- MQA: 1 × seq_len × (d_k + d_v) (shared, not per head!)
- Reduction: num_heads× (e.g., 32× for 32 heads)

**Parameters:**
- MHA: 3 × num_heads × d_model²
- MQA: num_heads × d_model² + 2 × d_model²
- Reduction: From 3×num_heads to (num_heads + 2)

**Example: 32 heads, seq_len=2048, d_k=128**
- MHA KV Cache: 32 × 2048 × 256 = 16.8M values
- MQA KV Cache: 1 × 2048 × 256 = 0.5M values
- Reduction: 32× (16.8M → 0.5M)

**Why It Works:**
- Queries represent "what am I looking for?" (different per head)
- Keys represent "what information do I have?" (can be shared)
- Values represent "what is the information?" (can be shared)
- Same information, different queries → similar quality

**Trade-offs:**
- Maximum memory reduction
- Slight quality loss compared to MHA
- Still achieves good quality
- Used when maximum efficiency needed

---

### Q107: What is Paged Attention? How does it improve memory efficiency?

**Answer:**

**Paged Attention:**
- Memory-efficient KV cache management
- Manages cache in non-contiguous pages (blocks)
- Similar to virtual memory in operating systems
- Core innovation behind vLLM

**The Problem: Memory Fragmentation**

**Standard KV Cache:**
- Store K, V for each sequence in contiguous memory
- Variable-length sequences → memory fragmentation
- When sequence finishes, memory freed but fragmented
- Cannot reuse efficiently → waste

**Example:**
```
Sequence 1: [12 tokens, finished] → 12 tokens freed
Sequence 2: [16 tokens, still generating]
New sequence needs 20 tokens → Cannot use the 12 freed tokens (fragmented)
```

**Paged Attention Solution:**

**1. Page Structure:**
- Divide KV cache into fixed-size pages (blocks)
- Each page stores K, V for block_size tokens (e.g., 16 tokens)
- Pages can be non-contiguous in memory

**2. Memory Management:**
- Maintain pool of free pages
- Allocate pages on-demand
- Return pages to pool when sequence finishes
- Pages can be reused immediately

**3. Benefits:**
- No memory fragmentation
- Efficient memory reuse
- Can handle variable-length sequences
- Better GPU memory utilization (95%+ vs ~70%)

**Example:**
- block_size = 16 tokens
- Sequence of 25 tokens: needs 2 pages (32 tokens allocated)
- Waste: Only 7 tokens (within last page)
- Much better than standard (could waste 50%+)

**Memory Efficiency:**
- Standard: ~70% utilization (due to fragmentation)
- Paged: ~95%+ utilization
- Enables serving more sequences with same memory

**See `05_attention_mechanisms/advanced_attention_mechanisms.md` for complete details!**

---

### Q108: Compare MHA, GQA, and MQA. When should you use each?

**Answer:**

**Comparison Table:**

| Aspect | MHA | GQA | MQA |
|-------|-----|-----|-----|
| **Q Projections** | num_heads | num_heads | num_heads |
| **K Projections** | num_heads | num_groups | 1 |
| **V Projections** | num_heads | num_groups | 1 |
| **KV Cache** | num_heads × seq_len × (d_k + d_v) | num_groups × seq_len × (d_k + d_v) | seq_len × (d_k + d_v) |
| **Quality** | Best | Very Good | Good |
| **Memory** | Highest | Medium | Lowest |
| **Use Case** | Training, research | Production (recommended) | Maximum efficiency |

**Example: 32 heads, 8 groups, seq_len=2048**

**MHA:**
- KV Cache: 32 × 2048 × 256 = 16.8M values
- Quality: Best
- Use: Training, maximum quality needed

**GQA:**
- KV Cache: 8 × 2048 × 256 = 4.2M values (4× reduction)
- Quality: Very Good (minimal loss)
- Use: Production inference (recommended)

**MQA:**
- KV Cache: 1 × 2048 × 256 = 0.5M values (32× reduction)
- Quality: Good (slight loss)
- Use: Maximum efficiency needed

**When to Use:**

**MHA:**
- Training: Maximum quality
- Research: Need best performance
- When: Have resources, quality is priority

**GQA:**
- Production inference: Best balance
- Recommended default
- When: Need efficiency but maintain quality

**MQA:**
- Maximum efficiency needed
- Quality loss acceptable
- When: Resource-constrained, high throughput

**Paged Attention:**
- Can be used with any of above
- Production serving (vLLM)
- When: Need efficient memory management

---

See `05_attention_mechanisms/advanced_attention_mechanisms.md` for complete theory!
See `05_attention_mechanisms/advanced_attention_code.py` for complete code!

---

## Mixture of Experts (MoE)

### Q109: What is Mixture of Experts? How does it work?

**Answer:**

**Mixture of Experts (MoE):**
- Architecture with multiple expert networks
- Router decides which experts to activate
- Only subset of experts process each input
- Enables models with trillions of parameters

**How It Works:**

**1. Multiple Experts:**
- 8-128 feed-forward networks
- Each expert is independent
- All experts have same architecture

**2. Router:**
- Takes input, outputs expert scores
- Computes probability distribution
- Selects top-k experts with highest scores

**3. Sparse Activation:**
- Only k experts activated per token (typically k=1 or 2)
- Most experts remain inactive
- Reduces computation significantly

**4. Weighted Combination:**
- Process through selected experts
- Weighted combination of outputs

**Efficiency:**
- Total parameters: num_experts × params_per_expert
- Active parameters: k × params_per_expert
- Example: 8 experts, k=2 → 4× reduction in computation

---

### Q110: How does MoE reduce computation? Compare with dense models.

**Answer:**

**Dense Model:**
- All parameters used for every input
- Computation: O(d_model²) per token
- Example: 7B parameters, all active

**MoE Model:**
- Total: num_experts × params_per_expert
- Active: k × params_per_expert
- Computation: O(k × d_model²) per token

**Example: Mixtral-8x7B**
- 8 experts × 7B = 56B total parameters
- k=2 → 2 × 7B = 14B active per token
- Computation: Only 14B parameters (not 56B!)

**Reduction:**
- Computation: (num_experts / k)× reduction
- 8 experts, k=2 → 4× reduction
- But total parameters: 8× more

**Trade-off:**
- More parameters (memory)
- Less computation (speed)
- Best of both worlds

---

### Q111: What is load balancing in MoE? Why is it important?

**Answer:**

**Load Balancing Problem:**
- Without balancing, router might always select same experts
- Some experts never used (waste)
- Others overloaded (bottleneck)
- Expert collapse: Only few experts ever used

**Solution: Load Balancing Loss**
```
L_balance = (1/num_experts) * sum(load_i)²
```

Where load_i is fraction of tokens routed to expert i.

**Goal:**
- Minimize variance of expert usage
- Distribute tokens evenly
- All experts used roughly equally

**Why Important:**
- Without: Experts 0-2 always used, 3-7 never used
- With: All experts used equally
- Better parameter utilization
- Prevents expert collapse

---

See `41_mixture_of_experts/moe_theory.md` for complete theory!
See `41_mixture_of_experts/moe_code.py` for complete code!
See `41_mixture_of_experts/moe_qa.md` for comprehensive Q&A!

---

## State Space Models (SSM)

### Q112: What are State Space Models? How do they work?

**Answer:**

**State Space Models (SSMs):**
- Sequence models using hidden state
- Process sequences with linear recurrence
- O(n) complexity (vs O(n²) for transformers)
- Better for very long sequences

**How They Work:**

**1. Hidden State:**
- Maintain state h[k] that evolves over time
- State captures information from all previous inputs
- Updated at each step

**2. State Evolution:**
```
h[k+1] = A_d h[k] + B_d u[k]  # State update
y[k] = C_d h[k] + D_d u[k]    # Output
```

**3. Linear Recurrence:**
- Each step: O(1) computation
- Total: O(n) for sequence of length n
- Much faster than attention: O(n²)

**Key Insight:**
- State summarizes past information
- Don't need to attend to all previous tokens
- More efficient than attention

---

### Q113: What is Mamba? How does it differ from standard SSMs?

**Answer:**

**Mamba:**
- Selective State Space Model
- Makes parameters input-dependent
- More expressive than fixed SSMs
- State-of-the-art for long sequences

**Key Difference:**

**Standard SSM:**
```
h[k+1] = A h[k] + B u[k]  # Fixed A, B
y[k] = C h[k]             # Fixed C
```

**Mamba (Selective):**
```
B[k] = Linear_B(u[k])  # Input-dependent B
C[k] = Linear_C(u[k])  # Input-dependent C
h[k+1] = A h[k] + B[k] u[k]
y[k] = C[k] h[k]
```

**Why This Works:**
- Different inputs need different transitions
- B[k] controls how input affects state
- C[k] controls what to extract
- More expressive while maintaining O(n) complexity

---

### Q114: Compare SSMs (Mamba) with Transformers. When to use each?

**Answer:**

**Complexity:**

| Aspect | Transformer | SSM (Mamba) |
|--------|-------------|-------------|
| **Time** | O(n²d) | O(nd) |
| **Space** | O(n²) | O(nd) |
| **Scaling** | Quadratic | Linear |

**When to Use:**

**Transformers:**
- Short-medium sequences (< 8K tokens)
- Need maximum quality
- Established architecture

**SSMs (Mamba):**
- Very long sequences (> 8K tokens)
- Need efficiency
- Sequences of length 100K+

**Crossover:**
- < 2K: Transformers faster
- > 8K: SSMs faster
- > 100K: SSMs much better

---

See `42_state_space_models/ssm_theory.md` for complete theory!
See `42_state_space_models/ssm_code.py` for complete code!
See `42_state_space_models/ssm_qa.md` for comprehensive Q&A!

## More Questions

See full `INTERVIEW_QA.md` for 100+ questions covering:
- All classical ML algorithms
- Complete LLM theory
- All inference techniques
- Training methods
- Optimization
- Regularization
- Information theory
- And more!

Good luck with your interviews! 🚀

