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

## More Questions

See full `INTERVIEW_QA.md` for 100+ questions covering:
- All classical ML algorithms
- Complete LLM theory
- All inference techniques
- Training methods
- Optimization
- Regularization
- And more!

Good luck with your interviews! 🚀

