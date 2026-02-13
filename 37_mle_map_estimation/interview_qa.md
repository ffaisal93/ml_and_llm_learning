# MLE and MAP: Interview Q&A

## Q1: Derive MLE for a coin flip (Bernoulli distribution).

**Answer:**

**Setup:**
- Data: n flips, k heads, (n-k) tails
- Model: P(heads) = θ, P(tails) = 1-θ
- Goal: Find θ that maximizes likelihood

**Step 1: Likelihood**
```
L(θ) = θᵏ × (1-θ)ⁿ⁻ᵏ
```

**Step 2: Log-Likelihood**
```
log L(θ) = k log θ + (n-k) log(1-θ)
```

**Step 3: Take Derivative**
```
d/dθ [log L(θ)] = k/θ - (n-k)/(1-θ)
```

**Step 4: Set to Zero**
```
k/θ - (n-k)/(1-θ) = 0
k(1-θ) = (n-k)θ
k = nθ
θ = k/n
```

**Result:**
```
θ̂_MLE = k/n = (number of heads) / (total flips)
```

**Intuition:** The MLE is simply the observed proportion!

---

## Q2: Derive MLE for linear regression.

**Answer:**

**Setup:**
- Model: y = Xw + ε, where ε ~ N(0, σ²)
- Data: {(x₁, y₁), ..., (xₙ, yₙ)}
- Goal: Find w that maximizes likelihood

**Step 1: Likelihood**
```
L(w, σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(yᵢ - xᵢᵀw)²/(2σ²))
```

**Step 2: Log-Likelihood**
```
log L(w, σ²) = -n/2 log(2πσ²) - (1/(2σ²)) Σᵢ(yᵢ - xᵢᵀw)²
```

**Step 3: Maximize with respect to w**

Since σ² doesn't depend on w:
```
argmax_w log L(w, σ²) = argmin_w [Σᵢ(yᵢ - xᵢᵀw)²]
                     = argmin_w ||y - Xw||²
```

**Step 4: Take Derivative and Set to Zero**
```
∂/∂w [||y - Xw||²] = -2Xᵀ(y - Xw) = 0
Xᵀy = XᵀXw
w = (XᵀX)⁻¹Xᵀy
```

**Result:**
```
ŵ_MLE = (XᵀX)⁻¹Xᵀy  (Ordinary Least Squares!)
```

**Key Insight:** MLE for linear regression with Gaussian noise = OLS!

---

## Q3: Explain the connection between MLE and MAP.

**Answer:**

**MLE:**
```
θ̂_MLE = argmax_θ log P(D|θ)
```

**MAP:**
```
θ̂_MAP = argmax_θ [log P(D|θ) + log P(θ)]
      = argmax_θ [log L(θ) + log P(θ)]
      = MLE + log(prior)
```

**Key Relationship:**
```
MAP = MLE + Prior
```

**When they're the same:**
1. **Uniform prior**: P(θ) = constant → log P(θ) = constant → MAP = MLE
2. **Large dataset**: Data overwhelms prior → MAP ≈ MLE

**When they differ:**
1. **Small dataset**: Prior has more influence
2. **Strong prior**: MAP pulled toward prior mean

**Regularization Connection:**
- **L2 (Ridge)**: MAP with Gaussian prior
- **L1 (Lasso)**: MAP with Laplace prior
- **Regularization = Bayesian prior**

---

## Q4: Derive MAP for linear regression with Gaussian prior (Ridge).

**Answer:**

**Setup:**
- Model: y = Xw + ε, where ε ~ N(0, σ²)
- Prior: w ~ N(0, σ²_prior I)
- Goal: Find w that maximizes posterior

**Step 1: Prior**
```
P(w) ∝ exp(-||w||²/(2σ²_prior))
```

**Step 2: Likelihood**
```
L(w) ∝ exp(-||y - Xw||²/(2σ²))
```

**Step 3: Log-Posterior**
```
log P(w|D) = log L(w) + log P(w) + constant
           = -||y - Xw||²/(2σ²) - ||w||²/(2σ²_prior) + constant
```

**Step 4: Maximize**

Take derivative:
```
∂/∂w [log P(w|D)] = -1/σ² × Xᵀ(y - Xw) - 1/σ²_prior × w
```

Set to zero:
```
-1/σ² × Xᵀ(y - Xw) - 1/σ²_prior × w = 0
Xᵀ(y - Xw) = -σ²/σ²_prior × w
Xᵀy = (XᵀX + λI)w  (where λ = σ²/σ²_prior)
w = (XᵀX + λI)⁻¹Xᵀy
```

**Result:**
```
ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy  (Ridge regression!)
```

**Key Insight:** MAP with Gaussian prior = Ridge regression (L2 regularization)!

---

## Q5: Why do we use log-likelihood instead of likelihood?

**Answer:**

**Reasons:**

**1. Numerical Stability:**
- Likelihood: Product of many small probabilities → very small numbers
- Log-likelihood: Sum of logs → more stable
- Example: 0.1 × 0.1 × ... × 0.1 (100 times) = 10⁻¹⁰⁰ (underflow!)
- Log: log(0.1) + ... + log(0.1) = -100 × log(10) (manageable)

**2. Mathematical Convenience:**
- Products become sums: log(∏ᵢ Pᵢ) = Σᵢ log Pᵢ
- Derivatives easier: d/dθ log f(θ) = (1/f(θ)) × f'(θ)
- No product rule needed

**3. Monotonicity:**
- log is monotonic: maximizing log L(θ) = maximizing L(θ)
- Same maximum, easier optimization

**4. Additive Properties:**
- Log-likelihoods can be added: log L₁(θ) + log L₂(θ) = log(L₁(θ) × L₂(θ))
- Useful for combining datasets

**Example:**
```
L(θ) = 0.1 × 0.1 × 0.1 = 0.001  (hard to work with)
log L(θ) = log(0.1) + log(0.1) + log(0.1) = -3 × log(10) ≈ -6.91  (easier!)
```

---

## Q6: What's the difference between MLE and MAP in practice?

**Answer:**

**MLE:**
- **Approach**: Frequentist
- **Prior**: None (or uniform)
- **Result**: Point estimate
- **Use when**: Large dataset, no prior knowledge
- **Example**: θ̂ = k/n for coin flip

**MAP:**
- **Approach**: Bayesian
- **Prior**: Informative prior
- **Result**: Point estimate (mode of posterior)
- **Use when**: Small dataset, have prior knowledge, need regularization
- **Example**: θ̂ = (k+α-1)/(n+α+β-2) for coin flip with Beta prior

**Practical Differences:**

**1. Small Data:**
- MLE: Can be extreme (e.g., 3/3 heads → θ = 1.0)
- MAP: More reasonable (incorporates prior)

**2. Regularization:**
- MLE: No regularization (can overfit)
- MAP: Natural regularization (prior prevents overfitting)

**3. Uncertainty:**
- MLE: No uncertainty estimate
- MAP: Can estimate uncertainty (though full Bayesian is better)

**4. Computation:**
- MLE: Usually simpler
- MAP: Similar complexity (just add prior term)

**When to use:**
- **MLE**: Large data, simple models, no prior knowledge
- **MAP**: Small data, need regularization, have domain knowledge

---

## Q7: How does MAP relate to regularization?

**Answer:**

**Key Insight:**
```
Regularization = Bayesian Prior
Regularized Loss = -log Posterior
```

**L2 Regularization (Ridge):**
```
Loss = MSE + λ||w||²
     = -log L(w) + λ||w||²
     = -[log L(w) + log P(w)]  (where P(w) ∝ exp(-λ||w||²))
```

**This is MAP with Gaussian prior!**

**L1 Regularization (Lasso):**
```
Loss = MSE + λ||w||₁
     = -[log L(w) + log P(w)]  (where P(w) ∝ exp(-λ||w||₁))
```

**This is MAP with Laplace prior!**

**Interpretation:**
- **Regularization strength λ**: Controls prior strength
- **Higher λ**: Stronger prior → more regularization
- **Lower λ**: Weaker prior → less regularization

**Why this matters:**
- Understanding regularization as prior helps choose λ
- Bayesian interpretation provides theoretical foundation
- Can use Bayesian methods to learn λ from data

---

## Summary

**Key Points:**
1. **MLE**: Maximizes P(data|parameters) - frequentist approach
2. **MAP**: Maximizes P(parameters|data) - Bayesian approach
3. **Relationship**: MAP = MLE + log(prior)
4. **Regularization**: MAP with appropriate prior
5. **Use MLE**: Large data, no prior knowledge
6. **Use MAP**: Small data, need regularization, have prior knowledge

Understanding MLE and MAP is fundamental to understanding:
- How models learn from data
- Why regularization works
- Bayesian vs Frequentist thinking
- Connection between different ML approaches

