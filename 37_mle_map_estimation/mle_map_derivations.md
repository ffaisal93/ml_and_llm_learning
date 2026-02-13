# MLE and MAP: Detailed Derivations with Intuitive Explanations

## Table of Contents

1. [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
2. [Maximum A Posteriori (MAP)](#maximum-a-posteriori-map)
3. [Connection: MLE vs MAP](#connection-mle-vs-map)
4. [Examples](#examples)
5. [When to Use Each](#when-to-use-each)

---

## Maximum Likelihood Estimation (MLE)

### Intuitive Explanation

**The Core Idea:**

Imagine you're a detective trying to figure out what happened. You have some evidence (data), and you want to figure out what scenario (parameters) is most likely to have produced that evidence.

**MLE asks:** "Given the data I observed, what parameter values make this data most probable?"

**Example:**
- You flip a coin 10 times and get 7 heads, 3 tails
- MLE asks: "What probability of heads makes 7 heads out of 10 flips most likely?"
- Answer: p = 0.7 (the observed proportion)

**Key Insight:**
MLE finds the parameter values that maximize the probability of observing the data you actually saw.

### Mathematical Setup

**Given:**
- Data: D = {x₁, x₂, ..., xₙ}
- Model: P(x|θ) (probability of data given parameters)
- Goal: Find θ that maximizes P(D|θ)

**Likelihood Function:**
```
L(θ) = P(D|θ) = P(x₁, x₂, ..., xₙ | θ)
```

**Key Point:** Likelihood is a function of parameters θ, not data. We treat data as fixed and vary parameters.

### Detailed Derivation

#### Step 1: Write the Likelihood

For independent observations:
```
L(θ) = P(x₁|θ) × P(x₂|θ) × ... × P(xₙ|θ)
     = ∏ᵢ P(xᵢ|θ)
```

**Why product?** Each observation is independent, so probability of all observations = product of individual probabilities.

#### Step 2: Log-Likelihood (Why We Use It)

**Problem:** Products are hard to work with (numerical issues, derivatives)

**Solution:** Take logarithm
```
log L(θ) = log ∏ᵢ P(xᵢ|θ)
         = Σᵢ log P(xᵢ|θ)
```

**Why this works:**
- log is monotonic: maximizing log L(θ) = maximizing L(θ)
- Products become sums (easier!)
- More numerically stable

#### Step 3: Maximize Log-Likelihood

**MLE estimate:**
```
θ̂_MLE = argmax_θ log L(θ)
      = argmax_θ Σᵢ log P(xᵢ|θ)
```

**How to find it:**
1. Take derivative with respect to θ
2. Set derivative to zero
3. Solve for θ

**Mathematical:**
```
∂/∂θ [Σᵢ log P(xᵢ|θ)] = 0
```

This gives us the maximum likelihood estimate.

### Example 1: MLE for Coin Flip (Bernoulli)

**Setup:**
- Data: n flips, k heads, (n-k) tails
- Model: P(heads) = θ, P(tails) = 1-θ
- Goal: Find θ that maximizes likelihood

**Step 1: Likelihood**
```
L(θ) = θᵏ × (1-θ)ⁿ⁻ᵏ

Why?
- k heads: each has probability θ → θᵏ
- (n-k) tails: each has probability (1-θ) → (1-θ)ⁿ⁻ᵏ
- Independent: multiply them
```

**Step 2: Log-Likelihood**
```
log L(θ) = log[θᵏ × (1-θ)ⁿ⁻ᵏ]
         = k log θ + (n-k) log(1-θ)
```

**Step 3: Take Derivative**
```
d/dθ [log L(θ)] = d/dθ [k log θ + (n-k) log(1-θ)]
                = k/θ - (n-k)/(1-θ)
```

**Step 4: Set to Zero**
```
k/θ - (n-k)/(1-θ) = 0
k/θ = (n-k)/(1-θ)
k(1-θ) = (n-k)θ
k - kθ = nθ - kθ
k = nθ
θ = k/n
```

**Result:**
```
θ̂_MLE = k/n = (number of heads) / (total flips)
```

**Intuition:**
The MLE estimate is simply the observed proportion! This makes sense: if you see 7 heads out of 10, the best estimate is 0.7.

### Example 2: MLE for Normal Distribution

**Setup:**
- Data: {x₁, x₂, ..., xₙ} from N(μ, σ²)
- Goal: Find μ and σ² that maximize likelihood

**Step 1: Likelihood**
```
L(μ, σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))
         = (1/√(2πσ²))ⁿ exp(-Σᵢ(xᵢ-μ)²/(2σ²))
```

**Step 2: Log-Likelihood**
```
log L(μ, σ²) = -n/2 log(2πσ²) - Σᵢ(xᵢ-μ)²/(2σ²)
```

**Step 3: Maximize with respect to μ**

Take derivative with respect to μ:
```
∂/∂μ [log L(μ, σ²)] = ∂/∂μ [-n/2 log(2πσ²) - Σᵢ(xᵢ-μ)²/(2σ²)]
                    = -1/(2σ²) × ∂/∂μ [Σᵢ(xᵢ-μ)²]
                    = -1/(2σ²) × Σᵢ 2(xᵢ-μ)(-1)
                    = 1/σ² × Σᵢ(xᵢ-μ)
```

Set to zero:
```
1/σ² × Σᵢ(xᵢ-μ) = 0
Σᵢ(xᵢ-μ) = 0
Σᵢ xᵢ - nμ = 0
μ = (1/n) Σᵢ xᵢ = x̄
```

**Result:**
```
μ̂_MLE = x̄ = (1/n) Σᵢ xᵢ  (sample mean)
```

**Step 4: Maximize with respect to σ²**

Take derivative with respect to σ²:
```
∂/∂σ² [log L(μ, σ²)] = -n/(2σ²) + Σᵢ(xᵢ-μ)²/(2(σ²)²)
```

Set to zero:
```
-n/(2σ²) + Σᵢ(xᵢ-μ)²/(2(σ²)²) = 0
-nσ² + Σᵢ(xᵢ-μ)² = 0
σ² = (1/n) Σᵢ(xᵢ-μ)²
```

**Result:**
```
σ̂²_MLE = (1/n) Σᵢ(xᵢ-μ̂)²  (sample variance, biased)
```

**Note:** This is the biased estimator. The unbiased version divides by (n-1) instead of n.

**Intuition:**
- MLE for mean = sample mean (makes sense!)
- MLE for variance = average squared deviation from mean

### Example 3: MLE for Linear Regression

**Setup:**
- Model: y = Xw + ε, where ε ~ N(0, σ²)
- Data: {(x₁, y₁), ..., (xₙ, yₙ)}
- Goal: Find w that maximizes likelihood

**Step 1: Likelihood**

For each data point:
```
P(yᵢ|xᵢ, w, σ²) = (1/√(2πσ²)) exp(-(yᵢ - xᵢᵀw)²/(2σ²))
```

For all data (independent):
```
L(w, σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(yᵢ - xᵢᵀw)²/(2σ²))
```

**Step 2: Log-Likelihood**
```
log L(w, σ²) = -n/2 log(2πσ²) - (1/(2σ²)) Σᵢ(yᵢ - xᵢᵀw)²
```

**Step 3: Maximize with respect to w**

Since σ² doesn't depend on w, we can ignore the first term:
```
argmax_w log L(w, σ²) = argmax_w [- (1/(2σ²)) Σᵢ(yᵢ - xᵢᵀw)²]
                     = argmin_w [Σᵢ(yᵢ - xᵢᵀw)²]
                     = argmin_w ||y - Xw||²
```

**Result:**
```
ŵ_MLE = (XᵀX)⁻¹Xᵀy  (ordinary least squares!)
```

**Key Insight:**
MLE for linear regression with Gaussian noise = Ordinary Least Squares (OLS)!

**Intuition:**
- We assume errors are normally distributed
- Maximizing likelihood = minimizing sum of squared errors
- This is exactly what OLS does!

---

## Maximum A Posteriori (MAP)

### Intuitive Explanation

**The Core Idea:**

MLE asks: "What parameters make the data most likely?"

MAP asks: "What parameters are most likely given both the data AND my prior beliefs?"

**Example:**
- You flip a coin 3 times and get 3 heads
- MLE says: p = 1.0 (100% heads) - but this seems wrong!
- MAP says: "Wait, I know coins are usually fair (p ≈ 0.5), so maybe p = 0.7" - incorporates prior knowledge

**Key Insight:**
MAP = MLE + Prior Beliefs

### Mathematical Setup

**Bayes' Theorem:**
```
P(θ|D) = P(D|θ) × P(θ) / P(D)

Where:
- P(θ|D): Posterior (what we want)
- P(D|θ): Likelihood (same as MLE)
- P(θ): Prior (our beliefs before seeing data)
- P(D): Evidence (normalizing constant, doesn't depend on θ)
```

**MAP Estimate:**
```
θ̂_MAP = argmax_θ P(θ|D)
      = argmax_θ P(D|θ) × P(θ)  (P(D) doesn't depend on θ)
      = argmax_θ [log P(D|θ) + log P(θ)]
      = argmax_θ [log L(θ) + log P(θ)]
```

**Key Insight:**
MAP = MLE + log(prior)

### Detailed Derivation

#### Step 1: Write the Posterior

```
P(θ|D) ∝ P(D|θ) × P(θ)
      ∝ L(θ) × P(θ)
```

**Why proportional?** P(D) is constant with respect to θ, so we can ignore it when maximizing.

#### Step 2: Log-Posterior

```
log P(θ|D) = log L(θ) + log P(θ) + constant
```

#### Step 3: Maximize Log-Posterior

```
θ̂_MAP = argmax_θ [log L(θ) + log P(θ)]
```

**How to find it:**
1. Take derivative with respect to θ
2. Set derivative to zero
3. Solve for θ

**Mathematical:**
```
∂/∂θ [log L(θ) + log P(θ)] = 0
```

### Example 1: MAP for Coin Flip with Beta Prior

**Setup:**
- Data: n flips, k heads
- Prior: θ ~ Beta(α, β) (conjugate prior for Bernoulli)
- Goal: Find θ that maximizes posterior

**Step 1: Prior**
```
P(θ) = (θ^(α-1) × (1-θ)^(β-1)) / B(α, β)

Where B(α, β) is the Beta function (normalizing constant)
```

**Step 2: Likelihood (same as MLE)**
```
L(θ) = θᵏ × (1-θ)ⁿ⁻ᵏ
```

**Step 3: Posterior**
```
P(θ|D) ∝ L(θ) × P(θ)
      ∝ θᵏ × (1-θ)ⁿ⁻ᵏ × θ^(α-1) × (1-θ)^(β-1)
      ∝ θ^(k+α-1) × (1-θ)^(n-k+β-1)
```

**This is Beta(k+α, n-k+β)!**

**Step 4: MAP Estimate**

Mode of Beta(α', β') is:
```
θ_mode = (α' - 1) / (α' + β' - 2)
```

So:
```
θ̂_MAP = (k + α - 1) / (n + α + β - 2)
```

**Special Case: Uniform Prior (α=1, β=1)**
```
θ̂_MAP = (k + 1 - 1) / (n + 1 + 1 - 2)
      = k / n
      = θ̂_MLE
```

**Intuition:**
- With uniform prior (no prior beliefs), MAP = MLE
- With informative prior, MAP incorporates prior knowledge
- Prior acts like "pseudo-observations": α-1 heads, β-1 tails

### Example 2: MAP for Linear Regression with Gaussian Prior

**Setup:**
- Model: y = Xw + ε, where ε ~ N(0, σ²)
- Prior: w ~ N(0, σ²_prior I)
- Goal: Find w that maximizes posterior

**Step 1: Prior**
```
P(w) ∝ exp(-||w||²/(2σ²_prior))
```

**Step 2: Likelihood (same as MLE)**
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
Xᵀy - XᵀXw = -λw  (where λ = σ²/σ²_prior)
Xᵀy = (XᵀX + λI)w
w = (XᵀX + λI)⁻¹Xᵀy
```

**Result:**
```
ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy  (Ridge regression!)
```

**Key Insight:**
MAP for linear regression with Gaussian prior = Ridge regression (L2 regularization)!

**Intuition:**
- Prior: w ~ N(0, σ²_prior) means we believe parameters should be small
- This is exactly L2 regularization!
- λ = σ²/σ²_prior controls regularization strength

---

## Connection: MLE vs MAP

### Mathematical Relationship

**MLE:**
```
θ̂_MLE = argmax_θ log L(θ)
      = argmax_θ log P(D|θ)
```

**MAP:**
```
θ̂_MAP = argmax_θ [log L(θ) + log P(θ)]
      = argmax_θ [log P(D|θ) + log P(θ)]
      = MLE + Prior
```

**Key Insight:**
MAP = MLE + log(prior)

### When They're the Same

**1. Uniform Prior:**
- P(θ) = constant
- log P(θ) = constant
- MAP = MLE (prior doesn't affect optimization)

**2. Large Dataset:**
- Data overwhelms prior
- MAP ≈ MLE (data dominates)

### When They Differ

**1. Small Dataset:**
- Prior has more influence
- MAP can be very different from MLE

**2. Strong Prior:**
- Prior strongly influences result
- MAP pulled toward prior mean

### Regularization Connection

**L2 Regularization (Ridge):**
```
Loss = MSE + λ||w||²
     = -log L(w) + λ||w||²
     = -[log L(w) + log P(w)]  (where P(w) ∝ exp(-λ||w||²))
```

**This is exactly MAP with Gaussian prior!**

**L1 Regularization (Lasso):**
```
Loss = MSE + λ||w||₁
     = -[log L(w) + log P(w)]  (where P(w) ∝ exp(-λ||w||₁))
```

**This is exactly MAP with Laplace prior!**

**Key Insight:**
Regularization = Bayesian prior
Regularized optimization = MAP estimation

---

## Examples

### Example 1: Coin Flip Comparison

**Data:** 3 flips, 3 heads

**MLE:**
```
θ̂_MLE = 3/3 = 1.0  (100% heads)
```

**MAP with Beta(2, 2) prior (slightly favors fair coin):**
```
θ̂_MAP = (3 + 2 - 1) / (3 + 2 + 2 - 2) = 4/5 = 0.8
```

**MAP with Beta(10, 10) prior (strongly favors fair coin):**
```
θ̂_MAP = (3 + 10 - 1) / (3 + 10 + 10 - 2) = 12/21 ≈ 0.57
```

**Intuition:**
- MLE: Only looks at data → extreme estimate
- MAP: Incorporates prior → more reasonable estimate
- Stronger prior → closer to prior mean (0.5)

### Example 2: Linear Regression Comparison

**Data:** Small dataset, many features

**MLE (OLS):**
```
ŵ_MLE = (XᵀX)⁻¹Xᵀy
```
- Can overfit with small data
- Parameters can be very large

**MAP (Ridge):**
```
ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy
```
- Prior shrinks parameters toward 0
- Prevents overfitting
- More stable with small data

---

## When to Use Each

### Use MLE When:

1. **Large dataset**: Data overwhelms any prior
2. **No prior knowledge**: Don't have strong beliefs
3. **Computational simplicity**: MLE is simpler
4. **Frequentist approach**: Want point estimates only

### Use MAP When:

1. **Small dataset**: Need to incorporate prior knowledge
2. **Strong prior beliefs**: Have domain knowledge
3. **Regularization needed**: Want to prevent overfitting
4. **Bayesian approach**: Want to incorporate uncertainty

### Practical Guidelines:

**For most ML problems:**
- **Large data**: MLE or MAP with weak prior (≈ MLE)
- **Small data**: MAP with informative prior
- **Regularization**: MAP (regularization = prior)

**For research:**
- **Frequentist**: MLE
- **Bayesian**: MAP (or full Bayesian inference)

---

## Summary

**MLE:**
- Maximizes P(data|parameters)
- Frequentist approach
- No prior beliefs
- Simple, data-driven

**MAP:**
- Maximizes P(parameters|data)
- Bayesian approach
- Incorporates prior beliefs
- MAP = MLE + Prior

**Connection:**
- Regularization = Bayesian prior
- Ridge = MAP with Gaussian prior
- Lasso = MAP with Laplace prior

**Key Insight:**
Understanding MLE and MAP helps understand:
- How models learn (MLE)
- Why regularization works (MAP)
- Bayesian vs Frequentist thinking

