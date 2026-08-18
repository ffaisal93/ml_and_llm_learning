# MLE and MAP: Detailed Derivations with Intuitive Explanations

## Table of Contents

1. [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
2. [Maximum A Posteriori (MAP)](#maximum-a-posteriori-map)
3. [Connection: MLE vs MAP](#connection-mle-vs-map)
4. [Examples](#examples)
5. [When to Use Each](#when-to-use-each)

---

## Maximum Likelihood Estimation (MLE)

*In plain language:* this whole section is one idea repeated on four different models. You have a knob (the parameter), you turn it until the data you actually collected looks as unsurprising as possible, and wherever the knob stops is your estimate. The calculus below is only the mechanical way of finding where it stops — take the log, differentiate, set to zero, solve.

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

> **Saying it out loud.** The recipe never changes, so it's worth being able to recite it as a procedure. Write down the probability of your whole dataset as a function of the parameter — that's a product, because the observations are independent. Take the log, which turns the product into a sum and keeps the maximum in the same place because log is monotonic. Differentiate the sum, set it equal to zero, solve. That's it; every derivation in this file is that same four-step loop. The reason the log step is non-negotiable in practice is numerical: multiplying a thousand probabilities underflows to exactly zero in double precision, since the smallest representable value is around $10^{-308}$.

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

> **Saying it out loud.** The coin flip is the derivation to have completely automatic. Likelihood is theta to the number of heads times one-minus-theta to the number of tails. Log it, and you get heads times log theta plus tails times log one-minus-theta. Differentiate, set to zero, and after one line of algebra you get theta equals heads over total flips — the observed proportion, exactly what anyone would have guessed. That's the reassuring part. The part I'd volunteer without being asked is the failure: three flips, three heads, and MLE says the coin never lands tails. It's the honest maximum-likelihood answer and it's a terrible prediction, which is the entire motivation for the MAP section.

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

> **Saying it out loud.** For a Gaussian you're fitting two parameters, and they come out in sequence. The mean estimate is just the sample average. Plug that back in and the variance estimate is the average squared distance from that average — with a $1/n$ in front, not the $1/(n-1)$ from your statistics course. That gap is the interesting part and it's a favorite follow-up. The MLE variance is biased low, because you're measuring spread around the sample mean, and the sample mean is by definition the point sitting closest to your own data — closer than the true mean would be. So your measured spread is systematically a little too small, and dividing by $n-1$ instead is Bessel's correction fixing exactly that.

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

> **Saying it out loud.** This is the derivation that changes how you see loss functions. Assume the targets are a linear function of the inputs plus Gaussian noise. Write the likelihood of the observed residuals, take the log, and every term except the sum of squared residuals is a constant that has nothing to do with $w$. So maximizing likelihood *is* minimizing squared error, and solving gives the normal equations. Least squares was never an arbitrary choice — it's the Gaussian assumption in disguise. And that tells you exactly when to abandon it: heavy-tailed noise breaks the assumption, a single outlier gets squared and dominates the fit, and the right move is a Laplace noise model, which hands you absolute error and the median instead of the mean.

---

## Maximum A Posteriori (MAP)

*In plain language:* MAP is the same procedure with one extra term. Before looking at the data you write down what you already believe about the parameter, and then you find the value that best balances your belief against the evidence. Because everything happens in log space, that belief shows up as something you simply add to the objective — which is why it ends up looking exactly like a regularization penalty.

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

> **Saying it out loud.** The Beta prior is the cleanest illustration of what a prior actually does, because it behaves like fake data you saw before the experiment. A Beta(α, β) prior on a coin acts like α−1 extra heads and β−1 extra tails already in your tally, so the MAP estimate is just the observed counts plus those pseudo-counts. That instantly fixes the three-heads-out-of-three disaster: the prior drags the answer back toward one half instead of letting it slam into 1.0. It also shows both boundary cases at once. A uniform Beta(1,1) contributes zero pseudo-counts, so MAP collapses back to MLE. And as real flips pile up, the fixed pseudo-counts become negligible, so MAP converges to MLE — the tradeoff is that a prior only buys you anything when data is scarce.

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

> **Saying it out loud.** Ridge regression is MAP with a Gaussian prior centered at zero, and the derivation is short enough to do live. The log of that prior is minus the squared norm of $w$ over twice the prior variance, so adding it to the Gaussian log-likelihood gives squared error plus an L2 penalty. Set the gradient to zero and you get $(X^\top X + \lambda I)^{-1}X^\top y$ — the OLS solution with a small amount added along the diagonal. The satisfying part is that lambda isn't a free knob, it's the noise variance divided by the prior variance. Noisy data means regularize harder; a confident prior means regularize harder. And there's a practical bonus: that $\lambda I$ makes the matrix invertible even with perfectly collinear features, which is precisely the case where plain OLS blows up.

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

> **Saying it out loud.** The single sentence to walk away with is that your regularized training objective is the negative log posterior. The data-fit term is the negative log-likelihood, the penalty term is the negative log-prior, and minimizing their sum is finding the mode of the posterior. That reframes L2 and L1 from arbitrary penalties into two different beliefs about weights: a Gaussian prior says weights are probably small, and a Laplace prior puts a sharp spike at zero, which is why lasso zeroes coefficients outright while ridge only shrinks them. The two cases where MAP and MLE coincide are worth naming too — a uniform prior, and a large dataset, since the likelihood grows with $n$ while the prior stays fixed.

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

> **Saying it out loud.** Side by side the difference is easiest to see on the two examples. On the coin, MLE gives you the raw proportion and MAP gives you the proportion with pseudo-counts added, which is what keeps small samples from producing absurd answers. On linear regression, MLE gives you OLS and MAP gives you ridge, which is OLS plus lambda on the diagonal. Same likelihood, one extra term, and the effect is entirely about stability. The number that makes it concrete: with two nearly collinear features, the OLS solution can have coefficients in the thousands that cancel each other out, and any tiny amount of ridge regularization collapses them to something sane with essentially no loss in fit.

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

> **Saying it out loud.** Practically, the choice almost makes itself. With a lot of data, use MLE, because the likelihood dominates and the prior is decoration. With little data, or when you genuinely know something about the parameters ahead of time, use MAP — and if you're using weight decay you're already doing MAP whether you call it that or not. The limitation to name so you don't get caught: MAP is still just a point estimate, the mode of the posterior, and it gives you no uncertainty at all. If someone needs a confidence interval or a risk estimate, neither MLE nor MAP delivers it, and you have to carry the full posterior with MCMC or variational inference, which costs orders of magnitude more compute.

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

