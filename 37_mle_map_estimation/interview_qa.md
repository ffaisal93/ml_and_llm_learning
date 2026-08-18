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

> **Saying it out loud.** For a coin, maximum likelihood gives you exactly what your gut says: the fraction of flips that came up heads. The derivation is four lines — write the likelihood as theta to the heads times one-minus-theta to the tails, take the log so the product becomes a sum, differentiate, set it to zero, and $k/n$ falls out. I'd say it that way because the interviewer wants to see that you can go from a probability model to an estimator mechanically. The part worth adding unprompted is the failure mode: with three flips and three heads, MLE confidently reports a probability of 1.0, which is nonsense, and that's precisely the gap a prior fills — Beta(1,1) turns that into 4/5.

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

> **Saying it out loud.** The punchline is that least squares isn't an arbitrary choice of loss — it's what you get when you assume Gaussian noise and do maximum likelihood. Write the likelihood of the residuals under a Gaussian, take the log, and everything except minus one over two sigma squared times the sum of squared residuals is a constant that doesn't touch $w$. So maximizing likelihood is minimizing squared error, and setting the gradient to zero gives the normal equations, $w = (X^\top X)^{-1}X^\top y$. That reframing is useful because it tells you when to stop using squared loss: if your errors have heavy tails, the Gaussian assumption is wrong and a single outlier, once squared, can drag the whole fit. Swap to a Laplace noise model and you get absolute error instead.

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

> **Saying it out loud.** MAP is MLE plus a prior, and because you're working in log space the prior shows up as an additive term rather than a multiplication. That single fact is why every regularizer you use has a Bayesian reading: L2 is a Gaussian prior centered at zero, L1 is a Laplace prior. Two boundary cases are worth stating because they show you understand the mechanics. With a uniform prior the log-prior is a constant, so MAP is identical to MLE. And as the dataset grows, the likelihood term grows with $n$ while the prior stays fixed, so MAP converges to MLE — the data drowns out your opinion. The practical read: priors matter enormously with a hundred examples and are essentially decorative with ten million.

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

> **Saying it out loud.** Ridge regression is MAP with a zero-mean Gaussian prior on the weights. The log of that prior is minus the squared norm of $w$ over twice the prior variance, so adding it to the Gaussian log-likelihood gives you squared error plus an L2 penalty, and setting the gradient to zero gives $(X^\top X + \lambda I)^{-1}X^\top y$. The satisfying part is where $\lambda$ comes from: it's the noise variance divided by the prior variance. So a tight prior means heavy regularization, and noisy data means heavy regularization, both for reasons you can state in English. And there's a numerical bonus worth mentioning — adding $\lambda I$ makes the matrix invertible even when features are perfectly collinear, which is exactly when plain OLS blows up.

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

> **Saying it out loud.** Three reasons, and the first one is the one that actually bites you in code. Multiply a thousand probabilities together and you underflow to exactly zero in floating point — a product of a thousand values around 0.1 is $10^{-1000}$, and double precision bottoms out near $10^{-308}$. Take logs and it's a sum around $-2300$, perfectly representable. Second, logs turn products into sums, so derivatives are term-by-term and you never touch the product rule. Third, and this is what makes it legal, log is monotonically increasing, so whatever maximizes the log-likelihood also maximizes the likelihood — you get the numerical benefits for free without changing the answer.

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

> **Saying it out loud.** The practical difference shows up entirely in the small-data regime. MLE has no opinion, so it commits fully to whatever it saw — three heads out of three means probability one. MAP brings a prior that acts like extra pseudo-observations, which pulls the estimate back toward something sane. Once you have a lot of data the two converge, since the likelihood scales with $n$ and the prior doesn't, so the argument becomes moot. Computationally they cost the same; MAP is just one extra additive term in the objective. The honest limitation worth naming: MAP is still a single point estimate, the mode of the posterior. It does not give you uncertainty. If you need error bars you have to go full Bayesian and carry the whole posterior.

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

> **Saying it out loud.** The cleanest way to say this: your regularized loss is just the negative log posterior. Data-fit term is the negative log-likelihood, penalty term is the negative log-prior, and minimizing the sum is finding the posterior mode. From there L2 and L1 stop being two arbitrary penalties and become two different beliefs about weights — a Gaussian prior says weights are small but nonzero, a Laplace prior has a sharp spike at zero, which is why lasso actually drives coefficients to exactly zero and ridge only shrinks them. And $\lambda$ isn't a magic knob, it's the ratio of noise variance to prior variance, so turning it up literally means "I trust my prior more than my data." That's a much better story than "I grid-searched it."

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

