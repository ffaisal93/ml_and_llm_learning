# Topic 18: Distribution Classification

> 🔥 **For interviews, read these first:**
> - **`DISTRIBUTIONS_DEEP_DIVE.md`** — frontier-lab deep dive: which distribution to use for which data type, exponential family unification, GLMs and canonical links, heavy-tailed distributions, common pitfalls.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic teaches you:
- Given a number, determine which distribution it came from
- Likelihood ratio test
- Maximum likelihood estimation
- Simple classification methods

## Why We Need This

### Interview Importance
- **Common question**: "Given data, which distribution?"
- **Statistical reasoning**: Shows statistical knowledge
- **Problem-solving**: Practical ML skill

### Real-World Application
- **Data analysis**: Understand data distribution
- **Model selection**: Choose right distribution
- **Anomaly detection**: Identify outliers

## Core Intuition

This is a probabilistic classification problem.

You have:
- two candidate data-generating sources
- one new observation

You want to know which source makes that observation more plausible.

That is why likelihood or posterior comparison is the right mental model, not just mean comparison.

### Why Means Are Not Enough

Two distributions can have:
- the same mean
- different variances
- different tail behavior

A point can be closer to one mean but still more likely under another density.

## Problem Statement

**Question**: You have 2 lists from 2 distributions. You get a new number. How do you determine which distribution it belongs to?

**Short Answer**: Use a generative classifier.

That means:
1. estimate each distribution from its samples
2. compute how likely the new value is under each distribution
3. include class priors if one distribution is more common
4. choose the distribution with the larger posterior score

In symbols:

`choose dist1 if p(x | dist1) * P(dist1) > p(x | dist2) * P(dist2)`

This is the cleanest interview answer because it shows:
- probabilistic reasoning
- awareness of assumptions
- understanding of priors

## Technical Details Interviewers Often Want

### Likelihood vs Posterior

If priors are equal, likelihood comparison is enough.

If priors differ, posterior comparison is more correct.

### Parametric vs Nonparametric

Parametric:
- assume a family like Gaussian
- estimate a few parameters

Nonparametric:
- estimate the density more flexibly with KDE or related methods

## How to Answer This in an Interview

Say it in this order:

### Case 1: If I Am Comfortable Assuming a Distribution Family

If I assume both arrays come from Gaussians:
- fit mean and variance for each array
- compute Gaussian density of the new value under each fitted distribution
- if priors are equal, choose the larger likelihood
- if priors differ, choose the larger posterior score

### Case 2: If I Do Not Want a Parametric Assumption

If I do not want to assume Gaussian:
- estimate density nonparametrically with KDE
- or use a simple nearest-neighbor density intuition in 1D
- then compare the estimated densities

### Case 3: If the Distributions Overlap Heavily

Then classification may be ambiguous.

In that case, report:
- predicted class
- posterior/confidence
- the fact that the point lies in an overlapping region

This is a stronger answer than pretending every point can be classified confidently.

## Industry-Standard Boilerplate Code

### Likelihood Ratio Test

```python
"""
Likelihood Ratio Test: Compare likelihoods
"""
import numpy as np
from scipy import stats

def classify_by_likelihood(new_value: float, 
                           dist1_samples: np.ndarray,
                           dist2_samples: np.ndarray) -> int:
    """
    Classify new value using likelihood ratio
    
    Returns: 1 if from dist1, 2 if from dist2
    """
    # Estimate parameters from samples
    mu1, sigma1 = np.mean(dist1_samples), np.std(dist1_samples)
    mu2, sigma2 = np.mean(dist2_samples), np.std(dist2_samples)
    
    # Compute likelihoods
    likelihood1 = stats.norm.pdf(new_value, mu1, sigma1)
    likelihood2 = stats.norm.pdf(new_value, mu2, sigma2)
    
    # Return distribution with higher likelihood
    return 1 if likelihood1 > likelihood2 else 2
```

### Maximum Likelihood Classification

```python
"""
Maximum Likelihood: Choose distribution with max likelihood
"""
def classify_by_mle(new_value: float,
                    dist1_samples: np.ndarray,
                    dist2_samples: np.ndarray) -> int:
    """
    Classify using maximum likelihood estimation
    """
    # Fit distributions to samples
    params1 = stats.norm.fit(dist1_samples)
    params2 = stats.norm.fit(dist2_samples)
    
    # Compute log-likelihoods
    log_likelihood1 = stats.norm.logpdf(new_value, *params1)
    log_likelihood2 = stats.norm.logpdf(new_value, *params2)
    
    # Return distribution with higher log-likelihood
    return 1 if log_likelihood1 > log_likelihood2 else 2
```

### Bayesian Classification

```python
"""
Bayesian: Use prior probabilities
"""
def classify_bayesian(new_value: float,
                     dist1_samples: np.ndarray,
                     dist2_samples: np.ndarray,
                     prior1: float = 0.5,
                     prior2: float = 0.5) -> int:
    """
    Classify using Bayesian approach
    
    P(dist|value) ∝ P(value|dist) × P(dist)
    """
    # Estimate parameters
    mu1, sigma1 = np.mean(dist1_samples), np.std(dist1_samples)
    mu2, sigma2 = np.mean(dist2_samples), np.std(dist2_samples)
    
    # Compute posterior probabilities
    likelihood1 = stats.norm.pdf(new_value, mu1, sigma1) * prior1
    likelihood2 = stats.norm.pdf(new_value, mu2, sigma2) * prior2
    
    # Normalize
    total = likelihood1 + likelihood2
    posterior1 = likelihood1 / total
    posterior2 = likelihood2 / total
    
    return 1 if posterior1 > posterior2 else 2
```

## Theory

### Likelihood Ratio
- Compare P(x|dist1) vs P(x|dist2)
- Choose distribution with higher likelihood
- Simple and effective

### Maximum Likelihood
- Fit distributions to samples
- Compute log-likelihood for new value
- Choose distribution with higher log-likelihood

### Bayesian Approach
- Use prior probabilities
- Compute posterior: P(dist|x) ∝ P(x|dist) × P(dist)
- More sophisticated

### Important Interview Follow-Ups

#### What assumptions are hidden here?

You are assuming:
- the training arrays are representative samples
- the fitted family is reasonable if you use a parametric model
- the new value is generated from one of those two candidate distributions

#### What if both distributions have the same mean?

Then variance still matters.

Example:
- one distribution may be narrow and concentrated
- the other may be wide and diffuse

A point near the shared mean may be more likely under the narrow one.
A point far from the mean may be more likely under the wide one.

#### What if you only have a few samples?

Then parameter estimates are noisy.

Good answer:
- say confidence should be lower
- consider Bayesian priors or bootstrap uncertainty
- avoid overconfident claims

#### What if the distributions are unknown?

Then use:
- KDE
- histogram density estimate
- nearest-neighbor density idea

The principle is still the same: compare estimated densities.

## Common Failure Modes

- comparing only means instead of densities
- ignoring class priors
- being overconfident when distributions overlap heavily
- assuming Gaussian without stating the assumption
- forgetting how noisy the answer can be with few samples

## What to Practice Saying Out Loud

1. Why this is really a Bayes classification problem
2. Why density comparison is stronger than mean comparison
3. Why overlap and priors both matter

## Exercises

1. Implement likelihood ratio test
2. Compare different methods
3. Test on various distributions
4. Handle edge cases
5. Practice the spoken answer for the exact interview question above

## Next Steps

- **Topic 19**: Advanced clustering
- **Topic 20**: Multi-turn conversations
