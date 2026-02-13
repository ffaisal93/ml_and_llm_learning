# Topic 18: Distribution Classification

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

## Problem Statement

**Question**: You have 2 lists from 2 distributions. You get a new number. How do you determine which distribution it belongs to?

**Answer**: Use likelihood ratio or maximum likelihood

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

## Exercises

1. Implement likelihood ratio test
2. Compare different methods
3. Test on various distributions
4. Handle edge cases

## Next Steps

- **Topic 19**: Advanced clustering
- **Topic 20**: Multi-turn conversations

