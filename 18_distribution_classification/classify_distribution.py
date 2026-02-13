"""
Distribution Classification
Interview question: "Given 2 distributions and a new number, which distribution?"
"""
import numpy as np
from scipy import stats

def classify_by_likelihood(new_value: float, 
                       dist1_samples: np.ndarray,
                       dist2_samples: np.ndarray) -> int:
    """
    Classify new value using likelihood ratio
    
    Simple approach: Compare likelihoods
    
    Args:
        new_value: The new number to classify
        dist1_samples: Samples from distribution 1
        dist2_samples: Samples from distribution 2
    Returns:
        1 if from dist1, 2 if from dist2
    """
    # Estimate parameters from samples (assuming normal distribution)
    mu1, sigma1 = np.mean(dist1_samples), np.std(dist1_samples)
    mu2, sigma2 = np.mean(dist2_samples), np.std(dist2_samples)
    
    # Compute likelihoods
    likelihood1 = stats.norm.pdf(new_value, mu1, sigma1)
    likelihood2 = stats.norm.pdf(new_value, mu2, sigma2)
    
    # Return distribution with higher likelihood
    return 1 if likelihood1 > likelihood2 else 2

def classify_by_mle(new_value: float,
                    dist1_samples: np.ndarray,
                    dist2_samples: np.ndarray) -> int:
    """
    Classify using maximum likelihood estimation
    
    Fits distributions to samples, then computes log-likelihood
    """
    # Fit distributions to samples
    params1 = stats.norm.fit(dist1_samples)  # (mu, sigma)
    params2 = stats.norm.fit(dist2_samples)
    
    # Compute log-likelihoods
    log_likelihood1 = stats.norm.logpdf(new_value, *params1)
    log_likelihood2 = stats.norm.logpdf(new_value, *params2)
    
    # Return distribution with higher log-likelihood
    return 1 if log_likelihood1 > log_likelihood2 else 2

def classify_bayesian(new_value: float,
                      dist1_samples: np.ndarray,
                      dist2_samples: np.ndarray,
                      prior1: float = 0.5,
                      prior2: float = 0.5) -> int:
    """
    Classify using Bayesian approach
    
    P(dist|value) ∝ P(value|dist) × P(dist)
    
    Uses prior probabilities (can be different if distributions
    have different frequencies)
    """
    # Estimate parameters
    mu1, sigma1 = np.mean(dist1_samples), np.std(dist1_samples)
    mu2, sigma2 = np.mean(dist2_samples), np.std(dist2_samples)
    
    # Compute likelihoods
    likelihood1 = stats.norm.pdf(new_value, mu1, sigma1)
    likelihood2 = stats.norm.pdf(new_value, mu2, sigma2)
    
    # Multiply by priors
    posterior1 = likelihood1 * prior1
    posterior2 = likelihood2 * prior2
    
    # Normalize (optional, but doesn't change comparison)
    total = posterior1 + posterior2
    posterior1 = posterior1 / total
    posterior2 = posterior2 / total
    
    return 1 if posterior1 > posterior2 else 2

def classify_with_confidence(new_value: float,
                            dist1_samples: np.ndarray,
                            dist2_samples: np.ndarray) -> tuple:
    """
    Classify and return confidence
    
    Returns: (distribution, confidence)
    """
    mu1, sigma1 = np.mean(dist1_samples), np.std(dist1_samples)
    mu2, sigma2 = np.mean(dist2_samples), np.std(dist2_samples)
    
    likelihood1 = stats.norm.pdf(new_value, mu1, sigma1)
    likelihood2 = stats.norm.pdf(new_value, mu2, sigma2)
    
    total = likelihood1 + likelihood2
    confidence1 = likelihood1 / total
    confidence2 = likelihood2 / total
    
    if confidence1 > confidence2:
        return (1, confidence1)
    else:
        return (2, confidence2)


# Usage Example
if __name__ == "__main__":
    print("Distribution Classification")
    print("=" * 60)
    
    # Generate samples from two distributions
    np.random.seed(42)
    dist1_samples = np.random.normal(0, 1, 1000)  # Mean=0, Std=1
    dist2_samples = np.random.normal(5, 1, 1000)   # Mean=5, Std=1
    
    # New value to classify
    new_value = 2.0
    
    print(f"Distribution 1: Mean={np.mean(dist1_samples):.2f}, Std={np.std(dist1_samples):.2f}")
    print(f"Distribution 2: Mean={np.mean(dist2_samples):.2f}, Std={np.std(dist2_samples):.2f}")
    print(f"New value: {new_value}")
    print()
    
    # Classify using different methods
    result1 = classify_by_likelihood(new_value, dist1_samples, dist2_samples)
    print(f"Likelihood ratio: Distribution {result1}")
    
    result2 = classify_by_mle(new_value, dist1_samples, dist2_samples)
    print(f"Maximum likelihood: Distribution {result2}")
    
    result3 = classify_bayesian(new_value, dist1_samples, dist2_samples)
    print(f"Bayesian: Distribution {result3}")
    
    result4, confidence = classify_with_confidence(new_value, dist1_samples, dist2_samples)
    print(f"With confidence: Distribution {result4} (confidence: {confidence:.4f})")

