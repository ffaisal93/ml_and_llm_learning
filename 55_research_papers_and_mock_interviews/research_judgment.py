"""
Research interview helpers:
- compare experiments
- decide which distribution a new value likely came from
"""
from typing import Dict, Tuple

import numpy as np
from scipy import stats


def compare_experiments(
    baseline_scores: np.ndarray, candidate_scores: np.ndarray
) -> Dict[str, float]:
    """
    Simple paired experiment summary.
    """
    deltas = candidate_scores - baseline_scores
    return {
        "baseline_mean": float(np.mean(baseline_scores)),
        "candidate_mean": float(np.mean(candidate_scores)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "candidate_wins_fraction": float(np.mean(candidate_scores > baseline_scores)),
    }


def classify_value_gaussian_bayes(
    x: float,
    dist1_samples: np.ndarray,
    dist2_samples: np.ndarray,
    prior1: float = 0.5,
    prior2: float = 0.5,
) -> Dict[str, float]:
    """
    Bayes classifier under Gaussian assumptions.

    Returns posterior-like normalized scores and predicted class.
    """
    mu1, sigma1 = float(np.mean(dist1_samples)), float(np.std(dist1_samples, ddof=0))
    mu2, sigma2 = float(np.mean(dist2_samples)), float(np.std(dist2_samples, ddof=0))

    sigma1 = max(sigma1, 1e-6)
    sigma2 = max(sigma2, 1e-6)

    score1 = stats.norm.pdf(x, loc=mu1, scale=sigma1) * prior1
    score2 = stats.norm.pdf(x, loc=mu2, scale=sigma2) * prior2
    total = score1 + score2
    posterior1 = score1 / total
    posterior2 = score2 / total

    return {
        "mu1": mu1,
        "sigma1": sigma1,
        "mu2": mu2,
        "sigma2": sigma2,
        "posterior1": float(posterior1),
        "posterior2": float(posterior2),
        "predicted_class": 1.0 if posterior1 >= posterior2 else 2.0,
    }


def classify_value_kde(
    x: float,
    dist1_samples: np.ndarray,
    dist2_samples: np.ndarray,
    prior1: float = 0.5,
    prior2: float = 0.5,
) -> Dict[str, float]:
    """
    Nonparametric 1D density comparison using KDE.
    Useful when Gaussian assumptions are weak.
    """
    kde1 = stats.gaussian_kde(dist1_samples)
    kde2 = stats.gaussian_kde(dist2_samples)
    score1 = float(kde1(x)[0]) * prior1
    score2 = float(kde2(x)[0]) * prior2
    total = score1 + score2
    posterior1 = score1 / total
    posterior2 = score2 / total
    return {
        "posterior1": float(posterior1),
        "posterior2": float(posterior2),
        "predicted_class": 1.0 if posterior1 >= posterior2 else 2.0,
    }


if __name__ == "__main__":
    print("Research Judgment and Mock Interview Helpers")
    print("=" * 60)

    baseline = np.array([0.71, 0.72, 0.70, 0.73])
    candidate = np.array([0.74, 0.73, 0.71, 0.75])
    print("Experiment comparison:")
    print(compare_experiments(baseline, candidate))

    np.random.seed(42)
    dist1 = np.random.normal(loc=0.0, scale=1.0, size=400)
    dist2 = np.random.normal(loc=3.0, scale=1.2, size=400)
    x = 1.4
    print("\nGaussian Bayes classification:")
    print(classify_value_gaussian_bayes(x, dist1, dist2))
    print("\nKDE classification:")
    print(classify_value_kde(x, dist1, dist2))
