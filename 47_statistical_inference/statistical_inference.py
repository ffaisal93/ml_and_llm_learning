"""
Statistical inference boilerplate for ML interviews.
Short functions, direct formulas, easy to rewrite under pressure.
"""
from typing import Callable, Iterable, Tuple

import numpy as np
from scipy import stats


def sample_mean(x: np.ndarray) -> float:
    """Sample mean."""
    return float(np.mean(x))


def sample_variance(x: np.ndarray, unbiased: bool = True) -> float:
    """
    Sample variance.

    unbiased=True uses ddof=1, which divides by n - 1.
    unbiased=False uses ddof=0, which divides by n.
    """
    ddof = 1 if unbiased else 0
    return float(np.var(x, ddof=ddof))


def bernoulli_mle(samples: np.ndarray) -> float:
    """MLE for Bernoulli(p) is just the sample mean."""
    return float(np.mean(samples))


def gaussian_mle(samples: np.ndarray) -> Tuple[float, float]:
    """
    MLE for Gaussian mean and variance.

    Variance uses division by n, not n - 1.
    """
    mu_hat = float(np.mean(samples))
    var_hat = float(np.mean((samples - mu_hat) ** 2))
    return mu_hat, var_hat


def mean_confidence_interval(
    samples: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Large-sample confidence interval for the mean using the t distribution.
    """
    n = len(samples)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    se = std / np.sqrt(n)
    alpha = 1.0 - confidence
    critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    margin = critical * se
    return float(mean - margin), float(mean + margin)


def bootstrap_confidence_interval(
    samples: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = sample_mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for any 1D statistic.
    """
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(samples)

    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(samples, size=n, replace=True)
        estimates.append(statistic_fn(bootstrap_sample))

    alpha = 1.0 - confidence
    lower = np.percentile(estimates, 100.0 * alpha / 2.0)
    upper = np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def two_sample_t_test(x: np.ndarray, y: np.ndarray, equal_var: bool = False) -> Tuple[float, float]:
    """
    Two-sample t-test.

    Returns:
        t_statistic, p_value
    """
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
    return float(t_stat), float(p_value)


def beta_bernoulli_posterior(
    heads: int, tails: int, alpha: float = 1.0, beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    Beta-Bernoulli conjugate update.

    Returns:
        posterior_alpha, posterior_beta, posterior_mean
    """
    posterior_alpha = alpha + heads
    posterior_beta = beta + tails
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    return float(posterior_alpha), float(posterior_beta), float(posterior_mean)


def estimate_bias_variance(
    estimator_fn: Callable[[np.ndarray], float],
    data_generator: Callable[[np.random.Generator, int], np.ndarray],
    true_value: float,
    n_samples: int = 50,
    n_trials: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Monte Carlo estimate of estimator bias and variance.
    """
    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n_trials):
        sample = data_generator(rng, n_samples)
        estimates.append(estimator_fn(sample))

    estimates = np.array(estimates)
    bias = float(np.mean(estimates) - true_value)
    variance = float(np.var(estimates))
    return bias, variance


if __name__ == "__main__":
    np.random.seed(42)

    print("Statistical Inference Boilerplate")
    print("=" * 60)

    x = np.array([2.1, 2.5, 1.9, 2.7, 2.4, 2.0, 2.3])
    print(f"Samples: {x}")
    print(f"Sample mean: {sample_mean(x):.4f}")
    print(f"Sample variance (unbiased): {sample_variance(x, unbiased=True):.4f}")
    print(f"Sample variance (MLE): {sample_variance(x, unbiased=False):.4f}")

    bernoulli_samples = np.array([1, 0, 1, 1, 0, 1, 1, 1])
    print("\nBernoulli MLE")
    print(f"Observed samples: {bernoulli_samples}")
    print(f"p_hat: {bernoulli_mle(bernoulli_samples):.4f}")

    gaussian_samples = np.array([1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0])
    mu_hat, var_hat = gaussian_mle(gaussian_samples)
    print("\nGaussian MLE")
    print(f"mu_hat: {mu_hat:.4f}")
    print(f"var_hat: {var_hat:.4f}")

    ci = mean_confidence_interval(x, confidence=0.95)
    boot_ci = bootstrap_confidence_interval(x, statistic_fn=sample_mean, n_bootstrap=2000)
    print("\nConfidence Intervals for the Mean")
    print(f"t-based 95% CI: {ci}")
    print(f"Bootstrap 95% CI: {boot_ci}")

    group_a = np.array([0.81, 0.78, 0.83, 0.79, 0.82, 0.80])
    group_b = np.array([0.76, 0.75, 0.77, 0.74, 0.78, 0.76])
    t_stat, p_value = two_sample_t_test(group_a, group_b, equal_var=False)
    print("\nTwo-Sample t-test")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    post_alpha, post_beta, post_mean = beta_bernoulli_posterior(
        heads=7, tails=3, alpha=2.0, beta=2.0
    )
    print("\nBeta-Bernoulli Posterior")
    print(f"posterior alpha: {post_alpha:.1f}")
    print(f"posterior beta: {post_beta:.1f}")
    print(f"posterior mean: {post_mean:.4f}")

    bias, variance = estimate_bias_variance(
        estimator_fn=sample_mean,
        data_generator=lambda rng, n: rng.normal(loc=3.0, scale=2.0, size=n),
        true_value=3.0,
    )
    print("\nMonte Carlo Bias / Variance of Sample Mean")
    print(f"Estimated bias: {bias:.4f}")
    print(f"Estimated variance: {variance:.4f}")
