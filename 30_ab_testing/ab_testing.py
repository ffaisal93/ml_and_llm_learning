"""
A/B Testing for ML Models
Statistical testing and interpretation
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple

def calculate_sample_size(effect_size: float, alpha: float = 0.05,
                         power: float = 0.8, baseline_rate: float = 0.5) -> int:
    """
    Calculate required sample size for A/B test
    
    Formula:
    n = 2 × (Z_α/2 + Z_β)² × p(1-p) / (p_A - p_B)²
    
    Where:
    - effect_size: Minimum detectable effect (p_A - p_B)
    - alpha: Significance level (default 0.05)
    - power: Statistical power (default 0.8, means 80% chance of detecting effect)
    - baseline_rate: Baseline conversion rate (p)
    """
    from scipy.stats import norm
    
    # Z-scores
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)
    
    # Variance (for proportion)
    variance = baseline_rate * (1 - baseline_rate)
    
    # Sample size
    n = 2 * (z_alpha + z_beta)**2 * variance / (effect_size**2)
    
    return int(np.ceil(n))

def run_ab_test(control_data: np.ndarray, treatment_data: np.ndarray,
                metric_type: str = 'continuous') -> Dict:
    """
    Run A/B test and return results
    
    Args:
        control_data: Results from control group (A)
        treatment_data: Results from treatment group (B)
        metric_type: 'continuous' (e.g., revenue) or 'binary' (e.g., conversion)
    
    Returns:
        Dictionary with test results
    """
    if metric_type == 'continuous':
        # T-test for continuous metrics
        statistic, p_value = stats.ttest_ind(treatment_data, control_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_data) + np.var(treatment_data)) / 2
        )
        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
        
        # Confidence interval
        diff = np.mean(treatment_data) - np.mean(control_data)
        se = np.sqrt(
            np.var(control_data)/len(control_data) + 
            np.var(treatment_data)/len(treatment_data)
        )
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        
    else:  # binary
        # Chi-square test for binary metrics
        # Create contingency table
        control_success = np.sum(control_data)
        control_total = len(control_data)
        treatment_success = np.sum(treatment_data)
        treatment_total = len(treatment_data)
        
        contingency = np.array([
            [control_success, control_total - control_success],
            [treatment_success, treatment_total - treatment_success]
        ])
        
        statistic, p_value, _, _ = stats.chi2_contingency(contingency)
        
        # Effect size (difference in proportions)
        control_rate = control_success / control_total
        treatment_rate = treatment_success / treatment_total
        diff = treatment_rate - control_rate
        cohens_d = diff  # Simplified for binary
        
        # Confidence interval for difference in proportions
        se = np.sqrt(
            control_rate * (1 - control_rate) / control_total +
            treatment_rate * (1 - treatment_rate) / treatment_total
        )
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
    
    # Interpretation
    is_significant = p_value < 0.05
    effect_direction = 'positive' if diff > 0 else 'negative'
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'effect_size': diff,
        'cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'effect_direction': effect_direction,
        'control_mean': np.mean(control_data),
        'treatment_mean': np.mean(treatment_data),
        'control_size': len(control_data),
        'treatment_size': len(treatment_data)
    }

def interpret_ab_test_results(results: Dict) -> str:
    """
    Interpret A/B test results in plain English
    """
    interpretation = []
    
    # Significance
    if results['is_significant']:
        interpretation.append(f"Statistically significant (p={results['p_value']:.4f})")
    else:
        interpretation.append(f"Not statistically significant (p={results['p_value']:.4f})")
    
    # Effect
    effect_pct = (results['effect_size'] / results['control_mean']) * 100
    interpretation.append(
        f"Effect: {results['effect_direction']} {abs(effect_pct):.2f}% "
        f"({results['treatment_mean']:.4f} vs {results['control_mean']:.4f})"
    )
    
    # Confidence interval
    ci = results['confidence_interval']
    interpretation.append(
        f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
    )
    
    # Recommendation
    if results['is_significant'] and results['effect_direction'] == 'positive':
        interpretation.append("Recommendation: ROLLOUT - Significant positive effect")
    elif results['is_significant'] and results['effect_direction'] == 'negative':
        interpretation.append("Recommendation: DON'T ROLLOUT - Significant negative effect")
    else:
        interpretation.append("Recommendation: NEED MORE DATA - Not significant")
    
    return "\n".join(interpretation)

def multiple_testing_correction(p_values: list, method: str = 'bonferroni') -> list:
    """
    Apply multiple testing correction
    
    Methods:
    - bonferroni: Divide alpha by number of tests (conservative)
    - fdr_bh: Benjamini-Hochberg FDR (less conservative)
    """
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        # Bonferroni: p_adjusted = p × n_tests
        adjusted = p_values * len(p_values)
        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR
        from statsmodels.stats.multitest import multipletests
        _, adjusted, _, _ = multipletests(p_values, method='fdr_bh')
    else:
        adjusted = p_values
    
    return adjusted.tolist()


# Usage Example
if __name__ == "__main__":
    print("A/B Testing for ML Models")
    print("=" * 60)
    
    # Example: Testing new recommendation model
    print("\nExample: Testing New Recommendation Model")
    print("-" * 60)
    
    # Simulate data
    np.random.seed(42)
    n_control = 10000
    n_treatment = 10000
    
    # Control: 2.5% CTR
    control_ctr = np.random.binomial(1, 0.025, n_control)
    
    # Treatment: 2.8% CTR (12% improvement)
    treatment_ctr = np.random.binomial(1, 0.028, n_treatment)
    
    # Run test
    results = run_ab_test(control_ctr, treatment_ctr, metric_type='binary')
    
    print("\nResults:")
    print(f"  Control CTR: {results['control_mean']:.4f}")
    print(f"  Treatment CTR: {results['treatment_mean']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Significant: {results['is_significant']}")
    print(f"  Effect: {results['effect_size']:.4f}")
    
    print("\nInterpretation:")
    print(interpret_ab_test_results(results))
    
    # Sample size calculation
    print("\n" + "=" * 60)
    print("Sample Size Calculation")
    print("-" * 60)
    
    # Want to detect 10% relative improvement (2.5% → 2.75%)
    effect_size = 0.0025  # Absolute difference
    baseline = 0.025
    
    sample_size = calculate_sample_size(
        effect_size=effect_size,
        baseline_rate=baseline,
        alpha=0.05,
        power=0.8
    )
    
    print(f"\nTo detect {effect_size*100:.2f}% absolute improvement:")
    print(f"  Required sample size per group: {sample_size:,}")
    print(f"  Total sample size: {sample_size * 2:,}")

