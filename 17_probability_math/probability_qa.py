"""
Probability Math Q&A
Common interview questions with solutions
"""
import numpy as np
from scipy import stats

# ==================== Bayes' Theorem ====================

def bayes_theorem(p_a_given_b: float, p_b: float, p_a: float) -> float:
    """
    Bayes' Theorem: P(B|A) = P(A|B) × P(B) / P(A)
    
    Example: Medical test
    - P(positive|disease) = 0.95
    - P(disease) = 0.01
    - P(positive) = 0.1
    - Find: P(disease|positive)
    """
    p_b_given_a = (p_a_given_b * p_b) / p_a
    return p_b_given_a

# ==================== Expected Value ====================

def expected_value(values: np.ndarray, probabilities: np.ndarray) -> float:
    """
    Expected Value: E[X] = Σ x × P(x)
    """
    return np.sum(values * probabilities)

def variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    """
    Variance: Var(X) = E[X²] - (E[X])²
    """
    e_x = expected_value(values, probabilities)
    e_x_squared = expected_value(values**2, probabilities)
    return e_x_squared - e_x**2

# ==================== Common Distributions ====================

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Binomial: P(k successes in n trials)
    """
    return stats.binom.pmf(k, n, p)

def poisson_probability(k: int, lambda_param: float) -> float:
    """
    Poisson: P(k events) with rate lambda
    """
    return stats.poisson.pmf(k, lambda_param)

def normal_probability(x: float, mu: float, sigma: float) -> float:
    """
    Normal: P(X = x) with mean mu, std sigma
    """
    return stats.norm.pdf(x, mu, sigma)

# ==================== Conditional Probability ====================

def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """
    Conditional: P(A|B) = P(A and B) / P(B)
    """
    return p_a_and_b / p_b if p_b > 0 else 0.0

# ==================== Independence ====================

def are_independent(p_a: float, p_b: float, p_a_and_b: float) -> bool:
    """
    Check if A and B are independent
    Independent if: P(A and B) = P(A) × P(B)
    """
    return abs(p_a_and_b - p_a * p_b) < 1e-6

# ==================== Interview Questions ====================

def interview_question_1():
    """
    Q: Medical test has 95% accuracy. Disease prevalence is 1%.
    If test is positive, what's probability of having disease?
    """
    p_positive_given_disease = 0.95
    p_disease = 0.01
    p_positive = 0.95 * 0.01 + 0.05 * 0.99  # Total probability
    
    p_disease_given_positive = bayes_theorem(
        p_positive_given_disease, p_disease, p_positive
    )
    
    print(f"Q: Medical test (95% accuracy), disease (1% prevalence)")
    print(f"   P(disease|positive) = {p_disease_given_positive:.4f}")
    return p_disease_given_positive

def interview_question_2():
    """
    Q: Expected value of rolling a die?
    """
    values = np.array([1, 2, 3, 4, 5, 6])
    probs = np.array([1/6] * 6)
    
    e_x = expected_value(values, probs)
    var_x = variance(values, probs)
    
    print(f"Q: Expected value of die roll")
    print(f"   E[X] = {e_x:.2f}")
    print(f"   Var(X) = {var_x:.2f}")
    return e_x

def interview_question_3():
    """
    Q: Two coins, what's P(both heads)?
    """
    p_head = 0.5
    p_both_heads = p_head * p_head
    
    print(f"Q: Two coin flips, P(both heads)")
    print(f"   P(both heads) = {p_both_heads:.2f}")
    return p_both_heads

# ==================== Usage ====================

if __name__ == "__main__":
    print("Probability Math Q&A")
    print("=" * 60)
    print()
    
    interview_question_1()
    print()
    
    interview_question_2()
    print()
    
    interview_question_3()
    print()
    
    # Bayes' theorem example
    print("Bayes' Theorem Example:")
    p_b_given_a = bayes_theorem(p_a_given_b=0.8, p_b=0.3, p_a=0.5)
    print(f"  P(B|A) = {p_b_given_a:.4f}")

