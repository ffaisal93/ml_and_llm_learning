"""
Information Theory Metrics from Scratch
Entropy, Cross-Entropy, KL Divergence, Mutual Information, Gini
"""
import numpy as np
from typing import Union

# ==================== 1. ENTROPY ====================

def entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """
    Entropy: H(X) = -Σ p(x) * log(p(x))
    
    Measures uncertainty/randomness in a distribution
    
    Args:
        probabilities: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats)
    
    Returns:
        Entropy value (bits or nats)
    
    Properties:
    - H(X) ≥ 0 (always non-negative)
    - Maximum when uniform distribution
    - Minimum (0) when deterministic (one outcome has prob=1)
    """
    # Remove zeros (log(0) is undefined)
    probabilities = probabilities[probabilities > 0]
    
    # Compute entropy
    log_probs = np.log(probabilities) / np.log(base)
    entropy_value = -np.sum(probabilities * log_probs)
    
    return entropy_value

def entropy_examples():
    """
    Examples of entropy
    """
    print("Entropy Examples:")
    print("=" * 60)
    
    # Fair coin: maximum entropy
    fair_coin = np.array([0.5, 0.5])
    h_fair = entropy(fair_coin)
    print(f"Fair coin: {h_fair:.4f} bits (maximum uncertainty)")
    
    # Biased coin: lower entropy
    biased_coin = np.array([0.9, 0.1])
    h_biased = entropy(biased_coin)
    print(f"Biased coin (90/10): {h_biased:.4f} bits (less uncertainty)")
    
    # Deterministic: zero entropy
    deterministic = np.array([1.0, 0.0])
    h_det = entropy(deterministic)
    print(f"Deterministic: {h_det:.4f} bits (no uncertainty)")
    
    # Uniform distribution: maximum entropy
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    h_uniform = entropy(uniform)
    print(f"Uniform (4 classes): {h_uniform:.4f} bits (maximum)")
    print(f"  Max entropy for 4 classes: {np.log2(4):.4f} bits")

# ==================== 2. CROSS-ENTROPY ====================

def cross_entropy(true_probs: np.ndarray, pred_probs: np.ndarray, 
                 base: float = 2.0) -> float:
    """
    Cross-Entropy: H(P, Q) = -Σ p(x) * log(q(x))
    
    Measures average bits needed to encode P using code optimized for Q
    Always ≥ H(P) (entropy of true distribution)
    Equal to H(P) when Q = P
    
    Args:
        true_probs: True distribution P
        pred_probs: Predicted distribution Q
        base: Logarithm base
    
    Returns:
        Cross-entropy value
    """
    # Remove zeros
    mask = (true_probs > 0) & (pred_probs > 0)
    true_probs = true_probs[mask]
    pred_probs = pred_probs[mask]
    
    # Compute cross-entropy
    log_pred = np.log(pred_probs) / np.log(base)
    cross_ent = -np.sum(true_probs * log_pred)
    
    return cross_ent

def cross_entropy_examples():
    """
    Examples of cross-entropy
    """
    print("\nCross-Entropy Examples:")
    print("=" * 60)
    
    # Perfect prediction: cross-entropy = entropy
    true_dist = np.array([0.5, 0.3, 0.2])
    perfect_pred = np.array([0.5, 0.3, 0.2])
    
    h_true = entropy(true_dist)
    ce_perfect = cross_entropy(true_dist, perfect_pred)
    
    print(f"True distribution entropy: {h_true:.4f} bits")
    print(f"Perfect prediction cross-entropy: {ce_perfect:.4f} bits")
    print(f"  → Equal when prediction is perfect")
    
    # Bad prediction: cross-entropy > entropy
    bad_pred = np.array([0.1, 0.1, 0.8])  # Wrong distribution
    ce_bad = cross_entropy(true_dist, bad_pred)
    
    print(f"Bad prediction cross-entropy: {ce_bad:.4f} bits")
    print(f"  → Higher when prediction is wrong")
    print(f"  → Penalizes confident wrong predictions")

# ==================== 3. KL DIVERGENCE ====================

def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    KL Divergence: KL(P || Q) = Σ p(x) * log(p(x) / q(x))
    
    Measures how different Q is from P
    Not symmetric: KL(P || Q) ≠ KL(Q || P)
    Not a metric (doesn't satisfy triangle inequality)
    
    Properties:
    - KL(P || Q) ≥ 0 (always non-negative)
    - KL(P || Q) = 0 if and only if P = Q
    - Asymmetric
    
    Args:
        p: True/reference distribution P
        q: Approximated distribution Q
        base: Logarithm base
    
    Returns:
        KL divergence value
    """
    # Remove zeros (only where p > 0, q can be 0)
    mask = p > 0
    p = p[mask]
    q = q[mask]
    
    # Avoid division by zero
    q = np.maximum(q, 1e-10)
    
    # Compute KL divergence
    ratio = p / q
    log_ratio = np.log(ratio) / np.log(base)
    kl = np.sum(p * log_ratio)
    
    return kl

def kl_divergence_examples():
    """
    Examples of KL divergence
    """
    print("\nKL Divergence Examples:")
    print("=" * 60)
    
    # Identical distributions: KL = 0
    p1 = np.array([0.5, 0.5])
    q1 = np.array([0.5, 0.5])
    kl1 = kl_divergence(p1, q1)
    print(f"Identical distributions: KL = {kl1:.4f}")
    
    # Different distributions: KL > 0
    p2 = np.array([0.5, 0.5])
    q2 = np.array([0.9, 0.1])
    kl2 = kl_divergence(p2, q2)
    print(f"Different distributions: KL = {kl2:.4f}")
    
    # Asymmetry demonstration
    kl_pq = kl_divergence(p2, q2)
    kl_qp = kl_divergence(q2, p2)
    print(f"\nAsymmetry:")
    print(f"  KL(P || Q) = {kl_pq:.4f}")
    print(f"  KL(Q || P) = {kl_qp:.4f}")
    print(f"  → Not equal (asymmetric)")

# ==================== 4. MUTUAL INFORMATION ====================

def mutual_information(joint_probs: np.ndarray, 
                      marginal_x: np.ndarray,
                      marginal_y: np.ndarray,
                      base: float = 2.0) -> float:
    """
    Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    Measures how much information X gives about Y
    Symmetric: I(X; Y) = I(Y; X)
    
    Properties:
    - I(X; Y) = 0 if X and Y are independent
    - I(X; Y) = H(X) if X completely determines Y
    - I(X; Y) ≥ 0 (always non-negative)
    
    Args:
        joint_probs: Joint distribution P(X, Y) (2D array)
        marginal_x: Marginal distribution P(X)
        marginal_y: Marginal distribution P(Y)
        base: Logarithm base
    
    Returns:
        Mutual information value
    """
    # Entropy of X
    h_x = entropy(marginal_x, base)
    
    # Entropy of Y
    h_y = entropy(marginal_y, base)
    
    # Joint entropy H(X, Y)
    joint_flat = joint_probs.flatten()
    h_xy = entropy(joint_flat, base)
    
    # Mutual information
    mi = h_x + h_y - h_xy
    
    return mi

def mutual_information_from_samples(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information from samples
    
    Estimates distributions from data, then computes MI
    """
    # Estimate joint distribution
    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)
    
    # Create joint distribution
    joint = np.zeros((len(unique_x), len(unique_y)))
    for i, val_x in enumerate(unique_x):
        for j, val_y in enumerate(unique_y):
            joint[i, j] = np.sum((x == val_x) & (y == val_y))
    
    joint = joint / len(x)  # Normalize
    
    # Marginal distributions
    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)
    
    return mutual_information(joint, marginal_x, marginal_y)

def mutual_information_examples():
    """
    Examples of mutual information
    """
    print("\nMutual Information Examples:")
    print("=" * 60)
    
    # Independent variables: MI = 0
    np.random.seed(42)
    x_indep = np.random.randint(0, 3, 1000)
    y_indep = np.random.randint(0, 3, 1000)
    mi_indep = mutual_information_from_samples(x_indep, y_indep)
    print(f"Independent variables: MI = {mi_indep:.4f} (should be ~0)")
    
    # Dependent variables: MI > 0
    x_dep = np.random.randint(0, 3, 1000)
    y_dep = (x_dep + np.random.randint(0, 2, 1000)) % 3  # y depends on x
    mi_dep = mutual_information_from_samples(x_dep, y_dep)
    print(f"Dependent variables: MI = {mi_dep:.4f} (should be > 0)")
    
    # Deterministic relationship: MI = H(X)
    x_det = np.random.randint(0, 3, 1000)
    y_det = x_det  # y = x (deterministic)
    mi_det = mutual_information_from_samples(x_det, y_det)
    h_x = entropy(np.bincount(x_det) / len(x_det))
    print(f"Deterministic (y=x): MI = {mi_det:.4f}, H(X) = {h_x:.4f}")
    print(f"  → MI ≈ H(X) when y completely determined by x")

# ==================== 5. GINI IMPURITY ====================

def gini_impurity(probabilities: np.ndarray) -> float:
    """
    Gini Impurity: Gini = 1 - Σ p_i²
    
    Measures probability of misclassifying a random element
    if labeled according to class distribution
    
    Properties:
    - Gini = 0: Pure (all same class)
    - Gini = 1 - 1/k: Maximum for k classes (uniform)
    - Range: [0, 1-1/k] for k classes
    - For binary: [0, 0.5]
    
    Args:
        probabilities: Class probabilities (must sum to 1)
    
    Returns:
        Gini impurity value
    """
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def gini_vs_entropy():
    """
    Compare Gini and Entropy
    """
    print("\nGini vs Entropy Comparison:")
    print("=" * 60)
    
    # Different distributions
    distributions = [
        ([1.0, 0.0], "Pure (deterministic)"),
        ([0.9, 0.1], "Highly biased"),
        ([0.7, 0.3], "Moderately biased"),
        ([0.5, 0.5], "Uniform (maximum)"),
    ]
    
    print("Distribution | Gini  | Entropy | Difference")
    print("-" * 50)
    
    for probs, name in distributions:
        probs = np.array(probs)
        gini = gini_impurity(probs)
        ent = entropy(probs)
        diff = abs(gini - ent)
        print(f"{name:20} | {gini:.4f} | {ent:.4f} | {diff:.4f}")
    
    print("\nKey Differences:")
    print("  - Gini: Faster to compute (no log)")
    print("  - Entropy: More information-theoretic")
    print("  - Both work similarly for decision trees")
    print("  - Gini: More sensitive to class probability changes")

# ==================== 6. JENSEN-SHANNON DIVERGENCE ====================

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, 
                              base: float = 2.0) -> float:
    """
    Jensen-Shannon Divergence: JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    
    Where M = 0.5 * (P + Q) is the average distribution
    
    Properties:
    - Symmetric: JS(P || Q) = JS(Q || P)
    - Bounded: JS(P || Q) ∈ [0, 1] (when using log base 2)
    - Metric: Satisfies triangle inequality
    - More stable than KL divergence
    
    Args:
        p: Distribution P
        q: Distribution Q
        base: Logarithm base
    
    Returns:
        JS divergence value
    """
    # Average distribution
    m = 0.5 * (p + q)
    
    # JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    js = 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)
    
    return js

def js_divergence_examples():
    """
    Examples of JS divergence
    """
    print("\nJensen-Shannon Divergence Examples:")
    print("=" * 60)
    
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    
    # Symmetry
    js_pq = jensen_shannon_divergence(p, q)
    js_qp = jensen_shannon_divergence(q, p)
    
    print(f"JS(P || Q) = {js_pq:.4f}")
    print(f"JS(Q || P) = {js_qp:.4f}")
    print(f"  → Equal (symmetric)")
    
    # Bounded
    print(f"\nJS divergence is bounded: [0, 1]")
    print(f"  Current value: {js_pq:.4f}")

# ==================== USAGE ====================

if __name__ == "__main__":
    print("Information Theory Metrics")
    print("=" * 60)
    
    entropy_examples()
    cross_entropy_examples()
    kl_divergence_examples()
    mutual_information_examples()
    gini_vs_entropy()
    js_divergence_examples()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Entropy: Uncertainty in distribution")
    print("  - Cross-Entropy: Loss function for classification")
    print("  - KL Divergence: Distance between distributions (asymmetric)")
    print("  - Mutual Information: Information shared between variables")
    print("  - Gini Impurity: Misclassification probability")
    print("  - JS Divergence: Symmetric version of KL")

