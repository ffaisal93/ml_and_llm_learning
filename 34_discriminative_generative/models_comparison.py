"""
Discriminative vs Generative Models
Simple implementations with detailed explanations
"""
import numpy as np
from typing import Tuple

# ==================== DISCRIMINATIVE MODELS ====================

class LogisticRegressionDiscriminative:
    """
    Logistic Regression: Discriminative Model
    
    What it does:
    - Learns P(Y|X) directly
    - Models: P(Y=1|X) = σ(β₀ + β₁X₁ + ... + βₙXₙ)
    - Finds decision boundary
    
    Assumptions:
    1. Binary outcome
    2. Linearity of log-odds
    3. Independence of observations
    4. No multicollinearity
    5. Large sample size
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression"""
        n_samples, n_features = X.shape
        
        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear = X.dot(self.weights) + self.bias
            predictions = self.sigmoid(linear)
            
            # Gradients
            dw = (1/n_samples) * X.T.dot(predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities P(Y=1|X)"""
        linear = X.dot(self.weights) + self.bias
        return self.sigmoid(linear)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return (self.predict_proba(X) >= 0.5).astype(int)


# ==================== GENERATIVE MODELS ====================

class NaiveBayesGenerative:
    """
    Naive Bayes: Generative Model
    
    What it does:
    - Learns P(X, Y) = P(X|Y) * P(Y)
    - Uses Bayes' theorem: P(Y|X) = P(X|Y) * P(Y) / P(X)
    - Assumes features are independent given class (naive assumption)
    
    Assumptions:
    1. Features are independent given class: P(X₁, X₂, ..., Xₙ|Y) = Π P(Xᵢ|Y)
    2. Can model P(X|Y) for each class
    3. Prior P(Y) can be estimated
    
    Why "Naive"?
    - Assumes independence (usually not true in practice)
    - But works surprisingly well despite this assumption
    """
    
    def __init__(self):
        self.class_priors = None
        self.class_means = None
        self.class_stds = None
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Naive Bayes"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Initialize
        self.class_priors = np.zeros(n_classes)
        self.class_means = np.zeros((n_classes, n_features))
        self.class_stds = np.zeros((n_classes, n_features))
        
        # Estimate parameters for each class
        for i, c in enumerate(self.classes):
            # Class prior: P(Y=c)
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(X)
            
            # Class-conditional: P(X|Y=c)
            # Assume Gaussian: mean and std for each feature
            self.class_means[i] = np.mean(X_c, axis=0)
            self.class_stds[i] = np.std(X_c, axis=0) + 1e-8  # Avoid division by zero
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Gaussian probability density function"""
        exponent = -0.5 * ((x - mean) / std) ** 2
        return np.prod(np.exp(exponent) / (std * np.sqrt(2 * np.pi)))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Bayes' theorem
        
        P(Y=c|X) = P(X|Y=c) * P(Y=c) / P(X)
        
        Since P(X) is same for all classes, we can ignore it
        and just compare numerators
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j, c in enumerate(self.classes):
                # P(X|Y=c) using naive assumption
                # P(X₁, X₂, ..., Xₙ|Y=c) = Π P(Xᵢ|Y=c)
                likelihood = self._gaussian_pdf(
                    X[i], 
                    self.class_means[j], 
                    self.class_stds[j]
                )
                
                # P(Y=c|X) ∝ P(X|Y=c) * P(Y=c)
                probabilities[i, j] = likelihood * self.class_priors[j]
            
            # Normalize (divide by P(X))
            probabilities[i] = probabilities[i] / probabilities[i].sum()
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


# ==================== BAYES' THEOREM ====================

def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """
    Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
    
    Detailed Explanation:
    
    This theorem allows us to update our beliefs about event A after
    observing evidence B. It combines:
    
    1. Prior P(A): What we believe about A before seeing B
    2. Likelihood P(B|A): How likely is B if A is true
    3. Evidence P(B): Total probability of observing B
    4. Posterior P(A|B): Updated belief about A after seeing B
    
    The key insight is that we start with prior knowledge, then update
    it with new evidence to get a more informed posterior probability.
    
    Args:
        prior: P(A) - Prior probability
        likelihood: P(B|A) - Likelihood of evidence given A
        evidence: P(B) - Total probability of evidence
    
    Returns:
        posterior: P(A|B) - Posterior probability
    """
    posterior = (likelihood * prior) / evidence
    return posterior

def bayes_theorem_example():
    """
    Medical diagnosis example
    
    Problem:
    - Disease prevalence: 1% (prior)
    - Test accuracy: 95% (likelihood)
    - If someone tests positive, what's probability they have disease?
    """
    print("Bayes' Theorem: Medical Diagnosis Example")
    print("=" * 60)
    
    # Prior: P(disease) = 0.01
    p_disease = 0.01
    p_no_disease = 0.99
    
    # Likelihood: P(positive|disease) = 0.95, P(positive|no disease) = 0.05
    p_positive_given_disease = 0.95
    p_positive_given_no_disease = 0.05
    
    # Evidence: P(positive) = P(positive|disease)*P(disease) + P(positive|no disease)*P(no disease)
    p_positive = (p_positive_given_disease * p_disease + 
                 p_positive_given_no_disease * p_no_disease)
    
    # Posterior: P(disease|positive)
    p_disease_given_positive = bayes_theorem(
        p_disease,
        p_positive_given_disease,
        p_positive
    )
    
    print(f"Prior P(disease): {p_disease:.4f} (1%)")
    print(f"Likelihood P(positive|disease): {p_positive_given_disease:.4f} (95%)")
    print(f"Evidence P(positive): {p_positive:.4f}")
    print(f"\nPosterior P(disease|positive): {p_disease_given_positive:.4f} ({p_disease_given_positive*100:.1f}%)")
    print("\nKey Insight:")
    print("  Even with 95% accurate test, if disease is rare (1%),")
    print("  positive test only means ~16% chance of disease!")
    print("  This is because false positives from large healthy population")
    print("  outweigh true positives from small diseased population.")


# ==================== MODEL ASSUMPTIONS ====================

def check_linear_regression_assumptions(X: np.ndarray, y: np.ndarray, 
                                       residuals: np.ndarray) -> dict:
    """
    Check assumptions of linear regression
    
    Returns dictionary with assumption checks
    """
    assumptions = {}
    
    # 1. Linearity: Check residual plot
    # (In practice, plot residuals vs predicted)
    assumptions['linearity'] = "Check: Plot residuals vs predicted (should be random)"
    
    # 2. Independence: Check for autocorrelation
    # (In practice, use Durbin-Watson test)
    assumptions['independence'] = "Check: Durbin-Watson test (should be ~2)"
    
    # 3. Homoscedasticity: Check variance of residuals
    # (In practice, plot residuals vs predicted, look for funnel)
    residual_variance = np.var(residuals)
    assumptions['homoscedasticity'] = f"Residual variance: {residual_variance:.4f} (should be constant)"
    
    # 4. Normality: Check distribution of residuals
    # (In practice, use Q-Q plot, Shapiro-Wilk test)
    from scipy import stats
    _, p_value = stats.shapiro(residuals[:5000]) if len(residuals) > 5000 else stats.shapiro(residuals)
    assumptions['normality'] = f"Shapiro-Wilk p-value: {p_value:.4f} (p>0.05 means normal)"
    
    # 5. Multicollinearity: Check correlation between features
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        max_corr = np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
        assumptions['multicollinearity'] = f"Max correlation: {max_corr:.4f} (should be <0.8)"
    
    return assumptions

def check_logistic_regression_assumptions(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Check assumptions of logistic regression
    """
    assumptions = {}
    
    # 1. Binary outcome
    unique_classes = np.unique(y)
    assumptions['binary_outcome'] = f"Classes: {unique_classes} (should be binary)"
    
    # 2. Linearity of log-odds
    assumptions['linearity_log_odds'] = "Check: Box-Tidwell test (log-odds should be linear)"
    
    # 3. Independence
    assumptions['independence'] = "Check: Observations should be independent"
    
    # 4. No multicollinearity
    if X.shape[1] > 1:
        corr_matrix = np.corrcoef(X.T)
        max_corr = np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
        assumptions['multicollinearity'] = f"Max correlation: {max_corr:.4f} (should be <0.8)"
    
    # 5. Large sample size
    assumptions['sample_size'] = f"Sample size: {len(X)} (should be large, especially with many features)"
    
    return assumptions


# ==================== COMPARISON ====================

def compare_discriminative_generative():
    """
    Compare discriminative vs generative models
    """
    print("Discriminative vs Generative Models")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    # Two classes
    X1 = np.random.randn(n_samples//2, 2) + np.array([-1, -1])
    X2 = np.random.randn(n_samples//2, 2) + np.array([1, 1])
    X = np.vstack([X1, X2])
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    # Train discriminative (Logistic Regression)
    lr = LogisticRegressionDiscriminative(learning_rate=0.1, n_iterations=1000)
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    lr_accuracy = np.mean(lr_pred == y)
    
    # Train generative (Naive Bayes)
    nb = NaiveBayesGenerative()
    nb.fit(X, y)
    nb_pred = nb.predict(X)
    nb_accuracy = np.mean(nb_pred == y)
    
    print(f"\nLogistic Regression (Discriminative):")
    print(f"  Accuracy: {lr_accuracy:.4f}")
    print(f"  Learns: P(Y|X) directly")
    print(f"  Focus: Decision boundary")
    
    print(f"\nNaive Bayes (Generative):")
    print(f"  Accuracy: {nb_accuracy:.4f}")
    print(f"  Learns: P(X, Y) = P(X|Y) * P(Y)")
    print(f"  Focus: Data distribution")
    
    print(f"\nKey Differences:")
    print(f"  - Discriminative: Simpler, faster, less data needed")
    print(f"  - Generative: Can generate data, handles missing data better")
    print(f"  - Both can work well, choice depends on use case")


# ==================== USAGE ====================

if __name__ == "__main__":
    print("Discriminative vs Generative Models")
    print("=" * 60)
    
    # Bayes' theorem example
    bayes_theorem_example()
    print()
    
    # Model comparison
    compare_discriminative_generative()
    print()
    
    # Assumptions check
    print("Model Assumptions:")
    print("=" * 60)
    
    # Linear regression assumptions
    X_lin = np.random.randn(100, 2)
    y_lin = 2 * X_lin[:, 0] + 1 * X_lin[:, 1] + 0.1 * np.random.randn(100)
    residuals_lin = y_lin - (2 * X_lin[:, 0] + 1 * X_lin[:, 1])
    
    print("\nLinear Regression Assumptions:")
    assumptions_lin = check_linear_regression_assumptions(X_lin, y_lin, residuals_lin)
    for key, value in assumptions_lin.items():
        print(f"  {key}: {value}")
    
    # Logistic regression assumptions
    X_log = np.random.randn(100, 2)
    y_log = ((X_log[:, 0] + X_log[:, 1]) > 0).astype(int)
    
    print("\nLogistic Regression Assumptions:")
    assumptions_log = check_logistic_regression_assumptions(X_log, y_log)
    for key, value in assumptions_log.items():
        print(f"  {key}: {value}")

