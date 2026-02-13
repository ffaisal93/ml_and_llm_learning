"""
Recommendation Systems
Simple implementations
"""
import numpy as np

def matrix_factorization(R: np.ndarray, k: int, 
                        learning_rate: float = 0.01,
                        iterations: int = 100,
                        lambda_reg: float = 0.01) -> tuple:
    """
    Matrix Factorization: R ≈ P @ Q^T
    
    R: User-item rating matrix (n_users, n_items)
    P: User factors (n_users, k)
    Q: Item factors (n_items, k)
    k: Number of latent factors
    """
    n_users, n_items = R.shape
    
    # Initialize factors
    P = np.random.randn(n_users, k) * 0.1
    Q = np.random.randn(n_items, k) * 0.1
    
    # Train on observed ratings
    for iteration in range(iterations):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:  # Observed rating
                    # Prediction
                    pred = P[i] @ Q[j]
                    error = R[i, j] - pred
                    
                    # Update with regularization
                    P[i] += learning_rate * (error * Q[j] - lambda_reg * P[i])
                    Q[j] += learning_rate * (error * P[i] - lambda_reg * Q[j])
    
    return P, Q

def predict_rating(user_id: int, item_id: int, P: np.ndarray, Q: np.ndarray) -> float:
    """Predict rating for user-item pair"""
    return P[user_id] @ Q[item_id]

def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """Precision@K: Of top K, how many are relevant?"""
    top_k = recommended[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / k if k > 0 else 0.0

def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    """Recall@K: Of all relevant, how many in top K?"""
    top_k = recommended[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / len(relevant) if len(relevant) > 0 else 0.0

def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain"""
    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    
    ideal_dcg = sum(1.0 / np.log2(i + 2) 
                   for i in range(min(len(relevant), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# Usage Example
if __name__ == "__main__":
    print("Recommendation Systems")
    print("=" * 60)
    
    # Sample rating matrix (0 = not rated)
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    
    print("Rating Matrix R:")
    print(R)
    print()
    
    # Matrix factorization
    k = 2
    P, Q = matrix_factorization(R, k, iterations=100)
    
    print(f"User factors P shape: {P.shape}")
    print(f"Item factors Q shape: {Q.shape}")
    print()
    
    # Predict ratings
    print("Predicted Ratings:")
    R_pred = P @ Q.T
    print(R_pred)
    print()
    
    # Evaluation
    recommended = [0, 1, 2, 3]
    relevant = [0, 2]
    
    print("Evaluation Metrics:")
    print(f"Precision@2: {precision_at_k(recommended, relevant, k=2):.4f}")
    print(f"Recall@2: {recall_at_k(recommended, relevant, k=2):.4f}")
    print(f"NDCG@2: {ndcg_at_k(recommended, relevant, k=2):.4f}")

