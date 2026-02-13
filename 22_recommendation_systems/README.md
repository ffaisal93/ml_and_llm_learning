# Topic 22: Recommendation Systems

## What You'll Learn

This topic teaches you recommendation systems:
- Collaborative Filtering
- Matrix Factorization
- Content-Based
- Evaluation metrics
- Simple implementations

## Why We Need This

### Interview Importance
- **Common question**: "How do recommendation systems work?"
- **Practical knowledge**: Used in many companies
- **Evaluation**: Know how to measure success

### Real-World Application
- **E-commerce**: Amazon, eBay
- **Streaming**: Netflix, Spotify
- **Social media**: Facebook, Twitter

## Industry Use Cases

### 1. **Collaborative Filtering**
**Use Case**: User-item interactions
- "Users who liked X also liked Y"
- Matrix factorization
- Most common approach

### 2. **Content-Based**
**Use Case**: Item features
- Recommend similar items
- Based on item properties
- No user data needed

### 3. **Hybrid**
**Use Case**: Best of both
- Combine collaborative + content
- Better recommendations
- More robust

## Industry-Standard Boilerplate Code

### Matrix Factorization (Simplified)

```python
"""
Matrix Factorization for Recommendations
Interview question: "Implement recommendation system"
"""
import numpy as np

def matrix_factorization(R: np.ndarray, k: int, 
                        learning_rate: float = 0.01,
                        iterations: int = 100) -> tuple:
    """
    Matrix Factorization: R ≈ P @ Q^T
    
    R: User-item rating matrix (n_users, n_items)
    P: User factors (n_users, k)
    Q: Item factors (n_items, k)
    k: Number of latent factors
    
    Returns: (P, Q)
    """
    n_users, n_items = R.shape
    
    # Initialize factors
    P = np.random.randn(n_users, k) * 0.1
    Q = np.random.randn(n_items, k) * 0.1
    
    # Only train on observed ratings
    for iteration in range(iterations):
        for i in range(n_users):
            for j in range(n_items):
                if R[i, j] > 0:  # Observed rating
                    # Prediction
                    pred = P[i] @ Q[j]
                    error = R[i, j] - pred
                    
                    # Update factors
                    P[i] += learning_rate * (error * Q[j])
                    Q[j] += learning_rate * (error * P[i])
    
    return P, Q

def predict_rating(user_id: int, item_id: int, P: np.ndarray, Q: np.ndarray) -> float:
    """Predict rating for user-item pair"""
    return P[user_id] @ Q[item_id]
```

### Evaluation Metrics

```python
"""
Recommendation System Evaluation
"""
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """
    Precision@K: Of top K recommendations, how many are relevant?
    """
    top_k = recommended[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / k if k > 0 else 0.0

def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    """
    Recall@K: Of all relevant items, how many in top K?
    """
    top_k = recommended[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant))
    return relevant_in_top_k / len(relevant) if len(relevant) > 0 else 0.0

def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    Higher score for relevant items at top
    """
    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Ideal DCG (all relevant items at top)
    ideal_dcg = sum(1.0 / np.log2(i + 2) 
                   for i in range(min(len(relevant), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

## Theory

### Matrix Factorization
- **Goal**: R ≈ P @ Q^T
- **P**: User preferences (latent factors)
- **Q**: Item characteristics (latent factors)
- **k**: Number of latent dimensions

### Evaluation Metrics
- **RMSE**: Rating prediction accuracy
- **Precision@K**: Relevance of top K
- **Recall@K**: Coverage of relevant items
- **NDCG@K**: Ranking quality

## Exercises

1. Implement matrix factorization
2. Evaluate recommendations
3. Compare different k values
4. Add regularization

## Next Steps

- **Topic 23**: Clustering evaluation
- **Topic 24**: Linear algebra Q&A

