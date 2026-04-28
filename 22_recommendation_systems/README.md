# Topic 22: Recommendation Systems

> 🔥 **For interviews, read these first:**
> - **`RECOMMENDATION_SYSTEMS_DEEP_DIVE.md`** — frontier-lab deep dive: collaborative filtering, matrix factorization (BPR loss), two-tower retrieval (in-batch negatives, ANN serving), sequential models (GRU4Rec, SASRec, BERT4Rec), two-stage architecture (retrieval + ranking), GBDT/DeepFM/DLRM rankers, NDCG/MAP/MRR, cold start, echo chamber + exploration.
> - **`INTERVIEW_GRILL.md`** — 55 active-recall questions.

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

## Core Intuition

Recommendation systems try to predict preference, not just similarity.

That means a good recommender needs to answer some version of:
- what will this user like next?
- what should we rank near the top?

### Collaborative Filtering

Collaborative filtering works because patterns of user behavior contain shared structure.

If users behave similarly, their preferences may transfer.

This is why "users who liked X also liked Y" is such a common intuition.

### Matrix Factorization

Matrix factorization compresses the user-item interaction matrix into:
- user latent factors
- item latent factors

The idea is that preference can be explained in a lower-dimensional latent space.

### Content-Based Recommendation

Content-based methods use item attributes rather than shared user behavior.

They are useful when:
- you have strong metadata
- user interaction history is limited

### Hybrid Systems

Hybrid systems matter because collaborative and content-based methods fail differently.

Combining them is often more robust than relying on either one alone.

## Technical Details Interviewers Often Want

### Cold Start

This is one of the most common recommendation follow-ups.

Collaborative filtering struggles when:
- a new user has little history
- a new item has few interactions

Content features can help in those cases.

### Ranking vs Rating Prediction

Predicting a numeric rating is not the same as ranking the best items.

In many production systems, ranking quality matters more than absolute rating accuracy.

That is why metrics like:
- Precision@K
- Recall@K
- NDCG@K

often matter more than RMSE.

## Common Failure Modes

- optimizing RMSE when the real task is top-k ranking
- ignoring cold-start problems
- recommending only popular items and reducing diversity
- overfitting sparse interaction matrices
- assuming collaborative filtering alone is enough in early-stage products

## Edge Cases and Follow-Up Questions

1. Why can a recommender with low RMSE still have poor top-k recommendations?
2. What is the cold-start problem?
3. Why are hybrid systems often stronger in practice?
4. Why is recommendation a ranking problem more than a pure regression problem?
5. Why can popularity bias distort evaluation?

## What to Practice Saying Out Loud

1. Why collaborative filtering works
2. Why matrix factorization uses latent factors
3. Why ranking metrics often matter more than rating-error metrics

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
