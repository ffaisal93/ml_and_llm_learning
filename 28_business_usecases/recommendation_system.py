"""
Recommendation System: Complete Business Solution
E-commerce product recommendations
"""
import numpy as np
import pandas as pd
from typing import List, Dict

class RecommendationSystem:
    """
    End-to-end recommendation system
    
    Business Problem:
    - Recommend products to users to increase sales
    - Improve user engagement and conversion
    - Increase average order value
    """
    
    def __init__(self):
        self.collaborative_model = None
        self.content_model = None
    
    def collaborative_filtering(self, user_item_matrix: np.ndarray, 
                               n_factors: int = 50) -> Dict:
        """
        Collaborative Filtering: Matrix Factorization
        
        R ≈ P @ Q^T
        
        Where:
        - R: User-item interaction matrix
        - P: User factors (n_users × n_factors)
        - Q: Item factors (n_items × n_factors)
        """
        from sklearn.decomposition import NMF
        
        # Non-negative Matrix Factorization
        model = NMF(n_components=n_factors, random_state=42)
        user_factors = model.fit_transform(user_item_matrix)
        item_factors = model.components_
        
        self.collaborative_model = {
            'user_factors': user_factors,
            'item_factors': item_factors,
            'model': model
        }
        
        return self.collaborative_model
    
    def content_based_recommendations(self, item_features: np.ndarray,
                                     user_history: np.ndarray) -> np.ndarray:
        """
        Content-Based: Recommend items similar to user's past purchases
        
        Steps:
        1. Get user's preferred item features (average of purchased items)
        2. Compute similarity to all items
        3. Recommend most similar items
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # User profile: average features of purchased items
        user_profile = np.mean(item_features[user_history], axis=0)
        
        # Similarity to all items
        similarities = cosine_similarity([user_profile], item_features)[0]
        
        return similarities
    
    def hybrid_recommendations(self, user_id: int, 
                              collaborative_scores: np.ndarray,
                              content_scores: np.ndarray,
                              weights: Dict = None) -> np.ndarray:
        """
        Hybrid: Combine collaborative and content-based
        
        Final Score = w1 × Collaborative + w2 × Content
        
        Default weights: 60% collaborative, 40% content
        """
        if weights is None:
            weights = {'collaborative': 0.6, 'content': 0.4}
        
        # Normalize scores
        collaborative_norm = (collaborative_scores - collaborative_scores.min()) / (
            collaborative_scores.max() - collaborative_scores.min() + 1e-8
        )
        content_norm = (content_scores - content_scores.min()) / (
            content_scores.max() - content_scores.min() + 1e-8
        )
        
        # Weighted combination
        final_scores = (weights['collaborative'] * collaborative_norm +
                       weights['content'] * content_norm)
        
        return final_scores
    
    def get_recommendations(self, user_id: int, user_item_matrix: np.ndarray,
                           item_features: np.ndarray, user_history: List[int],
                           top_k: int = 10) -> List[int]:
        """
        Get top K recommendations for user
        
        Steps:
        1. Collaborative filtering scores
        2. Content-based scores
        3. Hybrid combination
        4. Return top K items (excluding already purchased)
        """
        # Collaborative scores
        if self.collaborative_model is None:
            self.collaborative_filtering(user_item_matrix)
        
        user_factors = self.collaborative_model['user_factors'][user_id]
        item_factors = self.collaborative_model['item_factors']
        collaborative_scores = user_factors @ item_factors.T
        
        # Content-based scores
        content_scores = self.content_based_recommendations(
            item_features, user_history
        )
        
        # Hybrid
        final_scores = self.hybrid_recommendations(
            user_id, collaborative_scores, content_scores
        )
        
        # Exclude already purchased items
        final_scores[user_history] = -np.inf
        
        # Top K
        top_k_indices = np.argsort(final_scores)[-top_k:][::-1]
        
        return top_k_indices.tolist()
    
    def evaluate_recommendations(self, recommendations: List[int],
                               actual_purchases: List[int],
                               k: int = 10) -> Dict:
        """
        Evaluate recommendation quality
        
        Metrics:
        - Precision@K: Of recommended items, how many were purchased?
        - Recall@K: Of purchased items, how many were recommended?
        - NDCG@K: Ranking quality
        """
        # Precision@K
        recommended_set = set(recommendations[:k])
        purchased_set = set(actual_purchases)
        precision = len(recommended_set & purchased_set) / k
        
        # Recall@K
        recall = len(recommended_set & purchased_set) / len(purchased_set) if purchased_set else 0
        
        # NDCG@K (simplified)
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in purchased_set:
                dcg += 1.0 / np.log2(i + 2)
        
        ideal_dcg = sum(1.0 / np.log2(i + 2) 
                       for i in range(min(len(purchased_set), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'ndcg_at_k': ndcg
        }


def recommendation_system_pipeline():
    """
    Complete pipeline for recommendation system
    
    Business Impact:
    - Increase conversion rate by 15%
    - Improve user engagement
    - Increase average order value
    """
    print("Recommendation System Pipeline")
    print("=" * 60)
    
    print("\n1. Problem Definition:")
    print("   - Goal: Recommend products to increase sales")
    print("   - Metrics: CTR, conversion rate, revenue")
    print("   - Latency: <100ms per request")
    
    print("\n2. Data Collection:")
    print("   - User-item interactions (views, purchases, ratings)")
    print("   - Item features (category, price, brand, description)")
    print("   - User features (demographics, purchase history)")
    
    print("\n3. Model Architecture:")
    print("   - Collaborative Filtering (matrix factorization)")
    print("   - Content-Based (item similarity)")
    print("   - Hybrid (combine both)")
    
    print("\n4. Implementation:")
    print("   - Train models offline")
    print("   - Pre-compute recommendations (caching)")
    print("   - Real-time updates for new users/items")
    
    print("\n5. Evaluation:")
    print("   - Offline: Precision@K, Recall@K, NDCG@K")
    print("   - Online: A/B test (CTR, conversion rate)")
    
    print("\n6. Deployment:")
    print("   - API for real-time recommendations")
    print("   - Batch updates for model retraining")
    print("   - Monitoring: latency, throughput, metrics")


# Usage Example
if __name__ == "__main__":
    recommendation_system_pipeline()
    
    print("\n" + "=" * 60)
    print("Example Implementation")
    print("=" * 60)
    
    # Simulated data
    n_users = 1000
    n_items = 500
    
    # User-item matrix (interactions)
    user_item_matrix = np.random.rand(n_users, n_items) > 0.9
    
    # Item features (e.g., embeddings)
    item_features = np.random.randn(n_items, 50)
    
    # Example user
    user_id = 0
    user_history = [10, 20, 30]  # Items user has purchased
    
    # Initialize system
    rec_system = RecommendationSystem()
    
    # Get recommendations
    recommendations = rec_system.get_recommendations(
        user_id, user_item_matrix, item_features, user_history, top_k=10
    )
    
    print(f"\nTop 10 Recommendations for User {user_id}:")
    print(f"  Items: {recommendations}")
    
    # Evaluate (simulated)
    actual_purchases = [15, 25, 35]  # Items user actually purchased later
    metrics = rec_system.evaluate_recommendations(
        recommendations, actual_purchases, k=10
    )
    
    print(f"\nEvaluation Metrics:")
    print(f"  Precision@10: {metrics['precision_at_k']:.4f}")
    print(f"  Recall@10: {metrics['recall_at_k']:.4f}")
    print(f"  NDCG@10: {metrics['ndcg_at_k']:.4f}")

