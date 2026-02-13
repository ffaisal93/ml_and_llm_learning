"""
Clustering Evaluation Metrics
Simple implementations
"""
import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score: Measure clustering quality
    
    For each point:
    - a(i) = avg distance to same cluster
    - b(i) = avg distance to nearest other cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Range: -1 to 1 (higher is better)
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        # Distance to same cluster
        same_cluster_mask = labels == labels[i]
        same_cluster = X[same_cluster_mask]
        
        if len(same_cluster) > 1:
            distances = [np.linalg.norm(X[i] - x) 
                       for j, x in enumerate(same_cluster) 
                       if not np.array_equal(X[i], x)]
            a_i = np.mean(distances) if distances else 0.0
        else:
            a_i = 0.0
        
        # Distance to nearest other cluster
        min_b_i = float('inf')
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                distances = [np.linalg.norm(X[i] - x) for x in other_cluster]
                b_i = np.mean(distances) if distances else float('inf')
                min_b_i = min(min_b_i, b_i)
        
        b_i = min_b_i
        
        # Silhouette score
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)

def inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Inertia: Within-cluster sum of squares
    
    Sum of squared distances to centroids
    Lower is better
    """
    unique_labels = np.unique(labels)
    total_inertia = 0.0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            for point in cluster_points:
                total_inertia += np.linalg.norm(point - centroid)**2
    
    return total_inertia

def adjusted_rand_index(labels_true: np.ndarray, 
                       labels_pred: np.ndarray) -> float:
    """
    ARI: Adjusted Rand Index
    
    Compares clustering to ground truth
    Adjusted for chance
    Range: -1 to 1 (1 = perfect match)
    """
    n_samples = len(labels_true)
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    # Contingency table
    contingency = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency[i, j] = np.sum((labels_true == true_label) & 
                                     (labels_pred == pred_label))
    
    # Sum combinations
    sum_combinations = sum(np.sum(contingency[i]) * (np.sum(contingency[i]) - 1) / 2
                          for i in range(len(unique_true)))
    sum_pred = sum(np.sum(contingency[:, j]) * (np.sum(contingency[:, j]) - 1) / 2
                  for j in range(len(unique_pred)))
    sum_true = sum(contingency[i, j] * (contingency[i, j] - 1) / 2
                  for i in range(len(unique_true))
                  for j in range(len(unique_pred)))
    
    # ARI
    n_choose_2 = n_samples * (n_samples - 1) / 2
    expected_index = sum_combinations * sum_pred / n_choose_2
    max_index = (sum_combinations + sum_pred) / 2
    
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_true - expected_index) / (max_index - expected_index)
    return ari


# Usage Example
if __name__ == "__main__":
    print("Clustering Evaluation")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Two clusters
    cluster1 = np.random.randn(n_samples//2, 2) + np.array([0, 0])
    cluster2 = np.random.randn(n_samples//2, 2) + np.array([5, 5])
    X = np.vstack([cluster1, cluster2])
    
    # True labels
    labels_true = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    # Predicted labels (perfect clustering)
    labels_pred = labels_true.copy()
    
    # Evaluate
    silhouette = silhouette_score(X, labels_pred)
    inertia_score = inertia(X, labels_pred)
    ari = adjusted_rand_index(labels_true, labels_pred)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Inertia: {inertia_score:.4f}")
    print(f"ARI: {ari:.4f}")

