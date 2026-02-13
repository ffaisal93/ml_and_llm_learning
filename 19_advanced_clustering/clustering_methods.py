"""
Advanced Clustering Methods
Simple implementations
"""
import numpy as np

def hierarchical_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Hierarchical Clustering (Agglomerative)
    
    Builds tree by merging closest clusters
    """
    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]
    
    while len(clusters) > n_clusters:
        # Find two closest clusters
        min_dist = float('inf')
        merge_i, merge_j = 0, 1
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # Distance between centroids
                centroid_i = np.mean(X[clusters[i]], axis=0)
                centroid_j = np.mean(X[clusters[j]], axis=0)
                dist = np.linalg.norm(centroid_i - centroid_j)
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
        
        # Merge
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)
    
    # Assign labels
    labels = np.zeros(n_samples)
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    
    return labels

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN: Density-Based Clustering
    
    Finds clusters of arbitrary shape
    -1 = noise/outlier
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)
    cluster_id = 0
    
    def get_neighbors(point_idx: int) -> list:
        """Get neighbors within eps"""
        neighbors = []
        for i in range(n_samples):
            if np.linalg.norm(X[point_idx] - X[i]) <= eps:
                neighbors.append(i)
        return neighbors
    
    for i in range(n_samples):
        if labels[i] != -1:
            continue
        
        neighbors = get_neighbors(i)
        
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
            continue
        
        # Start cluster
        labels[i] = cluster_id
        seed_set = neighbors.copy()
        
        # Expand cluster
        j = 0
        while j < len(seed_set):
            neighbor = seed_set[j]
            
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            
            neighbor_neighbors = get_neighbors(neighbor)
            if len(neighbor_neighbors) >= min_samples:
                for nn in neighbor_neighbors:
                    if labels[nn] == -1:
                        seed_set.append(nn)
            
            j += 1
        
        cluster_id += 1
    
    return labels


# Usage Example
if __name__ == "__main__":
    print("Advanced Clustering Methods")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Two clusters
    cluster1 = np.random.randn(n_samples//2, 2) + np.array([0, 0])
    cluster2 = np.random.randn(n_samples//2, 2) + np.array([5, 5])
    X = np.vstack([cluster1, cluster2])
    
    # Hierarchical clustering
    labels_hier = hierarchical_clustering(X, n_clusters=2)
    print(f"Hierarchical: {len(np.unique(labels_hier))} clusters")
    
    # DBSCAN
    labels_dbscan = dbscan(X, eps=1.5, min_samples=5)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = np.sum(labels_dbscan == -1)
    print(f"DBSCAN: {n_clusters_dbscan} clusters, {n_noise} noise points")

