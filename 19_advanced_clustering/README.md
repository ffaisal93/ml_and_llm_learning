# Topic 19: Advanced Clustering Methods

## What You'll Learn

This topic teaches you different clustering algorithms:
- K-Means (review)
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)
- Simple implementations

## Why We Need This

### Interview Importance
- **Common question**: "Explain different clustering methods"
- **Algorithm knowledge**: Shows breadth
- **Problem-solving**: Choose right method

### Real-World Application
- **Customer segmentation**: Different methods for different data
- **Anomaly detection**: DBSCAN finds outliers
- **Data exploration**: Understand data structure

## Industry Use Cases

### 1. **K-Means**
**Use Case**: Spherical clusters, known k
- Customer segmentation
- Image compression
- Simple and fast

### 2. **Hierarchical Clustering**
**Use Case**: Unknown k, hierarchical structure
- Taxonomy creation
- Document clustering
- Dendrogram visualization

### 3. **DBSCAN**
**Use Case**: Arbitrary shapes, outlier detection
- Anomaly detection
- Non-spherical clusters
- No need to specify k

### 4. **GMM**
**Use Case**: Probabilistic clustering
- Soft assignments
- Overlapping clusters
- Probability-based

## Industry-Standard Boilerplate Code

### Hierarchical Clustering

```python
"""
Hierarchical Clustering: Build tree of clusters
"""
import numpy as np

def hierarchical_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Simple hierarchical clustering (agglomerative)
    
    Algorithm:
    1. Start with each point as cluster
    2. Merge closest clusters
    3. Repeat until n_clusters
    """
    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]
    
    while len(clusters) > n_clusters:
        # Find two closest clusters
        min_dist = float('inf')
        merge_i, merge_j = 0, 1
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # Distance between cluster centroids
                centroid_i = np.mean(X[clusters[i]], axis=0)
                centroid_j = np.mean(X[clusters[j]], axis=0)
                dist = np.linalg.norm(centroid_i - centroid_j)
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
        
        # Merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)
    
    # Assign labels
    labels = np.zeros(n_samples)
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    
    return labels
```

### DBSCAN

```python
"""
DBSCAN: Density-based clustering
Finds clusters of arbitrary shape
"""
def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN: Density-Based Spatial Clustering
    
    Args:
        eps: Maximum distance for neighbors
        min_samples: Minimum points to form cluster
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)  # -1 = noise
    cluster_id = 0
    
    def get_neighbors(point_idx: int) -> list:
        """Get neighbors within eps"""
        neighbors = []
        for i in range(n_samples):
            if np.linalg.norm(X[point_idx] - X[i]) <= eps:
                neighbors.append(i)
        return neighbors
    
    for i in range(n_samples):
        if labels[i] != -1:  # Already processed
            continue
        
        neighbors = get_neighbors(i)
        
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
            continue
        
        # Start new cluster
        labels[i] = cluster_id
        seed_set = neighbors.copy()
        
        # Expand cluster
        j = 0
        while j < len(seed_set):
            neighbor = seed_set[j]
            
            if labels[neighbor] == -1:  # Noise -> border point
                labels[neighbor] = cluster_id
            elif labels[neighbor] == -1:  # Unvisited
                labels[neighbor] = cluster_id
                neighbor_neighbors = get_neighbors(neighbor)
                if len(neighbor_neighbors) >= min_samples:
                    seed_set.extend(neighbor_neighbors)
            
            j += 1
        
        cluster_id += 1
    
    return labels
```

### GMM (Simplified)

```python
"""
GMM: Gaussian Mixture Model
Probabilistic clustering
"""
def gmm_clustering(X: np.ndarray, n_components: int, 
                   max_iter: int = 100) -> np.ndarray:
    """
    Simple GMM implementation
    
    Uses EM algorithm:
    1. Initialize means, covariances, weights
    2. E-step: Compute responsibilities
    3. M-step: Update parameters
    4. Repeat
    """
    n_samples, n_features = X.shape
    
    # Initialize
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = [np.eye(n_features) for _ in range(n_components)]
    weights = np.ones(n_components) / n_components
    
    for _ in range(max_iter):
        # E-step: Compute responsibilities
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            diff = X - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
            responsibilities[:, k] = weights[k] * exp_term
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        for k in range(n_components):
            resp_k = responsibilities[:, k]
            weights[k] = resp_k.sum() / n_samples
            means[k] = (resp_k[:, None] * X).sum(axis=0) / resp_k.sum()
            
            diff = X - means[k]
            covariances[k] = (resp_k[:, None, None] * 
                            diff[:, :, None] * diff[:, None, :]).sum(axis=0) / resp_k.sum()
    
    # Assign to most likely cluster
    labels = responsibilities.argmax(axis=1)
    return labels
```

## Theory

### When to Use Which

| Method | Use When | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **K-Means** | Spherical clusters, known k | Fast, simple | Need k, spherical only |
| **Hierarchical** | Unknown k, hierarchy | No k needed, dendrogram | Slow, O(n³) |
| **DBSCAN** | Arbitrary shapes, outliers | No k, finds outliers | Sensitive to parameters |
| **GMM** | Overlapping clusters | Probabilistic, soft | Slow, assumes Gaussian |

## Exercises

1. Implement all methods
2. Compare on different datasets
3. Visualize clusters
4. Choose right method

## Next Steps

- **Topic 20**: Multi-turn conversations
- **Topic 21**: Dimensionality reduction

