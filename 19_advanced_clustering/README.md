# Topic 19: Advanced Clustering Methods

> 🔥 **For interviews, read these first:**
> - **`CLUSTERING_DEEP_DIVE.md`** — frontier-lab interview deep dive: K-means as coordinate descent, GMM with EM derivation, DBSCAN core/border/noise, hierarchical linkage, spectral clustering, curse of dimensionality.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

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

## Core Intuition

Clustering methods differ because they assume different cluster structure.

That is why the real interview question is often:
- what kind of geometry does each method assume?
- when is that assumption reasonable?

### K-Means

K-means is best for compact centroid-like clusters.

### Hierarchical Clustering

Hierarchical clustering is useful when you want nested structure or do not want to commit to one fixed `k` immediately.

### DBSCAN

DBSCAN defines clusters by density, not by centroids.

That makes it good for:
- arbitrary shapes
- outlier detection
- settings where `k` is unknown

### GMM

GMM is probabilistic and gives soft assignments.

That matters when cluster membership is ambiguous or overlapping.

## Technical Details Interviewers Often Want

### DBSCAN Parameter Sensitivity

DBSCAN is powerful but sensitive to:
- `eps`
- `min_samples`

### Hierarchical Clustering Cost

Hierarchical methods can be very interpretable but often scale poorly.

### GMM Assumption

GMM assumes the data can be modeled as a mixture of Gaussian components.

That gives flexibility, but it is still a modeling assumption.

## Common Failure Modes

- using K-means on arbitrary-shaped clusters
- using DBSCAN without tuning density parameters
- treating GMM as if it were just K-means with probabilities
- picking a method without thinking about geometry

## Edge Cases and Follow-Up Questions

1. Why is DBSCAN good for outliers?
2. Why can K-means fail on non-spherical clusters?
3. Why is GMM stronger than K-means for overlapping clusters?
4. Why is hierarchical clustering useful when `k` is uncertain?
5. Why is there no universally best clustering algorithm?

## What to Practice Saying Out Loud

1. How clustering assumptions differ across methods
2. Why density-based clustering is conceptually different from centroid-based clustering
3. Why soft clustering can be more realistic than hard assignment

## Industry-Standard Boilerplate Code

### Hierarchical Clustering

```python
"""
Hierarchical Clustering: Build tree of clusters
"""
import numpy as np

def hierarchical_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Naive O(n^3) agglomerative clustering — recomputes centroid distances each merge.
    Production use scipy.cluster.hierarchy or sklearn's AgglomerativeClustering (Ward
    default; uses cached pairwise distance updates instead of recomputing).
    
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
DBSCAN: Density-based clustering. Finds clusters of arbitrary shape + noise.
Two parameters: eps (neighborhood radius), min_samples (density threshold).
Output: labels[i] = cluster_id (>=0) or -1 for noise.
"""
def dbscan(X, eps, min_samples):
    n = X.shape[0]
    labels = np.full(n, -2)              # -2 = unvisited; -1 = noise; >=0 = cluster id
    cluster_id = 0

    def neighbors(i):
        return [j for j in range(n) if np.linalg.norm(X[i] - X[j]) <= eps]

    for i in range(n):
        if labels[i] != -2:               # already visited
            continue
        N_i = neighbors(i)
        if len(N_i) < min_samples:
            labels[i] = -1                # mark as noise (may be relabeled later)
            continue

        # Start new cluster from core point i
        labels[i] = cluster_id
        seeds = list(N_i)
        k = 0
        while k < len(seeds):
            q = seeds[k]
            if labels[q] == -1:           # noise → flip to border (no expansion)
                labels[q] = cluster_id
            elif labels[q] == -2:         # unvisited
                labels[q] = cluster_id
                N_q = neighbors(q)
                if len(N_q) >= min_samples:   # core: expand the seed set
                    seeds.extend(N_q)
            k += 1
        cluster_id += 1

    labels[labels == -2] = -1             # any still-unvisited become noise
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
    
    eps_reg = 1e-6                        # covariance regularization (singular fix)

    for _ in range(max_iter):
        # E-step: posterior responsibilities γ_{ik}
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            cov_k = covariances[k] + eps_reg * np.eye(n_features)
            diff = X - means[k]                                          # [N, d]
            inv_cov = np.linalg.inv(cov_k)
            mahal = np.sum(diff @ inv_cov * diff, axis=1)                # [N]
            # Full Gaussian PDF (with normalization constant)
            norm = ((2 * np.pi) ** (n_features / 2)) * np.sqrt(np.linalg.det(cov_k))
            pdf = np.exp(-0.5 * mahal) / norm
            responsibilities[:, k] = weights[k] * pdf
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)  # normalize per row

        # M-step: weighted MLE updates
        for k in range(n_components):
            resp_k = responsibilities[:, k]
            Nk = resp_k.sum()
            weights[k] = Nk / n_samples
            means[k] = (resp_k[:, None] * X).sum(axis=0) / Nk
            diff = X - means[k]
            covariances[k] = (resp_k[:, None, None]
                              * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / Nk

    # Hard assignment to most likely cluster
    return responsibilities.argmax(axis=1)
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
