# Topic 23: Clustering Evaluation

> 🔥 **For interviews, read these first:**
> - **`CLUSTERING_EVALUATION_DEEP_DIVE.md`** — frontier-lab deep dive: internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn), external metrics (ARI, NMI, V-measure, purity, pairwise F), choosing $K$ (elbow, silhouette, gap statistic, stability), bootstrap stability validation, common pitfalls.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

## What You'll Learn

This topic teaches you how to evaluate clustering:
- Silhouette Score
- Inertia (Within-cluster sum of squares)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Simple implementations

## Why We Need This

### Interview Importance
- **Common question**: "How do you evaluate clustering?"
- **Practical knowledge**: Need to measure quality
- **Problem-solving**: Choose right metric

### Real-World Application
- **Model selection**: Choose best clustering
- **Parameter tuning**: Find optimal k
- **Quality assessment**: Measure cluster quality

## Industry Use Cases

### 1. **Silhouette Score**
**Use Case**: General clustering evaluation
- Works without ground truth
- Measures cohesion and separation
- Range: -1 to 1 (higher is better)

### 2. **ARI (Adjusted Rand Index)**
**Use Case**: When ground truth available
- Compares to true labels
- Adjusted for chance
- Range: -1 to 1 (higher is better)

### 3. **NMI (Normalized Mutual Information)**
**Use Case**: Information-theoretic measure
- Measures shared information
- Normalized to [0, 1]
- Higher is better

## Core Intuition

Clustering is unsupervised, which makes evaluation harder than standard classification.

The main question is:
- what does "good clustering" even mean?

Different metrics answer different versions of that question.

### Silhouette Score

Silhouette measures two things at once:
- how close a point is to its own cluster
- how far it is from the nearest competing cluster

That makes it useful when you do not have labels.

### Inertia

Inertia measures compactness inside clusters.

It is useful, but it has an important limitation:
- inertia almost always decreases as you increase `k`

So it cannot be interpreted alone without comparing model complexity.

### ARI and NMI

When ground-truth labels exist, ARI and NMI compare the clustering to a reference labeling.

That makes them external evaluation metrics rather than purely intrinsic clustering scores.

## Technical Details Interviewers Often Want

### Why Silhouette Is Not Always Enough

A clustering can have reasonable silhouette and still be wrong for the business or scientific task.

Why?
- clusters may be geometrically neat but semantically useless
- some data structures are not well captured by distance-based cohesion/separation

### Why Inertia Needs Context

Since inertia usually decreases with more clusters, a lower inertia does not automatically mean a better clustering.

This is why the elbow method is heuristic rather than a theorem.

### External vs Internal Metrics

This is a useful interview distinction:
- **internal metrics** use only the clustering and geometry
- **external metrics** compare against ground-truth labels

## Common Failure Modes

- treating inertia as a standalone model-selection criterion
- interpreting a high clustering metric as proof of business usefulness
- comparing clusterings without checking whether the metric matches the use case
- forgetting that some metrics need ground truth and some do not

## Edge Cases and Follow-Up Questions

1. Why does inertia almost always improve when `k` increases?
2. Why can silhouette be misleading for irregular cluster shapes?
3. When should you use ARI or NMI instead of silhouette?
4. Why is cluster evaluation more ambiguous than classification evaluation?
5. Why is the elbow method only a heuristic?

## What to Practice Saying Out Loud

1. The difference between intrinsic and extrinsic clustering metrics
2. Why lower inertia is not automatically better
3. Why clustering quality depends on the task definition

## Industry-Standard Boilerplate Code

### Silhouette Score

```python
"""
Silhouette Score: Measure clustering quality
"""
import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score
    
    For each point:
    - a(i) = average distance to points in same cluster
    - b(i) = average distance to points in nearest other cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Returns: Average silhouette score
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        # Distance to points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(X[i], x)])
        else:
            a_i = 0.0
        
        # Distance to nearest other cluster
        min_b_i = float('inf')
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                b_i = np.mean([np.linalg.norm(X[i] - x) for x in other_cluster])
                min_b_i = min(min_b_i, b_i)
        
        b_i = min_b_i
        
        # Silhouette score for this point
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)
```

### Inertia (WCSS)

```python
"""
Inertia: Within-cluster sum of squares
Lower is better
"""
def inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Inertia = Sum of squared distances to centroids
    """
    unique_labels = np.unique(labels)
    total_inertia = 0.0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        
        for point in cluster_points:
            total_inertia += np.linalg.norm(point - centroid)**2
    
    return total_inertia
```

### Adjusted Rand Index

```python
"""
ARI: Adjusted Rand Index
Compares clustering to ground truth
"""
def adjusted_rand_index(labels_true: np.ndarray, 
                       labels_pred: np.ndarray) -> float:
    """
    ARI: Measures agreement between two clusterings
    
    Adjusted for chance (random clustering gets ~0)
    Range: -1 to 1 (1 = perfect match)
    """
    # Contingency table
    n_samples = len(labels_true)
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    contingency = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency[i, j] = np.sum((labels_true == true_label) & 
                                     (labels_pred == pred_label))
    
    # Sum over contingency table
    sum_combinations = np.sum([np.sum(contingency[i]) * (np.sum(contingency[i]) - 1) / 2
                              for i in range(len(unique_true))])
    
    sum_pred = np.sum([np.sum(contingency[:, j]) * (np.sum(contingency[:, j]) - 1) / 2
                      for j in range(len(unique_pred))])
    
    sum_true = np.sum([contingency[i, j] * (contingency[i, j] - 1) / 2
                      for i in range(len(unique_true))
                      for j in range(len(unique_pred))])
    
    # ARI formula
    n_choose_2 = n_samples * (n_samples - 1) / 2
    expected_index = sum_combinations * sum_pred / n_choose_2
    max_index = (sum_combinations + sum_pred) / 2
    
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_true - expected_index) / (max_index - expected_index)
    return ari
```

## Theory

### When to Use Which Metric

| Metric | Ground Truth Needed | Use Case |
|--------|-------------------|----------|
| **Silhouette** | No | General evaluation |
| **Inertia** | No | Compare different k |
| **ARI** | Yes | Compare to true labels |
| **NMI** | Yes | Information-theoretic |

### Choosing k (Number of Clusters)

1. **Elbow Method**: Plot inertia vs k, find elbow
2. **Silhouette**: Choose k with highest silhouette
3. **Domain knowledge**: Use business requirements

## Exercises

1. Implement all metrics
2. Evaluate different clusterings
3. Find optimal k
4. Compare metrics

## Next Steps

- **Topic 24**: Linear algebra Q&A
- **Topic 25**: Final review
