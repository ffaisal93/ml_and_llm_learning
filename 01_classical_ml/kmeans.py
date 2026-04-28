"""
K-Means Clustering from Scratch - Pure Python/NumPy Version
Interview question: "Implement K-means"

Simple implementation using NumPy
"""
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means clustering algorithm
    """
    
    def __init__(self, k: int = 3, max_iters: int = 100, random_state: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_history = []
    
    def _initialize_centroids(self, X: np.ndarray):
        """Initialize centroids randomly"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        ## explain the line below with an example clearly   
        # Example:
        # X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        # n_samples = 5
        # n_features = 2
        # self.k = 2
        # np.random.choice(n_samples, self.k, replace=False) = [0, 1]
        # self.centroids = X[[0, 1]] = [[1, 2], [3, 4]]

        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid"""
        # Calculate distances from each point to each centroid
        # newaxis is used to add a new axis to the centroids array, so that it can be broadcasted to the X array
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign to nearest centroid
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray):
        """Update centroids based on cluster assignments"""
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = cluster_points.mean(axis=0)
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares"""
        ## what is the purpose of this function?
        # The purpose of this function is to calculate the within-cluster sum of squares.
        # The within-cluster sum of squares is the sum of the squared distances of each point to its centroid.
        # This is used to measure the quality of the clustering.
        # The lower the inertia, the better the clustering.
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia
    
    def fit(self, X: np.ndarray):
        """Fit K-means to data"""
        self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            old_centroids = self.centroids.copy()
            self._update_centroids(X, labels)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, labels)
            self.inertia_history.append(inertia)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        self.labels = labels
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster for new data"""
        return self._assign_clusters(X)


# Usage Example
if __name__ == "__main__":
    # Generate sample data (3 clusters)
    np.random.seed(42)
    n_samples = 300
    
    # Cluster 1
    X1 = np.random.randn(n_samples//3, 2) + np.array([2, 2])
    
    # Cluster 2
    X2 = np.random.randn(n_samples//3, 2) + np.array([-2, 2])
    
    # Cluster 3
    X3 = np.random.randn(n_samples//3, 2) + np.array([0, -2])
    
    # vstack is used to stack the arrays vertically, so that the data is combined into a single array
    # this is done to create a single dataset of all the data points from all the clusters
    # examples: np.vstack([[1, 2], [3, 4], [5, 6]]) = [[1, 2], [3, 4], [5, 6]]
    X = np.vstack([X1, X2, X3])
    
    # Fit K-means
    model = KMeans(k=3, max_iters=100)
    model.fit(X)
    
    # Predict
    labels = model.predict(X)
    
    print(f"Number of clusters: {model.k}")
    print(f"Centroids:\n{model.centroids}")
    print(f"Final inertia: {model.inertia_history[-1]:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.inertia_history)
    plt.xlabel('Iteration')
    plt.ylabel('Inertia')
    plt.title('Inertia History')
    
    plt.tight_layout()
    plt.savefig('kmeans.png')
    print("Plot saved to kmeans.png")

