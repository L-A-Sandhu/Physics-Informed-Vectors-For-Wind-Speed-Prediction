import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ErrorClusterer:
    def __init__(self):
        self.best_k = None
        self.cluster_centers_ = None

    def fit_predict(self, errors):
        """
        Clusters error values into 2-3 groups dynamically using silhouette score.
        Returns cluster labels ordered by error magnitude (0 = smallest errors).
        """
        data = errors.reshape(-1, 1)
        
        best_score = -1
        best_k = 2
        best_labels = None
        best_centers = None
        
        for k in [2, 3]:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            score = silhouette_score(data, kmeans.labels_)
            
            if score > best_score:
                best_score, best_k = score, k
                best_labels, best_centers = kmeans.labels_, kmeans.cluster_centers_
        
        # Order clusters by error magnitude
        sorted_indices = np.argsort(best_centers.flatten())
        label_mapping = {old: new for new, old in enumerate(sorted_indices)}
        remapped_labels = np.vectorize(label_mapping.get)(best_labels)
        
        self.best_k = best_k
        self.cluster_centers_ = best_centers[sorted_indices]
        
        return remapped_labels