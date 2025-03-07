import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

class EnergyAndClustering:
    def __init__(self, energy, correlation_results, label, n_clusters_range=(2, 10), checkpoint_path="model_checkpoint.pkl"):
        self.energy = energy  # Energy values (examples, features)
        self.correlation_results = correlation_results  # Correlation results (features x outputs)
        self.label = label  # 'train' or 'test'
        self.n_clusters_range = n_clusters_range  # Range of clusters to try (from 2 to 10)
        self.checkpoint_path = checkpoint_path  # Path to save/load checkpoints

        self.selected_features = None
        self.cluster_assignments = None
        self.best_n_clusters = None

    # def select_top_features(self):
    #     """Select the top 50% most correlated features based on correlation results."""
    #     # Sum the squared values of correlations for each feature (across all outputs)
    #     feature_correlation_sums = np.sum(self.correlation_results ** 2, axis=1)
        
    #     # Get the indices of the top 50% correlated features
    #     top_features_idx = np.argsort(feature_correlation_sums)[-len(feature_correlation_sums)//2:]
    #     self.selected_features = top_features_idx
    #     return top_features_idx
    def select_top_features(self):
        """Select the top 1 most correlated feature based on correlation results."""
        # Sum the squared values of correlations for each feature (across all outputs)
        feature_correlation_sums = np.sum(self.correlation_results ** 2, axis=1)
        
        # Get the index of the top 1 correlated feature
        top_feature_idx = np.argmax(feature_correlation_sums)
        self.selected_features = [top_feature_idx]
        
        return top_feature_idx
# def select_top_features(self):
#     """Select the top 2 most correlated features based on correlation results."""
#     # Sum the squared values of correlations for each feature (across all outputs)
#     feature_correlation_sums = np.sum(self.correlation_results ** 2, axis=1)
    
#     # Get the indices of the top 2 correlated features
#     top_features_idx = np.argsort(feature_correlation_sums)[-2:]  # Get the last 2 indices after sorting
#     self.selected_features = top_features_idx
    
#     return top_features_idx


    def compute_combined_energy(self):
        """Compute combined energy of selected features by squaring their energy values."""
        combined_energy = np.zeros(self.energy.shape[0])  # One value per example

        for feature_idx in self.selected_features:
            combined_energy += self.energy[:, feature_idx]**2  # Square the energy for each selected feature

        return combined_energy

    def apply_kmeans_clustering(self, combined_energy):
        """Apply KMeans clustering and return the best number of clusters."""
        best_silhouette = -1
        best_n_clusters = None
        best_kmeans = None

        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_assignments = kmeans.fit_predict(combined_energy.reshape(-1, 1))
            silhouette = silhouette_score(combined_energy.reshape(-1, 1), cluster_assignments)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_n_clusters = n_clusters
                best_kmeans = kmeans
                self.cluster_assignments = cluster_assignments

        self.best_n_clusters = best_n_clusters
        return self.cluster_assignments

    def save_checkpoint(self):
        """Save the checkpoint with selected features and clustering model."""
        checkpoint_data = {
            'selected_features': self.selected_features,
            'cluster_model': self.best_n_clusters,
            'cluster_assignments': self.cluster_assignments
        }
        joblib.dump(checkpoint_data, self.checkpoint_path)

    def load_checkpoint(self):
        """Load checkpoint if label is test."""
        checkpoint_data = joblib.load(self.checkpoint_path)
        self.selected_features = checkpoint_data['selected_features']
        self.best_n_clusters = checkpoint_data['cluster_model']
        self.cluster_assignments = checkpoint_data['cluster_assignments']

    def run_analysis(self):
        """Run the full analysis for both train and test data."""
        if self.label == 'train':
            # Step 1: Select top correlated features
            self.select_top_features()

            # Step 2: Compute combined energy for the selected features
            combined_energy = self.compute_combined_energy()

            # Step 3: Apply KMeans clustering to the combined energy
            cluster_assignments = self.apply_kmeans_clustering(combined_energy)

            # Step 4: Save the checkpoint for future use
            self.save_checkpoint()

            return cluster_assignments, self.best_n_clusters

        elif self.label == 'test':
            # Load previously saved checkpoint (for testing phase)
            self.load_checkpoint()

            # Step 1: Compute combined energy for the test data (based on the same selected features)
            combined_energy = self.compute_combined_energy()

            # Step 2: Apply the pre-trained KMeans clustering model
            cluster_assignments = self.best_n_clusters.predict(combined_energy.reshape(-1, 1))

            return cluster_assignments
