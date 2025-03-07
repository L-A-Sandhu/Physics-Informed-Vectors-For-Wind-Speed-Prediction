import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ClusterAnalysis:
    def __init__(self, cluster_assignments, energy_data, n_clusters=None):
        """
        Initializes the class with cluster assignments and energy data.
        
        Parameters:
        cluster_assignments: Array of shape (examples,) indicating the cluster assignment of each example.
        energy_data: Array of shape (examples, time_steps, features, locations) containing the energy data for each example, time, feature, and location.
        n_clusters: Number of clusters to visualize and analyze. If None, it's automatically determined from cluster_assignments.
        """
        self.cluster_assignments = cluster_assignments
        self.energy_data = energy_data
        self.n_clusters = n_clusters if n_clusters else len(np.unique(cluster_assignments))  # Automatically determine clusters if not provided

        # Print the sizes of the inputs for debugging purposes
        print(f"cluster_assignments shape: {self.cluster_assignments.shape}")
        print(f"energy_data shape: {self.energy_data.shape}")
        print(f"Number of clusters: {self.n_clusters}")

    def plot_pdf(self):
        """Plots the PDF of each feature for each cluster, aggregating by time and location."""
        num_features = self.energy_data.shape[2]  # Number of features (F)
        num_time_steps = self.energy_data.shape[1]  # Number of time steps (T)
        num_locations = self.energy_data.shape[3]  # Number of locations (L)
        num_examples = self.energy_data.shape[0]  # Number of examples (E)

        # Step 1: Reorder the data to (num_examples, features, time_steps, locations)
        energy_data_reordered = np.transpose(self.energy_data, (0, 2, 1, 3))  # Shape: (num_examples, features, time_steps, locations)
        print(f"Reordered energy_data shape: {energy_data_reordered.shape}")

        # Step 2: Flatten the time_steps and locations (T * L)
        flattened_data = energy_data_reordered.reshape(num_examples, num_features, num_time_steps * num_locations)  # Shape: [num_examples, features, T * L]
        print(f"Flattened data shape: {flattened_data.shape}")

        # Get unique cluster values
        unique_clusters = np.unique(self.cluster_assignments)
        print(f"Unique clusters: {unique_clusters}")

        # Dynamically calculate rows and columns based on the number of clusters and features
        rows = len(unique_clusters)  # One row for each cluster
        cols = num_features  # One column for each feature

        # Create a figure to hold the plots
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.ravel()  # Flatten the axes array for easy iteration

        # Loop over each cluster
        for cluster_idx, cluster_val in enumerate(unique_clusters):
            # Get the data corresponding to this cluster
            cluster_data = flattened_data[self.cluster_assignments == cluster_val]
            print(f"Cluster {cluster_val} data shape: {cluster_data.shape}")

            # Loop over each feature
            for feature_idx in range(num_features):
                # Get the data for the current feature (Shape: [examples, time_steps * locations])
                feature_data = cluster_data[:, feature_idx, :]
                
                # Plot the individual examples with transparency but no legend for each line
                for example in feature_data:
                    sns.kdeplot(example, color='blue', ax=axes[cluster_idx * num_features + feature_idx], alpha=0.1, legend=False)

                # Compute and display the mean and variance for this feature
                feature_mean = np.mean(feature_data, axis=0)
                feature_variance = np.var(feature_data, axis=0)

                # Plot the KDE of the mean and show mean/variance with the legend
                sns.kdeplot(feature_mean, color='red', linestyle='--', ax=axes[cluster_idx * num_features + feature_idx], label=f"Mean: {np.mean(feature_mean):.2f}")
                
                # Display variance as text
                axes[cluster_idx * num_features + feature_idx].text(np.mean(feature_mean), 0.02, f"Var: {feature_variance.mean():.2f}", color='green', ha='center')

                # Set the title and labels
                axes[cluster_idx * num_features + feature_idx].set_title(f"Feature {feature_idx + 1} - Cluster {cluster_val} PDF")
                axes[cluster_idx * num_features + feature_idx].set_xlabel("Energy")
                axes[cluster_idx * num_features + feature_idx].set_ylabel("Density")
                
            # Add a legend to each subplot
            axes[cluster_idx * num_features].legend()

        # Adjust the layout and show the plots
        plt.tight_layout()
        plt.show()

# # Example usage:
# if __name__ == "__main__":
#     # Example Data (mocked for demonstration purposes)
#     num_examples = 10514
#     num_time_steps = 24
#     num_features = 4  # Adjusted to match your data
#     num_locations = 9

#     # Mocked energy data (10514 examples, 24 time steps, 4 features, 9 locations)
#     energy_data = np.random.rand(num_examples, num_time_steps, num_features, num_locations)

#     # Mocked cluster assignments (10514 examples, each assigned to cluster 0 or 1)
#     cluster_assignments = np.random.choice([0, 1], num_examples)

#     # Create ClusterAnalysis object
#     cluster_analysis = ClusterAnalysis(cluster_assignments, energy_data, n_clusters=2)

#     # Plot PDFs of each feature for each cluster
#     cluster_analysis.plot_pdf()
