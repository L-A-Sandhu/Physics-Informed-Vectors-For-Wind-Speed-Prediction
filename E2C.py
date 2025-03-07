import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class EnergyAndCorrelation:
    def __init__(self, X, Y):
        self.X = X  # Input data (examples, time, features, locations)
        self.Y = Y  # Output data (examples, 3)

    def compute_derivatives(self):
        """Compute first-order and second-order derivatives."""
        examples, time, features, locations = self.X.shape

        # Initialize arrays for derivatives
        first_order_derivative = np.zeros_like(self.X)
        second_order_derivative = np.zeros_like(self.X)

        # Compute first-order derivatives (change along time axis)
        first_order_derivative[:, 1:, :, :] = self.X[:, 1:, :, :] - self.X[:, :-1, :, :]

        # Compute second-order derivatives (change of first-order derivative)
        second_order_derivative[:, 2:, :, :] = first_order_derivative[:, 2:, :, :] - first_order_derivative[:, 1:-1, :, :]

        return first_order_derivative, second_order_derivative

    def compute_total_energy(self, first_order_derivative, second_order_derivative):
        """Compute total energy for each feature based on the derivatives."""
        energy = np.zeros((first_order_derivative.shape[0], self.X.shape[2]))  # Number of features

        for i in range(self.X.shape[2]):  # Loop over features
            # Sum of squared first-order and second-order derivatives for each feature
            energy[:, i] = np.sum(first_order_derivative[:, :, i, :]**2, axis=(1, 2)) + np.sum(second_order_derivative[:, :, i, :]**2, axis=(1, 2))

        return energy

    def compute_correlation_and_plot(self, correlation_method="pearson"):
        """Compute correlation of energy with outputs and plot."""
        # Compute derivatives
        first_order_derivative, second_order_derivative = self.compute_derivatives()
        
        # Compute energy
        energy = self.compute_total_energy(first_order_derivative, second_order_derivative)
        
        # Plot correlations between energy of features and outputs (Y)
        features = energy.shape[1]
        outputs = self.Y.shape[1]

        correlation_results = np.zeros((features, outputs))

        for feature_idx in range(features):
            for output_idx in range(outputs):
                if correlation_method == "pearson":
                    # Pearson correlation
                    corr = np.corrcoef(energy[:, feature_idx], self.Y[:, output_idx])[0, 1]
                elif correlation_method == "spearman":
                    # Spearman rank correlation for non-linear relationships
                    corr, _ = spearmanr(energy[:, feature_idx], self.Y[:, output_idx])

                correlation_results[feature_idx, output_idx] = corr
        
        # Create a heatmap-style plot of the correlation results
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_results, annot=True, cmap="coolwarm", xticklabels=[f'Y{i+1}' for i in range(outputs)],
                    yticklabels=[f'F{i+1}' for i in range(features)], cbar=True)
        # plt.title("Correlation between Features' Energy and Outputs")
        # plt.xlabel('Outputs')
        # plt.ylabel('Features')
        # plt.show()

        return energy, correlation_results
