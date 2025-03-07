import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class FuzzyGaussianError:
    def __init__(self, error_values=None):
        if error_values is not None:
            self.train_errors = np.array(error_values)
            self.mean = np.mean(self.train_errors)
            self.std = np.std(self.train_errors)
            
            # Define the fuzzy membership functions using Gaussian functions
            self.low_mean = self.mean - self.std
            self.medium_mean = self.mean
            self.high_mean = self.mean + self.std
            self.sigma = self.std
            
            # Fuzzify the training data at initialization
            self.train_memberships = self.fuzzify(self.train_errors)
        else:
            self.mean = None
            self.std = None
            self.low_mean = None
            self.medium_mean = None
            self.high_mean = None
            self.sigma = None
            self.train_memberships = None

    def gaussian_membership(self, x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    def fuzzify(self, error_values):
        if self.mean is None or self.std is None:
            raise ValueError("Training data must be provided before fuzzifying.")
        
        memberships = np.array([[
            self.gaussian_membership(error, self.low_mean, self.sigma),
            self.gaussian_membership(error, self.medium_mean, self.sigma),
            self.gaussian_membership(error, self.high_mean, self.sigma)
        ] for error in error_values])
        
        return memberships

    def plot_fuzzy_sets(self, error_values=None, ax=None):
        # Plot the membership functions for Low, Medium, and High
        x = np.linspace(self.mean - 3*self.std, self.mean + 3*self.std, 500)
        low_membership = self.gaussian_membership(x, self.low_mean, self.sigma)
        medium_membership = self.gaussian_membership(x, self.medium_mean, self.sigma)
        high_membership = self.gaussian_membership(x, self.high_mean, self.sigma)

        if ax is None:
            ax = plt.gca()

        ax.plot(x, low_membership, label='Low', color='blue')
        ax.plot(x, medium_membership, label='Medium', color='green')
        ax.plot(x, high_membership, label='High', color='red')

        if error_values is not None:
            max_errors_to_plot = 10  # Limit the number of error lines plotted
            subset_errors = error_values[:min(len(error_values), max_errors_to_plot)]  # Take first 'max_errors_to_plot' errors
            ax.vlines(subset_errors, 0, 1, colors='black', linestyle='dashed', label="Errors" if len(subset_errors) <= max_errors_to_plot else "")

        ax.set_title("Fuzzy Membership Functions")
        ax.set_xlabel("Error Value")
        ax.set_ylabel("Membership Degree")
        ax.grid(True)

        # Only show the legend once for each plot
        if error_values is not None:
            ax.legend(loc='upper right', frameon=False)

    def get_memberships(self, error_values):
        return self.fuzzify(error_values)
