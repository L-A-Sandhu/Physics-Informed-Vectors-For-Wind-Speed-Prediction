import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def load_plot_combined_losses_and_slopes(model_dirs, plot_filename_combined_loss, plot_filename_combined_slope, N=None):
    plt.figure(figsize=(10, 8))

    for model_name, dir_path in model_dirs:
        filepath = os.path.join(dir_path, 'losses.pkl')

        with open(filepath, 'rb') as f:
            losses = pickle.load(f)

        train_losses = losses['train_losses'][:N]
        plt.plot(train_losses, label=f'{model_name} Train Loss')

    plt.title('Training Losses Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename_combined_loss)
    plt.close()

    plt.figure(figsize=(10, 8))

    for model_name, dir_path in model_dirs:
        filepath = os.path.join(dir_path, 'losses.pkl')

        with open(filepath, 'rb') as f:
            losses = pickle.load(f)

        train_losses = losses['train_losses']
        train_slopes = np.diff(train_losses)[:N-1]  # N-1 since diff reduces length by 1
        plt.plot(train_slopes, label=f'{model_name} Train Loss Slope')

    plt.title('Training Loss Slopes Comparison')
    plt.xlabel('Epoch Intervals')
    plt.ylabel('Loss Slope')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename_combined_slope)
    plt.close()

# Example usage:
model_dirs = [
    ('Model Custom', 'model_custom'),
    ('Model MSE', 'model_mse')
]
plot_filename_combined_loss = 'combined_training_losses.png'
plot_filename_combined_slope = 'combined_training_loss_slopes.png'
N = 100  # Adjust N based on your data
load_plot_combined_losses_and_slopes(model_dirs, plot_filename_combined_loss, plot_filename_combined_slope, N)
