import pickle
import os
import matplotlib.pyplot as plt

def load_and_plot_losses(model_dirs, plot_filename, N=None):
    """
    Load training and validation losses from specified model directories and plot the first N losses,
    excluding epochs where either loss exceeds 1.
    
    Args:
    - model_dirs (list): List of tuples containing model names and their directory paths.
    - plot_filename (str): Filename where the plot will be saved.
    - N (int, optional): Number of initial epochs to include in the plot. If None, plot all epochs.
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, dir_path in model_dirs:
        # Construct the file path
        filepath = os.path.join(dir_path, 'losses.pkl')
        
        # Load losses
        with open(filepath, 'rb') as f:
            losses = pickle.load(f)
            
        train_losses = losses['train_losses']
        val_losses = losses['val_losses']
        
        # Filter epochs where either loss exceeds 1
        filtered_epochs = [(train, val) for train, val in zip(train_losses, val_losses) if train <= 1 and val <= 1]
        
        # If N is not None, slice the first N values after filtering
        if N is not None:
            filtered_epochs = filtered_epochs[:N]
        
        # Unzip the filtered losses
        filtered_train_losses, filtered_val_losses = zip(*filtered_epochs)
        
        # Plot
        plt.plot(filtered_train_losses, label=f'{model_name} Train Loss')
        plt.plot(filtered_val_losses, label=f'{model_name} Validation Loss', linestyle='--')
    
    plt.title('Training and Validation Losses Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments
    
    print(f"Plot saved to {plot_filename}")

# Example usage with N value:
N = 10  # For example, to plot only the first 10 epochs after filtering
model_dirs = [
    ('Model Custom', 'model_custom'),
    ('Model MSE', 'model_mse')
]
plot_filename = 'combined_losses_comparison_filtered.png'
load_and_plot_losses(model_dirs, plot_filename, N)
