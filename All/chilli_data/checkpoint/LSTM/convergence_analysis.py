import pickle
import os
import matplotlib.pyplot as plt

def load_and_plot_train_losses(model_dirs, plot_filename, N=None):
    """
    Load training losses from specified model directories and plot the first N losses.
    
    Args:
    - model_dirs (list): List of tuples containing model names and their directory paths.
    - plot_filename (str): Filename where the plot will be saved.
    - N (int, optional): Number of initial epochs to include in the plot. If None, plot all epochs.
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, dir_path in model_dirs:
        # Assume `dir_path` is the correct path to the model directory
        filepath = os.path.join(dir_path, 'losses.pkl')
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        # Load losses
        with open(filepath, 'rb') as f:
            losses = pickle.load(f)
        
        train_losses = losses['train_losses']
        
        # If N is not None, slice the first N values
        if N is not None and len(train_losses) > N:
            train_losses = train_losses[:N]
        
        # Customize model name for the plot
        if model_name == 'Model Custom':
            plot_label = 'PIV Informed Model Train Loss'
        elif model_name == 'Model MSE':
            plot_label = 'Non PIV Informed Model Train Loss'
        else:
            plot_label = f'{model_name} Train Loss'  # Default case
        
        # Plot with bold lines
        plt.plot(train_losses, label=plot_label, linewidth=2)
    
    plt.title('Training Loss Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')  # Set x-axis label font size and weight
    plt.yticks(fontsize=16, fontweight='bold')  # Set y-axis label font size and weight
    plt.legend(fontsize=14)
    plt.grid(True)
    
    # Save plot
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments
    
    print(f"Plot saved to {plot_filename}")


model_dirs = [
    ('Model Custom', 'model_custom'),
    ('Model MSE', 'model_mse')
]
N=20
plot_filename = 'combined_losses_comparison_filtered.png'
load_and_plot_train_losses(model_dirs, plot_filename, N)
