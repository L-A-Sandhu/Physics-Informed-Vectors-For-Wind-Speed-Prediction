import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
from physics_models import PhysicsModelWindError
from fuzz import FuzzyGaussianError
# from custom_model import VolatilityAwareWindPredictor
from custom_model import VolatilityAwareWindPredictor
import numpy as np
import tensorflow as tf
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
from tensorflow.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_conf)

def load_data_from_directory(city_folder):
    # Lists to store all inputs (X) and outputs (Y)
    X_data = []
    Y_data = []

    # Iterate over all subdirectories (train, valid, test)
    for dataset_name in ['train', 'valid', 'test']:
        dataset_folder = os.path.join(city_folder, dataset_name)

        # Iterate over all numpy files in the directory
        for file_name in os.listdir(dataset_folder):
            if file_name.endswith('.npy'):
                # Load the numpy file
                npy_file_path = os.path.join(dataset_folder, file_name)
                data = np.load(npy_file_path)

                # The last dimension (index 2) contains the wind speed outputs
                # Extract the wind speed values from the filename (assumes format is "<example_number>_<value_1>_<value_2>_<value_3>.npy")
                parts = file_name.split('_')
                value_1 = float(parts[-3])
                value_2 = float(parts[-2])
                value_3 = float(parts[-1].replace('.npy', ''))

                # Append the input data (the entire 24 time steps window) and output data (the 3 future C_WS50M values)
                X_data.append(data)
                Y_data.append([value_1, value_2, value_3])
    
    # Convert lists to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    return X_data, Y_data


def plot_errors_comparison(errors_N, errors_A, errors_NA, num_examples=20):
    # Select first 'num_examples' for visualization
    indices = np.arange(num_examples)
    n_errors = errors_N[:num_examples]
    a_errors = errors_A[:num_examples]
    na_errors = errors_NA[:num_examples]

    # Set up plot
    plt.figure(figsize=(15, 6))
    bar_width = 0.25
    
    # Create bars
    bars1 = plt.bar(indices - bar_width, n_errors, width=bar_width,
                   label='Navier-Stokes (N)', alpha=0.7)
    bars2 = plt.bar(indices, a_errors, width=bar_width,
                   label='Advection (A)', alpha=0.7)
    bars3 = plt.bar(indices + bar_width, na_errors, width=bar_width,
                   label='Combined (NA)', alpha=0.7)

    # Formatting
    plt.title(f'First {num_examples} Examples: Model Comparison')
    plt.xlabel('Example Index')
    plt.ylabel('Normalized MSE')
    plt.xticks(indices, [f'Ex {i}' for i in range(num_examples)])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add error values on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', rotation=90, fontsize=8)

    plt.tight_layout()
    plt.show()


# Function to print the min, max, and shape of the dataset
def print_dataset_info(X_data, Y_data):
    print(f"X_data: Min = {X_data.min()}, Max = {X_data.max()}, Shape = {X_data.shape}")
    print(f"Y_data: Min = {Y_data.min()}, Max = {Y_data.max()}, Shape = {Y_data.shape}")

# Define the city folder path
city_folder = '../../data_final_3h/Esbjerg'  # Change this to the folder for the specific city

# Load the data for the specified city
X_data, Y_data = load_data_from_directory(city_folder)

# Since the data is already split into train, valid, and test subdirectories, we just need to load the data for each
X_train, Y_train = X_data[:len(os.listdir(os.path.join(city_folder, 'train')))], Y_data[:len(os.listdir(os.path.join(city_folder, 'train')))]
X_valid, Y_valid = X_data[len(os.listdir(os.path.join(city_folder, 'train'))):len(os.listdir(os.path.join(city_folder, 'train'))) + len(os.listdir(os.path.join(city_folder, 'valid')))], Y_data[len(os.listdir(os.path.join(city_folder, 'train'))):len(os.listdir(os.path.join(city_folder, 'train'))) + len(os.listdir(os.path.join(city_folder, 'valid')))]
X_test, Y_test = X_data[len(os.listdir(os.path.join(city_folder, 'train'))) + len(os.listdir(os.path.join(city_folder, 'valid'))):], Y_data[len(os.listdir(os.path.join(city_folder, 'train'))) + len(os.listdir(os.path.join(city_folder, 'valid'))):]

# Print the dataset information
print("Training Set Info:")
print_dataset_info(X_train, Y_train)
print("\nValidation Set Info:")
print_dataset_info(X_valid, Y_valid)
print("\nTest Set Info:")
print_dataset_info(X_test, Y_test)

print("\nFirst example spatial data (Hour 0):")
print("Locations: Center, N, S, E, W, NE, NW, SE, SW")
print("Wind Speeds:", X_train[0,0,1,:])
print("Pressures:", X_train[0,0,2,:])
# Usage
model = PhysicsModelWindError(X_train)
errors_N_train = model.compute_errors('N')['N']
model = PhysicsModelWindError(X_test)
errors_N_test = model.compute_errors('N')['N']
model = PhysicsModelWindError(X_valid)
errors_N_valid = model.compute_errors('N')['N']

print("Unique values of errors_N:::::::", np.shape(errors_N_train),np.min(errors_N_train),np.max(errors_N_train))
fuzzy_train = FuzzyGaussianError(errors_N_train)
memberships_train = fuzzy_train.get_memberships(errors_N_train)  # Pass the training data error values

# Get memberships for test and validation data
memberships_test = fuzzy_train.get_memberships(errors_N_test)
memberships_valid = fuzzy_train.get_memberships(errors_N_valid)

print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
print(np.shape(memberships_train),np.shape(memberships_test),np.shape(memberships_valid))

print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
print('####################################################################################################################################################################3')
# Create a figure for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot train, test, and valid memberships on the same figure with subplots
fuzzy_train.plot_fuzzy_sets(errors_N_train, ax=axes[0])  # Train
fuzzy_train.plot_fuzzy_sets(errors_N_test, ax=axes[1])   # Test
fuzzy_train.plot_fuzzy_sets(errors_N_valid, ax=axes[2])  # Valid

# Show the plot
plt.tight_layout()
# plt.show()












print('####################################################################################################################################################################3')
############################################################################################################################################################################
# old method
predictor = VolatilityAwareWindPredictor(input_shape=(24, 5, 9), output_steps=3, min_val=0.02, max_val=26.08)

# Build models
custom_model = predictor.build_custom_model()
baseline_model = predictor.build_baseline_model()

# Preprocess data
X_train, X_val, X_test = predictor.preprocess_data(X_train, X_valid, X_test)

# Train custom model with hybrid loss
custom_model = predictor.train(X_train, Y_train, X_valid, Y_valid, memberships_train, memberships_valid, model_type='custom')
custom_metrics = predictor.evaluate(X_test, Y_test, memberships_test=memberships_test, model_type='custom')
print("custom_metrics::::::::::", custom_metrics)
# Train baseline model with MSE loss
baseline_model = predictor.train(X_train, Y_train, X_valid, Y_valid, memberships_train=None, memberships_val=None, model_type='baseline')

# Evaluate models
custom_metrics = predictor.evaluate(X_test, Y_test, memberships_test=memberships_test, model_type='custom')
baseline_metrics = predictor.evaluate(X_test, Y_test, memberships_test=None, model_type='baseline')

print("custom_metrics::::::::::", custom_metrics)

print("baseline_metrics::::::::::", baseline_metrics)

# #############################################################################################################################################################################3

# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################

# #############################################################################################################################################################################
# #old method
# predictor = VolatilityAwareWindPredictor(input_shape=(24, 5, 9), output_steps=3, min_val=0.02, max_val=26.08,patience_1=10,patience_2=10)

# # Build models
# custom_model_MH = predictor.build_custom_model("MH")
# custom_model_LM = predictor.build_custom_model("LM")
# baseline_model = predictor.build_baseline_model()

# # Preprocess data
# X_train, X_val, X_test, X_MH_train, X_MH_val, X_MH_test, X_LM_train, X_LM_val, X_LM_test, Y_MH_train, Y_MH_val, Y_MH_test, Y_LM_train, Y_LM_val, Y_LM_test, mem_MH_train, mem_MH_val, mem_MH_test,mem_LM_train, mem_LM_val, mem_LM_test= predictor.preprocess_data(X_train, X_valid, X_test,memberships_train,memberships_valid,memberships_test,Y_train,Y_valid,Y_test)

# # Train custom model with hybrid loss and MH
# custom_model_MH = predictor.train(X_MH_train, Y_MH_train, X_MH_val, Y_MH_val, mem_MH_train, mem_MH_val, model_type='custom')
# y_test_unscaled_MH, y_pred_unscaled_MH = predictor.pred(X_MH_test, Y_MH_test, memberships_test=mem_MH_test, model_type='custom')
# print("MH_TRUE:::::",np.shape(y_test_unscaled_MH),"MH_PRED:::::",np.shape(y_pred_unscaled_MH) )

# # Train custom model with hybrid loss and LM
# custom_model_LM = predictor.train(X_LM_train, Y_LM_train, X_LM_val, Y_LM_val, mem_LM_train, mem_LM_val, model_type='custom')
# y_test_unscaled_LM, y_pred_unscaled_LM = predictor.pred(X_LM_test, Y_LM_test, memberships_test=mem_LM_test, model_type='custom')
# print("MH_TRUE:::::",np.shape(y_test_unscaled_LM),"MH_PRED:::::",np.shape(y_pred_unscaled_LM) )

# # Join their O/P
# TRUE_combined = np.concatenate((y_test_unscaled_MH, y_test_unscaled_LM), axis=0)
# PRED_combined = np.concatenate((y_pred_unscaled_MH, y_pred_unscaled_LM), axis=0)
# custom_metrics = predictor.evaluate(TRUE_combined, PRED_combined)
# print("custom_metrics MH::::::::::", custom_metrics)




# # # Train baseline model with MSE loss
# # baseline_model = predictor.train(X_train, Y_train, X_valid, Y_valid, memberships_train=None, memberships_val=None, model_type='baseline')

# # # Evaluate models
# # custom_metrics = predictor.evaluate(TRUE_combined, PRED_combined)
# # y_test_unscaled, y_pred_unscaled= predictor.pred(X_test, Y_test, memberships_test=None, model_type='baseline')
# # baseline_metrics = predictor.evaluate(y_test_unscaled, y_pred_unscaled)
# # print("custom_metrics::::::::::", custom_metrics)

# # print("baseline_metrics::::::::::", baseline_metrics)




# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################


# import json
# import pandas as pd

# # Assuming custom_metrics and baseline_metrics are defined
# combined_metrics = {
#     'custom_metrics': custom_metrics,
#     'baseline_metrics': baseline_metrics
# }

# # Save as JSON with clear tagging
# json_path = './checkpoint/results.json'
# with open(json_path, 'w') as json_file:
#     json.dump(combined_metrics, json_file, indent=4)

# # Save as CSV with clear differentiation
# csv_path = './checkpoint/results.csv'

# # Convert the combined dictionary into a format suitable for CSV
# # First, we'll flatten the structure for each group (custom_metrics and baseline_metrics)
# combined_metrics_flat = {
#     f"custom_{key}": value for key, value in custom_metrics.items()
# }
# combined_metrics_flat.update({
#     f"baseline_{key}": value for key, value in baseline_metrics.items()
# })

# # Now, save the combined and flattened dictionary to a CSV
# combined_metrics_df = pd.DataFrame([combined_metrics_flat])
# combined_metrics_df.to_csv(csv_path, index=False)  # Save DataFrame to CSV




# # Results
# print(f"Optimal clusters: {clusterer.best_k}")
# print(f"Cluster centers: {clusterer.cluster_centers_}")
# print(f"First 10 labels: {label[:10]}")
# print(f"Label shape: {label.shape}")  # Should be (12853,)
# print("Unique values of labels:::::::", np.shape(label),np.min(label),np.max(label), np.unique(label))
# # To access individual example errors:
# # errors_N[0] = MSE for first example using Navier-Stokes
# # errors_A[0] = MSE for first example using Advection

# # Usage - plot first 20 examples
# plot_errors_comparison(errors_N, errors_A, errors_NA, num_examples=50)

# # For statistical distribution (all examples):
# plt.figure(figsize=(10, 5))
# plt.hist([errors_N, errors_A, errors_NA], bins=50, 
#          label=['Navier-Stokes', 'Advection', 'Combined'])
# plt.title('Error Distribution Across All Examples')
# plt.xlabel('Normalized MSE')
# plt.ylabel('Count')
# plt.legend()
# plt.show()