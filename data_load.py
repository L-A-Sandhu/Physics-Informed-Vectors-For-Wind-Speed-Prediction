import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
from E2C import EnergyAndCorrelation
from EnergyAndClustering import EnergyAndClustering
from ClusterAnalysis import ClusterAnalysis 
class ProcessData:
    def __init__(self, X, Y):
        self.X = X  # Input data (examples, time, features, locations)
        self.Y = Y  # Output data (examples, 3)

    def run_analysis(self, correlation_method="pearson"):
        """Run the energy and correlation analysis on the given data."""
        # Create an instance of EnergyAndCorrelation class
        energy_corr = EnergyAndCorrelation(self.X, self.Y)
        
        # Compute energy and correlation
        energy_values, correlation_results = energy_corr.compute_correlation_and_plot(correlation_method)
        
        # Print energy and correlation results
        # print("Energy Values for Each Feature of Each Example:", energy_values)
        # print("Correlation Results:")
        # print(correlation_results)
        
        return energy_values, correlation_results

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

# Function to print the min, max, and shape of the dataset
def print_dataset_info(X_data, Y_data):
    print(f"X_data: Min = {X_data.min()}, Max = {X_data.max()}, Shape = {X_data.shape}")
    print(f"Y_data: Min = {Y_data.min()}, Max = {Y_data.max()}, Shape = {Y_data.shape}")

# Define the city folder path
city_folder = './data_final_3h/Esbjerg'  # Change this to the folder for the specific city

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
X_D=X_train[:, :, 1:, :] 

process_data = ProcessData(X_D, Y_train)

# Run analysis with Pearson correlation
energy_values, correlation_results= process_data.run_analysis(correlation_method="spearman")
print(f"Energy: Min = {energy_values.min()}, Max = {energy_values.max()}, Shape = {energy_values.shape}")


energy_clustering = EnergyAndClustering(energy_values, correlation_results, label="train", n_clusters_range=(2, 5))

# Run the analysis for training data
cluster_assignments, best_n_clusters = energy_clustering.run_analysis()
print(f"cluster_assignments = {cluster_assignments.min()}, Max = {cluster_assignments.max()}, Shape = {cluster_assignments.shape}")
print(best_n_clusters)
cluster_analysis = ClusterAnalysis(cluster_assignments, X_D, n_clusters=2)  # 2 clusters

cluster_analysis.plot_pdf()

