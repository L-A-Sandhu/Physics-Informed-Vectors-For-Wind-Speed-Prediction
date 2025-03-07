import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.preprocessing import MinMaxScaler

# Function to read the min and max values from the JSON file
def read_min_max_from_json(city):
    # Define the file path for the JSON
    file_path = f"./data_final/{city}/{city}_min_max.json"
    
    # Read the JSON file
    with open(file_path, 'r') as json_file:
        min_max_values = json.load(json_file)
    
    return min_max_values

# Function to normalize the P value
def normalize_P(P, city):
    # Read the min and max values from the saved JSON file
    min_max_values = read_min_max_from_json(city)
    
    # Get the min and max for C_WS50M
    C_WS50M_min = min_max_values['C_WS50M_min']
    C_WS50M_max = min_max_values['C_WS50M_max']
    
    # Normalize P using the min and max values
    P_normalized = (P - C_WS50M_min) / (C_WS50M_max - C_WS50M_min)
    
    return P_normalized


def normalize_all_columns(df, column_names):
    scaler = MinMaxScaler()
    df[column_names] = scaler.fit_transform(df[column_names])  # Normalize all columns
    return df, scaler

def save_min_max_values_for_C_WS50M(df, city):
    # Calculate and save min, max, and mean values BEFORE normalization
    min_max_dict = {
        'C_WS50M_min': df['C_WS50M'].min(),
        'C_WS50M_max': df['C_WS50M'].max(),
        'C_WS50M_mean': df['C_WS50M'].mean()
    }

    # Define the output directory for the city
    output_dir = f'./data_final/{city}'

    # Check if the directory exists, and if so, delete it and its contents
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and its contents
        print("Deleted old directory.")

    # Create the directory for the city
    os.makedirs(output_dir, exist_ok=True)  # Create city folder again

    # Define the file path for the JSON
    file_path = f"{output_dir}/{city}_min_max.json"

    # Write the min, max, and mean values to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(min_max_dict, json_file)

    print(f"Min-max values saved to {file_path}")

def process_city_data(city):
    # Load the combined CSV for the city
    file_path = f"./data/{city}/combined.csv"
    df = pd.read_csv(file_path)

    # List of all columns to normalize, excluding temporal columns
    column_names = ['YEAR', 'MO', 'DY', 'HR', 
                    'C_WD50M', 'C_WS50M', 'C_PS', 'C_T2M', 'C_QV2M', 
                    'N_WD50M', 'N_WS50M', 'N_PS', 'N_T2M', 'N_QV2M', 
                    'S_WD50M', 'S_WS50M', 'S_PS', 'S_T2M', 'S_QV2M', 
                    'E_WD50M', 'E_WS50M', 'E_PS', 'E_T2M', 'E_QV2M', 
                    'W_WD50M', 'W_WS50M', 'W_PS', 'W_T2M', 'W_QV2M', 
                    'NE_WD50M', 'NE_WS50M', 'NE_PS', 'NE_T2M', 'NE_QV2M', 
                    'NW_WD50M', 'NW_WS50M', 'NW_PS', 'NW_T2M', 'NW_QV2M', 
                    'SE_WD50M', 'SE_WS50M', 'SE_PS', 'SE_T2M', 'SE_QV2M', 
                    'SW_WD50M', 'SW_WS50M', 'SW_PS', 'SW_T2M', 'SW_QV2M']
    
    # Save the min, max, and mean values for C_WS50M BEFORE normalization
    save_min_max_values_for_C_WS50M(df, city)

    # Normalize the entire dataset (all columns)
    # df_normalized, scaler = normalize_all_columns(df, [col for col in column_names if col not in ['YEAR', 'MO', 'DY', 'HR']])
    df_normalized, scaler = normalize_all_columns(df, column_names)

    # Split the data into train, validation, and test sets
    total_rows = len(df_normalized)
    train_end = int(0.8 * total_rows)
    valid_end = int(0.9 * total_rows)

    train_data = df_normalized.iloc[:train_end]
    valid_data = df_normalized.iloc[train_end:valid_end]
    test_data = df_normalized.iloc[valid_end:]

    # Create the output directories for labels 0 and 1, and for train, valid, test sets
    city_folder = f'./data_final/{city}'
    os.makedirs(f'{city_folder}/train/0', exist_ok=True)
    os.makedirs(f'{city_folder}/train/1', exist_ok=True)
    os.makedirs(f'{city_folder}/valid/0', exist_ok=True)
    os.makedirs(f'{city_folder}/valid/1', exist_ok=True)
    os.makedirs(f'{city_folder}/test/0', exist_ok=True)
    os.makedirs(f'{city_folder}/test/1', exist_ok=True)

    # Function to save npy files for each dataset (train, valid, test)
    def save_npy_files(data, city, dataset_name):
        step = 6  # Step size to move forward (you can adjust this)
        T = 12  # T consecutive rows
        P = 1  # Threshold for the difference in C_WS50M to label as 1
        P_normalized = normalize_P(P, city)  # Normalize P using the pre-normalized value
        
        # Example counter to number the files
        example_counter = 1

        for i in range(0, len(data) - T, step):
            # Ensure we do not go out of bounds when accessing the next row
            if i + T + 1 < len(data):  # This prevents the out-of-bounds error
                X = data.iloc[i:i+T][column_names].values
                label = 0
                # Compare the values of C_WS50M
                if abs(data.iloc[i+T]['C_WS50M'] - data.iloc[i+T+1]['C_WS50M']) > P_normalized:
                    label = 1

                # Get the normalized C_WS50M value at the i+T+1 step
                C_WS50M_value = data.iloc[i+T+1]['C_WS50M']

                # Save as .npy in the respective train, valid, test folder and label folder
                npy_file_name = f"{city_folder}/{dataset_name}/{label}/{example_counter}_C_WS50M_{i+T+1}_{C_WS50M_value:.4f}.npy"
                np.save(npy_file_name, X)

                # Increment the example counter for the next file
                example_counter += 1

                # print(f"Saved {npy_file_name} with label {label}")

    # Process the datasets: train, valid, and test
    save_npy_files(train_data, city, 'train')
    save_npy_files(valid_data, city, 'valid')
    save_npy_files(test_data, city, 'test')

# Process data for both cities
process_city_data("AKTAU")
process_city_data("Esbjerg")
