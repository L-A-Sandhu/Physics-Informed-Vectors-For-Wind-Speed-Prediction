import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.preprocessing import MinMaxScaler
import pdb
# Function to read the min and max values from the JSON file
def read_min_max_from_json(city):
    file_path = f"./data_final_3h/{city}/{city}_min_max.json"
    with open(file_path, 'r') as json_file:
        min_max_values = json.load(json_file)
    return min_max_values

# Function to normalize all columns
def normalize_all_columns(df, column_names):
    scaler = MinMaxScaler()
    df[column_names] = scaler.fit_transform(df[column_names])  # Normalize all columns
    return df, scaler

# Function to save min-max values for C_WS50M
def save_min_max_values_for_C_WS50M(df, city):
    min_max_dict = {
        'C_WS50M_min': df['C_WS50M'].min(),
        'C_WS50M_max': df['C_WS50M'].max(),
        'C_WS50M_mean': df['C_WS50M'].mean()
    }

    output_dir = f'./data_final_3h/{city}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("Deleted old directory.")
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{output_dir}/{city}_min_max.json"
    with open(file_path, 'w') as json_file:
        json.dump(min_max_dict, json_file)

    print(f"Min-max values saved to {file_path}")

# Function to process the data
def process_city_data(city):
    # Load the combined CSV for the city
    file_path = f"./data/{city}/combined.csv"
    df = pd.read_csv(file_path)

    # Remove unwanted columns
    df = df.drop(columns=['YEAR', 'MO', 'DY', 'HR'])

    # Define the columns to normalize (all remaining columns)
    column_names = ['C_WD50M', 'C_WS50M', 'C_PS', 'C_T2M', 'C_QV2M', 
                    'N_WD50M', 'N_WS50M', 'N_PS', 'N_T2M', 'N_QV2M', 
                    'S_WD50M', 'S_WS50M', 'S_PS', 'S_T2M', 'S_QV2M', 
                    'E_WD50M', 'E_WS50M', 'E_PS', 'E_T2M', 'E_QV2M', 
                    'W_WD50M', 'W_WS50M', 'W_PS', 'W_T2M', 'W_QV2M', 
                    'NE_WD50M', 'NE_WS50M', 'NE_PS', 'NE_T2M', 'NE_QV2M', 
                    'NW_WD50M', 'NW_WS50M', 'NW_PS', 'NW_T2M', 'NW_QV2M', 
                    'SE_WD50M', 'SE_WS50M', 'SE_PS', 'SE_T2M', 'SE_QV2M', 
                    'SW_WD50M', 'SW_WS50M', 'SW_PS', 'SW_T2M', 'SW_QV2M']
    
    # Save min-max values for C_WS50M
    save_min_max_values_for_C_WS50M(df, city)

    # Normalize the entire dataset
    df_normalized, scaler = normalize_all_columns(df, column_names)

    # Split the data into train, validation, and test sets
    total_rows = len(df_normalized)
    train_end = int(0.8 * total_rows)
    valid_end = int(0.9 * total_rows)

    train_data = df_normalized.iloc[:train_end]
    valid_data = df_normalized.iloc[train_end:valid_end]
    test_data = df_normalized.iloc[valid_end:]

    # Create the output directories for train, valid, test sets
    city_folder = f'./data_final_3h/{city}'
    os.makedirs(f'{city_folder}/train', exist_ok=True)
    os.makedirs(f'{city_folder}/valid', exist_ok=True)
    os.makedirs(f'{city_folder}/test', exist_ok=True)

    # Function to save npy files for each dataset
    def save_npy_files(data, city, dataset_name, column_names):
        T = 24  # Number of consecutive rows in the window
        step = 6  # Step size to move forward (adjust this as needed)
        
        # Example counter to number the files
        example_counter = 1

        for i in range(0, len(data) - T - 3, step):  # Adjust range to prevent overlap
            # Ensure we do not go out of bounds when accessing the next row
            if i + T + 3 < len(data):  # Prevent out-of-bounds error
                # Extract the current window of 24 rows (this would be input data)
                X = data.iloc[i:i+T][column_names].values
                print("i is ",i, "i+T is :", i+T)
                reshaped_data = np.zeros((T, 5, 9))
                for j in range(9):
                    start_col = j * 5
                    end_col = (j + 1) * 5
                    reshaped_data[:, :, j] = X[:, start_col:end_col]

                # Get the values of C_WS50M at i+T+1, i+T+2, and i+T+3 for filename
                value_1 = data.iloc[i+T]['C_WS50M']
                value_2 = data.iloc[i+T+1]['C_WS50M']
                value_3 = data.iloc[i+T+2]['C_WS50M']

                # Display next 3 values (for filename)
                print("All data is:", data.iloc[i:i+T+3]['C_WS50M'])
                print("Input data is:", data.iloc[i:i+T]['C_WS50M'])
                print("output data is:", data.iloc[i+T]['C_WS50M'],data.iloc[i+T+1]['C_WS50M'],data.iloc[i+T+2]['C_WS50M'])

                # Save as .npy in the respective train, valid, test folder
                npy_file_name = f"{city_folder}/{dataset_name}/{example_counter}_{value_1}_{value_2}_{value_3}.npy"
                np.save(npy_file_name, reshaped_data)


                # Increment the example counter for the next file
                example_counter += 1


    # Process the datasets: train, valid, and test
    save_npy_files(train_data, city, 'train', column_names)
    save_npy_files(valid_data, city, 'valid', column_names)
    save_npy_files(test_data, city, 'test', column_names)

# Process data for both cities
process_city_data("Esbjerg")
