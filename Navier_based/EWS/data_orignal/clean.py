import os
import pandas as pd

def clean_csv(file_path):
    """
    Function to clean the CSV file by finding the line with the values matching
    'YEAR MO DY HR WD50M WS50M PS T2M QV2M' and making it the header.
    Deletes all rows before that.
    """
    # Read the CSV file into a DataFrame
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # The line that should be the header
    header_values = ['YEAR', 'MO', 'DY', 'HR', 'WD50M', 'WS50M', 'PS', 'T2M', 'QV2M']
    
    # Find the line where the values match the header
    header_index = None
    for i, line in enumerate(lines):
        # Strip whitespace and split line into values (assuming comma-separated values)
        line_values = line.strip().split(',')  # Using comma delimiter
        
        # Log each line for troubleshooting
        print(f"Checking line {i}: {line_values}")
        
        # Compare values case-insensitively and ignore extra spaces
        if [value.strip() for value in line_values] == header_values:
            header_index = i
            break
    
    if header_index is not None:
        # Read the CSV starting from the header line
        cleaned_data = pd.read_csv(file_path, skiprows=header_index, delimiter=',')  # Use comma as delimiter
        
        # Save the cleaned CSV back with the original name
        cleaned_data.to_csv(file_path, index=False)
        print(f"Cleaned and saved: {file_path}")
    else:
        print(f"Header not found in {file_path}.")

def process_directory(directory_path):
    """
    Function to process all CSV files in a directory and its subdirectories.
    """
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                clean_csv(file_path)

# Example usage: Process the current directory
directory_path = './'
process_directory(directory_path)
