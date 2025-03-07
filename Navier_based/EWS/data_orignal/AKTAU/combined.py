import pandas as pd

# Define the file names in the desired order
file_names = [
    "C.csv", "N.csv", "S.csv", "E.csv", "W.csv", 
    "NE.csv", "NW.csv", "SE.csv", "SW.csv"
]

# Read all files into pandas DataFrames
dfs = {file_name: pd.read_csv(file_name) for file_name in file_names}

# Start with the first file (C.csv) and extract the columns for merging
merged_df = dfs["C.csv"][["YEAR", "MO", "DY", "HR", "WD50M", "WS50M", "PS", "T2M", "QV2M"]]
# Rename columns from C.csv to have "C_" prefix
merged_df = merged_df.rename(columns={col: f"C_{col}" for col in ["WD50M", "WS50M", "PS", "T2M", "QV2M"]})

# Iterating through all files to merge data on YEAR, MO, DY, HR columns
for file_name, df in dfs.items():
    if file_name != "C.csv":
        # Remove .csv extension from the file name to use as the prefix
        file_prefix = file_name.replace(".csv", "")
        
        # Merge on the columns YEAR, MO, DY, HR with an 'inner' join to ensure only matching rows are kept
        merged_df = pd.merge(merged_df, df[["YEAR", "MO", "DY", "HR", "WD50M", "WS50M", "PS", "T2M", "QV2M"]], 
                             on=["YEAR", "MO", "DY", "HR"], how="inner")
        # Rename columns to prefix with the file name (without .csv)
        for col in ["WD50M", "WS50M", "PS", "T2M", "QV2M"]:
            merged_df = merged_df.rename(columns={col: f"{file_prefix}_{col}"})

# Save the combined DataFrame to a new CSV file
merged_df.to_csv("combined.csv", index=False)

print("Combined CSV file created successfully with C.csv data first.")
