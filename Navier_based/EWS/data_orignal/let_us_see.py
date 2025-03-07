import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot wind speed with updated logic for green and red dots and count occurrences
def plot_wind_speed(csv_folder, P, K):
    # Walk through the directory and read all CSV files
    for root, dirs, files in os.walk(csv_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Ensure the necessary columns are present
                if 'WS50M' not in df.columns:
                    print(f"Column 'WS50M' not found in {file}")
                    continue
                
                # Initialize counter for instances where wind speed difference exceeds P
                trigger_count = 0

                # Iterate through the rows to find consecutive differences greater than P
                for i in range(1, len(df)):
                    if abs(df['WS50M'][i] - df['WS50M'][i - 1]) > P:
                        trigger_count += 1  # Count the instance
                        
                        # Determine the range of indices to plot (previous K and next K)
                        start_idx = max(i - K, 0)
                        end_idx = min(i + K + 1, len(df))

                        # Prepare data for plotting
                        data_to_plot = df[start_idx:end_idx]
                        wind_speeds = data_to_plot['WS50M']

                        # Create the plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(data_to_plot.index, wind_speeds, label="Wind Speed", marker='o')

                        # Add red dot for the point that exceeds the P threshold
                        plt.plot(df.index[i], df['WS50M'][i], 'ro', label='Trigger Point')

                        # Add green dots for points within the window that also exceed the P threshold
                        for j in range(start_idx, end_idx - 1):
                            if abs(df['WS50M'][j + 1] - df['WS50M'][j]) > P:
                                plt.plot(df.index[j + 1], df['WS50M'][j + 1], 'go')

                        # Title and labels
                        plt.title(f"Wind Speed for {file} (Window around {i})")
                        plt.xlabel('Index')
                        plt.ylabel('Wind Speed (m/s)')
                        plt.legend()
                        plt.grid(True)
                        # plt.show()

                # Print summary for the current file
                print(f"Summary for {file}: {trigger_count} instances where the difference in wind speed exceeds {P} m/s.")
                
# Example usage
plot_wind_speed('./', P=2, K=100)
