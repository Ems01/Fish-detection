import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a folder called 'Results' if it doesn't exist
output_folder = 'Results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of CSV file names
csv_files = ['results1.csv', 'results2.csv', 'results3.csv', 'results4.csv', 'results5.csv']

# List to store the DataFrames
dfs = []

# Load all the CSV files, stripping extra spaces from the column names
for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
    dfs.append(df)

# Calculate the mean of the data across all files
mean_df = pd.concat(dfs).groupby(level=0).mean()

# Fields to plot
fields = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
          'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
          'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']

# Create separate files for each field
for i, field in enumerate(fields):
    plt.figure(figsize=(10, 6))  # Create a new figure with a higher resolution
    
    # Plot all curves for each CSV file with a label for the legend
    for idx, df in enumerate(dfs):
        plt.plot(df[field], alpha=0.5, label=f'Fold {idx+1}')
    
    # Plot the mean curve
    plt.plot(mean_df[field], label='Mean', color='black', linewidth=2)
    
    # Add title, legend, and labels
    plt.title(field)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    # Save the figure as a PNG file in the 'Results' folder
    filename = f'{output_folder}/{field.replace("/", "_")}.png'
    plt.savefig(filename)

    # Close the figure to save memory
    plt.close()

print("Plots saved in the 'Results' folder.")
