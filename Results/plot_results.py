import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the CSV files
csv_folder = 'Results'
csv_files = [os.path.join(csv_folder, f'results{i}.csv') for i in range(1, 6)]

# Create a folder for plots if it doesn't exist
output_folder = os.path.join(csv_folder, 'Plots')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List to store the DataFrames
dfs = []

# Load all the CSV files, stripping extra spaces from the column names
for file in csv_files:
    print(f"Trying to load: {file}")  # Print file path for debugging
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
    dfs.append(df)

# Calculate the mean of the data across all files
mean_df = pd.concat(dfs).groupby(level=0).mean()

# Pairs of fields to plot (train and val of the same type)
loss_pairs = [('train/box_loss', 'val/box_loss'), 
              ('train/cls_loss', 'val/cls_loss'), 
              ('train/dfl_loss', 'val/dfl_loss')]

# Metrics fields to plot separately (precision, recall, etc.)
metrics_fields = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'Mean Average Precision at IoU 50'),
    ('metrics/mAP50-95(B)', 'Mean Average Precision at IoU 50-95')
]

# 1. Plot comparison of train and val (only mean) for each loss type
for train_field, val_field in loss_pairs:
    plt.figure(figsize=(10, 6))  # Create a new figure
    
    # Plot the mean curves for train and val
    plt.plot(mean_df[train_field], label='Mean Train', color='blue', linewidth=2, linestyle='--')
    plt.plot(mean_df[val_field], label='Mean Validation', color='green', linewidth=2, linestyle='-')
    
    # Create pretty titles
    train_title = f'{train_field.split("/")[1].title()} Train'
    val_title = f'{val_field.split("/")[1].title()} Validation'
    
    # Add title, legend, and labels
    plt.title(f'Comparison: {train_title} vs {val_title}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    # Save the figure
    filename = f'{output_folder}/mean_comparison_{train_field.split("/")[1].lower()}_vs_{val_field.split("/")[1].lower()}.png'
    plt.savefig(filename)
    
    # Close the figure to save memory
    plt.close()

# 2. Plot separate graphs with folds and mean for train and val separately
for train_field, val_field in loss_pairs:
    # Train plot
    plt.figure(figsize=(10, 6))
    
    # Plot all curves for each CSV file (train)
    for idx, df in enumerate(dfs):
        plt.plot(df[train_field], alpha=0.5, label=f'Fold {idx+1} Train')
    
    # Plot the mean curve for train
    plt.plot(mean_df[train_field], label='Mean Train', color='black', linewidth=2, linestyle='--')
    
    # Create pretty title
    train_title = f'{train_field.split("/")[1].title()} Train'
    
    # Add title, legend, and labels
    plt.title(train_title)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    # Save the figure
    filename = f'{output_folder}/train_{train_field.split("/")[1].lower()}.png'
    plt.savefig(filename)
    plt.close()
    
    # Val plot
    plt.figure(figsize=(10, 6))
    
    # Plot all curves for each CSV file (val)
    for idx, df in enumerate(dfs):
        plt.plot(df[val_field], alpha=0.5, label=f'Fold {idx+1} Validation')
    
    # Plot the mean curve for val
    plt.plot(mean_df[val_field], label='Mean Validation', color='black', linewidth=2, linestyle='-')
    
    # Create pretty title
    val_title = f'{val_field.split("/")[1].title()} Validation'
    
    # Add title, legend, and labels
    plt.title(val_title)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    # Save the figure
    filename = f'{output_folder}/val_{val_field.split("/")[1].lower()}.png'
    plt.savefig(filename)
    plt.close()

# 3. Plot single graphs for each metric
for field, pretty_name in metrics_fields:
    plt.figure(figsize=(10, 6))  # Create a new figure
    
    # Plot all curves for each CSV file (metric)
    for idx, df in enumerate(dfs):
        plt.plot(df[field], alpha=0.5, label=f'Fold {idx+1}')
    
    # Plot the mean curve for the metric
    plt.plot(mean_df[field], label='Mean', color='black', linewidth=2)
    
    # Add title, legend, and labels
    plt.title(pretty_name)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    # Save the figure
    filename = f'{output_folder}/{pretty_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename)
    plt.close()

print("All plots saved in the 'Results/Plots' folder.")