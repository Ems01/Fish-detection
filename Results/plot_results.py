import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the data
base_folder = 'Results/Data'
output_base_folder = 'Results/Plots'

# Font size settings for better readability
axis_label_font_size = 18
tick_label_font_size = 16
title_font_size = 18
legend_font_size = 14

# Ensure the output base folder exists
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# Iterate through all subfolders in the base folder (large100, small100, etc.)
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(subfolder_path):  # Only process directories
        print(f"Processing folder: {subfolder_path}")

        # Prepare the list of CSV files for this subfolder
        csv_files = [os.path.join(subfolder_path, f'results{i}.csv') for i in range(1, 6)]

        # Create a specific output folder for this subfolder's plots
        output_folder = os.path.join(output_base_folder, subfolder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # List to store the DataFrames
        dfs = []

        # Load all the CSV files for this subfolder
        for file in csv_files:
            print(f"Trying to load: {file}")
            if os.path.exists(file):
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
                df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
                dfs.append(df)
            else:
                print(f"File {file} does not exist. Skipping.")

        # Check if any DataFrames were loaded
        if dfs:
            # Concatenate the DataFrames and calculate the mean of numeric columns only
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

            # 1. Plot comparison of train and val (mean and per fold) for each loss type
            for train_field, val_field in loss_pairs:
                # Train and Val combined comparison plot for box_loss and cls_loss
                if train_field in ['train/box_loss', 'train/cls_loss']:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot all curves for each CSV file (train and val)
                    for idx, df in enumerate(dfs):
                        plt.plot(df[train_field], alpha=0.5, label=f'Fold {idx+1} Train')
                        plt.plot(df[val_field], alpha=0.5, linestyle='--', label=f'Fold {idx+1} Validation')

                    # Plot the mean curves for train and val
                    plt.plot(mean_df[train_field], label='Mean Train', color='blue', linewidth=2)
                    plt.plot(mean_df[val_field], label='Mean Validation', color='red', linestyle='--', linewidth=2)
                    
                    # Add title, legend, and labels
                    combined_title = f'{train_field.split("/")[1].title()} Comparison'
                    plt.title(combined_title, fontsize=title_font_size)
                    plt.legend(fontsize=legend_font_size)
                    plt.xlabel('Epochs', fontsize=axis_label_font_size)
                    plt.ylabel('Value', fontsize=axis_label_font_size)
                    plt.xticks(fontsize=tick_label_font_size)
                    plt.yticks(fontsize=tick_label_font_size)
                    
                    # Save the figure
                    filename = os.path.join(output_folder, f'combined_{train_field.split("/")[1].lower()}.png')
                    plt.savefig(filename)
                    plt.close()

                # Individual train and val plots as before (not modified)
                # Train plot: Fold and Mean comparison
                plt.figure(figsize=(10, 6))
                for idx, df in enumerate(dfs):
                    plt.plot(df[train_field], alpha=0.5, label=f'Fold {idx+1} Train')
                plt.plot(mean_df[train_field], label='Mean Train', color='black', linewidth=2, linestyle='--')
                plt.title(f'{train_field.split("/")[1].title()} Train', fontsize=title_font_size)
                plt.legend(fontsize=legend_font_size)
                plt.xlabel('Epochs', fontsize=axis_label_font_size)
                plt.ylabel('Value', fontsize=axis_label_font_size)
                plt.xticks(fontsize=tick_label_font_size)
                plt.yticks(fontsize=tick_label_font_size)
                filename = os.path.join(output_folder, f'train_{train_field.split("/")[1].lower()}.png')
                plt.savefig(filename)
                plt.close()

                # Val plot: Fold and Mean comparison
                plt.figure(figsize=(10, 6))
                for idx, df in enumerate(dfs):
                    plt.plot(df[val_field], alpha=0.5, label=f'Fold {idx+1} Validation')
                plt.plot(mean_df[val_field], label='Mean Validation', color='black', linewidth=2)
                plt.title(f'{val_field.split("/")[1].title()} Validation', fontsize=title_font_size)
                plt.legend(fontsize=legend_font_size)
                plt.xlabel('Epochs', fontsize=axis_label_font_size)
                plt.ylabel('Value', fontsize=axis_label_font_size)
                plt.xticks(fontsize=tick_label_font_size)
                plt.yticks(fontsize=tick_label_font_size)
                filename = os.path.join(output_folder, f'val_{val_field.split("/")[1].lower()}.png')
                plt.savefig(filename)
                plt.close()

            # 2. Plot single graphs for each metric with folds and mean
            for field, pretty_name in metrics_fields:
                plt.figure(figsize=(10, 6))  # Create a new figure
                for idx, df in enumerate(dfs):
                    plt.plot(df[field], alpha=0.5, label=f'Fold {idx+1}')
                plt.plot(mean_df[field], label='Mean', color='black', linewidth=2)
                plt.title(f'{subfolder}: {pretty_name}', fontsize=title_font_size)
                plt.legend(fontsize=legend_font_size)
                plt.xlabel('Epochs', fontsize=axis_label_font_size)
                plt.ylabel('Value', fontsize=axis_label_font_size)
                plt.xticks(fontsize=tick_label_font_size)
                plt.yticks(fontsize=tick_label_font_size)
                filename = os.path.join(output_folder, f'{field.split("/")[1].lower()}_metric.png')
                plt.savefig(filename)
                plt.close()

        else:
            print(f"No valid data found in folder: {subfolder_path}")
