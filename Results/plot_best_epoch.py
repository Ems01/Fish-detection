import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the CSV files
csv_folder = 'Results'
csv_files = [os.path.join(csv_folder, f'results{i}.csv') for i in range(1, 6)]

# Create a folder for the plots if it doesn't exist
output_folder = os.path.join(csv_folder, 'Plots')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List to store the DataFrames
dfs = []

# Load all CSV files, removing extra spaces from column names
for file in csv_files:
    print(f"Trying to load: {file}")  # Print the file path for debugging
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove spaces at the beginning and end of column names
    dfs.append(df)

# Calculate the optimal epoch for each fold
for idx, df in enumerate(dfs):
    # Extract necessary metrics
    mAP50 = df['metrics/mAP50(B)']
    val_box_loss = df['val/box_loss']
    val_cls_loss = df['val/cls_loss']
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']

    # Coefficients for calculating the combined score
    coeff_box_loss = 2.0  # Higher weight for box losses
    coeff_cls_loss = 2.0  # Higher weight for class losses
    coeff_precision = 1.5  # Weight for precision
    coeff_recall = 1.5  # Weight for recall

    # Calculate a combined score
    combined_score = (mAP50 * 1.0 +  # Use mAP50 with a weight of 1
                      precision * coeff_precision + 
                      recall * coeff_recall - 
                      (val_box_loss * coeff_box_loss) - 
                      (val_cls_loss * coeff_cls_loss))

    # Find the epoch with the highest score
    best_epoch = combined_score.idxmax()
    best_score = combined_score.max()

    print(f"Fold {idx + 1}: Best epoch = {best_epoch + 1}, Combined Score = {best_score:.4f}")

    # Plotting for each fold
    plt.figure(figsize=(12, 6))

    # Plot the metrics
    plt.plot(mAP50, label='mAP50', color='blue', linestyle='--')
    plt.plot(val_box_loss, label='Validation Box Loss', color='green', linestyle='-')
    plt.plot(val_cls_loss, label='Validation Class Loss', color='orange', linestyle='-')
    plt.plot(precision, label='Precision', color='purple', linestyle='-.')
    plt.plot(recall, label='Recall', color='red', linestyle=':')

    # Add a dashed vertical line for the best epoch
    plt.axvline(x=best_epoch, color='red', linestyle=':', label='Best Epoch')

    # Add title, legend, and labels
    plt.title(f'Fold {idx + 1}: mAP50, Validation Box Loss, Class Loss, Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    # Update the legend to include the best epoch
    plt.legend(loc='upper right')
    plt.legend(title=f'Best Epoch: {best_epoch + 1}', title_fontsize='13')

    # Save the plot
    plt.savefig(os.path.join(output_folder, f'fold_{idx + 1}_combined_score.png'))
    plt.close()

print("All plots saved in the 'Results/Plots' folder.")
