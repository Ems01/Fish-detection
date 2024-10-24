import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the confusion matrix values for each fold
# Format: [TP, FP, FN, TN]
confusion_matrices = [
    [8630, 7645, 3809, 0],  # Fold 1
    [9227, 7212, 4788, 0],  # Fold 2
    [10666, 7281, 6010, 0],  # Fold 3
    [9951, 7058, 4653, 0],  # Fold 4
    [8141, 10149, 3796, 0],  # Fold 5
]

# Convert to a NumPy array for easier manipulation
confusion_matrices = np.array(confusion_matrices)

# Calculate the sum and average
sum_matrix = np.sum(confusion_matrices, axis=0)
average_matrix = np.mean(confusion_matrices, axis=0)

def save_confusion_matrix(matrix, title, filename, is_average=False):
    """Save the confusion matrix as a PNG file."""
    sns.set()
    labels = ['fish', 'background']
    confusion_matrix_reshaped = np.array([[matrix[0], matrix[1]], [matrix[2], matrix[3]]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_reshaped, annot=True, fmt='.2f' if is_average else 'd', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(title)

    # Create Results directory if it doesn't exist
    os.makedirs('Results', exist_ok=True)

    # Save the figure
    plt.savefig(f'Results/{filename}')
    plt.close()  # Close the plot to free up memory

# Save the sum confusion matrix
save_confusion_matrix(sum_matrix, 'Sum of Confusion Matrices', 'sum_confusion_matrix.png')

# Save the average confusion matrix
save_confusion_matrix(average_matrix, 'Average of Confusion Matrices', 'average_confusion_matrix.png', is_average=True)
