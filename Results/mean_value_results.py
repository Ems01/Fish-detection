import pandas as pd

# Function to calculate F1-measure
def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Function to extract the best values from a CSV file
def get_best_metrics(file_path):
    data = pd.read_csv(file_path)
    
    # Find the maximum value for precision, recall, mAP50 and calculate F1-measure
    best_precision = data['metrics/precision(B)'].max()
    best_recall = data['metrics/recall(B)'].max()
    best_mAP50 = data['metrics/mAP50(B)'].max()
    best_f1 = calculate_f1(best_precision, best_recall)
    
    return {
        'precision': best_precision,
        'recall': best_recall,
        'F1': best_f1,
        'mAP50': best_mAP50
    }

# List of CSV files
file_paths = ["file_path_1", "file_path_2", "file_path_3", "file_path_4", "file_path_5"]

# List to store the best values from each file
all_best_metrics = []

# Extract the best values for each file
for file_path in file_paths:
    best_metrics = get_best_metrics(file_path)
    all_best_metrics.append(best_metrics)

# Calculate the average of the best values
mean_metrics = {
    'precision': sum(d['precision'] for d in all_best_metrics) / len(all_best_metrics),
    'recall': sum(d['recall'] for d in all_best_metrics) / len(all_best_metrics),
    'F1': sum(d['F1'] for d in all_best_metrics) / len(all_best_metrics),
    'mAP50': sum(d['mAP50'] for d in all_best_metrics) / len(all_best_metrics)
}

# Print the absolute results for each file
print("Absolute values for each file:")
for i, metrics in enumerate(all_best_metrics, 1):
    print(f"File {i} - Precision: {metrics['precision']:.5f}, Recall: {metrics['recall']:.5f}, F1: {metrics['F1']:.5f}, mAP50: {metrics['mAP50']:.5f}")

# Print the average values
print("\nAverage of the best values for the 5 files:")
print(f"Precision: {mean_metrics['precision']:.5f}")
print(f"Recall: {mean_metrics['recall']:.5f}")
print(f"F1: {mean_metrics['F1']:.5f}")
print(f"mAP50: {mean_metrics['mAP50']:.5f}")
