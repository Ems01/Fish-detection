import os
import numpy as np  # To calculate the average performance
import torch
from codecarbon import EmissionsTracker  # Import the CO2 tracker
from ultralytics import YOLO

# Automatically get the current directory and append the dataset folder
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "dataset")

# Check if GPU is available, otherwise raise an exception
if not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled GPU not found. The script requires a GPU with CUDA support to run.")

# Set the device to GPU
device = 'cuda'

# Log file for emissions tracking
emissions_log_file = os.path.join(dataset_path, "co2_emissions_log.txt")

# Function to log emissions data
def log_emissions(emissions, total=False):
    with open(emissions_log_file, 'a') as f:
        if total:
            f.write(f"Total CO2 emissions for all folds: {emissions:.4f} kg\n")

# Function to perform cross-validation
def run_cross_validation(num_folds=5, epochs=200, checkpoint_interval=10, resume_epoch=None):
    fold_performances = []
    total_emissions = 0.0

    # Clear the emissions log file at the start
    with open(emissions_log_file, 'w') as f:
        f.write("CO2 Emissions Log\n==================\n")

    for fold in range(1, num_folds + 1):
        print(f"\n--- Fold {fold} ---")

        # Paths for the current fold
        train_images = os.path.join(dataset_path, f'fold_{fold}', 'training', 'images')
        test_images = os.path.join(dataset_path, f'fold_{fold}', 'test', 'images')

        # Create a temporary YAML file for each fold
        data_yaml_content = f"""
        train: {train_images}
        val: {test_images}
        nc: 1
        names: ['fish']
        """

        data_yaml_path = os.path.join(dataset_path, f'fold_{fold}_data.yaml')
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml_content)

        # Load the YOLOv8 model
        model = YOLO('yolov8s.pt')

        # Set the save directory for training and validation separately for clarity
        train_save_dir = os.path.join(current_dir, "runs", "detect", f"train_fold_{fold}")
        val_save_dir = os.path.join(current_dir, "runs", "detect", f"val_fold_{fold}")
        
        # Ensure directories exist
        os.makedirs(train_save_dir, exist_ok=True)
        os.makedirs(val_save_dir, exist_ok=True)

        # If resuming from a checkpoint, load the last checkpoint
        if resume_epoch is not None and resume_epoch > 0:
            checkpoint_path = os.path.join(train_save_dir, f'weights/epoch_{resume_epoch}.pt')
            if os.path.exists(checkpoint_path):
                print(f"Resuming training for fold {fold} from epoch {resume_epoch}")
                model = YOLO(checkpoint_path)  # Load model from checkpoint
            else:
                print(f"Checkpoint for epoch {resume_epoch} not found. Starting from scratch for fold {fold}.")
                resume_epoch = None  # Reset resume_epoch if checkpoint is not found

        # Start the emissions tracker
        tracker = EmissionsTracker(log_level="WARNING")
        try:
            tracker.start()
        except Exception as e:
            print(f"Error starting emissions tracker: {e}")

        # Training phase with augmentation
        model.train(
            data=data_yaml_path,  # Use the generated YAML file
            epochs=epochs,  # Total number of epochs
            imgsz=640,  # Image size
            batch=16,  # Batch size for training
            device=device,  # Use GPU
            workers=0,  # Number of workers for validation
            plots=True,  # Save training plots
            save_dir=train_save_dir,  # Save training results in train_fold_X
            save_period=checkpoint_interval,  # Save checkpoint every 'checkpoint_interval' epochs
            resume=resume_epoch,  # Resume from the specified epoch if provided
            augment=True,  # Enable default augmentations
            hsv_s=0.4,    # Saturation variation
            hsv_v=0.4,    # Value (brightness) variation
            scale=0.5,  # Scale image up/down by 50%
            fliplr=0.2,  # Horizontal flip
        )

        # Validation phase
        val_results = model.val(
            data=data_yaml_path,  # Use the same data.yaml
            device=device,  # Run validation on GPU
            batch=16,  # Batch size for validation
            workers=0,  # Number of workers for validation
            save_dir=val_save_dir,  # Save validation results in val_fold_X
        )
        print(f"Validation results for fold {fold}: {val_results}")

        # Append validation results to calculate average later
        fold_performances.append(val_results.results_dict)

        try:
            # Stop the emissions tracker and log the emissions produced
            emissions = tracker.stop()
            total_emissions += emissions
            print(f"CO2 emitted for fold {fold}: {emissions:.4f} kg")

        except Exception as e:
            print(f"Error stopping emissions tracker or saving data: {e}")

    # Calculate the average performance across all folds
    avg_map50 = np.mean([fold['metrics/mAP50(B)'] for fold in fold_performances])
    print(f"Average mAP50 across all folds: {avg_map50}")

    # Log total CO2 emissions to the file
    log_emissions(emissions=total_emissions, total=True)

    # Print the total CO2 emissions
    print(f"Total CO2 emissions for {num_folds} folds: {total_emissions:.4f} kg")

if __name__ == '__main__':
    # Start cross validation
    run_cross_validation(num_folds=5, epochs=300, checkpoint_interval=1)
