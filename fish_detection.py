import torch
from ultralytics import YOLO
import os

# Automatically get the current directory and append the dataset folder
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "Fish Detection CV 2024.v1i.yolov8")

# Load the pre-trained YOLOv8 model (you can change 'yolov8l' to any other model variant like 'yolov8s', 'yolov8m', etc.)
model = YOLO('yolov8l.pt')

# Check if GPU is available and set to 'cpu' for your local notebook since you're not using a GPU
device = 'cpu'

# Train the model using your dataset
# Adjust the 'save_dir' to a custom path where results (e.g., confusion matrix) will be saved
model.train(
    data=f"{dataset_path}/data.yaml",  # Path to your dataset YAML file
    epochs=50,                         # Number of epochs to train
    imgsz=800,                         # Image size
    batch=16,                          # Batch size for training
    device=device,                     # Selected devide
    workers=2,                         # Number of data loading workers
    cache=False,                       # Don't cache data to reduce memory usage on your notebook
    plots=True,                        # Save training plots
    save_dir=f"{dataset_path}/results", # Custom directory to save results
)

# Validate the model after training
results = model.val(
    device=device,                     # Run validation on 'cpu'
    batch=16,                          # Batch size for validation
    workers=2,                         # Number of workers for validation
    save_dir=f"{dataset_path}/results", # Save validation results in the same custom directory
)

# Inference with the trained model
# prediction = model.predict(source='path/to/image.jpg', save=True, save_dir="f"{dataset_path}/results/inference")
