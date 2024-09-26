from ultralytics import YOLO
import os
import torch  # To check if GPU is available

# Automatically get the current directory and append the dataset folder
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "Fish Detection CV 2024.v1i.yolov8")

# Check if GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Funzione per creare il file di configurazione YAML per ogni fold
def create_data_yaml(fold_num):
    yaml_content = f"""
    train: {os.path.join(dataset_path, f'fold {fold_num}/training/images')}
    val: {os.path.join(dataset_path, f'fold {fold_num}/test/images')}
    nc: 1  # Numero di classi, cambia se hai più classi
    names: ['fish']  # Nome della classe, cambia con il nome corretto
    """
    yaml_path = os.path.join(dataset_path, f'fold_{fold_num}_data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

# Funzione per eseguire la cross-validation
def run_cross_validation(num_folds=5, epochs=50):
    for fold in range(1, num_folds + 1):
        print(f"\n--- Fold {fold} ---")
        
        # Crea il file YAML per il fold corrente
        data_yaml = create_data_yaml(fold)
        
        # Carica il modello YOLOv8
        model = YOLO('yolov8l.pt')
        
        # Train the model using your dataset
        model.train(
            data=data_yaml,                    # Use the generated YAML file
            epochs=epochs,                     # Number of epochs to train
            imgsz=640,                         # Image size
            batch=16,                          # Batch size for training
            device=device,                     # Automatically select GPU if available, otherwise use CPU
            plots=True,                        # Save training plots
            save_dir=os.path.join(dataset_path, "results", f"fold_{fold}"),  # Save results for each fold
        )

        # Validate the model after training
        val_results = model.val(
            device=device,                     # Run validation on GPU if available
            batch=16,                          # Batch size for validation
            workers=2,                         # Number of workers for validation
            save_dir=os.path.join(dataset_path, "results", f"fold_{fold}"),  # Save validation results for each fold
        )
        print(f"Validation results for fold {fold}: {val_results}")

        # Inference with the trained model (optional)
        test_images_path = os.path.join(dataset_path, f'fold {fold}/test/images')
        test_results = model.predict(source=test_images_path, save=True, save_dir=os.path.join(dataset_path, "results", f"fold_{fold}/inference"))
        print(f"Test results for fold {fold}: {test_results}")

# Esegui la cross-validation
run_cross_validation(num_folds=5, epochs=100)
