from ultralytics import YOLO
import cv2
import os

# Function to extract fold number and model name from model path
def extract_info_from_model_path(model_path):
    # Example assumption: model path contains 'fold X' and the model file name contains the model name
    # e.g., "/Users/fede0/OneDrive/Desktop/trasferimento/fold 1/epoch225.pt"
    
    # Extract fold number from the directory name
    fold_part = os.path.basename(os.path.dirname(model_path))
    fold_number = int(fold_part.split(' ')[-1])  # Assuming format is 'fold X'
    
    # Extract model name from the file name
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Removes the '.pt' extension
    
    return fold_number, model_name

# Function to perform inference on a folder of images
def run_inference_on_folder(model_path, folder_path, confidence_threshold=0.25):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Extract fold number and model name from model path
    fold_number, model_name = extract_info_from_model_path(model_path)

    # Define directory to save the inference results
    save_dir = 'InferenceResults'
    os.makedirs(save_dir, exist_ok=True)

    # Loop through all images in the folder
    for image_file in os.listdir(folder_path):
        # Only process files with common image extensions
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Image at {image_path} not found or unable to open.")
                continue

            # Perform inference with the confidence threshold
            results = model.predict(source=image_path, conf=confidence_threshold)

            # Save the results with bounding boxes as images
            for idx, result in enumerate(results):
                # Plot bounding boxes on the image
                result_image = result.plot()

                # Define the save path using the required format
                image_name, ext = os.path.splitext(image_file)
                save_path = os.path.join(
                    save_dir, 
                    f'{image_name}_{fold_number}_{model_name}_{confidence_threshold:.2f}.jpg'
                )
                cv2.imwrite(save_path, result_image)
                print(f"Result saved to: {save_path}")

if __name__ == '__main__':
    # Path to the trained model (e.g., best.pt)
    model_path = '/Users/fede0/OneDrive/Desktop/trasferimento/fold 1/best.pt'

    # Path to the folder containing images
    folder_path = '/Users/fede0/OneDrive/Desktop/video_nuovo/video_nuovo/'

    # Confidence threshold
    confidence_threshold = 0.5

    # Run inference on the folder
    run_inference_on_folder(model_path, folder_path, confidence_threshold)
