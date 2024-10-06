from ultralytics import YOLO
import cv2
import os

# Function to perform inference on an image
def run_inference(model_path, image_path, confidence_threshold=0.25):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Perform inference with the confidence threshold
    results = model.predict(source=image_path, conf=confidence_threshold)

    # Define directory to save the inference results
    save_dir = 'InferenceResults'
    os.makedirs(save_dir, exist_ok=True)

    # Save the results with bounding boxes as images
    for idx, result in enumerate(results):
        # Save the image with bounding boxes plotted
        result_image = result.plot()
        
        # Define the save path for each result image
        save_path = os.path.join(save_dir, f'result_{idx}.jpg')
        cv2.imwrite(save_path, result_image)
        print(f"Result saved to: {save_path}")

if __name__ == '__main__':
    # Path to the trained model (e.g., last.pt)
    model_path = 'path/to/your/model/last.pt'

    # Path to the image for inference
    image_path = 'path/to/your/image.jpg'

    # Confidence threshold
    confidence_threshold = 0.5

    # Run inference
    run_inference(model_path, image_path, confidence_threshold)
