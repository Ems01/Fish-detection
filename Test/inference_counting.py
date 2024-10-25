from ultralytics import YOLO
import cv2
import os

# Function to extract fold number and model name from model path
def extract_info_from_model_path(model_path):
    # Extract fold number and model name from model path
    fold_part = os.path.basename(os.path.dirname(model_path))
    fold_number = int(fold_part.split(' ')[-1])  # Assuming format is 'fold X'
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Removes the '.pt' extension
    return fold_number, model_name

# Function to count bounding boxes from label file
def count_bounding_boxes_from_label(label_file_path):
    if not os.path.exists(label_file_path):
        print(f"Label file {label_file_path} not found.")
        return 0
    
    # Count lines in the label file (each line represents a bounding box)
    with open(label_file_path, 'r') as label_file:
        return len(label_file.readlines())

# Function to perform inference on a folder of images
def run_inference_on_folder(model_path, folder_path, labels_folder_path, confidence_threshold=0.25):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Extract fold number and model name from model path
    fold_number, model_name = extract_info_from_model_path(model_path)

    # Define directory to save the inference results
    save_dir = 'InferenceResults/Small300_best5'
    os.makedirs(save_dir, exist_ok=True)

    # Define file to save the results
    results_file_path = os.path.join(save_dir, 'results.txt')
    
    # Open file for writing results
    with open(results_file_path, 'w') as results_file:
        total_confidence_sum = 0.0
        total_boxes_count = 0
        total_actual_bboxes = 0
        image_counter = 1

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

                # Get the confidence values for each bounding box
                confidence_values = [box.conf.item() for box in results[0].boxes]

                # Sum of confidences and count of boxes for this image
                image_confidence_sum = sum(confidence_values)
                image_boxes_count = len(confidence_values)

                # Update total sums
                total_confidence_sum += image_confidence_sum
                total_boxes_count += image_boxes_count

                # Calculate the average confidence for this image
                image_avg_confidence = image_confidence_sum / image_boxes_count if image_boxes_count > 0 else 0

                # Get corresponding label file
                image_name, ext = os.path.splitext(image_file)
                label_file_path = os.path.join(labels_folder_path, f'{image_name}.txt')

                # Get actual bounding box count from the label file (ground truth)
                actual_bboxes_count = count_bounding_boxes_from_label(label_file_path)
                total_actual_bboxes += actual_bboxes_count

                # Write results for this image to the file
                results_file.write(f"Image {image_counter}: {image_file}\n")  # Write image name
                results_file.write(f"Box recognised: {image_boxes_count}\n")
                results_file.write(f"Ground truth Box: {actual_bboxes_count}\n")  # Write ground truth box count
                results_file.write(f"Average confidence: {image_avg_confidence:.2f}\n\n")

                image_counter += 1

            # Save the results with bounding boxes as images
            for idx, result in enumerate(results):
                # Plot bounding boxes on the image
                result_image = result.plot()

                # Define the save path using the required format
                save_path = os.path.join(
                    save_dir, 
                    f'{image_file}_{fold_number}_{model_name}_{confidence_threshold:.2f}.jpg'
                )
                cv2.imwrite(save_path, result_image)

        # Calculate overall average confidence across all images
        overall_avg_confidence = total_confidence_sum / total_boxes_count if total_boxes_count > 0 else 0

        # Write the total results at the end of the file
        results_file.write("Summary:\n")
        results_file.write(f"Total Box recognised: {total_boxes_count}\n")
        results_file.write(f"Total Ground truth Box: {total_actual_bboxes}\n")  # Total actual boxes
        results_file.write(f"Overall Average Confidence: {overall_avg_confidence:.2f}\n")

    print(f"Results saved to: {results_file_path}")

if __name__ == '__main__':
    # Path to the trained model (e.g., best.pt)
    model_path = '/Users/emsar/OneDrive/Desktop/materiale progetto cv/resultsv8Small300/epoche/fold 5/best.pt'

    # Path to the folder containing images
    folder_path = '/Users/emsar/OneDrive/Desktop/materiale progetto cv/Output fold/fold_5/test/images'

    # Path to the folder containing label files
    labels_folder_path = '/Users/emsar/OneDrive/Desktop/materiale progetto cv/Output fold/fold_5/test/labels'

    # Confidence threshold
    confidence_threshold = 0.5

    # Run inference on the folder
    run_inference_on_folder(model_path, folder_path, labels_folder_path, confidence_threshold)
