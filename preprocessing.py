import json
import os
import time

# Function to replace all labels with 'fish'
def replace_all_labels_with_fish(json_file, new_label):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Replace all labels with new_label
    for shape in data.get('shapes', []):
        shape['label'] = new_label
    
    # Save the modified JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Function to convert points to bounding boxes of size 20x20
def point_to_bbox(point, img_width, img_height, box_width=20, box_height=20):
    x_center = point[0]
    y_center = point[1]
    half_width = box_width / 2.0
    half_height = box_height / 2.0
    
    x_min = max(0, x_center - half_width)
    y_min = max(0, y_center - half_height)
    x_max = min(img_width, x_center + half_width)
    y_max = min(img_height, y_center + half_height)
    
    return {
        "x_min": x_min / img_width,
        "y_min": y_min / img_height,
        "x_max": x_max / img_width,
        "y_max": y_max / img_height
    }

# Function to replace points with bounding boxes in the JSON
def replace_points_with_bboxes(json_file, box_width=20, box_height=20):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    new_shapes = []
    
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'point':
            # Convert the point to a bounding box of size 20x20
            bbox = point_to_bbox(shape['points'][0], data['imageWidth'], data['imageHeight'], box_width, box_height)
            new_shapes.append({
                "label": shape['label'],
                "shape_type": "rectangle",
                "points": [
                    [bbox['x_min'], bbox['y_min']],
                    [bbox['x_max'], bbox['y_max']]
                ]
            })
        else:
            # Keep existing rectangles
            new_shapes.append(shape)
    
    # Update the shapes in the JSON
    data['shapes'] = new_shapes
    
    # Save the modified JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Automatically get the directory where this script is located
json_dir = os.path.dirname(os.path.realpath(__file__))

# New label to replace all others
new_label = 'fish'

# Track the start time
start_time = time.time()

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(json_dir):
    for i, filename in enumerate(files, 1):
        if filename.endswith('.json'):
            json_file = os.path.join(root, filename)
            
            try:
                # Print the current directory and file being processed
                print(f"Processing file: {filename} in directory: {root}")
                
                # First, replace all labels with 'fish'
                replace_all_labels_with_fish(json_file, new_label)
            
                # Then, replace points with 20x20 bounding boxes
                replace_points_with_bboxes(json_file, box_width=20, box_height=20)
            
                # Print the time elapsed for each file
                current_time = time.time()
                elapsed_time = current_time - start_time
                print(f"Processed {i} files. Time elapsed: {elapsed_time:.2f} seconds")
            
            except Exception as e:
                # Handle any errors that occur and continue with the next file
                print(f"Error processing file {filename} in directory {root}: {e}")

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time

# Print the total execution time
print(f"Total execution time: {total_time:.2f} seconds")
