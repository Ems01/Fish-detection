import os

# Function to calculate the area of a bounding box from YOLO format
def calculate_area(bbox):
    # The bbox is a list: [class, x_center, y_center, width, height]
    width = bbox[3]
    height = bbox[4]
    return width * height


# Function to process a single file and calculate the areas of the bounding boxes
def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    areas = []
    for line in lines:
        # Converts each line into a list of floats (space-separated)
        bbox = list(map(float, line.strip().split()))
        
        # Ignores the bounding box if the class is 0, as specified
        if bbox[0] == 0:
            area = calculate_area(bbox)
            areas.append(area)
    
    return areas

# Main function to process all files in the folder
def calculate_average_area(folder_path):
    total_areas = []
    
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            areas = process_file(file_path)
            total_areas.extend(areas)
    
    # Calculate the average area, if there are any areas calculated
    if total_areas:
        average_area = sum(total_areas) / len(total_areas)
        return average_area
    else:
        return 0

# Specify the path of the folder containing the text files
folder_path = '/Users/emsar/OneDrive/Desktop/box/'  # Modify with the correct path

# Calculate and print the average area of the bounding boxes
average_area = calculate_average_area(folder_path)
print(f"The average area of the bounding boxes is: {average_area}")
