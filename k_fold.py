import os
import random
import shutil

def divide_images_and_texts_into_groups(image_folder, output_folder, num_groups=5):
    # List all images in the folder
    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]  # Filtra solo le immagini
    
    # Sort and randomly shuffle the images
    images.sort()
    random.shuffle(images)
    
    # Determine the number of images per group
    num_images_per_group = len(images) // num_groups
    remainder = len(images) % num_groups

    # Create the groups of images
    groups = []
    index = 0
    for i in range(num_groups):
        group_size = num_images_per_group + (1 if i < remainder else 0)
        group = images[index:index + group_size]
        groups.append(group)
        index += group_size

    # Create the folder structure if it doesn't already exist
    for fold_index in range(num_groups):
        fold_folder = os.path.join(output_folder, f'fold_{fold_index + 1}')
        test_folder = os.path.join(fold_folder, 'test')
        training_folder = os.path.join(fold_folder, 'training')

        # Create the test and training folders with images and labels subfolders
        os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(training_folder, 'labels'), exist_ok=True)

        # Copy the current group to the test folder
        test_group = groups[fold_index]
        for image in test_group:
            image_name, image_ext = os.path.splitext(image)
            shutil.copy(os.path.join(image_folder, image), os.path.join(test_folder, 'images', image))
            
            # Copy the corresponding .txt file if it exists
            txt_file = f"{image_name}.txt"
            if os.path.exists(os.path.join(image_folder, txt_file)):
                shutil.copy(os.path.join(image_folder, txt_file), os.path.join(test_folder, 'labels', txt_file))

        # Copy the remaining groups to the training folder
        for i in range(num_groups):
            if i != fold_index:
                training_group = groups[i]
                for image in training_group:
                    image_name, image_ext = os.path.splitext(image)
                    shutil.copy(os.path.join(image_folder, image), os.path.join(training_folder, 'images', image))
                    
                    # Copy the corresponding .txt file if it exists
                    txt_file = f"{image_name}.txt"
                    if os.path.exists(os.path.join(image_folder, txt_file)):
                        shutil.copy(os.path.join(image_folder, txt_file), os.path.join(training_folder, 'labels', txt_file))

    print(f'Division completed. The images and corresponding .txt files have been divided into {num_groups} groups and organized into fold directories.')
    
# Input and output folder
image_folder = r'Dataset merged\FratelloCluster-BarbaraA'
output_folder = r'Output fold'

divide_images_and_texts_into_groups(image_folder, output_folder)