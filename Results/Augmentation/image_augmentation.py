import cv2
import numpy as np

def augment_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("The image cannot be loaded. Check the image path.")
    
    augmented_images = []

    # Augmentation 1: Fixed saturation adjustment of 40%
    def adjust_saturation(image, scale=0.4):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = 1 + scale
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Augmentation 2: Fixed brightness adjustment of 40%
    def adjust_brightness(image, scale=0.4):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = 1 + scale
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Augmentation 3: Fixed scaling to 50%
    def scale_image(image, scale=0.5):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Augmentation 4: Fixed horizontal flip
    def flip_image(image):
        return cv2.flip(image, 1)  # 1 indicates horizontal flip

    # Apply each augmentation
    augmented_images.append(adjust_saturation(image))
    augmented_images.append(adjust_brightness(image))
    augmented_images.append(scale_image(image))
    augmented_images.append(flip_image(image))

    # Save the images
    for i, aug_img in enumerate(augmented_images):
        cv2.imwrite(f"augmented_image_{i+1}.jpg", aug_img)
        print(f"Image augmented_image_{i+1}.jpg saved successfully.")

# Usage example
augment_image("/Users/emsar/OneDrive/Desktop/prova.png")
