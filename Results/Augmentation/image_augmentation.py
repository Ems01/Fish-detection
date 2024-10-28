import cv2
import numpy as np

def augment_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("The image cannot be loaded. Check the image path.")
    
    # Augmentation 1: saturation
    def adjust_saturation(image, scale=0.2):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = 1 + scale
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Augmentation 2: brightness
    def adjust_brightness(image, scale=0.4):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = 1 + scale
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Augmentation 3: zoom (center crop and resize back to original)
    def zoom_image(image, zoom_factor=0.5):
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        # Crop the center of the image
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = image[top:top + new_h, left:left + new_w]
        # Resize back to original size
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Augmentation 4: horizontal flip
    def flip_image(image):
        return cv2.flip(image, 1)  # 1 indicates horizontal flip

    # Apply augmentations sequentially
    augmented_image = adjust_saturation(image)
    augmented_image = adjust_brightness(augmented_image)
    augmented_image = zoom_image(augmented_image)
    augmented_image = flip_image(augmented_image)

    # Save the final augmented image
    output_path = "./augmented_combined_image.jpg"
    cv2.imwrite(output_path, augmented_image)
    print(f"Combined augmentation image saved as {output_path}")

# Usage example
augment_image("Results/Augmentation/img.jpg")
