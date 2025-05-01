import os
import cv2
import numpy as np

def post_process_images(img_path, save_path):
    """
    Perform post-processing on images in the input folder and save the processed images.
    Args:
        img_path: Input image folder path.
        save_path: Path to save processed images.
    """
    # Get the list of image files
    imagelist = os.listdir(img_path)
    # Iterate through the image file list
    for img_name in imagelist:
        # Check if the file ends with ".bmp"
        if img_name.lower().endswith(".png") or img_name.lower().endswith(".bmp") or img_name.lower().endswith(".jpg"):
            # Read grayscale image
            bf_img = cv2.imread(os.path.join(img_path, img_name), 0)
            # Define structuring element for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (7, 7))
            # Perform morphological closing operation to fill holes and connect broken edges
            img = cv2.morphologyEx(bf_img, cv2.MORPH_CLOSE, kernel)
            # Save the processed image
            cv2.imwrite(os.path.join(save_path, img_name), img)
        else:
            continue

# Usage example
# img_path = '../data/REFUGE/test/test-seg/'  # Image folder path
# save_path = '../data/REFUGE/test/test-seg-post/'  # Path to save processed images
# post_process_images(img_path, save_path)
