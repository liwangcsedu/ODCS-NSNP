import os
from PIL import Image


def process_image(input_path, output_path, target_color, replace_color):
    # Open the image
    img = Image.open(input_path)

    # Convert the image to RGB mode
    img = img.convert("RGB")

    # Get the width and height of the image
    width, height = img.size

    # Loop through each pixel
    for x in range(width):
        for y in range(height):
            # Get the color of the pixel
            r, g, b = img.getpixel((x, y))

            # Check if the pixel color is gray
            if r == g == b and 0 < r < 255:
                # Set the pixel color to target_color (white or black)
                img.putpixel((x, y), target_color)

    # Save the modified image
    img.save(output_path)


def process_images_in_folder(input_folder, output_folder_white, output_folder_black):
    # Create output folders if they don't exist
    os.makedirs(output_folder_white, exist_ok=True)
    os.makedirs(output_folder_black, exist_ok=True)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(".bmp") or filename.endswith(".png"):
            # Input image path
            input_image_path = os.path.join(input_folder, filename)

            # Output path for image with gray areas replaced with white
            output_path_white = os.path.join(output_folder_white, filename)

            # Output path for image with gray areas replaced with black
            output_path_black = os.path.join(output_folder_black, filename)

            # Process the image with gray areas replaced with white
            process_image(input_image_path, output_path_white, (255, 255, 255, 255), (255, 255, 255, 255))

            # Process the image with gray areas replaced with black
            #process_image(input_image_path, output_path_black, (0, 0, 0, 255), (255, 255, 255, 255))


# Input folder containing images
input_folder = "../data/crop/test-gt2/"

# Output folder for images with gray areas replaced with white
output_folder_white = "../data/crop/test-gt2-oc/"

# Output folder for images with gray areas replaced with black
output_folder_black = "../data/crop/test-gt2-od/"

# Process images in the input folder
process_images_in_folder(input_folder, output_folder_white, output_folder_black)

print("Images processed successfully!")
