from PIL import Image
import os


def convert_to_single_channel(image_path):
    # Open the image file
    img = Image.open(image_path)

    # If the image is not single channel, convert it
    if img.mode != 'L':
        # Convert to single channel image
        img = img.convert('L')

        # Save the converted image
        img.save(image_path)

        print(f"Converted {image_path} to single channel.")


def check_and_convert_images(folder_path):
    # Get all files in the specified folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file)

        # Check if the file is an image
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Call the conversion function
            convert_to_single_channel(file_path)


# Specify the folder path
folder_path = '../data/Refuge/od/train/labels/'

# Call the function to check and convert images
check_and_convert_images(folder_path)
