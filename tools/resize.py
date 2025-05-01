import shutil
import numpy as np
import cv2
import os
from torch import random

# Return a list of files with the specified type from the specified path
def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

# Data paths
train_img_path = '../data/Drishti-GS1_files/Test/Images/'
label_img_path = '../data/Drishti-GS1_files/Test/labels/'
save_path_img = '../data/Drishti/test/images-oc/'
save_path_label = '../data/Drishti/test/labels-oc/'

# Get training image list
file_list = return_list(train_img_path, '.png')  # Change file type to '.png'
n_original = len(file_list)

# Process each image
for i, file_name in enumerate(file_list):
    img_name = os.path.join(train_img_path, file_name)
    #label_name = os.path.join(label_img_path, file_name[:-4] + '_ODsegSoftmap.png')  # Change label file suffix
    label_name = os.path.join(label_img_path, file_name[:-4] + '_cupsegSoftmap.png')  # Change label file suffix to cupsegSoftmap.png

    # Read images
    img = cv2.imread(img_name)
    label = cv2.imread(label_name, 0)

    # Resize images and labels to 480x480
    img = cv2.resize(img, (480, 480))
    label = cv2.resize(label, (480, 480))

    # Save processed images
    cv2.imwrite(os.path.join(save_path_img, file_name), img)
    cv2.imwrite(os.path.join(save_path_label, file_name), label)

    print("Processed {} images.".format(i + 1))

print("Resizing completed for a total of {} images.".format(len(file_list)))


