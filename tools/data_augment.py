import shutil
import numpy as np
import cv2
import os
from torch import random

# Return a list of files with the specified type from the specified path
def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

# Define brightness adjustment function
def contrast_img(img, c, b):
    rows, cols, channels = img.shape  # Get the number of rows, columns, and channels of the image
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1 - c, b)
    return dst

# Define salt and pepper noise addition function
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randX = np.random.randint(0, src.shape[0] - 1)
        randY = np.random.randint(0, src.shape[1] - 1)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randX, randY] = 0
        else:
            SP_NoiseImg[randX, randY] = 255
    return SP_NoiseImg

# Define Gaussian noise addition function
def addGaussianNoise(image, percetage):
    G_Noiseimg = image
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg

# Define image translation function
def translate_img(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(image, M, (cols, rows))
    return translated_img

# Define image rotation function
def rotate_img(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))
    return rotated_img

# Data paths
train_img_path = '../data/ref-crop/data-800-img2/'
label_img_path = '../data/ref-crop/data-800-gt2-od/'
save_path_img = '../data/Refuge/od/train/images/'
save_path_label = '../data/Refuge/od/train/labels/'

# Get training image list
file_list = return_list(train_img_path, '.jpg')
n_original = len(file_list)
n_augmented = 15000 - n_original  # Total number of augmented images required

# Copy original images and labels to augmentation output directory
for file in file_list:
    img_name = os.path.join(train_img_path, file)
    label_name = os.path.join(label_img_path, file[:-4] + '.bmp')
    shutil.copy(img_name, save_path_img)
    shutil.copy(label_name, save_path_label)

# Process each image
for i in range(n_augmented):
    # Randomly select an original image
    temp_list = file_list[np.random.randint(0, n_original)]
    img_name = os.path.join(train_img_path, temp_list[:-4] + '.jpg')
    label_name = os.path.join(label_img_path, temp_list[:-4] + '.bmp')

    # Read images
    img = cv2.imread(img_name)
    label = cv2.imread(label_name, 0)

    # Randomly select an augmentation method and parameters
    augmentation_method = np.random.randint(1, 6)

    if augmentation_method == 1:  # Brightness adjustment
        img_augmented = contrast_img(img, np.random.uniform(0.5, 1.5), np.random.randint(-50, 50))
        label_augmented = label

    elif augmentation_method == 2:  # Salt and pepper noise addition
        img_augmented = SaltAndPepper(img, np.random.uniform(0.01, 0.05))
        label_augmented = label

    elif augmentation_method == 3:  # Gaussian noise addition
        r, g, b = cv2.split(img)
        r1 = addGaussianNoise(r, 0.005)
        g1 = addGaussianNoise(g, 0.005)
        b1 = addGaussianNoise(b, 0.005)
        img_augmented = cv2.merge([r1, g1, b1])
        label_augmented = label

    #Remove the translation augmentation method
    elif augmentation_method == 4:  # Translation
        tx = np.random.randint(-50, 50)
        ty = np.random.randint(-50, 50)
        img_augmented = translate_img(img, tx, ty)
        label_augmented = translate_img(label, tx, ty)

    else:  # Rotation
        angle = np.random.randint(-30, 30)
        img_augmented = rotate_img(img, angle)
        label_augmented = rotate_img(label, angle)

    # Save processed images
    cv2.imwrite(save_path_img + temp_list[:-4] + 'N_{}.jpg'.format(i), img_augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(save_path_label + temp_list[:-4] + 'N_{}.bmp'.format(i), label_augmented)

    print("Processed {} images.".format(i + 1))

print("Augmentation completed for a total of {} images.".format(n_augmented))
