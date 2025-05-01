from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import cv2
import numpy as np

class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, transformI=None):

        #self.labels_dir = labels_dir  # Set label directory
        self.images_dir = images_dir  # Set image directory
        self.transformI = transformI  # Set transformation operation for input images
        # self.transformM = transformM


    def __len__(self):
        return len(self.images_dir)  # Return the size of the dataset

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):  # Iterate through the image directory
            image = Image.open(self.images_dir[i]).convert('RGB')  # Open image and convert to RGB mode
            image_name = self.images_dir[i]  # Get image filename
            # image = cv2.imread(self.images_dir[i])  # Read image using OpenCV
            # image = np.transpose(image, (2, 0, 1))  # Transpose image array dimensions

            # r,g,b = image.split()
            # image = r
            # label = Image.open(self.labels_dir[i])
            # label = cv2.imread(self.labels_dir[i])
            # label = torch.from_numpy(label)

            imgs = self.transforms(image)  # Apply transformation operations to the image
            label = []  # Initialize label list
            sample = {'images': imgs, 'labels': label, 'name': image_name}  # Create sample dictionary

        return sample  # Return sample dictionary


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, transformI=None):
        self.images = sorted(os.listdir(images_dir))  # Get list of image files in the directory and sort alphabetically
        self.images_dir = images_dir  # Set image directory
        self.transformI = transformI  # Set transformation operation for input images

        if self.transformI:  # If there is a transformation operation for input images
            self.tx = self.transformI  # Set it as the transformation operation
        else:
            self.tx = torchvision.transforms.Compose([  # Otherwise use default transformation operations
                torchvision.transforms.Resize((240, 240)),  # Resize the image
                #   torchvision.transforms.CenterCrop(960),
                #   torchvision.transforms.RandomRotation((-10,10)),
                # torchvision.transforms.RandomHorizontalFlip(),
                #  torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),  # Convert image to tensor
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.images)  # Return the size of the dataset

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')  # Open image and convert to RGB mode
        i2 = self.images[i]  # Get image filename
        # i1 = cv2.imread(self.images_dir + self.images[i])  # Read image using OpenCV
        # i1 = np.transpose(i1, (2, 0, 1))  # Transpose image array dimensions
        # r,g,b = i1.split()
        # i1 = r
        # l1 = Image.open(self.labels_dir + self.labels[i])
        # l1 = cv2.imread(self.labels_dir + self.labels[i])
        # l1 = torch.from_numpy(l1)

        return self.tx(i1), i2  # Return transformed image and image filename


class Images_Dataset_folder_pre(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, transformI=None):
        self.images = sorted(os.listdir(images_dir))  # Get list of image files in the directory and sort alphabetically
        self.images_dir = images_dir  # Set image directory
        self.transformI = transformI  # Set transformation operation for input images

        if self.transformI:  # If there is a transformation operation for input images
            self.tx = self.transformI  # Set it as the transformation operation
        else:
            self.tx = torchvision.transforms.Compose([  # Otherwise use default transformation operations
                torchvision.transforms.Resize((256, 256)),  # Resize the image
                #   torchvision.transforms.CenterCrop(960),
                #   torchvision.transforms.RandomRotation((-10,10)),
                # torchvision.transforms.RandomHorizontalFlip(),
                #  torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),  # Convert image to tensor
                # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.images)  # Return the size of the dataset

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i]).convert('RGB')  # Open image and convert to RGB mode
        img_size = i1.size  # Get image size
        i2 = self.images[i]  # Get image filename
        # i1 = cv2.imread(self.images_dir + self.images[i])  # Read image using OpenCV
        # i1 = np.transpose(i1, (2, 0, 1))  # Transpose image array dimensions
        # r,g,b = i1.split()
        # i1 = r
        # l1 = Image.open(self.labels_dir + self.labels[i])
        # l1 = cv2.imread(self.labels_dir + self.labels[i])
        # l1 = torch.from_numpy(l1)

        return self.tx(i1), i2, img_size  # Return transformed image, image filename and image size
