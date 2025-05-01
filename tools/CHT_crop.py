import os
import cv2
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks

def crop_and_pad_image(image, label, center, size):
    x0 = center[0] - size // 2
    y0 = center[1] - size // 2
    x1 = x0 + size
    y1 = y0 + size

    # Crop image and label
    cropped_image = image[y0:y1, x0:x1]
    cropped_label = label[y0:y1, x0:x1]

    # Pad image and label if necessary
    if cropped_image.shape[0] < size:
        pad_x = size - cropped_image.shape[0]
        cropped_image = cv2.copyMakeBorder(cropped_image, pad_x, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cropped_label = cv2.copyMakeBorder(cropped_label, pad_x, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    if cropped_image.shape[1] < size:
        pad_y = size - cropped_image.shape[1]
        cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, pad_y, 0, cv2.BORDER_CONSTANT, value=0)
        cropped_label = cv2.copyMakeBorder(cropped_label, 0, 0, pad_y, 0, cv2.BORDER_CONSTANT, value=255)

    return cropped_image, cropped_label

def detect_circle(image):
    # Ensure the image has only one channel (if not grayscale)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess label image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 35)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel)

    # Detect circles using Hough transform
    hough_radii = np.arange(60, 200, 1)
    hough_res = hough_circle(edges, hough_radii)
    centers, _, _ = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(centers) > 0:
        center_x, center_y = centers[0]
        return center_x, center_y
    else:
        return None, None

def detect_drs_circle(image):
    # Ensure the image has only one channel (if not grayscale)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess label image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 35)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel)

    # Detect circles using Hough transform
    hough_radii = np.arange(60, 200, 1)
    hough_res = hough_circle(edges, hough_radii)
    _, y_centers, x_centers, _ = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(x_centers) > 0:
        center_x, center_y = x_centers[0], y_centers[0]
        return center_x, center_y
    else:
        return None, None

def process_dataset(input_path, output_path, label_path,output_label_path):
    # Get image list
    imagelist = os.listdir(input_path)

    # Process each image
    for filename in imagelist:
        if not filename.endswith(".jpg"):
            continue

        # Read image and label
        image = cv2.imread(os.path.join(input_path, filename))
        label = cv2.imread(os.path.join(label_path, filename[:-4] + '.bmp'), 0)

        # Detect circle center
        center_x, center_y = detect_circle(label)
        if center_x is None or center_y is None:
            print(f"No circle detected in {filename}")
            continue

        # Crop and pad image
        cropped_image, cropped_label = crop_and_pad_image(image, label, (center_x, center_y), 480)

        # Save cropped image and label
        cv2.imwrite(os.path.join(output_path, filename), cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(output_label_path, filename[:-4] + '.bmp'), cropped_label)

        print(f"Processed {filename}")

def process_drs_dataset(input_path, out_image1_path, label1_path, label2_path, output_label1_path, output_label2_path,out_image2_path):
    # Get image list
    imagelist = os.listdir(input_path)

    # Process each image
    for filename in imagelist:
        if not filename.endswith(".png"):
            continue

        # Read image
        image = cv2.imread(os.path.join(input_path, filename))

        # Find corresponding label files
        label1_filename = filename.replace(".png", "_ODsegSoftmap.png")
        label2_filename = filename.replace(".png", "_cupsegSoftmap.png")
        label1 = cv2.imread(os.path.join(label1_path, label1_filename), 0)
        label2 = cv2.imread(os.path.join(label2_path, label2_filename), 0)

        # Detect circle center using label1
        center_x1, center_y1 = detect_drs_circle(label1)
        if center_x1 is None or center_y1 is None:
            print(f"No circle detected in {filename}")
            continue
        # Detect circle center using label2
        center_x2, center_y2 = detect_drs_circle(label2)
        if center_x2 is None or center_y2 is None:
            print(f"No circle detected in {filename}")
            continue

        # Crop and pad image, label1, and label2
        cropped_image1, cropped_label1 = crop_and_pad_image(image, label1, (center_x1, center_y1), 480)
        cropped_image2, cropped_label2 = crop_and_pad_image(image, label2, (center_x2, center_y2), 480)

        # Save cropped image, label1, and label2
        cv2.imwrite(os.path.join(out_image1_path, filename), cropped_image1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(output_label1_path, filename), cropped_label1)

        cv2.imwrite(os.path.join(out_image2_path, filename), cropped_image2, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(os.path.join(output_label2_path, filename), cropped_label2)

        print(f"Processed {filename}")

# Process REFUGE dataset
# Process training dataset
process_dataset('../data/crop400/data/train/', '../data/crop/train2/', '../data/crop400/data/train-gt/','../data/crop/train-gt2/')
# Process validation dataset
process_dataset('../data/crop400/data/val/', '../data/crop/val2/', '../data/crop400/data/val-gt/','../data/crop/val-gt2/')
# Process test dataset
process_dataset('../data/crop400/data/test/', '../data/crop/test2/', '../data/crop400/data/test-gt/','../data/crop/test-gt2/')

# Process Drishti dataset
# Process training dataset
# process_drs_dataset('../data/Drishti-GS1_files/Training/Images/', '../data/Drishti-GS1_files/crop/train/images1/',
#                  '../data/Drishti-GS1_files/Training/labels/', '../data/Drishti-GS1_files/Training/labels/',
#                  '../data/Drishti-GS1_files/crop/train/labels1/', '../data/Drishti-GS1_files/crop/train/labels2/',
#                     '../data/Drishti-GS1_files/crop/train/images2/')
# # Process test dataset
# process_drs_dataset('../data/Drishti-GS1_files/Test/Images/', '../data/Drishti-GS1_files/crop/test/images1/',
#                  '../data/Drishti-GS1_files/Test/labels/', '../data/Drishti-GS1_files/Test/labels/',
#                  '../data/Drishti-GS1_files/crop/test/labels1/', '../data/Drishti-GS1_files/crop/test/labels2/',
#                     '../data/Drishti-GS1_files/crop/test/images2/')