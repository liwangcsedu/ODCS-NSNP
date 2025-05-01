import os
import cv2
import numpy as np

#train image
train_path = '../data/RIM-ONE/test/cup/' # Original Image
label_path = '../data/RIM-ONE/test/disk/' #Original Image label

####save path
train_save_path = '../data/rim-crop/oc/test/labels/'
label_save_path = '../data/rim-crop/oc/train/labels/'

imagelist_1 = os.listdir(train_path)

# crop train data Glaucoma
# #---train-----
for lineIdx in range(len(imagelist_1)):

    temp_txt = imagelist_1[lineIdx]
    print(temp_txt)
    #temp_txt_1 = temp_txt[:-4] + '.bmp'
    temp_txt_1 = temp_txt[:-4] + '.png'
    #temp_txt_1 = temp_txt[:-4] + '_cupsegSoftmap.png'
    #temp_txt_1 = temp_txt[:-4] + '_ODsegSoftmap.png'

    if (imagelist_1[lineIdx].endswith(".png")):
        org_img = cv2.imread(train_path + temp_txt)
        org_label = cv2.imread(label_path + temp_txt_1,0)
        gray_r = org_label
        gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
        edged = cv2.Canny(gray_blur, 15, 35)
        # cv2.imshow('edged1',edged)
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel)
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1500, param1=100, param2=10, minRadius=60,
                                   maxRadius=200)
        print('The centre point of img ' + str(temp_txt) + ' is ' + str(circles))
        circles = circles[0, 0]
        circles = np.uint16(np.around(circles))
        print(circles)

        x_center = circles[0]
        y_center = circles[1]

        # Calculate the top-left and bottom-right coordinates of the cropping rectangle
        x0 = x_center - 240
        y0 = y_center - 240
        x1 = x_center + 240
        y1 = y_center + 240

        # Ensure the cropping area does not exceed the image boundaries
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > org_img.shape[1]:
            x1 = org_img.shape[1]
        if y1 > org_img.shape[0]:
            y1 = org_img.shape[0]

        # Crop the images
        crop_img = org_img[y0:y1, x0:x1]
        crop_label = org_label[y0:y1, x0:x1]

        # Calculate the dimensions of the cropped image
        img_shape = crop_img.shape
        x_shape = img_shape[1]
        y_shape = img_shape[0]

        # If the cropped image size is less than 480x480, pad it
        if x_shape < 480:
            pad_xnum_left = (480 - x_shape) // 2
            pad_xnum_right = 480 - x_shape - pad_xnum_left
        else:
            pad_xnum_left = 0
            pad_xnum_right = 0
        if y_shape < 480:
            pad_ynum_top = (480 - y_shape) // 2
            pad_ynum_bottom = 480 - y_shape - pad_ynum_top
        else:
            pad_ynum_top = 0
            pad_ynum_bottom = 0

        # Pad the image boundaries
        crop_img = cv2.copyMakeBorder(crop_img, pad_ynum_top, pad_ynum_bottom, pad_xnum_left, pad_xnum_right,
                                      cv2.BORDER_CONSTANT, value=0)
        crop_label = cv2.copyMakeBorder(crop_label, pad_ynum_top, pad_ynum_bottom, pad_xnum_left, pad_xnum_right,
                                        cv2.BORDER_CONSTANT, value=255)

        # Save the cropped images and labels
        cv2.imwrite(train_save_path + temp_txt, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(label_save_path + temp_txt_1, crop_label)

    else:
        continue