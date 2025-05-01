# Description: Test the model
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tools.read_testdata import Images_Dataset_folder
import cv2
from models.ODCS_NSNP import ODCS_NSNP
import os
from tools.post_processing import post_process_images
from tools.evaluate import *
from tqdm import tqdm
from PIL import Image

# Function to load test data
def load_test_data(test_dir, valid_size=0.0):
    Testing_Data = Images_Dataset_folder(test_dir)
    num_train = len(Testing_Data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    test_idx = indices[split:]
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(Testing_Data, batch_size=1, sampler=test_sampler, num_workers=0)
    return test_loader

# Function to load model
def load_model(model_path, device):
    net = ODCS_NSNP(3, 3)
    net.to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

def test_model_and_evaluate(test_dir, save_path_img, model_path, post_process_save_path, gt_img_path, epoch, dataset, seg_class, flag='train'):
    device = torch.device('cuda', 2)

    # Load test data
    test_loader = load_test_data(test_dir)

    # Load model
    if flag == 'train':
        net = model_path
        net.eval()  # Set model to evaluation mode
    else:
        net = load_model(model_path, device)

    # Make predictions and save results
    with torch.no_grad():
        bar = tqdm(test_loader)
        for i, data in enumerate(bar):
            inputs, image_name = data
            inputs = inputs.to(device)
            image_name = image_name[0]
            output = net(inputs)
            output = torch.argmax(output, dim=1)
            output = output.squeeze().cpu().numpy()

            # Map prediction output to corresponding pixel values
            target = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)

            # When segmenting optic disc and cup separately
            if dataset == 'Dri' or dataset == 'Rim':
                target[output == 1] = 0  # Class 1 (optic disc/cup) set to 0
                target[output == 0] = 255  # Other areas (class 0) set to 255
            else:
                target[output == 1] = 0  # Class 1 (optic disc/cup) set to 0
                target[output == 0] = 255  # Other areas (class 0) set to 255

            # Resize image to 480x480
            target = cv2.resize(target, (480, 480), interpolation=cv2.INTER_CUBIC)

            # Save prediction results
            if dataset == 'Dri' or dataset == 'Rim':
                cv2.imwrite(os.path.join(save_path_img, image_name[:-4] + '.png'), target)
            else:
                cv2.imwrite(os.path.join(save_path_img, image_name[:-4] + '.bmp'), target)

            # Progress bar
            bar.set_description(f'Test process {image_name[:-4]}')

    # Post-processing
    post_process_images(save_path_img, post_process_save_path)

    # Evaluation
    if flag == 'train':
        if seg_class == 'OC':
            mean_disc_dc, mean_cup_dc = evaluate_segmentation_results_OC('Train_Dri_oc_snp_epoch' + str(epoch), post_process_save_path, gt_img_path, dataset)
        else:
            mean_disc_dc, mean_cup_dc = evaluate_segmentation_results_OD('Train_Dri_od_snp_epoch' + str(epoch), post_process_save_path, gt_img_path, dataset)
    else:
        if seg_class == 'OC':
            mean_disc_dc, mean_cup_dc = evaluate_segmentation_results_OC(os.path.basename(model_path), post_process_save_path, gt_img_path, dataset)
        else:
            mean_disc_dc, mean_cup_dc = evaluate_segmentation_results_OD(os.path.basename(model_path), post_process_save_path, gt_img_path, dataset)

    # Return evaluation results
    if flag == 'train':
        if seg_class == 'OC':
            return mean_cup_dc
        else:
            return mean_disc_dc


if __name__ == "__main__":
    test_dir = './data/Drishti/oc/test/images/'
    save_path_img = './data/Drishti/oc/test/seg/'
    model_path = './checkpoints/Dri_oc_snp_epoch1.pkl'
    post_process_save_path = './data/Drishti/oc/test/seg-post/'
    gt_img_path = './data/Drishti/oc/test/labels/'

    test_model_and_evaluate(test_dir, save_path_img, model_path, post_process_save_path, gt_img_path, 'Test', 'Dri', 'OC', flag='test')