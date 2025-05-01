import os
import cv2
import numpy as np
from tools.evaluation_metrics_for_segmentation import *


def evaluate_segmentation_results(model_name,pred_img_path, gt_img_path):
    # Function to return list of files with specified type in a directory
    def return_list(data_path, data_type):
        file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
        return file_list

    file_list = return_list(pred_img_path, '.bmp')
    n = len(file_list)
    DC_score_cup = {}
    DC_score_disc = {}
    JAC_score_cup = {}
    JAC_score_disc = {}
    ACC_score_cup = {}
    ACC_score_disc = {}
    SEN_score_cup = {}
    SEN_score_disc = {}
    SPC_score_cup = {}
    SPC_score_disc = {}
    CDR_score = {}

    cup_dices = []
    disc_dices = []
    cup_JAC = []
    disc_JAC = []
    cup_ACC = []
    disc_ACC = []
    cup_SEN = []
    disc_SEN = []
    cup_SPC = []
    disc_SPC = []
    CDR = []

    for i in range(n):
        temp_list = file_list[i]
        pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.bmp')
        gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.bmp')
        pred = cv2.imread(pred_name, 0)
        gt = cv2.imread(gt_name, 0)
        cup_dice, cup_jac, cup_acc, cup_sen, cup_spc, disc_dice, disc_jac, disc_acc, disc_sen, disc_spc, cdr = evaluate_binary_segmentation_1(pred, gt)
        DC_score_cup[temp_list] = cup_dice
        DC_score_disc[temp_list] = disc_dice
        JAC_score_cup[temp_list] = cup_jac
        JAC_score_disc[temp_list] = disc_jac
        ACC_score_cup[temp_list] = cup_acc
        ACC_score_disc[temp_list] = disc_acc
        SEN_score_cup[temp_list] = cup_sen
        SEN_score_disc[temp_list] = disc_sen
        SPC_score_cup[temp_list] = cup_spc
        SPC_score_disc[temp_list] = disc_spc
        CDR_score[temp_list] = cdr

        cup_dices.append(cup_dice)
        disc_dices.append(disc_dice)
        cup_JAC.append(cup_jac)
        disc_JAC.append(disc_jac)
        cup_ACC.append(cup_acc)
        disc_ACC.append(disc_acc)
        cup_SEN.append(cup_sen)
        disc_SEN.append(disc_sen)
        cup_SPC.append(cup_spc)
        disc_SPC.append(disc_spc)
        CDR.append(cdr)

    mean_cup_dice = np.mean(cup_dices)
    mean_disc_dice = np.mean(disc_dices)
    DC_score_cup['DC_cup_mean_score'] = mean_cup_dice
    DC_score_disc['DC_disc_mean_score'] = mean_disc_dice
    mean_cup_jac = np.mean(cup_JAC)
    mean_disc_jac = np.mean(disc_JAC)
    JAC_score_cup['JAC_cup_mean_score'] = mean_cup_jac
    JAC_score_disc['JAC_disc_mean_score'] = mean_disc_jac
    mean_cup_acc = np.mean(cup_ACC)
    mean_disc_acc = np.mean(disc_ACC)
    ACC_score_cup['ACC_cup_mean_score'] = mean_cup_acc
    ACC_score_disc['ACC_disc_mean_score'] = mean_disc_acc
    mean_cup_sen = np.mean(cup_SEN)
    mean_disc_sen = np.mean(disc_SEN)
    SEN_score_cup['SEN_cup_mean_score'] = mean_cup_sen
    SEN_score_disc['SEN_disc_mean_score'] = mean_disc_sen
    mean_cup_spc = np.mean(cup_SPC)
    mean_disc_spc = np.mean(disc_SPC)
    SPC_score_cup['SPC_cup_mean_score'] = mean_cup_spc
    SPC_score_disc['SPC_disc_mean_score'] = mean_disc_spc

    mean_cdr = np.mean(CDR)
    CDR_score['CDR_mean_score'] = mean_cdr

    # Write to file
    with open('test.txt', 'a') as f:
        f.write('#Test model: ' + model_name + '\r')
        f.write('OD: DC mean :{}        JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
            mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_cup_sen, mean_disc_spc))
        f.write('OC: DC mean :{}        JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
            mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))
        f.write('#\n')

    # Print test results
    print('#Test model: ' + model_name + '\n')
    print('OD: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}'.format(
            mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_cup_sen, mean_disc_spc))
    print('OC: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}'.format(
            mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))

    # Return the best model results for testing
    return {
        mean_disc_dice,
        mean_cup_dice
    }

def evaluate_segmentation_results_OD(model_name, pred_img_path, gt_img_path, flag):
    def return_list(data_path, data_type):
        file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
        return file_list

    if flag == 'Dri' or flag == 'Rim':
        file_list = return_list(pred_img_path, '.png')
    else:
        file_list = return_list(pred_img_path, '.bmp')

    n = len(file_list)
    DC_score_disc = {}
    JAC_score_disc = {}
    ACC_score_disc = {}
    SEN_score_disc = {}
    SPC_score_disc = {}
    CDR_score = {}

    disc_dices = []
    disc_JAC = []
    disc_ACC = []
    disc_SEN = []
    disc_SPC = []
    CDR = []

    for i in range(n):
        temp_list = file_list[i]
        if flag == 'Dri' or flag == 'Rim':
            pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.png')
            gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.png')
        else:
            pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.bmp')
            gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.bmp')

        pred = cv2.imread(pred_name, 0)
        gt = cv2.imread(gt_name, 0)

        disc_dice, disc_jac, disc_acc, disc_sen, disc_spc, cdr = evaluate_binary_segmentation_OD(pred, gt)
        DC_score_disc[temp_list] = disc_dice
        JAC_score_disc[temp_list] = disc_jac
        ACC_score_disc[temp_list] = disc_acc
        SEN_score_disc[temp_list] = disc_sen
        SPC_score_disc[temp_list] = disc_spc
        CDR_score[temp_list] = cdr

        disc_dices.append(disc_dice)
        disc_JAC.append(disc_jac)
        disc_ACC.append(disc_acc)
        disc_SEN.append(disc_sen)
        disc_SPC.append(disc_spc)
        CDR.append(cdr)

    mean_disc_dice = np.mean(disc_dices)
    DC_score_disc['DC_disc_mean_score'] = mean_disc_dice
    mean_disc_jac = np.mean(disc_JAC)
    JAC_score_disc['JAC_disc_mean_score'] = mean_disc_jac
    mean_disc_acc = np.mean(disc_ACC)
    ACC_score_disc['ACC_disc_mean_score'] = mean_disc_acc
    mean_disc_sen = np.mean(disc_SEN)
    SEN_score_disc['SEN_disc_mean_score'] = mean_disc_sen
    mean_disc_spc = np.mean(disc_SPC)
    SPC_score_disc['SPC_disc_mean_score'] = mean_disc_spc

    mean_cdr = np.mean(CDR)
    CDR_score['CDR_mean_score'] = mean_cdr

    if flag == 'Dri':
        # Write to file
        with open('od_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OD: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_disc_sen, mean_disc_spc))
            f.write('#\n')
    elif flag == 'Rim':
        # Write to file
        with open('Rim_od_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OD: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_disc_sen, mean_disc_spc))
            f.write('#\n')
    else:
        # Write to file
        with open('Ref_od_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OD: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_disc_sen, mean_disc_spc))
            f.write('#\n')

    # Print test results
    print('#Test model: ' + model_name + '\n')
    print('OD: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}'.format(
            mean_disc_dice, mean_disc_jac, mean_disc_acc, mean_disc_sen, mean_disc_spc))

    return mean_disc_dice, 0


def evaluate_segmentation_results_OC(model_name, pred_img_path, gt_img_path, flag):
    # Function to return list of files with specified type in a directory
    def return_list(data_path, data_type):
        file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
        return file_list

    if flag == 'Dri' or flag == 'Rim':
        file_list = return_list(pred_img_path, '.png')
    else:
        file_list = return_list(pred_img_path, '.bmp')

    n = len(file_list)
    DC_score_cup = {}
    JAC_score_cup = {}
    ACC_score_cup = {}
    SEN_score_cup = {}
    SPC_score_cup = {}
    CDR_score = {}

    cup_dices = []
    cup_JAC = []
    cup_ACC = []
    cup_SEN = []
    cup_SPC = []
    CDR = []

    for i in range(n):
        temp_list = file_list[i]
        if flag == 'Dri' or flag == 'Rim':
            pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.png')
            gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.png')
        else:
            pred_name = os.path.join(pred_img_path, temp_list[:-4] + '.bmp')
            gt_name = os.path.join(gt_img_path, temp_list[:-4] + '.bmp')

        pred = cv2.imread(pred_name, 0)
        gt = cv2.imread(gt_name, 0)

        cup_dice, cup_jac, cup_acc, cup_sen, cup_spc, cdr = evaluate_binary_segmentation_OC(pred, gt)
        DC_score_cup[temp_list] = cup_dice
        JAC_score_cup[temp_list] = cup_jac
        ACC_score_cup[temp_list] = cup_acc
        SEN_score_cup[temp_list] = cup_sen
        SPC_score_cup[temp_list] = cup_spc
        CDR_score[temp_list] = cdr

        cup_dices.append(cup_dice)
        cup_JAC.append(cup_jac)
        cup_ACC.append(cup_acc)
        cup_SEN.append(cup_sen)
        cup_SPC.append(cup_spc)
        CDR.append(cdr)

    mean_cup_dice = np.mean(cup_dices)
    DC_score_cup['DC_cup_mean_score'] = mean_cup_dice
    mean_cup_jac = np.mean(cup_JAC)
    JAC_score_cup['JAC_cup_mean_score'] = mean_cup_jac
    mean_cup_acc = np.mean(cup_ACC)
    ACC_score_cup['ACC_cup_mean_score'] = mean_cup_acc
    mean_cup_sen = np.mean(cup_SEN)
    SEN_score_cup['SEN_cup_mean_score'] = mean_cup_sen
    mean_cup_spc = np.mean(cup_SPC)
    SPC_score_cup['SPC_cup_mean_score'] = mean_cup_spc

    mean_cdr = np.mean(CDR)
    CDR_score['CDR_mean_score'] = mean_cdr

    if flag == 'Dri':
        # Write to file
        with open('oc_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OC: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))
            f.write('#\n')
    elif flag == 'Rim':
        # Write to file
        with open('Rim_oc_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OC: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))
            f.write('#\n')
    else:
        # Write to file
        with open('Ref_oc_test.txt', 'a') as f:
            f.write('#Test model: ' + model_name + '\n')
            f.write(
                'OC: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}\n'.format(
                    mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))
            f.write('#\n')


    # Print test results
    print('#Test model: ' + model_name + '\n')
    print('OC: DC mean :{}          JAC mean :{}        PRE(ACC) mean :{}           SEN mean :{}        SPC mean :{}'.format(
            mean_cup_dice, mean_cup_jac, mean_cup_acc, mean_cup_sen, mean_cup_spc))

    return 0, mean_cup_dice


