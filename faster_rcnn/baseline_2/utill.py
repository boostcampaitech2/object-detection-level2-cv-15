import os
import configparser
from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO

from setting import *
import wandb


# def read_config(paths):
#     values = configparser.ConfigParser()


# mAP 계산 (baseline code) - hyunsoo 10.12

def calculate_mAP(GT_JSON, outputs, score_threshold):

    # load prediction

    pred_df = make_valid_dataframe(GT_JSON, outputs, score_threshold)
    
    new_pred = []

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()
    
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        boxes = np.array(str(bbox).split(' '))
    
        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        for box in boxes:
            new_pred.append([file_name, box[0], box[1], float(box[2]), float(box[4]), float(box[3]), float(box[5])])

    gt = [] 

    coco = COCO(GT_JSON)

    for image_id in coco.getImgIds():

        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)

        file_name = image_info['file_name']

        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                       float(annotation['bbox'][0]),
                       float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                       float(annotation['bbox'][1]),
                       (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])

    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, new_pred, iou_threshold=0.5)

    return mean_ap, average_precisions

def logging_with_dict(result, epoch, train_loss_hist, val_mAP, fold_ind):
    result = {
        f"epoch": epoch,
        f"train_loss_hist" : train_loss_hist.value,
        f"val_mAP" : val_mAP,
        f"fold_ind" : fold_ind
    }

    return result

def logging_with_wandb(epoch, train_loss_hist, val_mAP, fold_ind):
    wandb.log({
                f"epoch": epoch,
                f"train_loss_hist" : train_loss_hist.value,
                f"val_mAP" : val_mAP,
                f"fold_ind" : fold_ind
                })

def logging_with_sysprint(epoch, train_loss_hist, val_mAP, fold_ind): 
    
    if fold_ind == -1:
        print(f"fold: {fold_ind} | "
            f"epoch: {epoch} | "
            f"train_loss:{train_loss_hist:.5f} | "
            f"val_mAP:{val_mAP:.5f} | "
            )
    else:
        print(f"fold: {fold_ind} | "
            f"epoch: {epoch} | "
            f"train_loss:{train_loss_hist:.5f} | "
            f"val_mAP:{val_mAP:.5f} | "
            )