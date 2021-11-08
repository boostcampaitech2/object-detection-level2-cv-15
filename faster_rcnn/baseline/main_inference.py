from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from Dataset_inference import CustomDataset
from inference_function import inference_fn
from setting import *




def main():

    conf = JsonConfigFileManager("../baseline_2/config.json")
    config = conf.values

    annotation = config["path"]["test_annotation_dir"] # annotation 경로
    data_dir = config["path"]["data_dir"] # data_dir 경로
    check_point=config["path"]["model_checkpoint"] #model 경로
  
    test_dataset = CustomDataset(annotation, data_dir)
    score_threshold = 0.05
    
    

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    # torchvision model 불러오기
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    num_classes = 11  # 10 class + background
    # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls                                                                                   _score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('./faster_rcnn_torchvision_submission_1010.csv', index=None)
    print(submission.head())

if __name__ == '__main__':
    main()