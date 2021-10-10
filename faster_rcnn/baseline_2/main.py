from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from configparser import ConfigParser
from trainer import train_fn #trainer
from trainer import collate_fn
from dataset import CustomDataset #dataset
from dataset import get_train_transform
from model import * #model list file 불러오기
from optimizer import *
from setting import *
#from inference import *

import sys

if __name__ == '__main__':
    conf = JsonConfigFileManager("./config.json")
    config = conf.values

    #=========================config====================================#
    
    start_id = config["training"]["id"]

    annotation = config["path"]["annotation_dir"] # annotation 경로
    #hyuns
    #annotation_test = config["path"]["annotation_test_dir"] 
    data_dir = config["path"]["data_dir"] # data_dir 경로

    for config in config["model"]:
        if config["id"] == start_id:
            num_epochs = config["num_epochs"]
            num_workers = config["num_workers"]
            num_classes = config["num_classes"]
            batch_size = config["batch_size"]
            lr = config["lr"]

            model = get_model(config["model_name"])
            box_model_name = get_box_model(config["box_model_name"])
            optimizer = get_optimizer(model, config["optimizer"], lr)

    #=========================config====================================#

    # Dataset

    train_dataset = CustomDataset(annotation, data_dir, get_train_transform()) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Roi heads 설정
    if box_model_name != None:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = box_model_name(in_features, num_classes)
    
    model.to(device)

    # training
    # hyuns add start_id
    train_fn(start_id, num_epochs, train_data_loader, optimizer, model, device)

    # inference
    #hyuns
    # test_dataset = CustomDataset(annotation, data_dir)
    # score_threshold = 0.05
    # check_point = './checkpoints/faster_rcnn_torchvision_checkpoints.pth' # 체크포인트 경로
    
    # test_data_loader = DataLoader(
    #     test_dataset,
    #     batch_size=8,
    #     shuffle=False,
    #     num_workers=4
    # )

    # infer_fn()

    
    # train_end
    print("-" * 20)

    print("end")
