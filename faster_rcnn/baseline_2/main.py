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
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import *


from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from configparser import ConfigParser
from trainer import train_fn #trainer
from trainer import collate_fn
from dataset import CustomDataset #dataset
from dataset import get_train_transform
from model import *
from optimizer import *
from setting import *

if __name__ == '__main__':
    conf = JsonConfigFileManager("./config.json")
    config = conf.values

    #=========================config====================================#

    num_epochs = config["h_param"]["num_epochs"]
    num_workers = config["h_param"]["num_workers"]
    num_classes = config["h_param"]["num_classes"]
    batch_size = config["h_param"]["batch_size"]
    lr = config["h_param"]["lr"]
    
    annotation = config["path"]["annotation_dir"] # annotation 경로
    data_dir = config["path"]["data_dir"] # data_dir 경로

    model = get_model(config["training"]["model_name"])
    box_model_name = get_model(config["training"]["box_model_name"])
    optimizer = get_optimizer(model, config["training"]["optimizer"], lr)

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
    #소정
    # custom_rpn = AnchorGenerator(sizes=((32, 64, 128, 512)), aspect_ratios=((0.5, 1.0, 2.0))) 
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    # torchvision model 불러오기

    #num_classes = 11 # class 개수= 10 + backgroundb
    # get number of input features for the classifier
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = get_box_model(config)
   # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # box model 점검
    model.to(device)

    # training
    train_fn(num_epochs, train_data_loader, optimizer, model, device)