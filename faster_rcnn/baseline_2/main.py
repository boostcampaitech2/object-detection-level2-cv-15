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
from model import * #model list file 불러오기
from optimizer import *


from torchvision.models.detection.rpn import AnchorGenerator



def main(config):
    # 데이터셋 불러오기

    # print(list(config))

    annotation = '../../datasets/train.json' # annotation 경로
    data_dir = '../../datasets' # data_dir 경로
    train_dataset = CustomDataset(annotation, data_dir, get_train_transform()) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.getint('h_param','batch_size'),
        shuffle=False,
        num_workers=config.getint('h_param','num_workers'),
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    custom_rpn = AnchorGenerator(sizes=((32, 64, 128, 512)), aspect_ratios=((0.5, 1.0, 2.0))) 
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    # torchvision model 불러오기
    model = get_model(config)
    num_classes = 11 # class 개수= 10 + backgroundb
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512)), aspect_ratios=((0.5, 1.0, 2.0)))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
    backbone, 
    num_classes=11, 
    rpn_anchor_generator=anchor_generator, 
    box_roi_pool=roi_pooler
    )

    
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    lr=config.getfloat('h_param','lr')
    optimizer = get_optimizer(model,config)
    num_epochs = config.getint('h_param','num_epochs')


    # training
    train_fn(num_epochs, train_data_loader, optimizer, model, device)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('parser.ini')
    main(config)