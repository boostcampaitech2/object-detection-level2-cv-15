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
from configparser import ConfigParser
from trainer import train_fn #trainer
from trainer import collate_fn
from dataset import CustomDataset #dataset
from dataset import get_train_transform
from model import * #model list file 불러오기
from optimizer import *

def get_model(config):
    if config.model_name == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif config.model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif config.model_name == 'mobilenet_v3_large_320_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    elif config.model_name == 'retinanet_resnet50_fpn':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)     
    elif config.model_name == 'ssd300_vgg16':
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)     
    elif config.model_name == 'maskrcnn_resnet50_fpn':
        model =torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)     
    elif config.model_name == 'keypointrcnn_resnet50_fpn':
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)   
    return model