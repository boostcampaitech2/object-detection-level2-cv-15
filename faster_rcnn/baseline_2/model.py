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
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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

from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation


# from .rpn import RPNHead, RegionProposalNetwork
# from .roi_heads import RoIHeads
# from .transform import GeneralizedRCNNTransform
# from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers, mobilenet_backbone

def get_model(model_name):
    
    if model_name == 'fasterrcnn_resnet50_fpn':
        model_name = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        model_name = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
        model_name = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    elif model_name == 'retinanet_resnet50_fpn':
        model_name = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)     
    elif model_name == 'ssd300_vgg16':
        model_name = torchvision.models.detection.ssd300_vgg16(pretrained=True)     
    elif model_name == 'maskrcnn_resnet50_fpn':
        model_name =torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)     
    elif model_name == 'keypointrcnn_resnet50_fpn':
        model_name = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)   
    
    return model_name

def get_box_model(model):

    if model == 'FastRCNNPredictor':
        model = FastRCNNPredictor
    else:
        model = None
    return model