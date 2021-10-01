import warnings
warnings.filterwarnings(action='ignore')

import os
import six
from collections import namedtuple
from importlib import import_module

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

import numpy as np  
import pandas as pd
from tqdm import tqdm

from torchvision.models import vgg16
from torchvision.ops import RoIPool
from torchvision.ops import nms

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils import data as data_

from torchnet.meter import ConfusionMeter, AverageValueMeter

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


# ---- faster_rcnn -----#

# Train dataset transform
def get_train_transform(h, w):
    return A.Compose([
        A.Resize(height = h, width = w),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# No transform
def no_transform():
    return A.Compose([
        ToTensorV2(p=1.0) # format for pytorch tensor
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# TrainDataset
class TrainCustom(Dataset):
    def __init__(self, annotation, data_dir, transforms = False):
        """
        Args:
            annotation: annotation 파일 위치
            data_dir: data가 존재하는 폴더 경로
            transforms : transform or not
        """

        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgIds=index)

        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]

        # 이미지 로드
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # 어노테이션 파일 로드
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # 박스 가져오기
        boxes = np.array([x['bbox'] for x in anns])

        # boxes (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # 레이블 가져오기
        labels = np.array([x['category_id'] for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # transform 함수 정의
        if self.transforms :
            scale = 1.0  # resize scale
            H, W, _ = image.shape
            resize_H = int(scale * H)
            resize_W = int(scale * W)
            transforms = get_train_transform(resize_H, resize_W)
        else :
            scale = 1.0
            transforms = no_transform()
        
        # transform
        sample = {
            'image': image,
            'bboxes': boxes,
            'labels': labels
        }
        sample = transforms(**sample)
        image = sample['image']
        bboxes = torch.tensor(sample['bboxes'], dtype=torch.float32)
        boxes = torch.tensor(sample['bboxes'], dtype=torch.float32)

        # bboxes (x_min, y_min, x_max, y_max) -> boxes (y_min, x_min, y_max, x_max)
        boxes[:, 0] = bboxes[:, 1]
        boxes[:, 1] = bboxes[:, 0]
        boxes[:, 2] = bboxes[:, 3]
        boxes[:, 3] = bboxes[:, 2]

        return image, boxes, labels, scale

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

# Test Datset
class TestCustom(Dataset):
    def __init__(self, annotation, data_dir):
        """
        Args:
            annotation: annotation 파일 위치
            data_dir: data가 존재하는 폴더 경로
        """

        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

    def __getitem__(self, index: int):
        
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgIds=index)

        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]

        # 이미지 로드
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = torch.tensor(image, dtype = torch.float).permute(2,0,1)
        
        return image, image.shape[1:]

    def __len__(self) -> int:
        return len(self.coco.getImgIds())


#-------------------------------------------faster r-cnn -------------------------------------#
