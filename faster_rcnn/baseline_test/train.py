from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os

from util import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

def train_fn(num_epochs, train_data_loader, optimizer, model, device):
    best_loss = 1000
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)
            # ---- loss 값을 어떻게 ? config hyun
            print("-"*20)
            print(f"loss_dict : {loss_dict}")
            print("-"*20)

            for loss in loss_dict.values():
                print(loss)
            losses = sum(loss for loss in loss_dict.values())

            print("-"*20)
            print(f"losses : {losses}")
            print(type(losses))
            print("-"*20)
            
            loss_value = losses.item()
            print(f"loss_value : {loss_value}")
            print("-"*20)

            loss_hist.send(loss_value)
            print(f"loss_hist : {loss_hist}")
            print("-"*20)
            break
            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = './checkpoints/faster_rcnn_torchvision_checkpoints.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value