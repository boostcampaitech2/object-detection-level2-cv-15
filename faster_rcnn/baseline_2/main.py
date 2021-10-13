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
from dataset import CustomTestDataset
from dataset import CustomDataset
from trainer import ensemble_submission, train_fn #trainer
from trainer import collate_fn
from trainer import infer_fn
# from dataset import stratified_group_k_fold 
# from dataset import CustomDataset #dataset
# from dataset import get_train_transform
from dataset import *
from model import * #model list file 불러오기

from setting import *
import wandb 
#from inference import *

import sys

if __name__ == '__main__':
    
    conf = JsonConfigFileManager("./config.json")
    configs = conf.values

    #=========================config====================================#
    
    #training se
    start_id = configs["training"]["id"]
    fold_num = configs["training"]["fold_num"]

    wandb_project_name = configs["wandb"].wandb_project_name
    wandb_entity = configs["wandb"].wandb_entity
    wandb_group_name = configs["wandb"].wandb_group_name

    annotation = configs["path"]["annotation_dir"] # annotation 경로
    annotation_test = configs["path"]["annotation_test_dir"] 
    data_dir = configs["path"]["data_dir"] # data_dir 경로
    save_model_path :  configs["path"]["save_model_path"]
    save_submission_path :  configs["path"]["save_submission_path"]

    for index, config_data in enumerate(configs["model"]):
        if config_data["id"] == start_id:
            config_id = config_data
            num_epochs = config_id["num_epochs"]
            num_workers = config_id["num_workers"]
            num_classes = config_id["num_classes"]
            batch_size = config_id["batch_size"]
            lr = config_id["lr"]

            model_name = config_id["model_name"]
            model = get_model(model_name)
            box_model_name = get_box_model(config_id["box_model_name"])
    #=========================config====================================#

## ================ train with validation set - by stratified_k_fold (hyunsoo) ===============#
    #make data frame for train set & data set
    # df = make_dataframe(annotation)
    # train_x = df["id"]
    # train_y = df["class"]
    # groups = df["file_name"]

    # for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(train_x, train_y, groups, fold_num=5)):
        
    #     print(f'{fold_ind} fold start -')
    #     # dev_ind, val_ind는 list 형태로 들어가서 Series형식의 id, Image id를 뽑는다.
    #     dev_y, val_y = train_y[dev_ind], train_y[val_ind] # train index, 
    #     dev_groups, val_groups = groups[dev_ind], groups[val_ind] # Image id index,
    #     dev_id, val_id = train_x[dev_ind], train_x[val_ind]

    #     # label 비율 검사
    #     distrs.append(get_distribution(dev_y))
    #     index.append(f'development set - fold {fold_ind}')
    #     distrs.append(get_distribution(val_y))
    #     index.append(f'validation set - fold {fold_ind}')

    #     # json file 생성
    #     dev_id_json = list(set(dev_id))
    #     val_id_json = list(set(val_id))
    #     make_json(fold_ind, dev_id_json, val_id_json) # make json file

    #     # 가정 설정문,  동일한게 image file이 있는 지 확인 True이면 그대로 진행 아니라면 Assertion Error 생성
    #    assert len(set(dev_groups) & set(val_groups)) == 0
    #     # dev_ind, val_ind는 list 형태로 들어가서 Series형식의 id, Image id를 뽑는다.
    #     dev_y, val_y = train_y[dev_ind], train_y[val_ind] # train index, 
    #     dev_groups, val_groups = groups[dev_ind], groups[val_ind] # Image id index,
    #     dev_id, val_id = train_x[dev_ind], train_x[val_ind]
    
## ================ train with validation set - by stratified_k_fold (hyunsoo) ===============#
    
## ================ train with validation set - by json (hyunsoo) ===============#

    annotation_foldDir = []
    if fold_num >= 1:
        for fold_idx in range(1, fold_num+1):
            anno_trainDir, anno_valDir = make_dir(annotation, fold_idx)
            annotation_foldDir.append([anno_trainDir, anno_valDir])

    checkpoints = []
    for fold_ind, (anno_trainDir, anno_valDir) in enumerate(annotation_foldDir):

        group_name = f"{wandb_group_name}_id_{start_id}"
        experiemnt_name = f"fold_{fold_ind}_detection"
        run = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        group=group_name,
        name=experiemnt_name,
        config=config_id
        )

        print(f'{fold_ind} fold start -')
        train_dataset = CustomDataset(anno_trainDir, data_dir, get_train_transform()) 
        valid_dataset = CustomDataset(anno_valDir, data_dir, get_train_transform()) 
        
        # DataLoader 부분
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)

        # Roi heads 설정 if 2 stages
        if box_model_name != None:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = box_model_name(in_features, num_classes)

        model.to(device)

        # training
        model_path = train_fn(train_data_loader, valid_data_loader, model, device, fold_ind, anno_valDir, config_id, configs)
        checkpoints.append(model_path)
        run.finish()
        
        # train_end
    
    print("-" * 40 + "train - end" + "-" * 40)
    print("-" * 40 + "test_start" + "-" * 40)

    test_dataset = CustomTestDataset(annotation_test, data_dir)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
        )
    
    # 체크포인트 경로
    # inference
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    submission_files=[]
    for indx, check_point in enumerate(checkpoints):
        submission_dir = infer_fn(indx+1, model, start_id, test_data_loader, check_point, annotation_test, config_id["score_threshold"], device, configs)
        submission_files.append(submission_dir)

    submission_df = [pd.read_csv(file) for file in submission_files]

    # ensemble_test
    ensemble_submission(submission_df, annotation_test, start_id, configs)

    print("-" * 40 + "test - end" + "-" * 40)

## ================ train with validation set - by json (hyunsoo) ===============#
