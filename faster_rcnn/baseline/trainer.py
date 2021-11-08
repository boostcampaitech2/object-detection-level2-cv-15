from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utill import *
from optimizer import *
import torch
import os
import sys

import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


def train_per_epoches(train_data_loader, model, optimizer, loss_hist, device):
    loss_hist.reset()
    model.train()
    for images, targets, image_ids in tqdm(train_data_loader):
    
        # gpu 계산을 위해 image.to(device)
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # calculate loss, loss 값을 뽑으니까 mAP를 계산하기가.. 힘들거 같은데 ?
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value)

        # backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    return loss_hist

def valid_per_epoches(valid_data_loader, model, anno_valDir, score_threshold, device):
    model.eval()
    outputs = []
    GT_JSON = anno_valDir
    for images, targets, image_ids in tqdm(valid_data_loader):
        
        # gpu 계산을 위해 image.to(device)
        images = list(image.float().to(device) for image in images)
        ids = list(image_id for image_id in image_ids)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # calculate loss
        pred = model(images)
        for id, out in zip(ids, pred):
            outputs.append({'id': id, 'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
            
        #print(len(image_ids))
        #print(len(outputs[0]["boxes"]),len(outputs[0]["scores"]),len(outputs[0]["labels"]))
    valid_mAP, average_precisions = calculate_mAP(GT_JSON, outputs, score_threshold)

    return valid_mAP, average_precisions

def train_fn(train_data_loader, valid_data_loader, model, device, fold_ind, anno_valDir, config_id, configs):
    num_epochs = config_id["num_epochs"]
    model_name = config_id["model_name"]
    lr = config_id["lr"]
    optimizer = get_optimizer(model, config_id["optimizer"], lr)
    scheduler = get_scheduler(optimizer, config_id["scheduler"])
    start_id = config_id["id"]
    score_threshold = config_id["score_threshold"]
    save_model_path = configs["path"]["save_model_path"]
    early_stop = config_id["early_stop"]
    early_stopping_count = 0
    best_mAP = 0
    best_loss = 1000
    loss_hist = Averager()
    
    result = {
        f"epoch": [],
        f"train_loss_hist" : [],
        f"val_mAP" : [],
        f"fold_ind" : []
    }


    wandb.watch(model)

    for epoch in range(num_epochs):
        
        train_loss_hist = train_per_epoches(train_data_loader, model, optimizer, loss_hist, device)
        print(f"{fold_ind} Epoch #{epoch+1} loss: {train_loss_hist.value}")
        
        val_mAP, val_precision = valid_per_epoches(valid_data_loader, model, anno_valDir, score_threshold, device)
        print(f"{fold_ind} Epoch #{epoch+1} mAP: {val_mAP}")

        # dict 기록
        result = logging_with_dict(result, epoch, train_loss_hist, val_mAP, fold_ind)
       
        # wandb log 기록
        logging_with_wandb(epoch, train_loss_hist, val_mAP, fold_ind)
        
        # 콘솔 기록
        logging_with_sysprint(epoch, train_loss_hist.value, val_mAP, fold_ind)
        scheduler.step()

        if train_loss_hist.value < best_loss:
            best_loss = train_loss_hist.value
            print("-"*10, f"best_loss {best_loss}!!", "-"*10)

        if val_mAP > best_mAP:
            print("-"*10, "Best model changed", "-"*10)
            print("-"*10, "Model_save", "-"*10)
            save_path = save_model_path + f'fold{fold_ind}_{model_name}_{start_id}.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(model.state_dict(), save_path)
            best_model = model
            best_mAP = val_mAP
            wandb.log({"best_mAP" : best_mAP})
            print("-"*10, f"Best mAP {best_mAP}!!", "-"*10)
            print("-"*10, "Saved!!", "-"*10)
        else:
            early_stopping_count += 1

        if early_stopping_count == early_stop:
            print("-"*10, "Early Stop!!!!", "-"*10)
            break

    return save_path


def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def infer_fn(indx, model, start_id, test_data_loader, check_point, annotation_test, score_threshold, device, configs):
    save_submission_path = configs["path"]["save_submission_path"]
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()

    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation_test)

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
    submission_path = save_submission_path + f'fold{indx}_submission_{start_id}.csv'
    submission.to_csv(submission_path, index=None)
    print(submission.head())

    return submission_path

def ensemble_submission(submission_df, annotation_test, start_id, configs):
    save_submission_path = configs["path"]["save_submission_path"]
    image_ids = submission_df[0]['image_id'].tolist()
    coco = COCO(annotation_test)
    
    prediction_strings = []
    file_names = []
    iou_thr = 0.4

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
    
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()
        
        if len(predict_list)==0 or len(predict_list)==1:
            continue
            
        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []
        
        for box in predict_list[:, 2:6].tolist():
            box[0] = float(box[0]) / image_info['width']
            box[1] = float(box[1]) / image_info['height']
            box[2] = float(box[2]) / image_info['width']
            box[3] = float(box[3]) / image_info['height']
            box_list.append(box)
            
        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    
    if len(boxes_list):
        boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
    prediction_strings.append(prediction_string)
    file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(save_submission_path + f'id_{start_id}_submission_ensemble_{iou_thr}_testset.csv')
    submission.head()