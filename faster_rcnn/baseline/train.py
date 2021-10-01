import warnings
warnings.filterwarnings(action='ignore')

import os
import six
from collections import namedtuple

from dataset import *
from Loss import *
from train import *
from utill import *
from modeel import *

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

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss',
                                     'roi_loc_loss', 'roi_cls_loss',
                                     'total_loss'])

# util 함수 정의
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    #bbox_a 1개와 bbox_b k개를 비교해야하므로 None을 이용해서 차원을 늘려서 연산한다.
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

# Anchor Target Creator

# ---- faster R cnn loss ----- #
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # Localization loss 구할 때는 positive example에 대해서만 계산
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss

# ---- faster R cnn loss----- #

class AnchorTargetCreator(object):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size

        n_anchor = len(anchor) # 9216
        inside_index = get_inside_index(anchor, img_H, img_W) # (2272,)
        anchor = anchor[inside_index] # (2272, 4)
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious]) # (2272, 4)

        # map up to original set of anchors
        label = unmap(label, n_anchor, inside_index, fill=-1) # (9216,)
        loc = unmap(loc, n_anchor, inside_index, fill=0) # (9216, 4)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label) 1 :positive, 0 : negative, -1 : dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0 # 0.3

        # 가장 iou가 큰 것은 positive label
        label[gt_argmax_ious] = 1

        # positive label
        label[max_ious >= self.pos_iou_thresh] = 1 # 0.7

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

# positive, neagtive sampling
class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh # positive iou threshold
        self.neg_iou_thresh_hi = neg_iou_thresh_hi # negitave iou threshold = (neg_iou_thresh_hi ~ neg_iou_thresh_lo)
        self.neg_iou_thresh_lo = neg_iou_thresh_lo 

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio) # positive image 갯수 = 32
        iou = bbox_iou(roi, bbox) # RoI와 bounding box IoU
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1 # class label [0, n_fg_class - 1] -> [1, n_fg_class].

        # positive sample 선택 (>= pos_iou_thresh IoU)
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Negative sample 선택 [neg_iou_thresh_lo, neg_iou_thresh_hi)
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative sample의 label = 0
        sample_roi = roi[keep_index] # (128, 4)

        # sample roi와 gt_bbox를 이용해 bbox regression에서 regression해야할 ground truth loc값(t_x, t_y, t_w, t_h) 계산
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) # (128, 4)
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

# Trainer 정의

class FasterRCNNTrainer(nn.Module):

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # training 상태 보여주는 지표
        self.rpn_cm = ConfusionMeter(2) # confusion matrix for classification
        self.roi_cm = ConfusionMeter(11)  # confusion matrix for classification
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # VGG (features extractor)
        features = self.faster_rcnn.extractor(imgs)
        
        # RPN (region proposal)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        """
        sample roi =  rpn에서 nms 거친 2000개의 roi들 중 positive/negative 비율 고려해 최종 sampling한 roi
        """
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        # NOTE it's all zero because now it only support for batch=1 now
        # Faster R-CNN head (prediction head)
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features,sample_roi,sample_roi_index) 

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox),anchor,img_size) 
        gt_rpn_label = totensor(gt_rpn_label).long() 
        gt_rpn_loc = totensor(gt_rpn_loc) 
        
        # rpn bounding box regression loss
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        # rpn classification loss
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0] 
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long() 
        gt_roi_loc = totensor(gt_roi_loc) 

        # faster rcnn bounding box regression loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # faster rcnn classification loss
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        
        self.roi_cm.add(totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] # total_loss == sum(losses)

        return LossTuple(*losses)
    
    # training
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses
    
    # checkpoint 만들기
    def save(self, save_optimizer=False, save_path=None):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = './checkpoints/faster_rcnn_scratch_checkpoints.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path
    
    # checkpoint load
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}



def train():
    # Train dataset 불러오기
    # dataset = TrainDataset()
    annotation = os.path.join(data_dir,'train.json')
    dataset = TrainCustom(annotation, data_dir, transforms=True)
    print('load data')
    dataloader = data_.DataLoader(dataset, 
                                  batch_size=1,     # only batch_size=1 support
                                  shuffle=True, 
                                  pin_memory=False,
                                  num_workers=4)
    
    # faster rcnn 불러오기
    faster_rcnn = FasterRCNNVGG16().cuda()
    print('model construct completed')
    
    # faster rcnn trainer 불러오기
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    
    # checkpoint load
    if train_load_path:
        trainer.load(train_load_path)
        print('load pretrained model from %s' % train_load_path)
    
    #lr_ = learning_rate
    best_loss = 1000
    for epoch in range(epochs):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in enumerate(tqdm(dataloader)):
            
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, float(scale))
        
        losses = trainer.get_meter_data()
        print(f"Epoch #{epoch+1} loss: {losses}")
        if losses['total_loss'] < best_loss :
            trainer.save()
            
        if epoch == 9:
            trainer.faster_rcnn.scale_lr(lr_decay)
            lr_ = lr_ * lr_decay

        if epoch == 9: # 변경
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories

    # epochs= 1
    # learning_rate = 1e-
    # lr_decay = 0.1
    # weight_decay = 0.0005
    # use_drop = False   # use dropout in RoIHead
    
    # rpn_sigma = 3.     # sigma for l1_smooth_loss (RPN loss)
    # roi_sigma = 1.     # sigma for l1_smooth_loss (ROI loss)
    
    # data_dir = '../dataset'   # 데이터 경로 
    # train_load_path = None  # train시 checkpoint 경로
    inf_load_path = './checkpoints/faster_rcnn_scratch_checkpoints.pth' # inference시 체크포인트 경로
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/detection/dataset'))
    #parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)