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

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # anchor_base는 하나의 pixel에 9개 종류의 anchor box를 나타냄
    # 이것을 enumerate시켜 전체 이미지의 pixel에 각각 9개의 anchor box를 가지게 함
    # 32x32 feature map에서는 32x32x9=9216개 anchor box가짐

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor # (9216, 4)


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    # model = vgg16()
    # model.load_state_dict(torch.load('./checkpoints/vgg16-397923af.pth'))
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict(),):
        
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios) # 9개의 anchorbox 생성
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params) # proposal_creator_params : 해당 네트워크가 training인지 testing인지 알려준다.
        n_anchor = self.anchor_base.shape[0] # anchor 개수
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  # 9*2
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)   # 9*4
        normal_init(self.conv1, 0, 0.01) # weight initalizer
        normal_init(self.score, 0, 0.01) # weight initalizer
        normal_init(self.loc, 0, 0.01)   # weight initalizer

    def forward(self, x, img_size, scale=1.):
        # x(feature map)
        n, _, hh, ww = x.shape

        # 전체 (h*w*9)개 anchor의 좌표값 # anchor_base:(9, 4)
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww) 
        n_anchor = anchor.shape[0] // (hh * ww) # anchor 개수
        
        middle = F.relu(self.conv1(x))
        
        # predicted bounding box offset
        rpn_locs = self.loc(middle)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) 

        # predicted scores for anchor (foreground or background)
        rpn_scores = self.score(middle)  
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() 
        
        # scores for foreground
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4) 
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()    
        rpn_fg_scores = rpn_fg_scores.view(n, -1)    
        
        rpn_scores = rpn_scores.view(n, -1, 2) 

        # proposal생성 (ProposalCreator)
        rois = list()        # proposal의 좌표값이 있는 bounding box array
        roi_indices = list() # roi에 해당하는 image 인덱스
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),rpn_fg_scores[i].cpu().data.numpy(),anchor, img_size,scale=scale) 
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor



class VGG16RoIHead(nn.Module):
    """
    Faster R-CNN head
    RoI pool 후에 classifier, regressior 통과
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier  
        self.cls_loc = nn.Linear(4096, n_class * 4) # bounding box regressor
        self.score = nn.Linear(4096, n_class) # Classifier

        normal_init(self.cls_loc, 0, 0.001)  # weight initialize
        normal_init(self.score, 0, 0.01)     # weight initialize

        self.n_class = n_class # 배경 포함한 class 수
        self.roi_size = roi_size # RoI-pooling 후 feature map의  높이, 너비
        self.spatial_scale = spatial_scale # roi resize scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # in case roi_indices is  ndarray
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous() 

        # 각 이미지 roi pooling 
        pool = self.roi(x, indices_and_rois) 
        # flatten 
        pool = pool.view(pool.size(0), -1)
        # fully connected
        fc7 = self.classifier(pool)
        # regression 
        roi_cls_locs = self.cls_loc(fc7)
        # softmax
        roi_scores = self.score(fc7)

        
        return roi_cls_locs, roi_scores
def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor  # extractor : vgg
        self.rpn = rpn              # rpn : region proposal network
        self.head = head            # head : RoiHead

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset()

    @property
    def n_class(self): # 최종 class 개수 (배경 포함)
        return self.head.n_class
        
    # predict 시 사용하는 forward
    # train 시 FasterRCNNTrainer을 사용하여 FasterRcnn에 있는 extractor, rpn, head를 모듈별로 불러와서 forward
    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x) # extractor 통과
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale) # rpn 통과
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices) # head 통과
        return roi_cls_locs, roi_scores, rois, roi_indices 

    def use_preset(self): # prediction 과정 쓰이는 threshold 정의
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs,sizes=None):
        """
        이미지에서 객체 검출
        Input : images
        Output : bboxes, labels, scores
        """
        self.eval()
        prepared_imgs = imgs
                
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale) # self = FasterRCNN
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = torch.Tensor(self.loc_normalize_mean).cuda(). repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda(). repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset()
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        '''
        Optimizer 선언
        '''
        lr = learning_rate
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


class FasterRCNNVGG16(FasterRCNN):

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self, n_fg_class=10, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32] ): # n_fg_class : 배경포함 하지 않은 class 개수        
        extractor, classifier = decom_vgg16()
        
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


