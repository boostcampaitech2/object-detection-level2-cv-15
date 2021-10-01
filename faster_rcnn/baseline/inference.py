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


def eval(dataloader, faster_rcnn):
    outputs = []
    for ii, (imgs, sizes) in enumerate(tqdm(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        for out in range(len(pred_bboxes_)):
            outputs.append({'boxes':pred_bboxes_[out], 'scores': pred_scores_[out], 'labels': pred_labels_[out]})
    
    return outputs

def inference():

    # Test dataset 불러오기
#     testset = TestDataset()
    annotation = os.path.join(data_dir,'test.json')
    testset = TestCustom(annotation, data_dir)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1, # only batch_size=1 support
                                       num_workers=4,
                                       shuffle=False, 
                                       pin_memory=False
                                       )
    # faster rcnn 불러오기
    faster_rcnn = FasterRCNNVGG16().cuda()
    state_dict = torch.load(inf_load_path)
    if 'model' in state_dict:
        faster_rcnn.load_state_dict(state_dict['model'])
    print('load pretrained model from %s' % inf_load_path)

    # evaluation
    outputs = eval(test_dataloader, faster_rcnn)
    score_threshold = 0.05
    prediction_strings = []
    file_names = []
    
    # submission file 작성
    coco = COCO(os.path.join(data_dir, 'test.json'))
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[1]) + ' ' + str(
                    box[0]) + ' ' + str(box[3]) + ' ' + str(box[2]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv("./faster_rcnn_scratch_submission.csv", index=False)
    
    print(submission.head())        

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # Data and model checkpoints directories
#     parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
#     parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
#     parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

#     # Container environment
#     parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
#     parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
#     parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

#     args = parser.parse_args()

#     data_dir = args.data_dir
#     model_dir = args.model_dir
#     output_dir = args.output_dir

#     os.makedirs(output_dir, exist_ok=True)

#     inference(data_dir, model_dir, output_dir, args)
