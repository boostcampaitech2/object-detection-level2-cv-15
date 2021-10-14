from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import os
import random

import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import json
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict

# faster rcnn model이 포함된 library
import torchvision

def make_json(fold_ind, dev_id_json, val_id_json, annotation_foldDir):
    
    train_path = "/opt/ml/detection/dataset/train.json"
    #train_file_name = f"../../dataset/fold{fold_ind}_train.json"
    train_file_name = annotation_foldDir[0]
    #valid_file_name = f"../../dataset/fold{fold_ind}_valid.json"
    valid_file_name = annotation_foldDir[1]
    
    # print(fold_ind)
    # print(dev_id_json[0])
    # print(val_id_json[0])
    with open(train_path, 'r') as f:
        json_datas = json.load(f) # python dict 처럼 접근하게끔 변환
    	#dict_keys(['info', 'licenses', 'images', 'categories', 'annotations'])
    # category = {}
    # file_info = {}

    info = json_datas["info"]
    licenses = json_datas["licenses"]
    categories = json_datas["licenses"]

    images_train = []
    images_valid = []

    annotations_train = []
    annotations_valid = []

    for item in json_datas["images"]:
        if item["id"] in dev_id_json:
            images_train.append(item)
        elif item["id"] in val_id_json:
            images_valid.append(item)
        else:
            print("no id in images")


    for item in json_datas["annotations"]:
        if item["image_id"] in dev_id_json:
            annotations_train.append(item)
        elif item["image_id"] in val_id_json:
            annotations_valid.append(item)
        else:
            print("no id in annotations")

    make_json_train = defaultdict(list)
    make_json_valid = defaultdict(list)
    make_json_train = {"info":info, "licenses" : licenses, "images": images_train, "categories":categories, "annotations":annotations_train}
    make_json_valid = {"info":info, "licenses" : licenses, "images": images_valid, "categories":categories, "annotations":annotations_valid}

    # 개수 파악
    if len(json_datas["images"]) != len(make_json_train["images"]) + len(make_json_valid["images"]):
        print("images length, diff")

    if len(json_datas["annotations"]) != len(make_json_train["annotations"]) + len(make_json_valid["annotations"]):
        print("annotations length, diff")

    # # 이름 중복 파악 - > 모든 성분 파악
    # for item in make_json_train["images"] (dict):
    #     if item["file_name"] in make_json_valid["images"]:
    #         print("same file name exist")
    
    # for item in make_json_train["annotations"]:
    #     if item["file_name"] in make_json_valid["annotations"]:
    #         print("same file name exist")

    # assert len(set(make_json_train["annotations"]) & set(make_json_valid["annotations"])) == 0

    # print(make_json)
    
    with open(train_file_name, 'w') as output:
        json.dump(make_json_train, output, indent=2)

    with open(valid_file_name, 'w') as output:
        json.dump(make_json_valid, output, indent=2)
    # train, valid 개수랑 겹치는 부분

# hyuns 10/11
def stratified_group_k_fold(X, y, groups, k, seed = None):

    #stratified_group_k_fold(train_x, train_y, groups, k=5)

    labels_num = np.max(y) + 1 # class num ()
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num)) # 그룹마다 클래스의 수 분포를 파악
    y_distr = Counter() # 모든 라벨의 개수를 세어서 dict 형태로 반환 

    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1 # 그룹마다 라벨의 위치에 클래스 개수를 늘려준다.
        y_distr[label] += 1 # 총 라벨의 개수 증가
    #print(y_distr)

    #print(y_counts_per_group["train/4882.jpg"])
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num)) # fold마다 클래스의 수 분포를 파악
    groups_per_fold = defaultdict(set) # fold별 group 만들기

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts # fold 하나에 Image label이 더해집니다. (계산하기 위한 용도?, Numpy는 list끼리 더하기가 가능하다.
        std_per_label = []
        #print(y_counts_per_fold[fold])
        for label in range(labels_num): # 10개
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]) # 모든 fold에 있는 라벨들에 대해 비율을 구하고 그 표준편차를 구한다.
            std_per_label.append(label_std) # 라벨 당 표준편차들을 구합니다.
            
        y_counts_per_fold[fold] -= y_counts # fold당 모든 이미지의 라벨을 다시 뺴줍니다.
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items()) # list화  [file_name, label_list]
    #random.Random(seed).shuffle(groups_and_y_counts) # list random shuffle 섞을 필요가 있나? 밑에서 sort를 하는데..?

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])): # image당 라벨 개수들의 분포에 -표준편차를 기준으로 sort
    #for g, y_counts in groups_and_y_counts:
        best_fold = None
        min_eval = None
        for i in range(k): # k: fold 개수
            fold_eval = eval_y_counts_per_fold(y_counts, i) # group이 가지고 있는 라벨에서 / fold 별로 각 label의 표준 편차를 구하고 / 그룹에 있는 라벨들의 표준 편차의 평균을 구한다.
            if min_eval is None or fold_eval < min_eval: # 라벨들의 표준편차 그리고 그에 평균 값이 가장 적은 곳에 best_fold를 해준다.
                min_eval = fold_eval
                best_fold = i

        # 결국 라벨들의 분포를 골고루 넣어주기 위한 과정

        y_counts_per_fold[best_fold] += y_counts  # 폴드 당 라벨들의 합이 들어있따.
        groups_per_fold[best_fold].add(g) # 폴드당 이미지들을 다 넣어줌

    all_groups = set(groups) # 중복 그룹 제거
    for i in range(k): # fold = 0, 1, 2, 3, 4 일 때,train/test group 만들기
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


# def get_train_transform():
#     return A.Compose([
#             A.Resize(1024,1024),
#             A.Blur(blur_limit=50, p=0.1),
#             #A.MedianBlur(blur_limit=51, p=0.1),
#             A.ToGray(p=0.3),
#             A.Flip(p=0.5),
#             ToTensorV2(p=1.0)],
#             bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''
    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms
    def __getitem__(self, index: int):
        #image_id = self.coco.getImgIds(imgIds=index) # getImgIds 는 Image의 아이디를 따오는 게 아니었다.. 그냥 입력값을 리스트로만,,
        image_id = self.predictions["images"][index]["id"]
        image_id = [image_id]

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정 
        labels = np.array([x['category_id']+1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
                                
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())


class CustomTestDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())

