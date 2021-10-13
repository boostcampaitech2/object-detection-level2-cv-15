# hyuns
from easydict import EasyDict
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def make_dir(annotation_dir, fold_idx):
    anno_trainName = os.path.splitext(annotation_dir)[0]
    anno_valName = os.path.splitext(annotation_dir)[0].split("/")[:-1]

    anno_trainName = anno_trainName + "_" + str(fold_idx)
    anno_trainExt = os.path.splitext(annotation_dir)[1]
    anno_foldTrainDir = anno_trainName + anno_trainExt

    anno_valName = "/".join(anno_valName) + "/" + "val_" + str(fold_idx)
    anno_foldValDir = anno_valName + anno_trainExt
    
    return anno_foldTrainDir, anno_foldValDir

def make_dataframe(target):
    # target 에 존재하는 train.json 파일을 엽니다.
    with open(target, 'r') as f:
        json_datas = json.load(f) # python dict 처럼 접근하게끔 변환
    	#dict_keys(['info', 'licenses', 'images', 'categories', 'annotations'])

    category = {}
    file_info = {}
    make_frame = defaultdict(list)

    # 이미지 정보 중 파일 경로와 아이디만 추출해서 file_info 에 저장
    for item in json_datas['images']:
        file_info[item['id']] = {'id' : item['id'], 'file_name' : item['file_name']}
    # 카테고리 정보를 category 에 저장
    for item in json_datas['categories']:
        category[item['id']] = item['name']
    # annotations 에 속하는 아이템들을 images 에 속하는 아이템의 정보와 합치기 위함
    for annotation in json_datas['annotations']:
        save_dict = file_info[annotation['image_id']]
        # 각 이미지에 해당하는 bounding box 정보와 class 정보 area(넓이) 정보를 추가
        bbox = np.array(annotation['bbox'])
        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]
        save_dict.update({
            'class': annotation['category_id'], # 배경은 0, 나머지 +1 in faster_rcnn
            'x_min': bbox[0],
            'y_min': bbox[1],
            'x_max': bbox[2],
            'y_max': bbox[3],
            'area':annotation['area']
            })

        for k,v in save_dict.items():
            # dataframe 으로 만들기 위해서 'key' : [item1,item2...] 형태로 저장
            make_frame[k].append(v)

    # dictionary 가 잘 만들어 졌는지 길이를 측정해서 확인해보세요!
    print(len(json_datas['annotations']))
    # dictionary to DataFrame
    df = pd.DataFrame.from_dict(make_frame)
    df.to_csv('./detection_info.csv',index=False)
    #print(df.head())

    return df

def make_valid_dataframe(GT_JSON, outputs, score_threshold):
    '''
    {'boxes': out['boxes'].tolist(),
     'scores': out['scores'].tolist(),
      'labels': out['labels'].tolist()}
    '''
    coco = COCO(GT_JSON)
    prediction_strings = []
    file_names = []
    #score_threshold = 0.05
    #score_threshold = 0.1

    for output in outputs:
        prediction_string = ''
        
        image_info = coco.loadImgs(output["id"])[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    valid_df = pd.DataFrame()
    valid_df["PredictionString"] = prediction_strings
    valid_df["image_id"] = file_names

    return valid_df


class JsonConfigFileManager:
    """Json설정파일을 관리한다"""
    def __init__(self, file_path):
        self.values = EasyDict()
        if file_path:
            self.file_path = file_path # 파일경로 저장
            self.reload()

    def reload(self):
        """설정을 리셋하고 설정파일을 다시 로딩한다"""
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f))

    def clear(self):
        """설정을 리셋한다"""
        self.values.clear()
                
    def update(self, in_dict):
        """기존 설정에 새로운 설정을 업데이트한다(최대 3레벨까지만)"""
        for (k1, v1) in in_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1     
            
    def export(self, save_file_name):
        """설정값을 json파일로 저장한다"""
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f)

    