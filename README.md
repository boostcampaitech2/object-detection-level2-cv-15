# Object Detection

## Table of Contents

- [Object Detection](#object-detection)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Usage](#usage)
    - [Structure](#structure)
    - [Install](#install)
    - [Getting_Started](#getting_started)
  - [Result](#result)
  - [Reference](#reference)

## Background

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## Usage
### Structure

```sh
|-- convert2Yolo
|-- datasets
|   |-- detection_info.csv
|   |-- test.json
|   `-- train.json
|-- detectron2
|   |-- configs
|   |-- datasets
|   |-- detectron2
|   `-- tools
|-- torchvision
|   `-- baseline
|-- mmdetection
|   |-- configs
|   `-- tools
|-- others
|-- requirements.txt
`-- yolov5
    |-- data
    |-- requirements.txt
    |-- train.py
    `-- val.py
```


### Install

1. torchvision
    ```sh
    $ cd torchvision/baseline
    $ pip install -r requirements.txt
    ```

2. MMDetection
    ```sh
    $ cd mmdetection
    $ conda install pytorch=1.7.1 cudatoolkit=11.0 torchvision -c pytorch
    $ pip install openmim
    $ mim install mmdet
    ```

3. Detectron2
    ```sh
    $ cd mmdetection
    $ python -m pip install -e detectron2
    ```

4. YOLOv5
    ```sh
    $ cd yolov5
    $ pip install -r requirements.txt
    ```

### Getting_Started
1. torchvision
    ```sh
    $ cd torchvision/baseline
    $ python main.py

    [Parameter setting : config.json, main.py (config)]
    ```
    
2. MMDetection
    ```sh
    $ cd mmdetection
    $ python ./tools/train.py [Config File]

    [ex) python ./tools/train.py ./configs/1.MyConfig/sm_config.py]
    ```

3. Detectron2
    ```sh
    $ cd detectron2
    $ python tools/train_net.py --config-file [Config File]

    [ex) python tools/train_net.py --config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml]
    ```

4. YOLOv5
    ```sh
    $ cd yolov5
    $ python train.py [Config File]

    [ex) python train.py —cfg cfg/file/locate —data data/folder/locate —epochs 500 —batch_size 32]
    ```

## Result

<img src="https://github.com/boostcampaitech2/object-detection-level2-cv-15/blob/master/results/img1.png" width="800px" height="400px"></img><br/>

## Reference

<details>
  <summary>MMDetection</summary>

- MMDetection
  ```
  @article{mmdetection,
    title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
    author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
              Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
              Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
              Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
              Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
              and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
    journal= {arXiv preprint arXiv:1906.07155},
    year={2019}
  }
  ```

- DETR
  ```BibTeX
  @inproceedings{detr,
    author    = {Nicolas Carion and
                Francisco Massa and
                Gabriel Synnaeve and
                Nicolas Usunier and
                Alexander Kirillov and
                Sergey Zagoruyko},
    title     = {End-to-End Object Detection with Transformers},
    booktitle = {ECCV},
    year      = {2020}
  }
  ```

- Swin Transformer
  ```latex
  @article{liu2021Swin,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={arXiv preprint arXiv:2103.14030},
    year={2021}
  }
  ```

- MaskRCNN
  ```latex
  @article{He_2017,
    title={Mask R-CNN},
    journal={2017 IEEE International Conference on Computer Vision (ICCV)},
    publisher={IEEE},
    author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
    year={2017},
    month={Oct}
  }
  ```

- YOLOv5 reference [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>

<details>
  <summary>Detectron2</summary>

- Detectron2
  ```BibTeX
  @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {\url{https://github.com/facebookresearch/detectron2}},
    year =         {2019}
  }
  ```

- Cascade R-CNN
  ```latex
  @article{Cai_2019,
    title={Cascade R-CNN: High Quality Object Detection and Instance Segmentation},
    ISSN={1939-3539},
    url={http://dx.doi.org/10.1109/tpami.2019.2956516},
    DOI={10.1109/tpami.2019.2956516},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    publisher={Institute of Electrical and Electronics Engineers (IEEE)},
    author={Cai, Zhaowei and Vasconcelos, Nuno},
    year={2019},
    pages={1–1}
  }
  ```
</details>

<details>
  <summary>faster-rcnn</summary>
  
- Faster-RCNN's reference in [torchvision.models.detection.fasterrcnn_resnet50_fpn](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn)

- RetinaNet's reference in [torchvision.models.detection.retinanet_resnet50_fpn](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html#retinanet_resnet50_fpn)

</details>

