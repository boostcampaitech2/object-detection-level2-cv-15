def get_model(config):
    if config.model_name == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif config.model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    elif config.model_name == 'mobilenet_v3_large_320_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    elif config.model_name == 'retinanet_resnet50_fpn':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)     
    elif config.model_name == 'ssd300_vgg16':
        model = torchvision.models.detection.ssd300_vgg16(pretrained=True)     
    elif config.model_name == 'maskrcnn_resnet50_fpn':
        model =torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)     
    elif config.model_name == 'keypointrcnn_resnet50_fpn':
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)   
    return model