# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
train_json = ''
val_json = ''
test_json = ''

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[123.651, 117.40, 110.08], std=[54.035, 53.37, 54.784], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Mosaic', img_scale=(1024, 1024), center_ratio_range=(0.5, 1.5), min_bbox_size=0, pad_val=114),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + train_json,
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=(1024, 1024))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + val_json,
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_json,
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')