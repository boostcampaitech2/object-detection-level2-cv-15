_base_ = [
    'model1.py',
    'dataset.py',
    'schedule.py', 'runtime.py'
]

data_root = '/opt/ml/detection/dataset/'
index = 1
train_json = f'train_{index}.json'
val_json = f'val_{index}.json'
test_json = 'test.json'
wandb_runname = f'mmdet_K-Fold_{index}'

data = dict(
    train=dict(ann_file=data_root + train_json),
    val=dict(ann_file=data_root + val_json),
    test=dict(ann_file=data_root + test_json)
    )

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project='garbage_ObjectDetection',
                entity = 'falling90',
                name = wandb_runname
            ),
        )
    ])

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         convert_weights=True,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[96, 192, 384, 768]))