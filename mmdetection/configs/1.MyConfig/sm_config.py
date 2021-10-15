_base_ = [
    'deformable_detr_twostage_refine_r50_16x2_50e_coco.py', #수정
    'dataset.py',
    'sm_schedule_1x.py', 'sm_runtime.py'
]
