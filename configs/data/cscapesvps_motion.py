dataset_type = 'MotionDataset'
data_root = 'data/cityscapes_vps/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1024, 512), (1024, 496), (1024, 480),(1024, 464), (1024, 448), (1024, 432)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=(300, 800)),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'flow', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'flow']),
            dict(type='Collect', keys=['img', 'flow']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_train_Annotations.json',
        img_prefix=data_root + 'train/img/',
        flow_prefix=data_root + 'train/flow/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_val_Annotations.json',
        img_prefix=data_root + 'val/img/',
        flow_prefix=data_root + 'val/flow/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_val_Annotations.json',
        img_prefix=data_root + 'val/img/',
        flow_prefix=data_root + 'val/flow/',
        pipeline=test_pipeline))

