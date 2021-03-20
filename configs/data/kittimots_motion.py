dataset_type = 'MotionDataset'
data_root = 'data/kittimots_moseg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1242, 375), (1242, 343), (1242, 311),
                                   (1242, 279), (1242, 247), (1242, 215)],
         multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'flow', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1242, 375),
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
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_train.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow/',
        pipeline=test_pipeline))
