dataset_type = 'MotionDataset'
data_root = 'data/kittimots_moseg/'
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
        with_seg=True, with_pid=True,
        # Cityscapes specific class mapping
        semantic2label={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
                        10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
                        17:17, 18:18, -1:255, 255:255},),
    dict(type='Resize', img_scale=[(1242, 375)], keep_ratio=True),
    dict(type='RandomFlip'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SegResizeFlipCropPadRescale', scale_factor=[1, 0.25]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'flow', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadRefImageFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1242, 375)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'ref_img']),
            dict(type='Collect', keys=['img', 'ref_img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_train_3classes_Annotations.json',
        img_prefix=data_root + 'JPEGImages_480/',
        flow_prefix=data_root + 'OpticalFlow_480/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val_3classes_Annotations.json',
        img_prefix=data_root + 'JPEGImages_480/',
        flow_prefix=data_root + 'OpticalFlow_480/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val_3classes_Annotations.json',
        img_prefix=data_root + 'JPEGImages_480/',
        flow_prefix=data_root + 'OpticalFlow_480/',
        pipeline=test_pipeline))
