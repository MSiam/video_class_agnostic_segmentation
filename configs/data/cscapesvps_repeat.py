dataset_type = 'CityscapesVPSDataset'
data_root = 'data/cityscapes_vps/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadRefImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
        with_seg=True, with_pid=True,
        # Cityscapes specific class mapping
        semantic2label={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
                        10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
                        17:17, 18:18, -1:255, 255:255},),
    dict(type='Resize', img_scale=[(2048, 1024)], keep_ratio=True,
        multiscale_mode='value', ratio_range=(0.8, 1.5)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=(800, 1600)),
    dict(type='Pad', size_divisor=32),
    dict(type='SegResizeFlipCropPadRescale', scale_factor=[1, 0.25]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
            'gt_obj_ids', 'gt_masks', 'gt_semantic_seg',
            'gt_semantic_seg_Nx', 'ref_img', 'ref_bboxes',
            'ref_labels', 'ref_obj_ids', 'ref_masks']),
]
test_pipeline = [
    dict(type='LoadRefImageFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 1024)],
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
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'instances_train_city_vps_rle.json',
            img_prefix=data_root + 'train/img/',
            ref_prefix=data_root + 'train/img/',
            seg_prefix=data_root + 'train/labelmap/',
            pipeline=train_pipeline,
            ref_ann_file=data_root +
            'instances_train_city_vps_rle.json',
            offsets=[-1,+1])),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'instances_val_city_vps_rle.json',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        #'im_all_info_val_city_vps.json',
        'instances_val_city_vps_rle.json',
        #img_prefix=data_root + 'val/img_all/',
        img_prefix=data_root + 'val/img/',
        ref_prefix=data_root + 'val/img/',
        #nframes_span_test=30,
        nframes_span_test=6,
        pipeline=test_pipeline))
