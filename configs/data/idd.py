dataset_type = 'MotionDataset'
data_root = 'data/debug_ca_idd_data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadFlowFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1280, 720)],
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
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'IDD_Test.json',
        img_prefix=data_root+'/JPEGImages_480/',
        flow_prefix=data_root+'OpticalFlow_480_0/',
        pipeline=test_pipeline))

