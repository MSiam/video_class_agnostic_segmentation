from configs.data.kittimots_motion import *

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_train.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow_suppressed/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow_suppressed/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/KITTIMOTS_MOSeg_val.json',
        img_prefix=data_root + 'images/',
        flow_prefix=data_root + 'flow_suppressed/',
        pipeline=test_pipeline))
