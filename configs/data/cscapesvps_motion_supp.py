from configs.data.cscapesvps_motion import *

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_train_Annotations.json',
        img_prefix=data_root + 'train/img/',
        flow_prefix=data_root + 'train/flow_suppressed/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_val_Annotations.json',
        img_prefix=data_root + 'val/img/',
        flow_prefix=data_root + 'val/flow_suppressed/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'CityscapesVPS_MOSeg_val_Annotations.json',
        img_prefix=data_root + 'val/img/',
        flow_prefix=data_root + 'val/flow_suppressed/',
        pipeline=test_pipeline))

