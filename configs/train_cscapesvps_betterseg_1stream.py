from configs.models.backbone_1stream import backbone
from configs.models.neck import neck
from configs.models.bbox_head import set_num_classes
from configs.models.panoptic_head import set_params as set_params_panoptic
from configs.experiments.general import *
from configs.data.cscapesvps_repeat import *


# model settings
bbox_head = set_num_classes(num_classes=9)
panoptic_head = set_params_panoptic(merge_fpn=False, stacked_convs=2)

# model settings
model = dict(
    type='SOLO',
    pretrained='torchvision://resnet50',
    backbone=backbone,
    neck=neck,
    panoptic_head=panoptic_head,
    bbox_head=bbox_head,
    )

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train= data['train'],
    val=data['val'],
    test=data['test'],)

train_cfg.update({'train_inst_seg': True, 'train_panoptic': True, 'endtoend': True})


# optimizer
total_epochs = 8
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[6,])
checkpoint_config = dict(interval=1)

# yapf:enable
work_dir = './work_dirs/betterseg_1stream/'
pretrain_weights = './trained_models/latest_multiscale.pth'
convert_dict={'backbone':'backbone'}
ignore_clf = False
same_nclasses = True
freeze_vars={'backbone':True}
