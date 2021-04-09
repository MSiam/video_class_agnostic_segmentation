from configs.models.backbone_2stream import backbone
from configs.models.neck import neck
from configs.models.bbox_head import set_num_classes
from configs.models.ca_motion_head import set_params
from configs.models.panoptic_head import panoptic_head
from configs.experiments.general import *
from configs.data.cscapesvps_motion_supp_2048 import data as cscapesvps_data
from configs.data.kittimots_motion_supp import data as kittimots_data
from configs.data.cscapesvps_motion_supp_2048 import *


# model settings
bbox_head = set_num_classes(num_classes=9)
ca_head = set_params(num_classes=3)

# model settings
model = dict(
    type='SOLO',
    pretrained='torchvision://resnet50',
    backbone=backbone,
    neck=neck,
    panoptic_head=panoptic_head,
    bbox_head=bbox_head,
    ca_head=ca_head,
    )

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=[kittimots_data['train'], cscapesvps_data['train']],
    val=cscapesvps_data['val'],
    test=cscapesvps_data['test'],)

# optimizer
total_epochs = 15
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[6, 8])
checkpoint_config = dict(interval=5)

# yapf:enable
work_dir = './work_dirs/ca_motion/'
pretrain_weights = './trained_models/panopticseg_cscapesvps.pth'
ignore_clf = False
same_nclasses = True
freeze_vars={'backbone.appearance_stream':True, 'neck':True, 'bbox_head':True, 'panoptic_head':True}
