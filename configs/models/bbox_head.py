bbox_head=dict(
    type='DecoupledSOLOHead',
    num_classes=9,
    in_channels=256,
    stacked_convs=7,
    seg_feat_channels=256,
    strides=[8, 8, 16, 32, 32],
    scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
    sigma=0.2,
    num_grids=[80, 72, 64, 48, 32],
    cate_down_pos=0,
    with_deform=False,
    loss_ins=dict(
        type='DiceLoss',
        use_sigmoid=True,
        loss_weight=3.0),
    loss_cate=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
)

def set_num_classes(num_classes):
    bbox_head['num_classes'] = num_classes
    return bbox_head
