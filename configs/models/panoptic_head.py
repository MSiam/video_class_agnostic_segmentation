panoptic_head=dict(type='SimpleSegHead',
    num_classes=19,
    in_channels=256,
    seg_feats_channel=256,
    stacked_convs=5,
    original_image_size=(1600, 800))

def set_params(num_classes=19):
    panoptic_head['num_classes'] = num_classes
    return panoptic_head
