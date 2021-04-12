panoptic_head=dict(type='SimpleSegHead',
    num_classes=19,
    in_channels=256,
    seg_feats_channel=256,
    stacked_convs=5,
    original_image_size=(1600, 800),
    merge_fpn=True)

def set_params(num_classes=19, merge_fpn=True, stacked_convs=5, in_channels=256):
    panoptic_head['num_classes'] = num_classes
    panoptic_head['merge_fpn'] = merge_fpn
    panoptic_head['stacked_convs'] = stacked_convs
    panoptic_head['in_channels'] = in_channels
    return panoptic_head
