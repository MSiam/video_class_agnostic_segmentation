ca_head=dict(type='MahalanobisAppearanceBasedClassAgnosticHead',
    n_convs=4,
    clustering_type='dbscan',
    norm_cfg = dict(type='GN', num_groups=32, requires_grad=True),
    num_classes=19,
    interm_channels=256,
    merge_fpn=True
)

def set_params(num_classes, ca_label, merge_fpn=True, merge_average=True):
    ca_head['num_classes'] = num_classes
    ca_head['ca_label'] = ca_label
    ca_head['merge_fpn'] = merge_fpn
    ca_head['merge_average'] = merge_average
    return ca_head
