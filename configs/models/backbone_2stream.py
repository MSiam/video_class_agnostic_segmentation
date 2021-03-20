backbone=dict(
    type='TwoStreamResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
    frozen_stages=1,
    style='pytorch')
