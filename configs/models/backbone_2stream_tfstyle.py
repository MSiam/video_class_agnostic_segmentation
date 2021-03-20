backbone=dict(
    type='TwoStreamResNetTFStyle',
    layers=[3, 4, 6, 3],
    width_multiplier=1,
    sk_ratio=0,
    out_indices=(0, 1, 2, 3), # C2, C3, C4, C5
    frozen_stages=3
    )

def set_frozen_stages(frozen_stages=1):
    backbone['frozen_stages'] = frozen_stages
    return backbone
