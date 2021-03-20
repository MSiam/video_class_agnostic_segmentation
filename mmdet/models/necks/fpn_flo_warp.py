import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch
import numpy as np
import copy

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from .fpn import FPN

@NECKS.register_module
class FPNFlowWarp(FPN):

    def __init__(self, **kwargs):
        super(FPNFlowWarp, self).__init__(**kwargs)

    @auto_fp16()
    def forward(self, inputs, flow):
        outs = super().forward(inputs)

        # Get even and odd indices
        odd_indices = torch.range(0, outs[0].shape[0]-1) % 2 != 0
        even_indices = torch.range(0, outs[0].shape[0]-1) % 2 == 0

        feat_shape = outs[0][odd_indices].shape[-2:]

        identity_grid = np.meshgrid(np.linspace(-1, 1, feat_shape[1]),
                                    np.linspace(-1, 1, feat_shape[0]))
        identity_grid = torch.tensor(identity_grid).float().cuda()
        identity_grid = identity_grid.permute(1,2,0).unsqueeze(0)

        warped_outs = []
        if flow is not None:
            for level in range(len(outs)):
                # Warp Odd indices , Previous Frame Features
                original_feat_shape = outs[level][odd_indices].shape[-2:]
                warping_flow = F.interpolate(flow[odd_indices],
                                             feat_shape,
                                             mode='bilinear', align_corners=True)
                warping_flow = warping_flow.permute(0, 2, 3, 1)
                warping_flow_normalize = copy.deepcopy(warping_flow)
                warping_flow_normalize[:, :, :, 0] = warping_flow[:, :, :, 0] / feat_shape[1]
                warping_flow_normalize[:, :, :, 1] = warping_flow[:, :, :, 1] / feat_shape[0]


                feats = F.interpolate(outs[level][odd_indices], feat_shape,
                                      mode='bilinear', align_corners=True)

                warped_feats = F.grid_sample(
                    feats, identity_grid - warping_flow_normalize
                )

                warped_feats = F.interpolate(warped_feats, original_feat_shape,
                                                 mode='bilinear', align_corners=True)

                # Construct final feats from warped and even indices features (current ones) as is
                final_warped_feats = torch.zeros(outs[level].shape).cuda()
                final_warped_feats[odd_indices] = warped_feats
                final_warped_feats[even_indices] = outs[level][even_indices]
                warped_outs.append(final_warped_feats)
        else:
            warped_outs = outs

        return tuple(warped_outs)
