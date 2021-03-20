"""
Class Agnostic Head Abstract Class
"""

import torch.nn as nn
import torch
from mmcv.cnn import normal_init
import torch.nn.functional as F

from ..utils import ConvModule
from ..utils import merge_fpn
import random
import numpy as np

class AppearanceBasedClassAgnosticAbstract(nn.Module):
    def __init__(self, clustering_type='dbscan', in_channels=256, interm_channels=256,
                 n_convs=7, norm_cfg=None, num_classes=19, merge_fpn=False):

        super().__init__()
        self.clustering_type = clustering_type
        self.stuff_idx = 11
        self.interm_channels = interm_channels
        self.conv_modules = nn.ModuleList()
        self.merge_fpn = merge_fpn
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg

        self.conv_modules = self.init_layers(
            n_convs, in_channels, interm_channels, norm_cfg, self.conv_modules
        )
        self.conv_modules = self.init_weights_module(self.conv_modules)

    def init_layers(self, n_convs, in_channels, interm_channels, norm_cfg, conv_modules):
        for idx in range(n_convs):
            if idx == 0:
                chn = in_channels
            else:
                chn = interm_channels

            conv_modules.append(
                    ConvModule(
                        chn,
                        interm_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None))
        return conv_modules

    def init_weights(self):
        self.conv_modules = self.init_weights_module(self.conv_modules)

    def init_weights_module(self, conv_modules):
        for m in conv_modules:
            normal_init(m.conv, std=0.01)
        return conv_modules

    def forward(self, feats, eval=0):
        if self.merge_fpn:
            feats = merge_fpn(feats)
        else:
            feats = feats[0]

        merged_fpn_feats = feats

        for conv_layer in self.conv_modules:
            feats = conv_layer(feats)

        out = {'class_agnostic_embeddings': feats}
        if not self.training:
            out['merged_fpn_embeddings'] = merged_fpn_feats
        return out

    def loss(self, **kwargs):
        pass

    def get_seg(self, **kwargs):
        pass
