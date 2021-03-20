"""
Masked Average Pooling Head for Feature Analysis
"""

import torch.nn as nn
import torch
from mmcv.cnn import normal_init
import torch.nn.functional as F

from ..registry import HEADS
from ..utils import ConvModule
from ..utils import merge_fpn
from mmdet.core import compute_ood_scores
import random
import numpy as np
import time


@HEADS.register_module
class MAPClassAgnosticHead(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

    def init_weights(self):
        pass

    def forward(self, feats, eval=0):
        out = {'class_agnostic_embeddings': feats,
               'merged_fpn_embeddings': feats}
        return out

    def get_seg(self, **kwargs):
        return [{}]
