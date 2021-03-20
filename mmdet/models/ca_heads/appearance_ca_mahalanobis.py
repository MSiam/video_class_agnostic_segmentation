"""
Class Agnostic Head Mahalanobis Based Segmentation
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
from mmdet.models.ca_heads.appearance_ca_abstract import AppearanceBasedClassAgnosticAbstract


@HEADS.register_module
class MahalanobisAppearanceBasedClassAgnosticHead(AppearanceBasedClassAgnosticAbstract):
    def __init__(self, clustering_type='dbscan', in_channels=256, interm_channels=256, n_convs=7,
                 norm_cfg=None, merge_fpn=True, num_classes=19, use_norm=False, ca_label=254, upsample=True,
                 merge_average=True):
        super().__init__(clustering_type=clustering_type, in_channels=in_channels, interm_channels=interm_channels,
                         n_convs=n_convs, norm_cfg=norm_cfg, num_classes=num_classes, merge_fpn=merge_fpn)
        self.ca_label = ca_label
        self.use_norm = use_norm
        self.upsample = upsample
        self.merge_average = merge_average
        self.init_mahalanobis_parameters()
        if not self.merge_average:
            self.fpn_conv = ConvModule(
                                1280,
                                256,
                                3,
                                stride=1,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                bias=False)


    def init_mahalanobis_parameters(self):
        self.means = nn.Parameter(torch.zeros((self.num_classes, self.interm_channels)), requires_grad=True)
        nn.init.normal_(self.means, mean=0, std=0.01)
        self.sigmas = nn.Parameter(torch.zeros((self.num_classes, 1)), requires_grad=True)
        nn.init.normal_(self.sigmas, mean=0, std=0.01)
        self.global_unknown = nn.Parameter(torch.zeros((1, 1)), requires_grad=True)
        nn.init.normal_(self.global_unknown, mean=0, std=0.01)

    def forward(self, feats, eval=0):
        if self.merge_fpn:
            feats = merge_fpn(feats, average=self.merge_average)
            if not self.merge_average:
                feats = self.fpn_conv(feats)
        else:
            feats = feats[0]

        if self.upsample:
            feats = F.interpolate(feats, (feats.shape[-2]*2, feats.shape[-1]*2),
                                  mode='bilinear', align_corners=True)

        if self.use_norm:
            feats = F.normalize(feats, dim=1)

        merged_fpn_feats = feats

        for conv_layer in self.conv_modules:
            feats = conv_layer(feats)

        out = {'class_agnostic_embeddings': feats,
               'merged_fpn_embeddings': merged_fpn_feats}
        return out

    def compute_preds(self, means, sigmas, global_unknown, embeddings):
        eps = 1e-10
        sigmas = sigmas.view(1, self.num_classes, 1, 1)
        means = means.view(1, self.num_classes, self.interm_channels, 1, 1)
        embeddings = embeddings.unsqueeze(1)

        if self.use_norm:
            means = F.normalize(means, dim=2)
            embeddings = F.normalize(embeddings, dim=2)
            cos_sim = (embeddings * means).sum(dim=2)
            preds = -1 * cos_sim / (2 * sigmas**2 + eps)
        else:
            mean_diff = means - embeddings
            preds = -1 * torch.norm(mean_diff, dim=2) / (2 * sigmas**2 + eps)

        preds_unknown = global_unknown.repeat(preds.shape[0], 1, *preds.shape[-2:])
        if self.ca_label != -1:
            preds = torch.cat((preds, preds_unknown), dim=1)
        return preds

    def loss(self, **kwargs):
        losses = {}
        x = kwargs['class_agnostic_embeddings']
        target_seg = kwargs['gt_semantic_seg']

        loss = torch.tensor([0.0]).cuda()

        target_seg = target_seg.data[:, 0]
        if self.ca_label != -1:
            target_seg[target_seg==self.ca_label] = self.num_classes

        preds = self.compute_preds(
            self.means, self.sigmas, self.global_unknown, x
        )
        preds = F.interpolate(preds, target_seg.shape[-2:])

        losses['mahalanobis_loss'] = F.cross_entropy(preds, target_seg.long(), ignore_index=255)
        return losses


    def get_seg(self, **kwargs):

        x = kwargs['class_agnostic_embeddings']
        preds = self.compute_preds(
                self.means, self.sigmas, self.global_unknown, x
            )
        preds = torch.argmax(preds, dim=1).unsqueeze(1)

        return [{'mahalanobis_seg': preds}]
