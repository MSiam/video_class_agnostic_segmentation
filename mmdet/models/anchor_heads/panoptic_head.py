import torch.nn.functional as F
import torch.nn as nn
from ..utils import bias_init_with_prob, ConvModule, merge_fpn
from ..registry import HEADS
import torch

@HEADS.register_module
class SimpleSegHead(nn.Module):
    def __init__(self, num_classes, in_channels, seg_feats_channel, stacked_convs, original_image_size,
                 merge_fpn=True):

        super().__init__()
        self.num_classes = num_classes
        self.original_image_size = original_image_size
        self.fcn = nn.ModuleList()
        self.stacked_convs = stacked_convs
        self.merge_fpn = merge_fpn

        chn = in_channels
        for i in range(stacked_convs):
            self.fcn.append(
                ConvModule(
                    chn,
                    seg_feats_channel,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=None,
                    bias=True))
            chn = seg_feats_channel

        self.upsample_conv = nn.Conv2d(chn, chn, 1)
        self.classifier = nn.Conv2d(chn, num_classes, 1)

    def forward(self, x):
        x = merge_fpn(x, average=self.merge_fpn)
        for i in range(self.stacked_convs):
            x = self.fcn[i](x)
        intermediate_feats = x
        x = F.interpolate(x, (x.shape[2]*2, x.shape[3]*2))
        x = F.relu(self.upsample_conv(x))
        x = self.classifier(x)
        x = F.interpolate(x, self.original_image_size[::-1])
        return x, intermediate_feats

    def loss(self, seg_map, gt_semantic_seg):
        #TODO: Add loss using two logits from instance and semantic seg
        gt_semantic_seg_up = F.interpolate(gt_semantic_seg.float(), seg_map.shape[-2:], mode='nearest')
        gt_semantic_seg_up = gt_semantic_seg_up.long().squeeze(1)
        loss_seg = F.cross_entropy(seg_map, gt_semantic_seg_up, ignore_index=255)
        return loss_seg
