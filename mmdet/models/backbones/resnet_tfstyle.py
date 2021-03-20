"""
ResNet TensorFlow Style to enable loading of pretrained SimCLR Weights
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON), nn.ReLU() if relu else nn.Identity())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2, bias=bias)


class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(conv(in_channels, 2 * out_channels, stride=stride),
                                       BatchNormRelu(2 * out_channels))
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(conv(out_channels, mid_dim, kernel_size=1), BatchNormRelu(mid_dim),
                                         conv(mid_dim, 2 * out_channels, kernel_size=1))

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                                          # kernel_size = 2 => padding = 1
                                          nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                                          conv(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [conv(in_channels, out_channels, kernel_size=1), BatchNormRelu(out_channels)]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList([BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


@BACKBONES.register_module
class ResNetTFStyle(nn.Module):
    def __init__(self, layers, width_multiplier, sk_ratio, out_indices, frozen_stages):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self.net[0](x)

        outs = []
        for i in range(len(self.net)-1):
            x = self.net[i+1](x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for stage_idx in range(self.frozen_stages+1):
            for param in self.net[stage_idx].parameters():
                param.requires_grad = False

@BACKBONES.register_module
class TwoStreamResNetTFStyle(nn.Module):
    def __init__(self, layers, width_multiplier, sk_ratio, out_indices, frozen_stages):
        super(TwoStreamResNetTFStyle, self).__init__()
        self.appearance_stream = ResNetTFStyle(layers, width_multiplier, sk_ratio, out_indices, frozen_stages)
        self.motion_stream = ResNetTFStyle(layers, width_multiplier, sk_ratio, out_indices, frozen_stages)
        self.projection = nn.ModuleList([nn.Conv2d(512, 256, 1),
                                         nn.Conv2d(1024, 512, 1),
                                         nn.Conv2d(2048, 1024, 1),
                                         nn.Conv2d(4096, 2048, 1)])

        self._freeze_stages()

    def _freeze_stages(self):
        self.appearance_stream._freeze_stages()
        self.motion_stream._freeze_stages()

    def init_weights(self, pretrained=None):
        self.appearance_stream.init_weights(pretrained=pretrained)
        self.motion_stream.init_weights(pretrained=pretrained)

    def sensor_fusion(self, appearance_features, motion_features, level_idx):
        fused = torch.cat((appearance_features, motion_features), dim=1)
        return fused

    def forward(self, x, x_second_modality=None):
        app_x = self.appearance_stream(x)
        if x_second_modality is not None:
            second_modality_x = self.motion_stream(x_second_modality)
            concat_feats = []
            for level_idx in range(len(app_x)):
                concat_feats.append(self.sensor_fusion(app_x[level_idx], second_modality_x[level_idx], level_idx))
                concat_feats[-1] = self.projection[level_idx](concat_feats[-1])
        else:
            concat_feats = None
        return concat_feats, app_x

class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


def get_resnet(depth=50, width_multiplier=1, sk_ratio=0):  # sk_ratio=0.0625 is recommended
    layers = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}[depth]
    resnet = ResNet(layers, width_multiplier, sk_ratio)
    return resnet, ContrastiveHead(resnet.channels_out)


def name_to_params(checkpoint):
    sk_ratio = 0.0625 if '_sk1' in checkpoint else 0
    if 'r50_' in checkpoint:
        depth = 50
    elif 'r101_' in checkpoint:
        depth = 101
    elif 'r152_' in checkpoint:
        depth = 152
    else:
        raise NotImplementedError

    if '_1x_' in checkpoint:
        width = 1
    elif '_2x_' in checkpoint:
        width = 2
    elif '_3x_' in checkpoint:
        width = 3
    else:
        raise NotImplementedError

    return depth, width, sk_ratio
