import torch.nn as nn

from ..registry import HEADS
from .. import builder

@HEADS.register_module
class ComposedClassAgnosticHead(nn.Module):
    def __init__(self, ca_heads):
        super().__init__()
        self.heads_list = nn.ModuleDict()
        for ca_name, ca_head in ca_heads.items():
            self.heads_list[ca_name] = builder.build_head(ca_head)

    def init_weights(self):
        for _, ca_head in self.heads_list.items():
            ca_head.init_weights()

    def forward(self, **kwargs):
        outs = {}
        for _, ca_head in self.heads_list.items():
            outs.update(ca_head(**kwargs))
        return outs

    def loss(self, **kwargs):
        losses = {}
        for _, ca_head in self.heads_list.items():
            losses.update(ca_head.loss(**kwargs))
        return losses

    def get_seg(self, **kwargs):
        seg_out = [{}]
        for _, ca_head in self.heads_list.items():
            ca_out = ca_head.get_seg(**kwargs)
            if ca_out[0] is not None:
                seg_out[0].update(ca_out[0])
        return seg_out
