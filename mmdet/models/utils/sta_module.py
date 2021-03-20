import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Aggregation

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
        self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        assert self.stride == 1, 'stride > 1 not implemented'
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x_ref, x_current):

        x1, x2, x3 = self.conv1(x_current), self.conv2(x_ref), self.conv3(x_current)
        x1 = x1.view(x_ref.shape[0], -1, 1, x_ref.shape[2]*x_ref.shape[3])
        x2 = self.unfold_j(self.pad(x2)).view(x_ref.shape[0], -1, 1, x1.shape[-1])
        # Refer to equation 5, R(i): 7x7, delta: concatenation, gamma: conv_w,
        w = self.conv_w(torch.cat([x1, x2], 1)).view(x_ref.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x


class STABottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes=8, kernel_size=3, stride=1):
        super(STABottleneck, self).__init__()
        self.sam = SAM(sa_type=sa_type, in_planes=in_planes, rel_planes=rel_planes,
                       out_planes=out_planes, share_planes=share_planes, kernel_size=kernel_size,
                       stride=stride)
        self.stride = stride

    def forward_single(self, x_ref, x_current):
       out = self.sam(x_ref, x_current)
       return out

    def forward(self, x_ref, x_current):
        out = []
        for xr, xc in zip(x_ref, x_current):
            out.append(self.forward_single(xr, xc))
        return out


