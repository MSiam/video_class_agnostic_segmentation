import torch
import torch.nn.functional as F

def merge_fpn(x, average=True):
    max_size = x[0].shape
    merged_fpn = []
    for i, _ in enumerate(x):
        merged_fpn.append(F.interpolate(x[i], max_size[-2:]))
    if average:
        return torch.stack(merged_fpn).mean(dim=0)
    else:
        concat = torch.stack(merged_fpn)
        return concat.permute(1,0,2,3,4).reshape(concat.shape[1], -1, *concat.shape[-2:])

