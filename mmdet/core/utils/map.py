import torch.nn.functional as F

def compute_map(mask, x):
    h, w = x.shape[-2:]
    if mask.ndim < 4:
        mask = mask.unsqueeze(1)

    masked_embedding = mask * x.unsqueeze(0)
    area = F.avg_pool2d(mask, x.shape[-2:]) * h * w + 0.0005
    map_embedding = F.avg_pool2d(input=masked_embedding, kernel_size=x.shape[-2:]) * h * w / area
    map_embedding = map_embedding.squeeze()
    if map_embedding.ndim < 2:
        map_embedding = map_embedding.unsqueeze(0)
    return map_embedding

