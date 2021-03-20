import argparse
import os
import os.path as osp
import shutil
import tempfile
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import cv2
import pickle
import numpy as np
import matplotlib.cm as cm
from mmdet.core import vis_seg, get_color_map
import matplotlib.pyplot as plt

def single_gpu_test(model, data_loader, args, cfg=None, verbose=True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    class_num = 1000 # ins
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(class_num)]
    seg_color_map = get_color_map('cityscapes')

    for color in colors: # remove instance colors overlapping with segmentaiton colormap
        if color in list(seg_color_map.values()):
            colors.remove(color)

    seglabels_ignore = range(11, 19)

    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            fname = data['img_meta'][0].data[0][0]['filename']
            seg_result = model(return_loss=False, rescale=True, temporal_consistency=args.consistent, **data)

            for batch_idx, _ in enumerate(seg_result):
                for key, _ in seg_result[batch_idx].items():
                    if key == 'track_ids': # Only first attributes are tensors, rest are np.ndarray
                        continue

                    if len(seg_result[batch_idx][key]) == 0:
                        continue
                    seg_result[batch_idx][key] = seg_result[batch_idx][key].cpu()

        if not os.path.exists(args.save_dir+fname.split('/')[0]):
            os.makedirs(args.save_dir+fname.split('/')[0])

        if verbose:
            mapping = {'KITTIMOTSTrackDataset':'kitti_mots', 'CityscapesVPSDataset':'cityscapesvps', 'MotionDataset': 'cityscapesvps'}
            dtype = mapping[cfg['dataset_type']]
            vis_seg(data, seg_result, cfg.img_norm_cfg, data_id=None, colors=colors,
                    score_thr=args.score_thr, save_dir=args.save_dir+fname,
                    dataset_type=dtype, vis_track=False, segmentation=True, seglabels_colors=seg_color_map,
                    seglabels_ignore=seglabels_ignore, vis_unknown=args.vis_unknown)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--vis_unknown', action='store_true', help='Flag to visualize the tracks')
    parser.add_argument('--consistent', action='store_true', help='Flag to visualize the tracks')
    parser.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
    parser.add_argument('--save_dir', help='dir for saveing visualized images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if type(cfg.data.test) == list:
        for test_data in cfg.data.test:
            test_data.test_mode = True
    else:
        cfg.data.test.test_mode = True

    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args, cfg=cfg)

if __name__ == '__main__':
    main()
