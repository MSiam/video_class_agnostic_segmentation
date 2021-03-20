
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
#### upsnet libraries
from tools.dataset import *
import pickle
import cv2
from mmdet.core.utils.misc import compute_panoptic, convert_and_load_checkpoint
from tools.dataset.base_dataset import PQStatCat
from mmdet.core.utils.misc import compute_mask_ious, process_gt_masks


def evaluate_uq(seg_result, gt_masks, score_thr):
    frame_pq_stat = PQStatCat()

    # Choose Label 0: Moving
    valid_inds = seg_result[0]['labels'] == 0
    masks = seg_result[0]['masks'][valid_inds]
    scores = seg_result[0]['scores'][valid_inds]

    # Choose ones go above score thr
    valid_inds = scores > score_thr
    masks = masks[valid_inds]

    if len(masks) != 0 and len(gt_masks) != 0:
        ious_matrix = compute_mask_ious(masks, gt_masks).cpu()
        ious, indices = ious_matrix.max(dim=1)
    else:
        ious = []
        indices = []

    for iou in ious:
        if iou > 0.5:
            frame_pq_stat.iou += iou
            frame_pq_stat.tp += 1
        else:
            frame_pq_stat.fp += 1
            frame_pq_stat.fn += 1

    for iteration, gt_mask in enumerate(gt_masks):
        if iteration not in indices:
            frame_pq_stat.fn += 1

    return frame_pq_stat

def evaluate_miou(seg_result, gt_masks, score_thr):
    # Choose Label 0: Moving
    valid_inds = seg_result[0]['labels'] == 0
    masks = seg_result[0]['masks'][valid_inds]
    scores = seg_result[0]['scores'][valid_inds]

    # Choose score > thr
    valid_inds = scores > score_thr
    masks = masks[valid_inds]

    if len(masks) == 0 and len(gt_masks) == 0:
        return np.array([0.0, 0.0])

    if len(masks) != 0:
        mask_shape = masks[0].shape
    elif len(gt_masks) != 0:
        mask_shape = gt_masks[0].shape

    full_pred = np.zeros(mask_shape)
    full_gt = np.zeros(mask_shape)

    for mask in masks:
        full_pred[mask==1] = 1

    for gt_mask in gt_masks:
        full_gt[gt_mask==1] = 1

    U = np.logical_or(full_gt, full_pred).sum()
    iou_fg = np.logical_and(full_gt, full_pred).sum()/ float(U)

    full_gt = 1 - full_gt
    full_pred = 1 - full_pred

    U = np.logical_or(full_gt, full_pred).sum()
    iou_bg = np.logical_and(full_gt, full_pred).sum()/ float(U)

    return np.asarray([iou_bg, iou_fg])

def single_gpu_test(model, data_loader, show=False, score_thr=0.3, cfg=None):
    model.eval()
    results = []
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    output_panos = {}
    accumulated_pq_stat = PQStatCat()

    ious = np.zeros([1, 2])
    count = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            idx = data['img_meta'][0].data[0][0]['idx']
            img_h, img_w = data['img_meta'][0].data[0][0]['img_shape'][:2]
            ann_info = data_loader.dataset.get_ann_info(idx)

            valid_unknown_indices = np.where(ann_info['labels'] == 1)[0]
            gt_masks = []
            for idx in valid_unknown_indices:
                gt_masks.append(ann_info['masks'][idx])

            seg_result = model(return_loss=False, rescale=True, detect_only=True, **data)

            for batch_idx, _ in enumerate(seg_result):
                for key, value in seg_result[batch_idx].items():
                    if key == 'track_ids': # Only first attributes are tensors, rest are np.ndarray
                        continue

                    if len(seg_result[batch_idx][key]) == 0:
                        continue

                    if type(seg_result[batch_idx][key]) == torch.Tensor:
                        seg_result[batch_idx][key] = seg_result[batch_idx][key].cpu().numpy()

            seg_result_ca = [{}]
            for key, value in seg_result[0].items():
                if 'ca' in key:
                    seg_result_ca[0][key[3:]] = seg_result[0][key]

            gt_masks = process_gt_masks(gt_masks, img_h, img_w)
            pq_stat = evaluate_uq(seg_result_ca, gt_masks, score_thr)
            accumulated_pq_stat += pq_stat

            ious += evaluate_miou(seg_result_ca, gt_masks, score_thr)
            count += 1
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    sq = accumulated_pq_stat.iou / accumulated_pq_stat.tp
    rq = accumulated_pq_stat.tp / (accumulated_pq_stat.tp + 0.5 * accumulated_pq_stat.fp + 0.5 * accumulated_pq_stat.fn)
    uq = sq * rq
    ious = ious / count
    return sq, rq, uq, ious

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
    parser.add_argument('--gpus', type=str, default='0' )
    args, rest = parser.parse_known_args()
    #### update config
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gpus = [int(_) for _ in args.gpus.split(',')]
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backedns.cudnn.benchmark = True
    cfg.model.pretrained = None
    if type(cfg.data.test) == list:
        for test_data in cfg.data.test:
            test_data.test_mode = True
    else:
        cfg.data.test.test_mode = True

    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model,
                           train_cfg=None,
                           test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if model.bbox_head is None:
        # Convert bbox_head to ca_head if needed first
        model, checkpoint = convert_and_load_checkpoint(model, args.checkpoint)
    else:
        checkpoint = load_checkpoint(model,
                                     args.checkpoint,
                                     map_location='cpu')

    # E.g., Cityscapes has 8 things CLASSES.
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[gpus[0]])
    # args.show = False
    sq, rq, uq, ious = single_gpu_test(model, data_loader, score_thr=args.score_thr, cfg=cfg)

    # EVAL: IMAGE Class Agnostic Segmentation SEGMENTATION
    # *******************************************
    print("==> Image Uknown Quality is ")
    print ('Segmentation Quality ', sq)
    print ('Recognition Quality ', rq)
    print ('Class Agnostic Segmentation Quality', uq)

    print('IoUs: [Bg, Fg] ', ious)
    print('IoUs: mean ', ious.mean())

if __name__ == '__main__':
    main()
