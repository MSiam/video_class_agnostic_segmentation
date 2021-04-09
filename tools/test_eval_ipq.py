
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
from tools.dataset.cityscapes_vps import CityscapesVps
import cv2
from mmdet.core.utils.misc import compute_panoptic

def correct_annotations(labels):
    mapping = {0:0, 1:1, 2:2, 3:3, 4:7, 5:6, 6:4, 7:5}
    for k, v in mapping.items():
        labels[labels==k] = v
    return labels

def single_gpu_test(model, data_loader, show=False, score_thr=0.3, img_scale=None, flip_annotations=False):
    model.eval()
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    output_panos = {}
    output_panos['all_ssegs'] = []
    output_panos['all_panos'] = []
    output_panos['all_pano_cls_inds'] = []
    output_panos['all_pano_obj_ids'] = []
    output_panos['all_names'] = []

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            fname = data['img_meta'][0].data[0][0]['filename']
            seg_result = model(return_loss=False, rescale=True, **data)

            for batch_idx, _ in enumerate(seg_result):
                for key, _ in seg_result[batch_idx].items():
                    if len(seg_result[batch_idx][key]) == 0:
                        continue
                    seg_result[batch_idx][key] = seg_result[batch_idx][key].cpu()

        if len(seg_result[0].keys()) > 3 and len(seg_result[0]['masks']) != 0:
            segmentation = cv2.resize(np.array(seg_result[0]['segmentation'], np.uint8), img_scale,
                                      interpolation=cv2.INTER_NEAREST)
            output_panos['all_ssegs'].append(segmentation)
            segmentation = np.asarray(segmentation, dtype=np.uint8)
            valid_inds = seg_result[0]['scores'] > score_thr
            output_panos['all_panos'].append(compute_panoptic(seg_result[0]['masks'][valid_inds], segmentation))
            # get_unified_... has last_id_stuff = 10 so in order to get Car: 2 to be become 13 need to add + 1
            output_panos['all_pano_cls_inds'].append(seg_result[0]['labels'][valid_inds]+1)
            output_panos['all_pano_obj_ids'].append(None)
            output_panos['all_names'].append(fname.split('/')[-1])
        else:
            # Handling Empty Instance Masks
            segmentation = cv2.resize(np.array(seg_result[0]['segmentation'], np.uint8), img_scale,
                                      interpolation=cv2.INTER_NEAREST)
            output_panos['all_ssegs'].append(segmentation)
            output_panos['all_panos'].append(segmentation)
            output_panos['all_pano_cls_inds'].append([])
            output_panos['all_pano_obj_ids'].append([])
            output_panos['all_names'].append(fname.split('/')[-1])

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return output_panos


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector PQ')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--panoptic_json_file', type=str, default='data/cityscapes_vps/panoptic_gt_val_city_vps.json')
    parser.add_argument('--panoptic_gt_folder', type=str, default='data/cityscapes_vps/val/panoptic_video/')
    parser.add_argument('--score_thr', type=float, default=0.3, help='score threshold for visualization')
    parser.add_argument('--flip_annotations', action='store_true')
    parser.add_argument('--gpus', type=str, default='0' )
    args, rest = parser.parse_known_args()
    #### update config
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    gpus = [int(_) for _ in args.gpus.split(',')]
    if args.out is not None and not args.out.endswith(('.pkl', 'pickle')):
        raise ValueError("The output file must be a .pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backedns.cudnn.benchmark = True
    cfg.model.pretrained = None
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
    checkpoint = load_checkpoint(model,
                                 args.checkpoint,
                                 map_location='cpu')
    # E.g., Cityscapes has 8 things CLASSES.
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # If .pkl and _pano.pkl results are saved already, load = True.
    model = MMDataParallel(model, device_ids=[gpus[0]])
    # args.show = False


    for element in cfg['data']['test']['pipeline']:
        if element['type'] == 'MultiScaleFlipAug':
            img_scale = element['img_scale']

    outputs_pano = single_gpu_test(model, data_loader, score_thr=args.score_thr,
                                   img_scale=img_scale,
                                   flip_annotations=args.flip_annotations)

    eval_helper_dataset = CityscapesVps(panoptic_json_file=args.panoptic_json_file, panoptic_gt_folder=args.panoptic_gt_folder)

    # EVAL: IMAGE PANOPTIC SEGMENTATION
    # *******************************************
    print("==> Image Panoptic Segmentation PNGs and PQ.TXT will be saved at:")
    print("---", args.out.split('.pkl')[0]+'_pans_unified/')
    # If _pred_pans_2ch.pkl is saved already, load = True.
    # TODO: Use img_scale
    panoptic_stuff_area_limit = img_scale[0]
    pred_pans_2ch_ = eval_helper_dataset.get_unified_pan_result(
        outputs_pano['all_ssegs'],
        outputs_pano['all_panos'],
        outputs_pano['all_pano_cls_inds'],
        stuff_area_limit=panoptic_stuff_area_limit,
        names=outputs_pano['all_names'])

    pred_keys = [_ for _ in pred_pans_2ch_.keys()]
    pred_keys.sort()
    pred_pans_2ch = [pred_pans_2ch_[k] for k in pred_keys]
    del pred_pans_2ch_

    # Evaluate IPQ
    eval_helper_dataset.evaluate_panoptic(
            pred_pans_2ch, args.out.replace('.pkl','_pans_unified'))


if __name__ == '__main__':
    main()
