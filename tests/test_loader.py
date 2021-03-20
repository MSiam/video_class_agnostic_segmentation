from mmdet.datasets import build_dataset, build_dataloader

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
import cv2
import sys
from mmcv import Config

def vis_masks(img, ref_img, masks, ref_masks, instance_ids, refinst_ids):
    fig2 = plt.figure(4)
    fig2.clf()
    fig1 = plt.figure(3)
    fig1.clf()
    ax1 = fig1.add_subplot(111, aspect='equal')
    current_full_mask = create_full_mask(masks, instance_ids)
    ax1.imshow(current_full_mask)

    if ref_masks is not None:
        ref_full_mask = create_full_mask(ref_masks, refinst_ids)
        ax2 = fig2.add_subplot(111, aspect='equal')
        ax2.imshow(ref_full_mask)


def create_full_mask(masks, ids):
    full_mask = np.zeros((masks.shape[-2], masks.shape[-1]), np.uint8)

    for mask, id_ in zip(masks, ids):
        full_mask[mask==1] = id_
    return full_mask

def vis_boxes(img, ref_img, boxes, ref_boxes, instance_ids, refinst_ids):
    fig2 = plt.figure(2)
    fig2.clf()
    fig1 = plt.figure(1)
    fig1.clf()

    if ref_img is not None:
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.imshow(np.array(ref_img));
        for ref_box, rid in zip(ref_boxes, refinst_ids):
            c = 'r'
            ax1.add_patch(
                 patches.Rectangle(
                     ref_box[:2],
                     ref_box[2]-ref_box[0],
                     ref_box[3]-ref_box[1],
                     fill=False,
                     edgecolor=c
                     ))

    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(np.array(img));
    for box, rid in zip(boxes, instance_ids):
        c = 'r'
        ax2.add_patch(
             patches.Rectangle(
                 box[:2],
                 box[2]-box[0],
                 box[3]-box[1],
                 fill=False,
                 edgecolor=c
                 ))
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--use_rgb', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.test_mode:
        dataset = build_dataset(cfg.data.test, default_args={'test_mode': True})
        dataloader = build_dataloader(dataset, 1, cfg.data.workers_per_gpu, dist=False, shuffle=False)
    else:
        dataset = build_dataset(cfg.data.train, default_args={'test_mode': False})
        dataloader = build_dataloader(dataset, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=False)

    plt.ion()
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot('111', aspect='equal')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot('111', aspect='equal')
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot('111', aspect='equal')

    for it, data in enumerate(dataloader):
        for batch in range(data['img'].data[0].shape[0]):
            if not args.test_mode:
                if 'ref_img' in data and 'ref_masks' in data:
                    ref_img = np.transpose(data['ref_img'].data[0][batch].int(), (1,2,0))
                    ref_masks = data['ref_masks'].data[0][batch]
                    ref_obj_ids = range(1, data['ref_obj_ids'].data[0][batch].shape[0]+1)
                    ref_bboxes = data['ref_bboxes'].data[0][batch]
                    gt_pids = data['gt_pids'].data[0][batch]
                else:
                    ref_img = None
                    ref_masks = None
                    ref_obj_ids = None
                    ref_bboxes = None
                    gt_pids = np.arange(1, data['gt_masks'].data[0][batch].shape[0]+1)

                vis_masks(np.transpose(data['img'].data[0][batch].int(), (1,2,0)),
                          ref_img,
                          data['gt_masks'].data[0][batch],
                          ref_masks,
                          gt_pids,
                          ref_obj_ids)

                if not args.use_rgb:
                    if 'depth' in data:
                        img = data['depth']
                    elif 'flow' in data:
                        img = data['flow']
                    else:
                        img = data['img']
                else:
                    img = data['img']

                vis_boxes(np.transpose(img.data[0][batch].int(), (1,2,0)),
                          ref_img,
                          data['gt_bboxes'].data[0][batch],
                          ref_bboxes,
                          gt_pids,
                          ref_obj_ids)

                assert data['gt_masks'].data[0][batch].shape[0] == data['gt_bboxes'].data[0][batch].shape[0]

                if ref_bboxes is not None:
                    assert data['gt_pids'].data[0][batch].shape[0] == data['gt_bboxes'].data[0][batch].shape[0]
                    assert data['ref_obj_ids'].data[0][batch].shape[0] == data['ref_bboxes'].data[0][batch].shape[0]
                    assert data['ref_masks'].data[0][batch].shape[0] == data['ref_bboxes'].data[0][batch].shape[0]

                plt.draw()
                #plt.waitforbuttonpress()
                plt.pause(0.01)
            else:
                plt.ion()
                for batch in range(cfg.data.imgs_per_gpu):
                    for i in range(len(data['img'][batch])):
                        if i == 0: # Show only one of flipped images
                            if 'ref_img' in data:
                                ref_img = np.transpose(data['ref_img'][i][batch].data, (1,2,0))
                            else:
                                ref_img = None

                            plt.clf()
                            plt.figure(1); plt.imshow(np.transpose(data['img'][i][batch].data, (1,2,0)));
                            if ref_img is not None:
                                plt.figure(2); plt.imshow(ref_img);
                            plt.draw()
                            plt.pause(0.01)
                            #plt.waitforbuttonpress()
