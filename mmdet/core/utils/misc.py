from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip
import torch.nn.functional as F
import torch
from ..evaluation.class_names import get_classes
from scipy import ndimage
import cv2
import functools
import pycocotools.mask as maskUtils
import os
import scipy.linalg as la
import time
from mmdet.core.utils.colormap import get_color_map
from mmdet.core.utils.colormap import labels_cityscapes as cscapes_colormap
from PIL import Image

def load_seg_gt(cfg, fname, semantic2label):
    # Load and convert segmentation gt
    img_prefix = cfg['data']['test']['img_prefix']
    seg_prefix = cfg['data']['test']['seg_prefix']
    cfg_test_pipeline = cfg['test_pipeline']
    assert semantic2label is not None

    fname = fname.replace(img_prefix, seg_prefix)
    if 'idd' in cfg['data_root']:
        fname = fname.replace('leftImg8bit', 'gtFine_labelcsTrainIds')
    elif 'cityscapes_ps' in cfg['data_root']:
        fname = fname.replace('leftImg8bit', 'gtFine_labelTrainIds')
    elif 'cityscapes_vps' in cfg['data_root']:
        fname = fname.replace('leftImg8bit', 'gtFine_color').replace('newImg8bit', 'final_mask')

    fname = fname.replace('jpg', 'png')

    seg_gt = np.array(Image.open(fname))
    if len(seg_gt.shape) > 2:
        seg_gt = seg_gt[:, :, 0]
    temp_seg_gt = seg_gt.copy()
    for cls in np.unique(seg_gt):
        temp_seg_gt[seg_gt==cls] = semantic2label[int(cls)]
    seg_gt = temp_seg_gt
    num_classes = cfg['model']['ca_head']['num_classes']
    seg_gt[seg_gt==254] = num_classes

    for test_pipeline in cfg_test_pipeline:
        for key, value in test_pipeline.items():
            if 'img_scale' in key:
                scale = value[0]
    seg_gt, _ = mmcv.imrescale(seg_gt, scale, return_scale=True, interpolation='nearest')
    seg_gt = mmcv.impad_to_multiple(seg_gt, 32, pad_val=250)
    return seg_gt

def compute_panoptic(masks, semantic_seg_mask):
    nstuff = 11
    panoptic_mask = np.ones(semantic_seg_mask.shape[-2:], dtype=np.uint8)*255
    for cls in np.unique(semantic_seg_mask):
        if cls < nstuff:
            panoptic_mask[semantic_seg_mask==cls] = cls

    index = nstuff
    for mask in masks:
        panoptic_mask[mask==1] = index
        index += 1
    return panoptic_mask


def masked_avg_pool(feats, masks, gt=False, levels=[0]):
    map_feats_final = []
    for i in range(len(masks)):
        if gt:
            mask_shape = masks[i][0].shape
            masks_ = torch.tensor(masks[i]).float().cuda()
        else:
            mask_shape = masks[i]['masks'][0].shape
            masks_ = masks[i]['masks'].float()
        map_feats_fpn = []
        for level, feat in enumerate(feats): # For every FPN resolution
            if level not in levels:
                continue
            feat = feat[i]
            feat = feat.unsqueeze(0)
            #feat = F.interpolate(feat.unsqueeze(0), (mask_shape[0]//2, mask_shape[1]//2), mode='bilinear', align_corners=True)
            h, w = feat.shape[-2:]
            map_feats = []
            for mask in masks_:
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                     feat.shape[-2:], mode='nearest')
                area = F.avg_pool2d(mask,
                                    feat.shape[-2:]) * h * w + 0.0005
                x = mask * feat
                x = F.avg_pool2d(input=x,
                                 kernel_size=feat.shape[-2:]) * h * w / area
                map_feats.append(x)
            map_feats_fpn.append(torch.stack(map_feats).squeeze())
        map_feats_fpn = torch.stack(map_feats_fpn).mean(dim=0)
        if len(map_feats_fpn.shape) < 2:
            map_feats_fpn = map_feats_fpn.unsqueeze(0)
        map_feats_final.append(map_feats_fpn)
    return map_feats_final

def convert_masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        fore_x, fore_y = np.where(mask.cpu()==1)
        boxes.append([fore_x.min(), fore_y.min(),
                          fore_x.max(), fore_y.max()])
    return boxes

def compute_box_ious(det_masks, tracked_masks):
    mask_ious = []

    # Convert to BBoxes
    det_boxes = convert_masks_to_boxes(det_masks)
    track_boxes = convert_masks_to_boxes(tracked_masks)
    for dbox in det_boxes:
        dmask_ious = []
        for tbox in track_boxes:
            xA = max(dbox[0], tbox[0])
            yA = max(dbox[1], tbox[1])
            xB = min(dbox[2], tbox[2])
            yB = min(dbox[3], tbox[3])
            interArea = max(0, xB-xA+1) * max(0, yB-yA+1)
            boxAArea = (dbox[2] - dbox[0] + 1) * (dbox[3] - dbox[1] + 1)
            boxBArea = (tbox[2] - tbox[0] + 1) * (tbox[3] - tbox[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            dmask_ious.append(torch.tensor(iou))
        mask_ious.append(torch.stack(dmask_ious))
    return torch.stack(mask_ious).cuda()

def compute_mask_ious(det_masks, tracked_masks):
    mask_ious = []
    for dmask in det_masks:
        dmask_ious = []
        for tmask in tracked_masks:
            if type(dmask) == torch.Tensor:
                dmask = dmask.cpu()

            U= np.logical_or(dmask, tmask).sum()
            if U == 0:
                raise Exception('Union shouldnt be zero')
            else:
                iou = np.logical_and(dmask, tmask).sum()/ float(U)
            dmask_ious.append(torch.tensor(iou))
        mask_ious.append(torch.stack(dmask_ious))
    return torch.stack(mask_ious).cuda()

def convert_and_load_checkpoint(model, weights_path):
    checkpoint = torch.load(weights_path)

    state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()

    convert_first = False

    for loaded_key in state_dict.keys():
        if model.bbox_head is None and 'bbox_head' in loaded_key:
            convert_first = True

    for k, v in state_dict.items():
        if convert_first and 'bbox_head' in k:
            model_state_dict[k.replace('bbox_head', 'ca_head')] = state_dict[k]
        else:
            model_state_dict[k] = state_dict[k]
    model.load_state_dict(model_state_dict)
    return model, checkpoint

def partial_load(model, weights_path, ignore_clf=False, samenclasses=False, convert_dict={}):
    state_dict = torch.load(weights_path)['state_dict']
    model_state_dict = model.state_dict()

    for k, v in state_dict.items():
        if k.split('.')[0] in convert_dict:
            if 'dsolo_cate' in k and ignore_clf:
                continue
            names = k.split('.')
            mapped_name = convert_dict[names[0]] + "".join(['.'+n for n in names[1:]])
            model_state_dict[mapped_name] = state_dict[k]
        # Next two ifs should be handled by convert_dict better
        elif 'dsolo_cate' in k: #TODO: should only happen in bbox_head not ca_head
            if not ignore_clf:
                if not samenclasses:
                    model_state_dict[k] = state_dict[k][:1]
                else:
                    model_state_dict[k] = state_dict[k]
        elif 'backbone' in k and 'stream' not in k and 'projection' not in k:
            # TODO: Use convert_dict instead of this way of hardcoding
            model_state_dict[k[:9]+'appearance_stream.'+k[9:]] = state_dict[k]
            model_state_dict[k[:9]+'motion_stream.'+k[9:]] = state_dict[k]
        elif 'panoptic_head.classifier' in k:
            if ignore_clf:
                continue
        elif k not in model_state_dict.keys():
            print(' Key doesnt exist in current model ', k)
            continue
        else:
            model_state_dict[k] = state_dict[k]

    model.load_state_dict(model_state_dict)
    return model

def compute_gaussian(embeddings, targets):
    means = {}
    sigmas = {}
    for target_class in np.unique(targets):
        class_embeddings = embeddings[:, targets==target_class]
        means[target_class] = class_embeddings.mean(axis=1)
        # sigmas[target_class] = class_embeddings.var(axis=1)
        sigmas[target_class] = np.cov(class_embeddings)
    return means, sigmas

def compute_ood_score_mu_sigma(embeddings, means, vars_):
    """
    embeddings: dxN
    means: [d]*C or Cxd
    sigmas: [dxd]*C or Cxd
    """
    if type(embeddings) != torch.Tensor:
        embeddings = torch.tensor(embeddings).cuda()

    d = embeddings.shape[0]
    scores = []

    eps = 1e-10
    for mean, var in zip(means, vars_):
        if type(mean) != torch.Tensor or type(var) != torch.Tensor:
            mean = torch.tensor(mean).cuda()
            var = torch.tensor(var).cuda()
        else:
            mean = mean.cuda()
            var = var.cuda()

        mean = mean.unsqueeze(1)
        mean_diff = (embeddings - mean).T
        inv_covar = torch.pinverse(var)
        # TODO: investigate why can I use it? Causes NaNs
        #score_normalizer =  torch.log((2 * np.pi)**d * torch.det(var) + eps)
        score_to_class_ = torch.matmul(mean_diff, inv_covar).reshape(-1) * mean_diff.reshape(-1)
        score_to_class_ = -1 * score_to_class_.view(-1, d).sum(dim=1) #- score_normalizer

        scores.append(score_to_class_)
    scores = torch.stack(scores)
    return scores

def compute_ood_scores(embeddings, means_things, sigmas_things, means_stuff, sigmas_stuff):
    things_instances = list(means_things.keys())
    means_things_ = []
    sigmas_things_ = []
    for k in things_instances:
        means_things_.append(torch.tensor(means_things[k]))
        sigmas_things_.append(torch.tensor(sigmas_things[k]))
    means_things_ = torch.stack(means_things_)
    sigmas_things_ = torch.stack(sigmas_things_)

    stuff_classes = list(means_stuff.keys())
    means_stuff_ = []
    sigmas_stuff_ = []
    for k in stuff_classes:
        means_stuff_.append(torch.tensor(means_stuff[k]))
        sigmas_stuff_.append(torch.tensor(sigmas_stuff[k]))
    means_stuff_ = torch.stack(means_stuff_)
    sigmas_stuff_ = torch.stack(sigmas_stuff_)

    scores_things = compute_ood_score_mu_sigma(embeddings, means_things_, sigmas_things_)

    scores_stuff = compute_ood_score_mu_sigma(embeddings, means_stuff_, sigmas_stuff_)

    things_instances = torch.tensor(things_instances)
    stuff_classes = torch.tensor(stuff_classes)

    return scores_things.max(0)[0], things_instances[torch.argmax(scores_things, dim=0)], \
            scores_stuff.max(0)[0], stuff_classes[torch.argmax(scores_stuff, dim=0)]

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def freeze_model_partially(model, freeze_vars=None):
    backbone_exists = False

    for key in freeze_vars.keys():
        if 'backbone' in key:
            backbone_exists = True

    if not backbone_exists:  # It means all backbone should be frozen
        modules = [model.backbone]
    else:
        modules = []

    for key, value in freeze_vars.items():
        try:
            if value:
                modules.append(rgetattr(model, key))
        except:
            raise Exception('Module %s doesnt exist in model'%key)

    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    return model


def process_seg_masks(seg_map, cfg_data, semantic2label):
    seg_prefix = None
    if type(cfg_data) == list:
        for test_cfg in cfg_data:
            if 'seg_prefix' in test_cfg:
                seg_prefix = test_cfg['seg_prefix']
    else:
        seg_prefix = cfg_data['seg_prefix']
    assert seg_prefix is not None

    seg_filename = os.path.join(seg_prefix, seg_map)
    seg_filename = seg_filename.replace('leftImg8bit', 'gtFine_color').replace( \
                        'newImg8bit', 'final_mask')
    gt_seg = mmcv.imread(seg_filename, flag='unchanged').squeeze()
    gt_seg_ = gt_seg.copy()
    gt_seg_unique = np.unique(gt_seg)

    for i in gt_seg_unique:
        gt_seg[gt_seg_==i] = semantic2label[i]
    return gt_seg

def poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def process_gt_masks(gt_masks, img_h, img_w):
    gt_masks_ = []
    for gt_mask in gt_masks:
        gt_mask = poly2mask(gt_mask, img_h, img_w)
        gt_masks_.append(gt_mask)
    return gt_masks_

def vis_gt_known_ood(data, gt, img_norm_cfg, save_dir, palette='cityscapes', last_cls=19):
    img_tensor = data['img'][0]
    seg_color_map = get_color_map(palette)

    if len(img_tensor.shape) < 4:
        img_tensor = img_tensor.unsqueeze(0)
    try:
        img_metas = data['img_meta'][0].data[0]
    except:
        img_metas = data['img_meta'][0]

    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    for img, img_meta, img_gt in zip(imgs, img_metas, gt):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        img_gt = torch.tensor(img_gt).unsqueeze(0).unsqueeze(0)
        full_seg = F.interpolate(img_gt.float(),
                                 img.shape[:2], mode='nearest')[0,0]
        full_seg = full_seg.cpu().numpy().astype(np.uint8)
        full_seg = full_seg[:h, :w]

        known_seg = full_seg.copy()
        ood_seg = np.zeros_like(full_seg)
        known_seg[full_seg > last_cls-1] = 255
        ood_seg[full_seg == last_cls] = 1

        seg_show = img_show.copy()

        for k, v in seg_color_map.items():
            cur_mask = (known_seg==k)
            seg_show[cur_mask] = seg_show[cur_mask] * 0.5 + np.asarray(v[::-1], np.uint8) * 0.5

        img_show[ood_seg==1] = 0.5 * img_show[ood_seg==1] + 0.5 * np.array((255, 0, 0)[::-1])

        mmcv.imwrite(img_show, '{}'.format(save_dir))
        mmcv.imwrite(seg_show, '{}'.format(save_dir.replace('ood', 'seg')))

def vis_seg_known_ood(data, result, img_norm_cfg, save_dir,
                      palette='cityscapes', last_cls=19, semantic2label=None):

    img_tensor = data['img'][0]
    seg_color_map = get_color_map(palette)

    if len(img_tensor.shape) < 4:
        img_tensor = img_tensor.unsqueeze(0)
    try:
        img_metas = data['img_meta'][0].data[0]
    except:
        img_metas = data['img_meta'][0]

    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        full_seg = F.interpolate(cur_result['ca_mahalanobis_seg'].float(),
                                 img.shape[:2], mode='nearest')[0,0]
        full_seg = full_seg.cpu().numpy().astype(np.uint8)
        full_seg = full_seg[:h, :w]

        known_seg = full_seg.copy()
        ood_seg = np.zeros_like(full_seg)
        known_seg[full_seg > last_cls-1] = 255
        ood_seg[full_seg == last_cls] = 1

        # Remap back for getting right colors
        if semantic2label is not None:
            temp = known_seg.copy()
            for key, value in semantic2label.items():
                temp[known_seg==value] = key
            known_seg = temp

        seg_show = img_show.copy()

        for k, v in seg_color_map.items():
            cur_mask = (known_seg==k)
            seg_show[cur_mask] = seg_show[cur_mask] * 0.5 + np.asarray(v[::-1], np.uint8) * 0.5

        img_show[ood_seg==1] = 0.5 * img_show[ood_seg==1] + 0.5 * np.array((255, 0, 0)[::-1])

        mmcv.imwrite(img_show, '{}'.format(save_dir))
        mmcv.imwrite(seg_show, '{}'.format(save_dir.replace('ood', 'seg')))

def vis_seg_ood(data, result, img_norm_cfg, save_dir, score_thr):

    img_tensor = data['img'][0]
    if len(img_tensor.shape) < 4:
        img_tensor = img_tensor.unsqueeze(0)
    try:
        img_metas = data['img_meta'][0].data[0]
    except:
        img_metas = data['img_meta'][0]

    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ood_seg = F.interpolate(cur_result['ca_ood_seg'].unsqueeze(0).unsqueeze(0), img_show.shape[:2])[0,0]
        img_show[ood_seg>score_thr] = 0.7 * img_show[ood_seg>score_thr] + 0.3 * np.array((0, 0, 255))

        mmcv.imwrite(img_show, '{}'.format(save_dir))

def vis_seg(data, result, img_norm_cfg, data_id, colors, score_thr,
            save_dir, dataset_type='cityscapes', inst_ids=None,
            vis_track=False, segmentation=False, seglabels_colors=None,
            seglabels_ignore=None, vis_unknown=False):

    img_tensor = data['img'][0]
    if len(img_tensor.shape) < 4:
        img_tensor = img_tensor.unsqueeze(0)
    try:
        img_metas = data['img_meta'][0].data[0]
    except:
        img_metas = data['img_meta'][0]

    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    class_names = get_classes(dataset_type)

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        if cur_result is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, ::-1]

        if len(cur_result['masks']) != 0:
            seg_label = cur_result['masks']
            seg_label = seg_label.cpu().numpy().astype(np.uint8)
            cate_label = cur_result['labels']
            cate_label = cate_label.cpu().numpy()
            score = cur_result['scores'].cpu().numpy()

        if vis_unknown:
            if 'ca_masks' in cur_result:
                prefix = 'ca_'
            else:
                prefix = '' # To visualise model for DAVIS no multitask

            ca_seg_label = cur_result[prefix+'masks'].cpu().numpy().astype(np.uint8)
            ca_class_labels = cur_result[prefix+'labels'].cpu().numpy().astype(np.uint8)
            ca_scores = cur_result[prefix+'scores'].cpu().numpy()
            ca_indices = ca_scores > score_thr
            ca_seg_label = ca_seg_label[ca_indices]
            ca_class_labels = ca_class_labels[ca_indices]

            ca_indices = ca_class_labels==0
            ca_seg_label = ca_seg_label[ca_indices]

        if segmentation and 'segmentation' in cur_result:
            seg_map = cur_result['segmentation']
            seg_map = cv2.resize(np.asarray(seg_map.cpu(), np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
        if vis_track:
            inst_ids = cur_result['track_ids']

        if len(cur_result['masks']) != 0:
            vis_inds = score > score_thr
            seg_label = seg_label[vis_inds]
            num_mask = seg_label.shape[0]
            cate_label = cate_label[vis_inds]
            cate_score = score[vis_inds]

            if inst_ids is not None:
                inst_ids = inst_ids[vis_inds]

            mask_density = []
            for idx in range(num_mask):
                cur_mask = seg_label[idx, :, :]
                cur_mask = mmcv.imresize(cur_mask, (w, h))
                cur_mask = (cur_mask > 0.5).astype(np.int32)
                mask_density.append(cur_mask.sum())

            orders = np.argsort(mask_density)
            seg_label = seg_label[orders]
            cate_label = cate_label[orders]
            cate_score = cate_score[orders]

            if inst_ids is not None:
                inst_ids = inst_ids[orders]
        else:
            num_mask = 0

        seg_show = img_show.copy()
        unknown_show = img_show.copy()
        if segmentation and 'segmentation' in cur_result:
            for k, v in seglabels_colors.items():
                if k in seglabels_ignore:
                    continue

                cur_mask = (seg_map==k)
                seg_show[cur_mask] = seg_show[cur_mask] * 0.5 + np.asarray(v, np.uint8) * 0.5

        for idx in range(num_mask):
            idx = -(idx+1)
            cur_mask = seg_label[idx, :,:]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
               continue

            cls_name = class_names[cate_label[idx]]
            for entry in cscapes_colormap:
                if entry[0] == cls_name:
                    color_mask = np.array(entry[-1])

            cur_mask_bool = cur_mask.astype(np.bool)
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 + color_mask * 0.5

            contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(seg_show, contours, 0, (0, 0, 0), 2)

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]
            label_text = class_names[cur_cate]

        if vis_unknown:
            for idx in range(ca_seg_label.shape[0]):
                cur_mask = ca_seg_label[idx, :,:]
                cur_mask = mmcv.imresize(cur_mask, (w, h))
                cur_mask = (cur_mask > 0.5).astype(np.uint8)
                if cur_mask.sum() == 0:
                   continue
                color_mask = np.array([255.0, 0.0, 0.0])

                cur_mask_bool = cur_mask.astype(np.bool)
                unknown_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.7 + color_mask * 0.3

                contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(unknown_show, contours, 0, (0, 0, 0), 2)

            path_splits = save_dir.split('/')
            unknown_save_path = ''
            for split in path_splits[:-1]:
                unknown_save_path += split + '/'
            unknown_save_path += 'unknwown_' + path_splits[-1]

            mmcv.imwrite(unknown_show[:, :, ::-1], unknown_save_path)

        if 'png' not in save_dir and 'jpg' not in save_dir and 'jpeg' not in save_dir:
            save_dir = save_dir + '%05d_panoptic.png'%data_id
        mmcv.imwrite(seg_show[:,:,::-1], '{}'.format(save_dir, data_id))

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
