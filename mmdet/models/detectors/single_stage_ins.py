import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import scipy.optimize as scipy_optimize
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.tracker_inference import TrackerInference
from mmdet.core.utils.post_proc_utils import  post_process_seg_result, process_bbox_outputs


@DETECTORS.register_module
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 track_head=None,
                 panoptic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 sta_config=None,
                 ca_head=None,
                 max_nottrack=20):

        super(SingleStageInsDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if sta_config is not None:
            self.sta_type = sta_config['sta_type']
            self.sta_kernel_size = sta_config['sta_kernel_size']
        else:
            self.sta_type = 0

        self.sta_module = None

        if neck is not None:
            self.neck_with_warp = False
            if 'Warp' in neck['type']:
                self.neck_with_warp = True

            self.neck = builder.build_neck(neck)

        if bbox_head is not None:
            self.bbox_head = builder.build_head(bbox_head)
        else:
            self.bbox_head = None

        if track_head is not None:
            self.track_head = builder.build_head(track_head)
        else:
            self.track_head = None

        if ca_head is not None:
            self.ca_head = builder.build_head(ca_head)
        else:
             self.ca_head = None

        if panoptic_head is not None:
            self.panoptic_head = builder.build_head(panoptic_head)
        else:
            self.panoptic_head = None

        assert self.bbox_head or self.ca_head

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.debug_frame_id = 0

        self.tracker_inference = TrackerInference(max_nottrack=max_nottrack)

        self.time_inference = 0
        self.time_all = 0
        self.ntimes = 0

    def init_weights(self, pretrained=None):
        super(SingleStageInsDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        if self.bbox_head is not None:
            self.bbox_head.init_weights()

        if self.ca_head is not None:
            self.ca_head.init_weights()

        if self.track_head is not None:
            self.track_head.init_weights()

    def extract_feat(self, img, flow=None, depth=None):
        if type(self.backbone) == ResNet:
            appearance = self.backbone(img)
            merged = None
        elif depth is not None:
            merged, appearance = self.backbone(img, depth)
        elif flow is not None:
            # Flow is used as input not to warp ref_img
            merged, appearance = self.backbone(img, flow)

        if self.with_neck:
            if merged is not None:
                if self.neck_with_warp:
                    merged = self.neck(merged, flow)
                else:
                    merged = self.neck(merged)

            if self.neck_with_warp:
                appearance = None
            else:
                appearance = self.neck(appearance)

        return [merged, appearance]

    def forward_dummy(self, img):
        merged, appearance = self.extract_feat(img)
        if merged is None:
            outs = self.bbox_head(appearance)
        else:
            outs = self.bbox_head(merged)
        return outs

    def forward_train_tracking(losses, appearance, merged, loss_inputs):
        # Removed this till I remember why "and gt_semantic_seg is None:"
        assert self.track_head is not None
        assert self.bbox_head is not None

        # Infer instance segmentation results
        if self.train_cfg.endtoend:
            outs = self.bbox_head(appearance, eval=2)
            loss_inputs.update(outs['train_tensors'])
            losses = self.bbox_head.loss(**loss_inputs)
        else:
            outs = self.bbox_head(appearance, eval=1)
            losses = {}

        seg_inputs = outs['eval_tensors']
        seg_inputs.update({'img_metas': img_metas, 'cfg': self.test_cfg,'rescale':  False})

        seg_result = self.bbox_head.get_seg(**seg_inputs)

        # TODO: Fix this bug of iterating on batch and checing None
        if seg_result is None:
            losses.update({'loss_match': torch.tensor([0.0]).cuda()})
            return losses

        # Filter out low conf seg
        seg_result, empty = post_process_seg_result(seg_result, img)

        if empty: # In case of no new detections passing score_thr
            losses.update({'loss_match': torch.tensor([0.0]).cuda()})
        else:
            # Compute corresponding IDs
            seg_gt_pids = self.track_head.compute_pids(seg_result, gt_masks, gt_pids)

            # Infer feats for reference image
            if x_ref is None:
                _, x_ref = self.extract_feat(ref_img)

            # Compute matching scores between current and ref
            match_score, _, exhaustive_scores, _ = self.track_head(appearance, x_ref, seg_result, ref_masks)

            # Compute final tracking loss
            losses.update(self.track_head.loss(match_score, seg_gt_pids, exhaustive_scores=exhaustive_scores))
        return losses

    def forward_train_ca(self, merged, appearance, loss_inputs):
        assert self.ca_head is not None  #TODO: Add flag to identify class agnostic gt
        x = merged if merged is not None else appearance

        ca_outs = self.ca_head(feats=x) # Cunknown

        if 'train_tensors' in ca_outs:
            loss_inputs.update(ca_outs['train_tensors'])

        if 'class_agnostic_embeddings' in ca_outs:
            loss_inputs.update({'class_agnostic_embeddings': ca_outs['class_agnostic_embeddings']})
            loss_inputs.update({'merged_fpn_embeddings': ca_outs['merged_fpn_embeddings']})
            if 'highres_feats' in ca_outs:
                loss_inputs.update({'highres_feats': ca_outs['highres_feats']})
            if 'scl_features' in ca_outs:
                loss_inputs.update({'scl_features': ca_outs['scl_features']})
        temp_losses = self.ca_head.loss(**loss_inputs)

        ca_losses = {}
        for key, _ in temp_losses.items():
            ca_losses['ca_'+key] = temp_losses[key]

        return ca_losses

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img=None,
                      ref_bboxes=None,
                      ref_masks=None,
                      gt_pids=None,
                      ref_pids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None, gt_semantic_seg=None, gt_obj_ids=None,
                      gt_semantic_seg_Nx=None, ref_labels=None, ref_obj_ids=None,
                      flow=None,
                      depth=None,
                      save_tensorboard=False):
        merged, appearance = self.extract_feat(img, flow, depth)

        x_ref = None
        if self.sta_module is not None:
            # TODO: This needs to be fixed to handle both merged and appearance
            assert ref_img is not None
            x_ref = self.extract_feat(ref_img)
            x = merged if merged is not None else appearance
            x = self.sta_module(x_ref, x)

        losses = {}
        loss_inputs = {'gt_bbox_list': gt_bboxes, 'gt_label_list': gt_labels, 'gt_mask_list': gt_masks,
                       'img_metas': img_metas, 'cfg': self.train_cfg, 'save_tensorboard': save_tensorboard,
                       'gt_bboxes_ignore': gt_bboxes_ignore, 'gt_semantic_seg': None}

        # Note KITTIMOTSTrack loader doesnt separate to list and rather 1 tensor
        if self.train_cfg.train_tracker:
           losses = self.forward_train_tracking(losses, appearance, merged, loss_inputs)

        elif self.train_cfg.train_inst_seg and not self.train_cfg.train_ca:  # TODO: support endtoend training with class agnostic head
            assert self.bbox_head is not None
            outs = self.bbox_head(appearance)
            loss_inputs.update(outs['train_tensors'])
            losses = self.bbox_head.loss(**loss_inputs)

        if self.train_cfg.train_panoptic:
            assert gt_semantic_seg is not None
            if not self.train_cfg.endtoend:
                pass
            else:
                seg_map, _ = self.panoptic_head(appearance)
                loss_seg = self.panoptic_head.loss(seg_map, gt_semantic_seg)
                losses['loss_seg'] = loss_seg

        if self.train_cfg.train_ca:
            loss_inputs['gt_semantic_seg'] = gt_semantic_seg
            ca_losses = self.forward_train_ca(merged, appearance, loss_inputs)
            losses.update(ca_losses)

        return losses

    def simple_test(
        self, img, img_meta, ref_img=None, flow=None, depth=None, rescale=False,
        detect_only=False, temporal_consistency=False
    ):
        # Computing Instance Segmentation Output
        start_time = time.time()

        merged, appearance = self.extract_feat(img, flow, depth)
        x = merged if merged is not None else appearance

        if self.bbox_head is not None:
            outs = self.bbox_head(appearance, eval=1)
            seg_result = process_bbox_outputs(outs, self.bbox_head, img_meta, rescale, cfg=self.test_cfg)
            # Compute instance ids
            seg_result, empty = post_process_seg_result(seg_result, img, train=False)
        else:
            seg_result = [{}]
            empty = True

        semantic_seg = None
        if self.panoptic_head is not None:
            if seg_result[0] is None:
                seg_result[0] = {}
            seg_map, _ = self.panoptic_head(appearance)
            seg_result[0]['segmentation'] = seg_map.argmax(dim=1)[0]
            semantic_seg = seg_result[0]['segmentation']

        #TODO: I should still return segmentation
        if empty:
            inst_ids = []
            if seg_result[0] is None:
                seg_result[0] = [{'masks': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}]
            else:
                seg_result[0].update({'masks': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])})

        if self.ca_head is not None:
            outs = self.ca_head(feats=x, eval=1)
            seg_result_ca = process_bbox_outputs(outs, self.ca_head, img_meta, rescale,
                                                  pred_semantic_seg=semantic_seg,
                                                  cfg=self.test_cfg)
            seg_result_ca, empty = post_process_seg_result(seg_result_ca, img, train=False)

            if temporal_consistency:
                seg_result_ca = self.tracker_inference.associate_mask_based_iou(img_meta, seg_result_ca, empty)

            if seg_result_ca[0] is None:
                seg_result_ca = [{'masks': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}]

            for key, value in seg_result_ca[0].items():
                seg_result[0]['ca_'+key] = seg_result_ca[0][key]

        if detect_only:
            return seg_result

        # TODO: Handle Merged VS Appearance in tracking
        # Assuming always batchsize 1 in inference
        if self.track_head is not None:
            start_time = time.time()
            # Indicate first frame in a sequence to reset memory
            inst_ids = self.tracker_inference.infer(appearance, img_meta, seg_result, empty, self.track_head)

            inst_ids = torch.tensor(inst_ids)
            seg_result[0]['track_ids'] = inst_ids
            self.time_all += (time.time() - start_time)
            self.ntimes += 1
            print('Inference time ', self.tracker_inference.time_inference / self.ntimes,
                  ' Inference+Postproc ', self.time_all / self.ntimes)

        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
