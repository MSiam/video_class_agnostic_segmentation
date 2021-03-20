import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import numpy as np
from mmdet.core import (masked_avg_pool, compute_mask_ious)
from ..registry import HEADS


@HEADS.register_module
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs = 2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 mask_dummy_iou=0,
                 dynamic=True,
                 metric_type='vanilla',
                 loss_type='ce'
                 ):
        super(TrackHead, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.mask_dummy_iou = mask_dummy_iou
        self.num_fcs = num_fcs
        self.metric_type = metric_type
        self.loss_type = loss_type

        if self.with_avg_pool:
            self.avg_pool = masked_avg_pool
        else: # This is not recommended
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):

            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic=dynamic

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, mask_ious, add_mask_dummy=False):
        if add_mask_dummy:
            mask_iou_dummy = torch.ones(mask_ious.shape[0], 1).cuda() * self.mask_dummy_iou
            mask_ious = torch.cat((mask_iou_dummy, mask_ious), dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            assert(len(self.match_coeff)==2), "IOUs Coeff not provided"
            return self.match_coeff[0] * match_ll + self.match_coeff[1] * mask_ious

    def compute_exhaustive_scores(self, ref_x, x):
        scores = []
        for batch, _ in enumerate(x):
            all_x = torch.cat((x[batch], ref_x[batch]), dim=0)
            scores.append(F.sigmoid(F.cosine_similarity(all_x.unsqueeze(0), all_x.unsqueeze(1), dim=2)))
        return scores

    def forward(self, x, ref_x, seg_result, ref_masks, memory_feats=None):
        # x and ref_x are the features of current and reference frame
        # seg_results [ins_masks, cates, cate_scores]

        # Extract features with masked average pooling
        if self.with_avg_pool:
            x = self.avg_pool(x, seg_result)
            map_out = x
            if ref_x is not None:
                ref_x = self.avg_pool(ref_x, ref_masks, gt=True)
            elif memory_feats is not None:
                ref_x = [memory_feats]
            else:
                raise Exception('Either reference image features or memory feats needed')

        # Infer through track head
        for i in range(len(x)):
            x[i] = x[i].view(x[i].size(0), -1)
            ref_x[i] = ref_x[i].view(ref_x[i].size(0), -1)
            for idx, fc in enumerate(self.fcs):
                x[i] = fc(x[i])
                ref_x[i] = fc(ref_x[i])
                if idx < len(self.fcs) - 1:
                    x[i] = self.relu(x[i])
                    ref_x[i] = self.relu(ref_x[i])

        # TODO: Decide on whether to merge FPN feats before matching or match on diff levels
        # Compute matching score
        n = len(x)
        prods = []
        for i in range(n):
            if self.metric_type == 'vanilla':
                prod = torch.mm(x[i], torch.transpose(ref_x[i], 0, 1))
            elif self.metric_type == 'cosine_sim':
                prod = F.cosine_similarity(x[i].unsqueeze(1), ref_x[i].unsqueeze(0), dim=2)
            prods.append(prod)
        if self.loss_type == 'cecl':
            exhaustive_score = self.compute_exhaustive_scores(ref_x, x)
        else:
            exhaustive_score = None
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros( m, 1, device=torch.cuda.current_device())

                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=torch.cuda.current_device())
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy,prods_all], dim=2)
        return match_score, x, exhaustive_score, map_out

    def contrastive_loss(self, exhaustive_scores, cur_ids, ndets, ntracks):
        ntracks = ntracks - 1
        binary_gt = torch.eye(exhaustive_scores.shape[0]).cuda()
        pairs = [torch.range(0, cur_ids.shape[0]-1), cur_ids - 1]
        for x, y in zip(pairs[0], pairs[1]):
            y = y + ndets
            binary_gt[int(x), int(y)] = 1
            binary_gt[int(y), int(x)] = 1
        preds = exhaustive_scores.view(-1)
        return F.binary_cross_entropy(preds, binary_gt.view(-1))

    def compute_pids(self, seg_result, gt_masks, gt_pids):
        seg_gt_pids_final = []
        for i in range(len(seg_result)):
            current_gt_masks = F.interpolate(torch.tensor(gt_masks[i]).float().unsqueeze(0),
                                             seg_result[i]['masks'].shape[-2:],
                                             mode='nearest')[0]
            mask_ious = compute_mask_ious(seg_result[i]['masks'], current_gt_masks)
            _, inds = mask_ious.max(dim=1)
            seg_gt_pids_final.append(gt_pids[i][inds])
        assert seg_gt_pids_final[0].shape[0] == seg_result[0]['masks'].shape[0]
        return seg_gt_pids_final

    def loss(self,
             match_score,
             ids,
             id_weights=None,
             reduce=True,
             exhaustive_scores=None):
        losses = dict()
        if self.dynamic:
            n = len(match_score)
            loss_match = torch.tensor([0.]).cuda()
            n_total = 0
            for batch, (score, cur_ids) in enumerate(zip(match_score, ids)):
                valid_idx = torch.nonzero(cur_ids).view(-1)
                if len(valid_idx) == 0:
                    continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                # TODO: cros entropy loss test after change
                loss_match += F.cross_entropy(score, cur_ids)

                if self.loss_type == 'cecl' and exhaustive_scores is not None:
                    losses['loss_contrastive']  = self.contrastive_loss(exhaustive_scores[batch], cur_ids, *score.shape)

            losses['loss_match'] = loss_match / n
        else:
          if match_score is not None:
              losses['loss_match'] = F.cross_entropy(match_score, ids)
        return losses

