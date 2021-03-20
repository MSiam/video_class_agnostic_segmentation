import numpy as np
import scipy.optimize as scipy_optimize
import time
import torch
import torch.nn.functional as F
from mmdet.core import vis_seg, compute_box_ious

class TrackerInference:
    def __init__(self, max_nottrack):
        # Memory components
        self.last_inst_id = 0
        self.memory_feats = {}
        self.memory_masks = None
        self.memory_nottrack = {}
        self.count_positive = {}
        self.nottrack_max = max_nottrack
        self.obj_ids = None

        self.time_inference = 0

    def associate_mask_based_iou(self, img_meta, seg_result, empty):
        is_first = img_meta[0]['is_first']
        final_masks = [{'masks': [], 'scores': [], 'labels': []}]
        if empty:
            pass
        elif is_first or self.memory_masks is None:
            # Create memory features and masks
            valid_inds = seg_result[0]['labels'] == 0
            moving_masks = seg_result[0]['masks'][valid_inds]
            self.memory_masks = None
            if moving_masks.shape[0] != 0:
                # Generate initial instance ids to be tracked
                inst_ids = np.arange(1, moving_masks.shape[0] + 1)
                self.memory_masks = {}
                self.memory_nottrack = {}
                self.count_positive = {}

                for it, iid in enumerate(inst_ids):
                    self.memory_masks[iid] = moving_masks[it]
                    self.memory_nottrack[iid] = 0
                    self.count_positive[iid] = 1
                self.last_inst_id = iid + 1
        else:
            # Compute Matching scores
            valid_inds = seg_result[0]['labels'] == 0
            moving_masks = seg_result[0]['masks'][valid_inds]
            if moving_masks.shape[0] != 0:
                prev_masks = []
                key_ids = []
                for k, v in self.memory_masks.items():
                    prev_masks.append(v)
                    key_ids.append(k)
                mask_ious = compute_box_ious(moving_masks, prev_masks)
                row_inds, col_inds = scipy_optimize.linear_sum_assignment(-1 * mask_ious.cpu())

                # Getting IDs of assigned ones
                assigned_dets = []
                assigned_tracks = []
                for detid, trackid in zip(row_inds, col_inds):
                    if mask_ious[detid, trackid] > 0.5:
                        self.memory_masks[key_ids[trackid]] = moving_masks[detid]
                        self.memory_nottrack[key_ids[trackid]] = 0
                        self.count_positive[key_ids[trackid]] += 1
                        assigned_dets.append(detid)
                        assigned_tracks.append(key_ids[trackid])

                # Check for Consistently Detected Masks and announce as moving and remove from queue
                for key, value in self.count_positive.items():
                    if self.count_positive[key] > 5 and key in assigned_tracks:
                        print('Ensure temporal consistency for moving objects')
                        final_masks[0]['masks'].append(self.memory_masks[key])
                        final_masks[0]['labels'].append(torch.tensor(0))
                        final_masks[0]['scores'].append(torch.tensor(1))

                for detid in range(mask_ious.shape[0]):
                    if detid not in assigned_dets:
                        iid = self.last_inst_id
                        self.memory_masks[iid] = moving_masks[detid]
                        self.memory_nottrack[iid] = 0
                        self.count_positive[iid] = 1
                        self.last_inst_id += 1

                for mid in key_ids:
                    if mid not in assigned_tracks:
                        self.memory_nottrack[mid] += 1

                for mid in key_ids:
                    if self.memory_nottrack[mid] > self.nottrack_max:
                        del self.memory_masks[mid]
                        del self.memory_nottrack[mid]
                        del self.count_positive[mid]

        if len(final_masks[0]['masks']) != 0:
            final_masks[0]['masks'] = torch.stack(final_masks[0]['masks'])
            final_masks[0]['scores'] = torch.stack(final_masks[0]['scores'] )
            final_masks[0]['labels'] = torch.stack(final_masks[0]['labels'] )
        else:
            for key in final_masks[0].keys():
                final_masks[0][key] = torch.tensor(final_masks[0][key])

        return final_masks

    ####### Tracking Functionality ################
    def create_new_instance(self, mask, feats):
        new_inst_id = self.last_inst_id
        self.last_inst_id += 1
        self.memory_masks[new_inst_id] = mask
        self.memory_feats[new_inst_id] = feats
        self.memory_nottrack[new_inst_id] = 0
        return torch.tensor(new_inst_id).cuda()

    def update_instance(self, cid, feat, mask):
        new_inst_id = int(cid)
        #[*self.memory_feats.keys()][cid-1]
        self.memory_feats[new_inst_id] = feat
        self.memory_masks[new_inst_id] = mask
        self.memory_nottrack[new_inst_id] = 0
        return torch.tensor(new_inst_id).cuda()

    def infer(self, x, img_meta, seg_result, empty, track_head):
        inst_ids = []
        is_first = img_meta[0]['is_first']
        if empty:
            if is_first:
                self.memory_masks = None  # Reset
            pass
        elif is_first or self.memory_masks is None:
            # Generate initial instance ids to be tracked
            inst_ids = np.arange(1, seg_result[0]['masks'].shape[0]+1)
            self.last_inst_id = seg_result[0]['masks'].shape[0] + 1
            _, new_feats, _, _ = track_head(x, x,
                                          seg_result,
                                          [seg_result[0]['masks']])
            self.memory_feats = {}
            self.memory_masks = {}
            self.memory_nottrack = {}

            # Create memory features and masks
            for it, iid in enumerate(inst_ids):
                self.memory_masks[iid] = seg_result[0]['masks'][it]
                self.memory_feats[iid] = new_feats[0][it]
                self.memory_nottrack[iid] = 0
        else:
            # Compute Matching scores

            prev_masks = []
            prev_feats = []
            key_ids = []
            for k, v in self.memory_masks.items():
                prev_masks.append(v)
                prev_feats.append(self.memory_feats[k])
                key_ids.append(k)

            prev_masks = torch.stack(prev_masks)
            prev_feats = torch.stack(prev_feats)

            start_time = time.time()
            match_score, new_feats, _, _ = track_head(x, None,
                                                     seg_result,
                                                     [prev_masks],
                                                     memory_feats=prev_feats)
            self.time_inference += (time.time() - start_time)

            match_score = F.softmax(match_score[0], dim=1)

            # Compute Mask IoUs between detected and tracked masks
            # PS: Inference always batchsize 1
            # TODO: use compute_comp_scores in track_head to match classes
            # Useful when #classes > 1
            mask_ious = compute_box_ious(seg_result[0]['masks'], prev_masks)
            match_score = track_head.compute_comp_scores(match_score,
                                                               mask_ious,
                                                               add_mask_dummy=True)
            #key_ids =  [*self.memory_feats.keys()]
            row_inds, col_inds = scipy_optimize.linear_sum_assignment(-1 * match_score[:, 1:].cpu())

            # Getting IDs of assigned ones
            inst_ids = []
            for detid, trackid in zip(row_inds, col_inds):
                inst_ids.append(torch.tensor(key_ids[trackid]))
                self.update_instance(inst_ids[-1],
                                     new_feats[0][detid],
                                     seg_result[0]['masks'][detid])

            # Setting new IDs for newly tracked ones
            for detid in range(match_score.shape[0]):
                if detid not in row_inds:
                    inst_ids.append(self.create_new_instance(seg_result[0]['masks'][detid],
                                                           new_feats[0][detid]))

            # Handle Objects not tracked for some X frames to be removed
            for mid in self.memory_feats.keys():
                if mid not in inst_ids:
                    self.memory_nottrack[mid] += 1

            for mid in [*self.memory_feats.keys()]:
                if self.memory_nottrack[mid] > self.nottrack_max:
                    del self.memory_feats[mid]
                    del self.memory_masks[mid]
                    del self.memory_nottrack[mid]

            # Converting ids to int then all array to numpy
            for iid, _ in enumerate(inst_ids):
                inst_ids[iid] = int(inst_ids[iid])
        return inst_ids
