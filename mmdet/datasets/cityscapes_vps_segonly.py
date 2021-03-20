import torch

from .cityscapes import CityscapesDataset
from .registry import DATASETS
import os
import os.path as osp
import copy
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module
class CityscapesVPSSegDataset(CityscapesDataset):
    """
    Cityscapes VPS Dataset loading semantic segmentation
        Support Contrastive Training (Supervised/Temporal)
    """
    def get_segmentationfile(self, seg_file, seg_dir):
        ftokens = seg_file.split('_')
        for f in os.listdir(seg_dir):
            tokens = f.split('_')
            if tokens[0] == ftokens[0] and tokens[2] == ftokens[2] and\
                    tokens[3] == ftokens[3] and tokens[4] == ftokens[4]:

                return osp.join(seg_dir, f)
        return None

    def select_ref_image(self, idx):
        curr_img_info = self.img_infos[idx]

        ref_idx = idx - self.skip
        prev_img_info = self.img_infos[ref_idx]

        curr_tokens = curr_img_info['file_name'].split('_')
        prev_tokens = prev_img_info['file_name'].split('_')

        if curr_tokens[2] == prev_tokens[2] and curr_tokens[0] == prev_tokens[0]:
            return ref_idx
        else:
            return -1

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        # Add reading of semantic segmentation labels
        seg_file = results['ann_info']['seg_map']
        seg_filename = osp.join(
                self.seg_prefix,
                seg_file.replace('leftImg8bit', 'gtFine_color')).replace(
                                 'newImg8bit', 'final_mask')
        if not osp.exists(seg_filename):
            newfile = self.get_segmentationfile(seg_file, self.seg_prefix)
            if newfile is not None:
                seg_filename = newfile

        results['ann_info']['seg_filename'] = seg_filename

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        if self.ref_prefix is not None:
            ref_idx = self.select_ref_image(idx)
            if ref_idx == -1:
                return None
            else:
                results['img_info']['ref_filename'] = self.img_infos[ref_idx]['filename']

        self.pre_pipeline(results)

        results = self.pipeline(results)

        if results is not None and 'gt_labels' not in results:
            results['gt_labels'] = []
            results['gt_bboxes'] = []

        results['img_meta'].data['cl_temporal'] = self.cl_temporal

        if self.cl_temporal:
            results_augment = {}
            for key in results:
                if 'ref' in key:
                    results_augment[key.replace('ref_', '')] = copy.deepcopy(results[key])
                elif key == 'img_meta':
                    results_augment[key] = copy.deepcopy(results[key])
                    results_augment[key].data['filename'] = self.img_infos[ref_idx]['filename']
                elif key not in ['img', 'depth', 'gt_semantic_seg']:
                    results_augment[key] = results[key]
            del results['ref_img']
            del results['ref_depth']

            if results['gt_semantic_seg'] is not None:
                results_augment['gt_semantic_seg'] = DC(
                        torch.ones(results['gt_semantic_seg'].data.shape).byte() * 255
                )
            else:
                del results['gt_semantic_seg']

            results = [results, results_augment]
        return results
