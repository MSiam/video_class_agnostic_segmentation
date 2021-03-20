from .cityscapes import CityscapesDataset
from .registry import DATASETS
import os.path as osp

@DATASETS.register_module
class CityscapesPanopticDataset(CityscapesDataset):
    """
    Cityscapes/Carla Dataset loading semantic segmentation
    without Instance support
    """

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        # Add reading of semantic segmentation labels
        seg_filename = osp.join(
                self.seg_prefix,
                results['ann_info']['seg_map'].replace(
                        'leftImg8bit', 'gtFine_labelTrainIds'))
        results['ann_info']['seg_filename'] = seg_filename

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        results = self.pipeline(results)

        if results is not None and 'gt_labels' not in results:
            results['gt_labels'] = []
            results['gt_bboxes'] = []
        return results
