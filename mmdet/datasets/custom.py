import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 ref_prefix=None,
                 flow_prefix=None,
                 depth_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 ref_ann_file=None,
                 filter_empty_gt=True,
                 enable_cl_augment=False,
                 cl_temporal=False,
                 skip=1):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.ref_prefix = ref_prefix
        self.flow_prefix = flow_prefix
        self.depth_prefix = depth_prefix
        self.enable_cl_augment = enable_cl_augment
        self.cl_temporal = cl_temporal
        self.skip = skip

        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.ref_ann_file = ref_ann_file

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.flow_prefix is None or osp.isabs(self.flow_prefix)):
                self.flow_prefix = osp.join(self.data_root, self.flow_prefix)
            if not (self.depth_prefix is None or osp.isabs(self.depth_prefix)):
                self.depth_prefix = osp.join(self.data_root, self.depth_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.ref_prefix is None or osp.isabs(self.ref_prefix)):
                self.ref_prefix = osp.join(self.data_root, self.ref_prefix)
            if not osp.isabs(self.ref_ann_file):
                self.ref_ann_file = osp.join(self.data_root, self.ref_ann_file)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.ann_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['ref_prefix'] = self.ref_prefix
        results['proposal_file'] = self.proposal_file
        results['flow_prefix'] = self.flow_prefix
        results['depth_prefix'] = self.depth_prefix

        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            data_augment = None

            # In Case of Contrastive Training Sample MultiViewedBatch
            if data is not None and self.enable_cl_augment:
                data_augment = self.prepare_train_img(idx)
                data = [data, data_augment]

            if data is None or (data_augment is None and self.enable_cl_augment):
                idx = self._rand_another(idx)
                continue

            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        results['img_meta'][0].data['idx'] = idx
        return results
