from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MotionDataset(CocoDataset):
    """
    Motion Dataset for Motion Instance Segmentation
    """

    CLASSES = ("moving", "static")
    def prepare_test_img(self, idx):
        results = super().prepare_test_img(idx)

        # TODO: Use more generic way to work with cityscapes as well not only kitti
        if 'kitti' in results['img_meta'][0].data['filename']:
            fileno = int(results['img_meta'][0].data['filename'].split('/')[-1].split('.')[0])
            is_first = fileno==0
        else:
            nframes_span_test = 6
            is_first = (idx % nframes_span_test == 0)

        results['img_meta'][0].data['is_first'] = is_first
        return results
