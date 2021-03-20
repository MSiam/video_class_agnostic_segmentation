from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class KITTIMOTSDataset(CocoDataset):
    """
    KITTI Dataset for Instance Segmentation
    """
    CLASSES = ("car", "person")
