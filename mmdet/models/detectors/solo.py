from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SOLO(SingleStageInsDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 track_head=None,
                 panoptic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 sta_config=None,
                 ca_head=None,
                 max_nottrack=20):
        super(SOLO, self).__init__(backbone, neck, bbox_head, track_head, panoptic_head, train_cfg,
                                   test_cfg, pretrained, sta_config=sta_config, ca_head=ca_head,
                                   max_nottrack=max_nottrack)
