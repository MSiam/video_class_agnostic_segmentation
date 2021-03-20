# Dataset Preparation

We build our motion annotations on KITTI-MOTS[1] and Cityscapes-VPS[2].

## KITTI

* Images: Download the original [KITTI-MOTS Dataset](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015).
* Flow: Download Precomupted Flow []().
* Annotations: Download motion annotations for KITTI.
* Construct Dataset Folder with Structure

    .
    ├── Images
    ├── Flow
    ├── Flow_Suppressed
    └── Annotations

## Cityscapes
* Download motion annotations for Cityscapes

# References

[1] Voigtlaender, Paul, et al. "Mots: Multi-object tracking and segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
[2] Kim, Dahun, et al. "Video panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
