# Dataset Preparation

We build our motion annotations on KITTI-MOTS[1] and Cityscapes-VPS[2].

## KITTI

* Images: Download the original [KITTI-MOTS Dataset](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015).
* Flow: Download Precomupted [Flow](https://drive.google.com/file/d/1tIyRKO5o9imAF3huUo0s-R-ys4znly5t/view?usp=sharing).
* Annotations: Download [motion annotations](https://drive.google.com/file/d/1YT5aQ8WBloFoQg1gu8OYtwxW238tR1Qt/view?usp=sharing).
* Construct Dataset Folder with Structure

    .
    +-- Images
    +-- Flow
    +-- Flow_Suppressed
    +-- Annotations

## Cityscapes
* Download motion annotations for Cityscapes
* Images: Download the original [Cityscapes-VPS](https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0). Follow full instructions [here](https://github.com/mcahny/vps/blob/master/docs/DATASET.md).
* Flow: Download Precomupted [Flow](https://drive.google.com/file/d/1HE4WTIW7HvjpQPU2wZ-eD6CVxmlAwigb/view?usp=sharing).
* Annotations: Download [motion annotations](https://drive.google.com/file/d/1tXnThgg6TIVfravqEicm3DKsYYEbFwRg/view?usp=sharing).
* Construct Dataset Folder with Structure
    .
    +-- train
    |   +-- images
    |   +-- flow
    |   +-- flow_suppressed
    +-- val
    |   +-- images
    |   +-- flow
    |   +-- flow_suppressed
    +-- annotations

# References

[1] Voigtlaender, Paul, et al. "Mots: Multi-object tracking and segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
[2] Kim, Dahun, et al. "Video panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
