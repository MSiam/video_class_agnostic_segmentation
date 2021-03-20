# Video Class Agnostic Segmentation
Official Datasets and Implementation from our Paper "Video Class Agnostic Segmentation in Autonomous Driving".

# Installation
* Download Required Packages
```
pip install -r requirements.txt
```

* Setup mmdet
```
python setup.py develop
```

# Dataset Preparation

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


## References

* KITTI-MOTS
* Cityscapes-VPS
* SOLO
* This Repository built upon [SOLO Code]()
