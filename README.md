# Video Class Agnostic Segmentation
Official Datasets and Implementation from our Paper "Video Class Agnostic Segmentation in Autonomous Driving".



# Installation
This repo is tested under Python 3.6, PyTorch 1.4

* Download Required Packages
```
pip install -r requirements.txt
```

* Setup mmdet
```
python setup.py develop
```

# Dataset Preparation

* Follow Dataset Preparation [Instructions](https://github.com/MSiam/video_class_agnostic_segmentation/blob/main/Motion_Dataset_Download.md).

# Inference

* Download Trained Weights on Ego Flow Suppressed, trained on Cityscapes and KITTI-MOTS

* Modify Configs according to dataset path + Image/Annotation/Flow prefix
```
configs/data/kittimots_motion_supp.py
configs/data/cscapesvps_motion_supp.py
```

* Evaluate CAQ, 
```
python tools/test_eval_caq.py CONFIG_FILE WEIGHTS_FILE
```
CONFIG_FILE: configs/infer_kittimots.py or configs/infer_cscapesvps.py

Note better CAQ than reported from paper on KITTIMOTS as the flow suppressed for training images was further masked to remove objects 
that were moving but do not belong to Car or Pedestrian (unlabelled objects).

* Qualitative Results
```
python tools/test_vis.py CONFIG_FILE WEIGHTS_FILE --vis_unknown --save_dir OUTS_DIR
```
# Training

Coming Soon ...

# Acknowledgements

Dataset and Repository relied on these sources:

* Voigtlaender, Paul, et al. "Mots: Multi-object tracking and segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
* Kim, Dahun, et al. "Video panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
* Wang, Xinlong, et al. "Solo: Segmenting objects by locations." European Conference on Computer Vision. Springer, Cham, 2020.
* This Repository built upon [SOLO Code](https://github.com/WXinlong/SOLO)

# Contact
If you have any questions regarding the dataset or repository, please contact menna.seyam@gmail.com.
