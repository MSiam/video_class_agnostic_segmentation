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

* Follow Dataset Preparation [Instructions]().

# Inference

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

# References

* KITTI-MOTS
* Cityscapes-VPS
* SOLO
* This Repository built upon [SOLO Code]()
