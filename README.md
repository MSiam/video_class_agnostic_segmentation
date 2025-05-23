# Video Class Agnostic Segmentation
\[[Method Paper]()\] \[[Benchmark Paper](https://arxiv.org/pdf/2103.11015.pdf)\] \[[Project](https://msiam.github.io/vca/)\] \[[Demo](https://www.youtube.com/watch?v=c9hMFHdTs6M)\]

Official Datasets and Implementation from our Paper "Video Class Agnostic Segmentation Benchmark in Autonomous Driving" in Workshop on Autonomous Driving, CVPR 2021.

<div align="center">
<img src="https://github.com/MSiam/video_class_agnostic_segmentation/blob/main/images/VCA_Teaser.png" width="70%" height="70%"><br><br>
</div>


# Installation
This repo is tested under Python 3.6, PyTorch 1.4

* Download Required Packages
```
pip install -r requirements.txt
pip install "git+https://github.com/cocodataset/panopticapi.git"
```

* Setup mmdet
```
python setup.py develop
```

# Motion Segmentation Track
## Dataset Preparation

* Follow Dataset Preparation [Instructions](https://github.com/MSiam/video_class_agnostic_segmentation/blob/main/Motion_Dataset_Download.md).
* Low resolution view of the [full dataset](https://www.youtube.com/playlist?list=PL4jKsHbreeuBhEmzcL94JxWzVear79r5z)
  
## Inference

* Download [Trained Weights](https://drive.google.com/file/d/16qEH0WoFVt0n6Ooi6zl4ymWKZYv1YVJ8/view?usp=sharing) on Ego Flow Suppressed, trained on Cityscapes and KITTI-MOTS

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


* Qualitative Results
```
python tools/test_vis.py CONFIG_FILE WEIGHTS_FILE --vis_unknown --save_dir OUTS_DIR
```

* Evaluate Image Panoptic Quality, Note: evaluated on 1024x2048 Images
```
python tools/test_eval_ipq.py configs/infer_cscapesvps_pq.py WEIGHTS_FILE --out PKL_FILE
```

## Training

Coming Soon ...

# Open-set Segmentation Track

Coming soon ...

# Acknowledgements

Dataset and Repository relied on these sources:

* Voigtlaender, Paul, et al. "Mots: Multi-object tracking and segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
* Kim, Dahun, et al. "Video panoptic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
* Wang, Xinlong, et al. "Solo: Segmenting objects by locations." European Conference on Computer Vision. Springer, Cham, 2020.
* This Repository built upon [SOLO Code](https://github.com/WXinlong/SOLO)

# Citation

```
@article{siam2021video,
      title={Video Class Agnostic Segmentation Benchmark for Autonomous Driving}, 
      author={Mennatullah Siam and Alex Kendall and Martin Jagersand},
      year={2021},
      eprint={2103.11015},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Contact
If you have any questions regarding the dataset or repository, please contact menna.seyam@gmail.com.
