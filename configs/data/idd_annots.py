from configs.data.idd import *

data = dict(
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'IDD_Annotated_Annotations.json',
        img_prefix=data_root+'/images/',
        flow_prefix=data_root+'/flow/',
        pipeline=test_pipeline))

