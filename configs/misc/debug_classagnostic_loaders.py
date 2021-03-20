# dataset settings
from configs.data.kittimots_motion_supp import data as data_kittimots_motion
from configs.data.cscapesvps_motion_supp import data as data_cscapesvps_motion

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)

for idx, pipeline in enumerate(data_kittimots_motion['train']['pipeline']):
    if pipeline['type'] == 'Normalize':
        data_kittimots_motion['train']['pipeline'][idx]['mean'] = img_norm_cfg['mean']
        data_kittimots_motion['train']['pipeline'][idx]['std'] = img_norm_cfg['std']
        data_kittimots_motion['train']['pipeline'][idx]['to_rgb'] = img_norm_cfg['to_rgb']

for idx, pipeline in enumerate(data_cscapesvps_motion['train']['pipeline']):
    if pipeline['type'] == 'Normalize':
        data_cscapesvps_motion['train']['pipeline'][idx]['mean'] = img_norm_cfg['mean']
        data_cscapesvps_motion['train']['pipeline'][idx]['std'] = img_norm_cfg['std']
        data_cscapesvps_motion['train']['pipeline'][idx]['to_rgb'] = img_norm_cfg['to_rgb']

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=[data_kittimots_motion['train'], data_cscapesvps_motion['train']],
    val=data_kittimots_motion['val'],
    test=data_kittimots_motion['test'],)
