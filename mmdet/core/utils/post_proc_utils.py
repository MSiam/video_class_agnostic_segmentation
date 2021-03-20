import torch.nn.functional as F

def post_process_seg_result(seg_result, img, train=False):
    score_thr = 0.3
    empty = True

    for i in range(len(seg_result)):
        if seg_result[i] is None or 'scores' not in seg_result[i]:  # Happens when only embeddings are there
            continue
        vis_inds = seg_result[i]['scores'] > score_thr
        seg_result[i]['masks'] = seg_result[i]['masks'][vis_inds]
        seg_result[i]['labels'] = seg_result[i]['labels'][vis_inds]
        seg_result[i]['scores'] = seg_result[i]['scores'][vis_inds]

        if seg_result[i]['masks'].shape[0] != 0:
            empty = False
            if train:
                seg_result[i]['masks'] = F.interpolate(seg_result[i]['masks'].unsqueeze(0).float(), \
                    img[i].shape[-2:], mode='nearest').squeeze()
                if len(seg_result[i]['masks'].shape) < 3:
                    seg_result[i]['masks'] = seg_result[i]['masks'].unsqueeze(0)

    return seg_result, empty

def process_bbox_outputs(outs, bbox_head, img_meta, rescale, pred_semantic_seg=None, cfg=None):
    if 'eval_tensors' in outs:  # BBox or Ca Head is using decoupled SOLO
        seg_inputs = outs['eval_tensors']
        seg_inputs.update({'img_metas': img_meta, 'cfg': cfg, 'rescale': rescale})
        seg_result = bbox_head.get_seg(**seg_inputs)
    else:
        seg_inputs = {'class_agnostic_embeddings': outs['class_agnostic_embeddings'],
                      'merged_fpn_embeddings': outs['merged_fpn_embeddings'],
                      'pred_semantic_seg': pred_semantic_seg}
        seg_result = bbox_head.get_seg(**seg_inputs)

    extra_keys = ['class_agnostic_embeddings', 'merged_fpn_embeddings']
    for key in extra_keys:
        if key in outs:
            seg_result[0][key] = outs[key]
    return seg_result
