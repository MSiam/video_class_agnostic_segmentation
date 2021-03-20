import torch
import mmcv
import numpy as np
from imagecorruptions import corrupt
from numpy import random
import torchvision.transforms as torchvision_transforms
from PIL import Image

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES
import pdb
import cv2

@PIPELINES.register_module
class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def _apply_onimage(self, sample):

        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return np.asarray(sample, np.float32)

    def __call__(self, results):
        keys = ['img']

        if 'ref_img' in results:
            keys.append('ref_img')

        for el in keys:
            results[el] = self._apply_onimage(results[el])
        return results

@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        els = ['ref_img', 'img'] if 'ref_img' in results else ['img']
        if 'flow' in results:
            els += ['flow']
        if 'depth' in results:
            els += ['depth']
        if 'ref_depth' in results:
            els += ['ref_depth']

        for key in ['flow', 'depth', 'ref_img', 'ref_depth']:
            if key in results:
                if results[key].shape != results['img'].shape:
                    img = results[key]
                    img = mmcv.imresize(img, (results['img'].shape[1],  results['img'].shape[0]))
                    results[key] = img

        for el in els:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[el], results['scale'], return_scale=True)
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[el], results['scale'], return_scale=True)
                scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                        dtype=np.float32)


            results[el] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        els = ['ref_bbox_fields', 'bbox_fields'] if 'ref_bbox_fields' in results else ['bbox_fields']
        for el in els:
            img_shape = results['img_shape']
            for key in results.get(el, []):
                bboxes = results[key] * results['scale_factor']
                bboxes[:, 0::2] = np.clip(
                        bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(
                        bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes

    def _resize_masks(self, results):
        els = ['ref_mask_fields', 'mask_fields'] if 'ref_mask_fields' in results else ['mask_fields']
        for el in els:
            for key in results.get(el, []):
                if results[key] is None:
                    continue
                if self.keep_ratio:
                    masks = [
                        mmcv.imrescale(
                            mask, results['scale_factor'],
                            interpolation='nearest')
                        for mask in results[key]
                    ]
                else:
                    mask_size = (results['img_shape'][1],
                        results['img_shape'][0])
                    masks = [
                        mmcv.imresize(mask, mask_size,
                            interpolation='nearest')
                        for mask in results[key]
                    ]
                results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        # self._resize_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        else:
            self.flip_ratio = 0

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            els = ['img']
            if 'ref_img' in results:
                els += ['ref_img']
            if 'flow' in results:
                els += ['flow']
            if 'depth' in results:
                els += ['depth']
            if 'ref_depth' in results:
                els += ['ref_depth']

            for el in els:
                results[el] = mmcv.imflip(results[el])

            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            for key in results.get('ref_bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
            for key in results.get('ref_mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        els = ['ref_img', 'img'] if 'ref_img' in results else ['img']
        if 'flow' in results:
            els += ['flow']
        if 'depth' in results:
            els += ['depth']
        if 'ref_depth' in results:
            els += ['ref_depth']

        for el in els:
            if self.size is not None:
                padded_img = mmcv.impad(results[el], self.size)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[el], self.size_divisor, pad_val=self.pad_val)
            results[el] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        els = ['ref_mask_fields', 'mask_fields'] if 'ref_mask_fields' in results else ['mask_fields']
        for el in els:
            pad_shape = results['pad_shape'][:2]
            for key in results.get(el, []):
                padded_masks = [
                    mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                    for mask in results[key]
                ]
                results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        els = ['img']
        if 'ref_img' in results:
            els += ['ref_img']
        if 'flow' in results:
            # 2 channel flow is raw no need to normalize used for warping solely
            if results['flow'].shape[-1] == 3:
                els += ['flow']
        if 'depth' in results:
            els += ['depth']
        if 'ref_depth' in results:
            els += ['ref_depth']

        for el in els:
            results[el] = mmcv.imnormalize(
                results[el], self.mean, self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
                self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        ori_shape = img.shape
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        for key in ['ref_img', 'flow', 'depth', 'ref_depth']:
            if key in results:
                img_ = results[key]
                img_ = img_[crop_y1:crop_y2, crop_x1:crop_x2, :]
                results[key] = img_

        results['img_shape'] = img_shape
        results['crop_coords'] = [crop_y1,crop_y2,crop_x1,crop_x2]

        # crop bboxes accordingly and clip to the image boundary
        els = ['ref_bbox_fields', 'bbox_fields'] if 'ref_bbox_fields' in results else ['bbox_fields']
        for el in els:
            for key in results.get(el, []):
                bbox_offset = np.array(
                        [offset_w, offset_h, offset_w, offset_h],
                        dtype=np.float32)
                bboxes = results[key] - bbox_offset
                bboxes[:, 0::2] = np.clip(
                        bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(
                        bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        els = ['ref_bboxes', 'gt_bboxes'] if 'ref_bboxes' in results else ['gt_bboxes']
        for el in els:
            if el in results:
                gt_bboxes = results[el]
                valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                    gt_bboxes[:, 3] > gt_bboxes[:, 1])
                # if no gt bbox remains after cropping, just skip this image
                if not np.any(valid_inds):
                    return None
                results[el] = gt_bboxes[valid_inds, :]
                ell = el.replace('_bboxes','_labels')
                if ell in results:
                    results[ell] = results[ell][valid_inds]
                #### filter gt_obj_ids just like gt_labes.
                elo = el.replace('_bboxes','_obj_ids')
                if elo in results:
                    results[elo] = results[elo][valid_inds]
                # filter and crop the masks
                elm = el.replace('_bboxes','_masks')
                if elm in results:
                    valid_gt_masks = []
                    for i in np.where(valid_inds)[0]:
                        gt_mask = results[elm][i][
                                crop_y1:crop_y2, crop_x1:crop_x2]
                        valid_gt_masks.append(gt_mask)
                    results[elm] = valid_gt_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegResizeFlipCropPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. crop
    4. pad
    5. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        if isinstance(scale_factor, list):
            self.scale_factor = scale_factor[0]
            self.another_scale = scale_factor[-1]
        else:
            self.scale_factor = scale_factor
            self.another_scale = None

    def __call__(self, results):
        els = (['ref_semantic_seg', 'gt_semantic_seg']
                if 'ref_semantic_seg' in results
                else ['gt_semantic_seg'])
        if 'gt_semantic_seg' not in results:
            els = []
        for el in els:
            if results['keep_ratio']:
                gt_seg = mmcv.imrescale(
                    results[el],
                    results['scale'],
                    interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[el],
                    results['scale'],
                    interpolation='nearest')
            if results['flip']:
                gt_seg = mmcv.imflip(gt_seg).copy()
            if 'crop_coords' in results:
                crds = results['crop_coords']
                gt_seg = gt_seg[crds[0]:crds[1], crds[2]:crds[3]]
            if gt_seg.shape != results['pad_shape'][:2]:
                # raise ValueError('gt_seg shape does not match with pad_shape')
                gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
            if self.scale_factor != 1:
                gt_seg = mmcv.imrescale(
                    gt_seg, self.scale_factor, interpolation='nearest')
            results[el] = gt_seg
            if self.another_scale is not None:
                gt_seg_Nx = mmcv.imrescale(
                    gt_seg, self.another_scale, interpolation='nearest')
                results[el+'_Nx'] = gt_seg_Nx
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)



@PIPELINES.register_module
class ImgResizeFlipNormCropPad(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. normalize
    4. crop
    5. pad

    Args:
        scale_factor (float): The scale factor of the final output.
    """
    def __init__(self):
        pass

    def single_call(self, results, img_ref):
        if results['keep_ratio']:
            img_ref = mmcv.imrescale(
                img_ref, results['scale'], return_scale=False)
        else:
            img_ref = mmcv.imresize(
                img_ref, results['scale'], return_scale=False)
        if results['flip']:
            img_ref = mmcv.imflip(img_ref)
        if results['img_norm_cfg']:
            img_norm_cfg = results['img_norm_cfg']
            img_ref = mmcv.imnormalize(
                    img_ref, img_norm_cfg['mean'],
                    img_norm_cfg['std'],
                    img_norm_cfg['to_rgb'])
        if 'crop_coords' in results:
            crds = results['crop_coords']
            img_ref = img_ref[crds[0]:crds[1], crds[2]:crds[3], :]
        if img_ref.shape != results['pad_shape']:
            img_ref = mmcv.impad(img_ref, results['pad_shape'][:2])
        return img_ref

    def __call__(self, results):
        if isinstance(results['ref_img'], list):
            results_img_ref = []
            for img_ref in results['ref_img']:
                results_img_ref.append(
                        self.single_call(results, img_ref))
            results['ref_img'] = results_img_ref
        else:
            results['ref_img'] = self.single_call(
                    results, results['ref_img'])
        return results


    def __repr__(self):

        return self.__class__.__name__

@PIPELINES.register_module
class ColorJitter(object):
    """
    Uses torchvision color jitter
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.jitter_transform = torchvision_transforms.ColorJitter(
            brightness, contrast, saturation, hue
        )
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation

        self.transform = torchvision_transforms.RandomApply([self.jitter_transform], p=0.8)

    def __call__(self, results):
        keys = ['img']

        if 'ref_img' in results:
            keys.append('ref_img')

        for el in keys:
            pil_img = Image.fromarray(np.asarray(results[el] * 255.0, dtype=np.uint8))
            results[el] = np.array(self.transform(pil_img)) / 255.0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness={}, contrast={}, '
                     'saturation={}, hue={})').format(
                         self.brightness, self.contrast,
                         self.saturation, self.hue)
        return repr_str

@PIPELINES.register_module
class RandGrayscale(object):
    """
    Uses torchvision color jitter
    """
    def __init__(self, p=0.1):
        self.grayscale_transform = torchvision_transforms.RandomGrayscale(p=p)
        self.p = p

    def __call__(self, results):
        keys = ['img']

        if 'ref_img' in results:
            keys.append('ref_img')

        for el in keys:
            pil_img = Image.fromarray(np.asarray(results[el] * 255.0, dtype=np.uint8))
            results[el] = np.array(self.grayscale_transform(pil_img)) / 255.0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(Grayscale Random prob={}, ').format(self.p)
        return repr_str

@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        if 'ref_img' in results:
            ref_img = results['ref_img']
        else:
            ref_img = None

        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta
            if ref_img is not None:
                ref_img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
                if ref_img is not None:
                    ref_img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)
        if ref_img is not None:
            ref_img = mmcv.bgr2hsv(ref_img)

        # random saturation
        if random.randint(2):
            random_saturation = random.uniform(self.saturation_lower,
                                               self.saturation_upper)

            img[..., 1] *= random_saturation
            if ref_img is not None:
                ref_img[..., 1] *= random_saturation


        # random hue
        if random.randint(2):
            random_hue = random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0] += random_hue
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
            if ref_img is not None:
                ref_img[..., 0] += random_hue
                ref_img[..., 0][ref_img[..., 0] > 360] -= 360
                ref_img[..., 0][ref_img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)
        if ref_img is not None:
            ref_img = mmcv.hsv2bgr(ref_img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha
                if ref_img is not None:
                    ref_img *= alpha

        # randomly swap channels
        if random.randint(2):
            random_permute = random.permutation(3)
            img = img[..., random_permute]
            if ref_img is not None:
                ref_img = ref_img[..., random_permute]


        results['img'] = img
        if ref_img is not None:
            results['ref_img'] = ref_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, results):
        if random.randint(2):
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = expand_gt_masks
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={})'.format(
            self.mean, self.to_rgb, self.ratio_range)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels, obj_ids = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels', 'gt_obj_ids')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]
                obj_ids = obj_ids[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels
                results['gt_obj_ids'] = obj_ids

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = [
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str
