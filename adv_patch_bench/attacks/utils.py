from __future__ import annotations

import ast
from copy import deepcopy
import os
import pickle
from argparse import Namespace
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as T
from adv_patch_bench.utils.image import (mask_to_box, resize_and_center,
                                         prepare_obj)
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from hparams import PATH_SYN_OBJ
from kornia import augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import resize


def coerce_rank(x, ndim):
    if x.ndim == ndim:
        return x

    ndim_diff = ndim - x.ndim
    if ndim_diff < 0:
        for _ in range(-ndim_diff):
            x.squeeze_(0)
        if x.ndim != ndim:
            raise ValueError('Can\'t coerce rank.')
        return x
    
    for _ in range(ndim_diff):
        x.unsqueeze_(0)
    if x.ndim != ndim:
        raise ValueError('Can\'t coerce rank.')
    return x

def prep_synthetic_eval(
    args: Namespace,
    img_size: Tuple[int, int],
    label_names: List[str],
    transform_prob: float = 1.,
    device: str = 'cuda',
):
    # Testing with synthetic signs
    # Add synthetic label to the label names at the last position
    # FIXME
    syn_sign_class = len(label_names)
    # label_names[syn_sign_class] = 'synthetic'
    # Set up random transforms for synthetic sign
    # obj_transforms = K.RandomAffine(20, translate=(0.45, 0.45), p=1.0,
    #                                 return_transform=True, scale=(0.25, 0.5))
    # mask_transforms = K.RandomAffine(20, translate=(0.45, 0.45), p=1.0,
    #                                  resample=Resample.NEAREST, scale=(0.25, 0.5))
    # obj_transforms = K.RandomAffine(20, translate=(0.45, 0.45), p=1.0, return_transform=True, scale=(0.3, 0.6))
    # mask_transforms = K.RandomAffine(20, translate=(0.45, 0.45), p=1.0, resample=Resample.NEAREST, scale=(0.3, 0.6))
    obj_transforms = K.RandomAffine(
        30, translate=(0.45, 0.45), p=transform_prob, return_transform=True)
    mask_transforms = K.RandomAffine(
        30, translate=(0.45, 0.45), p=transform_prob, resample=Resample.NEAREST)

    # Load synthetic object/sign from file
    _, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    obj_size = patch_mask.shape[-2]
    obj, obj_mask = prepare_obj(args.syn_obj_path, img_size, (obj_size, obj_size))
    obj = obj.to(device)
    obj_mask = obj_mask.to(device)  # .unsqueeze(0)
    return obj, obj_mask, obj_transforms, mask_transforms, syn_sign_class


def prep_attack(
    args: Namespace,
    img_size: Tuple[int, int],
    device: str = 'cuda',
):
    # Load patch from a pickle file if specified
    # TODO: make script to generate dummy patch
    adv_patch, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    adv_patch = coerce_rank(adv_patch, 3)
    patch_mask = coerce_rank(patch_mask, 3)
    patch_mask = patch_mask.to(device)
    patch_height, patch_width = adv_patch.shape[-2:]
    patch_loc = mask_to_box(patch_mask)

    if args.attack_type == 'debug':
        # Load 'arrow on checkboard' patch if specified (for debug)
        adv_patch = torchvision.io.read_image(
            os.path.join(PATH_SYN_OBJ, 'debug.png')).float()[:3, :, :] / 255
        adv_patch = resize(adv_patch, (patch_height, patch_width))
    elif args.attack_type == 'random':
        # Patch with uniformly random pixels
        adv_patch = torch.rand(3, patch_height, patch_width)

    # load csv file containing target points for transform
    df = pd.read_csv(args.tgt_csv_filepath)
    # converts 'tgt_final' from string to list format
    df['tgt_final'] = df['tgt_final'].apply(ast.literal_eval)
    # exclude shapes to which we do not apply the transform to
    df = df[df['final_shape'] != 'other-0.0-0.0']
    # print(df.shape)
    # print(df.groupby(by=['final_shape']).count())

    if args.synthetic:
        # Adv patch and mask have to be made compatible with random
        # transformation for synthetic signs
        h_w_ratio = patch_mask.shape[-2] / patch_mask.shape[-1]
        obj_size = patch_mask.shape[-2]
        if isinstance(obj_size, int):
            obj_size_px = (round(obj_size * h_w_ratio), obj_size)
        patch_mask = resize_and_center(
            patch_mask, img_size, obj_size_px, is_binary=True)
        patch_loc = mask_to_box(patch_mask)
        adv_patch = resize_and_center(
            adv_patch, img_size, obj_size_px, is_binary=False)
        patch_mask = patch_mask.to(device)
        adv_patch = adv_patch.to(device)
    else:
        obj_size = patch_mask.shape[-2:]
        adv_patch = resize_and_center(
            adv_patch, None, obj_size, is_binary=False)

    assert adv_patch.ndim == patch_mask.ndim == 3
    assert adv_patch.shape[-2:] == patch_mask.shape[-2:], \
        f'{adv_patch.shape} does not match {patch_mask.shape}.'

    return df, adv_patch, patch_mask, patch_loc


def apply_synthetic_sign(
    image: torch.FloatTensor,
    targets: Any,
    image_id: int,
    adv_patch: torch.FloatTensor,
    patch_mask: torch.FloatTensor,
    syn_obj: torch.FloatTensor,
    syn_obj_mask: torch.FloatTensor,
    obj_transforms: Any,
    mask_transforms: Any,
    syn_sign_class: int,
    device: str = 'cuda',
    use_attack: bool = True,
    return_target: bool = True,
    is_detectron: bool = False,
    other_sign_class: int = None
):
    adv_patch = coerce_rank(adv_patch, 4)
    patch_mask = coerce_rank(patch_mask, 4)
    syn_obj = coerce_rank(syn_obj, 4)
    image = coerce_rank(image, 4)
    h, w = image.shape[-2:]
    
    if use_attack:
        adv_obj = patch_mask * adv_patch + (1 - patch_mask) * syn_obj
    else:
        adv_obj = syn_obj
    adv_obj, tf_params = obj_transforms(adv_obj)
    adv_obj.clamp_(0, 1)
    o_mask = mask_transforms.apply_transform(
        syn_obj_mask, None, transform=tf_params.to(device))
    image = image.to(device).view_as(adv_obj) / 255
    adv_img = o_mask * adv_obj + (1 - o_mask) * image
    perturbed_image = adv_img.squeeze() * 255

    if not return_target:
        return perturbed_image

    # get top left and bottom right points
    bbox = mask_to_box(o_mask.cpu()[0][0] == 1)
    y_min, x_min, h_obj, w_obj = [b.item() for b in bbox]

    # Since we paste a new synthetic sign on image, we have to add
    # in a new synthetic label/target to compute the metrics
    if is_detectron:
        assert isinstance(targets, dict) and isinstance(other_sign_class, int)
        targets = deepcopy(targets)
        annotations = targets['annotations']
        instances = targets['instances']
        for anno in annotations:
            anno['category_id'] = other_sign_class
        new_bbox = [x_min, y_min, x_min + w_obj, y_min + h_obj]
        new_anno = {
            'bbox': new_bbox,
            'category_id': syn_sign_class,
            'bbox_mode': annotations[0]['bbox_mode'],
        }
        annotations.append(new_anno)
        # Create new gt instances
        new_instances = Instances(instances.image_size)
        new_gt_classes = torch.zeros(len(instances) + 1, dtype=torch.int64)
        new_gt_classes[:-1] = other_sign_class
        new_gt_classes[-1] = syn_sign_class
        new_bbox = torch.tensor(new_bbox, dtype=torch.float32)
        new_instances.gt_boxes = Boxes.cat([instances.gt_boxes,
                                            Boxes(new_bbox[None, :])])
        new_instances.gt_classes = new_gt_classes
        targets['instances'] = new_instances
        return perturbed_image, targets

    # Add new target for YOLO
    assert isinstance(targets, torch.Tensor)
    label = [
        image_id,
        syn_sign_class,
        (x_min + w_obj / 2) / w,  # relative center x
        (y_min + h_obj / 2) / h,  # relative center y
        w_obj / w,  # relative width
        h_obj / h,  # relative height
        -1
    ]
    targets = torch.cat((targets, torch.tensor(label).unsqueeze(0)))
    return perturbed_image, targets


def get_object_and_mask_from_numpy(
    obj_numpy: np.ndarray,
    obj_size: Tuple[int, int],
    img_size: Tuple[int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get object and its mask and resize to obj_size."""
    obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    if img_size is not None:
        obj = resize_and_center(obj, img_size, obj_size, is_binary=False)
        obj_mask = resize_and_center(obj_mask, img_size, obj_size, is_binary=True)
    return obj, obj_mask
