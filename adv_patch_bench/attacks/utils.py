import ast
import pickle
from argparse import Namespace
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as T
from adv_patch_bench.utils.image import (mask_to_box, pad_and_center,
                                         prepare_obj)
from kornia import augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import resize


def prep_synthetic_eval(
    args: Namespace,
    img_size: Tuple[int, int],
    label_names: List[str],
    device: str = 'cuda',
):
    # Testing with synthetic signs
    # Add synthetic label to the label names at the last position
    syn_sign_class = len(label_names)
    label_names[syn_sign_class] = 'synthetic'
    nc += 1
    # Set up random transforms for synthetic sign
    obj_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, return_transform=True)
    mask_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, resample=Resample.NEAREST)

    # Load synthetic object/sign from file
    _, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    obj_size = patch_mask.shape[1]
    obj, obj_mask = prepare_obj(args.syn_obj_path, img_size, (obj_size, obj_size))
    obj = obj.to(device)
    obj_mask = obj_mask.to(device).unsqueeze(0)

    return obj, obj_mask, obj_transforms, mask_transforms, syn_sign_class


def prep_attack(
    args: Namespace,
    img_size: Tuple[int, int],
    device: str = 'cuda',
):
    # Load patch from a pickle file if specified
    # TODO: make script to generate dummy patch
    adv_patch, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    obj_size = patch_mask.shape[1]
    patch_mask = patch_mask.to(device)
    patch_height, patch_width = adv_patch.shape[1:]
    patch_loc = mask_to_box(patch_mask)

    if args.attack_type == 'debug':
        # Load 'arrow on checkboard' patch if specified (for debug)
        adv_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
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
    print(df.shape)
    print(df.groupby(by=['final_shape']).count())

    if args.synthetic:
        # Adv patch and mask have to be made compatible with random
        # transformation for synthetic signs
        _, patch_mask = pad_and_center(None, patch_mask, img_size, (obj_size, obj_size))
        patch_loc = mask_to_box(patch_mask)
        # Pad adv patch
        pad_size = [
            patch_loc[1],  # left
            patch_loc[0],  # right
            img_size[1] - patch_loc[1] - patch_loc[3],  # top
            img_size[0] - patch_loc[0] - patch_loc[2],  # bottom
        ]
        adv_patch = T.pad(adv_patch, pad_size)
        patch_mask = patch_mask.to(device)
        adv_patch = adv_patch.to(device)

    return df, adv_patch, patch_mask, patch_loc


def apply_synthetic_sign(
    image: torch.Tensor,
    adv_patch: torch.Tensor,
    patch_mask: torch.Tensor,
    syn_obj: torch.Tensor,
    syn_obj_mask: torch.Tensor,
    obj_transforms: Any,
    mask_transforms: Any,
    syn_sign_class: int,
    device: str = 'cuda',
    use_attack: bool = True,
):
    _, h, w = image.shape
    if use_attack:
        adv_obj = patch_mask * adv_patch + (1 - patch_mask) * syn_obj
    else:
        adv_obj = syn_obj
    adv_obj, tf_params = obj_transforms(adv_obj)
    adv_obj.clamp_(0, 1)
    o_mask = mask_transforms.apply_transform(
        syn_obj_mask, None, transform=tf_params.to(device))

    # get top left and bottom right points
    # TODO: can we use mask_to_box here?
    indices = np.where(o_mask.cpu()[0][0] == 1)
    x_min, x_max = min(indices[1]), max(indices[1])
    y_min, y_max = min(indices[0]), max(indices[0])

    # Since we paste a new synthetic sign on image, we have to add
    # in a new synthetic label/target to compute the metrics
    label = [
        image,
        syn_sign_class,
        (x_min + x_max) / (2 * w),
        (y_min + y_max) / (2 * h),
        (x_max - x_min) / w,
        (y_max - y_min) / h,
        1,
        -1
    ]
    targets = torch.cat((targets, torch.tensor(label).unsqueeze(0)))

    adv_img = o_mask * adv_obj + (1 - o_mask) * perturbed_image.to(device) / 255
    perturbed_image = adv_img.squeeze() * 255
    return perturbed_image


def get_object_and_mask_from_numpy(
    obj_numpy: np.ndarray,
    obj_size: Tuple[int, int],
    pad_size: Union[Tuple, List] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get object and its mask and resize to obj_size"""
    obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    obj = T.resize(obj, obj_size, antialias=True)
    if pad_size is not None:
        obj = T.pad(obj, pad_size)
    # obj = obj.permute(1, 2, 0)
    obj_mask = T.resize(obj_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
    if pad_size is not None:
        obj_mask = T.pad(obj_mask, pad_size)
    # obj_mask = obj_mask.permute(1, 2, 0)
    return obj, obj_mask
