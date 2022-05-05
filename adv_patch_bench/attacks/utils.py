import ast
import pickle
from argparse import Namespace
from typing import List, Tuple

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

    if args.synthetic_eval:
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
