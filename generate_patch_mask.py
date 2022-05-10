import argparse
import os
import pickle
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from adv_patch_bench.attacks.utils import get_object_and_mask_from_numpy


def generate_mask(
    obj_numpy: np.ndarray,
    obj_size_px: Union[int, Tuple[int, int]],
    obj_width_inch: float,
) -> torch.Tensor:

    # Get height to width ratio of the object
    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]
    if isinstance(obj_size_px, int):
        obj_size_px = (round(obj_size_px * h_w_ratio), obj_size_px)
    obj_w_inch = obj_width_inch
    obj_h_inch = h_w_ratio * obj_w_inch
    obj_h_px, obj_w_px = obj_size_px
    patch_mask = torch.zeros((1, ) + obj_size_px)

    # EDIT: User defines from this point on
    patch_size_inch = 10
    if isinstance(patch_size_inch, int):
        patch_size_inch = (patch_size_inch, patch_size_inch)
    patch_h_inch, patch_w_inch = patch_size_inch
    patch_h_px = round(patch_h_inch / obj_h_inch * obj_h_px)
    patch_w_px = round(patch_w_inch / obj_w_inch * obj_w_px)

    # Define patch location and size
    # Example: 10x10-inch patch in the middle of 36x36-inch sign
    # (1) 1 square (bottom)
    mid_height, mid_width = obj_h_px // 2, obj_w_px // 2
    patch_x_shift = 0
    shift_inch = (obj_h_inch - patch_h_inch) / 2
    patch_y_shift = round(shift_inch / obj_h_inch * obj_h_px)
    patch_x_pos = mid_width + patch_x_shift
    patch_y_pos = mid_height + patch_y_shift
    hh, hw = patch_h_px // 2, patch_w_px // 2

    # # (2) 2 rectangles
    # patch_width = 20
    # patch_height = 6
    # h = round(patch_height / 36 / 2 * obj_size[0])
    # w = round(patch_width / 36 / 2 * obj_size[1])
    # offset_h = round(28 / 128 * obj_size[0])
    # offset_w = obj_size[1] // 2
    # patch_mask[:, offset_h - h:offset_h + h, offset_w - w:offset_w + w] = 1
    # patch_mask[:, obj_size[0] - offset_h - h:obj_size[0] - offset_h + h, offset_w - w:offset_w + w] = 1

    patch_mask[:, patch_y_pos - hh:patch_y_pos + hh, patch_x_pos - hw:patch_x_pos + hw] = 1
    return patch_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Generation', add_help=False)
    parser.add_argument('--syn-obj-path', type=str, default='', help='path to synthetic image of the object')
    parser.add_argument('--obj-size', type=int, required=True)
    parser.add_argument('--patch-size', type=int, required=True)
    parser.add_argument('--patch-name', type=str, default=None)
    parser.add_argument('--save-mask', action='store_true')
    args = parser.parse_args()
    # FIXME: get obj size in inch
    obj_numpy = np.array(Image.open(args.syn_obj_path).convert('RGBA')) / 255
    patch_mask = generate_mask(obj_numpy, **vars(args))

    if args.save_mask and args.patch_name is not None:
        save_dir = './masks/'
        patch_mask = patch_mask.permute(1, 2, 0)
        os.makedirs(save_dir, exist_ok=True)
        print(f'=> Saving patch mask in {save_dir}...')

        plt.imshow(patch_mask, cmap='gray')
        mask_save_path = os.path.join(save_dir, f'{args.patch_name}.png')
        plt.savefig(mask_save_path, bbox_inches='tight')
        plt.close()

        obj, _ = get_object_and_mask_from_numpy(obj_numpy, patch_mask.shape[1:])
        plot_image = obj * (1 - patch_mask)
        plt.imshow(plot_image)
        mask_save_path = os.path.join(save_dir, f'{args.patch_name}_mask_on_sign.jpg')
        plt.savefig(mask_save_path, bbox_inches='tight')
        plt.close()
