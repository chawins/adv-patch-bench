import argparse
import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

from adv_patch_bench.attacks.utils import get_object_and_mask_from_numpy
from adv_patch_bench.utils.argparse import parse_dataset_name
from adv_patch_bench.utils.image import get_obj_width
from hparams import LABEL_LIST


def gen_mask_10x10(obj_h_inch, obj_w_inch, obj_h_px, obj_w_px):
    patch_size_inch = (10, 10)
    patch_mask = torch.zeros((1, obj_h_px, obj_w_px))
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
    patch_mask[
        :,
        patch_y_pos - hh : patch_y_pos + hh,
        max(0, patch_x_pos - hw) : patch_x_pos + hw,
    ] = 1
    return patch_mask


def gen_mask_10x20(obj_h_inch, obj_w_inch, obj_h_px, obj_w_px, num_patches=1):
    patch_size_inch = (10, 20)
    patch_mask = torch.zeros((1, obj_h_px, obj_w_px))
    patch_h_inch, patch_w_inch = patch_size_inch
    patch_h_px = round(patch_h_inch / obj_h_inch * obj_h_px)
    patch_w_px = round(patch_w_inch / obj_w_inch * obj_w_px)

    # Define patch location and size
    mid_height, mid_width = obj_h_px // 2, obj_w_px // 2
    shift_inch = (obj_h_inch - patch_h_inch) / 2
    patch_y_shift = round(shift_inch / obj_h_inch * obj_h_px)
    patch_x_pos = mid_width
    patch_y_pos = mid_height + patch_y_shift
    hh, hw = patch_h_px // 2, patch_w_px // 2
    # Bottom patch
    patch_mask[
        :,
        patch_y_pos - hh : patch_y_pos + hh,
        max(0, patch_x_pos - hw) : patch_x_pos + hw,
    ] = 1

    if num_patches == 2:
        # Top patch
        patch_y_pos = mid_height - patch_y_shift
        top_edge = min(patch_y_pos + hh, obj_h_px)
        bot_edge = obj_h_px - patch_h_px
        patch_mask[
            :,
            bot_edge:top_edge,
            max(0, patch_x_pos - hw) : patch_x_pos + hw,
        ] = 1

    return patch_mask


def generate_mask(
    mask_name: str,
    obj_numpy: np.ndarray,
    obj_size_px: Union[int, Tuple[int, int]],
    obj_width_inch: float,
) -> torch.Tensor:
    assert mask_name in ("10x10", "10x20", "2_10x20")

    # Get height to width ratio of the object
    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]
    if isinstance(obj_size_px, int):
        obj_size_px = (round(obj_size_px * h_w_ratio), obj_size_px)
    obj_w_inch = obj_width_inch
    obj_h_inch = h_w_ratio * obj_w_inch
    obj_h_px, obj_w_px = obj_size_px

    if mask_name == "10x10":
        patch_mask = gen_mask_10x10(obj_h_inch, obj_w_inch, obj_h_px, obj_w_px)
    elif "10x20" in mask_name:
        num_patches = 2 if "2_" in mask_name else 1
        patch_mask = gen_mask_10x20(
            obj_h_inch, obj_w_inch, obj_h_px, obj_w_px, num_patches=num_patches
        )

    # (3) cover the whole sign
    # patch_mask = torch.ones_like(patch_mask)

    return patch_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask Generation", add_help=False
    )
    parser.add_argument(
        "--syn-obj-path",
        type=str,
        default="",
        help="path to synthetic image of the object",
    )
    parser.add_argument("--obj-size", type=int, required=True)
    # parser.add_argument('--patch-size', type=int, required=True)
    parser.add_argument("--patch-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--obj-class",
        type=int,
        default=-1,
        help="class of object to attack (-1: all classes)",
    )
    parser.add_argument("--save-mask", action="store_true")
    args = parser.parse_args()
    parse_dataset_name(args)

    # Get obj size in inch based on given obj class
    class_names = LABEL_LIST[args.dataset]
    obj_width_inch = get_obj_width(args.obj_class, class_names)
    obj_numpy = np.array(Image.open(args.syn_obj_path).convert("RGBA")) / 255
    patch_mask = generate_mask(obj_numpy, args.obj_size, obj_width_inch)

    if args.save_mask and args.patch_name is not None:
        save_dir = "./masks/"
        os.makedirs(save_dir, exist_ok=True)
        mask_save_path = os.path.join(save_dir, f"{args.patch_name}.png")
        print(f"=> Saving patch mask to {mask_save_path}")
        torchvision.utils.save_image(patch_mask[0], mask_save_path)

        # plt.imshow(patch_mask[0], cmap='gray')
        # plt.savefig(mask_save_path, bbox_inches='tight')
        # plt.close()

        obj, _ = get_object_and_mask_from_numpy(obj_numpy, patch_mask.shape[1:])
        plot_image = obj * (1 - patch_mask)
        plt.imshow(plot_image.permute(1, 2, 0).clamp(0, 1))
        mask_save_path = os.path.join(
            save_dir, f"{args.patch_name}_on_sign.jpg"
        )
        plt.savefig(mask_save_path, bbox_inches="tight")
        plt.close()
