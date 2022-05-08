import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image


def generate_mask(syn_obj_path, obj_size, patch_size, patch_name, save_mask=False):
    obj_numpy = np.array(Image.open(syn_obj_path).convert('RGBA')) / 255

    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]
    if isinstance(obj_size, int):
        obj_size = (round(obj_size * h_w_ratio), obj_size)

    obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    obj = T.resize(obj, obj_size, antialias=True)
    obj = obj.permute(1, 2, 0)
    obj_mask = T.resize(obj_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
    obj_mask = obj_mask.permute(1, 2, 0)

    # Define patch location and size
    patch_mask = torch.zeros((1, ) + obj_size)
    # Example: 10x10-inch patch in the middle of 36x36-inch sign
    # (1) 1 square
    mid_height = obj_size[0] // 2
    mid_width = obj_size[1] // 2

    patch_x_shift = 0
    patch_y_shift = round(40 / 128 * obj_size[0])

    patch_x_pos = mid_width + patch_x_shift
    patch_y_pos = mid_height + patch_y_shift

    h = round(patch_size / 36 / 2 * obj_size[0])
    w = round(patch_size / 36 / 2 * obj_size[1])

    # # (2) 2 rectangles
    # patch_width = 20
    # patch_height = 6
    # h = round(patch_height / 36 / 2 * obj_size[0])
    # w = round(patch_width / 36 / 2 * obj_size[1])
    # offset_h = round(28 / 128 * obj_size[0])
    # offset_w = obj_size[1] // 2
    # patch_mask[:, offset_h - h:offset_h + h, offset_w - w:offset_w + w] = 1
    # patch_mask[:, obj_size[0] - offset_h - h:obj_size[0] - offset_h + h, offset_w - w:offset_w + w] = 1

    patch_mask[:, patch_y_pos - h:patch_y_pos + h, patch_x_pos - w:patch_x_pos + w] = 1
    patch_mask = patch_mask.permute(1, 2, 0)

    if save_mask and patch_name is not None:
        save_dir = './masks/'
        os.makedirs(save_dir, exist_ok=True)

        plt.imshow(patch_mask, cmap='gray')
        mask_save_path = os.path.join(save_dir, f'{patch_name}.png')
        plt.savefig(mask_save_path, bbox_inches='tight')
        plt.close()

        plot_image = obj * (1 - patch_mask)
        plt.imshow(plot_image)
        mask_save_path = os.path.join(save_dir, f'{patch_name}_mask_on_sign.jpg')
        plt.savefig(mask_save_path, bbox_inches='tight')
        plt.close()

    # pickle.dump(patch_mask, open(mask_save_path, 'wb'))
    patch_mask = patch_mask.permute(2, 0, 1)
    return patch_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Generation', add_help=False)
    parser.add_argument('--syn-obj-path', type=str, default='', help='path to synthetic image of the object')
    parser.add_argument('--obj-size', type=int, required=True)
    parser.add_argument('--patch-size', type=int, required=True)
    parser.add_argument('--patch-name', type=str, required=True)
    parser.add_argument('--save-mask', action='store_true')
    args = parser.parse_args()
    generate_mask(**vars(args))
