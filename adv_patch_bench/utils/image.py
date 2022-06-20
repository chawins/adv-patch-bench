import json
import os
from os.path import join
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as T
from PIL import Image


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

def load_annotation(label_path, image_key):
    with open(join(label_path, '{:s}.json'.format(image_key)), 'r') as fid:
        anno = json.load(fid)
    return anno


def get_image_files(path):
    image_keys = []
    for entry in os.scandir(path):
        if (entry.path.endswith('.jpg') or entry.path.endswith('.png')) and entry.is_file():
            image_keys.append(entry.name)
    return image_keys


def pad_to_size(img: torch.Tensor, size: Tuple[int, int]):
    img_size = img.shape[-2:]
    pad_size = [
        (size[1] - img_size[1]) // 2,  # left
        (size[0] - img_size[0]) // 2,  # top
        size[1] - img_size[1] - (size[1] - img_size[1]) // 2,  # right
        size[0] - img_size[0] - (size[0] - img_size[0]) // 2,  # bottom
    ]
    bbox = [pad_size[0], pad_size[1], size[1] - pad_size[2], size[0] - pad_size[3]]
    return pad_image(img, pad_size), bbox


def pad_image(img, pad_size=0.1, pad_mode='constant', return_pad_size=False):
    pad_size = int(max(height, width) * pad_size) if isinstance(pad_size, float) else pad_size
    if isinstance(img, np.ndarray):
        height, width = img.shape[0], img.shape[1]
        pad_size_tuple = ((pad_size, pad_size), (pad_size, pad_size)) + ((0, 0), ) * (img.ndim - 2)
        img_padded = np.pad(img, pad_size_tuple, mode=pad_mode)
    else:
        height, width = img.shape[img.ndim - 2], img.shape[img.ndim - 1]
        img_padded = T.pad(img, pad_size, padding_mode=pad_mode)
    if return_pad_size:
        return img_padded, pad_size
    return img_padded


def crop(img_padded, mask, pad, offset):
    """Crop a square bounding box of an object with a correcponding 
    segmentation mask from an (padded) image.

    Args:
        img_padded (np.ndarray): Image of shape (height, width, channels)
        mask (np.ndarray): A boolean mask of shape (height, width)
        pad (float): Extra padding for the bounding box from each endpoint of
            the mask as a ratio of `max(height, width)` of the object
        offset (int): Offset in case the given image is already padded

    Returns:
        np.ndarray: Cropped image
    """
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    # Make sure that bounding box is square
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    xpad, ypad = int((size - width) / 2), int((size - height) / 2)
    extra_obj_pad = int(pad * size)
    size += 2 * extra_obj_pad
    xmin += offset - xpad - extra_obj_pad
    ymin += offset - ypad - extra_obj_pad
    xmax, ymax = xmin + size, ymin + size
    return img_padded[ymin:ymax, xmin:xmax]


def img_numpy_to_torch(img):
    assert img.ndim == 3 and isinstance(img, np.ndarray)
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.


def get_box(mask, pad):
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    # Make sure that bounding box is square
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    xpad, ypad = int((size - width) / 2), int((size - height) / 2)
    extra_obj_pad = int(pad * size)
    size += 2 * extra_obj_pad
    xmin -= xpad + extra_obj_pad
    ymin -= ypad + extra_obj_pad
    xmax, ymax = xmin + size, ymin + size
    return ymin, ymax, xmin, xmax


def draw_from_contours(img, contours, color=[0, 0, 255, 255]):
    if not isinstance(contours, list):
        contours = [contours]
    for contour in contours:
        if contour.ndim == 3:
            contour_coord = (contour[:, 0, 1], contour[:, 0, 0])
        else:
            contour_coord = (contour[:, 1], contour[:, 0])
        img[contour_coord] = color
    return img


def letterbox(im, new_shape=(640, 640), color=114, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = T.resize(im, new_unpad)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = T.pad(im, [left, top, right, bottom], fill=color)
    return im, ratio, (dw, dh)


def mask_to_box(mask):
    """Get a binary mask and returns a bounding box: y0, x0, h, w"""
    # if mask.ndim == 3:
    #     mask = mask.squeeze(0)
    # assert mask.ndim == 2
    mask = coerce_rank(mask, 2)
    if mask.sum() <= 0:
        raise ValueError('mask is all zeros!')
    y, x = torch.where(mask)
    y_min, x_min = y.min(), x.min()
    return y_min, x_min, y.max() - y_min, x.max() - x_min


def prepare_obj(obj_path, img_size, obj_size, interp):
    """Load image of an object and place it in the middle of an image tensor of
    size `img_size`. The object is also resized to `obj_size`.

    Args:
        obj_path (str): Path to image to load
        img_size (tuple): Size of image to place object on: (height, width)
        obj_size (tuple): Size of object in the image: (height, width)

    Returns:
        torch.Tensor, torch.Tensor: Object and its mask
    """
    obj_numpy = np.array(Image.open(obj_path).convert('RGBA')) / 255
    obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    obj = resize_and_center(
        obj, img_size, obj_size, is_binary=False, interp=interp)
    obj_mask = resize_and_center(obj_mask, img_size, obj_size, is_binary=True)
    obj.unsqueeze_(0)
    obj_mask.unsqueeze_(0)
    assert obj.ndim == obj_mask.ndim == 4
    assert obj.shape[-2:] == obj_mask.shape[-2:]
    return obj, obj_mask


def resize_and_center(obj: torch.Tensor,
                      img_size: Tuple[int, int],
                      obj_size: Tuple[int, int],
                      is_binary: bool = False,
                      interp: str = 'bicubic'):
    """
    Resize object to obj_size and then place it in the middle of zero 
    background.
    """
    if obj_size is not None:
        if is_binary or interp == 'nearest':
            interp = T.InterpolationMode.NEAREST
        elif interp == 'bicubic':
            interp = T.InterpolationMode.BICUBIC
        elif interp == 'bilinear':
            interp = T.InterpolationMode.BILINEAR
        else:
            raise NotImplementedError('interp not supported.')
        obj = T.resize(obj, obj_size, interpolation=interp)

    if img_size is not None:
        # left, top, right, bottom
        # left = (img_size[1] - obj_size[1]) // 2
        # top = (img_size[0] - obj_size[0]) // 2
        left = torch.div(img_size[1] - obj_size[1], 2, rounding_mode='trunc')
        top = torch.div(img_size[0] - obj_size[0], 2, rounding_mode='trunc')
        pad_size = [
            left,  # left
            top,  # top
            img_size[1] - obj_size[1] - left,  # right
            img_size[0] - obj_size[0] - top,  # bottom
        ]
        obj = T.pad(obj, pad_size)

    return obj


def get_obj_width(obj_class, class_names):
    return float(class_names[obj_class].split('-')[1]) * 0.0393701
