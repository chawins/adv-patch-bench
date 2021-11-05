import json
import os
from os.path import join

import numpy as np
import torch
import torchvision.transforms.functional as TF


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


def pad_image(img, pad_size=0.1, pad_mode='constant', return_pad_size=False):
    if isinstance(img, np.ndarray):
        height, width = img.shape[0], img.shape[1]
        pad_size = int(max(height, width) * pad_size) if isinstance(pad_size, float) else pad_size
        pad_size_tuple = ((pad_size, pad_size), (pad_size, pad_size)) + ((0, 0), ) * (img.ndim - 2)
        img_padded = np.pad(img, pad_size_tuple, mode=pad_mode)
    else:
        height, width = img.shape[img.ndim - 2], img.shape[img.ndim - 1]
        pad_size = int(max(height, width) * pad_size) if isinstance(pad_size, float) else pad_size
        img_padded = TF.pad(img, pad_size, padding_mode=pad_mode)
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