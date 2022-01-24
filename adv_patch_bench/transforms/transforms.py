import math

import cv2 as cv
import numpy as np


def gen_rect_mask(size, ratio=None):
    # ratio = height / width
    mask = np.zeros((size, size))
    height = ratio * size if ratio < 1 else size
    width = height / ratio
    if ratio > 1:
        pad = int((size - width) / 2)
        mask[:, pad:size - pad] = 1
        box = [[pad, 0], [size - pad, 0], [size - pad, size - 1], [pad, size - 1]]
    elif ratio < 1:
        pad = int((size - height) / 2)
        mask[pad:size - pad, :] = 1
        box = [[0, pad], [size - 1, pad], [size - 1, size - pad], [0, size - pad]]
    else:
        mask[:, :] = 1
        box = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    return mask, box

def gen_square_mask(size, ratio=None):
    # ratio = height / width
    mask = np.zeros((size, size))
    height = ratio * size if ratio < 1 else size
    width = height / ratio
    if ratio > 1:
        pad = int((size - width) / 2)
        mask[:, pad:size - pad] = 1
        box = [[pad, 0], [size - pad, 0], [size - pad, size - 1], [pad, size - 1]]
    elif ratio < 1:
        pad = int((size - height) / 2)
        mask[pad:size - pad, :] = 1
        box = [[0, pad], [size - 1, pad], [size - 1, size - pad], [0, size - pad]]
    else:
        mask[:, :] = 1
        box = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    return mask, box


def gen_diamond_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= mid) * (Y - X >= -mid) * (Y + X <= size + mid) * (Y - X <= mid)
    return mask, [[0, mid], [mid, 0], [size - 1, mid], [mid, size - 1]]


def gen_circle_mask(size, ratio=None):
    Y, X = np.ogrid[:size, :size]
    center = round(size / 2)
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= center
    return mask, [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]


def gen_triangle_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + 2 * X >= size) * (Y - 2 * X >= -size)
    return mask, [[mid, 0], [size - 1, size - 1], [0, size - 1]]


def gen_triangle_inverted_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y - 2 * X <= 0) * (Y + 2 * X <= 2 * size)
    return mask, [[0, 0], [size - 1, 0], [mid, size - 1]]


def gen_pentagon_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= mid) * (Y - X >= -mid)
    return mask, [[0, mid], [size - 1, mid], [size - 1, size - 1], [0, size - 1]]


def gen_octagon_mask(size, ratio=None):
    edge = round((2 - np.sqrt(2)) / 2 * size)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= edge) * (Y - X >= -(size - edge)) * (Y + X <= 2 * size - edge) * (Y + X <= size - edge)
    return mask, [[0, edge], [size - edge, 0], [size - 1, size - edge], [edge, size - 1]]


def gen_sign_mask(shape, size, ratio=None):
    return SHAPE_TO_MASK[shape](size, ratio=ratio)


SHAPE_TO_MASK = {
    'circle': gen_circle_mask,
    'triangle_inverted': gen_triangle_inverted_mask,
    'triangle': gen_triangle_mask,
    'rect': gen_rect_mask,
    'diamond': gen_diamond_mask,
    'pentagon': gen_pentagon_mask,
    'octagon': gen_octagon_mask,
    'square': gen_square_mask,
}
