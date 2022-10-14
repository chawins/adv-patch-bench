"""Utility functions for transforms."""

from typing import Optional, Tuple, Union

import kornia
import kornia.augmentation as K
import numpy as np
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    ImageTensor,
    TransformFn,
    TransformParamFn,
)


def _identity(x: Union[ImageTensor, BatchImageTensor]) -> BatchImageTensor:
    return x


def _identity_with_params(
    x: Union[ImageTensor, BatchImageTensor]
) -> Tuple[BatchImageTensor, None]:
    x = _identity(x)
    return x, None


def gen_rect_mask(size, ratio=None):
    height = round(ratio * size) if ratio > 1 else size
    width = size
    mask = np.zeros((height, width))
    if ratio > 1:
        pad = int((size - width) / 2)
        mask[:, pad : size - pad] = 1
        box = [
            [pad, 0],
            [size - pad, 0],
            [size - pad, size - 1],
            [pad, size - 1],
        ]
        box = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    elif ratio < 1:
        pad = int((size - height) / 2)
        mask[pad : size - pad, :] = 1
        box = [
            [0, pad],
            [size - 1, pad],
            [size - 1, size - pad],
            [0, size - pad],
        ]
    else:
        mask[:, :] = 1
        box = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    return mask, box


def gen_square_mask(size, ratio=None):
    # ratio = height / width
    assert ratio == 1
    mask = np.zeros((size, size))
    height = ratio * size if ratio < 1 else size
    width = height / ratio
    if ratio > 1:
        pad = int((size - width) / 2)
        mask[:, pad : size - pad] = 1
        box = [
            [pad, 0],
            [size - pad, 0],
            [size - pad, size - 1],
            [pad, size - 1],
        ]
    elif ratio < 1:
        pad = int((size - height) / 2)
        mask[pad : size - pad, :] = 1
        box = [
            [0, pad],
            [size - 1, pad],
            [size - 1, size - pad],
            [0, size - pad],
        ]
    else:
        mask[:, :] = 1
        box = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    return mask, box


def gen_diamond_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= mid)
        * (Y - X >= -mid)
        * (Y + X <= size + mid)
        * (Y - X <= mid)
    )
    return mask, [[0, mid], [mid, 0], [size - 1, mid], [mid, size - 1]]


def gen_circle_mask(size, ratio=None):
    Y, X = np.ogrid[:size, :size]
    center = round(size / 2)
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= center
    return mask, [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]


def gen_triangle_mask(size, ratio=None):
    # width: 1168, height: 1024
    height = int(1024 / 1168 * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height + 2 * X / size >= 1) * (Y / height - 2 * X / size >= -1)
    return mask, [[mid, 0], [size - 1, height - 1], [0, height - 1]]


def gen_triangle_inverted_mask(size, ratio=None):
    # width: 1024, height: 900
    height = int(900 / 1024 * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height - 2 * X / size <= 0) * (Y / height + 2 * X / size <= 2)
    return mask, [[0, 0], [size - 1, 0], [mid, height - 1]]


def gen_pentagon_mask(size, ratio=None):
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= mid) * (Y - X >= -mid)
    return mask, [
        [0, mid],
        [size - 1, mid],
        [size - 1, size - 1],
        [0, size - 1],
    ]


def gen_octagon_mask(size, ratio=None):
    edge = round((2 - np.sqrt(2)) / 2 * size)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= edge)
        * (Y - X >= -(size - edge))
        * (Y + X <= 2 * size - edge)
        * (Y - X <= (size - edge))
    )
    return mask, [
        [edge, 0],
        [size - 1, edge],
        [size - edge, size - 1],
        [0, size - edge],
    ]


def gen_sign_mask(shape: str, hw_ratio: float, obj_width_px: int):
    return _SHAPE_TO_MASK[shape](obj_width_px, ratio=hw_ratio)


_SHAPE_TO_MASK = {
    "circle": gen_circle_mask,
    "triangle_inverted": gen_triangle_inverted_mask,
    "triangle": gen_triangle_mask,
    "rect": gen_rect_mask,
    "diamond": gen_diamond_mask,
    "pentagon": gen_pentagon_mask,
    "octagon": gen_octagon_mask,
    "square": gen_square_mask,
}


def init_syn_transforms(
    prob_geo: Optional[float] = None,
    syn_rotate: Optional[float] = None,
    syn_scale: Optional[float] = None,
    syn_translate: Optional[float] = None,
    syn_3d_dist: Optional[float] = None,
    prob_colorjitter: Optional[float] = None,
    syn_colorjitter: Optional[float] = None,
    interp: str = "bilinear",
) -> Tuple[TransformFn, TransformFn, TransformFn]:
    """Initialize geometric (for object and mask) and lighting transforms.

    When transforms are not applied, they are returned as identity function.

    Args:
        prob_geo: Probability of applying geometric transform.
        syn_rotate: Rotation degrees. Defaults to None (or 0 = no rotate).
        syn_scale: Scaling ratio. Defaults to None (or 1 = no scale).
        syn_translate: Translation distance.  Defaults to None.
        syn_3d_dist: 3D distortion. If syn_3d_dist is set to any non-None
            value, 3D or perspective transform will be used instead of
            affine transform. Defaults to None.
        prob_colorjitter: Probability of applying lighting transform.
        syn_colorjitter: Colorjitter intensity. Defaults to None (no color
            jitter or no lighting transform).
        interp: Interpolation mode. Defaults to "bilinear".

    Returns:
        Geometric transforms for object and mask and lighting transform.
    """
    # Geometric transform
    if prob_geo is not None and prob_geo > 0:
        if syn_3d_dist is not None and syn_3d_dist > 0:
            transform_params = {
                "p": prob_geo,
                "distortion_scale": syn_3d_dist,
            }
            transform_fn = K.RandomPerspective
        else:
            transform_params = {
                "p": prob_geo,
                "degrees": syn_rotate,
                "translate": (syn_translate, syn_translate),
                "scale": None
                if syn_scale is None
                else (1 / syn_scale, syn_scale),
            }
            transform_fn = K.RandomAffine

        geo_transform: TransformFn = transform_fn(
            return_transform=True,
            resample=interp,
            **transform_params,
        )
        mask_transform: TransformFn = transform_fn(
            resample=kornia.constants.Resample.NEAREST, **transform_params
        )
    else:
        geo_transform: TransformParamFn = _identity_with_params
        mask_transform: TransformFn = _identity

    # Lighting transform (color jitter)
    if (
        prob_colorjitter is not None
        and prob_colorjitter > 0
        and syn_colorjitter is not None
    ):
        # Hue can't be change much; Otherwise, the color becomes wrong
        light_transform: TransformFn = K.ColorJitter(
            brightness=syn_colorjitter,
            contrast=syn_colorjitter,
            saturation=syn_colorjitter,
            hue=0.05,
            p=1.0,
        )
    else:
        light_transform: TransformFn = _identity

    return (
        geo_transform,
        mask_transform,
        light_transform,
    )
