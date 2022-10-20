"""Utility functions for transforms."""

from typing import List, NewType, Optional, Tuple, Union

import kornia
import kornia.augmentation as K
import numpy as np
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    ImageTensor,
    TransformFn,
    TransformParamFn,
)

_KeyPoints = NewType("_KeyPoints", List[Tuple[float, float]])


def _identity(x: Union[ImageTensor, BatchImageTensor]) -> BatchImageTensor:
    """Identity transform function."""
    return x


def _identity_with_params(
    x: Union[ImageTensor, BatchImageTensor]
) -> Tuple[BatchImageTensor, None]:
    """Indentity function that also returns None param."""
    x = _identity(x)
    return x, None


def _gen_rect_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate rectangular mask.

    The keypoints are a list of tuple (x, y) coordinates starting from the
    uppper left one and sorted clockwise. For example, keypoints of a
    rectangular mask are upper-left, upper-right, lower-right, and lower-left
    corners, respectively.

    Args:
        size: Width of object in pixels.
        ratio: Ratio between height and width.

    Returns:
        Binary mask and source keypoint for geometric transformation with
        respect to this mask.
    """
    height: int = round(ratio * size)
    width: int = size
    mask: np.ndarray = np.zeros((height, width))

    if ratio > 1:
        # Tall rectangle
        pad = round((height - width) / 2)
        assert pad < width
        mask[:, pad : width - pad] = 1
        box = [
            (pad, 0),
            (width - pad, 0),
            (width - pad, height - 1),
            (pad, height - 1),
        ]
    elif ratio < 1:
        # Wide rectangle
        pad = round((width - height) / 2)
        assert pad < height
        mask[pad : height - pad, :] = 1
        box = [
            (0, pad),
            (width - 1, pad),
            (width - 1, height - pad),
            (0, height - pad),
        ]
    else:
        # Square
        mask[:, :] = 1
        box = [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)]

    return mask, box


def _gen_diamond_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate diamond mask. See _gen_rect_mask()."""
    del ratio  # Unused
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= mid)
        * (Y - X >= -mid)
        * (Y + X <= size + mid)
        * (Y - X <= mid)
    )
    return mask, [(0, mid), (mid, 0), (size - 1, mid), (mid, size - 1)]


def _gen_circle_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate circle mask. See _gen_rect_mask()."""
    del ratio  # Unused
    Y, X = np.ogrid[:size, :size]
    center = round(size / 2)  # center is also radius
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= center
    return mask, [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)]


def _gen_triangle_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate triangle mask. See _gen_rect_mask()."""
    height = round(ratio * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height + 2 * X / size >= 1) * (Y / height - 2 * X / size >= -1)
    return mask, [(mid, 0), (size - 1, height - 1), (0, height - 1)]


def _gen_triangle_inverted_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate inverted triangle mask. See _gen_rect_mask()."""
    height = round(ratio * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height - 2 * X / size <= 0) * (Y / height + 2 * X / size <= 2)
    return mask, [(0, 0), (size - 1, 0), (mid, height - 1)]


def _gen_pentagon_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate pentagon mask. See _gen_rect_mask()."""
    del ratio  # Unused
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= mid) * (Y - X >= -mid)
    return mask, [
        (0, mid),
        (size - 1, mid),
        (size - 1, size - 1),
        (0, size - 1),
    ]


def _gen_octagon_mask(
    size: int, ratio: Optional[float] = None
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate octagon mask. See _gen_rect_mask()."""
    del ratio  # Unused
    edge = round((2 - np.sqrt(2)) / 2 * size)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= edge)
        * (Y - X >= -(size - edge))
        * (Y + X <= 2 * size - edge)
        * (Y - X <= (size - edge))
    )
    return mask, [
        (edge, 0),
        (size - 1, edge),
        (size - edge, size - 1),
        (0, size - edge),
    ]


def gen_sign_mask(
    shape: str, hw_ratio: float, obj_width_px: int
) -> Tuple[np.ndarray, _KeyPoints]:
    """Generate mask of object and source keypoints.

    The keypoints are a list of tuple (x, y) coordinates starting from the
    uppper left one and sorted clockwise. For example, keypoints of a
    rectangular mask are upper-left, upper-right, lower-right, and lower-left
    corners, respectively.

    Args:
        shape: Object shape defined based on classes in REAP.
        hw_ratio: Ratio of height over width of object.
        obj_width_px: Width of object in pixels.

    Returns:
        Binary mask and source keypoint for geometric transformation with
        respect to this mask.
    """
    shape_to_mask = {
        "circle": _gen_circle_mask,
        "triangle_inverted": _gen_triangle_inverted_mask,
        "triangle": _gen_triangle_mask,
        "rect": _gen_rect_mask,
        "diamond": _gen_diamond_mask,
        "pentagon": _gen_pentagon_mask,
        "octagon": _gen_octagon_mask,
        "square": _gen_rect_mask,
    }
    return shape_to_mask[shape](obj_width_px, ratio=hw_ratio)


def get_transform_fn(
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
        Tuple of three transform functions: (i) geometric for object, (ii)
        geometric for mask, and (iii) lighting for object.
    """
    # Geometric transform
    geo_transform: TransformParamFn = _identity_with_params
    mask_transform: TransformFn = _identity

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

        geo_transform = transform_fn(
            return_transform=True,
            resample=interp,
            **transform_params,
        )
        mask_transform = transform_fn(
            resample=kornia.constants.Resample.NEAREST, **transform_params
        )

    # Lighting transform (color jitter)
    light_transform: TransformFn = _identity
    if (
        prob_colorjitter is not None
        and prob_colorjitter > 0
        and syn_colorjitter is not None
        and syn_colorjitter > 0
    ):
        # Hue can't be change much; Otherwise, the color becomes wrong
        light_transform: TransformFn = K.ColorJitter(
            brightness=syn_colorjitter,
            contrast=syn_colorjitter,
            saturation=syn_colorjitter,
            hue=0.05,
            p=prob_colorjitter,
        )

    return (
        geo_transform,
        mask_transform,
        light_transform,
    )
