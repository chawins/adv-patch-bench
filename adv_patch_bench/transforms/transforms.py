from ast import literal_eval
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from cv2 import getAffineTransform
from kornia.geometry.transform import (get_perspective_transform, warp_affine,
                                       warp_perspective)


def gen_rect_mask(size, ratio=None):
    # ratio = height / width
    # mask = np.zeros((size, size))
    
    height = round(ratio * size) if ratio > 1 else size
    width = size
    mask = np.zeros((height, width))
    # height = ratio * size if ratio < 1 else size
    # width = height / ratio
    if ratio > 1:
        pad = int((size - width) / 2)
        mask[:, pad:size - pad] = 1
        box = [[pad, 0], [size - pad, 0], [size - pad, size - 1], [pad, size - 1]]
        box = [[0, 0], [width-1, 0], [width-1, height-1], [0, height - 1]]
        
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
    mask = (Y + X >= edge) * (Y - X >= -(size - edge)) * (Y + X <= 2 * size - edge) * (Y - X <= (size - edge))
    return mask, [[edge, 0], [size - 1, edge], [size - edge, size - 1], [0, size - edge]]


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


def get_sign_canonical(
    predicted_class: str,
    patch_size_in_pixel: int = None,
    patch_size_in_mm: float = None,
    sign_size_in_pixel: int = None,
) -> Tuple:
    """Generate the canonical mask of a sign with a specific shape.
    Args:
        predicted_class (str): Sign class in format 'shape-width-height' or 
            'shape-width'
        patch_size_in_pixel (int): Width of the patch to apply in pixels.
            Defaults to None (use `sign_size_in_pixel` insteal).
        patch_size_in_mm (float): Width of the patch to apply in mm.
            Defaults to None (use `sign_size_in_pixel` insteal).
        sign_size_in_pixel (int, optional): Optionally, sign size in pixels can
            be explicitly set. Defaults to None (use size relative to patch).
    Returns:
        - sign_canonical (torch.Tensor): sign canonical 
        - sign_mask (torch.Tensor): sign mask
        - src (list): List that specifies 4 key points of the canonical sign
    """
    assert sign_size_in_pixel is not None or patch_size_in_pixel is not None
    shape = predicted_class.split('-')[0]
    sign_width_in_mm = float(predicted_class.split('-')[1])
    if len(predicted_class.split('-')) == 3:
        sign_height_in_mm = float(predicted_class.split('-')[2])
        hw_ratio = sign_height_in_mm / sign_width_in_mm
    else:
        hw_ratio = 1
    if sign_size_in_pixel is None:
        if len(predicted_class.split('-')) == 3:
            sign_size_in_mm = max(sign_width_in_mm, sign_height_in_mm)
        else:
            sign_size_in_mm = float(sign_width_in_mm)
        pixel_mm_ratio = patch_size_in_pixel / patch_size_in_mm
        sign_size_in_pixel = round(sign_size_in_mm * pixel_mm_ratio)
    
    # sign_canonical = torch.zeros((4, sign_size_in_pixel, sign_size_in_pixel))
    sign_canonical = torch.zeros((4, round(hw_ratio * sign_size_in_pixel), sign_size_in_pixel))

    sign_mask, src = gen_sign_mask(shape, sign_size_in_pixel, ratio=hw_ratio)
    sign_mask = torch.from_numpy(sign_mask).float()[None, :, :]
    return sign_canonical, sign_mask, src


def get_transform(
    sign_size_in_pixel: int,
    predicted_class: str,
    row: pd.DataFrame,
    h0: float,
    w0: float,
    h_ratio: float,
    w_ratio: float,
    w_pad: float,
    h_pad: float,
    use_transform: bool = True,
) -> Tuple:
    """Get transformation matrix and parameters including relighting.
    Args:
        sign_size_in_pixel (int): _description_
        predicted_class (str): _description_
        row (pd.DataFrame): _description_
        h0 (float): _description_
        w0 (float): _description_
        h_ratio (float): _description_
        w_ratio (float): _description_
        w_pad (float): _description_
        h_pad (float): _description_
        use_transform (bool, optional): _description_. Defaults to True.
    Returns:
        Tuple: _description_
    """
    sign_canonical, sign_mask, src = get_sign_canonical(
        predicted_class, sign_size_in_pixel=sign_size_in_pixel)
    alpha = torch.tensor(row['alpha'])
    beta = torch.tensor(row['beta'])

    src = np.array(src, dtype=np.float32)
    # TODO: Fix this after unifying csv
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)

        offset_y = min(tgt[:, 1])
        offset_x = min(tgt[:, 0])

        # tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))
        tgt_shape = (max(tgt[:, 0]) - min(tgt[:, 0]), max(tgt[:, 1]) - min(tgt[:, 1]))

        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad

        if not use_transform:
            # Get the scaling factor
            src_shape = (max(src[:, 0]) - min(src[:, 0]), max(src[:, 1]) - min(src[:, 1]))
            # you have to flip because the image.shape is (y,x) but your corner points are (x,y)
            scale = np.divide(tgt_shape, src_shape)

            tgt_untransformed = src.copy()
            # rescale src
            tgt_untransformed[:, 1] = tgt_untransformed[:, 1] * scale[1]
            tgt_untransformed[:, 0] = tgt_untransformed[:, 0] * scale[0]
            # translate src
            tgt_untransformed[:, 1] += offset_y
            tgt_untransformed[:, 0] += offset_x
            tgt_untransformed[:, 1] = (tgt_untransformed[:, 1] * h_ratio) + h_pad
            tgt_untransformed[:, 0] = (tgt_untransformed[:, 0] * w_ratio) + w_pad
            tgt = tgt_untransformed
    else:
        tgt = row['tgt'] if pd.isna(row['tgt_polygon']) else row['tgt_polygon']
        tgt = np.array(literal_eval(tgt), dtype=np.float32)

        # tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))
        tgt_shape = (max(tgt[:, 0]) - min(tgt[:, 0]), max(tgt[:, 1]) - min(tgt[:, 1]))

        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved
        # TODO: this should be cleaned up with csv
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

        if not use_transform:
            # Get the scaling factor
            src_shape = (max(src[:, 0]) - min(src[:, 0]), max(src[:, 1]) - min(src[:, 1]))
            # you have to flip because the image.shape is (y,x) but your corner points are (x,y)
            scale = np.divide(tgt_shape, src_shape)
            tgt_untransformed = src.copy()
            # rescale src
            tgt_untransformed[:, 1] = tgt_untransformed[:, 1] * scale[1]
            tgt_untransformed[:, 0] = tgt_untransformed[:, 0] * scale[0]
            # translate src
            tgt_untransformed[:, 1] = (tgt_untransformed[:, 1] + y_min) * h_ratio + h_pad
            tgt_untransformed[:, 0] = (tgt_untransformed[:, 0] + x_min) * w_ratio + w_pad
            tgt = tgt_untransformed

    # Get transformation matrix and transform function (affine or perspective)
    # from source and target coordinates
    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    return transform_func, sign_canonical, sign_mask, M.squeeze(), alpha, beta


def apply_transform(
    image: torch.Tensor,
    adv_patch: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_loc: Tuple[float],
    transform_func: Any,
    tf_data: Tuple[Any],
    tf_patch: Any = None,
    tf_bg: Any = None,
    interp: str = 'bilinear',
    use_relight: bool = False,
) -> torch.Tensor:
    """
    Apply patch with transformation specified by `tf_data` and `transform_func`.
    This function is designed to be used with `get_transform` function.
    All Tensor inputs must have batch dimension.
    Args:
        image (torch.Tensor): Input image
        adv_patch (torch.Tensor): Patch in canonical size
        patch_mask (torch.Tensor): Mask for patch w.r.t. canonical sign
        patch_loc (Tuple[float]): Patch bounding box w.r.t. canonical sign
        transform_func (Any): Transform function from `get_transform`
        tf_data (Tuple[Any]): Parameters of the transform
        tf_patch (Any, optional): Additional transformation applied to
            `adv_patch`. Used for random augmentation. Defaults to None.
        tf_bg (Any, optional): Additional transformation applied to `image`.
            Used for random augmentation.. Defaults to None.
    Returns:
        torch.Tensor: Image with transformed patch
    """
    ymin, xmin, height, width = patch_loc
    sign_canonical, sign_mask, M, alpha, beta = tf_data
    adv_patch.clamp_(0, 1)
    if use_relight:
        adv_patch.mul_(alpha).add_(beta).clamp_(0, 1)
    sign_canonical = sign_canonical.clone()
    sign_canonical[:, :-1, ymin:ymin + height, xmin:xmin + width] = adv_patch
    sign_canonical[:, -1, ymin:ymin + height, xmin:xmin + width] = 1

    sign_canonical = sign_mask * patch_mask * sign_canonical
    # Apply augmentation on the patch
    if tf_patch is not None:
        sign_canonical = tf_patch(sign_canonical)

    warped_patch = transform_func(sign_canonical, M, image.shape[2:],
                                  mode=interp, padding_mode='zeros')
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[:, -1].unsqueeze(1)
    final_img = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch[:, :-1]
    # Apply augmentation on the entire image
    if tf_bg is not None:
        final_img = tf_bg(final_img)
    return final_img


def add_singleton_dim(x, total_dim):
    add_dim = total_dim - x.ndim
    for _ in range(add_dim):
        x = x.unsqueeze(0)
    return x


def transform_and_apply_patch(
    image: torch.Tensor,
    adv_patch: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_loc: Tuple[float],
    predicted_class: str,
    row: pd.DataFrame,
    img_data: List,
    use_transform: bool = True,
    use_relight: bool = True,
    interp: str = 'bilinear',
) -> torch.Tensor:

    # Does not support batch mode. Add singleton dims to 4D if needed.
    image = add_singleton_dim(image, 4)
    adv_patch = add_singleton_dim(adv_patch, 4)
    patch_mask = add_singleton_dim(patch_mask, 4)
    device = image.device

    sign_size_in_pixel = patch_mask.shape[-1]
    transform_func, sign_canonical, sign_mask, M, alpha, beta = get_transform(
        sign_size_in_pixel, predicted_class, row, *img_data, use_transform=use_transform)

    sign_canonical = add_singleton_dim(sign_canonical, 4).to(device)
    sign_mask = add_singleton_dim(sign_mask, 4).to(device)
    M = add_singleton_dim(M, 3).to(device)
    tf_data = (sign_canonical, sign_mask, M, alpha.to(device), beta.to(device))

    img = apply_transform(image, adv_patch, patch_mask, patch_loc,
                          transform_func, tf_data, interp=interp, use_relight=use_relight)
    return img