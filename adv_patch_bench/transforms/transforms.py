from ast import literal_eval
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from adv_patch_bench.attacks.utils import coerce_rank
from adv_patch_bench.transforms.verifier import sort_polygon_vertices
from cv2 import getAffineTransform
from kornia.geometry.transform import (get_perspective_transform, warp_affine,
                                       warp_perspective)


def gen_rect_mask(size, ratio=None):
    height = round(ratio * size) if ratio > 1 else size
    width = size
    mask = np.zeros((height, width))
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
    assert ratio == 1
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
    mask = ((Y + X >= edge) * (Y - X >= -(size - edge)) *
            (Y + X <= 2 * size - edge) * (Y - X <= (size - edge)))
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
        sign_canonical (torch.Tensor): sign canonical 
        sign_mask (torch.Tensor): sign mask
        src (list): List that specifies 4 key points of the canonical sign
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
    sign_canonical = torch.zeros(
        (4, round(hw_ratio * sign_size_in_pixel), sign_size_in_pixel))

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
    transform_mode: str,
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
        transform (str, required): _description_. 'transform_scale, affine or perspective'.
    Returns:
        Tuple: _description_
    """
    alpha = torch.tensor(row['alpha'])
    beta = torch.tensor(row['beta'])

    # Get target points from dataframe
    # TODO: Fix this after unifying csv
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)
        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad
        # print('not polygon')
    else:
        tgt = row['tgt'] if pd.isna(row['tgt_polygon']) else row['tgt_polygon']
        tgt = np.array(literal_eval(tgt), dtype=np.float32)
        # print('polygon')

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

    # FIXME: Correct in the csv file directly
    shape = predicted_class.split('-')[0]
    if shape != 'octagon':
        tgt = sort_polygon_vertices(tgt)

    # if row['use_polygon'] == 1:
    #     # nR-M2zUbIWJzatAuy2egrQ.jpg
    # if row['use_polygon'] == 1:
    #     print(tgt)
    #     import pdb
    #     pdb.set_trace()

    if shape == 'diamond':
        # Verify target points of diamonds. If they are very close to corners
        # of a square, the sign likely lies on another square surface. In this
        # case, use the square src points instead.
        x, y = np.abs(tgt[1] - tgt[0])
        angle10 = np.arctan2(y, x)
        x, y = np.abs(tgt[3] - tgt[2])
        angle32 = np.arctan2(y, x)
        mean_angle = (angle10 + angle32) / 2
        if mean_angle < np.pi / 180 * 15:
            predicted_class = f'square-{predicted_class.split("-")[1]}'

    sign_canonical, sign_mask, src = get_sign_canonical(
        predicted_class, sign_size_in_pixel=sign_size_in_pixel)
    src = np.array(src, dtype=np.float32)

    if shape == 'pentagon':
        # Verify that target points of pentagons align like rectangle (almost
        # parallel sides). If not, then there's an annotation error which is
        # then fixed by changing src points.
        angle10 = np.arctan2(*(tgt[1] - tgt[0]))
        angle21 = np.arctan2(*(tgt[2] - tgt[1]))
        angle23 = np.arctan2(*(tgt[2] - tgt[3]))
        angle30 = np.arctan2(*(tgt[3] - tgt[0]))
        mean_diff = (np.abs(angle10 - angle23) + np.abs(angle21 - angle30)) / 2
        if mean_diff > np.pi / 180 * 15:
            src[1, 0] = float(src[1, 1])
            src[1, 1] = 0

    # Get transformation matrix and transform function (affine or perspective)
    # from source and target coordinates
    if transform_mode == 'translate_scale':
        min_tgt_x = min(tgt[:, 0])
        max_tgt_x = max(tgt[:, 0])
        min_tgt_y = min(tgt[:, 1])
        max_tgt_y = max(tgt[:, 1])
        tgt = np.array([[min_tgt_x, min_tgt_y], [max_tgt_x, min_tgt_y],
                        [max_tgt_x, max_tgt_y], [min_tgt_x, max_tgt_y]])

        min_src_x = min(src[:, 0])
        max_src_x = max(src[:, 0])
        min_src_y = min(src[:, 1])
        max_src_y = max(src[:, 1])
        src = np.array([[min_src_x, min_src_y], [max_src_x, min_src_y],
                        [max_src_x, max_src_y], [min_src_x, max_src_y]])

    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    # if transform_mode in ('affine', 'translate_scale'):
    #     if len(src) > 3:
    #         src = src[:-1]
    #         tgt = tgt[:-1]
    #     M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
    #     transform_func = warp_affine

    #     if transform_mode == 'translate_scale':
    #         a = M[0][0][0]
    #         b = M[0][0][1]
    #         c = M[0][1][0]
    #         d = M[0][1][1]
    #         s_x = torch.sign(a) * ((a ** 2 + b ** 2) ** 0.5)
    #         s_y = torch.sign(d) * ((c ** 2 + d ** 2) ** 0.5)
    #         M[0][0][0] = s_x + 1e-15
    #         M[0][0][1] = 0
    #         M[0][1][0] = 0
    #         M[0][1][1] = s_y + 1e-15

    # elif transform_mode == 'perspective':
    #     if len(src) == 3:
    #         M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
    #         transform_func = warp_affine
    #     else:
    #         src = torch.from_numpy(src).unsqueeze(0)
    #         tgt = torch.from_numpy(tgt).unsqueeze(0)
    #         M = get_perspective_transform(src, tgt)
    #         transform_func = warp_perspective
    # else:
    #     raise NotImplementedError(f'transform_mode {transform_mode} does not exist.')

    return transform_func, sign_canonical, sign_mask, M.squeeze(), alpha, beta, tgt


def apply_transform(
    image: torch.FloatTensor,
    adv_patch: torch.FloatTensor,
    patch_mask: torch.FloatTensor,
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
        image (torch.Tensor): Input image (0-255)
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
    sign_canonical, sign_mask, M, alpha, beta = tf_data
    adv_patch = adv_patch.clone()
    adv_patch = coerce_rank(adv_patch, 4)
    patch_mask = coerce_rank(patch_mask, 4)
    sign_canonical = coerce_rank(sign_canonical, 4)
    obj_size = patch_mask.shape[-2:]
    assert adv_patch.shape[-2:] == sign_canonical.shape[-2:] == obj_size

    # ymin, xmin, height, width = patch_loc
    if use_relight:
        adv_patch.mul_(alpha).add_(beta)
    adv_patch.clamp_(0, 1)

    # Place adv_patch on sign_canonical and set alpha channel
    patch_on_obj = sign_canonical.clone()
    patch_on_obj[:, :-1] = patch_mask * adv_patch
    patch_on_obj[:, -1] = patch_mask
    # Crop with sign_mask and patch_mask
    patch_on_obj *= sign_mask * patch_mask

    # Apply augmentation on the patch
    if tf_patch is not None:
        patch_on_obj = tf_patch(patch_on_obj)
    warped_patch = transform_func(patch_on_obj, M, image.shape[2:],
                                  mode=interp, padding_mode='zeros')
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[:, -1]
    warped_patch = warped_patch[:, :-1]
    alpha_mask = coerce_rank(alpha_mask, 4)
    final_img = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch

    # Apply augmentation on the entire image
    if tf_bg is not None:
        final_img = tf_bg(final_img)

    warped_patch_num_pixels = torch.count_nonzero(alpha_mask).item()
    # print('warped_patch_num_pixels', warped_patch_num_pixels)
    final_img *= 255
    return final_img, warped_patch_num_pixels

@torch.no_grad()
def transform_and_apply_patch(
    image: torch.Tensor,
    adv_patch: torch.Tensor,
    patch_mask: torch.Tensor,
    predicted_class: str,
    row: pd.DataFrame,
    img_data: List,
    transform_mode: str,
    use_relight: bool = True,
    interp: str = 'bilinear',
) -> torch.Tensor:
    # Does not support batch mode. Add singleton dims to 4D if needed.
    image = coerce_rank(image, 4)
    adv_patch = coerce_rank(adv_patch, 4)
    patch_mask = coerce_rank(patch_mask, 4)
    device = image.device

    sign_size_in_pixel = patch_mask.shape[-1]
    transform_func, sign_canonical, sign_mask, M, alpha, beta, _ = get_transform(
        sign_size_in_pixel, predicted_class, row, *img_data, transform_mode)

    sign_canonical = coerce_rank(sign_canonical, 4).to(device)
    sign_mask = coerce_rank(sign_mask, 4).to(device)
    M = coerce_rank(M, 3).to(device)
    tf_data = (sign_canonical, sign_mask, M, alpha.to(device), beta.to(device))

    img, warped_patch_num_pixels = apply_transform(
        image, adv_patch, patch_mask, transform_func, tf_data,
        interp=interp, use_relight=use_relight)

    return img, warped_patch_num_pixels
