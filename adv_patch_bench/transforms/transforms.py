import cv2 as cv
import numpy as np
from ast import literal_eval
import pandas as pd
import torch
from cv2 import getAffineTransform
from kornia.geometry.transform import (get_perspective_transform, warp_affine,
                                       warp_perspective)

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


def get_sign_canonical(predicted_class, patch_size_in_pixel,
                       patch_size_in_mm, sign_size_in_pixel=None):
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
    sign_canonical = torch.zeros((4, sign_size_in_pixel, sign_size_in_pixel))
    sign_mask, src = gen_sign_mask(shape, sign_size_in_pixel, ratio=hw_ratio)
    sign_mask = torch.from_numpy(sign_mask).float()[None, :, :]
    return sign_canonical, sign_mask, src


def get_transform(sign_size_in_pixel, predicted_class, row, h0, w0,
                  h_ratio, w_ratio, w_pad, h_pad, no_transform=False):
    # TODO: This should directly come from patch mask
    sign_canonical, sign_mask, src = get_sign_canonical(
        predicted_class, None, None, sign_size_in_pixel=sign_size_in_pixel)
    alpha = torch.tensor(row['alpha'])
    beta = torch.tensor(row['beta'])

    src = np.array(src, dtype=np.float32)
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)

        offset_y = min(tgt[:, 1])
        offset_x = min(tgt[:, 0])

        tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))

        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad

        if no_transform:
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

        tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))

        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

        if no_transform:
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

    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    return transform_func, sign_canonical, sign_mask, M.squeeze(), alpha, beta


def apply_transform(image, adv_patch, patch_mask, patch_loc, transform_func, tf_data,
                    tf_patch=None, tf_bg=None, no_relighting=False):
    ymin, xmin, height, width = patch_loc
    sign_canonical, sign_mask, M, alpha, beta = tf_data
    if no_relighting:
        adv_patch.clamp_(0, 1).mul_(1).add_(0).clamp_(0, 1)
    else:
        adv_patch.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)
    sign_canonical[:, :-1, ymin:ymin + height, xmin:xmin + width] = adv_patch
    sign_canonical[:, -1, ymin:ymin + height, xmin:xmin + width] = 1
    sign_canonical = sign_mask * patch_mask * sign_canonical
    # Apply augmentation on the patch
    if tf_patch is not None:
        sign_canonical = tf_patch(sign_canonical)

    warped_patch = transform_func(sign_canonical,
                                  M, image.shape[2:],
                                #   mode='bicubic',
                                  mode='bilinear',  # TODO: try others?
                                  padding_mode='zeros')
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[:, -1].unsqueeze(1)
    final_img = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch[:, :-1]
    # Apply augmentation on the entire image
    if tf_bg is not None:
        final_img = tf_bg(final_img)
    return final_img





def transform_and_apply_patch(image, adv_patch, patch_mask, patch_loc,
                              predicted_class, row, img_data, no_transform=False, device=None):
    if adv_patch.ndim == 4:
        adv_patch = adv_patch[0]
    if patch_mask.ndim == 4:
        patch_mask = patch_mask[0]

    ymin, xmin, height, width = patch_loc
    h0, w0, h_ratio, w_ratio, w_pad, h_pad = img_data
    sign_canonical, sign_mask, src = get_sign_canonical(
        predicted_class, None, None, sign_size_in_pixel=patch_mask.size(-1))
    sign_canonical = sign_canonical.to(device)
    sign_mask = sign_mask.to(device)
    alpha = row['alpha']
    beta = row['beta']

    patch_cropped = adv_patch.clone()
    patch_cropped.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)
    sign_canonical[:-1, ymin:ymin + height, xmin:xmin + width] = patch_cropped
    sign_canonical[-1, ymin:ymin + height, xmin:xmin + width] = 1
    # print(patch_mask.device)
    # print(sign_mask.device)
    # print(sign_canonical.device)
    sign_canonical = sign_mask * patch_mask * sign_canonical

    src = np.array(src, dtype=np.float32)

    # if annotated, then use those points
    # TODO: unify `points` and other `tgt` in the csv file so this part can be
    # simplified into one function
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)

        offset_y = min(tgt[:, 1])
        offset_x = min(tgt[:, 0])

        tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))

        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad

        if no_transform:
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

        tgt_shape = (max(tgt[:, 1]) - min(tgt[:, 1]), max(tgt[:, 0]) - min(tgt[:, 0]))

        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

        if no_transform:
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

    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    cur_shape = image.shape[-2:]

    warped_patch = transform_func(sign_canonical.unsqueeze(0),
                                  M.to(device), cur_shape,
                                #   mode='bilinear',
                                  mode='bicubic',
                                  padding_mode='zeros')[0]
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[-1].unsqueeze(0)
    
    image_with_transformed_patch = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch[:-1]
    return image_with_transformed_patch