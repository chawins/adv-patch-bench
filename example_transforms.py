import json
import os
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.models as models
from kornia.geometry.transform import (get_perspective_transform, resize,
                                       warp_affine, warp_perspective)
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm

from adv_patch_bench.datasets.utils import (get_image_files,
                                            img_numpy_to_torch,
                                            load_annotation, pad_image)
from adv_patch_bench.models.common import Normalize
from adv_patch_bench.transforms import (gen_sign_mask, get_box_vertices,
                                        get_corners, get_shape_from_vertices)
from classify_traffic_signs import draw_from_contours

TRAFFIC_SIGN_LABEL = 95

CLASS_LIST = ['octagon-915.0-915.0',
              'diamond-915.0-915.0',
              'pentagon-915.0-915.0',
              'rect-915.0-1220.0',
              'rect-762.0-915.0',
              'triangle-900.0',
              'circle-750.0',
              'triangle_inverted-1220.0-1220.0',
              'rect-458.0-610.0',
              'other-0.0-0.0']


def compute_example_transform(filename, model, panoptic_per_image_id,
                              img_path, label_path, demo_patch,
                              min_area=0, pad=0.05, patch_size_in_mm=150,
                              patch_size_in_pixel=32):

    # Load Mapillary Vistas image
    img_id = filename.split('.')[0]
    segment = panoptic_per_image_id[img_id]['segments_info']
    panoptic = np.array(Image.open(join(label_path, f'{img_id}.png')))
    img_pil = Image.open(join(img_path, filename))
    img = np.array(img_pil)[:, :, :3]
    img_height, img_width, _ = img.shape

    # Pad image to avoid cutting varying shapes due to boundary
    img_padded, pad_size = pad_image(img, pad_mode='constant', return_pad_size=True)
    id_padded = pad_image(panoptic[:, :, 0], pad_mode='constant')

    # Crop the specified object
    for obj in segment:

        # Check if bounding box is cut off at the image boundary
        xmin, ymin, width, height = obj['bbox']
        is_oob = (xmin == 0) or (ymin == 0) or \
            ((xmin + width) >= img_width) or ((ymin + height) >= img_height)

        if obj['category_id'] != TRAFFIC_SIGN_LABEL or obj['area'] < min_area or is_oob:
            continue

        # Make sure that bounding box is square and add some padding to avoid
        # cutting into the sign
        size = max(width, height)
        xpad, ypad = int((size - width) / 2), int((size - height) / 2)
        extra_obj_pad = int(pad * size)
        size += 2 * extra_obj_pad
        xmin += pad_size - xpad - extra_obj_pad
        ymin += pad_size - ypad - extra_obj_pad
        xmax, ymax = xmin + size, ymin + size
        traffic_sign = img_numpy_to_torch(img_padded[ymin:ymax, xmin:xmax])
        # TODO: Consider running classifier outside once in batch
        y_hat = model(traffic_sign.unsqueeze(0).cuda())[0].argmax().item()
        predicted_class = CLASS_LIST[y_hat]
        predicted_shape = predicted_class.split('-')[0]
        print(f'==> predicted_class: {predicted_class}')

        # Collect mask
        bool_mask = (id_padded[ymin:ymax, xmin:xmax] == obj['id']).astype(np.uint8)
        # Get vertices of mask
        vertices, hull = get_corners(bool_mask)
        # Fit ellipse
        hull_mask = cv.drawContours(np.zeros_like(bool_mask), [hull], -1, (1, ), 1)
        hull_draw_points = np.stack(np.where(hull_mask), axis=1)[:, ::-1]
        ellipse = cv.fitEllipse(hull_draw_points)
        ellipse_mask = cv.ellipse(np.zeros_like(bool_mask, dtype=np.float32), ellipse, (1,), thickness=-1)
        ellipse_error = np.abs(ellipse_mask - bool_mask.astype(np.float32)).sum() / bool_mask.sum()

        # Determine polygon shape from vertices
        shape = get_shape_from_vertices(vertices)
        if predicted_shape == 'other':
            group = 3
        elif ellipse_error < 0.1:
            # Check circle based on ellipse fit error
            shape = 'circle'
            vertices = ellipse
            group = 1 if predicted_shape == 'circle' else 2
        else:
            if ((shape != 'other' and predicted_shape == shape) or
                    (shape == 'rect' and predicted_shape != 'other')):
                # Both classifier and verifier agree on some polygons or
                # the sign symbol is on a square sign (assume that dimension is
                # equal to the actual symbol)
                group = 1
            else:
                # Disagree but not other
                group = 2
        print(f'==> shape: {shape}, group: {group}')
        print(vertices)

        if shape != 'other':
            tgt = get_box_vertices(vertices, shape).astype(np.int64)
            # Filter some vertices that might be out of bound
            tgt = np.array([t for t in tgt if 0 <= t[0] < size and 0 <= t[1] < size])

            # If shape is not other, draw vertices
            vert = draw_from_contours(np.zeros((size, size, 3)), tgt, color=[0, 255, 0])
            vert = img_numpy_to_torch(cv.dilate(vert, None))
            vert_mask = (vert.sum(0, keepdim=True) > 0).float()
            traffic_sign = (1 - vert_mask) * traffic_sign + vert_mask * vert

            # Group 1: draw both vertices and patch
            if group == 1:
                if len(predicted_class.split('-')) == 3:
                    sign_width_in_mm = float(predicted_class.split('-')[1])
                    sign_height_in_mm = float(predicted_class.split('-')[2])
                    hw_ratio = sign_height_in_mm / sign_width_in_mm
                    sign_size_in_mm = max(sign_width_in_mm, sign_height_in_mm)
                else:
                    sign_size_in_mm = float(predicted_class.split('-')[1])
                    hw_ratio = 1
                pixel_mm_ratio = patch_size_in_pixel / patch_size_in_mm
                sign_size_in_pixel = round(sign_size_in_mm * pixel_mm_ratio)

                sign_canonical = torch.zeros((3, sign_size_in_pixel, sign_size_in_pixel))
                sign_mask, src = gen_sign_mask(shape, sign_size_in_pixel, ratio=hw_ratio)
                sign_mask = torch.from_numpy(sign_mask).float()[None, :, :]

                # TODO: run attack, optimize patch location, etc.
                begin = (sign_size_in_pixel - patch_size_in_pixel) // 2
                end = begin + patch_size_in_pixel
                sign_canonical[:, begin:end, begin:end] = demo_patch
                # Crop patch that is not on the sign
                sign_canonical *= sign_mask
                patch_mask = torch.zeros((1, sign_size_in_pixel, sign_size_in_pixel))
                patch_mask[:, begin:end, begin:end] = 1

                # Compute perspective transform
                src = np.array(src).astype(np.float32)
                tgt = tgt.astype(np.float32)
                if len(src) == 3:
                    M = torch.from_numpy(cv.getAffineTransform(src, tgt)).unsqueeze(0).float()
                    transform_func = warp_affine
                else:
                    src = torch.from_numpy(src).unsqueeze(0)
                    tgt = torch.from_numpy(tgt).unsqueeze(0)
                    M = get_perspective_transform(src, tgt)
                    transform_func = warp_perspective
                warped_patch = transform_func(sign_canonical.unsqueeze(0),
                                              M, (size, size),
                                              mode='bilinear',
                                              padding_mode='zeros')[0]
                warped_mask = transform_func(patch_mask.unsqueeze(0),
                                             M, (size, size),
                                             mode='nearest',
                                             padding_mode='zeros')[0]

                # Assume that 80% of pixels have one 255 (e.g., red, blue) and
                # 20% of pixels are white (255, 255, 255).
                old_patch = torch.masked_select(traffic_sign, torch.from_numpy(bool_mask).bool())
                mu_1, sigma_1 = old_patch.mean(), old_patch.std()
                # mu_0, sigma_0 = 0.4666667, 0.4988876
                old_patch -= old_patch.min()
                old_patch /= old_patch.max()
                old_patch_q = (old_patch > 0.5).float()
                mu_0, sigma_0 = old_patch_q.mean(), old_patch_q.std()
                alpha = sigma_1 / sigma_0
                beta = mu_1 - mu_0 * alpha
                warped_patch.mul_(alpha).add_(beta).clamp_(0, 1)

                traffic_sign = (1 - warped_mask) * traffic_sign + warped_mask * warped_patch

                # DEBUG
                # save_image(traffic_sign, 'test.png')
                # import pdb
                # pdb.set_trace()

        # DEBUG
        # if 'triangle' in shape:
        #     save_image(cropped_sign, 'test.png')
        #     import pdb
        #     pdb.set_trace()

        return traffic_sign


def main():

    # Arguments
    min_area = 1600
    max_num_imgs = 200
    data_dir = '/data/shared/mapillary_vistas/training/'
    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    model_path = '/home/nab_126/adv-patch-bench/model_weights/resnet18_cropped_signs_good_resolution_and_not_edge_10_labels.pth'

    device = 'cuda'
    # seed = 2021
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    # Create model
    mean = [0.3867, 0.3993, 0.3786]
    std = [0.1795, 0.1718, 0.1714]
    normalize = Normalize(mean, std)
    base = models.resnet18(pretrained=False)
    base.fc = nn.Linear(512, 10)

    if os.path.exists(model_path):
        print('Loading model weights...')
        base.load_state_dict(torch.load(model_path))
    else:
        raise ValueError('Model weight not found!')

    model = nn.Sequential(normalize, base).to(device).eval()

    # Read in panoptic file
    panoptic_json_path = f'{data_dir}/v2.0/panoptic/panoptic_2020.json'
    with open(panoptic_json_path) as panoptic_file:
        panoptic = json.load(panoptic_file)

    # Convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic['annotations']:
        panoptic_per_image_id[annotation['image_id']] = annotation

    # Convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic['categories']:
        panoptic_category_per_id[category['id']] = category

    img_path = join(data_dir, 'images')
    label_path = join(data_dir, 'v2.0/panoptic/')

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    np.random.shuffle(filenames)

    # demo_patch = img_numpy_to_torch(np.array(Image.open()))
    demo_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
    demo_patch = resize(demo_patch, (32, 32))

    for filename in tqdm(filenames):
        compute_example_transform(filename, model, panoptic_per_image_id,
                                  img_path, label_path, demo_patch,
                                  min_area=min_area, pad=0.05, patch_size_in_mm=150,
                                  patch_size_in_pixel=32)


if __name__ == '__main__':
    main()
