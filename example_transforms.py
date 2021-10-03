import json
import os
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from kornia.geometry.transform import (get_perspective_transform,
                                       homography_warp, resize)
from PIL import Image
from skimage.exposure import match_histograms
# from skimage.transform import resize
from torchvision.utils import save_image
from tqdm.auto import tqdm

from adv_patch_bench.datasets.utils import (get_image_files, load_annotation,
                                            pad_image, img_numpy_to_torch)
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
              'rect-458.0-610.0']


def compute_example_transform(filename, model, panoptic_per_image_id,
                              img_path, label_path, demo_patch,
                              min_area=0, pad=0.05, patch_size_in_mm=150,
                              patch_size_in_pixel=32):

    # Load Mapillary Vistas image
    img_id = filename.split('.')[0]
    segment = panoptic_per_image_id[img_id]['segments_info']
    panoptic = np.array(Image.open(join(label_path, f'{img_id}.png')))
    img_pil = Image.open(join(img_path, filename))
    img = np.array(img_pil)
    img_height, img_width, _ = img.shape

    # Pad image to avoid cutting varying shapes due to boundary
    img_padded, pad_size = pad_image(img, pad_mode='constant', return_pad_size=True)

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
        traffic_sign = img_padded[ymin:ymax, xmin:xmax]
        # TODO: Consider running classifier outside once in batch
        y_hat = model(img_numpy_to_torch(traffic_sign).unsqueeze(0).cuda())[0].argmax().item()
        predicted_class = CLASS_LIST[y_hat]
        predicted_shape = predicted_class.split('-')[0]

        # Collect mask
        bool_mask = (panoptic[:, :, 0] == obj['id']).astype(np.uint8)
        # Get vertices of mask
        vertices, hull = get_corners(bool_mask)
        # Fit ellipse
        hull_mask = cv.drawContours(np.zeros_like(bool_mask), [hull], -1, (1, ), 1)
        hull_draw_points = np.stack(np.where(hull_mask), axis=1)[:, ::-1]
        ellipse = cv.fitEllipse(hull_draw_points)
        ellipse_mask = cv.ellipse(np.zeros_like(bool_mask, dtype=np.float32), ellipse, (1,), thickness=-1)
        ellipse_error = np.abs(ellipse_mask - bool_mask.astype(np.float32)).sum() / bool_mask.sum()

        if ellipse_error < 0.1:
            # Both agree on circle
            shape = 'circle'
            vertices = ellipse
            group = 1 if predicted_shape == 'circle' else 2
        else:
            # Determine polygon shape from vertices
            shape = get_shape_from_vertices(vertices)
            if shape != 'other' and predicted_shape == shape:
                # Both agree on some polygons
                group = 1
            elif predicted_shape == 'other':
                group = 3
            else:
                # Disagree but not other
                group = 2

        if shape != 'other':
            src = get_box_vertices(vertices, shape)

            # If shape is not other, draw vertices
            vert = draw_from_contours(np.zeros_like(img), src, color=[0, 255, 0])
            vert = cv.dilate(vert, None)
            vert_mask = (vert.sum(-1) > 0).astype(np.float32)[:, :, None]
            img = (1 - vert_mask) * img + vert_mask * vert
            img_tensor = img_numpy_to_torch(img)

            # Group 1: draw both vertices and patch
            if group == 1:
                # DEBUG: ratio is height / width (which one is height/width?)
                sign_width_in_mm = float(predicted_class.split('-')[1])
                sign_height_in_mm = float(predicted_class.split('-')[2])
                hw_ratio = sign_height_in_mm / sign_width_in_mm
                pixel_mm_ratio = patch_size_in_pixel / patch_size_in_mm
                sign_size_in_pixel = round(sign_height_in_mm * pixel_mm_ratio)

                sign_canonical = torch.zeros((3, sign_size_in_pixel, sign_size_in_pixel))
                tgt, sign_mask = gen_sign_mask(predicted_shape, sign_size_in_pixel, ratio=hw_ratio)
                sign_mask = torch.from_numpy(sign_mask)[None, :, :]

                # TODO: run attack, optimize patch location, etc.
                begin = (sign_size_in_pixel - patch_size_in_pixel) // 2
                end = begin + patch_size_in_pixel
                sign_canonical[begin:end, begin:end] = demo_patch
                # Crop patch that is not on the sign
                sign_canonical *= sign_mask

                # Compute perspective transform
                # OpenCV version
                # M = cv.getPerspectiveTransform(src.astype(np.float32), tgt.astype(np.float32))
                # out = cv.warpPerspective(canonical, M, (img_width, img_height))
                # out_mask = cv.warpPerspective(canonical_mask, M, (img_width, img_height))
                # TODO: may need to swap x, y here
                src = torch.from_tensor(src).unsqueeze(0)
                tgt = torch.from_tensor(tgt).unsqueeze(0)
                M = get_perspective_transform(src, tgt)
                warped_patch = homography_warp(sign_canonical.unsqueeze(0),
                                               M, (img_height, img_width),
                                               mode='bilinear',
                                               padding_mode='zeros',
                                               normalized_coordinates=False)[0]
                warped_mask = homography_warp(sign_mask.unsqueeze(0),
                                              M, (img_height, img_width),
                                              mode='nearest',
                                              padding_mode='zeros',
                                              normalized_coordinates=False)[0]
                img_tensor = (1 - warped_mask) * img_tensor + warped_mask * warped_patch

        # If shape is other, not draw anything
        else:
            img_tensor = img_numpy_to_torch(img)

        save_image(img_tensor, 'test.png')

    # emask = pad_image(ellipse_mask, pad_mode='constant')
    # save_image(torch.from_numpy(emask[ymin:ymax, xmin:xmax]), 'test.png')
    # save_image(torch.from_numpy(mask_patch[:, :, :3] / 255.).permute(2, 0, 1), 'test_mask.png')
    # save_image(torch.from_numpy(patch / 255.).permute(2, 0, 1), 'test_img.png')


def main():

    # Arguments
    min_area = 1600
    max_num_imgs = 200
    data_dir = '/data/shared/mapillary_vistas/training/'
    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    model_path = '/home/nab_126/adv-patch-bench/model_weights/resnet18_cropped_signs_good_resolution_and_not_edge_10_labels.pth.pth'

    device = 'cuda'
    # seed = 2021
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    # Create model
    mean = [0.3891, 0.3978, 0.3728]
    std = [0.1688, 0.1622, 0.1601]
    normalize = Normalize(mean, std)
    base = models.resnet18(pretrained=False)
    base.fc = nn.Linear(512, 6)

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

    demo_patch = img_numpy_to_torch(np.array(Image.open('demo.png')))

    for filename in tqdm(filenames):
        compute_example_transform(filename, model, panoptic_per_image_id,
                                  img_path, label_path, demo_patch,
                                  min_area=min_area, pad=0.05, patch_size_in_mm=150,
                                  patch_size_in_pixel=32)


if __name__ == '__main__':
    main()
