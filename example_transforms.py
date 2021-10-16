import json
import os
from os import listdir
from os.path import isfile, join

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from adv_patch_bench.datasets.utils import (get_box, get_image_files,
                                            img_numpy_to_torch,
                                            load_annotation, pad_image)
from adv_patch_bench.models.common import Normalize
from adv_patch_bench.transforms import (gen_sign_mask, get_box_vertices,
                                        get_corners, get_shape_from_vertices, relight_range)
from classify_traffic_signs import draw_from_contours

DATASET = 'mapillaryvistas'
# DATASET = 'bdd100k'

if DATASET == 'mapillaryvistas':
    TRAFFIC_SIGN_LABEL = 95
elif DATASET == 'bdd100k':
    TRAFFIC_SIGN_LABEL = 'traffic sign'

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
                              min_area=0, pad=0.1, patch_size_in_mm=150,
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

    results = []
    show_list = []
    # Crop the specified object
    for obj in segment:

        # Check if bounding box is cut off at the image boundary
        xmin, ymin, width, height = obj['bbox']
        is_oob = (xmin == 0) or (ymin == 0) or \
            ((xmin + width) >= img_width - 1) or ((ymin + height) >= img_height - 1)
        if obj['category_id'] != TRAFFIC_SIGN_LABEL or is_oob:
            continue

        # Collect mask
        extra_pad = int(max(width, height) * 0.2)
        ymin, xmin = max(0, ymin + pad_size - extra_pad), max(0, xmin + pad_size - extra_pad)
        temp_mask = (id_padded[ymin:ymin + height + 2 * extra_pad,
                               xmin:xmin + width + 2 * extra_pad] == obj['id']).astype(np.uint8)

        if temp_mask.sum() < min_area:
            continue

        # Get refined crop patch
        ymin_, ymax_, xmin_, xmax_ = get_box(temp_mask, pad)
        ymin, ymax, xmin, xmax = ymin + ymin_, ymin + ymax_, xmin + xmin_, xmin + xmax_
        bool_mask = (id_padded[ymin:ymax, xmin:xmax] == obj['id']).astype(np.uint8)
        height, width = bool_mask.shape
        if height != width:
            # NOTE: This is very weird and rare corner case where the traffic
            # sign somehow takes majority of the whole image
            # print(height, width)
            # print(img.shape)
            # print(ymin, ymax, xmin, xmax)
            # import pdb
            # pdb.set_trace()
            # size = int((height + width) / 2)
            # ymax, xmax = ymin + size, xmin + size
            continue
        size = height

        # Make sure that bounding box is square and add some padding to avoid
        # cutting into the sign
        # size = max(width, height)
        # xpad, ypad = int((size - width) / 2), int((size - height) / 2)
        # extra_obj_pad = int(pad * size)
        # size += 2 * extra_obj_pad
        # xmin += pad_size - xpad - extra_obj_pad
        # ymin += pad_size - ypad - extra_obj_pad
        # xmax, ymax = xmin + size, ymin + size
        traffic_sign = img_numpy_to_torch(img_padded[ymin:ymax, xmin:xmax])
        # TODO: Consider running classifier outside once in batch
        y_hat = model(traffic_sign.unsqueeze(0).cuda())[0].argmax().item()
        predicted_class = CLASS_LIST[y_hat]
        predicted_shape = predicted_class.split('-')[0]
        # print(f'==> predicted_class: {predicted_class}')

        # DEBUG
        # save_image(traffic_sign, 'test.png')
        # import pdb
        # pdb.set_trace()

        # FIXME
        if DATASET == 'bdd100k':
            xmin_, ymin_, width_, height_ = obj['bbox']
            bool_mask[:max(0, ymin_-10), :] = 0
            bool_mask[min(ymin_+height_+10, img_height):, :] = 0
            bool_mask[:, :max(0, xmin_-10)] = 0
            bool_mask[:, min(xmin_+width_+10, img_width):] = 0

        # Get vertices of mask
        vertices, hull = get_corners(bool_mask)
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
        # print(f'==> shape: {shape}, group: {group}')
        # print(vertices)

        if shape != 'other':
            tgt = get_box_vertices(vertices, shape).astype(np.int64)
            # Filter some vertices that might be out of bound
            if (tgt < 0).any() or (tgt >= size).any():
                # group = 2
                # tgt = np.array([t for t in tgt if 0 <= t[0] < size and 0 <= t[1] < size])
                # This generally happens when the fitted ellipse is incorrect
                # and the vertices lie outside of the patch
                results.append([traffic_sign, shape, predicted_shape, predicted_class, 4])
                continue

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
                                              mode='bicubic',
                                              padding_mode='zeros')[0]
                warped_mask = transform_func(patch_mask.unsqueeze(0),
                                             M, (size, size),
                                             mode='nearest',
                                             padding_mode='zeros')[0]
                # print(np.histogram(warped_patch.numpy()))
                # print(warped_patch.min(), warped_patch.max())
                # # Assume that 80% of pixels have one 255 (e.g., red, blue) and
                # # 20% of pixels are white (255, 255, 255).
                # old_patch = torch.masked_select(traffic_sign, torch.from_numpy(bool_mask).bool())
                # mu_1, sigma_1 = old_patch.mean(), old_patch.std()
                # # mu_0, sigma_0 = 0.4666667, 0.4988876
                # old_patch -= old_patch.min()
                # old_patch /= old_patch.max()
                # old_patch_q = (old_patch > 0.5).float()
                # mu_0, sigma_0 = old_patch_q.mean(), old_patch_q.std()
                # alpha = sigma_1 / sigma_0
                # beta = mu_1 - mu_0 * alpha
                old_patch = torch.masked_select(traffic_sign, torch.from_numpy(bool_mask).bool())
                alpha, beta = relight_range(old_patch.numpy().reshape(-1, 1))
                warped_patch.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)

                traffic_sign = (1 - warped_mask) * traffic_sign + warped_mask * warped_patch

                # DEBUG
                save_image(traffic_sign, 'test.png')
                import pdb
                pdb.set_trace()

            # DEBUG
            # if group in (1, 2):
            #     show_list.append(traffic_sign)
            # if len(show_list) > 100:
            #     save_image(show_list, 'test.png')
            #     import pdb
            #     pdb.set_trace()

        # DEBUG
        # if 'triangle' in shape:
        #     save_image(cropped_sign, 'test.png')
        #     import pdb
        #     pdb.set_trace()

        results.append([traffic_sign, shape, predicted_shape, predicted_class, group])
    return results


def main():

    # Arguments
    min_area = 1600
    max_num_imgs = 200

    if DATASET == 'mapillaryvistas':
        data_dir = '/data/shared/mapillary_vistas/training/'
    elif DATASET == 'bdd100k':
        data_dir = '/data/shared/bdd100k/images/10k/train/'

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
    if DATASET == 'mapillaryvistas':
        panoptic_json_path = f'{data_dir}/v2.0/panoptic/panoptic_2020.json'
    elif DATASET == 'bdd100k':
        panoptic_json_path = '/data/shared/bdd100k/labels/pan_seg/polygons/pan_seg_train.json'

    with open(panoptic_json_path) as panoptic_file:
        panoptic = json.load(panoptic_file)

    if DATASET == 'mapillaryvistas':
        panoptic_per_image_id = {}
        for annotation in panoptic['annotations']:
            panoptic_per_image_id[annotation['image_id']] = annotation

        # Convert category infos to category_id indexed dictionary
        panoptic_category_per_id = {}
        for category in panoptic['categories']:
            panoptic_category_per_id[category['id']] = category

    elif DATASET == 'bdd100k':
        # creating same mapping for bdd100k
        panoptic_per_image_id = {}
        for image_annotation in tqdm(panoptic):
            filename = image_annotation['name']
            image_id = filename.split('.jpg')[0]
            annotation = {}
            annotation['filename'] = filename
            annotation['image_id'] = image_id
            segments_info = []
            for label in image_annotation['labels']:
                label_dict = {}

                # TODO: check if occluded and exclude if True
                if label['category'] == 'traffic sign':
                    # if label['category'] == 'traffic sign' or label['category'] == 'traffic sign frame':
                    # label_dict['id'] = label['id']
                    label_dict['id'] = 26
                    label_dict['category_id'] = label['category']
                    for sign in label['poly2d']:
                        vertices = sign['vertices']
                        vertices = np.array(vertices)

                        x_cords, y_cords = vertices[:, 0], vertices[:, 1]
                        xmin = min(x_cords)
                        xmax = max(x_cords)
                        ymin = min(y_cords)
                        ymax = max(y_cords)
                        width = xmax-xmin
                        height = ymax-ymin

                        label_dict['area'] = int(width) * int(height)
                        label_dict['bbox'] = [int(xmin), int(ymin), int(width), int(height)]
                        segments_info.append(label_dict)
            annotation['segments_info'] = segments_info
            panoptic_per_image_id[image_id] = annotation

    # mapillary
    if DATASET == 'mapillaryvistas':
        img_path = join(data_dir, 'images')
        label_path = join(data_dir, 'v2.0/panoptic/')
        filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    elif DATASET == 'bdd100k':
        # data_dir = '/data/shared/bdd100k/images/10k/train/'
        label_path = '/data/shared/bdd100k/labels/pan_seg/bitmasks/train/'
        filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        img_path = data_dir

    np.random.shuffle(filenames)

    # demo_patch = img_numpy_to_torch(np.array(Image.open()))
    demo_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
    demo_patch = resize(demo_patch, (32, 32))

    column_names = ['filename', 'shape', 'predicted_shape', 'predicted_class', 'group', 'row']
    df_data = []

    num_images_processed = 0
    num_type_errors = 0
    num_runtime_errors = 0

    num_plots = 300
    fig, ax = plt.subplots(int(num_plots/5), 5)
    fig.set_figheight(int(num_plots/5)*10)
    fig.set_figwidth(5 * 5)
    i = 0

    print('[INFO] running detection algorithm')
    for filename in tqdm(filenames):
        num_images_processed += 1
        # try:
        #     transformed_images = compute_example_transform(filename, model, panoptic_per_image_id,
        #                                                    img_path, label_path, demo_patch,
        #                                                    min_area=min_area, pad=0.1, patch_size_in_mm=150,
        #                                                    patch_size_in_pixel=32)

        #     for img in transformed_images:
        #         col = i % 5
        #         row = i // 5

        #         cropped_image, shape, predicted_shape, predicted_class, group = img
        #         cropped_image = cropped_image.permute(1, 2, 0)
        #         title = '{} contour:{} resnet:{}'.format(filename, shape, predicted_shape)
        #         if col == 0:
        #             title = 'row #{}  '.format(row) + title
        #         ax[row][col].set_title(title)
        #         ax[row][col].imshow(cropped_image.numpy())

        #         df_data.append([filename, shape, predicted_shape, predicted_class, group, row])

        #         if i == num_plots-1:
        #             plt.savefig('{}.png'.format(DATASET), bbox_inches='tight', pad_inches=0)
        #             break

        #         i += 1
        # except TypeError:
        #     num_type_errors += 1
        # except RuntimeError:
        #     num_runtime_errors += 1
        #     continue

        transformed_images = compute_example_transform(filename, model, panoptic_per_image_id,
                                                       img_path, label_path, demo_patch,
                                                       min_area=min_area, pad=0.1, patch_size_in_mm=150,
                                                       patch_size_in_pixel=32)

        # for img in transformed_images:
        #     col = i % 5
        #     row = i // 5

        #     cropped_image, shape, predicted_shape, predicted_class, group = img
        #     cropped_image = cropped_image.permute(1, 2, 0)
        #     title = '{} contour:{} resnet:{}'.format(filename, shape, predicted_shape)
        #     if col == 0:
        #         title = 'row #{}  '.format(row) + title
        #     ax[row][col].set_title(title)
        #     ax[row][col].imshow(cropped_image.numpy())

        #     df_data.append([filename, shape, predicted_shape, predicted_class, group, row])

        #     if i == num_plots-1:
        #         plt.savefig('{}.png'.format(DATASET), bbox_inches='tight', pad_inches=0)
        #         break

        #     i += 1

    print('[INFO] saving csv')
    df = pd.DataFrame(df_data, columns=column_names)
    df['dataset'] = DATASET
    df.to_csv('{}.csv'.format(DATASET), index=False)

    print('percentage type errors:', num_type_errors/num_images_processed)
    print('percentage runtime errors:', num_runtime_errors/num_images_processed)


if __name__ == '__main__':
    main()
