import argparse
import json
import os
from collections import OrderedDict
from os import listdir
from os.path import isfile, join

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from kornia.geometry.transform import (get_perspective_transform, resize,
                                       warp_affine, warp_perspective)
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm

from adv_patch_bench.models import build_classifier
from adv_patch_bench.transforms import (gen_sign_mask, get_box_vertices,
                                        get_corners, get_shape_from_vertices,
                                        relight_range)
from adv_patch_bench.utils import (draw_from_contours, get_box,
                                   img_numpy_to_torch, pad_image)

DATASET = 'mapillaryvistas'
# DATASET = 'bdd100k'

if DATASET == 'mapillaryvistas':
    TRAFFIC_SIGN_LABEL = 95
elif DATASET == 'bdd100k':
    TRAFFIC_SIGN_LABEL = 'traffic sign'

CLASS_LIST = [
    'circle-750.0',
    'triangle-900.0',
    'octagon-915.0',
    'other-0.0-0.0',
    'triangle_inverted-1220.0',
    'diamond-600.0',
    'diamond-915.0',
    'square-600.0',
    'rect-458.0-610.0',
    'rect-762.0-915.0',
    'rect-915.0-1220.0',
    'pentagon-915.0'
]

SHAPE_LIST = [
    'circle',
    'triangle',
    'triangle_inverted',
    'diamond',
    'square',
    'rect',
    'pentagon',
    'octagon',
    'other'
]

# CLASS_LIST = ['octagon-915.0-915.0',
#               'diamond-915.0-915.0',
#               'pentagon-915.0-915.0',
#               'rect-915.0-1220.0',
#               'rect-762.0-915.0',
#               'triangle-900.0',
#               'circle-750.0',
#               'triangle_inverted-1220.0-1220.0',
#               'rect-458.0-610.0',
#               'other-0.0-0.0']


def get_args_parser():
    parser = argparse.ArgumentParser(description='Part classification', add_help=False)
    parser.add_argument('--data', default='~/data/shared/', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model on ImageNet-1k')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size per device.')
    parser.add_argument('--full-precision', action='store_true')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB')
    # TODO
    parser.add_argument('--dataset', required=True, type=str, help='Dataset')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='Number of classes')
    parser.add_argument('--experiment', required=False, type=str,
                        help='Type of experiment to run')
    # Unused
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    return parser


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
        
        model_input = img_padded[ymin:ymax, xmin:xmax].astype('uint8')
        bool_mask_3d = np.expand_dims(bool_mask, axis=2)

        # element-wise multiplation
        # model_input = model_input * bool_mask_3d
        PIL_traffic_sign = Image.fromarray(model_input, 'RGB')

        transform_list = [
            # transforms.RandomEqualize(p=1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        model_traffic_sign = transform(PIL_traffic_sign)

        traffic_sign = img_numpy_to_torch(img_padded[ymin:ymax, xmin:xmax])
        # TODO: Consider running classifier outside once in batch

        # print(torch.min(model_traffic_sign))
        # print(torch.max(model_traffic_sign))
        y_hat = model(model_traffic_sign.unsqueeze(0).cuda())[0].argmax().item()

        predicted_class = CLASS_LIST[y_hat]
        predicted_shape = predicted_class.split('-')[0]
        # print(f'==> predicted_class: {predicted_class}')

        # DEBUG
        # save_image(traffic_sign_equalized, 'test.png')
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
                    # (shape == 'rect' and not (predicted_shape == 'other' or predicted_shape == 'diamond'))):
                    # (shape == 'rect' and (predicted_shape == 'circle' or predicted_shape == 'triangle'))):
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
                # print(shape, predicted_class, group)
                # save_image(traffic_sign, 'test.png')
                # import pdb
                # pdb.set_trace()

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

        results.append([traffic_sign, obj['id'], shape, predicted_shape, predicted_class, group])
    return results


def main(args):

    # Arguments
    min_area = 1600
    max_num_imgs = 200

    if DATASET == 'mapillaryvistas':
        data_dir = '/data/shared/mapillary_vistas/training/'
    elif DATASET == 'bdd100k':
        data_dir = '/data/shared/bdd100k/images/10k/train/'

    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    # model_path = '/home/nab_126/adv-patch-bench/model_weights/resnet18_cropped_signs_good_resolution_and_not_edge_10_labels.pth'

    device = 'cuda'
    # seed = 2021
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    # Create model
    # mean = [0.3867, 0.3993, 0.3786]
    # std = [0.1795, 0.1718, 0.1714]
    # normalize = Normalize(mean, std)
    # base = models.resnet18(pretrained=False)
    # base.fc = nn.Linear(512, 10)

    # if os.path.exists(model_path):
    #     print('Loading model weights...')
    #     base.load_state_dict(torch.load(model_path))
    # else:
    #     raise ValueError('Model weight not found!')

    # model = nn.Sequential(normalize, base).to(device).eval()

    # Compute stats of best model
    model = build_classifier(args)[0]
    model.eval()

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

    np.random.seed(1111)
    np.random.shuffle(filenames)

    # demo_patch = img_numpy_to_torch(np.array(Image.open()))
    demo_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
    demo_patch = resize(demo_patch, (32, 32))

    column_names = ['filename', 'object_id', 'shape', 'predicted_shape',
                    'predicted_class', 'group', 'batch_number', 'row', 'column']
    df_data = []

    num_plots = 100
    num_images_processed = 0

    fig_objects = []
    class_counts = np.zeros((len(SHAPE_LIST), 3), dtype=np.int8)
    class_batch_numbers = np.zeros((len(SHAPE_LIST), 3), dtype=np.int8)

    for i in range(len(SHAPE_LIST)):
        fig_objects.append([])
        for group in range(3):
            fig, ax = plt.subplots(int(num_plots/5), 5)
            fig.set_figheight(int(num_plots/5)*5)
            fig.set_figwidth(5 * 5)
            fig_objects[i].append((fig, ax))

    # print(len(fig_objects))
    # print(len(fig_objects[0]))
    # print(len(fig_objects[1]))

    COLUMN_NAMES = 'abcdefghijklmnopqrstuvwxyz'
    ROW_NAMES = list(range(num_plots))

    plot_folder = 'mapillaryvistas_plots_model_3'
    
    print('[INFO] running detection algorithm')
    for filename in tqdm(filenames):
        transformed_images = compute_example_transform(filename, model, panoptic_per_image_id,
                                                       img_path, label_path, demo_patch,
                                                       min_area=min_area, pad=0., patch_size_in_mm=150,
                                                       patch_size_in_pixel=32)

        for img in transformed_images:
            try:
                cropped_image, obj_id, shape, predicted_shape, predicted_class, group = img
            except ValueError:
                continue

            num_images_processed += 1
            cropped_image = cropped_image.permute(1, 2, 0)

            shape_index = SHAPE_LIST.index(shape)
            col = class_counts[shape_index, group-1] % 5
            row = class_counts[shape_index, group-1] // 5

            col_name = COLUMN_NAMES[col]
            row_name = ROW_NAMES[row]
            if shape != 'triangle_inverted':
                title = 'id:({}, {}) | pred({}, {}) | gridcord({}{})'.format(filename[:6], obj_id, shape, predicted_class, row_name, col_name)
            else:
                title = 'id:({}, {}) | pred({}, {}) | gridcord({}{})'.format(filename[:6], obj_id, 't_inv', predicted_class, row_name, col_name)

            fig_objects[shape_index][group-1][1][row][col].set_title(title)
            fig_objects[shape_index][group-1][1][row][col].imshow(cropped_image.numpy())

            df_data.append([filename, obj_id, shape, predicted_shape, predicted_class, group, class_batch_numbers[shape_index][group-1], row_name, col_name])
            
            if not os.path.exists('{}/{}/{}'.format(plot_folder, shape, group)):
                os.makedirs('{}/{}/{}'.format(plot_folder, shape, group), exist_ok=False)
            if class_counts[shape_index][group-1] == num_plots-1:
                fig_objects[shape_index][group-1][0].savefig('{}/{}/{}/batch_{}.png'.format(plot_folder, shape, group, class_batch_numbers[shape_index][group-1]), bbox_inches='tight', pad_inches=0)
                class_counts[shape_index][group-1] = -1
                class_batch_numbers[shape_index][group-1] += 1
 
            class_counts[shape_index][group-1] += 1
            
            if num_images_processed % 100 == 0:
                df = pd.DataFrame(df_data, columns=column_names)
                df.to_csv('{}_model_3.csv'.format(DATASET), index=False)

            if num_images_processed % 1000 == 0:
                for i in range(len(SHAPE_LIST)):
                    for group in range(1, 4):
                        if class_counts[i][group-1] < 100:
                            if not os.path.exists('{}/{}/{}'.format(plot_folder, SHAPE_LIST[i], group)):
                                os.makedirs('{}/{}/{}'.format(plot_folder, SHAPE_LIST[i], group), exist_ok=False)
                            fig_objects[i][group-1][0].savefig('{}/{}/{}/batch_{}.png'.format(plot_folder, SHAPE_LIST[i], group, class_batch_numbers[i][group-1]), bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Example Transform', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
