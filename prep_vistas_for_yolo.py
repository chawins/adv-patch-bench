import argparse
import json
from os import listdir
from os.path import isfile, join, expanduser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from adv_patch_bench.models import build_classifier


def write_yolo_labels(model, label, panoptic_per_image_id, data_dir, num_classes,
                      min_area=0, conf_thres=0., device='cuda'):
    img_path = join(data_dir, 'images')
    label_path = join(data_dir, 'labels_v2')

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    filenames.sort()

    bbox, traffic_signs, = [], []
    filename_to_idx = {}
    obj_idx = 0
    for filename in tqdm(filenames):
        img_id = filename.split('.')[0]
        segment = panoptic_per_image_id[img_id]['segments_info']
        img_pil = Image.open(join(img_path, filename))
        img = np.array(img_pil)
        img_height, img_width, _ = img.shape
        filename_to_idx[img_id] = []

        for obj in segment:
            # Check if bounding box is cut off at the image boundary
            xmin, ymin, width, height = obj['bbox']
            is_oob = (xmin == 0) or (ymin == 0) or \
                ((xmin + width) >= img_width) or ((ymin + height) >= img_height)

            if obj['category_id'] != label or obj['area'] < min_area or is_oob:
                continue

            x_center = (xmin + width / 2) / img_width
            y_center = (ymin + height / 2) / img_height
            obj_width = width / img_width
            obj_height = height / img_height
            bbox.append([x_center, y_center, obj_width, obj_height])
            traffic_signs.append(torch.from_numpy(img[ymin:ymin + height, xmin:xmin + width]).unsqueeze(0))
            filename_to_idx[img_id].append(obj_idx)
            obj_idx += 1

    # Classify all patches
    print('==> Classifying traffic signs...')
    traffic_signs = torch.cat(traffic_signs, dim=0)
    num_samples = len(traffic_signs)
    batch_size = 200
    num_batches = int(np.ceil(num_samples / batch_size))
    predicted_labels = torch.zeros(len(num_samples))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            begin, end = i * batch_size, (i + 1) * batch_size
            logits = model(traffic_signs[begin:end].to(device))
            outputs = logits.argmax(1)
            confidence = F.softmax(logits, dim=1)
            # Set output of low-confidence prediction to "other" class
            outputs[confidence.max(1)[0] < conf_thres] = num_classes - 1
            predicted_labels[begin:end] = outputs

    for filename, obj_idx in filename_to_idx.items():
        text = ''
        for idx in obj_idx:
            class_label = predicted_labels[idx]
            x_center, y_center, obj_width, obj_height = bbox[idx]
            text += f'{class_label} {x_center} {y_center} {obj_width} {obj_height}\n'
        with open(join(label_path, filename + '.txt'), 'w') as f:
            f.write(text)


def main():
    # Arguments
    min_area = 0  # NOTE: We will ignore small signs in YOLO
    label_to_classify = 95      # Class id of traffic signs on Vistas
    conf_thres = 0.
    num_classes = 16
    data_dir = expanduser('~/data/mapillary_vistas/training/')
    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    model_path = expanduser('~/adv-patch-bench/results/5/checkpoint_best.pt')

    device = 'cuda'
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Don't change this and don't set it in command line
    parser = argparse.ArgumentParser(description='Manually set args', add_help=False)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model on ImageNet-1k')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--full-precision', action='store_false')
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--resume', default=model_path, type=str, help='path to latest checkpoint')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dataset', default='mtsd', type=str, help='Dataset')
    parser.add_argument('--num-classes', default=num_classes, type=int, help='Number of classes')
    args = parser.parse_args()

    model, _, _ = build_classifier(args)

    # Read in panoptic file
    panoptic_json_path = f'{data_dir}/v2.0/panoptic/panoptic_2020.json'
    with open(expanduser(panoptic_json_path)) as panoptic_file:
        panoptic = json.load(panoptic_file)

    # Convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic['annotations']:
        panoptic_per_image_id[annotation['image_id']] = annotation

    # Convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic['categories']:
        panoptic_category_per_id[category['id']] = category

    write_yolo_labels(model, label_to_classify, panoptic_per_image_id, data_dir, num_classes,
                      min_area=min_area, conf_thres=conf_thres, device=device)


if __name__ == '__main__':
    main()
