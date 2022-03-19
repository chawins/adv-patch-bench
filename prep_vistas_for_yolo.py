import argparse
import json
import pdb
from os import listdir, makedirs
from os.path import expanduser, isfile, join

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

from adv_patch_bench.models import build_classifier
from hparams import TS_COLOR_DICT, TS_COLOR_LABEL_LIST, TS_COLOR_OFFSET_DICT

def write_yolo_labels(model, label, panoptic_per_image_id, data_dir,
                      num_classes, anno_df, min_area=0, conf_thres=0.,
                      device='cuda', batch_size=128):
    img_path = join(data_dir, 'images')
    label_path = join(data_dir, 'labels_v2')
    makedirs(label_path, exist_ok=True)

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    filenames.sort()

    bbox, traffic_signs, shapes = [], [], []
    filename_to_idx = {}
    obj_idx = 0
    print('Collecting traffic signs from all images...')

    # filenames_with_small_objects_only = []
    for filename in tqdm(filenames):
        img_id = filename.split('.')[0]
        segment = panoptic_per_image_id[img_id]['segments_info']
        img_pil = Image.open(join(img_path, filename))
        img = np.array(img_pil)
        img_height, img_width, _ = img.shape
        filename_to_idx[img_id] = []
        
        # image_only_had_small_objs = True

        for obj in segment:
            # Check if bounding box is cut off at the image boundary
            xmin, ymin, width, height = obj['bbox']
            is_oob = (xmin == 0) or (ymin == 0) or \
                ((xmin + width) >= img_width) or ((ymin + height) >= img_height)

            if (obj['category_id'] != label or obj['area'] < min_area or is_oob
                    or width * height < min_area):
                continue

            # is_small = width * height < (20 * 20)
            # image_only_had_small_objs = image_only_had_small_objs and is_small

            x_center = (xmin + width / 2) / img_width
            y_center = (ymin + height / 2) / img_height
            obj_width = width / img_width
            obj_height = height / img_height
            bbox.append([x_center, y_center, obj_width, obj_height])

            # Collect traffic signs and resize them to 128x128 (same resolution
            # that classifier is trained on)
            traffic_sign = torch.from_numpy(img[ymin:ymin + height, xmin:xmin + width])
            traffic_sign = traffic_sign.permute(2, 0, 1).unsqueeze(0) / 255
            traffic_signs.append(TF.resize(traffic_sign, [128, 128]))
            filename_to_idx[img_id].append(obj_idx)

            # Get available "final_shape" from our annotation
            traffic_sign_name = f"{img_id}_{obj['id']}.png"
            row = anno_df[anno_df['filename'] == traffic_sign_name]
            if len(row) == 1:
                shapes.append(row['final_shape'].values[0])
            else:
                shapes.append('no_annotation')

            obj_idx += 1

        # if image_only_had_small_objs:
        #     filenames_with_small_objects_only.apppend(filename)

        # DEBUG
        # if len(bbox) > 500:
        #     break

    # Classify all patches
    print('==> Classifying traffic signs...')
    traffic_signs = torch.cat(traffic_signs, dim=0)
    num_samples = len(traffic_signs)
    num_batches = int(np.ceil(num_samples / batch_size))
    predicted_scores = torch.zeros((num_samples, len(TS_COLOR_LABEL_LIST)))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            begin, end = i * batch_size, (i + 1) * batch_size
            logits = model(traffic_signs[begin:end].to(device))
            predicted_scores[begin:end] = F.softmax(logits, dim=1).cpu()
    predicted_labels = predicted_scores.argmax(1)
    # Set output of low-confidence prediction to "other" clss
    predicted_labels[predicted_scores.max(1)[0] < conf_thres] = num_classes - 1

    # Resolve some errors from predicted_labels
    prob_wrong, num_fix, no_anno = 0, 0, 0

    no_anno_octagon = 0

    for i, (correct_shape, y) in enumerate(zip(shapes, predicted_labels)):
        pred_shape = TS_COLOR_LABEL_LIST[y]
        # Remove the color name
        pred_shape = '-'.join(pred_shape.split('-')[:-1])
        if correct_shape == pred_shape:
            continue
        if correct_shape == 'no_annotation':
            no_anno += 1
            if 'octagon' in pred_shape:
                no_anno_octagon += 1
            continue
        num_colors = len(TS_COLOR_DICT[correct_shape])
        if num_colors == 0:
            # If there's a mismatch, we trust `correct_shape` and replace
            # `predicted_shape` if possible (i.e, no color ambiguity)
            num_fix += 1
            predicted_labels[i] = TS_COLOR_OFFSET_DICT[correct_shape]
        else:
            # If `correct_shape` can have multiple colors, we pick the color
            # with the highest softmax score
            prob_wrong += 1
            offset = TS_COLOR_OFFSET_DICT[correct_shape]
            color_idx = predicted_scores[i][offset:offset + num_colors].argmax()
            predicted_labels[i] = offset + color_idx
    num_samples = predicted_labels.size(0)
    print(f'=> {num_fix}/{num_samples} samples were re-assigned a new and correct class.')
    print(f'=> {prob_wrong}/{num_samples} samples were re-assigned a new class '
          'based on max confidence which *may* be incorrect.')
    print(f'=> {no_anno}/{num_samples} samples were not annotated.')

    print(f'=> {no_anno_octagon}/{num_samples} octagon samples were not annotated.')

    new_img_path = join(data_dir, 'images_v2')
    makedirs(new_img_path, exist_ok=True)

    for filename, obj_idx in tqdm(filename_to_idx.items()):
        if len(obj_idx) == 0:
            continue
        img_pil = Image.open(join(img_path, filename+'.jpg'))
        full_filename = join(new_img_path, filename+'.jpg')
        img_pil.save(full_filename)

        text = ''
        for idx in obj_idx:
            class_label = int(predicted_labels[idx].item())
            x_center, y_center, obj_width, obj_height = bbox[idx]
            text += f'{class_label:d} {x_center} {y_center} {obj_width} {obj_height}\n'
        with open(join(label_path, filename + '.txt'), 'w') as f:
            f.write(text)
    
    # print()
    # print(len(filenames_with_small_objects_only))

    # with open('filenames_with_small_objects_only.txt', 'w') as f:
    #     for line in filenames_with_small_objects_only:
    #         f.write(line)
    #         f.write('\n')

    # import pdb
    # pdb.set_trace()

def main():
    # Arguments
    min_area = 3  # NOTE: We will ignore small signs in YOLO
    label_to_classify = 95      # Class id of traffic signs on Vistas
    conf_thres = 0.
    num_classes = 16
    # data_dir = expanduser('~/data/mapillary_vistas/training/')
    # model_path = expanduser('~/adv-patch-bench/results/5/checkpoint_best.pt')
    data_dir = '/data/shared/mapillary_vistas/training/'
    model_path = '/data/shared/adv-patch-bench/results/6/checkpoint_best.pt'
    # The final CSV file with our annotation. This will be used to check
    # against the prediction of the classifier
    csv_path = './mapillary_vistas_final_merged.csv'
    anno_df = pd.read_csv(csv_path)

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
    parser.add_argument('--batch-size', default=128, type=int)
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
    model = model.module

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

    write_yolo_labels(
        model, label_to_classify, panoptic_per_image_id, data_dir, num_classes,
        anno_df, min_area=min_area, conf_thres=conf_thres, device=device,
        batch_size=args.batch_size)


if __name__ == '__main__':
    main()
