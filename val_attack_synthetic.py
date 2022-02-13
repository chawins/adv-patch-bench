# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import pdb
import pickle
import sys
from pathlib import Path
from threading import Thread
from ast import literal_eval


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as T
from kornia.geometry.transform import (get_perspective_transform, resize,
                                       warp_affine, warp_perspective)
from kornia import augmentation as K
from kornia.constants import Resample

from PIL import Image
from tqdm import tqdm
from cv2 import getAffineTransform

from adv_patch_bench.attacks.rp2 import RP2AttackModule
from adv_patch_bench.utils.image import pad_image
from example_transforms import get_sign_canonical
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (LOGGER, box_iou, check_dataset,
                                  check_img_size, check_requirements,
                                  check_yaml, coco80_to_coco91_class, colorstr,
                                  increment_path, non_max_suppression,
                                  print_args, scale_coords, xywh2xyxy,
                                  xyxy2xywh)
from yolov5.utils.metrics import ConfusionMatrix, ap_per_class
from yolov5.utils.plots import output_to_target, plot_images, plot_val_study
from yolov5.utils.torch_utils import select_device, time_sync


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def transform_and_apply_patch(image, adv_patch_cropped, shape, predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad):
    # print('shape', adv_patch_cropped.shape)
    # qqq
    patch_size_in_pixel = adv_patch_cropped.shape[1]
    patch_size_in_mm = 250
    sign_canonical, sign_mask, src = get_sign_canonical(
        shape, predicted_class, patch_size_in_pixel, patch_size_in_mm)

    alpha = row['alpha']
    beta = row['beta']
    patch_cropped = adv_patch_cropped.clone()
    patch_cropped.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)
    sign_size_in_pixel = sign_canonical.size(-1)
    # Patch location (here is center). Should match the
    # location used during attack
    begin = (sign_size_in_pixel - patch_size_in_pixel) // 2
    end = begin + patch_size_in_pixel
    sign_canonical[:-1, begin:end, begin:end] = patch_cropped
    sign_canonical[-1, begin:end, begin:end] = 1

    # Crop patch that is not on the sign
    sign_canonical *= sign_mask
    src = np.array(src, dtype=np.float32)

    # if annotated, then use those points
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)
        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad

    # if we flagged with 'use_polygon', then use those 'tgt_polygon'
    elif not pd.isna(row['tgt_polygon']):
        tgt = np.array(literal_eval(row['tgt_polygon']), dtype=np.float32)
        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size

        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

    else:
        tgt = np.array(literal_eval(row['tgt']), dtype=np.float32)
        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size

        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

    # moving patch
    sign_height = max(tgt[:, 1]) - min(tgt[:, 1])
    tgt[:, 1] += sign_height * 0.3
    
    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    cur_shape = image.shape[1:]
    warped_patch = transform_func(sign_canonical.unsqueeze(0),
                                M, cur_shape,
                                mode='bicubic',
                                padding_mode='zeros')[0]
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[-1].unsqueeze(0)
    image_with_transformed_patch = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch[:-1]
    return image_with_transformed_patch


@torch.no_grad()
def run(args,
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        **kwargs,
        ):

    apply_patch = args.apply_patch
    synthetic = args.synthetic
    random_patch = args.random_patch
    save_exp_metrics = args.save_exp_metrics
    plot_single_images = args.plot_single_images
    plot_octagons = args.plot_octagons
    num_bg = args.num_bg
    load_patch = args.load_patch

    torch.manual_seed(1111)
    np.random.seed(1111)

    DATASET_NAME = 'mapillary' if 'mapillary' in data else 'mtsd'
    try:
        metrics_df = pd.read_csv('runs/results.csv')
        metrics_df = metrics_df.replace({'apply_patch': 'True', 'random_patch': 'True'}, 1)
        metrics_df = metrics_df.replace({'apply_patch': 'False', 'random_patch': 'False'}, 0)
        metrics_df['apply_patch'] = metrics_df['apply_patch'].astype(float).astype(bool)
        metrics_df['random_patch'] = metrics_df['random_patch'].astype(float).astype(bool)
        metrics_df_grouped = metrics_df.groupby(by=['dataset', 'apply_patch', 'random_patch']).count().reset_index()
        metrics_df_grouped = metrics_df_grouped[(metrics_df_grouped['dataset'] == DATASET_NAME) & (
            metrics_df_grouped['apply_patch'] == apply_patch) & (metrics_df_grouped['random_patch'] == random_patch)]['name']
        exp_number = 0 if not len(metrics_df_grouped) else metrics_df_grouped.item()
    except FileNotFoundError:
        exp_number = 0

    if apply_patch:
        if random_patch:
            name += f'_{DATASET_NAME}_random_patch_{exp_number}'
        else:
            name += f'_{DATASET_NAME}_rp2_patch_{exp_number}'
    else:
        name += f'_{DATASET_NAME}_no_patch_{exp_number}'

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    print(data['train'])

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]

    img_size = (int(imgsz * 0.75), imgsz)
    img_height, img_width = img_size

    if apply_patch and not synthetic:
        # df = pd.read_csv('mapillary_vistas_final_merged_new.csv')
        df = pd.read_csv('mapillary_vistas_final_merged.csv')
        df['tgt_final'] = df['tgt_final'].apply(literal_eval)
        df = df[df['final_shape'] != 'other-0.0-0.0']
        print(df.shape)
        print(df.groupby(by=['final_shape']).count())

        df_use_polygons = df[~df['tgt_polygon'].isna()]
        # adv_patch_cropped = resize(adv_patch_cropped, (32, 32))
    elif synthetic:
        obj_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, return_transform=True)
        mask_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, resample=Resample.NEAREST)
        resize_transform = torchvision.transforms.Resize(size=img_size)
        
    # EDIT: Randomly select backgrounds and resize
    bg_size = img_size
    # bg_dir = '/data/shared/mtsd_v2_fully_annotated/test'
    bg_dir = '/data/shared/mtsd_v2_fully_annotated/train'
    all_bgs = os.listdir(os.path.expanduser(bg_dir))
    idx = np.arange(len(all_bgs))
    np.random.shuffle(idx)
    backgrounds = torch.zeros((num_bg, 3) + bg_size, )
    for i, index in enumerate(idx[:num_bg]):
        bg = torchvision.io.read_image(os.path.join(bg_dir, all_bgs[index])) / 255
        backgrounds[i] = T.resize(bg, bg_size, antialias=True)
    torchvision.utils.save_image(backgrounds, 'backgrounds.png')

    # EDIT: set up attack
    obj_size = int(min(bg_size) * 0.1) * 2        # (256, 256)
    obj_size = (obj_size, obj_size)
    asset_dir = 'attack_assets/'
    # TODO: Allow data parallel?
    attack = RP2AttackModule(None, model, None, None, None)
    obj = np.array(Image.open(os.path.join(asset_dir, 'stop_sign.png')).convert('RGBA')) / 255
    obj_mask = torch.from_numpy(obj[:, :, 0] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj[:, :, :-1]).float().permute(2, 0, 1)
    # Resize and put object in the middle of zero background
    pad_size = [(bg_size[1] - obj_size[1]) // 2, (bg_size[0] - obj_size[0]) // 2]  # left/right, top/bottom
    obj = T.resize(obj, obj_size, antialias=True)
    obj = T.pad(obj, pad_size)
    obj_mask = T.resize(obj_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
    obj_mask = T.pad(obj_mask, pad_size)
    # Define patch location and size
    patch_mask = torch.zeros_like(obj_mask)
    # Example: 10x10-inch patch in the middle of 36x36-inch sign
    mid_height = bg_size[0] // 2 + 60
    mid_width = bg_size[1] // 2
    patch_size = 10
    h = int(patch_size / 36 / 2 * obj_size[0])
    w = int(patch_size / 36 / 2 * obj_size[1])

    patch_mask[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w] = 1

    torchvision.utils.save_image(obj, 'obj.png')
    torchvision.utils.save_image(obj_mask, 'obj_mask.png')
    torchvision.utils.save_image(patch_mask, 'patch_mask.png')

    img = patch_mask + (1 - patch_mask) * (obj_mask * obj + (1 - obj_mask) * backgrounds[0])
    torchvision.utils.save_image(img, 'img.png')

    if apply_patch:
        if load_patch == 'arrow':
            # Load 'arrow on checkboard' patch if specified
            # adv_patch_cropped = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
            # adv_patch_cropped = resize(adv_patch_cropped, (2*h, 2*w))
            # print(adv_patch_cropped.shape)
            # adv_patch = torch.zeros_like(patch_mask)
            # adv_patch[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w] = adv_patch_cropped
            adv_patch_cropped = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
            adv_patch_cropped = resize(adv_patch_cropped, (2*h, 2*w))
            adv_patch = torch.zeros_like(patch_mask)
            adv_patch = adv_patch.repeat(3, 1, 1)
            adv_patch[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w] = adv_patch_cropped

        elif load_patch != '':
            # Load patch if specified
            adv_patch = pickle.load(open(load_patch, 'rb'))
            adv_patch_cropped = adv_patch[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w]
        elif random_patch:
            # Random patch
            adv_patch_cropped = torch.rand(3, 2*h, 2*w)
            adv_patch = torch.zeros_like(patch_mask)
            adv_patch = adv_patch.repeat(3, 1, 1)
            adv_patch[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w] = adv_patch_cropped
        else:
            # Otherwise, generate a new adversarial patch
            if True: # TODO: add argument for how to generate patch (using synthetic sign or real sign)
                with torch.enable_grad():
                    adv_patch = attack.attack(obj.cuda(), obj_mask.cuda(), patch_mask.cuda(), backgrounds.cuda())
                qqq
            elif True: # TODO: add argument for how to generate patch (using synthetic sign or real sign)
                attack_images = []
                s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
                pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

                # TODO: split into train and val set and change pbar
                for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
                    for image_i, path in enumerate(paths):
                        filename = path.split('/')[-1]

                        #TODO: only for debugging
                        if filename == '8lkcFc59-2RgSU203mlYEQ.jpg':
                            continue

                        img_df = df[df['filename_y'] == filename]
                        if len(img_df) == 0:
                            continue
                        for _, row in img_df.iterrows():
                            (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
                            predicted_class = row['final_shape']
                            shape = predicted_class.split('-')[0]

                            # only apply patch to octagons
                            if shape != 'octagon':
                                continue
                            # attack_images.append([im[image_i], [shape, predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad]])
                            attack_images.append([im[image_i], [shape, predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad], str(filename)])
                    if len(attack_images) > 10:
                        break
                # for i in attack_images:
                #     print(f'tmp/{i[2]}.png')
                #     print(i[0].shape)
                #     print(max(i[0][0][0]))
                #     torchvision.utils.save_image(i[0]/255, f'tmp/{i[2]}.png')
                # qqq

                with torch.enable_grad():
                    adv_patch = attack.transform_and_attack(attack_images)
                    # TODO: change name of patch which is saved
                    
            adv_patch = adv_patch[0].detach()
            adv_patch = adv_patch.cpu().float()
            patch_path = './adv_patch.pkl'
            print(f'Saving the generated adv patch to {patch_path}...')
            pickle.dump(adv_patch, open(patch_path, 'wb'))
            adv_patch_cropped = adv_patch[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w]
            
            # TODO: add argument
            # adv_patch_cropped = adv_patch

        f = os.path.join(save_dir, 'adversarial_patch.png')
        print(f'Saving the patch to {f}...')
        torchvision.utils.save_image(adv_patch, f)
        f = os.path.join(save_dir, 'adversarial_patch_cropped.png')
        print(f'Saving the cropped patch to {f}...')
        torchvision.utils.save_image(adv_patch_cropped, f)

    seen = 0

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = len(names)
    print('Class names: ', names)

    # TODO move to label file instead of adding synthetic stop sign as class here
    SYNTHETIC_STOP_SIGN_CLASS = len(names)
    names[SYNTHETIC_STOP_SIGN_CLASS] = 'synthetic_stop_sign'
    nc += 1

    confusion_matrix = ConfusionMatrix(nc=nc)

    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    # DEBUG
    # if apply_patch:
    #     demo_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
    #     # demo_patch = resize(adv_patch_cropped, (32, 32))
    #     demo_patch = resize(demo_patch, (32, 32))
    #     f = os.path.join(save_dir, 'adversarial_patch.png')
    #     torchvision.utils.save_image(demo_patch, f)

    

    num_errors = 0
    num_detected = 0
    num_octagon_labels = 0
    num_octagon_with_patch = 0

    if plot_octagons:
        shape_to_plot_data = {}
        shape_to_plot_data['octagon'] = []
    num_samples = 0
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # DEBUG
        num_samples += im.shape[0]
        if batch_i == 200:
            break
        for image_i, path in enumerate(paths):
            if apply_patch and not synthetic:
                filename = path.split('/')[-1]
                img_df = df[df['filename_y'] == filename]

                if len(img_df) == 0:
                    continue
                # Apply patch on all of the signs on this image
                for _, row in img_df.iterrows():
                    (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
                    predicted_class = row['final_shape']
                    shape = predicted_class.split('-')[0]

                    # only apply patch to octagons
                    if shape != 'octagon':
                        continue
                    num_octagon_with_patch += shape == 'octagon'

                    # traffic_sign = (1 - alpha_mask) * im[image_i] / 255 + alpha_mask * warped_patch[:-1]
                    # # traffic_sign.clamp_(0, 1)
                    # # im[image_i] = (traffic_sign * 255).byte()
                    # im[image_i] = traffic_sign * 255
                    
                    # def transform_and_apply_patch(image, adv_patch_cropped, shape, predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad):
                    im[image_i] = transform_and_apply_patch(im[image_i], adv_patch_cropped, shape, predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad) * 255
                    print(im[image_i].shape)

            elif synthetic:
                orig_shape = im[image_i].shape[1:]
                resized_img = resize_transform(im[image_i])

                if apply_patch:
                    adv_obj = patch_mask * adv_patch + (1 - patch_mask) * obj
                else:
                    adv_obj = obj

                adv_obj, tf_params = obj_transforms(adv_obj)
                adv_obj.clamp_(0, 1)
                # TODO: clean this part
                num_eot = 1
                obj_mask = obj_mask.cuda()
                obj_mask_dup = obj_mask.expand(num_eot, -1, -1, -1)

                tf_params = tf_params.cuda()
                o_mask = mask_transforms.apply_transform(obj_mask_dup, None, transform=tf_params)

                o_mask = o_mask.cpu()
                indices = np.where(o_mask[0][0] == 1)
                x_min, x_max = min(indices[1]), max(indices[1])
                y_min, y_max = min(indices[0]), max(indices[0])

                label = [
                    image_i,
                    SYNTHETIC_STOP_SIGN_CLASS,
                    (x_min + x_max) / (2 * img_width),
                    (y_min + y_max) / (2 * img_height),
                    (x_max - x_min) / img_width,
                    (y_max - y_min) / img_height,
                ]
                targets = torch.cat((targets, torch.unsqueeze(torch.Tensor(label), 0)))
                adv_img = o_mask * adv_obj + (1 - o_mask) * resized_img/255
                reresize_transform = torchvision.transforms.Resize(size=orig_shape)
                im[image_i] = reresize_transform(adv_img) * 255

            # qqq
            # DEBUG
            # pdb.set_trace()

        im = im[:num_samples]
        targets = targets[:num_samples]

        # continue

        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)

        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        # pred_for_plotting = []
        predictions_for_plotting = output_to_target(out)

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()

            # detections (Array[N, 6]), x1, y1, x2, y2, confidence, class
            # labels (Array[M, 5]), class, x1, y1, x2, y2

            # import pdb
            # pdb.set_trace()

            tbox = xywh2xyxy(labels[:, 1:5]).cpu()  # target boxes
            class_only = np.expand_dims(labels[:, 0].cpu(), axis=0)
            tbox = np.concatenate((class_only.T, tbox), axis=1)

            num_labels_changed = 0

            if synthetic:
                for lbl in tbox:
                    if lbl[0] == SYNTHETIC_STOP_SIGN_CLASS:
                        for pi, prd in enumerate(predn):
                            # [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
                            x1 = 0.9 * lbl[1] if 0.9 * lbl[1] > 5 else -20
                            y1 = 0.9 * lbl[2] if 0.9 * lbl[2] > 5 else -20

                            if prd[0] >= x1 and prd[1] >= y1 and prd[2] <= 1.1 * lbl[3] and prd[3] <= 1.1 * lbl[4]:
                                # 14 is octagon
                                if prd[5] == 14:
                                    predn[pi, 5] = SYNTHETIC_STOP_SIGN_CLASS
                                    pred[pi, 5] = SYNTHETIC_STOP_SIGN_CLASS

                                    # if prd[4] > 0.25:
                                    num_labels_changed += 1

                if num_labels_changed > 1:
                    num_errors += 1
                if num_labels_changed == 1:
                    num_detected += 1

            # for *box, conf, cls in predn.cpu().numpy():
            #     pred_for_plotting.append([si, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])

            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # 14 is octagon
            if plot_octagons and 14 in labels[:, 0]:
                num_octagon_labels += sum([1 for x in list(labels[:, 0]) if x == 14])
                if len(shape_to_plot_data['octagon']) < 50:
                    fn = str(path).split('/')[-1]

                    # TODO: remove if condition. only used to debug wrong transforms
                    # if fn not in df_use_polygons['filename_y'].values:
                    #     continue
                    # print('path', str(path).split('/')[-1])
                    shape_to_plot_data['octagon'].append(
                        [im[si: si + 1],
                        targets[targets[:, 0] == si, :],
                        path, predictions_for_plotting[predictions_for_plotting[:, 0] == si]])

        # Plot images
        if plots and batch_i < 30:
            if plot_single_images:
                save_dir_single_plots = increment_path(
                    save_dir / 'single_plots', exist_ok=exist_ok, mkdir=True)  # increment run
                for i in range(len(im)):
                    # labels
                    f = save_dir_single_plots / f'val_batch{batch_i}_image{i}_labels.jpg'  # labels
                    ti = targets[targets[:, 0] == i]
                    ti[:, 0] = 0
                    plot_images(im[i:i+1], ti, paths[i:i+1], f, names)

                    # predictions
                    f = save_dir_single_plots / f'val_batch{batch_i}_image{i}_pred.jpg'  # labels
                    ti = predictions_for_plotting[predictions_for_plotting[:, 0] == i]
                    ti[:, 0] = 0
                    plot_images(im[i:i+1], ti, paths[i:i+1], f, names)
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()
            print(f)

    if plot_octagons:
        save_dir_octagon = increment_path(save_dir / 'octagon', exist_ok=exist_ok, mkdir=True)  # increment run
        for i in range(len(shape_to_plot_data['octagon'])):
            im, targets, path, out = shape_to_plot_data['octagon'][i]
            # labels
            f = save_dir_octagon / f'image{i}_labels.jpg'  # labels
            targets[:, 0] = 0
            plot_images(im, targets, [path], f, names)

            # predictions
            f = save_dir_octagon / f'image{i}_pred.jpg'  # labels
            out[:, 0] = 0
            plot_images(im, out, [path], f, names)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    metrics_df_column_names = ["name", "apply_patch", "random_patch"]
    current_exp_metrics = {}

    metrics_df_column_names.append('num_bg')
    current_exp_metrics['num_bg'] = num_bg

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, fnr, fn = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    metrics_df_column_names.append('num_images')
    current_exp_metrics['num_images'] = seen
    metrics_df_column_names.append('num_targets_all')
    current_exp_metrics['num_targets_all'] = nt.sum()
    metrics_df_column_names.append('precision_all')
    current_exp_metrics['precision_all'] = mp
    metrics_df_column_names.append('recall_all')
    current_exp_metrics['recall_all'] = mr
    metrics_df_column_names.append('map_50_all')
    current_exp_metrics['map_50_all'] = map50
    metrics_df_column_names.append('map_50_95_all')
    current_exp_metrics['map_50_95_all'] = map
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    print('[INFO] results per class')
    for i, c in enumerate(ap_class):
        # metrics_df_column_names.append(f'num_images_{names[c]}')
        # current_exp_metrics[f'num_images_{names[c]}'] = seen
        metrics_df_column_names.append(f'num_targets_{names[c]}')
        current_exp_metrics[f'num_targets_{names[c]}'] = nt[c]
        metrics_df_column_names.append(f'precision_{names[c]}')
        current_exp_metrics[f'precision_{names[c]}'] = p[i]
        metrics_df_column_names.append(f'recall_{names[c]}')
        current_exp_metrics[f'recall_{names[c]}'] = r[i]
        metrics_df_column_names.append(f'fnr_{names[c]}')
        current_exp_metrics[f'fnr_{names[c]}'] = fnr[i]
        metrics_df_column_names.append(f'tp_{names[c]}')
        current_exp_metrics[f'tp_{names[c]}'] = tp[i]
        metrics_df_column_names.append(f'fn_{names[c]}')
        current_exp_metrics[f'fn_{names[c]}'] = fn[i]
        metrics_df_column_names.append(f'ap_50_{names[c]}')
        current_exp_metrics[f'ap_50_{names[c]}'] = ap50[i]
        metrics_df_column_names.append(f'ap_50_95_{names[c]}')
        current_exp_metrics[f'ap_50_95_{names[c]}'] = ap[i]

        LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    print('num octagon labels', num_octagon_labels)
    print('num octagon with patch', num_octagon_with_patch)

    metrics_df_column_names.append('num_errors')
    current_exp_metrics['num_errors'] = num_errors
    metrics_df_column_names.append('proportion_of_errors')
    current_exp_metrics['proportion_of_errors'] = num_errors/seen
    metrics_df_column_names.append('dataset')
    current_exp_metrics['dataset'] = DATASET_NAME

    metrics_df_column_names.append('num_octagon_labels')
    current_exp_metrics['num_octagon_labels'] = num_octagon_labels
    metrics_df_column_names.append('num_octagon_with_patch')
    current_exp_metrics['num_octagon_with_patch'] = num_octagon_with_patch
    
    if save_exp_metrics:
        try:
            metrics_df = pd.read_csv('runs/results.csv')
        except FileNotFoundError:
            metrics_df = pd.DataFrame(columns=metrics_df_column_names)
            metrics_df = pd.DataFrame()

        current_exp_metrics['name'] = name
        current_exp_metrics['apply_patch'] = apply_patch
        current_exp_metrics['random_patch'] = random_patch

        metrics_df = metrics_df.append(current_exp_metrics, ignore_index=True)
        metrics_df.to_csv('runs/results.csv', index=False)

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map

    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # Our attack and evaluate options
    parser.add_argument('--apply-patch', action='store_true', help='add adversarial patch to traffic signs if true')
    parser.add_argument('--synthetic', action='store_true', help='add adversarial patch to traffic signs if true')
    parser.add_argument('--random-patch', action='store_true', help='adversarial patch is random')
    parser.add_argument('--load-patch', type=str, default='', help='path to adv patch to load')
    parser.add_argument('--save-exp-metrics', action='store_true', help='save metrics for this experiment to dataframe')
    parser.add_argument('--plot-single-images', action='store_true',
                        help='save single images in a folder instead of batch images in a single plot')
    parser.add_argument('--plot-octagons', action='store_true',
                        help='save single images containing octagons in a folder')
    parser.add_argument('--num-bg', type=int, default=16, help='number of backgrounds to generate adversarial patch')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(opt, **vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(opt, **vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(opt, **vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
