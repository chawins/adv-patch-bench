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
import warnings
from ast import literal_eval
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as T
import yaml
from kornia import augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import resize
from tqdm import tqdm

from adv_patch_bench.attacks.rp2 import RP2AttackModule
from adv_patch_bench.transforms import transform_and_apply_patch
from adv_patch_bench.utils.image import (mask_to_box, pad_and_center,
                                         prepare_obj)
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

warnings.filterwarnings("ignore")


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
    iou = box_iou(labels[:, 1:5], detections[:, :4])

    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match

    matches = []
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            # sort matches by decreasing order of IOU
            matches = matches[matches[:, 2].argsort()[::-1]]
            # for each (label, detection) pair, select the one with highest IOU score
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct, matches


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

    # Load our new args
    attack_type = args.attack_type
    use_attack = args.attack_type != 'none'
    synthetic_eval = args.synthetic_eval
    save_exp_metrics = args.save_exp_metrics
    plot_single_images = args.plot_single_images
    plot_class_examples = [int(x) for x in args.plot_class_examples]
    adv_patch_path = args.adv_patch_path
    tgt_csv_filepath = args.tgt_csv_filepath
    attack_config_path = args.attack_config_path
    adv_sign_class = args.obj_class
    syn_obj_path = args.syn_obj_path

    no_transform = args.no_transform
    relighting = not args.no_relighting
    metrics_confidence_threshold = args.metrics_confidence_threshold
    img_size = tuple([int(x) for x in args.padded_imgsz.split(',')])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DATASET_NAME = 'mapillary' if 'mapillary' in data else 'mtsd'
    try:
        metrics_df = pd.read_csv('runs/results.csv')
        metrics_df = metrics_df.replace({'apply_patch': 'True', 'random_patch': 'True'}, 1)
        metrics_df = metrics_df.replace({'apply_patch': 'False', 'random_patch': 'False'}, 0)
        metrics_df['apply_patch'] = metrics_df['apply_patch'].astype(float).astype(bool)
        metrics_df['random_patch'] = metrics_df['random_patch'].astype(float).astype(bool)
        metrics_df_grouped = metrics_df.groupby(by=['dataset', 'apply_patch', 'random_patch']).count().reset_index()
        metrics_df_grouped = metrics_df_grouped[
            (metrics_df_grouped['dataset'] == DATASET_NAME) &
            (metrics_df_grouped['apply_patch'] == use_attack) &
            (metrics_df_grouped['random_patch'] == (attack_type == 'random'))]['name']
        exp_number = 0 if not len(metrics_df_grouped) else metrics_df_grouped.item()
    except FileNotFoundError:
        exp_number = 0

    # Set folder and experiment name
    name += f'_{DATASET_NAME}_{attack_type}_{exp_number}'
    print(f'=> Experiment name: {name}')

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
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride,
                                       single_cls, pad=pad, rect=pt,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    nc = len(names)
    assert adv_sign_class < nc, 'Obj Class to attack does not exist'
    print('Class names: ', names)

    confusion_matrix = ConfusionMatrix(nc=nc)

    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    # DEBUG
    # if use_attack:
    #     demo_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
    #     # demo_patch = resize(adv_patch_cropped, (32, 32))
    #     demo_patch = resize(demo_patch, (32, 32))
    #     f = os.path.join(save_dir, 'adversarial_patch.png')
    #     torchvision.utils.save_image(demo_patch, f)

    if synthetic_eval:
        # Testing with synthetic signs
        syn_sign_class = len(names)
        names[syn_sign_class] = 'synthetic'
        nc += 1
        # Set up random transforms for synthetic sign
        img_height, img_width = img_size
        obj_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, return_transform=True)
        mask_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, resample=Resample.NEAREST)

        # Load synthetic object/sign from file
        adv_patch, patch_mask = pickle.load(open(adv_patch_path, 'rb'))
        patch_mask = patch_mask.to(device)
        obj_size = patch_mask.shape[1]
        obj, obj_mask = prepare_obj(syn_obj_path, img_size, (obj_size, obj_size))
        obj = obj.to(device)
        obj_mask = obj_mask.to(device).unsqueeze(0)

    if use_attack:
        # Load patch from a pickle file if specified
        # TODO: make script to generate dummy patch
        adv_patch, patch_mask = pickle.load(open(adv_patch_path, 'rb'))
        patch_mask = patch_mask.to(device)
        patch_height, patch_width = adv_patch.shape[1:]
        patch_loc = mask_to_box(patch_mask)

        if attack_type == 'debug':
            # Load 'arrow on checkboard' patch if specified (for debug)
            adv_patch = torchvision.io.read_image('demo.png').float()[:3, :, :] / 255
            adv_patch = resize(adv_patch, (patch_height, patch_width))
        elif attack_type == 'random':
            # Patch with uniformly random pixels
            adv_patch = torch.rand(3, patch_height, patch_width)

        if synthetic_eval:
            # Adv patch and mask have to be made compatible with random
            # transformation for synthetic signs
            _, patch_mask = pad_and_center(None, patch_mask, img_size, (obj_size, obj_size))
            patch_loc = mask_to_box(patch_mask)
            # Pad adv patch
            pad_size = [
                patch_loc[1],  # left
                patch_loc[0],  # right
                img_size[1] - patch_loc[1] - patch_loc[3],  # top
                img_size[0] - patch_loc[0] - patch_loc[2],  # bottom
            ]
            adv_patch = T.pad(adv_patch, pad_size)
            patch_mask = patch_mask.to(device)
            adv_patch = adv_patch.to(device)
            obj = obj.to(device)
        else:
            # load csv file containing target points for transform
            df = pd.read_csv(tgt_csv_filepath)
            # converts 'tgt_final' from string to list format
            df['tgt_final'] = df['tgt_final'].apply(literal_eval)
            # exclude shapes to which we do not apply the transform to
            df = df[df['final_shape'] != 'other-0.0-0.0']
            print(df.shape)
            print(df.groupby(by=['final_shape']).count())

    # Initialize attack
    if attack_type == 'per-sign':
        try:
            with open(attack_config_path) as file:
                attack_config = yaml.load(file, Loader=yaml.FullLoader)
                attack_config['input_size'] = img_size

            attack = RP2AttackModule(attack_config, model, None, None, None,
                                     rescaling=False, interp=args.interp, verbose=verbose)
        except:
            raise Exception('Config file not provided for targeted attacks')

    # ======================================================================= #
    #                          BEGIN: Main eval loop                          #
    # ======================================================================= #
    total_num_patches = 0
    num_apply_imgs = 0

    # Loading file names from the specified text file
    filename_list = []
    if args.img_txt_path != '':
        with open(args.img_txt_path, 'r') as f:
            filename_list = f.read().splitlines()

    if plot_class_examples:
        shape_to_plot_data = {}
        for class_index in plot_class_examples:
            class_name = names[class_index]
            shape_to_plot_data[class_name] = []

    # TODO: remove 'metrics_per_image_df' in the future
    metrics_per_image_df = pd.DataFrame(columns=['filename', 'num_targeted_sign_class', 'num_patches', 'fn'])
    metrics_per_label_df = pd.DataFrame(
        columns=['filename', 'obj_id', 'label', 'correct_prediction', 'sign_width', 'sign_height', 'confidence'])

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 'targets' shape is # number of labels by 8 (image_id, class, x1, y1, label width, label height, number of patches applied, obj id)
        targets = torch.nn.functional.pad(targets, (0, 2), "constant", 0)  # effectively zero padding

        # DEBUG
        if args.debug and batch_i == 20:
            break

        if num_apply_imgs >= len(filename_list) and args.run_only_img_txt:
            break
        # ======================= BEGIN: apply patch ======================== #
        for image_i, path in enumerate(paths):
            # assign each label in the image an object id
            targets[targets[:, 0] == image_i, 7] = torch.arange(
                torch.sum(targets[:, 0] == image_i), dtype=targets.dtype)

            if use_attack and not synthetic_eval:
                filename = path.split('/')[-1]
                img_df = df[df['filename_y'] == filename]

                if len(img_df) == 0:
                    continue

                # Skip (or only run on) files listed in the txt file
                in_list = filename in filename_list
                if ((in_list and not args.run_only_img_txt) or
                        (not in_list and args.run_only_img_txt)):
                    continue
                num_apply_imgs += 1

                num_patches_applied_to_image = 0

                # Apply patch on all of the signs on this image
                for _, row in img_df.iterrows():
                    (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
                    img_data = (h0, w0, h_ratio, w_ratio, w_pad, h_pad)
                    predicted_class = row['final_shape']

                    shape = predicted_class.split('-')[0]
                    if shape != names[adv_sign_class]:
                        continue
                    total_num_patches += 1

                    # Run attack for each sign
                    if attack_type == 'per-sign':
                        print('=> Generating adv patch...')
                        data = [predicted_class, row, *img_data]
                        attack_images = [[im[image_i], data, str(filename)]]
                        with torch.enable_grad():
                            adv_patch = attack.transform_and_attack(
                                attack_images, patch_mask=patch_mask,
                                obj_class=adv_sign_class)[0]

                    # # Transform and apply patch on the image. `im` has range [0, 255]
                    im[image_i] = transform_and_apply_patch(
                        im[image_i].to(device), adv_patch.to(device), patch_mask, patch_loc,
                        predicted_class, row, img_data, no_transform=no_transform, relighting=relighting,
                        interp=args.interp) * 255
                    num_patches_applied_to_image += 1

                # set targets[6] to #patches_applied_to_image
                targets[targets[:, 0] == image_i, 6] = num_patches_applied_to_image

            elif synthetic_eval:
                if use_attack:
                    adv_obj = patch_mask * adv_patch + (1 - patch_mask) * obj
                else:
                    adv_obj = obj

                adv_obj, tf_params = obj_transforms(adv_obj)
                adv_obj.clamp_(0, 1)
                o_mask = mask_transforms.apply_transform(
                    obj_mask, None, transform=tf_params.to(device))

                # get top left and bottom right points
                # TODO: can we use mask_to_box here?
                indices = np.where(o_mask.cpu()[0][0] == 1)
                x_min, x_max = min(indices[1]), max(indices[1])
                y_min, y_max = min(indices[0]), max(indices[0])

                # Since we paste a new synthetic sign on image, we have to add
                # in a new synthetic label/target to compute the metrics
                label = [
                    image_i,
                    syn_sign_class,
                    (x_min + x_max) / (2 * img_width),
                    (y_min + y_max) / (2 * img_height),
                    (x_max - x_min) / img_width,
                    (y_max - y_min) / img_height,
                    int(use_attack),
                    -1
                ]
                targets = torch.cat((targets, torch.tensor(label).unsqueeze(0)))

                adv_img = o_mask * adv_obj + (1 - o_mask) * im[image_i].to(device) / 255
                im[image_i] = adv_img.squeeze() * 255

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
        targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        predictions_for_plotting = output_to_target(out)

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]

            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            filename = str(path).split('/')[-1]

            current_image_metrics = {}
            current_image_metrics['filename'] = filename
            current_image_metrics['num_targeted_sign_class'] = sum([1 for x in labels[:, 0] if x == adv_sign_class])
            current_image_metrics['num_patches'] = max(labels[:, 5].tolist() + [0])

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))

                for lbl_ in labels:
                    if lbl_[0] == adv_sign_class:
                        lbl_ = lbl_.cpu()
                        current_label_metric = {}
                        current_label_metric['filename'] = filename
                        current_label_metric['obj_id'] = lbl_[6].item()
                        current_label_metric['num_patches_applied_to_image'] = lbl_[5].item()
                        current_label_metric['label'] = lbl_[0].item()
                        current_label_metric['correct_prediction'] = 0
                        current_label_metric['sign_width'] = lbl_[3].item()
                        current_label_metric['sign_height'] = lbl_[4].item()
                        current_label_metric['confidence'] = None
                        metrics_per_label_df = metrics_per_label_df.append(current_label_metric, ignore_index=True)

                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()

            # detections (Array[N, 6]), x1, y1, x2, y2, confidence, class
            # labels (Array[M, 5]), class, x1, y1, x2, y2

            tbox = xywh2xyxy(labels[:, 1:5]).cpu()  # target boxes
            class_only = np.expand_dims(labels[:, 0].cpu(), axis=0)
            tbox = np.concatenate((class_only.T, tbox), axis=1)

            num_labels_changed = 0

            if synthetic_eval:
                for lbl in tbox:
                    if lbl[0] == syn_sign_class:
                        for pi, prd in enumerate(predn):
                            # [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
                            x1 = 0.9 * lbl[1] if 0.9 * lbl[1] > 5 else -20
                            y1 = 0.9 * lbl[2] if 0.9 * lbl[2] > 5 else -20

                            if prd[0] >= x1 and prd[1] >= y1 and prd[2] <= 1.1 * lbl[3] and prd[3] <= 1.1 * lbl[4]:
                                if prd[5] == adv_sign_class:
                                    predn[pi, 5] = syn_sign_class
                                    pred[pi, 5] = syn_sign_class

                                    # if prd[4] > 0.25:
                                    num_labels_changed += 1

            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                correct, matches = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct, matches = torch.zeros(pred.shape[0], niou, dtype=torch.bool), []

            # Collecting results
            for lbl_ in labels:
                if lbl_[0] == adv_sign_class:
                    lbl_ = lbl_.cpu()
                    current_label_metric = {}
                    current_label_metric['filename'] = filename
                    current_label_metric['obj_id'] = lbl_[6].item()
                    current_label_metric['num_patches_applied_to_image'] = lbl_[5].item()
                    current_label_metric['label'] = lbl_[0].item()
                    current_label_metric['correct_prediction'] = 0
                    current_label_metric['sign_width'] = lbl_[3].item()
                    current_label_metric['sign_height'] = lbl_[4].item()
                    current_label_metric['confidence'] = None

                    if len(matches) > 0:
                        # match on obj_id
                        match = matches[matches[:, 0] == lbl_[6]]
                        assert len(match) <= 1
                        if len(match) > 0:
                            detection_index = int(match[0, 1])
                            current_label_metric['confidence'] = pred.cpu().numpy()[detection_index, 4]
                            if pred.cpu().numpy()[detection_index, 5] == adv_sign_class and pred.cpu().numpy()[
                                    detection_index, 4] > metrics_confidence_threshold and correct[detection_index, 0]:
                                current_label_metric['correct_prediction'] = 1
                    metrics_per_label_df = metrics_per_label_df.append(current_label_metric, ignore_index=True)

            total_positives = sum([1 for x in tcls if x == adv_sign_class])
            sign_indices = np.logical_and(
                pred.cpu().numpy()[:, 5] == adv_sign_class, pred.cpu().numpy()[:, 4] >
                metrics_confidence_threshold)
            tp = sum(correct.cpu().numpy()[sign_indices, 0])
            current_image_metrics['fn'] = total_positives - tp
            metrics_per_image_df = metrics_per_image_df.append(current_image_metrics, ignore_index=True)

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            for class_index in labels[:, 0]:
                if class_index in plot_class_examples:
                    class_name = names[class_index.item()]
                    if len(shape_to_plot_data[class_name]) < 50:
                        fn = str(path).split('/')[-1]
                        shape_to_plot_data[class_name].append(
                            [im[si: si + 1], targets[targets[:, 0] == si, :], path,
                             predictions_for_plotting[predictions_for_plotting[:, 0] == si]])
                        break

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
                    plot_images(im[i:i+1], ti, paths[i:i+1], f, names, labels=True)

                    # predictions
                    f = save_dir_single_plots / f'val_batch{batch_i}_image{i}_pred.jpg'  # labels
                    ti = predictions_for_plotting[predictions_for_plotting[:, 0] == i]
                    ti[:, 0] = 0
                    plot_images(im[i:i+1], ti, paths[i:i+1], f, names, labels=False)
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names, 1920, 16, True), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names, 1920, 16, False), daemon=True).start()
            print(f)
    # ======================================================================= #
    #                            END: Main eval loop                          #
    # ======================================================================= #

    metrics_per_image_df.to_csv(f'{project}/{name}/results_per_image.csv', index=False)
    metrics_per_label_df.to_csv(f'{project}/{name}/results_per_label.csv', index=False)

    if plot_class_examples:
        for class_index in plot_class_examples:
            class_name = names[class_index]
            save_dir_class = increment_path(save_dir / class_name, exist_ok=exist_ok, mkdir=True)  # increment run
            for i in range(len(shape_to_plot_data[class_name])):
                im, targets, path, out = shape_to_plot_data[class_name][i]
                # labels
                f = save_dir_class / f'image{i}_labels.jpg'  # labels
                targets[:, 0] = 0
                plot_images(im, targets, [path], f, names, labels=True)

                # predictions
                f = save_dir_class / f'image{i}_pred.jpg'  # labels
                out[:, 0] = 0
                plot_images(im, out, [path], f, names, labels=False)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    metrics_df_column_names = ["name", "apply_patch", "random_patch"]
    current_exp_metrics = {}

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class, fnr, fn = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names, confidence_threshold=metrics_confidence_threshold)
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
    metrics_df_column_names.append('dataset')
    current_exp_metrics['dataset'] = DATASET_NAME

    metrics_df_column_names.append('total_num_patches')
    current_exp_metrics['total_num_patches'] = total_num_patches

    if save_exp_metrics:
        try:
            metrics_df = pd.read_csv('runs/results.csv')
        except FileNotFoundError:
            metrics_df = pd.DataFrame(columns=metrics_df_column_names)
            metrics_df = pd.DataFrame()

        current_exp_metrics['name'] = name
        current_exp_metrics['apply_patch'] = use_attack
        current_exp_metrics['random_patch'] = attack_type == 'random'

        if attack_type == 'per-sign':
            metrics_df_column_names.append('no_transform')
            current_exp_metrics['no_transform'] = attack_config['no_transform']
            metrics_df_column_names.append('relighting')
            current_exp_metrics['relighting'] = attack_config['relighting']

        try:
            metrics_df_column_names.append('generate_patch')
            metrics_df_column_names.append('rescaling')
            metrics_df_column_names.append('relighting')
            if adv_patch_path:
                patch_metadata_folder = '/'.join(adv_patch_path.split('/')[:-1])
                patch_metadata_path = os.path.join(patch_metadata_folder, 'patch_metadata.pkl')
                print(patch_metadata_path)
                with open(patch_metadata_path, 'rb') as f:
                    patch_metadata = pickle.load(f)
                current_exp_metrics['generate_patch'] = patch_metadata['generate_patch']
                current_exp_metrics['rescaling'] = patch_metadata['rescaling']
                current_exp_metrics['relighting'] = patch_metadata['relighting']
        except:
            pass

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
    # =========================== Model arguments =========================== #
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
    # ========================== Our misc arguments ========================= #
    parser.add_argument('--seed', type=int, default=0, help='set random seed')
    parser.add_argument('--padded_imgsz', type=str, default='992,1312',
                        help='final image size including padding (height,width). Default: 992,1312')
    parser.add_argument('--interp', type=str, default='bilinear',
                        help='interpolation method (nearest, bilinear, bicubic)')
    parser.add_argument('--synthetic-eval', action='store_true',
                        help='evaluate with pasted synthetic signs')
    parser.add_argument('--debug', action='store_true')
    # =========================== Attack arguments ========================== #
    # General
    parser.add_argument('--attack-type', type=str, required=True,
                        help='which attack evaluation to run (none, load, per-sign, random, debug)')
    parser.add_argument('--adv-patch-path', type=str, default='',
                        help='path to adv patch and mask to load')
    parser.add_argument('--obj-class', type=int, default=0, help='class of object to attack')
    parser.add_argument('--tgt-csv-filepath', required=True,
                        help='path to csv which contains target points for transform')
    parser.add_argument('--attack-config-path',
                        help='path to yaml file with attack configs (used when attack_type is per-sign)')
    parser.add_argument('--syn-obj-path', type=str, default='',
                        help='path to an image of a synthetic sign (used when synthetic_eval is True')
    parser.add_argument('--img-txt-path', type=str, default='',
                        help='path to a text file containing image filenames')
    parser.add_argument('--run-only-img-txt', action='store_true',
                        help='run evaluation on images listed in img-txt-path. Otherwise, exclude these images.')
    parser.add_argument('--no-transform', action='store_true',
                        help=('If True, do not apply patch to signs using '
                              '3D-transform. Patch will directly face camera.'))
    parser.add_argument('--no-relighting', action='store_true',
                        help=('If True, do not apply relighting transform to patch'))

    # ============================== Plot / log ============================= #
    parser.add_argument('--save-exp-metrics', action='store_true', help='save metrics for this experiment to dataframe')
    parser.add_argument('--plot-single-images', action='store_true',
                        help='save single images in a folder instead of batch images in a single plot')
    parser.add_argument('--plot-class-examples', type=str, default='', nargs='*',
                        help='save single images containing individual classes in different folders.')
    parser.add_argument('--metrics-confidence-threshold', type=float, default=0.5, help='confidence threshold')

    # TODO: remove when no bug
    # parser.add_argument('--num-bg', type=int, default=16, help='number of backgrounds to generate adversarial patch')
    # parser.add_argument('--patch-loc', type=str, default='', nargs='*',
    #                     help='location to place patch w.r.t. object in tuple (ymin, xmin)')
    # parser.add_argument('--obj-size', type=int, default=128, help='width of the object in pixels')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)

    if opt.synthetic_eval and opt.attack_type == 'per-sign':
        raise NotImplementedError('Synthetic evaluation with per-sign attack is not implemented.')

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
