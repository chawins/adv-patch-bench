"""
Generate adversarial patch
"""

import os
import pickle
import random
from ast import literal_eval
from os.path import join
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms.functional as T
import yaml
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from PIL import Image
from torch.nn import DataParallel
from tqdm import tqdm

from adv_patch_bench.attacks.rp2 import RP2AttackModule
from adv_patch_bench.attacks.utils import get_object_and_mask_from_numpy
from adv_patch_bench.dataloaders import (get_mapillary_dict, get_mtsd_dict,
                                         register_mapillary, register_mtsd)
from adv_patch_bench.dataloaders.detectron.mapper import BenignMapper
from adv_patch_bench.utils.argparse import (eval_args_parser,
                                            setup_detectron_test_args)
from adv_patch_bench.utils.detectron import ShuffleInferenceSampler
from adv_patch_bench.utils.image import get_obj_width
from gen_mask import generate_mask
from hparams import (DATASETS, LABEL_LIST, MAPILLARY_IMG_COUNTS_DICT,
                     OTHER_SIGN_CLASS, SAVE_DIR_DETECTRON)


def collect_backgrounds(dataloader, img_size, num_bg, device,
                        df=None, class_name=None):
    if num_bg < 1:
        assert class_name is not None
        print(f'num_bg is a fraction ({num_bg}).')
        num_bg = round(MAPILLARY_IMG_COUNTS_DICT[class_name] * num_bg)
        print(f'For {class_name}, this is {num_bg} images.')
    num_bg = int(num_bg)

    attack_images, metadata = [], []
    backgrounds = torch.zeros((num_bg, 3) + img_size, )
    num_collected = 0
    print('=> Collecting background images...')

    # DEBUG: count images
    # counts = [0] * 12
    # class_names = LABEL_LIST[args.dataset]

    for i, batch in tqdm(enumerate(dataloader)):
        file_name = batch[0]['file_name']
        filename = file_name.split('/')[-1]

        if df is not None:
            img_df = df[df['filename'] == filename]
            if len(img_df) == 0:
                continue

        found = False
        obj, obj_label = None, class_name
        if class_name is not None and df is not None:
            for _, obj in img_df.iterrows():
                obj_label = obj['final_shape']
                if obj_label == class_name:
                    found = True
                    break
            # DEBUG
            # obj_labels = np.unique(img_df['final_shape'])
            # for obj_label in obj_labels:
            #     lb = class_names.index(obj_label)
            #     counts[lb] += 1
        else:
            # No df provided or don't care about class
            found = True

        if found:
            # Flip BGR to RGB and then flip back when feeding to model
            image = batch[0]['image'].float().to(device).flip(0)
            h0, w0 = batch[0]['height'], batch[0]['width']
            _, h, w = image.shape

            assert w == img_size[1]
            if h > img_size[0]:
                # Just resize in this case and avoid padding
                image = T.resize(image, img_size, antialias=True)
                pad_top = 0
            else:
                # Pad height
                pad_top = (img_size[0] - h) // 2
                pad_bottom = img_size[0] - h - pad_top
                image = T.pad(image, [0, pad_top, 0, pad_bottom])

            _, h, w = image.shape
            assert (h, w) == img_size
            # NOTE: img_data: h_orig, w_orig, h, w, w_pad, h_pad
            # It's (w_pad, h_pad) and not (h_pad, w_pad) due to compatibility
            # with YOLO dataloader/augmentation
            img_data = (h0, w0, h / h0, w / w0, 0, pad_top)
            data = [obj_label, obj, *img_data]
            attack_images.append([image, data, str(filename), batch[0]])
            metadata.extend(batch)
            if num_collected < num_bg:
                backgrounds[num_collected] = image
            num_collected += 1

        # num_collected += 1
        if num_collected >= num_bg:
            break

    # print('======> ', counts, num_collected)

    print(f'=> {len(attack_images)} backgrounds collected.')
    return attack_images[:num_bg], metadata[:num_bg], backgrounds / 255


def generate_adv_patch(
    model: torch.nn.Module,
    obj_numpy: np.ndarray,
    patch_mask: torch.Tensor,
    device: str = 'cuda',
    img_size: Tuple[int, int] = (992, 1312),
    obj_class: int = None,
    obj_size: int = None,
    # bg_dir: str = './',
    # num_bg: int = 16,
    save_images: bool = False,
    save_dir: str = './',
    synthetic: bool = False,
    # rescaling: bool = False,
    tgt_csv_filepath: str = None,
    dataloader: Any = None,
    interp: str = 'bilinear',
    verbose: bool = False,
    debug: bool = False,
    dataset: str = None,
    attack_config: Dict = {},
    **kwargs,
):
    """Generate adversarial patch"""
    print(f'=> Initializing attack...')
    num_bg = attack_config['num_bg']

    # TODO: Allow data parallel?
    attack = RP2AttackModule(attack_config, model, None, None, None,
                             interp=interp, verbose=verbose, is_detectron=True)

    # Randomly select backgrounds from `bg_dir` and resize them
    # TODO: Remoev in the future
    # all_bgs = os.listdir(os.path.expanduser(bg_dir))
    # print(f'There are {len(all_bgs)} background images in {bg_dir}.')
    # idx = np.arange(len(all_bgs))
    # np.random.shuffle(idx)
    # bg_size = img_size
    # backgrounds = torch.zeros((num_bg, 3) + bg_size, )
    # for i, index in enumerate(idx[:num_bg]):
    #     bg = torchvision.io.read_image(join(bg_dir, all_bgs[index])) / 255
    #     backgrounds[i] = T.resize(bg, bg_size, antialias=True)

    df = None
    class_name = LABEL_LIST[dataset][obj_class]
    if not synthetic:
        # For synthetic sign, we don't care about transforms and classes
        df = pd.read_csv(tgt_csv_filepath)
        df['tgt_final'] = df['tgt_final'].apply(literal_eval)
        df = df[df['final_shape'] != 'other-0.0-0.0']

    attack_images, metadata, backgrounds = collect_backgrounds(
        dataloader, img_size, num_bg, device, df=df, class_name=class_name)

    # Generate an adversarial patch
    if synthetic:
        print('=> Generating adversarial patch on synthetic signs...')
        # Resize object to the specify size and pad obj and masks to image size
        # left, top, right, bottom
        pad_size = [(img_size[1] - obj_size[1]) // 2,
                    (img_size[0] - obj_size[0]) // 2,
                    (img_size[1] - obj_size[1]) // 2 + obj_size[1] % 2,
                    (img_size[0] - obj_size[0]) // 2 + obj_size[0] % 2]  
        obj, obj_mask = get_object_and_mask_from_numpy(obj_numpy, obj_size, 
                                                       pad_size=pad_size)
        patch_mask_ = T.resize(patch_mask, obj_size, 
                               interpolation=T.InterpolationMode.NEAREST)
        patch_mask_ = T.pad(patch_mask_, pad_size)

        print(f'=> Start attacking...')
        adv_patch = attack.attack(obj.to(device),
                                  obj_mask.to(device),
                                  patch_mask_.to(device),
                                  backgrounds.to(device),
                                  obj_class=obj_class,
                                  metadata=metadata)

        if save_images:
            torchvision.utils.save_image(obj, join(save_dir, 'obj.png'))
            torchvision.utils.save_image(obj_mask, 
                                         join(save_dir, 'obj_mask.png'))
            torchvision.utils.save_image(backgrounds, 
                                         join(save_dir, 'backgrounds.png'))

    else:
        print('=> Generating adversarial patch on real signs...')

        # DEBUG: Save all the background images
        if debug:
            for img in attack_images:
                os.makedirs(join(save_dir, 'backgrounds'), exist_ok=True)
                torchvision.utils.save_image(
                    img[0] / 255, join(save_dir, 'backgrounds', img[2]))

        # Save background filenames in txt file
        print(f'=> Saving used backgrounds in a txt file.')
        with open(join(save_dir, 'bg_filenames.txt'), 'w') as f:
            for img in attack_images:
                f.write(f'{img[2]}\n')

        print(f'=> Start attacking...')
        # with torch.enable_grad():
        adv_patch = attack.attack_real(attack_images,
                                       patch_mask=patch_mask.to(device),
                                       obj_class=obj_class,
                                       metadata=metadata)

    adv_patch = adv_patch[0].detach().cpu().float()

    if save_images:
        torchvision.utils.save_image(patch_mask, 
                                     join(save_dir, 'patch_mask.png'))
        torchvision.utils.save_image(adv_patch, 
                                     join(save_dir, 'adversarial_patch.png'))

    return adv_patch


def main(
    padded_imgsz: str = '992,1312',
    save_dir=Path(''),
    name: str = 'exp',  # save to project/name
    obj_class: int = None,
    obj_size: int = None,
    syn_obj_path: str = '',
    seed: int = 0,
    synthetic: bool = False,
    attack_config_path: str = None,
    num_samples: int = 0,
    **kwargs,
):
    cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    img_size = tuple([int(i) for i in padded_imgsz.split(',')])
    assert len(img_size) == 2

    # Set up directories
    class_names = LABEL_LIST[args.dataset]
    save_dir = os.path.join(SAVE_DIR_DETECTRON, name, class_names[obj_class])
    os.makedirs(save_dir, exist_ok=True)
    model = DefaultPredictor(cfg).model

    # Configure object size
    # NOTE: We assume that the target object fits the tensor in the same way
    # that we generate canonical masks (e.g., largest inscribed circle, octagon,
    # etc. in a square). Patch and patch mask are defined with respect to this
    # object tensor, and they should all have the same width and height.
    obj_numpy = np.array(Image.open(syn_obj_path).convert('RGBA')) / 255
    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]

    # Deterimine object size in pixels
    if obj_size is None:
        obj_size = int(min(img_size) * 0.1)
    if isinstance(obj_size, int):
        obj_size = (round(obj_size * h_w_ratio), obj_size)

    # Get object width in inch
    obj_width_inch = get_obj_width(obj_class, class_names)
    patch_mask = generate_mask(obj_numpy, obj_size, obj_width_inch)

    # Build dataloader
    dataloader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=BenignMapper(cfg, is_train=False),
        batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=ShuffleInferenceSampler(num_samples)
    )

    with open(attack_config_path) as file:
        attack_config = yaml.load(file, Loader=yaml.FullLoader)
    attack_config['input_size'] = img_size

    adv_patch = generate_adv_patch(
        model, obj_numpy, patch_mask, img_size=img_size, obj_size=obj_size,
        save_dir=save_dir, synthetic=synthetic, dataloader=dataloader,
        obj_class=obj_class, name=name, attack_config=attack_config, **kwargs)

    # Save adv patch
    patch_path = join(save_dir, 'adv_patch.pkl')
    print(f'Saving the generated adv patch to {patch_path}...')
    pickle.dump([adv_patch, patch_mask], open(patch_path, 'wb'))

    # Save attack config
    patch_metadata_path = join(save_dir, 'config.yaml')
    print(f'Saving the adv patch metadata to {patch_metadata_path}...')
    patch_metadata = {'synthetic': synthetic, **attack_config}
    with open(patch_metadata_path, 'w') as outfile:
        yaml.dump(patch_metadata, outfile)


if __name__ == "__main__":
    args = eval_args_parser(True)
    print('Command Line Args:', args)
    args.device = 'cuda'

    # Verify some args
    cfg = setup_detectron_test_args(args, OTHER_SIGN_CLASS)
    assert args.dataset in DATASETS
    split = cfg.DATASETS.TEST[0].split('_')[1]

    # Register dataset
    if 'mtsd' in args.dataset:
        assert 'mtsd' in cfg.DATASETS.TEST[0], \
            'MTSD is specified as dataset in args but not config file'
        dataset_params = register_mtsd(
            use_mtsd_original_labels='orig' in args.dataset,
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )
        data_list = get_mtsd_dict(split, *dataset_params)
    else:
        assert 'mapillary' in cfg.DATASETS.TEST[0], \
            'Mapillary is specified as dataset in args but not config file'
        dataset_params = register_mapillary(
            use_color=args.use_color,
            ignore_other=args.data_no_other,
            only_annotated=args.annotated_signs_only,
        )
        data_list = get_mapillary_dict(split, *dataset_params)
    num_samples = len(data_list)

    print(args)
    main(**vars(args), num_samples=num_samples)
