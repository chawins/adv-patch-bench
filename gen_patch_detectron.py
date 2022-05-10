"""
Generate adversarial patch
"""

import os
import pickle
from ast import literal_eval
from os.path import join
from pathlib import Path
from typing import Any, Tuple

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
from adv_patch_bench.dataloaders import register_mapillary, register_mtsd
from adv_patch_bench.dataloaders.detectron.mapper import BenignMapper
from adv_patch_bench.utils.argparse import (eval_args_parser,
                                            setup_detectron_test_args)
from adv_patch_bench.utils.image import get_obj_width
from gen_mask import generate_mask
from hparams import DATASETS, LABEL_LIST, OTHER_SIGN_CLASS, SAVE_DIR_DETECTRON


def generate_adv_patch(
    model: torch.nn.Module,
    obj_numpy: np.ndarray,
    patch_mask: torch.Tensor,
    device: str = 'cuda',
    img_size: Tuple[int, int] = (992, 1312),
    obj_class: int = 0,
    obj_size: int = None,
    bg_dir: str = './',
    num_bg: int = 16,
    save_images: bool = False,
    save_dir: str = './',
    synthetic: bool = False,
    # rescaling: bool = False,
    tgt_csv_filepath: str = None,
    dataloader: Any = None,
    attack_config_path: str = None,
    interp: str = 'bilinear',
    verbose: bool = False,
    debug: bool = False,
    **kwargs,
):
    """Generate adversarial patch

    Args:
        model (torch.nn.Module): _description_
        obj_numpy (np.ndarray): _description_
        patch_mask (torch.Tensor): _description_
        device (str, optional): _description_. Defaults to 'cuda'.
        img_size (Tuple[int, int], optional): _description_. Defaults to (992, 1312).
        obj_class (int, optional): _description_. Defaults to 0.
        obj_size (int, optional): _description_. Defaults to None.
        bg_dir (str, optional): _description_. Defaults to './'.
        num_bg (int, optional): _description_. Defaults to 16.
        save_images (bool, optional): _description_. Defaults to False.
        save_dir (str, optional): _description_. Defaults to './'.
        synthetic (bool, optional): _description_. Defaults to False.
        tgt_csv_filepath (str, optional): _description_. Defaults to None.
        dataloader (Any, optional): _description_. Defaults to None.
        attack_config_path (str, optional): _description_. Defaults to None.
        interp (str, optional): _description_. Defaults to 'bilinear'.
        verbose (bool, optional): _description_. Defaults to False.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    class_names = LABEL_LIST[args.dataset]
    print(f'=> Initializing attack...')
    with open(attack_config_path) as file:
        attack_config = yaml.load(file, Loader=yaml.FullLoader)
        attack_config['input_size'] = img_size

    # TODO: Allow data parallel?
    attack = RP2AttackModule(attack_config, model, None, None, None,
                             interp=interp, verbose=verbose, is_detectron=True)

    # Randomly select backgrounds from `bg_dir` and resize them
    # NOTE: we might not need this anymore?
    # all_bgs = os.listdir(os.path.expanduser(bg_dir))
    # print(f'There are {len(all_bgs)} background images in {bg_dir}.')
    # idx = np.arange(len(all_bgs))
    # np.random.shuffle(idx)
    # bg_size = img_size
    # backgrounds = torch.zeros((num_bg, 3) + bg_size, )
    # for i, index in enumerate(idx[:num_bg]):
    #     bg = torchvision.io.read_image(join(bg_dir, all_bgs[index])) / 255
    #     backgrounds[i] = T.resize(bg, bg_size, antialias=True)

    # Generate an adversarial patch
    if synthetic:
        print('=> Generating adversarial patch on synthetic signs...')
        # Resize object to the specify size and pad obj and masks to image size
        pad_size = [(img_size[1] - obj_size[1]) // 2,
                    (img_size[0] - obj_size[0]) // 2,
                    (img_size[1] - obj_size[1]) // 2 + obj_size[1] % 2,
                    (img_size[0] - obj_size[0]) // 2 + obj_size[0] % 2]  # left, top, right, bottom
        obj, obj_mask = get_object_and_mask_from_numpy(obj_numpy, obj_size, pad_size=pad_size)
        patch_mask_ = T.resize(patch_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
        patch_mask_ = T.pad(patch_mask_, pad_size)

        print(f'=> Start attacking...')
        with torch.enable_grad():
            adv_patch = attack.attack(obj.to(device),
                                      obj_mask.to(device),
                                      patch_mask_.to(device),
                                      backgrounds.to(device),
                                      obj_class=obj_class,
                                      obj_size=obj_size)

        if save_images:
            torchvision.utils.save_image(obj, join(save_dir, 'obj.png'))
            torchvision.utils.save_image(obj_mask, join(save_dir, 'obj_mask.png'))
            torchvision.utils.save_image(backgrounds, join(save_dir, 'backgrounds.png'))

    else:
        print('=> Generating adversarial patch on real signs...')
        df = pd.read_csv(tgt_csv_filepath)
        df['tgt_final'] = df['tgt_final'].apply(literal_eval)
        df = df[df['final_shape'] != 'other-0.0-0.0']

        attack_images, metadata = [], []
        print('=> Collecting background images...')
        for i, batch in tqdm(enumerate(dataloader)):
            file_name = batch[0]['file_name']
            filename = file_name.split('/')[-1]
            img_df = df[df['filename'] == filename]
            if len(img_df) == 0:
                continue

            # Flip BGR to RGB and then flip back when feeding to model
            image = batch[0]['image'].float().to(device).flip(0)
            h0, w0 = batch[0]['height'], batch[0]['width']
            _, h, w = image.shape

            for _, obj in img_df.iterrows():
                img_data = (h0, w0, h / h0, w / w0, 0, 0)
                obj_label = obj['final_shape']
                if obj_label != class_names[obj_class]:
                    continue
                data = [obj_label, obj, *img_data]
                # TODO: has to pad
                image = T.resize(image, img_size, antialias=True)
                attack_images.append([image, data, str(filename), batch[0]])
                metadata.extend(batch)
                break   # This prevents duplicating the background

            if len(attack_images) >= num_bg:
                break

        print(f'=> {len(attack_images)} backgrounds collected.')
        attack_images = attack_images[:num_bg]

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
        with torch.enable_grad():
            adv_patch = attack.attack_real(attack_images,
                                           patch_mask=patch_mask.to(device),
                                           obj_class=obj_class,
                                           metadata=metadata)

    adv_patch = adv_patch[0].detach().cpu().float()

    if save_images:
        torchvision.utils.save_image(patch_mask, join(save_dir, 'patch_mask.png'))
        torchvision.utils.save_image(adv_patch, join(save_dir, 'adversarial_patch.png'))

    return adv_patch


def main(
    padded_imgsz='992,1312',
    save_dir=Path(''),
    name='exp',  # save to project/name
    obj_class=0,
    obj_size=None,
    syn_obj_path='',
    seed=0,
    synthetic=False,
    **kwargs,
):
    cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    dataloader = None
    if not synthetic:
        # Build dataloader
        dataloader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[0], mapper=BenignMapper(cfg, is_train=False),
            # cfg, cfg.DATASETS.TEST[0],
            batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS
        )

    adv_patch = generate_adv_patch(
        model, obj_numpy, patch_mask, img_size=img_size, obj_size=obj_size,
        save_dir=save_dir, synthetic=synthetic, dataloader=dataloader, **kwargs)

    # Save adv patch
    patch_path = join(save_dir, f'adv_patch.pkl')
    print(f'Saving the generated adv patch to {patch_path}...')
    pickle.dump([adv_patch, patch_mask], open(patch_path, 'wb'))

    patch_metadata = {
        'synthetic': synthetic,
        # 'attack_type': attack_type,
        # 'rescaling': rescaling,
    }
    patch_metadata_path = join(save_dir, 'patch_metadata.pkl')
    print(f'Saving the generated adv patch metadata to {patch_metadata_path}...')
    pickle.dump(patch_metadata, open(patch_metadata_path, 'wb'))


if __name__ == "__main__":
    args = eval_args_parser(True)
    print('Command Line Args:', args)
    args.device = 'cuda'
    if args.patch_size_inch is not None:
        args.mask_path = None

    # Verify some args
    cfg = setup_detectron_test_args(args, OTHER_SIGN_CLASS)
    assert args.dataset in DATASETS

    # Register dataset
    if 'mtsd' in args.dataset:
        assert 'mtsd' in cfg.DATASETS.TEST[0], \
            'MTSD is specified as dataset in args but not config file'
        dataset_params = register_mtsd(
            use_mtsd_original_labels='orig' in args.dataset,
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )
    else:
        assert 'mapillary' in cfg.DATASETS.TEST[0], \
            'Mapillary is specified as dataset in args but not config file'
        dataset_params = register_mapillary(
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )

    print(args)
    main(**vars(args))
