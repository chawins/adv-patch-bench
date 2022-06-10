"""
Generate adversarial patch
"""

import os
import pickle
import sys
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
from PIL import Image
from torch.nn import DataParallel

from adv_patch_bench.attacks.rp2 import RP2AttackModule
from adv_patch_bench.utils.argparse import eval_args_parser, parse_dataset_name
from adv_patch_bench.utils.image import get_obj_width
from gen_mask import generate_mask
from hparams import LABEL_LIST, MAPILLARY_IMG_COUNTS_DICT
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (LOGGER, check_dataset, check_img_size,
                                  check_yaml, colorstr, increment_path)
from yolov5.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def load_yolov5(weights, device, imgsz, img_size, data, dnn, half):
    """Set up YOLOv5 model. This can be replaced with other models."""
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    else:
        LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

    # model.warmup(imgsz=(1, 3) + img_size, half=half)  # warmup
    data = check_yaml(data)  # check YAML
    data = check_dataset(data)
    imgsz = check_img_size(imgsz, s=stride)
    # TODO: try to get dataparallel working
    # model = DataParallel(model, device_ids=[0, 1]).to('cuda')
    return model, data


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
    tgt_csv_filepath: str = 'mapillary.csv',
    dataloader: Any = None,
    attack_config_path: str = None,
    **kwargs,
):
    """Generate adversarial patch

    Args:
        model ([type]): [description]
        obj_numpy (np.ndarray): Image of object in numpy with RGBA channels
        patch_mask ([type]): [description]
        device (str, optional): [description]. Defaults to 'cuda'.
        img_size (tuple, optional): [description]. Defaults to (960, 1280).
        obj_class (int, optional): [description]. Defaults to 0.
        obj_size ([type], optional): [description]. Defaults to None.
        bg_dir (str, optional): [description]. Defaults to './'.
        num_bg (int, optional): [description]. Defaults to 16.
        save_images (bool, optional): [description]. Defaults to False.
        save_dir (str, optional): [description]. Defaults to './'.

    Returns:
        [type]: [description]
    """

    # Randomly select backgrounds from `bg_dir` and resize them
    all_bgs = os.listdir(os.path.expanduser(bg_dir))
    print(f'There are {len(all_bgs)} background images in {bg_dir}.')
    idx = np.arange(len(all_bgs))
    np.random.shuffle(idx)
    # FIXME: does this break anything?
    # bg_size = (img_size[0] - 32, img_size[1] - 32)
    bg_size = img_size
        
    backgrounds = torch.zeros((num_bg, 3) + bg_size, )
    
    for i, index in enumerate(idx[:num_bg]):
        bg = torchvision.io.read_image(join(bg_dir, all_bgs[index])) / 255
        backgrounds[i] = T.resize(bg, bg_size, antialias=True)

    # getting object classes names
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_names = LABEL_LIST[args.dataset]

    print(f'=> Initializing attack...')
    with open(attack_config_path) as file:
        attack_config = yaml.load(file, Loader=yaml.FullLoader)
        attack_config['input_size'] = img_size

    # TODO: Allow data parallel?
    attack = RP2AttackModule(attack_config, model, None, None, None, verbose=True, interp=args.interp)

    # Generate an adversarial patch
    if synthetic:
        print('=> Generating adversarial patch on synthetic signs...')
        obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
        obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
        # Resize object to the specify size and pad obj and masks to image size
        pad_size = [(img_size[1] - obj_size[1]) // 2,
                    (img_size[0] - obj_size[0]) // 2,
                    (img_size[1] - obj_size[1]) // 2 + obj_size[1] % 2,
                    (img_size[0] - obj_size[0]) // 2 + obj_size[0] % 2]  # left, top, right, bottom
        obj = T.resize(obj, obj_size, antialias=True)
        obj = T.pad(obj, pad_size)

        obj_mask = T.resize(obj_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
        obj_mask = T.pad(obj_mask, pad_size)

        
        patch_mask = patch_mask.unsqueeze(dim=0)
        patch_mask_ = T.resize(patch_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
        
        patch_mask_ = T.pad(patch_mask_, pad_size)
        

        print(f'=> Start attacking...')
        with torch.enable_grad():
            adv_patch = attack.attack(obj.to(device),
                                      obj_mask.to(device),
                                      patch_mask_.to(device),
                                      backgrounds.to(device),
                                      obj_class=12,
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

        attack_images = []
        print('=> Collecting background images...')
        
        # filename_debug = '6obAf9CRQh_dBFHPAIiRFQ.jpg'
        filename_debug = 'HwBonTfmkfIOANA1O7B2OQ.jpg'

        from tqdm import tqdm
        for batch_i, (im, targets, paths, shapes) in tqdm(enumerate(dataloader)):
            # import pdb
            # pdb.set_trace()

            # DEBUG
            # if f'/datadrive/nab_126/data/mapillary_vistas/no_color/combined/images/{filename_debug}' not in paths:
                # continue

            for image_i, path in enumerate(paths):
                filename = path.split('/')[-1]

                # DEBUG
                # if filename != filename_debug:
                #     continue
                # tmp_df = pd.read_csv('results_per_label_errors.csv')
                # if filename not in tmp_df['filename'].values:
                #     continue
                # print('attack_images', len(attack_images))
                
                img_df = df[df['filename'] == filename]

                if len(img_df) == 0:
                    continue

                for _, row in img_df.iterrows():
                    (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
                    predicted_class = row['final_shape']

                    # Filter out images that do not have the obj_class
                    if predicted_class != class_names[obj_class]:
                        continue

                    # Pad to make sure all images are of same size
                    img = im[image_i]
                    data = [predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad]
                    attack_images.append([img, data, str(filename)])
                    break   # This prevents duplicating the background
                
                if len(attack_images) >= num_bg:
                    break
            if len(attack_images) >= num_bg:
                break

        print(f'=> {len(attack_images)} backgrounds collected.')
        attack_images = attack_images[:num_bg]

        # DEBUG: Save all the background images
        # for img in attack_images:
        #     os.makedirs('tmp', exist_ok=True)
        #     torchvision.utils.save_image(img[0] / 255, f'tmp/{img[2]}')

        # Save background filenames in txt file
        print(f'=> Saving used backgrounds in a txt file.')
        with open(join(save_dir, 'bg_filenames.txt'), 'w') as f:
            for img in attack_images:
                f.write(f'{img[2]}\n')

        print(f'=> Start attacking...')
        with torch.enable_grad():
            adv_patch = attack.attack_real(attack_images,
                                           patch_mask=patch_mask.to(device),
                                           obj_class=obj_class)

    adv_patch = adv_patch[0].detach().cpu().float()

    if save_images:
        torchvision.utils.save_image(patch_mask, join(save_dir, 'patch_mask.png'))
        torchvision.utils.save_image(adv_patch, join(save_dir, 'adversarial_patch.png'))

    return adv_patch


def main(
    device='',
    batch_size=32,  # batch size
    weights=None,  # model.pt path(s)
    imgsz=1280,  # image width
    padded_imgsz='992,1312',
    dnn=False,  # use OpenCV DNN for ONNX inference
    half=False,
    save_dir=Path(''),
    exist_ok=True,  # existing project/name ok, do not increment
    project=ROOT / 'runs/val',  # save to project/name
    name='exp',  # save to project/name
    # save_txt=False,  # save results to *.txt
    obj_class=0,
    obj_size=-1,
    syn_obj_path='',
    seed=0,
    synthetic: bool = False,
    # rescaling=False,
    data=None,
    task='test',
    mask_dir=None,
    **kwargs,
):
    cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    img_size = tuple([int(i) for i in padded_imgsz.split(',')])
    assert len(img_size) == 2
    class_names = LABEL_LIST[args.dataset]

    # Set up directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir = save_dir / class_names[obj_class]
    os.makedirs(save_dir, exist_ok=True)

    # Load model (YOLO)
    device = select_device(device, batch_size=batch_size)
    model, data = load_yolov5(weights, device, imgsz, img_size, data, dnn, half)

    num_bg = args.num_bg
    class_name = list(MAPILLARY_IMG_COUNTS_DICT.keys())[int(obj_class)]
    if num_bg < 1:
        assert class_name is not None
        print(f'num_bg is a fraction ({num_bg}).')
        num_bg = round(MAPILLARY_IMG_COUNTS_DICT[class_name] * num_bg)
        print(f'For {class_name}, this is {num_bg} images.')
    num_bg = int(num_bg)
    kwargs['num_bg'] = num_bg

    # Configure object size
    # NOTE: We assume that the target object fits the tensor in the same way
    # that we generate canonical masks (e.g., largest inscribed circle, octagon,
    # etc. in a square). Patch and patch mask are defined with respect to this
    # object tensor, and they should all have the same width and height.
    obj_numpy = np.array(Image.open(syn_obj_path).convert('RGBA')) / 255
    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]
    
    if obj_size == -1:
        obj_size = int(min(img_size) * 0.1)
    if isinstance(obj_size, int):
        obj_size = (round(obj_size * h_w_ratio), obj_size)

    if mask_dir is not None:
        # Load path mask from file if specified (gen_mask.py)
        mask_path = join(mask_dir, f'{name}.png')
        patch_mask = torchvision.io.read_image(mask_path)
        patch_mask = patch_mask.float() / 255   

        patch_mask = patch_mask[0]     
    else:
        # Otherwise, generate a new mask here
        # Get size in inch from sign class
        obj_width_inch = get_obj_width(obj_class, class_names)
        patch_mask = generate_mask(obj_numpy, obj_size, obj_width_inch)

    dataloader = None
    if not synthetic:
        stride, pt = model.stride, model.pt
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride,
                                       single_cls=False, pad=0.5, rect=pt, shuffle=True,
                                       workers=8, prefix=colorstr(f'{task}: '))[0]

    adv_patch = generate_adv_patch(
        model, obj_numpy, patch_mask, device=device, img_size=img_size,
        obj_size=obj_size, obj_class=obj_class, save_dir=save_dir, synthetic=synthetic,
        dataloader=dataloader, **kwargs)
    
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
    args = eval_args_parser(False, root=ROOT)
    parse_dataset_name(args)
    print(args)
    if args.patch_size_inch is not None:
        args.mask_path = None
    main(**vars(args))
