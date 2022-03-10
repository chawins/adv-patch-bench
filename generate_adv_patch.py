"""
Generate adversarial patch
"""

import argparse
import os
import pickle
import yaml
import sys
from ast import literal_eval
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as T
from PIL import Image

from adv_patch_bench.attacks.rp2 import RP2AttackModule
from hparams import TS_COLOR_LABEL_DICT
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
    elif engine:
        batch_size = model.batch_size
    else:
        LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

    model.warmup(imgsz=(1, 3) + img_size, half=half)  # warmup
    data = check_yaml(data)  # check YAML
    data = check_dataset(data)
    imgsz = check_img_size(imgsz, s=stride)
    return model, data


def generate_adv_patch(model, obj_numpy, patch_mask, device='cuda',
                       img_size=(736, 1312), obj_class=0, obj_size=None,
                       bg_dir='./', num_bg=16, save_images=False, save_dir='./',
                       generate_patch='synthetic', rescaling=False, relighting=False,
                       csv_path='mapillary.csv', dataloader=None, attack_config_path=None):
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
    backgrounds = torch.zeros((num_bg, 3) + img_size, )
    for i, index in enumerate(idx[:num_bg]):
        bg = torchvision.io.read_image(join(bg_dir, all_bgs[index])) / 255
        backgrounds[i] = T.resize(bg, img_size, antialias=True)

    print(f'=> Initializing attack...')
    with open(attack_config_path) as file:
        attack_config = yaml.load(file, Loader=yaml.FullLoader)
        attack_config['input_size'] = img_size

    # TODO: Allow data parallel?
    attack = RP2AttackModule(attack_config, model, None, None, None,
                             rescaling=rescaling, relighting=relighting, verbose=True)

    # Generate an adversarial patch
    if generate_patch == 'synthetic':
        print('=> Generating adversarial patch on synthetic signs...')
        obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
        obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
        # Resize object to the specify size and pad obj and masks to image size
        # pad_size = [(img_size[1] - obj_size[1]) // 2, (img_size[0] - obj_size[0]) // 2]  # left/right, top/bottom
        pad_size = [(img_size[1] - obj_size[1]) // 2,
                    (img_size[0] - obj_size[0]) // 2,
                    (img_size[1] - obj_size[1]) // 2 + obj_size[1] % 2,
                    (img_size[0] - obj_size[0]) // 2 + obj_size[0] % 2]  # left, top, right, bottom
        obj = T.resize(obj, obj_size, antialias=True)
        obj = T.pad(obj, pad_size)
        obj_mask = T.resize(obj_mask, obj_size, interpolation=T.InterpolationMode.NEAREST)
        obj_mask = T.pad(obj_mask, pad_size)
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

    elif generate_patch == 'real':
        print('=> Generating adversarial patch on real signs...')
        df = pd.read_csv(csv_path)
        df['tgt_final'] = df['tgt_final'].apply(literal_eval)
        df = df[df['final_shape'] != 'other-0.0-0.0']

        attack_images = []
        print('=> Collecting background images...')
        for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
            for image_i, path in enumerate(paths):
                filename = path.split('/')[-1]
                img_df = df[df['filename_y'] == filename]
                if len(img_df) == 0:
                    continue
                for _, row in img_df.iterrows():
                    (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
                    predicted_class = row['final_shape']
                    shape = predicted_class.split('-')[0]
                    # Filter out signs that are not form `obj_class`
                    # TODO: Add color to csv
                    # shape_color_classes = f'{predicted_class}-{row["color"]}'
                    # if obj_class != TS_COLOR_LABEL_DICT[shape_color_classes]:
                    #     continue
                    if shape != 'octagon':
                        continue
                    # Pad to make sure all images are of same size
                    img = im[image_i]
                    data = [predicted_class, row, h0, w0, h_ratio, w_ratio, w_pad, h_pad]
                    attack_images.append([img, data, str(filename)])
                    break   # This prevents duplicating the background

            if len(attack_images) >= num_bg:
                break

        print(f'=> {len(attack_images)} backgrounds collected.')
        attack_images = attack_images[:num_bg]

        # DEBUG: Save all the background images
        for img in attack_images:
            torchvision.utils.save_image(img[0] / 255, f'tmp/{img[2]}.png')

        # Save background filenames in txt file
        print(f'=> Saving used backgrounds in a txt file.')
        with open(join(save_dir, 'bg_filenames.txt'), 'w') as f:
            for img in attack_images:
                f.write(f'{img[2]}\n')

        print(f'=> Start attacking...')
        with torch.enable_grad():
            adv_patch = attack.transform_and_attack(attack_images,
                                                    patch_mask=patch_mask.to(device),
                                                    obj_class=obj_class)

    adv_patch = adv_patch[0].detach().cpu().float()

    if save_images:
        torchvision.utils.save_image(patch_mask, join(save_dir, 'patch_mask.png'))
        torchvision.utils.save_image(adv_patch, join(save_dir, 'adversarial_patch.png'))

    return adv_patch


def main(
    batch_size=32,  # batch size
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    weights=None,  # model.pt path(s)
    imgsz=1280,  # image width
    padded_imgsz='992,1312',
    dnn=False,  # use OpenCV DNN for ONNX inference
    half=False,
    save_dir=Path(''),
    exist_ok=True,  # existing project/name ok, do not increment
    project=ROOT / 'runs/val',  # save to project/name
    name='exp',  # save to project/name
    save_txt=False,  # save results to *.txt
    obj_class=0,
    obj_size=-1,
    obj_path='',
    bg_dir='./',
    num_bg=16,
    save_images=False,
    patch_name='adv_patch',
    seed=0,
    generate_patch='synthetic',
    rescaling=False,
    relighting=False,
    csv_path='',
    data=None,
    attack_config_path=None
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    img_size = tuple([int(i) for i in padded_imgsz.split(',')])
    assert len(img_size) == 2
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model (YOLOv5)
    model, data = load_yolov5(weights, device, imgsz, img_size, data, dnn, half)

    # Configure object size
    # NOTE: We assume that the target object fits the tensor in the same way
    # that we generate canonical masks (e.g., largest inscribed circle, octagon,
    # etc. in a square). Patch and patch mask are defined with respect to this
    # object tensor, and they should all have the same width and height.
    obj_numpy = np.array(Image.open(obj_path).convert('RGBA')) / 255

    # TODO: FIXED?
    h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]
    # h_w_ratio = img_size[0] / img_size[1]

    if obj_size == -1:
        obj_size = int(min(img_size) * 0.1)
    if isinstance(obj_size, int):
        obj_size = (int(obj_size * h_w_ratio), obj_size)

    print('obj size', obj_size)
    # Define patch location and size
    patch_mask = torch.zeros((1, ) + obj_size)
    # TODO: Move this to a separate script for generating patch size/location
    # Example: 10x10-inch patch in the middle of 36x36-inch sign
    mid_height = obj_size[0] // 2 + 40
    # mid_height = obj_size[0] // 2
    mid_width = obj_size[1] // 2
    patch_size = 10
    h = int(patch_size / 36 / 2 * obj_size[0])
    w = int(patch_size / 36 / 2 * obj_size[1])
    patch_mask[:, mid_height - h:mid_height + h, mid_width - w:mid_width + w] = 1

    dataloader = None
    if generate_patch == 'real':
        task = 'train'  # Use transform-annotated images from training set
        stride, pt = model.stride, model.pt
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride,
                                       single_cls=False, pad=0.5, rect=pt, shuffle=True,
                                       workers=8, prefix=colorstr(f'{task}: '))[0]

    adv_patch = generate_adv_patch(
        model, obj_numpy, patch_mask, device=device, img_size=img_size,
        obj_class=obj_class, obj_size=obj_size, bg_dir=bg_dir, num_bg=num_bg,
        save_images=save_images, save_dir=save_dir, generate_patch=generate_patch,
        rescaling=rescaling, relighting=relighting, csv_path=csv_path, dataloader=dataloader,
        attack_config_path=attack_config_path)

    # Save adv patch
    patch_path = join(save_dir, f'{patch_name}.pkl')
    print(f'Saving the generated adv patch to {patch_path}...')
    pickle.dump([adv_patch, patch_mask], open(patch_path, 'wb'))

    patch_metadata = {
        'generate_patch': generate_patch,
        'rescaling': rescaling,
        'relighting': relighting
    }
    patch_metadata_path = join(save_dir, 'patch_metadata.pkl')
    print(f'Saving the generated adv patch metadata to {patch_metadata_path}...')
    pickle.dump(patch_metadata, open(patch_metadata_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', type=int, default=1280, help='inference size (width)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # Our attack and evaluate
    parser.add_argument('--seed', type=int, default=0, help='set random seed')
    parser.add_argument('--padded_imgsz', type=str, default='992,1312',
                        help='final image size including padding (height,width); comma-separated')
    parser.add_argument('--patch-name', type=str, default='adv_patch',
                        help='name of pickle file to save the generated patch')
    parser.add_argument('--obj-class', type=int, default=0, help='class of object to attack')
    parser.add_argument('--obj-size', type=int, default=-1, help='object width in pixels')
    parser.add_argument('--obj-path', type=str, default='', help='path to synthetic image of the object')
    parser.add_argument('--bg-dir', type=str, default='', help='path to background directory')
    parser.add_argument('--num-bg', type=int, default=16, help='number of backgrounds to generate adversarial patch')
    parser.add_argument('--save-images', action='store_true', help='save generated patch')
    parser.add_argument('--generate-patch', type=str, default='synthetic',
                        help=("create patch using synthetic stop signs if 'synthetic', use real signs if 'real'"))
    parser.add_argument('--rescaling', action='store_true',
                        help=('randomly rescale synthetic sign size during patch generations'
                              '(use with --generate-patch synthetic)'))
    parser.add_argument('--relighting', action='store_true',
                        help='apply color jitter to patch during attack')
    parser.add_argument('--csv-path', type=str, default='',
                        help=('path to csv file with the annotated transform '
                              'data (required if --generate-patch real)'))
    parser.add_argument('--attack-config-path', help='path to yaml file with attack configs')
    opt = parser.parse_args()
    print(opt)

    main(**vars(opt))
