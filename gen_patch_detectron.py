"""
Generate adversarial patch
"""

import os
import pickle
import random
from ast import literal_eval
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms.functional as T
import yaml
from adv_patch_bench.dataloaders.detectron import custom_build
from detectron2.engine import DefaultPredictor
from PIL import Image
from tqdm import tqdm

from adv_patch_bench.attacks.detectron_attack_wrapper import (
    DetectronAttackWrapper,
)
from adv_patch_bench.attacks.rp2.rp2_detectron import RP2AttackDetectron
from adv_patch_bench.attacks.utils import get_object_and_mask_from_numpy
from adv_patch_bench.dataloaders import (
    get_mapillary_dict,
    get_mtsd_dict,
    register_mapillary,
    register_mtsd,
)
from adv_patch_bench.dataloaders.detectron.mapper import BenignMapper
from adv_patch_bench.utils.argparse import (
    eval_args_parser,
    setup_detectron_test_args,
)
from adv_patch_bench.utils.detectron.custom_sampler import (
    ShuffleInferenceSampler,
)
from adv_patch_bench.utils.image import resize_and_center
from gen_mask import get_mask_from_syn_image
from hparams import (
    DATASETS,
    LABEL_LIST,
    MAPILLARY_IMG_COUNTS_DICT,
    OTHER_SIGN_CLASS,
    PATH_BG_TXT_FILE,
)


def collect_backgrounds(
    dataloader: Any,
    img_size: Tuple[int, int],
    num_bg: Union[int, float],
    device: str,
    df: Optional[pd.DataFrame] = None,
    class_name: Optional[str] = None,
    filter_file_names: Optional[List[str]] = None,
) -> Tuple[List[Any], List[Any], np.ndarray]:
    """Collect background images to be used by the attack.

    Args:
        dataloader (Any): Detectron data loader.
        img_size (Tuple[int, int]): Desired background iamge size.
        num_bg (Union[int, float]): Num total background images to collect.
        device (str): Device to store background images.
        df (pd.DataFrame, optional): Our annotation DataFrame. If specified,
            only select images belong to df. Defaults to None.
        class_name (str, optional): Desired class. If specified, only
            select images from class class_name. Defaults to None.
        filter_file_names (List[str], optional): List of image file names to use
            as backgrounds.

    Returns:
        attack_images: List of background images and their metadata, used by the
            attack.
        metadata: List of metadata of original background image, used as part of
            input to detectron model.
        backgrond: Numpy array of background images
    """
    if num_bg < 1:
        assert class_name is not None
        print(f"num_bg is a fraction ({num_bg}).")
        num_bg = round(MAPILLARY_IMG_COUNTS_DICT[class_name] * num_bg)
        print(f"For {class_name}, this is {num_bg} images.")
    num_bg = int(num_bg)

    attack_images, metadata = [], []
    backgrounds = torch.zeros((num_bg, 3) + img_size)
    num_collected = 0

    print("=> Collecting background images...")

    # DEBUG: count images
    # counts = [0] * 12
    # class_names = LABEL_LIST[args.dataset]

    for _, batch in tqdm(enumerate(dataloader)):
        file_name = batch[0]["file_name"]
        filename = file_name.split("/")[-1]

        # If img_txt_path is specified, ignore other file names
        if filter_file_names is not None and filename not in filter_file_names:
            continue

        # If df is specified, ignore images that are not in df
        if df is not None:
            img_df = df[df["filename"] == filename]
            if img_df.empty:
                continue

        found = False
        obj, obj_label = None, class_name
        if class_name is not None and df is not None:
            # If class_name is also specified, make sure that there is at least
            # one sign with label class_name in image.
            for _, obj_i in img_df.iterrows():
                obj_label = obj_i["final_shape"]
                if obj_label == class_name:
                    found = True
                    obj = obj_i
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
            image = batch[0]["image"].float().to(device).flip(0)
            h0, w0 = batch[0]["height"], batch[0]["width"]
            _, h, w = image.shape

            assert w == img_size[1]
            if h > img_size[0]:
                # If actual height is larger than desired height, just resize
                # image and avoid padding
                image = T.resize(image, img_size, antialias=True)
                pad_top = 0
            else:
                # Otherwise, pad height
                pad_top = (img_size[0] - h) // 2
                pad_bottom = img_size[0] - h - pad_top
                image = T.pad(image, [0, pad_top, 0, pad_bottom])

            # Get new shape and assert that it is correct
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

    print(f"=> {len(attack_images)} backgrounds collected.")
    return attack_images[:num_bg], metadata[:num_bg], backgrounds / 255


def generate_adv_patch(
    model: torch.nn.Module,
    obj_numpy: np.ndarray,
    patch_mask: torch.Tensor,
    device: str = "cuda",
    img_size: Tuple[int, int] = (992, 1312),
    obj_class: int = None,
    obj_size: int = None,
    save_images: bool = False,
    save_dir: str = "./",
    synthetic: bool = False,
    tgt_csv_filepath: str = None,
    dataloader: Any = None,
    interp: str = "bilinear",
    verbose: bool = False,
    debug: bool = False,
    dataset: str = None,
    attack_config: Dict = {},
    filter_file_names: Optional[List[str]] = None,
    **kwargs,
):
    """Generate adversarial patch."""
    del kwargs  # Unused
    print(f"=> Initializing attack...")
    num_bg = attack_config["rp2"]["num_bg"]  # TODO: decouple from rp2

    # TODO: Allow data parallel?
    attack = RP2AttackDetectron(
        attack_config, model, None, None, None, interp=interp, verbose=verbose
    )

    df = None
    class_name = LABEL_LIST[dataset][obj_class]
    if not synthetic:
        # To attack with real signs, we have to pick background images that
        # contain target class and exists in our annotation (since we need to
        # know the transformation).
        df = pd.read_csv(tgt_csv_filepath)
        df["tgt_final"] = df["tgt_final"].apply(literal_eval)
        df = df[df["final_shape"] != "other-0.0-0.0"]

    attack_images, metadata, backgrounds = collect_backgrounds(
        dataloader,
        img_size,
        num_bg,
        device,
        df=df,
        class_name=class_name,
        filter_file_names=filter_file_names,
    )

    # Save background filenames in txt file
    print("=> Saving used backgrounds in a txt file.")
    with open(join(save_dir, f"bg_filenames-{num_bg}.txt"), "w") as f:
        for img in attack_images:
            f.write(f"{img[2]}\n")

    if debug:
        # Save all the background images
        for img in attack_images:
            os.makedirs(join(save_dir, "backgrounds"), exist_ok=True)
            torchvision.utils.save_image(
                img[0] / 255, join(save_dir, "backgrounds", img[2])
            )

    # Generate an adversarial patch
    if synthetic:
        print("=> Generating adversarial patch on synthetic signs...")
        # Resize object to the specify size and pad obj and masks to image size
        # left, top, right, bottom
        obj, obj_mask = get_object_and_mask_from_numpy(
            obj_numpy, obj_size=obj_size, img_size=img_size, interp=interp
        )
        patch_mask_padded = resize_and_center(
            patch_mask, img_size=img_size, obj_size=obj_size, is_binary=True
        )

        metadata = DetectronAttackWrapper.clone_metadata(backgrounds, metadata)
        adv_patch = attack.attack_synthetic(
            obj.to(device),
            obj_mask.to(device),
            patch_mask_padded.to(device),
            backgrounds.to(device),
            obj_class=obj_class,
            metadata=metadata,
        )
        if save_images:
            torchvision.utils.save_image(obj, join(save_dir, "obj.png"))
            torchvision.utils.save_image(
                obj_mask, join(save_dir, "obj_mask.png")
            )
    else:
        print("=> Generating adversarial patch on real signs...")
        # Clone metadata for each attack image ("background"). The metadata will
        # be automatically handled by the attack.
        metadata = DetectronAttackWrapper.clone_metadata(
            [obj[0] for obj in attack_images], metadata
        )
        adv_patch = attack.attack_real(
            attack_images,
            patch_mask=patch_mask.to(device),
            obj_class=obj_class,
            metadata=metadata,
        )

    adv_patch = adv_patch[0].detach().cpu().float()
    if save_images:
        torchvision.utils.save_image(
            patch_mask, join(save_dir, "patch_mask.png")
        )
        torchvision.utils.save_image(
            adv_patch, join(save_dir, "adversarial_patch.png")
        )

    return adv_patch


def main(
    dataset: str = "mapillary_no_color",
    padded_imgsz: str = "992,1312",
    save_dir: Path = Path(""),
    name: str = "exp",  # save to project/name
    obj_class: int = None,
    obj_size: int = None,
    syn_obj_path: str = "",
    seed: int = 0,
    synthetic: bool = False,
    attack_config_path: str = None,
    num_samples: int = 0,
    mask_name: str = "10x10",
    img_txt_path: str = "",
    **kwargs,
):
    cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get image size from padded_imgsz
    img_size = tuple([int(i) for i in padded_imgsz.split(",")])
    if len(img_size) != 2:
        raise ValueError(
            "padded_imgsz must be two numbers separated by a comma, but it is "
            f"{padded_imgsz}."
        )
    class_names = LABEL_LIST[dataset]

    # Set up model from config
    model = DefaultPredictor(cfg).model

    # Configure object size
    # NOTE: We assume that the target object fits the tensor in the same way
    # that we generate canonical masks (e.g., largest inscribed circle, octagon,
    # etc. in a square). Patch and patch mask are defined with respect to this
    # object tensor, and they should all have the same width and height.
    # obj_numpy = np.array(Image.open(syn_obj_path).convert("RGBA")) / 255
    # h_w_ratio = obj_numpy.shape[0] / obj_numpy.shape[1]

    # # Deterimine object size in pixels
    # if obj_size is None:
    #     obj_size = int(min(img_size) * 0.1)
    # if isinstance(obj_size, int):
    #     obj_size = (round(obj_size * h_w_ratio), obj_size)
    # assert isinstance(obj_size, tuple) and all(
    #     [isinstance(o, int) for o in obj_size]
    # )

    # # Get object width in inch
    # obj_width_inch = get_obj_width(obj_class, class_names)
    # # TODO: why don't we include mask_name in attack config?
    # patch_mask = generate_mask(
    #     mask_name,
    #     obj_numpy,
    #     obj_size,
    #     obj_width_inch,
    # )
    obj_numpy, patch_mask, obj_size = get_mask_from_syn_image(
        obj_class, syn_obj_path, obj_size, img_size, mask_name, class_names
    )

    if img_txt_path:
        img_txt_path = os.path.join(PATH_BG_TXT_FILE, img_txt_path)
        with open(img_txt_path, "r") as f:
            filter_file_names = set(f.read().splitlines())
    else:
        filter_file_names = None

    # Build dataloader
    # dataloader = build_detection_test_loader(
    dataloader = custom_build.build_detection_test_loader(
        cfg,
        cfg.DATASETS.TEST[0],
        mapper=BenignMapper(cfg, is_train=False),
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=ShuffleInferenceSampler(
            num_samples if filter_file_names is None else len(filter_file_names)
        ),
        pin_memory=True,
        filter_file_names=filter_file_names,
    )

    # Load attack config from a separate YAML file
    with open(attack_config_path) as f:
        attack_config = yaml.load(f, Loader=yaml.FullLoader)
    attack_config["input_size"] = img_size

    adv_patch = generate_adv_patch(
        model,
        obj_numpy,
        patch_mask,
        dataset=dataset,
        img_size=img_size,
        obj_size=obj_size,
        save_dir=save_dir,
        synthetic=synthetic,
        dataloader=dataloader,
        obj_class=obj_class,
        name=name,
        attack_config=attack_config,
        filter_file_names=filter_file_names,
        **kwargs,
    )

    # Save adv patch
    patch_path = join(save_dir, "adv_patch.pkl")
    print(f"Saving the generated adv patch to {patch_path}...")
    pickle.dump([adv_patch, patch_mask], open(patch_path, "wb"))

    # Save attack config
    patch_metadata_path = join(save_dir, "config.yaml")
    print(f"Saving the adv patch metadata to {patch_metadata_path}...")
    patch_metadata = {"synthetic": synthetic, **attack_config}
    with open(patch_metadata_path, "w") as outfile:
        yaml.dump(patch_metadata, outfile)


if __name__ == "__main__":
    args = eval_args_parser(True)
    print("Command Line Args:", args)
    args.device = "cuda"

    # Verify some args
    cfg = setup_detectron_test_args(args, OTHER_SIGN_CLASS)
    assert args.dataset in DATASETS
    split = cfg.DATASETS.TEST[0].split("_")[1]

    # Register dataset
    if "mtsd" in args.dataset:
        assert (
            "mtsd" in cfg.DATASETS.TEST[0]
        ), "MTSD is specified as dataset in args but not config file"
        dataset_params = register_mtsd(
            use_mtsd_original_labels="orig" in args.dataset,
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )
        data_list = get_mtsd_dict(split, *dataset_params)
    else:
        assert (
            "mapillary" in cfg.DATASETS.TEST[0]
        ), "Mapillary is specified as dataset in args but not config file"
        dataset_params = register_mapillary(
            use_color=args.use_color,
            ignore_other=args.data_no_other,
            only_annotated=args.annotated_signs_only,
        )
        data_list = get_mapillary_dict(split, *dataset_params)
    num_samples = len(data_list)

    print(args)
    args_dict = vars(args)
    del args
    main(**args_dict, num_samples=num_samples)
