import ast
import pickle
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision
from adv_patch_bench.utils.image import (
    coerce_rank,
    mask_to_box,
    prepare_obj,
    resize_and_center,
)
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from hparams import PATH_DEBUG_ADV_PATCH
from kornia import augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import resize


def prep_synthetic_eval(
    syn_obj_path: str,
    img_size: Tuple[int, int] = (1536, 2048),
    obj_size: Tuple[int, int] = (128, 128),
    transform_prob: float = 1.0,
    interp: str = "bilinear",
    syn_rotate_degree: float = 15,
    syn_use_scale: bool = True,
    syn_3d_transform: bool = False,
    syn_3d_distortion: float = 0.25,
    syn_use_colorjitter: bool = False,
    syn_colorjitter_intensity: float = 0.3,
    device: str = "cuda",
):
    if not (isinstance(obj_size, tuple) or isinstance(obj_size, int)):
        raise ValueError(
            f"obj_size must be tuple of two ints, but it is {obj_size}."
        )

    # Add synthetic label to the label names at the last position
    # TODO: needed?
    # syn_sign_class = len(label_names)
    # label_names[syn_sign_class] = 'synthetic'

    if syn_3d_transform:
        transform_params = {
            "distortion_scale": syn_3d_distortion,
        }
        tf_func = K.RandomPerspective
    else:
        # TODO: This depends on our experiment and maybe we want to make it easily
        # adjsutable.
        transform_params = {
            "degrees": syn_rotate_degree,
            "translate": (0.4, 0.4),
            "scale": (0.5, 2) if syn_use_scale else None,
        }
        tf_func = K.RandomAffine

    obj_transforms = tf_func(
        p=transform_prob,
        return_transform=True,
        resample=interp,
        **transform_params,
    )
    mask_transforms = tf_func(
        p=transform_prob, resample=Resample.NEAREST, **transform_params
    )

    if syn_use_colorjitter:
        jitter_transform = K.ColorJitter(
            brightness=syn_colorjitter_intensity,
            contrast=syn_colorjitter_intensity,
            saturation=syn_colorjitter_intensity,
            hue=0.05,  # Hue can't be change much
            p=transform_prob,
        )
    else:
        jitter_transform = None

    # Load synthetic sign from file and covert them to the right size and format
    obj, obj_mask = prepare_obj(
        syn_obj_path,
        img_size,
        obj_size,
        interp=interp,
    )
    obj = obj.to(device)
    obj_mask = obj_mask.to(device)
    return (
        obj,
        obj_mask,
        obj_transforms,
        mask_transforms,
        jitter_transform,
    )


def load_annotation_df(tgt_csv_filepath: str) -> pd.DataFrame:
    """Load CSV annotation (transforms and sign class) into pd.DataFrame."""
    df = pd.read_csv(tgt_csv_filepath)
    # Converts 'tgt_final' from string to list format
    df["tgt_final"] = df["tgt_final"].apply(ast.literal_eval)
    # Exclude shapes to which we do not apply the transform to
    df = df[df["final_shape"] != "other-0.0-0.0"]
    return df


def prep_adv_patch(
    img_size: Tuple[int, int] = (1536, 2048),
    adv_patch_path: Optional[str] = None,
    attack_type: str = "none",
    synthetic: bool = False,
    obj_size: Optional[Union[int, Tuple[int, int]]] = None,
    interp: str = "bilinear",
    device: str = "cuda",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load and prepare adversarial patch along with its mask.

    Args:
        img_size (Tuple[int, int]): Size of test image in pixels.
        adv_patch_path (Optional[str], optional): Path to pickle file containing
            adversarial patch and its mask. Defaults to None.
        attack_type (str, optional): Type of attack to run. Options are "none",
            "debug", "random", "load". Defaults to "none".
        synthetic (bool, optional): Whether we are attacking synthetic signs or
            real signs. Defaults to False.
        obj_size (Optional[int], optional): Object size (width) in pixels.
            Defaults to 128.
        interp (str, optional): Interpolation method. Defaults to "bilinear".
        device (str, optional): Device for torch.Tensor. Defaults to "cuda".

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: _description_
    """
    if attack_type == "none":
        return None, None

    # Load patch from a pickle file if specified
    # if attack_type == 'per-sign':
    # TODO: make script to generate dummy patch for per-sign attack
    # adv_patch = torch.zeros((3, obj_size, obj_size))
    adv_patch, patch_mask = pickle.load(open(adv_patch_path, "rb"))

    adv_patch = coerce_rank(adv_patch, 3)
    patch_mask = coerce_rank(patch_mask, 3)
    patch_mask = patch_mask.to(device)
    patch_height, patch_width = adv_patch.shape[-2:]

    if attack_type == "debug":
        # Load 'arrow on checkboard' patch if specified (for debug)
        adv_patch = torchvision.io.read_image(PATH_DEBUG_ADV_PATCH)
        adv_patch = adv_patch.float()[:3, :, :] / 255
        adv_patch = resize(adv_patch, (patch_height, patch_width))
    elif attack_type == "random":
        # Patch with uniformly random pixels
        adv_patch = torch.rand(3, patch_height, patch_width)

    if synthetic:
        # Adv patch and mask have to be made compatible with random
        # transformation for synthetic signs
        hw_ratio = patch_mask.shape[-2] / patch_mask.shape[-1]
        if isinstance(obj_size, int):
            obj_size_px = (round(obj_size * hw_ratio), obj_size)
        else:
            obj_size_px = obj_size
        assert isinstance(obj_size, tuple) or isinstance(
            obj_size, int
        ), f"obj_size is {obj_size}. It must be int or tuple of two ints!"

        # Resize patch_mask and adv_patch to obj_size_px and place them in
        # middle of image by padding to image_size.
        patch_mask = resize_and_center(
            patch_mask, img_size=img_size, obj_size=obj_size_px, is_binary=True
        )
        adv_patch = resize_and_center(
            adv_patch,
            img_size=img_size,
            obj_size=obj_size_px,
            is_binary=False,
            interp=interp,
        )
    else:
        # For real sign, just resize adv_patch to have same size as patch_mask
        obj_size = patch_mask.shape[-2:]
        adv_patch = resize_and_center(
            adv_patch,
            obj_size=obj_size,
            is_binary=False,
            interp=interp,
        )

    patch_mask = patch_mask.to(device)
    adv_patch = adv_patch.to(device)
    assert adv_patch.ndim == patch_mask.ndim == 3
    assert (
        adv_patch.shape[-2:] == patch_mask.shape[-2:]
    ), f"{adv_patch.shape} does not match {patch_mask.shape}."

    return adv_patch, patch_mask


def apply_synthetic_sign(
    image: torch.FloatTensor,
    targets: Any,
    image_id: int,
    adv_patch: torch.FloatTensor,
    patch_mask: torch.FloatTensor,
    syn_obj: torch.FloatTensor,
    syn_obj_mask: torch.FloatTensor,
    obj_transforms: Any,
    mask_transforms: Any,
    jitter_transform: Any,
    syn_sign_class: int,
    device: str = "cuda",
    use_attack: bool = True,
    return_target: bool = True,
    is_detectron: bool = False,
    other_sign_class: int = None,
):
    syn_obj = coerce_rank(syn_obj, 4)
    image = coerce_rank(image, 4)
    h, w = image.shape[-2:]

    if use_attack:
        adv_patch = coerce_rank(adv_patch, 4)
        patch_mask = coerce_rank(patch_mask, 4)
        adv_obj = patch_mask * adv_patch + (1 - patch_mask) * syn_obj
    else:
        adv_obj = syn_obj
    adv_obj, tf_params = obj_transforms(adv_obj)
    adv_obj.clamp_(0, 1)
    if jitter_transform is not None:
        adv_obj = jitter_transform(adv_obj)
        adv_obj.clamp_(0, 1)

    o_mask = mask_transforms.apply_transform(
        syn_obj_mask, None, transform=tf_params.to(device)
    )
    adv_img = o_mask * adv_obj + (1 - o_mask) * image / 255
    adv_img = coerce_rank(adv_img, 3)
    adv_img *= 255

    if not return_target:
        return adv_img

    # get top left and bottom right points
    bbox = mask_to_box(o_mask == 1)
    y_min, x_min, h_obj, w_obj = [b.cpu().item() for b in bbox]

    # Since we paste a new synthetic sign on image, we have to add
    # in a new synthetic label/target to compute the metrics
    # TODO: (DEPRECATED) This does not work because COCO evaluator seems to use
    # the registered dataset to evaluate. Modifying annotation after the fact
    # does not work.
    if is_detectron:
        assert isinstance(targets, dict) and isinstance(other_sign_class, int)
        targets = deepcopy(targets)
        new_bbox = [x_min, y_min, x_min + w_obj, y_min + h_obj]
        # annotations = targets["annotations"]
        # for anno in annotations:
        #     anno["category_id"] = other_sign_class
        # new_anno = {
        #     "bbox": new_bbox,
        #     "category_id": syn_sign_class,
        #     "bbox_mode": annotations[0]["bbox_mode"],
        # }
        # annotations.append(new_anno)
        # Create new gt instances
        instances = targets["instances"]
        new_instances = Instances(instances.image_size)
        new_gt_classes = torch.zeros(len(instances) + 1, dtype=torch.int64)
        new_gt_classes[:-1] = other_sign_class
        new_gt_classes[-1] = syn_sign_class
        new_bbox = torch.tensor(new_bbox, dtype=torch.float32)
        new_instances.gt_boxes = Boxes.cat(
            [instances.gt_boxes, Boxes(new_bbox[None, :])]
        )
        new_instances.gt_classes = new_gt_classes
        targets["instances"] = new_instances
        return adv_img, targets

    # Add new target for YOLO
    assert isinstance(targets, torch.Tensor)
    label = [
        image_id,
        syn_sign_class,
        (x_min + w_obj / 2) / w,  # relative center x
        (y_min + h_obj / 2) / h,  # relative center y
        w_obj / w,  # relative width
        h_obj / h,  # relative height
        -1,
    ]
    targets = torch.cat((targets, torch.tensor(label).unsqueeze(0)))
    return adv_img, targets


def get_object_and_mask_from_numpy(
    obj_numpy: np.ndarray,
    obj_size: Optional[Tuple[int, int]] = None,
    img_size: Optional[Tuple[int, int]] = None,
    interp: str = "bicubic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get object and its mask and resize to obj_size."""
    obj_mask = torch.from_numpy(obj_numpy[:, :, -1] == 1).float().unsqueeze(0)
    obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    if img_size is not None or obj_size is not None:
        obj = resize_and_center(
            obj,
            img_size=img_size,
            obj_size=obj_size,
            is_binary=False,
            interp=interp,
        )
        obj_mask = resize_and_center(
            obj_mask, img_size=img_size, obj_size=obj_size, is_binary=True
        )
    return obj, obj_mask
