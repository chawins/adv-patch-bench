"""
Evaluator that handles evaluation of Detectron2 model with and without applying
adversarial patch.

This code is inspired by
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
"""
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import pandas as pd
import numpy as np
import torch
from adv_patch_bench.attacks import base_attack
from adv_patch_bench.attacks.attack import setup_attack
from adv_patch_bench.attacks.utils import (
    apply_synthetic_sign,
    load_annotation_df,
    prep_adv_patch,
    prep_synthetic_eval,
)
from adv_patch_bench.transforms import transform_and_apply_patch
from adv_patch_bench.utils.detectron import build_evaluator
from adv_patch_bench.utils.image import coerce_rank, resize_and_center
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

_DEFAULT_IOU_THRESHOLDS = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)


def _pad_or_crop_height(
    img: torch.Tensor, size: Tuple[int, int]
) -> torch.Tensor:
    if img is None:
        return None
    cur_h, cur_w = img.shape[-2:]
    h, w = size
    # assert (
    #     w == cur_w
    # ), f"Width should already match ({img.shape[-2:]} vs {size})!"
    # if h > cur_h:
    #     # Pad to adv_patch/patch_mask to match cur image height
    #     img = resize_and_center(img, img_size=size)
    # elif h < cur_h:
    #     # Crop adv_patch/patch_mask to match cur image height
    #     top_crop = (cur_h - h) // 2
    #     bot_crop = cur_h - h - top_crop
    #     img = img[..., top_crop:-bot_crop, :]

    if cur_h < h or cur_w < w:
        # Pad if too small in either dimension
        img = resize_and_center(img, img_size=size)

    cur_h, cur_w = img.shape[-2:]
    assert cur_h >= h and cur_w >= w, f"{(cur_h, cur_w)} vs {size}"

    # Crop if too large
    top_crop = (cur_h - h) // 2
    bot_crop = h + top_crop
    left_crop = (cur_w - w) // 2
    right_crop = w + left_crop
    img = img[..., top_crop:bot_crop, left_crop:right_crop]

    assert img.shape[-2:] == size
    return img


class DetectronEvaluator:
    def __init__(
        self,
        cfg: CfgNode,
        config_eval: Dict[str, Any],
        config_attack: Dict[str, Any],
        model: torch.nn.Module,
        dataloader: Any,
        class_names: List[str],
        iou_thres: float = 0.5,
        all_iou_thres: np.ndarray = _DEFAULT_IOU_THRESHOLDS,
    ) -> None:
        """Evaluator wrapper for detectron model.

        Args:
            cfg: Detectron model config.
            args: Dictionary containing eval parameters.
            config_attack: Dictionary containing attack parameters.
            model: Target model.
            dataloader: Dataset to run attack on.
            class_names: List of class names in string.
            all_iou_thres: Array of IoU thresholds for computing score.
        """
        # cfg can be modified by model so we copy it first
        self.cfg = cfg.clone()
        self.model = model
        self.device = self.model.device
        self.dataloader = dataloader
        self.input_format: str = cfg.INPUT.FORMAT
        assert self.input_format == "BGR", "Only allow BGR input format for now"
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.verbose: bool = config_eval["verbose"]
        self.debug: bool = config_eval["debug"]
        self.img_size: Tuple[int, int] = config_eval["img_size"]
        num_eval = config_eval["num_eval"]
        self.num_test: int = num_eval if num_eval is not None else 1e9
        self.conf_thres: float = config_eval["conf_thres"]
        self.interp: str = config_eval["interp"]
        self.class_names: List[str] = class_names
        self.adv_sign_class_id: int = config_eval["obj_class"]
        self.other_sign_class: int = config_eval["other_sign_class"]
        self.transform_params = {
            "transform_mode": config_eval["reap_transform_mode"],
            "use_relight": config_eval["reap_use_relight"],
            "interp": self.interp,
        }
        # TODO(feature): Make this an option
        self.fixed_input_size = False

        # Synthetic benchmark params
        self.synthetic: bool = config_eval["synthetic"]
        self.obj_size: Tuple[int, int] = config_eval["syn_obj_size"]
        self.syn_transform_params = {
            "syn_rotate": config_eval["syn_rotate"],
            "syn_scale": config_eval["syn_scale"],
            "syn_colorjitter": config_eval["syn_colorjitter"],
            "syn_3d_dist": config_eval["syn_3d_dist"],
        }
        self.syn_obj_path = config_eval["syn_obj_path"]

        # Load annotation DataFrame. "Other" signs are discarded.
        self.df: pd.DataFrame = load_annotation_df(
            config_eval["tgt_csv_filepath"]
        )
        self.annotated_signs_only: bool = config_eval["annotated_signs_only"]

        # Loading file names from the specified text file
        self.skipped_filename_list = []
        img_txt_path = config_eval["img_txt_path"]
        if img_txt_path is not None:
            print(f"Loading file names from {img_txt_path}...")
            with open(img_txt_path, "r") as f:
                self.skipped_filename_list = f.read().splitlines()

        if config_eval["run_only_img_txt"] is not None:
            self.run_only_img_txt: bool = config_eval["run_only_img_txt"]
        else:
            self.run_only_img_txt: bool = False

        if self.run_only_img_txt:
            print("Evaluation will run only on these files.")
        else:
            print("Evaluation will skip these files.")

        # Build COCO evaluator
        self.evaluator = build_evaluator(
            cfg, cfg.DATASETS.TEST[0], synthetic=self.synthetic
        )

        # Set up list of IoU thresholds to consider
        self.iou_thres_idx = int(np.where(all_iou_thres == iou_thres)[0])
        self.all_iou_thres = torch.from_numpy(all_iou_thres).to(self.device)

        # Set up attack if applicable
        self.attack_type: str = config_eval["attack_type"]
        self.use_attack: bool = self.attack_type != "none"
        if config_attack is not None:
            self.attack: base_attack.DetectorAttackModule = setup_attack(
                config_attack=config_attack,
                is_detectron=True,
                model=model,
                input_size=self.img_size,
                verbose=self.verbose,
            )
        else:
            self.attack = None
        self.adv_patch_path: str = config_eval["adv_patch_path"]

        # Visualization
        self.num_vis: int = config_eval.get("num_vis", 0)
        self.vis_save_dir = os.path.join(config_eval["result_dir"], "vis")
        os.makedirs(self.vis_save_dir, exist_ok=True)
        if config_eval["vis_conf_thres"] is not None:
            self.vis_conf_thres: float = config_eval["vis_conf_thres"]
        else:
            self.vis_conf_thres: float = config_eval["conf_thres"]
        self.vis_show_bbox: bool = config_eval["vis_show_bbox"]

    @staticmethod
    def clone_metadata(imgs, metadata):
        metadata_clone = []
        for img, m in zip(imgs, metadata):
            data_dict = {}
            for keys in m:
                data_dict[keys] = m[keys]
            data_dict["image"] = None
            data_dict["height"], data_dict["width"] = img.shape[1:]
            metadata_clone.append(data_dict)
        metadata_clone = np.array(metadata_clone)
        return metadata_clone

    def log(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Runs evaluator and saves the prediction results.

        Returns:
            Dict[str, Any]: Prediction results as a dict
        """
        # Save predictions in coco format
        coco_instances_results = []

        # Prepare attack data
        adv_patch, patch_mask = None, None
        if self.use_attack:
            adv_patch, patch_mask = prep_adv_patch(
                img_size=self.img_size,
                adv_patch_path=self.adv_patch_path,
                attack_type=self.attack_type,
                synthetic=self.synthetic,
                obj_size=self.obj_size,
                interp=self.interp,
                device=self.device,
            )

        if self.synthetic:
            # Prepare evaluation with synthetic signs, syn_data: (syn_obj,
            # syn_obj_mask, obj_transforms, mask_transforms, syn_sign_class)
            syn_data = prep_synthetic_eval(
                syn_obj_path=self.syn_obj_path,
                img_size=self.img_size,
                obj_size=self.obj_size,
                transform_prob=1.0,
                interp=self.interp,
                device=self.device,
                **self.syn_transform_params,
            )
            # Append with synthetic sign class
            syn_obj, syn_obj_mask = syn_data[:2]
            syn_data = [*syn_data[2:], self.adv_sign_class_id]
            syn_scores = torch.zeros(
                (len(self.all_iou_thres), self.num_test), device=self.device
            )
            syn_matches = torch.zeros_like(syn_scores)

        total_num_images, total_num_patches, num_vis = 0, 0, 0
        self.evaluator.reset()

        for i, batch in tqdm(enumerate(self.dataloader)):

            if total_num_images >= self.num_test:
                break

            file_name: str = batch[0]["file_name"]
            filename: str = file_name.split("/")[-1]
            image_id: int = batch[0]["image_id"]
            h0, w0 = batch[0]["height"], batch[0]["width"]
            img_df: pd.DataFrame = self.df[self.df["filename"] == filename]
            is_included: bool = False

            if self.annotated_signs_only and img_df.empty:
                # Skip image if there's no annotation
                continue

            # Skip (or only run on) files listed in the txt file
            in_list = filename in self.skipped_filename_list
            if (in_list and not self.run_only_img_txt) or (
                not in_list and self.run_only_img_txt
            ):
                continue

            # Have to preprocess image here [0, 255] -> [-123.675, 151.470]
            # i.e., centered with mean [103.530, 116.280, 123.675], no std
            # normalization so scale is still 255
            # NOTE: this is done inside attack
            # images = self.model.preprocess_image(batch)

            # Image from dataloader is 'BGR' and has shape [C, H, W]
            images: torch.Tensor = batch[0]["image"].float().to(self.device)
            new_gt: Dict[str, Any] = batch[0]
            if self.input_format == "BGR":
                images = images.flip(0)
            perturbed_image: torch.Tensor = images.clone()
            _, h, w = images.shape

            # if self.fixed_input_size or w < self.img_size[1]:
            if self.fixed_input_size:
                # FIXME: This code does not work correctly because gt is not
                # adjusted in the same way (still requires padding).
                # Resize and pad perturbed_image to self.img_size preseving
                # aspect ratio. This also handles images whose width is the
                # shorter side in the varying input size case.
                if w < self.img_size[1]:
                    # If real width is smaller than desired one, then height
                    # must be longer than width so we scale down by height ratio
                    scale = self.img_size[0] / h
                else:
                    scale = 1
                resized_size = (int(h * scale), int(w * scale))
                perturbed_image = resize_and_center(
                    perturbed_image,
                    img_size=self.img_size,
                    obj_size=resized_size,
                    interp=self.interp,
                )
                h, w = self.img_size
                h_pad = h - resized_size[0]  # TODO: divided by 2?
                w_pad = w - resized_size[1]
            else:
                h_pad, w_pad = 0, 0

            # NOTE: w_pad, h_pad is not typo!
            img_dim_data = (h0, w0, h / h0, w / w0, w_pad, h_pad)

            if self.synthetic:
                # Attacking synthetic signs
                self.log(f"Attacking {file_name} ...")

                if not self.fixed_input_size:
                    # When image size is not fixed, other objects (sign, patch,
                    # masks) have to be resized instead.
                    cur_adv_patch, cur_patch_mask, cur_syn_obj, cur_obj_mask = [
                        _pad_or_crop_height(p, (h, w))
                        for p in [adv_patch, patch_mask, syn_obj, syn_obj_mask]
                    ]
                else:
                    # Otherwise, sizes should already match
                    cur_adv_patch, cur_patch_mask, cur_syn_obj, cur_obj_mask = (
                        adv_patch,
                        patch_mask,
                        syn_obj,
                        syn_obj_mask,
                    )

                # Apply synthetic sign and patch to image
                perturbed_image, new_gt = apply_synthetic_sign(
                    perturbed_image,
                    batch[0],
                    None,
                    cur_adv_patch,
                    cur_patch_mask,
                    cur_syn_obj,
                    cur_obj_mask,
                    *syn_data,
                    use_attack=self.use_attack,
                    return_target=True,
                    is_detectron=True,
                    other_sign_class=self.other_sign_class,
                )
                # Image is used as background only so we can include any of
                # them when evaluating synthetic signs.
                is_included = True
                total_num_patches += 1

            elif not img_df.empty:
                # Attacking real signs

                # Iterate through annotated objects in the current image
                for obj_idx, obj in img_df.iterrows():

                    obj_classname = obj["final_shape"]
                    obj_class_id = self.class_names.index(obj_classname)

                    # Skip if it is "other" class or not from desired class
                    if obj_class_id == self.other_sign_class or (
                        obj_class_id != self.adv_sign_class_id
                        and self.adv_sign_class_id != -1
                    ):
                        continue

                    is_included = True
                    total_num_patches += 1
                    if not self.use_attack:
                        continue

                    # Run attack for each sign to get a new `adv_patch`
                    if self.attack_type == "per-sign":
                        data = [obj_classname, obj, *img_dim_data]
                        attack_images = [[images, data, str(file_name)]]
                        cloned_metadata = self.clone_metadata(
                            [obj[0] for obj in attack_images], batch
                        )
                        with torch.enable_grad():
                            adv_patch = self.attack.attack_real(
                                attack_images,
                                patch_mask,
                                obj_class_id,
                                metadata=cloned_metadata,
                            )

                    # TODO: Should we put only one adversarial patch per image?
                    # i.e., attacking only one sign per image.
                    self.log(f"Attacking {file_name} on obj {obj_idx}...")

                    # Transform and apply patch on the image
                    img = perturbed_image.clone().to(self.device)
                    perturbed_image, _ = transform_and_apply_patch(
                        img,
                        adv_patch,
                        patch_mask,
                        obj_classname,
                        obj,
                        img_dim_data,
                        **self.transform_params,
                    )

            if not is_included:
                # Skip image without any adversarial patch when attacking
                continue

            # Perform inference on perturbed image
            # perturbed_image = self._post_process_image(perturbed_image)
            perturbed_image = coerce_rank(perturbed_image, 3)
            outputs = self.predict(perturbed_image)

            if self.synthetic:
                instances = outputs["instances"]
                # ious has shape [num_dts,]
                ious = pairwise_iou(
                    instances.pred_boxes,
                    new_gt["instances"]
                    .gt_boxes[-1]
                    .to(self.device),  # Last new gt bbox is synthetic sign
                )[:, 0]
                # Skip empty ious (no overlap)
                if len(ious) == 0:
                    continue
                # Save scores and gt-dt matches at each level of IoU thresholds
                # Find the match with highest IoU and has correct class
                ious *= instances.pred_classes == self.adv_sign_class_id
                matches = ious[None, :] >= self.all_iou_thres[:, None]
                # Zero out scores that have lowe IoU than threshold
                scores = instances.scores[None, :] * matches
                # Select matched dt with highest score
                idx_max_score = scores.argmax(-1)
                tmp_idx = torch.arange(10)
                syn_matches[:, total_num_images] = matches[
                    tmp_idx, idx_max_score
                ]
                syn_scores[:, total_num_images] = scores[tmp_idx, idx_max_score]
            else:
                new_outputs = deepcopy(outputs)
                new_instances = new_outputs["instances"]
                new_instances._image_size = (h0, w0)
                # Scale output to match original input size for evaluator
                h_ratio, w_ratio = h / h0, w / w0
                new_instances.pred_boxes.tensor[:, 0] /= h_ratio
                new_instances.pred_boxes.tensor[:, 1] /= w_ratio
                new_instances.pred_boxes.tensor[:, 2] /= h_ratio
                new_instances.pred_boxes.tensor[:, 3] /= w_ratio
                self.evaluator.process([new_gt], [new_outputs])

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs, image_id)
            coco_instances_results.extend(instance_dicts)
            total_num_images += 1

            if num_vis < self.num_vis and is_included:
                num_vis += 1
                self._visualize(i, batch[0], perturbed_image, outputs)

        if self.synthetic:
            metrics = {"bbox": {}}
            syn_scores = syn_scores.cpu().numpy()
            syn_matches = syn_matches.float().cpu().numpy()

            # Iterate over each IoU threshold
            # ap_t = np.zeros(len(self.all_iou_thres))
            # for t, (scores, matches) in enumerate(zip(syn_scores, syn_matches)):
            #     results = _compute_ap_recall(scores, matches, total_num_images)
            #     ap_t[t] = results["AP"]
            # metrics["bbox"]["syn_ap"] = ap_t.mean()
            # metrics["bbox"]["syn_ap50"] = ap_t[0]

            # Get detection for desired score and for all IoU thresholds
            detected = (syn_scores >= self.conf_thres) * syn_matches
            # Select desired IoU threshold
            tp = detected[self.iou_thres_idx].sum()
            fn = total_num_images - tp
            metrics["bbox"]["syn_total"] = total_num_images
            metrics["bbox"]["syn_tp"] = int(tp)
            metrics["bbox"]["syn_fn"] = int(fn)
            metrics["bbox"]["syn_tpr"] = tp / total_num_images
            metrics["bbox"]["syn_fnr"] = fn / total_num_images
            metrics["bbox"]["syn_scores"] = syn_scores
            metrics["bbox"]["syn_matches"] = syn_matches
        else:
            metrics = self.evaluator.evaluate()
            metrics["bbox"]["total_num_patches"] = total_num_patches

        return coco_instances_results, metrics

    def _visualize(
        self,
        index: int,
        input_dict: List[Dict[str, Any]],
        perturbed_image: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> None:
        """Visualize ground truth, clean and adversarial predictions."""
        # Visualize ground truth
        original_image: torch.Tensor = input_dict["image"].permute(1, 2, 0)
        original_image = original_image.numpy()[:, :, ::-1]
        visualizer = Visualizer(original_image, self.metadata, scale=0.5)
        if self.vis_show_bbox:
            vis_gt = visualizer.draw_dataset_dict(input_dict)
        else:
            vis_gt = visualizer.get_output()
        vis_gt.save(os.path.join(self.vis_save_dir, f"gt_{index}.jpg"))

        # Save adv predictions
        # Set confidence threshold
        # NOTE: output bbox follows original size instead of real input size
        instances = outputs["instances"]
        mask = instances.scores > self.vis_conf_thres
        instances = instances[mask]
        perturbed_image = perturbed_image.permute(1, 2, 0)
        perturbed_image = perturbed_image.cpu().numpy().astype(np.uint8)
        v = Visualizer(perturbed_image, self.metadata, scale=0.5)
        if self.vis_show_bbox:
            vis = v.draw_instance_predictions(instances.to("cpu")).get_image()
        else:
            vis = v.get_output().get_image()

        if self.use_attack:
            # Save original predictions
            outputs = self.predict(original_image)
            instances = outputs["instances"]
            mask = instances.scores > self.vis_conf_thres
            instances = instances[mask]
            v = Visualizer(original_image, self.metadata, scale=0.5)
            vis_og = v.draw_instance_predictions(
                instances.to("cpu")
            ).get_image()
            save_path = os.path.join(
                self.vis_save_dir, f"pred_clean_{index}.jpg"
            )
            cv2.imwrite(save_path, vis_og[:, :, ::-1])
            save_name = f"pred_adv_{index}.jpg"
        else:
            save_name = f"pred_clean_{index}.jpg"

        save_adv_path = os.path.join(self.vis_save_dir, save_name)
        cv2.imwrite(save_adv_path, vis[:, :, ::-1])
        self.log(f"Saved visualization to {save_adv_path}")

    def _create_instance_dicts(
        self, outputs: Dict[str, Any], image_id: int
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to coco predictions format.

        Args:
            outputs (Dict[str, Any]): Output dictionary from model output
            image_id (int): Image ID

        Returns:
            List[Dict[str, Any]]: List of per instance predictions
        """
        instance_dicts = []

        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes
        scores = instances.scores.cpu().numpy()

        # For each bounding box
        for i, box in enumerate(pred_boxes):
            i_dict = {
                "image_id": image_id,
                "category_id": int(pred_classes[i]),
                "bbox": [b.item() for b in box],
                "score": float(scores[i]),
            }
            instance_dicts.append(i_dict)

        return instance_dicts

    @torch.no_grad()
    def _post_process_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process image back to [0, 255] range, i.e. undo the normalization.

        Args:
            image: torch.Tensor [C, H, W]

        Returns:
            torch.Tensor [C, H, W]
        """
        image = image.detach()
        image = (image * self.model.pixel_std) + self.model.pixel_mean

        return torch.clamp(image, 0, 255)

    def predict(
        self,
        original_image: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """Simple inference on a single image.

        Args:
            original_image: Input image; could be a numpy array of shape
                (H, W, C) or torch.Tensor of shape (C, H, W) (in RGB order).

        Returns:
            predictions: the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if isinstance(original_image, np.ndarray):
                if self.input_format == "BGR":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = torch.from_numpy(
                    original_image.astype("float32").transpose(2, 0, 1)
                )
            else:
                # Torch image is already float and has shape [C, H, W]
                height, width = original_image.shape[1:]
                image = original_image
                if self.input_format == "BGR":
                    image = image.flip(0)

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
