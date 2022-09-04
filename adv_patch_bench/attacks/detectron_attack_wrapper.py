"""
This code is adapted from
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
"""
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from adv_patch_bench.attacks.rp2.rp2_detectron import RP2AttackDetectron
from adv_patch_bench.attacks.utils import (
    apply_synthetic_sign,
    load_annotation_df,
    prep_adv_patch,
    prep_synthetic_eval,
)
from adv_patch_bench.transforms import transform_and_apply_patch
from adv_patch_bench.utils.detectron import build_evaluator
from adv_patch_bench.utils.image import coerce_rank
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from hparams import SAVE_DIR_DETECTRON
from tqdm import tqdm


class DetectronAttackWrapper:
    def __init__(
        self,
        cfg: CfgNode,
        args: Namespace,
        attack_config: Dict,
        model: torch.nn.Module,
        dataloader: Any,
        class_names: List[str],
    ) -> None:
        """Attack wrapper for detectron model.

        Args:
            cfg (CfgNode): Detectron model config.
            args (Namespace): All command line args.
            attack_config (Dict): Dictionary containing attack parameters.
            model (torch.nn.Module): Target model.
            dataloader (Any): Dataset to run attack on.
            class_names (List[str]): List of class names in string.
        """
        self.args = args
        self.verbose = args.verbose
        self.debug = args.debug
        self.model = model
        self.num_vis = 25
        if isinstance(args.img_size, str):
            self.img_size = tuple(
                [int(size) for size in args.img_size.split(",")]
            )
        else:
            self.img_size = args.img_size
        assert isinstance(self.img_size, Tuple) and len(self.img_size) == 2

        # Modify config
        self.cfg = cfg.clone()  # cfg can be modified by model
        # TODO: To generate more dense proposals
        # self.cfg.MODEL.RPN.NMS_THRESH = nms_thresh
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == "BGR", "Only allow BGR input format for now"

        self.data_loader = dataloader
        self.device = self.model.device
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        # Load annotation DataFrame. "Other" signs are discarded.
        self.df = load_annotation_df(args.tgt_csv_filepath)

        # Attack params
        if attack_config is None:
            self.attack = RP2AttackDetectron(
                attack_config,
                model,
                None,
                None,
                None,
                rescaling=False,
                interp=args.interp,
                verbose=self.verbose,
            )
        else:
            self.attack = None
        self.attack_type = args.attack_type
        self.use_attack = self.attack_type != "none"
        self.adv_sign_class = args.obj_class
        self.class_names = class_names
        self.other_sign_class = len(self.class_names) - 1
        self.synthetic = args.synthetic
        self.annotated_signs_only = args.annotated_signs_only
        self.transform_params = {
            "transform_mode": args.transform_mode,
            "use_relight": not args.no_patch_relight,
            "interp": args.interp,
        }
        self.evaluator = build_evaluator(
            cfg, cfg.DATASETS.TEST[0], synthetic=self.synthetic
        )
        if self.synthetic:
            # TODO: self.syn_sign_class is not really used now.
            # self.syn_sign_class = self.other_sign_class + 1
            # self.metadata.thing_classes.append('synthetic')
            self.syn_sign_class = self.other_sign_class

        # Loading file names from the specified text file
        self.skipped_filename_list = []
        if args.img_txt_path != "":
            img_txt_path = os.path.join(SAVE_DIR_DETECTRON, args.img_txt_path)
            with open(img_txt_path, "r") as f:
                self.skipped_filename_list = f.read().splitlines()

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

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(
        self,
        vis_save_dir: str = None,
        vis_conf_thresh: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Runs the DAG algorithm and saves the prediction results.

        Args:
            vis_save_dir : str, optional
                Directory to save the visualized bbox prediction images
            vis_conf_thresh : float, optional
                Confidence threshold for visualized bbox predictions, by
                default 0.5

        Returns:
            Dict[str, Any]: Prediction results as a dict
        """
        # Save predictions in coco format
        coco_instances_results = []

        # Prepare attack data
        if self.use_attack:
            adv_patch, patch_mask = prep_adv_patch(
                img_size=self.img_size,
                adv_patch_path=self.args.adv_patch_path,
                attack_type=self.attack_type,
                synthetic=self.synthetic,
                obj_size=self.args.obj_size,
                interp=self.args.interp,
                device=self.device,
            )

        total_num_images, total_num_patches, num_vis = 0, 0, 0
        syn_is_detected = []
        syn_tp, syn_total = 0, 0
        self.evaluator.reset()

        for i, batch in tqdm(enumerate(self.data_loader)):
            if self.debug and total_num_patches >= 20:
                break
            if total_num_images >= self.args.num_test:
                break

            file_name = batch[0]["file_name"]
            filename = file_name.split("/")[-1]
            image_id = batch[0]["image_id"]
            h0, w0 = batch[0]["height"], batch[0]["width"]
            img_df = self.df[self.df["filename"] == filename]

            # Skip (or only run on) files listed in the txt file
            in_list = filename in self.skipped_filename_list
            if (in_list and not self.args.run_only_img_txt) or (
                not in_list and self.args.run_only_img_txt
            ):
                continue

            # Have to preprocess image here [0, 255] -> [-123.675, 151.470]
            # i.e., centered with mean [103.530, 116.280, 123.675], no std
            # normalization so scale is still 255
            # NOTE: this is done inside attack
            # images = self.model.preprocess_image(batch)

            # Image from dataloader is 'BGR' and has shape [C, H, W]
            images = batch[0]["image"].float().to(self.device)
            if self.input_format == "BGR":
                images = images.flip(0)
            perturbed_image = images.clone()
            _, h, w = images.shape
            img_data = (h0, w0, h / h0, w / w0, 0, 0)

            is_included = False
            if self.synthetic:
                self.log(f"Attacking {file_name} ...")
                adv_patch, patch_mask = prep_adv_patch(
                    img_size=(h, w),
                    adv_patch_path=self.args.adv_patch_path,
                    attack_type=self.attack_type,
                    synthetic=self.synthetic,
                    obj_size=self.args.obj_size,
                    interp=self.args.interp,
                    device=self.device,
                )

                # Prepare evaluation with synthetic signs, syn_data: (syn_obj,
                # syn_obj_mask, obj_transforms, mask_transforms, syn_sign_class)
                syn_data = prep_synthetic_eval(
                    self.args.syn_obj_path,
                    syn_use_scale=self.args.syn_use_scale,
                    syn_use_colorjitter=self.args.syn_use_colorjitter,
                    img_size=(h, w),
                    objh_size=self.args.obj_size,
                    transform_prob=1.0,
                    interp=self.args.interp,
                    device=self.device,
                )

                # Append with synthetic sign class
                syn_data = list(syn_data)
                syn_data.append(self.adv_sign_class)

                # Apply synthetic sign and patch to image
                perturbed_image, new_gt = apply_synthetic_sign(
                    images,
                    batch[0],
                    None,
                    adv_patch,
                    patch_mask,
                    *syn_data,
                    use_attack=self.use_attack,
                    return_target=True,
                    is_detectron=True,
                    other_sign_class=self.other_sign_class,
                )
                is_included = True
                total_num_patches += 1

            elif len(img_df) > 0:

                new_gt = batch[0]
                # Iterate through annotated objects in the current image
                for obj_idx, obj in img_df.iterrows():

                    obj_classname = obj["final_shape"]
                    obj_class_id = self.class_names.index(obj_classname)

                    # Skip if it is "other" class or not from desired class
                    if obj_class_id == self.other_sign_class or (
                        obj_class_id != self.adv_sign_class
                        and self.adv_sign_class != -1
                    ):
                        continue

                    is_included = True
                    total_num_patches += 1
                    if not self.use_attack:
                        continue

                    # Run attack for each sign to get a new `adv_patch`
                    if self.attack_type == "per-sign":
                        data = [obj_classname, obj, *img_data]
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
                    adv_patch_clone = adv_patch.clone().to(self.device)
                    img = perturbed_image.clone().to(self.device)
                    perturbed_image, _ = transform_and_apply_patch(
                        img,
                        adv_patch_clone,
                        patch_mask,
                        obj_classname,
                        obj,
                        img_data,
                        **self.transform_params,
                    )

            elif not self.annotated_signs_only:
                # Include all images if annotated_signs_only is not set to True
                is_included = True

            if not is_included:
                # Skip image without any adversarial patch when attacking
                continue
            total_num_images += 1

            # Perform inference on perturbed image
            # perturbed_image = self._post_process_image(perturbed_image)
            perturbed_image = coerce_rank(perturbed_image, 3)
            outputs = self(perturbed_image)
            new_outputs = deepcopy(outputs)
            new_instances = new_outputs["instances"]
            new_instances._image_size = (h0, w0)
            # Set predicted class of any object that overlaps with the gt bbox
            # of the synthetic sign to a new syn_sign_class
            if self.synthetic:
                ious = pairwise_iou(
                    new_instances.pred_boxes,
                    new_gt["instances"].gt_boxes[-1].to(self.device),
                )[:, 0]
                # TODO: compute mAP
                idx = (
                    (ious >= 0.5)
                    & (new_instances.scores >= self.args.conf_thres)
                    & (new_instances.pred_classes == self.adv_sign_class)
                )
                is_detected = idx.any().item()
                syn_is_detected.append(is_detected)
                syn_tp += is_detected
                syn_total += 1
                # FIXME: this is set to "other"
                new_instances.pred_classes[idx] = self.syn_sign_class
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

            if vis_save_dir and num_vis < self.num_vis and is_included:
                # Visualize ground truth
                original_image = (
                    batch[0]["image"].permute(1, 2, 0).numpy()[:, :, ::-1]
                )
                visualizer = Visualizer(
                    original_image, self.metadata, scale=0.5
                )
                vis_gt = visualizer.draw_dataset_dict(batch[0])
                vis_gt.save(os.path.join(vis_save_dir, f"gt_{i}.jpg"))

                # Save adv predictions
                # Set confidence threshold
                # NOTE: output bbox follows original size instead of real input size
                instances = outputs["instances"]
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]
                perturbed_image = perturbed_image.permute(1, 2, 0)
                perturbed_image = perturbed_image.cpu().numpy().astype(np.uint8)
                v = Visualizer(perturbed_image, self.metadata, scale=0.5)
                vis = v.draw_instance_predictions(
                    instances.to("cpu")
                ).get_image()

                if self.use_attack:
                    # Save original predictions
                    outputs = self(original_image)
                    instances = outputs["instances"]
                    mask = instances.scores > vis_conf_thresh
                    instances = instances[mask]
                    v = Visualizer(original_image, self.metadata, scale=0.5)
                    vis_og = v.draw_instance_predictions(
                        instances.to("cpu")
                    ).get_image()
                    save_path = os.path.join(
                        vis_save_dir, f"pred_clean_{i}.jpg"
                    )
                    cv2.imwrite(save_path, vis_og[:, :, ::-1])
                    save_name = f"pred_adv_{i}.jpg"
                else:
                    save_name = f"pred_clean_{i}.jpg"

                save_adv_path = os.path.join(vis_save_dir, save_name)
                cv2.imwrite(save_adv_path, vis[:, :, ::-1])
                self.log(f"Saved visualization to {save_adv_path}")
                num_vis += 1

        metrics = self.evaluator.evaluate()
        metrics["bbox"]["total_num_patches"] = total_num_patches
        if self.synthetic:
            metrics["bbox"]["syn_tp"] = syn_tp
            metrics["bbox"]["syn_total"] = syn_total
            metrics["bbox"]["syn_is_detected"] = syn_is_detected
        return coco_instances_results, metrics

    def _create_instance_dicts(
        self, outputs: Dict[str, Any], image_id: int
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to coco predictions format

        Args:
            outputs : Dict[str, Any]
                Output dictionary from model output
            image_id : int

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
        """Process image back to [0, 255] range, i.e. undo the normalization

        Args:
            image : torch.Tensor [C, H, W]

        Returns:
            torch.Tensor [C, H, W]
        """
        image = image.detach()
        image = (image * self.model.pixel_std) + self.model.pixel_mean

        return torch.clamp(image, 0, 255)

    def __call__(
        self,
        original_image: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """Simple inference on a single image

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in RGB order).
                or (torch.Tensor): an image of shape (C, H, W) (in RGB order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if isinstance(original_image, np.ndarray):
                if self.input_format == "BGR":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                # image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(
                    original_image.astype("float32").transpose(2, 0, 1)
                )
            else:
                # Torch image is already float and has shape [C, H, W]
                height, width = original_image.shape[1:]
                image = original_image
                if self.input_format == "BGR":
                    image = image.flip(0)

            inputs = {"image": image, "height": height, "width": width}
            # inputs = {"image": image, "height": size[0], "width": size[1]}
            predictions = self.model([inputs])[0]
            return predictions
