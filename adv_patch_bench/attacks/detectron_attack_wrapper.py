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
from adv_patch_bench.utils.image import coerce_rank, resize_and_center
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from hparams import SAVE_DIR_DETECTRON
from tqdm import tqdm


def _pad_or_crop_height(
    img: torch.Tensor, size: Tuple[int, int]
) -> torch.Tensor:
    if img is None:
        return None
    cur_h, cur_w = img.shape[-2:]
    h, w = size
    assert (
        w == cur_w
    ), f"Width should already match ({img.shape[-2:]} vs {size})!"
    if h > cur_h:
        # Pad to adv_patch/patch_mask to match cur image height
        img = resize_and_center(img, img_size=size)
    elif h < cur_h:
        # Crop adv_patch/patch_mask to match cur image height
        top_crop = (cur_h - h) // 2
        bot_crop = cur_h - h - top_crop
        img = img[..., top_crop:-bot_crop, :]
    assert img.shape[-2:] == size
    return img


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """
    (DEPRECATED) This curve tracing method has some quirks that do not appear
    when only unique confidence thresholds are used (i.e. Scikit-learn's
    implementation), however, in order to be consistent, the COCO's method is
    reproduced.
    """

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }


class DetectronAttackWrapper:
    def __init__(
        self,
        cfg: CfgNode,
        args: Namespace,
        attack_config: Dict,
        model: torch.nn.Module,
        dataloader: Any,
        class_names: List[str],
        iou_thres: float = 0.5,
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

        self.num_test = args.num_test if not self.debug else 20
        self.data_loader = dataloader
        self.device = self.model.device
        self.conf_thres = args.conf_thres
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        self.interp = args.interp

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
                interp=self.interp,
                verbose=self.verbose,
            )
        else:
            self.attack = None
        self.attack_type = args.attack_type
        self.use_attack = self.attack_type != "none"
        self.adv_sign_class_id = args.obj_class
        self.class_names = class_names
        self.other_sign_class = len(self.class_names) - 1
        self.synthetic = args.synthetic
        self.annotated_signs_only = args.annotated_signs_only
        self.transform_params = {
            "transform_mode": args.transform_mode,
            "use_relight": not args.no_patch_relight,
            "interp": self.interp,
        }
        self.fixed_input_size = False  # TODO: Make this an option

        if self.synthetic:
            # TODO: (DEPRECATED)
            # self.syn_sign_class = self.other_sign_class + 1
            # self.metadata.thing_classes.append("synthetic")
            # self.syn_sign_class = self.other_sign_class

            # Compute obj_size
            obj_size = args.obj_size
            obj_class_name = class_names[self.adv_sign_class_id]
            obj_class_name_list = obj_class_name.split("-")
            sign_width_in_mm = float(obj_class_name_list[1])
            if len(obj_class_name_list) == 3 and sign_width_in_mm != 0:
                sign_height_in_mm = float(obj_class_name_list[2])
                hw_ratio = sign_height_in_mm / sign_width_in_mm
            else:
                hw_ratio = 1
            if isinstance(obj_size, int):
                obj_size = (round(obj_size * hw_ratio), obj_size)

            assert isinstance(obj_size, tuple) and all(
                [isinstance(s, int) for s in obj_size]
            ), f"obj_size is {obj_size}. It must be int or tuple of two ints!"
            self.obj_size = obj_size
        else:
            self.obj_size = None

        # Loading file names from the specified text file
        self.skipped_filename_list = []
        if args.img_txt_path != "":
            img_txt_path = os.path.join(SAVE_DIR_DETECTRON, args.img_txt_path)
            with open(img_txt_path, "r") as f:
                self.skipped_filename_list = f.read().splitlines()

        self.evaluator = build_evaluator(
            cfg, cfg.DATASETS.TEST[0], synthetic=self.synthetic
        )

        # Set up list of IoU thresholds to consider
        all_iou_thres = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.iou_thres_idx = int(np.where(all_iou_thres == iou_thres)[0])
        self.all_iou_thres = torch.from_numpy(all_iou_thres).to(self.device)

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
        adv_patch, patch_mask = None, None
        if self.use_attack:
            adv_patch, patch_mask = prep_adv_patch(
                img_size=self.img_size,
                adv_patch_path=self.args.adv_patch_path,
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
                self.args.syn_obj_path,
                syn_use_scale=self.args.syn_use_scale,
                syn_use_colorjitter=self.args.syn_use_colorjitter,
                img_size=self.img_size,
                obj_size=self.obj_size,
                transform_prob=1.0,
                interp=self.interp,
                device=self.device,
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

        for i, batch in tqdm(enumerate(self.data_loader)):

            if total_num_images >= self.num_test:
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
            new_gt = batch[0]
            if self.input_format == "BGR":
                images = images.flip(0)
            perturbed_image = images.clone()
            _, h, w = images.shape

            if self.fixed_input_size or w < self.img_size[1]:
                # TODO: This code does not work correctly because gt is not
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

            # Include all images if annotated_signs_only is False
            is_included = not self.annotated_signs_only

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

            elif len(img_df) > 0:
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
            outputs = self(perturbed_image, (h0, w0))

            if self.synthetic:
                instances = outputs["instances"]
                ious = pairwise_iou(
                    instances.pred_boxes,
                    new_gt["instances"].gt_boxes[-1].to(self.device),
                )[:, 0]
                # Skip empty ious (no overlap)
                if len(ious) == 0:
                    continue
                # Save scores and gt-dt matches at each level of IoU thresholds
                # Find the match with highest IoU and has correct class
                ious *= instances.pred_classes == self.adv_sign_class_id
                max_idx = ious.argmax()
                # Compute matches at different values of IoU threshold
                matches = ious[max_idx] >= self.all_iou_thres
                syn_matches[:, total_num_images] = matches
                scores = instances.scores[max_idx] * matches
                syn_scores[:, total_num_images] = scores
            else:
                new_outputs = deepcopy(outputs)
                # new_instances = new_outputs["instances"]
                # new_instances._image_size = (h0, w0)
                # # Scale output to match original input size for evaluator
                # h_ratio, w_ratio = h / h0, w / w0
                # new_instances.pred_boxes.tensor[:, 0] /= h_ratio
                # new_instances.pred_boxes.tensor[:, 1] /= w_ratio
                # new_instances.pred_boxes.tensor[:, 2] /= h_ratio
                # new_instances.pred_boxes.tensor[:, 3] /= w_ratio
                self.evaluator.process([new_gt], [new_outputs])

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs, image_id)
            coco_instances_results.extend(instance_dicts)
            total_num_images += 1

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
                    outputs = self(original_image, (h0, w0))  # FIXME
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

    def __call__(
        self,
        original_image: Union[np.ndarray, torch.Tensor],
        height_width,
    ) -> Dict[str, Any]:
        """Simple inference on a single image

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in RGB order).
                or (torch.Tensor): an image of shape (C, H, W) (in RGB order).

        Returns:
            predictions (dict): the output of the model for one image only.
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

            # inputs = {"image": image, "height": size[0], "width": size[1]}
            height, width = height_width
            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions
