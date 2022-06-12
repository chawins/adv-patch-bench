'''
This code is adapted from
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
'''
from email.mime import image
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from adv_patch_bench.attacks.utils import (apply_synthetic_sign, prep_attack,
                                           prep_synthetic_eval)
from adv_patch_bench.utils.image import pad_image, pad_to_size
from adv_patch_bench.transforms import transform_and_apply_patch
from adv_patch_bench.utils.detectron import build_evaluator
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from .rp2 import RP2AttackModule


class DAGAttacker:
    def __init__(
        self,
        cfg: CfgNode,
        args: Namespace,
        attack_config: Dict,
        model: torch.nn.Module,
        dataloader: Any,
        class_names: List,
    ) -> None:
        """Attack wrapper for detectron model"""
        self.args = args
        self.verbose = args.verbose
        self.debug = args.debug
        self.model = model
        self.num_vis = 100
        if isinstance(args.img_size, str):
            self.img_size = tuple([int(size) for size in args.img_size.split(',')])
        else:
            self.img_size = args.img_size
        assert isinstance(self.img_size, Tuple) and len(self.img_size) == 2

        # Modify config
        self.cfg = cfg.clone()  # cfg can be modified by model
        # TODO: To generate more dense proposals
        # self.cfg.MODEL.RPN.NMS_THRESH = nms_thresh
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR', 'Only allow BGR input format for now'

        self.data_loader = dataloader
        self.device = self.model.device
        self.n_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        self.evaluator = build_evaluator(cfg, cfg.DATASETS.TEST[0])

        # Attack params
        self.attack = RP2AttackModule(attack_config, model, None, None, None,
                                      rescaling=False, interp=args.interp,
                                      verbose=self.verbose, is_detectron=True)
        self.attack_type = args.attack_type
        self.use_attack = self.attack_type != 'none'
        self.adv_sign_class = args.obj_class
        self.df = pd.read_csv(args.tgt_csv_filepath)
        self.class_names = class_names
        self.synthetic = args.synthetic
        self.transform_params = {
            'transform_mode': args.transform_mode,
            'use_relight': not args.no_patch_relight,
            'interp': args.interp,
        }

        # Loading file names from the specified text file
        self.skipped_filename_list = []
        if args.img_txt_path != '':
            with open(args.img_txt_path, 'r') as f:
                self.skipped_filename_list = f.read().splitlines()

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(
        self,
        patch_mask: torch.Tensor,
        vis_save_dir: str = None,
        vis_conf_thresh: float = 0.5,
    ) -> List:
        """Runs the DAG algorithm and saves the prediction results.

        Parameters
        ----------
        vis_save_dir : str, optional
            Directory to save the visualized bbox prediction images
        vis_conf_thresh : float, optional
            Confidence threshold for visualized bbox predictions, by default 0.5

        Returns
        -------
        Dict[str, Any]
            Prediction results as a dict
        """
        # Save predictions in coco format
        coco_instances_results = []

        if self.synthetic:
            # Prepare evaluation with synthetic signs
            # syn_data: (syn_obj, syn_obj_mask, obj_transforms, mask_transforms, syn_sign_class)
            syn_data = prep_synthetic_eval(
                self.args, self.img_size, self.class_names, device=self.device)

        # Prepare attack data
        if self.use_attack:
            _, adv_patch, patch_mask, patch_loc = prep_attack(
                self.args, self.img_size, self.device)

        total_num_patches, num_vis = 0, 0
        self.evaluator.reset()

        for i, batch in tqdm(enumerate(self.data_loader)):
            if self.debug and i >= 100:
                break

            file_name = batch[0]['file_name']
            filename = file_name.split('/')[-1]
            # basename = os.path.basename(file_name)
            image_id = batch[0]['image_id']
            h0, w0 = batch[0]['height'], batch[0]['width']
            img_df = self.df[self.df['filename'] == filename]

            # Skip (or only run on) files listed in the txt file
            in_list = filename in self.skipped_filename_list
            if ((in_list and not self.args.run_only_img_txt) or
                    (not in_list and self.args.run_only_img_txt)):
                continue

            # Have to preprocess image here [0, 255] -> [-123.675, 151.470]
            # i.e., centered with mean [103.530, 116.280, 123.675], no std
            # normalization so scale is still 255
            # NOTE: this is done inside attack
            # images = self.model.preprocess_image(batch)

            # Image from dataloader is 'BGR' and has shape [C, H, W]
            images = batch[0]['image'].float().to(self.device)
            if self.input_format == 'BGR':
                images = images.flip(0)
            perturbed_image = images.clone()
            _, h, w = images.shape
            img_data = (h0, w0, h / h0, w / w0, 0, 0)

            is_included = False
            if self.synthetic:
                self.log(f'Attacking {file_name} ...')
                # Apply synthetic sign and patch to image
                images, bbox = pad_to_size(images, self.img_size)
                assert images.shape[-2:] == self.img_size
                perturbed_image = apply_synthetic_sign(
                    images,
                    None,
                    None,
                    adv_patch,
                    patch_mask,
                    *syn_data,
                    use_attack=self.use_attack,
                    return_target=False,
                )
                left, top, right, bot = bbox
                perturbed_image = perturbed_image[:, top:bot, left:right]
                is_included = True
            elif len(img_df) > 0:
                # Iterate through objects in the current image
                for _, obj in img_df.iterrows():
                    obj_classname = obj['final_shape']
                    obj_class = self.class_names.index(obj_classname)
                    if obj_class != self.adv_sign_class and self.adv_sign_class != -1:
                        # Skip if object is not from desired class
                        continue
                    is_included = True
                    if not self.use_attack:
                        continue

                    # Run attack for each sign to get a new `adv_patch`
                    if self.attack_type == 'per-sign':
                        data = [obj_classname, obj, *img_data]
                        attack_images = [[images, data, str(file_name)]]
                        with torch.enable_grad():
                            adv_patch = self.attack.attack_real(
                                attack_images, patch_mask, obj_class, metadata=batch)

                    # TODO: Should we put only one adversarial patch per image?
                    # i.e., attacking only one sign per image.
                    self.log(f'Attacking {file_name} ...')

                    # Transform and apply patch on the image. `im` has range [0, 255]
                    perturbed_image = transform_and_apply_patch(
                        perturbed_image, adv_patch.to(self.device),
                        patch_mask, patch_loc, obj_classname, obj, img_data,
                        **self.transform_params) * 255
                    total_num_patches += 1

            if not is_included:
                # Skip image without any adversarial patch when attacking
                continue

            # Perform inference on perturbed image
            # perturbed_image = self._post_process_image(perturbed_image)
            if perturbed_image.ndim == 4:
                # TODO: Remove batch dim
                perturbed_image = perturbed_image[0]

            outputs = self(perturbed_image)
            # Scale output to match original input size for evaluator
            new_outputs = deepcopy(outputs)
            new_outputs['instances']._image_size = (h0, w0)
            h_ratio, w_ratio = h / h0, w / w0
            new_outputs['instances'].pred_boxes.tensor[:, 0] /= h_ratio
            new_outputs['instances'].pred_boxes.tensor[:, 1] /= w_ratio
            new_outputs['instances'].pred_boxes.tensor[:, 2] /= h_ratio
            new_outputs['instances'].pred_boxes.tensor[:, 3] /= w_ratio
            self.evaluator.process(batch, [new_outputs])

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs, image_id)
            coco_instances_results.extend(instance_dicts)

            if vis_save_dir and num_vis < self.num_vis and is_included:
                # Visualize ground truth
                original_image = batch[0]['image'].permute(1, 2, 0).numpy()[:, :, ::-1]
                visualizer = Visualizer(original_image, self.metadata, scale=0.5)
                vis_gt = visualizer.draw_dataset_dict(batch[0])
                vis_gt.save(os.path.join(vis_save_dir, f'gt_{i}.jpg'))

                # Save adv predictions
                # Set confidence threshold
                # NOTE: output bbox follows original size instead of real input size
                instances = outputs['instances']
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]
                perturbed_image = perturbed_image.permute(1, 2, 0)
                perturbed_image = perturbed_image.cpu().numpy().astype(np.uint8)
                v = Visualizer(perturbed_image, self.metadata, scale=0.5)
                vis = v.draw_instance_predictions(instances.to('cpu')).get_image()

                if self.use_attack:
                    # Save original predictions
                    outputs = self(original_image)
                    instances = outputs["instances"]
                    mask = instances.scores > vis_conf_thresh
                    instances = instances[mask]
                    v = Visualizer(original_image, self.metadata, scale=0.5)
                    vis_og = v.draw_instance_predictions(instances.to('cpu')).get_image()
                    save_path = os.path.join(vis_save_dir, f'pred_clean_{i}.jpg')
                    cv2.imwrite(save_path, vis_og[:, :, ::-1])
                    save_name = f'pred_adv_{i}.jpg'
                else:
                    save_name = f'pred_clean_{i}.jpg'

                save_adv_path = os.path.join(vis_save_dir, save_name)
                cv2.imwrite(save_adv_path, vis[:, :, ::-1])
                self.log(f'Saved visualization to {vis_save_dir}')
                num_vis += 1

        metrics = self.evaluator.evaluate()
        return coco_instances_results, metrics

    def _create_instance_dicts(
        self, outputs: Dict[str, Any], image_id: int
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to coco predictions format

        Parameters
        ----------
        outputs : Dict[str, Any]
            Output dictionary from model output
        image_id : int

        Returns
        -------
        List[Dict[str, Any]]
            List of per instance predictions
        """
        instance_dicts = []

        instances = outputs['instances']
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

        Parameters
        ----------
        image : torch.Tensor
            [C, H, W]

        Returns
        -------
        torch.Tensor
            [C, H, W]
        """
        image = image.detach()
        image = (image * self.model.pixel_std) + self.model.pixel_mean

        return torch.clamp(image, 0, 255)

    def __call__(
        self,
        original_image: Union[np.ndarray, torch.Tensor],
    ) -> Dict:
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
                if self.input_format == 'BGR':
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                # image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            else:
                # Torch image is already float and has shape [C, H, W]
                height, width = original_image.shape[1:]
                image = original_image
                if self.input_format == 'BGR':
                    image = image.flip(0)

            inputs = {"image": image, "height": height, "width": width}
            # inputs = {"image": image, "height": size[0], "width": size[1]}
            predictions = self.model([inputs])[0]
            return predictions
