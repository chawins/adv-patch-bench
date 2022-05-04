'''
This code is adapted from
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
'''
import json
import os
import random
from typing import Any, Callable, Dict, List, Tuple

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import (DatasetMapper, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.modeling import build_model
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.visualizer import Visualizer
from torch.nn import functional as F
from tqdm import tqdm

from .rp2 import RP2AttackModule
from detectron2.structures import BoxMode


class DAGAttacker:
    def __init__(
        self,
        cfg: CfgNode,
        args,
        attack_config: Dict,
        model,
        dataloader,
        # n_iter: int = 150,
        # gamma: float = 0.5,
        # nms_thresh: float = 0.9,
        # mapper: Callable = DatasetMapper,
        # verbose: bool = False
    ) -> None:
        """Implements the DAG algorithm

        Parameters
        ----------
        cfg : CfgNode
            Config object used to train the model
        n_iter : int, optional
            Number of iterations to run the algorithm on each image, by default 150
        gamma : float, optional
            Perturbation weight, by default 0.5
        nms_thresh : float, optional
            NMS threshold of RPN; higher it is, more dense set of proposals, by default 0.9
        mapper : Callable, optional
            Can specify own DatasetMapper logic, by default DatasetMapper
        """
        # self.n_iter = n_iter
        # self.gamma = gamma
        self.verbose = args.verbose
        self.debug = args.debug
        self.model = model
        self.num_vis = 30

        # Modify config
        self.cfg = cfg.clone()  # cfg can be modified by model
        # TODO: To generate more dense proposals
        # self.cfg.MODEL.RPN.NMS_THRESH = nms_thresh
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000

        # # Init model
        # self.model = build_model(self.cfg)

        # # Load weights
        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)

        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )
        self.input_format = cfg.INPUT.FORMAT
        # assert self.input_format in ["RGB", "BGR"], self.input_format

        # Init dataloader on test dataset
        # mapper = BenignMapper(cfg, is_train=False)
        # dataset_mapper = mapper(cfg, is_train=False)
        # self.data_loader = build_detection_test_loader(
        #     cfg, cfg.DATASETS.TEST[0], mapper=dataset_mapper
        # )
        self.data_loader = dataloader

        self.device = self.model.device
        self.n_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        # HACK Only specific for this dataset
        # self.metadata.thing_classes = ["box", "logo"]
        # self.contiguous_id_to_thing_id = {
        #     v: k for k, v in self.metadata.thing_dataset_id_to_contiguous_id.items()
        # }

        self.attack = RP2AttackModule(attack_config, model, None, None, None,
                                      rescaling=False, interp=args.interp,
                                      verbose=self.verbose)

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(
        self,
        obj_class: int,
        patch_mask: torch.Tensor,
        results_save_path: str = 'coco_instances_results.json',
        vis_save_dir: str = None,
        vis_conf_thresh: float = 0.5,
    ) -> List:
        """Runs the DAG algorithm and saves the prediction results.

        Parameters
        ----------
        results_save_path : str, optional
            Path to save the results JSON file, by default "coco_instances_results.json"
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

        for i, batch in tqdm(enumerate(self.data_loader)):
            original_image = batch[0]['image'].permute(1, 2, 0).numpy()
            file_name = batch[0]['file_name']
            basename = os.path.basename(file_name)
            image_id = batch[0]['image_id']

            # Peform DAG attack
            self.log(f'[{i}/{len(self.data_loader)}] Attacking {file_name} ...')
            # Have to preprocess image here [0, 255] -> [-123.675, 151.470]
            # i.e., centered with mean [103.530, 116.280, 123.675], no std
            # normalization so scale is still 255
            # images = self.model.preprocess_image(batch)
            if vis_save_dir and i < self.num_vis:
                visualizer = Visualizer(original_image[:, :, ::-1], self.metadata, scale=0.5)
                vis_gt = visualizer.draw_dataset_dict(batch[0])
                vis_gt.save(os.path.join(vis_save_dir, f'gt_{i}.jpg'))

            images = batch[0]['image'].float().to(self.device) / 255

            # TODO
            # h0, w0, h_ratio, w_ratio, w_pad, h_pad
            img_data = (h0, w0, h_ratio, w_ratio, 0, 0)
            predicted_class = row['final_shape']
            data = [predicted_class, row, *img_data]
            attack_images = [[im[image_i], data, str(filename)]]
            perturbed_image = self.attack.transform_and_attack(
                images, patch_mask, obj_class)

            # Perform inference on perturbed image
            perturbed_image = self._post_process_image(perturbed_image)
            perturbed_image = (
                perturbed_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
            outputs = self(perturbed_image)

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs, image_id)
            coco_instances_results.extend(instance_dicts)

            if vis_save_dir and i < self.num_vis:
                # Save adv predictions
                # Set confidence threshold
                instances = outputs['instances']
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]

                v = Visualizer(perturbed_image[:, :, ::-1], self.metadata, scale=0.5)
                vis_adv = v.draw_instance_predictions(instances.to('cpu')).get_image()

                # Save original predictions
                outputs = self(original_image)
                instances = outputs["instances"]
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]

                v = Visualizer(original_image[:, :, ::-1], self.metadata, scale=0.5)
                vis_og = v.draw_instance_predictions(instances.to('cpu')).get_image()

                # Save side-by-side
                # concat = np.concatenate((vis_og, vis_adv), axis=1)

                # save_path = os.path.join("saved/adv", basename)
                # cv2.imwrite(save_path, concat[:, :, ::-1])

                save_path = os.path.join(vis_save_dir, f'pred_clean_{i}.jpg')
                save_adv_path = os.path.join(vis_save_dir, f'pred_adv_{i}.jpg')

                cv2.imwrite(save_path, vis_og[:, :, ::-1])
                cv2.imwrite(save_adv_path, vis_adv[:, :, ::-1])
                self.log(f'Saved visualization to {save_path}')

        # Save predictions as COCO results json format
        with open(results_save_path, 'w') as f:
            json.dump(coco_instances_results, f)

        return coco_instances_results

    def attack_image(self, batched_inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """Attack an image from the test dataloader

        Parameters
        ----------
        batched_inputs : List[Dict[str, Any]]
            A list containing a single element, the dataset_dict from the test dataloader

        Returns
        -------
        torch.Tensor
            [C, H, W], Perturbed image
        """
        images = self.model.preprocess_image(batched_inputs)

        instances = batched_inputs[0]["instances"]
        # If no ground truth annotations, no choice but to skip
        if len(instances.gt_boxes) == 0:
            return self._post_process_image(images.tensor[0])

        # Acquire targets and corresponding labels to attack
        # FIXME What if no target_boxes?
        target_boxes, target_labels = self._get_targets(batched_inputs)

        # Record gradients for image
        images.tensor.requires_grad = True

        # if len(target_boxes) == 0:
        #     return self._post_process_image(images.tensor[0])

        # Start DAG
        for i in range(self.n_iter):
            # Get features
            features = self.model.backbone(images.tensor)

            # Get classification logits
            logits, _ = self._get_roi_heads_predictions(features, target_boxes)

            # FIXME
            if len(logits) == 0:
                break

            # Update active target set,
            # i.e. filter for correctly predicted targets;
            # only attack targets that are still correctly predicted so far
            # FIXME
            active_cond = logits.argmax(dim=1) == target_labels
            # active_cond = logits.argmax(dim=1) != adv_labels

            target_boxes = target_boxes[active_cond]
            logits = logits[active_cond]
            target_labels = target_labels[active_cond]
            # adv_labels = adv_labels[active_cond]

            # If active set is empty, end algo;
            # All targets are already wrongly predicted
            if len(target_boxes) == 0:
                break

            # Compute total loss
            # FIXME Use before or after softmax?
            target_loss = F.cross_entropy(logits, target_labels, reduction="sum")
            adv_loss = F.cross_entropy(logits, adv_labels, reduction="sum")
            # Make every target incorrectly predicted as the adversarial label
            total_loss = target_loss - adv_loss

            # Backprop and compute gradient wrt image
            total_loss.backward()
            image_grad = images.tensor.grad.detach()

            # Apply perturbation on image
            with torch.no_grad():
                # Normalize grad
                image_perturb = (
                    self.gamma / image_grad.norm(float("inf"))
                ) * image_grad
                images.tensor += image_perturb

            # Zero gradients
            image_grad.zero_()
            self.model.zero_grad()

        self.log(f"Done with attack. Total Iterations {i}")

        return self._post_process_image(images.tensor[0])

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

        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes
        scores = instances.scores.cpu().numpy()

        # For each bounding box
        for i, box in enumerate(pred_boxes):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width = x2 - x1
            height = y2 - y1

            # HACK Only specific for this dataset
            category_id = int(pred_classes[i] + 1)
            # category_id = self.contiguous_id_to_thing_id[pred_classes[i]]
            score = float(scores[i])

            i_dict = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": score,
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

    def __call__(self, original_image: np.ndarray) -> Dict:
        """Simple inference on a single image

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
