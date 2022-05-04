from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.functional as F
from detectron2.structures import Boxes, pairwise_iou


@torch.no_grad()
def get_targets(
    model,
    batched_inputs: List[Dict[Any, Any]],
    device: str = 'cuda',
) -> Tuple[Boxes, torch.Tensor]:
    """Select a set of initial targets for the DAG algo.

    Parameters
    ----------
    batched_inputs : List[Dict[Any]]
        A list containing a single dataset_dict, transformed by a DatasetMapper.

    Returns
    -------
    Tuple[Boxes, torch.Tensor]
        target_boxes, target_labels
    """
    instances = batched_inputs[0]["instances"]
    gt_boxes = instances.gt_boxes.to(device)
    gt_classes = instances.gt_classes

    images = model.preprocess_image(batched_inputs)

    # Get features
    features = model.backbone(images.tensor)

    # Get bounding box proposals
    # For API, see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L431
    proposals, _ = model.proposal_generator(images, features, None)
    proposal_boxes = proposals[0].proposal_boxes
    objectness_logits = proposals[0].objectness_logits

    # Get proposal boxes' classification scores
    predictions = get_roi_heads_predictions(features, proposal_boxes)
    # Scores (softmaxed) for a single image, [n_proposals, n_classes + 1]
    # For API, see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L547
    # scores = model.roi_heads.box_predictor.predict_probs(
    #     predictions, proposals
    # )[0]
    # Get logit scores without softmax
    class_logits, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    class_logits = class_logits.split(num_inst_per_image, dim=0)

    # TODO: check class_logits dim
    import pdb
    pdb.set_trace()

    return filter_positive_proposals(
        proposal_boxes, class_logits, gt_boxes, gt_classes, objectness_logits,
        device=device)


def get_roi_heads_predictions(
    model,
    features: Dict[str, torch.Tensor],
    proposal_boxes: Boxes,
) -> Tuple[torch.Tensor, torch.Tensor]:
    roi_heads = model.roi_heads
    features = [features[f] for f in roi_heads.box_in_features]
    box_features = roi_heads.box_pooler(features, [proposal_boxes])
    box_features = roi_heads.box_head(box_features)

    logits, proposal_deltas = roi_heads.box_predictor(box_features)
    del box_features

    return logits, proposal_deltas


def filter_positive_proposals(
    proposal_boxes: Boxes,
    scores: torch.Tensor,
    gt_boxes: Boxes,
    gt_classes: torch.Tensor,
    objectness_logits: torch.Tensor,
    device: str = 'cuda',
) -> Tuple[Boxes, torch.Tensor, torch.Tensor]:
    """Filter for desired targets for the DAG algo

    Parameters
    ----------
    proposal_boxes : Boxes
        Proposal boxes directly from RPN
    scores : torch.Tensor
        Softmaxed scores for each proposal box
    gt_boxes : Boxes
        Ground truth boxes
    gt_classes : torch.Tensor
        Ground truth classes

    Returns
    -------
    Tuple[Boxes, torch.Tensor, torch.Tensor]
        filtered_target_boxes, corresponding_class_labels, corresponding_scores
    """
    n_proposals = len(proposal_boxes)

    proposal_gt_ious = pairwise_iou(proposal_boxes, gt_boxes)

    # For each proposal_box, pair with a gt_box, i.e. find gt_box with highest IoU
    # IoU with paired gt_box, idx of paired gt_box
    paired_ious, paired_gt_idx = proposal_gt_ious.max(dim=1)

    # Filter for IoUs > 0.1
    iou_cond = paired_ious > 0.1

    # Filter for score of proposal > 0.1
    # Get class of paired gt_box
    gt_classes_repeat = gt_classes.repeat(n_proposals, 1)
    paired_gt_classes = gt_classes_repeat[torch.arange(n_proposals), paired_gt_idx]
    # Get scores of corresponding class
    paired_scores = scores[torch.arange(n_proposals), paired_gt_classes]
    score_cond = paired_scores > 0.1

    # Filter for positive proposals and their corresponding gt labels
    cond = iou_cond & score_cond

    return (proposal_boxes[cond], paired_gt_classes[cond].to(device),
            paired_scores[cond], objectness_logits[cond])
