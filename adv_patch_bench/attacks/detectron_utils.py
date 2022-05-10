from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, pairwise_iou


# @torch.no_grad()
def get_targets(
    model: torch.nn.Module,
    inputs: List[Dict[Any, Any]],
    device: str = 'cuda',
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
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
    images = model.preprocess_image(inputs)

    # Get features
    features = model.backbone(images.tensor)

    # Get bounding box proposals. For API, see
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L431
    proposals, _ = model.proposal_generator(images, features, None)
    proposal_boxes = [x.proposal_boxes for x in proposals]

    # Get proposal boxes' classification scores
    predictions = get_roi_heads_predictions(model, features, proposal_boxes)
    # Scores (softmaxed) for a single image, [n_proposals, n_classes + 1]
    # scores = model.roi_heads.box_predictor.predict_probs(
    #     predictions, proposals
    # )[0]
    # Instead, we want to get logit scores without softmax. For API, see
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L547
    class_logits, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    class_logits = class_logits.split(num_inst_per_image, dim=0)

    # NOTE: class_logits dim [[1000, num_classes + 1], ...]

    gt_boxes = [i['instances'].gt_boxes.to(device) for i in inputs]
    gt_classes = [i['instances'].gt_classes for i in inputs]
    objectness_logits = [x.objectness_logits for x in proposals]

    return filter_positive_proposals(
        proposal_boxes, class_logits, gt_boxes, gt_classes, objectness_logits,
        device=device, iou_thres=iou_thres, score_thres=score_thres)


def get_roi_heads_predictions(
    model,
    features: Dict[str, torch.Tensor],
    proposal_boxes: List[Boxes],
) -> Tuple[torch.Tensor, torch.Tensor]:
    roi_heads = model.roi_heads
    features = [features[f] for f in roi_heads.box_in_features]
    # Defn: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/poolers.py#L205
    # Usage: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/roi_heads.py#L780
    box_features = roi_heads.box_pooler(features, proposal_boxes)
    box_features = roi_heads.box_head(box_features)
    logits, proposal_deltas = roi_heads.box_predictor(box_features)
    del box_features

    return logits, proposal_deltas


# @torch.no_grad()
def filter_positive_proposals(
    proposal_boxes: List[Boxes],
    class_logits: List[torch.Tensor],
    gt_boxes: List[Boxes],
    gt_classes: List[torch.Tensor],
    objectness_logits: List[torch.Tensor],
    device: str = 'cuda',
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
) -> Tuple[Boxes, torch.Tensor, torch.Tensor]:

    outputs = [[], [], [], []]
    for inpt in zip(proposal_boxes, class_logits, gt_boxes, gt_classes, objectness_logits):
        out = filter_positive_proposals_single(
            *inpt,
            device=device,
            iou_thres=iou_thres,
            score_thres=score_thres,
        )
        for i in range(4):
            outputs[i].append(out[i])
    return outputs


def filter_positive_proposals_single(
    proposal_boxes: Boxes,
    class_logits: torch.Tensor,
    gt_boxes: Boxes,
    gt_classes: torch.Tensor,
    objectness_logits: torch.Tensor,
    device: str = 'cuda',
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
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

    # Filter for IoUs > iou_thres
    iou_cond = paired_ious > iou_thres

    # Filter for score of proposal > score_thres
    # Get class of paired gt_box
    gt_classes_repeat = gt_classes.repeat(n_proposals, 1)
    idx = torch.arange(n_proposals)
    paired_gt_classes = gt_classes_repeat[idx, paired_gt_idx]
    # Get scores of corresponding class
    scores = F.softmax(class_logits, dim=-1)
    paired_scores = scores[idx, paired_gt_classes]
    score_cond = paired_scores > score_thres

    # Filter for positive proposals and their corresponding gt labels
    cond = iou_cond & score_cond

    return (proposal_boxes[cond], paired_gt_classes[cond].to(device),
            class_logits[cond], objectness_logits[cond])
