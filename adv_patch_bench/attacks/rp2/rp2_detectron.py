from copy import deepcopy

import torch
import torch.nn.functional as F
from adv_patch_bench.attacks.detectron_utils import get_targets
from adv_patch_bench.utils.image import mask_to_box
from detectron2.structures import Boxes, Instances

from .rp2_base import RP2AttackModule

EPS = 1e-6


class RP2AttackDetectron(RP2AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super(RP2AttackDetectron, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs)

        detectron_config = attack_config['detectron']
        self.detectron_obj_const = detectron_config['obj_loss_const']
        self.detectron_iou_thres = detectron_config['iou_thres']

        # self.cfg.MODEL.RPN.NMS_THRESH = nms_thresh
        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000
        self.nms_thresh_orig = deepcopy(
            core_model.proposal_generator.nms_thresh)
        self.post_nms_topk_orig = deepcopy(
            core_model.proposal_generator.post_nms_topk)
        # self.nms_thresh = 0.9
        # self.post_nms_topk = {True: 5000, False: 5000}
        self.nms_thresh = self.nms_thresh_orig
        self.post_nms_topk = self.post_nms_topk_orig

    def _on_enter_attack(self, **kwargs):
        self.is_training = self.core_model.training
        self.core_model.eval()
        self.core_model.proposal_generator.nms_thresh = self.nms_thresh
        self.core_model.proposal_generator.post_nms_topk = self.post_nms_topk

    def _on_exit_attack(self, **kwargs):
        self.core_model.train(self.is_training)
        self.core_model.proposal_generator.nms_thresh = self.nms_thresh_orig
        self.core_model.proposal_generator.post_nms_topk = self.post_nms_topk_orig

    def _on_syn_attack_step(self, metadata, o_mask, bg_idx, obj_class, **kwargs):
        # Update metada with location of transformed synthetic sign
        for i in range(self.num_eot):
            m = metadata[bg_idx[i]]
            instances = m['instances']
            new_instances = Instances(instances.image_size)
            # Turn object mask to gt_boxes
            o_ymin, o_xmin, o_height, o_width = mask_to_box(o_mask[i])
            box = torch.tensor([[o_xmin, o_ymin, o_xmin + o_width, o_ymin + o_height]])
            new_instances.gt_boxes = Boxes(box)
            new_instances.gt_classes = torch.tensor([[obj_class]])
            m['instances'] = new_instances
        return metadata

    def _loss_func(self, adv_img, obj_class, metadata):
        """Compute loss for Faster R-CNN models"""
        for i, m in enumerate(metadata):
            # Flip image from RGB to BGR
            m['image'] = adv_img[i].flip(0) * 255
        # NOTE: IoU threshold for ROI is 0.5 and for RPN is 0.7
        _, target_labels, target_logits, obj_logits = get_targets(
            self.core_model, metadata, device=self.core_model.device,
            iou_thres=self.detectron_iou_thres, score_thres=self.min_conf,
            use_correct_only=False)

        # DEBUG
        # import cv2
        # from detectron2.utils.visualizer import Visualizer
        # from detectron2.data import MetadataCatalog
        # with torch.no_grad():
        #     idx = 0
        #     metadata[idx]['height'], metadata[idx]['width'] = adv_img.shape[2:]
        #     outputs = self.core_model(metadata)[idx]
        #     instances = outputs["instances"]
        #     mask = instances.scores > 0.5
        #     instances = instances[mask]
        #     self.metadata = MetadataCatalog.get('mapillary_combined')
        #     img = metadata[idx]['image'].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        #     v = Visualizer(img, self.metadata, scale=0.5)
        #     vis_og = v.draw_instance_predictions(instances.to('cpu')).get_image()
        #     cv2.imwrite('temp_pred.png', vis_og[:, :, ::-1])
        #     metadata[idx]['annotations'] = [{
        #         'bbox': metadata[idx]['instances'].gt_boxes.tensor[0].tolist(),
        #         'category_id': metadata[idx]['instances'].gt_classes.item(),
        #         'bbox_mode': metadata[idx]['annotations'][0]['bbox_mode'],
        #     }]
        #     vis_gt = v.draw_dataset_dict(metadata[0]).get_image()
        #     cv2.imwrite('temp_gt.png', vis_gt[:, :, ::-1])
        #     print('ok')
        # import pdb
        # pdb.set_trace()

        # Loop through each EoT image
        loss = 0
        for tgt_lb, tgt_log, obj_log in zip(target_labels, target_logits, obj_logits):
            # Filter obj_class
            if 'obj_class_only' in self.attack_mode:
                idx = obj_class == tgt_lb
                tgt_lb, tgt_log, obj_log = tgt_lb[idx], tgt_log[idx], obj_log[idx]
            else:
                tgt_lb = torch.zeros_like(tgt_lb) + obj_class
            # If there's no matched gt/prediction, then attack already succeeds.
            # TODO: This has to be changed for appearing or misclassification attacks.
            target_loss, obj_loss = 0, 0
            if len(tgt_log) > 0 and len(tgt_lb) > 0:
                # Ignore the background class on tgt_log
                target_loss = F.cross_entropy(tgt_log, tgt_lb, reduction='sum')
            if len(obj_logits) > 0 and self.detectron_obj_const != 0:
                obj_lb = torch.ones_like(obj_log)
                obj_loss = F.binary_cross_entropy_with_logits(obj_log, obj_lb,
                                                              reduction='sum')
            loss += target_loss + self.detectron_obj_const * obj_loss
        return -loss
