import math
import random

import cv2
import numpy as np

from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import union_of_bboxes
from albumentations.augmentations.geometric import functional as FGeometric
from albumentations import RandomSizedBBoxSafeCrop
# from albumentations import functional as F
from albumentations.augmentations.crops.functional import random_crop, bbox_random_crop

class RandomSizedBBoxSafeCrop_v2(RandomSizedBBoxSafeCrop):
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        self.crop_params = {"h_start": h_start, "w_start": w_start, "crop_height": self.height, "crop_width": self.width}
        crop = random_crop(img, self.height, self.width, h_start, w_start)
        return FGeometric.resize(crop, self.height, self.width, interpolation)

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return bbox_random_crop(bbox, self.height, self.width, h_start, w_start, rows, cols)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes_v2(
            width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        )
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)

        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}


def union_of_bboxes_v2(height, width, bboxes, erosion_rate=0.0):
    """Calculate union of bounding boxes.

    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[1:5]
        w, h = x_max - x_min, y_max - y_min
        lim_x1, lim_y1 = x_min + erosion_rate * w, y_min + erosion_rate * h
        lim_x2, lim_y2 = x_max - erosion_rate * w, y_max - erosion_rate * h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2