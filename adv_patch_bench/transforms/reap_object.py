"""Implement REAP patch rendering for each object."""

import ast
from typing import Any, Callable, Tuple

import adv_patch_bench.utils.image as img_util
import cv2
import kornia.geometry.transform as kornia_tf
import numpy as np
import pandas as pd
import torch
from adv_patch_bench.transforms import geometric_tf, render_object
from adv_patch_bench.utils.types import (
    ImageTensor,
    ImageTensorRGBA,
    TransformFn,
    Target,
    BatchImageTensorGeneric,
    BatchImageTensorRGBA,
)

_VALID_TRANSFORM_MODE = ("perspective", "translate_scale")


class ReapObject(render_object.RenderObject):
    """Object wrapper using REAP benchmark."""

    def __init__(
        self,
        patch_transform_mode: str = "perspective",
        use_patch_relight: bool = True,
        **kwargs,
    ) -> None:
        """Initialize ReapObject.

        Args:
            patch_transform_mode: Type of geometric transform functions to use.
                Defaults to "perspective".
            use_patch_relight: Whether to apply relighting transform to
                adversarial patch. Defaults to True.

        Raises:
            NotImplementedError: Invalid transform mode.
        """
        super().__init__(**kwargs)

        if patch_transform_mode not in _VALID_TRANSFORM_MODE:
            raise NotImplementedError(
                f"transform_mode {patch_transform_mode} is not implemented. "
                f"Only supports {_VALID_TRANSFORM_MODE}!"
            )
        self.patch_transform_mode: str = patch_transform_mode

        # Get REAP relighting transform params
        if use_patch_relight:
            alpha = torch.tensor(self.obj_df["alpha"], device=self._device)
            beta = torch.tensor(self.obj_df["beta"], device=self._device)
        else:
            alpha = torch.tensor(1.0, device=self._device)
            beta = torch.tensor(0.0, device=self._device)
        self.alpha: torch.Tensor = img_util.coerce_rank(alpha, 3)
        self.beta: torch.Tensor = img_util.coerce_rank(beta, 3)

        # Get REAP geometric transform params
        tf_data = self._get_reap_transforms(self.obj_df)
        self.transform_mat = tf_data[1].to(self._device)
        self.transform_fn: TransformFn = self._wrap_transform_fn(tf_data[0])

    def _wrap_transform_fn(
        self, transform_fn: Callable[..., BatchImageTensorGeneric]
    ) -> TransformFn:
        """Wrap kornia transform function to avoid passing arguments around.

        Args:
            transform_fn: kornia transform function to wrap.

        Returns:
            Wrapped transform function that only takes image as input.
        """

        def wrapper_tf_fn(
            x: BatchImageTensorGeneric,
        ) -> BatchImageTensorGeneric:
            return transform_fn(
                x,
                self.transform_mat,
                self.img_size,
                mode=self._interp,
                padding_mode="zeros",
            )

        return wrapper_tf_fn

    def _get_reap_transforms(
        self, df_row: pd.DataFrame
    ) -> Tuple[Callable[..., Any], torch.Tensor]:
        """Get transformation matrix and parameters.

        Returns:
            Tuple of (Transform function, transformation matrix, target points).
        """
        h_ratio, w_ratio = self.img_hw_ratio
        h0, w0 = self.img_size_orig
        h_pad, w_pad = self.img_pad_size

        # Get target points from dataframe
        # TODO: Fix this after unifying csv
        if not pd.isna(df_row["points"]):
            tgt = np.array(ast.literal_eval(df_row["points"]), dtype=np.float32)
            tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
            tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad
            # print('not polygon')
        else:
            tgt = (
                df_row["tgt"]
                if pd.isna(df_row["tgt_polygon"])
                else df_row["tgt_polygon"]
            )
            tgt = np.array(ast.literal_eval(tgt), dtype=np.float32)
            # print('polygon')

            offset_x_ratio = df_row["xmin_ratio"]
            offset_y_ratio = df_row["ymin_ratio"]
            # Have to correct for the padding when df is saved
            # TODO: this should be cleaned up with csv
            pad_size = int(max(h0, w0) * 0.25)
            x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
            y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
            # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
            tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
            tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

        # FIXME: Correct in the csv file directly
        shape = self.obj_class_name.split("-")[0]
        if shape != "octagon":
            tgt = geometric_tf.sort_polygon_vertices(tgt)

        # if row['use_polygon'] == 1:
        #     # nR-M2zUbIWJzatAuy2egrQ.jpg
        # if row['use_polygon'] == 1:
        #     print(tgt)
        #     import pdb
        #     pdb.set_trace()

        if shape == "diamond":
            # Verify target points of diamonds. If they are very close to corners
            # of a square, the sign likely lies on another square surface. In this
            # case, use the square src points instead.
            x, y = np.abs(tgt[1] - tgt[0])
            angle10 = np.arctan2(y, x)
            x, y = np.abs(tgt[3] - tgt[2])
            angle32 = np.arctan2(y, x)
            mean_angle = (angle10 + angle32) / 2
            if mean_angle < np.pi / 180 * 15:
                self.obj_class_name = "square-600.0"

        src = np.array(self.src_points, dtype=np.float32)

        if shape == "pentagon":
            # Verify that target points of pentagons align like rectangle (almost
            # parallel sides). If not, then there's an annotation error which is
            # then fixed by changing src points.
            angle10 = np.arctan2(*(tgt[1] - tgt[0]))
            angle21 = np.arctan2(*(tgt[2] - tgt[1]))
            angle23 = np.arctan2(*(tgt[2] - tgt[3]))
            angle30 = np.arctan2(*(tgt[3] - tgt[0]))
            mean_diff = (
                np.abs(angle10 - angle23) + np.abs(angle21 - angle30)
            ) / 2
            # FIXME: we should fix this by tgt points instead
            if mean_diff > np.pi / 180 * 15:
                src[1, 0] = float(src[1, 1])
                src[1, 1] = 0

        # Get transformation matrix and transform function (affine or perspective)
        # from source and target coordinates
        if self.patch_transform_mode == "translate_scale":
            # Use corners of axis-aligned bounding box for transform (translation
            # and scaling) instead of real corners.
            min_tgt_x = min(tgt[:, 0])
            max_tgt_x = max(tgt[:, 0])
            min_tgt_y = min(tgt[:, 1])
            max_tgt_y = max(tgt[:, 1])
            tgt = np.array(
                [
                    [min_tgt_x, min_tgt_y],
                    [max_tgt_x, min_tgt_y],
                    [max_tgt_x, max_tgt_y],
                    [min_tgt_x, max_tgt_y],
                ]
            )

            min_src_x = min(src[:, 0])
            max_src_x = max(src[:, 0])
            min_src_y = min(src[:, 1])
            max_src_y = max(src[:, 1])
            src = np.array(
                [
                    [min_src_x, min_src_y],
                    [max_src_x, min_src_y],
                    [max_src_x, max_src_y],
                    [min_src_x, max_src_y],
                ]
            )

        if len(src) == 3:
            M = (
                torch.from_numpy(cv2.getAffineTransform(src, tgt))
                .unsqueeze(0)
                .float()
            )
            transform_func = kornia_tf.warp_affine
        else:
            src = torch.from_numpy(src).unsqueeze(0)
            tgt = torch.from_numpy(tgt).unsqueeze(0)
            M = kornia_tf.get_perspective_transform(src, tgt)
            transform_func = kornia_tf.warp_perspective

        return transform_func, M

    def apply_object(
        self,
        image: ImageTensor,
        target: Target,
    ) -> Tuple[ImageTensor, Target]:
        """Apply adversarial patch to image using REAP approach.

        Args:
            image: Image to apply patch to.
            target: Target labels (unmodified).

        Returns:
            final_img: Image with transformed patch applied.
            target: Target with synthetic object label added.
        """
        adv_patch: ImageTensor = self.adv_patch.clone()
        patch_mask: ImageTensor = self.patch_mask.clone()

        adv_patch = img_util.coerce_rank(adv_patch, 3)
        patch_mask = img_util.coerce_rank(patch_mask, 3)
        if not (
            adv_patch.shape[-2:] == patch_mask.shape[-2:] == self.obj_size_px
        ):
            raise ValueError(
                f"Shape mismatched: adv_patch {adv_patch.shape}, patch_mask "
                f"{patch_mask.shape}, obj_size {self.obj_size_px}!"
            )

        # Apply relighting transform (brightness and contrast)
        adv_patch.mul_(self.alpha).add_(self.beta)
        adv_patch.clamp_(0, 1)

        # Apply extra lighting augmentation on patch
        adv_patch = self.aug_light(adv_patch)
        adv_patch.clamp_(0, 1)

        # Combine patch_mask with adv_patch as alpha channel
        alpha_patch: ImageTensorRGBA = torch.cat([adv_patch, patch_mask], dim=0)
        # Crop with sign_mask and patch_mask
        alpha_patch *= self.obj_mask * patch_mask

        # Apply extra geometric augmentation on patch
        alpha_patch: BatchImageTensorRGBA
        alpha_patch, _ = self.aug_geo(alpha_patch)
        alpha_patch = img_util.coerce_rank(alpha_patch, 4)

        # Apply transform on RGBA patch
        warped_patch: BatchImageTensorRGBA = self.transform_fn(alpha_patch)
        warped_patch.squeeze_(0)
        warped_patch.clamp_(0, 1)

        # Place patch on object using alpha channel
        alpha_mask = warped_patch[-1:]
        warped_patch: ImageTensor = warped_patch[:-1]
        final_img: ImageTensor = (
            1 - alpha_mask
        ) * image + alpha_mask * warped_patch

        return final_img, target
