import time
from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from adv_patch_bench.attacks.base_detector import DetectorAttackModule
from adv_patch_bench.transforms import apply_transform, get_transform
from adv_patch_bench.utils.image import (
    coerce_rank,
    mask_to_box,
    resize_and_center,
)
from kornia import augmentation as K
from kornia.constants import Resample
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import output_to_target, plot_images


class RP2AttackModule(DetectorAttackModule):
    def __init__(
        self,
        attack_config,
        core_model,
        loss_fn,
        norm,
        eps,
        rescaling=False,
        verbose=False,
        interp=None,
        **kwargs,
    ):
        super(RP2AttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs
        )
        self.input_size = attack_config["input_size"]

        rp2_config = attack_config["rp2"]
        self.num_steps = rp2_config["num_steps"]
        self.step_size = rp2_config["step_size"]
        self.optimizer = rp2_config["optimizer"]
        self.use_lr_schedule = rp2_config["use_lr_schedule"]
        self.num_eot = rp2_config["num_eot"]
        self.lmbda = rp2_config["lambda"]
        self.min_conf = rp2_config["min_conf"]
        self.patch_dim = rp2_config.get("patch_dim", None)
        self.attack_mode = rp2_config["attack_mode"].split("-")
        self.transform_mode = rp2_config["transform_mode"]
        self.use_relight = rp2_config["use_patch_relight"]
        self.interp = interp
        self.num_restarts = 1
        self.verbose = verbose
        self.is_training = None
        self.ema_const = 0.9

        # Use change of variable on delta with alpha and beta.
        # Mostly used with per-sign attack.
        self.use_var_change_ab = "var_change_ab" in self.attack_mode
        if not self.use_relight:
            self.use_var_change_ab = False
        if self.use_var_change_ab:
            # Does not work when num_eot > 1
            assert (
                self.num_eot == 1
            ), "When use_var_change_ab is used, num_eot can only be set to 1."
            # No need to relight further
            self.use_relight = False

        # TODO: We probably don't need this now
        # self.rescaling = rescaling
        self.rescaling = False

        # Define EoT augmentation for attacking synthetic signs
        p_geo = float(rp2_config["augment_prob_geometric"])
        rotate_degrees = float(rp2_config.get("augment_rotate_degree", 15))
        p_light = float(rp2_config["augment_prob_relight"])
        intensity_light = float(
            rp2_config.get("augment_intensity_relight", 0.3)
        )
        bg_size = self.input_size
        self.bg_transforms = K.RandomResizedCrop(
            bg_size, scale=(0.8, 1), p=p_geo, resample=self.interp
        )
        self.obj_transforms = K.RandomAffine(
            degrees=rotate_degrees,
            translate=(0.4, 0.4),
            p=p_geo,
            return_transform=True,
            resample=self.interp,
            # TODO: add scaling
        )
        # Args to mask_transforms can be set to anything because it will use
        # the same params as obj_transforms anyway (via apply_transform).
        self.mask_transforms = K.RandomAffine(
            rotate_degrees,
            translate=(0.40, 0.40),
            p=p_geo,
            resample=Resample.NEAREST,
        )
        self.jitter_transform = K.ColorJitter(
            brightness=intensity_light, contrast=intensity_light, p=p_light
        )

        # Define EoT augmentation for attacking real signs
        # Transforms patch and background when attacking real signs
        self.real_transform = {
            "tf_patch": K.RandomAffine(
                15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                p=p_geo,
                resample=self.interp,
            )
        }
        # Background should move very little because gt annotation is fixed
        # self.real_transform['tf_bg'] = self.bg_transforms

    @abstractmethod
    def _loss_func(self, delta, adv_img, obj_class, metadata):
        raise NotImplementedError("_loss_func not implemented!")

    def _on_enter_attack(self, **kwargs):
        """Method called at the begining of the attack call."""
        self.is_training = self.core_model.training
        self.core_model.eval()

    def _on_exit_attack(self, **kwargs):
        """Method called at the end of the attack call."""
        self.core_model.train(self.is_training)

    def _on_syn_attack_step(self, metadata, **kwargs):
        """
        Method called at the begining of every step in synthetic attack.
        Can be used to update metadata.
        """
        return metadata

    def _on_real_attack_step(self, metadata, **kwargs):
        """
        Method called at the begining of every step in real attack.
        Can be used to update metadata.
        """
        return metadata

    def compute_loss(self, delta, adv_img, obj_class, metadata):
        loss = self._loss_func(adv_img, obj_class, metadata)
        tv = (delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() + (
            delta[:, :, :, :-1] - delta[:, :, :, 1:]
        ).abs().mean()
        loss += self.lmbda * tv
        return loss

    @torch.no_grad()
    def attack_synthetic(
        self,
        obj: torch.Tensor,
        obj_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        backgrounds: torch.Tensor,
        obj_class: int = None,
        metadata: Optional[List] = None,
    ) -> torch.Tensor:
        """Run RP2 Attack.

        Args:
            obj (torch.Tesnor): Object to place the adversarial patch on
                (shape: [C, H, W])
            obj_mask (torch.Tesnor): Mask of object (shape: [1, H, W])
            patch_mask (torch.Tesnor): Mask of the patch (shape: [1, H, W])
            backgrounds (torch.Tesnor): Background images (shape: [N, C, H, W])

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        self._on_enter_attack()
        device = obj.device
        dtype = obj.dtype

        obj.detach_()
        obj_mask.detach_()

        patch_mask.detach_()
        backgrounds.detach_()
        _, _, obj_height, obj_width = mask_to_box(obj_mask)
        # TODO: Allow patch to be non-square
        if self.patch_dim is not None:
            patch_dim = self.patch_dim
        else:
            patch_dim = max(obj_height, obj_width)
        obj_mask = coerce_rank(obj_mask, 3)
        all_bg_idx = np.arange(len(backgrounds))

        obj_mask = coerce_rank(obj_mask, 4)
        patch_mask = coerce_rank(patch_mask, 4)
        obj = coerce_rank(obj, 4)
        obj_mask_eot = obj_mask.expand(self.num_eot, -1, -1, -1)
        patch_mask_eot = patch_mask.expand(self.num_eot, -1, -1, -1)
        obj_eot = obj.expand(self.num_eot, -1, -1, -1)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta = torch.zeros(
                (1, 3, patch_dim, patch_dim),
                device=device,
                dtype=dtype,
            )
            z_delta.uniform_(0, 1)

            opt, lr_schedule = self._setup_opt(z_delta)
            self.start_time = time.time()
            self.ema_loss = None
            counter = 0

            for step in range(self.num_steps):
                # Randomly select background and apply transforms (crop and scale)
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[: self.num_eot]
                bgs = backgrounds[bg_idx]
                bgs = self.bg_transforms(bgs)

                # UNUSED
                # if self.rescaling:
                #     synthetic_sign_size = obj_size[0]
                #     old_ratio = synthetic_sign_size / self.input_size[0]
                #     prob_array = [0.38879158, 0.26970227, 0.16462349, 0.07530647, 0.04378284,
                #                   0.03327496, 0.01050788, 0.00700525, 0.00350263, 0.00350263]
                #     new_possible_ratios = [0.05340427, 0.11785139, 0.18229851, 0.24674563,
                #                            0.31119275, 0.3756399, 0.440087, 0.5045341, 0.56898123, 0.6334284, 0.6978755]
                #     index_array = np.arange(0, len(new_possible_ratios) - 1)
                #     sampled_index = np.random.choice(index_array, None, p=prob_array)
                #     low_bin_edge, high_bin_edge = new_possible_ratios[sampled_index], new_possible_ratios[sampled_index+1]
                #     self.obj_transforms = K.RandomAffine(
                #         30, translate=(0.45, 0.45),
                #         p=1.0, return_transform=True, scale=(low_bin_edge / old_ratio, high_bin_edge / old_ratio))

                # Apply random transformations to synthetic sign and masks
                adv_obj, tf_params = self.obj_transforms(obj_eot)
                o_mask = self.mask_transforms.apply_transform(
                    obj_mask_eot, None, transform=tf_params
                )
                p_mask = self.mask_transforms.apply_transform(
                    patch_mask_eot, None, transform=tf_params
                )

                metadata = self._on_syn_attack_step(
                    metadata, o_mask=o_mask, bg_idx=bg_idx, obj_class=obj_class
                )

                with torch.enable_grad():
                    z_delta.requires_grad_()
                    delta = self._to_model_space(z_delta, 0, 1)
                    delta_padded = resize_and_center(
                        delta,
                        img_size=self.input_size,
                        obj_size=(obj_height, obj_width),
                        is_binary=False,
                        interp=self.interp,
                    )
                    delta_eot = delta_padded.expand(self.num_eot, -1, -1, -1)
                    delta_eot = self.obj_transforms.apply_transform(
                        delta_eot, None, transform=tf_params
                    )
                    adv_obj = p_mask * delta_eot + (1 - p_mask) * adv_obj
                    # Augment sign and patch with relighting
                    adv_obj = self.jitter_transform(adv_obj)

                    # Apply sign on background
                    adv_img = o_mask * adv_obj + (1 - o_mask) * bgs
                    adv_img = adv_img.clamp(0, 1)

                    # DEBUG
                    # if step % 100 == 0:
                    #     torchvision.utils.save_image(
                    #         adv_img[0], f'gen_adv_syn_{step}.png')

                    mdata = None if metadata is None else metadata[bg_idx]
                    loss = self.compute_loss(delta, adv_img, obj_class, mdata)
                    loss.backward()
                    z_delta = self._step_opt(z_delta, opt)

                    # counter += 1
                    # if counter < 5:
                    #     continue

                if lr_schedule is not None:
                    lr_schedule.step(self.ema_loss)
                self._print_loss(loss, step)

            # if self.num_restarts == 1:
            #     x_adv_worst = x_adv
            # else:
            #     # Update worst-case inputs with itemized final losses
            #     fin_losses = self.loss_fn(self.core_model(x_adv), y).reshape(worst_losses.shape)
            #     up_mask = (fin_losses >= worst_losses).float()
            #     x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
            #     worst_losses = fin_losses * up_mask + worst_losses * (1 - up_mask)

        # DEBUG: YOLO
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.6)
        # plot_images(adv_img.clamp(0, 1).detach(), c)

        self._on_exit_attack()
        # Return worst-case perturbed input logits
        return self._to_model_space(z_delta.detach(), 0, 1)

    @torch.no_grad()
    def attack_real(
        self,
        objs: List,
        patch_mask: torch.Tensor,
        obj_class: int,
        metadata: Optional[List] = None,
    ):
        """Run RP2 Attack.

        Args:

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        self._on_enter_attack()
        device = patch_mask.device

        # Process transform data and create batch tensors
        obj_size = patch_mask.shape[-2:]
        obj_width_px = obj_size[-1]

        # TODO: Assume that every signs use the same transform function
        # i.e., warp_perspetive. Have to fix this for triangles
        tf_function = get_transform(
            obj_width_px, *objs[0][1], self.transform_mode
        )[0]
        tf_data_temp = [
            get_transform(obj_width_px, *obj[1], self.transform_mode)[1:-1]
            for obj in objs
        ]

        # tf_data contains [sign_canonical, sign_mask, M, alpha, beta]
        tf_data = []
        for i in range(5):
            data_i = []
            for data in tf_data_temp:
                data_i.append(data[i].unsqueeze(0))
            data_i = torch.cat(data_i, dim=0).to(device)
            # Add singletons for alpha and beta ([B, ] -> [B, 1, 1, 1])
            if i in (3, 4):
                data_i = data_i[:, None, None, None]
            tf_data.append(data_i)

        all_bg_idx = np.arange(len(tf_data_temp))
        backgrounds = torch.cat(
            [obj[0].unsqueeze(0) for obj in objs], dim=0
        ).to(device)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            # TODO: check if this is not buggy and works as expected
            z_delta = torch.zeros(
                (1, 3) + patch_mask.shape[-2:],
                # (1, 3, self.patch_dim, self.patch_dim),
                device=device,
                dtype=torch.float32,
            )
            z_delta.uniform_(0, 1)

            # Set up optimizer
            opt, lr_schedule = self._setup_opt(z_delta)
            self.ema_loss = None
            self.start_time = time.time()

            # Run PGD on inputs for specified number of steps
            for step in range(self.num_steps):

                # Randomly select background and place patch with transforms
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[: self.num_eot]
                curr_tf_data = [data[bg_idx] for data in tf_data]
                bgs = backgrounds[bg_idx].clone()

                metadata = self._on_real_attack_step(metadata)
                with torch.enable_grad():
                    z_delta.requires_grad_()
                    # Determine how perturbation is projected
                    if self.use_var_change_ab:
                        # Does not work when num_eot > 1
                        alpha, beta = curr_tf_data[-2:]
                        delta = self._to_model_space(
                            z_delta, beta, alpha + beta
                        )
                    else:
                        delta = self._to_model_space(z_delta, 0, 1)
                    delta_resized = resize_and_center(
                        delta,
                        img_size=None,
                        obj_size=obj_size,
                        is_binary=False,
                        interp=self.interp,
                    )
                    delta_eot = delta_resized.repeat(self.num_eot, 1, 1, 1)
                    adv_img, _ = apply_transform(
                        bgs,
                        delta_eot,
                        patch_mask,
                        tf_function,
                        curr_tf_data,
                        interp=self.interp,
                        **self.real_transform,
                        use_relight=self.use_relight,
                    )
                    adv_img /= 255

                    # DEBUG
                    # if step % 100 == 0:
                    #     torchvision.utils.save_image(
                    #         adv_img[0], f'gen_adv_real_{step}.png')

                    mdata = None if metadata is None else metadata[bg_idx]
                    loss = self.compute_loss(delta, adv_img, obj_class, mdata)
                    loss.backward()
                    z_delta = self._step_opt(z_delta, opt)

                if lr_schedule is not None:
                    lr_schedule.step(loss)
                self._print_loss(loss, step)

                # DEBUG
                # import os
                # for idx in range(self.num_eot):
                #     if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
                #         os.makedirs(f'tmp/{idx}/', exist_ok=True)
                #     torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')

        # DEBUG
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        # plot_images(adv_img.detach(), output_to_target(outt))

        self._on_exit_attack()
        # Return worst-case perturbed input logits
        return delta.detach()

    def _setup_opt(self, z_delta):
        # Set up optimizer
        if self.optimizer == "sgd":
            opt = optim.SGD([z_delta], lr=self.step_size, momentum=0.999)
        elif self.optimizer == "adam":
            opt = optim.Adam([z_delta], lr=self.step_size)
        elif self.optimizer == "rmsprop":
            opt = optim.RMSprop([z_delta], lr=self.step_size)
        elif self.optimizer == "pgd":
            opt = None
        else:
            raise NotImplementedError("Given optimizer not implemented.")

        lr_schedule = None
        if self.use_lr_schedule and opt is not None:
            # lr_schedule = optim.lr_scheduler.MultiStepLR(opt, [500, 1000, 1500], gamma=0.1)
            lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=0.5,
                patience=int(self.num_steps / 10),
                threshold=1e-9,
                min_lr=self.step_size * 1e-6,
                verbose=self.verbose,
            )

        return opt, lr_schedule

    def _step_opt(self, z_delta, opt):
        if self.optimizer == "pgd":
            grad = z_delta.grad.detach()
            grad = torch.sign(grad)
            z_delta = z_delta.detach() - self.step_size * grad
            z_delta.clamp_(0, 1)
        else:
            opt.step()
        return z_delta

    def _to_model_space(self, x, min_, max_):
        """Transforms an input from the attack space to the model space.
        This transformation and the returned gradient are elementwise."""
        if "pgd" in self.attack_mode:
            return x

        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)

        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x

    def _print_loss(self, loss, step):
        if self.ema_loss is None:
            self.ema_loss = loss.item()
        else:
            self.ema_loss = (
                self.ema_const * self.ema_loss
                + (1 - self.ema_const) * loss.item()
            )

        if step % 100 == 0 and self.verbose:
            print(
                f"step: {step:4d}  loss: {self.ema_loss:.4f}  "
                f"time: {time.time() - self.start_time:.2f}s"
            )
            self.start_time = time.time()
