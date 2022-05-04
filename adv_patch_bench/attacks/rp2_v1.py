import time

import numpy as np
import torch
import torch.optim as optim
import torchvision
from adv_patch_bench.transforms import apply_transform, get_transform
from kornia import augmentation as K
from kornia.constants import Resample
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import output_to_target, plot_images

from ..utils.image import letterbox, mask_to_box
from .base_detector import DetectorAttackModule

EPS = 1e-6


class RP2AttackModule(DetectorAttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 rescaling=False, verbose=False, interp=None, **kwargs):
        super(RP2AttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs)
        self.num_steps = attack_config['rp2_num_steps']
        self.step_size = attack_config['rp2_step_size']
        self.optimizer = attack_config['rp2_optimizer']
        self.num_eot = attack_config['rp2_num_eot']
        self.lmbda = attack_config['rp2_lambda']
        self.min_conf = attack_config['rp2_min_conf']
        self.input_size = attack_config['input_size']
        self.attack_mode = attack_config['attack_mode']

        # TODO: rename a lot of these to be less confusing
        self.no_transform = attack_config['no_transform']
        self.relighting = attack_config['relighting']
        self.rescaling = rescaling
        self.augment_real = attack_config['rp2_augment_real']
        self.interp = attack_config['interp'] if interp is None else interp

        self.num_restarts = 1
        self.verbose = verbose

        # Defind EoT data augmentation when generating adversarial examples
        bg_size = (self.input_size[0] - 32, self.input_size[1] - 32)
        self.bg_transforms = K.RandomResizedCrop(bg_size, scale=(0.8, 1), p=1.0,
                                                 resample=self.interp)
        self.obj_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0,
                                             return_transform=True, resample=self.interp)
        self.mask_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0,
                                              resample=Resample.NEAREST)
        self.jitter_transform = K.ColorJitter(brightness=0.3, contrast=0.3, p=1.0)
        self.real_transform = {}
        if self.augment_real:
            self.real_transform['tf_patch'] = K.RandomAffine(
                15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0, resample=self.interp)
            self.real_transform['tf_bg'] = self.bg_transforms

    def _setup_opt(self, z_delta):
        # Set up optimizer
        if self.optimizer == 'sgd':
            opt = optim.SGD([z_delta], lr=self.step_size, momentum=0.999)
        elif self.optimizer == 'adam':
            opt = optim.Adam([z_delta], lr=self.step_size)
        elif self.optimizer == 'rmsprop':
            opt = optim.RMSprop([z_delta], lr=self.step_size)
        else:
            raise NotImplementedError('Given optimizer not implemented.')
        return opt

    def compute_loss(self, adv_img, obj_class):
        # Compute logits, loss, gradients
        out, _ = self.core_model(adv_img, val=True)
        conf = out[:, :, 4:5] * out[:, :, 5:]
        conf, labels = conf.max(-1)
        if obj_class is not None:
            loss = 0
            for c, l in zip(conf, labels):
                c_l = c[l == obj_class]
                if c_l.size(0) > 0:
                    # Select prediction from box with max confidence and ignore
                    # ones with already low confidence
                    loss += c_l.max().clamp_min(self.min_conf)
            loss /= self.num_eot
        else:
            loss = conf.max(1)[0].clamp_min(self.min_conf).mean()
        return loss

    def attack(
        self,
        obj: torch.Tensor,
        obj_mask: torch.Tensor,
        patch_mask: torch.Tensor,
        backgrounds: torch.Tensor,
        obj_class: int = None,
        obj_size: tuple = None,
    ) -> torch.Tensor:
        """Run RP2 Attack.

        Args:
            obj (torch.Tesnor): Object to place the adversarial patch on, shape [C, H, W]
            obj_mask (torch.Tesnor): Mask of object, must have shape [1, H, W]
            patch_mask (torch.Tesnor): Mask of the patch, must have shape [1, H, W]
            backgrounds (torch.Tesnor): Background images, shape [N, C, H, W]

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        mode = self.core_model.training
        self.core_model.eval()
        device = obj.device
        dtype = obj.dtype

        obj.detach_()
        obj_mask.detach_()
        patch_mask.detach_()
        backgrounds.detach_()
        obj_mask_dup = obj_mask.expand(self.num_eot, -1, -1, -1)
        ymin, xmin, height, width = mask_to_box(patch_mask)
        # patch_full = torch.zeros_like(obj)
        all_bg_idx = np.arange(len(backgrounds))

        # TODO: Initialize worst-case inputs, use moving average
        # x_adv_worst = x.clone().detach()
        ema_const = 0.99
        # ema_const = 0
        ema_loss = None

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
            z_delta.uniform_(-10, 10)

            opt = self._setup_opt(z_delta)
            # lr_schedule = optim.lr_scheduler.MultiStepLR(opt, [500, 1000, 1500], gamma=0.1)
            lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=int(self.num_steps / 10),
                threshold=1e-9, min_lr=self.step_size * 1e-6, verbose=True)
            start_time = time.time()

            # Run PGD on inputs for specified number of steps
            for step in range(self.num_steps):
                z_delta.requires_grad_()
                delta = self._to_model_space(z_delta, 0, 1)

                # Randomly select background and apply transforms (crop and scale)
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[:self.num_eot]
                bgs = backgrounds[bg_idx]
                bgs = self.bg_transforms(bgs)
                # Patch image the same way as YOLO
                bgs = letterbox(bgs, new_shape=self.input_size, color=114/255)[0]
                synthetic_sign_size = obj_size[0]

                if self.rescaling:
                    # TODO: clean this and generalize this to other signs?
                    old_ratio = synthetic_sign_size / self.input_size[0]
                    prob_array = [0.38879158, 0.26970227, 0.16462349, 0.07530647, 0.04378284,
                                  0.03327496, 0.01050788, 0.00700525, 0.00350263, 0.00350263]
                    new_possible_ratios = [0.05340427, 0.11785139, 0.18229851, 0.24674563,
                                           0.31119275, 0.3756399, 0.440087, 0.5045341, 0.56898123, 0.6334284, 0.6978755]
                    index_array = np.arange(0, len(new_possible_ratios) - 1)
                    sampled_index = np.random.choice(index_array, None, p=prob_array)
                    low_bin_edge, high_bin_edge = new_possible_ratios[sampled_index], new_possible_ratios[sampled_index+1]
                    self.obj_transforms = K.RandomAffine(
                        30, translate=(0.45, 0.45),
                        p=1.0, return_transform=True, scale=(low_bin_edge / old_ratio, high_bin_edge / old_ratio))

                # Copy patch to synthetic sign
                adv_obj = obj.clone()
                adv_obj[:, ymin:ymin + height, xmin:xmin + width] = delta

                # Apply random transformations to both sign and patch
                adv_obj = adv_obj.expand(self.num_eot, -1, -1, -1)
                if self.relighting:
                    adv_obj = self.jitter_transform(adv_obj)
                adv_obj, tf_params = self.obj_transforms(adv_obj)
                o_mask = self.mask_transforms.apply_transform(
                    obj_mask_dup, None, transform=tf_params)

                # Apply sign on background
                adv_img = o_mask * adv_obj + (1 - o_mask) * bgs
                adv_img = adv_img.clamp(0, 1)

                # DEBUG
                # if step % 100 == 0:
                #     torchvision.utils.save_image(adv_img[0], f'gen_synthetic_adv_{step}.png')
                #     import pdb
                #     pdb.set_trace()

                tv = ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() +
                      (delta[:, :, :, :-1] - delta[:, :, :, 1:]).abs().mean())
                loss = self.compute_loss(adv_img, obj_class)
                loss += self.lmbda * tv
                # out, _ = self.core_model(adv_img, val=True)
                # loss = out[:, :, 4].mean() + self.lmbda * tv
                loss.backward(retain_graph=True)
                opt.step()

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                # lr_schedule.step(ema_loss)

                if step % 100 == 0 and self.verbose:
                    # print(self.compute_loss(adv_img, obj_class))
                    print(f'step: {step}   loss: {ema_loss:.6f}   time: {time.time() - start_time:.2f}s')
                    start_time = time.time()

            # if self.num_restarts == 1:
            #     x_adv_worst = x_adv
            # else:
            #     # Update worst-case inputs with itemized final losses
            #     fin_losses = self.loss_fn(self.core_model(x_adv), y).reshape(worst_losses.shape)
            #     up_mask = (fin_losses >= worst_losses).float()
            #     x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
            #     worst_losses = fin_losses * up_mask + worst_losses * (1 - up_mask)

        # DEBUG
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.6)
        # plot_images(adv_img.clamp(0, 1).detach(), c)

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return delta.detach()

    def transform_and_attack(self, objs, patch_mask, obj_class):
        """Run RP2 Attack.

        Args:

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        device = patch_mask.device
        mode = self.core_model.training
        self.core_model.eval()
        ymin, xmin, height, width = mask_to_box(patch_mask)
        patch_loc = (ymin, xmin, height, width)

        # ema_const = 0.99
        ema_const = 0
        ema_loss = None

        self.alpha = 1e-2

        # Process transform data and create batch tensors
        sign_size_in_pixel = patch_mask.size(-1)
        # TODO: Assume that every signs use the same transform function
        # i.e., warp_perspetive. Have to fix this for triangles
        tf_function = get_transform(sign_size_in_pixel, *objs[0][1], no_transform=self.no_transform)[0]
        tf_data_temp = [get_transform(sign_size_in_pixel, *obj[1], no_transform=self.no_transform)[1:] for obj in objs]
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
        backgrounds = torch.cat([obj[0].unsqueeze(0) for obj in objs], dim=0).to(device)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, height, width), device=device, dtype=torch.float32)
            z_delta.uniform_(-10, 10)

            # Set up optimizer
            opt = self._setup_opt(z_delta)
            start_time = time.time()

            # Run PGD on inputs for specified number of steps
            for step in range(self.num_steps):
                z_delta.requires_grad_()

                if self.attack_mode == 'pgd':
                    delta = z_delta
                else:
                    delta = self._to_model_space(z_delta, 0, 1)

                # Randomly select background and apply transforms (crop and scale)
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[:self.num_eot]

                curr_tf_data = [data[bg_idx] for data in tf_data]
                delta = delta.repeat(self.num_eot, 1, 1, 1)
                adv_img = apply_transform(
                    backgrounds[bg_idx].clone(), delta.clone(), patch_mask, patch_loc,
                    tf_function, curr_tf_data, interp=self.interp,
                    **self.real_transform, relighting=self.relighting)

                # DEBUG
                # torchvision.utils.save_image(adv_img, 'temp.png')
                # import pdb
                # pdb.set_trace()

                loss = self.compute_loss(adv_img, obj_class)
                tv = ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() +
                      (delta[:, :, :, :-1] - delta[:, :, :, 1:]).abs().mean())
                # loss = out[:, :, 4].mean() + self.lmbda * tv
                loss += self.lmbda * tv
                loss.backward(retain_graph=True)

                if self.attack_mode == 'pgd':
                    grad = z_delta.grad.detach()
                    grad = torch.sign(grad)
                    z_delta = z_delta.detach() - self.alpha * grad
                    z_delta = z_delta.clamp_(0, 1)
                else:
                    opt.step()

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                # lr_schedule.step(loss)

                if step % 100 == 0 and self.verbose:
                    print(f'step: {step:4d}   loss: {ema_loss:.4f}   time: {time.time() - start_time:.2f}s')
                    start_time = time.time()
                    # DEBUG
                    # import os
                    # for idx in range(self.num_eot):
                    #     if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
                    #         os.makedirs(f'tmp/{idx}/', exist_ok=True)
                    #     torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')

        # DEBUG
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        # plot_images(adv_img.detach(), output_to_target(outt))

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return delta.detach()

    def _to_attack_space(self, x, min_, max_):
        # map from [min_, max_] to [-1, +1]
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = (x - a) / b

        # from [-1, +1] to approx. (-1, +1)
        x = x * 0.99999

        # from (-1, +1) to (-inf, +inf): atanh(x)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _to_model_space(self, x, min_, max_):
        """Transforms an input from the attack space to the model space. 
        This transformation and the returned gradient are elementwise."""
        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)

        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x