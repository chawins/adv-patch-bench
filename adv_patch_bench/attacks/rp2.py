import time
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from cv2 import getAffineTransform
from example_transforms import get_sign_canonical
from kornia import augmentation as K
from kornia.constants import Resample
from kornia.geometry.transform import (get_perspective_transform, warp_affine,
                                       warp_perspective)
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import output_to_target, plot_images

from ..utils.image import letterbox, mask_to_box
from .base_detector import DetectorAttackModule

EPS = 1e-6


class RP2AttackModule(DetectorAttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps,
                 rescaling=False, relighting=False, verbose=False, **kwargs):
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

        self.num_restarts = 1
        self.rescaling = rescaling
        self.relighting = relighting
        self.verbose = verbose
        self.augment_real = attack_config['rp2_augment_real']

        self.bg_transforms = K.RandomResizedCrop(self.input_size, scale=(0.25, 1), p=1.0)
        # self.obj_transforms = K.container.AugmentationSequential(
        #     K.RandomAffine(30, translate=(0.5, 0.5)),      # Only translate and rotate as in Eykholt et al.
        #     # RandomAffine(30, translate=(0.5, 0.5), scale=(0.25, 4), shear=(0.1, 0.1), p=1.0),
        #     # TODO: add more transforms in the future
        #     return_transform=True,
        # )
        # self.mask_transforms = K.container.AugmentationSequential(
        #     K.RandomAffine(30, translate=(0.5, 0.5), resample=Resample.NEAREST),
        # )
        self.obj_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, return_transform=True)
        self.mask_transforms = K.RandomAffine(30, translate=(0.45, 0.45), p=1.0, resample=Resample.NEAREST)
        self.jitter_transform = K.ColorJitter(brightness=0.3, contrast=0.3, p=1.0)
        self.real_transform = {}
        if self.augment_real:
            self.real_transform['tf_patch'] = K.RandomAffine(
                15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0, resample=Resample.BILINEAR)
            self.real_transform['tf_bg'] = self.bg_transforms

    def attack(self,
               obj: torch.Tensor,
               obj_mask: torch.Tensor,
               patch_mask: torch.Tensor,
               backgrounds: torch.Tensor,
               obj_class: int = None,
               obj_size: tuple = None) -> torch.Tensor:
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

        obj_mask_dup = obj_mask.expand(self.num_eot, -1, -1, -1)
        ymin, xmin, height, width = mask_to_box(patch_mask)
        patch_full = torch.zeros_like(obj)

        # TODO: Initialize worst-case inputs, use moving average
        # x_adv_worst = x.clone().detach()
        # worst_losses =
        ema_const = 0.99
        ema_loss = None

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
            z_delta.uniform_(-10, 10)

            # Set up optimizer
            if self.optimizer == 'sgd':
                opt = optim.SGD([z_delta], lr=self.step_size, momentum=0.999)
            elif self.optimizer == 'adam':
                opt = optim.Adam([z_delta], lr=self.step_size)
            elif self.optimizer == 'rmsprop':
                opt = optim.RMSprop([z_delta], lr=self.step_size)
            else:
                raise NotImplementedError('Given optimizer not implemented.')
            lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.2, patience=int(self.num_steps / 10),
                threshold=1e-9, min_lr=self.step_size * 1e-6, verbose=True)

            # Run PGD on inputs for specified number of steps
            with torch.autograd.set_detect_anomaly(True):
                for step in range(self.num_steps):
                    z_delta.requires_grad_()
                    delta = self._to_model_space(z_delta, 0, 1)

                    # Randomly select background and apply transforms (crop and scale)
                    bg_idx = torch.randint(0, len(backgrounds), size=(self.num_eot, ))
                    bgs = backgrounds[bg_idx]
                    bgs = self.bg_transforms(bgs)

                    # indices = np.where(obj_mask.cpu()[0] > 0)
                    # x_min, x_max = min(indices[1]), max(indices[1])
                    # y_min, y_max = min(indices[0]), max(indices[0])
                    # synthetic_sign_height = y_max - y_min
                    # synthetic_sign_width = x_max - x_min
                    # synthetic_sign_size = x_max - x_min
                    synthetic_sign_size = obj_size[0]
                    # print(synthetic_sign_size)

                    # old_ratio = synthetic_sign_size/960
                    if self.rescaling:
                        old_ratio = synthetic_sign_size/self.input_size[0]
                        prob_array = [0.38879158, 0.26970227, 0.16462349, 0.07530647, 0.04378284,
                                      0.03327496, 0.01050788, 0.00700525, 0.00350263, 0.00350263]
                        new_possible_ratios = [0.05340427, 0.11785139, 0.18229851, 0.24674563,
                                               0.31119275, 0.3756399, 0.440087, 0.5045341, 0.56898123, 0.6334284, 0.6978755]
                        index_array = np.arange(0, len(new_possible_ratios)-1)
                        sampled_index = np.random.choice(index_array, None, p=prob_array)
                        low_bin_edge, high_bin_edge = new_possible_ratios[sampled_index], new_possible_ratios[sampled_index+1]
                        self.obj_transforms = K.RandomAffine(
                            30, translate=(0.45, 0.45),
                            p=1.0, return_transform=True, scale=(low_bin_edge / old_ratio, high_bin_edge / old_ratio))
                    else:
                        self.obj_transforms = K.RandomAffine(
                            30, translate=(0.45, 0.45),
                            p=1.0, return_transform=True, scale=None)

                    # print((low_bin_edge/old_ratio, high_bin_edge/old_ratio))
                    # new_size = (int(self.input_size[0] * new_synthetic_sign_ratio / old_ratio), int(self.input_size[1] * new_synthetic_sign_ratio / old_ratio))
                    # self.resize_transforms = T.Resize(size=new_size)

                    # Apply random transformations
                    patch_full[:, ymin:ymin + height, xmin:xmin + width] = delta
                    adv_obj = patch_mask * patch_full + (1 - patch_mask) * obj

                    adv_obj = adv_obj.expand(self.num_eot, -1, -1, -1)
                    if self.relighting:
                        adv_obj = self.jitter_transform(adv_obj)
                    adv_obj, tf_params = self.obj_transforms(adv_obj)
                    adv_obj = adv_obj.clamp(0, 1)

                    o_mask = self.mask_transforms.apply_transform(
                        obj_mask_dup, None, transform=tf_params)
                    adv_img = o_mask * adv_obj + (1 - o_mask) * bgs
                    # Patch image the same way as YOLO
                    adv_img = letterbox(adv_img, new_shape=self.input_size[1])[0]

                    # if step % 100 == 0:
                    #     torchvision.utils.save_image(adv_img[0], f'tmp/synthetic/test_synthetic_adv_img_{step}.png')

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

                loss /= self.num_eot
                tv = ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() +
                      (delta[:, :, :, :-1] - delta[:, :, :, 1:]).abs().mean())
                # loss = out[:, :, 4].mean() + self.lmbda * tv
                loss += self.lmbda * tv
                loss.backward(retain_graph=True)
                opt.step()
                # lr_schedule.step(loss)

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                if step % 100 == 0 and self.verbose:
                    print(f'step: {step}   loss: {ema_loss:.6f}')

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
        self.objs = objs
        resize_transform = torchvision.transforms.Resize(size=self.input_size)
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
        tf_function = get_transform(sign_size_in_pixel, *objs[0][1])[0]
        tf_data_temp = [get_transform(sign_size_in_pixel, *obj[1])[1:] for obj in objs]
        tf_data = []
        for i in range(5):
            data_i = []
            for data in tf_data_temp:
                data_i.append(data[i].unsqueeze(0))
            data_i = torch.cat(data_i, dim=0).to(device)
            if i in (3, 4):
                # Add singletons for alpha and beta
                data_i = data_i[:, None, None, None]
            tf_data.append(data_i)

        all_bg_idx = np.arange(len(tf_data_temp))
        backgrounds = torch.cat([obj[0].unsqueeze(0) for obj in objs], dim=0).to(device)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, height, width), device=device, dtype=torch.float32)
            z_delta.uniform_(0, 1)

            # Set up optimizer
            if self.optimizer == 'sgd':
                opt = optim.SGD([z_delta], lr=self.step_size, momentum=0.9)
            elif self.optimizer == 'adam':
                opt = optim.Adam([z_delta], lr=self.step_size)
            elif self.optimizer == 'rmsprop':
                opt = optim.RMSprop([z_delta], lr=self.step_size)
            else:
                raise NotImplementedError('Given optimizer not implemented.')
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

                # adv_img = apply_transform(backgrounds[bg_idx], delta, patch_mask, patch_loc, tf_function, curr_tf_data)
                adv_img = apply_transform(
                    backgrounds[bg_idx], delta, patch_mask, patch_loc,
                    tf_function, curr_tf_data, **self.real_transform)
                adv_img = resize_transform(adv_img)

                # DEBUG
                # torchvision.utils.save_image(adv_img, 'temp.png')
                # import pdb
                # pdb.set_trace()

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

                # lr_schedule.step(loss)

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                if step % 100 == 0 and self.verbose:
                    print(f'step: {step:4d}   loss: {ema_loss:.4f}   time: {time.time() - start_time:.2f}s')
                    start_time = time.time()
                    # DEBUG
                    import os
                    for idx in range(self.num_eot):
                        if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
                            os.makedirs(f'tmp/{idx}/', exist_ok=True)
                        torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')

        # DEBUG
        outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        plot_images(adv_img.detach(), output_to_target(outt))

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return delta.detach()

    def transform_and_attack_pgd(self, objs, patch_mask, obj_class):
        """Run RP2 Attack.

        Args:

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        device = patch_mask.device
        mode = self.core_model.training
        self.core_model.eval()
        self.objs = objs
        resize_transform = torchvision.transforms.Resize(size=self.input_size)
        ymin, xmin, height, width = mask_to_box(patch_mask)
        patch_loc = (ymin, xmin, height, width)

        ema_const = 0.99
        ema_loss = None

        self.alpha = 1e-2

        # Process transform data and create batch tensors
        sign_size_in_pixel = patch_mask.size(-1)
        # Assume that every signs use the same transform function (len(tgt) = 4)
        tf_function = get_transform(sign_size_in_pixel, *objs[0][1])[0]
        tf_data_temp = [get_transform(sign_size_in_pixel, *obj[1])[1:] for obj in objs]
        tf_data = []
        for i in range(5):
            data_i = []
            for data in tf_data_temp:
                data_i.append(data[i].unsqueeze(0))
            data_i = torch.cat(data_i, dim=0).to(device)
            if i in (3, 4):
                # Add singletons for alpha and beta
                data_i = data_i[:, None, None, None]
            tf_data.append(data_i)

        # sign_canonical, sign_mask, M, alpha, beta = tf_data

        all_bg_idx = np.arange(len(tf_data_temp))
        backgrounds = torch.cat([obj[0].unsqueeze(0) for obj in objs], dim=0).to(device)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            delta = torch.autograd.Variable(
                torch.rand((1, 3, height, width),
                           device=device, dtype=torch.float32),
                requires_grad=True)

            # Set up optimizer
            # if self.optimizer == 'sgd':
            #     opt = optim.SGD([delta], lr=self.step_size, momentum=0.9)
            # elif self.optimizer == 'adam':
            #     opt = optim.Adam([delta], lr=self.step_size)
            # elif self.optimizer == 'rmsprop':
            #     opt = optim.RMSprop([delta], lr=self.step_size)
            # else:
            #     raise NotImplementedError('Given optimizer not implemented.')
            start_time = time.time()

            # Run PGD on inputs for specified number of steps
            for step in range(self.num_steps):
                delta.requires_grad_()

                # Randomly select background and apply transforms (crop and scale)
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[:self.num_eot]

                curr_tf_data = [data[bg_idx] for data in tf_data]

                delta_repeat = delta.repeat(self.num_eot, 1, 1, 1)

                adv_img = apply_transform(backgrounds[bg_idx], delta_repeat,
                                          patch_mask, patch_loc, tf_function, curr_tf_data)
                adv_img = resize_transform(adv_img)

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

                tv = ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() +
                      (delta[:, :, :, :-1] - delta[:, :, :, 1:]).abs().mean())

                loss += self.lmbda * tv

                # delta_repeat.retain_grad()
                loss.backward(retain_graph=True)

                grad = delta.grad.detach()

                print(delta)
                # print(torch.sum(grad))
                import pdb
                pdb.set_trace()
                # qqq

                # grad = torch.mean(delta_repeat, axis=0, keepdim=True)
                grad = torch.sign(grad)
                delta = delta.detach() + self.alpha * grad
                delta = delta.clamp_(0, 1)

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                if step % 100 == 0:
                    print(f'step: {step:4d}   loss: {ema_loss:.4f}   time: {time.time() - start_time:.2f}s')
                    start_time = time.time()
                    # DEBUG
                    import os
                    for idx in range(self.num_eot):
                        if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
                            os.makedirs(f'tmp/{idx}/', exist_ok=True)
                        torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')

        # DEBUG
        outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        plot_images(adv_img.detach(), output_to_target(outt))

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return delta.detach()

    # def _forward_linf(self, x, y):
    #     mode = self.core_model.training
    #     self.core_model.eval()

    #     # Initialize worst-case inputs
    #     x_adv_worst = x.clone().detach()
    #     worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

    #     # Repeat PGD for specified number of restarts
    #     for _ in range(self.num_restarts):
    #         x_adv = x.clone().detach()

    #         # Initialize adversarial inputs
    #         x_adv += torch.zeros_like(x_adv).uniform_(-self.eps, self.eps)
    #         x_adv = torch.clamp(x_adv, 0, 1)

    #         # Run PGD on inputs for specified number of steps
    #         for _ in range(self.num_steps):
    #             x_adv.requires_grad_()

    #             # Compute logits, loss, gradients
    #             with torch.enable_grad():
    #                 logits = self.core_model(x_adv, **self.forward_args)
    #                 loss = self.loss_fn(logits, y).mean()
    #                 grads = torch.autograd.grad(loss, x_adv, allow_unused=True)[0].detach()

    #             with torch.no_grad():
    #                 # Perform gradient update, project to norm ball
    #                 x_adv = x_adv.detach() + self.step_size * torch.sign(grads)
    #                 x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
    #                 # Clip perturbed inputs to image domain
    #                 x_adv = torch.clamp(x_adv, 0, 1)

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


def get_transform(sign_size_in_pixel, shape, predicted_class, row, h0, w0,
                  h_ratio, w_ratio, w_pad, h_pad):
    # TODO: This should directly come from patch mask
    sign_canonical, sign_mask, src = get_sign_canonical(
        shape, predicted_class, None, None, sign_size_in_pixel=sign_size_in_pixel)
    alpha = torch.tensor(row['alpha'])
    beta = torch.tensor(row['beta'])

    # if annotated, then use those points
    if not pd.isna(row['points']):
        tgt = np.array(literal_eval(row['points']), dtype=np.float32)
        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad
    else:
        tgt = row['tgt'] if pd.isna(row['tgt_polygon']) else row['tgt_polygon']
        tgt = np.array(literal_eval(tgt), dtype=np.float32)
        offset_x_ratio = row['xmin_ratio']
        offset_y_ratio = row['ymin_ratio']
        # Have to correct for the padding when df is saved (TODO: this should be simplified)
        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt[:, 1] = (tgt[:, 1] + y_min) * h_ratio + h_pad
        tgt[:, 0] = (tgt[:, 0] + x_min) * w_ratio + w_pad

    src = np.array(src, dtype=np.float32)
    if len(src) == 3:
        M = torch.from_numpy(getAffineTransform(src, tgt)).unsqueeze(0).float()
        transform_func = warp_affine
    else:
        src = torch.from_numpy(src).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        M = get_perspective_transform(src, tgt)
        transform_func = warp_perspective

    return transform_func, sign_canonical, sign_mask, M.squeeze(), alpha, beta


def apply_transform(image, adv_patch, patch_mask, patch_loc, transform_func, tf_data,
                    tf_patch=None, tf_bg=None):
    ymin, xmin, height, width = patch_loc
    sign_canonical, sign_mask, M, alpha, beta = tf_data
    adv_patch.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)
    sign_canonical[:, :-1, ymin:ymin + height, xmin:xmin + width] = adv_patch
    sign_canonical[:, -1, ymin:ymin + height, xmin:xmin + width] = 1
    sign_canonical = sign_mask * patch_mask * sign_canonical
    if tf_patch is not None:
        sign_canonical = tf_patch(sign_canonical)

    warped_patch = transform_func(sign_canonical,
                                  M, image.shape[2:],
                                  mode='bilinear',  # TODO: try others?
                                  padding_mode='zeros')
    warped_patch.clamp_(0, 1)
    alpha_mask = warped_patch[:, -1].unsqueeze(1)
    final_img = (1 - alpha_mask) * image / 255 + alpha_mask * warped_patch[:, :-1]
    if tf_bg is not None:
        final_img = tf_bg(final_img)
    return final_img
