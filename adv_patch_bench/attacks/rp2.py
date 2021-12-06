import numpy as np
import torch
import torch.optim as optim
from kornia import augmentation as K
from kornia.constants import Resample

from .base_detector import DetectorAttackModule

EPS = 1e-6


class RP2AttackModule(DetectorAttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super(RP2AttackModule, self).__init__(attack_config, core_model, loss_fn, norm, eps, **kwargs)
        # self.num_steps = attack_config['rp2_steps']
        # self.step_size = attack_config['rp2_step_size']
        # self.num_restarts = attack_config['num_restarts']
        # self.optimizer = attack_config['optimizer']

        self.num_steps = 100
        self.step_size = 1e-3
        self.num_restarts = 1
        self.optimizer = 'adam'
        self.input_size = 1280      # TODO: rectangle?
        self.num_eot = 10
        self.bg_transforms = K.RandomResizedCrop(self.input_size)
        self.obj_transforms = K.container.AugmentationSequential(
            K.RandomAffine(30, translate=(0.5, 0.5)),      # Only translate and rotate as in Eykholt et al.
            # RandomAffine(30, translate=(0.5, 0.5), scale=(0.25, 4), shear=(0.1, 0.1), p=1.0),
            # TODO: add more transforms in the future
            return_transform=True,
        )
        self.mask_transforms = K.container.AugmentationSequential(
            K.RandomAffine(30, translate=(0.5, 0.5), resample=Resample.NEAREST),
        )

    def attack(self,
               obj: torch.Tensor,
               obj_mask: torch.Tensor,
               patch_mask: torch.Tensor,
               backgrounds: torch.Tensor) -> torch.Tensor:
        """Run RP2 Attack.

        Args:
            obj (torch.Tesnor): Object to place the adversarial patch on, shape [C, H, W]
            patch_mask (torch.Tesnor): Boolean mask of the patch, must have shape [H, W]
            backgrounds (torch.Tesnor): Background images, shape [N, C, H, W]

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """

        mode = self.core_model.training
        self.core_model.eval()
        device = obj.device
        dtype = obj.dtype
        p_mask = patch_mask.unsqueeze(0)
        obj_mask_dup = obj_mask.unsqueeze(0).expand(self.num_eot, 1, 1, 1)

        # TODO: Initialize worst-case inputs, use moving average
        # x_adv_worst = x.clone().detach()
        # worst_losses =

        for _ in range(self.num_restarts):

            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, ) + patch_mask.shape, device=device, dtype=dtype)
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

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                z_delta.requires_grad_()

                # Randomly select background and apply transforms (crop and scale)
                bg_idx = torch.randint(0, len(backgrounds), size=self.num_eot)
                bgs = backgrounds[bg_idx]
                bgs = self.bg_transforms(bgs)

                # TODO: Apply random transformations

                adv_obj = p_mask * z_delta + (1 - p_mask) * obj
                # DEBUG
                adv_obj = adv_obj.unsqueeze(0).expand(self.num_eot, 1, 1, 1)
                adv_obj, tf_params = self.obj_transforms(adv_obj)
                adv_obj.clamp_(0, 1)
                o_mask = self.mask_transforms.apply_transform(obj_mask_dup, tf_params)
                print(o_mask.shape, adv_obj.shape, bgs.shape)
                adv_img = o_mask * adv_obj + (1 - o_mask) * bgs

                # Compute logits, loss, gradients
                outputs = self.core_model(adv_img)
                loss = self.loss_fn(outputs, y).mean()
                loss.backward()
                opt.step()
                z_delta.clamp_(0, 1)

            if self.num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self.loss_fn(self.core_model(x_adv), y).reshape(worst_losses.shape)
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (1 - up_mask)

        # Return worst-case perturbed input logits
        self.core_model.train(mode)
        return x_adv_worst.detach()
