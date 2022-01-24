import torch
import torch.optim as optim
from kornia import augmentation as K
from kornia.constants import Resample
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import output_to_target, plot_images

from ..utils.image import letterbox
from .base_detector import DetectorAttackModule

EPS = 1e-6


class RP2AttackModule(DetectorAttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super(RP2AttackModule, self).__init__(attack_config, core_model, loss_fn, norm, eps, **kwargs)
        # self.num_steps = attack_config['rp2_steps']
        # self.step_size = attack_config['rp2_step_size']
        # self.num_restarts = attack_config['num_restarts']
        # self.optimizer = attack_config['optimizer']

        self.num_steps = 1000
        self.step_size = 1e-1
        self.num_restarts = 1
        self.optimizer = 'adam'
        self.input_size = (960, 1280)      # TODO: rectangle?
        self.num_eot = 5
        self.bg_transforms = K.RandomResizedCrop(self.input_size, p=1.0)
        # self.obj_transforms = K.container.AugmentationSequential(
        #     K.RandomAffine(30, translate=(0.5, 0.5)),      # Only translate and rotate as in Eykholt et al.
        #     # RandomAffine(30, translate=(0.5, 0.5), scale=(0.25, 4), shear=(0.1, 0.1), p=1.0),
        #     # TODO: add more transforms in the future
        #     return_transform=True,
        # )
        # self.mask_transforms = K.container.AugmentationSequential(
        #     K.RandomAffine(30, translate=(0.5, 0.5), resample=Resample.NEAREST),
        # )
        self.obj_transforms = K.RandomAffine(30, translate=(0.5, 0.5), p=1.0, return_transform=True)
        self.mask_transforms = K.RandomAffine(30, translate=(0.5, 0.5), p=1.0, resample=Resample.NEAREST)
        self.lmbda = 1e-2

    def attack(self,
               obj: torch.Tensor,
               obj_mask: torch.Tensor,
               patch_mask: torch.Tensor,
               backgrounds: torch.Tensor) -> torch.Tensor:
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
        height, width = obj.shape[-2:]
        device = obj.device
        dtype = obj.dtype
        obj_mask_dup = obj_mask.expand(self.num_eot, -1, -1, -1)

        # TODO: Initialize worst-case inputs, use moving average
        # x_adv_worst = x.clone().detach()
        # worst_losses =
        ema_loss = None

        for _ in range(self.num_restarts):

            # Initialize adversarial perturbation
            z_delta = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
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
            for step in range(self.num_steps):
                z_delta.requires_grad_()
                delta = self._to_model_space(z_delta, 0, 1)

                # Randomly select background and apply transforms (crop and scale)
                bg_idx = torch.randint(0, len(backgrounds), size=(self.num_eot, ))
                bgs = backgrounds[bg_idx]
                bgs = self.bg_transforms(bgs)

                # Apply random transformations
                adv_obj = patch_mask * delta + (1 - patch_mask) * obj
                adv_obj = adv_obj.expand(self.num_eot, -1, -1, -1)
                adv_obj, tf_params = self.obj_transforms(adv_obj)
                adv_obj = adv_obj.clamp(0, 1)

                # print(obj_mask_dup.shape)
                # print(tf_params)

                o_mask = self.mask_transforms.apply_transform(
                    obj_mask_dup, None, transform=tf_params)
                adv_img = o_mask * adv_obj + (1 - o_mask) * bgs
                adv_img = letterbox(adv_img, new_shape=self.input_size[1])[0]

                # Compute logits, loss, gradients
                out, _ = self.core_model(adv_img, val=True)

                # DEBUG
                # outt = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45)
                # plot_images(adv_img, output_to_target(outt))
                # import pdb
                # pdb.set_trace()

                tv = ((delta[:, :, :-1, :] - delta[:, :, 1:, :]).abs().mean() +
                      (delta[:, :, :, :-1] - delta[:, :, :, 1:]).abs().mean())
                loss = out[:, :, 4].mean() + self.lmbda * tv
                loss.backward(retain_graph=True)
                opt.step()

                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_const = 0.99
                    ema_loss = ema_const * ema_loss + (1 - ema_const) * loss.item()
                if step % 100 == 0:
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
                # if step >= 802:
                #     outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
                #     plot_images(adv_img.detach(), output_to_target(outt), fname=f'rp2_{step-800}.png')
                # if step == 820:
                #     break

        # DEBUG
        outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        plot_images(adv_img.detach(), output_to_target(outt))

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
