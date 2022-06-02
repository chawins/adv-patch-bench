import torch
from .base import AttackModule

EPS = 1e-6


class PatchAttackModule(AttackModule):

    def __init__(self, attack_config, core_model, loss_fn, norm, eps, **kwargs):
        super(PatchAttackModule, self).__init__(
            attack_config, core_model, loss_fn, norm, eps, **kwargs)
        self.num_steps = attack_config['pgd_steps']
        self.step_size = attack_config['pgd_step_size']
        self.num_restarts = attack_config['num_restarts']

    def forward(self, x, y):
        mode = self.core_model.training
        self.core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self.num_restarts):
            x_adv = x.clone().detach()
            # Fix patch location on top left corner
            delta = torch.zeros_like(x_adv)[:, :, :self.eps, :self.eps]
            # Initialize adversarial inputs
            delta.uniform_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self.num_steps):
                delta.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    x_adv[:, :, :self.eps, :self.eps] = delta
                    logits = self.core_model(x_adv)
                    loss = self.loss_fn(logits, y).mean()
                    grads = torch.autograd.grad(loss, delta)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    delta = delta.detach() + self.step_size * torch.sign(grads)
                    delta = torch.clamp(delta, 0, 1)

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
