import torch.nn as nn


class DetectorAttackModule(nn.Module):
    def __init__(
        self,
        attack_config,
        core_model,
        loss_fn,
        norm,
        eps,
        verbose=False,
        **kwargs
    ):
        super(DetectorAttackModule, self).__init__()
        self.core_model = core_model
        self.loss_fn = loss_fn
        self.eps = eps
        self.verbose = verbose
