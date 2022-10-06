import torch.nn as nn


class DetectorAttackModule(nn.Module):
    def __init__(
        self,
        attack_config,
        core_model,
        input_size=(1536, 2048),
        verbose=False,
        **kwargs
    ):
        super().__init__()
        self.core_model = core_model
        self.input_size = input_size
        self.verbose = verbose
