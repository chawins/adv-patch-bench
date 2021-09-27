import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, sd, *args, **kwargs):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None])
        self.register_buffer('sd', torch.tensor(sd)[None, :, None, None])

    def forward(self, x):
        return (x - self.mean) / self.sd
