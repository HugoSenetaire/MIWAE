from torch.distributions import Distribution
import torch.nn as nn


class ReparamTrick(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.distribution = None
        self.multiplyer_for_param = None

    def forward(self, params, n_samples=None):
        raise NotImplementedError
        