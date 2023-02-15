from torch.distributions import Distribution
import torch.nn as nn


class ReparamTrick(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.distribution = None
        self.multiplyer_for_param = None

    def log_prob(self, z, parameters):
        raise NotImplementedError

    def rsamples(self, parameters, pathwise_sample):
        """
        The goal of such function is to completely decorrelate the samples from the parameters
        and allow for vmap calculation
        """
        raise NotImplementedError

    def forward(self, params, n_samples=None):
        raise NotImplementedError
        