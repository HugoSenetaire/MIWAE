from .abstract_reparam import ReparamTrick
from torch.distributions import RelaxedBernoulli, Bernoulli
import torch

class ReparamTrickBernoulli(ReparamTrick):
    """
    It's not really a reparam trick but useful anyway
    """
    def __init__(self,) -> None:
        super().__init__()
        self.distribution = Bernoulli
        self.multiplyer_for_param = 1

    def log_prob(self, z, parameters):
        parameters = torch.sigmoid(parameters)
        log_prob = torch.log(parameters+1e-8) * z + torch.log(1 - parameters + 1e-8) * (1 - z)
        return log_prob

    def forward(self, parameters):
        parameters = torch.sigmoid(parameters)
        dist = self.distribution(probs=parameters)
        return dist