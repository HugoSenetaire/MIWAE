from .abstract_reparam import ReparamTrick
from torch.distributions import RelaxedBernoulli, Bernoulli
import torch

class ReparamTrickBernoulli(ReparamTrick):
    def __init__(self,) -> None:
        super().__init__()
        self.distribution = Bernoulli
        self.multiplyer_for_param = 1

    def forward(self, parameters):
        parameters = torch.sigmoid(parameters)
        dist = self.distribution(probs=parameters)
        return dist