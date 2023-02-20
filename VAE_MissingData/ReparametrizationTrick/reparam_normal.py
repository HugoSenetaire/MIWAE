from .abstract_reparam import ReparamTrick
from torch.distributions import Normal
import torch
import math

class ReparamTrickNormal(ReparamTrick):
    def __init__(self,) -> None:
        super().__init__()
        self.distribution = Normal
        self.multiplyer_for_param = 2
        self.mu = None
        self.logvar = None
        self.dist = None

    def forward(self, parameters):
        self.mu, self.log_var = parameters.chunk(2, dim=-1)
        self.dist = self.distribution(self.mu, (0.5 * self.log_var).exp())
        return self.dist
    

    def rsample(self, parameters, pathwise_sample = None, ):
        mu, log_var = parameters.chunk(2, dim=-1)
        if pathwise_sample is None:
            pathwise_sample = self.sample_pathwise(mu.shape).to(mu.device)

        z = pathwise_sample * (0.5 * log_var).exp() + mu
        return z

    def sample_pathwise(self, shape):
        distribution = Normal(torch.zeros(shape), torch.ones(shape))
        return distribution.sample()

    def log_prob(self, z, parameters):
        mu, log_var = parameters.chunk(2, dim=-1)
        log_prob = -((z - mu) ** 2) / (2 * log_var.exp()) - log_var/2 - math.log(math.sqrt(2 * math.pi))
        return log_prob