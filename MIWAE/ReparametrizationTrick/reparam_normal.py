from .abstract_reparam import ReparamTrick
from torch.distributions import Normal

class ReparamTrickNormal(ReparamTrick):
    def __init__(self,) -> None:
        super().__init__()
        self.distribution = Normal
        self.multiplyer_for_param = 2

    def forward(self, parameters):
        mu, log_var = parameters.chunk(2, dim=1)
        dist = self.distribution(mu, (0.5 * log_var).exp())
        return dist