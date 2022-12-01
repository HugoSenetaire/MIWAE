from .abstract_reparam import ReparamTrick
from torch.distributions import Normal
import torch

class ReparamTrickNormal(ReparamTrick):
    def __init__(self,) -> None:
        super().__init__()
        self.distribution = Normal
        self.multiplyer_for_param = 2
        self.mu = None
        self.logvar = None
        self.dist = None

    def forward(self, parameters):
        self.mu, self.log_var = parameters.chunk(2, dim=1)
        # print(mu, log_var)
        self.dist = self.distribution(self.mu, (0.5 * self.log_var).exp())
        return self.dist
    
    def rsample(self, shape):
        # z = self.dist.rsample(shape)
        origin_z = Normal(torch.zeros_like(self.mu), torch.ones_like(self.log_var)).sample(shape)
        current_log_var = self.log_var.unsqueeze(0).unsqueeze(0).expand(*shape, *self.log_var.shape)
        current_mu = self.mu.unsqueeze(0).unsqueeze(0).expand(*shape, *self.mu.shape)
        z = origin_z * (0.5 * current_log_var).exp() + current_mu
        if torch.any(torch.isnan(z)):
            # print(mu, log_var)
            index_nan = torch.where(torch.isnan(z))
            print("index",index_nan)
            print("z",z[index_nan])
            print("originz", origin_z[index_nan])
            print("other z", z[:,:,index_nan[-2], index_nan[-1]])
            print("other origin z", origin_z[:,:,index_nan[-2], index_nan[-1]])
            print("mu", self.mu.shape)
            # print(z.shape)

            print("mu",self.mu[index_nan[-2], index_nan[-1]])
            print("logvar",(self.log_var[index_nan[-2], index_nan[-1]]/2).exp())
            print("NAN")
            assert 1==0
        return z