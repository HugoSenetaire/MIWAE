import torch.nn as nn
from .ReparametrizationTrick import ReparamTrick, ReparamTrickNormal

class Encoder(nn.Module):
    def __init__(self, reparam_trick : ReparamTrick=ReparamTrickNormal(),  encoder_network: nn.Module = None, latent_dim: int = 10):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.reparam_trick = reparam_trick
        self.encoder_network = encoder_network

        self.dist = None

    def forward(self, x, mc_sample_z=None, iwae_sample_z = None,):
        _out = self.encoder_network(x)

        self.dist = self.reparam_trick(_out)
        if mc_sample_z is None:
            mc_sample_z = 1
        if iwae_sample_z is None :
            iwae_sample_z = 1

        _z = self.reparam_trick.rsample([iwae_sample_z, mc_sample_z])


        

        return _out, _z, self.dist