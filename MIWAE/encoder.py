import torch.nn as nn
from torch.distributions import Normal, Distribution
from .ReparametrizationTrick import ReparamTrick, ReparamTrickNormal
from .NetworkMiwae import ConvFeatureExtractor

class Encoder(nn.Module):
    def __init__(self, reparam_trick : ReparamTrick=ReparamTrickNormal(),  feature_extractor: nn.Module = ConvFeatureExtractor(input_channel=1), latent_dim: int = 10):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.reparam_trick = reparam_trick
        self.multiplyer_for_param = self.reparam_trick.multiplyer_for_param
        self.feature_extractor = feature_extractor
        self.output_layers = nn.Sequential(nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.multiplyer_for_param*self.latent_dim))
        self.dist = None

    def forward(self, x, mc_sample_z=None, iwae_sample_z = None,):
        _out = self.feature_extractor(x)
        _out = _out.flatten(1)
        _out = self.output_layers(_out)

        self.dist = self.reparam_trick(_out)
        if mc_sample_z is None:
            mc_sample_z = 1
        if iwae_sample_z is None :
            iwae_sample_z = 1

        _z = self.dist.rsample([iwae_sample_z, mc_sample_z])

        

        return _out, _z, self.dist