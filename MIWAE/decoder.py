import torch.nn as nn
from torch.distributions import Bernoulli, Distribution, Normal
from .NetworkMiwae import ConvDecoder

from .ReparametrizationTrick import ReparamTrick, ReparamTrickBernoulli


class Decoder(nn.Module):
    def __init__(self, reparam_trick : ReparamTrick = ReparamTrickBernoulli, decoder_network : nn.Module = ConvDecoder, latent_dim: int = 10, output_channel: int = 1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channel = output_channel
        self.reparam_trick = reparam_trick
        self.multiplyer_for_param = self.reparam_trick.multiplyer_for_param
        self.decoder_network = decoder_network(latent_dim = self.latent_dim, output_channel = self.multiplyer_for_param * self.output_channel)

        self.dist = None

    def forward(self, latent, n_samples=None):
        _out =  self.decoder_network(latent)
        if n_samples is None:
            n_samples = 1

        self.dist = self.reparam_trick(_out)
        _x = self.dist.sample([n_samples])

        return _x.flatten(0,1), self.dist