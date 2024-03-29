import torch.nn as nn

from .ReparametrizationTrick import ReparamTrick, ReparamTrickBernoulli


class Decoder(nn.Module):
    def __init__(self, reparam_trick : ReparamTrick = ReparamTrickBernoulli, decoder_network : nn.Module = None,):
        super(Decoder, self).__init__()
        self.reparam_trick = reparam_trick
        self.multiplyer_for_param = self.reparam_trick.multiplyer_for_param
        self.decoder_network = decoder_network
        self.dist = None

    def forward(self, latent, n_samples=None):
        _out =  self.decoder_network(latent)
        if n_samples is None:
            n_samples = 1

        self.dist = self.reparam_trick(_out)
        _out = _out.unsqueeze(1).expand(_out.shape[0], n_samples, *_out.shape[1:])
        _x = self.reparam_trick(_out).sample()

        return _x.flatten(0,1), self.dist